//! HTTP request handlers for the gateway API.

use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures::stream::StreamExt;
use gateway_core::{GatewayRequest, ModelsResponse, ModelObject};
use gateway_telemetry::RequestInfo;
use serde::Serialize;
use std::{convert::Infallible, time::Instant};
use tracing::{debug, error, info, instrument};

use crate::{
    error::ApiError,
    extractors::{JsonBody, RequestId, TenantId},
    state::AppState,
};

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,
    /// Version
    pub version: String,
    /// Uptime in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uptime_seconds: Option<u64>,
}

/// Health check endpoint
pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: None,
    })
}

/// Readiness check endpoint
pub async fn readiness_check(State(state): State<AppState>) -> impl IntoResponse {
    // Check if we have any providers
    let provider_count = state.providers.len();

    if provider_count > 0 {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "no providers available")
    }
}

/// Liveness check endpoint
pub async fn liveness_check() -> impl IntoResponse {
    (StatusCode::OK, "alive")
}

/// Metrics endpoint (Prometheus format)
pub async fn metrics_endpoint(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = state.metrics.gather();
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        metrics,
    )
}

/// List models endpoint (OpenAI compatible)
#[instrument(skip(state))]
pub async fn list_models(State(state): State<AppState>) -> Result<Json<ModelsResponse>, ApiError> {
    let models = state.providers.get_all_models();

    let data: Vec<ModelObject> = models
        .into_iter()
        .map(|m| ModelObject::new(&m.id, "system"))
        .collect();

    Ok(Json(ModelsResponse::new(data)))
}

/// Get model endpoint
#[instrument(skip(state))]
pub async fn get_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelObject>, ApiError> {
    let models = state.providers.get_all_models();

    let model = models
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| ApiError::not_found(format!("Model not found: {model_id}")))?;

    Ok(Json(ModelObject::new(&model.id, "system")))
}

/// Chat completion request (OpenAI compatible)
#[instrument(skip(state, body), fields(model = %body.model))]
pub async fn chat_completion(
    State(state): State<AppState>,
    RequestId(request_id): RequestId,
    TenantId(tenant_id): TenantId,
    JsonBody(body): JsonBody<GatewayRequest>,
) -> Result<Response, ApiError> {
    let request = body;
    let streaming = request.stream;

    debug!(
        request_id = %request_id,
        model = %request.model,
        streaming = streaming,
        tenant = ?tenant_id,
        "Processing chat completion request"
    );

    // Track request
    let request_info = RequestInfo::new(&request_id, &request.model)
        .with_streaming(streaming);
    state.tracker.start(request_info);

    // Route the request
    let (provider, _decision) = match state.router.route(&request, tenant_id.as_deref()) {
        Ok(result) => result,
        Err(e) => {
            state.tracker.complete_error(&request_id, 503, e.to_string());
            return Err(e.into());
        }
    };

    state.tracker.update_provider(&request_id, provider.id());

    // Get circuit breaker for the provider
    let circuit_breaker = state.circuit_breakers.get_or_create(provider.id());

    // Check circuit breaker
    if let Err(err) = circuit_breaker.check() {
        state.tracker.complete_error(&request_id, 503, err.to_string());
        return Err(err.into());
    }

    let start = Instant::now();

    // Handle streaming vs non-streaming
    if streaming {
        handle_streaming_request(state, request, request_id, provider, circuit_breaker, start).await
    } else {
        handle_non_streaming_request(state, request, request_id, provider, circuit_breaker, start).await
    }
}

async fn handle_non_streaming_request(
    state: AppState,
    request: GatewayRequest,
    request_id: String,
    provider: std::sync::Arc<dyn gateway_core::LLMProvider>,
    circuit_breaker: std::sync::Arc<gateway_resilience::CircuitBreaker>,
    start: Instant,
) -> Result<Response, ApiError> {
    // Execute with retry
    let result = state
        .retry_policy
        .execute(|| async {
            provider.chat_completion(&request).await
        })
        .await;

    let duration = start.elapsed();

    match result {
        Ok(response) => {
            circuit_breaker.record_success();

            // Record metrics
            let usage = &response.usage;
            state.tracker.complete_success(
                &request_id,
                200,
                Some(usage.prompt_tokens),
                Some(usage.completion_tokens),
            );

            state.metrics.record_request(&gateway_telemetry::RequestMetrics {
                model: request.model.clone(),
                provider: provider.id().to_string(),
                latency: duration,
                success: true,
                status_code: 200,
                input_tokens: Some(usage.prompt_tokens),
                output_tokens: Some(usage.completion_tokens),
                streaming: false,
                tenant_id: None,
            });

            state.router.record_completion(provider.id(), duration, true);

            info!(
                request_id = %request_id,
                provider = %provider.id(),
                duration_ms = duration.as_millis(),
                "Chat completion successful"
            );

            Ok(Json(response).into_response())
        }
        Err(e) => {
            circuit_breaker.record_failure();

            state.tracker.complete_error(&request_id, 500, e.to_string());
            state.metrics.record_error(provider.id(), &e.to_string());
            state.router.record_completion(provider.id(), duration, false);

            error!(
                request_id = %request_id,
                provider = %provider.id(),
                error = %e,
                "Chat completion failed"
            );

            Err(e.into())
        }
    }
}

async fn handle_streaming_request(
    state: AppState,
    request: GatewayRequest,
    request_id: String,
    provider: std::sync::Arc<dyn gateway_core::LLMProvider>,
    circuit_breaker: std::sync::Arc<gateway_resilience::CircuitBreaker>,
    _start: Instant,
) -> Result<Response, ApiError> {
    // Get streaming response
    let stream_result = provider.chat_completion_stream(&request).await;

    match stream_result {
        Ok(chunk_stream) => {
            // Record first chunk time
            let first_chunk_received = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let first_chunk_flag = first_chunk_received;
            let tracker = state.tracker.clone();
            let request_id_clone = request_id.clone();

            // Create SSE stream
            let sse_stream = chunk_stream.map(move |chunk_result| {
                match chunk_result {
                    Ok(chunk) => {
                        // Record first token time
                        if !first_chunk_flag.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            tracker.record_first_token(&request_id_clone);
                        }

                        // Count tokens
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                // Rough token estimate: ~4 chars per token
                                let token_count = (content.len() / 4).max(1) as u32;
                                tracker.record_tokens(&request_id_clone, token_count);
                            }
                        }

                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        Ok::<_, Infallible>(Event::default().data(data))
                    }
                    Err(e) => {
                        let error_event = serde_json::json!({
                            "error": {
                                "message": e.to_string(),
                                "type": "stream_error"
                            }
                        });
                        Ok(Event::default().data(error_event.to_string()))
                    }
                }
            });

            // Add [DONE] event at the end
            let done_stream = futures::stream::once(async {
                Ok::<_, Infallible>(Event::default().data("[DONE]"))
            });

            let full_stream = sse_stream.chain(done_stream);

            // Record success after stream setup
            circuit_breaker.record_success();

            // Note: actual completion tracking happens in stream events
            // This is tracked when the stream ends

            Ok(Sse::new(full_stream)
                .keep_alive(axum::response::sse::KeepAlive::default())
                .into_response())
        }
        Err(e) => {
            circuit_breaker.record_failure();
            state.tracker.complete_error(&request_id, 500, e.to_string());

            error!(
                request_id = %request_id,
                provider = %provider.id(),
                error = %e,
                "Streaming request failed"
            );

            Err(e.into())
        }
    }
}

/// Provider status response
#[derive(Debug, Serialize)]
pub struct ProviderStatus {
    /// Provider ID
    pub id: String,
    /// Provider type
    pub provider_type: String,
    /// Health status
    pub health: String,
    /// Number of models
    pub model_count: usize,
}

/// List providers endpoint
pub async fn list_providers(State(state): State<AppState>) -> Json<Vec<ProviderStatus>> {
    let provider_ids = state.providers.provider_ids();

    let statuses: Vec<ProviderStatus> = provider_ids
        .iter()
        .filter_map(|id| {
            state.providers.get(id).map(|p| ProviderStatus {
                id: id.clone(),
                provider_type: format!("{:?}", p.provider_type()),
                health: "unknown".to_string(),
                model_count: p.models().len(),
            })
        })
        .collect();

    Json(statuses)
}

/// Gateway statistics response
#[derive(Debug, Serialize)]
pub struct GatewayStats {
    /// Active request count
    pub active_requests: usize,
    /// Total requests processed
    pub total_requests: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average latency in ms
    pub avg_latency_ms: f64,
    /// Registered providers
    pub providers: usize,
}

/// Get gateway statistics
pub async fn gateway_stats(State(state): State<AppState>) -> Json<GatewayStats> {
    let tracker_stats = state.tracker.stats();

    Json(GatewayStats {
        active_requests: tracker_stats.active_requests,
        total_requests: tracker_stats.total_completed,
        success_rate: tracker_stats.success_rate,
        avg_latency_ms: tracker_stats.avg_duration.as_millis() as f64,
        providers: state.providers.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check() {
        let response = health_check().await;
        assert_eq!(response.0.status, "healthy");
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "healthy".to_string(),
            version: "0.1.0".to_string(),
            uptime_seconds: Some(100),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("healthy"));
        assert!(json.contains("0.1.0"));
    }
}
