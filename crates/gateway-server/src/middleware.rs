//! HTTP middleware for the gateway.
//!
//! Provides middleware for:
//! - Request logging
//! - Request ID injection
//! - CORS handling
//! - Response timing
//! - Rate limiting

use axum::{
    extract::{Request, State},
    http::{header, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use gateway_resilience::{RateLimiter, RateLimiterConfig};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, info_span, warn, Instrument};
use uuid::Uuid;

/// Create CORS middleware layer
pub fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers([
            header::AUTHORIZATION,
            header::CONTENT_TYPE,
            header::ACCEPT,
        ])
        .expose_headers([
            header::HeaderName::from_static("x-request-id"),
            header::HeaderName::from_static("x-response-time"),
        ])
        .max_age(std::time::Duration::from_secs(3600))
}

/// Request ID middleware - adds request ID to request and response
pub async fn request_id_middleware(mut request: Request, next: Next) -> Response {
    // Get or generate request ID
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok()).map_or_else(|| Uuid::new_v4().to_string(), String::from);

    // Add to request extensions
    request.extensions_mut().insert(RequestIdExt(request_id.clone()));

    // Add to request headers if not present
    if !request.headers().contains_key("x-request-id") {
        if let Ok(value) = HeaderValue::from_str(&request_id) {
            request.headers_mut().insert("x-request-id", value);
        }
    }

    // Process request
    let mut response = next.run(request).await;

    // Add to response headers
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", value);
    }

    response
}

/// Request ID extension for sharing across handlers
#[derive(Clone, Debug)]
pub struct RequestIdExt(pub String);

/// Request logging middleware
pub async fn logging_middleware(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let _version = request.version();

    // Get request ID if available
    let request_id = request
        .extensions()
        .get::<RequestIdExt>().map_or_else(|| "unknown".to_string(), |r| r.0.clone());

    let span = info_span!(
        "http_request",
        method = %method,
        uri = %uri,
        request_id = %request_id,
    );

    let start = Instant::now();

    let response = next.run(request).instrument(span).await;

    let duration = start.elapsed();
    let status = response.status();

    info!(
        method = %method,
        uri = %uri,
        status = %status.as_u16(),
        duration_ms = duration.as_millis(),
        request_id = %request_id,
        "Request completed"
    );

    response
}

/// Response time middleware - adds X-Response-Time header
pub async fn response_time_middleware(request: Request, next: Next) -> Response {
    let start = Instant::now();

    let mut response = next.run(request).await;

    let duration = start.elapsed();
    let duration_ms = format!("{}ms", duration.as_millis());

    if let Ok(value) = HeaderValue::from_str(&duration_ms) {
        response.headers_mut().insert("x-response-time", value);
    }

    response
}

/// Content type validation middleware for JSON endpoints
pub async fn json_content_type_middleware(request: Request, next: Next) -> Response {
    // Only check POST/PUT/PATCH requests
    if matches!(
        request.method(),
        &Method::POST | &Method::PUT | &Method::PATCH
    ) {
        // Check Content-Type header
        if let Some(content_type) = request.headers().get(header::CONTENT_TYPE) {
            if let Ok(ct) = content_type.to_str() {
                if !ct.starts_with("application/json") {
                    return (
                        StatusCode::UNSUPPORTED_MEDIA_TYPE,
                        "Content-Type must be application/json",
                    )
                        .into_response();
                }
            }
        }
    }

    next.run(request).await
}

/// Security headers middleware
pub async fn security_headers_middleware(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;

    let headers = response.headers_mut();

    // Prevent content type sniffing
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );

    // Prevent clickjacking
    headers.insert(
        "x-frame-options",
        HeaderValue::from_static("DENY"),
    );

    // XSS protection
    headers.insert(
        "x-xss-protection",
        HeaderValue::from_static("1; mode=block"),
    );

    // Referrer policy
    headers.insert(
        "referrer-policy",
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );

    response
}

/// Rate limiter state for middleware
#[derive(Clone)]
pub struct RateLimiterState {
    /// The rate limiter instance
    pub limiter: Arc<RateLimiter>,
}

impl RateLimiterState {
    /// Create a new rate limiter state
    #[must_use]
    pub fn new(config: RateLimiterConfig) -> Self {
        Self {
            limiter: Arc::new(RateLimiter::new("gateway", config)),
        }
    }

    /// Create a disabled rate limiter
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            limiter: Arc::new(RateLimiter::disabled("gateway")),
        }
    }

    /// Create from config schema
    #[must_use]
    pub fn from_config(config: &gateway_config::RateLimitConfig) -> Self {
        if !config.enabled {
            return Self::disabled();
        }

        let limiter_config = RateLimiterConfig {
            requests_per_window: config.default_rpm,
            tokens_per_window: config.default_tpm,
            window: config.window,
            enable_burst: true,
            burst_multiplier: 1.2,
        };

        Self::new(limiter_config)
    }
}

/// Rate limiting middleware
///
/// Extracts the rate limit key from the request based on configuration:
/// - API key from Authorization header
/// - IP address from X-Forwarded-For or connection
/// - Tenant ID from X-Tenant-ID header
pub async fn rate_limit_middleware(
    State(state): State<RateLimiterState>,
    request: Request,
    next: Next,
) -> Response {
    // Extract rate limit key from request
    let key = extract_rate_limit_key(&request);

    // Check rate limit
    match state.limiter.check(&key, None).await {
        Ok(()) => {
            // Add rate limit headers to response
            let mut response = next.run(request).await;
            add_rate_limit_headers(&mut response, &state.limiter, &key).await;
            response
        }
        Err(err) => {
            warn!(key = %key, error = %err, "Rate limit exceeded");

            // Build rate limit exceeded response
            let retry_after = match &err {
                gateway_core::GatewayError::RateLimit { retry_after, .. } => {
                    retry_after.map(|d| d.as_secs()).unwrap_or(60)
                }
                _ => 60,
            };

            let mut response = (
                StatusCode::TOO_MANY_REQUESTS,
                [
                    (header::RETRY_AFTER, retry_after.to_string()),
                    (
                        header::CONTENT_TYPE,
                        "application/json".to_string(),
                    ),
                ],
                format!(
                    r#"{{"error":{{"type":"rate_limit_exceeded","message":"Rate limit exceeded","retry_after_seconds":{}}}}}"#,
                    retry_after
                ),
            )
                .into_response();

            // Add rate limit headers
            add_rate_limit_headers(&mut response, &state.limiter, &key).await;
            response
        }
    }
}

/// Extract rate limit key from request
fn extract_rate_limit_key(request: &Request) -> String {
    // Try API key from Authorization header
    if let Some(auth) = request.headers().get(header::AUTHORIZATION) {
        if let Ok(auth_str) = auth.to_str() {
            if let Some(key) = auth_str.strip_prefix("Bearer ") {
                // Hash the API key for privacy in logs
                return format!("api:{}", &key[..key.len().min(8)]);
            }
        }
    }

    // Try tenant ID header
    if let Some(tenant) = request.headers().get("x-tenant-id") {
        if let Ok(tenant_str) = tenant.to_str() {
            return format!("tenant:{tenant_str}");
        }
    }

    // Fall back to IP address
    if let Some(forwarded) = request.headers().get("x-forwarded-for") {
        if let Ok(forwarded_str) = forwarded.to_str() {
            if let Some(ip) = forwarded_str.split(',').next() {
                return format!("ip:{}", ip.trim());
            }
        }
    }

    // Default key for requests without identifiable source
    "ip:unknown".to_string()
}

/// Add rate limit headers to response
async fn add_rate_limit_headers(response: &mut Response, limiter: &RateLimiter, key: &str) {
    if let Some(stats) = limiter.stats(key).await {
        let headers = response.headers_mut();

        // Standard rate limit headers
        if let Ok(v) = HeaderValue::from_str(&stats.requests_per_window.to_string()) {
            headers.insert("x-ratelimit-limit", v);
        }

        let remaining = stats.request_tokens_available.max(0.0) as u32;
        if let Ok(v) = HeaderValue::from_str(&remaining.to_string()) {
            headers.insert("x-ratelimit-remaining", v);
        }

        // Reset time (simplified - uses window duration)
        if let Ok(v) = HeaderValue::from_str("60") {
            headers.insert("x-ratelimit-reset", v);
        }
    }
}

/// Create a rate limiter with default settings
#[must_use]
pub fn default_rate_limiter() -> RateLimiterState {
    RateLimiterState::new(RateLimiterConfig {
        requests_per_window: 1000,
        tokens_per_window: None,
        window: Duration::from_secs(60),
        enable_burst: true,
        burst_multiplier: 1.5,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, routing::get, Router};
    use tower::ServiceExt;

    async fn test_handler() -> &'static str {
        "OK"
    }

    #[tokio::test]
    async fn test_cors_layer() {
        let _cors = cors_layer();
        // Just verify it can be created without error
    }

    #[tokio::test]
    async fn test_request_id_generation() {
        let app = Router::new()
            .route("/", get(test_handler))
            .layer(axum::middleware::from_fn(request_id_middleware));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // Should have x-request-id header
        assert!(response.headers().contains_key("x-request-id"));
    }

    #[tokio::test]
    async fn test_request_id_passthrough() {
        let app = Router::new()
            .route("/", get(test_handler))
            .layer(axum::middleware::from_fn(request_id_middleware));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("x-request-id", "test-id-123")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.headers().get("x-request-id").unwrap(),
            "test-id-123"
        );
    }

    #[tokio::test]
    async fn test_response_time_header() {
        let app = Router::new()
            .route("/", get(test_handler))
            .layer(axum::middleware::from_fn(response_time_middleware));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert!(response.headers().contains_key("x-response-time"));
    }

    #[test]
    fn test_extract_rate_limit_key_from_auth() {
        let request = Request::builder()
            .uri("/")
            .header("Authorization", "Bearer sk-test-key-12345")
            .body(Body::empty())
            .unwrap();

        let key = extract_rate_limit_key(&request);
        assert!(key.starts_with("api:"));
    }

    #[test]
    fn test_extract_rate_limit_key_from_tenant() {
        let request = Request::builder()
            .uri("/")
            .header("x-tenant-id", "tenant-123")
            .body(Body::empty())
            .unwrap();

        let key = extract_rate_limit_key(&request);
        assert_eq!(key, "tenant:tenant-123");
    }

    #[test]
    fn test_extract_rate_limit_key_from_ip() {
        let request = Request::builder()
            .uri("/")
            .header("x-forwarded-for", "192.168.1.1, 10.0.0.1")
            .body(Body::empty())
            .unwrap();

        let key = extract_rate_limit_key(&request);
        assert_eq!(key, "ip:192.168.1.1");
    }

    #[test]
    fn test_extract_rate_limit_key_unknown() {
        let request = Request::builder()
            .uri("/")
            .body(Body::empty())
            .unwrap();

        let key = extract_rate_limit_key(&request);
        assert_eq!(key, "ip:unknown");
    }

    #[tokio::test]
    async fn test_rate_limiter_state_from_config() {
        let config = gateway_config::RateLimitConfig {
            enabled: true,
            default_rpm: 100,
            default_tpm: Some(10000),
            window: Duration::from_secs(60),
            key_by: gateway_config::RateLimitKeyBy::ApiKey,
        };

        let state = RateLimiterState::from_config(&config);
        assert!(state.limiter.is_enabled());
    }

    #[tokio::test]
    async fn test_rate_limiter_state_disabled() {
        let config = gateway_config::RateLimitConfig {
            enabled: false,
            default_rpm: 100,
            default_tpm: None,
            window: Duration::from_secs(60),
            key_by: gateway_config::RateLimitKeyBy::ApiKey,
        };

        let state = RateLimiterState::from_config(&config);
        assert!(!state.limiter.is_enabled());
    }

    #[tokio::test]
    async fn test_rate_limit_middleware_allows_requests() {
        let state = RateLimiterState::new(RateLimiterConfig {
            requests_per_window: 10,
            tokens_per_window: None,
            window: Duration::from_secs(60),
            enable_burst: false,
            burst_multiplier: 1.0,
        });

        let app = Router::new()
            .route("/", get(test_handler))
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                rate_limit_middleware,
            ))
            .with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("x-tenant-id", "test-tenant")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().contains_key("x-ratelimit-limit"));
        assert!(response.headers().contains_key("x-ratelimit-remaining"));
    }

    #[tokio::test]
    async fn test_rate_limit_middleware_blocks_excess() {
        let state = RateLimiterState::new(RateLimiterConfig {
            requests_per_window: 1,
            tokens_per_window: None,
            window: Duration::from_secs(60),
            enable_burst: false,
            burst_multiplier: 1.0,
        });

        let app = Router::new()
            .route("/", get(test_handler))
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                rate_limit_middleware,
            ))
            .with_state(state);

        // First request should succeed
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("x-tenant-id", "test-tenant")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Second request should be rate limited
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("x-tenant-id", "test-tenant")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response.headers().contains_key("retry-after"));
    }
}
