//! HTTP middleware for the gateway.
//!
//! Provides middleware for:
//! - Request logging
//! - Request ID injection
//! - CORS handling
//! - Response timing

use axum::{
    extract::Request,
    http::{header, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, info_span, Instrument};
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
        let cors = cors_layer();
        // Just verify it can be created
        assert!(true);
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
}
