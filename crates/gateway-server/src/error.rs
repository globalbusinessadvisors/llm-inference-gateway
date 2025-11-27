//! API error handling.
//!
//! Provides consistent error responses following OpenAI's error format.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use gateway_core::GatewayError;
use serde::{Deserialize, Serialize};
use tracing::error;

/// API error response (OpenAI-compatible format)
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiErrorResponse {
    /// Error details
    pub error: ApiErrorDetail,
}

/// Error detail
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiErrorDetail {
    /// Error message
    pub message: String,
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,
    /// Parameter that caused the error (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    /// Error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// API error wrapper
#[derive(Debug)]
pub struct ApiError {
    /// HTTP status code
    pub status: StatusCode,
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Optional parameter
    pub param: Option<String>,
    /// Optional error code
    pub code: Option<String>,
}

impl ApiError {
    /// Create a new API error
    pub fn new(status: StatusCode, error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status,
            error_type: error_type.into(),
            message: message.into(),
            param: None,
            code: None,
        }
    }

    /// Add parameter info
    #[must_use]
    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    /// Add error code
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Bad request error
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    /// Unauthorized error
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "authentication_error", message)
    }

    /// Forbidden error
    pub fn forbidden(message: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, "permission_error", message)
    }

    /// Not found error
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found_error", message)
    }

    /// Rate limited error
    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::new(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", message)
    }

    /// Internal server error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", message)
    }

    /// Service unavailable error
    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, "server_error", message)
    }

    /// Bad gateway error
    pub fn bad_gateway(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_GATEWAY, "upstream_error", message)
    }

    /// Gateway timeout error
    pub fn gateway_timeout(message: impl Into<String>) -> Self {
        Self::new(StatusCode::GATEWAY_TIMEOUT, "timeout_error", message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ApiErrorResponse {
            error: ApiErrorDetail {
                message: self.message.clone(),
                error_type: self.error_type,
                param: self.param,
                code: self.code,
            },
        };

        error!(
            status = %self.status,
            message = %self.message,
            "API error response"
        );

        (self.status, Json(body)).into_response()
    }
}

impl From<GatewayError> for ApiError {
    fn from(err: GatewayError) -> Self {
        match &err {
            GatewayError::Validation { message, field, .. } => {
                let mut api_err = Self::bad_request(message);
                if let Some(f) = field {
                    api_err = api_err.with_param(f);
                }
                api_err
            }
            GatewayError::Authentication { message } => {
                Self::unauthorized(message)
            }
            GatewayError::Authorization { message } => {
                Self::forbidden(message)
            }
            GatewayError::ModelNotFound { model } => {
                Self::not_found(format!("Model not found: {model}"))
            }
            GatewayError::ProviderNotFound { provider } => {
                Self::not_found(format!("Provider not found: {provider}"))
            }
            GatewayError::NoHealthyProviders { model } => {
                Self::service_unavailable(format!("No healthy provider available for model: {model}"))
            }
            GatewayError::RateLimit { retry_after, .. } => {
                let msg = if let Some(after) = retry_after {
                    format!("Rate limited. Retry after {} seconds", after.as_secs())
                } else {
                    "Rate limited".to_string()
                };
                Self::rate_limited(msg)
            }
            GatewayError::Timeout { duration } => {
                Self::gateway_timeout(format!("Request timed out after {duration:?}"))
            }
            GatewayError::Provider { provider, message, retryable, .. } => {
                if *retryable {
                    Self::bad_gateway(format!("{provider}: {message}"))
                } else {
                    Self::internal(format!("{provider}: {message}"))
                }
            }
            GatewayError::CircuitBreakerOpen { provider } => {
                Self::service_unavailable(format!("Provider {provider} is temporarily unavailable"))
            }
            GatewayError::PayloadTooLarge { size, limit } => {
                Self::bad_request(format!(
                    "Payload too large: {size} bytes exceeds limit of {limit} bytes"
                ))
            }
            GatewayError::Streaming { message } => {
                Self::internal(format!("Streaming error: {message}"))
            }
            GatewayError::Configuration { message } => {
                Self::internal(format!("Configuration error: {message}"))
            }
            GatewayError::Internal { message } => {
                Self::internal(message)
            }
        }
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(err: serde_json::Error) -> Self {
        Self::bad_request(format!("JSON parse error: {err}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = ApiError::bad_request("Invalid parameter")
            .with_param("model")
            .with_code("invalid_model");

        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.error_type, "invalid_request_error");
        assert_eq!(err.param, Some("model".to_string()));
        assert_eq!(err.code, Some("invalid_model".to_string()));
    }

    #[test]
    fn test_gateway_error_conversion() {
        let gateway_err = GatewayError::Validation {
            message: "Test error".to_string(),
            field: None,
            code: "test_code".to_string(),
        };
        let api_err: ApiError = gateway_err.into();

        assert_eq!(api_err.status, StatusCode::BAD_REQUEST);
        assert!(api_err.message.contains("Test error"));
    }

    #[test]
    fn test_rate_limit_error() {
        let gateway_err = GatewayError::RateLimit {
            retry_after: Some(std::time::Duration::from_secs(60)),
            limit: None,
        };
        let api_err: ApiError = gateway_err.into();

        assert_eq!(api_err.status, StatusCode::TOO_MANY_REQUESTS);
        assert!(api_err.message.contains("60"));
    }
}
