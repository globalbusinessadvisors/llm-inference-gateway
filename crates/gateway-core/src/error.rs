//! Error types and handling for the gateway.
//!
//! This module provides a comprehensive error hierarchy that maps to appropriate
//! HTTP status codes and provides structured error information for clients.

use crate::types::ValidationError;
use http::StatusCode;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;
use thiserror::Error;

/// Result type alias using `GatewayError`
pub type GatewayResult<T> = Result<T, GatewayError>;

/// Comprehensive gateway error type covering all error scenarios
#[derive(Debug, Error)]
pub enum GatewayError {
    /// Request validation failed
    #[error("Validation error: {message}")]
    Validation {
        /// Error message
        message: String,
        /// Field that failed validation (if applicable)
        field: Option<String>,
        /// Error code for programmatic handling
        code: String,
    },

    /// Authentication failed
    #[error("Authentication failed: {message}")]
    Authentication {
        /// Error message
        message: String,
    },

    /// Authorization denied
    #[error("Authorization denied: {message}")]
    Authorization {
        /// Error message
        message: String,
    },

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimit {
        /// Duration to wait before retrying
        retry_after: Option<Duration>,
        /// Rate limit that was exceeded
        limit: Option<u32>,
    },

    /// Provider error
    #[error("Provider error: {provider} - {message}")]
    Provider {
        /// Provider that returned the error
        provider: String,
        /// Error message
        message: String,
        /// HTTP status code from provider (if applicable)
        status_code: Option<u16>,
        /// Whether this error is retryable
        retryable: bool,
    },

    /// Circuit breaker is open
    #[error("Circuit breaker open for provider: {provider}")]
    CircuitBreakerOpen {
        /// Provider with open circuit breaker
        provider: String,
    },

    /// Request timed out
    #[error("Request timeout after {duration:?}")]
    Timeout {
        /// Duration after which the request timed out
        duration: Duration,
    },

    /// No healthy providers available
    #[error("No healthy providers available for model: {model}")]
    NoHealthyProviders {
        /// Model that was requested
        model: String,
    },

    /// Model not found
    #[error("Model not found: {model}")]
    ModelNotFound {
        /// Model that was not found
        model: String,
    },

    /// Provider not found
    #[error("Provider not found: {provider}")]
    ProviderNotFound {
        /// Provider that was not found
        provider: String,
    },

    /// Request payload too large
    #[error("Request payload too large: {size} bytes exceeds limit of {limit} bytes")]
    PayloadTooLarge {
        /// Actual size of the payload
        size: usize,
        /// Maximum allowed size
        limit: usize,
    },

    /// Streaming error
    #[error("Streaming error: {message}")]
    Streaming {
        /// Error message
        message: String,
    },

    /// Configuration error
    #[error("Configuration error: {message}")]
    Configuration {
        /// Error message
        message: String,
    },

    /// Internal server error
    #[error("Internal error: {message}")]
    Internal {
        /// Error message
        message: String,
    },
}

impl GatewayError {
    /// Get the HTTP status code for this error
    #[must_use]
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::Validation { .. } => StatusCode::BAD_REQUEST,
            Self::Authentication { .. } => StatusCode::UNAUTHORIZED,
            Self::Authorization { .. } => StatusCode::FORBIDDEN,
            Self::RateLimit { .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::Provider { status_code, .. } => status_code
                .and_then(|code| StatusCode::from_u16(code).ok())
                .unwrap_or(StatusCode::BAD_GATEWAY),
            Self::CircuitBreakerOpen { .. } | Self::NoHealthyProviders { .. } => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            Self::Timeout { .. } => StatusCode::GATEWAY_TIMEOUT,
            Self::ModelNotFound { .. } | Self::ProviderNotFound { .. } => StatusCode::NOT_FOUND,
            Self::PayloadTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            Self::Streaming { .. } | Self::Configuration { .. } | Self::Internal { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    }

    /// Check if this error is retryable
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Provider { retryable, .. } => *retryable,
            Self::Timeout { .. }
            | Self::RateLimit { .. }
            | Self::CircuitBreakerOpen { .. }
            | Self::NoHealthyProviders { .. }
            | Self::Streaming { .. } => true,
            _ => false,
        }
    }

    /// Get the error type string for API responses
    #[must_use]
    pub fn error_type(&self) -> &'static str {
        match self {
            Self::Validation { .. } | Self::PayloadTooLarge { .. } => "invalid_request_error",
            Self::Authentication { .. } => "authentication_error",
            Self::Authorization { .. } => "authorization_error",
            Self::RateLimit { .. } => "rate_limit_error",
            Self::Provider { .. } => "provider_error",
            Self::CircuitBreakerOpen { .. } | Self::NoHealthyProviders { .. } => {
                "service_unavailable_error"
            }
            Self::Timeout { .. } => "timeout_error",
            Self::ModelNotFound { .. } | Self::ProviderNotFound { .. } => "not_found_error",
            Self::Streaming { .. } => "streaming_error",
            Self::Configuration { .. } | Self::Internal { .. } => "internal_error",
        }
    }

    /// Get the error code for programmatic handling
    #[must_use]
    pub fn error_code(&self) -> &str {
        match self {
            Self::Validation { code, .. } => code,
            Self::Authentication { .. } => "authentication_failed",
            Self::Authorization { .. } => "authorization_denied",
            Self::RateLimit { .. } => "rate_limit_exceeded",
            Self::Provider { .. } => "provider_error",
            Self::CircuitBreakerOpen { .. } => "circuit_breaker_open",
            Self::Timeout { .. } => "timeout",
            Self::NoHealthyProviders { .. } => "no_healthy_providers",
            Self::ModelNotFound { .. } => "model_not_found",
            Self::ProviderNotFound { .. } => "provider_not_found",
            Self::PayloadTooLarge { .. } => "payload_too_large",
            Self::Streaming { .. } => "streaming_error",
            Self::Configuration { .. } => "configuration_error",
            Self::Internal { .. } => "internal_error",
        }
    }

    /// Create a validation error
    #[must_use]
    pub fn validation(message: impl Into<String>, field: Option<String>, code: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
            field,
            code: code.into(),
        }
    }

    /// Create an authentication error
    #[must_use]
    pub fn authentication(message: impl Into<String>) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }

    /// Create an authorization error
    #[must_use]
    pub fn authorization(message: impl Into<String>) -> Self {
        Self::Authorization {
            message: message.into(),
        }
    }

    /// Create a provider error
    #[must_use]
    pub fn provider(
        provider: impl Into<String>,
        message: impl Into<String>,
        status_code: Option<u16>,
        retryable: bool,
    ) -> Self {
        Self::Provider {
            provider: provider.into(),
            message: message.into(),
            status_code,
            retryable,
        }
    }

    /// Create an internal error
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Create a timeout error
    #[must_use]
    pub fn timeout(duration: Duration) -> Self {
        Self::Timeout { duration }
    }

    /// Create a model not found error
    #[must_use]
    pub fn model_not_found(model: impl Into<String>) -> Self {
        Self::ModelNotFound {
            model: model.into(),
        }
    }

    /// Create a no healthy providers error
    #[must_use]
    pub fn no_healthy_providers(model: impl Into<String>) -> Self {
        Self::NoHealthyProviders {
            model: model.into(),
        }
    }

    /// Create a circuit breaker open error
    #[must_use]
    pub fn circuit_breaker_open(provider: impl Into<String>) -> Self {
        Self::CircuitBreakerOpen {
            provider: provider.into(),
        }
    }

    /// Create a rate limit error
    #[must_use]
    pub fn rate_limit(retry_after: Option<Duration>, limit: Option<u32>) -> Self {
        Self::RateLimit { retry_after, limit }
    }

    /// Create a streaming error
    #[must_use]
    pub fn streaming(message: impl Into<String>) -> Self {
        Self::Streaming {
            message: message.into(),
        }
    }
}

impl From<ValidationError> for GatewayError {
    fn from(err: ValidationError) -> Self {
        let (field, code) = match &err {
            ValidationError::InvalidTemperature { .. } => (Some("temperature".to_string()), "invalid_temperature"),
            ValidationError::InvalidMaxTokens { .. } => (Some("max_tokens".to_string()), "invalid_max_tokens"),
            ValidationError::InvalidTopP { .. } => (Some("top_p".to_string()), "invalid_top_p"),
            ValidationError::InvalidTopK { .. } => (Some("top_k".to_string()), "invalid_top_k"),
            ValidationError::InvalidModelId { .. } => (Some("model".to_string()), "invalid_model_id"),
            ValidationError::InvalidRequestId { .. } => (Some("request_id".to_string()), "invalid_request_id"),
            ValidationError::InvalidTenantId { .. } => (Some("tenant_id".to_string()), "invalid_tenant_id"),
            ValidationError::InvalidProviderId { .. } => (Some("provider_id".to_string()), "invalid_provider_id"),
            ValidationError::InvalidApiKey { .. } => (Some("api_key".to_string()), "invalid_api_key"),
        };
        Self::Validation {
            message: err.to_string(),
            field,
            code: code.to_string(),
        }
    }
}

/// API error response format (OpenAI compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiErrorResponse {
    /// Error details
    pub error: ApiError,
}

/// API error details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Parameter that caused the error
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

impl From<&GatewayError> for ApiErrorResponse {
    fn from(err: &GatewayError) -> Self {
        let param = match err {
            GatewayError::Validation { field, .. } => field.clone(),
            _ => None,
        };

        Self {
            error: ApiError {
                error_type: err.error_type().to_string(),
                message: err.to_string(),
                code: Some(err.error_code().to_string()),
                param,
            },
        }
    }
}

impl fmt::Display for ApiErrorResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_status_codes() {
        assert_eq!(
            GatewayError::validation("test", None, "test_code").status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            GatewayError::authentication("test").status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            GatewayError::authorization("test").status_code(),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            GatewayError::rate_limit(None, None).status_code(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            GatewayError::timeout(Duration::from_secs(30)).status_code(),
            StatusCode::GATEWAY_TIMEOUT
        );
        assert_eq!(
            GatewayError::model_not_found("gpt-4").status_code(),
            StatusCode::NOT_FOUND
        );
    }

    #[test]
    fn test_error_retryability() {
        assert!(!GatewayError::validation("test", None, "test").is_retryable());
        assert!(!GatewayError::authentication("test").is_retryable());
        assert!(GatewayError::rate_limit(None, None).is_retryable());
        assert!(GatewayError::timeout(Duration::from_secs(30)).is_retryable());
        assert!(GatewayError::circuit_breaker_open("openai").is_retryable());
        assert!(GatewayError::provider("openai", "error", Some(500), true).is_retryable());
        assert!(!GatewayError::provider("openai", "error", Some(400), false).is_retryable());
    }

    #[test]
    fn test_api_error_response() {
        let err = GatewayError::validation("Invalid temperature", Some("temperature".to_string()), "invalid_temperature");
        let response = ApiErrorResponse::from(&err);

        assert_eq!(response.error.error_type, "invalid_request_error");
        assert_eq!(response.error.param, Some("temperature".to_string()));
        assert_eq!(response.error.code, Some("invalid_temperature".to_string()));
    }
}
