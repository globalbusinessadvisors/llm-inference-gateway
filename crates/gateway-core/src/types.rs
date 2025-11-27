//! Validated domain types (newtypes) for type-safe API contracts.
//!
//! All domain values use newtype wrappers with compile-time and runtime validation
//! to ensure correctness at the type system level.

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::num::NonZeroU32;
use thiserror::Error;

/// Validation error for domain types
#[derive(Debug, Error)]
pub enum ValidationError {
    /// Temperature value out of range
    #[error("Invalid temperature {value}: must be between {min} and {max}")]
    InvalidTemperature {
        /// The invalid value provided
        value: f32,
        /// Minimum allowed value
        min: f32,
        /// Maximum allowed value
        max: f32,
    },

    /// Max tokens value out of range
    #[error("Invalid max_tokens {value}: must be between {min} and {max}")]
    InvalidMaxTokens {
        /// The invalid value provided
        value: u32,
        /// Minimum allowed value
        min: u32,
        /// Maximum allowed value
        max: u32,
    },

    /// Top-p value out of range
    #[error("Invalid top_p {value}: must be between {min} and {max}")]
    InvalidTopP {
        /// The invalid value provided
        value: f32,
        /// Minimum allowed value
        min: f32,
        /// Maximum allowed value
        max: f32,
    },

    /// Top-k value out of range
    #[error("Invalid top_k {value}: must be at least {min}")]
    InvalidTopK {
        /// The invalid value provided
        value: u32,
        /// Minimum allowed value
        min: u32,
    },

    /// Model ID validation failed
    #[error("Invalid model_id: {reason}")]
    InvalidModelId {
        /// Reason for validation failure
        reason: String,
    },

    /// Request ID validation failed
    #[error("Invalid request_id: {reason}")]
    InvalidRequestId {
        /// Reason for validation failure
        reason: String,
    },

    /// Tenant ID validation failed
    #[error("Invalid tenant_id: {reason}")]
    InvalidTenantId {
        /// Reason for validation failure
        reason: String,
    },

    /// Provider ID validation failed
    #[error("Invalid provider_id: {reason}")]
    InvalidProviderId {
        /// Reason for validation failure
        reason: String,
    },

    /// API key validation failed
    #[error("Invalid api_key: {reason}")]
    InvalidApiKey {
        /// Reason for validation failure
        reason: String,
    },
}

/// Temperature for sampling (0.0 to 2.0)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct Temperature(f32);

impl Temperature {
    /// Minimum allowed temperature
    pub const MIN: f32 = 0.0;
    /// Maximum allowed temperature
    pub const MAX: f32 = 2.0;
    /// Default temperature
    pub const DEFAULT: f32 = 1.0;

    /// Create a new temperature value with validation
    ///
    /// # Errors
    /// Returns `ValidationError::InvalidTemperature` if value is outside [0.0, 2.0]
    pub fn new(value: f32) -> Result<Self, ValidationError> {
        if (Self::MIN..=Self::MAX).contains(&value) {
            Ok(Self(value))
        } else {
            Err(ValidationError::InvalidTemperature {
                value,
                min: Self::MIN,
                max: Self::MAX,
            })
        }
    }

    /// Get the inner value
    #[must_use]
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Default for Temperature {
    fn default() -> Self {
        Self(Self::DEFAULT)
    }
}

impl TryFrom<f32> for Temperature {
    type Error = ValidationError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Temperature> for f32 {
    fn from(temp: Temperature) -> Self {
        temp.0
    }
}

impl fmt::Display for Temperature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Maximum tokens to generate (1 to 128,000)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct MaxTokens(NonZeroU32);

impl MaxTokens {
    /// Minimum allowed max_tokens
    pub const MIN: u32 = 1;
    /// Maximum allowed max_tokens
    pub const MAX: u32 = 128_000;

    /// Create a new max_tokens value with validation
    ///
    /// # Errors
    /// Returns `ValidationError::InvalidMaxTokens` if value is outside [1, 128000]
    pub fn new(value: u32) -> Result<Self, ValidationError> {
        if !(Self::MIN..=Self::MAX).contains(&value) {
            return Err(ValidationError::InvalidMaxTokens {
                value,
                min: Self::MIN,
                max: Self::MAX,
            });
        }
        // SAFETY: We've verified value >= 1
        NonZeroU32::new(value)
            .map(Self)
            .ok_or(ValidationError::InvalidMaxTokens {
                value,
                min: Self::MIN,
                max: Self::MAX,
            })
    }

    /// Get the inner value
    #[must_use]
    pub fn value(&self) -> u32 {
        self.0.get()
    }
}

impl TryFrom<u32> for MaxTokens {
    type Error = ValidationError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<MaxTokens> for u32 {
    fn from(tokens: MaxTokens) -> Self {
        tokens.value()
    }
}

impl fmt::Display for MaxTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Top-p (nucleus sampling) parameter (0.0 < p <= 1.0)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct TopP(f32);

impl TopP {
    /// Minimum allowed top_p (exclusive)
    pub const MIN: f32 = 0.0;
    /// Maximum allowed top_p (inclusive)
    pub const MAX: f32 = 1.0;

    /// Create a new top_p value with validation
    ///
    /// # Errors
    /// Returns `ValidationError::InvalidTopP` if value is outside (0.0, 1.0]
    pub fn new(value: f32) -> Result<Self, ValidationError> {
        if value > Self::MIN && value <= Self::MAX {
            Ok(Self(value))
        } else {
            Err(ValidationError::InvalidTopP {
                value,
                min: Self::MIN,
                max: Self::MAX,
            })
        }
    }

    /// Get the inner value
    #[must_use]
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for TopP {
    type Error = ValidationError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<TopP> for f32 {
    fn from(top_p: TopP) -> Self {
        top_p.0
    }
}

impl fmt::Display for TopP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Top-k sampling parameter (>= 1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct TopK(NonZeroU32);

impl TopK {
    /// Minimum allowed top_k
    pub const MIN: u32 = 1;

    /// Create a new top_k value with validation
    ///
    /// # Errors
    /// Returns `ValidationError::InvalidTopK` if value is less than 1
    pub fn new(value: u32) -> Result<Self, ValidationError> {
        NonZeroU32::new(value)
            .map(Self)
            .ok_or(ValidationError::InvalidTopK {
                value,
                min: Self::MIN,
            })
    }

    /// Get the inner value
    #[must_use]
    pub fn value(&self) -> u32 {
        self.0.get()
    }
}

impl TryFrom<u32> for TopK {
    type Error = ValidationError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<TopK> for u32 {
    fn from(top_k: TopK) -> Self {
        top_k.value()
    }
}

impl fmt::Display for TopK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Model identifier (non-empty, max 256 chars)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ModelId(String);

impl ModelId {
    /// Maximum length for model ID
    pub const MAX_LENGTH: usize = 256;

    /// Create a new model ID with validation
    ///
    /// # Errors
    /// Returns `ValidationError::InvalidModelId` if empty or exceeds max length
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ValidationError::InvalidModelId {
                reason: "model_id cannot be empty".to_string(),
            });
        }
        if value.len() > Self::MAX_LENGTH {
            return Err(ValidationError::InvalidModelId {
                reason: format!("model_id exceeds maximum length of {}", Self::MAX_LENGTH),
            });
        }
        Ok(Self(value))
    }

    /// Get the inner value as a string slice
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ModelId {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ModelId> for String {
    fn from(id: ModelId) -> Self {
        id.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for ModelId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Request identifier (non-empty, max 128 chars)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RequestId(String);

impl RequestId {
    /// Maximum length for request ID
    pub const MAX_LENGTH: usize = 128;

    /// Create a new request ID with validation
    ///
    /// # Errors
    /// Returns `ValidationError::InvalidRequestId` if empty or exceeds max length
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ValidationError::InvalidRequestId {
                reason: "request_id cannot be empty".to_string(),
            });
        }
        if value.len() > Self::MAX_LENGTH {
            return Err(ValidationError::InvalidRequestId {
                reason: format!("request_id exceeds maximum length of {}", Self::MAX_LENGTH),
            });
        }
        Ok(Self(value))
    }

    /// Generate a new UUID-based request ID
    #[must_use]
    pub fn generate() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    /// Get the inner value as a string slice
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for RequestId {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<RequestId> for String {
    fn from(id: RequestId) -> Self {
        id.0
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::generate()
    }
}

/// Tenant identifier (alphanumeric plus hyphen/underscore)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct TenantId(String);

impl TenantId {
    /// Maximum length for tenant ID
    pub const MAX_LENGTH: usize = 64;

    /// Create a new tenant ID with validation
    ///
    /// # Errors
    /// Returns `ValidationError::InvalidTenantId` if empty, exceeds max length,
    /// or contains invalid characters
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ValidationError::InvalidTenantId {
                reason: "tenant_id cannot be empty".to_string(),
            });
        }
        if value.len() > Self::MAX_LENGTH {
            return Err(ValidationError::InvalidTenantId {
                reason: format!("tenant_id exceeds maximum length of {}", Self::MAX_LENGTH),
            });
        }
        // Validate characters: alphanumeric, hyphen, underscore
        if !value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(ValidationError::InvalidTenantId {
                reason: "tenant_id must contain only alphanumeric characters, hyphens, or underscores".to_string(),
            });
        }
        Ok(Self(value))
    }

    /// Get the inner value as a string slice
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for TenantId {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<TenantId> for String {
    fn from(id: TenantId) -> Self {
        id.0
    }
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Provider identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ProviderId(String);

impl ProviderId {
    /// Maximum length for provider ID
    pub const MAX_LENGTH: usize = 64;

    /// Create a new provider ID with validation
    ///
    /// # Errors
    /// Returns `ValidationError::InvalidProviderId` if empty or exceeds max length
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ValidationError::InvalidProviderId {
                reason: "provider_id cannot be empty".to_string(),
            });
        }
        if value.len() > Self::MAX_LENGTH {
            return Err(ValidationError::InvalidProviderId {
                reason: format!("provider_id exceeds maximum length of {}", Self::MAX_LENGTH),
            });
        }
        Ok(Self(value))
    }

    /// Get the inner value as a string slice
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ProviderId {
    type Error = ValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ProviderId> for String {
    fn from(id: ProviderId) -> Self {
        id.0
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for ProviderId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// API key (sensitive, never logged)
#[derive(Clone)]
pub struct ApiKey(SecretString);

impl ApiKey {
    /// Create a new API key
    ///
    /// # Errors
    /// Returns `ValidationError::InvalidApiKey` if the key is empty
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ValidationError::InvalidApiKey {
                reason: "api_key cannot be empty".to_string(),
            });
        }
        Ok(Self(SecretString::new(value)))
    }

    /// Expose the secret value (use sparingly)
    #[must_use]
    pub fn expose_secret(&self) -> &str {
        self.0.expose_secret()
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ApiKey([REDACTED])")
    }
}

impl fmt::Display for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[REDACTED]")
    }
}

impl<'de> Deserialize<'de> for ApiKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::new(s).map_err(serde::de::Error::custom)
    }
}

impl Serialize for ApiKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Never serialize the actual key
        serializer.serialize_str("[REDACTED]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_valid() {
        assert!(Temperature::new(0.0).is_ok());
        assert!(Temperature::new(1.0).is_ok());
        assert!(Temperature::new(2.0).is_ok());
        assert!(Temperature::new(0.7).is_ok());
    }

    #[test]
    fn test_temperature_invalid() {
        assert!(Temperature::new(-0.1).is_err());
        assert!(Temperature::new(2.1).is_err());
        assert!(Temperature::new(f32::NAN).is_err());
    }

    #[test]
    fn test_max_tokens_valid() {
        assert!(MaxTokens::new(1).is_ok());
        assert!(MaxTokens::new(1000).is_ok());
        assert!(MaxTokens::new(128_000).is_ok());
    }

    #[test]
    fn test_max_tokens_invalid() {
        assert!(MaxTokens::new(0).is_err());
        assert!(MaxTokens::new(128_001).is_err());
    }

    #[test]
    fn test_top_p_valid() {
        assert!(TopP::new(0.1).is_ok());
        assert!(TopP::new(0.5).is_ok());
        assert!(TopP::new(1.0).is_ok());
    }

    #[test]
    fn test_top_p_invalid() {
        assert!(TopP::new(0.0).is_err());
        assert!(TopP::new(-0.1).is_err());
        assert!(TopP::new(1.1).is_err());
    }

    #[test]
    fn test_model_id_valid() {
        assert!(ModelId::new("gpt-4").is_ok());
        assert!(ModelId::new("claude-3-opus").is_ok());
    }

    #[test]
    fn test_model_id_invalid() {
        assert!(ModelId::new("").is_err());
        assert!(ModelId::new("a".repeat(257)).is_err());
    }

    #[test]
    fn test_tenant_id_valid() {
        assert!(TenantId::new("tenant-123").is_ok());
        assert!(TenantId::new("my_tenant").is_ok());
        assert!(TenantId::new("ABC123").is_ok());
    }

    #[test]
    fn test_tenant_id_invalid() {
        assert!(TenantId::new("").is_err());
        assert!(TenantId::new("tenant@invalid").is_err());
        assert!(TenantId::new("tenant with space").is_err());
    }

    #[test]
    fn test_api_key_redacted() {
        let key = ApiKey::new("sk-secret-key").expect("valid key");
        assert_eq!(format!("{key}"), "[REDACTED]");
        assert_eq!(format!("{key:?}"), "ApiKey([REDACTED])");
        assert_eq!(key.expose_secret(), "sk-secret-key");
    }
}
