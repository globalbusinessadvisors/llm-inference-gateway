//! Provider traits and abstractions.
//!
//! This module defines the core trait that all LLM providers must implement,
//! along with supporting types for capabilities and health status.

use crate::error::GatewayError;
use crate::request::GatewayRequest;
use crate::response::GatewayResponse;
use crate::streaming::ChatChunk;
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;

/// Core trait for all LLM providers
///
/// All provider implementations must implement this trait to be usable
/// with the gateway's routing and load balancing systems.
#[async_trait]
pub trait LLMProvider: Send + Sync + 'static {
    /// Unique provider instance identifier
    fn id(&self) -> &str;

    /// Provider type (e.g., OpenAI, Anthropic)
    fn provider_type(&self) -> ProviderType;

    /// Execute a synchronous chat completion
    ///
    /// # Errors
    /// Returns `GatewayError` on provider errors, timeouts, or validation failures
    async fn chat_completion(
        &self,
        request: &GatewayRequest,
    ) -> Result<GatewayResponse, GatewayError>;

    /// Execute a streaming chat completion
    ///
    /// # Errors
    /// Returns `GatewayError` on provider errors, timeouts, or validation failures
    async fn chat_completion_stream(
        &self,
        request: &GatewayRequest,
    ) -> Result<BoxStream<'static, Result<ChatChunk, GatewayError>>, GatewayError>;

    /// Perform a health check on this provider
    async fn health_check(&self) -> HealthStatus;

    /// Get provider capabilities
    fn capabilities(&self) -> &ProviderCapabilities;

    /// Get list of supported models
    fn models(&self) -> &[ModelInfo];

    /// Get the base URL for this provider
    fn base_url(&self) -> &str;

    /// Check if a specific model is supported
    fn supports_model(&self, model: &str) -> bool {
        self.models().iter().any(|m| m.id == model || m.aliases.contains(&model.to_string()))
    }

    /// Get the configured timeout for this provider
    fn timeout(&self) -> Duration {
        Duration::from_secs(120)
    }

    /// Check if provider is currently enabled
    fn is_enabled(&self) -> bool {
        true
    }
}

/// Provider type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// OpenAI API
    OpenAI,
    /// Anthropic API
    Anthropic,
    /// Google AI (Gemini)
    Google,
    /// Azure OpenAI Service
    Azure,
    /// AWS Bedrock
    Bedrock,
    /// vLLM (self-hosted)
    VLLM,
    /// Ollama (self-hosted)
    Ollama,
    /// Together AI
    Together,
    /// Custom/other provider
    Custom,
}

impl fmt::Display for ProviderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenAI => write!(f, "openai"),
            Self::Anthropic => write!(f, "anthropic"),
            Self::Google => write!(f, "google"),
            Self::Azure => write!(f, "azure"),
            Self::Bedrock => write!(f, "bedrock"),
            Self::VLLM => write!(f, "vllm"),
            Self::Ollama => write!(f, "ollama"),
            Self::Together => write!(f, "together"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

impl std::str::FromStr for ProviderType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Self::OpenAI),
            "anthropic" => Ok(Self::Anthropic),
            "google" | "gemini" => Ok(Self::Google),
            "azure" | "azure_openai" | "azure-openai" => Ok(Self::Azure),
            "bedrock" | "aws_bedrock" | "aws-bedrock" => Ok(Self::Bedrock),
            "vllm" => Ok(Self::VLLM),
            "ollama" => Ok(Self::Ollama),
            "together" | "together_ai" | "together-ai" => Ok(Self::Together),
            "custom" => Ok(Self::Custom),
            _ => Err(format!("Unknown provider type: {s}")),
        }
    }
}

/// Provider health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum HealthStatus {
    /// Provider is healthy and accepting requests
    Healthy,
    /// Provider is degraded (slow or partial failures)
    Degraded,
    /// Provider is unhealthy and should not receive requests
    Unhealthy,
    /// Health status is unknown (e.g., never checked)
    #[default]
    Unknown,
}

impl HealthStatus {
    /// Check if the provider should receive traffic
    #[must_use]
    pub fn should_route(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }

    /// Check if the status indicates a problem
    #[must_use]
    pub fn is_problematic(&self) -> bool {
        matches!(self, Self::Unhealthy)
    }
}


impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::Unhealthy => write!(f, "unhealthy"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Provider capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Supports chat completion
    #[serde(default = "default_true")]
    pub chat: bool,

    /// Supports streaming responses
    #[serde(default = "default_true")]
    pub streaming: bool,

    /// Supports function/tool calling
    #[serde(default)]
    pub function_calling: bool,

    /// Supports vision (image inputs)
    #[serde(default)]
    pub vision: bool,

    /// Supports embeddings
    #[serde(default)]
    pub embeddings: bool,

    /// Supports JSON mode
    #[serde(default)]
    pub json_mode: bool,

    /// Supports seed for reproducibility
    #[serde(default)]
    pub seed: bool,

    /// Supports log probabilities
    #[serde(default)]
    pub logprobs: bool,

    /// Maximum context window size
    #[serde(default)]
    pub max_context_length: Option<u32>,

    /// Maximum output tokens
    #[serde(default)]
    pub max_output_tokens: Option<u32>,

    /// Supports parallel function calls
    #[serde(default)]
    pub parallel_tool_calls: bool,
}

fn default_true() -> bool {
    true
}

impl ProviderCapabilities {
    /// Create capabilities for a basic chat provider
    #[must_use]
    pub fn basic_chat() -> Self {
        Self {
            chat: true,
            streaming: true,
            ..Default::default()
        }
    }

    /// Create capabilities for a full-featured provider
    #[must_use]
    pub fn full_featured() -> Self {
        Self {
            chat: true,
            streaming: true,
            function_calling: true,
            vision: true,
            embeddings: true,
            json_mode: true,
            seed: true,
            logprobs: true,
            max_context_length: Some(128_000),
            max_output_tokens: Some(4096),
            parallel_tool_calls: true,
        }
    }

    /// Check if a specific capability is supported
    #[must_use]
    pub fn supports(&self, capability: &str) -> bool {
        match capability {
            "chat" => self.chat,
            "streaming" => self.streaming,
            "function_calling" | "tools" => self.function_calling,
            "vision" => self.vision,
            "embeddings" => self.embeddings,
            "json_mode" => self.json_mode,
            "seed" => self.seed,
            "logprobs" => self.logprobs,
            "parallel_tool_calls" => self.parallel_tool_calls,
            _ => false,
        }
    }
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier
    pub id: String,

    /// Human-readable name
    #[serde(default)]
    pub name: Option<String>,

    /// Model description
    #[serde(default)]
    pub description: Option<String>,

    /// Alternative names/aliases for this model
    #[serde(default)]
    pub aliases: Vec<String>,

    /// Maximum context window size
    #[serde(default)]
    pub context_length: Option<u32>,

    /// Maximum output tokens
    #[serde(default)]
    pub max_output_tokens: Option<u32>,

    /// Cost per 1K input tokens (USD)
    #[serde(default)]
    pub input_cost_per_1k: Option<f64>,

    /// Cost per 1K output tokens (USD)
    #[serde(default)]
    pub output_cost_per_1k: Option<f64>,

    /// Model capabilities (inherits from provider if not specified)
    #[serde(default)]
    pub capabilities: Option<ProviderCapabilities>,

    /// Whether this model is deprecated
    #[serde(default)]
    pub deprecated: bool,
}

impl ModelInfo {
    /// Create a new model info
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            description: None,
            aliases: Vec::new(),
            context_length: None,
            max_output_tokens: None,
            input_cost_per_1k: None,
            output_cost_per_1k: None,
            capabilities: None,
            deprecated: false,
        }
    }

    /// Set the model name
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the context length
    #[must_use]
    pub fn with_context_length(mut self, length: u32) -> Self {
        self.context_length = Some(length);
        self
    }

    /// Set the max output tokens
    #[must_use]
    pub fn with_max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Add an alias
    #[must_use]
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }

    /// Set pricing
    #[must_use]
    pub fn with_pricing(mut self, input_cost: f64, output_cost: f64) -> Self {
        self.input_cost_per_1k = Some(input_cost);
        self.output_cost_per_1k = Some(output_cost);
        self
    }

    /// Check if this model matches the given identifier
    #[must_use]
    pub fn matches(&self, model_id: &str) -> bool {
        self.id == model_id || self.aliases.iter().any(|a| a == model_id)
    }
}

/// Provider statistics for monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderStats {
    /// Total requests made
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Total tokens used (prompt + completion)
    pub total_tokens: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// P95 latency in milliseconds
    pub p95_latency_ms: f64,
    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,
    /// Current requests per second
    pub requests_per_second: f64,
    /// Current health status
    pub health_status: HealthStatus,
    /// Last health check timestamp
    pub last_health_check: Option<i64>,
}

impl ProviderStats {
    /// Calculate success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            1.0
        } else {
            self.successful_requests as f64 / self.total_requests as f64
        }
    }

    /// Calculate error rate
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        1.0 - self.success_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type_display() {
        assert_eq!(ProviderType::OpenAI.to_string(), "openai");
        assert_eq!(ProviderType::Anthropic.to_string(), "anthropic");
        assert_eq!(ProviderType::Google.to_string(), "google");
    }

    #[test]
    fn test_provider_type_parse() {
        assert_eq!("openai".parse::<ProviderType>().unwrap(), ProviderType::OpenAI);
        assert_eq!("OPENAI".parse::<ProviderType>().unwrap(), ProviderType::OpenAI);
        assert_eq!("azure_openai".parse::<ProviderType>().unwrap(), ProviderType::Azure);
        assert_eq!("aws-bedrock".parse::<ProviderType>().unwrap(), ProviderType::Bedrock);
        assert!("invalid".parse::<ProviderType>().is_err());
    }

    #[test]
    fn test_health_status() {
        assert!(HealthStatus::Healthy.should_route());
        assert!(HealthStatus::Degraded.should_route());
        assert!(!HealthStatus::Unhealthy.should_route());
        assert!(!HealthStatus::Unknown.should_route());

        assert!(!HealthStatus::Healthy.is_problematic());
        assert!(HealthStatus::Unhealthy.is_problematic());
    }

    #[test]
    fn test_capabilities() {
        let basic = ProviderCapabilities::basic_chat();
        assert!(basic.chat);
        assert!(basic.streaming);
        assert!(!basic.vision);

        let full = ProviderCapabilities::full_featured();
        assert!(full.chat);
        assert!(full.vision);
        assert!(full.function_calling);

        assert!(full.supports("chat"));
        assert!(full.supports("vision"));
        assert!(full.supports("tools"));
        assert!(!full.supports("unknown"));
    }

    #[test]
    fn test_model_info() {
        let model = ModelInfo::new("gpt-4")
            .with_name("GPT-4")
            .with_context_length(128_000)
            .with_alias("gpt-4-0613")
            .with_pricing(0.03, 0.06);

        assert_eq!(model.id, "gpt-4");
        assert_eq!(model.name, Some("GPT-4".to_string()));
        assert!(model.matches("gpt-4"));
        assert!(model.matches("gpt-4-0613"));
        assert!(!model.matches("gpt-3.5"));
    }

    #[test]
    fn test_provider_stats() {
        let stats = ProviderStats {
            total_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            ..Default::default()
        };

        assert!((stats.success_rate() - 0.95).abs() < 0.001);
        assert!((stats.error_rate() - 0.05).abs() < 0.001);
    }
}
