//! OpenAI provider implementation.
//!
//! Supports OpenAI API including GPT-4, GPT-3.5-turbo, and other models.

use async_stream::try_stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures_util::StreamExt;
use gateway_core::{
    ChatChunk, ChatMessage, Choice, ChunkChoice, ChunkDelta, FinishReason, FunctionCall,
    GatewayError, GatewayRequest, GatewayResponse, HealthStatus, LLMProvider, MessageContent,
    MessageRole, ModelInfo, ProviderCapabilities, ProviderType, ToolCall, Usage,
};
use gateway_core::response::ResponseMessage;
use reqwest::Client;
use reqwest_eventsource::{Event, EventSource};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, trace, warn};

/// OpenAI provider configuration
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// Provider instance ID
    pub id: String,
    /// API key
    pub api_key: SecretString,
    /// Base URL (default: https://api.openai.com)
    pub base_url: String,
    /// Organization ID (optional)
    pub organization_id: Option<String>,
    /// Request timeout
    pub timeout: Duration,
    /// Supported models
    pub models: Vec<ModelInfo>,
}

impl OpenAIConfig {
    /// Create a new OpenAI configuration
    #[must_use]
    pub fn new(id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            api_key: SecretString::new(api_key.into()),
            base_url: "https://api.openai.com".to_string(),
            organization_id: None,
            timeout: Duration::from_secs(120),
            models: Self::default_models(),
        }
    }

    /// Set the base URL
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the organization ID
    #[must_use]
    pub fn with_organization(mut self, org_id: impl Into<String>) -> Self {
        self.organization_id = Some(org_id.into());
        self
    }

    /// Set the timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set custom models
    #[must_use]
    pub fn with_models(mut self, models: Vec<ModelInfo>) -> Self {
        self.models = models;
        self
    }

    /// Default OpenAI models
    #[must_use]
    pub fn default_models() -> Vec<ModelInfo> {
        vec![
            ModelInfo::new("gpt-4o")
                .with_name("GPT-4o")
                .with_context_length(128_000)
                .with_max_output_tokens(16_384)
                .with_pricing(0.005, 0.015),
            ModelInfo::new("gpt-4o-mini")
                .with_name("GPT-4o Mini")
                .with_context_length(128_000)
                .with_max_output_tokens(16_384)
                .with_pricing(0.00015, 0.0006),
            ModelInfo::new("gpt-4-turbo")
                .with_name("GPT-4 Turbo")
                .with_context_length(128_000)
                .with_max_output_tokens(4_096)
                .with_pricing(0.01, 0.03),
            ModelInfo::new("gpt-4")
                .with_name("GPT-4")
                .with_context_length(8_192)
                .with_max_output_tokens(8_192)
                .with_pricing(0.03, 0.06),
            ModelInfo::new("gpt-3.5-turbo")
                .with_name("GPT-3.5 Turbo")
                .with_context_length(16_385)
                .with_max_output_tokens(4_096)
                .with_pricing(0.0005, 0.0015),
        ]
    }
}

/// OpenAI provider implementation
pub struct OpenAIProvider {
    config: OpenAIConfig,
    client: Client,
    capabilities: ProviderCapabilities,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider
    ///
    /// # Errors
    /// Returns error if HTTP client cannot be created
    pub fn new(config: OpenAIConfig) -> Result<Self, GatewayError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .pool_max_idle_per_host(100)
            .build()
            .map_err(|e| GatewayError::internal(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            config,
            client,
            capabilities: ProviderCapabilities {
                chat: true,
                streaming: true,
                function_calling: true,
                vision: true,
                embeddings: true,
                json_mode: true,
                seed: true,
                logprobs: true,
                max_context_length: Some(128_000),
                max_output_tokens: Some(16_384),
                parallel_tool_calls: true,
            },
        })
    }

    /// Get the chat completions endpoint URL
    fn completions_url(&self) -> String {
        format!("{}/v1/chat/completions", self.config.base_url)
    }

    /// Transform gateway request to OpenAI format
    fn transform_request(&self, request: &GatewayRequest) -> OpenAIRequest {
        let messages: Vec<OpenAIMessage> = request
            .messages
            .iter()
            .map(OpenAIMessage::from_gateway_message)
            .collect();

        OpenAIRequest {
            model: request.model.clone(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stop: request.stop.clone(),
            stream: Some(request.stream),
            n: request.n,
            seed: request.seed,
            user: request.user.clone(),
            tools: request.tools.as_ref().map(|tools| {
                tools
                    .iter()
                    .map(|t| OpenAITool {
                        tool_type: t.tool_type.clone(),
                        function: OpenAIFunction {
                            name: t.function.name.clone(),
                            description: t.function.description.clone(),
                            parameters: t.function.parameters.clone(),
                        },
                    })
                    .collect()
            }),
            tool_choice: request.tool_choice.as_ref().map(|tc| {
                match tc {
                    gateway_core::request::ToolChoice::String(s) => {
                        serde_json::Value::String(s.clone())
                    }
                    gateway_core::request::ToolChoice::Tool { tool_type, function } => {
                        serde_json::json!({
                            "type": tool_type,
                            "function": { "name": function.name }
                        })
                    }
                }
            }),
            response_format: request.response_format.as_ref().map(|rf| OpenAIResponseFormat {
                format_type: rf.format_type.clone(),
            }),
        }
    }

    /// Transform OpenAI response to gateway format
    fn transform_response(&self, response: OpenAIResponse) -> GatewayResponse {
        let choices: Vec<Choice> = response
            .choices
            .into_iter()
            .map(|c| Choice {
                index: c.index,
                message: ResponseMessage {
                    role: MessageRole::Assistant,
                    content: c.message.content,
                    tool_calls: c.message.tool_calls.map(|tcs| {
                        tcs.into_iter()
                            .map(|tc| ToolCall {
                                id: tc.id,
                                tool_type: tc.tool_type,
                                function: FunctionCall {
                                    name: tc.function.name,
                                    arguments: tc.function.arguments,
                                },
                            })
                            .collect()
                    }),
                    function_call: None,
                },
                finish_reason: c.finish_reason.map(|r| match r.as_str() {
                    "length" => FinishReason::Length,
                    "tool_calls" => FinishReason::ToolCalls,
                    "content_filter" => FinishReason::ContentFilter,
                    _ => FinishReason::Stop,
                }),
                logprobs: None,
            })
            .collect();

        GatewayResponse {
            id: response.id,
            object: response.object,
            created: response.created,
            model: response.model,
            choices,
            usage: response.usage.map_or_else(Usage::default, |u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
            system_fingerprint: response.system_fingerprint,
            provider: Some(self.config.id.clone()),
        }
    }

}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    fn id(&self) -> &str {
        &self.config.id
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }

    async fn chat_completion(
        &self,
        request: &GatewayRequest,
    ) -> Result<GatewayResponse, GatewayError> {
        let openai_request = self.transform_request(request);

        debug!(
            provider = %self.config.id,
            model = %request.model,
            "Sending chat completion request to OpenAI"
        );

        let mut req_builder = self
            .client
            .post(self.completions_url())
            .header("Authorization", format!("Bearer {}", self.config.api_key.expose_secret()))
            .header("Content-Type", "application/json");

        if let Some(ref org_id) = self.config.organization_id {
            req_builder = req_builder.header("OpenAI-Organization", org_id);
        }

        let response = req_builder
            .json(&openai_request)
            .send()
            .await
            .map_err(|e| {
                GatewayError::provider(
                    &self.config.id,
                    format!("Request failed: {e}"),
                    None,
                    e.is_timeout() || e.is_connect(),
                )
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            let retryable = status.as_u16() >= 500 || status.as_u16() == 429;

            error!(
                provider = %self.config.id,
                status = %status,
                error = %error_body,
                "OpenAI API error"
            );

            return Err(GatewayError::provider(
                &self.config.id,
                error_body,
                Some(status.as_u16()),
                retryable,
            ));
        }

        let openai_response: OpenAIResponse = response.json().await.map_err(|e| {
            GatewayError::provider(
                &self.config.id,
                format!("Failed to parse response: {e}"),
                None,
                false,
            )
        })?;

        Ok(self.transform_response(openai_response))
    }

    async fn chat_completion_stream(
        &self,
        request: &GatewayRequest,
    ) -> Result<BoxStream<'static, Result<ChatChunk, GatewayError>>, GatewayError> {
        let mut openai_request = self.transform_request(request);
        openai_request.stream = Some(true);

        debug!(
            provider = %self.config.id,
            model = %request.model,
            "Starting streaming chat completion to OpenAI"
        );

        let mut req_builder = self
            .client
            .post(self.completions_url())
            .header("Authorization", format!("Bearer {}", self.config.api_key.expose_secret()))
            .header("Content-Type", "application/json");

        if let Some(ref org_id) = self.config.organization_id {
            req_builder = req_builder.header("OpenAI-Organization", org_id);
        }

        let req_builder = req_builder.json(&openai_request);

        let provider_id = self.config.id.clone();

        let stream = try_stream! {
            let es = EventSource::new(req_builder).map_err(|e| {
                GatewayError::streaming(format!("Failed to create event source: {e}"))
            })?;

            let mut es = Box::pin(es);

            while let Some(event) = es.next().await {
                match event {
                    Ok(Event::Open) => {
                        trace!(provider = %provider_id, "SSE connection opened");
                    }
                    Ok(Event::Message(message)) => {
                        let data = message.data.trim();
                        if data == "[DONE]" {
                            trace!(provider = %provider_id, "SSE stream done");
                            break;
                        }

                        match serde_json::from_str::<OpenAIChunk>(data) {
                            Ok(chunk) => {
                                let choices: Vec<ChunkChoice> = chunk
                                    .choices
                                    .into_iter()
                                    .map(|c| ChunkChoice {
                                        index: c.index,
                                        delta: ChunkDelta {
                                            role: c.delta.role.map(|_| MessageRole::Assistant),
                                            content: c.delta.content,
                                            tool_calls: None,
                                            function_call: None,
                                        },
                                        finish_reason: c.finish_reason.map(|r| match r.as_str() {
                                            "length" => FinishReason::Length,
                                            "tool_calls" => FinishReason::ToolCalls,
                                            _ => FinishReason::Stop,
                                        }),
                                        logprobs: None,
                                    })
                                    .collect();

                                yield ChatChunk {
                                    id: chunk.id,
                                    object: chunk.object,
                                    created: chunk.created,
                                    model: chunk.model,
                                    choices,
                                    usage: None,
                                    system_fingerprint: chunk.system_fingerprint,
                                };
                            }
                            Err(e) => {
                                warn!(provider = %provider_id, error = %e, "Failed to parse chunk");
                            }
                        }
                    }
                    Err(e) => {
                        error!(provider = %provider_id, error = %e, "SSE error");
                        Err(GatewayError::streaming(format!("SSE error: {e}")))?;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    async fn health_check(&self) -> HealthStatus {
        // Use models endpoint for health check
        let url = format!("{}/v1/models", self.config.base_url);

        match self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key.expose_secret()))
            .timeout(Duration::from_secs(10))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => HealthStatus::Healthy,
            Ok(response) if response.status().as_u16() == 429 => HealthStatus::Degraded,
            Ok(_) | Err(_) => HealthStatus::Unhealthy,
        }
    }

    fn capabilities(&self) -> &ProviderCapabilities {
        &self.capabilities
    }

    fn models(&self) -> &[ModelInfo] {
        &self.config.models
    }

    fn base_url(&self) -> &str {
        &self.config.base_url
    }

    fn timeout(&self) -> Duration {
        self.config.timeout
    }
}

// OpenAI API types

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OpenAIResponseFormat>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl OpenAIMessage {
    fn from_gateway_message(msg: &ChatMessage) -> Self {
        let role = match msg.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };

        let content = match &msg.content {
            MessageContent::Text(s) => Some(serde_json::Value::String(s.clone())),
            MessageContent::Parts(parts) => {
                let json_parts: Vec<serde_json::Value> = parts
                    .iter()
                    .map(|p| match p {
                        gateway_core::request::ContentPart::Text { text } => {
                            serde_json::json!({"type": "text", "text": text})
                        }
                        gateway_core::request::ContentPart::ImageUrl { image_url } => {
                            serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url.url,
                                    "detail": image_url.detail.map(|d| match d {
                                        gateway_core::request::ImageDetail::Auto => "auto",
                                        gateway_core::request::ImageDetail::Low => "low",
                                        gateway_core::request::ImageDetail::High => "high",
                                    })
                                }
                            })
                        }
                    })
                    .collect();
                Some(serde_json::Value::Array(json_parts))
            }
        };

        Self {
            role: role.to_string(),
            content,
            name: msg.name.clone(),
            tool_calls: msg.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| OpenAIToolCall {
                        id: tc.id.clone(),
                        tool_type: tc.tool_type.clone(),
                        function: OpenAIFunctionCall {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    })
                    .collect()
            }),
            tool_call_id: msg.tool_call_id.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize)]
struct OpenAIFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAIResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
    system_fingerprint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    index: u32,
    message: OpenAIResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponseMessage {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAIChunk {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<OpenAIChunkChoice>,
    #[allow(dead_code)]
    usage: Option<OpenAIUsage>,
    system_fingerprint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChunkChoice {
    index: u32,
    delta: OpenAIChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChunkDelta {
    role: Option<String>,
    content: Option<String>,
    #[allow(dead_code)]
    tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct OpenAIToolCallDelta {
    index: u32,
    id: Option<String>,
    #[serde(rename = "type")]
    tool_type: Option<String>,
    function: Option<OpenAIFunctionCallDelta>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct OpenAIFunctionCallDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = OpenAIConfig::new("openai-1", "sk-test-key")
            .with_base_url("https://custom.openai.com")
            .with_organization("org-123")
            .with_timeout(Duration::from_secs(60));

        assert_eq!(config.id, "openai-1");
        assert_eq!(config.base_url, "https://custom.openai.com");
        assert_eq!(config.organization_id, Some("org-123".to_string()));
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_default_models() {
        let models = OpenAIConfig::default_models();
        assert!(!models.is_empty());

        let gpt4o = models.iter().find(|m| m.id == "gpt-4o");
        assert!(gpt4o.is_some());
    }

    #[test]
    fn test_provider_creation() {
        let config = OpenAIConfig::new("test", "sk-test");
        let provider = OpenAIProvider::new(config).expect("create provider");

        assert_eq!(provider.id(), "test");
        assert_eq!(provider.provider_type(), ProviderType::OpenAI);
        assert!(provider.capabilities().chat);
        assert!(provider.capabilities().streaming);
    }

    #[test]
    fn test_completions_url() {
        let config = OpenAIConfig::new("test", "sk-test")
            .with_base_url("https://api.openai.com");
        let provider = OpenAIProvider::new(config).expect("create provider");

        assert_eq!(
            provider.completions_url(),
            "https://api.openai.com/v1/chat/completions"
        );
    }
}
