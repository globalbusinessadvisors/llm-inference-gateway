//! Anthropic Claude provider implementation.
//!
//! Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku models.

use async_stream::try_stream;
use futures::stream::BoxStream;
use futures_util::StreamExt;
use gateway_core::{
    ChatChunk, FinishReason, GatewayError, GatewayRequest, GatewayResponse,
    HealthStatus, LLMProvider, MessageContent, MessageRole, ModelInfo, ProviderCapabilities,
    ProviderType, Usage,
};
use reqwest::Client;
use reqwest_eventsource::{Event, EventSource};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, error, instrument, warn};

/// Anthropic API version header value
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Default Anthropic API base URL
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Anthropic provider configuration
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// API key for authentication
    pub api_key: SecretString,
    /// Base URL for the API
    pub base_url: String,
    /// Request timeout
    pub timeout: Duration,
    /// Available models
    pub models: Vec<ModelInfo>,
}

impl AnthropicConfig {
    /// Create a new configuration with the given API key
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: SecretString::new(api_key.into()),
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout: Duration::from_secs(120),
            models: default_anthropic_models(),
        }
    }

    /// Set the base URL
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the request timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set available models
    #[must_use]
    pub fn with_models(mut self, models: Vec<ModelInfo>) -> Self {
        self.models = models;
        self
    }
}

/// Get default Anthropic models
fn default_anthropic_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo::new("claude-3-5-sonnet-20241022")
            .with_name("Claude 3.5 Sonnet")
            .with_context_length(200_000)
            .with_max_output_tokens(8192)
            .with_alias("claude-3-5-sonnet")
            .with_alias("claude-3.5-sonnet"),
        ModelInfo::new("claude-3-5-haiku-20241022")
            .with_name("Claude 3.5 Haiku")
            .with_context_length(200_000)
            .with_max_output_tokens(8192)
            .with_alias("claude-3-5-haiku")
            .with_alias("claude-3.5-haiku"),
        ModelInfo::new("claude-3-opus-20240229")
            .with_name("Claude 3 Opus")
            .with_context_length(200_000)
            .with_max_output_tokens(4096)
            .with_alias("claude-3-opus"),
        ModelInfo::new("claude-3-sonnet-20240229")
            .with_name("Claude 3 Sonnet")
            .with_context_length(200_000)
            .with_max_output_tokens(4096)
            .with_alias("claude-3-sonnet"),
        ModelInfo::new("claude-3-haiku-20240307")
            .with_name("Claude 3 Haiku")
            .with_context_length(200_000)
            .with_max_output_tokens(4096)
            .with_alias("claude-3-haiku"),
    ]
}

/// Anthropic LLM provider
pub struct AnthropicProvider {
    /// Provider ID
    id: String,
    /// HTTP client
    client: Client,
    /// Configuration
    config: AnthropicConfig,
    /// Provider capabilities
    capabilities: ProviderCapabilities,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider
    ///
    /// # Errors
    /// Returns error if HTTP client cannot be built
    pub fn new(config: AnthropicConfig) -> Result<Self, GatewayError> {
        Self::with_id("anthropic", config)
    }

    /// Create a new provider with a custom ID
    ///
    /// # Errors
    /// Returns error if HTTP client cannot be built
    pub fn with_id(id: impl Into<String>, config: AnthropicConfig) -> Result<Self, GatewayError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| GatewayError::Configuration {
                message: format!("Failed to build HTTP client: {e}"),
            })?;

        Ok(Self {
            id: id.into(),
            client,
            config,
            capabilities: ProviderCapabilities {
                chat: true,
                streaming: true,
                function_calling: true,
                vision: true,
                embeddings: false,
                json_mode: false,
                seed: false,
                logprobs: false,
                max_context_length: Some(200_000),
                max_output_tokens: Some(8192),
                parallel_tool_calls: true,
            },
        })
    }

    /// Get the API URL for a given endpoint
    fn api_url(&self, endpoint: &str) -> String {
        format!("{}/v1{}", self.config.base_url.trim_end_matches('/'), endpoint)
    }

    /// Build request headers
    ///
    /// # Panics
    /// Panics if header values cannot be parsed (should not happen with static strings)
    #[allow(clippy::expect_used)]
    fn build_headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "x-api-key",
            self.config
                .api_key
                .expose_secret()
                .parse()
                .expect("valid header value"),
        );
        headers.insert(
            "anthropic-version",
            ANTHROPIC_VERSION.parse().expect("valid header value"),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().expect("valid header value"),
        );
        headers
    }
}

#[async_trait::async_trait]
impl LLMProvider for AnthropicProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }

    #[instrument(skip(self, request), fields(provider = %self.id, model = %request.model))]
    async fn chat_completion(&self, request: &GatewayRequest) -> Result<GatewayResponse, GatewayError> {
        let anthropic_request = transform_request(request)?;
        let url = self.api_url("/messages");

        debug!(url = %url, "Sending chat completion request to Anthropic");

        let response = self
            .client
            .post(&url)
            .headers(self.build_headers())
            .json(&anthropic_request)
            .send()
            .await
            .map_err(|e| {
                let retryable = e.is_timeout() || e.is_connect();
                GatewayError::Provider {
                    provider: self.id.clone(),
                    message: format!("Request failed: {e}"),
                    status_code: None,
                    retryable,
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(parse_error_response(status, &error_body, &self.id));
        }

        let anthropic_response: AnthropicResponse = response.json().await.map_err(|e| {
            GatewayError::Provider {
                provider: self.id.clone(),
                message: format!("Failed to parse response: {e}"),
                status_code: None,
                retryable: false,
            }
        })?;

        transform_response(anthropic_response, &request.model)
    }

    #[instrument(skip(self, request), fields(provider = %self.id, model = %request.model))]
    async fn chat_completion_stream(
        &self,
        request: &GatewayRequest,
    ) -> Result<BoxStream<'static, Result<ChatChunk, GatewayError>>, GatewayError> {
        let mut anthropic_request = transform_request(request)?;
        anthropic_request.stream = Some(true);

        let url = self.api_url("/messages");
        let provider_id = self.id.clone();
        let model = request.model.clone();

        debug!(url = %url, "Starting streaming chat completion to Anthropic");

        let request_builder = self
            .client
            .post(&url)
            .headers(self.build_headers())
            .json(&anthropic_request);

        let event_source = EventSource::new(request_builder).map_err(|e| GatewayError::Provider {
            provider: provider_id.clone(),
            message: format!("Failed to create event source: {e}"),
            status_code: None,
            retryable: false,
        })?;

        let stream = try_stream! {
            let mut es = event_source;
            let mut input_tokens = 0u32;
            let mut output_tokens = 0u32;
            let mut current_index = 0usize;

            while let Some(event) = es.next().await {
                match event {
                    Ok(Event::Open) => {
                        debug!("SSE connection opened");
                    }
                    Ok(Event::Message(msg)) => {
                        // Skip ping events
                        if msg.event == "ping" {
                            continue;
                        }

                        // Handle message_start (contains input tokens)
                        if msg.event == "message_start" {
                            if let Ok(start) = serde_json::from_str::<MessageStartEvent>(&msg.data) {
                                if let Some(usage) = start.message.usage {
                                    input_tokens = usage.input_tokens.unwrap_or(0);
                                }
                            }
                            continue;
                        }

                        // Handle message_delta (contains output tokens and stop reason)
                        if msg.event == "message_delta" {
                            if let Ok(delta) = serde_json::from_str::<MessageDeltaEvent>(&msg.data) {
                                if let Some(usage) = delta.usage {
                                    output_tokens = usage.output_tokens.unwrap_or(0);
                                }
                                if let Some(stop_reason) = delta.delta.stop_reason {
                                    let chunk = ChatChunk::builder()
                                        .id(format!("chunk-{}", uuid::Uuid::new_v4()))
                                        .model(model.clone())
                                        .choice(gateway_core::ChunkChoice::with_finish(
                                            current_index as u32,
                                            map_stop_reason(&stop_reason),
                                        ))
                                        .usage(Usage::new(input_tokens, output_tokens))
                                        .build();
                                    yield chunk;
                                }
                            }
                            continue;
                        }

                        // Handle content_block_start
                        if msg.event == "content_block_start" {
                            if let Ok(block) = serde_json::from_str::<ContentBlockStartEvent>(&msg.data) {
                                current_index = block.index;
                            }
                            continue;
                        }

                        // Handle content_block_delta (the actual content)
                        if msg.event == "content_block_delta" {
                            if let Ok(delta) = serde_json::from_str::<ContentBlockDeltaEvent>(&msg.data) {
                                if let Some(text) = delta.delta.text {
                                    let chunk = ChatChunk::builder()
                                        .id(format!("chunk-{}", uuid::Uuid::new_v4()))
                                        .model(model.clone())
                                        .choice(gateway_core::ChunkChoice::with_content(
                                            current_index as u32,
                                            text,
                                        ))
                                        .build();
                                    yield chunk;
                                }
                            }
                            continue;
                        }

                        // Handle message_stop
                        if msg.event == "message_stop" {
                            debug!("Received message_stop event");
                            break;
                        }

                        // Handle error events
                        if msg.event == "error" {
                            if let Ok(err) = serde_json::from_str::<StreamErrorEvent>(&msg.data) {
                                Err(GatewayError::Provider {
                                    provider: provider_id.clone(),
                                    message: err.error.message,
                                    status_code: None,
                                    retryable: false,
                                })?;
                            }
                        }
                    }
                    Err(e) => {
                        // Check if it's a normal close
                        if matches!(e, reqwest_eventsource::Error::StreamEnded) {
                            break;
                        }
                        error!(error = %e, "SSE error");
                        Err(GatewayError::Provider {
                            provider: provider_id.clone(),
                            message: format!("Stream error: {e}"),
                            status_code: None,
                            retryable: false,
                        })?;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    async fn health_check(&self) -> HealthStatus {
        // Anthropic doesn't have a dedicated health endpoint, so we make a minimal request
        // We use the messages endpoint with minimal tokens to check connectivity
        let test_request = AnthropicRequest {
            model: "claude-3-haiku-20240307".to_string(),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicContent::Text("Hi".to_string()),
            }],
            max_tokens: 1,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            tool_choice: None,
        };

        let url = self.api_url("/messages");

        match self
            .client
            .post(&url)
            .headers(self.build_headers())
            .json(&test_request)
            .timeout(Duration::from_secs(10))
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    HealthStatus::Healthy
                } else if response.status().is_server_error() {
                    HealthStatus::Unhealthy
                } else {
                    // Client errors (like auth) mean the service is up
                    HealthStatus::Healthy
                }
            }
            Err(e) => {
                warn!(error = %e, "Anthropic health check failed");
                if e.is_timeout() {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Unhealthy
                }
            }
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
}

// ============================================================================
// Anthropic API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[allow(dead_code)]
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize)]
struct ImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: Option<String>,
    input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum AnthropicToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    #[serde(rename = "type")]
    _type: String,
    #[allow(dead_code)]
    role: String,
    content: Vec<AnthropicResponseContent>,
    #[allow(dead_code)]
    model: String,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicResponseContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    error_type: String,
    error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    error_type: String,
    message: String,
}

// Streaming event types
#[derive(Debug, Deserialize)]
struct MessageStartEvent {
    message: MessageStartMessage,
}

#[derive(Debug, Deserialize)]
struct MessageStartMessage {
    usage: Option<StreamUsage>,
}

#[derive(Debug, Deserialize)]
struct StreamUsage {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaEvent {
    delta: MessageDelta,
    usage: Option<StreamUsage>,
}

#[derive(Debug, Deserialize)]
struct MessageDelta {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStartEvent {
    index: usize,
}

#[derive(Debug, Deserialize)]
struct ContentBlockDeltaEvent {
    delta: ContentDelta,
}

#[derive(Debug, Deserialize)]
struct ContentDelta {
    #[serde(rename = "type")]
    _type: Option<String>,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamErrorEvent {
    error: StreamError,
}

#[derive(Debug, Deserialize)]
struct StreamError {
    #[serde(rename = "type")]
    _type: String,
    message: String,
}

// ============================================================================
// Transform Functions
// ============================================================================

/// Transform a gateway request to an Anthropic request
fn transform_request(request: &GatewayRequest) -> Result<AnthropicRequest, GatewayError> {
    let mut system_message = None;
    let mut messages = Vec::new();

    for msg in &request.messages {
        match msg.role {
            MessageRole::System => {
                // Anthropic uses a separate system parameter
                system_message = Some(extract_text_content(&msg.content));
            }
            MessageRole::User => {
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: transform_content(&msg.content),
                });
            }
            MessageRole::Assistant => {
                messages.push(AnthropicMessage {
                    role: "assistant".to_string(),
                    content: transform_content(&msg.content),
                });
            }
            MessageRole::Tool => {
                // Tool results need special handling
                if let Some(tool_call_id) = &msg.tool_call_id {
                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Blocks(vec![AnthropicContentBlock::ToolResult {
                            tool_use_id: tool_call_id.clone(),
                            content: extract_text_content(&msg.content),
                        }]),
                    });
                }
            }
        }
    }

    // Transform tools
    let tools = request.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.function.name.clone(),
                description: t.function.description.clone(),
                input_schema: t.function.parameters.clone().unwrap_or(serde_json::json!({})),
            })
            .collect()
    });

    // Transform tool choice
    let tool_choice = request.tool_choice.as_ref().and_then(|tc| match tc {
        gateway_core::ToolChoice::String(s) => match s.as_str() {
            "auto" => Some(AnthropicToolChoice::Auto),
            "none" => None, // Anthropic doesn't have "none"
            "required" => Some(AnthropicToolChoice::Any),
            _ => None,
        },
        gateway_core::ToolChoice::Tool { function, .. } => {
            Some(AnthropicToolChoice::Tool { name: function.name.clone() })
        },
    });

    Ok(AnthropicRequest {
        model: request.model.clone(),
        messages,
        max_tokens: request.max_tokens.unwrap_or(4096),
        system: system_message,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        stop_sequences: request.stop.clone(),
        stream: None,
        tools,
        tool_choice,
    })
}

/// Transform message content for Anthropic
fn transform_content(content: &MessageContent) -> AnthropicContent {
    match content {
        MessageContent::Text(text) => AnthropicContent::Text(text.clone()),
        MessageContent::Parts(parts) => {
            let blocks: Vec<AnthropicContentBlock> = parts
                .iter()
                .filter_map(|part| match part {
                    gateway_core::ContentPart::Text { text } => {
                        Some(AnthropicContentBlock::Text { text: text.clone() })
                    }
                    gateway_core::ContentPart::ImageUrl { image_url } => {
                        // Try to extract base64 data from data URLs
                        if let Some((media_type, data)) = parse_data_url(&image_url.url) {
                            Some(AnthropicContentBlock::Image {
                                source: ImageSource {
                                    source_type: "base64".to_string(),
                                    media_type,
                                    data,
                                },
                            })
                        } else {
                            // URL-based images not directly supported, skip
                            None
                        }
                    }
                })
                .collect();
            AnthropicContent::Blocks(blocks)
        }
    }
}

/// Extract text from message content
fn extract_text_content(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                gateway_core::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Parse a data URL into media type and base64 data
fn parse_data_url(url: &str) -> Option<(String, String)> {
    if let Some(stripped) = url.strip_prefix("data:") {
        let parts: Vec<&str> = stripped.splitn(2, ',').collect();
        if parts.len() == 2 {
            let meta = parts[0];
            let data = parts[1].to_string();
            let media_type = meta.split(';').next().unwrap_or("image/png").to_string();
            return Some((media_type, data));
        }
    }
    None
}

/// Transform Anthropic response to gateway response
fn transform_response(
    response: AnthropicResponse,
    model: &str,
) -> Result<GatewayResponse, GatewayError> {
    let mut content = String::new();
    let mut tool_calls = Vec::new();

    for block in response.content {
        match block {
            AnthropicResponseContent::Text { text } => {
                content.push_str(&text);
            }
            AnthropicResponseContent::ToolUse { id, name, input } => {
                tool_calls.push(gateway_core::ToolCall {
                    id,
                    tool_type: "function".to_string(),
                    function: gateway_core::FunctionCall {
                        name,
                        arguments: serde_json::to_string(&input).unwrap_or_default(),
                    },
                });
            }
        }
    }

    let finish_reason = response
        .stop_reason
        .as_deref()
        .map_or(FinishReason::Stop, map_stop_reason);

    // Build choice - with tool calls or content
    let choice = if tool_calls.is_empty() {
        gateway_core::Choice::new(0, content, finish_reason)
    } else {
        gateway_core::Choice::with_tool_calls(0, tool_calls, finish_reason)
    };

    let response = GatewayResponse::builder()
        .id(response.id)
        .model(model.to_string())
        .choice(choice)
        .usage(Usage::new(
            response.usage.input_tokens,
            response.usage.output_tokens,
        ))
        .build();

    Ok(response)
}

/// Map Anthropic stop reason to gateway finish reason
fn map_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" => FinishReason::Stop,
        "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        _ => FinishReason::Stop,
    }
}

/// Parse error response from Anthropic
fn parse_error_response(
    status: reqwest::StatusCode,
    body: &str,
    provider_id: &str,
) -> GatewayError {
    let message = if let Ok(err) = serde_json::from_str::<AnthropicError>(body) {
        err.error.message
    } else {
        format!("HTTP {status}: {body}")
    };

    match status.as_u16() {
        401 => GatewayError::Authentication { message },
        403 => GatewayError::Authorization { message },
        429 => GatewayError::RateLimit {
            retry_after: None,
            limit: None,
        },
        400 => GatewayError::Validation {
            message,
            field: None,
            code: "bad_request".to_string(),
        },
        500..=599 => GatewayError::Provider {
            provider: provider_id.to_string(),
            message,
            status_code: Some(status.as_u16()),
            retryable: true,
        },
        _ => GatewayError::Provider {
            provider: provider_id.to_string(),
            message,
            status_code: Some(status.as_u16()),
            retryable: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = AnthropicConfig::new("test-key")
            .with_base_url("https://custom.api")
            .with_timeout(Duration::from_secs(60));

        assert_eq!(config.base_url, "https://custom.api");
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_default_models() {
        let models = default_anthropic_models();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.id == "claude-3-5-sonnet-20241022"));
        assert!(models.iter().any(|m| m.id == "claude-3-opus-20240229"));
    }

    #[test]
    fn test_transform_request_basic() {
        let request = GatewayRequest::builder()
            .model("claude-3-sonnet-20240229")
            .message(gateway_core::ChatMessage::user("Hello"))
            .build()
            .unwrap();

        let anthropic_req = transform_request(&request).unwrap();
        assert_eq!(anthropic_req.model, "claude-3-sonnet-20240229");
        assert_eq!(anthropic_req.messages.len(), 1);
    }

    #[test]
    fn test_transform_request_with_system() {
        let request = GatewayRequest::builder()
            .model("claude-3-sonnet-20240229")
            .message(gateway_core::ChatMessage::system("You are helpful"))
            .message(gateway_core::ChatMessage::user("Hello"))
            .build()
            .unwrap();

        let anthropic_req = transform_request(&request).unwrap();
        assert_eq!(anthropic_req.system, Some("You are helpful".to_string()));
        assert_eq!(anthropic_req.messages.len(), 1); // System not in messages
    }

    #[test]
    fn test_map_stop_reason() {
        assert_eq!(map_stop_reason("end_turn"), FinishReason::Stop);
        assert_eq!(map_stop_reason("max_tokens"), FinishReason::Length);
        assert_eq!(map_stop_reason("tool_use"), FinishReason::ToolCalls);
    }

    #[test]
    fn test_parse_data_url() {
        let url = "data:image/png;base64,iVBORw0KGgoAAAANS";
        let result = parse_data_url(url);
        assert!(result.is_some());
        let (media_type, data) = result.unwrap();
        assert_eq!(media_type, "image/png");
        assert_eq!(data, "iVBORw0KGgoAAAANS");
    }
}
