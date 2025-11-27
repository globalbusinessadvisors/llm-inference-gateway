//! Response types for the gateway.
//!
//! This module defines the unified response format that is OpenAI-compatible.

use crate::request::{FunctionCall, MessageRole, ToolCall};
use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Unified gateway response (OpenAI-compatible format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayResponse {
    /// Unique response identifier
    pub id: String,

    /// Object type (always "chat.completion")
    pub object: String,

    /// Creation timestamp (Unix epoch seconds)
    pub created: i64,

    /// Model used for completion
    pub model: String,

    /// Completion choices
    pub choices: Vec<Choice>,

    /// Token usage statistics
    pub usage: Usage,

    /// System fingerprint for reproducibility
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,

    /// Provider that served this request (gateway extension)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

impl GatewayResponse {
    /// Create a new response builder
    #[must_use]
    pub fn builder() -> GatewayResponseBuilder {
        GatewayResponseBuilder::default()
    }

    /// Get the first choice's message content (convenience method)
    #[must_use]
    pub fn content(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.message.content.as_deref())
    }

    /// Get the first choice's finish reason
    #[must_use]
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.choices.first().and_then(|c| c.finish_reason)
    }

    /// Check if response has tool calls
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        self.choices
            .first()
            .is_some_and(|c| c.message.tool_calls.is_some())
    }

    /// Get tool calls from the first choice
    #[must_use]
    pub fn tool_calls(&self) -> Option<&[ToolCall]> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.as_deref())
    }
}

/// Builder for `GatewayResponse`
#[derive(Debug, Default)]
pub struct GatewayResponseBuilder {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<Choice>,
    usage: Option<Usage>,
    system_fingerprint: Option<String>,
    provider: Option<String>,
}

impl GatewayResponseBuilder {
    /// Set the response ID
    #[must_use]
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the model
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the choices
    #[must_use]
    pub fn choices(mut self, choices: Vec<Choice>) -> Self {
        self.choices = choices;
        self
    }

    /// Add a choice
    #[must_use]
    pub fn choice(mut self, choice: Choice) -> Self {
        self.choices.push(choice);
        self
    }

    /// Set the usage
    #[must_use]
    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set the system fingerprint
    #[must_use]
    pub fn system_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.system_fingerprint = Some(fingerprint.into());
        self
    }

    /// Set the provider
    #[must_use]
    pub fn provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Build the response
    #[must_use]
    pub fn build(self) -> GatewayResponse {
        GatewayResponse {
            id: self
                .id
                .unwrap_or_else(|| format!("chatcmpl-{}", uuid::Uuid::new_v4())),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: self.model.unwrap_or_default(),
            choices: self.choices,
            usage: self.usage.unwrap_or_default(),
            system_fingerprint: self.system_fingerprint,
            provider: self.provider,
        }
    }
}

/// Completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    /// Index of this choice
    pub index: u32,

    /// The generated message
    pub message: ResponseMessage,

    /// Reason for finishing
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Log probabilities (if requested)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
}

impl Choice {
    /// Create a new choice with content
    #[must_use]
    pub fn new(index: u32, content: impl Into<String>, finish_reason: FinishReason) -> Self {
        Self {
            index,
            message: ResponseMessage {
                role: MessageRole::Assistant,
                content: Some(content.into()),
                tool_calls: None,
                function_call: None,
            },
            finish_reason: Some(finish_reason),
            logprobs: None,
        }
    }

    /// Create a new choice with tool calls
    #[must_use]
    pub fn with_tool_calls(
        index: u32,
        tool_calls: Vec<ToolCall>,
        finish_reason: FinishReason,
    ) -> Self {
        Self {
            index,
            message: ResponseMessage {
                role: MessageRole::Assistant,
                content: None,
                tool_calls: Some(tool_calls),
                function_call: None,
            },
            finish_reason: Some(finish_reason),
            logprobs: None,
        }
    }
}

/// Response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    /// Role (always "assistant" for responses)
    pub role: MessageRole,

    /// Message content
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool calls made by the assistant
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Deprecated: Function call (use tool_calls instead)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

/// Reason for finishing generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural stop (end of message)
    Stop,
    /// Hit max_tokens limit
    Length,
    /// Model made tool calls
    ToolCalls,
    /// Content was filtered
    ContentFilter,
    /// Deprecated: Function call
    FunctionCall,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stop => write!(f, "stop"),
            Self::Length => write!(f, "length"),
            Self::ToolCalls => write!(f, "tool_calls"),
            Self::ContentFilter => write!(f, "content_filter"),
            Self::FunctionCall => write!(f, "function_call"),
        }
    }
}

/// Log probabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProbs {
    /// Content log probabilities
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TokenLogProb>>,
}

/// Token log probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogProb {
    /// The token
    pub token: String,
    /// Log probability
    pub logprob: f32,
    /// Bytes representation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    /// Top log probabilities
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<TopLogProb>>,
}

/// Top log probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogProb {
    /// The token
    pub token: String,
    /// Log probability
    pub logprob: f32,
    /// Bytes representation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// Token usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,

    /// Number of tokens in the completion
    pub completion_tokens: u32,

    /// Total number of tokens used
    pub total_tokens: u32,
}

impl Usage {
    /// Create a new usage record
    #[must_use]
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }

    /// Add another usage record to this one
    pub fn add(&mut self, other: &Self) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// Models list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    /// Object type
    pub object: String,
    /// List of models
    pub data: Vec<ModelObject>,
}

impl ModelsResponse {
    /// Create a new models response
    #[must_use]
    pub fn new(models: Vec<ModelObject>) -> Self {
        Self {
            object: "list".to_string(),
            data: models,
        }
    }
}

/// Model object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObject {
    /// Model ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: i64,
    /// Owner/organization
    pub owned_by: String,
}

impl ModelObject {
    /// Create a new model object
    #[must_use]
    pub fn new(id: impl Into<String>, owned_by: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model".to_string(),
            created: Utc::now().timestamp(),
            owned_by: owned_by.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_builder() {
        let response = GatewayResponse::builder()
            .model("gpt-4")
            .choice(Choice::new(0, "Hello!", FinishReason::Stop))
            .usage(Usage::new(10, 5))
            .provider("openai")
            .build();

        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.content(), Some("Hello!"));
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        assert_eq!(response.provider, Some("openai".to_string()));
    }

    #[test]
    fn test_usage_add() {
        let mut usage1 = Usage::new(10, 5);
        let usage2 = Usage::new(20, 10);

        usage1.add(&usage2);

        assert_eq!(usage1.prompt_tokens, 30);
        assert_eq!(usage1.completion_tokens, 15);
        assert_eq!(usage1.total_tokens, 45);
    }

    #[test]
    fn test_choice_with_tool_calls() {
        let tool_calls = vec![ToolCall {
            id: "call_123".to_string(),
            tool_type: "function".to_string(),
            function: crate::request::FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location": "NYC"}"#.to_string(),
            },
        }];

        let choice = Choice::with_tool_calls(0, tool_calls.clone(), FinishReason::ToolCalls);

        assert_eq!(choice.message.tool_calls, Some(tool_calls));
        assert!(choice.message.content.is_none());
        assert_eq!(choice.finish_reason, Some(FinishReason::ToolCalls));
    }

    #[test]
    fn test_response_serialization() {
        let response = GatewayResponse::builder()
            .id("test-123")
            .model("gpt-4")
            .choice(Choice::new(0, "Hello", FinishReason::Stop))
            .usage(Usage::new(5, 1))
            .build();

        let json = serde_json::to_string(&response).expect("serialize");
        let parsed: GatewayResponse = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.id, "test-123");
        assert_eq!(parsed.model, "gpt-4");
        assert_eq!(parsed.content(), Some("Hello"));
    }

    #[test]
    fn test_finish_reason_display() {
        assert_eq!(FinishReason::Stop.to_string(), "stop");
        assert_eq!(FinishReason::Length.to_string(), "length");
        assert_eq!(FinishReason::ToolCalls.to_string(), "tool_calls");
    }
}
