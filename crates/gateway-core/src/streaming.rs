//! Streaming response types for the gateway.
//!
//! This module defines the types used for Server-Sent Events (SSE) streaming responses.

use crate::request::MessageRole;
use crate::response::{FinishReason, Usage};
use serde::{Deserialize, Serialize};

/// Streaming chat chunk (SSE data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunk {
    /// Unique response identifier
    pub id: String,

    /// Object type (always "chat.completion.chunk")
    pub object: String,

    /// Creation timestamp (Unix epoch seconds)
    pub created: i64,

    /// Model used for completion
    pub model: String,

    /// Chunk choices
    pub choices: Vec<ChunkChoice>,

    /// Usage statistics (only in last chunk if stream_options.include_usage is true)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// System fingerprint for reproducibility
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

impl ChatChunk {
    /// Create a new chunk builder
    #[must_use]
    pub fn builder() -> ChatChunkBuilder {
        ChatChunkBuilder::default()
    }

    /// Get the delta content from the first choice
    #[must_use]
    pub fn content(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.delta.content.as_deref())
    }

    /// Get the finish reason from the first choice
    #[must_use]
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.choices.first().and_then(|c| c.finish_reason)
    }

    /// Check if this is the final chunk
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.choices
            .first()
            .is_some_and(|c| c.finish_reason.is_some())
    }

    /// Convert to SSE data format
    #[must_use]
    pub fn to_sse_data(&self) -> String {
        format!(
            "data: {}\n\n",
            serde_json::to_string(self).unwrap_or_default()
        )
    }
}

/// Builder for `ChatChunk`
#[derive(Debug, Default)]
pub struct ChatChunkBuilder {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<ChunkChoice>,
    usage: Option<Usage>,
    system_fingerprint: Option<String>,
}

impl ChatChunkBuilder {
    /// Set the chunk ID
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
    pub fn choices(mut self, choices: Vec<ChunkChoice>) -> Self {
        self.choices = choices;
        self
    }

    /// Add a choice
    #[must_use]
    pub fn choice(mut self, choice: ChunkChoice) -> Self {
        self.choices.push(choice);
        self
    }

    /// Set the usage (for final chunk)
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

    /// Build the chunk
    #[must_use]
    pub fn build(self) -> ChatChunk {
        ChatChunk {
            id: self
                .id
                .unwrap_or_else(|| format!("chatcmpl-{}", uuid::Uuid::new_v4())),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: self.model.unwrap_or_default(),
            choices: self.choices,
            usage: self.usage,
            system_fingerprint: self.system_fingerprint,
        }
    }
}

/// Streaming choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    /// Index of this choice
    pub index: u32,

    /// The delta (incremental content)
    pub delta: ChunkDelta,

    /// Reason for finishing (only in last chunk)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Log probabilities (if requested)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

impl ChunkChoice {
    /// Create a new chunk choice with content delta
    #[must_use]
    pub fn with_content(index: u32, content: impl Into<String>) -> Self {
        Self {
            index,
            delta: ChunkDelta {
                role: None,
                content: Some(content.into()),
                tool_calls: None,
                function_call: None,
            },
            finish_reason: None,
            logprobs: None,
        }
    }

    /// Create a new chunk choice with role (first chunk)
    #[must_use]
    pub fn with_role(index: u32, role: MessageRole) -> Self {
        Self {
            index,
            delta: ChunkDelta {
                role: Some(role),
                content: None,
                tool_calls: None,
                function_call: None,
            },
            finish_reason: None,
            logprobs: None,
        }
    }

    /// Create a new chunk choice with finish reason (last chunk)
    #[must_use]
    pub fn with_finish(index: u32, finish_reason: FinishReason) -> Self {
        Self {
            index,
            delta: ChunkDelta::default(),
            finish_reason: Some(finish_reason),
            logprobs: None,
        }
    }

    /// Create a new chunk choice with tool call delta
    #[must_use]
    pub fn with_tool_call(index: u32, tool_calls: Vec<ToolCallDelta>) -> Self {
        Self {
            index,
            delta: ChunkDelta {
                role: None,
                content: None,
                tool_calls: Some(tool_calls),
                function_call: None,
            },
            finish_reason: None,
            logprobs: None,
        }
    }
}

/// Delta (incremental) content
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkDelta {
    /// Role (only in first chunk)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<MessageRole>,

    /// Content delta
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool calls delta
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,

    /// Deprecated: Function call delta
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCallDelta>,
}

/// Tool call delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Index of the tool call
    pub index: u32,

    /// Tool call ID (only in first chunk for this tool call)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Tool type (only in first chunk for this tool call)
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,

    /// Function call delta
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// Function call delta for streaming
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    /// Function name (only in first chunk)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Arguments delta
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Stream options for requests
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Include usage in final chunk
    #[serde(default)]
    pub include_usage: bool,
}

/// SSE message for the stream
#[derive(Debug, Clone)]
pub enum StreamMessage {
    /// Data chunk
    Data(ChatChunk),
    /// Stream done marker
    Done,
    /// Error message
    Error(String),
}

impl StreamMessage {
    /// Convert to SSE format
    #[must_use]
    pub fn to_sse(&self) -> String {
        match self {
            Self::Data(chunk) => chunk.to_sse_data(),
            Self::Done => "data: [DONE]\n\n".to_string(),
            Self::Error(err) => format!("event: error\ndata: {err}\n\n"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_builder() {
        let chunk = ChatChunk::builder()
            .id("test-123")
            .model("gpt-4")
            .choice(ChunkChoice::with_content(0, "Hello"))
            .build();

        assert_eq!(chunk.id, "test-123");
        assert_eq!(chunk.model, "gpt-4");
        assert_eq!(chunk.content(), Some("Hello"));
        assert!(!chunk.is_done());
    }

    #[test]
    fn test_chunk_choice_with_role() {
        let choice = ChunkChoice::with_role(0, MessageRole::Assistant);
        assert_eq!(choice.delta.role, Some(MessageRole::Assistant));
        assert!(choice.delta.content.is_none());
    }

    #[test]
    fn test_chunk_choice_with_finish() {
        let choice = ChunkChoice::with_finish(0, FinishReason::Stop);
        assert_eq!(choice.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_chunk_is_done() {
        let chunk = ChatChunk::builder()
            .choice(ChunkChoice::with_finish(0, FinishReason::Stop))
            .build();

        assert!(chunk.is_done());
        assert_eq!(chunk.finish_reason(), Some(FinishReason::Stop));
    }

    #[test]
    fn test_sse_format() {
        let chunk = ChatChunk::builder()
            .id("test")
            .model("gpt-4")
            .choice(ChunkChoice::with_content(0, "Hi"))
            .build();

        let sse = chunk.to_sse_data();
        assert!(sse.starts_with("data: "));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn test_stream_message() {
        let data = StreamMessage::Data(ChatChunk::builder().build());
        assert!(data.to_sse().starts_with("data: "));

        let done = StreamMessage::Done;
        assert_eq!(done.to_sse(), "data: [DONE]\n\n");

        let error = StreamMessage::Error("test error".to_string());
        assert!(error.to_sse().contains("event: error"));
    }

    #[test]
    fn test_tool_call_delta() {
        let delta = ToolCallDelta {
            index: 0,
            id: Some("call_123".to_string()),
            tool_type: Some("function".to_string()),
            function: Some(FunctionCallDelta {
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"loc"#.to_string()),
            }),
        };

        let choice = ChunkChoice::with_tool_call(0, vec![delta]);
        assert!(choice.delta.tool_calls.is_some());
    }
}
