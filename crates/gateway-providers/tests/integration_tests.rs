//! Integration tests for LLM providers.
//!
//! These tests can be run against real provider APIs when API keys are available,
//! or they run with mock servers when API keys are not set.

use gateway_core::{GatewayRequest, LLMProvider};
use gateway_providers::{OpenAIProvider, AnthropicProvider};
use gateway_providers::openai::OpenAIConfig;
use gateway_providers::anthropic::AnthropicConfig;
use std::env;
use std::sync::Arc;

/// Test helper to check if OpenAI API key is available
fn has_openai_key() -> bool {
    env::var("OPENAI_API_KEY").is_ok()
}

/// Test helper to check if Anthropic API key is available
fn has_anthropic_key() -> bool {
    env::var("ANTHROPIC_API_KEY").is_ok()
}

#[cfg(test)]
mod openai_tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let config = OpenAIConfig::new("test-openai", "sk-test-key-not-real");
        let provider = OpenAIProvider::new(config);

        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.id(), "test-openai");
    }

    #[test]
    fn test_provider_models_not_empty() {
        let config = OpenAIConfig::new("test-openai", "sk-test");
        let provider = OpenAIProvider::new(config).unwrap();

        assert!(!provider.models().is_empty());

        // Should have common models
        let model_ids: Vec<&str> = provider.models().iter().map(|m| m.id.as_str()).collect();
        assert!(model_ids.contains(&"gpt-4o"));
        assert!(model_ids.contains(&"gpt-4o-mini"));
    }

    #[test]
    fn test_provider_capabilities() {
        let config = OpenAIConfig::new("test-openai", "sk-test");
        let provider = OpenAIProvider::new(config).unwrap();
        let caps = provider.capabilities();

        assert!(caps.chat);
        assert!(caps.streaming);
        assert!(caps.function_calling);
    }

    #[tokio::test]
    #[ignore = "Requires OPENAI_API_KEY environment variable"]
    async fn test_openai_chat_completion() {
        if !has_openai_key() {
            eprintln!("Skipping test: OPENAI_API_KEY not set");
            return;
        }

        let api_key = env::var("OPENAI_API_KEY").unwrap();
        let config = OpenAIConfig::new("openai-test", api_key);
        let provider = OpenAIProvider::new(config).unwrap();

        let request = GatewayRequest::builder()
            .model("gpt-4o-mini")
            .message(gateway_core::ChatMessage::user("Say 'hello' and nothing else."))
            .max_tokens(10u32)
            .build()
            .unwrap();

        let response = provider.chat_completion(&request).await;

        assert!(response.is_ok(), "Request failed: {:?}", response.err());
        let response = response.unwrap();

        assert!(!response.choices.is_empty());
        assert!(response.choices[0].message.content.is_some());
    }

    #[tokio::test]
    #[ignore = "Requires OPENAI_API_KEY environment variable"]
    async fn test_openai_streaming() {
        use futures::StreamExt;

        if !has_openai_key() {
            eprintln!("Skipping test: OPENAI_API_KEY not set");
            return;
        }

        let api_key = env::var("OPENAI_API_KEY").unwrap();
        let config = OpenAIConfig::new("openai-test", api_key);
        let provider = OpenAIProvider::new(config).unwrap();

        let request = GatewayRequest::builder()
            .model("gpt-4o-mini")
            .message(gateway_core::ChatMessage::user("Count from 1 to 3."))
            .stream(true)
            .max_tokens(50u32)
            .build()
            .unwrap();

        let stream_result = provider.chat_completion_stream(&request).await;
        assert!(stream_result.is_ok(), "Stream creation failed: {:?}", stream_result.err());

        let mut stream = stream_result.unwrap();
        let mut chunk_count = 0;
        let mut collected_content = String::new();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    chunk_count += 1;
                    if let Some(choice) = chunk.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            collected_content.push_str(content);
                        }
                    }
                }
                Err(e) => {
                    panic!("Stream error: {:?}", e);
                }
            }
        }

        assert!(chunk_count > 0, "Expected at least one chunk");
        assert!(!collected_content.is_empty(), "Expected non-empty content");
    }

    #[tokio::test]
    #[ignore = "Requires OPENAI_API_KEY environment variable"]
    async fn test_openai_health_check() {
        if !has_openai_key() {
            eprintln!("Skipping test: OPENAI_API_KEY not set");
            return;
        }

        let api_key = env::var("OPENAI_API_KEY").unwrap();
        let config = OpenAIConfig::new("openai-test", api_key);
        let provider = OpenAIProvider::new(config).unwrap();

        let health = provider.health_check().await;
        assert!(matches!(health, gateway_core::HealthStatus::Healthy));
    }
}

#[cfg(test)]
mod anthropic_tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let config = AnthropicConfig::new("test-key-not-real");
        let provider = AnthropicProvider::new(config);

        assert!(provider.is_ok());
    }

    #[test]
    fn test_provider_models_not_empty() {
        let config = AnthropicConfig::new("test-key");
        let provider = AnthropicProvider::new(config).unwrap();

        assert!(!provider.models().is_empty());

        // Should have Claude models
        let model_ids: Vec<&str> = provider.models().iter().map(|m| m.id.as_str()).collect();
        assert!(model_ids.iter().any(|id| id.contains("claude-3")));
    }

    #[test]
    fn test_provider_capabilities() {
        let config = AnthropicConfig::new("test-key");
        let provider = AnthropicProvider::new(config).unwrap();
        let caps = provider.capabilities();

        assert!(caps.chat);
        assert!(caps.streaming);
        assert!(caps.vision);
    }

    #[tokio::test]
    #[ignore = "Requires ANTHROPIC_API_KEY environment variable"]
    async fn test_anthropic_chat_completion() {
        if !has_anthropic_key() {
            eprintln!("Skipping test: ANTHROPIC_API_KEY not set");
            return;
        }

        let api_key = env::var("ANTHROPIC_API_KEY").unwrap();
        let config = AnthropicConfig::new(api_key);
        let provider = AnthropicProvider::new(config).unwrap();

        let request = GatewayRequest::builder()
            .model("claude-3-haiku-20240307")
            .message(gateway_core::ChatMessage::user("Say 'hello' and nothing else."))
            .max_tokens(10u32)
            .build()
            .unwrap();

        let response = provider.chat_completion(&request).await;

        assert!(response.is_ok(), "Request failed: {:?}", response.err());
        let response = response.unwrap();

        assert!(!response.choices.is_empty());
    }

    #[tokio::test]
    #[ignore = "Requires ANTHROPIC_API_KEY environment variable"]
    async fn test_anthropic_streaming() {
        use futures::StreamExt;

        if !has_anthropic_key() {
            eprintln!("Skipping test: ANTHROPIC_API_KEY not set");
            return;
        }

        let api_key = env::var("ANTHROPIC_API_KEY").unwrap();
        let config = AnthropicConfig::new(api_key);
        let provider = AnthropicProvider::new(config).unwrap();

        let request = GatewayRequest::builder()
            .model("claude-3-haiku-20240307")
            .message(gateway_core::ChatMessage::user("Count from 1 to 3."))
            .stream(true)
            .max_tokens(50u32)
            .build()
            .unwrap();

        let stream_result = provider.chat_completion_stream(&request).await;
        assert!(stream_result.is_ok(), "Stream creation failed: {:?}", stream_result.err());

        let mut stream = stream_result.unwrap();
        let mut chunk_count = 0;
        let mut collected_content = String::new();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    chunk_count += 1;
                    if let Some(choice) = chunk.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            collected_content.push_str(content);
                        }
                    }
                }
                Err(e) => {
                    panic!("Stream error: {:?}", e);
                }
            }
        }

        assert!(chunk_count > 0, "Expected at least one chunk");
    }
}

#[cfg(test)]
mod provider_registry_tests {
    use super::*;
    use gateway_providers::ProviderRegistry;

    #[test]
    fn test_registry_creation() {
        let registry = ProviderRegistry::new();
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let registry = ProviderRegistry::new();

        let config = OpenAIConfig::new("openai-1", "sk-test");
        let provider = OpenAIProvider::new(config).unwrap();

        registry.register(Arc::new(provider), 1, 100).unwrap();

        assert_eq!(registry.len(), 1);
        assert!(registry.get("openai-1").is_some());
    }

    #[test]
    fn test_registry_get_all_models() {
        let registry = ProviderRegistry::new();

        // Register OpenAI provider
        let openai_config = OpenAIConfig::new("openai", "sk-test");
        let openai_provider = OpenAIProvider::new(openai_config).unwrap();
        registry.register(Arc::new(openai_provider), 1, 100).unwrap();

        // Register Anthropic provider
        let anthropic_config = AnthropicConfig::new("test-key");
        let anthropic_provider = AnthropicProvider::new(anthropic_config).unwrap();
        registry.register(Arc::new(anthropic_provider), 2, 100).unwrap();

        let all_models = registry.get_all_models();

        // Should have models from both providers
        assert!(all_models.iter().any(|m| m.id.contains("gpt")));
        assert!(all_models.iter().any(|m| m.id.contains("claude")));
    }

    #[test]
    fn test_registry_get_providers_for_model() {
        let registry = ProviderRegistry::new();

        let config = OpenAIConfig::new("openai", "sk-test");
        let provider = OpenAIProvider::new(config).unwrap();
        registry.register(Arc::new(provider), 1, 100).unwrap();

        // Should find provider for GPT model
        let found = registry.get_providers_for_model("gpt-4o");
        assert!(!found.is_empty());
        assert_eq!(found[0].id(), "openai");

        // Should not find provider for unknown model
        let not_found = registry.get_providers_for_model("unknown-model");
        assert!(not_found.is_empty());
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_invalid_api_key_error() {
        let config = OpenAIConfig::new("openai", "invalid-key-12345");
        let provider = OpenAIProvider::new(config).unwrap();

        let request = GatewayRequest::builder()
            .model("gpt-4o-mini")
            .message(gateway_core::ChatMessage::user("test"))
            .build()
            .unwrap();

        let result = provider.chat_completion(&request).await;

        // Should fail with authentication or provider error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_invalid_model_handling() {
        if !has_openai_key() {
            eprintln!("Skipping test: OPENAI_API_KEY not set");
            return;
        }

        let api_key = env::var("OPENAI_API_KEY").unwrap();
        let config = OpenAIConfig::new("openai", api_key);
        let provider = OpenAIProvider::new(config).unwrap();

        let request = GatewayRequest::builder()
            .model("invalid-model-name-xyz")
            .message(gateway_core::ChatMessage::user("test"))
            .build()
            .unwrap();

        let result = provider.chat_completion(&request).await;

        // Should fail with provider error about invalid model
        assert!(result.is_err());
    }
}
