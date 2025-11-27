//! Provider registry for managing multiple LLM providers.
//!
//! The registry provides:
//! - Dynamic provider registration
//! - Model to provider mapping
//! - Health status caching

use dashmap::DashMap;
use gateway_core::{GatewayError, HealthStatus, LLMProvider, ModelInfo};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Cached health status
#[derive(Debug, Clone)]
pub struct CachedHealth {
    /// Health status
    pub status: HealthStatus,
    /// When the status was last updated
    pub updated_at: Instant,
    /// Time-to-live for the cache
    pub ttl: Duration,
}

impl CachedHealth {
    /// Create a new cached health entry
    #[must_use]
    pub fn new(status: HealthStatus, ttl: Duration) -> Self {
        Self {
            status,
            updated_at: Instant::now(),
            ttl,
        }
    }

    /// Check if the cached value is still valid
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.updated_at.elapsed() < self.ttl
    }
}

/// Provider entry in the registry
pub struct ProviderEntry {
    /// The provider instance
    pub provider: Arc<dyn LLMProvider>,
    /// Priority (lower = higher priority)
    pub priority: u32,
    /// Weight for load balancing
    pub weight: u32,
    /// Whether the provider is enabled
    pub enabled: bool,
}

/// Provider registry for managing multiple LLM providers
pub struct ProviderRegistry {
    /// Registered providers by ID
    providers: DashMap<String, ProviderEntry>,
    /// Health status cache
    health_cache: DashMap<String, CachedHealth>,
    /// Model to provider mappings
    model_index: DashMap<String, Vec<String>>,
    /// Health cache TTL
    health_ttl: Duration,
}

impl ProviderRegistry {
    /// Create a new provider registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            providers: DashMap::new(),
            health_cache: DashMap::new(),
            model_index: DashMap::new(),
            health_ttl: Duration::from_secs(30),
        }
    }

    /// Create with custom health cache TTL
    #[must_use]
    pub fn with_health_ttl(health_ttl: Duration) -> Self {
        Self {
            providers: DashMap::new(),
            health_cache: DashMap::new(),
            model_index: DashMap::new(),
            health_ttl,
        }
    }

    /// Register a new provider
    ///
    /// # Errors
    /// Returns error if provider ID is already registered
    pub fn register(
        &self,
        provider: Arc<dyn LLMProvider>,
        priority: u32,
        weight: u32,
    ) -> Result<(), GatewayError> {
        let id = provider.id().to_string();

        if self.providers.contains_key(&id) {
            return Err(GatewayError::Configuration {
                message: format!("Provider already registered: {id}"),
            });
        }

        // Index models
        for model in provider.models() {
            self.model_index
                .entry(model.id.clone())
                .or_default()
                .push(id.clone());

            // Also index aliases
            for alias in &model.aliases {
                self.model_index
                    .entry(alias.clone())
                    .or_default()
                    .push(id.clone());
            }
        }

        self.providers.insert(
            id.clone(),
            ProviderEntry {
                provider,
                priority,
                weight,
                enabled: true,
            },
        );

        info!(provider_id = %id, "Provider registered");
        Ok(())
    }

    /// Deregister a provider
    pub fn deregister(&self, id: &str) -> Option<Arc<dyn LLMProvider>> {
        if let Some((_, entry)) = self.providers.remove(id) {
            // Remove from model index
            self.model_index.retain(|_, providers| {
                providers.retain(|p| p != id);
                !providers.is_empty()
            });

            // Remove from health cache
            self.health_cache.remove(id);

            info!(provider_id = %id, "Provider deregistered");
            Some(entry.provider)
        } else {
            None
        }
    }

    /// Get a provider by ID
    #[must_use]
    pub fn get(&self, id: &str) -> Option<Arc<dyn LLMProvider>> {
        self.providers.get(id).map(|e| Arc::clone(&e.provider))
    }

    /// Get a provider entry by ID
    #[must_use]
    pub fn get_entry(&self, id: &str) -> Option<dashmap::mapref::one::Ref<'_, String, ProviderEntry>> {
        self.providers.get(id)
    }

    /// Get all provider IDs
    #[must_use]
    pub fn provider_ids(&self) -> Vec<String> {
        self.providers.iter().map(|e| e.key().clone()).collect()
    }

    /// Get providers that support a specific model
    #[must_use]
    pub fn get_providers_for_model(&self, model: &str) -> Vec<Arc<dyn LLMProvider>> {
        self.model_index
            .get(model)
            .map(|provider_ids| {
                provider_ids
                    .iter()
                    .filter_map(|id| self.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all enabled providers
    #[must_use]
    pub fn get_enabled_providers(&self) -> Vec<Arc<dyn LLMProvider>> {
        self.providers
            .iter()
            .filter(|e| e.enabled)
            .map(|e| Arc::clone(&e.provider))
            .collect()
    }

    /// Get healthy providers (from cache or by checking)
    pub async fn get_healthy_providers(&self) -> Vec<Arc<dyn LLMProvider>> {
        let mut healthy = Vec::new();

        for entry in &self.providers {
            if !entry.enabled {
                continue;
            }

            let health = self.get_health(entry.key()).await;
            if health.should_route() {
                healthy.push(Arc::clone(&entry.provider));
            }
        }

        healthy
    }

    /// Get healthy providers for a specific model
    pub async fn get_healthy_providers_for_model(&self, model: &str) -> Vec<Arc<dyn LLMProvider>> {
        let mut healthy = Vec::new();

        let provider_ids = self
            .model_index
            .get(model)
            .map(|r| r.value().clone())
            .unwrap_or_default();

        for id in provider_ids {
            if let Some(entry) = self.providers.get(&id) {
                if !entry.enabled {
                    continue;
                }

                let health = self.get_health(&id).await;
                if health.should_route() {
                    healthy.push(Arc::clone(&entry.provider));
                }
            }
        }

        healthy
    }

    /// Get cached health status or check provider
    pub async fn get_health(&self, provider_id: &str) -> HealthStatus {
        // Check cache first
        if let Some(cached) = self.health_cache.get(provider_id) {
            if cached.is_valid() {
                return cached.status;
            }
        }

        // Get fresh health status
        let status = if let Some(provider) = self.get(provider_id) {
            provider.health_check().await
        } else {
            HealthStatus::Unknown
        };

        // Update cache
        self.health_cache.insert(
            provider_id.to_string(),
            CachedHealth::new(status, self.health_ttl),
        );

        status
    }

    /// Update health status in cache
    pub fn update_health(&self, provider_id: &str, status: HealthStatus) {
        self.health_cache.insert(
            provider_id.to_string(),
            CachedHealth::new(status, self.health_ttl),
        );

        debug!(
            provider_id = %provider_id,
            status = %status,
            "Health status updated"
        );
    }

    /// Enable a provider
    pub fn enable(&self, id: &str) -> bool {
        if let Some(mut entry) = self.providers.get_mut(id) {
            entry.enabled = true;
            info!(provider_id = %id, "Provider enabled");
            true
        } else {
            false
        }
    }

    /// Disable a provider
    pub fn disable(&self, id: &str) -> bool {
        if let Some(mut entry) = self.providers.get_mut(id) {
            entry.enabled = false;
            info!(provider_id = %id, "Provider disabled");
            true
        } else {
            false
        }
    }

    /// Get all available models across all providers
    #[must_use]
    pub fn get_all_models(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for entry in &self.providers {
            if !entry.enabled {
                continue;
            }

            for model in entry.provider.models() {
                if seen.insert(model.id.clone()) {
                    models.push(model.clone());
                }
            }
        }

        models
    }

    /// Get number of registered providers
    #[must_use]
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Check if registry is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Clear all providers
    pub fn clear(&self) {
        self.providers.clear();
        self.model_index.clear();
        self.health_cache.clear();
        info!("Provider registry cleared");
    }

    /// Run health checks on all providers
    pub async fn refresh_health(&self) {
        for entry in &self.providers {
            let id = entry.key().clone();
            let provider = Arc::clone(&entry.provider);

            let status = provider.health_check().await;
            self.update_health(&id, status);
        }

        debug!("Health check completed for all providers");
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gateway_core::{
        ChatChunk, GatewayRequest, GatewayResponse, ProviderCapabilities, ProviderType,
    };
    use futures::stream::BoxStream;

    struct MockProvider {
        id: String,
        models: Vec<ModelInfo>,
        health: HealthStatus,
    }

    impl MockProvider {
        fn new(id: &str, models: Vec<&str>) -> Self {
            Self {
                id: id.to_string(),
                models: models
                    .into_iter()
                    .map(|m| ModelInfo::new(m))
                    .collect(),
                health: HealthStatus::Healthy,
            }
        }
    }

    #[async_trait::async_trait]
    impl LLMProvider for MockProvider {
        fn id(&self) -> &str {
            &self.id
        }

        fn provider_type(&self) -> ProviderType {
            ProviderType::Custom
        }

        async fn chat_completion(&self, _: &GatewayRequest) -> Result<GatewayResponse, GatewayError> {
            unimplemented!()
        }

        async fn chat_completion_stream(
            &self,
            _: &GatewayRequest,
        ) -> Result<BoxStream<'static, Result<ChatChunk, GatewayError>>, GatewayError> {
            unimplemented!()
        }

        async fn health_check(&self) -> HealthStatus {
            self.health
        }

        fn capabilities(&self) -> &ProviderCapabilities {
            static CAPS: ProviderCapabilities = ProviderCapabilities {
                chat: true,
                streaming: true,
                function_calling: false,
                vision: false,
                embeddings: false,
                json_mode: false,
                seed: false,
                logprobs: false,
                max_context_length: None,
                max_output_tokens: None,
                parallel_tool_calls: false,
            };
            &CAPS
        }

        fn models(&self) -> &[ModelInfo] {
            &self.models
        }

        fn base_url(&self) -> &str {
            "http://localhost"
        }
    }

    #[test]
    fn test_register_provider() {
        let registry = ProviderRegistry::new();
        let provider = Arc::new(MockProvider::new("test", vec!["model-1"]));

        assert!(registry.register(provider, 100, 100).is_ok());
        assert!(registry.get("test").is_some());
    }

    #[test]
    fn test_duplicate_registration() {
        let registry = ProviderRegistry::new();
        let provider1 = Arc::new(MockProvider::new("test", vec!["model-1"]));
        let provider2 = Arc::new(MockProvider::new("test", vec!["model-2"]));

        assert!(registry.register(provider1, 100, 100).is_ok());
        assert!(registry.register(provider2, 100, 100).is_err());
    }

    #[test]
    fn test_model_index() {
        let registry = ProviderRegistry::new();
        let provider1 = Arc::new(MockProvider::new("openai", vec!["gpt-4", "gpt-3.5"]));
        let provider2 = Arc::new(MockProvider::new("azure", vec!["gpt-4"]));

        registry.register(provider1, 100, 100).expect("register");
        registry.register(provider2, 100, 100).expect("register");

        let providers = registry.get_providers_for_model("gpt-4");
        assert_eq!(providers.len(), 2);

        let providers = registry.get_providers_for_model("gpt-3.5");
        assert_eq!(providers.len(), 1);
    }

    #[test]
    fn test_deregister() {
        let registry = ProviderRegistry::new();
        let provider = Arc::new(MockProvider::new("test", vec!["model-1"]));

        registry.register(provider, 100, 100).expect("register");
        assert!(registry.get("test").is_some());

        let removed = registry.deregister("test");
        assert!(removed.is_some());
        assert!(registry.get("test").is_none());
    }

    #[test]
    fn test_enable_disable() {
        let registry = ProviderRegistry::new();
        let provider = Arc::new(MockProvider::new("test", vec!["model-1"]));

        registry.register(provider, 100, 100).expect("register");

        let enabled = registry.get_enabled_providers();
        assert_eq!(enabled.len(), 1);

        registry.disable("test");
        let enabled = registry.get_enabled_providers();
        assert_eq!(enabled.len(), 0);

        registry.enable("test");
        let enabled = registry.get_enabled_providers();
        assert_eq!(enabled.len(), 1);
    }

    #[tokio::test]
    async fn test_health_cache() {
        let registry = ProviderRegistry::with_health_ttl(Duration::from_secs(60));
        let provider = Arc::new(MockProvider::new("test", vec!["model-1"]));

        registry.register(provider, 100, 100).expect("register");

        let health = registry.get_health("test").await;
        assert_eq!(health, HealthStatus::Healthy);

        // Should use cached value
        let health = registry.get_health("test").await;
        assert_eq!(health, HealthStatus::Healthy);
    }
}
