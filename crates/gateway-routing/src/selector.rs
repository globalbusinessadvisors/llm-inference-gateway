//! Provider selection logic.
//!
//! Handles selecting the best provider from a set of candidates
//! based on various criteria including health, capability, and cost.

use gateway_core::{GatewayRequest, HealthStatus, LLMProvider, ProviderCapabilities};
use std::sync::Arc;
use tracing::debug;

/// Criteria for selecting a provider
#[derive(Debug, Clone, Default)]
pub struct SelectionCriteria {
    /// Required model support
    pub model: Option<String>,
    /// Required capabilities
    pub capabilities: RequiredCapabilities,
    /// Prefer providers with lower latency
    pub prefer_low_latency: bool,
    /// Prefer providers with higher success rate
    pub prefer_high_success: bool,
    /// Minimum health status
    pub min_health: HealthStatus,
    /// Excluded provider IDs
    pub exclude_providers: Vec<String>,
    /// Preferred provider IDs (hint, not requirement)
    pub prefer_providers: Vec<String>,
}

impl SelectionCriteria {
    /// Create new selection criteria
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set required model
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set required capabilities
    #[must_use]
    pub fn with_capabilities(mut self, capabilities: RequiredCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Require streaming support
    #[must_use]
    pub fn require_streaming(mut self) -> Self {
        self.capabilities.streaming = true;
        self
    }

    /// Require function calling support
    #[must_use]
    pub fn require_function_calling(mut self) -> Self {
        self.capabilities.function_calling = true;
        self
    }

    /// Require vision support
    #[must_use]
    pub fn require_vision(mut self) -> Self {
        self.capabilities.vision = true;
        self
    }

    /// Set minimum health status
    #[must_use]
    pub fn with_min_health(mut self, health: HealthStatus) -> Self {
        self.min_health = health;
        self
    }

    /// Exclude a provider
    #[must_use]
    pub fn exclude_provider(mut self, provider_id: impl Into<String>) -> Self {
        self.exclude_providers.push(provider_id.into());
        self
    }

    /// Prefer a provider
    #[must_use]
    pub fn prefer_provider(mut self, provider_id: impl Into<String>) -> Self {
        self.prefer_providers.push(provider_id.into());
        self
    }

    /// Create criteria from a gateway request
    #[must_use]
    pub fn from_request(request: &GatewayRequest) -> Self {
        let mut criteria = Self::new().with_model(&request.model);

        // Check if request needs streaming
        if request.stream {
            criteria = criteria.require_streaming();
        }

        // Check if request needs function calling
        if request.tools.is_some() {
            criteria = criteria.require_function_calling();
        }

        // Check if request has images (vision)
        let has_images = request.messages.iter().any(|msg| {
            if let gateway_core::MessageContent::Parts(parts) = &msg.content {
                parts.iter().any(|p| {
                    matches!(p, gateway_core::ContentPart::ImageUrl { .. })
                })
            } else {
                false
            }
        });
        if has_images {
            criteria = criteria.require_vision();
        }

        criteria
    }
}

/// Required capabilities for provider selection
#[derive(Debug, Clone, Default)]
pub struct RequiredCapabilities {
    /// Requires streaming support
    pub streaming: bool,
    /// Requires function calling
    pub function_calling: bool,
    /// Requires vision support
    pub vision: bool,
    /// Requires embeddings
    pub embeddings: bool,
    /// Requires JSON mode
    pub json_mode: bool,
    /// Minimum context length
    pub min_context_length: Option<usize>,
}

impl RequiredCapabilities {
    /// Check if a provider's capabilities meet these requirements
    #[must_use]
    pub fn satisfied_by(&self, caps: &ProviderCapabilities) -> bool {
        if self.streaming && !caps.streaming {
            return false;
        }
        if self.function_calling && !caps.function_calling {
            return false;
        }
        if self.vision && !caps.vision {
            return false;
        }
        if self.embeddings && !caps.embeddings {
            return false;
        }
        if self.json_mode && !caps.json_mode {
            return false;
        }
        if let Some(min_ctx) = self.min_context_length {
            if let Some(max_ctx) = caps.max_context_length {
                if (max_ctx as usize) < min_ctx {
                    return false;
                }
            }
        }
        true
    }
}

/// A candidate provider for selection
#[derive(Clone)]
pub struct ProviderCandidate {
    /// Provider identifier
    pub id: String,
    /// Provider reference
    pub provider: Arc<dyn LLMProvider>,
    /// Current health status
    pub health: HealthStatus,
    /// Weight for load balancing
    pub weight: u32,
    /// Priority (lower = higher priority)
    pub priority: u32,
    /// Average latency in milliseconds
    pub avg_latency_ms: Option<f64>,
    /// Success rate (0.0 - 1.0)
    pub success_rate: Option<f64>,
    /// Whether this is a preferred provider
    pub preferred: bool,
}

impl ProviderCandidate {
    /// Create a new provider candidate
    #[must_use]
    pub fn new(provider: Arc<dyn LLMProvider>) -> Self {
        Self {
            id: provider.id().to_string(),
            provider,
            health: HealthStatus::Unknown,
            weight: 100,
            priority: 100,
            avg_latency_ms: None,
            success_rate: None,
            preferred: false,
        }
    }

    /// Set health status
    #[must_use]
    pub fn with_health(mut self, health: HealthStatus) -> Self {
        self.health = health;
        self
    }

    /// Set weight
    #[must_use]
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    /// Set priority
    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Mark as preferred
    #[must_use]
    pub fn preferred(mut self) -> Self {
        self.preferred = true;
        self
    }

    /// Calculate a composite score for this candidate
    #[must_use]
    pub fn score(&self) -> f64 {
        let mut score = 100.0;

        // Health factor
        match self.health {
            HealthStatus::Healthy => {}
            HealthStatus::Degraded => score *= 0.7,
            HealthStatus::Unhealthy => score *= 0.1,
            HealthStatus::Unknown => score *= 0.5,
        }

        // Priority factor (lower priority number = higher score)
        score *= 1.0 / (1.0 + (f64::from(self.priority) / 100.0));

        // Latency factor
        if let Some(latency) = self.avg_latency_ms {
            // Penalize high latency
            score *= 1000.0 / (1000.0 + latency);
        }

        // Success rate factor
        if let Some(rate) = self.success_rate {
            score *= rate;
        }

        // Preferred bonus
        if self.preferred {
            score *= 1.5;
        }

        // Weight factor
        score *= f64::from(self.weight) / 100.0;

        score
    }
}

/// Provider selector for choosing the best provider
pub struct ProviderSelector;

impl ProviderSelector {
    /// Filter candidates based on selection criteria
    #[must_use]
    pub fn filter(
        candidates: &[ProviderCandidate],
        criteria: &SelectionCriteria,
    ) -> Vec<ProviderCandidate> {
        candidates
            .iter()
            .filter(|c| {
                // Check exclusions
                if criteria.exclude_providers.contains(&c.id) {
                    debug!(provider = %c.id, "Excluded by criteria");
                    return false;
                }

                // Check health
                if !meets_health_requirement(c.health, criteria.min_health) {
                    debug!(
                        provider = %c.id,
                        health = %c.health,
                        required = %criteria.min_health,
                        "Health requirement not met"
                    );
                    return false;
                }

                // Check model support
                if let Some(model) = &criteria.model {
                    let models = c.provider.models();
                    let supports_model = models.iter().any(|m| {
                        m.id == *model || m.aliases.contains(model)
                    });
                    if !supports_model {
                        debug!(
                            provider = %c.id,
                            model = %model,
                            "Model not supported"
                        );
                        return false;
                    }
                }

                // Check capabilities
                if !criteria.capabilities.satisfied_by(c.provider.capabilities()) {
                    debug!(
                        provider = %c.id,
                        "Capabilities not satisfied"
                    );
                    return false;
                }

                true
            })
            .cloned()
            .map(|mut c| {
                // Mark preferred providers
                if criteria.prefer_providers.contains(&c.id) {
                    c.preferred = true;
                }
                c
            })
            .collect()
    }

    /// Select the best candidate from a filtered list
    #[must_use]
    pub fn select_best(candidates: &[ProviderCandidate]) -> Option<&ProviderCandidate> {
        if candidates.is_empty() {
            return None;
        }

        candidates.iter().max_by(|a, b| {
            a.score().partial_cmp(&b.score()).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Select multiple candidates sorted by score
    #[must_use]
    pub fn select_ranked(
        candidates: &[ProviderCandidate],
        limit: usize,
    ) -> Vec<&ProviderCandidate> {
        let mut sorted: Vec<_> = candidates.iter().collect();
        sorted.sort_by(|a, b| {
            b.score().partial_cmp(&a.score()).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(limit).collect()
    }
}

/// Check if a health status meets the minimum requirement
fn meets_health_requirement(actual: HealthStatus, required: HealthStatus) -> bool {
    match required {
        HealthStatus::Healthy => actual == HealthStatus::Healthy,
        HealthStatus::Degraded => matches!(actual, HealthStatus::Healthy | HealthStatus::Degraded),
        HealthStatus::Unhealthy | HealthStatus::Unknown => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gateway_core::{ChatChunk, GatewayError, GatewayResponse, ModelInfo, ProviderType};
    use futures::stream::BoxStream;

    struct MockProvider {
        id: String,
        models: Vec<ModelInfo>,
        capabilities: ProviderCapabilities,
    }

    impl MockProvider {
        fn new(id: &str) -> Self {
            Self {
                id: id.to_string(),
                models: vec![ModelInfo::new("test-model")],
                capabilities: ProviderCapabilities::default(),
            }
        }

        fn with_model(mut self, model: &str) -> Self {
            self.models.push(ModelInfo::new(model));
            self
        }

        fn with_streaming(mut self) -> Self {
            self.capabilities.streaming = true;
            self
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
            HealthStatus::Healthy
        }

        fn capabilities(&self) -> &ProviderCapabilities {
            &self.capabilities
        }

        fn models(&self) -> &[ModelInfo] {
            &self.models
        }

        fn base_url(&self) -> &str {
            "http://localhost"
        }
    }

    #[test]
    fn test_filter_by_model() {
        let provider1 = Arc::new(MockProvider::new("p1").with_model("gpt-4"));
        let provider2 = Arc::new(MockProvider::new("p2").with_model("claude-3"));

        let candidates = vec![
            ProviderCandidate::new(provider1).with_health(HealthStatus::Healthy),
            ProviderCandidate::new(provider2).with_health(HealthStatus::Healthy),
        ];

        let criteria = SelectionCriteria::new().with_model("gpt-4");
        let filtered = ProviderSelector::filter(&candidates, &criteria);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "p1");
    }

    #[test]
    fn test_filter_by_capabilities() {
        let provider1 = Arc::new(MockProvider::new("p1").with_streaming());
        let provider2 = Arc::new(MockProvider::new("p2"));

        let candidates = vec![
            ProviderCandidate::new(provider1).with_health(HealthStatus::Healthy),
            ProviderCandidate::new(provider2).with_health(HealthStatus::Healthy),
        ];

        let criteria = SelectionCriteria::new().require_streaming();
        let filtered = ProviderSelector::filter(&candidates, &criteria);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "p1");
    }

    #[test]
    fn test_filter_by_health() {
        let provider1 = Arc::new(MockProvider::new("p1"));
        let provider2 = Arc::new(MockProvider::new("p2"));

        let candidates = vec![
            ProviderCandidate::new(provider1).with_health(HealthStatus::Healthy),
            ProviderCandidate::new(provider2).with_health(HealthStatus::Unhealthy),
        ];

        let criteria = SelectionCriteria::new().with_min_health(HealthStatus::Healthy);
        let filtered = ProviderSelector::filter(&candidates, &criteria);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "p1");
    }

    #[test]
    fn test_exclude_providers() {
        let provider1 = Arc::new(MockProvider::new("p1"));
        let provider2 = Arc::new(MockProvider::new("p2"));

        let candidates = vec![
            ProviderCandidate::new(provider1).with_health(HealthStatus::Healthy),
            ProviderCandidate::new(provider2).with_health(HealthStatus::Healthy),
        ];

        let criteria = SelectionCriteria::new().exclude_provider("p1");
        let filtered = ProviderSelector::filter(&candidates, &criteria);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "p2");
    }

    #[test]
    fn test_prefer_providers() {
        let provider1 = Arc::new(MockProvider::new("p1"));
        let provider2 = Arc::new(MockProvider::new("p2"));

        let candidates = vec![
            ProviderCandidate::new(provider1).with_health(HealthStatus::Healthy),
            ProviderCandidate::new(provider2).with_health(HealthStatus::Healthy),
        ];

        let criteria = SelectionCriteria::new().prefer_provider("p2");
        let filtered = ProviderSelector::filter(&candidates, &criteria);

        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().find(|c| c.id == "p2").map(|c| c.preferred).unwrap_or(false));
    }

    #[test]
    fn test_select_best() {
        let provider1 = Arc::new(MockProvider::new("p1"));
        let provider2 = Arc::new(MockProvider::new("p2"));

        let candidates = vec![
            ProviderCandidate::new(provider1)
                .with_health(HealthStatus::Healthy)
                .with_priority(100),
            ProviderCandidate::new(provider2)
                .with_health(HealthStatus::Healthy)
                .with_priority(10), // Lower = better
        ];

        let best = ProviderSelector::select_best(&candidates);
        assert!(best.is_some());
        assert_eq!(best.map(|b| &b.id), Some(&"p2".to_string()));
    }

    #[test]
    fn test_candidate_score() {
        let provider: Arc<dyn LLMProvider> = Arc::new(MockProvider::new("test"));

        let healthy = ProviderCandidate::new(Arc::clone(&provider))
            .with_health(HealthStatus::Healthy);
        let degraded = ProviderCandidate::new(Arc::clone(&provider))
            .with_health(HealthStatus::Degraded);

        assert!(healthy.score() > degraded.score());
    }
}
