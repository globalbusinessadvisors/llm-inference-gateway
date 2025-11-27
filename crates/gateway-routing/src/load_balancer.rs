//! Load balancer for distributing requests across providers.
//!
//! Combines load balancing strategies with provider selection
//! to make intelligent routing decisions.

use crate::selector::{ProviderCandidate, ProviderSelector, SelectionCriteria};
use crate::strategy::{LoadBalancingStrategy, ProviderStats, StrategyFactory};
use gateway_core::{GatewayError, HealthStatus, LLMProvider};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Load balancer configuration
#[derive(Debug, Clone)]
pub struct LoadBalancerConfig {
    /// Default load balancing strategy
    pub strategy: String,
    /// Whether to consider health status
    pub health_aware: bool,
    /// Minimum healthy providers to serve traffic
    pub min_healthy: usize,
    /// Enable sticky sessions (route same tenant to same provider)
    pub sticky_sessions: bool,
    /// Sticky session TTL
    pub sticky_ttl: Duration,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            strategy: "round_robin".to_string(),
            health_aware: true,
            min_healthy: 1,
            sticky_sessions: false,
            sticky_ttl: Duration::from_secs(300),
        }
    }
}

impl LoadBalancerConfig {
    /// Create a new configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the strategy
    #[must_use]
    pub fn with_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.strategy = strategy.into();
        self
    }

    /// Enable/disable health awareness
    #[must_use]
    pub fn with_health_aware(mut self, enabled: bool) -> Self {
        self.health_aware = enabled;
        self
    }

    /// Enable sticky sessions
    #[must_use]
    pub fn with_sticky_sessions(mut self, enabled: bool, ttl: Duration) -> Self {
        self.sticky_sessions = enabled;
        self.sticky_ttl = ttl;
        self
    }
}

/// Load balancer for provider selection
pub struct LoadBalancer {
    /// Configuration
    config: LoadBalancerConfig,
    /// Load balancing strategy
    strategy: Box<dyn LoadBalancingStrategy>,
    /// Provider statistics
    stats: RwLock<HashMap<String, ProviderMetrics>>,
    /// Sticky session mappings (tenant_id -> provider_id)
    sticky_map: RwLock<HashMap<String, StickyEntry>>,
    /// Active connections per provider
    connections: dashmap::DashMap<String, AtomicU64>,
}

/// Metrics tracked per provider
#[derive(Debug, Clone)]
struct ProviderMetrics {
    /// Total requests
    total_requests: u64,
    /// Successful requests
    successful_requests: u64,
    /// Failed requests
    failed_requests: u64,
    /// Total latency (for average calculation)
    total_latency_ms: u64,
    /// Last updated
    last_updated: Instant,
}

impl Default for ProviderMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_latency_ms: 0,
            last_updated: Instant::now(),
        }
    }
}

impl ProviderMetrics {
    fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            1.0
        } else {
            self.successful_requests as f64 / self.total_requests as f64
        }
    }

    fn avg_latency_ms(&self) -> f64 {
        if self.successful_requests == 0 {
            0.0
        } else {
            self.total_latency_ms as f64 / self.successful_requests as f64
        }
    }
}

/// Sticky session entry
#[derive(Debug, Clone)]
struct StickyEntry {
    provider_id: String,
    created_at: Instant,
    ttl: Duration,
}

impl StickyEntry {
    fn is_valid(&self) -> bool {
        self.created_at.elapsed() < self.ttl
    }
}

impl LoadBalancer {
    /// Create a new load balancer
    #[must_use]
    pub fn new(config: LoadBalancerConfig) -> Self {
        let strategy = StrategyFactory::create(&config.strategy);
        info!(strategy = %config.strategy, "Load balancer initialized");

        Self {
            config,
            strategy,
            stats: RwLock::new(HashMap::new()),
            sticky_map: RwLock::new(HashMap::new()),
            connections: dashmap::DashMap::new(),
        }
    }

    /// Create with a specific strategy
    #[must_use]
    pub fn with_strategy(config: LoadBalancerConfig, strategy: Box<dyn LoadBalancingStrategy>) -> Self {
        info!(strategy = %strategy.name(), "Load balancer initialized with custom strategy");

        Self {
            config,
            strategy,
            stats: RwLock::new(HashMap::new()),
            sticky_map: RwLock::new(HashMap::new()),
            connections: dashmap::DashMap::new(),
        }
    }

    /// Select a provider from candidates
    pub fn select(
        &self,
        candidates: &[ProviderCandidate],
        criteria: &SelectionCriteria,
        tenant_id: Option<&str>,
    ) -> Result<Arc<dyn LLMProvider>, GatewayError> {
        // Check sticky session first
        if self.config.sticky_sessions {
            if let Some(tenant) = tenant_id {
                if let Some(provider) = self.get_sticky_provider(tenant, candidates) {
                    debug!(tenant = %tenant, provider = %provider.id(), "Using sticky session");
                    return Ok(provider);
                }
            }
        }

        // Filter candidates
        let filtered = ProviderSelector::filter(candidates, criteria);

        if filtered.is_empty() {
            return Err(GatewayError::NoHealthyProviders {
                model: criteria.model.clone().unwrap_or_default(),
            });
        }

        // Check minimum healthy requirement
        let healthy_count = filtered
            .iter()
            .filter(|c| c.health == HealthStatus::Healthy)
            .count();

        if self.config.health_aware && healthy_count < self.config.min_healthy {
            warn!(
                healthy = healthy_count,
                required = self.config.min_healthy,
                "Not enough healthy providers"
            );
            // Continue anyway with degraded providers
        }

        // Convert to provider stats for strategy
        let provider_stats: Vec<ProviderStats> = filtered
            .iter()
            .map(|c| {
                let stats = self.stats.read();
                let metrics = stats.get(&c.id);

                ProviderStats {
                    id: c.id.clone(),
                    weight: c.weight,
                    active_connections: self
                        .connections
                        .get(&c.id)
                        .map_or(0, |c| c.load(Ordering::Relaxed)),
                    avg_latency_ms: metrics.map_or(0.0, ProviderMetrics::avg_latency_ms),
                    success_rate: metrics.map_or(1.0, ProviderMetrics::success_rate),
                    is_healthy: c.health.should_route(),
                }
            })
            .collect();

        // Use strategy to select
        let selected_idx = self
            .strategy
            .select(&provider_stats)
            .ok_or_else(|| GatewayError::NoHealthyProviders {
                model: criteria.model.clone().unwrap_or_default(),
            })?;

        let selected = &filtered[selected_idx];
        let provider = Arc::clone(&selected.provider);

        // Record sticky session
        if self.config.sticky_sessions {
            if let Some(tenant) = tenant_id {
                self.set_sticky_provider(tenant, &selected.id);
            }
        }

        // Increment connection count
        self.connections
            .entry(selected.id.clone())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        debug!(
            provider = %selected.id,
            strategy = %self.strategy.name(),
            "Provider selected"
        );

        Ok(provider)
    }

    /// Record completion of a request
    pub fn record_completion(
        &self,
        provider_id: &str,
        latency: Duration,
        success: bool,
    ) {
        // Update internal stats
        {
            let mut stats = self.stats.write();
            let metrics = stats.entry(provider_id.to_string()).or_default();

            metrics.total_requests += 1;
            if success {
                metrics.successful_requests += 1;
                metrics.total_latency_ms += latency.as_millis() as u64;
            } else {
                metrics.failed_requests += 1;
            }
            metrics.last_updated = Instant::now();
        }

        // Notify strategy
        if let Some(idx) = self.get_provider_index(provider_id) {
            self.strategy.record_completion(idx, latency, success);
        }

        // Decrement connection count
        if let Some(count) = self.connections.get(provider_id) {
            count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Get provider statistics
    #[must_use]
    pub fn get_stats(&self, provider_id: &str) -> Option<LoadBalancerStats> {
        let stats = self.stats.read();
        stats.get(provider_id).map(|m| LoadBalancerStats {
            total_requests: m.total_requests,
            successful_requests: m.successful_requests,
            failed_requests: m.failed_requests,
            success_rate: m.success_rate(),
            avg_latency_ms: m.avg_latency_ms(),
            active_connections: self
                .connections
                .get(provider_id)
                .map_or(0, |c| c.load(Ordering::Relaxed)),
        })
    }

    /// Get all provider statistics
    #[must_use]
    pub fn get_all_stats(&self) -> HashMap<String, LoadBalancerStats> {
        let stats = self.stats.read();
        stats
            .iter()
            .map(|(id, m)| {
                (
                    id.clone(),
                    LoadBalancerStats {
                        total_requests: m.total_requests,
                        successful_requests: m.successful_requests,
                        failed_requests: m.failed_requests,
                        success_rate: m.success_rate(),
                        avg_latency_ms: m.avg_latency_ms(),
                        active_connections: self
                            .connections
                            .get(id)
                            .map_or(0, |c| c.load(Ordering::Relaxed)),
                    },
                )
            })
            .collect()
    }

    /// Clear sticky sessions
    pub fn clear_sticky_sessions(&self) {
        self.sticky_map.write().clear();
    }

    /// Clear all statistics
    pub fn clear_stats(&self) {
        self.stats.write().clear();
    }

    fn get_sticky_provider(
        &self,
        tenant_id: &str,
        candidates: &[ProviderCandidate],
    ) -> Option<Arc<dyn LLMProvider>> {
        let sticky_map = self.sticky_map.read();
        if let Some(entry) = sticky_map.get(tenant_id) {
            if entry.is_valid() {
                // Find the provider in candidates
                return candidates
                    .iter()
                    .find(|c| c.id == entry.provider_id && c.health.should_route())
                    .map(|c| Arc::clone(&c.provider));
            }
        }
        None
    }

    fn set_sticky_provider(&self, tenant_id: &str, provider_id: &str) {
        let mut sticky_map = self.sticky_map.write();
        sticky_map.insert(
            tenant_id.to_string(),
            StickyEntry {
                provider_id: provider_id.to_string(),
                created_at: Instant::now(),
                ttl: self.config.sticky_ttl,
            },
        );
    }

    fn get_provider_index(&self, provider_id: &str) -> Option<usize> {
        let stats = self.stats.read();
        stats.keys().position(|k| k == provider_id)
    }
}

/// Public load balancer statistics
#[derive(Debug, Clone)]
pub struct LoadBalancerStats {
    /// Total requests routed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Success rate
    pub success_rate: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Current active connections
    pub active_connections: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use gateway_core::{ChatChunk, GatewayRequest, GatewayResponse, ModelInfo, ProviderCapabilities, ProviderType};
    use futures::stream::BoxStream;

    struct MockProvider {
        id: String,
        models: Vec<ModelInfo>,
    }

    impl MockProvider {
        fn new(id: &str) -> Self {
            Self {
                id: id.to_string(),
                models: vec![ModelInfo::new("test-model")],
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
            HealthStatus::Healthy
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

    fn create_candidates(count: usize) -> Vec<ProviderCandidate> {
        (0..count)
            .map(|i| {
                let provider = Arc::new(MockProvider::new(&format!("provider-{i}")));
                ProviderCandidate::new(provider).with_health(HealthStatus::Healthy)
            })
            .collect()
    }

    #[test]
    fn test_basic_selection() {
        let lb = LoadBalancer::new(LoadBalancerConfig::new());
        let candidates = create_candidates(3);
        let criteria = SelectionCriteria::new();

        let result = lb.select(&candidates, &criteria, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_round_robin() {
        let config = LoadBalancerConfig::new().with_strategy("round_robin");
        let lb = LoadBalancer::new(config);
        let candidates = create_candidates(3);
        let criteria = SelectionCriteria::new();

        // Select multiple times and verify distribution
        let mut selections = Vec::new();
        for _ in 0..6 {
            let provider = lb.select(&candidates, &criteria, None).unwrap();
            selections.push(provider.id().to_string());
            // Simulate completion to release connection
            lb.record_completion(provider.id(), Duration::from_millis(100), true);
        }

        // Should have selected each provider at least once
        assert!(selections.contains(&"provider-0".to_string()));
        assert!(selections.contains(&"provider-1".to_string()));
        assert!(selections.contains(&"provider-2".to_string()));
    }

    #[test]
    fn test_sticky_sessions() {
        let config = LoadBalancerConfig::new()
            .with_sticky_sessions(true, Duration::from_secs(60));
        let lb = LoadBalancer::new(config);
        let candidates = create_candidates(3);
        let criteria = SelectionCriteria::new();

        // First selection for tenant-1
        let first = lb.select(&candidates, &criteria, Some("tenant-1")).unwrap();
        let first_id = first.id().to_string();
        lb.record_completion(&first_id, Duration::from_millis(100), true);

        // Subsequent selections for tenant-1 should return same provider
        for _ in 0..5 {
            let selected = lb.select(&candidates, &criteria, Some("tenant-1")).unwrap();
            assert_eq!(selected.id(), first_id);
            lb.record_completion(selected.id(), Duration::from_millis(100), true);
        }
    }

    #[test]
    fn test_record_completion() {
        let lb = LoadBalancer::new(LoadBalancerConfig::new());
        let candidates = create_candidates(1);
        let criteria = SelectionCriteria::new();

        let provider = lb.select(&candidates, &criteria, None).unwrap();

        // Record some completions
        lb.record_completion(provider.id(), Duration::from_millis(100), true);
        lb.record_completion(provider.id(), Duration::from_millis(200), true);
        lb.record_completion(provider.id(), Duration::from_millis(300), false);

        let stats = lb.get_stats(provider.id()).unwrap();
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.successful_requests, 2);
        assert_eq!(stats.failed_requests, 1);
        assert!((stats.success_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_no_available_provider() {
        let lb = LoadBalancer::new(LoadBalancerConfig::new());
        let candidates: Vec<ProviderCandidate> = vec![];
        let criteria = SelectionCriteria::new();

        let result = lb.select(&candidates, &criteria, None);
        assert!(matches!(result, Err(GatewayError::NoHealthyProviders { .. })));
    }

    #[test]
    fn test_filter_unhealthy() {
        let config = LoadBalancerConfig::new().with_health_aware(true);
        let lb = LoadBalancer::new(config);

        let provider1 = Arc::new(MockProvider::new("healthy"));
        let provider2 = Arc::new(MockProvider::new("unhealthy"));

        let candidates = vec![
            ProviderCandidate::new(provider1).with_health(HealthStatus::Healthy),
            ProviderCandidate::new(provider2).with_health(HealthStatus::Unhealthy),
        ];

        let criteria = SelectionCriteria::new().with_min_health(HealthStatus::Healthy);
        let provider = lb.select(&candidates, &criteria, None).unwrap();

        assert_eq!(provider.id(), "healthy");
    }
}
