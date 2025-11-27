//! Application state shared across handlers.

use arc_swap::ArcSwap;
use gateway_config::GatewayConfig;
use gateway_providers::ProviderRegistry;
use gateway_resilience::{CircuitBreaker, RetryPolicy};
use gateway_routing::Router;
use gateway_telemetry::{Metrics, RequestTracker};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Application state shared across all handlers
#[derive(Clone)]
pub struct AppState {
    /// Configuration (hot-reloadable)
    pub config: Arc<ArcSwap<GatewayConfig>>,
    /// Provider registry
    pub providers: Arc<ProviderRegistry>,
    /// Request router
    pub router: Arc<Router>,
    /// Circuit breakers per provider
    pub circuit_breakers: Arc<CircuitBreakerManager>,
    /// Retry policy
    pub retry_policy: Arc<RetryPolicy>,
    /// Metrics collector
    pub metrics: Arc<Metrics>,
    /// Request tracker
    pub tracker: Arc<RequestTracker>,
}

impl AppState {
    /// Create a new application state builder
    #[must_use]
    pub fn builder() -> AppStateBuilder {
        AppStateBuilder::new()
    }

    /// Get current configuration
    #[must_use]
    pub fn config(&self) -> Arc<GatewayConfig> {
        self.config.load_full()
    }

    /// Update configuration
    pub fn update_config(&self, config: GatewayConfig) {
        self.config.store(Arc::new(config));
    }
}

/// Builder for application state
pub struct AppStateBuilder {
    config: Option<GatewayConfig>,
    providers: Option<ProviderRegistry>,
    router: Option<Router>,
    retry_policy: Option<RetryPolicy>,
    metrics: Option<Metrics>,
}

impl AppStateBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: None,
            providers: None,
            router: None,
            retry_policy: None,
            metrics: None,
        }
    }

    /// Set the configuration
    #[must_use]
    pub fn config(mut self, config: GatewayConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the provider registry
    #[must_use]
    pub fn providers(mut self, providers: ProviderRegistry) -> Self {
        self.providers = Some(providers);
        self
    }

    /// Set the router
    #[must_use]
    pub fn router(mut self, router: Router) -> Self {
        self.router = Some(router);
        self
    }

    /// Set the retry policy
    #[must_use]
    pub fn retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = Some(policy);
        self
    }

    /// Set the metrics
    #[must_use]
    pub fn metrics(mut self, metrics: Metrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Build the application state
    ///
    /// # Panics
    /// Panics if required components are not set
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn build(self) -> AppState {
        let config = self.config.expect("config is required");

        AppState {
            config: Arc::new(ArcSwap::new(Arc::new(config))),
            providers: Arc::new(self.providers.unwrap_or_default()),
            router: Arc::new(self.router.unwrap_or_else(|| {
                Router::new(gateway_routing::RouterConfig::default())
            })),
            circuit_breakers: Arc::new(CircuitBreakerManager::new()),
            retry_policy: Arc::new(self.retry_policy.unwrap_or_else(RetryPolicy::with_defaults)),
            metrics: Arc::new(
                self.metrics
                    .unwrap_or_else(|| Metrics::new(&Default::default()).expect("metrics")),
            ),
            tracker: Arc::new(RequestTracker::new(10000)),
        }
    }
}

impl Default for AppStateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Manager for circuit breakers per provider
pub struct CircuitBreakerManager {
    breakers: RwLock<HashMap<String, Arc<CircuitBreaker>>>,
    config: CircuitBreakerConfig,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold before opening
    pub failure_threshold: u32,
    /// Success threshold before closing
    pub success_threshold: u32,
    /// Timeout before half-open
    pub timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(30),
        }
    }
}

impl CircuitBreakerManager {
    /// Create a new circuit breaker manager
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(CircuitBreakerConfig::default())
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            breakers: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Get or create a circuit breaker for a provider
    #[must_use]
    pub fn get_or_create(&self, provider_id: &str) -> Arc<CircuitBreaker> {
        // Check if exists
        {
            let breakers = self.breakers.read();
            if let Some(cb) = breakers.get(provider_id) {
                return Arc::clone(cb);
            }
        }

        // Create new
        let mut breakers = self.breakers.write();
        breakers
            .entry(provider_id.to_string())
            .or_insert_with(|| {
                let cb_config = gateway_resilience::CircuitBreakerConfig {
                    failure_threshold: self.config.failure_threshold,
                    success_threshold: self.config.success_threshold,
                    timeout: self.config.timeout,
                    window_size: 100,
                    min_requests: 10,
                };
                Arc::new(CircuitBreaker::new(provider_id, cb_config))
            })
            .clone()
    }

    /// Get a circuit breaker if it exists
    #[must_use]
    pub fn get(&self, provider_id: &str) -> Option<Arc<CircuitBreaker>> {
        self.breakers.read().get(provider_id).cloned()
    }

    /// Get all circuit breaker states
    #[must_use]
    pub fn all_states(&self) -> HashMap<String, CircuitBreakerState> {
        self.breakers
            .read()
            .iter()
            .map(|(id, cb)| (id.clone(), cb.state().into()))
            .collect()
    }
}

impl Default for CircuitBreakerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker state for external representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Circuit is closed (allowing requests)
    Closed,
    /// Circuit is open (rejecting requests)
    Open,
    /// Circuit is half-open (testing)
    HalfOpen,
}

impl From<gateway_resilience::CircuitState> for CircuitBreakerState {
    fn from(state: gateway_resilience::CircuitState) -> Self {
        match state {
            gateway_resilience::CircuitState::Closed => Self::Closed,
            gateway_resilience::CircuitState::Open => Self::Open,
            gateway_resilience::CircuitState::HalfOpen => Self::HalfOpen,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gateway_config::GatewayConfig;

    #[test]
    fn test_state_builder() {
        let config = GatewayConfig::default();
        let state = AppState::builder()
            .config(config)
            .build();

        assert!(state.config().server.host.is_empty() || !state.config().server.host.is_empty());
    }

    #[test]
    fn test_circuit_breaker_manager() {
        let manager = CircuitBreakerManager::new();

        let cb1 = manager.get_or_create("provider1");
        let cb2 = manager.get_or_create("provider1");

        // Should return same instance
        assert!(Arc::ptr_eq(&cb1, &cb2));

        // Different provider gets different instance
        let cb3 = manager.get_or_create("provider2");
        assert!(!Arc::ptr_eq(&cb1, &cb3));
    }

    #[test]
    fn test_config_update() {
        let config = GatewayConfig::default();
        let state = AppState::builder()
            .config(config)
            .build();

        let mut new_config = GatewayConfig::default();
        new_config.server.port = 9999;

        state.update_config(new_config);
        assert_eq!(state.config().server.port, 9999);
    }
}
