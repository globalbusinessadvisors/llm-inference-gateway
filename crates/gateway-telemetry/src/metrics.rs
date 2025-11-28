//! Prometheus metrics for the gateway.
//!
//! Provides metrics for:
//! - Request counts and latencies
//! - Token usage
//! - Provider health and availability
//! - Error rates

use parking_lot::RwLock;
use prometheus::{
    CounterVec, Encoder, GaugeVec, HistogramOpts, HistogramVec, Opts,
    Registry, TextEncoder,
};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, error, info};

/// Metrics configuration
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    /// Metrics path for HTTP endpoint
    pub path: String,
    /// Histogram buckets for latency
    pub latency_buckets: Vec<f64>,
    /// Custom labels to add to all metrics
    pub labels: HashMap<String, String>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "/metrics".to_string(),
            latency_buckets: vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
            ],
            labels: HashMap::new(),
        }
    }
}

/// Metrics for a single request
#[derive(Debug, Clone)]
pub struct RequestMetrics {
    /// Request model
    pub model: String,
    /// Provider used
    pub provider: String,
    /// Request latency
    pub latency: Duration,
    /// Whether the request was successful
    pub success: bool,
    /// HTTP status code
    pub status_code: u16,
    /// Input tokens
    pub input_tokens: Option<u32>,
    /// Output tokens
    pub output_tokens: Option<u32>,
    /// Whether it was a streaming request
    pub streaming: bool,
    /// Tenant ID if available
    pub tenant_id: Option<String>,
}

/// Main metrics registry and collectors
pub struct Metrics {
    /// Prometheus registry
    registry: Registry,
    /// Request counter
    requests_total: CounterVec,
    /// Request latency histogram
    request_latency: HistogramVec,
    /// Token usage counter
    tokens_total: CounterVec,
    /// Active requests gauge
    active_requests: GaugeVec,
    /// Provider health gauge
    provider_health: GaugeVec,
    /// Error counter by type
    errors_total: CounterVec,
    /// Circuit breaker state gauge
    circuit_breaker_state: GaugeVec,
    /// Rate limit hits counter
    rate_limit_hits: CounterVec,
    /// Cache hits/misses
    cache_operations: CounterVec,
    /// Time to first token histogram (streaming)
    ttft: HistogramVec,
    /// Tokens per second gauge
    tokens_per_second: GaugeVec,
    /// Internal state
    state: RwLock<MetricsState>,
}

#[derive(Default)]
struct MetricsState {
    active_counts: HashMap<String, i64>,
}

impl Metrics {
    /// Create a new metrics instance
    ///
    /// # Errors
    /// Returns error if metrics cannot be registered
    pub fn new(config: &MetricsConfig) -> Result<Self, prometheus::Error> {
        let registry = Registry::new();

        // Request counter
        let requests_total = CounterVec::new(
            Opts::new("llm_gateway_requests_total", "Total number of requests")
                .namespace("llm_gateway"),
            &["model", "provider", "status", "streaming"],
        )?;
        registry.register(Box::new(requests_total.clone()))?;

        // Request latency
        let request_latency = HistogramVec::new(
            HistogramOpts::new(
                "llm_gateway_request_duration_seconds",
                "Request latency in seconds",
            )
            .namespace("llm_gateway")
            .buckets(config.latency_buckets.clone()),
            &["model", "provider", "streaming"],
        )?;
        registry.register(Box::new(request_latency.clone()))?;

        // Token usage
        let tokens_total = CounterVec::new(
            Opts::new("llm_gateway_tokens_total", "Total tokens processed")
                .namespace("llm_gateway"),
            &["model", "provider", "type"],
        )?;
        registry.register(Box::new(tokens_total.clone()))?;

        // Active requests
        let active_requests = GaugeVec::new(
            Opts::new("llm_gateway_active_requests", "Number of active requests")
                .namespace("llm_gateway"),
            &["provider"],
        )?;
        registry.register(Box::new(active_requests.clone()))?;

        // Provider health
        let provider_health = GaugeVec::new(
            Opts::new("llm_gateway_provider_health", "Provider health status")
                .namespace("llm_gateway"),
            &["provider"],
        )?;
        registry.register(Box::new(provider_health.clone()))?;

        // Errors
        let errors_total = CounterVec::new(
            Opts::new("llm_gateway_errors_total", "Total number of errors")
                .namespace("llm_gateway"),
            &["provider", "error_type"],
        )?;
        registry.register(Box::new(errors_total.clone()))?;

        // Circuit breaker state
        let circuit_breaker_state = GaugeVec::new(
            Opts::new(
                "llm_gateway_circuit_breaker_state",
                "Circuit breaker state (0=closed, 1=open, 2=half-open)",
            )
            .namespace("llm_gateway"),
            &["provider"],
        )?;
        registry.register(Box::new(circuit_breaker_state.clone()))?;

        // Rate limiting
        let rate_limit_hits = CounterVec::new(
            Opts::new(
                "llm_gateway_rate_limit_hits_total",
                "Number of rate limit hits",
            )
            .namespace("llm_gateway"),
            &["tenant", "limit_type"],
        )?;
        registry.register(Box::new(rate_limit_hits.clone()))?;

        // Cache operations
        let cache_operations = CounterVec::new(
            Opts::new("llm_gateway_cache_operations_total", "Cache operations")
                .namespace("llm_gateway"),
            &["operation", "result"],
        )?;
        registry.register(Box::new(cache_operations.clone()))?;

        // Time to first token
        let ttft = HistogramVec::new(
            HistogramOpts::new(
                "llm_gateway_ttft_seconds",
                "Time to first token in seconds",
            )
            .namespace("llm_gateway")
            .buckets(vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
            &["model", "provider"],
        )?;
        registry.register(Box::new(ttft.clone()))?;

        // Tokens per second
        let tokens_per_second = GaugeVec::new(
            Opts::new(
                "llm_gateway_tokens_per_second",
                "Token generation rate",
            )
            .namespace("llm_gateway"),
            &["model", "provider"],
        )?;
        registry.register(Box::new(tokens_per_second.clone()))?;

        info!("Metrics initialized");

        Ok(Self {
            registry,
            requests_total,
            request_latency,
            tokens_total,
            active_requests,
            provider_health,
            errors_total,
            circuit_breaker_state,
            rate_limit_hits,
            cache_operations,
            ttft,
            tokens_per_second,
            state: RwLock::new(MetricsState::default()),
        })
    }

    /// Record a completed request
    pub fn record_request(&self, metrics: &RequestMetrics) {
        let status = if metrics.success { "success" } else { "error" };
        let streaming = if metrics.streaming { "true" } else { "false" };
        let model = metrics.model.as_str();
        let provider = metrics.provider.as_str();

        // Increment request counter
        self.requests_total
            .with_label_values(&[model, provider, status, streaming])
            .inc();

        // Record latency
        self.request_latency
            .with_label_values(&[model, provider, streaming])
            .observe(metrics.latency.as_secs_f64());

        // Record tokens
        if let Some(input) = metrics.input_tokens {
            self.tokens_total
                .with_label_values(&[model, provider, "input"])
                .inc_by(f64::from(input));
        }
        if let Some(output) = metrics.output_tokens {
            self.tokens_total
                .with_label_values(&[model, provider, "output"])
                .inc_by(f64::from(output));
        }

        debug!(
            model = %metrics.model,
            provider = %metrics.provider,
            latency_ms = metrics.latency.as_millis(),
            success = metrics.success,
            "Request metrics recorded"
        );
    }

    /// Record request start (increment active count)
    pub fn record_request_start(&self, provider: &str) {
        self.active_requests.with_label_values(&[provider]).inc();

        let mut state = self.state.write();
        *state.active_counts.entry(provider.to_string()).or_default() += 1;
    }

    /// Record request end (decrement active count)
    pub fn record_request_end(&self, provider: &str) {
        self.active_requests.with_label_values(&[provider]).dec();

        let mut state = self.state.write();
        if let Some(count) = state.active_counts.get_mut(provider) {
            *count = (*count - 1).max(0);
        }
    }

    /// Update provider health status
    pub fn update_provider_health(&self, provider: &str, healthy: bool) {
        let value = if healthy { 1.0 } else { 0.0 };
        self.provider_health
            .with_label_values(&[provider])
            .set(value);
    }

    /// Record an error
    pub fn record_error(&self, provider: &str, error_type: &str) {
        self.errors_total
            .with_label_values(&[provider, error_type])
            .inc();
    }

    /// Update circuit breaker state
    pub fn update_circuit_breaker(&self, provider: &str, state: CircuitBreakerState) {
        let value = match state {
            CircuitBreakerState::Closed => 0.0,
            CircuitBreakerState::Open => 1.0,
            CircuitBreakerState::HalfOpen => 2.0,
        };
        self.circuit_breaker_state
            .with_label_values(&[provider])
            .set(value);
    }

    /// Record a rate limit hit
    pub fn record_rate_limit_hit(&self, tenant: &str, limit_type: &str) {
        self.rate_limit_hits
            .with_label_values(&[tenant, limit_type])
            .inc();
    }

    /// Record cache operation
    pub fn record_cache_operation(&self, operation: &str, hit: bool) {
        let result = if hit { "hit" } else { "miss" };
        self.cache_operations
            .with_label_values(&[operation, result])
            .inc();
    }

    /// Record time to first token
    pub fn record_ttft(&self, model: &str, provider: &str, ttft: Duration) {
        self.ttft
            .with_label_values(&[model, provider])
            .observe(ttft.as_secs_f64());
    }

    /// Update tokens per second
    pub fn update_tokens_per_second(&self, model: &str, provider: &str, rate: f64) {
        self.tokens_per_second
            .with_label_values(&[model, provider])
            .set(rate);
    }

    /// Get metrics as Prometheus text format
    #[must_use]
    pub fn gather(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();

        let mut buffer = Vec::new();
        if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
            error!(error = %e, "Failed to encode metrics");
            return String::new();
        }

        String::from_utf8(buffer).unwrap_or_default()
    }

    /// Get the Prometheus registry
    #[must_use]
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

/// Circuit breaker state for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Circuit is closed (normal operation)
    Closed,
    /// Circuit is open (failing fast)
    Open,
    /// Circuit is half-open (testing)
    HalfOpen,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let config = MetricsConfig::default();
        let metrics = Metrics::new(&config);
        assert!(metrics.is_ok());
    }

    #[test]
    fn test_record_request() {
        let config = MetricsConfig::default();
        let metrics = Metrics::new(&config).unwrap();

        let request_metrics = RequestMetrics {
            model: "gpt-4".to_string(),
            provider: "openai".to_string(),
            latency: Duration::from_millis(500),
            success: true,
            status_code: 200,
            input_tokens: Some(100),
            output_tokens: Some(50),
            streaming: false,
            tenant_id: None,
        };

        metrics.record_request(&request_metrics);

        let output = metrics.gather();
        assert!(output.contains("llm_gateway_requests_total"));
        assert!(output.contains("gpt-4"));
    }

    #[test]
    fn test_active_requests() {
        let config = MetricsConfig::default();
        let metrics = Metrics::new(&config).unwrap();

        metrics.record_request_start("openai");
        metrics.record_request_start("openai");

        let output = metrics.gather();
        assert!(output.contains("llm_gateway_active_requests"));

        metrics.record_request_end("openai");
        metrics.record_request_end("openai");
    }

    #[test]
    fn test_provider_health() {
        let config = MetricsConfig::default();
        let metrics = Metrics::new(&config).unwrap();

        metrics.update_provider_health("openai", true);
        metrics.update_provider_health("anthropic", false);

        let output = metrics.gather();
        assert!(output.contains("llm_gateway_provider_health"));
    }

    #[test]
    fn test_circuit_breaker_state() {
        let config = MetricsConfig::default();
        let metrics = Metrics::new(&config).unwrap();

        metrics.update_circuit_breaker("openai", CircuitBreakerState::Closed);
        metrics.update_circuit_breaker("anthropic", CircuitBreakerState::Open);

        let output = metrics.gather();
        assert!(output.contains("llm_gateway_circuit_breaker_state"));
    }

    #[test]
    fn test_gather_output() {
        let config = MetricsConfig::default();
        let metrics = Metrics::new(&config).unwrap();

        // Record at least one metric to ensure output is not empty
        let request_metrics = RequestMetrics {
            model: "gpt-4".to_string(),
            provider: "openai".to_string(),
            latency: std::time::Duration::from_millis(100),
            success: true,
            status_code: 200,
            input_tokens: Some(10),
            output_tokens: Some(20),
            streaming: false,
            tenant_id: None,
        };
        metrics.record_request(&request_metrics);

        let output = metrics.gather();

        // Should be valid Prometheus format
        assert!(!output.is_empty());
        assert!(output.contains("# HELP"));
        assert!(output.contains("# TYPE"));
    }
}
