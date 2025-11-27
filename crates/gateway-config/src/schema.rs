//! Configuration schema definitions.
//!
//! This module defines all configuration types with validation and defaults.

use gateway_core::ProviderType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use validator::Validate;

/// Main gateway configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
#[derive(Default)]
pub struct GatewayConfig {
    /// Server configuration
    #[validate(nested)]
    pub server: ServerConfig,

    /// Provider configurations
    #[validate(nested)]
    pub providers: Vec<ProviderConfig>,

    /// Routing configuration
    #[validate(nested)]
    pub routing: RoutingConfig,

    /// Resilience configuration
    #[validate(nested)]
    pub resilience: ResilienceConfig,

    /// Observability configuration
    #[validate(nested)]
    pub observability: ObservabilityConfig,

    /// Security configuration
    #[validate(nested)]
    pub security: SecurityConfig,
}


impl GatewayConfig {
    /// Validate the configuration
    ///
    /// # Errors
    /// Returns validation errors if configuration is invalid
    pub fn validate_config(&self) -> Result<(), validator::ValidationErrors> {
        self.validate()
    }

    /// Get a provider config by ID
    #[must_use]
    pub fn get_provider(&self, id: &str) -> Option<&ProviderConfig> {
        self.providers.iter().find(|p| p.id == id)
    }

    /// Get all enabled providers
    #[must_use]
    pub fn enabled_providers(&self) -> Vec<&ProviderConfig> {
        self.providers.iter().filter(|p| p.enabled).collect()
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
pub struct ServerConfig {
    /// Bind host
    #[validate(length(min = 1))]
    pub host: String,

    /// Bind port
    #[validate(range(min = 1, max = 65535))]
    pub port: u16,

    /// Number of worker threads (0 = auto-detect)
    pub workers: usize,

    /// Request timeout
    #[serde(with = "humantime_serde")]
    pub request_timeout: Duration,

    /// Graceful shutdown timeout
    #[serde(with = "humantime_serde")]
    pub graceful_shutdown_timeout: Duration,

    /// Keep-alive timeout
    #[serde(with = "humantime_serde")]
    pub keep_alive_timeout: Duration,

    /// Maximum request body size in bytes
    pub max_request_body_size: usize,

    /// Enable HTTP/2
    pub http2: bool,

    /// TLS configuration (optional)
    #[validate(nested)]
    pub tls: Option<TlsConfig>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            workers: 0,
            request_timeout: Duration::from_secs(120),
            graceful_shutdown_timeout: Duration::from_secs(30),
            keep_alive_timeout: Duration::from_secs(60),
            max_request_body_size: 10 * 1024 * 1024, // 10MB
            http2: true,
            tls: None,
        }
    }
}

impl ServerConfig {
    /// Get the socket address
    #[must_use]
    pub fn socket_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TlsConfig {
    /// Path to certificate file
    pub cert_path: PathBuf,

    /// Path to private key file
    pub key_path: PathBuf,

    /// Minimum TLS version (default: 1.2)
    #[serde(default = "default_min_tls_version")]
    pub min_version: String,
}

fn default_min_tls_version() -> String {
    "1.2".to_string()
}

/// Provider configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ProviderConfig {
    /// Unique provider instance ID
    #[validate(length(min = 1, max = 64))]
    pub id: String,

    /// Provider type
    #[serde(rename = "type")]
    pub provider_type: ProviderType,

    /// Base URL/endpoint
    #[validate(url)]
    pub endpoint: String,

    /// API key (can be env var reference like ${OPENAI_API_KEY})
    #[serde(default)]
    pub api_key: Option<String>,

    /// API key environment variable name
    #[serde(default)]
    pub api_key_env: Option<String>,

    /// Supported models
    #[serde(default)]
    pub models: Vec<String>,

    /// Request timeout for this provider
    #[serde(default, with = "humantime_serde")]
    pub timeout: Option<Duration>,

    /// Rate limit configuration
    #[serde(default)]
    #[validate(nested)]
    pub rate_limit: Option<ProviderRateLimitConfig>,

    /// Whether this provider is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Priority for routing (lower = higher priority)
    #[serde(default = "default_priority")]
    pub priority: u32,

    /// Weight for weighted load balancing
    #[serde(default = "default_weight")]
    pub weight: u32,

    /// Custom headers to include in requests
    #[serde(default)]
    pub headers: HashMap<String, String>,

    /// Provider-specific options
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

fn default_true() -> bool {
    true
}

fn default_priority() -> u32 {
    100
}

fn default_weight() -> u32 {
    100
}

impl ProviderConfig {
    /// Resolve the API key from config or environment
    #[must_use]
    pub fn resolve_api_key(&self) -> Option<String> {
        // First try direct api_key
        if let Some(ref key) = self.api_key {
            // Check if it's an env var reference
            if key.starts_with("${") && key.ends_with('}') {
                let env_var = &key[2..key.len() - 1];
                return std::env::var(env_var).ok();
            }
            return Some(key.clone());
        }

        // Then try api_key_env
        if let Some(ref env_var) = self.api_key_env {
            return std::env::var(env_var).ok();
        }

        None
    }
}

/// Provider-specific rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
#[derive(Default)]
pub struct ProviderRateLimitConfig {
    /// Requests per minute
    pub requests_per_minute: Option<u32>,

    /// Tokens per minute
    pub tokens_per_minute: Option<u32>,

    /// Concurrent request limit
    pub max_concurrent: Option<u32>,
}


/// Routing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
pub struct RoutingConfig {
    /// Default load balancing strategy
    #[serde(default = "default_strategy")]
    pub default_strategy: LoadBalancingStrategy,

    /// Routing rules
    #[serde(default)]
    pub rules: Vec<RoutingRule>,

    /// Model to provider mappings
    #[serde(default)]
    pub model_mappings: HashMap<String, Vec<String>>,

    /// Enable health-aware routing
    #[serde(default = "default_true")]
    pub health_aware: bool,
}

fn default_strategy() -> LoadBalancingStrategy {
    LoadBalancingStrategy::RoundRobin
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            default_strategy: LoadBalancingStrategy::RoundRobin,
            rules: Vec::new(),
            model_mappings: HashMap::new(),
            health_aware: true,
        }
    }
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    #[default]
    RoundRobin,
    /// Route to least latency provider
    LeastLatency,
    /// Route to lowest cost provider
    CostOptimized,
    /// Weighted random distribution
    Weighted,
    /// Random selection
    Random,
    /// Always use primary, failover on error
    PrimaryBackup,
}


/// Routing rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    /// Rule name
    pub name: String,

    /// Match conditions
    pub match_conditions: MatchConditions,

    /// Target provider IDs
    pub route_to: Vec<String>,

    /// Rule priority (lower = higher priority)
    #[serde(default = "default_priority")]
    pub priority: u32,

    /// Load balancing strategy for this rule
    #[serde(default)]
    pub strategy: Option<LoadBalancingStrategy>,

    /// Whether rule is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
}

/// Match conditions for routing rules
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MatchConditions {
    /// Match model pattern (glob)
    #[serde(default)]
    pub model: Option<String>,

    /// Match tenant ID
    #[serde(default)]
    pub tenant_id: Option<String>,

    /// Match tags
    #[serde(default)]
    pub tags: HashMap<String, String>,

    /// Match priority threshold
    #[serde(default)]
    pub min_priority: Option<u8>,

    /// Match header values
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

/// Resilience configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
#[derive(Default)]
pub struct ResilienceConfig {
    /// Circuit breaker configuration
    #[validate(nested)]
    pub circuit_breaker: CircuitBreakerConfig,

    /// Retry configuration
    #[validate(nested)]
    pub retry: RetryConfig,

    /// Bulkhead configuration
    #[validate(nested)]
    pub bulkhead: BulkheadConfig,

    /// Timeout configuration
    #[validate(nested)]
    pub timeout: TimeoutConfig,
}


/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
pub struct CircuitBreakerConfig {
    /// Whether circuit breaker is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Number of failures to trip the circuit
    #[validate(range(min = 1, max = 100))]
    pub failure_threshold: u32,

    /// Number of successes to close the circuit
    #[validate(range(min = 1, max = 100))]
    pub success_threshold: u32,

    /// Time to wait before testing the circuit
    #[serde(with = "humantime_serde")]
    pub timeout: Duration,

    /// Sliding window size for failure rate calculation
    #[validate(range(min = 1, max = 1000))]
    pub window_size: u32,

    /// Minimum requests before failure rate is calculated
    #[validate(range(min = 1, max = 100))]
    pub min_requests: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            window_size: 100,
            min_requests: 10,
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
pub struct RetryConfig {
    /// Whether retry is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Maximum number of retries
    #[validate(range(min = 0, max = 10))]
    pub max_retries: u32,

    /// Base delay between retries
    #[serde(with = "humantime_serde")]
    pub base_delay: Duration,

    /// Maximum delay between retries
    #[serde(with = "humantime_serde")]
    pub max_delay: Duration,

    /// Backoff multiplier
    pub multiplier: f64,

    /// Jitter factor (0.0 - 1.0)
    #[validate(range(min = 0.0, max = 1.0))]
    pub jitter: f64,

    /// HTTP status codes to retry on
    #[serde(default = "default_retry_codes")]
    pub retry_on_status: Vec<u16>,
}

fn default_retry_codes() -> Vec<u16> {
    vec![429, 500, 502, 503, 504]
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
            jitter: 0.25,
            retry_on_status: default_retry_codes(),
        }
    }
}

/// Bulkhead configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
pub struct BulkheadConfig {
    /// Whether bulkhead is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Maximum concurrent requests per provider
    #[validate(range(min = 1, max = 10000))]
    pub max_concurrent: u32,

    /// Queue size when max concurrent is reached
    #[validate(range(min = 0, max = 10000))]
    pub queue_size: u32,

    /// Queue timeout
    #[serde(with = "humantime_serde")]
    pub queue_timeout: Duration,
}

impl Default for BulkheadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent: 100,
            queue_size: 100,
            queue_timeout: Duration::from_secs(10),
        }
    }
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
pub struct TimeoutConfig {
    /// Connection timeout
    #[serde(with = "humantime_serde")]
    pub connect: Duration,

    /// Read timeout
    #[serde(with = "humantime_serde")]
    pub read: Duration,

    /// Write timeout
    #[serde(with = "humantime_serde")]
    pub write: Duration,

    /// Overall request timeout
    #[serde(with = "humantime_serde")]
    pub request: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(10),
            read: Duration::from_secs(120),
            write: Duration::from_secs(30),
            request: Duration::from_secs(120),
        }
    }
}

/// Observability configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
#[derive(Default)]
pub struct ObservabilityConfig {
    /// Metrics configuration
    #[validate(nested)]
    pub metrics: MetricsConfig,

    /// Tracing configuration
    #[validate(nested)]
    pub tracing: TracingConfig,

    /// Logging configuration
    #[validate(nested)]
    pub logging: LoggingConfig,
}


/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
pub struct MetricsConfig {
    /// Whether metrics are enabled
    pub enabled: bool,

    /// Metrics endpoint path
    pub endpoint: String,

    /// Include histograms (more memory, more detail)
    pub histograms: bool,

    /// Histogram buckets for latency
    #[serde(default = "default_latency_buckets")]
    pub latency_buckets: Vec<f64>,
}

fn default_latency_buckets() -> Vec<f64> {
    vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "/metrics".to_string(),
            histograms: true,
            latency_buckets: default_latency_buckets(),
        }
    }
}

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
pub struct TracingConfig {
    /// Whether tracing is enabled
    pub enabled: bool,

    /// OTLP endpoint
    pub endpoint: Option<String>,

    /// Sample rate (0.0 - 1.0)
    #[validate(range(min = 0.0, max = 1.0))]
    pub sample_rate: f64,

    /// Service name
    pub service_name: String,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: None,
            sample_rate: 0.1,
            service_name: "llm-inference-gateway".to_string(),
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,

    /// Log format (json, pretty)
    pub format: LogFormat,

    /// Include request/response bodies in logs
    pub log_bodies: bool,

    /// Redact sensitive fields in logs
    pub redact_sensitive: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: LogFormat::Json,
            log_bodies: false,
            redact_sensitive: true,
        }
    }
}

/// Log format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum LogFormat {
    /// JSON formatted logs
    #[default]
    Json,
    /// Human-readable pretty logs
    Pretty,
    /// Compact single-line logs
    Compact,
}


/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
#[derive(Default)]
pub struct SecurityConfig {
    /// Authentication configuration
    #[validate(nested)]
    pub auth: AuthConfig,

    /// Rate limiting configuration
    #[validate(nested)]
    pub rate_limiting: RateLimitConfig,

    /// CORS configuration
    pub cors: CorsConfig,
}


/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
pub struct AuthConfig {
    /// Whether authentication is enabled
    pub enabled: bool,

    /// Authentication methods
    pub methods: Vec<AuthMethod>,

    /// API keys (for development/testing)
    #[serde(default)]
    pub api_keys: Vec<ApiKeyConfig>,

    /// JWT configuration
    #[serde(default)]
    pub jwt: Option<JwtConfig>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            methods: vec![AuthMethod::ApiKey],
            api_keys: Vec::new(),
            jwt: None,
        }
    }
}

/// Authentication method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    /// API key authentication
    ApiKey,
    /// JWT bearer token
    Jwt,
    /// No authentication
    None,
}

/// API key configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    /// Key identifier
    pub id: String,

    /// The API key value (or env var reference)
    pub key: String,

    /// Description
    #[serde(default)]
    pub description: Option<String>,

    /// Associated tenant ID
    #[serde(default)]
    pub tenant_id: Option<String>,

    /// Allowed roles
    #[serde(default)]
    pub roles: Vec<String>,

    /// Whether key is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
}

/// JWT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtConfig {
    /// JWT secret or public key
    pub secret: String,

    /// JWT issuer
    #[serde(default)]
    pub issuer: Option<String>,

    /// JWT audience
    #[serde(default)]
    pub audience: Option<String>,

    /// Algorithm (HS256, RS256, etc.)
    #[serde(default = "default_jwt_algorithm")]
    pub algorithm: String,
}

fn default_jwt_algorithm() -> String {
    "HS256".to_string()
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
pub struct RateLimitConfig {
    /// Whether rate limiting is enabled
    pub enabled: bool,

    /// Default requests per minute
    #[validate(range(min = 1))]
    pub default_rpm: u32,

    /// Default tokens per minute
    #[serde(default)]
    pub default_tpm: Option<u32>,

    /// Window size
    #[serde(with = "humantime_serde")]
    pub window: Duration,

    /// Rate limit key (ip, api_key, tenant)
    pub key_by: RateLimitKeyBy,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_rpm: 1000,
            default_tpm: None,
            window: Duration::from_secs(60),
            key_by: RateLimitKeyBy::ApiKey,
        }
    }
}

/// Rate limit key type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum RateLimitKeyBy {
    /// Rate limit by IP address
    Ip,
    /// Rate limit by API key
    #[default]
    ApiKey,
    /// Rate limit by tenant ID
    Tenant,
}


/// CORS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CorsConfig {
    /// Whether CORS is enabled
    pub enabled: bool,

    /// Allowed origins
    pub allowed_origins: Vec<String>,

    /// Allowed methods
    pub allowed_methods: Vec<String>,

    /// Allowed headers
    pub allowed_headers: Vec<String>,

    /// Max age for preflight cache
    pub max_age: u64,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "PUT".to_string(),
                "DELETE".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec!["*".to_string()],
            max_age: 86400,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GatewayConfig::default();
        assert_eq!(config.server.port, 8080);
        assert!(config.providers.is_empty());
        assert!(config.server.http2);
    }

    #[test]
    fn test_server_socket_addr() {
        let server = ServerConfig::default();
        assert_eq!(server.socket_addr(), "0.0.0.0:8080");
    }

    #[test]
    fn test_provider_config_resolve_api_key() {
        std::env::set_var("TEST_API_KEY", "test-key-123");

        let config = ProviderConfig {
            id: "test".to_string(),
            provider_type: ProviderType::OpenAI,
            endpoint: "https://api.openai.com".to_string(),
            api_key: Some("${TEST_API_KEY}".to_string()),
            api_key_env: None,
            models: vec![],
            timeout: None,
            rate_limit: None,
            enabled: true,
            priority: 100,
            weight: 100,
            headers: HashMap::new(),
            options: HashMap::new(),
        };

        assert_eq!(config.resolve_api_key(), Some("test-key-123".to_string()));

        std::env::remove_var("TEST_API_KEY");
    }

    #[test]
    fn test_circuit_breaker_config_defaults() {
        let config = CircuitBreakerConfig::default();
        assert!(config.enabled);
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.success_threshold, 3);
        assert_eq!(config.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_retry_config_defaults() {
        let config = RetryConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_retries, 3);
        assert!(config.retry_on_status.contains(&429));
        assert!(config.retry_on_status.contains(&503));
    }

    #[test]
    fn test_yaml_serialization() {
        let config = GatewayConfig::default();
        let yaml = serde_yaml::to_string(&config).expect("serialize");
        assert!(yaml.contains("server:"));
        assert!(yaml.contains("port: 8080"));
    }

    #[test]
    fn test_load_balancing_strategy() {
        let strategy: LoadBalancingStrategy =
            serde_yaml::from_str("round_robin").expect("deserialize");
        assert_eq!(strategy, LoadBalancingStrategy::RoundRobin);

        let strategy: LoadBalancingStrategy =
            serde_yaml::from_str("least_latency").expect("deserialize");
        assert_eq!(strategy, LoadBalancingStrategy::LeastLatency);
    }
}
