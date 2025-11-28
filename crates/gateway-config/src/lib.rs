//! # Gateway Configuration
//!
//! Configuration management for the LLM Inference Gateway, including:
//! - Configuration schema and validation
//! - Loading from YAML/TOML files
//! - Hot reload support via file watching
//! - Environment variable substitution

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod loader;
pub mod schema;
pub mod hot_reload;

// Re-export main types
pub use loader::{load_config, ConfigError, ConfigLoader, ConfigSource};
pub use schema::{
    GatewayConfig, ServerConfig, ProviderConfig, RoutingConfig,
    ResilienceConfig, ObservabilityConfig, SecurityConfig,
    CircuitBreakerConfig, RetryConfig, RateLimitConfig, RateLimitKeyBy,
    AuthConfig, TlsConfig,
};
pub use hot_reload::ConfigWatcher;
