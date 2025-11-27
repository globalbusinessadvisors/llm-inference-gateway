//! Configuration loading from files and environment.
//!
//! This module provides configuration loading from YAML and TOML files,
//! with support for environment variable substitution.

use crate::schema::GatewayConfig;
use std::path::Path;
use thiserror::Error;
use tokio::fs;
use tracing::{debug, info, warn};

/// Configuration loading errors
#[derive(Debug, Error)]
pub enum ConfigError {
    /// File not found
    #[error("Configuration file not found: {path}")]
    FileNotFound {
        /// The path to the file that was not found
        path: String,
    },

    /// IO error
    #[error("IO error reading configuration: {0}")]
    Io(#[from] std::io::Error),

    /// YAML parsing error
    #[error("YAML parsing error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// TOML parsing error
    #[error("TOML parsing error: {0}")]
    Toml(#[from] toml::de::Error),

    /// JSON parsing error
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    /// Validation error
    #[error("Configuration validation error: {0}")]
    Validation(String),

    /// Unsupported format
    #[error("Unsupported configuration format: {extension}")]
    UnsupportedFormat {
        /// The file extension that was not supported
        extension: String,
    },

    /// Environment variable not found
    #[error("Environment variable not found: {name}")]
    EnvVarNotFound {
        /// The name of the environment variable that was not found
        name: String,
    },
}

/// Configuration source
#[derive(Debug, Clone)]
pub enum ConfigSource {
    /// File path
    File(String),
    /// Raw YAML string
    Yaml(String),
    /// Raw TOML string
    Toml(String),
    /// Raw JSON string
    Json(String),
    /// Default configuration
    Default,
}

/// Configuration loader
pub struct ConfigLoader {
    sources: Vec<ConfigSource>,
    env_prefix: Option<String>,
}

impl ConfigLoader {
    /// Create a new config loader
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            env_prefix: None,
        }
    }

    /// Add a configuration source
    #[must_use]
    pub fn with_source(mut self, source: ConfigSource) -> Self {
        self.sources.push(source);
        self
    }

    /// Add a file source
    #[must_use]
    pub fn with_file(self, path: impl Into<String>) -> Self {
        self.with_source(ConfigSource::File(path.into()))
    }

    /// Set environment variable prefix for overrides
    #[must_use]
    pub fn with_env_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.env_prefix = Some(prefix.into());
        self
    }

    /// Load configuration from all sources
    ///
    /// # Errors
    /// Returns error if any source fails to load or validate
    pub async fn load(self) -> Result<GatewayConfig, ConfigError> {
        let mut config = GatewayConfig::default();

        // Load from each source in order
        for source in self.sources {
            let source_config = Self::load_source(&source).await?;
            config = Self::merge_configs(config, source_config);
        }

        // Apply environment variable overrides
        if let Some(ref prefix) = self.env_prefix {
            config = Self::apply_env_overrides(config, prefix)?;
        }

        // Validate final configuration
        config
            .validate_config()
            .map_err(|e| ConfigError::Validation(format!("{e:?}")))?;

        info!("Configuration loaded successfully");
        Ok(config)
    }

    /// Load from a single source
    async fn load_source(source: &ConfigSource) -> Result<GatewayConfig, ConfigError> {
        match source {
            ConfigSource::File(path) => Self::load_file(path).await,
            ConfigSource::Yaml(content) => Self::parse_yaml(content),
            ConfigSource::Toml(content) => Self::parse_toml(content),
            ConfigSource::Json(content) => Self::parse_json(content),
            ConfigSource::Default => Ok(GatewayConfig::default()),
        }
    }

    /// Load configuration from a file
    async fn load_file(path: &str) -> Result<GatewayConfig, ConfigError> {
        let path = Path::new(path);

        if !path.exists() {
            return Err(ConfigError::FileNotFound {
                path: path.display().to_string(),
            });
        }

        let content = fs::read_to_string(path).await?;
        let content = Self::substitute_env_vars(&content)?;

        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        debug!("Loading configuration from {} (format: {})", path.display(), extension);

        match extension.as_str() {
            "yaml" | "yml" => Self::parse_yaml(&content),
            "toml" => Self::parse_toml(&content),
            "json" => Self::parse_json(&content),
            ext => Err(ConfigError::UnsupportedFormat {
                extension: ext.to_string(),
            }),
        }
    }

    /// Parse YAML content
    fn parse_yaml(content: &str) -> Result<GatewayConfig, ConfigError> {
        Ok(serde_yaml::from_str(content)?)
    }

    /// Parse TOML content
    fn parse_toml(content: &str) -> Result<GatewayConfig, ConfigError> {
        Ok(toml::from_str(content)?)
    }

    /// Parse JSON content
    fn parse_json(content: &str) -> Result<GatewayConfig, ConfigError> {
        Ok(serde_json::from_str(content)?)
    }

    /// Substitute environment variables in content
    ///
    /// Supports ${VAR} and ${VAR:-default} syntax
    ///
    /// # Panics
    /// Panics if the regex is invalid (should not happen with static patterns)
    #[allow(clippy::expect_used)]
    fn substitute_env_vars(content: &str) -> Result<String, ConfigError> {
        let re = regex::Regex::new(r"\$\{([^}]+)\}").expect("valid regex");
        let mut result = content.to_string();

        for cap in re.captures_iter(content) {
            let full_match = cap.get(0).expect("match exists").as_str();
            let var_spec = cap.get(1).expect("group exists").as_str();

            // Handle ${VAR:-default} syntax
            let (var_name, default) = if let Some(idx) = var_spec.find(":-") {
                (&var_spec[..idx], Some(&var_spec[idx + 2..]))
            } else {
                (var_spec, None)
            };

            match std::env::var(var_name) {
                Ok(value) => {
                    result = result.replace(full_match, &value);
                }
                Err(_) => {
                    if let Some(default_val) = default {
                        result = result.replace(full_match, default_val);
                    } else {
                        warn!("Environment variable not found: {}", var_name);
                    }
                }
            }
        }

        // Don't fail on missing env vars - they might be optional
        // Just warn about them
        Ok(result)
    }

    /// Merge two configurations (later overrides earlier)
    fn merge_configs(base: GatewayConfig, overlay: GatewayConfig) -> GatewayConfig {
        // For now, we do a simple overlay where non-default values override
        // In production, this would be more sophisticated
        GatewayConfig {
            server: if overlay.server.port == ServerConfig::default().port {
                base.server
            } else {
                overlay.server
            },
            providers: if overlay.providers.is_empty() {
                base.providers
            } else {
                overlay.providers
            },
            routing: overlay.routing,
            resilience: overlay.resilience,
            observability: overlay.observability,
            security: overlay.security,
        }
    }

    /// Apply environment variable overrides
    fn apply_env_overrides(
        mut config: GatewayConfig,
        prefix: &str,
    ) -> Result<GatewayConfig, ConfigError> {
        // Server overrides
        if let Ok(port) = std::env::var(format!("{prefix}_SERVER_PORT")) {
            if let Ok(port) = port.parse() {
                config.server.port = port;
            }
        }

        if let Ok(host) = std::env::var(format!("{prefix}_SERVER_HOST")) {
            config.server.host = host;
        }

        // Log level override
        if let Ok(level) = std::env::var(format!("{prefix}_LOG_LEVEL")) {
            config.observability.logging.level = level;
        }

        // Metrics enabled override
        if let Ok(enabled) = std::env::var(format!("{prefix}_METRICS_ENABLED")) {
            config.observability.metrics.enabled = enabled.parse().unwrap_or(true);
        }

        Ok(config)
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

use crate::schema::ServerConfig;

/// Load configuration from default locations
///
/// Looks for configuration in order:
/// 1. Path from CONFIG_PATH environment variable
/// 2. ./config.yaml
/// 3. ./config/default.yaml
/// 4. /etc/llm-gateway/config.yaml
///
/// # Errors
/// Returns error if no configuration is found or parsing fails
pub async fn load_config() -> Result<GatewayConfig, ConfigError> {
    let config_path = std::env::var("CONFIG_PATH").ok();

    let search_paths = if let Some(ref path) = config_path {
        vec![path.as_str()]
    } else {
        vec![
            "config.yaml",
            "config.yml",
            "config/default.yaml",
            "config/default.yml",
            "/etc/llm-gateway/config.yaml",
        ]
    };

    for path in &search_paths {
        if Path::new(path).exists() {
            info!("Loading configuration from: {}", path);
            return ConfigLoader::new()
                .with_file(*path)
                .with_env_prefix("LLM_GATEWAY")
                .load()
                .await;
        }
    }

    // No config file found, use defaults
    warn!("No configuration file found, using defaults");
    Ok(GatewayConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_var_substitution() {
        std::env::set_var("TEST_VAR", "test_value");

        let content = "key: ${TEST_VAR}";
        let result = ConfigLoader::substitute_env_vars(content).expect("substitute");
        assert_eq!(result, "key: test_value");

        std::env::remove_var("TEST_VAR");
    }

    #[test]
    fn test_env_var_with_default() {
        let content = "key: ${NONEXISTENT_VAR:-default_value}";
        let result = ConfigLoader::substitute_env_vars(content).expect("substitute");
        assert_eq!(result, "key: default_value");
    }

    #[tokio::test]
    async fn test_load_yaml_content() {
        let yaml = r#"
server:
  port: 9090
  host: "127.0.0.1"
"#;

        let config = ConfigLoader::new()
            .with_source(ConfigSource::Yaml(yaml.to_string()))
            .load()
            .await
            .expect("load config");

        assert_eq!(config.server.port, 9090);
        assert_eq!(config.server.host, "127.0.0.1");
    }

    #[tokio::test]
    async fn test_load_default_config() {
        let config = ConfigLoader::new()
            .with_source(ConfigSource::Default)
            .load()
            .await
            .expect("load config");

        assert_eq!(config.server.port, 8080);
    }

    #[tokio::test]
    async fn test_env_overrides() {
        std::env::set_var("TEST_PREFIX_SERVER_PORT", "3000");

        let config = ConfigLoader::new()
            .with_source(ConfigSource::Default)
            .with_env_prefix("TEST_PREFIX")
            .load()
            .await
            .expect("load config");

        assert_eq!(config.server.port, 3000);

        std::env::remove_var("TEST_PREFIX_SERVER_PORT");
    }
}
