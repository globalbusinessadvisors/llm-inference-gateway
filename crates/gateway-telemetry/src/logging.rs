//! Structured logging configuration.
//!
//! Provides configurable logging with:
//! - JSON or pretty format
//! - Log level filtering
//! - Request context enrichment

use serde::{Deserialize, Serialize};
use tracing::Level;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer,
};

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Enable logging
    pub enabled: bool,
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Output format (json or pretty)
    pub format: LogFormat,
    /// Include timestamps
    pub timestamps: bool,
    /// Include source location
    pub include_location: bool,
    /// Include span events
    pub span_events: SpanEvents,
    /// Filter directives (e.g., "hyper=warn,tower=info")
    pub filter: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: "info".to_string(),
            format: LogFormat::Pretty,
            timestamps: true,
            include_location: true,
            span_events: SpanEvents::None,
            filter: None,
        }
    }
}

impl LoggingConfig {
    /// Create a new logging configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the log level
    #[must_use]
    pub fn with_level(mut self, level: impl Into<String>) -> Self {
        self.level = level.into();
        self
    }

    /// Set the output format
    #[must_use]
    pub fn with_format(mut self, format: LogFormat) -> Self {
        self.format = format;
        self
    }

    /// Enable JSON format
    #[must_use]
    pub fn json(mut self) -> Self {
        self.format = LogFormat::Json;
        self
    }

    /// Enable pretty format
    #[must_use]
    pub fn pretty(mut self) -> Self {
        self.format = LogFormat::Pretty;
        self
    }

    /// Set span events
    #[must_use]
    pub fn with_span_events(mut self, events: SpanEvents) -> Self {
        self.span_events = events;
        self
    }

    /// Set filter directives
    #[must_use]
    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    /// Get the tracing Level
    #[must_use]
    pub fn tracing_level(&self) -> Level {
        match self.level.to_lowercase().as_str() {
            "trace" => Level::TRACE,
            "debug" => Level::DEBUG,
            "info" => Level::INFO,
            "warn" | "warning" => Level::WARN,
            "error" => Level::ERROR,
            _ => Level::INFO,
        }
    }
}

/// Log output format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    /// JSON format (structured)
    Json,
    /// Pretty format (human-readable)
    #[default]
    Pretty,
    /// Compact format
    Compact,
}

/// Span event configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SpanEvents {
    /// No span events
    #[default]
    None,
    /// Log when spans are entered
    Enter,
    /// Log when spans are exited
    Exit,
    /// Log both enter and exit
    Full,
    /// Log when spans are created and closed
    Lifecycle,
}

impl SpanEvents {
    fn to_fmt_span(self) -> FmtSpan {
        match self {
            Self::None => FmtSpan::NONE,
            Self::Enter => FmtSpan::ENTER,
            Self::Exit => FmtSpan::EXIT,
            Self::Full => FmtSpan::ENTER | FmtSpan::EXIT,
            Self::Lifecycle => FmtSpan::NEW | FmtSpan::CLOSE,
        }
    }
}

/// Initialize logging with the given configuration
///
/// # Errors
/// Returns error if logging cannot be initialized
pub fn init_logging(config: &LoggingConfig) -> Result<(), LoggingError> {
    if !config.enabled {
        return Ok(());
    }

    // Build the env filter
    let filter = build_filter(config)?;

    // Initialize based on format
    match config.format {
        LogFormat::Json => init_json_logging(config, filter),
        LogFormat::Pretty => init_pretty_logging(config, filter),
        LogFormat::Compact => init_compact_logging(config, filter),
    }
}

fn build_filter(config: &LoggingConfig) -> Result<EnvFilter, LoggingError> {
    // Start with RUST_LOG env var or default level
    let base_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.level));

    // Add custom filter directives if specified
    if let Some(ref filter_str) = config.filter {
        EnvFilter::try_new(format!("{},{}", config.level, filter_str))
            .map_err(|e| LoggingError::FilterParse(e.to_string()))
    } else {
        Ok(base_filter)
    }
}

fn init_json_logging(config: &LoggingConfig, filter: EnvFilter) -> Result<(), LoggingError> {
    let layer = fmt::layer()
        .json()
        .with_span_events(config.span_events.to_fmt_span())
        .with_file(config.include_location)
        .with_line_number(config.include_location)
        .with_target(true)
        .with_thread_ids(true);

    // Note: timestamps are always included in JSON format
    let _ = config.timestamps; // Silence unused warning

    tracing_subscriber::registry()
        .with(layer.with_filter(filter))
        .try_init()
        .map_err(|e| LoggingError::Init(e.to_string()))
}

fn init_pretty_logging(config: &LoggingConfig, filter: EnvFilter) -> Result<(), LoggingError> {
    let layer = fmt::layer()
        .pretty()
        .with_span_events(config.span_events.to_fmt_span())
        .with_file(config.include_location)
        .with_line_number(config.include_location)
        .with_target(true)
        .with_thread_ids(false);

    let layer = if config.timestamps {
        layer.boxed()
    } else {
        layer.without_time().boxed()
    };

    tracing_subscriber::registry()
        .with(layer.with_filter(filter))
        .try_init()
        .map_err(|e| LoggingError::Init(e.to_string()))
}

fn init_compact_logging(config: &LoggingConfig, filter: EnvFilter) -> Result<(), LoggingError> {
    let layer = fmt::layer()
        .compact()
        .with_span_events(config.span_events.to_fmt_span())
        .with_file(config.include_location)
        .with_line_number(config.include_location)
        .with_target(true);

    let layer = if config.timestamps {
        layer.boxed()
    } else {
        layer.without_time().boxed()
    };

    tracing_subscriber::registry()
        .with(layer.with_filter(filter))
        .try_init()
        .map_err(|e| LoggingError::Init(e.to_string()))
}

/// Logging initialization error
#[derive(Debug, thiserror::Error)]
pub enum LoggingError {
    /// Failed to initialize logging
    #[error("Failed to initialize logging: {0}")]
    Init(String),
    /// Failed to parse filter
    #[error("Failed to parse log filter: {0}")]
    FilterParse(String),
}

/// Log context for enriching log entries
#[derive(Debug, Clone, Default)]
pub struct LogContext {
    /// Request ID
    pub request_id: Option<String>,
    /// Tenant ID
    pub tenant_id: Option<String>,
    /// Provider ID
    pub provider_id: Option<String>,
    /// Model
    pub model: Option<String>,
    /// Additional fields
    pub fields: std::collections::HashMap<String, String>,
}

impl LogContext {
    /// Create a new log context
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set request ID
    #[must_use]
    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    /// Set tenant ID
    #[must_use]
    pub fn with_tenant_id(mut self, id: impl Into<String>) -> Self {
        self.tenant_id = Some(id.into());
        self
    }

    /// Set provider ID
    #[must_use]
    pub fn with_provider_id(mut self, id: impl Into<String>) -> Self {
        self.provider_id = Some(id.into());
        self
    }

    /// Set model
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Add a custom field
    #[must_use]
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = LoggingConfig::new()
            .with_level("debug")
            .json()
            .with_filter("hyper=warn");

        assert_eq!(config.level, "debug");
        assert_eq!(config.format, LogFormat::Json);
        assert_eq!(config.filter, Some("hyper=warn".to_string()));
    }

    #[test]
    fn test_tracing_level() {
        assert_eq!(LoggingConfig::new().with_level("trace").tracing_level(), Level::TRACE);
        assert_eq!(LoggingConfig::new().with_level("DEBUG").tracing_level(), Level::DEBUG);
        assert_eq!(LoggingConfig::new().with_level("Info").tracing_level(), Level::INFO);
        assert_eq!(LoggingConfig::new().with_level("WARN").tracing_level(), Level::WARN);
        assert_eq!(LoggingConfig::new().with_level("error").tracing_level(), Level::ERROR);
        assert_eq!(LoggingConfig::new().with_level("invalid").tracing_level(), Level::INFO);
    }

    #[test]
    fn test_log_context() {
        let ctx = LogContext::new()
            .with_request_id("req-123")
            .with_tenant_id("tenant-456")
            .with_provider_id("openai")
            .with_model("gpt-4")
            .with_field("custom", "value");

        assert_eq!(ctx.request_id, Some("req-123".to_string()));
        assert_eq!(ctx.tenant_id, Some("tenant-456".to_string()));
        assert_eq!(ctx.provider_id, Some("openai".to_string()));
        assert_eq!(ctx.model, Some("gpt-4".to_string()));
        assert_eq!(ctx.fields.get("custom"), Some(&"value".to_string()));
    }

    #[test]
    fn test_span_events() {
        assert_eq!(SpanEvents::None.to_fmt_span(), FmtSpan::NONE);
        assert_eq!(SpanEvents::Enter.to_fmt_span(), FmtSpan::ENTER);
        assert_eq!(SpanEvents::Exit.to_fmt_span(), FmtSpan::EXIT);
    }
}
