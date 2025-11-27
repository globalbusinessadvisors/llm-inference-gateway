//! Timeout management for requests.
//!
//! Provides hierarchical timeout handling with connect, read, and request timeouts.

use gateway_core::GatewayError;
use std::future::Future;
use std::time::Duration;
use tracing::warn;

/// Timeout configuration
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Connection timeout
    pub connect: Duration,
    /// Read timeout
    pub read: Duration,
    /// Write timeout
    pub write: Duration,
    /// Overall request timeout
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

/// Timeout manager
#[derive(Debug, Clone)]
pub struct TimeoutManager {
    config: TimeoutConfig,
}

impl TimeoutManager {
    /// Create a new timeout manager
    #[must_use]
    pub fn new(config: TimeoutConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(TimeoutConfig::default())
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &TimeoutConfig {
        &self.config
    }

    /// Get the connect timeout
    #[must_use]
    pub fn connect_timeout(&self) -> Duration {
        self.config.connect
    }

    /// Get the read timeout
    #[must_use]
    pub fn read_timeout(&self) -> Duration {
        self.config.read
    }

    /// Get the write timeout
    #[must_use]
    pub fn write_timeout(&self) -> Duration {
        self.config.write
    }

    /// Get the request timeout
    #[must_use]
    pub fn request_timeout(&self) -> Duration {
        self.config.request
    }

    /// Execute an operation with the request timeout
    ///
    /// # Errors
    /// Returns `GatewayError::Timeout` if the operation times out
    pub async fn with_timeout<F, T>(&self, future: F) -> Result<T, GatewayError>
    where
        F: Future<Output = Result<T, GatewayError>>,
    {
        self.with_custom_timeout(future, self.config.request).await
    }

    /// Execute an operation with a custom timeout
    ///
    /// # Errors
    /// Returns `GatewayError::Timeout` if the operation times out
    pub async fn with_custom_timeout<F, T>(
        &self,
        future: F,
        timeout: Duration,
    ) -> Result<T, GatewayError>
    where
        F: Future<Output = Result<T, GatewayError>>,
    {
        if let Ok(result) = tokio::time::timeout(timeout, future).await { result } else {
            warn!(timeout_ms = timeout.as_millis(), "Request timed out");
            Err(GatewayError::timeout(timeout))
        }
    }

    /// Create a timeout manager with adjusted timeouts for streaming
    #[must_use]
    pub fn for_streaming(&self) -> Self {
        Self::new(TimeoutConfig {
            connect: self.config.connect,
            read: Duration::from_secs(300), // 5 minutes for streaming
            write: self.config.write,
            request: Duration::from_secs(600), // 10 minutes total
        })
    }

    /// Create a timeout manager with adjusted timeouts for health checks
    #[must_use]
    pub fn for_health_check(&self) -> Self {
        Self::new(TimeoutConfig {
            connect: Duration::from_secs(5),
            read: Duration::from_secs(5),
            write: Duration::from_secs(5),
            request: Duration::from_secs(10),
        })
    }
}

/// Builder for timeout configuration
#[derive(Debug, Default)]
pub struct TimeoutConfigBuilder {
    config: TimeoutConfig,
}

impl TimeoutConfigBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set connect timeout
    #[must_use]
    pub fn connect(mut self, timeout: Duration) -> Self {
        self.config.connect = timeout;
        self
    }

    /// Set read timeout
    #[must_use]
    pub fn read(mut self, timeout: Duration) -> Self {
        self.config.read = timeout;
        self
    }

    /// Set write timeout
    #[must_use]
    pub fn write(mut self, timeout: Duration) -> Self {
        self.config.write = timeout;
        self
    }

    /// Set request timeout
    #[must_use]
    pub fn request(mut self, timeout: Duration) -> Self {
        self.config.request = timeout;
        self
    }

    /// Build the configuration
    #[must_use]
    pub fn build(self) -> TimeoutConfig {
        self.config
    }
}

/// Extension trait for adding timeout to futures
#[allow(async_fn_in_trait)]
pub trait TimeoutExt: Sized {
    /// Add a timeout to this future
    ///
    /// # Errors
    /// Returns `GatewayError::Timeout` if the operation times out
    async fn with_timeout(self, timeout: Duration) -> Result<Self::Output, GatewayError>
    where
        Self: Future;
}

impl<F: Future> TimeoutExt for F {
    async fn with_timeout(self, timeout: Duration) -> Result<F::Output, GatewayError> {
        match tokio::time::timeout(timeout, self).await {
            Ok(result) => Ok(result),
            Err(_) => Err(GatewayError::timeout(timeout)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_timeout_success() {
        let tm = TimeoutManager::with_defaults();

        let result: Result<u32, GatewayError> = tm
            .with_custom_timeout(
                async {
                    sleep(Duration::from_millis(10)).await;
                    Ok(42)
                },
                Duration::from_secs(1),
            )
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_timeout_exceeded() {
        let tm = TimeoutManager::with_defaults();

        let result: Result<u32, GatewayError> = tm
            .with_custom_timeout(
                async {
                    sleep(Duration::from_secs(10)).await;
                    Ok(42)
                },
                Duration::from_millis(50),
            )
            .await;

        assert!(result.is_err());
        match result {
            Err(GatewayError::Timeout { .. }) => {}
            _ => panic!("Expected timeout error"),
        }
    }

    #[test]
    fn test_timeout_config_defaults() {
        let config = TimeoutConfig::default();
        assert_eq!(config.connect, Duration::from_secs(10));
        assert_eq!(config.read, Duration::from_secs(120));
        assert_eq!(config.write, Duration::from_secs(30));
        assert_eq!(config.request, Duration::from_secs(120));
    }

    #[test]
    fn test_timeout_for_streaming() {
        let tm = TimeoutManager::with_defaults();
        let streaming_tm = tm.for_streaming();

        assert_eq!(streaming_tm.config().read, Duration::from_secs(300));
        assert_eq!(streaming_tm.config().request, Duration::from_secs(600));
    }

    #[test]
    fn test_timeout_for_health_check() {
        let tm = TimeoutManager::with_defaults();
        let health_tm = tm.for_health_check();

        assert_eq!(health_tm.config().connect, Duration::from_secs(5));
        assert_eq!(health_tm.config().request, Duration::from_secs(10));
    }

    #[test]
    fn test_builder() {
        let config = TimeoutConfigBuilder::new()
            .connect(Duration::from_secs(5))
            .read(Duration::from_secs(60))
            .write(Duration::from_secs(15))
            .request(Duration::from_secs(90))
            .build();

        assert_eq!(config.connect, Duration::from_secs(5));
        assert_eq!(config.read, Duration::from_secs(60));
        assert_eq!(config.write, Duration::from_secs(15));
        assert_eq!(config.request, Duration::from_secs(90));
    }

    #[tokio::test]
    async fn test_timeout_ext() {
        let result = async { 42 }
            .with_timeout(Duration::from_secs(1))
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        let result = sleep(Duration::from_secs(10))
            .with_timeout(Duration::from_millis(50))
            .await;

        assert!(result.is_err());
    }
}
