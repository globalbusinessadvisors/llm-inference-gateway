//! HTTP server implementation.

use crate::{routes::create_router, state::AppState};
use std::net::SocketAddr;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::signal;
use tracing::{error, info};

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Host to bind to
    pub host: String,
    /// Port to listen on
    pub port: u16,
    /// Request timeout
    pub request_timeout: Duration,
    /// Keep-alive timeout
    pub keep_alive_timeout: Duration,
    /// Maximum request body size
    pub max_body_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            request_timeout: Duration::from_secs(120),
            keep_alive_timeout: Duration::from_secs(60),
            max_body_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

impl ServerConfig {
    /// Create a new server configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the host
    #[must_use]
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set the port
    #[must_use]
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set the request timeout
    #[must_use]
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Get the socket address
    ///
    /// # Panics
    /// Panics if the host and port cannot be parsed into a valid socket address
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn socket_addr(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port)
            .parse()
            .expect("valid socket address")
    }
}

/// HTTP server for the gateway
pub struct Server {
    config: ServerConfig,
    state: AppState,
}

impl Server {
    /// Create a new server
    #[must_use]
    pub fn new(config: ServerConfig, state: AppState) -> Self {
        Self { config, state }
    }

    /// Run the server
    ///
    /// # Errors
    /// Returns error if the server fails to start or encounters a fatal error
    pub async fn run(self) -> Result<(), ServerError> {
        let addr = self.config.socket_addr();
        let router = create_router(self.state);

        info!(
            host = %self.config.host,
            port = self.config.port,
            "Starting HTTP server"
        );

        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| ServerError::Bind(e.to_string()))?;

        info!(address = %addr, "Server listening");

        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| ServerError::Serve(e.to_string()))?;

        info!("Server shutdown complete");

        Ok(())
    }

    /// Run the server with a custom shutdown signal
    ///
    /// # Errors
    /// Returns error if the server fails to start
    pub async fn run_until<F>(self, shutdown: F) -> Result<(), ServerError>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let addr = self.config.socket_addr();
        let router = create_router(self.state);

        info!(
            host = %self.config.host,
            port = self.config.port,
            "Starting HTTP server"
        );

        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| ServerError::Bind(e.to_string()))?;

        info!(address = %addr, "Server listening");

        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown)
            .await
            .map_err(|e| ServerError::Serve(e.to_string()))?;

        info!("Server shutdown complete");

        Ok(())
    }
}

/// Server error type
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// Failed to bind to address
    #[error("Failed to bind to address: {0}")]
    Bind(String),
    /// Server error during operation
    #[error("Server error: {0}")]
    Serve(String),
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
}

/// Shutdown signal handler
///
/// # Panics
/// Panics if signal handlers cannot be installed (should not happen on supported platforms)
#[allow(clippy::expect_used)]
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            info!("Received Ctrl+C, starting graceful shutdown");
        }
        () = terminate => {
            info!("Received SIGTERM, starting graceful shutdown");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
    }

    #[test]
    fn test_server_config_builder() {
        let config = ServerConfig::new()
            .with_host("127.0.0.1")
            .with_port(9000)
            .with_request_timeout(Duration::from_secs(60));

        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 9000);
        assert_eq!(config.request_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_socket_addr() {
        let config = ServerConfig::new()
            .with_host("127.0.0.1")
            .with_port(8080);

        let addr = config.socket_addr();
        assert_eq!(addr.to_string(), "127.0.0.1:8080");
    }
}
