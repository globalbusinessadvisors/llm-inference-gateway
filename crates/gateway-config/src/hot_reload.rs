//! Hot reload support for configuration changes.
//!
//! This module provides file watching and automatic configuration reload
//! when configuration files change.

use crate::loader::{ConfigError, ConfigLoader};
use crate::schema::GatewayConfig;
use arc_swap::ArcSwap;
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Configuration watcher for hot reload
pub struct ConfigWatcher {
    /// Current configuration (atomic swappable)
    config: Arc<ArcSwap<GatewayConfig>>,
    /// File watcher
    _watcher: Option<RecommendedWatcher>,
    /// Shutdown signal
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl ConfigWatcher {
    /// Create a new config watcher with the given initial configuration
    #[must_use]
    pub fn new(config: GatewayConfig) -> Self {
        Self {
            config: Arc::new(ArcSwap::from_pointee(config)),
            _watcher: None,
            shutdown_tx: None,
        }
    }

    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> Arc<ArcSwap<GatewayConfig>> {
        Arc::clone(&self.config)
    }

    /// Get a snapshot of the current configuration
    #[must_use]
    pub fn load(&self) -> Arc<GatewayConfig> {
        self.config.load_full()
    }

    /// Update the configuration
    pub fn update(&self, new_config: GatewayConfig) {
        self.config.store(Arc::new(new_config));
        info!("Configuration updated");
    }

    /// Start watching a configuration file for changes
    ///
    /// # Errors
    /// Returns error if file watching cannot be started
    #[allow(clippy::unused_async)] // Async used for spawned task
    pub async fn watch_file(&mut self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let path = path.as_ref().to_path_buf();
        let config = Arc::clone(&self.config);

        // Create channel for file events
        let (tx, mut rx) = mpsc::channel::<PathBuf>(10);
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

        // Set up file watcher
        let tx_clone = tx;
        let path_clone = path.clone();

        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    if event.kind.is_modify() || event.kind.is_create() {
                        debug!("Configuration file changed: {:?}", event.paths);
                        if let Err(e) = tx_clone.blocking_send(path_clone.clone()) {
                            warn!("Failed to send file change event: {}", e);
                        }
                    }
                }
                Err(e) => {
                    error!("File watch error: {}", e);
                }
            }
        })
        .map_err(|e| ConfigError::Io(std::io::Error::other(e)))?;

        // Watch the config file's parent directory
        let watch_path = path.parent().unwrap_or(&path);
        watcher
            .watch(watch_path, RecursiveMode::NonRecursive)
            .map_err(|e| ConfigError::Io(std::io::Error::other(e)))?;

        info!("Watching configuration file: {}", path.display());

        // Spawn reload task
        let path_for_task = path.clone();
        tokio::spawn(async move {
            let mut debounce_timer: Option<tokio::time::Instant> = None;
            let debounce_duration = Duration::from_millis(500);

            loop {
                tokio::select! {
                    Some(changed_path) = rx.recv() => {
                        // Debounce rapid changes
                        let now = tokio::time::Instant::now();
                        if let Some(last) = debounce_timer {
                            if now.duration_since(last) < debounce_duration {
                                continue;
                            }
                        }
                        debounce_timer = Some(now);

                        info!("Reloading configuration from: {}", changed_path.display());
                        match Self::reload_config(&path_for_task).await {
                            Ok(new_config) => {
                                config.store(Arc::new(new_config));
                                info!("Configuration reloaded successfully");
                            }
                            Err(e) => {
                                error!("Failed to reload configuration: {}", e);
                            }
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Configuration watcher shutting down");
                        break;
                    }
                }
            }
        });

        self._watcher = Some(watcher);
        self.shutdown_tx = Some(shutdown_tx);

        Ok(())
    }

    /// Reload configuration from file
    async fn reload_config(path: &Path) -> Result<GatewayConfig, ConfigError> {
        ConfigLoader::new()
            .with_file(path.to_string_lossy().to_string())
            .with_env_prefix("LLM_GATEWAY")
            .load()
            .await
    }

    /// Stop watching for configuration changes
    pub async fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }
        self._watcher = None;
    }

    /// Manually trigger a configuration reload
    ///
    /// # Errors
    /// Returns error if reload fails
    pub async fn reload(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let new_config = Self::reload_config(path.as_ref()).await?;
        self.config.store(Arc::new(new_config));
        info!("Configuration manually reloaded");
        Ok(())
    }
}

impl Drop for ConfigWatcher {
    fn drop(&mut self) {
        // Watcher cleanup happens automatically
        debug!("ConfigWatcher dropped");
    }
}

/// Create a shared configuration that can be used across the application
pub fn shared_config(config: GatewayConfig) -> Arc<ArcSwap<GatewayConfig>> {
    Arc::new(ArcSwap::from_pointee(config))
}

/// Helper trait for accessing configuration from Arc<ArcSwap<T>>
pub trait ConfigAccess {
    /// Get a snapshot of the current configuration
    fn snapshot(&self) -> Arc<GatewayConfig>;
}

impl ConfigAccess for Arc<ArcSwap<GatewayConfig>> {
    fn snapshot(&self) -> Arc<GatewayConfig> {
        self.load_full()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use tokio::fs;
    use tokio::time::sleep;

    #[test]
    fn test_config_watcher_new() {
        let config = GatewayConfig::default();
        let watcher = ConfigWatcher::new(config);

        let loaded = watcher.load();
        assert_eq!(loaded.server.port, 8080);
    }

    #[test]
    fn test_config_update() {
        let config = GatewayConfig::default();
        let watcher = ConfigWatcher::new(config);

        let mut new_config = GatewayConfig::default();
        new_config.server.port = 9090;
        watcher.update(new_config);

        let loaded = watcher.load();
        assert_eq!(loaded.server.port, 9090);
    }

    #[tokio::test]
    async fn test_shared_config() {
        let config = GatewayConfig::default();
        let shared = shared_config(config);

        // Clone for "another thread"
        let shared_clone = Arc::clone(&shared);

        // Update original
        let mut new_config = GatewayConfig::default();
        new_config.server.port = 3000;
        shared.store(Arc::new(new_config));

        // Clone should see the update
        let snapshot = shared_clone.snapshot();
        assert_eq!(snapshot.server.port, 3000);
    }

    #[tokio::test]
    async fn test_manual_reload() {
        use tempfile::Builder;
        // Create temp config file with .yaml extension
        let temp_file = Builder::new()
            .suffix(".yaml")
            .tempfile()
            .expect("create temp file");

        let config_content = r#"
server:
  port: 8080
"#;
        fs::write(temp_file.path(), config_content)
            .await
            .expect("write config");

        let config = GatewayConfig::default();
        let watcher = ConfigWatcher::new(config);

        // Update file
        let new_content = r#"
server:
  port: 9999
"#;
        fs::write(temp_file.path(), new_content)
            .await
            .expect("write config");

        // Manual reload
        watcher.reload(temp_file.path()).await.expect("reload");

        let loaded = watcher.load();
        assert_eq!(loaded.server.port, 9999);
    }
}
