//! Request tracking and lifecycle management.
//!
//! Provides unified tracking of request lifecycle:
//! - Request start/end timestamps
//! - Token counting
//! - Provider selection tracking
//! - Error categorization

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Request tracker for monitoring request lifecycle
pub struct RequestTracker {
    /// Active requests
    active: RwLock<HashMap<String, TrackedRequest>>,
    /// Completed requests (ring buffer)
    completed: RwLock<Vec<RequestOutcome>>,
    /// Maximum completed requests to keep
    max_completed: usize,
}

/// Information about an active request
#[derive(Debug, Clone)]
pub struct RequestInfo {
    /// Request ID
    pub request_id: String,
    /// Model requested
    pub model: String,
    /// Provider selected
    pub provider: Option<String>,
    /// Tenant ID
    pub tenant_id: Option<String>,
    /// Whether streaming
    pub streaming: bool,
    /// Start time
    pub started_at: DateTime<Utc>,
    /// Client IP
    pub client_ip: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
}

impl RequestInfo {
    /// Create new request info
    #[must_use]
    pub fn new(request_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            model: model.into(),
            provider: None,
            tenant_id: None,
            streaming: false,
            started_at: Utc::now(),
            client_ip: None,
            user_agent: None,
        }
    }

    /// Set provider
    #[must_use]
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Set tenant
    #[must_use]
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant.into());
        self
    }

    /// Set streaming
    #[must_use]
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Set client IP
    #[must_use]
    pub fn with_client_ip(mut self, ip: impl Into<String>) -> Self {
        self.client_ip = Some(ip.into());
        self
    }
}

/// Tracked request with timing
struct TrackedRequest {
    info: RequestInfo,
    start_instant: Instant,
    first_token_at: Option<Instant>,
    tokens_received: u32,
}

/// Outcome of a completed request
#[derive(Debug, Clone)]
pub struct RequestOutcome {
    /// Request information
    pub info: RequestInfo,
    /// Total duration
    pub duration: Duration,
    /// Time to first token (streaming only)
    pub time_to_first_token: Option<Duration>,
    /// Input tokens
    pub input_tokens: Option<u32>,
    /// Output tokens
    pub output_tokens: Option<u32>,
    /// HTTP status code
    pub status_code: u16,
    /// Whether successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Completion time
    pub completed_at: DateTime<Utc>,
}

impl RequestTracker {
    /// Create a new request tracker
    #[must_use]
    pub fn new(max_completed: usize) -> Self {
        Self {
            active: RwLock::new(HashMap::new()),
            completed: RwLock::new(Vec::with_capacity(max_completed)),
            max_completed,
        }
    }

    /// Start tracking a request
    pub fn start(&self, info: RequestInfo) {
        let request_id = info.request_id.clone();

        debug!(
            request_id = %request_id,
            model = %info.model,
            provider = ?info.provider,
            streaming = info.streaming,
            "Request started"
        );

        let tracked = TrackedRequest {
            info,
            start_instant: Instant::now(),
            first_token_at: None,
            tokens_received: 0,
        };

        self.active.write().insert(request_id, tracked);
    }

    /// Record first token received (for streaming)
    pub fn record_first_token(&self, request_id: &str) {
        if let Some(tracked) = self.active.write().get_mut(request_id) {
            if tracked.first_token_at.is_none() {
                tracked.first_token_at = Some(Instant::now());
                debug!(
                    request_id = %request_id,
                    ttft_ms = tracked.first_token_at.map(|t| t.duration_since(tracked.start_instant).as_millis()),
                    "First token received"
                );
            }
        }
    }

    /// Record tokens received (for streaming)
    pub fn record_tokens(&self, request_id: &str, count: u32) {
        if let Some(tracked) = self.active.write().get_mut(request_id) {
            tracked.tokens_received += count;
        }
    }

    /// Update provider for a request
    pub fn update_provider(&self, request_id: &str, provider: &str) {
        if let Some(tracked) = self.active.write().get_mut(request_id) {
            tracked.info.provider = Some(provider.to_string());
        }
    }

    /// Complete a request successfully
    pub fn complete_success(
        &self,
        request_id: &str,
        status_code: u16,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    ) {
        self.complete(request_id, status_code, true, None, input_tokens, output_tokens);
    }

    /// Complete a request with an error
    pub fn complete_error(&self, request_id: &str, status_code: u16, error: impl Into<String>) {
        self.complete(request_id, status_code, false, Some(error.into()), None, None);
    }

    /// Complete a request
    fn complete(
        &self,
        request_id: &str,
        status_code: u16,
        success: bool,
        error: Option<String>,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    ) {
        let tracked = self.active.write().remove(request_id);

        if let Some(tracked) = tracked {
            let duration = tracked.start_instant.elapsed();
            let time_to_first_token = tracked
                .first_token_at
                .map(|t| t.duration_since(tracked.start_instant));

            let outcome = RequestOutcome {
                info: tracked.info.clone(),
                duration,
                time_to_first_token,
                input_tokens,
                output_tokens: output_tokens.or(Some(tracked.tokens_received)),
                status_code,
                success,
                error: error.clone(),
                completed_at: Utc::now(),
            };

            if success {
                info!(
                    request_id = %request_id,
                    model = %tracked.info.model,
                    provider = ?tracked.info.provider,
                    duration_ms = duration.as_millis(),
                    input_tokens = ?input_tokens,
                    output_tokens = ?output_tokens,
                    status = status_code,
                    "Request completed successfully"
                );
            } else {
                warn!(
                    request_id = %request_id,
                    model = %tracked.info.model,
                    provider = ?tracked.info.provider,
                    duration_ms = duration.as_millis(),
                    status = status_code,
                    error = ?error,
                    "Request failed"
                );
            }

            // Store in completed ring buffer
            let mut completed = self.completed.write();
            if completed.len() >= self.max_completed {
                completed.remove(0);
            }
            completed.push(outcome);
        }
    }

    /// Get number of active requests
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.read().len()
    }

    /// Get active request info
    #[must_use]
    pub fn get_active(&self, request_id: &str) -> Option<RequestInfo> {
        self.active.read().get(request_id).map(|t| t.info.clone())
    }

    /// Get all active requests
    #[must_use]
    pub fn get_all_active(&self) -> Vec<RequestInfo> {
        self.active.read().values().map(|t| t.info.clone()).collect()
    }

    /// Get recent completed requests
    #[must_use]
    pub fn get_recent_completed(&self, limit: usize) -> Vec<RequestOutcome> {
        let completed = self.completed.read();
        completed.iter().rev().take(limit).cloned().collect()
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> TrackerStats {
        let completed = self.completed.read();

        let total = completed.len();
        let successful = completed.iter().filter(|r| r.success).count();
        let failed = total - successful;

        let avg_duration = if total > 0 {
            let sum: Duration = completed.iter().map(|r| r.duration).sum();
            sum / total as u32
        } else {
            Duration::ZERO
        };

        let streaming_requests: Vec<_> = completed
            .iter()
            .filter(|r| r.info.streaming && r.time_to_first_token.is_some())
            .collect();

        let avg_ttft = if streaming_requests.is_empty() {
            None
        } else {
            let sum: Duration = streaming_requests
                .iter()
                .filter_map(|r| r.time_to_first_token)
                .sum();
            Some(sum / streaming_requests.len() as u32)
        };

        let total_input_tokens: u32 = completed
            .iter()
            .filter_map(|r| r.input_tokens)
            .sum();

        let total_output_tokens: u32 = completed
            .iter()
            .filter_map(|r| r.output_tokens)
            .sum();

        TrackerStats {
            active_requests: self.active.read().len(),
            total_completed: total,
            successful,
            failed,
            success_rate: if total > 0 {
                successful as f64 / total as f64
            } else {
                1.0
            },
            avg_duration,
            avg_ttft,
            total_input_tokens,
            total_output_tokens,
        }
    }

    /// Clear completed requests
    pub fn clear_completed(&self) {
        self.completed.write().clear();
    }
}

/// Tracker statistics
#[derive(Debug, Clone)]
pub struct TrackerStats {
    /// Number of active requests
    pub active_requests: usize,
    /// Total completed requests (in buffer)
    pub total_completed: usize,
    /// Successful requests
    pub successful: usize,
    /// Failed requests
    pub failed: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average request duration
    pub avg_duration: Duration,
    /// Average time to first token (streaming)
    pub avg_ttft: Option<Duration>,
    /// Total input tokens processed
    pub total_input_tokens: u32,
    /// Total output tokens generated
    pub total_output_tokens: u32,
}

impl Default for RequestTracker {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_lifecycle() {
        let tracker = RequestTracker::new(100);

        let info = RequestInfo::new("req-1", "gpt-4")
            .with_provider("openai")
            .with_streaming(false);

        tracker.start(info);
        assert_eq!(tracker.active_count(), 1);

        tracker.complete_success("req-1", 200, Some(100), Some(50));
        assert_eq!(tracker.active_count(), 0);

        let completed = tracker.get_recent_completed(10);
        assert_eq!(completed.len(), 1);
        assert!(completed[0].success);
    }

    #[test]
    fn test_streaming_tracking() {
        let tracker = RequestTracker::new(100);

        let info = RequestInfo::new("req-1", "gpt-4")
            .with_streaming(true);

        tracker.start(info);

        // Simulate streaming
        std::thread::sleep(Duration::from_millis(10));
        tracker.record_first_token("req-1");
        tracker.record_tokens("req-1", 10);
        tracker.record_tokens("req-1", 20);

        tracker.complete_success("req-1", 200, Some(50), None);

        let completed = tracker.get_recent_completed(1);
        assert_eq!(completed.len(), 1);
        assert!(completed[0].time_to_first_token.is_some());
        assert_eq!(completed[0].output_tokens, Some(30));
    }

    #[test]
    fn test_error_tracking() {
        let tracker = RequestTracker::new(100);

        let info = RequestInfo::new("req-1", "gpt-4");
        tracker.start(info);

        tracker.complete_error("req-1", 500, "Internal error");

        let completed = tracker.get_recent_completed(1);
        assert!(!completed[0].success);
        assert_eq!(completed[0].error, Some("Internal error".to_string()));
    }

    #[test]
    fn test_ring_buffer() {
        let tracker = RequestTracker::new(3);

        for i in 0..5 {
            let info = RequestInfo::new(format!("req-{i}"), "gpt-4");
            tracker.start(info);
            tracker.complete_success(&format!("req-{i}"), 200, None, None);
        }

        let completed = tracker.get_recent_completed(10);
        assert_eq!(completed.len(), 3);
        // Should have most recent
        assert_eq!(completed[0].info.request_id, "req-4");
    }

    #[test]
    fn test_stats() {
        let tracker = RequestTracker::new(100);

        // Add some successful requests
        for i in 0..3 {
            let info = RequestInfo::new(format!("req-{i}"), "gpt-4");
            tracker.start(info);
            tracker.complete_success(&format!("req-{i}"), 200, Some(100), Some(50));
        }

        // Add a failed request
        let info = RequestInfo::new("req-fail", "gpt-4");
        tracker.start(info);
        tracker.complete_error("req-fail", 500, "Error");

        let stats = tracker.stats();
        assert_eq!(stats.total_completed, 4);
        assert_eq!(stats.successful, 3);
        assert_eq!(stats.failed, 1);
        assert!((stats.success_rate - 0.75).abs() < 0.01);
        assert_eq!(stats.total_input_tokens, 300);
        assert_eq!(stats.total_output_tokens, 150);
    }

    #[test]
    fn test_request_info_builder() {
        let info = RequestInfo::new("req-1", "gpt-4")
            .with_provider("openai")
            .with_tenant("tenant-1")
            .with_streaming(true)
            .with_client_ip("127.0.0.1");

        assert_eq!(info.request_id, "req-1");
        assert_eq!(info.model, "gpt-4");
        assert_eq!(info.provider, Some("openai".to_string()));
        assert_eq!(info.tenant_id, Some("tenant-1".to_string()));
        assert!(info.streaming);
        assert_eq!(info.client_ip, Some("127.0.0.1".to_string()));
    }
}
