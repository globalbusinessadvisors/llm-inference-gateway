//! Rate limiting using the token bucket algorithm.
//!
//! Provides rate limiting for requests per minute (RPM) and tokens per minute (TPM).
//! Supports multiple keys for per-tenant, per-IP, or per-API-key rate limiting.

use gateway_core::GatewayError;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Rate limiter configuration
#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
    /// Requests per window (e.g., requests per minute)
    pub requests_per_window: u32,
    /// Tokens per window (optional, for token-based limits)
    pub tokens_per_window: Option<u32>,
    /// Window duration
    pub window: Duration,
    /// Whether to enable burst handling
    pub enable_burst: bool,
    /// Burst multiplier (e.g., 1.5 means 50% more for bursts)
    pub burst_multiplier: f32,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            requests_per_window: 1000,
            tokens_per_window: None,
            window: Duration::from_secs(60),
            enable_burst: true,
            burst_multiplier: 1.5,
        }
    }
}

/// Token bucket state for a single key
#[derive(Debug, Clone)]
struct TokenBucket {
    /// Available request tokens
    request_tokens: f64,
    /// Available token tokens (for TPM limits)
    token_tokens: Option<f64>,
    /// Last refill time
    last_refill: Instant,
    /// Configuration for this bucket
    config: RateLimiterConfig,
}

impl TokenBucket {
    fn new(config: RateLimiterConfig) -> Self {
        let burst_mult = if config.enable_burst {
            f64::from(config.burst_multiplier)
        } else {
            1.0
        };
        let request_tokens = f64::from(config.requests_per_window) * burst_mult;
        let token_tokens = config
            .tokens_per_window
            .map(|t| f64::from(t) * burst_mult);
        Self {
            request_tokens,
            token_tokens,
            last_refill: Instant::now(),
            config,
        }
    }

    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        let window_secs = self.config.window.as_secs_f64();

        if window_secs > 0.0 {
            let refill_rate = elapsed.as_secs_f64() / window_secs;

            // Refill request tokens
            let max_requests = f64::from(self.config.requests_per_window)
                * if self.config.enable_burst {
                    f64::from(self.config.burst_multiplier)
                } else {
                    1.0
                };
            self.request_tokens =
                (self.request_tokens + f64::from(self.config.requests_per_window) * refill_rate)
                    .min(max_requests);

            // Refill token tokens if configured
            if let (Some(tokens), Some(max_tokens)) =
                (&mut self.token_tokens, self.config.tokens_per_window)
            {
                let max_token_tokens = f64::from(max_tokens)
                    * if self.config.enable_burst {
                        f64::from(self.config.burst_multiplier)
                    } else {
                        1.0
                    };
                *tokens = (*tokens + f64::from(max_tokens) * refill_rate).min(max_token_tokens);
            }
        }

        self.last_refill = now;
    }

    /// Try to consume tokens for a request
    fn try_consume(&mut self, token_count: Option<u32>) -> Result<(), RateLimitExceeded> {
        self.refill();

        // Check request tokens
        if self.request_tokens < 1.0 {
            return Err(RateLimitExceeded {
                limit_type: RateLimitType::Requests,
                limit: self.config.requests_per_window,
                window: self.config.window,
                retry_after: self.estimate_retry_after(1.0, RateLimitType::Requests),
            });
        }

        // Check token tokens if applicable
        if let (Some(tokens), Some(count)) = (&self.token_tokens, token_count) {
            if *tokens < f64::from(count) {
                return Err(RateLimitExceeded {
                    limit_type: RateLimitType::Tokens,
                    limit: self.config.tokens_per_window.unwrap_or(0),
                    window: self.config.window,
                    retry_after: self
                        .estimate_retry_after(f64::from(count), RateLimitType::Tokens),
                });
            }
        }

        // Consume tokens
        self.request_tokens -= 1.0;
        if let (Some(tokens), Some(count)) = (&mut self.token_tokens, token_count) {
            *tokens -= f64::from(count);
        }

        Ok(())
    }

    /// Estimate time until enough tokens are available
    fn estimate_retry_after(&self, needed: f64, limit_type: RateLimitType) -> Duration {
        let window_secs = self.config.window.as_secs_f64();
        let (current, rate) = match limit_type {
            RateLimitType::Requests => (self.request_tokens, self.config.requests_per_window),
            RateLimitType::Tokens => (
                self.token_tokens.unwrap_or(0.0),
                self.config.tokens_per_window.unwrap_or(1),
            ),
        };

        if rate == 0 {
            return Duration::from_secs(60);
        }

        let tokens_needed = needed - current;
        if tokens_needed <= 0.0 {
            return Duration::ZERO;
        }

        let refill_rate = f64::from(rate) / window_secs;
        let secs = tokens_needed / refill_rate;
        Duration::from_secs_f64(secs.min(window_secs))
    }

    /// Get current statistics
    fn stats(&self) -> BucketStats {
        BucketStats {
            request_tokens_available: self.request_tokens,
            token_tokens_available: self.token_tokens,
            requests_per_window: self.config.requests_per_window,
            tokens_per_window: self.config.tokens_per_window,
        }
    }
}

/// Rate limit type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitType {
    /// Request count limit
    Requests,
    /// Token count limit
    Tokens,
}

/// Rate limit exceeded error details
#[derive(Debug, Clone)]
pub struct RateLimitExceeded {
    /// Type of limit exceeded
    pub limit_type: RateLimitType,
    /// The limit that was exceeded
    pub limit: u32,
    /// Window duration
    pub window: Duration,
    /// Estimated time until tokens are available
    pub retry_after: Duration,
}

/// Bucket statistics
#[derive(Debug, Clone)]
pub struct BucketStats {
    /// Available request tokens
    pub request_tokens_available: f64,
    /// Available token tokens (if configured)
    pub token_tokens_available: Option<f64>,
    /// Requests per window limit
    pub requests_per_window: u32,
    /// Tokens per window limit (if configured)
    pub tokens_per_window: Option<u32>,
}

impl BucketStats {
    /// Calculate request utilization percentage
    #[must_use]
    pub fn request_utilization(&self) -> f64 {
        if self.requests_per_window == 0 {
            return 0.0;
        }
        let used = f64::from(self.requests_per_window) - self.request_tokens_available;
        (used / f64::from(self.requests_per_window) * 100.0).max(0.0)
    }

    /// Calculate token utilization percentage
    #[must_use]
    pub fn token_utilization(&self) -> Option<f64> {
        self.tokens_per_window.map(|max| {
            if max == 0 {
                return 0.0;
            }
            let available = self.token_tokens_available.unwrap_or(0.0);
            let used = f64::from(max) - available;
            (used / f64::from(max) * 100.0).max(0.0)
        })
    }
}

/// Rate limiter with support for multiple keys
pub struct RateLimiter {
    /// Identifier
    id: String,
    /// Default configuration
    default_config: RateLimiterConfig,
    /// Per-key buckets
    buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    /// Whether rate limiting is enabled
    enabled: bool,
}

impl RateLimiter {
    /// Create a new rate limiter
    #[must_use]
    pub fn new(id: impl Into<String>, config: RateLimiterConfig) -> Self {
        Self {
            id: id.into(),
            default_config: config,
            buckets: Arc::new(RwLock::new(HashMap::new())),
            enabled: true,
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_defaults(id: impl Into<String>) -> Self {
        Self::new(id, RateLimiterConfig::default())
    }

    /// Create a disabled rate limiter (always allows requests)
    #[must_use]
    pub fn disabled(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            default_config: RateLimiterConfig::default(),
            buckets: Arc::new(RwLock::new(HashMap::new())),
            enabled: false,
        }
    }

    /// Get the rate limiter ID
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Check if rate limiting is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Check rate limit for a key
    ///
    /// # Arguments
    /// * `key` - The rate limit key (e.g., tenant ID, IP address, API key)
    /// * `token_count` - Optional token count for TPM limiting
    ///
    /// # Errors
    /// Returns error if rate limit is exceeded
    pub async fn check(&self, key: &str, token_count: Option<u32>) -> Result<(), GatewayError> {
        if !self.enabled {
            return Ok(());
        }

        let mut buckets = self.buckets.write().await;

        let bucket = buckets
            .entry(key.to_string())
            .or_insert_with(|| TokenBucket::new(self.default_config.clone()));

        match bucket.try_consume(token_count) {
            Ok(()) => {
                debug!(
                    rate_limiter = %self.id,
                    key = %key,
                    tokens_remaining = bucket.request_tokens,
                    "Rate limit check passed"
                );
                Ok(())
            }
            Err(exceeded) => {
                warn!(
                    rate_limiter = %self.id,
                    key = %key,
                    limit_type = ?exceeded.limit_type,
                    limit = exceeded.limit,
                    retry_after_ms = exceeded.retry_after.as_millis(),
                    "Rate limit exceeded"
                );
                Err(GatewayError::RateLimit {
                    retry_after: Some(exceeded.retry_after),
                    limit: Some(exceeded.limit),
                })
            }
        }
    }

    /// Acquire rate limit permit (alias for check with no token count)
    ///
    /// # Errors
    /// Returns error if rate limit is exceeded
    pub async fn acquire(&self, key: &str) -> Result<(), GatewayError> {
        self.check(key, None).await
    }

    /// Get statistics for a specific key
    pub async fn stats(&self, key: &str) -> Option<BucketStats> {
        let buckets = self.buckets.read().await;
        buckets.get(key).map(|b| b.stats())
    }

    /// Get all keys with their statistics
    pub async fn all_stats(&self) -> HashMap<String, BucketStats> {
        let buckets = self.buckets.read().await;
        buckets.iter().map(|(k, v)| (k.clone(), v.stats())).collect()
    }

    /// Clear expired buckets (buckets that haven't been used recently)
    pub async fn cleanup(&self, max_age: Duration) {
        let mut buckets = self.buckets.write().await;
        let now = Instant::now();

        buckets.retain(|key, bucket| {
            let age = now.duration_since(bucket.last_refill);
            if age > max_age {
                debug!(
                    rate_limiter = %self.id,
                    key = %key,
                    "Cleaned up expired rate limit bucket"
                );
                false
            } else {
                true
            }
        });
    }

    /// Get number of tracked keys
    pub async fn key_count(&self) -> usize {
        self.buckets.read().await.len()
    }

    /// Set custom configuration for a specific key
    pub async fn set_key_config(&self, key: &str, config: RateLimiterConfig) {
        let mut buckets = self.buckets.write().await;
        buckets.insert(key.to_string(), TokenBucket::new(config));
    }
}

/// Rate limiter middleware result
#[derive(Debug, Clone)]
pub struct RateLimitResult {
    /// Whether the request was allowed
    pub allowed: bool,
    /// Remaining requests in current window
    pub remaining: u32,
    /// Total limit
    pub limit: u32,
    /// Time until limit resets
    pub reset_after: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_rate_limiter_allows_within_limit() {
        let limiter = RateLimiter::new(
            "test",
            RateLimiterConfig {
                requests_per_window: 10,
                tokens_per_window: None,
                window: Duration::from_secs(60),
                enable_burst: false,
                burst_multiplier: 1.0,
            },
        );

        // Should allow up to 10 requests
        for i in 0..10 {
            let result = limiter.acquire(&format!("key-{i}")).await;
            assert!(result.is_ok(), "Request {i} should be allowed");
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_blocks_over_limit() {
        let limiter = RateLimiter::new(
            "test",
            RateLimiterConfig {
                requests_per_window: 5,
                tokens_per_window: None,
                window: Duration::from_secs(60),
                enable_burst: false,
                burst_multiplier: 1.0,
            },
        );

        let key = "test-key";

        // Should allow 5 requests
        for i in 0..5 {
            let result = limiter.acquire(key).await;
            assert!(result.is_ok(), "Request {i} should be allowed");
        }

        // 6th request should be blocked
        let result = limiter.acquire(key).await;
        assert!(result.is_err(), "6th request should be blocked");
    }

    #[tokio::test]
    async fn test_rate_limiter_refills_over_time() {
        let limiter = RateLimiter::new(
            "test",
            RateLimiterConfig {
                requests_per_window: 10,
                tokens_per_window: None,
                window: Duration::from_millis(100), // Short window for testing
                enable_burst: false,
                burst_multiplier: 1.0,
            },
        );

        let key = "test-key";

        // Exhaust all tokens
        for _ in 0..10 {
            limiter.acquire(key).await.ok();
        }

        // Should be blocked
        assert!(limiter.acquire(key).await.is_err());

        // Wait for refill
        sleep(Duration::from_millis(150)).await;

        // Should be allowed again
        assert!(limiter.acquire(key).await.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limiter_per_key() {
        let limiter = RateLimiter::new(
            "test",
            RateLimiterConfig {
                requests_per_window: 2,
                tokens_per_window: None,
                window: Duration::from_secs(60),
                enable_burst: false,
                burst_multiplier: 1.0,
            },
        );

        // Each key has its own bucket
        assert!(limiter.acquire("key1").await.is_ok());
        assert!(limiter.acquire("key1").await.is_ok());
        assert!(limiter.acquire("key1").await.is_err()); // key1 exhausted

        assert!(limiter.acquire("key2").await.is_ok()); // key2 still has tokens
        assert!(limiter.acquire("key2").await.is_ok());
        assert!(limiter.acquire("key2").await.is_err()); // key2 exhausted
    }

    #[tokio::test]
    async fn test_rate_limiter_disabled() {
        let limiter = RateLimiter::disabled("test");

        // Should always allow when disabled
        for _ in 0..100 {
            assert!(limiter.acquire("key").await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_burst() {
        let limiter = RateLimiter::new(
            "test",
            RateLimiterConfig {
                requests_per_window: 10,
                tokens_per_window: None,
                window: Duration::from_secs(60),
                enable_burst: true,
                burst_multiplier: 1.5, // Can burst to 15
            },
        );

        let key = "test-key";

        // Should allow more than 10 due to burst
        for i in 0..15 {
            let result = limiter.acquire(key).await;
            assert!(result.is_ok(), "Request {i} should be allowed (burst)");
        }

        // 16th should fail
        assert!(limiter.acquire(key).await.is_err());
    }

    #[tokio::test]
    async fn test_token_based_limiting() {
        let limiter = RateLimiter::new(
            "test",
            RateLimiterConfig {
                requests_per_window: 100, // High request limit
                tokens_per_window: Some(1000),
                window: Duration::from_secs(60),
                enable_burst: false,
                burst_multiplier: 1.0,
            },
        );

        let key = "test-key";

        // Request using 500 tokens should work
        assert!(limiter.check(key, Some(500)).await.is_ok());

        // Request using 400 tokens should work
        assert!(limiter.check(key, Some(400)).await.is_ok());

        // Request using 200 tokens should fail (only 100 left)
        assert!(limiter.check(key, Some(200)).await.is_err());
    }

    #[tokio::test]
    async fn test_rate_limiter_stats() {
        let limiter = RateLimiter::new(
            "test",
            RateLimiterConfig {
                requests_per_window: 10,
                tokens_per_window: Some(1000),
                window: Duration::from_secs(60),
                enable_burst: false,
                burst_multiplier: 1.0,
            },
        );

        let key = "test-key";

        // Make some requests
        limiter.acquire(key).await.ok();
        limiter.check(key, Some(100)).await.ok();

        let stats = limiter.stats(key).await.expect("stats");
        assert!(stats.request_tokens_available < 10.0);
        assert!(stats.token_tokens_available.unwrap_or(0.0) < 1000.0);
    }

    #[tokio::test]
    async fn test_rate_limiter_cleanup() {
        let limiter = RateLimiter::new(
            "test",
            RateLimiterConfig {
                requests_per_window: 10,
                tokens_per_window: None,
                window: Duration::from_secs(60),
                enable_burst: false,
                burst_multiplier: 1.0,
            },
        );

        // Create some buckets
        limiter.acquire("key1").await.ok();
        limiter.acquire("key2").await.ok();

        assert_eq!(limiter.key_count().await, 2);

        // Cleanup with very short max age
        sleep(Duration::from_millis(10)).await;
        limiter.cleanup(Duration::from_millis(5)).await;

        assert_eq!(limiter.key_count().await, 0);
    }

    #[tokio::test]
    async fn test_custom_key_config() {
        let limiter = RateLimiter::new(
            "test",
            RateLimiterConfig {
                requests_per_window: 10,
                tokens_per_window: None,
                window: Duration::from_secs(60),
                enable_burst: false,
                burst_multiplier: 1.0,
            },
        );

        // Set custom config for premium key
        limiter
            .set_key_config(
                "premium",
                RateLimiterConfig {
                    requests_per_window: 100,
                    tokens_per_window: None,
                    window: Duration::from_secs(60),
                    enable_burst: true,
                    burst_multiplier: 2.0,
                },
            )
            .await;

        // Premium key should have higher limit
        for i in 0..50 {
            let result = limiter.acquire("premium").await;
            assert!(result.is_ok(), "Premium request {i} should be allowed");
        }

        // Regular key still has default limit
        for _ in 0..10 {
            limiter.acquire("regular").await.ok();
        }
        assert!(limiter.acquire("regular").await.is_err());
    }

    #[test]
    fn test_bucket_stats_utilization() {
        let stats = BucketStats {
            request_tokens_available: 3.0,
            token_tokens_available: Some(250.0),
            requests_per_window: 10,
            tokens_per_window: Some(1000),
        };

        // 7 out of 10 used = 70%
        assert!((stats.request_utilization() - 70.0).abs() < 0.1);

        // 750 out of 1000 used = 75%
        assert!((stats.token_utilization().unwrap() - 75.0).abs() < 0.1);
    }
}
