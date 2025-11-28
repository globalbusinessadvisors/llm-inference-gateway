//! # Gateway Resilience
//!
//! Resilience patterns for the LLM Inference Gateway:
//! - Circuit breaker for preventing cascading failures
//! - Retry policy with exponential backoff
//! - Bulkhead pattern for resource isolation
//! - Timeout management
//! - Rate limiting with token bucket algorithm
//! - Response caching for performance optimization

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod circuit_breaker;
pub mod retry;
pub mod bulkhead;
pub mod timeout;
pub mod rate_limiter;
pub mod cache;

// Re-export main types
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
pub use retry::{RetryPolicy, RetryConfig, RetryResult};
pub use bulkhead::{Bulkhead, BulkheadConfig, BulkheadPermit};
pub use timeout::{TimeoutManager, TimeoutConfig};
pub use rate_limiter::{RateLimiter, RateLimiterConfig, RateLimitType, RateLimitExceeded, BucketStats};
pub use cache::{ResponseCache, CacheConfig, CacheKey, CacheStats, CacheLookupResult};
