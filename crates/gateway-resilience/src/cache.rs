//! Response caching for LLM completions.
//!
//! Provides an in-memory cache for caching identical requests to reduce
//! latency and provider costs. Uses a hash of the request as the cache key.

use gateway_core::{GatewayRequest, GatewayResponse};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Whether caching is enabled
    pub enabled: bool,
    /// Maximum number of entries in the cache
    pub max_entries: usize,
    /// Default TTL for cache entries
    pub default_ttl: Duration,
    /// Whether to cache streaming responses (may be memory intensive)
    pub cache_streaming: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 10000,
            default_ttl: Duration::from_secs(3600), // 1 hour
            cache_streaming: false,
        }
    }
}

/// A cached response entry
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached response
    response: GatewayResponse,
    /// When the entry was created
    created_at: Instant,
    /// TTL for this entry
    ttl: Duration,
    /// Number of times this entry has been accessed
    hits: u64,
}

impl CacheEntry {
    fn new(response: GatewayResponse, ttl: Duration) -> Self {
        Self {
            response,
            created_at: Instant::now(),
            ttl,
            hits: 0,
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }
}

/// Cache key derived from request
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Model name
    model: String,
    /// Hash of the messages
    messages_hash: u64,
    /// Temperature (discretized for caching)
    temperature_bucket: u32,
    /// Max tokens
    max_tokens: Option<u32>,
}

impl CacheKey {
    /// Create a cache key from a request
    pub fn from_request(request: &GatewayRequest) -> Self {
        // Hash the messages
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for msg in &request.messages {
            msg.role.hash(&mut hasher);
            // Hash the message content
            match &msg.content {
                gateway_core::MessageContent::Text(text) => text.hash(&mut hasher),
                gateway_core::MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            gateway_core::ContentPart::Text { text } => text.hash(&mut hasher),
                            gateway_core::ContentPart::ImageUrl { image_url } => {
                                image_url.url.hash(&mut hasher);
                            }
                        }
                    }
                }
            }
        }
        let messages_hash = hasher.finish();

        // Discretize temperature into buckets (0.0-0.1, 0.1-0.2, etc.)
        let temperature_bucket = request
            .temperature
            .map(|t| (t * 10.0) as u32)
            .unwrap_or(7); // Default temperature ~0.7

        Self {
            model: request.model.clone(),
            messages_hash,
            temperature_bucket,
            max_tokens: request.max_tokens,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Current number of entries
    pub entries: usize,
    /// Number of evictions
    pub evictions: u64,
}

impl CacheStats {
    /// Calculate hit rate
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64 * 100.0
        }
    }
}

/// Response cache for LLM completions
pub struct ResponseCache {
    /// Cache configuration
    config: CacheConfig,
    /// The cache entries
    entries: Arc<RwLock<HashMap<CacheKey, CacheEntry>>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
}

impl ResponseCache {
    /// Create a new response cache
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Create a disabled cache
    #[must_use]
    pub fn disabled() -> Self {
        Self::new(CacheConfig {
            enabled: false,
            ..Default::default()
        })
    }

    /// Check if caching is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Check if a request is cacheable
    #[must_use]
    pub fn is_cacheable(&self, request: &GatewayRequest) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Don't cache streaming requests unless configured
        if request.stream && !self.config.cache_streaming {
            return false;
        }

        // Don't cache requests with very high temperatures (too random)
        if let Some(temp) = request.temperature {
            if temp > 1.5 {
                return false;
            }
        }

        true
    }

    /// Get a cached response
    pub async fn get(&self, request: &GatewayRequest) -> Option<GatewayResponse> {
        if !self.is_cacheable(request) {
            return None;
        }

        let key = CacheKey::from_request(request);

        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;

        if let Some(entry) = entries.get_mut(&key) {
            if entry.is_expired() {
                entries.remove(&key);
                stats.misses += 1;
                stats.entries = entries.len();
                debug!(model = %request.model, "Cache miss (expired)");
                None
            } else {
                entry.hits += 1;
                stats.hits += 1;
                debug!(
                    model = %request.model,
                    hits = entry.hits,
                    "Cache hit"
                );
                Some(entry.response.clone())
            }
        } else {
            stats.misses += 1;
            debug!(model = %request.model, "Cache miss");
            None
        }
    }

    /// Put a response in the cache
    pub async fn put(&self, request: &GatewayRequest, response: GatewayResponse) {
        if !self.is_cacheable(request) {
            return;
        }

        let key = CacheKey::from_request(request);

        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;

        // Check if we need to evict entries
        if entries.len() >= self.config.max_entries {
            self.evict_lru(&mut entries, &mut stats);
        }

        entries.insert(key, CacheEntry::new(response, self.config.default_ttl));
        stats.entries = entries.len();

        debug!(
            model = %request.model,
            entries = stats.entries,
            "Response cached"
        );
    }

    /// Put a response with custom TTL
    pub async fn put_with_ttl(
        &self,
        request: &GatewayRequest,
        response: GatewayResponse,
        ttl: Duration,
    ) {
        if !self.is_cacheable(request) {
            return;
        }

        let key = CacheKey::from_request(request);

        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;

        // Check if we need to evict entries
        if entries.len() >= self.config.max_entries {
            self.evict_lru(&mut entries, &mut stats);
        }

        entries.insert(key, CacheEntry::new(response, ttl));
        stats.entries = entries.len();
    }

    /// Evict least recently used entries
    fn evict_lru(&self, entries: &mut HashMap<CacheKey, CacheEntry>, stats: &mut CacheStats) {
        // First, remove all expired entries
        let before = entries.len();
        entries.retain(|_, entry| !entry.is_expired());
        let removed_expired = before - entries.len();

        // If we still need to evict, remove entries with lowest hit counts
        if entries.len() >= self.config.max_entries {
            let to_remove = entries.len() - self.config.max_entries + 1;

            // Find entries with lowest hits
            let mut hit_counts: Vec<(CacheKey, u64)> = entries
                .iter()
                .map(|(k, v)| (k.clone(), v.hits))
                .collect();
            hit_counts.sort_by_key(|(_, hits)| *hits);

            for (key, _) in hit_counts.into_iter().take(to_remove) {
                entries.remove(&key);
            }
        }

        let removed = before - entries.len();
        stats.evictions += removed as u64;

        if removed > 0 {
            info!(
                removed_expired,
                removed_total = removed,
                "Cache eviction completed"
            );
        }
    }

    /// Clear all cache entries
    pub async fn clear(&self) {
        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;

        entries.clear();
        stats.entries = 0;

        info!("Cache cleared");
    }

    /// Remove expired entries
    pub async fn cleanup_expired(&self) {
        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;

        let before = entries.len();
        entries.retain(|_, entry| !entry.is_expired());
        let removed = before - entries.len();

        stats.entries = entries.len();
        stats.evictions += removed as u64;

        if removed > 0 {
            debug!(removed, "Expired cache entries removed");
        }
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Invalidate cache for a specific model
    pub async fn invalidate_model(&self, model: &str) {
        let mut entries = self.entries.write().await;
        let mut stats = self.stats.write().await;

        let before = entries.len();
        entries.retain(|key, _| key.model != model);
        let removed = before - entries.len();

        stats.entries = entries.len();
        stats.evictions += removed as u64;

        if removed > 0 {
            info!(model, removed, "Model cache invalidated");
        }
    }
}

/// Cache lookup result for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLookupResult {
    /// Cache hit
    Hit,
    /// Cache miss
    Miss,
    /// Request not cacheable
    NotCacheable,
    /// Cache disabled
    Disabled,
}

#[cfg(test)]
mod tests {
    use super::*;
    use gateway_core::ChatMessage;

    fn make_request(model: &str, content: &str) -> GatewayRequest {
        GatewayRequest::builder()
            .model(model)
            .message(ChatMessage::user(content))
            .temperature(0.7)
            .max_tokens(100_u32)
            .build()
            .expect("valid request")
    }

    fn make_response() -> GatewayResponse {
        GatewayResponse {
            id: "test-id".to_string(),
            object: "chat.completion".to_string(),
            model: "gpt-4o".to_string(),
            choices: vec![],
            usage: gateway_core::Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            },
            created: 1234567890,
            provider: Some("test".to_string()),
            system_fingerprint: None,
        }
    }

    #[tokio::test]
    async fn test_cache_hit() {
        let cache = ResponseCache::with_defaults();
        let request = make_request("gpt-4o", "Hello");
        let response = make_response();

        // Put in cache
        cache.put(&request, response.clone()).await;

        // Should hit
        let cached = cache.get(&request).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().id, response.id);
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let cache = ResponseCache::with_defaults();
        let request = make_request("gpt-4o", "Hello");

        // Should miss
        let cached = cache.get(&request).await;
        assert!(cached.is_none());
    }

    #[tokio::test]
    async fn test_cache_different_messages() {
        let cache = ResponseCache::with_defaults();

        let request1 = make_request("gpt-4o", "Hello");
        let request2 = make_request("gpt-4o", "Goodbye");

        let response = make_response();

        cache.put(&request1, response.clone()).await;

        // Same model but different message should miss
        let cached = cache.get(&request2).await;
        assert!(cached.is_none());
    }

    #[tokio::test]
    async fn test_cache_different_models() {
        let cache = ResponseCache::with_defaults();

        let request1 = make_request("gpt-4o", "Hello");
        let request2 = make_request("gpt-4o-mini", "Hello");

        let response = make_response();

        cache.put(&request1, response.clone()).await;

        // Same message but different model should miss
        let cached = cache.get(&request2).await;
        assert!(cached.is_none());
    }

    #[tokio::test]
    async fn test_cache_expiry() {
        let cache = ResponseCache::new(CacheConfig {
            enabled: true,
            max_entries: 100,
            default_ttl: Duration::from_millis(50), // Very short TTL
            cache_streaming: false,
        });

        let request = make_request("gpt-4o", "Hello");
        let response = make_response();

        cache.put(&request, response).await;

        // Should hit immediately
        assert!(cache.get(&request).await.is_some());

        // Wait for expiry
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should miss after expiry
        assert!(cache.get(&request).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_disabled() {
        let cache = ResponseCache::disabled();
        let request = make_request("gpt-4o", "Hello");
        let response = make_response();

        cache.put(&request, response).await;

        // Should always miss when disabled
        assert!(cache.get(&request).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_streaming_not_cached() {
        let cache = ResponseCache::with_defaults();
        // Create a streaming request using the builder
        let request = GatewayRequest::builder()
            .model("gpt-4o")
            .message(ChatMessage::user("Hello"))
            .stream(true)
            .build()
            .expect("valid request");

        let response = make_response();

        cache.put(&request, response).await;

        // Should not cache streaming requests by default
        assert!(cache.get(&request).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = ResponseCache::with_defaults();
        let request = make_request("gpt-4o", "Hello");
        let response = make_response();

        // Miss
        cache.get(&request).await;

        let stats = cache.stats().await;
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        // Put and hit
        cache.put(&request, response).await;
        cache.get(&request).await;

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 50.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let cache = ResponseCache::new(CacheConfig {
            enabled: true,
            max_entries: 2,
            default_ttl: Duration::from_secs(3600),
            cache_streaming: false,
        });

        let request1 = make_request("gpt-4o", "First");
        let request2 = make_request("gpt-4o", "Second");
        let request3 = make_request("gpt-4o", "Third");
        let response = make_response();

        cache.put(&request1, response.clone()).await;
        cache.put(&request2, response.clone()).await;

        // Access request2 to increase its hit count
        cache.get(&request2).await;

        // This should trigger eviction
        cache.put(&request3, response.clone()).await;

        // request1 should have been evicted (lowest hits)
        assert!(cache.get(&request1).await.is_none());
        // request2 should still be cached (had higher hits)
        assert!(cache.get(&request2).await.is_some());
        // request3 should be cached
        assert!(cache.get(&request3).await.is_some());
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let cache = ResponseCache::with_defaults();
        let request = make_request("gpt-4o", "Hello");
        let response = make_response();

        cache.put(&request, response).await;
        assert!(cache.get(&request).await.is_some());

        cache.clear().await;
        assert!(cache.get(&request).await.is_none());
    }

    #[tokio::test]
    async fn test_invalidate_model() {
        let cache = ResponseCache::with_defaults();
        let request1 = make_request("gpt-4o", "Hello");
        let request2 = make_request("gpt-4o-mini", "Hello");
        let response = make_response();

        cache.put(&request1, response.clone()).await;
        cache.put(&request2, response.clone()).await;

        cache.invalidate_model("gpt-4o").await;

        // gpt-4o should be gone
        assert!(cache.get(&request1).await.is_none());
        // gpt-4o-mini should still be there
        assert!(cache.get(&request2).await.is_some());
    }

    #[test]
    fn test_cache_key_stability() {
        let request = make_request("gpt-4o", "Hello world");

        let key1 = CacheKey::from_request(&request);
        let key2 = CacheKey::from_request(&request);

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_key_different_content() {
        let request1 = make_request("gpt-4o", "Hello");
        let request2 = make_request("gpt-4o", "World");

        let key1 = CacheKey::from_request(&request1);
        let key2 = CacheKey::from_request(&request2);

        assert_ne!(key1, key2);
    }
}
