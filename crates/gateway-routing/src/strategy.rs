//! Load balancing strategies.
//!
//! Provides various algorithms for distributing requests across providers:
//! - Round Robin
//! - Weighted Round Robin
//! - Random
//! - Least Connections
//! - Latency-based

use parking_lot::Mutex;
use rand::Rng;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::debug;

/// Load balancing strategy trait
pub trait LoadBalancingStrategy: Send + Sync {
    /// Select the next provider index from a list of available providers
    fn select(&self, providers: &[ProviderStats]) -> Option<usize>;

    /// Notify the strategy of a completed request
    fn record_completion(&self, provider_index: usize, latency: Duration, success: bool);

    /// Get the strategy name
    fn name(&self) -> &'static str;
}

/// Statistics about a provider for load balancing decisions
#[derive(Debug, Clone)]
pub struct ProviderStats {
    /// Provider identifier
    pub id: String,
    /// Weight for weighted strategies
    pub weight: u32,
    /// Current active connections
    pub active_connections: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Whether the provider is healthy
    pub is_healthy: bool,
}

impl ProviderStats {
    /// Create new provider stats
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            weight: 100,
            active_connections: 0,
            avg_latency_ms: 0.0,
            success_rate: 1.0,
            is_healthy: true,
        }
    }

    /// Set the weight
    #[must_use]
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }
}

/// Round Robin load balancing strategy
pub struct RoundRobinStrategy {
    counter: AtomicUsize,
}

impl RoundRobinStrategy {
    /// Create a new round robin strategy
    #[must_use]
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

impl Default for RoundRobinStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingStrategy for RoundRobinStrategy {
    fn select(&self, providers: &[ProviderStats]) -> Option<usize> {
        let healthy: Vec<usize> = providers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_healthy)
            .map(|(i, _)| i)
            .collect();

        if healthy.is_empty() {
            return None;
        }

        let counter = self.counter.fetch_add(1, Ordering::Relaxed);
        let index = counter % healthy.len();
        Some(healthy[index])
    }

    fn record_completion(&self, _provider_index: usize, _latency: Duration, _success: bool) {
        // Round robin doesn't use completion data
    }

    fn name(&self) -> &'static str {
        "round_robin"
    }
}

/// Weighted Round Robin load balancing strategy
pub struct WeightedRoundRobinStrategy {
    state: Mutex<WeightedRRState>,
}

struct WeightedRRState {
    current_weights: HashMap<usize, i32>,
    gcd: u32,
    max_weight: u32,
}

impl WeightedRoundRobinStrategy {
    /// Create a new weighted round robin strategy
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(WeightedRRState {
                current_weights: HashMap::new(),
                gcd: 1,
                max_weight: 100,
            }),
        }
    }

    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 {
            a
        } else {
            Self::gcd(b, a % b)
        }
    }
}

impl Default for WeightedRoundRobinStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingStrategy for WeightedRoundRobinStrategy {
    fn select(&self, providers: &[ProviderStats]) -> Option<usize> {
        let healthy: Vec<(usize, &ProviderStats)> = providers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_healthy && p.weight > 0)
            .collect();

        if healthy.is_empty() {
            return None;
        }

        let mut state = self.state.lock();

        // Recalculate GCD and max weight
        let weights: Vec<u32> = healthy.iter().map(|(_, p)| p.weight).collect();
        state.gcd = weights.iter().fold(weights[0], |acc, &w| Self::gcd(acc, w));
        state.max_weight = *weights.iter().max().unwrap_or(&100);

        // Find the provider with the highest current weight
        let mut selected_idx = None;
        let mut max_current_weight = i32::MIN;

        for &(idx, stats) in &healthy {
            let current = *state.current_weights.entry(idx).or_insert(0);
            let weighted = current + stats.weight as i32;
            state.current_weights.insert(idx, weighted);

            if weighted > max_current_weight {
                max_current_weight = weighted;
                selected_idx = Some(idx);
            }
        }

        // Decrease the selected provider's weight
        if let Some(idx) = selected_idx {
            let total_weight: i32 = healthy.iter().map(|(_, p)| p.weight as i32).sum();
            if let Some(w) = state.current_weights.get_mut(&idx) {
                *w -= total_weight;
            }
        }

        selected_idx
    }

    fn record_completion(&self, _provider_index: usize, _latency: Duration, _success: bool) {
        // Weighted round robin doesn't adapt based on completion
    }

    fn name(&self) -> &'static str {
        "weighted_round_robin"
    }
}

/// Random load balancing strategy
pub struct RandomStrategy;

impl RandomStrategy {
    /// Create a new random strategy
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for RandomStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingStrategy for RandomStrategy {
    fn select(&self, providers: &[ProviderStats]) -> Option<usize> {
        let healthy: Vec<usize> = providers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_healthy)
            .map(|(i, _)| i)
            .collect();

        if healthy.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..healthy.len());
        Some(healthy[index])
    }

    fn record_completion(&self, _provider_index: usize, _latency: Duration, _success: bool) {
        // Random doesn't use completion data
    }

    fn name(&self) -> &'static str {
        "random"
    }
}

/// Weighted Random load balancing strategy
pub struct WeightedRandomStrategy;

impl WeightedRandomStrategy {
    /// Create a new weighted random strategy
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for WeightedRandomStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingStrategy for WeightedRandomStrategy {
    fn select(&self, providers: &[ProviderStats]) -> Option<usize> {
        let healthy: Vec<(usize, u32)> = providers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_healthy && p.weight > 0)
            .map(|(i, p)| (i, p.weight))
            .collect();

        if healthy.is_empty() {
            return None;
        }

        let total_weight: u32 = healthy.iter().map(|(_, w)| w).sum();
        let mut rng = rand::thread_rng();
        let mut random = rng.gen_range(0..total_weight);

        for (idx, weight) in &healthy {
            if random < *weight {
                return Some(*idx);
            }
            random -= *weight;
        }

        // Fallback (shouldn't happen)
        healthy.last().map(|(i, _)| *i)
    }

    fn record_completion(&self, _provider_index: usize, _latency: Duration, _success: bool) {
        // Weighted random doesn't adapt
    }

    fn name(&self) -> &'static str {
        "weighted_random"
    }
}

/// Least Connections load balancing strategy
pub struct LeastConnectionsStrategy {
    connections: dashmap::DashMap<usize, AtomicU64>,
}

impl LeastConnectionsStrategy {
    /// Create a new least connections strategy
    #[must_use]
    pub fn new() -> Self {
        Self {
            connections: dashmap::DashMap::new(),
        }
    }

    /// Increment connection count for a provider
    pub fn increment(&self, provider_index: usize) {
        self.connections
            .entry(provider_index)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement connection count for a provider
    pub fn decrement(&self, provider_index: usize) {
        if let Some(count) = self.connections.get(&provider_index) {
            count.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Default for LeastConnectionsStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingStrategy for LeastConnectionsStrategy {
    fn select(&self, providers: &[ProviderStats]) -> Option<usize> {
        let healthy: Vec<(usize, u64)> = providers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_healthy)
            .map(|(i, p)| {
                let connections = self
                    .connections
                    .get(&i)
                    .map_or(0, |c| c.load(Ordering::Relaxed))
                    + p.active_connections;
                (i, connections)
            })
            .collect();

        if healthy.is_empty() {
            return None;
        }

        // Find minimum connections
        let min_conns = healthy.iter().map(|(_, c)| *c).min().unwrap_or(0);

        // Get all providers with minimum connections and pick randomly
        let candidates: Vec<usize> = healthy
            .iter()
            .filter(|(_, c)| *c == min_conns)
            .map(|(i, _)| *i)
            .collect();

        if candidates.len() == 1 {
            return Some(candidates[0]);
        }

        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..candidates.len());
        Some(candidates[idx])
    }

    fn record_completion(&self, provider_index: usize, _latency: Duration, _success: bool) {
        self.decrement(provider_index);
    }

    fn name(&self) -> &'static str {
        "least_connections"
    }
}

/// Latency-based load balancing strategy
pub struct LatencyBasedStrategy {
    latencies: dashmap::DashMap<usize, LatencyTracker>,
    /// Minimum samples before using latency data
    min_samples: usize,
    /// Exploration probability (0.0 - 1.0)
    exploration_rate: f64,
}

struct LatencyTracker {
    samples: Mutex<Vec<(Instant, Duration)>>,
    window: Duration,
}

impl LatencyTracker {
    fn new(window: Duration) -> Self {
        Self {
            samples: Mutex::new(Vec::new()),
            window,
        }
    }

    fn add_sample(&self, latency: Duration) {
        let mut samples = self.samples.lock();
        let now = Instant::now();

        // Remove old samples
        samples.retain(|(t, _)| now.duration_since(*t) < self.window);

        // Add new sample
        samples.push((now, latency));
    }

    fn average_latency(&self) -> Option<Duration> {
        let samples = self.samples.lock();
        if samples.is_empty() {
            return None;
        }

        let now = Instant::now();
        let valid: Vec<&Duration> = samples
            .iter()
            .filter(|(t, _)| now.duration_since(*t) < self.window)
            .map(|(_, d)| d)
            .collect();

        if valid.is_empty() {
            return None;
        }

        let sum: Duration = valid.iter().copied().sum();
        Some(sum / valid.len() as u32)
    }

    fn sample_count(&self) -> usize {
        self.samples.lock().len()
    }
}

impl LatencyBasedStrategy {
    /// Create a new latency-based strategy
    #[must_use]
    pub fn new() -> Self {
        Self {
            latencies: dashmap::DashMap::new(),
            min_samples: 5,
            exploration_rate: 0.1,
        }
    }

    /// Set minimum samples before using latency data
    #[must_use]
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Set exploration rate (probability of random selection)
    #[must_use]
    pub fn with_exploration_rate(mut self, rate: f64) -> Self {
        self.exploration_rate = rate.clamp(0.0, 1.0);
        self
    }
}

impl Default for LatencyBasedStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingStrategy for LatencyBasedStrategy {
    fn select(&self, providers: &[ProviderStats]) -> Option<usize> {
        let healthy: Vec<usize> = providers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_healthy)
            .map(|(i, _)| i)
            .collect();

        if healthy.is_empty() {
            return None;
        }

        // Exploration: randomly select with some probability
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.exploration_rate {
            let idx = rng.gen_range(0..healthy.len());
            debug!(provider_index = healthy[idx], "Exploration: random selection");
            return Some(healthy[idx]);
        }

        // Get latencies for all healthy providers
        let mut with_latency: Vec<(usize, Duration)> = Vec::new();
        let mut without_latency: Vec<usize> = Vec::new();

        for idx in healthy {
            if let Some(tracker) = self.latencies.get(&idx) {
                if tracker.sample_count() >= self.min_samples {
                    if let Some(avg) = tracker.average_latency() {
                        with_latency.push((idx, avg));
                        continue;
                    }
                }
            }
            without_latency.push(idx);
        }

        // If we have providers without sufficient latency data, prefer them for exploration
        if !without_latency.is_empty() {
            let idx = rng.gen_range(0..without_latency.len());
            debug!(
                provider_index = without_latency[idx],
                "Selecting provider without latency data"
            );
            return Some(without_latency[idx]);
        }

        // Select provider with lowest latency
        with_latency.sort_by_key(|(_, latency)| *latency);

        // Use power of two choices: pick two random, choose the better one
        if with_latency.len() >= 2 {
            let idx1 = rng.gen_range(0..with_latency.len());
            let mut idx2 = rng.gen_range(0..with_latency.len());
            while idx2 == idx1 {
                idx2 = rng.gen_range(0..with_latency.len());
            }

            let (provider1, latency1) = with_latency[idx1];
            let (provider2, latency2) = with_latency[idx2];

            if latency1 <= latency2 {
                debug!(
                    provider_index = provider1,
                    latency_ms = latency1.as_millis(),
                    "Selected by latency (power of two)"
                );
                return Some(provider1);
            }
            debug!(
                provider_index = provider2,
                latency_ms = latency2.as_millis(),
                "Selected by latency (power of two)"
            );
            return Some(provider2);
        }

        with_latency.first().map(|(idx, _)| *idx)
    }

    fn record_completion(&self, provider_index: usize, latency: Duration, success: bool) {
        if success {
            self.latencies
                .entry(provider_index)
                .or_insert_with(|| LatencyTracker::new(Duration::from_secs(300)))
                .add_sample(latency);
        }
    }

    fn name(&self) -> &'static str {
        "latency_based"
    }
}

/// Factory for creating load balancing strategies
pub struct StrategyFactory;

impl StrategyFactory {
    /// Create a strategy by name
    #[must_use]
    pub fn create(name: &str) -> Box<dyn LoadBalancingStrategy> {
        match name.to_lowercase().as_str() {
            "round_robin" | "roundrobin" => Box::new(RoundRobinStrategy::new()),
            "weighted_round_robin" | "weightedroundrobin" => {
                Box::new(WeightedRoundRobinStrategy::new())
            }
            "random" => Box::new(RandomStrategy::new()),
            "weighted_random" | "weightedrandom" => Box::new(WeightedRandomStrategy::new()),
            "least_connections" | "leastconnections" => Box::new(LeastConnectionsStrategy::new()),
            "latency" | "latency_based" => Box::new(LatencyBasedStrategy::new()),
            _ => Box::new(RoundRobinStrategy::new()), // Default fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_providers(count: usize) -> Vec<ProviderStats> {
        (0..count)
            .map(|i| ProviderStats::new(format!("provider-{i}")).with_weight(100))
            .collect()
    }

    #[test]
    fn test_round_robin_selection() {
        let strategy = RoundRobinStrategy::new();
        let providers = create_test_providers(3);

        let selections: Vec<usize> = (0..6)
            .filter_map(|_| strategy.select(&providers))
            .collect();

        // Should cycle through all providers
        assert_eq!(selections, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_round_robin_skips_unhealthy() {
        let strategy = RoundRobinStrategy::new();
        let mut providers = create_test_providers(3);
        providers[1].is_healthy = false;

        let selections: Vec<usize> = (0..4)
            .filter_map(|_| strategy.select(&providers))
            .collect();

        // Should only select healthy providers (0 and 2)
        assert_eq!(selections, vec![0, 2, 0, 2]);
    }

    #[test]
    fn test_weighted_round_robin() {
        let strategy = WeightedRoundRobinStrategy::new();
        let mut providers = create_test_providers(2);
        providers[0].weight = 100;
        providers[1].weight = 50;

        let mut counts = [0, 0];
        for _ in 0..30 {
            if let Some(idx) = strategy.select(&providers) {
                counts[idx] += 1;
            }
        }

        // Provider 0 should get roughly 2x the requests
        assert!(counts[0] > counts[1]);
    }

    #[test]
    fn test_random_selection() {
        let strategy = RandomStrategy::new();
        let providers = create_test_providers(3);

        let mut counts = [0, 0, 0];
        for _ in 0..300 {
            if let Some(idx) = strategy.select(&providers) {
                counts[idx] += 1;
            }
        }

        // All should have been selected at least once
        assert!(counts.iter().all(|&c| c > 0));
    }

    #[test]
    fn test_weighted_random() {
        let strategy = WeightedRandomStrategy::new();
        let mut providers = create_test_providers(2);
        providers[0].weight = 90;
        providers[1].weight = 10;

        let mut counts = [0, 0];
        for _ in 0..1000 {
            if let Some(idx) = strategy.select(&providers) {
                counts[idx] += 1;
            }
        }

        // Provider 0 should get significantly more
        assert!(counts[0] > counts[1] * 5);
    }

    #[test]
    fn test_least_connections() {
        let strategy = LeastConnectionsStrategy::new();
        let providers = create_test_providers(3);

        // Simulate connections
        strategy.increment(0);
        strategy.increment(0);
        strategy.increment(1);

        // Should select provider 2 (0 connections)
        let selected = strategy.select(&providers);
        assert_eq!(selected, Some(2));
    }

    #[test]
    fn test_latency_based() {
        let strategy = LatencyBasedStrategy::new().with_min_samples(2);
        let providers = create_test_providers(3);

        // Record some latencies
        for _ in 0..5 {
            strategy.record_completion(0, Duration::from_millis(100), true);
            strategy.record_completion(1, Duration::from_millis(50), true);
            strategy.record_completion(2, Duration::from_millis(200), true);
        }

        // Should prefer provider 1 (lowest latency) more often
        let mut counts = [0, 0, 0];
        for _ in 0..100 {
            if let Some(idx) = strategy.select(&providers) {
                counts[idx] += 1;
            }
        }

        // Provider 1 should be selected most often
        // (accounting for exploration)
        assert!(counts[1] > counts[0].min(counts[2]));
    }

    #[test]
    fn test_strategy_factory() {
        let strategies = ["round_robin", "random", "least_connections", "latency"];

        for name in strategies {
            let strategy = StrategyFactory::create(name);
            assert!(!strategy.name().is_empty());
        }
    }

    #[test]
    fn test_empty_providers() {
        let strategy = RoundRobinStrategy::new();
        let providers: Vec<ProviderStats> = vec![];

        assert!(strategy.select(&providers).is_none());
    }

    #[test]
    fn test_all_unhealthy() {
        let strategy = RoundRobinStrategy::new();
        let mut providers = create_test_providers(3);
        for p in &mut providers {
            p.is_healthy = false;
        }

        assert!(strategy.select(&providers).is_none());
    }
}
