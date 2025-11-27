//! Main router for request routing decisions.
//!
//! Combines rules engine, load balancer, and provider selection
//! to make intelligent routing decisions.

use crate::load_balancer::{LoadBalancer, LoadBalancerConfig};
use crate::rules::{MatchContext, RuleAction, RoutingRule, RulesEngine};
use crate::selector::{ProviderCandidate, SelectionCriteria};
use gateway_core::{GatewayError, GatewayRequest, HealthStatus, LLMProvider};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info, instrument};

/// Router configuration
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Load balancer configuration
    pub load_balancer: LoadBalancerConfig,
    /// Default providers when no rules match
    pub default_providers: Vec<String>,
    /// Enable rule-based routing
    pub rules_enabled: bool,
    /// Default strategy when no rule specifies one
    pub default_strategy: String,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            load_balancer: LoadBalancerConfig::default(),
            default_providers: Vec::new(),
            rules_enabled: true,
            default_strategy: "round_robin".to_string(),
        }
    }
}

impl RouterConfig {
    /// Create a new router configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set default providers
    #[must_use]
    pub fn with_default_providers(mut self, providers: Vec<String>) -> Self {
        self.default_providers = providers;
        self
    }

    /// Set load balancer config
    #[must_use]
    pub fn with_load_balancer(mut self, config: LoadBalancerConfig) -> Self {
        self.load_balancer = config;
        self
    }

    /// Enable/disable rules
    #[must_use]
    pub fn with_rules_enabled(mut self, enabled: bool) -> Self {
        self.rules_enabled = enabled;
        self
    }
}

/// Route decision made by the router
#[derive(Debug, Clone)]
pub struct RouteDecision {
    /// Selected provider
    pub provider_id: String,
    /// Model to use (may be transformed)
    pub model: String,
    /// Additional headers to add
    pub headers: HashMap<String, String>,
    /// Rules that matched
    pub matched_rules: Vec<String>,
    /// Strategy used
    pub strategy: String,
}

/// Main router for making routing decisions
pub struct Router {
    /// Configuration
    config: RouterConfig,
    /// Rules engine
    rules: RwLock<RulesEngine>,
    /// Load balancer
    load_balancer: LoadBalancer,
    /// Registered providers
    providers: RwLock<HashMap<String, ProviderEntry>>,
}

/// Provider entry in the router
struct ProviderEntry {
    provider: Arc<dyn LLMProvider>,
    weight: u32,
    priority: u32,
    health: HealthStatus,
}

impl Router {
    /// Create a new router
    #[must_use]
    pub fn new(config: RouterConfig) -> Self {
        let load_balancer = LoadBalancer::new(config.load_balancer.clone());

        Self {
            config,
            rules: RwLock::new(RulesEngine::new()),
            load_balancer,
            providers: RwLock::new(HashMap::new()),
        }
    }

    /// Register a provider
    pub fn register_provider(
        &self,
        provider: Arc<dyn LLMProvider>,
        weight: u32,
        priority: u32,
    ) {
        let id = provider.id().to_string();
        let mut providers = self.providers.write();
        providers.insert(
            id.clone(),
            ProviderEntry {
                provider,
                weight,
                priority,
                health: HealthStatus::Unknown,
            },
        );
        info!(provider_id = %id, weight, priority, "Provider registered with router");
    }

    /// Deregister a provider
    pub fn deregister_provider(&self, id: &str) {
        let mut providers = self.providers.write();
        if providers.remove(id).is_some() {
            info!(provider_id = %id, "Provider deregistered from router");
        }
    }

    /// Update provider health status
    pub fn update_health(&self, provider_id: &str, health: HealthStatus) {
        let mut providers = self.providers.write();
        if let Some(entry) = providers.get_mut(provider_id) {
            entry.health = health;
            debug!(provider_id = %provider_id, health = %health, "Health updated");
        }
    }

    /// Add a routing rule
    pub fn add_rule(&self, rule: RoutingRule) {
        let mut rules = self.rules.write();
        info!(rule_id = %rule.id, rule_name = %rule.name, "Adding routing rule");
        rules.add_rule(rule);
    }

    /// Remove a routing rule
    pub fn remove_rule(&self, id: &str) -> Option<RoutingRule> {
        let mut rules = self.rules.write();
        rules.remove_rule(id)
    }

    /// Set all routing rules (replaces existing)
    pub fn set_rules(&self, rules: Vec<RoutingRule>) {
        let mut engine = self.rules.write();
        info!(count = rules.len(), "Setting routing rules");
        engine.set_rules(rules);
    }

    /// Route a request to a provider
    #[instrument(skip(self, request), fields(model = %request.model))]
    pub fn route(
        &self,
        request: &GatewayRequest,
        tenant_id: Option<&str>,
    ) -> Result<(Arc<dyn LLMProvider>, RouteDecision), GatewayError> {
        // Build match context
        let context = self.build_match_context(request, tenant_id);

        // Get matching rule actions - collect into owned data
        let matched_actions: Vec<RuleAction> = if self.config.rules_enabled {
            let rules = self.rules.read();
            rules.evaluate(&context).into_iter().cloned().collect()
        } else {
            vec![]
        };
        let matched_action_refs: Vec<&RuleAction> = matched_actions.iter().collect();

        // Merge actions to get routing parameters
        let (target_providers, strategy, model_transform, headers, matched_rules) =
            self.merge_actions(&matched_action_refs, &request.model);

        // Get provider candidates
        let candidates = self.build_candidates(&target_providers);

        if candidates.is_empty() {
            return Err(GatewayError::NoHealthyProviders {
                model: request.model.clone(),
            });
        }

        // Build selection criteria from request
        let criteria = SelectionCriteria::from_request(request);

        // Select provider via load balancer
        let provider = self.load_balancer.select(&candidates, &criteria, tenant_id)?;

        // Apply model transform if any
        let model = model_transform.map_or_else(|| request.model.clone(), |t| t.apply(&request.model));

        let decision = RouteDecision {
            provider_id: provider.id().to_string(),
            model,
            headers,
            matched_rules,
            strategy,
        };

        debug!(
            provider = %decision.provider_id,
            model = %decision.model,
            rules = ?decision.matched_rules,
            "Route decision made"
        );

        Ok((provider, decision))
    }

    /// Record request completion for load balancer
    pub fn record_completion(
        &self,
        provider_id: &str,
        latency: Duration,
        success: bool,
    ) {
        self.load_balancer.record_completion(provider_id, latency, success);
    }

    /// Get the load balancer for direct access
    #[must_use]
    pub fn load_balancer(&self) -> &LoadBalancer {
        &self.load_balancer
    }

    fn build_match_context(&self, request: &GatewayRequest, tenant_id: Option<&str>) -> MatchContext {
        let mut context = MatchContext::new().with_model(&request.model);

        if let Some(tenant) = tenant_id {
            context = context.with_tenant(tenant);
        }

        // Add any request metadata
        if let Some(metadata) = &request.metadata {
            if let Some(tenant) = &metadata.tenant_id {
                context = context.with_metadata("tenant_id", tenant.clone());
            }
            if let Some(project) = &metadata.project_id {
                context = context.with_metadata("project_id", project.clone());
            }
            if let Some(env) = &metadata.environment {
                context = context.with_metadata("environment", env.clone());
            }
            for (key, value) in &metadata.tags {
                context = context.with_metadata(key.clone(), value.clone());
            }
        }

        context
    }

    fn merge_actions(
        &self,
        actions: &[&RuleAction],
        _default_model: &str,
    ) -> (
        Vec<String>,
        String,
        Option<crate::rules::ModelTransform>,
        HashMap<String, String>,
        Vec<String>,
    ) {
        let mut providers = Vec::new();
        let mut strategy = self.config.default_strategy.clone();
        let mut model_transform = None;
        let mut headers = HashMap::new();
        let matched_rules = Vec::new();

        for action in actions {
            // Collect providers
            providers.extend(action.providers.clone());

            // Use first specified strategy
            if let Some(s) = &action.strategy {
                if strategy == self.config.default_strategy {
                    strategy = s.clone();
                }
            }

            // Use first model transform
            if model_transform.is_none() {
                model_transform = action.model_transform.clone();
            }

            // Merge headers
            headers.extend(action.add_headers.clone());
        }

        // Fall back to default providers if none specified
        if providers.is_empty() {
            providers = self.config.default_providers.clone();
        }

        // If still no providers, use all registered providers
        if providers.is_empty() {
            let all_providers = self.providers.read();
            providers = all_providers.keys().cloned().collect();
        }

        (providers, strategy, model_transform, headers, matched_rules)
    }

    fn build_candidates(&self, target_providers: &[String]) -> Vec<ProviderCandidate> {
        let providers = self.providers.read();

        target_providers
            .iter()
            .filter_map(|id| {
                providers.get(id).map(|entry| {
                    ProviderCandidate::new(Arc::clone(&entry.provider))
                        .with_health(entry.health)
                        .with_weight(entry.weight)
                        .with_priority(entry.priority)
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{RuleAction, RuleMatcher};
    use gateway_core::{ChatChunk, GatewayResponse, ModelInfo, ProviderCapabilities, ProviderType};
    use futures::stream::BoxStream;

    struct MockProvider {
        id: String,
        models: Vec<ModelInfo>,
    }

    impl MockProvider {
        fn new(id: &str, models: Vec<&str>) -> Self {
            Self {
                id: id.to_string(),
                models: models.into_iter().map(ModelInfo::new).collect(),
            }
        }
    }

    #[async_trait::async_trait]
    impl LLMProvider for MockProvider {
        fn id(&self) -> &str {
            &self.id
        }

        fn provider_type(&self) -> ProviderType {
            ProviderType::Custom
        }

        async fn chat_completion(&self, _: &GatewayRequest) -> Result<GatewayResponse, GatewayError> {
            unimplemented!()
        }

        async fn chat_completion_stream(
            &self,
            _: &GatewayRequest,
        ) -> Result<BoxStream<'static, Result<ChatChunk, GatewayError>>, GatewayError> {
            unimplemented!()
        }

        async fn health_check(&self) -> HealthStatus {
            HealthStatus::Healthy
        }

        fn capabilities(&self) -> &ProviderCapabilities {
            static CAPS: ProviderCapabilities = ProviderCapabilities {
                chat: true,
                streaming: true,
                function_calling: false,
                vision: false,
                embeddings: false,
                json_mode: false,
                seed: false,
                logprobs: false,
                max_context_length: None,
                max_output_tokens: None,
                parallel_tool_calls: false,
            };
            &CAPS
        }

        fn models(&self) -> &[ModelInfo] {
            &self.models
        }

        fn base_url(&self) -> &str {
            "http://localhost"
        }
    }

    fn create_test_router() -> Router {
        let config = RouterConfig::new();
        let router = Router::new(config);

        // Register providers
        let openai = Arc::new(MockProvider::new("openai", vec!["gpt-4", "gpt-3.5-turbo"]));
        let anthropic = Arc::new(MockProvider::new("anthropic", vec!["claude-3"]));

        router.register_provider(openai, 100, 100);
        router.register_provider(anthropic, 100, 100);

        // Update health
        router.update_health("openai", HealthStatus::Healthy);
        router.update_health("anthropic", HealthStatus::Healthy);

        router
    }

    #[test]
    fn test_basic_routing() {
        let router = create_test_router();

        let request = GatewayRequest::builder()
            .model("gpt-4")
            .message(gateway_core::ChatMessage::user("Hello"))
            .build()
            .unwrap();

        let result = router.route(&request, None);
        assert!(result.is_ok());

        let (provider, decision) = result.unwrap();
        assert!(!decision.provider_id.is_empty());
    }

    #[test]
    fn test_rule_based_routing() {
        let router = create_test_router();

        // Add rule to route gpt-* to openai
        let rule = RoutingRule::new("gpt-rule", "GPT Models")
            .with_priority(10)
            .with_matcher(RuleMatcher::new().with_model("gpt-*"))
            .with_action(RuleAction::new().with_providers(vec!["openai".to_string()]));

        router.add_rule(rule);

        let request = GatewayRequest::builder()
            .model("gpt-4")
            .message(gateway_core::ChatMessage::user("Hello"))
            .build()
            .unwrap();

        let (provider, decision) = router.route(&request, None).unwrap();
        assert_eq!(provider.id(), "openai");
    }

    #[test]
    fn test_no_provider_error() {
        let config = RouterConfig::new().with_default_providers(vec!["nonexistent".to_string()]);
        let router = Router::new(config);

        let request = GatewayRequest::builder()
            .model("gpt-4")
            .message(gateway_core::ChatMessage::user("Hello"))
            .build()
            .unwrap();

        let result = router.route(&request, None);
        assert!(matches!(result, Err(GatewayError::NoHealthyProviders { .. })));
    }

    #[test]
    fn test_provider_registration() {
        let router = Router::new(RouterConfig::new());

        let provider = Arc::new(MockProvider::new("test", vec!["test-model"]));
        router.register_provider(provider, 100, 50);

        // Should be able to route to it
        router.update_health("test", HealthStatus::Healthy);

        let request = GatewayRequest::builder()
            .model("test-model")
            .message(gateway_core::ChatMessage::user("Hello"))
            .build()
            .unwrap();

        let result = router.route(&request, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_health_update() {
        let router = create_test_router();

        // Mark openai as unhealthy
        router.update_health("openai", HealthStatus::Unhealthy);

        // Add rule to try openai first, but it should fall back
        // Use "claude-3" model that anthropic supports
        let rule = RoutingRule::new("claude-3", "Test")
            .with_action(
                RuleAction::new()
                    .with_providers(vec!["openai".to_string(), "anthropic".to_string()]),
            );

        router.add_rule(rule);

        let request = GatewayRequest::builder()
            .model("claude-3")
            .message(gateway_core::ChatMessage::user("Hello"))
            .build()
            .unwrap();

        // Should route to anthropic since openai is unhealthy
        let (provider, _) = router.route(&request, None).unwrap();
        assert_eq!(provider.id(), "anthropic");
    }
}
