//! Routing rules engine.
//!
//! Provides rule-based routing with pattern matching on:
//! - Model names
//! - Tenant IDs
//! - Request headers
//! - Request metadata

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// A routing rule that matches requests and determines routing behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    /// Rule identifier
    pub id: String,
    /// Rule name for display
    pub name: String,
    /// Rule priority (lower = higher priority)
    pub priority: u32,
    /// Whether the rule is enabled
    pub enabled: bool,
    /// Matcher conditions
    pub matcher: RuleMatcher,
    /// Action to take when matched
    pub action: RuleAction,
}

impl RoutingRule {
    /// Create a new routing rule
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            priority: 100,
            enabled: true,
            matcher: RuleMatcher::default(),
            action: RuleAction::default(),
        }
    }

    /// Set the priority
    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the matcher
    #[must_use]
    pub fn with_matcher(mut self, matcher: RuleMatcher) -> Self {
        self.matcher = matcher;
        self
    }

    /// Set the action
    #[must_use]
    pub fn with_action(mut self, action: RuleAction) -> Self {
        self.action = action;
        self
    }

    /// Check if this rule matches the given context
    #[must_use]
    pub fn matches(&self, context: &MatchContext) -> bool {
        if !self.enabled {
            return false;
        }
        self.matcher.matches(context)
    }
}

/// Context for rule matching
#[derive(Debug, Clone, Default)]
pub struct MatchContext {
    /// Model being requested
    pub model: Option<String>,
    /// Tenant ID
    pub tenant_id: Option<String>,
    /// Request headers
    pub headers: HashMap<String, String>,
    /// Request metadata
    pub metadata: HashMap<String, String>,
    /// Source IP address
    pub source_ip: Option<String>,
    /// Request path
    pub path: Option<String>,
}

impl MatchContext {
    /// Create a new match context
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the tenant ID
    #[must_use]
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Add a header
    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Add metadata
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Rule matcher configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuleMatcher {
    /// Model patterns to match (regex)
    #[serde(default)]
    pub models: Vec<String>,
    /// Tenant IDs to match
    #[serde(default)]
    pub tenants: Vec<String>,
    /// Header conditions (key -> value pattern)
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Metadata conditions
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Source IP patterns
    #[serde(default)]
    pub source_ips: Vec<String>,
    /// Match mode for multiple conditions
    #[serde(default)]
    pub match_mode: MatchMode,
}

impl RuleMatcher {
    /// Create a new matcher
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a model pattern
    #[must_use]
    pub fn with_model(mut self, pattern: impl Into<String>) -> Self {
        self.models.push(pattern.into());
        self
    }

    /// Add a tenant ID
    #[must_use]
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenants.push(tenant_id.into());
        self
    }

    /// Add a header condition
    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, pattern: impl Into<String>) -> Self {
        self.headers.insert(key.into(), pattern.into());
        self
    }

    /// Set match mode
    #[must_use]
    pub fn with_match_mode(mut self, mode: MatchMode) -> Self {
        self.match_mode = mode;
        self
    }

    /// Check if this matcher matches the given context
    #[must_use]
    pub fn matches(&self, context: &MatchContext) -> bool {
        let conditions = self.evaluate_conditions(context);

        if conditions.is_empty() {
            // No conditions = always match
            return true;
        }

        match self.match_mode {
            MatchMode::All => conditions.iter().all(|&b| b),
            MatchMode::Any => conditions.iter().any(|&b| b),
            MatchMode::None => conditions.iter().all(|&b| !b),
        }
    }

    fn evaluate_conditions(&self, context: &MatchContext) -> Vec<bool> {
        let mut conditions = Vec::new();

        // Check model patterns
        if !self.models.is_empty() {
            if let Some(model) = &context.model {
                let matched = self.models.iter().any(|pattern| {
                    matches_pattern(pattern, model)
                });
                conditions.push(matched);
            } else {
                conditions.push(false);
            }
        }

        // Check tenants
        if !self.tenants.is_empty() {
            if let Some(tenant) = &context.tenant_id {
                let matched = self.tenants.contains(tenant);
                conditions.push(matched);
            } else {
                conditions.push(false);
            }
        }

        // Check headers
        for (key, pattern) in &self.headers {
            if let Some(value) = context.headers.get(key) {
                conditions.push(matches_pattern(pattern, value));
            } else {
                conditions.push(false);
            }
        }

        // Check metadata
        for (key, pattern) in &self.metadata {
            if let Some(value) = context.metadata.get(key) {
                conditions.push(matches_pattern(pattern, value));
            } else {
                conditions.push(false);
            }
        }

        // Check source IPs
        if !self.source_ips.is_empty() {
            if let Some(ip) = &context.source_ip {
                let matched = self.source_ips.iter().any(|pattern| {
                    matches_ip_pattern(pattern, ip)
                });
                conditions.push(matched);
            } else {
                conditions.push(false);
            }
        }

        conditions
    }
}

/// Match mode for multiple conditions
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MatchMode {
    /// All conditions must match
    #[default]
    All,
    /// Any condition must match
    Any,
    /// No condition must match (negation)
    None,
}

/// Action to take when a rule matches
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuleAction {
    /// Target providers (by ID)
    #[serde(default)]
    pub providers: Vec<String>,
    /// Provider weights (provider_id -> weight)
    #[serde(default)]
    pub weights: HashMap<String, u32>,
    /// Load balancing strategy override
    pub strategy: Option<String>,
    /// Transform the model name
    pub model_transform: Option<ModelTransform>,
    /// Add headers to the request
    #[serde(default)]
    pub add_headers: HashMap<String, String>,
    /// Request priority boost
    pub priority_boost: Option<i32>,
    /// Whether to skip other rules after this one
    #[serde(default)]
    pub terminal: bool,
}

impl RuleAction {
    /// Create a new action
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target providers
    #[must_use]
    pub fn with_providers(mut self, providers: Vec<String>) -> Self {
        self.providers = providers;
        self
    }

    /// Add a provider with weight
    #[must_use]
    pub fn with_provider_weight(mut self, provider: impl Into<String>, weight: u32) -> Self {
        let provider = provider.into();
        self.providers.push(provider.clone());
        self.weights.insert(provider, weight);
        self
    }

    /// Set strategy
    #[must_use]
    pub fn with_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.strategy = Some(strategy.into());
        self
    }

    /// Set model transform
    #[must_use]
    pub fn with_model_transform(mut self, transform: ModelTransform) -> Self {
        self.model_transform = Some(transform);
        self
    }

    /// Make this a terminal rule
    #[must_use]
    pub fn terminal(mut self) -> Self {
        self.terminal = true;
        self
    }
}

/// Model name transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModelTransform {
    /// Replace the model name entirely
    #[serde(rename = "replace")]
    Replace {
        /// The value to replace the model name with
        value: String,
    },
    /// Map model names
    #[serde(rename = "map")]
    Map {
        /// Mapping from original model names to transformed names
        mappings: HashMap<String, String>,
    },
    /// Apply regex replacement
    #[serde(rename = "regex")]
    Regex {
        /// The regex pattern to match
        pattern: String,
        /// The replacement string
        replacement: String,
    },
    /// Strip prefix
    #[serde(rename = "strip_prefix")]
    StripPrefix {
        /// The prefix to strip from the model name
        prefix: String,
    },
    /// Strip suffix
    #[serde(rename = "strip_suffix")]
    StripSuffix {
        /// The suffix to strip from the model name
        suffix: String,
    },
}

impl ModelTransform {
    /// Apply the transformation to a model name
    #[must_use]
    pub fn apply(&self, model: &str) -> String {
        match self {
            Self::Replace { value } => value.clone(),
            Self::Map { mappings } => {
                mappings.get(model).cloned().unwrap_or_else(|| model.to_string())
            }
            Self::Regex { pattern, replacement } => {
                if let Ok(re) = Regex::new(pattern) {
                    re.replace_all(model, replacement.as_str()).to_string()
                } else {
                    model.to_string()
                }
            }
            Self::StripPrefix { prefix } => {
                model.strip_prefix(prefix.as_str()).unwrap_or(model).to_string()
            }
            Self::StripSuffix { suffix } => {
                model.strip_suffix(suffix.as_str()).unwrap_or(model).to_string()
            }
        }
    }
}

/// Rules engine for evaluating routing rules
pub struct RulesEngine {
    rules: Vec<RoutingRule>,
}

impl RulesEngine {
    /// Create a new rules engine
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Create with initial rules
    #[must_use]
    pub fn with_rules(rules: Vec<RoutingRule>) -> Self {
        let mut engine = Self::new();
        engine.set_rules(rules);
        engine
    }

    /// Set rules (replaces existing rules and sorts by priority)
    pub fn set_rules(&mut self, mut rules: Vec<RoutingRule>) {
        rules.sort_by_key(|r| r.priority);
        self.rules = rules;
    }

    /// Add a rule
    pub fn add_rule(&mut self, rule: RoutingRule) {
        self.rules.push(rule);
        self.rules.sort_by_key(|r| r.priority);
    }

    /// Remove a rule by ID
    pub fn remove_rule(&mut self, id: &str) -> Option<RoutingRule> {
        if let Some(pos) = self.rules.iter().position(|r| r.id == id) {
            Some(self.rules.remove(pos))
        } else {
            None
        }
    }

    /// Evaluate rules and return matching actions
    #[must_use]
    pub fn evaluate(&self, context: &MatchContext) -> Vec<&RuleAction> {
        let mut actions = Vec::new();

        for rule in &self.rules {
            if rule.matches(context) {
                debug!(
                    rule_id = %rule.id,
                    rule_name = %rule.name,
                    "Rule matched"
                );
                actions.push(&rule.action);

                if rule.action.terminal {
                    break;
                }
            }
        }

        actions
    }

    /// Get the first matching action
    #[must_use]
    pub fn evaluate_first(&self, context: &MatchContext) -> Option<&RuleAction> {
        self.evaluate(context).into_iter().next()
    }

    /// Get all rules
    #[must_use]
    pub fn rules(&self) -> &[RoutingRule] {
        &self.rules
    }
}

impl Default for RulesEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a pattern matches a value (supports glob-like patterns)
fn matches_pattern(pattern: &str, value: &str) -> bool {
    // Handle exact match
    if !pattern.contains('*') && !pattern.contains('?') {
        return pattern == value;
    }

    // Convert glob to regex
    let regex_pattern = glob_to_regex(pattern);
    if let Ok(re) = Regex::new(&regex_pattern) {
        re.is_match(value)
    } else {
        pattern == value
    }
}

/// Convert a glob pattern to regex
fn glob_to_regex(glob: &str) -> String {
    let mut regex = String::from("^");
    for c in glob.chars() {
        match c {
            '*' => regex.push_str(".*"),
            '?' => regex.push('.'),
            '.' | '+' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\' => {
                regex.push('\\');
                regex.push(c);
            }
            _ => regex.push(c),
        }
    }
    regex.push('$');
    regex
}

/// Check if an IP pattern matches an IP address
fn matches_ip_pattern(pattern: &str, ip: &str) -> bool {
    // Handle CIDR notation (simplified)
    if pattern.contains('/') {
        // For now, just do prefix matching on the network part
        let parts: Vec<&str> = pattern.split('/').collect();
        if parts.len() == 2 {
            let network = parts[0];
            return ip.starts_with(network.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.'));
        }
    }

    // Handle wildcards
    if pattern.contains('*') {
        return matches_pattern(pattern, ip);
    }

    // Exact match
    pattern == ip
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_matching() {
        let rule = RoutingRule::new("test", "Test Rule")
            .with_matcher(RuleMatcher::new().with_model("gpt-*"));

        let context = MatchContext::new().with_model("gpt-4");
        assert!(rule.matches(&context));

        let context = MatchContext::new().with_model("claude-3");
        assert!(!rule.matches(&context));
    }

    #[test]
    fn test_multiple_conditions_all() {
        let matcher = RuleMatcher::new()
            .with_model("gpt-*")
            .with_tenant("tenant-1")
            .with_match_mode(MatchMode::All);

        let context = MatchContext::new()
            .with_model("gpt-4")
            .with_tenant("tenant-1");
        assert!(matcher.matches(&context));

        let context = MatchContext::new()
            .with_model("gpt-4")
            .with_tenant("tenant-2");
        assert!(!matcher.matches(&context));
    }

    #[test]
    fn test_multiple_conditions_any() {
        let matcher = RuleMatcher::new()
            .with_model("gpt-*")
            .with_tenant("tenant-1")
            .with_match_mode(MatchMode::Any);

        let context = MatchContext::new()
            .with_model("claude-3")
            .with_tenant("tenant-1");
        assert!(matcher.matches(&context));

        let context = MatchContext::new()
            .with_model("gpt-4")
            .with_tenant("tenant-2");
        assert!(matcher.matches(&context));

        let context = MatchContext::new()
            .with_model("claude-3")
            .with_tenant("tenant-2");
        assert!(!matcher.matches(&context));
    }

    #[test]
    fn test_header_matching() {
        let matcher = RuleMatcher::new()
            .with_header("x-priority", "high");

        let context = MatchContext::new()
            .with_header("x-priority", "high");
        assert!(matcher.matches(&context));

        let context = MatchContext::new()
            .with_header("x-priority", "low");
        assert!(!matcher.matches(&context));
    }

    #[test]
    fn test_model_transform_replace() {
        let transform = ModelTransform::Replace {
            value: "gpt-4-turbo".to_string(),
        };
        assert_eq!(transform.apply("gpt-4"), "gpt-4-turbo");
    }

    #[test]
    fn test_model_transform_map() {
        let mut mappings = HashMap::new();
        mappings.insert("gpt-4".to_string(), "gpt-4-turbo".to_string());

        let transform = ModelTransform::Map { mappings };
        assert_eq!(transform.apply("gpt-4"), "gpt-4-turbo");
        assert_eq!(transform.apply("gpt-3.5"), "gpt-3.5"); // Not in map
    }

    #[test]
    fn test_model_transform_strip() {
        let transform = ModelTransform::StripPrefix {
            prefix: "openai/".to_string(),
        };
        assert_eq!(transform.apply("openai/gpt-4"), "gpt-4");

        let transform = ModelTransform::StripSuffix {
            suffix: "-latest".to_string(),
        };
        assert_eq!(transform.apply("gpt-4-latest"), "gpt-4");
    }

    #[test]
    fn test_rules_engine() {
        let rules = vec![
            RoutingRule::new("rule-1", "High Priority GPT")
                .with_priority(10)
                .with_matcher(RuleMatcher::new().with_model("gpt-4*"))
                .with_action(RuleAction::new().with_providers(vec!["openai".to_string()])),
            RoutingRule::new("rule-2", "Default")
                .with_priority(100)
                .with_action(RuleAction::new().with_providers(vec!["fallback".to_string()])),
        ];

        let engine = RulesEngine::with_rules(rules);

        let context = MatchContext::new().with_model("gpt-4-turbo");
        let actions = engine.evaluate(&context);
        assert_eq!(actions.len(), 2);
        assert!(actions[0].providers.contains(&"openai".to_string()));
    }

    #[test]
    fn test_terminal_rule() {
        let rules = vec![
            RoutingRule::new("rule-1", "Terminal")
                .with_priority(10)
                .with_action(RuleAction::new().terminal()),
            RoutingRule::new("rule-2", "Should not match")
                .with_priority(20),
        ];

        let engine = RulesEngine::with_rules(rules);
        let context = MatchContext::new();
        let actions = engine.evaluate(&context);

        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn test_glob_pattern() {
        assert!(matches_pattern("gpt-*", "gpt-4"));
        assert!(matches_pattern("gpt-*", "gpt-4-turbo"));
        assert!(!matches_pattern("gpt-*", "claude-3"));
        assert!(matches_pattern("gpt-?", "gpt-4"));
        assert!(!matches_pattern("gpt-?", "gpt-4-turbo"));
    }

    #[test]
    fn test_ip_pattern() {
        assert!(matches_ip_pattern("192.168.1.1", "192.168.1.1"));
        assert!(matches_ip_pattern("192.168.*.*", "192.168.1.100"));
        assert!(!matches_ip_pattern("192.168.1.1", "192.168.1.2"));
    }
}
