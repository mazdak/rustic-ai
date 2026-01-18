use std::collections::{HashMap, HashSet};

use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedModelConfig {
    pub primary: String,
    pub backup: Option<String>,
    pub retry_limit: u32,
    pub failover_on: HashSet<String>,
    pub settings: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub recovery_timeout: u32,
    pub window: u32,
    pub trigger_on: Vec<String>,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            recovery_timeout: 60,
            window: 300,
            trigger_on: vec![
                "http_403".to_string(),
                "http_401".to_string(),
                "connect_error".to_string(),
                "http_5xx".to_string(),
            ],
        }
    }
}

pub trait ModelConfigResolver: Send + Sync {
    fn resolve_model_config(
        &self,
        agent_name: &str,
        requested_model: Option<&str>,
        environment: Option<&str>,
    ) -> ResolvedModelConfig;

    fn resolve_utility_config(
        &self,
        utility_name: &str,
        environment: Option<&str>,
    ) -> ResolvedModelConfig;

    fn circuit_breaker_config(&self, _environment: Option<&str>) -> CircuitBreakerConfig {
        CircuitBreakerConfig::default()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelConfigEntry {
    pub primary: Option<String>,
    pub backup: Option<String>,
    pub retry_limit: Option<u32>,
    pub failover_on: Option<HashSet<String>>,
    pub settings: Option<Map<String, Value>>,
}

impl ModelConfigEntry {
    pub fn new(primary: impl Into<String>) -> Self {
        Self {
            primary: Some(primary.into()),
            ..Default::default()
        }
    }

    pub fn backup(mut self, backup: impl Into<String>) -> Self {
        self.backup = Some(backup.into());
        self
    }

    pub fn retry_limit(mut self, retry_limit: u32) -> Self {
        self.retry_limit = Some(retry_limit);
        self
    }

    pub fn failover_on<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let set = values.into_iter().map(Into::into).collect::<HashSet<_>>();
        self.failover_on = Some(set);
        self
    }

    pub fn setting(mut self, key: impl Into<String>, value: Value) -> Self {
        self.settings
            .get_or_insert_with(Map::new)
            .insert(key.into(), value);
        self
    }
}

#[derive(Clone, Debug)]
pub struct InMemoryResolver {
    defaults: ModelConfigEntry,
    agents: HashMap<String, ModelConfigEntry>,
    environments: HashMap<String, ModelConfigEntry>,
    utilities: HashMap<String, ModelConfigEntry>,
    circuit_breaker: CircuitBreakerConfig,
    fallback_model: Option<String>,
}

impl InMemoryResolver {
    pub fn new(primary: impl Into<String>) -> Self {
        Self {
            defaults: ModelConfigEntry::new(primary),
            agents: HashMap::new(),
            environments: HashMap::new(),
            utilities: HashMap::new(),
            circuit_breaker: CircuitBreakerConfig::default(),
            fallback_model: None,
        }
    }

    pub fn with_defaults(mut self, defaults: ModelConfigEntry) -> Self {
        self.defaults = defaults;
        self
    }

    pub fn with_fallback_model(mut self, model: impl Into<String>) -> Self {
        self.fallback_model = Some(model.into());
        self
    }

    pub fn insert_agent(&mut self, name: impl Into<String>, entry: ModelConfigEntry) {
        self.agents.insert(name.into(), entry);
    }

    pub fn insert_environment(&mut self, name: impl Into<String>, entry: ModelConfigEntry) {
        self.environments.insert(name.into().to_lowercase(), entry);
    }

    pub fn insert_utility(&mut self, name: impl Into<String>, entry: ModelConfigEntry) {
        self.utilities.insert(name.into(), entry);
    }

    pub fn set_circuit_breaker(&mut self, config: CircuitBreakerConfig) {
        self.circuit_breaker = config;
    }

    fn resolve_entries(
        &self,
        name: &str,
        environment: Option<&str>,
        map: &HashMap<String, ModelConfigEntry>,
    ) -> ModelConfigEntry {
        let env_key = environment.map(|env| env.to_lowercase());
        let env_entry = env_key
            .as_ref()
            .and_then(|key| self.environments.get(key))
            .cloned()
            .unwrap_or_default();
        let name_entry = map.get(name).cloned().unwrap_or_default();
        merge_entries(&[self.defaults.clone(), env_entry, name_entry])
    }

    fn build_resolved(
        &self,
        merged: ModelConfigEntry,
        requested_model: Option<&str>,
    ) -> ResolvedModelConfig {
        let primary = requested_model
            .map(str::to_string)
            .or(merged.primary)
            .or_else(|| self.fallback_model.clone())
            .unwrap_or_default();
        let retry_limit = merged.retry_limit.unwrap_or(0);
        let failover_on = merged.failover_on.unwrap_or_default();
        let settings = merged.settings.unwrap_or_default();

        ResolvedModelConfig {
            primary,
            backup: merged.backup,
            retry_limit,
            failover_on,
            settings,
        }
    }
}

impl ModelConfigResolver for InMemoryResolver {
    fn resolve_model_config(
        &self,
        agent_name: &str,
        requested_model: Option<&str>,
        environment: Option<&str>,
    ) -> ResolvedModelConfig {
        let merged = self.resolve_entries(agent_name, environment, &self.agents);
        self.build_resolved(merged, requested_model)
    }

    fn resolve_utility_config(
        &self,
        utility_name: &str,
        environment: Option<&str>,
    ) -> ResolvedModelConfig {
        let merged = self.resolve_entries(utility_name, environment, &self.utilities);
        self.build_resolved(merged, None)
    }

    fn circuit_breaker_config(&self, _environment: Option<&str>) -> CircuitBreakerConfig {
        self.circuit_breaker.clone()
    }
}

fn merge_entries(entries: &[ModelConfigEntry]) -> ModelConfigEntry {
    let mut merged = ModelConfigEntry::default();
    for entry in entries {
        if let Some(primary) = &entry.primary {
            merged.primary = Some(primary.clone());
        }
        if let Some(backup) = &entry.backup {
            merged.backup = Some(backup.clone());
        }
        if let Some(retry_limit) = entry.retry_limit {
            merged.retry_limit = Some(retry_limit);
        }
        if let Some(failover_on) = &entry.failover_on {
            merged.failover_on = Some(failover_on.clone());
        }
        if let Some(settings) = &entry.settings {
            let merged_settings = merged.settings.get_or_insert_with(Map::new);
            for (key, value) in settings {
                merged_settings.insert(key.clone(), value.clone());
            }
        }
    }
    merged
}
