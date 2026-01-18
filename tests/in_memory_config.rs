use std::collections::HashSet;

use rustic_ai::{InMemoryResolver, ModelConfigEntry, ModelConfigResolver};
use serde_json::{Map, json};

#[test]
fn in_memory_resolver_merges_defaults_env_and_agent() {
    let mut resolver = InMemoryResolver::new("openai:gpt-4o-mini");
    resolver.insert_environment(
        "Local",
        ModelConfigEntry::new("openai:gpt-4o").setting("temperature", json!(0.4)),
    );
    resolver.insert_agent(
        "agent",
        ModelConfigEntry::default()
            .backup("anthropic:claude-3.5-sonnet")
            .retry_limit(2)
            .failover_on(["http_429", "http_5xx"])
            .setting("max_tokens", json!(512)),
    );

    let resolved = resolver.resolve_model_config("agent", None, Some("LOCAL"));

    assert_eq!(resolved.primary, "openai:gpt-4o");
    assert_eq!(
        resolved.backup.as_deref(),
        Some("anthropic:claude-3.5-sonnet")
    );
    assert_eq!(resolved.retry_limit, 2);
    assert!(resolved.failover_on.contains("http_429"));
    assert_eq!(
        resolved
            .settings
            .get("temperature")
            .and_then(|value| value.as_f64()),
        Some(0.4)
    );
    assert_eq!(
        resolved
            .settings
            .get("max_tokens")
            .and_then(|value| value.as_i64()),
        Some(512)
    );
}

#[test]
fn in_memory_resolver_requested_model_overrides_primary() {
    let resolver = InMemoryResolver::new("openai:gpt-4o-mini");
    let resolved = resolver.resolve_model_config("agent", Some("openai:gpt-4o"), None);

    assert_eq!(resolved.primary, "openai:gpt-4o");
}

#[test]
fn in_memory_resolver_utility_uses_utility_entry() {
    let mut resolver = InMemoryResolver::new("openai:gpt-4o-mini");
    resolver.insert_utility(
        "summarizer",
        ModelConfigEntry::new("anthropic:claude-3.5-sonnet").retry_limit(1),
    );

    let resolved = resolver.resolve_utility_config("summarizer", None);

    assert_eq!(resolved.primary, "anthropic:claude-3.5-sonnet");
    assert_eq!(resolved.retry_limit, 1);
}

#[test]
fn in_memory_resolver_preserves_defaults_when_missing_entries() {
    let mut resolver = InMemoryResolver::new("openai:gpt-4o-mini");
    resolver.insert_environment("local", ModelConfigEntry::default());
    resolver.insert_agent("agent", ModelConfigEntry::default());
    resolver.insert_utility("utility", ModelConfigEntry::default());

    let resolved = resolver.resolve_model_config("agent", None, Some("local"));
    assert_eq!(resolved.primary, "openai:gpt-4o-mini");
    assert_eq!(resolved.retry_limit, 0);
    assert_eq!(resolved.failover_on, HashSet::new());
    assert_eq!(resolved.settings, Map::new());
}
