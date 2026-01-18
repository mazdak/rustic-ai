use std::sync::{Arc, Mutex};

use rustic_ai::{
    FailoverResult, InMemoryResolver, ModelConfigEntry, ModelError, ResolvedModelConfig,
    run_with_config_and_classifier, run_with_failover,
};
use serde_json::Map;

#[tokio::test]
async fn run_with_config_fails_over_to_backup() {
    let config = ResolvedModelConfig {
        primary: "primary".to_string(),
        backup: Some("backup".to_string()),
        retry_limit: 0,
        failover_on: ["http_429"].into_iter().map(String::from).collect(),
        settings: Map::new(),
    };

    let calls = Arc::new(Mutex::new(Vec::new()));
    let calls_clone = Arc::clone(&calls);

    let result: FailoverResult<String> = run_with_config_and_classifier(
        config,
        move |model| {
            let calls = Arc::clone(&calls_clone);
            let model_name = model.to_string();
            async move {
                calls.lock().expect("lock calls").push(model_name.clone());
                if model_name == "primary" {
                    Err(ModelError::HttpStatus { status: 429 })
                } else {
                    Ok("ok".to_string())
                }
            }
        },
        |error| rustic_ai::classify_error_kind(error),
    )
    .await
    .expect("failover succeeded");

    assert_eq!(result.value, "ok");
    assert_eq!(result.model_used, "backup");
    assert!(result.failed_over);
    assert_eq!(result.primary_attempts, 1);
    assert_eq!(
        calls.lock().expect("lock calls").as_slice(),
        ["primary", "backup"]
    );
}

#[tokio::test]
async fn run_with_failover_uses_resolver() {
    let mut resolver = InMemoryResolver::new("primary");
    resolver.insert_agent(
        "agent",
        ModelConfigEntry::default()
            .backup("backup")
            .failover_on(["http_429"]),
    );

    let calls = Arc::new(Mutex::new(Vec::new()));
    let calls_clone = Arc::clone(&calls);

    let result = run_with_failover(&resolver, "agent", None, None, move |model| {
        let calls = Arc::clone(&calls_clone);
        let model_name = model.to_string();
        async move {
            calls.lock().expect("lock calls").push(model_name.clone());
            if model_name == "primary" {
                Err(ModelError::HttpStatus { status: 429 })
            } else {
                Ok("ok".to_string())
            }
        }
    })
    .await
    .expect("failover succeeded");

    assert_eq!(result.model_used, "backup");
    assert!(result.failed_over);
    assert_eq!(
        calls.lock().expect("lock calls").as_slice(),
        ["primary", "backup"]
    );
}

#[tokio::test]
async fn run_with_config_returns_error_when_not_retryable() {
    let config = ResolvedModelConfig {
        primary: "primary".to_string(),
        backup: Some("backup".to_string()),
        retry_limit: 0,
        failover_on: ["http_429"].into_iter().map(String::from).collect(),
        settings: Map::new(),
    };

    let calls = Arc::new(Mutex::new(Vec::new()));
    let calls_clone = Arc::clone(&calls);

    let result: Result<FailoverResult<String>, ModelError> = run_with_config_and_classifier(
        config,
        move |model| {
            let calls = Arc::clone(&calls_clone);
            let model_name = model.to_string();
            async move {
                calls.lock().expect("lock calls").push(model_name);
                Err(ModelError::HttpStatus { status: 401 })
            }
        },
        |error| rustic_ai::classify_error_kind(error),
    )
    .await;

    assert!(result.is_err());
    assert_eq!(calls.lock().expect("lock calls").as_slice(), ["primary"]);
}
