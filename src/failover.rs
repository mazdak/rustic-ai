use std::error::Error;
use std::future::Future;

use crate::model_config::{ModelConfigResolver, ResolvedModelConfig};

use crate::error::AgentError;
use crate::model::ModelError;

pub fn classify_error_kind(error: &(dyn Error + 'static)) -> Option<&'static str> {
    if let Some(agent_error) = error.downcast_ref::<AgentError>() {
        return classify_agent_error(agent_error);
    }
    if let Some(model_error) = error.downcast_ref::<ModelError>() {
        return classify_model_error(model_error);
    }
    None
}

fn classify_agent_error(error: &AgentError) -> Option<&'static str> {
    match error {
        AgentError::Model(model_error) => classify_model_error(model_error),
        _ => None,
    }
}

fn classify_model_error(error: &ModelError) -> Option<&'static str> {
    match error {
        ModelError::Timeout => Some("timeout"),
        ModelError::Transport(_) => Some("connect_error"),
        ModelError::HttpStatus { status } => match *status {
            401 => Some("http_401"),
            403 => Some("http_403"),
            429 => Some("http_429"),
            status if status >= 500 => Some("http_5xx"),
            _ => None,
        },
        ModelError::Provider(_) | ModelError::Serialization(_) => Some("model_error"),
        ModelError::Unsupported(_) => None,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FailoverResult<T> {
    pub value: T,
    pub model_used: String,
    pub failed_over: bool,
    pub primary_attempts: u32,
}

pub async fn run_with_failover<T, E, F, Fut>(
    resolver: &dyn ModelConfigResolver,
    agent_name: &str,
    requested_model: Option<&str>,
    environment: Option<&str>,
    invoke: F,
) -> Result<FailoverResult<T>, E>
where
    E: Error + Send + Sync + 'static,
    F: FnMut(&str) -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    run_with_failover_with_classifier(
        resolver,
        agent_name,
        requested_model,
        environment,
        invoke,
        |error| classify_error_kind(error),
    )
    .await
}

pub async fn run_with_failover_with_classifier<T, E, F, Fut, C>(
    resolver: &dyn ModelConfigResolver,
    agent_name: &str,
    requested_model: Option<&str>,
    environment: Option<&str>,
    invoke: F,
    classifier: C,
) -> Result<FailoverResult<T>, E>
where
    E: Error + Send + Sync + 'static,
    F: FnMut(&str) -> Fut,
    Fut: Future<Output = Result<T, E>>,
    C: Fn(&E) -> Option<&'static str>,
{
    let config = resolver.resolve_model_config(agent_name, requested_model, environment);
    run_with_config_and_classifier(config, invoke, classifier).await
}

pub async fn run_with_utility_failover<T, E, F, Fut>(
    resolver: &dyn ModelConfigResolver,
    utility_name: &str,
    environment: Option<&str>,
    invoke: F,
) -> Result<FailoverResult<T>, E>
where
    E: Error + Send + Sync + 'static,
    F: FnMut(&str) -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    run_with_utility_failover_with_classifier(
        resolver,
        utility_name,
        environment,
        invoke,
        |error| classify_error_kind(error),
    )
    .await
}

pub async fn run_with_utility_failover_with_classifier<T, E, F, Fut, C>(
    resolver: &dyn ModelConfigResolver,
    utility_name: &str,
    environment: Option<&str>,
    invoke: F,
    classifier: C,
) -> Result<FailoverResult<T>, E>
where
    E: Error + Send + Sync + 'static,
    F: FnMut(&str) -> Fut,
    Fut: Future<Output = Result<T, E>>,
    C: Fn(&E) -> Option<&'static str>,
{
    let config = resolver.resolve_utility_config(utility_name, environment);
    run_with_config_and_classifier(config, invoke, classifier).await
}

pub async fn run_with_config<T, E, F, Fut>(
    config: ResolvedModelConfig,
    invoke: F,
) -> Result<FailoverResult<T>, E>
where
    E: Error + Send + Sync + 'static,
    F: FnMut(&str) -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    run_with_config_and_classifier(config, invoke, |error| classify_error_kind(error)).await
}

pub async fn run_with_config_and_classifier<T, E, F, Fut, C>(
    config: ResolvedModelConfig,
    mut invoke: F,
    classifier: C,
) -> Result<FailoverResult<T>, E>
where
    E: Error + Send + Sync + 'static,
    F: FnMut(&str) -> Fut,
    Fut: Future<Output = Result<T, E>>,
    C: Fn(&E) -> Option<&'static str>,
{
    let mut last_kind = None;
    let mut last_error = None;

    for attempt in 0..=config.retry_limit {
        match invoke(&config.primary).await {
            Ok(value) => {
                return Ok(FailoverResult {
                    value,
                    model_used: config.primary.clone(),
                    failed_over: false,
                    primary_attempts: attempt + 1,
                });
            }
            Err(error) => {
                let kind = classifier(&error);
                last_kind = kind;
                if !kind.is_some_and(|kind| config.failover_on.contains(kind)) {
                    return Err(error);
                }
                last_error = Some(error);
                if attempt < config.retry_limit {
                    continue;
                }
                break;
            }
        }
    }

    let should_failover =
        config.backup.is_some() && last_kind.is_some_and(|kind| config.failover_on.contains(kind));
    if !should_failover && let Some(error) = last_error {
        return Err(error);
    }

    let backup = config.backup.clone().unwrap_or_default();
    let result = invoke(&backup).await?;
    Ok(FailoverResult {
        value: result,
        model_used: backup,
        failed_over: true,
        primary_attempts: config.retry_limit + 1,
    })
}
