use std::sync::Arc;

use thiserror::Error;

use crate::model::{Model, ModelSettings};

pub mod anthropic;
pub mod gemini;
pub mod grok;
pub mod openai;

pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn model(&self, model: &str, settings: Option<ModelSettings>) -> Arc<dyn Model>;
}

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("unknown provider: {0}")]
    UnknownProvider(String),
    #[error("missing API key for provider: {0}")]
    MissingApiKey(String),
    #[error("invalid model string: {0}")]
    InvalidModel(String),
}

pub fn infer_provider(name: &str) -> Result<Box<dyn Provider>, ProviderError> {
    match name {
        "openai" => openai::OpenAIProvider::from_env().map(|p| Box::new(p) as Box<dyn Provider>),
        "grok" => grok::GrokProvider::from_env().map(|p| Box::new(p) as Box<dyn Provider>),
        "anthropic" => {
            anthropic::AnthropicProvider::from_env().map(|p| Box::new(p) as Box<dyn Provider>)
        }
        "gemini" => gemini::GeminiProvider::from_env().map(|p| Box::new(p) as Box<dyn Provider>),
        other => Err(ProviderError::UnknownProvider(other.to_string())),
    }
}

pub fn infer_model(
    model: impl AsRef<str>,
    provider_factory: impl Fn(&str) -> Result<Box<dyn Provider>, ProviderError>,
) -> Result<Arc<dyn Model>, ProviderError> {
    let model = model.as_ref();
    let (provider_name, model_name) = match model.split_once(':') {
        Some((provider, name)) => (provider, name),
        None => (infer_provider_from_model(model)?, model),
    };

    let provider = provider_factory(provider_name)?;
    Ok(provider.model(model_name, None))
}

fn infer_provider_from_model(model: &str) -> Result<&'static str, ProviderError> {
    let lowered = model.to_lowercase();
    if lowered.starts_with("gpt") || lowered.starts_with("o1") || lowered.starts_with("o3") {
        return Ok("openai");
    }
    if lowered.starts_with("claude") {
        return Ok("anthropic");
    }
    if lowered.starts_with("gemini") {
        return Ok("gemini");
    }
    if lowered.starts_with("grok") {
        return Ok("grok");
    }
    Err(ProviderError::InvalidModel(model.to_string()))
}
