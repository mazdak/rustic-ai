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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct StubProvider;

    impl Provider for StubProvider {
        fn name(&self) -> &str {
            "stub"
        }

        fn model(&self, model: &str, _settings: Option<ModelSettings>) -> Arc<dyn Model> {
            struct StubModel {
                name: String,
            }
            #[async_trait]
            impl Model for StubModel {
                fn name(&self) -> &str {
                    &self.name
                }

                async fn request(
                    &self,
                    _messages: &[crate::messages::ModelMessage],
                    _settings: Option<&ModelSettings>,
                    _params: &crate::model::ModelRequestParameters,
                ) -> Result<crate::messages::ModelResponse, crate::model::ModelError>
                {
                    Err(crate::model::ModelError::Unsupported(
                        "not implemented".to_string(),
                    ))
                }
            }

            Arc::new(StubModel {
                name: model.to_string(),
            })
        }
    }

    #[test]
    fn infer_provider_from_model_matches_prefixes() {
        assert_eq!(infer_provider_from_model("gpt-4o").unwrap(), "openai");
        assert_eq!(infer_provider_from_model("o1-mini").unwrap(), "openai");
        assert_eq!(infer_provider_from_model("o3-mini").unwrap(), "openai");
        assert_eq!(infer_provider_from_model("claude-3").unwrap(), "anthropic");
        assert_eq!(infer_provider_from_model("gemini-1.5").unwrap(), "gemini");
        assert_eq!(infer_provider_from_model("grok-2").unwrap(), "grok");
    }

    #[test]
    fn infer_provider_from_model_rejects_unknown() {
        let err = infer_provider_from_model("unknown-model").expect_err("unknown provider");
        assert!(matches!(err, ProviderError::InvalidModel(_)));
    }

    #[test]
    fn infer_model_uses_explicit_provider() {
        let model = infer_model("stub:example", |_| Ok(Box::new(StubProvider)))
            .expect("model from stub provider");
        assert_eq!(model.name(), "example");
    }

    #[test]
    fn infer_model_infers_provider_without_prefix() {
        let model = infer_model("gpt-4o-mini", |_| Ok(Box::new(StubProvider)))
            .expect("model from inferred provider");
        assert_eq!(model.name(), "gpt-4o-mini");
    }
}
