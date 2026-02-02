use std::sync::Arc;

use reqwest::Url;

use crate::model::{Model, ModelSettings};
use crate::providers::openai::{OpenAIChatCapabilities, OpenAIChatModel};
use crate::providers::{Provider, ProviderError};

#[derive(Clone, Debug)]
pub struct GrokProvider {
    api_key: String,
    base_url: Url,
}

impl GrokProvider {
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl AsRef<str>,
    ) -> Result<Self, ProviderError> {
        let url = Url::parse(base_url.as_ref())
            .map_err(|_| ProviderError::InvalidModel(base_url.as_ref().to_string()))?;
        Ok(Self {
            api_key: api_key.into(),
            base_url: url,
        })
    }

    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = std::env::var("XAI_API_KEY")
            .or_else(|_| std::env::var("GROK_API_KEY"))
            .map_err(|_| ProviderError::MissingApiKey("grok".to_string()))?;
        Self::new(api_key, "https://api.x.ai/v1")
    }
}

impl Provider for GrokProvider {
    fn name(&self) -> &str {
        "grok"
    }

    fn model(&self, model: &str, settings: Option<ModelSettings>) -> Arc<dyn Model> {
        let capabilities = OpenAIChatCapabilities {
            supports_response_format: false,
            supports_parallel_tool_calls: true,
            reject_binary_images: true,
        };
        Arc::new(OpenAIChatModel::new_with_capabilities(
            model,
            self.api_key.clone(),
            self.base_url.clone(),
            settings,
            capabilities,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grok_provider_rejects_invalid_url() {
        let err = GrokProvider::new("key", "not a url").expect_err("invalid url");
        assert!(matches!(err, ProviderError::InvalidModel(_)));
    }

    #[test]
    fn grok_provider_builds_model() {
        let provider = GrokProvider::new("key", "https://api.x.ai/v1").expect("valid provider");
        assert_eq!(provider.name(), "grok");
        let model = provider.model("grok-test", None);
        assert_eq!(model.name(), "grok-test");
    }
}
