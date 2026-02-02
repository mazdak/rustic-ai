use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use rustic_ai::{Model, ModelError, ModelRequestParameters, ModelResponse, ModelSettings};

#[derive(Debug)]
pub struct ScriptedModel {
    name: String,
    responses: Arc<Mutex<VecDeque<ModelResponse>>>,
}

impl ScriptedModel {
    pub fn new(name: impl Into<String>, responses: Vec<ModelResponse>) -> Self {
        Self {
            name: name.into(),
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
        }
    }

    pub async fn push_response(&self, response: ModelResponse) {
        let mut guard = self.responses.lock().await;
        guard.push_back(response);
    }
}

#[async_trait]
impl Model for ScriptedModel {
    fn name(&self) -> &str {
        &self.name
    }

    async fn request(
        &self,
        _messages: &[rustic_ai::ModelMessage],
        _settings: Option<&ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        let mut guard = self.responses.lock().await;
        guard.pop_front().ok_or_else(|| {
            ModelError::Unsupported("scripted model has no queued responses".to_string())
        })
    }
}
