use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use tokio::sync::Mutex;

use rustic_ai::{
    Model, ModelError, ModelRequestParameters, ModelResponse, ModelSettings, ModelStream,
    StreamChunk,
};

#[derive(Debug)]
pub struct StreamedModel {
    name: String,
    chunks: Arc<Mutex<VecDeque<Result<StreamChunk, ModelError>>>>,
}

impl StreamedModel {
    pub fn new(name: impl Into<String>, chunks: Vec<Result<StreamChunk, ModelError>>) -> Self {
        Self {
            name: name.into(),
            chunks: Arc::new(Mutex::new(VecDeque::from(chunks))),
        }
    }

    pub async fn push_chunk(&self, chunk: Result<StreamChunk, ModelError>) {
        let mut guard = self.chunks.lock().await;
        guard.push_back(chunk);
    }
}

#[async_trait]
impl Model for StreamedModel {
    fn name(&self) -> &str {
        &self.name
    }

    async fn request(
        &self,
        _messages: &[rustic_ai::ModelMessage],
        _settings: Option<&ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        Err(ModelError::Unsupported(
            "streamed model does not support non-streaming requests".to_string(),
        ))
    }

    async fn request_stream(
        &self,
        _messages: &[rustic_ai::ModelMessage],
        _settings: Option<&ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<ModelStream, ModelError> {
        let mut guard = self.chunks.lock().await;
        let items: Vec<_> = guard.drain(..).collect();
        Ok(Box::pin(stream::iter(items)))
    }
}
