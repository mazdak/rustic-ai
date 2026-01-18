use std::pin::Pin;

use async_trait::async_trait;
use futures::stream::Stream;
use serde_json::{Map, Value};
use thiserror::Error;

use crate::messages::{ModelMessage, ModelResponse, ToolCallPart};
use crate::tools::ToolDefinition;
use crate::usage::RequestUsage;

pub type ModelSettings = Map<String, Value>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum OutputMode {
    #[default]
    Text,
    JsonSchema,
}

#[derive(Clone, Debug)]
pub struct ModelRequestParameters {
    pub function_tools: Vec<ToolDefinition>,
    pub output_schema: Option<Value>,
    pub output_mode: OutputMode,
    pub allow_text_output: bool,
}

impl ModelRequestParameters {
    pub fn new(function_tools: Vec<ToolDefinition>) -> Self {
        Self {
            function_tools,
            output_schema: None,
            output_mode: OutputMode::Text,
            allow_text_output: true,
        }
    }

    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self.output_mode = OutputMode::JsonSchema;
        self.allow_text_output = false;
        self
    }
}

impl Default for ModelRequestParameters {
    fn default() -> Self {
        Self {
            function_tools: Vec::new(),
            output_schema: None,
            output_mode: OutputMode::Text,
            allow_text_output: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StreamChunk {
    pub text_delta: Option<String>,
    pub tool_call: Option<ToolCallPart>,
    pub finish_reason: Option<String>,
    pub usage: Option<RequestUsage>,
}

pub type ModelStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, ModelError>> + Send>>;

#[async_trait]
pub trait Model: Send + Sync {
    fn name(&self) -> &str;

    async fn request(
        &self,
        messages: &[ModelMessage],
        settings: Option<&ModelSettings>,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError>;

    async fn count_tokens(
        &self,
        _messages: &[ModelMessage],
        _settings: Option<&ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<RequestUsage, ModelError> {
        Err(ModelError::Unsupported(
            "token counting not supported".to_string(),
        ))
    }

    async fn request_stream(
        &self,
        _messages: &[ModelMessage],
        _settings: Option<&ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<ModelStream, ModelError> {
        Err(ModelError::Unsupported(
            "streaming not supported".to_string(),
        ))
    }
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("provider error: {0}")]
    Provider(String),
    #[error("http error status: {status}")]
    HttpStatus { status: u16 },
    #[error("transport error: {0}")]
    Transport(String),
    #[error("timeout error")]
    Timeout,
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
