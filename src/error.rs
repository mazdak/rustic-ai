use thiserror::Error;

use crate::usage::UsageError;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("model error: {0}")]
    Model(#[from] crate::model::ModelError),
    #[error("tool error: {0}")]
    Tool(#[from] crate::tools::ToolError),
    #[error("usage limit exceeded: {0}")]
    Usage(#[from] UsageError),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("output validation failed: {0}")]
    OutputValidation(String),
    #[error("invalid configuration: {0}")]
    Config(String),
}
