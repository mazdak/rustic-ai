use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::messages::{ModelMessage, UserContent};
use crate::model::Model;
use crate::usage::RunUsage;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolKind {
    Function,
    Output,
    External,
    Unapproved,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters_json_schema: Value,
    pub kind: ToolKind,
    pub sequential: bool,
    pub metadata: Option<Value>,
    pub timeout: Option<f64>,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: Option<String>,
        parameters_json_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description,
            parameters_json_schema,
            kind: ToolKind::Function,
            sequential: false,
            metadata: None,
            timeout: None,
        }
    }

    pub fn with_kind(mut self, kind: ToolKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn with_sequential(mut self, sequential: bool) -> Self {
        self.sequential = sequential;
        self
    }

    pub fn with_timeout(mut self, timeout: f64) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

#[derive(Clone)]
pub struct RunContext<Deps> {
    pub run_id: String,
    pub deps: Arc<Deps>,
    pub model: Arc<dyn Model>,
    pub usage: RunUsage,
    pub prompt: Option<Arc<Vec<UserContent>>>,
    pub messages: Arc<Vec<ModelMessage>>,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
}

impl<Deps> RunContext<Deps> {
    pub fn for_tool_call(&self, tool_call_id: String, tool_name: String) -> Self {
        Self {
            run_id: self.run_id.clone(),
            deps: Arc::clone(&self.deps),
            model: Arc::clone(&self.model),
            usage: self.usage.clone(),
            prompt: self.prompt.clone(),
            messages: Arc::clone(&self.messages),
            tool_call_id: Some(tool_call_id),
            tool_name: Some(tool_name),
        }
    }
}

#[async_trait]
pub trait Tool<Deps>: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    async fn call(&self, ctx: RunContext<Deps>, args: Value) -> Result<Value, ToolError>;
}

#[async_trait]
pub trait Toolset<Deps>: Send + Sync {
    async fn list_tools(&self, ctx: &RunContext<Deps>) -> Result<Vec<ToolDefinition>, ToolError>;

    async fn call_tool(
        &self,
        ctx: &RunContext<Deps>,
        name: &str,
        args: Value,
    ) -> Result<Value, ToolError>;

    async fn enter(&self) -> Result<(), ToolError> {
        Ok(())
    }

    async fn exit(&self) -> Result<(), ToolError> {
        Ok(())
    }

    fn name(&self) -> &str {
        "toolset"
    }
}

pub struct FunctionTool<Deps> {
    definition: ToolDefinition,
    handler: Arc<ToolHandler<Deps>>,
}

type ToolHandler<Deps> =
    dyn Fn(RunContext<Deps>, Value) -> BoxFuture<'static, Result<Value, ToolError>> + Send + Sync;

impl<Deps> FunctionTool<Deps>
where
    Deps: Send + Sync + 'static,
{
    pub fn new<Args, Output, Func, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        func: Func,
    ) -> Result<Self, ToolError>
    where
        Args: DeserializeOwned + JsonSchema + Send + 'static,
        Output: Serialize + Send + 'static,
        Func: Fn(RunContext<Deps>, Args) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Output, ToolError>> + Send + 'static,
    {
        let name = name.into();
        let description = Some(description.into());
        let schema = schemars::schema_for!(Args);
        let parameters_json_schema = serde_json::to_value(&schema).map_err(ToolError::Serde)?;

        let definition = ToolDefinition::new(name, description, parameters_json_schema);
        let func = Arc::new(func);
        let handler = Arc::new(move |ctx: RunContext<Deps>, args: Value| {
            let parsed = serde_json::from_value(args).map_err(ToolError::InvalidArgs);
            let func = Arc::clone(&func);
            let fut = async move {
                let parsed = parsed?;
                let output = func(ctx, parsed).await?;
                let value = serde_json::to_value(output).map_err(ToolError::Serde)?;
                Ok(value)
            };
            Box::pin(fut) as BoxFuture<'static, Result<Value, ToolError>>
        });

        Ok(Self {
            definition,
            handler,
        })
    }

    pub fn with_kind(mut self, kind: ToolKind) -> Self {
        self.definition.kind = kind;
        self
    }

    pub fn with_sequential(mut self, sequential: bool) -> Self {
        self.definition.sequential = sequential;
        self
    }

    pub fn with_timeout(mut self, timeout: f64) -> Self {
        self.definition.timeout = Some(timeout);
        self
    }
}

#[async_trait]
impl<Deps> Tool<Deps> for FunctionTool<Deps>
where
    Deps: Send + Sync + 'static,
{
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn call(&self, ctx: RunContext<Deps>, args: Value) -> Result<Value, ToolError> {
        (self.handler)(ctx, args).await
    }
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("invalid tool arguments: {0}")]
    InvalidArgs(serde_json::Error),
    #[error("tool execution failed: {0}")]
    Execution(String),
    #[error("serialization error: {0}")]
    Serde(serde_json::Error),
    #[error("toolset error: {0}")]
    Toolset(String),
}
