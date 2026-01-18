use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::future::BoxFuture;
use rustic_ai::{
    Agent, FunctionTool, Model, ModelMessage, ModelRequestParameters, ModelResponse,
    ModelResponsePart, RunContext, RunInput, ToolCallPart, ToolDefinition, ToolError,
    ToolReturnPart, Toolset, UsageLimits, UserContent,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

struct SequenceModel {
    responses: Arc<Vec<ModelResponse>>,
    call_index: AtomicUsize,
    seen_params: Arc<Mutex<Vec<ModelRequestParameters>>>,
}

impl SequenceModel {
    fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            responses: Arc::new(responses),
            call_index: AtomicUsize::new(0),
            seen_params: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn seen_params(&self) -> Arc<Mutex<Vec<ModelRequestParameters>>> {
        Arc::clone(&self.seen_params)
    }
}

#[async_trait]
impl Model for SequenceModel {
    fn name(&self) -> &str {
        "sequence-model"
    }

    async fn request(
        &self,
        _messages: &[ModelMessage],
        _settings: Option<&rustic_ai::ModelSettings>,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, rustic_ai::model::ModelError> {
        self.seen_params
            .lock()
            .expect("params lock")
            .push(params.clone());

        let index = self.call_index.fetch_add(1, Ordering::SeqCst);
        let response = if index >= self.responses.len() {
            self.responses
                .last()
                .cloned()
                .unwrap_or_else(|| text_response(""))
        } else {
            self.responses[index].clone()
        };
        Ok(response)
    }
}

fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        parts: vec![ModelResponsePart::Text(rustic_ai::TextPart {
            content: text.to_string(),
        })],
        usage: None,
        model_name: Some("sequence".to_string()),
        finish_reason: Some("stop".to_string()),
    }
}

fn tool_call_response(name: &str, args: Value) -> ModelResponse {
    ModelResponse {
        parts: vec![ModelResponsePart::ToolCall(ToolCallPart {
            id: "call-1".to_string(),
            name: name.to_string(),
            arguments: args,
        })],
        usage: None,
        model_name: Some("sequence".to_string()),
        finish_reason: Some("tool_call".to_string()),
    }
}

fn tool_returns(messages: &[ModelMessage]) -> Vec<ToolReturnPart> {
    let mut returns = Vec::new();
    for message in messages {
        if let ModelMessage::Request(request) = message {
            for part in &request.parts {
                if let rustic_ai::ModelRequestPart::ToolReturn(tool_return) = part {
                    returns.push(tool_return.clone());
                }
            }
        }
    }
    returns
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct AddArgs {
    a: i32,
    b: i32,
}

#[derive(Debug, Clone, Serialize)]
struct AddResult {
    sum: i32,
}

fn add_tool(name: &str) -> FunctionTool<()> {
    FunctionTool::new(name, "add two numbers", |_, args: AddArgs| async move {
        Ok(AddResult {
            sum: args.a + args.b,
        })
    })
    .expect("tool creation should succeed")
}

struct StaticToolset {
    name: String,
    tool_name: String,
    call_count: Arc<AtomicUsize>,
}

impl StaticToolset {
    fn new(name: &str, tool_name: &str, call_count: Arc<AtomicUsize>) -> Self {
        Self {
            name: name.to_string(),
            tool_name: tool_name.to_string(),
            call_count,
        }
    }
}

#[async_trait]
impl Toolset<()> for StaticToolset {
    async fn list_tools(&self, _ctx: &RunContext<()>) -> Result<Vec<ToolDefinition>, ToolError> {
        Ok(vec![ToolDefinition::new(
            self.tool_name.clone(),
            Some("toolset tool".to_string()),
            json!({"type": "object", "properties": {}}),
        )])
    }

    async fn call_tool(
        &self,
        _ctx: &RunContext<()>,
        _name: &str,
        _args: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"source": "toolset"}))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[tokio::test]
async fn prepare_tools_filters_defs() {
    let model = Arc::new(SequenceModel::new(vec![
        tool_call_response("allowed", json!({"a": 1, "b": 2})),
        text_response("done"),
    ]));

    let seen_params = model.seen_params();
    let mut agent = Agent::new(model);
    agent.tool(add_tool("allowed"));
    agent.tool(add_tool("blocked"));

    let prepare = Arc::new(|_ctx: &RunContext<()>, defs: Vec<ToolDefinition>| {
        let fut = async move {
            Ok(defs
                .into_iter()
                .filter(|def| def.name == "allowed")
                .collect())
        };
        Box::pin(fut) as BoxFuture<'static, Result<Vec<ToolDefinition>, ToolError>>
    });

    let agent = agent.prepare_tools(prepare);

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await.expect("run succeeds");
    assert_eq!(result.output, "done");

    let params = seen_params.lock().expect("params lock");
    let first = params.first().expect("params recorded");
    let tool_names: Vec<String> = first
        .function_tools
        .iter()
        .map(|d| d.name.clone())
        .collect();
    assert_eq!(tool_names, vec!["allowed".to_string()]);
}

#[tokio::test]
async fn tool_name_collision_prefers_local() {
    let model = Arc::new(SequenceModel::new(vec![
        tool_call_response("dup", json!({"a": 1, "b": 2})),
        text_response("done"),
    ]));

    let mut agent = Agent::new(model);
    agent.tool(add_tool("dup"));

    let call_count = Arc::new(AtomicUsize::new(0));
    let toolset = StaticToolset::new("remote", "dup", Arc::clone(&call_count));
    agent.toolset(toolset);

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await.expect("run succeeds");
    let returns = tool_returns(&result.messages);
    assert_eq!(returns.len(), 1);
    assert_eq!(returns[0].tool_name, "dup");
    assert_eq!(returns[0].content, json!({"sum": 3}));
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        0,
        "toolset should not be called"
    );
}
