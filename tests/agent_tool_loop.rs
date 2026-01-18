use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use rustic_ai::{
    Agent, FunctionTool, Model, ModelMessage, ModelRequestParameters, ModelResponse,
    ModelResponsePart, RequestUsage, RunInput, ToolCallPart, ToolError, UsageLimits, UserContent,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::time::{Duration, sleep};

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct AddArgs {
    a: i32,
    b: i32,
}

#[derive(Debug, Clone, Serialize)]
struct AddResult {
    sum: i32,
}

struct StepModel {
    step: AtomicUsize,
}

impl StepModel {
    fn new() -> Self {
        Self {
            step: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl Model for StepModel {
    fn name(&self) -> &str {
        "step-model"
    }

    async fn request(
        &self,
        _messages: &[ModelMessage],
        _settings: Option<&rustic_ai::ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, rustic_ai::model::ModelError> {
        let step = self.step.fetch_add(1, Ordering::SeqCst);
        if step == 0 {
            Ok(ModelResponse {
                parts: vec![ModelResponsePart::ToolCall(ToolCallPart {
                    id: "call-1".to_string(),
                    name: "add".to_string(),
                    arguments: json!({"a": 1, "b": 2}),
                })],
                usage: Some(RequestUsage {
                    input_tokens: 3,
                    output_tokens: 1,
                    ..Default::default()
                }),
                model_name: Some("step-model".to_string()),
                finish_reason: Some("tool_call".to_string()),
            })
        } else {
            Ok(ModelResponse {
                parts: vec![ModelResponsePart::Text(rustic_ai::TextPart {
                    content: "done".to_string(),
                })],
                usage: Some(RequestUsage {
                    input_tokens: 2,
                    output_tokens: 2,
                    ..Default::default()
                }),
                model_name: Some("step-model".to_string()),
                finish_reason: Some("stop".to_string()),
            })
        }
    }
}

#[tokio::test]
async fn agent_executes_tool_and_returns_text() {
    let model = Arc::new(StepModel::new());
    let mut agent = Agent::new(model);

    let tool = FunctionTool::new("add", "add two numbers", |_, args: AddArgs| async move {
        Ok(AddResult {
            sum: args.a + args.b,
        })
    })
    .expect("tool creation should succeed");

    agent.tool(tool);

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits {
            request_limit: Some(5),
            ..Default::default()
        },
    );

    let result = agent.run(input).await.expect("agent run succeeds");
    assert_eq!(result.output, "done");
    assert_eq!(result.usage.requests, 2);
    assert_eq!(result.usage.tool_calls, 1);
}

#[tokio::test]
async fn tool_context_includes_latest_response() {
    let model = Arc::new(StepModel::new());
    let mut agent = Agent::new(model);

    let tool = FunctionTool::new("add", "add two numbers", |ctx, args: AddArgs| async move {
        let has_tool_call = ctx.messages.iter().any(|message| match message {
            ModelMessage::Response(response) => response
                .tool_calls()
                .iter()
                .any(|call| call.id == "call-1" && call.name == "add"),
            _ => false,
        });
        if !has_tool_call {
            return Err(ToolError::Execution(
                "tool context missing latest tool call".to_string(),
            ));
        }
        Ok(AddResult {
            sum: args.a + args.b,
        })
    })
    .expect("tool creation should succeed");

    agent.tool(tool);

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await.expect("agent run succeeds");
    assert_eq!(result.output, "done");
}

#[tokio::test]
async fn tool_timeout_returns_error() {
    let model = Arc::new(StepModel::new());
    let mut agent = Agent::new(model);

    let tool = FunctionTool::new("add", "add two numbers", |_, _args: AddArgs| async move {
        sleep(Duration::from_millis(50)).await;
        Ok(AddResult { sum: 3 })
    })
    .expect("tool creation should succeed")
    .with_timeout(0.01);

    agent.tool(tool);

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await;
    match result {
        Err(rustic_ai::AgentError::Tool(ToolError::Execution(_))) => {}
        other => panic!("expected tool timeout error, got: {other:?}"),
    }
}
