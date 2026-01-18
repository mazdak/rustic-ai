use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use rustic_ai::{
    Agent, AgentError, FunctionTool, Model, ModelMessage, ModelRequestParameters, ModelResponse,
    ModelResponsePart, RequestUsage, RunInput, ToolCallPart, UsageError, UsageLimits, UserContent,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

struct SequenceModel {
    responses: Arc<Vec<ModelResponse>>,
    call_index: AtomicUsize,
}

impl SequenceModel {
    fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            responses: Arc::new(responses),
            call_index: AtomicUsize::new(0),
        }
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
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, rustic_ai::model::ModelError> {
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

fn tool_call_response_with_usage(
    name: &str,
    args: Value,
    input_tokens: u64,
    output_tokens: u64,
) -> ModelResponse {
    ModelResponse {
        parts: vec![ModelResponsePart::ToolCall(ToolCallPart {
            id: "call-1".to_string(),
            name: name.to_string(),
            arguments: args,
        })],
        usage: Some(RequestUsage {
            input_tokens,
            output_tokens,
            ..Default::default()
        }),
        model_name: Some("sequence".to_string()),
        finish_reason: Some("tool_call".to_string()),
    }
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

fn add_tool() -> FunctionTool<()> {
    FunctionTool::new("add", "add two numbers", |_, args: AddArgs| async move {
        Ok(AddResult {
            sum: args.a + args.b,
        })
    })
    .expect("tool creation should succeed")
}

#[tokio::test]
async fn request_limit_exceeded_on_tool_loop() {
    let model = Arc::new(SequenceModel::new(vec![
        tool_call_response_with_usage("add", json!({"a": 1, "b": 2}), 2, 1),
        text_response("done"),
    ]));

    let mut agent = Agent::new(model);
    agent.tool(add_tool());

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits {
            request_limit: Some(1),
            ..Default::default()
        },
    );

    let err = match agent.run(input).await {
        Ok(_) => panic!("expected error"),
        Err(err) => err,
    };
    match err {
        AgentError::Usage(UsageError::RequestLimitExceeded { .. }) => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn tool_call_limit_exceeded() {
    let model = Arc::new(SequenceModel::new(vec![tool_call_response(
        "add",
        json!({"a": 1, "b": 2}),
    )]));

    let mut agent = Agent::new(model);
    agent.tool(add_tool());

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits {
            tool_calls_limit: Some(0),
            ..Default::default()
        },
    );

    let err = match agent.run(input).await {
        Ok(_) => panic!("expected error"),
        Err(err) => err,
    };
    match err {
        AgentError::Usage(UsageError::ToolCallsLimitExceeded { .. }) => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn total_token_limit_exceeded() {
    let model = Arc::new(SequenceModel::new(vec![tool_call_response_with_usage(
        "add",
        json!({"a": 1, "b": 2}),
        3,
        3,
    )]));

    let mut agent = Agent::new(model);
    agent.tool(add_tool());

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits {
            total_tokens_limit: Some(5),
            ..Default::default()
        },
    );

    let err = match agent.run(input).await {
        Ok(_) => panic!("expected error"),
        Err(err) => err,
    };
    match err {
        AgentError::Usage(UsageError::TotalTokensLimitExceeded { .. }) => {}
        other => panic!("unexpected error: {other:?}"),
    }
}
