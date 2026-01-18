use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use rustic_ai::{
    Agent, FunctionTool, Model, ModelMessage, ModelRequestParameters, ModelResponse,
    ModelResponsePart, RequestUsage, RunInput, ToolCallPart, UsageLimits, UserContent,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

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

fn text_response_with_usage(text: &str, input_tokens: u64, output_tokens: u64) -> ModelResponse {
    ModelResponse {
        parts: vec![ModelResponsePart::Text(rustic_ai::TextPart {
            content: text.to_string(),
        })],
        usage: Some(RequestUsage {
            input_tokens,
            output_tokens,
            ..Default::default()
        }),
        model_name: Some("sequence".to_string()),
        finish_reason: Some("stop".to_string()),
    }
}

fn tool_call_response_with_usage(
    name: &str,
    args: serde_json::Value,
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
async fn usage_requests_increments_without_usage_payload() {
    let model = Arc::new(SequenceModel::new(vec![text_response("ok")]));
    let agent = Agent::new(model);

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await.expect("run succeeds");
    assert_eq!(result.usage.requests, 1);
}

#[tokio::test]
async fn usage_accumulates_tokens_from_payload() {
    let model = Arc::new(SequenceModel::new(vec![
        tool_call_response_with_usage("add", json!({"a": 1, "b": 2}), 3, 2),
        text_response_with_usage("ok", 1, 4),
    ]));
    let mut agent = Agent::new(model);
    agent.tool(add_tool());

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits {
            request_limit: Some(2),
            ..Default::default()
        },
    );

    let result = agent.run(input).await.expect("run succeeds");
    assert_eq!(result.usage.requests, 2);
    assert_eq!(result.usage.input_tokens, 4);
    assert_eq!(result.usage.output_tokens, 6);
}
