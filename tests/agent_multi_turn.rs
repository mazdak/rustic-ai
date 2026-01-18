use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use rustic_ai::messages::ModelRequestPart;
use rustic_ai::model::ModelError;
use rustic_ai::{
    Agent, FunctionTool, Model, ModelMessage, ModelRequestParameters, ModelResponse,
    ModelResponsePart, RunInput, ToolCallPart, UsageLimits, UserContent,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct AddArgs {
    a: i32,
    b: i32,
}

#[derive(Debug, Clone, Serialize)]
struct AddResult {
    sum: i32,
}

struct MultiTurnModel {
    step: AtomicUsize,
}

impl MultiTurnModel {
    fn new() -> Self {
        Self {
            step: AtomicUsize::new(0),
        }
    }

    fn ensure_tool_history(messages: &[ModelMessage]) -> Result<(), ModelError> {
        let mut tool_call_index = None;
        let mut tool_return_index = None;

        for (idx, message) in messages.iter().enumerate() {
            match message {
                ModelMessage::Response(response) => {
                    if response
                        .tool_calls()
                        .iter()
                        .any(|call| call.id == "call-1" && call.name == "add")
                    {
                        tool_call_index = Some(idx);
                    }
                }
                ModelMessage::Request(request) => {
                    let has_tool_return = request.parts.iter().any(|part| match part {
                        ModelRequestPart::ToolReturn(tool_return) => {
                            tool_return.tool_call_id == "call-1" && tool_return.tool_name == "add"
                        }
                        _ => false,
                    });
                    if has_tool_return {
                        tool_return_index = Some(idx);
                    }
                }
            }
        }

        let call_index = tool_call_index.ok_or_else(|| {
            ModelError::Provider("missing tool call in message history".to_string())
        })?;
        let return_index = tool_return_index.ok_or_else(|| {
            ModelError::Provider("missing tool return in message history".to_string())
        })?;
        if call_index >= return_index {
            return Err(ModelError::Provider(
                "tool return appears before tool call".to_string(),
            ));
        }
        Ok(())
    }
}

#[async_trait]
impl Model for MultiTurnModel {
    fn name(&self) -> &str {
        "multi-turn-model"
    }

    async fn request(
        &self,
        messages: &[ModelMessage],
        _settings: Option<&rustic_ai::ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        let step = self.step.fetch_add(1, Ordering::SeqCst);
        match step {
            0 => Ok(ModelResponse {
                parts: vec![ModelResponsePart::ToolCall(ToolCallPart {
                    id: "call-1".to_string(),
                    name: "add".to_string(),
                    arguments: json!({"a": 1, "b": 2}),
                })],
                usage: None,
                model_name: Some("multi-turn-model".to_string()),
                finish_reason: Some("tool_call".to_string()),
            }),
            1 => Ok(ModelResponse {
                parts: vec![ModelResponsePart::Text(rustic_ai::TextPart {
                    content: "done".to_string(),
                })],
                usage: None,
                model_name: Some("multi-turn-model".to_string()),
                finish_reason: Some("stop".to_string()),
            }),
            _ => {
                Self::ensure_tool_history(messages)?;
                Ok(ModelResponse {
                    parts: vec![ModelResponsePart::Text(rustic_ai::TextPart {
                        content: "second".to_string(),
                    })],
                    usage: None,
                    model_name: Some("multi-turn-model".to_string()),
                    finish_reason: Some("stop".to_string()),
                })
            }
        }
    }
}

#[tokio::test]
async fn multi_turn_preserves_tool_history() {
    let model: Arc<dyn Model> = Arc::new(MultiTurnModel::new());
    let mut agent = Agent::new(Arc::clone(&model));

    let tool = FunctionTool::new("add", "add two numbers", |_, args: AddArgs| async move {
        Ok(AddResult {
            sum: args.a + args.b,
        })
    })
    .expect("tool creation should succeed");
    agent.tool(tool);

    let input = RunInput::new(
        vec![UserContent::Text("first".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );
    let result = agent.run(input).await.expect("first run succeeds");
    assert_eq!(result.output, "done");

    let input = RunInput::new(
        vec![UserContent::Text("second".to_string())],
        result.messages,
        (),
        UsageLimits::default(),
    );
    let result = agent.run(input).await.expect("second run succeeds");
    assert_eq!(result.output, "second");
}
