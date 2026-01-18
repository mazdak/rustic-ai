use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rustic_ai::{
    Agent, FunctionTool, Model, ModelMessage, ModelRequestParameters, ModelResponse,
    ModelResponsePart, RunInput, SystemPromptPart, ToolCallPart, ToolReturnPart, UsageLimits,
    UserContent,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

struct SequenceModel {
    responses: Arc<Vec<ModelResponse>>,
    call_index: AtomicUsize,
    seen_messages: Arc<Mutex<Vec<Vec<ModelMessage>>>>,
}

impl SequenceModel {
    fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            responses: Arc::new(responses),
            call_index: AtomicUsize::new(0),
            seen_messages: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn seen_messages(&self) -> Arc<Mutex<Vec<Vec<ModelMessage>>>> {
        Arc::clone(&self.seen_messages)
    }
}

#[async_trait]
impl Model for SequenceModel {
    fn name(&self) -> &str {
        "sequence-model"
    }

    async fn request(
        &self,
        messages: &[ModelMessage],
        _settings: Option<&rustic_ai::ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, rustic_ai::model::ModelError> {
        self.seen_messages
            .lock()
            .expect("messages lock")
            .push(messages.to_vec());

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

fn has_system_prompt(messages: &[ModelMessage]) -> bool {
    messages.iter().any(|message| match message {
        ModelMessage::Request(request) => request.parts.iter().any(|part| {
            matches!(
                part,
                rustic_ai::ModelRequestPart::SystemPrompt(SystemPromptPart { .. })
            )
        }),
        _ => false,
    })
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

fn add_tool() -> FunctionTool<()> {
    FunctionTool::new("add", "add two numbers", |_, args: AddArgs| async move {
        Ok(AddResult {
            sum: args.a + args.b,
        })
    })
    .expect("tool creation should succeed")
}

#[tokio::test]
async fn includes_system_prompt_by_default() {
    let model = Arc::new(SequenceModel::new(vec![text_response("ok")]));
    let seen_messages = model.seen_messages();

    let agent = Agent::new(model).system_prompt("system");

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await.expect("run succeeds");
    assert_eq!(result.output, "ok");

    let messages = seen_messages.lock().expect("messages lock");
    let first = messages.first().expect("first call recorded");
    assert!(has_system_prompt(first));
}

#[tokio::test]
async fn allows_disabling_system_prompt() {
    let model = Arc::new(SequenceModel::new(vec![text_response("ok")]));
    let seen_messages = model.seen_messages();

    let agent = Agent::new(model).system_prompt("system");

    let mut input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );
    input.include_system_prompt = false;

    let result = agent.run(input).await.expect("run succeeds");
    assert_eq!(result.output, "ok");

    let messages = seen_messages.lock().expect("messages lock");
    let first = messages.first().expect("first call recorded");
    assert!(!has_system_prompt(first));
}

#[tokio::test]
async fn appends_tool_return_to_messages() {
    let model = Arc::new(SequenceModel::new(vec![
        tool_call_response("add", json!({"a": 1, "b": 2})),
        text_response("done"),
    ]));

    let mut agent = Agent::new(model);
    agent.tool(add_tool());

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await.expect("run succeeds");
    let returns = tool_returns(&result.messages);
    assert_eq!(returns.len(), 1);
    assert_eq!(returns[0].tool_name, "add");
    assert_eq!(returns[0].tool_call_id, "call-1");
}
