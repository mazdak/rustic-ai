use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use rustic_ai::{
    Agent, Model, ModelMessage, ModelRequestParameters, ModelResponse, ModelResponsePart,
    RunContext, RunInput, ToolCallPart, ToolDefinition, ToolError, Toolset, UsageLimits,
    UserContent,
};
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

struct CountingToolset {
    name: String,
    tool_name: String,
    call_count: Arc<AtomicUsize>,
}

impl CountingToolset {
    fn new(name: &str, tool_name: &str, call_count: Arc<AtomicUsize>) -> Self {
        Self {
            name: name.to_string(),
            tool_name: tool_name.to_string(),
            call_count,
        }
    }
}

#[async_trait]
impl Toolset<()> for CountingToolset {
    async fn list_tools(&self, _ctx: &RunContext<()>) -> Result<Vec<ToolDefinition>, ToolError> {
        Ok(vec![ToolDefinition::new(
            self.tool_name.clone(),
            Some("counting tool".to_string()),
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
        Ok(json!({"ok": true}))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[tokio::test]
async fn toolset_call_executes_and_usage_increments() {
    let model = Arc::new(SequenceModel::new(vec![
        tool_call_response("remote", json!({})),
        text_response("done"),
    ]));

    let mut agent = Agent::new(model);
    let call_count = Arc::new(AtomicUsize::new(0));
    let toolset = CountingToolset::new("remote", "remote", Arc::clone(&call_count));
    agent.toolset(toolset);

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await.expect("run succeeds");
    assert_eq!(result.output, "done");
    assert_eq!(call_count.load(Ordering::SeqCst), 1);
    assert_eq!(result.usage.tool_calls, 1);
}
