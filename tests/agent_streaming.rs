#[path = "support/streamed_model.rs"]
mod streamed_support;

use std::sync::Arc;

use futures::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use rustic_ai::{
    Agent, AgentRunState, AgentStreamEvent, FunctionTool, RequestUsage, RunInput, StreamChunk,
    UsageLimits, UserContent,
};
use streamed_support::StreamedModel;

#[tokio::test]
async fn streaming_text_emits_deltas_and_done() {
    let model = Arc::new(StreamedModel::new("stream-model", Vec::new()));
    model
        .push_chunk(Ok(StreamChunk {
            text_delta: Some("he".to_string()),
            tool_call: None,
            finish_reason: None,
            usage: None,
        }))
        .await;
    model
        .push_chunk(Ok(StreamChunk {
            text_delta: Some("llo".to_string()),
            tool_call: None,
            finish_reason: Some("stop".to_string()),
            usage: Some(RequestUsage {
                input_tokens: 3,
                output_tokens: 2,
                ..Default::default()
            }),
        }))
        .await;

    let agent = Agent::new(model);
    let input = RunInput::new(
        vec![UserContent::Text("hi".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let mut stream = agent.run_stream(input).await.expect("stream starts");
    let mut deltas = Vec::new();
    let mut done = None;

    while let Some(event) = stream.next().await {
        match event.expect("stream event") {
            AgentStreamEvent::TextDelta(delta) => deltas.push(delta),
            AgentStreamEvent::Done(result) => done = Some(result),
            AgentStreamEvent::ToolCall(_) => {}
        }
    }

    assert_eq!(deltas.join(""), "hello");
    let result = done.expect("done event");
    assert_eq!(result.output, "hello");
    assert_eq!(result.usage.requests, 1);
    assert_eq!(result.usage.input_tokens, 3);
    assert_eq!(result.usage.output_tokens, 2);
    assert_eq!(result.state, AgentRunState::Completed);
}

#[derive(Debug, Deserialize, JsonSchema)]
struct PingArgs {}

#[derive(Debug, Serialize)]
struct PingResult {
    ok: bool,
}

#[tokio::test]
async fn streaming_tool_call_is_deferred() {
    let model = Arc::new(StreamedModel::new("stream-tool-model", Vec::new()));
    model
        .push_chunk(Ok(StreamChunk {
            text_delta: None,
            tool_call: Some(rustic_ai::ToolCallPart {
                id: "call-1".to_string(),
                name: "ping".to_string(),
                arguments: json!({}),
            }),
            finish_reason: Some("tool_call".to_string()),
            usage: Some(RequestUsage {
                input_tokens: 1,
                output_tokens: 0,
                ..Default::default()
            }),
        }))
        .await;

    let tool = FunctionTool::new("ping", "ping tool", |_ctx, _args: PingArgs| async move {
        Ok(PingResult { ok: true })
    })
    .expect("tool creation");

    let mut agent = Agent::new(model);
    agent.tool(tool);

    let input = RunInput::new(
        vec![UserContent::Text("ping".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let mut stream = agent.run_stream(input).await.expect("stream starts");
    let mut saw_tool_call = false;
    let mut done = None;

    while let Some(event) = stream.next().await {
        match event.expect("stream event") {
            AgentStreamEvent::ToolCall(call) => {
                assert_eq!(call.name, "ping");
                saw_tool_call = true;
            }
            AgentStreamEvent::Done(result) => done = Some(result),
            AgentStreamEvent::TextDelta(_) => {}
        }
    }

    assert!(saw_tool_call);
    let result = done.expect("done event");
    assert_eq!(result.deferred_calls.len(), 1);
    assert_eq!(result.deferred_calls[0].tool_name, "ping");
    assert_eq!(result.state, AgentRunState::Deferred);
}
