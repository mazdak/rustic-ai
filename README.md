# RusticAI

RusticAI is a Rust-native agent framework inspired by PydanticAI. It focuses on
clear abstractions, memory safety, and performance while keeping agent orchestration
and tool calling ergonomic.

## Highlights

- **Agent orchestration** with tool calling, usage limits, and message history
- **Providers** for OpenAI, Gemini, Anthropic, and Grok (XAI)
- **Streaming** support with structured events
- **Structured output** validation (JSON schema, strict by default)
- **Deferred tools** for approval flows and external tooling
- **Tool execution controls** (timeouts + sequential execution)
- **MCP toolsets** (HTTP + SSE extras)
- **Instrumentation hooks** (Tracing/OTEL-compatible telemetry)
- **Configurable failover** with pluggable resolvers
- **Realtime Grok helpers** for voice agents (audio + tool calls)

## Provider notes

- OpenAI defaults to the Responses API; audio inputs and streaming fall back to Chat Completions when supported by the model.
- Grok uses the OpenAI-compatible Chat Completions API for tool calling.
- Structured output schemas are normalized for OpenAI-compatible strict mode where supported.
- Grok vision expects image URLs; binary image inputs are rejected.

## Install

```toml
[dependencies]
rustic-ai = { path = "../rustic-ai" }
```

## Canonical example

```rust
use std::sync::Arc;

use async_trait::async_trait;
use rustic_ai::{
    Agent, FunctionTool, Model, ModelMessage, ModelRequestParameters, ModelResponse,
    ModelResponsePart, RunInput, UsageLimits, UserContent,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema)]
struct AddArgs {
    a: i64,
    b: i64,
}

#[derive(Serialize)]
struct AddResult {
    sum: i64,
}

struct EchoModel;

#[async_trait]
impl Model for EchoModel {
    fn name(&self) -> &str {
        "echo"
    }

    async fn request(
        &self,
        _messages: &[ModelMessage],
        _settings: Option<&rustic_ai::ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, rustic_ai::model::ModelError> {
        Ok(ModelResponse {
            parts: vec![ModelResponsePart::Text(rustic_ai::TextPart {
                content: "hello".to_string(),
            })],
            usage: None,
            model_name: Some("echo".to_string()),
            finish_reason: Some("stop".to_string()),
        })
    }
}

#[tokio::main]
async fn main() {
    let model = Arc::new(EchoModel);
    let mut agent = Agent::new(model).system_prompt("You are helpful.");

    let tool = FunctionTool::new("add", "add two numbers", |_, args: AddArgs| async move {
        Ok(AddResult { sum: args.a + args.b })
    })
    .expect("tool creation should succeed");

    agent.tool(tool);

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await.expect("run succeeds");
    println!("{}", result.output);
}
```

For detailed developer documentation, see `DEVELOPMENT.md`.
