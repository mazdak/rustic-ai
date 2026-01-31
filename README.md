# RusticAI

[![Crates.io](https://img.shields.io/crates/v/rustic-ai.svg)](https://crates.io/crates/rustic-ai)
[![Documentation](https://docs.rs/rustic-ai/badge.svg)](https://docs.rs/rustic-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Rust-native agent framework inspired by PydanticAI. RusticAI focuses on clear
abstractions, memory safety, and performance while keeping agent orchestration
and tool calling ergonomic.

## Features

- **Agent orchestration** with tool calling, usage limits, and message history
- **Multi-provider support** for OpenAI, Gemini, Anthropic, and Grok (XAI)
- **Streaming** with structured events (text deltas, tool calls)
- **Structured output** validation via JSON schema
- **Deferred tools** for approval flows and human-in-the-loop workflows
- **Tool execution controls** including timeouts and sequential execution
- **MCP toolsets** for remote tool integration (HTTP + SSE)
- **Instrumentation hooks** with tracing/OpenTelemetry support
- **Configurable failover** with pluggable model resolvers
- **Realtime support** for Grok voice agents (audio + tool calls)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustic-ai = "0.1"
```

## Quick Start

```rust
use rustic_ai::{Agent, RunInput, UsageLimits, UserContent, infer_model, infer_provider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a model (requires OPENAI_API_KEY environment variable)
    let model = infer_model("openai:gpt-4o-mini", infer_provider)?;

    // Build an agent with a system prompt
    let agent = Agent::new(model)
        .system_prompt("You are a helpful assistant.");

    // Run the agent
    let input = RunInput::new(
        vec![UserContent::Text("What is the capital of France?".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await?;
    println!("{}", result.output);

    Ok(())
}
```

## Adding Tools

Tools are defined with typed arguments using `serde` and `schemars`:

```rust
use rustic_ai::FunctionTool;
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

let tool = FunctionTool::new("add", "Add two numbers", |_ctx, args: AddArgs| async move {
    Ok(AddResult { sum: args.a + args.b })
})?;

let mut agent = Agent::new(model).system_prompt("You can do math.");
agent.tool(tool);
```

## Providers

RusticAI supports multiple LLM providers:

| Provider | Environment Variable | Example Model |
|----------|---------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | `openai:gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic:claude-sonnet-4-5` |
| Gemini | `GEMINI_API_KEY` | `gemini:gemini-2.0-flash` |
| Grok | `XAI_API_KEY` | `grok:grok-3-mini-fast` |

```rust
// Use any supported provider
let openai = infer_model("openai:gpt-4o", infer_provider)?;
let anthropic = infer_model("anthropic:claude-sonnet-4-5", infer_provider)?;
let gemini = infer_model("gemini:gemini-2.0-flash", infer_provider)?;
let grok = infer_model("grok:grok-3-mini-fast", infer_provider)?;
```

## Streaming

Stream responses with text deltas and tool call events:

```rust
use futures::StreamExt;
use rustic_ai::agent::AgentStreamEvent;

let mut stream = agent.run_stream(input).await?;

while let Some(event) = stream.next().await {
    match event? {
        AgentStreamEvent::TextDelta(text) => print!("{text}"),
        AgentStreamEvent::ToolCall(call) => println!("\n[Tool: {}]", call.name),
        AgentStreamEvent::Done(result) => println!("\n\nDone: {}", result.output),
    }
}
```

## Telemetry (Optional)

Enable OpenTelemetry or Datadog exporters with feature flags:

```toml
[dependencies]
rustic-ai = { version = "0.1", features = ["telemetry-otel"] }
# or: features = ["telemetry-datadog"]
```

```rust
use opentelemetry_otlp::Protocol;
use rustic_ai::telemetry::init_otlp_tracing;

let _guard = init_otlp_tracing("my-service", Protocol::Grpc, None, None)?;
```

## Documentation

- [API Documentation](https://docs.rs/rustic-ai) - Full API reference
- [Development Guide](DEVELOPMENT.md) - Architecture, internals, and contributor guide

## License

MIT License - see [LICENSE](LICENSE) for details.
