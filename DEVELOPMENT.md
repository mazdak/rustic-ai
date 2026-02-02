# RusticAI Development Guide

This document describes the full feature set and internal architecture of RusticAI,
with practical guidance for contributors and advanced users.

## Table of contents

- Overview
- Core concepts
- Agent
- Model interface
- Providers
- Messages
- Tools
- Deferred tool flow
- Streaming
- Usage limits and accounting
- Structured output
- Failover and config resolvers
- MCP toolsets (HTTP + SSE extras)
- Instrumentation
- Errors
- Testing and linting

## Overview

RusticAI is an agent framework built around a small set of extensible traits:

- `Model`: makes LLM requests (sync and streaming)
- `Tool`: typed local tool callable by the agent
- `Toolset`: remote tool registry (e.g., MCP)
- `Instrumenter`: hooks for tracing/telemetry
- `ModelConfigResolver`: resolves primary/backup models + settings

The `Agent` orchestrates tool calling, manages message history, enforces usage
limits, validates structured output, and returns a detailed `AgentRunResult`.

Example:

```rust
use rustic_ai::{Agent, RunInput, UsageLimits, UserContent, infer_model, infer_provider};

let model = infer_model("openai:gpt-4o-mini", infer_provider)?;
let agent = Agent::new(model).system_prompt("Be helpful.");
let input = RunInput::new(
    vec![UserContent::Text("Hello!".to_string())],
    vec![],
    (),
    UsageLimits::default(),
);
let result = agent.run(input).await?;
println!("{}", result.output);
```

Type-state builder alternative:

```rust
let input = RunInput::builder(())
    .user_text("Hello!")
    .usage_limits(UsageLimits::default())
    .build();
```

## Core concepts

Example:

```rust
use rustic_ai::{Agent, infer_model, infer_provider};

let model = infer_model("openai:gpt-4o-mini", infer_provider)?;
let _agent = Agent::new(model);
```

### Agent

`Agent<Deps>` owns:

- a `Model` instance
- tools and toolsets
- system prompt
- model settings
- usage limits and output validation configuration

Key entry points:

- `run`: standard single-call run loop
- `run_stream`: streaming output (text/tool calls)
- `run_with_failover`: uses a resolver to select primary/backup models

The run loop:

1. Build messages (system prompt + history + user prompt)
2. Call the model
3. If tool calls are returned, execute them
4. Append tool results and continue
5. Validate output schema (if configured)

Example:

```rust
let input = RunInput::new(
    vec![UserContent::Text("Summarize this.".to_string())],
    vec![],
    (),
    UsageLimits::default(),
);
let result = agent.run(input).await?;
println!("{}", result.output);
```

### Model interface

```rust
#[async_trait]
pub trait Model: Send + Sync {
    fn name(&self) -> &str;

    async fn request(
        &self,
        messages: &[ModelMessage],
        settings: Option<&ModelSettings>,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError>;

    async fn request_stream(
        &self,
        messages: &[ModelMessage],
        settings: Option<&ModelSettings>,
        params: &ModelRequestParameters,
    ) -> Result<ModelStream, ModelError>;
}
```

Example:

```rust
struct EchoModel;

#[async_trait]
impl Model for EchoModel {
    fn name(&self) -> &str {
        "echo"
    }

    async fn request(
        &self,
        _messages: &[ModelMessage],
        _settings: Option<&ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        Ok(ModelResponse {
            parts: vec![ModelResponsePart::Text(TextPart {
                content: "ok".to_string(),
            })],
            usage: None,
            model_name: Some("echo".to_string()),
            finish_reason: Some("stop".to_string()),
        })
    }
}
```

`ModelSettings` is a JSON map, allowing provider-specific parameters without
hard-coding every field.

## Providers

Built-in providers live in `src/providers`:

- **OpenAI** (`OPENAI_API_KEY`) – Responses API (preferred) with Chat Completions fallback for audio input and streaming
- **Grok** (`XAI_API_KEY` or `GROK_API_KEY`) – OpenAI-compatible Chat Completions API
- **Anthropic** (`ANTHROPIC_API_KEY`) – Messages API
- **Gemini** (`GEMINI_API_KEY` or `GOOGLE_API_KEY`) – GenerateContent API

`infer_provider` and `infer_model` resolve provider + model names:

```rust
let model = rustic_ai::infer_model("openai:gpt-4o-mini", rustic_ai::infer_provider)?;
```

Example (provider-specific settings):

```rust
use serde_json::json;

let model = rustic_ai::infer_model("openai:gpt-4o-mini", rustic_ai::infer_provider)?;
let settings = json!({ "temperature": 0.2 })
    .as_object()
    .expect("settings object")
    .clone();
let _agent = Agent::new(model).model_settings(settings);
```

Providers serialize:

- system/user messages
- tool definitions
- tool calls and tool results
- media inputs (images, audio, documents)

### Media serialization

- OpenAI Responses: uses `input_text`, `input_image`, and `input_file` (data URLs for binary images/PDFs)
- OpenAI/Grok Chat: uses `image_url` (data URLs for binary images) and `input_audio` for audio binaries/data URLs
- OpenAI Chat fallback is used when audio input is present or when streaming is requested (Responses does not accept audio inputs or streaming)
- Anthropic: base64-encodes binary images/PDFs; text-like binaries are inlined as text
- Gemini: uses `inlineData` for binaries and `fileData` for URLs (with MIME inference from URL when missing)

Example:

```rust
use rustic_ai::{BinaryContent, RunInput, UsageLimits, UserContent};

let input = RunInput::new(
    vec![
        UserContent::Binary(BinaryContent {
            data: vec![0_u8; 16],
            media_type: "image/png".to_string(),
        }),
        UserContent::Text("What is this?".to_string()),
    ],
    vec![],
    (),
    UsageLimits::default(),
);
```

### Realtime (voice)

`rustic-ai` provides a Grok Realtime client (`realtime::grok`) for voice agents. It supports:

- WebSocket session setup (voice, tools, temperature, audio format)
- Audio input/output events
- Function-call tool events + helpers to convert to `ToolCallPart`

Minimal usage sketch:

```rust
use rustic_ai::realtime::grok::{GrokClient, SessionConfig};

let client = GrokClient::new(ws_url, api_key);
let session = SessionConfig::new("You are a helpful voice agent.");
let (sender, mut events) = client.connect(session).await?;
sender.send_user_text("Hello".to_string()).await?;
sender.request_response(Some(vec!["text".to_string(), "audio".to_string()])).await?;
```

## Messages

Key types in `src/messages.rs`:

- `ModelMessage`: request/response wrapper
- `ModelRequestPart`: system/user prompt, tool return, retry prompt
- `ModelResponsePart`: text, tool call, or provider-specific items
- `UserContent`: text, image, audio, video, document, binary

Example:

```rust
use rustic_ai::{ModelMessage, ModelRequest, ModelRequestPart, UserContent, UserPromptPart};

let message = ModelMessage::Request(ModelRequest {
    parts: vec![ModelRequestPart::UserPrompt(UserPromptPart {
        content: vec![UserContent::Text("Hello".to_string())],
    })],
    instructions: None,
});
```

## Tools

Tools are strongly typed via `serde` and `schemars`:

```rust
let tool = FunctionTool::new("add", "add two numbers", |ctx, args: AddArgs| async move {
    Ok(AddResult { sum: args.a + args.b })
})?;
```

Example (register a tool on an agent):

```rust
let mut agent = Agent::new(model);
agent.tool(tool);
```

Tools return JSON values and are automatically embedded into `ToolReturnPart`.

### Execution controls

Each tool definition supports:

- `sequential`: forces tool calls to execute one-at-a-time (and disables provider parallel tool calls)
- `timeout`: per-tool timeout in seconds (timeout produces a tool error)

Example:

```rust
let tool = tool.with_sequential(true).with_timeout(15.0);
```

### Tool kinds

`ToolKind` controls execution behavior:

- `Function`: normal local or MCP tool execution
- `External`: returned as deferred (requires approval)
- `Unapproved`: returned as deferred

Example:

```rust
let tool = tool.with_kind(ToolKind::External);
```

## Deferred tool flow

When the model emits a tool call for `External` or `Unapproved` tools, the agent
returns an `AgentRunResult` with:

- `state = AgentRunState::Deferred`
- `deferred_calls` listing tool name, arguments, and call id

This enables human approval or external execution before continuing.

Example:

```rust
if result.state == AgentRunState::Deferred {
    for call in &result.deferred_calls {
        println!("deferred tool: {} {}", call.tool_name, call.tool_call_id);
    }
}
```

## Streaming

`Agent::run_stream` returns an `AgentEventStream` of:

- `TextDelta`
- `ToolCall`
- `Done(AgentRunResult)`

Streaming accounts for:

- usage tokens if provided
- tool-call limits and usage counters
- request count incremented even when no usage payload is provided

Note: tool calls are returned as deferred; execution in streaming mode is caller-controlled.

Example:

```rust
use futures::StreamExt;
use rustic_ai::agent::AgentStreamEvent;

let mut stream = agent.run_stream(input).await?;
while let Some(event) = stream.next().await {
    match event? {
        AgentStreamEvent::TextDelta(delta) => print!("{delta}"),
        AgentStreamEvent::ToolCall(call) => println!("tool call: {}", call.name),
        AgentStreamEvent::Done(result) => println!("done: {}", result.output),
    }
}
```

## Usage limits and accounting

`UsageLimits` can constrain:

- request count
- tool calls
- input tokens
- output tokens
- total tokens

`RunUsage` tracks totals across the run, and is updated both for standard and
streaming paths.

Example:

```rust
use rustic_ai::UsageLimits;

let limits = UsageLimits {
    request_limit: Some(5),
    tool_calls_limit: Some(3),
    ..Default::default()
};
```

## Structured output

`Agent::output_schema(schema)` enables JSON Schema validation of final output.
On validation failure, the agent will retry up to `output_retries` and inject a
`RetryPromptPart` instructing the model to correct its output.

By default, output schema runs in strict JSON-only mode (text output is rejected).
Use `Agent::allow_text_output(true)` if you want to accept text responses when
the provider cannot comply with the schema.

When an output schema is set, RusticAI also injects a system-level instruction
requesting JSON that matches the schema. This provides prompted-output fallback
for providers without native JSON schema response formats.

Example:

```rust
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Summary {
    title: String,
    bullets: Vec<String>,
}

let schema = schemars::schema_for!(Summary);
let agent = Agent::new(model).output_schema(serde_json::to_value(schema)?);
```

Typed helper:

```rust
let agent = Agent::new(model).output_schema_for::<Summary>();
```

## Failover and config resolvers

Failover is based on `ResolvedModelConfig`:

```rust
pub struct ResolvedModelConfig {
    pub primary: String,
    pub backup: Option<String>,
    pub retry_limit: u32,
    pub failover_on: HashSet<String>,
    pub settings: Map<String, Value>,
}
```

Example:

```rust
use rustic_ai::run_with_failover;

let result = run_with_failover(
    &resolver,
    "agent",
    None,
    None,
    |model_name| async move { agent.run(input_for(model_name)).await },
)
.await?;
println!("used {}", result.model_used);
```

### Resolver trait

```rust
pub trait ModelConfigResolver {
    fn resolve_model_config(&self, agent_name: &str, requested_model: Option<&str>, environment: Option<&str>)
        -> ResolvedModelConfig;

    fn resolve_utility_config(&self, utility_name: &str, environment: Option<&str>)
        -> ResolvedModelConfig;

    fn circuit_breaker_config(&self, environment: Option<&str>) -> CircuitBreakerConfig;
}
```

Example:

```rust
use std::collections::HashSet;
use serde_json::Map;
use rustic_ai::{ModelConfigResolver, ResolvedModelConfig};

struct StaticResolver;

impl ModelConfigResolver for StaticResolver {
    fn resolve_model_config(
        &self,
        _agent_name: &str,
        _requested_model: Option<&str>,
        _environment: Option<&str>,
    ) -> ResolvedModelConfig {
        ResolvedModelConfig {
            primary: "openai:gpt-4o-mini".to_string(),
            backup: None,
            retry_limit: 0,
            failover_on: HashSet::new(),
            settings: Map::new(),
        }
    }

    fn resolve_utility_config(
        &self,
        _utility_name: &str,
        _environment: Option<&str>,
    ) -> ResolvedModelConfig {
        self.resolve_model_config("", None, None)
    }
}
```

### In-memory resolver

`InMemoryResolver` is a minimal, non-opinionated resolver used for tests or
custom integrations:

Example:

```rust
let mut resolver = InMemoryResolver::new("openai:gpt-4o-mini");
resolver.insert_agent(
    "agent",
    ModelConfigEntry::default()
        .backup("anthropic:claude-3.5-sonnet")
        .retry_limit(2)
        .failover_on(["http_429", "http_5xx"]),
);
```

## MCP toolsets (HTTP + SSE extras)

`McpServerStreamableHttp` supports:

- `tools/list` and `tools/call`
- optional caching
- list-resources/prompts/templates
- prompt retrieval
- sampling
- SSE event stream for cache invalidation and notifications

Example:

```rust
use rustic_ai::mcp::McpServerStreamableHttp;

let toolset = McpServerStreamableHttp::new("http://localhost:3333")?;
let mut agent = Agent::new(model);
agent.toolset(toolset);
let result = agent.run_with_toolsets(input).await?;
```

See `src/mcp.rs` for request/response shapes and SSE handling.

## Instrumentation

Instrumentation is pluggable via `Instrumenter`:

Run lifecycle:

- `on_run_start`
- `on_run_end`
- `on_run_error`

Model lifecycle:

- `on_model_request`
- `on_model_response`
- `on_model_error`

Tool lifecycle:

- `on_tool_call`
- `on_tool_start`
- `on_tool_end`
- `on_tool_error`

Other:

- `on_usage_limit`
- `on_output_validation_error`

Built-ins:

- `NoopInstrumenter`
- `TracingInstrumenter`

Example:

```rust
use std::sync::Arc;
use rustic_ai::TracingInstrumenter;

let agent = Agent::new(model).instrumenter(Arc::new(TracingInstrumenter::default()));
```

These hooks are compatible with OpenTelemetry (OTEL)-style telemetry layers.
`RunContext` includes a `run_id` field for correlation across logs and spans.

Optional telemetry helpers are available behind feature flags:

- `telemetry-otel` (OTLP exporter)
- `telemetry-datadog` (Datadog exporter)

Example (OTLP):

```rust
use opentelemetry_otlp::Protocol;
use rustic_ai::telemetry::init_otlp_tracing;

let _guard = init_otlp_tracing("rustic-ai", Protocol::Grpc, None, None)?;
```

## Errors

Key error types:

- `AgentError` – orchestration errors and validation
- `ModelError` – provider errors and HTTP/transport failures
- `ToolError` – tool execution errors
- `UsageError` – usage limit breaches

`classify_error_kind` converts errors to canonical strings used by failover:

- `timeout`
- `connect_error`
- `http_401`, `http_403`, `http_429`, `http_5xx`
- `model_error`

Example:

```rust
use rustic_ai::classify_error_kind;

match agent.run(input).await {
    Ok(result) => println!("{}", result.output),
    Err(err) => {
        let kind = classify_error_kind(&err as &dyn std::error::Error);
        eprintln!("run failed ({:?}): {err}", kind);
    }
}
```

## Testing and linting

Run tests:

```bash
cargo test
```

Deterministic test helpers live in `tests/support`:

- `ScriptedModel` queues canned `ModelResponse` values for unit tests.
- `StreamedModel` queues `StreamChunk` values for streaming tests.

These are useful for message/usage assertions without live API calls.

Example (run a single live test):

```bash
RUSTIC_AI_LIVE_TESTS=1 cargo test --test live_providers -- --ignored live_openai_tool_roundtrip
```

Run live provider integration tests (opt-in, ignored by default):

```bash
RUSTIC_AI_LIVE_TESTS=1 OPENAI_API_KEY=... cargo test --test live_providers -- --ignored
```

The live tests will auto-load a local `.env` file if present.

Environment variables for live tests:

- `OPENAI_API_KEY`, `OPENAI_MODEL` (defaults to `gpt-5-mini`)
- `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` (defaults to `claude-sonnet-4-5`)
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`, `GEMINI_MODEL` (defaults to `gemini-2.5-flash`)
- `XAI_API_KEY` or `GROK_API_KEY`, `GROK_MODEL` (required; multimodal requires a vision-capable Grok model)

Run live multimodal tests (opt-in, ignored by default):

```bash
RUSTIC_AI_LIVE_TESTS=1 cargo test --test live_multimodal -- --ignored
```

Notes:

- Live tests prefer explicit `*_MODEL` entries. Set `RUSTIC_AI_LIVE_ALLOW_FALLBACKS=1` to use default model fallbacks.
- Multimodal tests assume:
  - OpenAI models support image input (audio input uses Chat Completions; Responses is image/text only).
  - Anthropic models support image + PDF document input.
  - Gemini models support image + PDF + audio input.
  - Grok multimodal tests use `GROK_MODEL` with a hosted image URL input.

Run clippy:

```bash
cargo clippy --all-targets
```

### Pre-push hooks (pre-commit)

Hooks are managed by pre-commit via `.pre-commit-config.yaml` and run on **pre-push**.
Install them with:

```bash
uvx pre-commit install --hook-type pre-push
```

If you already have `pre-commit` on your PATH, you can run:

```bash
pre-commit install --hook-type pre-push
```
