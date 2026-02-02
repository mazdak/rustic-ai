//! # RusticAI
//!
//! A Rust-native agent framework with tool calling, streaming, and multi-provider
//! support for OpenAI, Anthropic, Gemini, and Grok.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rustic_ai::{Agent, RunInput, UsageLimits, UserContent, infer_model, infer_provider};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model = infer_model("openai:gpt-4o-mini", infer_provider)?;
//! let agent = Agent::new(model)
//!     .system_prompt("You are a helpful assistant.");
//!
//! let input = RunInput::new(
//!     vec![UserContent::Text("Hello!".to_string())],
//!     vec![],
//!     (),
//!     UsageLimits::default(),
//! );
//!
//! let result = agent.run(input).await?;
//! println!("{}", result.output);
//! # Ok(())
//! # }
//! ```
//!
//! Type-state builder:
//!
//! ```rust,no_run
//! use rustic_ai::{RunInput, UsageLimits};
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let input = RunInput::builder(())
//!     .user_text("Hello!")
//!     .usage_limits(UsageLimits::default())
//!     .build();
//! # let _ = input;
//! # Ok(())
//! # }
//! ```
//!
//! Typed structured output:
//!
//! ```rust,no_run
//! use rustic_ai::{Agent, RunInput, UsageLimits, UserContent, infer_model, infer_provider};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! #[derive(Deserialize, JsonSchema)]
//! struct Answer {
//!     answer: String,
//! }
//! let model = infer_model("openai:gpt-4o-mini", infer_provider)?;
//! let agent = Agent::new(model).output_schema_for::<Answer>();
//! let input = RunInput::new(
//!     vec![UserContent::Text("Respond with {\"answer\": \"ok\"}".to_string())],
//!     vec![],
//!     (),
//!     UsageLimits::default(),
//! );
//! let _ = agent.run(input).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Features
//!
//! - **Agent orchestration** with tool calling, usage limits, and message history
//! - **Multi-provider support** for OpenAI, Gemini, Anthropic, and Grok
//! - **Streaming** with structured events
//! - **Structured output** validation via JSON schema
//! - **Deferred tools** for approval flows
//! - **MCP toolsets** for remote tool integration
//! - **Instrumentation hooks** with tracing/OpenTelemetry support
//!
//! ## Optional Features
//!
//! - `telemetry-otel` - OpenTelemetry/OTLP exporter support
//! - `telemetry-datadog` - Datadog exporter support

#![forbid(unsafe_code)]

pub mod agent;
pub mod error;
pub mod failover;
pub mod instrumentation;
mod json_schema;
pub mod mcp;
pub mod messages;
pub mod model;
pub mod model_config;
pub mod providers;
pub mod realtime;
#[cfg(any(feature = "telemetry-otel", feature = "telemetry-datadog"))]
pub mod telemetry;
pub mod tools;
pub mod usage;

pub use agent::{
    Agent, AgentEventStream, AgentRunResult, AgentRunState, AgentStreamEvent, DeferredToolCall,
    RunInput, RunInputBuilder,
};
pub use error::AgentError;
pub use failover::{
    FailoverResult, classify_error_kind, run_with_config, run_with_config_and_classifier,
    run_with_failover, run_with_failover_with_classifier, run_with_utility_failover,
    run_with_utility_failover_with_classifier,
};
pub use instrumentation::{Instrumenter, NoopInstrumenter, TracingInstrumenter};
pub use messages::{
    AudioUrl, BinaryContent, DocumentUrl, ImageUrl, ModelMessage, ModelRequest, ModelRequestPart,
    ModelResponse, ModelResponsePart, ProviderItemPart, RetryPromptPart, SystemPromptPart,
    TextPart, ToolCallPart, ToolReturnPart, UserContent, UserPromptPart, VideoUrl,
};
pub use model::{
    Model, ModelError, ModelRequestParameters, ModelSettings, ModelStream, OutputMode, StreamChunk,
};
pub use model_config::{
    CircuitBreakerConfig, InMemoryResolver, ModelConfigEntry, ModelConfigResolver,
    ResolvedModelConfig,
};
pub use providers::{Provider, ProviderError, infer_model, infer_provider};
pub use realtime::grok::{
    GrokClient as GrokRealtimeClient, GrokSender as GrokRealtimeSender,
    ServerEvent as GrokRealtimeEvent, SessionConfig as GrokSessionConfig,
};
pub use tools::ToolError;
pub use tools::{FunctionTool, RunContext, Tool, ToolDefinition, ToolKind, Toolset};
pub use usage::{RequestUsage, RunUsage, UsageError, UsageLimits};
