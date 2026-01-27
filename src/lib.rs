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

pub use agent::{Agent, AgentRunResult, AgentRunState, DeferredToolCall, RunInput};
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
