use std::time::Duration;

use crate::agent::AgentRunState;
use crate::tools::ToolKind;
use crate::usage::{RunUsage, UsageLimits};

#[derive(Clone, Debug)]
pub struct RunStartInfo {
    pub run_id: String,
    pub model_name: String,
    pub message_count: usize,
    pub tool_count: usize,
    pub output_schema: bool,
    pub streaming: bool,
    pub allow_text_output: bool,
    pub output_retries: u32,
    pub usage_limits: UsageLimits,
}

#[derive(Clone, Debug)]
pub struct RunEndInfo {
    pub run_id: String,
    pub model_name: String,
    pub state: AgentRunState,
    pub usage: RunUsage,
    pub output_len: usize,
    pub deferred_calls: usize,
    pub tool_calls: usize,
    pub duration: Duration,
}

#[derive(Clone, Debug)]
pub struct RunErrorInfo {
    pub run_id: String,
    pub model_name: String,
    pub error: String,
    pub error_kind: Option<String>,
    pub streaming: bool,
    pub duration: Duration,
}

#[derive(Clone, Debug)]
pub struct ModelRequestInfo {
    pub run_id: String,
    pub model_name: String,
    pub step: u64,
    pub message_count: usize,
    pub tool_count: usize,
    pub output_schema: bool,
    pub streaming: bool,
    pub allow_text_output: bool,
}

#[derive(Clone, Debug)]
pub struct ModelResponseInfo {
    pub run_id: String,
    pub model_name: String,
    pub step: u64,
    pub finish_reason: Option<String>,
    pub usage: RunUsage,
    pub tool_calls: usize,
    pub output_len: usize,
    pub duration: Duration,
    pub streaming: bool,
}

#[derive(Clone, Debug)]
pub struct ModelErrorInfo {
    pub run_id: String,
    pub model_name: String,
    pub step: u64,
    pub error: String,
    pub error_kind: Option<String>,
    pub duration: Duration,
    pub streaming: bool,
}

#[derive(Clone, Debug)]
pub struct ToolCallInfo {
    pub run_id: String,
    pub tool_name: String,
    pub tool_call_id: Option<String>,
    pub deferred: bool,
    pub kind: ToolKind,
    pub sequential: bool,
}

#[derive(Clone, Debug)]
pub struct ToolStartInfo {
    pub run_id: String,
    pub tool_name: String,
    pub tool_call_id: Option<String>,
    pub timeout_secs: Option<f64>,
    pub sequential: bool,
}

#[derive(Clone, Debug)]
pub struct ToolEndInfo {
    pub run_id: String,
    pub tool_name: String,
    pub tool_call_id: Option<String>,
    pub duration: Duration,
}

#[derive(Clone, Debug)]
pub struct ToolErrorInfo {
    pub run_id: String,
    pub tool_name: String,
    pub tool_call_id: Option<String>,
    pub error: String,
    pub duration: Duration,
}

#[derive(Clone, Debug)]
pub enum UsageLimitKind {
    Requests,
    ToolCalls,
    InputTokens,
    OutputTokens,
    TotalTokens,
}

#[derive(Clone, Debug)]
pub struct UsageLimitInfo {
    pub run_id: String,
    pub model_name: String,
    pub kind: UsageLimitKind,
    pub limit: u64,
    pub usage: RunUsage,
}

#[derive(Clone, Debug)]
pub struct OutputValidationErrorInfo {
    pub run_id: String,
    pub model_name: String,
    pub error: String,
    pub output_len: usize,
}

pub trait Instrumenter: Send + Sync {
    fn on_run_start(&self, _info: &RunStartInfo) {}
    fn on_run_end(&self, _info: &RunEndInfo) {}
    fn on_run_error(&self, _info: &RunErrorInfo) {}
    fn on_model_request(&self, _info: &ModelRequestInfo) {}
    fn on_model_response(&self, _info: &ModelResponseInfo) {}
    fn on_model_error(&self, _info: &ModelErrorInfo) {}
    fn on_tool_call(&self, _info: &ToolCallInfo) {}
    fn on_tool_start(&self, _info: &ToolStartInfo) {}
    fn on_tool_end(&self, _info: &ToolEndInfo) {}
    fn on_tool_error(&self, _info: &ToolErrorInfo) {}
    fn on_usage_limit(&self, _info: &UsageLimitInfo) {}
    fn on_output_validation_error(&self, _info: &OutputValidationErrorInfo) {}
}

#[derive(Clone, Default)]
pub struct NoopInstrumenter;

impl Instrumenter for NoopInstrumenter {}

#[derive(Clone, Default)]
pub struct TracingInstrumenter;

impl Instrumenter for TracingInstrumenter {
    fn on_run_start(&self, info: &RunStartInfo) {
        tracing::info!(
            run_id = info.run_id.as_str(),
            model = info.model_name.as_str(),
            message_count = info.message_count,
            tool_count = info.tool_count,
            output_schema = info.output_schema,
            streaming = info.streaming,
            allow_text_output = info.allow_text_output,
            output_retries = info.output_retries,
            request_limit = info.usage_limits.request_limit.unwrap_or(0),
            tool_calls_limit = info.usage_limits.tool_calls_limit.unwrap_or(0),
            input_tokens_limit = info.usage_limits.input_tokens_limit.unwrap_or(0),
            output_tokens_limit = info.usage_limits.output_tokens_limit.unwrap_or(0),
            total_tokens_limit = info.usage_limits.total_tokens_limit.unwrap_or(0),
            "agent run started"
        );
    }

    fn on_run_end(&self, info: &RunEndInfo) {
        tracing::info!(
            run_id = info.run_id.as_str(),
            model = info.model_name.as_str(),
            state = ?info.state,
            output_len = info.output_len,
            deferred_calls = info.deferred_calls,
            tool_calls = info.tool_calls,
            requests = info.usage.requests,
            input_tokens = info.usage.input_tokens,
            output_tokens = info.usage.output_tokens,
            cache_write_tokens = info.usage.cache_write_tokens,
            cache_read_tokens = info.usage.cache_read_tokens,
            input_audio_tokens = info.usage.input_audio_tokens,
            output_audio_tokens = info.usage.output_audio_tokens,
            duration_ms = info.duration.as_millis() as u64,
            "agent run completed"
        );
    }

    fn on_run_error(&self, info: &RunErrorInfo) {
        tracing::error!(
            run_id = info.run_id.as_str(),
            model = info.model_name.as_str(),
            streaming = info.streaming,
            error = info.error.as_str(),
            error_kind = info.error_kind.as_deref().unwrap_or(""),
            duration_ms = info.duration.as_millis() as u64,
            "agent run failed"
        );
    }

    fn on_model_request(&self, info: &ModelRequestInfo) {
        tracing::info!(
            run_id = info.run_id.as_str(),
            model = info.model_name.as_str(),
            step = info.step,
            message_count = info.message_count,
            tool_count = info.tool_count,
            output_schema = info.output_schema,
            streaming = info.streaming,
            allow_text_output = info.allow_text_output,
            "model request"
        );
    }

    fn on_model_response(&self, info: &ModelResponseInfo) {
        tracing::info!(
            run_id = info.run_id.as_str(),
            model = info.model_name.as_str(),
            step = info.step,
            finish_reason = info.finish_reason.as_deref().unwrap_or(""),
            tool_calls = info.tool_calls,
            requests = info.usage.requests,
            input_tokens = info.usage.input_tokens,
            output_tokens = info.usage.output_tokens,
            output_len = info.output_len,
            duration_ms = info.duration.as_millis() as u64,
            streaming = info.streaming,
            "model response"
        );
    }

    fn on_model_error(&self, info: &ModelErrorInfo) {
        tracing::error!(
            run_id = info.run_id.as_str(),
            model = info.model_name.as_str(),
            step = info.step,
            error = info.error.as_str(),
            error_kind = info.error_kind.as_deref().unwrap_or(""),
            duration_ms = info.duration.as_millis() as u64,
            streaming = info.streaming,
            "model request failed"
        );
    }

    fn on_tool_call(&self, info: &ToolCallInfo) {
        tracing::info!(
            run_id = info.run_id.as_str(),
            tool = info.tool_name.as_str(),
            tool_call_id = info.tool_call_id.as_deref().unwrap_or(""),
            deferred = info.deferred,
            kind = ?info.kind,
            sequential = info.sequential,
            "tool call"
        );
    }

    fn on_tool_start(&self, info: &ToolStartInfo) {
        tracing::debug!(
            run_id = info.run_id.as_str(),
            tool = info.tool_name.as_str(),
            tool_call_id = info.tool_call_id.as_deref().unwrap_or(""),
            timeout_secs = info.timeout_secs.unwrap_or(0.0),
            sequential = info.sequential,
            "tool execution started"
        );
    }

    fn on_tool_end(&self, info: &ToolEndInfo) {
        tracing::info!(
            run_id = info.run_id.as_str(),
            tool = info.tool_name.as_str(),
            tool_call_id = info.tool_call_id.as_deref().unwrap_or(""),
            duration_ms = info.duration.as_millis() as u64,
            "tool execution completed"
        );
    }

    fn on_tool_error(&self, info: &ToolErrorInfo) {
        tracing::error!(
            run_id = info.run_id.as_str(),
            tool = info.tool_name.as_str(),
            tool_call_id = info.tool_call_id.as_deref().unwrap_or(""),
            error = info.error.as_str(),
            duration_ms = info.duration.as_millis() as u64,
            "tool execution failed"
        );
    }

    fn on_usage_limit(&self, info: &UsageLimitInfo) {
        tracing::warn!(
            run_id = info.run_id.as_str(),
            model = info.model_name.as_str(),
            kind = ?info.kind,
            limit = info.limit,
            requests = info.usage.requests,
            tool_calls = info.usage.tool_calls,
            input_tokens = info.usage.input_tokens,
            output_tokens = info.usage.output_tokens,
            "usage limit exceeded"
        );
    }

    fn on_output_validation_error(&self, info: &OutputValidationErrorInfo) {
        tracing::warn!(
            run_id = info.run_id.as_str(),
            model = info.model_name.as_str(),
            error = info.error.as_str(),
            output_len = info.output_len,
            "output validation failed"
        );
    }
}
