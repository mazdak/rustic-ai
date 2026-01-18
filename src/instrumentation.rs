use crate::usage::RunUsage;

#[derive(Clone, Debug)]
pub struct ModelRequestInfo {
    pub model_name: String,
    pub message_count: usize,
    pub tool_count: usize,
    pub output_schema: bool,
}

#[derive(Clone, Debug)]
pub struct ModelResponseInfo {
    pub model_name: String,
    pub finish_reason: Option<String>,
    pub usage: RunUsage,
    pub tool_calls: usize,
}

#[derive(Clone, Debug)]
pub struct ToolCallInfo {
    pub tool_name: String,
    pub tool_call_id: Option<String>,
    pub deferred: bool,
}

pub trait Instrumenter: Send + Sync {
    fn on_model_request(&self, _info: &ModelRequestInfo) {}
    fn on_model_response(&self, _info: &ModelResponseInfo) {}
    fn on_tool_call(&self, _info: &ToolCallInfo) {}
}

#[derive(Clone, Default)]
pub struct NoopInstrumenter;

impl Instrumenter for NoopInstrumenter {}

#[derive(Clone, Default)]
pub struct TracingInstrumenter;

impl Instrumenter for TracingInstrumenter {
    fn on_model_request(&self, info: &ModelRequestInfo) {
        tracing::info!(
            model = info.model_name.as_str(),
            message_count = info.message_count,
            tool_count = info.tool_count,
            output_schema = info.output_schema,
            "model request"
        );
    }

    fn on_model_response(&self, info: &ModelResponseInfo) {
        tracing::info!(
            model = info.model_name.as_str(),
            finish_reason = info.finish_reason.as_deref().unwrap_or(""),
            tool_calls = info.tool_calls,
            requests = info.usage.requests,
            input_tokens = info.usage.input_tokens,
            output_tokens = info.usage.output_tokens,
            "model response"
        );
    }

    fn on_tool_call(&self, info: &ToolCallInfo) {
        tracing::info!(
            tool = info.tool_name.as_str(),
            tool_call_id = info.tool_call_id.as_deref().unwrap_or(""),
            deferred = info.deferred,
            "tool call"
        );
    }
}
