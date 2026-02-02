use std::time::Duration;

use rustic_ai::instrumentation::{
    ModelErrorInfo, ModelRequestInfo, ModelResponseInfo, OutputValidationErrorInfo, RunEndInfo,
    RunErrorInfo, RunStartInfo, ToolCallInfo, ToolEndInfo, ToolErrorInfo, ToolStartInfo,
    UsageLimitInfo, UsageLimitKind,
};
use rustic_ai::{
    AgentRunState, Instrumenter, NoopInstrumenter, RunUsage, ToolKind, TracingInstrumenter,
    UsageLimits,
};

#[test]
fn tracing_instrumenter_handles_all_hooks() {
    let instrumenter = TracingInstrumenter;
    let usage_limits = UsageLimits {
        request_limit: Some(2),
        tool_calls_limit: Some(3),
        input_tokens_limit: Some(10),
        output_tokens_limit: Some(10),
        total_tokens_limit: Some(20),
    };
    let usage = RunUsage {
        requests: 1,
        tool_calls: 1,
        input_tokens: 2,
        output_tokens: 3,
        ..Default::default()
    };

    instrumenter.on_run_start(&RunStartInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        message_count: 1,
        tool_count: 2,
        output_schema: true,
        streaming: false,
        allow_text_output: false,
        output_retries: 1,
        usage_limits: usage_limits.clone(),
    });

    instrumenter.on_model_request(&ModelRequestInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        step: 0,
        message_count: 2,
        tool_count: 1,
        output_schema: false,
        streaming: false,
        allow_text_output: true,
    });

    instrumenter.on_model_response(&ModelResponseInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        step: 0,
        finish_reason: Some("stop".to_string()),
        usage: usage.clone(),
        tool_calls: 1,
        output_len: 3,
        duration: Duration::from_millis(5),
        streaming: false,
    });

    instrumenter.on_model_error(&ModelErrorInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        step: 0,
        error: "oops".to_string(),
        error_kind: Some("timeout".to_string()),
        duration: Duration::from_millis(5),
        streaming: false,
    });

    instrumenter.on_tool_call(&ToolCallInfo {
        run_id: "run".to_string(),
        tool_name: "tool".to_string(),
        tool_call_id: Some("call".to_string()),
        deferred: false,
        kind: ToolKind::Function,
        sequential: false,
    });

    instrumenter.on_tool_start(&ToolStartInfo {
        run_id: "run".to_string(),
        tool_name: "tool".to_string(),
        tool_call_id: Some("call".to_string()),
        timeout_secs: Some(1.0),
        sequential: false,
    });

    instrumenter.on_tool_end(&ToolEndInfo {
        run_id: "run".to_string(),
        tool_name: "tool".to_string(),
        tool_call_id: Some("call".to_string()),
        duration: Duration::from_millis(2),
    });

    instrumenter.on_tool_error(&ToolErrorInfo {
        run_id: "run".to_string(),
        tool_name: "tool".to_string(),
        tool_call_id: Some("call".to_string()),
        error: "failed".to_string(),
        duration: Duration::from_millis(2),
    });

    instrumenter.on_usage_limit(&UsageLimitInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        kind: UsageLimitKind::TotalTokens,
        limit: 10,
        usage: usage.clone(),
    });

    instrumenter.on_output_validation_error(&OutputValidationErrorInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        error: "invalid".to_string(),
        output_len: 5,
    });

    instrumenter.on_run_end(&RunEndInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        state: AgentRunState::Completed,
        usage: usage.clone(),
        output_len: 5,
        deferred_calls: 0,
        tool_calls: 1,
        duration: Duration::from_millis(10),
    });

    instrumenter.on_run_error(&RunErrorInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        error: "boom".to_string(),
        error_kind: Some("model_error".to_string()),
        streaming: false,
        duration: Duration::from_millis(10),
    });
}

#[test]
fn noop_instrumenter_handles_all_hooks() {
    let instrumenter = NoopInstrumenter;
    let usage_limits = UsageLimits::default();
    let usage = RunUsage::default();

    instrumenter.on_run_start(&RunStartInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        message_count: 0,
        tool_count: 0,
        output_schema: false,
        streaming: false,
        allow_text_output: true,
        output_retries: 0,
        usage_limits,
    });

    instrumenter.on_model_request(&ModelRequestInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        step: 0,
        message_count: 0,
        tool_count: 0,
        output_schema: false,
        streaming: false,
        allow_text_output: true,
    });

    instrumenter.on_model_response(&ModelResponseInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        step: 0,
        finish_reason: None,
        usage: usage.clone(),
        tool_calls: 0,
        output_len: 0,
        duration: Duration::from_secs(0),
        streaming: false,
    });

    instrumenter.on_model_error(&ModelErrorInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        step: 0,
        error: "".to_string(),
        error_kind: None,
        duration: Duration::from_secs(0),
        streaming: false,
    });

    instrumenter.on_tool_call(&ToolCallInfo {
        run_id: "run".to_string(),
        tool_name: "tool".to_string(),
        tool_call_id: None,
        deferred: false,
        kind: ToolKind::Function,
        sequential: false,
    });

    instrumenter.on_tool_start(&ToolStartInfo {
        run_id: "run".to_string(),
        tool_name: "tool".to_string(),
        tool_call_id: None,
        timeout_secs: None,
        sequential: false,
    });

    instrumenter.on_tool_end(&ToolEndInfo {
        run_id: "run".to_string(),
        tool_name: "tool".to_string(),
        tool_call_id: None,
        duration: Duration::from_secs(0),
    });

    instrumenter.on_tool_error(&ToolErrorInfo {
        run_id: "run".to_string(),
        tool_name: "tool".to_string(),
        tool_call_id: None,
        error: "".to_string(),
        duration: Duration::from_secs(0),
    });

    instrumenter.on_usage_limit(&UsageLimitInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        kind: UsageLimitKind::Requests,
        limit: 0,
        usage: usage.clone(),
    });

    instrumenter.on_output_validation_error(&OutputValidationErrorInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        error: "".to_string(),
        output_len: 0,
    });

    instrumenter.on_run_end(&RunEndInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        state: AgentRunState::Completed,
        usage,
        output_len: 0,
        deferred_calls: 0,
        tool_calls: 0,
        duration: Duration::from_secs(0),
    });

    instrumenter.on_run_error(&RunErrorInfo {
        run_id: "run".to_string(),
        model_name: "model".to_string(),
        error: "".to_string(),
        error_kind: None,
        streaming: false,
        duration: Duration::from_secs(0),
    });
}
