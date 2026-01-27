use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_stream::try_stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::future::join_all;
use futures::stream::Stream;
use jsonschema::{Draft, JSONSchema};
use serde_json::Value;
use tokio::time::timeout;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::error::AgentError;
use crate::failover::{FailoverResult, classify_error_kind, run_with_config_and_classifier};
use crate::instrumentation::{
    Instrumenter, ModelErrorInfo, ModelRequestInfo, ModelResponseInfo, NoopInstrumenter,
    OutputValidationErrorInfo, RunEndInfo, RunErrorInfo, RunStartInfo, ToolCallInfo, ToolEndInfo,
    ToolErrorInfo, ToolStartInfo, UsageLimitInfo, UsageLimitKind,
};
use crate::messages::{
    ModelMessage, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart,
    RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart, UserContent, UserPromptPart,
};
use crate::model::{Model, ModelRequestParameters, ModelSettings};
use crate::model_config::{ModelConfigResolver, ResolvedModelConfig};
use crate::tools::{RunContext, Tool, ToolDefinition, ToolError, ToolKind, Toolset};
use crate::usage::{RunUsage, UsageError, UsageLimits};

pub type PrepareToolsFn<Deps> = Arc<
    dyn Fn(
            &RunContext<Deps>,
            Vec<ToolDefinition>,
        ) -> BoxFuture<'static, Result<Vec<ToolDefinition>, ToolError>>
        + Send
        + Sync,
>;

pub struct Agent<Deps> {
    model: Arc<dyn Model>,
    system_prompt: Option<String>,
    model_settings: Option<ModelSettings>,
    tools: HashMap<String, Arc<dyn Tool<Deps>>>,
    toolsets: Vec<Arc<dyn Toolset<Deps>>>,
    prepare_tools: Option<PrepareToolsFn<Deps>>,
    instrumenter: Arc<dyn Instrumenter>,
    output_schema: Option<Value>,
    output_retries: u32,
    allow_text_output: bool,
}

impl<Deps> Agent<Deps>
where
    Deps: Send + Sync + 'static,
{
    fn prepare_run_input(&self, input: RunInput<Deps>) -> PreparedRunInput<Deps> {
        PreparedRunInput {
            user_prompt: input.user_prompt,
            message_history: input.message_history,
            deps: Arc::new(input.deps),
            usage_limits: input.usage_limits,
            include_system_prompt: input.include_system_prompt,
            run_id: resolve_run_id(input.run_id),
        }
    }

    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            system_prompt: None,
            model_settings: None,
            tools: HashMap::new(),
            toolsets: Vec::new(),
            prepare_tools: None,
            instrumenter: Arc::new(NoopInstrumenter),
            output_schema: None,
            output_retries: 0,
            allow_text_output: false,
        }
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn model_settings(mut self, settings: ModelSettings) -> Self {
        self.model_settings = Some(settings);
        self
    }

    pub fn instrumenter(mut self, instrumenter: Arc<dyn Instrumenter>) -> Self {
        self.instrumenter = instrumenter;
        self
    }

    pub fn output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    pub fn output_retries(mut self, retries: u32) -> Self {
        self.output_retries = retries;
        self
    }

    pub fn allow_text_output(mut self, allow: bool) -> Self {
        self.allow_text_output = allow;
        self
    }

    pub fn tool(&mut self, tool: impl Tool<Deps> + 'static) {
        let def = tool.definition();
        self.tools.insert(def.name.clone(), Arc::new(tool));
    }

    pub fn toolset(&mut self, toolset: impl Toolset<Deps> + 'static) {
        self.toolsets.push(Arc::new(toolset));
    }

    pub fn prepare_tools(mut self, func: PrepareToolsFn<Deps>) -> Self {
        self.prepare_tools = Some(func);
        self
    }

    pub async fn enter_toolsets(&self) -> Result<(), AgentError> {
        for toolset in &self.toolsets {
            toolset.enter().await.map_err(AgentError::Tool)?;
        }
        Ok(())
    }

    pub async fn exit_toolsets(&self) -> Result<(), AgentError> {
        for toolset in self.toolsets.iter().rev() {
            toolset.exit().await.map_err(AgentError::Tool)?;
        }
        Ok(())
    }

    pub async fn run_with_toolsets(
        &self,
        input: RunInput<Deps>,
    ) -> Result<AgentRunResult, AgentError> {
        self.enter_toolsets().await?;
        let result = self.run(input).await;
        self.exit_toolsets().await?;
        result
    }

    pub async fn run(&self, input: RunInput<Deps>) -> Result<AgentRunResult, AgentError> {
        let prepared = self.prepare_run_input(input);
        self.run_prepared(Arc::clone(&self.model), prepared, None)
            .await
    }

    async fn run_prepared(
        &self,
        model: Arc<dyn Model>,
        prepared: PreparedRunInput<Deps>,
        settings_override: Option<ModelSettings>,
    ) -> Result<AgentRunResult, AgentError> {
        let PreparedRunInput {
            user_prompt,
            mut message_history,
            deps,
            usage_limits,
            include_system_prompt,
            run_id,
        } = prepared;

        let mut messages = Vec::new();
        let output_instructions = self.output_schema.as_ref().map(build_output_instructions);

        if include_system_prompt && let Some(prompt) = &self.system_prompt {
            messages.push(ModelMessage::Request(ModelRequest {
                parts: vec![ModelRequestPart::SystemPrompt(
                    crate::messages::SystemPromptPart {
                        content: prompt.clone(),
                    },
                )],
                instructions: None,
            }));
        }

        messages.append(&mut message_history);
        messages.push(ModelMessage::Request(ModelRequest {
            parts: vec![ModelRequestPart::UserPrompt(UserPromptPart {
                content: user_prompt.clone(),
            })],
            instructions: output_instructions.clone(),
        }));

        let mut usage = RunUsage::default();
        let mut output_attempts = 0u32;
        let mut step = 0u64;
        let max_steps = usage_limits
            .request_limit
            .map(|limit| limit.saturating_add(1).max(1))
            .unwrap_or(u64::MAX);
        let run_started_at = Instant::now();
        let model_name = model.name().to_string();
        let mut run_started = false;

        let result = 'run: loop {
            let run_ctx = RunContext {
                run_id: run_id.clone(),
                deps: Arc::clone(&deps),
                model: Arc::clone(&model),
                usage: usage.clone(),
                prompt: Some(user_prompt.clone()),
                messages: messages.clone(),
                tool_call_id: None,
                tool_name: None,
            };

            let (tool_defs, tool_map) = match self.collect_tools(&run_ctx).await {
                Ok(result) => result,
                Err(err) => break 'run Err(AgentError::Tool(err)),
            };
            let (tool_defs, tool_map) = match self
                .apply_prepare_tools(&run_ctx, tool_defs, tool_map)
                .await
            {
                Ok(result) => result,
                Err(err) => break 'run Err(AgentError::Tool(err)),
            };
            let mut params = ModelRequestParameters::new(tool_defs);
            if let Some(schema) = &self.output_schema {
                params = params.with_output_schema(schema.clone());
                params.allow_text_output = self.allow_text_output;
            }

            if !run_started {
                self.instrumenter.on_run_start(&RunStartInfo {
                    run_id: run_id.clone(),
                    model_name: model_name.clone(),
                    message_count: messages.len(),
                    tool_count: params.function_tools.len(),
                    output_schema: params.output_schema.is_some(),
                    streaming: false,
                    allow_text_output: self.allow_text_output,
                    output_retries: self.output_retries,
                    usage_limits: usage_limits.clone(),
                });
                run_started = true;
            }

            if let Err(err) = usage_limits.check_request(usage.requests) {
                record_usage_limit(&self.instrumenter, &run_id, &model_name, &usage, &err);
                break 'run Err(AgentError::Usage(err));
            }

            self.instrumenter.on_model_request(&ModelRequestInfo {
                run_id: run_id.clone(),
                model_name: model_name.clone(),
                step,
                message_count: messages.len(),
                tool_count: params.function_tools.len(),
                output_schema: params.output_schema.is_some(),
                streaming: false,
                allow_text_output: self.allow_text_output,
            });

            let response_settings = settings_override.as_ref().or(self.model_settings.as_ref());
            let request_started = Instant::now();
            let mut response = match model.request(&messages, response_settings, &params).await {
                Ok(response) => response,
                Err(err) => {
                    self.instrumenter.on_model_error(&ModelErrorInfo {
                        run_id: run_id.clone(),
                        model_name: model_name.clone(),
                        step,
                        error: err.to_string(),
                        error_kind: classify_error_kind(&err as &dyn std::error::Error)
                            .map(str::to_string),
                        duration: request_started.elapsed(),
                        streaming: false,
                    });
                    break 'run Err(AgentError::Model(err));
                }
            };

            if response.model_name.is_none() {
                response.model_name = Some(model_name.clone());
            }

            if let Some(request_usage) = &response.usage {
                usage.incr_request(request_usage);
            } else {
                usage.requests += 1;
            }

            if let Err(err) = usage_limits.check_after_response(&usage) {
                record_usage_limit(&self.instrumenter, &run_id, &model_name, &usage, &err);
                break 'run Err(AgentError::Usage(err));
            }
            messages.push(ModelMessage::Response(response.clone()));

            let output_len = response.text().map(|text| text.len()).unwrap_or(0);
            self.instrumenter.on_model_response(&ModelResponseInfo {
                run_id: run_id.clone(),
                model_name: model_name.clone(),
                step,
                finish_reason: response.finish_reason.clone(),
                usage: usage.clone(),
                tool_calls: response.tool_calls().len(),
                output_len,
                duration: request_started.elapsed(),
                streaming: false,
            });

            let tool_calls = response.tool_calls();
            if tool_calls.is_empty() {
                let output = response.text().unwrap_or_default();
                let parsed_output = match self.output_schema.as_ref() {
                    Some(schema) => {
                        match validate_output(schema, &output, self.allow_text_output) {
                            Ok(parsed) => parsed,
                            Err(err) => {
                                if output_attempts < self.output_retries {
                                    output_attempts += 1;
                                    messages.push(ModelMessage::Request(ModelRequest {
                                        parts: vec![ModelRequestPart::RetryPrompt(
                                            RetryPromptPart {
                                                content: err.clone(),
                                                tool_name: None,
                                                tool_call_id: None,
                                            },
                                        )],
                                        instructions: None,
                                    }));
                                    continue;
                                }
                                self.instrumenter.on_output_validation_error(
                                    &OutputValidationErrorInfo {
                                        run_id: run_id.clone(),
                                        model_name: model_name.clone(),
                                        error: err.clone(),
                                        output_len: output.len(),
                                    },
                                );
                                break 'run Err(AgentError::OutputValidation(err));
                            }
                        }
                    }
                    None => None,
                };
                break 'run Ok(AgentRunResult {
                    output,
                    usage,
                    messages,
                    response,
                    parsed_output,
                    deferred_calls: Vec::new(),
                    state: AgentRunState::Completed,
                });
            }

            let mut deferred_calls = Vec::new();
            let mut executable_calls: Vec<(usize, ToolCallPart, ToolEntry<Deps>)> = Vec::new();
            for (index, call) in tool_calls.into_iter().enumerate() {
                if let Err(err) = usage_limits.check_tool_call(usage.tool_calls) {
                    record_usage_limit(&self.instrumenter, &run_id, &model_name, &usage, &err);
                    break 'run Err(AgentError::Usage(err));
                }
                usage.incr_tool_call();
                let entry = match tool_map.get(&call.name) {
                    Some(entry) => entry,
                    None => {
                        let err = AgentError::UnknownTool(call.name.clone());
                        self.instrumenter.on_tool_error(&ToolErrorInfo {
                            run_id: run_id.clone(),
                            tool_name: call.name.clone(),
                            tool_call_id: Some(call.id.clone()),
                            error: err.to_string(),
                            duration: Duration::from_millis(0),
                        });
                        break 'run Err(err);
                    }
                };

                let is_deferred = matches!(
                    entry.definition.kind,
                    ToolKind::External | ToolKind::Unapproved
                );

                self.instrumenter.on_tool_call(&ToolCallInfo {
                    run_id: run_id.clone(),
                    tool_name: call.name.clone(),
                    tool_call_id: Some(call.id.clone()),
                    deferred: is_deferred,
                    kind: entry.definition.kind.clone(),
                    sequential: entry.definition.sequential,
                });

                if is_deferred {
                    deferred_calls.push(DeferredToolCall {
                        tool_name: call.name.clone(),
                        tool_call_id: call.id.clone(),
                        arguments: call.arguments.clone(),
                        kind: entry.definition.kind.clone(),
                    });
                    continue;
                }
                executable_calls.push((index, call, entry.clone()));
            }

            let should_run_sequentially = executable_calls
                .iter()
                .any(|(_, _, entry)| entry.definition.sequential);
            let mut tool_results: Vec<(usize, ToolReturnPart)> = Vec::new();
            if should_run_sequentially {
                for (index, call, entry) in executable_calls {
                    let tool_ctx = RunContext {
                        run_id: run_id.clone(),
                        deps: Arc::clone(&deps),
                        model: Arc::clone(&model),
                        usage: usage.clone(),
                        prompt: Some(user_prompt.clone()),
                        messages: messages.clone(),
                        tool_call_id: None,
                        tool_name: None,
                    };
                    let tool_result = match self
                        .execute_tool_with_timeout(&tool_ctx, &entry, &call)
                        .await
                    {
                        Ok(result) => result,
                        Err(err) => break 'run Err(err),
                    };
                    tool_results.push((
                        index,
                        ToolReturnPart {
                            tool_name: call.name.clone(),
                            tool_call_id: call.id.clone(),
                            content: tool_result,
                        },
                    ));
                }
            } else if !executable_calls.is_empty() {
                let mut futures = Vec::new();
                for (index, call, entry) in executable_calls {
                    let tool_ctx = RunContext {
                        run_id: run_id.clone(),
                        deps: Arc::clone(&deps),
                        model: Arc::clone(&model),
                        usage: usage.clone(),
                        prompt: Some(user_prompt.clone()),
                        messages: messages.clone(),
                        tool_call_id: None,
                        tool_name: None,
                    };
                    let call_clone = call.clone();
                    let entry_clone = entry.clone();
                    futures.push(async move {
                        let result = self
                            .execute_tool_with_timeout(&tool_ctx, &entry_clone, &call_clone)
                            .await;
                        (index, call_clone, result)
                    });
                }
                for (index, call, result) in join_all(futures).await {
                    let tool_result = match result {
                        Ok(result) => result,
                        Err(err) => break 'run Err(err),
                    };
                    tool_results.push((
                        index,
                        ToolReturnPart {
                            tool_name: call.name.clone(),
                            tool_call_id: call.id.clone(),
                            content: tool_result,
                        },
                    ));
                }
            }

            tool_results.sort_by_key(|(index, _)| *index);
            for (_, tool_return) in tool_results {
                messages.push(ModelMessage::Request(ModelRequest {
                    parts: vec![ModelRequestPart::ToolReturn(tool_return)],
                    instructions: None,
                }));
            }

            if !deferred_calls.is_empty() {
                break 'run Ok(AgentRunResult {
                    output: String::new(),
                    usage,
                    messages,
                    response,
                    parsed_output: None,
                    deferred_calls,
                    state: AgentRunState::Deferred,
                });
            }

            step += 1;
            if step >= max_steps {
                break 'run Err(AgentError::Config(
                    "tool execution loop exceeded request limit".to_string(),
                ));
            }
        };

        match result {
            Ok(result) => {
                self.instrumenter.on_run_end(&RunEndInfo {
                    run_id: run_id.clone(),
                    model_name: model_name.clone(),
                    state: result.state.clone(),
                    usage: result.usage.clone(),
                    output_len: result.output.len(),
                    deferred_calls: result.deferred_calls.len(),
                    tool_calls: result.usage.tool_calls as usize,
                    duration: run_started_at.elapsed(),
                });
                Ok(result)
            }
            Err(err) => {
                self.instrumenter.on_run_error(&RunErrorInfo {
                    run_id: run_id.clone(),
                    model_name: model_name.clone(),
                    error: err.to_string(),
                    error_kind: classify_error_kind(&err as &dyn std::error::Error)
                        .map(str::to_string),
                    streaming: false,
                    duration: run_started_at.elapsed(),
                });
                Err(err)
            }
        }
    }

    pub async fn run_with_failover(
        &self,
        input: RunInput<Deps>,
        resolver: &dyn ModelConfigResolver,
        agent_name: &str,
        requested_model: Option<&str>,
        environment: Option<&str>,
        model_factory: impl Fn(&str) -> Result<Arc<dyn Model>, AgentError> + Send + Sync,
    ) -> Result<FailoverResult<AgentRunResult>, AgentError> {
        let config = resolver.resolve_model_config(agent_name, requested_model, environment);
        self.run_with_resolved_failover(input, config, model_factory)
            .await
    }

    pub async fn run_with_resolved_failover(
        &self,
        input: RunInput<Deps>,
        config: ResolvedModelConfig,
        model_factory: impl Fn(&str) -> Result<Arc<dyn Model>, AgentError> + Send + Sync,
    ) -> Result<FailoverResult<AgentRunResult>, AgentError> {
        let prepared = self.prepare_run_input(input);
        let settings_override = (!config.settings.is_empty()).then(|| config.settings.clone());
        run_with_config_and_classifier(
            config,
            |model_name| {
                let prepared = prepared.clone();
                let model = model_factory(model_name);
                let settings_override = settings_override.clone();
                async move {
                    let model = model?;
                    self.run_prepared(model, prepared, settings_override).await
                }
            },
            |error| classify_error_kind(error),
        )
        .await
    }

    pub async fn run_with_failover_with_toolsets(
        &self,
        input: RunInput<Deps>,
        resolver: &dyn ModelConfigResolver,
        agent_name: &str,
        requested_model: Option<&str>,
        environment: Option<&str>,
        model_factory: impl Fn(&str) -> Result<Arc<dyn Model>, AgentError> + Send + Sync,
    ) -> Result<FailoverResult<AgentRunResult>, AgentError> {
        self.enter_toolsets().await?;
        let result = self
            .run_with_failover(
                input,
                resolver,
                agent_name,
                requested_model,
                environment,
                model_factory,
            )
            .await;
        self.exit_toolsets().await?;
        result
    }

    pub async fn run_with_resolved_failover_with_toolsets(
        &self,
        input: RunInput<Deps>,
        config: ResolvedModelConfig,
        model_factory: impl Fn(&str) -> Result<Arc<dyn Model>, AgentError> + Send + Sync,
    ) -> Result<FailoverResult<AgentRunResult>, AgentError> {
        self.enter_toolsets().await?;
        let result = self
            .run_with_resolved_failover(input, config, model_factory)
            .await;
        self.exit_toolsets().await?;
        result
    }

    pub async fn run_stream(&self, input: RunInput<Deps>) -> Result<AgentEventStream, AgentError> {
        let RunInput {
            user_prompt,
            mut message_history,
            deps,
            usage_limits,
            include_system_prompt,
            run_id,
        } = input;

        let deps = Arc::new(deps);
        let mut messages = Vec::new();
        let output_instructions = self.output_schema.as_ref().map(build_output_instructions);
        let run_id = resolve_run_id(run_id);
        let run_started_at = Instant::now();
        let model_name = self.model.name().to_string();

        if include_system_prompt && let Some(prompt) = &self.system_prompt {
            messages.push(ModelMessage::Request(ModelRequest {
                parts: vec![ModelRequestPart::SystemPrompt(
                    crate::messages::SystemPromptPart {
                        content: prompt.clone(),
                    },
                )],
                instructions: None,
            }));
        }

        messages.append(&mut message_history);
        messages.push(ModelMessage::Request(ModelRequest {
            parts: vec![ModelRequestPart::UserPrompt(UserPromptPart {
                content: user_prompt.clone(),
            })],
            instructions: output_instructions.clone(),
        }));

        let run_ctx = RunContext {
            run_id: run_id.clone(),
            deps: Arc::clone(&deps),
            model: Arc::clone(&self.model),
            usage: RunUsage::default(),
            prompt: Some(user_prompt.clone()),
            messages: messages.clone(),
            tool_call_id: None,
            tool_name: None,
        };

        let (tool_defs, tool_map) = match self.collect_tools(&run_ctx).await {
            Ok(result) => result,
            Err(err) => {
                let agent_err = AgentError::Tool(err);
                self.instrumenter.on_run_error(&RunErrorInfo {
                    run_id: run_id.clone(),
                    model_name: model_name.clone(),
                    error: agent_err.to_string(),
                    error_kind: classify_error_kind(&agent_err as &dyn std::error::Error)
                        .map(str::to_string),
                    streaming: true,
                    duration: run_started_at.elapsed(),
                });
                return Err(agent_err);
            }
        };
        let (tool_defs, tool_map) = match self
            .apply_prepare_tools(&run_ctx, tool_defs, tool_map)
            .await
        {
            Ok(result) => result,
            Err(err) => {
                let agent_err = AgentError::Tool(err);
                self.instrumenter.on_run_error(&RunErrorInfo {
                    run_id: run_id.clone(),
                    model_name: model_name.clone(),
                    error: agent_err.to_string(),
                    error_kind: classify_error_kind(&agent_err as &dyn std::error::Error)
                        .map(str::to_string),
                    streaming: true,
                    duration: run_started_at.elapsed(),
                });
                return Err(agent_err);
            }
        };

        let mut params = ModelRequestParameters::new(tool_defs);
        if let Some(schema) = &self.output_schema {
            params = params.with_output_schema(schema.clone());
            params.allow_text_output = self.allow_text_output;
        }

        self.instrumenter.on_run_start(&RunStartInfo {
            run_id: run_id.clone(),
            model_name: model_name.clone(),
            message_count: messages.len(),
            tool_count: params.function_tools.len(),
            output_schema: params.output_schema.is_some(),
            streaming: true,
            allow_text_output: self.allow_text_output,
            output_retries: self.output_retries,
            usage_limits: usage_limits.clone(),
        });

        if let Err(err) = usage_limits.check_request(0) {
            record_usage_limit(
                &self.instrumenter,
                &run_id,
                &model_name,
                &RunUsage::default(),
                &err,
            );
            let agent_err = AgentError::Usage(err);
            self.instrumenter.on_run_error(&RunErrorInfo {
                run_id: run_id.clone(),
                model_name: model_name.clone(),
                error: agent_err.to_string(),
                error_kind: classify_error_kind(&agent_err as &dyn std::error::Error)
                    .map(str::to_string),
                streaming: true,
                duration: run_started_at.elapsed(),
            });
            return Err(agent_err);
        }

        self.instrumenter.on_model_request(&ModelRequestInfo {
            run_id: run_id.clone(),
            model_name: model_name.clone(),
            step: 0,
            message_count: messages.len(),
            tool_count: params.function_tools.len(),
            output_schema: params.output_schema.is_some(),
            streaming: true,
            allow_text_output: self.allow_text_output,
        });

        let response_settings = self.model_settings.as_ref();
        let request_started = Instant::now();
        let stream = match self
            .model
            .request_stream(&messages, response_settings, &params)
            .await
        {
            Ok(stream) => stream,
            Err(err) => {
                self.instrumenter.on_model_error(&ModelErrorInfo {
                    run_id: run_id.clone(),
                    model_name: model_name.clone(),
                    step: 0,
                    error: err.to_string(),
                    error_kind: classify_error_kind(&err as &dyn std::error::Error)
                        .map(str::to_string),
                    duration: request_started.elapsed(),
                    streaming: true,
                });
                let agent_err = AgentError::Model(err);
                self.instrumenter.on_run_error(&RunErrorInfo {
                    run_id: run_id.clone(),
                    model_name: model_name.clone(),
                    error: agent_err.to_string(),
                    error_kind: classify_error_kind(&agent_err as &dyn std::error::Error)
                        .map(str::to_string),
                    streaming: true,
                    duration: run_started_at.elapsed(),
                });
                return Err(agent_err);
            }
        };

        let instrumenter = Arc::clone(&self.instrumenter);
        let output_schema = self.output_schema.clone();
        let allow_text_output = self.allow_text_output;
        let run_id_for_stream = run_id.clone();
        let model_name_for_stream = model_name.clone();
        let run_started_at_for_stream = run_started_at;
        let request_started_for_stream = request_started;
        let usage_limits_for_stream = usage_limits.clone();
        let tool_map_for_stream = tool_map;

        let s = try_stream! {
            let mut usage = RunUsage::default();
            let mut output_text = String::new();
            let mut tool_calls: Vec<ToolCallPart> = Vec::new();
            let mut finish_reason = None;
            let mut saw_usage = false;

            let mut stream = stream;
            while let Some(chunk) = stream.as_mut().next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(err) => {
                        instrumenter.on_model_error(&ModelErrorInfo {
                            run_id: run_id_for_stream.clone(),
                            model_name: model_name_for_stream.clone(),
                            step: 0,
                            error: err.to_string(),
                            error_kind: classify_error_kind(&err as &dyn std::error::Error)
                                .map(str::to_string),
                            duration: request_started_for_stream.elapsed(),
                            streaming: true,
                        });
                        let agent_err = AgentError::Model(err);
                        instrumenter.on_run_error(&RunErrorInfo {
                            run_id: run_id_for_stream.clone(),
                            model_name: model_name_for_stream.clone(),
                            error: agent_err.to_string(),
                            error_kind: classify_error_kind(&agent_err as &dyn std::error::Error)
                                .map(str::to_string),
                            streaming: true,
                            duration: run_started_at_for_stream.elapsed(),
                        });
                        Err(agent_err)?
                    }
                };
                if let Some(delta) = chunk.text_delta {
                    output_text.push_str(&delta);
                    yield AgentStreamEvent::TextDelta(delta);
                }
                if let Some(call) = chunk.tool_call {
                    if let Err(err) = usage_limits_for_stream.check_tool_call(usage.tool_calls) {
                        record_usage_limit(
                            &instrumenter,
                            &run_id_for_stream,
                            &model_name_for_stream,
                            &usage,
                            &err,
                        );
                        let agent_err = AgentError::Usage(err);
                        instrumenter.on_run_error(&RunErrorInfo {
                            run_id: run_id_for_stream.clone(),
                            model_name: model_name_for_stream.clone(),
                            error: agent_err.to_string(),
                            error_kind: classify_error_kind(&agent_err as &dyn std::error::Error)
                                .map(str::to_string),
                            streaming: true,
                            duration: run_started_at_for_stream.elapsed(),
                        });
                        Err(agent_err)?;
                    }
                    usage.incr_tool_call();
                    let kind = tool_map_for_stream
                        .get(&call.name)
                        .map(|entry| entry.definition.kind.clone())
                        .unwrap_or(ToolKind::Function);
                    let sequential = tool_map_for_stream
                        .get(&call.name)
                        .map(|entry| entry.definition.sequential)
                        .unwrap_or(false);
                    let deferred = matches!(kind, ToolKind::External | ToolKind::Unapproved);
                    instrumenter.on_tool_call(&ToolCallInfo {
                        run_id: run_id_for_stream.clone(),
                        tool_name: call.name.clone(),
                        tool_call_id: Some(call.id.clone()),
                        deferred,
                        kind,
                        sequential,
                    });
                    tool_calls.push(call.clone());
                    yield AgentStreamEvent::ToolCall(call);
                }
                if let Some(reason) = chunk.finish_reason {
                    finish_reason = Some(reason);
                }
                if let Some(req_usage) = chunk.usage {
                    saw_usage = true;
                    usage.incr_request(&req_usage);
                }
                if let Err(err) = usage_limits_for_stream.check_after_response(&usage) {
                    record_usage_limit(
                        &instrumenter,
                        &run_id_for_stream,
                        &model_name_for_stream,
                        &usage,
                        &err,
                    );
                    let agent_err = AgentError::Usage(err);
                    instrumenter.on_run_error(&RunErrorInfo {
                        run_id: run_id_for_stream.clone(),
                        model_name: model_name_for_stream.clone(),
                        error: agent_err.to_string(),
                        error_kind: classify_error_kind(&agent_err as &dyn std::error::Error)
                            .map(str::to_string),
                        streaming: true,
                        duration: run_started_at_for_stream.elapsed(),
                    });
                    Err(agent_err)?;
                }
            }

            if !saw_usage {
                usage.requests += 1;
            }

            let mut parts = Vec::new();
            if !output_text.is_empty() {
                parts.push(ModelResponsePart::Text(TextPart {
                    content: output_text.clone(),
                }));
            }
            for call in &tool_calls {
                parts.push(ModelResponsePart::ToolCall(call.clone()));
            }

            let response = ModelResponse {
                parts,
                usage: None,
                model_name: Some(model_name.clone()),
                finish_reason,
            };
            messages.push(ModelMessage::Response(response.clone()));

            instrumenter.on_model_response(&ModelResponseInfo {
                run_id: run_id_for_stream.clone(),
                model_name: model_name_for_stream.clone(),
                step: 0,
                finish_reason: response.finish_reason.clone(),
                usage: usage.clone(),
                tool_calls: tool_calls.len(),
                output_len: output_text.len(),
                duration: request_started_for_stream.elapsed(),
                streaming: true,
            });

            let mut deferred_calls = Vec::new();
            for call in tool_calls {
                let kind = tool_map_for_stream
                    .get(&call.name)
                    .map(|entry| entry.definition.kind.clone())
                    .unwrap_or(ToolKind::Function);
                deferred_calls.push(DeferredToolCall {
                    tool_name: call.name.clone(),
                    tool_call_id: call.id.clone(),
                    arguments: call.arguments.clone(),
                    kind,
                });
            }

            let parsed_output = match output_schema.as_ref() {
                Some(schema) => match validate_output(schema, &output_text, allow_text_output) {
                    Ok(parsed) => parsed,
                    Err(err) => {
                        instrumenter.on_output_validation_error(&OutputValidationErrorInfo {
                            run_id: run_id_for_stream.clone(),
                            model_name: model_name_for_stream.clone(),
                            error: err.clone(),
                            output_len: output_text.len(),
                        });
                        let agent_err = AgentError::OutputValidation(err);
                        instrumenter.on_run_error(&RunErrorInfo {
                            run_id: run_id_for_stream.clone(),
                            model_name: model_name_for_stream.clone(),
                            error: agent_err.to_string(),
                            error_kind: classify_error_kind(&agent_err as &dyn std::error::Error)
                                .map(str::to_string),
                            streaming: true,
                            duration: run_started_at_for_stream.elapsed(),
                        });
                        Err(agent_err)?
                    }
                },
                None => None,
            };

            let state = if deferred_calls.is_empty() {
                AgentRunState::Completed
            } else {
                AgentRunState::Deferred
            };

            let result = AgentRunResult {
                output: output_text,
                usage,
                messages,
                response,
                parsed_output,
                deferred_calls,
                state,
            };

            instrumenter.on_run_end(&RunEndInfo {
                run_id: run_id_for_stream.clone(),
                model_name: model_name_for_stream.clone(),
                state: result.state.clone(),
                usage: result.usage.clone(),
                output_len: result.output.len(),
                deferred_calls: result.deferred_calls.len(),
                tool_calls: result.usage.tool_calls as usize,
                duration: run_started_at_for_stream.elapsed(),
            });

            yield AgentStreamEvent::Done(Box::new(result));
        };

        Ok(Box::pin(s))
    }

    async fn collect_tools(
        &self,
        ctx: &RunContext<Deps>,
    ) -> Result<(Vec<ToolDefinition>, HashMap<String, ToolEntry<Deps>>), ToolError> {
        let mut defs = Vec::new();
        let mut executors: HashMap<String, ToolEntry<Deps>> = HashMap::new();

        for (name, tool) in &self.tools {
            let def = tool.definition();
            executors.insert(
                name.clone(),
                ToolEntry {
                    definition: def.clone(),
                    executor: ToolExecutor::Local(Arc::clone(tool)),
                },
            );
            defs.push(def);
        }

        for toolset in &self.toolsets {
            let list = toolset.list_tools(ctx).await?;
            for def in list {
                if executors.contains_key(&def.name) {
                    warn!(
                        tool = def.name.as_str(),
                        toolset = toolset.name(),
                        "tool name collision, keeping first registration",
                    );
                    continue;
                }
                executors.insert(
                    def.name.clone(),
                    ToolEntry {
                        definition: def.clone(),
                        executor: ToolExecutor::Toolset(Arc::clone(toolset)),
                    },
                );
                defs.push(def);
            }
        }

        Ok((defs, executors))
    }

    async fn apply_prepare_tools(
        &self,
        ctx: &RunContext<Deps>,
        tool_defs: Vec<ToolDefinition>,
        mut tool_map: HashMap<String, ToolEntry<Deps>>,
    ) -> Result<(Vec<ToolDefinition>, HashMap<String, ToolEntry<Deps>>), ToolError> {
        if let Some(prepare) = &self.prepare_tools {
            let filtered = (prepare)(ctx, tool_defs).await?;
            let allowed: HashSet<String> = filtered.iter().map(|def| def.name.clone()).collect();
            debug!(count = allowed.len(), "prepare_tools filtered tool list");
            tool_map.retain(|name, _| allowed.contains(name));
            Ok((filtered, tool_map))
        } else {
            Ok((tool_defs, tool_map))
        }
    }

    async fn execute_tool(
        &self,
        ctx: &RunContext<Deps>,
        entry: &ToolEntry<Deps>,
        call: &ToolCallPart,
    ) -> Result<serde_json::Value, AgentError> {
        let tool_ctx = ctx.for_tool_call(call.id.clone(), call.name.clone());
        match &entry.executor {
            ToolExecutor::Local(tool) => Ok(tool.call(tool_ctx, call.arguments.clone()).await?),
            ToolExecutor::Toolset(toolset) => Ok(toolset
                .call_tool(&tool_ctx, &call.name, call.arguments.clone())
                .await?),
        }
    }

    async fn execute_tool_with_timeout(
        &self,
        ctx: &RunContext<Deps>,
        entry: &ToolEntry<Deps>,
        call: &ToolCallPart,
    ) -> Result<serde_json::Value, AgentError> {
        let started_at = Instant::now();
        self.instrumenter.on_tool_start(&ToolStartInfo {
            run_id: ctx.run_id.clone(),
            tool_name: call.name.clone(),
            tool_call_id: Some(call.id.clone()),
            timeout_secs: entry.definition.timeout,
            sequential: entry.definition.sequential,
        });

        let result = if let Some(timeout_secs) = entry.definition.timeout {
            let duration = Duration::from_secs_f64(timeout_secs.max(0.0));
            match timeout(duration, self.execute_tool(ctx, entry, call)).await {
                Ok(result) => result,
                Err(_) => Err(AgentError::Tool(ToolError::Execution(format!(
                    "tool call timed out after {timeout_secs}s"
                )))),
            }
        } else {
            self.execute_tool(ctx, entry, call).await
        };

        match result {
            Ok(value) => {
                self.instrumenter.on_tool_end(&ToolEndInfo {
                    run_id: ctx.run_id.clone(),
                    tool_name: call.name.clone(),
                    tool_call_id: Some(call.id.clone()),
                    duration: started_at.elapsed(),
                });
                Ok(value)
            }
            Err(err) => {
                self.instrumenter.on_tool_error(&ToolErrorInfo {
                    run_id: ctx.run_id.clone(),
                    tool_name: call.name.clone(),
                    tool_call_id: Some(call.id.clone()),
                    error: err.to_string(),
                    duration: started_at.elapsed(),
                });
                Err(err)
            }
        }
    }
}

fn build_output_instructions(schema: &Value) -> String {
    let schema_text = serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
    format!(
        "Return a JSON object that matches this JSON Schema. Respond with only JSON.\n\n{}",
        schema_text
    )
}

fn validate_output(
    schema: &Value,
    output: &str,
    allow_text: bool,
) -> Result<Option<Value>, String> {
    let parsed: Value = match serde_json::from_str(output) {
        Ok(value) => value,
        Err(err) => {
            if allow_text {
                return Ok(None);
            }
            return Err(format!("Invalid JSON output: {err}"));
        }
    };

    let compiled = JSONSchema::options()
        .with_draft(Draft::Draft7)
        .compile(schema)
        .map_err(|err| format!("Invalid JSON schema: {err}"))?;

    if let Err(errors) = compiled.validate(&parsed) {
        let mut messages = Vec::new();
        for error in errors {
            messages.push(error.to_string());
        }
        return Err(format!(
            "Output did not match schema: {}",
            messages.join("; ")
        ));
    }

    Ok(Some(parsed))
}

fn resolve_run_id(run_id: Option<String>) -> String {
    match run_id {
        Some(id) if !id.trim().is_empty() => id,
        _ => Uuid::new_v4().to_string(),
    }
}

fn record_usage_limit(
    instrumenter: &Arc<dyn Instrumenter>,
    run_id: &str,
    model_name: &str,
    usage: &RunUsage,
    err: &UsageError,
) {
    let (kind, limit) = match *err {
        UsageError::RequestLimitExceeded { limit } => (UsageLimitKind::Requests, limit),
        UsageError::ToolCallsLimitExceeded { limit } => (UsageLimitKind::ToolCalls, limit),
        UsageError::InputTokensLimitExceeded { limit } => (UsageLimitKind::InputTokens, limit),
        UsageError::OutputTokensLimitExceeded { limit } => (UsageLimitKind::OutputTokens, limit),
        UsageError::TotalTokensLimitExceeded { limit } => (UsageLimitKind::TotalTokens, limit),
    };

    instrumenter.on_usage_limit(&UsageLimitInfo {
        run_id: run_id.to_string(),
        model_name: model_name.to_string(),
        kind,
        limit,
        usage: usage.clone(),
    });
}

struct ToolEntry<Deps> {
    definition: ToolDefinition,
    executor: ToolExecutor<Deps>,
}

impl<Deps> Clone for ToolEntry<Deps> {
    fn clone(&self) -> Self {
        Self {
            definition: self.definition.clone(),
            executor: self.executor.clone(),
        }
    }
}

enum ToolExecutor<Deps> {
    Local(Arc<dyn Tool<Deps>>),
    Toolset(Arc<dyn Toolset<Deps>>),
}

impl<Deps> Clone for ToolExecutor<Deps> {
    fn clone(&self) -> Self {
        match self {
            ToolExecutor::Local(tool) => ToolExecutor::Local(Arc::clone(tool)),
            ToolExecutor::Toolset(toolset) => ToolExecutor::Toolset(Arc::clone(toolset)),
        }
    }
}

pub struct RunInput<Deps> {
    pub user_prompt: Vec<UserContent>,
    pub message_history: Vec<ModelMessage>,
    pub deps: Deps,
    pub usage_limits: UsageLimits,
    pub include_system_prompt: bool,
    pub run_id: Option<String>,
}

struct PreparedRunInput<Deps> {
    user_prompt: Vec<UserContent>,
    message_history: Vec<ModelMessage>,
    deps: Arc<Deps>,
    usage_limits: UsageLimits,
    include_system_prompt: bool,
    run_id: String,
}

impl<Deps> Clone for PreparedRunInput<Deps> {
    fn clone(&self) -> Self {
        Self {
            user_prompt: self.user_prompt.clone(),
            message_history: self.message_history.clone(),
            deps: Arc::clone(&self.deps),
            usage_limits: self.usage_limits.clone(),
            include_system_prompt: self.include_system_prompt,
            run_id: self.run_id.clone(),
        }
    }
}

impl<Deps> RunInput<Deps> {
    pub fn new(
        user_prompt: Vec<UserContent>,
        message_history: Vec<ModelMessage>,
        deps: Deps,
        usage_limits: UsageLimits,
    ) -> Self {
        Self {
            user_prompt,
            message_history,
            deps,
            usage_limits,
            include_system_prompt: true,
            run_id: None,
        }
    }

    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentRunState {
    Completed,
    Deferred,
}

#[derive(Clone, Debug)]
pub struct DeferredToolCall {
    pub tool_name: String,
    pub tool_call_id: String,
    pub arguments: Value,
    pub kind: ToolKind,
}

#[derive(Clone, Debug)]
pub struct AgentRunResult {
    pub output: String,
    pub usage: RunUsage,
    pub messages: Vec<ModelMessage>,
    pub response: ModelResponse,
    pub parsed_output: Option<Value>,
    pub deferred_calls: Vec<DeferredToolCall>,
    pub state: AgentRunState,
}

#[derive(Clone, Debug)]
pub enum AgentStreamEvent {
    TextDelta(String),
    ToolCall(ToolCallPart),
    Done(Box<AgentRunResult>),
}

pub type AgentEventStream =
    Pin<Box<dyn Stream<Item = Result<AgentStreamEvent, AgentError>> + Send>>;
