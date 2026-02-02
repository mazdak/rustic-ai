use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_stream::try_stream;
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::lock::Mutex;
use futures::stream::StreamExt;
use reqwest::header::HeaderMap;
use reqwest::{Client, Url};
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::tools::{RunContext, ToolDefinition, ToolError, ToolKind, Toolset};

#[derive(Clone, Debug)]
pub struct McpServerStreamableHttp {
    url: Url,
    headers: HeaderMap,
    timeout: Duration,
    tool_prefix: Option<String>,
    client: Client,
    events_url: Option<Url>,
    cache_tools: bool,
    cache_resources: bool,
    cache_prompts: bool,
    cached_tools: Arc<Mutex<Option<Vec<ToolDefinition>>>>,
    cached_resources: Arc<Mutex<Option<Vec<McpResource>>>>,
    cached_prompts: Arc<Mutex<Option<Vec<McpPrompt>>>>,
}

impl McpServerStreamableHttp {
    pub fn new(url: impl AsRef<str>) -> Result<Self, ToolError> {
        let url = Url::parse(url.as_ref())
            .map_err(|e| ToolError::Toolset(format!("invalid MCP URL: {e}")))?;
        let timeout = Duration::from_secs(10);
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| ToolError::Toolset(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            url,
            headers: HeaderMap::new(),
            timeout,
            tool_prefix: None,
            client,
            events_url: None,
            cache_tools: true,
            cache_resources: true,
            cache_prompts: true,
            cached_tools: Arc::new(Mutex::new(None)),
            cached_resources: Arc::new(Mutex::new(None)),
            cached_prompts: Arc::new(Mutex::new(None)),
        })
    }

    pub fn with_headers(mut self, headers: HeaderMap) -> Self {
        self.headers = headers;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self.client = Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| Client::new());
        self
    }

    pub fn with_tool_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.tool_prefix = Some(prefix.into());
        self
    }

    pub fn with_events_url(mut self, url: impl AsRef<str>) -> Result<Self, ToolError> {
        self.events_url = Some(
            Url::parse(url.as_ref())
                .map_err(|e| ToolError::Toolset(format!("invalid MCP events URL: {e}")))?,
        );
        Ok(self)
    }

    pub fn cache_tools(mut self, enabled: bool) -> Self {
        self.cache_tools = enabled;
        self
    }

    pub fn cache_resources(mut self, enabled: bool) -> Self {
        self.cache_resources = enabled;
        self
    }

    pub fn cache_prompts(mut self, enabled: bool) -> Self {
        self.cache_prompts = enabled;
        self
    }

    pub async fn invalidate_tools_cache(&self) {
        *self.cached_tools.lock().await = None;
    }

    pub async fn invalidate_resources_cache(&self) {
        *self.cached_resources.lock().await = None;
    }

    pub async fn invalidate_prompts_cache(&self) {
        *self.cached_prompts.lock().await = None;
    }

    async fn rpc(&self, method: &str, params: Value) -> Result<Value, ToolError> {
        let request_id = Uuid::new_v4().to_string();
        let payload = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        });
        let response = self
            .client
            .post(self.url.clone())
            .headers(self.headers.clone())
            .json(&payload)
            .send()
            .await
            .map_err(|e| ToolError::Toolset(format!("MCP request failed: {e}")))?;

        let status = response.status();
        let value: Value = response
            .json()
            .await
            .map_err(|e| ToolError::Toolset(format!("MCP response parse failed: {e}")))?;

        if let Some(error) = value.get("error") {
            return Err(ToolError::Toolset(format!(
                "MCP error (status {status}): {error}"
            )));
        }
        value
            .get("result")
            .cloned()
            .ok_or_else(|| ToolError::Toolset("MCP response missing result".to_string()))
    }

    fn prefix_name(&self, name: &str) -> String {
        if let Some(prefix) = &self.tool_prefix {
            format!("{}__{}", prefix, name)
        } else {
            name.to_string()
        }
    }

    fn unprefix_name<'a>(&self, name: &'a str) -> &'a str {
        if let Some(prefix) = &self.tool_prefix {
            let expected = format!("{}__", prefix);
            name.strip_prefix(&expected).unwrap_or(name)
        } else {
            name
        }
    }

    pub async fn list_resources(&self) -> Result<Vec<McpResource>, ToolError> {
        if self.cache_resources
            && let Some(cached) = self.cached_resources.lock().await.clone()
        {
            return Ok(cached);
        }

        let result = self.rpc("resources/list", json!({})).await?;
        let resources: RpcResourcesList = serde_json::from_value(result)
            .map_err(|e| ToolError::Toolset(format!("invalid MCP resources list: {e}")))?;
        if self.cache_resources {
            *self.cached_resources.lock().await = Some(resources.resources.clone());
        }
        Ok(resources.resources)
    }

    pub async fn list_resource_templates(&self) -> Result<Vec<McpResourceTemplate>, ToolError> {
        let result = self.rpc("resources/templates/list", json!({})).await?;
        let templates: RpcResourceTemplatesList = serde_json::from_value(result)
            .map_err(|e| ToolError::Toolset(format!("invalid MCP resource templates list: {e}")))?;
        Ok(templates.resource_templates)
    }

    pub async fn read_resource(&self, uri: &str) -> Result<Value, ToolError> {
        let result = self.rpc("resources/read", json!({ "uri": uri })).await?;
        Ok(result)
    }

    pub async fn list_prompts(&self) -> Result<Vec<McpPrompt>, ToolError> {
        if self.cache_prompts
            && let Some(cached) = self.cached_prompts.lock().await.clone()
        {
            return Ok(cached);
        }

        let result = self.rpc("prompts/list", json!({})).await?;
        let prompts: RpcPromptsList = serde_json::from_value(result)
            .map_err(|e| ToolError::Toolset(format!("invalid MCP prompts list: {e}")))?;
        if self.cache_prompts {
            *self.cached_prompts.lock().await = Some(prompts.prompts.clone());
        }
        Ok(prompts.prompts)
    }

    pub async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> Result<Vec<McpPromptMessage>, ToolError> {
        let mut params = json!({ "name": name });
        if let Some(arguments) = arguments
            && let Value::Object(map) = &mut params
        {
            map.insert("arguments".to_string(), arguments);
        }
        let result = self.rpc("prompts/get", params).await?;
        let prompt: RpcPromptGet = serde_json::from_value(result)
            .map_err(|e| ToolError::Toolset(format!("invalid MCP prompt: {e}")))?;
        Ok(prompt.messages)
    }

    pub async fn sample(&self, params: Value) -> Result<Value, ToolError> {
        self.rpc("sampling/createMessage", params).await
    }

    pub async fn notifications(&self) -> Result<McpNotificationStream, ToolError> {
        let events_url = self
            .events_url
            .clone()
            .ok_or_else(|| ToolError::Toolset("MCP events URL not configured".to_string()))?;

        let response = self
            .client
            .get(events_url)
            .headers(self.headers.clone())
            .send()
            .await
            .map_err(|e| ToolError::Toolset(format!("MCP events request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(ToolError::Toolset(format!(
                "MCP events error status {}",
                response.status()
            )));
        }

        let mut event_stream = response.bytes_stream().eventsource();
        let cached_tools = Arc::clone(&self.cached_tools);
        let cached_resources = Arc::clone(&self.cached_resources);
        let cached_prompts = Arc::clone(&self.cached_prompts);

        let stream = try_stream! {
            while let Some(event) = event_stream.next().await {
                let event = event.map_err(|e| ToolError::Toolset(format!("MCP events stream error: {e}")))?;
                let notification: McpNotification = serde_json::from_str(&event.data)
                    .map_err(|e| ToolError::Toolset(format!("MCP notification parse error: {e}")))?;

                match notification.method.as_str() {
                    "notifications/tools/list_changed" => {
                        *cached_tools.lock().await = None;
                    }
                    "notifications/resources/list_changed" => {
                        *cached_resources.lock().await = None;
                    }
                    "notifications/prompts/list_changed" => {
                        *cached_prompts.lock().await = None;
                    }
                    _ => {}
                }

                yield notification;
            }
        };

        Ok(Box::pin(stream))
    }
}

#[derive(Debug, Deserialize)]
struct RpcToolsList {
    tools: Vec<RpcTool>,
}

#[derive(Debug, Deserialize)]
struct RpcTool {
    name: String,
    description: Option<String>,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
    meta: Option<Value>,
    annotations: Option<Value>,
    #[serde(rename = "outputSchema")]
    output_schema: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct RpcResourcesList {
    resources: Vec<McpResource>,
}

#[derive(Debug, Deserialize)]
struct RpcResourceTemplatesList {
    #[serde(rename = "resourceTemplates")]
    resource_templates: Vec<McpResourceTemplate>,
}

#[derive(Debug, Deserialize)]
struct RpcPromptsList {
    prompts: Vec<McpPrompt>,
}

#[derive(Debug, Deserialize)]
struct RpcPromptGet {
    messages: Vec<McpPromptMessage>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct McpResource {
    pub uri: String,
    pub name: Option<String>,
    pub description: Option<String>,
    #[serde(rename = "mimeType")]
    pub mime_type: Option<String>,
    pub metadata: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct McpResourceTemplate {
    pub name: String,
    pub description: Option<String>,
    pub uri_template: Option<String>,
    pub metadata: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct McpPrompt {
    pub name: String,
    pub description: Option<String>,
    pub arguments: Option<Vec<McpPromptArgument>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct McpPromptArgument {
    pub name: String,
    pub description: Option<String>,
    pub required: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct McpPromptMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Clone, Debug, Deserialize)]
pub struct McpNotification {
    pub method: String,
    pub params: Option<Value>,
}

pub type McpNotificationStream =
    Pin<Box<dyn futures::stream::Stream<Item = Result<McpNotification, ToolError>> + Send>>;

#[async_trait]
impl<Deps> Toolset<Deps> for McpServerStreamableHttp
where
    Deps: Send + Sync,
{
    async fn list_tools(&self, _ctx: &RunContext<Deps>) -> Result<Vec<ToolDefinition>, ToolError> {
        if self.cache_tools
            && let Some(cached) = self.cached_tools.lock().await.clone()
        {
            return Ok(cached);
        }

        let result = self.rpc("tools/list", json!({})).await?;
        let tools: RpcToolsList = serde_json::from_value(result)
            .map_err(|e| ToolError::Toolset(format!("invalid MCP tools list: {e}")))?;
        let mapped: Vec<ToolDefinition> = tools
            .tools
            .into_iter()
            .map(|tool| {
                let mut def = ToolDefinition::new(
                    self.prefix_name(&tool.name),
                    tool.description,
                    tool.input_schema,
                );
                def.kind = ToolKind::Function;
                def.metadata = Some(json!({
                    "meta": tool.meta,
                    "annotations": tool.annotations,
                    "output_schema": tool.output_schema,
                }));
                def
            })
            .collect();

        if self.cache_tools {
            *self.cached_tools.lock().await = Some(mapped.clone());
        }

        Ok(mapped)
    }

    async fn call_tool(
        &self,
        _ctx: &RunContext<Deps>,
        name: &str,
        args: Value,
    ) -> Result<Value, ToolError> {
        let name = self.unprefix_name(name).to_string();
        let result = self
            .rpc("tools/call", json!({"name": name, "arguments": args}))
            .await?;

        if let Some(structured) = result.get("structuredContent") {
            return Ok(structured.clone());
        }

        if let Some(content) = result.get("content")
            && let Some(array) = content.as_array()
            && array.len() == 1
            && let Some(text) = array[0].get("text").and_then(|v| v.as_str())
        {
            return Ok(Value::String(text.to_string()));
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "mcp-http"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpListener;

    #[derive(Default)]
    struct RpcState {
        tool_list_calls: usize,
        tool_call_calls: usize,
        last_tool_name: Option<String>,
        resource_list_calls: usize,
        resource_template_calls: usize,
        resource_read_calls: usize,
        prompt_list_calls: usize,
        prompt_get_calls: usize,
        last_resource_uri: Option<String>,
    }

    async fn spawn_rpc_server(
        state: Arc<Mutex<RpcState>>,
    ) -> Result<(String, tokio::task::JoinHandle<()>), ToolError> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| ToolError::Toolset(format!("bind failed: {e}")))?;
        let addr = listener
            .local_addr()
            .map_err(|e| ToolError::Toolset(format!("addr failed: {e}")))?;
        let base_url = format!("http://{}", addr);

        let handle = tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(_) => break,
                };
                let state = Arc::clone(&state);
                tokio::spawn(async move {
                    let mut reader = BufReader::new(stream);
                    loop {
                        let mut content_length: usize = 0;
                        let mut saw_header = false;
                        loop {
                            let mut line = String::new();
                            let bytes = reader.read_line(&mut line).await.unwrap_or(0);
                            if bytes == 0 {
                                return;
                            }
                            if line == "\r\n" {
                                break;
                            }
                            saw_header = true;
                            let lower = line.to_ascii_lowercase();
                            if lower.starts_with("content-length:")
                                && let Some(value) = line.split(':').nth(1)
                            {
                                content_length = value.trim().parse().unwrap_or(0);
                            }
                        }

                        if !saw_header {
                            return;
                        }

                        let mut body = vec![0u8; content_length];
                        if content_length > 0 && reader.read_exact(&mut body).await.is_err() {
                            return;
                        }

                        let request: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                        let method = request.get("method").and_then(Value::as_str).unwrap_or("");
                        let id = request.get("id").cloned().unwrap_or(Value::Null);
                        let params = request.get("params").cloned().unwrap_or(Value::Null);

                        let result = match method {
                            "tools/list" => {
                                let mut guard = state.lock().unwrap();
                                guard.tool_list_calls += 1;
                                json!({
                                    "tools": [{
                                        "name": "echo",
                                        "description": "Echo input",
                                        "inputSchema": {"type": "object", "properties": {}},
                                        "meta": {"version": 1},
                                        "annotations": {"note": "test"},
                                        "outputSchema": {"type": "object"}
                                    }]
                                })
                            }
                            "tools/call" => {
                                let mut guard = state.lock().unwrap();
                                guard.tool_call_calls += 1;
                                guard.last_tool_name = params
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .map(|v| v.to_string());
                                match guard.last_tool_name.as_deref() {
                                    Some("structured") => {
                                        json!({ "structuredContent": { "ok": true } })
                                    }
                                    Some("text") => {
                                        json!({ "content": [ { "text": "hi" } ] })
                                    }
                                    _ => json!({ "value": 42 }),
                                }
                            }
                            "resources/list" => {
                                let mut guard = state.lock().unwrap();
                                guard.resource_list_calls += 1;
                                json!({ "resources": [{
                                    "uri": "file:///tmp/example.txt",
                                    "name": "example",
                                    "description": "example resource",
                                    "mimeType": "text/plain",
                                    "metadata": {"version": 1}
                                }] })
                            }
                            "resources/templates/list" => {
                                let mut guard = state.lock().unwrap();
                                guard.resource_template_calls += 1;
                                json!({ "resourceTemplates": [{
                                    "name": "example-template",
                                    "description": "example template",
                                    "uriTemplate": "file:///tmp/{name}",
                                    "metadata": {"source": "test"}
                                }] })
                            }
                            "resources/read" => {
                                let mut guard = state.lock().unwrap();
                                guard.resource_read_calls += 1;
                                guard.last_resource_uri = params
                                    .get("uri")
                                    .and_then(Value::as_str)
                                    .map(|v| v.to_string());
                                json!({ "contents": [{
                                    "uri": guard.last_resource_uri.clone().unwrap_or_default(),
                                    "text": "hello"
                                }] })
                            }
                            "prompts/list" => {
                                let mut guard = state.lock().unwrap();
                                guard.prompt_list_calls += 1;
                                json!({ "prompts": [{
                                    "name": "example-prompt",
                                    "description": "example prompt",
                                    "arguments": [{"name": "topic", "required": true}]
                                }] })
                            }
                            "prompts/get" => {
                                let mut guard = state.lock().unwrap();
                                guard.prompt_get_calls += 1;
                                json!({ "messages": [{
                                    "role": "user",
                                    "content": "hi"
                                }] })
                            }
                            _ => {
                                json!({ "error": { "message": "unknown method" } })
                            }
                        };

                        let response = if result.get("error").is_some() {
                            json!({ "jsonrpc": "2.0", "id": id, "error": result["error"] })
                        } else {
                            json!({ "jsonrpc": "2.0", "id": id, "result": result })
                        };
                        let response_bytes = response.to_string();
                        let stream = reader.get_mut();
                        let _ = stream
                            .write_all(
                                format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                                    response_bytes.len(),
                                    response_bytes
                                )
                                .as_bytes(),
                            )
                            .await;
                    }
                });
            }
        });

        Ok((base_url, handle))
    }

    #[tokio::test]
    async fn list_tools_applies_prefix_and_caches() {
        let state = Arc::new(Mutex::new(RpcState::default()));
        let (base_url, handle) = spawn_rpc_server(Arc::clone(&state)).await.expect("server");

        let client = McpServerStreamableHttp::new(base_url)
            .expect("client")
            .with_tool_prefix("remote");

        let ctx = RunContext {
            run_id: "run".to_string(),
            deps: Arc::new(()),
            model: Arc::new(crate::providers::openai::OpenAIChatModel::new(
                "gpt-test",
                "key".to_string(),
                Url::parse("https://example.com/").expect("url"),
                None,
            )),
            usage: crate::usage::RunUsage::default(),
            prompt: None,
            messages: Arc::new(Vec::new()),
            tool_call_id: None,
            tool_name: None,
        };

        let tools = client.list_tools(&ctx).await.expect("tools");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "remote__echo");
        assert!(tools[0].metadata.is_some());

        let tools_again = client.list_tools(&ctx).await.expect("tools again");
        assert_eq!(tools_again.len(), 1);

        let calls = state.lock().unwrap().tool_list_calls;
        assert_eq!(calls, 1, "tools list should be cached");

        client.invalidate_tools_cache().await;
        let _ = client
            .list_tools(&ctx)
            .await
            .expect("tools after invalidate");
        let calls = state.lock().unwrap().tool_list_calls;
        assert_eq!(calls, 2, "tools list should be refreshed");

        handle.abort();
    }

    #[tokio::test]
    async fn call_tool_returns_structured_or_text() {
        let state = Arc::new(Mutex::new(RpcState::default()));
        let (base_url, handle) = spawn_rpc_server(Arc::clone(&state)).await.expect("server");

        let client = McpServerStreamableHttp::new(base_url)
            .expect("client")
            .with_tool_prefix("remote");

        let ctx = RunContext {
            run_id: "run".to_string(),
            deps: Arc::new(()),
            model: Arc::new(crate::providers::openai::OpenAIChatModel::new(
                "gpt-test",
                "key".to_string(),
                Url::parse("https://example.com/").expect("url"),
                None,
            )),
            usage: crate::usage::RunUsage::default(),
            prompt: None,
            messages: Arc::new(Vec::new()),
            tool_call_id: None,
            tool_name: None,
        };

        let structured = client
            .call_tool(&ctx, "remote__structured", json!({}))
            .await
            .expect("structured");
        assert_eq!(structured, json!({"ok": true}));

        let text = client
            .call_tool(&ctx, "remote__text", json!({}))
            .await
            .expect("text");
        assert_eq!(text, Value::String("hi".to_string()));

        let calls = state.lock().unwrap().tool_call_calls;
        assert_eq!(calls, 2);

        handle.abort();
    }

    #[tokio::test]
    async fn list_resources_caches_and_invalidates() {
        let state = Arc::new(Mutex::new(RpcState::default()));
        let (base_url, handle) = spawn_rpc_server(Arc::clone(&state)).await.expect("server");

        let client = McpServerStreamableHttp::new(base_url)
            .expect("client")
            .cache_resources(true);

        let resources = client.list_resources().await.expect("resources");
        assert_eq!(resources.len(), 1);
        let resources_again = client.list_resources().await.expect("resources again");
        assert_eq!(resources_again.len(), 1);

        let calls = state.lock().unwrap().resource_list_calls;
        assert_eq!(calls, 1);

        client.invalidate_resources_cache().await;
        let _ = client.list_resources().await.expect("resources refreshed");
        let calls = state.lock().unwrap().resource_list_calls;
        assert_eq!(calls, 2);

        handle.abort();
    }

    #[tokio::test]
    async fn list_prompts_and_get_prompt() {
        let state = Arc::new(Mutex::new(RpcState::default()));
        let (base_url, handle) = spawn_rpc_server(Arc::clone(&state)).await.expect("server");

        let client = McpServerStreamableHttp::new(base_url)
            .expect("client")
            .cache_prompts(true);

        let prompts = client.list_prompts().await.expect("prompts");
        assert_eq!(prompts.len(), 1);
        let prompts_again = client.list_prompts().await.expect("prompts again");
        assert_eq!(prompts_again.len(), 1);

        let calls = state.lock().unwrap().prompt_list_calls;
        assert_eq!(calls, 1);

        let messages = client
            .get_prompt("example-prompt", None)
            .await
            .expect("prompt messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");

        let calls = state.lock().unwrap().prompt_get_calls;
        assert_eq!(calls, 1);

        handle.abort();
    }

    #[tokio::test]
    async fn list_templates_and_read_resource() {
        let state = Arc::new(Mutex::new(RpcState::default()));
        let (base_url, handle) = spawn_rpc_server(Arc::clone(&state)).await.expect("server");

        let client = McpServerStreamableHttp::new(base_url).expect("client");
        let templates = client.list_resource_templates().await.expect("templates");
        assert_eq!(templates.len(), 1);
        assert_eq!(templates[0].name, "example-template");

        let content = client
            .read_resource("file:///tmp/example.txt")
            .await
            .expect("read resource");
        let text = content
            .get("contents")
            .and_then(|value| value.as_array())
            .and_then(|items| items.first())
            .and_then(|item| item.get("text"))
            .and_then(|value| value.as_str())
            .expect("text");
        assert_eq!(text, "hello");

        let state = state.lock().unwrap();
        assert_eq!(state.resource_template_calls, 1);
        assert_eq!(state.resource_read_calls, 1);
        assert_eq!(
            state.last_resource_uri.as_deref(),
            Some("file:///tmp/example.txt")
        );

        handle.abort();
    }

    #[tokio::test]
    async fn cache_prompts_can_be_disabled() {
        let state = Arc::new(Mutex::new(RpcState::default()));
        let (base_url, handle) = spawn_rpc_server(Arc::clone(&state)).await.expect("server");

        let client = McpServerStreamableHttp::new(base_url)
            .expect("client")
            .cache_prompts(false);

        let _ = client.list_prompts().await.expect("prompts");
        let _ = client.list_prompts().await.expect("prompts again");

        let calls = state.lock().unwrap().prompt_list_calls;
        assert_eq!(calls, 2);

        handle.abort();
    }
}
