use std::sync::Arc;

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose};
use reqwest::{Client, Url};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::messages::{
    ModelMessage, ModelRequestPart, ModelResponse, ModelResponsePart, TextPart, ToolCallPart,
    UserContent,
};
use crate::model::{Model, ModelError, ModelRequestParameters, ModelSettings, OutputMode};
use crate::providers::{Provider, ProviderError};
use crate::usage::RequestUsage;

fn map_reqwest_error(label: &str, error: reqwest::Error) -> ModelError {
    if error.is_timeout() {
        return ModelError::Timeout;
    }
    if error.is_connect() {
        return ModelError::Transport(format!("{label} connect error: {error}"));
    }
    ModelError::Transport(format!("{label} request failed: {error}"))
}

fn truncate_error_body(body: &str) -> String {
    const LIMIT: usize = 512;
    if body.len() <= LIMIT {
        body.to_string()
    } else {
        format!("{}... ({} bytes)", &body[..LIMIT], body.len())
    }
}

fn normalize_tool_call_id(id: Option<String>) -> String {
    match id {
        Some(value) if !value.trim().is_empty() => value,
        _ => format!("call_{}", Uuid::new_v4().simple()),
    }
}

fn normalize_tool_call_id_str(id: &str) -> String {
    if id.trim().is_empty() {
        format!("call_{}", Uuid::new_v4().simple())
    } else {
        id.to_string()
    }
}

fn tool_return_content(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| value.to_string()),
    }
}

fn is_text_like_media_type(media_type: &str) -> bool {
    media_type.starts_with("text/")
        || matches!(
            media_type,
            "application/json"
                | "application/xml"
                | "application/xhtml+xml"
                | "application/javascript"
                | "application/x-www-form-urlencoded"
        )
}

fn is_pdf_url(url: &str) -> bool {
    url.split('?')
        .next()
        .is_some_and(|path| path.to_lowercase().ends_with(".pdf"))
}

#[derive(Clone, Debug)]
pub struct AnthropicProvider {
    api_key: String,
    base_url: Url,
}

impl AnthropicProvider {
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl AsRef<str>,
    ) -> Result<Self, ProviderError> {
        let url = Url::parse(base_url.as_ref())
            .map_err(|_| ProviderError::InvalidModel(base_url.as_ref().to_string()))?;
        Ok(Self {
            api_key: api_key.into(),
            base_url: url,
        })
    }

    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| ProviderError::MissingApiKey("anthropic".to_string()))?;
        Self::new(api_key, "https://api.anthropic.com")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{
        BinaryContent, DocumentUrl, ModelMessage, ModelRequest, ModelRequestPart, ModelResponse,
        ModelResponsePart, ToolCallPart, ToolReturnPart,
    };
    use base64::engine::general_purpose::STANDARD;
    use serde_json::{Value, json};
    use std::path::PathBuf;

    fn fixture_bytes(name: &str) -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name);
        std::fs::read(path).expect("fixture read")
    }

    #[test]
    fn convert_user_content_handles_documents_and_images() {
        let image_bytes = fixture_bytes("fixture.jpg");
        let pdf_bytes = fixture_bytes("fixture.pdf");

        let content = vec![
            UserContent::Binary(BinaryContent {
                data: image_bytes.clone(),
                media_type: "image/jpeg".to_string(),
            }),
            UserContent::Binary(BinaryContent {
                data: pdf_bytes.clone(),
                media_type: "application/pdf".to_string(),
            }),
            UserContent::Document(DocumentUrl {
                url: "https://example.com/fixture.pdf".to_string(),
                media_type: None,
            }),
        ];

        let parts = convert_user_content(&content);
        assert_eq!(parts.len(), 3);

        let image = &parts[0];
        assert_eq!(image.get("type"), Some(&Value::String("image".to_string())));
        let image_source = image.get("source").expect("image source");
        assert_eq!(
            image_source.get("type"),
            Some(&Value::String("base64".to_string()))
        );
        assert_eq!(
            image_source.get("media_type"),
            Some(&Value::String("image/jpeg".to_string()))
        );
        assert_eq!(
            image_source.get("data"),
            Some(&Value::String(STANDARD.encode(&image_bytes)))
        );

        let pdf = &parts[1];
        assert_eq!(
            pdf.get("type"),
            Some(&Value::String("document".to_string()))
        );
        let pdf_source = pdf.get("source").expect("pdf source");
        assert_eq!(
            pdf_source.get("type"),
            Some(&Value::String("base64".to_string()))
        );
        assert_eq!(
            pdf_source.get("media_type"),
            Some(&Value::String("application/pdf".to_string()))
        );
        assert_eq!(
            pdf_source.get("data"),
            Some(&Value::String(STANDARD.encode(&pdf_bytes)))
        );

        let doc = &parts[2];
        assert_eq!(
            doc.get("type"),
            Some(&Value::String("document".to_string()))
        );
        let doc_source = doc.get("source").expect("doc source");
        assert_eq!(
            doc_source.get("type"),
            Some(&Value::String("url".to_string()))
        );
        assert_eq!(
            doc_source.get("url"),
            Some(&Value::String(
                "https://example.com/fixture.pdf".to_string()
            ))
        );
    }

    #[test]
    fn split_system_replays_tool_calls() {
        let messages = vec![
            ModelMessage::Response(ModelResponse {
                parts: vec![ModelResponsePart::ToolCall(ToolCallPart {
                    id: "call-1".to_string(),
                    name: "get_data".to_string(),
                    arguments: json!({"a": 1}),
                })],
                usage: None,
                model_name: None,
                finish_reason: None,
            }),
            ModelMessage::Request(ModelRequest {
                parts: vec![ModelRequestPart::ToolReturn(ToolReturnPart {
                    tool_name: "get_data".to_string(),
                    tool_call_id: "call-1".to_string(),
                    content: json!({"ok": true}),
                })],
                instructions: None,
            }),
        ];

        let (_system, out) = AnthropicModel::split_system(&messages);
        assert_eq!(out.len(), 2);

        let assistant = out[0].as_object().expect("assistant message");
        assert_eq!(
            assistant.get("role"),
            Some(&Value::String("assistant".to_string()))
        );
        let assistant_content = assistant
            .get("content")
            .and_then(|value| value.as_array())
            .expect("assistant content");
        let tool_use = assistant_content
            .iter()
            .find(|part| part.get("type") == Some(&Value::String("tool_use".to_string())))
            .expect("tool_use part");
        assert_eq!(
            tool_use.get("id"),
            Some(&Value::String("call-1".to_string()))
        );
        assert_eq!(
            tool_use.get("name"),
            Some(&Value::String("get_data".to_string()))
        );
        assert_eq!(tool_use.get("input"), Some(&json!({"a": 1})));

        let user = out[1].as_object().expect("tool result message");
        assert_eq!(user.get("role"), Some(&Value::String("user".to_string())));
        let user_content = user
            .get("content")
            .and_then(|value| value.as_array())
            .expect("user content");
        let tool_result = user_content
            .iter()
            .find(|part| part.get("type") == Some(&Value::String("tool_result".to_string())))
            .expect("tool_result part");
        assert_eq!(
            tool_result.get("tool_use_id"),
            Some(&Value::String("call-1".to_string()))
        );
        assert_eq!(
            tool_result.get("content"),
            Some(&Value::String("{\"ok\":true}".to_string()))
        );
    }

    #[test]
    fn helper_functions_cover_ids_and_media() {
        let id = normalize_tool_call_id(Some("".to_string()));
        assert!(id.starts_with("call_"));
        let id = normalize_tool_call_id_str("");
        assert!(id.starts_with("call_"));

        assert!(is_text_like_media_type("text/plain"));
        assert!(is_text_like_media_type("application/json"));
        assert!(!is_text_like_media_type("image/png"));

        assert_eq!(tool_return_content(&json!("ok")), "ok");
        assert_eq!(tool_return_content(&json!({"a": 1})), "{\"a\":1}");

        assert!(is_pdf_url("https://example.com/doc.pdf"));
        assert!(is_pdf_url("https://example.com/doc.pdf?x=1"));
        assert!(!is_pdf_url("https://example.com/doc.txt"));
    }

    #[test]
    fn truncate_error_body_limits_length() {
        let truncated = truncate_error_body(&"a".repeat(600));
        assert!(truncated.contains("bytes"));
    }
}

impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self, model: &str, settings: Option<ModelSettings>) -> Arc<dyn Model> {
        Arc::new(AnthropicModel::new(
            model,
            self.api_key.clone(),
            self.base_url.clone(),
            settings,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct AnthropicModel {
    model: String,
    api_key: String,
    base_url: Url,
    client: Client,
    default_settings: Option<ModelSettings>,
}

impl AnthropicModel {
    pub fn new(
        model: impl Into<String>,
        api_key: String,
        base_url: Url,
        settings: Option<ModelSettings>,
    ) -> Self {
        Self {
            model: model.into(),
            api_key,
            base_url,
            client: Client::new(),
            default_settings: settings,
        }
    }

    fn endpoint(&self) -> Result<Url, ModelError> {
        self.base_url
            .join("v1/messages")
            .map_err(|e| ModelError::Provider(format!("invalid base url: {e}")))
    }

    fn split_system(messages: &[ModelMessage]) -> (Option<String>, Vec<Value>) {
        let mut system_parts = Vec::new();
        let mut out = Vec::new();

        for message in messages {
            match message {
                ModelMessage::Request(req) => {
                    if let Some(instructions) = req
                        .instructions
                        .as_ref()
                        .filter(|value| !value.trim().is_empty())
                    {
                        system_parts.push(instructions.to_string());
                    }
                    for part in &req.parts {
                        match part {
                            ModelRequestPart::SystemPrompt(prompt) => {
                                system_parts.push(prompt.content.clone());
                            }
                            ModelRequestPart::UserPrompt(prompt) => {
                                out.push(json!({
                                    "role": "user",
                                    "content": convert_user_content(&prompt.content)
                                }));
                            }
                            ModelRequestPart::ToolReturn(tool_return) => {
                                let content = tool_return_content(&tool_return.content);
                                out.push(json!({
                                    "role": "user",
                                    "content": [{
                                        "type": "tool_result",
                                        "tool_use_id": normalize_tool_call_id_str(&tool_return.tool_call_id),
                                        "content": content,
                                        "is_error": false,
                                    }]
                                }));
                            }
                            ModelRequestPart::RetryPrompt(retry) => {
                                if retry.tool_name.is_some() {
                                    out.push(json!({
                                        "role": "user",
                                        "content": [{
                                            "type": "tool_result",
                                            "tool_use_id": normalize_tool_call_id(retry.tool_call_id.clone()),
                                            "content": retry.content,
                                            "is_error": true,
                                        }]
                                    }));
                                } else {
                                    out.push(json!({
                                        "role": "user",
                                        "content": [{"type": "text", "text": retry.content}]
                                    }));
                                }
                            }
                        }
                    }
                }
                ModelMessage::Response(res) => {
                    let mut content = Vec::new();
                    if let Some(text) = res.text() {
                        content.push(json!({"type": "text", "text": text}));
                    }
                    for call in res.tool_calls() {
                        content.push(json!({
                            "type": "tool_use",
                            "id": normalize_tool_call_id_str(&call.id),
                            "name": call.name,
                            "input": call.arguments,
                        }));
                    }

                    if !content.is_empty() {
                        out.push(json!({
                            "role": "assistant",
                            "content": content
                        }));
                    }
                }
            }
        }

        let system = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n\n"))
        };

        (system, out)
    }
}

fn convert_user_content(content: &[UserContent]) -> Vec<Value> {
    let mut parts = Vec::new();
    for item in content {
        match item {
            UserContent::Text(text) => parts.push(json!({"type": "text", "text": text})),
            UserContent::Image(image) => parts.push(json!({
                "type": "image",
                "source": {"type": "url", "url": image.url}
            })),
            UserContent::Binary(binary) => {
                if binary.media_type.starts_with("image/") {
                    let encoded = general_purpose::STANDARD.encode(&binary.data);
                    parts.push(json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": binary.media_type,
                            "data": encoded
                        }
                    }));
                } else if is_text_like_media_type(&binary.media_type) {
                    match std::str::from_utf8(&binary.data) {
                        Ok(text) => parts.push(json!({"type": "text", "text": text})),
                        Err(_) => parts.push(json!({
                            "type": "text",
                            "text": format!("[binary content: {} bytes]", binary.data.len())
                        })),
                    }
                } else if binary.media_type == "application/pdf" {
                    let encoded = general_purpose::STANDARD.encode(&binary.data);
                    parts.push(json!({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": binary.media_type,
                            "data": encoded
                        }
                    }));
                } else {
                    parts.push(json!({
                        "type": "text",
                        "text": format!("[binary content: {} bytes]", binary.data.len())
                    }));
                }
            }
            UserContent::Audio(audio) => parts.push(json!({
                "type": "text",
                "text": format!("[audio: {}]", audio.url)
            })),
            UserContent::Video(video) => parts.push(json!({
                "type": "text",
                "text": format!("[video: {}]", video.url)
            })),
            UserContent::Document(doc) => {
                let media_type = doc.media_type.as_deref();
                if media_type == Some("application/pdf")
                    || (media_type.is_none() && is_pdf_url(&doc.url))
                {
                    parts.push(json!({
                        "type": "document",
                        "source": {"type": "url", "url": doc.url}
                    }))
                } else {
                    parts.push(json!({
                        "type": "text",
                        "text": format!("[document: {}]", doc.url)
                    }))
                }
            }
        }
    }
    parts
}

#[async_trait]
impl Model for AnthropicModel {
    fn name(&self) -> &str {
        &self.model
    }

    async fn request(
        &self,
        messages: &[ModelMessage],
        settings: Option<&ModelSettings>,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        tracing::debug!(
            model = %self.model,
            tool_count = params.function_tools.len(),
            output_schema = params.output_schema.is_some(),
            "Anthropic request"
        );
        let (system, messages) = Self::split_system(messages);
        let mut body = Map::new();
        body.insert("model".to_string(), Value::String(self.model.clone()));
        body.insert("messages".to_string(), Value::Array(messages));
        if let Some(system) = system {
            body.insert("system".to_string(), Value::String(system));
        }

        if !params.function_tools.is_empty() {
            let tools = params
                .function_tools
                .iter()
                .map(|tool| {
                    json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters_json_schema,
                    })
                })
                .collect();
            body.insert("tools".to_string(), Value::Array(tools));
            let mut tool_choice = json!({"type": "auto"});
            if params.function_tools.iter().any(|tool| tool.sequential)
                && let Value::Object(map) = &mut tool_choice
            {
                map.insert("disable_parallel_tool_use".to_string(), Value::Bool(true));
            }
            body.insert("tool_choice".to_string(), tool_choice);
        }

        if params.output_mode == OutputMode::JsonSchema
            && let Some(schema) = params.output_schema.clone()
        {
            body.insert(
                "output_format".to_string(),
                json!({
                    "type": "json_schema",
                    "schema": schema
                }),
            );
        }

        if let Some(settings) = &self.default_settings {
            for (key, value) in settings {
                body.insert(key.clone(), value.clone());
            }
        }

        if let Some(settings) = settings {
            for (key, value) in settings {
                body.insert(key.clone(), value.clone());
            }
        }

        if !body.contains_key("max_tokens") {
            body.insert("max_tokens".to_string(), Value::Number(1024.into()));
        }

        let mut request = self
            .client
            .post(self.endpoint()?)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01");

        if params.output_mode == OutputMode::JsonSchema && params.output_schema.is_some() {
            request = request.header("anthropic-beta", "structured-outputs-2025-11-13");
        }

        let response = request
            .json(&Value::Object(body))
            .send()
            .await
            .map_err(|e| map_reqwest_error("Anthropic", e))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            tracing::error!(
                status = status.as_u16(),
                model = %self.model,
                body = %truncate_error_body(&body),
                "Anthropic request failed"
            );
            return Err(ModelError::HttpStatus {
                status: status.as_u16(),
            });
        }

        let body: AnthropicResponse = response.json().await.map_err(|e| {
            tracing::error!(
                error = %e,
                model = %self.model,
                "Anthropic response parse failed"
            );
            ModelError::Provider(format!("Anthropic response parse failed: {e}"))
        })?;

        let mut parts = Vec::new();
        for content in body.content {
            match content.kind.as_str() {
                "text" => {
                    if let Some(text) = content.text {
                        parts.push(ModelResponsePart::Text(TextPart { content: text }));
                    }
                }
                "tool_use" => {
                    parts.push(ModelResponsePart::ToolCall(ToolCallPart {
                        id: normalize_tool_call_id(content.id),
                        name: content.name.unwrap_or_else(|| "tool".to_string()),
                        arguments: content.input.unwrap_or_else(|| Value::Object(Map::new())),
                    }));
                }
                _ => {}
            }
        }

        let usage = body.usage.map(|usage| RequestUsage {
            input_tokens: usage.input_tokens.unwrap_or(0),
            output_tokens: usage.output_tokens.unwrap_or(0),
            ..Default::default()
        });

        Ok(ModelResponse {
            parts,
            usage,
            model_name: Some(self.model.clone()),
            finish_reason: body.stop_reason,
        })
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    stop_reason: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
    id: Option<String>,
    name: Option<String>,
    input: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}
