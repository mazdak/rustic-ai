use std::collections::HashMap;
use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose};
use eventsource_stream::Eventsource;
use futures::stream::StreamExt;
use reqwest::{Client, Url};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::json_schema::transform_openai_schema;
use crate::messages::{
    BinaryContent, ModelMessage, ModelRequestPart, ModelResponse, ModelResponsePart,
    ProviderItemPart, TextPart, ToolCallPart, UserContent,
};
use crate::model::{
    Model, ModelError, ModelRequestParameters, ModelSettings, ModelStream, OutputMode, StreamChunk,
};
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

fn join_path(base: &Url, path: &str) -> Result<Url, ModelError> {
    let mut url = base.clone();
    let base_path = url.path().trim_end_matches('/');
    let path = path.trim_start_matches('/');
    let new_path = if base_path.is_empty() || base_path == "/" {
        format!("/{path}")
    } else {
        format!("{base_path}/{path}")
    };
    url.set_path(&new_path);
    Ok(url)
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

fn tool_call_arguments(value: &Value) -> String {
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

fn audio_format_from_media_type(media_type: &str) -> Option<&'static str> {
    match media_type {
        "audio/wav" | "audio/x-wav" => Some("wav"),
        "audio/mpeg" | "audio/mp3" => Some("mp3"),
        "audio/ogg" | "audio/ogg;codecs=opus" => Some("ogg"),
        "audio/flac" => Some("flac"),
        "audio/aiff" => Some("aiff"),
        "audio/aac" => Some("aac"),
        _ => None,
    }
}

fn parse_data_url_base64(url: &str) -> Option<(String, String)> {
    let data_url = url.strip_prefix("data:")?;
    let (meta, data) = data_url.split_once(',')?;
    let (media_type, encoding) = meta.split_once(';')?;
    if encoding != "base64" || media_type.trim().is_empty() {
        return None;
    }
    Some((media_type.to_string(), data.to_string()))
}

fn normalize_stream_tool_call_id(id: Option<String>, index: Option<usize>) -> String {
    if let Some(value) = id.filter(|value| !value.trim().is_empty()) {
        value
    } else if let Some(index) = index {
        format!("call_{index}")
    } else {
        normalize_tool_call_id(None)
    }
}

fn contains_audio(messages: &[ModelMessage]) -> bool {
    for message in messages {
        if let ModelMessage::Request(req) = message {
            for part in &req.parts {
                if let ModelRequestPart::UserPrompt(prompt) = part {
                    for item in &prompt.content {
                        match item {
                            UserContent::Audio(_) => return true,
                            UserContent::Binary(binary) => {
                                if binary.media_type.starts_with("audio/") {
                                    return true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    false
}

fn is_responses_only_model(model: &str) -> bool {
    let lowered = model.to_lowercase();
    lowered.starts_with("gpt-5")
        || lowered.starts_with("gpt-4.1")
        || lowered.starts_with("o1")
        || lowered.starts_with("o3")
}

fn prefers_responses(model: &str) -> bool {
    let lowered = model.to_lowercase();
    is_responses_only_model(model)
        || lowered.starts_with("gpt-4o")
        || lowered.starts_with("gpt-4.1")
        || lowered.starts_with("o1")
        || lowered.starts_with("o3")
}

#[derive(Clone, Debug)]
pub(crate) struct OpenAIChatCapabilities {
    pub(crate) supports_response_format: bool,
    pub(crate) supports_parallel_tool_calls: bool,
    pub(crate) reject_binary_images: bool,
}

impl Default for OpenAIChatCapabilities {
    fn default() -> Self {
        Self {
            supports_response_format: true,
            supports_parallel_tool_calls: true,
            reject_binary_images: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OpenAIProvider {
    api_key: String,
    base_url: Url,
}

impl OpenAIProvider {
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
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| ProviderError::MissingApiKey("openai".to_string()))?;
        Self::new(api_key, "https://api.openai.com/v1")
    }

    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Result<Self, ProviderError> {
        self.base_url = Url::parse(base_url.as_ref())
            .map_err(|_| ProviderError::InvalidModel(base_url.as_ref().to_string()))?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::STANDARD;
    use serde_json::{Value, json};
    use std::path::PathBuf;

    use crate::messages::{
        ModelMessage, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart,
        ProviderItemPart, TextPart, ToolCallPart, ToolReturnPart,
    };

    fn fixture_bytes(name: &str) -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name);
        std::fs::read(path).expect("fixture read")
    }

    #[test]
    fn convert_user_content_handles_binary_media() {
        let model = OpenAIChatModel::new(
            "gpt-4o-mini",
            "test-key".to_string(),
            Url::parse("https://example.com/").expect("valid url"),
            None,
        );

        let image_bytes = fixture_bytes("fixture.jpg");
        let audio_bytes = fixture_bytes("fixture.m4a");
        let pdf_bytes = fixture_bytes("fixture.pdf");

        let content = vec![
            UserContent::Binary(BinaryContent {
                data: image_bytes.clone(),
                media_type: "image/jpeg".to_string(),
            }),
            UserContent::Binary(BinaryContent {
                data: audio_bytes.clone(),
                media_type: "audio/aac".to_string(),
            }),
            UserContent::Binary(BinaryContent {
                data: pdf_bytes.clone(),
                media_type: "application/pdf".to_string(),
            }),
        ];

        let value = model
            .convert_user_content(&content)
            .expect("convert user content");
        let parts = value.as_array().expect("parts array");
        assert_eq!(parts.len(), 3);

        let image = &parts[0];
        assert_eq!(
            image.get("type"),
            Some(&Value::String("image_url".to_string()))
        );
        let image_url = image
            .get("image_url")
            .and_then(|value| value.get("url"))
            .and_then(|value| value.as_str())
            .expect("image url");
        let expected_image = format!("data:image/jpeg;base64,{}", STANDARD.encode(&image_bytes));
        assert_eq!(image_url, expected_image);

        let audio = &parts[1];
        assert_eq!(
            audio.get("type"),
            Some(&Value::String("input_audio".to_string()))
        );
        let audio_input = audio.get("input_audio").expect("input_audio");
        assert_eq!(
            audio_input.get("format"),
            Some(&Value::String("aac".to_string()))
        );
        let audio_data = audio_input
            .get("data")
            .and_then(|value| value.as_str())
            .expect("audio data");
        assert_eq!(audio_data, STANDARD.encode(&audio_bytes));

        let pdf = &parts[2];
        assert_eq!(pdf.get("type"), Some(&Value::String("text".to_string())));
        let pdf_text = pdf
            .get("text")
            .and_then(|value| value.as_str())
            .expect("pdf text");
        let expected_text = format!("[binary content: {} bytes]", pdf_bytes.len());
        assert_eq!(pdf_text, expected_text);
    }

    #[test]
    fn make_messages_replays_tool_calls() {
        let model = OpenAIChatModel::new(
            "gpt-4o-mini",
            "test-key".to_string(),
            Url::parse("https://example.com/").expect("valid url"),
            None,
        );

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

        let out = model.make_messages(&messages).expect("make messages");
        assert_eq!(out.len(), 2);

        let assistant = out[0].as_object().expect("assistant message");
        assert_eq!(
            assistant.get("role"),
            Some(&Value::String("assistant".to_string()))
        );
        assert_eq!(assistant.get("content"), Some(&Value::Null));
        let tool_calls = assistant
            .get("tool_calls")
            .and_then(|value| value.as_array())
            .expect("tool_calls");
        assert_eq!(tool_calls.len(), 1);
        let call = &tool_calls[0];
        assert_eq!(call.get("id"), Some(&Value::String("call-1".to_string())));
        let function = call.get("function").expect("function");
        assert_eq!(
            function.get("name"),
            Some(&Value::String("get_data".to_string()))
        );
        assert_eq!(
            function.get("arguments"),
            Some(&Value::String("{\"a\":1}".to_string()))
        );

        let tool = out[1].as_object().expect("tool message");
        assert_eq!(tool.get("role"), Some(&Value::String("tool".to_string())));
        assert_eq!(
            tool.get("tool_call_id"),
            Some(&Value::String("call-1".to_string()))
        );
        assert_eq!(
            tool.get("content"),
            Some(&Value::String("{\"ok\":true}".to_string()))
        );
    }

    #[test]
    fn responses_replays_tool_calls() {
        let model = OpenAIResponsesModel::new(
            "gpt-5-mini",
            "test-key".to_string(),
            Url::parse("https://example.com/").expect("valid url"),
            None,
        );

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

        let out = model
            .make_input_messages(&messages)
            .expect("make input messages");
        assert_eq!(out.len(), 2);

        let call = out[0].as_object().expect("function call item");
        assert_eq!(
            call.get("type"),
            Some(&Value::String("function_call".to_string()))
        );
        assert_eq!(
            call.get("call_id"),
            Some(&Value::String("call-1".to_string()))
        );
        assert_eq!(
            call.get("name"),
            Some(&Value::String("get_data".to_string()))
        );
        assert_eq!(
            call.get("arguments"),
            Some(&Value::String("{\"a\":1}".to_string()))
        );

        let output = out[1].as_object().expect("function call output");
        assert_eq!(
            output.get("type"),
            Some(&Value::String("function_call_output".to_string()))
        );
        assert_eq!(
            output.get("call_id"),
            Some(&Value::String("call-1".to_string()))
        );
        assert_eq!(
            output.get("output"),
            Some(&Value::String("{\"ok\":true}".to_string()))
        );
    }

    #[test]
    fn responses_replays_provider_items() {
        let model = OpenAIResponsesModel::new(
            "gpt-5-mini",
            "test-key".to_string(),
            Url::parse("https://example.com/").expect("valid url"),
            None,
        );

        let raw_item = json!({
            "type": "reasoning",
            "summary": "ok"
        });

        let messages = vec![ModelMessage::Response(ModelResponse {
            parts: vec![
                ModelResponsePart::ProviderItem(ProviderItemPart {
                    provider: "openai_responses".to_string(),
                    payload: raw_item.clone(),
                }),
                ModelResponsePart::Text(TextPart {
                    content: "ignored".to_string(),
                }),
            ],
            usage: None,
            model_name: None,
            finish_reason: None,
        })];

        let out = model
            .make_input_messages(&messages)
            .expect("make input messages");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], raw_item);
    }

    #[test]
    fn unified_model_streaming_prefers_chat_when_available() {
        let model = OpenAIUnifiedModel::new(
            "gpt-4o-mini",
            "test-key".to_string(),
            Url::parse("https://example.com/").expect("valid url"),
            None,
        );

        let mode = model.select_api(&[], true).expect("select api for stream");
        assert!(matches!(mode, OpenAIApiMode::Chat));
    }

    #[test]
    fn unified_model_streaming_errors_for_responses_only() {
        let model = OpenAIUnifiedModel::new(
            "gpt-5-mini",
            "test-key".to_string(),
            Url::parse("https://example.com/").expect("valid url"),
            None,
        );

        let err = model.select_api(&[], true).expect_err("streaming error");
        assert!(matches!(err, ModelError::Unsupported(_)));
    }
}

impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn model(&self, model: &str, settings: Option<ModelSettings>) -> Arc<dyn Model> {
        Arc::new(OpenAIUnifiedModel::new(
            model,
            self.api_key.clone(),
            self.base_url.clone(),
            settings,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct OpenAIChatModel {
    model: String,
    api_key: String,
    base_url: Url,
    client: Client,
    default_settings: Option<ModelSettings>,
    capabilities: OpenAIChatCapabilities,
}

impl OpenAIChatModel {
    pub fn new(
        model: impl Into<String>,
        api_key: String,
        base_url: Url,
        settings: Option<ModelSettings>,
    ) -> Self {
        Self::new_with_capabilities(
            model,
            api_key,
            base_url,
            settings,
            OpenAIChatCapabilities::default(),
        )
    }

    pub(crate) fn new_with_capabilities(
        model: impl Into<String>,
        api_key: String,
        base_url: Url,
        settings: Option<ModelSettings>,
        capabilities: OpenAIChatCapabilities,
    ) -> Self {
        Self {
            model: model.into(),
            api_key,
            base_url,
            client: Client::new(),
            default_settings: settings,
            capabilities,
        }
    }

    fn endpoint(&self) -> Result<Url, ModelError> {
        join_path(&self.base_url, "chat/completions")
    }

    fn make_messages(&self, messages: &[ModelMessage]) -> Result<Vec<Value>, ModelError> {
        let mut out = Vec::new();
        for message in messages {
            match message {
                ModelMessage::Request(req) => {
                    if let Some(instructions) = req
                        .instructions
                        .as_ref()
                        .filter(|value| !value.trim().is_empty())
                    {
                        out.push(json!({"role": "system", "content": instructions}));
                    }
                    for part in &req.parts {
                        match part {
                            ModelRequestPart::SystemPrompt(prompt) => {
                                out.push(json!({"role": "system", "content": prompt.content}))
                            }
                            ModelRequestPart::UserPrompt(prompt) => {
                                let content = self.convert_user_content(&prompt.content)?;
                                out.push(json!({"role": "user", "content": content}))
                            }
                            ModelRequestPart::ToolReturn(tool_return) => {
                                let content = tool_return_content(&tool_return.content);
                                out.push(json!({
                                    "role": "tool",
                                    "tool_call_id": normalize_tool_call_id_str(&tool_return.tool_call_id),
                                    "content": content,
                                }))
                            }
                            ModelRequestPart::RetryPrompt(retry) => {
                                if retry.tool_name.is_some() {
                                    out.push(json!({
                                        "role": "tool",
                                        "tool_call_id": normalize_tool_call_id(retry.tool_call_id.clone()),
                                        "content": retry.content,
                                    }));
                                } else {
                                    out.push(json!({
                                        "role": "user",
                                        "content": retry.content,
                                    }));
                                }
                            }
                        }
                    }
                }
                ModelMessage::Response(res) => {
                    let text = res.text();
                    let tool_calls = res.tool_calls();

                    if text.is_none() && tool_calls.is_empty() {
                        continue;
                    }

                    let mut msg = Map::new();
                    msg.insert("role".to_string(), Value::String("assistant".to_string()));

                    if let Some(text) = text {
                        msg.insert("content".to_string(), Value::String(text));
                    } else if !tool_calls.is_empty() {
                        msg.insert("content".to_string(), Value::Null);
                    }

                    if !tool_calls.is_empty() {
                        let calls = tool_calls
                            .into_iter()
                            .map(|call| {
                                let args = tool_call_arguments(&call.arguments);
                                json!({
                                    "id": normalize_tool_call_id_str(&call.id),
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": args,
                                    }
                                })
                            })
                            .collect::<Vec<_>>();
                        msg.insert("tool_calls".to_string(), Value::Array(calls));
                    }

                    out.push(Value::Object(msg));
                }
            }
        }
        Ok(out)
    }

    fn convert_user_content(&self, content: &[UserContent]) -> Result<Value, ModelError> {
        let mut parts = Vec::new();
        for item in content {
            match item {
                UserContent::Text(text) => parts.push(json!({"type": "text", "text": text})),
                UserContent::Image(image) => parts.push(json!({
                    "type": "image_url",
                    "image_url": {"url": image.url}
                })),
                UserContent::Binary(BinaryContent { data, media_type }) => {
                    if media_type.starts_with("image/") {
                        if self.capabilities.reject_binary_images {
                            return Err(ModelError::Unsupported(
                                "binary image inputs are not supported; provide an image URL"
                                    .to_string(),
                            ));
                        }
                        let encoded = general_purpose::STANDARD.encode(data);
                        let data_url = format!("data:{};base64,{}", media_type, encoded);
                        parts.push(json!({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        }))
                    } else if media_type.starts_with("audio/") {
                        if let Some(format) = audio_format_from_media_type(media_type) {
                            let encoded = general_purpose::STANDARD.encode(data);
                            parts.push(json!({
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded,
                                    "format": format
                                }
                            }))
                        } else {
                            parts.push(json!({
                                "type": "text",
                                "text": format!("[audio content: {} bytes]", data.len())
                            }))
                        }
                    } else if is_text_like_media_type(media_type) {
                        match std::str::from_utf8(data) {
                            Ok(text) => parts.push(json!({"type": "text", "text": text})),
                            Err(_) => parts.push(json!({
                                "type": "text",
                                "text": format!("[binary content: {} bytes]", data.len())
                            })),
                        }
                    } else {
                        parts.push(json!({
                            "type": "text",
                            "text": format!("[binary content: {} bytes]", data.len())
                        }))
                    }
                }
                UserContent::Audio(audio) => {
                    if let Some((media_type, data)) = parse_data_url_base64(&audio.url)
                        && let Some(format) = audio_format_from_media_type(&media_type)
                    {
                        parts.push(json!({
                            "type": "input_audio",
                            "input_audio": {
                                "data": data,
                                "format": format
                            }
                        }))
                    } else {
                        parts.push(json!({
                            "type": "text",
                            "text": format!("[audio: {}]", audio.url)
                        }))
                    }
                }
                UserContent::Video(video) => parts.push(json!({
                    "type": "text",
                    "text": format!("[video: {}]", video.url)
                })),
                UserContent::Document(doc) => {
                    if let Some((media_type, data)) = parse_data_url_base64(&doc.url)
                        && is_text_like_media_type(&media_type)
                    {
                        match general_purpose::STANDARD.decode(data.as_bytes()) {
                            Ok(bytes) => match String::from_utf8(bytes) {
                                Ok(text) => parts.push(json!({"type": "text", "text": text})),
                                Err(_) => parts.push(json!({
                                    "type": "text",
                                    "text": format!("[document: {}]", doc.url)
                                })),
                            },
                            Err(_) => parts.push(json!({
                                "type": "text",
                                "text": format!("[document: {}]", doc.url)
                            })),
                        }
                    } else {
                        parts.push(json!({
                            "type": "text",
                            "text": format!("[document: {}]", doc.url)
                        }))
                    }
                }
            }
        }

        Ok(Value::Array(parts))
    }

    fn build_body(
        &self,
        messages: &[ModelMessage],
        params: &ModelRequestParameters,
        stream: bool,
    ) -> Result<Value, ModelError> {
        let mut body = Map::new();
        body.insert("model".to_string(), Value::String(self.model.clone()));
        body.insert(
            "messages".to_string(),
            Value::Array(self.make_messages(messages)?),
        );

        if !params.function_tools.is_empty() {
            let tools = params
                .function_tools
                .iter()
                .map(|tool| {
                    let (schema, _strict_ok) =
                        transform_openai_schema(&tool.parameters_json_schema, None);
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": schema,
                        }
                    })
                })
                .collect();
            body.insert("tools".to_string(), Value::Array(tools));
            body.insert("tool_choice".to_string(), Value::String("auto".to_string()));
            if self.capabilities.supports_parallel_tool_calls
                && params.function_tools.iter().any(|tool| tool.sequential)
            {
                body.insert("parallel_tool_calls".to_string(), Value::Bool(false));
            }
        }

        if params.output_mode == OutputMode::JsonSchema
            && let Some(schema) = params.output_schema.clone()
            && self.capabilities.supports_response_format
        {
            let strict = !params.allow_text_output;
            let (schema, _strict_ok) = transform_openai_schema(&schema, Some(strict));
            body.insert(
                "response_format".to_string(),
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "output",
                        "schema": schema,
                        "strict": strict,
                    }
                }),
            );
        }

        if stream {
            body.insert("stream".to_string(), Value::Bool(true));
            body.insert("stream_options".to_string(), json!({"include_usage": true}));
        }

        if let Some(settings) = &self.default_settings {
            for (key, value) in settings {
                body.entry(key.clone()).or_insert(value.clone());
            }
        }

        Ok(Value::Object(body))
    }

    fn parse_tool_call(tool_call: &OpenAIToolCall) -> ToolCallPart {
        let args = tool_call
            .function
            .arguments
            .as_ref()
            .and_then(|arg| serde_json::from_str::<Value>(arg).ok())
            .unwrap_or_else(|| {
                tool_call
                    .function
                    .arguments
                    .clone()
                    .map(Value::String)
                    .unwrap_or_else(|| Value::Object(Map::new()))
            });

        ToolCallPart {
            id: normalize_tool_call_id(tool_call.id.clone()),
            name: tool_call
                .function
                .name
                .clone()
                .unwrap_or_else(|| "tool".to_string()),
            arguments: args,
        }
    }
}

#[async_trait]
impl Model for OpenAIChatModel {
    fn name(&self) -> &str {
        &self.model
    }

    async fn request(
        &self,
        messages: &[ModelMessage],
        settings: Option<&ModelSettings>,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        let mut body = self.build_body(messages, params, false)?;
        if let Some(settings) = settings
            && let Value::Object(map) = &mut body
        {
            for (key, value) in settings {
                map.insert(key.clone(), value.clone());
            }
        }

        let response = self
            .client
            .post(self.endpoint()?)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| map_reqwest_error("OpenAI", e))?;

        let status = response.status();
        if !status.is_success() {
            return Err(ModelError::HttpStatus {
                status: status.as_u16(),
            });
        }

        let body: OpenAIChatResponse = response
            .json()
            .await
            .map_err(|e| ModelError::Provider(format!("OpenAI response parse failed: {e}")))?;

        let choice =
            body.choices.into_iter().next().ok_or_else(|| {
                ModelError::Provider("OpenAI response missing choices".to_string())
            })?;

        let mut parts = Vec::new();
        if let Some(content) = choice.message.content {
            parts.push(ModelResponsePart::Text(TextPart { content }));
        }

        if let Some(tool_calls) = choice.message.tool_calls {
            for call in tool_calls {
                parts.push(ModelResponsePart::ToolCall(Self::parse_tool_call(&call)));
            }
        } else if let Some(function_call) = choice.message.function_call {
            parts.push(ModelResponsePart::ToolCall(ToolCallPart {
                id: normalize_tool_call_id(None),
                name: function_call.name.unwrap_or_else(|| "tool".to_string()),
                arguments: function_call
                    .arguments
                    .as_ref()
                    .and_then(|arg| serde_json::from_str::<Value>(arg).ok())
                    .unwrap_or_else(|| {
                        function_call
                            .arguments
                            .clone()
                            .map(Value::String)
                            .unwrap_or_else(|| Value::Object(Map::new()))
                    }),
            }));
        }

        let usage = body.usage.map(|usage| RequestUsage {
            input_tokens: usage.prompt_tokens.unwrap_or(0),
            output_tokens: usage.completion_tokens.unwrap_or(0),
            ..Default::default()
        });

        Ok(ModelResponse {
            parts,
            usage,
            model_name: Some(self.model.clone()),
            finish_reason: choice.finish_reason,
        })
    }

    async fn request_stream(
        &self,
        messages: &[ModelMessage],
        settings: Option<&ModelSettings>,
        params: &ModelRequestParameters,
    ) -> Result<ModelStream, ModelError> {
        let mut body = self.build_body(messages, params, true)?;
        if let Some(settings) = settings
            && let Value::Object(map) = &mut body
        {
            for (key, value) in settings {
                map.insert(key.clone(), value.clone());
            }
        }

        let response = self
            .client
            .post(self.endpoint()?)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| map_reqwest_error("OpenAI stream", e))?;

        if !response.status().is_success() {
            return Err(ModelError::HttpStatus {
                status: response.status().as_u16(),
            });
        }

        let mut event_stream = response.bytes_stream().eventsource();

        let s = try_stream! {
            let mut tool_accumulator: HashMap<String, ToolAccumulator> = HashMap::new();
            while let Some(event) = event_stream.next().await {
                let event = event.map_err(|e| ModelError::Provider(format!("OpenAI stream error: {e}")))?;
                let data = event.data;
                if data.trim() == "[DONE]" {
                    if !tool_accumulator.is_empty() {
                        for (_id, acc) in tool_accumulator.drain() {
                            let args = serde_json::from_str::<Value>(&acc.arguments)
                                .unwrap_or_else(|_| Value::String(acc.arguments.clone()));
                            yield StreamChunk {
                                text_delta: None,
                                tool_call: Some(ToolCallPart {
                                    id: acc.id.clone(),
                                    name: acc.name.unwrap_or_else(|| "tool".to_string()),
                                    arguments: args,
                                }),
                                finish_reason: None,
                                usage: None,
                            };
                        }
                    }
                    break;
                }

                let chunk: OpenAIChatStreamResponse = serde_json::from_str(&data)
                    .map_err(|e| ModelError::Provider(format!("OpenAI stream parse error: {e}")))?;
                if let Some(choice) = chunk.choices.into_iter().next() {
                    if let Some(content) = choice.delta.content {
                        yield StreamChunk {
                            text_delta: Some(content),
                            tool_call: None,
                            finish_reason: None,
                            usage: None,
                        };
                    }

                    if let Some(tool_calls) = choice.delta.tool_calls {
                        for call in tool_calls {
                            let id = normalize_stream_tool_call_id(call.id.clone(), call.index);
                            let entry = tool_accumulator.entry(id.clone()).or_insert_with(|| ToolAccumulator {
                                id,
                                name: None,
                                arguments: String::new(),
                            });
                            if let Some(name) = call.function.name {
                                entry.name = Some(name);
                            }
                            if let Some(args) = call.function.arguments {
                                entry.arguments.push_str(&args);
                            }
                        }
                    }

                    if let Some(reason) = choice.finish_reason.clone() {
                        if !tool_accumulator.is_empty() {
                            for (_id, acc) in tool_accumulator.drain() {
                                let args = serde_json::from_str::<Value>(&acc.arguments)
                                    .unwrap_or_else(|_| Value::String(acc.arguments.clone()));
                                yield StreamChunk {
                                    text_delta: None,
                                    tool_call: Some(ToolCallPart {
                                        id: acc.id.clone(),
                                        name: acc.name.unwrap_or_else(|| "tool".to_string()),
                                        arguments: args,
                                    }),
                                    finish_reason: Some(reason.clone()),
                                    usage: None,
                                };
                            }
                        }
                        yield StreamChunk {
                            text_delta: None,
                            tool_call: None,
                            finish_reason: Some(reason),
                            usage: chunk.usage.map(|usage| RequestUsage {
                                input_tokens: usage.prompt_tokens.unwrap_or(0),
                                output_tokens: usage.completion_tokens.unwrap_or(0),
                                ..Default::default()
                            }),
                        };
                    }
                }
            }
        };

        Ok(Box::pin(s))
    }
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
    function_call: Option<OpenAIFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct OpenAIToolCall {
    id: Option<String>,
    function: OpenAIToolFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAIToolFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIFunctionCall {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatStreamResponse {
    choices: Vec<OpenAIChatStreamChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatStreamChoice {
    delta: OpenAIChatStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamToolCall {
    id: Option<String>,
    index: Option<usize>,
    function: OpenAIStreamToolFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamToolFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug)]
struct ToolAccumulator {
    id: String,
    name: Option<String>,
    arguments: String,
}

#[derive(Clone, Debug)]
pub struct OpenAIUnifiedModel {
    model: String,
    chat: OpenAIChatModel,
    responses: OpenAIResponsesModel,
    responses_only: bool,
    prefer_responses: bool,
}

impl OpenAIUnifiedModel {
    pub fn new(
        model: impl Into<String>,
        api_key: String,
        base_url: Url,
        settings: Option<ModelSettings>,
    ) -> Self {
        let model = model.into();
        let responses_only = is_responses_only_model(&model);
        let prefer_responses = prefers_responses(&model);
        Self {
            chat: OpenAIChatModel::new(
                model.clone(),
                api_key.clone(),
                base_url.clone(),
                settings.clone(),
            ),
            responses: OpenAIResponsesModel::new(model.clone(), api_key, base_url, settings),
            model,
            responses_only,
            prefer_responses,
        }
    }

    fn select_api(
        &self,
        messages: &[ModelMessage],
        stream: bool,
    ) -> Result<OpenAIApiMode, ModelError> {
        if contains_audio(messages) {
            if self.responses_only {
                return Err(ModelError::Unsupported(
                    "OpenAI Responses API does not support audio input".to_string(),
                ));
            }
            return Ok(OpenAIApiMode::Chat);
        }
        if stream {
            if self.responses_only {
                return Err(ModelError::Unsupported(
                    "streaming not supported for OpenAI Responses API".to_string(),
                ));
            }
            return Ok(OpenAIApiMode::Chat);
        }
        if self.prefer_responses || self.responses_only {
            Ok(OpenAIApiMode::Responses)
        } else {
            Ok(OpenAIApiMode::Chat)
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum OpenAIApiMode {
    Chat,
    Responses,
}

#[async_trait]
impl Model for OpenAIUnifiedModel {
    fn name(&self) -> &str {
        &self.model
    }

    async fn request(
        &self,
        messages: &[ModelMessage],
        settings: Option<&ModelSettings>,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        match self.select_api(messages, false)? {
            OpenAIApiMode::Chat => self.chat.request(messages, settings, params).await,
            OpenAIApiMode::Responses => self.responses.request(messages, settings, params).await,
        }
    }

    async fn request_stream(
        &self,
        messages: &[ModelMessage],
        settings: Option<&ModelSettings>,
        params: &ModelRequestParameters,
    ) -> Result<ModelStream, ModelError> {
        match self.select_api(messages, true)? {
            OpenAIApiMode::Chat => self.chat.request_stream(messages, settings, params).await,
            OpenAIApiMode::Responses => Err(ModelError::Unsupported(
                "streaming not supported for OpenAI Responses API".to_string(),
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OpenAIResponsesModel {
    model: String,
    api_key: String,
    base_url: Url,
    client: Client,
    default_settings: Option<ModelSettings>,
}

impl OpenAIResponsesModel {
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
        join_path(&self.base_url, "responses")
    }

    fn filename_for_media_type(media_type: &str) -> String {
        let ext = match media_type {
            "application/pdf" => "pdf",
            "text/plain" => "txt",
            "text/markdown" => "md",
            "application/json" => "json",
            _ => "bin",
        };
        format!("file.{ext}")
    }

    fn make_input_messages(&self, messages: &[ModelMessage]) -> Result<Vec<Value>, ModelError> {
        let mut out = Vec::new();
        for message in messages {
            match message {
                ModelMessage::Request(req) => {
                    if let Some(instructions) = req
                        .instructions
                        .as_ref()
                        .filter(|value| !value.trim().is_empty())
                    {
                        out.push(json!({"role": "system", "content": instructions}));
                    }
                    for part in &req.parts {
                        match part {
                            ModelRequestPart::SystemPrompt(prompt) => {
                                out.push(json!({"role": "system", "content": prompt.content}))
                            }
                            ModelRequestPart::UserPrompt(prompt) => {
                                let content = self.convert_user_content(&prompt.content)?;
                                out.push(json!({"role": "user", "content": content}))
                            }
                            ModelRequestPart::ToolReturn(tool_return) => {
                                let content = tool_return_content(&tool_return.content);
                                out.push(json!({
                                    "type": "function_call_output",
                                    "call_id": normalize_tool_call_id_str(&tool_return.tool_call_id),
                                    "output": content,
                                }))
                            }
                            ModelRequestPart::RetryPrompt(retry) => {
                                if retry.tool_name.is_some() {
                                    out.push(json!({
                                        "type": "function_call_output",
                                        "call_id": normalize_tool_call_id(retry.tool_call_id.clone()),
                                        "output": retry.content,
                                    }));
                                } else {
                                    out.push(json!({
                                        "role": "user",
                                        "content": [ { "type": "input_text", "text": retry.content } ],
                                    }));
                                }
                            }
                        }
                    }
                }
                ModelMessage::Response(res) => {
                    let provider_items: Vec<Value> = res
                        .parts
                        .iter()
                        .filter_map(|part| match part {
                            ModelResponsePart::ProviderItem(item)
                                if item.provider == "openai_responses" =>
                            {
                                Some(item.payload.clone())
                            }
                            _ => None,
                        })
                        .collect();
                    if !provider_items.is_empty() {
                        out.extend(provider_items);
                        continue;
                    }
                    if let Some(text) = res.text() {
                        out.push(json!({"role": "assistant", "content": text}));
                    }
                    for call in res.tool_calls() {
                        let args = tool_call_arguments(&call.arguments);
                        out.push(json!({
                            "type": "function_call",
                            "call_id": normalize_tool_call_id_str(&call.id),
                            "name": call.name,
                            "arguments": args,
                        }));
                    }
                }
            }
        }
        Ok(out)
    }

    fn convert_user_content(&self, content: &[UserContent]) -> Result<Value, ModelError> {
        let mut parts = Vec::new();
        for item in content {
            match item {
                UserContent::Text(text) => parts.push(json!({"type": "input_text", "text": text})),
                UserContent::Image(image) => parts.push(json!({
                    "type": "input_image",
                    "image_url": image.url
                })),
                UserContent::Binary(BinaryContent { data, media_type }) => {
                    if media_type.starts_with("image/") {
                        let encoded = general_purpose::STANDARD.encode(data);
                        let data_url = format!("data:{};base64,{}", media_type, encoded);
                        parts.push(json!({
                            "type": "input_image",
                            "image_url": data_url
                        }));
                    } else if media_type == "application/pdf" {
                        let encoded = general_purpose::STANDARD.encode(data);
                        let data_url = format!("data:{};base64,{}", media_type, encoded);
                        parts.push(json!({
                            "type": "input_file",
                            "file_data": data_url,
                            "filename": Self::filename_for_media_type(media_type),
                        }));
                    } else if is_text_like_media_type(media_type) {
                        match std::str::from_utf8(data) {
                            Ok(text) => parts.push(json!({"type": "input_text", "text": text})),
                            Err(_) => parts.push(json!({
                                "type": "input_text",
                                "text": format!("[binary content: {} bytes]", data.len())
                            })),
                        }
                    } else {
                        parts.push(json!({
                            "type": "input_text",
                            "text": format!("[binary content: {} bytes]", data.len())
                        }))
                    }
                }
                UserContent::Document(doc) => {
                    if let Some((media_type, data)) = parse_data_url_base64(&doc.url) {
                        let data_url = format!("data:{};base64,{}", media_type, data);
                        parts.push(json!({
                            "type": "input_file",
                            "file_data": data_url,
                            "filename": Self::filename_for_media_type(&media_type),
                        }));
                    } else {
                        parts.push(json!({
                            "type": "input_file",
                            "file_url": doc.url
                        }));
                    }
                }
                UserContent::Audio(audio) => parts.push(json!({
                    "type": "input_text",
                    "text": format!("[audio: {}]", audio.url)
                })),
                UserContent::Video(video) => parts.push(json!({
                    "type": "input_text",
                    "text": format!("[video: {}]", video.url)
                })),
            }
        }
        Ok(Value::Array(parts))
    }

    fn build_body(
        &self,
        messages: &[ModelMessage],
        params: &ModelRequestParameters,
    ) -> Result<Value, ModelError> {
        let mut body = Map::new();
        body.insert("model".to_string(), Value::String(self.model.clone()));
        body.insert(
            "input".to_string(),
            Value::Array(self.make_input_messages(messages)?),
        );

        if !params.function_tools.is_empty() {
            let tools = params
                .function_tools
                .iter()
                .map(|tool| {
                    let (schema, _strict_ok) =
                        transform_openai_schema(&tool.parameters_json_schema, None);
                    json!({
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    })
                })
                .collect();
            body.insert("tools".to_string(), Value::Array(tools));
            if params.function_tools.iter().any(|tool| tool.sequential) {
                body.insert("parallel_tool_calls".to_string(), Value::Bool(false));
            }
        }

        if params.output_mode == OutputMode::JsonSchema
            && let Some(schema) = params.output_schema.clone()
        {
            let strict = !params.allow_text_output;
            let (schema, _strict_ok) = transform_openai_schema(&schema, Some(strict));
            body.insert(
                "text".to_string(),
                json!({
                    "format": {
                        "type": "json_schema",
                        "name": "output",
                        "schema": schema,
                        "strict": strict,
                    }
                }),
            );
        }

        if let Some(settings) = &self.default_settings {
            for (key, value) in settings {
                if key == "max_tokens" && !body.contains_key("max_output_tokens") {
                    body.insert("max_output_tokens".to_string(), value.clone());
                } else {
                    body.insert(key.clone(), value.clone());
                }
            }
        }

        Ok(Value::Object(body))
    }
}

#[async_trait]
impl Model for OpenAIResponsesModel {
    fn name(&self) -> &str {
        &self.model
    }

    async fn request(
        &self,
        messages: &[ModelMessage],
        settings: Option<&ModelSettings>,
        params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        let mut body = self.build_body(messages, params)?;
        if let Some(settings) = settings
            && let Value::Object(map) = &mut body
        {
            for (key, value) in settings {
                if key == "max_tokens" && !map.contains_key("max_output_tokens") {
                    map.insert("max_output_tokens".to_string(), value.clone());
                } else {
                    map.insert(key.clone(), value.clone());
                }
            }
        }

        let response = self
            .client
            .post(self.endpoint()?)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| map_reqwest_error("OpenAI Responses", e))?;

        let status = response.status();
        if !status.is_success() {
            return Err(ModelError::HttpStatus {
                status: status.as_u16(),
            });
        }

        let body: OpenAIResponsesResponse = response
            .json()
            .await
            .map_err(|e| ModelError::Provider(format!("OpenAI response parse failed: {e}")))?;

        let mut parts = Vec::new();
        for item in body.output {
            parts.push(ModelResponsePart::ProviderItem(ProviderItemPart {
                provider: "openai_responses".to_string(),
                payload: item.clone(),
            }));

            if let Some(item_type) = item.get("type").and_then(|value| value.as_str()) {
                match item_type {
                    "message" => {
                        if let Some(content) =
                            item.get("content").and_then(|value| value.as_array())
                        {
                            for part in content {
                                if part.get("type").and_then(|value| value.as_str())
                                    == Some("output_text")
                                    && let Some(text) =
                                        part.get("text").and_then(|value| value.as_str())
                                {
                                    parts.push(ModelResponsePart::Text(TextPart {
                                        content: text.to_string(),
                                    }));
                                }
                            }
                        }
                    }
                    "function_call" => {
                        let name = item
                            .get("name")
                            .and_then(|value| value.as_str())
                            .unwrap_or("tool")
                            .to_string();
                        let call_id = item
                            .get("call_id")
                            .and_then(|value| value.as_str())
                            .map(str::to_string);
                        let arguments = item.get("arguments").cloned().unwrap_or(Value::Null);
                        let args = match arguments {
                            Value::String(value) => serde_json::from_str::<Value>(&value)
                                .unwrap_or(Value::String(value)),
                            other => other,
                        };
                        parts.push(ModelResponsePart::ToolCall(ToolCallPart {
                            id: normalize_tool_call_id(call_id),
                            name,
                            arguments: args,
                        }));
                    }
                    _ => {}
                }
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
            model_name: body.model.or_else(|| Some(self.model.clone())),
            finish_reason: body.finish_reason,
        })
    }
}

#[derive(Debug, Deserialize)]
struct OpenAIResponsesResponse {
    output: Vec<Value>,
    usage: Option<OpenAIResponsesUsage>,
    model: Option<String>,
    #[serde(rename = "finish_reason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponsesUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}
