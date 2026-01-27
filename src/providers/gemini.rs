use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
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

fn gemini_response_object(value: &Value) -> Value {
    match value {
        Value::Object(_) => value.clone(),
        _ => {
            let mut wrapped = Map::new();
            wrapped.insert("return_value".to_string(), value.clone());
            Value::Object(wrapped)
        }
    }
}

fn is_null_schema(value: &Value) -> bool {
    matches!(
        value,
        Value::Object(map) if matches!(map.get("type"), Some(Value::String(t)) if t == "null")
    )
}

fn sanitize_gemini_schema(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            if let Some(variants) = map.get("anyOf").and_then(|val| val.as_array()) {
                let mut cleaned = variants
                    .iter()
                    .filter(|variant| !is_null_schema(variant))
                    .map(sanitize_gemini_schema)
                    .collect::<Vec<_>>();
                if cleaned.len() == 1 {
                    return cleaned.pop().unwrap_or(Value::Null);
                }
            }
            if let Some(variants) = map.get("oneOf").and_then(|val| val.as_array()) {
                let mut cleaned = variants
                    .iter()
                    .filter(|variant| !is_null_schema(variant))
                    .map(sanitize_gemini_schema)
                    .collect::<Vec<_>>();
                if cleaned.len() == 1 {
                    return cleaned.pop().unwrap_or(Value::Null);
                }
            }

            let mut out = Map::new();
            for (key, val) in map {
                if matches!(
                    key.as_str(),
                    "additionalProperties" | "$schema" | "$id" | "title"
                ) {
                    continue;
                }
                if key == "type"
                    && let Value::Array(types) = val
                {
                    if let Some(first) = types
                        .iter()
                        .find(|item| !matches!(item, Value::String(t) if t == "null"))
                    {
                        out.insert(key.clone(), first.clone());
                    }
                    continue;
                }
                out.insert(key.clone(), sanitize_gemini_schema(val));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(sanitize_gemini_schema).collect()),
        _ => value.clone(),
    }
}

fn infer_media_type_from_url(url: &str) -> Option<String> {
    let path = url.split('?').next()?;
    let ext = path.rsplit('.').next()?.to_lowercase();
    let media_type = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "md" | "markdown" => "text/markdown",
        "csv" => "text/csv",
        "json" => "application/json",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" | "oga" => "audio/ogg",
        "flac" => "audio/flac",
        "m4a" | "aac" => "audio/aac",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        "mkv" => "video/x-matroska",
        _ => return None,
    };
    Some(media_type.to_string())
}

fn file_data_part(url: &str, media_type: &Option<String>) -> Value {
    let mut file_data = Map::new();
    file_data.insert("fileUri".to_string(), Value::String(url.to_string()));
    let inferred = media_type
        .clone()
        .or_else(|| infer_media_type_from_url(url));
    if let Some(media_type) = inferred {
        file_data.insert("mimeType".to_string(), Value::String(media_type.clone()));
    }
    let mut wrapper = Map::new();
    wrapper.insert("fileData".to_string(), Value::Object(file_data));
    Value::Object(wrapper)
}

#[derive(Clone, Debug)]
pub struct GeminiProvider {
    api_key: String,
    base_url: Url,
}

impl GeminiProvider {
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
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| ProviderError::MissingApiKey("gemini".to_string()))?;
        Self::new(api_key, "https://generativelanguage.googleapis.com")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{
        BinaryContent, ImageUrl, ModelMessage, ModelRequest, ModelRequestPart, ModelResponse,
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
    fn convert_user_content_handles_inline_and_file_data() {
        let pdf_bytes = fixture_bytes("fixture.pdf");
        let audio_bytes = fixture_bytes("fixture.m4a");

        let content = vec![
            UserContent::Binary(BinaryContent {
                data: pdf_bytes.clone(),
                media_type: "application/pdf".to_string(),
            }),
            UserContent::Binary(BinaryContent {
                data: audio_bytes.clone(),
                media_type: "audio/aac".to_string(),
            }),
            UserContent::Image(ImageUrl {
                url: "https://example.com/fixture.jpg".to_string(),
                media_type: None,
            }),
        ];

        let parts = convert_user_content(&content);
        assert_eq!(parts.len(), 3);

        let pdf = &parts[0];
        let pdf_inline = pdf.get("inlineData").expect("pdf inline");
        assert_eq!(
            pdf_inline.get("mimeType"),
            Some(&Value::String("application/pdf".to_string()))
        );
        assert_eq!(
            pdf_inline.get("data"),
            Some(&Value::String(STANDARD.encode(&pdf_bytes)))
        );

        let audio = &parts[1];
        let audio_inline = audio.get("inlineData").expect("audio inline");
        assert_eq!(
            audio_inline.get("mimeType"),
            Some(&Value::String("audio/aac".to_string()))
        );
        assert_eq!(
            audio_inline.get("data"),
            Some(&Value::String(STANDARD.encode(&audio_bytes)))
        );

        let image = &parts[2];
        let file_data = image.get("fileData").expect("file data");
        assert_eq!(
            file_data.get("fileUri"),
            Some(&Value::String(
                "https://example.com/fixture.jpg".to_string()
            ))
        );
        assert_eq!(
            file_data.get("mimeType"),
            Some(&Value::String("image/jpeg".to_string()))
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

        let (_system, contents) = GeminiModel::split_system(&messages);
        assert_eq!(contents.len(), 2);

        let model_msg = contents[0].as_object().expect("model message");
        assert_eq!(
            model_msg.get("role"),
            Some(&Value::String("model".to_string()))
        );
        let model_parts = model_msg
            .get("parts")
            .and_then(|value| value.as_array())
            .expect("model parts");
        let function_call = model_parts
            .iter()
            .find_map(|part| part.get("functionCall"))
            .expect("functionCall");
        assert_eq!(
            function_call.get("name"),
            Some(&Value::String("get_data".to_string()))
        );
        assert_eq!(function_call.get("args"), Some(&json!({"a": 1})));

        let user_msg = contents[1].as_object().expect("user message");
        assert_eq!(
            user_msg.get("role"),
            Some(&Value::String("user".to_string()))
        );
        let user_parts = user_msg
            .get("parts")
            .and_then(|value| value.as_array())
            .expect("user parts");
        let function_response = user_parts
            .iter()
            .find_map(|part| part.get("functionResponse"))
            .expect("functionResponse");
        assert_eq!(
            function_response.get("name"),
            Some(&Value::String("get_data".to_string()))
        );
        assert_eq!(
            function_response.get("response"),
            Some(&json!({"ok": true}))
        );
    }
}

impl Provider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    fn model(&self, model: &str, settings: Option<ModelSettings>) -> Arc<dyn Model> {
        Arc::new(GeminiModel::new(
            model,
            self.api_key.clone(),
            self.base_url.clone(),
            settings,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct GeminiModel {
    model: String,
    api_key: String,
    base_url: Url,
    client: Client,
    default_settings: Option<ModelSettings>,
}

impl GeminiModel {
    pub fn new(
        model: impl Into<String>,
        api_key: String,
        base_url: Url,
        settings: Option<ModelSettings>,
    ) -> Self {
        let mut model = model.into();
        if !model.starts_with("models/") {
            model = format!("models/{model}");
        }
        Self {
            model,
            api_key,
            base_url,
            client: Client::new(),
            default_settings: settings,
        }
    }

    fn endpoint(&self) -> Result<Url, ModelError> {
        let path = format!("v1beta/{}:generateContent", self.model);
        let mut url = self
            .base_url
            .join(&path)
            .map_err(|e| ModelError::Provider(format!("invalid base url: {e}")))?;
        url.query_pairs_mut().append_pair("key", &self.api_key);
        Ok(url)
    }

    fn split_system(messages: &[ModelMessage]) -> (Option<String>, Vec<Value>) {
        let mut system_parts = Vec::new();
        let mut contents = Vec::new();

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
                            ModelRequestPart::UserPrompt(prompt) => contents.push(json!({
                                "role": "user",
                                "parts": convert_user_content(&prompt.content)
                            })),
                            ModelRequestPart::ToolReturn(tool_return) => contents.push(json!({
                                "role": "user",
                                "parts": [{
                                    "functionResponse": {
                                        "name": tool_return.tool_name,
                                        "response": gemini_response_object(&tool_return.content),
                                    }
                                }]
                            })),
                            ModelRequestPart::RetryPrompt(retry) => {
                                let parts = if let Some(tool_name) = &retry.tool_name {
                                    vec![json!({
                                        "functionResponse": {
                                            "name": tool_name,
                                            "response": {"call_error": retry.content}
                                        }
                                    })]
                                } else {
                                    vec![json!({"text": retry.content})]
                                };
                                contents.push(json!({
                                    "role": "user",
                                    "parts": parts
                                }));
                            }
                        }
                    }
                }
                ModelMessage::Response(res) => {
                    let mut parts = Vec::new();
                    if let Some(text) = res.text() {
                        parts.push(json!({"text": text}));
                    }
                    for call in res.tool_calls() {
                        parts.push(json!({
                            "functionCall": {
                                "name": call.name,
                                "args": call.arguments,
                            }
                        }));
                    }

                    if !parts.is_empty() {
                        contents.push(json!({
                            "role": "model",
                            "parts": parts
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

        (system, contents)
    }
}

fn convert_user_content(content: &[UserContent]) -> Vec<Value> {
    let mut parts = Vec::new();
    for item in content {
        match item {
            UserContent::Text(text) => parts.push(json!({"text": text})),
            UserContent::Image(image) => parts.push(file_data_part(&image.url, &image.media_type)),
            UserContent::Video(video) => parts.push(file_data_part(&video.url, &video.media_type)),
            UserContent::Audio(audio) => parts.push(file_data_part(&audio.url, &audio.media_type)),
            UserContent::Document(doc) => parts.push(file_data_part(&doc.url, &doc.media_type)),
            UserContent::Binary(binary) => parts.push(json!({
                "inlineData": {
                    "mimeType": binary.media_type,
                    "data": base64::engine::general_purpose::STANDARD.encode(&binary.data)
                }
            })),
        }
    }
    parts
}

#[async_trait]
impl Model for GeminiModel {
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
            "Gemini request"
        );
        let (system, contents) = Self::split_system(messages);
        let mut body = Map::new();
        body.insert("contents".to_string(), Value::Array(contents));
        if let Some(system) = system {
            body.insert(
                "systemInstruction".to_string(),
                json!({"parts": [{"text": system}]}),
            );
        }

        if !params.function_tools.is_empty() {
            let tools = params
                .function_tools
                .iter()
                .map(|tool| {
                    let schema = sanitize_gemini_schema(&tool.parameters_json_schema);
                    json!({
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    })
                })
                .collect::<Vec<_>>();
            body.insert(
                "tools".to_string(),
                json!([{ "functionDeclarations": tools }]),
            );
            body.insert(
                "toolConfig".to_string(),
                json!({"functionCallingConfig": {"mode": "AUTO"}}),
            );
        }

        if params.output_mode == OutputMode::JsonSchema
            && let Some(schema) = params.output_schema.clone()
        {
            let schema = sanitize_gemini_schema(&schema);
            body.insert(
                "generationConfig".to_string(),
                json!({
                    "responseMimeType": "application/json",
                    "responseSchema": schema
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

        let response = self
            .client
            .post(self.endpoint()?)
            .json(&Value::Object(body))
            .send()
            .await
            .map_err(|e| map_reqwest_error("Gemini", e))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            tracing::error!(
                status = status.as_u16(),
                model = %self.model,
                body = %truncate_error_body(&body),
                "Gemini request failed"
            );
            return Err(ModelError::HttpStatus {
                status: status.as_u16(),
            });
        }

        let body: GeminiResponse = response.json().await.map_err(|e| {
            tracing::error!(
                error = %e,
                model = %self.model,
                "Gemini response parse failed"
            );
            ModelError::Provider(format!("Gemini response parse failed: {e}"))
        })?;

        let candidate = body.candidates.into_iter().next().ok_or_else(|| {
            tracing::error!(model = %self.model, "Gemini response missing candidates");
            ModelError::Provider("Gemini response missing candidates".to_string())
        })?;

        let mut parts = Vec::new();
        if let Some(content) = candidate.content {
            for part in content.parts {
                if let Some(text) = part.text {
                    parts.push(ModelResponsePart::Text(TextPart { content: text }));
                }
                if let Some(call) = part.function_call {
                    parts.push(ModelResponsePart::ToolCall(ToolCallPart {
                        id: normalize_tool_call_id(call.id),
                        name: call.name.unwrap_or_else(|| "tool".to_string()),
                        arguments: call.args.unwrap_or_else(|| Value::Object(Map::new())),
                    }));
                }
            }
        }

        let usage = body.usage_metadata.map(|usage| RequestUsage {
            input_tokens: usage.prompt_token_count.unwrap_or(0),
            output_tokens: usage.candidates_token_count.unwrap_or(0),
            ..Default::default()
        });

        Ok(ModelResponse {
            parts,
            usage,
            model_name: Some(self.model.clone()),
            finish_reason: candidate.finish_reason,
        })
    }
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiPart {
    text: Option<String>,
    #[serde(rename = "functionCall")]
    function_call: Option<GeminiFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct GeminiFunctionCall {
    id: Option<String>,
    name: Option<String>,
    args: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct GeminiUsage {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u64>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u64>,
}
