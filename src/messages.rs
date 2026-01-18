use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::usage::RequestUsage;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    pub media_type: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoUrl {
    pub url: String,
    pub media_type: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioUrl {
    pub url: String,
    pub media_type: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentUrl {
    pub url: String,
    pub media_type: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BinaryContent {
    pub data: Vec<u8>,
    pub media_type: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UserContent {
    Text(String),
    Image(ImageUrl),
    Video(VideoUrl),
    Audio(AudioUrl),
    Document(DocumentUrl),
    Binary(BinaryContent),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemPromptPart {
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserPromptPart {
    pub content: Vec<UserContent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolReturnPart {
    pub tool_name: String,
    pub tool_call_id: String,
    pub content: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetryPromptPart {
    pub content: String,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelRequestPart {
    SystemPrompt(SystemPromptPart),
    UserPrompt(UserPromptPart),
    ToolReturn(ToolReturnPart),
    RetryPrompt(RetryPromptPart),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRequest {
    pub parts: Vec<ModelRequestPart>,
    pub instructions: Option<String>,
}

impl ModelRequest {
    pub fn user_text_prompt(prompt: impl Into<String>) -> Self {
        Self {
            parts: vec![ModelRequestPart::UserPrompt(UserPromptPart {
                content: vec![UserContent::Text(prompt.into())],
            })],
            instructions: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextPart {
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallPart {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderItemPart {
    pub provider: String,
    pub payload: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelResponsePart {
    Text(TextPart),
    ToolCall(ToolCallPart),
    ProviderItem(ProviderItemPart),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelResponse {
    pub parts: Vec<ModelResponsePart>,
    pub usage: Option<RequestUsage>,
    pub model_name: Option<String>,
    pub finish_reason: Option<String>,
}

impl ModelResponse {
    pub fn text(&self) -> Option<String> {
        let mut texts = Vec::new();
        for part in &self.parts {
            if let ModelResponsePart::Text(text) = part {
                texts.push(text.content.clone());
            }
        }
        if texts.is_empty() {
            None
        } else {
            Some(texts.join("\n\n"))
        }
    }

    pub fn tool_calls(&self) -> Vec<ToolCallPart> {
        self.parts
            .iter()
            .filter_map(|part| match part {
                ModelResponsePart::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelMessage {
    Request(ModelRequest),
    Response(ModelResponse),
}
