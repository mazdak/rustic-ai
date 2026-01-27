//! Grok Realtime WebSocket client and event types.
//!
//! Based on the OpenAI Realtime API specification which Grok is compatible with.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{Message, http::Request},
};
use tracing::{debug, error, info, trace, warn};

use crate::messages::ToolCallPart;

#[derive(Debug, Error)]
pub enum Error {
    #[error("connection closed")]
    ConnectionClosed,
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("websocket error: {0}")]
    WebSocket(String),
    #[error("provider error: {0}")]
    Provider(String),
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

impl From<tokio_tungstenite::tungstenite::Error> for Error {
    fn from(err: tokio_tungstenite::tungstenite::Error) -> Self {
        Self::WebSocket(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Events sent from client to Grok
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientEvent {
    /// Update session configuration
    #[serde(rename = "session.update")]
    SessionUpdate { session: SessionUpdatePayload },

    /// Append audio to input buffer
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        audio: String, // base64 encoded
    },

    /// Commit the audio buffer (create user message)
    #[serde(rename = "conversation.item.commit")]
    ConversationItemCommit {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },

    /// Clear the audio buffer
    #[serde(rename = "input_audio_buffer.clear")]
    InputAudioBufferClear {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },

    /// Create a conversation item (e.g., tool result)
    #[serde(rename = "conversation.item.create")]
    ConversationItemCreate {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        item: ConversationItem,
    },

    /// Trigger a response from the model
    #[serde(rename = "response.create")]
    ResponseCreate {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<ResponseCreatePayload>,
    },

    /// Cancel an in-progress response
    #[serde(rename = "response.cancel")]
    ResponseCancel {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionUpdatePayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_detection: Option<TurnDetection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GrokToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioConfig>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TurnDetection {
    #[serde(rename = "type")]
    pub detection_type: String, // "server_vad"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_padding_ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub silence_duration_ms: Option<u32>,
}

impl Default for TurnDetection {
    fn default() -> Self {
        Self {
            detection_type: "server_vad".to_string(),
            threshold: Some(0.5),
            prefix_padding_ms: Some(300),
            silence_duration_ms: Some(200),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioConfig {
    pub input: AudioChannelConfig,
    pub output: AudioChannelConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioChannelConfig {
    pub format: AudioFormat,
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioFormat {
    #[serde(rename = "type")]
    pub format_type: String, // "audio/pcm", "audio/pcmu", "audio/pcma"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrokToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>, // JSON Schema
}

impl GrokToolDefinition {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            name: name.into(),
            description: Some(description.into()),
            parameters: Some(parameters),
        }
    }
}

impl From<&crate::tools::ToolDefinition> for GrokToolDefinition {
    fn from(tool: &crate::tools::ToolDefinition) -> Self {
        Self {
            tool_type: "function".to_string(),
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: Some(tool.parameters_json_schema.clone()),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ConversationItem {
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ContentPart>>,
}

impl ConversationItem {
    /// Create a function call output item
    pub fn function_call_output(call_id: String, output: String) -> Self {
        Self {
            item_type: "function_call_output".to_string(),
            id: None,
            call_id: Some(call_id),
            output: Some(output),
            role: None,
            content: None,
        }
    }

    /// Create a user text message item
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            item_type: "message".to_string(),
            id: None,
            call_id: None,
            output: None,
            role: Some("user".to_string()),
            content: Some(vec![ContentPart {
                content_type: "input_text".to_string(),
                text: Some(text.into()),
                audio: None,
            }]),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseCreatePayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
}

/// Events received from Grok server
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerEvent {
    /// Session created
    #[serde(rename = "session.created")]
    SessionCreated { session: SessionInfo },

    /// Session updated
    #[serde(rename = "session.updated")]
    SessionUpdated { session: SessionInfo },

    /// Conversation created
    #[serde(rename = "conversation.created")]
    ConversationCreated {
        event_id: String,
        conversation: ConversationInfo,
        #[serde(default)]
        previous_item_id: Option<String>,
    },

    /// Audio delta from model response
    #[serde(rename = "response.audio.delta")]
    ResponseAudioDelta {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String, // base64 encoded audio
    },

    /// Alternative event name for audio delta
    #[serde(rename = "response.output_audio.delta")]
    ResponseOutputAudioDelta {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },

    /// Function call arguments streaming
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        call_id: String,
        delta: String,
    },

    /// Function call arguments complete
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone {
        event_id: String,
        response_id: String,
        item_id: String,
        output_index: u32,
        call_id: String,
        name: String,
        arguments: String,
    },

    /// Response completed
    #[serde(rename = "response.done")]
    ResponseDone {
        event_id: String,
        response_id: String,
        #[serde(default)]
        response: Option<ResponseInfo>,
    },

    /// Speech started in input buffer
    #[serde(rename = "input_audio_buffer.speech_started")]
    InputAudioBufferSpeechStarted {
        event_id: String,
        audio_start_ms: u64,
        item_id: String,
    },

    /// Speech stopped in input buffer
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    InputAudioBufferSpeechStopped {
        event_id: String,
        audio_end_ms: u64,
        item_id: String,
    },

    /// Input audio buffer committed
    #[serde(rename = "input_audio_buffer.committed")]
    InputAudioBufferCommitted {
        event_id: String,
        item_id: String,
        previous_item_id: Option<String>,
    },

    /// Input audio transcription completed
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    InputAudioTranscriptionCompleted {
        event_id: String,
        item_id: String,
        transcript: String,
        content_index: u32,
        status: String,
        #[serde(default)]
        previous_item_id: Option<String>,
    },

    /// Output audio transcript delta
    #[serde(rename = "response.output_audio_transcript.delta")]
    ResponseOutputAudioTranscriptDelta {
        event_id: String,
        item_id: String,
        response_id: String,
        delta: String,
        content_index: u32,
        output_index: u32,
        #[serde(default)]
        start_time: Option<f32>,
        #[serde(default)]
        previous_item_id: Option<String>,
    },

    /// Output audio transcript completed
    #[serde(rename = "response.output_audio_transcript.done")]
    ResponseOutputAudioTranscriptDone {
        event_id: String,
        item_id: String,
        response_id: String,
        transcript: String,
        content_index: u32,
        output_index: u32,
        #[serde(default)]
        previous_item_id: Option<String>,
    },

    /// Rate limits updated
    #[serde(rename = "rate_limits.updated")]
    RateLimitsUpdated {
        event_id: String,
        rate_limits: Vec<RateLimit>,
    },

    /// Error from server
    #[serde(rename = "error")]
    Error { event_id: String, error: ErrorInfo },

    /// Catch-all for unknown events
    #[serde(other)]
    Unknown,
}

impl ServerEvent {
    /// Extract audio delta if this is an audio event
    pub fn audio_delta(&self) -> Option<&str> {
        match self {
            Self::ResponseAudioDelta { delta, .. } => Some(delta),
            Self::ResponseOutputAudioDelta { delta, .. } => Some(delta),
            _ => None,
        }
    }

    /// Check if this is a function call completion
    pub fn function_call(&self) -> Option<FunctionCall> {
        match self {
            Self::ResponseFunctionCallArgumentsDone {
                call_id,
                name,
                arguments,
                ..
            } => Some(FunctionCall {
                call_id: call_id.clone(),
                name: name.clone(),
                arguments: arguments.clone(),
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

impl FunctionCall {
    pub fn to_tool_call_part(&self) -> ToolCallPart {
        let args = serde_json::from_str::<Value>(&self.arguments)
            .unwrap_or_else(|_| Value::String(self.arguments.clone()));
        ToolCallPart {
            id: self.call_id.clone(),
            name: self.name.clone(),
            arguments: args,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConversationInfo {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SessionInfo {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseInfo {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RateLimit {
    pub name: String,
    pub limit: u32,
    pub remaining: u32,
    pub reset_seconds: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorInfo {
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
    pub message: String,
}

/// Configuration for a Grok Realtime session
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub instructions: String,
    pub voice: String,
    pub tools: Vec<GrokToolDefinition>,
    pub temperature: f32,
    pub audio_format: AudioFormat,
    pub turn_detection: TurnDetection,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            instructions: "You are a helpful voice assistant.".to_string(),
            voice: "Ara".to_string(),
            tools: Vec::new(),
            temperature: 0.8,
            audio_format: AudioFormat {
                format_type: "audio/pcmu".to_string(),
                rate: None,
            },
            turn_detection: TurnDetection::default(),
        }
    }
}

impl SessionConfig {
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            ..Default::default()
        }
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = voice.into();
        self
    }

    pub fn with_tools(mut self, tools: Vec<GrokToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_rustic_tools(mut self, tools: &[crate::tools::ToolDefinition]) -> Self {
        self.tools = tools.iter().map(GrokToolDefinition::from).collect();
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_audio_format(mut self, format_type: impl Into<String>, rate: Option<u32>) -> Self {
        self.audio_format = AudioFormat {
            format_type: format_type.into(),
            rate,
        };
        self
    }

    pub fn with_turn_detection(mut self, detection: TurnDetection) -> Self {
        self.turn_detection = detection;
        self
    }

    /// Convert to session update payload for the API
    pub fn to_update_payload(&self) -> SessionUpdatePayload {
        SessionUpdatePayload {
            instructions: Some(self.instructions.clone()),
            voice: Some(self.voice.clone()),
            turn_detection: Some(self.turn_detection.clone()),
            tools: if self.tools.is_empty() {
                None
            } else {
                Some(self.tools.clone())
            },
            temperature: Some(self.temperature),
            audio: Some(AudioConfig {
                input: AudioChannelConfig {
                    format: self.audio_format.clone(),
                },
                output: AudioChannelConfig {
                    format: self.audio_format.clone(),
                },
            }),
        }
    }
}

/// Handle for sending events to Grok
#[derive(Clone)]
pub struct GrokSender {
    tx: mpsc::Sender<ClientEvent>,
}

impl GrokSender {
    /// Send audio data to Grok
    pub async fn send_audio(&self, audio_base64: String) -> Result<()> {
        self.tx
            .send(ClientEvent::InputAudioBufferAppend {
                event_id: None,
                audio: audio_base64,
            })
            .await
            .map_err(|_| Error::ConnectionClosed)
    }

    /// Send a tool result back to Grok
    pub async fn send_tool_result(&self, call_id: String, result: String) -> Result<()> {
        self.tx
            .send(ClientEvent::ConversationItemCreate {
                event_id: None,
                item: ConversationItem::function_call_output(call_id, result),
            })
            .await
            .map_err(|_| Error::ConnectionClosed)?;

        self.tx
            .send(ClientEvent::ResponseCreate {
                event_id: None,
                response: None,
            })
            .await
            .map_err(|_| Error::ConnectionClosed)
    }

    /// Send a user text message to Grok
    pub async fn send_user_text(&self, text: String) -> Result<()> {
        self.tx
            .send(ClientEvent::ConversationItemCreate {
                event_id: None,
                item: ConversationItem::user_text(text),
            })
            .await
            .map_err(|_| Error::ConnectionClosed)
    }

    /// Request a model response
    pub async fn request_response(&self, modalities: Option<Vec<String>>) -> Result<()> {
        self.tx
            .send(ClientEvent::ResponseCreate {
                event_id: None,
                response: Some(ResponseCreatePayload { modalities }),
            })
            .await
            .map_err(|_| Error::ConnectionClosed)
    }

    /// Cancel the current response (e.g., on interruption)
    pub async fn cancel_response(&self) -> Result<()> {
        self.tx
            .send(ClientEvent::ResponseCancel { event_id: None })
            .await
            .map_err(|_| Error::ConnectionClosed)
    }

    /// Commit the current input audio buffer
    pub async fn commit_audio(&self) -> Result<()> {
        self.tx
            .send(ClientEvent::ConversationItemCommit { event_id: None })
            .await
            .map_err(|_| Error::ConnectionClosed)
    }
}

/// Grok Realtime API client
pub struct GrokClient {
    ws_url: String,
    api_key: String,
}

impl GrokClient {
    pub fn new(ws_url: String, api_key: String) -> Self {
        Self { ws_url, api_key }
    }

    /// Connect to Grok and return sender/receiver handles
    ///
    /// Returns:
    /// - `GrokSender`: For sending audio and tool results
    /// - `mpsc::Receiver<ServerEvent>`: For receiving events from Grok
    pub async fn connect(
        &self,
        session_config: SessionConfig,
    ) -> Result<(GrokSender, mpsc::Receiver<ServerEvent>)> {
        let request = Request::builder()
            .uri(&self.ws_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Sec-WebSocket-Key", generate_ws_key())
            .header("Sec-WebSocket-Version", "13")
            .header("Connection", "Upgrade")
            .header("Upgrade", "websocket")
            .header("Host", extract_host(&self.ws_url))
            .body(())
            .map_err(|e| Error::Provider(format!("failed to build request: {e}")))?;

        info!(url = %self.ws_url, "Connecting to Grok Realtime API");

        let (ws_stream, _response) = connect_async(request)
            .await
            .map_err(|e| Error::Provider(format!("websocket connection failed: {e}")))?;

        info!("Connected to Grok Realtime API");

        let (mut ws_sink, mut ws_stream_rx) = ws_stream.split();

        let (client_tx, mut client_rx) = mpsc::channel::<ClientEvent>(256);
        let (server_tx, server_rx) = mpsc::channel::<ServerEvent>(256);

        let session_update = ClientEvent::SessionUpdate {
            session: session_config.to_update_payload(),
        };
        let msg = serde_json::to_string(&session_update)?;
        ws_sink
            .send(Message::Text(msg))
            .await
            .map_err(|e| Error::Provider(format!("failed to send session update: {e}")))?;
        debug!("Sent session.update");

        tokio::spawn(async move {
            while let Some(event) = client_rx.recv().await {
                match serde_json::to_string(&event) {
                    Ok(msg) => {
                        if let Err(e) = ws_sink.send(Message::Text(msg)).await {
                            error!(error = %e, "Failed to send to Grok WebSocket");
                            break;
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Failed to serialize client event");
                    }
                }
            }
            debug!("Grok sender task ended");
        });

        tokio::spawn(async move {
            while let Some(msg_result) = ws_stream_rx.next().await {
                match msg_result {
                    Ok(Message::Text(text)) => match serde_json::from_str::<Value>(&text) {
                        Ok(value) => {
                            let event_type = value
                                .get("type")
                                .and_then(|val| val.as_str())
                                .unwrap_or("unknown");
                            match serde_json::from_value::<ServerEvent>(value.clone()) {
                                Ok(event) => {
                                    if matches!(event, ServerEvent::Unknown) {
                                        trace!(event_type = %event_type, raw = %text, "Unhandled Grok event");
                                    } else if event.audio_delta().is_none() {
                                        debug!(?event, "Received Grok event");
                                    }
                                    if server_tx.send(event).await.is_err() {
                                        debug!("Server event receiver dropped");
                                        break;
                                    }
                                }
                                Err(e) => {
                                    warn!(
                                        error = %e,
                                        event_type = %event_type,
                                        "Failed to parse Grok event"
                                    );
                                    trace!(raw = %text, "Grok event parse failure payload");
                                }
                            }
                        }
                        Err(e) => {
                            warn!(error = %e, "Failed to parse Grok event");
                            trace!(raw = %text, "Grok event parse failure payload");
                        }
                    },
                    Ok(Message::Close(_)) => {
                        info!("Grok WebSocket closed");
                        break;
                    }
                    Ok(Message::Ping(data)) => {
                        debug!("Received ping from Grok");
                        let _ = data;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        error!(error = %e, "Grok WebSocket error");
                        break;
                    }
                }
            }
            debug!("Grok receiver task ended");
        });

        Ok((GrokSender { tx: client_tx }, server_rx))
    }
}

fn generate_ws_key() -> String {
    let mut key = [0u8; 16];
    for (i, byte) in key.iter_mut().enumerate() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0));
        *byte = (now.as_nanos() as u8).wrapping_add(i as u8);
    }
    base64::engine::general_purpose::STANDARD.encode(key)
}

fn extract_host(url: &str) -> String {
    url.replace("wss://", "")
        .replace("ws://", "")
        .split('/')
        .next()
        .unwrap_or("api.x.ai")
        .to_string()
}
