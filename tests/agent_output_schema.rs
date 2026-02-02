use std::sync::Arc;

use async_trait::async_trait;
use rustic_ai::{
    Agent, AgentError, Model, ModelMessage, ModelRequestParameters, ModelResponse,
    ModelResponsePart, RunInput, TextPart, UsageLimits, UserContent,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

struct StaticModel {
    response: ModelResponse,
}

impl StaticModel {
    fn new(response: ModelResponse) -> Self {
        Self { response }
    }
}

#[async_trait]
impl Model for StaticModel {
    fn name(&self) -> &str {
        "static-model"
    }

    async fn request(
        &self,
        _messages: &[ModelMessage],
        _settings: Option<&rustic_ai::ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, rustic_ai::model::ModelError> {
        Ok(self.response.clone())
    }
}

fn text_response(text: &str) -> ModelResponse {
    ModelResponse {
        parts: vec![ModelResponsePart::Text(TextPart {
            content: text.to_string(),
        })],
        usage: None,
        model_name: Some("static-model".to_string()),
        finish_reason: Some("stop".to_string()),
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AnswerOutput {
    answer: String,
}

#[tokio::test]
async fn output_schema_rejects_non_json_by_default() {
    let model = Arc::new(StaticModel::new(text_response("not json")));
    let schema = json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    });

    let agent = Agent::new(model).output_schema(schema);

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await;
    match result {
        Err(AgentError::OutputValidation(_)) => {}
        other => panic!("expected output validation error, got: {other:?}"),
    }
}

#[tokio::test]
async fn output_schema_allows_text_when_configured() {
    let model = Arc::new(StaticModel::new(text_response("not json")));
    let schema = json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    });

    let agent = Agent::new(model)
        .output_schema(schema)
        .allow_text_output(true);

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await.expect("run succeeds");
    assert_eq!(result.output, "not json");
    assert!(result.parsed_output.is_none());
}

#[tokio::test]
async fn output_schema_for_accepts_typed_schema() {
    let sample = AnswerOutput {
        answer: "ok".to_string(),
    };
    assert_eq!(sample.answer, "ok");

    let model = Arc::new(StaticModel::new(text_response(r#"{"answer":"ok"}"#)));
    let agent = Agent::new(model).output_schema_for::<AnswerOutput>();

    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await.expect("run succeeds");
    let parsed = result.parsed_output.expect("parsed output present");
    assert_eq!(
        parsed.get("answer").and_then(|value| value.as_str()),
        Some("ok")
    );
}
