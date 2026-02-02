#[path = "support/scripted_model.rs"]
mod scripted_support;

use std::sync::Arc;

use rustic_ai::{
    Agent, ModelResponse, ModelResponsePart, RunInput, TextPart, UsageLimits, UserContent,
};
use scripted_support::ScriptedModel;

#[tokio::test]
async fn scripted_model_returns_queued_response() {
    let response = ModelResponse {
        parts: vec![ModelResponsePart::Text(TextPart {
            content: "ok".to_string(),
        })],
        usage: None,
        model_name: Some("scripted".to_string()),
        finish_reason: Some("stop".to_string()),
    };

    let model = Arc::new(ScriptedModel::new("scripted", Vec::new()));
    model.push_response(response).await;
    let agent = Agent::new(model);
    let input = RunInput::new(
        vec![UserContent::Text("hello".to_string())],
        vec![],
        (),
        UsageLimits::default(),
    );

    let result = agent.run(input).await.expect("run succeeds");
    assert_eq!(result.output, "ok");
}
