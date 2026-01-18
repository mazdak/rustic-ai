use std::sync::Arc;

use async_trait::async_trait;
use rustic_ai::{
    Agent, InMemoryResolver, Model, ModelConfigEntry, ModelError, ModelRequestParameters,
    ModelResponse, ModelResponsePart, RunInput, TextPart, UsageLimits, UserContent,
};

struct FixedModel {
    name: String,
    should_fail: bool,
}

impl FixedModel {
    fn new(name: impl Into<String>, should_fail: bool) -> Self {
        Self {
            name: name.into(),
            should_fail,
        }
    }
}

#[async_trait]
impl Model for FixedModel {
    fn name(&self) -> &str {
        &self.name
    }

    async fn request(
        &self,
        _messages: &[rustic_ai::ModelMessage],
        _settings: Option<&rustic_ai::ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        if self.should_fail {
            return Err(ModelError::HttpStatus { status: 429 });
        }

        Ok(ModelResponse {
            parts: vec![ModelResponsePart::Text(TextPart {
                content: "ok".to_string(),
            })],
            usage: None,
            model_name: Some(self.name.clone()),
            finish_reason: Some("stop".to_string()),
        })
    }
}

#[tokio::test]
async fn agent_run_with_failover_uses_backup() {
    let agent = Agent::new(Arc::new(FixedModel::new("base", false)));

    let mut resolver = InMemoryResolver::new("primary");
    resolver.insert_agent(
        "agent",
        ModelConfigEntry::default()
            .backup("backup")
            .failover_on(["http_429"]),
    );

    let input = RunInput::new(
        vec![UserContent::Text("hi".to_string())],
        Vec::new(),
        (),
        UsageLimits::default(),
    );

    let result = agent
        .run_with_failover(input, &resolver, "agent", None, None, |model_name| {
            let should_fail = model_name == "primary";
            Ok(Arc::new(FixedModel::new(model_name, should_fail)) as Arc<dyn Model>)
        })
        .await
        .expect("failover result");

    assert!(result.failed_over);
    assert_eq!(result.model_used, "backup");
    assert_eq!(result.value.output, "ok");
}
