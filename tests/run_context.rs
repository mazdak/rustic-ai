use std::sync::Arc;

use async_trait::async_trait;
use rustic_ai::{
    Model, ModelError, ModelMessage, ModelRequestParameters, ModelResponse, RunContext, RunUsage,
};

struct DummyModel;

#[async_trait]
impl Model for DummyModel {
    fn name(&self) -> &str {
        "dummy"
    }

    async fn request(
        &self,
        _messages: &[ModelMessage],
        _settings: Option<&rustic_ai::ModelSettings>,
        _params: &ModelRequestParameters,
    ) -> Result<ModelResponse, ModelError> {
        Err(ModelError::Unsupported("dummy".to_string()))
    }
}

#[test]
fn run_context_for_tool_call_sets_fields() {
    let ctx = RunContext {
        run_id: "run".to_string(),
        deps: Arc::new(()),
        model: Arc::new(DummyModel),
        usage: RunUsage::default(),
        prompt: None,
        messages: Arc::new(Vec::new()),
        tool_call_id: None,
        tool_name: None,
    };

    let child = ctx.for_tool_call("call-1".to_string(), "tool".to_string());
    assert_eq!(child.run_id, "run");
    assert_eq!(child.tool_call_id.as_deref(), Some("call-1"));
    assert_eq!(child.tool_name.as_deref(), Some("tool"));
}
