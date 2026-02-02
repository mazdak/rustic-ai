use serde_json::json;

use rustic_ai::{ToolDefinition, ToolKind};

#[test]
fn tool_definition_builder_sets_fields() {
    let def = ToolDefinition::new("tool", Some("desc".to_string()), json!({"type": "object"}))
        .with_kind(ToolKind::External)
        .with_metadata(json!({"version": 1}))
        .with_sequential(true)
        .with_timeout(1.5);

    assert_eq!(def.name, "tool");
    assert_eq!(def.description.as_deref(), Some("desc"));
    assert_eq!(def.kind, ToolKind::External);
    assert_eq!(def.metadata, Some(json!({"version": 1})));
    assert!(def.sequential);
    assert_eq!(def.timeout, Some(1.5));
}
