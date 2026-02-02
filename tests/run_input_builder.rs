use rustic_ai::{ModelMessage, ModelRequest, RunInput, UsageLimits, UserContent};

#[test]
fn run_input_builder_sets_prompt_and_defaults() {
    let input = RunInput::builder(()).user_text("hello").build();

    assert!(matches!(
        input.user_prompt.as_slice(),
        [UserContent::Text(text)] if text == "hello"
    ));
    assert!(input.message_history.is_empty());
    assert!(input.include_system_prompt);
    assert!(input.run_id.is_none());
}

#[test]
fn run_input_builder_allows_customization() {
    let history = vec![ModelMessage::Request(ModelRequest::user_text_prompt(
        "prior",
    ))];
    let limits = UsageLimits {
        request_limit: Some(1),
        ..Default::default()
    };

    let input = RunInput::builder("deps")
        .message_history(history)
        .usage_limits(limits.clone())
        .include_system_prompt(false)
        .run_id("run-123")
        .prompt(vec![UserContent::Text("question".to_string())])
        .build();

    assert_eq!(input.message_history.len(), 1);
    assert_eq!(input.usage_limits.request_limit, limits.request_limit);
    assert!(!input.include_system_prompt);
    assert_eq!(input.run_id.as_deref(), Some("run-123"));
}
