use std::error::Error;
use std::sync::Arc;
use std::sync::Once;
use std::time::{SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use rustic_ai::{
    Agent, FunctionTool, RunInput, UsageLimits, UserContent, infer_model, infer_provider,
};

#[derive(Deserialize, JsonSchema)]
struct TokenArgs {
    #[serde(default)]
    nonce: Option<String>,
}

#[derive(Serialize)]
struct TokenResult {
    token: String,
}

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn load_env() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = dotenvy::dotenv();
    });
}

fn live_tests_enabled() -> bool {
    env_truthy("RUSTIC_AI_LIVE_TESTS")
}

fn allow_fallbacks() -> bool {
    env_truthy("RUSTIC_AI_LIVE_ALLOW_FALLBACKS")
}

fn error(msg: impl Into<String>) -> Box<dyn Error + Send + Sync> {
    std::io::Error::other(msg.into()).into()
}

fn fresh_token() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("tok_{nanos}")
}

fn resolve_model(
    provider: &str,
    api_key_envs: &[&str],
    model_env: &str,
    fallback: Option<&str>,
) -> Option<String> {
    load_env();
    if !live_tests_enabled() {
        eprintln!("live tests disabled; set RUSTIC_AI_LIVE_TESTS=1 to run");
        return None;
    }
    if !api_key_envs
        .iter()
        .any(|env_name| std::env::var(env_name).is_ok())
    {
        eprintln!("skipping: missing {:?}", api_key_envs);
        return None;
    }
    let model = std::env::var(model_env).ok().or_else(|| {
        if allow_fallbacks() {
            fallback.map(|v| v.to_string())
        } else {
            None
        }
    });
    if model.is_none() {
        eprintln!(
            "skipping: missing {model_env} for provider {provider} (set RUSTIC_AI_LIVE_ALLOW_FALLBACKS=1 to use defaults)"
        );
    }
    model
}

fn normalize_model(provider: &str, model: &str) -> String {
    if model.contains(':') {
        model.to_string()
    } else {
        format!("{provider}:{model}")
    }
}

async fn run_tool_roundtrip(
    provider: &str,
    api_key_envs: &[&str],
    model_env: &str,
    fallback: Option<&str>,
    strict_json: bool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let model_name = match resolve_model(provider, api_key_envs, model_env, fallback) {
        Some(model) => model,
        None => return Ok(()),
    };

    let model = infer_model(normalize_model(provider, &model_name), infer_provider)?;
    let mut agent = Agent::new(model).system_prompt(
        "You must call the get_token tool before answering. Respond with JSON only.",
    );

    let token = Arc::new(fresh_token());
    let tool_token = Arc::clone(&token);
    let tool = FunctionTool::new(
        "get_token",
        "return a one-time token",
        move |_, args: TokenArgs| {
            let tool_token = Arc::clone(&tool_token);
            let _ = args.nonce;
            async move {
                Ok(TokenResult {
                    token: (*tool_token).clone(),
                })
            }
        },
    )?;

    agent.tool(tool);

    let schema = json!({
        "type": "object",
        "properties": {
            "token": { "type": "string" }
        },
        "required": ["token"],
        "additionalProperties": false
    });
    let agent = if strict_json {
        agent.output_schema(schema)
    } else {
        agent
    };

    let input = RunInput::new(
        vec![UserContent::Text(
            "Call the get_token tool (no args) and return {\"token\":\"...\"} only.".to_string(),
        )],
        vec![],
        (),
        UsageLimits::default(),
    );
    let result = agent.run(input).await?;
    let parsed = if strict_json {
        result
            .parsed_output
            .ok_or_else(|| error("missing parsed output"))?
    } else {
        serde_json::from_str(&result.output)?
    };
    let token_value = parsed
        .get("token")
        .and_then(|value| value.as_str())
        .ok_or_else(|| error("missing token field"))?;
    if token_value != token.as_ref() {
        return Err(error(format!(
            "token mismatch: expected {}, got {}",
            token.as_ref(),
            token_value
        )));
    }

    let input = RunInput::new(
        vec![UserContent::Text(
            "Return the same token again as {\"token\":\"...\"} only.".to_string(),
        )],
        result.messages,
        (),
        UsageLimits::default(),
    );
    let result = agent.run(input).await?;
    let parsed = if strict_json {
        result
            .parsed_output
            .ok_or_else(|| error("missing parsed output on roundtrip"))?
    } else {
        serde_json::from_str(&result.output)?
    };
    let token_value = parsed
        .get("token")
        .and_then(|value| value.as_str())
        .ok_or_else(|| error("missing token field on roundtrip"))?;
    if token_value != token.as_ref() {
        return Err(error(format!(
            "roundtrip token mismatch: expected {}, got {}",
            token.as_ref(),
            token_value
        )));
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn live_openai_tool_roundtrip() -> Result<(), Box<dyn Error + Send + Sync>> {
    run_tool_roundtrip(
        "openai",
        &["OPENAI_API_KEY"],
        "OPENAI_MODEL",
        Some("gpt-5-mini"),
        true,
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn live_anthropic_tool_roundtrip() -> Result<(), Box<dyn Error + Send + Sync>> {
    run_tool_roundtrip(
        "anthropic",
        &["ANTHROPIC_API_KEY"],
        "ANTHROPIC_MODEL",
        Some("claude-sonnet-4-5"),
        true,
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn live_gemini_tool_roundtrip() -> Result<(), Box<dyn Error + Send + Sync>> {
    run_tool_roundtrip(
        "gemini",
        &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "GEMINI_MODEL",
        Some("gemini-2.5-flash"),
        false,
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn live_grok_tool_roundtrip() -> Result<(), Box<dyn Error + Send + Sync>> {
    run_tool_roundtrip(
        "grok",
        &["XAI_API_KEY", "GROK_API_KEY"],
        "GROK_MODEL",
        None,
        true,
    )
    .await
}
