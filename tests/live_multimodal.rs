use std::error::Error;
use std::sync::Once;

use rustic_ai::{
    Agent, BinaryContent, ImageUrl, RunInput, UsageLimits, UserContent, infer_model, infer_provider,
};

fn load_env() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = dotenvy::dotenv();
    });
}

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn live_tests_enabled() -> bool {
    env_truthy("RUSTIC_AI_LIVE_TESTS")
}

fn allow_fallbacks() -> bool {
    env_truthy("RUSTIC_AI_LIVE_ALLOW_FALLBACKS")
}

fn resolve_model(api_key_envs: &[&str], model_env: &str, fallback: Option<&str>) -> Option<String> {
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
            "skipping: missing {model_env} (set RUSTIC_AI_LIVE_ALLOW_FALLBACKS=1 to use defaults)"
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

fn fixture_bytes(name: &str) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name);
    Ok(std::fs::read(path)?)
}

struct MultimodalCase {
    fixture: &'static str,
    media_type: &'static str,
    prompt: &'static str,
    expected_tokens: &'static [&'static str],
}

async fn run_multimodal_suite_for_model(
    provider: &str,
    model_name: &str,
    cases: &[MultimodalCase],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let model = infer_model(normalize_model(provider, model_name), infer_provider)?;
    let agent = Agent::new(model).system_prompt(
        "Be concise. If you see text in the media, repeat it exactly. If you see a color, answer the color.",
    );

    for case in cases {
        let data = fixture_bytes(case.fixture)?;
        let input = RunInput::new(
            vec![
                UserContent::Binary(BinaryContent {
                    data,
                    media_type: case.media_type.to_string(),
                }),
                UserContent::Text(case.prompt.to_string()),
            ],
            vec![],
            (),
            UsageLimits::default(),
        );
        let result = agent.run(input).await?;
        let output = result.output.to_lowercase();
        let missing = case
            .expected_tokens
            .iter()
            .filter(|needle| !output.contains(&needle.to_lowercase()))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(format!(
                "expected output to contain {:?}, got {:?}",
                case.expected_tokens, output
            )
            .into());
        }
    }

    Ok(())
}

async fn run_multimodal_suite(
    provider: &str,
    api_key_envs: &[&str],
    model_env: &str,
    fallback: Option<&str>,
    cases: &[MultimodalCase],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let model_name = match resolve_model(api_key_envs, model_env, fallback) {
        Some(model) => model,
        None => return Ok(()),
    };
    run_multimodal_suite_for_model(provider, &model_name, cases).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn live_openai_multimodal() -> Result<(), Box<dyn Error + Send + Sync>> {
    run_multimodal_suite(
        "openai",
        &["OPENAI_API_KEY"],
        "OPENAI_MODEL",
        Some("gpt-5-mini"),
        &[MultimodalCase {
            fixture: "fixture.jpg",
            media_type: "image/jpeg",
            prompt: "What is the dominant color?",
            expected_tokens: &["red"],
        }],
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn live_anthropic_multimodal() -> Result<(), Box<dyn Error + Send + Sync>> {
    run_multimodal_suite(
        "anthropic",
        &["ANTHROPIC_API_KEY"],
        "ANTHROPIC_MODEL",
        Some("claude-sonnet-4-5"),
        &[
            MultimodalCase {
                fixture: "fixture.jpg",
                media_type: "image/jpeg",
                prompt: "What is the dominant color?",
                expected_tokens: &["red"],
            },
            MultimodalCase {
                fixture: "fixture.pdf",
                media_type: "application/pdf",
                prompt: "What text is in the document?",
                expected_tokens: &["rustic", "test"],
            },
        ],
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn live_gemini_multimodal() -> Result<(), Box<dyn Error + Send + Sync>> {
    run_multimodal_suite(
        "gemini",
        &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "GEMINI_MODEL",
        Some("gemini-2.5-flash"),
        &[
            MultimodalCase {
                fixture: "fixture.jpg",
                media_type: "image/jpeg",
                prompt: "What is the dominant color?",
                expected_tokens: &["red"],
            },
            MultimodalCase {
                fixture: "fixture.pdf",
                media_type: "application/pdf",
                prompt: "What text is in the document?",
                expected_tokens: &["rust", "test"],
            },
            MultimodalCase {
                fixture: "fixture.m4a",
                media_type: "audio/aac",
                prompt: "Transcribe the audio.",
                expected_tokens: &["rust", "test"],
            },
        ],
    )
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn live_grok_multimodal() -> Result<(), Box<dyn Error + Send + Sync>> {
    let model_name = match resolve_model(&["XAI_API_KEY", "GROK_API_KEY"], "GROK_MODEL", None) {
        Some(model) => model,
        None => return Ok(()),
    };

    let model = infer_model(normalize_model("grok", &model_name), infer_provider)?;
    let agent = Agent::new(model).system_prompt(
        "Be concise. If you see text in the media, repeat it exactly. If you see a color, answer the color.",
    );

    let input = RunInput::new(
        vec![
            UserContent::Image(ImageUrl {
                url: "https://dummyimage.com/256x256/ff0000/ff0000.png".to_string(),
                media_type: Some("image/png".to_string()),
            }),
            UserContent::Text("What is the dominant color?".to_string()),
        ],
        vec![],
        (),
        UsageLimits::default(),
    );
    let result = agent.run(input).await?;
    let output = result.output.to_lowercase();
    if !output.contains("red") {
        return Err(format!("expected output to contain \"red\", got {:?}", output).into());
    }
    Ok(())
}
