#![forbid(unsafe_code)]

use std::error::Error;
use std::io::{IsTerminal, Write};

use clap::Parser;
use rustic_ai::mcp::McpServerStreamableHttp;
use rustic_ai::{
    Agent, ModelMessage, RunInput, UsageLimits, UserContent, infer_model, infer_provider,
};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

const DEFAULT_PROMPT: &str = "I am patient P123. Book me with dr_smith on 2026-01-23 at 09:30 for a checkup. My phone is 555-0100.";

#[derive(Parser, Debug)]
#[command(name = "assistant")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:9099/rpc")]
    mcp_url: String,
    #[arg(long, default_value = "gemini:gemini-2.0-flash")]
    model: String,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value = "Rustic Care Clinic")]
    clinic_name: String,
    #[arg(long, default_value = "America/Los_Angeles")]
    timezone: String,
}

#[derive(Clone)]
struct ClinicDeps {
    clinic_name: String,
    timezone: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let Args {
        mcp_url,
        model,
        prompt,
        clinic_name,
        timezone,
    } = Args::parse();

    let model = infer_model(&model, infer_provider)?;
    let deps = ClinicDeps {
        clinic_name,
        timezone,
    };
    let mut agent = Agent::new(model).system_prompt(build_system_prompt(&deps));

    let mcp = McpServerStreamableHttp::new(&mcp_url)?;
    agent.toolset(mcp);

    let interactive = std::io::stdin().is_terminal();
    if interactive {
        run_interactive(&agent, deps, prompt).await?;
    } else {
        let prompt = resolve_noninteractive(prompt).await?;
        let input = RunInput::new(
            vec![UserContent::Text(prompt)],
            vec![],
            deps,
            UsageLimits::default(),
        );
        let result = agent.run_with_toolsets(input).await?;
        println!("{}", result.output);
    }

    Ok(())
}

fn build_system_prompt(deps: &ClinicDeps) -> String {
    format!(
        "You are a medical office scheduling assistant for {clinic} in the {tz} time zone.\n\
Use the provided tools to check availability and schedule appointments.\n\
Only schedule if all required fields are present: patient_id, provider_id, date (YYYY-MM-DD), time (HH:MM 24-hour), reason, contact_phone.\n\
If anything is missing or ambiguous, ask a concise follow-up question.\n\
Confirm details after scheduling. Do not provide medical advice.\n\
Available providers: dr_smith, dr_lee, dr_patel.",
        clinic = deps.clinic_name,
        tz = deps.timezone
    )
}

async fn resolve_noninteractive(
    prompt: Option<String>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if let Some(prompt) = prompt {
        if !prompt.trim().is_empty() {
            return Ok(prompt);
        }
    }

    let mut input = String::new();
    tokio::io::stdin().read_to_string(&mut input).await?;
    let trimmed = input.trim();
    if trimmed.is_empty() {
        Ok(DEFAULT_PROMPT.to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

async fn run_interactive(
    agent: &Agent<ClinicDeps>,
    deps: ClinicDeps,
    prompt: Option<String>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("Interactive mode. Type /quit to exit.");

    let mut reader = BufReader::new(tokio::io::stdin());
    let mut message_history: Vec<ModelMessage> = Vec::new();
    let mut first = true;
    let mut pending_prompt = prompt.filter(|value| !value.trim().is_empty());

    loop {
        let user_text = if let Some(value) = pending_prompt.take() {
            value
        } else {
            let allow_default = first;
            match prompt_for_input(&mut reader, allow_default).await? {
                Some(value) => value,
                None => break,
            }
        };

        if user_text.is_empty() {
            continue;
        }
        if user_text == "/quit" || user_text == "/exit" {
            break;
        }

        let mut input = RunInput::new(
            vec![UserContent::Text(user_text)],
            message_history,
            deps.clone(),
            UsageLimits::default(),
        );
        if !first {
            input.include_system_prompt = false;
        }

        let result = agent.run_with_toolsets(input).await?;
        println!("{}", result.output);

        message_history = result.messages;
        first = false;
    }

    Ok(())
}

async fn prompt_for_input(
    reader: &mut BufReader<tokio::io::Stdin>,
    allow_default: bool,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    if allow_default {
        println!("Enter an appointment request (press Enter for the default example):");
    } else {
        println!("Enter a follow-up message (or /quit to exit):");
    }
    print!("> ");
    std::io::stdout().flush()?;

    let mut line = String::new();
    let bytes = reader.read_line(&mut line).await?;
    if bytes == 0 {
        return Ok(None);
    }
    let trimmed = line.trim();
    if trimmed.is_empty() {
        if allow_default {
            return Ok(Some(DEFAULT_PROMPT.to_string()));
        }
        return Ok(Some(String::new()));
    }

    Ok(Some(trimmed.to_string()))
}
