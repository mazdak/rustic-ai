#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{Local, NaiveDate, NaiveTime};
use clap::Parser;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

const SLOT_TIMES: [&str; 10] = [
    "09:00",
    "09:30",
    "10:00",
    "10:30",
    "13:00",
    "13:30",
    "14:00",
    "14:30",
    "15:00",
    "15:30",
];

#[derive(Parser, Debug)]
#[command(name = "mcp_server")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:9099")]
    addr: SocketAddr,
}

#[derive(Clone)]
struct AppState {
    tools: Vec<RpcTool>,
    clinic: Arc<Mutex<ClinicState>>,
}

#[derive(Debug, Deserialize)]
struct RpcRequest {
    jsonrpc: String,
    id: Value,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize, Clone)]
struct RpcTool {
    name: String,
    description: Option<String>,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
    #[serde(rename = "outputSchema")]
    output_schema: Option<Value>,
    meta: Option<Value>,
    annotations: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ToolsCallParams {
    name: String,
    arguments: Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GetAvailableSlotsArgs {
    provider_id: String,
    date: String,
}

#[derive(Debug, Serialize, JsonSchema, Clone)]
struct Slot {
    time: String,
    duration_minutes: u32,
}

#[derive(Debug, Serialize, JsonSchema)]
struct AvailabilityResult {
    provider_id: String,
    provider_name: String,
    date: String,
    slots: Vec<Slot>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ScheduleAppointmentArgs {
    patient_id: String,
    provider_id: String,
    date: String,
    time: String,
    reason: String,
    contact_phone: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct AppointmentConfirmation {
    appointment_id: String,
    provider_id: String,
    provider_name: String,
    date: String,
    time: String,
    status: String,
    instructions: String,
}

#[derive(Debug, Clone)]
struct Provider {
    name: String,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct BookingKey {
    provider_id: String,
    date: String,
    time: String,
}

#[derive(Debug)]
struct ClinicState {
    providers: HashMap<String, Provider>,
    bookings: HashSet<BookingKey>,
}

impl ClinicState {
    fn new() -> Self {
        let mut providers = HashMap::new();
        providers.insert(
            "dr_smith".to_string(),
            Provider {
                name: "Dr. Smith".to_string(),
            },
        );
        providers.insert(
            "dr_lee".to_string(),
            Provider {
                name: "Dr. Lee".to_string(),
            },
        );
        providers.insert(
            "dr_patel".to_string(),
            Provider {
                name: "Dr. Patel".to_string(),
            },
        );
        Self {
            providers,
            bookings: HashSet::new(),
        }
    }

    fn booked_times_for(&self, provider_id: &str, date: &str) -> HashSet<String> {
        self.bookings
            .iter()
            .filter(|key| key.provider_id == provider_id && key.date == date)
            .map(|key| key.time.clone())
            .collect()
    }

    fn is_slot_booked(&self, provider_id: &str, date: &str, time: &str) -> bool {
        let key = BookingKey {
            provider_id: provider_id.to_string(),
            date: date.to_string(),
            time: time.to_string(),
        };
        self.bookings.contains(&key)
    }

    fn book_slot(&mut self, provider_id: &str, date: &str, time: &str) {
        let key = BookingKey {
            provider_id: provider_id.to_string(),
            date: date.to_string(),
            time: time.to_string(),
        };
        self.bookings.insert(key);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let args = Args::parse();
    let tools = build_tools()?;
    let state = AppState {
        tools,
        clinic: Arc::new(Mutex::new(ClinicState::new())),
    };

    let app = Router::new()
        .route("/", get(root))
        .route("/rpc", post(rpc_handler))
        .with_state(state);

    info!("MCP server listening on {}", args.addr);
    let listener = tokio::net::TcpListener::bind(args.addr).await?;
    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;

    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("shutdown signal received");
}

async fn root() -> &'static str {
    "MCP scheduling server is running"
}

async fn rpc_handler(
    State(state): State<AppState>,
    Json(request): Json<RpcRequest>,
) -> Json<Value> {
    if request.jsonrpc != "2.0" {
        return rpc_error(request.id, -32600, "Invalid JSON-RPC version");
    }

    info!(method = %request.method, "rpc request");
    match request.method.as_str() {
        "tools/list" => {
            let result = json!({ "tools": state.tools });
            rpc_ok(request.id, result)
        }
        "tools/call" => match parse_params::<ToolsCallParams>(request.params) {
            Ok(call) => handle_tool_call(state, request.id, call).await,
            Err(message) => rpc_error(request.id, -32602, message),
        },
        other => {
            warn!("unknown method: {}", other);
            rpc_error(request.id, -32601, "Method not found")
        }
    }
}

async fn handle_tool_call(
    state: AppState,
    id: Value,
    call: ToolsCallParams,
) -> Json<Value> {
    info!(tool = %call.name, "tool call");
    match call.name.as_str() {
        "get_available_slots" => match parse_args::<GetAvailableSlotsArgs>(call.arguments) {
            Ok(args) => match handle_get_available_slots(state, args).await {
                Ok(result) => rpc_ok(id, json!({ "structuredContent": result })),
                Err(message) => rpc_error(id, -32602, message),
            },
            Err(message) => rpc_error(id, -32602, message),
        },
        "schedule_appointment" => match parse_args::<ScheduleAppointmentArgs>(call.arguments) {
            Ok(args) => match handle_schedule_appointment(state, args).await {
                Ok(result) => rpc_ok(id, json!({ "structuredContent": result })),
                Err(message) => rpc_error(id, -32602, message),
            },
            Err(message) => rpc_error(id, -32602, message),
        },
        other => {
            warn!("unknown tool: {}", other);
            rpc_error(id, -32601, "Tool not found")
        }
    }
}

async fn handle_get_available_slots(
    state: AppState,
    args: GetAvailableSlotsArgs,
) -> Result<AvailabilityResult, String> {
    validate_date(&args.date)?;

    let clinic = state.clinic.lock().await;
    let provider = clinic
        .providers
        .get(&args.provider_id)
        .ok_or_else(|| "Unknown provider_id".to_string())?;

    let booked = clinic.booked_times_for(&args.provider_id, &args.date);
    let slots = SLOT_TIMES
        .iter()
        .filter(|time| !booked.contains(&time.to_string()))
        .map(|time| Slot {
            time: (*time).to_string(),
            duration_minutes: 30,
        })
        .collect();

    Ok(AvailabilityResult {
        provider_id: args.provider_id,
        provider_name: provider.name.clone(),
        date: args.date,
        slots,
    })
}

async fn handle_schedule_appointment(
    state: AppState,
    args: ScheduleAppointmentArgs,
) -> Result<AppointmentConfirmation, String> {
    require_nonempty("patient_id", &args.patient_id)?;
    require_nonempty("provider_id", &args.provider_id)?;
    require_nonempty("reason", &args.reason)?;
    require_nonempty("contact_phone", &args.contact_phone)?;
    validate_phone(&args.contact_phone)?;

    let date = validate_date(&args.date)?;
    let time = validate_time(&args.time)?;

    let today = Local::now().date_naive();
    if date < today {
        return Err("Date cannot be in the past".to_string());
    }

    if !SLOT_TIMES.contains(&args.time.as_str()) {
        return Err("Time must match an available 30-minute slot".to_string());
    }

    let mut clinic = state.clinic.lock().await;
    let provider_name = clinic
        .providers
        .get(&args.provider_id)
        .ok_or_else(|| "Unknown provider_id".to_string())?
        .name
        .clone();

    if clinic.is_slot_booked(&args.provider_id, &args.date, &args.time) {
        return Err("Slot is already booked".to_string());
    }

    let appointment_id = Uuid::new_v4().to_string();
    clinic.book_slot(&args.provider_id, &args.date, &args.time);

    let instructions = "Arrive 10 minutes early. Bring a photo ID and insurance card."
        .to_string();

    info!(
        appointment_id = %appointment_id,
        provider_id = %args.provider_id,
        date = %args.date,
        time = %args.time,
        "appointment scheduled"
    );

    Ok(AppointmentConfirmation {
        appointment_id,
        provider_id: args.provider_id,
        provider_name,
        date: args.date,
        time: time.format("%H:%M").to_string(),
        status: "scheduled".to_string(),
        instructions,
    })
}

fn parse_params<T: for<'de> Deserialize<'de>>(params: Option<Value>) -> Result<T, String> {
    let params = params.unwrap_or_else(|| json!({}));
    serde_json::from_value(params).map_err(|err| format!("Invalid params: {err}"))
}

fn parse_args<T: for<'de> Deserialize<'de>>(args: Value) -> Result<T, String> {
    serde_json::from_value(args).map_err(|err| format!("Invalid arguments: {err}"))
}

fn validate_date(date: &str) -> Result<NaiveDate, String> {
    NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .map_err(|_| "Date must be in YYYY-MM-DD format".to_string())
}

fn validate_time(time: &str) -> Result<NaiveTime, String> {
    NaiveTime::parse_from_str(time, "%H:%M")
        .map_err(|_| "Time must be in HH:MM 24-hour format".to_string())
}

fn validate_phone(phone: &str) -> Result<(), String> {
    let digits = phone.chars().filter(|c| c.is_ascii_digit()).count();
    if digits < 7 {
        return Err("Contact phone must include at least 7 digits".to_string());
    }
    Ok(())
}

fn require_nonempty(field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} is required"));
    }
    Ok(())
}

fn build_tools() -> Result<Vec<RpcTool>, Box<dyn std::error::Error + Send + Sync>> {
    let availability_schema = serde_json::to_value(&schemars::schema_for!(GetAvailableSlotsArgs))?;
    let availability_out = serde_json::to_value(&schemars::schema_for!(AvailabilityResult))?;
    let schedule_schema = serde_json::to_value(&schemars::schema_for!(ScheduleAppointmentArgs))?;
    let schedule_out = serde_json::to_value(&schemars::schema_for!(AppointmentConfirmation))?;

    Ok(vec![
        RpcTool {
            name: "get_available_slots".to_string(),
            description: Some("List available appointment slots for a provider on a date".to_string()),
            input_schema: availability_schema,
            output_schema: Some(availability_out),
            meta: None,
            annotations: None,
        },
        RpcTool {
            name: "schedule_appointment".to_string(),
            description: Some("Schedule an appointment for a patient".to_string()),
            input_schema: schedule_schema,
            output_schema: Some(schedule_out),
            meta: None,
            annotations: None,
        },
    ])
}

fn rpc_ok(id: Value, result: Value) -> Json<Value> {
    Json(json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    }))
}

fn rpc_error(id: Value, code: i64, message: impl Into<String>) -> Json<Value> {
    Json(json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into(),
        }
    }))
}
