#[cfg(any(feature = "telemetry-otel", feature = "telemetry-datadog"))]
use thiserror::Error;

#[cfg(any(feature = "telemetry-otel", feature = "telemetry-datadog"))]
use opentelemetry_sdk::trace::SdkTracerProvider;

#[cfg(any(feature = "telemetry-otel", feature = "telemetry-datadog"))]
#[derive(Debug, Error)]
pub enum TelemetryError {
    #[error("opentelemetry error: {0}")]
    OTel(String),
    #[error("subscriber error: {0}")]
    Subscriber(String),
}

#[cfg(any(feature = "telemetry-otel", feature = "telemetry-datadog"))]
#[derive(Debug)]
pub struct TelemetryGuard {
    provider: SdkTracerProvider,
}

#[cfg(any(feature = "telemetry-otel", feature = "telemetry-datadog"))]
impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        let _ = self.provider.shutdown();
    }
}

#[cfg(feature = "telemetry-otel")]
pub fn init_otlp_tracing(
    service_name: &str,
    protocol: opentelemetry_otlp::Protocol,
    endpoint: Option<&str>,
    env_filter: Option<&str>,
) -> Result<TelemetryGuard, TelemetryError> {
    use opentelemetry::global;
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig};
    use opentelemetry_sdk::Resource;
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let exporter = match protocol {
        Protocol::Grpc => {
            let mut builder = SpanExporter::builder().with_tonic();
            if let Some(endpoint) = endpoint {
                builder = builder.with_endpoint(endpoint.to_string());
            }
            builder
                .build()
                .map_err(|e| TelemetryError::OTel(e.to_string()))?
        }
        Protocol::HttpBinary | Protocol::HttpJson => {
            let mut builder = SpanExporter::builder().with_http().with_protocol(protocol);
            if let Some(endpoint) = endpoint {
                builder = builder.with_endpoint(endpoint.to_string());
            }
            builder
                .build()
                .map_err(|e| TelemetryError::OTel(e.to_string()))?
        }
    };

    let tracer_provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            Resource::builder()
                .with_service_name(service_name.to_string())
                .build(),
        )
        .build();

    global::set_tracer_provider(tracer_provider.clone());
    let tracer = tracer_provider.tracer("rustic-ai");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    let filter = match env_filter {
        Some(filter) => EnvFilter::new(filter),
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .with(otel_layer)
        .try_init()
        .map_err(|e| TelemetryError::Subscriber(e.to_string()))?;

    Ok(TelemetryGuard {
        provider: tracer_provider,
    })
}

#[cfg(feature = "telemetry-datadog")]
pub fn init_datadog_tracing(
    service_name: &str,
    agent_endpoint: Option<&str>,
    env_filter: Option<&str>,
) -> Result<TelemetryGuard, TelemetryError> {
    use opentelemetry::global;
    use opentelemetry::trace::TracerProvider as _;
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let mut pipeline = opentelemetry_datadog::new_pipeline().with_service_name(service_name);
    if let Some(endpoint) = agent_endpoint {
        pipeline = pipeline.with_agent_endpoint(endpoint);
    }

    let tracer_provider = pipeline
        .install_batch()
        .map_err(|e| TelemetryError::OTel(e.to_string()))?;

    global::set_tracer_provider(tracer_provider.clone());
    let tracer = tracer_provider.tracer("rustic-ai");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    let filter = match env_filter {
        Some(filter) => EnvFilter::new(filter),
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
    };

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .with(otel_layer)
        .try_init()
        .map_err(|e| TelemetryError::Subscriber(e.to_string()))?;

    Ok(TelemetryGuard {
        provider: tracer_provider,
    })
}
