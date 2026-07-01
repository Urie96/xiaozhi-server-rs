mod audio;
mod config;
mod http;
mod protocol;
mod services;
mod session;
mod speaker_id;
mod text_filter;

use anyhow::Context;
use config::Config;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    install_rustls_crypto_provider();

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "xiaozhi_server_rs=debug,tower_http=info,axum=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    init_onnx_runtime().context("initialize ONNX Runtime")?;

    let config = Config::from_env().context("load config")?;
    http::serve(config).await
}

fn init_onnx_runtime() -> anyhow::Result<()> {
    if !vad_enabled_from_env() && !speaker_id_enabled_from_env() {
        return Ok(());
    }

    ort::init()
        .commit()
        .context("load ONNX Runtime dynamic library; make sure libonnxruntime.so is on LD_LIBRARY_PATH")?;
    tracing::info!("ONNX Runtime initialized");
    Ok(())
}

fn vad_enabled_from_env() -> bool {
    let provider = std::env::var("XIAOZHI_VAD_PROVIDER")
        .unwrap_or_else(|_| "silero".to_string())
        .to_ascii_lowercase();
    !matches!(
        provider.as_str(),
        "none" | "off" | "disabled" | "false" | "0"
    )
}

fn speaker_id_enabled_from_env() -> bool {
    let provider = std::env::var("XIAOZHI_SPEAKER_PROVIDER")
        .or_else(|_| std::env::var("SPEAKER_PROVIDER"))
        .unwrap_or_else(|_| "none".to_string())
        .to_ascii_lowercase();
    !matches!(
        provider.as_str(),
        "none" | "off" | "disabled" | "false" | "0"
    )
}
fn install_rustls_crypto_provider() {
    // tokio-tungstenite uses rustls for Volcengine's wss:// endpoint. With
    // rustls 0.23 the process must pick a crypto provider before any TLS config
    // is built; otherwise the first outbound TLS connection can panic.
    let _ = rustls::crypto::ring::default_provider().install_default();
}
