mod audio;
mod config;
mod http;
mod protocol;
mod services;
mod session;
mod text_filter;

use std::path::Path;

use anyhow::{Context, bail};
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
    if !vad_enabled_from_env() {
        return Ok(());
    }

    let dylib_path = std::env::var("ORT_DYLIB_PATH")
        .ok()
        .filter(|path| !path.is_empty())
        .or_else(|| option_env!("ORT_DYLIB_PATH").map(ToOwned::to_owned))
        .context(
            "ORT_DYLIB_PATH is not set; run through nix-shell/direnv so shell.nix can provide onnxruntime",
        )?;

    if !Path::new(&dylib_path).exists() {
        bail!("ORT_DYLIB_PATH does not exist: {dylib_path}");
    }

    ort::init_from(&dylib_path)
        .commit()
        .context("load ONNX Runtime dynamic library")?;
    tracing::info!(%dylib_path, "ONNX Runtime initialized");
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

fn install_rustls_crypto_provider() {
    // tokio-tungstenite uses rustls for Volcengine's wss:// endpoint. With
    // rustls 0.23 the process must pick a crypto provider before any TLS config
    // is built; otherwise the first outbound TLS connection can panic.
    let _ = rustls::crypto::ring::default_provider().install_default();
}
