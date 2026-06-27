mod audio;
mod config;
mod http;
mod protocol;
mod services;
mod session;

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

    let config = Config::from_env().context("load config")?;
    http::serve(config).await
}

fn install_rustls_crypto_provider() {
    // tokio-tungstenite uses rustls for Volcengine's wss:// endpoint. With
    // rustls 0.23 the process must pick a crypto provider before any TLS config
    // is built; otherwise the first outbound TLS connection can panic.
    let _ = rustls::crypto::ring::default_provider().install_default();
}
