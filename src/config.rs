use std::{env, net::SocketAddr};

use anyhow::{Context, Result};

#[derive(Clone, Debug)]
pub struct Config {
    pub bind: SocketAddr,
    pub public_ws_url: String,
    pub token: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let bind = env::var("XIAOZHI_BIND")
            .unwrap_or_else(|_| "0.0.0.0:8080".to_string())
            .parse::<SocketAddr>()
            .context("invalid XIAOZHI_BIND, expected host:port")?;

        let public_ws_url = env::var("XIAOZHI_PUBLIC_WS_URL")
            .unwrap_or_else(|_| format!("ws://127.0.0.1:{}/ws", bind.port()));
        let token = env::var("XIAOZHI_TOKEN").unwrap_or_else(|_| "dev-token".to_string());

        Ok(Self {
            bind,
            public_ws_url,
            token,
        })
    }
}
