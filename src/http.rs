use std::{
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::Context;
use axum::{
    Json, Router,
    body::Body,
    extract::{Request, State, WebSocketUpgrade, ws::WebSocket},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;

use crate::{
    config::Config,
    protocol::BinaryProtocolVersion,
    session::{self, SessionMeta},
};

#[derive(Clone)]
pub struct AppState {
    config: Arc<Config>,
}

pub async fn serve(config: Config) -> anyhow::Result<()> {
    let bind = config.bind;
    let state = AppState {
        config: Arc::new(config),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/ota", get(ota).post(ota))
        .route("/ota/activate", post(activate))
        .route("/ws", get(ws_handler))
        .with_state(state)
        .layer(middleware::from_fn(log_request_response))
        .layer(TraceLayer::new_for_http());

    let listener = TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind {bind}"))?;
    tracing::info!(%bind, "xiaozhi server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("serve http")
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::warn!(%err, "failed to listen for shutdown signal");
    }
}

async fn log_request_response(request: Request<Body>, next: Next) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let version = request.version();
    let user_agent = request
        .headers()
        .get("User-Agent")
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);
    let device_id = request
        .headers()
        .get("Device-Id")
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);
    let client_id = request
        .headers()
        .get("Client-Id")
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);

    tracing::debug!(
        %method,
        %uri,
        ?version,
        user_agent = ?user_agent,
        device_id = ?device_id,
        client_id = ?client_id,
        "http request started"
    );

    let started = Instant::now();
    let response = next.run(request).await;
    let status = response.status();
    let elapsed_ms = started.elapsed().as_millis();

    tracing::debug!(
        %method,
        %uri,
        ?version,
        %status,
        elapsed_ms,
        "http request finished"
    );

    response
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn ota(State(state): State<AppState>) -> Json<Value> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default();

    Json(json!({
        "firmware": {
            "version": "0.0.0",
            "force": 0
        },
        "websocket": {
            "url": state.config.public_ws_url,
            "token": state.config.token,
            "version": 1
        },
        "server_time": {
            "timestamp": timestamp,
            "timezone_offset": 480
        }
    }))
}

async fn activate() -> StatusCode {
    StatusCode::OK
}

async fn ws_handler(
    State(state): State<AppState>,
    ws: WebSocketUpgrade,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !is_authorized(&state.config, &headers) {
        tracing::warn!("rejecting websocket connection: invalid authorization header");
        return (StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    let protocol_version = BinaryProtocolVersion::from_header(
        headers
            .get("Protocol-Version")
            .and_then(|value| value.to_str().ok()),
    );
    let meta = SessionMeta {
        device_id: header_string(&headers, "Device-Id"),
        client_id: header_string(&headers, "Client-Id"),
    };

    tracing::debug!(
        version = ?protocol_version,
        device_id = ?meta.device_id,
        client_id = ?meta.client_id,
        "websocket upgrade accepted"
    );

    ws.on_upgrade(move |socket: WebSocket| {
        session::handle_websocket(socket, protocol_version, meta)
    })
    .into_response()
}

fn is_authorized(config: &Config, headers: &HeaderMap) -> bool {
    if config.token.is_empty() {
        return true;
    }

    let Some(header) = headers.get("Authorization").and_then(|v| v.to_str().ok()) else {
        // The initial development server intentionally allows missing auth so
        // boards can be debugged before OTA/NVS is fully configured.
        return true;
    };

    let header = header.trim();
    header == config.token || header == format!("Bearer {}", config.token)
}

fn header_string(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}
