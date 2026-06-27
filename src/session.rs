use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::{
    sync::{Mutex, mpsc},
    task::JoinHandle,
};
use uuid::Uuid;

use crate::{
    protocol::{self, AudioFrame, BinaryProtocolVersion, decode_audio_frame, encode_audio_frame},
    services::{ServiceBundle, TtsEvent},
};

const MOCK_ASR_TRIGGER_FRAMES: usize = 20;

#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub device_id: Option<String>,
    pub client_id: Option<String>,
}

#[derive(Debug)]
enum Outbound {
    Text(Value),
    Binary(Bytes),
    Pong(Bytes),
}

#[derive(Debug, Default)]
struct SessionState {
    listening: bool,
    already_triggered: bool,
    listen_mode: Option<String>,
    frames: Vec<AudioFrame>,
    pipeline: Option<JoinHandle<()>>,
}

#[derive(Clone)]
struct SessionContext {
    id: String,
    version: Arc<Mutex<BinaryProtocolVersion>>,
    tx: mpsc::Sender<Outbound>,
    services: ServiceBundle,
}

pub async fn handle_websocket(
    socket: WebSocket,
    protocol_version: BinaryProtocolVersion,
    meta: SessionMeta,
    services: ServiceBundle,
) {
    let session_id = Uuid::new_v4().to_string();
    tracing::info!(
        session_id,
        device_id = ?meta.device_id,
        client_id = ?meta.client_id,
        version = ?protocol_version,
        "websocket session connected"
    );

    let (mut ws_tx, mut ws_rx) = socket.split();
    let (out_tx, mut out_rx) = mpsc::channel::<Outbound>(128);

    let writer_session_id = session_id.clone();
    let writer = tokio::spawn(async move {
        while let Some(outbound) = out_rx.recv().await {
            let message = match outbound {
                Outbound::Text(value) => {
                    let text = value.to_string();
                    tracing::debug!(session_id = writer_session_id, %text, "websocket send text");
                    Message::Text(text.into())
                }
                Outbound::Binary(bytes) => Message::Binary(bytes),
                Outbound::Pong(bytes) => {
                    tracing::debug!(
                        session_id = writer_session_id,
                        bytes = bytes.len(),
                        "websocket send pong"
                    );
                    Message::Pong(bytes)
                }
            };

            if let Err(err) = ws_tx.send(message).await {
                tracing::debug!(%err, "websocket writer stopped");
                break;
            }
        }
    });

    let ctx = SessionContext {
        id: session_id.clone(),
        version: Arc::new(Mutex::new(protocol_version)),
        tx: out_tx.clone(),
        services,
    };
    let state = Arc::new(Mutex::new(SessionState::default()));

    while let Some(message) = ws_rx.next().await {
        match message {
            Ok(Message::Text(text)) => {
                tracing::debug!(session_id, %text, "websocket recv text");
                if !handle_text(text.to_string(), &ctx, &state).await {
                    break;
                }
            }
            Ok(Message::Binary(data)) => {
                handle_binary(&data, &ctx, &state).await;
            }
            Ok(Message::Ping(data)) => {
                tracing::debug!(session_id, bytes = data.len(), "websocket recv ping");
                let _ = out_tx.send(Outbound::Pong(data)).await;
            }
            Ok(Message::Pong(data)) => {
                tracing::debug!(session_id, bytes = data.len(), "websocket recv pong");
            }
            Ok(Message::Close(frame)) => {
                tracing::info!(session_id, ?frame, "websocket closed by client");
                break;
            }
            Err(err) => {
                tracing::debug!(session_id, %err, "websocket receive error");
                break;
            }
        }
    }

    abort_pipeline(&ctx, &state, false).await;
    drop(out_tx);
    if let Err(err) = writer.await {
        tracing::debug!(session_id, %err, "websocket writer join error");
    }
    tracing::info!(session_id, "websocket session disconnected");
}

async fn handle_text(text: String, ctx: &SessionContext, state: &Arc<Mutex<SessionState>>) -> bool {
    let incoming = match serde_json::from_str::<protocol::IncomingJson>(&text) {
        Ok(incoming) => incoming,
        Err(err) => {
            tracing::warn!(session_id = ctx.id, %err, text, "invalid json from client");
            return true;
        }
    };

    if let Some(incoming_session_id) = incoming.session_id.as_deref() {
        if incoming_session_id != ctx.id {
            tracing::debug!(
                session_id = ctx.id,
                incoming_session_id,
                "ignoring mismatched incoming session_id"
            );
        }
    }

    match incoming.typ.as_str() {
        "hello" => {
            let mut version = ctx.version.lock().await;
            *version = BinaryProtocolVersion::from_hello(incoming.version, *version);
            tracing::info!(
                session_id = ctx.id,
                version = ?*version,
                client_hello = %incoming.extra,
                "client hello received"
            );
            send_json(ctx, protocol::hello(&ctx.id)).await;
        }
        "listen" => match incoming.state() {
            Some("start") => {
                abort_pipeline(ctx, state, true).await;
                let mut guard = state.lock().await;
                guard.listening = true;
                guard.already_triggered = false;
                guard.listen_mode = incoming.mode().map(ToOwned::to_owned);
                guard.frames.clear();
                tracing::info!(
                    session_id = ctx.id,
                    mode = ?guard.listen_mode,
                    "listen started"
                );
            }
            Some("stop") => {
                let should_trigger = {
                    let mut guard = state.lock().await;
                    guard.listening = false;
                    !guard.already_triggered
                };
                tracing::info!(session_id = ctx.id, "listen stopped");
                if should_trigger {
                    trigger_conversation(ctx.clone(), state.clone()).await;
                }
            }
            Some("detect") => {
                let wake_word = incoming
                    .extra
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                tracing::info!(session_id = ctx.id, wake_word, "wake word detected");
                abort_pipeline(ctx, state, true).await;
            }
            other => tracing::debug!(session_id = ctx.id, ?other, "unknown listen state"),
        },
        "abort" => {
            let reason = incoming
                .extra
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("manual");
            tracing::info!(session_id = ctx.id, reason, "abort requested");
            abort_pipeline(ctx, state, true).await;
        }
        "mcp" => tracing::debug!(session_id = ctx.id, "mcp message ignored in v1"),
        "goodbye" => return false,
        other => tracing::debug!(session_id = ctx.id, typ = other, "ignored json message"),
    }

    true
}

async fn handle_binary(data: &[u8], ctx: &SessionContext, state: &Arc<Mutex<SessionState>>) {
    let version = *ctx.version.lock().await;
    let frame = match decode_audio_frame(version, data) {
        Ok(Some(frame)) => frame,
        Ok(None) => return,
        Err(err) => {
            tracing::warn!(session_id = ctx.id, %err, "invalid binary audio frame");
            return;
        }
    };

    let should_trigger = {
        let mut guard = state.lock().await;
        if !guard.listening {
            tracing::trace!(session_id = ctx.id, "discarding audio while not listening");
            return;
        }

        guard.frames.push(frame);
        !guard.already_triggered && guard.frames.len() >= MOCK_ASR_TRIGGER_FRAMES
    };

    if should_trigger {
        trigger_conversation(ctx.clone(), state.clone()).await;
    }
}

async fn trigger_conversation(ctx: SessionContext, state: Arc<Mutex<SessionState>>) {
    let frames = {
        let mut guard = state.lock().await;
        if guard.already_triggered {
            return;
        }
        guard.already_triggered = true;
        guard.listening = false;
        guard.frames.clone()
    };

    tracing::info!(
        session_id = ctx.id,
        frames = frames.len(),
        "starting mock conversation pipeline"
    );
    let pipeline_ctx = ctx.clone();
    let handle = tokio::spawn(async move {
        if let Err(err) = run_pipeline(pipeline_ctx.clone(), frames).await {
            tracing::warn!(session_id = pipeline_ctx.id, %err, "conversation pipeline failed");
            send_json(&pipeline_ctx, protocol::tts_stop(&pipeline_ctx.id)).await;
        }
    });

    let mut guard = state.lock().await;
    if let Some(old) = guard.pipeline.replace(handle) {
        old.abort();
    }
}

async fn run_pipeline(ctx: SessionContext, frames: Vec<AudioFrame>) -> anyhow::Result<()> {
    let text = ctx.services.asr.recognize(&frames).await?;
    send_json(&ctx, protocol::stt(&ctx.id, &text)).await;
    send_json(&ctx, protocol::llm_emotion(&ctx.id, "happy", "😊")).await;
    send_json(&ctx, protocol::tts_start(&ctx.id)).await;

    let llm_stream = ctx.services.llm.chat_stream(text);
    let mut tts_stream = ctx.services.tts.synthesize_stream(llm_stream);

    while let Some(event) = tts_stream.next().await {
        match event? {
            TtsEvent::SentenceStart(text) => {
                send_json(&ctx, protocol::tts_sentence_start(&ctx.id, &text)).await;
            }
            TtsEvent::Audio(frame) => {
                let version = *ctx.version.lock().await;
                let bytes = encode_audio_frame(version, &frame);
                if ctx.tx.send(Outbound::Binary(bytes)).await.is_err() {
                    break;
                }
            }
        }
    }

    send_json(&ctx, protocol::tts_stop(&ctx.id)).await;
    Ok(())
}

async fn abort_pipeline(ctx: &SessionContext, state: &Arc<Mutex<SessionState>>, send_stop: bool) {
    let handle = {
        let mut guard = state.lock().await;
        guard.pipeline.take()
    };

    if let Some(handle) = handle {
        if handle.is_finished() {
            return;
        }

        handle.abort();
        tracing::info!(session_id = ctx.id, "conversation pipeline aborted");
        if send_stop {
            send_json(ctx, protocol::tts_stop(&ctx.id)).await;
        }
    }
}

async fn send_json(ctx: &SessionContext, value: Value) {
    if ctx.tx.send(Outbound::Text(value)).await.is_err() {
        tracing::debug!(
            session_id = ctx.id,
            "failed to enqueue json: websocket closed"
        );
    }
}
