use std::{sync::Arc, time::Instant};

use async_stream::try_stream;
use axum::extract::ws::{Message, WebSocket};
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::{
    sync::{Mutex, mpsc},
    task::JoinHandle,
    time::{Duration, sleep},
};
use uuid::Uuid;

use crate::{
    protocol::{self, BinaryProtocolVersion, decode_audio_frame, encode_audio_frame},
    services::{AsrStream, ServiceBundle, TextStream, TtsEvent},
};

const TEMP_LISTEN_TIMEOUT: Duration = Duration::from_secs(5);

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

#[derive(Default)]
struct SessionState {
    listening: bool,
    already_triggered: bool,
    listen_mode: Option<String>,
    asr: Option<Box<dyn AsrStream>>,
    listen_timeout: Option<JoinHandle<()>>,
    pipeline: Option<JoinHandle<()>>,
    listen_started_at: Option<Instant>,
    first_audio_logged: bool,
    input_frames: u64,
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

    abort_active(&ctx, &state, false).await;
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
                abort_active(ctx, state, true).await;
                let asr_start = Instant::now();
                tracing::debug!(session_id = ctx.id, "starting asr stream");
                match ctx.services.asr.start_stream().await {
                    Ok(asr) => {
                        tracing::debug!(
                            session_id = ctx.id,
                            elapsed_ms = asr_start.elapsed().as_millis(),
                            "asr stream started"
                        );
                        let timeout_ctx = ctx.clone();
                        let timeout_state = state.clone();
                        let listen_timeout = tokio::spawn(async move {
                            sleep(TEMP_LISTEN_TIMEOUT).await;
                            finish_listening(
                                timeout_ctx,
                                timeout_state,
                                "temporary listen timeout",
                                false,
                            )
                            .await;
                        });

                        let mut guard = state.lock().await;
                        guard.listening = true;
                        guard.already_triggered = false;
                        guard.listen_mode = incoming.mode().map(ToOwned::to_owned);
                        guard.asr = Some(asr);
                        guard.listen_started_at = Some(Instant::now());
                        guard.first_audio_logged = false;
                        guard.input_frames = 0;
                        if let Some(old) = guard.listen_timeout.replace(listen_timeout) {
                            old.abort();
                        }
                        tracing::info!(
                            session_id = ctx.id,
                            mode = ?guard.listen_mode,
                            timeout_ms = TEMP_LISTEN_TIMEOUT.as_millis(),
                            "listen started"
                        );
                    }
                    Err(err) => {
                        let error_chain = format_error_chain(err.as_ref());
                        tracing::warn!(
                            session_id = ctx.id,
                            error_chain,
                            "failed to start asr stream"
                        );
                    }
                }
            }
            Some("stop") => {
                finish_listening(ctx.clone(), state.clone(), "listen stopped", true).await;
            }
            Some("detect") => {
                let wake_word = incoming
                    .extra
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                tracing::info!(session_id = ctx.id, wake_word, "wake word detected");
                abort_active(ctx, state, true).await;
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
            abort_active(ctx, state, true).await;
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

    let mut guard = state.lock().await;
    if !guard.listening {
        tracing::trace!(session_id = ctx.id, "discarding audio while not listening");
        return;
    }

    guard.input_frames = guard.input_frames.saturating_add(1);
    if !guard.first_audio_logged {
        guard.first_audio_logged = true;
        tracing::debug!(
            session_id = ctx.id,
            elapsed_ms = guard
                .listen_started_at
                .map(|started| started.elapsed().as_millis()),
            "first input audio frame received"
        );
    }

    if let Some(asr) = guard.asr.as_mut() {
        if let Err(err) = asr.push_audio(frame).await {
            let error_chain = format_error_chain(err.as_ref());
            tracing::warn!(
                session_id = ctx.id,
                error_chain,
                "failed to push audio to asr stream"
            );
        }
    }
}

async fn finish_listening(
    ctx: SessionContext,
    state: Arc<Mutex<SessionState>>,
    reason: &'static str,
    abort_timeout: bool,
) {
    let (asr, timeout, listen_elapsed_ms, input_frames) = {
        let mut guard = state.lock().await;
        guard.listening = false;
        let listen_elapsed_ms = guard
            .listen_started_at
            .map(|started| started.elapsed().as_millis());
        let input_frames = guard.input_frames;
        if guard.already_triggered {
            (None, None, listen_elapsed_ms, input_frames)
        } else {
            guard.already_triggered = true;
            guard.listen_started_at = None;
            guard.first_audio_logged = false;
            guard.input_frames = 0;
            (
                guard.asr.take(),
                if abort_timeout {
                    guard.listen_timeout.take()
                } else {
                    None
                },
                listen_elapsed_ms,
                input_frames,
            )
        }
    };

    if let Some(timeout) = timeout {
        timeout.abort();
    }

    tracing::info!(
        session_id = ctx.id,
        reason,
        listen_elapsed_ms,
        input_frames,
        "listen finished"
    );
    if let Some(asr) = asr {
        trigger_conversation(ctx, state, asr).await;
    }
}

async fn trigger_conversation(
    ctx: SessionContext,
    state: Arc<Mutex<SessionState>>,
    asr: Box<dyn AsrStream>,
) {
    tracing::info!(session_id = ctx.id, "starting conversation pipeline");
    let pipeline_ctx = ctx.clone();
    let handle = tokio::spawn(async move {
        if let Err(err) = run_pipeline(pipeline_ctx.clone(), asr).await {
            let error_chain = format_error_chain(err.as_ref());
            tracing::warn!(
                session_id = pipeline_ctx.id,
                error_chain,
                "conversation pipeline failed"
            );
            send_json(&pipeline_ctx, protocol::tts_stop(&pipeline_ctx.id)).await;
        }
    });

    let mut guard = state.lock().await;
    if let Some(old) = guard.pipeline.replace(handle) {
        old.abort();
    }
}

async fn run_pipeline(ctx: SessionContext, mut asr: Box<dyn AsrStream>) -> anyhow::Result<()> {
    let pipeline_started = Instant::now();
    tracing::debug!(session_id = ctx.id, "pipeline timing started");

    let asr_finish_started = Instant::now();
    let text = asr.finish().await?;
    tracing::debug!(
        session_id = ctx.id,
        elapsed_ms = asr_finish_started.elapsed().as_millis(),
        total_elapsed_ms = pipeline_started.elapsed().as_millis(),
        text_chars = text.chars().count(),
        "asr finish returned final text"
    );

    send_json(&ctx, protocol::stt(&ctx.id, &text)).await;
    send_json(&ctx, protocol::llm_emotion(&ctx.id, "happy", "😊")).await;
    send_json(&ctx, protocol::tts_start(&ctx.id)).await;
    tracing::debug!(
        session_id = ctx.id,
        total_elapsed_ms = pipeline_started.elapsed().as_millis(),
        "stt/llm-emotion/tts-start events enqueued"
    );

    let llm_started = Instant::now();
    let raw_llm_stream = ctx.services.llm.chat_stream(text);
    let llm_stream = instrument_llm_stream(
        ctx.id.clone(),
        pipeline_started,
        llm_started,
        raw_llm_stream,
    );

    let tts_started = Instant::now();
    let mut tts_stream = ctx.services.tts.synthesize_stream(llm_stream);
    tracing::debug!(
        session_id = ctx.id,
        total_elapsed_ms = pipeline_started.elapsed().as_millis(),
        "tts stream created"
    );

    let mut sentence_events = 0u64;
    let mut audio_frames = 0u64;
    let mut first_sentence_logged = false;
    let mut first_audio_logged = false;

    while let Some(event) = tts_stream.next().await {
        match event? {
            TtsEvent::SentenceStart(text) => {
                sentence_events = sentence_events.saturating_add(1);
                if !first_sentence_logged {
                    first_sentence_logged = true;
                    tracing::debug!(
                        session_id = ctx.id,
                        tts_elapsed_ms = tts_started.elapsed().as_millis(),
                        total_elapsed_ms = pipeline_started.elapsed().as_millis(),
                        text_chars = text.chars().count(),
                        "first tts sentence_start"
                    );
                }
                send_json(&ctx, protocol::tts_sentence_start(&ctx.id, &text)).await;
            }
            TtsEvent::Audio(frame) => {
                audio_frames = audio_frames.saturating_add(1);
                if !first_audio_logged {
                    first_audio_logged = true;
                    tracing::debug!(
                        session_id = ctx.id,
                        tts_elapsed_ms = tts_started.elapsed().as_millis(),
                        total_elapsed_ms = pipeline_started.elapsed().as_millis(),
                        frame_bytes = frame.payload.len(),
                        "first tts audio frame ready"
                    );
                }
                let version = *ctx.version.lock().await;
                let bytes = encode_audio_frame(version, &frame);
                if ctx.tx.send(Outbound::Binary(bytes)).await.is_err() {
                    tracing::debug!(
                        session_id = ctx.id,
                        audio_frames,
                        "websocket outbound queue closed while sending tts audio"
                    );
                    break;
                }
            }
        }
    }

    tracing::debug!(
        session_id = ctx.id,
        total_elapsed_ms = pipeline_started.elapsed().as_millis(),
        sentence_events,
        audio_frames,
        "tts stream finished"
    );
    send_json(&ctx, protocol::tts_stop(&ctx.id)).await;
    tracing::debug!(
        session_id = ctx.id,
        total_elapsed_ms = pipeline_started.elapsed().as_millis(),
        "pipeline finished"
    );
    Ok(())
}

fn instrument_llm_stream(
    session_id: String,
    pipeline_started: Instant,
    llm_started: Instant,
    mut input: TextStream,
) -> TextStream {
    Box::pin(try_stream! {
        let mut chunks = 0u64;
        let mut chars = 0usize;
        let mut first_chunk_logged = false;

        while let Some(chunk) = input.next().await {
            let chunk = chunk?;
            let chunk_chars = chunk.chars().count();
            chunks = chunks.saturating_add(1);
            chars = chars.saturating_add(chunk_chars);

            if !first_chunk_logged {
                first_chunk_logged = true;
                tracing::debug!(
                    session_id,
                    llm_elapsed_ms = llm_started.elapsed().as_millis(),
                    total_elapsed_ms = pipeline_started.elapsed().as_millis(),
                    chunk_chars,
                    "first llm text chunk"
                );
            }

            yield chunk;
        }

        tracing::debug!(
            session_id,
            llm_elapsed_ms = llm_started.elapsed().as_millis(),
            total_elapsed_ms = pipeline_started.elapsed().as_millis(),
            chunks,
            chars,
            "llm stream finished"
        );
    })
}

async fn abort_active(ctx: &SessionContext, state: &Arc<Mutex<SessionState>>, send_stop: bool) {
    let (pipeline, mut asr, listen_timeout) = {
        let mut guard = state.lock().await;
        guard.listening = false;
        guard.already_triggered = false;
        (
            guard.pipeline.take(),
            guard.asr.take(),
            guard.listen_timeout.take(),
        )
    };

    if let Some(listen_timeout) = listen_timeout {
        listen_timeout.abort();
    }

    if let Some(handle) = pipeline {
        if !handle.is_finished() {
            handle.abort();
            tracing::info!(session_id = ctx.id, "conversation pipeline aborted");
            if send_stop {
                send_json(ctx, protocol::tts_stop(&ctx.id)).await;
            }
        }
    }

    if let Some(asr) = asr.as_mut() {
        asr.abort().await;
        tracing::info!(session_id = ctx.id, "asr stream aborted");
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

fn format_error_chain(err: &(dyn std::error::Error + 'static)) -> String {
    let mut out = err.to_string();
    let mut source = err.source();
    while let Some(err) = source {
        out.push_str(": ");
        out.push_str(&err.to_string());
        source = err.source();
    }
    out
}
