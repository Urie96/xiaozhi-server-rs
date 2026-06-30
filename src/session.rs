use std::{
    collections::VecDeque,
    sync::{Arc, Mutex as StdMutex},
    time::Instant,
};

use async_stream::try_stream;
use axum::extract::ws::{Message, WebSocket};
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::{
    sync::{Mutex, Notify, mpsc, oneshot},
    task::JoinHandle,
    time::{Duration, sleep, timeout},
};
use uuid::Uuid;

use crate::{
    audio::{
        opus_decode::OpusPcmDecoder,
        silero_vad::{SileroVadStream, VadEvent},
    },
    protocol::{self, BinaryProtocolVersion, decode_audio_frame, encode_audio_frame},
    services::{AsrStream, LlmSession, LlmSessionMeta, ServiceBundle, TextStream, TtsEvent},
    text_filter::filter_tts_text_stream,
};

const CLIENT_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_LISTEN_MAX_TIMEOUT: Duration = Duration::from_secs(120);

/// Default idle window before the server simulates a "goodbye" prompt
/// for the user. Configurable via `XIAOZHI_IDLE_CLOSE_SECONDS`.
const DEFAULT_IDLE_CLOSE_SECONDS: u64 = 90;

/// After the TTS stream for an exit-intent round has fully drained, wait
/// this long before closing the WebSocket so the final Opus frame has
/// time to reach the device and finish playing. The 240 ms headroom
/// covers one 60 ms Opus packet of network jitter plus the prebuffer
/// pacing the TTS service already introduced.
const POST_TTS_FLUSH_DELAY: Duration = Duration::from_millis(240);

/// How long the server waits for a Pong response after sending a
/// WebSocket Ping during idle-timeout probing.
const WEBSOCKET_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Default exit-command keywords. Matched as exact, trimmed equality
/// against the final ASR text. Configurable via `XIAOZHI_EXIT_COMMANDS`
/// (comma-separated).
const DEFAULT_EXIT_COMMANDS: &[&str] = &["退出", "关闭"];

/// Default end-of-conversation prompt fed to the LLM when the idle
/// watchdog fires. Configurable via `XIAOZHI_END_PROMPT`.
const DEFAULT_END_PROMPT: &str =
    "请你以\"时间过得真快\"开头，用富有感情、依依不舍的话来结束这场对话吧！";

/// Fixed-size PCM ring buffer used as a pre-roll for ASR: it keeps the most
/// recent ~1 second of decoded PCM samples so that when VAD finally confirms
/// speech onset, ASR receives the audio leading up to (and including) the
/// trigger frame, instead of starting from the trigger point and missing the
/// first syllable.
///
/// 1 second @ 16 kHz mono = 16 000 samples of i16 (~32 KB).
const PCM_RING_BUFFER_SAMPLES: usize = 16_000;

#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub device_id: Option<String>,
    pub client_id: Option<String>,
}

#[derive(Debug)]
enum Outbound {
    Text(Value),
    Binary(Bytes),
    Ping(Bytes),
    Pong(Bytes),
    /// Server-initiated WebSocket close frame. The writer sends
    /// `Message::Close(None)` to the client and stops the send loop;
    /// this lets the device recognise that the server is hanging up
    /// (rather than just dropping the TCP connection), so it stops
    /// transitioning back into listening.
    Close,
}

#[derive(Default)]
struct SessionState {
    listening: bool,
    already_triggered: bool,
    listen_mode: Option<String>,
    asr: Option<Box<dyn AsrStream>>,
    audio_decoder: Option<OpusPcmDecoder>,
    vad: Option<SileroVadStream>,
    /// Holds the most recent PCM samples (FIFO, fixed capacity) until VAD
    /// triggers, so the ASR session can be primed with the leading audio.
    pcm_ring_buffer: Option<PcmRingBuffer>,
    listen_timeout: Option<JoinHandle<()>>,
    pipeline: Option<JoinHandle<()>>,
    listen_started_at: Option<Instant>,
    first_audio_logged: bool,
    input_frames: u64,
    /// Active WebSocket probe waiting for a matching Pong. Used by the
    /// idle watchdog to distinguish a live client from a dead one before
    /// it decides whether to synthesize a goodbye prompt.
    ws_probe: Option<WebSocketProbe>,
    /// Set when the user said something that signals end-of-conversation
    /// (matched against `XIAOZHI_EXIT_COMMANDS`) OR when the idle
    /// watchdog fired and queued a synthetic goodbye prompt. The next
    /// pipeline to finish its TTS stream will, after a brief flush delay,
    /// signal `handle_websocket` to close the WebSocket.
    close_after_chat: bool,
    /// Last moment any user audio was observed (`handle_binary`) or the
    /// user explicitly entered listening (`listen.start`). Used by the
    /// idle watchdog to decide when to fire a synthetic goodbye.
    last_voice_ts: Option<Instant>,
    /// Owns the LLM session for the lifetime of the WebSocket connection.
    ///
    /// Held behind `Arc<Mutex<...>>` so the in-flight `run_pipeline` task can
    /// borrow it (calling `chat_stream` synchronously under the lock) without
    /// moving it out of `SessionState`. This is critical: if the pipeline
    /// task is later notified to abort, it must NOT be holding the only
    /// reference to the LLM, because abort/shutdown must stay routed through
    /// the provider implementation. The lock is held only for the synchronous
    /// `chat_stream` call; the lock is never held across an `await`.
    llm: Arc<Mutex<Option<Box<dyn LlmSession>>>>,
    /// Notified by `abort_active` so the running `run_pipeline` task can
    /// observe the abort, emit a `tts.stop`, and exit cleanly. The pipeline
    /// task is never cancelled with `JoinHandle::abort`, because that
    /// would drop its stack-locals before the provider can cleanly abort
    /// its active request.
    abort_notify: Arc<Notify>,
}

/// Fixed-capacity FIFO ring buffer for `i16` PCM samples.
///
/// Internally backed by `VecDeque` for simple push/pop-front semantics. The
/// capacity is fixed; pushing past it evicts the oldest sample.
#[derive(Debug)]
struct PcmRingBuffer {
    storage: VecDeque<i16>,
    capacity: usize,
}

impl PcmRingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            storage: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, samples: &[i16]) {
        for &sample in samples {
            if self.storage.len() == self.capacity {
                self.storage.pop_front();
            }
            self.storage.push_back(sample);
        }
    }

    /// Return all currently-buffered samples in chronological order.
    fn drain(&self) -> Vec<i16> {
        self.storage.iter().copied().collect()
    }
}

#[derive(Clone)]
struct SessionContext {
    id: String,
    version: Arc<Mutex<BinaryProtocolVersion>>,
    tx: mpsc::Sender<Outbound>,
    services: ServiceBundle,
    /// Signals `handle_websocket` to break out of its main receive loop
    /// and tear down the connection. Used by the conversation pipeline
    /// once an exit-intent round has finished playing its goodbye audio.
    shutdown_tx: mpsc::Sender<()>,
}

struct WebSocketProbe {
    payload: Bytes,
    done_tx: oneshot::Sender<()>,
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

    let llm_meta = LlmSessionMeta {
        session_id: session_id.clone(),
        device_id: meta.device_id.clone(),
        client_id: meta.client_id.clone(),
    };
    let llm = match services.llm.create_session(llm_meta).await {
        Ok(llm) => {
            tracing::info!(session_id, "llm session created");
            Some(llm)
        }
        Err(err) => {
            let error_chain = format_error_chain(err.as_ref());
            tracing::warn!(session_id, error_chain, "failed to create llm session");
            None
        }
    };
    let llm_slot = Arc::new(Mutex::new(llm));

    let (mut ws_tx, mut ws_rx) = socket.split();
    let (out_tx, mut out_rx) = mpsc::channel::<Outbound>(128);
    let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);

    let writer_session_id = session_id.clone();
    let writer = tokio::spawn(async move {
        while let Some(outbound) = out_rx.recv().await {
            let message = match &outbound {
                Outbound::Text(value) => {
                    let text = value.to_string();
                    let is_error = serde_json::from_str::<Value>(&text)
                        .ok()
                        .and_then(|v| v.get("type").and_then(Value::as_str).map(str::to_string))
                        .as_deref()
                        == Some("error");
                    if is_error {
                        tracing::error!(
                            session_id = writer_session_id,
                            %text,
                            "websocket send error message"
                        );
                    } else {
                        tracing::debug!(
                            session_id = writer_session_id,
                            %text,
                            "websocket send text"
                        );
                    }
                    Message::Text(text.into())
                }
                Outbound::Binary(bytes) => Message::Binary(bytes.clone()),
                Outbound::Ping(bytes) => {
                    tracing::debug!(
                        session_id = writer_session_id,
                        bytes = bytes.len(),
                        "websocket send ping"
                    );
                    Message::Ping(bytes.clone())
                }
                Outbound::Pong(bytes) => {
                    tracing::debug!(
                        session_id = writer_session_id,
                        bytes = bytes.len(),
                        "websocket send pong"
                    );
                    Message::Pong(bytes.clone())
                }
                Outbound::Close => {
                    tracing::info!(
                        session_id = writer_session_id,
                        "sending websocket close frame"
                    );
                    Message::Close(None)
                }
            };

            if let Err(err) = ws_tx.send(message).await {
                tracing::debug!(%err, "websocket writer stopped");
                break;
            }
            if matches!(outbound, Outbound::Close) {
                tracing::debug!(
                    session_id = writer_session_id,
                    "websocket writer exiting after close frame"
                );
                break;
            }
        }
    });

    let ctx = SessionContext {
        id: session_id.clone(),
        version: Arc::new(Mutex::new(protocol_version)),
        tx: out_tx.clone(),
        services,
        shutdown_tx: shutdown_tx.clone(),
    };
    let state = Arc::new(Mutex::new(SessionState {
        llm: llm_slot,
        abort_notify: Arc::new(Notify::new()),
        // Seed the idle clock at session start so the watchdog doesn't
        // fire immediately for a freshly-connected device.
        last_voice_ts: Some(Instant::now()),
        ..SessionState::default()
    }));

    // Idle watchdog: after a quiet window with no audio and no
    // in-flight pipeline, synthesize a goodbye prompt and let the LLM
    // close out the session. Drop happens automatically when the main
    // loop exits below.
    let watchdog_ctx = ctx.clone();
    let watchdog_state = state.clone();
    let watchdog = tokio::spawn(async move {
        idle_watchdog(watchdog_ctx, watchdog_state).await;
    });

    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.recv() => {
                tracing::info!(
                    session_id,
                    "shutdown signal received; closing websocket"
                );
                break;
            }
            message = ws_rx.next() => {
                let Some(message) = message else {
                    break;
                };
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
                        if complete_ws_probe(&state, &data).await {
                            tracing::debug!(
                                session_id,
                                bytes = data.len(),
                                "websocket recv pong matched probe"
                            );
                        } else {
                            tracing::debug!(session_id, bytes = data.len(), "websocket recv pong");
                        }
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
        }
    }

    watchdog.abort();

    abort_active(&ctx, &state, false).await;
    {
        let llm_slot = state.lock().await.llm.clone();
        if let Some(mut llm) = llm_slot.lock().await.take() {
            llm.shutdown().await;
            tracing::info!(session_id, "llm session shutdown complete");
        }
    }
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
                let audio_decoder = match OpusPcmDecoder::new(CLIENT_SAMPLE_RATE) {
                    Ok(decoder) => decoder,
                    Err(err) => {
                        let error_chain = format_error_chain(err.as_ref());
                        tracing::warn!(
                            session_id = ctx.id,
                            error_chain,
                            "failed to create opus decoder for input audio"
                        );
                        return true;
                    }
                };

                let vad = match ctx
                    .services
                    .vad
                    .as_ref()
                    .map(|vad| vad.start_stream())
                    .transpose()
                {
                    Ok(vad) => vad,
                    Err(err) => {
                        let error_chain = format_error_chain(err.as_ref());
                        tracing::warn!(
                            session_id = ctx.id,
                            error_chain,
                            "failed to start VAD stream"
                        );
                        return true;
                    }
                };

                let max_timeout = listen_max_timeout();
                let timeout_ctx = ctx.clone();
                let timeout_state = state.clone();
                let listen_timeout = tokio::spawn(async move {
                    sleep(max_timeout).await;
                    finish_listening(timeout_ctx, timeout_state, "maximum listen timeout", false)
                        .await;
                });

                let mut guard = state.lock().await;
                guard.listening = true;
                guard.already_triggered = false;
                guard.last_voice_ts = Some(Instant::now());
                guard.listen_mode = incoming.mode().map(ToOwned::to_owned);
                // ASR is intentionally NOT started here. It will be opened on
                // VAD `SpeechStart` so the first ASR bytes include the pre-roll
                // audio that the VAD just confirmed.
                guard.asr = None;
                guard.audio_decoder = Some(audio_decoder);
                guard.vad = vad;
                guard.pcm_ring_buffer = Some(PcmRingBuffer::new(PCM_RING_BUFFER_SAMPLES));
                guard.listen_started_at = Some(Instant::now());
                guard.first_audio_logged = false;
                guard.input_frames = 0;
                if let Some(old) = guard.listen_timeout.replace(listen_timeout) {
                    old.abort();
                }
                tracing::info!(
                    session_id = ctx.id,
                    mode = ?guard.listen_mode,
                    vad_enabled = guard.vad.is_some(),
                    timeout_ms = max_timeout.as_millis(),
                    "listen started"
                );
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
    guard.last_voice_ts = Some(Instant::now());
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

    let samples = match guard
        .audio_decoder
        .as_mut()
        .map(|decoder| decoder.decode_to_pcm_i16(&frame.payload))
    {
        Some(Ok(samples)) => samples,
        Some(Err(err)) => {
            let error_chain = format_error_chain(err.as_ref());
            tracing::warn!(
                session_id = ctx.id,
                error_chain,
                "failed to decode input opus frame"
            );
            return;
        }
        None => return,
    };

    if let Some(asr) = guard.asr.as_mut() {
        if let Err(err) = asr.push_pcm(&samples).await {
            let error_chain = format_error_chain(err.as_ref());
            tracing::warn!(
                session_id = ctx.id,
                error_chain,
                "failed to push pcm to asr stream"
            );
        }
    } else if let Some(ring) = guard.pcm_ring_buffer.as_mut() {
        // Pre-roll: ASR is not yet open, accumulate PCM so it can be
        // drained into ASR when VAD finally confirms speech onset.
        ring.push(&samples);
    }

    let mut finish_reason = None;
    let mut asr_open_failure: Option<String> = None;
    if let Some(vad) = guard.vad.as_mut() {
        match vad.accept_pcm(&samples) {
            Ok(events) => {
                for event in events {
                    match event {
                        VadEvent::SpeechStart { sample } => {
                            tracing::info!(
                                session_id = ctx.id,
                                sample,
                                "VAD speech started; opening ASR"
                            );
                            match ctx.services.asr.start_stream().await {
                                Ok(mut asr) => {
                                    if let Some(ring) = guard.pcm_ring_buffer.take() {
                                        let prefill = ring.drain();
                                        if !prefill.is_empty() {
                                            tracing::debug!(
                                                session_id = ctx.id,
                                                samples = prefill.len(),
                                                "draining pre-roll PCM to ASR"
                                            );
                                            if let Err(err) = asr.push_pcm(&prefill).await {
                                                let error_chain = format_error_chain(err.as_ref());
                                                tracing::warn!(
                                                    session_id = ctx.id,
                                                    error_chain,
                                                    "failed to prefill ASR with pcm"
                                                );
                                            }
                                        }
                                    }
                                    guard.asr = Some(asr);
                                    // VAD has confirmed speech; the listen
                                    // timeout is no longer needed as a backstop.
                                    if let Some(timeout) = guard.listen_timeout.take() {
                                        timeout.abort();
                                    }
                                }
                                Err(err) => {
                                    let error_chain = format_error_chain(err.as_ref());
                                    tracing::error!(
                                        session_id = ctx.id,
                                        error_chain,
                                        "failed to start ASR on VAD trigger"
                                    );
                                    asr_open_failure = Some(error_chain);
                                }
                            }
                        }
                        VadEvent::SpeechEnd {
                            start_sample,
                            end_sample,
                        } => {
                            tracing::info!(
                                session_id = ctx.id,
                                start_sample,
                                end_sample,
                                "VAD speech ended"
                            );
                            finish_reason = Some("vad speech ended");
                        }
                    }
                }
            }
            Err(err) => {
                let error_chain = format_error_chain(err.as_ref());
                tracing::warn!(session_id = ctx.id, error_chain, "VAD processing failed");
            }
        }
    }
    drop(guard);

    if let Some(reason) = finish_reason {
        finish_listening(ctx.clone(), state.clone(), reason, true).await;
    }
    if let Some(error_chain) = asr_open_failure {
        tracing::error!(
            session_id = ctx.id,
            error_chain,
            "aborting listen due to ASR open failure"
        );
        abort_active(ctx, state, false).await;
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
            guard.audio_decoder = None;
            guard.vad = None;
            guard.pcm_ring_buffer = None;
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
    let pipeline_state = state.clone();
    let handle = tokio::spawn(async move {
        if let Err(err) = run_pipeline(pipeline_ctx.clone(), pipeline_state, asr).await {
            let error_chain = format_error_chain(err.as_ref());
            tracing::error!(
                session_id = pipeline_ctx.id,
                error_chain,
                "conversation pipeline failed"
            );
            send_json(&pipeline_ctx, protocol::tts_stop(&pipeline_ctx.id)).await;
        }
    });

    let mut guard = state.lock().await;
    if let Some(old) = guard.pipeline.replace(handle) {
        // `old` is the previous pipeline task. Under normal flow
        // `abort_active` already notified it to exit cleanly before we
        // got here, so this should normally be a no-op `await` of an
        // already-finished task. The `abort()` is a belt-and-braces
        // fallback for the rare race where a new listen round started
        // before the previous one finished (e.g. a client sends
        // `listen.start` while a previous `listen.stop` is still in
        // flight). Even in that case the LLM now lives in
        // `state.llm` (not on the pipeline stack), so cancelling the
        // task no longer drops the provider-owned session handle.
        if !old.is_finished() {
            tracing::debug!(
                session_id = ctx.id,
                "previous pipeline still running while starting a new one; aborting it"
            );
            old.abort();
        }
    }
}

async fn run_pipeline(
    ctx: SessionContext,
    state: Arc<Mutex<SessionState>>,
    mut asr: Box<dyn AsrStream>,
) -> anyhow::Result<()> {
    let pipeline_started = Instant::now();
    tracing::debug!(session_id = ctx.id, "pipeline timing started");

    // Abort notifier is cloned here and again inside the loop. We hold a
    // `Notified` future for the *outer* wait (during ASR finish) so that an
    // abort arriving before the pipeline reaches the TTS loop is also
    // observed.
    let abort_notify = {
        let guard = state.lock().await;
        guard.abort_notify.clone()
    };

    let asr_finish_started = Instant::now();
    let asr_result = tokio::select! {
        biased;
        _ = abort_notify.notified() => {
            tracing::info!(
                session_id = ctx.id,
                "pipeline aborted during asr finish"
            );
            // Drain the asr stream so its underlying connection is closed
            // cleanly even though we won't run the LLM/TTS pipeline.
            let _ = asr.abort().await;
            return Ok(());
        }
        result = asr.finish() => result,
    };
    let text = asr_result?;
    tracing::debug!(
        session_id = ctx.id,
        elapsed_ms = asr_finish_started.elapsed().as_millis(),
        total_elapsed_ms = pipeline_started.elapsed().as_millis(),
        text_chars = text.chars().count(),
        "asr finish returned final text"
    );

    run_pipeline_with_text(ctx, state, pipeline_started, text).await
}

async fn run_pipeline_with_text(
    ctx: SessionContext,
    state: Arc<Mutex<SessionState>>,
    pipeline_started: Instant,
    text: String,
) -> anyhow::Result<()> {
    // Abort notifier is cloned here and again inside the loop. We hold a
    // `Notified` future for the *outer* wait (during ASR finish) so that an
    // abort arriving before the pipeline reaches the TTS loop is also
    // observed.
    let abort_notify = {
        let guard = state.lock().await;
        guard.abort_notify.clone()
    };

    // Exit-command detection: full, trimmed equality against the
    // configured keywords. If matched we still feed the text to the LLM
    // so it can craft a goodbye, and after the TTS stream finishes we
    // close the WebSocket. This deliberately does not interfere with the
    // abort path: `aborted=true` short-circuits the shutdown below.
    if let Some(cmd) = check_exit_command(&text) {
        state.lock().await.close_after_chat = true;
        tracing::info!(
            session_id = ctx.id,
            matched = cmd,
            text = %text,
            "exit command matched; will close after tts finishes"
        );
    }

    if text.trim().is_empty() {
        tracing::info!(
            session_id = ctx.id,
            total_elapsed_ms = pipeline_started.elapsed().as_millis(),
            "asr returned empty text; skipping llm and returning to listen"
        );
        // Send a paired `tts.start` + `tts.stop` so the client
        // transitions through `kDeviceStateSpeaking` and back to
        // `kDeviceStateListening`. The ESP32 firmware only acts on
        // `tts.stop` when it is currently in `kDeviceStateSpeaking`,
        // and the transition into `kDeviceStateListening` is what
        // triggers it to re-send `listen.start`. Without `tts.start`
        // first, the client stays in `kDeviceStateListening` (its
        // state never changes), never re-sends `listen.start`, and
        // every subsequent audio frame is dropped server-side by
        // `discarding audio while not listening`. This is the
        // "ASR returned empty text and nothing responds afterwards"
        // failure mode.
        send_json(&ctx, protocol::tts_start(&ctx.id)).await;
        send_json(&ctx, protocol::tts_stop(&ctx.id)).await;
        maybe_schedule_close_after_chat(&ctx, &state, false).await;
        return Ok(());
    }

    send_json(&ctx, protocol::stt(&ctx.id, &text)).await;
    send_json(&ctx, protocol::llm_emotion(&ctx.id, "happy", "😊")).await;
    send_json(&ctx, protocol::tts_start(&ctx.id)).await;
    tracing::debug!(
        session_id = ctx.id,
        total_elapsed_ms = pipeline_started.elapsed().as_millis(),
        "stt/llm-emotion/tts-start events enqueued"
    );

    let llm_started = Instant::now();
    // Borrow the LLM session under the lock for the synchronous
    // `chat_stream` call only. The lock is dropped as soon as we have the
    // `TextStream` in hand; we never hold the lock across an `await`. This
    // is what guarantees `abort_active` can always reach the LLM to abort
    // the active provider request, and that the LLM session is never dropped
    // because the pipeline task was cancelled.
    let llm_slot = state.lock().await.llm.clone();
    let raw_llm_stream = {
        let mut guard = llm_slot.lock().await;
        match guard.as_mut() {
            Some(llm) => llm.chat_stream(text),
            None => {
                tracing::error!(
                    session_id = ctx.id,
                    "llm session not available; aborting conversation pipeline"
                );
                drop(guard);
                let _ = ctx
                    .tx
                    .send(Outbound::Text(serde_json::json!({
                        "session_id": ctx.id,
                        "type": "error",
                        "message": "llm session not available"
                    })))
                    .await;
                return Ok(());
            }
        }
    };
    let llm_stream = instrument_llm_stream(
        ctx.id.clone(),
        pipeline_started,
        llm_started,
        raw_llm_stream,
    );

    let tts_started = Instant::now();
    let tts_input = filter_tts_text_stream(llm_stream);
    let collected_text = Arc::new(StdMutex::new(String::new()));
    let tts_input = tee_collect_text(tts_input, collected_text.clone());
    let mut tts_stream = ctx.services.tts.synthesize_stream(tts_input);
    tracing::debug!(
        session_id = ctx.id,
        total_elapsed_ms = pipeline_started.elapsed().as_millis(),
        "tts stream created"
    );

    let mut sentence_events = 0u64;
    let mut audio_frames = 0u64;
    let mut first_sentence_logged = false;
    let mut first_audio_logged = false;
    let mut aborted = false;

    // TTS event loop with abort awareness. We `select!` on the next TTS
    // event vs. the abort notifier so an abort mid-pipeline is observed
    // promptly without cancelling the task (and thus without dropping the
    // provider-owned LLM session handle).
    loop {
        let next_event = tts_stream.next();
        tokio::pin!(next_event);
        let event = tokio::select! {
            biased;
            _ = abort_notify.notified() => {
                tracing::info!(
                    session_id = ctx.id,
                    audio_frames,
                    "pipeline aborted mid-tts"
                );
                aborted = true;
                break;
            }
            event = &mut next_event => event,
        };
        let event = match event {
            Some(Ok(event)) => event,
            Some(Err(err)) => return Err(err),
            None => break,
        };
        match event {
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
        aborted,
        "tts stream finished"
    );
    // Always emit `tts.stop` on the way out, including the abort path, so
    // the client reliably transitions out of the speaking state. On normal
    // completion this is the pair of `tts.start` we emitted earlier; on
    // abort it cancels whatever audio the client was already playing.
    send_json(&ctx, protocol::tts_stop(&ctx.id)).await;
    if !aborted {
        let llm_reply = std::mem::take(&mut *collected_text.lock().unwrap());
        tracing::info!(
            session_id = ctx.id,
            chars = llm_reply.chars().count(),
            text = %llm_reply,
            total_elapsed_ms = pipeline_started.elapsed().as_millis(),
            "llm reply"
        );
    }
    tracing::debug!(
        session_id = ctx.id,
        total_elapsed_ms = pipeline_started.elapsed().as_millis(),
        "pipeline finished"
    );
    maybe_schedule_close_after_chat(&ctx, &state, aborted).await;
    Ok(())
}

/// If `close_after_chat` is set on the session state and the pipeline
/// wasn't aborted, wait briefly for the final TTS audio to flush out to
/// the device, then send a WebSocket close frame and signal the main
/// loop to exit.
///
/// Sending an explicit close frame (rather than just dropping the TCP
/// connection) is what tells the ESP32 firmware that the server is
/// hanging up. Without it the device would treat the disconnect as an
/// error and resume its listening loop, which is the
/// "说关闭之后又进入监听状态了" failure mode the user reported.
async fn maybe_schedule_close_after_chat(
    ctx: &SessionContext,
    state: &Arc<Mutex<SessionState>>,
    aborted: bool,
) {
    if aborted {
        return;
    }
    let should_close = state.lock().await.close_after_chat;
    if !should_close {
        return;
    }
    tracing::info!(
        session_id = ctx.id,
        "close_after_chat set; flushing final audio before closing websocket"
    );
    // Give the final TTS frame time to reach the device and play out.
    sleep(POST_TTS_FLUSH_DELAY).await;
    // Send an explicit WebSocket close frame so the device knows this
    // is a graceful hang-up, not a network error. After this the
    // writer task exits and the device transitions out of listening.
    if ctx.tx.send(Outbound::Close).await.is_err() {
        tracing::debug!(
            session_id = ctx.id,
            "outbound queue closed; main loop already gone"
        );
    }
    if ctx.shutdown_tx.send(()).await.is_err() {
        tracing::debug!(
            session_id = ctx.id,
            "shutdown_tx closed; main loop already gone"
        );
    }
}

// (helper removed: we now `tokio::pin!` the next-event future inline in
//  `run_pipeline` so the abort notifier can race it directly.)

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

fn tee_collect_text(mut input: TextStream, sink: Arc<StdMutex<String>>) -> TextStream {
    Box::pin(try_stream! {
        while let Some(chunk) = input.next().await {
            let chunk = chunk?;
            sink.lock().unwrap().push_str(&chunk);
            yield chunk;
        }
    })
}

async fn abort_active(ctx: &SessionContext, state: &Arc<Mutex<SessionState>>, send_stop: bool) {
    // Gather the things we need to talk to. We deliberately do NOT take
    // `state.llm` here: the LLM session lives in `state.llm` for the
    // lifetime of the WebSocket connection, and we only borrow it to send
    // an abort command. This is the fix for the `llm session not
    // available` bug: previously the pipeline task would take the LLM out
    // of state, and aborting the pipeline task would drop it before the
    // provider saw an abort. Now the LLM stays put and the pipeline task is
    // notified (not cancelled) to exit cleanly.
    let (abort_notify, mut asr) = {
        let mut guard = state.lock().await;
        guard.listening = false;
        guard.already_triggered = false;
        guard.audio_decoder = None;
        guard.vad = None;
        guard.pcm_ring_buffer = None;
        guard.listen_started_at = None;
        guard.first_audio_logged = false;
        guard.input_frames = 0;
        if let Some(timeout) = guard.listen_timeout.take() {
            timeout.abort();
        }
        let asr = guard.asr.take();
        (guard.abort_notify.clone(), asr)
    };

    // Notify the pipeline task so it can stop at the next TTS event
    // boundary, emit a `tts.stop`, and exit. We don't cancel the task
    // because that would drop its stack-locals and could leave the
    // websocket writer in a weird state. The pipeline task will itself
    // emit `tts.stop` on its way out (see `run_pipeline`).
    abort_notify.notify_waiters();

    // Send abort to the LLM. For pi-server this drops the in-flight HTTP
    // response body, which causes the server to abort the active generation.
    {
        let llm_slot = state.lock().await.llm.clone();
        let mut llm_guard = llm_slot.lock().await;
        if let Some(llm) = llm_guard.as_mut() {
            llm.abort().await;
        }
    }

    // Abort the in-flight ASR stream (if any). This is per-listen-round
    // state, not the long-lived LLM.
    if let Some(asr) = asr.as_mut() {
        asr.abort().await;
        tracing::info!(session_id = ctx.id, "asr stream aborted");
    }

    if send_stop {
        send_json(ctx, protocol::tts_stop(&ctx.id)).await;
    }
    tracing::info!(session_id = ctx.id, "abort dispatched");
}

async fn send_json(ctx: &SessionContext, value: Value) {
    if ctx.tx.send(Outbound::Text(value)).await.is_err() {
        tracing::debug!(
            session_id = ctx.id,
            "failed to enqueue json: websocket closed"
        );
    }
}

async fn probe_websocket_alive(ctx: &SessionContext, state: &Arc<Mutex<SessionState>>) -> bool {
    let payload = Uuid::new_v4().as_bytes().to_vec();
    let (done_tx, done_rx) = oneshot::channel();

    {
        let mut guard = state.lock().await;
        if guard.ws_probe.is_some() {
            tracing::debug!(session_id = ctx.id, "websocket probe already in flight");
            return true;
        }
        guard.ws_probe = Some(WebSocketProbe {
            payload: Bytes::from(payload.clone()),
            done_tx,
        });
    }

    if ctx
        .tx
        .send(Outbound::Ping(Bytes::from(payload.clone())))
        .await
        .is_err()
    {
        tracing::debug!(session_id = ctx.id, "failed to enqueue websocket ping");
        let mut guard = state.lock().await;
        guard.ws_probe = None;
        return false;
    }

    match timeout(WEBSOCKET_PROBE_TIMEOUT, done_rx).await {
        Ok(Ok(())) => {
            tracing::debug!(session_id = ctx.id, "websocket probe succeeded");
            true
        }
        Ok(Err(_)) => {
            tracing::debug!(session_id = ctx.id, "websocket probe channel closed");
            let mut guard = state.lock().await;
            guard.ws_probe = None;
            false
        }
        Err(_) => {
            tracing::debug!(session_id = ctx.id, "websocket probe timed out");
            let mut guard = state.lock().await;
            guard.ws_probe = None;
            false
        }
    }
}

async fn complete_ws_probe(state: &Arc<Mutex<SessionState>>, pong: &Bytes) -> bool {
    let probe = {
        let mut guard = state.lock().await;
        let Some(probe) = guard.ws_probe.take() else {
            return false;
        };
        if probe.payload != *pong {
            guard.ws_probe = Some(probe);
            return false;
        }
        probe
    };

    let _ = probe.done_tx.send(());
    true
}

fn listen_max_timeout() -> Duration {
    std::env::var("XIAOZHI_LISTEN_MAX_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_LISTEN_MAX_TIMEOUT)
}

fn idle_close_after() -> Duration {
    std::env::var("XIAOZHI_IDLE_CLOSE_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(DEFAULT_IDLE_CLOSE_SECONDS))
}

fn exit_commands() -> Vec<String> {
    std::env::var("XIAOZHI_EXIT_COMMANDS")
        .ok()
        .map(|value| {
            // Split on commas and the CJK enumeration comma (、) only.
            // Whitespace is NOT a separator, so phrases like "see you"
            // survive as one entry. We still trim each entry so authors
            // can use `, ` or `,，` comfortably.
            value
                .split([',', '\u{3001}'])
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        })
        .filter(|cmds: &Vec<String>| !cmds.is_empty())
        .unwrap_or_else(|| {
            DEFAULT_EXIT_COMMANDS
                .iter()
                .map(|s| s.to_string())
                .collect()
        })
}

fn end_prompt_text() -> String {
    std::env::var("XIAOZHI_END_PROMPT")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_END_PROMPT.to_string())
}

fn end_prompt_enabled() -> bool {
    !matches!(
        std::env::var("XIAOZHI_END_PROMPT_ENABLED")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("0") | Some("false") | Some("off") | Some("no") | Some("disabled")
    )
}

/// Returns the matched command keyword if `text` is exactly (after
/// trimming whitespace and removing common trailing punctuation) one of
/// the configured exit commands. Comparison is case-sensitive on the
/// trimmed core but tolerant of trailing `。`, `！`, `?`, `？`, `!`,
/// `~`, `…` and similar, in both ASCII and CJK fullwidth forms.
fn check_exit_command(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let stripped = trimmed
        .trim_end_matches(|c: char| {
            // CJK fullwidth punctuation first (more common in Chinese
            // ASR output) then ASCII fallbacks. Whitespace is included
            // so a stray trailing space doesn't block the match.
            matches!(c, '。' | '！' | '？' | '!' | '?' | '~' | '…' | ' ' | '\t')
        })
        .trim_end();
    exit_commands()
        .into_iter()
        .find(|cmd| stripped == cmd.as_str() || trimmed == cmd.as_str())
}

/// Watchdog that fires when the device has been idle for longer than the
/// configured threshold with no audio and no in-flight pipeline. When it
/// fires it seeds `close_after_chat=true` and feeds the end-prompt as a
/// synthetic user turn into the pipeline, so the LLM produces a goodbye
/// and the normal exit path closes the WebSocket. Drop is the cancel
/// mechanism: when `handle_websocket` returns it aborts the spawned
/// task.
async fn idle_watchdog(ctx: SessionContext, state: Arc<Mutex<SessionState>>) {
    if !end_prompt_enabled() {
        tracing::info!(
            session_id = ctx.id,
            "end prompt disabled; idle watchdog will not fire"
        );
        return;
    }
    let threshold = idle_close_after();
    let interval = Duration::from_secs(5);
    let prompt = end_prompt_text();
    tracing::info!(
        session_id = ctx.id,
        threshold_seconds = threshold.as_secs(),
        "idle watchdog armed"
    );

    loop {
        sleep(interval).await;

        // First verify the session is idle, then ping the client before
        // we spend tokens on a synthetic goodbye. If the probe fails we
        // close the session quietly instead of speaking to a dead socket.
        let idle = {
            let guard = state.lock().await;
            let idle = guard
                .last_voice_ts
                .map(|ts| ts.elapsed() >= threshold)
                .unwrap_or(false);
            let busy = guard.listening || guard.pipeline.is_some();
            !guard.close_after_chat && !busy && idle
        };
        if !idle {
            continue;
        }

        if !probe_websocket_alive(&ctx, &state).await {
            tracing::info!(
                session_id = ctx.id,
                threshold_seconds = threshold.as_secs(),
                "idle timeout reached but websocket probe failed; closing session without goodbye"
            );
            let _ = ctx.shutdown_tx.send(()).await;
            return;
        }

        // Re-check after the probe: a user may have resumed speaking
        // while the ping was in flight, and we should not claim the
        // session in that case.
        let should_fire = {
            let mut guard = state.lock().await;
            let idle = guard
                .last_voice_ts
                .map(|ts| ts.elapsed() >= threshold)
                .unwrap_or(false);
            let busy = guard.listening || guard.pipeline.is_some();
            if !guard.close_after_chat && !busy && idle {
                guard.close_after_chat = true;
                guard.last_voice_ts = Some(Instant::now());
                true
            } else {
                false
            }
        };
        if !should_fire {
            continue;
        }

        tracing::info!(
            session_id = ctx.id,
            threshold_seconds = threshold.as_secs(),
            "idle timeout reached; feeding synthetic end-prompt to llm"
        );

        let prompt = prompt.clone();
        let ctx_for_pipeline = ctx.clone();
        let state_for_pipeline = state.clone();
        let handle = tokio::spawn(async move {
            let pipeline_started = Instant::now();
            if let Err(err) = run_pipeline_with_text(
                ctx_for_pipeline.clone(),
                state_for_pipeline.clone(),
                pipeline_started,
                prompt,
            )
            .await
            {
                tracing::error!(
                    session_id = ctx_for_pipeline.id,
                    error = %format_error_chain(err.as_ref()),
                    "idle-triggered pipeline failed"
                );
                // Best-effort: still try to close so the device doesn't
                // hang on a wedged session.
                let _ = ctx_for_pipeline.shutdown_tx.send(()).await;
            }
        });
        state.lock().await.pipeline = Some(handle);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// `check_exit_command` / `idle_close_after` / `end_prompt_enabled`
    /// all read env at call time. Cargo runs tests in parallel, but env
    /// is process-global state — so any test that touches an env var
    /// must hold this mutex for its entire body. The 2024 edition also
    /// marks `set_var` / `remove_var` as `unsafe` because of the data
    /// race; holding the lock is the synchronisation that makes the
    /// unsafe call sound.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// `check_exit_command` reads `XIAOZHI_EXIT_COMMANDS` at call time,
    /// so tests must set/clear the env var explicitly. We isolate each
    /// case with a tiny RAII guard.
    struct ExitCommandsGuard {
        prev: Option<String>,
    }

    impl ExitCommandsGuard {
        fn new(value: Option<&str>) -> Self {
            let prev = std::env::var("XIAOZHI_EXIT_COMMANDS").ok();
            // SAFETY: the test holds ENV_LOCK for the duration of the
            // guard, so no other thread reads this env var concurrently.
            unsafe {
                match value {
                    Some(v) => std::env::set_var("XIAOZHI_EXIT_COMMANDS", v),
                    None => std::env::remove_var("XIAOZHI_EXIT_COMMANDS"),
                }
            }
            Self { prev }
        }
    }

    impl Drop for ExitCommandsGuard {
        fn drop(&mut self) {
            // SAFETY: see ExitCommandsGuard::new.
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var("XIAOZHI_EXIT_COMMANDS", v),
                    None => std::env::remove_var("XIAOZHI_EXIT_COMMANDS"),
                }
            }
        }
    }

    #[test]
    fn exit_command_matches_default_keywords_exactly() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = ExitCommandsGuard::new(None);
        let r = check_exit_command("退出");
        let d = r.as_deref();
        assert_eq!(d, Some("退出"));
        let r = check_exit_command("关闭");
        let d = r.as_deref();
        assert_eq!(d, Some("关闭"));
    }

    #[test]
    fn exit_command_tolerates_trailing_punctuation() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = ExitCommandsGuard::new(None);
        let r = check_exit_command("退出。");
        let d = r.as_deref();
        assert_eq!(d, Some("退出"));
        let r = check_exit_command("退出!");
        let d = r.as_deref();
        assert_eq!(d, Some("退出"));
        let r = check_exit_command("退出？");
        let d = r.as_deref();
        assert_eq!(d, Some("退出"));
        let r = check_exit_command("关闭…");
        let d = r.as_deref();
        assert_eq!(d, Some("关闭"));
    }

    #[test]
    fn exit_command_does_not_match_substring_or_sentence() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = ExitCommandsGuard::new(None);
        // "退出系统" is a phrase, not the bare keyword.
        assert!(check_exit_command("退出系统").is_none());
        assert!(check_exit_command("我想退出").is_none());
        assert!(check_exit_command("请关闭一下吧").is_none());
        assert!(check_exit_command("").is_none());
        assert!(check_exit_command("   ").is_none());
    }

    #[test]
    fn exit_command_respects_custom_env_var() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = ExitCommandsGuard::new(Some("bye,see you,晚安"));
        let r = check_exit_command("bye");
        let d = r.as_deref();
        assert_eq!(d, Some("bye"));
        let r = check_exit_command("see you");
        let d = r.as_deref();
        assert_eq!(d, Some("see you"));
        let r = check_exit_command("晚安");
        let d = r.as_deref();
        assert_eq!(d, Some("晚安"));
        // Defaults no longer apply once env is set.
        assert!(check_exit_command("退出").is_none());
    }

    #[test]
    fn idle_close_after_reads_env_or_default() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("XIAOZHI_IDLE_CLOSE_SECONDS").ok();
        // SAFETY: ENV_LOCK is held; no concurrent env readers.
        unsafe {
            std::env::remove_var("XIAOZHI_IDLE_CLOSE_SECONDS");
            assert_eq!(idle_close_after(), Duration::from_secs(90));
            std::env::set_var("XIAOZHI_IDLE_CLOSE_SECONDS", "45");
            assert_eq!(idle_close_after(), Duration::from_secs(45));
            std::env::set_var("XIAOZHI_IDLE_CLOSE_SECONDS", "0");
            // 0 is treated as "use default" (not a valid threshold).
            assert_eq!(idle_close_after(), Duration::from_secs(90));
            std::env::set_var("XIAOZHI_IDLE_CLOSE_SECONDS", "bogus");
            assert_eq!(idle_close_after(), Duration::from_secs(90));
            match prev {
                Some(v) => std::env::set_var("XIAOZHI_IDLE_CLOSE_SECONDS", v),
                None => std::env::remove_var("XIAOZHI_IDLE_CLOSE_SECONDS"),
            }
        }
    }

    #[test]
    fn end_prompt_disabled_short_circuits_watchdog() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("XIAOZHI_END_PROMPT_ENABLED").ok();
        // SAFETY: ENV_LOCK is held; no concurrent env readers.
        unsafe {
            std::env::set_var("XIAOZHI_END_PROMPT_ENABLED", "false");
            assert!(!end_prompt_enabled());
            std::env::set_var("XIAOZHI_END_PROMPT_ENABLED", "0");
            assert!(!end_prompt_enabled());
            std::env::set_var("XIAOZHI_END_PROMPT_ENABLED", "off");
            assert!(!end_prompt_enabled());
            std::env::remove_var("XIAOZHI_END_PROMPT_ENABLED");
            assert!(end_prompt_enabled());
            std::env::set_var("XIAOZHI_END_PROMPT_ENABLED", "true");
            assert!(end_prompt_enabled());
            match prev {
                Some(v) => std::env::set_var("XIAOZHI_END_PROMPT_ENABLED", v),
                None => std::env::remove_var("XIAOZHI_END_PROMPT_ENABLED"),
            }
        }
    }
}
