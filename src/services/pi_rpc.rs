use std::{
    fs,
    path::PathBuf,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use anyhow::{Context, Result, anyhow};
use async_stream::try_stream;
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, Command},
    sync::{broadcast, mpsc, oneshot},
};

use super::{LlmSession, LlmSessionFactory, LlmSessionMeta, TextStream};

#[derive(Clone, Debug)]
pub struct PiRpcLlmFactoryConfig {
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub stream_idle_timeout: std::time::Duration,
}

impl PiRpcLlmFactoryConfig {
    pub fn from_env() -> Result<Option<Self>> {
        let provider = std::env::var("XIAOZHI_LLM_PROVIDER")
            .or_else(|_| std::env::var("LLM_PROVIDER"))
            .unwrap_or_else(|_| "mock".to_string())
            .to_ascii_lowercase();
        if provider != "pi" && provider != "pi-rpc" && provider != "pi_rpc" {
            return Ok(None);
        }

        let command = std::env::var("XIAOZHI_PI_RPC_COMMAND").unwrap_or_else(|_| "pi".to_string());

        // --mode rpc first, then either the user-supplied flags or our safe defaults,
        // then the system prompt file content appended as `--system-prompt <text>`.
        let mut args: Vec<String> = vec!["--mode".to_string(), "rpc".to_string()];

        let flag_string = env_str("XIAOZHI_PI_RPC_FLAGS");
        let user_flags: Vec<String> = match flag_string.as_deref() {
            None | Some("") => default_safe_flags(),
            Some(raw) => raw.split_whitespace().map(|s| s.to_string()).collect(),
        };
        args.extend(user_flags);

        if let Some(path) = env_str("XIAOZHI_PI_RPC_SYSTEM_PROMPT_FILE") {
            let prompt_text = fs::read_to_string(&path)
                .with_context(|| format!("read pi system prompt file {path}"))?;
            let trimmed = prompt_text.trim_end();
            if !trimmed.is_empty() {
                args.push("--system-prompt".to_string());
                args.push(trimmed.to_string());
            }
        }

        let cwd = env_str("XIAOZHI_PI_RPC_CWD").map(PathBuf::from);
        let stream_idle_timeout_ms = env_u64("XIAOZHI_PI_RPC_IDLE_TIMEOUT_MS", 60_000);

        let config = Self {
            command: command.clone(),
            args: args.clone(),
            cwd,
            stream_idle_timeout: std::time::Duration::from_millis(stream_idle_timeout_ms),
        };

        Ok(Some(config))
    }
}

#[derive(Clone)]
pub struct PiRpcLlmFactory {
    config: Arc<PiRpcLlmFactoryConfig>,
}

impl PiRpcLlmFactory {
    pub fn new(config: PiRpcLlmFactoryConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl LlmSessionFactory for PiRpcLlmFactory {
    async fn create_session(&self, meta: LlmSessionMeta) -> Result<Box<dyn LlmSession>> {
        let session = PiRpcLlmSession::spawn(self.config.clone(), meta)
            .await
            .context("spawn pi --mode rpc subprocess")?;
        Ok(Box::new(session))
    }
}

// -----------------------------------------------------------------------------
// Session implementation
// -----------------------------------------------------------------------------

struct PiRpcLlmSession {
    meta: LlmSessionMeta,
    writer_tx: mpsc::Sender<WriterCommand>,
    event_tx: broadcast::Sender<PiEvent>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    next_request_id: Arc<AtomicU64>,
}

enum WriterCommand {
    Prompt {
        request_id: u64,
        message: String,
        reply: oneshot::Sender<Result<()>>,
    },
    Abort {
        request_id: u64,
    },
    Shutdown,
}

impl PiRpcLlmSession {
    async fn spawn(config: Arc<PiRpcLlmFactoryConfig>, meta: LlmSessionMeta) -> Result<Self> {
        let mut command = Command::new(&config.command);
        command
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(cwd) = &config.cwd {
            command.current_dir(cwd);
        }
        // Pass through a small allowlist of pi-specific env knobs when set.
        for key in ["PI_NO_LOCAL_LLM", "PI_OFFLINE"] {
            if let Ok(value) = std::env::var(key) {
                command.env(key, value);
            }
        }

        let mut child: Child = command
            .spawn()
            .with_context(|| format!("spawn pi rpc command `{}`", config.command))?;

        let stdin: ChildStdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("pi rpc child stdin missing"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("pi rpc child stdout missing"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("pi rpc child stderr missing"))?;

        let (writer_tx, mut writer_rx) = mpsc::channel::<WriterCommand>(32);
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        // Writer task: serializes all child stdin writes from the channel.
        tokio::spawn(async move {
            let mut stdin = stdin;
            while let Some(command) = writer_rx.recv().await {
                match command {
                    WriterCommand::Prompt {
                        request_id,
                        message,
                        reply,
                    } => {
                        let payload = json!({
                            "id": format!("req-{request_id}"),
                            "type": "prompt",
                            "message": message,
                        });
                        let line = format!("{payload}\n");
                        let result = match stdin.write_all(line.as_bytes()).await {
                            Ok(()) => stdin.flush().await.context("flush pi rpc prompt"),
                            Err(err) => {
                                Err(anyhow::Error::from(err).context("write pi rpc prompt"))
                            }
                        };
                        let _ = reply.send(result);
                    }
                    WriterCommand::Abort { request_id } => {
                        let payload = json!({
                            "id": format!("abort-{request_id}"),
                            "type": "abort",
                        });
                        let line = format!("{payload}\n");
                        if let Err(err) = stdin.write_all(line.as_bytes()).await {
                            tracing::debug!(error = %err, "failed to send pi rpc abort");
                            continue;
                        }
                        let _ = stdin.flush().await;
                    }
                    WriterCommand::Shutdown => {
                        let _ = stdin.write_all(b"{\"type\":\"abort\"}\n").await;
                        let _ = stdin.flush().await;
                        break;
                    }
                }
            }
        });

        // Stderr drain task: forward pi's diagnostics without blocking the child.
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr).lines();
            loop {
                match reader.next_line().await {
                    Ok(Some(line)) => {
                        tracing::debug!(target: "pi_rpc_stderr", "{}", line);
                    }
                    Ok(None) => break,
                    Err(err) => {
                        tracing::debug!(error = %err, "pi rpc stderr read error");
                        break;
                    }
                }
            }
        });

        // Reader task: drives the child stdout JSONL stream into per-request replies.
        // The first turn has no active request, so we just discard events until the
        // first prompt. After that, prompts are serialized via the writer.
        let (event_tx, _) = broadcast::channel::<PiEvent>(64);
        let reader_event_tx = event_tx.clone();
        tokio::spawn(async move {
            reader_loop(stdout, reader_event_tx, config.stream_idle_timeout).await;
        });

        // Coordinator task: dispatches lifecycle events (idle, child exit) to kill the
        // child when the upstream pipe goes silent or the child dies unexpectedly.
        let coordinator_event_rx = event_tx.subscribe();
        tokio::spawn(coordinator_loop(
            coordinator_event_rx,
            writer_tx.clone(),
            child,
            shutdown_rx,
        ));

        Ok(Self {
            meta,
            writer_tx,
            event_tx,
            shutdown_tx: Some(shutdown_tx),
            next_request_id: Arc::new(AtomicU64::new(1)),
        })
    }
}

#[async_trait]
impl LlmSession for PiRpcLlmSession {
    fn chat_stream(&mut self, prompt: String) -> TextStream {
        let (chunk_tx, mut chunk_rx) = mpsc::channel::<Result<String>>(32);
        let mut event_rx = self.event_tx.subscribe();
        let (result_tx, _result_rx) = oneshot::channel::<Result<()>>();

        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let request_id_str = request_id.to_string();

        let writer_tx = self.writer_tx.clone();
        let meta = self.meta.clone();
        let prompt_started = Instant::now();
        let idle_timeout = std::time::Duration::from_secs(60);

        // Dispatch a single prompt and forward events to the result stream.
        tokio::spawn(async move {
            if let Err(err) = writer_tx
                .send(WriterCommand::Prompt {
                    request_id,
                    message: prompt,
                    reply: result_tx,
                })
                .await
            {
                let _ = chunk_tx
                    .send(Err(anyhow!(err).context("dispatch pi rpc prompt")))
                    .await;
                return;
            }

            loop {
                match tokio::time::timeout(idle_timeout, event_rx.recv()).await {
                    Ok(Ok(event)) => {
                        match event {
                            PiEvent::TextDelta(text) => {
                                if chunk_tx.send(Ok(text)).await.is_err() {
                                    tracing::debug!(
                                        request_id = %request_id_str,
                                        "pi rpc text delta consumer dropped"
                                    );
                                    return;
                                }
                            }
                            PiEvent::PromptRejected { reason } => {
                                let _ = chunk_tx
                                    .send(Err(anyhow!("pi rpc prompt rejected: {reason}")))
                                    .await;
                                return;
                            }
                            PiEvent::PromptFailed { reason } => {
                                let _ = chunk_tx
                                    .send(Err(anyhow!("pi rpc prompt failed: {reason}")))
                                    .await;
                                return;
                            }
                            PiEvent::AgentEnd { will_retry } => {
                                if will_retry {
                                    tracing::info!(
                                        session_id = %meta.session_id,
                                        request_id = %request_id_str,
                                        "pi rpc agent_end with will_retry"
                                    );
                                }
                                break;
                            }
                            PiEvent::Idle => {
                                let _ = chunk_tx
                                    .send(Err(anyhow!("pi rpc stdout idle for too long")))
                                    .await;
                                return;
                            }
                            PiEvent::ChildExited(status) => {
                                let _ = chunk_tx
                                    .send(Err(anyhow!(
                                        "pi rpc child exited before agent_end: {status:?}"
                                    )))
                                    .await;
                                return;
                            }
                            PiEvent::PromptStarted { .. }
                            | PiEvent::Compaction { .. }
                            | PiEvent::Retry { .. } => {
                                // Informational; keep reading.
                            }
                        }
                    }
                    Ok(Err(broadcast::error::RecvError::Lagged(_))) => {
                        // Slow consumer; skip stale events.
                        continue;
                    }
                    Ok(Err(_)) => {
                        let _ = chunk_tx
                            .send(Err(anyhow!("pi rpc event channel closed")))
                            .await;
                        return;
                    }
                    Err(_) => {
                        let _ = chunk_tx
                            .send(Err(anyhow!(
                                "pi rpc prompt idle timeout after {:?}",
                                prompt_started.elapsed()
                            )))
                            .await;
                        return;
                    }
                }
            }
        });

        Box::pin(try_stream! {
            while let Some(item) = chunk_rx.recv().await {
                yield item?;
            }
        })
    }

    async fn abort(&mut self) {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let _ = self
            .writer_tx
            .send(WriterCommand::Abort { request_id })
            .await;
    }

    async fn shutdown(&mut self) {
        if let Some(shutdown) = self.shutdown_tx.take() {
            let _ = shutdown.send(());
        }
        let _ = self.writer_tx.send(WriterCommand::Shutdown).await;
    }
}

// -----------------------------------------------------------------------------
// Event protocol
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum PiEvent {
    PromptStarted {
        request_id: String,
    },
    TextDelta(String),
    PromptRejected {
        reason: String,
    },
    PromptFailed {
        reason: String,
    },
    AgentEnd {
        will_retry: bool,
    },
    ChildExited(Option<i32>),
    Idle,
    Compaction {
        kind: String,
        reason: Option<String>,
    },
    Retry {
        kind: String,
        attempt: Option<u64>,
        success: Option<bool>,
    },
}

async fn reader_loop<R>(
    stdout: R,
    event_tx: broadcast::Sender<PiEvent>,
    idle_timeout: std::time::Duration,
) where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    let mut reader = BufReader::new(stdout).lines();
    let mut last_event = Instant::now();
    loop {
        let deadline = last_event + idle_timeout;
        let now = Instant::now();
        let wait = deadline.saturating_duration_since(now);
        let line = match tokio::time::timeout(wait, reader.next_line()).await {
            Ok(Ok(Some(line))) => line,
            Ok(Ok(None)) => {
                let _ = event_tx.send(PiEvent::ChildExited(None));
                return;
            }
            Ok(Err(err)) => {
                tracing::debug!(error = %err, "pi rpc stdout read error");
                return;
            }
            Err(_) => {
                let _ = event_tx.send(PiEvent::Idle);
                return;
            }
        };
        last_event = Instant::now();

        if line.is_empty() {
            continue;
        }

        let value: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(err) => {
                tracing::debug!(error = %err, raw = %line, "ignoring non-JSON pi rpc line");
                continue;
            }
        };

        let Some(kind) = value.get("type").and_then(Value::as_str) else {
            continue;
        };

        match kind {
            "response" => {
                if let Some(success) = value.get("success").and_then(Value::as_bool) {
                    if success {
                        if let Some(id) = value.get("id").and_then(Value::as_str) {
                            let _ = event_tx.send(PiEvent::PromptStarted {
                                request_id: id.to_string(),
                            });
                        }
                    } else {
                        let command = value.get("command").and_then(Value::as_str).unwrap_or("");
                        let reason = value
                            .get("error")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown")
                            .to_string();
                        let _ = event_tx.send(match command {
                            "prompt" => PiEvent::PromptRejected { reason },
                            _ => PiEvent::PromptFailed { reason },
                        });
                    }
                }
            }
            "message_update" => {
                if let Some(event) = value.get("assistantMessageEvent") {
                    match event.get("type").and_then(Value::as_str) {
                        Some("text_delta") => {
                            if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                                let _ = event_tx.send(PiEvent::TextDelta(delta.to_string()));
                            }
                        }
                        Some("thinking_delta") | Some("thinking_start") | Some("thinking_end") => {
                            // Drop thinking deltas; only speak the actual answer.
                        }
                        _ => {}
                    }
                }
            }
            "agent_end" => {
                let will_retry = value
                    .get("willRetry")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let _ = event_tx.send(PiEvent::AgentEnd { will_retry });
            }
            "auto_retry_start" => {
                let _ = event_tx.send(PiEvent::Retry {
                    kind: "start".to_string(),
                    attempt: value.get("attempt").and_then(Value::as_u64),
                    success: None,
                });
            }
            "auto_retry_end" => {
                let _ = event_tx.send(PiEvent::Retry {
                    kind: "end".to_string(),
                    attempt: value.get("attempt").and_then(Value::as_u64),
                    success: value.get("success").and_then(Value::as_bool),
                });
            }
            "compaction_start" => {
                let _ = event_tx.send(PiEvent::Compaction {
                    kind: "start".to_string(),
                    reason: value
                        .get("reason")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                });
            }
            "compaction_end" => {
                let _ = event_tx.send(PiEvent::Compaction {
                    kind: "end".to_string(),
                    reason: value
                        .get("reason")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                });
            }
            "extension_error" => {
                tracing::warn!(
                    extension_path = ?value.get("extensionPath").map(|v| v.to_string()),
                    event = ?value.get("event").map(|v| v.to_string()),
                    error = ?value.get("error").map(|v| v.to_string()),
                    "pi rpc extension error"
                );
            }
            "queue_update" => {
                // Informational; no-op.
            }
            other => {
                tracing::debug!(event = other, "pi rpc event");
            }
        }
    }
}

async fn coordinator_loop(
    mut event_rx: broadcast::Receiver<PiEvent>,
    _writer_tx: mpsc::Sender<WriterCommand>,
    mut child: Child,
    mut shutdown_rx: oneshot::Receiver<()>,
) {
    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown_rx => {
                tracing::info!("pi rpc session shutdown requested");
                let _ = child.start_kill();
                let _ = child.wait().await;
                return;
            }
            event = event_rx.recv() => {
                match event {
                    Ok(PiEvent::ChildExited(status)) => {
                        tracing::warn!(?status, "pi rpc child exited unexpectedly");
                        if let Ok(Some(exit)) = child.try_wait() {
                            tracing::debug!(?exit, "pi rpc child reaped after early exit");
                        }
                        return;
                    }
                    Ok(PiEvent::Idle) => {
                        tracing::warn!("pi rpc stdout idle; killing child");
                        let _ = child.start_kill();
                        let _ = child.wait().await;
                        return;
                    }
                    Ok(_) => continue,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => {
                        let _ = child.start_kill();
                        let _ = child.wait().await;
                        return;
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// env helpers
// -----------------------------------------------------------------------------

fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// Safe default flag set used when `XIAOZHI_PI_RPC_FLAGS` is unset. Disables tool
/// calls, skills, prompt templates, context files, themes, and extension discovery,
/// keeps the run ephemeral, and turns off thinking so the model streams answers
/// only (no `thinking_*` deltas to filter later).
fn default_safe_flags() -> Vec<String> {
    [
        "--no-tools",
        "--no-builtin-tools",
        "--no-skills",
        "--no-prompt-templates",
        "--no-context-files",
        "--no-themes",
        "--no-extensions",
        "--no-session",
        "--thinking",
        "off",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn env_u64(name: &str, default: u64) -> u64 {
    env_str(name)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}
