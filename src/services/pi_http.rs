use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::Value;
use tokio::{sync::mpsc, task::JoinHandle};

use super::{LlmSession, LlmSessionFactory, LlmSessionMeta, TextStream};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8081";
const DEFAULT_AGENT_ID: &str = "zhuzhu";
const DEFAULT_IDLE_TIMEOUT_MS: u64 = 300_000;

#[derive(Clone, Debug)]
pub struct PiHttpLlmConfig {
    pub base_url: String,
    pub agent_id: String,
    pub stream_idle_timeout: Duration,
}

impl PiHttpLlmConfig {
    pub fn from_env() -> Result<Option<Self>> {
        let provider = std::env::var("XIAOZHI_LLM_PROVIDER")
            .or_else(|_| std::env::var("LLM_PROVIDER"))
            .unwrap_or_else(|_| "mock".to_string())
            .to_ascii_lowercase();
        if !matches!(
            provider.as_str(),
            "pi" | "pi-http" | "pi_http" | "pi-server" | "pi_server"
        ) {
            return Ok(None);
        }

        let base_url = env_any(&["XIAOZHI_PI_HTTP_BASE_URL", "PI_SERVER_BASE_URL"])
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let agent_id = env_any(&["XIAOZHI_PI_AGENT_ID", "PI_SERVER_AGENT_ID"])
            .unwrap_or_else(|| DEFAULT_AGENT_ID.to_string());
        if agent_id.trim().is_empty() {
            return Err(anyhow!("pi-server agent id must not be empty"));
        }
        let stream_idle_timeout = Duration::from_millis(env_u64_any(
            &[
                "XIAOZHI_PI_HTTP_IDLE_TIMEOUT_MS",
                "PI_SERVER_IDLE_TIMEOUT_MS",
            ],
            DEFAULT_IDLE_TIMEOUT_MS,
        ));

        Ok(Some(Self {
            base_url,
            agent_id,
            stream_idle_timeout,
        }))
    }

    pub fn chat_url(&self) -> String {
        format!("{}/chat", self.base_url.trim_end_matches('/'))
    }
}

#[derive(Clone, Debug)]
pub struct PiHttpLlmFactory {
    config: PiHttpLlmConfig,
    client: reqwest::Client,
}

impl PiHttpLlmFactory {
    pub fn new(config: PiHttpLlmConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LlmSessionFactory for PiHttpLlmFactory {
    async fn create_session(&self, meta: LlmSessionMeta) -> Result<Box<dyn LlmSession>> {
        Ok(Box::new(PiHttpLlmSession {
            meta,
            config: self.config.clone(),
            client: self.client.clone(),
            active_request: None,
        }))
    }
}

#[derive(Debug)]
struct PiHttpLlmSession {
    meta: LlmSessionMeta,
    config: PiHttpLlmConfig,
    client: reqwest::Client,
    active_request: Option<JoinHandle<()>>,
}

#[async_trait]
impl LlmSession for PiHttpLlmSession {
    fn chat_stream(&mut self, prompt: String) -> TextStream {
        if let Some(handle) = self.active_request.take() {
            handle.abort();
        }

        let (chunk_tx, mut chunk_rx) = mpsc::channel::<Result<String>>(32);
        let client = self.client.clone();
        let config = self.config.clone();
        let meta = self.meta.clone();

        self.active_request = Some(tokio::spawn(async move {
            if let Err(err) = run_chat_request(client, config, meta, prompt, chunk_tx.clone()).await
            {
                let _ = chunk_tx.send(Err(err)).await;
            }
        }));

        Box::pin(try_stream! {
            while let Some(item) = chunk_rx.recv().await {
                yield item?;
            }
        })
    }

    async fn abort(&mut self) {
        if let Some(handle) = self.active_request.take() {
            handle.abort();
        }
    }

    async fn shutdown(&mut self) {
        self.abort().await;
    }
}

async fn run_chat_request(
    client: reqwest::Client,
    config: PiHttpLlmConfig,
    meta: LlmSessionMeta,
    prompt: String,
    chunk_tx: mpsc::Sender<Result<String>>,
) -> Result<()> {
    let url = config.chat_url();
    let started = Instant::now();
    tracing::debug!(
        session_id = %meta.session_id,
        url = %url,
        agent_id = %config.agent_id,
        "starting pi-server LLM stream request"
    );

    let response = client
        .post(&url)
        .bearer_auth(&config.agent_id)
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .json(&serde_json::json!({ "prompt": prompt }))
        .send()
        .await
        .context("send pi-server chat request")?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "pi-server chat failed: status={status}, body={body}"
        ));
    }

    let mut raw_chunks = 0u64;
    let mut text_chunks = 0u64;
    let mut buffer = Vec::<u8>::new();
    let mut parser = SseParser::default();
    let mut saw_agent_end = false;
    let mut stream = response.bytes_stream();

    loop {
        let next = match tokio::time::timeout(config.stream_idle_timeout, stream.next()).await {
            Ok(next) => next,
            Err(_) => {
                return Err(anyhow!(
                    "pi-server stream idle timeout after {:?}",
                    config.stream_idle_timeout
                ));
            }
        };

        let Some(chunk) = next else {
            break;
        };
        let chunk = chunk.context("read pi-server stream chunk")?;
        raw_chunks = raw_chunks.saturating_add(1);
        buffer.extend_from_slice(&chunk);

        while let Some(line) = take_line(&mut buffer) {
            if let Some(event) = parser.push_line(&line)? {
                match event {
                    PiServerEvent::TextDelta(delta) => {
                        if !delta.is_empty() {
                            text_chunks = text_chunks.saturating_add(1);
                            if chunk_tx.send(Ok(delta)).await.is_err() {
                                tracing::debug!(
                                    session_id = %meta.session_id,
                                    "pi-server text delta consumer dropped"
                                );
                                return Ok(());
                            }
                        }
                    }
                    PiServerEvent::AgentEnd => {
                        saw_agent_end = true;
                        break;
                    }
                    PiServerEvent::Error(reason) => {
                        return Err(anyhow!("pi-server chat error: {reason}"));
                    }
                    PiServerEvent::Ignored => {}
                }
            }
        }

        if saw_agent_end {
            break;
        }
    }

    if !buffer.is_empty() {
        if let Some(event) = parser.push_line(&buffer)? {
            match event {
                PiServerEvent::TextDelta(delta) if !delta.is_empty() => {
                    text_chunks = text_chunks.saturating_add(1);
                    if chunk_tx.send(Ok(delta)).await.is_err() {
                        return Ok(());
                    }
                }
                PiServerEvent::AgentEnd => saw_agent_end = true,
                PiServerEvent::Error(reason) => {
                    return Err(anyhow!("pi-server chat error: {reason}"));
                }
                PiServerEvent::TextDelta(_) | PiServerEvent::Ignored => {}
            }
        }
    }
    if let Some(event) = parser.finish()? {
        match event {
            PiServerEvent::TextDelta(delta) if !delta.is_empty() => {
                text_chunks = text_chunks.saturating_add(1);
                if chunk_tx.send(Ok(delta)).await.is_err() {
                    return Ok(());
                }
            }
            PiServerEvent::AgentEnd => saw_agent_end = true,
            PiServerEvent::Error(reason) => {
                return Err(anyhow!("pi-server chat error: {reason}"));
            }
            PiServerEvent::TextDelta(_) | PiServerEvent::Ignored => {}
        }
    }

    if !saw_agent_end {
        return Err(anyhow!("pi-server stream ended before agent_end"));
    }

    tracing::debug!(
        session_id = %meta.session_id,
        elapsed_ms = started.elapsed().as_millis(),
        raw_chunks,
        text_chunks,
        "pi-server LLM stream finished"
    );
    Ok(())
}

#[derive(Default)]
struct SseParser {
    event_name: Option<String>,
    data_lines: Vec<String>,
}

impl SseParser {
    fn push_line(&mut self, line: &[u8]) -> Result<Option<PiServerEvent>> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(b":") {
            return Ok(None);
        }

        let (field, value) = match line.iter().position(|byte| *byte == b':') {
            Some(index) => (&line[..index], trim_single_space(&line[index + 1..])),
            None => (line, &b""[..]),
        };

        match field {
            b"event" => {
                self.event_name = Some(std::str::from_utf8(value)?.to_string());
            }
            b"data" => {
                self.data_lines
                    .push(std::str::from_utf8(value)?.to_string());
            }
            _ => {}
        }
        Ok(None)
    }

    fn finish(&mut self) -> Result<Option<PiServerEvent>> {
        self.dispatch()
    }

    fn dispatch(&mut self) -> Result<Option<PiServerEvent>> {
        if self.event_name.is_none() && self.data_lines.is_empty() {
            return Ok(None);
        }

        let event_name = self
            .event_name
            .take()
            .unwrap_or_else(|| "message".to_string());
        let data = std::mem::take(&mut self.data_lines).join("\n");
        parse_pi_server_event(&event_name, &data).map(Some)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum PiServerEvent {
    TextDelta(String),
    AgentEnd,
    Error(String),
    Ignored,
}

fn parse_pi_server_event(event_name: &str, data: &str) -> Result<PiServerEvent> {
    match event_name {
        "text_delta" => {
            let value: Value = serde_json::from_str(data).context("parse text_delta event")?;
            Ok(PiServerEvent::TextDelta(
                value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            ))
        }
        "agent_end" => Ok(PiServerEvent::AgentEnd),
        "error" => {
            let reason = serde_json::from_str::<Value>(data)
                .ok()
                .and_then(|value| {
                    value
                        .get("message")
                        .or_else(|| value.get("error"))
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .unwrap_or_else(|| data.to_string());
            Ok(PiServerEvent::Error(reason))
        }
        "thinking_start" | "thinking_delta" | "thinking_end" | "message" => {
            Ok(PiServerEvent::Ignored)
        }
        other => {
            tracing::debug!(event = other, "ignoring pi-server SSE event");
            Ok(PiServerEvent::Ignored)
        }
    }
}

fn take_line(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let pos = buffer
        .iter()
        .position(|byte| *byte == b'\n' || *byte == b'\r')?;
    let mut drain_len = pos + 1;
    if buffer[pos] == b'\r' && buffer.get(pos + 1) == Some(&b'\n') {
        drain_len += 1;
    }
    let line = buffer[..pos].to_vec();
    buffer.drain(..drain_len);
    Some(line)
}

fn trim_single_space(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(b" ").unwrap_or(bytes)
}

fn env_any(names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| std::env::var(name).ok().filter(|value| !value.is_empty()))
}

fn env_u64_any(names: &[&str], default: u64) -> u64 {
    env_any(names)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_text_delta_event() {
        assert_eq!(
            parse_pi_server_event("text_delta", r#"{"delta":"你好"}"#).unwrap(),
            PiServerEvent::TextDelta("你好".to_string())
        );
    }

    #[test]
    fn parses_error_message() {
        assert_eq!(
            parse_pi_server_event("error", r#"{"message":"superseded"}"#).unwrap(),
            PiServerEvent::Error("superseded".to_string())
        );
    }

    #[test]
    fn parses_sse_frames() {
        let mut parser = SseParser::default();
        assert!(parser.push_line(b"event: text_delta").unwrap().is_none());
        assert!(
            parser
                .push_line(br#"data: {"delta":"Hi"}"#)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            parser.push_line(b"").unwrap(),
            Some(PiServerEvent::TextDelta("Hi".to_string()))
        );
    }

    #[test]
    fn takes_crlf_lines() {
        let mut buffer = b"event: agent_end\r\ndata: {}\r\n\r\n".to_vec();
        assert_eq!(take_line(&mut buffer).unwrap(), b"event: agent_end");
        assert_eq!(take_line(&mut buffer).unwrap(), b"data: {}");
        assert_eq!(take_line(&mut buffer).unwrap(), b"");
        assert!(buffer.is_empty());
    }
}
