use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use async_stream::try_stream;
use futures_util::StreamExt;
use serde_json::{Value, json};

use async_trait::async_trait;

use super::{LlmSession, LlmSessionFactory, LlmSessionMeta, TextStream};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-4o-mini";

#[derive(Clone, Debug)]
pub struct OpenAiLlmConfig {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub disable_thinking: bool,
    pub thinking_style: ThinkingStyle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingStyle {
    Auto,
    DeepSeek,
    EnableThinking,
    Both,
    None,
}

impl ThinkingStyle {
    fn from_env() -> Self {
        match env_any(&["XIAOZHI_LLM_THINKING_STYLE", "OPENAI_THINKING_STYLE"])
            .unwrap_or_else(|| "auto".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "deepseek" | "thinking" => Self::DeepSeek,
            "enable_thinking" | "enable-thinking" | "qwen" | "dashscope" | "aliyun" => {
                Self::EnableThinking
            }
            "both" | "all" => Self::Both,
            "none" | "off" | "disabled" => Self::None,
            _ => Self::Auto,
        }
    }
}

impl OpenAiLlmConfig {
    pub fn from_env() -> Result<Option<Self>> {
        let provider = std::env::var("XIAOZHI_LLM_PROVIDER")
            .or_else(|_| std::env::var("LLM_PROVIDER"))
            .unwrap_or_else(|_| "mock".to_string())
            .to_ascii_lowercase();
        if provider != "openai"
            && provider != "openai-compatible"
            && provider != "openai_compatible"
        {
            return Ok(None);
        }

        let model = env_any(&["XIAOZHI_LLM_MODEL", "OPENAI_MODEL"])
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let base_url = env_any(&["XIAOZHI_LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"])
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let thinking_style = ThinkingStyle::from_env();
        let disable_thinking =
            env_bool_any(&["XIAOZHI_LLM_DISABLE_THINKING", "OPENAI_DISABLE_THINKING"])
                .unwrap_or(true);

        Ok(Some(Self {
            api_key: required_env_any(&["XIAOZHI_LLM_API_KEY", "OPENAI_API_KEY"])
                .context("load OpenAI-compatible LLM api key")?,
            model,
            base_url,
            system_prompt: env_any(&["XIAOZHI_LLM_SYSTEM_PROMPT", "OPENAI_SYSTEM_PROMPT"]),
            temperature: env_f32_any(&["XIAOZHI_LLM_TEMPERATURE", "OPENAI_TEMPERATURE"]),
            max_tokens: env_u32_any(&["XIAOZHI_LLM_MAX_TOKENS", "OPENAI_MAX_TOKENS"]),
            disable_thinking,
            thinking_style,
        }))
    }

    pub fn chat_completions_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if base.ends_with("/chat/completions") {
            base.to_string()
        } else {
            format!("{base}/chat/completions")
        }
    }
}

#[derive(Clone, Debug)]
pub struct OpenAiLlmFactory {
    config: OpenAiLlmConfig,
    client: reqwest::Client,
}

impl OpenAiLlmFactory {
    pub fn new(config: OpenAiLlmConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LlmSessionFactory for OpenAiLlmFactory {
    async fn create_session(&self, _session_meta: LlmSessionMeta) -> Result<Box<dyn LlmSession>> {
        Ok(Box::new(OpenAiLlmSession {
            config: self.config.clone(),
            client: self.client.clone(),
        }))
    }
}

#[derive(Clone, Debug)]
struct OpenAiLlmSession {
    config: OpenAiLlmConfig,
    client: reqwest::Client,
}

#[async_trait]
impl LlmSession for OpenAiLlmSession {
    fn chat_stream(&mut self, prompt: String) -> TextStream {
        let config = self.config.clone();
        let client = self.client.clone();

        Box::pin(try_stream! {
            let mut messages = Vec::new();
            if let Some(system_prompt) = config.system_prompt.as_deref().filter(|s| !s.is_empty()) {
                messages.push(json!({"role": "system", "content": system_prompt}));
            }
            messages.push(json!({"role": "user", "content": prompt}));

            let mut body = json!({
                "model": config.model,
                "messages": messages,
                "stream": true,
            });
            if let Some(temperature) = config.temperature {
                body["temperature"] = json!(temperature);
            }
            if let Some(max_tokens) = config.max_tokens {
                body["max_tokens"] = json!(max_tokens);
            }
            apply_thinking_config(&mut body, &config);

            let url = config.chat_completions_url();
            let request_started = Instant::now();
            tracing::debug!(
                %url,
                model = %config.model,
                disable_thinking = config.disable_thinking,
                thinking_style = ?effective_thinking_style(&config),
                "starting OpenAI-compatible LLM stream request"
            );
            let response = client
                .post(&url)
                .bearer_auth(&config.api_key)
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .json(&body)
                .send()
                .await
                .context("send OpenAI-compatible chat completions request")?;

            let status = response.status();
            tracing::debug!(
                status = status.as_u16(),
                elapsed_ms = request_started.elapsed().as_millis(),
                "OpenAI-compatible LLM response headers received"
            );
            if status.is_success() {
                let mut stream = response.bytes_stream();
                let mut buffer = Vec::<u8>::new();
                let mut raw_chunks = 0u64;
                let mut content_chunks = 0u64;
                let mut content_chars = 0usize;
                let mut first_raw_chunk_logged = false;
                let mut first_content_logged = false;

                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.context("read OpenAI-compatible stream chunk")?;
                    raw_chunks = raw_chunks.saturating_add(1);
                    if !first_raw_chunk_logged {
                        first_raw_chunk_logged = true;
                        tracing::debug!(
                            elapsed_ms = request_started.elapsed().as_millis(),
                            bytes = chunk.len(),
                            "first OpenAI-compatible LLM raw stream chunk received"
                        );
                    }
                    buffer.extend_from_slice(&chunk);

                    while let Some(line_end) = find_line_end(&buffer) {
                        let mut line = buffer.drain(..line_end).collect::<Vec<u8>>();
                        if buffer.first() == Some(&b'\n') {
                            buffer.drain(..1);
                        }
                        if line.last() == Some(&b'\r') {
                            line.pop();
                        }

                        if let Some(content) = parse_sse_line(&line)? {
                            if !content.is_empty() {
                                let chunk_chars = content.chars().count();
                                content_chunks = content_chunks.saturating_add(1);
                                content_chars = content_chars.saturating_add(chunk_chars);
                                if !first_content_logged {
                                    first_content_logged = true;
                                    tracing::debug!(
                                        elapsed_ms = request_started.elapsed().as_millis(),
                                        chunk_chars,
                                        "first OpenAI-compatible LLM content delta received"
                                    );
                                }
                                yield content;
                            }
                        }
                    }
                }

                if !buffer.is_empty() {
                    if let Some(content) = parse_sse_line(&buffer)? {
                        if !content.is_empty() {
                            let chunk_chars = content.chars().count();
                            content_chunks = content_chunks.saturating_add(1);
                            content_chars = content_chars.saturating_add(chunk_chars);
                            if !first_content_logged {
                                tracing::debug!(
                                    elapsed_ms = request_started.elapsed().as_millis(),
                                    chunk_chars,
                                    "first OpenAI-compatible LLM content delta received"
                                );
                            }
                            yield content;
                        }
                    }
                }

                tracing::debug!(
                    elapsed_ms = request_started.elapsed().as_millis(),
                    raw_chunks,
                    content_chunks,
                    content_chars,
                    "OpenAI-compatible LLM stream finished"
                );
            } else {
                let text = response.text().await.unwrap_or_default();
                Err(anyhow!("OpenAI-compatible chat completions failed: status={status}, body={text}"))?;
            }
        })
    }

    async fn abort(&mut self) {}

    async fn shutdown(&mut self) {}
}

fn apply_thinking_config(body: &mut Value, config: &OpenAiLlmConfig) {
    if !config.disable_thinking {
        return;
    }

    match effective_thinking_style(config) {
        ThinkingStyle::DeepSeek => {
            body["thinking"] = json!({"type": "disabled"});
        }
        ThinkingStyle::EnableThinking => {
            body["enable_thinking"] = json!(false);
        }
        ThinkingStyle::Both => {
            body["thinking"] = json!({"type": "disabled"});
            body["enable_thinking"] = json!(false);
        }
        ThinkingStyle::None | ThinkingStyle::Auto => {}
    }
}

fn effective_thinking_style(config: &OpenAiLlmConfig) -> ThinkingStyle {
    match config.thinking_style {
        ThinkingStyle::Auto => {
            let marker = format!(
                "{} {}",
                config.base_url.to_ascii_lowercase(),
                config.model.to_ascii_lowercase()
            );
            if marker.contains("deepseek") {
                ThinkingStyle::DeepSeek
            } else if marker.contains("qwen")
                || marker.contains("dashscope")
                || marker.contains("aliyun")
                || marker.contains("alibaba")
            {
                ThinkingStyle::EnableThinking
            } else {
                ThinkingStyle::Both
            }
        }
        style => style,
    }
}

fn parse_sse_line(line: &[u8]) -> Result<Option<String>> {
    let line = trim_ascii_start(line);
    if line.is_empty() || line.starts_with(b":") {
        return Ok(None);
    }

    let Some(data) = line.strip_prefix(b"data:") else {
        return Ok(None);
    };
    let data = trim_ascii_start(data);
    if data == b"[DONE]" {
        return Ok(None);
    }

    let value: Value = serde_json::from_slice(data).context("parse OpenAI-compatible SSE data")?;
    Ok(extract_delta_content(&value))
}

fn extract_delta_content(value: &Value) -> Option<String> {
    let choices = value.get("choices")?.as_array()?;
    for choice in choices {
        let Some(delta) = choice.get("delta") else {
            continue;
        };
        if let Some(content) = delta.get("content") {
            if let Some(text) = content.as_str() {
                return Some(text.to_string());
            }
            if let Some(parts) = content.as_array() {
                let text = parts
                    .iter()
                    .filter_map(|part| {
                        part.as_str()
                            .or_else(|| part.get("text").and_then(|text| text.as_str()))
                    })
                    .collect::<String>();
                if !text.is_empty() {
                    return Some(text);
                }
            }
        }
    }
    None
}

fn find_line_end(buffer: &[u8]) -> Option<usize> {
    buffer
        .iter()
        .position(|byte| *byte == b'\n' || *byte == b'\r')
}

fn trim_ascii_start(mut bytes: &[u8]) -> &[u8] {
    while matches!(bytes.first(), Some(b' ' | b'\t')) {
        bytes = &bytes[1..];
    }
    bytes
}

fn required_env_any(names: &[&str]) -> Result<String> {
    env_any(names)
        .ok_or_else(|| anyhow!("missing required env; expected one of {}", names.join(", ")))
}

fn env_any(names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| std::env::var(name).ok().filter(|s| !s.is_empty()))
}

fn env_f32_any(names: &[&str]) -> Option<f32> {
    env_any(names).and_then(|s| s.parse::<f32>().ok())
}

fn env_u32_any(names: &[&str]) -> Option<u32> {
    env_any(names).and_then(|s| s.parse::<u32>().ok())
}

fn env_bool_any(names: &[&str]) -> Option<bool> {
    env_any(names).and_then(|s| match s.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_content_delta() {
        let line =
            r#"data: {"choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}"#;
        assert_eq!(
            parse_sse_line(line.as_bytes()).unwrap().as_deref(),
            Some("你好")
        );
    }

    #[test]
    fn ignores_done_marker() {
        assert_eq!(parse_sse_line(b"data: [DONE]").unwrap(), None);
    }

    #[test]
    fn builds_chat_completions_url() {
        let config = test_config(
            "model",
            "https://example.com/v1/",
            false,
            ThinkingStyle::None,
        );
        assert_eq!(
            config.chat_completions_url(),
            "https://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn disables_deepseek_thinking() {
        let config = test_config(
            "deepseek-v4-flash",
            "https://api.deepseek.com",
            true,
            ThinkingStyle::Auto,
        );
        let mut body = json!({"model": config.model, "messages": [], "stream": true});
        apply_thinking_config(&mut body, &config);
        assert_eq!(body["thinking"], json!({"type": "disabled"}));
        assert!(body.get("enable_thinking").is_none());
    }

    #[test]
    fn disables_enable_thinking_style() {
        let config = test_config(
            "qwen-plus",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            true,
            ThinkingStyle::Auto,
        );
        let mut body = json!({"model": config.model, "messages": [], "stream": true});
        apply_thinking_config(&mut body, &config);
        assert_eq!(body["enable_thinking"], json!(false));
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn disables_unknown_provider_with_both_styles() {
        let config = test_config(
            "some-model",
            "https://example.com/v1",
            true,
            ThinkingStyle::Auto,
        );
        let mut body = json!({"model": config.model, "messages": [], "stream": true});
        apply_thinking_config(&mut body, &config);
        assert_eq!(body["thinking"], json!({"type": "disabled"}));
        assert_eq!(body["enable_thinking"], json!(false));
    }

    fn test_config(
        model: &str,
        base_url: &str,
        disable_thinking: bool,
        thinking_style: ThinkingStyle,
    ) -> OpenAiLlmConfig {
        OpenAiLlmConfig {
            api_key: "key".to_string(),
            model: model.to_string(),
            base_url: base_url.to_string(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            disable_thinking,
            thinking_style,
        }
    }
}
