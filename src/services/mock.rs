use std::time::Duration;

use anyhow::Result;
use async_stream::try_stream;
use async_trait::async_trait;
use futures_util::{StreamExt, stream};
use tokio::time::sleep;

use crate::{
    audio::opus_silence,
    protocol::{AudioFrame, SERVER_FRAME_DURATION_MS},
};

use super::{
    AsrService, AsrStream, LlmSession, LlmSessionFactory, LlmSessionMeta, TextStream, TtsEvent,
    TtsService, TtsStream,
};

#[derive(Clone, Debug, Default)]
pub struct MockAsr;

#[async_trait]
impl AsrService for MockAsr {
    async fn start_stream(&self) -> Result<Box<dyn AsrStream>> {
        Ok(Box::new(MockAsrStream::default()))
    }
}

#[derive(Debug, Default)]
struct MockAsrStream {
    frames: usize,
}

#[async_trait]
impl AsrStream for MockAsrStream {
    async fn push_pcm(&mut self, _samples: &[i16]) -> Result<()> {
        self.frames += 1;
        Ok(())
    }

    async fn finish(&mut self) -> Result<String> {
        tracing::info!(frames = self.frames, "mock asr recognized utterance");
        Ok("你好小智".to_string())
    }

    async fn abort(&mut self) {
        tracing::debug!(frames = self.frames, "mock asr stream aborted");
    }
}

#[derive(Clone, Debug, Default)]
pub struct MockLlmFactory;

#[async_trait]
impl LlmSessionFactory for MockLlmFactory {
    async fn create_session(&self, _session_meta: LlmSessionMeta) -> Result<Box<dyn LlmSession>> {
        Ok(Box::new(MockLlmSession))
    }
}

#[derive(Debug)]
struct MockLlmSession;

#[async_trait]
impl LlmSession for MockLlmSession {
    fn chat_stream(&mut self, prompt: String) -> TextStream {
        let chunks = vec![
            "你好，".to_string(),
            "我是 Rust 版小智服务端。".to_string(),
            format!("我已经收到你的消息：{prompt}。"),
            "这是一个 mock 回复。".to_string(),
        ];

        Box::pin(stream::iter(chunks).then(|chunk| async move {
            sleep(Duration::from_millis(180)).await;
            Ok(chunk)
        }))
    }

    async fn abort(&mut self) {}

    async fn shutdown(&mut self) {}
}

#[derive(Clone, Debug)]
pub struct MockTts {
    frames_per_text_chunk: usize,
}

impl Default for MockTts {
    fn default() -> Self {
        Self {
            frames_per_text_chunk: 6,
        }
    }
}

impl TtsService for MockTts {
    fn synthesize_stream(&self, mut input: TextStream) -> TtsStream {
        let frames_per_text_chunk = self.frames_per_text_chunk;

        Box::pin(try_stream! {
            let mut timestamp = 0u32;
            let mut announced = false;
            let mut subtitle = String::new();

            while let Some(chunk) = input.next().await {
                let chunk = chunk?;
                if chunk.is_empty() {
                    continue;
                }

                subtitle.push_str(&chunk);
                if !announced {
                    announced = true;
                    yield TtsEvent::SentenceStart(subtitle.clone());
                }

                for _ in 0..frames_per_text_chunk {
                    yield TtsEvent::Audio(AudioFrame {
                        timestamp,
                        payload: opus_silence::packet(),
                    });
                    timestamp = timestamp.saturating_add(SERVER_FRAME_DURATION_MS);
                    sleep(Duration::from_millis(SERVER_FRAME_DURATION_MS as u64)).await;
                }
            }
        })
    }
}
