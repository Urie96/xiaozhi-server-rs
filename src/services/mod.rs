pub mod mock;

use std::pin::Pin;

use anyhow::Result;
use async_trait::async_trait;
use futures_util::Stream;

use crate::protocol::AudioFrame;

pub type TextStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;
pub type TtsStream = Pin<Box<dyn Stream<Item = Result<TtsEvent>> + Send>>;

#[async_trait]
pub trait AsrService: Send + Sync + 'static {
    async fn recognize(&self, frames: &[AudioFrame]) -> Result<String>;
}

pub trait LlmService: Send + Sync + 'static {
    fn chat_stream(&self, prompt: String) -> TextStream;
}

pub trait TtsService: Send + Sync + 'static {
    fn synthesize_stream(&self, input: TextStream) -> TtsStream;
}

#[derive(Debug, Clone)]
pub enum TtsEvent {
    SentenceStart(String),
    Audio(AudioFrame),
}
