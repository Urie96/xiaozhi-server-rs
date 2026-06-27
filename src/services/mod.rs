pub mod mock;
pub mod volcengine;
pub mod volcengine_asr;

use std::{pin::Pin, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use futures_util::Stream;

use crate::protocol::AudioFrame;

use self::{
    mock::{MockAsr, MockLlm, MockTts},
    volcengine::{VolcengineTts, VolcengineTtsConfig},
    volcengine_asr::{VolcengineAsr, VolcengineAsrConfig},
};

pub type TextStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;
pub type TtsStream = Pin<Box<dyn Stream<Item = Result<TtsEvent>> + Send>>;

#[async_trait]
pub trait AsrService: Send + Sync + 'static {
    async fn start_stream(&self) -> Result<Box<dyn AsrStream>>;
}

#[async_trait]
pub trait AsrStream: Send + 'static {
    async fn push_audio(&mut self, frame: AudioFrame) -> Result<()>;
    async fn finish(&mut self) -> Result<String>;
    async fn abort(&mut self);
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

#[derive(Clone)]
pub struct ServiceBundle {
    pub asr: Arc<dyn AsrService>,
    pub llm: Arc<dyn LlmService>,
    pub tts: Arc<dyn TtsService>,
}

impl ServiceBundle {
    pub fn from_env() -> Result<Self> {
        let tts: Arc<dyn TtsService> = match VolcengineTtsConfig::from_env()? {
            Some(config) => {
                tracing::info!(
                    endpoint = %config.endpoint,
                    resource_id = %config.resource_id,
                    voice_type = %config.voice_type,
                    encoding = ?config.encoding,
                    prebuffer_ms = config.prebuffer_ms,
                    "using volcengine bidirectional tts"
                );
                Arc::new(VolcengineTts::new(config))
            }
            None => {
                tracing::info!("using mock tts");
                Arc::new(MockTts::default())
            }
        };

        let asr: Arc<dyn AsrService> = match VolcengineAsrConfig::from_env()? {
            Some(config) => {
                tracing::info!(
                    endpoint = %config.endpoint,
                    resource_id = %config.resource_id,
                    language = %config.language,
                    chunk_ms = config.chunk_ms,
                    "using volcengine streaming asr"
                );
                Arc::new(VolcengineAsr::new(config))
            }
            None => {
                tracing::info!("using mock asr");
                Arc::new(MockAsr)
            }
        };

        Ok(Self {
            asr,
            llm: Arc::new(MockLlm),
            tts,
        })
    }
}
