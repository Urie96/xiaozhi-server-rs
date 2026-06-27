pub mod mock;
pub mod openai;
pub mod pi_rpc;
pub mod volcengine;
pub mod volcengine_asr;

use std::{pin::Pin, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use futures_util::Stream;

use crate::{
    audio::silero_vad::{SileroVadConfig, SileroVadService},
    protocol::AudioFrame,
};

use self::{
    mock::{MockAsr, MockLlmFactory, MockTts},
    openai::{OpenAiLlmConfig, OpenAiLlmFactory},
    pi_rpc::{PiRpcLlmFactory, PiRpcLlmFactoryConfig},
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
    async fn push_pcm(&mut self, samples: &[i16]) -> Result<()>;
    async fn finish(&mut self) -> Result<String>;
    async fn abort(&mut self);
}

#[async_trait]
pub trait LlmSessionFactory: Send + Sync + 'static {
    async fn create_session(&self, session_meta: LlmSessionMeta) -> Result<Box<dyn LlmSession>>;
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct LlmSessionMeta {
    pub session_id: String,
    pub device_id: Option<String>,
    pub client_id: Option<String>,
}

#[async_trait]
pub trait LlmSession: Send + 'static {
    fn chat_stream(&mut self, prompt: String) -> TextStream;
    async fn abort(&mut self);
    async fn shutdown(&mut self);
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
    pub llm: Arc<dyn LlmSessionFactory>,
    pub tts: Arc<dyn TtsService>,
    pub vad: Option<Arc<SileroVadService>>,
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

        let vad = match SileroVadConfig::from_env()? {
            Some(config) => {
                tracing::info!(
                    model_path = %config.model_path.display(),
                    threshold = config.threshold,
                    min_silence_ms = config.min_silence_duration_ms,
                    min_speech_ms = config.min_speech_duration_ms,
                    max_speech_seconds = config.max_speech_duration_s,
                    "using Silero VAD"
                );
                Some(Arc::new(SileroVadService::new(config)))
            }
            None => {
                tracing::info!("VAD disabled; using timeout/manual stop only");
                None
            }
        };

        let llm: Arc<dyn LlmSessionFactory> =
            if let Some(config) = PiRpcLlmFactoryConfig::from_env()? {
                let rendered = std::iter::once(config.command.clone())
                    .chain(config.args.iter().cloned())
                    .map(|arg| {
                        if arg.chars().all(|c| {
                            c.is_ascii_alphanumeric()
                                || matches!(c, '_' | '-' | '.' | '/' | '=' | ':' | ',')
                        }) {
                            arg
                        } else {
                            format!("'{}'", arg.replace('\'', "'\\''"))
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                tracing::info!(
                    command = %config.command,
                    cwd = ?config.cwd,
                    rendered_args = %rendered,
                    "using pi rpc llm"
                );
                Arc::new(PiRpcLlmFactory::new(config))
            } else if let Some(config) = OpenAiLlmConfig::from_env()? {
                tracing::info!(
                    base_url = %config.base_url,
                    model = %config.model,
                    disable_thinking = config.disable_thinking,
                    thinking_style = ?config.thinking_style,
                    "using OpenAI-compatible streaming llm"
                );
                Arc::new(OpenAiLlmFactory::new(config))
            } else {
                tracing::info!("using mock llm");
                Arc::new(MockLlmFactory)
            };

        Ok(Self { asr, llm, tts, vad })
    }
}
