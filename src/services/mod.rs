pub mod mock;
pub mod openai;
pub mod pi_http;
pub mod volcengine;
pub mod volcengine_asr;

use std::{pin::Pin, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use futures_util::Stream;

use crate::{
    audio::silero_vad::{SileroVadConfig, SileroVadService},
    protocol::AudioFrame,
    speaker_id::{SpeakerIdConfig, SpeakerIdService},
};

use self::{
    mock::{MockAsr, MockLlmFactory, MockTts},
    openai::{OpenAiLlmConfig, OpenAiLlmFactory},
    pi_http::{PiHttpLlmConfig, PiHttpLlmFactory},
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
    pub agent_id: Option<String>,
    pub speaker_name: Option<String>,
}

#[async_trait]
pub trait LlmSession: Send + 'static {
    /// Begin a chat completion and return a stream of text deltas.
    ///
    /// **Implementor contract (do not violate):** this method MUST be
    /// synchronous (no `.await`) and MUST NOT be invoked while holding an
    /// `Arc<Mutex<LlmSession>>` across an `await` point. The session
    /// runtime relies on being able to call `abort()` at any time, which
    /// requires being able to take the `LlmSession` lock without
    /// contending with a pipeline task that is still parked inside
    /// `chat_stream`. Implementations are expected to return a fresh
    /// `TextStream` (typically backed by an internal channel) and
    /// immediately return, with the actual LLM work driven by a
    /// background task that has already been moved out of `&mut self`.
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
    pub speaker_id: Option<Arc<SpeakerIdService>>,
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

        let speaker_id = match SpeakerIdConfig::from_env()? {
            Some(config) => {
                tracing::info!(
                    model_path = %config.model_path.display(),
                    db_dir = %config.db_dir.display(),
                    default_agent_id = %config.default_agent_id,
                    min_similarity = config.min_similarity,
                    "using speaker identification"
                );
                Some(Arc::new(SpeakerIdService::new(config)?))
            }
            None => {
                tracing::info!("speaker identification disabled");
                None
            }
        };

        let llm: Arc<dyn LlmSessionFactory> = if let Some(config) = PiHttpLlmConfig::from_env()? {
            tracing::info!(
                base_url = %config.base_url,
                agent_id = %config.agent_id,
                idle_timeout_ms = config.stream_idle_timeout.as_millis(),
                "using pi-server HTTP streaming llm"
            );
            Arc::new(PiHttpLlmFactory::new(config))
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

        Ok(Self {
            asr,
            llm,
            tts,
            vad,
            speaker_id,
        })
    }
}
