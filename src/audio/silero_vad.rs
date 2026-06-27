use std::{mem::take, path::PathBuf};

use anyhow::{Context, Result, bail};
use ndarray::{Array, Array1, Array2, ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr};
use ort::{session::Session, value::Value};

const DEFAULT_SAMPLE_RATE: usize = 16_000;
const DEFAULT_FRAME_SIZE_MS: usize = 32;
const DEFAULT_MODEL_PATH: &str = "models/silero_vad.onnx";
const FALLBACK_MODEL_PATH: &str = "/home/urie/temp/silero-vad/src/silero_vad/data/silero_vad.onnx";

#[derive(Clone, Debug)]
pub struct SileroVadConfig {
    pub model_path: PathBuf,
    pub sample_rate: usize,
    pub frame_size_ms: usize,
    pub threshold: f32,
    pub min_silence_duration_ms: usize,
    pub speech_pad_ms: usize,
    pub min_speech_duration_ms: usize,
    pub max_speech_duration_s: f32,
}

impl SileroVadConfig {
    pub fn from_env() -> Result<Option<Self>> {
        let provider = env_or("XIAOZHI_VAD_PROVIDER", "silero").to_ascii_lowercase();
        if matches!(
            provider.as_str(),
            "none" | "off" | "disabled" | "false" | "0"
        ) {
            return Ok(None);
        }
        if provider != "silero" {
            bail!("unsupported XIAOZHI_VAD_PROVIDER={provider}; expected silero or none");
        }

        let configured_path = std::env::var("SILERO_VAD_MODEL_PATH")
            .or_else(|_| std::env::var("XIAOZHI_VAD_MODEL_PATH"))
            .ok()
            .map(PathBuf::from);
        let model_path = configured_path.unwrap_or_else(default_model_path);

        Ok(Some(Self {
            model_path,
            sample_rate: DEFAULT_SAMPLE_RATE,
            frame_size_ms: env_usize("XIAOZHI_VAD_FRAME_SIZE_MS", DEFAULT_FRAME_SIZE_MS),
            threshold: env_f32("XIAOZHI_VAD_THRESHOLD", 0.5),
            min_silence_duration_ms: env_usize("XIAOZHI_VAD_MIN_SILENCE_MS", 600),
            speech_pad_ms: env_usize("XIAOZHI_VAD_SPEECH_PAD_MS", 64),
            min_speech_duration_ms: env_usize("XIAOZHI_VAD_MIN_SPEECH_MS", 160),
            max_speech_duration_s: env_f32("XIAOZHI_VAD_MAX_SPEECH_SECONDS", 15.0),
        }))
    }
}

#[derive(Clone, Debug)]
pub struct SileroVadService {
    config: SileroVadConfig,
}

impl SileroVadService {
    pub fn new(config: SileroVadConfig) -> Self {
        Self { config }
    }

    pub fn start_stream(&self) -> Result<SileroVadStream> {
        SileroVadStream::new(self.config.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    SpeechStart {
        sample: usize,
    },
    SpeechEnd {
        start_sample: usize,
        end_sample: usize,
    },
}

pub struct SileroVadStream {
    silero: Silero,
    params: VadParams,
    state: VadState,
    pending: Vec<i16>,
    speech_started_logged: bool,
}

impl SileroVadStream {
    pub fn new(config: SileroVadConfig) -> Result<Self> {
        if config.sample_rate != 16_000 && config.sample_rate != 8_000 {
            bail!(
                "Silero VAD only supports 8kHz or 16kHz, got {}",
                config.sample_rate
            );
        }
        if config.frame_size_ms == 0 {
            bail!("Silero VAD frame size must be positive");
        }

        let silero = Silero::new(config.sample_rate, &config.model_path)
            .with_context(|| format!("load Silero VAD model {}", config.model_path.display()))?;
        Ok(Self {
            silero,
            params: VadParams::from_config(&config),
            state: VadState::default(),
            pending: Vec::new(),
            speech_started_logged: false,
        })
    }

    pub fn accept_pcm(&mut self, samples: &[i16]) -> Result<Vec<VadEvent>> {
        self.pending.extend_from_slice(samples);
        let mut events = Vec::new();

        while self.pending.len() >= self.params.frame_size_samples {
            let frame: Vec<i16> = self
                .pending
                .drain(..self.params.frame_size_samples)
                .collect();
            let speech_prob = self.silero.calc_level(&frame)?;
            if let Some(event) = self.state.update(&self.params, speech_prob) {
                if matches!(event, VadEvent::SpeechStart { .. }) {
                    self.speech_started_logged = true;
                }
                events.push(event);
            }
        }

        Ok(events)
    }
}

struct Silero {
    session: Session,
    sample_rate: ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>>,
    state: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    context: Array1<f32>,
    context_size: usize,
}

impl Silero {
    fn new(sample_rate: usize, model_path: &PathBuf) -> Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        let state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
        let context_size = if sample_rate == 16_000 { 64 } else { 32 };
        let context = Array1::<f32>::zeros(context_size);
        let sample_rate = Array::from_shape_vec([1], vec![sample_rate as i64]).unwrap();
        Ok(Self {
            session,
            sample_rate,
            state,
            context,
            context_size,
        })
    }

    fn calc_level(&mut self, audio_frame: &[i16]) -> Result<f32> {
        let data = audio_frame
            .iter()
            .map(|x| (*x as f32) / (i16::MAX as f32))
            .collect::<Vec<_>>();

        let mut input_with_context = Vec::with_capacity(self.context_size + data.len());
        input_with_context.extend_from_slice(self.context.as_slice().unwrap());
        input_with_context.extend_from_slice(&data);

        let frame =
            Array2::<f32>::from_shape_vec([1, input_with_context.len()], input_with_context)
                .unwrap();

        let frame_value = Value::from_array(frame)?;
        let state_value = Value::from_array(take(&mut self.state))?;
        let sr_value = Value::from_array(self.sample_rate.clone())?;

        let res = self.session.run([
            (&frame_value).into(),
            (&state_value).into(),
            (&sr_value).into(),
        ])?;

        let (shape, state_data) = res["stateN"].try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = shape.as_ref().iter().map(|&d| d as usize).collect();
        self.state = ArrayD::from_shape_vec(shape_usize.as_slice(), state_data.to_vec()).unwrap();

        if data.len() >= self.context_size {
            self.context = Array1::from_vec(data[data.len() - self.context_size..].to_vec());
        }

        let prob = *res["output"]
            .try_extract_tensor::<f32>()?
            .1
            .first()
            .context("missing Silero VAD output probability")?;
        Ok(prob)
    }
}

#[derive(Debug)]
struct VadParams {
    threshold: f32,
    frame_size_samples: usize,
    min_speech_samples: usize,
    max_speech_samples: f32,
    min_silence_samples: usize,
    min_silence_samples_at_max_speech: usize,
}

impl VadParams {
    fn from_config(config: &SileroVadConfig) -> Self {
        let sr_per_ms = config.sample_rate / 1000;
        let frame_size_samples = config.frame_size_ms * sr_per_ms;
        let min_speech_samples = sr_per_ms * config.min_speech_duration_ms;
        let speech_pad_samples = sr_per_ms * config.speech_pad_ms;
        let max_speech_samples = config.sample_rate as f32 * config.max_speech_duration_s
            - frame_size_samples as f32
            - 2.0 * speech_pad_samples as f32;
        let min_silence_samples = sr_per_ms * config.min_silence_duration_ms;
        let min_silence_samples_at_max_speech = sr_per_ms * 98;
        Self {
            threshold: config.threshold,
            frame_size_samples,
            min_speech_samples,
            max_speech_samples,
            min_silence_samples,
            min_silence_samples_at_max_speech,
        }
    }
}

#[derive(Debug, Default)]
struct VadState {
    current_sample: usize,
    temp_end: usize,
    next_start: usize,
    prev_end: usize,
    triggered: bool,
    current_start: usize,
}

impl VadState {
    fn update(&mut self, params: &VadParams, speech_prob: f32) -> Option<VadEvent> {
        self.current_sample = self
            .current_sample
            .saturating_add(params.frame_size_samples);

        if speech_prob > params.threshold {
            if self.temp_end != 0 {
                self.temp_end = 0;
                if self.next_start < self.prev_end {
                    self.next_start = self
                        .current_sample
                        .saturating_sub(params.frame_size_samples);
                }
            }
            if !self.triggered {
                self.triggered = true;
                self.current_start = self
                    .current_sample
                    .saturating_sub(params.frame_size_samples);
                return Some(VadEvent::SpeechStart {
                    sample: self.current_start,
                });
            }
            return None;
        }

        if self.triggered
            && (self.current_sample.saturating_sub(self.current_start)) as f32
                > params.max_speech_samples
        {
            let end = if self.prev_end > 0 {
                self.prev_end
            } else {
                self.current_sample
            };
            return self.finish_speech(end, params);
        }

        if self.triggered && speech_prob < (params.threshold - 0.15) {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if self.current_sample.saturating_sub(self.temp_end)
                > params.min_silence_samples_at_max_speech
            {
                self.prev_end = self.temp_end;
            }
            if self.current_sample.saturating_sub(self.temp_end) >= params.min_silence_samples {
                return self.finish_speech(self.temp_end, params);
            }
        }

        None
    }

    fn finish_speech(&mut self, end: usize, params: &VadParams) -> Option<VadEvent> {
        let start = self.current_start;
        self.prev_end = 0;
        self.next_start = 0;
        self.temp_end = 0;
        self.triggered = false;
        self.current_start = 0;

        if end.saturating_sub(start) > params.min_speech_samples {
            Some(VadEvent::SpeechEnd {
                start_sample: start,
                end_sample: end,
            })
        } else {
            None
        }
    }
}

fn default_model_path() -> PathBuf {
    let project_path = PathBuf::from(DEFAULT_MODEL_PATH);
    if project_path.exists() {
        project_path
    } else {
        PathBuf::from(FALLBACK_MODEL_PATH)
    }
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vad_state_emits_start_and_end_after_enough_silence() {
        let params = VadParams {
            threshold: 0.5,
            frame_size_samples: 512,
            min_speech_samples: 512,
            max_speech_samples: 16_000.0,
            min_silence_samples: 1024,
            min_silence_samples_at_max_speech: 1568,
        };
        let mut state = VadState::default();

        assert_eq!(
            state.update(&params, 0.8),
            Some(VadEvent::SpeechStart { sample: 0 })
        );
        assert_eq!(state.update(&params, 0.8), None);
        assert_eq!(state.update(&params, 0.1), None);
        assert_eq!(state.update(&params, 0.1), None);
        assert_eq!(
            state.update(&params, 0.1),
            Some(VadEvent::SpeechEnd {
                start_sample: 0,
                end_sample: 1536
            })
        );
    }
}
