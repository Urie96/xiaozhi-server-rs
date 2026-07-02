use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow, bail};
use hound;
use ndarray::Array3;
use ort::{session::Session, value::TensorRef};
use realfft::{RealFftPlanner, RealToComplex};
use tokio::sync::Mutex;

const DEFAULT_MODEL_PATH: &str = "models/voxceleb_resnet34.onnx";
const DEFAULT_DB_DIR: &str = "speakers";
const DEFAULT_AGENT_ID: &str = "zhuzhu";
const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.35;
const SPEAKER_SCORE_TOP_K: usize = 3;

#[derive(Clone, Debug)]
pub struct SpeakerIdConfig {
    pub model_path: PathBuf,
    pub db_dir: PathBuf,
    pub default_agent_id: String,
    pub min_similarity: f32,
}

impl SpeakerIdConfig {
    pub fn from_env() -> Result<Option<Self>> {
        let provider = env_any(&["XIAOZHI_SPEAKER_PROVIDER", "SPEAKER_PROVIDER"])
            .unwrap_or_else(|| "none".to_string())
            .to_ascii_lowercase();
        if matches!(
            provider.as_str(),
            "none" | "off" | "disabled" | "false" | "0"
        ) {
            return Ok(None);
        }
        if !matches!(provider.as_str(), "speaker_id" | "speaker" | "voiceprint") {
            bail!("unsupported XIAOZHI_SPEAKER_PROVIDER={provider}; expected speaker_id or none");
        }

        let model_path = env_path_any(&["XIAOZHI_SPEAKER_MODEL_PATH", "SPEAKER_MODEL_PATH"])
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
        let db_dir = env_path_any(&["XIAOZHI_SPEAKER_DB_DIR", "SPEAKER_DB_DIR"])
            .unwrap_or_else(|| PathBuf::from(DEFAULT_DB_DIR));
        let default_agent_id = env_any(&[
            "XIAOZHI_SPEAKER_DEFAULT_AGENT_ID",
            "SPEAKER_DEFAULT_AGENT_ID",
        ])
        .unwrap_or_else(|| DEFAULT_AGENT_ID.to_string());
        let min_similarity = env_f32_any(
            &["XIAOZHI_SPEAKER_MIN_SIMILARITY", "SPEAKER_MIN_SIMILARITY"],
            DEFAULT_SIMILARITY_THRESHOLD,
        );

        Ok(Some(Self {
            model_path,
            db_dir,
            default_agent_id,
            min_similarity,
        }))
    }
}

#[derive(Clone, Debug)]
pub struct SpeakerTurnIdentity {
    pub speaker_name: Option<String>,
    pub agent_id: String,
    pub similarity: f32,
}

pub struct SpeakerIdService {
    embedder: Arc<Mutex<SpeakerEmbedder>>,
    registry: SpeakerRegistry,
    default_agent_id: String,
    min_similarity: f32,
}

impl SpeakerIdService {
    pub fn new(config: SpeakerIdConfig) -> Result<Self> {
        let mut embedder = SpeakerEmbedder::new(&config.model_path)
            .with_context(|| format!("load speaker model {}", config.model_path.display()))?;
        let registry = SpeakerRegistry::from_wavs(&config.db_dir, &mut embedder)
            .with_context(|| format!("load speaker wavs from {}", config.db_dir.display()))?;
        Ok(Self {
            embedder: Arc::new(Mutex::new(embedder)),
            registry,
            default_agent_id: config.default_agent_id,
            min_similarity: config.min_similarity,
        })
    }

    pub fn default_agent_id(&self) -> &str {
        &self.default_agent_id
    }

    pub async fn identify_turn(&self, pcm: &[i16]) -> Result<Option<SpeakerTurnIdentity>> {
        if pcm.is_empty() {
            return Ok(None);
        }

        let embedding = {
            let mut embedder = self.embedder.lock().await;
            embedder.embed_pcm(pcm)?
        };

        Ok(Some(self.registry.identify(
            &embedding,
            &self.default_agent_id,
            self.min_similarity,
        )))
    }
}

#[derive(Debug)]
struct SpeakerRegistry {
    speakers: Vec<RegisteredSpeaker>,
    agent_map: HashMap<String, String>,
}

impl SpeakerRegistry {
    fn from_wavs(db_dir: &Path, embedder: &mut SpeakerEmbedder) -> Result<Self> {
        let mut speakers = Vec::new();
        if db_dir.exists() {
            let entries =
                fs::read_dir(db_dir).with_context(|| format!("read {}", db_dir.display()))?;
            let mut entries = entries
                .flatten()
                .map(|entry| entry.path())
                .collect::<Vec<_>>();
            entries.sort();

            for path in entries {
                if !path.is_dir() {
                    continue;
                }

                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                match register_speaker_from_dir(&path, &name, embedder) {
                    Ok(Some(speaker)) => {
                        tracing::info!(
                            speaker = %name,
                            path = %path.display(),
                            samples = speaker.embeddings.len(),
                            dim = speaker.embeddings.first().map_or(0, Vec::len),
                            "registered speaker from wav samples"
                        );
                        speakers.push(speaker);
                    }
                    Ok(None) => tracing::warn!(
                        speaker = %name,
                        path = %path.display(),
                        "skip speaker directory without valid wav samples"
                    ),
                    Err(err) => tracing::warn!(
                        path = %path.display(),
                        error = %err,
                        "skip invalid speaker directory"
                    ),
                }
            }
        }

        let agent_map = load_agent_map(&db_dir.join("agents.json"))?;
        tracing::info!(
            db_dir = %db_dir.display(),
            speakers = speakers.len(),
            agent_map_entries = agent_map.len(),
            "speaker database loaded from speaker directories"
        );
        Ok(Self {
            speakers,
            agent_map,
        })
    }

    fn identify(
        &self,
        embedding: &[f32],
        default_agent_id: &str,
        min_similarity: f32,
    ) -> SpeakerTurnIdentity {
        let mut best_name = None;
        let mut best_sim = -1.0_f32;

        for speaker in &self.speakers {
            let sim = speaker_similarity(embedding, &speaker.embeddings, SPEAKER_SCORE_TOP_K);
            if sim > best_sim {
                best_sim = sim;
                best_name = Some(speaker.name.clone());
            }
        }

        if let Some(name) = best_name {
            if best_sim >= min_similarity {
                let agent_id = self
                    .agent_map
                    .get(&name)
                    .cloned()
                    .unwrap_or_else(|| name.clone());
                return SpeakerTurnIdentity {
                    speaker_name: Some(name),
                    agent_id,
                    similarity: best_sim,
                };
            }
        }

        SpeakerTurnIdentity {
            speaker_name: None,
            agent_id: default_agent_id.to_string(),
            similarity: best_sim,
        }
    }
}

fn speaker_similarity(query: &[f32], embeddings: &[Vec<f32>], top_k: usize) -> f32 {
    if embeddings.is_empty() || top_k == 0 {
        return -1.0;
    }

    let mut scores = embeddings
        .iter()
        .map(|embedding| cosine_similarity(query, embedding))
        .collect::<Vec<_>>();
    scores.sort_by(|a, b| b.total_cmp(a));

    let count = scores.len().min(top_k);
    scores.iter().take(count).sum::<f32>() / count as f32
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[derive(Debug)]
struct RegisteredSpeaker {
    name: String,
    embeddings: Vec<Vec<f32>>,
}

#[derive(Clone)]
struct FbankConfig {
    sample_rate: u32,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
    pre_emphasis: f32,
}

impl Default for FbankConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            n_mels: 80,
            f_min: 20.0,
            f_max: 7_600.0,
            pre_emphasis: 0.97,
        }
    }
}

struct Fbank {
    config: FbankConfig,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>,
}

impl Fbank {
    fn new(config: FbankConfig) -> Self {
        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(config.n_fft);
        let window = hamming_window(config.win_length);
        let mel_filters = mel_filterbank(
            config.n_fft,
            config.n_mels,
            config.sample_rate,
            config.f_min,
            config.f_max,
        );
        Self {
            config,
            fft,
            window,
            mel_filters,
        }
    }

    fn extract(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        if samples.len() < self.config.win_length {
            return Vec::new();
        }

        let mut pre = Vec::with_capacity(samples.len());
        pre.push(samples[0]);
        for i in 1..samples.len() {
            pre.push(samples[i] - self.config.pre_emphasis * samples[i - 1]);
        }

        let hop = self.config.hop_length;
        let win = self.config.win_length;
        let n_fft = self.config.n_fft;
        let n_mels = self.config.n_mels;
        let num_frames = 1 + (pre.len().saturating_sub(win)) / hop;

        let mut melspec = Vec::with_capacity(num_frames);
        let mut spectrum = self.fft.make_output_vec();

        for i in 0..num_frames {
            let start = i * hop;
            let mut buf = vec![0.0_f32; n_fft];
            for (j, &v) in pre[start..start + win].iter().enumerate() {
                buf[j] = v * self.window[j];
            }

            if self.fft.process(&mut buf, &mut spectrum).is_err() {
                continue;
            }

            let power: Vec<f32> = spectrum.iter().map(|c| c.norm_sqr()).collect();
            let mut mel = vec![0.0_f32; n_mels];
            for (m, filter) in self.mel_filters.iter().enumerate() {
                mel[m] = filter
                    .iter()
                    .zip(power.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    .max(1e-10)
                    .ln();
            }
            melspec.push(mel);
        }

        melspec
    }
}

struct SpeakerEmbedder {
    session: Session,
    fbank: Fbank,
}

impl SpeakerEmbedder {
    fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()
            .map_err(|err| anyhow!("session builder: {err}"))?
            .with_intra_threads(1)
            .map_err(|err| anyhow!("speaker model intra threads: {err}"))?
            .commit_from_file(model_path.as_ref())
            .map_err(|err| anyhow!("commit speaker model: {err}"))?;

        Ok(Self {
            session,
            fbank: Fbank::new(FbankConfig::default()),
        })
    }

    fn embed_pcm(&mut self, samples: &[i16]) -> Result<Vec<f32>> {
        let mut audio = samples
            .iter()
            .map(|sample| *sample as f32 / i16::MAX as f32)
            .collect::<Vec<_>>();
        if audio.len() < self.fbank.config.win_length {
            audio.resize(self.fbank.config.win_length, 0.0);
        }

        let mut feats = self.fbank.extract(&audio);
        if feats.is_empty() {
            bail!("audio too short for speaker embedding");
        }
        feats = apply_cmvn(&feats);

        let time = feats.len();
        let n_mels = self.fbank.config.n_mels;
        let flat: Vec<f32> = feats.into_iter().flatten().collect();
        let input_tensor = Array3::from_shape_vec((1, time, n_mels), flat)
            .map_err(|err| anyhow!("speaker embedding tensor shape: {err}"))?;
        let input_ref = TensorRef::from_array_view(&input_tensor)
            .map_err(|err| anyhow!("speaker embedding tensor view: {err}"))?;
        let outputs = self
            .session
            .run(ort::inputs![input_ref])
            .map_err(|err| anyhow!("speaker model inference: {err}"))?;

        let embedding_view = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|err| anyhow!("speaker embedding output: {err}"))?;
        let mut embedding: Vec<f32> = embedding_view.iter().copied().collect();
        l2_normalize(&mut embedding);
        Ok(embedding)
    }
}

fn register_speaker_from_dir(
    dir: &Path,
    name: &str,
    embedder: &mut SpeakerEmbedder,
) -> Result<Option<RegisteredSpeaker>> {
    let entries = fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?;
    let mut wav_paths = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("wav"))
        .collect::<Vec<_>>();
    wav_paths.sort();

    let mut embeddings = Vec::new();
    for path in wav_paths {
        match embed_speaker_wav(&path, embedder) {
            Ok(embedding) => embeddings.push(embedding),
            Err(err) => tracing::warn!(
                path = %path.display(),
                error = %err,
                "skip invalid speaker wav sample"
            ),
        }
    }

    if embeddings.is_empty() {
        return Ok(None);
    }

    Ok(Some(RegisteredSpeaker {
        name: name.to_string(),
        embeddings,
    }))
}

fn embed_speaker_wav(path: &Path, embedder: &mut SpeakerEmbedder) -> Result<Vec<f32>> {
    let samples =
        read_wav_16khz_mono(path).with_context(|| format!("read wav {}", path.display()))?;
    let pcm: Vec<i16> = samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();
    embedder
        .embed_pcm(&pcm)
        .with_context(|| format!("embed speaker {}", path.display()))
}

fn read_wav_16khz_mono(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open WAV {}", path.display()))?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    let mono = if spec.channels == 2 {
        samples.chunks(2).map(|c| (c[0] + c[1]) * 0.5).collect()
    } else {
        samples
    };

    Ok(if spec.sample_rate == 16000 {
        mono
    } else {
        resample(&mono, spec.sample_rate, 16000)
    })
}

fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (input.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src_idx = i as f64 / ratio;
        let left = src_idx.floor() as usize;
        let right = (left + 1).min(input.len() - 1);
        let frac = src_idx - left as f64;
        let val = (input[left] as f64 * (1.0 - frac) + input[right] as f64 * frac) as f32;
        output.push(val);
    }
    output
}

fn load_agent_map(path: &Path) -> Result<HashMap<String, String>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let map = serde_json::from_slice::<HashMap<String, String>>(&data)
        .with_context(|| format!("parse {}", path.display()))?;
    Ok(map)
}

fn hamming_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos())
        .collect()
}

fn mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    sample_rate: u32,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    let sr = sample_rate as f32;
    let mel_min = 2595.0 * (1.0 + f_min / 700.0).log10();
    let mel_max = 2595.0 * (1.0 + f_max / 700.0).log10();

    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points
        .iter()
        .map(|&m| 700.0 * (10.0_f32.powf(m / 2595.0) - 1.0))
        .collect();

    let bins: Vec<usize> = hz_points
        .iter()
        .map(|&hz| (hz * (n_fft as f32 + 1.0) / sr).floor() as usize)
        .collect();

    let mut filters = vec![vec![0.0_f32; n_fft / 2 + 1]; n_mels];
    for m in 0..n_mels {
        let left = bins[m];
        let center = bins[m + 1];
        let right = bins[m + 2];

        for i in left..center {
            let denom = (center - left) as f32;
            if denom > 0.0 {
                filters[m][i] = (i - left) as f32 / denom;
            }
        }
        for i in center..right {
            let denom = (right - center) as f32;
            if denom > 0.0 {
                filters[m][i] = (right - i) as f32 / denom;
            }
        }
    }

    filters
}

fn apply_cmvn(frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if frames.is_empty() {
        return Vec::new();
    }

    let n_bins = frames[0].len();
    let n_frames = frames.len() as f32;
    let mut means = vec![0.0_f32; n_bins];
    for frame in frames {
        for (i, value) in frame.iter().enumerate() {
            means[i] += value;
        }
    }
    for mean in &mut means {
        *mean /= n_frames;
    }

    frames
        .iter()
        .map(|frame| {
            frame
                .iter()
                .zip(means.iter())
                .map(|(&v, &m)| v - m)
                .collect()
        })
        .collect()
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for value in v.iter_mut() {
            *value /= norm;
        }
    }
}

fn env_any(names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| std::env::var(name).ok().filter(|value| !value.is_empty()))
}

fn env_path_any(names: &[&str]) -> Option<PathBuf> {
    env_any(names).map(PathBuf::from)
}

fn env_f32_any(names: &[&str], default: f32) -> f32 {
    env_any(names)
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_agent_map_json() {
        let map: HashMap<String, String> =
            serde_json::from_str(r#"{"alice":"agent-a","bob":"agent-b"}"#).unwrap();
        assert_eq!(map.get("alice").map(String::as_str), Some("agent-a"));
        assert_eq!(map.get("bob").map(String::as_str), Some("agent-b"));
    }

    #[test]
    fn speaker_similarity_uses_top_k_mean() {
        let query = vec![1.0, 0.0];
        let embeddings = vec![
            vec![0.1, 0.0],
            vec![0.9, 0.0],
            vec![0.6, 0.0],
            vec![0.3, 0.0],
        ];

        let score = speaker_similarity(&query, &embeddings, 3);

        assert!((score - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn registry_identify_scores_each_speaker_from_multiple_samples() {
        let registry = SpeakerRegistry {
            speakers: vec![
                RegisteredSpeaker {
                    name: "alice".to_string(),
                    embeddings: vec![vec![0.7, 0.0], vec![0.8, 0.0], vec![0.9, 0.0]],
                },
                RegisteredSpeaker {
                    name: "bob".to_string(),
                    embeddings: vec![vec![0.95, 0.0], vec![0.1, 0.0], vec![0.1, 0.0]],
                },
            ],
            agent_map: HashMap::from([("alice".to_string(), "agent-a".to_string())]),
        };

        let identity = registry.identify(&[1.0, 0.0], "default-agent", 0.35);

        assert_eq!(identity.speaker_name.as_deref(), Some("alice"));
        assert_eq!(identity.agent_id, "agent-a");
        assert!((identity.similarity - 0.8).abs() < f32::EPSILON);
    }
}
