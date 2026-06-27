use anyhow::{Context, Result};
use opus::{Channels, Decoder};

/// Decodes Xiaozhi client raw Opus packets into 16 kHz mono PCM S16LE bytes.
///
/// ESP32 encodes microphone audio as raw Opus packets at 16 kHz mono. Volcengine
/// ASR accepts PCM S16LE, so the ASR provider can stream decoded PCM chunks
/// without waiting for the whole utterance.
pub struct OpusPcmDecoder {
    decoder: Decoder,
    scratch: Vec<i16>,
}

impl OpusPcmDecoder {
    pub fn new(sample_rate: u32) -> Result<Self> {
        Ok(Self {
            decoder: Decoder::new(sample_rate, Channels::Mono).context("create opus decoder")?,
            // Max Opus packet duration is 120 ms. At 16 kHz mono this is 1920 samples.
            scratch: vec![0; (sample_rate as usize * 120 / 1000).max(1920)],
        })
    }

    pub fn decode_to_pcm_i16(&mut self, packet: &[u8]) -> Result<Vec<i16>> {
        let samples = self
            .decoder
            .decode(packet, &mut self.scratch, false)
            .context("decode opus packet")?;
        Ok(self.scratch[..samples].to_vec())
    }
}

pub fn pcm_i16_to_le_bytes(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}
