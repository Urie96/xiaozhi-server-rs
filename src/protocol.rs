use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

pub const SERVER_SAMPLE_RATE: u32 = 24_000;
pub const SERVER_CHANNELS: u8 = 1;
pub const SERVER_FRAME_DURATION_MS: u32 = 60;

#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub timestamp: u32,
    pub payload: Bytes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryProtocolVersion {
    V1,
    V2,
    V3,
}

impl BinaryProtocolVersion {
    pub fn from_header(value: Option<&str>) -> Self {
        match value.and_then(|v| v.parse::<u8>().ok()) {
            Some(2) => Self::V2,
            Some(3) => Self::V3,
            _ => Self::V1,
        }
    }

    pub fn from_hello(value: Option<u64>, fallback: Self) -> Self {
        match value {
            Some(2) => Self::V2,
            Some(3) => Self::V3,
            Some(1) => Self::V1,
            _ => fallback,
        }
    }
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("binary frame too short for protocol v{version}: {len} bytes")]
    ShortFrame { version: u8, len: usize },
    #[error("declared payload size {declared} exceeds frame length {available}")]
    InvalidPayloadSize { declared: usize, available: usize },
}

pub fn decode_audio_frame(
    version: BinaryProtocolVersion,
    data: &[u8],
) -> Result<Option<AudioFrame>, ProtocolError> {
    match version {
        BinaryProtocolVersion::V1 => Ok(Some(AudioFrame {
            timestamp: 0,
            payload: Bytes::copy_from_slice(data),
        })),
        BinaryProtocolVersion::V2 => {
            if data.len() < 16 {
                return Err(ProtocolError::ShortFrame {
                    version: 2,
                    len: data.len(),
                });
            }
            let typ = u16::from_be_bytes([data[2], data[3]]);
            if typ != 0 {
                return Ok(None);
            }
            let timestamp = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
            let payload_size =
                u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
            if payload_size > data.len() - 16 {
                return Err(ProtocolError::InvalidPayloadSize {
                    declared: payload_size,
                    available: data.len() - 16,
                });
            }
            Ok(Some(AudioFrame {
                timestamp,
                payload: Bytes::copy_from_slice(&data[16..16 + payload_size]),
            }))
        }
        BinaryProtocolVersion::V3 => {
            if data.len() < 4 {
                return Err(ProtocolError::ShortFrame {
                    version: 3,
                    len: data.len(),
                });
            }
            let typ = data[0];
            if typ != 0 {
                return Ok(None);
            }
            let payload_size = u16::from_be_bytes([data[2], data[3]]) as usize;
            if payload_size > data.len() - 4 {
                return Err(ProtocolError::InvalidPayloadSize {
                    declared: payload_size,
                    available: data.len() - 4,
                });
            }
            Ok(Some(AudioFrame {
                timestamp: 0,
                payload: Bytes::copy_from_slice(&data[4..4 + payload_size]),
            }))
        }
    }
}

pub fn encode_audio_frame(version: BinaryProtocolVersion, frame: &AudioFrame) -> Bytes {
    match version {
        BinaryProtocolVersion::V1 => frame.payload.clone(),
        BinaryProtocolVersion::V2 => {
            let mut out = Vec::with_capacity(16 + frame.payload.len());
            out.extend_from_slice(&2u16.to_be_bytes());
            out.extend_from_slice(&0u16.to_be_bytes());
            out.extend_from_slice(&0u32.to_be_bytes());
            out.extend_from_slice(&frame.timestamp.to_be_bytes());
            out.extend_from_slice(&(frame.payload.len() as u32).to_be_bytes());
            out.extend_from_slice(&frame.payload);
            Bytes::from(out)
        }
        BinaryProtocolVersion::V3 => {
            let mut out = Vec::with_capacity(4 + frame.payload.len());
            out.push(0);
            out.push(0);
            out.extend_from_slice(&(frame.payload.len() as u16).to_be_bytes());
            out.extend_from_slice(&frame.payload);
            Bytes::from(out)
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct IncomingJson {
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(rename = "type")]
    pub typ: String,
    #[serde(default)]
    pub version: Option<u64>,
    #[serde(flatten)]
    pub extra: Value,
}

impl IncomingJson {
    pub fn state(&self) -> Option<&str> {
        self.extra.get("state")?.as_str()
    }

    pub fn mode(&self) -> Option<&str> {
        self.extra.get("mode")?.as_str()
    }
}

#[derive(Debug, Serialize)]
pub struct AudioParams {
    pub format: &'static str,
    pub sample_rate: u32,
    pub channels: u8,
    pub frame_duration: u32,
}

pub fn audio_params() -> AudioParams {
    AudioParams {
        format: "opus",
        sample_rate: SERVER_SAMPLE_RATE,
        channels: SERVER_CHANNELS,
        frame_duration: SERVER_FRAME_DURATION_MS,
    }
}

pub fn hello(session_id: &str) -> Value {
    json!({
        "type": "hello",
        "transport": "websocket",
        "session_id": session_id,
        "audio_params": audio_params(),
    })
}

pub fn stt(session_id: &str, text: &str) -> Value {
    json!({"session_id": session_id, "type": "stt", "text": text})
}

pub fn llm_emotion(session_id: &str, emotion: &str, text: &str) -> Value {
    json!({"session_id": session_id, "type": "llm", "emotion": emotion, "text": text})
}

pub fn tts_start(session_id: &str) -> Value {
    json!({"session_id": session_id, "type": "tts", "state": "start"})
}

pub fn tts_sentence_start(session_id: &str, text: &str) -> Value {
    json!({"session_id": session_id, "type": "tts", "state": "sentence_start", "text": text})
}

pub fn tts_stop(session_id: &str) -> Value {
    json!({"session_id": session_id, "type": "tts", "state": "stop"})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_incoming_json_with_extra_fields() {
        let msg: IncomingJson = serde_json::from_str(
            r#"{"session_id":"s","type":"listen","state":"start","mode":"auto"}"#,
        )
        .unwrap();
        assert_eq!(msg.typ, "listen");
        assert_eq!(msg.state(), Some("start"));
        assert_eq!(msg.mode(), Some("auto"));
    }

    #[test]
    fn binary_v2_roundtrip() {
        let frame = AudioFrame {
            timestamp: 42,
            payload: Bytes::from_static(b"opus"),
        };
        let encoded = encode_audio_frame(BinaryProtocolVersion::V2, &frame);
        let decoded = decode_audio_frame(BinaryProtocolVersion::V2, &encoded)
            .unwrap()
            .unwrap();
        assert_eq!(decoded.timestamp, 42);
        assert_eq!(&decoded.payload[..], b"opus");
    }

    #[test]
    fn binary_v3_roundtrip() {
        let frame = AudioFrame {
            timestamp: 0,
            payload: Bytes::from_static(b"opus"),
        };
        let encoded = encode_audio_frame(BinaryProtocolVersion::V3, &frame);
        let decoded = decode_audio_frame(BinaryProtocolVersion::V3, &encoded)
            .unwrap()
            .unwrap();
        assert_eq!(&decoded.payload[..], b"opus");
    }
}
