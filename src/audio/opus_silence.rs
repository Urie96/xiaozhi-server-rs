use bytes::Bytes;

/// A compact valid Opus packet commonly used to represent silence/comfort-noise.
///
/// The ESP32 client only needs decodable Opus payloads for the first mock
/// implementation. Real TTS providers can replace this module later.
pub const OPUS_SILENCE_PACKET: &[u8] = &[0xF8, 0xFF, 0xFE];

pub fn packet() -> Bytes {
    Bytes::copy_from_slice(OPUS_SILENCE_PACKET)
}
