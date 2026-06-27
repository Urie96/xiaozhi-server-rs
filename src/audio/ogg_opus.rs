use anyhow::{Result, bail};
use bytes::Bytes;

/// Incrementally demuxes Ogg Opus pages into raw Opus packets.
///
/// The Xiaozhi ESP32 client expects each WebSocket binary audio payload to be a
/// decodable Opus packet, while Volcengine's documented streaming Opus format is
/// `ogg_opus`. This small demuxer strips the Ogg container and skips OpusHead /
/// OpusTags packets.
#[derive(Debug, Default)]
pub struct OggOpusPacketizer {
    buffer: Vec<u8>,
    pending_packet: Vec<u8>,
    head_seen: bool,
    tags_seen: bool,
}

impl OggOpusPacketizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, data: &[u8]) -> Result<Vec<Bytes>> {
        self.buffer.extend_from_slice(data);
        let mut packets = Vec::new();

        loop {
            self.resync()?;

            if self.buffer.len() < 27 {
                break;
            }

            let segment_count = self.buffer[26] as usize;
            if self.buffer.len() < 27 + segment_count {
                break;
            }

            let payload_len: usize = self.buffer[27..27 + segment_count]
                .iter()
                .map(|&n| n as usize)
                .sum();
            let page_len = 27 + segment_count + payload_len;
            if self.buffer.len() < page_len {
                break;
            }

            let laces = self.buffer[27..27 + segment_count].to_vec();
            let payload = self.buffer[27 + segment_count..page_len].to_vec();
            self.buffer.drain(..page_len);

            let mut offset = 0usize;
            for lace in laces {
                let len = lace as usize;
                if offset + len > payload.len() {
                    bail!("invalid ogg page: segment exceeds payload");
                }

                self.pending_packet
                    .extend_from_slice(&payload[offset..offset + len]);
                offset += len;

                if lace < 255 {
                    let completed = std::mem::take(&mut self.pending_packet);
                    self.accept_packet(completed, &mut packets);
                }
            }
        }

        Ok(packets)
    }

    fn resync(&mut self) -> Result<()> {
        if self.buffer.len() < 4 || self.buffer.starts_with(b"OggS") {
            return Ok(());
        }

        if let Some(pos) = self.buffer.windows(4).position(|w| w == b"OggS") {
            self.buffer.drain(..pos);
            Ok(())
        } else {
            // Keep a small suffix in case the next push completes the magic.
            let keep = self.buffer.len().min(3);
            let suffix = self.buffer.split_off(self.buffer.len() - keep);
            self.buffer = suffix;
            Ok(())
        }
    }

    fn accept_packet(&mut self, packet: Vec<u8>, out: &mut Vec<Bytes>) {
        if packet.is_empty() {
            return;
        }

        if packet.starts_with(b"OpusHead") {
            self.head_seen = true;
            return;
        }

        if packet.starts_with(b"OpusTags") {
            self.tags_seen = true;
            return;
        }

        if self.head_seen && self.tags_seen {
            out.push(Bytes::from(packet));
        } else {
            tracing::debug!(
                bytes = packet.len(),
                "dropping ogg opus packet before headers"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(packet: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"OggS");
        out.extend_from_slice(&[0; 22]);
        let mut remaining = packet.len();
        let mut laces = Vec::new();
        while remaining >= 255 {
            laces.push(255u8);
            remaining -= 255;
        }
        laces.push(remaining as u8);
        out.push(laces.len() as u8);
        out.extend_from_slice(&laces);
        out.extend_from_slice(packet);
        out
    }

    #[test]
    fn extracts_opus_packets_after_headers() {
        let mut demux = OggOpusPacketizer::new();
        assert!(demux.push(&page(b"OpusHeadxxxx")).unwrap().is_empty());
        assert!(demux.push(&page(b"OpusTagsxxxx")).unwrap().is_empty());
        let packets = demux.push(&page(b"audio")).unwrap();
        assert_eq!(packets.len(), 1);
        assert_eq!(&packets[0][..], b"audio");
    }
}
