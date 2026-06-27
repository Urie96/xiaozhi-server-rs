use std::time::Duration;

/// Returns the playback duration encoded in an Opus packet's TOC byte.
///
/// Based on RFC 6716 section 3.1. The duration is independent of the output
/// sample rate; Opus frame sizes are defined in milliseconds.
pub fn packet_duration(packet: &[u8]) -> Option<Duration> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    let code = toc & 0b11;

    let frame_duration_ms_x2 = frame_duration_ms_x2(config)?;
    let frame_count = match code {
        0 => 1u32,
        1 | 2 => 2u32,
        3 => (*packet.get(1)? & 0b0011_1111) as u32,
        _ => unreachable!(),
    };

    if frame_count == 0 {
        return None;
    }

    let total_ms_x2 = frame_duration_ms_x2.checked_mul(frame_count)?;
    Some(Duration::from_micros((total_ms_x2 as u64) * 500))
}

fn frame_duration_ms_x2(config: u8) -> Option<u32> {
    match config {
        // SILK-only: 10, 20, 40, 60 ms.
        0..=11 => match config % 4 {
            0 => Some(20),
            1 => Some(40),
            2 => Some(80),
            3 => Some(120),
            _ => unreachable!(),
        },
        // Hybrid: 10, 20 ms.
        12..=15 => match config % 2 {
            0 => Some(20),
            1 => Some(40),
            _ => unreachable!(),
        },
        // CELT-only: 2.5, 5, 10, 20 ms.
        16..=31 => match config % 4 {
            0 => Some(5),
            1 => Some(10),
            2 => Some(20),
            3 => Some(40),
            _ => unreachable!(),
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_20ms_celt_frame() {
        // config=31, code=0 => one 20ms CELT-only frame.
        assert_eq!(
            packet_duration(&[0b1111_1000]).unwrap(),
            Duration::from_millis(20)
        );
    }

    #[test]
    fn parses_two_20ms_celt_frames() {
        // config=31, code=1 => two 20ms CELT-only frames.
        assert_eq!(
            packet_duration(&[0b1111_1001]).unwrap(),
            Duration::from_millis(40)
        );
    }

    #[test]
    fn parses_code3_frame_count() {
        // config=31, code=3, count=3 => three 20ms CELT-only frames.
        assert_eq!(
            packet_duration(&[0b1111_1011, 3]).unwrap(),
            Duration::from_millis(60)
        );
    }
}
