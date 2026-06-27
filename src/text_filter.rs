use async_stream::try_stream;
use futures_util::StreamExt;

use crate::services::TextStream;

#[derive(Debug)]
pub struct TtsTextFilter {
    paren_depth: usize,
    square_depth: usize,
    line_start: bool,
    pending_hashes: usize,
    pending_gt: bool,
}

impl Default for TtsTextFilter {
    fn default() -> Self {
        Self {
            paren_depth: 0,
            square_depth: 0,
            line_start: true,
            pending_hashes: 0,
            pending_gt: false,
        }
    }
}

impl TtsTextFilter {
    pub fn filter_chunk(&mut self, chunk: &str) -> String {
        let mut out = String::with_capacity(chunk.len());

        for ch in chunk.chars() {
            if self.paren_depth > 0 {
                match ch {
                    '(' | '（' => self.paren_depth += 1,
                    ')' | '）' => self.paren_depth = self.paren_depth.saturating_sub(1),
                    _ => {}
                }
                continue;
            }

            if self.square_depth > 0 {
                match ch {
                    '[' | '【' => self.square_depth += 1,
                    ']' | '】' => self.square_depth = self.square_depth.saturating_sub(1),
                    _ => {}
                }
                continue;
            }

            if self.pending_hashes > 0 {
                if ch == '#' && self.pending_hashes < 6 {
                    self.pending_hashes += 1;
                    continue;
                }
                if ch.is_whitespace() {
                    self.pending_hashes = 0;
                    self.line_start = false;
                    continue;
                }
                for _ in 0..self.pending_hashes {
                    out.push('#');
                }
                self.pending_hashes = 0;
            }

            if self.pending_gt {
                if ch.is_whitespace() {
                    self.pending_gt = false;
                    self.line_start = false;
                    continue;
                }
                out.push('>');
                self.pending_gt = false;
            }

            match ch {
                '(' | '（' => self.paren_depth = 1,
                '[' | '【' => self.square_depth = 1,
                '*' | '_' | '`' | '~' | '|' => {}
                '#' if self.line_start => {
                    self.pending_hashes = (self.pending_hashes + 1).min(6);
                }
                '>' if self.line_start => self.pending_gt = true,
                ch if is_emoji(ch) => {}
                ch => {
                    out.push(ch);
                    self.line_start = ch == '\n';
                }
            }
        }

        out
    }

    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        if self.pending_hashes > 0 {
            for _ in 0..self.pending_hashes {
                out.push('#');
            }
            self.pending_hashes = 0;
        }
        if self.pending_gt {
            out.push('>');
            self.pending_gt = false;
        }
        self.line_start = out.ends_with('\n');
        out
    }
}

pub fn filter_tts_text_stream(mut input: TextStream) -> TextStream {
    Box::pin(try_stream! {
        let mut filter = TtsTextFilter::default();

        while let Some(chunk) = input.next().await {
            let filtered = filter.filter_chunk(&chunk?);
            if !filtered.trim().is_empty() {
                yield filtered;
            }
        }

        let tail = filter.finish();
        if !tail.trim().is_empty() {
            yield tail;
        }
    })
}

fn is_emoji(ch: char) -> bool {
    matches!(
        ch as u32,
        0x1F000..=0x1FFFF | 0x2600..=0x26FF | 0x2700..=0x27BF
    )
}

#[cfg(test)]
mod tests {
    use super::TtsTextFilter;

    #[test]
    fn removes_markdown_and_emoji() {
        let mut filter = TtsTextFilter::default();
        assert_eq!(
            filter.filter_chunk("**你好**，这是 `Rust` ~~服务~~ 😊"),
            "你好，这是 Rust 服务 "
        );
    }

    #[test]
    fn removes_parenthesized_stage_directions_across_chunks() {
        let mut filter = TtsTextFilter::default();
        assert_eq!(filter.filter_chunk("你好（突然"), "你好");
        assert_eq!(filter.filter_chunk("小声）世界"), "世界");
    }

    #[test]
    fn removes_markdown_heading_quote_and_link_text() {
        let mut filter = TtsTextFilter::default();
        assert_eq!(
            filter.filter_chunk("## 标题\n> 引用\n请看[链接](url)"),
            "标题\n引用\n请看"
        );
    }

    #[test]
    fn preserves_literal_hash_and_gt_when_not_markdown_prefixes() {
        let mut filter = TtsTextFilter::default();
        assert_eq!(filter.filter_chunk("C# 和 1>0"), "C# 和 1>0");
    }
}
