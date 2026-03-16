// NeuronCite -- local, privacy-preserving semantic document search engine.
// Copyright (C) 2026 NeuronCite Contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! Word-window chunking strategy.
//!
//! Splits text into fixed-size windows measured in whitespace-delimited words.
//! Each window overlaps with its predecessor by a configurable number of words.
//! Word boundaries are determined by splitting on Unicode whitespace.
//!
//! The algorithm concatenates all pages with "\n" separators, normalizes the
//! text, splits into words, and slides a window of `window_size` words forward
//! by `(window_size - overlap)` words at each step. Each window becomes one
//! Chunk with correct byte offsets within the normalized concatenated text and
//! page range resolved via per-page byte counts.

use neuroncite_core::error::NeuronCiteError;
use neuroncite_core::traits::ChunkStrategy;
use neuroncite_core::types::{Chunk, PageText};

use crate::normalize::normalize_text;
use crate::offset::compute_page_range;

/// Sliding word-window chunking strategy with configurable window size and
/// word overlap between consecutive chunks.
pub struct WordWindowStrategy {
    /// The number of whitespace-delimited words per chunk.
    pub window_size: usize,
    /// The number of words shared between consecutive chunks. Must be
    /// strictly less than `window_size`.
    pub overlap: usize,
}

impl ChunkStrategy for WordWindowStrategy {
    /// Splits the concatenated, normalized document text into overlapping
    /// word-window chunks.
    ///
    /// The concatenation joins all pages with "\n" separators. After
    /// normalization, the text is split into words on Unicode whitespace
    /// boundaries. A sliding window of `window_size` words advances by
    /// `(window_size - overlap)` words at each step.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Chunk` if the input pages are empty.
    fn chunk(&self, pages: &[PageText]) -> Result<Vec<Chunk>, NeuronCiteError> {
        if pages.is_empty() {
            return Err(NeuronCiteError::Chunk(
                "cannot chunk an empty page list".into(),
            ));
        }

        let source_file = pages[0].source_file.clone();

        // Normalize each page individually and derive both the per-page byte
        // counts and the concatenated normalized text from the same data. This
        // avoids the redundant (N+1)-th normalization of the full concatenation
        // and ensures byte counts are consistent with the concatenated text.
        let normalized_pages: Vec<String> =
            pages.iter().map(|p| normalize_text(&p.content)).collect();

        let byte_counts: Vec<usize> = normalized_pages.iter().map(|s| s.len()).collect();

        let normalized = normalized_pages.join("\n");

        // Collect word positions as (byte_start, byte_end) pairs within the
        // normalized text. Words are delimited by Unicode whitespace.
        let word_positions: Vec<(usize, usize)> = find_word_positions(&normalized);

        if word_positions.is_empty() {
            return Ok(Vec::new());
        }

        let step = self.window_size - self.overlap;
        let mut chunks = Vec::new();
        let mut chunk_index: usize = 0;
        let mut word_offset: usize = 0;

        while word_offset < word_positions.len() {
            let window_end = (word_offset + self.window_size).min(word_positions.len());
            let window_words = &word_positions[word_offset..window_end];

            // The chunk's byte range spans from the start of the first word
            // to the end of the last word in this window.
            let doc_offset_start = window_words[0].0;
            let doc_offset_end = window_words[window_words.len() - 1].1;

            let content = normalized[doc_offset_start..doc_offset_end].to_string();
            let content_hash = Chunk::compute_content_hash(&content);

            let (page_start, page_end) =
                compute_page_range(&byte_counts, doc_offset_start, doc_offset_end)?;

            chunks.push(Chunk {
                source_file: source_file.clone(),
                page_start,
                page_end,
                chunk_index,
                doc_text_offset_start: doc_offset_start,
                doc_text_offset_end: doc_offset_end,
                content,
                content_hash,
            });

            chunk_index += 1;

            // If the window reached the end of the word list, stop.
            if window_end >= word_positions.len() {
                break;
            }

            word_offset += step;
        }

        Ok(chunks)
    }
}

/// Finds the byte positions of all whitespace-delimited words in the text.
///
/// Returns a vector of `(start, end)` tuples where `start` is the byte index
/// of the first character of the word and `end` is the byte index one past
/// the last character. Words are separated by any Unicode whitespace character.
fn find_word_positions(text: &str) -> Vec<(usize, usize)> {
    let mut positions = Vec::new();
    let mut chars = text.char_indices().peekable();

    while let Some(&(i, ch)) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }

        // Start of a word
        let start = i;
        let mut end = i;

        while let Some(&(j, c)) = chars.peek() {
            if c.is_whitespace() {
                break;
            }
            end = j + c.len_utf8();
            chars.next();
        }

        positions.push((start, end));
    }

    positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_pages;

    /// T-CHK-003: Each chunk (except the last) contains exactly window_size
    /// whitespace-separated words.
    #[test]
    fn t_chk_003_correct_window_size() {
        let text = "one two three four five six seven eight nine ten eleven twelve";
        let pages = make_pages(&[text]);
        let strategy = WordWindowStrategy {
            window_size: 4,
            overlap: 1,
        };
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        // All chunks except the last must have exactly window_size words
        for chunk in &chunks[..chunks.len() - 1] {
            let word_count = chunk.content.split_whitespace().count();
            assert_eq!(
                word_count, 4,
                "non-final chunk has {word_count} words instead of 4"
            );
        }
    }

    /// T-CHK-004: The last `overlap` words of chunk i are identical to the
    /// first `overlap` words of chunk i+1.
    #[test]
    fn t_chk_004_correct_overlap() {
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let pages = make_pages(&[text]);
        let strategy = WordWindowStrategy {
            window_size: 4,
            overlap: 2,
        };
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        for i in 0..chunks.len() - 1 {
            let words_i: Vec<&str> = chunks[i].content.split_whitespace().collect();
            let words_next: Vec<&str> = chunks[i + 1].content.split_whitespace().collect();

            let tail_of_i = &words_i[words_i.len() - 2..];
            let head_of_next = &words_next[..2];

            assert_eq!(
                tail_of_i,
                head_of_next,
                "overlap mismatch between chunk {i} and chunk {}",
                i + 1
            );
        }
    }

    /// T-CHK-005: The last chunk contains between 1 and window_size words
    /// (inclusive).
    #[test]
    fn t_chk_005_final_chunk_remainder() {
        let text = "a b c d e f g h i j k";
        let pages = make_pages(&[text]);
        let strategy = WordWindowStrategy {
            window_size: 4,
            overlap: 1,
        };
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        let last_chunk = chunks.last().expect("no chunks produced");
        let last_word_count = last_chunk.content.split_whitespace().count();

        assert!(
            (1..=4).contains(&last_word_count),
            "last chunk has {last_word_count} words, expected 1..=4"
        );
    }

    /// T-CHK-006: A chunk whose word range spans a page boundary has
    /// page_start < page_end.
    #[test]
    fn t_chk_006_cross_page_chunks() {
        // Page 1 has 3 words, page 2 has 3 words. With window_size=4,
        // the first chunk must span across both pages.
        let pages = make_pages(&["word1 word2 word3", "word4 word5 word6"]);
        let strategy = WordWindowStrategy {
            window_size: 4,
            overlap: 0,
        };
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        // The first chunk should contain words from both pages
        let cross_page_chunk = &chunks[0];
        assert!(
            cross_page_chunk.page_start < cross_page_chunk.page_end,
            "expected cross-page chunk to have page_start ({}) < page_end ({})",
            cross_page_chunk.page_start,
            cross_page_chunk.page_end
        );
    }

    /// Verifies that an empty page list produces an error.
    #[test]
    fn empty_pages_returns_error() {
        let strategy = WordWindowStrategy {
            window_size: 4,
            overlap: 1,
        };
        let result = strategy.chunk(&[]);
        assert!(result.is_err());
    }

    /// Verifies that zero overlap produces non-overlapping chunks.
    #[test]
    fn zero_overlap() {
        let text = "one two three four five six";
        let pages = make_pages(&[text]);
        let strategy = WordWindowStrategy {
            window_size: 3,
            overlap: 0,
        };
        let chunks = strategy.chunk(&pages).expect("chunking failed");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].content, "one two three");
        assert_eq!(chunks[1].content, "four five six");
    }

    /// Regression test for issue #18: pages with whitespace that collapses
    /// during normalization must produce consistent byte counts. The old
    /// code normalized the full concatenation separately from individual pages,
    /// which could produce inconsistent byte mappings.
    #[test]
    fn normalization_consistency_across_pages() {
        // Pages with collapsible whitespace and a cross-page word.
        let pages = make_pages(&[
            "word1   word2  word3",  // extra spaces collapse
            "word4\tword5\t\tword6", // tabs collapse
        ]);
        let strategy = WordWindowStrategy {
            window_size: 3,
            overlap: 0,
        };
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        // All chunks must have valid content (no panic from inconsistent offsets).
        assert!(!chunks.is_empty(), "must produce at least one chunk");

        // Each chunk's page range must be within bounds.
        for chunk in &chunks {
            assert!(chunk.page_start >= 1 && chunk.page_start <= 2);
            assert!(chunk.page_end >= 1 && chunk.page_end <= 2);
            assert!(chunk.page_start <= chunk.page_end);
        }
    }
}
