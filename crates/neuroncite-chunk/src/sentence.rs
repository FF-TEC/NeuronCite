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

//! Sentence-based chunking strategy.
//!
//! Groups consecutive sentences into chunks that fit within a configurable
//! maximum word count. Sentence boundaries are detected using the Unicode
//! Standard Annex #29 sentence segmentation algorithm (via the
//! `unicode-segmentation` crate's `split_sentence_bound_indices` method),
//! with additional heuristics from the `abbreviations` module to suppress
//! false breaks at common academic abbreviations like "et al.", "Fig.", and
//! "Eq.".

use unicode_segmentation::UnicodeSegmentation;

use neuroncite_core::error::NeuronCiteError;
use neuroncite_core::traits::ChunkStrategy;
use neuroncite_core::types::{Chunk, PageText};

use crate::abbreviations::is_abbreviation;
use crate::normalize::normalize_text;
use crate::offset::compute_page_range;

/// Sentence-based chunking strategy that accumulates sentences until the
/// word count exceeds a configurable maximum.
pub struct SentenceStrategy {
    /// Maximum number of whitespace-delimited words per chunk. When adding
    /// the next sentence would exceed this limit, the current buffer is
    /// flushed as a chunk and accumulation restarts.
    pub max_words: usize,
}

impl ChunkStrategy for SentenceStrategy {
    /// Splits the concatenated, normalized document text into sentence-based
    /// chunks. Sentences are accumulated until the next sentence would push
    /// the word count over `max_words`, at which point the accumulated
    /// sentences are flushed as a chunk.
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

        // Concatenate all page contents with newline separators.
        let raw_concat = pages
            .iter()
            .map(|p| p.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let normalized = normalize_text(&raw_concat);

        // Compute per-page byte counts from individually normalized pages.
        let byte_counts: Vec<usize> = pages
            .iter()
            .map(|p| normalize_text(&p.content).len())
            .collect();

        // Split the normalized text into sentences using Unicode segmentation,
        // then merge segments that are separated by false breaks at abbreviations.
        let sentences = split_sentences_with_abbreviations(&normalized);

        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        let mut chunks = Vec::new();
        let mut chunk_index: usize = 0;

        // Accumulator for sentences being grouped into the current chunk.
        // Each entry is (byte_offset_start, byte_offset_end, word_count).
        let mut buffer: Vec<(usize, usize, usize)> = Vec::new();
        let mut buffer_word_count: usize = 0;

        for &(sent_start, sent_end) in &sentences {
            let sentence_text = &normalized[sent_start..sent_end];
            let sentence_words = sentence_text.split_whitespace().count();

            // If adding this sentence would exceed max_words and the buffer
            // is non-empty, flush the buffer as a chunk first.
            if buffer_word_count + sentence_words > self.max_words && !buffer.is_empty() {
                let doc_start = buffer[0].0;
                let doc_end = buffer[buffer.len() - 1].1;
                let content = normalized[doc_start..doc_end].to_string();
                let content_hash = Chunk::compute_content_hash(&content);
                let (page_start, page_end) = compute_page_range(&byte_counts, doc_start, doc_end)?;

                chunks.push(Chunk {
                    source_file: source_file.clone(),
                    page_start,
                    page_end,
                    chunk_index,
                    doc_text_offset_start: doc_start,
                    doc_text_offset_end: doc_end,
                    content,
                    content_hash,
                });

                chunk_index += 1;
                buffer.clear();
                buffer_word_count = 0;
            }

            buffer.push((sent_start, sent_end, sentence_words));
            buffer_word_count += sentence_words;
        }

        // Flush remaining sentences in the buffer as the final chunk.
        if !buffer.is_empty() {
            let doc_start = buffer[0].0;
            let doc_end = buffer[buffer.len() - 1].1;
            let content = normalized[doc_start..doc_end].to_string();
            let content_hash = Chunk::compute_content_hash(&content);
            let (page_start, page_end) = compute_page_range(&byte_counts, doc_start, doc_end)?;

            chunks.push(Chunk {
                source_file: source_file.clone(),
                page_start,
                page_end,
                chunk_index,
                doc_text_offset_start: doc_start,
                doc_text_offset_end: doc_end,
                content,
                content_hash,
            });
        }

        Ok(chunks)
    }
}

/// Splits the text into sentences using Unicode segmentation, then merges
/// segments that were falsely split at abbreviation periods.
///
/// The Unicode segmentation algorithm treats every period followed by
/// whitespace as a sentence boundary. This function post-processes the raw
/// segments to rejoin sentences that were split at abbreviation periods
/// (e.g., "et al." should not cause a sentence break when followed by a
/// lowercase word).
///
/// Returns a vector of `(byte_start, byte_end)` pairs for each merged
/// sentence.
fn split_sentences_with_abbreviations(text: &str) -> Vec<(usize, usize)> {
    // Collect raw sentence segments from the Unicode segmentation algorithm.
    // `split_sentence_bound_indices` returns (byte_offset, &str) tuples for
    // every segment between sentence boundaries (including whitespace-only
    // segments).
    let raw_segments: Vec<(usize, &str)> = text.split_sentence_bound_indices().collect();

    if raw_segments.is_empty() {
        return Vec::new();
    }

    // Merge adjacent segments into logical sentences. Abbreviation-terminated
    // segments are joined with the following segment instead of creating a
    // sentence break.
    let mut merged: Vec<(usize, usize)> = Vec::new();
    let mut current_start = raw_segments[0].0;
    let mut current_end = raw_segments[0].0 + raw_segments[0].1.len();

    for &(seg_offset, seg_text) in &raw_segments[1..] {
        let prev_text = &text[current_start..current_end];

        // Check if the previous accumulated segment ends with an abbreviation.
        // If so, merge the current segment with the previous one rather than
        // creating a sentence break.
        if ends_with_abbreviation(prev_text) {
            current_end = seg_offset + seg_text.len();
        } else {
            // Finalize the previous segment if it contains non-whitespace content.
            let trimmed = text[current_start..current_end].trim();
            if !trimmed.is_empty() {
                // Find the byte position of the trimmed content within the
                // segment range.
                let leading_ws = text[current_start..current_end].len()
                    - text[current_start..current_end].trim_start().len();
                let trim_start = current_start + leading_ws;
                let trim_end = trim_start + trimmed.len();
                merged.push((trim_start, trim_end));
            }
            current_start = seg_offset;
            current_end = seg_offset + seg_text.len();
        }
    }

    // Finalize the last accumulated segment.
    let trimmed = text[current_start..current_end].trim();
    if !trimmed.is_empty() {
        let leading_ws = text[current_start..current_end].len()
            - text[current_start..current_end].trim_start().len();
        let trim_start = current_start + leading_ws;
        let trim_end = trim_start + trimmed.len();
        merged.push((trim_start, trim_end));
    }

    merged
}

/// Checks whether the given text segment ends with one of the recognized
/// academic abbreviations.
///
/// Scans backwards from the end of the text to extract only the last two
/// whitespace-delimited tokens, avoiding an O(n) collection of all words.
/// This function is called once per sentence boundary in the merging loop,
/// where the input text can grow to the full document length during
/// abbreviation chaining. The backward-scan approach keeps each call O(1)
/// in stack and heap usage regardless of input size.
fn ends_with_abbreviation(text: &str) -> bool {
    let bytes = text.as_bytes();
    let len = bytes.len();

    // Skip trailing whitespace to find the end of the last word.
    let mut end = len;
    while end > 0 && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    if end == 0 {
        return false;
    }

    // Scan backwards past non-whitespace characters to find the start of
    // the last word.
    let mut start = end;
    while start > 0 && !bytes[start - 1].is_ascii_whitespace() {
        start -= 1;
    }
    let last_word = &text[start..end];

    // Check the last word against single-word abbreviations.
    if is_abbreviation(last_word) {
        return true;
    }

    // Scan further back to find the second-to-last word for multi-word
    // abbreviations like "et al."
    let mut gap_end = start;
    while gap_end > 0 && bytes[gap_end - 1].is_ascii_whitespace() {
        gap_end -= 1;
    }
    if gap_end == 0 {
        return false;
    }
    let mut second_start = gap_end;
    while second_start > 0 && !bytes[second_start - 1].is_ascii_whitespace() {
        second_start -= 1;
    }

    let two_word = format!("{} {last_word}", &text[second_start..gap_end]);
    is_abbreviation(&two_word)
}

#[cfg(test)]
mod tests {
    use unicode_segmentation::UnicodeSegmentation;

    use super::*;
    use crate::test_helpers::make_pages;

    /// T-CHK-010: No sentence boundary inside chunk interior (sentences are
    /// not split mid-sentence). Each chunk contains only complete sentences.
    #[test]
    fn t_chk_010_sentence_boundary_splits() {
        let text = "First sentence here. Second sentence follows. Third sentence arrives. Fourth sentence ends.";
        let pages = make_pages(&[text]);
        let strategy = SentenceStrategy { max_words: 6 };
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        // Each chunk must contain complete sentences. Verify by checking
        // that no chunk's interior contains a sentence-ending period followed
        // by a space and an uppercase letter (a heuristic for sentence breaks).
        for chunk in &chunks {
            let content = &chunk.content;
            // Re-segment the chunk content using `unicode_sentences`.
            // The number of sentences should match what was grouped together.
            let inner_sentences: Vec<&str> = content.unicode_sentences().collect();
            for sentence in &inner_sentences {
                let trimmed = sentence.trim();
                if !trimmed.is_empty() {
                    // Each inner sentence should end with a period or be the
                    // last (possibly period-less) sentence in the chunk.
                    assert!(
                        trimmed.ends_with('.') || trimmed == inner_sentences.last().unwrap().trim(),
                        "found incomplete sentence in chunk: '{trimmed}'"
                    );
                }
            }
        }
    }

    /// T-CHK-011: No chunk exceeds `max_words` whitespace-separated words.
    #[test]
    fn t_chk_011_max_words_limit() {
        let text = "One two three four. Five six seven eight. Nine ten eleven twelve. Thirteen fourteen fifteen sixteen.";
        let pages = make_pages(&[text]);
        let strategy = SentenceStrategy { max_words: 8 };
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        for chunk in &chunks {
            let word_count = chunk.content.split_whitespace().count();
            assert!(
                word_count <= 8,
                "chunk has {word_count} words, exceeds max_words of 8: '{}'",
                chunk.content
            );
        }
    }

    /// T-CHK-012: Text containing "et al." followed by a lowercase word does
    /// not produce a sentence break at the period after "al".
    #[test]
    fn t_chk_012_abbreviation_handling() {
        let text = "Smith et al. reported significant results. The study was large.";
        let pages = make_pages(&[text]);
        let strategy = SentenceStrategy { max_words: 50 };
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        // With max_words=50, all text should fit in one chunk. The "et al."
        // period should not split the first sentence.
        assert_eq!(chunks.len(), 1, "expected one chunk");
        assert!(
            chunks[0].content.contains("et al. reported"),
            "abbreviation 'et al.' was incorrectly split into separate sentences"
        );
    }

    /// Verifies that empty pages produce an error.
    #[test]
    fn empty_pages_returns_error() {
        let strategy = SentenceStrategy { max_words: 10 };
        let result = strategy.chunk(&[]);
        assert!(result.is_err());
    }

    /// Verifies that a single very long sentence does not get broken.
    /// It becomes a single chunk even if it exceeds `max_words`, because
    /// sentences are not split mid-sentence.
    #[test]
    fn single_long_sentence_becomes_one_chunk() {
        let text = "One two three four five six seven eight nine ten.";
        let pages = make_pages(&[text]);
        let strategy = SentenceStrategy { max_words: 5 };
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        // A single sentence that exceeds max_words is still emitted as one
        // chunk because the strategy does not break within sentences.
        assert_eq!(chunks.len(), 1);
    }
}
