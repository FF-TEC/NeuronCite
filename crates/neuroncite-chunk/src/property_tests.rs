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

// Property-based tests for all chunking strategies and the Unicode normalizer.
//
// These tests use `proptest` to generate arbitrary inputs and verify that
// structural invariants hold for every generated case. Unlike the example-based
// unit tests (T-CHK-001..025), property tests exercise the full input space
// by running each property against 256 distinct randomized cases (proptest
// default) with automatic shrinking on failure.
//
// Invariants verified:
//   - Offset validity: all byte offsets within concatenated text length.
//   - Content preservation: every word in the input appears in some chunk.
//   - Chunk index sequence: chunk_index is 0-based and consecutive.
//   - No empty content: every chunk has non-empty content after trimming.
//   - Monotonic offsets: doc_text_offset_start is non-decreasing.
//   - Determinism: same input always produces the same chunks.
//   - NFC idempotency: normalizing twice equals normalizing once.
//   - Whitespace collapse: no two consecutive ASCII spaces after normalization.
//   - Hyphenation repair: alphabetic text around hyphen-newline is preserved.

use std::path::PathBuf;

use neuroncite_core::traits::ChunkStrategy;
use neuroncite_core::types::{Chunk, ExtractionBackend, PageText};
use proptest::prelude::*;

use crate::normalize::normalize_text;
use crate::page::PageStrategy;
use crate::sentence::SentenceStrategy;
use crate::word_window::WordWindowStrategy;

/// Constructs a single-page `PageText` with the given content string.
/// Uses `ExtractionBackend::PdfExtract` and page number 1 as defaults.
fn make_page(content: impl Into<String>) -> PageText {
    PageText {
        source_file: PathBuf::from("test.pdf"),
        page_number: 1,
        content: content.into(),
        backend: ExtractionBackend::PdfExtract,
    }
}

/// Constructs a multi-page `PageText` slice from a vec of content strings.
/// Pages are 1-indexed.
fn make_pages(contents: Vec<String>) -> Vec<PageText> {
    contents
        .into_iter()
        .enumerate()
        .map(|(i, content)| PageText {
            source_file: PathBuf::from("test.pdf"),
            page_number: i + 1,
            content,
            backend: ExtractionBackend::PdfExtract,
        })
        .collect()
}

/// Asserts the offset invariants shared by all chunking strategies.
///
/// This function does NOT check content emptiness because the page strategy
/// preserves whitespace-only content verbatim. Callers that require non-empty
/// content (word-window, sentence) check that separately.
///
/// Invariants checked:
///   1. `chunk_index` starts at 0 and increments by 1.
///   2. `doc_text_offset_start <= doc_text_offset_end`.
///   3. `doc_text_offset_end <= concatenated_text.len()`.
///   4. `doc_text_offset_start` is non-decreasing across chunks.
fn assert_offset_invariants(chunks: &[Chunk], concatenated_len: usize) {
    let mut prev_start: Option<usize> = None;
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk.chunk_index, i,
            "chunk_index must be 0-based sequential: expected {i}, got {}",
            chunk.chunk_index
        );
        assert!(
            chunk.doc_text_offset_start <= chunk.doc_text_offset_end,
            "chunk {i}: byte_start ({}) must not exceed byte_end ({})",
            chunk.doc_text_offset_start,
            chunk.doc_text_offset_end
        );
        assert!(
            chunk.doc_text_offset_end <= concatenated_len,
            "chunk {i}: byte_end ({}) must not exceed text length ({concatenated_len})",
            chunk.doc_text_offset_end
        );
        if let Some(ps) = prev_start {
            assert!(
                chunk.doc_text_offset_start >= ps,
                "chunk {i}: doc_text_offset_start ({}) must be >= previous ({ps})",
                chunk.doc_text_offset_start
            );
        }
        prev_start = Some(chunk.doc_text_offset_start);
    }
}

// ---- Normalization properties -----------------------------------------------

proptest! {
    /// NFC normalization is idempotent: applying `normalize_text` twice
    /// produces the same result as applying it once.
    #[test]
    fn prop_nfc_idempotent(s in "\\PC{0,200}") {
        let once = normalize_text(&s);
        let twice = normalize_text(&once);
        prop_assert_eq!(once, twice, "normalize_text must be idempotent");
    }

    /// After normalization, no two consecutive ASCII space characters appear
    /// in the output. The whitespace-collapse pass reduces all runs of spaces,
    /// tabs, and form feeds to a single space.
    #[test]
    fn prop_no_consecutive_spaces(s in "[A-Za-z \t\x0c]{1,200}") {
        let result = normalize_text(&s);
        prop_assert!(
            !result.contains("  "),
            "normalized output must not contain consecutive spaces, got: {result:?}"
        );
    }

    /// A word hyphenated at a line break is correctly joined. The alphabetic
    /// characters on both sides of the hyphen-newline boundary are preserved.
    #[test]
    fn prop_hyphenation_preserves_alpha(
        word1 in "[a-z]{2,10}",
        word2 in "[a-z]{2,10}",
    ) {
        let input = format!("{word1}-\n{word2}");
        let result = normalize_text(&input);
        let expected = format!("{word1}{word2}");
        prop_assert_eq!(
            result.trim(),
            expected.as_str(),
            "hyphenation repair must join the two word halves"
        );
    }

    /// A hyphen not immediately followed by a newline is preserved verbatim.
    #[test]
    fn prop_non_linebreak_hyphen_preserved(
        word1 in "[a-z]{2,10}",
        word2 in "[a-z]{2,10}",
    ) {
        let input = format!("{word1}-{word2}");
        let result = normalize_text(&input);
        // NFC normalization may change combining characters but not ASCII hyphens.
        prop_assert!(
            result.contains('-'),
            "a hyphen not before newline must be preserved in: {result:?}"
        );
    }

    /// Newlines are preserved by normalization because they serve as
    /// inter-page separators in concatenated document text.
    #[test]
    fn prop_newlines_preserved(
        lines in prop::collection::vec("[A-Za-z]{1,20}", 2..=5),
    ) {
        let input = lines.join("\n");
        let result = normalize_text(&input);
        let newline_count_input = input.chars().filter(|&c| c == '\n').count();
        let newline_count_result = result.chars().filter(|&c| c == '\n').count();
        prop_assert_eq!(
            newline_count_input,
            newline_count_result,
            "newlines must be preserved by normalization"
        );
    }
}

// ---- Page strategy properties -----------------------------------------------

proptest! {
    /// The page strategy produces exactly one chunk per input page.
    #[test]
    fn prop_page_strategy_chunk_count_equals_page_count(
        pages in prop::collection::vec("[A-Za-z0-9 ,.;:]{1,100}", 1..=10),
    ) {
        let page_texts = make_pages(pages.clone());
        let chunks = PageStrategy.chunk(&page_texts).unwrap();
        prop_assert_eq!(
            chunks.len(),
            pages.len(),
            "page strategy must produce exactly one chunk per page"
        );
    }

    /// Page strategy chunk indices are 0-based and sequential.
    #[test]
    fn prop_page_strategy_chunk_indices_sequential(
        pages in prop::collection::vec("[A-Za-z0-9 ]{1,50}", 1..=8),
    ) {
        let page_texts = make_pages(pages);
        let chunks = PageStrategy.chunk(&page_texts).unwrap();
        for (i, chunk) in chunks.iter().enumerate() {
            prop_assert_eq!(
                chunk.chunk_index, i,
                "page strategy chunk_index must be sequential"
            );
        }
    }

    /// Page strategy byte offsets are within bounds and non-overlapping in
    /// a monotonically non-decreasing order.
    #[test]
    fn prop_page_strategy_offsets_valid(
        pages in prop::collection::vec("[A-Za-z0-9 ,.]{1,80}", 1..=8),
    ) {
        let page_texts = make_pages(pages);
        let chunks = PageStrategy.chunk(&page_texts).unwrap();

        // The page strategy uses raw content without normalization and joins
        // pages with a single "\n" separator. Compute concatenated_len using
        // the same formula the strategy uses internally.
        let raw_content_refs: Vec<&str> = page_texts.iter().map(|p| p.content.as_str()).collect();
        let concatenated_len = raw_content_refs.join("\n").len();

        assert_offset_invariants(&chunks, concatenated_len);
    }

    /// Page strategy is deterministic: the same input always produces the
    /// same chunks.
    #[test]
    fn prop_page_strategy_deterministic(
        pages in prop::collection::vec("[A-Za-z0-9 ]{1,60}", 1..=6),
    ) {
        let page_texts = make_pages(pages);
        let a = PageStrategy.chunk(&page_texts).unwrap();
        let b = PageStrategy.chunk(&page_texts).unwrap();
        prop_assert_eq!(
            a.len(), b.len(),
            "page strategy must produce same chunk count on repeated calls"
        );
        for (ca, cb) in a.iter().zip(b.iter()) {
            prop_assert_eq!(
                &ca.content, &cb.content,
                "page strategy content must be deterministic"
            );
        }
    }
}

// ---- Word window strategy properties ----------------------------------------

proptest! {
    /// Word window strategy produces chunks with valid byte offsets.
    #[test]
    fn prop_word_window_offsets_valid(
        words in prop::collection::vec("[A-Za-z]{2,12}", 6..=40),
        window in 2usize..=6,
        overlap in 0usize..=2,
    ) {
        prop_assume!(overlap < window);
        let text = words.join(" ");
        let page_texts = vec![make_page(text)];
        let strategy = WordWindowStrategy { window_size: window, overlap };
        let chunks = strategy.chunk(&page_texts).unwrap();

        // The concatenated length is the normalized single page (no inter-page
        // "\n" added because there is only one page, but the strategy still
        // joins with "\n" separator — for a single page this means no separator).
        let normalized = normalize_text(&page_texts[0].content);
        assert_offset_invariants(&chunks, normalized.len());
    }

    /// Every input word appears in at least one chunk produced by the word
    /// window strategy (content preservation).
    #[test]
    fn prop_word_window_content_preserved(
        words in prop::collection::vec("[A-Za-z]{3,10}", 6..=30),
        window in 2usize..=6,
        overlap in 0usize..=2,
    ) {
        prop_assume!(overlap < window);
        let text = words.join(" ");
        let page_texts = vec![make_page(text)];
        let strategy = WordWindowStrategy { window_size: window, overlap };
        let chunks = strategy.chunk(&page_texts).unwrap();
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join(" ");
        for word in &words {
            prop_assert!(
                all_content.contains(word.as_str()),
                "word '{word}' must appear in some chunk; all content: {all_content:?}"
            );
        }
    }

    /// Word window strategy is deterministic: identical inputs produce
    /// identical chunks.
    #[test]
    fn prop_word_window_deterministic(
        words in prop::collection::vec("[A-Za-z]{3,10}", 6..=30),
        window in 2usize..=6,
    ) {
        let text = words.join(" ");
        let page_texts = vec![make_page(text)];
        let strategy = WordWindowStrategy { window_size: window, overlap: 0 };
        let a = strategy.chunk(&page_texts).unwrap();
        let b = strategy.chunk(&page_texts).unwrap();
        prop_assert_eq!(a.len(), b.len());
        for (ca, cb) in a.iter().zip(b.iter()) {
            prop_assert_eq!(&ca.content, &cb.content);
            prop_assert_eq!(ca.doc_text_offset_start, cb.doc_text_offset_start);
            prop_assert_eq!(ca.doc_text_offset_end, cb.doc_text_offset_end);
        }
    }

    /// Word window strategy produces no chunks with empty content.
    #[test]
    fn prop_word_window_no_empty_chunks(
        words in prop::collection::vec("[A-Za-z]{3,10}", 6..=30),
        window in 2usize..=6,
        overlap in 0usize..=2,
    ) {
        prop_assume!(overlap < window);
        let text = words.join(" ");
        let page_texts = vec![make_page(text)];
        let strategy = WordWindowStrategy { window_size: window, overlap };
        let chunks = strategy.chunk(&page_texts).unwrap();
        for chunk in &chunks {
            prop_assert!(
                !chunk.content.trim().is_empty(),
                "word window must not produce empty chunks"
            );
        }
    }

    /// doc_text_offset_start values are non-decreasing across all chunks
    /// from the word window strategy.
    #[test]
    fn prop_word_window_monotonic_offsets(
        words in prop::collection::vec("[A-Za-z]{3,10}", 6..=30),
        window in 2usize..=6,
        overlap in 0usize..=2,
    ) {
        prop_assume!(overlap < window);
        let text = words.join(" ");
        let page_texts = vec![make_page(text)];
        let strategy = WordWindowStrategy { window_size: window, overlap };
        let chunks = strategy.chunk(&page_texts).unwrap();
        let starts: Vec<usize> = chunks.iter().map(|c| c.doc_text_offset_start).collect();
        prop_assert!(
            starts.windows(2).all(|w| w[0] <= w[1]),
            "word window offsets must be monotonically non-decreasing: {starts:?}"
        );
    }
}

// ---- Sentence strategy properties -------------------------------------------

proptest! {
    /// Sentence strategy produces chunks with valid byte offsets into the
    /// normalized concatenated text.
    #[test]
    fn prop_sentence_offsets_valid(
        sentences in prop::collection::vec("[A-Z][a-z ]{5,25}\\.", 2..=8),
        max_words in 10usize..=80,
    ) {
        let text = sentences.join(" ");
        let page_texts = vec![make_page(text)];
        let strategy = SentenceStrategy { max_words };
        let chunks = strategy.chunk(&page_texts).unwrap();
        let normalized = normalize_text(&page_texts[0].content);
        assert_offset_invariants(&chunks, normalized.len());
    }

    /// Sentence strategy chunk offsets are monotonically non-decreasing.
    #[test]
    fn prop_sentence_monotonic_offsets(
        sentences in prop::collection::vec("[A-Z][a-z ]{5,25}\\.", 2..=8),
        max_words in 15usize..=80,
    ) {
        let text = sentences.join(" ");
        let page_texts = vec![make_page(text)];
        let strategy = SentenceStrategy { max_words };
        let chunks = strategy.chunk(&page_texts).unwrap();
        let starts: Vec<usize> = chunks.iter().map(|c| c.doc_text_offset_start).collect();
        prop_assert!(
            starts.windows(2).all(|w| w[0] <= w[1]),
            "sentence offsets must be monotonically non-decreasing: {starts:?}"
        );
    }

    /// Sentence strategy produces no chunks with empty content.
    #[test]
    fn prop_sentence_no_empty_chunks(
        sentences in prop::collection::vec("[A-Z][a-z ]{5,25}\\.", 2..=8),
        max_words in 10usize..=80,
    ) {
        let text = sentences.join(" ");
        let page_texts = vec![make_page(text)];
        let strategy = SentenceStrategy { max_words };
        let chunks = strategy.chunk(&page_texts).unwrap();
        for chunk in &chunks {
            prop_assert!(
                !chunk.content.trim().is_empty(),
                "sentence strategy must not produce empty chunks"
            );
        }
    }

    /// Sentence strategy is deterministic.
    #[test]
    fn prop_sentence_deterministic(
        sentences in prop::collection::vec("[A-Z][a-z ]{5,25}\\.", 2..=8),
        max_words in 20usize..=80,
    ) {
        let text = sentences.join(" ");
        let page_texts = vec![make_page(text)];
        let strategy = SentenceStrategy { max_words };
        let a = strategy.chunk(&page_texts).unwrap();
        let b = strategy.chunk(&page_texts).unwrap();
        prop_assert_eq!(a.len(), b.len());
        for (ca, cb) in a.iter().zip(b.iter()) {
            prop_assert_eq!(&ca.content, &cb.content);
            prop_assert_eq!(ca.doc_text_offset_start, cb.doc_text_offset_start);
        }
    }

    /// The chunk content stored in each sentence chunk matches the byte slice
    /// of the normalized concatenated text at [doc_text_offset_start, doc_text_offset_end).
    /// This verifies that byte offsets and content are internally consistent.
    #[test]
    fn prop_sentence_content_matches_offset_slice(
        sentences in prop::collection::vec("[A-Z][a-z ]{5,25}\\.", 2..=8),
        max_words in 20usize..=80,
    ) {
        let text = sentences.join(" ");
        let page_texts = vec![make_page(text)];
        let normalized = normalize_text(&page_texts[0].content);
        let strategy = SentenceStrategy { max_words };
        let chunks = strategy.chunk(&page_texts).unwrap();
        for chunk in &chunks {
            let slice = &normalized[chunk.doc_text_offset_start..chunk.doc_text_offset_end];
            prop_assert_eq!(
                slice.trim(),
                chunk.content.trim(),
                "sentence chunk content must match the byte slice at its offsets"
            );
        }
    }
}

// ---- Cross-strategy properties ----------------------------------------------

proptest! {
    /// All chunking strategies produce chunks whose chunk_index values are
    /// 0-based and strictly sequential. Tested across page, word-window, and
    /// sentence strategies with a shared input.
    #[test]
    fn prop_all_strategies_sequential_chunk_index(
        words in prop::collection::vec("[A-Za-z]{3,10}", 8..=30),
    ) {
        let text = words.join(" ");

        // Page strategy: single page.
        let pages = vec![make_page(text.clone())];
        let page_chunks = PageStrategy.chunk(&pages).unwrap();
        for (i, c) in page_chunks.iter().enumerate() {
            prop_assert_eq!(c.chunk_index, i);
        }

        // Word window strategy: window=4, overlap=1.
        let ww_chunks = WordWindowStrategy { window_size: 4, overlap: 1 }
            .chunk(&pages)
            .unwrap();
        for (i, c) in ww_chunks.iter().enumerate() {
            prop_assert_eq!(c.chunk_index, i);
        }

        // Sentence strategy: max_words=50.
        let sent_chunks = SentenceStrategy { max_words: 50 }
            .chunk(&pages)
            .unwrap();
        for (i, c) in sent_chunks.iter().enumerate() {
            prop_assert_eq!(c.chunk_index, i);
        }
    }
}
