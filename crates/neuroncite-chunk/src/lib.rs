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

//! neuroncite-chunk: Text splitting strategies for extracted PDF content.
//!
//! This crate takes the plain-text output produced by `neuroncite-pdf` and splits
//! it into smaller overlapping chunks suitable for embedding. Four chunking
//! strategies are provided:
//!
//! - **Page** -- One chunk per PDF page (no splitting within pages).
//! - **Word window** -- Fixed-size windows measured in whitespace-delimited words
//!   with configurable overlap.
//! - **Token window** -- Fixed-size windows measured in subword tokens using a
//!   caller-provided tokenizer, with configurable overlap.
//! - **Sentence** -- Groups of consecutive sentences that fit within a word budget,
//!   with abbreviation-aware boundary suppression to avoid mid-citation breaks.
//!
//! All strategies produce chunks annotated with byte offsets back into the original
//! page text, enabling precise page-and-character citation references.

#![forbid(unsafe_code)]
// `deny(warnings)` is inherited from [workspace.lints.rust] in the root Cargo.toml.

pub mod abbreviations;
pub mod error;
pub mod normalize;
pub mod offset;
pub mod page;
pub mod sentence;
pub mod token_window;
pub mod word_window;

pub use error::ChunkError;

use neuroncite_core::traits::ChunkStrategy;

use crate::page::PageStrategy;
use crate::sentence::SentenceStrategy;
use crate::token_window::TokenWindowStrategy;
use crate::word_window::WordWindowStrategy;

/// Factory function that constructs a boxed `ChunkStrategy` from a strategy
/// identifier string and optional parameters.
///
/// # Arguments
///
/// * `strategy` - The strategy name: "page", "word", "token", or "sentence".
/// * `chunk_size` - The window size in words (for "word") or token limit
///   (for "token"). Required for "word" and "token" strategies.
/// * `overlap` - The overlap in words (for "word") or tokens (for "token").
///   Defaults to 0 if not provided.
/// * `max_words` - The maximum word count per chunk (for "sentence").
///   Required for the "sentence" strategy.
/// * `tokenizer_json` - The embedding model's tokenizer serialized as a JSON
///   string (for "token"). Required for the "token" strategy. The JSON is
///   produced by `EmbeddingBackend::tokenizer_json()` and deserialized here
///   via `tokenizers::Tokenizer::from_str()`. This design keeps the
///   `tokenizers` crate encapsulated within neuroncite-chunk so that calling
///   crates (neuroncite, neuroncite-api) do not need a direct dependency on it.
///
/// # Errors
///
/// Returns `ChunkError::InvalidConfig` if the strategy name is unrecognized,
/// if required parameters are missing for the chosen strategy, or if the
/// tokenizer JSON fails to deserialize.
///
/// Returns `ChunkError::OverlapExceedsWindow` if the overlap value equals or
/// exceeds the window/chunk size.
pub fn create_strategy(
    strategy: &str,
    chunk_size: Option<usize>,
    overlap: Option<usize>,
    max_words: Option<usize>,
    tokenizer_json: Option<&str>,
) -> Result<Box<dyn ChunkStrategy>, ChunkError> {
    match strategy {
        "page" => Ok(Box::new(PageStrategy)),

        "word" => {
            let window_size = chunk_size.ok_or_else(|| {
                ChunkError::InvalidConfig("word strategy requires chunk_size parameter".into())
            })?;

            if window_size == 0 {
                return Err(ChunkError::InvalidConfig(
                    "window_size must be greater than zero".into(),
                ));
            }

            let word_overlap = overlap.unwrap_or(0);

            if word_overlap >= window_size {
                return Err(ChunkError::OverlapExceedsWindow {
                    overlap: word_overlap,
                    window: window_size,
                });
            }

            Ok(Box::new(WordWindowStrategy {
                window_size,
                overlap: word_overlap,
            }))
        }

        "token" => {
            let token_limit = chunk_size.ok_or_else(|| {
                ChunkError::InvalidConfig("token strategy requires chunk_size parameter".into())
            })?;

            if token_limit == 0 {
                return Err(ChunkError::InvalidConfig(
                    "token_limit must be greater than zero".into(),
                ));
            }

            let token_overlap = overlap.unwrap_or(0);

            if token_overlap >= token_limit {
                return Err(ChunkError::OverlapExceedsWindow {
                    overlap: token_overlap,
                    window: token_limit,
                });
            }

            let json = tokenizer_json.ok_or_else(|| {
                ChunkError::InvalidConfig(
                    "token strategy requires a tokenizer (load an embedding model first)".into(),
                )
            })?;

            let tok = tokenizers::Tokenizer::from_bytes(json).map_err(|e| {
                ChunkError::InvalidConfig(format!("failed to deserialize tokenizer JSON: {e}"))
            })?;

            Ok(Box::new(TokenWindowStrategy {
                token_limit,
                token_overlap,
                tokenizer: tok,
            }))
        }

        "sentence" => {
            // Default to 256 words when max_words is not provided. This matches
            // the default chunk_size used by the index handler and prevents
            // callers from getting an error when calling preview_chunks with
            // chunk_strategy="sentence" without explicit max_words or chunk_size.
            let words = max_words.unwrap_or(256);

            if words == 0 {
                return Err(ChunkError::InvalidConfig(
                    "max_words must be greater than zero".into(),
                ));
            }

            Ok(Box::new(SentenceStrategy { max_words: words }))
        }

        unknown => Err(ChunkError::InvalidConfig(format!(
            "unrecognized strategy: '{unknown}'"
        ))),
    }
}

/// Property-based tests using `proptest` that verify structural invariants
/// across all chunking strategies and the Unicode normalizer. Each property
/// runs against 256 randomly generated inputs with automatic shrinking on
/// failure, covering the full input space rather than hand-picked examples.
#[cfg(test)]
mod property_tests;

/// Shared test utilities for the neuroncite-chunk crate. Contains helper
/// functions used by test modules across multiple sub-modules (page,
/// word_window, token_window, sentence) to avoid code duplication.
#[cfg(test)]
pub(crate) mod test_helpers {
    use std::path::PathBuf;

    use neuroncite_core::types::{ExtractionBackend, PageText};

    /// Creates a vector of `PageText` values from an array of content strings.
    /// Each string becomes one page with a 1-indexed page number and a fixed
    /// source file path ("/test/doc.pdf"). The `PdfExtract` backend is assigned
    /// to all pages. This helper is used across all chunking strategy test
    /// modules to construct consistent test input.
    pub(crate) fn make_pages(contents: &[&str]) -> Vec<PageText> {
        contents
            .iter()
            .enumerate()
            .map(|(i, &text)| PageText {
                source_file: PathBuf::from("/test/doc.pdf"),
                page_number: i + 1,
                content: text.into(),
                backend: ExtractionBackend::PdfExtract,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use neuroncite_core::types::Chunk;

    use super::*;
    use crate::test_helpers::make_pages;

    /// T-CHK-017: Processing the same PDF with the same strategy and
    /// parameters twice produces identical chunk lists (same text, same
    /// offsets, same hashes).
    #[test]
    fn t_chk_017_chunk_determinism() {
        let pages = make_pages(&[
            "First page with some text content here.",
            "Second page has different words and sentences.",
            "Third page concludes the document.",
        ]);

        let strategy = create_strategy("word", Some(5), Some(1), None, None)
            .expect("strategy creation failed");

        let chunks_run1 = strategy.chunk(&pages).expect("first run failed");
        let chunks_run2 = strategy.chunk(&pages).expect("second run failed");

        assert_eq!(
            chunks_run1.len(),
            chunks_run2.len(),
            "chunk count differs between runs"
        );

        for (c1, c2) in chunks_run1.iter().zip(chunks_run2.iter()) {
            assert_eq!(c1.content, c2.content, "content differs");
            assert_eq!(
                c1.doc_text_offset_start, c2.doc_text_offset_start,
                "offset_start differs"
            );
            assert_eq!(
                c1.doc_text_offset_end, c2.doc_text_offset_end,
                "offset_end differs"
            );
            assert_eq!(c1.content_hash, c2.content_hash, "content_hash differs");
            assert_eq!(c1.chunk_index, c2.chunk_index, "chunk_index differs");
            assert_eq!(c1.page_start, c2.page_start, "page_start differs");
            assert_eq!(c1.page_end, c2.page_end, "page_end differs");
        }
    }

    /// T-CHK-018: The `content_hash` of each chunk equals the SHA-256 hex
    /// digest of the chunk's content bytes.
    #[test]
    fn t_chk_018_content_hash_correctness() {
        let pages = make_pages(&[
            "Some sample text for hashing verification.",
            "Another page with different content.",
        ]);

        let strategy = create_strategy("word", Some(4), Some(1), None, None)
            .expect("strategy creation failed");
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        for chunk in &chunks {
            let recomputed = Chunk::compute_content_hash(&chunk.content);
            assert_eq!(
                chunk.content_hash, recomputed,
                "content_hash does not match SHA-256 of content for chunk {}",
                chunk.chunk_index
            );
        }
    }

    /// Verifies that the factory function rejects unrecognized strategy names.
    #[test]
    fn unknown_strategy_rejected() {
        let result = create_strategy("unknown", None, None, None, None);
        assert!(result.is_err());
    }

    /// Verifies that the factory function rejects overlap >= window_size.
    #[test]
    fn overlap_exceeds_window_rejected() {
        let result = create_strategy("word", Some(5), Some(5), None, None);
        assert!(result.is_err());

        let result = create_strategy("word", Some(5), Some(10), None, None);
        assert!(result.is_err());
    }

    /// Verifies that the "word" strategy requires chunk_size.
    #[test]
    fn word_strategy_requires_chunk_size() {
        let result = create_strategy("word", None, None, None, None);
        assert!(result.is_err());
    }

    /// Verifies that the "sentence" strategy defaults to 256 max_words when
    /// the parameter is omitted, instead of returning an error.
    #[test]
    fn sentence_strategy_defaults_max_words() {
        let result = create_strategy("sentence", None, None, None, None);
        assert!(
            result.is_ok(),
            "sentence strategy must succeed with default max_words=256"
        );
    }

    /// Verifies that the "token" strategy requires both chunk_size and
    /// tokenizer.
    #[test]
    fn token_strategy_requires_params() {
        let result = create_strategy("token", None, None, None, None);
        assert!(result.is_err());

        let result = create_strategy("token", Some(10), None, None, None);
        assert!(result.is_err());
    }

    /// Verifies that the "page" strategy can be created without parameters.
    #[test]
    fn page_strategy_creation() {
        let result = create_strategy("page", None, None, None, None);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------
    // Token strategy factory tests
    // -------------------------------------------------------------------

    /// Builds a minimal word-level tokenizer for testing and returns its
    /// JSON serialization. Each whitespace-delimited word becomes one token.
    fn build_test_tokenizer_json() -> String {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;

        let vocab_entries: Vec<(String, u32)> = vec![
            ("hello".into(), 0),
            ("world".into(), 1),
            ("foo".into(), 2),
            ("bar".into(), 3),
            ("baz".into(), 4),
            ("[UNK]".into(), 5),
        ];

        let vocab = vocab_entries.into_iter().collect();

        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .expect("WordLevel model construction failed");

        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));

        tokenizer
            .to_string(false)
            .expect("tokenizer JSON serialization failed")
    }

    /// T-CHK-019: The factory function creates a token strategy when given
    /// valid chunk_size and tokenizer JSON parameters.
    #[test]
    fn t_chk_019_token_strategy_creation_with_valid_json() {
        let json = build_test_tokenizer_json();
        let result = create_strategy("token", Some(128), Some(16), None, Some(&json));
        assert!(
            result.is_ok(),
            "token strategy creation must succeed with valid tokenizer JSON"
        );
    }

    /// T-CHK-020: The factory function rejects a token strategy when the
    /// tokenizer JSON is malformed (cannot be deserialized).
    #[test]
    fn t_chk_020_token_strategy_rejects_invalid_json() {
        let result = create_strategy("token", Some(128), Some(16), None, Some("not valid json"));
        assert!(
            result.is_err(),
            "token strategy creation must fail with invalid tokenizer JSON"
        );
    }

    /// T-CHK-021: The factory function rejects a token strategy when
    /// token_overlap >= token_limit.
    #[test]
    fn t_chk_021_token_overlap_exceeds_limit() {
        let json = build_test_tokenizer_json();

        // overlap == limit
        let result = create_strategy("token", Some(10), Some(10), None, Some(&json));
        assert!(
            result.is_err(),
            "token strategy must reject overlap equal to token_limit"
        );

        // overlap > limit
        let result = create_strategy("token", Some(10), Some(20), None, Some(&json));
        assert!(
            result.is_err(),
            "token strategy must reject overlap greater than token_limit"
        );
    }

    /// T-CHK-022: The factory function rejects a token strategy with
    /// zero chunk_size.
    #[test]
    fn t_chk_022_token_zero_chunk_size() {
        let json = build_test_tokenizer_json();
        let result = create_strategy("token", Some(0), None, None, Some(&json));
        assert!(
            result.is_err(),
            "token strategy must reject zero token_limit"
        );
    }

    /// T-CHK-023: End-to-end test of the token strategy created via the
    /// factory function. Verifies that chunking produces non-empty results
    /// with valid page ranges and byte offsets.
    #[test]
    fn t_chk_023_token_strategy_end_to_end_via_factory() {
        let json = build_test_tokenizer_json();
        let strategy = create_strategy("token", Some(3), Some(1), None, Some(&json))
            .expect("token strategy creation failed");

        let pages = make_pages(&["hello world foo bar baz"]);
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        assert!(
            !chunks.is_empty(),
            "token strategy must produce at least one chunk"
        );

        for chunk in &chunks {
            assert!(!chunk.content.is_empty(), "chunk content must not be empty");
            assert!(
                chunk.doc_text_offset_end > chunk.doc_text_offset_start,
                "chunk byte range must be non-empty"
            );
            assert!(chunk.page_start >= 1, "page_start must be >= 1");
            assert!(
                chunk.page_end >= chunk.page_start,
                "page_end must be >= page_start"
            );
        }

        // Verify chunk indices are sequential starting from 0.
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(
                chunk.chunk_index, i,
                "chunk_index must equal sequential position"
            );
        }
    }

    /// T-CHK-024: Token strategy with default overlap (None) produces
    /// non-overlapping chunks. When overlap is not specified, the factory
    /// defaults to 0.
    #[test]
    fn t_chk_024_token_strategy_default_overlap_zero() {
        let json = build_test_tokenizer_json();
        let strategy = create_strategy("token", Some(2), None, None, Some(&json))
            .expect("token strategy creation failed");

        let pages = make_pages(&["hello world foo bar"]);
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        // With 4 words (tokens) and limit=2, overlap=0, there should be 2 chunks.
        assert_eq!(
            chunks.len(),
            2,
            "4 tokens / 2 per chunk / 0 overlap must produce 2 chunks"
        );

        // Chunks must not overlap: second chunk starts where first ends.
        assert!(
            chunks[1].doc_text_offset_start >= chunks[0].doc_text_offset_end,
            "with zero overlap, chunks must not share content"
        );
    }

    /// T-CHK-025: Token strategy determinism -- identical input produces
    /// identical output when using the same tokenizer.
    #[test]
    fn t_chk_025_token_strategy_determinism() {
        let json = build_test_tokenizer_json();

        let pages = make_pages(&["hello world foo", "bar baz hello"]);

        let strategy1 = create_strategy("token", Some(3), Some(1), None, Some(&json))
            .expect("first strategy creation failed");
        let strategy2 = create_strategy("token", Some(3), Some(1), None, Some(&json))
            .expect("second strategy creation failed");

        let chunks1 = strategy1.chunk(&pages).expect("first run failed");
        let chunks2 = strategy2.chunk(&pages).expect("second run failed");

        assert_eq!(
            chunks1.len(),
            chunks2.len(),
            "chunk count must be identical across runs"
        );

        for (c1, c2) in chunks1.iter().zip(chunks2.iter()) {
            assert_eq!(c1.content, c2.content, "content must be identical");
            assert_eq!(
                c1.doc_text_offset_start, c2.doc_text_offset_start,
                "offset_start must be identical"
            );
            assert_eq!(
                c1.doc_text_offset_end, c2.doc_text_offset_end,
                "offset_end must be identical"
            );
            assert_eq!(
                c1.content_hash, c2.content_hash,
                "content_hash must be identical"
            );
        }
    }

    /// T-CHK-026: The word strategy zero chunk_size is rejected.
    #[test]
    fn t_chk_026_word_zero_chunk_size() {
        let result = create_strategy("word", Some(0), None, None, None);
        assert!(
            result.is_err(),
            "word strategy must reject zero window_size"
        );
    }

    /// T-CHK-027: The sentence strategy zero max_words is rejected.
    #[test]
    fn t_chk_027_sentence_zero_max_words() {
        let result = create_strategy("sentence", None, None, Some(0), None);
        assert!(
            result.is_err(),
            "sentence strategy must reject zero max_words"
        );
    }

    /// T-CHK-028: The sentence strategy uses explicit max_words when provided.
    /// Regression test for NEW-1: verifies that explicit max_words overrides
    /// the default and is not ignored.
    #[test]
    fn t_chk_028_sentence_explicit_max_words() {
        let result = create_strategy("sentence", None, None, Some(100), None);
        assert!(
            result.is_ok(),
            "sentence strategy must succeed with explicit max_words=100"
        );
    }

    /// T-CHK-029: The sentence strategy defaults to 256 when chunk_size is
    /// provided but max_words is not. The chunk_size parameter is mapped to
    /// max_words by the preview_chunks handler, but the factory function
    /// itself only looks at max_words. Verifies the default still applies.
    #[test]
    fn t_chk_029_sentence_defaults_with_chunk_size_only() {
        // chunk_size is not used by the sentence strategy factory path;
        // it is only mapped to max_words by the handler. The factory
        // receives max_words=None and should default to 256.
        let result = create_strategy("sentence", Some(100), None, None, None);
        assert!(
            result.is_ok(),
            "sentence strategy must default max_words when chunk_size is provided but max_words is not"
        );
    }
}
