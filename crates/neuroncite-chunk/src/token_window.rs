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

//! Token-window chunking strategy.
//!
//! Splits text into fixed-size windows measured in subword tokens using a
//! caller-provided `tokenizers::Tokenizer` instance. The tokenizer is injected
//! by the API handler or binary crate (which obtains it from the embedding
//! backend), avoiding a compile-time dependency from this crate to
//! neuroncite-embed.
//!
//! Each window overlaps with its predecessor by a configurable number of
//! tokens. Chunk content is extracted by slicing the normalized document
//! string using the tokenizer's byte offset mappings (the `offsets` field
//! of each token), rather than by calling the tokenizer's `decode()` method.
//! This guarantees that each chunk is a verbatim substring of the normalized
//! document text, preserving exact byte offsets for citation generation.

use neuroncite_core::error::NeuronCiteError;
use neuroncite_core::traits::ChunkStrategy;
use neuroncite_core::types::{Chunk, PageText};

use crate::normalize::normalize_text;
use crate::offset::compute_page_range;

/// Token-window chunking strategy that measures chunk boundaries in subword
/// tokens using the model's tokenizer.
pub struct TokenWindowStrategy {
    /// Maximum number of tokens per chunk.
    pub token_limit: usize,
    /// Number of tokens shared between consecutive chunks. Must be strictly
    /// less than `token_limit`.
    pub token_overlap: usize,
    /// The Hugging Face `tokenizers` crate instance used for subword
    /// tokenization. This is the same tokenizer that the embedding model
    /// uses, ensuring that chunk boundaries align with the model's token
    /// budget.
    pub tokenizer: tokenizers::Tokenizer,
}

impl ChunkStrategy for TokenWindowStrategy {
    /// Splits the concatenated, normalized document text into overlapping
    /// token-window chunks.
    ///
    /// The algorithm tokenizes the full normalized text, then slides a window
    /// of `token_limit` tokens forward by `(token_limit - token_overlap)` at
    /// each step. Content is extracted from the original text using tokenizer
    /// byte offsets, not via `decode()`.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Chunk` if the input pages are empty or if
    /// the tokenizer fails to process the text.
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

        // Tokenize the full normalized text. The encoding result provides
        // token IDs and byte offset mappings for each token.
        let encoding = self
            .tokenizer
            .encode(normalized.as_str(), false)
            .map_err(|e| NeuronCiteError::Chunk(format!("tokenizer encoding failed: {e}")))?;

        // Collect the byte offset pairs for each token. Each offset is a
        // (start, end) pair in the original text. Special tokens (CLS, SEP)
        // produced by the tokenizer may have (0, 0) offsets and are filtered
        // out to avoid producing empty or incorrect chunks.
        let offsets: Vec<(usize, usize)> = encoding
            .get_offsets()
            .iter()
            .filter(|&&(s, e)| s != e) // skip special tokens with zero-width offsets
            .copied()
            .collect();

        if offsets.is_empty() {
            return Ok(Vec::new());
        }

        let step = self.token_limit - self.token_overlap;
        let mut chunks = Vec::new();
        let mut chunk_index: usize = 0;
        let mut token_offset: usize = 0;

        while token_offset < offsets.len() {
            let window_end = (token_offset + self.token_limit).min(offsets.len());
            let window_offsets = &offsets[token_offset..window_end];

            // The chunk spans from the start of the first token to the end
            // of the last token in this window.
            let doc_offset_start = window_offsets[0].0;
            let doc_offset_end = window_offsets[window_offsets.len() - 1].1;

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

            if window_end >= offsets.len() {
                break;
            }

            token_offset += step;
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_pages;

    /// Builds a minimal word-level tokenizer suitable for testing. Each
    /// whitespace-delimited word becomes one token. The tokenizer provides
    /// byte offset mappings that map back to the original text. No special
    /// tokens (CLS/SEP) are added.
    fn build_test_tokenizer() -> tokenizers::Tokenizer {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;

        // Word-level vocabulary. Each (word, id) pair is a vocabulary entry.
        // The vocabulary is built as an array of tuples and collected into
        // the `AHashMap` type expected by the `WordLevel` builder.
        let vocab_entries: Vec<(String, u32)> = vec![
            ("hello".into(), 0),
            ("world".into(), 1),
            ("foo".into(), 2),
            ("bar".into(), 3),
            ("baz".into(), 4),
            ("one".into(), 5),
            ("two".into(), 6),
            ("three".into(), 7),
            ("four".into(), 8),
            ("five".into(), 9),
            ("six".into(), 10),
            ("seven".into(), 11),
            ("eight".into(), 12),
            ("nine".into(), 13),
            ("ten".into(), 14),
            ("abcde".into(), 15),
            ("fghij".into(), 16),
            ("klmno".into(), 17),
            ("pqrst".into(), 18),
            ("uvwxy".into(), 19),
            ("zabcd".into(), 20),
            ("efghi".into(), 21),
            ("the".into(), 22),
            ("quick".into(), 23),
            ("brown".into(), 24),
            ("fox".into(), 25),
            ("jumps".into(), 26),
            ("over".into(), 27),
            ("lazy".into(), 28),
            ("dog".into(), 29),
            ("[UNK]".into(), 30),
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
    }

    /// T-CHK-007: Each chunk, when tokenized by the model's tokenizer,
    /// produces at most `token_limit` tokens.
    #[test]
    fn t_chk_007_token_count_within_limit() {
        let tokenizer = build_test_tokenizer();
        let text = "abcde fghij klmno pqrst uvwxy";
        let pages = make_pages(&[text]);

        let strategy = TokenWindowStrategy {
            token_limit: 3,
            token_overlap: 1,
            tokenizer: tokenizer.clone(),
        };

        let chunks = strategy.chunk(&pages).expect("chunking failed");

        for chunk in &chunks {
            let encoding = tokenizer
                .encode(chunk.content.as_str(), false)
                .expect("encoding failed");
            let token_count = encoding.get_ids().len();
            assert!(
                token_count <= 3,
                "chunk has {token_count} tokens, exceeds limit of 3"
            );
        }
    }

    /// T-CHK-008: The last `token_overlap` tokens of chunk i match the first
    /// `token_overlap` tokens of chunk i+1 (verified via tokenizer output).
    #[test]
    fn t_chk_008_token_overlap_matches() {
        let tokenizer = build_test_tokenizer();
        let text = "abcde fghij klmno pqrst uvwxy zabcd efghi";
        let pages = make_pages(&[text]);

        let overlap = 1;
        let strategy = TokenWindowStrategy {
            token_limit: 3,
            token_overlap: overlap,
            tokenizer: tokenizer.clone(),
        };

        let chunks = strategy.chunk(&pages).expect("chunking failed");

        for i in 0..chunks.len() - 1 {
            let enc_i = tokenizer
                .encode(chunks[i].content.as_str(), false)
                .expect("encoding failed");
            let enc_next = tokenizer
                .encode(chunks[i + 1].content.as_str(), false)
                .expect("encoding failed");

            let ids_i = enc_i.get_ids();
            let ids_next = enc_next.get_ids();

            let tail_of_i = &ids_i[ids_i.len() - overlap..];
            let head_of_next = &ids_next[..overlap.min(ids_next.len())];

            assert_eq!(
                tail_of_i,
                head_of_next,
                "token overlap mismatch between chunk {i} and chunk {}",
                i + 1
            );
        }
    }

    /// T-CHK-009: Each chunk's content is a verbatim (byte-exact) substring
    /// of the concatenated, normalized document text at the offsets
    /// `[doc_text_offset_start, doc_text_offset_end)`.
    #[test]
    fn t_chk_009_verbatim_substring() {
        let tokenizer = build_test_tokenizer();
        let pages = make_pages(&["hello world", "foo bar baz"]);

        let strategy = TokenWindowStrategy {
            token_limit: 3,
            token_overlap: 1,
            tokenizer,
        };

        let chunks = strategy.chunk(&pages).expect("chunking failed");

        // Reconstruct the normalized concatenated text
        let raw_concat = "hello world\nfoo bar baz";
        let normalized = crate::normalize::normalize_text(raw_concat);

        for chunk in &chunks {
            let expected = &normalized[chunk.doc_text_offset_start..chunk.doc_text_offset_end];
            assert_eq!(
                chunk.content, expected,
                "chunk content does not match the verbatim substring"
            );
        }
    }

    /// Verifies that empty pages produce an error.
    #[test]
    fn empty_pages_returns_error() {
        let tokenizer = build_test_tokenizer();
        let strategy = TokenWindowStrategy {
            token_limit: 10,
            token_overlap: 2,
            tokenizer,
        };
        let result = strategy.chunk(&[]);
        assert!(result.is_err());
    }
}
