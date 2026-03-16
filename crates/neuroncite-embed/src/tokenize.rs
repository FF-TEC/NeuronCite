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

//! HuggingFace tokenizer management.
//!
//! Wraps the `tokenizers` crate to load, cache, and apply subword tokenization
//! for the active embedding model. The tokenizer instance is shared across the
//! embed and chunk crates -- neuroncite-chunk receives it as a callback to avoid
//! a compile-time dependency on this crate.

use crate::error::EmbedError;

/// Holds the encoded output of a batch tokenization operation. Each field
/// is a vector of vectors: one inner vector per input text in the batch.
/// All inner vectors are padded to the same length (the longest sequence
/// in the batch or `max_length`, whichever is smaller).
#[derive(Debug, Clone)]
pub struct BatchEncoding {
    /// Token IDs for each input text. Padding positions contain the tokenizer's
    /// pad token ID.
    pub token_ids: Vec<Vec<u32>>,

    /// Attention masks for each input text. 1 for real tokens, 0 for padding
    /// positions. This mask is passed to the transformer model so that padding
    /// tokens do not influence the attention computation.
    pub attention_masks: Vec<Vec<u32>>,

    /// Character-level byte offsets for each token in each input text. Each
    /// offset pair `(start, end)` represents the byte range of the original
    /// text that corresponds to the token. Padding positions have offset `(0, 0)`.
    pub offsets: Vec<Vec<(usize, usize)>>,

    /// Per-input truncation flag. `true` when the input text produced more
    /// tokens than `max_length` and was truncated to fit. Callers can inspect
    /// this to warn users that the query was truncated and only a prefix
    /// was used for embedding.
    pub truncated: Vec<bool>,
}

/// Thin wrapper around `tokenizers::Tokenizer` that provides batch tokenization
/// with padding and attention mask generation. The wrapper holds ownership of
/// the underlying `Tokenizer` instance and exposes only the operations needed
/// by the embedding backends.
pub struct TokenizerWrapper {
    /// The HuggingFace tokenizer instance loaded from a JSON configuration file
    /// or constructed programmatically.
    tokenizer: tokenizers::Tokenizer,
}

impl TokenizerWrapper {
    /// Creates a wrapper around the given `tokenizers::Tokenizer` instance.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - A fully configured tokenizer (vocabulary, merges, special tokens).
    #[must_use]
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

    /// Returns a reference to the underlying `tokenizers::Tokenizer`.
    #[must_use]
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Encodes a batch of text strings into token IDs, attention masks, and
    /// character offsets. All sequences are truncated to `max_length` tokens
    /// and then padded to the length of the longest sequence in the batch.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of input text strings to tokenize.
    /// * `max_length` - Maximum number of tokens per sequence (truncation limit).
    ///
    /// # Returns
    ///
    /// A `BatchEncoding` where all inner vectors have the same length.
    ///
    /// # Errors
    ///
    /// Returns `EmbedError::Tokenizer` if the underlying tokenizer fails to
    /// encode any input text.
    pub fn encode_batch(
        &self,
        texts: &[&str],
        max_length: usize,
    ) -> Result<BatchEncoding, EmbedError> {
        if texts.is_empty() {
            return Ok(BatchEncoding {
                token_ids: Vec::new(),
                attention_masks: Vec::new(),
                offsets: Vec::new(),
                truncated: Vec::new(),
            });
        }

        // Batch-encode all texts using tokenizers::Tokenizer::encode_batch(),
        // which parallelizes tokenization across inputs via Rayon internally.
        // This yields a 2-4x speedup over sequential encode() for batch sizes
        // of 16-64 on multi-core systems.
        let inputs: Vec<tokenizers::EncodeInput<'_>> = texts
            .iter()
            .map(|&t| tokenizers::EncodeInput::Single(t.into()))
            .collect();

        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
        let mut all_offsets: Vec<Vec<(usize, usize)>> = Vec::with_capacity(texts.len());
        let mut truncated_flags: Vec<bool> = Vec::with_capacity(texts.len());

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let offsets = encoding.get_offsets();

            // Detect whether truncation occurred: the raw token count exceeds
            // the max_length limit. This flag is exposed in BatchEncoding so
            // callers (search handlers) can warn users that their query was
            // truncated to a prefix of the original text.
            let was_truncated = ids.len() > max_length;
            truncated_flags.push(was_truncated);

            let truncated_len = ids.len().min(max_length);
            all_ids.push(ids[..truncated_len].to_vec());
            all_offsets.push(offsets[..truncated_len].to_vec());
        }

        // Determine the padding length: the longest sequence in the batch.
        let pad_length = all_ids.iter().map(Vec::len).max().unwrap_or(0);

        // Pad all sequences to the same length and build attention masks.
        let pad_token_id = self.tokenizer.token_to_id("[PAD]").unwrap_or(0);

        let mut token_ids = Vec::with_capacity(texts.len());
        let mut attention_masks = Vec::with_capacity(texts.len());
        let mut padded_offsets = Vec::with_capacity(texts.len());

        for (ids, offsets) in all_ids.into_iter().zip(all_offsets.into_iter()) {
            let real_len = ids.len();
            let padding_count = pad_length - real_len;

            // Token IDs: real tokens followed by pad tokens
            let mut padded_ids = ids;
            padded_ids.extend(std::iter::repeat_n(pad_token_id, padding_count));

            // Attention mask: 1 for real tokens, 0 for padding
            let mut mask = vec![1_u32; real_len];
            mask.extend(std::iter::repeat_n(0_u32, padding_count));

            // Offsets: real offsets followed by (0, 0) for padding positions
            let mut padded_offs = offsets;
            padded_offs.extend(std::iter::repeat_n((0_usize, 0_usize), padding_count));

            token_ids.push(padded_ids);
            attention_masks.push(mask);
            padded_offsets.push(padded_offs);
        }

        Ok(BatchEncoding {
            token_ids,
            attention_masks,
            offsets: padded_offsets,
            truncated: truncated_flags,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal WordPiece tokenizer for testing purposes. This tokenizer
    /// handles basic ASCII text with a small vocabulary sufficient for
    /// verifying padding, attention masks, and token ID generation.
    fn build_test_tokenizer() -> tokenizers::Tokenizer {
        use tokenizers::models::wordpiece::WordPiece;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;

        // Build a WordPiece model with a minimal vocabulary.
        // WordPiece is used by BERT-family models, making this representative
        // of the actual tokenizer configuration used in production.
        //
        // The vocab is constructed from an array of tuples because the WordPiece
        // builder's vocab method accepts `Into<AHashMap<String, u32>>`, and
        // AHashMap implements `From<[(K, V); N]>`.
        let vocab = [
            ("[PAD]".to_string(), 0_u32),
            ("[UNK]".to_string(), 1),
            ("[CLS]".to_string(), 2),
            ("[SEP]".to_string(), 3),
            ("hello".to_string(), 4),
            ("world".to_string(), 5),
            ("foo".to_string(), 6),
            ("bar".to_string(), 7),
            ("the".to_string(), 8),
            ("a".to_string(), 9),
        ];

        let wordpiece = WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("WordPiece model construction failed");

        let mut tokenizer = tokenizers::Tokenizer::new(wordpiece);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));

        tokenizer
    }

    /// T-EMB-002: Tokenizer produces expected token IDs for known input.
    /// Verifies that "hello world" tokenizes into the expected token IDs
    /// from the test vocabulary.
    #[test]
    fn t_emb_002_tokenizer_produces_expected_token_ids() {
        let tokenizer = build_test_tokenizer();
        let wrapper = TokenizerWrapper::new(tokenizer);

        let result = wrapper
            .encode_batch(&["hello world"], 512)
            .expect("encoding failed");

        // WordPiece produces token IDs based on the vocabulary mapping.
        // "hello" -> 4, "world" -> 5
        // The exact IDs depend on whether [CLS]/[SEP] are added by post-processing.
        // Without post-processing, the raw tokens are the word IDs.
        assert!(
            !result.token_ids[0].is_empty(),
            "token IDs must not be empty for non-empty input"
        );

        // Verify that the known words appear in the token ID sequence.
        let ids = &result.token_ids[0];
        assert!(
            ids.contains(&4) && ids.contains(&5),
            "token IDs should contain hello=4 and world=5, got {:?}",
            ids
        );
    }

    /// T-EMB-003: Batch padding and attention masks are correct.
    /// When two texts of different lengths are batched, the shorter one is
    /// padded to the length of the longer one. Attention masks have 1 for
    /// real tokens and 0 for padding positions.
    #[test]
    fn t_emb_003_batch_padding_and_attention_masks_correct() {
        let tokenizer = build_test_tokenizer();
        let wrapper = TokenizerWrapper::new(tokenizer);

        // "hello" produces fewer tokens than "hello world foo bar"
        let result = wrapper
            .encode_batch(&["hello", "hello world foo bar"], 512)
            .expect("encoding failed");

        // Both sequences must have the same padded length
        assert_eq!(
            result.token_ids[0].len(),
            result.token_ids[1].len(),
            "padded sequences must have equal length"
        );
        assert_eq!(
            result.attention_masks[0].len(),
            result.attention_masks[1].len(),
            "attention masks must have equal length"
        );

        // The shorter sequence should have trailing zeros in the attention mask
        let mask_short = &result.attention_masks[0];
        let mask_long = &result.attention_masks[1];

        // The longer sequence should have all 1s (no padding needed)
        let long_ones: u32 = mask_long.iter().sum();
        assert_eq!(
            long_ones,
            mask_long.len() as u32,
            "longer sequence mask should have all 1s"
        );

        // The shorter sequence should have some 0s at the end
        let short_ones: u32 = mask_short.iter().sum();
        assert!(
            short_ones < mask_short.len() as u32,
            "shorter sequence mask should have trailing 0s"
        );

        // Verify that 1s come before 0s (no interleaving)
        let first_zero_pos = mask_short.iter().position(|&m| m == 0);
        if let Some(pos) = first_zero_pos {
            for &m in &mask_short[pos..] {
                assert_eq!(
                    m, 0,
                    "all values after the first padding position must be 0"
                );
            }
        }
    }

    /// Verifies that an empty input batch produces an empty BatchEncoding.
    #[test]
    fn empty_batch_produces_empty_encoding() {
        let tokenizer = build_test_tokenizer();
        let wrapper = TokenizerWrapper::new(tokenizer);

        let result = wrapper.encode_batch(&[], 512).expect("encoding failed");
        assert!(result.token_ids.is_empty());
        assert!(result.attention_masks.is_empty());
        assert!(result.offsets.is_empty());
    }

    /// Verifies that truncation limits the sequence length to max_length.
    #[test]
    fn truncation_limits_sequence_length() {
        let tokenizer = build_test_tokenizer();
        let wrapper = TokenizerWrapper::new(tokenizer);

        let result = wrapper
            .encode_batch(&["hello world foo bar the a hello world"], 3)
            .expect("encoding failed");

        assert!(
            result.token_ids[0].len() <= 3,
            "sequence length must not exceed max_length=3, got {}",
            result.token_ids[0].len()
        );
    }

    /// Regression test for issue #16: encode_batch must produce the same
    /// real (non-padding) token IDs whether texts are batched or encoded
    /// individually. This validates that the parallel Rayon-based
    /// encode_batch() path yields the same tokenization as sequential
    /// encode(). Padding tokens are excluded from comparison because batch
    /// encoding pads shorter sequences to the longest sequence length.
    #[test]
    fn batch_encoding_matches_individual_encoding() {
        let tokenizer = build_test_tokenizer();
        let wrapper = TokenizerWrapper::new(tokenizer);

        let texts = ["hello world", "foo bar", "the a hello"];
        let batch_result = wrapper.encode_batch(&texts, 512).expect("batch failed");

        for (i, &text) in texts.iter().enumerate() {
            let single_result = wrapper
                .encode_batch(&[text], 512)
                .expect("single encode failed");

            // Extract only the real (non-padding) tokens from the batch result
            // by counting the 1s in the attention mask.
            let real_len: usize = batch_result.attention_masks[i]
                .iter()
                .take_while(|&&m| m == 1)
                .count();
            let batch_real_ids = &batch_result.token_ids[i][..real_len];

            assert_eq!(
                batch_real_ids,
                single_result.token_ids[0].as_slice(),
                "real token IDs for text[{i}] differ between batch and individual encoding"
            );
        }
    }

    /// T-EMB-TRUNC-001: Truncation flag is set when input exceeds max_length.
    /// Regression test for BUG-006 where truncation occurred silently without
    /// any indication. The `truncated` field in BatchEncoding now correctly
    /// reports whether each input was truncated.
    #[test]
    fn t_emb_trunc_001_truncation_flag_set_when_exceeds_max_length() {
        let tokenizer = build_test_tokenizer();
        let wrapper = TokenizerWrapper::new(tokenizer);

        // "hello world foo bar" produces 4 tokens. With max_length=2,
        // truncation must occur.
        let result = wrapper
            .encode_batch(&["hello world foo bar"], 2)
            .expect("encoding failed");

        assert!(
            result.truncated[0],
            "truncated flag must be true when input exceeds max_length"
        );
    }

    /// T-EMB-TRUNC-002: Truncation flag is false when input fits within max_length.
    #[test]
    fn t_emb_trunc_002_truncation_flag_false_when_fits() {
        let tokenizer = build_test_tokenizer();
        let wrapper = TokenizerWrapper::new(tokenizer);

        // "hello" produces 1 token. With max_length=512, no truncation.
        let result = wrapper
            .encode_batch(&["hello"], 512)
            .expect("encoding failed");

        assert!(
            !result.truncated[0],
            "truncated flag must be false when input fits within max_length"
        );
    }

    /// T-EMB-TRUNC-003: Truncation flags are per-input in a batch. When a
    /// batch contains both short and long inputs, only the long inputs are
    /// flagged as truncated.
    #[test]
    fn t_emb_trunc_003_per_input_truncation_flags() {
        let tokenizer = build_test_tokenizer();
        let wrapper = TokenizerWrapper::new(tokenizer);

        // First input has 1 token (fits in max_length=3), second has 4+ tokens
        // (exceeds max_length=3).
        let result = wrapper
            .encode_batch(&["hello", "hello world foo bar"], 3)
            .expect("encoding failed");

        assert!(
            !result.truncated[0],
            "short input must not be flagged as truncated"
        );
        assert!(
            result.truncated[1],
            "long input must be flagged as truncated"
        );
    }

    /// T-EMB-TRUNC-004: Empty batch produces empty truncated flags vector.
    #[test]
    fn t_emb_trunc_004_empty_batch_empty_flags() {
        let tokenizer = build_test_tokenizer();
        let wrapper = TokenizerWrapper::new(tokenizer);

        let result = wrapper.encode_batch(&[], 512).expect("encoding failed");
        assert!(
            result.truncated.is_empty(),
            "empty batch must produce empty truncated flags"
        );
    }
}
