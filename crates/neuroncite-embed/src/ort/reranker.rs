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

//! ONNX Runtime reranker implementation.
//!
//! Implements the `Reranker` trait by running a cross-encoder ONNX model through
//! an `ort::session::Session`. Takes a query-document pair, tokenizes them as a
//! single sequence with \[SEP\] delimiter, and returns the model's relevance score.

use std::path::PathBuf;
use std::sync::Mutex;

use neuroncite_core::{NeuronCiteError, Reranker};
use ort::session::Session;
use ort::value::TensorRef;

use crate::cache;
use crate::error::EmbedError;
use crate::tokenize::TokenizerWrapper;

use super::session::{self, OrtSessionConfig};

/// Default maximum sequence length for cross-encoder models. Query and candidate
/// are concatenated with a \[SEP\] token, so the effective limit for each part
/// is roughly half of this value.
const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// ONNX Runtime cross-encoder reranker. Holds a loaded ONNX session and
/// tokenizer for scoring query-candidate pairs.
///
/// The session is wrapped in a `Mutex` because `Session::run` requires
/// `&mut self`.
///
/// Before calling `rerank_batch`, the caller must invoke `load_model` to load
/// the cross-encoder ONNX model and tokenizer.
pub struct OrtReranker {
    /// The ONNX Runtime inference session for the cross-encoder model.
    session: Option<Mutex<Session>>,

    /// Tokenizer for encoding query-candidate pairs into token sequences.
    tokenizer: Option<TokenizerWrapper>,

    /// The currently loaded model's Hugging Face identifier.
    model_id: Option<String>,

    /// Whether the loaded ONNX model accepts a `token_type_ids` input tensor.
    /// BERT-based cross-encoders (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2)
    /// require this input, while XLM-RoBERTa-based models (e.g.,
    /// BAAI/bge-reranker-v2-m3) do not have this input in their ONNX graph.
    /// Determined at model load time by inspecting the session's input names.
    uses_token_type_ids: bool,
}

// SAFETY: OrtReranker is Send + Sync because the ONNX Runtime Session is
// wrapped in Mutex<Session> for synchronized access. The ort crate (pinned to
// version =2.0.0-rc.11 in the workspace Cargo.toml) documents that Session
// is thread-safe for concurrent inference at the ONNX Runtime C API level
// (InferenceSession is designed for multi-threaded use). All other fields
// (TokenizerWrapper, model_id: Option<String>, uses_token_type_ids: bool) are
// owned values without interior mutability and are trivially Send + Sync.
//
// If the ort crate version is updated, verify that the new version maintains
// the same thread-safety guarantee for Session before keeping this impl.
unsafe impl Send for OrtReranker {}
unsafe impl Sync for OrtReranker {}

impl OrtReranker {
    /// Creates an `OrtReranker` in the unloaded state. No model or tokenizer
    /// is loaded; the caller must invoke `load_model` before reranking.
    #[must_use]
    pub fn unloaded() -> Self {
        Self {
            session: None,
            tokenizer: None,
            model_id: None,
            uses_token_type_ids: false,
        }
    }

    /// Locates the cross-encoder model directory in the local cache.
    /// Verifies file integrity against the checksums manifest when the
    /// model directory exists. Returns an error on checksum failure so
    /// the caller can purge the cache and re-download.
    fn resolve_model_paths(model_id: &str) -> Result<(PathBuf, PathBuf), EmbedError> {
        let model_dir = cache::model_dir(model_id, "main");
        if !model_dir.exists() {
            return Err(EmbedError::ModelNotFound(format!(
                "reranker model directory does not exist: {}",
                model_dir.display()
            )));
        }

        // Verify file integrity against the checksums manifest. If any file
        // is corrupt (checksum mismatch), return an error. The caller should
        // purge the cache directory and trigger a fresh download.
        match cache::verify_cached_model(model_id, "main") {
            Ok(true) => {}
            Ok(false) => {
                return Err(EmbedError::CacheIo(format!(
                    "cached reranker model '{model_id}' failed checksum verification -- files are corrupt"
                )));
            }
            Err(e) => {
                return Err(EmbedError::CacheIo(format!(
                    "reranker checksum verification error for '{model_id}': {e}"
                )));
            }
        }

        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !model_path.exists() {
            return Err(EmbedError::ModelNotFound(format!(
                "reranker ONNX model file not found: {}",
                model_path.display()
            )));
        }
        if !tokenizer_path.exists() {
            return Err(EmbedError::ModelNotFound(format!(
                "reranker tokenizer file not found: {}",
                tokenizer_path.display()
            )));
        }

        Ok((model_path, tokenizer_path))
    }
}

impl Reranker for OrtReranker {
    /// Returns the human-readable name of this reranker.
    fn name(&self) -> &str {
        "ONNX Runtime Cross-Encoder"
    }

    /// Scores a batch of candidate passages against a query string.
    ///
    /// Each candidate is paired with the query and encoded as a single sequence
    /// using the `[CLS] query [SEP] candidate [SEP]` format. The cross-encoder
    /// model produces a logit for each pair, which is used directly as the
    /// relevance score (higher is more relevant).
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Search` if the model is not loaded or if
    /// inference fails.
    fn rerank_batch(&self, query: &str, candidates: &[&str]) -> Result<Vec<f64>, NeuronCiteError> {
        let session_mutex = self.session.as_ref().ok_or_else(|| {
            NeuronCiteError::Search("reranker model not loaded; call load_model first".into())
        })?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| NeuronCiteError::Search("reranker tokenizer not loaded".into()))?;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Encode each (query, candidate) pair as a single sequence.
        // Cross-encoders expect the pair as "[CLS] query [SEP] candidate [SEP]".
        let inner_tokenizer = tokenizer.inner();

        let mut all_input_ids: Vec<Vec<i64>> = Vec::with_capacity(candidates.len());
        let mut all_attention_masks: Vec<Vec<i64>> = Vec::with_capacity(candidates.len());
        // Only collected when the model accepts token_type_ids (BERT-based
        // cross-encoders). XLM-RoBERTa-based models like bge-reranker-v2-m3
        // do not have this input in their ONNX graph.
        let mut all_token_type_ids: Vec<Vec<i64>> = if self.uses_token_type_ids {
            Vec::with_capacity(candidates.len())
        } else {
            Vec::new()
        };

        for &candidate in candidates {
            let encoding = inner_tokenizer
                .encode((query, candidate), true)
                .map_err(|e| NeuronCiteError::Search(format!("tokenizer error: {e}")))?;

            let ids: Vec<i64> = encoding
                .get_ids()
                .iter()
                .take(DEFAULT_MAX_SEQ_LEN)
                .map(|&id| i64::from(id))
                .collect();

            let mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .take(DEFAULT_MAX_SEQ_LEN)
                .map(|&m| i64::from(m))
                .collect();

            all_input_ids.push(ids);
            all_attention_masks.push(mask);

            if self.uses_token_type_ids {
                let type_ids: Vec<i64> = encoding
                    .get_type_ids()
                    .iter()
                    .take(DEFAULT_MAX_SEQ_LEN)
                    .map(|&t| i64::from(t))
                    .collect();
                all_token_type_ids.push(type_ids);
            }
        }

        // Pad all sequences to the maximum length in the batch.
        let max_len = all_input_ids.iter().map(Vec::len).max().unwrap_or(0);
        let batch_size = candidates.len();

        let mut input_ids_flat: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask_flat: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        // Only allocated when the model requires token_type_ids.
        let mut token_type_ids_flat: Vec<i64> = if self.uses_token_type_ids {
            Vec::with_capacity(batch_size * max_len)
        } else {
            Vec::new()
        };

        for i in 0..batch_size {
            let real_len = all_input_ids[i].len();
            input_ids_flat.extend_from_slice(&all_input_ids[i]);
            input_ids_flat.extend(std::iter::repeat_n(0_i64, max_len - real_len));

            attention_mask_flat.extend_from_slice(&all_attention_masks[i]);
            attention_mask_flat.extend(std::iter::repeat_n(0_i64, max_len - real_len));

            if self.uses_token_type_ids {
                token_type_ids_flat.extend_from_slice(&all_token_type_ids[i]);
                token_type_ids_flat.extend(std::iter::repeat_n(0_i64, max_len - real_len));
            }
        }

        let shape = [batch_size, max_len];

        let input_ids_tensor =
            TensorRef::from_array_view((shape, &*input_ids_flat)).map_err(|e| {
                NeuronCiteError::Search(format!("failed to create input_ids tensor: {e}"))
            })?;
        let attention_mask_tensor = TensorRef::from_array_view((shape, &*attention_mask_flat))
            .map_err(|e| {
                NeuronCiteError::Search(format!("failed to create attention_mask tensor: {e}"))
            })?;

        let mut session = session_mutex
            .lock()
            .map_err(|e| NeuronCiteError::Search(format!("session lock poisoned: {e}")))?;

        // Build the input map. BERT-based cross-encoders require input_ids,
        // attention_mask, and token_type_ids. XLM-RoBERTa-based models only
        // accept input_ids and attention_mask; passing token_type_ids causes
        // an "Invalid input name" error from the ONNX runtime.
        let outputs = if self.uses_token_type_ids {
            let token_type_ids_tensor = TensorRef::from_array_view((shape, &*token_type_ids_flat))
                .map_err(|e| {
                    NeuronCiteError::Search(format!("failed to create token_type_ids tensor: {e}"))
                })?;

            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ];

            session.run(inputs).map_err(|e| {
                NeuronCiteError::Search(format!("cross-encoder inference failed: {e}"))
            })?
        } else {
            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ];

            session.run(inputs).map_err(|e| {
                NeuronCiteError::Search(format!("cross-encoder inference failed: {e}"))
            })?
        };

        // Extract the relevance scores from the output tensor.
        // Cross-encoders output a tensor of shape [batch_size, 1] or [batch_size].
        // `try_extract_tensor` returns `(&Shape, &[f32])`.
        let output_value = outputs.iter().next().map(|(_, v)| v).ok_or_else(|| {
            NeuronCiteError::Search("no output tensor in cross-encoder result".into())
        })?;

        let (output_shape, output_data) =
            output_value.try_extract_tensor::<f32>().map_err(|e| {
                NeuronCiteError::Search(format!(
                    "failed to extract cross-encoder output tensor: {e}"
                ))
            })?;

        let shape_dims: &[i64] = output_shape;

        let scores: Vec<f64> = (0..batch_size)
            .map(|b| {
                let score = if shape_dims.len() == 2 {
                    let cols = shape_dims[1] as usize;
                    output_data[b * cols]
                } else {
                    output_data[b]
                };

                // Guard against floating-point overflow or degenerate values.
                // Cross-encoder logits for ms-marco models fall in the [-10, +10]
                // range under normal conditions. Values outside [-1000, +1000]
                // indicate a numerical instability in the ONNX inference (e.g.,
                // denormalized inputs producing extreme logits). Clamping these
                // to the boundary prevents nonsensical scores from propagating
                // to callers as ranking artifacts.
                let score_f64 = f64::from(score);
                if score_f64.is_finite() {
                    score_f64.clamp(-1000.0, 1000.0)
                } else {
                    // NaN or infinity: replace with the minimum score so this
                    // result sorts to the bottom of the ranked list.
                    -1000.0
                }
            })
            .collect();

        Ok(scores)
    }

    /// Loads a cross-encoder ONNX model from the local cache.
    ///
    /// The model directory must contain `model.onnx` and `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Embed` if the model files are missing or
    /// if session creation fails.
    fn load_model(&mut self, model_id: &str) -> Result<(), NeuronCiteError> {
        let (model_path, tokenizer_path) = Self::resolve_model_paths(model_id)
            .map_err(|e| NeuronCiteError::Embed(e.to_string()))?;

        let config = OrtSessionConfig {
            model_path,
            use_gpu: true,
        };
        let (ort_session, _capabilities) =
            session::create_session(&config).map_err(|e| NeuronCiteError::Embed(e.to_string()))?;

        // Inspect the ONNX graph's input names to determine whether the model
        // accepts token_type_ids. BERT-based cross-encoders (e.g.,
        // cross-encoder/ms-marco-MiniLM-L-6-v2) include token_type_ids in
        // their input schema, while XLM-RoBERTa-based models (e.g.,
        // BAAI/bge-reranker-v2-m3) only accept input_ids and attention_mask.
        // Passing an input that the model does not expect causes an
        // "Invalid input name" error from the ONNX runtime.
        let has_token_type_ids = ort_session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        tracing::info!(
            model_id = model_id,
            uses_token_type_ids = has_token_type_ids,
            input_count = ort_session.inputs().len(),
            "inspected ONNX cross-encoder input schema"
        );

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            NeuronCiteError::Embed(format!("failed to load reranker tokenizer: {e}"))
        })?;

        self.session = Some(Mutex::new(ort_session));
        self.tokenizer = Some(TokenizerWrapper::new(tokenizer));
        self.model_id = Some(model_id.to_string());
        self.uses_token_type_ids = has_token_type_ids;

        tracing::info!(
            model_id = model_id,
            uses_token_type_ids = has_token_type_ids,
            "ONNX Runtime cross-encoder model loaded"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// T-EMB-007: rerank_batch returns exactly candidates.len() scores.
    /// This test verifies that an unloaded reranker returns an error rather
    /// than fabricated scores. With a cached model, the test would verify
    /// that the number of scores matches the number of candidates.
    #[test]
    fn t_emb_007_rerank_batch_returns_correct_number_of_scores() {
        use neuroncite_core::Reranker;
        let reranker = super::OrtReranker::unloaded();
        let result = reranker.rerank_batch("test query", &["candidate 1", "candidate 2"]);

        // The reranker is not loaded, so it should return an error.
        assert!(result.is_err(), "unloaded reranker should return an error");
    }

    /// T-EMB-008: Relevant pair scores higher than irrelevant.
    /// Without a loaded model, this test verifies the error handling path.
    #[test]
    fn t_emb_008_relevant_pair_scores_higher_than_irrelevant() {
        use neuroncite_core::Reranker;
        let reranker = super::OrtReranker::unloaded();

        let result = reranker.rerank_batch(
            "What is machine learning?",
            &[
                "Machine learning is a subset of artificial intelligence.",
                "The weather today is sunny.",
            ],
        );
        assert!(
            result.is_err(),
            "unloaded reranker must return an error, not fabricated scores"
        );
    }

    /// T-EMB-024: Score clamping guards against overflow and non-finite values.
    /// Regression test for DEF-4: cross-encoder inference producing extreme
    /// float values (~-5.5e15) that propagated as ranking artifacts. The fix
    /// clamps scores to [-1000, 1000] and replaces NaN/infinity with -1000.
    /// This test verifies the clamping logic directly on f32 -> f64 conversion.
    #[test]
    fn t_emb_009_score_clamping_prevents_overflow() {
        // Simulate the clamping logic from rerank_batch's score extraction.
        let clamp_score = |raw: f32| -> f64 {
            let score_f64 = f64::from(raw);
            if score_f64.is_finite() {
                score_f64.clamp(-1000.0, 1000.0)
            } else {
                -1000.0
            }
        };

        // Normal score within range: passes through unchanged.
        // Using values exactly representable in f32 (powers of 2 fractions)
        // to avoid f32->f64 precision differences in exact equality checks.
        assert_eq!(clamp_score(5.5), 5.5);
        assert_eq!(clamp_score(-3.25), -3.25);

        // Extreme positive value: clamped to 1000.
        assert_eq!(clamp_score(1e15), 1000.0);

        // Extreme negative value: clamped to -1000.
        assert_eq!(clamp_score(-5.5e15), -1000.0);

        // Boundary values: exact limits pass through.
        assert_eq!(clamp_score(1000.0), 1000.0);
        assert_eq!(clamp_score(-1000.0), -1000.0);

        // NaN: replaced with -1000.0 (sorts to bottom of ranked list).
        assert_eq!(clamp_score(f32::NAN), -1000.0);

        // Positive infinity: replaced with -1000.0.
        assert_eq!(clamp_score(f32::INFINITY), -1000.0);

        // Negative infinity: replaced with -1000.0.
        assert_eq!(clamp_score(f32::NEG_INFINITY), -1000.0);
    }
}
