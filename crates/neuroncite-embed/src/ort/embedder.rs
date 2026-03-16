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

//! ONNX Runtime embedder implementation.
//!
//! Implements the `EmbeddingBackend` trait by running transformer ONNX models through
//! an `ort::session::Session`. Supports both encoder-based models (BERT, BGE, MiniLM,
//! GTE-Large) and decoder-based models (Qwen3-Embedding) via configurable pooling
//! strategies, input tensor layouts, and optional position_ids defined in
//! `EmbeddingModelConfig`.

use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

use half::f16;
use neuroncite_core::{
    EmbeddingBackend, EmbeddingModelConfig, ModelInfo, NeuronCiteError, PoolingStrategy,
};
use ort::session::Session;
use ort::value::TensorRef;

use crate::cache;
use crate::error::EmbedError;
use crate::normalize::l2_normalize;
use crate::tokenize::TokenizerWrapper;

use super::session::{self, OrtSessionConfig};

/// Lazily-initialized static catalog of all embedding models supported by the
/// ONNX Runtime backend. The `LazyLock` ensures the `Vec<EmbeddingModelConfig>`
/// is constructed exactly once on first access and lives for the entire program
/// lifetime. All subsequent calls to `supported_model_configs()` return a
/// `&'static [EmbeddingModelConfig]` slice into this vector, avoiding repeated
/// heap allocations of 8 String-heavy structs per call.
static MODEL_CATALOG: LazyLock<Vec<EmbeddingModelConfig>> = LazyLock::new(|| {
    vec![
        // ---------------------------------------------------------------
        // Encoder models (BERT architecture, CLS pooling)
        // These models have absolute position embeddings baked in and do
        // not need explicit position_ids or instruction prefixes.
        // ---------------------------------------------------------------
        EmbeddingModelConfig {
            model_id: "BAAI/bge-small-en-v1.5".into(),
            display_name: "BGE Small EN v1.5".into(),
            vector_dimension: 384,
            max_seq_len: 512,
            pooling: PoolingStrategy::Cls,
            uses_token_type_ids: true,
            needs_position_ids: false,
            query_prefix: String::new(),
            document_prefix: String::new(),
            release_year: 2023,
            quality_rating: "Good".into(),
            language_scope: "EN".into(),
            de_en_retrieval: "Not supported".into(),
            cpu_suitability: "Very good".into(),
            gpu_recommendation: "Optional".into(),
            ram_requirement: "1-2 GB".into(),
            typical_use_case: "Fast EN search".into(),
            model_size_mb: 130,
        },
        EmbeddingModelConfig {
            model_id: "BAAI/bge-base-en-v1.5".into(),
            display_name: "BGE Base EN v1.5".into(),
            vector_dimension: 768,
            max_seq_len: 512,
            pooling: PoolingStrategy::Cls,
            uses_token_type_ids: true,
            needs_position_ids: false,
            query_prefix: String::new(),
            document_prefix: String::new(),
            release_year: 2023,
            quality_rating: "Good+".into(),
            language_scope: "EN".into(),
            de_en_retrieval: "Not supported".into(),
            cpu_suitability: "Good".into(),
            gpu_recommendation: "Optional".into(),
            ram_requirement: "2-4 GB".into(),
            typical_use_case: "Precise EN search".into(),
            model_size_mb: 440,
        },
        EmbeddingModelConfig {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
            display_name: "MiniLM L6 v2".into(),
            vector_dimension: 384,
            max_seq_len: 512,
            pooling: PoolingStrategy::Cls,
            uses_token_type_ids: true,
            needs_position_ids: false,
            query_prefix: String::new(),
            document_prefix: String::new(),
            release_year: 2021,
            quality_rating: "Solid".into(),
            language_scope: "EN".into(),
            de_en_retrieval: "Not supported".into(),
            cpu_suitability: "Very good".into(),
            gpu_recommendation: "Not required".into(),
            ram_requirement: "<1.5 GB".into(),
            typical_use_case: "Lightweight systems".into(),
            model_size_mb: 90,
        },
        // ---------------------------------------------------------------
        // GTE Large EN v1.5 (Alibaba-NLP, BERT++ with RoPE, CLS pooling)
        // Uses a custom "NewModel" architecture (BERT + RoPE + GLU) but
        // follows the same input contract as standard BERT models: requires
        // token_type_ids and does not need explicit position_ids (RoPE is
        // applied internally from the model's config). The official ONNX
        // export is hosted in the onnx/ directory of the HuggingFace repo.
        // ---------------------------------------------------------------
        EmbeddingModelConfig {
            model_id: "Alibaba-NLP/gte-large-en-v1.5".into(),
            display_name: "GTE Large EN v1.5".into(),
            vector_dimension: 1024,
            max_seq_len: 8192,
            pooling: PoolingStrategy::Cls,
            uses_token_type_ids: true,
            needs_position_ids: false,
            query_prefix: String::new(),
            document_prefix: String::new(),
            release_year: 2023,
            quality_rating: "High".into(),
            language_scope: "EN".into(),
            de_en_retrieval: "Not supported".into(),
            cpu_suitability: "Slow".into(),
            gpu_recommendation: "Recommended".into(),
            ram_requirement: "6-10 GB".into(),
            typical_use_case: "Long EN documents".into(),
            model_size_mb: 1340,
        },
        // ---------------------------------------------------------------
        // Decoder models (Qwen3 LLM-based, mean pooling, instruction prefix)
        // The ONNX exports require explicit position_ids as input because
        // Qwen3 uses rotary position embeddings (RoPE) that are computed
        // from the position_ids tensor at runtime.
        // ---------------------------------------------------------------
        EmbeddingModelConfig {
            model_id: "Qwen/Qwen3-Embedding-0.6B".into(),
            display_name: "Qwen3 Embedding 0.6B".into(),
            vector_dimension: 1024,
            max_seq_len: 8192,
            pooling: PoolingStrategy::MeanPooling,
            uses_token_type_ids: false,
            needs_position_ids: true,
            query_prefix: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ".into(),
            document_prefix: String::new(),
            release_year: 2025,
            quality_rating: "Very good".into(),
            language_scope: "Multilingual".into(),
            de_en_retrieval: "Good".into(),
            cpu_suitability: "Heavy".into(),
            gpu_recommendation: "Recommended".into(),
            ram_requirement: "8-12 GB".into(),
            typical_use_case: "RAG systems".into(),
            model_size_mb: 1200,
        },
        EmbeddingModelConfig {
            model_id: "Qwen/Qwen3-Embedding-4B".into(),
            display_name: "Qwen3 Embedding 4B".into(),
            vector_dimension: 2560,
            max_seq_len: 8192,
            pooling: PoolingStrategy::MeanPooling,
            uses_token_type_ids: false,
            needs_position_ids: true,
            query_prefix: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ".into(),
            document_prefix: String::new(),
            release_year: 2025,
            quality_rating: "Very high".into(),
            language_scope: "Multilingual".into(),
            de_en_retrieval: "Very good".into(),
            cpu_suitability: "Not recommended".into(),
            gpu_recommendation: "Strongly recommended".into(),
            ram_requirement: "16-24 GB".into(),
            typical_use_case: "High-quality retrieval".into(),
            model_size_mb: 8000,
        },
        EmbeddingModelConfig {
            model_id: "Qwen/Qwen3-Embedding-8B".into(),
            display_name: "Qwen3 Embedding 8B".into(),
            vector_dimension: 4096,
            max_seq_len: 8192,
            pooling: PoolingStrategy::MeanPooling,
            uses_token_type_ids: false,
            needs_position_ids: true,
            query_prefix: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ".into(),
            document_prefix: String::new(),
            release_year: 2025,
            quality_rating: "SOTA Open-Source".into(),
            language_scope: "Multilingual".into(),
            de_en_retrieval: "SOTA".into(),
            cpu_suitability: "Not recommended".into(),
            gpu_recommendation: "Strongly recommended".into(),
            ram_requirement: "32-40 GB".into(),
            typical_use_case: "Enterprise / High Precision".into(),
            model_size_mb: 16500,
        },
        // ---------------------------------------------------------------
        // BGE-M3 (BAAI, XLM-RoBERTa encoder, CLS pooling, multilingual)
        // XLM-RoBERTa architecture does not use token_type_ids (unlike
        // BERT-based BGE models). The ONNX model accepts input_ids and
        // attention_mask only. The official HuggingFace repository ships
        // the ONNX export in the onnx/ subdirectory. BGE-M3 supports
        // dense, sparse, and ColBERT retrieval; this config uses the
        // dense CLS embedding.
        // ---------------------------------------------------------------
        EmbeddingModelConfig {
            model_id: "BAAI/bge-m3".into(),
            display_name: "BGE-M3".into(),
            vector_dimension: 1024,
            max_seq_len: 8192,
            pooling: PoolingStrategy::Cls,
            uses_token_type_ids: false,
            needs_position_ids: false,
            query_prefix: String::new(),
            document_prefix: String::new(),
            release_year: 2024,
            quality_rating: "High".into(),
            language_scope: "Multilingual".into(),
            de_en_retrieval: "Very good".into(),
            cpu_suitability: "Slow".into(),
            gpu_recommendation: "Recommended".into(),
            ram_requirement: "8-16 GB".into(),
            typical_use_case: "Multilingual Retrieval".into(),
            model_size_mb: 2270,
        },
    ]
});

/// Returns a static slice referencing the lazily-initialized model catalog.
/// The catalog contains all embedding models supported by the ONNX Runtime
/// backend. Each entry is an `EmbeddingModelConfig` describing the model's
/// architecture, pooling strategy, sequence length, and optional instruction
/// prefixes.
///
/// Because the catalog is stored in a `LazyLock<Vec<...>>`, the first call
/// initializes the vector and all subsequent calls return the same `&'static`
/// slice without allocation.
///
/// Encoder models (BERT-family) use CLS pooling, token_type_ids, and have
/// absolute position embeddings baked into the model weights.
/// Decoder models (Qwen-family) use mean pooling, lack token_type_ids,
/// require explicit position_ids for rotary position embeddings, and
/// use instruction prefixes for queries.
#[must_use]
pub fn supported_model_configs() -> &'static [EmbeddingModelConfig] {
    &MODEL_CATALOG
}

/// Looks up the `EmbeddingModelConfig` for a given model ID from the static
/// catalog. Returns a `&'static` reference to the config entry, or `None` if
/// the model ID is not registered. Because the catalog lives in a `LazyLock`,
/// the returned reference is valid for the entire program lifetime.
#[must_use]
pub fn find_model_config(model_id: &str) -> Option<&'static EmbeddingModelConfig> {
    supported_model_configs()
        .iter()
        .find(|cfg| cfg.model_id == model_id)
}

/// ONNX Runtime embedding backend. Holds a loaded ONNX session and the
/// corresponding tokenizer for encoding input text into token IDs.
///
/// The session is wrapped in a `Mutex` because `ort::session::Session::run`
/// requires `&mut self`. The Mutex allows shared access from multiple threads
/// while serializing actual inference calls.
///
/// Before calling `embed_batch` or `embed_single`, the caller must invoke
/// `load_model` to load the ONNX model file and tokenizer configuration.
pub struct OrtEmbedder {
    /// The ONNX Runtime inference session, populated after `load_model` succeeds.
    /// Wrapped in Mutex because `Session::run` requires `&mut self`.
    session: Option<Mutex<Session>>,

    /// Tokenizer wrapper for encoding input text into token IDs and attention masks.
    tokenizer: Option<TokenizerWrapper>,

    /// Dimensionality of the embedding vectors produced by the loaded model.
    dimension: usize,

    /// The currently loaded model's Hugging Face identifier.
    model_id: Option<String>,

    /// Model-specific configuration controlling pooling, sequence length, and
    /// input tensor layout. Populated during `load_model` from the static catalog.
    model_config: Option<EmbeddingModelConfig>,

    /// Runtime-detected hardware capabilities for the active ONNX session.
    /// Populated during `load_model` from `create_session`'s hardware detection.
    /// Used by the pipeline to adapt batch sizes and threading to the platform.
    capabilities: neuroncite_core::InferenceCapabilities,
}

// SAFETY: OrtEmbedder is Send + Sync because the ONNX Runtime Session is
// wrapped in Mutex<Session> for synchronized access. The ort crate (pinned to
// version =2.0.0-rc.11 in the workspace Cargo.toml) documents that Session
// is thread-safe for concurrent inference calls at the ONNX Runtime C API
// level (InferenceSession is designed for multi-threaded use). The Mutex
// serializes mutable access from this crate's perspective. All other fields
// (TokenizerWrapper, model_id: Option<String>, dimension: usize,
// model_config: Option<...>) are owned values without interior mutability
// and are trivially Send + Sync.
//
// If the ort crate version is updated, verify that the new version maintains
// the same thread-safety guarantee for Session before keeping this impl.
unsafe impl Send for OrtEmbedder {}
unsafe impl Sync for OrtEmbedder {}

impl OrtEmbedder {
    /// Creates an `OrtEmbedder` in the unloaded state. No model or tokenizer
    /// is loaded; the caller must invoke `load_model` before inference.
    #[must_use]
    pub fn unloaded() -> Self {
        Self {
            session: None,
            tokenizer: None,
            dimension: 384, // Default dimension for BGE-small
            model_id: None,
            model_config: None,
            capabilities: neuroncite_core::InferenceCapabilities::default(),
        }
    }

    /// Locates the model directory in the local cache and returns the paths
    /// to the ONNX model file and tokenizer JSON file. If the model is not
    /// cached locally, downloads it from HuggingFace before resolving paths.
    fn resolve_model_paths(model_id: &str) -> Result<(PathBuf, PathBuf), EmbedError> {
        let model_dir = cache::model_dir(model_id, "main");
        if !model_dir.exists() {
            tracing::info!(
                model_id,
                "model not cached locally, downloading from HuggingFace"
            );
            eprintln!(
                "  Downloading embedding model '{model_id}' from HuggingFace (one-time download, please wait)..."
            );
            cache::download_model(model_id, "main")?;
            eprintln!("  Model download complete.");
        } else {
            // Verify file integrity against the checksums manifest written
            // during the original download. Detects partial downloads or
            // on-disk corruption before the ONNX runtime attempts to parse
            // the model file (which produces opaque errors on corrupt data).
            match cache::verify_cached_model(model_id, "main") {
                Ok(true) => {}
                Ok(false) => {
                    tracing::warn!(
                        model_id,
                        "cached model failed checksum verification; re-downloading"
                    );
                    cache::download_model(model_id, "main")?;
                }
                Err(e) => {
                    tracing::warn!(
                        model_id,
                        "checksum verification error: {e}; proceeding with cached files"
                    );
                }
            }
        }

        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !model_path.exists() {
            return Err(EmbedError::ModelNotFound(format!(
                "ONNX model file not found: {}",
                model_path.display()
            )));
        }
        if !tokenizer_path.exists() {
            return Err(EmbedError::ModelNotFound(format!(
                "tokenizer file not found: {}",
                tokenizer_path.display()
            )));
        }

        Ok((model_path, tokenizer_path))
    }

    /// Computes mean pooling over the transformer output tensor, weighted by the
    /// attention mask. Padding positions (mask == 0) are excluded from the average.
    ///
    /// For a 3-D output of shape [batch_size, seq_len, hidden_dim], the result is:
    ///   embedding[d] = sum(output[t][d] * mask[t]) / sum(mask[t])  for each d in hidden_dim
    ///
    /// If all mask values are zero (degenerate case), returns the zero vector.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Embed` when the ONNX output's `seq_len` differs
    /// from the tokenizer's `batch_seq_len`, which indicates a model/tokenizer
    /// configuration mismatch. Also errors when the computed tensor slice is
    /// out-of-bounds (truncated output tensor from a mismatched export).
    fn mean_pool(
        output_data: &[f32],
        attention_mask_flat: &[i64],
        batch_idx: usize,
        seq_len: usize,
        hidden_dim: usize,
        batch_seq_len: usize,
    ) -> Result<Vec<f32>, NeuronCiteError> {
        // Guard: the ONNX output sequence length must match the tokenizer's.
        // A mismatch indicates that the model was exported with a fixed
        // sequence length that differs from the tokenizer's padding length,
        // or that a wrong tokenizer.json was paired with the model.
        if seq_len != batch_seq_len {
            return Err(NeuronCiteError::Embed(format!(
                "mean_pool: ONNX output seq_len {} != tokenizer seq_len {} — \
                 model and tokenizer sequence lengths do not match",
                seq_len, batch_seq_len
            )));
        }

        let mut embedding = vec![0.0_f32; hidden_dim];
        let mut mask_sum = 0.0_f32;

        for t in 0..seq_len {
            let mask_val = attention_mask_flat[batch_idx * batch_seq_len + t] as f32;
            if mask_val > 0.0 {
                mask_sum += mask_val;
                let token_offset = batch_idx.saturating_mul(seq_len).saturating_mul(hidden_dim)
                    + t.saturating_mul(hidden_dim);
                let token_end = token_offset + hidden_dim;
                if token_end > output_data.len() {
                    return Err(NeuronCiteError::Embed(format!(
                        "mean_pool: token slice [{}..{}] out of bounds for tensor len {} \
                         (batch_idx={}, t={}, seq_len={}, hidden_dim={})",
                        token_offset,
                        token_end,
                        output_data.len(),
                        batch_idx,
                        t,
                        seq_len,
                        hidden_dim
                    )));
                }
                for d in 0..hidden_dim {
                    embedding[d] += output_data[token_offset + d] * mask_val;
                }
            }
        }

        if mask_sum > f32::EPSILON {
            let inv_sum = 1.0 / mask_sum;
            for val in embedding.iter_mut().take(hidden_dim) {
                *val *= inv_sum;
            }
        }

        Ok(embedding)
    }

    /// Extracts the embedding at the last non-padding token position.
    /// Scans the attention mask from right to left to find the last position
    /// where mask == 1, then copies the hidden_dim values at that position
    /// via a contiguous slice copy.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Embed` when the computed slice index is
    /// out-of-bounds, indicating a shape mismatch between the ONNX output
    /// tensor and the expected [batch_size, seq_len, hidden_dim] layout.
    fn last_token_pool(
        output_data: &[f32],
        attention_mask_flat: &[i64],
        batch_idx: usize,
        seq_len: usize,
        hidden_dim: usize,
        batch_seq_len: usize,
    ) -> Result<Vec<f32>, NeuronCiteError> {
        // Find the index of the last non-padding token by scanning backward.
        let mask_start = batch_idx * batch_seq_len;
        let mut last_real_pos = 0;
        for t in (0..seq_len).rev() {
            if attention_mask_flat[mask_start + t] != 0 {
                last_real_pos = t;
                break;
            }
        }

        let base = batch_idx * seq_len * hidden_dim + last_real_pos * hidden_dim;
        let end = base
            .checked_add(hidden_dim)
            .filter(|&e| e <= output_data.len())
            .ok_or_else(|| {
                NeuronCiteError::Embed(format!(
                    "last_token_pool: slice [{}..{}] out of bounds for tensor len {} \
                     (batch_idx={}, last_real_pos={}, seq_len={}, hidden_dim={})",
                    base,
                    base + hidden_dim,
                    output_data.len(),
                    batch_idx,
                    last_real_pos,
                    seq_len,
                    hidden_dim
                ))
            })?;
        Ok(output_data[base..end].to_vec())
    }

    /// Extracts the \[CLS\] token embedding at sequence position 0.
    /// Uses a contiguous slice copy which the compiler can lower to memcpy.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Embed` when the computed slice index is
    /// out-of-bounds, indicating that the ONNX output tensor has fewer
    /// elements than expected by the [batch_size, seq_len, hidden_dim] shape.
    fn cls_pool(
        output_data: &[f32],
        batch_idx: usize,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<Vec<f32>, NeuronCiteError> {
        // [batch_idx, 0, d] in flat row-major layout is batch_idx * seq_len * hidden_dim + d.
        let base = batch_idx * seq_len * hidden_dim;
        let end = base
            .checked_add(hidden_dim)
            .filter(|&e| e <= output_data.len())
            .ok_or_else(|| {
                NeuronCiteError::Embed(format!(
                    "cls_pool: slice [{}..{}] out of bounds for tensor len {} \
                     (batch_idx={}, seq_len={}, hidden_dim={})",
                    base,
                    base + hidden_dim,
                    output_data.len(),
                    batch_idx,
                    seq_len,
                    hidden_dim
                ))
            })?;
        Ok(output_data[base..end].to_vec())
    }
}

impl EmbeddingBackend for OrtEmbedder {
    /// Returns the human-readable name of this backend.
    fn name(&self) -> &str {
        "ONNX Runtime"
    }

    /// Returns the dimensionality of embedding vectors produced by the loaded model.
    fn vector_dimension(&self) -> usize {
        self.dimension
    }

    /// Loads an ONNX model from the local cache and initializes the tokenizer.
    ///
    /// The model directory is located via `cache::model_dir` with revision "main".
    /// The directory must contain `model.onnx` and `tokenizer.json` files.
    /// The model's architecture-specific configuration (pooling strategy, sequence
    /// length, input tensor layout) is resolved from the static catalog.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Embed` if the model files are missing, if the
    /// model ID is not in the supported catalog, or if the ONNX session cannot
    /// be created.
    fn load_model(&mut self, model_id: &str) -> Result<(), NeuronCiteError> {
        // Skip reloading if the requested model is already the active model
        // with a valid ONNX session. This avoids recreating the session on
        // every indexing run when the model has not changed.
        if self.model_id.as_deref() == Some(model_id) && self.session.is_some() {
            tracing::debug!(model_id = model_id, "model already loaded, skipping reload");
            return Ok(());
        }

        let (model_path, tokenizer_path) = Self::resolve_model_paths(model_id)
            .map_err(|e| NeuronCiteError::Embed(e.to_string()))?;

        // Look up the model's configuration from the static catalog.
        // `find_model_config` returns a `&'static` reference; the config
        // is cloned into `self.model_config` because `OrtEmbedder` owns
        // its config for the lifetime of the loaded model.
        let model_cfg = find_model_config(model_id).ok_or_else(|| {
            NeuronCiteError::Embed(format!(
                "model '{}' is not in the supported model catalog; available models: {}",
                model_id,
                supported_model_configs()
                    .iter()
                    .map(|c| c.model_id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;

        let dimension = model_cfg.vector_dimension;

        // On macOS, skip CoreML and use CPU with maximum threads (Level3 +
        // NEON SIMD). CoreML partitions the ONNX graph into many subgraphs
        // (38 for BGE-small, similar for other BERT-family models) because
        // ~65 operators (Shape, Gather, Unsqueeze, etc.) are not supported
        // by CoreML's MLProgram format. Each partition boundary requires a
        // synchronous data copy between CoreML memory and CPU memory. For
        // BERT-class encoder models this makes CoreML inference SLOWER than
        // pure CPU on Apple Silicon:
        //
        //   - CUDA (RTX 4090): 1 partition, all ops on GPU, ~5-10ms/batch
        //   - CoreML (M5 Pro): 38 partitions, 38 data transfers/batch,
        //     plus 42s shape compilation for [32,512] on first use
        //   - CPU (M5 Pro): 0 partitions, Level3 fused ops, NEON SIMD,
        //     12 cores, ~200-500ms/batch — much faster in practice
        //
        // CoreML would only be beneficial for models where nearly ALL
        // operators are supported (< 5 partitions) AND the model is large
        // enough that ANE throughput outweighs the partition overhead.
        // Standard ONNX BERT exports do not meet this criterion because
        // they use operators that CoreML assigns to CPU, fragmenting the
        // graph. A Metal/MPS execution provider (direct GPU access without
        // CoreML's partitioning) does not exist in ONNX Runtime yet.
        //
        // On Windows/Linux, GPU acceleration (CUDA, DirectML, ROCm) runs
        // the entire graph in a single partition and is dramatically faster.
        let use_gpu = !cfg!(target_os = "macos");
        let config = OrtSessionConfig {
            model_path,
            use_gpu,
        };
        let (ort_session, capabilities) =
            session::create_session(&config).map_err(|e| NeuronCiteError::Embed(e.to_string()))?;

        tracing::info!(
            active_ep = %capabilities.active_ep,
            system_memory_gb = capabilities.system_memory_bytes / (1024 * 1024 * 1024),
            unified_memory = capabilities.unified_memory,
            "inference capabilities detected"
        );

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| NeuronCiteError::Embed(format!("failed to load tokenizer: {e}")))?;

        self.capabilities = capabilities;
        self.session = Some(Mutex::new(ort_session));
        self.tokenizer = Some(TokenizerWrapper::new(tokenizer));
        self.dimension = dimension;
        self.model_id = Some(model_id.to_string());

        tracing::info!(
            model_id = model_id,
            dimension = dimension,
            pooling = %model_cfg.pooling,
            max_seq_len = model_cfg.max_seq_len,
            uses_token_type_ids = model_cfg.uses_token_type_ids,
            "ONNX Runtime model loaded"
        );

        self.model_config = Some(model_cfg.clone());

        Ok(())
    }

    /// Computes dense vector embeddings for a batch of text strings.
    ///
    /// The inference pipeline adapts to the loaded model's configuration:
    ///
    /// - **Tokenization**: Uses the model's `max_seq_len` (512 for BERT, 8192 for decoders).
    /// - **Input tensors**: Constructs `input_ids` and `attention_mask` for all models.
    ///   Adds `token_type_ids` (all zeros) only for BERT-family models that require it.
    /// - **Instruction prefix**: Prepends `document_prefix` to each input text for decoder
    ///   models that use instruction-tuned embeddings.
    /// - **Pooling**: Applies the model's configured pooling strategy (CLS, mean, last-token)
    ///   to extract a fixed-size vector from the variable-length sequence output.
    /// - **Normalization**: L2-normalizes each embedding to enable cosine similarity as dot product.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Embed` if the model is not loaded, if
    /// tokenization fails, or if the ONNX session inference fails.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
        let session_mutex = self.session.as_ref().ok_or_else(|| {
            NeuronCiteError::Embed("model not loaded; call load_model first".into())
        })?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| NeuronCiteError::Embed("tokenizer not loaded".into()))?;
        let model_cfg = self.model_config.as_ref().ok_or_else(|| {
            NeuronCiteError::Embed("model config not set; call load_model first".into())
        })?;

        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Prepend instruction prefix for decoder models that require it
        // (e.g., Qwen3 "Instruct: ..." prefix). For encoder models (5 of 8
        // supported models), document_prefix is empty and the original text
        // slice is passed directly to the tokenizer without allocating
        // intermediate String copies or cloning the reference slice.
        let encoding = if model_cfg.document_prefix.is_empty() {
            tokenizer
                .encode_batch(texts, model_cfg.max_seq_len)
                .map_err(|e| NeuronCiteError::Embed(e.to_string()))?
        } else {
            let prefixed_storage: Vec<String> = texts
                .iter()
                .map(|&t| format!("{}{}", model_cfg.document_prefix, t))
                .collect();
            let text_refs: Vec<&str> = prefixed_storage.iter().map(String::as_str).collect();
            tokenizer
                .encode_batch(&text_refs, model_cfg.max_seq_len)
                .map_err(|e| NeuronCiteError::Embed(e.to_string()))?
        };

        let batch_size = encoding.token_ids.len();

        // Guard: the tokenizer must return exactly one sequence per input text.
        // An empty result for a non-empty input indicates a tokenizer config
        // mismatch (wrong model directory, corrupted tokenizer.json). Without
        // this check, encoding.token_ids[0] on the next line would panic with
        // an index-out-of-bounds on production input, crashing the worker thread.
        if encoding.token_ids.is_empty() {
            return Err(NeuronCiteError::Embed(
                "tokenizer returned zero sequences for non-empty input batch — \
                 check that the model directory contains a valid tokenizer.json"
                    .into(),
            ));
        }
        // A count mismatch (returned != requested) indicates internal tokenizer
        // inconsistency. Downstream code assumes 1-to-1 alignment between input
        // texts and output sequences when indexing into encoding.token_ids[i].
        if batch_size != texts.len() {
            return Err(NeuronCiteError::Embed(format!(
                "tokenizer returned {} sequences for {} inputs — \
                 tokenizer config mismatch or corrupted batch",
                batch_size,
                texts.len()
            )));
        }

        let seq_len = encoding.token_ids[0].len();

        // Build flat i64 arrays for input tensors. All ONNX models expect i64
        // tensors of shape [batch_size, seq_len].
        let mut input_ids_flat: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_flat: Vec<i64> = Vec::with_capacity(batch_size * seq_len);

        for i in 0..batch_size {
            for j in 0..seq_len {
                input_ids_flat.push(i64::from(encoding.token_ids[i][j]));
                attention_mask_flat.push(i64::from(encoding.attention_masks[i][j]));
            }
        }

        let shape = [batch_size, seq_len];

        let input_ids_tensor =
            TensorRef::from_array_view((shape, &*input_ids_flat)).map_err(|e| {
                NeuronCiteError::Embed(format!("failed to create input_ids tensor: {e}"))
            })?;
        let attention_mask_tensor = TensorRef::from_array_view((shape, &*attention_mask_flat))
            .map_err(|e| {
                NeuronCiteError::Embed(format!("failed to create attention_mask tensor: {e}"))
            })?;

        // Acquire the session lock for the mutable inference call.
        let mut session = session_mutex
            .lock()
            .map_err(|e| NeuronCiteError::Embed(format!("session lock poisoned: {e}")))?;

        // Construct input tensor map based on the model's requirements.
        //
        // Three input configurations exist:
        //
        // 1. Encoder models with token_type_ids (BERT, BGE, MiniLM, GTE-Large):
        //    input_ids + attention_mask + token_type_ids (all zeros for single-segment).
        //
        // 2. Decoder models with position_ids (Qwen3-Embedding):
        //    input_ids + attention_mask + position_ids (sequential [0..seq_len-1]).
        //    The ONNX export of Qwen3 requires explicit position_ids because rotary
        //    position embeddings (RoPE) are computed from this tensor at runtime.
        //
        // 3. Decoder models without position_ids:
        //    input_ids + attention_mask only.
        let outputs = if model_cfg.uses_token_type_ids {
            let token_type_ids_flat: Vec<i64> = vec![0_i64; batch_size * seq_len];
            let token_type_ids_tensor = TensorRef::from_array_view((shape, &*token_type_ids_flat))
                .map_err(|e| {
                    NeuronCiteError::Embed(format!("failed to create token_type_ids tensor: {e}"))
                })?;

            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ];
            session
                .run(inputs)
                .map_err(|e| NeuronCiteError::Embed(format!("ONNX inference failed: {e}")))?
        } else if model_cfg.needs_position_ids {
            // Build position_ids as [0, 1, 2, ..., seq_len-1] repeated for each
            // batch element. Padding positions are still numbered sequentially;
            // the attention_mask ensures they do not affect the output.
            let mut position_ids_flat: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
            for _b in 0..batch_size {
                for pos in 0..seq_len {
                    position_ids_flat.push(pos as i64);
                }
            }
            let position_ids_tensor = TensorRef::from_array_view((shape, &*position_ids_flat))
                .map_err(|e| {
                    NeuronCiteError::Embed(format!("failed to create position_ids tensor: {e}"))
                })?;

            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "position_ids" => position_ids_tensor,
            ];
            session
                .run(inputs)
                .map_err(|e| NeuronCiteError::Embed(format!("ONNX inference failed: {e}")))?
        } else {
            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ];
            session
                .run(inputs)
                .map_err(|e| NeuronCiteError::Embed(format!("ONNX inference failed: {e}")))?
        };

        // Extract the output tensor. The shape depends on the model architecture:
        // - Encoder models: [batch_size, seq_len, hidden_dim] (3-D)
        // - Some models: [batch_size, hidden_dim] (2-D, pre-pooled)
        let output_value = outputs.iter().next().map(|(_, v)| v).ok_or_else(|| {
            NeuronCiteError::Embed("no output tensor in ONNX session result".into())
        })?;

        // Some ONNX models (e.g. Qwen3-Embedding-0.6B) produce f16 output tensors
        // while others (e.g. BAAI/bge-small-en-v1.5) produce f32. Attempt f32
        // extraction first; on type mismatch, fall back to f16 extraction and
        // convert each element to f32 for the downstream pooling logic.
        let (shape_dims_owned, output_data_owned): (Vec<i64>, Vec<f32>) =
            if let Ok((shape, data)) = output_value.try_extract_tensor::<f32>() {
                (shape.to_vec(), data.to_vec())
            } else {
                let (shape, data_f16) = output_value.try_extract_tensor::<f16>().map_err(|e| {
                    NeuronCiteError::Embed(format!(
                        "failed to extract output tensor as f32 or f16: {e}"
                    ))
                })?;
                let converted = convert_f16_to_f32(data_f16);
                (shape.to_vec(), converted)
            };

        let output_data: &[f32] = &output_data_owned;

        // Shape contains dimension sizes as i64; e.g. [batch_size, seq_len, hidden_dim].
        let shape_dims: &[i64] = &shape_dims_owned;
        let ndims = shape_dims.len();

        let mut embeddings = Vec::with_capacity(batch_size);

        if ndims == 3 {
            // Shape: [batch_size, seq_len, hidden_dim].
            // Use checked_dim to reject negative (dynamic) ONNX axis values
            // that would wrap to enormous usize values and cause OOB panics.
            let s1 = checked_dim(shape_dims[1], "seq_len")?;
            let hidden_dim = checked_dim(shape_dims[2], "hidden_dim")?;

            for b in 0..batch_size {
                // All three pool strategies now return Result. Propagate errors
                // immediately so that a single malformed batch does not silently
                // corrupt the embedding vector list.
                let mut embedding = match model_cfg.pooling {
                    PoolingStrategy::Cls => Self::cls_pool(output_data, b, s1, hidden_dim)?,
                    PoolingStrategy::MeanPooling => Self::mean_pool(
                        output_data,
                        &attention_mask_flat,
                        b,
                        s1,
                        hidden_dim,
                        seq_len,
                    )?,
                    PoolingStrategy::LastToken => Self::last_token_pool(
                        output_data,
                        &attention_mask_flat,
                        b,
                        s1,
                        hidden_dim,
                        seq_len,
                    )?,
                };
                l2_normalize(&mut embedding);
                embeddings.push(embedding);
            }
        } else if ndims == 2 {
            // Shape: [batch_size, hidden_dim] — some models output pooled embeddings directly.
            let hidden_dim = checked_dim(shape_dims[1], "hidden_dim")?;
            for b in 0..batch_size {
                let base = b * hidden_dim;
                // Bounds check prevents panic when the tensor is shorter than expected.
                let end = base
                    .checked_add(hidden_dim)
                    .filter(|&e| e <= output_data.len())
                    .ok_or_else(|| {
                        NeuronCiteError::Embed(format!(
                            "2D output slice [{}..{}] out of bounds for tensor len {} \
                         (batch_idx={}, hidden_dim={})",
                            base,
                            base + hidden_dim,
                            output_data.len(),
                            b,
                            hidden_dim
                        ))
                    })?;
                let mut embedding = output_data[base..end].to_vec();
                l2_normalize(&mut embedding);
                embeddings.push(embedding);
            }
        } else {
            return Err(NeuronCiteError::Embed(format!(
                "unexpected output tensor shape: {shape_dims:?}"
            )));
        }

        Ok(embeddings)
    }

    /// Computes the embedding for a single text string. Delegates to `embed_batch`
    /// with a single-element slice to reuse the batched inference path.
    fn embed_single(&self, text: &str) -> Result<Vec<f32>, NeuronCiteError> {
        let results = self.embed_batch(&[text])?;
        results.into_iter().next().ok_or_else(|| {
            NeuronCiteError::Embed("embed_batch returned empty results for single input".into())
        })
    }

    /// Returns whether hardware-accelerated GPU inference is available.
    /// Checks CUDA (NVIDIA) on Windows/Linux, DirectML (DirectX 12) on
    /// Windows, and CoreML (Apple Neural Engine + GPU) on macOS.
    /// Returns true if any GPU execution provider can be used.
    ///
    /// GPU detection is deferred until the ORT shared library is loaded
    /// into the process. Before that point (e.g., during first-run startup
    /// before the user confirms the welcome dialog), this returns false
    /// to prevent triggering an automatic ORT download (~450-900 MB).
    /// ORT is loaded by `create_session()` when a model is activated.
    fn supports_gpu(&self) -> bool {
        if !session::is_ort_library_loaded() {
            return false;
        }
        session::is_cuda_available()
            || session::is_directml_available()
            || session::is_coreml_available()
    }

    /// Returns the list of embedding models supported by the ONNX Runtime backend.
    /// Includes both encoder (BERT-family) and decoder (LLM-based) models.
    fn available_models(&self) -> Vec<ModelInfo> {
        supported_model_configs()
            .iter()
            .map(|cfg| ModelInfo {
                id: cfg.model_id.clone(),
                display_name: cfg.display_name.clone(),
                vector_dimension: cfg.vector_dimension,
                backend: "ort".to_string(),
            })
            .collect()
    }

    /// Returns the Hugging Face identifier of the currently loaded model,
    /// or an empty string if no model is loaded.
    fn loaded_model_id(&self) -> String {
        self.model_id.clone().unwrap_or_default()
    }

    /// Returns the maximum input sequence length of the loaded model.
    /// The value is read from the `EmbeddingModelConfig` populated during
    /// `load_model`. If no model is loaded, returns the default 512.
    ///
    /// Models with long context windows (e.g. Qwen3-Embedding with 8192,
    /// GTE-Large with 8192) return their actual maximum to allow the
    /// indexing pipeline to compute appropriately small batch sizes.
    fn max_sequence_length(&self) -> usize {
        self.model_config
            .as_ref()
            .map_or(512, |cfg| cfg.max_seq_len)
    }

    /// Serializes the loaded model's tokenizer to a JSON string. The tokenizer
    /// JSON is consumed by `TokenWindowStrategy` (in neuroncite-chunk) to
    /// reconstruct a `tokenizers::Tokenizer` instance for token-based chunking.
    ///
    /// Returns `None` if no model (and therefore no tokenizer) is loaded.
    fn tokenizer_json(&self) -> Option<String> {
        self.tokenizer
            .as_ref()
            .and_then(|tw| tw.inner().to_string(false).ok())
    }

    fn inference_capabilities(&self) -> neuroncite_core::InferenceCapabilities {
        self.capabilities.clone()
    }
}

/// Converts an ONNX output tensor dimension from the raw `i64` shape value to
/// a `usize` index. ONNX models may export dynamic axes as -1 (or any negative
/// value); casting a negative i64 directly to usize wraps to a very large number
/// on 64-bit platforms, causing out-of-bounds panics in the pooling logic.
///
/// Returns `NeuronCiteError::Embed` when the dimension is negative (dynamic axis
/// or malformed shape) with a message that names the axis for diagnostics.
fn checked_dim(val: i64, axis_name: &str) -> Result<usize, NeuronCiteError> {
    if val < 0 {
        return Err(NeuronCiteError::Embed(format!(
            "ONNX output tensor has a dynamic axis '{}' (value={val}). \
             The model must be exported with static output shapes for all axes. \
             Re-export the model with a fixed batch size and sequence length.",
            axis_name
        )));
    }
    Ok(val as usize)
}

/// Converts a slice of IEEE 754 half-precision (f16) values to single-precision
/// (f32). Used when ONNX models (e.g. Qwen3-Embedding-0.6B) produce f16 output
/// tensors that the downstream pooling and normalization logic expects as f32.
fn convert_f16_to_f32(data: &[f16]) -> Vec<f32> {
    data.iter().map(|v| v.to_f32()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Unit tests for pooling strategies (no model download required)
    // -----------------------------------------------------------------------

    /// T-EMB-POOL-001: CLS pooling extracts position 0 for each batch element.
    /// Verifies that `cls_pool` returns the first token's hidden state from a
    /// synthetic 3-D output tensor.
    #[test]
    fn t_emb_pool_001_cls_pooling_extracts_position_zero() {
        // Synthetic output: batch=2, seq_len=3, hidden_dim=4
        // Batch 0: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        // Batch 1: [[13,14,15,16], [17,18,19,20], [21,22,23,24]]
        let output_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();

        let emb0 = OrtEmbedder::cls_pool(&output_data, 0, 3, 4).unwrap();
        assert_eq!(emb0, vec![1.0, 2.0, 3.0, 4.0]);

        let emb1 = OrtEmbedder::cls_pool(&output_data, 1, 3, 4).unwrap();
        assert_eq!(emb1, vec![13.0, 14.0, 15.0, 16.0]);
    }

    /// T-EMB-POOL-002: Mean pooling computes attention-weighted average, excluding
    /// padding positions (mask == 0).
    #[test]
    fn t_emb_pool_002_mean_pooling_excludes_padding() {
        // Synthetic output: batch=1, seq_len=3, hidden_dim=2
        // Token 0: [2.0, 4.0] (mask=1)
        // Token 1: [6.0, 8.0] (mask=1)
        // Token 2: [100.0, 200.0] (mask=0, padding -- excluded)
        let output_data = vec![2.0, 4.0, 6.0, 8.0, 100.0, 200.0];
        let attention_mask = vec![1_i64, 1, 0];

        let emb = OrtEmbedder::mean_pool(&output_data, &attention_mask, 0, 3, 2, 3).unwrap();

        // Mean of token 0 and token 1: [(2+6)/2, (4+8)/2] = [4.0, 6.0]
        assert!(
            (emb[0] - 4.0).abs() < 1e-6,
            "dim 0: expected 4.0, got {}",
            emb[0]
        );
        assert!(
            (emb[1] - 6.0).abs() < 1e-6,
            "dim 1: expected 6.0, got {}",
            emb[1]
        );
    }

    /// T-EMB-POOL-003: Mean pooling returns zero vector when all mask values are zero.
    #[test]
    fn t_emb_pool_003_mean_pooling_zero_mask_returns_zero_vector() {
        let output_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let attention_mask = vec![0_i64, 0, 0];

        let emb = OrtEmbedder::mean_pool(&output_data, &attention_mask, 0, 3, 2, 3).unwrap();
        assert_eq!(emb, vec![0.0, 0.0]);
    }

    /// T-EMB-POOL-004: Last-token pooling extracts the final non-padding position.
    #[test]
    fn t_emb_pool_004_last_token_extracts_final_real_token() {
        // Synthetic output: batch=1, seq_len=4, hidden_dim=3
        // Token 0: [1,2,3] (mask=1)
        // Token 1: [4,5,6] (mask=1)  <-- last real token
        // Token 2: [7,8,9] (mask=0, padding)
        // Token 3: [10,11,12] (mask=0, padding)
        let output_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let attention_mask = vec![1_i64, 1, 0, 0];

        let emb = OrtEmbedder::last_token_pool(&output_data, &attention_mask, 0, 4, 3, 4).unwrap();

        // Last real token is at position 1: [4.0, 5.0, 6.0]
        assert_eq!(emb, vec![4.0, 5.0, 6.0]);
    }

    /// T-EMB-POOL-005: Last-token pooling works when all tokens are real (no padding).
    #[test]
    fn t_emb_pool_005_last_token_no_padding() {
        // batch=1, seq_len=3, hidden_dim=2, all tokens real
        let output_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let attention_mask = vec![1_i64, 1, 1];

        let emb = OrtEmbedder::last_token_pool(&output_data, &attention_mask, 0, 3, 2, 3).unwrap();

        // Last real token is at position 2: [5.0, 6.0]
        assert_eq!(emb, vec![5.0, 6.0]);
    }

    /// T-EMB-POOL-006: Mean pooling works correctly for batched inputs where
    /// different batch elements have different padding amounts.
    #[test]
    fn t_emb_pool_006_mean_pooling_batched_different_padding() {
        // batch=2, seq_len=3, hidden_dim=2
        // Batch 0: [[2,4], [6,8], [0,0]] masks=[1,1,0] -> mean=[(2+6)/2, (4+8)/2]=[4,6]
        // Batch 1: [[1,3], [5,7], [9,11]] masks=[1,1,1] -> mean=[(1+5+9)/3, (3+7+11)/3]=[5,7]
        let output_data = vec![
            2.0, 4.0, 6.0, 8.0, 0.0, 0.0, // batch 0
            1.0, 3.0, 5.0, 7.0, 9.0, 11.0, // batch 1
        ];
        let attention_mask = vec![
            1_i64, 1, 0, // batch 0
            1, 1, 1, // batch 1
        ];

        let emb0 = OrtEmbedder::mean_pool(&output_data, &attention_mask, 0, 3, 2, 3).unwrap();
        assert!((emb0[0] - 4.0).abs() < 1e-6);
        assert!((emb0[1] - 6.0).abs() < 1e-6);

        let emb1 = OrtEmbedder::mean_pool(&output_data, &attention_mask, 1, 3, 2, 3).unwrap();
        assert!((emb1[0] - 5.0).abs() < 1e-6);
        assert!((emb1[1] - 7.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Model config catalog tests
    // -----------------------------------------------------------------------

    /// T-EMB-CFG-001: Every model in the catalog has a valid, non-empty model_id
    /// and a positive vector_dimension.
    #[test]
    fn t_emb_cfg_001_catalog_entries_are_valid() {
        for cfg in supported_model_configs() {
            assert!(!cfg.model_id.is_empty(), "model_id must not be empty");
            assert!(
                !cfg.display_name.is_empty(),
                "display_name must not be empty for '{}'",
                cfg.model_id
            );
            assert!(
                cfg.vector_dimension > 0,
                "vector_dimension must be positive for '{}'",
                cfg.model_id
            );
            assert!(
                cfg.max_seq_len > 0,
                "max_seq_len must be positive for '{}'",
                cfg.model_id
            );
        }
    }

    /// T-EMB-CFG-002: No duplicate model IDs exist in the catalog.
    #[test]
    fn t_emb_cfg_002_no_duplicate_model_ids() {
        let configs = supported_model_configs();
        let mut seen = std::collections::HashSet::new();
        for cfg in configs {
            assert!(
                seen.insert(&cfg.model_id),
                "duplicate model_id in catalog: '{}'",
                cfg.model_id
            );
        }
    }

    /// T-EMB-CFG-003: `find_model_config` returns Some for registered models
    /// and None for unknown models.
    #[test]
    fn t_emb_cfg_003_find_model_config_known_and_unknown() {
        assert!(
            find_model_config("BAAI/bge-small-en-v1.5").is_some(),
            "BGE small should be in catalog"
        );
        assert!(
            find_model_config("Qwen/Qwen3-Embedding-0.6B").is_some(),
            "Qwen3-Embedding should be in catalog"
        );
        assert!(
            find_model_config("nonexistent/model-xyz").is_none(),
            "unknown model should return None"
        );
    }

    /// T-EMB-CFG-004: Decoder models (MeanPooling, LastToken) do not use
    /// token_type_ids. BERT-based CLS models use token_type_ids, while
    /// XLM-RoBERTa-based CLS models (BGE-M3) do not. The test verifies
    /// decoder models never have token_type_ids and that the CLS/token_type_ids
    /// relationship is consistent within each architecture family.
    #[test]
    fn t_emb_cfg_004_encoder_decoder_config_consistency() {
        for cfg in supported_model_configs() {
            match cfg.pooling {
                PoolingStrategy::Cls => {
                    // CLS pooling is used by both BERT models (which need
                    // token_type_ids) and XLM-RoBERTa models like BGE-M3
                    // (which do not). No universal assertion possible here.
                }
                PoolingStrategy::MeanPooling | PoolingStrategy::LastToken => {
                    assert!(
                        !cfg.uses_token_type_ids,
                        "decoder model '{}' should not use token_type_ids",
                        cfg.model_id
                    );
                }
            }
        }
    }

    /// T-EMB-CFG-005: Decoder models have a non-empty query_prefix.
    /// Encoder models have an empty query_prefix.
    #[test]
    fn t_emb_cfg_005_decoder_models_have_query_prefix() {
        for cfg in supported_model_configs() {
            match cfg.pooling {
                PoolingStrategy::MeanPooling | PoolingStrategy::LastToken => {
                    assert!(
                        !cfg.query_prefix.is_empty(),
                        "decoder model '{}' should have a non-empty query_prefix",
                        cfg.model_id
                    );
                }
                PoolingStrategy::Cls => {
                    assert!(
                        cfg.query_prefix.is_empty(),
                        "encoder model '{}' should have an empty query_prefix",
                        cfg.model_id
                    );
                }
            }
        }
    }

    /// T-EMB-CFG-006: Decoder models in the catalog use a max_seq_len greater
    /// than the default BERT 512, reflecting their longer context windows.
    /// Encoder models have a max_seq_len of 512 or higher (GTE-Large supports 8192).
    #[test]
    fn t_emb_cfg_006_decoder_models_have_extended_seq_len() {
        for cfg in supported_model_configs() {
            match cfg.pooling {
                PoolingStrategy::MeanPooling | PoolingStrategy::LastToken => {
                    assert!(
                        cfg.max_seq_len > 512,
                        "decoder model '{}' should have max_seq_len > 512, got {}",
                        cfg.model_id,
                        cfg.max_seq_len
                    );
                }
                PoolingStrategy::Cls => {
                    assert!(
                        cfg.max_seq_len >= 512,
                        "encoder model '{}' should have max_seq_len >= 512, got {}",
                        cfg.model_id,
                        cfg.max_seq_len
                    );
                }
            }
        }
    }

    /// T-EMB-CFG-008: Models with `needs_position_ids == true` do not use
    /// token_type_ids, and vice versa. These two input requirements are
    /// mutually exclusive across the current catalog.
    #[test]
    fn t_emb_cfg_008_position_ids_and_token_type_ids_exclusive() {
        for cfg in supported_model_configs() {
            if cfg.needs_position_ids {
                assert!(
                    !cfg.uses_token_type_ids,
                    "model '{}' with needs_position_ids should not use token_type_ids",
                    cfg.model_id
                );
            }
        }
    }

    /// T-EMB-CFG-007: `PoolingStrategy` Display trait produces expected strings.
    #[test]
    fn t_emb_cfg_007_pooling_strategy_display() {
        assert_eq!(format!("{}", PoolingStrategy::Cls), "cls");
        assert_eq!(format!("{}", PoolingStrategy::MeanPooling), "mean-pooling");
        assert_eq!(format!("{}", PoolingStrategy::LastToken), "last-token");
    }

    // -----------------------------------------------------------------------
    // Structural tests (no model download required)
    // -----------------------------------------------------------------------

    /// T-EMB-ORT-001: Unloaded embedder returns error on embed_batch.
    #[test]
    fn t_emb_ort_001_unloaded_returns_error() {
        let embedder = OrtEmbedder::unloaded();
        let result = embedder.embed_batch(&["test"]);
        assert!(result.is_err(), "unloaded embedder must return error");
    }

    /// T-EMB-ORT-002: available_models returns entries for all catalog models.
    #[test]
    fn t_emb_ort_002_available_models_matches_catalog() {
        let embedder = OrtEmbedder::unloaded();
        let models = embedder.available_models();
        let configs = supported_model_configs();
        assert_eq!(
            models.len(),
            configs.len(),
            "available_models must return one entry per catalog model"
        );
        for (model, cfg) in models.iter().zip(configs.iter()) {
            assert_eq!(model.id, cfg.model_id);
            assert_eq!(model.vector_dimension, cfg.vector_dimension);
            assert_eq!(model.backend, "ort");
        }
    }

    // -----------------------------------------------------------------------
    // f16 -> f32 conversion tests (regression prevention for half-precision
    // output models like Qwen3-Embedding-0.6B)
    // -----------------------------------------------------------------------

    /// T-EMB-F16-001: `convert_f16_to_f32` correctly converts a slice of known
    /// f16 values to their f32 equivalents. Verifies that standard IEEE 754
    /// half-precision values survive the round-trip within representable precision.
    #[test]
    fn t_emb_f16_001_convert_f16_to_f32_known_values() {
        let f16_values = vec![
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(-1.0),
            f16::from_f32(0.5),
            f16::from_f32(0.333_251_95), // closest f16 representable value to 1/3
            f16::from_f32(65504.0),      // f16 max normal value
            f16::from_f32(-65504.0),     // f16 min normal value
        ];

        let f32_result = convert_f16_to_f32(&f16_values);

        assert_eq!(f32_result.len(), 7);
        assert_eq!(f32_result[0], 0.0);
        assert_eq!(f32_result[1], 1.0);
        assert_eq!(f32_result[2], -1.0);
        assert_eq!(f32_result[3], 0.5);
        assert!(
            (f32_result[4] - 0.333_251_95).abs() < 1e-3,
            "1/3 approx mismatch"
        );
        assert_eq!(f32_result[5], 65504.0);
        assert_eq!(f32_result[6], -65504.0);
    }

    /// T-EMB-F16-002: `convert_f16_to_f32` handles an empty slice without panic.
    #[test]
    fn t_emb_f16_002_convert_f16_empty_slice() {
        let result = convert_f16_to_f32(&[]);
        assert!(result.is_empty());
    }

    /// T-EMB-F16-003: `convert_f16_to_f32` preserves the relative ordering of
    /// embedding-like values. When downstream pooling computes a mean, the
    /// converted values must produce the same ranking as the original f16 data.
    #[test]
    fn t_emb_f16_003_convert_f16_preserves_ordering() {
        // Simulate a small embedding vector in f16 with known ordering.
        let ascending: Vec<f16> = (-5..=5).map(|i| f16::from_f32(i as f32 * 0.1)).collect();

        let converted = convert_f16_to_f32(&ascending);

        // Verify monotonically non-decreasing after conversion.
        for window in converted.windows(2) {
            assert!(
                window[0] <= window[1],
                "ordering violated: {} > {}",
                window[0],
                window[1]
            );
        }
    }

    /// T-EMB-F16-004: Simulates the full output processing path (extraction +
    /// pooling) with synthetic f16 data to verify that mean-pooled embeddings
    /// from f16 output tensors produce valid, non-zero, L2-normalized vectors.
    /// This test exercises the same code path that failed for Qwen3-Embedding
    /// before the f16 fallback was added.
    #[test]
    fn t_emb_f16_004_f16_mean_pooling_produces_valid_embeddings() {
        // Simulate a 3-D output tensor: batch=1, seq_len=4, hidden_dim=3
        // representing what a Qwen model would produce as f16.
        let f16_data: Vec<f16> = vec![
            // Token 0
            f16::from_f32(0.5),
            f16::from_f32(0.3),
            f16::from_f32(-0.1),
            // Token 1
            f16::from_f32(0.7),
            f16::from_f32(-0.2),
            f16::from_f32(0.4),
            // Token 2
            f16::from_f32(0.1),
            f16::from_f32(0.6),
            f16::from_f32(0.2),
            // Token 3 (padding, mask=0)
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
        ];

        // Convert f16 -> f32 (the path that was previously broken).
        let f32_data = convert_f16_to_f32(&f16_data);
        assert_eq!(f32_data.len(), 12, "converted data length mismatch");

        // Apply mean pooling with attention mask [1, 1, 1, 0].
        // mean_pool returns Result because it validates seq_len vs batch_seq_len.
        // In this test both are 4, so the call must succeed.
        let attention_mask = vec![1_i64, 1, 1, 0];
        let embedding = OrtEmbedder::mean_pool(&f32_data, &attention_mask, 0, 4, 3, 4)
            .expect("mean_pool must succeed when seq_len equals batch_seq_len");

        // Expected mean over tokens 0-2:
        // dim 0: (0.5 + 0.7 + 0.1) / 3 = 0.4333...
        // dim 1: (0.3 + -0.2 + 0.6) / 3 = 0.2333...
        // dim 2: (-0.1 + 0.4 + 0.2) / 3 = 0.1666...
        assert!(
            (embedding[0] - 0.4333).abs() < 0.01,
            "dim 0 mismatch: {}",
            embedding[0]
        );
        assert!(
            (embedding[1] - 0.2333).abs() < 0.01,
            "dim 1 mismatch: {}",
            embedding[1]
        );
        assert!(
            (embedding[2] - 0.1666).abs() < 0.01,
            "dim 2 mismatch: {}",
            embedding[2]
        );

        // Verify L2 normalization produces a unit vector.
        let mut normalized = embedding.clone();
        crate::normalize::l2_normalize(&mut normalized);
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "L2-normalized vector must have unit length, got {norm}"
        );
    }

    // -----------------------------------------------------------------------
    // max_sequence_length tests
    // -----------------------------------------------------------------------

    /// T-EMB-SEQ-001: Unloaded OrtEmbedder returns the default max_sequence_length
    /// of 512 (matching the BERT-family default).
    #[test]
    fn t_emb_seq_001_unloaded_returns_default_max_seq_len() {
        let embedder = OrtEmbedder::unloaded();
        assert_eq!(
            embedder.max_sequence_length(),
            512,
            "unloaded embedder must return default 512"
        );
    }

    /// T-EMB-SEQ-002: Every model in the catalog has a max_seq_len that would
    /// produce a valid (>= 1) batch size through the adaptive computation.
    #[test]
    fn t_emb_seq_002_catalog_models_produce_valid_batch_sizes() {
        for cfg in supported_model_configs() {
            let batch_size = (32_usize * 512 / cfg.max_seq_len.max(1)).clamp(1, 32);
            assert!(
                batch_size >= 1,
                "model '{}' (max_seq_len={}) produces invalid batch_size={}",
                cfg.model_id,
                cfg.max_seq_len,
                batch_size
            );
        }
    }

    // -----------------------------------------------------------------------
    // Dimension safety tests (regression prevention for the mismatch bug
    // where Qwen3 sessions were stored with dimension=384)
    // -----------------------------------------------------------------------

    /// T-EMB-DIM-001: `OrtEmbedder::unloaded()` exposes an unsafe default
    /// dimension. This test documents the value so that any change to the
    /// default is a conscious decision. The dimension of an unloaded embedder
    /// must never be relied upon for session creation -- callers must always
    /// use the model catalog (`find_model_config`) for authoritative dimensions.
    #[test]
    fn t_emb_dim_001_unloaded_dimension_is_documented_default() {
        let embedder = OrtEmbedder::unloaded();
        // The unloaded embedder has dimension 384 as a historical default.
        // This value is a placeholder and must never be used to create sessions.
        assert_eq!(
            embedder.vector_dimension(),
            384,
            "unloaded embedder default dimension is 384 (placeholder only)"
        );
    }

    /// T-EMB-DIM-002: Every model in the catalog has a vector_dimension that
    /// matches the known specification. Prevents silent regression if someone
    /// changes a dimension constant in the catalog.
    #[test]
    fn t_emb_dim_002_catalog_dimensions_match_specifications() {
        let known_dimensions: &[(&str, usize)] = &[
            ("BAAI/bge-small-en-v1.5", 384),
            ("BAAI/bge-base-en-v1.5", 768),
            ("Qwen/Qwen3-Embedding-0.6B", 1024),
            ("Qwen/Qwen3-Embedding-4B", 2560),
        ];

        for &(model_id, expected_dim) in known_dimensions {
            let config = find_model_config(model_id);
            assert!(
                config.is_some(),
                "model '{}' must be in the catalog",
                model_id
            );
            let config = config.unwrap();
            assert_eq!(
                config.vector_dimension, expected_dim,
                "model '{}' dimension mismatch: expected {}, got {}",
                model_id, expected_dim, config.vector_dimension
            );
        }
    }

    /// T-EMB-DIM-003: `find_model_config` returns the correct dimension for
    /// each model in the catalog. This ensures that index handlers using
    /// `find_model_config` to resolve dimensions will always get the right
    /// value, independent of which model is loaded in the backend.
    #[test]
    fn t_emb_dim_003_find_model_config_dimension_independent_of_loaded_state() {
        // Regardless of what the OrtEmbedder's current dimension field is,
        // find_model_config must return the catalog value.
        let embedder = OrtEmbedder::unloaded();
        assert_eq!(embedder.vector_dimension(), 384);

        // find_model_config is a static catalog lookup, not dependent on
        // the embedder state.
        let qwen_cfg = find_model_config("Qwen/Qwen3-Embedding-0.6B").unwrap();
        assert_eq!(qwen_cfg.vector_dimension, 1024);

        let bge_cfg = find_model_config("BAAI/bge-small-en-v1.5").unwrap();
        assert_eq!(bge_cfg.vector_dimension, 384);
    }

    /// T-EMB-DIM-004: The unloaded embedder's dimension (384) is a real
    /// dimension shared by multiple models in the catalog. This test verifies
    /// that the default model (BAAI/bge-small-en-v1.5) is among those with
    /// dimension 384, and that models with different dimensions (e.g. Qwen3
    /// at 1024) are correctly distinguished. The core safety against the
    /// unloaded-dimension bug is the catalog-based resolution in the index
    /// handlers, not the uniqueness of dimension 384.
    #[test]
    fn t_emb_dim_004_default_model_has_dimension_384() {
        let all_configs = supported_model_configs();
        let models_with_384: Vec<&str> = all_configs
            .iter()
            .filter(|c| c.vector_dimension == 384)
            .map(|c| c.model_id.as_str())
            .collect();

        assert!(
            models_with_384.contains(&"BAAI/bge-small-en-v1.5"),
            "the default model BAAI/bge-small-en-v1.5 must have dimension 384, \
             found models with 384: {:?}",
            models_with_384
        );

        // Qwen3 models must NOT have dimension 384, ensuring the catalog
        // correctly distinguishes between model families.
        let qwen_models: Vec<&str> = all_configs
            .iter()
            .filter(|c| c.model_id.contains("Qwen3"))
            .filter(|c| c.vector_dimension == 384)
            .map(|c| c.model_id.as_str())
            .collect();

        assert!(
            qwen_models.is_empty(),
            "no Qwen3 model should have dimension 384 (that would mask the \
             unloaded-dimension bug), found: {:?}",
            qwen_models
        );
    }

    // -----------------------------------------------------------------------
    // LazyLock catalog tests
    // -----------------------------------------------------------------------

    /// T-EMB-LAZY-001: `supported_model_configs` returns the same pointer on
    /// repeated calls, confirming that the catalog is initialized once and
    /// reused without reallocation.
    #[test]
    fn t_emb_lazy_001_catalog_returns_stable_pointer() {
        let first = supported_model_configs();
        let second = supported_model_configs();
        assert!(
            std::ptr::eq(first.as_ptr(), second.as_ptr()),
            "supported_model_configs must return the same static slice on every call"
        );
    }

    /// T-EMB-LAZY-002: `find_model_config` returns a reference into the static
    /// catalog. The returned reference must point into the same memory region
    /// as the catalog slice.
    #[test]
    fn t_emb_lazy_002_find_returns_reference_into_catalog() {
        let catalog = supported_model_configs();
        let found = find_model_config("BAAI/bge-small-en-v1.5").unwrap();
        // Verify that the found reference is the same object as catalog[0].
        assert!(
            std::ptr::eq(found, &catalog[0]),
            "find_model_config must return a reference into the static catalog, not a clone"
        );
    }
}
