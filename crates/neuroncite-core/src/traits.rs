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

// Trait definitions for the three pluggable subsystems in NeuronCite:
// embedding computation, text chunking, and result reranking.
//
// These traits are defined in neuroncite-core so that downstream crates can
// depend on the trait contract without depending on any concrete implementation.
// Implementations live in neuroncite-embed (for `EmbeddingBackend` and `Reranker`)
// and neuroncite-chunk (for `ChunkStrategy`).
//
// All traits require Send + Sync to allow sharing across tokio tasks and
// thread pool workers.

use crate::error::NeuronCiteError;
use crate::types::{Chunk, InferenceCapabilities, ModelInfo, PageText};

/// Abstraction over dense vector embedding backends (ONNX Runtime, Candle, Burn).
///
/// Implementors load a transformer model, expose its vector dimensionality,
/// and provide batch and single-text embedding methods. The trait is object-safe
/// to allow dynamic dispatch via `Box<dyn EmbeddingBackend>` in the API layer.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` because the embedding backend is shared
/// across the tokio runtime's worker threads via `Arc`.
pub trait EmbeddingBackend: Send + Sync {
    /// Returns the human-readable name of this backend (e.g., "ort").
    fn name(&self) -> &str;

    /// Returns the dimensionality of the embedding vectors produced by the
    /// currently loaded model. This value determines the HNSW index dimension
    /// and the expected length of all embedding vectors.
    fn vector_dimension(&self) -> usize;

    /// Loads the specified model into memory and prepares it for inference.
    ///
    /// The `model_id` corresponds to a Hugging Face model identifier
    /// (e.g., "BAAI/bge-small-en-v1.5").
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Embed` if the model files are not found
    /// in the cache or if loading fails.
    fn load_model(&mut self, model_id: &str) -> Result<(), NeuronCiteError>;

    /// Computes dense vector embeddings for a batch of text strings.
    /// Returns one embedding vector per input text. The length of each
    /// vector equals `self.vector_dimension()`.
    ///
    /// Batch embedding is more efficient than repeated single-text calls
    /// because it amortizes model inference overhead across multiple inputs.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Embed` if inference fails for any input
    /// in the batch.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError>;

    /// Computes the dense vector embedding for a single text string.
    /// The default implementation delegates to `embed_batch` with a
    /// single-element slice, which is correct but may be less efficient
    /// than a specialized single-text path in some backends.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Embed` if inference fails or if the
    /// batch result is unexpectedly empty.
    fn embed_single(&self, text: &str) -> Result<Vec<f32>, NeuronCiteError> {
        let results = self.embed_batch(&[text])?;
        results.into_iter().next().ok_or_else(|| {
            NeuronCiteError::Embed("embed_batch returned empty results for single input".into())
        })
    }

    /// Returns whether this backend supports GPU-accelerated inference.
    /// The GUI displays this information to help users select a backend.
    fn supports_gpu(&self) -> bool;

    /// Returns the list of models available through this backend.
    /// Each `ModelInfo` includes the model identifier, display name, vector
    /// dimension, and the backend name.
    fn available_models(&self) -> Vec<ModelInfo>;

    /// Returns the Hugging Face model identifier of the currently loaded
    /// model, or an empty string if no model is loaded. Used by the GUI
    /// to track which model is active for mismatch detection.
    fn loaded_model_id(&self) -> String;

    /// Returns the maximum input sequence length (in tokens) accepted by the
    /// loaded model. The indexing pipeline uses this value to compute an
    /// adaptive embedding batch size that fits within GPU memory.
    ///
    /// GPU self-attention memory scales as O(batch_size * seq_len^2), so
    /// models with long context windows (e.g., 8192 tokens for Qwen3) require
    /// smaller batch sizes than short-context models (512 tokens for BGE).
    ///
    /// The default value of 512 matches the BERT-family encoder models that
    /// constitute the majority of supported models. Backends with non-BERT
    /// models (e.g., OrtEmbedder with Qwen3-Embedding) override this method
    /// to return the loaded model's actual maximum sequence length.
    fn max_sequence_length(&self) -> usize {
        512
    }

    /// Returns the loaded model's tokenizer serialized as a JSON string.
    ///
    /// The token-based chunking strategy (`TokenWindowStrategy`) requires a
    /// `tokenizers::Tokenizer` instance to measure chunk boundaries in subword
    /// tokens. Since the `tokenizers` crate is not a dependency of this core
    /// crate, the tokenizer is exchanged as a JSON string that the consumer
    /// deserializes via `tokenizers::Tokenizer::from_str()`.
    ///
    /// Returns `None` if no model is loaded or if the backend does not support
    /// tokenizer export. The default implementation returns `None`.
    fn tokenizer_json(&self) -> Option<String> {
        None
    }

    /// Returns the runtime-detected hardware capabilities for the active
    /// inference session. Used by the pipeline to adapt batch sizes and
    /// threading to the platform's execution provider and memory architecture.
    ///
    /// The default implementation returns conservative CPU-only capabilities
    /// (8 GB RAM, no unified memory). Backends that detect hardware at session
    /// creation time (e.g., `OrtEmbedder`) override this to return the actual
    /// detected capabilities.
    fn inference_capabilities(&self) -> InferenceCapabilities {
        InferenceCapabilities::default()
    }
}

/// Abstraction over text chunking strategies (page-based, word-window,
/// sentence-based).
///
/// Implementors receive a sequence of `PageText` values (one per page of a
/// single document) and produce a sequence of `Chunk` values with computed
/// page ranges, document-level byte offsets, and content hashes.
pub trait ChunkStrategy: Send + Sync {
    /// Splits the extracted pages of a single document into chunks.
    ///
    /// The `pages` slice contains one `PageText` per page, ordered by
    /// `page_number` (1-indexed, ascending). The implementor concatenates
    /// the page texts with newline separators, applies its splitting
    /// logic, and returns chunks with correct `page_start`, `page_end`,
    /// `doc_text_offset_start`, `doc_text_offset_end`, and `content_hash`
    /// fields.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Chunk` if the input pages are empty or
    /// if the splitting logic encounters an unrecoverable condition.
    fn chunk(&self, pages: &[PageText]) -> Result<Vec<Chunk>, NeuronCiteError>;
}

/// Abstraction over cross-encoder reranking models.
///
/// Implementors load a cross-encoder model and score query-document pairs
/// to reorder search results by semantic relevance. Reranking is optional;
/// if no reranker is configured, the search pipeline skips this stage.
pub trait Reranker: Send + Sync {
    /// Returns the human-readable name of this reranker implementation.
    fn name(&self) -> &str;

    /// Scores a batch of candidate passages against a query string.
    /// Returns one f64 score per candidate, where higher scores indicate
    /// greater relevance. The order of scores corresponds to the order
    /// of the candidates slice.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Search` if cross-encoder inference fails.
    fn rerank_batch(&self, query: &str, candidates: &[&str]) -> Result<Vec<f64>, NeuronCiteError>;

    /// Loads the specified cross-encoder model into memory.
    ///
    /// The `model_id` corresponds to a Hugging Face model identifier.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Embed` if the model files are not found
    /// or loading fails.
    fn load_model(&mut self, model_id: &str) -> Result<(), NeuronCiteError>;
}
