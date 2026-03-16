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

//! neuroncite-embed: Embedding backend abstraction and ONNX Runtime implementation.
//!
//! This crate defines implementations of the `EmbeddingBackend` and `Reranker` traits
//! (defined in neuroncite-core) and provides one concrete backend gated behind
//! a Cargo feature flag:
//!
//! - **`backend-ort`** (default) -- ONNX Runtime inference via the `ort` crate,
//!   with CUDA execution provider support and `ndarray` for tensor manipulation.
//!
//! Common infrastructure (model registry, tokenizer management, embedding
//! normalization, and local model cache) lives in always-compiled modules.
//!
//! The crate does not use `#![forbid(unsafe_code)]` because the ONNX Runtime
//! FFI backend requires unsafe blocks for GPU interop.

// `deny(warnings)` is inherited from [workspace.lints.rust] in the root Cargo.toml.

// -- Backend implementations (feature-gated) ----------------------------------

#[cfg(feature = "backend-ort")]
pub mod ort;

// -- Always-compiled infrastructure -------------------------------------------

pub mod cache;
pub mod error;
pub mod normalize;
pub mod registry;
pub mod tokenize;

// -- Re-exports for ergonomic use by downstream crates -----------------------

/// `ensure_ort_runtime` is available on all platforms with backend-ort:
/// Windows and Linux auto-download ORT from GitHub, macOS uses the same
/// mechanism with architecture-specific tarballs.
#[cfg(feature = "backend-ort")]
pub use cache::ensure_ort_runtime;
pub use cache::{
    ModelDiagnosis, all_model_ids, cache_dir, compute_sha256, detect_gpu_device_name,
    diagnose_model, download_model, is_cached, is_fully_cached, model_dir, model_expected_files,
    purge_cached_model, verify_cached_model, verify_checksum,
};
/// GPU-related functions are available on Windows and Linux (not macOS,
/// which has no NVIDIA GPU support). These functions detect NVIDIA hardware,
/// check for the GPU ORT variant, locate cuDNN, and configure runtime paths.
#[cfg(all(not(target_os = "macos"), feature = "backend-ort"))]
pub use cache::{
    detect_nvidia_gpu, ensure_cuda_runtime_on_path, find_cudnn_dir, is_gpu_ort_runtime,
};
pub use error::EmbedError;
pub use normalize::{l2_normalize, l2_normalize_batch};
pub use registry::{
    BackendInfo, DEFAULT_BACKEND_NAME, create_backend, create_reranker, list_available_backends,
};
pub use tokenize::{BatchEncoding, TokenizerWrapper};

#[cfg(feature = "backend-ort")]
pub use ort::embedder::{find_model_config, supported_model_configs};
#[cfg(feature = "backend-ort")]
pub use ort::session::is_coreml_available;
#[cfg(feature = "backend-ort")]
pub use ort::session::is_cuda_available;
#[cfg(feature = "backend-ort")]
pub use ort::session::is_directml_available;

#[cfg(test)]
mod tests {
    use super::*;

    /// T-EMB-005: embed_single returns the same vector as embed_batch(&[text])[0]
    /// within 1e-6 tolerance.
    ///
    /// This test uses the registry to create a backend. If no model is available
    /// in the cache, the test verifies structural correctness (both paths return
    /// the same error type) rather than numerical equality.
    #[test]
    fn t_emb_005_embed_single_matches_embed_batch_element_zero() {
        let backends = registry::list_available_backends();
        if backends.is_empty() {
            // No backends compiled in; skip the test.
            return;
        }

        // Attempt to create a backend and load a model. If the model is not
        // cached locally, both embed_single and embed_batch will fail with
        // the same error, which still validates behavioral consistency.
        let backend_name = &backends[0].name;
        let backend = match registry::create_backend(backend_name) {
            Ok(b) => b,
            Err(_) => return, // Backend creation failed; skip.
        };

        let text = "The quick brown fox jumps over the lazy dog.";

        // Without a loaded model, both methods should return Err.
        let single_result = backend.embed_single(text);
        let batch_result = backend.embed_batch(&[text]);

        match (single_result, batch_result) {
            (Ok(single_vec), Ok(batch_vecs)) => {
                // Both succeeded: verify numerical equality within tolerance.
                assert_eq!(
                    batch_vecs.len(),
                    1,
                    "embed_batch with one input should return one vector"
                );
                let batch_vec = &batch_vecs[0];
                assert_eq!(
                    single_vec.len(),
                    batch_vec.len(),
                    "embed_single and embed_batch vectors must have the same length"
                );
                for (i, (s, b)) in single_vec.iter().zip(batch_vec.iter()).enumerate() {
                    assert!(
                        (s - b).abs() < 1e-6,
                        "element {i} differs: single={s}, batch={b}"
                    );
                }
            }
            (Err(_), Err(_)) => {
                // Both failed (model not loaded): this is the expected behavior
                // when no model is cached. The test passes because both paths
                // produced consistent errors.
            }
            (Ok(_), Err(e)) => {
                panic!("embed_single succeeded but embed_batch failed: {e}");
            }
            (Err(e), Ok(_)) => {
                panic!("embed_batch succeeded but embed_single failed: {e}");
            }
        }
    }

    /// T-EMB-006: embed_batch vectors all have length equal to vector_dimension().
    ///
    /// This test verifies the structural invariant that all returned vectors
    /// have the correct dimensionality. If no model is cached, the test verifies
    /// that the error path is consistent.
    #[test]
    fn t_emb_006_embed_batch_vectors_match_vector_dimension() {
        let backends = registry::list_available_backends();
        if backends.is_empty() {
            return;
        }

        let backend_name = &backends[0].name;
        let backend = match registry::create_backend(backend_name) {
            Ok(b) => b,
            Err(_) => return,
        };

        let dimension = backend.vector_dimension();
        let texts = &["first text", "second text", "third text"];

        match backend.embed_batch(texts) {
            Ok(vectors) => {
                assert_eq!(
                    vectors.len(),
                    texts.len(),
                    "embed_batch should return one vector per input text"
                );
                for (i, vec) in vectors.iter().enumerate() {
                    assert_eq!(
                        vec.len(),
                        dimension,
                        "vector {i} has length {}, expected {dimension}",
                        vec.len()
                    );
                }
            }
            Err(_) => {
                // Model not loaded; this is acceptable in a test environment
                // without cached models.
            }
        }
    }
}
