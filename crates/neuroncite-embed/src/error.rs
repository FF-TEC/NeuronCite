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

//! Error types for the neuroncite-embed crate.
//!
//! `EmbedError` covers failures across all embedding backends, the model cache,
//! tokenizer initialization, and normalization. Backend-specific error variants
//! are included unconditionally so that downstream consumers can match on them
//! regardless of which backend feature is active.

use neuroncite_core::NeuronCiteError;

/// Represents all error conditions that can occur during embedding or reranking
/// operations, model loading, tokenizer initialization, and cache management.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// The requested model was not found in the local cache and could not be
    /// resolved from the registry.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// The model file could not be loaded into the inference session. This
    /// covers deserialization failures, incompatible formats, and corrupted
    /// weight files.
    #[error("model load error: {0}")]
    ModelLoad(String),

    /// The downloaded model file's SHA-256 checksum does not match the expected
    /// value recorded in the registry.
    #[error("checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch {
        /// The expected SHA-256 hex digest from the registry.
        expected: String,
        /// The computed SHA-256 hex digest of the downloaded file.
        actual: String,
    },

    /// The tokenizer failed to initialize or to encode input text.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// The ML backend failed during the forward pass (inference).
    #[error("inference error: {0}")]
    Inference(String),

    /// GPU initialization failed (CUDA driver, device allocation, or
    /// execution provider registration).
    #[error("GPU initialization error: {0}")]
    GpuInit(String),

    /// An I/O error occurred while reading or writing model files in the cache
    /// directory.
    #[error("cache I/O error: {0}")]
    CacheIo(String),

    /// The model download from a remote source failed.
    #[error("download error: {0}")]
    Download(String),

    /// A backend-specific error that does not fit any other category.
    #[error("backend error: {0}")]
    Backend(String),
}

impl From<EmbedError> for NeuronCiteError {
    /// Converts an `EmbedError` into `NeuronCiteError::Embed` by formatting the
    /// error message as a string. This conversion is used at the trait boundary
    /// where `EmbeddingBackend` and `Reranker` methods return `NeuronCiteError`.
    fn from(err: EmbedError) -> Self {
        NeuronCiteError::Embed(err.to_string())
    }
}

impl From<std::io::Error> for EmbedError {
    /// Converts a standard I/O error into `EmbedError::CacheIo`. This conversion
    /// is used when reading or writing model files in the cache directory.
    fn from(err: std::io::Error) -> Self {
        EmbedError::CacheIo(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuroncite_core::NeuronCiteError;

    // -----------------------------------------------------------------------
    // #49 -- From<EmbedError> for NeuronCiteError conversion tests
    //
    // The From impl converts every EmbedError variant into
    // NeuronCiteError::Embed(String) by calling .to_string() on the
    // EmbedError. These tests verify that each variant is routed to the
    // correct NeuronCiteError variant and that the error message is preserved.
    // -----------------------------------------------------------------------

    /// Validates that `EmbedError::ModelNotFound` converts to
    /// `NeuronCiteError::Embed` with the message containing the model name.
    #[test]
    fn t_err_001_model_not_found_converts_to_neuroncite_embed() {
        let embed_err = EmbedError::ModelNotFound("bge-small-en-v1.5".into());
        let core_err: NeuronCiteError = embed_err.into();

        match core_err {
            NeuronCiteError::Embed(msg) => {
                assert!(
                    msg.contains("bge-small-en-v1.5"),
                    "message must contain the model name, got: {msg}"
                );
                assert!(
                    msg.contains("model not found"),
                    "message must contain the error prefix, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Embed, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `EmbedError::ModelLoad` converts to
    /// `NeuronCiteError::Embed` with the load failure reason preserved.
    #[test]
    fn t_err_002_model_load_converts_to_neuroncite_embed() {
        let embed_err = EmbedError::ModelLoad("corrupted ONNX file".into());
        let core_err: NeuronCiteError = embed_err.into();

        match core_err {
            NeuronCiteError::Embed(msg) => {
                assert!(
                    msg.contains("corrupted ONNX file"),
                    "message must contain the load failure reason, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Embed, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `EmbedError::ChecksumMismatch` converts to
    /// `NeuronCiteError::Embed` with both expected and actual digests present
    /// in the error message.
    #[test]
    fn t_err_003_checksum_mismatch_converts_to_neuroncite_embed() {
        let embed_err = EmbedError::ChecksumMismatch {
            expected: "aabb".into(),
            actual: "ccdd".into(),
        };
        let core_err: NeuronCiteError = embed_err.into();

        match core_err {
            NeuronCiteError::Embed(msg) => {
                assert!(
                    msg.contains("aabb"),
                    "message must contain the expected digest, got: {msg}"
                );
                assert!(
                    msg.contains("ccdd"),
                    "message must contain the actual digest, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Embed, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `EmbedError::Tokenizer` converts to
    /// `NeuronCiteError::Embed` with the tokenizer failure message preserved.
    #[test]
    fn t_err_004_tokenizer_converts_to_neuroncite_embed() {
        let embed_err = EmbedError::Tokenizer("unknown token".into());
        let core_err: NeuronCiteError = embed_err.into();

        match core_err {
            NeuronCiteError::Embed(msg) => {
                assert!(
                    msg.contains("unknown token"),
                    "message must contain the tokenizer error, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Embed, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `EmbedError::Inference` converts to
    /// `NeuronCiteError::Embed` with the inference failure message preserved.
    #[test]
    fn t_err_005_inference_converts_to_neuroncite_embed() {
        let embed_err = EmbedError::Inference("CUDA out of memory".into());
        let core_err: NeuronCiteError = embed_err.into();

        match core_err {
            NeuronCiteError::Embed(msg) => {
                assert!(
                    msg.contains("CUDA out of memory"),
                    "message must contain the inference error, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Embed, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `EmbedError::GpuInit` converts to
    /// `NeuronCiteError::Embed` with the GPU initialization message preserved.
    #[test]
    fn t_err_006_gpu_init_converts_to_neuroncite_embed() {
        let embed_err = EmbedError::GpuInit("no CUDA device found".into());
        let core_err: NeuronCiteError = embed_err.into();

        match core_err {
            NeuronCiteError::Embed(msg) => {
                assert!(
                    msg.contains("no CUDA device found"),
                    "message must contain the GPU init error, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Embed, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `EmbedError::CacheIo` converts to
    /// `NeuronCiteError::Embed` with the I/O error message preserved.
    #[test]
    fn t_err_007_cache_io_converts_to_neuroncite_embed() {
        let embed_err = EmbedError::CacheIo("permission denied".into());
        let core_err: NeuronCiteError = embed_err.into();

        match core_err {
            NeuronCiteError::Embed(msg) => {
                assert!(
                    msg.contains("permission denied"),
                    "message must contain the I/O error, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Embed, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `EmbedError::Download` converts to
    /// `NeuronCiteError::Embed` with the download failure message preserved.
    #[test]
    fn t_err_008_download_converts_to_neuroncite_embed() {
        let embed_err = EmbedError::Download("HTTP 503 from CDN".into());
        let core_err: NeuronCiteError = embed_err.into();

        match core_err {
            NeuronCiteError::Embed(msg) => {
                assert!(
                    msg.contains("HTTP 503 from CDN"),
                    "message must contain the download error, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Embed, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `EmbedError::Backend` converts to
    /// `NeuronCiteError::Embed` with the backend-specific message preserved.
    #[test]
    fn t_err_009_backend_converts_to_neuroncite_embed() {
        let embed_err = EmbedError::Backend("unsupported operator".into());
        let core_err: NeuronCiteError = embed_err.into();

        match core_err {
            NeuronCiteError::Embed(msg) => {
                assert!(
                    msg.contains("unsupported operator"),
                    "message must contain the backend error, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Embed, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that the subsystem() method on the converted NeuronCiteError
    /// returns "embed" for all EmbedError variants. This confirms all variants
    /// route to the Embed arm of NeuronCiteError, not to any other subsystem.
    #[test]
    fn t_err_010_all_variants_map_to_embed_subsystem() {
        let variants: Vec<EmbedError> = vec![
            EmbedError::ModelNotFound("x".into()),
            EmbedError::ModelLoad("x".into()),
            EmbedError::ChecksumMismatch {
                expected: "a".into(),
                actual: "b".into(),
            },
            EmbedError::Tokenizer("x".into()),
            EmbedError::Inference("x".into()),
            EmbedError::GpuInit("x".into()),
            EmbedError::CacheIo("x".into()),
            EmbedError::Download("x".into()),
            EmbedError::Backend("x".into()),
        ];

        for variant in variants {
            let display = variant.to_string();
            let core_err: NeuronCiteError = variant.into();
            assert_eq!(
                core_err.subsystem(),
                "embed",
                "EmbedError variant with display '{display}' must map to 'embed' subsystem"
            );
        }
    }

    // -----------------------------------------------------------------------
    // From<std::io::Error> for EmbedError conversion tests
    //
    // The From impl converts std::io::Error into EmbedError::CacheIo by
    // formatting the I/O error message as a string. This is used when model
    // cache file operations fail.
    // -----------------------------------------------------------------------

    /// Validates that `std::io::Error` converts to `EmbedError::CacheIo` with
    /// the I/O error message preserved. This tests the From<std::io::Error>
    /// impl which is used by the ? operator in cache file read/write paths.
    #[test]
    fn t_err_011_io_error_converts_to_cache_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "model.onnx not found");
        let embed_err: EmbedError = io_err.into();

        match embed_err {
            EmbedError::CacheIo(msg) => {
                assert!(
                    msg.contains("model.onnx not found"),
                    "CacheIo message must contain the original I/O error text, got: {msg}"
                );
            }
            other => panic!("expected EmbedError::CacheIo, got: {other:?}"),
        }
    }

    /// Validates that the full conversion chain std::io::Error -> EmbedError ->
    /// NeuronCiteError produces NeuronCiteError::Embed with the original I/O
    /// message embedded in the string. This tests composability of the two
    /// From impls in sequence.
    #[test]
    fn t_err_012_io_to_embed_to_neuroncite_chain() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "read-only fs");
        let embed_err: EmbedError = io_err.into();
        let core_err: NeuronCiteError = embed_err.into();

        match core_err {
            NeuronCiteError::Embed(msg) => {
                assert!(
                    msg.contains("read-only fs"),
                    "chained conversion must preserve the original I/O message, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Embed, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    // -----------------------------------------------------------------------
    // Display trait tests
    //
    // The thiserror #[error(...)] attributes define the Display output for
    // each variant. These tests verify the exact format strings.
    // -----------------------------------------------------------------------

    /// Validates that every EmbedError variant produces a non-empty Display
    /// string that starts with the expected prefix defined in the #[error(...)]
    /// attribute.
    #[test]
    fn t_err_013_display_format_all_variants() {
        let cases: Vec<(EmbedError, &str)> = vec![
            (
                EmbedError::ModelNotFound("test".into()),
                "model not found: test",
            ),
            (
                EmbedError::ModelLoad("test".into()),
                "model load error: test",
            ),
            (
                EmbedError::ChecksumMismatch {
                    expected: "aa".into(),
                    actual: "bb".into(),
                },
                "checksum mismatch: expected aa, got bb",
            ),
            (
                EmbedError::Tokenizer("test".into()),
                "tokenizer error: test",
            ),
            (
                EmbedError::Inference("test".into()),
                "inference error: test",
            ),
            (
                EmbedError::GpuInit("test".into()),
                "GPU initialization error: test",
            ),
            (EmbedError::CacheIo("test".into()), "cache I/O error: test"),
            (EmbedError::Download("test".into()), "download error: test"),
            (EmbedError::Backend("test".into()), "backend error: test"),
        ];

        for (err, expected) in cases {
            assert_eq!(
                err.to_string(),
                expected,
                "Display format mismatch for EmbedError variant"
            );
        }
    }
}
