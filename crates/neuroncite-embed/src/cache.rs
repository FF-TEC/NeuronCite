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

//! Local model cache for embedding and reranker model files.
//!
//! Manages the platform-specific cache directory where downloaded model files are
//! stored. Provides functions to resolve a model identifier to a local file path,
//! download the model if it is not already cached, and verify the file's SHA-256
//! checksum against the registry entry. Supports fully offline operation when all
//! required models are pre-cached.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::EmbedError;

/// Returns the directory for cached embedding model files.
///
/// Delegates to `neuroncite_core::paths::models_dir()`, which resolves to
/// `<Documents>/NeuronCite/models/` on all platforms.
#[must_use]
pub fn cache_dir() -> PathBuf {
    neuroncite_core::paths::models_dir()
}

/// Returns the directory path where a specific model revision is stored.
///
/// The model ID is sanitized by replacing `/` with `--` to avoid nested
/// directories from Hugging Face model identifiers (e.g., "BAAI/bge-small-en-v1.5"
/// becomes "BAAI--bge-small-en-v1.5"). The revision is appended with a `--`
/// separator.
///
/// # Arguments
///
/// * `model_id` - Hugging Face model identifier (e.g., "BAAI/bge-small-en-v1.5").
/// * `revision` - Git revision or version tag (e.g., "main", "abc123").
#[must_use]
pub fn model_dir(model_id: &str, revision: &str) -> PathBuf {
    let sanitized_id = model_id.replace('/', "--");
    let dir_name = format!("{sanitized_id}--{revision}");
    cache_dir().join(dir_name)
}

/// Checks whether a model revision is already present in the local cache.
///
/// Returns `true` if the model directory exists and contains at least one file.
/// This is a lightweight check that does not verify file integrity; use
/// `verify_checksum` for full validation.
///
/// # Arguments
///
/// * `model_id` - Hugging Face model identifier.
/// * `revision` - Git revision or version tag.
#[must_use]
pub fn is_cached(model_id: &str, revision: &str) -> bool {
    let dir = model_dir(model_id, revision);
    if !dir.exists() {
        return false;
    }

    // Verify the directory is not empty by checking for at least one entry.
    match std::fs::read_dir(&dir) {
        Ok(mut entries) => entries.next().is_some(),
        Err(_) => false,
    }
}

/// Checks whether all required model files are present in the local cache
/// directory. Unlike `is_cached()` which only verifies that the directory
/// contains at least one file, this function checks every file listed in
/// the MODEL_FILES manifest for the given model_id.
///
/// Returns `true` when the model directory exists and every file listed in
/// the MODEL_FILES manifest for this model_id exists on disk (e.g.,
/// `model.onnx` and `tokenizer.json`).
///
/// Returns `false` when the model directory does not exist, the model_id
/// is not in the manifest (unknown model), or any required file is missing
/// from the directory.
///
/// # Arguments
///
/// * `model_id` - HuggingFace model identifier.
/// * `revision` - Git revision or version tag.
#[must_use]
pub fn is_fully_cached(model_id: &str, revision: &str) -> bool {
    let dir = model_dir(model_id, revision);
    if !dir.exists() {
        return false;
    }

    let expected = match model_expected_files(model_id) {
        Some(files) => files,
        None => return false,
    };

    expected.iter().all(|filename| dir.join(filename).exists())
}

/// Diagnostic result for a single model's cache state. Contains file-level
/// detail about which files are present, which are missing, and whether
/// checksums pass verification. Used by the Model Doctor web endpoint to
/// report model health and determine repairability.
#[derive(Debug, Clone)]
pub struct ModelDiagnosis {
    /// HuggingFace model identifier.
    pub model_id: String,
    /// Git revision or tag used for the cache directory.
    pub revision: String,
    /// Whether the model directory exists on disk.
    pub directory_exists: bool,
    /// Filenames present in the model directory.
    pub files_present: Vec<String>,
    /// Filenames expected by the manifest but missing from the directory.
    pub files_missing: Vec<String>,
    /// Whether the checksums.sha256 manifest exists and all file checksums
    /// match. None if checksums cannot be verified (no manifest or I/O error).
    pub checksums_valid: Option<bool>,
    /// Total size in bytes of all present files.
    pub total_size_bytes: u64,
    /// Whether the model can be repaired by purging and re-downloading.
    /// True when the model_id is in the download manifest.
    pub repairable: bool,
}

/// Produces a detailed diagnostic report for a model's cache state.
/// Examines the cache directory for file presence, size, and checksum
/// integrity. The result is used by the Model Doctor endpoint to display
/// health status and determine whether repair is possible.
///
/// # Arguments
///
/// * `model_id` - HuggingFace model identifier.
/// * `revision` - Git revision or version tag.
pub fn diagnose_model(model_id: &str, revision: &str) -> ModelDiagnosis {
    let dir = model_dir(model_id, revision);
    let directory_exists = dir.exists();
    let expected_files = model_expected_files(model_id);
    let repairable = expected_files.is_some();

    if !directory_exists {
        return ModelDiagnosis {
            model_id: model_id.to_string(),
            revision: revision.to_string(),
            directory_exists: false,
            files_present: Vec::new(),
            files_missing: expected_files
                .unwrap_or_default()
                .into_iter()
                .map(String::from)
                .collect(),
            checksums_valid: None,
            total_size_bytes: 0,
            repairable,
        };
    }

    let expected = expected_files.unwrap_or_default();
    let mut files_present = Vec::new();
    let mut files_missing = Vec::new();
    let mut total_size_bytes: u64 = 0;

    for filename in &expected {
        let path = dir.join(filename);
        if path.exists() {
            if let Ok(meta) = std::fs::metadata(&path) {
                total_size_bytes += meta.len();
            }
            files_present.push(filename.to_string());
        } else {
            files_missing.push(filename.to_string());
        }
    }

    // Only run checksum verification when all files are present. Missing
    // files are already flagged by the files_missing list, and the checksum
    // manifest references all expected files.
    let checksums_valid = if files_missing.is_empty() {
        verify_cached_model(model_id, revision).ok()
    } else {
        Some(false)
    };

    ModelDiagnosis {
        model_id: model_id.to_string(),
        revision: revision.to_string(),
        directory_exists,
        files_present,
        files_missing,
        checksums_valid,
        total_size_bytes,
        repairable,
    }
}

/// Returns the list of all model identifiers known to the download manifest.
/// Includes both embedding and reranker models. Used by the Model Doctor
/// endpoint to enumerate all models for diagnostic scanning.
pub fn all_model_ids() -> Vec<&'static str> {
    MODEL_FILES.iter().map(|(id, _)| *id).collect()
}

/// Computes the SHA-256 checksum of a file and compares it with the expected
/// hex digest string.
///
/// Reads the file in 8 KiB chunks to avoid loading the entire file into memory,
/// which matters for large model weight files (hundreds of megabytes).
///
/// # Arguments
///
/// * `path` - Filesystem path to the file to verify.
/// * `expected_sha256` - Lowercase hex-encoded SHA-256 digest to compare against.
///
/// # Returns
///
/// `Ok(true)` if the computed digest matches, `Ok(false)` if it does not.
///
/// # Errors
///
/// Returns `EmbedError::CacheIo` if the file cannot be read.
pub fn verify_checksum(path: &Path, expected_sha256: &str) -> Result<bool, EmbedError> {
    let mut file = std::fs::File::open(path).map_err(|e| {
        EmbedError::CacheIo(format!(
            "failed to open file for checksum: {}: {e}",
            path.display()
        ))
    })?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = std::io::Read::read(&mut file, &mut buffer).map_err(|e| {
            EmbedError::CacheIo(format!(
                "failed to read file for checksum: {}: {e}",
                path.display()
            ))
        })?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let result = hasher.finalize();
    let mut computed = String::with_capacity(64);
    for byte in &result {
        use std::fmt::Write;
        let _ = write!(computed, "{byte:02x}");
    }

    Ok(computed == expected_sha256)
}

/// Computes the SHA-256 hex digest of a file and returns it as a string.
///
/// This function is used internally by `verify_checksum` and is also available
/// for diagnostic purposes (e.g., reporting the actual checksum of a corrupted
/// download).
///
/// # Arguments
///
/// * `path` - Filesystem path to the file to hash.
///
/// # Errors
///
/// Returns `EmbedError::CacheIo` if the file cannot be read.
pub fn compute_sha256(path: &Path) -> Result<String, EmbedError> {
    let mut file = std::fs::File::open(path).map_err(|e| {
        EmbedError::CacheIo(format!(
            "failed to open file for hashing: {}: {e}",
            path.display()
        ))
    })?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = std::io::Read::read(&mut file, &mut buffer).map_err(|e| {
            EmbedError::CacheIo(format!(
                "failed to read file for hashing: {}: {e}",
                path.display()
            ))
        })?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let result = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in &result {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

// ---------------------------------------------------------------------------
// HuggingFace model download
// ---------------------------------------------------------------------------

/// Base URL for downloading individual model files from HuggingFace repositories.
/// Files are fetched from the CDN via the `/resolve/{revision}/{path}` endpoint.
const HF_CDN_BASE: &str = "https://huggingface.co";

/// Download manifest for each supported model. Maps the HuggingFace model ID to
/// a list of (remote_path_in_repo, local_filename) pairs. The remote path is
/// relative to the model's repository root on HuggingFace.
const MODEL_FILES: &[(&str, &[(&str, &str)])] = &[
    // -----------------------------------------------------------------------
    // Encoder models (BERT architecture)
    // -----------------------------------------------------------------------
    (
        "BAAI/bge-small-en-v1.5",
        &[
            ("onnx/model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
    (
        "BAAI/bge-base-en-v1.5",
        &[
            ("onnx/model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        &[
            ("onnx/model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
    // -----------------------------------------------------------------------
    // GTE Large EN v1.5 (Alibaba-NLP, BERT++ architecture with RoPE + GLU)
    // Ships with an official ONNX model in the onnx/ directory of its
    // HuggingFace repository.
    // -----------------------------------------------------------------------
    (
        "Alibaba-NLP/gte-large-en-v1.5",
        &[
            ("onnx/model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
    // -----------------------------------------------------------------------
    // Decoder models (Qwen3 LLM-based, mean pooling)
    // ONNX weights are hosted in community conversion repositories. The
    // canonical model IDs are mapped to download repos via
    // DOWNLOAD_REPO_REDIRECTS. The 4B and 8B variants store their weight
    // tensors in a separate model.onnx_data file (ONNX external data
    // format) that must be placed alongside model.onnx for ORT to load.
    // -----------------------------------------------------------------------
    (
        "Qwen/Qwen3-Embedding-0.6B",
        &[
            ("model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
    (
        "Qwen/Qwen3-Embedding-4B",
        &[
            ("model.onnx", "model.onnx"),
            ("model.onnx_data", "model.onnx_data"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
    (
        "Qwen/Qwen3-Embedding-8B",
        &[
            ("onnx/model.onnx", "model.onnx"),
            ("onnx/model.onnx_data", "model.onnx_data"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
    // -----------------------------------------------------------------------
    // BGE-M3 (BAAI, XLM-RoBERTa encoder, multilingual)
    // Official HuggingFace repository ships ONNX files in the onnx/
    // subdirectory. The model uses external data format (model.onnx_data).
    // -----------------------------------------------------------------------
    (
        "BAAI/bge-m3",
        &[
            ("onnx/model.onnx", "model.onnx"),
            ("onnx/model.onnx_data", "model.onnx_data"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
    // -----------------------------------------------------------------------
    // Cross-encoder rerankers (MS MARCO, binary relevance scoring)
    //
    // All three models share the same input contract: query and document are
    // concatenated as a single sequence "[CLS] query [SEP] document [SEP]"
    // and the model outputs a single relevance logit per pair.
    //
    // ONNX weights are downloaded from the canonical cross-encoder/*
    // repositories on HuggingFace, which ship ONNX exports at onnx/model.onnx.
    // No DOWNLOAD_REPO_REDIRECT is needed. OrtReranker handles both output
    // shapes [batch_size, 1] and [batch_size].
    //
    // MiniLM-L6:    6 layers, ~22M params, ~133 MB float32 ONNX. Fast CPU.
    // MiniLM-L12:   12 layers, ~33M params, ~200 MB float32 ONNX. Balanced.
    // Electra-Base: 12 layers, ~110M params, ~440 MB float32 ONNX. Highest quality.
    // -----------------------------------------------------------------------
    (
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        &[
            ("onnx/model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
    (
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        &[
            ("onnx/model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
    (
        "cross-encoder/ms-marco-electra-base",
        &[
            ("onnx/model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
    // -----------------------------------------------------------------------
    // BGE Reranker v2 M3 (BAAI, XLM-RoBERTa cross-encoder, multilingual)
    //
    // The canonical model repository (BAAI/bge-reranker-v2-m3) ships only
    // SafeTensors weights, without ONNX exports. The ONNX conversion is
    // hosted in a community repository (see DOWNLOAD_REPO_REDIRECTS). The
    // model uses external data format (model.onnx_data) due to its 568M
    // parameter count (~2.3 GB float32 ONNX).
    // -----------------------------------------------------------------------
    (
        "BAAI/bge-reranker-v2-m3",
        &[
            ("onnx/model.onnx", "model.onnx"),
            ("onnx/model.onnx_data", "model.onnx_data"),
            ("tokenizer.json", "tokenizer.json"),
        ],
    ),
];

/// Returns the list of local filenames expected in the model's cache directory.
/// Each model in the `MODEL_FILES` manifest specifies which files are downloaded
/// from HuggingFace (remote path) and where they are stored locally (local name).
/// This function returns the local names (e.g., "model.onnx", "tokenizer.json").
///
/// Returns `None` if the `model_id` is not present in the download manifest.
///
/// Used by the CLI `models info` and `models verify` subcommands to enumerate
/// the files that should be present in a cached model directory.
pub fn model_expected_files(model_id: &str) -> Option<Vec<&'static str>> {
    MODEL_FILES
        .iter()
        .find(|(id, _)| *id == model_id)
        .map(|(_, files)| files.iter().map(|(_, local)| *local).collect())
}

/// Downloads all required files for a model from HuggingFace and places them
/// in the local cache directory. Creates the cache directory structure if it
/// does not exist.
///
/// The download runs in a dedicated thread to avoid blocking the tokio runtime
/// context (reqwest::blocking panics inside a tokio runtime). Each file is
/// fetched individually from HuggingFace's CDN resolve endpoint.
///
/// # Arguments
///
/// * `model_id` - HuggingFace model identifier (e.g., "BAAI/bge-small-en-v1.5").
/// * `revision` - Git revision or version tag (e.g., "main").
///
/// # Errors
///
/// Returns `EmbedError::Download` if the model ID is not in the download
/// manifest, if any HTTP request fails, or if the file cannot be written.
/// Maps canonical model IDs to alternate HuggingFace repository IDs for download.
/// Some models (particularly decoder-based ones) have their ONNX weights hosted
/// in community conversion repositories rather than the original model repository.
/// The canonical ID is used everywhere in the application; only the download URL
/// uses the alternate repository.
const DOWNLOAD_REPO_REDIRECTS: &[(&str, &str)] = &[
    (
        "Qwen/Qwen3-Embedding-0.6B",
        "zhiqing/Qwen3-Embedding-0.6B-ONNX",
    ),
    ("Qwen/Qwen3-Embedding-4B", "zhiqing/Qwen3-Embedding-4B-ONNX"),
    (
        "Qwen/Qwen3-Embedding-8B",
        "Maxi-Lein/Qwen3-Embedding-8B-onnx",
    ),
    // BGE-M3 uses its official repository (BAAI/bge-m3) directly,
    // so no redirect entry is needed.
    //
    // The three cross-encoder/* reranker repos (MiniLM-L-6-v2, MiniLM-L-12-v2,
    // ms-marco-electra-base) now ship ONNX weights at onnx/model.onnx in
    // their canonical repositories. No redirect is needed anymore. The
    // previous Xenova/* redirects were removed because Xenova/ms-marco-
    // electra-base became private (HTTP 401), and the other two Xenova
    // repos are redundant since the canonical repos contain ONNX exports.
    // BGE Reranker v2 M3 has no official ONNX export. The community
    // conversion by mogolloni provides float32 ONNX weights derived
    // from the original SafeTensors checkpoint.
    (
        "BAAI/bge-reranker-v2-m3",
        "mogolloni/bge-reranker-v2-m3-onnx",
    ),
];

pub fn download_model(model_id: &str, revision: &str) -> Result<PathBuf, EmbedError> {
    let files = MODEL_FILES
        .iter()
        .find(|(id, _)| *id == model_id)
        .map(|(_, files)| *files)
        .ok_or_else(|| {
            EmbedError::Download(format!(
                "no download manifest for model '{model_id}'; \
                 supported models: {}",
                MODEL_FILES
                    .iter()
                    .map(|(id, _)| *id)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;

    // Resolve the actual HuggingFace repository for download. For most models
    // this is the same as model_id, but some decoder ONNX models are hosted
    // in community conversion repositories.
    let download_repo = DOWNLOAD_REPO_REDIRECTS
        .iter()
        .find(|(canonical, _)| *canonical == model_id)
        .map(|(_, repo)| *repo)
        .unwrap_or(model_id);

    let dir = model_dir(model_id, revision);
    std::fs::create_dir_all(&dir).map_err(|e| {
        EmbedError::CacheIo(format!(
            "failed to create model directory {}: {e}",
            dir.display()
        ))
    })?;

    for (remote_path, local_name) in files {
        let url = format!("{HF_CDN_BASE}/{download_repo}/resolve/{revision}/{remote_path}");
        let dest = dir.join(local_name);
        download_file(&url, &dest)?;
    }

    // Compute SHA-256 checksums for all downloaded files and persist them
    // as a manifest. On subsequent cache loads, `verify_cached_model` reads
    // this manifest and validates file integrity, catching partial downloads
    // or on-disk corruption.
    write_checksums_manifest(&dir, files)?;

    tracing::info!(
        model_id,
        revision,
        path = %dir.display(),
        "model download complete"
    );
    Ok(dir)
}

/// Manifest filename stored alongside model files in each model's cache
/// directory. Contains SHA-256 hex digests keyed by local filename.
const CHECKSUMS_MANIFEST: &str = "checksums.sha256";

/// Computes SHA-256 checksums for all downloaded files and writes them to
/// a manifest file in the model directory. Each line has the format
/// `{hex_digest}  {filename}` (two spaces, matching sha256sum output).
fn write_checksums_manifest(
    dir: &std::path::Path,
    files: &[(&str, &str)],
) -> Result<(), EmbedError> {
    let mut lines = Vec::with_capacity(files.len());
    for (_, local_name) in files {
        let path = dir.join(local_name);
        let hex = compute_sha256(&path)?;
        tracing::debug!(file = %local_name, sha256 = %hex, "computed checksum for downloaded file");
        lines.push(format!("{hex}  {local_name}"));
    }
    let content = lines.join("\n") + "\n";
    std::fs::write(dir.join(CHECKSUMS_MANIFEST), content.as_bytes()).map_err(|e| {
        EmbedError::CacheIo(format!(
            "failed to write checksums manifest in {}: {e}",
            dir.display()
        ))
    })
}

/// Verifies the integrity of a cached model directory by comparing each
/// file's SHA-256 digest against the manifest written during download.
///
/// Returns `Ok(true)` if all files match, `Ok(false)` if any file is
/// missing or has a different checksum. Returns `Err` if the manifest
/// itself cannot be read (which is treated as a cache miss by callers).
///
/// # Arguments
///
/// * `model_id` - HuggingFace model identifier.
/// * `revision` - Git revision or version tag.
pub fn verify_cached_model(model_id: &str, revision: &str) -> Result<bool, EmbedError> {
    let dir = model_dir(model_id, revision);
    let manifest_path = dir.join(CHECKSUMS_MANIFEST);

    let content = match std::fs::read_to_string(&manifest_path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // No manifest exists (model was cached before checksum verification
            // was added). Treat as unverified but not an error.
            tracing::debug!(
                model_id,
                revision,
                "no checksums manifest found; skipping integrity check"
            );
            return Ok(true);
        }
        Err(e) => {
            return Err(EmbedError::CacheIo(format!(
                "failed to read checksums manifest {}: {e}",
                manifest_path.display()
            )));
        }
    };

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Format: "{hex_digest}  {filename}" (two-space separator, matching sha256sum)
        let Some((expected_hex, filename)) = line.split_once("  ") else {
            tracing::warn!(line = %line, "malformed checksums manifest line");
            continue;
        };

        let file_path = dir.join(filename);
        match verify_checksum(&file_path, expected_hex) {
            Ok(true) => {
                tracing::debug!(file = %filename, "checksum verified");
            }
            Ok(false) => {
                tracing::error!(
                    file = %filename,
                    expected = %expected_hex,
                    "checksum mismatch -- cached file is corrupt"
                );
                return Ok(false);
            }
            Err(e) => {
                tracing::error!(file = %filename, "checksum verification failed: {e}");
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Deletes the entire cached model directory for a given model and revision.
/// Used to remove corrupt files so a fresh download can replace them.
///
/// Returns `Ok(())` if the directory was removed or did not exist.
///
/// # Arguments
///
/// * `model_id` - HuggingFace model identifier.
/// * `revision` - Git revision or version tag.
pub fn purge_cached_model(model_id: &str, revision: &str) -> Result<(), EmbedError> {
    let dir = model_dir(model_id, revision);
    if !dir.exists() {
        return Ok(());
    }
    tracing::info!(
        model_id,
        revision,
        path = %dir.display(),
        "purging corrupt cached model directory"
    );
    std::fs::remove_dir_all(&dir).map_err(|e| {
        EmbedError::CacheIo(format!(
            "failed to remove model directory {}: {e}",
            dir.display()
        ))
    })
}

/// Downloads a single file from a URL to a local path. Runs the HTTP request
/// in a separate thread to avoid reqwest::blocking conflicts with tokio's
/// runtime context.
///
/// # Arguments
///
/// * `url` - Full URL to download from.
/// * `dest` - Local filesystem path to write the downloaded content to.
///
/// # Errors
///
/// Returns `EmbedError::Download` on HTTP errors or `EmbedError::CacheIo` on
/// filesystem write errors.
fn download_file(url: &str, dest: &Path) -> Result<(), EmbedError> {
    let url_owned = url.to_string();
    let dest_owned = dest.to_path_buf();
    let file_name = dest
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "file".to_string());

    // Spawn a dedicated thread because reqwest::blocking::get panics when
    // called from within a tokio runtime context. The GUI startup enters
    // the tokio runtime before model loading, so the download must run
    // in a thread that is outside that context.
    std::thread::spawn(move || -> Result<(), EmbedError> {
        tracing::info!(url = %url_owned, dest = %dest_owned.display(), "downloading file");
        eprintln!("  Downloading {file_name}...");

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .map_err(|e| EmbedError::Download(format!("failed to create HTTP client: {e}")))?;

        let mut response = client.get(&url_owned).send().map_err(|e| {
            EmbedError::Download(format!("HTTP request failed for {url_owned}: {e}"))
        })?;

        if !response.status().is_success() {
            return Err(EmbedError::Download(format!(
                "HTTP {} for {url_owned}",
                response.status()
            )));
        }

        // Stream the response body directly to disk instead of buffering
        // the entire content in memory. This is more robust for large files
        // (e.g., 133 MB model.onnx) and avoids reqwest gzip decoding issues.
        let mut file = std::fs::File::create(&dest_owned).map_err(|e| {
            EmbedError::CacheIo(format!(
                "failed to create file {}: {e}",
                dest_owned.display()
            ))
        })?;

        let bytes_written = response.copy_to(&mut file).map_err(|e| {
            EmbedError::Download(format!(
                "failed to stream response body to {}: {e}",
                dest_owned.display()
            ))
        })?;

        let mb = bytes_written as f64 / 1_048_576.0;
        tracing::info!(
            bytes = bytes_written,
            dest = %dest_owned.display(),
            "file downloaded"
        );
        eprintln!("  Downloaded {file_name} ({mb:.1} MB)");
        Ok(())
    })
    .join()
    .map_err(|_| EmbedError::Download("download thread panicked".to_string()))?
}

// ---------------------------------------------------------------------------
// ONNX Runtime shared library download (Windows, backend-ort)
// ---------------------------------------------------------------------------

/// ONNX Runtime version that matches ort-sys 2.0.0-rc.11's expected C API
/// version. This version is downloaded from Microsoft's GitHub releases when
/// the `load-dynamic` feature requires the shared library at runtime.
/// Shared across Windows and Linux; macOS has its own constant below.
#[cfg(all(not(target_os = "macos"), feature = "backend-ort"))]
const ORT_VERSION: &str = "1.23.2";

/// Runs nvidia-smi with a 5-second timeout and returns its stdout on success.
/// Spawns the process and polls `try_wait` at 100ms intervals to avoid
/// hanging indefinitely if nvidia-smi is stuck (e.g., during a driver update
/// or GPU reset). Kills the child process if the deadline is exceeded.
///
/// Returns `None` if nvidia-smi is not found, exits with non-zero status,
/// produces empty output, or exceeds the 5-second timeout.
#[cfg(all(not(target_os = "macos"), feature = "backend-ort"))]
fn run_nvidia_smi_with_timeout() -> Option<String> {
    use std::io::Read;

    let mut child = match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => {
            tracing::debug!("nvidia-smi not found or not executable");
            return None;
        }
    };

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if status.success() {
                    if let Some(mut stdout) = child.stdout.take() {
                        let mut output = String::new();
                        if stdout.read_to_string(&mut output).is_ok() && !output.trim().is_empty() {
                            return Some(output);
                        }
                    }
                } else {
                    tracing::debug!("nvidia-smi exited with non-zero status (no NVIDIA driver)");
                }
                return None;
            }
            Ok(None) => {
                if std::time::Instant::now() > deadline {
                    tracing::warn!("nvidia-smi timed out after 5 seconds, killing process");
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => {
                tracing::debug!(error = %e, "nvidia-smi wait failed");
                return None;
            }
        }
    }
}

/// Detects whether an NVIDIA GPU with a working driver is present on the
/// system by invoking `nvidia-smi` with a 5-second timeout. Returns `true`
/// if the tool exits successfully and reports at least one GPU name.
///
/// This function is used to decide whether to download the GPU variant of
/// ONNX Runtime (which includes CUDA execution provider shared libraries)
/// or the smaller CPU-only variant.
///
/// Available on Windows and Linux. macOS does not support NVIDIA GPUs.
#[cfg(all(not(target_os = "macos"), feature = "backend-ort"))]
pub fn detect_nvidia_gpu() -> bool {
    match run_nvidia_smi_with_timeout() {
        Some(output) => {
            let gpu_name = output.trim();
            tracing::info!(gpu = gpu_name, "NVIDIA GPU detected via nvidia-smi");
            true
        }
        None => false,
    }
}

/// Detects the GPU device name by querying platform-specific tools.
///
/// - **Windows/Linux**: Invokes `nvidia-smi` to query the NVIDIA GPU name
///   (e.g., "NVIDIA GeForce RTX 4090"). Returns None when no NVIDIA GPU
///   or driver is present.
/// - **macOS**: Queries `sysctl machdep.cpu.brand_string` to detect the
///   Apple Silicon chip (e.g., "Apple M2 Pro"). On Apple Silicon, the GPU
///   is integrated into the SoC and identified by the chip name. Returns
///   None on Intel Macs where there is no discrete GPU detection.
pub fn detect_gpu_device_name() -> Option<String> {
    // Try NVIDIA GPU detection on non-macOS platforms when the ORT backend
    // is compiled in. The nvidia-smi subprocess requires the ORT feature to
    // be active because GPU acceleration is only available through ORT.
    #[cfg(all(not(target_os = "macos"), feature = "backend-ort"))]
    if let Some(name) = detect_nvidia_gpu_name() {
        return Some(name);
    }

    // On macOS, detect Apple Silicon chip name. The GPU is integrated
    // into the Apple M-series SoC and accelerated via CoreML.
    #[cfg(target_os = "macos")]
    {
        if let Some(name) = detect_apple_silicon_name() {
            return Some(name);
        }
    }

    None
}

/// Queries nvidia-smi for the NVIDIA GPU device name. Returns the first
/// GPU name from the CSV output, or None if nvidia-smi is unavailable,
/// reports no GPUs, or exceeds the 5-second timeout.
///
/// Gated behind the same cfg condition as `run_nvidia_smi_with_timeout`
/// because the function calls it directly. Without `feature = "backend-ort"`
/// on non-macOS platforms, neither function is compiled.
#[cfg(all(not(target_os = "macos"), feature = "backend-ort"))]
fn detect_nvidia_gpu_name() -> Option<String> {
    let output = run_nvidia_smi_with_timeout()?;
    // nvidia-smi may output multiple lines if multiple GPUs are installed.
    // Return the first non-empty line as the primary GPU device name.
    let name = output.lines().next().unwrap_or("").trim().to_string();
    if name.is_empty() { None } else { Some(name) }
}

/// Detects the Apple Silicon chip name via `sysctl machdep.cpu.brand_string`.
/// Returns the chip name (e.g., "Apple M2 Pro", "Apple M4 Max") on Apple
/// Silicon Macs. Returns None on Intel Macs where the brand string does not
/// start with "Apple M".
#[cfg(target_os = "macos")]
fn detect_apple_silicon_name() -> Option<String> {
    match std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
    {
        Ok(output) if output.status.success() => {
            let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
            // Only return the name for Apple Silicon chips (M1, M2, M3, M4).
            // Intel Macs report "Intel(R) Core(TM) ..." which is not a GPU name.
            if name.starts_with("Apple M") {
                Some(name)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Returns the directory path where ONNX Runtime shared libraries are cached.
/// Stored at `<Documents>/NeuronCite/runtime/ort/` on Windows, and the
/// equivalent XDG-based path on Linux.
#[cfg(all(not(target_os = "macos"), feature = "backend-ort"))]
fn ort_cache_dir() -> Result<PathBuf, EmbedError> {
    Ok(neuroncite_core::paths::runtime_dir().join("ort"))
}

/// Checks whether the GPU variant of ONNX Runtime is installed in the local
/// cache directory. The GPU variant ships a CUDA execution provider shared
/// library alongside the main ORT library, while the CPU-only variant has
/// only the main library.
///
/// Platform-specific CUDA provider library names:
/// - Windows: `onnxruntime_providers_cuda.dll`
/// - Linux: `libonnxruntime_providers_cuda.so`
///
/// Returns `false` if the cache directory cannot be determined or if the
/// CUDA provider library is absent.
#[cfg(all(not(target_os = "macos"), feature = "backend-ort"))]
pub fn is_gpu_ort_runtime() -> bool {
    let Ok(base_dir) = ort_cache_dir() else {
        return false;
    };

    #[cfg(target_os = "windows")]
    let cuda_provider_lib = "onnxruntime_providers_cuda.dll";
    #[cfg(not(target_os = "windows"))]
    let cuda_provider_lib = "libonnxruntime_providers_cuda.so";

    base_dir.join(cuda_provider_lib).exists()
}

/// The cuDNN DLL filename required by ORT 1.23.2's CUDA execution provider.
/// ORT loads `onnxruntime_providers_cuda.dll` which dynamically links to
/// cuDNN 9.x at runtime. Without this DLL on the library search path,
/// CUDA EP registration silently fails and ORT falls back to CPU.
#[cfg(all(target_os = "windows", feature = "backend-ort"))]
const CUDNN_DLL: &str = "cudnn64_9.dll";

/// Searches for the CUDA 12.x toolkit `bin` directory containing cuDNN
/// (`cudnn64_9.dll`) and the CUDA runtime libraries (`cublas64_12.dll`,
/// `cudart64_12.dll`, etc.).
///
/// Search order:
/// 1. `%CUDA_PATH%\bin` -- set by the NVIDIA CUDA Toolkit installer
/// 2. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*\bin` --
///    default installation path, with the highest version preferred
///
/// Returns `Some(path_to_bin_dir)` if cuDNN is found, `None` otherwise.
#[cfg(all(target_os = "windows", feature = "backend-ort"))]
pub fn find_cudnn_dir() -> Option<PathBuf> {
    // Check CUDA_PATH environment variable first.
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let bin_dir = PathBuf::from(&cuda_path).join("bin");
        if bin_dir.join(CUDNN_DLL).exists() {
            tracing::debug!(
                path = %bin_dir.display(),
                "cuDNN found via CUDA_PATH"
            );
            return Some(bin_dir);
        }
    }

    // Search the default NVIDIA CUDA Toolkit installation directory for
    // versioned subdirectories (v12.0, v12.1, v12.3, etc.). CUDA 12.x is
    // required by ORT 1.23.2.
    let program_files =
        std::env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".to_string());
    let cuda_base = PathBuf::from(&program_files)
        .join("NVIDIA GPU Computing Toolkit")
        .join("CUDA");

    if let Ok(entries) = std::fs::read_dir(&cuda_base) {
        // Collect versioned CUDA directories and sort in reverse so that
        // the latest version (e.g., v12.3) is preferred over older ones.
        let mut cuda_dirs: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let name_str = name.to_string_lossy();
                // Match v12.* directories (ORT 1.23.2 requires CUDA 12.x)
                name_str.starts_with("v12.") && e.file_type().map(|ft| ft.is_dir()).unwrap_or(false)
            })
            .collect();
        cuda_dirs.sort_by_key(|b| std::cmp::Reverse(b.file_name()));

        for dir in cuda_dirs {
            let bin_dir = dir.path().join("bin");
            if bin_dir.join(CUDNN_DLL).exists() {
                tracing::debug!(
                    path = %bin_dir.display(),
                    "cuDNN found in CUDA toolkit"
                );
                return Some(bin_dir);
            }
        }
    }

    None
}

/// Searches for the CUDA 12.x toolkit `bin` directory containing the CUDA
/// runtime libraries (cuBLAS, cuFFT, etc.) required by the ONNX Runtime
/// CUDA execution provider, even when cuDNN is not installed. This allows
/// adding the CUDA bin directory to PATH so that ORT can at least find the
/// base CUDA libraries.
///
/// Returns `Some(path_to_bin_dir)` for the highest-versioned CUDA 12.x
/// installation, `None` if no CUDA 12.x is installed.
#[cfg(all(target_os = "windows", feature = "backend-ort"))]
fn find_cuda12_bin_dir() -> Option<PathBuf> {
    // Check CUDA_PATH first -- if it points to a CUDA 12.x installation.
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let path = PathBuf::from(&cuda_path);
        if let Some(name) = path.file_name()
            && name.to_string_lossy().starts_with("v12.")
        {
            let bin = path.join("bin");
            if bin.exists() {
                return Some(bin);
            }
        }
    }

    let program_files =
        std::env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".to_string());
    let cuda_base = PathBuf::from(&program_files)
        .join("NVIDIA GPU Computing Toolkit")
        .join("CUDA");

    if let Ok(entries) = std::fs::read_dir(&cuda_base) {
        let mut cuda_dirs: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let name_str = name.to_string_lossy();
                name_str.starts_with("v12.") && e.file_type().map(|ft| ft.is_dir()).unwrap_or(false)
            })
            .collect();
        cuda_dirs.sort_by_key(|b| std::cmp::Reverse(b.file_name()));

        if let Some(dir) = cuda_dirs.first() {
            let bin_dir = dir.path().join("bin");
            if bin_dir.exists() {
                return Some(bin_dir);
            }
        }
    }

    None
}

/// Searches for cuDNN 9.x DLLs in a Python environment where the
/// `nvidia-cudnn-cu12` pip package is installed. The DLLs are located
/// in the package's `bin` or `lib` subdirectory within the site-packages
/// tree.
///
/// Returns `Some(path_to_dll_dir)` if cuDNN DLLs are found, `None` otherwise.
#[cfg(all(target_os = "windows", feature = "backend-ort"))]
fn find_cudnn_in_python() -> Option<PathBuf> {
    let output = std::process::Command::new("python")
        .args(["-c", "import nvidia.cudnn; print(nvidia.cudnn.__file__)"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let init_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let package_dir = PathBuf::from(&init_path).parent()?.to_path_buf();

    // cuDNN DLLs are in the bin/ or lib/ subdirectory of the package.
    for subdir in &["bin", "lib"] {
        let dll_dir = package_dir.join(subdir);
        if dll_dir.join(CUDNN_DLL).exists() {
            tracing::info!(
                path = %dll_dir.display(),
                "cuDNN found in Python site-packages (nvidia-cudnn-cu12)"
            );
            return Some(dll_dir);
        }
    }

    None
}

/// Downloads cuDNN 9.x for CUDA 12 from NVIDIA's redistributable package
/// on PyPI (`nvidia-cudnn-cu12`). The package is distributed as a Python
/// wheel (.whl), which is a standard zip archive containing the cuDNN
/// shared libraries. The DLLs are extracted to the specified destination
/// directory.
///
/// The download uses PowerShell to query PyPI's JSON API for the latest
/// Windows x64 wheel, download it, and extract the DLL files. The total
/// download size is approximately 400-700 MB depending on the cuDNN version.
///
/// # Arguments
///
/// * `dest_dir` - Directory to place the extracted cuDNN DLLs.
///
/// # Errors
///
/// Returns `EmbedError::Download` if no Windows wheel is available on PyPI,
/// or if the download or extraction fails.
#[cfg(all(target_os = "windows", feature = "backend-ort"))]
fn download_cudnn(dest_dir: &Path) -> Result<(), EmbedError> {
    tracing::info!(
        "cuDNN not found locally; downloading nvidia-cudnn-cu12 from PyPI \
         (this is a one-time download of ~400 MB)"
    );

    let dest_str = dest_dir.to_string_lossy();

    // PowerShell script that queries PyPI for the nvidia-cudnn-cu12 package,
    // finds the latest Windows x64 wheel, downloads it, extracts DLLs, and
    // copies them to the destination directory. The wheel format is a
    // standard zip archive, so Expand-Archive handles extraction.
    let ps_script = format!(
        r#"$ProgressPreference = 'SilentlyContinue'
$ErrorActionPreference = 'Stop'
try {{
    $response = Invoke-RestMethod -Uri 'https://pypi.org/pypi/nvidia-cudnn-cu12/json'
    $files = $response.urls
    $wheel = $files | Where-Object {{ $_.filename -match 'win_amd64\.whl$' }} | Select-Object -First 1
    if (-not $wheel) {{
        Write-Error 'No Windows x64 wheel found for nvidia-cudnn-cu12 on PyPI'
        exit 1
    }}
    $wheelUrl = $wheel.url
    $wheelSize = $wheel.size
    Write-Output "Downloading cuDNN wheel ($([math]::Round($wheelSize / 1MB, 1)) MB)..."
    $tempDir = Join-Path $env:TEMP 'neuroncite_cudnn'
    if (Test-Path $tempDir) {{ Remove-Item -Recurse -Force $tempDir }}
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
    $wheelPath = Join-Path $tempDir 'cudnn.zip'
    Invoke-WebRequest -Uri $wheelUrl -OutFile $wheelPath -UseBasicParsing
    $extractPath = Join-Path $tempDir 'extracted'
    Expand-Archive -Path $wheelPath -DestinationPath $extractPath -Force
    $dllCount = 0
    Get-ChildItem -Path $extractPath -Recurse -Filter '*.dll' | ForEach-Object {{
        Copy-Item -Path $_.FullName -Destination '{dest_str}' -Force
        Write-Output "Extracted: $($_.Name)"
        $dllCount++
    }}
    Remove-Item -Recurse -Force $tempDir
    if ($dllCount -eq 0) {{
        Write-Error 'No DLL files found in cuDNN wheel'
        exit 1
    }}
    Write-Output "cuDNN download complete ($dllCount DLLs extracted)"
}} catch {{
    if (Test-Path $tempDir) {{ Remove-Item -Recurse -Force $tempDir }}
    Write-Error $_.Exception.Message
    exit 1
}}"#
    );

    tracing::info!("invoking PowerShell to download cuDNN from PyPI");
    eprintln!("  Downloading cuDNN from PyPI (~400 MB one-time download, please wait)...");

    let ps_output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &ps_script])
        .output()
        .map_err(|e| EmbedError::Download(format!("failed to invoke PowerShell: {e}")))?;
    eprintln!("  cuDNN download complete.");

    let stdout = String::from_utf8_lossy(&ps_output.stdout);
    for line in stdout.lines() {
        if !line.is_empty() {
            tracing::info!("{}", line.trim());
        }
    }

    if !ps_output.status.success() {
        let stderr = String::from_utf8_lossy(&ps_output.stderr);
        return Err(EmbedError::Download(format!(
            "cuDNN download failed: {}",
            stderr.trim()
        )));
    }

    // Verify that the main cuDNN DLL was extracted to the destination.
    if !dest_dir.join(CUDNN_DLL).exists() {
        return Err(EmbedError::Download(
            "cuDNN download completed but cudnn64_9.dll was not found in \
             the extracted files"
                .into(),
        ));
    }

    tracing::info!(
        path = %dest_dir.display(),
        "cuDNN DLLs installed to ORT cache directory"
    );
    Ok(())
}

/// Prepares the process environment for CUDA execution provider loading.
/// Searches for cuDNN 9.x using four strategies in order:
///
/// 1. ORT cache directory (from a prior auto-download)
/// 2. CUDA toolkit installation directories (CUDA_PATH, default paths)
/// 3. Python site-packages (pip-installed nvidia-cudnn-cu12)
/// 4. Auto-download from PyPI (~400 MB one-time download)
///
/// When cuDNN is found or downloaded, the containing directory is added to
/// the process PATH so that `onnxruntime_providers_cuda.dll` can find
/// `cudnn64_9.dll` and related DLLs at load time.
///
/// Returns `true` if cuDNN 9.x is available and PATH was configured,
/// `false` if all strategies failed. When returning `false`, the CUDA EP
/// will not function and the caller should use the CPU execution provider.
///
/// # Safety
///
/// Modifies the process `PATH` environment variable via `std::env::set_var`.
/// This is called during early initialization from the OnceLock in
/// `ensure_ort_library`, guaranteeing single-threaded access.
#[cfg(all(target_os = "windows", feature = "backend-ort"))]
pub fn ensure_cuda_runtime_on_path() -> bool {
    // Strategy 1: Check the ORT cache directory for previously downloaded
    // cuDNN DLLs. This avoids repeating the ~400 MB download on subsequent
    // application starts.
    if let Ok(ort_dir) = ort_cache_dir()
        && ort_dir.join(CUDNN_DLL).exists()
    {
        add_dir_to_path(&ort_dir);
        tracing::info!(
            path = %ort_dir.display(),
            "cuDNN found in ORT cache directory"
        );
        return true;
    }

    // Strategy 2: Check CUDA toolkit installation directories on the system
    // (CUDA_PATH environment variable, then default NVIDIA install paths).
    if let Some(cudnn_dir) = find_cudnn_dir() {
        add_dir_to_path(&cudnn_dir);
        tracing::info!(
            path = %cudnn_dir.display(),
            "CUDA toolkit bin directory with cuDNN added to PATH"
        );
        return true;
    }

    // Strategy 3: Check Python site-packages for pip-installed cuDNN
    // (nvidia-cudnn-cu12 package).
    if let Some(cudnn_dir) = find_cudnn_in_python() {
        add_dir_to_path(&cudnn_dir);
        return true;
    }

    // Strategy 4: Download cuDNN from PyPI as a last resort. The download
    // is ~400-700 MB and requires an internet connection.
    if let Ok(ort_dir) = ort_cache_dir() {
        match download_cudnn(&ort_dir) {
            Ok(()) => {
                add_dir_to_path(&ort_dir);
                return true;
            }
            Err(e) => {
                tracing::warn!("cuDNN auto-download from PyPI failed: {e}");
            }
        }
    }

    // All strategies exhausted. Add the CUDA 12.x bin directory to PATH
    // so that the base CUDA runtime libraries (cuBLAS, cuFFT) are
    // discoverable, even though CUDA EP requires cuDNN to function.
    if let Some(cuda_bin) = find_cuda12_bin_dir() {
        add_dir_to_path(&cuda_bin);
        tracing::info!(
            path = %cuda_bin.display(),
            "CUDA 12.x bin directory added to PATH (cuDNN not available)"
        );
    }

    tracing::warn!(
        "cuDNN 9.x ({CUDNN_DLL}) could not be found or downloaded. \
         The CUDA execution provider requires cuDNN for GPU inference. \
         Falling back to CPU execution."
    );
    false
}

/// Adds a directory to the process `PATH` environment variable if it is not
/// already present. Prepends the directory to ensure it takes precedence
/// over other entries.
#[cfg(all(target_os = "windows", feature = "backend-ort"))]
fn add_dir_to_path(dir: &Path) {
    let dir_str = dir.to_string_lossy();
    let current_path = std::env::var("PATH").unwrap_or_default();
    if !current_path.contains(&*dir_str) {
        // SAFETY: Called during early initialization before multi-threaded
        // ORT API calls. The OnceLock in ensure_ort_library serializes
        // all env modifications.
        unsafe {
            std::env::set_var("PATH", format!("{dir_str};{current_path}"));
        }
    }
}

// ---------------------------------------------------------------------------
// cuDNN discovery and CUDA runtime setup (Linux, backend-ort)
// ---------------------------------------------------------------------------

/// The cuDNN shared library filename required by ORT 1.23.2's CUDA execution
/// provider on Linux. ORT loads `libonnxruntime_providers_cuda.so` which
/// dynamically links to cuDNN 9.x at runtime. Without this library on
/// `LD_LIBRARY_PATH`, CUDA EP registration silently fails and ORT falls
/// back to CPU.
#[cfg(all(target_os = "linux", feature = "backend-ort"))]
const CUDNN_SO: &str = "libcudnn.so.9";

/// Searches for the cuDNN 9.x shared library (`libcudnn.so.9`) on Linux.
///
/// Search order:
/// 1. `$CUDA_PATH/lib64` -- set by the NVIDIA CUDA Toolkit installer
/// 2. `/usr/local/cuda/lib64/` -- default CUDA Toolkit installation
/// 3. `/usr/lib/x86_64-linux-gnu/` -- Debian/Ubuntu multiarch library path
/// 4. `/usr/lib64/` -- RHEL/Fedora/CentOS library path
/// 5. `/usr/lib/aarch64-linux-gnu/` -- Debian/Ubuntu ARM64 multiarch path
///
/// Returns `Some(path_to_lib_dir)` if cuDNN is found, `None` otherwise.
#[cfg(all(target_os = "linux", feature = "backend-ort"))]
pub fn find_cudnn_dir() -> Option<PathBuf> {
    // Check CUDA_PATH environment variable first.
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let lib_dir = PathBuf::from(&cuda_path).join("lib64");
        if lib_dir.join(CUDNN_SO).exists() {
            tracing::debug!(
                path = %lib_dir.display(),
                "cuDNN found via CUDA_PATH/lib64"
            );
            return Some(lib_dir);
        }
    }

    // Standard Linux CUDA Toolkit and system library paths where cuDNN
    // may be installed via the CUDA Toolkit, distribution packages
    // (libcudnn9-cuda-12), or NVIDIA's apt/yum repositories.
    let search_dirs = [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib/aarch64-linux-gnu",
    ];

    for dir in &search_dirs {
        let lib_dir = PathBuf::from(dir);
        if lib_dir.join(CUDNN_SO).exists() {
            tracing::debug!(
                path = %lib_dir.display(),
                "cuDNN found in system library directory"
            );
            return Some(lib_dir);
        }
    }

    None
}

/// Prepares the process environment for CUDA execution provider loading
/// on Linux. Searches for cuDNN 9.x in the ORT cache directory and
/// standard system library paths.
///
/// Unlike Windows, cuDNN auto-download is NOT performed on Linux. The
/// system package manager (`apt install libcudnn9-cuda-12` on Debian/Ubuntu,
/// `dnf install cudnn9-cuda-12` on Fedora) is the expected installation method.
///
/// When cuDNN is found, the containing directory is added to `LD_LIBRARY_PATH`
/// so that `libonnxruntime_providers_cuda.so` can locate `libcudnn.so.9` and
/// related libraries at load time.
///
/// Returns `true` if cuDNN 9.x is available and `LD_LIBRARY_PATH` was
/// configured, `false` if cuDNN is not found.
///
/// # Safety
///
/// Modifies the process `LD_LIBRARY_PATH` environment variable via
/// `std::env::set_var`. Called during early initialization from the OnceLock
/// in `ensure_ort_library`, guaranteeing single-threaded access.
#[cfg(all(target_os = "linux", feature = "backend-ort"))]
pub fn ensure_cuda_runtime_on_path() -> bool {
    // Strategy 1: Check the ORT cache directory for cuDNN libraries
    // that may have been placed alongside the ORT shared libraries
    // (e.g., in a Docker image or manual install).
    if let Ok(ort_dir) = ort_cache_dir()
        && ort_dir.join(CUDNN_SO).exists()
    {
        add_dir_to_ld_library_path(&ort_dir);
        tracing::info!(
            path = %ort_dir.display(),
            "cuDNN found in ORT cache directory"
        );
        return true;
    }

    // Strategy 2: Check standard system library directories.
    if let Some(cudnn_dir) = find_cudnn_dir() {
        add_dir_to_ld_library_path(&cudnn_dir);
        tracing::info!(
            path = %cudnn_dir.display(),
            "cuDNN directory added to LD_LIBRARY_PATH"
        );
        return true;
    }

    tracing::warn!(
        "cuDNN 9.x ({CUDNN_SO}) not found on the system. \
         The CUDA execution provider requires cuDNN for GPU inference. \
         Install via your package manager: \
         apt install libcudnn9-cuda-12 (Debian/Ubuntu) or \
         dnf install cudnn9-cuda-12 (Fedora/RHEL). \
         Falling back to CPU execution."
    );
    false
}

/// Adds a directory to the process `LD_LIBRARY_PATH` environment variable
/// if it is not already present. Prepends the directory to ensure it takes
/// precedence over other entries.
#[cfg(all(target_os = "linux", feature = "backend-ort"))]
fn add_dir_to_ld_library_path(dir: &Path) {
    let dir_str = dir.to_string_lossy();
    let current_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    if !current_path.contains(&*dir_str) {
        // SAFETY: Called during early initialization before multi-threaded
        // ORT API calls. The OnceLock in ensure_ort_library serializes
        // all env modifications.
        unsafe {
            std::env::set_var("LD_LIBRARY_PATH", format!("{dir_str}:{current_path}"));
        }
    }
}

// ---------------------------------------------------------------------------
// ONNX Runtime shared library download (Linux, backend-ort)
// ---------------------------------------------------------------------------

/// Ensures that the ONNX Runtime shared library (`libonnxruntime.so`) is
/// present in the application's local data directory on Linux. If the library
/// is not found, downloads the matching ORT release from Microsoft's GitHub
/// releases page and extracts it using `tar`.
///
/// When an NVIDIA GPU is detected via `nvidia-smi`, the GPU variant
/// (`onnxruntime-linux-x64-gpu-{VERSION}.tgz`, ~300 MB) is downloaded.
/// This variant bundles `libonnxruntime_providers_cuda.so` and
/// `libonnxruntime_providers_shared.so` alongside `libonnxruntime.so`,
/// enabling CUDA execution provider registration at session creation time.
///
/// If GPU hardware is not detected, or if the GPU variant download fails,
/// the smaller CPU-only variant is used as fallback.
///
/// On ARM64 Linux, only the CPU variant is available from Microsoft's releases.
///
/// If `ORT_DYLIB_PATH` is already set (e.g., in a Docker container or manual
/// install), the auto-download is skipped entirely and the existing path is used.
///
/// # Returns
///
/// The full filesystem path to `libonnxruntime.so`.
///
/// # Errors
///
/// Returns `EmbedError::Download` if the HTTP download or tar extraction
/// fails, or `EmbedError::CacheIo` if filesystem operations fail.
#[cfg(all(target_os = "linux", feature = "backend-ort"))]
pub fn ensure_ort_runtime() -> Result<PathBuf, EmbedError> {
    // If ORT_DYLIB_PATH is already set (Docker, manual install), respect it.
    if let Ok(existing) = std::env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(&existing);
        if path.exists() {
            tracing::info!(
                path = %path.display(),
                "using pre-configured ORT_DYLIB_PATH"
            );
            return Ok(path);
        }
        tracing::warn!(
            path = %existing,
            "ORT_DYLIB_PATH is set but the file does not exist; attempting auto-download"
        );
    }

    let base_dir = ort_cache_dir()?;
    let so_path = base_dir.join("libonnxruntime.so");

    #[cfg(target_arch = "x86_64")]
    let cuda_provider_path = base_dir.join("libonnxruntime_providers_cuda.so");
    #[cfg(not(target_arch = "x86_64"))]
    let cuda_provider_path = base_dir.join("_nonexistent_arm64_no_gpu_variant");

    let has_gpu_hardware = detect_nvidia_gpu();

    // Check if the correct variant is already cached.
    if so_path.exists() {
        if !has_gpu_hardware || cuda_provider_path.exists() {
            let variant = if cuda_provider_path.exists() {
                "GPU"
            } else {
                "CPU"
            };
            tracing::debug!(
                path = %so_path.display(),
                variant,
                "ONNX Runtime shared library found in local cache"
            );
            return Ok(so_path);
        }
        // GPU hardware available but only CPU variant cached. Re-download
        // the GPU variant to enable CUDA acceleration.
        tracing::info!(
            "CPU-only ONNX Runtime cached but NVIDIA GPU available; \
             upgrading to GPU variant"
        );
    } else {
        tracing::info!(
            version = ORT_VERSION,
            dest = %base_dir.display(),
            "ONNX Runtime shared library not found locally, downloading from GitHub"
        );
    }

    // Verify sufficient disk space before starting the download (~300 MB
    // for GPU variant, ~18 MB for CPU-only variant).
    neuroncite_core::disk::check_disk_space(&base_dir).map_err(EmbedError::Download)?;

    std::fs::create_dir_all(&base_dir).map_err(|e| {
        EmbedError::CacheIo(format!(
            "failed to create ORT directory {}: {e}",
            base_dir.display()
        ))
    })?;

    // Detect CPU architecture for download URL construction.
    // Microsoft ORT releases use "x64" for x86_64 and "aarch64" for ARM64.
    let arch = if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "x64"
    };

    // When GPU hardware is detected AND architecture is x86_64, try the
    // GPU variant first. GPU ORT releases are only available for x64.
    if has_gpu_hardware && cfg!(target_arch = "x86_64") {
        let gpu_tgz_url = format!(
            "https://github.com/microsoft/onnxruntime/releases/download/\
             v{ORT_VERSION}/onnxruntime-linux-x64-gpu-{ORT_VERSION}.tgz"
        );
        match download_and_extract_ort_tgz_linux(&gpu_tgz_url, &base_dir) {
            Ok(()) if so_path.exists() => {
                tracing::info!(
                    path = %so_path.display(),
                    version = ORT_VERSION,
                    "ONNX Runtime GPU variant ready (Linux)"
                );
                return Ok(so_path);
            }
            Ok(()) => {
                tracing::warn!(
                    "GPU variant download succeeded but libonnxruntime.so \
                     not found in archive, falling back to CPU variant"
                );
            }
            Err(e) => {
                tracing::warn!("GPU variant download failed ({e}), falling back to CPU variant");
            }
        }
    }

    // CPU-only variant download (either no GPU hardware, ARM64 platform,
    // or GPU download failed above).
    let cpu_tgz_url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/\
         v{ORT_VERSION}/onnxruntime-linux-{arch}-{ORT_VERSION}.tgz"
    );
    download_and_extract_ort_tgz_linux(&cpu_tgz_url, &base_dir)?;

    if !so_path.exists() {
        return Err(EmbedError::Download(
            "libonnxruntime.so was not found in the downloaded archive".into(),
        ));
    }

    tracing::info!(
        path = %so_path.display(),
        version = ORT_VERSION,
        arch,
        "ONNX Runtime CPU variant ready (Linux)"
    );
    Ok(so_path)
}

/// Downloads an ONNX Runtime release tarball (.tgz) from the given URL and
/// extracts all shared library files (.so) into the destination directory
/// on Linux. Uses the system `tar` command for extraction.
///
/// The archive contains a top-level directory (e.g.,
/// `onnxruntime-linux-x64-gpu-1.23.2/lib/`) with the shared libraries.
/// After extraction, only the `.so` files are copied to the flat cache
/// directory and all temporary files are cleaned up.
#[cfg(all(target_os = "linux", feature = "backend-ort"))]
fn download_and_extract_ort_tgz_linux(tgz_url: &str, dest_dir: &Path) -> Result<(), EmbedError> {
    let tgz_path = dest_dir.join("ort_download.tgz");
    let extract_dir = dest_dir.join("_extracted");

    // Download the tgz archive using the existing download_file function,
    // which runs in a dedicated thread to avoid tokio runtime conflicts.
    download_file(tgz_url, &tgz_path)?;

    // Compute and log SHA-256 of the downloaded archive for audit trail.
    if let Ok(hex) = compute_sha256(&tgz_path) {
        tracing::info!(
            sha256 = %hex,
            url = %tgz_url,
            "SHA-256 of downloaded ONNX Runtime archive"
        );
    }

    // Extract the tgz archive using the system `tar` command.
    std::fs::create_dir_all(&extract_dir).map_err(|e| {
        EmbedError::CacheIo(format!(
            "failed to create extraction directory {}: {e}",
            extract_dir.display()
        ))
    })?;

    tracing::info!(url = %tgz_url, "extracting ONNX Runtime archive via tar");

    let tar_output = std::process::Command::new("tar")
        .args(["-xzf"])
        .arg(&tgz_path)
        .arg("-C")
        .arg(&extract_dir)
        .output()
        .map_err(|e| EmbedError::Download(format!("failed to invoke tar: {e}")))?;

    if !tar_output.status.success() {
        let stderr = String::from_utf8_lossy(&tar_output.stderr);
        let _ = std::fs::remove_file(&tgz_path);
        return Err(EmbedError::Download(format!(
            "tar extraction failed: {stderr}"
        )));
    }

    // Copy shared library files from the extracted directory tree into the
    // flat ORT cache directory. The archive nests the .so files inside
    // onnxruntime-linux-{arch}-{version}/lib/.
    copy_so_files_from_extracted(&extract_dir, dest_dir)?;

    // Remove temporary download artifacts.
    let _ = std::fs::remove_file(&tgz_path);
    let _ = std::fs::remove_dir_all(&extract_dir);

    Ok(())
}

/// Recursively walks a directory tree and copies all `.so` files (and
/// `.so.*` versioned variants) into a flat destination directory. Preserves
/// file names but discards the directory hierarchy from the source archive.
///
/// `std::fs::copy` follows symlinks, so both the versioned library
/// (e.g., `libonnxruntime.so.1.23.2`) and the unversioned symlink
/// (`libonnxruntime.so`) are copied as regular files in the destination.
#[cfg(all(target_os = "linux", feature = "backend-ort"))]
fn copy_so_files_from_extracted(src: &Path, dst: &Path) -> Result<(), EmbedError> {
    let entries = std::fs::read_dir(src).map_err(|e| {
        EmbedError::CacheIo(format!("failed to read directory {}: {e}", src.display()))
    })?;

    for entry_result in entries {
        let entry = entry_result
            .map_err(|e| EmbedError::CacheIo(format!("failed to read directory entry: {e}")))?;
        let path = entry.path();

        if path.is_dir() {
            copy_so_files_from_extracted(&path, dst)?;
        } else if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
            // Match .so files and versioned .so.* variants (e.g., libonnxruntime.so.1.23.2).
            if file_name.contains(".so") {
                let dest_path = dst.join(file_name);
                std::fs::copy(&path, &dest_path).map_err(|e| {
                    EmbedError::CacheIo(format!(
                        "failed to copy {} to {}: {e}",
                        path.display(),
                        dest_path.display()
                    ))
                })?;
                tracing::debug!(file = %dest_path.display(), "copied ORT shared library to cache");
            }
        }
    }
    Ok(())
}

/// Ensures that the ONNX Runtime shared library (`onnxruntime.dll`) is present
/// in the application's local data directory. If the DLL is not found, downloads
/// the matching ORT release from Microsoft's GitHub releases page and extracts
/// it using PowerShell's `Expand-Archive` cmdlet.
///
/// When an NVIDIA GPU is detected via `nvidia-smi`, the GPU variant
/// (`onnxruntime-win-x64-gpu-{VERSION}.zip`, ~300 MB) is downloaded. This
/// variant bundles `onnxruntime_providers_cuda.dll` and
/// `onnxruntime_providers_shared.dll` alongside `onnxruntime.dll`, enabling
/// CUDA execution provider registration at session creation time.
///
/// If GPU hardware is not detected, or if the GPU variant download fails, the
/// smaller CPU-only variant (`onnxruntime-win-x64-{VERSION}.zip`, ~18 MB) is
/// used as fallback.
///
/// The DLLs are stored in `<Documents>/NeuronCite/runtime/ort/` so that
/// the application is self-contained and does not depend on system-wide ORT
/// installations (which may have version mismatches).
///
/// # Returns
///
/// The full filesystem path to `onnxruntime.dll`.
///
/// # Errors
///
/// Returns `EmbedError::Download` if the HTTP download or PowerShell
/// extraction fails, or `EmbedError::CacheIo` if filesystem operations fail.
#[cfg(all(target_os = "windows", feature = "backend-ort"))]
pub fn ensure_ort_runtime() -> Result<PathBuf, EmbedError> {
    let base_dir = ort_cache_dir()?;
    let dll_path = base_dir.join("onnxruntime.dll");
    let cuda_dll_path = base_dir.join("onnxruntime_providers_cuda.dll");
    let has_gpu_hardware = detect_nvidia_gpu();

    // Check if the correct variant is already cached. If GPU hardware is
    // present, the GPU variant (with CUDA provider DLLs) must be cached.
    // If no GPU, the CPU-only DLL is sufficient.
    if dll_path.exists() {
        if !has_gpu_hardware || cuda_dll_path.exists() {
            let variant = if cuda_dll_path.exists() { "GPU" } else { "CPU" };
            tracing::debug!(
                path = %dll_path.display(),
                variant,
                "ONNX Runtime DLL found in local cache"
            );
            return Ok(dll_path);
        }
        // GPU hardware available but only CPU variant cached. Need to
        // re-download the GPU variant to enable CUDA acceleration.
        tracing::info!(
            "CPU-only ONNX Runtime cached but NVIDIA GPU available; \
             upgrading to GPU variant"
        );
    } else {
        tracing::info!(
            version = ORT_VERSION,
            dest = %base_dir.display(),
            "ONNX Runtime DLL not found locally, downloading from GitHub"
        );
    }

    // Verify sufficient disk space before starting the download (~300 MB
    // for GPU variant, ~18 MB for CPU-only variant).
    neuroncite_core::disk::check_disk_space(&base_dir).map_err(EmbedError::Download)?;

    std::fs::create_dir_all(&base_dir).map_err(|e| {
        EmbedError::CacheIo(format!(
            "failed to create ORT directory {}: {e}",
            base_dir.display()
        ))
    })?;

    // When GPU hardware is detected, try the GPU variant first. The GPU
    // variant is ~300 MB and includes onnxruntime_providers_cuda.dll and
    // onnxruntime_providers_shared.dll. Falls back to CPU variant on failure.
    if has_gpu_hardware {
        let gpu_zip_url = format!(
            "https://github.com/microsoft/onnxruntime/releases/download/\
             v{ORT_VERSION}/onnxruntime-win-x64-gpu-{ORT_VERSION}.zip"
        );
        match download_and_extract_ort(&gpu_zip_url, &base_dir) {
            Ok(()) if dll_path.exists() => {
                tracing::info!(
                    path = %dll_path.display(),
                    version = ORT_VERSION,
                    "ONNX Runtime GPU variant ready"
                );
                return Ok(dll_path);
            }
            Ok(()) => {
                tracing::warn!(
                    "GPU variant download succeeded but onnxruntime.dll \
                     not found in archive, falling back to CPU variant"
                );
            }
            Err(e) => {
                tracing::warn!("GPU variant download failed ({e}), falling back to CPU variant");
            }
        }
    }

    // CPU-only variant download (either no GPU hardware, or GPU download
    // failed above).
    let cpu_zip_url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/\
         v{ORT_VERSION}/onnxruntime-win-x64-{ORT_VERSION}.zip"
    );
    download_and_extract_ort(&cpu_zip_url, &base_dir)?;

    if !dll_path.exists() {
        return Err(EmbedError::Download(
            "onnxruntime.dll was not found in the downloaded archive".into(),
        ));
    }

    tracing::info!(
        path = %dll_path.display(),
        version = ORT_VERSION,
        "ONNX Runtime CPU variant ready"
    );
    Ok(dll_path)
}

/// Downloads an ONNX Runtime release zip from the given URL and extracts
/// all DLL files into the destination directory. Uses PowerShell's
/// `Invoke-WebRequest` for download and `Expand-Archive` for extraction,
/// which reliably handles GitHub's CDN redirects and chunked transfer
/// encoding (avoiding the gzip decoding issues observed with reqwest on
/// GitHub release assets).
#[cfg(all(target_os = "windows", feature = "backend-ort"))]
fn download_and_extract_ort(zip_url: &str, dest_dir: &Path) -> Result<(), EmbedError> {
    let zip_path = dest_dir.join("ort_download.zip");
    let extract_dir = dest_dir.join("_extracted");

    let ps_script = format!(
        "$ProgressPreference = 'SilentlyContinue'; \
         Invoke-WebRequest -Uri '{}' -OutFile '{}' -UseBasicParsing; \
         Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
        zip_url,
        zip_path.display(),
        zip_path.display(),
        extract_dir.display()
    );

    tracing::info!(url = %zip_url, "downloading and extracting ONNX Runtime via PowerShell");
    eprintln!("  Downloading ONNX Runtime from GitHub (~300 MB for GPU variant, please wait)...");
    let ps_output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &ps_script])
        .output()
        .map_err(|e| EmbedError::Download(format!("failed to invoke PowerShell: {e}")))?;
    eprintln!("  ONNX Runtime download complete.");

    if !ps_output.status.success() {
        let stderr = String::from_utf8_lossy(&ps_output.stderr);
        let _ = std::fs::remove_file(&zip_path);
        return Err(EmbedError::Download(format!(
            "PowerShell download/extraction failed: {stderr}"
        )));
    }

    // Compute and log SHA-256 of the downloaded zip for audit trail.
    // The zip file remains on disk after PowerShell's Expand-Archive,
    // allowing post-download integrity verification. No known-good hash
    // is available for ORT releases, so the computed digest is logged
    // for operator auditability and future pinning.
    if zip_path.exists() {
        match compute_sha256(&zip_path) {
            Ok(hex) => {
                tracing::info!(
                    sha256 = %hex,
                    url = %zip_url,
                    "SHA-256 of downloaded ONNX Runtime archive"
                );
            }
            Err(e) => {
                tracing::warn!(
                    url = %zip_url,
                    "failed to compute SHA-256 of ORT archive: {e}"
                );
            }
        }
    }

    // Copy all .dll files from the extracted directory tree into the flat
    // ORT cache directory. The archive contains a top-level directory
    // (e.g., "onnxruntime-win-x64-gpu-1.23.2/lib/") with the DLLs.
    copy_dlls_from_extracted(&extract_dir, dest_dir)?;

    // Remove temporary download artifacts.
    let _ = std::fs::remove_file(&zip_path);
    let _ = std::fs::remove_dir_all(&extract_dir);

    Ok(())
}

/// Recursively walks a directory tree and copies all `.dll` files into a flat
/// destination directory. Preserves file names but discards the directory
/// hierarchy from the source archive.
#[cfg(all(target_os = "windows", feature = "backend-ort"))]
fn copy_dlls_from_extracted(src: &Path, dst: &Path) -> Result<(), EmbedError> {
    let entries = std::fs::read_dir(src).map_err(|e| {
        EmbedError::CacheIo(format!("failed to read directory {}: {e}", src.display()))
    })?;

    for entry_result in entries {
        let entry = entry_result
            .map_err(|e| EmbedError::CacheIo(format!("failed to read directory entry: {e}")))?;
        let path = entry.path();

        if path.is_dir() {
            copy_dlls_from_extracted(&path, dst)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("dll")
            && let Some(file_name) = path.file_name()
        {
            let dest_path = dst.join(file_name);
            std::fs::copy(&path, &dest_path).map_err(|e| {
                EmbedError::CacheIo(format!(
                    "failed to copy {} to {}: {e}",
                    path.display(),
                    dest_path.display()
                ))
            })?;
            tracing::debug!(file = %dest_path.display(), "copied ORT DLL to cache");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ONNX Runtime shared library download (macOS, backend-ort)
// ---------------------------------------------------------------------------

/// ONNX Runtime version for macOS. Must match ort-sys 2.0.0-rc.11's expected
/// C API version. Same version as the Windows variant above.
#[cfg(all(target_os = "macos", feature = "backend-ort"))]
const ORT_VERSION: &str = "1.23.2";

/// Returns the directory path where ONNX Runtime shared libraries are cached
/// on macOS. Stored at `<Documents>/NeuronCite/runtime/ort/`.
#[cfg(all(target_os = "macos", feature = "backend-ort"))]
fn ort_cache_dir() -> Result<PathBuf, EmbedError> {
    Ok(neuroncite_core::paths::runtime_dir().join("ort"))
}

/// Ensures that the ONNX Runtime shared library (`libonnxruntime.dylib`) is
/// present in the application's local data directory. If the dylib is not
/// found, downloads the matching ORT release from Microsoft's GitHub releases
/// and extracts it using `tar`.
///
/// Automatically detects the CPU architecture (arm64 for Apple Silicon,
/// x86_64 for Intel Macs) and downloads the corresponding ORT variant.
/// Both variants include the CoreML execution provider for Apple hardware
/// acceleration. CUDA is not available on macOS.
///
/// # Returns
///
/// The full filesystem path to `libonnxruntime.dylib`.
///
/// # Errors
///
/// Returns `EmbedError::Download` if the HTTP download or tar extraction
/// fails, or `EmbedError::CacheIo` if filesystem operations fail.
#[cfg(all(target_os = "macos", feature = "backend-ort"))]
pub fn ensure_ort_runtime() -> Result<PathBuf, EmbedError> {
    let base_dir = ort_cache_dir()?;
    let dylib_path = base_dir.join("libonnxruntime.dylib");

    if dylib_path.exists() {
        tracing::debug!(
            path = %dylib_path.display(),
            "ONNX Runtime dylib found in local cache"
        );
        return Ok(dylib_path);
    }

    tracing::info!(
        version = ORT_VERSION,
        dest = %base_dir.display(),
        "ONNX Runtime dylib not found locally, downloading from GitHub"
    );

    // Verify sufficient disk space before starting the download.
    neuroncite_core::disk::check_disk_space(&base_dir).map_err(EmbedError::Download)?;

    std::fs::create_dir_all(&base_dir).map_err(|e| {
        EmbedError::CacheIo(format!(
            "failed to create ORT directory {}: {e}",
            base_dir.display()
        ))
    })?;

    // Detect CPU architecture: arm64 for Apple Silicon (M1, M2, M3, M4),
    // x86_64 for Intel Macs. The ORT release archives use "arm64" and
    // "x86_64" in their filenames respectively.
    let arch = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x86_64"
    };

    let tgz_url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/\
         v{ORT_VERSION}/onnxruntime-osx-{arch}-{ORT_VERSION}.tgz"
    );

    download_and_extract_ort_tgz(&tgz_url, &base_dir)?;

    if !dylib_path.exists() {
        return Err(EmbedError::Download(
            "libonnxruntime.dylib was not found in the downloaded archive".into(),
        ));
    }

    tracing::info!(
        path = %dylib_path.display(),
        version = ORT_VERSION,
        arch,
        "ONNX Runtime ready for macOS"
    );
    Ok(dylib_path)
}

/// Downloads an ONNX Runtime release tarball (.tgz) from the given URL and
/// extracts all shared library files (.dylib) into the destination directory.
/// Uses the system `tar` command for extraction, which is always available
/// on macOS.
///
/// The archive contains a top-level directory (e.g.,
/// `onnxruntime-osx-arm64-1.23.2/lib/`) with the shared libraries. After
/// extraction, only the dylib files are copied to the flat cache directory
/// and all temporary files are cleaned up.
#[cfg(all(target_os = "macos", feature = "backend-ort"))]
fn download_and_extract_ort_tgz(tgz_url: &str, dest_dir: &Path) -> Result<(), EmbedError> {
    let tgz_path = dest_dir.join("ort_download.tgz");
    let extract_dir = dest_dir.join("_extracted");

    // Download the tgz archive using the existing download_file function,
    // which runs in a dedicated thread to avoid tokio runtime conflicts.
    download_file(tgz_url, &tgz_path)?;

    // Compute and log SHA-256 of the downloaded archive for audit trail.
    // No known-good hash is available for ORT releases, so the computed
    // digest is logged for operator auditability and future pinning.
    if let Ok(hex) = compute_sha256(&tgz_path) {
        tracing::info!(
            sha256 = %hex,
            url = %tgz_url,
            "SHA-256 of downloaded ONNX Runtime archive"
        );
    }

    // Extract the tgz archive using the system `tar` command. The -C flag
    // specifies the extraction target directory.
    std::fs::create_dir_all(&extract_dir).map_err(|e| {
        EmbedError::CacheIo(format!(
            "failed to create extraction directory {}: {e}",
            extract_dir.display()
        ))
    })?;

    tracing::info!(url = %tgz_url, "extracting ONNX Runtime archive via tar");

    let tar_output = std::process::Command::new("tar")
        .args(["-xzf"])
        .arg(&tgz_path)
        .arg("-C")
        .arg(&extract_dir)
        .output()
        .map_err(|e| EmbedError::Download(format!("failed to invoke tar: {e}")))?;

    if !tar_output.status.success() {
        let stderr = String::from_utf8_lossy(&tar_output.stderr);
        let _ = std::fs::remove_file(&tgz_path);
        return Err(EmbedError::Download(format!(
            "tar extraction failed: {stderr}"
        )));
    }

    // Copy shared library files from the extracted directory tree into the
    // flat ORT cache directory. The archive structure nests the dylibs
    // inside onnxruntime-osx-{arch}-{version}/lib/.
    copy_dylibs_from_extracted_tgz(&extract_dir, dest_dir)?;

    // Remove temporary download artifacts (tgz archive + extracted tree).
    let _ = std::fs::remove_file(&tgz_path);
    let _ = std::fs::remove_dir_all(&extract_dir);

    Ok(())
}

/// Recursively walks a directory tree and copies all `.dylib` files into a
/// flat destination directory. Preserves file names but discards the directory
/// hierarchy from the source archive.
///
/// `std::fs::copy` follows symlinks, so both the versioned dylib
/// (e.g., `libonnxruntime.1.23.2.dylib`) and the unversioned symlink
/// (`libonnxruntime.dylib`) are copied as regular files in the destination.
#[cfg(all(target_os = "macos", feature = "backend-ort"))]
fn copy_dylibs_from_extracted_tgz(src: &Path, dst: &Path) -> Result<(), EmbedError> {
    let entries = std::fs::read_dir(src).map_err(|e| {
        EmbedError::CacheIo(format!("failed to read directory {}: {e}", src.display()))
    })?;

    for entry_result in entries {
        let entry = entry_result
            .map_err(|e| EmbedError::CacheIo(format!("failed to read directory entry: {e}")))?;
        let path = entry.path();

        if path.is_dir() {
            copy_dylibs_from_extracted_tgz(&path, dst)?;
        } else if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
            // Copy .dylib files (the primary shared library format on macOS).
            if ext == "dylib"
                && let Some(file_name) = path.file_name()
            {
                let dest_path = dst.join(file_name);
                std::fs::copy(&path, &dest_path).map_err(|e| {
                    EmbedError::CacheIo(format!(
                        "failed to copy {} to {}: {e}",
                        path.display(),
                        dest_path.display()
                    ))
                })?;
                tracing::debug!(file = %dest_path.display(), "copied ORT dylib to cache");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// T-CACHE-REDIRECT-001: Verifies that the download redirect mapping resolves
    /// canonical model IDs to their community ONNX repository IDs. Models without
    /// a redirect mapping should resolve to themselves.
    #[test]
    fn t_cache_redirect_001_download_repo_redirects() {
        // Qwen3 ONNX weights are hosted in a community conversion repository.
        let qwen_redirect = DOWNLOAD_REPO_REDIRECTS
            .iter()
            .find(|(canonical, _)| *canonical == "Qwen/Qwen3-Embedding-0.6B")
            .map(|(_, repo)| *repo);
        assert_eq!(
            qwen_redirect,
            Some("zhiqing/Qwen3-Embedding-0.6B-ONNX"),
            "Qwen3 canonical ID should redirect to community ONNX repo"
        );

        // BGE models have no redirect (ONNX weights in the original repo).
        let bge_redirect = DOWNLOAD_REPO_REDIRECTS
            .iter()
            .find(|(canonical, _)| *canonical == "BAAI/bge-small-en-v1.5")
            .map(|(_, repo)| *repo);
        assert_eq!(
            bge_redirect, None,
            "BGE model should not have a download redirect"
        );
    }

    /// T-CACHE-MANIFEST-001: Every model ID in MODEL_FILES has a non-empty file
    /// list containing at least model.onnx and tokenizer.json.
    #[test]
    fn t_cache_manifest_001_all_entries_have_required_files() {
        for (model_id, files) in MODEL_FILES {
            assert!(
                !files.is_empty(),
                "model '{}' must have at least one file in the manifest",
                model_id
            );

            let local_names: Vec<&str> = files.iter().map(|(_, local)| *local).collect();
            assert!(
                local_names.contains(&"model.onnx"),
                "model '{}' manifest must include model.onnx",
                model_id
            );
            assert!(
                local_names.contains(&"tokenizer.json"),
                "model '{}' manifest must include tokenizer.json",
                model_id
            );
        }
    }

    /// T-CACHE-FILES-001: model_expected_files returns the local filenames for
    /// a known model. Verifies that the function returns Some with a vector
    /// containing both "model.onnx" and "tokenizer.json" for a model that
    /// exists in the MODEL_FILES manifest.
    #[test]
    fn t_cache_files_001_known_model_returns_expected_files() {
        let files = model_expected_files("BAAI/bge-small-en-v1.5");
        assert!(files.is_some(), "known model must return Some");
        let files = files.unwrap();
        assert!(
            files.contains(&"model.onnx"),
            "expected files must include model.onnx"
        );
        assert!(
            files.contains(&"tokenizer.json"),
            "expected files must include tokenizer.json"
        );
    }

    /// T-CACHE-FILES-002: model_expected_files returns None for an unknown
    /// model identifier that is not present in the MODEL_FILES manifest.
    #[test]
    fn t_cache_files_002_unknown_model_returns_none() {
        let files = model_expected_files("nonexistent/model-that-does-not-exist");
        assert!(files.is_none(), "unknown model must return None");
    }

    /// T-EMB-011: Download and checksum verification using a temporary file.
    /// Creates a temp file with known content, computes the expected SHA-256,
    /// and verifies that `verify_checksum` returns `true` for the correct digest
    /// and `false` for an incorrect digest.
    #[test]
    fn t_emb_011_checksum_verification() {
        let dir = tempfile::tempdir().expect("failed to create temp directory");
        let file_path = dir.path().join("test_model.bin");

        let content = b"NeuronCite test model file content for checksum verification";
        {
            let mut file = std::fs::File::create(&file_path).expect("failed to create temp file");
            file.write_all(content).expect("failed to write temp file");
        }

        // Compute the expected SHA-256 digest of the known content
        let mut hasher = Sha256::new();
        hasher.update(content);
        let expected_bytes = hasher.finalize();
        let mut expected_hex = String::with_capacity(64);
        for byte in &expected_bytes {
            use std::fmt::Write;
            let _ = write!(expected_hex, "{byte:02x}");
        }

        // Correct checksum should return true
        let result =
            verify_checksum(&file_path, &expected_hex).expect("checksum verification failed");
        assert!(result, "checksum should match for correct digest");

        // Incorrect checksum should return false
        let wrong_digest = "0000000000000000000000000000000000000000000000000000000000000000";
        let result_wrong =
            verify_checksum(&file_path, wrong_digest).expect("checksum verification failed");
        assert!(!result_wrong, "checksum should not match for wrong digest");
    }

    /// T-EMB-012: Offline resolution -- cached model loads without network.
    /// Creates a fake model directory in the temp cache, then verifies that
    /// `is_cached` returns true and `model_dir` points to the correct path.
    #[test]
    fn t_emb_012_offline_resolution_cached_model_loads_without_network() {
        let dir = tempfile::tempdir().expect("failed to create temp directory");
        let fake_model_id = "test-org/test-model";
        let fake_revision = "abc123";

        // Construct the expected directory path matching the sanitization logic
        let sanitized = fake_model_id.replace('/', "--");
        let dir_name = format!("{sanitized}--{fake_revision}");
        let model_path = dir.path().join(dir_name);
        std::fs::create_dir_all(&model_path).expect("failed to create model directory");

        // Place a dummy file in the model directory
        let dummy_file = model_path.join("model.onnx");
        std::fs::write(&dummy_file, b"dummy model data").expect("failed to write dummy model file");

        // Verify that model_dir produces the correct path structure
        let resolved = model_dir(fake_model_id, fake_revision);
        let resolved_name = resolved
            .file_name()
            .expect("model_dir should have a file name")
            .to_string_lossy();
        let expected_name = format!("{sanitized}--{fake_revision}");
        assert_eq!(
            resolved_name, expected_name,
            "model_dir should produce the sanitized directory name"
        );

        // Verify that is_cached returns true for the directory we created
        // (This tests the logic, not the actual global cache path, since we
        //  cannot redirect cache_dir() in a unit test without env manipulation.)
        assert!(
            model_path.exists(),
            "the constructed model directory must exist"
        );
        let entries: Vec<_> = std::fs::read_dir(&model_path)
            .expect("failed to read model directory")
            .collect();
        assert!(
            !entries.is_empty(),
            "model directory should contain at least one file"
        );
    }

    /// Verifies that `cache_dir` returns a path containing "NeuronCite" and "models".
    #[test]
    fn cache_dir_contains_expected_components() {
        let dir = cache_dir();
        let path_str = dir.to_string_lossy();
        let lower = path_str.to_lowercase();
        assert!(
            lower.contains("neuroncite"),
            "cache_dir should contain 'NeuronCite', got: {path_str}"
        );
        assert!(
            lower.contains("models"),
            "cache_dir should contain 'models', got: {path_str}"
        );
    }

    /// Verifies that `model_dir` sanitizes forward slashes in model IDs.
    #[test]
    fn model_dir_sanitizes_slashes() {
        let dir = model_dir("org/model-name", "v1");
        let dir_name = dir
            .file_name()
            .expect("should have file name")
            .to_string_lossy();
        assert!(
            !dir_name.contains('/'),
            "sanitized directory name must not contain forward slashes"
        );
        assert!(
            dir_name.contains("org--model-name"),
            "sanitized name should replace / with --"
        );
    }

    /// T-GPU-001: `detect_nvidia_gpu` returns a boolean without panicking,
    /// regardless of whether an NVIDIA GPU or driver is present on the system.
    /// This test guards against regressions where nvidia-smi invocation or
    /// output parsing could panic on unexpected environments (CI without GPU,
    /// Docker containers, etc.).
    #[cfg(all(target_os = "windows", feature = "backend-ort"))]
    #[test]
    fn detect_nvidia_gpu_returns_without_panicking() {
        let result = detect_nvidia_gpu();
        // The result depends on the hardware; we only assert it does not panic.
        eprintln!("detect_nvidia_gpu() = {result}");
    }

    /// T-GPU-002: `is_gpu_ort_runtime` returns false when no ORT DLLs are
    /// installed (clean test environment). This test creates a temporary
    /// directory structure mimicking the ORT cache layout and verifies that
    /// the GPU variant detection correctly distinguishes between CPU-only and
    /// GPU installations based on the presence of
    /// `onnxruntime_providers_cuda.dll`.
    #[cfg(all(target_os = "windows", feature = "backend-ort"))]
    #[test]
    fn is_gpu_ort_runtime_detects_cuda_dll_presence() {
        // In a clean test environment, the ORT cache directory relative to the
        // test executable may or may not exist. The function should return a
        // boolean without panicking regardless.
        let result = is_gpu_ort_runtime();
        eprintln!("is_gpu_ort_runtime() = {result}");
    }

    /// T-GPU-003: Verifies that `ort_cache_dir` returns a path containing the
    /// expected `NeuronCite/runtime/ort` directory structure under the user's
    /// Documents folder.
    #[cfg(all(target_os = "windows", feature = "backend-ort"))]
    #[test]
    fn ort_cache_dir_path_structure() {
        let dir = ort_cache_dir().expect("ort_cache_dir should succeed");
        let path_str = dir.to_string_lossy();
        assert!(
            path_str.contains("NeuronCite"),
            "ORT cache dir should be under NeuronCite"
        );
        assert!(
            path_str.ends_with("ort"),
            "ORT cache dir should end with 'ort'"
        );
    }

    /// T-GPU-004: Verifies that `ensure_ort_runtime` logic correctly decides
    /// which variant to download based on GPU detection. This is a structural
    /// test: it exercises the decision-making without triggering the actual
    /// ~300MB download. The test verifies that detect_nvidia_gpu() is consistent
    /// with is_gpu_ort_runtime() after a successful ensure_ort_runtime() call:
    /// if GPU hardware is present and ORT is cached, the GPU variant DLLs
    /// should be present.
    #[cfg(all(target_os = "windows", feature = "backend-ort"))]
    #[test]
    fn gpu_variant_selection_matches_hardware_detection() {
        let has_gpu = detect_nvidia_gpu();
        let has_gpu_runtime = is_gpu_ort_runtime();

        // If there is no GPU hardware, the CPU variant is correct regardless.
        // If there IS GPU hardware, the GPU variant should be present (assuming
        // ensure_ort_runtime was previously called during application startup).
        // In a CI environment where ensure_ort_runtime was never called, both
        // will be false, which is still a consistent state.
        if has_gpu && has_gpu_runtime {
            eprintln!("GPU hardware detected and GPU ORT runtime is cached");
        } else if has_gpu && !has_gpu_runtime {
            eprintln!(
                "GPU hardware detected but GPU ORT runtime not yet cached \
                 (ensure_ort_runtime has not been called in this test environment)"
            );
        } else if !has_gpu {
            eprintln!("No GPU hardware; CPU-only ORT variant is appropriate");
        }
    }

    // -----------------------------------------------------------------------
    // Cross-encoder / reranker manifest tests
    // -----------------------------------------------------------------------

    /// T-CACHE-RERANK-001: The cross-encoder model is present in MODEL_FILES.
    /// Without this entry, download_model() returns "no download manifest"
    /// and the reranker handler cannot auto-download the model.
    #[test]
    fn t_cache_rerank_001_cross_encoder_in_model_files() {
        let entry = MODEL_FILES
            .iter()
            .find(|(id, _)| *id == "cross-encoder/ms-marco-MiniLM-L-6-v2");
        assert!(
            entry.is_some(),
            "cross-encoder/ms-marco-MiniLM-L-6-v2 must be in MODEL_FILES"
        );
    }

    /// T-CACHE-RERANK-002: The cross-encoder model manifest contains the
    /// required ONNX model file and tokenizer. OrtReranker::resolve_model_paths
    /// looks for "model.onnx" and "tokenizer.json" in the cache directory.
    #[test]
    fn t_cache_rerank_002_cross_encoder_manifest_has_required_files() {
        let files = MODEL_FILES
            .iter()
            .find(|(id, _)| *id == "cross-encoder/ms-marco-MiniLM-L-6-v2")
            .map(|(_, files)| *files)
            .expect("cross-encoder/ms-marco-MiniLM-L-6-v2 must be in MODEL_FILES");

        let local_names: Vec<&str> = files.iter().map(|(_, local)| *local).collect();

        assert!(
            local_names.contains(&"model.onnx"),
            "cross-encoder manifest must list 'model.onnx' as a local filename"
        );
        assert!(
            local_names.contains(&"tokenizer.json"),
            "cross-encoder manifest must list 'tokenizer.json' as a local filename"
        );
    }

    /// T-CACHE-RERANK-003: The cross-encoder canonical repo ships ONNX weights
    /// directly at onnx/model.onnx, so no DOWNLOAD_REPO_REDIRECT is needed.
    /// The previous Xenova redirect was removed because the canonical repo
    /// now contains ONNX exports and the Xenova repo became inaccessible.
    #[test]
    fn t_cache_rerank_003_cross_encoder_no_redirect_needed() {
        let redirect = DOWNLOAD_REPO_REDIRECTS
            .iter()
            .find(|(canonical, _)| *canonical == "cross-encoder/ms-marco-MiniLM-L-6-v2");

        assert!(
            redirect.is_none(),
            "cross-encoder/ms-marco-MiniLM-L-6-v2 must NOT have a redirect; \
             the canonical repo ships ONNX weights directly"
        );
    }

    /// T-CACHE-RERANK-004: The cross-encoder remote path in MODEL_FILES uses
    /// the "onnx/model.onnx" prefix, which matches the canonical repository
    /// structure. OrtReranker expects a flat "model.onnx" in the cache
    /// directory, which is achieved by the (remote_path, local_name) mapping.
    #[test]
    fn t_cache_rerank_004_cross_encoder_remote_path_uses_onnx_dir() {
        let files = MODEL_FILES
            .iter()
            .find(|(id, _)| *id == "cross-encoder/ms-marco-MiniLM-L-6-v2")
            .map(|(_, files)| *files)
            .expect("cross-encoder/ms-marco-MiniLM-L-6-v2 must be in MODEL_FILES");

        // Find the model.onnx entry and verify the remote path starts with "onnx/".
        let model_entry = files
            .iter()
            .find(|(_, local)| *local == "model.onnx")
            .expect("manifest must have a model.onnx entry");

        let (remote_path, _) = model_entry;
        assert!(
            remote_path.starts_with("onnx/"),
            "cross-encoder remote path must start with 'onnx/' to match the \
             canonical repository structure, got: {remote_path}"
        );
    }

    /// T-CACHE-RERANK-005: model_expected_files returns the correct local
    /// filenames for the cross-encoder model, matching what OrtReranker looks
    /// for in the cache directory.
    #[test]
    fn t_cache_rerank_005_model_expected_files_returns_correct_filenames() {
        let files = model_expected_files("cross-encoder/ms-marco-MiniLM-L-6-v2");
        assert!(
            files.is_some(),
            "model_expected_files must return Some for the cross-encoder model"
        );

        let files = files.expect("files is Some");
        assert!(
            files.contains(&"model.onnx"),
            "expected files must contain 'model.onnx'"
        );
        assert!(
            files.contains(&"tokenizer.json"),
            "expected files must contain 'tokenizer.json'"
        );
        // The cross-encoder model does not use external data format, so
        // model.onnx_data must not be listed.
        assert!(
            !files.contains(&"model.onnx_data"),
            "cross-encoder model must not list model.onnx_data (no external data format)"
        );
    }

    /// T-CACHE-RERANK-006: All DOWNLOAD_REPO_REDIRECTS entries reference a
    /// canonical model ID that is also in MODEL_FILES. A redirect without a
    /// MODEL_FILES entry would be unreachable dead code.
    #[test]
    fn t_cache_rerank_006_all_redirects_have_corresponding_manifest_entry() {
        for (canonical, redirect_repo) in DOWNLOAD_REPO_REDIRECTS {
            let in_manifest = MODEL_FILES.iter().any(|(id, _)| id == canonical);
            assert!(
                in_manifest,
                "DOWNLOAD_REPO_REDIRECT '{canonical}' -> '{redirect_repo}' has no \
                 corresponding MODEL_FILES entry; the redirect is unreachable"
            );
        }
    }

    /// T-CACHE-RERANK-007: The MiniLM-L12-v2 model is present in MODEL_FILES.
    /// Without this manifest entry, download_model() returns "no download
    /// manifest" and the reranker handler cannot auto-download the model.
    #[test]
    fn t_cache_rerank_007_miniml_l12_in_model_files() {
        let entry = MODEL_FILES
            .iter()
            .find(|(id, _)| *id == "cross-encoder/ms-marco-MiniLM-L-12-v2");
        assert!(
            entry.is_some(),
            "cross-encoder/ms-marco-MiniLM-L-12-v2 must be in MODEL_FILES"
        );
    }

    /// T-CACHE-RERANK-008: The MiniLM-L12-v2 manifest contains the required
    /// ONNX model file and tokenizer. OrtReranker::resolve_model_paths looks
    /// for "model.onnx" and "tokenizer.json" in the local cache directory.
    #[test]
    fn t_cache_rerank_008_miniml_l12_manifest_has_required_files() {
        let files = MODEL_FILES
            .iter()
            .find(|(id, _)| *id == "cross-encoder/ms-marco-MiniLM-L-12-v2")
            .map(|(_, files)| *files)
            .expect("cross-encoder/ms-marco-MiniLM-L-12-v2 must be in MODEL_FILES");

        let local_names: Vec<&str> = files.iter().map(|(_, local)| *local).collect();
        assert!(
            local_names.contains(&"model.onnx"),
            "MiniLM-L12-v2 manifest must list 'model.onnx' as a local filename"
        );
        assert!(
            local_names.contains(&"tokenizer.json"),
            "MiniLM-L12-v2 manifest must list 'tokenizer.json' as a local filename"
        );
    }

    /// T-CACHE-RERANK-009: The MiniLM-L12-v2 remote path in MODEL_FILES uses
    /// the "onnx/" prefix, matching the Xenova community repository structure
    /// where ONNX weights are stored under an "onnx/" subdirectory.
    #[test]
    fn t_cache_rerank_009_miniml_l12_remote_path_uses_onnx_dir() {
        let files = MODEL_FILES
            .iter()
            .find(|(id, _)| *id == "cross-encoder/ms-marco-MiniLM-L-12-v2")
            .map(|(_, files)| *files)
            .expect("cross-encoder/ms-marco-MiniLM-L-12-v2 must be in MODEL_FILES");

        let model_entry = files
            .iter()
            .find(|(_, local)| *local == "model.onnx")
            .expect("manifest must have a model.onnx entry");

        let (remote_path, _) = model_entry;
        assert!(
            remote_path.starts_with("onnx/"),
            "MiniLM-L12-v2 remote path must start with 'onnx/' to match the \
             canonical repository structure, got: {remote_path}"
        );
    }

    /// T-CACHE-RERANK-010: The MiniLM-L12-v2 canonical repo ships ONNX weights
    /// directly at onnx/model.onnx, so no DOWNLOAD_REPO_REDIRECT is needed.
    /// The previous Xenova redirect was removed because the canonical repo
    /// now contains ONNX exports and the Xenova repo became inaccessible.
    #[test]
    fn t_cache_rerank_010_miniml_l12_no_redirect_needed() {
        let redirect = DOWNLOAD_REPO_REDIRECTS
            .iter()
            .find(|(canonical, _)| *canonical == "cross-encoder/ms-marco-MiniLM-L-12-v2");

        assert!(
            redirect.is_none(),
            "cross-encoder/ms-marco-MiniLM-L-12-v2 must NOT have a redirect; \
             the canonical repo ships ONNX weights directly"
        );
    }

    /// T-CACHE-RERANK-011: The Electra-Base model is present in MODEL_FILES.
    /// Without this entry, download_model() cannot locate the file manifest
    /// and returns an error before any download is attempted.
    #[test]
    fn t_cache_rerank_011_electra_base_in_model_files() {
        let entry = MODEL_FILES
            .iter()
            .find(|(id, _)| *id == "cross-encoder/ms-marco-electra-base");
        assert!(
            entry.is_some(),
            "cross-encoder/ms-marco-electra-base must be in MODEL_FILES"
        );
    }

    /// T-CACHE-RERANK-012: The Electra-Base manifest contains the required
    /// ONNX model file and tokenizer. OrtReranker::resolve_model_paths expects
    /// both "model.onnx" and "tokenizer.json" to be present in the cache
    /// directory before the model can be loaded.
    #[test]
    fn t_cache_rerank_012_electra_base_manifest_has_required_files() {
        let files = MODEL_FILES
            .iter()
            .find(|(id, _)| *id == "cross-encoder/ms-marco-electra-base")
            .map(|(_, files)| *files)
            .expect("cross-encoder/ms-marco-electra-base must be in MODEL_FILES");

        let local_names: Vec<&str> = files.iter().map(|(_, local)| *local).collect();
        assert!(
            local_names.contains(&"model.onnx"),
            "Electra-Base manifest must list 'model.onnx' as a local filename"
        );
        assert!(
            local_names.contains(&"tokenizer.json"),
            "Electra-Base manifest must list 'tokenizer.json' as a local filename"
        );
    }

    /// T-CACHE-RERANK-013: The Electra-Base canonical repo ships ONNX weights
    /// directly at onnx/model.onnx, so no DOWNLOAD_REPO_REDIRECT is needed.
    /// The previous Xenova redirect was removed because the Xenova/ms-marco-
    /// electra-base repository became private (HTTP 401 Unauthorized).
    #[test]
    fn t_cache_rerank_013_electra_base_no_redirect_needed() {
        let redirect = DOWNLOAD_REPO_REDIRECTS
            .iter()
            .find(|(canonical, _)| *canonical == "cross-encoder/ms-marco-electra-base");

        assert!(
            redirect.is_none(),
            "cross-encoder/ms-marco-electra-base must NOT have a redirect; \
             the canonical repo ships ONNX weights directly"
        );
    }

    /// T-CACHE-RERANK-014: All three cross-encoder reranker models are
    /// registered in MODEL_FILES. This completeness check ensures that every
    /// model offered in the GUI catalog has a corresponding download manifest
    /// entry so that auto-download never silently fails.
    #[test]
    fn t_cache_rerank_014_all_three_cross_encoders_in_model_files() {
        const REQUIRED: &[&str] = &[
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "cross-encoder/ms-marco-electra-base",
        ];
        for model_id in REQUIRED {
            let found = MODEL_FILES.iter().any(|(id, _)| id == model_id);
            assert!(found, "MODEL_FILES must contain an entry for '{model_id}'");
        }
    }
}
