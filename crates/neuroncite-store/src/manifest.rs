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

// Model manifest (`models.lock`) serialization and deserialization.
//
// The manifest is a TOML file stored in the `.neuroncite/` directory alongside
// the `SQLite` database. It records model identifiers, revisions, SHA-256
// checksums (both model weights and tokenizer), source URLs, licenses, and
// optional local cache paths. This file enables the embed crate to verify
// downloaded model files and allows offline operation by recording which
// models have been cached locally.
//
// The manifest uses a TOML array-of-tables format where each entry is a
// `ModelManifest` struct defined in `neuroncite-core`.

use std::fs;
use std::path::Path;

use neuroncite_core::ModelManifest;
use serde::{Deserialize, Serialize};

use crate::error::StoreError;

/// Wrapper struct for TOML serialization. TOML requires a top-level table,
/// so the array of manifests is wrapped in a `models` field.
#[derive(Debug, Serialize, Deserialize)]
struct ManifestFile {
    /// The list of model manifests stored in the lock file.
    models: Vec<ModelManifest>,
}

/// File name of the manifest lock file within the `.neuroncite/` directory.
const MANIFEST_FILENAME: &str = "models.lock";

/// Reads the `models.lock` TOML file from the given directory and returns
/// the list of `ModelManifest` entries.
///
/// # Arguments
///
/// * `dir` - Path to the `.neuroncite/` directory containing `models.lock`.
///
/// # Errors
///
/// Returns `StoreError::Io` if the file cannot be read, or
/// `StoreError::Manifest` if the TOML content is malformed.
///
/// # Returns
///
/// An empty `Vec` if the manifest file does not exist.
pub fn read_manifest(dir: &Path) -> Result<Vec<ModelManifest>, StoreError> {
    let path = dir.join(MANIFEST_FILENAME);

    if !path.exists() {
        return Ok(Vec::new());
    }

    let content = fs::read_to_string(&path).map_err(|e| StoreError::io(path.clone(), e))?;

    let manifest_file: ManifestFile =
        toml::from_str(&content).map_err(|e| StoreError::manifest(e.to_string()))?;

    Ok(manifest_file.models)
}

/// Writes the given list of `ModelManifest` entries to `models.lock` in the
/// specified directory. Overwrites any existing file.
///
/// # Arguments
///
/// * `dir` - Path to the `.neuroncite/` directory where `models.lock` is stored.
/// * `manifests` - Slice of `ModelManifest` entries to persist.
///
/// # Errors
///
/// Returns `StoreError::Manifest` if serialization fails, or
/// `StoreError::Io` if the file cannot be written.
pub fn write_manifest(dir: &Path, manifests: &[ModelManifest]) -> Result<(), StoreError> {
    let path = dir.join(MANIFEST_FILENAME);
    let tmp_path = dir.join(".models.lock.tmp");

    let manifest_file = ManifestFile {
        models: manifests.to_vec(),
    };

    let content =
        toml::to_string(&manifest_file).map_err(|e| StoreError::manifest(e.to_string()))?;

    // Write to a temporary file first, then atomically rename to the
    // final path. This prevents a half-written manifest if the process
    // crashes or loses power during the write. On the same filesystem
    // volume, rename is atomic on both POSIX and Windows (NTFS).
    fs::write(&tmp_path, &content).map_err(|e| StoreError::io(tmp_path.clone(), e))?;
    fs::rename(&tmp_path, &path).map_err(|e| StoreError::io(path, e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Verifies that reading a nonexistent manifest returns an empty list.
    #[test]
    fn read_missing_manifest_returns_empty() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let result = read_manifest(tmp.path()).expect("read_manifest failed");
        assert!(result.is_empty());
    }

    /// Verifies that a write-then-read roundtrip preserves all manifest fields.
    #[test]
    fn manifest_roundtrip() {
        let tmp = TempDir::new().expect("failed to create temp dir");

        let manifests = vec![ModelManifest {
            model_id: "BAAI/bge-small-en-v1.5".into(),
            revision: "abc123".into(),
            sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into(),
            tokenizer_sha256: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
                .into(),
            source_url: "https://huggingface.co/BAAI/bge-small-en-v1.5".into(),
            license: "MIT".into(),
            local_path: Some(PathBuf::from("/cache/models/bge-small")),
        }];

        write_manifest(tmp.path(), &manifests).expect("write_manifest failed");
        let loaded = read_manifest(tmp.path()).expect("read_manifest failed");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0], manifests[0]);
    }

    /// T-STORE-033: Atomic write leaves no temporary file. Verifies that after
    /// write_manifest completes, only the final `models.lock` file exists in
    /// the directory and the temporary `.models.lock.tmp` file has been removed
    /// by the rename operation. A leftover temp file would indicate the atomic
    /// rename failed or was skipped.
    #[test]
    fn t_store_033_atomic_write_no_temp_file() {
        let tmp = TempDir::new().expect("failed to create temp dir");

        let manifests = vec![ModelManifest {
            model_id: "test-model".into(),
            revision: "v1".into(),
            sha256: "abc123".into(),
            tokenizer_sha256: "def456".into(),
            source_url: "https://example.com".into(),
            license: "Apache-2.0".into(),
            local_path: None,
        }];

        write_manifest(tmp.path(), &manifests).expect("write_manifest failed");

        // The final manifest file must exist.
        let final_path = tmp.path().join(MANIFEST_FILENAME);
        assert!(
            final_path.exists(),
            "models.lock must exist after write_manifest"
        );

        // The temporary file must not exist after a completed write.
        // Its presence would mean the rename step was skipped.
        let tmp_path = tmp.path().join(".models.lock.tmp");
        assert!(
            !tmp_path.exists(),
            ".models.lock.tmp must not exist after a completed atomic write"
        );

        // Verify the final file is valid TOML that roundtrips correctly.
        let loaded = read_manifest(tmp.path())
            .expect("read_manifest must succeed on the atomically written file");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].model_id, "test-model");
    }
}
