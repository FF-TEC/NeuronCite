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

//! Centralized storage path resolution for the NeuronCite application.
//!
//! All persistent artifacts (databases, runtime DLLs, embedding models, exports)
//! are stored under a single root directory in the user's Documents folder:
//!
//! ```text
//! Documents/NeuronCite/
//! +-- runtime/              ONNX Runtime DLLs, pdfium, Tesseract binaries
//! |   +-- ort/
//! |   +-- pdfium/
//! |   +-- tesseract/
//! +-- models/               Downloaded ONNX embedding model files
//! |   +-- BAAI--bge-small-en-v1.5--main/
//! |   +-- ...
//! +-- indexes/              Per-directory SQLite databases and HNSW indices
//!     +-- <sanitized-path>/ One subfolder per indexed directory
//!     |   +-- index.db
//!     |   +-- hnsw_<session_id>/
//!     |   +-- exports/
//!     +-- ...
//! ```
//!
//! Path sanitization converts absolute filesystem paths to flat directory names
//! by replacing colons with underscores and path separators with double dashes.
//! Example: `D:\Papers\Finance` becomes `D_--Papers--Finance`.
//!
//! This module is the single source of truth for all storage locations. Every
//! other crate in the workspace calls these functions instead of computing
//! paths independently.

use std::path::{Path, PathBuf};

/// Returns the root directory for all NeuronCite persistent data.
///
/// Resolution order:
///
/// - **Windows**: `%APPDATA%\NeuronCite`, then `%USERPROFILE%\Documents\NeuronCite`
/// - **Linux**: `$XDG_DATA_HOME/NeuronCite`, then `$HOME/Documents/NeuronCite`
/// - **macOS**: `$HOME/Documents/NeuronCite`
///
/// Falls back to `./NeuronCite` relative to the current working directory
/// if no environment variable is set.
#[must_use]
pub fn base_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        // Prefer APPDATA (%APPDATA%\NeuronCite) which is the standard location
        // for application data on Windows and survives OneDrive folder redirection
        // that sometimes moves the Documents folder to a cloud-synced location.
        if let Ok(appdata) = std::env::var("APPDATA") {
            return PathBuf::from(appdata).join("NeuronCite");
        }
        if let Ok(profile) = std::env::var("USERPROFILE") {
            return PathBuf::from(profile).join("Documents").join("NeuronCite");
        }
    }

    #[cfg(target_os = "linux")]
    {
        // Prefer XDG_DATA_HOME ($XDG_DATA_HOME/NeuronCite) which is the
        // freedesktop.org standard for persistent application data on Linux.
        // Falls back to $HOME/Documents/NeuronCite if XDG_DATA_HOME is not set
        // (the XDG spec defaults to $HOME/.local/share, but using Documents
        // keeps backward compatibility with existing installations).
        if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
            return PathBuf::from(xdg).join("NeuronCite");
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join("Documents").join("NeuronCite");
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join("Documents").join("NeuronCite");
        }
    }

    // Fallback when no home directory environment variable is set.
    // On Unix, a more robust fallback would use getpwuid(getuid()).pw_dir,
    // but that requires libc and unsafe code, which this crate forbids via
    // #![forbid(unsafe_code)]. The relative path fallback is acceptable since
    // HOME is always set in standard Linux/macOS environments and Docker
    // containers with a proper entrypoint.
    tracing::warn!(
        "no home directory environment variable found, falling back to relative path NeuronCite/"
    );
    PathBuf::from("NeuronCite")
}

/// Returns the directory for runtime dependency binaries (ONNX Runtime DLLs,
/// pdfium shared library, Tesseract executable and tessdata).
///
/// Path: `<base_dir>/runtime/`
#[must_use]
pub fn runtime_dir() -> PathBuf {
    base_dir().join("runtime")
}

/// Returns the directory for cached ONNX embedding model files downloaded
/// from HuggingFace.
///
/// Path: `<base_dir>/models/`
#[must_use]
pub fn models_dir() -> PathBuf {
    base_dir().join("models")
}

/// Returns the parent directory containing all per-path index subdirectories.
///
/// Path: `<base_dir>/indexes/`
#[must_use]
pub fn indexes_dir() -> PathBuf {
    base_dir().join("indexes")
}

/// Returns the index directory for the GUI mode's central database. This
/// directory is independent of the current working directory, so sessions
/// persist across application restarts regardless of how the executable is
/// launched (double-click, Start Menu, terminal, etc.).
///
/// Path: `<base_dir>/indexes/_gui_/`
#[must_use]
pub fn gui_index_dir() -> PathBuf {
    indexes_dir().join("_gui_")
}

/// Returns the path of the marker file that is created once the user has
/// completed the first-run setup dialog (either by clicking "Download and
/// install everything" or "I will set it up myself"). The presence of this
/// file suppresses the welcome dialog on subsequent application launches.
///
/// Path: `<base_dir>/.setup_complete`
#[must_use]
pub fn setup_complete_path() -> PathBuf {
    base_dir().join(".setup_complete")
}

/// Returns the index directory for a specific indexed PDF directory path.
/// The input path is sanitized to produce a flat directory name.
///
/// Path: `<base_dir>/indexes/<sanitized_path>/`
///
/// # Arguments
///
/// * `pdf_directory` - The absolute path to the PDF directory that was indexed.
///   This path is sanitized by [`sanitize_path`] to produce a valid directory name.
#[must_use]
pub fn index_dir_for_path(pdf_directory: &Path) -> PathBuf {
    indexes_dir().join(sanitize_path(pdf_directory))
}

/// Resolves a directory path to its canonical absolute form. On Windows,
/// strips the `\\?\` extended-length path prefix that `std::fs::canonicalize`
/// adds, because the prefix breaks string comparisons in the session database
/// (e.g., `\\?\D:\Papers` != `D:\Papers`).
///
/// Returns `Err` when `std::fs::canonicalize` fails (e.g., the path does not
/// exist on disk). Callers that need a guaranteed `PathBuf` for non-critical
/// paths (read-only inspection, test fixtures) can apply
/// `.unwrap_or_else(|_| path.to_path_buf())`. API handlers that receive a
/// directory from an untrusted client should propagate the error as a
/// `BadRequest` response so the client receives a clear rejection rather than
/// silently operating on an unverified path.
///
/// This is the single source of truth for normalizing directory paths before
/// session creation, search, or deletion.
pub fn canonicalize_directory(path: &Path) -> Result<PathBuf, std::io::Error> {
    let canonical = std::fs::canonicalize(path)?;
    Ok(strip_extended_length_prefix_path(&canonical))
}

/// Strips the Windows `\\?\` extended-length prefix from a `PathBuf`. This
/// prefix is an internal Windows API detail added by `std::fs::canonicalize`
/// and must not appear in user-facing output or database-stored paths, where
/// it would break equality comparisons with paths stored without the prefix.
///
/// Returns the input unchanged when the prefix is absent.
fn strip_extended_length_prefix_path(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    // Strip the Windows extended-length path prefix `\\?\` that
    // std::fs::canonicalize adds on Windows. This prefix is valid for Win32
    // API calls but breaks string equality comparisons in SQLite queries where
    // paths were stored without the prefix.
    if let Some(stripped) = s.strip_prefix(r"\\?\") {
        PathBuf::from(stripped)
    } else {
        path.to_path_buf()
    }
}

/// Converts an absolute filesystem path to a flat directory name suitable
/// for use as a subdirectory under the indexes root.
///
/// Transformation rules:
/// - Drive letter colons (e.g., `D:`) are replaced with an underscore (`D_`).
/// - Both forward slashes and backslashes are replaced with `--`.
/// - Leading and trailing separators are stripped before replacement.
///
/// # Examples
///
/// - `D:\Papers\Finance` -> `D_--Papers--Finance`
/// - `/home/user/docs` -> `home--user--docs`
/// - `C:\Users\alice\OneDrive\PDFs` -> `C_--Users--alice--OneDrive--PDFs`
#[must_use]
pub fn sanitize_path(path: &Path) -> String {
    let s = path.to_string_lossy();

    // Replace drive letter colon (Windows paths like "D:\...")
    let s = s.replace(':', "_");

    // Normalize all separators to forward slashes, then replace with --
    let s = s.replace('\\', "/");

    // Strip leading and trailing slashes before splitting
    let s = s.trim_matches('/');

    // Replace remaining slashes with double dashes
    s.replace('/', "--")
}

/// Strips the Windows extended-length path prefix (`\\?\`) from a path
/// string. This prefix is an internal Windows API detail added by
/// `std::fs::canonicalize` and should not appear in user-facing output
/// (MCP tool responses, BibTeX entries, search result citations).
///
/// Returns the input unchanged on non-Windows paths or paths without the
/// prefix.
#[must_use]
pub fn strip_extended_length_prefix(path: &str) -> &str {
    path.strip_prefix(r"\\?\").unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-PATH-001: sanitize_path replaces Windows drive colon with underscore.
    #[test]
    fn t_path_001_sanitize_windows_drive() {
        let path = Path::new("D:\\Papers\\Finance");
        let result = sanitize_path(path);
        assert_eq!(result, "D_--Papers--Finance");
    }

    /// T-PATH-002: sanitize_path handles Unix absolute paths by stripping
    /// the leading slash.
    #[test]
    fn t_path_002_sanitize_unix_absolute() {
        let path = Path::new("/home/user/docs");
        let result = sanitize_path(path);
        assert_eq!(result, "home--user--docs");
    }

    /// T-PATH-003: sanitize_path handles complex Windows paths with multiple
    /// directory levels.
    #[test]
    fn t_path_003_sanitize_complex_windows() {
        let path = Path::new("C:\\Users\\alice\\OneDrive\\PDFs");
        let result = sanitize_path(path);
        assert_eq!(result, "C_--Users--alice--OneDrive--PDFs");
    }

    /// T-PATH-004: sanitize_path handles paths with trailing separator.
    #[test]
    fn t_path_004_sanitize_trailing_separator() {
        let path = Path::new("D:\\Docs\\");
        let result = sanitize_path(path);
        assert_eq!(result, "D_--Docs");
    }

    /// T-PATH-005: sanitize_path produces a non-empty string for any
    /// non-empty input path.
    #[test]
    fn t_path_005_sanitize_nonempty() {
        let path = Path::new("some_dir");
        let result = sanitize_path(path);
        assert!(!result.is_empty());
        assert_eq!(result, "some_dir");
    }

    /// T-PATH-006: base_dir returns a path containing "NeuronCite".
    #[test]
    fn t_path_006_base_dir_contains_neuroncite() {
        let dir = base_dir();
        let s = dir.to_string_lossy();
        assert!(
            s.contains("NeuronCite"),
            "base_dir must contain 'NeuronCite', got: {s}"
        );
    }

    /// T-PATH-007: runtime_dir, models_dir, indexes_dir are subdirectories of base_dir.
    #[test]
    fn t_path_007_subdirectories_under_base() {
        let base = base_dir();
        assert!(runtime_dir().starts_with(&base));
        assert!(models_dir().starts_with(&base));
        assert!(indexes_dir().starts_with(&base));
    }

    /// T-PATH-008: index_dir_for_path returns a path under indexes_dir.
    #[test]
    fn t_path_008_index_dir_under_indexes() {
        let idx = index_dir_for_path(Path::new("D:\\Papers"));
        assert!(idx.starts_with(indexes_dir()));
    }

    /// T-PATH-009: Two different input paths produce different index directories.
    #[test]
    fn t_path_009_different_paths_different_dirs() {
        let dir_a = index_dir_for_path(Path::new("D:\\Papers\\Finance"));
        let dir_b = index_dir_for_path(Path::new("D:\\Papers\\Biology"));
        assert_ne!(dir_a, dir_b);
    }

    /// T-PATH-010: sanitize_path produces a string that contains no
    /// path separators (forward slash or backslash), making it safe
    /// to use as a single directory name component.
    #[test]
    fn t_path_010_sanitize_no_separators() {
        let path = Path::new("C:\\Users\\test\\deep\\nested\\path");
        let result = sanitize_path(path);
        assert!(!result.contains('/'), "sanitized name must not contain /");
        assert!(!result.contains('\\'), "sanitized name must not contain \\");
    }

    /// T-PATH-014: strip_extended_length_prefix removes the `\\?\` prefix
    /// from a Windows extended-length path.
    #[test]
    fn t_path_014_strip_extended_length_prefix() {
        assert_eq!(
            strip_extended_length_prefix(r"\\?\D:\OneDrive\Papers"),
            r"D:\OneDrive\Papers"
        );
    }

    /// T-PATH-015: strip_extended_length_prefix returns the input unchanged
    /// when no prefix is present.
    #[test]
    fn t_path_015_strip_no_prefix() {
        assert_eq!(
            strip_extended_length_prefix(r"D:\OneDrive\Papers"),
            r"D:\OneDrive\Papers"
        );
        assert_eq!(
            strip_extended_length_prefix("/home/user/docs"),
            "/home/user/docs"
        );
    }

    /// T-PATH-011: canonicalize_directory returns Ok with a valid path for an
    /// existing directory and the result does not start with the Windows `\\?\`
    /// extended-length prefix.
    #[test]
    fn t_path_011_canonicalize_existing_directory() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let canonical = canonicalize_directory(tmp.path())
            .expect("canonicalize_directory must succeed for an existing directory");

        // The canonical path must exist and must not start with `\\?\`.
        assert!(
            canonical.is_dir(),
            "canonical path must be a directory: {}",
            canonical.display()
        );
        let s = canonical.to_string_lossy();
        assert!(
            !s.starts_with(r"\\?\"),
            "canonical path must not have \\\\?\\ prefix: {s}"
        );
    }

    /// T-PATH-012: canonicalize_directory returns Err for a path that does
    /// not exist on disk. The error propagates to callers instead of silently
    /// returning the unverified input path.
    #[test]
    fn t_path_012_canonicalize_nonexistent_returns_err() {
        let nonexistent = Path::new("/this/path/does/not/exist/at/all");
        let result = canonicalize_directory(nonexistent);
        assert!(
            result.is_err(),
            "canonicalize_directory must return Err for a non-existent path"
        );
    }

    /// T-PATH-013: canonicalize_directory produces consistent results for
    /// the same directory across multiple calls (idempotent when the path
    /// already has no `\\?\` prefix).
    #[test]
    fn t_path_013_canonicalize_idempotent() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let first =
            canonicalize_directory(tmp.path()).expect("first canonicalization must succeed");
        let second = canonicalize_directory(&first).expect("second canonicalization must succeed");
        assert_eq!(first, second, "canonicalization must be idempotent");
    }

    /// T-PATH-016: Stripping the `\\?\` prefix from both sides of a path
    /// comparison produces equality for the same underlying filesystem path.
    /// This is the core logic that fixes BUG-007 where the discover handler
    /// compared `\\?\D:\...\paper.pdf` (from the database) against
    /// `D:\...\paper.pdf` (from read_dir) and the comparison always failed.
    #[test]
    fn t_path_016_stripped_paths_match_for_discover() {
        let db_path = r"\\?\D:\OneDrive\Papers\test.pdf";
        let fs_path = r"D:\OneDrive\Papers\test.pdf";

        let stripped_db = strip_extended_length_prefix(db_path);
        let stripped_fs = strip_extended_length_prefix(fs_path);

        assert_eq!(
            stripped_db, stripped_fs,
            "database path and filesystem path must match after stripping prefix"
        );
    }

    /// T-PATH-017: A HashSet comparison works correctly after stripping the
    /// `\\?\` prefix from both indexed and filesystem paths. This tests the
    /// exact pattern used in the discover handler's unindexed PDF detection.
    #[test]
    fn t_path_017_hashset_comparison_with_stripped_paths() {
        use std::collections::HashSet;

        // Simulate indexed paths from the database (with `\\?\` prefix).
        let indexed_db_paths = [
            r"\\?\D:\Papers\paper_a.pdf".to_string(),
            r"\\?\D:\Papers\paper_b.pdf".to_string(),
        ];

        // Simulate filesystem paths from read_dir (without prefix).
        let fs_paths = [
            r"D:\Papers\paper_a.pdf".to_string(),
            r"D:\Papers\paper_b.pdf".to_string(),
            r"D:\Papers\paper_c.pdf".to_string(),
        ];

        // Build the indexed set with prefix stripping (the BUG-007 fix).
        let indexed_set: HashSet<String> = indexed_db_paths
            .iter()
            .map(|p| strip_extended_length_prefix(p).to_string())
            .collect();

        // Find unindexed files with prefix stripping on both sides.
        let unindexed: Vec<&str> = fs_paths
            .iter()
            .filter(|p| {
                let stripped = strip_extended_length_prefix(p);
                !indexed_set.contains(stripped)
            })
            .map(|p| strip_extended_length_prefix(p))
            .collect();

        assert_eq!(unindexed.len(), 1, "only paper_c.pdf should be unindexed");
        assert_eq!(
            unindexed[0], r"D:\Papers\paper_c.pdf",
            "the unindexed file should be paper_c.pdf"
        );
    }
}
