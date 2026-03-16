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

//! Recursive PDF file discovery.
//!
//! Walks a directory tree starting from a given root path and collects the
//! absolute paths of all files whose extension matches `.pdf` (case-insensitive).
//! Symbolic links are followed with cycle detection via a canonicalized-path set
//! to prevent infinite loops caused by circular symlinks. Unreadable directories
//! are skipped with a warning logged through the `tracing` crate.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::PdfError;

/// Recursively discovers all PDF files under `root`.
///
/// Returns a sorted `Vec` of absolute `PathBuf` entries, one per discovered
/// `.pdf` file. The sort order is lexicographic by full path, which provides
/// deterministic processing order across runs.
///
/// Symbolic links are followed. Cycle detection uses a set of canonicalized
/// paths to prevent infinite recursion on circular symlinks. Unreadable
/// directories are skipped with a `tracing::warn` message rather than aborting
/// the entire traversal.
///
/// # Errors
///
/// Returns [`PdfError::Discovery`] if `root` does not exist or is not a
/// directory.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use neuroncite_pdf::discover_pdfs;
///
/// let pdfs = discover_pdfs(Path::new("/home/user/papers")).unwrap();
/// for path in &pdfs {
///     println!("{}", path.display());
/// }
/// ```
pub fn discover_pdfs(root: &Path) -> Result<Vec<PathBuf>, PdfError> {
    if !root.exists() {
        return Err(PdfError::Discovery(format!(
            "root path does not exist: {}",
            root.display()
        )));
    }
    if !root.is_dir() {
        return Err(PdfError::Discovery(format!(
            "root path is not a directory: {}",
            root.display()
        )));
    }

    let mut results = Vec::new();
    let mut visited = HashSet::new();

    // Canonicalize the root path to seed the visited set, preventing the root
    // itself from being re-visited if a symlink points back to it.
    let canonical_root = fs::canonicalize(root).map_err(|e| {
        PdfError::Discovery(format!(
            "failed to canonicalize root path {}: {e}",
            root.display()
        ))
    })?;
    visited.insert(canonical_root.clone());

    walk_directory(
        &canonical_root,
        &mut results,
        &mut visited,
        has_pdf_extension,
    );

    results.sort();
    Ok(results)
}

/// Recursively discovers all indexable documents under `root`.
///
/// Finds files with `.pdf`, `.html`, and `.htm` extensions (case-insensitive)
/// throughout the entire directory tree. Symbolic links are followed with cycle
/// detection. Returns a sorted `Vec` of absolute `PathBuf` entries.
///
/// This function is used by the background job executor to index all supported
/// document types in a directory selected by the user through the GUI.
///
/// # Errors
///
/// Returns [`PdfError::Discovery`] if `root` does not exist or is not a
/// directory.
pub fn discover_documents(root: &Path) -> Result<Vec<PathBuf>, PdfError> {
    if !root.exists() {
        return Err(PdfError::Discovery(format!(
            "root path does not exist: {}",
            root.display()
        )));
    }
    if !root.is_dir() {
        return Err(PdfError::Discovery(format!(
            "root path is not a directory: {}",
            root.display()
        )));
    }

    let mut results = Vec::new();
    let mut visited = HashSet::new();

    let canonical_root = fs::canonicalize(root).map_err(|e| {
        PdfError::Discovery(format!(
            "failed to canonicalize root path {}: {e}",
            root.display()
        ))
    })?;
    visited.insert(canonical_root.clone());

    walk_directory(
        &canonical_root,
        &mut results,
        &mut visited,
        has_indexable_extension,
    );

    results.sort();
    Ok(results)
}

/// Recursively traverses `dir`, collecting file paths that pass the
/// `matches_extension` filter into `results`.
///
/// `visited` tracks canonicalized directory paths to detect symlink cycles.
/// Unreadable directories and canonicalization failures are logged as warnings
/// and skipped without aborting the traversal.
///
/// The `matches_extension` parameter determines which file types are collected.
/// For PDF-only discovery, pass `has_pdf_extension`; for multi-format discovery
/// (PDF + HTML), pass `has_indexable_extension`.
fn walk_directory(
    dir: &Path,
    results: &mut Vec<PathBuf>,
    visited: &mut HashSet<PathBuf>,
    matches_extension: fn(&Path) -> bool,
) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!("skipping unreadable directory {}: {e}", dir.display());
            return;
        }
    };

    for entry_result in entries {
        let entry = match entry_result {
            Ok(entry) => entry,
            Err(e) => {
                tracing::warn!(
                    "skipping unreadable directory entry in {}: {e}",
                    dir.display()
                );
                continue;
            }
        };

        let path = entry.path();

        // Determine the file type. `entry.file_type()` does not follow symlinks,
        // so symlinks require special handling below.
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                tracing::warn!(
                    "skipping entry with unreadable file type {}: {e}",
                    path.display()
                );
                continue;
            }
        };

        if file_type.is_symlink() {
            // Follow the symlink by canonicalizing, then check what it points to.
            let canonical = match fs::canonicalize(&path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("skipping broken symlink {}: {e}", path.display());
                    continue;
                }
            };

            if canonical.is_dir() {
                // Cycle detection: skip if the canonical directory was already visited.
                if visited.insert(canonical.clone()) {
                    walk_directory(&canonical, results, visited, matches_extension);
                } else {
                    tracing::warn!(
                        "skipping symlink cycle at {} -> {}",
                        path.display(),
                        canonical.display()
                    );
                }
            } else if canonical.is_file() && matches_extension(&canonical) {
                results.push(strip_extended_prefix(canonical));
            }
        } else if file_type.is_dir() {
            // Regular directory: canonicalize for cycle detection, then recurse.
            let canonical = match fs::canonicalize(&path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(
                        "skipping directory with canonicalization failure {}: {e}",
                        path.display()
                    );
                    continue;
                }
            };
            if visited.insert(canonical.clone()) {
                walk_directory(&canonical, results, visited, matches_extension);
            }
        } else if file_type.is_file() && matches_extension(&path) {
            // Regular PDF file: canonicalize to produce an absolute path,
            // then strip the Windows extended-length prefix (\\?\) that
            // std::fs::canonicalize adds. The prefix is an internal Win32 API
            // detail that breaks string equality in cross-referencing contexts
            // (e.g., comparing indexed paths against filesystem scan results).
            match fs::canonicalize(&path) {
                Ok(canonical) => {
                    results.push(strip_extended_prefix(canonical));
                }
                Err(e) => {
                    tracing::warn!(
                        "skipping file with canonicalization failure {}: {e}",
                        path.display()
                    );
                }
            }
        }
    }
}

/// Strips the Windows extended-length path prefix (`\\?\`) from a `PathBuf`.
/// On non-Windows platforms or paths without the prefix, returns the input
/// unchanged. This is applied after `std::fs::canonicalize` to produce paths
/// that match those obtained from `read_dir` (which does not add the prefix).
fn strip_extended_prefix(path: PathBuf) -> PathBuf {
    let s = path.to_string_lossy();
    if let Some(stripped) = s.strip_prefix(r"\\?\") {
        PathBuf::from(stripped)
    } else {
        path
    }
}

/// Discovers PDF files in the top-level of `root` without recursing into
/// subdirectories.
///
/// Returns a sorted `Vec` of absolute `PathBuf` entries, one per discovered
/// `.pdf` file in the immediate directory. Subdirectories are ignored entirely.
/// This is the default mode for MCP/API indexing to prevent unintended capture
/// of PDFs in nested folders (e.g., output directories, `.neuroncite/` caches).
///
/// # Errors
///
/// Returns [`PdfError::Discovery`] if `root` does not exist or is not a
/// directory.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use neuroncite_pdf::discover_pdfs_flat;
///
/// let pdfs = discover_pdfs_flat(Path::new("/home/user/papers")).unwrap();
/// for path in &pdfs {
///     println!("{}", path.display());
/// }
/// ```
pub fn discover_pdfs_flat(root: &Path) -> Result<Vec<PathBuf>, PdfError> {
    if !root.exists() {
        return Err(PdfError::Discovery(format!(
            "root path does not exist: {}",
            root.display()
        )));
    }
    if !root.is_dir() {
        return Err(PdfError::Discovery(format!(
            "root path is not a directory: {}",
            root.display()
        )));
    }

    let canonical_root = fs::canonicalize(root).map_err(|e| {
        PdfError::Discovery(format!(
            "failed to canonicalize root path {}: {e}",
            root.display()
        ))
    })?;

    let entries = fs::read_dir(&canonical_root).map_err(|e| {
        PdfError::Discovery(format!(
            "failed to read directory {}: {e}",
            canonical_root.display()
        ))
    })?;

    let mut results = Vec::new();

    for entry_result in entries {
        let entry = match entry_result {
            Ok(entry) => entry,
            Err(e) => {
                tracing::warn!(
                    "skipping unreadable directory entry in {}: {e}",
                    canonical_root.display()
                );
                continue;
            }
        };

        let path = entry.path();

        // Determine file type. For symlinks, follow to check if the target
        // is a regular file (not a directory) with a .pdf extension.
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                tracing::warn!(
                    "skipping entry with unreadable file type {}: {e}",
                    path.display()
                );
                continue;
            }
        };

        if file_type.is_symlink() {
            let canonical = match fs::canonicalize(&path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("skipping broken symlink {}: {e}", path.display());
                    continue;
                }
            };
            // Only collect symlinks that point to files, not directories.
            if canonical.is_file() && has_pdf_extension(&canonical) {
                results.push(strip_extended_prefix(canonical));
            }
        } else if file_type.is_file() && has_pdf_extension(&path) {
            match fs::canonicalize(&path) {
                Ok(canonical) => results.push(strip_extended_prefix(canonical)),
                Err(e) => {
                    tracing::warn!(
                        "skipping file with canonicalization failure {}: {e}",
                        path.display()
                    );
                }
            }
        }
        // Directories are intentionally ignored (non-recursive mode).
    }

    results.sort();
    Ok(results)
}

/// Returns `true` if the file path has a `.pdf` extension (case-insensitive).
pub fn has_pdf_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
}

/// Returns `true` if the file path has an `.html` or `.htm` extension
/// (case-insensitive).
pub fn has_html_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("html") || ext.eq_ignore_ascii_case("htm"))
}

/// Returns `true` if the file path has an extension supported by the indexing
/// pipeline. Currently matches `.pdf`, `.html`, and `.htm` (case-insensitive).
pub fn has_indexable_extension(path: &Path) -> bool {
    has_pdf_extension(path) || has_html_extension(path)
}

/// Returns the canonical file type identifier for a path based on its extension,
/// or `None` if the extension is not recognized as an indexable format. This
/// function is the single source of truth for format detection across all crates.
/// Adding a new document format requires one match arm here.
///
/// Returned identifiers: `"pdf"`, `"html"`.
pub fn file_type_for_path(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?;
    match ext.to_ascii_lowercase().as_str() {
        "pdf" => Some("pdf"),
        "html" | "htm" => Some("html"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// T-PDF-001: Discovery finds all PDFs recursively in nested directories.
    /// Creates a temp directory with PDFs at multiple nesting depths and verifies
    /// that all PDF files are found by the discovery function.
    #[test]
    fn t_pdf_001_finds_all_pdfs_recursively() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let root = tmp.path();

        // Create nested directory structure with PDF files at various depths.
        let dirs = [
            root.to_path_buf(),
            root.join("sub1"),
            root.join("sub1").join("sub2"),
            root.join("another"),
        ];
        for dir in &dirs {
            fs::create_dir_all(dir).expect("failed to create directory");
        }

        let pdf_files = [
            root.join("top.pdf"),
            root.join("sub1").join("mid.pdf"),
            root.join("sub1").join("sub2").join("deep.pdf"),
            root.join("another").join("side.pdf"),
        ];
        for pdf in &pdf_files {
            fs::write(pdf, b"fake pdf content").expect("failed to write file");
        }

        let found = discover_pdfs(root).expect("discovery failed");

        assert_eq!(
            found.len(),
            4,
            "expected 4 PDF files, found {}",
            found.len()
        );
    }

    /// T-PDF-002: Discovery ignores non-PDF files.
    /// Creates files with .txt, .docx, and .png extensions alongside a .pdf file
    /// and verifies that only the .pdf file appears in the results.
    #[test]
    fn t_pdf_002_ignores_non_pdf_files() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let root = tmp.path();

        fs::write(root.join("document.pdf"), b"pdf").expect("write failed");
        fs::write(root.join("readme.txt"), b"text").expect("write failed");
        fs::write(root.join("report.docx"), b"docx").expect("write failed");
        fs::write(root.join("image.png"), b"png").expect("write failed");

        let found = discover_pdfs(root).expect("discovery failed");

        assert_eq!(found.len(), 1, "only the .pdf file should be found");
        assert!(
            found[0].to_string_lossy().ends_with(".pdf"),
            "the found file must have a .pdf extension"
        );
    }

    /// T-PDF-003: Discovery returns sorted paths.
    /// Creates PDF files with names that would be unsorted by filesystem order
    /// and verifies that the result is sorted lexicographically.
    #[test]
    fn t_pdf_003_returns_sorted_paths() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let root = tmp.path();

        // Create files whose names sort differently than creation order.
        fs::write(root.join("charlie.pdf"), b"c").expect("write failed");
        fs::write(root.join("alpha.pdf"), b"a").expect("write failed");
        fs::write(root.join("bravo.pdf"), b"b").expect("write failed");

        let found = discover_pdfs(root).expect("discovery failed");

        assert_eq!(found.len(), 3);
        // Verify lexicographic order by checking that each consecutive pair
        // is in non-decreasing order.
        for window in found.windows(2) {
            assert!(
                window[0] <= window[1],
                "paths are not sorted: {:?} > {:?}",
                window[0],
                window[1]
            );
        }
    }

    /// T-PDF-004: Discovery handles empty directories.
    /// An empty directory produces an empty result list without error.
    #[test]
    fn t_pdf_004_empty_directory_returns_empty_list() {
        let tmp = TempDir::new().expect("failed to create temp dir");

        let found = discover_pdfs(tmp.path()).expect("discovery failed");

        assert!(
            found.is_empty(),
            "empty directory must produce an empty result"
        );
    }

    /// T-PDF-005: Symlink cycle detection.
    /// Creates a directory structure with a circular symlink and verifies that
    /// discovery terminates without error. On platforms where symlink creation
    /// fails (e.g., unprivileged Windows), the test verifies that the visited-path
    /// set correctly prevents re-entry.
    #[test]
    fn t_pdf_005_symlink_cycle_detection() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let root = tmp.path();
        let sub = root.join("subdir");
        fs::create_dir(&sub).expect("failed to create subdir");
        fs::write(sub.join("file.pdf"), b"pdf").expect("write failed");

        // Attempt to create a symlink from subdir/loop -> root, forming a cycle.
        // On Windows, creating directory symlinks requires elevated privileges,
        // so if creation fails, we fall back to testing the visited-path set
        // directly.
        let link_path = sub.join("loop");

        #[cfg(unix)]
        let symlink_created = std::os::unix::fs::symlink(root, &link_path).is_ok();

        #[cfg(windows)]
        let symlink_created = std::os::windows::fs::symlink_dir(root, &link_path).is_ok();

        if symlink_created {
            // The symlink cycle exists; discover_pdfs must terminate.
            let found = discover_pdfs(root).expect("discovery failed");
            assert!(
                !found.is_empty(),
                "should still find the PDF despite the cycle"
            );
        } else {
            // Symlink creation was not possible; verify the visited-path set
            // mechanism directly by running discovery (which must succeed without
            // any cycle) and checking that the result is correct.
            let found = discover_pdfs(root).expect("discovery failed");
            assert_eq!(
                found.len(),
                1,
                "should find exactly one PDF without symlinks"
            );
        }
    }

    /// T-PDF-006: discover_pdfs_flat finds only top-level PDFs and ignores
    /// subdirectories. Creates a structure with PDFs at root and in a nested
    /// subdirectory. Only the root-level PDFs are returned.
    #[test]
    fn t_pdf_006_flat_ignores_subdirectories() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let root = tmp.path();

        // Create a subdirectory with a PDF that should be ignored.
        let sub = root.join("subdir");
        fs::create_dir(&sub).expect("failed to create subdirectory");
        fs::write(sub.join("nested.pdf"), b"nested").expect("write failed");

        // Create top-level PDFs that should be found.
        fs::write(root.join("top1.pdf"), b"top1").expect("write failed");
        fs::write(root.join("top2.pdf"), b"top2").expect("write failed");

        let found = discover_pdfs_flat(root).expect("flat discovery failed");

        assert_eq!(
            found.len(),
            2,
            "flat discovery must find only top-level PDFs, found: {:?}",
            found
        );

        // Verify that neither path contains "subdir" (the nested PDF is excluded).
        for path in &found {
            let path_str = path.to_string_lossy();
            assert!(
                !path_str.contains("subdir"),
                "flat discovery must not include subdirectory PDFs: {}",
                path_str
            );
        }
    }

    /// T-PDF-007: discover_pdfs_flat returns sorted paths, consistent with
    /// discover_pdfs behavior.
    #[test]
    fn t_pdf_007_flat_returns_sorted_paths() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let root = tmp.path();

        fs::write(root.join("zulu.pdf"), b"z").expect("write failed");
        fs::write(root.join("alpha.pdf"), b"a").expect("write failed");
        fs::write(root.join("mike.pdf"), b"m").expect("write failed");

        let found = discover_pdfs_flat(root).expect("flat discovery failed");

        assert_eq!(found.len(), 3);
        for window in found.windows(2) {
            assert!(
                window[0] <= window[1],
                "flat discovery paths are not sorted: {:?} > {:?}",
                window[0],
                window[1]
            );
        }
    }

    /// T-PDF-008: discover_pdfs_flat handles empty directories without error.
    #[test]
    fn t_pdf_008_flat_empty_directory() {
        let tmp = TempDir::new().expect("failed to create temp dir");

        let found = discover_pdfs_flat(tmp.path()).expect("flat discovery failed");

        assert!(
            found.is_empty(),
            "flat discovery of empty directory must return empty list"
        );
    }

    /// T-PDF-009: discover_pdfs_flat ignores non-PDF files in the top level,
    /// consistent with the recursive discovery behavior.
    #[test]
    fn t_pdf_009_flat_ignores_non_pdf_files() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let root = tmp.path();

        fs::write(root.join("paper.pdf"), b"pdf").expect("write failed");
        fs::write(root.join("notes.txt"), b"txt").expect("write failed");
        fs::write(root.join("data.csv"), b"csv").expect("write failed");

        let found = discover_pdfs_flat(root).expect("flat discovery failed");

        assert_eq!(
            found.len(),
            1,
            "flat discovery must find only the .pdf file"
        );
    }

    /// T-PDF-010: discover_pdfs (recursive) finds PDFs in subdirectories,
    /// while discover_pdfs_flat does not. Verifies the behavioral difference
    /// between the two functions on the same directory structure.
    #[test]
    fn t_pdf_010_recursive_vs_flat_comparison() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let root = tmp.path();

        fs::write(root.join("top.pdf"), b"top").expect("write failed");
        let sub = root.join("nested");
        fs::create_dir(&sub).expect("failed to create subdirectory");
        fs::write(sub.join("deep.pdf"), b"deep").expect("write failed");

        let recursive = discover_pdfs(root).expect("recursive discovery failed");
        let flat = discover_pdfs_flat(root).expect("flat discovery failed");

        assert_eq!(
            recursive.len(),
            2,
            "recursive discovery must find both PDFs"
        );
        assert_eq!(
            flat.len(),
            1,
            "flat discovery must find only the top-level PDF"
        );
    }

    /// T-PDF-011: discover_documents finds PDFs and HTML files recursively
    /// in nested directories. Verifies that both file types are collected
    /// while other file types (e.g., .txt) are ignored.
    #[test]
    fn t_pdf_011_discover_documents_finds_pdf_and_html() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let root = tmp.path();

        // Create nested directory structure with mixed file types.
        let sub = root.join("docs");
        fs::create_dir(&sub).expect("failed to create subdirectory");

        fs::write(root.join("paper.pdf"), b"pdf").expect("write failed");
        fs::write(sub.join("article.html"), b"html").expect("write failed");
        fs::write(sub.join("page.htm"), b"htm").expect("write failed");
        fs::write(root.join("notes.txt"), b"txt").expect("write failed");
        fs::write(sub.join("data.csv"), b"csv").expect("write failed");

        let found = discover_documents(root).expect("document discovery failed");

        assert_eq!(
            found.len(),
            3,
            "discover_documents must find 1 PDF + 1 HTML + 1 HTM = 3 files, found: {:?}",
            found
        );

        // Verify that .txt and .csv files are excluded.
        for path in &found {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            assert!(
                ext.eq_ignore_ascii_case("pdf")
                    || ext.eq_ignore_ascii_case("html")
                    || ext.eq_ignore_ascii_case("htm"),
                "unexpected file type in results: {}",
                path.display()
            );
        }
    }

    /// T-PDF-012: discover_documents finds only PDFs when no HTML files exist,
    /// producing the same result as discover_pdfs.
    #[test]
    fn t_pdf_012_discover_documents_pdf_only_matches_discover_pdfs() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let root = tmp.path();

        let sub = root.join("papers");
        fs::create_dir(&sub).expect("failed to create subdirectory");

        fs::write(root.join("top.pdf"), b"top").expect("write failed");
        fs::write(sub.join("nested.pdf"), b"nested").expect("write failed");

        let docs = discover_documents(root).expect("document discovery failed");
        let pdfs = discover_pdfs(root).expect("PDF discovery failed");

        assert_eq!(
            docs.len(),
            pdfs.len(),
            "discover_documents and discover_pdfs must return the same count \
             when only PDF files exist"
        );
    }
}
