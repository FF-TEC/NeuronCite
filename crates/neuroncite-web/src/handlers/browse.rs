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

//! File system browsing handlers for the web frontend.
//!
//! The browser cannot access the local filesystem directly, so these endpoints
//! provide directory listing, native OS folder selection, and PDF discovery
//! functionality. The native folder dialog uses platform-specific commands
//! (PowerShell on Windows, osascript on macOS, zenity/kdialog on Linux) to
//! open a standard OS folder picker window.
//!
//! The scan-pdfs endpoint enriches discovered PDF files with their indexing
//! status (indexed, outdated, pending) by cross-referencing against the
//! `indexed_file` table in the active session's database when a `session_id`
//! is provided in the request.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use neuroncite_core::AppConfig;

use crate::WebState;

/// Validates that the requested path is within the configured browse roots.
///
/// When `allowed_browse_roots` in AppConfig is empty (default), all paths are
/// permitted. When non-empty, the canonical form of the requested path must be
/// a descendant of at least one of the allowed roots. This prevents directory
/// traversal attacks via symlinks or `..` components.
///
/// # Security implications of empty `allowed_browse_roots`
///
/// When `allowed_browse_roots` is empty, this function permits access to the
/// entire filesystem -- every directory and file that the server process has
/// read permission for becomes enumerable through the browse and scan-pdfs
/// endpoints. This includes system directories, user home directories, and
/// any mounted volumes. The empty-list default exists for backward
/// compatibility with localhost-only deployments where the operator has full
/// local filesystem access anyway.
///
/// However, when the server binds to a non-loopback address (e.g., 0.0.0.0
/// for LAN access), the empty allowlist means any network client can browse
/// the entire filesystem. This is a significant information disclosure risk:
/// an attacker can enumerate directory structures, discover file names and
/// sizes, and locate sensitive files (configuration, credentials, database
/// files) without needing further exploits. Operators exposing the server
/// to a network MUST populate `allowed_browse_roots` in the configuration
/// file to restrict browsing to intended directories only.
///
/// The `AppConfig::validate()` method logs a `warn!`-level message at
/// startup when this dangerous combination (non-loopback bind + empty
/// allowlist) is detected.
///
/// Returns `Ok(())` when the path is allowed, or `Err((StatusCode, String))`
/// with a 403 Forbidden response when the path falls outside the allowlist.
fn validate_browse_path(path: &Path, config: &AppConfig) -> Result<(), (StatusCode, String)> {
    if config.allowed_browse_roots.is_empty() {
        if neuroncite_core::config::is_loopback(&config.bind_address) {
            // Loopback bind with empty allowlist: all paths are permitted
            // (backwards-compatible default for localhost-only use where the
            // operator has full filesystem access via other means anyway).
            return Ok(());
        }
        // Non-loopback bind with empty allowlist: deny all browse requests.
        // Without an explicit allowlist, binding to a network interface would
        // expose the entire filesystem to any network client. The operator
        // must populate `allowed_browse_roots` in the configuration file.
        return Err((
            StatusCode::FORBIDDEN,
            "filesystem browsing is disabled when no allowed_browse_roots are configured \
             and the server is bound to a non-loopback address"
                .to_string(),
        ));
    }

    // Canonicalize the requested path to resolve symlinks and ".." components.
    // If canonicalization fails (e.g. path does not exist), reject the request
    // because we cannot verify containment without a canonical path.
    // A fixed error message is used (no path interpolation) to prevent the
    // endpoint from confirming whether a specific path exists or not. Reflecting
    // the requested path in a 403 response leaks filesystem structure to callers
    // that probe for path existence by observing which paths are "cannot resolve"
    // vs "outside allowed roots".
    let canonical = std::fs::canonicalize(path).map_err(|_| {
        (
            StatusCode::FORBIDDEN,
            "the requested path could not be resolved".to_string(),
        )
    })?;

    for root in &config.allowed_browse_roots {
        let root_path = PathBuf::from(root);
        // Canonicalize the root as well to ensure consistent comparison.
        if let Ok(canonical_root) = std::fs::canonicalize(&root_path)
            && canonical.starts_with(&canonical_root)
        {
            return Ok(());
        }
    }

    Err((
        StatusCode::FORBIDDEN,
        "path is outside the allowed browse roots".to_string(),
    ))
}

/// Request body for the directory browsing endpoint.
#[derive(Deserialize)]
pub struct BrowseRequest {
    /// Absolute path to list. Empty string or "/" returns the root/drive listing.
    pub path: String,
}

/// Single entry in a directory listing.
#[derive(Serialize)]
pub struct DirEntry {
    /// File or directory name (not the full path).
    pub name: String,
    /// "directory" or "file".
    pub kind: String,
    /// File size in bytes (directories report 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

/// Response body for the directory browsing endpoint.
#[derive(Serialize)]
pub struct BrowseResponse {
    /// The listed directory path.
    pub path: String,
    /// Parent directory path (empty string for root/drive level).
    pub parent: String,
    /// Directory contents sorted alphabetically with directories first.
    pub entries: Vec<DirEntry>,
    /// Available drive letters (Windows only, empty on Unix).
    pub drives: Vec<String>,
}

/// Lists the contents of a directory on the local filesystem. On Windows, when
/// the path is empty, returns a list of available drive letters. Directories
/// are listed before files, both sorted alphabetically.
pub async fn browse_directory(
    State(state): State<Arc<WebState>>,
    Json(req): Json<BrowseRequest>,
) -> Result<Json<BrowseResponse>, (StatusCode, String)> {
    let drives = list_drives();

    // Empty path returns drive/root listing
    if req.path.is_empty() || req.path == "/" {
        return Ok(Json(BrowseResponse {
            path: String::new(),
            parent: String::new(),
            entries: drives
                .iter()
                .map(|d| DirEntry {
                    name: d.clone(),
                    kind: "directory".to_string(),
                    size: None,
                })
                .collect(),
            drives,
        }));
    }

    let path = PathBuf::from(&req.path);

    // Validate the requested path against the configured browse roots allowlist.
    validate_browse_path(&path, &state.app_state.config)?;

    if !path.exists() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Path does not exist: {}", req.path),
        ));
    }
    if !path.is_dir() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Path is not a directory: {}", req.path),
        ));
    }

    let parent = path
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();

    let mut dirs = Vec::new();
    let mut files = Vec::new();

    match std::fs::read_dir(&path) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let metadata = entry.metadata().ok();
                let name = entry.file_name().to_string_lossy().to_string();

                if metadata.as_ref().is_some_and(|m| m.is_dir()) {
                    dirs.push(DirEntry {
                        name,
                        kind: "directory".to_string(),
                        size: None,
                    });
                } else {
                    files.push(DirEntry {
                        name,
                        kind: "file".to_string(),
                        size: metadata.map(|m| m.len()),
                    });
                }
            }
        }
        Err(e) => {
            return Err((StatusCode::FORBIDDEN, format!("Cannot read directory: {e}")));
        }
    }

    // Sort alphabetically, directories first
    dirs.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    files.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

    let mut entries = dirs;
    entries.append(&mut files);

    Ok(Json(BrowseResponse {
        path: req.path,
        parent,
        entries,
        drives,
    }))
}

/// Request body for the document scan endpoint.
#[derive(Deserialize)]
pub struct ScanDocumentsRequest {
    /// Absolute path to the directory to scan recursively for indexable documents.
    pub path: String,
    /// Session ID for cross-referencing file index status. When provided,
    /// each document entry is enriched with its status ("indexed", "outdated",
    /// or "pending") by comparing filesystem metadata against the
    /// `indexed_file` table records for this session.
    pub session_id: Option<i64>,
}

/// Single indexable document discovered during a recursive directory scan.
#[derive(Serialize)]
pub struct DocumentEntry {
    /// Absolute path to the document file.
    pub path: String,
    /// Filename without directory components.
    pub name: String,
    /// Subfolder path relative to the scan root (empty string for root-level files).
    pub subfolder: String,
    /// File size in bytes.
    pub size: u64,
    /// File modification time as UNIX timestamp (seconds since epoch).
    pub mtime: i64,
    /// Canonical file type identifier returned by `file_type_for_path`
    /// (e.g. "pdf", "html"). Used by the frontend for format badges.
    pub file_type: String,
    /// Indexing status relative to the active session: "indexed" when the file
    /// path exists in the `indexed_file` table and the stored mtime/size match,
    /// "outdated" when the path exists but metadata differs, "pending" when
    /// the file is not in the database. Always "pending" when no session_id
    /// is provided.
    pub status: String,
    /// Database file ID from the indexed_file table. Only populated when the
    /// file is found in the active session (status is "indexed" or "outdated").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<i64>,
    /// Number of extracted text pages stored in the database for this file.
    /// Only populated for indexed/outdated files.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_count: Option<i64>,
    /// Number of active (non-deleted) chunks produced from this file.
    /// Only populated for indexed/outdated files.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_count: Option<i64>,
}

/// Maximum directory recursion depth for the document scan walk.
///
/// A walk that descends beyond this depth is truncated. Symlink loops and
/// abnormally deep directory trees (mounted archives, recursive bind-mounts)
/// are bounded by this value even when the symlink cycle detection does not
/// fire (e.g. a legitimate but very deep tree).
const MAX_WALK_DEPTH: usize = 20;

/// Maximum number of document files collected per scan request.
///
/// When the walk reaches this count, it stops immediately and sets the
/// `truncated` flag in the response. This prevents excessive memory use
/// and response payload size for directories with very large file counts.
const MAX_WALK_FILES: usize = 10_000;

/// Wall-clock timeout for the blocking filesystem scan, in seconds.
///
/// If the scan does not complete within this time (e.g. due to a very large
/// tree or slow network share), the handler returns HTTP 504 Gateway Timeout.
const SCAN_TIMEOUT_SECS: u64 = 60;

/// Response body for the document scan endpoint.
#[derive(Serialize)]
pub struct ScanDocumentsResponse {
    /// All indexable documents found recursively under the scanned directory.
    pub files: Vec<DocumentEntry>,
    /// True when the scan was stopped early because the file count reached
    /// MAX_WALK_FILES. The caller should prompt the user to select a more
    /// specific subdirectory.
    pub truncated: bool,
}

/// Recursively discovers all indexable documents in a directory and returns
/// them with their filesystem metadata. Called from within a `spawn_blocking`
/// closure. Collects all formats recognized by `file_type_for_path`.
///
/// Returns the list of discovered document entries and a boolean indicating
/// whether the walk was stopped early because it reached MAX_WALK_FILES.
fn scan_documents_recursive(root: &Path) -> (Vec<DocumentEntry>, bool) {
    let mut result = Vec::new();
    // The visited set holds canonical paths of every directory entered.
    // When canonicalize() succeeds and the canonical path is already in the
    // set, the walk skips that directory to break symlink cycles. When
    // canonicalize() fails (e.g. dangling symlink), the directory is also
    // skipped because we cannot verify it has not been visited before.
    let mut visited: HashSet<PathBuf> = HashSet::new();
    walk_dir(root, root, &mut result, 0, &mut visited);
    let truncated = result.len() >= MAX_WALK_FILES;
    result.sort_by(|a, b| a.path.cmp(&b.path));
    (result, truncated)
}

/// Recursive directory walker that collects indexable documents into `result`.
///
/// Uses `entry.file_type()` (which does not follow symlinks) to classify
/// each entry instead of `path.is_dir()` (which does follow symlinks). This
/// prevents symlink traversal to directories outside the scan root. Symlink
/// entries are explicitly skipped -- they are neither descended into nor
/// collected as document files.
///
/// `depth` starts at 0 for the root and increments with each recursion level.
/// The walk stops descending when depth exceeds MAX_WALK_DEPTH, protecting
/// against stack overflow on abnormally deep trees.
///
/// `visited` accumulates canonical paths of every directory entered so that
/// symlink cycles are detected and broken before entering an already-visited
/// directory a second time.
fn walk_dir(
    root: &Path,
    current: &Path,
    result: &mut Vec<DocumentEntry>,
    depth: usize,
    visited: &mut HashSet<PathBuf>,
) {
    // Depth limit protects against stack overflow on pathological trees.
    if depth > MAX_WALK_DEPTH {
        return;
    }
    // File count cap prevents excessive memory and response payload size.
    if result.len() >= MAX_WALK_FILES {
        return;
    }

    // Symlink cycle detection: canonicalize the current directory path and
    // check whether we have entered it before. canonicalize() resolves all
    // symlink components, so two different directory paths that point to the
    // same inode (via symlinks) will produce the same canonical path. When
    // canonicalize() fails, skip the directory — we cannot verify it is safe.
    match current.canonicalize() {
        Ok(canonical) => {
            if !visited.insert(canonical) {
                // Already visited this physical directory. Skip to break the cycle.
                return;
            }
        }
        Err(_) => return,
    }

    let entries = match std::fs::read_dir(current) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        if result.len() >= MAX_WALK_FILES {
            break;
        }

        // Use entry.file_type() to classify the entry without following
        // symlinks. This is the safe alternative to path.is_dir(), which
        // follows symlinks and can escape the scan root via a symlink to
        // an ancestor directory.
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };

        if file_type.is_dir() {
            walk_dir(root, &entry.path(), result, depth + 1, visited);
        } else if file_type.is_file() {
            let path = entry.path();
            // Determine format type via the central format registry.
            // Files with unrecognized extensions are skipped.
            let ft = match neuroncite_pdf::file_type_for_path(&path) {
                Some(ft) => ft,
                None => continue,
            };

            // Use entry.metadata() instead of std::fs::metadata(&path) so
            // that the metadata call does not follow symlinks for the file
            // itself. entry.metadata() is equivalent to lstat() on Unix.
            let metadata = entry.metadata().ok();
            let subfolder = path
                .parent()
                .and_then(|p| p.strip_prefix(root).ok())
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();

            let mtime = metadata
                .as_ref()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);

            result.push(DocumentEntry {
                path: path.to_string_lossy().to_string(),
                name: path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default(),
                subfolder,
                size: metadata.map(|m| m.len()).unwrap_or(0),
                mtime,
                file_type: ft.to_string(),
                status: "pending".to_string(),
                file_id: None,
                page_count: None,
                chunk_count: None,
            });
        }
        // file_type.is_symlink() entries are intentionally skipped. Following
        // symlinks during a recursive scan can escape the intended directory
        // tree and introduce symlink cycles. The scan root itself may already
        // be a symlink (resolved by validate_browse_path via canonicalize),
        // but entries within the tree are not followed.
    }
}

/// Scans a directory tree for indexable documents and returns their metadata
/// and indexing status. When a `session_id` is provided, queries the
/// `indexed_file` table to determine each file's status (indexed, outdated,
/// or pending). The scan runs on a blocking thread with a wall-clock timeout
/// of SCAN_TIMEOUT_SECS and a file count cap of MAX_WALK_FILES. When either
/// limit is reached the response includes `truncated: true`.
pub async fn scan_documents(
    State(state): State<Arc<WebState>>,
    Json(req): Json<ScanDocumentsRequest>,
) -> Result<Json<ScanDocumentsResponse>, (StatusCode, String)> {
    let root = PathBuf::from(&req.path);

    // Validate the requested path against the configured browse roots allowlist.
    validate_browse_path(&root, &state.app_state.config)?;

    if !root.exists() || !root.is_dir() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Invalid directory: {}", req.path),
        ));
    }

    // Run the filesystem scan on a blocking thread with a wall-clock timeout.
    // The timeout prevents indefinite blocking on slow network shares or
    // abnormally deep directory trees.
    let scan_result = tokio::time::timeout(
        Duration::from_secs(SCAN_TIMEOUT_SECS),
        tokio::task::spawn_blocking(move || scan_documents_recursive(&root)),
    )
    .await;

    let (mut files, truncated) = match scan_result {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Scan failed: {e}"),
            ));
        }
        Err(_elapsed) => {
            return Err((
                StatusCode::GATEWAY_TIMEOUT,
                format!(
                    "Filesystem scan did not complete within {SCAN_TIMEOUT_SECS} seconds. \
                     Select a more specific subdirectory."
                ),
            ));
        }
    };

    // Cross-reference against the indexed_file table when a session is active.
    // Also loads per-file chunk statistics for the detail view.
    //
    // The pool.get() call and the two SELECT queries run inside spawn_blocking
    // so the axum handler thread is not blocked during connection acquisition
    // or query execution. The file-status enrichment loop runs outside the
    // blocking task since it does not touch the database.
    if let Some(session_id) = req.session_id {
        let pool = state.app_state.pool.clone();
        // Type alias for the pair returned by the blocking database query.
        // The tuple captures (file_id, mtime_secs, size_bytes, hash_stamp) per path
        // and chunk_count per file_id from the indexed corpus.
        type FileIndexMap = HashMap<String, (i64, i64, i64, i64)>;
        type ChunkCountMap = HashMap<i64, i64>;
        let (index_map, chunk_map): (FileIndexMap, ChunkCountMap) =
            tokio::task::spawn_blocking(move || {
            let conn = pool.get().map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Database connection failed: {e}"),
                )
            })?;

            let indexed_files =
                neuroncite_store::list_files_by_session(&conn, session_id).map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to query indexed files: {e}"),
                    )
                })?;

            // Load per-file chunk counts for indexed files.
            let chunk_stats = neuroncite_store::file_chunk_stats_by_session(&conn, session_id)
                .unwrap_or_default();
            let chunk_map: HashMap<i64, i64> = chunk_stats
                .into_iter()
                .map(|s| (s.file_id, s.chunk_count))
                .collect();

            // Build a lookup table keyed by file_path for O(1) status resolution.
            // Stores (mtime, size, file_id, page_count) for each indexed file.
            let index_map: HashMap<String, (i64, i64, i64, i64)> = indexed_files
                .into_iter()
                .map(|f| {
                    // Strip the Windows extended-length prefix (\\?\) from stored
                    // paths so they match the filesystem scan paths produced by
                    // walk_dir (which uses entry.path() without canonicalization).
                    let clean_path =
                        neuroncite_core::paths::strip_extended_length_prefix(&f.file_path)
                            .to_string();
                    (clean_path, (f.mtime, f.size, f.id, f.page_count))
                })
                .collect();

            Ok::<(HashMap<String, (i64, i64, i64, i64)>, HashMap<i64, i64>), (StatusCode, String)>((index_map, chunk_map))
        })
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Database task panicked: {e}"),
            )
        })??;

        for entry in &mut files {
            let lookup_key = neuroncite_core::paths::strip_extended_length_prefix(&entry.path);
            if let Some(&(stored_mtime, stored_size, file_id, page_count)) =
                index_map.get(lookup_key)
            {
                entry.file_id = Some(file_id);
                entry.page_count = Some(page_count);
                entry.chunk_count = chunk_map.get(&file_id).copied();
                if stored_mtime == entry.mtime && stored_size == entry.size as i64 {
                    entry.status = "indexed".to_string();
                } else {
                    entry.status = "outdated".to_string();
                }
            }
        }
    }

    Ok(Json(ScanDocumentsResponse { files, truncated }))
}

/// Returns a list of filesystem roots. On Windows, returns available drive
/// letters (A:\ through Z:\). On Linux/macOS, returns the single root "/".
fn list_drives() -> Vec<String> {
    #[cfg(target_os = "windows")]
    {
        let mut drives = Vec::new();
        // Check drive letters A-Z by attempting to read their root directory.
        for letter in b'A'..=b'Z' {
            let drive = format!("{}:\\", letter as char);
            if Path::new(&drive).exists() {
                drives.push(drive);
            }
        }
        drives
    }

    #[cfg(not(target_os = "windows"))]
    {
        vec!["/".to_string()]
    }
}

/// Response body for the native folder dialog endpoint.
#[derive(Serialize)]
pub struct NativeFolderResponse {
    /// The selected folder path, or empty string if the user cancelled.
    pub path: String,
    /// Whether a folder was selected (false when the user cancelled the dialog).
    pub selected: bool,
}

/// Opens a native OS folder selection dialog and returns the selected path.
/// Uses `rfd::AsyncFileDialog` which delegates to the platform's native picker:
/// - Windows: `IFileOpenDialog` COM interface on a dedicated STA thread with a
///   message loop, ensuring the dialog receives foreground focus.
/// - macOS: `NSOpenPanel` dispatched to the main thread via GCD. Requires the
///   main thread to run a platform event loop (tao in GUI mode). In non-GUI
///   mode, the main thread is parked in tokio's block_on and cannot process
///   GCD dispatches, causing a deadlock. The `native_dialogs` flag on WebState
///   prevents this by returning 503 before rfd is called.
/// - Linux: `xdg-desktop-portal` D-Bus with GTK3 fallback
///
/// The async API is used instead of the synchronous `FileDialog` because the
/// sync variant runs on `spawn_blocking` worker threads that lack foreground
/// rights on Windows, causing the dialog to appear behind the browser window.
///
/// The rfd call runs in a separate tokio task so that a panic inside rfd
/// (e.g., missing display server, Cocoa assertion failure) does not crash
/// the handler task. The handler receives the result via a oneshot channel
/// and returns 503 if the spawned task panicked.
///
/// Returns an empty path with `selected: false` when the user cancels.
pub async fn native_folder_dialog(
    State(state): State<Arc<WebState>>,
) -> Result<Json<NativeFolderResponse>, (StatusCode, String)> {
    if !state.native_dialogs {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "native dialogs not available in this mode".to_string(),
        ));
    }

    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let result = rfd::AsyncFileDialog::new()
            .set_title("Select Document Folder")
            .pick_folder()
            .await;
        let _ = tx.send(result.map(|h| h.path().to_string_lossy().to_string()));
    });

    match rx.await {
        Ok(Some(path)) => Ok(Json(NativeFolderResponse {
            path,
            selected: true,
        })),
        Ok(None) => Ok(Json(NativeFolderResponse {
            path: String::new(),
            selected: false,
        })),
        Err(_) => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "native folder dialog failed (task panicked)".to_string(),
        )),
    }
}

/// Query parameters for the native file dialog endpoint.
/// The `filter` field controls the file type filter shown in the dialog.
#[derive(Deserialize)]
pub struct NativeFileQuery {
    /// File type filter: "bib" for BibTeX, "csv" for CSV, "tex" for LaTeX.
    /// Defaults to "bib" when omitted for backward compatibility.
    #[serde(default = "default_bib_filter")]
    pub filter: String,
}

fn default_bib_filter() -> String {
    "bib".to_string()
}

/// Opens a native OS file selection dialog filtered by the `filter` query
/// parameter. Supported filters: "bib" (BibTeX), "csv" (CSV), "tex" (LaTeX).
/// Defaults to BibTeX when no filter is specified. Uses `rfd::AsyncFileDialog`
/// with `pick_file()` and returns the selected path or an empty unselected
/// response if the user cancels.
///
/// Returns 503 when native dialogs are disabled (non-GUI mode) or when the
/// rfd task panics. The frontend falls back to the custom browser modal.
pub async fn native_file_dialog(
    axum::extract::Query(params): axum::extract::Query<NativeFileQuery>,
    State(state): State<Arc<WebState>>,
) -> Result<Json<NativeFolderResponse>, (StatusCode, String)> {
    if !state.native_dialogs {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "native dialogs not available in this mode".to_string(),
        ));
    }

    let (title, label, extensions): (&str, &str, &[&str]) = match params.filter.as_str() {
        "csv" => ("Select CSV File", "CSV", &["csv"]),
        "tex" => ("Select LaTeX File", "LaTeX", &["tex"]),
        _ => ("Select BibTeX File", "BibTeX", &["bib"]),
    };

    // Owned copies for the spawned task (rfd builder requires &str, so
    // the strings must live in the same task that calls pick_file).
    let title = title.to_string();
    let label = label.to_string();
    let extensions: Vec<String> = extensions.iter().map(|s| s.to_string()).collect();

    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let ext_refs: Vec<&str> = extensions.iter().map(|s| s.as_str()).collect();
        let result = rfd::AsyncFileDialog::new()
            .set_title(&title)
            .add_filter(&label, &ext_refs)
            .add_filter("All Files", &["*"])
            .pick_file()
            .await;
        let _ = tx.send(result.map(|h| h.path().to_string_lossy().to_string()));
    });

    match rx.await {
        Ok(Some(path)) => Ok(Json(NativeFolderResponse {
            path,
            selected: true,
        })),
        Ok(None) => Ok(Json(NativeFolderResponse {
            path: String::new(),
            selected: false,
        })),
        Err(_) => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "native file dialog failed (task panicked)".to_string(),
        )),
    }
}
