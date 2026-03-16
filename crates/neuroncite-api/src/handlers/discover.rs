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

//! Directory discovery endpoint handler.
//!
//! Scans a filesystem directory for PDF files and queries all existing index
//! sessions for that directory, reporting coverage and unindexed files.

use std::collections::HashSet;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;

use neuroncite_core::paths::strip_extended_length_prefix;
use neuroncite_store::{self as store};

use crate::dto::{
    API_VERSION, DiscoverRequest, DiscoverResponse, DiscoverSessionDto, FilesystemSummaryDto,
};
use crate::error::ApiError;
use crate::state::AppState;

/// POST /api/v1/discover
///
/// Discovers what is indexed at a given directory path.
#[utoipa::path(
    post,
    path = "/api/v1/discover",
    request_body = DiscoverRequest,
    responses(
        (status = 200, description = "Discovery result", body = DiscoverResponse),
    )
)]
pub async fn discover(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DiscoverRequest>,
) -> Result<Json<DiscoverResponse>, ApiError> {
    if req.directory.is_empty() {
        return Err(ApiError::BadRequest {
            reason: "directory must not be empty".to_string(),
        });
    }

    // Path containment: verify the discover directory is within the allowed
    // roots when the server is configured with an allowlist.
    crate::util::validate_path_access(std::path::Path::new(&req.directory), &state.config)?;

    // Canonicalize the requested directory for consistent DB path comparisons.
    // This is a read-only inspection endpoint; when the directory does not exist
    // yet the canonical form falls back to the raw input so the response can
    // still report zero indexed files.
    let canonical =
        neuroncite_core::paths::canonicalize_directory(std::path::Path::new(&req.directory))
            .unwrap_or_else(|_| std::path::PathBuf::from(&req.directory));
    let canonical_str = canonical.to_string_lossy().to_string();
    let dir_exists = canonical.is_dir();

    // Scan for indexable documents on disk (non-recursive, top-level directory only).
    // Tracks per-type file counts for the response DTO.
    let mut files_on_disk: Vec<(String, u64)> = Vec::new();
    let mut total_size: u64 = 0;
    let mut type_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    if dir_exists && let Ok(entries) = std::fs::read_dir(&canonical) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ft) = neuroncite_pdf::file_type_for_path(&path)
                && let Ok(meta) = entry.metadata()
            {
                let size = meta.len();
                total_size += size;
                files_on_disk.push((path.to_string_lossy().into_owned(), size));
                *type_counts.entry(ft.to_string()).or_insert(0) += 1;
            }
        }
    }

    let conn = state.pool.get().map_err(ApiError::from)?;

    // Find all sessions that reference this directory path.
    let sessions =
        store::find_sessions_by_directory(&conn, &canonical_str).map_err(ApiError::from)?;

    // Fetch aggregate statistics for all sessions in one query.
    let all_aggs = store::all_session_aggregates(&conn).map_err(ApiError::from)?;
    let agg_map: std::collections::HashMap<i64, store::SessionAggregates> =
        all_aggs.into_iter().map(|a| (a.session_id, a)).collect();

    // Collect all indexed file paths across sessions to determine unindexed PDFs.
    let mut all_indexed: HashSet<String> = HashSet::new();
    let mut session_array: Vec<DiscoverSessionDto> = Vec::new();

    for s in &sessions {
        let agg = agg_map.get(&s.id);
        let files = store::list_files_by_session(&conn, s.id).map_err(ApiError::from)?;
        // Strip the Windows extended-length path prefix (`\\?\`) from indexed
        // file paths before inserting into the comparison set. On Windows,
        // `std::fs::canonicalize()` (called during indexing in `discover_pdfs_flat`)
        // adds the `\\?\` prefix to paths, but `std::fs::read_dir()` entries
        // (used to scan the filesystem above) do NOT include this prefix.
        // Without stripping, the HashSet comparison always fails and every
        // PDF is reported as "unindexed" even when it is fully indexed.
        for f in &files {
            all_indexed.insert(strip_extended_length_prefix(&f.file_path).to_string());
        }

        let bytes = agg.map_or(0, |a| a.total_content_bytes);
        session_array.push(DiscoverSessionDto {
            session_id: s.id,
            label: s.label.clone(),
            model_name: s.model_name.clone(),
            chunk_strategy: s.chunk_strategy.clone(),
            chunk_size: s.chunk_size,
            file_count: agg.map_or(0, |a| a.file_count),
            total_chunks: agg.map_or(0, |a| a.total_chunks),
            total_pages: agg.map_or(0, |a| a.total_pages),
            total_content_bytes: bytes,
            total_words: bytes / 6,
            created_at: s.created_at,
        });
    }

    // Identify documents on disk that are not indexed in any session.
    // Both sides of the comparison are stripped of the `\\?\` prefix to
    // ensure consistent matching regardless of how paths were obtained.
    let unindexed: Vec<String> = files_on_disk
        .iter()
        .filter(|(path, _)| {
            let stripped = strip_extended_length_prefix(path);
            !all_indexed.contains(stripped)
        })
        .map(|(path, _)| strip_extended_length_prefix(path).to_string())
        .collect();

    Ok(Json(DiscoverResponse {
        api_version: API_VERSION.to_string(),
        directory: canonical_str,
        directory_exists: dir_exists,
        filesystem: FilesystemSummaryDto {
            type_counts,
            total_size_bytes: total_size,
        },
        sessions: session_array,
        unindexed_files: unindexed,
    }))
}
