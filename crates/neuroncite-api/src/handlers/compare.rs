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

//! File comparison endpoint handler.
//!
//! Compares the same PDF across different index sessions, reporting per-session
//! chunk statistics and content hash consistency.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;

use neuroncite_store::{self as store};

use crate::dto::{
    API_VERSION, FileCompareRequest, FileCompareResponse, FileComparisonDto,
    FileComparisonSessionDto,
};
use crate::error::ApiError;
use crate::state::AppState;

/// Escapes SQL LIKE metacharacters in user-provided input so that `%`, `_`,
/// and the escape character `\` are treated as literal characters rather than
/// wildcards.
///
/// The escaped string is intended for use with `LIKE ?1 ESCAPE '\'` in SQL
/// queries. Without escaping, user input containing `%` or `_` would be
/// interpreted as wildcard characters, causing unintended broad matches
/// (e.g., a file name containing "100%" would match any path).
///
/// Escape rules:
/// - `\` -> `\\` (the escape character itself must be escaped first)
/// - `%` -> `\%` (LIKE "any" wildcard)
/// - `_` -> `\_` (LIKE "single char" wildcard)
fn escape_like_pattern(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '%' => escaped.push_str("\\%"),
            '_' => escaped.push_str("\\_"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// POST /api/v1/files/compare
///
/// Finds all indexed instances of a file by path or name pattern and compares
/// them across sessions.
#[utoipa::path(
    post,
    path = "/api/v1/files/compare",
    request_body = FileCompareRequest,
    responses(
        (status = 200, description = "File comparison", body = FileCompareResponse),
    )
)]
pub async fn compare_files(
    State(state): State<Arc<AppState>>,
    Json(req): Json<FileCompareRequest>,
) -> Result<Json<FileCompareResponse>, ApiError> {
    // When file_path is provided, the user supplies a literal file path that
    // should be matched as a suffix (prefixed with `%`). The path itself must
    // have its LIKE metacharacters escaped so that characters like `%` and `_`
    // in file names are treated literally. When file_name_pattern is provided,
    // the user supplies their own SQL LIKE pattern and is responsible for
    // wildcards; the metacharacters in the pattern are intentional.
    // Both fields being set simultaneously is ambiguous: the caller must
    // provide exactly one search mode. An error here is preferable to silently
    // preferring one field over the other.
    if req.file_path.is_some() && req.file_name_pattern.is_some() {
        return Err(ApiError::BadRequest {
            reason: "provide either 'file_path' or 'file_name_pattern', not both".to_string(),
        });
    }

    let pattern = match (&req.file_path, &req.file_name_pattern) {
        (Some(path), _) => format!("%{}", escape_like_pattern(path)),
        (None, Some(pat)) => pat.clone(),
        (None, None) => {
            return Err(ApiError::BadRequest {
                reason: "provide either 'file_path' or 'file_name_pattern'".to_string(),
            });
        }
    };

    let conn = state.pool.get().map_err(ApiError::from)?;

    let files = store::find_files_by_path_pattern(&conn, &pattern).map_err(ApiError::from)?;

    // Group files by canonical file name for cross-session comparison.
    let mut groups: HashMap<String, Vec<&store::FileRow>> = HashMap::new();
    for f in &files {
        let name = std::path::Path::new(&f.file_path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        groups.entry(name).or_default().push(f);
    }

    // Cache session metadata to avoid repeated lookups for the same session.
    let mut session_cache: HashMap<i64, store::SessionRow> = HashMap::new();
    let mut comparisons: Vec<FileComparisonDto> = Vec::new();

    for (file_name, file_group) in &groups {
        let mut sessions_data: Vec<FileComparisonSessionDto> = Vec::new();
        let mut chunk_counts: Vec<i64> = Vec::new();
        let mut hashes: Vec<&str> = Vec::new();

        for f in file_group {
            if let std::collections::hash_map::Entry::Vacant(e) = session_cache.entry(f.session_id)
                && let Ok(session) = store::get_session(&conn, f.session_id)
            {
                e.insert(session);
            }

            let cs = store::single_file_chunk_stats(&conn, f.id).map_err(ApiError::from)?;
            chunk_counts.push(cs.chunk_count);
            hashes.push(&f.file_hash);

            let session_info = session_cache.get(&f.session_id);
            sessions_data.push(FileComparisonSessionDto {
                session_id: f.session_id,
                file_id: f.id,
                label: session_info.and_then(|s| s.label.clone()),
                model_name: session_info
                    .map(|s| s.model_name.clone())
                    .unwrap_or_default(),
                chunk_strategy: session_info
                    .map(|s| s.chunk_strategy.clone())
                    .unwrap_or_default(),
                pages_extracted: f.page_count,
                pdf_page_count: f.pdf_page_count,
                chunks: cs.chunk_count,
                avg_chunk_bytes: cs.avg_content_len,
            });
        }

        // Determine if all instances have identical content hashes.
        let same_content = hashes.windows(2).all(|w| w[0] == w[1]);
        let min_chunks = chunk_counts.iter().copied().min().unwrap_or(0);
        let max_chunks = chunk_counts.iter().copied().max().unwrap_or(0);

        comparisons.push(FileComparisonDto {
            file_name: file_name.clone(),
            instances: file_group.len(),
            sessions: sessions_data,
            same_content,
            chunk_count_range: [min_chunks, max_chunks],
        });
    }

    Ok(Json(FileCompareResponse {
        api_version: API_VERSION.to_string(),
        pattern,
        matched_files: groups.len(),
        comparisons,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-SEC-010: escape_like_pattern escapes the backslash escape character
    /// itself, then SQL LIKE wildcards `%` and `_`.
    #[test]
    fn t_sec_010_escape_like_pattern_basic() {
        assert_eq!(escape_like_pattern("report.pdf"), "report.pdf");
        assert_eq!(escape_like_pattern("100%_done"), "100\\%\\_done");
        assert_eq!(escape_like_pattern("path\\to\\file"), "path\\\\to\\\\file");
    }

    /// T-SEC-011: escape_like_pattern handles empty input.
    #[test]
    fn t_sec_011_escape_like_pattern_empty() {
        assert_eq!(escape_like_pattern(""), "");
    }

    /// T-SEC-012: escape_like_pattern handles input that is entirely metacharacters.
    #[test]
    fn t_sec_012_escape_like_pattern_all_meta() {
        assert_eq!(escape_like_pattern("%_\\"), "\\%\\_\\\\");
    }

    /// T-SEC-013: The compare handler wraps file_path with a leading `%` for
    /// suffix matching, with the path content escaped. This test verifies the
    /// pattern construction logic without needing a database connection.
    #[test]
    fn t_sec_013_file_path_pattern_construction() {
        let path = "docs/100%_report.pdf";
        let pattern = format!("%{}", escape_like_pattern(path));
        assert_eq!(pattern, "%docs/100\\%\\_report.pdf");
    }
}
