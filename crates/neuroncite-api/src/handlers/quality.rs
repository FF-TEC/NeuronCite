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

//! Quality report endpoint handler.
//!
//! Generates a text extraction quality overview for all files in a session,
//! reporting extraction method distribution and per-file quality flags.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};

use neuroncite_store::{self as store};

use crate::dto::{
    API_VERSION, ExtractionSummaryDto, QualityFlagDetailsDto, QualityFlagDto, QualityReportResponse,
};
use crate::error::ApiError;
use crate::state::AppState;

/// GET /api/v1/sessions/{session_id}/quality
///
/// Returns extraction quality data for all files in a session.
#[utoipa::path(
    get,
    path = "/api/v1/sessions/{session_id}/quality",
    params(("session_id" = i64, Path, description = "Session ID")),
    responses(
        (status = 200, description = "Quality report", body = QualityReportResponse),
        (status = 404, description = "Session not found"),
    )
)]
pub async fn quality_report(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<i64>,
) -> Result<Json<QualityReportResponse>, ApiError> {
    let conn = state.pool.get().map_err(ApiError::from)?;

    // Verify the session exists (returns 404 if missing).
    store::get_session(&conn, session_id).map_err(ApiError::from)?;

    let rows = store::session_quality_data(&conn, session_id).map_err(ApiError::from)?;

    let mut native_count = 0_i64;
    let mut ocr_count = 0_i64;
    let mut mixed_count = 0_i64;
    let mut total_pages = 0_i64;
    let mut total_empty = 0_i64;
    let mut total_bytes = 0_i64;
    let mut quality_flags: Vec<QualityFlagDto> = Vec::new();

    for row in &rows {
        total_pages += row.page_count;
        total_empty += row.empty_pages;
        total_bytes += row.total_bytes;

        // Classify files by extraction method: native text, OCR, or mixed.
        if row.ocr_pages > 0 && row.native_pages > 0 {
            mixed_count += 1;
        } else if row.ocr_pages > 0 {
            ocr_count += 1;
        } else {
            native_count += 1;
        }

        let mut file_flags: Vec<String> = Vec::new();

        // Check if the extracted page count differs from the PDF metadata page count.
        if let Some(pdf_pc) = row.pdf_page_count {
            if row.page_count < pdf_pc {
                file_flags.push("incomplete_extraction".to_string());
            } else if row.page_count > pdf_pc {
                file_flags.push("page_count_mismatch".to_string());
            }
        }

        // Flag files where OCR pages outnumber native text pages.
        if row.ocr_pages > row.native_pages && row.ocr_pages > 0 {
            file_flags.push("ocr_heavy".to_string());
        }

        // Flag files with average content below 100 bytes per page.
        if row.page_count > 0 && row.total_bytes / row.page_count < 100 {
            file_flags.push("low_text_density".to_string());
        }

        // Flag files where more than 10% of pages are empty.
        if row.empty_pages > 0 && row.page_count > 0 && row.empty_pages * 10 > row.page_count {
            file_flags.push("many_empty_pages".to_string());
        }

        if !file_flags.is_empty() {
            let file_name = std::path::Path::new(&row.file_path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();

            quality_flags.push(QualityFlagDto {
                file_id: row.file_id,
                file_name,
                flags: file_flags,
                details: QualityFlagDetailsDto {
                    page_count: row.page_count,
                    pdf_page_count: row.pdf_page_count,
                    native_pages: row.native_pages,
                    ocr_pages: row.ocr_pages,
                    empty_pages: row.empty_pages,
                    total_bytes: row.total_bytes,
                },
            });
        }
    }

    let avg_bytes_per_page = if total_pages > 0 {
        total_bytes / total_pages
    } else {
        0
    };

    // Capture the count before moving quality_flags into the response struct.
    let files_with_issues = quality_flags.len();
    let files_clean = rows.len().saturating_sub(files_with_issues) as i64;

    Ok(Json(QualityReportResponse {
        api_version: API_VERSION.to_string(),
        session_id,
        extraction_summary: ExtractionSummaryDto {
            total_files: rows.len(),
            native_text_count: native_count,
            ocr_required_count: ocr_count,
            mixed_count,
            total_pages,
            total_empty_pages: total_empty,
            avg_bytes_per_page,
        },
        quality_flags,
        files_with_issues,
        files_clean,
    }))
}
