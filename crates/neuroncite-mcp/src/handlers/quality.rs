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

//! Handler for the `neuroncite_quality_report` MCP tool.
//!
//! Generates a text extraction quality overview for all files in a session.
//! Reports extraction method distribution (native text vs OCR), page count
//! mismatches, empty pages, and per-file quality flags.

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_core::paths::strip_extended_length_prefix;

/// Generates a quality report for a session.
///
/// # Parameters (from MCP tool call)
///
/// - `session_id` (required): Session to generate the quality report for.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_id = params["session_id"]
        .as_i64()
        .ok_or("missing required parameter: session_id")?;

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    neuroncite_store::get_session(&conn, session_id)
        .map_err(|_| format!("session {session_id} not found"))?;

    let rows = neuroncite_store::session_quality_data(&conn, session_id)
        .map_err(|e| format!("fetching quality data: {e}"))?;

    // Fetch per-page quality details (empty and OCR pages) in a single query.
    // Build a lookup map: file_id -> (empty_page_numbers, ocr_page_numbers).
    let page_details = neuroncite_store::page_quality_details(&conn, session_id)
        .map_err(|e| format!("fetching page quality details: {e}"))?;

    let mut page_detail_map: std::collections::HashMap<i64, (Vec<i64>, Vec<i64>)> =
        std::collections::HashMap::new();
    for detail in &page_details {
        let entry = page_detail_map.entry(detail.file_id).or_default();
        if detail.byte_count == 0 {
            entry.0.push(detail.page_number);
        }
        if detail.backend == "ocr" {
            entry.1.push(detail.page_number);
        }
    }

    // Classify each file by extraction method and collect quality flags.
    let mut native_count = 0_i64;
    let mut ocr_count = 0_i64;
    let mut mixed_count = 0_i64;
    let mut total_pages = 0_i64;
    let mut total_empty = 0_i64;
    let mut total_bytes = 0_i64;
    let mut flags: Vec<serde_json::Value> = Vec::new();

    for row in &rows {
        total_pages += row.page_count;
        total_empty += row.empty_pages;
        total_bytes += row.total_bytes;

        // Classify extraction method.
        if row.ocr_pages > 0 && row.native_pages > 0 {
            mixed_count += 1;
        } else if row.ocr_pages > 0 {
            ocr_count += 1;
        } else {
            native_count += 1;
        }

        // Per-file text density: words per page using the same bytes/6
        // heuristic as the neuroncite_files handler.
        let words_per_page = if row.page_count > 0 {
            (row.total_bytes / 6) as f64 / row.page_count as f64
        } else {
            0.0
        };

        // Collect per-file quality flags.
        let mut file_flags: Vec<&str> = Vec::new();

        if let Some(pdf_pc) = row.pdf_page_count {
            if row.page_count < pdf_pc {
                file_flags.push("incomplete_extraction");
            } else if row.page_count > pdf_pc {
                file_flags.push("page_count_mismatch");
            }
        }

        if row.ocr_pages > row.native_pages && row.ocr_pages > 0 {
            file_flags.push("ocr_heavy");
        }

        if row.page_count > 0 && row.total_bytes / row.page_count < 100 {
            file_flags.push("low_text_density");
        }

        if row.empty_pages > 0 && row.page_count > 0 && row.empty_pages * 10 > row.page_count {
            file_flags.push("many_empty_pages");
        }

        // Look up per-page detail for this file (empty and OCR page numbers).
        let (empty_page_numbers, ocr_page_numbers) = page_detail_map
            .get(&row.file_id)
            .cloned()
            .unwrap_or_default();

        if !file_flags.is_empty() {
            let file_name = std::path::Path::new(strip_extended_length_prefix(&row.file_path))
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();

            // Generate actionable per-file recommendations based on the
            // detected quality flags. Each recommendation describes what the
            // flag means and suggests a concrete remediation step.
            let recommendations: Vec<&str> = file_flags
                .iter()
                .map(|flag| match *flag {
                    "incomplete_extraction" => {
                        "some pages failed text extraction; re-index with OCR \
                         enabled (ocr_language parameter) to extract text from \
                         image-based pages"
                    }
                    "page_count_mismatch" => {
                        "extracted page count exceeds the structural PDF page count; \
                         verify the PDF is not corrupted and consider re-indexing"
                    }
                    "ocr_heavy" => {
                        "majority of pages required OCR fallback; this PDF is likely \
                         a scanned document -- ensure Tesseract OCR is installed and \
                         the correct ocr_language is configured for the document's language"
                    }
                    "low_text_density" => {
                        "very little text per page (< 100 bytes); the PDF may contain \
                         primarily images, charts, or formulas -- consider using a \
                         larger chunk_size to aggregate sparse text across pages"
                    }
                    "many_empty_pages" => {
                        "more than 10% of pages contain no extractable text; these \
                         may be cover pages, blank separators, or image-only pages -- \
                         empty pages do not affect search accuracy but reduce chunk density"
                    }
                    _ => "unknown quality flag",
                })
                .collect();

            flags.push(serde_json::json!({
                "file_id": row.file_id,
                "file_name": file_name,
                "flags": file_flags,
                "recommendations": recommendations,
                "details": {
                    "page_count": row.page_count,
                    "pdf_page_count": row.pdf_page_count,
                    "native_pages": row.native_pages,
                    "ocr_pages": row.ocr_pages,
                    "empty_pages": row.empty_pages,
                    "total_bytes": row.total_bytes,
                    "words_per_page": (words_per_page * 10.0).round() / 10.0,
                    "empty_page_numbers": empty_page_numbers,
                    "ocr_page_numbers": ocr_page_numbers,
                },
            }));
        }
    }

    let avg_bytes_per_page = if total_pages > 0 {
        total_bytes / total_pages
    } else {
        0
    };

    // Session-level word estimate using the same bytes/6 heuristic.
    let total_words = total_bytes / 6;
    let avg_words_per_page = if total_pages > 0 {
        total_words as f64 / total_pages as f64
    } else {
        0.0
    };

    // Build session-level recommendations based on aggregate patterns across
    // all files. These complement the per-file recommendations by identifying
    // corpus-wide issues that affect the session as a whole.
    let mut session_recommendations: Vec<String> = Vec::new();

    if ocr_count > 0 || mixed_count > 0 {
        let ocr_file_count = ocr_count + mixed_count;
        session_recommendations.push(format!(
            "{ocr_file_count} of {} files required OCR; verify Tesseract is \
             installed and the ocr_language parameter matches the document language",
            rows.len()
        ));
    }

    if total_pages > 0 && total_empty > 0 {
        let empty_pct = (total_empty as f64 / total_pages as f64 * 100.0).round() as i64;
        if empty_pct > 5 {
            session_recommendations.push(format!(
                "{empty_pct}% of pages across the session are empty ({total_empty} of \
                 {total_pages}); consider reviewing source PDFs for blank or image-only pages"
            ));
        }
    }

    if total_pages > 0 && avg_bytes_per_page < 200 {
        session_recommendations.push(format!(
            "average text density is low ({avg_bytes_per_page} bytes/page); \
             a larger chunk_size may help aggregate sparse text for search"
        ));
    }

    Ok(serde_json::json!({
        "session_id": session_id,
        "extraction_summary": {
            "total_files": rows.len(),
            "native_text_count": native_count,
            "ocr_required_count": ocr_count,
            "mixed_count": mixed_count,
            "total_pages": total_pages,
            "total_empty_pages": total_empty,
            "avg_bytes_per_page": avg_bytes_per_page,
            "total_words": total_words,
            "avg_words_per_page": (avg_words_per_page * 10.0).round() / 10.0,
        },
        "quality_flags": flags,
        "files_with_issues": flags.len(),
        "files_clean": rows.len() as i64 - flags.len() as i64,
        "recommendations": session_recommendations,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates an in-memory SQLite connection pool with foreign keys enabled
    /// and the full neuroncite schema applied via migration.
    fn test_pool() -> r2d2::Pool<r2d2_sqlite::SqliteConnectionManager> {
        let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA foreign_keys = ON;")?;
            Ok(())
        });
        let pool = r2d2::Pool::builder()
            .max_size(2)
            .build(manager)
            .expect("pool build");
        {
            let conn = pool.get().expect("get conn");
            neuroncite_store::migrate(&conn).expect("migrate");
        }
        pool
    }

    /// Creates an AppState backed by a stub embedding backend and the given
    /// connection pool. The stub backend returns zero-vectors of dimension 384.
    fn test_state(pool: r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>) -> Arc<AppState> {
        use neuroncite_core::{AppConfig, EmbeddingBackend, ModelInfo, NeuronCiteError};

        struct StubBackend;
        impl EmbeddingBackend for StubBackend {
            fn name(&self) -> &str {
                "stub"
            }
            fn vector_dimension(&self) -> usize {
                384
            }
            fn load_model(&mut self, _: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                Ok(texts.iter().map(|_| vec![0.0; 384]).collect())
            }
            fn supports_gpu(&self) -> bool {
                false
            }
            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn loaded_model_id(&self) -> String {
                String::new()
            }
        }

        let backend: Arc<dyn EmbeddingBackend> = Arc::new(StubBackend);
        let worker_handle = neuroncite_api::spawn_worker(backend, None);
        AppState::new(pool, worker_handle, AppConfig::default(), true, None, 384)
            .expect("test AppState construction must succeed")
    }

    /// Returns a standard IndexConfig used across quality report tests.
    fn test_config() -> neuroncite_core::IndexConfig {
        neuroncite_core::IndexConfig {
            directory: std::path::PathBuf::from("/test/dir"),
            model_name: "test-model".to_string(),
            chunk_strategy: "word".to_string(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".to_string(),
            embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
            vector_dimension: 384,
        }
    }

    /// T-MCP-QUAL-001: Calling the quality handler without a `session_id`
    /// parameter returns an error indicating the missing required field.
    #[tokio::test]
    async fn t_mcp_qual_001_missing_session_id_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({});
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "quality report without session_id must fail"
        );
        assert!(
            result.unwrap_err().contains("session_id"),
            "error message must reference the missing session_id parameter"
        );
    }

    /// T-MCP-QUAL-002: Providing a session_id that does not exist in the
    /// database returns a "not found" error.
    #[tokio::test]
    async fn t_mcp_qual_002_nonexistent_session_returns_error() {
        let pool = test_pool();
        let state = test_state(pool);

        let params = serde_json::json!({ "session_id": 999999 });
        let result = handle(&state, &params).await;

        assert!(
            result.is_err(),
            "quality report for non-existent session must fail"
        );
        assert!(
            result.unwrap_err().contains("not found"),
            "error message must indicate the session was not found"
        );
    }

    /// T-MCP-QUAL-003: A session containing files that have no quality issues
    /// produces a report with files_with_issues=0 and files_clean equal to the
    /// number of inserted files.
    ///
    /// A "clean" file has:
    /// - page_count == pdf_page_count (no incomplete extraction or mismatch)
    /// - native_pages > ocr_pages (not OCR-heavy)
    /// - total_bytes / page_count >= 100 (sufficient text density)
    /// - empty_pages * 10 <= page_count (not many empty pages)
    #[tokio::test]
    async fn t_mcp_qual_003_clean_files_report_zero_issues() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // Insert two files, each with 5 native pages and matching pdf_page_count.
            // Each page has 500 bytes of content, well above the 100 bytes/page threshold.
            let file_a = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/clean_a.pdf",
                "aaa",
                1000,
                5000,
                5,
                Some(5),
            )
            .unwrap();

            let file_b = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/clean_b.pdf",
                "bbb",
                1000,
                3000,
                5,
                Some(5),
            )
            .unwrap();

            // Content string of 500 bytes to ensure text density is above 100 bytes/page.
            let content = "x".repeat(500);
            let pages_a: Vec<(i64, &str, &str)> = (1..=5)
                .map(|i| (i, content.as_str(), "pdf-extract"))
                .collect();
            let pages_b: Vec<(i64, &str, &str)> = (1..=5)
                .map(|i| (i, content.as_str(), "pdf-extract"))
                .collect();

            neuroncite_store::bulk_insert_pages(&conn, file_a, &pages_a).unwrap();
            neuroncite_store::bulk_insert_pages(&conn, file_b, &pages_b).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        assert_eq!(
            result["files_with_issues"], 0,
            "clean files must produce zero issues"
        );
        assert_eq!(
            result["files_clean"], 2,
            "both files must be classified as clean"
        );
        assert_eq!(
            result["extraction_summary"]["total_files"], 2,
            "total_files must match the number of inserted files"
        );
        assert_eq!(
            result["extraction_summary"]["native_text_count"], 2,
            "both files use native text extraction only"
        );
        assert_eq!(
            result["quality_flags"]
                .as_array()
                .expect("quality_flags is an array")
                .len(),
            0,
            "quality_flags array must be empty for clean files"
        );
    }

    /// T-MCP-QUAL-004: A file with page_count < pdf_page_count triggers the
    /// `incomplete_extraction` flag. The handler detects that not all pages from
    /// the structural PDF page tree were successfully extracted.
    #[tokio::test]
    async fn t_mcp_qual_004_incomplete_extraction_flag() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // pdf_page_count=10 but only 5 pages extracted: triggers incomplete_extraction.
            // Each page has 500 bytes to avoid triggering low_text_density.
            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/partial.pdf",
                "ccc",
                1000,
                8000,
                5,
                Some(10),
            )
            .unwrap();

            let content = "y".repeat(500);
            let pages: Vec<(i64, &str, &str)> = (1..=5)
                .map(|i| (i, content.as_str(), "pdf-extract"))
                .collect();
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        assert_eq!(
            result["files_with_issues"], 1,
            "one file has incomplete extraction"
        );

        let quality_flags = result["quality_flags"]
            .as_array()
            .expect("quality_flags is an array");
        assert_eq!(quality_flags.len(), 1);

        let flags = quality_flags[0]["flags"]
            .as_array()
            .expect("flags is an array");
        let flag_strings: Vec<&str> = flags.iter().filter_map(|f| f.as_str()).collect();
        assert!(
            flag_strings.contains(&"incomplete_extraction"),
            "incomplete_extraction flag must be present when page_count < pdf_page_count"
        );
    }

    /// T-MCP-QUAL-005: A file where all pages are OCR-extracted (ocr_pages >
    /// native_pages, with both > 0 not required -- only ocr_pages > native_pages
    /// and ocr_pages > 0) triggers the `ocr_heavy` flag.
    ///
    /// The test inserts a file with 3 OCR pages and 1 native page. Since
    /// ocr_pages (3) > native_pages (1) and ocr_pages > 0, the flag fires.
    #[tokio::test]
    async fn t_mcp_qual_005_ocr_heavy_flag() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // 4 pages total, page_count matches pdf_page_count to avoid
            // incomplete_extraction. 3 OCR + 1 native triggers ocr_heavy.
            // Each page has 500 bytes to avoid low_text_density.
            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/ocr_file.pdf",
                "ddd",
                1000,
                6000,
                4,
                Some(4),
            )
            .unwrap();

            let content = "z".repeat(500);
            let pages: Vec<(i64, &str, &str)> = vec![
                (1, content.as_str(), "pdf-extract"), // native
                (2, content.as_str(), "ocr"),         // ocr
                (3, content.as_str(), "ocr"),         // ocr
                (4, content.as_str(), "ocr"),         // ocr
            ];
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        assert_eq!(result["files_with_issues"], 1, "one file is OCR-heavy");

        let quality_flags = result["quality_flags"]
            .as_array()
            .expect("quality_flags is an array");
        assert_eq!(quality_flags.len(), 1);

        let flags = quality_flags[0]["flags"]
            .as_array()
            .expect("flags is an array");
        let flag_strings: Vec<&str> = flags.iter().filter_map(|f| f.as_str()).collect();
        assert!(
            flag_strings.contains(&"ocr_heavy"),
            "ocr_heavy flag must be present when ocr_pages > native_pages"
        );

        // Verify the extraction summary counts the file as mixed (has both OCR and native pages).
        assert_eq!(
            result["extraction_summary"]["mixed_count"], 1,
            "file with both native and OCR pages is classified as mixed"
        );

        // Verify per-page OCR page numbers are listed in the details.
        let details = &quality_flags[0]["details"];
        let ocr_page_numbers = details["ocr_page_numbers"]
            .as_array()
            .expect("ocr_page_numbers is an array");
        assert_eq!(ocr_page_numbers.len(), 3, "3 OCR pages must be listed");
        assert_eq!(ocr_page_numbers[0], 2);
        assert_eq!(ocr_page_numbers[1], 3);
        assert_eq!(ocr_page_numbers[2], 4);

        // Verify words_per_page is present.
        assert!(
            details["words_per_page"].as_f64().is_some(),
            "words_per_page must be present in details"
        );
    }

    /// T-MCP-QUAL-006: An empty session (no files indexed) returns an
    /// extraction_summary with all zero values and empty quality_flags.
    #[tokio::test]
    async fn t_mcp_qual_006_empty_session_returns_zeros() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap()
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        let summary = &result["extraction_summary"];
        assert_eq!(summary["total_files"], 0, "no files in session");
        assert_eq!(summary["native_text_count"], 0);
        assert_eq!(summary["ocr_required_count"], 0);
        assert_eq!(summary["mixed_count"], 0);
        assert_eq!(summary["total_pages"], 0);
        assert_eq!(summary["total_empty_pages"], 0);
        assert_eq!(summary["avg_bytes_per_page"], 0);
        assert_eq!(summary["total_words"], 0);

        assert_eq!(result["files_with_issues"], 0);
        assert_eq!(result["files_clean"], 0);
        assert_eq!(
            result["quality_flags"]
                .as_array()
                .expect("quality_flags is an array")
                .len(),
            0,
            "quality_flags must be empty for a session with no files"
        );
    }

    /// T-MCP-QUAL-007: A file with empty pages (byte_count = 0) triggers
    /// the `many_empty_pages` flag and lists the specific empty page numbers
    /// in the details.
    #[tokio::test]
    async fn t_mcp_qual_007_empty_page_numbers_listed() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // 5 pages, 3 of which are empty (byte_count = 0). With
            // empty_pages (3) * 10 > page_count (5), triggers many_empty_pages.
            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/sparse.pdf",
                "eee",
                1000,
                5000,
                5,
                Some(5),
            )
            .unwrap();

            let content = "x".repeat(500);
            let pages: Vec<(i64, &str, &str)> = vec![
                (1, content.as_str(), "pdf-extract"), // has content
                (2, "", "pdf-extract"),               // empty
                (3, content.as_str(), "pdf-extract"), // has content
                (4, "", "pdf-extract"),               // empty
                (5, "", "pdf-extract"),               // empty
            ];
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        assert_eq!(result["files_with_issues"], 1);

        let quality_flags = result["quality_flags"]
            .as_array()
            .expect("quality_flags is an array");
        let details = &quality_flags[0]["details"];

        // Verify empty_page_numbers lists pages 2, 4, 5.
        let empty_page_numbers = details["empty_page_numbers"]
            .as_array()
            .expect("empty_page_numbers is an array");
        assert_eq!(empty_page_numbers.len(), 3, "3 empty pages");
        assert_eq!(empty_page_numbers[0], 2);
        assert_eq!(empty_page_numbers[1], 4);
        assert_eq!(empty_page_numbers[2], 5);
    }

    /// T-MCP-QUAL-008: The extraction_summary includes total_words and
    /// avg_words_per_page computed from the total_bytes using the bytes/6
    /// heuristic.
    #[tokio::test]
    async fn t_mcp_qual_008_words_per_page_in_summary() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/words.pdf",
                "fff",
                1000,
                5000,
                3,
                Some(3),
            )
            .unwrap();

            // 600 bytes across 3 pages: total_words = 600/6 = 100,
            // avg_words_per_page = 100/3 = 33.3
            let content = "w".repeat(200);
            let pages: Vec<(i64, &str, &str)> = (1..=3)
                .map(|i| (i, content.as_str(), "pdf-extract"))
                .collect();
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        let summary = &result["extraction_summary"];
        assert_eq!(
            summary["total_words"], 100,
            "total_words = 600 bytes / 6 = 100"
        );
        let avg_wpg = summary["avg_words_per_page"]
            .as_f64()
            .expect("avg_words_per_page is a number");
        assert!(
            (avg_wpg - 33.3).abs() < 0.2,
            "avg_words_per_page should be ~33.3, got {avg_wpg}"
        );
    }

    /// T-MCP-QUAL-009: A file with very low text density (total_bytes /
    /// page_count < 100) triggers the `low_text_density` flag. This indicates
    /// the extraction produced very little text per page, which can happen with
    /// scanned PDFs where OCR failed or image-heavy documents.
    #[tokio::test]
    async fn t_mcp_qual_009_low_text_density_flag() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // 10 pages with only 50 bytes each = total 500 bytes.
            // Density: 500 / 10 = 50 bytes/page, below the 100 threshold.
            // pdf_page_count matches page_count to avoid incomplete_extraction.
            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/sparse_text.pdf",
                "ggg",
                1000,
                2000,
                10,
                Some(10),
            )
            .unwrap();

            let content = "x".repeat(50);
            let pages: Vec<(i64, &str, &str)> = (1..=10)
                .map(|i| (i, content.as_str(), "pdf-extract"))
                .collect();
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        assert_eq!(result["files_with_issues"], 1);

        let quality_flags = result["quality_flags"]
            .as_array()
            .expect("quality_flags is an array");
        let flags = quality_flags[0]["flags"]
            .as_array()
            .expect("flags is an array");
        let flag_strings: Vec<&str> = flags.iter().filter_map(|f| f.as_str()).collect();
        assert!(
            flag_strings.contains(&"low_text_density"),
            "low_text_density flag must be present when bytes/page < 100, got flags: {flag_strings:?}"
        );

        // Verify per-file words_per_page in details reflects the low density.
        let words_per_page = quality_flags[0]["details"]["words_per_page"]
            .as_f64()
            .expect("words_per_page is a number");
        assert!(
            words_per_page < 20.0,
            "words_per_page should be low (~8.3), got {words_per_page}"
        );
    }

    /// T-MCP-QUAL-010: A file where the extracted page count exceeds the
    /// structural PDF page count triggers the `page_count_mismatch` flag.
    /// This can occur when the extractor produces duplicate pages or when
    /// the structural page count is inconsistent with the actual content.
    #[tokio::test]
    async fn t_mcp_qual_010_page_count_mismatch_flag() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // page_count=8 but pdf_page_count=5: extracted more pages than
            // structurally declared. Each page has 500 bytes (sufficient density).
            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/extra_pages.pdf",
                "hhh",
                1000,
                8000,
                8,
                Some(5),
            )
            .unwrap();

            let content = "y".repeat(500);
            let pages: Vec<(i64, &str, &str)> = (1..=8)
                .map(|i| (i, content.as_str(), "pdf-extract"))
                .collect();
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        assert_eq!(result["files_with_issues"], 1);

        let quality_flags = result["quality_flags"]
            .as_array()
            .expect("quality_flags is an array");
        let flags = quality_flags[0]["flags"]
            .as_array()
            .expect("flags is an array");
        let flag_strings: Vec<&str> = flags.iter().filter_map(|f| f.as_str()).collect();
        assert!(
            flag_strings.contains(&"page_count_mismatch"),
            "page_count_mismatch flag must be present when page_count > pdf_page_count, got: {flag_strings:?}"
        );

        // Verify the details contain both counts for diagnostic purposes.
        let details = &quality_flags[0]["details"];
        assert_eq!(details["page_count"], 8);
        assert_eq!(details["pdf_page_count"], 5);
    }

    /// T-MCP-QUAL-011: A pure-OCR file (all pages extracted via OCR, no
    /// native text) is classified as `ocr_required_count=1` in the summary.
    /// This is distinct from `mixed_count` which requires both OCR and
    /// native pages in the same file.
    #[tokio::test]
    async fn t_mcp_qual_011_pure_ocr_file_classification() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // 3 pages, all OCR-extracted. Sufficient text density (500 bytes/page).
            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/scanned.pdf",
                "iii",
                1000,
                6000,
                3,
                Some(3),
            )
            .unwrap();

            let content = "z".repeat(500);
            let pages: Vec<(i64, &str, &str)> =
                (1..=3).map(|i| (i, content.as_str(), "ocr")).collect();
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        let summary = &result["extraction_summary"];
        assert_eq!(
            summary["ocr_required_count"], 1,
            "pure-OCR file must be counted as ocr_required, not mixed"
        );
        assert_eq!(
            summary["mixed_count"], 0,
            "a file with only OCR pages is not mixed"
        );
        assert_eq!(
            summary["native_text_count"], 0,
            "a file with only OCR pages has zero native text files"
        );

        // The file should also trigger ocr_heavy flag (ocr_pages > native_pages).
        assert_eq!(result["files_with_issues"], 1);
        let quality_flags = result["quality_flags"]
            .as_array()
            .expect("quality_flags is an array");
        let flags = quality_flags[0]["flags"]
            .as_array()
            .expect("flags is an array");
        let flag_strings: Vec<&str> = flags.iter().filter_map(|f| f.as_str()).collect();
        assert!(
            flag_strings.contains(&"ocr_heavy"),
            "pure-OCR file must trigger ocr_heavy flag"
        );
    }

    /// T-MCP-QUAL-012: avg_bytes_per_page in the extraction summary is
    /// computed as total_bytes / total_pages (integer division). Verifies
    /// the value is present and consistent with the inserted data.
    #[tokio::test]
    async fn t_mcp_qual_012_avg_bytes_per_page_in_summary() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // 4 pages, each with 300 bytes = 1200 bytes total.
            // avg_bytes_per_page = 1200 / 4 = 300.
            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/avg_bytes.pdf",
                "jjj",
                1000,
                5000,
                4,
                Some(4),
            )
            .unwrap();

            let content = "a".repeat(300);
            let pages: Vec<(i64, &str, &str)> = (1..=4)
                .map(|i| (i, content.as_str(), "pdf-extract"))
                .collect();
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        let summary = &result["extraction_summary"];
        assert_eq!(
            summary["avg_bytes_per_page"], 300,
            "avg_bytes_per_page = 1200 total bytes / 4 pages = 300"
        );
    }

    /// T-MCP-QUAL-013: A file with multiple quality issues triggers all
    /// applicable flags simultaneously. Inserts a file that has low text
    /// density AND many empty pages AND incomplete extraction, verifying
    /// all three flags appear in the same quality entry.
    #[tokio::test]
    async fn t_mcp_qual_013_multiple_flags_on_single_file() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // page_count=5 but pdf_page_count=10: triggers incomplete_extraction.
            // 3 of 5 pages are empty: 3*10 > 5, triggers many_empty_pages.
            // Only 2 pages with 30 bytes each = 60 total bytes: 60/5 = 12 < 100,
            // triggers low_text_density.
            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/troubled.pdf",
                "kkk",
                1000,
                1000,
                5,
                Some(10),
            )
            .unwrap();

            let content = "x".repeat(30);
            let pages: Vec<(i64, &str, &str)> = vec![
                (1, content.as_str(), "pdf-extract"), // has content
                (2, "", "pdf-extract"),               // empty
                (3, content.as_str(), "pdf-extract"), // has content
                (4, "", "pdf-extract"),               // empty
                (5, "", "pdf-extract"),               // empty
            ];
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        assert_eq!(result["files_with_issues"], 1);

        let quality_flags = result["quality_flags"]
            .as_array()
            .expect("quality_flags is an array");
        let flags = quality_flags[0]["flags"]
            .as_array()
            .expect("flags is an array");
        let flag_strings: Vec<&str> = flags.iter().filter_map(|f| f.as_str()).collect();

        assert!(
            flag_strings.contains(&"incomplete_extraction"),
            "incomplete_extraction must be present, got: {flag_strings:?}"
        );
        assert!(
            flag_strings.contains(&"low_text_density"),
            "low_text_density must be present, got: {flag_strings:?}"
        );
        assert!(
            flag_strings.contains(&"many_empty_pages"),
            "many_empty_pages must be present, got: {flag_strings:?}"
        );
    }

    /// T-MCP-QUAL-014: Per-file recommendations are generated for each
    /// quality flag. Each flag produces a corresponding actionable
    /// recommendation in the `recommendations` array.
    #[tokio::test]
    async fn t_mcp_qual_014_per_file_recommendations_present() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // OCR-heavy file: 3 OCR pages, 1 native page.
            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/ocr_reco.pdf",
                "lll",
                1000,
                6000,
                4,
                Some(4),
            )
            .unwrap();

            let content = "z".repeat(500);
            let pages: Vec<(i64, &str, &str)> = vec![
                (1, content.as_str(), "pdf-extract"),
                (2, content.as_str(), "ocr"),
                (3, content.as_str(), "ocr"),
                (4, content.as_str(), "ocr"),
            ];
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        let quality_flags = result["quality_flags"]
            .as_array()
            .expect("quality_flags is an array");
        assert_eq!(quality_flags.len(), 1);

        // The recommendations array must be present alongside the flags.
        let recommendations = quality_flags[0]["recommendations"]
            .as_array()
            .expect("recommendations must be an array");

        // The ocr_heavy flag must produce a recommendation mentioning Tesseract.
        assert_eq!(
            recommendations.len(),
            1,
            "one flag produces one recommendation"
        );
        let reco_text = recommendations[0].as_str().unwrap();
        assert!(
            reco_text.contains("Tesseract") || reco_text.contains("OCR"),
            "OCR-heavy recommendation must mention Tesseract or OCR, got: {reco_text}"
        );
    }

    /// T-MCP-QUAL-015: Multiple quality flags produce multiple per-file
    /// recommendations (one per flag). Verifies the 1:1 correspondence
    /// between flags and recommendations.
    #[tokio::test]
    async fn t_mcp_qual_015_recommendations_count_matches_flags() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // File with incomplete_extraction + low_text_density + many_empty_pages.
            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/multi_reco.pdf",
                "mmm",
                1000,
                1000,
                5,
                Some(10),
            )
            .unwrap();

            let content = "x".repeat(30);
            let pages: Vec<(i64, &str, &str)> = vec![
                (1, content.as_str(), "pdf-extract"),
                (2, "", "pdf-extract"),
                (3, content.as_str(), "pdf-extract"),
                (4, "", "pdf-extract"),
                (5, "", "pdf-extract"),
            ];
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        let quality_flags = result["quality_flags"]
            .as_array()
            .expect("quality_flags is an array");
        assert_eq!(quality_flags.len(), 1);

        let flags = quality_flags[0]["flags"]
            .as_array()
            .expect("flags is an array");
        let recommendations = quality_flags[0]["recommendations"]
            .as_array()
            .expect("recommendations is an array");

        assert_eq!(
            flags.len(),
            recommendations.len(),
            "each flag must produce one recommendation: flags={}, recommendations={}",
            flags.len(),
            recommendations.len()
        );

        // Verify at least 3 flags are present (incomplete, low_density, empty).
        assert!(
            flags.len() >= 3,
            "file should have at least 3 flags, got {}: {:?}",
            flags.len(),
            flags
        );
    }

    /// T-MCP-QUAL-016: Session-level recommendations array is present in
    /// the response. For a session with OCR files, it includes a
    /// recommendation about Tesseract.
    #[tokio::test]
    async fn t_mcp_qual_016_session_level_recommendations() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // OCR-only file to trigger the session-level OCR recommendation.
            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/sess_reco.pdf",
                "nnn",
                1000,
                6000,
                3,
                Some(3),
            )
            .unwrap();

            let content = "z".repeat(500);
            let pages: Vec<(i64, &str, &str)> =
                (1..=3).map(|i| (i, content.as_str(), "ocr")).collect();
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        // Session-level recommendations must be an array.
        let recommendations = result["recommendations"]
            .as_array()
            .expect("session-level recommendations must be an array");

        // At least one recommendation for the OCR file.
        assert!(
            !recommendations.is_empty(),
            "session with OCR files must produce recommendations"
        );

        // The recommendation must mention OCR/Tesseract.
        let has_ocr_reco = recommendations.iter().any(|r| {
            let text = r.as_str().unwrap_or("");
            text.contains("OCR") || text.contains("Tesseract") || text.contains("ocr_language")
        });
        assert!(
            has_ocr_reco,
            "session-level recommendations must include OCR guidance, got: {:?}",
            recommendations
        );
    }

    /// T-MCP-QUAL-017: A clean session (no quality issues) produces an
    /// empty session-level recommendations array.
    #[tokio::test]
    async fn t_mcp_qual_017_clean_session_empty_recommendations() {
        let pool = test_pool();
        let state = test_state(pool);
        let config = test_config();

        let session_id = {
            let conn = state.pool.get().unwrap();
            let sid = neuroncite_store::create_session(&conn, &config, "0.1.0").unwrap();

            // Clean file: native text, matching page count, sufficient density.
            let file_id = neuroncite_store::insert_file(
                &conn,
                sid,
                "/test/dir/clean_reco.pdf",
                "ooo",
                1000,
                5000,
                5,
                Some(5),
            )
            .unwrap();

            let content = "x".repeat(500);
            let pages: Vec<(i64, &str, &str)> = (1..=5)
                .map(|i| (i, content.as_str(), "pdf-extract"))
                .collect();
            neuroncite_store::bulk_insert_pages(&conn, file_id, &pages).unwrap();

            sid
        };

        let params = serde_json::json!({ "session_id": session_id });
        let result = handle(&state, &params).await.unwrap();

        let recommendations = result["recommendations"]
            .as_array()
            .expect("recommendations must be an array");
        assert!(
            recommendations.is_empty(),
            "clean session must produce empty recommendations, got: {:?}",
            recommendations
        );
    }
}
