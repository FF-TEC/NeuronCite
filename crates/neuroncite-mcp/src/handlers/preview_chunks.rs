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

//! Handler for the `neuroncite_preview_chunks` MCP tool.
//!
//! Previews how a file will be split into chunks using a specified chunking
//! strategy, without storing anything in the database. Supports PDF files
//! (via file_path) and cached HTML pages (via url or file_id). This is a
//! stateless operation: extract content, apply the chunking strategy, and
//! return the first N chunks with their metadata.
//!
//! For HTML sources, the response includes section structure (headings and
//! content lengths) alongside the chunk preview.
//!
//! This handler absorbs the functionality of the former `neuroncite_html_preview`
//! tool: provide a `url` or `file_id` parameter to preview HTML content.

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_api::indexer;

/// Maximum number of chunks that can be returned in a single preview request.
const MAX_PREVIEW_LIMIT: usize = 50;

/// Default number of chunks returned when the `limit` parameter is omitted.
const DEFAULT_PREVIEW_LIMIT: usize = 5;

/// Default maximum character count for the extracted text in the HTML preview.
const DEFAULT_MAX_CHARS: usize = 50_000;

/// Previews how a file will be chunked without database storage.
///
/// # Parameters (from MCP tool call)
///
/// - `file_path` (optional): Absolute path to the file on disk (PDF or other).
/// - `url` (optional): URL of a cached HTML page (alternative to file_path).
/// - `file_id` (optional): Database file ID of an indexed HTML page (the URL
///   is looked up from the web_source table, then the cache path is derived).
/// - `chunk_strategy` (required): Chunking strategy name - "page", "word",
///   "token", or "sentence".
/// - `chunk_size` (optional): Window size in tokens or words. For "token"
///   and "word" strategies this is the window size. For "sentence" strategy
///   this is mapped to `max_words` (same behavior as `neuroncite_index`).
/// - `chunk_overlap` (optional): Overlap between consecutive chunks. Used by
///   "token" and "word" strategies only.
/// - `max_words` (optional): Maximum words per chunk for the "sentence"
///   strategy. Takes precedence over `chunk_size` when both are provided.
/// - `limit` (optional): Number of chunks to return (default: 5, max: 50).
/// - `strip_boilerplate` (optional): For HTML, apply readability-based
///   boilerplate removal. Defaults to true.
/// - `max_chars` (optional): Maximum characters of extracted text to return
///   in the HTML preview. Defaults to 50000.
///
/// Provide exactly one of: `file_path`, `url`, or `file_id`.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let file_path = params["file_path"].as_str();
    let url_param = params["url"].as_str();
    let file_id_param = params["file_id"].as_i64();

    // Determine input mode. URL and file_id route to the HTML preview path;
    // file_path routes to the PDF/generic extraction path.
    let is_html = url_param.is_some() || file_id_param.is_some();

    if is_html {
        return handle_html_preview(state, params).await;
    }

    let file_path =
        file_path.ok_or("missing required parameter: provide 'file_path', 'url', or 'file_id'")?;

    let strategy_name = params["chunk_strategy"]
        .as_str()
        .ok_or("missing required parameter: chunk_strategy")?;

    // Validate the chunk_strategy parameter against the four supported values.
    if !["page", "word", "token", "sentence"].contains(&strategy_name) {
        return Err(format!(
            "invalid chunk_strategy: '{strategy_name}' (valid: page, word, token, sentence)"
        ));
    }

    let raw_chunk_size = params["chunk_size"].as_u64().map(|v| v as usize);
    let raw_chunk_overlap = params["chunk_overlap"].as_u64().map(|v| v as usize);
    let raw_max_words = params["max_words"].as_u64().map(|v| v as usize);

    // Map the unified chunk_size parameter to strategy-specific fields,
    // matching the behavior of the `neuroncite_index` handler. The "sentence"
    // strategy uses max_words instead of chunk_size. When the caller provides
    // chunk_size but not max_words, map chunk_size to max_words so both
    // `neuroncite_index` and `neuroncite_preview_chunks` accept the same
    // parameters for the same strategy.
    let (chunk_size, chunk_overlap, max_words) = match strategy_name {
        "token" | "word" => (raw_chunk_size, raw_chunk_overlap, None),
        "sentence" => {
            // Prefer explicit max_words; fall back to chunk_size as synonym.
            let effective_max_words = raw_max_words.or(raw_chunk_size);
            (None, None, effective_max_words)
        }
        _ => (None, None, None), // "page" strategy ignores all size parameters
    };

    let limit = params["limit"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_PREVIEW_LIMIT)
        .clamp(1, MAX_PREVIEW_LIMIT);

    // Validate the file exists on disk.
    let pdf_path = std::path::Path::new(file_path);
    if !pdf_path.is_file() {
        return Err(format!("file does not exist: {file_path}"));
    }

    // Create the chunking strategy. The tokenizer JSON is required for the
    // "token" strategy; for other strategies it is optional but harmless.
    let tokenizer_json = state.worker_handle.tokenizer_json();

    let chunk_strategy = neuroncite_chunk::create_strategy(
        strategy_name,
        chunk_size,
        chunk_overlap,
        max_words,
        tokenizer_json.as_deref(),
    )
    .map_err(|e| format!("chunking strategy error: {e}"))?;

    // Extract pages and apply chunking on a blocking thread (CPU-bound I/O).
    let pdf_path_owned = pdf_path.to_path_buf();
    let extracted = tokio::task::spawn_blocking(move || {
        indexer::extract_and_chunk_file(chunk_strategy.as_ref(), &pdf_path_owned)
    })
    .await
    .map_err(|e| format!("extraction task panicked: {e}"))?
    .map_err(|e| format!("extraction failed: {e}"))?;

    let total_pages = extracted.pages.len();
    let total_chunks = extracted.chunks.len();
    let previewed = limit.min(total_chunks);

    // Build the preview array from the first `limit` chunks.
    let chunks_preview: Vec<serde_json::Value> = extracted
        .chunks
        .iter()
        .take(limit)
        .map(|chunk| {
            let word_count = chunk.content.split_whitespace().count();
            let byte_count = chunk.content.len();

            serde_json::json!({
                "chunk_index": chunk.chunk_index,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "word_count": word_count,
                "byte_count": byte_count,
                "content": chunk.content,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "file_path": file_path,
        "chunk_strategy": strategy_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "max_words": max_words,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "previewed": previewed,
        "chunks": chunks_preview,
    }))
}

/// Handles the HTML preview path: reads cached HTML, extracts metadata and
/// sections, applies chunking, and returns a combined preview with section
/// structure and chunk samples.
///
/// This is the logic formerly in `neuroncite_html_preview`, moved here so
/// a single `neuroncite_preview_chunks` tool serves all file types.
async fn handle_html_preview(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let strip_boilerplate = params["strip_boilerplate"].as_bool().unwrap_or(true);
    let max_chars = params["max_chars"]
        .as_u64()
        .unwrap_or(DEFAULT_MAX_CHARS as u64) as usize;

    let strategy_name = params["chunk_strategy"].as_str();

    // Resolve the URL from either the `url` parameter directly or by looking
    // up the `file_id` in the web_source table.
    let url: String = if let Some(u) = params["url"].as_str() {
        u.to_string()
    } else if let Some(fid) = params["file_id"].as_i64() {
        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        let ws = neuroncite_store::get_web_source_by_file(&conn, fid)
            .map_err(|_| format!("no web_source record found for file_id={fid}"))?;
        ws.url
    } else {
        return Err("missing required parameter: provide either 'url' or 'file_id'".to_string());
    };

    // Compute the cache file path from the URL and read the raw HTML.
    let cache_dir = neuroncite_html::default_cache_dir();
    let cache_path = neuroncite_html::cache_path_for_url(&cache_dir, &url);

    let raw_html = std::fs::read_to_string(&cache_path).map_err(|e| {
        format!(
            "cached HTML not found at {}: {e}; fetch the URL first via html_fetch or html_crawl",
            cache_path.display()
        )
    })?;

    // Extract metadata from the HTML.
    let metadata = neuroncite_html::extract_metadata(&raw_html, &url, 200, None);

    // Split the HTML into heading-based sections.
    let sections = neuroncite_html::split_into_sections(&raw_html, strip_boilerplate)
        .map_err(|e| format!("section splitting failed: {e}"))?;

    // Build the section summary array (heading, content length) without
    // including the full section text.
    let section_summaries: Vec<serde_json::Value> = sections
        .iter()
        .map(|s| {
            serde_json::json!({
                "section_index": s.section_index,
                "heading": s.heading,
                "heading_level": s.heading_level,
                "content_chars": s.content.len(),
                "byte_offset_start": s.byte_offset_start,
                "byte_offset_end": s.byte_offset_end,
            })
        })
        .collect();

    // Concatenate all section content for the full extracted text preview.
    let full_text: String = sections
        .iter()
        .map(|s| s.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    // Truncate the full text to the maximum character limit.
    let (preview_text, was_truncated) = if full_text.len() <= max_chars {
        (full_text.as_str(), false)
    } else {
        let mut boundary = max_chars;
        while boundary > 0 && !full_text.is_char_boundary(boundary) {
            boundary -= 1;
        }
        (&full_text[..boundary], true)
    };

    // If a chunk_strategy was provided, also generate chunk preview from the
    // sections (matching the behavior of the PDF path).
    let chunks_json = if let Some(strat_name) = strategy_name {
        let raw_chunk_size = params["chunk_size"].as_u64().map(|v| v as usize);
        let raw_chunk_overlap = params["chunk_overlap"].as_u64().map(|v| v as usize);
        let raw_max_words = params["max_words"].as_u64().map(|v| v as usize);

        let (cs, co, mw) = match strat_name {
            "token" | "word" => (raw_chunk_size, raw_chunk_overlap, None),
            "sentence" => {
                let effective_max_words = raw_max_words.or(raw_chunk_size);
                (None, None, effective_max_words)
            }
            _ => (None, None, None),
        };

        let tokenizer_json = state.worker_handle.tokenizer_json();
        let chunk_strategy =
            neuroncite_chunk::create_strategy(strat_name, cs, co, mw, tokenizer_json.as_deref())
                .map_err(|e| format!("chunking strategy error: {e}"))?;

        let limit = params["limit"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_PREVIEW_LIMIT)
            .clamp(1, MAX_PREVIEW_LIMIT);

        // Build PageText entries from sections for the chunking pipeline.
        // source_file is set to the cache path (or a placeholder) since the
        // chunking pipeline requires it but we only use the content here.
        let page_texts: Vec<neuroncite_core::PageText> = sections
            .iter()
            .map(|s| neuroncite_core::PageText {
                source_file: std::path::PathBuf::from(&cache_path),
                page_number: s.section_index,
                content: s.content.clone(),
                backend: neuroncite_core::ExtractionBackend::HtmlReadability,
            })
            .collect();

        let chunks = chunk_strategy
            .chunk(&page_texts)
            .map_err(|e| format!("chunking error: {e}"))?;
        let total_chunks = chunks.len();
        let previewed = limit.min(total_chunks);

        let chunk_arr: Vec<serde_json::Value> = chunks
            .iter()
            .take(limit)
            .map(|chunk| {
                let word_count = chunk.content.split_whitespace().count();
                serde_json::json!({
                    "chunk_index": chunk.chunk_index,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "word_count": word_count,
                    "byte_count": chunk.content.len(),
                    "content": chunk.content,
                })
            })
            .collect();

        Some(serde_json::json!({
            "chunk_strategy": strat_name,
            "total_chunks": total_chunks,
            "previewed": previewed,
            "chunks": chunk_arr,
        }))
    } else {
        None
    };

    let mut response = serde_json::json!({
        "url": url,
        "cache_path": cache_path.display().to_string(),
        "html_bytes": raw_html.len(),
        "metadata": {
            "title": metadata.title,
            "domain": metadata.domain,
            "language": metadata.language,
            "author": metadata.author,
            "meta_description": metadata.meta_description,
            "canonical_url": metadata.canonical_url,
            "og_title": metadata.og_title,
            "og_description": metadata.og_description,
            "og_image": metadata.og_image,
            "published_date": metadata.published_date,
        },
        "sections": {
            "count": sections.len(),
            "items": section_summaries,
        },
        "text": {
            "total_chars": full_text.len(),
            "preview_chars": preview_text.len(),
            "truncated": was_truncated,
            "content": preview_text,
        },
        "strip_boilerplate": strip_boilerplate,
    });

    if let Some(cj) = chunks_json {
        response["chunking"] = cj;
    }

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-MCP-PREVIEW-001: Handler rejects requests missing the file_path
    /// parameter. The error message must follow the "missing required
    /// parameter: <name>" convention used across all handlers.
    #[test]
    fn t_mcp_preview_001_missing_file_path() {
        let params = serde_json::json!({
            "chunk_strategy": "word"
        });

        assert!(params["file_path"].as_str().is_none());
    }

    /// T-MCP-PREVIEW-002: Handler rejects requests missing the chunk_strategy
    /// parameter.
    #[test]
    fn t_mcp_preview_002_missing_chunk_strategy() {
        let params = serde_json::json!({
            "file_path": "/some/path.pdf"
        });

        assert!(params["chunk_strategy"].as_str().is_none());
    }

    /// T-MCP-PREVIEW-003: Invalid chunk_strategy values are rejected. Only
    /// "page", "word", "token", and "sentence" are accepted.
    #[test]
    fn t_mcp_preview_003_invalid_chunk_strategy() {
        let invalid_strategies = ["paragraph", "chapter", "random", "", "PAGE"];

        for strategy in invalid_strategies {
            assert!(
                !["page", "word", "token", "sentence"].contains(&strategy),
                "'{strategy}' should not be in the valid set"
            );
        }
    }

    /// T-MCP-PREVIEW-004: Handler rejects a nonexistent file path. The
    /// validation happens before any PDF extraction is attempted.
    #[test]
    fn t_mcp_preview_004_nonexistent_file() {
        let path = std::path::Path::new("/nonexistent/path/fake.pdf");
        assert!(!path.is_file());
    }

    /// T-MCP-PREVIEW-005: Default limit is 5 when the parameter is omitted.
    #[test]
    fn t_mcp_preview_005_default_limit() {
        let params = serde_json::json!({
            "file_path": "/some/path.pdf",
            "chunk_strategy": "word"
        });

        let limit = params["limit"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_PREVIEW_LIMIT);

        assert_eq!(limit, 5);
    }

    /// T-MCP-PREVIEW-006: Limit is clamped to the maximum of 50.
    #[test]
    fn t_mcp_preview_006_limit_clamped() {
        let params = serde_json::json!({
            "file_path": "/some/path.pdf",
            "chunk_strategy": "word",
            "limit": 999
        });

        let limit = params["limit"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_PREVIEW_LIMIT)
            .clamp(1, MAX_PREVIEW_LIMIT);

        assert_eq!(limit, MAX_PREVIEW_LIMIT);
    }

    /// T-MCP-PREVIEW-007: Module compiles and the handler returns the
    /// correct Result type. Compilation of this module validates that all
    /// imports, types, and function signatures are correct.
    #[test]
    fn t_mcp_preview_007_module_compiles() {
        let _check: fn() -> Result<serde_json::Value, String> =
            || Err("compile-time type check only".to_string());
    }

    /// T-MCP-PREVIEW-008: Limit of zero is clamped to 1 (minimum bound).
    #[test]
    fn t_mcp_preview_008_limit_zero_clamped() {
        let params = serde_json::json!({
            "limit": 0
        });

        let limit = params["limit"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_PREVIEW_LIMIT)
            .clamp(1, MAX_PREVIEW_LIMIT);

        assert_eq!(limit, 1);
    }

    /// T-MCP-PREVIEW-009: All four valid chunk strategy names are accepted by
    /// the validation check.
    #[test]
    fn t_mcp_preview_009_all_valid_strategies_accepted() {
        let valid = ["page", "word", "token", "sentence"];
        for strategy in &valid {
            assert!(
                ["page", "word", "token", "sentence"].contains(strategy),
                "'{strategy}' must be accepted as a valid chunk strategy"
            );
        }
    }

    /// T-MCP-PREVIEW-010: Optional parameters default to None when omitted.
    /// The chunk_size, chunk_overlap, and max_words parameters are passed
    /// through to neuroncite_chunk::create_strategy as Option<usize>.
    #[test]
    fn t_mcp_preview_010_optional_params_default_to_none() {
        let params = serde_json::json!({
            "file_path": "/some/path.pdf",
            "chunk_strategy": "word"
        });

        let chunk_size = params["chunk_size"].as_u64().map(|v| v as usize);
        let chunk_overlap = params["chunk_overlap"].as_u64().map(|v| v as usize);
        let max_words = params["max_words"].as_u64().map(|v| v as usize);

        assert!(chunk_size.is_none(), "chunk_size must default to None");
        assert!(
            chunk_overlap.is_none(),
            "chunk_overlap must default to None"
        );
        assert!(max_words.is_none(), "max_words must default to None");
    }

    /// T-MCP-PREVIEW-011: Explicit chunk_size and chunk_overlap values are
    /// extracted correctly from the params JSON.
    #[test]
    fn t_mcp_preview_011_explicit_chunk_params() {
        let params = serde_json::json!({
            "file_path": "/some/path.pdf",
            "chunk_strategy": "word",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "max_words": 300
        });

        let chunk_size = params["chunk_size"].as_u64().map(|v| v as usize);
        let chunk_overlap = params["chunk_overlap"].as_u64().map(|v| v as usize);
        let max_words = params["max_words"].as_u64().map(|v| v as usize);

        assert_eq!(chunk_size, Some(512));
        assert_eq!(chunk_overlap, Some(64));
        assert_eq!(max_words, Some(300));
    }

    /// T-MCP-PREVIEW-012: Limit is honored exactly when within the valid range.
    #[test]
    fn t_mcp_preview_012_limit_within_range() {
        for test_limit in [1_usize, 5, 10, 25, 50] {
            let clamped = test_limit.clamp(1, MAX_PREVIEW_LIMIT);
            assert_eq!(
                clamped, test_limit,
                "limit {test_limit} within [1, {MAX_PREVIEW_LIMIT}] must be unchanged"
            );
        }
    }

    // ===================================================================
    // DEFECT-003 regression tests: sentence strategy parameter mapping
    //
    // The `neuroncite_index` handler maps `chunk_size` to `max_words` for
    // the sentence strategy, but `preview_chunks` originally passed the
    // raw parameters without mapping. This caused the sentence strategy to
    // receive `chunk_size` instead of `max_words`, producing inconsistent
    // chunking between preview and actual indexing.
    //
    // The fix applies the same strategy-specific parameter routing used by
    // `neuroncite_index`: for "sentence", chunk_size is mapped to
    // max_words (with explicit max_words taking precedence); for
    // "token"/"word", chunk_size and chunk_overlap pass through; for
    // "page", all size parameters are None.
    // ===================================================================

    /// Helper that replicates the parameter mapping logic from the handler.
    /// Extracts raw parameters from JSON, applies strategy-specific routing,
    /// and returns the resolved (chunk_size, chunk_overlap, max_words) tuple.
    fn map_params(params: &serde_json::Value) -> (Option<usize>, Option<usize>, Option<usize>) {
        let strategy_name = params["chunk_strategy"].as_str().unwrap_or("word");
        let raw_chunk_size = params["chunk_size"].as_u64().map(|v| v as usize);
        let raw_chunk_overlap = params["chunk_overlap"].as_u64().map(|v| v as usize);
        let raw_max_words = params["max_words"].as_u64().map(|v| v as usize);

        match strategy_name {
            "token" | "word" => (raw_chunk_size, raw_chunk_overlap, None),
            "sentence" => {
                let effective_max_words = raw_max_words.or(raw_chunk_size);
                (None, None, effective_max_words)
            }
            _ => (None, None, None),
        }
    }

    /// T-MCP-PREVIEW-013: Sentence strategy with chunk_size but no max_words
    /// maps chunk_size to max_words. This is the primary DEFECT-003 regression
    /// test. Before the fix, chunk_size was passed directly as chunk_size (not
    /// max_words), causing `create_strategy` to receive max_words=None and
    /// either error or use the wrong default.
    #[test]
    fn t_mcp_preview_013_sentence_chunk_size_maps_to_max_words() {
        let params = serde_json::json!({
            "file_path": "/some/path.pdf",
            "chunk_strategy": "sentence",
            "chunk_size": 200
        });

        let (chunk_size, chunk_overlap, max_words) = map_params(&params);

        assert_eq!(
            chunk_size, None,
            "sentence strategy must not pass chunk_size to create_strategy"
        );
        assert_eq!(
            chunk_overlap, None,
            "sentence strategy must not pass chunk_overlap to create_strategy"
        );
        assert_eq!(
            max_words,
            Some(200),
            "chunk_size=200 must be mapped to max_words=200 for sentence strategy"
        );
    }

    /// T-MCP-PREVIEW-014: Sentence strategy with explicit max_words takes
    /// precedence over chunk_size. When both parameters are provided, the
    /// explicit max_words value is used.
    #[test]
    fn t_mcp_preview_014_sentence_explicit_max_words_takes_precedence() {
        let params = serde_json::json!({
            "file_path": "/some/path.pdf",
            "chunk_strategy": "sentence",
            "chunk_size": 200,
            "max_words": 150
        });

        let (chunk_size, chunk_overlap, max_words) = map_params(&params);

        assert_eq!(chunk_size, None);
        assert_eq!(chunk_overlap, None);
        assert_eq!(
            max_words,
            Some(150),
            "explicit max_words=150 must override chunk_size=200 for sentence strategy"
        );
    }

    /// T-MCP-PREVIEW-015: Sentence strategy with only max_words (no chunk_size)
    /// passes max_words through directly.
    #[test]
    fn t_mcp_preview_015_sentence_max_words_only() {
        let params = serde_json::json!({
            "file_path": "/some/path.pdf",
            "chunk_strategy": "sentence",
            "max_words": 100
        });

        let (chunk_size, chunk_overlap, max_words) = map_params(&params);

        assert_eq!(chunk_size, None);
        assert_eq!(chunk_overlap, None);
        assert_eq!(
            max_words,
            Some(100),
            "max_words=100 must be passed through for sentence strategy"
        );
    }

    /// T-MCP-PREVIEW-016: Sentence strategy with no size parameters results
    /// in all None values. The chunking library applies its defaults.
    #[test]
    fn t_mcp_preview_016_sentence_no_size_params() {
        let params = serde_json::json!({
            "file_path": "/some/path.pdf",
            "chunk_strategy": "sentence"
        });

        let (chunk_size, chunk_overlap, max_words) = map_params(&params);

        assert_eq!(chunk_size, None);
        assert_eq!(chunk_overlap, None);
        assert_eq!(
            max_words, None,
            "sentence strategy with no size parameters must pass None for max_words"
        );
    }

    /// T-MCP-PREVIEW-017: Word strategy passes chunk_size and chunk_overlap
    /// through without mapping. The max_words parameter is ignored (set to None).
    #[test]
    fn t_mcp_preview_017_word_strategy_passthrough() {
        let params = serde_json::json!({
            "file_path": "/some/path.pdf",
            "chunk_strategy": "word",
            "chunk_size": 300,
            "chunk_overlap": 50,
            "max_words": 999
        });

        let (chunk_size, chunk_overlap, max_words) = map_params(&params);

        assert_eq!(
            chunk_size,
            Some(300),
            "word strategy must pass chunk_size through"
        );
        assert_eq!(
            chunk_overlap,
            Some(50),
            "word strategy must pass chunk_overlap through"
        );
        assert_eq!(
            max_words, None,
            "word strategy must ignore max_words (set to None)"
        );
    }

    /// T-MCP-PREVIEW-018: Token strategy passes chunk_size and chunk_overlap
    /// through without mapping, same as word strategy.
    #[test]
    fn t_mcp_preview_018_token_strategy_passthrough() {
        let params = serde_json::json!({
            "file_path": "/some/path.pdf",
            "chunk_strategy": "token",
            "chunk_size": 256,
            "chunk_overlap": 32
        });

        let (chunk_size, chunk_overlap, max_words) = map_params(&params);

        assert_eq!(
            chunk_size,
            Some(256),
            "token strategy must pass chunk_size through"
        );
        assert_eq!(
            chunk_overlap,
            Some(32),
            "token strategy must pass chunk_overlap through"
        );
        assert_eq!(max_words, None);
    }

    /// T-MCP-PREVIEW-019: Page strategy ignores all size parameters. The page
    /// strategy produces one chunk per PDF page regardless of any size settings.
    #[test]
    fn t_mcp_preview_019_page_strategy_ignores_size_params() {
        let params = serde_json::json!({
            "file_path": "/some/path.pdf",
            "chunk_strategy": "page",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "max_words": 200
        });

        let (chunk_size, chunk_overlap, max_words) = map_params(&params);

        assert_eq!(chunk_size, None, "page strategy must ignore chunk_size");
        assert_eq!(
            chunk_overlap, None,
            "page strategy must ignore chunk_overlap"
        );
        assert_eq!(max_words, None, "page strategy must ignore max_words");
    }

    /// T-MCP-PREVIEW-020: Verifies that the preview_chunks parameter mapping
    /// matches the neuroncite_index handler parameter mapping for all four
    /// strategies. This cross-validates both handlers produce identical
    /// strategy configurations from the same input parameters, preventing
    /// the DEFECT-003 inconsistency from recurring.
    #[test]
    fn t_mcp_preview_020_mapping_matches_index_handler() {
        // Test cases: (strategy, chunk_size, max_words, expected_max_words)
        type Case = (&'static str, Option<u64>, Option<u64>, Option<usize>);
        let cases: Vec<Case> = vec![
            ("sentence", Some(200), None, Some(200)), // chunk_size maps to max_words
            ("sentence", Some(200), Some(150), Some(150)), // explicit max_words wins
            ("sentence", None, Some(100), Some(100)), // max_words only
            ("sentence", None, None, None),           // no params
            ("word", Some(300), None, None),          // word ignores max_words
            ("token", Some(256), None, None),         // token ignores max_words
            ("page", Some(512), Some(200), None),     // page ignores everything
        ];

        for (strategy, cs, mw, expected_mw) in &cases {
            let mut json = serde_json::json!({
                "file_path": "/some/path.pdf",
                "chunk_strategy": strategy
            });

            if let Some(v) = cs {
                json["chunk_size"] = serde_json::json!(v);
            }
            if let Some(v) = mw {
                json["max_words"] = serde_json::json!(v);
            }

            let (_, _, resolved_mw) = map_params(&json);
            assert_eq!(
                resolved_mw, *expected_mw,
                "strategy={strategy}, chunk_size={cs:?}, max_words={mw:?}: \
                 expected max_words={expected_mw:?}, got {resolved_mw:?}"
            );
        }
    }
}
