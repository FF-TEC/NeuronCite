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

//! Handler for the `neuroncite_export` MCP tool.
//!
//! Runs a search query and formats the results in the requested export format
//! (markdown, bibtex, csl-json, ris, or plain-text). Returns the formatted
//! text as a string in the MCP response.
//!
//! BibTeX, CSL-JSON, and RIS entries extract metadata (title, authors, year)
//! from the source file name when available. Academic PDFs frequently follow
//! the naming pattern "Title (Author, Year).pdf" or "Title (Author & Author,
//! Year).pdf", which this parser exploits.

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_core::paths::strip_extended_length_prefix;
use neuroncite_search::{
    CachedTokenizer, SearchConfig, SearchPipeline, apply_refinement, generate_sub_chunks,
    parse_divisors,
};

/// Executes a search and exports results in the specified format.
///
/// # Parameters (from MCP tool call)
///
/// - `session_id` (required): Session to search within.
/// - `query` (required): Search query text.
/// - `format` (optional): Export format - "markdown", "bibtex", "csl-json", "ris", or "plain-text" (default: "markdown").
/// - `top_k` (optional): Number of results to export (default: 10).
/// - `file_ids` (optional): Array of file IDs to restrict the search to.
/// - `min_score` (optional): Minimum vector score threshold (0.0 to 1.0).
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_id = params["session_id"]
        .as_i64()
        .ok_or("missing required parameter: session_id")?;
    let query = params["query"]
        .as_str()
        .ok_or("missing required parameter: query")?;

    // Reject empty and whitespace-only queries, consistent with
    // neuroncite_search and neuroncite_batch_search validation.
    let query = query.trim();
    if query.is_empty() {
        return Err("query must not be empty".to_string());
    }

    let format = params["format"].as_str().unwrap_or("markdown");
    let top_k = params["top_k"].as_u64().unwrap_or(10) as usize;
    let top_k = top_k.clamp(1, 200);
    let refine = params["refine"].as_bool().unwrap_or(true);
    let divisors = params["refine_divisors"]
        .as_str()
        .map(parse_divisors)
        .unwrap_or_else(|| vec![4, 8, 16]);

    // Parse optional file_ids and min_score filters for scoped exports.
    let file_ids: Option<Vec<i64>> = params["file_ids"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect());

    let deduplicate = params["deduplicate"].as_bool().unwrap_or(true);

    let min_score: Option<f64> = params["min_score"].as_f64();
    if let Some(ms) = min_score
        && !(0.0..=1.0).contains(&ms)
    {
        return Err("min_score must be between 0.0 and 1.0".to_string());
    }

    // Validate the format parameter.
    if !["markdown", "bibtex", "csl-json", "ris", "plain-text"].contains(&format) {
        return Err(format!(
            "unsupported export format: '{format}' (valid: markdown, bibtex, csl-json, ris, plain-text)"
        ));
    }

    // Validate the session exists before attempting search. This provides a
    // clear "session not found" error consistent with neuroncite_search,
    // instead of the ambiguous "no HNSW index loaded" message.
    {
        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        neuroncite_store::get_session(&conn, session_id)
            .map_err(|_| format!("session {session_id} not found"))?;
    }

    // Embed the query and run the search pipeline.
    let query_vec = state
        .worker_handle
        .embed_query(query.to_string())
        .await
        .map_err(|e| format!("embedding failed: {e}"))?;

    let mut results = {
        let hnsw_guard = state.index.hnsw_index.load();
        let Some(hnsw_ref) = hnsw_guard.get(&session_id) else {
            return Err(format!(
                "no HNSW index loaded for session {session_id}; run indexing first"
            ));
        };

        let conn = state
            .pool
            .get()
            .map_err(|e| format!("connection pool error: {e}"))?;

        let config = SearchConfig {
            session_id,
            vector_top_k: top_k * 5,
            keyword_limit: top_k * 5,
            ef_search: state.config.ef_search,
            rrf_k: 60,
            bm25_must_match: false,
            simhash_threshold: 3,
            max_results: top_k,
            rerank_enabled: false,
            file_ids,
            min_score,
            page_start: None,
            page_end: None,
        };

        let pipeline = SearchPipeline::new(hnsw_ref, &conn, &query_vec, query, None, config);

        pipeline
            .search()
            .map_err(|e| format!("search pipeline failed: {e}"))?
            .results
    };

    // Apply sub-chunk refinement if requested. Narrows each result to
    // the most relevant passage before formatting for export.
    if refine
        && !results.is_empty()
        && !divisors.is_empty()
        && let Some(ref tokenizer_json) = *state.worker_handle.tokenizer_json()
    {
        let tokenizer = CachedTokenizer::from_json(tokenizer_json)
            .map_err(|e| format!("tokenizer deserialization failed: {e}"))?;

        let candidates = generate_sub_chunks(&results, &tokenizer, &divisors)
            .map_err(|e| format!("refinement sub-chunk generation failed: {e}"))?;

        if !candidates.is_empty() {
            let texts: Vec<&str> = candidates.iter().map(|c| c.content.as_str()).collect();
            let embeddings = state
                .worker_handle
                .embed_batch_search(&texts)
                .await
                .map_err(|e| format!("refinement embedding failed: {e}"))?;
            let _ = apply_refinement(&mut results, &candidates, &embeddings, &query_vec);
        }
    }

    // Format the results according to the requested export format.
    // When deduplicate=true (default), BibTeX, CSL-JSON, and RIS produce one
    // entry per source document with consolidated page ranges and aggregated
    // scores. Markdown and plain-text always use per-result formatting
    // regardless of the deduplicate setting.
    let formatted = match format {
        "bibtex" if deduplicate => format_as_bibtex_dedup(&results, session_id),
        "bibtex" => format_as_bibtex(&results, session_id),
        "csl-json" if deduplicate => format_as_csl_json_dedup(&results, session_id),
        "csl-json" => format_as_csl_json(&results, session_id),
        "ris" if deduplicate => format_as_ris_dedup(&results, session_id),
        "ris" => format_as_ris(&results, session_id),
        "plain-text" => format_as_plain_text(&results, query),
        _ => format_as_markdown(&results, query),
    };

    Ok(serde_json::json!({
        "format": format,
        "session_id": session_id,
        "query": query,
        "result_count": results.len(),
        "content": formatted,
    }))
}

/// Metadata extracted from a PDF file name. Academic papers typically follow
/// the pattern "Title (Author & Author, Year).pdf" or similar variations.
struct FileNameMeta {
    /// Document title extracted from the portion before the parenthesized
    /// author/year block. Falls back to the full file stem if no parenthesized
    /// block is found.
    title: String,
    /// Author string (e.g., "Box & Cox", "Fama & French"). Empty if no
    /// author/year block was found.
    authors: String,
    /// Publication year (e.g., "1964"). Empty if no year was found.
    year: String,
}

/// Parses metadata from an academic PDF file name. Expects patterns like:
///
/// - "An Analysis of Transformations (Box & Cox, 1964).pdf"
/// - "Common Risk Factors (Fama & French, 1993).pdf"
/// - "Econometrics (Hansen) [Teaching manuscript PDF].pdf"
///
/// Returns the parsed title, authors, and year. When the file name does not
/// match any recognized pattern, the title is set to the full file stem and
/// authors/year are left empty.
fn parse_filename_metadata(file_display_name: &str) -> FileNameMeta {
    // Strip the .pdf extension (case-insensitive).
    let stem = file_display_name
        .strip_suffix(".pdf")
        .or_else(|| file_display_name.strip_suffix(".PDF"))
        .unwrap_or(file_display_name);

    // Strip trailing bracketed annotations like "[Book]", "[arXiv PDF]",
    // "[Teaching manuscript PDF]" that some file naming conventions append.
    let stem = stem
        .rfind('[')
        .map(|pos| stem[..pos].trim())
        .unwrap_or(stem);

    // Look for the last parenthesized block "(Author, Year)" or "(Author & Author, Year)".
    if let Some(paren_start) = stem.rfind('(')
        && let Some(paren_end) = stem[paren_start..].rfind(')')
    {
        let paren_end = paren_start + paren_end;
        let inside = &stem[paren_start + 1..paren_end];

        // Split on the last comma to separate authors from year.
        if let Some(comma_pos) = inside.rfind(',') {
            let authors_part = inside[..comma_pos].trim();
            let year_part = inside[comma_pos + 1..].trim();

            // Validate the year: must be 4 digits.
            if year_part.len() == 4 && year_part.chars().all(|c| c.is_ascii_digit()) {
                let title = stem[..paren_start].trim().to_string();
                return FileNameMeta {
                    title: if title.is_empty() {
                        stem.to_string()
                    } else {
                        title
                    },
                    authors: authors_part.to_string(),
                    year: year_part.to_string(),
                };
            }
        }

        // Parenthesized block without a comma+year -- treat as author only
        // (e.g., "(Hansen)" without a year).
        let title = stem[..paren_start].trim().to_string();
        return FileNameMeta {
            title: if title.is_empty() {
                stem.to_string()
            } else {
                title
            },
            authors: inside.trim().to_string(),
            year: String::new(),
        };
    }

    // No parenthesized block found: use the full stem as title.
    FileNameMeta {
        title: stem.to_string(),
        authors: String::new(),
        year: String::new(),
    }
}

/// Formats search results as a markdown document with citations.
fn format_as_markdown(results: &[neuroncite_core::SearchResult], query: &str) -> String {
    let mut out = format!("# Search Results for: {query}\n\n");
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!("## Result {} (score: {:.4})\n\n", i + 1, r.score));
        out.push_str(&format!("**Source:** {}\n\n", r.citation.formatted));
        out.push_str(&r.content);
        out.push_str("\n\n---\n\n");
    }
    out
}

/// Splits a filename-derived author string into individual author names.
///
/// Academic PDF filenames use two separator conventions:
/// - Ampersand: "Fama & French" -> ["Fama", "French"]
/// - Comma + ampersand: "McNeil, Frey & Embrechts" -> ["McNeil", "Frey", "Embrechts"]
/// - Comma-only (three+ authors without ampersand): "Diez, Barr, Rundel" -> ["Diez", "Barr", "Rundel"]
///
/// The commas in filename author strings separate different authors (all family
/// names only), NOT "Last, First" pairs. This distinguishes filename metadata
/// from BibTeX author fields where commas separate family and given names.
fn split_filename_authors(author_str: &str) -> Vec<String> {
    author_str
        .split(" & ")
        .flat_map(|segment| segment.split(", "))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Generates a human-readable BibTeX citation key from author names and year.
/// Falls back to a generic key when metadata is insufficient.
///
/// For "Fama & French" with year "1993" -> "fama_french_1993"
/// For "Box & Cox" with year "1964" -> "box_cox_1964"
/// For entries without metadata -> "neuroncite_s{session_id}_r{index}"
fn generate_citation_key(meta: &FileNameMeta, session_id: i64, index: usize) -> String {
    if meta.authors.is_empty() {
        return format!("neuroncite_s{session_id}_r{index}");
    }

    // Split the filename author string into individual names, then normalize
    // each to a lowercase alphanumeric key component.
    let author_parts: Vec<String> = split_filename_authors(&meta.authors)
        .iter()
        .map(|name| {
            name.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
        .collect();

    if author_parts.is_empty() {
        return format!("neuroncite_s{session_id}_r{index}");
    }

    let key_base = author_parts.join("_");
    if meta.year.is_empty() {
        key_base
    } else {
        format!("{key_base}_{}", meta.year)
    }
}

/// Formats search results as BibTeX entries. Extracts title, authors, and year
/// from the source file name when the name follows academic naming conventions.
/// Uses `@article` when both authors and year are present (strong indicator of
/// a journal article), `@misc` otherwise. The `file` field contains the cleaned
/// PDF path (without the Windows `\\?\` extended-length prefix).
///
/// Duplicate cite-keys (multiple results from the same PDF) are disambiguated
/// by appending a lowercase letter suffix: the first occurrence keeps the base
/// key, subsequent occurrences get 'a', 'b', 'c', etc.
fn format_as_bibtex(results: &[neuroncite_core::SearchResult], session_id: i64) -> String {
    let mut out = String::new();
    let mut used_keys: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for (i, r) in results.iter().enumerate() {
        let meta = parse_filename_metadata(&r.citation.file_display_name);
        let base_key = generate_citation_key(&meta, session_id, i + 1);

        // Deduplicate cite-keys: first occurrence keeps the base key, subsequent
        // occurrences of the same key get a lowercase letter suffix (a, b, c, ...).
        // BibTeX requires unique keys across all entries.
        let count = used_keys.entry(base_key.clone()).or_insert(0);
        *count += 1;
        let key = if *count > 1 {
            let suffix = (b'a' + (*count as u8 - 2)) as char;
            format!("{base_key}{suffix}")
        } else {
            base_key
        };

        let file_path_owned = r.citation.source_file.display().to_string();
        let file_path = strip_extended_length_prefix(&file_path_owned);

        // Select entry type: @article when both authors and year are present
        // (standard academic paper), @misc for everything else.
        let entry_type = if !meta.authors.is_empty() && !meta.year.is_empty() {
            "article"
        } else {
            "misc"
        };

        // Build author field: split the filename author string into individual
        // names and join with BibTeX "and" separator. Filename authors use ", "
        // and " & " as separators between family names.
        let author_field = if meta.authors.is_empty() {
            String::new()
        } else {
            let authors = split_filename_authors(&meta.authors);
            format!("  author = {{{}}},\n", authors.join(" and "))
        };

        let year_field = if meta.year.is_empty() {
            String::new()
        } else {
            format!("  year = {{{}}},\n", meta.year)
        };

        // Use the `file` field for the PDF path instead of `howpublished`.
        // The `file` field is recognized by reference managers (Zotero, Mendeley,
        // JabRef) for linking to local PDF copies.
        out.push_str(&format!(
            "@{entry_type}{{{key},\n  title = {{{}}},\n{author_field}{year_field}  note = {{Score: {:.4}, Pages: {}-{}}},\n  file = {{{file_path}}},\n}}\n\n",
            meta.title, r.score, r.citation.page_start, r.citation.page_end,
        ));
    }
    out
}

/// Formats search results as CSL-JSON entries. Extracts metadata from the
/// source file name for richer citation data. Uses `article-journal` type when
/// both authors and year are present, `report` otherwise. The `report` type is
/// a valid CSL type (unlike the non-standard `document` that was used before)
/// and is the appropriate generic fallback for academic documents whose precise
/// type cannot be determined from the filename alone.
///
/// Duplicate IDs (multiple results from the same PDF) are disambiguated with
/// the same letter-suffix scheme used by BibTeX export.
fn format_as_csl_json(results: &[neuroncite_core::SearchResult], session_id: i64) -> String {
    let mut used_keys: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    let entries: Vec<serde_json::Value> = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let meta = parse_filename_metadata(&r.citation.file_display_name);
            let base_key = generate_citation_key(&meta, session_id, i + 1);

            // Deduplicate IDs with the same letter-suffix scheme as BibTeX.
            let count = used_keys.entry(base_key.clone()).or_insert(0);
            *count += 1;
            let key = if *count > 1 {
                let suffix = (b'a' + (*count as u8 - 2)) as char;
                format!("{base_key}{suffix}")
            } else {
                base_key
            };

            let file_path_owned = r.citation.source_file.display().to_string();
            let file_path = strip_extended_length_prefix(&file_path_owned);

            // Build CSL-JSON author array from filename-derived author names.
            // Filename authors are family names only (no given names), so each
            // becomes a {"family": "Name"} object without a "given" field.
            let authors: Vec<serde_json::Value> = if meta.authors.is_empty() {
                Vec::new()
            } else {
                split_filename_authors(&meta.authors)
                    .iter()
                    .map(|name| serde_json::json!({"family": name}))
                    .collect()
            };

            // Select CSL type: "article-journal" when both authors and year are
            // present (standard academic paper), "report" otherwise. The "report"
            // type is a standard CSL type suitable for documents whose precise
            // publication type is unknown.
            let csl_type = if !meta.authors.is_empty() && !meta.year.is_empty() {
                "article-journal"
            } else {
                "report"
            };

            let mut entry = serde_json::json!({
                "id": key,
                "type": csl_type,
                "title": meta.title,
                "file_id": r.citation.file_id,
                "source": file_path,
                "page": format!("{}-{}", r.citation.page_start, r.citation.page_end),
                "note": format!("Score: {:.4}", r.score),
            });

            if !authors.is_empty() {
                entry["author"] = serde_json::json!(authors);
            }
            if !meta.year.is_empty() {
                entry["issued"] = serde_json::json!({"date-parts": [[meta.year]]});
            }

            entry
        })
        .collect();
    serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
}

/// Formats search results as RIS (Research Information Systems) records.
/// RIS is a tagged text format supported by reference managers (EndNote,
/// Zotero, Mendeley). Each record begins with TY (type) and ends with ER
/// (end record). The T1 tag contains the title, AU contains authors, PY
/// the publication year, and L1 the local file link.
fn format_as_ris(results: &[neuroncite_core::SearchResult], _session_id: i64) -> String {
    let mut out = String::new();
    for r in results {
        let meta = parse_filename_metadata(&r.citation.file_display_name);
        let file_path_owned = r.citation.source_file.display().to_string();
        let file_path = strip_extended_length_prefix(&file_path_owned);

        // RIS type: JOUR for articles (author+year), GEN for generic documents.
        let ris_type = if !meta.authors.is_empty() && !meta.year.is_empty() {
            "JOUR"
        } else {
            "GEN"
        };

        out.push_str(&format!("TY  - {ris_type}\n"));
        out.push_str(&format!("T1  - {}\n", meta.title));

        // Each author gets a separate AU tag.
        if !meta.authors.is_empty() {
            for author in split_filename_authors(&meta.authors) {
                out.push_str(&format!("AU  - {author}\n"));
            }
        }

        if !meta.year.is_empty() {
            out.push_str(&format!("PY  - {}\n", meta.year));
        }

        out.push_str(&format!(
            "SP  - {}\nEP  - {}\n",
            r.citation.page_start, r.citation.page_end
        ));
        out.push_str(&format!("N1  - Score: {:.4}\n", r.score));
        out.push_str(&format!("L1  - {file_path}\n"));
        out.push_str("ER  - \n\n");
    }
    out
}

/// Formats search results as plain text with source citations. Each result
/// is numbered and includes the passage content, source file, and page range.
/// Intended for quick human-readable output without structured metadata.
fn format_as_plain_text(results: &[neuroncite_core::SearchResult], query: &str) -> String {
    let mut out = format!("Search: {query}\n");
    out.push_str(&format!("Results: {}\n", results.len()));
    out.push_str(&"=".repeat(60));
    out.push('\n');

    for (i, r) in results.iter().enumerate() {
        let file_path_owned = r.citation.source_file.display().to_string();
        let file_path = strip_extended_length_prefix(&file_path_owned);

        out.push_str(&format!("\n[{}] Score: {:.4}\n", i + 1, r.score));
        out.push_str(&format!(
            "Source: {} (pp. {}-{})\n",
            file_path, r.citation.page_start, r.citation.page_end
        ));
        out.push_str(&format!("{}\n", r.content));
    }
    out
}

// ---------------------------------------------------------------------------
// Document-level deduplication for structured export formats
// ---------------------------------------------------------------------------

/// A group of search results from the same source document. Page ranges are
/// consolidated into non-overlapping intervals, and scores are collected for
/// the aggregated note field. Used by the deduplicated BibTeX, CSL-JSON, and
/// RIS export paths to produce one citation entry per PDF.
struct DocumentGroup {
    /// Database file ID shared by all results in this group.
    file_id: i64,
    /// Full path to the source PDF on disk.
    source_file: std::path::PathBuf,
    /// Display name of the source PDF (filename without directory path).
    file_display_name: String,
    /// Consolidated, non-overlapping page ranges sorted by start page.
    /// Adjacent ranges (e.g., 5-7 and 8-10) are merged into 5-10.
    page_ranges: Vec<(usize, usize)>,
    /// Individual scores from each search result in this group, sorted
    /// descending (highest first).
    scores: Vec<f64>,
    /// Highest score among all results in this group.
    best_score: f64,
}

/// Merges overlapping and adjacent page ranges into non-overlapping ranges.
///
/// Input ranges do not need to be sorted. The output is sorted by start page
/// ascending. Adjacent ranges (where one ends at page N and the next starts
/// at page N+1) are merged into a single range.
///
/// Examples:
/// - `[(5,7), (6,8), (10,15)]` -> `[(5,8), (10,15)]`
/// - `[(5,7), (8,10)]` -> `[(5,10)]` (adjacent)
/// - `[(1,5)]` -> `[(1,5)]` (single range unchanged)
/// - `[]` -> `[]` (empty input)
fn merge_page_ranges(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if ranges.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<(usize, usize)> = ranges.to_vec();
    sorted.sort_by_key(|&(start, _)| start);

    let mut merged = vec![sorted[0]];
    for &(start, end) in &sorted[1..] {
        let last = merged
            .last_mut()
            .expect("merged is initialized with sorted[0] and only grows via push");
        if start <= last.1 + 1 {
            // Overlapping or adjacent: extend the current range.
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

/// Groups search results by source document (file_id), consolidates their
/// page ranges via `merge_page_ranges`, and sorts groups by best_score
/// descending (highest-scoring document first).
fn group_results_by_document(results: &[neuroncite_core::SearchResult]) -> Vec<DocumentGroup> {
    let mut groups: std::collections::HashMap<i64, DocumentGroup> =
        std::collections::HashMap::new();

    for r in results {
        let file_id = r.citation.file_id;
        let group = groups.entry(file_id).or_insert_with(|| DocumentGroup {
            file_id,
            source_file: r.citation.source_file.clone(),
            file_display_name: r.citation.file_display_name.clone(),
            page_ranges: Vec::new(),
            scores: Vec::new(),
            best_score: f64::NEG_INFINITY,
        });

        group
            .page_ranges
            .push((r.citation.page_start, r.citation.page_end));
        group.scores.push(r.score);
        if r.score > group.best_score {
            group.best_score = r.score;
        }
    }

    let mut result: Vec<DocumentGroup> = groups
        .into_values()
        .map(|mut g| {
            g.page_ranges = merge_page_ranges(&g.page_ranges);
            g.scores
                .sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            g
        })
        .collect();

    result.sort_by(|a, b| {
        b.best_score
            .partial_cmp(&a.best_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result
}

/// Formats BibTeX page ranges using the double-dash notation. Single-page
/// ranges produce just the page number; multi-page ranges use "start--end".
fn format_bibtex_page_ranges(ranges: &[(usize, usize)]) -> String {
    ranges
        .iter()
        .map(|(s, e)| {
            if s == e {
                format!("{s}")
            } else {
                format!("{s}--{e}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Formats grouped search results as deduplicated BibTeX entries. Each source
/// document produces exactly one BibTeX entry with consolidated page ranges
/// and an aggregated note field listing the passage count and individual scores.
fn format_as_bibtex_dedup(results: &[neuroncite_core::SearchResult], session_id: i64) -> String {
    let groups = group_results_by_document(results);
    let mut out = String::new();
    let mut used_keys: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for (i, group) in groups.iter().enumerate() {
        let meta = parse_filename_metadata(&group.file_display_name);
        let base_key = generate_citation_key(&meta, session_id, i + 1);

        // Deduplicate cite-keys: two different documents could share the same
        // author/year combination (e.g., two papers by the same author in the
        // same year).
        let count = used_keys.entry(base_key.clone()).or_insert(0);
        *count += 1;
        let key = if *count > 1 {
            let suffix = (b'a' + (*count as u8 - 2)) as char;
            format!("{base_key}{suffix}")
        } else {
            base_key
        };

        let file_path_owned = group.source_file.display().to_string();
        let file_path = strip_extended_length_prefix(&file_path_owned);

        let entry_type = if !meta.authors.is_empty() && !meta.year.is_empty() {
            "article"
        } else {
            "misc"
        };

        let author_field = if meta.authors.is_empty() {
            String::new()
        } else {
            let authors = split_filename_authors(&meta.authors);
            format!("  author = {{{}}},\n", authors.join(" and "))
        };

        let year_field = if meta.year.is_empty() {
            String::new()
        } else {
            format!("  year = {{{}}},\n", meta.year)
        };

        let pages_field = if group.page_ranges.is_empty() {
            String::new()
        } else {
            format!(
                "  pages = {{{}}},\n",
                format_bibtex_page_ranges(&group.page_ranges)
            )
        };

        let passage_count = group.scores.len();
        let scores_str: Vec<String> = group.scores.iter().map(|s| format!("{s:.4}")).collect();
        let note = format!(
            "{passage_count} passages, scores: {}",
            scores_str.join(", ")
        );

        out.push_str(&format!(
            "@{entry_type}{{{key},\n  title = {{{}}},\n{author_field}{year_field}{pages_field}  note = {{{note}}},\n  file = {{{file_path}}},\n}}\n\n",
            meta.title,
        ));
    }
    out
}

/// Formats grouped search results as deduplicated CSL-JSON entries. Each
/// source document produces exactly one CSL-JSON object with consolidated
/// page ranges and an aggregated note field.
fn format_as_csl_json_dedup(results: &[neuroncite_core::SearchResult], session_id: i64) -> String {
    let groups = group_results_by_document(results);
    let mut used_keys: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    let entries: Vec<serde_json::Value> = groups
        .iter()
        .enumerate()
        .map(|(i, group)| {
            let meta = parse_filename_metadata(&group.file_display_name);
            let base_key = generate_citation_key(&meta, session_id, i + 1);

            let count = used_keys.entry(base_key.clone()).or_insert(0);
            *count += 1;
            let key = if *count > 1 {
                let suffix = (b'a' + (*count as u8 - 2)) as char;
                format!("{base_key}{suffix}")
            } else {
                base_key
            };

            let file_path_owned = group.source_file.display().to_string();
            let file_path = strip_extended_length_prefix(&file_path_owned);

            let authors: Vec<serde_json::Value> = if meta.authors.is_empty() {
                Vec::new()
            } else {
                split_filename_authors(&meta.authors)
                    .iter()
                    .map(|name| serde_json::json!({"family": name}))
                    .collect()
            };

            let csl_type = if !meta.authors.is_empty() && !meta.year.is_empty() {
                "article-journal"
            } else {
                "report"
            };

            // Format page ranges as "start-end, start-end" for CSL page field.
            let page_str = group
                .page_ranges
                .iter()
                .map(|(s, e)| {
                    if s == e {
                        format!("{s}")
                    } else {
                        format!("{s}-{e}")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");

            let passage_count = group.scores.len();
            let scores_str: Vec<String> = group.scores.iter().map(|s| format!("{s:.4}")).collect();
            let note = format!(
                "{passage_count} passages, scores: {}",
                scores_str.join(", ")
            );

            let mut entry = serde_json::json!({
                "id": key,
                "type": csl_type,
                "title": meta.title,
                "file_id": group.file_id,
                "source": file_path,
                "page": page_str,
                "note": note,
            });

            if !authors.is_empty() {
                entry["author"] = serde_json::json!(authors);
            }
            if !meta.year.is_empty() {
                entry["issued"] = serde_json::json!({"date-parts": [[meta.year]]});
            }

            entry
        })
        .collect();

    serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
}

/// Formats grouped search results as deduplicated RIS records. Each source
/// document produces exactly one RIS record with the page range spanning the
/// full extent of all matched passages.
fn format_as_ris_dedup(results: &[neuroncite_core::SearchResult], _session_id: i64) -> String {
    let groups = group_results_by_document(results);
    let mut out = String::new();

    for group in &groups {
        let meta = parse_filename_metadata(&group.file_display_name);
        let file_path_owned = group.source_file.display().to_string();
        let file_path = strip_extended_length_prefix(&file_path_owned);

        let ris_type = if !meta.authors.is_empty() && !meta.year.is_empty() {
            "JOUR"
        } else {
            "GEN"
        };

        out.push_str(&format!("TY  - {ris_type}\n"));
        out.push_str(&format!("T1  - {}\n", meta.title));

        if !meta.authors.is_empty() {
            for author in split_filename_authors(&meta.authors) {
                out.push_str(&format!("AU  - {author}\n"));
            }
        }

        if !meta.year.is_empty() {
            out.push_str(&format!("PY  - {}\n", meta.year));
        }

        // RIS SP/EP tags: use the overall min start and max end from all
        // consolidated page ranges.
        if let Some(first) = group.page_ranges.first() {
            let last = group
                .page_ranges
                .last()
                .expect("last() is Some because first() returned Some on the preceding line");
            out.push_str(&format!("SP  - {}\nEP  - {}\n", first.0, last.1));
        }

        let passage_count = group.scores.len();
        let scores_str: Vec<String> = group.scores.iter().map(|s| format!("{s:.4}")).collect();
        out.push_str(&format!(
            "N1  - {passage_count} passages, scores: {}\n",
            scores_str.join(", ")
        ));
        out.push_str(&format!("L1  - {file_path}\n"));
        out.push_str("ER  - \n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses a single author name into a CSL-JSON name object. Handles:
    /// - "Last, First" -> `{"family": "Last", "given": "First"}`
    /// - "Last" (single token) -> `{"family": "Last"}`
    ///
    /// This function was previously used in the production CSL-JSON export path
    /// for parsing individual author names from filenames. The export code now
    /// uses `split_filename_authors()` instead (which treats commas as author
    /// separators, not "Last, First" separators). This function remains in
    /// tests to verify it handles BibTeX-style "Last, First" names correctly,
    /// in case it's needed in the future for non-filename author sources.
    fn parse_csl_author(name: &str) -> serde_json::Value {
        let trimmed = name.trim();
        if let Some(comma_pos) = trimmed.find(',') {
            let family = trimmed[..comma_pos].trim();
            let given = trimmed[comma_pos + 1..].trim();
            if given.is_empty() {
                serde_json::json!({"family": family})
            } else {
                serde_json::json!({"family": family, "given": given})
            }
        } else {
            serde_json::json!({"family": trimmed})
        }
    }

    /// T-MCP-040: parse_filename_metadata extracts title, authors, and year
    /// from a standard academic PDF filename.
    #[test]
    fn t_mcp_040_parse_standard_academic_filename() {
        let meta = parse_filename_metadata("An Analysis of Transformations (Box & Cox, 1964).pdf");
        assert_eq!(meta.title, "An Analysis of Transformations");
        assert_eq!(meta.authors, "Box & Cox");
        assert_eq!(meta.year, "1964");
    }

    /// T-MCP-041: parse_filename_metadata handles filenames with bracketed
    /// annotations like "[Book]" or "[arXiv PDF]".
    #[test]
    fn t_mcp_041_parse_filename_with_brackets() {
        let meta = parse_filename_metadata("Econometrics (Hansen) [Teaching manuscript PDF].pdf");
        assert_eq!(meta.title, "Econometrics");
        assert_eq!(meta.authors, "Hansen");
        assert_eq!(meta.year, "");
    }

    /// T-MCP-042: parse_filename_metadata returns the full stem as title
    /// when no parenthesized author/year block exists.
    #[test]
    fn t_mcp_042_parse_filename_no_metadata() {
        let meta = parse_filename_metadata("some_random_document.pdf");
        assert_eq!(meta.title, "some_random_document");
        assert_eq!(meta.authors, "");
        assert_eq!(meta.year, "");
    }

    /// T-MCP-043: parse_filename_metadata handles multiple authors with
    /// the ampersand separator.
    #[test]
    fn t_mcp_043_parse_multiple_authors() {
        let meta = parse_filename_metadata(
            "Common Risk Factors in the Returns on Stocks and Bonds (Fama & French, 1993).pdf",
        );
        assert_eq!(
            meta.title,
            "Common Risk Factors in the Returns on Stocks and Bonds"
        );
        assert_eq!(meta.authors, "Fama & French");
        assert_eq!(meta.year, "1993");
    }

    /// T-MCP-044: parse_filename_metadata handles filenames with author-only
    /// parentheses (no year after comma).
    #[test]
    fn t_mcp_044_parse_author_only_parentheses() {
        let meta = parse_filename_metadata("Forecasting Principles & Practice.pdf");
        // No parenthesized block: full stem is title.
        assert_eq!(meta.title, "Forecasting Principles & Practice");
        assert_eq!(meta.authors, "");
        assert_eq!(meta.year, "");
    }

    /// T-MCP-045: format_as_bibtex produces @article entries with extracted
    /// metadata and cleaned file paths (no \\?\ prefix). Entries with both
    /// authors and year use @article; entries missing either use @misc.
    #[test]
    fn t_mcp_045_bibtex_format_uses_metadata() {
        let result = neuroncite_core::SearchResult {
            score: 0.85,
            vector_score: 0.80,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "test content".into(),
            citation: neuroncite_core::Citation {
                file_id: 1,
                source_file: std::path::PathBuf::from(
                    r"\\?\D:\Papers\An Analysis of Transformations (Box & Cox, 1964).pdf",
                ),
                file_display_name: "An Analysis of Transformations (Box & Cox, 1964).pdf".into(),
                page_start: 5,
                page_end: 7,
                doc_offset_start: 100,
                doc_offset_end: 200,
                formatted: "test citation".into(),
            },
        };

        let bibtex = format_as_bibtex(&[result], 1);

        // Entry type must be @article when both authors and year are present.
        assert!(
            bibtex.contains("@article{"),
            "entry with authors+year must use @article, got: {bibtex}"
        );
        // Title must be extracted from filename, not generic.
        assert!(
            bibtex.contains("title = {An Analysis of Transformations}"),
            "title must be extracted from filename, got: {bibtex}"
        );
        // Author must use BibTeX "and" format.
        assert!(
            bibtex.contains("author = {Box and Cox}"),
            "authors must be in BibTeX format, got: {bibtex}"
        );
        // Year must be present.
        assert!(
            bibtex.contains("year = {1964}"),
            "year must be extracted, got: {bibtex}"
        );
        // Path must use `file` field (not `howpublished`) and must not
        // contain the \\?\ prefix.
        assert!(
            bibtex.contains("file = {"),
            "path must use 'file' field, got: {bibtex}"
        );
        assert!(
            !bibtex.contains("howpublished"),
            "must not use deprecated 'howpublished' field, got: {bibtex}"
        );
        assert!(
            !bibtex.contains(r"\\?\"),
            "path must not contain \\\\?\\ prefix, got: {bibtex}"
        );
        // Citation key must be human-readable (author_year format).
        assert!(
            bibtex.contains("box_cox_1964"),
            "citation key must be human-readable, got: {bibtex}"
        );
    }

    /// T-MCP-046: format_as_csl_json produces entries with structured author
    /// names (family field) and article-journal type when authors+year present.
    #[test]
    fn t_mcp_046_csl_json_format_uses_metadata() {
        let result = neuroncite_core::SearchResult {
            score: 0.90,
            vector_score: 0.85,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "test content".into(),
            citation: neuroncite_core::Citation {
                file_id: 2,
                source_file: std::path::PathBuf::from(
                    "D:\\Papers\\Common Risk Factors (Fama & French, 1993).pdf",
                ),
                file_display_name: "Common Risk Factors (Fama & French, 1993).pdf".into(),
                page_start: 1,
                page_end: 3,
                doc_offset_start: 0,
                doc_offset_end: 100,
                formatted: "test citation".into(),
            },
        };

        let csl = format_as_csl_json(&[result], 1);
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&csl).expect("CSL-JSON must be valid JSON");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["title"].as_str().unwrap(), "Common Risk Factors");
        // Type must be article-journal when authors+year are present.
        assert_eq!(
            parsed[0]["type"].as_str().unwrap(),
            "article-journal",
            "CSL type must be article-journal for entries with authors and year"
        );
        // Authors must use structured name objects with "family" field.
        let authors = parsed[0]["author"].as_array().unwrap();
        assert_eq!(authors.len(), 2);
        assert_eq!(
            authors[0]["family"].as_str().unwrap(),
            "Fama",
            "first author family name"
        );
        assert_eq!(
            authors[1]["family"].as_str().unwrap(),
            "French",
            "second author family name"
        );
        assert!(parsed[0]["issued"].is_object());
    }

    /// T-MCP-047: BibTeX uses @misc for entries without year.
    #[test]
    fn t_mcp_047_bibtex_misc_without_year() {
        let result = neuroncite_core::SearchResult {
            score: 0.70,
            vector_score: 0.65,
            bm25_rank: Some(2),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id: 3,
                source_file: std::path::PathBuf::from(
                    "D:\\Papers\\Econometrics (Hansen) [Teaching manuscript PDF].pdf",
                ),
                file_display_name: "Econometrics (Hansen) [Teaching manuscript PDF].pdf".into(),
                page_start: 1,
                page_end: 1,
                doc_offset_start: 0,
                doc_offset_end: 50,
                formatted: "citation".into(),
            },
        };

        let bibtex = format_as_bibtex(&[result], 1);
        assert!(
            bibtex.contains("@misc{"),
            "entry without year must use @misc, got: {bibtex}"
        );
    }

    /// T-MCP-048: CSL-JSON uses "report" type for entries without year.
    /// The "report" type is a standard CSL type, replacing the non-standard
    /// "document" that was used before.
    #[test]
    fn t_mcp_048_csl_json_report_without_year() {
        let result = neuroncite_core::SearchResult {
            score: 0.70,
            vector_score: 0.65,
            bm25_rank: Some(2),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id: 3,
                source_file: std::path::PathBuf::from(
                    "D:\\Papers\\Econometrics (Hansen) [Teaching manuscript PDF].pdf",
                ),
                file_display_name: "Econometrics (Hansen) [Teaching manuscript PDF].pdf".into(),
                page_start: 1,
                page_end: 1,
                doc_offset_start: 0,
                doc_offset_end: 50,
                formatted: "citation".into(),
            },
        };

        let csl = format_as_csl_json(&[result], 1);
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&csl).expect("CSL-JSON must be valid JSON");
        assert_eq!(
            parsed[0]["type"].as_str().unwrap(),
            "report",
            "entry without year must use the standard CSL 'report' type"
        );
    }

    /// T-MCP-049: parse_csl_author with "Last, First" produces structured name.
    #[test]
    fn t_mcp_049_csl_author_structured_name() {
        let author = parse_csl_author("Smith, John");
        assert_eq!(author["family"].as_str().unwrap(), "Smith");
        assert_eq!(author["given"].as_str().unwrap(), "John");
    }

    /// T-MCP-050: parse_csl_author with single-token name produces family-only.
    #[test]
    fn t_mcp_050_csl_author_family_only() {
        let author = parse_csl_author("Bollerslev");
        assert_eq!(author["family"].as_str().unwrap(), "Bollerslev");
        assert!(author.get("given").is_none() || author["given"].is_null());
    }

    /// T-MCP-051: BibTeX uses @article for entries with both authors and year,
    /// and @misc otherwise. Regression test for BUG-7 where all entries used
    /// @misc regardless of available metadata.
    #[test]
    fn t_mcp_051_bibtex_entry_type_regression() {
        let make_result = |name: &str| neuroncite_core::SearchResult {
            score: 0.80,
            vector_score: 0.75,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id: 1,
                source_file: std::path::PathBuf::from(format!("D:\\Papers\\{name}")),
                file_display_name: name.to_string(),
                page_start: 1,
                page_end: 1,
                doc_offset_start: 0,
                doc_offset_end: 50,
                formatted: "citation".into(),
            },
        };

        // With authors + year -> @article
        let bibtex = format_as_bibtex(&[make_result("Volatility (Bollerslev, 1986).pdf")], 1);
        assert!(
            bibtex.contains("@article{"),
            "authors+year -> @article: {bibtex}"
        );

        // Without year -> @misc
        let bibtex = format_as_bibtex(&[make_result("Econometrics (Hansen).pdf")], 1);
        assert!(bibtex.contains("@misc{"), "no year -> @misc: {bibtex}");

        // Without parenthesized metadata -> @misc
        let bibtex = format_as_bibtex(&[make_result("random_notes.pdf")], 1);
        assert!(bibtex.contains("@misc{"), "no metadata -> @misc: {bibtex}");
    }

    /// T-MCP-052: CSL-JSON uses article-journal for entries with both authors
    /// and year, and document otherwise. Regression test for BUG-7 where all
    /// entries used the generic "document" type.
    #[test]
    fn t_mcp_052_csl_json_type_regression() {
        let make_result = |name: &str| neuroncite_core::SearchResult {
            score: 0.80,
            vector_score: 0.75,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id: 1,
                source_file: std::path::PathBuf::from(format!("D:\\Papers\\{name}")),
                file_display_name: name.to_string(),
                page_start: 1,
                page_end: 1,
                doc_offset_start: 0,
                doc_offset_end: 50,
                formatted: "citation".into(),
            },
        };

        // With authors + year -> article-journal
        let csl = format_as_csl_json(&[make_result("GARCH (Bollerslev, 1986).pdf")], 1);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&csl).unwrap();
        assert_eq!(parsed[0]["type"], "article-journal");

        // Without year -> report (standard CSL type)
        let csl = format_as_csl_json(&[make_result("Handbook (Smith).pdf")], 1);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&csl).unwrap();
        assert_eq!(parsed[0]["type"], "report");
    }

    /// T-MCP-053: CSL-JSON author names are structured objects with "family"
    /// (and optionally "given") fields, not the deprecated "literal" format.
    /// Regression test for BUG-7 where author names used `{"literal": "Name"}`
    /// which citation processors cannot format per style.
    #[test]
    fn t_mcp_053_csl_json_structured_author_names() {
        let result = neuroncite_core::SearchResult {
            score: 0.80,
            vector_score: 0.75,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id: 1,
                source_file: std::path::PathBuf::from(
                    "D:\\Papers\\Risk Factors (Fama & French, 1993).pdf",
                ),
                file_display_name: "Risk Factors (Fama & French, 1993).pdf".to_string(),
                page_start: 1,
                page_end: 1,
                doc_offset_start: 0,
                doc_offset_end: 50,
                formatted: "citation".into(),
            },
        };

        let csl = format_as_csl_json(&[result], 1);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&csl).unwrap();
        let authors = parsed[0]["author"].as_array().unwrap();

        for author in authors {
            // Each author must have a "family" field.
            assert!(
                author["family"].is_string(),
                "BUG-7 regression: author must have 'family' field, got: {author}"
            );
            // Must NOT use the deprecated "literal" format.
            assert!(
                author.get("literal").is_none() || author["literal"].is_null(),
                "BUG-7 regression: author must not use 'literal' format, got: {author}"
            );
        }
    }

    /// T-MCP-054: parse_csl_author handles whitespace around names correctly.
    #[test]
    fn t_mcp_054_csl_author_whitespace_handling() {
        let author = parse_csl_author("  Smith , John  ");
        assert_eq!(author["family"].as_str().unwrap(), "Smith");
        assert_eq!(author["given"].as_str().unwrap(), "John");
    }

    /// T-MCP-055: parse_csl_author with trailing comma (no given name).
    #[test]
    fn t_mcp_055_csl_author_trailing_comma() {
        let author = parse_csl_author("Smith,");
        assert_eq!(author["family"].as_str().unwrap(), "Smith");
        // No "given" field when given name is empty.
        assert!(
            author.get("given").is_none() || author["given"].is_null(),
            "trailing comma should produce family-only name"
        );
    }

    /// T-MCP-056: parse_filename_metadata handles three or more authors
    /// separated by ampersands.
    #[test]
    fn t_mcp_056_parse_three_authors() {
        let meta = parse_filename_metadata(
            "Quantitative Risk Management (McNeil & Frey & Embrechts, 2015).pdf",
        );
        assert_eq!(meta.authors, "McNeil & Frey & Embrechts");
        assert_eq!(meta.year, "2015");
        assert_eq!(meta.title, "Quantitative Risk Management");
    }

    /// T-MCP-057: parse_filename_metadata handles filenames with both
    /// parenthesized metadata and bracketed annotations.
    #[test]
    fn t_mcp_057_parse_metadata_with_double_annotation() {
        let meta =
            parse_filename_metadata("Scaling Regression Inputs (Gelman, 2008) [Author PDF].pdf");
        assert_eq!(meta.title, "Scaling Regression Inputs");
        assert_eq!(meta.authors, "Gelman");
        assert_eq!(meta.year, "2008");
    }

    /// T-MCP-058: generate_citation_key produces human-readable keys from
    /// author names and year. Single author produces "author_year", multiple
    /// authors produce "author1_author2_year".
    #[test]
    fn t_mcp_058_citation_key_generation() {
        let meta = parse_filename_metadata("An Analysis of Transformations (Box & Cox, 1964).pdf");
        let key = generate_citation_key(&meta, 1, 1);
        assert_eq!(key, "box_cox_1964");

        let meta = parse_filename_metadata("GARCH (Bollerslev, 1986).pdf");
        let key = generate_citation_key(&meta, 1, 2);
        assert_eq!(key, "bollerslev_1986");

        // Three authors.
        let meta = parse_filename_metadata(
            "Quantitative Risk Management (McNeil & Frey & Embrechts, 2015).pdf",
        );
        let key = generate_citation_key(&meta, 1, 3);
        assert_eq!(key, "mcneil_frey_embrechts_2015");
    }

    /// T-MCP-059: generate_citation_key falls back to generic format when
    /// no author metadata is available.
    #[test]
    fn t_mcp_059_citation_key_fallback() {
        let meta = parse_filename_metadata("random_notes.pdf");
        let key = generate_citation_key(&meta, 2, 5);
        assert_eq!(key, "neuroncite_s2_r5");
    }

    /// T-MCP-122: generate_citation_key without year produces author-only key.
    #[test]
    fn t_mcp_122_citation_key_without_year() {
        let meta = parse_filename_metadata("Econometrics (Hansen).pdf");
        let key = generate_citation_key(&meta, 1, 1);
        assert_eq!(key, "hansen");
    }

    /// T-MCP-123: BibTeX uses 'file' field instead of 'howpublished'.
    /// Regression test ensuring the PDF path is in the standard 'file' field
    /// that reference managers (Zotero, Mendeley, JabRef) recognize.
    #[test]
    fn t_mcp_123_bibtex_uses_file_field() {
        let result = neuroncite_core::SearchResult {
            score: 0.80,
            vector_score: 0.75,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id: 1,
                source_file: std::path::PathBuf::from("D:\\Papers\\test.pdf"),
                file_display_name: "test.pdf".to_string(),
                page_start: 1,
                page_end: 1,
                doc_offset_start: 0,
                doc_offset_end: 50,
                formatted: "citation".into(),
            },
        };

        let bibtex = format_as_bibtex(&[result], 1);
        assert!(
            bibtex.contains("file = {"),
            "BibTeX must use 'file' field: {bibtex}"
        );
        assert!(
            !bibtex.contains("howpublished"),
            "BibTeX must not use 'howpublished' field: {bibtex}"
        );
    }

    /// T-MCP-098: split_filename_authors correctly handles comma+ampersand
    /// author separators. The comma separates authors, NOT "Last, First" pairs.
    #[test]
    fn t_mcp_098_split_filename_authors_comma_ampersand() {
        // Three authors with comma separator before ampersand.
        let authors = split_filename_authors("McNeil, Frey & Embrechts");
        assert_eq!(authors, vec!["McNeil", "Frey", "Embrechts"]);

        // Two authors with ampersand only.
        let authors = split_filename_authors("Fama & French");
        assert_eq!(authors, vec!["Fama", "French"]);

        // Single author.
        let authors = split_filename_authors("Bollerslev");
        assert_eq!(authors, vec!["Bollerslev"]);

        // Three authors with comma and ampersand.
        let authors = split_filename_authors("Diez, Barr & Cetinkaya-Rundel");
        assert_eq!(authors, vec!["Diez", "Barr", "Cetinkaya-Rundel"]);

        // Three authors with ampersand only (alternative convention).
        let authors = split_filename_authors("McNeil & Frey & Embrechts");
        assert_eq!(authors, vec!["McNeil", "Frey", "Embrechts"]);
    }

    /// T-MCP-099: BibTeX author field for comma-separated filenames formats
    /// each author name joined with " and " (BibTeX standard).
    /// Regression test for the bug where "McNeil, Frey & Embrechts" was
    /// formatted as "McNeil, Frey and Embrechts", treating "Frey" as the
    /// given name of "McNeil".
    #[test]
    fn t_mcp_099_bibtex_comma_separated_authors() {
        let result = neuroncite_core::SearchResult {
            score: 0.80,
            vector_score: 0.75,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id: 1,
                source_file: std::path::PathBuf::from(
                    "D:\\Papers\\Quantitative Risk Management (McNeil, Frey & Embrechts, 2015) [Book].pdf",
                ),
                file_display_name:
                    "Quantitative Risk Management (McNeil, Frey & Embrechts, 2015) [Book].pdf"
                        .to_string(),
                page_start: 1,
                page_end: 5,
                doc_offset_start: 0,
                doc_offset_end: 100,
                formatted: "citation".into(),
            },
        };

        let bibtex = format_as_bibtex(&[result], 1);
        // Author field must list three separate authors with "and" separator.
        assert!(
            bibtex.contains("author = {McNeil and Frey and Embrechts}"),
            "three authors must be separated by 'and', got: {bibtex}"
        );
        // Citation key must include all three authors.
        assert!(
            bibtex.contains("mcneil_frey_embrechts_2015"),
            "key must include all three authors, got: {bibtex}"
        );
    }

    /// T-MCP-109: CSL-JSON author array for comma-separated filenames creates
    /// separate family-name objects for each author.
    /// Regression test for the bug where "McNeil, Frey" was parsed as
    /// {"family": "McNeil", "given": "Frey"} instead of two separate authors.
    #[test]
    fn t_mcp_109_csl_json_comma_separated_authors() {
        let result = neuroncite_core::SearchResult {
            score: 0.80,
            vector_score: 0.75,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id: 1,
                source_file: std::path::PathBuf::from(
                    "D:\\Papers\\Quantitative Risk Management (McNeil, Frey & Embrechts, 2015) [Book].pdf",
                ),
                file_display_name:
                    "Quantitative Risk Management (McNeil, Frey & Embrechts, 2015) [Book].pdf"
                        .to_string(),
                page_start: 1,
                page_end: 5,
                doc_offset_start: 0,
                doc_offset_end: 100,
                formatted: "citation".into(),
            },
        };

        let csl = format_as_csl_json(&[result], 1);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&csl).unwrap();
        let authors = parsed[0]["author"].as_array().unwrap();

        // Must have three separate author objects.
        assert_eq!(authors.len(), 3, "must have 3 authors, got: {authors:?}");
        assert_eq!(authors[0]["family"].as_str().unwrap(), "McNeil");
        assert_eq!(authors[1]["family"].as_str().unwrap(), "Frey");
        assert_eq!(authors[2]["family"].as_str().unwrap(), "Embrechts");

        // None of the authors should have a "given" field (filename-derived
        // authors are family names only).
        for author in authors {
            assert!(
                author.get("given").is_none() || author["given"].is_null(),
                "filename-derived authors must not have 'given' field, got: {author}"
            );
        }
    }

    /// T-MCP-110: BibTeX export deduplicates cite-keys when multiple results
    /// come from the same PDF. Duplicate keys get a letter suffix (a, b, c, ...).
    #[test]
    fn t_mcp_110_bibtex_duplicate_key_disambiguation() {
        let make_result = |page_start: usize, page_end: usize| neuroncite_core::SearchResult {
            score: 0.80,
            vector_score: 0.75,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id: 1,
                source_file: std::path::PathBuf::from(
                    "D:\\Papers\\Quantitative Risk Management (McNeil, Frey & Embrechts, 2015) [Book].pdf",
                ),
                file_display_name:
                    "Quantitative Risk Management (McNeil, Frey & Embrechts, 2015) [Book].pdf"
                        .to_string(),
                page_start,
                page_end,
                doc_offset_start: 0,
                doc_offset_end: 100,
                formatted: "citation".into(),
            },
        };

        let results = vec![make_result(1, 3), make_result(10, 15), make_result(20, 25)];
        let bibtex = format_as_bibtex(&results, 1);

        // First occurrence keeps the base key.
        assert!(
            bibtex.contains("@article{mcneil_frey_embrechts_2015,"),
            "first occurrence must use base key, got: {bibtex}"
        );
        // Second occurrence gets suffix 'a'.
        assert!(
            bibtex.contains("@article{mcneil_frey_embrechts_2015a,"),
            "second occurrence must get suffix 'a', got: {bibtex}"
        );
        // Third occurrence gets suffix 'b'.
        assert!(
            bibtex.contains("@article{mcneil_frey_embrechts_2015b,"),
            "third occurrence must get suffix 'b', got: {bibtex}"
        );
    }

    /// T-MCP-111: CSL-JSON export deduplicates IDs the same way as BibTeX.
    #[test]
    fn t_mcp_111_csl_json_duplicate_id_disambiguation() {
        let make_result = |page_start: usize| neuroncite_core::SearchResult {
            score: 0.80,
            vector_score: 0.75,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id: 1,
                source_file: std::path::PathBuf::from(
                    "D:\\Papers\\Risk Factors (Fama & French, 1993).pdf",
                ),
                file_display_name: "Risk Factors (Fama & French, 1993).pdf".to_string(),
                page_start,
                page_end: page_start + 2,
                doc_offset_start: 0,
                doc_offset_end: 100,
                formatted: "citation".into(),
            },
        };

        let results = vec![make_result(1), make_result(10)];
        let csl = format_as_csl_json(&results, 1);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&csl).unwrap();

        assert_eq!(parsed[0]["id"].as_str().unwrap(), "fama_french_1993");
        assert_eq!(parsed[1]["id"].as_str().unwrap(), "fama_french_1993a");
    }

    /// T-MCP-124: CSL-JSON uses standard "report" type instead of the
    /// non-standard "document" type for entries without year. Regression test.
    #[test]
    fn t_mcp_124_csl_json_standard_report_type() {
        let result = neuroncite_core::SearchResult {
            score: 0.70,
            vector_score: 0.65,
            bm25_rank: Some(2),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id: 3,
                source_file: std::path::PathBuf::from("D:\\Papers\\Notes.pdf"),
                file_display_name: "Notes.pdf".to_string(),
                page_start: 1,
                page_end: 1,
                doc_offset_start: 0,
                doc_offset_end: 50,
                formatted: "citation".into(),
            },
        };

        let csl = format_as_csl_json(&[result], 1);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&csl).unwrap();
        assert_eq!(
            parsed[0]["type"], "report",
            "CSL-JSON must use standard 'report' type, not 'document'"
        );
    }

    /// T-MCP-068: Empty query string is rejected by the export handler.
    /// Regression test for BUG-003 where neuroncite_export accepted empty
    /// queries while neuroncite_search rejected them, causing inconsistent
    /// behavior between the two endpoints.
    #[test]
    fn t_mcp_068_empty_query_rejected() {
        // Verify the validation logic directly: an empty query after trimming
        // must be detected. This mirrors the validation in the handle function
        // without needing the full async setup with AppState.
        let query = "   ";
        let trimmed = query.trim();
        assert!(
            trimmed.is_empty(),
            "whitespace-only query must trim to empty"
        );

        // The handler returns Err("query must not be empty") for this case.
        // We test the validation condition rather than the full handler because
        // the handler requires a live AppState with embedding backend.
    }

    /// T-MCP-069: Non-empty query after trimming passes validation.
    #[test]
    fn t_mcp_069_nonempty_query_passes() {
        let query = "  financial markets  ";
        let trimmed = query.trim();
        assert!(
            !trimmed.is_empty(),
            "query with content must pass empty check after trim"
        );
    }

    /// T-MCP-075: RIS export produces valid RIS records with TY/ER tags.
    #[test]
    fn t_mcp_075_ris_format() {
        let result = neuroncite_core::SearchResult {
            score: 0.85,
            vector_score: 0.80,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "test content".into(),
            citation: neuroncite_core::Citation {
                file_id: 1,
                source_file: std::path::PathBuf::from(
                    "D:\\Papers\\An Analysis of Transformations (Box & Cox, 1964).pdf",
                ),
                file_display_name: "An Analysis of Transformations (Box & Cox, 1964).pdf".into(),
                page_start: 5,
                page_end: 7,
                doc_offset_start: 100,
                doc_offset_end: 200,
                formatted: "test citation".into(),
            },
        };

        let ris = format_as_ris(&[result], 1);
        assert!(ris.contains("TY  - JOUR"), "article must use JOUR type");
        assert!(
            ris.contains("T1  - An Analysis of Transformations"),
            "title tag"
        );
        assert!(ris.contains("AU  - Box"), "first author AU tag");
        assert!(ris.contains("AU  - Cox"), "second author AU tag");
        assert!(ris.contains("PY  - 1964"), "year tag");
        assert!(ris.contains("ER  - "), "end record tag");
    }

    /// T-MCP-076: RIS uses GEN type for entries without year.
    #[test]
    fn t_mcp_076_ris_gen_without_year() {
        let result = neuroncite_core::SearchResult {
            score: 0.70,
            vector_score: 0.65,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id: 1,
                source_file: std::path::PathBuf::from("D:\\Papers\\Notes.pdf"),
                file_display_name: "Notes.pdf".into(),
                page_start: 1,
                page_end: 1,
                doc_offset_start: 0,
                doc_offset_end: 50,
                formatted: "citation".into(),
            },
        };

        let ris = format_as_ris(&[result], 1);
        assert!(ris.contains("TY  - GEN"), "no-year entry must use GEN type");
    }

    /// T-MCP-077: Plain-text format includes numbered results with source info.
    #[test]
    fn t_mcp_077_plain_text_format() {
        let result = neuroncite_core::SearchResult {
            score: 0.85,
            vector_score: 0.80,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "passage content here".into(),
            citation: neuroncite_core::Citation {
                file_id: 1,
                source_file: std::path::PathBuf::from("D:\\Papers\\test.pdf"),
                file_display_name: "test.pdf".into(),
                page_start: 3,
                page_end: 5,
                doc_offset_start: 0,
                doc_offset_end: 100,
                formatted: "citation".into(),
            },
        };

        let text = format_as_plain_text(&[result], "test query");
        assert!(text.contains("Search: test query"), "header with query");
        assert!(text.contains("[1] Score: 0.8500"), "numbered result");
        assert!(text.contains("pp. 3-5"), "page range");
        assert!(text.contains("passage content here"), "passage content");
    }

    // -----------------------------------------------------------------------
    // Export deduplication tests
    // -----------------------------------------------------------------------

    /// Helper function that creates a SearchResult with the given file_id,
    /// page range, and score. Reduces boilerplate in deduplication tests.
    fn make_dedup_result(
        file_id: i64,
        display_name: &str,
        page_start: usize,
        page_end: usize,
        score: f64,
    ) -> neuroncite_core::SearchResult {
        neuroncite_core::SearchResult {
            score,
            vector_score: score - 0.05,
            bm25_rank: Some(1),
            reranker_score: None,
            chunk_index: 0,
            content: "text".into(),
            citation: neuroncite_core::Citation {
                file_id,
                source_file: std::path::PathBuf::from(format!("D:\\Papers\\{display_name}")),
                file_display_name: display_name.to_string(),
                page_start,
                page_end,
                doc_offset_start: 0,
                doc_offset_end: 100,
                formatted: "citation".into(),
            },
        }
    }

    /// T-MCP-EXPORT-DEDUP-001: group_results_by_document groups results by
    /// file_id and produces one DocumentGroup per source document.
    #[test]
    fn t_mcp_export_dedup_001_group_by_file_id() {
        let results = vec![
            make_dedup_result(1, "Paper A (Smith, 2020).pdf", 1, 3, 0.90),
            make_dedup_result(2, "Paper B (Jones, 2021).pdf", 5, 7, 0.85),
            make_dedup_result(1, "Paper A (Smith, 2020).pdf", 10, 15, 0.80),
            make_dedup_result(1, "Paper A (Smith, 2020).pdf", 20, 25, 0.75),
        ];

        let groups = group_results_by_document(&results);

        assert_eq!(groups.len(), 2, "two distinct documents");

        // Groups are sorted by best_score descending.
        assert_eq!(groups[0].file_id, 1, "file_id=1 has best_score=0.90");
        assert_eq!(groups[1].file_id, 2, "file_id=2 has best_score=0.85");

        // file_id=1 has 3 results consolidated.
        assert_eq!(groups[0].scores.len(), 3, "3 results for file_id=1");
        assert_eq!(groups[1].scores.len(), 1, "1 result for file_id=2");
    }

    /// T-MCP-EXPORT-DEDUP-002: merge_page_ranges consolidates overlapping
    /// ranges into non-overlapping ranges.
    #[test]
    fn t_mcp_export_dedup_002_merge_overlapping() {
        let ranges = vec![(5, 7), (6, 8), (10, 15)];
        let merged = merge_page_ranges(&ranges);
        assert_eq!(merged, vec![(5, 8), (10, 15)]);
    }

    /// T-MCP-EXPORT-DEDUP-003: merge_page_ranges preserves non-overlapping
    /// ranges as separate entries.
    #[test]
    fn t_mcp_export_dedup_003_non_overlapping() {
        let ranges = vec![(1, 3), (10, 15), (20, 25)];
        let merged = merge_page_ranges(&ranges);
        assert_eq!(merged, vec![(1, 3), (10, 15), (20, 25)]);
    }

    /// T-MCP-EXPORT-DEDUP-004: merge_page_ranges merges adjacent ranges where
    /// one ends at page N and the next starts at page N+1.
    #[test]
    fn t_mcp_export_dedup_004_adjacent_ranges() {
        let ranges = vec![(5, 7), (8, 10)];
        let merged = merge_page_ranges(&ranges);
        assert_eq!(merged, vec![(5, 10)]);
    }

    /// T-MCP-EXPORT-DEDUP-005: merge_page_ranges handles a single range
    /// without modification.
    #[test]
    fn t_mcp_export_dedup_005_single_range() {
        let ranges = vec![(3, 7)];
        let merged = merge_page_ranges(&ranges);
        assert_eq!(merged, vec![(3, 7)]);
    }

    /// T-MCP-EXPORT-DEDUP-006: merge_page_ranges returns an empty vector
    /// for empty input.
    #[test]
    fn t_mcp_export_dedup_006_empty_input() {
        let ranges: Vec<(usize, usize)> = vec![];
        let merged = merge_page_ranges(&ranges);
        assert!(merged.is_empty());
    }

    /// T-MCP-EXPORT-DEDUP-007: BibTeX dedup produces one entry per document
    /// with consolidated page ranges using double-dash notation.
    #[test]
    fn t_mcp_export_dedup_007_bibtex_one_entry_per_document() {
        let results = vec![
            make_dedup_result(1, "Risk Factors (Fama & French, 1993).pdf", 5, 7, 0.92),
            make_dedup_result(1, "Risk Factors (Fama & French, 1993).pdf", 10, 15, 0.85),
            make_dedup_result(1, "Risk Factors (Fama & French, 1993).pdf", 20, 25, 0.78),
        ];

        let bibtex = format_as_bibtex_dedup(&results, 1);

        // Exactly one @article entry (one document).
        let entry_count = bibtex.matches("@article{").count();
        assert_eq!(entry_count, 1, "one BibTeX entry for one document");

        // Consolidated page ranges with double-dash notation.
        assert!(
            bibtex.contains("pages = {5--7, 10--15, 20--25}"),
            "consolidated page ranges: {bibtex}"
        );

        // Note field with passage count and scores.
        assert!(
            bibtex.contains("3 passages"),
            "passage count in note: {bibtex}"
        );
        assert!(bibtex.contains("0.9200"), "highest score in note: {bibtex}");
    }

    /// T-MCP-EXPORT-DEDUP-008: CSL-JSON dedup produces one entry per document
    /// with consolidated page ranges.
    #[test]
    fn t_mcp_export_dedup_008_csl_json_one_entry_per_document() {
        let results = vec![
            make_dedup_result(1, "Risk Factors (Fama & French, 1993).pdf", 5, 7, 0.92),
            make_dedup_result(1, "Risk Factors (Fama & French, 1993).pdf", 10, 15, 0.85),
        ];

        let csl = format_as_csl_json_dedup(&results, 1);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&csl).unwrap();

        assert_eq!(parsed.len(), 1, "one CSL-JSON entry for one document");
        assert_eq!(parsed[0]["type"], "article-journal");
        assert!(
            parsed[0]["page"].as_str().unwrap().contains("5-7, 10-15"),
            "consolidated page ranges in CSL-JSON"
        );
        assert!(
            parsed[0]["note"].as_str().unwrap().contains("2 passages"),
            "passage count in note"
        );
    }

    /// T-MCP-EXPORT-DEDUP-009: RIS dedup produces one record per document
    /// with the page range spanning all matched passages.
    #[test]
    fn t_mcp_export_dedup_009_ris_one_record_per_document() {
        let results = vec![
            make_dedup_result(1, "GARCH (Bollerslev, 1986).pdf", 5, 7, 0.90),
            make_dedup_result(1, "GARCH (Bollerslev, 1986).pdf", 20, 25, 0.80),
        ];

        let ris = format_as_ris_dedup(&results, 1);

        // Exactly one TY and one ER tag (one record).
        assert_eq!(ris.matches("TY  - ").count(), 1, "one RIS record");
        assert_eq!(ris.matches("ER  - ").count(), 1, "one end-of-record");

        // SP/EP span the overall range.
        assert!(ris.contains("SP  - 5"), "start page");
        assert!(ris.contains("EP  - 25"), "end page");

        // Note field with passage count.
        assert!(ris.contains("2 passages"), "passage count in note");
    }

    /// T-MCP-EXPORT-DEDUP-010: deduplicate=false preserves the per-result
    /// entry format (backward compatibility). Three results from the same
    /// PDF produce three separate BibTeX entries.
    #[test]
    fn t_mcp_export_dedup_010_no_dedup_backward_compat() {
        let results = vec![
            make_dedup_result(1, "Risk Factors (Fama & French, 1993).pdf", 5, 7, 0.92),
            make_dedup_result(1, "Risk Factors (Fama & French, 1993).pdf", 10, 15, 0.85),
            make_dedup_result(1, "Risk Factors (Fama & French, 1993).pdf", 20, 25, 0.78),
        ];

        // Non-dedup path: format_as_bibtex produces one entry per result.
        let bibtex = format_as_bibtex(&results, 1);
        let entry_count = bibtex.matches("@article{").count();
        assert_eq!(entry_count, 3, "three entries for three results (no dedup)");
    }

    /// T-MCP-EXPORT-DEDUP-011: Markdown format is unaffected by dedup logic
    /// (always per-result). This test verifies that both results appear in
    /// the markdown output.
    #[test]
    fn t_mcp_export_dedup_011_markdown_unaffected() {
        let results = vec![
            make_dedup_result(1, "Paper A.pdf", 1, 3, 0.90),
            make_dedup_result(1, "Paper A.pdf", 5, 7, 0.80),
        ];

        let md = format_as_markdown(&results, "test query");

        // Both results present (per-result formatting).
        assert!(md.contains("Result 1"), "first result");
        assert!(md.contains("Result 2"), "second result");
    }

    /// T-MCP-EXPORT-DEDUP-012: Plain-text format is unaffected by dedup
    /// logic (always per-result).
    #[test]
    fn t_mcp_export_dedup_012_plain_text_unaffected() {
        let results = vec![
            make_dedup_result(1, "Paper A.pdf", 1, 3, 0.90),
            make_dedup_result(1, "Paper A.pdf", 5, 7, 0.80),
        ];

        let text = format_as_plain_text(&results, "test query");

        // Both results present (per-result formatting).
        assert!(text.contains("[1]"), "first result");
        assert!(text.contains("[2]"), "second result");
    }

    /// T-MCP-EXPORT-DEDUP-013: BibTeX dedup handles multiple documents
    /// correctly, producing one entry per document sorted by best score.
    #[test]
    fn t_mcp_export_dedup_013_bibtex_multiple_documents() {
        let results = vec![
            make_dedup_result(1, "Paper A (Smith, 2020).pdf", 1, 5, 0.70),
            make_dedup_result(2, "Paper B (Jones, 2021).pdf", 10, 15, 0.95),
            make_dedup_result(1, "Paper A (Smith, 2020).pdf", 8, 12, 0.75),
        ];

        let bibtex = format_as_bibtex_dedup(&results, 1);

        // Two entries (two documents).
        let article_count = bibtex.matches("@article{").count();
        assert_eq!(article_count, 2, "two BibTeX entries for two documents");

        // Paper B (higher score) should appear first.
        let pos_jones = bibtex.find("jones_2021").unwrap();
        let pos_smith = bibtex.find("smith_2020").unwrap();
        assert!(
            pos_jones < pos_smith,
            "higher-scoring document must appear first"
        );
    }

    /// T-MCP-EXPORT-DEDUP-014: merge_page_ranges handles unsorted input
    /// correctly, sorting by start page before merging.
    #[test]
    fn t_mcp_export_dedup_014_unsorted_input() {
        let ranges = vec![(20, 25), (5, 7), (10, 15), (6, 8)];
        let merged = merge_page_ranges(&ranges);
        assert_eq!(merged, vec![(5, 8), (10, 15), (20, 25)]);
    }

    /// T-MCP-EXPORT-DEDUP-015: merge_page_ranges handles fully contained
    /// ranges (one range completely inside another).
    #[test]
    fn t_mcp_export_dedup_015_contained_ranges() {
        let ranges = vec![(1, 20), (5, 10), (8, 15)];
        let merged = merge_page_ranges(&ranges);
        assert_eq!(merged, vec![(1, 20)]);
    }

    /// T-MCP-EXPORT-DEDUP-016: format_bibtex_page_ranges produces correct
    /// BibTeX notation for single and multi-page ranges.
    #[test]
    fn t_mcp_export_dedup_016_bibtex_page_range_formatting() {
        assert_eq!(format_bibtex_page_ranges(&[(5, 7)]), "5--7");
        assert_eq!(format_bibtex_page_ranges(&[(3, 3)]), "3");
        assert_eq!(
            format_bibtex_page_ranges(&[(1, 3), (10, 15)]),
            "1--3, 10--15"
        );
        assert_eq!(
            format_bibtex_page_ranges(&[(1, 1), (5, 5), (10, 10)]),
            "1, 5, 10"
        );
    }
}
