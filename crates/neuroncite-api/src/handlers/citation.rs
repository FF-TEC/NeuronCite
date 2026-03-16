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

//! Citation verification endpoint handlers.
//!
//! Implements five endpoints that form the complete citation verification
//! workflow: create -> claim -> submit -> status -> export. Each handler
//! extracts parameters from axum extractors, delegates to the
//! neuroncite-citation crate for business logic, and returns structured
//! JSON responses. The same business logic is shared with the MCP handlers.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};

use neuroncite_citation::{
    Alert, CitationJobParams, CitationRow, CitationRowInsert, CorrectionType, SubmitEntry,
};
use neuroncite_llm::ollama::OllamaBackend;
use neuroncite_store::{self as store, JobState};

use crate::agent::{
    AgentConfig, AutoVerifyRequest, AutoVerifyResponse, CitationAgent, llm_config_from_request,
};

use crate::dto::{
    API_VERSION, CitationClaimRequest, CitationClaimResponse, CitationCreateRequest,
    CitationExportRequest, CitationExportResponse, CitationRowsQuery, CitationRowsResponse,
    CitationStatusResponse, CitationSubmitRequest, CitationSubmitResponse,
};
use crate::error::ApiError;
use crate::state::AppState;

/// Validates a URL from a BibTeX entry before issuing an HTTP request.
///
/// Rejects any URL that:
/// - Cannot be parsed as a valid URL.
/// - Uses a scheme other than `http` or `https` (e.g. `file://`, `ftp://`).
/// - Has a host that matches a loopback address, link-local range, or
///   private IP range. This is a static host-string check; it does not
///   perform DNS resolution.
///
/// A crafted BibTeX `url` field pointing at `http://localhost:9200/` or
/// `http://169.254.169.254/latest/meta-data/` would allow the server to act
/// as an SSRF proxy. This function prevents that.
fn validate_fetch_url(url: &str) -> Result<(), ApiError> {
    let parsed = url::Url::parse(url).map_err(|_| ApiError::BadRequest {
        reason: "BibTeX entry contains an invalid URL that cannot be parsed".to_string(),
    })?;

    match parsed.scheme() {
        "http" | "https" => {}
        other => {
            return Err(ApiError::BadRequest {
                reason: format!(
                    "URL scheme '{other}' is not permitted in fetch_sources; \
                     only http and https are allowed"
                ),
            });
        }
    }

    if let Some(host) = parsed.host_str() {
        block_internal_host(host)?;
    }

    Ok(())
}

/// Returns an error when the given host string matches a private, loopback, or
/// link-local address. Checked via string prefix comparison without DNS resolution.
fn block_internal_host(host: &str) -> Result<(), ApiError> {
    let lower = host.to_ascii_lowercase();

    // Loopback and special hostnames.
    if lower == "localhost" || lower == "ip6-localhost" || lower == "ip6-loopback" {
        return Err(ApiError::BadRequest {
            reason: "requests to localhost are not permitted in fetch_sources".to_string(),
        });
    }

    // Strip optional brackets from IPv6 literals (e.g. "[::1]" -> "::1").
    let addr_str = lower.trim_start_matches('[').trim_end_matches(']');

    // IPv4 loopback: 127.0.0.0/8
    if addr_str.starts_with("127.") {
        return Err(ApiError::BadRequest {
            reason: "requests to the loopback address range are not permitted".to_string(),
        });
    }
    // RFC 1918 private ranges.
    if addr_str.starts_with("10.") {
        return Err(ApiError::BadRequest {
            reason: "requests to private address range 10.0.0.0/8 are not permitted".to_string(),
        });
    }
    if addr_str.starts_with("192.168.") {
        return Err(ApiError::BadRequest {
            reason: "requests to private address range 192.168.0.0/16 are not permitted"
                .to_string(),
        });
    }
    // RFC 1918 172.16.0.0/12 covers 172.16.x.x through 172.31.x.x.
    if let Some(rest) = addr_str.strip_prefix("172.")
        && let Some(octet_str) = rest.split('.').next()
        && let Ok(octet) = octet_str.parse::<u8>()
        && (16..=31).contains(&octet)
    {
        return Err(ApiError::BadRequest {
            reason: "requests to private address range 172.16.0.0/12 are not permitted".to_string(),
        });
    }
    // Link-local range: 169.254.0.0/16 (includes AWS EC2 instance metadata endpoint).
    if addr_str.starts_with("169.254.") {
        return Err(ApiError::BadRequest {
            reason: "requests to link-local range 169.254.0.0/16 are not permitted \
                     (includes cloud metadata endpoints)"
                .to_string(),
        });
    }
    // IPv6 loopback.
    if addr_str == "::1" {
        return Err(ApiError::BadRequest {
            reason: "requests to the IPv6 loopback address are not permitted".to_string(),
        });
    }
    // IPv6 unique local addresses: fc00::/7 (fc00:: and fd00::).
    if addr_str.starts_with("fd") || addr_str.starts_with("fc") {
        return Err(ApiError::BadRequest {
            reason: "requests to IPv6 unique local addresses are not permitted".to_string(),
        });
    }
    // Unspecified addresses.
    if addr_str == "0.0.0.0" || addr_str == "::" {
        return Err(ApiError::BadRequest {
            reason: "requests to the unspecified address are not permitted".to_string(),
        });
    }

    Ok(())
}

/// POST /api/v1/citation/create
///
/// Parses the .tex file for citation commands, resolves cite-keys against
/// the .bib file, matches cited works to indexed PDFs using token-overlap
/// similarity, assigns citations to batches, and inserts all citation_row
/// records into the database. Creates a job with kind = "citation_verify"
/// and state = "running" (agent-driven, not executor-driven).
///
/// Dry-run mode: When `dry_run` is true, the pipeline runs without creating
/// a job or inserting rows. Returns a match preview with per-cite-key data
/// so incorrect matches can be identified and corrected via `file_overrides`.
///
/// File overrides: A map of cite_key -> file_id that overrides automatic
/// PDF matching. Applied after the automatic matching step.
#[utoipa::path(
    post,
    path = "/api/v1/citation/create",
    request_body = CitationCreateRequest,
    responses(
        (status = 202, description = "Citation verification job created"),
        (status = 200, description = "Dry-run match preview"),
        (status = 400, description = "Invalid request parameters"),
        (status = 404, description = "Session not found"),
    )
)]
pub async fn create(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CitationCreateRequest>,
) -> Result<(StatusCode, HeaderMap, Json<serde_json::Value>), ApiError> {
    // Validate required fields via the DTO validate() method, which returns
    // ApiError::UnprocessableEntity for structural request errors (empty paths,
    // zero batch_size). This replaces the inline validation that was previously
    // duplicated between the REST and MCP handlers.
    req.validate()?;

    let batch_size = req.batch_size.unwrap_or(5);

    let dry_run = req.dry_run.unwrap_or(false);
    let file_overrides = req.file_overrides.unwrap_or_default();

    // Resolve the effective session ID list. When `session_ids` is provided
    // and non-empty, it overrides the single `session_id` field. This
    // supports multi-session citation verification where files from multiple
    // index sessions are aggregated for cite-key matching.
    let effective_session_ids: Vec<i64> = match req.session_ids {
        Some(ref ids) if !ids.is_empty() => ids.clone(),
        _ => vec![req.session_id],
    };

    // Path containment: verify tex and bib file paths are within the allowed
    // roots. validate_path_access returns the canonicalized path to prevent
    // TOCTOU attacks where a symlink is swapped between validation and read.
    let canonical_tex =
        crate::util::validate_path_access(std::path::Path::new(&req.tex_path), &state.config)?;
    let canonical_bib =
        crate::util::validate_path_access(std::path::Path::new(&req.bib_path), &state.config)?;

    // Validate file paths exist using the canonical paths.
    if !canonical_tex.is_file() {
        return Err(ApiError::BadRequest {
            reason: format!("tex file does not exist: {}", req.tex_path),
        });
    }
    if !canonical_bib.is_file() {
        return Err(ApiError::BadRequest {
            reason: format!("bib file does not exist: {}", req.bib_path),
        });
    }

    // Read and parse files on a blocking thread to avoid stalling the tokio
    // event loop. std::fs::read_to_string and the LaTeX/BibTeX parsers perform
    // synchronous I/O and CPU-bound text processing that must not run on the
    // async executor's cooperative threads. Uses the canonical paths from
    // validate_path_access to prevent TOCTOU symlink swaps.
    let tex_path_owned = canonical_tex;
    let bib_path_owned = canonical_bib;

    let (raw_citations, bib_entries) = tokio::task::spawn_blocking(move || {
        let tex_content = std::fs::read_to_string(&tex_path_owned)
            .map_err(|e| format!("failed to read tex file: {e}"))?;
        let bib_content = std::fs::read_to_string(&bib_path_owned)
            .map_err(|e| format!("failed to read bib file: {e}"))?;

        let mut raw_citations = neuroncite_citation::latex::parse_citations(&tex_content);
        if raw_citations.is_empty() {
            return Err("no citation commands found in the LaTeX file".to_string());
        }

        let bib_entries = neuroncite_citation::bibtex::parse_bibtex(&bib_content);
        neuroncite_citation::batch::assign_batches(&mut raw_citations, batch_size);

        Ok((raw_citations, bib_entries))
    })
    .await
    .map_err(|e| ApiError::Internal {
        reason: format!("blocking task panicked: {e}"),
    })?
    .map_err(|e: String| ApiError::Internal { reason: e })?;

    let conn = state.pool.get().map_err(ApiError::from)?;

    // Validate all sessions exist.
    for &sid in &effective_session_ids {
        store::get_session(&conn, sid).map_err(|_| ApiError::NotFound {
            resource: format!("session {}", sid),
        })?;
    }

    // Match cite-keys to indexed files using token-overlap on the display name.
    // When multiple sessions are selected, files from all sessions are
    // aggregated into a single lookup list so cite-keys can match files
    // regardless of which session they belong to.
    let mut all_indexed_files = Vec::new();
    for &sid in &effective_session_ids {
        let session_files =
            store::list_files_by_session(&conn, sid).map_err(|e| ApiError::Internal {
                reason: format!("listing files for session {sid}: {e}"),
            })?;
        all_indexed_files.extend(session_files);
    }

    // Build a map of file_id -> web_source title for HTML-sourced files so
    // they can be matched by page title rather than the URL stem. The URL
    // stored in indexed_file.file_path for HTML entries (e.g.
    // "https://arxiv.org/abs/1706.03762") produces a meaningless file_stem
    // that yields zero token overlap with BibTeX metadata. The web_source
    // title (e.g. "Attention Is All You Need") contains the actual paper
    // title words and matches the BibTeX title field reliably.
    let mut html_titles: std::collections::HashMap<i64, String> = std::collections::HashMap::new();
    for &sid in &effective_session_ids {
        match store::list_web_sources(&conn, sid) {
            Ok(sources) => {
                for ws in sources {
                    // Prefer og_title over title because og_title is more
                    // likely to contain only the article title without
                    // domain-specific suffixes (e.g. "| Nature" or "- ACL").
                    let display = ws.og_title.or(ws.title).unwrap_or_default();
                    if !display.is_empty() {
                        html_titles.insert(ws.file_id, display);
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    session_id = sid,
                    "failed to load web_source titles for HTML matching: {e}"
                );
            }
        }
    }

    let file_lookup: Vec<(i64, String)> = all_indexed_files
        .iter()
        .map(|f| {
            // For HTML-sourced files, use the web_source title when available.
            // For PDF files (and HTML files without a web_source title), fall
            // back to the file stem of the stored path.
            let display_name = if f.source_type == "html" {
                html_titles
                    .get(&f.id)
                    .map(|t| t.to_lowercase())
                    .unwrap_or_else(|| {
                        std::path::Path::new(&f.file_path)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_lowercase()
                    })
            } else {
                std::path::Path::new(&f.file_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_lowercase()
            };
            (f.id, display_name)
        })
        .collect();

    // Track per-cite-key match data for dry_run response and deduplication.
    struct MatchInfo {
        author: String,
        title: String,
        year: Option<String>,
        matched_file_id: Option<i64>,
        matched_filename: String,
        overlap_score: f64,
    }

    let mut insert_rows = Vec::new();
    let mut unresolved_keys: Vec<String> = Vec::new();
    let mut unique_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut match_info_per_key: HashMap<String, MatchInfo> = HashMap::new();

    for cit in &raw_citations {
        let is_new_key = unique_keys.insert(cit.cite_key.clone());

        let bib_entry = bib_entries.get(&cit.cite_key);

        let (author, title, year, bib_abstract, keywords) = match bib_entry {
            Some(entry) => (
                entry.author.clone(),
                entry.title.clone(),
                entry.year.clone(),
                entry.bib_abstract.clone(),
                entry.keywords.clone(),
            ),
            None => {
                if is_new_key {
                    unresolved_keys.push(cit.cite_key.clone());
                }
                (String::new(), cit.cite_key.clone(), None, None, None)
            }
        };

        // Match against indexed files using token-overlap on author + title,
        // with year validation to reject candidates from different years.
        let match_result = find_best_file_match(&author, &title, year.as_deref(), &file_lookup);
        let (mut matched_file_id, overlap_score) = match match_result {
            Some((id, score)) => (Some(id), score),
            None => (None, 0.0),
        };

        // Apply file_overrides if the caller specified a manual correction.
        if let Some(&override_id) = file_overrides.get(&cit.cite_key) {
            matched_file_id = Some(override_id);
        }

        // Track match info for dry-run and first-occurrence deduplication.
        if is_new_key {
            let matched_filename = matched_file_id
                .and_then(|id| file_lookup.iter().find(|(fid, _)| *fid == id))
                .map(|(_, name)| name.clone())
                .unwrap_or_default();

            match_info_per_key.insert(
                cit.cite_key.clone(),
                MatchInfo {
                    author: author.clone(),
                    title: title.clone(),
                    year: year.clone(),
                    matched_file_id,
                    matched_filename,
                    overlap_score,
                },
            );
        }

        insert_rows.push(CitationRowInsert {
            job_id: String::new(), // Set after job creation.
            group_id: cit.group_id as i64,
            cite_key: cit.cite_key.clone(),
            author,
            title,
            year,
            tex_line: cit.line as i64,
            anchor_before: cit.anchor_before.clone(),
            anchor_after: cit.anchor_after.clone(),
            section_title: cit.section_title.clone(),
            matched_file_id,
            bib_abstract,
            bib_keywords: keywords,
            tex_context: cit.tex_context.clone(),
            batch_id: cit.batch_id.map(|b| b as i64),
            co_citation_json: None,
        });
    }

    // Compute co-citation metadata from group_id membership. A group_id is
    // shared by all cite-keys from the same \cite{a,b,c} command. Groups
    // with more than one distinct cite-key produce co-citation relationships.
    let mut group_keys: std::collections::HashMap<i64, Vec<String>> =
        std::collections::HashMap::new();
    for row in &insert_rows {
        group_keys
            .entry(row.group_id)
            .or_default()
            .push(row.cite_key.clone());
    }
    // Deduplicate keys within each group.
    for keys in group_keys.values_mut() {
        keys.sort();
        keys.dedup();
    }
    // Set co_citation_json on rows whose group contains >1 distinct cite-key.
    for row in &mut insert_rows {
        if let Some(keys) = group_keys.get(&row.group_id)
            && keys.len() > 1
        {
            let co_cited_with: Vec<String> = keys
                .iter()
                .filter(|k| *k != &row.cite_key)
                .cloned()
                .collect();
            let info = neuroncite_citation::CoCitationInfo {
                is_co_citation: true,
                co_cited_with,
            };
            row.co_citation_json = serde_json::to_string(&info).ok();
        }
    }

    // Count matched vs unmatched cite-keys (per unique cite-key, not per
    // individual citation row). A cite-key is "matched" when token-overlap
    // matching found a corresponding PDF file in the indexed session. The
    // field names use "cite_keys_" prefix to distinguish from total_citations
    // which counts individual citation rows (multiple rows can share the
    // same cite-key when a key appears in multiple \cite commands).
    let cite_keys_matched = match_info_per_key
        .values()
        .filter(|m| m.matched_file_id.is_some())
        .count();
    let cite_keys_unmatched = unique_keys.len().saturating_sub(cite_keys_matched);

    // Count total batches.
    let total_batches = insert_rows
        .iter()
        .filter_map(|r| r.batch_id)
        .collect::<std::collections::HashSet<_>>()
        .len();

    // --- Dry-run mode: return match preview without creating job or rows ---
    if dry_run {
        let matches: Vec<serde_json::Value> = match_info_per_key
            .iter()
            .map(|(key, info)| {
                serde_json::json!({
                    "cite_key": key,
                    "author": info.author,
                    "title": info.title,
                    "year": info.year,
                    "matched_file_id": info.matched_file_id,
                    "matched_filename": info.matched_filename,
                    "overlap_score": info.overlap_score,
                })
            })
            .collect();

        return Ok((
            StatusCode::OK,
            HeaderMap::new(),
            Json(serde_json::json!({
                "api_version": "v1",
                "dry_run": true,
                "matches": matches,
                "total_citations": insert_rows.len(),
                "total_batches": total_batches,
                "unique_cite_keys": unique_keys.len(),
                "cite_keys_matched": cite_keys_matched,
                "cite_keys_unmatched": cite_keys_unmatched,
                "unresolved_cite_keys": unresolved_keys,
            })),
        ));
    }

    // --- Normal mode: create job and insert rows ---
    let job_id = uuid::Uuid::new_v4().to_string();
    let primary_session_id = effective_session_ids[0];
    let job_params = CitationJobParams {
        tex_path: req.tex_path,
        bib_path: req.bib_path,
        session_id: primary_session_id,
        session_ids: effective_session_ids.clone(),
        batch_size,
    };
    let params_json = serde_json::to_string(&job_params).map_err(|e| ApiError::Internal {
        reason: format!("params serialization: {e}"),
    })?;

    // Citation verification jobs use state 'running' from the start because
    // they are agent-driven (not executor-driven). The executor ignores
    // jobs with kind = 'citation_verify'.
    store::create_job_with_params(
        &conn,
        &job_id,
        "citation_verify",
        Some(primary_session_id),
        Some(&params_json),
    )
    .map_err(ApiError::from)?;

    store::update_job_state(&conn, &job_id, JobState::Running, None).map_err(ApiError::from)?;

    // Set job_id on all insert rows and insert into database.
    for row in &mut insert_rows {
        row.job_id = job_id.clone();
    }

    let total_inserted = neuroncite_citation::db::insert_citation_rows(&conn, &insert_rows)
        .map_err(|e| ApiError::Internal {
            reason: format!("inserting citation rows: {e}"),
        })?;

    // Update job progress total.
    store::update_job_progress(&conn, &job_id, 0, total_inserted as i64).map_err(ApiError::from)?;

    let mut headers = HeaderMap::new();
    let location = format!("/api/v1/jobs/{job_id}");
    if let Ok(v) = HeaderValue::from_str(&location) {
        headers.insert(axum::http::header::LOCATION, v);
    }
    Ok((
        StatusCode::ACCEPTED,
        headers,
        Json(serde_json::json!({
            "api_version": "v1",
            "job_id": job_id,
            "session_id": primary_session_id,
            "session_ids": effective_session_ids,
            "total_citations": total_inserted,
            "total_batches": total_batches,
            "unique_cite_keys": unique_keys.len(),
            "cite_keys_matched": cite_keys_matched,
            "cite_keys_unmatched": cite_keys_unmatched,
            "unresolved_cite_keys": unresolved_keys,
        })),
    ))
}

/// POST /api/v1/citation/claim
///
/// Claims a batch of citations for sub-agent processing. When `batch_id` is
/// specified, claims that specific batch (allowing retries of failed batches).
/// When absent, claims the next pending batch in FIFO order.
///
/// Stale batches (claimed > 5 minutes ago) are automatically reclaimed before
/// the claim attempt.
#[utoipa::path(
    post,
    path = "/api/v1/citation/claim",
    request_body = CitationClaimRequest,
    responses(
        (status = 200, description = "Batch claimed or no batches available", body = CitationClaimResponse),
        (status = 404, description = "Job not found"),
        (status = 400, description = "Job is not a citation_verify job"),
    )
)]
pub async fn claim(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CitationClaimRequest>,
) -> Result<Json<CitationClaimResponse>, ApiError> {
    let conn = state.pool.get().map_err(ApiError::from)?;

    // Validate job exists and is a citation_verify job.
    let job = store::get_job(&conn, &req.job_id).map_err(|_| ApiError::NotFound {
        resource: format!("job {}", req.job_id),
    })?;
    if job.kind != "citation_verify" {
        return Err(ApiError::BadRequest {
            reason: format!("job '{}' is not a citation_verify job", req.job_id),
        });
    }

    // If batch_id is provided, claim that specific batch. Otherwise, claim
    // the next FIFO batch.
    let claim = if let Some(batch_id) = req.batch_id {
        neuroncite_citation::db::claim_specific_batch(&conn, &req.job_id, batch_id).map_err(
            |e| ApiError::Internal {
                reason: format!("claiming batch {batch_id}: {e}"),
            },
        )?
    } else {
        neuroncite_citation::db::claim_next_batch(&conn, &req.job_id).map_err(|e| {
            ApiError::Internal {
                reason: format!("claiming batch: {e}"),
            }
        })?
    };

    match claim {
        Some(batch) => {
            let rows: Vec<serde_json::Value> = batch
                .rows
                .iter()
                .map(|r| serde_json::to_value(r).unwrap_or_default())
                .collect();

            Ok(Json(CitationClaimResponse {
                api_version: API_VERSION.to_string(),
                batch_id: Some(batch.batch_id),
                tex_path: Some(batch.tex_path),
                bib_path: Some(batch.bib_path),
                session_id: Some(batch.session_id),
                rows,
                remaining_batches: None,
                message: None,
            }))
        }
        None => {
            // Distinguish between "specific batch not found / not claimable"
            // and "no pending batches remain in the FIFO queue". When the caller
            // specified a batch_id, returning a generic "no pending batches"
            // message is misleading because the real cause is that the specific
            // batch either does not exist or is already in a non-pending state.
            let batch_counts = neuroncite_citation::db::count_batches_by_status(&conn, &req.job_id)
                .map_err(|e| ApiError::Internal {
                    reason: format!("counting batches: {e}"),
                })?;

            let message = if let Some(bid) = req.batch_id {
                format!(
                    "batch {} not found or not claimable (total: {}, pending: {}, claimed: {}, done: {})",
                    bid,
                    batch_counts.total,
                    batch_counts.pending,
                    batch_counts.claimed,
                    batch_counts.done,
                )
            } else {
                "no pending batches available".to_string()
            };

            Ok(Json(CitationClaimResponse {
                api_version: API_VERSION.to_string(),
                batch_id: None,
                tex_path: None,
                bib_path: None,
                session_id: None,
                rows: Vec::new(),
                remaining_batches: Some(batch_counts.pending),
                message: Some(message),
            }))
        }
    }
}

/// GET /api/v1/citation/{job_id}/rows
///
/// Returns citation rows for a job with optional status filtering and pagination.
/// Provides direct read-only access to rows including their result_json field,
/// without the file-writing overhead of the export endpoint.
#[utoipa::path(
    get,
    path = "/api/v1/citation/{job_id}/rows",
    params(
        ("job_id" = String, Path, description = "Citation verification job ID"),
        CitationRowsQuery,
    ),
    responses(
        (status = 200, description = "Citation rows", body = CitationRowsResponse),
        (status = 404, description = "Job not found"),
        (status = 400, description = "Invalid query parameters"),
    )
)]
pub async fn rows(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    axum::extract::Query(query): axum::extract::Query<CitationRowsQuery>,
) -> Result<Json<CitationRowsResponse>, ApiError> {
    let conn = state.pool.get().map_err(ApiError::from)?;

    // Validate job exists and is a citation_verify job.
    let job = store::get_job(&conn, &job_id).map_err(|_| ApiError::NotFound {
        resource: format!("job {job_id}"),
    })?;
    if job.kind != "citation_verify" {
        return Err(ApiError::BadRequest {
            reason: format!("job '{job_id}' is not a citation_verify job"),
        });
    }

    // Validate optional status filter.
    if let Some(ref s) = query.status
        && !["pending", "claimed", "done", "failed"].contains(&s.as_str())
    {
        return Err(ApiError::BadRequest {
            reason: format!(
                "invalid status filter '{s}': must be pending, claimed, done, or failed"
            ),
        });
    }

    let offset = query.offset.unwrap_or(0);
    if offset < 0 {
        return Err(ApiError::BadRequest {
            reason: "offset must be >= 0".to_string(),
        });
    }

    let limit = query.limit.unwrap_or(100);
    if !(1..=500).contains(&limit) {
        return Err(ApiError::BadRequest {
            reason: "limit must be between 1 and 500".to_string(),
        });
    }

    let status_ref = query.status.as_deref();
    let (row_list, total) =
        neuroncite_citation::db::list_rows_filtered(&conn, &job_id, status_ref, offset, limit)
            .map_err(|e| ApiError::Internal {
                reason: format!("listing rows: {e}"),
            })?;

    let rows_json: Vec<serde_json::Value> = row_list
        .iter()
        .map(|r| serde_json::to_value(r).unwrap_or_default())
        .collect();

    Ok(Json(CitationRowsResponse {
        api_version: API_VERSION.to_string(),
        job_id,
        rows: rows_json,
        total,
        offset,
        limit,
    }))
}

/// POST /api/v1/citation/submit
///
/// Accepts verification results for a batch of citation rows. Each entry
/// contains the sub-agent's verdict, passages, reasoning, confidence, and
/// optional LaTeX correction. Rows are validated to belong to the specified
/// job and to be in 'claimed' status. Auto-completes the job when all
/// rows are done or failed.
#[utoipa::path(
    post,
    path = "/api/v1/citation/submit",
    request_body = CitationSubmitRequest,
    responses(
        (status = 200, description = "Results accepted", body = CitationSubmitResponse),
        (status = 400, description = "Invalid results format"),
        (status = 404, description = "Job not found"),
    )
)]
pub async fn submit(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CitationSubmitRequest>,
) -> Result<Json<CitationSubmitResponse>, ApiError> {
    if req.results.is_empty() {
        return Err(ApiError::BadRequest {
            reason: "results array must not be empty".to_string(),
        });
    }

    // Deserialize the results array into SubmitEntry values.
    let results: Vec<SubmitEntry> = req
        .results
        .into_iter()
        .map(serde_json::from_value)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::BadRequest {
            reason: format!("invalid results format: {e}"),
        })?;

    let conn = state.pool.get().map_err(ApiError::from)?;

    let count = neuroncite_citation::db::submit_batch_results(&conn, &req.job_id, &results)
        .map_err(|e| ApiError::Internal {
            reason: format!("submitting results: {e}"),
        })?;

    // Check if all rows are done/failed and complete the job if so.
    let status_counts =
        neuroncite_citation::db::count_by_status(&conn, &req.job_id).map_err(|e| {
            ApiError::Internal {
                reason: format!("counting status: {e}"),
            }
        })?;

    let total =
        status_counts.pending + status_counts.claimed + status_counts.done + status_counts.failed;
    let is_complete = status_counts.pending == 0 && status_counts.claimed == 0;

    if is_complete {
        store::update_job_state(&conn, &req.job_id, JobState::Completed, None)
            .map_err(ApiError::from)?;
    }

    Ok(Json(CitationSubmitResponse {
        api_version: API_VERSION.to_string(),
        status: "accepted".to_string(),
        rows_submitted: count,
        total,
        is_complete,
    }))
}

/// GET /api/v1/citation/{job_id}/status
///
/// Returns counts by row status, counts by batch status, verdict
/// aggregates, and flagged alert entries for the specified citation
/// verification job.
#[utoipa::path(
    get,
    path = "/api/v1/citation/{job_id}/status",
    params(
        ("job_id" = String, Path, description = "Citation verification job ID")
    ),
    responses(
        (status = 200, description = "Job status", body = CitationStatusResponse),
        (status = 404, description = "Job not found"),
    )
)]
pub async fn status(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<CitationStatusResponse>, ApiError> {
    let conn = state.pool.get().map_err(ApiError::from)?;

    let job = store::get_job(&conn, &job_id).map_err(|_| ApiError::NotFound {
        resource: format!("job {job_id}"),
    })?;

    let status_counts = neuroncite_citation::db::count_by_status(&conn, &job_id).map_err(|e| {
        ApiError::Internal {
            reason: format!("counting status: {e}"),
        }
    })?;

    let batch_counts =
        neuroncite_citation::db::count_batches_by_status(&conn, &job_id).map_err(|e| {
            ApiError::Internal {
                reason: format!("counting batches: {e}"),
            }
        })?;

    let verdicts = neuroncite_citation::db::count_verdicts(&conn, &job_id).map_err(|e| {
        ApiError::Internal {
            reason: format!("counting verdicts: {e}"),
        }
    })?;

    let alerts =
        neuroncite_citation::db::build_alerts(&conn, &job_id).map_err(|e| ApiError::Internal {
            reason: format!("building alerts: {e}"),
        })?;

    let total =
        status_counts.pending + status_counts.claimed + status_counts.done + status_counts.failed;
    let is_complete = status_counts.pending == 0 && status_counts.claimed == 0;

    // Calculate elapsed time from job timestamps.
    let elapsed = if let Some(started) = job.started_at {
        let now = neuroncite_core::unix_timestamp_secs();
        if let Some(finished) = job.finished_at {
            finished - started
        } else {
            now - started
        }
    } else {
        0
    };

    // `verdicts` is already a HashMap<String, i64> returned by count_verdicts.
    // It maps directly onto the typed DTO field without any JSON conversion.
    let alerts_json: Vec<serde_json::Value> = alerts
        .iter()
        .map(|a| serde_json::to_value(a).unwrap_or_default())
        .collect();

    Ok(Json(CitationStatusResponse {
        api_version: API_VERSION.to_string(),
        job_id,
        job_state: crate::dto::JobStateDto::from(job.state),
        total,
        pending: status_counts.pending,
        claimed: status_counts.claimed,
        done: status_counts.done,
        failed: status_counts.failed,
        total_batches: batch_counts.total,
        batches_done: batch_counts.done,
        batches_pending: batch_counts.pending,
        batches_claimed: batch_counts.claimed,
        verdicts,
        alerts: alerts_json,
        elapsed_seconds: elapsed,
        is_complete,
    }))
}

/// POST /api/v1/citation/{job_id}/export
///
/// Generates three output files from the completed citation verification
/// results: an annotation CSV for the neuroncite-annotate pipeline, a
/// corrections JSON with suggested LaTeX fixes sorted by line number, and
/// a summary report JSON. Triggers the annotation pipeline when the CSV
/// contains data rows.
#[utoipa::path(
    post,
    path = "/api/v1/citation/{job_id}/export",
    request_body = CitationExportRequest,
    params(
        ("job_id" = String, Path, description = "Citation verification job ID")
    ),
    responses(
        (status = 200, description = "Export completed", body = CitationExportResponse),
        (status = 404, description = "Job not found"),
        (status = 400, description = "Invalid export parameters"),
    )
)]
pub async fn export(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    Json(req): Json<CitationExportRequest>,
) -> Result<Json<CitationExportResponse>, ApiError> {
    if req.output_directory.is_empty() {
        return Err(ApiError::BadRequest {
            reason: "output_directory must not be empty".to_string(),
        });
    }
    if req.source_directory.is_empty() {
        return Err(ApiError::BadRequest {
            reason: "source_directory must not be empty".to_string(),
        });
    }

    let conn = state.pool.get().map_err(ApiError::from)?;

    let job = store::get_job(&conn, &job_id).map_err(|_| ApiError::NotFound {
        resource: format!("job {job_id}"),
    })?;

    // Read all done rows.
    let done_rows = neuroncite_citation::db::list_done_rows(&conn, &job_id).map_err(|e| {
        ApiError::Internal {
            reason: format!("listing done rows: {e}"),
        }
    })?;

    let verdicts = neuroncite_citation::db::count_verdicts(&conn, &job_id).map_err(|e| {
        ApiError::Internal {
            reason: format!("counting verdicts: {e}"),
        }
    })?;

    let alerts =
        neuroncite_citation::db::build_alerts(&conn, &job_id).map_err(|e| ApiError::Internal {
            reason: format!("building alerts: {e}"),
        })?;

    // Calculate elapsed time.
    let elapsed = if let Some(started) = job.started_at {
        job.finished_at
            .unwrap_or_else(neuroncite_core::unix_timestamp_secs)
            - started
    } else {
        0
    };

    // Generate output files: 6 total (annotation CSV, data CSV, Excel,
    // corrections JSON, summary report, full-detail JSON).
    let annotation_csv =
        neuroncite_citation::export::build_annotation_csv(&done_rows).map_err(|e| {
            ApiError::Internal {
                reason: format!("building annotation CSV: {e}"),
            }
        })?;

    let data_csv = neuroncite_citation::export::build_data_csv(&done_rows).map_err(|e| {
        ApiError::Internal {
            reason: format!("building data CSV: {e}"),
        }
    })?;

    let excel_bytes =
        neuroncite_citation::export::build_excel(&done_rows).map_err(|e| ApiError::Internal {
            reason: format!("building Excel workbook: {e}"),
        })?;

    let corrections_content = neuroncite_citation::export::build_corrections_json(&done_rows)
        .map_err(|e| ApiError::Internal {
            reason: format!("building corrections JSON: {e}"),
        })?;

    let report_content = neuroncite_citation::export::build_summary_report(
        &job_id, &done_rows, &alerts, &verdicts, elapsed,
    )
    .map_err(|e| ApiError::Internal {
        reason: format!("building summary report: {e}"),
    })?;

    let full_detail_content = neuroncite_citation::export::build_full_detail_json(&done_rows)
        .map_err(|e| ApiError::Internal {
            reason: format!("building full detail JSON: {e}"),
        })?;

    // Path containment: verify output and source directories are within the
    // allowed roots. Uses the canonical path returned by validate_path_access
    // for the subsequent create_dir_all to prevent TOCTOU symlink swaps.
    let canonical_out_dir = crate::util::validate_path_access(
        std::path::Path::new(&req.output_directory),
        &state.config,
    )?;
    crate::util::validate_path_access(std::path::Path::new(&req.source_directory), &state.config)?;

    // Create the output directory using the canonical path.
    std::fs::create_dir_all(&canonical_out_dir).map_err(|e| ApiError::Internal {
        reason: format!("creating output directory: {e}"),
    })?;

    // Write output files. The annotation CSV is named "annotation_pipeline_input.csv"
    // to distinguish it from the full 38-column "citation_data.csv".
    let annotation_csv_path = canonical_out_dir.join("annotation_pipeline_input.csv");
    let data_csv_path = canonical_out_dir.join("citation_data.csv");
    let excel_path = canonical_out_dir.join("citation_data.xlsx");
    let corrections_path = canonical_out_dir.join("corrections.json");
    let report_path = canonical_out_dir.join("citation_report.json");
    let full_detail_path = canonical_out_dir.join("citation_full_detail.json");

    std::fs::write(&annotation_csv_path, &annotation_csv).map_err(|e| ApiError::Internal {
        reason: format!("writing annotation CSV: {e}"),
    })?;
    std::fs::write(&data_csv_path, &data_csv).map_err(|e| ApiError::Internal {
        reason: format!("writing data CSV: {e}"),
    })?;
    std::fs::write(&excel_path, &excel_bytes).map_err(|e| ApiError::Internal {
        reason: format!("writing Excel: {e}"),
    })?;
    std::fs::write(&corrections_path, &corrections_content).map_err(|e| ApiError::Internal {
        reason: format!("writing corrections: {e}"),
    })?;
    std::fs::write(&report_path, &report_content).map_err(|e| ApiError::Internal {
        reason: format!("writing report: {e}"),
    })?;
    std::fs::write(&full_detail_path, &full_detail_content).map_err(|e| ApiError::Internal {
        reason: format!("writing full detail: {e}"),
    })?;

    // Trigger annotation pipeline with the annotation CSV if it has data rows
    // and the caller requested annotation (annotate=true, the default).
    // Annotated PDFs are written to an `annotated_pdfs/` subfolder inside the
    // output directory so that the main directory contains only CSV, Excel,
    // and JSON export files.
    let csv_lines = annotation_csv.lines().count();
    let annotation_job_id = if csv_lines > 1 && req.annotate {
        let ann_job_id = uuid::Uuid::new_v4().to_string();
        let ann_output_dir = canonical_out_dir.join("annotated_pdfs");
        let ann_params = serde_json::json!({
            "input_data": annotation_csv,
            "source_directory": req.source_directory,
            "output_directory": ann_output_dir.to_string_lossy(),
            "default_color": "#FFFF00",
        });
        let ann_params_json =
            serde_json::to_string(&ann_params).map_err(|e| ApiError::Internal {
                reason: format!("annotation params: {e}"),
            })?;

        store::create_job_with_params(&conn, &ann_job_id, "annotate", None, Some(&ann_params_json))
            .map_err(ApiError::from)?;

        // Wake the executor so it picks up the annotation job immediately.
        state.job_notify.notify_one();

        Some(ann_job_id)
    } else {
        None
    };

    // Build summary counts.
    let summary = build_export_summary(&verdicts, &done_rows, &alerts);

    Ok(Json(CitationExportResponse {
        api_version: API_VERSION.to_string(),
        annotation_job_id,
        annotation_csv_path: annotation_csv_path.to_string_lossy().to_string(),
        data_csv_path: data_csv_path.to_string_lossy().to_string(),
        excel_path: excel_path.to_string_lossy().to_string(),
        report_path: report_path.to_string_lossy().to_string(),
        corrections_path: corrections_path.to_string_lossy().to_string(),
        full_detail_path: full_detail_path.to_string_lossy().to_string(),
        summary,
    }))
}

/// POST /api/v1/citation/{job_id}/auto-verify
///
/// Starts a background citation agent that uses a local LLM via Ollama to
/// autonomously verify all pending citation rows in the job. The agent claims
/// batches, generates search queries, executes them against the indexed corpus,
/// evaluates passages, and submits structured verdicts -- all without external
/// API calls or human intervention.
///
/// The endpoint returns 202 Accepted immediately. Progress is reported via
/// SSE events on the `/api/v1/events/citation` channel.
#[utoipa::path(
    post,
    path = "/api/v1/citation/{job_id}/auto-verify",
    request_body = AutoVerifyRequest,
    params(
        ("job_id" = String, Path, description = "Citation verification job ID")
    ),
    responses(
        (status = 202, description = "Agent started", body = AutoVerifyResponse),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Job not found"),
    )
)]
pub async fn auto_verify(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    Json(req): Json<AutoVerifyRequest>,
) -> Result<(StatusCode, Json<AutoVerifyResponse>), ApiError> {
    // Validate the model field is not empty.
    if req.model.is_empty() {
        return Err(ApiError::BadRequest {
            reason: "model field must not be empty".to_string(),
        });
    }

    let conn = state.pool.get().map_err(ApiError::from)?;

    // Validate job exists and is a running citation_verify job.
    let job = store::get_job(&conn, &job_id).map_err(|_| ApiError::NotFound {
        resource: format!("job {job_id}"),
    })?;
    if job.kind != "citation_verify" {
        return Err(ApiError::BadRequest {
            reason: format!("job '{job_id}' is not a citation_verify job"),
        });
    }

    // Prevent spawning a second agent for the same job. The DashSet is
    // populated by CitationAgent::run() on start and cleared on finish.
    // Without this guard, a second auto_verify call would spawn a competing
    // agent that claims the same batches.
    if state.active_citation_agents.contains(&job_id) {
        return Err(ApiError::Conflict {
            reason: format!("a citation agent is already running for job '{job_id}'"),
        });
    }

    // Read the session_ids from the job's params_json. Jobs created before
    // multi-session support have only `session_id`; the `session_ids` field
    // defaults to an empty Vec via serde, and we fall back to wrapping the
    // single `session_id` in a Vec.
    let params_json = job.params_json.as_deref().unwrap_or("{}");
    let job_params: neuroncite_citation::CitationJobParams = serde_json::from_str(params_json)
        .map_err(|e| ApiError::Internal {
            reason: format!("failed to parse job params: {e}"),
        })?;

    let effective_session_ids = if job_params.session_ids.is_empty() {
        vec![job_params.session_id]
    } else {
        job_params.session_ids.clone()
    };

    // Construct the LLM backend from the request parameters.
    let llm_config = llm_config_from_request(&req);
    let ollama_backend = OllamaBackend::new(llm_config).map_err(|e| ApiError::Internal {
        reason: format!("failed to create Ollama HTTP client: {e}"),
    })?;
    let llm: Arc<dyn neuroncite_llm::LlmBackend> = Arc::new(ollama_backend);

    // Clone the SSE broadcast sender for citation events. The channel is
    // initialised unconditionally in AppState::new(); in headless mode there
    // are no subscribers so events are silently discarded by the runtime.
    let citation_tx = Some(state.sse.citation_tx.clone());

    // Construct the agent and spawn it as a background task.
    // Verification tuning parameters fall back to sensible defaults when
    // not provided in the request (matching the previous hardcoded values).
    let agent_config = AgentConfig {
        top_k: req.top_k.unwrap_or(5),
        cross_corpus_queries: req.cross_corpus_queries.unwrap_or(2).min(4),
        max_retry_attempts: req.max_retry_attempts.unwrap_or(3).clamp(1, 5),
        min_score: req.min_score,
        rerank: req.rerank.unwrap_or(false),
        source_directory: req.source_directory.clone(),
        fetch_sources: req.fetch_sources.unwrap_or(false),
        unpaywall_email: req.unpaywall_email.clone(),
    };

    // Wrap the agent in Arc (required by the run() method signature).
    let agent = Arc::new(CitationAgent::new(
        Arc::clone(&state),
        llm,
        job_id.clone(),
        effective_session_ids,
        citation_tx,
        agent_config,
    ));

    // The JoinHandle from tokio::spawn is intentionally not stored. The
    // citation agent manages its own lifecycle through the job state machine
    // in the database (queued -> running -> completed/failed). If the agent
    // panics, the spawned task catches it and the error is logged below.
    // The job state in the database transitions to "failed" via the agent's
    // internal error handling, allowing the frontend to display the failure.
    let job_id_for_log = job_id.clone();
    tokio::spawn(async move {
        if let Err(e) = agent.run().await {
            tracing::error!(job_id = %job_id_for_log, error = %e, "Citation agent failed");
        }
    });

    Ok((
        StatusCode::ACCEPTED,
        Json(AutoVerifyResponse {
            api_version: API_VERSION.to_string(),
            status: "started".to_string(),
            job_id,
            message: format!("Citation agent started with model '{}'", req.model),
        }),
    ))
}

/// POST /api/v1/citation/parse-bib
///
/// Lightweight endpoint that parses a .bib file and returns structured metadata
/// for all entries without performing any network operations or downloads. Used
/// by the Sources tab to show a live preview of BibTeX entries as soon as the
/// user selects a .bib file, before clicking "Fetch Sources".
#[utoipa::path(
    post,
    path = "/api/v1/citation/parse-bib",
    request_body = crate::dto::ParseBibRequest,
    responses(
        (status = 200, description = "BibTeX entries parsed", body = crate::dto::ParseBibResponse),
        (status = 400, description = "Invalid bib file path"),
    )
)]
pub async fn parse_bib(
    Json(req): Json<crate::dto::ParseBibRequest>,
) -> Result<Json<crate::dto::ParseBibResponse>, ApiError> {
    if req.bib_path.is_empty() {
        return Err(ApiError::BadRequest {
            reason: "bib_path must not be empty".to_string(),
        });
    }

    let bib_file = std::path::Path::new(&req.bib_path);
    if !bib_file.is_file() {
        return Err(ApiError::BadRequest {
            reason: format!("bib file does not exist: {}", req.bib_path),
        });
    }

    let bib_path_owned = req.bib_path.clone();
    let bib_entries = tokio::task::spawn_blocking(move || {
        let content = std::fs::read_to_string(&bib_path_owned)
            .map_err(|e| format!("failed to read bib file: {e}"))?;
        Ok::<_, String>(neuroncite_citation::bibtex::parse_bibtex(&content))
    })
    .await
    .map_err(|e| ApiError::Internal {
        reason: format!("blocking task panicked: {e}"),
    })?
    .map_err(|e: String| ApiError::Internal { reason: e })?;

    // When output_directory is provided, recursively scan it and all
    // subdirectories at arbitrary depth for PDF and HTML files. Each file
    // is stored as (lowercased_filename, relative_path_from_output_dir).
    let existing_files: Vec<(String, String)> = if let Some(ref out_dir) = req.output_directory {
        let dir = std::path::Path::new(out_dir);
        let mut collected: Vec<(String, String)> = Vec::new();
        if dir.is_dir() {
            collect_files_recursive(dir, dir, &mut collected);
        }
        collected
    } else {
        Vec::new()
    };

    let mut entries: Vec<crate::dto::BibEntryPreview> = bib_entries
        .into_iter()
        .map(|(cite_key, entry)| {
            // Check if files for this entry exist in the output directory
            // using token-overlap matching. Returns ALL matches so duplicate
            // files (same entry found in multiple subdirectories) can be
            // detected and reported to the user.
            let all_matches = if !existing_files.is_empty() {
                find_all_disk_file_matches(
                    &entry.author,
                    &entry.title,
                    entry.year.as_deref(),
                    &cite_key,
                    &existing_files,
                )
            } else {
                Vec::new()
            };

            let file_exists = !all_matches.is_empty();
            let existing_file = all_matches.first().map(|(path, _)| path.clone());

            // When two or more files match the same BibTeX entry, store all
            // paths so the UI can show a "duplicate" status with full paths.
            let duplicate_files: Option<Vec<String>> = if all_matches.len() >= 2 {
                Some(all_matches.iter().map(|(path, _)| path.clone()).collect())
            } else {
                None
            };

            // Compute the canonical filename that the download pipeline generates
            // from BibTeX metadata. Displayed in the Sources tab so users can
            // see what filename the system produces when fetching sources.
            let expected_filename = {
                let stem = neuroncite_html::build_source_filename(
                    &entry.title,
                    &entry.author,
                    entry.year.as_deref(),
                    &cite_key,
                );
                if stem.is_empty() {
                    None
                } else {
                    Some(format!("{stem}.pdf"))
                }
            };

            crate::dto::BibEntryPreview {
                cite_key,
                entry_type: entry.entry_type,
                author: entry.author,
                title: entry.title,
                year: entry.year,
                has_url: entry.url.is_some(),
                has_doi: entry.doi.is_some(),
                url: entry.url,
                doi: entry.doi,
                bib_abstract: entry.bib_abstract,
                keywords: entry.keywords,
                file_exists,
                existing_file,
                expected_filename,
                duplicate_files,
                extra_fields: entry.extra_fields,
            }
        })
        .collect();

    // Sort by cite_key for stable ordering in the frontend preview.
    entries.sort_by(|a, b| a.cite_key.cmp(&b.cite_key));

    Ok(Json(crate::dto::ParseBibResponse {
        api_version: API_VERSION.to_string(),
        entries,
    }))
}

/// POST /api/v1/citation/bib-report
///
/// Generates CSV and XLSX report files from a BibTeX file. Parses the .bib
/// file, checks file existence in the output directory, and writes a report
/// listing all entries with cite_key, title, author, year, link type
/// (URL/DOI/none), and download status (exists/missing/no_link). Existing
/// report files are overwritten.
///
/// The CSV file includes a UTF-8 BOM for correct encoding in Excel. The XLSX
/// file has a formatted header row, status-colored rows, and frozen header
/// for scrolling convenience.
#[utoipa::path(
    post,
    path = "/api/v1/citation/bib-report",
    request_body = crate::dto::BibReportRequest,
    responses(
        (status = 200, description = "Report generated", body = crate::dto::BibReportResponse),
        (status = 400, description = "Invalid request parameters"),
    )
)]
pub async fn bib_report(
    Json(req): Json<crate::dto::BibReportRequest>,
) -> Result<Json<crate::dto::BibReportResponse>, ApiError> {
    if req.bib_path.is_empty() {
        return Err(ApiError::BadRequest {
            reason: "bib_path must not be empty".to_string(),
        });
    }
    if req.output_directory.is_empty() {
        return Err(ApiError::BadRequest {
            reason: "output_directory must not be empty".to_string(),
        });
    }

    let bib_file = std::path::Path::new(&req.bib_path);
    if !bib_file.is_file() {
        return Err(ApiError::BadRequest {
            reason: format!("bib file does not exist: {}", req.bib_path),
        });
    }

    let output_dir = std::path::Path::new(&req.output_directory);
    std::fs::create_dir_all(output_dir).map_err(|e| ApiError::Internal {
        reason: format!("failed to create output directory: {e}"),
    })?;

    // Parse BibTeX on a blocking thread (CPU-bound text processing).
    let bib_path_owned = req.bib_path.clone();
    let bib_entries = tokio::task::spawn_blocking(move || {
        let content = std::fs::read_to_string(&bib_path_owned)
            .map_err(|e| format!("failed to read bib file: {e}"))?;
        Ok::<_, String>(neuroncite_citation::bibtex::parse_bibtex(&content))
    })
    .await
    .map_err(|e| ApiError::Internal {
        reason: format!("blocking task panicked: {e}"),
    })?
    .map_err(|e: String| ApiError::Internal { reason: e })?;

    // Recursively scan the output directory and all subdirectories at
    // arbitrary depth for PDF and HTML files. Stores each file as
    // (lowercased_filename, relative_path_from_output_dir).
    let mut existing_files: Vec<(String, String)> = Vec::new();
    if output_dir.is_dir() {
        collect_files_recursive(output_dir, output_dir, &mut existing_files);
    }

    // Build report rows sorted by cite_key.
    struct ReportEntry {
        cite_key: String,
        title: String,
        author: String,
        year: String,
        link_type: String, // "URL", "DOI", or ""
        status: String,    // "exists", "duplicate", "missing", or "no_link"
        url: String,
        doi: String,
        existing_file: String,
        expected_filename: String,
    }

    let mut rows: Vec<ReportEntry> = bib_entries
        .into_iter()
        .map(|(cite_key, entry)| {
            let link_type = if entry.url.is_some() {
                "URL"
            } else if entry.doi.is_some() {
                "DOI"
            } else {
                ""
            };

            // Find all matching files using token-overlap matching. Multiple
            // matches indicate duplicate files (same entry in different
            // subdirectories). The report uses "duplicate" status for these.
            let all_matches = if !existing_files.is_empty() {
                find_all_disk_file_matches(
                    &entry.author,
                    &entry.title,
                    entry.year.as_deref(),
                    &cite_key,
                    &existing_files,
                )
            } else {
                Vec::new()
            };

            let existing_file = all_matches
                .first()
                .map(|(path, _)| path.clone())
                .unwrap_or_default();

            let status = if all_matches.len() >= 2 {
                // Multiple files match the same BibTeX entry. Report status
                // "duplicate" so users can consolidate their file copies.
                "duplicate"
            } else if !all_matches.is_empty() {
                "exists"
            } else if entry.url.is_some() || entry.doi.is_some() {
                "missing"
            } else {
                "no_link"
            };

            let expected_filename = format!(
                "{}.pdf",
                neuroncite_html::build_source_filename(
                    &entry.title,
                    &entry.author,
                    entry.year.as_deref(),
                    &cite_key,
                )
            );

            // When duplicates exist, list all paths in the existing_file
            // column separated by " | " for CSV/XLSX readability.
            let existing_file_display = if all_matches.len() >= 2 {
                all_matches
                    .iter()
                    .map(|(path, _)| path.as_str())
                    .collect::<Vec<_>>()
                    .join(" | ")
            } else {
                existing_file
            };

            ReportEntry {
                cite_key,
                title: entry.title,
                author: entry.author,
                year: entry.year.unwrap_or_default(),
                link_type: link_type.to_string(),
                status: status.to_string(),
                url: entry.url.unwrap_or_default(),
                doi: entry.doi.unwrap_or_default(),
                existing_file: existing_file_display,
                expected_filename,
            }
        })
        .collect();

    rows.sort_by(|a, b| a.cite_key.cmp(&b.cite_key));

    // Aggregate counts for the response. "duplicate" entries count as
    // existing (the file is present, just in multiple locations).
    let total_entries = rows.len();
    let duplicate_count = rows.iter().filter(|r| r.status == "duplicate").count();
    let existing_count = rows
        .iter()
        .filter(|r| r.status == "exists" || r.status == "duplicate")
        .count();
    let missing_count = rows.iter().filter(|r| r.status == "missing").count();
    let no_link_count = rows.iter().filter(|r| r.status == "no_link").count();

    // Column definitions: (header_name, xlsx_width).
    const COLUMNS: &[(&str, f64)] = &[
        ("cite_key", 22.0),
        ("title", 50.0),
        ("author", 35.0),
        ("year", 8.0),
        ("link_type", 10.0),
        ("status", 12.0),
        ("url", 45.0),
        ("doi", 25.0),
        ("existing_file", 45.0),
        ("expected_filename", 50.0),
    ];

    // --- CSV report ---
    let csv_escape = |s: &str| -> String {
        if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    };

    let mut csv = String::new();
    // UTF-8 BOM so Excel opens the file with correct encoding.
    csv.push('\u{FEFF}');
    csv.push_str(
        &COLUMNS
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(","),
    );
    csv.push('\n');

    for row in &rows {
        let fields = [
            &row.cite_key,
            &row.title,
            &row.author,
            &row.year,
            &row.link_type,
            &row.status,
            &row.url,
            &row.doi,
            &row.existing_file,
            &row.expected_filename,
        ];
        csv.push_str(
            &fields
                .iter()
                .map(|f| csv_escape(f))
                .collect::<Vec<_>>()
                .join(","),
        );
        csv.push('\n');
    }

    let csv_path = output_dir.join("bib_report.csv");
    std::fs::write(&csv_path, &csv).map_err(|e| ApiError::Internal {
        reason: format!("failed to write CSV report: {e}"),
    })?;
    tracing::info!(path = %csv_path.display(), entries = total_entries, "bib report CSV written");

    // --- XLSX report ---
    let xlsx_path = output_dir.join("bib_report.xlsx");
    let xlsx_result = (|| -> Result<(), rust_xlsxwriter::XlsxError> {
        use rust_xlsxwriter::{Color, Format, FormatAlign, FormatBorder, Workbook};

        let mut workbook = Workbook::new();
        let ws = workbook.add_worksheet();
        ws.set_name("BibTeX Report")?;

        let border_color = Color::RGB(0xBBBBBB);

        // Header format: dark blue background, white bold text.
        let header_fmt = Format::new()
            .set_background_color(Color::RGB(0x1F3864))
            .set_font_color(Color::White)
            .set_bold()
            .set_font_size(10.0)
            .set_align(FormatAlign::Center)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        // Write header and set column widths.
        for (col, (name, width)) in COLUMNS.iter().enumerate() {
            ws.set_column_width(col as u16, *width)?;
            ws.write_with_format(0, col as u16, *name, &header_fmt)?;
        }
        ws.set_freeze_panes(1, 0)?;

        // Row formats per status.
        let exists_fmt = Format::new()
            .set_background_color(Color::RGB(0xE0F7FA))
            .set_font_size(10.0)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color)
            .set_text_wrap();

        let duplicate_fmt = Format::new()
            .set_background_color(Color::RGB(0xFFF8E1))
            .set_font_size(10.0)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color)
            .set_text_wrap();

        let missing_fmt = Format::new()
            .set_background_color(Color::RGB(0xFCE4EC))
            .set_font_size(10.0)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color)
            .set_text_wrap();

        let no_link_fmt = Format::new()
            .set_background_color(Color::RGB(0xF5F5F5))
            .set_font_size(10.0)
            .set_font_color(Color::RGB(0x999999))
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color)
            .set_text_wrap();

        // Status badge formats.
        let exists_badge = Format::new()
            .set_background_color(Color::RGB(0x00897B))
            .set_font_color(Color::White)
            .set_bold()
            .set_font_size(10.0)
            .set_align(FormatAlign::Center)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        let duplicate_badge = Format::new()
            .set_background_color(Color::RGB(0xF9A825))
            .set_font_color(Color::White)
            .set_bold()
            .set_font_size(10.0)
            .set_align(FormatAlign::Center)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        let missing_badge = Format::new()
            .set_background_color(Color::RGB(0xD81B60))
            .set_font_color(Color::White)
            .set_bold()
            .set_font_size(10.0)
            .set_align(FormatAlign::Center)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        let no_link_badge = Format::new()
            .set_background_color(Color::RGB(0xBDBDBD))
            .set_font_color(Color::White)
            .set_bold()
            .set_font_size(10.0)
            .set_align(FormatAlign::Center)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        for (row_idx, row) in rows.iter().enumerate() {
            let excel_row = (row_idx + 1) as u32;
            let row_fmt = match row.status.as_str() {
                "exists" => &exists_fmt,
                "duplicate" => &duplicate_fmt,
                "missing" => &missing_fmt,
                _ => &no_link_fmt,
            };

            let fields = [
                &row.cite_key,
                &row.title,
                &row.author,
                &row.year,
                &row.link_type,
                &row.status,
                &row.url,
                &row.doi,
                &row.existing_file,
                &row.expected_filename,
            ];

            for (col, value) in fields.iter().enumerate() {
                let col16 = col as u16;
                if col == 5 {
                    // Status column: use badge format.
                    let badge = match row.status.as_str() {
                        "exists" => &exists_badge,
                        "duplicate" => &duplicate_badge,
                        "missing" => &missing_badge,
                        _ => &no_link_badge,
                    };
                    ws.write_with_format(excel_row, col16, value.as_str(), badge)?;
                } else {
                    ws.write_with_format(excel_row, col16, value.as_str(), row_fmt)?;
                }
            }
            ws.set_row_height(excel_row, 30.0)?;
        }

        // Sheet 2: Summary statistics.
        let ws2 = workbook.add_worksheet();
        ws2.set_name("Summary")?;

        let label_fmt = Format::new()
            .set_bold()
            .set_align(FormatAlign::Left)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        let value_fmt = Format::new()
            .set_align(FormatAlign::Right)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        let title_fmt = Format::new()
            .set_background_color(Color::RGB(0x1F3864))
            .set_font_color(Color::White)
            .set_bold()
            .set_font_size(14.0)
            .set_align(FormatAlign::Center)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        ws2.set_column_width(0, 28.0)?;
        ws2.set_column_width(1, 16.0)?;
        ws2.merge_range(0, 0, 0, 1, "BibTeX Report Summary", &title_fmt)?;

        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

        let summary_rows: &[(&str, String)] = &[
            ("Generated", timestamp),
            ("BibTeX file", req.bib_path.clone()),
            ("Output directory", req.output_directory.clone()),
            ("", String::new()),
            ("Total entries", total_entries.to_string()),
            (
                "Entries with URL/DOI",
                (existing_count + missing_count).to_string(),
            ),
            ("Entries without URL/DOI", no_link_count.to_string()),
            ("Already downloaded (exists)", existing_count.to_string()),
            (
                "Duplicates (multiple copies found)",
                duplicate_count.to_string(),
            ),
            ("Missing (not yet downloaded)", missing_count.to_string()),
        ];

        for (idx, (label, value)) in summary_rows.iter().enumerate() {
            let r = (idx + 1) as u32;
            if label.is_empty() {
                continue;
            }
            ws2.write_with_format(r, 0, *label, &label_fmt)?;
            ws2.write_with_format(r, 1, value.as_str(), &value_fmt)?;
        }

        workbook.save(&xlsx_path)?;
        Ok(())
    })();

    if let Err(e) = xlsx_result {
        return Err(ApiError::Internal {
            reason: format!("failed to write XLSX report: {e}"),
        });
    }
    tracing::info!(path = %xlsx_path.display(), entries = total_entries, "bib report XLSX written");

    Ok(Json(crate::dto::BibReportResponse {
        api_version: API_VERSION.to_string(),
        csv_path: csv_path.display().to_string(),
        xlsx_path: xlsx_path.display().to_string(),
        total_entries,
        existing_count,
        duplicate_count,
        missing_count,
        no_link_count,
    }))
}

/// POST /api/v1/citation/fetch-sources
///
/// Standalone endpoint for downloading cited source documents from BibTeX
/// URL/DOI fields. Parses the .bib file, resolves DOIs through the
/// multi-source resolution chain (Unpaywall -> Semantic Scholar -> OpenAlex
/// -> doi.org), classifies each URL as PDF or HTML, downloads the files,
/// indexes them into the specified session, and rebuilds the HNSW index.
///
/// Sends per-entry SSE events through the source_tx broadcast channel so the
/// frontend can display live download progress. The response still contains
/// the complete results for clients that do not use SSE.
#[utoipa::path(
    post,
    path = "/api/v1/citation/fetch-sources",
    request_body = crate::dto::FetchSourcesRequest,
    responses(
        (status = 200, description = "Source fetching completed", body = crate::dto::FetchSourcesResponse),
        (status = 400, description = "Invalid request parameters"),
        (status = 404, description = "Session not found"),
    )
)]
pub async fn fetch_sources(
    State(state): State<Arc<AppState>>,
    Json(req): Json<crate::dto::FetchSourcesRequest>,
) -> Result<Json<crate::dto::FetchSourcesResponse>, ApiError> {
    // Validate required fields.
    if req.bib_path.is_empty() {
        return Err(ApiError::BadRequest {
            reason: "bib_path must not be empty".to_string(),
        });
    }
    if req.output_directory.is_empty() {
        return Err(ApiError::BadRequest {
            reason: "output_directory must not be empty".to_string(),
        });
    }

    // Path containment: verify bib and output paths are within the allowed
    // roots. validate_path_access returns the canonical path to prevent
    // TOCTOU attacks where a symlink is swapped between validation and read.
    let canonical_bib =
        crate::util::validate_path_access(std::path::Path::new(&req.bib_path), &state.config)?;
    crate::util::validate_path_access(std::path::Path::new(&req.output_directory), &state.config)?;

    // Validate bib file exists using the canonical path.
    if !canonical_bib.is_file() {
        return Err(ApiError::BadRequest {
            reason: format!("bib file does not exist: {}", req.bib_path),
        });
    }

    let delay_ms = req.delay_ms.unwrap_or(1000);
    let email = req.email.clone();

    // Parse the BibTeX file on a blocking thread (CPU-bound text processing).
    // Uses the canonical path from validate_path_access.
    let bib_path_owned = canonical_bib;
    let bib_entries = tokio::task::spawn_blocking(move || {
        let content = std::fs::read_to_string(&bib_path_owned)
            .map_err(|e| format!("failed to read bib file: {e}"))?;
        Ok::<_, String>(neuroncite_citation::bibtex::parse_bibtex(&content))
    })
    .await
    .map_err(|e| ApiError::Internal {
        reason: format!("blocking task panicked: {e}"),
    })?
    .map_err(|e: String| ApiError::Internal { reason: e })?;

    let total_entries = bib_entries.len();

    // Build the HTTP client for URL classification, DOI resolution, and downloads.
    let client = neuroncite_html::build_http_client().map_err(|e| ApiError::Internal {
        reason: format!("HTTP client initialization failed: {e}"),
    })?;

    // Resolve URLs for entries: prefer the explicit `url` field, then try
    // the multi-source DOI resolution chain.
    struct SourceEntry {
        cite_key: String,
        url: String,
        doi_source: Option<neuroncite_html::DoiSource>,
    }

    // Clone the indexing progress broadcast sender for source fetch SSE events.
    // Source events are routed through progress_tx (instead of the dedicated
    // source_tx) because the browser already holds 5 permanent SSE connections
    // at this point: logs, progress, jobs, models, citation (all opened by
    // App.tsx on mount). HTTP/1.1 allows 6 connections per host. Opening a
    // 6th SSE connection for the sources stream consumed the last slot, so the
    // fetch POST that follows had no slot available and blocked indefinitely.
    // Routing through progress_tx lets source events reach the browser over the
    // already-open progress EventSource. The progress_stream SSE handler
    // extracts the "event" field from each JSON payload and uses it as the SSE
    // event name, so source_entry_update events are dispatched to the browser's
    // source_entry_update handler on that EventSource, separate from the
    // index_progress and index_complete events from the job executor.
    let source_tx_early = state.sse.progress_tx.clone();
    let send_resolve_event = |entry: &serde_json::Value| {
        // All other SSE channels (model_tx, progress_tx) send flat JSON where
        // the "event" discriminator is a sibling of the payload fields, not a
        // wrapper around them. The frontend isSourceEntryUpdate type guard
        // checks for cite_key and status at the top level; the previous
        // {"event":..., "data":{...}} nesting placed them under "data" and
        // caused every event to be rejected by the type guard.
        let _ = source_tx_early.send(build_source_sse_payload(entry));
    };

    // Count entries that need DOI resolution (have doi but no url).
    let doi_entries_total = bib_entries
        .iter()
        .filter(|(_, e)| e.url.is_none() && e.doi.is_some())
        .count();
    let mut doi_entries_done = 0usize;

    let mut sources: Vec<SourceEntry> = Vec::new();
    for (cite_key, entry) in &bib_entries {
        if let Some(url) = &entry.url {
            // Validate the URL from the BibTeX `url` field before issuing any
            // HTTP request. This prevents the server from acting as an SSRF proxy
            // for internal services when a crafted BibTeX file is submitted.
            if let Err(e) = validate_fetch_url(url) {
                tracing::warn!(
                    cite_key = %cite_key,
                    url = %url,
                    "fetch_sources: rejecting BibTeX url field due to SSRF validation: {e}"
                );
                continue;
            }
            sources.push(SourceEntry {
                cite_key: cite_key.clone(),
                url: url.clone(),
                doi_source: None,
            });
        } else if let Some(doi) = &entry.doi {
            // Emit SSE event before starting DOI resolution for this entry,
            // so the frontend can show "resolving" status and update progress.
            send_resolve_event(&serde_json::json!({
                "cite_key": cite_key,
                "status": "resolving",
                "doi": doi,
                "doi_done": doi_entries_done,
                "doi_total": doi_entries_total,
            }));

            let resolved = neuroncite_html::resolve_doi(&client, doi, email.as_deref()).await;
            doi_entries_done += 1;
            tracing::info!(
                cite_key = %cite_key,
                doi = %doi,
                source = %resolved.source,
                url = %resolved.url,
                "DOI resolved"
            );

            // Emit SSE event after DOI resolution completes for this entry.
            send_resolve_event(&serde_json::json!({
                "cite_key": cite_key,
                "status": "resolved",
                "doi": doi,
                "url": &resolved.url,
                "doi_done": doi_entries_done,
                "doi_total": doi_entries_total,
            }));

            // Validate the DOI-resolved URL for the same SSRF reasons as the
            // explicit `url` field. The resolved URL comes from external DOI
            // resolvers (DOI.org, Crossref) and should always be http/https,
            // but a misconfigured resolver or redirect chain could return an
            // internal address.
            if let Err(e) = validate_fetch_url(&resolved.url) {
                tracing::warn!(
                    cite_key = %cite_key,
                    doi = %doi,
                    resolved_url = %resolved.url,
                    "fetch_sources: rejecting DOI-resolved url due to SSRF validation: {e}"
                );
                continue;
            }
            sources.push(SourceEntry {
                cite_key: cite_key.clone(),
                url: resolved.url,
                doi_source: Some(resolved.source),
            });
        }
    }

    let entries_with_url = sources.len();

    // Create the output directory and type-separated subfolders.
    // Downloaded PDFs are stored in output_dir/pdf/ and HTML pages in
    // output_dir/html/ so the user can browse sources grouped by format.
    let output_dir = std::path::Path::new(&req.output_directory);
    let pdf_dir = output_dir.join("pdf");
    let html_dir = output_dir.join("html");
    std::fs::create_dir_all(&pdf_dir).map_err(|e| ApiError::Internal {
        reason: format!("failed to create pdf output directory: {e}"),
    })?;
    std::fs::create_dir_all(&html_dir).map_err(|e| ApiError::Internal {
        reason: format!("failed to create html output directory: {e}"),
    })?;

    let html_cache_dir = neuroncite_html::default_cache_dir();

    // Known title/content patterns indicating bot-detection or access-denied pages.
    let bot_patterns: &[&str] = &[
        "just a moment",
        "attention required",
        "checking your browser",
        "access denied",
        "please enable cookies",
        "security check",
        "redirecting",
        "please wait",
        "verify you are human",
        "enable javascript",
        "one more step",
        "403 forbidden",
    ];

    // Reuse the progress_tx clone from the DOI resolution phase
    // for download-phase SSE events. The event type field in each payload
    // ensures the browser dispatches them to source_entry_update handlers.
    let send_source_event = |entry: &serde_json::Value| {
        // Same flat-JSON convention as send_resolve_event above.
        let _ = source_tx_early.send(build_source_sse_payload(entry));
    };

    // Recursively scan the output directory and all subdirectories for
    // existing files so already-downloaded sources can be skipped. Collects
    // lowercased filenames (without path) into a HashSet for O(1) lookup.
    let mut existing_files: std::collections::HashSet<String> = std::collections::HashSet::new();
    fn collect_filenames_recursive(
        dir: &std::path::Path,
        set: &mut std::collections::HashSet<String>,
    ) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in rd.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                collect_filenames_recursive(&path, set);
            } else {
                set.insert(entry.file_name().to_string_lossy().to_lowercase());
            }
        }
    }
    collect_filenames_recursive(output_dir, &mut existing_files);

    // --- Phase 1: Classify and download each source ---
    let mut results: Vec<serde_json::Value> = Vec::new();
    let mut _downloaded_pdfs: Vec<String> = Vec::new();
    let mut pdfs_downloaded = 0usize;
    let mut pdfs_failed = 0usize;
    let mut pdfs_skipped = 0usize;
    let mut html_fetched = 0usize;
    let mut html_failed = 0usize;
    let mut html_blocked = 0usize;
    let mut html_skipped = 0usize;

    for (i, source) in sources.iter().enumerate() {
        if i > 0 && delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }

        // Build the expected filename from BibTeX metadata for both the
        // skip-check and the actual download.
        let bib_entry = bib_entries.get(&source.cite_key);
        let base_filename = neuroncite_html::build_source_filename(
            bib_entry.map(|e| e.title.as_str()).unwrap_or(""),
            bib_entry.map(|e| e.author.as_str()).unwrap_or(""),
            bib_entry.and_then(|e| e.year.as_deref()),
            &source.cite_key,
        );

        // Check whether a file for this entry already exists in the output
        // directory. Matches the expected filename with .pdf or .html extension,
        // and also checks for cite_key-based filenames.
        let base_lower = base_filename.to_lowercase();
        let cite_lower = source.cite_key.to_lowercase();
        let already_exists = existing_files.iter().any(|f| {
            let stem = f
                .strip_suffix(".pdf")
                .or_else(|| f.strip_suffix(".html"))
                .unwrap_or(f);
            stem == base_lower || stem == cite_lower || f.contains(&cite_lower)
        });

        if already_exists {
            let source_type = if existing_files.contains(&format!("{base_lower}.pdf"))
                || existing_files.contains(&format!("{cite_lower}.pdf"))
            {
                "pdf"
            } else {
                "html"
            };
            if source_type == "pdf" {
                pdfs_skipped += 1;
            } else {
                html_skipped += 1;
            }
            let entry = serde_json::json!({
                "cite_key": source.cite_key,
                "url": source.url,
                "type": source_type,
                "status": "skipped",
                "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
            });
            send_source_event(&entry);
            results.push(entry);
            continue;
        }

        let source_type = neuroncite_html::classify_url(&client, &source.url).await;

        match source_type {
            neuroncite_html::UrlSourceType::Pdf => {
                // Download the PDF into the pdf/ subfolder.
                match neuroncite_html::download_pdf(&client, &source.url, &pdf_dir, &base_filename)
                    .await
                {
                    Ok(pdf_path) => {
                        let path_str = pdf_path.display().to_string();
                        pdfs_downloaded += 1;
                        _downloaded_pdfs.push(path_str.clone());
                        let entry = serde_json::json!({
                            "cite_key": source.cite_key,
                            "url": source.url,
                            "type": "pdf",
                            "status": "downloaded",
                            "file_path": path_str,
                            "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
                        });
                        send_source_event(&entry);
                        results.push(entry);
                    }
                    Err(e) => {
                        pdfs_failed += 1;
                        let entry = serde_json::json!({
                            "cite_key": source.cite_key,
                            "url": source.url,
                            "type": "pdf",
                            "status": "failed",
                            "error": format!("{e}"),
                            "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
                        });
                        send_source_event(&entry);
                        results.push(entry);
                    }
                }
            }
            neuroncite_html::UrlSourceType::Html => {
                match neuroncite_html::fetch_url(&client, &source.url, &html_cache_dir, true).await
                {
                    Ok(fetch_result) => {
                        let raw_html = match std::fs::read(&fetch_result.cache_path) {
                            Ok(bytes) => bytes,
                            Err(e) => {
                                html_failed += 1;
                                let entry = serde_json::json!({
                                    "cite_key": source.cite_key,
                                    "url": source.url,
                                    "type": "html",
                                    "status": "failed",
                                    "error": format!("reading cached HTML: {e}"),
                                    "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
                                });
                                send_source_event(&entry);
                                results.push(entry);
                                continue;
                            }
                        };

                        // Quality check: detect bot-detection stubs. Pages with
                        // fewer than 50 words that match a known pattern are excluded.
                        let raw_html_str = String::from_utf8_lossy(&raw_html);
                        let visible_text = neuroncite_html::extract_visible_text(&raw_html_str);
                        let word_count = visible_text.split_whitespace().count();
                        let is_blocked = if word_count < 50 {
                            let title_lower = fetch_result
                                .metadata
                                .title
                                .as_deref()
                                .unwrap_or("")
                                .to_lowercase();
                            let text_lower = visible_text.to_lowercase();
                            bot_patterns
                                .iter()
                                .any(|p| title_lower.contains(p) || text_lower.contains(p))
                        } else {
                            false
                        };

                        if is_blocked {
                            html_blocked += 1;
                            let entry = serde_json::json!({
                                "cite_key": source.cite_key,
                                "url": source.url,
                                "type": "html",
                                "status": "blocked",
                                "reason": format!("blocked: {word_count} words, bot-detection pattern matched"),
                                "title": fetch_result.metadata.title,
                                "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
                            });
                            send_source_event(&entry);
                            results.push(entry);
                            continue;
                        }

                        // Save the HTML page into the html/ subfolder within
                        // the output directory for type-separated organization.
                        let html_filename = format!("{base_filename}.html");
                        let html_path = html_dir.join(&html_filename);
                        if let Err(e) = std::fs::write(&html_path, &raw_html) {
                            tracing::warn!(
                                path = %html_path.display(),
                                error = %e,
                                "failed to save HTML to output directory"
                            );
                        }

                        html_fetched += 1;
                        let entry = serde_json::json!({
                            "cite_key": source.cite_key,
                            "url": source.url,
                            "type": "html",
                            "status": "fetched",
                            "file_path": html_path.display().to_string(),
                            "cache_path": fetch_result.cache_path.display().to_string(),
                            "title": fetch_result.metadata.title,
                            "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
                        });
                        send_source_event(&entry);
                        results.push(entry);
                    }
                    Err(e) => {
                        html_failed += 1;
                        let entry = serde_json::json!({
                            "cite_key": source.cite_key,
                            "url": source.url,
                            "type": "html",
                            "status": "failed",
                            "error": format!("{e}"),
                            "doi_resolved_via": source.doi_source.as_ref().map(|s| s.to_string()),
                        });
                        send_source_event(&entry);
                        results.push(entry);
                    }
                }
            }
        }
    }

    // --- Phase 2: Write failed-sources report (CSV + XLSX) ---
    //
    // Collects every source that could not be acquired: entries whose download
    // failed, entries blocked by bot-detection, and BibTeX entries that had
    // no URL or DOI at all. Both a machine-readable .csv and a formatted
    // .xlsx workbook are written to the output directory so the user can
    // review and manually retrieve the missing sources.

    // Collect BibTeX entries that had no URL and no DOI (skipped entirely).
    let mut no_link_entries: Vec<(&String, &neuroncite_citation::types::BibEntry)> = Vec::new();
    for (cite_key, entry) in &bib_entries {
        if entry.url.is_none() && entry.doi.is_none() {
            no_link_entries.push((cite_key, entry));
        }
    }

    // Collect failed and blocked entries from the results vector.
    let failed_results: Vec<&serde_json::Value> = results
        .iter()
        .filter(|r| {
            let status = r["status"].as_str().unwrap_or("");
            status == "failed" || status == "blocked"
        })
        .collect();

    let has_failures = !failed_results.is_empty() || !no_link_entries.is_empty();

    if has_failures {
        // Column definitions shared by CSV and XLSX: (header_name, xlsx_width).
        const REPORT_COLUMNS: &[(&str, f64)] = &[
            ("cite_key", 18.0),
            ("title", 45.0),
            ("author", 30.0),
            ("year", 8.0),
            ("status", 12.0),
            ("type", 8.0),
            ("url", 50.0),
            ("doi_resolved_via", 18.0),
            ("error", 50.0),
            ("reason", 40.0),
        ];

        // Build a flat row list that both CSV and XLSX write from. Each row is
        // a fixed-size array of 10 string fields matching REPORT_COLUMNS order.
        struct ReportRow {
            fields: [String; 10],
            /// Status value for XLSX row coloring: "failed", "blocked", or "no_link".
            status: String,
        }

        let mut report_rows: Vec<ReportRow> = Vec::new();

        // Rows for failed/blocked downloads.
        for entry in &failed_results {
            let cite_key = entry["cite_key"].as_str().unwrap_or("");
            let bib = bib_entries.get(cite_key);
            report_rows.push(ReportRow {
                fields: [
                    cite_key.to_string(),
                    bib.map(|b| b.title.as_str()).unwrap_or("").to_string(),
                    bib.map(|b| b.author.as_str()).unwrap_or("").to_string(),
                    bib.and_then(|b| b.year.as_deref())
                        .unwrap_or("")
                        .to_string(),
                    entry["status"].as_str().unwrap_or("").to_string(),
                    entry["type"].as_str().unwrap_or("").to_string(),
                    entry["url"].as_str().unwrap_or("").to_string(),
                    entry["doi_resolved_via"].as_str().unwrap_or("").to_string(),
                    entry["error"].as_str().unwrap_or("").to_string(),
                    entry["reason"].as_str().unwrap_or("").to_string(),
                ],
                status: entry["status"].as_str().unwrap_or("").to_string(),
            });
        }

        // Rows for entries without URL/DOI.
        for (cite_key, entry) in &no_link_entries {
            report_rows.push(ReportRow {
                fields: [
                    cite_key.to_string(),
                    entry.title.clone(),
                    entry.author.clone(),
                    entry.year.as_deref().unwrap_or("").to_string(),
                    "no_link".to_string(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                ],
                status: "no_link".to_string(),
            });
        }

        // --- CSV report ---
        let mut csv = String::new();
        // UTF-8 BOM so Excel opens the file with correct encoding.
        csv.push('\u{FEFF}');
        csv.push_str(
            &REPORT_COLUMNS
                .iter()
                .map(|(name, _)| *name)
                .collect::<Vec<_>>()
                .join(","),
        );
        csv.push('\n');

        // Helper: escapes a CSV field by wrapping in double quotes and doubling
        // any internal double-quote characters (RFC 4180 compliant).
        let csv_escape = |s: &str| -> String {
            if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
                format!("\"{}\"", s.replace('"', "\"\""))
            } else {
                s.to_string()
            }
        };

        for row in &report_rows {
            let escaped: Vec<String> = row.fields.iter().map(|f| csv_escape(f)).collect();
            csv.push_str(&escaped.join(","));
            csv.push('\n');
        }

        let csv_path = output_dir.join("failed_sources.csv");
        if let Err(e) = std::fs::write(&csv_path, &csv) {
            tracing::warn!(path = %csv_path.display(), error = %e, "failed to write CSV report");
        } else {
            tracing::info!(path = %csv_path.display(), "failed sources CSV report written");
        }

        // --- XLSX report ---
        // Formatted Excel workbook with status-colored rows, frozen header,
        // autofilter, and a summary sheet with aggregate statistics.
        let xlsx_result = (|| -> Result<(), rust_xlsxwriter::XlsxError> {
            use rust_xlsxwriter::{Color, Format, FormatAlign, FormatBorder, Workbook};

            let mut workbook = Workbook::new();
            let border_color = Color::RGB(0xBBBBBB);

            // -- Format definitions --

            // Header: dark navy background, white bold text, centered.
            let header_fmt = Format::new()
                .set_background_color(Color::RGB(0x1F3864))
                .set_font_color(Color::White)
                .set_bold()
                .set_text_wrap()
                .set_align(FormatAlign::Center)
                .set_align(FormatAlign::VerticalCenter)
                .set_border(FormatBorder::Thin)
                .set_border_color(border_color);

            // Base cell format with text wrap and thin borders.
            let base_fmt = Format::new()
                .set_text_wrap()
                .set_align(FormatAlign::Top)
                .set_border(FormatBorder::Thin)
                .set_border_color(border_color);

            // Status-specific row backgrounds.
            let failed_fmt = base_fmt.clone().set_background_color(Color::RGB(0xFCE4EC)); // light red
            let blocked_fmt = base_fmt.clone().set_background_color(Color::RGB(0xFFF8E1)); // light amber
            let no_link_fmt = base_fmt.clone().set_background_color(Color::RGB(0xF3E5F5)); // light purple

            // Status badge formats (bold, centered, colored background).
            let failed_badge = Format::new()
                .set_background_color(Color::RGB(0xE53935))
                .set_font_color(Color::White)
                .set_bold()
                .set_align(FormatAlign::Center)
                .set_align(FormatAlign::VerticalCenter)
                .set_border(FormatBorder::Thin)
                .set_border_color(border_color);
            let blocked_badge = Format::new()
                .set_background_color(Color::RGB(0xFFA000))
                .set_font_color(Color::White)
                .set_bold()
                .set_align(FormatAlign::Center)
                .set_align(FormatAlign::VerticalCenter)
                .set_border(FormatBorder::Thin)
                .set_border_color(border_color);
            let no_link_badge = Format::new()
                .set_background_color(Color::RGB(0x8E24AA))
                .set_font_color(Color::White)
                .set_bold()
                .set_align(FormatAlign::Center)
                .set_align(FormatAlign::VerticalCenter)
                .set_border(FormatBorder::Thin)
                .set_border_color(border_color);

            // URL column format: blue underlined text on the row background.
            let url_failed_fmt = failed_fmt
                .clone()
                .set_font_color(Color::RGB(0x1565C0))
                .set_underline(rust_xlsxwriter::FormatUnderline::Single);
            let url_blocked_fmt = blocked_fmt
                .clone()
                .set_font_color(Color::RGB(0x1565C0))
                .set_underline(rust_xlsxwriter::FormatUnderline::Single);

            // ---- Sheet 1: Failed Sources (detail rows) ----
            let ws = workbook.add_worksheet();
            ws.set_name("Failed Sources")?;

            // Write header row.
            for (col, (name, width)) in REPORT_COLUMNS.iter().enumerate() {
                ws.write_with_format(0, col as u16, *name, &header_fmt)?;
                ws.set_column_width(col as u16, *width)?;
            }
            ws.set_row_height(0, 22.0)?;
            ws.set_freeze_panes(1, 0)?;
            ws.autofilter(0, 0, 0, (REPORT_COLUMNS.len() - 1) as u16)?;

            // Write data rows with status-based coloring.
            for (seq, row) in report_rows.iter().enumerate() {
                let excel_row = (seq + 1) as u32;
                let (row_fmt, badge_fmt) = match row.status.as_str() {
                    "failed" => (&failed_fmt, &failed_badge),
                    "blocked" => (&blocked_fmt, &blocked_badge),
                    _ => (&no_link_fmt, &no_link_badge),
                };

                for (col, value) in row.fields.iter().enumerate() {
                    let col16 = col as u16;
                    let col_name = REPORT_COLUMNS[col].0;

                    if col_name == "status" {
                        // Status column: badge format with colored background.
                        ws.write_with_format(excel_row, col16, value.as_str(), badge_fmt)?;
                    } else if col_name == "url" && !value.is_empty() {
                        // URL column: clickable hyperlink with blue underline.
                        let url_fmt = match row.status.as_str() {
                            "failed" => &url_failed_fmt,
                            "blocked" => &url_blocked_fmt,
                            _ => row_fmt,
                        };
                        ws.write_url_with_format(excel_row, col16, value.as_str(), url_fmt)?;
                    } else {
                        ws.write_with_format(excel_row, col16, value.as_str(), row_fmt)?;
                    }
                }
                ws.set_row_height(excel_row, 40.0)?;
            }

            // ---- Sheet 2: Summary (aggregate statistics) ----
            let ws2 = workbook.add_worksheet();
            ws2.set_name("Summary")?;

            // Summary label format: bold, left-aligned.
            let label_fmt = Format::new()
                .set_bold()
                .set_align(FormatAlign::Left)
                .set_border(FormatBorder::Thin)
                .set_border_color(border_color);

            // Summary value format: right-aligned number.
            let value_fmt = Format::new()
                .set_align(FormatAlign::Right)
                .set_border(FormatBorder::Thin)
                .set_border_color(border_color);

            // Title row spanning two columns.
            let title_fmt = Format::new()
                .set_background_color(Color::RGB(0x1F3864))
                .set_font_color(Color::White)
                .set_bold()
                .set_font_size(14.0)
                .set_align(FormatAlign::Center)
                .set_border(FormatBorder::Thin)
                .set_border_color(border_color);

            ws2.set_column_width(0, 28.0)?;
            ws2.set_column_width(1, 16.0)?;
            ws2.merge_range(0, 0, 0, 1, "Failed Sources Report", &title_fmt)?;

            let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

            let summary_rows: &[(&str, String)] = &[
                ("Generated", timestamp),
                ("BibTeX file", req.bib_path.clone()),
                ("Output directory", req.output_directory.clone()),
                ("", String::new()),
                ("Total BibTeX entries", total_entries.to_string()),
                ("Entries with URL/DOI", entries_with_url.to_string()),
                ("Entries without URL/DOI", no_link_entries.len().to_string()),
                (
                    "Downloads successful",
                    (pdfs_downloaded + html_fetched).to_string(),
                ),
                ("Downloads failed", (pdfs_failed + html_failed).to_string()),
                ("Blocked (bot detection)", html_blocked.to_string()),
                ("", String::new()),
                (
                    "Total failed/missing",
                    (failed_results.len() + no_link_entries.len()).to_string(),
                ),
            ];

            for (row_idx, (label, value)) in summary_rows.iter().enumerate() {
                let r = (row_idx + 1) as u32;
                if label.is_empty() {
                    // Empty separator row.
                    continue;
                }
                ws2.write_with_format(r, 0, *label, &label_fmt)?;
                ws2.write_with_format(r, 1, value.as_str(), &value_fmt)?;
            }

            let xlsx_path = output_dir.join("failed_sources.xlsx");
            workbook.save(&xlsx_path)?;
            tracing::info!(path = %xlsx_path.display(), "failed sources XLSX report written");
            Ok(())
        })();

        if let Err(e) = xlsx_result {
            tracing::warn!(error = %e, "failed to write XLSX report");
        }
    }

    Ok(Json(crate::dto::FetchSourcesResponse {
        api_version: API_VERSION.to_string(),
        total_entries,
        entries_with_url,
        pdfs_downloaded,
        pdfs_failed,
        pdfs_skipped,
        html_fetched,
        html_failed,
        html_blocked,
        html_skipped,
        results,
    }))
}

/// Penalty factor applied to the overlap score when the BibTeX year and the
/// filename year do not match. A value of 0.5 means the score is halved,
/// making a year-mismatched candidate less likely to win but not rejecting
/// it outright. This handles cases where the BibTeX year differs from the
/// filename year (e.g., a preprint year vs publication year, or a minor
/// metadata error) while the title and author match strongly.
const YEAR_MISMATCH_PENALTY: f64 = 0.5;

/// Finds the best matching indexed file for a given BibTeX entry using
/// token-based matching on the filename. Returns the file_id and Jaccard
/// similarity score of the best match, or None if no match exceeds the threshold.
///
/// Token-based matching is used instead of Jaro-Winkler because Jaro-Winkler
/// produces misleading results for long strings (it is designed for short
/// person names). Token matching correctly handles the common case where the
/// BibTeX title and author family names appear as tokens in the filename.
///
/// Similarity metric: Jaccard similarity |A ∩ B| / |A ∪ B|.
/// The Jaccard denominator is the union size (always >= max(|A|, |B|)), which
/// makes short-file false positives much less likely than the overlap coefficient
/// (which uses min(|A|, |B|) as denominator). A 3-token file that shares 2
/// common domain words (e.g., "financial", "markets") with a 12-token BibTeX
/// entry scores 2/13 = 0.15 under Jaccard, not 2/3 = 0.67 under overlap.
///
/// Minimum token length is 4 characters. This excludes 3-letter stopwords
/// ("and", "the", "out", "can", "for", "but", "are", "its") that appear
/// across many academic titles and would otherwise generate false intersections.
///
/// Year handling: When a 4-digit year is present in both the BibTeX entry and
/// the filename, matching years receive no penalty while mismatched years
/// receive a score penalty (YEAR_MISMATCH_PENALTY). This allows strong title
/// matches to overcome minor year discrepancies (e.g., preprint vs publication
/// year, BibTeX metadata errors) while still preferring year-matching candidates.
pub(crate) fn find_best_file_match(
    author: &str,
    title: &str,
    year: Option<&str>,
    file_lookup: &[(i64, String)],
) -> Option<(i64, f64)> {
    if file_lookup.is_empty() || (author.is_empty() && title.is_empty()) {
        return None;
    }

    // Extract the 4-digit year string from the BibTeX year field if present.
    // Year fields in BibTeX can contain extra text (e.g., "1993/04"), so we
    // extract just the first 4-digit sequence.
    let bib_year: Option<String> = year.and_then(|y| {
        let digits: String = y.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() >= 4 {
            Some(digits[..4].to_string())
        } else {
            None
        }
    });

    let combined = format!("{} {}", author, title).to_lowercase();
    // Minimum token length of 4 excludes 3-letter stopwords ("and", "the",
    // "out", "can", "for", "but", "are", "its") that appear across many
    // academic titles and would otherwise inflate intersection counts.
    let search_tokens: std::collections::HashSet<String> = combined
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 4)
        .map(|w| w.to_string())
        .collect();

    if search_tokens.is_empty() {
        return None;
    }

    let mut best_score = 0.0_f64;
    let mut best_id = None;

    for (file_id, filename) in file_lookup {
        // Determine the year penalty for this candidate. When both the BibTeX
        // entry and the filename have year-like tokens and none match, apply
        // YEAR_MISMATCH_PENALTY. When either has no year or they match, no
        // penalty is applied.
        let year_factor = if let Some(ref by) = bib_year {
            let file_years: Vec<String> = filename
                .split(|c: char| !c.is_ascii_digit())
                .filter(|s| s.len() == 4 && s.starts_with(['1', '2']))
                .map(|s| s.to_string())
                .collect();

            if !file_years.is_empty() && !file_years.contains(by) {
                YEAR_MISMATCH_PENALTY
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Lowercase the filename for case-insensitive comparison.
        // Search tokens are already lowercased from the combined string.
        let filename_lower = filename.to_lowercase();
        // Minimum token length of 4 mirrors the search_tokens filter above.
        let file_tokens: std::collections::HashSet<&str> = filename_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4)
            .collect();

        if file_tokens.is_empty() {
            continue;
        }

        let intersection = search_tokens
            .iter()
            .filter(|t| file_tokens.contains(t.as_str()))
            .count();

        // Jaccard similarity: |A ∩ B| / |A ∪ B|, with year penalty applied
        // to the final score. The union denominator is always >= max(|A|, |B|),
        // which prevents short-filename false positives that the overlap
        // coefficient (min denominator) was prone to.
        let union_size = search_tokens.len() + file_tokens.len() - intersection;
        let raw_score = if union_size > 0 {
            intersection as f64 / union_size as f64
        } else {
            0.0
        };
        let score = raw_score * year_factor;

        if score > best_score {
            best_score = score;
            best_id = Some(*file_id);
        }
    }

    // Require at least 30% Jaccard similarity to accept a match.
    if best_score >= 0.3 {
        best_id.map(|id| (id, best_score))
    } else {
        None
    }
}

/// Finds the best matching file on disk for a BibTeX entry by applying
/// multi-strategy matching against a list of existing filenames.
///
/// Strategy 1 (fast path): exact cite_key match. Checks whether the file
/// stem equals the cite_key or the filename contains the cite_key. This
/// handles files downloaded by NeuronCite's built-in fetcher (which names
/// files by cite_key) and backward-compatible naming.
///
/// Strategy 2 (Jaccard similarity): combines author + title from BibTeX,
/// tokenizes on non-alphanumeric boundaries (keeping tokens with length >= 4
/// to exclude 3-letter stopwords), and computes Jaccard similarity
/// |A ∩ B| / |A ∪ B| against each candidate filename. Year mismatch penalty
/// (YEAR_MISMATCH_PENALTY) is applied when both the BibTeX entry and the
/// filename contain different 4-digit years.
///
/// Returns (original_filename, score) for the best match above the 0.30
/// Jaccard threshold, or None when no candidate qualifies.
/// Recursively collects all PDF and HTML files from a directory tree.
/// Each entry is stored as (lowercased_filename, relative_path_from_root).
/// The relative path uses forward slashes for cross-platform display
/// consistency. Descends into all subdirectories at arbitrary depth.
fn collect_files_recursive(
    dir: &std::path::Path,
    root: &std::path::Path,
    collected: &mut Vec<(String, String)>,
) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            collect_files_recursive(&path, root, collected);
        } else {
            let name = entry.file_name().to_string_lossy().to_string();
            let lower = name.to_lowercase();
            if lower.ends_with(".pdf") || lower.ends_with(".html") {
                // Build a relative path from the root output directory.
                // Uses forward slashes for consistent display on all platforms.
                let rel_path = path
                    .strip_prefix(root)
                    .map(|p| p.to_string_lossy().replace('\\', "/"))
                    .unwrap_or(name);
                collected.push((lower, rel_path));
            }
        }
    }
}

/// Finds ALL matching files in a directory listing for a BibTeX entry.
/// Returns a Vec of (relative_path, score) for every file that exceeds
/// the 0.30 Jaccard similarity threshold, sorted by score descending.
///
/// Matching strategy:
/// 1. Exact cite_key match against file stem or substring (score 1.0).
/// 2. Jaccard similarity on author+title vs filename tokens with year-mismatch
///    penalty. Uses the same algorithm as find_best_file_match: minimum token
///    length of 4 (excludes 3-letter stopwords) and Jaccard |A∩B|/|A∪B|
///    instead of the overlap coefficient, preventing false positives from
///    short filenames that share only common domain words.
///
/// Used by parse_bib and bib_report to detect file existence AND
/// identify duplicate files (same BibTeX entry matched by multiple
/// files in different subdirectories).
fn find_all_disk_file_matches(
    author: &str,
    title: &str,
    year: Option<&str>,
    cite_key: &str,
    existing_files: &[(String, String)], // (lowercased_filename, relative_path)
) -> Vec<(String, f64)> {
    if existing_files.is_empty() {
        return Vec::new();
    }

    let cite_lower = cite_key.to_lowercase();

    // Strategy 1: collect all cite_key exact matches. Multiple files in
    // different subdirectories can match the same cite_key.
    let mut cite_matches: Vec<(String, f64)> = Vec::new();
    for (lower, rel_path) in existing_files {
        let stem = lower
            .strip_suffix(".pdf")
            .or_else(|| lower.strip_suffix(".html"))
            .unwrap_or(lower);
        if stem == cite_lower || lower.contains(&cite_lower) {
            cite_matches.push((rel_path.clone(), 1.0));
        }
    }
    if !cite_matches.is_empty() {
        return cite_matches;
    }

    // Strategy 2: token-overlap matching using BibTeX author + title.
    if author.is_empty() && title.is_empty() {
        return Vec::new();
    }

    let bib_year: Option<String> = year.and_then(|y| {
        let digits: String = y.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() >= 4 {
            Some(digits[..4].to_string())
        } else {
            None
        }
    });

    let combined = format!("{} {}", author, title).to_lowercase();
    // Minimum token length of 4 excludes 3-letter stopwords ("and", "the",
    // "out", "can", "for", "but", "are", "its") that appear in many academic
    // titles and filenames and would otherwise create false intersections.
    let search_tokens: std::collections::HashSet<String> = combined
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 4)
        .map(|w| w.to_string())
        .collect();

    if search_tokens.is_empty() {
        return Vec::new();
    }

    let mut matches: Vec<(String, f64)> = Vec::new();

    for (lower, rel_path) in existing_files {
        let stem = lower
            .strip_suffix(".pdf")
            .or_else(|| lower.strip_suffix(".html"))
            .unwrap_or(lower);

        let year_factor = if let Some(ref by) = bib_year {
            let file_years: Vec<String> = stem
                .split(|c: char| !c.is_ascii_digit())
                .filter(|s| s.len() == 4 && s.starts_with(['1', '2']))
                .map(|s| s.to_string())
                .collect();
            if !file_years.is_empty() && !file_years.contains(by) {
                YEAR_MISMATCH_PENALTY
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Minimum token length of 4 mirrors the search_tokens filter above.
        let file_tokens: std::collections::HashSet<&str> = stem
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4)
            .collect();

        if file_tokens.is_empty() {
            continue;
        }

        let intersection = search_tokens
            .iter()
            .filter(|t| file_tokens.contains(t.as_str()))
            .count();

        // Jaccard similarity: |A ∩ B| / |A ∪ B|, with year penalty applied.
        // The union denominator prevents short-filename false positives that
        // the overlap coefficient (min denominator) was prone to produce.
        let union_size = search_tokens.len() + file_tokens.len() - intersection;
        let raw_score = if union_size > 0 {
            intersection as f64 / union_size as f64
        } else {
            0.0
        };
        let score = raw_score * year_factor;

        if score >= 0.3 {
            matches.push((rel_path.clone(), score));
        }
    }

    // Sort by score descending so the best match is first.
    matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    matches
}

/// Builds a JSON value containing verdict counts, correction statistics,
/// and critical alert count from the export data.
fn build_export_summary(
    verdicts: &HashMap<String, i64>,
    done_rows: &[CitationRow],
    alerts: &[Alert],
) -> serde_json::Value {
    let corrections_suggested = done_rows
        .iter()
        .filter(|r| r.result_json.is_some())
        .filter_map(|r| {
            let entry: SubmitEntry = serde_json::from_str(r.result_json.as_ref()?).ok()?;
            Some(entry.latex_correction)
        })
        .filter(|c| c.correction_type != CorrectionType::None)
        .count();

    let critical_alerts = alerts.iter().filter(|a| a.flag == "critical").count();

    serde_json::json!({
        "supported": verdicts.get("supported").copied().unwrap_or(0),
        "partial": verdicts.get("partial").copied().unwrap_or(0),
        "unsupported": verdicts.get("unsupported").copied().unwrap_or(0),
        "not_found": verdicts.get("not_found").copied().unwrap_or(0),
        "wrong_source": verdicts.get("wrong_source").copied().unwrap_or(0),
        "unverifiable": verdicts.get("unverifiable").copied().unwrap_or(0),
        "corrections_suggested": corrections_suggested,
        "critical_alerts": critical_alerts,
    })
}

/// Constructs the flat JSON string for a `source_entry_update` SSE event.
///
/// Clones the entry object and inserts the `"event"` discriminator at the
/// top level, producing a flat structure such as:
///
///   {"event":"source_entry_update","cite_key":"fama1970","status":"resolved",...}
///
/// All SSE channels in this application (model_tx, progress_tx) follow the
/// same flat convention where the event name is a sibling of the payload
/// fields. The previous `{"event":...,"data":{...}}` nesting placed `cite_key`
/// and `status` under `"data"`, causing the frontend `isSourceEntryUpdate`
/// type guard to reject every event and leave the progress bar stuck at zero.
fn build_source_sse_payload(entry: &serde_json::Value) -> String {
    let mut payload = entry.clone();
    if let serde_json::Value::Object(ref mut map) = payload {
        map.insert(
            "event".to_string(),
            serde_json::Value::String("source_entry_update".to_string()),
        );
    }
    payload.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-CIT-MATCH-001: BUG-003 regression: `find_best_file_match` returns the
    /// correct file even when the BibTeX year (2000) differs from the filename
    /// year (1997). Before the fix, year mismatch caused hard rejection and
    /// the function returned None. The fix applies a penalty factor instead,
    /// so a strong title match can overcome a year discrepancy.
    #[test]
    fn t_cit_match_001_year_mismatch_penalty_not_reject() {
        let files = vec![
            (1, "Herd behavior and aggregate fluctuations in financial markets (Cont & Bouchaud, 1997) [arXiv PDF].pdf".to_string()),
            (2, "Efficient Capital Markets A Review of Theory and Empirical Work (Fama, 1970).pdf".to_string()),
        ];

        // BibTeX year is 2000 (publication), filename year is 1997 (arXiv preprint).
        let result = find_best_file_match(
            "Cont, Rama and Bouchaud, Jean-Philippe",
            "Herd behavior and aggregate fluctuations in financial markets",
            Some("2000"),
            &files,
        );

        assert!(
            result.is_some(),
            "year mismatch must not reject a strong title match"
        );
        let (file_id, score) = result.unwrap();
        assert_eq!(
            file_id, 1,
            "must match file ID 1 (Cont & Bouchaud), got file ID {file_id}"
        );
        // Score is penalized but still above the 0.3 threshold.
        assert!(
            score >= 0.3,
            "penalized score must still exceed the 0.3 threshold, got {score}"
        );
        assert!(
            score < 1.0,
            "penalized score must be below 1.0, got {score}"
        );
    }

    /// T-CIT-MATCH-002: When the BibTeX year matches the filename year, no
    /// penalty is applied and the match score is higher than the year-mismatch
    /// case.
    #[test]
    fn t_cit_match_002_year_match_no_penalty() {
        let files = vec![(
            1,
            "Efficient Capital Markets A Review of Theory and Empirical Work (Fama, 1970).pdf"
                .to_string(),
        )];

        let result = find_best_file_match(
            "Fama, Eugene F.",
            "Efficient Capital Markets: A Review of Theory and Empirical Work",
            Some("1970"),
            &files,
        );

        assert!(result.is_some(), "matching year must produce a result");
        let (file_id, score) = result.unwrap();
        assert_eq!(file_id, 1);
        // With matching year, no penalty is applied.
        assert!(score > 0.5, "year-matching score must be high, got {score}");
    }

    /// T-CIT-MATCH-003: When the BibTeX entry has no year field, no year
    /// penalty is applied regardless of whether the filename contains a year.
    #[test]
    fn t_cit_match_003_no_bib_year_no_penalty() {
        let files = vec![(
            1,
            "Common Risk Factors in the Returns on Stocks and Bonds (Fama & French, 1993).pdf"
                .to_string(),
        )];

        let result = find_best_file_match(
            "Fama, Eugene F. and French, Kenneth R.",
            "Common Risk Factors in the Returns on Stocks and Bonds",
            None, // No year in BibTeX
            &files,
        );

        assert!(result.is_some(), "missing year must not block matching");
        let (file_id, _score) = result.unwrap();
        assert_eq!(file_id, 1);
    }

    /// T-CIT-MATCH-004: When the file lookup is empty, returns None.
    #[test]
    fn t_cit_match_004_empty_file_lookup() {
        let result = find_best_file_match("Author", "Title", Some("2020"), &[]);
        assert!(result.is_none(), "empty file lookup must return None");
    }

    /// T-CIT-MATCH-005: When both author and title are empty, returns None.
    #[test]
    fn t_cit_match_005_empty_author_and_title() {
        let files = vec![(1, "Some Paper (Author, 2020).pdf".to_string())];
        let result = find_best_file_match("", "", Some("2020"), &files);
        assert!(result.is_none(), "empty author and title must return None");
    }

    /// T-CIT-MATCH-006: The function correctly picks the best matching file
    /// when multiple candidates exist, preferring the one with higher token
    /// overlap even when another has a matching year.
    #[test]
    fn t_cit_match_006_best_overlap_wins() {
        let files = vec![
            (
                1,
                "Autoregressive Conditional Heteroskedasticity (Engle, 1982).pdf".to_string(),
            ),
            (
                2,
                "Generalized Autoregressive Conditional Heteroskedasticity (Bollerslev, 1986).pdf"
                    .to_string(),
            ),
        ];

        let result = find_best_file_match(
            "Bollerslev, Tim",
            "Generalized Autoregressive Conditional Heteroskedasticity",
            Some("1986"),
            &files,
        );

        assert!(result.is_some());
        let (file_id, _) = result.unwrap();
        assert_eq!(file_id, 2, "must pick Bollerslev 1986, not Engle 1982");
    }

    /// T-CIT-MATCH-007: The YEAR_MISMATCH_PENALTY constant is between 0
    /// (exclusive) and 1 (exclusive) -- it must penalize but not zero out
    /// the score. Uses a runtime binding to avoid clippy's
    /// `assertions_on_constants` lint.
    #[test]
    fn t_cit_match_007_penalty_constant_range() {
        let penalty = YEAR_MISMATCH_PENALTY;
        assert!(
            penalty > 0.0 && penalty < 1.0,
            "YEAR_MISMATCH_PENALTY must be in (0.0, 1.0), got {penalty}"
        );
    }

    /// T-CIT-MATCH-008: Regression -- a BibTeX entry for a stock-returns
    /// paper must not match "Comparing measures of sample skewness and
    /// kurtosis.pdf". The previous overlap-coefficient algorithm accepted this
    /// because the shared 3-letter token "and" and the word "sample" together
    /// reached the 0.30 threshold. After switching to Jaccard similarity with
    /// minimum token length 4, both false matches are rejected.
    #[test]
    fn t_cit_match_008_no_false_positive_shared_common_word() {
        let files = vec![
            (
                1,
                "Predicting Excess Stock Returns Out of Sample Can Anything Beat the Historical Average (Campbell et al., 2008).pdf".to_string(),
            ),
            (
                2,
                "Comparing measures of sample skewness and kurtosis.pdf".to_string(),
            ),
        ];

        let result = find_best_file_match(
            "Campbell, John Y. and Thompson, Samuel B.",
            "Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?",
            Some("2008"),
            &files,
        );

        assert!(result.is_some(), "correct file must be found");
        let (file_id, _score) = result.unwrap();
        assert_eq!(
            file_id, 1,
            "must match the correct Campbell paper, not the skewness/kurtosis file"
        );
    }

    /// T-CIT-MATCH-009: Regression -- "The econometrics of financial markets.pdf"
    /// must not match cont2000 ("Herd Behavior and Aggregate Fluctuations in
    /// Financial Markets"). The old overlap coefficient gave score 2/3 = 0.67
    /// because the 3-token file shared "financial" and "markets". Jaccard gives
    /// 2/12 = 0.17, which is below threshold, so only the correct Cont/Bouchaud
    /// file is returned.
    #[test]
    fn t_cit_match_009_no_false_positive_financial_markets() {
        let files = vec![
            (
                1,
                "Herd behavior and aggregate fluctuations in financial markets (Cont & Bouchaud, 1997) [arXiv PDF].pdf".to_string(),
            ),
            (
                2,
                "The econometrics of financial markets.pdf".to_string(),
            ),
        ];

        let result = find_best_file_match(
            "Cont, Rama and Bouchaud, Jean-Philippe",
            "Herd Behavior and Aggregate Fluctuations in Financial Markets",
            Some("2000"),
            &files,
        );

        assert!(result.is_some(), "correct file must be found");
        let (file_id, _score) = result.unwrap();
        assert_eq!(
            file_id, 1,
            "must match the Cont & Bouchaud file, not the econometrics textbook"
        );
    }

    /// T-CIT-MATCH-010: Regression -- "The econometrics of financial markets.pdf"
    /// must not match a short-title book entry whose only meaningful token is
    /// "econometrics". The old overlap coefficient gave score 1/min(2,3) = 0.5.
    /// Jaccard gives 1/(2+3-1) = 0.25, which is below threshold.
    #[test]
    fn t_cit_match_010_no_false_positive_single_word_title() {
        let files = vec![
            (1, "Econometrics (Hansen, 2022).pdf".to_string()),
            (2, "The econometrics of financial markets.pdf".to_string()),
        ];

        let result = find_best_file_match("Hansen, Bruce E.", "Econometrics", Some("2022"), &files);

        assert!(result.is_some(), "correct file must be found");
        let (file_id, _score) = result.unwrap();
        assert_eq!(
            file_id, 1,
            "must match the Hansen Econometrics book, not the financial markets textbook"
        );
    }

    /// T-CIT-MATCH-011: Regression -- "The Akaike Information Criterion
    /// Background Derivation Properties Application Interpretation and
    /// Refinements.pdf" (the cavanaugh2019 file) must not be flagged as a
    /// match for akaike1973 ("Information Theory and an Extension of the
    /// Maximum Likelihood Principle"). The two titles share only "akaike" and
    /// "information" as 4+ char tokens. Jaccard = 2/15 = 0.13 < 0.30 threshold.
    #[test]
    fn t_cit_match_011_no_cross_entry_false_positive_akaike() {
        let files = vec![
            (
                1,
                "Information Theory and an Extension of the Maximum Likelihood Principle (Akaike, 1973).pdf".to_string(),
            ),
            (
                2,
                "The Akaike Information Criterion Background Derivation Properties Application Interpretation and Refinements.pdf".to_string(),
            ),
        ];

        let result = find_best_file_match(
            "Akaike, Hirotugu",
            "Information Theory and an Extension of the Maximum Likelihood Principle",
            Some("1973"),
            &files,
        );

        assert!(result.is_some(), "correct file must be found");
        let (file_id, _score) = result.unwrap();
        assert_eq!(
            file_id, 1,
            "must match the original Akaike 1973 paper, not the Cavanaugh survey"
        );
    }

    /// T-CIT-MATCH-012: Regression -- find_all_disk_file_matches must not
    /// return more than one result when only one file genuinely matches the
    /// BibTeX entry. Before the fix, shared stopwords ("and", "the") and
    /// common domain tokens caused a second unrelated file to exceed the
    /// overlap threshold, triggering a false "duplicate" status in the UI.
    #[test]
    fn t_cit_match_012_no_spurious_duplicate_in_find_all() {
        // The correct file for hyndman2006 and an unrelated file that shares
        // only "measures" as a 4+ char token.
        let existing_files = vec![
            (
                "another look at measures of forecast accuracy (hyndman & koehler, 2006) [author pdf].pdf".to_string(),
                "Another Look at Measures of Forecast Accuracy (Hyndman & Koehler, 2006) [Author PDF].pdf".to_string(),
            ),
            (
                "comparing measures of sample skewness and kurtosis.pdf".to_string(),
                "Comparing measures of sample skewness and kurtosis.pdf".to_string(),
            ),
        ];

        let matches = find_all_disk_file_matches(
            "Hyndman, Rob J. and Koehler, Anne B.",
            "Another Look at Measures of Forecast Accuracy",
            Some("2006"),
            "hyndman2006",
            &existing_files,
        );

        assert_eq!(
            matches.len(),
            1,
            "only one file must match hyndman2006; spurious duplicate found: {:?}",
            matches
        );
        assert!(
            matches[0].0.contains("Hyndman"),
            "the single match must be the correct Hyndman file, got: {}",
            matches[0].0
        );
    }

    /// T-SOURCE-SSE-001: Regression -- build_source_sse_payload must produce
    /// flat JSON where cite_key and status are at the top level. Before the fix,
    /// the closures wrapped the entry under a nested "data" field, causing the
    /// frontend isSourceEntryUpdate type guard to reject every event (it checks
    /// for cite_key and status at the top level). The progress bar stayed at
    /// zero and the results list never updated during a fetch.
    #[test]
    fn t_source_sse_001_payload_is_flat_for_doi_resolve_event() {
        let entry = serde_json::json!({
            "cite_key": "fama1970",
            "status": "resolving",
            "doi": "10.1093/rfs/hhm055",
            "doi_done": 2,
            "doi_total": 10,
        });
        let payload_str = build_source_sse_payload(&entry);
        let parsed: serde_json::Value =
            serde_json::from_str(&payload_str).expect("payload must be valid JSON");

        assert_eq!(
            parsed["cite_key"], "fama1970",
            "cite_key must be at the top level, not nested under 'data'"
        );
        assert_eq!(
            parsed["status"], "resolving",
            "status must be at the top level, not nested under 'data'"
        );
        assert_eq!(
            parsed["event"], "source_entry_update",
            "event discriminator must be present at the top level"
        );
        assert!(
            parsed.get("data").is_none(),
            "no nested 'data' key: the old wrapper format is rejected by the frontend type guard"
        );
    }

    /// T-SOURCE-SSE-002: Regression -- build_source_sse_payload must flatten a
    /// download-status entry the same way it flattens a DOI-resolve entry.
    /// Covers the send_source_event closure used during Phase 2 (downloading).
    #[test]
    fn t_source_sse_002_payload_is_flat_for_download_event() {
        let entry = serde_json::json!({
            "cite_key": "campbell2008",
            "url": "https://example.com/paper.pdf",
            "type": "pdf",
            "status": "downloaded",
            "file_path": "/output/paper.pdf",
            "doi_resolved_via": "unpaywall",
        });
        let payload_str = build_source_sse_payload(&entry);
        let parsed: serde_json::Value =
            serde_json::from_str(&payload_str).expect("payload must be valid JSON");

        assert_eq!(parsed["cite_key"], "campbell2008");
        assert_eq!(parsed["status"], "downloaded");
        assert_eq!(parsed["url"], "https://example.com/paper.pdf");
        assert_eq!(parsed["event"], "source_entry_update");
        assert!(
            parsed.get("data").is_none(),
            "no nested 'data' key: the old wrapper format is rejected by the frontend type guard"
        );
    }

    /// T-SOURCE-SSE-003: build_source_sse_payload must not modify the original
    /// entry (it clones before inserting the event field). This matters because
    /// the same entry value is pushed into the HTTP response results vector
    /// after being passed to send_source_event; HTTP results must not carry
    /// the "event" discriminator field.
    #[test]
    fn t_source_sse_003_original_entry_is_not_modified() {
        let entry = serde_json::json!({
            "cite_key": "brock1992",
            "status": "failed",
            "error": "403 Forbidden",
        });
        let _payload = build_source_sse_payload(&entry);

        assert!(
            entry.get("event").is_none(),
            "the original entry must not be mutated: HTTP response results must not carry 'event'"
        );
    }
}
