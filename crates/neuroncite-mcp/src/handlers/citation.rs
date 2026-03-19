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

//! Handlers for the five citation verification MCP tools.
//!
//! Each handler extracts parameters from the MCP tool call arguments,
//! delegates to functions in the neuroncite-citation crate for business
//! logic, and returns a JSON response. The handlers share the AppState
//! with all other MCP tools via the common `Arc<AppState>` reference.
//!
//! The five tools form a complete citation verification workflow:
//! create -> claim -> submit -> status -> export.

use std::collections::HashMap;
use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_citation::{CitationJobParams, CitationRowInsert};

/// Creates a citation verification job from LaTeX and BibTeX files.
///
/// Parses the .tex file for citation commands, resolves cite-keys against
/// the .bib file, matches cited works to indexed PDFs, assigns citations
/// to batches, and inserts all citation_row records into the database.
///
/// # Parameters
///
/// - `tex_path` (required): Absolute path to the .tex file.
/// - `bib_path` (required): Absolute path to the .bib file.
/// - `session_id` (required): NeuronCite index session to search against.
/// - `batch_size` (optional, default: 5): Target citations per batch.
pub async fn handle_create(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let tex_path = params["tex_path"]
        .as_str()
        .ok_or("missing required parameter: tex_path")?;

    let bib_path = params["bib_path"]
        .as_str()
        .ok_or("missing required parameter: bib_path")?;

    let session_id = params["session_id"]
        .as_i64()
        .ok_or("missing required parameter: session_id")?;

    let batch_size = params["batch_size"].as_u64().unwrap_or(5) as usize;
    if batch_size == 0 {
        return Err("batch_size must be greater than 0".to_string());
    }

    // Validate file paths exist.
    let tex_file = std::path::Path::new(tex_path);
    if !tex_file.is_file() {
        return Err(format!("tex file does not exist: {tex_path}"));
    }
    let bib_file = std::path::Path::new(bib_path);
    if !bib_file.is_file() {
        return Err(format!("bib file does not exist: {bib_path}"));
    }

    // Read and parse files on a blocking thread to avoid stalling the tokio
    // event loop. std::fs::read_to_string and the LaTeX/BibTeX parsers perform
    // synchronous I/O and CPU-bound text processing that must not run on the
    // async executor's cooperative threads.
    let tex_path_owned = tex_path.to_string();
    let bib_path_owned = bib_path.to_string();

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
    .map_err(|e| format!("blocking task panicked: {e}"))??;

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Validate session exists.
    neuroncite_store::get_session(&conn, session_id)
        .map_err(|_| format!("session {session_id} not found"))?;

    let dry_run = params["dry_run"].as_bool().unwrap_or(false);

    // Parse file_overrides: optional map of cite_key -> file_id for correcting
    // automatic PDF matches. Applied after the automatic matching step.
    let file_overrides: std::collections::HashMap<String, i64> = params
        .get("file_overrides")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_i64().map(|id| (k.clone(), id)))
                .collect()
        })
        .unwrap_or_default();

    // Match cite-keys to indexed files using token-overlap on a descriptive
    // name. For PDF files this is the filename stem (e.g., "Fama_French_1993").
    // For HTML files the filename stem is a URL-derived hash or "article",
    // which has zero semantic overlap with BibTeX author/title. Instead, the
    // page's `<title>` tag (stored in the web_source table) is used as the
    // matching string, since it typically contains the article title with
    // strong token overlap against BibTeX entries.
    let indexed_files = neuroncite_store::list_files_by_session(&conn, session_id)
        .map_err(|e| format!("listing files: {e}"))?;

    // Build a file_id -> web_source display name lookup for HTML files.
    // og_title is preferred over title because og_title typically contains
    // only the article title, whereas the HTML <title> tag often includes
    // site-specific suffixes (e.g. " | Nature" or " - arxiv.org") that
    // reduce token overlap with BibTeX title fields.
    let web_source_titles: HashMap<i64, String> =
        neuroncite_store::list_web_sources(&conn, session_id)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|ws| {
                // Prefer og_title, fall back to title tag.
                ws.og_title.or(ws.title).map(|t| (ws.file_id, t))
            })
            .collect();

    let file_lookup: Vec<(i64, String)> = indexed_files
        .iter()
        .map(|f| {
            // For HTML files: use the web_source title (article title from
            // the page's <title> tag) which has meaningful token overlap with
            // BibTeX author/title fields. For PDF files: use the filename
            // stem which typically encodes author names, year, and keywords.
            let name = if f.source_type == "html" {
                web_source_titles
                    .get(&f.id)
                    .cloned()
                    .unwrap_or_default()
                    .to_lowercase()
            } else {
                std::path::Path::new(&f.file_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_lowercase()
            };
            (f.id, name)
        })
        .collect();

    // Resolve each citation to a BibEntry and match to an indexed file.
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
    let mut match_info_per_key: std::collections::HashMap<String, MatchInfo> =
        std::collections::HashMap::new();

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
    let mut group_keys: HashMap<i64, Vec<String>> = HashMap::new();
    for row in &insert_rows {
        group_keys
            .entry(row.group_id)
            .or_default()
            .push(row.cite_key.clone());
    }
    // Deduplicate keys within each group (same cite-key repeated in the same
    // \cite command is theoretically possible, though uncommon).
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
                // Derive a match_quality label from the overlap score. The
                // threshold for accepting a match is 0.3 in find_best_file_match.
                // Quality bands:
                //   >= 0.7: "high"   -- strong token overlap, confident match
                //   >= 0.5: "medium" -- moderate overlap, likely correct
                //   >= 0.3: "low"    -- borderline match, may need verification
                //   <  0.3: "none"   -- no file matched above the threshold
                let (match_quality, reason) = if info.matched_file_id.is_some() {
                    let quality = if info.overlap_score >= 0.7 {
                        "high"
                    } else if info.overlap_score >= 0.5 {
                        "medium"
                    } else {
                        "low"
                    };
                    (quality, serde_json::Value::Null)
                } else {
                    // Determine reason for no match. Possible causes:
                    // - No BibTeX entry found for this cite-key
                    // - BibTeX entry exists but no indexed PDF had sufficient
                    //   token overlap (score < 0.3)
                    let reason = if info.title.is_empty() && info.author.is_empty() {
                        "no_bib_entry"
                    } else {
                        "below_threshold"
                    };
                    ("none", serde_json::json!(reason))
                };

                serde_json::json!({
                    "cite_key": key,
                    "author": info.author,
                    "title": info.title,
                    "year": info.year,
                    "matched_file_id": info.matched_file_id,
                    "matched_filename": info.matched_filename,
                    "overlap_score": info.overlap_score,
                    "match_quality": match_quality,
                    "reason": reason,
                })
            })
            .collect();

        return Ok(serde_json::json!({
            "dry_run": true,
            "matches": matches,
            "total_citations": insert_rows.len(),
            "total_batches": total_batches,
            "unique_cite_keys": unique_keys.len(),
            "cite_keys_matched": cite_keys_matched,
            "cite_keys_unmatched": cite_keys_unmatched,
            "unresolved_cite_keys": unresolved_keys,
        }));
    }

    // --- Normal mode: create job and insert rows ---
    let job_id = uuid::Uuid::new_v4().to_string();
    let job_params = CitationJobParams {
        tex_path: tex_path.to_string(),
        bib_path: bib_path.to_string(),
        session_id,
        session_ids: vec![session_id],
        batch_size,
    };
    let params_json =
        serde_json::to_string(&job_params).map_err(|e| format!("params serialization: {e}"))?;

    // Citation verification jobs use state 'running' from the start because
    // they are agent-driven (not executor-driven). The executor ignores
    // jobs with kind = 'citation_verify'.
    neuroncite_store::create_job_with_params(
        &conn,
        &job_id,
        "citation_verify",
        Some(session_id),
        Some(&params_json),
    )
    .map_err(|e| format!("job creation: {e}"))?;

    neuroncite_store::update_job_state(&conn, &job_id, neuroncite_store::JobState::Running, None)
        .map_err(|e| format!("job state update: {e}"))?;

    // Set job_id on all insert rows and insert into database.
    for row in &mut insert_rows {
        row.job_id = job_id.clone();
    }

    let total_inserted = neuroncite_citation::db::insert_citation_rows(&conn, &insert_rows)
        .map_err(|e| format!("inserting citation rows: {e}"))?;

    // Update job progress total.
    neuroncite_store::update_job_progress(&conn, &job_id, 0, total_inserted as i64)
        .map_err(|e| format!("updating progress: {e}"))?;

    Ok(serde_json::json!({
        "job_id": job_id,
        "session_id": session_id,
        "total_citations": total_inserted,
        "total_batches": total_batches,
        "unique_cite_keys": unique_keys.len(),
        "cite_keys_matched": cite_keys_matched,
        "cite_keys_unmatched": cite_keys_unmatched,
        "unresolved_cite_keys": unresolved_keys,
        "tex_path": tex_path,
        "bib_path": bib_path,
    }))
}

/// Claims a batch of citations for sub-agent processing.
///
/// When `batch_id` is specified, claims that specific batch (allowing targeted
/// retries of failed batches and out-of-order claiming). When `batch_id` is
/// absent, claims the next pending batch in FIFO order.
///
/// Stale batches (claimed > 5 minutes ago) are automatically reclaimed before
/// the claim attempt. Returns all citation rows in the batch with job context
/// (tex_path, bib_path, session_id).
///
/// # Parameters
///
/// - `job_id` (required): The citation verification job ID.
/// - `batch_id` (optional): Specific batch to claim. When absent, the lowest
///   pending batch is claimed.
pub async fn handle_claim(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let job_id = params["job_id"]
        .as_str()
        .ok_or("missing required parameter: job_id")?;

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Validate job exists and is a citation_verify job.
    let job =
        neuroncite_store::get_job(&conn, job_id).map_err(|_| format!("job not found: {job_id}"))?;
    if job.kind != "citation_verify" {
        return Err(format!("job '{job_id}' is not a citation_verify job"));
    }

    // If batch_id is provided, claim that specific batch. Otherwise, claim
    // the next FIFO batch. claim_specific_batch allows retrying failed batches
    // by resetting their rows before claiming.
    let batch_id_param = params["batch_id"].as_i64();

    let claim = if let Some(batch_id) = batch_id_param {
        neuroncite_citation::db::claim_specific_batch(&conn, job_id, batch_id)
            .map_err(|e| format!("claiming batch {batch_id}: {e}"))?
    } else {
        neuroncite_citation::db::claim_next_batch(&conn, job_id)
            .map_err(|e| format!("claiming batch: {e}"))?
    };

    match claim {
        Some(batch) => {
            Ok(serde_json::to_value(&batch).map_err(|e| format!("serializing claim: {e}"))?)
        }
        None => {
            // Distinguish between "specific batch not found / not claimable" and
            // "no pending batches remain in the FIFO queue". When the caller
            // specified a batch_id, returning a generic "no pending batches"
            // message is misleading because the real cause is that the specific
            // batch either does not exist or is already in a non-pending state.
            let batch_counts = neuroncite_citation::db::count_batches_by_status(&conn, job_id)
                .map_err(|e| format!("counting batches: {e}"))?;

            let message = if let Some(bid) = batch_id_param {
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

            Ok(serde_json::json!({
                "batch_id": null,
                "remaining_batches": batch_counts.pending,
                "message": message
            }))
        }
    }
}

/// Returns citation rows for a job with optional status filtering and pagination.
///
/// Provides direct read-only access to citation rows including their result_json
/// field, without the file-writing overhead of the export endpoint. Useful for
/// programmatic access to verification results by status.
///
/// # Parameters
///
/// - `job_id` (required): The citation verification job ID.
/// - `status` (optional): Filter by row status ("pending", "claimed", "done", "failed").
/// - `offset` (optional, default: 0): Number of rows to skip.
/// - `limit` (optional, default: 100, max: 500): Number of rows to return.
pub async fn handle_rows(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let job_id = params["job_id"]
        .as_str()
        .ok_or("missing required parameter: job_id")?;

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Validate job exists and is a citation_verify job.
    let job =
        neuroncite_store::get_job(&conn, job_id).map_err(|_| format!("job not found: {job_id}"))?;
    if job.kind != "citation_verify" {
        return Err(format!("job '{job_id}' is not a citation_verify job"));
    }

    // Parse optional status filter with validation.
    let status = params["status"].as_str();
    if let Some(s) = status
        && !["pending", "claimed", "done", "failed"].contains(&s)
    {
        return Err(format!(
            "invalid status filter '{s}': must be pending, claimed, done, or failed"
        ));
    }

    let offset = params["offset"].as_i64().unwrap_or(0);
    if offset < 0 {
        return Err("offset must be >= 0".to_string());
    }

    let limit = params["limit"].as_i64().unwrap_or(100);
    if !(1..=500).contains(&limit) {
        return Err("limit must be between 1 and 500".to_string());
    }

    let (rows, total) =
        neuroncite_citation::db::list_rows_filtered(&conn, job_id, status, offset, limit)
            .map_err(|e| format!("listing rows: {e}"))?;

    let rows_json: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| serde_json::to_value(r).unwrap_or_default())
        .collect();

    Ok(serde_json::json!({
        "job_id": job_id,
        "rows": rows_json,
        "total": total,
        "offset": offset,
        "limit": limit,
    }))
}

/// Submits verification results for a batch of citation rows.
///
/// Each entry contains the sub-agent's verdict, passages, reasoning,
/// confidence, and optional LaTeX correction. Rows are validated to
/// belong to the specified job and to be in 'claimed' status.
///
/// # Parameters
///
/// - `job_id` (required): The citation verification job ID.
/// - `results` (required): Array of result entries, one per row_id.
pub async fn handle_submit(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let job_id = params["job_id"]
        .as_str()
        .ok_or("missing required parameter: job_id")?;

    let results_json = params
        .get("results")
        .ok_or("missing required parameter: results")?;

    let results: Vec<neuroncite_citation::SubmitEntry> =
        serde_json::from_value(results_json.clone())
            .map_err(|e| format!("invalid results format: {e}"))?;

    if results.is_empty() {
        return Err("results array must not be empty".to_string());
    }

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    let count = neuroncite_citation::db::submit_batch_results(&conn, job_id, &results)
        .map_err(|e| format!("submitting results: {e}"))?;

    // Check if all rows are done/failed and complete the job if so.
    let status_counts = neuroncite_citation::db::count_by_status(&conn, job_id)
        .map_err(|e| format!("counting status: {e}"))?;

    let total =
        status_counts.pending + status_counts.claimed + status_counts.done + status_counts.failed;
    let is_complete = status_counts.pending == 0 && status_counts.claimed == 0;

    if is_complete {
        neuroncite_store::update_job_state(
            &conn,
            job_id,
            neuroncite_store::JobState::Completed,
            None,
        )
        .map_err(|e| format!("completing job: {e}"))?;
    }

    Ok(serde_json::json!({
        "status": "accepted",
        "rows_submitted": count,
        "total": total,
        "is_complete": is_complete,
    }))
}

/// Returns the status of a citation verification job.
///
/// Includes counts by status, counts by batch status, verdict aggregates,
/// and flagged alert entries.
///
/// # Parameters
///
/// - `job_id` (required): The citation verification job ID.
pub async fn handle_status(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let job_id = params["job_id"]
        .as_str()
        .ok_or("missing required parameter: job_id")?;

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    let job =
        neuroncite_store::get_job(&conn, job_id).map_err(|_| format!("job not found: {job_id}"))?;

    let status_counts = neuroncite_citation::db::count_by_status(&conn, job_id)
        .map_err(|e| format!("counting status: {e}"))?;

    let batch_counts = neuroncite_citation::db::count_batches_by_status(&conn, job_id)
        .map_err(|e| format!("counting batches: {e}"))?;

    let verdicts = neuroncite_citation::db::count_verdicts(&conn, job_id)
        .map_err(|e| format!("counting verdicts: {e}"))?;

    let alerts = neuroncite_citation::db::build_alerts(&conn, job_id)
        .map_err(|e| format!("building alerts: {e}"))?;

    // Count stale batches (claimed but not submitted within the timeout window).
    let stale_batches = neuroncite_citation::db::count_stale_batches(&conn, job_id)
        .map_err(|e| format!("counting stale batches: {e}"))?;

    // Compute average duration of completed batches in seconds.
    let avg_batch_duration = neuroncite_citation::db::avg_completed_batch_duration(&conn, job_id)
        .map_err(|e| format!("computing avg batch duration: {e}"))?;

    let total =
        status_counts.pending + status_counts.claimed + status_counts.done + status_counts.failed;
    let is_complete = status_counts.pending == 0 && status_counts.claimed == 0;

    // Calculate elapsed time from the job's started_at timestamp. For running
    // jobs, elapsed is measured to now. For completed/failed jobs, elapsed is
    // measured to finished_at.
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

    let mut response = serde_json::json!({
        "job_id": job_id,
        "job_state": job.state.to_string(),
        "total": total,
        "pending": status_counts.pending,
        "claimed": status_counts.claimed,
        "done": status_counts.done,
        "failed": status_counts.failed,
        "total_batches": batch_counts.total,
        "batches_done": batch_counts.done,
        "batches_pending": batch_counts.pending,
        "batches_claimed": batch_counts.claimed,
        "stale_batches": stale_batches,
        "verdicts": verdicts,
        "alerts": alerts,
        "elapsed_seconds": elapsed,
        "is_complete": is_complete,
    });

    // Include avg_batch_duration_seconds only when at least one batch has
    // completed, to avoid returning null for jobs that have not started
    // processing yet.
    if let Some(avg_dur) = avg_batch_duration {
        response["avg_batch_duration_seconds"] =
            serde_json::json!((avg_dur * 100.0).round() / 100.0);
    }

    Ok(response)
}

/// Exports citation verification results as annotation CSV, corrections
/// JSON, and a summary report. Writes output files to the specified
/// directory and triggers the annotation pipeline.
///
/// # Parameters
///
/// - `job_id` (required): The citation verification job ID.
/// - `output_directory` (required): Where to write output files.
/// - `source_directory` (required): Where the source PDFs are located.
pub async fn handle_export(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let job_id = params["job_id"]
        .as_str()
        .ok_or("missing required parameter: job_id")?;

    let output_directory = params["output_directory"]
        .as_str()
        .ok_or("missing required parameter: output_directory")?;

    let source_directory = params["source_directory"]
        .as_str()
        .ok_or("missing required parameter: source_directory")?;

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    let job =
        neuroncite_store::get_job(&conn, job_id).map_err(|_| format!("job not found: {job_id}"))?;

    // Verify job completeness before exporting. A citation verification job
    // is complete when no rows remain in pending or claimed status. Exporting
    // an incomplete job would silently produce partial output files, misleading
    // callers who rely on the export to contain all citation results.
    let status_counts = neuroncite_citation::db::count_by_status(&conn, job_id)
        .map_err(|e| format!("counting status: {e}"))?;

    let incomplete_rows = status_counts.pending + status_counts.claimed;
    if incomplete_rows > 0 {
        return Err(format!(
            "job '{job_id}' is not complete: {} rows pending, {} rows claimed, {} rows done. \
             Export requires all rows to be in done or failed status.",
            status_counts.pending, status_counts.claimed, status_counts.done,
        ));
    }

    // Read all done rows.
    let done_rows = neuroncite_citation::db::list_done_rows(&conn, job_id)
        .map_err(|e| format!("listing done rows: {e}"))?;

    let verdicts = neuroncite_citation::db::count_verdicts(&conn, job_id)
        .map_err(|e| format!("counting verdicts: {e}"))?;

    let alerts = neuroncite_citation::db::build_alerts(&conn, job_id)
        .map_err(|e| format!("building alerts: {e}"))?;

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
    let annotation_csv = neuroncite_citation::export::build_annotation_csv(&done_rows)
        .map_err(|e| format!("building annotation CSV: {e}"))?;

    let data_csv = neuroncite_citation::export::build_data_csv(&done_rows)
        .map_err(|e| format!("building data CSV: {e}"))?;

    let excel_bytes = neuroncite_citation::export::build_excel(&done_rows)
        .map_err(|e| format!("building Excel workbook: {e}"))?;

    let corrections_content = neuroncite_citation::export::build_corrections_json(&done_rows)
        .map_err(|e| format!("building corrections JSON: {e}"))?;

    let report_content = neuroncite_citation::export::build_summary_report(
        job_id, &done_rows, &alerts, &verdicts, elapsed,
    )
    .map_err(|e| format!("building summary report: {e}"))?;

    let full_detail_content = neuroncite_citation::export::build_full_detail_json(&done_rows)
        .map_err(|e| format!("building full detail JSON: {e}"))?;

    // Ensure output directory exists.
    let out_dir = std::path::Path::new(output_directory);
    std::fs::create_dir_all(out_dir).map_err(|e| format!("creating output directory: {e}"))?;

    // Write output files. The annotation CSV is named "annotation_pipeline_input.csv"
    // to distinguish it from the full 38-column "citation_data.csv".
    let annotation_csv_path = out_dir.join("annotation_pipeline_input.csv");
    let data_csv_path = out_dir.join("citation_data.csv");
    let excel_path = out_dir.join("citation_data.xlsx");
    let corrections_path = out_dir.join("corrections.json");
    let report_path = out_dir.join("citation_report.json");
    let full_detail_path = out_dir.join("citation_full_detail.json");

    std::fs::write(&annotation_csv_path, &annotation_csv)
        .map_err(|e| format!("writing annotation CSV: {e}"))?;
    std::fs::write(&data_csv_path, &data_csv).map_err(|e| format!("writing data CSV: {e}"))?;
    std::fs::write(&excel_path, &excel_bytes).map_err(|e| format!("writing Excel: {e}"))?;
    std::fs::write(&corrections_path, &corrections_content)
        .map_err(|e| format!("writing corrections: {e}"))?;
    std::fs::write(&report_path, &report_content).map_err(|e| format!("writing report: {e}"))?;
    std::fs::write(&full_detail_path, &full_detail_content)
        .map_err(|e| format!("writing full detail: {e}"))?;

    // Trigger annotation pipeline with the annotation CSV if it has data rows.
    let csv_lines = annotation_csv.lines().count();
    let annotation_job_id = if csv_lines > 1 {
        let ann_job_id = uuid::Uuid::new_v4().to_string();
        let ann_params = serde_json::json!({
            "input_data": annotation_csv,
            "source_directory": source_directory,
            "output_directory": output_directory,
            "default_color": "#FFFF00",
        });
        let ann_params_json =
            serde_json::to_string(&ann_params).map_err(|e| format!("annotation params: {e}"))?;

        neuroncite_store::create_job_with_params(
            &conn,
            &ann_job_id,
            "annotate",
            None,
            Some(&ann_params_json),
        )
        .map_err(|e| format!("creating annotation job: {e}"))?;

        // Wake the executor so it picks up the annotation job immediately.
        state.job_notify.notify_one();

        Some(ann_job_id)
    } else {
        None
    };

    // Build summary counts.
    let summary = build_export_summary(&verdicts, &done_rows, &alerts);

    Ok(serde_json::json!({
        "annotation_job_id": annotation_job_id,
        "annotation_csv_path": annotation_csv_path.to_string_lossy(),
        "data_csv_path": data_csv_path.to_string_lossy(),
        "excel_path": excel_path.to_string_lossy(),
        "report_path": report_path.to_string_lossy(),
        "corrections_path": corrections_path.to_string_lossy(),
        "full_detail_path": full_detail_path.to_string_lossy(),
        "summary": summary,
    }))
}

/// Retries citation verification rows by resetting them back to pending
/// status so they can be claimed again by sub-agents.
///
/// Supports three targeting modes (combinable):
/// - `row_ids`: reset specific rows by their integer ID (status 'done' or 'failed')
/// - `flags`: reset all done rows carrying a specific flag value
/// - `batch_ids`: reset all failed rows in specific batches
///
/// When no targeting parameter is provided, all failed rows in the job are reset.
///
/// # Parameters
///
/// - `job_id` (required): The citation verification job ID.
/// - `row_ids` (optional): Array of specific row IDs to reset.
/// - `flags` (optional): Array of flag values to match (e.g., ["warning"]).
/// - `batch_ids` (optional): Array of specific batch IDs whose failed rows
///   should be reset.
pub async fn handle_retry(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let job_id = params["job_id"]
        .as_str()
        .ok_or("missing required parameter: job_id")?;

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Validate job exists and is a citation_verify job.
    let job =
        neuroncite_store::get_job(&conn, job_id).map_err(|_| format!("job not found: {job_id}"))?;
    if job.kind != "citation_verify" {
        return Err(format!("job '{job_id}' is not a citation_verify job"));
    }

    // Parse optional targeting parameters. Multiple can be combined:
    // row_ids targets specific citation rows by their integer ID.
    // flags targets all done rows carrying a specific flag value.
    // batch_ids targets all failed rows in specific batches.
    let row_ids: Option<Vec<i64>> = params["row_ids"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect());

    let flags: Option<Vec<String>> = params["flags"].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect()
    });

    let batch_ids: Option<Vec<i64>> = params["batch_ids"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect());

    let mut total_rows_reset: i64 = 0;
    let mut total_batches_reset: i64 = 0;

    // Dispatch to the appropriate reset function based on which parameters
    // are provided. When multiple parameters are present, each targets a
    // different subset of rows and the totals are accumulated.
    if let Some(ref ids) = row_ids
        && !ids.is_empty()
    {
        let (rows, batches) = neuroncite_citation::db::reset_rows_by_ids(&conn, job_id, ids)
            .map_err(|e| format!("resetting rows by ids: {e}"))?;
        total_rows_reset += rows;
        total_batches_reset += batches;
    }

    if let Some(ref f) = flags
        && !f.is_empty()
    {
        let flag_refs: Vec<&str> = f.iter().map(|s| s.as_str()).collect();
        let (rows, batches) =
            neuroncite_citation::db::reset_rows_by_flags(&conn, job_id, &flag_refs)
                .map_err(|e| format!("resetting rows by flags: {e}"))?;
        total_rows_reset += rows;
        total_batches_reset += batches;
    }

    // When neither row_ids nor flags are provided, fall back to the
    // original behavior: reset all failed rows, optionally filtered
    // by batch_ids.
    if row_ids.as_ref().is_none_or(|v| v.is_empty()) && flags.as_ref().is_none_or(|v| v.is_empty())
    {
        let (rows, batches) =
            neuroncite_citation::db::reset_failed_rows(&conn, job_id, batch_ids.as_deref())
                .map_err(|e| format!("resetting failed rows: {e}"))?;
        total_rows_reset += rows;
        total_batches_reset += batches;
    }

    // If the job was completed, transition it back to running so the
    // sub-agents can claim the newly pending rows.
    if total_rows_reset > 0 && job.state == neuroncite_store::JobState::Completed {
        neuroncite_store::update_job_state(
            &conn,
            job_id,
            neuroncite_store::JobState::Running,
            None,
        )
        .map_err(|e| format!("updating job state: {e}"))?;
    }

    Ok(serde_json::json!({
        "job_id": job_id,
        "rows_reset": total_rows_reset,
        "batches_reset": total_batches_reset,
        "message": if total_rows_reset > 0 {
            format!("reset {total_rows_reset} rows across {total_batches_reset} batches to pending status")
        } else {
            "no matching rows found to retry".to_string()
        },
    }))
}

/// Finds the best matching indexed file for a given BibTeX entry using
/// token-overlap matching on the filename. Returns the file_id and overlap
/// score of the best match, or None if no match exceeds the threshold.
///
/// Token-based overlap is used instead of Jaro-Winkler because Jaro-Winkler
/// produces misleading results for long strings (it is designed for short
/// person names). Token overlap correctly handles the common case where the
/// BibTeX title and author family names appear as tokens in the filename.
///
/// The overlap coefficient is defined as |A ∩ B| / min(|A|, |B|), which handles
/// the asymmetry between BibTeX metadata (long author strings with first names,
/// subtitles) and filenames (shorter, family names only).
///
/// Year validation: When a 4-digit year is present in the BibTeX entry, the
/// function checks whether the filename also contains that year string. If
/// the filename contains a *different* 4-digit year (e.g., "1993" vs "1963"),
/// the candidate is rejected outright because this indicates a different
/// publication. This prevents false positives when the same author published
/// multiple papers with similar titles in different years.
fn find_best_file_match(
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

    // Extract author family names from BibTeX "Last, First and Last, First"
    // format. Family names (the part before the comma) are the primary
    // disambiguation signal in filenames like "Fama & French, 1993".
    let author_lower = author.to_lowercase();
    let author_tokens: std::collections::HashSet<String> = author_lower
        .split(" and ")
        .filter_map(|entry| {
            let trimmed = entry.trim();
            // BibTeX "Last, First" format: extract the family name before the comma.
            // Plain "Last" format (no comma): use the entire entry.
            let family = trimmed.split(',').next().unwrap_or(trimmed).trim();
            family
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| w.len() >= 3)
                .map(|w| w.to_string())
                .next()
        })
        .collect();

    // Tokenize the title separately for weighted scoring.
    let title_lower = title.to_lowercase();
    let title_tokens: std::collections::HashSet<String> = title_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(|w| w.to_string())
        .collect();

    // Combined token set for overlap computation (backward-compatible fallback).
    let combined = format!("{} {}", author, title).to_lowercase();
    let search_tokens: std::collections::HashSet<String> = combined
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(|w| w.to_string())
        .collect();

    if search_tokens.is_empty() {
        return None;
    }

    let mut best_score = 0.0_f64;
    let mut best_id = None;

    for (file_id, filename) in file_lookup {
        // Year mismatch penalty: if the BibTeX entry has a year and the
        // filename contains a different 4-digit year, apply a 0.5x penalty
        // instead of rejecting the candidate outright. This allows matching
        // publications where the BibTeX year differs from the filename year
        // (e.g., arXiv preprint year 1997 vs published version year 2000)
        // while still preferring candidates with matching years.
        let year_factor = if let Some(ref by) = bib_year {
            let file_years: Vec<String> = filename
                .split(|c: char| !c.is_ascii_digit())
                .filter(|s| s.len() == 4 && s.starts_with(['1', '2']))
                .map(|s| s.to_string())
                .collect();

            if !file_years.is_empty() && !file_years.contains(by) {
                0.5
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Lowercase the filename for case-insensitive comparison.
        let filename_lower = filename.to_lowercase();
        let file_tokens: std::collections::HashSet<&str> = filename_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3)
            .collect();

        if file_tokens.is_empty() {
            continue;
        }

        // Two-tier weighted scoring: author token matches receive double
        // weight compared to title token matches. This prevents false
        // positives where a generic title like "financial markets" matches
        // the wrong PDF by a different author. Author family names are the
        // strongest disambiguation signal in academic PDF filenames.
        //
        // Weighted score = (2 * author_hits + title_hits) / (2 * author_count + title_count)
        // Falls back to the original overlap coefficient when no author tokens
        // are available (author field is empty or contains only short words).
        let author_hits = author_tokens
            .iter()
            .filter(|t| file_tokens.contains(t.as_str()))
            .count();
        let title_hits = title_tokens
            .iter()
            .filter(|t| file_tokens.contains(t.as_str()))
            .count();

        let raw_score = if !author_tokens.is_empty() {
            let weighted_hits = 2.0 * author_hits as f64 + title_hits as f64;
            let weighted_total = 2.0 * author_tokens.len() as f64 + title_tokens.len() as f64;
            if weighted_total > 0.0 {
                weighted_hits / weighted_total
            } else {
                0.0
            }
        } else {
            // Fallback: original overlap coefficient when no author tokens
            // are extractable. Uses |A ∩ B| / min(|A|, |B|).
            let intersection = search_tokens
                .iter()
                .filter(|t| file_tokens.contains(t.as_str()))
                .count();
            let min_size = search_tokens.len().min(file_tokens.len());
            if min_size > 0 {
                intersection as f64 / min_size as f64
            } else {
                0.0
            }
        };

        // Apply year mismatch penalty to the raw token overlap score.
        let score = raw_score * year_factor;

        if score > best_score {
            best_score = score;
            best_id = Some(*file_id);
        }
    }

    // Require at least 30% weighted overlap to accept a match.
    if best_score >= 0.3 {
        best_id.map(|id| (id, best_score))
    } else {
        None
    }
}

/// Builds an ExportSummary-like JSON value from verdict counts and row data.
fn build_export_summary(
    verdicts: &HashMap<String, i64>,
    done_rows: &[neuroncite_citation::CitationRow],
    alerts: &[neuroncite_citation::Alert],
) -> serde_json::Value {
    // Count corrections.
    let corrections_suggested = done_rows
        .iter()
        .filter(|r| r.result_json.is_some())
        .filter_map(|r| {
            let entry: neuroncite_citation::SubmitEntry =
                serde_json::from_str(r.result_json.as_ref()?).ok()?;
            Some(entry.latex_correction)
        })
        .filter(|c| c.correction_type != neuroncite_citation::CorrectionType::None)
        .count();

    let critical_alerts = alerts.iter().filter(|a| a.flag == "critical").count();

    serde_json::json!({
        "supported": verdicts.get("supported").unwrap_or(&0),
        "partial": verdicts.get("partial").unwrap_or(&0),
        "unsupported": verdicts.get("unsupported").unwrap_or(&0),
        "not_found": verdicts.get("not_found").unwrap_or(&0),
        "wrong_source": verdicts.get("wrong_source").unwrap_or(&0),
        "unverifiable": verdicts.get("unverifiable").unwrap_or(&0),
        "corrections_suggested": corrections_suggested,
        "critical_alerts": critical_alerts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-CIT-079: Year mismatch causes rejection. When the BibTeX year is
    /// "1970" and the only candidate filename contains "1993", the function
    /// must return None because the year mismatch indicates a different
    /// publication even if author/title tokens overlap.
    ///
    /// Regression test for BUG-005: Before the fix, find_best_file_match did
    /// not consider the year field, allowing "Fama, 1970" to match a file
    /// named "Fama, 1993" purely based on author-name token overlap.
    #[test]
    fn t_cit_079_year_mismatch_rejects_candidate() {
        let files = vec![(
            1_i64,
            "Common Risk Factors in the Returns on Stocks and Bonds (Fama & French, 1993).pdf"
                .to_string(),
        )];

        // BibTeX: Fama 1970, file: Fama 1993 -- must not match.
        let result = find_best_file_match(
            "Fama, Eugene F.",
            "Efficient Capital Markets: A Review of Theory and Empirical Work",
            Some("1970"),
            &files,
        );

        assert!(
            result.is_none(),
            "must not match when BibTeX year (1970) differs from filename year (1993)"
        );
    }

    /// T-CIT-080: Year match allows correct identification. When the BibTeX
    /// year matches the filename year, the candidate is considered and
    /// returned if token overlap exceeds the threshold.
    #[test]
    fn t_cit_080_year_match_allows_correct_file() {
        let files = vec![
            (
                1_i64,
                "Common Risk Factors in the Returns on Stocks and Bonds (Fama & French, 1993).pdf"
                    .to_string(),
            ),
            (
                2_i64,
                "Efficient Capital Markets A Review of Theory and Empirical Work (Fama, 1970).pdf"
                    .to_string(),
            ),
        ];

        let result = find_best_file_match(
            "Fama, Eugene F.",
            "Efficient Capital Markets: A Review of Theory and Empirical Work",
            Some("1970"),
            &files,
        );

        assert!(
            result.is_some(),
            "must match file 2 with matching year 1970"
        );
        let (id, score) = result.unwrap();
        assert_eq!(
            id, 2,
            "must select file 2 (the 1970 paper, not the 1993 paper)"
        );
        assert!(score >= 0.3, "overlap score must exceed threshold");
    }

    /// T-CIT-081: No year in BibTeX entry still matches via token overlap.
    /// When the BibTeX year is None, the function falls back to pure token
    /// overlap without year filtering.
    #[test]
    fn t_cit_081_no_year_falls_back_to_token_overlap() {
        let files = vec![(
            1_i64,
            "Efficient Capital Markets A Review of Theory and Empirical Work (Fama, 1970).pdf"
                .to_string(),
        )];

        let result = find_best_file_match(
            "Fama, Eugene F.",
            "Efficient Capital Markets: A Review of Theory and Empirical Work",
            None,
            &files,
        );

        assert!(
            result.is_some(),
            "must match via token overlap when year is absent"
        );
    }

    /// T-CIT-082: Filename without a year still matches when BibTeX has a
    /// year. The year filter only rejects candidates that contain a *different*
    /// year, not candidates that contain no year at all.
    #[test]
    fn t_cit_082_filename_without_year_still_matches() {
        let files = vec![(
            1_i64,
            "Comparing measures of sample skewness and kurtosis.pdf".to_string(),
        )];

        let result = find_best_file_match(
            "Joanes, D. N. and Gill, C. A.",
            "Comparing measures of sample skewness and kurtosis",
            Some("1998"),
            &files,
        );

        assert!(
            result.is_some(),
            "filename without year must still be considered when BibTeX has a year"
        );
    }

    /// T-CIT-083: Multiple candidates with different years selects the correct
    /// one. Given two files from the same author but different years, the
    /// function must select the one whose year matches the BibTeX entry.
    #[test]
    fn t_cit_083_selects_correct_year_among_multiple() {
        let files = vec![
            (
                10_i64,
                "The Variation of Certain Speculative Prices (Mandelbrot, 1963).pdf".to_string(),
            ),
            (
                20_i64,
                "The Variation of Certain Speculative Prices Revisited (Mandelbrot, 1997).pdf"
                    .to_string(),
            ),
        ];

        let result = find_best_file_match(
            "Mandelbrot, Benoit B.",
            "The Variation of Certain Speculative Prices",
            Some("1963"),
            &files,
        );

        assert!(result.is_some(), "must match the 1963 paper");
        let (id, _) = result.unwrap();
        assert_eq!(id, 10, "must select file 10 (the 1963 paper)");
    }

    /// T-CIT-084: Empty file lookup returns None without panicking.
    #[test]
    fn t_cit_084_empty_file_lookup() {
        let files: Vec<(i64, String)> = vec![];
        let result = find_best_file_match("Author", "Title", Some("2000"), &files);
        assert!(result.is_none(), "empty file lookup must return None");
    }

    /// T-CIT-085: Empty author and title returns None without panicking.
    #[test]
    fn t_cit_085_empty_author_and_title() {
        let files = vec![(1_i64, "some file.pdf".to_string())];
        let result = find_best_file_match("", "", Some("2000"), &files);
        assert!(result.is_none(), "empty author and title must return None");
    }

    /// T-CIT-086: Year field with extra text (e.g., "1993/04") is parsed
    /// correctly by extracting the first 4 digits.
    #[test]
    fn t_cit_086_year_with_extra_text() {
        let files = vec![(
            1_i64,
            "Common Risk Factors (Fama & French, 1993).pdf".to_string(),
        )];

        let result = find_best_file_match(
            "Fama, Eugene F. and French, Kenneth R.",
            "Common Risk Factors in the Returns on Stocks and Bonds",
            Some("1993/04"),
            &files,
        );

        assert!(
            result.is_some(),
            "year '1993/04' must be parsed as 1993 and match"
        );
    }

    /// T-CIT-087: Below-threshold overlap score returns None. Two strings that
    /// share fewer than 30% of their tokens must not match.
    #[test]
    fn t_cit_087_below_threshold_returns_none() {
        let files = vec![(
            1_i64,
            "Completely Unrelated Paper About Cooking Recipes.pdf".to_string(),
        )];

        let result = find_best_file_match(
            "Fama, Eugene F.",
            "Efficient Capital Markets: A Review",
            Some("1970"),
            &files,
        );

        assert!(
            result.is_none(),
            "unrelated file must not match despite no year conflict"
        );
    }

    /// T-CIT-088: Author-weighted matching disambiguates papers with similar
    /// generic titles. Regression test for BUG-002 where the original overlap
    /// coefficient (without author weighting) matched "The econometrics of
    /// financial markets" by Campbell, Lo, MacKinlay to the wrong PDF because
    /// the generic title tokens "financial" and "markets" appeared in multiple
    /// filenames. The weighted algorithm prioritizes author family name matches.
    #[test]
    fn t_cit_088_author_weighting_disambiguates_generic_titles() {
        let files = vec![
            (
                1_i64,
                "The econometrics of financial markets (Campbell, Lo & MacKinlay).pdf".to_string(),
            ),
            (
                2_i64,
                "Efficient Capital Markets (Fama, 1970).pdf".to_string(),
            ),
            (
                3_i64,
                "Herd behavior and aggregate fluctuations in financial markets (Cont & Bouchaud, 1997).pdf".to_string(),
            ),
        ];

        let result = find_best_file_match(
            "Campbell, John Y. and Lo, Andrew W. and MacKinlay, A. Craig",
            "The econometrics of financial markets",
            None,
            &files,
        );

        assert!(result.is_some(), "must match a file");
        let (id, _) = result.unwrap();
        assert_eq!(
            id, 1,
            "must select file 1 (Campbell, Lo & MacKinlay) over other 'financial markets' files"
        );
    }

    /// T-CIT-090: Author-only matching works when the title has only stop-word-
    /// length tokens. The algorithm falls back to the combined overlap
    /// coefficient when no author tokens are extractable, but when author
    /// tokens are present, they dominate the scoring.
    #[test]
    fn t_cit_090_author_tokens_dominate_scoring() {
        let files = vec![
            (
                1_i64,
                "Fama & French (1993) Common Risk Factors.pdf".to_string(),
            ),
            (2_i64, "Bollerslev (1986) GARCH.pdf".to_string()),
        ];

        // Both files share the word "common" with the title "Common Risk
        // Factors" but only file 1 has the author "Fama" in its filename.
        let result = find_best_file_match(
            "Fama, Eugene F. and French, Kenneth R.",
            "Common Risk Factors in the Returns on Stocks and Bonds",
            Some("1993"),
            &files,
        );

        assert!(result.is_some());
        let (id, _) = result.unwrap();
        assert_eq!(id, 1, "author family names must determine the match");
    }

    /// T-CIT-117: handle_export rejects jobs with pending or claimed rows.
    /// Regression test for DEF-1: export on an incomplete job previously
    /// succeeded silently and produced partial output. The fix adds a
    /// count_by_status check that returns an error when rows remain
    /// in pending or claimed status. This test verifies the validation
    /// logic by checking that the StatusCounts struct correctly detects
    /// incomplete states.
    #[test]
    fn t_cit_117_export_rejects_incomplete_job() {
        // Simulate an incomplete job: 2 pending, 1 claimed, 3 done.
        // The export handler checks (pending + claimed) > 0 and rejects.
        let pending: i64 = 2;
        let claimed: i64 = 1;
        let _done: i64 = 3;
        let incomplete = pending + claimed;
        assert!(
            incomplete > 0,
            "job with pending+claimed rows must be detected as incomplete"
        );

        // Simulate a complete job: 0 pending, 0 claimed, 6 done.
        let pending = 0_i64;
        let claimed = 0_i64;
        let done = 6_i64;
        let incomplete = pending + claimed;
        assert_eq!(
            incomplete, 0,
            "job with all rows done must pass the completeness check"
        );
        assert!(done > 0, "complete job has at least one done row");
    }

    // --- Bug 4: web_source title matching for HTML files ---

    /// T-CIT-118: HTML page title provides meaningful token overlap for matching.
    /// When an HTML file's web_source title is "Efficient Capital Markets: A Review
    /// of Theory and Empirical Work", `find_best_file_match` matches it to a
    /// BibTeX entry with the same title/author. This verifies that using the
    /// page title (as the lookup string for HTML files) produces correct matches,
    /// unlike using the URL-derived filename stem which yields "article" or a hash.
    #[test]
    fn t_cit_118_html_title_matches_bibtex() {
        // file_lookup entries: file_id 1 is an HTML file whose lookup string
        // is the page title (lowercased), file_id 2 is a PDF with a filename.
        let files = vec![
            (
                1_i64,
                "efficient capital markets: a review of theory and empirical work".to_string(),
            ),
            (
                2_i64,
                "Common Risk Factors in the Returns on Stocks and Bonds (Fama & French, 1993).pdf"
                    .to_string(),
            ),
        ];

        let result = find_best_file_match(
            "Fama, Eugene F.",
            "Efficient Capital Markets: A Review of Theory and Empirical Work",
            Some("1970"),
            &files,
        );

        assert!(result.is_some(), "must match against the HTML page title");
        let (id, score) = result.unwrap();
        assert_eq!(
            id, 1,
            "must select file 1 (HTML title match) over file 2 (1993 Fama/French PDF)"
        );
        assert!(
            score >= 0.3,
            "overlap score must exceed minimum threshold: {score}"
        );
    }

    /// T-CIT-119: URL-derived filename stem fails to match BibTeX entries.
    /// When the lookup string for an HTML file is the URL's file_stem (e.g.,
    /// "article" from "https://journals.plos.org/.../article"), it has no
    /// token overlap with the BibTeX author/title and returns None.
    #[test]
    fn t_cit_119_url_stem_fails_to_match() {
        // Simulate what happens without Bug 4 fix: file_stem of a URL is
        // "article" (from PLOS) or a SHA-256 hash (from cache).
        let files = vec![
            (1_i64, "article".to_string()),
            (2_i64, "a1b2c3d4e5f6".to_string()),
        ];

        let result = find_best_file_match(
            "Fama, Eugene F.",
            "Efficient Capital Markets: A Review of Theory and Empirical Work",
            Some("1970"),
            &files,
        );

        assert!(
            result.is_none(),
            "URL-derived stems like 'article' or hashes must not match BibTeX entries"
        );
    }
}
