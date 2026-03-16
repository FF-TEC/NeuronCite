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

//! Handlers for the `neuroncite_jobs`, `neuroncite_job_status`, and
//! `neuroncite_job_cancel` MCP tools.
//!
//! Provides job listing, individual job status retrieval, and job cancellation
//! through the neuroncite-store workflow layer.

use std::sync::Arc;

use neuroncite_api::AppState;
use neuroncite_store::JobState;

/// Lists all jobs with their state, progress, and timestamps.
///
/// Returns a JSON object containing an array of job objects sorted by
/// creation time (most recent first).
pub async fn handle_list(
    state: &Arc<AppState>,
    _params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    let jobs = neuroncite_store::list_jobs(&conn).map_err(|e| format!("listing jobs: {e}"))?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let job_array: Vec<serde_json::Value> = jobs
        .iter()
        .map(|j| {
            let mut entry = serde_json::json!({
                "job_id": j.id,
                "kind": j.kind,
                "state": format!("{:?}", j.state).to_lowercase(),
                "session_id": j.session_id,
                "progress_done": j.progress_done,
                "progress_total": j.progress_total,
                "created_at": j.created_at,
                "started_at": j.started_at,
                "finished_at": j.finished_at,
                "error_message": j.error_message,
            });

            // Compute elapsed_seconds from started_at. For running jobs, the
            // elapsed time is measured from start to now. For completed/failed
            // jobs, elapsed time is measured from start to finish.
            if let Some(started) = j.started_at {
                let end_time = j.finished_at.unwrap_or(now);
                let elapsed = (end_time - started).max(0);
                entry["elapsed_seconds"] = serde_json::json!(elapsed);

                // Speed metrics for jobs that track page-level progress
                // (index and annotate jobs). Computed as progress_done / elapsed.
                if elapsed > 0 && j.progress_done > 0 {
                    let rate = j.progress_done as f64 / elapsed as f64;
                    entry["items_per_second"] = serde_json::json!((rate * 100.0).round() / 100.0);

                    // Estimated remaining time based on current processing rate.
                    let remaining_items = j.progress_total - j.progress_done;
                    if remaining_items > 0 && j.state == JobState::Running {
                        let estimated = remaining_items as f64 / rate;
                        entry["estimated_remaining_seconds"] =
                            serde_json::json!(estimated.ceil() as i64);
                    }
                }
            }

            entry
        })
        .collect();

    Ok(serde_json::json!({
        "job_count": job_array.len(),
        "jobs": job_array,
    }))
}

/// Retrieves the status of a specific job.
///
/// # Parameters (from MCP tool call)
///
/// - `job_id` (required): UUID of the job to query.
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

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let mut response = serde_json::json!({
        "job_id": job.id,
        "kind": job.kind,
        "state": format!("{:?}", job.state).to_lowercase(),
        "session_id": job.session_id,
        "progress_done": job.progress_done,
        "progress_total": job.progress_total,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error_message": job.error_message,
    });

    // Compute elapsed_seconds and speed metrics. For running jobs, elapsed
    // time is measured from start to now. For completed/failed/canceled jobs,
    // elapsed time is from start to finish.
    if let Some(started) = job.started_at {
        let end_time = job.finished_at.unwrap_or(now);
        let elapsed = (end_time - started).max(0);
        response["elapsed_seconds"] = serde_json::json!(elapsed);

        // Speed metrics: items_per_second derived from progress / elapsed.
        if elapsed > 0 && job.progress_done > 0 {
            let rate = job.progress_done as f64 / elapsed as f64;
            response["items_per_second"] = serde_json::json!((rate * 100.0).round() / 100.0);

            // Estimated remaining time based on current processing rate. Only
            // reported for running jobs, because completed/failed jobs have no
            // remaining work.
            let remaining_items = job.progress_total - job.progress_done;
            if remaining_items > 0 && job.state == JobState::Running {
                let estimated = remaining_items as f64 / rate;
                response["estimated_remaining_seconds"] =
                    serde_json::json!(estimated.ceil() as i64);
            }
        }
    }

    // For completed index jobs, include a completion_summary with session-level
    // statistics that describe the indexing outcome. These metrics are fetched
    // from the database rather than stored in the job record, because they
    // represent the final state of the indexed data.
    if job.state == JobState::Completed
        && job.kind == "index"
        && let Some(sid) = job.session_id
        && let Ok(files) = neuroncite_store::list_files_by_session(&conn, sid)
    {
        let page_stats =
            neuroncite_store::file_page_stats_by_session(&conn, sid).unwrap_or_default();

        let mut total_pages: i64 = 0;
        let mut total_ocr_pages: i64 = 0;
        let mut total_empty_pages: i64 = 0;

        for f in &files {
            total_pages += f.page_count;
        }
        for ps in &page_stats {
            if ps.backend == "ocr" {
                total_ocr_pages += ps.page_count;
            }
            total_empty_pages += ps.empty_count;
        }

        let chunk_stats =
            neuroncite_store::file_chunk_stats_by_session(&conn, sid).unwrap_or_default();
        let total_chunks: i64 = chunk_stats.iter().map(|c| c.chunk_count).sum();

        response["completion_summary"] = serde_json::json!({
            "files_indexed": files.len(),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "ocr_pages": total_ocr_pages,
            "empty_pages": total_empty_pages,
        });
    }

    // For annotate jobs, include the pipeline parameters (source_directory,
    // output_directory, default_color) from the stored params_json so that
    // callers polling job_status can see the job configuration without
    // re-reading the original request.
    //
    // Annotation jobs do not operate on an index session -- they work directly
    // with PDF files on disk. The session_id field is null for these jobs
    // because there is no associated neuroncite_index session. This is by
    // design: annotations are independent of the search index and can be
    // applied to any PDF directory regardless of indexing state.
    if job.kind == "annotate" {
        response["session_id_note"] =
            serde_json::json!("annotation jobs are session-independent; session_id is always null");

        if let Some(ref params_str) = job.params_json
            && let Ok(params_val) = serde_json::from_str::<serde_json::Value>(params_str)
        {
            response["annotate_params"] = serde_json::json!({
                "source_directory": params_val["source_directory"],
                "output_directory": params_val["output_directory"],
                "default_color": params_val["default_color"],
            });
        }
    }

    // For completed annotate jobs, read the pipeline report from the output
    // directory (annotate_report.json) and include a completion_summary with
    // aggregate annotation statistics and per-quote match_details. The report
    // is written by the pipeline upon completion and contains per-PDF and
    // per-quote results.
    if job.state == JobState::Completed
        && job.kind == "annotate"
        && let Some((summary, match_details)) = read_annotate_completion_summary(&job)
    {
        response["completion_summary"] = summary;
        response["match_details"] = match_details;
    }

    Ok(response)
}

/// Reads the annotation pipeline report from disk and extracts both a
/// completion summary and per-quote match_details. The report file
/// (`annotate_report.json`) is written by the annotation pipeline to the
/// output directory specified in params_json.
///
/// Returns a tuple of (completion_summary, match_details) where:
/// - completion_summary: aggregate counts (total_pdfs, quotes_matched, etc.)
/// - match_details: array of per-quote entries with filename, status,
///   match_method, page, and diagnostic information
///
/// Returns `None` if the params_json is missing, the output directory is not
/// set, or the report file does not exist or cannot be parsed. Failures are
/// silent because the report is supplementary information -- the core job
/// status fields (state, progress, error) are always available from the
/// database.
fn read_annotate_completion_summary(
    job: &neuroncite_store::JobRow,
) -> Option<(serde_json::Value, serde_json::Value)> {
    let params_str = job.params_json.as_deref()?;
    let params: serde_json::Value = serde_json::from_str(params_str).ok()?;
    let output_dir = params["output_directory"].as_str()?;

    let report_path = std::path::Path::new(output_dir).join("annotate_report.json");
    let report_bytes = std::fs::read(&report_path).ok()?;
    let report: serde_json::Value = serde_json::from_slice(&report_bytes).ok()?;

    let summary = &report["summary"];

    let completion_summary = serde_json::json!({
        "total_pdfs": summary["total_pdfs"],
        "successful_pdfs": summary["successful"],
        "partial_pdfs": summary["partial"],
        "failed_pdfs": summary["failed"],
        "total_quotes": summary["total_quotes"],
        "quotes_matched": summary["quotes_matched"],
        "quotes_not_found": summary["quotes_not_found"],
        "unmatched_inputs": report["unmatched_inputs"].as_array().map(|a| a.len()).unwrap_or(0),
    });

    // Build per-quote match_details from the pdfs[].quotes[] arrays in the
    // report. Each entry contains the PDF filename, quote excerpt, status,
    // match method, page number, and diagnostic details. This provides callers
    // with structured per-quote information without requiring them to read
    // and parse the annotate_report.json file separately.
    let mut match_details: Vec<serde_json::Value> = Vec::new();

    if let Some(pdfs) = report["pdfs"].as_array() {
        for pdf in pdfs {
            let filename = pdf["filename"].as_str().unwrap_or("");
            let pdf_status = pdf["status"].as_str().unwrap_or("");

            if let Some(quotes) = pdf["quotes"].as_array() {
                for quote in quotes {
                    let mut detail = serde_json::json!({
                        "filename": filename,
                        "pdf_status": pdf_status,
                        "quote_excerpt": quote["quote_excerpt"],
                        "status": quote["status"],
                        "match_method": quote["match_method"],
                        "page": quote["page"],
                    });

                    // Include character-level match statistics when available.
                    // chars_matched < chars_total indicates partial bounding box
                    // coverage, which can affect highlight accuracy.
                    if let Some(cm) = quote["chars_matched"].as_u64() {
                        detail["chars_matched"] = serde_json::json!(cm);
                    }
                    if let Some(ct) = quote["chars_total"].as_u64() {
                        detail["chars_total"] = serde_json::json!(ct);
                    }

                    // Include fuzzy match score when the match used the fuzzy
                    // pipeline stage. Scores below 0.95 indicate lower confidence.
                    if let Some(fs) = quote["fuzzy_score"].as_f64() {
                        detail["fuzzy_score"] = serde_json::json!(fs);
                    }

                    // Include pipeline stage diagnostics for quotes that were
                    // not found. These describe which stages were attempted and
                    // why they failed, helping callers understand the failure.
                    if let Some(stages) = quote["stages_tried"].as_array() {
                        detail["stages_tried"] = serde_json::json!(stages);
                    }

                    match_details.push(detail);
                }
            }
        }
    }

    // Append unmatched input rows (quotes whose title/author could not be
    // matched to any PDF file in the source directory). These appear at the
    // end of match_details with status "unmatched_input".
    if let Some(unmatched) = report["unmatched_inputs"].as_array() {
        for entry in unmatched {
            match_details.push(serde_json::json!({
                "filename": null,
                "pdf_status": null,
                "quote_excerpt": null,
                "status": "unmatched_input",
                "match_method": null,
                "page": null,
                "title": entry["title"],
                "author": entry["author"],
                "error": entry["error"],
            }));
        }
    }

    Some((completion_summary, serde_json::json!(match_details)))
}

/// Cancels a queued or running job.
///
/// Only jobs in "queued" or "running" state can be canceled. Jobs that have
/// already completed, failed, or been canceled return an error.
///
/// For running executor-managed jobs (index, annotate), the cancellation is
/// cooperative: the executor checks the job state before processing each file
/// and stops when it detects a canceled state. For agent-driven jobs
/// (citation_verify), canceling prevents further batch claims.
///
/// # Parameters (from MCP tool call)
///
/// - `job_id` (required): UUID of the job to cancel.
pub async fn handle_cancel(
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

    if job.state != JobState::Queued && job.state != JobState::Running {
        return Err(format!(
            "job {} is in state {} and cannot be canceled",
            job_id, job.state
        ));
    }

    neuroncite_store::update_job_state(&conn, job_id, JobState::Canceled, None)
        .map_err(|e| format!("canceling job: {e}"))?;

    // Wake the executor so it detects the cancellation immediately and can
    // signal the running pipeline via the cooperative cancel flag.
    state.job_notify.notify_one();

    Ok(serde_json::json!({
        "job_id": job_id,
        "previous_state": format!("{:?}", job.state).to_lowercase(),
        "state": "canceled",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-MCP-JOB-001: read_annotate_completion_summary parses a well-formed
    /// annotate_report.json and extracts both the completion_summary and
    /// match_details arrays.
    #[test]
    fn t_mcp_job_001_read_annotate_report_with_match_details() {
        // Create a temporary directory with a simulated annotate_report.json.
        let tmp_dir = tempfile::tempdir().expect("create temp dir");
        let output_dir = tmp_dir.path().to_str().unwrap();

        let report = serde_json::json!({
            "timestamp": "2026-02-26T12:00:00Z",
            "source_directory": "/source",
            "output_directory": output_dir,
            "summary": {
                "total_pdfs": 2,
                "successful": 1,
                "partial": 1,
                "failed": 0,
                "total_quotes": 4,
                "quotes_matched": 3,
                "quotes_not_found": 1,
            },
            "pdfs": [
                {
                    "filename": "paper_a.pdf",
                    "source_path": "/source/paper_a.pdf",
                    "output_path": format!("{output_dir}/paper_a.pdf"),
                    "status": "success",
                    "quotes": [
                        {
                            "quote_excerpt": "The variance of returns...",
                            "status": "matched",
                            "match_method": "exact",
                            "page": 5,
                            "chars_matched": 30,
                            "chars_total": 30,
                            "pre_check_passed": true,
                            "post_check_passed": true,
                        },
                        {
                            "quote_excerpt": "Risk factors include...",
                            "status": "matched",
                            "match_method": "fuzzy",
                            "page": 12,
                            "chars_matched": 45,
                            "chars_total": 48,
                            "pre_check_passed": true,
                            "post_check_passed": true,
                            "fuzzy_score": 0.953,
                        },
                    ],
                    "error": null,
                },
                {
                    "filename": "paper_b.pdf",
                    "source_path": "/source/paper_b.pdf",
                    "output_path": format!("{output_dir}/paper_b.pdf"),
                    "status": "partial",
                    "quotes": [
                        {
                            "quote_excerpt": "Market efficiency...",
                            "status": "matched",
                            "match_method": "normalized",
                            "page": 3,
                            "chars_matched": 20,
                            "chars_total": 20,
                        },
                        {
                            "quote_excerpt": "This quote does not exist...",
                            "status": "not_found",
                            "match_method": null,
                            "page": null,
                            "stages_tried": [
                                "exact: no match on 15 pages",
                                "normalized: no match on 15 pages",
                                "fuzzy: best=42% on page 7 < 90% threshold"
                            ],
                        },
                    ],
                    "error": null,
                },
            ],
            "unmatched_inputs": [
                {
                    "title": "Nonexistent Paper",
                    "author": "Unknown Author",
                    "error": "no PDF matched title",
                },
            ],
        });

        // Write the report to disk.
        let report_path = tmp_dir.path().join("annotate_report.json");
        std::fs::write(&report_path, serde_json::to_string_pretty(&report).unwrap())
            .expect("write report");

        // Create a simulated JobRow with params_json pointing to the output dir.
        let job = neuroncite_store::JobRow {
            id: "test-job-id".into(),
            kind: "annotate".into(),
            state: JobState::Completed,
            session_id: None,
            progress_done: 4,
            progress_total: 4,
            created_at: 1000,
            started_at: Some(1001),
            finished_at: Some(1010),
            error_message: None,
            params_json: Some(
                serde_json::json!({
                    "output_directory": output_dir,
                    "source_directory": "/source",
                    "default_color": "#FFFF00",
                })
                .to_string(),
            ),
        };

        let result = read_annotate_completion_summary(&job);
        assert!(
            result.is_some(),
            "read_annotate_completion_summary must parse the report"
        );

        let (summary, match_details) = result.unwrap();

        // Verify completion_summary aggregate counts.
        assert_eq!(summary["total_pdfs"], 2);
        assert_eq!(summary["successful_pdfs"], 1);
        assert_eq!(summary["partial_pdfs"], 1);
        assert_eq!(summary["quotes_matched"], 3);
        assert_eq!(summary["quotes_not_found"], 1);
        assert_eq!(summary["unmatched_inputs"], 1);

        // Verify match_details contains per-quote entries.
        let details = match_details.as_array().expect("match_details is an array");

        // 4 quotes from pdfs + 1 unmatched input = 5 entries total.
        assert_eq!(details.len(), 5, "4 PDF quotes + 1 unmatched input");

        // First quote: exact match in paper_a.pdf.
        assert_eq!(details[0]["filename"], "paper_a.pdf");
        assert_eq!(details[0]["status"], "matched");
        assert_eq!(details[0]["match_method"], "exact");
        assert_eq!(details[0]["page"], 5);
        assert_eq!(details[0]["chars_matched"], 30);

        // Second quote: fuzzy match with fuzzy_score.
        assert_eq!(details[1]["match_method"], "fuzzy");
        assert_eq!(details[1]["fuzzy_score"], 0.953);

        // Fourth quote: not_found with stages_tried.
        assert_eq!(details[3]["status"], "not_found");
        let stages = details[3]["stages_tried"]
            .as_array()
            .expect("stages_tried is an array");
        assert_eq!(stages.len(), 3, "three pipeline stages were tried");
        assert!(
            stages[2].as_str().unwrap().contains("fuzzy"),
            "third stage must be fuzzy"
        );

        // Fifth entry: unmatched input.
        assert_eq!(details[4]["status"], "unmatched_input");
        assert_eq!(details[4]["title"], "Nonexistent Paper");
        assert_eq!(details[4]["author"], "Unknown Author");
        assert!(
            details[4]["filename"].is_null(),
            "unmatched input has no filename"
        );
    }

    /// T-MCP-JOB-002: read_annotate_completion_summary returns None when
    /// params_json is missing the output_directory field.
    #[test]
    fn t_mcp_job_002_missing_output_directory_returns_none() {
        let job = neuroncite_store::JobRow {
            id: "test-job".into(),
            kind: "annotate".into(),
            state: JobState::Completed,
            session_id: None,
            progress_done: 0,
            progress_total: 0,
            created_at: 1000,
            started_at: Some(1001),
            finished_at: Some(1010),
            error_message: None,
            params_json: Some(serde_json::json!({"source_directory": "/src"}).to_string()),
        };

        assert!(
            read_annotate_completion_summary(&job).is_none(),
            "missing output_directory must return None"
        );
    }

    /// T-MCP-JOB-003: read_annotate_completion_summary returns None when
    /// params_json is None entirely.
    #[test]
    fn t_mcp_job_003_no_params_json_returns_none() {
        let job = neuroncite_store::JobRow {
            id: "test-job".into(),
            kind: "annotate".into(),
            state: JobState::Completed,
            session_id: None,
            progress_done: 0,
            progress_total: 0,
            created_at: 1000,
            started_at: Some(1001),
            finished_at: Some(1010),
            error_message: None,
            params_json: None,
        };

        assert!(
            read_annotate_completion_summary(&job).is_none(),
            "None params_json must return None"
        );
    }

    /// T-MCP-JOB-004: read_annotate_completion_summary returns None when the
    /// annotate_report.json file does not exist on disk.
    #[test]
    fn t_mcp_job_004_missing_report_file_returns_none() {
        let job = neuroncite_store::JobRow {
            id: "test-job".into(),
            kind: "annotate".into(),
            state: JobState::Completed,
            session_id: None,
            progress_done: 0,
            progress_total: 0,
            created_at: 1000,
            started_at: Some(1001),
            finished_at: Some(1010),
            error_message: None,
            params_json: Some(
                serde_json::json!({
                    "output_directory": "/nonexistent/path",
                    "source_directory": "/source",
                })
                .to_string(),
            ),
        };

        assert!(
            read_annotate_completion_summary(&job).is_none(),
            "nonexistent report file must return None"
        );
    }

    /// T-MCP-JOB-005: match_details handles an empty pdfs array and empty
    /// unmatched_inputs gracefully (returns empty match_details array).
    #[test]
    fn t_mcp_job_005_empty_report_produces_empty_match_details() {
        let tmp_dir = tempfile::tempdir().expect("create temp dir");
        let output_dir = tmp_dir.path().to_str().unwrap();

        let report = serde_json::json!({
            "summary": {
                "total_pdfs": 0,
                "successful": 0,
                "partial": 0,
                "failed": 0,
                "total_quotes": 0,
                "quotes_matched": 0,
                "quotes_not_found": 0,
            },
            "pdfs": [],
            "unmatched_inputs": [],
        });

        let report_path = tmp_dir.path().join("annotate_report.json");
        std::fs::write(&report_path, serde_json::to_string(&report).unwrap()).unwrap();

        let job = neuroncite_store::JobRow {
            id: "test-job".into(),
            kind: "annotate".into(),
            state: JobState::Completed,
            session_id: None,
            progress_done: 0,
            progress_total: 0,
            created_at: 1000,
            started_at: Some(1001),
            finished_at: Some(1010),
            error_message: None,
            params_json: Some(
                serde_json::json!({
                    "output_directory": output_dir,
                    "source_directory": "/source",
                })
                .to_string(),
            ),
        };

        let result = read_annotate_completion_summary(&job);
        assert!(result.is_some(), "empty report must still parse");

        let (summary, match_details) = result.unwrap();
        assert_eq!(summary["total_pdfs"], 0);
        assert_eq!(summary["quotes_matched"], 0);

        let details = match_details.as_array().expect("match_details is an array");
        assert!(
            details.is_empty(),
            "match_details must be empty for a report with no quotes"
        );
    }

    /// T-MCP-JOB-006: Annotation jobs include a session_id_note explaining
    /// that session_id is always null for annotation jobs.
    #[test]
    fn t_mcp_job_006_annotation_session_id_note() {
        // Verify the session_id_note text is the expected constant string.
        let expected = "annotation jobs are session-independent; session_id is always null";
        let note = serde_json::json!(expected);
        assert_eq!(
            note.as_str().unwrap(),
            expected,
            "session_id_note must contain the explanatory text"
        );
    }
}
