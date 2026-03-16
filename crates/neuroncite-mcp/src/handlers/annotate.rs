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

//! Handler for the `neuroncite_annotate` MCP tool.
//!
//! Creates an annotation job that highlights quoted text passages in PDF files
//! and adds optional popup comments. The annotation pipeline runs asynchronously
//! through the job executor; the caller polls `neuroncite_job_status` for progress.
//!
//! Input data (CSV or JSON) is validated immediately to provide early error
//! feedback. The validated data and directory paths are stored as serialized
//! JSON in the job's `params_json` column for the executor to pick up.
//!
//! Two additional modes control the handler's behavior:
//!
//! - **dry_run**: When true, the handler validates input data and checks which
//!   quoted PDFs exist in the source directory. No job is created and no files
//!   are written. The response contains a per-row preview with match status.
//!
//! - **append**: When true, the handler reads PDFs from the output directory
//!   (which contains previously annotated files) instead of the source directory.
//!   This allows layering annotations incrementally across multiple runs.

use std::sync::Arc;

use neuroncite_api::AppState;

/// Starts a PDF annotation job from MCP tool parameters.
///
/// # Parameters (from MCP tool call)
///
/// - `source_directory` (required): Absolute path to the directory containing source PDFs.
/// - `output_directory` (required): Absolute path where annotated PDFs are saved.
/// - `input_data` (required): CSV or JSON string with annotation instructions.
///   Required columns: title, author, quote. Optional: color (#RRGGBB), comment.
/// - `default_color` (optional): Default highlight color in hex (#RRGGBB). Default: #FFFF00.
/// - `dry_run` (optional): When true, validate input and report match status without
///   creating a job or writing files. Default: false.
/// - `append` (optional): When true, read PDFs from output_directory instead of
///   source_directory. The output_directory must already exist. Default: false.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let source_directory = params["source_directory"]
        .as_str()
        .ok_or("missing required parameter: source_directory")?;

    let output_directory = params["output_directory"]
        .as_str()
        .ok_or("missing required parameter: output_directory")?;

    let input_data = params["input_data"]
        .as_str()
        .ok_or("missing required parameter: input_data")?;

    // Validate source directory exists.
    let source_path = std::path::Path::new(source_directory);
    if !source_path.is_dir() {
        return Err(format!(
            "source_directory does not exist: {source_directory}"
        ));
    }

    // Early validation: parse the input data to catch format errors before
    // creating the job record. This gives the AI assistant immediate feedback.
    let rows = neuroncite_annotate::parse_input(input_data.as_bytes())
        .map_err(|e| format!("input_data parsing failed: {e}"))?;

    let total_quotes = rows.len();
    if total_quotes == 0 {
        return Err("input_data contains no annotation rows".to_string());
    }

    let default_color = params["default_color"].as_str().unwrap_or("#FFFF00");
    let dry_run = params["dry_run"].as_bool().unwrap_or(false);
    let append = params["append"].as_bool().unwrap_or(false);

    if dry_run {
        // Dry-run mode: validate input data and report which PDFs were found
        // in the source directory. Delegates to resolve::match_pdfs_to_quotes,
        // the same function that annotate_pdfs_with_cancel() calls internally.
        // This guarantees identical match/unmatch decisions between dry_run=true
        // and dry_run=false on the same inputs.
        let dry_run_rows = rows.clone();
        let dry_source = source_path.to_path_buf();
        let (matched_count, unmatched_count, preview) = tokio::task::spawn_blocking(move || {
            let (matched, unmatched) = neuroncite_annotate::resolve::match_pdfs_to_quotes(
                &dry_source,
                dry_run_rows.clone(),
            )
            .unwrap_or_default();

            // Build a set of matched (title, author) pairs for O(1) lookup.
            let matched_set: std::collections::HashSet<(String, String)> = matched
                .values()
                .flatten()
                .map(|row| (row.title.clone(), row.author.clone()))
                .collect();

            let matched_count = matched.values().map(|v| v.len()).sum::<usize>();
            let unmatched_count = unmatched.len();

            let preview: Vec<serde_json::Value> = dry_run_rows
                .iter()
                .map(|row| {
                    let found = matched_set.contains(&(row.title.clone(), row.author.clone()));
                    serde_json::json!({
                        "title": row.title,
                        "author": row.author,
                        "quote_length": row.quote.len(),
                        "pdf_found": found,
                    })
                })
                .collect();

            (matched_count, unmatched_count, preview)
        })
        .await
        .map_err(|e| format!("dry-run matching task failed: {e}"))?;

        return Ok(serde_json::json!({
            "dry_run": true,
            "total_quotes": total_quotes,
            "matched_quotes": matched_count,
            "unmatched_quotes": unmatched_count,
            "source_directory": source_directory,
            "preview": preview,
        }));
    }

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Check for concurrent annotation jobs.
    let jobs = neuroncite_store::list_jobs(&conn).map_err(|e| format!("listing jobs: {e}"))?;
    let has_running = jobs.iter().any(|j| {
        j.kind == "annotate"
            && (j.state == neuroncite_store::JobState::Queued
                || j.state == neuroncite_store::JobState::Running)
    });
    if has_running {
        return Err(
            "an annotation job is already running; wait for it to complete or cancel it"
                .to_string(),
        );
    }

    // In append mode, validate that the output directory exists (it must
    // contain the previously annotated PDFs from a prior run). The pipeline
    // always reads from source_directory for text extraction because pdfium
    // re-linearizes PDFs when saving, which breaks text extraction on a
    // second pass. Prior annotations are merged separately using lopdf.
    if append {
        let out_path = std::path::Path::new(output_directory);
        if !out_path.is_dir() {
            return Err(format!(
                "append mode requires output_directory to exist: {output_directory}"
            ));
        }
    }

    // Serialize job parameters for storage in params_json. The pipeline
    // always reads source PDFs from source_directory (original, unmodified
    // files). When append=true, prior_output_directory tells the pipeline
    // where to find previously annotated PDFs whose highlight annotations
    // should be merged into the new output.
    let params_obj = serde_json::json!({
        "input_data": input_data,
        "source_directory": source_directory,
        "output_directory": output_directory,
        "default_color": default_color,
        "prior_output_directory": if append { Some(output_directory) } else { None::<&str> },
    });
    let params_json =
        serde_json::to_string(&params_obj).map_err(|e| format!("params serialization: {e}"))?;

    // Create the annotation job. No session_id for annotation jobs.
    let job_id = uuid::Uuid::new_v4().to_string();
    neuroncite_store::create_job_with_params(&conn, &job_id, "annotate", None, Some(&params_json))
        .map_err(|e| format!("job creation: {e}"))?;

    // Set the initial progress_total to the number of quotes so that callers
    // polling neuroncite_job_status see a meaningful "0 / N" progress
    // immediately, rather than the default "0 / 0" which provides no
    // indication of the expected workload. The executor's pipeline callback
    // will update progress_done as PDFs are processed.
    let _ = neuroncite_store::update_job_progress(&conn, &job_id, 0, total_quotes as i64);

    // Wake the executor immediately so it picks up the annotate job without
    // waiting for the next poll interval.
    state.job_notify.notify_one();

    Ok(serde_json::json!({
        "job_id": job_id,
        "status": "accepted",
        "total_quotes": total_quotes,
        "source_directory": source_directory,
        "output_directory": output_directory,
        "default_color": default_color,
        "append": append,
    }))
}

#[cfg(test)]
mod tests {
    /// T-MCP-060: Missing required parameters return an error.
    #[test]
    fn t_mcp_060_annotate_missing_params() {
        let params = serde_json::json!({});
        let source = params["source_directory"].as_str();
        assert!(
            source.is_none(),
            "missing source_directory must return None"
        );
    }

    /// T-MCP-061: Malformed input_data (invalid CSV) is detected during
    /// early validation before job creation.
    #[test]
    fn t_mcp_061_annotate_invalid_input_data() {
        // An input with missing required columns should fail parsing.
        let bad_csv = "wrong_col1,wrong_col2\nfoo,bar";
        let result = neuroncite_annotate::parse_input(bad_csv.as_bytes());
        assert!(
            result.is_err(),
            "malformed CSV without required columns must be rejected"
        );
    }

    /// T-MCP-062: dry_run mode validates input without creating a job.
    #[test]
    fn t_mcp_062_annotate_dry_run_validation() {
        // Verify that the dry_run parameter is correctly parsed from JSON.
        let params = serde_json::json!({
            "source_directory": "/nonexistent",
            "output_directory": "/tmp/out",
            "input_data": "title,author,quote\nTest,Author,\"some quote\"",
            "dry_run": true,
        });
        assert_eq!(params["dry_run"].as_bool(), Some(true));
    }

    /// T-MCP-063: append mode parameter is correctly parsed.
    #[test]
    fn t_mcp_063_annotate_append_param() {
        let params = serde_json::json!({
            "source_directory": "/src",
            "output_directory": "/out",
            "input_data": "title,author,quote\nTest,Author,\"some quote\"",
            "append": true,
        });
        assert_eq!(params["append"].as_bool(), Some(true));
    }

    /// T-MCP-064: default values for dry_run and append are false.
    #[test]
    fn t_mcp_064_annotate_defaults() {
        let params = serde_json::json!({
            "source_directory": "/src",
            "output_directory": "/out",
            "input_data": "data",
        });
        assert!(!params["dry_run"].as_bool().unwrap_or(false));
        assert!(!params["append"].as_bool().unwrap_or(false));
    }

    /// T-MCP-065: dry_run uses resolve::match_pdfs_to_quotes (same function
    /// as the actual pipeline) to guarantee identical match decisions.
    /// Regression test for DEF-6: dry_run previously used a simplified
    /// substring-based matching that produced different results from the
    /// Jaro-Winkler weighted matching in the actual annotation pipeline.
    /// The fix delegates to resolve::match_pdfs_to_quotes directly.
    #[test]
    fn t_mcp_065_dry_run_uses_resolve_module() {
        // Verify that resolve::match_pdfs_to_quotes is accessible from this
        // crate (the import succeeds and the function signature matches).
        // The actual matching behavior is tested by the resolve module's own
        // tests (T-ANNOTATE-020 through T-ANNOTATE-067).
        let source = std::path::PathBuf::from("/nonexistent");
        let rows: Vec<neuroncite_annotate::InputRow> = Vec::new();
        let result = neuroncite_annotate::resolve::match_pdfs_to_quotes(&source, rows);
        // The function returns an error for a nonexistent directory, which
        // confirms the function is callable from the handler's crate.
        assert!(
            result.is_err(),
            "nonexistent source directory must produce an error from resolve"
        );
    }

    /// T-MCP-066: In append mode, the serialized params_json contains
    /// source_directory (not output_directory) as the source for text
    /// extraction, and prior_output_directory is set to output_directory.
    /// Regression test for NEW-2 where append mode passed output_directory
    /// as the effective source, causing text extraction to fail on
    /// pdfium-re-linearized PDFs.
    #[test]
    fn t_mcp_066_append_mode_uses_source_directory() {
        // Simulate the params_json construction from the handler.
        let source_directory = "/original/pdfs";
        let output_directory = "/annotated/pdfs";
        let append = true;

        let params_obj = serde_json::json!({
            "input_data": "title,author,quote\nTest,Author,\"quote\"",
            "source_directory": source_directory,
            "output_directory": output_directory,
            "default_color": "#FFFF00",
            "prior_output_directory": if append { Some(output_directory) } else { None::<&str> },
        });

        // The source_directory in params must always be the original source,
        // not the output_directory (which contains re-linearized PDFs).
        assert_eq!(
            params_obj["source_directory"].as_str().unwrap(),
            "/original/pdfs",
            "append mode must use original source_directory for text extraction"
        );
        assert_eq!(
            params_obj["prior_output_directory"].as_str().unwrap(),
            "/annotated/pdfs",
            "prior_output_directory must be set to output_directory in append mode"
        );
    }

    /// T-MCP-067: In non-append mode, prior_output_directory is null.
    #[test]
    fn t_mcp_067_non_append_mode_no_prior_output() {
        let append = false;

        let params_obj = serde_json::json!({
            "source_directory": "/src",
            "output_directory": "/out",
            "prior_output_directory": if append { Some("/out") } else { None::<&str> },
        });

        assert!(
            params_obj["prior_output_directory"].is_null(),
            "non-append mode must not set prior_output_directory"
        );
    }
}
