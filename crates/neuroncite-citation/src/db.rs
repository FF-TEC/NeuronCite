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

// SQLite CRUD operations for citation_row records.
//
// Provides functions for the complete citation verification lifecycle:
// inserting rows after parsing, claiming batches for sub-agent processing,
// submitting verification results, and querying status/verdict aggregates.
// All functions accept a borrowed rusqlite::Connection and operate within
// the caller's transaction context.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, params};

use crate::error::CitationError;
use crate::types::{
    Alert, BatchClaim, BatchCounts, CitationJobParams, CitationRow, CitationRowInsert,
    StatusCounts, SubmitEntry, Verdict,
};

/// Stale batch timeout in seconds. Claimed batches that have not been
/// submitted within this window are reclaimed and reset to pending.
const STALE_TIMEOUT_SECS: i64 = 300;

/// Inserts citation rows into the database. Each row corresponds to one
/// cite-key from the parsed LaTeX document. Includes the co_citation_json
/// column for co-citation metadata computed during parsing.
///
/// # Returns
///
/// The number of rows inserted.
pub fn insert_citation_rows(
    conn: &Connection,
    rows: &[CitationRowInsert],
) -> Result<usize, CitationError> {
    let mut stmt = conn.prepare_cached(
        "INSERT INTO citation_row (
            job_id, group_id, cite_key, author, title, year, tex_line,
            anchor_before, anchor_after, section_title, matched_file_id,
            bib_abstract, bib_keywords, tex_context, batch_id, status, co_citation_json
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, 'pending', ?16)",
    )?;

    for row in rows {
        stmt.execute(params![
            row.job_id,
            row.group_id,
            row.cite_key,
            row.author,
            row.title,
            row.year,
            row.tex_line,
            row.anchor_before,
            row.anchor_after,
            row.section_title,
            row.matched_file_id,
            row.bib_abstract,
            row.bib_keywords,
            row.tex_context,
            row.batch_id,
            row.co_citation_json,
        ])?;
    }

    Ok(rows.len())
}

/// Claims the next available batch for processing. Uses BEGIN IMMEDIATE to
/// acquire a write lock before reading, ensuring correct behavior.
///
/// Steps within the transaction:
/// 1. Reclaims stale batches: rows with `status = 'claimed'` and
///    `claimed_at < now - STALE_TIMEOUT_SECS` are reset to `pending`.
/// 2. Finds the lowest `batch_id` where ALL rows have `status = 'pending'`.
/// 3. Updates all rows in that batch to `status = 'claimed'` with
///    `claimed_at = now`.
/// 4. Reads back the claimed rows and job parameters.
///
/// Returns None if no pending batches remain.
pub fn claim_next_batch(
    conn: &Connection,
    job_id: &str,
) -> Result<Option<BatchClaim>, CitationError> {
    // BEGIN IMMEDIATE acquires a reserved lock before any reads, serializing
    // concurrent claim calls from parallel worker loops. Without this, two
    // workers could SELECT the same pending batch_id before either UPDATEs.
    // Uses raw SQL because rusqlite's transaction_with_behavior() requires
    // &mut Connection, which is not available from the r2d2 pool.
    conn.execute_batch("BEGIN IMMEDIATE")?;
    match claim_next_batch_inner(conn, job_id) {
        Ok(result) => {
            conn.execute_batch("COMMIT")?;
            Ok(result)
        }
        Err(e) => {
            // Roll back on any error to release the write lock and undo
            // partial state changes (e.g. stale reclaims without a claim).
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

/// Inner implementation of claim_next_batch, called within a BEGIN IMMEDIATE
/// transaction. Separated to allow the outer function to handle COMMIT/ROLLBACK
/// around all possible error paths.
fn claim_next_batch_inner(
    conn: &Connection,
    job_id: &str,
) -> Result<Option<BatchClaim>, CitationError> {
    let now = unix_now();
    let stale_cutoff = now - STALE_TIMEOUT_SECS;

    // Step 1: Reclaim stale batches.
    conn.execute(
        "UPDATE citation_row SET status = 'pending', claimed_at = NULL
         WHERE job_id = ?1 AND status = 'claimed' AND claimed_at < ?2",
        params![job_id, stale_cutoff],
    )?;

    // Step 2: Find the lowest pending batch_id. A batch is pending only if
    // ALL its rows are in 'pending' status (no partial claims).
    let batch_id: Option<i64> = conn
        .query_row(
            "SELECT batch_id FROM citation_row
             WHERE job_id = ?1 AND status = 'pending' AND batch_id IS NOT NULL
             GROUP BY batch_id
             HAVING COUNT(*) = SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END)
             ORDER BY batch_id ASC
             LIMIT 1",
            params![job_id],
            |row| row.get(0),
        )
        .ok(); // Returns None if no rows match.

    let batch_id = match batch_id {
        Some(id) => id,
        None => return Ok(None),
    };

    // Step 3: Claim all rows in this batch.
    conn.execute(
        "UPDATE citation_row SET status = 'claimed', claimed_at = ?1
         WHERE job_id = ?2 AND batch_id = ?3 AND status = 'pending'",
        params![now, job_id, batch_id],
    )?;

    // Step 4: Read back all rows in the claimed batch.
    let rows = list_rows_by_batch(conn, job_id, batch_id)?;

    // Read job params (tex_path, bib_path, session_id) from the job record.
    let params_json: String = conn
        .query_row(
            "SELECT params_json FROM job WHERE id = ?1",
            params![job_id],
            |row| row.get(0),
        )
        .map_err(|_| CitationError::JobNotFound {
            job_id: job_id.to_string(),
        })?;

    let job_params: CitationJobParams = serde_json::from_str(&params_json)?;

    Ok(Some(BatchClaim {
        batch_id,
        tex_path: job_params.tex_path,
        bib_path: job_params.bib_path,
        session_id: job_params.session_id,
        rows,
    }))
}

/// Validates a SubmitEntry for business rule compliance before storage.
/// Returns an error message string if validation fails, None if the entry
/// is valid.
///
/// Validation rules:
/// - flag must be "critical", "warning", or empty string.
/// - verdict=wrong_source, partial, peripheral_match require other_source_list
///   to be non-empty.
/// - Each verdict has a valid meta-confidence [floor, ceiling] range:
///   supported [0.50, 1.00], partial [0.00, 0.80], unsupported [0.50, 1.00],
///   not_found [0.70, 1.00], wrong_source [0.50, 1.00], unverifiable [0.50, 1.00],
///   peripheral_match [0.00, 0.69].
fn validate_submit_entry(entry: &SubmitEntry) -> Option<String> {
    // Validate flag values.
    if !entry.flag.is_empty() && entry.flag != "critical" && entry.flag != "warning" {
        return Some(format!(
            "row {}: flag must be 'critical', 'warning', or empty, got '{}'",
            entry.row_id, entry.flag
        ));
    }

    // Validate that wrong_source, partial, and peripheral_match verdicts have
    // at least one entry in other_source_list. For peripheral_match this
    // enforces a cross-corpus search for body-text evidence in other sources.
    if matches!(
        entry.verdict,
        Verdict::WrongSource | Verdict::Partial | Verdict::PeripheralMatch
    ) && entry.other_source_list.is_empty()
    {
        let label = match entry.verdict {
            Verdict::WrongSource => "wrong_source",
            Verdict::Partial => "partial",
            Verdict::PeripheralMatch => "peripheral_match",
            _ => unreachable!(),
        };
        return Some(format!(
            "row {}: verdict '{}' requires at least one entry in other_source_list",
            entry.row_id, label
        ));
    }

    // Validate verdict-specific meta-confidence bounds. Each verdict has a
    // valid [floor, ceiling] range. Violations indicate the submitting agent
    // used an inconsistent confidence interpretation or bypassed the
    // enforce_confidence_caps safety net (MCP external agents).
    let bounds_err = match entry.verdict {
        Verdict::Supported if entry.confidence < 0.50 => Some(("supported", ">= 0.50")),
        Verdict::Partial if entry.confidence > 0.80 => Some(("partial", "<= 0.80")),
        Verdict::Unsupported if entry.confidence < 0.50 => Some(("unsupported", ">= 0.50")),
        Verdict::NotFound if entry.confidence < 0.70 => Some(("not_found", ">= 0.70")),
        Verdict::WrongSource if entry.confidence < 0.50 => Some(("wrong_source", ">= 0.50")),
        Verdict::Unverifiable if entry.confidence < 0.50 => Some(("unverifiable", ">= 0.50")),
        Verdict::PeripheralMatch if entry.confidence > 0.69 => {
            Some(("peripheral_match", "<= 0.69"))
        }
        _ => None,
    };
    if let Some((label, requirement)) = bounds_err {
        return Some(format!(
            "row {}: verdict '{}' requires confidence {}, got {:.2}",
            entry.row_id, label, requirement, entry.confidence
        ));
    }

    None
}

/// Submits verification results for a batch of citation rows. For each
/// entry, validates that the row belongs to the specified job and is in
/// 'claimed' status, runs business rule validation, then serializes the
/// result into result_json, sets status to 'done', and stores the flag.
///
/// After submission, updates the job's progress_done counter to reflect
/// the total number of done + failed rows.
///
/// # Returns
///
/// The number of rows updated.
pub fn submit_batch_results(
    conn: &Connection,
    job_id: &str,
    results: &[SubmitEntry],
) -> Result<usize, CitationError> {
    let mut update_stmt = conn.prepare_cached(
        "UPDATE citation_row SET status = 'done', result_json = ?1, flag = ?2
         WHERE id = ?3 AND job_id = ?4 AND status = 'claimed'",
    )?;

    let mut count = 0;

    for entry in results {
        // Validate business rules before processing.
        if let Some(error_msg) = validate_submit_entry(entry) {
            return Err(CitationError::InvalidRowState {
                row_id: entry.row_id,
                current_status: error_msg,
                expected_status: "valid submit entry".to_string(),
            });
        }

        // Verify the row belongs to this job and is claimed.
        let (row_job_id, row_status): (String, String) = conn
            .query_row(
                "SELECT job_id, status FROM citation_row WHERE id = ?1",
                params![entry.row_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|_| CitationError::InvalidRowState {
                row_id: entry.row_id,
                current_status: "not_found".to_string(),
                expected_status: "claimed".to_string(),
            })?;

        if row_job_id != job_id {
            return Err(CitationError::RowJobMismatch {
                row_id: entry.row_id,
                job_id: job_id.to_string(),
            });
        }

        if row_status != "claimed" {
            // When the row is already 'done', a second agent processed it after
            // the batch was reclaimed (stale batch timeout >5 min). The first
            // agent's result is valid. Skip this submission silently to avoid
            // overwriting a correct result with a 'failed' marker.
            if row_status == "done" {
                tracing::warn!(
                    row_id = entry.row_id,
                    "Skipping duplicate submission: row is already done. \
                     The batch was likely reclaimed after a timeout and two \
                     agents completed it concurrently."
                );
                continue;
            }
            return Err(CitationError::InvalidRowState {
                row_id: entry.row_id,
                current_status: row_status,
                expected_status: "claimed".to_string(),
            });
        }

        let result_json = serde_json::to_string(entry)?;

        // Store flag as NULL if empty string, so existing flag queries
        // using IS NOT NULL continue to work correctly.
        let flag_value: Option<&str> = if entry.flag.is_empty() {
            None
        } else {
            Some(&entry.flag)
        };

        update_stmt.execute(params![result_json, flag_value, entry.row_id, job_id])?;
        count += 1;
    }

    // Update the job's progress_done to the total count of done + failed rows.
    let done_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM citation_row WHERE job_id = ?1 AND status IN ('done', 'failed')",
        params![job_id],
        |row| row.get(0),
    )?;

    let total_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM citation_row WHERE job_id = ?1",
        params![job_id],
        |row| row.get(0),
    )?;

    conn.execute(
        "UPDATE job SET progress_done = ?1, progress_total = ?2 WHERE id = ?3",
        params![done_count, total_count, job_id],
    )?;

    Ok(count)
}

/// Returns the count of citation rows grouped by status for the given job.
pub fn count_by_status(conn: &Connection, job_id: &str) -> Result<StatusCounts, CitationError> {
    let mut counts = StatusCounts::default();

    let mut stmt = conn.prepare_cached(
        "SELECT status, COUNT(*) FROM citation_row WHERE job_id = ?1 GROUP BY status",
    )?;

    let rows = stmt.query_map(params![job_id], |row| {
        let status: String = row.get(0)?;
        let count: i64 = row.get(1)?;
        Ok((status, count))
    })?;

    for row in rows {
        let (status, count) = row?;
        match status.as_str() {
            "pending" => counts.pending = count,
            "claimed" => counts.claimed = count,
            "done" => counts.done = count,
            "failed" => counts.failed = count,
            _ => {} // Ignore unknown statuses.
        }
    }

    Ok(counts)
}

/// Returns the count of batches grouped by their aggregate status.
/// A batch is "done" if all its rows are done or failed, "pending" if all
/// are pending, and "claimed" if at least one row is claimed.
pub fn count_batches_by_status(
    conn: &Connection,
    job_id: &str,
) -> Result<BatchCounts, CitationError> {
    // Total distinct batches.
    let total = conn.query_row(
        "SELECT COUNT(DISTINCT batch_id) FROM citation_row WHERE job_id = ?1 AND batch_id IS NOT NULL",
        params![job_id],
        |row| row.get(0),
    )?;

    // Batches where all rows are done or failed.
    let done = conn.query_row(
        "SELECT COUNT(*) FROM (
            SELECT batch_id FROM citation_row
            WHERE job_id = ?1 AND batch_id IS NOT NULL
            GROUP BY batch_id
            HAVING SUM(CASE WHEN status IN ('done', 'failed') THEN 1 ELSE 0 END) = COUNT(*)
        )",
        params![job_id],
        |row| row.get(0),
    )?;

    // Batches where all rows are pending.
    let pending = conn.query_row(
        "SELECT COUNT(*) FROM (
            SELECT batch_id FROM citation_row
            WHERE job_id = ?1 AND batch_id IS NOT NULL
            GROUP BY batch_id
            HAVING SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) = COUNT(*)
        )",
        params![job_id],
        |row| row.get(0),
    )?;

    // Batches with at least one claimed row.
    let claimed = conn.query_row(
        "SELECT COUNT(DISTINCT batch_id) FROM citation_row
         WHERE job_id = ?1 AND batch_id IS NOT NULL AND status = 'claimed'",
        params![job_id],
        |row| row.get(0),
    )?;

    Ok(BatchCounts {
        total,
        done,
        pending,
        claimed,
    })
}

/// Returns all citation rows with status 'done' for the given job.
pub fn list_done_rows(conn: &Connection, job_id: &str) -> Result<Vec<CitationRow>, CitationError> {
    list_rows_by_status(conn, job_id, "done")
}

/// Returns all citation rows for a job, regardless of status.
/// Ordered by id ascending.
pub fn list_all_rows(conn: &Connection, job_id: &str) -> Result<Vec<CitationRow>, CitationError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, job_id, group_id, cite_key, author, title, year, tex_line,
                anchor_before, anchor_after, section_title, matched_file_id,
                bib_abstract, bib_keywords, tex_context, batch_id, status, flag,
                claimed_at, result_json, co_citation_json
         FROM citation_row WHERE job_id = ?1
         ORDER BY id ASC",
    )?;

    let rows = stmt.query_map(params![job_id], row_to_citation_row)?;
    let mut result = Vec::new();
    for row in rows {
        result.push(row?);
    }
    Ok(result)
}

/// Updates the matched_file_id for a citation row. Used by the autonomous
/// agent after source-fetching to re-match previously unmatched rows against
/// newly indexed files. Only rows with `matched_file_id IS NULL` and status
/// 'pending' are updated to avoid overwriting manually overridden matches
/// or already-processed rows.
///
/// Returns the number of rows updated (0 or 1).
pub fn update_matched_file_id(
    conn: &Connection,
    row_id: i64,
    file_id: i64,
) -> Result<usize, CitationError> {
    let updated = conn.execute(
        "UPDATE citation_row SET matched_file_id = ?1
         WHERE id = ?2 AND matched_file_id IS NULL AND status = 'pending'",
        params![file_id, row_id],
    )?;
    Ok(updated)
}

/// Returns citation rows for a job with optional status filter and pagination.
/// When `status` is None, returns rows in all statuses.
/// `offset` is the number of rows to skip (0-based), `limit` is the maximum
/// number of rows to return.
///
/// Also returns the total count of matching rows (before pagination) so the
/// caller can construct pagination metadata.
pub fn list_rows_filtered(
    conn: &Connection,
    job_id: &str,
    status: Option<&str>,
    offset: i64,
    limit: i64,
) -> Result<(Vec<CitationRow>, i64), CitationError> {
    let (rows, total) = if let Some(s) = status {
        // Count total matching rows.
        let total: i64 = conn.query_row(
            "SELECT COUNT(*) FROM citation_row WHERE job_id = ?1 AND status = ?2",
            params![job_id, s],
            |row| row.get(0),
        )?;

        let mut stmt = conn.prepare_cached(
            "SELECT id, job_id, group_id, cite_key, author, title, year, tex_line,
                    anchor_before, anchor_after, section_title, matched_file_id,
                    bib_abstract, bib_keywords, tex_context, batch_id, status, flag,
                    claimed_at, result_json, co_citation_json
             FROM citation_row WHERE job_id = ?1 AND status = ?2
             ORDER BY id ASC
             LIMIT ?3 OFFSET ?4",
        )?;

        let mapped = stmt.query_map(params![job_id, s, limit, offset], row_to_citation_row)?;
        let mut result = Vec::new();
        for row in mapped {
            result.push(row?);
        }
        (result, total)
    } else {
        // No status filter -- return all rows.
        let total: i64 = conn.query_row(
            "SELECT COUNT(*) FROM citation_row WHERE job_id = ?1",
            params![job_id],
            |row| row.get(0),
        )?;

        let mut stmt = conn.prepare_cached(
            "SELECT id, job_id, group_id, cite_key, author, title, year, tex_line,
                    anchor_before, anchor_after, section_title, matched_file_id,
                    bib_abstract, bib_keywords, tex_context, batch_id, status, flag,
                    claimed_at, result_json, co_citation_json
             FROM citation_row WHERE job_id = ?1
             ORDER BY id ASC
             LIMIT ?2 OFFSET ?3",
        )?;

        let mapped = stmt.query_map(params![job_id, limit, offset], row_to_citation_row)?;
        let mut result = Vec::new();
        for row in mapped {
            result.push(row?);
        }
        (result, total)
    };

    Ok((rows, total))
}

/// Claims a specific batch by batch_id for sub-agent processing.
/// Allows targeted retry of failed batches or out-of-order claiming.
///
/// Steps:
/// 1. Reclaims stale batches (same as `claim_next_batch`).
/// 2. Checks that the target batch exists and all its rows are in 'pending'
///    or 'failed' status (not 'claimed' or 'done').
/// 3. Resets any 'failed' rows in the batch to 'pending'.
/// 4. Atomically claims all rows in the batch.
///
/// Returns None if the batch does not exist, has no rows, or contains rows
/// that are already claimed or done.
pub fn claim_specific_batch(
    conn: &Connection,
    job_id: &str,
    batch_id: i64,
) -> Result<Option<BatchClaim>, CitationError> {
    let now = unix_now();
    let stale_cutoff = now - STALE_TIMEOUT_SECS;

    // Step 1: Reclaim stale batches (same as claim_next_batch).
    conn.execute(
        "UPDATE citation_row SET status = 'pending', claimed_at = NULL
         WHERE job_id = ?1 AND status = 'claimed' AND claimed_at < ?2",
        params![job_id, stale_cutoff],
    )?;

    // Step 2: Check that the batch exists and all rows are claimable
    // (pending or failed). If any row is 'claimed' or 'done', reject.
    let batch_row_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM citation_row
         WHERE job_id = ?1 AND batch_id = ?2",
        params![job_id, batch_id],
        |row| row.get(0),
    )?;

    if batch_row_count == 0 {
        return Ok(None);
    }

    let claimable_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM citation_row
         WHERE job_id = ?1 AND batch_id = ?2 AND status IN ('pending', 'failed')",
        params![job_id, batch_id],
        |row| row.get(0),
    )?;

    if claimable_count != batch_row_count {
        // Some rows are 'claimed' or 'done' -- batch is not fully claimable.
        return Ok(None);
    }

    // Step 3: Reset any 'failed' rows to 'pending' before claiming.
    conn.execute(
        "UPDATE citation_row SET status = 'pending', result_json = NULL, flag = NULL
         WHERE job_id = ?1 AND batch_id = ?2 AND status = 'failed'",
        params![job_id, batch_id],
    )?;

    // Step 4: Claim all rows in the batch.
    conn.execute(
        "UPDATE citation_row SET status = 'claimed', claimed_at = ?1
         WHERE job_id = ?2 AND batch_id = ?3 AND status = 'pending'",
        params![now, job_id, batch_id],
    )?;

    // Read back the claimed rows.
    let rows = list_rows_by_batch(conn, job_id, batch_id)?;

    // Read job params.
    let params_json: String = conn
        .query_row(
            "SELECT params_json FROM job WHERE id = ?1",
            params![job_id],
            |row| row.get(0),
        )
        .map_err(|_| CitationError::JobNotFound {
            job_id: job_id.to_string(),
        })?;

    let job_params: CitationJobParams = serde_json::from_str(&params_json)?;

    Ok(Some(BatchClaim {
        batch_id,
        tex_path: job_params.tex_path,
        bib_path: job_params.bib_path,
        session_id: job_params.session_id,
        rows,
    }))
}

/// Returns all citation rows that have a non-NULL flag (critical or warning).
pub fn list_flagged_rows(
    conn: &Connection,
    job_id: &str,
) -> Result<Vec<CitationRow>, CitationError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, job_id, group_id, cite_key, author, title, year, tex_line,
                anchor_before, anchor_after, section_title, matched_file_id,
                bib_abstract, bib_keywords, tex_context, batch_id, status, flag,
                claimed_at, result_json, co_citation_json
         FROM citation_row WHERE job_id = ?1 AND flag IS NOT NULL
         ORDER BY id ASC",
    )?;

    let rows = stmt.query_map(params![job_id], row_to_citation_row)?;
    let mut result = Vec::new();
    for row in rows {
        result.push(row?);
    }
    Ok(result)
}

/// Aggregates verdict counts from the result_json column of done rows.
/// Parses each result_json to extract the verdict field and counts occurrences.
pub fn count_verdicts(
    conn: &Connection,
    job_id: &str,
) -> Result<HashMap<String, i64>, CitationError> {
    let mut verdicts: HashMap<String, i64> = HashMap::new();

    let mut stmt = conn.prepare_cached(
        "SELECT result_json FROM citation_row
         WHERE job_id = ?1 AND status = 'done' AND result_json IS NOT NULL",
    )?;

    let rows = stmt.query_map(params![job_id], |row| {
        let json: String = row.get(0)?;
        Ok(json)
    })?;

    for row in rows {
        let json = row?;
        // Parse the verdict field from the result_json.
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json)
            && let Some(verdict) = parsed.get("verdict").and_then(|v| v.as_str())
        {
            *verdicts.entry(verdict.to_string()).or_insert(0) += 1;
        }
    }

    Ok(verdicts)
}

/// Builds Alert entries from flagged citation rows. Each alert contains
/// the row_id, cite_key, flag, and optionally the verdict and reasoning
/// extracted from result_json.
pub fn build_alerts(conn: &Connection, job_id: &str) -> Result<Vec<Alert>, CitationError> {
    let flagged = list_flagged_rows(conn, job_id)?;
    let mut alerts = Vec::new();

    for row in &flagged {
        let (verdict, reasoning) = if let Some(ref json) = row.result_json {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json) {
                let v = parsed
                    .get("verdict")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let r = parsed
                    .get("reasoning")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                (v, r)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        alerts.push(Alert {
            row_id: row.id,
            cite_key: row.cite_key.clone(),
            flag: row.flag.clone().unwrap_or_default(),
            verdict,
            reasoning,
        });
    }

    Ok(alerts)
}

/// Returns the count of stale batches -- batches whose rows are in 'claimed'
/// status with `claimed_at` older than STALE_TIMEOUT_SECS from now. These
/// batches were claimed by a sub-agent that has not submitted results within
/// the timeout window and will be reclaimed on the next claim attempt.
pub fn count_stale_batches(conn: &Connection, job_id: &str) -> Result<i64, CitationError> {
    let now = unix_now();
    let stale_cutoff = now - STALE_TIMEOUT_SECS;

    let count: i64 = conn.query_row(
        "SELECT COUNT(DISTINCT batch_id) FROM citation_row
         WHERE job_id = ?1 AND status = 'claimed' AND claimed_at < ?2 AND batch_id IS NOT NULL",
        params![job_id, stale_cutoff],
        |row| row.get(0),
    )?;

    Ok(count)
}

/// Computes the average duration in seconds for completed batches. A batch
/// is considered "completed" when all its rows are in 'done' or 'failed'
/// status. The duration is measured from the batch's `claimed_at` timestamp
/// to the last `result_json`-bearing row's effective completion time.
///
/// Since there is no explicit `completed_at` column on citation_row, the
/// duration is approximated using `claimed_at` of the batch and the current
/// time for batches that just transitioned to done. For historical analysis,
/// this approximation is sufficient because batch processing durations are
/// typically short (seconds to minutes).
///
/// Returns None if no completed batches exist.
pub fn avg_completed_batch_duration(
    conn: &Connection,
    job_id: &str,
) -> Result<Option<f64>, CitationError> {
    let now = unix_now();

    let result: Option<f64> = conn
        .query_row(
            "SELECT AVG(batch_duration) FROM (
                SELECT (?1 - MIN(claimed_at)) AS batch_duration
                FROM citation_row
                WHERE job_id = ?2 AND batch_id IS NOT NULL AND claimed_at IS NOT NULL
                GROUP BY batch_id
                HAVING SUM(CASE WHEN status IN ('done', 'failed') THEN 1 ELSE 0 END) = COUNT(*)
            )",
            params![now, job_id],
            |row| row.get(0),
        )
        .ok()
        .flatten();

    Ok(result)
}

/// Resets failed citation rows back to pending status so they can be claimed
/// and processed again by sub-agents. Clears the previous result_json, flag,
/// and claimed_at fields to ensure a clean retry.
///
/// When `batch_ids` is `Some`, only rows belonging to the specified batch IDs
/// are reset. When `batch_ids` is `None`, all failed rows in the job are reset.
///
/// Returns a tuple of (rows_reset, batches_affected) where batches_affected
/// is the count of distinct batch_ids among the reset rows.
pub fn reset_failed_rows(
    conn: &Connection,
    job_id: &str,
    batch_ids: Option<&[i64]>,
) -> Result<(i64, i64), CitationError> {
    // Count distinct batches that will be affected before the update,
    // so the caller knows how many batches were reset.
    let batches_affected: i64 = match batch_ids {
        Some(ids) if !ids.is_empty() => {
            let placeholders: String = std::iter::repeat_n("?", ids.len())
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT COUNT(DISTINCT batch_id) FROM citation_row \
                 WHERE job_id = ?1 AND status = 'failed' AND batch_id IN ({placeholders})"
            );
            let mut stmt = conn.prepare(&sql)?;
            let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> =
                vec![Box::new(job_id.to_string())];
            for &id in ids {
                params_vec.push(Box::new(id));
            }
            let param_refs: Vec<&dyn rusqlite::types::ToSql> =
                params_vec.iter().map(|p| p.as_ref()).collect();
            stmt.query_row(param_refs.as_slice(), |row| row.get(0))?
        }
        _ => conn.query_row(
            "SELECT COUNT(DISTINCT batch_id) FROM citation_row \
             WHERE job_id = ?1 AND status = 'failed'",
            params![job_id],
            |row| row.get(0),
        )?,
    };

    // Reset failed rows: clear status back to pending, remove stale
    // result data so the sub-agent starts fresh.
    let rows_reset: i64 = match batch_ids {
        Some(ids) if !ids.is_empty() => {
            let placeholders: String = std::iter::repeat_n("?", ids.len())
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "UPDATE citation_row \
                 SET status = 'pending', result_json = NULL, flag = NULL, claimed_at = NULL \
                 WHERE job_id = ?1 AND status = 'failed' AND batch_id IN ({placeholders})"
            );
            let mut stmt = conn.prepare(&sql)?;
            let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> =
                vec![Box::new(job_id.to_string())];
            for &id in ids {
                params_vec.push(Box::new(id));
            }
            let param_refs: Vec<&dyn rusqlite::types::ToSql> =
                params_vec.iter().map(|p| p.as_ref()).collect();
            stmt.execute(param_refs.as_slice())? as i64
        }
        _ => conn.execute(
            "UPDATE citation_row \
             SET status = 'pending', result_json = NULL, flag = NULL, claimed_at = NULL \
             WHERE job_id = ?1 AND status = 'failed'",
            params![job_id],
        )? as i64,
    };

    Ok((rows_reset, batches_affected))
}

/// Resets specific citation rows identified by their integer IDs back to
/// pending status. Only rows in 'done' or 'failed' status belonging to the
/// specified job are affected. Rows in 'pending' or 'claimed' status are
/// not touched because they either have no result yet or are actively being
/// processed by a sub-agent.
///
/// Clears result_json, flag, and claimed_at so the sub-agent starts with
/// a clean slate on the next claim cycle.
///
/// Returns (rows_reset, batches_affected) where batches_affected is the
/// count of distinct batch_ids among the reset rows.
pub fn reset_rows_by_ids(
    conn: &Connection,
    job_id: &str,
    row_ids: &[i64],
) -> Result<(i64, i64), CitationError> {
    if row_ids.is_empty() {
        return Ok((0, 0));
    }

    let placeholders: String = std::iter::repeat_n("?", row_ids.len())
        .collect::<Vec<_>>()
        .join(", ");

    // Count distinct batches that will be affected.
    let count_sql = format!(
        "SELECT COUNT(DISTINCT batch_id) FROM citation_row \
         WHERE job_id = ?1 AND id IN ({placeholders}) AND status IN ('done', 'failed')"
    );
    let mut stmt = conn.prepare(&count_sql)?;
    let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(job_id.to_string())];
    for &id in row_ids {
        params_vec.push(Box::new(id));
    }
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        params_vec.iter().map(|p| p.as_ref()).collect();
    let batches_affected: i64 = stmt.query_row(param_refs.as_slice(), |row| row.get(0))?;

    // Reset matching rows to pending.
    let update_sql = format!(
        "UPDATE citation_row \
         SET status = 'pending', result_json = NULL, flag = NULL, claimed_at = NULL \
         WHERE job_id = ?1 AND id IN ({placeholders}) AND status IN ('done', 'failed')"
    );
    let mut stmt = conn.prepare(&update_sql)?;
    let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(job_id.to_string())];
    for &id in row_ids {
        params_vec.push(Box::new(id));
    }
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        params_vec.iter().map(|p| p.as_ref()).collect();
    let rows_reset = stmt.execute(param_refs.as_slice())? as i64;

    Ok((rows_reset, batches_affected))
}

/// Resets all citation rows whose flag column matches any value in the
/// provided flags slice. Targets rows in 'done' status that carry the
/// specified flag values (e.g., "warning", "critical"). Rows in 'pending',
/// 'claimed', or 'failed' status are not touched.
///
/// Clears result_json, flag, and claimed_at so the sub-agent starts with
/// a clean slate on the next claim cycle.
///
/// Returns (rows_reset, batches_affected) where batches_affected is the
/// count of distinct batch_ids among the reset rows.
pub fn reset_rows_by_flags(
    conn: &Connection,
    job_id: &str,
    flags: &[&str],
) -> Result<(i64, i64), CitationError> {
    if flags.is_empty() {
        return Ok((0, 0));
    }

    let placeholders: String = std::iter::repeat_n("?", flags.len())
        .collect::<Vec<_>>()
        .join(", ");

    // Count distinct batches that will be affected.
    let count_sql = format!(
        "SELECT COUNT(DISTINCT batch_id) FROM citation_row \
         WHERE job_id = ?1 AND status = 'done' AND flag IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&count_sql)?;
    let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(job_id.to_string())];
    for &f in flags {
        params_vec.push(Box::new(f.to_string()));
    }
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        params_vec.iter().map(|p| p.as_ref()).collect();
    let batches_affected: i64 = stmt.query_row(param_refs.as_slice(), |row| row.get(0))?;

    // Reset matching rows to pending.
    let update_sql = format!(
        "UPDATE citation_row \
         SET status = 'pending', result_json = NULL, flag = NULL, claimed_at = NULL \
         WHERE job_id = ?1 AND status = 'done' AND flag IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&update_sql)?;
    let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(job_id.to_string())];
    for &f in flags {
        params_vec.push(Box::new(f.to_string()));
    }
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        params_vec.iter().map(|p| p.as_ref()).collect();
    let rows_reset = stmt.execute(param_refs.as_slice())? as i64;

    Ok((rows_reset, batches_affected))
}

// -------------------------------------------------------------------------
// Internal helpers
// -------------------------------------------------------------------------

/// Returns all rows for a specific batch within a job.
fn list_rows_by_batch(
    conn: &Connection,
    job_id: &str,
    batch_id: i64,
) -> Result<Vec<CitationRow>, CitationError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, job_id, group_id, cite_key, author, title, year, tex_line,
                anchor_before, anchor_after, section_title, matched_file_id,
                bib_abstract, bib_keywords, tex_context, batch_id, status, flag,
                claimed_at, result_json, co_citation_json
         FROM citation_row WHERE job_id = ?1 AND batch_id = ?2
         ORDER BY id ASC",
    )?;

    let rows = stmt.query_map(params![job_id, batch_id], row_to_citation_row)?;
    let mut result = Vec::new();
    for row in rows {
        result.push(row?);
    }
    Ok(result)
}

/// Returns all rows for a specific status within a job.
fn list_rows_by_status(
    conn: &Connection,
    job_id: &str,
    status: &str,
) -> Result<Vec<CitationRow>, CitationError> {
    let mut stmt = conn.prepare_cached(
        "SELECT id, job_id, group_id, cite_key, author, title, year, tex_line,
                anchor_before, anchor_after, section_title, matched_file_id,
                bib_abstract, bib_keywords, tex_context, batch_id, status, flag,
                claimed_at, result_json, co_citation_json
         FROM citation_row WHERE job_id = ?1 AND status = ?2
         ORDER BY id ASC",
    )?;

    let rows = stmt.query_map(params![job_id, status], row_to_citation_row)?;
    let mut result = Vec::new();
    for row in rows {
        result.push(row?);
    }
    Ok(result)
}

/// Maps a rusqlite row (21 columns) to a CitationRow struct.
fn row_to_citation_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CitationRow> {
    Ok(CitationRow {
        id: row.get(0)?,
        job_id: row.get(1)?,
        group_id: row.get(2)?,
        cite_key: row.get(3)?,
        author: row.get(4)?,
        title: row.get(5)?,
        year: row.get(6)?,
        tex_line: row.get(7)?,
        anchor_before: row.get(8)?,
        anchor_after: row.get(9)?,
        section_title: row.get(10)?,
        matched_file_id: row.get(11)?,
        bib_abstract: row.get(12)?,
        bib_keywords: row.get(13)?,
        tex_context: row.get(14)?,
        batch_id: row.get(15)?,
        status: row.get(16)?,
        flag: row.get(17)?,
        claimed_at: row.get(18)?,
        result_json: row.get(19)?,
        co_citation_json: row.get(20)?,
    })
}

/// Returns the current Unix epoch timestamp in seconds (UTC).
/// Falls back to 0 if the system clock is before the UNIX epoch (theoretical
/// impossibility on supported platforms, but avoids a panic).
fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(std::time::Duration::ZERO)
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CoCitationInfo, CorrectionType, LatexCorrection, OtherSourceEntry, OtherSourcePassage,
    };
    use neuroncite_store::migrate;

    /// Helper: creates an in-memory database with the full schema applied.
    fn setup_db() -> Connection {
        let conn = Connection::open_in_memory().expect("failed to open in-memory db");
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .expect("failed to enable foreign keys");
        migrate(&conn).expect("migration failed");
        conn
    }

    /// Helper: creates a job record and returns its ID.
    fn create_test_job(conn: &Connection, job_id: &str, params: &CitationJobParams) {
        let params_json = serde_json::to_string(params).expect("serialize params");
        conn.execute(
            "INSERT INTO job (id, kind, state, progress_done, progress_total, created_at, params_json)
             VALUES (?1, 'citation_verify', 'running', 0, 0, ?2, ?3)",
            params![job_id, unix_now(), params_json],
        )
        .expect("create test job failed");
    }

    /// Helper: creates sample citation row inserts for testing.
    fn sample_inserts(job_id: &str, count: usize) -> Vec<CitationRowInsert> {
        (0..count)
            .map(|i| CitationRowInsert {
                job_id: job_id.to_string(),
                group_id: i as i64,
                cite_key: format!("key{i}"),
                author: format!("Author {i}"),
                title: format!("Title {i}"),
                year: Some(format!("{}", 2000 + i)),
                tex_line: (10 + i * 5) as i64,
                anchor_before: "context words before".to_string(),
                anchor_after: "context words after".to_string(),
                section_title: Some("Introduction".to_string()),
                matched_file_id: if i % 2 == 0 { Some(i as i64) } else { None },
                bib_abstract: Some(format!("Abstract for key{i}")),
                bib_keywords: Some(format!("keyword{i}")),
                tex_context: format!("surrounding LaTeX text for key{i} with citation context"),
                batch_id: Some((i / 3) as i64),
                co_citation_json: None,
            })
            .collect()
    }

    /// Helper: creates a SubmitEntry with the mandatory field structure.
    fn make_submit_entry(row_id: i64, verdict: crate::types::Verdict) -> SubmitEntry {
        SubmitEntry {
            row_id,
            verdict,
            claim_original: "test claim".to_string(),
            claim_english: "test claim in English".to_string(),
            source_match: true,
            other_source_list: vec![],
            passages: vec![],
            reasoning: "found matching passage".to_string(),
            confidence: 0.75,
            search_rounds: 2,
            flag: String::new(),
            better_source: vec![],
            latex_correction: LatexCorrection {
                correction_type: CorrectionType::None,
                original_text: String::new(),
                suggested_text: String::new(),
                explanation: String::new(),
            },
        }
    }

    fn default_params() -> CitationJobParams {
        CitationJobParams {
            tex_path: "/test/paper.tex".to_string(),
            bib_path: "/test/refs.bib".to_string(),
            session_id: 1,
            session_ids: vec![1],
            batch_size: 5,
        }
    }

    /// T-CIT-107: insert_citation_rows stores all fields correctly.
    #[test]
    fn t_cit_024_insert_rows() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-024", &params);

        let inserts = sample_inserts("job-024", 3);
        let count = insert_citation_rows(&conn, &inserts).expect("insert failed");
        assert_eq!(count, 3);

        // Verify fields are stored correctly.
        let row: CitationRow = conn
            .query_row(
                "SELECT id, job_id, group_id, cite_key, author, title, year, tex_line,
                        anchor_before, anchor_after, section_title, matched_file_id,
                        bib_abstract, bib_keywords, tex_context, batch_id, status, flag,
                        claimed_at, result_json, co_citation_json
                 FROM citation_row WHERE cite_key = 'key0'",
                [],
                row_to_citation_row,
            )
            .expect("query row failed");

        assert_eq!(row.job_id, "job-024");
        assert_eq!(row.group_id, 0);
        assert_eq!(row.author, "Author 0");
        assert_eq!(row.title, "Title 0");
        assert_eq!(row.year.as_deref(), Some("2000"));
        assert_eq!(row.tex_line, 10);
        assert_eq!(row.status, "pending");
        assert!(row.flag.is_none());
        assert!(row.result_json.is_none());
    }

    /// T-CIT-108: claim_next_batch returns the lowest pending batch.
    #[test]
    fn t_cit_025_claim_lowest_batch() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-025", &params);

        let inserts = sample_inserts("job-025", 6); // batch_ids: 0,0,0,1,1,1
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        let claim = claim_next_batch(&conn, "job-025")
            .expect("claim failed")
            .expect("expected a batch");

        assert_eq!(claim.batch_id, 0);
        assert_eq!(claim.rows.len(), 3);
        assert_eq!(claim.tex_path, "/test/paper.tex");
        assert_eq!(claim.session_id, 1);

        // All rows in the claimed batch have status 'claimed'.
        for row in &claim.rows {
            assert_eq!(row.status, "claimed");
        }
    }

    /// T-CIT-026: claim_next_batch returns None when all batches are done.
    #[test]
    fn t_cit_026_no_pending_batches() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-026", &params);

        let inserts = sample_inserts("job-026", 3);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Mark all rows as done.
        conn.execute(
            "UPDATE citation_row SET status = 'done' WHERE job_id = 'job-026'",
            [],
        )
        .expect("update failed");

        let claim = claim_next_batch(&conn, "job-026").expect("claim failed");
        assert!(claim.is_none());
    }

    /// T-CIT-027: Two sequential claims take different batches (atomic).
    #[test]
    fn t_cit_027_sequential_claims_different_batches() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-027", &params);

        let inserts = sample_inserts("job-027", 6); // batch 0 and batch 1
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        let claim1 = claim_next_batch(&conn, "job-027")
            .expect("claim 1 failed")
            .expect("expected batch 0");
        let claim2 = claim_next_batch(&conn, "job-027")
            .expect("claim 2 failed")
            .expect("expected batch 1");

        assert_eq!(claim1.batch_id, 0);
        assert_eq!(claim2.batch_id, 1);
        assert_ne!(claim1.batch_id, claim2.batch_id);
    }

    /// T-CIT-028: Stale batch reclaim after timeout. A claimed batch older
    /// than STALE_TIMEOUT_SECS is reset to pending and can be claimed again.
    #[test]
    fn t_cit_028_stale_reclaim() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-028", &params);

        let inserts = sample_inserts("job-028", 3);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Simulate a claim from 10 minutes ago (stale).
        let stale_time = unix_now() - 600;
        conn.execute(
            "UPDATE citation_row SET status = 'claimed', claimed_at = ?1 WHERE job_id = 'job-028'",
            params![stale_time],
        )
        .expect("simulate stale claim");

        // claim_next_batch should reclaim the stale batch.
        let claim = claim_next_batch(&conn, "job-028")
            .expect("claim failed")
            .expect("stale batch should be reclaimable");

        assert_eq!(claim.batch_id, 0);
        for row in &claim.rows {
            assert_eq!(row.status, "claimed");
        }
    }

    /// T-CIT-029: submit_batch_results updates status and stores result_json.
    #[test]
    fn t_cit_029_submit_results() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-029", &params);

        let inserts = sample_inserts("job-029", 3);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Claim the batch.
        let claim = claim_next_batch(&conn, "job-029")
            .expect("claim failed")
            .expect("expected batch");

        // Submit results for all rows in the batch.
        let results: Vec<SubmitEntry> = claim
            .rows
            .iter()
            .map(|row| make_submit_entry(row.id, crate::types::Verdict::Supported))
            .collect();

        let count = submit_batch_results(&conn, "job-029", &results).expect("submit failed");
        assert_eq!(count, 3);

        // Verify rows are marked as done.
        let status_counts = count_by_status(&conn, "job-029").expect("count failed");
        assert_eq!(status_counts.done, 3);
        assert_eq!(status_counts.pending, 0);

        // Verify result_json is stored.
        let row: CitationRow = conn
            .query_row(
                "SELECT id, job_id, group_id, cite_key, author, title, year, tex_line,
                        anchor_before, anchor_after, section_title, matched_file_id,
                        bib_abstract, bib_keywords, tex_context, batch_id, status, flag,
                        claimed_at, result_json, co_citation_json
                 FROM citation_row WHERE id = ?1",
                params![claim.rows[0].id],
                row_to_citation_row,
            )
            .expect("query failed");

        assert_eq!(row.status, "done");
        assert!(row.result_json.is_some());
    }

    /// T-CIT-030: submit rejects row that is not in claimed status.
    #[test]
    fn t_cit_030_submit_rejects_unclaimed() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-030", &params);

        let inserts = sample_inserts("job-030", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Row is still 'pending', not 'claimed'.
        let row_id: i64 = conn
            .query_row(
                "SELECT id FROM citation_row WHERE job_id = 'job-030' LIMIT 1",
                [],
                |row| row.get(0),
            )
            .expect("query id failed");

        let results = vec![make_submit_entry(row_id, crate::types::Verdict::Supported)];

        let err = submit_batch_results(&conn, "job-030", &results);
        assert!(err.is_err(), "submitting unclaimed row must fail");
    }

    /// T-CIT-031: submit rejects row that belongs to a different job.
    #[test]
    fn t_cit_031_submit_rejects_wrong_job() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-031a", &params);
        create_test_job(&conn, "job-031b", &params);

        let inserts_a = vec![CitationRowInsert {
            job_id: "job-031a".to_string(),
            group_id: 0,
            cite_key: "keya".to_string(),
            author: "A".to_string(),
            title: "T".to_string(),
            year: None,
            tex_line: 1,
            anchor_before: "b".to_string(),
            anchor_after: "a".to_string(),
            section_title: None,
            matched_file_id: None,
            bib_abstract: None,
            bib_keywords: None,
            tex_context: "surrounding context".to_string(),
            batch_id: Some(0),
            co_citation_json: None,
        }];
        insert_citation_rows(&conn, &inserts_a).expect("insert a");

        // Claim the row in job-031a.
        conn.execute(
            "UPDATE citation_row SET status = 'claimed', claimed_at = ?1 WHERE job_id = 'job-031a'",
            params![unix_now()],
        )
        .expect("claim");

        let row_id: i64 = conn
            .query_row(
                "SELECT id FROM citation_row WHERE job_id = 'job-031a' LIMIT 1",
                [],
                |row| row.get(0),
            )
            .expect("query id");

        // Submit with job_id = 'job-031b' (wrong job).
        let results = vec![make_submit_entry(row_id, crate::types::Verdict::Supported)];

        let err = submit_batch_results(&conn, "job-031b", &results);
        assert!(err.is_err(), "submitting with wrong job_id must fail");
    }

    /// T-CIT-032: count_by_status returns correct counts.
    #[test]
    fn t_cit_032_count_by_status() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-032", &params);

        let inserts = sample_inserts("job-032", 5);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Set different statuses: 2 pending, 1 claimed, 1 done, 1 failed.
        conn.execute_batch(
            "UPDATE citation_row SET status = 'claimed' WHERE job_id = 'job-032' AND cite_key = 'key2';
             UPDATE citation_row SET status = 'done' WHERE job_id = 'job-032' AND cite_key = 'key3';
             UPDATE citation_row SET status = 'failed' WHERE job_id = 'job-032' AND cite_key = 'key4';",
        )
        .expect("update statuses");

        let counts = count_by_status(&conn, "job-032").expect("count failed");
        assert_eq!(counts.pending, 2);
        assert_eq!(counts.claimed, 1);
        assert_eq!(counts.done, 1);
        assert_eq!(counts.failed, 1);
    }

    /// T-CIT-033: count_verdicts aggregates from result_json.
    #[test]
    fn t_cit_033_count_verdicts() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-033", &params);

        let inserts = sample_inserts("job-033", 3);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Set all to done with different verdicts in result_json.
        conn.execute(
            "UPDATE citation_row SET status = 'done', result_json = '{\"verdict\":\"supported\"}' WHERE cite_key = 'key0' AND job_id = 'job-033'",
            [],
        ).expect("update 0");
        conn.execute(
            "UPDATE citation_row SET status = 'done', result_json = '{\"verdict\":\"supported\"}' WHERE cite_key = 'key1' AND job_id = 'job-033'",
            [],
        ).expect("update 1");
        conn.execute(
            "UPDATE citation_row SET status = 'done', result_json = '{\"verdict\":\"unsupported\"}' WHERE cite_key = 'key2' AND job_id = 'job-033'",
            [],
        ).expect("update 2");

        let verdicts = count_verdicts(&conn, "job-033").expect("count verdicts failed");
        assert_eq!(verdicts.get("supported"), Some(&2));
        assert_eq!(verdicts.get("unsupported"), Some(&1));
    }

    /// T-CIT-034: list_flagged_rows returns only rows with non-NULL flag.
    #[test]
    fn t_cit_034_list_flagged() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-034", &params);

        let inserts = sample_inserts("job-034", 3);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Set one row with a critical flag.
        conn.execute(
            "UPDATE citation_row SET flag = 'critical', status = 'done', result_json = '{\"verdict\":\"wrong_source\",\"reasoning\":\"bad source\"}'
             WHERE cite_key = 'key1' AND job_id = 'job-034'",
            [],
        )
        .expect("set flag");

        let flagged = list_flagged_rows(&conn, "job-034").expect("list flagged failed");
        assert_eq!(flagged.len(), 1);
        assert_eq!(flagged[0].cite_key, "key1");
        assert_eq!(flagged[0].flag.as_deref(), Some("critical"));
    }

    /// T-CIT-035: progress_done updates correctly after submit.
    #[test]
    fn t_cit_035_progress_updates() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-035", &params);

        let inserts = sample_inserts("job-035", 6); // 2 batches of 3
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Claim and submit batch 0.
        let claim = claim_next_batch(&conn, "job-035")
            .expect("claim failed")
            .expect("batch 0");

        let results: Vec<SubmitEntry> = claim
            .rows
            .iter()
            .map(|r| make_submit_entry(r.id, crate::types::Verdict::Supported))
            .collect();

        submit_batch_results(&conn, "job-035", &results).expect("submit");

        // Check job progress.
        let (done, total): (i64, i64) = conn
            .query_row(
                "SELECT progress_done, progress_total FROM job WHERE id = 'job-035'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("query progress");

        assert_eq!(done, 3, "3 rows submitted");
        assert_eq!(total, 6, "6 total rows");
    }

    /// T-CIT-044: list_all_rows returns rows in all statuses.
    #[test]
    fn t_cit_044_list_all_rows() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-044", &params);

        let inserts = sample_inserts("job-044", 5);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Set mixed statuses.
        conn.execute_batch(
            "UPDATE citation_row SET status = 'done' WHERE job_id = 'job-044' AND cite_key = 'key0';
             UPDATE citation_row SET status = 'claimed' WHERE job_id = 'job-044' AND cite_key = 'key1';
             UPDATE citation_row SET status = 'failed' WHERE job_id = 'job-044' AND cite_key = 'key2';",
        )
        .expect("set statuses");

        let all = list_all_rows(&conn, "job-044").expect("list_all_rows failed");
        assert_eq!(all.len(), 5, "all 5 rows returned regardless of status");

        // Verify statuses are mixed.
        let statuses: Vec<&str> = all.iter().map(|r| r.status.as_str()).collect();
        assert!(statuses.contains(&"done"));
        assert!(statuses.contains(&"claimed"));
        assert!(statuses.contains(&"failed"));
        assert!(statuses.contains(&"pending"));
    }

    /// T-CIT-045: list_rows_filtered with status filter returns only matching rows.
    #[test]
    fn t_cit_045_list_rows_filtered_by_status() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-045", &params);

        let inserts = sample_inserts("job-045", 5);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Set 2 rows to done.
        conn.execute_batch(
            "UPDATE citation_row SET status = 'done' WHERE job_id = 'job-045' AND cite_key IN ('key0', 'key2');",
        )
        .expect("set done");

        let (rows, total) =
            list_rows_filtered(&conn, "job-045", Some("done"), 0, 100).expect("filtered failed");
        assert_eq!(total, 2, "2 done rows total");
        assert_eq!(rows.len(), 2, "2 done rows returned");
        for row in &rows {
            assert_eq!(row.status, "done");
        }
    }

    /// T-CIT-046: list_rows_filtered with pagination returns correct slice.
    #[test]
    fn t_cit_046_list_rows_filtered_pagination() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-046", &params);

        // Insert 10 rows (batch_ids: 0,0,0,1,1,1,2,2,2,3).
        let inserts: Vec<CitationRowInsert> = (0..10)
            .map(|i| CitationRowInsert {
                job_id: "job-046".to_string(),
                group_id: i as i64,
                cite_key: format!("key{i}"),
                author: format!("Author {i}"),
                title: format!("Title {i}"),
                year: None,
                tex_line: i as i64,
                anchor_before: "b".to_string(),
                anchor_after: "a".to_string(),
                section_title: None,
                matched_file_id: None,
                bib_abstract: None,
                bib_keywords: None,
                tex_context: format!("context for key{i}"),
                batch_id: Some((i / 3) as i64),
                co_citation_json: None,
            })
            .collect();
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Paginate: offset=3, limit=2 (no status filter).
        let (rows, total) =
            list_rows_filtered(&conn, "job-046", None, 3, 2).expect("paginate failed");
        assert_eq!(total, 10, "10 total rows");
        assert_eq!(rows.len(), 2, "2 rows returned with limit=2");

        // With status filter and pagination.
        conn.execute_batch("UPDATE citation_row SET status = 'done' WHERE job_id = 'job-046';")
            .expect("set all done");

        let (rows, total) =
            list_rows_filtered(&conn, "job-046", Some("done"), 5, 3).expect("paginate done");
        assert_eq!(total, 10, "10 done rows total");
        assert_eq!(rows.len(), 3, "3 rows returned with limit=3 offset=5");
    }

    /// T-CIT-047: claim_specific_batch claims the requested batch directly.
    #[test]
    fn t_cit_047_claim_specific_batch() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-047", &params);

        let inserts = sample_inserts("job-047", 6); // batch_ids: 0,0,0,1,1,1
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Claim batch 1 specifically (skipping batch 0).
        let claim = claim_specific_batch(&conn, "job-047", 1)
            .expect("claim failed")
            .expect("expected batch 1");

        assert_eq!(claim.batch_id, 1);
        assert_eq!(claim.rows.len(), 3);
        for row in &claim.rows {
            assert_eq!(row.status, "claimed");
            assert_eq!(row.batch_id, Some(1));
        }

        // Batch 0 should still be pending.
        let counts = count_by_status(&conn, "job-047").expect("count failed");
        assert_eq!(counts.pending, 3, "batch 0 still pending");
        assert_eq!(counts.claimed, 3, "batch 1 claimed");
    }

    /// T-CIT-048: claim_specific_batch returns None for already claimed batch.
    #[test]
    fn t_cit_048_claim_specific_already_claimed() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-048", &params);

        let inserts = sample_inserts("job-048", 3); // batch 0
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Claim batch 0 via the normal path.
        claim_next_batch(&conn, "job-048")
            .expect("initial claim failed")
            .expect("expected batch 0");

        // Trying to claim batch 0 again should return None.
        let retry = claim_specific_batch(&conn, "job-048", 0).expect("retry claim failed");
        assert!(retry.is_none(), "already-claimed batch returns None");
    }

    /// T-CIT-049: claim_specific_batch retries failed batches by resetting
    /// failed rows to pending before claiming.
    #[test]
    fn t_cit_049_claim_specific_retry_failed() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-049", &params);

        let inserts = sample_inserts("job-049", 3); // batch 0
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Set all rows to 'failed' (simulating a failed batch).
        conn.execute(
            "UPDATE citation_row SET status = 'failed' WHERE job_id = 'job-049'",
            [],
        )
        .expect("set failed");

        // claim_specific_batch should reset failed -> pending -> claimed.
        let claim = claim_specific_batch(&conn, "job-049", 0)
            .expect("retry claim failed")
            .expect("expected batch 0 after retry");

        assert_eq!(claim.batch_id, 0);
        assert_eq!(claim.rows.len(), 3);
        for row in &claim.rows {
            assert_eq!(row.status, "claimed");
        }
    }

    /// T-CIT-040: reset_failed_rows resets all failed rows when no batch_ids filter.
    #[test]
    fn t_cit_040_reset_all_failed_rows() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-040", &params);

        let inserts = sample_inserts("job-040", 6); // batch 0 (3 rows), batch 1 (3 rows)
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Mark some rows as failed.
        conn.execute(
            "UPDATE citation_row SET status = 'failed', result_json = '{\"verdict\":\"not_found\"}' \
             WHERE job_id = 'job-040' AND cite_key IN ('key1', 'key4')",
            [],
        )
        .expect("set failed");

        let (rows_reset, batches) =
            reset_failed_rows(&conn, "job-040", None).expect("reset failed");
        assert_eq!(rows_reset, 2, "two rows were failed");
        assert_eq!(batches, 2, "failed rows span two batches");

        // Verify rows are back to pending with cleared fields.
        let counts = count_by_status(&conn, "job-040").expect("count");
        assert_eq!(counts.failed, 0, "no failed rows after reset");
        assert_eq!(counts.pending, 6, "all rows pending after reset");
    }

    /// T-CIT-041: reset_failed_rows with specific batch_ids only resets those batches.
    #[test]
    fn t_cit_041_reset_specific_batches() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-041", &params);

        let inserts = sample_inserts("job-041", 6);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Mark rows in both batches as failed.
        conn.execute(
            "UPDATE citation_row SET status = 'failed' WHERE job_id = 'job-041'",
            [],
        )
        .expect("set failed");

        // Reset only batch 0.
        let (rows_reset, batches) =
            reset_failed_rows(&conn, "job-041", Some(&[0])).expect("reset failed");
        assert_eq!(rows_reset, 3, "batch 0 has 3 rows");
        assert_eq!(batches, 1, "only one batch reset");

        // Batch 1 rows remain failed.
        let counts = count_by_status(&conn, "job-041").expect("count");
        assert_eq!(counts.pending, 3);
        assert_eq!(counts.failed, 3);
    }

    /// T-CIT-042: reset_failed_rows returns (0, 0) when no failed rows exist.
    #[test]
    fn t_cit_042_reset_no_failed_rows() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-042", &params);

        let inserts = sample_inserts("job-042", 3);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        let (rows_reset, batches) = reset_failed_rows(&conn, "job-042", None).expect("reset");
        assert_eq!(rows_reset, 0);
        assert_eq!(batches, 0);
    }

    /// T-CIT-043: reset_failed_rows clears result_json and flag fields.
    #[test]
    fn t_cit_043_reset_clears_result_fields() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-043", &params);

        let inserts = sample_inserts("job-043", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        conn.execute(
            "UPDATE citation_row SET status = 'failed', result_json = '{\"verdict\":\"not_found\"}', \
             flag = 'critical', claimed_at = 12345 WHERE job_id = 'job-043'",
            [],
        )
        .expect("set failed with data");

        reset_failed_rows(&conn, "job-043", None).expect("reset");

        let row: CitationRow = conn
            .query_row(
                "SELECT id, job_id, group_id, cite_key, author, title, year, tex_line,
                        anchor_before, anchor_after, section_title, matched_file_id,
                        bib_abstract, bib_keywords, tex_context, batch_id, status, flag,
                        claimed_at, result_json, co_citation_json
                 FROM citation_row WHERE job_id = 'job-043' LIMIT 1",
                [],
                row_to_citation_row,
            )
            .expect("query");

        assert_eq!(row.status, "pending");
        assert!(row.result_json.is_none(), "result_json must be cleared");
        assert!(row.flag.is_none(), "flag must be cleared");
        assert!(row.claimed_at.is_none(), "claimed_at must be cleared");
    }

    // -----------------------------------------------------------------------
    // Validation tests for the submit business rules
    // -----------------------------------------------------------------------

    /// T-CIT-050: submit rejects wrong_source verdict without other_source_list.
    #[test]
    fn t_cit_050_wrong_source_requires_other_sources() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-050", &params);

        let inserts = sample_inserts("job-050", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        let claim = claim_next_batch(&conn, "job-050")
            .expect("claim failed")
            .expect("batch");

        let mut entry = make_submit_entry(claim.rows[0].id, Verdict::WrongSource);
        entry.other_source_list = vec![]; // Empty -- must be rejected.

        let err = submit_batch_results(&conn, "job-050", &[entry]);
        assert!(
            err.is_err(),
            "wrong_source without other_source_list must fail"
        );
    }

    /// T-CIT-051: submit rejects partial verdict with confidence > 0.80.
    #[test]
    fn t_cit_051_partial_confidence_cap() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-051", &params);

        let inserts = sample_inserts("job-051", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        let claim = claim_next_batch(&conn, "job-051")
            .expect("claim failed")
            .expect("batch");

        let mut entry = make_submit_entry(claim.rows[0].id, Verdict::Partial);
        entry.confidence = 0.85; // Exceeds 0.80 cap for partial.
        entry.other_source_list = vec![OtherSourceEntry {
            cite_key_or_title: "alt_source".to_string(),
            passages: vec![OtherSourcePassage {
                text: "relevant text".to_string(),
                page: 1,
                score: 0.8,
            }],
        }];

        let err = submit_batch_results(&conn, "job-051", &[entry]);
        assert!(err.is_err(), "partial with confidence > 0.80 must fail");
    }

    /// T-CIT-052: submit accepts partial verdict with confidence <= 0.80.
    #[test]
    fn t_cit_052_partial_valid_confidence() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-052", &params);

        let inserts = sample_inserts("job-052", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        let claim = claim_next_batch(&conn, "job-052")
            .expect("claim failed")
            .expect("batch");

        let mut entry = make_submit_entry(claim.rows[0].id, Verdict::Partial);
        entry.confidence = 0.75;
        entry.other_source_list = vec![OtherSourceEntry {
            cite_key_or_title: "alt_source".to_string(),
            passages: vec![OtherSourcePassage {
                text: "relevant text".to_string(),
                page: 1,
                score: 0.8,
            }],
        }];

        let count =
            submit_batch_results(&conn, "job-052", &[entry]).expect("submit should succeed");
        assert_eq!(count, 1);
    }

    /// T-CIT-053: submit rejects invalid flag values.
    #[test]
    fn t_cit_053_invalid_flag_rejected() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-053", &params);

        let inserts = sample_inserts("job-053", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        let claim = claim_next_batch(&conn, "job-053")
            .expect("claim failed")
            .expect("batch");

        let mut entry = make_submit_entry(claim.rows[0].id, Verdict::Supported);
        entry.flag = "invalid_flag".to_string();

        let err = submit_batch_results(&conn, "job-053", &[entry]);
        assert!(err.is_err(), "invalid flag value must be rejected");
    }

    /// T-CIT-054: submit stores empty flag as NULL in the database.
    #[test]
    fn t_cit_054_empty_flag_stored_as_null() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-054", &params);

        let inserts = sample_inserts("job-054", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        let claim = claim_next_batch(&conn, "job-054")
            .expect("claim failed")
            .expect("batch");

        let entry = make_submit_entry(claim.rows[0].id, Verdict::Supported);
        submit_batch_results(&conn, "job-054", &[entry]).expect("submit");

        let flag: Option<String> = conn
            .query_row(
                "SELECT flag FROM citation_row WHERE id = ?1",
                params![claim.rows[0].id],
                |row| row.get(0),
            )
            .expect("query flag");

        assert!(flag.is_none(), "empty flag string must be stored as NULL");
    }

    /// T-CIT-055: co_citation_json is stored and retrieved correctly.
    #[test]
    fn t_cit_055_co_citation_json_roundtrip() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-055", &params);

        let co_info = CoCitationInfo {
            is_co_citation: true,
            co_cited_with: vec!["key_b".to_string(), "key_c".to_string()],
        };

        let inserts = vec![CitationRowInsert {
            job_id: "job-055".to_string(),
            group_id: 0,
            cite_key: "key_a".to_string(),
            author: "Author A".to_string(),
            title: "Title A".to_string(),
            year: None,
            tex_line: 1,
            anchor_before: "before".to_string(),
            anchor_after: "after".to_string(),
            section_title: None,
            matched_file_id: None,
            bib_abstract: None,
            bib_keywords: None,
            tex_context: "context".to_string(),
            batch_id: Some(0),
            co_citation_json: Some(serde_json::to_string(&co_info).expect("serialize")),
        }];

        insert_citation_rows(&conn, &inserts).expect("insert failed");

        let rows = list_all_rows(&conn, "job-055").expect("list failed");
        assert_eq!(rows.len(), 1);

        let retrieved = rows[0].co_citation_info();
        assert!(retrieved.is_co_citation);
        assert_eq!(retrieved.co_cited_with, vec!["key_b", "key_c"]);
    }

    /// T-CIT-056: co_citation_info returns default when JSON is NULL.
    #[test]
    fn t_cit_056_co_citation_null_defaults() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-056", &params);

        let inserts = sample_inserts("job-056", 1); // co_citation_json = None
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        let rows = list_all_rows(&conn, "job-056").expect("list failed");
        let info = rows[0].co_citation_info();
        assert!(!info.is_co_citation);
        assert!(info.co_cited_with.is_empty());
    }

    // -----------------------------------------------------------------------
    // Tests for reset_rows_by_ids and reset_rows_by_flags
    // -----------------------------------------------------------------------

    /// T-CIT-057: reset_rows_by_ids resets only the specified rows and leaves
    /// other rows in their original status.
    #[test]
    fn t_cit_057_reset_rows_by_ids_partial() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-057", &params);

        let inserts = sample_inserts("job-057", 3);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Claim and submit all rows to set them to 'done'.
        let batch = claim_next_batch(&conn, "job-057")
            .expect("claim failed")
            .expect("batch should exist");

        let entries: Vec<SubmitEntry> = batch
            .rows
            .iter()
            .map(|r| make_submit_entry(r.id, Verdict::Supported))
            .collect();
        submit_batch_results(&conn, "job-057", &entries).expect("submit failed");

        // Verify all 3 rows are done.
        let counts = count_by_status(&conn, "job-057").expect("count failed");
        assert_eq!(counts.done, 3);

        // Reset only the first two rows by ID.
        let row_ids: Vec<i64> = batch.rows.iter().take(2).map(|r| r.id).collect();
        let (rows_reset, batches_affected) =
            reset_rows_by_ids(&conn, "job-057", &row_ids).expect("reset failed");
        assert_eq!(rows_reset, 2);
        assert!(batches_affected >= 1);

        // Verify: 2 pending, 1 still done.
        let counts = count_by_status(&conn, "job-057").expect("count failed");
        assert_eq!(counts.pending, 2);
        assert_eq!(counts.done, 1);
    }

    /// T-CIT-058: reset_rows_by_ids with empty slice returns (0, 0) without
    /// modifying any rows.
    #[test]
    fn t_cit_058_reset_rows_by_ids_empty_slice() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-058", &params);

        let (rows_reset, batches_affected) =
            reset_rows_by_ids(&conn, "job-058", &[]).expect("reset failed");
        assert_eq!(rows_reset, 0);
        assert_eq!(batches_affected, 0);
    }

    /// T-CIT-059: reset_rows_by_flags resets only rows with matching flag
    /// values and leaves rows with other flags or no flag unchanged.
    #[test]
    fn t_cit_059_reset_rows_by_flags_warning_only() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-059", &params);

        let inserts = sample_inserts("job-059", 3);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Claim and submit all rows.
        let batch = claim_next_batch(&conn, "job-059")
            .expect("claim failed")
            .expect("batch should exist");

        let mut entries: Vec<SubmitEntry> = batch
            .rows
            .iter()
            .map(|r| make_submit_entry(r.id, Verdict::Supported))
            .collect();
        // Set flags: row 0 = warning, row 1 = critical, row 2 = no flag.
        entries[0].flag = "warning".to_string();
        entries[1].flag = "critical".to_string();
        entries[2].flag = String::new();
        submit_batch_results(&conn, "job-059", &entries).expect("submit failed");

        // Reset only warning-flagged rows.
        let (rows_reset, _batches) =
            reset_rows_by_flags(&conn, "job-059", &["warning"]).expect("reset failed");
        assert_eq!(rows_reset, 1);

        // Verify: 1 pending (warning reset), 2 still done (critical + no flag).
        let counts = count_by_status(&conn, "job-059").expect("count failed");
        assert_eq!(counts.pending, 1);
        assert_eq!(counts.done, 2);
    }

    /// T-CIT-060: reset_rows_by_ids returns correct batches_affected count
    /// when resetting rows across multiple batches.
    #[test]
    fn t_cit_060_reset_rows_by_ids_counts_batches() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-060", &params);

        // Create 6 rows in 2 batches (batch_id 0 has rows 0-2, batch_id 1 has rows 3-5).
        let inserts = sample_inserts("job-060", 6);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Claim and submit batch 0.
        let batch0 = claim_next_batch(&conn, "job-060")
            .expect("claim failed")
            .expect("batch0 should exist");
        let entries0: Vec<SubmitEntry> = batch0
            .rows
            .iter()
            .map(|r| make_submit_entry(r.id, Verdict::Supported))
            .collect();
        submit_batch_results(&conn, "job-060", &entries0).expect("submit0 failed");

        // Claim and submit batch 1.
        let batch1 = claim_next_batch(&conn, "job-060")
            .expect("claim failed")
            .expect("batch1 should exist");
        let entries1: Vec<SubmitEntry> = batch1
            .rows
            .iter()
            .map(|r| make_submit_entry(r.id, Verdict::Supported))
            .collect();
        submit_batch_results(&conn, "job-060", &entries1).expect("submit1 failed");

        // Reset one row from each batch.
        let ids = vec![batch0.rows[0].id, batch1.rows[0].id];
        let (rows_reset, batches_affected) =
            reset_rows_by_ids(&conn, "job-060", &ids).expect("reset failed");
        assert_eq!(rows_reset, 2);
        assert_eq!(
            batches_affected, 2,
            "rows from 2 different batches were reset"
        );
    }

    /// T-CIT-061: reset_rows_by_flags clears result_json and flag columns
    /// so the sub-agent starts with a clean slate.
    #[test]
    fn t_cit_061_reset_rows_by_flags_clears_result_json() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-061", &params);

        let inserts = sample_inserts("job-061", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");

        // Claim and submit with warning flag.
        let batch = claim_next_batch(&conn, "job-061")
            .expect("claim failed")
            .expect("batch should exist");
        let mut entry = make_submit_entry(batch.rows[0].id, Verdict::Supported);
        entry.flag = "warning".to_string();
        submit_batch_results(&conn, "job-061", &[entry]).expect("submit failed");

        // Verify result_json is set before reset.
        let row_before: (Option<String>, Option<String>) = conn
            .query_row(
                "SELECT result_json, flag FROM citation_row WHERE job_id = 'job-061'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("query before failed");
        assert!(
            row_before.0.is_some(),
            "result_json should be set before reset"
        );
        assert_eq!(row_before.1.as_deref(), Some("warning"));

        // Reset by flag.
        reset_rows_by_flags(&conn, "job-061", &["warning"]).expect("reset failed");

        // Verify result_json and flag are cleared.
        let row_after: (Option<String>, Option<String>) = conn
            .query_row(
                "SELECT result_json, flag FROM citation_row WHERE job_id = 'job-061'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("query after failed");
        assert!(
            row_after.0.is_none(),
            "result_json must be NULL after reset"
        );
        assert!(row_after.1.is_none(), "flag must be NULL after reset");
    }

    // -- Meta-confidence bounds validation tests --

    /// T-CIT-062: submit rejects supported verdict with confidence below 0.50.
    /// Under the meta-confidence model, a supported verdict with confidence
    /// below the 0.50 floor contradicts the requirement that the agent must
    /// be at least moderately certain the source supports the claim.
    #[test]
    fn t_cit_062_supported_confidence_floor() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-062", &params);

        let inserts = sample_inserts("job-062", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");
        let claim = claim_next_batch(&conn, "job-062")
            .expect("claim failed")
            .expect("batch");

        let mut entry = make_submit_entry(claim.rows[0].id, Verdict::Supported);
        entry.confidence = 0.40;

        let err = submit_batch_results(&conn, "job-062", &[entry]);
        assert!(err.is_err(), "supported with confidence < 0.50 must fail");
    }

    /// T-CIT-063: submit accepts supported verdict with confidence at the
    /// floor boundary (0.50).
    #[test]
    fn t_cit_063_supported_confidence_valid() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-063", &params);

        let inserts = sample_inserts("job-063", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");
        let claim = claim_next_batch(&conn, "job-063")
            .expect("claim failed")
            .expect("batch");

        let mut entry = make_submit_entry(claim.rows[0].id, Verdict::Supported);
        entry.confidence = 0.50;

        let count =
            submit_batch_results(&conn, "job-063", &[entry]).expect("submit should succeed");
        assert_eq!(count, 1);
    }

    /// T-CIT-064: submit rejects not_found verdict with confidence below 0.70.
    /// Corpus absence is a near-binary determination, so the meta-confidence
    /// floor is higher (0.70) than for interpretive verdicts.
    #[test]
    fn t_cit_064_not_found_confidence_floor() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-064", &params);

        let inserts = sample_inserts("job-064", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");
        let claim = claim_next_batch(&conn, "job-064")
            .expect("claim failed")
            .expect("batch");

        let mut entry = make_submit_entry(claim.rows[0].id, Verdict::NotFound);
        entry.confidence = 0.50;
        entry.source_match = false;

        let err = submit_batch_results(&conn, "job-064", &[entry]);
        assert!(err.is_err(), "not_found with confidence < 0.70 must fail");
    }

    /// T-CIT-065: submit rejects unsupported verdict with confidence below 0.50.
    #[test]
    fn t_cit_065_unsupported_confidence_floor() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-065", &params);

        let inserts = sample_inserts("job-065", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");
        let claim = claim_next_batch(&conn, "job-065")
            .expect("claim failed")
            .expect("batch");

        let mut entry = make_submit_entry(claim.rows[0].id, Verdict::Unsupported);
        entry.confidence = 0.30;

        let err = submit_batch_results(&conn, "job-065", &[entry]);
        assert!(err.is_err(), "unsupported with confidence < 0.50 must fail");
    }

    /// T-CIT-066: submit accepts unsupported verdict with high confidence
    /// (0.85). Under meta-confidence, a high value on unsupported means the
    /// agent is very certain the source does not support the claim.
    #[test]
    fn t_cit_066_unsupported_confidence_valid() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-066", &params);

        let inserts = sample_inserts("job-066", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");
        let claim = claim_next_batch(&conn, "job-066")
            .expect("claim failed")
            .expect("batch");

        let mut entry = make_submit_entry(claim.rows[0].id, Verdict::Unsupported);
        entry.confidence = 0.85;

        let count =
            submit_batch_results(&conn, "job-066", &[entry]).expect("submit should succeed");
        assert_eq!(count, 1);
    }

    /// T-CIT-067: submit rejects unverifiable verdict with confidence below
    /// 0.50. The agent must be at least moderately certain the claim cannot
    /// be checked.
    #[test]
    fn t_cit_067_unverifiable_confidence_floor() {
        let conn = setup_db();
        let params = default_params();
        create_test_job(&conn, "job-067", &params);

        let inserts = sample_inserts("job-067", 1);
        insert_citation_rows(&conn, &inserts).expect("insert failed");
        let claim = claim_next_batch(&conn, "job-067")
            .expect("claim failed")
            .expect("batch");

        let mut entry = make_submit_entry(claim.rows[0].id, Verdict::Unverifiable);
        entry.confidence = 0.30;

        let err = submit_batch_results(&conn, "job-067", &[entry]);
        assert!(
            err.is_err(),
            "unverifiable with confidence < 0.50 must fail"
        );
    }
}
