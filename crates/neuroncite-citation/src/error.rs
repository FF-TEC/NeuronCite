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

// Error types for the citation verification pipeline.
//
// Covers failures in LaTeX/BibTeX parsing, batch assignment, database
// operations (claim, submit, query), file matching, and export generation.

use thiserror::Error;

/// All error variants that can occur during citation verification.
#[derive(Debug, Error)]
pub enum CitationError {
    /// The LaTeX file could not be read from disk.
    #[error("failed to read LaTeX file: {path}: {reason}")]
    TexReadError { path: String, reason: String },

    /// The BibTeX file could not be read from disk.
    #[error("failed to read BibTeX file: {path}: {reason}")]
    BibReadError { path: String, reason: String },

    /// A cite-key referenced in the LaTeX file has no corresponding entry
    /// in the BibTeX file. The `cite_key` field contains the unresolved key.
    #[error("cite-key '{cite_key}' not found in BibTeX file")]
    UnresolvedCiteKey { cite_key: String },

    /// The LaTeX file contains no citation commands.
    #[error("no citation commands found in LaTeX file")]
    NoCitationsFound,

    /// A database operation failed during citation row insertion, claiming,
    /// or result submission.
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// A store-level error propagated from neuroncite-store (e.g., job
    /// creation, session lookup).
    #[error("store error: {0}")]
    Store(#[from] neuroncite_store::StoreError),

    /// The requested job does not exist or has an incompatible kind.
    #[error("job not found or wrong kind: {job_id}")]
    JobNotFound { job_id: String },

    /// A citation row referenced by row_id does not belong to the specified
    /// job, or is not in the expected status for the requested operation.
    #[error("invalid row state: row {row_id} is '{current_status}', expected '{expected_status}'")]
    InvalidRowState {
        row_id: i64,
        current_status: String,
        expected_status: String,
    },

    /// A citation row referenced by row_id does not belong to the specified job.
    #[error("row {row_id} does not belong to job '{job_id}'")]
    RowJobMismatch { row_id: i64, job_id: String },

    /// No pending batches remain for claiming.
    #[error("no pending batches available for job '{job_id}'")]
    NoPendingBatches { job_id: String },

    /// JSON serialization or deserialization failed for result_json or
    /// params_json content.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// CSV writing or formatting error during annotation CSV generation.
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// File I/O error during export (writing CSV, JSON, or report files).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The session_id referenced in the job parameters does not exist
    /// in the database.
    #[error("session {session_id} not found")]
    SessionNotFound { session_id: i64 },
}
