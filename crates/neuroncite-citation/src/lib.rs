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

//! neuroncite-citation: LaTeX citation verification pipeline.
//!
//! This crate provides the deterministic backend for verifying citations in
//! LaTeX documents against indexed PDFs in a NeuronCite session. The pipeline
//! consists of six stages:
//!
//! 1. **LaTeX parsing** (`latex`) -- Extracts all citation commands (`\cite`,
//!    `\citep`, `\citet`, `\autocite`, `\parencite`, `\textcite`) with line
//!    numbers, anchor words, section headings, and cite-group assignments.
//!
//! 2. **BibTeX parsing** (`bibtex`) -- Resolves cite-keys to author, title,
//!    year, abstract, and keywords from a .bib file.
//!
//! 3. **Batch assignment** (`batch`) -- Groups nearby citations from the same
//!    section into batches of configurable size, respecting cite-group
//!    boundaries so `\cite{a,b,c}` keys are never split across batches.
//!
//! 4. **Database operations** (`db`) -- CRUD operations for citation_row
//!    records: insertion, atomic batch claiming with stale timeout recovery,
//!    result submission, and status/verdict aggregation queries.
//!
//! 5. **Export** (`export`) -- Generates annotation CSV (for the neuroncite-annotate
//!    pipeline), corrections JSON (sorted by line number), and a summary
//!    report with statistics and alerts.
//!
//! Both the MCP handler layer (neuroncite-mcp) and the REST handler layer
//! (neuroncite-api) call functions in this crate. The transport-specific
//! code in those crates only handles parameter extraction and response
//! formatting -- all business logic lives here.

#![forbid(unsafe_code)]
// `deny(warnings)` is inherited from [workspace.lints.rust] in the root Cargo.toml.
#![allow(
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_panics_doc,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

pub mod batch;
pub mod bibtex;
pub mod db;
pub mod error;
pub mod export;
pub mod latex;
pub mod types;

// Re-export primary types at crate root for ergonomic imports.
pub use error::CitationError;
pub use types::{
    Alert, BatchClaim, BatchCounts, BibEntry, CitationJobParams, CitationRow, CitationRowInsert,
    CoCitationInfo, CorrectionType, EXPORT_COLUMNS, ExportResult, ExportSummary, LatexCorrection,
    OtherSourceEntry, OtherSourcePassage, PassageLocation, PassageRef, RawCitation, StatusCounts,
    StatusReport, SubmitEntry, Verdict,
};
