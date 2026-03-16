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

//! neuroncite-annotate: PDF annotation crate for highlighting quoted text
//! passages and adding comment annotations to PDF documents.
//!
//! Operates standalone without requiring an indexed session. Takes CSV or JSON
//! input with title, author, and quote fields, matches PDFs by filename using
//! Jaro-Winkler similarity, locates text via a 4-stage pipeline (exact,
//! normalized, fuzzy, OCR), and creates pdfium highlight annotations.
//!
//! The crate exposes three public functions:
//!   - annotate_pdfs(): runs the full pipeline (resolve -> locate -> annotate -> verify -> save)
//!   - parse_input(): parses and validates CSV/JSON input data
//!   - detect_format(): inspects content to determine CSV vs JSON format

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

mod annotate;
pub mod appearance;
pub mod error;
mod input;
pub mod inspect;
mod locate;
mod pipeline;
pub mod remove;
pub mod report;
pub mod resolve;
pub mod types;
pub mod verify;

// Re-export the primary public API at crate root.
pub use appearance::{inject_appearance_streams, merge_prior_annotations};
pub use input::{detect_format, parse_input};
pub use inspect::{AnnotationDetail, InspectionResult, inspect_pdf, inspect_pdf_with_pdfium};
pub use pipeline::{annotate_pdfs, annotate_pdfs_with_cancel};
pub use remove::{RemoveMode, RemoveResult, remove_highlights};
pub use types::{AnnotateConfig, InputFormat, InputRow};
