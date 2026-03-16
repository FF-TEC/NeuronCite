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

//! neuroncite-pdf: PDF file discovery and page-by-page text extraction.
//!
//! This crate provides three major capabilities:
//!
//! 1. **Discovery** -- Collects paths to indexable files in a directory tree.
//!    Three modes are available: `discover_pdfs` walks the directory tree
//!    recursively for PDF files only, `discover_pdfs_flat` scans the top-level
//!    directory only (used by the MCP/API indexer to avoid capturing
//!    subdirectories), and `discover_documents` walks recursively for all
//!    supported file types (PDF, HTML, HTM).
//! 2. **Extraction** -- Extracts text from each page of a PDF file using one of
//!    two backends: the default `pdf-extract` (pure Rust) or the optional
//!    `pdfium-render` backend (requires the pdfium shared library at runtime).
//! 3. **OCR fallback** -- When the `ocr` feature is enabled, pages whose
//!    extracted text falls below a quality threshold are rendered to raster
//!    images and passed through Tesseract OCR via CLI subprocess invocation.
//!
//! The crate does not use `#![forbid(unsafe_code)]` because the optional pdfium
//! FFI backend may require unsafe blocks in its implementation.

// `deny(warnings)` is inherited from [workspace.lints.rust] in the root Cargo.toml.

mod discovery;
mod error;
mod extract;
pub mod quality;

// Download logic for optional runtime dependencies (pdfium shared library,
// Tesseract binary, tessdata language packs). Always compiled (no feature gate)
// so the GUI crate can call the public download and probe functions regardless
// of which features are enabled. The module's functions only use std library
// types and external process invocations, so they have zero dependency on
// feature-gated external crates.
pub mod deps;

// Shared pdfium library binding logic. Both the pdfium extraction backend
// and the OCR module use this to locate the pdfium shared library at runtime.
// Only compiled when the pdfium feature is enabled.
#[cfg(feature = "pdfium")]
pub mod pdfium_binding;

// The ocr module is public to allow integration tests to directly invoke
// `ocr_page()` and `ocr_pdf_pages_batch()`. Its contents are conditionally
// compiled based on the "ocr" feature flag. When the feature is disabled,
// the module file contains only documentation comments and no compiled code.
pub mod ocr;

pub use discovery::discover_documents;
pub use discovery::discover_pdfs;
pub use discovery::discover_pdfs_flat;
pub use discovery::file_type_for_path;
pub use discovery::has_html_extension;
pub use discovery::has_indexable_extension;
pub use discovery::has_pdf_extension;
pub use error::PdfError;
pub use extract::ExtractionResult;
pub use extract::extract_pages;

// Re-export OCR types and batch functions at the crate root for convenience.
#[cfg(feature = "ocr")]
pub use ocr::HocrWord;
#[cfg(feature = "ocr")]
pub use ocr::TesseractPaths;
#[cfg(feature = "ocr")]
pub use ocr::ocr_page_with_hocr;
#[cfg(feature = "ocr")]
pub use ocr::ocr_pdf_pages_batch;
#[cfg(feature = "ocr")]
pub use ocr::ocr_pdf_pages_batch_hocr;
#[cfg(feature = "ocr")]
pub use ocr::render_page_to_png_with_dpi;
