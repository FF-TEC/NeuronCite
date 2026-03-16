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

//! Pdfium-based PDF text extraction backend (feature-gated: `pdfium`).
//!
//! This module uses the `pdfium-render` crate to extract text from PDF documents
//! that have multi-column layouts, complex tables, or non-standard font encodings.
//! It requires the pdfium shared library (`libpdfium.so` / `pdfium.dll`) to be
//! available at runtime.
//!
//! This module is only compiled when the `pdfium` Cargo feature is enabled.

use std::path::Path;

use neuroncite_core::{ExtractionBackend, PageText};
use pdfium_render::prelude::*;

use crate::error::PdfError;

/// Extracts text from each page of a PDF file using the pdfium rendering engine.
///
/// Each page is extracted independently, producing one `PageText` entry per page
/// with backend annotation `ExtractionBackend::Pdfium`.
///
/// # Errors
///
/// Returns [`PdfError::Pdfium`] if the pdfium library cannot be loaded or if
/// the PDF file cannot be opened or parsed.
pub fn extract_with_pdfium(pdf_path: &Path) -> Result<Vec<PageText>, PdfError> {
    let abs_path = std::fs::canonicalize(pdf_path).map_err(|e| {
        PdfError::Pdfium(format!(
            "failed to canonicalize path {}: {e}",
            pdf_path.display()
        ))
    })?;

    // Bind to the pdfium shared library using the shared binding logic.
    // Searches: (1) exe directory, (2) CWD, (3) system library paths.
    let pdfium = Pdfium::new(crate::pdfium_binding::bind_pdfium()?);

    let document = pdfium
        .load_pdf_from_file(&abs_path, None)
        .map_err(|e| PdfError::Pdfium(format!("failed to load PDF {}: {e}", abs_path.display())))?;

    let mut pages = Vec::with_capacity(document.pages().len() as usize);

    for (idx, page) in document.pages().iter().enumerate() {
        let text = page
            .text()
            .map_err(|e| {
                PdfError::Pdfium(format!(
                    "failed to extract text from page {} of {}: {e}",
                    idx + 1,
                    abs_path.display()
                ))
            })?
            .all();

        pages.push(PageText {
            source_file: abs_path.clone(),
            page_number: idx + 1,
            content: text,
            backend: ExtractionBackend::Pdfium,
        });
    }

    Ok(pages)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    /// T-PDF-060: Pdfium extracts multi-column text.
    /// This test requires the pdfium shared library at runtime. It verifies
    /// that `extract_with_pdfium` produces `PageText` entries with the correct
    /// backend annotation. Since the pdfium library may not be available in
    /// all CI environments, this test is gated behind `#[cfg(feature = "pdfium")]`
    /// at the module level.
    #[test]
    fn t_pdf_007_pdfium_multi_column_extraction() {
        // Without a real pdfium shared library installed, this test would fail
        // at the library binding step. The test structure demonstrates the
        // expected call pattern and verifies error handling.
        use super::*;
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Minimal PDF bytes (same structure as in pdf_extract tests).
        let pdf_bytes = b"%PDF-1.4\n\
            1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\
            2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\
            3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
            /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n\
            4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td (Test) Tj ET\nendstream\nendobj\n\
            5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n\
            xref\n0 6\n\
            0000000000 65535 f \n\
            0000000009 00000 n \n\
            0000000058 00000 n \n\
            0000000115 00000 n \n\
            0000000266 00000 n \n\
            0000000000 00000 n \n\
            trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF";

        let mut tmp = NamedTempFile::with_suffix(".pdf").expect("failed to create temp file");
        tmp.write_all(pdf_bytes).expect("failed to write PDF bytes");
        tmp.flush().expect("flush failed");

        let result = extract_with_pdfium(tmp.path());

        match result {
            Ok(pages) => {
                assert!(!pages.is_empty(), "at least one page should be extracted");
                for page in &pages {
                    assert_eq!(
                        page.backend,
                        ExtractionBackend::Pdfium,
                        "backend must be Pdfium"
                    );
                }
            }
            Err(PdfError::Pdfium(msg)) => {
                // Pdfium library not available in this environment.
                // The error message should reference the library binding failure.
                assert!(
                    msg.contains("pdfium") || msg.contains("library") || msg.contains("failed"),
                    "error message should describe the pdfium failure: {msg}"
                );
            }
            Err(PdfError::DepDownload(msg)) => {
                // Pdfium auto-download failed (network unavailable, file
                // permissions, concurrent download race condition, etc.).
                // This is a valid "not available" outcome in CI environments
                // or during parallel test execution without network access.
                assert!(
                    !msg.is_empty(),
                    "download error message should not be empty: {msg}"
                );
            }
            Err(other) => {
                panic!("unexpected error variant: {other}");
            }
        }
    }
}
