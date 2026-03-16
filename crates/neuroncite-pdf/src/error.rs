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

//! Error types for the neuroncite-pdf crate.
//!
//! `PdfError` is the single error enum returned by all public functions in this
//! crate. It covers I/O failures during directory traversal, text extraction
//! failures from the PDF backends, OCR failures, and special conditions like
//! password-protected or empty documents.

use neuroncite_core::error::NeuronCiteError;

/// Represents all error conditions that can occur during PDF discovery,
/// text extraction, and OCR processing within this crate.
#[derive(Debug, thiserror::Error)]
pub enum PdfError {
    /// An I/O error occurred while traversing the file system or reading a PDF
    /// file from disk. The `#[from]` attribute enables automatic conversion
    /// from `std::io::Error` via the `?` operator.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The pdf-extract backend failed to parse or extract text from a PDF file.
    /// The string contains a human-readable description of the extraction failure.
    #[error("PDF extraction failed: {0}")]
    Extraction(String),

    /// The pdfium backend failed during rendering or text extraction.
    /// This variant is reachable only when the `pdfium` feature is enabled.
    #[error("Pdfium backend error: {0}")]
    Pdfium(String),

    /// Tesseract OCR processing failed for a rendered page image.
    /// This variant is reachable only when the `ocr` feature is enabled.
    #[error("OCR failed: {0}")]
    Ocr(String),

    /// Auto-download of a runtime dependency (pdfium shared library, Tesseract
    /// binary, or tessdata language pack) failed. The string contains the URL,
    /// platform, or filesystem error details.
    #[error("dependency download failed: {0}")]
    DepDownload(String),

    /// An error occurred during recursive directory traversal for PDF discovery.
    /// Distinct from `Io` to allow callers to distinguish discovery-phase errors
    /// from extraction-phase I/O errors.
    #[error("Discovery error: {0}")]
    Discovery(String),

    /// The PDF file is encrypted with a password and cannot be opened without
    /// authentication credentials.
    #[error("Password-protected PDF: {0}")]
    PasswordProtected(String),

    /// The PDF file contains zero pages or zero extractable content streams.
    #[error("Empty document: {0}")]
    EmptyDocument(String),
}

/// Converts a `PdfError` into the workspace-wide `NeuronCiteError` by
/// mapping it to the `NeuronCiteError::Pdf` variant. The error message
/// is preserved via `Display` formatting. This impl lives in neuroncite-pdf
/// (rather than neuroncite-core) because neuroncite-core is a leaf crate
/// with no workspace dependencies and cannot reference `PdfError`.
impl From<PdfError> for NeuronCiteError {
    fn from(err: PdfError) -> Self {
        NeuronCiteError::Pdf(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuroncite_core::error::NeuronCiteError;

    // -----------------------------------------------------------------------
    // #49 -- From<PdfError> for NeuronCiteError conversion tests
    //
    // The From impl converts every PdfError variant into
    // NeuronCiteError::Pdf(String) by calling .to_string() on the PdfError.
    // These tests verify that each variant is routed to the Pdf arm of
    // NeuronCiteError and that the error message is preserved in the string.
    // -----------------------------------------------------------------------

    /// Validates that `PdfError::Io` converts to `NeuronCiteError::Pdf` with
    /// the I/O error message preserved. The Io variant wraps std::io::Error
    /// via #[from], so we construct it from a standard I/O error.
    #[test]
    fn t_err_001_io_converts_to_neuroncite_pdf() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let pdf_err = PdfError::Io(io_err);
        let core_err: NeuronCiteError = pdf_err.into();

        match core_err {
            NeuronCiteError::Pdf(msg) => {
                assert!(
                    msg.contains("file missing"),
                    "message must contain the original I/O error text, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Pdf, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `PdfError::Extraction` converts to `NeuronCiteError::Pdf`
    /// with the extraction failure message preserved. This variant represents
    /// failures in the pdf-extract backend during text extraction.
    #[test]
    fn t_err_002_extraction_converts_to_neuroncite_pdf() {
        let pdf_err = PdfError::Extraction("invalid cross-reference table".into());
        let core_err: NeuronCiteError = pdf_err.into();

        match core_err {
            NeuronCiteError::Pdf(msg) => {
                assert!(
                    msg.contains("invalid cross-reference table"),
                    "message must contain the extraction error, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Pdf, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `PdfError::Pdfium` converts to `NeuronCiteError::Pdf`
    /// with the pdfium backend error message preserved. This variant is
    /// reachable when the pdfium feature flag is active.
    #[test]
    fn t_err_003_pdfium_converts_to_neuroncite_pdf() {
        let pdf_err = PdfError::Pdfium("failed to load shared library".into());
        let core_err: NeuronCiteError = pdf_err.into();

        match core_err {
            NeuronCiteError::Pdf(msg) => {
                assert!(
                    msg.contains("failed to load shared library"),
                    "message must contain the pdfium error, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Pdf, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `PdfError::Ocr` converts to `NeuronCiteError::Pdf` with
    /// the OCR failure message preserved. This variant is reachable when the
    /// ocr feature flag is active.
    #[test]
    fn t_err_004_ocr_converts_to_neuroncite_pdf() {
        let pdf_err = PdfError::Ocr("tesseract not found on PATH".into());
        let core_err: NeuronCiteError = pdf_err.into();

        match core_err {
            NeuronCiteError::Pdf(msg) => {
                assert!(
                    msg.contains("tesseract not found on PATH"),
                    "message must contain the OCR error, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Pdf, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `PdfError::DepDownload` converts to `NeuronCiteError::Pdf`
    /// with the download failure message preserved. This variant covers failed
    /// auto-downloads of pdfium, tesseract, or tessdata language packs.
    #[test]
    fn t_err_005_dep_download_converts_to_neuroncite_pdf() {
        let pdf_err = PdfError::DepDownload("HTTP 404 for pdfium binary".into());
        let core_err: NeuronCiteError = pdf_err.into();

        match core_err {
            NeuronCiteError::Pdf(msg) => {
                assert!(
                    msg.contains("HTTP 404 for pdfium binary"),
                    "message must contain the download error, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Pdf, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `PdfError::Discovery` converts to `NeuronCiteError::Pdf`
    /// with the discovery failure message preserved. This variant is distinct
    /// from Io to allow callers to differentiate discovery-phase errors from
    /// extraction-phase I/O errors.
    #[test]
    fn t_err_006_discovery_converts_to_neuroncite_pdf() {
        let pdf_err = PdfError::Discovery("symlink loop in /data/docs".into());
        let core_err: NeuronCiteError = pdf_err.into();

        match core_err {
            NeuronCiteError::Pdf(msg) => {
                assert!(
                    msg.contains("symlink loop"),
                    "message must contain the discovery error, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Pdf, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `PdfError::PasswordProtected` converts to
    /// `NeuronCiteError::Pdf` with the filename or path preserved in the
    /// message. This variant represents encrypted PDFs that require credentials.
    #[test]
    fn t_err_007_password_protected_converts_to_neuroncite_pdf() {
        let pdf_err = PdfError::PasswordProtected("secret_report.pdf".into());
        let core_err: NeuronCiteError = pdf_err.into();

        match core_err {
            NeuronCiteError::Pdf(msg) => {
                assert!(
                    msg.contains("secret_report.pdf"),
                    "message must contain the filename, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Pdf, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that `PdfError::EmptyDocument` converts to
    /// `NeuronCiteError::Pdf` with the document identifier preserved.
    /// This variant represents PDFs with zero pages or zero extractable content.
    #[test]
    fn t_err_008_empty_document_converts_to_neuroncite_pdf() {
        let pdf_err = PdfError::EmptyDocument("blank.pdf has 0 pages".into());
        let core_err: NeuronCiteError = pdf_err.into();

        match core_err {
            NeuronCiteError::Pdf(msg) => {
                assert!(
                    msg.contains("blank.pdf has 0 pages"),
                    "message must contain the empty document description, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Pdf, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    /// Validates that the subsystem() method on the converted NeuronCiteError
    /// returns "pdf" for all PdfError variants. This confirms all variants
    /// route to the Pdf arm of NeuronCiteError, not to any other subsystem.
    #[test]
    fn t_err_009_all_variants_map_to_pdf_subsystem() {
        let variants: Vec<PdfError> = vec![
            PdfError::Io(std::io::Error::other("test")),
            PdfError::Extraction("test".into()),
            PdfError::Pdfium("test".into()),
            PdfError::Ocr("test".into()),
            PdfError::DepDownload("test".into()),
            PdfError::Discovery("test".into()),
            PdfError::PasswordProtected("test".into()),
            PdfError::EmptyDocument("test".into()),
        ];

        for variant in variants {
            let display = variant.to_string();
            let core_err: NeuronCiteError = variant.into();
            assert_eq!(
                core_err.subsystem(),
                "pdf",
                "PdfError variant with display '{display}' must map to 'pdf' subsystem"
            );
        }
    }

    // -----------------------------------------------------------------------
    // From<std::io::Error> for PdfError conversion tests
    //
    // The #[from] attribute on PdfError::Io enables automatic conversion from
    // std::io::Error. These tests verify that the conversion produces the Io
    // variant with the original error accessible.
    // -----------------------------------------------------------------------

    /// Validates that `std::io::Error` converts to `PdfError::Io` via the
    /// #[from] attribute. The original I/O error is wrapped directly (not
    /// stringified) because PdfError::Io uses #[from] rather than ToString.
    #[test]
    fn t_err_010_std_io_error_converts_to_pdf_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let pdf_err: PdfError = io_err.into();

        // Verify the Display output contains the original error message
        let display = pdf_err.to_string();
        assert!(
            display.contains("access denied"),
            "PdfError::Io display must contain the original I/O message, got: {display}"
        );
    }

    /// Validates the full conversion chain: std::io::Error -> PdfError::Io ->
    /// NeuronCiteError::Pdf. This tests that both From impls compose correctly
    /// and the original error text survives the double conversion.
    #[test]
    fn t_err_011_io_to_pdf_to_neuroncite_chain() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "broken pipe");
        let pdf_err: PdfError = io_err.into();
        let core_err: NeuronCiteError = pdf_err.into();

        match core_err {
            NeuronCiteError::Pdf(msg) => {
                assert!(
                    msg.contains("broken pipe"),
                    "chained conversion must preserve the original I/O message, got: {msg}"
                );
            }
            other => panic!(
                "expected NeuronCiteError::Pdf, got subsystem: {}",
                other.subsystem()
            ),
        }
    }

    // -----------------------------------------------------------------------
    // Display trait tests
    //
    // The thiserror #[error(...)] attributes define the Display output for
    // each variant. These tests verify the exact format strings.
    // -----------------------------------------------------------------------

    /// Validates that every PdfError variant produces a Display string that
    /// matches the format defined in the #[error(...)] attribute.
    #[test]
    fn t_err_012_display_format_all_variants() {
        let cases: Vec<(PdfError, &str)> = vec![
            (
                PdfError::Extraction("bad xref".into()),
                "PDF extraction failed: bad xref",
            ),
            (
                PdfError::Pdfium("lib missing".into()),
                "Pdfium backend error: lib missing",
            ),
            (
                PdfError::Ocr("no eng data".into()),
                "OCR failed: no eng data",
            ),
            (
                PdfError::DepDownload("timeout".into()),
                "dependency download failed: timeout",
            ),
            (PdfError::Discovery("loop".into()), "Discovery error: loop"),
            (
                PdfError::PasswordProtected("f.pdf".into()),
                "Password-protected PDF: f.pdf",
            ),
            (
                PdfError::EmptyDocument("e.pdf".into()),
                "Empty document: e.pdf",
            ),
        ];

        for (err, expected) in cases {
            assert_eq!(
                err.to_string(),
                expected,
                "Display format mismatch for PdfError variant"
            );
        }

        // The Io variant wraps std::io::Error, so its Display is "I/O error: <io message>"
        let io_err = std::io::Error::other("disk full");
        let pdf_io = PdfError::Io(io_err);
        assert_eq!(
            pdf_io.to_string(),
            "I/O error: disk full",
            "PdfError::Io display must follow the 'I/O error: <msg>' format"
        );
    }
}
