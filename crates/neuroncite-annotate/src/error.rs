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

// Error types for the PDF annotation pipeline.
//
// Each variant corresponds to a distinct failure category in the annotation
// workflow: input parsing, PDF discovery, filename matching, text location,
// annotation creation, verification, and I/O operations.

use thiserror::Error;

/// Top-level error type for the annotation pipeline. Returned by the
/// public `annotate_pdfs` function and propagated through all pipeline
/// stages.
#[derive(Debug, Error)]
pub enum AnnotateError {
    /// The input data (CSV or JSON) could not be parsed or validated.
    /// The message includes the row number or field name that caused
    /// the failure.
    #[error("input parsing failed: {0}")]
    InputParse(String),

    /// PDF file discovery in the source directory failed. The message
    /// contains the underlying I/O error or permission issue.
    #[error("PDF discovery failed: {0}")]
    PdfDiscovery(String),

    /// No PDF file in the source directory matched the given title and
    /// author combination from the input data.
    #[error("no PDF matched title={title}, author={author}")]
    PdfNotMatched { title: String, author: String },

    /// The PDF file could not be opened or loaded by pdfium. The message
    /// includes the file path and the pdfium error description.
    #[error("PDF loading failed for {path}: {reason}")]
    PdfLoad { path: String, reason: String },

    /// The quoted text could not be located in the PDF after attempting
    /// all pipeline stages (exact, normalized, fuzzy, fallback extraction, OCR).
    /// The `stages_tried` field contains diagnostic messages for each stage
    /// that was attempted and the reason it did not produce a match.
    #[error("quote not found in PDF after all location stages: {quote_excerpt}")]
    QuoteNotFound {
        quote_excerpt: String,
        /// Diagnostic messages for each pipeline stage that was attempted.
        /// Each entry describes the stage name and why it failed. Example:
        /// `["exact: no match on 21 pages", "fuzzy: best=42% on page 7 < 90% threshold"]`
        stages_tried: Vec<String>,
    },

    /// Annotation creation on the PDF page failed. The message includes
    /// the pdfium error or coordinate issue.
    #[error("annotation creation failed: {0}")]
    AnnotationFailed(String),

    /// The annotated PDF could not be saved to the output directory.
    #[error("PDF save failed for {path}: {reason}")]
    SaveFailed { path: String, reason: String },

    /// Pre-check or post-check verification detected a mismatch between
    /// expected and actual annotation coverage.
    #[error("verification failed: {0}")]
    Verification(String),

    /// File system I/O error (directory creation, file copy, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error from the pdfium library binding or rendering layer.
    #[error("pdfium error: {0}")]
    Pdfium(String),

    /// Error from the OCR subsystem (Tesseract invocation or hOCR parsing).
    #[error("OCR error: {0}")]
    Ocr(String),

    /// The annotation pipeline was canceled via cooperative cancellation.
    /// This variant is returned when the caller-supplied cancel callback
    /// returns `true` before processing the next PDF in the pipeline loop.
    /// The pipeline stops cleanly without leaving partial output files.
    #[error("annotation canceled: processed {completed} of {total} PDFs before cancellation")]
    Canceled {
        /// Number of PDFs that were fully processed before cancellation.
        completed: usize,
        /// Total number of PDFs that were scheduled for processing.
        total: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-ANNOTATE-ERR-001: QuoteNotFound error message includes the quote
    /// excerpt but not the stages_tried data (stages_tried is a structured
    /// field, not part of the Display format).
    #[test]
    fn t_annotate_err_001_quote_not_found_display() {
        let err = AnnotateError::QuoteNotFound {
            quote_excerpt: "test quote".into(),
            stages_tried: vec![
                "exact: no match on 5 pages".into(),
                "normalized: no match on 5 pages".into(),
            ],
        };

        let msg = format!("{err}");
        assert!(
            msg.contains("test quote"),
            "error Display must contain the quote excerpt"
        );
        assert!(
            msg.contains("all location stages"),
            "error Display must indicate all stages were tried"
        );
    }

    /// T-ANNOTATE-ERR-002: QuoteNotFound error carries stages_tried data
    /// accessible via pattern matching.
    #[test]
    fn t_annotate_err_002_quote_not_found_stages_extraction() {
        let stages = vec![
            "exact: no match on 10 pages".to_string(),
            "fuzzy: best=85% on page 3 < 90% threshold".to_string(),
        ];

        let err = AnnotateError::QuoteNotFound {
            quote_excerpt: "some quote".into(),
            stages_tried: stages.clone(),
        };

        match err {
            AnnotateError::QuoteNotFound {
                stages_tried,
                quote_excerpt,
            } => {
                assert_eq!(quote_excerpt, "some quote");
                assert_eq!(stages_tried.len(), 2);
                assert_eq!(stages_tried, stages);
            }
            _ => panic!("expected QuoteNotFound variant"),
        }
    }

    /// T-ANNOTATE-ERR-003: QuoteNotFound with empty stages_tried (for edge
    /// cases like empty quotes where no stages are attempted).
    #[test]
    fn t_annotate_err_003_quote_not_found_empty_stages() {
        let err = AnnotateError::QuoteNotFound {
            quote_excerpt: "(empty quote)".into(),
            stages_tried: vec![],
        };

        match err {
            AnnotateError::QuoteNotFound { stages_tried, .. } => {
                assert!(
                    stages_tried.is_empty(),
                    "empty quote must have empty stages_tried"
                );
            }
            _ => panic!("expected QuoteNotFound variant"),
        }
    }
}
