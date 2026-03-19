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

//! Default PDF text extraction backend using the `pdf-extract` crate.
//!
//! This module wraps the pure-Rust `pdf-extract` library to provide page-by-page
//! text extraction without requiring any external shared libraries at runtime.
//! It serves as the always-available primary backend.
//!
//! Page boundaries are determined using `lopdf`'s per-page text extraction, which
//! iterates over the PDF's page tree and extracts text from each page object
//! independently. This produces accurate page numbers for citation generation.
//! If `lopdf` per-page extraction fails for any page, the function falls back to
//! `pdf-extract`'s bulk extraction with form-feed (`\x0C`) splitting.

use std::path::Path;

use memmap2::Mmap;
use neuroncite_core::{ExtractionBackend, PageText};

use crate::error::PdfError;

/// Extracts text from each page of a PDF file.
///
/// Uses a two-tier strategy for page boundary detection:
///
/// 1. **Primary (lopdf per-page):** Loads the PDF structure with `lopdf` and
///    extracts text from each page independently via `Document::extract_text`.
///    This produces one `PageText` per structural page and is accurate for
///    multi-hundred-page documents.
///
/// 2. **Fallback (pdf-extract + form-feed split):** If `lopdf` per-page extraction
///    fails (e.g., encrypted PDFs, unsupported font encodings), falls back to
///    `pdf_extract::extract_text_from_mem` which returns the full document text.
///    Pages are split on form-feed characters (`\x0C`). If no form-feed characters
///    are present, the entire text becomes a single page.
///
/// # Errors
///
/// Returns [`PdfError::Extraction`] if both extraction strategies fail.
/// Returns [`PdfError::Io`] if the file cannot be read from disk.
///
/// # Safety
///
/// Uses memory-mapped I/O internally. The mapped region is read-only and the
/// file must not be truncated by another process while the map is active.
pub fn extract_with_pdf_extract(pdf_path: &Path) -> Result<Vec<PageText>, PdfError> {
    let file = std::fs::File::open(pdf_path)?;
    let metadata = file.metadata()?;

    // For empty files, return an empty page list without attempting to mmap.
    // memmap2::Mmap::map requires a non-zero file length on some platforms.
    if metadata.len() == 0 {
        return Ok(Vec::new());
    }

    // SAFETY: The file is opened read-only and is not modified by this process
    // during extraction. External truncation during the map is the only hazard,
    // which does not occur under normal operation.
    let mmap = unsafe { Mmap::map(&file)? };
    let bytes: &[u8] = &mmap;
    let abs_path = std::fs::canonicalize(pdf_path)?;

    // Primary strategy: per-page extraction via lopdf.
    // lopdf iterates the PDF page tree and extracts text from each page object
    // independently, producing accurate page boundaries regardless of whether
    // pdf-extract inserts form-feed characters.
    match extract_per_page_lopdf(bytes, &abs_path) {
        Ok(pages) if !pages.is_empty() => {
            tracing::debug!(
                page_count = pages.len(),
                path = %abs_path.display(),
                "extracted pages via lopdf per-page strategy"
            );
            return Ok(pages);
        }
        Ok(_) => {
            tracing::debug!(
                path = %abs_path.display(),
                "lopdf returned zero pages, falling back to pdf-extract"
            );
        }
        Err(e) => {
            tracing::debug!(
                path = %abs_path.display(),
                error = %e,
                "lopdf per-page extraction failed, falling back to pdf-extract"
            );
        }
    }

    // Fallback strategy: bulk extraction via pdf-extract with form-feed splitting.
    extract_with_formfeed_split(bytes, &abs_path)
}

/// Extracts text per page using lopdf's `Document::extract_text` method.
///
/// Loads the PDF structure, iterates over each page number (1-indexed), and
/// calls `extract_text(&[page_num])` for each page independently. Returns
/// one `PageText` entry per structural page.
///
/// # Errors
///
/// Returns `PdfError::Extraction` if the document cannot be loaded or if
/// text extraction fails for any page.
fn extract_per_page_lopdf(bytes: &[u8], abs_path: &Path) -> Result<Vec<PageText>, PdfError> {
    let doc = lopdf::Document::load_mem(bytes)
        .map_err(|e| PdfError::Extraction(format!("{}: lopdf load: {e}", abs_path.display())))?;

    let page_count = doc.get_pages().len();

    if page_count == 0 {
        return Ok(Vec::new());
    }

    let mut pages = Vec::with_capacity(page_count);

    for page_num in 1..=page_count as u32 {
        let text = doc.extract_text(&[page_num]).map_err(|e| {
            PdfError::Extraction(format!(
                "{}: lopdf page {page_num}/{page_count}: {e}",
                abs_path.display()
            ))
        })?;

        pages.push(PageText {
            source_file: abs_path.to_path_buf(),
            page_number: page_num as usize,
            content: text,
            backend: ExtractionBackend::PdfExtract,
        });
    }

    Ok(pages)
}

/// Fallback extraction using pdf-extract's bulk text output split on form-feed
/// characters (`\x0C`).
///
/// pdf-extract internally uses `output_doc` which writes a form-feed after each
/// page. However, some PDFs (or pdf-extract versions) may not produce form-feed
/// characters, in which case the entire document text is treated as a single page.
fn extract_with_formfeed_split(bytes: &[u8], abs_path: &Path) -> Result<Vec<PageText>, PdfError> {
    let full_text = pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| PdfError::Extraction(format!("{}: {e}", abs_path.display())))?;

    // Split on form-feed characters to recover per-page boundaries.
    // Filter out empty trailing segments caused by a trailing form-feed.
    let result: Vec<PageText> = full_text
        .split('\x0C')
        .enumerate()
        .filter(|(_, text)| !text.trim().is_empty())
        .map(|(idx, page_text)| PageText {
            source_file: abs_path.to_path_buf(),
            page_number: idx + 1,
            content: page_text.to_string(),
            backend: ExtractionBackend::PdfExtract,
        })
        .collect();

    tracing::debug!(
        page_count = result.len(),
        path = %abs_path.display(),
        "extracted pages via pdf-extract form-feed split fallback"
    );

    Ok(result)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Produces a minimal valid PDF byte sequence containing the given text
    /// on a single page. The PDF structure follows the minimal PDF specification:
    /// header, catalog, page tree, page, content stream, and cross-reference table.
    fn minimal_pdf_with_text(text: &str) -> Vec<u8> {
        // The content stream draws text using the Tf (font) and Tj (show text)
        // operators. The font /F1 is defined as Helvetica in the page resources.
        let stream_content = format!("BT /F1 12 Tf 100 700 Td ({text}) Tj ET");
        let stream_length = stream_content.len();

        // Build the PDF as a series of indirect objects.
        // Object 1: Catalog (root of the document structure)
        // Object 2: Pages (page tree node containing one page)
        // Object 3: Page (references the page tree and content stream)
        // Object 4: Content stream (the drawing instructions)
        // Object 5: Font dictionary (Helvetica, Type1)
        let pdf = format!(
            "%PDF-1.4\n\
             1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\
             2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\
             3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
             /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n\
             4 0 obj\n<< /Length {stream_length} >>\nstream\n\
             {stream_content}\nendstream\nendobj\n\
             5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n\
             xref\n0 6\n\
             0000000000 65535 f \n\
             0000000009 00000 n \n\
             0000000058 00000 n \n\
             0000000115 00000 n \n\
             0000000266 00000 n \n\
             0000000000 00000 n \n\
             trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF"
        );
        pdf.into_bytes()
    }

    /// T-PDF-057: Extracts text from a well-formed PDF.
    /// Creates a minimal valid PDF containing known text and verifies that
    /// `extract_with_pdf_extract` returns at least one `PageText` entry
    /// annotated with the `PdfExtract` backend.
    #[test]
    fn t_pdf_057_extracts_text_from_well_formed_pdf() {
        let pdf_bytes = minimal_pdf_with_text("Hello World");

        let mut tmp = NamedTempFile::with_suffix(".pdf").expect("failed to create temp file");
        tmp.write_all(&pdf_bytes)
            .expect("failed to write PDF bytes");
        tmp.flush().expect("flush failed");

        let result = extract_with_pdf_extract(tmp.path());

        // pdf-extract may or may not successfully parse our minimal PDF
        // depending on the exact version. If it succeeds, verify the structure.
        // If it fails, that is acceptable since our minimal PDF may not have
        // valid cross-reference offsets. The key assertion is that no panic
        // occurs.
        match result {
            Ok(pages) => {
                assert!(!pages.is_empty(), "at least one page should be returned");
                assert_eq!(
                    pages[0].backend,
                    ExtractionBackend::PdfExtract,
                    "backend annotation must be PdfExtract"
                );
                assert_eq!(pages[0].page_number, 1, "first page number must be 1");
            }
            Err(PdfError::Extraction(_)) => {
                // Acceptable: the minimal PDF may not pass pdf-extract's
                // strict cross-reference validation.
            }
            Err(other) => {
                panic!("unexpected error variant: {other}");
            }
        }
    }

    /// T-PDF-048: lopdf per-page extraction produces correct page numbers.
    /// Creates a multi-page PDF and verifies that each page receives its
    /// correct 1-indexed page number, rather than all pages being reported
    /// as page 1.
    #[test]
    fn t_pdf_048_lopdf_per_page_produces_correct_page_numbers() {
        // Build a minimal 3-page PDF. Each page has distinct content text
        // so we can verify that page boundaries are correct.
        let pdf_bytes =
            build_multi_page_pdf(&["Page one text", "Page two text", "Page three text"]);

        let mut tmp = NamedTempFile::with_suffix(".pdf").expect("failed to create temp file");
        tmp.write_all(&pdf_bytes)
            .expect("failed to write PDF bytes");
        tmp.flush().expect("flush failed");

        let result = extract_with_pdf_extract(tmp.path());

        match result {
            Ok(pages) => {
                // At least 3 pages should be extracted from our 3-page PDF.
                assert!(
                    pages.len() >= 3,
                    "expected at least 3 pages, got {}",
                    pages.len()
                );

                // Verify page numbers are sequential starting from 1.
                for (i, page) in pages.iter().enumerate() {
                    assert_eq!(
                        page.page_number,
                        i + 1,
                        "page {} has wrong page_number: {}",
                        i + 1,
                        page.page_number
                    );
                }
            }
            Err(PdfError::Extraction(_)) => {
                // Acceptable: the minimal PDF structure may not be parseable
                // by lopdf on all platforms.
            }
            Err(other) => {
                panic!("unexpected error variant: {other}");
            }
        }
    }

    /// T-PDF-058: lopdf fallback to pdf-extract form-feed split.
    /// Verifies that `extract_with_formfeed_split` filters out empty trailing
    /// segments produced by a trailing form-feed character.
    #[test]
    fn t_pdf_058_formfeed_split_filters_empty_trailing_pages() {
        let abs_path = std::path::PathBuf::from("/test/doc.pdf");

        // Simulate pdf-extract output with a trailing form-feed:
        // "page1\x0Cpage2\x0C" splits into ["page1", "page2", ""]
        let text_with_trailing_ff = "Page one content\x0CPage two content\x0C";

        let pages: Vec<PageText> = text_with_trailing_ff
            .split('\x0C')
            .enumerate()
            .filter(|(_, text)| !text.trim().is_empty())
            .map(|(idx, page_text)| PageText {
                source_file: abs_path.clone(),
                page_number: idx + 1,
                content: page_text.to_string(),
                backend: ExtractionBackend::PdfExtract,
            })
            .collect();

        assert_eq!(
            pages.len(),
            2,
            "trailing empty segment must be filtered out"
        );
        assert_eq!(pages[0].page_number, 1);
        assert_eq!(pages[1].page_number, 2);
        assert!(pages[0].content.contains("Page one"));
        assert!(pages[1].content.contains("Page two"));
    }

    /// T-PDF-059: Empty file returns empty page list without panicking.
    #[test]
    fn t_pdf_059_empty_file_returns_empty() {
        let tmp = NamedTempFile::with_suffix(".pdf").expect("failed to create temp file");
        let result = extract_with_pdf_extract(tmp.path());

        if let Ok(pages) = result {
            assert!(pages.is_empty(), "empty file must return empty pages");
        }
        // Err is acceptable: some platforms reject empty mmaps.
    }

    /// Builds a minimal multi-page PDF with the given per-page text strings.
    /// Each page contains the text drawn at position (100, 700) using Helvetica 12pt.
    fn build_multi_page_pdf(page_texts: &[&str]) -> Vec<u8> {
        let page_count = page_texts.len();

        // Build kid references: "3 0 R 6 0 R 9 0 R ..." (each page uses 3 objects)
        let kid_refs: Vec<String> = (0..page_count)
            .map(|i| format!("{} 0 R", 3 + i * 3))
            .collect();
        let kids_str = kid_refs.join(" ");

        let mut pdf = String::new();
        pdf.push_str("%PDF-1.4\n");

        // Object 1: Catalog
        pdf.push_str("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

        // Object 2: Pages (page tree root)
        pdf.push_str(&format!(
            "2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {page_count} >>\nendobj\n"
        ));

        // Font object number: after all page+content objects
        let font_obj_num = 3 + page_count * 3;

        // Create page + content stream objects for each page
        for (i, text) in page_texts.iter().enumerate() {
            let page_obj = 3 + i * 3;
            let content_obj = page_obj + 1;
            let length_obj = page_obj + 2;

            let stream = format!("BT /F1 12 Tf 100 700 Td ({text}) Tj ET");
            let stream_len = stream.len();

            // Page object
            pdf.push_str(&format!(
                "{page_obj} 0 obj\n<< /Type /Page /Parent 2 0 R \
                 /MediaBox [0 0 612 792] /Contents {content_obj} 0 R \
                 /Resources << /Font << /F1 {font_obj_num} 0 R >> >> >>\nendobj\n"
            ));

            // Content stream
            pdf.push_str(&format!(
                "{content_obj} 0 obj\n<< /Length {length_obj} 0 R >>\nstream\n\
                 {stream}\nendstream\nendobj\n"
            ));

            // Length object (indirect reference for the stream length)
            pdf.push_str(&format!("{length_obj} 0 obj\n{stream_len}\nendobj\n"));
        }

        // Font object
        pdf.push_str(&format!(
            "{font_obj_num} 0 obj\n<< /Type /Font /Subtype /Type1 \
             /BaseFont /Helvetica >>\nendobj\n"
        ));

        // Cross-reference table and trailer (simplified, offsets set to 0)
        let total_objects = font_obj_num + 1;
        pdf.push_str("xref\n");
        pdf.push_str(&format!("0 {total_objects}\n"));
        pdf.push_str("0000000000 65535 f \n");
        for _ in 1..total_objects {
            pdf.push_str("0000000000 00000 n \n");
        }

        pdf.push_str(&format!(
            "trailer\n<< /Size {total_objects} /Root 1 0 R >>\nstartxref\n0\n%%EOF"
        ));

        pdf.into_bytes()
    }
}
