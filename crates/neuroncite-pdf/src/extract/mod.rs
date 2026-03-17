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

//! PDF text extraction backends and fallback dispatch logic.
//!
//! This module contains the extraction implementations that read raw PDF bytes
//! and produce plain-text output for each page. The default backend uses the
//! `pdf-extract` crate (pure Rust, no external dependencies). When the `pdfium`
//! feature is enabled, an alternative backend based on `pdfium-render` becomes
//! available for documents with multi-column layouts, complex tables, or
//! non-standard font encodings.
//!
//! The `extract_pages` function implements the automatic backend selection and
//! fallback chain described in the architecture document: try pdf-extract first,
//! assess quality, fall back to pdfium if quality is insufficient, and trigger
//! OCR for pages that remain below a minimum character threshold after all
//! text-based backends have been tried.

pub mod pdf_extract;

#[cfg(feature = "pdfium")]
pub mod pdfium;

use std::path::Path;

use neuroncite_core::PageText;

use crate::error::PdfError;

/// Result of a PDF text extraction operation. Contains the extracted page texts
/// and the structural page count read from the PDF's page tree. The structural
/// count is independent of how many pages produced extractable text -- it
/// reflects the total number of pages defined in the PDF file structure.
pub struct ExtractionResult {
    /// Per-page extracted text content. The vector length may differ from
    /// `pdf_page_count` when pages are blank, image-only, or filtered.
    pub pages: Vec<PageText>,
    /// Total page count from the PDF's structural page tree (lopdf or pdfium).
    /// `None` if the structural count could not be determined (e.g., when the
    /// PDF page tree is corrupted or unreadable).
    pub pdf_page_count: Option<usize>,
}

/// Minimum number of non-whitespace characters a page must contain before
/// OCR fallback is considered. Pages with fewer non-whitespace characters
/// than this threshold are assumed to be image-only or blank.
#[cfg(feature = "ocr")]
const MIN_NON_WHITESPACE_CHARS: usize = 20;

/// Extracts text from each page of a PDF file, applying backend selection
/// and quality-based fallback logic.
///
/// # Backend Selection
///
/// - If `backend_preference` is `Some("pdf-extract")`, only the pdf-extract
///   backend is used.
/// - If `backend_preference` is `Some("pdfium")` and the `pdfium` feature is
///   enabled, only the pdfium backend is used.
/// - If `backend_preference` is `None` or `Some("auto")`, the automatic
///   fallback chain is applied: pdf-extract is tried first, quality heuristics
///   are checked, and pdfium is used as fallback for pages that fail quality
///   checks.
///
/// After all text-based backends, pages with fewer than 20 non-whitespace
/// characters trigger OCR (if the `ocr` feature is enabled).
///
/// # Parameters
///
/// - `pdf_path`: Absolute path to the PDF file.
/// - `backend_preference`: Backend selection override (see above).
/// - `ocr_language`: Tesseract language code for OCR fallback. When `None`,
///   defaults to `"eng"`. Ignored when the `ocr` feature is disabled.
///
/// # Errors
///
/// Returns [`PdfError`] if the PDF file cannot be read or all backends fail.
pub fn extract_pages(
    pdf_path: &Path,
    backend_preference: Option<&str>,
    ocr_language: Option<&str>,
) -> Result<ExtractionResult, PdfError> {
    // Read the structural page count from the PDF page tree before running
    // text extraction. This count is independent of extraction success and
    // reflects the total number of pages in the PDF file structure.
    let pdf_page_count = read_structural_page_count(pdf_path);

    let mut pages = match backend_preference {
        Some("pdf-extract") => pdf_extract::extract_with_pdf_extract(pdf_path),
        #[cfg(feature = "pdfium")]
        Some("pdfium") => pdfium::extract_with_pdfium(pdf_path),
        #[cfg(not(feature = "pdfium"))]
        Some("pdfium") => Err(PdfError::Pdfium(
            "pdfium feature is not enabled in this build".to_string(),
        )),
        Some("auto") | None => auto_extract(pdf_path, ocr_language, pdf_page_count),
        Some(other) => Err(PdfError::Extraction(format!(
            "unknown backend preference: {other}"
        ))),
    }?;

    // Strip C0 control characters (U+0000..U+001F) from all page content,
    // preserving tab (0x09), newline (0x0A), and carriage return (0x0D).
    // PDF text extraction backends sometimes emit NUL bytes, form feeds,
    // vertical tabs, and other control characters that cause display
    // artifacts in search results, exports, and JSON serialization.
    for page in &mut pages {
        strip_control_characters(&mut page.content);
    }

    Ok(ExtractionResult {
        pages,
        pdf_page_count,
    })
}

/// Reads the structural page count from the PDF's page tree.
///
/// Uses pdfium as the preferred page-count source when the `pdfium` feature
/// is enabled, because pdfium handles linearized PDFs, incremental updates,
/// and XRef stream objects more reliably than lopdf. Falls back to lopdf
/// when pdfium is unavailable or fails. When both sources succeed but disagree on the
/// page count, pdfium's count is preferred and a warning is logged. This
/// guards against inconsistent pdf_page_count values across sessions that
/// arise when parallel rayon extraction causes transient pdfium binding
/// failures, forcing the lopdf fallback path.
///
/// Returns `None` if the file cannot be read or the PDF structure is invalid.
fn read_structural_page_count(pdf_path: &Path) -> Option<usize> {
    #[cfg(feature = "pdfium")]
    {
        let pdfium_count = read_structural_page_count_pdfium(pdf_path);

        // Also read via lopdf for cross-validation. When both succeed and
        // disagree, prefer pdfium and log a warning so discrepancies are
        // visible in logs. This catches PDFs where lopdf's `get_pages()`
        // underreports (e.g., linearized PDFs, XRef stream objects).
        let lopdf_count = read_structural_page_count_lopdf(pdf_path);

        if let (Some(pc), Some(lc)) = (pdfium_count, lopdf_count)
            && pc != lc
        {
            tracing::warn!(
                pdf = %pdf_path.display(),
                pdfium_pages = pc,
                lopdf_pages = lc,
                "page count mismatch between pdfium and lopdf, using pdfium count"
            );
        }

        if pdfium_count.is_some() {
            return pdfium_count;
        }

        // Pdfium failed (binding or load error). Fall back to lopdf.
        tracing::debug!(
            pdf = %pdf_path.display(),
            "pdfium page count unavailable, falling back to lopdf"
        );
        lopdf_count
    }

    // When pdfium feature is disabled, lopdf is the only option.
    #[cfg(not(feature = "pdfium"))]
    {
        read_structural_page_count_lopdf(pdf_path)
    }
}

/// Maximum PDF file size that `read_structural_page_count_lopdf` will load into
/// memory. Files larger than this cap are skipped to prevent OOM when indexing
/// scan-heavy PDFs with hundreds of embedded high-resolution images. 256 MiB is
/// sufficient for all well-formed text-based PDFs; image-heavy PDFs that exceed
/// this limit are handled by the pdfium backend which streams pages individually.
const MAX_LOPDF_PROBE_BYTES: u64 = 256 * 1024 * 1024;

/// Reads the structural page count using lopdf (pure Rust PDF parser).
/// This parser works without external dependencies but may underreport the
/// page count for PDFs with linearized structures, incremental updates,
/// or XRef stream objects.
///
/// Returns `None` if the file exceeds `MAX_LOPDF_PROBE_BYTES`, cannot be read,
/// or the PDF structure is invalid.
fn read_structural_page_count_lopdf(pdf_path: &Path) -> Option<usize> {
    // Guard against loading very large PDFs into memory in one call.
    // std::fs::metadata is a stat(2) call and does not read file data.
    let file_size = std::fs::metadata(pdf_path).ok()?.len();
    if file_size > MAX_LOPDF_PROBE_BYTES {
        return None;
    }
    let bytes = std::fs::read(pdf_path).ok()?;
    let doc = lopdf::Document::load_mem(&bytes).ok()?;
    let count = doc.get_pages().len();
    if count > 0 { Some(count) } else { None }
}

/// Reads the structural page count using pdfium. Opens the PDF document
/// with pdfium and reads the page count from pdfium's internal page tree
/// parser, which handles more PDF structural variations than lopdf.
///
/// Returns `None` if pdfium cannot be loaded or the PDF cannot be opened.
#[cfg(feature = "pdfium")]
fn read_structural_page_count_pdfium(pdf_path: &Path) -> Option<usize> {
    let bindings = crate::pdfium_binding::bind_pdfium().ok()?;
    let pdfium = pdfium_render::prelude::Pdfium::new(bindings);
    let document = pdfium.load_pdf_from_file(pdf_path, None).ok()?;
    let count = document.pages().len() as usize;
    if count > 0 { Some(count) } else { None }
}

/// Removes C0 control characters (U+0000..U+001F) from a string in place,
/// preserving horizontal tab (0x09), newline (0x0A), and carriage return
/// (0x0D) which are legitimate whitespace in extracted text.
///
/// Characters removed include NUL (0x00), form feed (0x0C), vertical tab
/// (0x0B), backspace (0x08), and escape (0x1B). These appear in PDF text
/// extraction output when font encoding tables are incomplete or when
/// binary data leaks into content streams.
///
/// Operates in-place by retaining only non-control characters (or the three
/// allowed whitespace characters). This avoids a reallocation when no
/// control characters are present.
fn strip_control_characters(text: &mut String) {
    text.retain(|c| !c.is_control() || c == '\t' || c == '\n' || c == '\r');
}

/// Implements the automatic backend selection and fallback chain.
///
/// Phase 1: Attempt extraction with pdf-extract (pure Rust, no external
///          dependencies). Both errors and panics from pdf-extract are
///          caught so the fallback chain can continue.
///
/// Phase 2: When pdf-extract fails completely (error or panic) and the
///          pdfium feature is enabled, extract the entire document with
///          pdfium as a wholesale replacement.
///          When pdf-extract succeeds but individual pages have poor
///          quality (garbled text, high replacement character ratio),
///          re-extract those specific pages with pdfium.
///          When pdf-extract returns significantly fewer pages than the
///          structural page count, pdfium extracts the full document as
///          a wholesale replacement (the pdf-extract form-feed split
///          likely merged pages into a single block).
///
/// Phase 3: Collect all pages still below the minimum non-whitespace
///          character threshold and perform batch OCR if the `ocr`
///          feature is enabled. The batch call loads pdfium and Tesseract
///          once for all pages, avoiding per-page reinitialization.
///          Pages that are in the PDF structure but missing from the
///          extracted result (never produced by any text backend) are
///          created as placeholder entries and included in the OCR batch.
///
/// # Parameters
///
/// - `pdf_path`: Absolute path to the PDF file.
/// - `ocr_language`: Tesseract language code for OCR fallback.
/// - `pdf_page_count`: Structural page count from the PDF page tree,
///   used to detect missing pages in the extraction result.
fn auto_extract(
    pdf_path: &Path,
    ocr_language: Option<&str>,
    pdf_page_count: Option<usize>,
) -> Result<Vec<PageText>, PdfError> {
    // Suppress unused-variable warnings when neither pdfium nor ocr features
    // are enabled: ocr_language is only used in the OCR phase, pdf_page_count
    // is used in both the pdfium fallback (page coverage check) and OCR
    // fallback (missing page detection).
    let _ = ocr_language;
    let _ = pdf_page_count;

    // Phase 1: Try pdf-extract first. Catch both errors and panics so
    // the fallback chain can continue. The pdf-extract crate panics on
    // certain malformed PDFs (broken CIDRange tables, invalid Type1 font
    // encoding data), and returns errors for others (invalid cross-reference
    // tables). Both failure modes must be caught here, not propagated.
    let pdf_extract_result: Result<Vec<PageText>, PdfError> =
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pdf_extract::extract_with_pdf_extract(pdf_path)
        })) {
            Ok(result) => result,
            Err(panic_payload) => {
                let msg = if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else {
                    "unknown panic".to_string()
                };
                Err(PdfError::Extraction(format!(
                    "pdf-extract panicked for {}: {msg}",
                    pdf_path.display()
                )))
            }
        };

    // Phase 2: Apply pdfium fallback depending on how pdf-extract fared.
    // The structural page count is passed to detect cases where pdf-extract
    // returned far fewer pages than exist in the PDF (e.g., form-feed split
    // collapsed 500+ pages into a single entry).
    #[cfg(feature = "pdfium")]
    let pages = apply_pdfium_fallback(pdf_path, pdf_extract_result, pdf_page_count)?;

    // When pdfium is not available, propagate pdf-extract failures directly.
    #[cfg(not(feature = "pdfium"))]
    let pages = pdf_extract_result?;

    // Phase 3: Batch OCR fallback for pages with very little extractable text.
    // Collects all page numbers below the threshold first, then performs a
    // single batch OCR call that loads pdfium + Tesseract once for all pages.
    // The structural page count is passed so that pages missing from the
    // extraction result (never produced by any text backend) can be created
    // as placeholder entries and included in the OCR batch.
    #[cfg(feature = "ocr")]
    let pages = apply_ocr_fallback(pdf_path, pages, ocr_language, pdf_page_count)?;

    Ok(pages)
}

/// Minimum ratio of extracted pages to structural page count below which
/// pdfium wholesale fallback is triggered. When pdf-extract produces fewer
/// than 50% of the expected pages, the form-feed-based page splitting
/// likely failed (e.g., all pages merged into a single block), and the
/// entire document is re-extracted with pdfium.
#[cfg(feature = "pdfium")]
const PAGE_COVERAGE_THRESHOLD: f64 = 0.5;

/// Applies the pdfium fallback chain to the result of pdf-extract.
///
/// Three fallback triggers:
///
/// 1. **Complete failure**: pdf-extract errored or panicked. The entire
///    document is extracted with pdfium as wholesale replacement.
///
/// 2. **Low page coverage**: pdf-extract returned pages, but significantly
///    fewer than the structural page count (below PAGE_COVERAGE_THRESHOLD).
///    This indicates the form-feed page splitting collapsed multiple pages
///    into a single entry. Pdfium re-extracts the full document.
///
/// 3. **Per-page quality failure**: pdf-extract returned the expected number
///    of pages, but individual pages have poor text quality (garbled chars,
///    high replacement char ratio). Those pages are individually replaced
///    with pdfium output. Pdfium is loaded lazily on the first quality
///    failure to avoid unnecessary library binding for clean PDFs.
///
/// If pdfium also fails, the original pdf-extract result (or error) is
/// returned because it describes the primary failure cause.
///
/// # Parameters
///
/// - `pdf_path`: Absolute path to the PDF file.
/// - `pdf_extract_result`: The result from pdf-extract (pages or error).
/// - `pdf_page_count`: Structural page count from the PDF page tree,
///   used to detect low page coverage.
#[cfg(feature = "pdfium")]
fn apply_pdfium_fallback(
    pdf_path: &Path,
    pdf_extract_result: Result<Vec<PageText>, PdfError>,
    pdf_page_count: Option<usize>,
) -> Result<Vec<PageText>, PdfError> {
    use neuroncite_core::ExtractionBackend;

    match pdf_extract_result {
        Ok(mut pages) => {
            // Check whether pdf-extract returned significantly fewer pages
            // than the PDF structure contains. When the form-feed-based page
            // split fails (common for PDFs without form-feed characters),
            // all content ends up in a single PageText entry. In that case,
            // re-extract the entire document with pdfium.
            if let Some(structural_count) = pdf_page_count
                && structural_count > 1
            {
                let coverage = pages.len() as f64 / structural_count as f64;
                if coverage < PAGE_COVERAGE_THRESHOLD {
                    tracing::info!(
                        "pdf-extract returned {} pages but PDF has {} structural pages \
                         (coverage {:.1}% < {:.0}%), triggering pdfium wholesale fallback for {}",
                        pages.len(),
                        structural_count,
                        coverage * 100.0,
                        PAGE_COVERAGE_THRESHOLD * 100.0,
                        pdf_path.display(),
                    );

                    match pdfium::extract_with_pdfium(pdf_path) {
                        Ok(pdfium_pages) => return Ok(pdfium_pages),
                        Err(e) => {
                            tracing::warn!(
                                "pdfium wholesale fallback failed for {}: {e}",
                                pdf_path.display()
                            );
                            // Continue with the sparse pdf-extract results.
                        }
                    }
                }
            }

            // pdf-extract returned sufficient page coverage. Check per-page
            // quality and replace individual pages with pdfium output where
            // quality is poor.
            let mut pdfium_pages: Option<Vec<PageText>> = None;

            for page in &mut pages {
                let text_quality = crate::quality::check_text_quality(&page.content);
                if crate::quality::should_fallback(&text_quality) {
                    // Lazily load pdfium pages on first quality failure.
                    if pdfium_pages.is_none() {
                        match pdfium::extract_with_pdfium(pdf_path) {
                            Ok(pp) => pdfium_pages = Some(pp),
                            Err(e) => {
                                tracing::warn!(
                                    "pdfium fallback failed for {}: {e}",
                                    pdf_path.display()
                                );
                                // Continue with pdf-extract results.
                                break;
                            }
                        }
                    }

                    // Replace the page content with pdfium output if available.
                    if let Some(ref pp) = pdfium_pages
                        && let Some(pdfium_page) =
                            pp.iter().find(|p| p.page_number == page.page_number)
                    {
                        page.content.clone_from(&pdfium_page.content);
                        page.backend = ExtractionBackend::Pdfium;
                    }
                }
            }

            Ok(pages)
        }
        Err(pdf_extract_err) => {
            // pdf-extract failed completely (error or panic). Attempt
            // full-document extraction with pdfium as wholesale fallback.
            tracing::info!(
                "pdf-extract failed for {}, trying pdfium fallback: {pdf_extract_err}",
                pdf_path.display()
            );

            match pdfium::extract_with_pdfium(pdf_path) {
                Ok(pages) => Ok(pages),
                Err(pdfium_err) => {
                    // Both backends failed. Return the original pdf-extract
                    // error because it describes the primary failure cause.
                    tracing::warn!(
                        "pdfium fallback also failed for {}: {pdfium_err}",
                        pdf_path.display()
                    );
                    Err(pdf_extract_err)
                }
            }
        }
    }
}

/// Applies batch OCR to pages with very little extractable text and to
/// pages that are missing from the extraction result entirely.
///
/// Missing pages are detected by comparing the extracted page numbers
/// against the full range 1..=pdf_page_count. For each missing page,
/// a placeholder `PageText` entry is created (empty content, OCR backend)
/// and added to the pages vector before the OCR batch is submitted.
///
/// Pages already present but with fewer than MIN_NON_WHITESPACE_CHARS
/// non-whitespace characters are also included in the OCR batch.
///
/// # Parameters
///
/// - `pdf_path`: Absolute path to the PDF file.
/// - `pages`: Pages from text-based extraction backends.
/// - `ocr_language`: Tesseract language code (defaults to "eng").
/// - `pdf_page_count`: Structural page count from the PDF page tree.
///   When `None`, only existing pages below the threshold are OCR'd
///   (missing page detection is skipped).
#[cfg(feature = "ocr")]
fn apply_ocr_fallback(
    pdf_path: &Path,
    mut pages: Vec<PageText>,
    ocr_language: Option<&str>,
    pdf_page_count: Option<usize>,
) -> Result<Vec<PageText>, PdfError> {
    use std::collections::HashSet;

    use neuroncite_core::ExtractionBackend;

    let language = ocr_language.unwrap_or("eng");

    // Detect pages that are in the PDF structure but missing from the
    // extraction result. These pages were never produced by any text
    // backend (e.g., image-only pages skipped by pdf-extract, or pages
    // lost when pdfium was not available). Create placeholder entries
    // so the OCR batch can process them.
    if let Some(structural_count) = pdf_page_count {
        let extracted_page_numbers: HashSet<usize> = pages.iter().map(|p| p.page_number).collect();

        for page_num in 1..=structural_count {
            if !extracted_page_numbers.contains(&page_num) {
                pages.push(PageText {
                    source_file: pdf_path.to_path_buf(),
                    page_number: page_num,
                    content: String::new(),
                    backend: ExtractionBackend::Ocr,
                });
            }
        }

        // Sort pages by page number after adding placeholders to maintain
        // a consistent ordering for downstream consumers.
        pages.sort_by_key(|p| p.page_number);
    }

    // Collect 1-indexed page numbers of pages needing OCR (both existing
    // pages below the threshold and newly added placeholder entries).
    let ocr_page_numbers: Vec<usize> = pages
        .iter()
        .filter(|page| {
            let non_ws_count = page.content.chars().filter(|c| !c.is_whitespace()).count();
            non_ws_count < MIN_NON_WHITESPACE_CHARS
        })
        .map(|page| page.page_number)
        .collect();

    if !ocr_page_numbers.is_empty() {
        tracing::info!(
            "triggering batch OCR for {} pages of {} (language: {language})",
            ocr_page_numbers.len(),
            pdf_path.display(),
        );

        // Perform batch OCR: loads pdfium library, PDF document, and
        // Tesseract instance once for all pages.
        match crate::ocr::ocr_pdf_pages_batch(pdf_path, &ocr_page_numbers, language) {
            Ok(ocr_results) => {
                // Apply OCR text to the corresponding pages. This covers
                // both pre-existing pages that had insufficient text and
                // placeholder entries for missing pages.
                for page in &mut pages {
                    if let Some(ocr_text) = ocr_results.get(&page.page_number) {
                        page.content.clone_from(ocr_text);
                        page.backend = ExtractionBackend::Ocr;
                    }
                }
            }
            Err(e) => {
                tracing::warn!("batch OCR failed for {}: {e}", pdf_path.display());
                // Keep existing text rather than losing it. Placeholder
                // entries for missing pages remain with empty content.
            }
        }
    }

    Ok(pages)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// T-PDF-013: A corrupted PDF file (random bytes with .pdf extension)
    /// returns a `PdfError` variant and does not panic.
    #[test]
    fn t_pdf_013_corrupted_pdf_returns_error() {
        let corrupt_bytes: Vec<u8> = (0..256).map(|i| (i * 37 + 13) as u8).collect();

        let mut tmp = NamedTempFile::with_suffix(".pdf").expect("failed to create temp file");
        tmp.write_all(&corrupt_bytes)
            .expect("failed to write corrupt bytes");
        tmp.flush().expect("flush failed");

        let result = extract_pages(tmp.path(), None, None);

        assert!(
            result.is_err(),
            "corrupted PDF must produce an error, not a success"
        );
    }

    /// T-PDF-014: A valid PDF with no extractable text returns `PageText`
    /// entries with empty or whitespace-only content.
    #[test]
    fn t_pdf_014_empty_text_pdf() {
        // Construct a minimal valid PDF with an empty content stream.
        // The page has no text operators, so extraction produces empty text.
        let pdf = b"%PDF-1.4\n\
            1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\
            2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\
            3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
            /Contents 4 0 R /Resources << >> >>\nendobj\n\
            4 0 obj\n<< /Length 0 >>\nstream\n\nendstream\nendobj\n\
            xref\n0 5\n\
            0000000000 65535 f \n\
            0000000009 00000 n \n\
            0000000058 00000 n \n\
            0000000115 00000 n \n\
            0000000230 00000 n \n\
            trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n0\n%%EOF";

        let mut tmp = NamedTempFile::with_suffix(".pdf").expect("failed to create temp file");
        tmp.write_all(pdf).expect("failed to write PDF bytes");
        tmp.flush().expect("flush failed");

        let result = extract_pages(tmp.path(), None, None);

        // The extraction may succeed with empty content or fail due to
        // strict parsing. Both outcomes are acceptable for this test.
        match result {
            Ok(extraction) => {
                // If pages are returned, their content should be empty or
                // whitespace-only (no text operators in the content stream).
                for page in &extraction.pages {
                    let non_ws: usize = page.content.chars().filter(|c| !c.is_whitespace()).count();
                    // Allow a small number of artifacts from pdf-extract.
                    assert!(
                        non_ws < 20,
                        "page {} should have minimal text, found {} non-whitespace chars",
                        page.page_number,
                        non_ws
                    );
                }
            }
            Err(_) => {
                // Acceptable: the minimal PDF may fail strict parsing.
            }
        }
    }

    /// T-PDF-015: QUALITY-004 regression: `strip_control_characters` removes
    /// NUL bytes, form feeds, vertical tabs, and other C0 control characters
    /// while preserving tab, newline, and carriage return.
    #[test]
    fn t_pdf_015_strip_control_characters_basic() {
        let mut text = "Hello\x00World\x0BFoo\x0CBar\x1BEsc".to_string();
        strip_control_characters(&mut text);
        assert_eq!(text, "HelloWorldFooBarEsc");
    }

    /// T-PDF-016: QUALITY-004 regression: `strip_control_characters` preserves
    /// tab, newline, and carriage return characters since these are legitimate
    /// whitespace in extracted PDF text.
    #[test]
    fn t_pdf_016_strip_preserves_legitimate_whitespace() {
        let mut text = "Column A\tColumn B\nLine 2\r\nLine 3".to_string();
        strip_control_characters(&mut text);
        assert_eq!(text, "Column A\tColumn B\nLine 2\r\nLine 3");
    }

    /// T-PDF-017: QUALITY-004 regression: `strip_control_characters` is a
    /// no-op on strings that contain no control characters.
    #[test]
    fn t_pdf_017_strip_noop_on_clean_text() {
        let mut text = "The quick brown fox jumps over the lazy dog.".to_string();
        let original = text.clone();
        strip_control_characters(&mut text);
        assert_eq!(text, original);
    }

    /// T-PDF-018: QUALITY-004 regression: `strip_control_characters` handles
    /// strings composed entirely of control characters, producing an empty
    /// string.
    #[test]
    fn t_pdf_018_strip_all_control_chars() {
        let mut text = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0B\x0C\x0E\x0F\x1B\x1F".to_string();
        strip_control_characters(&mut text);
        assert!(text.is_empty(), "all control chars must be removed");
    }

    /// T-PDF-019: QUALITY-004 regression: `strip_control_characters` handles
    /// an empty string without error.
    #[test]
    fn t_pdf_019_strip_empty_string() {
        let mut text = String::new();
        strip_control_characters(&mut text);
        assert!(text.is_empty());
    }

    /// T-PDF-020: QUALITY-004 regression: `strip_control_characters` correctly
    /// handles multi-byte UTF-8 characters mixed with control characters.
    /// Unicode characters outside the C0 range must be preserved.
    #[test]
    fn t_pdf_020_strip_preserves_unicode() {
        let mut text = "Schr\u{00F6}dinger\x00 Pr\u{00FC}fung\x1B \u{2603}".to_string();
        strip_control_characters(&mut text);
        assert_eq!(text, "Schr\u{00F6}dinger Pr\u{00FC}fung \u{2603}");
    }
}
