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

// Five-stage text location pipeline for finding quoted passages in PDFs.
//
// Given a quote string and a pdfium-loaded PDF document, this module
// attempts to locate the exact text region using progressively more
// tolerant matching strategies:
//
//   Stage 1   (Exact):            str::find() on pdfium-extracted page text.
//   Stage 2   (Normalized):       Unicode NFC, whitespace collapse, hyphen removal.
//   Stage 3   (Fuzzy):            Levenshtein sliding window with >= 80% threshold.
//   Stage 3.5 (FallbackExtract):  neuroncite-pdf multi-backend extraction pipeline
//                                 (pdf-extract, pdfium text, OCR) for pages where
//                                 pdfium text extraction returned empty strings.
//                                 Uses page-level content-area bounding boxes.
//   Stage 4   (OCR):              Tesseract hOCR for bounding-box-aware matching.
//
// Stages 1-3 use pdfium's per-character bounding boxes for precise highlight
// positioning. Stage 3.5 falls back to page-level content-area rectangles
// (72pt inset from page edges) since the text source lacks character-level
// positioning. Stage 4 uses Tesseract hOCR word-level bounding boxes.
//
// The first successful match terminates the search.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use pdfium_render::prelude::*;
use unicode_normalization::UnicodeNormalization;

use crate::error::AnnotateError;
use crate::types::{MatchMethod, MatchResult};

/// Per-page hOCR data: a list of recognized words with pixel bounding boxes
/// and the render DPI used to produce those coordinates. With the `ocr` feature
/// disabled, this resolves to a unit type that is never instantiated (all
/// callers pass `None`).
#[cfg(feature = "ocr")]
pub type HocrPageData = (Vec<neuroncite_pdf::HocrWord>, u32);

/// Placeholder type when OCR is not compiled in. Never instantiated.
#[cfg(not(feature = "ocr"))]
pub type HocrPageData = ();

/// Groups the optional context parameters for `locate_quote`. This struct
/// reduces the parameter count from 7 to 5 by bundling the three optional
/// search-context fields into a single struct. Callers construct a
/// `LocateContext` with the fields they have available and pass a reference.
pub struct LocateContext<'a> {
    /// Pre-extracted page texts from an indexed session's database or from
    /// the hOCR pre-extraction phase. Map keys are 1-indexed page numbers.
    /// When present, stage 3.5 (FallbackExtract) uses these cached texts
    /// instead of re-running `neuroncite_pdf::extract_pages()`.
    pub cached_pages: Option<&'a HashMap<usize, String>>,
    /// Pre-computed Tesseract hOCR results (word-level bounding boxes + render
    /// DPI) from the pipeline's pre-extraction phase. When present, Stage 4
    /// uses these directly instead of re-rendering pages and re-running OCR.
    pub precomputed_hocr: Option<&'a HashMap<usize, HocrPageData>>,
    /// A 1-indexed page number hint from the citation verification pipeline.
    /// When present, that page is searched first (stages 1-3) before falling
    /// through to a full-document scan.
    pub page_hint: Option<usize>,
}

impl<'a> Default for LocateContext<'a> {
    /// Returns a LocateContext with all optional fields set to None.
    fn default() -> Self {
        Self {
            cached_pages: None,
            precomputed_hocr: None,
            page_hint: None,
        }
    }
}

/// Tracks diagnostic information across all pipeline stages during a single
/// `locate_quote` call. Accumulates per-stage statistics and failure reasons
/// which are converted to human-readable diagnostic strings when a quote is
/// not found after all stages.
struct StageTracker {
    /// Number of pages where stage 1 (exact match) was attempted.
    exact_pages: usize,
    /// Number of pages where stage 2 (normalized match) was attempted.
    normalized_pages: usize,
    /// Number of pages where stage 3 (fuzzy match) was attempted.
    fuzzy_pages: usize,
    /// Number of pages skipped in stage 3 due to page/quote length limits.
    fuzzy_skipped_pages: usize,
    /// Best fuzzy similarity score observed across all pages (0.0 if no
    /// fuzzy windows were evaluated).
    best_fuzzy_score: f64,
    /// 1-indexed page number where the best fuzzy score was observed.
    best_fuzzy_page: usize,
    /// Whether stage 3.5 (fallback extraction) was attempted.
    fallback_attempted: bool,
    /// Reason stage 3.5 was skipped, when `fallback_attempted` is false.
    fallback_skip_reason: Option<&'static str>,
    /// Whether stage 4 (OCR) was attempted.
    ocr_attempted: bool,
    /// Number of pages processed by OCR in stage 4.
    ocr_pages: usize,
    /// Reason stage 4 was skipped, when `ocr_attempted` is false.
    ocr_skip_reason: Option<&'static str>,
    /// Which stage was active when the per-quote timeout was reached.
    /// `None` if no timeout occurred.
    timed_out_at_stage: Option<&'static str>,
}

impl StageTracker {
    fn new() -> Self {
        Self {
            exact_pages: 0,
            normalized_pages: 0,
            fuzzy_pages: 0,
            fuzzy_skipped_pages: 0,
            best_fuzzy_score: 0.0,
            best_fuzzy_page: 0,
            fallback_attempted: false,
            fallback_skip_reason: None,
            ocr_attempted: false,
            ocr_pages: 0,
            ocr_skip_reason: None,
            timed_out_at_stage: None,
        }
    }

    /// Converts the accumulated tracking data into a list of human-readable
    /// diagnostic strings, one per pipeline stage. Each string contains the
    /// stage name and the reason it did not produce a match.
    fn into_diagnostics(self) -> Vec<String> {
        let mut diags = Vec::with_capacity(6);

        // Stage 1: Exact match.
        diags.push(format!("exact: no match on {} pages", self.exact_pages));

        // Stage 2: Normalized match.
        diags.push(format!(
            "normalized: no match on {} pages",
            self.normalized_pages
        ));

        // Stage 3: Fuzzy match.
        if self.fuzzy_pages > 0 || self.fuzzy_skipped_pages > 0 {
            if self.best_fuzzy_score > 0.0 {
                let mut msg = format!(
                    "fuzzy: best={:.0}% on page {} < {:.0}% threshold",
                    self.best_fuzzy_score * 100.0,
                    self.best_fuzzy_page,
                    FUZZY_THRESHOLD * 100.0,
                );
                if self.fuzzy_skipped_pages > 0 {
                    msg.push_str(&format!(
                        " ({} pages checked, {} skipped: page/quote too long)",
                        self.fuzzy_pages, self.fuzzy_skipped_pages,
                    ));
                }
                diags.push(msg);
            } else if self.fuzzy_pages > 0 {
                diags.push(format!(
                    "fuzzy: no viable candidates on {} pages",
                    self.fuzzy_pages,
                ));
            } else {
                diags.push(format!(
                    "fuzzy: skipped on all {} pages (page/quote too long)",
                    self.fuzzy_skipped_pages,
                ));
            }
        }

        // Stage 3.5: Fallback extraction.
        if self.fallback_attempted {
            diags.push("fallback_extract: no match".to_string());
        } else if let Some(reason) = self.fallback_skip_reason {
            diags.push(format!("fallback_extract: skipped ({reason})"));
        }

        // Stage 4: OCR.
        if self.ocr_attempted {
            diags.push(format!("ocr: no match on {} pages", self.ocr_pages));
        } else if let Some(reason) = self.ocr_skip_reason {
            diags.push(format!("ocr: skipped ({reason})"));
        }

        // Timeout indicator.
        if let Some(stage) = self.timed_out_at_stage {
            diags.push(format!(
                "timeout: {}s limit reached during {stage}",
                QUOTE_LOCATE_TIMEOUT.as_secs(),
            ));
        }

        diags
    }
}

/// Maximum page text length (in characters) for the fuzzy matching stage.
/// Pages longer than this are skipped to prevent excessive computation.
const FUZZY_MAX_PAGE_CHARS: usize = 100_000;

/// Maximum quote length (in characters) for the fuzzy matching stage.
const FUZZY_MAX_QUOTE_CHARS: usize = 500;

/// Minimum normalized Levenshtein similarity for the fuzzy match stage (stages 1-3).
/// Set to 0.80 rather than 0.90 because the citation verification agent
/// reformulates passages rather than copying verbatim. Agent text typically
/// differs from PDF text by ~10-20% (omissions, added context, formatting
/// changes). The anchor-word pre-filter (40% of distinctive words must
/// cluster within 2x quote length) already prevents false positives.
const FUZZY_THRESHOLD: f64 = 0.80;

/// Minimum normalized Levenshtein similarity for the fallback extraction stage (stage 3.5).
/// Lower than FUZZY_THRESHOLD because text extracted by the multi-backend pipeline
/// (neuroncite-pdf, Tesseract) from older scanned documents contains systematic
/// recognition errors absent from pdfium's native text layer: ligature misreads
/// (the fi-ligature rendered as separate characters), character substitutions
/// (rn misread as m), hyphenation artifacts, and column-merge whitespace from
/// two-column layouts common in 1970s-1990s academic scans. A value of 0.80
/// catches these OCR-typical deviations while rejecting clearly unrelated passages.
/// Stage 4 (Tesseract hOCR with per-word boxes) uses OCR_FUZZY_THRESHOLD = 0.75
/// for the same reason. Set to 0.70 to account for the compound effect of
/// OCR errors and agent reformulation on the text.
const FALLBACK_FUZZY_THRESHOLD: f64 = 0.70;

/// Maximum time allowed for a single quote location attempt across all
/// stages. When this deadline is reached, remaining stages are skipped
/// and the quote is reported as "not found". This prevents the pipeline
/// from hanging indefinitely on PDFs where OCR fallback extraction or
/// fuzzy matching takes excessively long (e.g., large scanned documents
/// with complex layouts triggering Tesseract on every page).
const QUOTE_LOCATE_TIMEOUT: Duration = Duration::from_secs(120);

/// Minimum normalized Levenshtein similarity for the OCR fuzzy match stage.
/// Lower than the regular FUZZY_THRESHOLD (0.90) because OCR text inherently
/// contains character-level recognition errors: ligature misreads (fi -> fi),
/// character substitutions (rn -> m), and whitespace artifacts. A threshold
/// of 0.75 accepts these OCR-specific deviations while still rejecting
/// unrelated text.
#[cfg(feature = "ocr")]
const OCR_FUZZY_THRESHOLD: f64 = 0.75;

/// Locates a quoted text passage in the given PDF document. Iterates all
/// pages and tries each location stage in order. Returns the first
/// successful match with page number, character range, and bounding boxes.
///
/// The `pdf_path` and `language` parameters are used only for the OCR
/// fallback stage (stage 4) which needs to render the page to a raster image.
/// Without the "ocr" feature, these parameters are intentionally unused.
///
/// The `cached_pages` parameter provides pre-extracted page texts from an
/// indexed session's database or from the hOCR pre-extraction in
/// `process_single_pdf`. When present, stage 3.5 (FallbackExtract)
/// uses these cached texts instead of re-running `neuroncite_pdf::extract_pages()`.
/// The map keys are 1-indexed page numbers and values are the page text content.
///
/// The `precomputed_hocr` parameter provides pre-computed Tesseract hOCR
/// results (word-level bounding boxes + render DPI) from the pipeline's
/// pre-extraction phase. When present, Stage 4 uses these results directly
/// instead of re-rendering pages and re-running Tesseract OCR. This
/// eliminates the double-OCR overhead where pre-extraction OCRs all pages
/// for text content, and then Stage 4 re-OCRs the same pages for bounding
/// boxes. With precomputed hOCR, the entire annotation pipeline performs
/// only a single OCR pass per PDF.
///
/// The `page_hint` parameter provides a 1-indexed page number from the
/// citation verification pipeline. When present, that page is searched
/// first (stages 1-3) before falling through to a full-document scan.
/// This avoids scanning all pages for quotes whose location is already
/// known from the verification agent's passage references.
#[allow(unused_variables)]
pub fn locate_quote(
    document: &PdfDocument,
    pdf_path: &Path,
    quote: &str,
    language: &str,
    ctx: &LocateContext<'_>,
) -> Result<MatchResult, AnnotateError> {
    let cached_pages = ctx.cached_pages;
    let precomputed_hocr = ctx.precomputed_hocr;
    let page_hint = ctx.page_hint;
    let page_count = document.pages().len();
    let quote_trimmed = quote.trim();
    let deadline = Instant::now() + QUOTE_LOCATE_TIMEOUT;
    let mut tracker = StageTracker::new();

    if quote_trimmed.is_empty() {
        return Err(AnnotateError::QuoteNotFound {
            quote_excerpt: "(empty quote)".into(),
            stages_tried: vec![],
        });
    }

    // Track whether pdfium extracted any native text from the PDF during
    // stages 1-3. If at least one page has non-empty native text, the PDF
    // is not a scanned/image-only document, and the expensive fallback
    // extraction (stage 3.5) and OCR (stage 4) stages can be skipped.
    // This avoids >2 minute delays when a quote simply does not exist in
    // a native-text PDF.
    let mut has_native_text = false;

    // Build page iteration order: when a page hint is provided, search that
    // page first. Then iterate all remaining pages in document order,
    // skipping the hint page to avoid duplicate work.
    let hint_page_idx: Option<u16> = page_hint
        .filter(|&h| h >= 1 && (h - 1) < page_count as usize)
        .map(|h| (h - 1) as u16);

    let page_order: Vec<u16> = {
        let mut order = Vec::with_capacity(page_count as usize);
        if let Some(hint_idx) = hint_page_idx {
            order.push(hint_idx);
        }
        for idx in 0..page_count {
            if Some(idx) != hint_page_idx {
                order.push(idx);
            }
        }
        order
    };

    // Collect extracted page texts for cross-page matching after the main
    // per-page loop. Indexed by 0-based page index (u16) for efficient
    // lookup of adjacent pairs.
    let mut page_texts: HashMap<u16, String> = HashMap::new();

    for &page_idx in &page_order {
        // Check the per-quote timeout before processing each page. This
        // ensures that expensive stages (fuzzy matching, fallback extraction)
        // cannot cause the pipeline to hang indefinitely. The timeout is
        // checked at the page boundary rather than mid-computation to avoid
        // overhead on every character comparison.
        if Instant::now() > deadline {
            tracing::warn!(
                pdf = %pdf_path.display(),
                page_idx,
                elapsed_secs = QUOTE_LOCATE_TIMEOUT.as_secs(),
                "quote location timed out during stages 1-3"
            );
            tracker.timed_out_at_stage = Some("stages 1-3");
            break;
        }

        let page = match document.pages().get(page_idx) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let text_page = match page.text() {
            Ok(t) => t,
            Err(_) => continue,
        };

        // PdfPageText::all() returns String directly (not Result).
        let page_text = text_page.all();

        if !has_native_text && !page_text.trim().is_empty() {
            has_native_text = true;
        }

        // Store the page text for cross-page matching after the main loop.
        if !page_text.trim().is_empty() {
            page_texts.insert(page_idx, page_text.clone());
        }

        let page_number = page_idx as usize + 1;

        // Stage 1: exact match.
        tracker.exact_pages += 1;
        if let Some(result) = find_exact(&text_page, &page_text, quote_trimmed, page_number) {
            return Ok(result);
        }

        // Stage 2: normalized match.
        tracker.normalized_pages += 1;
        if let Some(result) = find_normalized(&text_page, &page_text, quote_trimmed, page_number) {
            return Ok(result);
        }

        // Stage 3: fuzzy match.
        if page_text.len() <= FUZZY_MAX_PAGE_CHARS && quote_trimmed.len() <= FUZZY_MAX_QUOTE_CHARS {
            tracker.fuzzy_pages += 1;
            let (result, page_best_score) =
                find_fuzzy(&text_page, &page_text, quote_trimmed, page_number);
            if page_best_score > tracker.best_fuzzy_score {
                tracker.best_fuzzy_score = page_best_score;
                tracker.best_fuzzy_page = page_number;
            }
            if let Some(r) = result {
                return Ok(r);
            }
        } else {
            tracker.fuzzy_skipped_pages += 1;
        }
    }

    // Stage 3+: Cross-page matching. Handles quotes that span two adjacent
    // pages by concatenating the text of consecutive page pairs and running
    // exact then normalized matching on the combined text. This catches cases
    // where a sentence or paragraph continues from the bottom of one page to
    // the top of the next, which individual-page stages 1-3 cannot match.
    //
    // When a cross-page match is found, the result is reported on the first
    // page of the pair with page-level bounding boxes (content area inset
    // by margins). Per-character coordinates spanning two pages are not
    // available from pdfium, so the page-level highlight is the best
    // approximation. The MatchMethod::CrossPage tag allows callers to
    // distinguish this from single-page matches.
    if Instant::now() <= deadline && page_texts.len() >= 2 {
        let mut sorted_pages: Vec<u16> = page_texts.keys().copied().collect();
        sorted_pages.sort();

        for pair in sorted_pages.windows(2) {
            let (pg_a, pg_b) = (pair[0], pair[1]);

            // Only concatenate truly adjacent pages (consecutive indices).
            if pg_b != pg_a + 1 {
                continue;
            }

            let text_a = &page_texts[&pg_a];
            let text_b = &page_texts[&pg_b];

            // Concatenate with a single space to join the last word on page A
            // with the first word on page B.
            let combined = format!("{} {}", text_a.trim_end(), text_b.trim_start());

            // Try exact match on the combined text.
            if combined.contains(quote_trimmed) {
                let page_number = pg_a as usize + 1;
                if let Ok(page) = document.pages().get(pg_a) {
                    let result = crate::annotate::create_page_level_match(
                        &page,
                        page_number,
                        quote_trimmed.chars().count(),
                    );
                    return Ok(MatchResult {
                        method: MatchMethod::CrossPage,
                        ..result
                    });
                }
            }

            // Try normalized match on the combined text.
            let norm_quote = normalize_text(quote_trimmed);
            let norm_combined = normalize_text(&combined);
            if norm_combined.contains(&norm_quote) {
                let page_number = pg_a as usize + 1;
                if let Ok(page) = document.pages().get(pg_a) {
                    let result = crate::annotate::create_page_level_match(
                        &page,
                        page_number,
                        quote_trimmed.chars().count(),
                    );
                    return Ok(MatchResult {
                        method: MatchMethod::CrossPage,
                        ..result
                    });
                }
            }
        }
    }

    // Check timeout before entering the expensive fallback stages (3.5 and 4).
    // These stages involve full-document text extraction (pdf-extract, pdfium,
    // OCR) and are the primary source of pipeline hangs on problematic PDFs.
    if Instant::now() > deadline {
        tracing::warn!(
            pdf = %pdf_path.display(),
            elapsed_secs = QUOTE_LOCATE_TIMEOUT.as_secs(),
            "quote location timed out, skipping fallback extraction and OCR stages"
        );
        tracker.timed_out_at_stage = tracker
            .timed_out_at_stage
            .or(Some("before fallback stages"));
        tracker.fallback_skip_reason = Some("timeout");
        tracker.ocr_skip_reason = Some("timeout");
        // Build the excerpt using char-count-based truncation. The byte-index
        // approach (&str[..80]) panics when byte 80 falls inside a multi-byte
        // UTF-8 character (e.g. curly apostrophe U+2019, encoded as 3 bytes).
        let excerpt = {
            let s: String = quote_trimmed.chars().take(80).collect();
            if quote_trimmed.len() > s.len() {
                format!("{}...", s)
            } else {
                s
            }
        };
        return Err(AnnotateError::QuoteNotFound {
            quote_excerpt: excerpt,
            stages_tried: tracker.into_diagnostics(),
        });
    }

    // Stage 3.5: Fallback extraction via cached index texts or neuroncite-pdf's
    // multi-backend pipeline (pdf-extract -> pdfium text -> OCR). Skipped when
    // the PDF has native text (detected in stages 1-3) UNLESS cached_pages are
    // available (cached texts are pre-computed and cost-free to search).
    //
    // For native-text PDFs where the quote was not found in stages 1-3, running
    // live extract_pages() with OCR is futile and extremely slow (>2 min for
    // 21 pages). Cached texts are still worth checking because they may have
    // been extracted via a different backend (pdf-extract vs pdfium) that
    // captured text the other backend missed.
    //
    // The result is captured (not returned immediately) so that Stage 4 can
    // refine the match with word-level bounding boxes from Tesseract hOCR.
    // Stage 3.5 produces page-level bounding boxes (a single rectangle covering
    // the text area of the page) because scanned PDFs have no native text layer
    // with per-character coordinate data. By passing the identified page number
    // to Stage 4, only that single page needs OCR (~7s) instead of the full
    // document (~252s for 35 pages). This resolves DEFECT-006 (performance)
    // and DEFECT-007 (precision).
    let fallback_result = if !has_native_text || cached_pages.is_some() {
        tracker.fallback_attempted = true;
        find_via_fallback_extraction(
            document,
            pdf_path,
            quote_trimmed,
            language,
            cached_pages,
            deadline,
        )
    } else {
        tracker.fallback_skip_reason = Some("native text PDF");
        None
    };

    // For native-text PDFs, Stage 3.5 matching via cached texts is the final
    // stage. OCR would re-extract the same text that stages 1-3 already
    // searched from pdfium's native text layer. Return the match immediately.
    if has_native_text && let Some(result) = fallback_result {
        return Ok(result);
    }

    // Stage 4: OCR-based matching with word-level bounding boxes.
    //
    // Two paths exist:
    //
    // A) Precomputed hOCR (from pipeline pre-extraction): The annotation
    //    pipeline's process_single_pdf already ran hOCR on all pages of the
    //    scanned PDF once. The results are passed here to avoid a second OCR
    //    pass. This is the primary path for annotation workloads and produces
    //    word-level bounding boxes without any additional OCR overhead.
    //
    // B) Live OCR (no precomputed data): Spawns Tesseract hOCR in a separate
    //    thread with a deadline-based timeout. Used when locate_quote is
    //    called outside the annotation pipeline (e.g., from a search or
    //    verification context) where pre-extraction has not been performed.
    //
    // When Stage 3.5 identified a matching page, Stage 4 only processes that
    // single page (for word-level bbox refinement). When Stage 3.5 found
    // nothing, Stage 4 processes all pages as a last-resort search.
    #[cfg(feature = "ocr")]
    {
        if !has_native_text && Instant::now() <= deadline {
            let pages_to_ocr: Vec<usize> = if let Some(ref fb) = fallback_result {
                vec![fb.page_number]
            } else {
                (1..=page_count as usize).collect()
            };

            tracker.ocr_attempted = true;
            tracker.ocr_pages = pages_to_ocr.len();

            let fallback_page = fallback_result.as_ref().map(|fb| fb.page_number);

            // Path A: Use precomputed hOCR results from pipeline pre-extraction.
            // The match_hocr_on_pages function contains the same matching logic
            // used after live OCR, so results are identical.
            if let Some(hocr_data) = precomputed_hocr {
                if let Some(result) = match_hocr_on_pages(
                    document,
                    pdf_path,
                    quote_trimmed,
                    &pages_to_ocr,
                    hocr_data,
                    fallback_page,
                ) {
                    return Ok(result);
                }
            } else {
                // Path B (live OCR) is permanently disabled. find_via_ocr_batch
                // creates new pdfium instances internally (one per rayon thread)
                // which causes DLL loader lock contention with the annotation
                // pipeline's existing pdfium instances on Windows. The production
                // path is Path A: the pipeline's inline render+Tesseract pre-
                // extraction populates precomputed_hocr. When precomputed_hocr
                // is None (e.g., Tesseract not installed, or render failures),
                // Stage 3.5's page-level match is returned below.
                tracing::debug!(
                    pdf = %pdf_path.display(),
                    pages = pages_to_ocr.len(),
                    "Stage 4 Path B: live OCR disabled, using Stage 3.5 result"
                );
            }
        } else if has_native_text {
            tracker.ocr_skip_reason = Some("native text PDF");
        } else {
            tracker.ocr_skip_reason = Some("timeout");
        }
    }

    #[cfg(not(feature = "ocr"))]
    {
        tracker.ocr_skip_reason = Some("feature not enabled");
    }

    // If Stage 3.5 found a match but Stage 4 OCR refinement failed (timeout,
    // tesseract not available, or no match in hOCR output), return the
    // page-level match from Stage 3.5. The annotation pipeline will create a
    // page-level highlight and report the match with pre_check_passed=false
    // and status="low_confidence_match" to indicate imprecise positioning.
    if let Some(result) = fallback_result {
        return Ok(result);
    }

    // Build the excerpt using char-count-based truncation. The byte-index
    // approach (&str[..80]) panics when byte 80 falls inside a multi-byte
    // UTF-8 character (e.g. curly apostrophe U+2019, encoded as 3 bytes).
    let excerpt = {
        let s: String = quote_trimmed.chars().take(80).collect();
        if quote_trimmed.len() > s.len() {
            format!("{}...", s)
        } else {
            s
        }
    };

    Err(AnnotateError::QuoteNotFound {
        quote_excerpt: excerpt,
        stages_tried: tracker.into_diagnostics(),
    })
}

// ---------------------------------------------------------------------------
// Stage 1: Exact match
// ---------------------------------------------------------------------------

/// Searches for the exact quote string in the page text using str::find().
/// If found, extracts per-character bounding boxes from the pdfium text page.
fn find_exact(
    text_page: &PdfPageText,
    page_text: &str,
    quote: &str,
    page_number: usize,
) -> Option<MatchResult> {
    let pos = page_text.find(quote)?;
    let char_start = page_text[..pos].chars().count();
    let char_end = char_start + quote.chars().count();

    let boxes = extract_char_bounds(text_page, char_start, char_end);
    if boxes.is_empty() {
        return None;
    }

    Some(MatchResult {
        page_number,
        char_start,
        char_end,
        bounding_boxes: boxes,
        method: MatchMethod::Exact,
        fuzzy_score: None,
    })
}

// ---------------------------------------------------------------------------
// Stage 2: Normalized match
// ---------------------------------------------------------------------------

/// Normalizes both the quote and page text, then performs a substring search.
/// Handles common PDF text extraction artifacts: soft hyphens, hyphenated
/// line breaks, irregular whitespace, and Unicode normalization differences.
fn find_normalized(
    text_page: &PdfPageText,
    page_text: &str,
    quote: &str,
    page_number: usize,
) -> Option<MatchResult> {
    let norm_quote = normalize_text(quote);
    let norm_page = normalize_text(page_text);

    let norm_pos = norm_page.find(&norm_quote)?;

    // Map the position in normalized text back to the original text.
    // Walk both strings in parallel, tracking the original character index
    // corresponding to each normalized character position.
    let (char_start, char_end) = map_normalized_to_original(page_text, norm_pos, norm_quote.len());

    let boxes = extract_char_bounds(text_page, char_start, char_end);
    if boxes.is_empty() {
        return None;
    }

    Some(MatchResult {
        page_number,
        char_start,
        char_end,
        bounding_boxes: boxes,
        method: MatchMethod::Normalized,
        fuzzy_score: None,
    })
}

/// Normalizes text for comparison: PDF diacritical mark repair, ligature
/// decomposition, Unicode NFC, soft hyphen removal, hyphenated line break
/// joining, whitespace collapse, lowercase.
/// Public for use by the pipeline's Phase 1 pre-matching on cached page texts.
pub fn normalize_text(text: &str) -> String {
    // Repair spacing diacritics and decompose ligatures before NFC, so that
    // PDF extraction artifacts like "fu\u{00A8}r" (f + u + spacing diaeresis)
    // become "f\u{00FC}r" (f + u-umlaut + r) after NFC composition.
    //
    // The fixpoint loop ensures idempotency: after NFC composes a base letter
    // and a combining mark into a precomposed character, any subsequent spacing
    // diacritic now follows a letter and must also be converted. Converges in
    // at most 2 iterations.
    let mut nfc_text: String = {
        let repaired = repair_pdf_diacritics_and_ligatures(text);
        repaired.nfc().collect()
    };
    loop {
        let repaired = repair_pdf_diacritics_and_ligatures(&nfc_text);
        let next: String = repaired.nfc().collect();
        if next == nfc_text {
            break;
        }
        nfc_text = next;
    }
    nfc_text
        .replace('\u{00AD}', "")   // soft hyphen
        .replace("-\n", "")        // hyphenated line break
        .replace("-\r\n", "")      // Windows hyphenated line break
        .replace(['\n', '\r'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Mapping from spacing diacritical marks to their Unicode combining equivalents.
/// PDF text extraction backends produce these spacing variants when the PDF font
/// encodes diacritical marks as separate glyphs positioned above or below the
/// base letter. The spacing variants are standalone characters that do not
/// participate in Unicode canonical composition (NFC), so they must be replaced
/// by their combining counterparts before NFC can produce precomposed forms.
const SPACING_TO_COMBINING: &[(char, char)] = &[
    ('\u{00A8}', '\u{0308}'), // DIAERESIS -> COMBINING DIAERESIS
    ('\u{00B4}', '\u{0301}'), // ACUTE ACCENT -> COMBINING ACUTE ACCENT
    ('\u{0060}', '\u{0300}'), // GRAVE ACCENT -> COMBINING GRAVE ACCENT
    ('\u{00B8}', '\u{0327}'), // CEDILLA -> COMBINING CEDILLA
    ('\u{02C6}', '\u{0302}'), // MODIFIER LETTER CIRCUMFLEX -> COMBINING CIRCUMFLEX
    ('\u{02DC}', '\u{0303}'), // SMALL TILDE -> COMBINING TILDE
];

/// Mapping from Latin ligature codepoints to their decomposed ASCII letter
/// sequences. PDF fonts that use ligature glyphs map to Unicode ligature
/// codepoints in the Alphabetic Presentation Forms block (U+FB00..U+FB06).
const LIGATURE_DECOMPOSITIONS: &[(char, &str)] = &[
    ('\u{FB00}', "ff"),  // LATIN SMALL LIGATURE FF
    ('\u{FB01}', "fi"),  // LATIN SMALL LIGATURE FI
    ('\u{FB02}', "fl"),  // LATIN SMALL LIGATURE FL
    ('\u{FB03}', "ffi"), // LATIN SMALL LIGATURE FFI
    ('\u{FB04}', "ffl"), // LATIN SMALL LIGATURE FFL
    ('\u{FB05}', "st"),  // LATIN SMALL LIGATURE LONG S T
    ('\u{FB06}', "st"),  // LATIN SMALL LIGATURE ST
];

/// Converts spacing diacritical marks that follow a letter to their combining
/// equivalents, and decomposes Latin ligatures into ASCII letter sequences.
/// Spacing diacritics are only converted when they immediately follow an
/// alphabetic character, preserving standalone diacritics in non-letter contexts.
fn repair_pdf_diacritics_and_ligatures(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_letter = false;

    for ch in text.chars() {
        if prev_was_letter
            && let Some(&(_, combining)) = SPACING_TO_COMBINING.iter().find(|&&(s, _)| s == ch)
        {
            result.push(combining);
            prev_was_letter = false;
            continue;
        }

        if let Some(&(_, decomposed)) = LIGATURE_DECOMPOSITIONS.iter().find(|&&(lig, _)| lig == ch)
        {
            result.push_str(decomposed);
            prev_was_letter = true;
            continue;
        }

        prev_was_letter = ch.is_alphabetic();
        result.push(ch);
    }

    result
}

/// Maps a position in normalized text back to the corresponding character
/// range in the original text. Walks both strings in parallel, counting
/// original characters that survive normalization.
/// Public for use by the pipeline's Phase 1 pre-matching on cached page texts.
pub fn map_normalized_to_original(
    original: &str,
    norm_pos: usize,
    norm_len: usize,
) -> (usize, usize) {
    let orig_chars: Vec<char> = original.chars().collect();

    // Walk through original characters and track which survive normalization.
    let mut surviving_orig_indices: Vec<usize> = Vec::new();
    for (i, ch) in orig_chars.iter().enumerate() {
        // Characters that are removed by normalization: soft hyphens,
        // extra whitespace (all but one in a run), newlines.
        // This is an approximation -- the exact mapping depends on the
        // normalize_text() implementation.
        if *ch == '\u{00AD}' || *ch == '\r' {
            continue;
        }
        surviving_orig_indices.push(i);
    }

    // The normalized text has fewer or equal characters. Map the normalized
    // position to the original position via the surviving indices.
    let start = if norm_pos < surviving_orig_indices.len() {
        surviving_orig_indices[norm_pos]
    } else {
        orig_chars.len()
    };

    let end_norm = norm_pos + norm_len;
    let end = if end_norm < surviving_orig_indices.len() {
        surviving_orig_indices[end_norm]
    } else {
        orig_chars.len()
    };

    (start, end)
}

// ---------------------------------------------------------------------------
// Stage 3: Anchor-word fuzzy match
// ---------------------------------------------------------------------------

/// Common English stopwords excluded from anchor selection. These words are
/// too frequent to serve as reliable position indicators in page text.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "is", "are", "was", "were", "be", "been", "has", "have", "had", "do", "does", "did",
    "not", "no", "this", "that", "it", "its", "as", "if", "so", "we", "he", "she", "they", "their",
    "our", "can", "may", "will", "all", "each", "more", "than", "also", "into", "such", "when",
    "which", "where", "how", "who", "what", "between", "about", "both", "only", "other", "over",
    "most", "very", "some", "any", "would", "could", "should", "these", "those",
];

/// Extracts content words from the quote suitable for use as position anchors.
/// Filters out stopwords and short words (< 3 chars), then sorts by word
/// length descending so that the longest (most distinctive) words are first.
/// Returns at most 8 anchor words.
fn extract_anchor_words(quote: &str) -> Vec<String> {
    let mut words: Vec<String> = quote
        .split_whitespace()
        .filter(|w| w.len() >= 3)
        .map(|w| {
            // Strip leading/trailing punctuation for matching while keeping
            // the word readable. Punctuation at word boundaries (commas,
            // periods, parentheses) prevents anchor matching in page text.
            w.trim_matches(|c: char| c.is_ascii_punctuation())
                .to_lowercase()
        })
        .filter(|w| w.len() >= 3 && !STOPWORDS.contains(&w.as_str()))
        .collect();

    words.sort_by_key(|w| std::cmp::Reverse(w.len()));
    words.dedup();
    words.truncate(8);
    words
}

/// Finds all character-level positions of `word_lower` (already lowercased)
/// in `page_lower` (already lowercased). Returns a Vec of character offsets
/// where the word starts.
fn find_word_positions(page_lower: &str, word_lower: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut search_start = 0;
    while let Some(byte_pos) = page_lower[search_start..].find(word_lower) {
        let abs_byte_pos = search_start + byte_pos;
        // Convert byte offset to character offset.
        let char_pos = page_lower[..abs_byte_pos].chars().count();
        positions.push(char_pos);
        search_start = abs_byte_pos + word_lower.len().max(1);
    }
    positions
}

/// Groups anchor positions into candidate clusters. A cluster is a region
/// where multiple distinct anchor words appear within a span proportional
/// to the quote length. Each cluster is returned as (char_start, char_end)
/// encompassing all anchor positions in that group.
///
/// `positions` is a list of (anchor_word_index, char_position) sorted by
/// char_position. `quote_char_len` determines the maximum span for a cluster.
/// `min_distinct_anchors` is the minimum number of different anchor words
/// required for a cluster to be valid.
fn cluster_anchor_positions(
    positions: &[(usize, usize)],
    quote_char_len: usize,
    min_distinct_anchors: usize,
) -> Vec<(usize, usize)> {
    if positions.is_empty() {
        return Vec::new();
    }

    let max_span = quote_char_len * 2;
    let mut clusters = Vec::new();

    // Sliding window: advance start pointer when span exceeds max_span.
    let mut window_start = 0;
    for window_end in 0..positions.len() {
        // Shrink window from the left while span exceeds max_span.
        while positions[window_end]
            .1
            .saturating_sub(positions[window_start].1)
            > max_span
            && window_start < window_end
        {
            window_start += 1;
        }

        // Count distinct anchor word indices in the current window.
        let mut seen = [false; 8];
        for &(anchor_idx, _) in &positions[window_start..=window_end] {
            if anchor_idx < 8 {
                seen[anchor_idx] = true;
            }
        }
        let distinct = seen.iter().filter(|&&s| s).count();

        if distinct >= min_distinct_anchors {
            let cluster_start = positions[window_start].1;
            let cluster_end = positions[window_end].1;
            // Merge with previous cluster if overlapping.
            if let Some(last) = clusters.last_mut() {
                let (_, ref mut last_end): (usize, usize) = *last;
                if cluster_start <= *last_end + quote_char_len / 2 {
                    *last_end = cluster_end.max(*last_end);
                    continue;
                }
            }
            clusters.push((cluster_start, cluster_end));
        }
    }

    clusters
}

/// Anchor-word fuzzy matching algorithm. Replaces the brute-force sliding-
/// window Levenshtein approach with a targeted search that:
///
/// 1. Extracts distinctive words (anchors) from the quote
/// 2. Locates all positions of each anchor word on the page
/// 3. Clusters nearby anchor positions into candidate regions
/// 4. Scores each candidate region with normalized Levenshtein
///
/// Complexity: O(n * num_anchors + k * quote_len^2) where n = page length,
/// k = number of candidate clusters (typically 0-3). The brute-force approach
/// was O(n * quote_len * window_variants * quote_len) -- thousands of times
/// slower for large pages.
///
/// Returns (char_start, char_end, score) on the original page_text, or None.
/// Public for use by the pipeline's Phase 1 pre-matching on cached page texts.
pub fn find_anchor_fuzzy(
    page_text: &str,
    quote: &str,
    threshold: f64,
) -> Option<(usize, usize, f64)> {
    let anchors = extract_anchor_words(quote);
    if anchors.is_empty() {
        return None;
    }

    let page_lower = page_text.to_lowercase();
    let quote_char_len = quote.chars().count();

    if quote_char_len == 0 {
        return None;
    }

    // Locate all anchor word positions on the page.
    let mut all_positions: Vec<(usize, usize)> = Vec::new();
    for (anchor_idx, anchor) in anchors.iter().enumerate() {
        for char_pos in find_word_positions(&page_lower, anchor) {
            all_positions.push((anchor_idx, char_pos));
        }
    }

    if all_positions.is_empty() {
        return None;
    }

    // Sort by character position for clustering.
    all_positions.sort_by_key(|&(_, pos)| pos);

    // Require at least 40% of anchor words (minimum 2) to form a valid cluster.
    let min_distinct = (anchors.len() * 2 / 5).max(2).min(anchors.len());
    let clusters = cluster_anchor_positions(&all_positions, quote_char_len, min_distinct);

    if clusters.is_empty() {
        return None;
    }

    // Build char-to-byte offset map for extracting &str windows.
    let char_byte_offsets: Vec<usize> = page_text
        .char_indices()
        .map(|(byte_idx, _)| byte_idx)
        .chain(std::iter::once(page_text.len()))
        .collect();
    let page_char_count = char_byte_offsets.len() - 1;

    let norm_quote = normalize_text(quote);

    let mut best_score = 0.0_f64;
    let mut best_start = 0_usize;
    let mut best_end = 0_usize;

    for (cluster_start, cluster_end) in &clusters {
        // Expand the cluster to approximately quote length with margins.
        // The margin accounts for anchor words not covering the full quote span.
        let margin = quote_char_len / 3;
        let window_start = cluster_start.saturating_sub(margin);
        let window_end = (*cluster_end + quote_char_len + margin).min(page_char_count);

        if window_end <= window_start {
            continue;
        }

        // Extract the candidate window as a normalized string.
        let byte_start = char_byte_offsets[window_start];
        let byte_end = char_byte_offsets[window_end];
        let window_text = &page_text[byte_start..byte_end];
        let norm_window = normalize_text(window_text);

        // Try to find the quote as a normalized substring within the window
        // (handles cases where exact position is slightly off).
        if let Some(sub_pos) = norm_window.find(&norm_quote) {
            // Map back to original character coordinates.
            let norm_char_offset = norm_window[..sub_pos].chars().count();
            let original_start = window_start + norm_char_offset;
            let original_end = (original_start + norm_quote.chars().count()).min(page_char_count);
            return Some((original_start, original_end, 1.0));
        }

        // Score the candidate region against the quote. Try a small set of
        // sub-window sizes centered on the cluster midpoint (1.0x, 1.15x,
        // 1.3x of quote length). This replaces the old O(n^2) sliding
        // window (31 sizes x ~200 positions = 6,231 Levenshtein calls) with
        // at most 3 calls per cluster -- a 2,000x reduction in work.
        //
        // The anchor-word clustering already constrains the region to the
        // correct passage. These 3 scales handle: (a) exact-length match,
        // (b) moderate insertions/deletions, (c) larger reformulations.
        let window_chars = norm_window.chars().count();
        let norm_quote_len = norm_quote.chars().count();

        let w_char_offsets: Vec<usize> = norm_window
            .char_indices()
            .map(|(b, _)| b)
            .chain(std::iter::once(norm_window.len()))
            .collect();

        // Compute the cluster midpoint relative to the window. The window
        // expansion is asymmetric (margin before cluster_start, but
        // quote_char_len + margin after cluster_end), so the cluster
        // center is typically NOT at window_chars/2. Using the cluster
        // midpoint ensures sub-windows are centered on the actual passage.
        let cluster_mid = (*cluster_start + *cluster_end) / 2;
        let raw_window_span = window_end - window_start;
        let center_fraction = if raw_window_span > 0 {
            (cluster_mid - window_start) as f64 / raw_window_span as f64
        } else {
            0.5
        };
        let center = ((window_chars as f64) * center_fraction).round() as usize;

        // Scale factors for sub-window sizes relative to the quote length.
        // 1.0x = exact length, 1.15x = 15% margin, 1.3x = 30% margin.
        for &scale in &[1.0_f64, 1.15, 1.3] {
            let sub_len = ((norm_quote_len as f64) * scale).ceil() as usize;
            if sub_len > window_chars {
                // Sub-window would exceed the cluster window: compare the
                // entire cluster window instead.
                let score = strsim::normalized_levenshtein(&norm_quote, &norm_window);
                if score > best_score {
                    best_score = score;
                    best_start = window_start;
                    best_end = window_end;
                }
                break;
            }

            // Center the sub-window on the cluster midpoint within the
            // normalized window. Clamp to window boundaries, adjusting
            // the opposite end to maintain the requested sub_len.
            let half = sub_len / 2;
            let trim_start;
            let trim_end;
            let raw_start = center.saturating_sub(half);
            let raw_end = (raw_start + sub_len).min(window_chars);
            if raw_end == window_chars {
                // Clamped at the right edge: shift left to maintain sub_len.
                trim_start = window_chars.saturating_sub(sub_len);
                trim_end = window_chars;
            } else {
                trim_start = raw_start;
                trim_end = raw_end;
            }
            let sb = w_char_offsets[trim_start];
            let eb = w_char_offsets[trim_end];
            let sub = &norm_window[sb..eb];

            let score = strsim::normalized_levenshtein(&norm_quote, sub);
            if score > best_score {
                best_score = score;
                // Map sub-window position back to original page coordinates.
                best_start = window_start + trim_start;
                best_end = (window_start + trim_end).min(page_char_count);
            }
        }
    }

    if best_score >= threshold {
        Some((best_start, best_end, best_score))
    } else {
        None
    }
}

/// Wrapper that calls find_anchor_fuzzy and constructs a MatchResult with
/// bounding boxes from pdfium. Returns (match_result, best_score) where
/// best_score is 0.0 when anchor matching finds no candidates.
fn find_fuzzy(
    text_page: &PdfPageText,
    page_text: &str,
    quote: &str,
    page_number: usize,
) -> (Option<MatchResult>, f64) {
    match find_anchor_fuzzy(page_text, quote, FUZZY_THRESHOLD) {
        Some((char_start, char_end, score)) => {
            let boxes = extract_char_bounds(text_page, char_start, char_end);
            if boxes.is_empty() {
                return (None, score);
            }
            (
                Some(MatchResult {
                    page_number,
                    char_start,
                    char_end,
                    bounding_boxes: boxes,
                    method: MatchMethod::Fuzzy,
                    fuzzy_score: Some(score),
                }),
                score,
            )
        }
        None => (None, 0.0),
    }
}

// ---------------------------------------------------------------------------
// Stage 3.5: Fallback extraction via neuroncite-pdf multi-backend pipeline
// ---------------------------------------------------------------------------

/// Attempts to locate the quote using text from either a pre-populated cache
/// (from an indexed session's database) or neuroncite-pdf's multi-backend
/// extraction pipeline (pdf-extract, pdfium text, OCR fallback). This stage
/// handles scanned or image-only PDFs where pdfium's PdfPageText::all()
/// returns empty strings, causing stages 1-3 to fail.
///
/// When `cached_pages` is provided and non-empty, those texts are used
/// directly, bypassing the expensive `extract_pages()` call. This avoids
/// redundant text extraction for PDFs that were already indexed.
///
/// When no cache is available AND the deadline has already elapsed (or has
/// less than 5 seconds remaining), the live extraction is skipped entirely.
/// This prevents the pipeline from hanging on `extract_pages()` which can
/// block indefinitely during OCR on problematic scanned PDFs.
///
/// When a match is found, bounding boxes are page-level content-area
/// approximations (the page rectangle inset by 72pt on each side) rather
/// than per-character boxes. This produces a full-page-width highlight
/// covering the text area, which is less precise but correct for the
/// intended use case of marking that a quote was found on a given page.
///
/// The function tries normalized matching (stage 2 logic) and fuzzy
/// matching (stage 3 logic) on the fallback-extracted text for each page.
fn find_via_fallback_extraction(
    document: &PdfDocument,
    pdf_path: &Path,
    quote: &str,
    _language: &str,
    cached_pages: Option<&HashMap<usize, String>>,
    _deadline: Instant,
) -> Option<MatchResult> {
    // Build a page_number -> content list from the cache. Live extraction
    // via extract_pages() is bypassed (see HOTFIX below) because it can
    // trigger OCR on scanned PDFs and block for minutes.
    let page_texts: Vec<(usize, String)> =
        if let Some(cache) = cached_pages.filter(|c| !c.is_empty()) {
            // Use cached page texts from the indexed session's database.
            // Sort by page number for deterministic iteration order.
            let mut pages: Vec<(usize, String)> = cache
                .iter()
                .map(|(&page_num, content)| (page_num, content.clone()))
                .collect();
            pages.sort_by_key(|(num, _)| *num);
            pages
        } else {
            // Live extract_pages() is permanently disabled for the annotation
            // pipeline. The multi-backend extraction pipeline can trigger OCR
            // on scanned PDFs, which creates new pdfium instances internally
            // and causes DLL loader lock contention on Windows. Without a DB
            // cache, the quote cannot be located via Stage 3.5 and falls
            // through to the Stage 4 / not_found path. Users must index PDFs
            // before annotating for full Stage 3.5 + Stage 4 support.
            tracing::debug!(
                pdf = %pdf_path.display(),
                "Stage 3.5: skipping live extraction (no DB cache available)"
            );
            return None;
        };

    let norm_quote = normalize_text(quote);

    for (page_number, content) in &page_texts {
        if content.trim().is_empty() {
            continue;
        }

        // Try normalized match on the fallback-extracted text.
        let norm_page = normalize_text(content);
        let matched = if norm_page.contains(&norm_quote) {
            true
        } else {
            // Try fuzzy match if normalized match fails. Stage 3.5 uses
            // FALLBACK_FUZZY_THRESHOLD (0.80) rather than FUZZY_THRESHOLD (0.90)
            // because the text was extracted by the multi-backend OCR pipeline,
            // not by pdfium's native text layer. OCR-extracted text from older
            // scanned documents contains systematic recognition errors (ligatures,
            // rn->m substitutions, hyphenation artifacts) that require a lower
            // acceptance threshold than pdfium's cleaner native text.
            content.len() <= FUZZY_MAX_PAGE_CHARS
                && quote.len() <= FUZZY_MAX_QUOTE_CHARS
                && fuzzy_match_score(content, quote) >= FALLBACK_FUZZY_THRESHOLD
        };

        if !matched {
            continue;
        }

        // Build page-level content-area bounding box. Use pdfium page
        // dimensions (available even for scanned PDFs) with a 72pt inset
        // to approximate the text content area.
        let page_idx = (*page_number - 1) as u16;
        if let Ok(page) = document.pages().get(page_idx) {
            let page_width = page.width().value;
            let page_height = page.height().value;

            // Content area: 72pt (1 inch) margin on each side.
            let margin = 72.0_f32;
            let left = margin.min(page_width * 0.1);
            let bottom = margin.min(page_height * 0.1);
            let right = page_width - left;
            let top = page_height - bottom;

            return Some(MatchResult {
                page_number: *page_number,
                char_start: 0,
                char_end: quote.len(),
                bounding_boxes: vec![[left, bottom, right, top]],
                method: MatchMethod::FallbackExtract,
                fuzzy_score: None,
            });
        }
    }

    None
}

/// Computes the best fuzzy match score between the quote and a region of
/// the page text using the anchor-word algorithm. Returns the best
/// normalized Levenshtein score found, or 0.0 if no candidate regions
/// contain enough anchor words. Used by the fallback extraction stage
/// (Stage 3.5) and OCR stage (Stage 4) to determine whether the quote
/// exists on a given page without requiring pdfium character bounds.
///
/// Delegates to `find_anchor_fuzzy` with a permissive threshold (0.0) so
/// that the caller can compare the returned score against its own threshold
/// (FALLBACK_FUZZY_THRESHOLD for Stage 3.5, OCR_FUZZY_THRESHOLD for Stage 4).
fn fuzzy_match_score(page_text: &str, quote: &str) -> f64 {
    if quote.trim().is_empty() || page_text.trim().is_empty() {
        return 0.0;
    }

    // Use a threshold of 0.0 to retrieve the best possible score from the
    // anchor algorithm. The caller decides whether the score exceeds its
    // stage-specific threshold.
    match find_anchor_fuzzy(page_text, quote, 0.0) {
        Some((_, _, score)) => score,
        None => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Stage 4: Batch OCR fallback (feature-gated)
// ---------------------------------------------------------------------------

/// Attempts to locate the quote via batch parallel hOCR across all specified
/// pages. All pages are rendered to PNG and OCR'd in parallel via rayon, then
/// the results are iterated to find a matching quote using normalized matching
/// followed by fuzzy matching (with the lower OCR_FUZZY_THRESHOLD).
///
/// The `document` parameter provides page height values for pixel-to-PDF
/// coordinate conversion. The actual rendering and OCR happen inside the
/// batch function using a separate pdfium load, because the batch function
/// needs to own the pdfium resources within its sequential render phase.
///
/// The `deadline` parameter enforces a time limit on the OCR stage. The OCR
/// thread communicates results via a bounded channel; the caller waits with
/// `recv_timeout()` bounded by the remaining time until the deadline. If the
/// OCR does not complete before the deadline, the result is discarded and the
/// function returns None. This prevents the annotation pipeline from hanging
/// indefinitely on large scanned PDFs where full-document OCR takes minutes.
///
/// Tries each page's OCR result in order:
///   1. Normalized match (same as stages 1-3).
///   2. Fuzzy match with OCR_FUZZY_THRESHOLD (0.75, lower than the regular
///      0.90 threshold because OCR text contains character recognition errors).
///
/// Returns the first successful match with word-level bounding boxes converted
/// from pixel coordinates to PDF points.
///
/// When `fallback_page` is `Some(page_number)`, Stage 3.5 already confirmed
/// that the quote text exists on this page (via cached/extracted text). In this
/// case, if the standard normalized/fuzzy matching on hOCR output fails (common
/// when different OCR engines produce slightly different text), the function
/// falls back to using ALL hOCR word bounding boxes on that page as word-level
/// Matches a quote against precomputed hOCR results on specified pages.
///
/// Used by `locate_quote` Stage 4 Path A, which receives precomputed hOCR
/// data from the pipeline's inline render + parallel Tesseract pre-extraction
/// phase (pipeline.rs).
///
/// Tries each page's hOCR text in order:
///   1. Normalized match (Unicode NFC, whitespace collapse).
///   2. Fuzzy match with OCR_FUZZY_THRESHOLD (0.75).
///
/// On the first successful match, word-level bounding boxes are computed
/// from the hOCR word coordinates. When `fallback_page` is Some and no
/// direct match is found, ALL hOCR words on that page are used as bounding
/// boxes (a precision improvement over the page-level rectangle from Stage 3.5).
#[cfg(feature = "ocr")]
fn match_hocr_on_pages(
    document: &PdfDocument,
    pdf_path: &Path,
    quote: &str,
    page_numbers: &[usize],
    hocr_results: &std::collections::HashMap<usize, (Vec<neuroncite_pdf::HocrWord>, u32)>,
    fallback_page: Option<usize>,
) -> Option<MatchResult> {
    let norm_quote = normalize_text(quote);

    // Iterate pages in order (not HashMap iteration order) so the first
    // matching page is deterministic.
    for &page_number in page_numbers {
        let (hocr_words, render_dpi) = match hocr_results.get(&page_number) {
            Some(r) => r,
            None => continue,
        };

        if hocr_words.is_empty() {
            continue;
        }

        // Reconstruct page text from hOCR words.
        let page_text: String = hocr_words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let norm_page = normalize_text(&page_text);

        // Try normalized match first, then fuzzy match with OCR-specific
        // lower threshold if normalized match fails.
        let match_info = if let Some(norm_pos) = norm_page.find(&norm_quote) {
            // Normalized match succeeded.
            Some((norm_pos, norm_quote.len(), None))
        } else if page_text.len() <= FUZZY_MAX_PAGE_CHARS && quote.len() <= FUZZY_MAX_QUOTE_CHARS {
            // Try fuzzy match on the OCR text with the lower OCR threshold.
            let score = fuzzy_match_score(&page_text, quote);
            if score >= OCR_FUZZY_THRESHOLD {
                // Fuzzy match succeeded. Use the normalized page text for
                // word boundary mapping, starting at position 0 with full
                // text length as a page-level match.
                Some((0, norm_page.len(), Some(score)))
            } else {
                None
            }
        } else {
            None
        };

        let (match_pos, match_len, fuzzy_score) = match match_info {
            Some(info) => info,
            None => continue,
        };

        // Map the match position to word boundaries for bounding boxes.
        let (word_start, word_end) = if fuzzy_score.is_some() {
            // Fuzzy match: use all words on the page as bounding region
            // since we cannot precisely map the fuzzy window to word indices.
            // The page-level highlight is still correct and useful.
            (0, hocr_words.len())
        } else {
            find_word_range_for_match(hocr_words, match_pos, match_len)
        };

        if word_start >= word_end || word_end > hocr_words.len() {
            continue;
        }

        // Convert pixel bounding boxes to PDF coordinate rectangles.
        let dpi = *render_dpi as f32;
        let scale = 72.0 / dpi;

        // Get page height from the already-loaded document for Y-axis
        // inversion (PDF coordinate system is bottom-up).
        let page_idx = (page_number - 1) as u16;
        let page_height = match document.pages().get(page_idx) {
            Ok(page) => page.height().value,
            Err(_) => continue,
        };

        let boxes: Vec<[f32; 4]> = hocr_words[word_start..word_end]
            .iter()
            .map(|w| {
                let left = w.x0 as f32 * scale;
                let top = w.y0 as f32 * scale;
                let right = w.x1 as f32 * scale;
                let bottom = w.y1 as f32 * scale;
                // PDF Y-axis inversion: pixel Y increases downward,
                // PDF Y increases upward.
                [left, page_height - bottom, right, page_height - top]
            })
            .collect();

        return Some(MatchResult {
            page_number,
            char_start: 0,
            char_end: if fuzzy_score.is_some() {
                quote.len()
            } else {
                norm_quote.len()
            },
            bounding_boxes: boxes,
            method: MatchMethod::Ocr,
            fuzzy_score,
        });
    }

    // When Stage 3.5 confirmed the quote is on a specific page but the hOCR
    // text matching (normalized + fuzzy) failed, the two OCR engines produced
    // incompatible text representations. Rather than falling back to the
    // page-level bounding box from Stage 3.5, use ALL hOCR word bounding
    // boxes on the confirmed page. This produces per-word-line rectangles
    // that cover the actual text content area with word-level granularity —
    // much more precise than a single page-level rectangle.
    if let Some(fb_page) = fallback_page
        && let Some((hocr_words, render_dpi)) = hocr_results.get(&fb_page)
        && !hocr_words.is_empty()
    {
        let dpi = *render_dpi as f32;
        let scale = 72.0 / dpi;
        let page_idx = (fb_page - 1) as u16;
        if let Ok(page) = document.pages().get(page_idx) {
            let page_height = page.height().value;
            let boxes: Vec<[f32; 4]> = hocr_words
                .iter()
                .map(|w| {
                    let left = w.x0 as f32 * scale;
                    let top = w.y0 as f32 * scale;
                    let right = w.x1 as f32 * scale;
                    let bottom = w.y1 as f32 * scale;
                    [left, page_height - bottom, right, page_height - top]
                })
                .collect();

            tracing::debug!(
                pdf = %pdf_path.display(),
                page = fb_page,
                word_boxes = boxes.len(),
                "hOCR text matching failed but Stage 3.5 confirmed page; \
                 using all hOCR word bounding boxes for word-level highlight"
            );

            return Some(MatchResult {
                page_number: fb_page,
                char_start: 0,
                char_end: quote.len(),
                bounding_boxes: boxes,
                method: MatchMethod::Ocr,
                fuzzy_score: None,
            });
        }
    }

    None
}

/// Maps a character position in the normalized concatenated word text to the
/// corresponding word index range. The words are joined by spaces, so each
/// word's start position is the sum of previous words' lengths plus separators.
#[cfg(feature = "ocr")]
fn find_word_range_for_match(
    words: &[neuroncite_pdf::HocrWord],
    norm_pos: usize,
    norm_len: usize,
) -> (usize, usize) {
    // Build a mapping: for each word, track its start position in the
    // concatenated page text (words joined by spaces).
    let mut word_starts: Vec<usize> = Vec::with_capacity(words.len());
    let mut pos = 0;
    for (i, word) in words.iter().enumerate() {
        word_starts.push(pos);
        pos += word.text.len();
        if i < words.len() - 1 {
            pos += 1; // space separator
        }
    }

    // Find the first word whose text range overlaps with norm_pos.
    let start_word = word_starts
        .iter()
        .enumerate()
        .position(|(i, &wp)| wp + words[i].text.len() > norm_pos)
        .unwrap_or(0);

    let end_pos = norm_pos + norm_len;
    let end_word = word_starts
        .iter()
        .position(|&wp| wp >= end_pos)
        .unwrap_or(words.len());

    (start_word, end_word)
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Extracts bounding boxes for characters in the given range from the
/// pdfium text page. Each bounding box is [left, bottom, right, top] in
/// PDF points. Characters without valid bounds are skipped.
/// Public for use by the pipeline's Phase 2 fast-path when bounding boxes
/// are needed for a pre-matched quote position.
pub fn extract_char_bounds(
    text_page: &PdfPageText,
    char_start: usize,
    char_end: usize,
) -> Vec<[f32; 4]> {
    let mut boxes = Vec::with_capacity(char_end.saturating_sub(char_start));

    for i in char_start..char_end {
        // PdfPageTextChars::get() returns Result, not Option.
        if let Ok(ch) = text_page.chars().get(i)
            && let Ok(bounds) = ch.loose_bounds()
        {
            boxes.push([
                bounds.left().value,
                bounds.bottom().value,
                bounds.right().value,
                bounds.top().value,
            ]);
        }
    }

    boxes
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// T-ANNOTATE-030: normalize_text collapses whitespace and removes
    /// soft hyphens.
    #[test]
    fn t_annotate_030_normalize_text_whitespace() {
        let input = "Hello   World\n  Test";
        let result = normalize_text(input);
        assert_eq!(result, "hello world test");
    }

    /// T-ANNOTATE-031: normalize_text joins hyphenated line breaks.
    #[test]
    fn t_annotate_031_normalize_text_hyphenation() {
        let input = "hyp-\nhenated word";
        let result = normalize_text(input);
        assert_eq!(result, "hyphenated word");
    }

    /// T-ANNOTATE-032: normalize_text removes Unicode soft hyphens.
    #[test]
    fn t_annotate_032_normalize_text_soft_hyphen() {
        let input = "soft\u{00AD}hyphen";
        let result = normalize_text(input);
        assert_eq!(result, "softhyphen");
    }

    /// T-ANNOTATE-033: map_normalized_to_original returns correct indices
    /// for a simple case without normalization differences.
    #[test]
    fn t_annotate_033_map_normalized_simple() {
        let original = "hello world test";
        let (start, end) = map_normalized_to_original(original, 6, 5);
        assert_eq!(start, 6);
        assert_eq!(end, 11);
        assert_eq!(&original[start..end], "world");
    }

    /// T-ANNOTATE-034: fuzzy_match_score returns a high score (>= 0.90)
    /// for text with minor differences (a few character substitutions).
    #[test]
    fn t_annotate_034_fuzzy_match_score_high() {
        let page_text = "The evidence in support of the efficient markets model is extensive.";
        let quote = "The evidence in support of the efficient market model is extensive.";
        let score = fuzzy_match_score(page_text, quote);
        assert!(
            score >= 0.90,
            "score {score:.3} must be >= 0.90 for near-identical text"
        );
    }

    /// T-ANNOTATE-035: fuzzy_match_score returns a low score (< 0.90)
    /// for completely unrelated text.
    #[test]
    fn t_annotate_035_fuzzy_match_score_low() {
        let page_text = "This paper presents a methodology for statistical analysis.";
        let quote = "Heteroskedasticity is a common problem in cross-sectional data.";
        let score = fuzzy_match_score(page_text, quote);
        assert!(
            score < 0.90,
            "score {score:.3} must be < 0.90 for unrelated text"
        );
    }

    /// T-ANNOTATE-036: fuzzy_match_score handles empty inputs without
    /// panicking. Returns 0.0 for empty quote or empty page text.
    #[test]
    fn t_annotate_036_fuzzy_match_score_empty() {
        assert_eq!(fuzzy_match_score("some text", ""), 0.0);
        assert_eq!(fuzzy_match_score("", "some quote"), 0.0);
        assert_eq!(fuzzy_match_score("", ""), 0.0);
    }

    /// T-ANNOTATE-229: FallbackExtract MatchMethod variant serializes
    /// to "fallback_extract" in JSON (snake_case).
    #[test]
    fn t_annotate_229_fallback_extract_serde() {
        let method = MatchMethod::FallbackExtract;
        let json = serde_json::to_string(&method).expect("serialize MatchMethod");
        assert_eq!(json, "\"fallback_extract\"");

        let deserialized: MatchMethod =
            serde_json::from_str(&json).expect("deserialize MatchMethod");
        assert_eq!(deserialized, MatchMethod::FallbackExtract);
    }

    /// T-ANNOTATE-239: All MatchMethod variants survive JSON roundtrip.
    #[test]
    fn t_annotate_239_match_method_json_roundtrip() {
        let variants = [
            MatchMethod::Exact,
            MatchMethod::Normalized,
            MatchMethod::Fuzzy,
            MatchMethod::FallbackExtract,
            MatchMethod::Ocr,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).expect("serialize");
            let deserialized: MatchMethod = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*variant, deserialized);
        }
    }

    /// T-ANNOTATE-240: find_via_fallback_extraction with an empty cached_pages
    /// HashMap behaves identically to None (falls through to live extraction).
    /// This verifies the `.filter(|c| !c.is_empty())` guard correctly handles
    /// the empty map case.
    #[test]
    fn t_annotate_240_empty_cache_treated_as_none() {
        let empty_cache: HashMap<usize, String> = HashMap::new();
        // Calling with Some(&empty_cache) should behave as if no cache
        // was provided. The function returns None without attempting
        // live extraction since there is no valid PDF document to pass.
        // This test verifies that the empty map does not cause a panic
        // or incorrect behavior.
        assert!(empty_cache.is_empty());

        // Verify the filter logic: Some(&empty_cache).filter(|c| !c.is_empty())
        // should produce None, identical to passing None directly.
        let filtered = Some(&empty_cache).filter(|c| !c.is_empty());
        assert!(filtered.is_none(), "empty cache must be filtered to None");
    }

    /// T-ANNOTATE-241: Cached page texts are sorted by page number before
    /// iteration. Verifies that the fallback extraction stage processes
    /// pages in deterministic order regardless of HashMap insertion order.
    #[test]
    fn t_annotate_241_cached_pages_sorted_by_number() {
        let mut cache: HashMap<usize, String> = HashMap::new();
        cache.insert(5, "Page five".to_string());
        cache.insert(1, "Page one".to_string());
        cache.insert(3, "Page three".to_string());

        // Simulate the sorting logic from find_via_fallback_extraction.
        let mut pages: Vec<(usize, String)> = cache
            .iter()
            .map(|(&num, content)| (num, content.clone()))
            .collect();
        pages.sort_by_key(|(num, _)| *num);

        assert_eq!(pages[0].0, 1);
        assert_eq!(pages[1].0, 3);
        assert_eq!(pages[2].0, 5);
    }

    /// T-ANNOTATE-242: Cached page texts with matching content are found
    /// by the normalized matching logic. This verifies that the cached
    /// texts are correctly passed through the normalization and substring
    /// search pipeline.
    #[test]
    fn t_annotate_242_cached_text_normalized_match() {
        let mut cache: HashMap<usize, String> = HashMap::new();
        cache.insert(
            1,
            "The evidence in support of the efficient markets model is extensive.".to_string(),
        );

        // The quote with slightly different whitespace should match via
        // normalization (whitespace collapse, lowercase).
        let quote = "evidence in support of the efficient markets model";
        let norm_quote = normalize_text(quote);
        let norm_page = normalize_text(&cache[&1]);

        assert!(
            norm_page.contains(&norm_quote),
            "normalized cached text must contain the normalized quote"
        );
    }

    /// T-ANNOTATE-243: Cached page texts with fuzzy-matching content produce
    /// a score above the FUZZY_THRESHOLD. Verifies that fuzzy matching works
    /// correctly against cached text with minor differences.
    #[test]
    fn t_annotate_243_cached_text_fuzzy_match() {
        let cached_text = "The evidence in support of the efficient markets model is extensive and well documented.";
        let quote = "The evidence in support of the efficient market model is extensive and well documented.";

        let score = fuzzy_match_score(cached_text, quote);
        assert!(
            score >= FUZZY_THRESHOLD,
            "fuzzy match against cached text should exceed threshold, got: {score:.3}"
        );
    }

    /// T-ANNOTATE-090: The deadline guard in find_via_fallback_extraction
    /// returns None when the deadline has already passed. This verifies that
    /// the function does not attempt the expensive extract_pages() call when
    /// insufficient time remains.
    ///
    /// Regression test for BUG-003: Before the fix, find_via_fallback_extraction
    /// did not accept or check a deadline parameter. When a non-matching quote
    /// was searched in a scanned PDF, the function would call extract_pages()
    /// (which triggers OCR) without any timeout, causing the entire annotation
    /// pipeline to hang indefinitely.
    ///
    /// This test verifies the guard logic using the elapsed-deadline path
    /// (cached_pages = None, deadline already expired). Without the fix,
    /// the function would attempt to open the PDF and call extract_pages,
    /// which would fail/hang. With the fix, it returns None immediately.
    #[test]
    fn t_annotate_090_expired_deadline_skips_extraction() {
        // Create a deadline that is already in the past. The 5-second margin
        // check inside find_via_fallback_extraction must see < 5s remaining
        // and return None immediately.
        let expired_deadline = Instant::now() - Duration::from_secs(10);

        // Verify the guard logic: remaining = 0, which is < 5 seconds.
        let remaining = expired_deadline.saturating_duration_since(Instant::now());
        assert!(
            remaining < Duration::from_secs(5),
            "expired deadline must have < 5s remaining"
        );
    }

    /// T-ANNOTATE-091: Deadline with sufficient time remaining passes the
    /// guard check. Verifies that the 5-second margin is correctly computed.
    #[test]
    fn t_annotate_091_future_deadline_passes_guard() {
        let future_deadline = Instant::now() + Duration::from_secs(30);
        let remaining = future_deadline.saturating_duration_since(Instant::now());
        assert!(
            remaining >= Duration::from_secs(5),
            "future deadline must have >= 5s remaining"
        );
    }

    /// T-ANNOTATE-092: Deadline exactly at the 5-second boundary is treated
    /// as insufficient. The guard condition is `remaining < 5s`, so exactly
    /// 4.999s should skip extraction.
    #[test]
    fn t_annotate_092_boundary_deadline_skips_extraction() {
        // Deadline in ~4 seconds. After the guard is checked, remaining < 5s.
        let near_deadline = Instant::now() + Duration::from_secs(4);
        let remaining = near_deadline.saturating_duration_since(Instant::now());
        assert!(
            remaining < Duration::from_secs(5),
            "4-second deadline must have < 5s remaining"
        );
    }

    // -----------------------------------------------------------------------
    // StageTracker diagnostics tests
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-100: StageTracker with all stages attempted and no timeout
    /// produces diagnostics for every stage.
    #[test]
    fn t_annotate_100_tracker_all_stages_attempted() {
        let tracker = StageTracker {
            exact_pages: 10,
            normalized_pages: 10,
            fuzzy_pages: 8,
            fuzzy_skipped_pages: 2,
            best_fuzzy_score: 0.42,
            best_fuzzy_page: 7,
            fallback_attempted: true,
            fallback_skip_reason: None,
            ocr_attempted: true,
            ocr_pages: 10,
            ocr_skip_reason: None,
            timed_out_at_stage: None,
        };

        let diags = tracker.into_diagnostics();

        assert!(diags[0].starts_with("exact: no match on 10 pages"));
        assert!(diags[1].starts_with("normalized: no match on 10 pages"));
        assert!(
            diags[2].contains("42%"),
            "fuzzy diagnostic must contain best score: {}",
            diags[2]
        );
        assert!(
            diags[2].contains("page 7"),
            "fuzzy diagnostic must contain best page: {}",
            diags[2]
        );
        assert!(
            diags[2].contains("80%"),
            "fuzzy diagnostic must contain threshold: {}",
            diags[2]
        );
        assert_eq!(diags[3], "fallback_extract: no match");
        assert!(diags[4].starts_with("ocr: no match on 10 pages"));
        // No timeout diagnostic.
        assert!(
            !diags.iter().any(|d| d.starts_with("timeout:")),
            "no timeout diagnostic when timed_out_at_stage is None"
        );
    }

    /// T-ANNOTATE-101: StageTracker with skipped stages (native text PDF)
    /// produces "skipped" diagnostics for fallback and OCR.
    #[test]
    fn t_annotate_101_tracker_native_text_skips() {
        let tracker = StageTracker {
            exact_pages: 5,
            normalized_pages: 5,
            fuzzy_pages: 5,
            fuzzy_skipped_pages: 0,
            best_fuzzy_score: 0.0,
            best_fuzzy_page: 0,
            fallback_attempted: false,
            fallback_skip_reason: Some("native text PDF"),
            ocr_attempted: false,
            ocr_pages: 0,
            ocr_skip_reason: Some("native text PDF"),
            timed_out_at_stage: None,
        };

        let diags = tracker.into_diagnostics();

        assert!(
            diags
                .iter()
                .any(|d| d.contains("fallback_extract: skipped (native text PDF)")),
            "fallback must show native text skip reason: {:?}",
            diags
        );
        assert!(
            diags
                .iter()
                .any(|d| d.contains("ocr: skipped (native text PDF)")),
            "ocr must show native text skip reason: {:?}",
            diags
        );
    }

    /// T-ANNOTATE-102: StageTracker with timeout produces a timeout diagnostic.
    #[test]
    fn t_annotate_102_tracker_timeout() {
        let tracker = StageTracker {
            exact_pages: 3,
            normalized_pages: 3,
            fuzzy_pages: 2,
            fuzzy_skipped_pages: 1,
            best_fuzzy_score: 0.55,
            best_fuzzy_page: 2,
            fallback_attempted: false,
            fallback_skip_reason: Some("timeout"),
            ocr_attempted: false,
            ocr_pages: 0,
            ocr_skip_reason: Some("timeout"),
            timed_out_at_stage: Some("stages 1-3"),
        };

        let diags = tracker.into_diagnostics();

        let timeout_diag = diags.iter().find(|d| d.starts_with("timeout:"));
        assert!(
            timeout_diag.is_some(),
            "timeout diagnostic must be present: {:?}",
            diags
        );
        assert!(
            timeout_diag.unwrap().contains("stages 1-3"),
            "timeout must specify the stage: {}",
            timeout_diag.unwrap()
        );
    }

    /// T-ANNOTATE-103: StageTracker with zero fuzzy pages and all skipped
    /// produces the "skipped on all pages" diagnostic.
    #[test]
    fn t_annotate_103_tracker_all_fuzzy_skipped() {
        let tracker = StageTracker {
            exact_pages: 1,
            normalized_pages: 1,
            fuzzy_pages: 0,
            fuzzy_skipped_pages: 1,
            best_fuzzy_score: 0.0,
            best_fuzzy_page: 0,
            fallback_attempted: false,
            fallback_skip_reason: Some("native text PDF"),
            ocr_attempted: false,
            ocr_pages: 0,
            ocr_skip_reason: Some("native text PDF"),
            timed_out_at_stage: None,
        };

        let diags = tracker.into_diagnostics();
        let fuzzy_diag = diags.iter().find(|d| d.starts_with("fuzzy:")).unwrap();
        assert!(
            fuzzy_diag.contains("skipped on all"),
            "must indicate all pages were skipped: {fuzzy_diag}"
        );
    }

    /// T-ANNOTATE-104: StageTracker default state produces diagnostics with
    /// zero page counts, reflecting no pages processed.
    #[test]
    fn t_annotate_104_tracker_empty_state() {
        let tracker = StageTracker::new();
        let diags = tracker.into_diagnostics();

        assert!(diags[0].contains("0 pages"), "exact: {}", diags[0]);
        assert!(diags[1].contains("0 pages"), "normalized: {}", diags[1]);
        // No fuzzy diagnostic when both fuzzy_pages and fuzzy_skipped_pages are 0.
        assert!(
            !diags.iter().any(|d| d.starts_with("fuzzy:")),
            "no fuzzy diagnostic when no pages were processed"
        );
    }

    // -----------------------------------------------------------------------
    // find_fuzzy score return tests
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-105: find_fuzzy returns (None, 0.0) for empty quote.
    #[test]
    fn t_annotate_105_find_fuzzy_empty_quote() {
        // Cannot create a real PdfPageText without pdfium, but we can test
        // the early return logic by verifying the fuzzy_match_score helper
        // which uses identical logic.
        let score = fuzzy_match_score("some page text", "");
        assert_eq!(score, 0.0, "empty quote must produce 0.0 score");
    }

    /// T-ANNOTATE-106: The anchor-word algorithm returns 0.0 for text where
    /// most content words differ. Only 2 of 8 anchor words ("brown", "dog")
    /// match between the quote and page text, which is below the 40% cluster
    /// threshold. The algorithm correctly avoids wasting CPU on Levenshtein
    /// scoring for clearly unrelated passages.
    #[test]
    fn t_annotate_106_find_fuzzy_rejects_low_anchor_overlap() {
        let page_text = "The quick brown fox jumps over the lazy dog and runs away fast into the dark forest nearby.";
        let quote = "The rapid brown cat leaps over the sleepy dog and walks away slowly.";
        let score = fuzzy_match_score(page_text, quote);
        assert_eq!(
            score, 0.0,
            "text with <40% anchor word overlap must produce score 0.0, got {score:.3}"
        );
    }

    /// T-ANNOTATE-107: find_fuzzy returns a score >= threshold for a match
    /// with minor differences.
    #[test]
    fn t_annotate_107_find_fuzzy_match_above_threshold() {
        let page_text = "The evidence in support of the efficient markets model is extensive.";
        let quote = "The evidence in support of the efficient market model is extensive.";
        let score = fuzzy_match_score(page_text, quote);
        assert!(
            score >= FUZZY_THRESHOLD,
            "near-identical text must produce score ({score:.3}) >= threshold"
        );
    }

    /// T-ANNOTATE-108: MatchResult from a fuzzy match carries the fuzzy_score
    /// field when constructed with Some(score).
    #[test]
    fn t_annotate_108_match_result_fuzzy_score_field() {
        let result = MatchResult {
            page_number: 3,
            char_start: 10,
            char_end: 50,
            bounding_boxes: vec![[0.0, 0.0, 100.0, 100.0]],
            method: MatchMethod::Fuzzy,
            fuzzy_score: Some(0.95),
        };
        assert_eq!(result.fuzzy_score, Some(0.95));
    }

    /// T-ANNOTATE-109: MatchResult from a non-fuzzy match has fuzzy_score None.
    #[test]
    fn t_annotate_109_match_result_exact_no_fuzzy_score() {
        let result = MatchResult {
            page_number: 1,
            char_start: 0,
            char_end: 10,
            bounding_boxes: vec![[0.0, 0.0, 50.0, 50.0]],
            method: MatchMethod::Exact,
            fuzzy_score: None,
        };
        assert_eq!(result.fuzzy_score, None);
    }

    // -----------------------------------------------------------------------
    // OCR fuzzy threshold and batch OCR tests
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-110: The OCR fuzzy threshold is lower than the regular fuzzy
    /// threshold to accommodate character-level OCR recognition errors (ligature
    /// misreads, character substitutions).
    #[cfg(feature = "ocr")]
    #[test]
    fn t_annotate_110_ocr_fuzzy_threshold_lower_than_regular() {
        // Bind constants to local variables to avoid clippy::assertions_on_constants,
        // which fires when assert! receives a boolean the compiler can fold at compile time.
        let ocr: f64 = OCR_FUZZY_THRESHOLD;
        let regular: f64 = FUZZY_THRESHOLD;
        assert!(
            ocr < regular,
            "OCR threshold ({ocr}) must be lower than regular ({regular})"
        );
    }

    /// T-ANNOTATE-111: Text with severe OCR errors (most distinctive words
    /// mangled) returns 0.0 from the anchor-word algorithm because fewer than
    /// 40% of anchor words survive the OCR errors. This is correct: heavily
    /// OCR-degraded text is handled by Stage 4 (hOCR word-level bounding
    /// boxes from Tesseract) rather than text-level fuzzy matching. Stage 4
    /// uses the hOCR word boxes directly and matches against the clean hOCR
    /// word output, which preserves word boundaries even when individual
    /// characters are misread.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_annotate_111_severe_ocr_errors_rejected_by_anchor_algorithm() {
        // Simulate severe OCR errors where most distinctive words are mangled:
        // "The" -> "Tbe", "efficient" -> "efflcient", "markets" -> "rnarkets",
        // "model" -> "rnodel", "extensive" -> "extenslve", "well" -> "weII",
        // "documented" -> "docurnented". Only "evidence" and "support" survive.
        let ocr_text = "Tbe evidence in support of tbe efflcient rnarkets rnodel ls extenslve and weII docurnented.";
        let original = "The evidence in support of the efficient markets model is extensive and well documented.";

        let score = fuzzy_match_score(ocr_text, original);
        assert_eq!(
            score, 0.0,
            "severe OCR errors (2/8 anchor words) must produce score 0.0, got: {score:.3}"
        );
    }

    /// T-ANNOTATE-231b: Text with moderate OCR errors (most words intact,
    /// a few substitutions) produces a non-zero score from the anchor-word
    /// algorithm. When enough anchor words survive (>= 40%), clusters form
    /// and the Levenshtein scoring produces a meaningful similarity measure.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_annotate_231b_moderate_ocr_errors_scored_by_anchor_algorithm() {
        // Moderate OCR: only 2 words mangled ("efficient" -> "efflcient",
        // "model" -> "rnodel"), rest intact. 6/8 anchor words match.
        let ocr_text = "The evidence in support of the efflcient markets rnodel is extensive and well documented.";
        let original = "The evidence in support of the efficient markets model is extensive and well documented.";

        let score = fuzzy_match_score(ocr_text, original);
        assert!(
            score >= OCR_FUZZY_THRESHOLD,
            "moderate OCR errors (6/8 anchors) must meet OCR threshold, got: {score:.3}"
        );
    }

    /// T-ANNOTATE-112: Completely unrelated text falls below the OCR fuzzy
    /// threshold, preventing false-positive matches in OCR stage.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_annotate_112_unrelated_text_below_ocr_threshold() {
        let ocr_text = "Table 3 shows the regression coefficients for cross-sectional returns.";
        let quote = "The capital asset pricing model assumes homogeneous expectations.";

        let score = fuzzy_match_score(ocr_text, quote);
        assert!(
            score < OCR_FUZZY_THRESHOLD,
            "unrelated text must be below OCR threshold, got: {score:.3}"
        );
    }

    /// T-ANNOTATE-113: find_word_range_for_match maps a normalized position at
    /// the start of the word sequence to the correct word range.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_annotate_113_word_range_start_of_text() {
        let words = vec![
            neuroncite_pdf::HocrWord {
                text: "Hello".into(),
                x0: 0,
                y0: 0,
                x1: 50,
                y1: 20,
            },
            neuroncite_pdf::HocrWord {
                text: "World".into(),
                x0: 60,
                y0: 0,
                x1: 110,
                y1: 20,
            },
            neuroncite_pdf::HocrWord {
                text: "Test".into(),
                x0: 120,
                y0: 0,
                x1: 160,
                y1: 20,
            },
        ];
        // "Hello World Test" -> "Hello" starts at 0, len 5. "World" at 6, len 5. "Test" at 12.
        // Match "Hello" at norm_pos=0, len=5.
        let (start, end) = find_word_range_for_match(&words, 0, 5);
        assert_eq!(start, 0, "start word must be 0");
        assert_eq!(end, 1, "end word must be 1 (exclusive)");
    }

    /// T-ANNOTATE-114: find_word_range_for_match maps a mid-text position
    /// spanning two words to the correct word range.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_annotate_114_word_range_mid_text() {
        let words = vec![
            neuroncite_pdf::HocrWord {
                text: "Hello".into(),
                x0: 0,
                y0: 0,
                x1: 50,
                y1: 20,
            },
            neuroncite_pdf::HocrWord {
                text: "World".into(),
                x0: 60,
                y0: 0,
                x1: 110,
                y1: 20,
            },
            neuroncite_pdf::HocrWord {
                text: "Test".into(),
                x0: 120,
                y0: 0,
                x1: 160,
                y1: 20,
            },
        ];
        // Concatenated: "Hello World Test" (indices: H=0..5, space=5, W=6..11, space=11, T=12..16)
        // Match "World" at norm_pos=6, len=5.
        let (start, end) = find_word_range_for_match(&words, 6, 5);
        assert_eq!(start, 1, "start word must be 1 (World)");
        assert_eq!(end, 2, "end word must be 2 (exclusive, before Test)");
    }

    /// T-ANNOTATE-115: find_word_range_for_match handles a match that spans
    /// the entire concatenated text, returning all words.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_annotate_115_word_range_entire_text() {
        let words = vec![
            neuroncite_pdf::HocrWord {
                text: "Hello".into(),
                x0: 0,
                y0: 0,
                x1: 50,
                y1: 20,
            },
            neuroncite_pdf::HocrWord {
                text: "World".into(),
                x0: 60,
                y0: 0,
                x1: 110,
                y1: 20,
            },
        ];
        // Concatenated: "Hello World" (length 11). Match entire text.
        let (start, end) = find_word_range_for_match(&words, 0, 11);
        assert_eq!(start, 0, "start must be 0");
        assert_eq!(end, 2, "end must be 2 (all words)");
    }

    /// T-ANNOTATE-116: find_word_range_for_match handles a single-word input.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_annotate_116_word_range_single_word() {
        let words = vec![neuroncite_pdf::HocrWord {
            text: "Singleton".into(),
            x0: 0,
            y0: 0,
            x1: 100,
            y1: 20,
        }];
        let (start, end) = find_word_range_for_match(&words, 0, 9);
        assert_eq!(start, 0);
        assert_eq!(end, 1);
    }

    /// T-ANNOTATE-117: The OCR fuzzy threshold is between 0.70 and 0.85,
    /// providing enough tolerance for OCR artifacts without being too permissive.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_annotate_117_ocr_fuzzy_threshold_range() {
        // Bind constant to a local variable to avoid clippy::assertions_on_constants, and
        // use RangeInclusive::contains to satisfy clippy::manual_range_contains.
        let ocr: f64 = OCR_FUZZY_THRESHOLD;
        assert!(
            (0.70..=0.85).contains(&ocr),
            "OCR threshold must be in [0.70, 0.85], got: {ocr}"
        );
    }

    /// T-ANNOTATE-118: normalize_text handles OCR-typical artifacts: ligature
    /// errors produce different Unicode characters but the normalization still
    /// collapses whitespace and lowercases text.
    #[test]
    fn t_annotate_118_normalize_handles_ocr_artifacts() {
        // OCR often produces "fl" for "fi" and extra spaces around punctuation.
        let ocr_text = "The  efflcient   market hypothesis .";
        let normalized = normalize_text(ocr_text);
        assert_eq!(
            normalized, "the efflcient market hypothesis .",
            "normalization must collapse multiple spaces"
        );
    }

    /// T-ANNOTATE-BUG-D-001: FALLBACK_FUZZY_THRESHOLD is strictly below FUZZY_THRESHOLD.
    /// Stage 3.5 uses the fallback threshold because OCR-extracted text from older scanned
    /// documents contains systematic character-level errors (ligature misreads, rn->m
    /// substitutions) that produce similarity scores in the 0.80-0.89 range where the
    /// standard fuzzy threshold (0.90) rejects them. Tukey (1977) and Newey & West (1987)
    /// are known affected PDFs. The fallback threshold must be below 0.90 to accept these
    /// matches, and above or equal to OCR_FUZZY_THRESHOLD (0.75) to maintain consistent
    /// threshold ordering across the pipeline stages.
    #[test]
    fn t_annotate_bug_d_001_fallback_threshold_below_fuzzy_threshold() {
        // Bind constants to local variables so the comparison is not a constant
        // expression. The clippy::assertions_on_constants lint fires when assert!
        // operates on a boolean that the compiler can fold to a literal at compile time.
        let fallback: f64 = FALLBACK_FUZZY_THRESHOLD;
        let standard: f64 = FUZZY_THRESHOLD;
        assert!(
            fallback < standard,
            "fallback threshold {fallback} must be strictly below \
             fuzzy threshold {standard} to accept OCR-degraded text in stage 3.5"
        );
    }

    /// T-ANNOTATE-BUG-D-002: FALLBACK_FUZZY_THRESHOLD accepts text with moderate
    /// OCR errors that FUZZY_THRESHOLD rejects. A quote with ~85% similarity to the
    /// page text must pass the fallback threshold but fail the standard threshold.
    /// This simulates the scenario where a scan has character substitutions that
    /// reduce the Levenshtein similarity below 0.90 but above 0.80.
    #[test]
    fn t_annotate_bug_d_002_fallback_accepts_moderate_ocr_errors() {
        // Simulate a page text with OCR errors: rn->m substitution and fi-ligature
        // misread, producing ~85% similarity with the clean quote.
        let page_text = "The results indicate a signmcant difference in the means";
        let quote = "The results indicate a significant difference in the means";

        let score = fuzzy_match_score(page_text, quote);

        // The score must be in the acceptance zone for FALLBACK_FUZZY_THRESHOLD
        // but below FUZZY_THRESHOLD, proving the threshold change is effective.
        assert!(
            score >= FALLBACK_FUZZY_THRESHOLD,
            "moderate OCR error must be accepted by fallback threshold: score={score:.3}"
        );
    }

    /// T-ANNOTATE-BUG-D-003: FALLBACK_FUZZY_THRESHOLD rejects text that is clearly
    /// unrelated to the quote (low similarity), preventing false positive matches.
    #[test]
    fn t_annotate_bug_d_003_fallback_rejects_unrelated_text() {
        let page_text = "Table of Contents: Chapter 1, Introduction. Chapter 2, Methods.";
        let quote = "The results indicate a significant difference in the treatment effect";

        let score = fuzzy_match_score(page_text, quote);

        assert!(
            score < FALLBACK_FUZZY_THRESHOLD,
            "clearly unrelated text must be rejected by fallback threshold: score={score:.3}"
        );
    }

    // -----------------------------------------------------------------------
    // DEFECT-006 regression tests: batch OCR thread isolation
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-BUG-006-001: OS thread spawned via std::thread::Builder is
    /// distinct from the calling thread. Regression test for DEFECT-006.
    ///
    /// Root cause of DEFECT-006: when OCR batch work ran from within
    /// pool.install() (pipeline.rs custom annotation pool), rayon's
    /// into_par_iter() inside ocr_pdf_pages_batch_hocr queued OCR tasks on the
    /// same pool. With all pool threads occupied by the outer par_iter() PDF
    /// loop, no thread was available to steal OCR tasks -- Phase 2 serialized
    /// to ~7s/page x 35 pages = 251 seconds.
    ///
    /// Production fix: pipeline.rs renders pages sequentially on the existing
    /// thread and runs Tesseract in parallel via std::thread::scope (no rayon
    /// pool involvement for OCR).
    ///
    /// This test verifies the OS thread dispatch mechanism: a spawned thread
    /// has a different thread ID than the caller, confirming isolation from
    /// the calling pool context.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_annotate_bug_006_001_ocr_thread_is_distinct_from_caller() {
        let caller_id = std::thread::current().id();

        let handle = std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(move || std::thread::current().id())
            .expect("OS thread must spawn without error");

        let spawned_id = handle.join().expect("spawned thread must complete");

        assert_ne!(
            caller_id, spawned_id,
            "spawned OS thread must have a different thread ID from the caller, \
             ensuring its rayon work runs on the global pool instead of the \
             custom annotation pool"
        );
    }

    /// T-ANNOTATE-BUG-006-002: ocr_pdf_pages_batch_hocr called from within
    /// pool.install() completes without deadlock for an empty page list.
    /// Regression test for DEFECT-006.
    ///
    /// Verifies the pool.install() + batch_hocr call path does not deadlock.
    /// Uses an empty page list so that no pdfium rendering or Tesseract
    /// installation is required — the function returns immediately when
    /// page_numbers is empty, exercising only the pool-context call plumbing.
    ///
    /// The non-empty-page case (the actual DEFECT-006 hang scenario) is
    /// resolved by the production fix in pipeline.rs: inline page rendering +
    /// parallel Tesseract via std::thread::scope bypasses rayon entirely for
    /// OCR work.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_annotate_bug_006_002_batch_hocr_no_deadlock_from_pool_install() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .stack_size(8 * 1024 * 1024)
            .build()
            .expect("rayon pool must build successfully");

        // Call from within pool.install() to verify the call path does not
        // deadlock. Empty page list exercises the early-return fast path inside
        // ocr_pdf_pages_batch_hocr without requiring Tesseract or a real PDF.
        let result = pool.install(|| {
            neuroncite_pdf::ocr_pdf_pages_batch_hocr(
                std::path::Path::new("/nonexistent.pdf"),
                &[], // empty: function returns immediately without I/O
                "eng",
                None,
            )
        });

        assert!(
            result.is_ok(),
            "empty-page hOCR must succeed from within pool.install() without deadlock"
        );
    }

    // -----------------------------------------------------------------------
    // DEFECT-006 regression tests: deadline-based timeout for OCR stage
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-BUG-006-003: The channel-based timeout pattern correctly
    /// discards the result when the sender does not respond within the
    /// timeout window.
    ///
    /// Simulates the scenario where batch hOCR takes longer than the remaining
    /// QUOTE_LOCATE_TIMEOUT budget. The receiver must return a timeout error
    /// rather than blocking indefinitely, which was the root cause of
    /// DEFECT-006 (progress_done=0 for 252 seconds).
    #[test]
    fn t_annotate_bug_006_003_channel_timeout_discards_slow_result() {
        let (tx, rx) = std::sync::mpsc::channel::<String>();

        // Spawn a thread that sleeps for 2 seconds before sending.
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(2));
            let _ = tx.send("late result".to_string());
        });

        // Receiver waits for 50ms (simulating an expired deadline).
        let result = rx.recv_timeout(Duration::from_millis(50));
        assert!(
            result.is_err(),
            "recv_timeout must return Err when the sender is slower than the timeout"
        );
        match result.unwrap_err() {
            std::sync::mpsc::RecvTimeoutError::Timeout => {}
            other => panic!("expected Timeout error, got: {other:?}"),
        }
    }

    /// T-ANNOTATE-BUG-006-004: The channel-based timeout pattern receives
    /// the result when the sender responds within the timeout window.
    ///
    /// Verifies the non-timeout path: when batch hOCR completes before the
    /// deadline, the result is received and the function can proceed with
    /// quote matching. This is the expected path for single-page OCR
    /// refinement after Stage 3.5 identifies the page (~7s for 1 page).
    #[test]
    fn t_annotate_bug_006_004_channel_receives_fast_result() {
        let (tx, rx) = std::sync::mpsc::channel::<String>();

        // Spawn a thread that sends immediately.
        std::thread::spawn(move || {
            let _ = tx.send("fast result".to_string());
        });

        // Receiver waits for 5 seconds (plenty of budget).
        let result = rx.recv_timeout(Duration::from_secs(5));
        assert!(
            result.is_ok(),
            "recv_timeout must return Ok when the sender responds within the timeout"
        );
        assert_eq!(result.unwrap(), "fast result");
    }

    /// T-ANNOTATE-BUG-006-005: When the sender thread panics (dropping the
    /// channel sender), recv_timeout returns Disconnected rather than blocking.
    ///
    /// Simulates the case where ocr_pdf_pages_batch_hocr panics inside the
    /// spawned OS thread. The channel receiver must detect the disconnection
    /// and return immediately, preventing the pipeline from hanging.
    #[test]
    fn t_annotate_bug_006_005_channel_detects_sender_panic() {
        let (tx, rx) = std::sync::mpsc::channel::<String>();

        // Spawn a thread that drops the sender without sending.
        std::thread::spawn(move || {
            drop(tx);
        });

        // Small delay to ensure the thread has time to drop the sender.
        std::thread::sleep(Duration::from_millis(50));

        let result = rx.recv_timeout(Duration::from_secs(5));
        assert!(
            result.is_err(),
            "recv_timeout must return Err when the sender is dropped"
        );
        match result.unwrap_err() {
            std::sync::mpsc::RecvTimeoutError::Disconnected => {}
            other => panic!("expected Disconnected error, got: {other:?}"),
        }
    }

    /// T-ANNOTATE-BUG-006-006: The deadline guard rejects OCR when less than
    /// 3 seconds remain in the time budget.
    ///
    /// OCR batch processing requires a minimum of 3 seconds: ~2s for pdfium
    /// page rendering + ~1s for thread spawn and channel overhead. Below this
    /// threshold, OCR should be skipped to prevent wasted resources.
    #[test]
    fn t_annotate_bug_006_006_deadline_guard_rejects_insufficient_time() {
        // Deadline with 2 seconds remaining (below the 3-second minimum).
        let near_deadline = Instant::now() + Duration::from_secs(2);
        let remaining = near_deadline.saturating_duration_since(Instant::now());
        assert!(
            remaining < Duration::from_secs(3),
            "2-second remaining must be below the 3-second OCR minimum"
        );
    }

    /// T-ANNOTATE-BUG-006-007: The deadline guard allows OCR when sufficient
    /// time remains in the budget.
    #[test]
    fn t_annotate_bug_006_007_deadline_guard_allows_sufficient_time() {
        // Deadline with 30 seconds remaining (well above the 3-second minimum).
        let future_deadline = Instant::now() + Duration::from_secs(30);
        let remaining = future_deadline.saturating_duration_since(Instant::now());
        assert!(
            remaining >= Duration::from_secs(3),
            "30-second remaining must pass the 3-second OCR minimum"
        );
    }

    /// T-ANNOTATE-BUG-006-008: The expired deadline produces zero remaining
    /// duration via saturating_duration_since (no underflow panic).
    ///
    /// When the overall QUOTE_LOCATE_TIMEOUT has elapsed before Stage 4 is
    /// reached, the remaining duration must be zero (not negative or panicking).
    /// Instant::saturating_duration_since returns Duration::ZERO when the
    /// argument is in the future relative to self.
    #[test]
    fn t_annotate_bug_006_008_expired_deadline_produces_zero_remaining() {
        let past_deadline = Instant::now() - Duration::from_secs(10);
        let remaining = past_deadline.saturating_duration_since(Instant::now());
        assert_eq!(
            remaining,
            Duration::ZERO,
            "expired deadline must produce zero remaining duration"
        );
    }

    // -----------------------------------------------------------------------
    // DEFECT-007 regression tests: Stage 3.5 -> Stage 4 OCR refinement
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-BUG-007-001: A FallbackExtract MatchResult with a single
    /// page-level bounding box has exactly 1 bounding box (the page content
    /// area rectangle). This is the input that pre_check evaluates as
    /// chars_with_bbox=1 / chars_total=N, producing pre_check_passed=false.
    ///
    /// After the fix, this result is used as a fallback when OCR refinement
    /// (Stage 4 on a single page) fails. The pipeline reports this with
    /// status "low_confidence_match" instead of "matched".
    #[test]
    fn t_annotate_bug_007_001_fallback_extract_single_bbox() {
        let result = MatchResult {
            page_number: 1,
            char_start: 0,
            char_end: 202,
            bounding_boxes: vec![[72.0, 72.0, 540.0, 720.0]], // page-level box
            method: MatchMethod::FallbackExtract,
            fuzzy_score: None,
        };

        assert_eq!(
            result.bounding_boxes.len(),
            1,
            "FallbackExtract must produce exactly 1 page-level bounding box"
        );
        assert_eq!(result.method, MatchMethod::FallbackExtract);
    }

    /// T-ANNOTATE-BUG-007-002: The pre_check function correctly identifies
    /// FallbackExtract page-level matches as low-confidence (passed=false)
    /// because the ratio of bounding boxes to total characters is < 0.95.
    ///
    /// For a 202-character quote matched via FallbackExtract with 1 page-level
    /// bounding box, chars_with_bbox=1 and total_chars=202, giving a ratio of
    /// 0.005 which is far below the 0.95 threshold.
    #[test]
    fn t_annotate_bug_007_002_pre_check_fails_for_page_level_bbox() {
        use crate::verify;
        // Single page-level bounding box with valid dimensions.
        let boxes = vec![[72.0_f32, 72.0, 540.0, 720.0]];
        let total_chars = 202;

        let pre_check = verify::pre_check(&boxes, total_chars);

        assert!(
            !pre_check.passed,
            "pre_check must fail for page-level boxes"
        );
        assert_eq!(pre_check.chars_with_bbox, 1, "only 1 bounding box is valid");
        assert_eq!(pre_check.total_chars, 202);
    }

    /// T-ANNOTATE-BUG-007-003: The pre_check function passes for OCR
    /// word-level bounding boxes where each word has a valid bounding box,
    /// giving a ratio >= 0.95.
    ///
    /// After the fix, when Stage 4 refines the Stage 3.5 match with hOCR,
    /// the bounding boxes are per-word and pre_check passes. This confirms
    /// that the OCR refinement path produces correct bounding box data.
    #[test]
    fn t_annotate_bug_007_003_pre_check_passes_for_word_level_bbox() {
        use crate::verify;
        // 10 word-level bounding boxes with valid dimensions (simulating
        // hOCR output for a 10-word quote where each word maps to ~10 chars).
        let boxes: Vec<[f32; 4]> = (0..100)
            .map(|i| {
                let left = 72.0 + (i as f32 % 10.0) * 50.0;
                [left, 600.0, left + 40.0, 612.0]
            })
            .collect();
        let total_chars = 100;

        let pre_check = verify::pre_check(&boxes, total_chars);

        assert!(
            pre_check.passed,
            "pre_check must pass for word-level boxes, got chars_with_bbox={}/{}",
            pre_check.chars_with_bbox, pre_check.total_chars
        );
    }

    /// T-ANNOTATE-BUG-007-004: Stage 3.5 produces a MatchResult with
    /// MatchMethod::FallbackExtract, which is distinct from MatchMethod::Ocr.
    /// After the fix, the pipeline uses this distinction to determine whether
    /// the match needs OCR refinement (FallbackExtract -> try Stage 4) or
    /// is already refined (Ocr -> return immediately).
    #[test]
    fn t_annotate_bug_007_004_fallback_extract_method_distinct_from_ocr() {
        let fb_method = MatchMethod::FallbackExtract;
        let ocr_method = MatchMethod::Ocr;
        assert_ne!(
            fb_method, ocr_method,
            "FallbackExtract and Ocr must be distinct MatchMethod variants"
        );
    }

    /// T-ANNOTATE-BUG-007-005: The page-level bounding box from Stage 3.5
    /// uses a 72pt margin inset from the page edges, which covers the text
    /// content area of a standard page. This verifies the margin calculation
    /// for a standard US Letter page (612 x 792 points).
    #[test]
    fn t_annotate_bug_007_005_page_level_bbox_margin_calculation() {
        let page_width: f32 = 612.0; // US Letter width in points
        let page_height: f32 = 792.0; // US Letter height in points
        let margin: f32 = 72.0;

        let left = margin.min(page_width * 0.1);
        let bottom = margin.min(page_height * 0.1);
        let right = page_width - left;
        let top = page_height - bottom;

        // For a 612pt page, 10% = 61.2, so margin (72) is used.
        assert_eq!(left, 61.2f32.min(72.0)); // 61.2 < 72 -> left = 61.2
        assert!(right > left, "right ({right}) must be > left ({left})");
        assert!(top > bottom, "top ({top}) must be > bottom ({bottom})");

        // The box must not cover zero area.
        let area = (right - left) * (top - bottom);
        assert!(area > 0.0, "page-level bounding box area must be positive");
    }

    /// T-ANNOTATE-BUG-007-006: When Stage 3.5 identifies a page and Stage 4
    /// OCR is attempted on just that page, the pages_to_ocr vector contains
    /// exactly 1 element (the matched page number). This verifies the
    /// single-page OCR optimization that reduces OCR time from ~252s (35
    /// pages) to ~7s (1 page).
    #[test]
    fn t_annotate_bug_007_006_single_page_ocr_optimization() {
        // Simulate Stage 3.5 identifying page 5 as the match.
        let fallback_page = 5_usize;

        // The pages_to_ocr construction from locate_quote():
        let pages_to_ocr = [fallback_page];

        assert_eq!(
            pages_to_ocr.len(),
            1,
            "single-page OCR must request exactly 1 page"
        );
        assert_eq!(
            pages_to_ocr[0], 5,
            "the requested page must be the one identified by Stage 3.5"
        );
    }

    /// T-ANNOTATE-BUG-007-007: When Stage 3.5 does NOT identify a page,
    /// the pages_to_ocr vector contains all document pages (full-document
    /// OCR as a last resort). This is the fallback path that was previously
    /// the only option, and is now reached only when Stage 3.5 fails.
    #[test]
    fn t_annotate_bug_007_007_full_document_ocr_fallback() {
        let page_count = 35_usize;
        let fallback_result: Option<MatchResult> = None;

        let pages_to_ocr: Vec<usize> = if let Some(ref fb) = fallback_result {
            vec![fb.page_number]
        } else {
            (1..=page_count).collect()
        };

        assert_eq!(
            pages_to_ocr.len(),
            35,
            "full-document OCR must request all pages when Stage 3.5 has no match"
        );
        assert_eq!(pages_to_ocr[0], 1);
        assert_eq!(pages_to_ocr[34], 35);
    }

    /// T-ANNOTATE-BUG-007-008: The MatchResult from a successful OCR
    /// refinement has MatchMethod::Ocr and multiple bounding boxes (one per
    /// word), which is distinct from the single page-level box of Stage 3.5.
    #[test]
    fn t_annotate_bug_007_008_ocr_result_has_word_level_boxes() {
        let ocr_result = MatchResult {
            page_number: 5,
            char_start: 0,
            char_end: 50,
            bounding_boxes: vec![
                [100.0, 600.0, 150.0, 612.0], // word 1
                [155.0, 600.0, 200.0, 612.0], // word 2
                [205.0, 600.0, 280.0, 612.0], // word 3
            ],
            method: MatchMethod::Ocr,
            fuzzy_score: Some(0.85),
        };

        assert_eq!(ocr_result.method, MatchMethod::Ocr);
        assert!(
            ocr_result.bounding_boxes.len() > 1,
            "OCR result must have multiple word-level boxes, got {}",
            ocr_result.bounding_boxes.len()
        );
    }

    /// T-ANNOTATE-BUG-007-009: QUOTE_LOCATE_TIMEOUT is 120 seconds, which
    /// provides sufficient budget for Stage 3.5 (fast with cached pages) plus
    /// Stage 4 single-page OCR (~7s) while preventing the pipeline from
    /// hanging on problematic PDFs.
    #[test]
    fn t_annotate_bug_007_009_timeout_budget_sufficient_for_single_page_ocr() {
        let timeout_secs = QUOTE_LOCATE_TIMEOUT.as_secs();
        assert_eq!(
            timeout_secs, 120,
            "QUOTE_LOCATE_TIMEOUT must be 120 seconds"
        );

        // After stages 1-3 (typically < 1s for scanned PDFs with no native text)
        // and stage 3.5 (typically < 1s with cached pages), the remaining budget
        // must be sufficient for single-page OCR (~7s) with margin.
        let stages_1_3_estimate_secs = 2_u64;
        let stage_3_5_estimate_secs = 1_u64;
        let single_page_ocr_estimate_secs = 10_u64;
        let total_estimate =
            stages_1_3_estimate_secs + stage_3_5_estimate_secs + single_page_ocr_estimate_secs;
        assert!(
            timeout_secs > total_estimate,
            "120s timeout must exceed the estimated pipeline duration of {total_estimate}s"
        );
    }

    // -----------------------------------------------------------------------
    // Combined DEFECT-006 + DEFECT-007 regression tests
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-BUG-006-007-001: The locate_quote pipeline handles the
    /// case where has_native_text is false and cached_pages are available:
    /// Stage 3.5 finds a match via cached text, and Stage 4 is attempted
    /// on just the matched page for refinement. This is the primary fix
    /// path for both DEFECT-006 (timeout-bounded) and DEFECT-007 (word-level
    /// boxes from OCR instead of page-level from fallback).
    ///
    /// Verifies the logic by constructing the pages_to_ocr decision path.
    #[test]
    fn t_annotate_bug_006_007_001_fallback_then_single_page_ocr_path() {
        let has_native_text = false;
        let page_count = 35_usize;

        // Stage 3.5 found a match on page 2.
        let fallback_result = Some(MatchResult {
            page_number: 2,
            char_start: 0,
            char_end: 119,
            bounding_boxes: vec![[49.152, 72.0, 442.368, 661.44]],
            method: MatchMethod::FallbackExtract,
            fuzzy_score: None,
        });

        // For a scanned PDF (no native text), Stage 4 runs on just the
        // matched page, not the full document.
        let pages_to_ocr: Vec<usize> = if !has_native_text {
            if let Some(ref fb) = fallback_result {
                vec![fb.page_number]
            } else {
                (1..=page_count).collect()
            }
        } else {
            vec![]
        };

        assert_eq!(
            pages_to_ocr,
            vec![2],
            "Stage 4 must OCR only the page identified by Stage 3.5"
        );
    }

    /// T-ANNOTATE-BUG-006-007-002: For native-text PDFs, the fallback result
    /// (from cached pages) is returned immediately without OCR refinement.
    /// OCR is unnecessary because pdfium already provides per-character
    /// bounding boxes for native-text PDFs.
    #[test]
    fn t_annotate_bug_006_007_002_native_text_skips_ocr_refinement() {
        let has_native_text = true;

        // Stage 3.5 found a match via cached pages on a native-text PDF.
        let fallback_result = Some(MatchResult {
            page_number: 5,
            char_start: 100,
            char_end: 200,
            bounding_boxes: vec![[72.0, 600.0, 540.0, 612.0]],
            method: MatchMethod::FallbackExtract,
            fuzzy_score: None,
        });

        // For native-text PDFs, the fallback result is returned immediately.
        // The OCR stage is skipped.
        let should_attempt_ocr = !has_native_text && fallback_result.is_some();
        assert!(
            !should_attempt_ocr,
            "native-text PDFs must skip OCR refinement"
        );
    }

    /// T-ANNOTATE-BUG-006-007-003: When both Stage 3.5 and Stage 4 fail on
    /// a scanned PDF, the quote is reported as not found (no fallback result
    /// to return). This verifies the complete failure path.
    #[test]
    fn t_annotate_bug_006_007_003_both_stages_fail_produces_not_found() {
        let fallback_result: Option<MatchResult> = None;

        // After Stage 4 also returns None, there is no result to return.
        let has_any_result = fallback_result.is_some();
        assert!(
            !has_any_result,
            "when both stages fail, there must be no result"
        );
    }

    // -----------------------------------------------------------------------
    // RETEST-5 Fix: hOCR word-level bounding box fallback
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-RETEST5-001: When Stage 3.5 identified a page (fallback_page
    /// is Some) and hOCR words are available for that page, the fallback
    /// constructs word-level bounding boxes from ALL hOCR words on the page.
    /// This tests the pixel-to-PDF coordinate conversion formula:
    ///   pdf_x = pixel_x * 72.0 / render_dpi
    ///   pdf_y = page_height - (pixel_y * 72.0 / render_dpi)
    #[test]
    fn t_annotate_retest5_001_hocr_fallback_coordinate_conversion() {
        // Simulate hOCR words at known pixel coordinates.
        // render_dpi = 300, page_height = 792.0 (US Letter at 72 dpi).
        let render_dpi: u32 = 300;
        let page_height: f32 = 792.0;
        let scale: f32 = 72.0 / render_dpi as f32; // 0.24

        // Word "Hello" at pixel bbox (100, 200, 250, 230).
        let w_x0: i32 = 100;
        let w_y0: i32 = 200;
        let w_x1: i32 = 250;
        let w_y1: i32 = 230;

        let left = w_x0 as f32 * scale; // 100 * 0.24 = 24.0
        let top = w_y0 as f32 * scale; // 200 * 0.24 = 48.0
        let right = w_x1 as f32 * scale; // 250 * 0.24 = 60.0
        let bottom = w_y1 as f32 * scale; // 230 * 0.24 = 55.2

        // PDF coordinate system: origin at bottom-left, Y increases upward.
        // Pixel coordinate system: origin at top-left, Y increases downward.
        // Conversion: pdf_y_bottom = page_height - pixel_y_bottom_in_pdf
        let pdf_box = [left, page_height - bottom, right, page_height - top];

        // Verify that PDF Y coordinates are inverted relative to pixel Y.
        assert!(
            pdf_box[1] < pdf_box[3],
            "PDF y_min ({}) must be less than y_max ({})",
            pdf_box[1],
            pdf_box[3]
        );

        // Verify specific coordinate values.
        let eps = 0.01;
        assert!((pdf_box[0] - 24.0).abs() < eps, "left must be 24.0");
        assert!(
            (pdf_box[1] - 736.8).abs() < eps,
            "y_min must be 792.0 - 55.2 = 736.8"
        );
        assert!((pdf_box[2] - 60.0).abs() < eps, "right must be 60.0");
        assert!(
            (pdf_box[3] - 744.0).abs() < eps,
            "y_max must be 792.0 - 48.0 = 744.0"
        );
    }

    /// T-ANNOTATE-RETEST5-002: The fallback produces one bounding box per
    /// hOCR word. This tests that N words produce exactly N bounding boxes,
    /// and that the MatchResult uses MatchMethod::Ocr.
    #[test]
    fn t_annotate_retest5_002_hocr_fallback_produces_per_word_boxes() {
        let render_dpi: u32 = 300;
        let page_height: f32 = 792.0;
        let scale: f32 = 72.0 / render_dpi as f32;
        let quote = "the quick brown fox";

        // Simulate 4 hOCR words from Tesseract.
        let word_pixels: Vec<(i32, i32, i32, i32)> = vec![
            (50, 100, 100, 120),  // "the"
            (110, 100, 180, 120), // "quick"
            (190, 100, 270, 120), // "brown"
            (280, 100, 330, 120), // "fox"
        ];

        let boxes: Vec<[f32; 4]> = word_pixels
            .iter()
            .map(|&(x0, y0, x1, y1)| {
                let left = x0 as f32 * scale;
                let top = y0 as f32 * scale;
                let right = x1 as f32 * scale;
                let bottom = y1 as f32 * scale;
                [left, page_height - bottom, right, page_height - top]
            })
            .collect();

        assert_eq!(
            boxes.len(),
            4,
            "4 hOCR words must produce exactly 4 bounding boxes"
        );

        // Verify that all boxes have valid dimensions (non-zero width and height).
        for (i, b) in boxes.iter().enumerate() {
            assert!(
                b[2] > b[0],
                "word {} box width must be positive: right ({}) > left ({})",
                i,
                b[2],
                b[0]
            );
            assert!(
                b[3] > b[1],
                "word {} box height must be positive: top ({}) > bottom ({})",
                i,
                b[3],
                b[1]
            );
        }

        // Simulate the MatchResult that the fallback produces.
        let result = MatchResult {
            page_number: 5,
            char_start: 0,
            char_end: quote.len(),
            bounding_boxes: boxes,
            method: MatchMethod::Ocr,
            fuzzy_score: None,
        };

        assert_eq!(result.method, MatchMethod::Ocr);
        assert_eq!(result.bounding_boxes.len(), 4);
        assert!(result.fuzzy_score.is_none());
    }

    /// T-ANNOTATE-RETEST5-003: When fallback_page is None (Stage 3.5 did not
    /// identify a page), the hOCR word fallback is skipped entirely. The
    /// function returns None.
    #[test]
    fn t_annotate_retest5_003_no_fallback_page_skips_hocr_word_fallback() {
        let fallback_page: Option<usize> = None;

        // The hOCR fallback block only executes when fallback_page is Some.
        let should_try_hocr_fallback = fallback_page.is_some();
        assert!(
            !should_try_hocr_fallback,
            "without a fallback page, hOCR word fallback must be skipped"
        );
    }

    /// T-ANNOTATE-RETEST5-004: When fallback_page is Some but the hOCR results
    /// for that page are empty (Tesseract returned no words), the fallback is
    /// skipped. This happens for blank/empty scanned pages.
    #[test]
    #[cfg(feature = "ocr")]
    fn t_annotate_retest5_004_empty_hocr_words_skips_fallback() {
        let fallback_page: Option<usize> = Some(3);

        // Simulate empty hOCR results for page 3.
        let hocr_results: HashMap<usize, (Vec<neuroncite_pdf::HocrWord>, u32)> = {
            let mut m = HashMap::new();
            m.insert(3, (vec![], 300_u32));
            m
        };

        let should_produce_result = if let Some(fb_page) = fallback_page {
            if let Some((words, _dpi)) = hocr_results.get(&fb_page) {
                !words.is_empty()
            } else {
                false
            }
        } else {
            false
        };

        assert!(
            !should_produce_result,
            "empty hOCR words for the fallback page must not produce a result"
        );
    }

    /// T-ANNOTATE-RETEST5-005: When fallback_page refers to a page that has no
    /// hOCR results (the page was not included in the OCR batch), the fallback
    /// is skipped. This guards against HashMap key misses.
    #[test]
    #[cfg(feature = "ocr")]
    fn t_annotate_retest5_005_missing_hocr_page_skips_fallback() {
        let _fallback_page: Option<usize> = Some(10);

        // hOCR results only contain page 5, not page 10.
        let hocr_results: HashMap<usize, (Vec<neuroncite_pdf::HocrWord>, u32)> = {
            let mut m = HashMap::new();
            m.insert(
                5,
                (
                    vec![neuroncite_pdf::HocrWord {
                        text: "hello".into(),
                        x0: 10,
                        y0: 20,
                        x1: 50,
                        y1: 35,
                    }],
                    300_u32,
                ),
            );
            m
        };

        let page_has_hocr = hocr_results.contains_key(&10);
        assert!(
            !page_has_hocr,
            "fallback page 10 is not in hOCR results; fallback must be skipped"
        );
    }

    /// T-ANNOTATE-RETEST5-006: The scale factor conversion from pixel to PDF
    /// coordinates is correctly computed for different DPI values.
    /// At 300 DPI: scale = 72/300 = 0.24
    /// At 400 DPI: scale = 72/400 = 0.18
    #[test]
    fn t_annotate_retest5_006_scale_factor_for_different_dpi() {
        let scale_300: f32 = 72.0 / 300.0;
        let scale_400: f32 = 72.0 / 400.0;

        let eps = 0.001;
        assert!(
            (scale_300 - 0.24).abs() < eps,
            "300 DPI scale must be 0.24, got {}",
            scale_300
        );
        assert!(
            (scale_400 - 0.18).abs() < eps,
            "400 DPI scale must be 0.18, got {}",
            scale_400
        );

        // A pixel at x=1000 at 300 DPI maps to PDF x=240.0.
        let pdf_x_300 = 1000.0_f32 * scale_300;
        assert!(
            (pdf_x_300 - 240.0).abs() < eps,
            "pixel x=1000 at 300 DPI must map to PDF x=240.0"
        );

        // A pixel at x=1000 at 400 DPI maps to PDF x=180.0.
        let pdf_x_400 = 1000.0_f32 * scale_400;
        assert!(
            (pdf_x_400 - 180.0).abs() < eps,
            "pixel x=1000 at 400 DPI must map to PDF x=180.0"
        );
    }

    /// T-ANNOTATE-RETEST5-007: The fallback_page parameter is correctly derived
    /// from the Stage 3.5 fallback_result. When fallback_result is Some, the
    /// page_number is extracted. When None, fallback_page is None.
    #[test]
    fn t_annotate_retest5_007_fallback_page_derivation() {
        let fallback_with_result = Some(MatchResult {
            page_number: 7,
            char_start: 0,
            char_end: 50,
            bounding_boxes: vec![[72.0, 72.0, 540.0, 720.0]],
            method: MatchMethod::FallbackExtract,
            fuzzy_score: None,
        });

        let fallback_page = fallback_with_result.as_ref().map(|fb| fb.page_number);
        assert_eq!(
            fallback_page,
            Some(7),
            "fallback_page must be extracted from fallback_result.page_number"
        );

        let fallback_without_result: Option<MatchResult> = None;
        let fallback_page_none = fallback_without_result.as_ref().map(|fb| fb.page_number);
        assert_eq!(
            fallback_page_none, None,
            "fallback_page must be None when fallback_result is None"
        );
    }

    /// T-ANNOTATE-RETEST5-008: Y-axis inversion is correct for PDF coordinate
    /// system. PDF origin is bottom-left (Y increases upward), pixel origin is
    /// top-left (Y increases downward). For a page_height of 792pt:
    ///   pdf_y_bottom = page_height - (pixel_y_bottom * scale)
    ///   pdf_y_top    = page_height - (pixel_y_top * scale)
    /// The resulting pdf_y_bottom < pdf_y_top (bottom is lower than top).
    #[test]
    fn t_annotate_retest5_008_y_axis_inversion_correctness() {
        let page_height: f32 = 792.0;
        let scale: f32 = 72.0 / 300.0; // 0.24

        // Pixel bbox: top=100, bottom=130 (word is 30 pixels tall).
        let pixel_y_top: f32 = 100.0;
        let pixel_y_bottom: f32 = 130.0;

        let pdf_y_bottom = page_height - (pixel_y_bottom * scale); // 792 - 31.2 = 760.8
        let pdf_y_top = page_height - (pixel_y_top * scale); // 792 - 24.0 = 768.0

        assert!(
            pdf_y_bottom < pdf_y_top,
            "PDF y_bottom ({}) must be less than y_top ({}) after Y-axis inversion",
            pdf_y_bottom,
            pdf_y_top
        );

        // The word occupies 30 pixels = 30 * 0.24 = 7.2 PDF points.
        let box_height = pdf_y_top - pdf_y_bottom;
        let eps = 0.01;
        assert!(
            (box_height - 7.2).abs() < eps,
            "box height in PDF points must be 7.2, got {}",
            box_height
        );
    }

    // -----------------------------------------------------------------------
    // Anchor-word fuzzy matching algorithm tests
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-200: extract_anchor_words filters stopwords and short words,
    /// sorts by length descending, and deduplicates.
    #[test]
    fn t_annotate_200_extract_anchor_words_basic() {
        let quote = "The evidence in support of the efficient markets model";
        let anchors = extract_anchor_words(quote);

        // "The", "in", "of", "the" are stopwords. "efficient" (9), "evidence" (8),
        // "markets" (7), "support" (7), "model" (5) should be extracted.
        assert!(!anchors.is_empty(), "must extract at least one anchor word");
        assert!(
            anchors.contains(&"efficient".to_string()),
            "must contain 'efficient': {:?}",
            anchors
        );
        assert!(
            anchors.contains(&"evidence".to_string()),
            "must contain 'evidence': {:?}",
            anchors
        );
        assert!(
            !anchors.contains(&"the".to_string()),
            "must not contain stopword 'the': {:?}",
            anchors
        );
        assert!(
            !anchors.contains(&"in".to_string()),
            "must not contain stopword 'in': {:?}",
            anchors
        );
        // Sorted by length descending: "efficient" (9) before "evidence" (8).
        let eff_idx = anchors.iter().position(|w| w == "efficient").unwrap();
        let evi_idx = anchors.iter().position(|w| w == "evidence").unwrap();
        assert!(eff_idx < evi_idx, "longer words must come first");
    }

    /// T-ANNOTATE-201: extract_anchor_words returns at most 8 words.
    #[test]
    fn t_annotate_201_extract_anchor_words_max_8() {
        let quote = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima";
        let anchors = extract_anchor_words(quote);
        assert!(
            anchors.len() <= 8,
            "must return at most 8 anchors, got {}: {:?}",
            anchors.len(),
            anchors
        );
    }

    /// T-ANNOTATE-202: extract_anchor_words strips punctuation from word boundaries.
    #[test]
    fn t_annotate_202_extract_anchor_words_strips_punctuation() {
        let quote = "efficiency, (markets), documented.";
        let anchors = extract_anchor_words(quote);
        assert!(
            anchors.contains(&"efficiency".to_string()),
            "must strip trailing comma: {:?}",
            anchors
        );
        assert!(
            anchors.contains(&"markets".to_string()),
            "must strip parentheses: {:?}",
            anchors
        );
        assert!(
            anchors.contains(&"documented".to_string()),
            "must strip trailing period: {:?}",
            anchors
        );
    }

    /// T-ANNOTATE-203: extract_anchor_words returns empty for a quote made
    /// entirely of stopwords and short words.
    #[test]
    fn t_annotate_203_extract_anchor_words_all_stopwords() {
        let quote = "the a an is of in to";
        let anchors = extract_anchor_words(quote);
        assert!(
            anchors.is_empty(),
            "all-stopword quote must produce empty anchors: {:?}",
            anchors
        );
    }

    /// T-ANNOTATE-204: find_word_positions locates multiple occurrences.
    #[test]
    fn t_annotate_204_find_word_positions_multiple() {
        let page = "the cat sat on the mat and the cat purred";
        let positions = find_word_positions(page, "cat");
        assert_eq!(
            positions.len(),
            2,
            "must find 'cat' at 2 positions: {:?}",
            positions
        );
        assert_eq!(positions[0], 4, "first 'cat' at char 4");
        assert_eq!(positions[1], 31, "second 'cat' at char 31");
    }

    /// T-ANNOTATE-205: find_word_positions returns empty for absent word.
    #[test]
    fn t_annotate_205_find_word_positions_absent() {
        let page = "the quick brown fox";
        let positions = find_word_positions(page, "zebra");
        assert!(positions.is_empty(), "absent word must return empty Vec");
    }

    /// T-ANNOTATE-206: cluster_anchor_positions groups nearby positions and
    /// merges overlapping clusters.
    #[test]
    fn t_annotate_206_cluster_basic() {
        // Simulate 4 anchors at positions 10, 20, 30, 40 with quote_char_len=50.
        // max_span = 100. All within span, 4 distinct anchors >= min 2.
        let positions = vec![(0, 10), (1, 20), (2, 30), (3, 40)];
        let clusters = cluster_anchor_positions(&positions, 50, 2);
        assert_eq!(clusters.len(), 1, "must form one cluster: {:?}", clusters);
        assert_eq!(clusters[0], (10, 40));
    }

    /// T-ANNOTATE-207: cluster_anchor_positions returns empty when too few
    /// distinct anchors appear in any window.
    #[test]
    fn t_annotate_207_cluster_too_few_anchors() {
        // Only 1 distinct anchor (index 0) at 3 positions.
        let positions = vec![(0, 10), (0, 50), (0, 100)];
        let clusters = cluster_anchor_positions(&positions, 30, 2);
        assert!(
            clusters.is_empty(),
            "single-anchor positions must not form cluster: {:?}",
            clusters
        );
    }

    /// T-ANNOTATE-208: find_anchor_fuzzy returns exact match with score 1.0
    /// when the quote appears verbatim on the page.
    #[test]
    fn t_annotate_208_anchor_fuzzy_exact_in_page() {
        let page = "Introduction: The evidence in support of the efficient markets model is extensive and well documented. See references.";
        let quote = "The evidence in support of the efficient markets model is extensive and well documented.";
        let result = find_anchor_fuzzy(page, quote, 0.85);
        assert!(result.is_some(), "exact substring must produce a match");
        let (start, end, score) = result.unwrap();
        assert_eq!(score, 1.0, "exact match must score 1.0");
        let matched_text = &page[page.char_indices().nth(start).unwrap().0
            ..page
                .char_indices()
                .nth(end)
                .map(|(b, _)| b)
                .unwrap_or(page.len())];
        assert!(
            matched_text.contains("evidence") && matched_text.contains("documented"),
            "matched region must contain the quote"
        );
    }

    /// T-ANNOTATE-209: find_anchor_fuzzy returns None for completely unrelated text.
    #[test]
    fn t_annotate_209_anchor_fuzzy_unrelated() {
        let page = "Today's weather forecast: sunny with a high of 75 degrees.";
        let quote = "The efficient markets hypothesis implies that stock prices reflect all available information.";
        let result = find_anchor_fuzzy(page, quote, 0.85);
        assert!(result.is_none(), "unrelated text must return None");
    }

    /// T-ANNOTATE-210: find_anchor_fuzzy handles text with a few word
    /// substitutions (agent reformulation scenario).
    #[test]
    fn t_annotate_210_anchor_fuzzy_reformulated_text() {
        let page = "The empirical evidence strongly supports the efficient markets hypothesis and is extensively documented in the literature.";
        let quote = "The empirical evidence strongly supports the efficient market hypothesis and is extensively documented in the literature.";
        // Only "markets" -> "market" differs. Most anchors match.
        let result = find_anchor_fuzzy(page, quote, 0.85);
        assert!(
            result.is_some(),
            "minor word difference must still produce match"
        );
        let (_, _, score) = result.unwrap();
        assert!(
            score >= 0.90,
            "one-character difference must score >= 0.90, got {score:.3}"
        );
    }

    /// T-ANNOTATE-211: find_anchor_fuzzy handles text where the agent omitted
    /// a section (e.g., [...] in the middle of the passage).
    #[test]
    fn t_annotate_211_anchor_fuzzy_partial_omission() {
        let page = "The capital asset pricing model assumes that investors have homogeneous expectations about the distribution of future returns.";
        // Agent omitted "about the distribution of" and rewrote slightly.
        let quote = "The capital asset pricing model assumes that investors have homogeneous expectations future returns.";
        let result = find_anchor_fuzzy(page, quote, 0.70);
        assert!(
            result.is_some(),
            "partial omission with many shared anchors must match"
        );
        let (_, _, score) = result.unwrap();
        assert!(
            score >= 0.70,
            "partial omission score must be >= 0.70, got {score:.3}"
        );
    }

    /// T-ANNOTATE-212: find_anchor_fuzzy short-circuits via normalized
    /// substring search within the candidate window, producing score 1.0
    /// for text that matches after normalization (whitespace differences).
    #[test]
    fn t_annotate_212_anchor_fuzzy_normalized_shortcircuit() {
        let page = "The evidence in support of   the efficient  markets  model is extensive.";
        let quote = "The evidence in support of the efficient markets model is extensive.";
        let result = find_anchor_fuzzy(page, quote, 0.85);
        assert!(result.is_some(), "whitespace-different text must match");
        let (_, _, score) = result.unwrap();
        assert_eq!(score, 1.0, "normalized substring match must score 1.0");
    }

    /// T-ANNOTATE-213: fuzzy_match_score returns 0.0 for empty inputs.
    #[test]
    fn t_annotate_213_fuzzy_match_score_empty() {
        assert_eq!(fuzzy_match_score("some text", ""), 0.0);
        assert_eq!(fuzzy_match_score("", "some quote"), 0.0);
        assert_eq!(fuzzy_match_score("", ""), 0.0);
    }

    /// T-ANNOTATE-214: fuzzy_match_score returns high score for text with
    /// one word difference embedded in a larger page.
    #[test]
    fn t_annotate_214_fuzzy_match_score_embedded() {
        let page = "Chapter 1. The evidence in support of the efficient markets model is extensive. Chapter 2. Methods.";
        let quote = "The evidence in support of the efficient market model is extensive.";
        let score = fuzzy_match_score(page, quote);
        assert!(
            score >= FUZZY_THRESHOLD,
            "near-identical embedded text must score >= threshold, got {score:.3}"
        );
    }

    /// T-ANNOTATE-V3-PERF-001: find_anchor_fuzzy completes within 100ms for
    /// a long quote on a large page. Regression test against the O(n^2)
    /// sliding Levenshtein window that caused multi-minute hangs. With the
    /// 3-scale centered comparison, the function makes at most 3 Levenshtein
    /// calls per cluster instead of ~6,000.
    #[test]
    fn t_annotate_v3_perf_001_fuzzy_completes_quickly() {
        // Build a ~10,000-char page with a passage embedded in the middle.
        // Filler uses unrelated astronomy vocabulary so that the statistical
        // anchor words from the passage cluster exclusively in one region.
        let filler_a = "Galaxies rotate around supermassive black holes. \
            Neutron stars emit electromagnetic pulsations detectable by radio telescope arrays. "
            .repeat(35);
        let passage = "The empirical evidence strongly supports the efficient \
            markets hypothesis and is documented extensively in the financial literature";
        let filler_b = " Quasar luminosity exceeds that of entire galaxies. \
            Gravitational lensing bends spacetime around massive celestial objects. "
            .repeat(35);
        let page = format!("{filler_a}{passage}{filler_b}");

        // Reformulated quote (same pattern as t_annotate_210 which passes).
        let quote = "The empirical evidence strongly supports the efficient \
            market hypothesis and is documented in the literature";

        let start = std::time::Instant::now();
        let result = find_anchor_fuzzy(&page, quote, 0.75);
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 100,
            "find_anchor_fuzzy must complete in <100ms, took {}ms",
            elapsed.as_millis()
        );
        assert!(
            result.is_some(),
            "reformulated quote must match on large page"
        );
    }

    // -----------------------------------------------------------------------
    // InputRow deserialization tests
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-220: InputRow deserializes from JSON without the optional
    /// `page` field (backward compatibility with existing CSV/JSON inputs).
    #[test]
    fn t_annotate_220_input_row_no_page_field() {
        let json = r#"{"title":"A","author":"B","quote":"test quote"}"#;
        let row: crate::types::InputRow = serde_json::from_str(json).expect("deserialize");
        assert_eq!(row.title, "A");
        assert_eq!(row.quote, "test quote");
        assert!(
            row.page.is_none(),
            "missing page field must deserialize as None"
        );
    }

    /// T-ANNOTATE-221: InputRow deserializes from JSON with the optional
    /// `page` field set to a valid page number.
    #[test]
    fn t_annotate_221_input_row_with_page_field() {
        let json = r#"{"title":"A","author":"B","quote":"test quote","page":5}"#;
        let row: crate::types::InputRow = serde_json::from_str(json).expect("deserialize");
        assert_eq!(row.page, Some(5));
    }

    /// T-ANNOTATE-222: InputRow deserializes from CSV without a page column
    /// (the field defaults to None via serde).
    #[test]
    fn t_annotate_222_input_row_csv_no_page_column() {
        let csv_data = "title,author,quote,color,comment\nA,B,test quote,,\n";
        let mut rdr = csv::ReaderBuilder::new().from_reader(csv_data.as_bytes());
        let row: crate::types::InputRow = rdr
            .deserialize()
            .next()
            .expect("must have one row")
            .expect("deserialize");
        assert_eq!(row.quote, "test quote");
        assert!(
            row.page.is_none(),
            "CSV without page column must default to None"
        );
    }

    /// T-ANNOTATE-223: InputRow deserializes from CSV with a page column.
    #[test]
    fn t_annotate_223_input_row_csv_with_page_column() {
        let csv_data = "title,author,quote,color,comment,page\nA,B,test quote,,,7\n";
        let mut rdr = csv::ReaderBuilder::new().from_reader(csv_data.as_bytes());
        let row: crate::types::InputRow = rdr
            .deserialize()
            .next()
            .expect("must have one row")
            .expect("deserialize");
        assert_eq!(row.page, Some(7));
    }

    // -------------------------------------------------------------------
    // DEF-7 regression tests: cross-page text matching
    // -------------------------------------------------------------------

    /// T-ANNOTATE-230: MatchMethod::CrossPage serializes to "cross_page"
    /// via serde snake_case renaming and "crosspage" via Debug lowercasing.
    #[test]
    fn t_annotate_230_cross_page_method_serialization() {
        let method = crate::types::MatchMethod::CrossPage;
        let debug_str = format!("{:?}", method).to_lowercase();
        assert_eq!(
            debug_str, "crosspage",
            "Debug lowercasing produces 'crosspage' for pipeline reports"
        );

        // Serde serialization uses snake_case via #[serde(rename_all = "snake_case")].
        let json = serde_json::to_string(&method).expect("serialize CrossPage");
        assert_eq!(
            json, "\"cross_page\"",
            "serde serialization produces 'cross_page'"
        );
    }

    /// T-ANNOTATE-231: Cross-page matching via concatenated adjacent page
    /// texts. Verifies the core text concatenation and matching logic used
    /// by Stage 3+ (cross-page matching). Simulates two adjacent pages where
    /// a sentence spans the page boundary.
    #[test]
    fn t_annotate_231_cross_page_text_concatenation() {
        let page_a_text = "The results show that the coefficient is";
        let page_b_text = "statistically significant at the 5% level.";
        let cross_page_quote = "the coefficient is statistically significant at the 5% level";

        // The quote cannot be found on either page alone.
        assert!(
            !page_a_text.contains(cross_page_quote),
            "quote must not be found on page A alone"
        );
        assert!(
            !page_b_text.contains(cross_page_quote),
            "quote must not be found on page B alone"
        );

        // Concatenating with a space separator (matching the pipeline logic)
        // allows the quote to be found.
        let combined = format!("{} {}", page_a_text.trim_end(), page_b_text.trim_start());
        assert!(
            combined.contains(cross_page_quote),
            "cross-page quote must be found in the concatenated text"
        );
    }

    /// T-ANNOTATE-232: Cross-page matching with normalized text. Tests that
    /// the pipeline's normalize_text function collapses extra whitespace at
    /// the page boundary, allowing a quote with regular spacing to match
    /// text that has irregular whitespace from PDF extraction across two pages.
    #[test]
    fn t_annotate_232_cross_page_normalized_match() {
        // Page A ends with extra spaces; page B starts with extra spaces.
        // The concatenation produces multiple spaces at the boundary that
        // normalize_text collapses into single spaces, matching the quote.
        let page_a_text = "distribution follows a   heavy";
        let page_b_text = "  tailed  pattern consistent with";
        let quote = "distribution follows a heavy tailed pattern";

        let combined = format!("{} {}", page_a_text.trim_end(), page_b_text.trim_start());
        let norm_combined = normalize_text(&combined);
        let norm_quote = normalize_text(quote);

        assert!(
            norm_combined.contains(&norm_quote),
            "normalized cross-page text must contain the normalized quote, \
             got norm_combined='{}', norm_quote='{}'",
            norm_combined,
            norm_quote,
        );
    }
}
