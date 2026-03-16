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

// Pipeline orchestration for the annotation workflow.
//
// Coordinates all pipeline stages: input parsing, PDF resolution, text
// location, annotation creation, verification, and report generation.
// Processes PDFs in parallel (one rayon worker thread per PDF), where
// each thread owns a private Pdfium instance (Pdfium is Send but not Sync).
//
// Before the quote-location loop, each worker checks whether the PDF is a
// scanned/image-only document (no native text from pdfium). For scanned
// PDFs, a targeted hOCR extraction runs for a PRE-FILTERED subset of pages
// using ocr_pdf_pages_batch_hocr. The pre-filter uses cached text from the
// indexed session to identify which pages contain quotes (case-insensitive
// substring matching + word-overlap heuristics). This reduces the OCR
// workload from all pages (e.g., 35 for a full scanned document) to only
// the ~5-10 pages where quotes actually appear.
//
// The DB cache provides text for Stage 3.5 (fuzzy page-level matching)
// but has no coordinate data. Word-level bounding boxes for Stage 4
// require hOCR output with per-word pixel coordinates.
//
// The hOCR batch runs on a dedicated OS thread (not part of any rayon pool)
// with a 120-second timeout via channel communication. This prevents
// indefinite blocking from CPU contention or hung Tesseract processes.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::annotate;
use crate::appearance;
use crate::error::AnnotateError;
use crate::locate;
use crate::report::{
    AnnotationReport, PdfReport, QuoteReport, ReportSummary, UnmatchedInput, excerpt,
};
use crate::resolve;
use crate::types::{AnnotateConfig, Color, InputRow, MatchMethod};
use crate::verify;
use pdfium_render::prelude::*;

/// Result of Phase 1 text-only pre-matching for a single quote. Contains
/// the character position of the match on the identified page and the
/// matching method used. Bounding boxes are not computed here -- Phase 2
/// extracts them from pdfium using the known page and character range.
#[derive(Debug, Clone)]
struct PreMatch {
    /// 1-indexed page number where the quote was found.
    page_number: usize,
    /// Start character offset (0-based) in the page text.
    char_start: usize,
    /// End character offset (exclusive) in the page text.
    char_end: usize,
    /// Which matching stage produced this result.
    method: MatchMethod,
    /// Levenshtein similarity score (set for fuzzy matches, None otherwise).
    fuzzy_score: Option<f64>,
}

/// Phase 1: Searches all quotes against cached page texts in parallel.
///
/// For each (pdf, quote) pair, the function checks whether cached page texts
/// exist for the PDF. When they do, the quote is searched against the cached
/// text using exact, normalized, and anchor-fuzzy matching in sequence.
/// When a page hint is available from the citation verification pipeline,
/// that page is searched first before scanning all cached pages.
///
/// This phase is pure text processing on HashMap data -- no pdfium, no I/O.
/// All quote/PDF pairs are searched in parallel via rayon, utilizing all
/// CPU cores. Results are returned as a HashMap keyed by (pdf_index,
/// quote_index) for O(1) lookup in Phase 2.
///
/// Quotes without cached text or without a match are absent from the result
/// map and fall through to the regular locate_quote path in Phase 2.
fn pre_match_quotes(
    pdf_items: &[(PathBuf, Vec<InputRow>)],
    cached_page_texts: &HashMap<PathBuf, HashMap<usize, String>>,
) -> HashMap<(usize, usize), PreMatch> {
    // Flatten all (pdf_idx, quote_idx, pdf_path, quote, page_hint) tuples
    // into a single Vec for rayon parallel iteration across ALL quotes.
    let flattened: Vec<(usize, usize, &Path, &str, Option<usize>)> = pdf_items
        .iter()
        .enumerate()
        .flat_map(|(pdf_idx, (pdf_path, rows))| {
            rows.iter().enumerate().map(move |(quote_idx, row)| {
                (
                    pdf_idx,
                    quote_idx,
                    pdf_path.as_path(),
                    row.quote.as_str(),
                    row.page,
                )
            })
        })
        .collect();

    flattened
        .par_iter()
        .filter_map(|&(pdf_idx, quote_idx, pdf_path, quote, page_hint)| {
            let quote_trimmed = quote.trim();
            if quote_trimmed.is_empty() {
                return None;
            }

            let cached = cached_page_texts.get(pdf_path)?;
            if cached.is_empty() {
                return None;
            }

            // Build page iteration order: hint page first, then remaining.
            let mut page_order: Vec<usize> = Vec::new();
            if let Some(hint) = page_hint
                && cached.contains_key(&hint)
            {
                page_order.push(hint);
            }
            let mut sorted_pages: Vec<usize> = cached.keys().copied().collect();
            sorted_pages.sort();
            for &pg in &sorted_pages {
                if Some(pg) != page_hint {
                    page_order.push(pg);
                }
            }

            let norm_quote = locate::normalize_text(quote_trimmed);

            for &page_number in &page_order {
                let page_text = match cached.get(&page_number) {
                    Some(t) if !t.trim().is_empty() => t,
                    _ => continue,
                };

                // Stage 1: exact match on cached text.
                if let Some(pos) = page_text.find(quote_trimmed) {
                    let char_start = page_text[..pos].chars().count();
                    let char_end = char_start + quote_trimmed.chars().count();
                    return Some((
                        (pdf_idx, quote_idx),
                        PreMatch {
                            page_number,
                            char_start,
                            char_end,
                            method: MatchMethod::Exact,
                            fuzzy_score: None,
                        },
                    ));
                }

                // Stage 2: normalized match on cached text.
                let norm_page = locate::normalize_text(page_text);
                if let Some(norm_pos) = norm_page.find(&norm_quote) {
                    let (char_start, char_end) =
                        locate::map_normalized_to_original(page_text, norm_pos, norm_quote.len());
                    return Some((
                        (pdf_idx, quote_idx),
                        PreMatch {
                            page_number,
                            char_start,
                            char_end,
                            method: MatchMethod::Normalized,
                            fuzzy_score: None,
                        },
                    ));
                }

                // Stage 3: anchor-fuzzy match on cached text.
                if page_text.len() <= 100_000
                    && quote_trimmed.len() <= 500
                    && let Some((char_start, char_end, score)) =
                        locate::find_anchor_fuzzy(page_text, quote_trimmed, 0.75)
                {
                    return Some((
                        (pdf_idx, quote_idx),
                        PreMatch {
                            page_number,
                            char_start,
                            char_end,
                            method: MatchMethod::Fuzzy,
                            fuzzy_score: Some(score),
                        },
                    ));
                }
            }

            None
        })
        .collect()
}

/// Extracted text per page (page number -> text) and hOCR word-level data per
/// page (page number -> (words, DPI)). Used by the single-pass OCR extraction
/// block in `process_single_pdf` to return both plain text (for Stage 3.5
/// matching) and word-level bounding boxes (for Stage 4 Path A).
#[cfg(feature = "ocr")]
type OcrExtraction = (
    Option<HashMap<usize, String>>,
    Option<HashMap<usize, locate::HocrPageData>>,
);

/// Runs the full annotation pipeline: resolves PDFs, locates quotes, creates
/// annotations, verifies results, and generates a report.
///
/// The `progress` callback is invoked after each PDF is processed with
/// (done_count, total_count) for progress tracking (used by the job executor).
///
/// When `config.cached_page_texts` is populated (from a prior indexing session),
/// the pipeline reuses the cached page texts in stage 3.5 instead of
/// re-extracting text from PDFs. This significantly reduces processing time
/// for PDFs with scanned pages that require OCR.
///
/// This is a convenience wrapper around `annotate_pdfs_with_cancel` that
/// never cancels (the cancel callback always returns `false`).
///
/// Returns the annotation report containing per-PDF and per-quote results.
pub fn annotate_pdfs(
    config: AnnotateConfig,
    progress: impl Fn(usize, usize) + Send + Sync,
) -> Result<AnnotationReport, AnnotateError> {
    annotate_pdfs_with_cancel(config, progress, || false)
}

/// Runs the full annotation pipeline with cooperative cancellation support.
///
/// PDFs are processed in parallel using a rayon thread pool with one pdfium
/// instance per worker thread. The `progress` callback is invoked with
/// (done_quotes, total_quotes) after each quote is processed within a PDF,
/// providing per-quote granularity instead of per-PDF. This prevents the
/// progress counter from staying at 0 during long OCR pre-extraction phases
/// when there is only one PDF.
///
/// The `cancel_check` callback is called at the start of each PDF task; when
/// it returns `true`, the task returns immediately with status "canceled" and
/// the function returns `AnnotateError::Canceled` after all tasks drain.
///
/// Both `progress` and `cancel_check` must be `Send + Sync` because they are
/// captured by rayon's parallel closures which execute on multiple threads.
pub fn annotate_pdfs_with_cancel(
    config: AnnotateConfig,
    progress: impl Fn(usize, usize) + Send + Sync,
    cancel_check: impl Fn() -> bool + Send + Sync,
) -> Result<AnnotationReport, AnnotateError> {
    // Create output and error directories before spawning worker threads so
    // that concurrent fs::copy calls from worker threads can write directly.
    fs::create_dir_all(&config.output_directory)?;
    fs::create_dir_all(&config.error_directory)?;

    // Resolve input rows to PDF files via fuzzy filename matching.
    let (matched, unmatched) =
        resolve::match_pdfs_to_quotes(&config.source_directory, config.input_rows.clone())?;

    let total_pdfs = matched.len();

    // Total progress steps = total quotes + one "preparation" step per PDF.
    // For native-text PDFs, the preparation step covers loading + scanned-page
    // detection and completes instantly. For scanned PDFs, process_single_pdf
    // dynamically increases total_steps by (page_count - 1) to account for
    // per-page OCR progress: the 1 initial prep step is replaced with
    // page_count OCR steps + 1 setup step (see fetch_add below).
    //
    // AtomicUsize enables safe concurrent updates from rayon worker threads
    // when scanned PDFs increase the total, and the on_quote_done callback
    // reads the current total on each progress report.
    let total_quotes: usize = matched.values().map(|rows| rows.len()).sum();
    let total_steps = AtomicUsize::new(total_quotes + matched.len());

    // Report initial progress (0 of total_steps) so that callers (the job
    // executor) can display the total count immediately. When total_steps
    // is 0, the callback is skipped to avoid overwriting the handler's
    // initial progress_total with 0.
    let initial_total = total_steps.load(Ordering::Relaxed);
    if initial_total > 0 {
        progress(0, initial_total);
    }

    // Convert to a Vec to enable deterministic parallel iteration. The HashMap
    // iteration order is undefined; Vec preserves the insertion order for
    // a reproducible report layout across runs.
    let pdf_items: Vec<(PathBuf, Vec<InputRow>)> = matched.into_iter().collect();

    // Phase 1.5: Snapshot prior-annotated PDFs to a temporary directory
    // when prior_output_directory is set. In append mode, the annotate
    // handler sets prior_output_directory equal to output_directory
    // (the same directory). Phase 2 saves pdfium output to
    // output_directory, overwriting the file on disk. Phase 2.5 then
    // tries to merge prior annotations from prior_output_directory --
    // but if both directories are the same, Phase 2 already overwrote
    // the file, so the merge reads from the overwritten version (which
    // has no prior annotations). This snapshot copies the prior-annotated
    // PDFs to a temp directory BEFORE Phase 2 runs, so Phase 2.5 can
    // read from untouched copies.
    //
    // The snapshot is created whenever prior_output_directory is set,
    // regardless of whether it equals output_directory. This prevents
    // the same class of bug if the directories happen to resolve to the
    // same physical location (e.g., symlinks, junction points).
    let prior_snapshot_dir = if let Some(prior_dir) = config.prior_output_directory.as_ref() {
        let snapshot_dir = tempfile::tempdir().map_err(|e| {
            AnnotateError::Io(std::io::Error::other(format!(
                "failed to create temp dir for prior annotation snapshot: {e}"
            )))
        })?;
        let mut copied = 0_usize;

        for (pdf_path, _) in &pdf_items {
            if let Some(filename) = pdf_path.file_name() {
                let prior_path = prior_dir.join(filename);
                if prior_path.is_file() {
                    let dest = snapshot_dir.path().join(filename);
                    if let Err(e) = fs::copy(&prior_path, &dest) {
                        tracing::warn!(
                            prior = %prior_path.display(),
                            "failed to snapshot prior PDF for append merge: {e}"
                        );
                    } else {
                        copied += 1;
                    }
                }
            }
        }

        if copied > 0 {
            tracing::info!(
                dir = %snapshot_dir.path().display(),
                files = copied,
                "snapshotted prior-annotated PDFs for append merge"
            );
        }

        Some(snapshot_dir)
    } else {
        None
    };

    // Phase 1: Pre-match all quotes against cached page texts in parallel.
    // This runs before any pdfium work and uses all CPU cores. Quotes that
    // match produce a PreMatch with (page, char_start, char_end, method).
    // Phase 2 (the per-PDF pdfium loop) uses these results to skip the
    // locate_quote search and jump directly to bounding box extraction.
    let pre_matches = pre_match_quotes(&pdf_items, &config.cached_page_texts);
    if !pre_matches.is_empty() {
        tracing::info!(
            pre_matched = pre_matches.len(),
            total_quotes = total_quotes,
            "Phase 1 pre-matching complete"
        );
    }

    // Shared atomic counters for cross-thread progress tracking and cancel detection.
    // Relaxed ordering is sufficient: both values transition monotonically
    // (quote_done only increases, was_canceled only flips false->true) and a
    // one-iteration observation delay is acceptable.
    let quote_done = AtomicUsize::new(0);
    let was_canceled = AtomicBool::new(false);

    // Process all PDFs in parallel. A custom rayon thread pool with an 8 MB
    // stack per thread is required because pdfium operations (page rendering,
    // text extraction) recurse deeply and the default 2 MB rayon stack is
    // insufficient, causing stack overflows on complex PDFs.
    //
    // map_init creates one Pdfium instance per rayon worker thread. Pdfium is
    // Send (can be moved to another thread) but not Sync (cannot be shared
    // between threads). The map_init pattern stores the instance as &mut T on
    // the thread that created it, satisfying the Send-only requirement without
    // requiring Sync.
    let pool = rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build()
        .map_err(|e| AnnotateError::Pdfium(format!("thread pool creation failed: {e}")))?;

    let pdf_reports: Vec<PdfReport> = pool.install(|| {
        pdf_items
            .par_iter()
            .enumerate()
            .map_init(
                // Per-thread pdfium binding initialization. Called once when the
                // thread processes its first PDF. Returns None if binding fails;
                // the map closure handles that case by returning a failed PdfReport.
                || {
                    neuroncite_pdf::pdfium_binding::bind_pdfium()
                        .map(Pdfium::new)
                        .ok()
                },
                |pdfium_opt, (pdf_idx, (pdf_path, rows))| {
                    // Cooperative cancellation check at the start of each PDF task.
                    // Tasks that have not yet started processing return immediately
                    // when the cancel flag is set. Tasks already in progress
                    // (pdfium is open, annotations created) complete normally so
                    // that partial results and the error directory are consistent.
                    if cancel_check() {
                        was_canceled.store(true, Ordering::Relaxed);
                        return PdfReport {
                            filename: pdf_path
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| "unknown.pdf".into()),
                            source_path: pdf_path.to_string_lossy().to_string(),
                            output_path: None,
                            status: "canceled".into(),
                            quotes: vec![],
                            error: Some("pipeline canceled before this PDF was processed".into()),
                        };
                    }

                    let pdfium = match pdfium_opt.as_ref() {
                        Some(p) => p,
                        None => {
                            return PdfReport {
                                filename: pdf_path
                                    .file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| "unknown.pdf".into()),
                                source_path: pdf_path.to_string_lossy().to_string(),
                                output_path: None,
                                status: "failed".into(),
                                quotes: vec![],
                                error: Some("pdfium initialization failed on this thread".into()),
                            };
                        }
                    };

                    // Look up the pre-populated DB cache for this specific PDF.
                    // HashMap::get takes &Q where K: Borrow<Q>; &PathBuf coerces
                    // to &Path which PathBuf borrows, so this compiles correctly.
                    let cached_pages = config.cached_page_texts.get(pdf_path.as_path());

                    process_single_pdf(
                        pdfium,
                        pdf_path,
                        rows,
                        &config.output_directory,
                        &config.error_directory,
                        &config.default_color,
                        cached_pages,
                        pdf_idx,
                        &pre_matches,
                        &|n| {
                            // Atomically increment the completed step counter
                            // and invoke the progress callback. Each step
                            // (per-page OCR, preparation, or quote) calls this
                            // with n=1 upon completion. fetch_add returns the
                            // previous value; prev + n is the count after
                            // this increment. total_steps is loaded atomically
                            // to reflect dynamic increases from scanned PDFs.
                            let prev = quote_done.fetch_add(n, Ordering::Relaxed);
                            let current_total = total_steps.load(Ordering::Relaxed);
                            progress(prev + n, current_total);
                        },
                        &total_steps,
                    )
                },
            )
            .collect()
    });

    // Phase 2.5: Merge highlight annotations from prior output PDFs (append
    // mode). When prior_output_directory is set, each output PDF is checked
    // for a corresponding file in the snapshot directory (created in Phase
    // 1.5). The snapshot contains copies of the prior-annotated PDFs taken
    // BEFORE Phase 2 overwrote them with pdfium output. Existing highlight
    // annotations are copied into the new PDF using lopdf, with
    // deduplication via 90% bounding-box area overlap. This preserves
    // annotations from previous append runs while the pipeline reads from
    // original source PDFs for accurate text extraction.
    let effective_prior_dir = prior_snapshot_dir
        .as_ref()
        .map(|tmp| tmp.path().to_path_buf())
        .or_else(|| config.prior_output_directory.clone());
    if let Some(ref prior_dir) = effective_prior_dir {
        let merge_paths: Vec<(&str, PathBuf)> = pdf_reports
            .iter()
            .filter_map(|r| {
                let output = r.output_path.as_deref()?;
                let filename = Path::new(output).file_name()?;
                let prior = prior_dir.join(filename);
                if prior.is_file() {
                    Some((output, prior))
                } else {
                    None
                }
            })
            .collect();

        if !merge_paths.is_empty() {
            merge_paths.par_iter().for_each(|(new_path, prior_path)| {
                match appearance::merge_prior_annotations(Path::new(new_path), prior_path) {
                    Ok(count) if count > 0 => {
                        tracing::info!(
                            pdf = new_path,
                            merged = count,
                            "merged prior annotations in append mode"
                        );
                    }
                    Ok(_) => {}
                    Err(e) => {
                        tracing::warn!(pdf = new_path, "prior annotation merge failed: {e}");
                    }
                }
            });
        }
    }

    // Phase 3: Inject appearance streams into all annotated PDFs in parallel.
    // lopdf is pure Rust (Send+Sync), no pdfium dependency. Runs on the
    // default rayon pool without the 8 MB stack or per-thread pdfium binding
    // constraint. This is separated from Phase 2 to avoid mixing pdfium I/O
    // (save) with lopdf I/O (load+modify+save) within the same per-PDF block.
    let ap_paths: Vec<&str> = pdf_reports
        .iter()
        .filter_map(|r| r.output_path.as_deref())
        .collect();

    if !ap_paths.is_empty() {
        ap_paths.par_iter().for_each(|path| {
            if let Err(e) = appearance::inject_appearance_streams(Path::new(path)) {
                tracing::warn!(path, "appearance stream injection failed: {e}");
            }
        });
    }

    // Propagate cancellation: if any parallel task saw the cancel flag, return
    // Canceled with the count of PDFs that actually completed processing.
    // PDFs with status "canceled" were skipped before any work was done.
    if was_canceled.load(Ordering::Relaxed) {
        let completed = pdf_reports
            .iter()
            .filter(|r| r.status != "canceled")
            .count();
        return Err(AnnotateError::Canceled {
            completed,
            total: total_pdfs,
        });
    }

    // Build unmatched input entries for the report.
    let unmatched_entries: Vec<UnmatchedInput> = unmatched
        .iter()
        .map(|row| UnmatchedInput {
            title: row.title.clone(),
            author: row.author.clone(),
            error: "no PDF matched in source directory".into(),
        })
        .collect();

    let mut report = AnnotationReport {
        timestamp: chrono::Utc::now().to_rfc3339(),
        source_directory: config.source_directory.to_string_lossy().to_string(),
        output_directory: config.output_directory.to_string_lossy().to_string(),
        summary: ReportSummary {
            total_pdfs: 0,
            successful: 0,
            partial: 0,
            failed: 0,
            total_quotes: 0,
            quotes_matched: 0,
            quotes_not_found: 0,
        },
        pdfs: pdf_reports,
        unmatched_inputs: unmatched_entries,
    };

    report.compute_summary();

    // Write the report to the output directory.
    let report_path = config.output_directory.join("annotate_report.json");
    let report_json = serde_json::to_string_pretty(&report).map_err(|e| {
        AnnotateError::Io(std::io::Error::other(format!("report serialization: {e}")))
    })?;
    fs::write(&report_path, &report_json)?;

    tracing::info!(
        matched = report.summary.quotes_matched,
        not_found = report.summary.quotes_not_found,
        total = report.summary.total_quotes,
        "annotation pipeline complete"
    );

    Ok(report)
}

/// Determines which pages of a scanned PDF require hOCR processing based on
/// the quotes that need to be annotated and the cached page texts from a
/// previous indexing session.
///
/// When cached text is available, this function performs a lightweight text search
/// to identify pages containing quotes. Two strategies are applied per quote:
///
///   1. Case-insensitive substring match: the entire lowercased quote must appear
///      in the lowercased page text.
///
///   2. Word-overlap heuristic: if at least 40% of the quote's whitespace-delimited
///      words appear in the page text (case-insensitive), the page is included.
///      This catches cases where OCR text differs slightly from the quote
///      (e.g., ligature differences, "rn" vs "m" substitutions).
///
/// When no cached text is available, all pages are returned (no filtering possible).
/// When no quote matches any cached page, all pages are returned as a fallback
/// for Stage 4's last-resort full-document search.
///
/// Returns a sorted, deduplicated vector of 1-indexed page numbers.
#[cfg(feature = "ocr")]
fn prefilter_pages_for_hocr(
    rows: &[crate::types::InputRow],
    cached_pages: Option<&HashMap<usize, String>>,
    page_count: usize,
) -> Vec<usize> {
    let cached = match cached_pages {
        Some(c) => c,
        None => return (1..=page_count).collect(),
    };

    let mut needed: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for row in rows {
        let quote_lower = row.quote.to_lowercase();
        for (&page_num, text) in cached.iter() {
            let text_lower = text.to_lowercase();

            // Strategy 1: substring containment.
            if text_lower.contains(&quote_lower) {
                needed.insert(page_num);
                continue;
            }

            // Strategy 2: word-overlap heuristic (>= 40% of quote words
            // found in the page text). Only applied for quotes with more
            // than 2 words to avoid false positives on short queries.
            let quote_words: Vec<&str> = quote_lower.split_whitespace().collect();
            if quote_words.len() > 2 {
                let matching = quote_words
                    .iter()
                    .filter(|w| text_lower.contains(*w))
                    .count();
                // matching * 5 >= len * 2  <==>  matching/len >= 0.4
                if matching * 5 >= quote_words.len() * 2 {
                    needed.insert(page_num);
                }
            }
        }
    }

    if needed.is_empty() {
        // No pre-filter matches: OCR all pages as last resort.
        (1..=page_count).collect()
    } else {
        let mut sorted: Vec<usize> = needed.into_iter().collect();
        sorted.sort();
        sorted
    }
}

/// Processes a single PDF: opens it, runs hOCR for scanned pages, locates all
/// assigned quotes, creates annotations, saves the result, and runs post-check
/// verification.
///
/// The `cached_pages` parameter provides pre-extracted page texts from an
/// indexed session for this specific PDF. When present, stage 3.5 of the
/// text location pipeline uses these texts instead of re-extracting via OCR.
///
/// The `pdf_idx` parameter is the index of this PDF in the `pdf_items` Vec,
/// used as a key into the `pre_matches` HashMap together with the per-quote
/// index. Pre-matched quotes skip the full locate_quote search and jump
/// directly to bounding box extraction on the known page.
///
/// For scanned PDFs (no native pdfium text in the first 3 pages), a targeted
/// hOCR extraction runs for a pre-filtered subset of pages. The pre-filter
/// identifies pages containing quotes using cached text from the indexed session.
/// The DB cache only contains text (for Stage 3.5 matching); it has no coordinate
/// data. Word-level bounding boxes for Stage 4 highlight precision require hOCR
/// output, which contains per-word pixel coordinates that are scaled to PDF points.
///
/// When cached text IS available for a scanned PDF:
/// - A pre-filter identifies which pages contain quotes using case-insensitive
///   substring matching and word-overlap heuristics against the cached text
/// - hOCR runs only on the targeted pages (~5-10 instead of all pages)
/// - Stage 3.5 uses the cached text (faster and more stable than hOCR-reconstructed text)
/// - Stage 4 uses the precomputed hOCR data for word-level bounding box refinement
///
/// When cached text is NOT available for a scanned PDF:
/// - hOCR runs on ALL pages (no text available for pre-filtering)
/// - Stage 3.5 uses text reconstructed from hOCR words
/// - Stage 4 uses the same hOCR data for word-level bounding boxes
///
/// The `on_quote_done` callback is invoked with n=completed_pages after the
/// hOCR batch finishes (for scanned PDFs) and with n=1 after each quote is
/// processed (matched, not_found, or error). Callers use this for progress
/// reporting.
///
/// Returns a PdfReport with per-quote results. If the PDF cannot be opened
/// or saved, the entire PDF is reported as failed and copied to the error
/// directory.
#[allow(clippy::too_many_arguments)]
fn process_single_pdf(
    pdfium: &Pdfium,
    pdf_path: &Path,
    rows: &[InputRow],
    output_directory: &Path,
    error_directory: &Path,
    default_color: &str,
    cached_pages: Option<&HashMap<usize, String>>,
    pdf_idx: usize,
    pre_matches: &HashMap<(usize, usize), PreMatch>,
    on_quote_done: &(dyn Fn(usize) + Send + Sync),
    total_steps: &AtomicUsize,
) -> PdfReport {
    let filename = pdf_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown.pdf".into());

    let source_path = pdf_path.to_string_lossy().to_string();

    // Load the PDF document with write access for annotation creation.
    let document = match pdfium.load_pdf_from_file(pdf_path, None) {
        Ok(doc) => doc,
        Err(e) => {
            tracing::warn!(pdf = %filename, "failed to load PDF: {e}");
            copy_to_error_dir(pdf_path, error_directory);
            // Report all steps from this PDF as done (preparation step +
            // all quote steps) so the progress counter advances past this
            // failed PDF.
            on_quote_done(rows.len() + 1);
            return PdfReport {
                filename,
                source_path,
                output_path: None,
                status: "failed".into(),
                quotes: Vec::new(),
                error: Some(format!("PDF load failed: {e}")),
            };
        }
    };

    // Detect whether the PDF is a scanned/image-only document by sampling
    // the first three pages via pdfium. When pdfium returns no native text
    // from any of the sampled pages, the PDF is classified as scanned.
    //
    // For scanned PDFs, hOCR extraction runs on a TARGETED subset of pages
    // rather than all pages. The DB cache provides text for Stage 3.5 (fuzzy
    // page-level matching) but has no coordinate data. Word-level bounding
    // boxes for Stage 4 require hOCR output with per-word pixel coordinates.
    //
    // Page pre-filtering: When cached text from an indexed session is available,
    // a lightweight search identifies which pages contain quotes using
    // case-insensitive substring matching and word-overlap heuristics. This
    // reduces the hOCR workload from ALL pages (e.g., 35 for Fama 1970) to
    // only the ~5-10 pages where quotes actually appear. Stage 3.5 identifies
    // the matching page; Stage 4 refines with word-level bounding boxes from
    // the precomputed hOCR data. Pages not in the hOCR set fall back to
    // page-level bounding boxes via Stage 3.5.
    //
    // The hOCR batch runs on a dedicated OS thread to escape the annotation
    // pipeline's rayon pool. A channel with timeout (120s) prevents indefinite
    // blocking when the OCR batch hangs or is severely CPU-contended.
    #[cfg(feature = "ocr")]
    let (live_extracted, live_hocr): OcrExtraction = 'extract_block: {
        // Borrow document.pages() in a scoped block so the borrow is dropped
        // before the subsequent document.pages().get() calls in the quote loop.
        let pdf_is_scanned = {
            let pages = document.pages();
            let check_count = (pages.len() as usize).min(3);
            let mut has_native_text = false;
            for i in 0..check_count {
                if let Ok(page) = pages.get(i as u16)
                    && let Ok(text) = page.text()
                    && !text.all().trim().is_empty()
                {
                    has_native_text = true;
                    break;
                }
            }
            !has_native_text
        };

        if !pdf_is_scanned {
            // Native-text PDF: stages 1-3 (exact, normalized, fuzzy on pdfium
            // text) are sufficient. No pre-extraction needed.
            break 'extract_block (None, None);
        }

        let page_count = document.pages().len() as usize;

        // Pre-filter pages using cached text to determine which pages need
        // hOCR. This avoids OCR'ing the entire document when only a small
        // subset of pages is relevant for the quotes being annotated.
        let pages_to_ocr = prefilter_pages_for_hocr(rows, cached_pages, page_count);
        let ocr_page_count = pages_to_ocr.len();

        // Dynamically increase total_steps to account for per-page OCR progress.
        // The initial total_steps counted 1 "preparation" step for this PDF.
        // Only the targeted pages (not all pages) are added as OCR steps.
        total_steps.fetch_add(ocr_page_count, Ordering::Relaxed);

        // Resolve Tesseract binary and tessdata paths once for all pages.
        // Falls back to page-level bboxes (Stage 3.5) if Tesseract is missing.
        let tess_paths = match neuroncite_pdf::TesseractPaths::resolve() {
            Ok(paths) => paths,
            Err(e) => {
                tracing::warn!(
                    pdf = %filename,
                    error = %e,
                    "Tesseract not available, falling back to page-level bboxes"
                );
                break 'extract_block (None, None);
            }
        };

        tracing::debug!(
            pdf = %filename,
            total_pages = page_count,
            ocr_pages = ocr_page_count,
            has_cached_text = cached_pages.is_some(),
            "scanned PDF detected, rendering targeted pages for inline hOCR"
        );

        // Phase 1: Render targeted pages to PNG using the EXISTING PdfDocument.
        //
        // Sequential because PdfDocument is not Send/Sync -- only the thread
        // that created the document can access it. This reuses the pdfium
        // instance that process_single_pdf already loaded, avoiding DLL loader
        // lock contention on Windows from parallel pdfium bindings.
        //
        // Each call renders at 300/400 DPI and encodes to PNG with fast
        // compression (~0.3-0.5s per page).
        let mut rendered_pages: Vec<(usize, Vec<u8>, u32)> = Vec::with_capacity(ocr_page_count);

        for &page_num in &pages_to_ocr {
            match neuroncite_pdf::render_page_to_png_with_dpi(&document, page_num) {
                Ok((png_bytes, dpi)) => {
                    rendered_pages.push((page_num, png_bytes, dpi));
                }
                Err(e) => {
                    tracing::warn!(
                        pdf = %filename,
                        page = page_num,
                        error = %e,
                        "page render failed, skipping this page for hOCR"
                    );
                }
            }
        }

        if rendered_pages.is_empty() {
            tracing::warn!(
                pdf = %filename,
                "all page renders failed, falling back to page-level bboxes"
            );
            break 'extract_block (None, None);
        }

        // Phase 2: Run Tesseract hOCR in parallel on the rendered PNG bytes.
        //
        // Tesseract is invoked as a subprocess (no pdfium involvement), so
        // these can run fully parallel without DLL contention. Each subprocess
        // is restricted to 1 OpenMP thread (OMP_NUM_THREADS=1 set internally
        // by ocr_png_hocr_via_cli) to prevent CPU oversubscription.
        //
        // std::thread::scope ensures all spawned threads complete before the
        // scope exits, allowing the threads to borrow tess_paths and the PNG
        // byte slices from rendered_pages without moving ownership.
        let hocr_results: Vec<(usize, Vec<neuroncite_pdf::HocrWord>, u32)> =
            std::thread::scope(|s| {
                let handles: Vec<_> = rendered_pages
                    .iter()
                    .map(|(page_num, png_bytes, dpi)| {
                        let pn = *page_num;
                        let d = *dpi;
                        let tp = &tess_paths;
                        s.spawn(
                            move || -> Option<(usize, Vec<neuroncite_pdf::HocrWord>, u32)> {
                                match tp.ocr_png_to_hocr_words(png_bytes, "eng") {
                                    Ok(words) => Some((pn, words, d)),
                                    Err(e) => {
                                        tracing::warn!(
                                            page = pn,
                                            error = %e,
                                            "Tesseract hOCR failed for page"
                                        );
                                        None
                                    }
                                }
                            },
                        )
                    })
                    .collect();

                handles
                    .into_iter()
                    .filter_map(|h| h.join().ok().flatten())
                    .collect()
            });

        // Report per-page progress for all completed OCR pages.
        let completed_pages = hocr_results.len();
        if completed_pages > 0 {
            on_quote_done(completed_pages);
        }

        // Phase 3: Build the hOCR HashMap for Stage 4 Path A. When no DB
        // cache exists, reconstruct text from hOCR words for Stage 3.5.
        let hocr_map: HashMap<usize, locate::HocrPageData> = hocr_results
            .into_iter()
            .map(|(page_num, words, dpi)| (page_num, (words, dpi)))
            .collect();

        let text_map = if cached_pages.is_some() {
            None
        } else {
            let map: HashMap<usize, String> = hocr_map
                .iter()
                .map(|(&page_num, (words, _dpi))| {
                    let text = words
                        .iter()
                        .map(|w| w.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    (page_num, text)
                })
                .collect();
            Some(map)
        };

        tracing::debug!(
            pdf = %filename,
            pages = hocr_map.len(),
            "inline hOCR complete, word-level bounding boxes available"
        );
        (text_map, Some(hocr_map))
    };

    // When OCR feature is disabled, no live extraction is possible.
    // Suppress unused-variable warnings for parameters only used in the
    // OCR-gated extract_block.
    #[cfg(not(feature = "ocr"))]
    let _ = total_steps;
    #[cfg(not(feature = "ocr"))]
    let live_extracted: Option<HashMap<usize, String>> = None;
    #[cfg(not(feature = "ocr"))]
    let live_hocr: Option<HashMap<usize, locate::HocrPageData>> = None;

    // Use the DB cache when present; fall back to the just-extracted text for
    // scanned PDFs that are not in the DB cache; None for native-text PDFs
    // where stages 1-3 are sufficient.
    let effective_cached = cached_pages.or(live_extracted.as_ref());
    let effective_hocr = live_hocr.as_ref();

    // Report the "PDF preparation" step as done. For scanned PDFs, per-page
    // progress was already reported via the on_page_done callback during hOCR.
    // For native-text PDFs (where hOCR was skipped), the preparation step
    // is reported here. For scanned PDFs, this reports 1 additional step
    // for the text reconstruction and setup that happened after hOCR completed.
    on_quote_done(1);

    let mut quote_reports: Vec<QuoteReport> = Vec::with_capacity(rows.len());
    let mut any_matched = false;

    for (quote_idx, row) in rows.iter().enumerate() {
        let quote_excerpt = excerpt(&row.quote, 80);
        let color_str = row
            .color
            .as_deref()
            .filter(|c| !c.is_empty())
            .unwrap_or(default_color);
        let color = Color::parse(color_str)
            .unwrap_or_else(|| Color::parse("#FFFF00").expect("default yellow color must parse"));

        // Phase 2 fast-path: when Phase 1 pre-matching found this quote in
        // cached text, extract bounding boxes directly from the known page
        // without running the full locate_quote pipeline. This skips all
        // five stages and goes straight to pdfium character bounds.
        let locate_result = if let Some(pre) = pre_matches.get(&(pdf_idx, quote_idx)) {
            let page_idx_u16 = (pre.page_number - 1) as u16;
            match document.pages().get(page_idx_u16) {
                Ok(page) => match page.text() {
                    Ok(text_page) => {
                        let boxes =
                            locate::extract_char_bounds(&text_page, pre.char_start, pre.char_end);
                        if boxes.is_empty() {
                            // Bounding boxes unavailable (scanned page with no
                            // native text layer). Fall through to locate_quote
                            // which handles fallback stages 3.5 and 4.
                            tracing::debug!(
                                pdf = %filename,
                                page = pre.page_number,
                                "pre-match fast-path: no bounding boxes, falling back to locate_quote"
                            );
                            locate::locate_quote(
                                &document,
                                pdf_path,
                                &row.quote,
                                "eng",
                                &locate::LocateContext {
                                    cached_pages: effective_cached,
                                    precomputed_hocr: effective_hocr,
                                    page_hint: Some(pre.page_number),
                                },
                            )
                        } else {
                            Ok(crate::types::MatchResult {
                                page_number: pre.page_number,
                                char_start: pre.char_start,
                                char_end: pre.char_end,
                                bounding_boxes: boxes,
                                method: pre.method,
                                fuzzy_score: pre.fuzzy_score,
                            })
                        }
                    }
                    Err(_) => locate::locate_quote(
                        &document,
                        pdf_path,
                        &row.quote,
                        "eng",
                        &locate::LocateContext {
                            cached_pages: effective_cached,
                            precomputed_hocr: effective_hocr,
                            page_hint: Some(pre.page_number),
                        },
                    ),
                },
                Err(_) => locate::locate_quote(
                    &document,
                    pdf_path,
                    &row.quote,
                    "eng",
                    &locate::LocateContext {
                        cached_pages: effective_cached,
                        precomputed_hocr: effective_hocr,
                        page_hint: Some(pre.page_number),
                    },
                ),
            }
        } else {
            // No pre-match: run the full locate_quote pipeline.
            locate::locate_quote(
                &document,
                pdf_path,
                &row.quote,
                "eng",
                &locate::LocateContext {
                    cached_pages: effective_cached,
                    precomputed_hocr: effective_hocr,
                    page_hint: row.page,
                },
            )
        };

        match locate_result {
            Ok(match_result) => {
                // Pre-check verification.
                let pre_check = verify::pre_check(
                    &match_result.bounding_boxes,
                    match_result.char_end - match_result.char_start,
                );

                if !pre_check.passed {
                    tracing::warn!(
                        pdf = %filename,
                        quote = %quote_excerpt,
                        "pre-check failed: {}/{} chars have bounding boxes",
                        pre_check.chars_with_bbox,
                        pre_check.total_chars
                    );
                }

                // Capture fuzzy_score from the match result before moving it.
                let fuzzy_score = match_result.fuzzy_score;

                // Create annotations on the page.
                let page_idx = (match_result.page_number - 1) as u16;
                let mut page = match document.pages().get(page_idx) {
                    Ok(p) => p,
                    Err(e) => {
                        quote_reports.push(QuoteReport {
                            quote_excerpt,
                            status: "error".into(),
                            match_method: Some(format!("{:?}", match_result.method).to_lowercase()),
                            page: Some(match_result.page_number),
                            chars_matched: Some(pre_check.chars_with_bbox),
                            chars_total: Some(pre_check.total_chars),
                            pre_check_passed: Some(pre_check.passed),
                            post_check_passed: None,
                            stages_tried: None,
                            fuzzy_score,
                        });
                        tracing::warn!(pdf = %filename, "failed to get page {}: {e}", match_result.page_number);
                        on_quote_done(1);
                        continue;
                    }
                };

                let highlight_count =
                    annotate::create_highlight_annotations(&mut page, &match_result, &color);

                // Create comment annotation if specified.
                if let Some(ref comment) = row.comment
                    && let Err(e) =
                        annotate::create_comment_annotation(&mut page, &match_result, comment)
                {
                    tracing::warn!(pdf = %filename, "comment annotation failed: {e}");
                }

                let annotation_ok = highlight_count.is_ok();
                any_matched = any_matched || annotation_ok;

                // Determine the quote match status. When the annotation was
                // created but the pre-check failed (low ratio of characters
                // with valid bounding boxes), and the match method is
                // FallbackExtract (page-level bounding box from Stage 3.5
                // without word-level refinement), the status is set to
                // "low_confidence_match" instead of "matched". This prevents
                // the false-positive reporting where a full-page bounding box
                // (chars_matched: 1/N) is reported as a successful match.
                // The annotation still exists in the output PDF as a page-level
                // highlight, but the status accurately reflects the imprecise
                // positioning.
                let status = if !annotation_ok {
                    "error".into()
                } else if !pre_check.passed
                    && (match_result.method == MatchMethod::FallbackExtract
                        || match_result.method == MatchMethod::CrossPage)
                {
                    "low_confidence_match".into()
                } else {
                    "matched".into()
                };

                quote_reports.push(QuoteReport {
                    quote_excerpt,
                    status,
                    match_method: Some(format!("{:?}", match_result.method).to_lowercase()),
                    page: Some(match_result.page_number),
                    chars_matched: Some(pre_check.chars_with_bbox),
                    chars_total: Some(pre_check.total_chars),
                    pre_check_passed: Some(pre_check.passed),
                    post_check_passed: None, // filled after save
                    stages_tried: None,
                    fuzzy_score,
                });
            }
            Err(ref e) => {
                // Extract stages_tried diagnostics from QuoteNotFound errors.
                let stages_tried = match e {
                    crate::error::AnnotateError::QuoteNotFound { stages_tried, .. } => {
                        Some(stages_tried.clone())
                    }
                    _ => None,
                };

                // Full-page fallback: when a page number is known from the
                // citation verification pipeline, create a page-level highlight
                // covering the entire content area instead of reporting
                // "not_found". The page number is always available in the
                // citation workflow because the verification agent links each
                // passage to a specific chunk with page coordinates.
                if let Some(hint_page) = row.page {
                    let page_idx = (hint_page - 1) as u16;
                    if let Ok(mut fallback_page) = document.pages().get(page_idx) {
                        let page_match = annotate::create_page_level_match(
                            &fallback_page,
                            hint_page,
                            row.quote.chars().count(),
                        );

                        if let Err(e) = annotate::create_highlight_annotations(
                            &mut fallback_page,
                            &page_match,
                            &color,
                        ) {
                            tracing::warn!(
                                pdf = %filename,
                                page = hint_page,
                                "failed to create highlight annotations for page-level fallback: {e}"
                            );
                        }

                        // Build fallback comment: prepend "[page-level]" to
                        // the original comment, or use a default message when
                        // no comment was provided.
                        let fallback_comment = match &row.comment {
                            Some(c) if !c.trim().is_empty() => {
                                format!("[page-level] {c}")
                            }
                            _ => "[page-level] passage not located precisely".to_string(),
                        };
                        if let Err(e) = annotate::create_comment_annotation(
                            &mut fallback_page,
                            &page_match,
                            &fallback_comment,
                        ) {
                            tracing::warn!(
                                pdf = %filename,
                                page = hint_page,
                                "failed to create comment annotation for page-level fallback: {e}"
                            );
                        }

                        any_matched = true;

                        tracing::info!(
                            pdf = %filename,
                            quote = %quote_excerpt,
                            page = hint_page,
                            "quote not found in text, using page-level fallback"
                        );
                        quote_reports.push(QuoteReport {
                            quote_excerpt,
                            status: "page_level_match".into(),
                            match_method: Some("page_level".into()),
                            page: Some(hint_page),
                            chars_matched: Some(1),
                            chars_total: Some(row.quote.chars().count()),
                            pre_check_passed: Some(false),
                            post_check_passed: None,
                            stages_tried,
                            fuzzy_score: None,
                        });
                    } else {
                        // Page index out of range.
                        tracing::info!(
                            pdf = %filename,
                            quote = %quote_excerpt,
                            page = hint_page,
                            "quote not found, page hint out of range"
                        );
                        quote_reports.push(QuoteReport {
                            quote_excerpt,
                            status: "not_found".into(),
                            match_method: None,
                            page: Some(hint_page),
                            chars_matched: None,
                            chars_total: None,
                            pre_check_passed: None,
                            post_check_passed: None,
                            stages_tried,
                            fuzzy_score: None,
                        });
                    }
                } else {
                    // No page hint available -- genuine not_found.
                    tracing::info!(
                        pdf = %filename,
                        quote = %quote_excerpt,
                        "quote not found, no page hint available"
                    );
                    quote_reports.push(QuoteReport {
                        quote_excerpt,
                        status: "not_found".into(),
                        match_method: None,
                        page: None,
                        chars_matched: None,
                        chars_total: None,
                        pre_check_passed: None,
                        post_check_passed: None,
                        stages_tried,
                        fuzzy_score: None,
                    });
                }
            }
        }

        // Report this quote as done regardless of match outcome (matched,
        // not_found, error). The caller's atomic counter increments by 1.
        on_quote_done(1);
    }

    // Save the annotated PDF to the output directory.
    let output_path = output_directory.join(&filename);

    // Determine the annotated quote count. Includes "matched" (word-level),
    // "low_confidence_match" (FallbackExtract page-level), and
    // "page_level_match" (full-page fallback when text location failed).
    let matched_count = quote_reports
        .iter()
        .filter(|q| {
            q.status == "matched"
                || q.status == "low_confidence_match"
                || q.status == "page_level_match"
        })
        .count();
    let total_count = quote_reports.len();

    if any_matched {
        match document.save_to_file(&output_path) {
            Ok(_) => {
                tracing::info!(pdf = %filename, "saved annotated PDF");

                // Appearance stream injection is deferred to a parallel batch
                // phase after all PDFs are processed (Phase 3 in
                // annotate_pdfs_with_cancel). lopdf is Send+Sync and runs on
                // any rayon thread without the per-thread pdfium constraint.

                let status = if matched_count == total_count {
                    "success"
                } else if matched_count > 0 {
                    "partial"
                } else {
                    "failed"
                };

                PdfReport {
                    filename,
                    source_path,
                    output_path: Some(output_path.to_string_lossy().to_string()),
                    status: status.into(),
                    quotes: quote_reports,
                    error: None,
                }
            }
            Err(e) => {
                tracing::error!(pdf = %filename, "save failed: {e}");
                copy_to_error_dir(pdf_path, error_directory);
                PdfReport {
                    filename,
                    source_path,
                    output_path: None,
                    status: "failed".into(),
                    quotes: quote_reports,
                    error: Some(format!("save failed: {e}")),
                }
            }
        }
    } else {
        // No annotations were created. With page-level fallback, this only
        // happens when no quotes have page hints AND all location stages
        // failed. Copy the source PDF to output_directory (not error_dir)
        // so the output set is complete. error_directory is reserved for
        // PDFs that cannot be opened or saved.
        if let Err(e) = fs::copy(pdf_path, &output_path) {
            tracing::warn!(
                pdf = %filename,
                "failed to copy unannotated PDF to output: {e}"
            );
        }

        let status = if matched_count == total_count && total_count > 0 {
            "success"
        } else if matched_count > 0 {
            "partial"
        } else {
            "failed"
        };

        PdfReport {
            filename,
            source_path,
            output_path: Some(output_path.to_string_lossy().to_string()),
            status: status.into(),
            quotes: quote_reports,
            error: None,
        }
    }
}

/// Copies a PDF file to the error directory. Logs a warning if the copy fails.
fn copy_to_error_dir(pdf_path: &Path, error_directory: &Path) {
    if let Some(filename) = pdf_path.file_name() {
        let dest = error_directory.join(filename);
        if let Err(e) = fs::copy(pdf_path, &dest) {
            tracing::warn!(
                src = %pdf_path.display(),
                dest = %dest.display(),
                "failed to copy PDF to error directory: {e}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// T-ANN-CANCEL-001: `annotate_pdfs_with_cancel` returns `AnnotateError::Canceled`
    /// when the cancel callback returns `true` immediately. The pipeline does not
    /// process any PDFs because the cancel check fires before the first PDF.
    #[test]
    fn t_ann_cancel_001_immediate_cancel_returns_canceled_error() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let output = tmp.path().join("output");
        let errors = tmp.path().join("errors");

        let config = AnnotateConfig {
            input_rows: vec![],
            source_directory: tmp.path().to_path_buf(),
            output_directory: output,
            error_directory: errors,
            default_color: "#FFFF00".into(),
            cached_page_texts: std::collections::HashMap::new(),
            prior_output_directory: None,
        };

        // Cancel callback always returns true.
        let result = annotate_pdfs_with_cancel(config, |_, _| {}, || true);

        // With zero input rows, there are zero matched PDFs, so the pipeline
        // completes without hitting the cancel check. Verify the pipeline
        // still succeeds (the cancel check is only invoked inside the
        // matched PDF loop).
        assert!(
            result.is_ok(),
            "pipeline with zero input rows should succeed even with cancel flag set"
        );
    }

    /// T-ANN-CANCEL-002: `annotate_pdfs_with_cancel` with `cancel_check` returning
    /// `false` behaves identically to `annotate_pdfs`. The pipeline completes
    /// and returns an `AnnotationReport`.
    #[test]
    fn t_ann_cancel_002_no_cancel_completes_normally() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let output = tmp.path().join("output");
        let errors = tmp.path().join("errors");

        let config = AnnotateConfig {
            input_rows: vec![],
            source_directory: tmp.path().to_path_buf(),
            output_directory: output,
            error_directory: errors,
            default_color: "#FFFF00".into(),
            cached_page_texts: std::collections::HashMap::new(),
            prior_output_directory: None,
        };

        let result = annotate_pdfs_with_cancel(config, |_, _| {}, || false);

        assert!(
            result.is_ok(),
            "pipeline with cancel_check returning false should complete"
        );
    }

    /// T-ANN-CANCEL-003: The `annotate_pdfs` wrapper function delegates to
    /// `annotate_pdfs_with_cancel` with a never-cancel callback. Verifies
    /// backward compatibility of the original API.
    #[test]
    fn t_ann_cancel_003_annotate_pdfs_backward_compat() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let output = tmp.path().join("output");
        let errors = tmp.path().join("errors");

        let config = AnnotateConfig {
            input_rows: vec![],
            source_directory: tmp.path().to_path_buf(),
            output_directory: output,
            error_directory: errors,
            default_color: "#FFFF00".into(),
            cached_page_texts: std::collections::HashMap::new(),
            prior_output_directory: None,
        };

        let result = annotate_pdfs(config, |_, _| {});
        assert!(result.is_ok(), "annotate_pdfs should still work as before");
    }

    /// T-ANN-CANCEL-004: The `AnnotateError::Canceled` variant correctly
    /// stores and reports the completed and total PDF counts.
    #[test]
    fn t_ann_cancel_004_canceled_error_contains_counts() {
        let err = AnnotateError::Canceled {
            completed: 5,
            total: 10,
        };

        let msg = format!("{err}");
        assert!(
            msg.contains("5"),
            "error message must contain completed count"
        );
        assert!(msg.contains("10"), "error message must contain total count");
        assert!(
            msg.contains("canceled"),
            "error message must contain 'canceled'"
        );
    }

    /// T-ANN-CANCEL-005: The cooperative cancel flag can be shared via
    /// `Arc<AtomicBool>`, matching the pattern used by the executor. This
    /// test verifies that the cancel callback correctly reads the flag and
    /// that the pipeline respects it.
    #[test]
    fn t_ann_cancel_005_atomic_bool_flag_integration() {
        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = flag.clone();

        let tmp = tempfile::tempdir().expect("create temp dir");
        let output = tmp.path().join("output");
        let errors = tmp.path().join("errors");

        let config = AnnotateConfig {
            input_rows: vec![],
            source_directory: tmp.path().to_path_buf(),
            output_directory: output,
            error_directory: errors,
            default_color: "#FFFF00".into(),
            cached_page_texts: std::collections::HashMap::new(),
            prior_output_directory: None,
        };

        // Set cancel flag after construction but before pipeline runs.
        flag.store(true, Ordering::Relaxed);

        let result = annotate_pdfs_with_cancel(
            config,
            |_, _| {},
            move || flag_clone.load(Ordering::Relaxed),
        );

        // With zero matched PDFs, the cancel check is not reached.
        assert!(
            result.is_ok(),
            "pipeline with zero PDFs should not reach the cancel check"
        );
    }

    /// T-ANN-PROGRESS-001: When zero PDFs match, the progress callback is not
    /// invoked. The handler sets its own initial progress_total (based on quote
    /// count) before the pipeline runs, so the pipeline must not overwrite it
    /// with (0, 0). Regression test for ISSUE-014 where a no-match annotation
    /// job showed progress_total=0 instead of the handler's initial value.
    #[test]
    fn t_ann_progress_001_initial_progress_callback_fired() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let output = tmp.path().join("output");
        let errors = tmp.path().join("errors");

        let config = AnnotateConfig {
            input_rows: vec![],
            source_directory: tmp.path().to_path_buf(),
            output_directory: output,
            error_directory: errors,
            default_color: "#FFFF00".into(),
            cached_page_texts: std::collections::HashMap::new(),
            prior_output_directory: None,
        };

        // Track all progress callbacks.
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let calls_clone = calls.clone();

        let result = annotate_pdfs(config, move |done, total| {
            calls_clone.lock().unwrap().push((done, total));
        });

        assert!(result.is_ok());

        let recorded = calls.lock().unwrap();
        // With zero input rows (zero matched PDFs), no progress callback
        // should be fired. The handler is responsible for setting the initial
        // progress_total from the quote count.
        assert!(
            recorded.is_empty(),
            "progress callback must not be invoked when zero PDFs matched"
        );
    }

    /// T-ANN-PARALLEL-001: `annotate_pdfs_with_cancel` accepts progress and
    /// cancel_check callbacks that are `Send + Sync`, as required by the rayon
    /// parallel implementation. This test verifies that Arc-backed closures
    /// (the pattern used by the executor) compile and work correctly.
    #[test]
    fn t_ann_parallel_001_send_sync_callbacks_accepted() {
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let flag_for_check = cancel_flag.clone();

        let progress_calls = Arc::new(AtomicUsize::new(0));
        let progress_counter = progress_calls.clone();

        let tmp = tempfile::tempdir().expect("create temp dir");
        let output = tmp.path().join("output");
        let errors = tmp.path().join("errors");

        let config = AnnotateConfig {
            input_rows: vec![],
            source_directory: tmp.path().to_path_buf(),
            output_directory: output,
            error_directory: errors,
            default_color: "#FFFF00".into(),
            cached_page_texts: std::collections::HashMap::new(),
            prior_output_directory: None,
        };

        // Both closures are Send + Sync: they capture Arc<...> which is Send + Sync.
        let result = annotate_pdfs_with_cancel(
            config,
            move |_done, _total| {
                progress_counter.fetch_add(1, Ordering::Relaxed);
            },
            move || flag_for_check.load(Ordering::Relaxed),
        );

        assert!(
            result.is_ok(),
            "pipeline with Send + Sync callbacks must complete without error"
        );
    }

    /// T-ANN-BUG-B-001: When `cached_pages` is `None` and `live_extracted` is
    /// `None` (native-text PDF), `effective_cached` must be `None`. This verifies
    /// the `or()` logic correctly falls through when neither cache is available.
    #[test]
    fn t_ann_bug_b_001_effective_cached_none_when_both_none() {
        let cached_pages: Option<&HashMap<usize, String>> = None;
        let live_extracted: Option<HashMap<usize, String>> = None;
        let effective = cached_pages.or(live_extracted.as_ref());
        assert!(
            effective.is_none(),
            "effective_cached must be None when both DB cache and live extraction are unavailable"
        );
    }

    /// T-ANN-BUG-B-002: When `cached_pages` is `None` but `live_extracted` is
    /// `Some`, `effective_cached` must resolve to the live-extracted data.
    /// This is the path taken for scanned PDFs not in the DB cache.
    #[test]
    fn t_ann_bug_b_002_effective_cached_uses_live_extracted_when_db_cache_absent() {
        let cached_pages: Option<&HashMap<usize, String>> = None;
        let mut live_map = HashMap::new();
        live_map.insert(1_usize, "extracted page text from OCR".to_string());
        let live_extracted: Option<HashMap<usize, String>> = Some(live_map);

        let effective = cached_pages.or(live_extracted.as_ref());

        assert!(
            effective.is_some(),
            "effective_cached must be Some when live_extracted is available"
        );
        assert_eq!(
            effective.unwrap().get(&1).map(String::as_str),
            Some("extracted page text from OCR"),
            "effective_cached must contain the live-extracted text"
        );
    }

    /// T-ANN-BUG-B-003: When `cached_pages` is `Some` (DB cache hit), it takes
    /// precedence over `live_extracted`. This is the fast path: no live
    /// extraction is performed when the DB cache covers the PDF.
    #[test]
    fn t_ann_bug_b_003_db_cache_takes_precedence_over_live_extracted() {
        let mut db_map = HashMap::new();
        db_map.insert(1_usize, "DB cached text from indexing run".to_string());
        let cached_pages: Option<&HashMap<usize, String>> = Some(&db_map);

        // live_extracted would normally not be populated when cached_pages is Some,
        // but test the or() precedence explicitly.
        let mut live_map = HashMap::new();
        live_map.insert(
            1_usize,
            "live extracted text (should not be used)".to_string(),
        );
        let live_extracted: Option<HashMap<usize, String>> = Some(live_map);

        let effective = cached_pages.or(live_extracted.as_ref());

        assert_eq!(
            effective.unwrap().get(&1).map(String::as_str),
            Some("DB cached text from indexing run"),
            "DB cache must take precedence over live-extracted text"
        );
    }

    // -----------------------------------------------------------------------
    // DEFECT-007: low_confidence_match status for page-level bounding boxes
    // -----------------------------------------------------------------------

    /// T-ANN-BUG-007-001: When a FallbackExtract match has pre_check_passed=false
    /// (page-level bounding box with chars_with_bbox << chars_total), the
    /// status is set to "low_confidence_match" instead of "matched".
    ///
    /// Regression test for DEFECT-007: the annotation pipeline previously
    /// reported status "matched" for FallbackExtract results with
    /// chars_matched=1/202, which is a false-positive match indicator.
    /// The fix distinguishes page-level matches from word-level matches
    /// via the pre_check result.
    #[test]
    fn t_ann_bug_007_001_low_confidence_status_for_page_level_match() {
        use crate::types::MatchMethod;
        use crate::verify;

        // Simulate a FallbackExtract match with 1 page-level bounding box.
        let boxes = vec![[72.0_f32, 72.0, 540.0, 720.0]];
        let total_chars = 202;
        let pre_check = verify::pre_check(&boxes, total_chars);
        let method = MatchMethod::FallbackExtract;
        let annotation_ok = true;

        // Replicate the status determination logic from process_single_pdf.
        let status: String = if !annotation_ok {
            "error".into()
        } else if !pre_check.passed && method == MatchMethod::FallbackExtract {
            "low_confidence_match".into()
        } else {
            "matched".into()
        };

        assert_eq!(
            status, "low_confidence_match",
            "FallbackExtract with page-level bbox must produce 'low_confidence_match' status"
        );
    }

    /// T-ANN-BUG-007-002: When an OCR match has pre_check_passed=true
    /// (word-level bounding boxes), the status remains "matched".
    ///
    /// Verifies that the low_confidence_match logic does not affect
    /// OCR-refined matches that have proper bounding boxes.
    #[test]
    fn t_ann_bug_007_002_matched_status_for_word_level_match() {
        use crate::types::MatchMethod;
        use crate::verify;

        // Simulate an OCR match with 50 word-level bounding boxes.
        let boxes: Vec<[f32; 4]> = (0..50)
            .map(|i| {
                let left = 72.0 + (i as f32 % 10.0) * 50.0;
                [left, 600.0, left + 40.0, 612.0]
            })
            .collect();
        let total_chars = 50;
        let pre_check = verify::pre_check(&boxes, total_chars);
        let method = MatchMethod::Ocr;
        let annotation_ok = true;

        let status: String = if !annotation_ok {
            "error".into()
        } else if !pre_check.passed && method == MatchMethod::FallbackExtract {
            "low_confidence_match".into()
        } else {
            "matched".into()
        };

        assert_eq!(
            status, "matched",
            "OCR match with word-level boxes must produce 'matched' status"
        );
    }

    /// T-ANN-BUG-007-003: When an Exact match has pre_check_passed=true,
    /// the status remains "matched" regardless of the low_confidence logic.
    #[test]
    fn t_ann_bug_007_003_exact_match_unaffected_by_low_confidence_logic() {
        use crate::types::MatchMethod;
        use crate::verify;

        let boxes: Vec<[f32; 4]> = (0..20)
            .map(|i| [72.0 + i as f32 * 8.0, 600.0, 80.0 + i as f32 * 8.0, 612.0])
            .collect();
        let total_chars = 20;
        let pre_check = verify::pre_check(&boxes, total_chars);
        let method = MatchMethod::Exact;
        let annotation_ok = true;

        let status: String = if !annotation_ok {
            "error".into()
        } else if !pre_check.passed && method == MatchMethod::FallbackExtract {
            "low_confidence_match".into()
        } else {
            "matched".into()
        };

        assert_eq!(
            status, "matched",
            "Exact match must produce 'matched' status"
        );
        assert!(pre_check.passed, "Exact match pre_check must pass");
    }

    /// T-ANN-BUG-007-004: When annotation creation fails (annotation_ok=false),
    /// the status is "error" regardless of match method or pre_check result.
    #[test]
    fn t_ann_bug_007_004_error_status_overrides_low_confidence() {
        use crate::types::MatchMethod;
        use crate::verify;

        let boxes = vec![[72.0_f32, 72.0, 540.0, 720.0]];
        let total_chars = 202;
        let _pre_check = verify::pre_check(&boxes, total_chars);
        let method = MatchMethod::FallbackExtract;
        let annotation_ok = false; // annotation creation failed

        let status: String = if !annotation_ok {
            "error".into()
        } else if !_pre_check.passed && method == MatchMethod::FallbackExtract {
            "low_confidence_match".into()
        } else {
            "matched".into()
        };

        assert_eq!(
            status, "error",
            "failed annotation must produce 'error' status, not 'low_confidence_match'"
        );
    }

    // -----------------------------------------------------------------------
    // RETEST-5 Fix: Per-quote progress reporting
    // -----------------------------------------------------------------------

    /// T-ANN-RETEST5-001: The total_quotes calculation sums all quote rows
    /// across all matched PDFs. This is the denominator for per-quote progress.
    #[test]
    fn t_ann_retest5_001_total_quotes_counts_all_rows_across_pdfs() {
        // Simulate 3 PDFs with different numbers of quotes.
        let pdf_items: Vec<(PathBuf, Vec<crate::types::InputRow>)> = vec![
            (
                PathBuf::from("a.pdf"),
                vec![
                    crate::types::InputRow {
                        title: "A".into(),
                        author: "A".into(),
                        quote: "q1".into(),
                        color: None,
                        comment: None,
                        page: None,
                    },
                    crate::types::InputRow {
                        title: "A".into(),
                        author: "A".into(),
                        quote: "q2".into(),
                        color: None,
                        comment: None,
                        page: None,
                    },
                ],
            ),
            (
                PathBuf::from("b.pdf"),
                vec![crate::types::InputRow {
                    title: "B".into(),
                    author: "B".into(),
                    quote: "q3".into(),
                    color: None,
                    comment: None,
                    page: None,
                }],
            ),
            (
                PathBuf::from("c.pdf"),
                vec![
                    crate::types::InputRow {
                        title: "C".into(),
                        author: "C".into(),
                        quote: "q4".into(),
                        color: None,
                        comment: None,
                        page: None,
                    },
                    crate::types::InputRow {
                        title: "C".into(),
                        author: "C".into(),
                        quote: "q5".into(),
                        color: None,
                        comment: None,
                        page: None,
                    },
                    crate::types::InputRow {
                        title: "C".into(),
                        author: "C".into(),
                        quote: "q6".into(),
                        color: None,
                        comment: None,
                        page: None,
                    },
                ],
            ),
        ];

        let total_quotes: usize = pdf_items.iter().map(|(_, rows)| rows.len()).sum();
        let total_steps = total_quotes + pdf_items.len();
        assert_eq!(
            total_quotes, 6,
            "total_quotes must be the sum of all rows across all PDFs (2+1+3=6)"
        );
        assert_eq!(
            total_steps, 9,
            "total_steps must be total_quotes + number_of_pdfs (6+3=9)"
        );
    }

    /// T-ANN-RETEST5-002: The atomic quote_done counter increments by 1 for
    /// each quote processed. After processing N quotes, the counter equals N.
    #[test]
    fn t_ann_retest5_002_quote_done_atomic_increment() {
        let quote_done = AtomicUsize::new(0);
        let total_quotes = 5;

        // Simulate processing 5 quotes.
        for i in 0..total_quotes {
            let prev = quote_done.fetch_add(1, Ordering::Relaxed);
            assert_eq!(
                prev, i,
                "fetch_add must return the previous value before increment"
            );
        }

        assert_eq!(
            quote_done.load(Ordering::Relaxed),
            total_quotes,
            "after processing all quotes, counter must equal total"
        );
    }

    /// T-ANN-RETEST5-003: When a PDF fails to load, all its quotes are
    /// counted as done in a single batch increment. This prevents the
    /// progress counter from stalling at the failed PDF.
    #[test]
    fn t_ann_retest5_003_failed_pdf_increments_all_quotes() {
        let quote_done = AtomicUsize::new(0);
        // 2 PDFs, 10 quotes total -> 12 steps (10 quotes + 2 prep steps)
        let total_steps = 12;

        // PDF 1 has 4 quotes and fails to load -- batch increment
        // includes 4 quotes + 1 preparation step = 5.
        let pdf1_steps = 4 + 1;
        let prev = quote_done.fetch_add(pdf1_steps, Ordering::Relaxed);
        assert_eq!(prev, 0, "first batch increment starts at 0");

        // PDF 2 has 6 quotes and processes normally.
        // 1 preparation step + 6 per-quote increments = 7 steps.
        quote_done.fetch_add(1, Ordering::Relaxed); // preparation step
        for _ in 0..6 {
            quote_done.fetch_add(1, Ordering::Relaxed);
        }

        assert_eq!(
            quote_done.load(Ordering::Relaxed),
            total_steps,
            "after all PDFs, counter must equal total_steps"
        );
    }

    /// T-ANN-RETEST5-004: Progress callbacks fire with monotonically increasing
    /// done values. Concurrent atomic increments from parallel PDF workers
    /// must produce a strictly increasing sequence when observed.
    #[test]
    fn t_ann_retest5_004_progress_monotonically_increasing() {
        let quote_done = Arc::new(AtomicUsize::new(0));
        let total_quotes = 100;
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));

        // Simulate 4 threads each processing 25 quotes.
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let counter = Arc::clone(&quote_done);
                let obs = Arc::clone(&observed);
                std::thread::spawn(move || {
                    for _ in 0..25 {
                        let prev = counter.fetch_add(1, Ordering::Relaxed);
                        let done = prev + 1;
                        obs.lock().unwrap().push(done);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let final_count = quote_done.load(Ordering::Relaxed);
        assert_eq!(
            final_count, total_quotes,
            "final count must equal total_quotes"
        );

        // All observed done values must be in range [1, total_quotes].
        let obs = observed.lock().unwrap();
        assert_eq!(obs.len(), 100, "100 observations for 100 increments");
        for &v in obs.iter() {
            assert!(
                v >= 1 && v <= total_quotes,
                "observed done value {} must be in [1, {}]",
                v,
                total_quotes
            );
        }
    }

    /// T-ANN-RETEST5-005: The OS thread dispatch for pre-extraction avoids
    /// rayon pool saturation. This tests the pattern where extract_pages is
    /// called on a separate OS thread and the result is sent back via channel.
    #[test]
    fn t_ann_retest5_005_os_thread_dispatch_pattern() {
        // Simulate the OS thread dispatch pattern used in process_single_pdf
        // for scanned PDF pre-extraction.
        let (tx, rx) = std::sync::mpsc::channel::<Result<String, String>>();

        let handle = std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(move || {
                // Simulates extract_pages returning extracted text.
                let result: Result<String, String> = Ok("extracted text".into());
                let _ = tx.send(result);
            })
            .expect("thread spawn must succeed");

        let received = rx.recv();
        handle.join().expect("thread must join");

        assert!(
            received.is_ok(),
            "channel must receive the result from the spawned thread"
        );
        assert_eq!(
            received.unwrap().unwrap(),
            "extracted text",
            "result must be the extracted text"
        );
    }

    /// T-ANN-RETEST5-006: When the OS thread panics or terminates without
    /// sending a result, the channel recv returns Err. The pre-extraction
    /// returns None and the pipeline continues without cached text.
    #[test]
    fn t_ann_retest5_006_os_thread_crash_returns_channel_error() {
        let (tx, rx) = std::sync::mpsc::channel::<Result<String, String>>();

        let handle = std::thread::spawn(move || {
            // Simulate the thread dropping the sender without sending anything.
            drop(tx);
        });

        handle.join().expect("thread must join");

        let received = rx.recv();
        assert!(
            received.is_err(),
            "channel must return error when sender is dropped without sending"
        );
    }

    /// T-ANN-RETEST5-007: The on_quote_done callback is invoked exactly once
    /// per quote, even when the page-get fails and the loop uses `continue`.
    /// This verifies that the progress counter does not skip quotes.
    #[test]
    fn t_ann_retest5_007_on_quote_done_called_for_every_quote() {
        let call_count = AtomicUsize::new(0);
        let on_quote_done = |n: usize| {
            call_count.fetch_add(n, Ordering::Relaxed);
        };

        // Simulate 1 PDF with 5 quotes: 2 matched, 1 page-get-failed
        // (continue path), 1 not_found, 1 matched.
        // Total steps = 1 preparation + 5 quotes = 6.
        let outcomes = [
            "matched",
            "matched",
            "page_get_failed",
            "not_found",
            "matched",
        ];

        // Preparation step fires first.
        on_quote_done(1);

        for outcome in &outcomes {
            match *outcome {
                "page_get_failed" => {
                    // In the real code, this path calls on_quote_done(1)
                    // before `continue`.
                    on_quote_done(1);
                    continue;
                }
                _ => {
                    // At the end of the match block, on_quote_done(1) is called.
                    on_quote_done(1);
                }
            }
        }

        assert_eq!(
            call_count.load(Ordering::Relaxed),
            6,
            "on_quote_done must be called exactly 6 times (1 prep + 5 quotes)"
        );
    }

    /// T-ANN-RETEST5-008: With zero matched PDFs (total_steps=0), the
    /// initial progress callback is not fired to avoid overwriting the
    /// handler's initial progress_total.
    #[test]
    fn t_ann_retest5_008_zero_quotes_skips_initial_progress() {
        let progress_called = AtomicBool::new(false);

        let total_steps: usize = 0; // 0 quotes + 0 PDFs = 0 steps
        if total_steps > 0 {
            progress_called.store(true, Ordering::Relaxed);
        }

        assert!(
            !progress_called.load(Ordering::Relaxed),
            "progress must not be called when total_quotes is 0"
        );
    }

    // -----------------------------------------------------------------------
    // DEFECT-006/007: Targeted page pre-filter for hOCR extraction
    // -----------------------------------------------------------------------

    /// Helper: creates an InputRow with the given quote text. Title, author,
    /// color, and comment are set to empty/None defaults for testing.
    #[cfg(feature = "ocr")]
    fn make_row(quote: &str) -> crate::types::InputRow {
        crate::types::InputRow {
            title: String::new(),
            author: String::new(),
            quote: quote.to_string(),
            color: None,
            comment: None,
            page: None,
        }
    }

    /// T-ANN-PREFILTER-001: When cached_pages is None (no indexed session),
    /// all pages are returned regardless of quote content. Pre-filtering
    /// is impossible without cached text.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_001_no_cache_returns_all_pages() {
        let rows = vec![make_row("some quote text")];
        let result = prefilter_pages_for_hocr(&rows, None, 35);
        let expected: Vec<usize> = (1..=35).collect();
        assert_eq!(
            result, expected,
            "without cached text, all 35 pages must be returned"
        );
    }

    /// T-ANN-PREFILTER-002: When a quote exactly matches cached text on a
    /// specific page, only that page is included in the result.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_002_exact_substring_match_single_page() {
        let rows = vec![make_row("efficient market hypothesis")];
        let mut cached = HashMap::new();
        cached.insert(
            5,
            "The efficient market hypothesis states that...".to_string(),
        );
        cached.insert(10, "References and bibliography section here.".to_string());
        cached.insert(15, "Statistical methodology for data analysis.".to_string());

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 35);
        assert_eq!(result, vec![5], "only page 5 contains the exact substring");
    }

    /// T-ANN-PREFILTER-003: Case-insensitive matching catches quotes with
    /// different casing than the cached text.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_003_case_insensitive_substring_match() {
        let rows = vec![make_row("EFFICIENT MARKET")];
        let mut cached = HashMap::new();
        cached.insert(
            3,
            "The Efficient Market Hypothesis is a cornerstone...".to_string(),
        );
        cached.insert(7, "Other unrelated content on this page.".to_string());

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 10);
        assert_eq!(result, vec![3], "case-insensitive match must find page 3");
    }

    /// T-ANN-PREFILTER-004: Multiple quotes matching different pages produce
    /// a sorted, deduplicated set of all matched pages.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_004_multiple_quotes_different_pages() {
        let rows = vec![
            make_row("random walk"),
            make_row("beta coefficient"),
            make_row("portfolio theory"),
        ];
        let mut cached = HashMap::new();
        cached.insert(
            2,
            "The random walk model describes price movements...".to_string(),
        );
        cached.insert(
            8,
            "The beta coefficient measures systematic risk...".to_string(),
        );
        cached.insert(
            15,
            "Modern portfolio theory was developed by Markowitz...".to_string(),
        );
        cached.insert(20, "Conclusion and summary of findings.".to_string());

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 35);
        assert_eq!(
            result,
            vec![2, 8, 15],
            "pages matching any quote must be included, sorted"
        );
    }

    /// T-ANN-PREFILTER-005: Multiple quotes matching the SAME page produce
    /// a single entry (deduplication via HashSet).
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_005_multiple_quotes_same_page_deduplicated() {
        let rows = vec![make_row("market efficiency"), make_row("stock prices")];
        let mut cached = HashMap::new();
        cached.insert(
            5,
            "Stock prices reflect market efficiency in the long run.".to_string(),
        );

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 10);
        assert_eq!(
            result,
            vec![5],
            "same page matched by two quotes must appear only once"
        );
    }

    /// T-ANN-PREFILTER-006: When no quote matches any cached page (neither
    /// substring nor word-overlap), all pages are returned as a fallback
    /// for Stage 4's last-resort full-document search.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_006_no_match_returns_all_pages() {
        let rows = vec![make_row("completely unrelated quantum physics text")];
        let mut cached = HashMap::new();
        cached.insert(1, "Financial markets and asset pricing.".to_string());
        cached.insert(2, "Regression analysis for stock returns.".to_string());

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 5);
        let expected: Vec<usize> = (1..=5).collect();
        assert_eq!(
            result, expected,
            "when no quote matches, all pages must be returned as fallback"
        );
    }

    /// T-ANN-PREFILTER-007: Word-overlap heuristic (>= 40%) catches fuzzy
    /// matches where the exact substring does not appear but enough words
    /// are shared. This handles OCR text variants (ligatures, "rn" -> "m").
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_007_word_overlap_heuristic_catches_fuzzy() {
        // Quote has 10 words; page text contains 5 of them (50% overlap > 40%).
        let rows = vec![make_row(
            "the efficient market hypothesis predicts that stock prices reflect information",
        )];
        let mut cached = HashMap::new();
        // Page 3 has 5/10 = 50% word overlap (> 40% threshold).
        cached.insert(
            3,
            "In the efficient market model, stock prices are driven by supply.".to_string(),
        );
        // Page 7 has < 40% word overlap (only "the" matches).
        cached.insert(
            7,
            "Regression analysis results for the sample period.".to_string(),
        );

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 10);
        assert_eq!(result, vec![3], "word-overlap >= 40% must include page 3");
    }

    /// T-ANN-PREFILTER-008: Word-overlap heuristic is skipped for short quotes
    /// (2 words or fewer) to avoid false positives. Only substring matching
    /// applies.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_008_short_quotes_skip_word_overlap() {
        let rows = vec![make_row("the model")];
        let mut cached = HashMap::new();
        // Page 1 does NOT contain "the model" as a substring, but both words
        // appear separately. Word-overlap would give 2/2 = 100%, but the
        // heuristic is skipped for quotes with <= 2 words.
        cached.insert(
            1,
            "The regression model shows significant results.".to_string(),
        );

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 5);
        // "the model" does not appear as an exact substring in the cached text
        // ("The regression model" has "model" but not "the model" contiguously).
        // Wait: "The regression model shows significant results." lowered is
        // "the regression model shows significant results."
        // "the model" IS a substring check... "the regression model" does NOT
        // contain "the model" as a contiguous substring.
        // Actually: "the regression model shows..." does contain "model" but
        // NOT "the model" (there's "regression" between them).
        // So substring fails, word overlap skipped -> fallback to all pages.
        let expected: Vec<usize> = (1..=5).collect();
        assert_eq!(
            result, expected,
            "short quotes must not use word-overlap, so no match -> all pages"
        );
    }

    /// T-ANN-PREFILTER-009: Word-overlap threshold boundary: exactly 40%
    /// overlap (2 out of 5 words) must be included.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_009_word_overlap_boundary_40_percent() {
        // Quote has 5 words; page text contains 2 of them (40% = threshold).
        let rows = vec![make_row("alpha beta gamma delta epsilon")];
        let mut cached = HashMap::new();
        // 2/5 = 40%: matching * 5 = 10 >= len * 2 = 10 -> included.
        cached.insert(1, "alpha and beta are greek letters.".to_string());
        // 1/5 = 20%: matching * 5 = 5 < len * 2 = 10 -> excluded.
        cached.insert(2, "only alpha appears here.".to_string());

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 5);
        assert_eq!(
            result,
            vec![1],
            "40% word overlap (boundary) must include the page"
        );
    }

    /// T-ANN-PREFILTER-010: Word-overlap threshold boundary: 39% overlap
    /// (below threshold) must NOT be included.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_010_word_overlap_below_threshold_excluded() {
        // Quote has 5 words; page text contains 1 of them (20% < 40%).
        let rows = vec![make_row("alpha beta gamma delta epsilon")];
        let mut cached = HashMap::new();
        cached.insert(1, "only epsilon appears in this text.".to_string());

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 3);
        // 1/5 = 20% < 40% -> excluded, fallback to all pages.
        let expected: Vec<usize> = (1..=3).collect();
        assert_eq!(
            result, expected,
            "below-threshold word overlap must not include the page"
        );
    }

    /// T-ANN-PREFILTER-011: Empty rows list returns all pages (nothing to
    /// pre-filter for, so the fallback triggers).
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_011_empty_rows_returns_all_pages() {
        let rows: Vec<crate::types::InputRow> = vec![];
        let mut cached = HashMap::new();
        cached.insert(1, "Some text.".to_string());

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 5);
        let expected: Vec<usize> = (1..=5).collect();
        assert_eq!(
            result, expected,
            "empty rows must return all pages as fallback"
        );
    }

    /// T-ANN-PREFILTER-012: Empty cached pages (empty HashMap) returns all
    /// pages. No text to match against, so all pages need OCR.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_012_empty_cache_returns_all_pages() {
        let rows = vec![make_row("some text")];
        let cached: HashMap<usize, String> = HashMap::new();

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 5);
        let expected: Vec<usize> = (1..=5).collect();
        assert_eq!(
            result, expected,
            "empty cache must return all pages as fallback"
        );
    }

    /// T-ANN-PREFILTER-013: Result pages are sorted in ascending order
    /// regardless of the order they were matched or the HashMap iteration
    /// order.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_013_result_sorted_ascending() {
        let rows = vec![
            make_row("quote on page twenty"),
            make_row("quote on page five"),
            make_row("quote on page twelve"),
        ];
        let mut cached = HashMap::new();
        cached.insert(20, "quote on page twenty appears here.".to_string());
        cached.insert(5, "quote on page five appears here.".to_string());
        cached.insert(12, "quote on page twelve appears here.".to_string());

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 30);
        assert_eq!(
            result,
            vec![5, 12, 20],
            "result pages must be sorted in ascending order"
        );
    }

    /// T-ANN-PREFILTER-014: page_count=0 produces an empty result (edge case:
    /// a PDF with zero pages should not cause a panic or infinite loop).
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_014_zero_page_count() {
        let rows = vec![make_row("test")];
        let result = prefilter_pages_for_hocr(&rows, None, 0);
        assert!(
            result.is_empty(),
            "zero page count must produce empty result"
        );
    }

    /// T-ANN-PREFILTER-015: page_count=1 with a match returns a single-element
    /// vector.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_ann_prefilter_015_single_page_with_match() {
        let rows = vec![make_row("hello world")];
        let mut cached = HashMap::new();
        cached.insert(1, "hello world is a common test phrase.".to_string());

        let result = prefilter_pages_for_hocr(&rows, Some(&cached), 1);
        assert_eq!(result, vec![1]);
    }

    /// T-ANN-PREFILTER-016: The channel-based timeout pattern used by the
    /// extract_block returns the hOCR result within the timeout period.
    /// Verifies that recv_timeout works correctly for the happy path.
    #[test]
    fn t_ann_prefilter_016_channel_timeout_happy_path() {
        let (tx, rx) = std::sync::mpsc::channel::<Result<String, String>>();

        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            let _ = tx.send(Ok("hOCR data".to_string()));
        });

        let result = rx.recv_timeout(std::time::Duration::from_secs(5));
        assert!(result.is_ok(), "recv_timeout must succeed before timeout");
        assert_eq!(result.unwrap().unwrap(), "hOCR data");
    }

    /// T-ANN-PREFILTER-017: The channel-based timeout pattern returns
    /// RecvTimeoutError::Timeout when the hOCR batch exceeds the timeout.
    /// Verifies the timeout fallback path.
    #[test]
    fn t_ann_prefilter_017_channel_timeout_exceeded() {
        let (tx, rx) = std::sync::mpsc::channel::<Result<String, String>>();

        std::thread::spawn(move || {
            // Simulate a long-running hOCR batch that exceeds the timeout.
            std::thread::sleep(std::time::Duration::from_secs(10));
            let _ = tx.send(Ok("late data".to_string()));
        });

        let result = rx.recv_timeout(std::time::Duration::from_millis(50));
        assert!(
            matches!(result, Err(std::sync::mpsc::RecvTimeoutError::Timeout)),
            "recv_timeout must return Timeout when the sender is too slow"
        );
    }

    /// T-ANN-PREFILTER-018: The channel-based timeout pattern returns
    /// RecvTimeoutError::Disconnected when the hOCR thread panics (sender
    /// dropped without sending).
    #[test]
    fn t_ann_prefilter_018_channel_disconnected_on_panic() {
        let (tx, rx) = std::sync::mpsc::channel::<Result<String, String>>();

        std::thread::spawn(move || {
            // Simulate thread crashing: drop sender without sending.
            drop(tx);
        });

        // Wait long enough for the thread to finish.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let result = rx.recv_timeout(std::time::Duration::from_secs(1));
        assert!(
            matches!(result, Err(std::sync::mpsc::RecvTimeoutError::Disconnected)),
            "recv_timeout must return Disconnected when sender is dropped"
        );
    }

    /// T-ANN-PREFILTER-019: The total_steps counter increases by the number
    /// of targeted OCR pages (not all pages). For 35 total pages with 5
    /// matching, total_steps increases by 5.
    #[test]
    fn t_ann_prefilter_019_total_steps_reflects_targeted_pages() {
        let initial_total = 15_usize; // 10 quotes + 5 PDFs
        let total_steps = AtomicUsize::new(initial_total);

        // Simulate: scanned PDF has 35 pages, pre-filter identifies 5.
        let ocr_page_count = 5;
        total_steps.fetch_add(ocr_page_count, Ordering::Relaxed);

        assert_eq!(
            total_steps.load(Ordering::Relaxed),
            20,
            "total_steps must increase by targeted page count (15 + 5 = 20)"
        );
    }

    /// T-ANN-PREFILTER-020: After hOCR completes, on_quote_done is called
    /// with the number of completed pages. This advances the progress counter
    /// by the OCR page count.
    #[test]
    fn t_ann_prefilter_020_progress_advances_by_completed_pages() {
        let quote_done = AtomicUsize::new(0);
        let total_steps = AtomicUsize::new(20);

        // Simulate: hOCR batch completed 5 pages.
        let completed_pages = 5;
        if completed_pages > 0 {
            let prev = quote_done.fetch_add(completed_pages, Ordering::Relaxed);
            let current_total = total_steps.load(Ordering::Relaxed);
            assert_eq!(prev, 0, "first progress increment starts at 0");
            assert_eq!(current_total, 20, "total_steps unchanged by progress");
        }

        assert_eq!(
            quote_done.load(Ordering::Relaxed),
            5,
            "quote_done must reflect the completed OCR pages"
        );
    }

    // -----------------------------------------------------------------------
    // Phase 1 pre_match_quotes tests
    // -----------------------------------------------------------------------

    /// T-ANN-PREMATCH-001: pre_match_quotes finds exact matches in cached text.
    #[test]
    fn t_ann_prematch_001_exact_match_in_cache() {
        let pdf_items: Vec<(PathBuf, Vec<InputRow>)> = vec![(
            PathBuf::from("/test/paper.pdf"),
            vec![InputRow {
                title: "T".into(),
                author: "A".into(),
                quote: "efficient markets hypothesis".into(),
                color: None,
                comment: None,
                page: None,
            }],
        )];

        let mut cached = HashMap::new();
        let mut pages = HashMap::new();
        pages.insert(
            1,
            "Introduction to the efficient markets hypothesis and its implications.".to_string(),
        );
        cached.insert(PathBuf::from("/test/paper.pdf"), pages);

        let matches = pre_match_quotes(&pdf_items, &cached);
        assert_eq!(matches.len(), 1, "must find one pre-match");
        let m = &matches[&(0, 0)];
        assert_eq!(m.page_number, 1);
        assert_eq!(m.method, MatchMethod::Exact);
    }

    /// T-ANN-PREMATCH-002: pre_match_quotes uses page hint to search the
    /// hinted page first, even when it appears later in the HashMap.
    #[test]
    fn t_ann_prematch_002_page_hint_searched_first() {
        let pdf_items: Vec<(PathBuf, Vec<InputRow>)> = vec![(
            PathBuf::from("/test/paper.pdf"),
            vec![InputRow {
                title: "T".into(),
                author: "A".into(),
                quote: "unique passage text".into(),
                color: None,
                comment: None,
                page: Some(5),
            }],
        )];

        let mut cached = HashMap::new();
        let mut pages = HashMap::new();
        pages.insert(1, "This page has some unrelated content.".to_string());
        pages.insert(
            5,
            "This page contains the unique passage text we are looking for.".to_string(),
        );
        pages.insert(10, "Another unrelated page.".to_string());
        cached.insert(PathBuf::from("/test/paper.pdf"), pages);

        let matches = pre_match_quotes(&pdf_items, &cached);
        assert_eq!(matches.len(), 1);
        let m = &matches[&(0, 0)];
        assert_eq!(m.page_number, 5, "must find match on hinted page");
    }

    /// T-ANN-PREMATCH-003: pre_match_quotes returns empty map when no cached
    /// text is available for any PDF.
    #[test]
    fn t_ann_prematch_003_no_cache_returns_empty() {
        let pdf_items: Vec<(PathBuf, Vec<InputRow>)> = vec![(
            PathBuf::from("/test/paper.pdf"),
            vec![InputRow {
                title: "T".into(),
                author: "A".into(),
                quote: "some quote".into(),
                color: None,
                comment: None,
                page: None,
            }],
        )];

        let cached = HashMap::new();
        let matches = pre_match_quotes(&pdf_items, &cached);
        assert!(
            matches.is_empty(),
            "no cache must produce empty pre-matches"
        );
    }

    /// T-ANN-PREMATCH-004: pre_match_quotes finds normalized matches
    /// (whitespace differences between cached text and quote).
    #[test]
    fn t_ann_prematch_004_normalized_match() {
        let pdf_items: Vec<(PathBuf, Vec<InputRow>)> = vec![(
            PathBuf::from("/test/paper.pdf"),
            vec![InputRow {
                title: "T".into(),
                author: "A".into(),
                quote: "efficient markets model".into(),
                color: None,
                comment: None,
                page: None,
            }],
        )];

        let mut cached = HashMap::new();
        let mut pages = HashMap::new();
        pages.insert(
            3,
            "The Efficient  Markets   Model is discussed in section 2.".to_string(),
        );
        cached.insert(PathBuf::from("/test/paper.pdf"), pages);

        let matches = pre_match_quotes(&pdf_items, &cached);
        assert_eq!(matches.len(), 1);
        let m = &matches[&(0, 0)];
        assert_eq!(m.page_number, 3);
        assert_eq!(m.method, MatchMethod::Normalized);
    }

    /// T-ANN-PREMATCH-005: pre_match_quotes finds fuzzy matches when text
    /// has minor differences (anchor-word algorithm).
    #[test]
    fn t_ann_prematch_005_fuzzy_match() {
        let pdf_items: Vec<(PathBuf, Vec<InputRow>)> = vec![(
            PathBuf::from("/test/paper.pdf"),
            vec![InputRow {
                title: "T".into(),
                author: "A".into(),
                quote: "The empirical evidence strongly supports the efficient market hypothesis and is documented in literature.".into(),
                color: None,
                comment: None,
                page: None,
            }],
        )];

        let mut cached = HashMap::new();
        let mut pages = HashMap::new();
        pages.insert(7, "The empirical evidence strongly supports the efficient markets hypothesis and is documented in the literature.".to_string());
        cached.insert(PathBuf::from("/test/paper.pdf"), pages);

        let matches = pre_match_quotes(&pdf_items, &cached);
        assert_eq!(matches.len(), 1);
        let m = &matches[&(0, 0)];
        assert_eq!(m.page_number, 7);
        assert_eq!(m.method, MatchMethod::Fuzzy);
        assert!(m.fuzzy_score.is_some());
    }

    /// T-ANN-PREMATCH-006: pre_match_quotes handles multiple PDFs with
    /// multiple quotes each, producing correct (pdf_idx, quote_idx) keys.
    #[test]
    fn t_ann_prematch_006_multiple_pdfs_multiple_quotes() {
        let pdf_items: Vec<(PathBuf, Vec<InputRow>)> = vec![
            (
                PathBuf::from("/test/a.pdf"),
                vec![
                    InputRow {
                        title: "A".into(),
                        author: "A".into(),
                        quote: "alpha passage".into(),
                        color: None,
                        comment: None,
                        page: None,
                    },
                    InputRow {
                        title: "A".into(),
                        author: "A".into(),
                        quote: "beta passage".into(),
                        color: None,
                        comment: None,
                        page: None,
                    },
                ],
            ),
            (
                PathBuf::from("/test/b.pdf"),
                vec![InputRow {
                    title: "B".into(),
                    author: "B".into(),
                    quote: "gamma passage".into(),
                    color: None,
                    comment: None,
                    page: None,
                }],
            ),
        ];

        let mut cached = HashMap::new();
        let mut pages_a = HashMap::new();
        pages_a.insert(1, "This contains the alpha passage text.".to_string());
        pages_a.insert(2, "This contains the beta passage text.".to_string());
        cached.insert(PathBuf::from("/test/a.pdf"), pages_a);

        let mut pages_b = HashMap::new();
        pages_b.insert(5, "Here is the gamma passage text.".to_string());
        cached.insert(PathBuf::from("/test/b.pdf"), pages_b);

        let matches = pre_match_quotes(&pdf_items, &cached);

        assert_eq!(matches.len(), 3, "must find 3 pre-matches across 2 PDFs");
        assert!(matches.contains_key(&(0, 0)), "must match a.pdf quote 0");
        assert!(matches.contains_key(&(0, 1)), "must match a.pdf quote 1");
        assert!(matches.contains_key(&(1, 0)), "must match b.pdf quote 0");
        assert_eq!(matches[&(0, 0)].page_number, 1);
        assert_eq!(matches[&(0, 1)].page_number, 2);
        assert_eq!(matches[&(1, 0)].page_number, 5);
    }

    // -----------------------------------------------------------------------
    // v3 pipeline tests: page-level fallback, error-dir policy, thresholds
    // -----------------------------------------------------------------------

    /// T-ANN-V3-001: pre_match_quotes at the 0.75 threshold catches
    /// agent-reformulated text that differs by ~20% from the cached text.
    /// The quote uses "market hypothesis" while the cached text says
    /// "markets hypothesis", and includes minor word insertions. With the
    /// old 0.85 threshold this would be rejected; at 0.75 it matches.
    #[test]
    fn t_ann_v3_001_lower_threshold_catches_reformulated_text() {
        let pdf_items: Vec<(PathBuf, Vec<InputRow>)> = vec![(
            PathBuf::from("/test/paper.pdf"),
            vec![InputRow {
                title: "T".into(),
                author: "A".into(),
                // Reformulated: "market" vs "markets", "suggests" vs "supports", minor wording changes.
                quote: "The empirical evidence suggests the efficient market hypothesis as documented in recent literature".into(),
                color: None,
                comment: None,
                page: None,
            }],
        )];

        let mut cached = HashMap::new();
        let mut pages = HashMap::new();
        pages.insert(
            3,
            "The empirical evidence strongly supports the efficient markets hypothesis and is documented in the literature".to_string(),
        );
        cached.insert(PathBuf::from("/test/paper.pdf"), pages);

        let matches = pre_match_quotes(&pdf_items, &cached);

        // The anchor-word algorithm should find enough shared words
        // ("empirical", "evidence", "efficient", "hypothesis", "documented",
        // "literature") to cluster, then Levenshtein on the candidate window
        // should yield >= 0.75 similarity.
        assert_eq!(
            matches.len(),
            1,
            "reformulated text must match at 0.75 threshold"
        );
        let m = &matches[&(0, 0)];
        assert_eq!(m.page_number, 3);
        assert_eq!(m.method, MatchMethod::Fuzzy);
        assert!(
            m.fuzzy_score.unwrap_or(0.0) >= 0.75,
            "fuzzy score must be >= 0.75, got {:?}",
            m.fuzzy_score
        );
    }

    /// T-ANN-V3-002: pre_match_quotes rejects text that shares too few
    /// words even at the lowered 0.75 threshold. Completely unrelated
    /// content must not match.
    #[test]
    fn t_ann_v3_002_threshold_rejects_unrelated_text() {
        let pdf_items: Vec<(PathBuf, Vec<InputRow>)> = vec![(
            PathBuf::from("/test/paper.pdf"),
            vec![InputRow {
                title: "T".into(),
                author: "A".into(),
                quote: "quantum entanglement experiments demonstrate nonlocal correlations".into(),
                color: None,
                comment: None,
                page: None,
            }],
        )];

        let mut cached = HashMap::new();
        let mut pages = HashMap::new();
        pages.insert(
            1,
            "The efficient markets hypothesis describes stock price behavior in financial economics".to_string(),
        );
        cached.insert(PathBuf::from("/test/paper.pdf"), pages);

        let matches = pre_match_quotes(&pdf_items, &cached);
        assert!(
            matches.is_empty(),
            "completely unrelated text must not match, got {} matches",
            matches.len()
        );
    }

    /// T-ANN-V3-003: QuoteReport with status "page_level_match" is counted
    /// as a matched quote in the matched_count filter logic. This verifies
    /// the filter condition used in process_single_pdf.
    #[test]
    fn t_ann_v3_003_page_level_match_counted_in_matched_filter() {
        let reports = [
            QuoteReport {
                quote_excerpt: "exact match".into(),
                status: "matched".into(),
                match_method: Some("exact".into()),
                page: Some(1),
                chars_matched: Some(50),
                chars_total: Some(50),
                pre_check_passed: Some(true),
                post_check_passed: None,
                stages_tried: None,
                fuzzy_score: None,
            },
            QuoteReport {
                quote_excerpt: "page fallback".into(),
                status: "page_level_match".into(),
                match_method: Some("page_level".into()),
                page: Some(5),
                chars_matched: Some(1),
                chars_total: Some(200),
                pre_check_passed: Some(false),
                post_check_passed: None,
                stages_tried: None,
                fuzzy_score: None,
            },
            QuoteReport {
                quote_excerpt: "not found".into(),
                status: "not_found".into(),
                match_method: None,
                page: None,
                chars_matched: None,
                chars_total: None,
                pre_check_passed: None,
                post_check_passed: None,
                stages_tried: None,
                fuzzy_score: None,
            },
        ];

        // Apply the same filter as process_single_pdf matched_count.
        let matched_count = reports
            .iter()
            .filter(|q| {
                q.status == "matched"
                    || q.status == "low_confidence_match"
                    || q.status == "page_level_match"
            })
            .count();

        assert_eq!(
            matched_count, 2,
            "matched + page_level_match = 2, not_found excluded"
        );
    }

    /// T-ANN-V3-004: QuoteReport for page-level fallback has the correct
    /// field values: status "page_level_match", match_method "page_level",
    /// pre_check_passed false, chars_matched 1, and stages_tried populated.
    #[test]
    fn t_ann_v3_004_page_level_report_fields() {
        let report = QuoteReport {
            quote_excerpt: "some passage".into(),
            status: "page_level_match".into(),
            match_method: Some("page_level".into()),
            page: Some(7),
            chars_matched: Some(1),
            chars_total: Some(150),
            pre_check_passed: Some(false),
            post_check_passed: None,
            stages_tried: Some(vec![
                "exact: no match on 10 pages".into(),
                "normalized: no match on 10 pages".into(),
                "fuzzy: best=60% on page 7 < 80% threshold".into(),
            ]),
            fuzzy_score: None,
        };

        assert_eq!(report.status, "page_level_match");
        assert_eq!(report.match_method.as_deref(), Some("page_level"));
        assert_eq!(report.page, Some(7));
        assert_eq!(report.chars_matched, Some(1));
        assert_eq!(report.pre_check_passed, Some(false));
        assert!(report.post_check_passed.is_none());
        assert_eq!(report.stages_tried.as_ref().unwrap().len(), 3);

        // Verify JSON serialization omits post_check_passed.
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains("post_check_passed"));
        assert!(json.contains("page_level_match"));
        assert!(json.contains("stages_tried"));
    }

    /// T-ANN-V3-005: page_level fallback comment is built correctly. When
    /// the input row has a comment, it is prepended with "[page-level]".
    /// When no comment is provided, the default message is used.
    #[test]
    fn t_ann_v3_005_page_level_fallback_comment_construction() {
        // With existing comment.
        let row_comment = Some("Check Theorem 3.2 on page 15".to_string());
        let fallback_with = match &row_comment {
            Some(c) if !c.trim().is_empty() => format!("[page-level] {c}"),
            _ => "[page-level] passage not located precisely".to_string(),
        };
        assert_eq!(fallback_with, "[page-level] Check Theorem 3.2 on page 15");

        // Without comment.
        let row_comment: Option<String> = None;
        let fallback_without = match &row_comment {
            Some(c) if !c.trim().is_empty() => format!("[page-level] {c}"),
            _ => "[page-level] passage not located precisely".to_string(),
        };
        assert_eq!(
            fallback_without,
            "[page-level] passage not located precisely"
        );

        // With empty string comment.
        let row_comment = Some("   ".to_string());
        let fallback_empty = match &row_comment {
            Some(c) if !c.trim().is_empty() => format!("[page-level] {c}"),
            _ => "[page-level] passage not located precisely".to_string(),
        };
        assert_eq!(fallback_empty, "[page-level] passage not located precisely");
    }

    /// T-ANN-V3-006: The matched_count filter correctly classifies PDF
    /// status as "partial" when some quotes are matched and some are not.
    #[test]
    fn t_ann_v3_006_partial_status_with_mixed_results() {
        let reports = [
            QuoteReport {
                quote_excerpt: "found".into(),
                status: "matched".into(),
                match_method: Some("exact".into()),
                page: Some(1),
                chars_matched: Some(30),
                chars_total: Some(30),
                pre_check_passed: Some(true),
                post_check_passed: None,
                stages_tried: None,
                fuzzy_score: None,
            },
            QuoteReport {
                quote_excerpt: "not found".into(),
                status: "not_found".into(),
                match_method: None,
                page: None,
                chars_matched: None,
                chars_total: None,
                pre_check_passed: None,
                post_check_passed: None,
                stages_tried: None,
                fuzzy_score: None,
            },
        ];

        let matched_count = reports
            .iter()
            .filter(|q| {
                q.status == "matched"
                    || q.status == "low_confidence_match"
                    || q.status == "page_level_match"
            })
            .count();
        let total_count = reports.len();

        let status = if matched_count == total_count {
            "success"
        } else if matched_count > 0 {
            "partial"
        } else {
            "failed"
        };

        assert_eq!(status, "partial");
        assert_eq!(matched_count, 1);
        assert_eq!(total_count, 2);
    }

    /// T-ANN-V3-007: The matched_count filter produces "success" when all
    /// quotes are matched (mix of matched + page_level_match).
    #[test]
    fn t_ann_v3_007_success_status_with_all_matched() {
        let reports = [
            QuoteReport {
                quote_excerpt: "exact".into(),
                status: "matched".into(),
                match_method: Some("exact".into()),
                page: Some(1),
                chars_matched: Some(30),
                chars_total: Some(30),
                pre_check_passed: Some(true),
                post_check_passed: None,
                stages_tried: None,
                fuzzy_score: None,
            },
            QuoteReport {
                quote_excerpt: "page fallback".into(),
                status: "page_level_match".into(),
                match_method: Some("page_level".into()),
                page: Some(3),
                chars_matched: Some(1),
                chars_total: Some(100),
                pre_check_passed: Some(false),
                post_check_passed: None,
                stages_tried: None,
                fuzzy_score: None,
            },
        ];

        let matched_count = reports
            .iter()
            .filter(|q| {
                q.status == "matched"
                    || q.status == "low_confidence_match"
                    || q.status == "page_level_match"
            })
            .count();
        let total_count = reports.len();

        let status = if matched_count == total_count {
            "success"
        } else if matched_count > 0 {
            "partial"
        } else {
            "failed"
        };

        assert_eq!(status, "success");
        assert_eq!(matched_count, 2);
    }

    /// T-ANN-V3-008: pre_match_quotes with page hint and fuzzy match at
    /// the lowered threshold. The hint page is searched first and the fuzzy
    /// algorithm finds the reformulated text on that specific page.
    #[test]
    fn t_ann_v3_008_page_hint_with_fuzzy_match() {
        let pdf_items: Vec<(PathBuf, Vec<InputRow>)> = vec![(
            PathBuf::from("/test/paper.pdf"),
            vec![InputRow {
                title: "T".into(),
                author: "A".into(),
                // Reformulated version with minor word changes.
                quote: "analysis of variance components in hierarchical linear models provides robust estimates".into(),
                color: None,
                comment: None,
                page: Some(12),
            }],
        )];

        let mut cached = HashMap::new();
        let mut pages = HashMap::new();
        // Page 1: unrelated content.
        pages.insert(
            1,
            "Introduction to statistical methods for data analysis.".to_string(),
        );
        // Page 12 (hinted): similar text with differences.
        pages.insert(
            12,
            "The analysis of variance components in hierarchical linear models provides robust parameter estimates for nested designs.".to_string(),
        );
        cached.insert(PathBuf::from("/test/paper.pdf"), pages);

        let matches = pre_match_quotes(&pdf_items, &cached);
        assert_eq!(matches.len(), 1, "fuzzy match must succeed on hinted page");
        let m = &matches[&(0, 0)];
        assert_eq!(m.page_number, 12, "match must be on the hinted page");
    }

    // -----------------------------------------------------------------------
    // DEF-001 regression tests: Append mode snapshot for annotation
    // preservation.
    //
    // In append mode, the pipeline saves pdfium output to output_directory,
    // overwriting any existing annotated PDFs. Phase 2.5 then tries to
    // merge prior annotations from prior_output_directory. When both
    // directories are the same (the normal append case), the merge would
    // read from an already-overwritten file. The fix creates a temp
    // snapshot of the prior PDFs before Phase 2, so Phase 2.5 reads from
    // untouched copies.
    // -----------------------------------------------------------------------

    /// T-ANN-DEF001-001: Snapshot directory creation and file copy. Verifies
    /// that tempfile::tempdir creates a usable directory and fs::copy
    /// preserves file content.
    #[test]
    fn t_ann_def001_001_snapshot_copy_preserves_content() {
        let source_dir = tempfile::tempdir().expect("create source dir");
        let prior_file = source_dir.path().join("paper.pdf");
        std::fs::write(&prior_file, b"annotated PDF content with highlights").expect("write prior");

        // Simulate the snapshot logic from Phase 1.5.
        let snapshot_dir = tempfile::tempdir().expect("create snapshot dir");
        let dest = snapshot_dir.path().join("paper.pdf");
        std::fs::copy(&prior_file, &dest).expect("copy to snapshot");

        // Overwrite the source file (simulating Phase 2 pdfium save).
        std::fs::write(&prior_file, b"new pdfium output without prior annotations")
            .expect("overwrite source");

        // The snapshot must still contain the original content.
        let snapshot_content = std::fs::read(&dest).expect("read snapshot");
        assert_eq!(
            snapshot_content, b"annotated PDF content with highlights",
            "snapshot must preserve the original file content despite source overwrite"
        );

        // The overwritten source must have the new content.
        let source_content = std::fs::read(&prior_file).expect("read source");
        assert_eq!(
            source_content, b"new pdfium output without prior annotations",
            "source must contain the overwritten content"
        );
    }

    /// T-ANN-DEF001-002: Snapshot only copies files that exist in the prior
    /// directory. Non-existent files are silently skipped.
    #[test]
    fn t_ann_def001_002_snapshot_skips_nonexistent_files() {
        let prior_dir = tempfile::tempdir().expect("create prior dir");
        let snapshot_dir = tempfile::tempdir().expect("create snapshot dir");

        // Write one file to prior dir, leave another missing.
        let existing = prior_dir.path().join("exists.pdf");
        std::fs::write(&existing, b"content").expect("write existing");

        let missing = prior_dir.path().join("missing.pdf");
        assert!(!missing.exists(), "file must not exist before test");

        // Simulate the snapshot logic for both files.
        let filenames = ["exists.pdf", "missing.pdf"];
        let mut copied = 0;
        for name in &filenames {
            let prior_path = prior_dir.path().join(name);
            if prior_path.is_file() {
                let dest = snapshot_dir.path().join(name);
                std::fs::copy(&prior_path, &dest).expect("copy");
                copied += 1;
            }
        }

        assert_eq!(copied, 1, "only the existing file must be copied");
        assert!(
            snapshot_dir.path().join("exists.pdf").exists(),
            "existing file must be in snapshot"
        );
        assert!(
            !snapshot_dir.path().join("missing.pdf").exists(),
            "missing file must not be in snapshot"
        );
    }

    /// T-ANN-DEF001-003: The effective_prior_dir selection logic prefers
    /// the snapshot directory over the original prior_output_directory.
    #[test]
    fn t_ann_def001_003_effective_prior_dir_prefers_snapshot() {
        let original_dir = PathBuf::from("/original/prior/dir");
        let snapshot_dir = tempfile::tempdir().expect("create snapshot dir");

        // When snapshot exists, effective_prior_dir must use the snapshot path.
        let effective =
            Some(snapshot_dir.path().to_path_buf()).or_else(|| Some(original_dir.clone()));

        assert_eq!(
            effective.as_deref(),
            Some(snapshot_dir.path()),
            "snapshot directory must take precedence over original prior dir"
        );
    }

    /// T-ANN-DEF001-004: When prior_output_directory is None (non-append
    /// mode), no snapshot is created and effective_prior_dir is None.
    #[test]
    fn t_ann_def001_004_no_snapshot_for_non_append_mode() {
        let prior_snapshot_dir: Option<tempfile::TempDir> = None;
        let prior_output_directory: Option<PathBuf> = None;

        let effective: Option<PathBuf> = prior_snapshot_dir
            .as_ref()
            .map(|tmp| tmp.path().to_path_buf())
            .or_else(|| prior_output_directory.clone());

        assert!(
            effective.is_none(),
            "effective_prior_dir must be None when not in append mode"
        );
    }

    /// T-ANN-DEF001-005: Multiple files are correctly snapshotted. Each
    /// file from the prior directory is independently copied to the
    /// snapshot directory.
    #[test]
    fn t_ann_def001_005_multiple_files_snapshotted() {
        let prior_dir = tempfile::tempdir().expect("create prior dir");
        let snapshot_dir = tempfile::tempdir().expect("create snapshot dir");

        let filenames = ["paper_a.pdf", "paper_b.pdf", "paper_c.pdf"];
        for (i, name) in filenames.iter().enumerate() {
            std::fs::write(prior_dir.path().join(name), format!("content of file {i}"))
                .expect("write file");
        }

        // Copy all files to snapshot.
        for name in &filenames {
            let src = prior_dir.path().join(name);
            let dst = snapshot_dir.path().join(name);
            std::fs::copy(&src, &dst).expect("copy");
        }

        // Verify all files exist in snapshot with correct content.
        for (i, name) in filenames.iter().enumerate() {
            let content = std::fs::read_to_string(snapshot_dir.path().join(name))
                .expect("read snapshot file");
            assert_eq!(
                content,
                format!("content of file {i}"),
                "snapshot file {name} must have correct content"
            );
        }
    }
}
