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

//! Tesseract OCR fallback for image-heavy PDF pages (feature-gated: `ocr`).
//!
//! When a page's extracted text falls below the quality threshold defined in the
//! `quality` module, this module renders the page to a raster image via pdfium
//! and passes it through Tesseract OCR via CLI subprocess invocation.
//!
//! ## Fully parallel render+OCR pipeline
//!
//! The batch functions (`ocr_pdf_pages_batch`, `ocr_pdf_pages_batch_hocr`) use
//! a fully parallel pipeline where each rayon thread independently:
//!
//! 1. Loads its own pdfium library instance (thread-local state)
//! 2. Opens the PDF document
//! 3. Renders its assigned page to a PNG byte buffer
//! 4. Drops pdfium resources
//! 5. Spawns a Tesseract CLI subprocess (`tesseract stdin stdout`) to OCR the PNG
//!
//! pdfium's `PdfDocument` is not `Send`/`Sync`, preventing shared access across
//! threads. However, pdfium uses thread-local storage for its global state
//! (`FPDF_InitLibraryWithConfig`), so each thread can safely create its own
//! independent `Pdfium` + `PdfDocument` instances. This parallelizes both the
//! render and OCR phases, eliminating the sequential rendering bottleneck
//! (~2.5s/page × 35 pages = ~87s sequential → ~10s parallel with 8 threads).
//!
//! Tesseract processes are restricted to 1 OpenMP thread each
//! (`OMP_NUM_THREADS=1`), with rayon controlling the inter-page parallelism.
//!
//! Tesseract is invoked as a CLI subprocess rather than linked via FFI. This
//! eliminates the build-time dependency on Leptonica and Tesseract C libraries
//! (previously required by the `leptess` crate), making the OCR feature
//! accessible without vcpkg or system package installation.
//!
//! This module is only compiled when the `ocr` Cargo feature is enabled. The
//! `ocr` feature implicitly enables the `pdfium` feature because page rendering
//! requires the pdfium backend.

#[cfg(feature = "ocr")]
use std::path::{Path, PathBuf};
#[cfg(feature = "ocr")]
use std::sync::LazyLock;

#[cfg(feature = "ocr")]
use crate::error::PdfError;

/// Maximum pixel dimension (width or height) allowed when rendering a PDF page
/// for Tesseract OCR. i32 is the type accepted by pdfium's render configuration.
/// Clamping to 32_767 avoids i32 overflow and bitmap memory exhaustion for PDFs
/// with extreme page dimensions (e.g., engineering drawings, poster-size scans).
#[cfg(feature = "ocr")]
const MAX_OCR_DIMENSION_PX: f64 = 32_767.0;

/// Shared rayon thread pool used for all parallel OCR operations.
/// A single static pool prevents the thread count from growing unboundedly when
/// multiple large PDFs are indexed concurrently. `LazyLock` initializes the pool
/// on first use. `num_threads(4)` matches typical hardware without overcommitting
/// CPU to OCR when other tasks (embedding, search) are also running.
#[cfg(feature = "ocr")]
static OCR_THREAD_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    tracing::info!(threads, "OCR thread pool initialized");
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("ocr-worker-{i}"))
        .build()
        .expect("OCR thread pool creation requires no special privileges")
});

/// Compiled regex for parsing hOCR word spans from Tesseract HTML output.
/// Stored as a module-level static via `LazyLock` so the regex is compiled
/// exactly once across all calls to `parse_hocr_words`, rather than being
/// recompiled on every invocation. The regex matches `<span class="ocrx_word">`
/// elements with both single-quote and double-quote attribute syntax.
#[cfg(feature = "ocr")]
static HOCR_WORD_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(
        r#"<span\s+class=['"]ocrx_word['"][^>]*title=['"]bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)[^'"]*['"][^>]*>([^<]+)</span>"#,
    )
    .expect("hOCR word regex must compile")
});

/// Allowlist of valid Tesseract language codes. Only these codes (and "+"
/// separated combinations of them) are accepted as the `language` parameter
/// for OCR functions. This prevents command injection through the `-l` flag
/// passed to the Tesseract CLI subprocess.
///
/// The list covers the most widely used Tesseract traineddata language packs
/// from the tessdata repository.
#[cfg(feature = "ocr")]
static ALLOWED_TESSERACT_LANGUAGES: &[&str] = &[
    "eng", "deu", "fra", "spa", "ita", "por", "nld", "pol", "rus", "chi_sim", "chi_tra", "jpn",
    "kor", "ara", "hin", "tur", "swe", "nor", "dan", "fin", "ces", "ron", "hun", "bul", "hrv",
    "slk", "slv", "ukr", "ell", "heb", "tha", "vie", "ind", "msa", "cat", "eus", "glg", "lat",
];

/// Validates a Tesseract language parameter against the allowlist before it is
/// passed to the Tesseract CLI as the `-l` argument.
///
/// Tesseract accepts "+" separated language codes (e.g., "eng+deu"). This
/// function splits on "+" and checks each code against `ALLOWED_TESSERACT_LANGUAGES`.
///
/// Rejects inputs that:
/// - Exceed 100 characters (prevents abuse with excessively long arguments)
/// - Contain path separators (`/`, `\`) or shell metacharacters (`;`, `|`, `&`, `$`, `` ` ``, `(`, `)`)
/// - Contain any language code not present in the allowlist
///
/// # Errors
///
/// Returns [`PdfError::Ocr`] with a descriptive message if validation fails.
#[cfg(feature = "ocr")]
fn validate_tesseract_language(language: &str) -> Result<(), PdfError> {
    // Reject excessively long language strings to prevent abuse.
    if language.len() > 100 {
        return Err(PdfError::Ocr(format!(
            "language parameter exceeds 100 characters (got {})",
            language.len()
        )));
    }

    // Reject characters that could be interpreted by the shell or filesystem.
    // The Tesseract CLI receives `language` as a single argument via
    // `Command::arg`, which does not invoke a shell. However, path separators
    // and metacharacters have no legitimate use in language codes, so rejecting
    // them provides defense-in-depth.
    const FORBIDDEN_CHARS: &[char] = &['/', '\\', ';', '|', '&', '$', '`', '(', ')'];
    if let Some(bad) = language.chars().find(|c| FORBIDDEN_CHARS.contains(c)) {
        return Err(PdfError::Ocr(format!(
            "language parameter contains forbidden character '{bad}'"
        )));
    }

    // Split on "+" and validate each individual language code. Tesseract
    // uses "+" as the separator for multi-language OCR (e.g., "eng+deu").
    for code in language.split('+') {
        if code.is_empty() {
            return Err(PdfError::Ocr(
                "language parameter contains empty segment (consecutive '+' or leading/trailing '+')".into()
            ));
        }
        if !ALLOWED_TESSERACT_LANGUAGES.contains(&code) {
            return Err(PdfError::Ocr(format!(
                "language code '{code}' is not in the allowed Tesseract language list"
            )));
        }
    }

    Ok(())
}

/// Threshold for "small" page dimensions in millimeters. Pages with physical
/// dimensions smaller than 150mm x 200mm (approximately A5 or smaller) are
/// rendered at a higher DPI to capture fine detail.
#[cfg(feature = "ocr")]
const SMALL_PAGE_WIDTH_MM: f64 = 150.0;
#[cfg(feature = "ocr")]
const SMALL_PAGE_HEIGHT_MM: f64 = 200.0;

/// DPI used for rendering small pages (below the dimension threshold).
#[cfg(feature = "ocr")]
const HIGH_DPI: u32 = 400;

/// DPI used for rendering standard-sized pages.
#[cfg(feature = "ocr")]
const STANDARD_DPI: u32 = 300;

/// Performs OCR on raw image bytes by writing them to a temporary file and
/// invoking the Tesseract CLI binary as a subprocess.
///
/// The `image_data` parameter contains raw PNG image bytes. The `language`
/// parameter specifies the Tesseract language code (e.g., "eng", "deu",
/// or "eng+deu" for multiple languages).
///
/// Tesseract is located via PATH or auto-downloaded on first use. The tessdata
/// directory is resolved from the auto-download cache when using a cached
/// Tesseract binary.
///
/// # Errors
///
/// Returns [`PdfError::Ocr`] if Tesseract cannot be found, the subprocess
/// fails, the output is not valid UTF-8, or the language parameter fails
/// validation against the allowlist.
/// Returns [`PdfError::DepDownload`] if Tesseract auto-download fails.
#[cfg(feature = "ocr")]
pub fn ocr_page(image_data: &[u8], language: &str) -> Result<String, PdfError> {
    validate_tesseract_language(language)?;
    let tess_path = crate::deps::ensure_tesseract()?;
    let tessdata = resolve_tessdata_dir(&tess_path);
    ocr_png_via_cli(&tess_path, tessdata.as_deref(), image_data, language)
}

/// Invokes the Tesseract CLI binary on PNG image bytes piped via stdin.
/// Captures stdout as the OCR text output.
///
/// Uses `tesseract stdin stdout` mode (Tesseract 4+/5+) where the image is
/// written to the subprocess's stdin pipe. This avoids the overhead of creating,
/// writing, and cleaning up temporary files on disk for each page.
///
/// The `tesseract_path` is the absolute path to the Tesseract executable (or
/// just "tesseract" when using the system PATH). The optional `tessdata_dir`
/// overrides the default tessdata search path via the `--tessdata-dir` CLI flag.
///
/// Each Tesseract subprocess is restricted to 1 OpenMP thread via the
/// `OMP_NUM_THREADS` environment variable. Page-level parallelism is managed
/// by rayon in `ocr_pdf_pages_batch`; allowing Tesseract's internal threading
/// would oversubscribe the CPU when multiple pages are processed concurrently.
///
/// The caller is responsible for validating the `language` parameter against
/// the allowlist via `validate_tesseract_language` before invoking this function.
/// The batch entry points (`ocr_pdf_pages_batch`, `ocr_pdf_pages_batch_hocr`)
/// and single-page entry points (`ocr_page`, `ocr_pdf_page`) perform this
/// validation. Direct callers of this internal function must do the same.
///
/// # Errors
///
/// Returns [`PdfError::Ocr`] if Tesseract cannot be spawned, the stdin pipe
/// fails, it exits with a non-zero status, or its output is not valid UTF-8.
#[cfg(feature = "ocr")]
fn ocr_png_via_cli(
    tesseract_path: &Path,
    tessdata_dir: Option<&Path>,
    png_bytes: &[u8],
    language: &str,
) -> Result<String, PdfError> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut cmd = Command::new(tesseract_path);

    // "stdin" tells Tesseract to read image data from standard input instead
    // of a file path. "stdout" directs OCR text output to standard output.
    cmd.arg("stdin").arg("stdout").arg("-l").arg(language);

    // When using a cached (auto-downloaded) Tesseract binary, the tessdata
    // directory must be specified explicitly because the binary does not know
    // the default tessdata location outside its installation prefix.
    if let Some(td) = tessdata_dir {
        cmd.arg("--tessdata-dir").arg(td);
    }

    // Restrict Tesseract to 1 OpenMP thread per process. Rayon manages
    // inter-page parallelism; allowing Tesseract's internal multi-threading
    // would oversubscribe the CPU when multiple pages run concurrently.
    // The standard OpenMP environment variable is OMP_NUM_THREADS (not
    // OMP_THREAD_NUM, which Tesseract ignores silently).
    cmd.env("OMP_NUM_THREADS", "1");

    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| PdfError::Ocr(format!("failed to spawn Tesseract subprocess: {e}")))?;

    // Write PNG bytes to Tesseract's stdin pipe. The stdin handle is taken
    // (moved out of the child) and dropped after writing to signal EOF,
    // which allows Tesseract to begin processing.
    {
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| PdfError::Ocr("failed to open Tesseract stdin pipe".into()))?;
        let mut writer = std::io::BufWriter::new(stdin);
        writer
            .write_all(png_bytes)
            .map_err(|e| PdfError::Ocr(format!("failed to write PNG to Tesseract stdin: {e}")))?;
        // BufWriter and stdin handle dropped here -> EOF signaled to Tesseract.
    }

    let output = child
        .wait_with_output()
        .map_err(|e| PdfError::Ocr(format!("Tesseract wait failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PdfError::Ocr(format!(
            "Tesseract exited with status {}: {}",
            output.status,
            stderr.trim()
        )));
    }

    String::from_utf8(output.stdout)
        .map_err(|e| PdfError::Ocr(format!("Tesseract output is not valid UTF-8: {e}")))
}

/// Resolves the tessdata directory for a given Tesseract binary path.
///
/// When the Tesseract binary is the bare name "tesseract" (system-installed),
/// the system handles tessdata resolution and no override is needed. When the
/// binary is a cached path (auto-downloaded), tessdata is expected in the
/// cache's `tessdata/` subdirectory.
///
/// Returns `None` for system-installed Tesseract, `Some(path)` for cached
/// Tesseract if the tessdata directory exists.
#[cfg(feature = "ocr")]
fn resolve_tessdata_dir(tesseract_path: &Path) -> Option<PathBuf> {
    // System-installed Tesseract: no override needed.
    if tesseract_path == Path::new("tesseract") {
        return None;
    }

    // For cached Tesseract, check the standard tessdata location within the
    // auto-download cache directory structure.
    if let Ok(td) = crate::deps::tessdata_dir()
        && td.is_dir()
    {
        return Some(td);
    }

    // For the Windows UB-Mannheim installer, tessdata is placed in the same
    // directory as the binary: <install_dir>/tessdata/.
    if let Some(parent) = tesseract_path.parent() {
        let local_tessdata = parent.join("tessdata");
        if local_tessdata.is_dir() {
            return Some(local_tessdata);
        }
    }

    None
}

/// Selects the render DPI based on page dimensions in points.
///
/// PDF page dimensions are expressed in points (1 point = 1/72 inch).
/// Converts to millimeters (1 inch = 25.4 mm) and compares against the
/// small-page threshold. Pages smaller than 150mm x 200mm are rendered
/// at 400 DPI; all others at 300 DPI.
#[cfg(feature = "ocr")]
fn select_dpi(width_points: f64, height_points: f64) -> u32 {
    // Convert points to millimeters: points / 72 * 25.4
    let width_mm = width_points / 72.0 * 25.4;
    let height_mm = height_points / 72.0 * 25.4;

    if width_mm < SMALL_PAGE_WIDTH_MM && height_mm < SMALL_PAGE_HEIGHT_MM {
        HIGH_DPI
    } else {
        STANDARD_DPI
    }
}

/// Renders a single page to PNG bytes using pdfium, with adaptive DPI
/// selection based on page dimensions.
///
/// This is a helper for `ocr_pdf_pages_batch` that handles the render logic
/// for one page. The `document` and `page_number` (1-indexed) parameters
/// identify the page to render.
///
/// # Errors
///
/// Returns [`PdfError::Ocr`] if the page cannot be accessed or rendered.
#[cfg(feature = "ocr")]
fn render_page_to_png(
    document: &pdfium_render::prelude::PdfDocument<'_>,
    page_number: usize,
) -> Result<Vec<u8>, PdfError> {
    use pdfium_render::prelude::*;

    // Page numbers are 1-indexed; pdfium uses 0-indexed page access.
    let page_index = page_number.checked_sub(1).ok_or_else(|| {
        PdfError::Ocr(format!("invalid page number: {page_number} (must be >= 1)"))
    })?;

    // Pdfium's pages().get() accepts a u16. Verify the page index fits within
    // the u16 range to prevent silent truncation of page numbers above 65535.
    let page_index_u16: u16 = u16::try_from(page_index).map_err(|_| {
        PdfError::Ocr(format!(
            "page number {page_number} exceeds the maximum supported page index (65536)"
        ))
    })?;

    let page = document
        .pages()
        .get(page_index_u16)
        .map_err(|e| PdfError::Ocr(format!("failed to access page {page_number}: {e}")))?;

    // Determine page dimensions in points and select render DPI.
    let width_points = page.width().value as f64;
    let height_points = page.height().value as f64;
    let dpi = select_dpi(width_points, height_points);

    // Calculate pixel dimensions at the selected DPI.
    // pixels = points / 72 * dpi
    // Clamp to MAX_OCR_DIMENSION_PX to prevent i32 overflow and OOM for PDFs
    // with extreme dimensions (e.g., engineering drawings at 300 DPI).
    let width_px = (width_points / 72.0 * f64::from(dpi)).clamp(1.0, MAX_OCR_DIMENSION_PX) as i32;
    let height_px = (height_points / 72.0 * f64::from(dpi)).clamp(1.0, MAX_OCR_DIMENSION_PX) as i32;

    // Render the page to a bitmap.
    let bitmap = page
        .render_with_config(
            &PdfRenderConfig::new()
                .set_target_width(width_px)
                .set_target_height(height_px),
        )
        .map_err(|e| {
            PdfError::Ocr(format!(
                "failed to render page {page_number} at {dpi} DPI: {e}"
            ))
        })?;

    // Convert the bitmap to PNG bytes for Tesseract using fast compression.
    // CompressionType::Fast (level 1) reduces per-page encoding time from
    // ~2-3 seconds (default level 6) to ~100-200ms for a 300 DPI A4 page
    // without affecting pixel data: both levels produce lossless PNG and
    // Tesseract OCR output is identical regardless of compression level.
    let image = bitmap.as_image();
    let mut png_bytes = Vec::new();
    {
        use image::codecs::png::{CompressionType, FilterType, PngEncoder};
        image
            .write_with_encoder(PngEncoder::new_with_quality(
                std::io::Cursor::new(&mut png_bytes),
                CompressionType::Fast,
                FilterType::NoFilter,
            ))
            .map_err(|e| {
                PdfError::Ocr(format!("failed to encode page {page_number} as PNG: {e}"))
            })?;
    }

    Ok(png_bytes)
}

/// Performs OCR on multiple pages of a PDF file using a fully parallel
/// render+OCR pipeline.
///
/// Each page is processed independently on a dedicated rayon thread that:
///   1. Loads the pdfium library and opens the PDF
///   2. Renders its assigned page to a PNG image
///   3. Drops pdfium resources
///   4. Invokes Tesseract CLI on the PNG bytes
///
/// pdfium's `PdfDocument` is not `Send`/`Sync`, so each thread creates its
/// own independent pdfium instance. The pdfium shared library uses thread-local
/// storage for its global state (via `FPDF_InitLibraryWithConfig`), so multiple
/// threads can safely load and use pdfium concurrently as long as each thread
/// has its own `Pdfium` and `PdfDocument` instances. This parallelizes the
/// page rendering phase which was previously the dominant bottleneck (~2.5s
/// per page x 35 pages = ~87s sequential vs ~10s parallel with 8 threads).
///
/// A dedicated rayon thread pool guarantees parallel execution regardless of
/// the calling context (e.g., even when called from within another rayon pool
/// where all threads are occupied).
///
/// The returned `HashMap` maps each 1-indexed page number to its OCR text.
/// Pages that fail rendering or OCR are logged as warnings and omitted from
/// the result.
///
/// # Parameters
///
/// - `pdf_path`: Absolute path to the PDF file.
/// - `page_numbers`: Slice of 1-indexed page numbers to OCR.
/// - `language`: Tesseract language code (e.g., "eng", "deu", "eng+deu").
///
/// # Errors
///
/// Returns [`PdfError::Ocr`] if the language parameter fails validation.
/// Returns [`PdfError::DepDownload`] if Tesseract is not installed.
/// Returns [`PdfError::Ocr`] if the dedicated thread pool cannot be created.
/// Individual page render/OCR failures are logged and skipped (partial
/// results are returned).
#[cfg(feature = "ocr")]
pub fn ocr_pdf_pages_batch(
    pdf_path: &std::path::Path,
    page_numbers: &[usize],
    language: &str,
) -> Result<std::collections::HashMap<usize, String>, PdfError> {
    if page_numbers.is_empty() {
        return Ok(std::collections::HashMap::new());
    }

    // Validate the language parameter before spawning any threads or resolving
    // the Tesseract binary. This ensures invalid language codes are rejected
    // early with a clear error message.
    validate_tesseract_language(language)?;

    // Resolve the Tesseract binary path once for all pages.
    let tess_path = crate::deps::ensure_tesseract()?;
    let tessdata = resolve_tessdata_dir(&tess_path);

    // Fully parallel pipeline: each rayon thread independently loads pdfium,
    // opens the PDF, renders its assigned page, drops pdfium, and runs
    // Tesseract OCR. This parallelizes both the render and OCR phases.
    use rayon::prelude::*;

    // Use the process-wide OCR thread pool singleton to bound total thread count.
    let ocr_pool = &*OCR_THREAD_POOL;

    let pdf_path_ref = pdf_path;
    let tess_ref = &tess_path;
    let tessdata_ref = tessdata.as_deref();
    let page_list = page_numbers.to_vec();

    let results: std::collections::HashMap<usize, String> = ocr_pool.install(|| {
        page_list
            .into_par_iter()
            .filter_map(|page_num| {
                // Each thread loads its own pdfium instance. pdfium uses
                // thread-local state, so concurrent loads are safe.
                let png_bytes = {
                    use pdfium_render::prelude::*;
                    let bindings = match crate::pdfium_binding::bind_pdfium() {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::warn!(page = page_num, "pdfium bind failed: {e}");
                            return None;
                        }
                    };
                    let pdfium = Pdfium::new(bindings);
                    let document = match pdfium.load_pdf_from_file(pdf_path_ref, None) {
                        Ok(d) => d,
                        Err(e) => {
                            tracing::warn!(page = page_num, "PDF load failed: {e}");
                            return None;
                        }
                    };
                    match render_page_to_png(&document, page_num) {
                        Ok(bytes) => bytes,
                        Err(e) => {
                            tracing::warn!(
                                page = page_num,
                                pdf = %pdf_path_ref.display(),
                                error = %e,
                                "skipping OCR: page render failed"
                            );
                            return None;
                        }
                    }
                    // pdfium resources dropped here at block end.
                };

                match ocr_png_via_cli(tess_ref, tessdata_ref, &png_bytes, language) {
                    Ok(text) => Some((page_num, text)),
                    Err(e) => {
                        tracing::warn!(
                            page = page_num,
                            pdf = %pdf_path_ref.display(),
                            error = %e,
                            "skipping OCR: Tesseract failed"
                        );
                        None
                    }
                }
            })
            .collect()
    });

    tracing::debug!(
        pages = results.len(),
        pdf = %pdf_path.display(),
        "parallel render+OCR complete"
    );

    Ok(results)
}

/// Renders a specific page of a PDF to a raster image using pdfium and
/// performs OCR on the rendered image.
///
/// This is a convenience wrapper around `ocr_pdf_pages_batch` for single-page
/// OCR. For multiple pages, call `ocr_pdf_pages_batch` directly to avoid
/// repeated pdfium library binding, PDF loading, and Tesseract path resolution.
///
/// # Parameters
///
/// - `pdf_path`: Absolute path to the PDF file.
/// - `page_number`: 1-indexed page number to render and OCR.
/// - `language`: Tesseract language code (e.g., "eng", "deu", "eng+deu").
///
/// # Errors
///
/// Returns [`PdfError::Ocr`] if pdfium rendering, Tesseract processing, or
/// language validation fails.
/// Returns [`PdfError::Pdfium`] if the pdfium library cannot be loaded.
#[cfg(feature = "ocr")]
pub fn ocr_pdf_page(
    pdf_path: &std::path::Path,
    page_number: usize,
    language: &str,
) -> Result<String, PdfError> {
    let results = ocr_pdf_pages_batch(pdf_path, &[page_number], language)?;

    results.into_values().next().ok_or_else(|| {
        PdfError::Ocr(format!(
            "OCR produced no text for page {page_number} of {}",
            pdf_path.display()
        ))
    })
}

// ---------------------------------------------------------------------------
// hOCR support: word-level bounding boxes from Tesseract OCR output
// ---------------------------------------------------------------------------

/// A single word recognized by Tesseract hOCR with its pixel bounding box.
/// Coordinates are in the image pixel space at the render DPI, with the
/// origin at the top-left corner of the rendered page image.
#[cfg(feature = "ocr")]
#[derive(Debug, Clone)]
pub struct HocrWord {
    /// The recognized text content of this word.
    pub text: String,
    /// Left edge of the bounding box in pixels.
    pub x0: i32,
    /// Top edge of the bounding box in pixels.
    pub y0: i32,
    /// Right edge of the bounding box in pixels.
    pub x1: i32,
    /// Bottom edge of the bounding box in pixels.
    pub y1: i32,
}

/// Pre-resolved Tesseract binary and tessdata directory paths.
///
/// Calling `ensure_tesseract()` and `resolve_tessdata_dir()` involves
/// filesystem probing (executable search, tessdata directory detection).
/// When processing multiple PDF pages, resolving paths once and reusing
/// them avoids redundant I/O. The struct is `Send + Sync` (contains only
/// `PathBuf` and `Option<PathBuf>`), so it can be shared across scoped
/// threads that run parallel Tesseract subprocesses.
#[cfg(feature = "ocr")]
pub struct TesseractPaths {
    /// Absolute path to the Tesseract binary.
    pub binary: std::path::PathBuf,
    /// Absolute path to the tessdata directory (language pack files).
    /// None when Tesseract locates tessdata via its compiled-in default.
    pub tessdata: Option<std::path::PathBuf>,
}

#[cfg(feature = "ocr")]
impl TesseractPaths {
    /// Resolves the Tesseract binary and tessdata directory paths by probing
    /// the filesystem. Returns an error if Tesseract is not installed or
    /// cannot be downloaded.
    pub fn resolve() -> Result<Self, PdfError> {
        let binary = crate::deps::ensure_tesseract()?;
        let tessdata = resolve_tessdata_dir(&binary);
        Ok(Self { binary, tessdata })
    }

    /// Runs Tesseract hOCR on raw PNG image bytes using the pre-resolved
    /// paths and parses the result into word-level bounding boxes.
    ///
    /// The `png_bytes` parameter contains a rendered PDF page image. The
    /// `language` parameter specifies the Tesseract language code (e.g.,
    /// "eng", "deu", "eng+deu").
    ///
    /// Returns a vector of `HocrWord` structs with pixel-space bounding
    /// boxes. The caller must track the render DPI separately (from
    /// `render_page_to_png_with_dpi`) to convert pixel coordinates to
    /// PDF points.
    pub fn ocr_png_to_hocr_words(
        &self,
        png_bytes: &[u8],
        language: &str,
    ) -> Result<Vec<HocrWord>, PdfError> {
        validate_tesseract_language(language)?;
        let hocr_html =
            ocr_png_hocr_via_cli(&self.binary, self.tessdata.as_deref(), png_bytes, language)?;
        Ok(parse_hocr_words(&hocr_html))
    }
}

/// Renders a page from an already-loaded `PdfDocument` to PNG bytes and
/// reports the render DPI used.
///
/// This is the contention-free rendering path for contexts where a
/// `PdfDocument` is already loaded (e.g., the annotation pipeline).
/// It reuses the caller's pdfium instance instead of loading a new one,
/// avoiding DLL loader lock contention on Windows when multiple threads
/// hold pdfium bindings simultaneously.
///
/// The DPI is selected adaptively based on page dimensions: 400 DPI for
/// small pages (business cards, receipts), 300 DPI for standard pages.
///
/// # Parameters
///
/// - `document`: Reference to an already-loaded pdfium PDF document.
/// - `page_number`: 1-indexed page number to render.
///
/// # Returns
///
/// A tuple of `(png_bytes, render_dpi)` where `png_bytes` is the lossless
/// PNG image and `render_dpi` is the DPI used for rendering (needed for
/// pixel-to-PDF-point coordinate conversion).
#[cfg(feature = "ocr")]
pub fn render_page_to_png_with_dpi(
    document: &pdfium_render::prelude::PdfDocument<'_>,
    page_number: usize,
) -> Result<(Vec<u8>, u32), PdfError> {
    let page_index = page_number.checked_sub(1).ok_or_else(|| {
        PdfError::Ocr(format!("invalid page number: {page_number} (must be >= 1)"))
    })?;
    let page_index_u16: u16 = u16::try_from(page_index).map_err(|_| {
        PdfError::Ocr(format!(
            "page number {page_number} exceeds the maximum supported page index (65536)"
        ))
    })?;

    let page = document
        .pages()
        .get(page_index_u16)
        .map_err(|e| PdfError::Ocr(format!("failed to access page {page_number}: {e}")))?;

    let width_points = page.width().value as f64;
    let height_points = page.height().value as f64;
    let dpi = select_dpi(width_points, height_points);

    let png_bytes = render_page_to_png(document, page_number)?;
    Ok((png_bytes, dpi))
}

/// Renders a specific PDF page to a raster image and performs OCR with hOCR
/// output to obtain word-level bounding boxes.
///
/// hOCR (HTML OCR) is a Tesseract output format that embeds bounding box
/// coordinates for each recognized word in HTML span elements. This function
/// renders the page via pdfium, invokes Tesseract with hOCR output mode, and
/// parses the resulting HTML to extract word text and pixel-space coordinates.
///
/// The returned bounding boxes are in the rendered image's pixel coordinate
/// system. To convert to PDF points:
///   `pdf_x = pixel_x * 72.0 / render_dpi`
///   `pdf_y = page_height_pts - (pixel_y * 72.0 / render_dpi)`
/// The Y-axis inversion accounts for PDF's bottom-up coordinate system vs.
/// the image's top-down coordinate system.
///
/// # Parameters
///
/// - `pdf_path`: Absolute path to the PDF file.
/// - `page_number`: 1-indexed page number to render and OCR.
/// - `language`: Tesseract language code (e.g., "eng", "deu", "eng+deu").
///
/// # Returns
///
/// A tuple of `(Vec<HocrWord>, render_dpi)` where `render_dpi` is the DPI
/// used for rendering (needed for pixel-to-PDF-points conversion).
///
/// # Errors
///
/// Returns [`PdfError::Ocr`] if the language parameter fails validation.
/// Returns [`PdfError::Pdfium`] if the pdfium library cannot be loaded or
/// the PDF cannot be opened.
/// Returns [`PdfError::Ocr`] if rendering, Tesseract invocation, or hOCR
/// parsing fails.
#[cfg(feature = "ocr")]
pub fn ocr_page_with_hocr(
    pdf_path: &Path,
    page_number: usize,
    language: &str,
) -> Result<(Vec<HocrWord>, u32), PdfError> {
    use pdfium_render::prelude::*;

    // Validate language before performing any expensive rendering or subprocess work.
    validate_tesseract_language(language)?;

    let tess_path = crate::deps::ensure_tesseract()?;
    let tessdata = resolve_tessdata_dir(&tess_path);

    // Load pdfium and the PDF to render the page.
    let bindings = crate::pdfium_binding::bind_pdfium()?;
    let pdfium = Pdfium::new(bindings);
    let document = pdfium
        .load_pdf_from_file(pdf_path, None)
        .map_err(|e| PdfError::Pdfium(format!("failed to load PDF {}: {e}", pdf_path.display())))?;

    // Determine the render DPI from page dimensions before rendering.
    let page_index = page_number.checked_sub(1).ok_or_else(|| {
        PdfError::Ocr(format!("invalid page number: {page_number} (must be >= 1)"))
    })?;
    let page_index_u16: u16 = u16::try_from(page_index).map_err(|_| {
        PdfError::Ocr(format!(
            "page number {page_number} exceeds the maximum supported page index (65536)"
        ))
    })?;
    let page = document
        .pages()
        .get(page_index_u16)
        .map_err(|e| PdfError::Ocr(format!("failed to access page {page_number}: {e}")))?;

    let width_points = page.width().value as f64;
    let height_points = page.height().value as f64;
    let dpi = select_dpi(width_points, height_points);

    // Render the page to PNG bytes.
    let png_bytes = render_page_to_png(&document, page_number)?;

    // Drop pdfium resources before invoking Tesseract.
    drop(page);
    drop(document);

    // Invoke Tesseract with hOCR output mode.
    let hocr_html = ocr_png_hocr_via_cli(&tess_path, tessdata.as_deref(), &png_bytes, language)?;

    // Parse hOCR HTML to extract word bounding boxes.
    let words = parse_hocr_words(&hocr_html);

    Ok((words, dpi))
}

/// Performs hOCR on multiple pages of a PDF file using a fully parallel
/// render+OCR pipeline, producing word-level bounding boxes for each page.
///
/// Each page is processed independently on a dedicated rayon thread that:
///   1. Loads the pdfium library and opens the PDF
///   2. Renders its assigned page to a PNG image and records the render DPI
///   3. Drops pdfium resources
///   4. Invokes Tesseract CLI with hOCR output mode
///   5. Parses the hOCR HTML to extract word bounding boxes
///
/// This uses the same independent-pdfium-per-thread strategy as
/// [`ocr_pdf_pages_batch`]: pdfium's `PdfDocument` is not `Send`/`Sync`, but
/// each thread can safely load its own pdfium instance because pdfium uses
/// thread-local storage for its global state. This parallelizes the page
/// rendering phase which was the dominant bottleneck (~2.5s per page
/// sequential × 35 pages = ~87s vs ~10s parallel with 8 threads).
///
/// The optional `on_page_done` callback is invoked after each page completes
/// (both render and hOCR). It fires from the rayon worker thread that
/// processed the page, so it must be `Send + Sync`. The annotation pipeline
/// uses this for per-page progress reporting during the OCR phase.
///
/// # Parameters
///
/// - `pdf_path`: Absolute path to the PDF file.
/// - `page_numbers`: Slice of 1-indexed page numbers to OCR.
/// - `language`: Tesseract language code (e.g., "eng", "deu", "eng+deu").
/// - `on_page_done`: Optional callback invoked after each page completes OCR.
///
/// # Returns
///
/// A `HashMap` mapping each 1-indexed page number to a tuple of
/// `(Vec<HocrWord>, render_dpi)`. Pages that fail rendering or hOCR
/// are logged as warnings and omitted from the result.
///
/// # Errors
///
/// Returns [`PdfError::Ocr`] if the language parameter fails validation.
/// Returns [`PdfError::DepDownload`] if Tesseract is not installed.
/// Returns [`PdfError::Ocr`] if the dedicated thread pool cannot be created.
/// Individual page render/hOCR failures are logged and skipped (partial
/// results are returned).
#[cfg(feature = "ocr")]
pub fn ocr_pdf_pages_batch_hocr(
    pdf_path: &std::path::Path,
    page_numbers: &[usize],
    language: &str,
    on_page_done: Option<&(dyn Fn() + Send + Sync)>,
) -> Result<std::collections::HashMap<usize, (Vec<HocrWord>, u32)>, PdfError> {
    if page_numbers.is_empty() {
        return Ok(std::collections::HashMap::new());
    }

    // Validate the language parameter before spawning any threads or resolving
    // the Tesseract binary. This ensures invalid language codes are rejected
    // early with a clear error message.
    validate_tesseract_language(language)?;

    // Resolve the Tesseract binary path once for all pages.
    let tess_path = crate::deps::ensure_tesseract()?;
    let tessdata = resolve_tessdata_dir(&tess_path);

    // Fully parallel pipeline: each rayon thread independently loads pdfium,
    // opens the PDF, renders its assigned page (recording DPI), drops pdfium,
    // and runs Tesseract hOCR. This parallelizes both the render and OCR phases.
    use rayon::prelude::*;

    // Use the process-wide OCR thread pool singleton to bound total thread count.
    let ocr_pool = &*OCR_THREAD_POOL;

    let pdf_path_ref = pdf_path;
    let tess_ref = &tess_path;
    let tessdata_ref = tessdata.as_deref();
    let page_list = page_numbers.to_vec();

    let results: std::collections::HashMap<usize, (Vec<HocrWord>, u32)> = ocr_pool.install(|| {
        page_list
            .into_par_iter()
            .filter_map(|page_num| {
                // Each thread loads its own pdfium instance for rendering.
                // pdfium uses thread-local state, so concurrent loads are safe.
                let (png_bytes, dpi) = {
                    use pdfium_render::prelude::*;
                    let bindings = match crate::pdfium_binding::bind_pdfium() {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::warn!(page = page_num, "pdfium bind failed: {e}");
                            return None;
                        }
                    };
                    let pdfium = Pdfium::new(bindings);
                    let document = match pdfium.load_pdf_from_file(pdf_path_ref, None) {
                        Ok(d) => d,
                        Err(e) => {
                            tracing::warn!(page = page_num, "PDF load failed: {e}");
                            return None;
                        }
                    };

                    // Determine render DPI from page dimensions before rendering.
                    let page_index = match page_num.checked_sub(1) {
                        Some(i) => i,
                        None => {
                            tracing::warn!(page = page_num, "invalid page number (must be >= 1)");
                            return None;
                        }
                    };
                    let page_index_u16 = match u16::try_from(page_index) {
                        Ok(i) => i,
                        Err(_) => {
                            tracing::warn!(page = page_num, "page number exceeds u16 range");
                            return None;
                        }
                    };
                    let page = match document.pages().get(page_index_u16) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::warn!(page = page_num, error = %e, "page access failed");
                            return None;
                        }
                    };
                    let dpi = select_dpi(page.width().value as f64, page.height().value as f64);

                    match render_page_to_png(&document, page_num) {
                        Ok(bytes) => (bytes, dpi),
                        Err(e) => {
                            tracing::warn!(
                                page = page_num,
                                pdf = %pdf_path_ref.display(),
                                error = %e,
                                "skipping hOCR: page render failed"
                            );
                            return None;
                        }
                    }
                    // pdfium resources dropped here at block end.
                };

                let result =
                    match ocr_png_hocr_via_cli(tess_ref, tessdata_ref, &png_bytes, language) {
                        Ok(hocr_html) => {
                            let words = parse_hocr_words(&hocr_html);
                            Some((page_num, (words, dpi)))
                        }
                        Err(e) => {
                            tracing::warn!(
                                page = page_num,
                                pdf = %pdf_path_ref.display(),
                                error = %e,
                                "skipping hOCR: Tesseract failed"
                            );
                            None
                        }
                    };

                // Signal page completion to the caller for progress reporting.
                if let Some(cb) = on_page_done {
                    cb();
                }

                result
            })
            .collect()
    });

    tracing::debug!(
        pages = results.len(),
        pdf = %pdf_path.display(),
        "parallel render+hOCR complete"
    );

    Ok(results)
}

/// Invokes the Tesseract CLI binary on PNG image bytes with hOCR output mode.
/// Returns the raw hOCR HTML string containing word-level bounding boxes.
///
/// Uses `tesseract stdin stdout -c tessedit_create_hocr=1` to produce HTML
/// output with embedded bounding box coordinates for each recognized word.
///
/// The caller is responsible for validating the `language` parameter against
/// the allowlist via `validate_tesseract_language` before invoking this function.
#[cfg(feature = "ocr")]
fn ocr_png_hocr_via_cli(
    tesseract_path: &Path,
    tessdata_dir: Option<&Path>,
    png_bytes: &[u8],
    language: &str,
) -> Result<String, PdfError> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut cmd = Command::new(tesseract_path);
    cmd.arg("stdin")
        .arg("stdout")
        .arg("-l")
        .arg(language)
        .arg("-c")
        .arg("tessedit_create_hocr=1");

    if let Some(td) = tessdata_dir {
        cmd.arg("--tessdata-dir").arg(td);
    }

    // Restrict Tesseract to 1 OpenMP thread per process. Rayon manages
    // inter-page parallelism; allowing Tesseract's internal multi-threading
    // would oversubscribe the CPU when multiple pages run concurrently.
    // The standard OpenMP environment variable is OMP_NUM_THREADS (not
    // OMP_THREAD_NUM, which Tesseract ignores silently).
    cmd.env("OMP_NUM_THREADS", "1");
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| PdfError::Ocr(format!("failed to spawn Tesseract hOCR subprocess: {e}")))?;

    {
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| PdfError::Ocr("failed to open Tesseract stdin pipe".into()))?;
        let mut writer = std::io::BufWriter::new(stdin);
        writer
            .write_all(png_bytes)
            .map_err(|e| PdfError::Ocr(format!("failed to write PNG to Tesseract stdin: {e}")))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| PdfError::Ocr(format!("Tesseract hOCR wait failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PdfError::Ocr(format!(
            "Tesseract hOCR exited with status {}: {}",
            output.status,
            stderr.trim()
        )));
    }

    String::from_utf8(output.stdout)
        .map_err(|e| PdfError::Ocr(format!("Tesseract hOCR output is not valid UTF-8: {e}")))
}

/// Parses hOCR HTML output to extract word text and bounding boxes.
///
/// hOCR word elements have two common attribute quoting styles depending on
/// the Tesseract version and platform:
///   Tesseract 3.x/4.x: `<span class='ocrx_word' ... title='bbox 10 20 100 40; ...'>word</span>`
///   Tesseract 5.x:      `<span class="ocrx_word" ... title="bbox 10 20 100 40; ...">word</span>`
///
/// The regex accepts both single-quote and double-quote attribute syntax via
/// `['"]` character classes. This prevents empty word lists (and consequent
/// fallback to page-level bounding boxes) when the installed Tesseract version
/// uses a different quoting style than the one the regex was originally written
/// for.
///
/// Words with empty text (whitespace-only) are filtered out.
#[cfg(feature = "ocr")]
fn parse_hocr_words(hocr_html: &str) -> Vec<HocrWord> {
    // Reference the module-level compiled regex via LazyLock. The regex is
    // compiled once on first access and reused across all subsequent calls.
    let re = &*HOCR_WORD_RE;

    re.captures_iter(hocr_html)
        .filter_map(|cap| {
            let x0: i32 = cap[1].parse().ok()?;
            let y0: i32 = cap[2].parse().ok()?;
            let x1: i32 = cap[3].parse().ok()?;
            let y1: i32 = cap[4].parse().ok()?;
            let text = cap[5].trim().to_string();
            if text.is_empty() {
                return None;
            }
            Some(HocrWord {
                text,
                x0,
                y0,
                x1,
                y1,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "ocr"))]
mod tests {
    use super::*;

    /// T-PDF-061: OCR fallback triggers for image-only pages.
    /// This test verifies the `ocr_page` function by passing a valid blank PNG
    /// through Tesseract. Tesseract must be installed on the system or in the
    /// NeuronCite runtime cache for this test to pass. When Tesseract is
    /// missing, the test fails with a descriptive assertion message.
    #[test]
    fn t_pdf_008_ocr_fallback_triggers_for_image_only_pages() {
        // Generate a valid 1x1 white PNG image using the image crate.
        // Hand-crafted PNG byte arrays are fragile (CRC, zlib checksums),
        // so the image crate produces a structurally correct file.
        let img = image::ImageBuffer::from_pixel(1, 1, image::Rgb([255u8, 255, 255]));
        let mut png_buf = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut png_buf),
            image::ImageFormat::Png,
        )
        .expect("encoding a 1x1 PNG to memory should not fail");

        let result = ocr_page(&png_buf, "eng");

        // Tesseract may not be installed on CI runners. When the error indicates
        // a missing dependency (DepDownload), the test skips gracefully rather
        // than failing the CI pipeline.
        let text = match result {
            Ok(t) => t,
            Err(ref e) if format!("{e:?}").contains("DepDownload") => {
                eprintln!("T-PDF-008: skipping -- Tesseract not installed");
                return;
            }
            Err(e) => panic!("OCR failed with unexpected error: {e:?}"),
        };

        // A 1x1 white pixel should produce empty or whitespace-only text.
        assert!(
            text.trim().is_empty() || text.len() < 50,
            "OCR of a blank image should produce minimal text"
        );
    }

    /// Verifies that `select_dpi` returns 400 DPI for small pages and 300 DPI
    /// for standard-sized pages.
    #[test]
    fn select_dpi_adaptive_resolution() {
        // A5 page in points: ~420 x 595 points -> ~148mm x 210mm.
        // 148mm < 150mm and 210mm > 200mm, so the height exceeds the threshold.
        // This should use standard DPI.
        let a5_width_points = 420.0;
        let a5_height_points = 595.0;
        assert_eq!(select_dpi(a5_width_points, a5_height_points), STANDARD_DPI);

        // Very small page: 100mm x 100mm -> ~283 x 283 points.
        // Both dimensions are below the threshold, so high DPI applies.
        let small_width_points = 100.0 / 25.4 * 72.0; // ~283 points
        let small_height_points = 100.0 / 25.4 * 72.0;
        assert_eq!(
            select_dpi(small_width_points, small_height_points),
            HIGH_DPI
        );
    }

    /// T-PERF-003: Batch OCR processes an empty page list without error.
    /// Verifies that `ocr_pdf_pages_batch` returns an empty map when given
    /// an empty slice of page numbers, without loading pdfium or Tesseract.
    #[test]
    fn t_perf_003_batch_ocr_empty_pages() {
        let dummy_path = std::path::Path::new("/nonexistent.pdf");
        let result = ocr_pdf_pages_batch(dummy_path, &[], "eng");
        // An empty page list should return Ok with an empty map, without
        // attempting to load pdfium or resolve the Tesseract binary.
        assert!(result.is_ok(), "empty page list must succeed");
        assert!(result.unwrap().is_empty(), "result must be an empty map");
    }

    /// Verifies that `ocr_png_via_cli` correctly handles stdin piping for
    /// a larger image (100x100 white pixels). The image is large enough to
    /// exercise the BufWriter buffering path and validates that the full PNG
    /// byte stream is delivered to Tesseract via stdin without truncation.
    #[test]
    fn stdin_pipe_handles_larger_image() {
        let tess_path = match crate::deps::ensure_tesseract() {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Tesseract not installed, skipping stdin_pipe_handles_larger_image");
                return;
            }
        };
        let tessdata = resolve_tessdata_dir(&tess_path);

        // Generate a 100x100 white PNG (~300 bytes compressed).
        let img = image::ImageBuffer::from_pixel(100, 100, image::Rgb([255u8, 255, 255]));
        let mut png_buf = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut png_buf),
            image::ImageFormat::Png,
        )
        .expect("encoding a 100x100 PNG to memory should not fail");

        let result = ocr_png_via_cli(&tess_path, tessdata.as_deref(), &png_buf, "eng");
        let text = result.expect("OCR of a 100x100 white image via stdin should succeed");

        // A blank white image should produce empty or near-empty OCR output.
        assert!(
            text.trim().is_empty() || text.len() < 50,
            "OCR of a blank 100x100 image should produce minimal text, got: {text:?}"
        );
    }

    /// Verifies that `ocr_png_via_cli` returns an error when given invalid
    /// (non-PNG) bytes. Tesseract cannot decode garbage data and should exit
    /// with a non-zero status, which the stdin pipe path must handle gracefully.
    #[test]
    fn stdin_pipe_rejects_invalid_png() {
        let tess_path = match crate::deps::ensure_tesseract() {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Tesseract not installed, skipping stdin_pipe_rejects_invalid_png");
                return;
            }
        };
        let tessdata = resolve_tessdata_dir(&tess_path);

        // 16 bytes of garbage data that are not a valid image format.
        let garbage = [
            0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
            0x0A, 0x0B,
        ];

        let result = ocr_png_via_cli(&tess_path, tessdata.as_deref(), &garbage, "eng");
        assert!(
            result.is_err(),
            "OCR of invalid image data should return an error"
        );
    }

    /// Verifies that `ocr_page` works with a larger image containing actual
    /// rendered content (solid gray). Tests the full pipeline: resolve
    /// Tesseract path, resolve tessdata, invoke via stdin, parse output.
    #[test]
    fn ocr_page_full_pipeline_gray_image() {
        // Generate a 50x50 solid gray PNG.
        let img = image::ImageBuffer::from_pixel(50, 50, image::Rgb([128u8, 128, 128]));
        let mut png_buf = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut png_buf),
            image::ImageFormat::Png,
        )
        .expect("encoding a 50x50 PNG to memory should not fail");

        let result = ocr_page(&png_buf, "eng");

        // Tesseract may not be installed on CI runners. Skip gracefully when
        // the error indicates a missing dependency rather than failing the
        // entire test suite.
        let text = match result {
            Ok(t) => t,
            Err(ref e) if format!("{e:?}").contains("DepDownload") => {
                eprintln!("ocr_page_full_pipeline_gray_image: skipping -- Tesseract not installed");
                return;
            }
            Err(e) => panic!("OCR failed with unexpected error: {e:?}"),
        };

        // A solid gray image should produce empty or near-empty text.
        assert!(
            text.trim().is_empty() || text.len() < 100,
            "OCR of a solid gray image should produce minimal text, got: {text:?}"
        );
    }

    /// Verifies that `ocr_png_via_cli` handles empty input bytes gracefully.
    /// An empty byte slice is not a valid PNG; Tesseract should reject it.
    #[test]
    fn stdin_pipe_rejects_empty_bytes() {
        let tess_path = match crate::deps::ensure_tesseract() {
            Ok(p) => p,
            Err(_) => {
                eprintln!("Tesseract not installed, skipping stdin_pipe_rejects_empty_bytes");
                return;
            }
        };
        let tessdata = resolve_tessdata_dir(&tess_path);

        let result = ocr_png_via_cli(&tess_path, tessdata.as_deref(), &[], "eng");
        assert!(
            result.is_err(),
            "OCR of empty input bytes should return an error"
        );
    }

    /// T-PERF-004: Batch hOCR processes an empty page list without error.
    /// Mirrors T-PERF-012 for the hOCR variant. Returns an empty map without
    /// loading pdfium or resolving the Tesseract binary.
    #[test]
    fn t_perf_004_batch_hocr_empty_pages() {
        let dummy_path = std::path::Path::new("/nonexistent.pdf");
        let result = ocr_pdf_pages_batch_hocr(dummy_path, &[], "eng", None);
        assert!(result.is_ok(), "empty page list must succeed");
        assert!(result.unwrap().is_empty(), "result must be an empty map");
    }

    /// T-PERF-005: parse_hocr_words extracts word text and bounding box
    /// coordinates from well-formed hOCR HTML output.
    #[test]
    fn t_perf_005_parse_hocr_words_extraction() {
        let hocr = concat!(
            "<span class='ocrx_word' id='word_1_1' title='bbox 10 20 100 40; ",
            "x_wconf 95'>Hello</span> ",
            "<span class='ocrx_word' id='word_1_2' title='bbox 110 20 200 40; ",
            "x_wconf 93'>World</span>"
        );

        let words = parse_hocr_words(hocr);
        assert_eq!(words.len(), 2, "must extract 2 words from hOCR");

        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[0].x0, 10);
        assert_eq!(words[0].y0, 20);
        assert_eq!(words[0].x1, 100);
        assert_eq!(words[0].y1, 40);

        assert_eq!(words[1].text, "World");
        assert_eq!(words[1].x0, 110);
        assert_eq!(words[1].y0, 20);
        assert_eq!(words[1].x1, 200);
        assert_eq!(words[1].y1, 40);
    }

    /// T-PERF-006: parse_hocr_words filters out whitespace-only words that
    /// occasionally appear in Tesseract output for blank page regions.
    #[test]
    fn t_perf_006_parse_hocr_words_filters_whitespace() {
        let hocr = concat!(
            "<span class='ocrx_word' id='word_1_1' title='bbox 10 20 100 40'>",
            "   </span> ",
            "<span class='ocrx_word' id='word_1_2' title='bbox 110 20 200 40'>",
            "Visible</span>"
        );

        let words = parse_hocr_words(hocr);
        assert_eq!(words.len(), 1, "whitespace-only words must be filtered out");
        assert_eq!(words[0].text, "Visible");
    }

    /// T-PERF-007: parse_hocr_words returns an empty vector for HTML input
    /// without any ocrx_word spans (e.g., a blank page with no recognized text).
    #[test]
    fn t_perf_007_parse_hocr_words_empty_input() {
        let hocr = "<div class='ocr_page'></div>";
        let words = parse_hocr_words(hocr);
        assert!(words.is_empty(), "blank page hOCR must produce no words");
    }

    /// T-PERF-009: render_page_to_png uses fast PNG compression (CompressionType::Fast)
    /// which produces valid, decodable PNG bytes. Regression test for DEFECT-006:
    /// the rendering phase (Phase 1) of ocr_pdf_pages_batch_hocr previously used
    /// default PNG compression (level 6, ~2-3s per A4 page), contributing to the
    /// observed 251-second annotation hang on a 35-page scanned PDF.
    ///
    /// Verifies that the fast encoder produces lossless output with correct pixel
    /// values -- the same data Tesseract reads via its stdin pipe. A synthetic
    /// 50x50 RGB image is used so that no pdfium or PDF file is required.
    #[cfg(feature = "ocr")]
    #[test]
    fn t_perf_009_fast_png_encoding_produces_valid_decodable_png() {
        // Construct a synthetic DynamicImage with a solid color (blue channel
        // only) to verify that the fast PNG roundtrip preserves pixel values.
        let raw: Vec<u8> = std::iter::repeat_n([0u8, 0, 200], 50 * 50)
            .flatten()
            .collect();
        let img = image::DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(50, 50, raw).expect("raw buffer has correct dimensions"),
        );

        let mut buf = Vec::new();
        {
            use image::codecs::png::{CompressionType, FilterType, PngEncoder};
            img.write_with_encoder(PngEncoder::new_with_quality(
                std::io::Cursor::new(&mut buf),
                CompressionType::Fast,
                FilterType::NoFilter,
            ))
            .expect("fast PNG encoding must succeed on a synthetic image");
        }

        // PNG signature: first 8 bytes of every valid PNG file.
        assert_eq!(
            &buf[..8],
            &[137u8, 80, 78, 71, 13, 10, 26, 10],
            "encoded buffer must start with PNG signature bytes"
        );

        // Decoding the bytes back must produce the original dimensions and pixels.
        let decoded = image::load_from_memory_with_format(&buf, image::ImageFormat::Png)
            .expect("fast-encoded PNG must be decodable by the image crate");
        assert_eq!(decoded.width(), 50, "decoded width must be 50");
        assert_eq!(decoded.height(), 50, "decoded height must be 50");

        // Verify that a center pixel survived the lossless roundtrip.
        let pixel = decoded.to_rgb8().get_pixel(25, 25).0;
        assert_eq!(
            pixel,
            [0, 0, 200],
            "pixel values must be preserved through fast PNG compression"
        );
    }

    /// T-PERF-010: parse_hocr_words accepts double-quote attribute syntax produced
    /// by Tesseract 5.x builds. Earlier versions use single quotes; the regex must
    /// handle both to prevent empty word lists that cause fallback to page-level
    /// bounding boxes (DEFECT-007 root cause).
    #[test]
    fn t_perf_010_parse_hocr_words_double_quote_attributes() {
        let hocr = concat!(
            r#"<span class="ocrx_word" id="word_1_1" title="bbox 10 20 100 40; "#,
            r#"x_wconf 95">Hello</span> "#,
            r#"<span class="ocrx_word" id="word_1_2" title="bbox 110 20 200 40; "#,
            r#"x_wconf 93">World</span>"#
        );

        let words = parse_hocr_words(hocr);
        assert_eq!(
            words.len(),
            2,
            "must extract 2 words from double-quoted hOCR"
        );

        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[0].x0, 10);
        assert_eq!(words[0].y0, 20);
        assert_eq!(words[0].x1, 100);
        assert_eq!(words[0].y1, 40);

        assert_eq!(words[1].text, "World");
        assert_eq!(words[1].x0, 110);
    }

    /// T-PERF-011: parse_hocr_words handles mixed single/double quote attributes
    /// (defensive against non-standard or hand-crafted hOCR).
    #[test]
    fn t_perf_011_parse_hocr_words_mixed_quote_styles() {
        // Single-quoted class, single-quoted title.
        let hocr_single = "<span class='ocrx_word' id='w1' \
            title='bbox 5 10 50 30'>Single</span>";
        // Double-quoted class, double-quoted title.
        let hocr_double =
            r#"<span class="ocrx_word" id="w2" title="bbox 55 10 100 30">Double</span>"#;

        let words_single = parse_hocr_words(hocr_single);
        let words_double = parse_hocr_words(hocr_double);

        assert_eq!(words_single.len(), 1, "single-quoted hOCR must parse");
        assert_eq!(words_double.len(), 1, "double-quoted hOCR must parse");

        assert_eq!(words_single[0].text, "Single");
        assert_eq!(words_double[0].text, "Double");
    }

    /// T-PERF-008: parse_hocr_words handles multi-digit bounding box coordinates
    /// that occur with high-DPI rendering (400 DPI on A4 produces coordinates
    /// above 3000 pixels).
    #[test]
    fn t_perf_008_parse_hocr_words_large_coordinates() {
        let hocr = "<span class='ocrx_word' id='word_1_1' \
            title='bbox 2450 3100 3200 3150'>LargeCoord</span>";

        let words = parse_hocr_words(hocr);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].x0, 2450);
        assert_eq!(words[0].y0, 3100);
        assert_eq!(words[0].x1, 3200);
        assert_eq!(words[0].y1, 3150);
    }

    // -----------------------------------------------------------------------
    // Language validation tests
    // -----------------------------------------------------------------------

    /// T-SEC-001: validate_tesseract_language accepts a single valid language code.
    #[test]
    fn t_sec_001_valid_single_language() {
        assert!(validate_tesseract_language("eng").is_ok());
        assert!(validate_tesseract_language("deu").is_ok());
        assert!(validate_tesseract_language("chi_sim").is_ok());
        assert!(validate_tesseract_language("lat").is_ok());
    }

    /// T-SEC-002: validate_tesseract_language accepts "+" separated valid codes.
    #[test]
    fn t_sec_002_valid_multi_language() {
        assert!(validate_tesseract_language("eng+deu").is_ok());
        assert!(validate_tesseract_language("eng+deu+fra").is_ok());
        assert!(validate_tesseract_language("chi_sim+chi_tra").is_ok());
    }

    /// T-SEC-003: validate_tesseract_language rejects unknown language codes.
    #[test]
    fn t_sec_003_unknown_language_rejected() {
        let result = validate_tesseract_language("klingon");
        assert!(result.is_err(), "unknown language code must be rejected");
    }

    /// T-SEC-004: validate_tesseract_language rejects path separators.
    #[test]
    fn t_sec_004_path_separator_rejected() {
        assert!(validate_tesseract_language("../../etc/passwd").is_err());
        assert!(validate_tesseract_language("eng;rm -rf /").is_err());
        assert!(validate_tesseract_language("eng|cat /etc/shadow").is_err());
    }

    /// T-SEC-005: validate_tesseract_language rejects strings exceeding 100 chars.
    #[test]
    fn t_sec_005_excessive_length_rejected() {
        let long = "eng+".repeat(30);
        assert!(validate_tesseract_language(&long).is_err());
    }

    /// T-SEC-006: validate_tesseract_language rejects empty segments (leading/trailing "+").
    #[test]
    fn t_sec_006_empty_segment_rejected() {
        assert!(validate_tesseract_language("+eng").is_err());
        assert!(validate_tesseract_language("eng+").is_err());
        assert!(validate_tesseract_language("eng++deu").is_err());
    }

    /// T-SEC-007: validate_tesseract_language rejects shell metacharacters.
    #[test]
    fn t_sec_007_shell_metacharacters_rejected() {
        assert!(validate_tesseract_language("eng$(whoami)").is_err());
        assert!(validate_tesseract_language("eng`id`").is_err());
        assert!(validate_tesseract_language("eng&deu").is_err());
    }
}
