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

// Read-only inspection of PDF annotation properties.
//
// Scans all annotations in a PDF file and extracts their type, fill color,
// opacity, and bounding rectangle. This module is used both by the
// `neuroncite_inspect_annotations` MCP tool (for agent-driven verification)
// and by integration tests that verify highlight colors were physically
// written to disk by the annotation pipeline.
//
// The inspection is completely read-only: it opens the PDF, iterates pages
// and annotations, reads their properties via pdfium, and returns structured
// results. It does not modify the PDF in any way.

use std::path::Path;

use pdfium_render::prelude::*;
use serde::Serialize;

use crate::error::AnnotateError;

/// A single annotation extracted from a PDF page. Contains the annotation
/// type, color information, bounding rectangle, and QuadPoints status.
///
/// For highlight annotations, the visible color displayed by PDF viewers comes
/// from the `/C` entry in the annotation dictionary (exposed as stroke_color
/// by pdfium-render). The `/IC` entry (exposed as fill_color) is only used
/// by Circle/Square annotations for their interior. Both fields are reported
/// so agents can verify which entry was actually set.
///
/// The `quad_points_count` field is critical for highlight annotations: without
/// `/QuadPoints`, PDF viewers create the annotation object (visible in panels,
/// with working comments) but do NOT render any visible highlight color on the
/// page. A highlight annotation with quad_points_count == 0 is invisible.
#[derive(Debug, Clone, Serialize)]
pub struct AnnotationDetail {
    /// 1-indexed page number where this annotation appears.
    pub page_number: usize,
    /// PDF annotation type as a lowercase string (e.g. "highlight", "text",
    /// "link", "circle", "square", "underline", "strikeout", "squiggly",
    /// "ink", "stamp", "popup", "free_text", "widget", "redacted").
    pub annotation_type: String,
    /// Stroke color as a hex string "#RRGGBB" (the `/C` entry in the PDF
    /// annotation dictionary). For highlight annotations, this is the color
    /// that PDF viewers display. None if pdfium cannot read the `/C` entry.
    pub stroke_color: Option<String>,
    /// Fill color as a hex string "#RRGGBB" (the `/IC` entry in the PDF
    /// annotation dictionary). For Circle/Square annotations, this is the
    /// interior fill. For highlights, this is NOT the displayed color.
    /// None if pdfium cannot read the `/IC` entry.
    pub fill_color: Option<String>,
    /// Alpha channel value (0-255) of the primary color. For highlights,
    /// this is the alpha from stroke_color (`/C`). For other annotation
    /// types, this is the alpha from fill_color (`/IC`). None when the
    /// corresponding color field is None.
    pub alpha: Option<u8>,
    /// Bounding rectangle in PDF points: [left, bottom, right, top].
    /// None if pdfium fails to return bounds for this annotation.
    pub bounds: Option<[f32; 4]>,
    /// Number of QuadPoints entries (/QuadPoints array groups) on this
    /// annotation. For highlight annotations, this must be >= 1 for the
    /// highlight to be visually rendered by PDF viewers. A value of 0
    /// means the annotation exists but is invisible. For non-highlight
    /// annotation types, this field is 0 (QuadPoints are not applicable).
    pub quad_points_count: usize,
}

/// Aggregated inspection result for a PDF file. Contains per-annotation
/// details and summary statistics (highlight count, unique color list).
#[derive(Debug, Clone, Serialize)]
pub struct InspectionResult {
    /// Absolute path of the inspected PDF file.
    pub file_path: String,
    /// Total number of pages in the PDF document.
    pub total_pages: usize,
    /// All annotations found, ordered by page number then by position
    /// within the page's annotation list.
    pub annotations: Vec<AnnotationDetail>,
    /// Number of annotations with type "highlight".
    pub highlight_count: usize,
    /// Deduplicated list of hex color strings found across all highlight
    /// annotations, sorted alphabetically. Colors from non-highlight
    /// annotations are excluded from this list.
    pub unique_colors: Vec<String>,
}

/// Converts a `PdfPageAnnotationType` enum variant to a lowercase string
/// representation suitable for serialization and display.
fn annotation_type_to_string(ann_type: PdfPageAnnotationType) -> String {
    match ann_type {
        PdfPageAnnotationType::Highlight => "highlight",
        PdfPageAnnotationType::Text => "text",
        PdfPageAnnotationType::Link => "link",
        PdfPageAnnotationType::FreeText => "free_text",
        PdfPageAnnotationType::Circle => "circle",
        PdfPageAnnotationType::Square => "square",
        PdfPageAnnotationType::Ink => "ink",
        PdfPageAnnotationType::Stamp => "stamp",
        PdfPageAnnotationType::Popup => "popup",
        PdfPageAnnotationType::Underline => "underline",
        PdfPageAnnotationType::Strikeout => "strikeout",
        PdfPageAnnotationType::Squiggly => "squiggly",
        PdfPageAnnotationType::Redacted => "redacted",
        PdfPageAnnotationType::Widget => "widget",
        PdfPageAnnotationType::XfaWidget => "xfa_widget",
        _ => "unknown",
    }
    .to_string()
}

/// Converts a `PdfColor` to a hex string "#RRGGBB".
fn color_to_hex(color: &PdfColor) -> String {
    format!(
        "#{:02X}{:02X}{:02X}",
        color.red(),
        color.green(),
        color.blue()
    )
}

/// Inspects all annotations in a PDF file and returns structured details
/// about each one, including type, fill color, and bounding rectangle.
///
/// When `page_filter` is `Some(n)`, only annotations on page `n` (1-indexed)
/// are returned. When `None`, annotations from all pages are scanned.
///
/// This function uses lopdf (pure Rust) to read annotation properties directly
/// from the PDF binary structure. Unlike the pdfium-based path, lopdf handles
/// both pdfium-created inline annotation dicts and lopdf-promoted indirect
/// annotation references without invoking any C library code. This prevents
/// the Windows SEH crash caused by pdfium's FPDF_Annot_GetAttachmentPoints
/// C API when reading PDFs whose /Annots array entries have been reformatted
/// from inline dicts to indirect objects by lopdf's promote_inline_annotations().
pub fn inspect_pdf(
    pdf_path: &Path,
    page_filter: Option<usize>,
) -> Result<InspectionResult, AnnotateError> {
    inspect_pdf_lopdf(pdf_path, page_filter)
}

// ---------------------------------------------------------------------------
// lopdf-based annotation inspection (private implementation)
// ---------------------------------------------------------------------------

// Source of a single annotation entry inside a page's /Annots array.
// After lopdf's promote_inline_annotations(), entries are indirect References.
// pdfium-created PDFs (without lopdf post-processing) store entries as inline
// Dictionary objects. Collecting into this enum before processing allows all
// Document borrows to be dropped before the second pass resolves references.
pub(crate) enum AnnotSource {
    // Indirect reference (lopdf-promoted format).
    Ref(lopdf::ObjectId),
    // Inline dictionary (pdfium-created format). Cloned to release the
    // immutable borrow chain on the Document.
    Inline(lopdf::Dictionary),
}

// Converts a lopdf Object (Integer or Real) to f32. Returns None for all
// other object types (Boolean, Name, String, Array, Dictionary, Reference,
// Stream, Null).
pub(crate) fn lopdf_obj_to_f32(obj: &lopdf::Object) -> Option<f32> {
    match obj {
        lopdf::Object::Integer(i) => Some(*i as f32),
        lopdf::Object::Real(f) => Some(*f),
        _ => None,
    }
}

// Reads the /Subtype Name entry from an annotation dictionary and returns it
// as a lowercase ASCII string (e.g. "highlight", "text", "link"). Returns
// "unknown" when the entry is absent, not a Name object, or not valid UTF-8.
pub(crate) fn read_subtype_str(dict: &lopdf::Dictionary) -> String {
    dict.get(b"Subtype")
        .ok()
        .and_then(|o| o.as_name().ok())
        .and_then(|n| std::str::from_utf8(n).ok())
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| "unknown".to_string())
}

// Reads a PDF color array entry (/C or /IC) from a dictionary. Returns the
// color as a hex string "#RRGGBB" when the entry is a 3-element RGB array
// with values in the range 0.0..1.0 (PDF normalized color space). Returns
// None when the entry is absent, not an array, or not a 3-element array
// (0-element = transparent, 1-element = gray, 4-element = CMYK).
pub(crate) fn read_color_entry(dict: &lopdf::Dictionary, key: &[u8]) -> Option<String> {
    let arr = dict.get(key).ok()?.as_array().ok()?;
    if arr.len() != 3 {
        return None;
    }
    let r = (lopdf_obj_to_f32(&arr[0])? * 255.0).round() as u8;
    let g = (lopdf_obj_to_f32(&arr[1])? * 255.0).round() as u8;
    let b = (lopdf_obj_to_f32(&arr[2])? * 255.0).round() as u8;
    Some(format!("#{r:02X}{g:02X}{b:02X}"))
}

// Reads the /CA constant alpha entry from an annotation dictionary. Returns
// the alpha as a byte value 0..=255 (scaled from the 0.0..1.0 PDF range).
// Returns None when the entry is absent or not a numeric object.
fn read_ca_alpha(dict: &lopdf::Dictionary) -> Option<u8> {
    let v = lopdf_obj_to_f32(dict.get(b"CA").ok()?)?;
    Some((v.clamp(0.0, 1.0) * 255.0).round() as u8)
}

// Reads the /Rect bounding box entry from an annotation dictionary. Returns
// [left, bottom, right, top] as an f32 array. Returns None when the entry
// is absent, not an array, or contains fewer than 4 numeric elements.
pub(crate) fn read_rect_entry(dict: &lopdf::Dictionary) -> Option<[f32; 4]> {
    let arr = dict.get(b"Rect").ok()?.as_array().ok()?;
    if arr.len() < 4 {
        return None;
    }
    Some([
        lopdf_obj_to_f32(&arr[0])?,
        lopdf_obj_to_f32(&arr[1])?,
        lopdf_obj_to_f32(&arr[2])?,
        lopdf_obj_to_f32(&arr[3])?,
    ])
}

// Reads the /QuadPoints array entry from a dictionary and returns the count
// of point groups (each group is 8 numbers = 4 xy coordinate pairs defining
// a quadrilateral). Returns 0 when the entry is absent or the array length
// is not a positive multiple of 8.
pub(crate) fn read_quad_points_count(dict: &lopdf::Dictionary) -> usize {
    dict.get(b"QuadPoints")
        .ok()
        .and_then(|o| o.as_array().ok())
        .map(|arr| arr.len() / 8)
        .unwrap_or(0)
}

// Builds an AnnotationDetail from a lopdf annotation dictionary and the
// 1-indexed page number. Reads annotation type (/Subtype), stroke color
// (/C), fill color (/IC), constant alpha (/CA), bounding rectangle (/Rect),
// and QuadPoints count (/QuadPoints).
pub(crate) fn build_annotation_detail(
    dict: &lopdf::Dictionary,
    page_number: usize,
) -> AnnotationDetail {
    let annotation_type = read_subtype_str(dict);
    let is_highlight = annotation_type == "highlight";
    let stroke_color = read_color_entry(dict, b"C");
    let fill_color = read_color_entry(dict, b"IC");
    let alpha = read_ca_alpha(dict);
    let bounds = read_rect_entry(dict);
    let quad_points_count = if is_highlight {
        read_quad_points_count(dict)
    } else {
        0
    };
    AnnotationDetail {
        page_number,
        annotation_type,
        stroke_color,
        fill_color,
        alpha,
        bounds,
        quad_points_count,
    }
}

// Collects all annotation details from a single page. Returns an empty Vec
// when the page has no /Annots entry or it cannot be resolved.
//
// Uses a two-pass approach to avoid simultaneous immutable borrows on the
// Document:
//   Pass 1 (scoped block): iterates the /Annots array and collects
//     AnnotSource items (cloned inline dicts or ObjectId refs). All Document
//     borrows from the page/annots chain are dropped when the block ends.
//   Pass 2: resolves each ObjectId with a fresh Document borrow and calls
//     build_annotation_detail for each annotation.
pub(crate) fn read_page_annotations(
    doc: &lopdf::Document,
    page_id: lopdf::ObjectId,
    page_number: usize,
) -> Vec<AnnotationDetail> {
    // Pass 1: collect annotation sources. All doc borrows are dropped at the
    // end of this block.
    let sources: Vec<AnnotSource> = {
        let page_dict = match doc.get_object(page_id).ok().and_then(|o| o.as_dict().ok()) {
            Some(d) => d,
            None => return Vec::new(),
        };
        let annots_obj = match page_dict.get(b"Annots").ok() {
            Some(o) => o,
            None => return Vec::new(),
        };
        // /Annots can be a direct array or an indirect reference to an array.
        let annots_arr: Vec<lopdf::Object> = match annots_obj {
            lopdf::Object::Array(arr) => arr.clone(),
            lopdf::Object::Reference(r) => {
                match doc.get_object(*r).ok().and_then(|o| o.as_array().ok()) {
                    Some(arr) => arr.clone(),
                    None => return Vec::new(),
                }
            }
            _ => return Vec::new(),
        };
        // Classify each element as an indirect reference or an inline dict.
        let mut sources = Vec::with_capacity(annots_arr.len());
        for item in &annots_arr {
            match item {
                lopdf::Object::Reference(id) => {
                    // Indirect reference (lopdf-promoted annotation format).
                    sources.push(AnnotSource::Ref(*id));
                }
                lopdf::Object::Dictionary(d) => {
                    // Inline dictionary (pdfium-created annotation format).
                    // Clone to release the borrow on annots_arr.
                    sources.push(AnnotSource::Inline(d.clone()));
                }
                _ => {}
            }
        }
        sources
    }; // All doc borrows from the page/annots chain are dropped here.

    // Pass 2: resolve indirect references and build AnnotationDetail structs.
    let mut details = Vec::with_capacity(sources.len());
    for source in &sources {
        match source {
            AnnotSource::Ref(id) => {
                if let Some(dict) = doc.get_object(*id).ok().and_then(|o| o.as_dict().ok()) {
                    details.push(build_annotation_detail(dict, page_number));
                }
            }
            AnnotSource::Inline(dict) => {
                details.push(build_annotation_detail(dict, page_number));
            }
        }
    }
    details
}

// lopdf-based PDF annotation inspector. Reads annotation properties directly
// from the PDF binary structure without invoking the pdfium C library.
//
// Handles both annotation formats produced by the annotation pipeline:
//   - Inline dicts (pdfium-created, /Annots [<< /Subtype /Highlight ... >>])
//   - Indirect references (lopdf-promoted, /Annots [5 0 R 6 0 R])
//
// pdfium's FPDF_Annot_GetAttachmentPoints C API (called via
// annotation.attachment_points().len()) crashes on lopdf-promoted PDFs
// because it does not handle the indirect reference annotation format.
// This implementation avoids that call entirely by reading /QuadPoints
// directly from the lopdf dictionary.
fn inspect_pdf_lopdf(
    pdf_path: &Path,
    page_filter: Option<usize>,
) -> Result<InspectionResult, AnnotateError> {
    let doc = lopdf::Document::load(pdf_path).map_err(|e| AnnotateError::PdfLoad {
        path: pdf_path.to_string_lossy().to_string(),
        reason: format!("{e}"),
    })?;

    // get_pages() returns BTreeMap<u32, ObjectId> with 1-indexed page numbers.
    let pages = doc.get_pages();
    let total_pages = pages.len();

    let page_range: Vec<(usize, lopdf::ObjectId)> = match page_filter {
        Some(n) => {
            if n == 0 || n > total_pages {
                return Err(AnnotateError::Verification(format!(
                    "page_number {n} is out of range (1..{total_pages})"
                )));
            }
            match pages.get(&(n as u32)) {
                Some(&id) => vec![(n, id)],
                None => {
                    return Err(AnnotateError::Verification(format!(
                        "page_number {n} not found in document structure"
                    )));
                }
            }
        }
        None => pages.iter().map(|(&pn, &id)| (pn as usize, id)).collect(),
    };

    let mut annotations: Vec<AnnotationDetail> = Vec::new();

    for (page_number, page_id) in &page_range {
        let page_annots = read_page_annotations(&doc, *page_id, *page_number);
        annotations.extend(page_annots);
    }

    let highlight_count = annotations
        .iter()
        .filter(|a| a.annotation_type == "highlight")
        .count();

    let mut color_set: Vec<String> = annotations
        .iter()
        .filter(|a| a.annotation_type == "highlight")
        .filter_map(|a| a.stroke_color.clone().or_else(|| a.fill_color.clone()))
        .collect();
    color_set.sort();
    color_set.dedup();

    Ok(InspectionResult {
        file_path: pdf_path.to_string_lossy().to_string(),
        total_pages,
        annotations,
        highlight_count,
        unique_colors: color_set,
    })
}

/// Inspects annotations using an already-loaded pdfium instance. This avoids
/// redundant library loading when calling inspect_pdf multiple times (e.g.
/// in a loop or from tests that share a single pdfium instance).
pub fn inspect_pdf_with_pdfium(
    pdfium: &Pdfium,
    pdf_path: &Path,
    page_filter: Option<usize>,
) -> Result<InspectionResult, AnnotateError> {
    let document =
        pdfium
            .load_pdf_from_file(pdf_path, None)
            .map_err(|e| AnnotateError::PdfLoad {
                path: pdf_path.to_string_lossy().to_string(),
                reason: format!("{e}"),
            })?;

    let total_pages = document.pages().len() as usize;
    let mut annotations: Vec<AnnotationDetail> = Vec::new();

    // Determine which pages to scan based on the optional filter.
    let page_range: Vec<usize> = match page_filter {
        Some(page_num) => {
            if page_num == 0 || page_num > total_pages {
                return Err(AnnotateError::Verification(format!(
                    "page_number {page_num} is out of range (1..{total_pages})"
                )));
            }
            vec![page_num]
        }
        None => (1..=total_pages).collect(),
    };

    for page_num in page_range {
        let page_idx = (page_num - 1) as u16;
        let page = match document.pages().get(page_idx) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(page = page_num, "failed to access page: {e}");
                continue;
            }
        };

        for annotation in page.annotations().iter() {
            let ann_type = annotation.annotation_type();
            let type_str = annotation_type_to_string(ann_type);
            let is_highlight = ann_type == PdfPageAnnotationType::Highlight;

            // Read stroke color (the `/C` entry). For highlight annotations,
            // this is the color PDF viewers display.
            let (stroke_hex, stroke_alpha) = match annotation.stroke_color() {
                Ok(color) => (Some(color_to_hex(&color)), Some(color.alpha())),
                Err(_) => (None, None),
            };

            // Read fill color (the `/IC` entry). For Circle/Square annotations,
            // this is the interior fill. For highlights, this is secondary.
            let (fill_hex, fill_alpha) = match annotation.fill_color() {
                Ok(color) => (Some(color_to_hex(&color)), Some(color.alpha())),
                Err(_) => (None, None),
            };

            // For highlights, the primary alpha comes from stroke_color (`/C`).
            // For other annotation types, it comes from fill_color (`/IC`).
            let alpha = if is_highlight {
                stroke_alpha
            } else {
                fill_alpha
            };

            // Read bounding rectangle.
            let bounds = annotation.bounds().ok().map(|rect| {
                [
                    rect.left().value,
                    rect.bottom().value,
                    rect.right().value,
                    rect.top().value,
                ]
            });

            // Read QuadPoints count. For highlight annotations, the /QuadPoints
            // entry defines the visible highlighted region. Without it, the
            // annotation exists but is invisible in PDF viewers. The
            // attachment_points() collection maps directly to the /QuadPoints
            // array in the PDF annotation dictionary.
            let quad_points_count = if is_highlight {
                annotation.attachment_points().len()
            } else {
                0
            };

            annotations.push(AnnotationDetail {
                page_number: page_num,
                annotation_type: type_str,
                stroke_color: stroke_hex,
                fill_color: fill_hex,
                alpha,
                bounds,
                quad_points_count,
            });
        }
    }

    // Compute summary statistics from the collected annotations.
    let highlight_count = annotations
        .iter()
        .filter(|a| a.annotation_type == "highlight")
        .count();

    // Collect unique colors from highlight annotations. Uses stroke_color
    // (the `/C` entry) because that is the color PDF viewers display for
    // highlights. Falls back to fill_color (`/IC`) when stroke_color is absent,
    // covering PDFs created by older tools that only set `/IC`.
    let mut color_set: Vec<String> = annotations
        .iter()
        .filter(|a| a.annotation_type == "highlight")
        .filter_map(|a| a.stroke_color.clone().or_else(|| a.fill_color.clone()))
        .collect();
    color_set.sort();
    color_set.dedup();

    Ok(InspectionResult {
        file_path: pdf_path.to_string_lossy().to_string(),
        total_pages,
        annotations,
        highlight_count,
        unique_colors: color_set,
    })
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// T-INSPECT-001: annotation_type_to_string returns "highlight" for the
    /// Highlight variant.
    #[test]
    fn t_inspect_001_type_to_string_highlight() {
        assert_eq!(
            annotation_type_to_string(PdfPageAnnotationType::Highlight),
            "highlight"
        );
    }

    /// T-INSPECT-002: annotation_type_to_string returns "text" for the
    /// Text variant.
    #[test]
    fn t_inspect_002_type_to_string_text() {
        assert_eq!(
            annotation_type_to_string(PdfPageAnnotationType::Text),
            "text"
        );
    }

    /// T-INSPECT-003: annotation_type_to_string returns "unknown" for
    /// annotation types without an explicit mapping.
    #[test]
    fn t_inspect_003_type_to_string_unknown() {
        assert_eq!(
            annotation_type_to_string(PdfPageAnnotationType::Unknown),
            "unknown"
        );
    }

    /// T-INSPECT-004: color_to_hex produces uppercase hex with leading '#'.
    #[test]
    fn t_inspect_004_color_to_hex_format() {
        let color = PdfColor::new(255, 128, 0, 200);
        assert_eq!(color_to_hex(&color), "#FF8000");
    }

    /// T-INSPECT-005: color_to_hex handles pure black (all zero RGB).
    #[test]
    fn t_inspect_005_color_to_hex_black() {
        let color = PdfColor::new(0, 0, 0, 255);
        assert_eq!(color_to_hex(&color), "#000000");
    }

    /// T-INSPECT-006: color_to_hex handles pure white (all 255 RGB).
    #[test]
    fn t_inspect_006_color_to_hex_white() {
        let color = PdfColor::new(255, 255, 255, 255);
        assert_eq!(color_to_hex(&color), "#FFFFFF");
    }

    /// T-INSPECT-007: inspect_pdf returns an error for a nonexistent file path.
    #[test]
    fn t_inspect_007_nonexistent_file() {
        let result = inspect_pdf(Path::new("/nonexistent/file.pdf"), None);
        assert!(result.is_err(), "must fail for nonexistent PDF");
    }

    /// T-INSPECT-008: inspect_pdf with page_filter=0 returns an out-of-range error.
    /// Page numbers are 1-indexed, so 0 is invalid.
    #[test]
    fn t_inspect_008_page_zero_is_invalid() {
        // This test requires a valid PDF to reach the page validation logic.
        // We create a minimal PDF using pdfium.
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return, // Skip if pdfium is not available.
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create empty PDF");
        document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create blank page");

        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document.save_to_file(tmp.path()).expect("save PDF");

        let result = inspect_pdf_with_pdfium(&pdfium, tmp.path(), Some(0));
        assert!(result.is_err(), "page 0 must be rejected");
    }

    /// T-INSPECT-009: inspect_pdf on a blank PDF (one page, no annotations)
    /// returns an InspectionResult with zero annotations.
    #[test]
    fn t_inspect_009_empty_pdf_no_annotations() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);

        let mut document = pdfium.create_new_pdf().expect("create empty PDF");
        document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create blank page");

        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document.save_to_file(tmp.path()).expect("save PDF");

        let result =
            inspect_pdf_with_pdfium(&pdfium, tmp.path(), None).expect("inspect must succeed");
        assert_eq!(result.total_pages, 1);
        assert_eq!(result.annotations.len(), 0);
        assert_eq!(result.highlight_count, 0);
        assert!(result.unique_colors.is_empty());
    }

    /// T-INSPECT-010: Round-trip test that creates a highlight annotation with
    /// a specific color (setting /Rect, /QuadPoints, /C, and /IC entries as the
    /// production annotate.rs does), saves the PDF, then inspects it to verify
    /// all entries are physically present in the saved file.
    #[test]
    fn t_inspect_010_roundtrip_highlight_color() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        // Add a blank A4 page.
        let mut page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create page");

        // Create a highlight annotation with red color, mirroring the production
        // code in annotate.rs: set /Rect, /QuadPoints, /C, and /IC.
        let annotations = page.annotations_mut();
        if let Ok(mut highlight) = annotations.create_highlight_annotation() {
            let rect = PdfRect::new(
                PdfPoints::new(100.0),
                PdfPoints::new(100.0),
                PdfPoints::new(120.0),
                PdfPoints::new(300.0),
            );
            let _ = highlight.set_bounds(rect);
            let quad_points = PdfQuadPoints::from_rect(&rect);
            let _ = highlight
                .attachment_points_mut()
                .create_attachment_point_at_end(quad_points);
            let pdf_color = PdfColor::new(255, 0, 0, 128);
            let _ = highlight.set_stroke_color(pdf_color);
            let _ = highlight.set_fill_color(pdf_color);
        }

        // Save to a temporary file.
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document.save_to_file(tmp.path()).expect("save PDF");

        // Inspect the saved PDF.
        let result =
            inspect_pdf_with_pdfium(&pdfium, tmp.path(), None).expect("inspect must succeed");

        assert_eq!(result.highlight_count, 1, "must find one highlight");
        assert!(
            result.unique_colors.contains(&"#FF0000".to_string()),
            "must contain red color, found: {:?}",
            result.unique_colors
        );

        // Verify the single highlight annotation's properties.
        let highlight = result
            .annotations
            .iter()
            .find(|a| a.annotation_type == "highlight")
            .expect("must find a highlight annotation");
        assert_eq!(highlight.page_number, 1);
        assert_eq!(
            highlight.stroke_color.as_deref(),
            Some("#FF0000"),
            "stroke_color (/C entry) must be red"
        );
        assert_eq!(
            highlight.fill_color.as_deref(),
            Some("#FF0000"),
            "fill_color (/IC entry) must be red"
        );
        assert_eq!(highlight.alpha, Some(128));
        assert!(highlight.bounds.is_some());
        assert_eq!(
            highlight.quad_points_count, 1,
            "highlight must have exactly one QuadPoints entry for visible rendering"
        );
    }

    /// T-INSPECT-011: Round-trip with multiple colors on the same page.
    /// Creates two highlight annotations (red and green) with /Rect, /QuadPoints,
    /// /C, and /IC entries, saves, then verifies both colors and QuadPoints
    /// appear in the inspection result.
    #[test]
    fn t_inspect_011_multiple_colors_same_page() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        let mut page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create page");

        // Highlight 1: red (/Rect, /QuadPoints, /C, /IC set).
        {
            let annotations = page.annotations_mut();
            if let Ok(mut h) = annotations.create_highlight_annotation() {
                let rect = PdfRect::new(
                    PdfPoints::new(100.0),
                    PdfPoints::new(50.0),
                    PdfPoints::new(112.0),
                    PdfPoints::new(200.0),
                );
                let _ = h.set_bounds(rect);
                let _ = h
                    .attachment_points_mut()
                    .create_attachment_point_at_end(PdfQuadPoints::from_rect(&rect));
                let pdf_color = PdfColor::new(255, 0, 0, 128);
                let _ = h.set_stroke_color(pdf_color);
                let _ = h.set_fill_color(pdf_color);
            }
        }

        // Highlight 2: green (/Rect, /QuadPoints, /C, /IC set).
        {
            let annotations = page.annotations_mut();
            if let Ok(mut h) = annotations.create_highlight_annotation() {
                let rect = PdfRect::new(
                    PdfPoints::new(80.0),
                    PdfPoints::new(50.0),
                    PdfPoints::new(92.0),
                    PdfPoints::new(200.0),
                );
                let _ = h.set_bounds(rect);
                let _ = h
                    .attachment_points_mut()
                    .create_attachment_point_at_end(PdfQuadPoints::from_rect(&rect));
                let pdf_color = PdfColor::new(0, 255, 0, 128);
                let _ = h.set_stroke_color(pdf_color);
                let _ = h.set_fill_color(pdf_color);
            }
        }

        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document.save_to_file(tmp.path()).expect("save PDF");

        let result =
            inspect_pdf_with_pdfium(&pdfium, tmp.path(), None).expect("inspect must succeed");

        assert_eq!(result.highlight_count, 2);
        assert!(result.unique_colors.contains(&"#FF0000".to_string()));
        assert!(result.unique_colors.contains(&"#00FF00".to_string()));

        // Verify both highlights have QuadPoints for visible rendering.
        for ann in result
            .annotations
            .iter()
            .filter(|a| a.annotation_type == "highlight")
        {
            assert_eq!(
                ann.quad_points_count, 1,
                "each highlight must have QuadPoints"
            );
        }
    }

    /// T-INSPECT-012: Page filter restricts results to the specified page.
    /// Creates highlights on pages 1 and 2 (with /Rect, /QuadPoints, /C, /IC),
    /// then verifies that filtering to page 2 only returns the annotation
    /// from page 2.
    #[test]
    fn t_inspect_012_page_filter() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        // Page 1 with a red highlight.
        {
            let mut page1 = document
                .pages_mut()
                .create_page_at_start(PdfPagePaperSize::a4())
                .expect("create page 1");
            let annotations = page1.annotations_mut();
            if let Ok(mut h) = annotations.create_highlight_annotation() {
                let rect = PdfRect::new(
                    PdfPoints::new(100.0),
                    PdfPoints::new(50.0),
                    PdfPoints::new(112.0),
                    PdfPoints::new(200.0),
                );
                let _ = h.set_bounds(rect);
                let _ = h
                    .attachment_points_mut()
                    .create_attachment_point_at_end(PdfQuadPoints::from_rect(&rect));
                let pdf_color = PdfColor::new(255, 0, 0, 128);
                let _ = h.set_stroke_color(pdf_color);
                let _ = h.set_fill_color(pdf_color);
            }
        }

        // Page 2 with a blue highlight.
        {
            let mut page2 = document
                .pages_mut()
                .create_page_at_end(PdfPagePaperSize::a4())
                .expect("create page 2");
            let annotations = page2.annotations_mut();
            if let Ok(mut h) = annotations.create_highlight_annotation() {
                let rect = PdfRect::new(
                    PdfPoints::new(100.0),
                    PdfPoints::new(50.0),
                    PdfPoints::new(112.0),
                    PdfPoints::new(200.0),
                );
                let _ = h.set_bounds(rect);
                let _ = h
                    .attachment_points_mut()
                    .create_attachment_point_at_end(PdfQuadPoints::from_rect(&rect));
                let pdf_color = PdfColor::new(0, 0, 255, 128);
                let _ = h.set_stroke_color(pdf_color);
                let _ = h.set_fill_color(pdf_color);
            }
        }

        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document.save_to_file(tmp.path()).expect("save PDF");

        // Inspect only page 2.
        let result = inspect_pdf_with_pdfium(&pdfium, tmp.path(), Some(2))
            .expect("inspect page 2 must succeed");

        assert_eq!(result.total_pages, 2, "total pages is the full document");
        assert_eq!(result.highlight_count, 1, "only page 2 annotations");
        assert!(result.unique_colors.contains(&"#0000FF".to_string()));
        assert!(
            !result.unique_colors.contains(&"#FF0000".to_string()),
            "page 1 red color must not appear when filtering to page 2"
        );
    }

    /// T-INSPECT-013: Page filter with a page number exceeding the document's
    /// page count returns an error.
    #[test]
    fn t_inspect_013_page_filter_out_of_range() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");
        document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create page");

        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document.save_to_file(tmp.path()).expect("save PDF");

        let result = inspect_pdf_with_pdfium(&pdfium, tmp.path(), Some(99));
        assert!(result.is_err(), "page 99 of a 1-page PDF must fail");
    }

    /// T-INSPECT-014: unique_colors is sorted alphabetically and deduplicated.
    /// Creates three highlights (with /Rect, /QuadPoints, /C, /IC) using only
    /// two distinct colors to verify deduplication.
    #[test]
    fn t_inspect_014_unique_colors_dedup_and_sorted() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        let mut page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create page");

        // Three highlights: two green, one red.
        let colors = [
            PdfColor::new(0, 255, 0, 128),
            PdfColor::new(255, 0, 0, 128),
            PdfColor::new(0, 255, 0, 128), // duplicate green
        ];

        for (i, color) in colors.iter().enumerate() {
            let annotations = page.annotations_mut();
            if let Ok(mut h) = annotations.create_highlight_annotation() {
                let y_offset = 100.0 + (i as f32 * 20.0);
                let rect = PdfRect::new(
                    PdfPoints::new(y_offset),
                    PdfPoints::new(50.0),
                    PdfPoints::new(y_offset + 12.0),
                    PdfPoints::new(200.0),
                );
                let _ = h.set_bounds(rect);
                let _ = h
                    .attachment_points_mut()
                    .create_attachment_point_at_end(PdfQuadPoints::from_rect(&rect));
                let _ = h.set_stroke_color(*color);
                let _ = h.set_fill_color(*color);
            }
        }

        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document.save_to_file(tmp.path()).expect("save PDF");

        let result =
            inspect_pdf_with_pdfium(&pdfium, tmp.path(), None).expect("inspect must succeed");

        assert_eq!(result.highlight_count, 3);
        assert_eq!(
            result.unique_colors.len(),
            2,
            "three highlights but only two distinct colors"
        );
        // Sorted alphabetically: "#00FF00" < "#FF0000".
        assert_eq!(result.unique_colors[0], "#00FF00");
        assert_eq!(result.unique_colors[1], "#FF0000");
    }

    /// T-INSPECT-015: Text annotations (popup notes) are reported with type
    /// "text" and do not contribute to highlight_count or unique_colors.
    /// Text annotations have quad_points_count == 0 because QuadPoints are
    /// only applicable to highlight annotations.
    #[test]
    fn t_inspect_015_text_annotation_excluded_from_highlights() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        let mut page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create page");

        // Create a text annotation (popup note).
        {
            let annotations = page.annotations_mut();
            let _text_ann = annotations.create_text_annotation("Test comment");
        }

        // Create a highlight annotation with /Rect, /QuadPoints, /C, /IC.
        {
            let annotations = page.annotations_mut();
            if let Ok(mut h) = annotations.create_highlight_annotation() {
                let rect = PdfRect::new(
                    PdfPoints::new(100.0),
                    PdfPoints::new(50.0),
                    PdfPoints::new(112.0),
                    PdfPoints::new(200.0),
                );
                let _ = h.set_bounds(rect);
                let _ = h
                    .attachment_points_mut()
                    .create_attachment_point_at_end(PdfQuadPoints::from_rect(&rect));
                let pdf_color = PdfColor::new(255, 255, 0, 128);
                let _ = h.set_stroke_color(pdf_color);
                let _ = h.set_fill_color(pdf_color);
            }
        }

        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document.save_to_file(tmp.path()).expect("save PDF");

        let result =
            inspect_pdf_with_pdfium(&pdfium, tmp.path(), None).expect("inspect must succeed");

        // There should be annotations of different types.
        assert!(result.annotations.len() >= 2, "at least highlight + text");
        assert_eq!(result.highlight_count, 1, "only the highlight counts");
        assert_eq!(
            result.unique_colors,
            vec!["#FFFF00".to_string()],
            "only highlight colors in unique_colors"
        );

        // Text annotations must have quad_points_count == 0.
        for ann in result
            .annotations
            .iter()
            .filter(|a| a.annotation_type == "text")
        {
            assert_eq!(
                ann.quad_points_count, 0,
                "text annotations have no QuadPoints"
            );
        }
    }

    /// T-INSPECT-016: annotation_type_to_string covers all explicitly mapped
    /// annotation types.
    #[test]
    fn t_inspect_016_all_type_mappings() {
        let cases = vec![
            (PdfPageAnnotationType::Highlight, "highlight"),
            (PdfPageAnnotationType::Text, "text"),
            (PdfPageAnnotationType::Link, "link"),
            (PdfPageAnnotationType::FreeText, "free_text"),
            (PdfPageAnnotationType::Circle, "circle"),
            (PdfPageAnnotationType::Square, "square"),
            (PdfPageAnnotationType::Ink, "ink"),
            (PdfPageAnnotationType::Stamp, "stamp"),
            (PdfPageAnnotationType::Popup, "popup"),
            (PdfPageAnnotationType::Underline, "underline"),
            (PdfPageAnnotationType::Strikeout, "strikeout"),
            (PdfPageAnnotationType::Squiggly, "squiggly"),
            (PdfPageAnnotationType::Redacted, "redacted"),
            (PdfPageAnnotationType::Widget, "widget"),
            (PdfPageAnnotationType::XfaWidget, "xfa_widget"),
        ];

        for (ann_type, expected) in cases {
            assert_eq!(
                annotation_type_to_string(ann_type),
                expected,
                "mapping for {:?}",
                ann_type
            );
        }
    }

    // -----------------------------------------------------------------------
    // Tests for the lopdf-based inspect_pdf code path (DEFECT-004 regression)
    // -----------------------------------------------------------------------

    /// T-INSPECT-017: Round-trip through lopdf AP injection. Creates a pdfium
    /// PDF with a yellow highlight annotation, runs inject_appearance_streams
    /// (the lopdf post-processing step from the production annotation pipeline
    /// that converts inline annotation dicts to indirect object references and
    /// adds /AP appearance streams), then calls inspect_pdf.
    ///
    /// This test is the primary regression guard for DEFECT-004: before the
    /// fix, inspect_pdf called pdfium's FPDF_Annot_GetAttachmentPoints C API
    /// on lopdf-modified PDFs, triggering a Windows SEH access violation that
    /// killed the MCP server process and produced "MCP error -32000: Connection
    /// closed" on every neuroncite_inspect_annotations call.
    #[test]
    fn t_inspect_017_inspect_after_ap_injection() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");
        let mut page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create page");

        // Create a yellow highlight as the production annotate.rs pipeline does:
        // set /Rect, /QuadPoints, /C (stroke), and /IC (fill).
        {
            let annotations = page.annotations_mut();
            if let Ok(mut h) = annotations.create_highlight_annotation() {
                let rect = PdfRect::new(
                    PdfPoints::new(100.0),
                    PdfPoints::new(100.0),
                    PdfPoints::new(120.0),
                    PdfPoints::new(300.0),
                );
                let _ = h.set_bounds(rect);
                let _ = h
                    .attachment_points_mut()
                    .create_attachment_point_at_end(PdfQuadPoints::from_rect(&rect));
                let pdf_color = PdfColor::new(255, 255, 0, 128);
                let _ = h.set_stroke_color(pdf_color);
                let _ = h.set_fill_color(pdf_color);
            }
        }

        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document.save_to_file(tmp.path()).expect("save PDF");

        // Run the production annotation pipeline's lopdf post-processing step.
        // This promotes inline annotation dicts to indirect references and
        // injects /AP appearance streams -- the transformation that caused
        // pdfium's FPDF_Annot_GetAttachmentPoints to crash in the old code.
        crate::appearance::inject_appearance_streams(tmp.path())
            .expect("AP injection must succeed");

        // inspect_pdf uses the lopdf path and must not crash on the modified PDF.
        let result = inspect_pdf(tmp.path(), None)
            .expect("inspect_pdf must not crash on lopdf-processed PDF");

        assert_eq!(result.total_pages, 1);
        assert_eq!(
            result.highlight_count, 1,
            "must find the highlight annotation after AP injection"
        );
        assert!(
            result.unique_colors.contains(&"#FFFF00".to_string()),
            "must report yellow color from /C entry after AP injection, found: {:?}",
            result.unique_colors
        );
        let h = result
            .annotations
            .iter()
            .find(|a| a.annotation_type == "highlight")
            .expect("must find highlight annotation");
        assert!(
            h.quad_points_count >= 1,
            "highlight must have QuadPoints for visible rendering after AP injection"
        );
    }

    /// T-INSPECT-018: A PDF written by pdfium only (without lopdf
    /// post-processing) is correctly readable through inspect_pdf's lopdf code
    /// path. Verifies that lopdf handles pdfium's inline annotation dict format
    /// (annotations stored as inline dicts directly inside the /Annots array,
    /// as opposed to indirect references).
    #[test]
    fn t_inspect_018_pdfium_only_pdf_via_lopdf_path() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");
        let mut page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create page");

        {
            let annotations = page.annotations_mut();
            if let Ok(mut h) = annotations.create_highlight_annotation() {
                let rect = PdfRect::new(
                    PdfPoints::new(50.0),
                    PdfPoints::new(50.0),
                    PdfPoints::new(70.0),
                    PdfPoints::new(200.0),
                );
                let _ = h.set_bounds(rect);
                let _ = h
                    .attachment_points_mut()
                    .create_attachment_point_at_end(PdfQuadPoints::from_rect(&rect));
                let pdf_color = PdfColor::new(255, 0, 0, 128);
                let _ = h.set_stroke_color(pdf_color);
                let _ = h.set_fill_color(pdf_color);
            }
        }

        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document.save_to_file(tmp.path()).expect("save PDF");

        // No inject_appearance_streams -- pure pdfium-written PDF with inline
        // annotation dicts. lopdf must handle this format correctly.
        let result =
            inspect_pdf(tmp.path(), None).expect("inspect_pdf must succeed on pdfium-only PDF");

        assert_eq!(result.highlight_count, 1);
        assert!(
            result.unique_colors.contains(&"#FF0000".to_string()),
            "lopdf path must detect red color from pdfium inline annotation dict, \
             found: {:?}",
            result.unique_colors
        );
    }

    /// T-INSPECT-019: read_quad_points_count returns 0 for an annotation
    /// dictionary without a /QuadPoints entry.
    #[test]
    fn t_inspect_019_quad_points_count_absent_returns_zero() {
        let mut dict = lopdf::Dictionary::new();
        dict.set(b"Subtype", lopdf::Object::Name(b"Highlight".to_vec()));
        assert_eq!(
            read_quad_points_count(&dict),
            0,
            "absent /QuadPoints entry must return count 0"
        );
    }

    /// T-INSPECT-020: read_quad_points_count returns 2 for a /QuadPoints
    /// array with 16 elements (two groups of 8 floats each, where each group
    /// defines one highlighted line region).
    #[test]
    fn t_inspect_020_quad_points_count_two_groups() {
        let mut dict = lopdf::Dictionary::new();
        let quad_vals: Vec<lopdf::Object> = (0..16).map(|_| lopdf::Object::Real(0.0)).collect();
        dict.set(b"QuadPoints", lopdf::Object::Array(quad_vals));
        assert_eq!(
            read_quad_points_count(&dict),
            2,
            "/QuadPoints with 16 elements must yield 2 groups (16 / 8)"
        );
    }

    /// T-INSPECT-021: read_color_entry returns None for an absent color entry.
    #[test]
    fn t_inspect_021_read_color_entry_absent() {
        let dict = lopdf::Dictionary::new();
        assert!(
            read_color_entry(&dict, b"C").is_none(),
            "absent /C entry must return None"
        );
    }

    /// T-INSPECT-022: read_color_entry returns the correct uppercase hex
    /// string for a 3-element RGB array [1.0, 0.0, 0.0] (pure red).
    #[test]
    fn t_inspect_022_read_color_entry_rgb_red() {
        let mut dict = lopdf::Dictionary::new();
        dict.set(
            b"C",
            lopdf::Object::Array(vec![
                lopdf::Object::Real(1.0),
                lopdf::Object::Real(0.0),
                lopdf::Object::Real(0.0),
            ]),
        );
        assert_eq!(
            read_color_entry(&dict, b"C").as_deref(),
            Some("#FF0000"),
            "RGB [1.0, 0.0, 0.0] must map to #FF0000"
        );
    }
}
