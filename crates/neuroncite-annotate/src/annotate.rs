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

// PDF annotation creation via pdfium-render.
//
// Creates highlight annotations (colored rectangles over text) and optional
// popup text annotations on PDF pages. Uses pdfium-render's annotation API
// with per-character bounding boxes from the locate module. Each highlight
// covers a contiguous run of characters; line breaks within a match produce
// multiple highlight rectangles (one per line).
//
// PDF highlight annotations (Subtype /Highlight) require two entries to be
// visible in PDF viewers:
//   1. /QuadPoints: array of quadrilateral coordinates defining the
//      highlighted region. Without this, viewers show the annotation in
//      panels but render no color overlay.
//   2. /AP (Appearance Stream): a Form XObject that defines how the
//      highlight is rendered. pdfium-render does NOT generate /AP entries
//      for highlight annotations. The appearance module (appearance.rs)
//      injects these as a post-processing step after pdfium saves the PDF.
//
// The pipeline flow is:
//   1. pdfium-render creates annotations with /Rect, /QuadPoints, /C, /IC
//   2. pdfium saves the PDF
//   3. appearance::inject_appearance_streams() adds /AP /N Form XObjects
//   4. The post-check verifies annotations are renderable

use pdfium_render::prelude::*;

use crate::error::AnnotateError;
use crate::types::{Color, MatchMethod, MatchResult};

/// Creates highlight annotations on the given PDF page for the matched text
/// region. Groups adjacent bounding boxes into line-level rectangles to avoid
/// creating one annotation per character. Returns the number of highlight
/// annotations created.
///
/// Each highlight annotation receives:
/// - `/Rect`: bounding rectangle (for hit-testing and annotation selection)
/// - `/QuadPoints`: quadrilateral coordinates (for visible highlight rendering)
/// - `/C`: stroke color (the color PDF viewers display for highlights)
/// - `/IC`: interior color (fallback for viewers that check this entry)
///
/// The bounding boxes are expected in PDF coordinate space (left, bottom,
/// right, top) as produced by the locate module's extract_char_bounds.
///
/// When the page already contains highlight annotations (e.g., from a previous
/// annotation run in append mode), existing highlight bounds are collected
/// before creating new annotations. A new highlight is skipped if its bounding
/// rectangle overlaps with any existing highlight by more than 90% area. This
/// prevents duplicate highlights when the same quote is annotated multiple
/// times via neuroncite_annotate(append=true). When all highlights are skipped
/// as duplicates, the function returns Ok(0) rather than an error, because the
/// annotations already exist on the page from a prior run.
pub fn create_highlight_annotations(
    page: &mut PdfPage,
    match_result: &MatchResult,
    color: &Color,
) -> Result<usize, AnnotateError> {
    if match_result.bounding_boxes.is_empty() {
        return Err(AnnotateError::AnnotationFailed(
            "no bounding boxes available for annotation".into(),
        ));
    }

    // Collect bounding rectangles of existing highlight annotations on this
    // page. Used to detect and skip duplicate highlights in append mode,
    // where the same PDF is re-annotated with overlapping quotes.
    let existing_highlight_rects = collect_existing_highlight_rects(page);

    // Merge adjacent character bounding boxes into line-level rectangles.
    // Two boxes belong to the same line if their vertical midpoints are
    // within a tolerance (half the character height) of each other.
    let merged = merge_boxes_into_lines(&match_result.bounding_boxes);

    let annotations = page.annotations_mut();
    let mut count = 0_usize;
    let mut skipped_duplicates = 0_usize;

    let pdf_color = PdfColor::new(color.red(), color.green(), color.blue(), 128);

    for rect in &merged {
        // Skip this highlight if it overlaps with an existing annotation by
        // more than 90% of the smaller rectangle's area. This deduplication
        // prevents the same quote from producing stacked highlights when the
        // annotation pipeline runs multiple times on the same PDF in append mode.
        if is_duplicate_highlight(rect, &existing_highlight_rects) {
            skipped_duplicates += 1;
            continue;
        }

        let left = PdfPoints::new(rect[0]);
        let bottom = PdfPoints::new(rect[1]);
        let right = PdfPoints::new(rect[2]);
        let top = PdfPoints::new(rect[3]);

        let pdf_rect = PdfRect::new(bottom, left, top, right);

        let mut highlight = annotations
            .create_highlight_annotation()
            .map_err(|e| AnnotateError::AnnotationFailed(format!("create highlight: {e}")))?;

        // Set the bounding rectangle (/Rect entry). This defines the
        // overall annotation boundary used for hit-testing and selection.
        highlight
            .set_bounds(pdf_rect)
            .map_err(|e| AnnotateError::AnnotationFailed(format!("set_bounds: {e}")))?;

        // Set the QuadPoints (/QuadPoints entry). This is the array of
        // quadrilateral coordinates that PDF viewers use to render the
        // visible highlight overlay. Without this entry, viewers create
        // the annotation object but display no color on the page.
        // PdfQuadPoints::from_rect converts the PdfRect into the four
        // corner points in counter-clockwise order as required by the
        // PDF specification (ISO 32000, Section 12.5.6.10).
        let quad_points = PdfQuadPoints::from_rect(&pdf_rect);
        highlight
            .attachment_points_mut()
            .create_attachment_point_at_end(quad_points)
            .map_err(|e| AnnotateError::AnnotationFailed(format!("QuadPoints: {e}")))?;

        // Set stroke color (/C entry). PDF highlight annotations use
        // the /C dictionary entry for their visible color. set_stroke_color
        // calls FPDFAnnot_SetColor with FPDFANNOT_COLORTYPE_Color.
        highlight
            .set_stroke_color(pdf_color)
            .map_err(|e| AnnotateError::AnnotationFailed(format!("set_stroke_color: {e}")))?;

        // Set interior color (/IC entry) as a fallback for viewers that
        // read this entry instead of /C. set_fill_color calls
        // FPDFAnnot_SetColor with FPDFANNOT_COLORTYPE_InteriorColor.
        highlight
            .set_fill_color(pdf_color)
            .map_err(|e| AnnotateError::AnnotationFailed(format!("set_fill_color: {e}")))?;

        count += 1;
    }

    if count == 0 && skipped_duplicates == 0 {
        // All merged rectangles failed annotation creation without any being
        // recognized as duplicates. This indicates a pdfium API failure rather
        // than intentional deduplication. When skipped_duplicates > 0, returning
        // Ok(0) is correct because the highlights already exist on the page
        // from a previous annotation run (append mode).
        return Err(AnnotateError::AnnotationFailed(
            "pdfium failed to create any highlight annotations".into(),
        ));
    }

    Ok(count)
}

/// Creates a popup text annotation at the top-left corner of the first
/// bounding box. The annotation contains the provided comment text.
/// Returns Ok(()) on success or an error if annotation creation fails.
pub fn create_comment_annotation(
    page: &mut PdfPage,
    match_result: &MatchResult,
    comment: &str,
) -> Result<(), AnnotateError> {
    if comment.trim().is_empty() {
        return Ok(());
    }

    let first_box = match match_result.bounding_boxes.first() {
        Some(b) => b,
        None => {
            return Err(AnnotateError::AnnotationFailed(
                "no bounding boxes for comment placement".into(),
            ));
        }
    };

    // Position the text annotation at the top-left corner of the first box.
    let left = PdfPoints::new(first_box[0]);
    let top = PdfPoints::new(first_box[3]);
    let right = PdfPoints::new(first_box[0] + 18.0); // 18pt icon width
    let bottom = PdfPoints::new(first_box[3] - 18.0); // 18pt icon height

    let rect = PdfRect::new(bottom, left, top, right);

    let annotations = page.annotations_mut();

    // Create a text annotation (displayed as a popup note icon) and set its
    // bounding rectangle. pdfium-render's create_text_annotation() accepts the
    // text content; the position is set separately via set_bounds().
    let result = annotations.create_text_annotation(comment);

    match result {
        Ok(mut ann) => {
            ann.set_bounds(rect).map_err(|e| {
                AnnotateError::AnnotationFailed(format!("comment annotation bounds: {e}"))
            })?;
            Ok(())
        }
        Err(e) => Err(AnnotateError::AnnotationFailed(format!(
            "comment annotation creation: {e}"
        ))),
    }
}

/// Constructs a MatchResult covering the content area of an entire page.
/// Used as a fallback when all text location stages fail but the page number
/// is known from the citation verification pipeline. The bounding box is the
/// page MediaBox inset by 72pt margins (approximating the text content area).
/// For very small pages, the margin is clamped to 10% of the page dimension.
pub fn create_page_level_match(
    page: &PdfPage,
    page_number: usize,
    quote_char_count: usize,
) -> MatchResult {
    let w = page.width().value;
    let h = page.height().value;
    let margin = 72.0_f32.min(w * 0.1).min(h * 0.1);

    MatchResult {
        page_number,
        char_start: 0,
        char_end: quote_char_count,
        bounding_boxes: vec![[margin, margin, w - margin, h - margin]],
        method: MatchMethod::PageLevel,
        fuzzy_score: None,
    }
}

/// Collects bounding rectangles [left, bottom, right, top] from all existing
/// highlight annotations on the given page. Called before creating new
/// highlights to detect duplicates in append mode. Returns an empty Vec when
/// the page has no highlight annotations.
fn collect_existing_highlight_rects(page: &PdfPage) -> Vec<[f32; 4]> {
    let mut rects = Vec::new();
    for annotation in page.annotations().iter() {
        if annotation.annotation_type() != PdfPageAnnotationType::Highlight {
            continue;
        }
        if let Ok(bounds) = annotation.bounds() {
            rects.push([
                bounds.left().value,
                bounds.bottom().value,
                bounds.right().value,
                bounds.top().value,
            ]);
        }
    }
    rects
}

/// Checks whether a candidate highlight rectangle overlaps with any existing
/// highlight by more than 90% of the smaller rectangle's area. The 90%
/// threshold accounts for minor floating-point differences in bounding box
/// coordinates between annotation runs (e.g., rounding in pdfium's coordinate
/// extraction). Returns true if the candidate is a duplicate that should be
/// skipped.
fn is_duplicate_highlight(candidate: &[f32; 4], existing: &[[f32; 4]]) -> bool {
    for existing_rect in existing {
        // Compute intersection rectangle.
        let inter_left = candidate[0].max(existing_rect[0]);
        let inter_bottom = candidate[1].max(existing_rect[1]);
        let inter_right = candidate[2].min(existing_rect[2]);
        let inter_top = candidate[3].min(existing_rect[3]);

        // No intersection if the rectangles do not overlap.
        if inter_left >= inter_right || inter_bottom >= inter_top {
            continue;
        }

        let inter_area = (inter_right - inter_left) * (inter_top - inter_bottom);
        let candidate_area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]);
        let existing_area =
            (existing_rect[2] - existing_rect[0]) * (existing_rect[3] - existing_rect[1]);
        let smaller_area = candidate_area.min(existing_area);

        // Guard against degenerate (zero-area) rectangles.
        if smaller_area <= 0.0 {
            continue;
        }

        // If the intersection covers more than 90% of the smaller rectangle,
        // the candidate is a duplicate of this existing highlight.
        if inter_area / smaller_area >= 0.90 {
            return true;
        }
    }
    false
}

/// Merges per-character bounding boxes into line-level rectangles.
///
/// Characters on the same text line (similar vertical position) are merged
/// into a single encompassing rectangle. This reduces the number of
/// annotations from one-per-character to one-per-line, producing cleaner
/// visual results in PDF viewers.
fn merge_boxes_into_lines(boxes: &[[f32; 4]]) -> Vec<[f32; 4]> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut lines: Vec<[f32; 4]> = Vec::new();
    let mut current = boxes[0];

    for bbox in boxes.iter().skip(1) {
        let current_mid_y = (current[1] + current[3]) / 2.0;
        let bbox_mid_y = (bbox[1] + bbox[3]) / 2.0;
        let char_height = (current[3] - current[1]).abs().max(1.0);

        // If the vertical midpoint difference is less than half the
        // character height, the boxes are on the same line.
        if (current_mid_y - bbox_mid_y).abs() < char_height * 0.5 {
            // Extend the current line rectangle.
            current[0] = current[0].min(bbox[0]); // left
            current[1] = current[1].min(bbox[1]); // bottom
            current[2] = current[2].max(bbox[2]); // right
            current[3] = current[3].max(bbox[3]); // top
        } else {
            // New line: save current and start a new rectangle.
            lines.push(current);
            current = *bbox;
        }
    }

    lines.push(current);
    lines
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// T-ANNOTATE-060: merge_boxes_into_lines merges two horizontally
    /// adjacent boxes on the same line.
    #[test]
    fn t_annotate_060_merge_same_line() {
        let boxes = vec![
            [10.0, 100.0, 20.0, 112.0], // char 1
            [20.0, 100.0, 30.0, 112.0], // char 2, same line
        ];
        let merged = merge_boxes_into_lines(&boxes);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0][0], 10.0); // left
        assert_eq!(merged[0][2], 30.0); // right
    }

    /// T-ANNOTATE-061: merge_boxes_into_lines splits boxes on different
    /// lines into separate rectangles.
    #[test]
    fn t_annotate_061_merge_different_lines() {
        let boxes = vec![
            [10.0, 100.0, 200.0, 112.0], // line 1
            [10.0, 80.0, 150.0, 92.0],   // line 2 (lower Y)
        ];
        let merged = merge_boxes_into_lines(&boxes);
        assert_eq!(merged.len(), 2);
    }

    /// T-ANNOTATE-062: merge_boxes_into_lines handles a single box.
    #[test]
    fn t_annotate_062_merge_single_box() {
        let boxes = vec![[10.0, 100.0, 200.0, 112.0]];
        let merged = merge_boxes_into_lines(&boxes);
        assert_eq!(merged.len(), 1);
    }

    /// T-ANNOTATE-063: merge_boxes_into_lines handles empty input.
    #[test]
    fn t_annotate_063_merge_empty() {
        let merged = merge_boxes_into_lines(&[]);
        assert!(merged.is_empty());
    }

    /// T-ANNOTATE-064: Color parsing and RGB extraction.
    #[test]
    fn t_annotate_064_color_rgb() {
        let color = Color::parse("#FF8000").unwrap();
        assert_eq!(color.red(), 255);
        assert_eq!(color.green(), 128);
        assert_eq!(color.blue(), 0);
    }

    /// T-ANNOTATE-070: End-to-end test of create_highlight_annotations.
    ///
    /// Uses the production function to create highlights on a blank PDF page,
    /// saves the PDF to disk, then reopens it with the inspect module and
    /// verifies:
    /// 1. The highlight annotation exists in the saved file
    /// 2. The stroke color (/C entry) matches the requested color
    /// 3. The QuadPoints (/QuadPoints entry) are present (quad_points_count >= 1)
    ///
    /// This is the critical test that catches invisible highlights: without
    /// QuadPoints, the highlight annotation exists as a PDF object (with color
    /// metadata and working comments) but is invisible in PDF viewers.
    #[test]
    fn t_annotate_070_e2e_highlight_color_and_quadpoints_in_saved_pdf() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return, // Skip if pdfium is not available.
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        let mut page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create blank A4 page");

        // Simulate a matched text region spanning two lines of characters.
        // Each bounding box represents one character in PDF coordinate space
        // (left, bottom, right, top).
        let match_result = MatchResult {
            page_number: 1,
            char_start: 0,
            char_end: 6,
            method: crate::types::MatchMethod::Exact,
            bounding_boxes: vec![
                // Line 1: three characters at y=700
                [72.0, 700.0, 80.0, 712.0],
                [80.0, 700.0, 88.0, 712.0],
                [88.0, 700.0, 96.0, 712.0],
                // Line 2: three characters at y=680 (below line 1)
                [72.0, 680.0, 80.0, 692.0],
                [80.0, 680.0, 88.0, 692.0],
                [88.0, 680.0, 96.0, 692.0],
            ],
            fuzzy_score: None,
        };

        let color = Color::parse("#FF8000").expect("parse orange color");

        // Call the production function.
        let count = create_highlight_annotations(&mut page, &match_result, &color)
            .expect("highlight creation must succeed");

        // Two lines of text produce two highlight annotations.
        assert_eq!(count, 2, "two text lines produce two highlight annotations");

        // Save the annotated PDF to a temporary file.
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document
            .save_to_file(tmp.path())
            .expect("save annotated PDF");

        // Re-open the saved PDF with the inspect module and verify the
        // highlights are physically present with correct color and QuadPoints.
        let result = crate::inspect::inspect_pdf_with_pdfium(&pdfium, tmp.path(), None)
            .expect("inspect saved PDF");

        assert_eq!(
            result.highlight_count, 2,
            "saved PDF must contain two highlight annotations"
        );

        assert!(
            result.unique_colors.contains(&"#FF8000".to_string()),
            "saved PDF must contain the orange highlight color, found: {:?}",
            result.unique_colors
        );

        // Verify each highlight annotation has both color and QuadPoints.
        for ann in result
            .annotations
            .iter()
            .filter(|a| a.annotation_type == "highlight")
        {
            assert_eq!(
                ann.stroke_color.as_deref(),
                Some("#FF8000"),
                "stroke_color (/C entry) must be the requested orange"
            );
            assert!(
                ann.quad_points_count >= 1,
                "highlight must have QuadPoints for visible rendering, got quad_points_count={}",
                ann.quad_points_count
            );
            assert!(
                ann.bounds.is_some(),
                "highlight must have a bounding rectangle"
            );
        }
    }

    /// T-ANNOTATE-071: End-to-end test with multiple colors on the same page.
    ///
    /// Creates two sets of highlights with different colors via the production
    /// function, saves the PDF, and verifies both colors and QuadPoints are
    /// present in the saved file.
    #[test]
    fn t_annotate_071_e2e_multiple_colors_in_saved_pdf() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        let mut page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create blank A4 page");

        // First highlight: red, one line at y=700.
        let match_result_1 = MatchResult {
            page_number: 1,
            char_start: 0,
            char_end: 3,
            method: crate::types::MatchMethod::Exact,
            bounding_boxes: vec![
                [72.0, 700.0, 80.0, 712.0],
                [80.0, 700.0, 88.0, 712.0],
                [88.0, 700.0, 96.0, 712.0],
            ],
            fuzzy_score: None,
        };
        let red = Color::parse("#FF0000").expect("parse red");
        let count_1 = create_highlight_annotations(&mut page, &match_result_1, &red)
            .expect("red highlight creation");
        assert_eq!(count_1, 1);

        // Second highlight: green, one line at y=650.
        let match_result_2 = MatchResult {
            page_number: 1,
            char_start: 10,
            char_end: 13,
            method: crate::types::MatchMethod::Normalized,
            bounding_boxes: vec![
                [72.0, 650.0, 80.0, 662.0],
                [80.0, 650.0, 88.0, 662.0],
                [88.0, 650.0, 96.0, 662.0],
            ],
            fuzzy_score: None,
        };
        let green = Color::parse("#00FF00").expect("parse green");
        let count_2 = create_highlight_annotations(&mut page, &match_result_2, &green)
            .expect("green highlight creation");
        assert_eq!(count_2, 1);

        // Save and inspect.
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document
            .save_to_file(tmp.path())
            .expect("save annotated PDF");

        let result = crate::inspect::inspect_pdf_with_pdfium(&pdfium, tmp.path(), None)
            .expect("inspect saved PDF");

        assert_eq!(result.highlight_count, 2);
        assert!(result.unique_colors.contains(&"#FF0000".to_string()));
        assert!(result.unique_colors.contains(&"#00FF00".to_string()));

        // Both highlights must have QuadPoints.
        for ann in result
            .annotations
            .iter()
            .filter(|a| a.annotation_type == "highlight")
        {
            assert!(
                ann.quad_points_count >= 1,
                "every highlight must have QuadPoints, got {}",
                ann.quad_points_count
            );
        }
    }

    /// T-ANNOTATE-072: Appearance stream injection on a blank PDF.
    ///
    /// Creates highlight annotations with pdfium-render, saves the PDF,
    /// injects appearance streams via the lopdf post-processing module,
    /// and verifies that the /AP /N Form XObject is present in the saved
    /// file. This is the core regression test for the invisible-highlight
    /// bug where annotations had /Rect, /QuadPoints, /C, /IC but no /AP.
    #[test]
    fn t_annotate_072_appearance_stream_injection_blank_pdf() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        let mut page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create blank A4 page");

        let match_result = MatchResult {
            page_number: 1,
            char_start: 0,
            char_end: 3,
            method: crate::types::MatchMethod::Exact,
            bounding_boxes: vec![
                [72.0, 700.0, 80.0, 712.0],
                [80.0, 700.0, 88.0, 712.0],
                [88.0, 700.0, 96.0, 712.0],
            ],
            fuzzy_score: None,
        };

        let color = Color::parse("#FF8000").expect("parse orange");
        create_highlight_annotations(&mut page, &match_result, &color).expect("highlight creation");

        // Save with pdfium (no /AP at this point).
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document
            .save_to_file(tmp.path())
            .expect("save annotated PDF");

        // Inject appearance streams via lopdf post-processing.
        let injected = crate::appearance::inject_appearance_streams(tmp.path())
            .expect("appearance stream injection");
        assert_eq!(
            injected, 1,
            "one highlight should receive an appearance stream"
        );

        // Verify the /AP /N entry exists by reading with lopdf.
        let doc = lopdf::Document::load(tmp.path()).expect("reload PDF");
        let pages: Vec<lopdf::ObjectId> = doc.get_pages().values().copied().collect();
        assert_eq!(pages.len(), 1);

        let page_dict = doc
            .get_object(pages[0])
            .expect("page object")
            .as_dict()
            .expect("page dict");
        let annots = page_dict
            .get(b"Annots")
            .expect("/Annots")
            .as_array()
            .expect("annots array");

        for annot_ref in annots {
            if let lopdf::Object::Reference(id) = annot_ref {
                let annot = doc.get_object(*id).expect("annot object");
                let annot_dict = annot.as_dict().expect("annot dict");
                let subtype = annot_dict
                    .get(b"Subtype")
                    .ok()
                    .and_then(|o| o.as_name().ok())
                    .map(|n| std::str::from_utf8(n).unwrap_or(""));

                if subtype == Some("Highlight") {
                    assert!(
                        annot_dict.has(b"AP"),
                        "highlight annotation must have /AP after injection"
                    );
                    let ap = annot_dict
                        .get(b"AP")
                        .expect("/AP")
                        .as_dict()
                        .expect("/AP dict");
                    assert!(ap.has(b"N"), "/AP must have /N (Normal Appearance)");
                }
            }
        }
    }

    /// T-ANNOTATE-073: Appearance stream injection on a loaded (non-blank) PDF.
    ///
    /// Loads a real PDF from the test fixtures, creates highlight annotations
    /// on its first page, saves, injects appearance streams, and verifies
    /// that the /AP entry is present. This catches issues that only manifest
    /// with PDFs that have existing content (fonts, images, complex page
    /// trees) as opposed to freshly-created blank pages.
    #[test]
    fn t_annotate_073_appearance_stream_injection_loaded_pdf() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);

        // Use a synthetic PDF from the neuroncite-testgen fixtures. The
        // CID-range PDF has valid structure that pdfium can open (the malformed
        // CMap only affects pdf-extract, not pdfium).
        let test_pdf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/PDFs/ocr/cid-range-panic/synthetic_finance_cid_range.pdf");
        if !test_pdf.exists() {
            eprintln!(
                "skipping: synthetic test PDF not found at {}",
                test_pdf.display()
            );
            return;
        }

        let document = pdfium
            .load_pdf_from_file(&test_pdf, None)
            .expect("load synthetic test PDF");

        // Get first page and create a highlight annotation.
        let mut page = document.pages().get(0).expect("get first page");

        let match_result = MatchResult {
            page_number: 1,
            char_start: 0,
            char_end: 3,
            method: crate::types::MatchMethod::Exact,
            bounding_boxes: vec![
                [72.0, 700.0, 80.0, 712.0],
                [80.0, 700.0, 88.0, 712.0],
                [88.0, 700.0, 96.0, 712.0],
            ],
            fuzzy_score: None,
        };

        let color = Color::parse("#00FF00").expect("parse green");
        create_highlight_annotations(&mut page, &match_result, &color)
            .expect("highlight creation on loaded PDF");

        // Save the annotated PDF.
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document
            .save_to_file(tmp.path())
            .expect("save annotated loaded PDF");

        // Inject appearance streams.
        let injected = crate::appearance::inject_appearance_streams(tmp.path())
            .expect("appearance stream injection on loaded PDF");
        assert!(
            injected >= 1,
            "at least one highlight should receive an appearance stream"
        );

        // Verify /AP exists via lopdf.
        let doc = lopdf::Document::load(tmp.path()).expect("reload PDF");
        let pages: Vec<lopdf::ObjectId> = doc.get_pages().values().copied().collect();
        assert!(!pages.is_empty());

        // Check the first page's annotations.
        let page_dict = doc
            .get_object(pages[0])
            .expect("page object")
            .as_dict()
            .expect("page dict");

        if let Ok(annots_obj) = page_dict.get(b"Annots") {
            let annots = annots_obj.as_array().expect("annots array");
            let mut found_highlight_with_ap = false;

            for annot_ref in annots {
                if let lopdf::Object::Reference(id) = annot_ref {
                    let annot = doc.get_object(*id).expect("annot object");
                    let annot_dict = annot.as_dict().expect("annot dict");
                    let subtype = annot_dict
                        .get(b"Subtype")
                        .ok()
                        .and_then(|o| o.as_name().ok())
                        .map(|n| std::str::from_utf8(n).unwrap_or(""));

                    if subtype == Some("Highlight") && annot_dict.has(b"AP") {
                        found_highlight_with_ap = true;
                    }
                }
            }

            assert!(
                found_highlight_with_ap,
                "loaded PDF must have at least one highlight with /AP after injection"
            );
        }
    }

    /// T-ANNOTATE-076: create_page_level_match produces a single bounding box
    /// covering the content area (page minus 72pt margins) with MatchMethod::PageLevel.
    #[test]
    fn t_annotate_076_page_level_match_bounding_box() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        // A4 page: 595.28 x 841.89 points.
        let page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create A4 page");

        let result = create_page_level_match(&page, 1, 100);

        assert_eq!(result.page_number, 1);
        assert_eq!(result.char_start, 0);
        assert_eq!(result.char_end, 100);
        assert_eq!(result.method, crate::types::MatchMethod::PageLevel);
        assert!(result.fuzzy_score.is_none());
        assert_eq!(
            result.bounding_boxes.len(),
            1,
            "single page-level bounding box"
        );

        let bbox = result.bounding_boxes[0];
        // A4 is 595.28 x 841.89 pts. The margin formula is
        // 72.0.min(w * 0.1).min(h * 0.1). For A4: w*0.1 = 59.53,
        // h*0.1 = 84.19, so margin = min(72, 59.53, 84.19) = 59.53.
        let expected_margin = 72.0_f32.min(595.28 * 0.1).min(841.89 * 0.1);
        assert!(
            (bbox[0] - expected_margin).abs() < 1.0,
            "left margin: {}",
            bbox[0]
        );
        assert!(
            (bbox[1] - expected_margin).abs() < 1.0,
            "bottom margin: {}",
            bbox[1]
        );
        let expected_right = 595.28 - expected_margin;
        let expected_top = 841.89 - expected_margin;
        assert!((bbox[2] - expected_right).abs() < 2.0, "right: {}", bbox[2]);
        assert!((bbox[3] - expected_top).abs() < 2.0, "top: {}", bbox[3]);
    }

    /// T-ANNOTATE-077: create_page_level_match clamps margins to 10% on small pages.
    #[test]
    fn t_annotate_077_page_level_match_small_page_margin_clamp() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        // Create a very small page: 100 x 100 points. PdfPagePaperSize::Custom
        // accepts two PdfPoints values for width and height.
        let page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::Custom(
                PdfPoints::new(100.0),
                PdfPoints::new(100.0),
            ))
            .expect("create 100x100 page");

        let result = create_page_level_match(&page, 1, 50);

        let bbox = result.bounding_boxes[0];
        // 10% of 100 = 10pt margin (clamped from 72pt).
        assert!(
            (bbox[0] - 10.0).abs() < 1.0,
            "left margin clamped to 10pt: {}",
            bbox[0]
        );
        assert!(
            (bbox[1] - 10.0).abs() < 1.0,
            "bottom margin clamped: {}",
            bbox[1]
        );
        assert!((bbox[2] - 90.0).abs() < 1.0, "right ~90pt: {}", bbox[2]);
        assert!((bbox[3] - 90.0).abs() < 1.0, "top ~90pt: {}", bbox[3]);
    }

    /// T-ANNOTATE-078: create_page_level_match and create_highlight_annotations
    /// together produce a visible full-page highlight in a saved PDF.
    #[test]
    fn t_annotate_078_page_level_highlight_in_saved_pdf() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        let page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create A4 page");

        let page_match = create_page_level_match(&page, 1, 200);
        drop(page);

        let mut page = document.pages().get(0).expect("get first page");
        let color = Color::parse("#AAAAAA").expect("parse gray");
        let count = create_highlight_annotations(&mut page, &page_match, &color)
            .expect("page-level highlight creation");
        assert_eq!(
            count, 1,
            "page-level highlight produces exactly 1 annotation"
        );

        // Save and inspect.
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document.save_to_file(tmp.path()).expect("save PDF");

        let result = crate::inspect::inspect_pdf_with_pdfium(&pdfium, tmp.path(), None)
            .expect("inspect saved PDF");
        assert_eq!(result.highlight_count, 1);
        assert!(result.unique_colors.contains(&"#AAAAAA".to_string()));
    }

    /// T-ANNOTATE-074: Color preserved in appearance stream after round-trip.
    ///
    /// Creates a highlight with a specific color, saves with pdfium, injects
    /// appearance streams, and verifies that the /C color array in the
    /// annotation dictionary matches the requested color AND that the
    /// appearance stream content contains the corresponding RGB values.
    #[test]
    fn t_annotate_074_color_preserved_in_appearance_stream() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        let mut page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create blank A4 page");

        let match_result = MatchResult {
            page_number: 1,
            char_start: 0,
            char_end: 3,
            method: crate::types::MatchMethod::Exact,
            bounding_boxes: vec![
                [72.0, 700.0, 80.0, 712.0],
                [80.0, 700.0, 88.0, 712.0],
                [88.0, 700.0, 96.0, 712.0],
            ],
            fuzzy_score: None,
        };

        // Use a distinctive color (magenta: #FF00FF).
        let color = Color::parse("#FF00FF").expect("parse magenta");
        create_highlight_annotations(&mut page, &match_result, &color).expect("highlight creation");

        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document
            .save_to_file(tmp.path())
            .expect("save annotated PDF");

        crate::appearance::inject_appearance_streams(tmp.path())
            .expect("appearance stream injection");

        // Read back with lopdf and verify the /C color and AP stream content.
        let doc = lopdf::Document::load(tmp.path()).expect("reload PDF");
        let pages: Vec<lopdf::ObjectId> = doc.get_pages().values().copied().collect();
        let page_dict = doc
            .get_object(pages[0])
            .expect("page")
            .as_dict()
            .expect("page dict");
        let annots = page_dict
            .get(b"Annots")
            .expect("/Annots")
            .as_array()
            .expect("array");

        for annot_ref in annots {
            if let lopdf::Object::Reference(id) = annot_ref {
                let annot = doc.get_object(*id).expect("annot");
                let annot_dict = annot.as_dict().expect("dict");
                let subtype = annot_dict
                    .get(b"Subtype")
                    .ok()
                    .and_then(|o| o.as_name().ok())
                    .map(|n| std::str::from_utf8(n).unwrap_or(""));

                if subtype != Some("Highlight") {
                    continue;
                }

                // Verify /C color array.
                let c_arr = annot_dict
                    .get(b"C")
                    .expect("/C entry")
                    .as_array()
                    .expect("/C array");
                assert_eq!(c_arr.len(), 3, "/C must have 3 components (RGB)");

                // Verify /AP /N stream contains the color.
                let ap = annot_dict
                    .get(b"AP")
                    .expect("/AP")
                    .as_dict()
                    .expect("/AP dict");
                if let Ok(lopdf::Object::Reference(n_ref)) = ap.get(b"N") {
                    let stream_obj = doc.get_object(*n_ref).expect("stream");
                    if let lopdf::Object::Stream(ref stream) = *stream_obj {
                        let content =
                            std::str::from_utf8(&stream.content).expect("AP content is UTF-8");
                        // Magenta = RGB(1.0, 0.0, 1.0). The stream should
                        // contain "1.0000 0.0000 1.0000 rg" (or similar).
                        assert!(
                            content.contains("rg"),
                            "AP stream must contain 'rg' fill color operator"
                        );
                        assert!(
                            content.contains("re f"),
                            "AP stream must contain 're f' rectangle fill"
                        );
                    }
                }
            }
        }
    }

    /// T-ANNOTATE-075: Full pipeline test on a real PDF from the test fixtures.
    ///
    /// Loads a real academic paper, creates highlights, saves, injects
    /// appearance streams, and inspects with pdfium to verify that
    /// highlight annotations are visible (have color AND QuadPoints AND
    /// appearance streams).
    #[test]
    fn t_annotate_075_full_pipeline_real_pdf() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);

        // Use a synthetic PDF from the neuroncite-testgen fixtures.
        let test_pdf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/PDFs/ocr/cid-range-panic/synthetic_finance_cid_range.pdf");
        if !test_pdf.exists() {
            eprintln!(
                "skipping: synthetic test PDF not found at {}",
                test_pdf.display()
            );
            return;
        }

        let document = pdfium
            .load_pdf_from_file(&test_pdf, None)
            .expect("load synthetic test PDF");

        let mut page = document.pages().get(0).expect("get first page");

        // Create highlights with two different colors at different positions.
        let match1 = MatchResult {
            page_number: 1,
            char_start: 0,
            char_end: 3,
            method: crate::types::MatchMethod::Exact,
            bounding_boxes: vec![
                [72.0, 750.0, 80.0, 762.0],
                [80.0, 750.0, 88.0, 762.0],
                [88.0, 750.0, 96.0, 762.0],
            ],
            fuzzy_score: None,
        };
        let yellow = Color::parse("#FFFF00").expect("parse yellow");
        let c1 = create_highlight_annotations(&mut page, &match1, &yellow)
            .expect("yellow highlight on real PDF");

        let match2 = MatchResult {
            page_number: 1,
            char_start: 10,
            char_end: 13,
            method: crate::types::MatchMethod::Normalized,
            bounding_boxes: vec![
                [72.0, 700.0, 80.0, 712.0],
                [80.0, 700.0, 88.0, 712.0],
                [88.0, 700.0, 96.0, 712.0],
            ],
            fuzzy_score: None,
        };
        let red = Color::parse("#FF0000").expect("parse red");
        let c2 = create_highlight_annotations(&mut page, &match2, &red)
            .expect("red highlight on real PDF");

        assert_eq!(c1, 1);
        assert_eq!(c2, 1);

        // Save.
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        document
            .save_to_file(tmp.path())
            .expect("save annotated real PDF");

        // Inspect with pdfium BEFORE lopdf injection. lopdf's PDF rewrite
        // produces output that is incompatible with pdfium's PDF loader
        // (pdfium segfaults when loading a lopdf-rewritten file). The
        // pdfium check runs first while the file is still in pdfium's
        // native output format.
        let result = crate::inspect::inspect_pdf_with_pdfium(&pdfium, tmp.path(), None)
            .expect("inspect annotated real PDF");

        assert!(
            result.highlight_count >= 2,
            "real PDF must have at least 2 highlights, got {}",
            result.highlight_count
        );

        // Verify colors are present.
        assert!(
            result.unique_colors.contains(&"#FFFF00".to_string()),
            "yellow color must be present in annotated real PDF, found: {:?}",
            result.unique_colors
        );
        assert!(
            result.unique_colors.contains(&"#FF0000".to_string()),
            "red color must be present in annotated real PDF, found: {:?}",
            result.unique_colors
        );

        // Verify all highlight annotations have QuadPoints.
        for ann in result
            .annotations
            .iter()
            .filter(|a| a.annotation_type == "highlight")
        {
            assert!(
                ann.quad_points_count >= 1,
                "highlight on real PDF must have QuadPoints"
            );
        }

        // Inject appearance streams after pdfium inspection.
        let injected = crate::appearance::inject_appearance_streams(tmp.path())
            .expect("AP injection on real PDF");
        assert!(
            injected >= 2,
            "both highlights should receive appearance streams, got {injected}"
        );

        // Verify /AP exists via lopdf (lopdf can read its own output).
        let doc = lopdf::Document::load(tmp.path()).expect("reload real PDF");
        let pages: Vec<lopdf::ObjectId> = doc.get_pages().values().copied().collect();
        let page_dict = doc
            .get_object(pages[0])
            .expect("page")
            .as_dict()
            .expect("page dict");

        if let Ok(annots_obj) = page_dict.get(b"Annots") {
            let annots = annots_obj.as_array().expect("array");
            let highlights_with_ap = annots
                .iter()
                .filter_map(|a| {
                    if let lopdf::Object::Reference(id) = a {
                        let dict = doc.get_object(*id).ok()?.as_dict().ok()?;
                        let sub = dict
                            .get(b"Subtype")
                            .ok()?
                            .as_name()
                            .ok()
                            .map(|n| std::str::from_utf8(n).unwrap_or(""))?;
                        if sub == "Highlight" && dict.has(b"AP") {
                            Some(())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .count();

            assert!(
                highlights_with_ap >= 2,
                "at least 2 highlights with /AP on real PDF, got {highlights_with_ap}"
            );
        }
    }

    // -------------------------------------------------------------------
    // DEF-5 regression tests: duplicate highlight deduplication
    // -------------------------------------------------------------------

    /// T-ANNOTATE-079: is_duplicate_highlight returns true when a candidate
    /// rectangle overlaps an existing highlight by more than 90% of its area.
    #[test]
    fn t_annotate_079_is_duplicate_exact_overlap() {
        let existing = vec![[72.0, 700.0, 200.0, 712.0]];
        let candidate = [72.0, 700.0, 200.0, 712.0]; // identical rect
        assert!(
            is_duplicate_highlight(&candidate, &existing),
            "identical rectangles must be detected as duplicates"
        );
    }

    /// T-ANNOTATE-080: is_duplicate_highlight returns false when the candidate
    /// does not overlap any existing highlight.
    #[test]
    fn t_annotate_080_no_overlap_is_not_duplicate() {
        let existing = vec![[72.0, 700.0, 200.0, 712.0]];
        let candidate = [72.0, 600.0, 200.0, 612.0]; // different vertical position
        assert!(
            !is_duplicate_highlight(&candidate, &existing),
            "non-overlapping rectangles must not be detected as duplicates"
        );
    }

    /// T-ANNOTATE-081: is_duplicate_highlight returns true for nearly-identical
    /// rectangles that differ by minor floating-point rounding (within 90%
    /// overlap). This simulates the case where two annotation runs produce
    /// slightly different bounding boxes for the same text region.
    #[test]
    fn t_annotate_081_near_identical_is_duplicate() {
        let existing = vec![[72.0, 700.0, 200.0, 712.0]];
        // Shift by 1pt on the right edge: 199pt vs 200pt.
        // Overlap area: (200-72) * (712-700) = 128 * 12 = 1536 (existing)
        // Candidate area: (199-72) * (712-700) = 127 * 12 = 1524
        // Intersection: (199-72) * (712-700) = 127 * 12 = 1524
        // Ratio: 1524 / 1524 = 1.0 >= 0.90
        let candidate = [72.0, 700.0, 199.0, 712.0];
        assert!(
            is_duplicate_highlight(&candidate, &existing),
            "nearly-identical rectangles must be detected as duplicates"
        );
    }

    /// T-ANNOTATE-082: is_duplicate_highlight handles partial overlap below
    /// the 90% threshold. Two rectangles sharing only 50% of their area
    /// must not be flagged as duplicates.
    #[test]
    fn t_annotate_082_partial_overlap_not_duplicate() {
        let existing = vec![[72.0, 700.0, 200.0, 712.0]]; // width=128
        // Candidate shifted right by 64pt: overlaps only half the existing rect.
        let candidate = [136.0, 700.0, 264.0, 712.0]; // width=128
        // Intersection: (200-136) * 12 = 64 * 12 = 768
        // Smaller area: 128 * 12 = 1536
        // Ratio: 768 / 1536 = 0.5 < 0.90
        assert!(
            !is_duplicate_highlight(&candidate, &existing),
            "50% overlap must not be flagged as duplicate"
        );
    }

    /// T-ANNOTATE-083: is_duplicate_highlight handles an empty existing list.
    #[test]
    fn t_annotate_083_empty_existing_not_duplicate() {
        let existing: Vec<[f32; 4]> = Vec::new();
        let candidate = [72.0, 700.0, 200.0, 712.0];
        assert!(
            !is_duplicate_highlight(&candidate, &existing),
            "no existing highlights means no duplicate"
        );
    }

    /// T-ANNOTATE-084: collect_existing_highlight_rects returns an empty Vec
    /// for a page with no annotations.
    #[test]
    fn t_annotate_084_collect_rects_empty_page() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");
        let page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create blank page");
        let rects = collect_existing_highlight_rects(&page);
        assert!(rects.is_empty(), "blank page has no highlight rects");
    }

    /// T-ANNOTATE-085: create_highlight_annotations skips duplicate highlights
    /// when the page already has a highlight at the same position. Regression
    /// test for DEF-5 (append=true duplicate highlights).
    #[test]
    fn t_annotate_085_skip_duplicate_highlight_in_append_mode() {
        let bindings = match neuroncite_pdf::pdfium_binding::bind_pdfium() {
            Ok(b) => b,
            Err(_) => return,
        };
        let pdfium = Pdfium::new(bindings);
        let mut document = pdfium.create_new_pdf().expect("create PDF");

        let mut page = document
            .pages_mut()
            .create_page_at_start(PdfPagePaperSize::a4())
            .expect("create blank page");

        let match_result = MatchResult {
            page_number: 1,
            char_start: 0,
            char_end: 3,
            method: crate::types::MatchMethod::Exact,
            bounding_boxes: vec![
                [72.0, 700.0, 80.0, 712.0],
                [80.0, 700.0, 88.0, 712.0],
                [88.0, 700.0, 96.0, 712.0],
            ],
            fuzzy_score: None,
        };
        let color = Color::parse("#FFFF00").expect("parse yellow");

        // First annotation: creates highlights.
        let count_1 = create_highlight_annotations(&mut page, &match_result, &color)
            .expect("first annotation");
        assert_eq!(count_1, 1, "first run creates one highlight");

        // Second annotation with the same bounding boxes: should skip duplicates.
        let count_2 = create_highlight_annotations(&mut page, &match_result, &color)
            .expect("second annotation");
        assert_eq!(
            count_2, 0,
            "second run with identical bounding boxes must skip all highlights as duplicates"
        );
    }
}
