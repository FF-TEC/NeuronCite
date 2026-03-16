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

// Post-processing module that injects PDF-compliant appearance streams (/AP)
// into highlight annotations that were created by pdfium-render.
//
// pdfium-render's create_highlight_annotation() creates annotations with
// /Rect, /QuadPoints, /C, and /IC dictionary entries, but does NOT generate
// an /AP (Appearance Stream) entry. Many PDF viewers (Adobe Acrobat, Okular,
// Evince) require the /AP /N (Normal Appearance) Form XObject to render the
// highlight color overlay on the page. Without /AP, the annotation exists as
// a PDF object (with working comments and selectable metadata) but the
// colored overlay is invisible.
//
// This module opens a saved PDF with lopdf (pure Rust PDF library), finds
// all highlight annotations that lack an /AP entry, reads their /C color
// array and /Rect bounding box, generates a Form XObject appearance stream
// with the correct fill color and Multiply blend mode, and writes the
// modified PDF back to disk.
//
// The appearance stream content is a minimal PDF graphics program:
//   q /GS0 gs R G B rg 0 0 W H re f Q
//
// Where:
//   q / Q     = save / restore graphics state
//   /GS0 gs   = apply ExtGState with /BM /Multiply and /ca 0.5
//   R G B rg  = set non-stroking fill color (normalized 0.0..1.0)
//   re        = append rectangle to path
//   f         = fill path
//
// The Form XObject dictionary contains:
//   /Type /XObject
//   /Subtype /Form
//   /BBox [0 0 width height]
//   /Resources << /ExtGState << /GS0 << /Type /ExtGState /BM /Multiply /ca 0.5 >> >> >>

use std::path::Path;

use crate::error::AnnotateError;

/// Opens a saved PDF, injects Normal Appearance (/AP /N) Form XObjects into
/// all highlight annotations that lack an /AP entry, and overwrites the file.
///
/// This function is idempotent: annotations that already have an /AP entry
/// are skipped. Annotations that are not of subtype /Highlight are skipped.
///
/// Returns the number of annotations that received a newly generated
/// appearance stream.
pub fn inject_appearance_streams(pdf_path: &Path) -> Result<usize, AnnotateError> {
    let mut doc = lopdf::Document::load(pdf_path).map_err(|e| {
        AnnotateError::Pdfium(format!("lopdf failed to load {}: {e}", pdf_path.display()))
    })?;

    // pdfium-render serializes annotations as inline Dictionary objects
    // inside the /Annots array (direct objects), while the rest of this
    // module expects annotations as indirect object References. This
    // pre-processing step promotes any inline annotation dictionaries to
    // indirect objects so the main loop can address them by ObjectId.
    promote_inline_annotations(&mut doc);

    let mut injected_count = 0_usize;

    // Collect all page object IDs. lopdf's get_pages() returns a BTreeMap
    // of 1-indexed page numbers to object IDs.
    let pages: Vec<lopdf::ObjectId> = doc.get_pages().values().copied().collect();

    for page_id in pages {
        // Resolve the page dictionary and extract the /Annots array.
        let annot_ids = match get_annotation_ids(&doc, page_id) {
            Some(ids) => ids,
            None => continue,
        };

        for annot_id in annot_ids {
            let needs_ap = annotation_needs_appearance(&doc, annot_id);
            if !needs_ap {
                continue;
            }

            // Read the annotation's /Rect and /C entries for the appearance
            // stream rectangle and fill color.
            let (rect, color) = match read_rect_and_color(&doc, annot_id) {
                Some(rc) => rc,
                None => continue,
            };

            // Build the Form XObject appearance stream.
            let ap_stream = build_appearance_xobject(&rect, &color);

            // Add the stream as an indirect object in the PDF.
            let ap_obj_id = doc.add_object(ap_stream);

            // Set /AP << /N <reference to the stream> >> on the annotation.
            if let Ok(annot_obj) = doc.get_object_mut(annot_id)
                && let lopdf::Object::Dictionary(ref mut dict) = *annot_obj
            {
                let ap_dict = lopdf::Dictionary::from_iter(vec![(
                    b"N".to_vec(),
                    lopdf::Object::Reference(ap_obj_id),
                )]);
                dict.set(b"AP".to_vec(), lopdf::Object::Dictionary(ap_dict));
                injected_count += 1;
            }
        }
    }

    if injected_count > 0 {
        doc.save(pdf_path).map_err(|e| AnnotateError::SaveFailed {
            path: pdf_path.display().to_string(),
            reason: format!("lopdf save after AP injection: {e}"),
        })?;

        tracing::info!(
            path = %pdf_path.display(),
            count = injected_count,
            "injected appearance streams into highlight annotations"
        );
    }

    Ok(injected_count)
}

/// Converts inline annotation dictionaries inside /Annots arrays to indirect
/// objects (References). PDF viewers and authoring tools store annotations in
/// two valid formats per ISO 32000:
///
/// 1. Indirect references: `/Annots [5 0 R 6 0 R]` -- lopdf and most PDF
///    libraries use this format.
/// 2. Inline dictionaries: `/Annots [<< /Type /Annot ... >> << ... >>]` --
///    pdfium-render uses this format when saving PDFs via `save_to_file()`.
///
/// The main injection loop in `inject_appearance_streams` addresses
/// annotations by ObjectId, which requires them to exist as indirect objects.
/// This function iterates all pages, finds /Annots arrays with inline
/// Dictionary entries, promotes each to an indirect object via
/// `doc.add_object()`, and replaces the /Annots array with one that contains
/// only References.
fn promote_inline_annotations(doc: &mut lopdf::Document) {
    let pages: Vec<lopdf::ObjectId> = doc.get_pages().values().copied().collect();

    for page_id in pages {
        let annots_array = match extract_annots_clone(doc, page_id) {
            Some(arr) => arr,
            None => continue,
        };

        // Skip pages where all annotation entries are already References.
        let has_inline = annots_array
            .iter()
            .any(|item| matches!(item, lopdf::Object::Dictionary(_)));
        if !has_inline {
            continue;
        }

        // Build a replacement array: inline dicts become indirect objects,
        // existing References pass through unchanged.
        let mut promoted = Vec::with_capacity(annots_array.len());
        let mut promoted_count = 0_usize;
        for item in annots_array {
            match item {
                lopdf::Object::Dictionary(dict) => {
                    let obj_id = doc.add_object(lopdf::Object::Dictionary(dict));
                    promoted.push(lopdf::Object::Reference(obj_id));
                    promoted_count += 1;
                }
                other => promoted.push(other),
            }
        }

        // Replace the page's /Annots entry with the promoted array.
        if let Ok(page_obj) = doc.get_object_mut(page_id)
            && let lopdf::Object::Dictionary(ref mut dict) = *page_obj
        {
            dict.set(b"Annots", lopdf::Object::Array(promoted));
        }

        tracing::trace!(
            page = ?page_id,
            count = promoted_count,
            "promoted inline annotation dictionaries to indirect objects"
        );
    }
}

/// Extracts the /Annots array from a page dictionary, cloning the contents
/// to release the immutable borrow on the Document. Handles both direct
/// arrays and indirect References to arrays.
fn extract_annots_clone(
    doc: &lopdf::Document,
    page_id: lopdf::ObjectId,
) -> Option<Vec<lopdf::Object>> {
    let page_dict = doc.get_object(page_id).ok()?.as_dict().ok()?;
    let annots_obj = page_dict.get(b"Annots").ok()?;
    match annots_obj {
        lopdf::Object::Array(arr) => Some(arr.clone()),
        lopdf::Object::Reference(r) => doc.get_object(*r).ok()?.as_array().ok().cloned(),
        _ => None,
    }
}

/// Extracts annotation object IDs from a page's /Annots array.
/// Returns None if the page has no /Annots entry or it cannot be resolved.
fn get_annotation_ids(
    doc: &lopdf::Document,
    page_id: lopdf::ObjectId,
) -> Option<Vec<lopdf::ObjectId>> {
    let page_dict = doc.get_object(page_id).ok()?.as_dict().ok()?;

    // /Annots can be a direct array or a reference to an indirect array.
    let annots_obj = page_dict.get(b"Annots").ok()?;
    let annots_array = match annots_obj {
        lopdf::Object::Array(arr) => arr.clone(),
        lopdf::Object::Reference(r) => {
            let resolved = doc.get_object(*r).ok()?;
            resolved.as_array().ok()?.clone()
        }
        _ => return None,
    };

    let mut ids = Vec::with_capacity(annots_array.len());
    for item in &annots_array {
        if let lopdf::Object::Reference(id) = item {
            ids.push(*id);
        }
    }

    Some(ids)
}

/// Checks whether an annotation is a /Highlight that lacks an /AP entry.
/// Returns true if the annotation needs an appearance stream injection.
fn annotation_needs_appearance(doc: &lopdf::Document, annot_id: lopdf::ObjectId) -> bool {
    let dict = match doc.get_object(annot_id).ok().and_then(|o| o.as_dict().ok()) {
        Some(d) => d,
        None => return false,
    };

    // Check /Subtype is /Highlight.
    let subtype = dict
        .get(b"Subtype")
        .ok()
        .and_then(|o| o.as_name().ok())
        .map(|n| std::str::from_utf8(n).unwrap_or(""));

    if subtype != Some("Highlight") {
        return false;
    }

    // Check that /AP is absent. If /AP already exists, the annotation
    // already has an appearance stream (from a previous injection or
    // from the original PDF authoring tool).
    !dict.has(b"AP")
}

/// Reads the /Rect and /C color array from a highlight annotation.
/// Returns the bounding rectangle as [left, bottom, right, top] and
/// the RGB color as [r, g, b] normalized to 0.0..1.0.
///
/// Falls back to /IC (interior color) if /C is absent. If both /C and
/// /IC are absent, uses yellow (1.0, 1.0, 0.0) as the default color.
fn read_rect_and_color(
    doc: &lopdf::Document,
    annot_id: lopdf::ObjectId,
) -> Option<([f32; 4], [f32; 3])> {
    let dict = doc.get_object(annot_id).ok()?.as_dict().ok()?;

    // Parse /Rect [left, bottom, right, top].
    let rect_arr = dict.get(b"Rect").ok()?.as_array().ok()?;
    if rect_arr.len() != 4 {
        return None;
    }
    let rect = [
        object_to_f32(&rect_arr[0])?,
        object_to_f32(&rect_arr[1])?,
        object_to_f32(&rect_arr[2])?,
        object_to_f32(&rect_arr[3])?,
    ];

    // Parse /C (stroke color) or /IC (interior color) as RGB array.
    // PDF highlight annotations store color in /C as an array of 3 floats
    // in the range 0.0..1.0. Some authoring tools write /IC instead.
    let color = parse_color_array(dict, b"C")
        .or_else(|| parse_color_array(dict, b"IC"))
        .unwrap_or([1.0, 1.0, 0.0]); // default: yellow

    Some((rect, color))
}

/// Parses an RGB color array from a dictionary entry. Returns None if the
/// key is absent, not an array, or has fewer than 3 elements.
fn parse_color_array(dict: &lopdf::Dictionary, key: &[u8]) -> Option<[f32; 3]> {
    let arr = dict.get(key).ok()?.as_array().ok()?;
    if arr.len() < 3 {
        return None;
    }
    Some([
        object_to_f32(&arr[0])?,
        object_to_f32(&arr[1])?,
        object_to_f32(&arr[2])?,
    ])
}

/// Converts a lopdf Object (Integer or Real) to f32.
fn object_to_f32(obj: &lopdf::Object) -> Option<f32> {
    match obj {
        lopdf::Object::Integer(i) => Some(*i as f32),
        lopdf::Object::Real(f) => Some(*f),
        _ => None,
    }
}

/// Builds a Form XObject stream that renders a colored, semi-transparent
/// rectangle covering the annotation's bounding box. The rectangle uses
/// Multiply blend mode (/BM /Multiply) with 50% opacity (/ca 0.5) to
/// simulate a physical highlighter pen overlay.
///
/// The Form XObject's /BBox is set to [0, 0, width, height] in the
/// annotation's local coordinate space. The appearance stream content
/// fills a rectangle covering the full bounding box.
fn build_appearance_xobject(rect: &[f32; 4], color: &[f32; 3]) -> lopdf::Stream {
    let width = (rect[2] - rect[0]).abs();
    let height = (rect[3] - rect[1]).abs();

    // Build the content stream: apply graphics state, set color, draw rect.
    let content = format!(
        "q /GS0 gs {r:.4} {g:.4} {b:.4} rg 0 0 {w:.2} {h:.2} re f Q",
        r = color[0],
        g = color[1],
        b = color[2],
        w = width,
        h = height,
    );

    let content_bytes = content.into_bytes();

    // ExtGState dictionary for Multiply blend mode and 50% opacity.
    let gs0 = lopdf::Dictionary::from_iter(vec![
        (b"Type".to_vec(), lopdf::Object::Name(b"ExtGState".to_vec())),
        (b"BM".to_vec(), lopdf::Object::Name(b"Multiply".to_vec())),
        (b"ca".to_vec(), lopdf::Object::Real(0.5)),
    ]);

    let ext_gstate_dict =
        lopdf::Dictionary::from_iter(vec![(b"GS0".to_vec(), lopdf::Object::Dictionary(gs0))]);

    let resources = lopdf::Dictionary::from_iter(vec![(
        b"ExtGState".to_vec(),
        lopdf::Object::Dictionary(ext_gstate_dict),
    )]);

    // Form XObject dictionary.
    let bbox = lopdf::Object::Array(vec![
        lopdf::Object::Real(0.0),
        lopdf::Object::Real(0.0),
        lopdf::Object::Real(width),
        lopdf::Object::Real(height),
    ]);

    let mut stream_dict = lopdf::Dictionary::from_iter(vec![
        (b"Type".to_vec(), lopdf::Object::Name(b"XObject".to_vec())),
        (b"Subtype".to_vec(), lopdf::Object::Name(b"Form".to_vec())),
        (b"BBox".to_vec(), bbox),
        (b"Resources".to_vec(), lopdf::Object::Dictionary(resources)),
    ]);

    // The /Length entry is automatically handled by lopdf when writing.
    stream_dict.set(
        b"Length".to_vec(),
        lopdf::Object::Integer(content_bytes.len() as i64),
    );

    lopdf::Stream::new(stream_dict, content_bytes)
}

/// Merges highlight annotations from a prior output PDF into a newly annotated
/// PDF. Used in append mode where the pipeline reads original source PDFs for
/// text extraction (to avoid pdfium re-linearization issues) and then copies
/// existing annotations from the previous output.
///
/// The function opens both PDFs with lopdf, iterates all pages in the new PDF,
/// finds matching pages in the prior PDF (by page index), and copies highlight
/// annotations that do not already exist in the new PDF. Deduplication uses
/// 90% bounding-box area overlap (identical to the deduplication in
/// `annotate::create_highlight_annotations`).
///
/// Each copied annotation includes its full dictionary (/Rect, /QuadPoints,
/// /C, /IC, /Contents, /Subtype, /Type) and its /AP entry (appearance stream).
/// The /AP /N Form XObject is deep-copied as a new indirect object in the
/// destination document so cross-document references are avoided.
///
/// Returns the total number of annotations merged across all pages.
pub fn merge_prior_annotations(
    new_pdf_path: &Path,
    prior_pdf_path: &Path,
) -> Result<usize, AnnotateError> {
    // Load both documents. The prior PDF may not exist if this is the first
    // append run for a PDF that had no matches in the prior run (the prior
    // pipeline copied the source PDF unchanged to the output directory).
    let prior_doc = lopdf::Document::load(prior_pdf_path).map_err(|e| {
        AnnotateError::Pdfium(format!(
            "lopdf failed to load prior PDF {}: {e}",
            prior_pdf_path.display()
        ))
    })?;

    let mut new_doc = lopdf::Document::load(new_pdf_path).map_err(|e| {
        AnnotateError::Pdfium(format!(
            "lopdf failed to load new PDF {}: {e}",
            new_pdf_path.display()
        ))
    })?;

    // Ensure inline annotations in the new document are promoted to indirect
    // objects so the deduplication logic can address them by ObjectId.
    promote_inline_annotations(&mut new_doc);

    let prior_pages: Vec<(u32, lopdf::ObjectId)> = prior_doc.get_pages().into_iter().collect();
    let new_pages: Vec<(u32, lopdf::ObjectId)> = new_doc.get_pages().into_iter().collect();

    let mut total_merged = 0_usize;

    for &(page_num, new_page_id) in &new_pages {
        // Find the corresponding page in the prior document by page number.
        let prior_page_id = match prior_pages.iter().find(|(pn, _)| *pn == page_num) {
            Some((_, id)) => *id,
            None => continue,
        };

        // Collect existing highlight rects from the new PDF for deduplication.
        let existing_rects = collect_highlight_rects(&new_doc, new_page_id);

        // Get highlight annotations from the prior PDF on this page.
        let prior_annot_ids = match get_annotation_ids(&prior_doc, prior_page_id) {
            Some(ids) => ids,
            None => continue,
        };

        let mut page_merged = Vec::new();

        for prior_annot_id in prior_annot_ids {
            // Only merge highlight annotations (skip text notes, links, etc.).
            let prior_dict = match prior_doc
                .get_object(prior_annot_id)
                .ok()
                .and_then(|o| o.as_dict().ok())
            {
                Some(d) => d,
                None => continue,
            };

            let subtype = prior_dict
                .get(b"Subtype")
                .ok()
                .and_then(|o| o.as_name().ok())
                .map(|n| std::str::from_utf8(n).unwrap_or(""));

            if subtype != Some("Highlight") {
                continue;
            }

            // Read the annotation's /Rect for deduplication.
            let prior_rect = match prior_dict.get(b"Rect").ok().and_then(|o| o.as_array().ok()) {
                Some(arr) if arr.len() == 4 => {
                    let vals: Option<[f32; 4]> = (|| {
                        Some([
                            object_to_f32(&arr[0])?,
                            object_to_f32(&arr[1])?,
                            object_to_f32(&arr[2])?,
                            object_to_f32(&arr[3])?,
                        ])
                    })();
                    match vals {
                        Some(r) => r,
                        None => continue,
                    }
                }
                _ => continue,
            };

            // Check for 90% bounding-box area overlap with existing annotations.
            let is_duplicate = existing_rects
                .iter()
                .any(|existing| rects_overlap_90_percent(existing, &prior_rect));

            if is_duplicate {
                continue;
            }

            // Deep-copy the annotation dictionary to the new document.
            // Clone the dictionary and resolve any /AP references.
            let mut new_annot_dict = prior_dict.clone();

            // If the prior annotation has an /AP /N entry, deep-copy the
            // referenced Form XObject stream to the new document.
            if let Ok(ap_obj) = prior_dict.get(b"AP")
                && let Ok(ap_dict) = ap_obj.as_dict()
            {
                let mut new_ap_dict = lopdf::Dictionary::new();
                if let Ok(lopdf::Object::Reference(n_ref)) = ap_dict.get(b"N") {
                    // Copy the /N stream object to the new document.
                    if let Ok(stream_obj) = prior_doc.get_object(*n_ref) {
                        let new_stream_id = new_doc.add_object(stream_obj.clone());
                        new_ap_dict.set(b"N", lopdf::Object::Reference(new_stream_id));
                    }
                }
                if !new_ap_dict.is_empty() {
                    new_annot_dict.set(b"AP", lopdf::Object::Dictionary(new_ap_dict));
                }
            }

            // Add the copied annotation as an indirect object in the new doc.
            let new_annot_id = new_doc.add_object(lopdf::Object::Dictionary(new_annot_dict));
            page_merged.push(new_annot_id);
            total_merged += 1;
        }

        // Append the merged annotation references to the page's /Annots array.
        if !page_merged.is_empty() {
            append_annots_to_page(&mut new_doc, new_page_id, &page_merged);
        }
    }

    if total_merged > 0 {
        new_doc
            .save(new_pdf_path)
            .map_err(|e| AnnotateError::SaveFailed {
                path: new_pdf_path.display().to_string(),
                reason: format!("lopdf save after annotation merge: {e}"),
            })?;

        tracing::info!(
            path = %new_pdf_path.display(),
            count = total_merged,
            "merged prior highlight annotations into annotated PDF"
        );
    }

    Ok(total_merged)
}

/// Collects the /Rect bounding boxes of all highlight annotations on a page.
/// Used for deduplication when merging prior annotations.
fn collect_highlight_rects(doc: &lopdf::Document, page_id: lopdf::ObjectId) -> Vec<[f32; 4]> {
    let annot_ids = match get_annotation_ids(doc, page_id) {
        Some(ids) => ids,
        None => return Vec::new(),
    };

    let mut rects = Vec::new();
    for annot_id in annot_ids {
        let dict = match doc.get_object(annot_id).ok().and_then(|o| o.as_dict().ok()) {
            Some(d) => d,
            None => continue,
        };

        let subtype = dict
            .get(b"Subtype")
            .ok()
            .and_then(|o| o.as_name().ok())
            .map(|n| std::str::from_utf8(n).unwrap_or(""));

        if subtype != Some("Highlight") {
            continue;
        }

        if let Some(arr) = dict.get(b"Rect").ok().and_then(|o| o.as_array().ok())
            && arr.len() == 4
            && let (Some(a), Some(b), Some(c), Some(d)) = (
                object_to_f32(&arr[0]),
                object_to_f32(&arr[1]),
                object_to_f32(&arr[2]),
                object_to_f32(&arr[3]),
            )
        {
            rects.push([a, b, c, d]);
        }
    }

    rects
}

/// Checks whether two rectangles have >= 90% area overlap, using the same
/// algorithm as `annotate::is_duplicate_highlight`. Two rects are considered
/// duplicates if the intersection area divided by the smaller rect's area
/// is >= 0.90.
fn rects_overlap_90_percent(a: &[f32; 4], b: &[f32; 4]) -> bool {
    let inter_left = a[0].max(b[0]);
    let inter_bottom = a[1].max(b[1]);
    let inter_right = a[2].min(b[2]);
    let inter_top = a[3].min(b[3]);

    let inter_width = (inter_right - inter_left).max(0.0);
    let inter_height = (inter_top - inter_bottom).max(0.0);
    let inter_area = inter_width * inter_height;

    let area_a = (a[2] - a[0]).abs() * (a[3] - a[1]).abs();
    let area_b = (b[2] - b[0]).abs() * (b[3] - b[1]).abs();
    let min_area = area_a.min(area_b);

    if min_area < f32::EPSILON {
        return false;
    }

    inter_area / min_area >= 0.90
}

/// Appends annotation References to a page's /Annots array. Creates the
/// /Annots array if it does not exist.
fn append_annots_to_page(
    doc: &mut lopdf::Document,
    page_id: lopdf::ObjectId,
    annot_ids: &[lopdf::ObjectId],
) {
    // Read the existing /Annots array (cloned to release the borrow).
    let existing = extract_annots_clone(doc, page_id).unwrap_or_default();

    let mut combined = existing;
    for &aid in annot_ids {
        combined.push(lopdf::Object::Reference(aid));
    }

    if let Ok(page_obj) = doc.get_object_mut(page_id)
        && let lopdf::Object::Dictionary(ref mut dict) = *page_obj
    {
        dict.set(b"Annots", lopdf::Object::Array(combined));
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// T-APPEAR-001: build_appearance_xobject generates a valid Form XObject
    /// stream with the correct BBox, color, and graphics state.
    #[test]
    fn t_appear_001_xobject_stream_structure() {
        let rect = [72.0, 700.0, 200.0, 712.0];
        let color = [1.0, 0.5, 0.0]; // orange

        let stream = build_appearance_xobject(&rect, &color);

        // Verify dictionary entries.
        let dict = &stream.dict;
        assert_eq!(dict.get(b"Type").unwrap().as_name().unwrap(), b"XObject");
        assert_eq!(dict.get(b"Subtype").unwrap().as_name().unwrap(), b"Form");

        // BBox should be [0, 0, 128, 12] (width=200-72=128, height=712-700=12).
        let bbox = dict.get(b"BBox").unwrap().as_array().unwrap();
        assert_eq!(bbox.len(), 4);

        // Content stream should contain the color and rectangle operators.
        let content = std::str::from_utf8(&stream.content).unwrap();
        assert!(
            content.contains("rg"),
            "stream must contain 'rg' color operator"
        );
        assert!(
            content.contains("re f"),
            "stream must contain 're f' rectangle fill"
        );
        assert!(content.contains("/GS0 gs"), "stream must apply ExtGState");
        assert!(
            content.starts_with("q "),
            "stream must start with graphics state save"
        );
        assert!(
            content.ends_with(" Q"),
            "stream must end with graphics state restore"
        );
    }

    /// T-APPEAR-002: parse_color_array correctly extracts RGB from a PDF
    /// color array with Real values.
    #[test]
    fn t_appear_002_parse_color_array_real() {
        let dict = lopdf::Dictionary::from_iter(vec![(
            b"C".to_vec(),
            lopdf::Object::Array(vec![
                lopdf::Object::Real(1.0),
                lopdf::Object::Real(0.5),
                lopdf::Object::Real(0.0),
            ]),
        )]);
        let color = parse_color_array(&dict, b"C").unwrap();
        assert!((color[0] - 1.0).abs() < f32::EPSILON);
        assert!((color[1] - 0.5).abs() < f32::EPSILON);
        assert!((color[2] - 0.0).abs() < f32::EPSILON);
    }

    /// T-APPEAR-003: parse_color_array returns None for missing key.
    #[test]
    fn t_appear_003_parse_color_array_missing() {
        let dict = lopdf::Dictionary::new();
        assert!(parse_color_array(&dict, b"C").is_none());
    }

    /// T-APPEAR-004: parse_color_array handles Integer color values
    /// (some PDF generators use Integer 0/1 instead of Real 0.0/1.0).
    #[test]
    fn t_appear_004_parse_color_array_integer() {
        let dict = lopdf::Dictionary::from_iter(vec![(
            b"C".to_vec(),
            lopdf::Object::Array(vec![
                lopdf::Object::Integer(1),
                lopdf::Object::Integer(0),
                lopdf::Object::Integer(0),
            ]),
        )]);
        let color = parse_color_array(&dict, b"C").unwrap();
        assert!((color[0] - 1.0).abs() < f32::EPSILON);
        assert!((color[1] - 0.0).abs() < f32::EPSILON);
    }

    /// T-APPEAR-005: annotation_needs_appearance returns true for a Highlight
    /// annotation dictionary without /AP.
    #[test]
    fn t_appear_005_needs_appearance_highlight_no_ap() {
        let mut doc = lopdf::Document::new();
        let dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
            (
                b"Subtype".to_vec(),
                lopdf::Object::Name(b"Highlight".to_vec()),
            ),
        ]);
        let id = doc.add_object(lopdf::Object::Dictionary(dict));
        assert!(annotation_needs_appearance(&doc, id));
    }

    /// T-APPEAR-006: annotation_needs_appearance returns false for a Highlight
    /// annotation that already has /AP.
    #[test]
    fn t_appear_006_needs_appearance_highlight_with_ap() {
        let mut doc = lopdf::Document::new();
        let ap_dict = lopdf::Dictionary::from_iter(vec![(b"N".to_vec(), lopdf::Object::Null)]);
        let dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
            (
                b"Subtype".to_vec(),
                lopdf::Object::Name(b"Highlight".to_vec()),
            ),
            (b"AP".to_vec(), lopdf::Object::Dictionary(ap_dict)),
        ]);
        let id = doc.add_object(lopdf::Object::Dictionary(dict));
        assert!(!annotation_needs_appearance(&doc, id));
    }

    /// T-APPEAR-007: annotation_needs_appearance returns false for non-Highlight
    /// annotations (e.g. /Text annotations).
    #[test]
    fn t_appear_007_needs_appearance_text_annotation() {
        let mut doc = lopdf::Document::new();
        let dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
            (b"Subtype".to_vec(), lopdf::Object::Name(b"Text".to_vec())),
        ]);
        let id = doc.add_object(lopdf::Object::Dictionary(dict));
        assert!(!annotation_needs_appearance(&doc, id));
    }

    /// T-APPEAR-008: inject_appearance_streams round-trip test.
    /// Creates a minimal valid PDF with a highlight annotation lacking /AP,
    /// runs the injection, and verifies the /AP entry was added.
    #[test]
    fn t_appear_008_inject_roundtrip() {
        // Build a minimal PDF with one page and one highlight annotation.
        let mut doc = lopdf::Document::new();
        doc.version = "1.7".to_string();

        // Create the annotation dictionary (Highlight, no /AP).
        let annot_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
            (
                b"Subtype".to_vec(),
                lopdf::Object::Name(b"Highlight".to_vec()),
            ),
            (
                b"Rect".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(72.0),
                    lopdf::Object::Real(700.0),
                    lopdf::Object::Real(200.0),
                    lopdf::Object::Real(712.0),
                ]),
            ),
            (
                b"C".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(1.0),
                    lopdf::Object::Real(0.5),
                    lopdf::Object::Real(0.0),
                ]),
            ),
        ]);
        let annot_id = doc.add_object(lopdf::Object::Dictionary(annot_dict));

        // Create a minimal page dictionary with the annotation in /Annots.
        let page_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
            (
                b"Annots".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(annot_id)]),
            ),
        ]);
        let page_id = doc.add_object(lopdf::Object::Dictionary(page_dict));

        // Create the Pages node.
        let pages_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let pages_id = doc.add_object(lopdf::Object::Dictionary(pages_dict));

        // Set /Parent on the page.
        if let Ok(lopdf::Object::Dictionary(d)) = doc.get_object_mut(page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(pages_id));
        }

        // Create the catalog.
        let catalog_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(pages_id)),
        ]);
        let catalog_id = doc.add_object(lopdf::Object::Dictionary(catalog_dict));

        doc.trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(catalog_id));

        // Save the PDF to a temp file.
        let mut tmp = tempfile::NamedTempFile::new().expect("create temp file");
        doc.save_to(&mut tmp).expect("save test PDF");
        tmp.flush().expect("flush");

        let path = tmp.path().to_path_buf();

        // Run the appearance stream injection.
        let count = inject_appearance_streams(&path).expect("injection must succeed");
        assert_eq!(
            count, 1,
            "one annotation should receive an appearance stream"
        );

        // Reload and verify /AP is present.
        let doc2 = lopdf::Document::load(&path).expect("reload test PDF");
        let annot = doc2.get_object(annot_id).expect("annotation object");
        let annot_dict = annot.as_dict().expect("annotation dictionary");

        assert!(
            annot_dict.has(b"AP"),
            "annotation must have /AP after injection"
        );

        let ap = annot_dict
            .get(b"AP")
            .expect("/AP entry")
            .as_dict()
            .expect("/AP is a dictionary");

        assert!(ap.has(b"N"), "/AP must have /N (Normal Appearance) entry");

        // Verify the /N reference points to a valid Form XObject.
        if let Ok(lopdf::Object::Reference(n_ref)) = ap.get(b"N") {
            let stream_obj = doc2.get_object(*n_ref).expect("/N reference target");
            if let lopdf::Object::Stream(ref stream) = *stream_obj {
                let sd = &stream.dict;
                assert_eq!(sd.get(b"Type").unwrap().as_name().unwrap(), b"XObject");
                assert_eq!(sd.get(b"Subtype").unwrap().as_name().unwrap(), b"Form");
                assert!(sd.has(b"BBox"), "Form XObject must have /BBox");
                assert!(sd.has(b"Resources"), "Form XObject must have /Resources");
            } else {
                panic!("/N target must be a Stream object");
            }
        } else {
            panic!("/AP /N must be a Reference");
        }
    }

    /// T-APPEAR-009: inject_appearance_streams is idempotent -- calling it
    /// twice does not add a second /AP or modify the existing one.
    #[test]
    fn t_appear_009_idempotent() {
        let mut doc = lopdf::Document::new();
        doc.version = "1.7".to_string();

        let annot_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
            (
                b"Subtype".to_vec(),
                lopdf::Object::Name(b"Highlight".to_vec()),
            ),
            (
                b"Rect".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(72.0),
                    lopdf::Object::Real(700.0),
                    lopdf::Object::Real(200.0),
                    lopdf::Object::Real(712.0),
                ]),
            ),
            (
                b"C".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(1.0),
                    lopdf::Object::Real(0.0),
                    lopdf::Object::Real(0.0),
                ]),
            ),
        ]);
        let annot_id = doc.add_object(lopdf::Object::Dictionary(annot_dict));

        let page_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
            (
                b"Annots".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(annot_id)]),
            ),
        ]);
        let page_id = doc.add_object(lopdf::Object::Dictionary(page_dict));

        let pages_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let pages_id = doc.add_object(lopdf::Object::Dictionary(pages_dict));

        if let Ok(lopdf::Object::Dictionary(d)) = doc.get_object_mut(page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(pages_id));
        }

        let catalog_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(pages_id)),
        ]);
        let catalog_id = doc.add_object(lopdf::Object::Dictionary(catalog_dict));

        doc.trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(catalog_id));

        let mut tmp = tempfile::NamedTempFile::new().expect("create temp file");
        doc.save_to(&mut tmp).expect("save test PDF");
        tmp.flush().expect("flush");

        let path = tmp.path().to_path_buf();

        // First injection.
        let count1 = inject_appearance_streams(&path).expect("first injection");
        assert_eq!(count1, 1);

        // Second injection (should be idempotent).
        let count2 = inject_appearance_streams(&path).expect("second injection");
        assert_eq!(
            count2, 0,
            "second injection must not modify any annotations"
        );
    }

    /// T-APPEAR-010: read_rect_and_color falls back to /IC when /C is absent.
    #[test]
    fn t_appear_010_color_fallback_to_ic() {
        let mut doc = lopdf::Document::new();
        let dict = lopdf::Dictionary::from_iter(vec![
            (
                b"Rect".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(10.0),
                    lopdf::Object::Real(20.0),
                    lopdf::Object::Real(100.0),
                    lopdf::Object::Real(50.0),
                ]),
            ),
            (
                b"IC".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(0.0),
                    lopdf::Object::Real(1.0),
                    lopdf::Object::Real(0.0),
                ]),
            ),
        ]);
        let id = doc.add_object(lopdf::Object::Dictionary(dict));
        let (rect, color) = read_rect_and_color(&doc, id).unwrap();
        assert!((rect[0] - 10.0).abs() < f32::EPSILON);
        assert!(
            (color[1] - 1.0).abs() < f32::EPSILON,
            "green channel from /IC"
        );
    }

    /// T-APPEAR-011: read_rect_and_color uses yellow default when both /C
    /// and /IC are absent.
    #[test]
    fn t_appear_011_color_default_yellow() {
        let mut doc = lopdf::Document::new();
        let dict = lopdf::Dictionary::from_iter(vec![(
            b"Rect".to_vec(),
            lopdf::Object::Array(vec![
                lopdf::Object::Real(10.0),
                lopdf::Object::Real(20.0),
                lopdf::Object::Real(100.0),
                lopdf::Object::Real(50.0),
            ]),
        )]);
        let id = doc.add_object(lopdf::Object::Dictionary(dict));
        let (_rect, color) = read_rect_and_color(&doc, id).unwrap();
        assert!((color[0] - 1.0).abs() < f32::EPSILON, "default red=1.0");
        assert!((color[1] - 1.0).abs() < f32::EPSILON, "default green=1.0");
        assert!((color[2] - 0.0).abs() < f32::EPSILON, "default blue=0.0");
    }

    /// T-APPEAR-012: promote_inline_annotations converts inline Dictionary
    /// objects inside /Annots arrays to indirect References. Simulates the
    /// serialization format that pdfium-render produces when saving PDFs.
    #[test]
    fn t_appear_012_promote_inline_annotations() {
        let mut doc = lopdf::Document::new();
        doc.version = "1.7".to_string();

        // Build an inline highlight annotation dictionary (as pdfium writes it).
        let inline_annot = lopdf::Object::Dictionary(lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
            (
                b"Subtype".to_vec(),
                lopdf::Object::Name(b"Highlight".to_vec()),
            ),
            (
                b"Rect".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(72.0),
                    lopdf::Object::Real(700.0),
                    lopdf::Object::Real(200.0),
                    lopdf::Object::Real(712.0),
                ]),
            ),
            (
                b"C".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(1.0),
                    lopdf::Object::Real(0.0),
                    lopdf::Object::Real(0.0),
                ]),
            ),
        ]));

        // Create a page with the inline annotation directly in the /Annots array
        // (not as an indirect reference).
        let page_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
            (b"Annots".to_vec(), lopdf::Object::Array(vec![inline_annot])),
        ]);
        let page_id = doc.add_object(lopdf::Object::Dictionary(page_dict));

        let pages_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let pages_id = doc.add_object(lopdf::Object::Dictionary(pages_dict));

        #[allow(clippy::collapsible_match)]
        if let Ok(lopdf::Object::Dictionary(d)) = doc.get_object_mut(page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(pages_id));
        }

        let catalog_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(pages_id)),
        ]);
        let catalog_id = doc.add_object(lopdf::Object::Dictionary(catalog_dict));
        doc.trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(catalog_id));

        // Before promotion: get_annotation_ids returns empty because the
        // annotation is an inline Dictionary, not a Reference.
        let before = get_annotation_ids(&doc, page_id).unwrap_or_default();
        assert!(
            before.is_empty(),
            "inline annotations must not appear as references before promotion"
        );

        // Run the promotion.
        promote_inline_annotations(&mut doc);

        // After promotion: the /Annots array contains a Reference, and the
        // annotation dictionary is accessible as an indirect object.
        let after = get_annotation_ids(&doc, page_id).unwrap_or_default();
        assert_eq!(
            after.len(),
            1,
            "one annotation must be present after promotion"
        );

        // Verify the promoted annotation retains its /Subtype and /Rect.
        let annot_dict = doc
            .get_object(after[0])
            .expect("promoted annotation object")
            .as_dict()
            .expect("promoted annotation dictionary");
        let subtype = annot_dict
            .get(b"Subtype")
            .expect("/Subtype")
            .as_name()
            .expect("name");
        assert_eq!(subtype, b"Highlight");
        assert!(annot_dict.has(b"Rect"), "promoted annotation retains /Rect");
        assert!(annot_dict.has(b"C"), "promoted annotation retains /C");
    }

    /// T-APPEAR-013: promote_inline_annotations is a no-op when all
    /// annotations in the /Annots array are already indirect References.
    #[test]
    fn t_appear_013_promote_noop_for_references() {
        let mut doc = lopdf::Document::new();
        doc.version = "1.7".to_string();

        // Create annotation as an indirect object (the standard lopdf way).
        let annot_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
            (
                b"Subtype".to_vec(),
                lopdf::Object::Name(b"Highlight".to_vec()),
            ),
            (
                b"Rect".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(72.0),
                    lopdf::Object::Real(700.0),
                    lopdf::Object::Real(200.0),
                    lopdf::Object::Real(712.0),
                ]),
            ),
        ]);
        let annot_id = doc.add_object(lopdf::Object::Dictionary(annot_dict));

        let page_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
            (
                b"Annots".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(annot_id)]),
            ),
        ]);
        let page_id = doc.add_object(lopdf::Object::Dictionary(page_dict));

        let pages_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let pages_id = doc.add_object(lopdf::Object::Dictionary(pages_dict));

        #[allow(clippy::collapsible_match)]
        if let Ok(lopdf::Object::Dictionary(d)) = doc.get_object_mut(page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(pages_id));
        }

        let catalog_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(pages_id)),
        ]);
        let catalog_id = doc.add_object(lopdf::Object::Dictionary(catalog_dict));
        doc.trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(catalog_id));

        // Before promotion: one annotation reference.
        let before = get_annotation_ids(&doc, page_id).unwrap_or_default();
        assert_eq!(before.len(), 1);
        assert_eq!(before[0], annot_id);

        // Run promotion (should be a no-op).
        promote_inline_annotations(&mut doc);

        // After promotion: the same annotation reference, unchanged.
        let after = get_annotation_ids(&doc, page_id).unwrap_or_default();
        assert_eq!(after.len(), 1);
        assert_eq!(
            after[0], annot_id,
            "annotation ObjectId must be unchanged when already a Reference"
        );
    }

    /// T-APPEAR-014: inject_appearance_streams works end-to-end on a PDF
    /// with inline annotation dictionaries (simulating pdfium output).
    /// Verifies the full pipeline: promote -> detect -> inject -> save.
    #[test]
    fn t_appear_014_inject_with_inline_annotations() {
        let mut doc = lopdf::Document::new();
        doc.version = "1.7".to_string();

        // Inline highlight annotation dictionary (no /AP entry).
        let inline_annot = lopdf::Object::Dictionary(lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
            (
                b"Subtype".to_vec(),
                lopdf::Object::Name(b"Highlight".to_vec()),
            ),
            (
                b"Rect".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(72.0),
                    lopdf::Object::Real(700.0),
                    lopdf::Object::Real(200.0),
                    lopdf::Object::Real(712.0),
                ]),
            ),
            (
                b"C".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(0.0),
                    lopdf::Object::Real(0.5),
                    lopdf::Object::Real(1.0),
                ]),
            ),
        ]));

        let page_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
            (b"Annots".to_vec(), lopdf::Object::Array(vec![inline_annot])),
        ]);
        let page_id = doc.add_object(lopdf::Object::Dictionary(page_dict));

        let pages_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let pages_id = doc.add_object(lopdf::Object::Dictionary(pages_dict));

        #[allow(clippy::collapsible_match)]
        if let Ok(lopdf::Object::Dictionary(d)) = doc.get_object_mut(page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(pages_id));
        }

        let catalog_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(pages_id)),
        ]);
        let catalog_id = doc.add_object(lopdf::Object::Dictionary(catalog_dict));
        doc.trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(catalog_id));

        // Save to a temp file.
        let mut tmp = tempfile::NamedTempFile::new().expect("create temp file");
        doc.save_to(&mut tmp).expect("save test PDF");
        tmp.flush().expect("flush");

        let path = tmp.path().to_path_buf();

        // Run the full injection pipeline (includes inline promotion).
        let count = inject_appearance_streams(&path).expect("injection must succeed");
        assert_eq!(
            count, 1,
            "one inline annotation should receive an appearance stream"
        );

        // Verify /AP was added to the promoted annotation.
        let doc2 = lopdf::Document::load(&path).expect("reload test PDF");
        let pages: Vec<lopdf::ObjectId> = doc2.get_pages().values().copied().collect();
        let page_dict = doc2
            .get_object(pages[0])
            .expect("page object")
            .as_dict()
            .expect("page dict");
        let annots = page_dict
            .get(b"Annots")
            .expect("/Annots")
            .as_array()
            .expect("annots array");

        // After injection, the annotation should be a Reference (promoted)
        // with /AP /N present.
        let annot_ref = &annots[0];
        if let lopdf::Object::Reference(id) = annot_ref {
            let annot_dict = doc2
                .get_object(*id)
                .expect("annotation object")
                .as_dict()
                .expect("annotation dict");
            assert!(
                annot_dict.has(b"AP"),
                "promoted annotation must have /AP after injection"
            );
        } else {
            panic!("annotation must be a Reference after promotion and save");
        }
    }

    /// T-APPEAR-015: promote_inline_annotations handles mixed arrays
    /// containing both inline Dictionary annotations and indirect References.
    #[test]
    fn t_appear_015_promote_mixed_array() {
        let mut doc = lopdf::Document::new();
        doc.version = "1.7".to_string();

        // First annotation: indirect Reference (standard lopdf format).
        let ref_annot = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
            (
                b"Subtype".to_vec(),
                lopdf::Object::Name(b"Highlight".to_vec()),
            ),
            (
                b"Rect".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(72.0),
                    lopdf::Object::Real(700.0),
                    lopdf::Object::Real(200.0),
                    lopdf::Object::Real(712.0),
                ]),
            ),
        ]);
        let ref_id = doc.add_object(lopdf::Object::Dictionary(ref_annot));

        // Second annotation: inline Dictionary (pdfium format).
        let inline_annot = lopdf::Object::Dictionary(lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
            (
                b"Subtype".to_vec(),
                lopdf::Object::Name(b"Highlight".to_vec()),
            ),
            (
                b"Rect".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(72.0),
                    lopdf::Object::Real(680.0),
                    lopdf::Object::Real(200.0),
                    lopdf::Object::Real(692.0),
                ]),
            ),
        ]));

        let page_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
            (
                b"Annots".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(ref_id), inline_annot]),
            ),
        ]);
        let page_id = doc.add_object(lopdf::Object::Dictionary(page_dict));

        let pages_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let pages_id = doc.add_object(lopdf::Object::Dictionary(pages_dict));

        #[allow(clippy::collapsible_match)]
        if let Ok(lopdf::Object::Dictionary(d)) = doc.get_object_mut(page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(pages_id));
        }

        let catalog_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(pages_id)),
        ]);
        let catalog_id = doc.add_object(lopdf::Object::Dictionary(catalog_dict));
        doc.trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(catalog_id));

        // Before: only the indirect reference is visible.
        let before = get_annotation_ids(&doc, page_id).unwrap_or_default();
        assert_eq!(
            before.len(),
            1,
            "only indirect reference visible before promotion"
        );

        promote_inline_annotations(&mut doc);

        // After: both annotations are visible as References.
        let after = get_annotation_ids(&doc, page_id).unwrap_or_default();
        assert_eq!(
            after.len(),
            2,
            "both annotations must be visible after promotion"
        );

        // The first entry should still be the original reference.
        assert_eq!(
            after[0], ref_id,
            "original indirect reference must be preserved"
        );
    }

    /// T-APPEAR-016: promote_inline_annotations is a no-op for pages
    /// without any /Annots entry.
    #[test]
    fn t_appear_016_promote_no_annots() {
        let mut doc = lopdf::Document::new();
        doc.version = "1.7".to_string();

        // Page without /Annots.
        let page_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
        ]);
        let page_id = doc.add_object(lopdf::Object::Dictionary(page_dict));

        let pages_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let pages_id = doc.add_object(lopdf::Object::Dictionary(pages_dict));

        #[allow(clippy::collapsible_match)]
        if let Ok(lopdf::Object::Dictionary(d)) = doc.get_object_mut(page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(pages_id));
        }

        let catalog_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(pages_id)),
        ]);
        let catalog_id = doc.add_object(lopdf::Object::Dictionary(catalog_dict));
        doc.trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(catalog_id));

        // Running promotion on a document with no annotations is safe.
        promote_inline_annotations(&mut doc);

        let annots = get_annotation_ids(&doc, page_id);
        assert!(
            annots.is_none(),
            "pages without /Annots remain unchanged after promotion"
        );
    }

    // -----------------------------------------------------------------------
    // Annotation merge tests (NEW-2 regression tests)
    // -----------------------------------------------------------------------

    /// T-APPEAR-017: rects_overlap_90_percent returns true for identical rects.
    #[test]
    fn t_appear_017_identical_rects_overlap() {
        let rect = [72.0, 700.0, 200.0, 712.0];
        assert!(
            rects_overlap_90_percent(&rect, &rect),
            "identical rectangles must have 100% overlap"
        );
    }

    /// T-APPEAR-018: rects_overlap_90_percent returns false for non-overlapping
    /// rects (completely disjoint).
    #[test]
    fn t_appear_018_disjoint_rects_no_overlap() {
        let a = [72.0, 700.0, 200.0, 712.0];
        let b = [300.0, 700.0, 400.0, 712.0]; // far to the right
        assert!(
            !rects_overlap_90_percent(&a, &b),
            "disjoint rectangles must have zero overlap"
        );
    }

    /// T-APPEAR-019: rects_overlap_90_percent returns false for partial overlap
    /// below 90% threshold.
    #[test]
    fn t_appear_019_partial_overlap_below_threshold() {
        let a = [0.0, 0.0, 100.0, 100.0]; // 10000 area
        let b = [50.0, 0.0, 200.0, 100.0]; // overlap: 50*100=5000 / 10000=50%
        assert!(
            !rects_overlap_90_percent(&a, &b),
            "50% overlap must be below the 90% threshold"
        );
    }

    /// T-APPEAR-020: rects_overlap_90_percent returns true for slightly offset
    /// rects with > 90% overlap.
    #[test]
    fn t_appear_020_high_overlap_passes() {
        let a = [0.0, 0.0, 100.0, 100.0]; // 10000 area
        let b = [5.0, 0.0, 105.0, 100.0]; // overlap: 95*100=9500 / 10000=95%
        assert!(
            rects_overlap_90_percent(&a, &b),
            "95% overlap must be above the 90% threshold"
        );
    }

    /// T-APPEAR-021: rects_overlap_90_percent returns false for zero-area rects.
    #[test]
    fn t_appear_021_zero_area_rects() {
        let a = [72.0, 700.0, 72.0, 700.0]; // zero width and height
        let b = [72.0, 700.0, 200.0, 712.0];
        assert!(
            !rects_overlap_90_percent(&a, &b),
            "zero-area rectangle must return false"
        );
    }

    /// T-APPEAR-022: merge_prior_annotations copies a highlight annotation
    /// from a prior PDF into a new PDF. End-to-end test using temporary files.
    #[test]
    fn t_appear_022_merge_prior_annotations_roundtrip() {
        // Build a minimal "prior" PDF with one highlight annotation (with /AP).
        let mut prior = lopdf::Document::new();
        prior.version = "1.7".to_string();

        let prior_annot_dict = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
            (
                b"Subtype".to_vec(),
                lopdf::Object::Name(b"Highlight".to_vec()),
            ),
            (
                b"Rect".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(72.0),
                    lopdf::Object::Real(700.0),
                    lopdf::Object::Real(200.0),
                    lopdf::Object::Real(712.0),
                ]),
            ),
            (
                b"C".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(1.0),
                    lopdf::Object::Real(0.0),
                    lopdf::Object::Real(0.0),
                ]),
            ),
        ]);
        let prior_annot_id = prior.add_object(lopdf::Object::Dictionary(prior_annot_dict));

        let prior_page = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
            (
                b"Annots".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(prior_annot_id)]),
            ),
        ]);
        let prior_page_id = prior.add_object(lopdf::Object::Dictionary(prior_page));

        let prior_pages = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(prior_page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let prior_pages_id = prior.add_object(lopdf::Object::Dictionary(prior_pages));

        if let Ok(lopdf::Object::Dictionary(d)) = prior.get_object_mut(prior_page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(prior_pages_id));
        }

        let prior_catalog = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(prior_pages_id)),
        ]);
        let prior_catalog_id = prior.add_object(lopdf::Object::Dictionary(prior_catalog));
        prior
            .trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(prior_catalog_id));

        // Build a "new" PDF with one page and NO annotations.
        let mut new_doc = lopdf::Document::new();
        new_doc.version = "1.7".to_string();

        let new_page = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
        ]);
        let new_page_id = new_doc.add_object(lopdf::Object::Dictionary(new_page));

        let new_pages = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(new_page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let new_pages_id = new_doc.add_object(lopdf::Object::Dictionary(new_pages));

        if let Ok(lopdf::Object::Dictionary(d)) = new_doc.get_object_mut(new_page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(new_pages_id));
        }

        let new_catalog = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(new_pages_id)),
        ]);
        let new_catalog_id = new_doc.add_object(lopdf::Object::Dictionary(new_catalog));
        new_doc
            .trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(new_catalog_id));

        // Save both to temp files.
        let dir = tempfile::tempdir().expect("create temp dir");
        let prior_path = dir.path().join("prior.pdf");
        let new_path = dir.path().join("new.pdf");

        prior.save(&prior_path).expect("save prior PDF");
        new_doc.save(&new_path).expect("save new PDF");

        // Merge prior annotations into the new PDF.
        let count = merge_prior_annotations(&new_path, &prior_path).expect("merge must succeed");
        assert_eq!(count, 1, "one highlight annotation must be merged");

        // Verify the merged annotation is present in the new PDF.
        let reloaded = lopdf::Document::load(&new_path).expect("reload merged PDF");
        let page_ids: Vec<lopdf::ObjectId> = reloaded.get_pages().values().copied().collect();
        let annot_ids = get_annotation_ids(&reloaded, page_ids[0]).unwrap_or_default();
        assert_eq!(
            annot_ids.len(),
            1,
            "merged PDF must have one annotation on page 1"
        );

        // Verify the merged annotation is a Highlight with the correct /Rect.
        let merged_dict = reloaded
            .get_object(annot_ids[0])
            .expect("merged annotation")
            .as_dict()
            .expect("dict");
        let subtype = merged_dict
            .get(b"Subtype")
            .expect("/Subtype")
            .as_name()
            .expect("name");
        assert_eq!(subtype, b"Highlight");
    }

    /// T-APPEAR-023: merge_prior_annotations deduplicates annotations.
    /// When the new PDF already has a highlight at the same position as the
    /// prior PDF, the prior annotation is not copied (90% overlap).
    #[test]
    fn t_appear_023_merge_deduplicates_existing_annotations() {
        // Build a "prior" PDF with one highlight annotation.
        let rect_arr = || {
            lopdf::Object::Array(vec![
                lopdf::Object::Real(72.0),
                lopdf::Object::Real(700.0),
                lopdf::Object::Real(200.0),
                lopdf::Object::Real(712.0),
            ])
        };

        let annot_dict_fn = || {
            lopdf::Dictionary::from_iter(vec![
                (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
                (
                    b"Subtype".to_vec(),
                    lopdf::Object::Name(b"Highlight".to_vec()),
                ),
                (b"Rect".to_vec(), rect_arr()),
                (
                    b"C".to_vec(),
                    lopdf::Object::Array(vec![
                        lopdf::Object::Real(1.0),
                        lopdf::Object::Real(0.0),
                        lopdf::Object::Real(0.0),
                    ]),
                ),
            ])
        };

        // Prior PDF with one highlight.
        let mut prior = lopdf::Document::new();
        prior.version = "1.7".to_string();
        let prior_annot_id = prior.add_object(lopdf::Object::Dictionary(annot_dict_fn()));
        let prior_page = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
            (
                b"Annots".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(prior_annot_id)]),
            ),
        ]);
        let prior_page_id = prior.add_object(lopdf::Object::Dictionary(prior_page));
        let prior_pages = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(prior_page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let prior_pages_id = prior.add_object(lopdf::Object::Dictionary(prior_pages));
        if let Ok(lopdf::Object::Dictionary(d)) = prior.get_object_mut(prior_page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(prior_pages_id));
        }
        let prior_cat = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(prior_pages_id)),
        ]);
        let prior_cat_id = prior.add_object(lopdf::Object::Dictionary(prior_cat));
        prior
            .trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(prior_cat_id));

        // New PDF with the SAME highlight annotation at the same position.
        let mut new_doc = lopdf::Document::new();
        new_doc.version = "1.7".to_string();
        let new_annot_id = new_doc.add_object(lopdf::Object::Dictionary(annot_dict_fn()));
        let new_page = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
            (
                b"Annots".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(new_annot_id)]),
            ),
        ]);
        let new_page_id = new_doc.add_object(lopdf::Object::Dictionary(new_page));
        let new_pages = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(new_page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let new_pages_id = new_doc.add_object(lopdf::Object::Dictionary(new_pages));
        if let Ok(lopdf::Object::Dictionary(d)) = new_doc.get_object_mut(new_page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(new_pages_id));
        }
        let new_cat = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(new_pages_id)),
        ]);
        let new_cat_id = new_doc.add_object(lopdf::Object::Dictionary(new_cat));
        new_doc
            .trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(new_cat_id));

        // Save both.
        let dir = tempfile::tempdir().expect("create temp dir");
        let prior_path = dir.path().join("prior.pdf");
        let new_path = dir.path().join("new.pdf");
        prior.save(&prior_path).expect("save prior PDF");
        new_doc.save(&new_path).expect("save new PDF");

        // Merge: the prior annotation should be deduplicated (same rect).
        let count = merge_prior_annotations(&new_path, &prior_path).expect("merge must succeed");
        assert_eq!(
            count, 0,
            "duplicate annotation must be skipped during merge"
        );

        // Verify the new PDF still has exactly one annotation.
        let reloaded = lopdf::Document::load(&new_path).expect("reload");
        let page_ids: Vec<lopdf::ObjectId> = reloaded.get_pages().values().copied().collect();
        let annot_ids = get_annotation_ids(&reloaded, page_ids[0]).unwrap_or_default();
        assert_eq!(
            annot_ids.len(),
            1,
            "deduplicated merge must not add a second annotation"
        );
    }

    // -----------------------------------------------------------------------
    // DEF-001 regression tests: Append mode annotation preservation via
    // snapshot-based merge.
    //
    // In append mode, the pipeline overwrites the output PDF with pdfium
    // output (Phase 2), then merges prior annotations from a snapshot
    // directory (Phase 2.5). These tests verify that the merge correctly
    // restores prior annotations from a snapshot file that is separate
    // from the overwritten output file.
    // -----------------------------------------------------------------------

    /// T-APPEAR-DEF001-001: Merge from a snapshot file restores prior
    /// annotations into a new (overwritten) PDF. Simulates the full
    /// DEF-001 scenario: prior PDF has 1 red highlight, new PDF has
    /// 1 blue highlight at a different position, merge adds the red
    /// highlight from the snapshot into the new PDF.
    #[test]
    fn t_appear_def001_001_merge_from_snapshot_restores_annotations() {
        // Helper to build a minimal PDF with one highlight annotation.
        let build_pdf_with_highlight = |color: [f32; 3], rect: [f32; 4]| -> lopdf::Document {
            let mut doc = lopdf::Document::new();
            doc.version = "1.7".to_string();

            let annot = lopdf::Dictionary::from_iter(vec![
                (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
                (
                    b"Subtype".to_vec(),
                    lopdf::Object::Name(b"Highlight".to_vec()),
                ),
                (
                    b"Rect".to_vec(),
                    lopdf::Object::Array(vec![
                        lopdf::Object::Real(rect[0]),
                        lopdf::Object::Real(rect[1]),
                        lopdf::Object::Real(rect[2]),
                        lopdf::Object::Real(rect[3]),
                    ]),
                ),
                (
                    b"C".to_vec(),
                    lopdf::Object::Array(vec![
                        lopdf::Object::Real(color[0]),
                        lopdf::Object::Real(color[1]),
                        lopdf::Object::Real(color[2]),
                    ]),
                ),
            ]);
            let annot_id = doc.add_object(lopdf::Object::Dictionary(annot));

            let page = lopdf::Dictionary::from_iter(vec![
                (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
                (
                    b"MediaBox".to_vec(),
                    lopdf::Object::Array(vec![
                        lopdf::Object::Integer(0),
                        lopdf::Object::Integer(0),
                        lopdf::Object::Integer(612),
                        lopdf::Object::Integer(792),
                    ]),
                ),
                (
                    b"Annots".to_vec(),
                    lopdf::Object::Array(vec![lopdf::Object::Reference(annot_id)]),
                ),
            ]);
            let page_id = doc.add_object(lopdf::Object::Dictionary(page));

            let pages = lopdf::Dictionary::from_iter(vec![
                (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
                (
                    b"Kids".to_vec(),
                    lopdf::Object::Array(vec![lopdf::Object::Reference(page_id)]),
                ),
                (b"Count".to_vec(), lopdf::Object::Integer(1)),
            ]);
            let pages_id = doc.add_object(lopdf::Object::Dictionary(pages));

            if let Ok(lopdf::Object::Dictionary(d)) = doc.get_object_mut(page_id) {
                d.set(b"Parent".to_vec(), lopdf::Object::Reference(pages_id));
            }

            let cat = lopdf::Dictionary::from_iter(vec![
                (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
                (b"Pages".to_vec(), lopdf::Object::Reference(pages_id)),
            ]);
            let cat_id = doc.add_object(lopdf::Object::Dictionary(cat));
            doc.trailer
                .set(b"Root".to_vec(), lopdf::Object::Reference(cat_id));

            doc
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let snapshot_dir = tempfile::tempdir().expect("create snapshot dir");

        // Step 1: Create the "prior" PDF with a red highlight at y=700.
        let mut prior_pdf = build_pdf_with_highlight(
            [1.0, 0.0, 0.0], // red
            [72.0, 700.0, 200.0, 712.0],
        );
        let output_path = dir.path().join("paper.pdf");
        prior_pdf.save(&output_path).expect("save prior");

        // Step 2: Snapshot the prior PDF (simulating Phase 1.5).
        let snapshot_path = snapshot_dir.path().join("paper.pdf");
        std::fs::copy(&output_path, &snapshot_path).expect("snapshot copy");

        // Step 3: Overwrite the output file with a new PDF containing a
        // blue highlight at y=500 (simulating Phase 2 pdfium save).
        let mut new_pdf = build_pdf_with_highlight(
            [0.0, 0.0, 1.0], // blue
            [72.0, 500.0, 200.0, 512.0],
        );
        new_pdf.save(&output_path).expect("overwrite with new");

        // Step 4: Merge from the SNAPSHOT (not from output_path, which
        // was overwritten). This is the Phase 2.5 fix.
        let merged = merge_prior_annotations(&output_path, &snapshot_path)
            .expect("merge from snapshot must succeed");
        assert_eq!(
            merged, 1,
            "one prior annotation (red highlight) must be merged from the snapshot"
        );

        // Step 5: Verify the final PDF has both annotations.
        let final_doc = lopdf::Document::load(&output_path).expect("reload final");
        let page_ids: Vec<lopdf::ObjectId> = final_doc.get_pages().values().copied().collect();
        let annot_ids = get_annotation_ids(&final_doc, page_ids[0]).unwrap_or_default();
        assert_eq!(
            annot_ids.len(),
            2,
            "final PDF must contain both the blue (new) and red (merged from snapshot) highlights"
        );
    }

    /// T-APPEAR-DEF001-002: Without the snapshot fix, merging from the
    /// overwritten output file produces 0 merged annotations (the
    /// overwritten file has no prior annotations to merge). This test
    /// demonstrates the bug scenario.
    #[test]
    fn t_appear_def001_002_merge_from_overwritten_file_merges_nothing() {
        let dir = tempfile::tempdir().expect("create temp dir");

        // Create a PDF with a red highlight (simulating prior annotation).
        let mut prior = lopdf::Document::new();
        prior.version = "1.7".to_string();
        let annot = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Annot".to_vec())),
            (
                b"Subtype".to_vec(),
                lopdf::Object::Name(b"Highlight".to_vec()),
            ),
            (
                b"Rect".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Real(72.0),
                    lopdf::Object::Real(700.0),
                    lopdf::Object::Real(200.0),
                    lopdf::Object::Real(712.0),
                ]),
            ),
        ]);
        let annot_id = prior.add_object(lopdf::Object::Dictionary(annot));
        let page = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
            (
                b"Annots".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(annot_id)]),
            ),
        ]);
        let page_id = prior.add_object(lopdf::Object::Dictionary(page));
        let pages = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let pages_id = prior.add_object(lopdf::Object::Dictionary(pages));
        if let Ok(lopdf::Object::Dictionary(d)) = prior.get_object_mut(page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(pages_id));
        }
        let cat = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(pages_id)),
        ]);
        let cat_id = prior.add_object(lopdf::Object::Dictionary(cat));
        prior
            .trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(cat_id));

        let output_path = dir.path().join("paper.pdf");
        prior.save(&output_path).expect("save prior");

        // Overwrite with a clean PDF (no annotations) -- simulating
        // Phase 2 pdfium save without the snapshot fix.
        let mut clean = lopdf::Document::new();
        clean.version = "1.7".to_string();
        let clean_page = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Page".to_vec())),
            (
                b"MediaBox".to_vec(),
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            ),
        ]);
        let clean_page_id = clean.add_object(lopdf::Object::Dictionary(clean_page));
        let clean_pages = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Pages".to_vec())),
            (
                b"Kids".to_vec(),
                lopdf::Object::Array(vec![lopdf::Object::Reference(clean_page_id)]),
            ),
            (b"Count".to_vec(), lopdf::Object::Integer(1)),
        ]);
        let clean_pages_id = clean.add_object(lopdf::Object::Dictionary(clean_pages));
        if let Ok(lopdf::Object::Dictionary(d)) = clean.get_object_mut(clean_page_id) {
            d.set(b"Parent".to_vec(), lopdf::Object::Reference(clean_pages_id));
        }
        let clean_cat = lopdf::Dictionary::from_iter(vec![
            (b"Type".to_vec(), lopdf::Object::Name(b"Catalog".to_vec())),
            (b"Pages".to_vec(), lopdf::Object::Reference(clean_pages_id)),
        ]);
        let clean_cat_id = clean.add_object(lopdf::Object::Dictionary(clean_cat));
        clean
            .trailer
            .set(b"Root".to_vec(), lopdf::Object::Reference(clean_cat_id));
        clean.save(&output_path).expect("overwrite with clean");

        // Try to merge from the SAME overwritten file (the bug scenario).
        // The overwritten file has no annotations, so nothing is merged.
        let merged = merge_prior_annotations(&output_path, &output_path)
            .expect("merge from self must succeed");
        assert_eq!(
            merged, 0,
            "merging from the overwritten file (no prior annotations) must produce 0 merges"
        );
    }
}
