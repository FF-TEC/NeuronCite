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

// Highlight annotation removal from PDF files using lopdf.
//
// Removes highlight annotations from PDFs based on a filter mode: all highlights,
// highlights matching specific hex colors, or highlights on specific pages. Uses
// lopdf for pure-Rust PDF manipulation (not pdfium) to avoid the Windows SEH crash
// that occurs when pdfium reads lopdf-promoted indirect annotation references.
//
// The removal operates in two passes:
//   Pass 1 (read-only): Iterates all pages and their /Annots arrays, collects
//     ObjectIds of highlight annotations that match the removal criteria.
//   Pass 2 (mutating): For each page, filters the collected ObjectIds from the
//     /Annots array and deletes the orphaned annotation objects from the document.
//
// Supports dry_run mode: when enabled, pass 1 runs but pass 2 is skipped and the
// document is not saved. The returned RemoveResult still reports what would be removed.

use std::path::Path;

use serde::Serialize;

use crate::error::AnnotateError;
use crate::inspect::{AnnotSource, read_color_entry, read_subtype_str};

/// Filter mode controlling which highlight annotations are removed.
#[derive(Debug, Clone)]
pub enum RemoveMode {
    /// Remove all highlight annotations from the PDF.
    All,
    /// Remove only highlights whose stroke color (/C entry) matches one of the
    /// provided hex strings (e.g., "#FF0000", "#FFFF00"). Comparison is
    /// case-insensitive. Highlights without a readable /C entry are skipped.
    ByColor(Vec<String>),
    /// Remove only highlights on the specified pages (1-indexed). Highlights on
    /// pages not in this list are preserved.
    ByPage(Vec<usize>),
}

/// Result of a highlight removal operation. Reports the number of annotations
/// removed, which pages were affected, and how many orphaned appearance stream
/// objects were cleaned up.
#[derive(Debug, Clone, Serialize)]
pub struct RemoveResult {
    /// Number of highlight annotations that were removed (or would be removed
    /// in dry_run mode).
    pub annotations_removed: usize,
    /// 1-indexed page numbers where at least one annotation was removed.
    /// Sorted in ascending order, deduplicated.
    pub pages_affected: Vec<usize>,
    /// Number of orphaned annotation objects deleted from the document's object
    /// table. Each removed annotation contributes one object deletion. In
    /// dry_run mode, this field reports the count that would be deleted.
    pub appearance_objects_cleaned: usize,
}

/// Collected information about a single highlight annotation that matches the
/// removal criteria. Used between pass 1 (collection) and pass 2 (deletion).
struct HighlightTarget {
    /// 1-indexed page number where this annotation appears.
    page_number: usize,
    /// ObjectId of the page containing this annotation. Used to look up the
    /// page's /Annots array during pass 2.
    page_obj_id: lopdf::ObjectId,
    /// ObjectId of the annotation itself (indirect reference format only).
    /// None for inline dictionary annotations, which are removed by index
    /// rather than by ObjectId.
    annot_obj_id: Option<lopdf::ObjectId>,
    /// Index of this annotation within the /Annots array. Used for inline
    /// annotation removal where no ObjectId is available.
    annot_index: usize,
}

/// Removes highlight annotations from a PDF file based on the specified mode.
///
/// Opens the PDF at `pdf_path`, identifies highlight annotations matching the
/// `mode` criteria, removes them, and saves the result to `output_path`. When
/// `dry_run` is true, the removal analysis runs but no file is written.
///
/// The function uses lopdf (pure Rust) for all PDF operations. pdfium is not
/// invoked, avoiding the Windows SEH crash that occurs when pdfium reads
/// lopdf-modified annotation arrays.
///
/// # Errors
///
/// Returns `AnnotateError::PdfLoad` when the input PDF cannot be opened.
/// Returns `AnnotateError::SaveFailed` when the output PDF cannot be written.
pub fn remove_highlights(
    pdf_path: &Path,
    output_path: &Path,
    mode: &RemoveMode,
    dry_run: bool,
) -> Result<RemoveResult, AnnotateError> {
    let mut doc = lopdf::Document::load(pdf_path).map_err(|e| AnnotateError::PdfLoad {
        path: pdf_path.to_string_lossy().to_string(),
        reason: format!("{e}"),
    })?;

    let pages = doc.get_pages();

    // Pass 1: Collect all highlight annotations matching the removal criteria.
    // All Document borrows from the page/annotation iteration are scoped so
    // that pass 2 can take mutable borrows for modification.
    let targets = collect_targets(&doc, &pages, mode);

    let annotations_removed = targets.len();
    let mut pages_affected: Vec<usize> = targets.iter().map(|t| t.page_number).collect();
    pages_affected.sort_unstable();
    pages_affected.dedup();

    if dry_run || targets.is_empty() {
        return Ok(RemoveResult {
            annotations_removed,
            pages_affected,
            appearance_objects_cleaned: annotations_removed,
        });
    }

    // Pass 2: Remove annotations from each affected page's /Annots array and
    // delete the orphaned annotation objects from the document.
    let objects_cleaned = remove_targets(&mut doc, &targets);

    // Save the modified document to the output path.
    doc.save(output_path)
        .map_err(|e| AnnotateError::SaveFailed {
            path: output_path.to_string_lossy().to_string(),
            reason: format!("{e}"),
        })?;

    Ok(RemoveResult {
        annotations_removed,
        pages_affected,
        appearance_objects_cleaned: objects_cleaned,
    })
}

/// Pass 1: Iterates all pages and their /Annots arrays, collecting information
/// about highlight annotations that match the removal mode. All Document borrows
/// are dropped before this function returns.
fn collect_targets(
    doc: &lopdf::Document,
    pages: &std::collections::BTreeMap<u32, lopdf::ObjectId>,
    mode: &RemoveMode,
) -> Vec<HighlightTarget> {
    let mut targets = Vec::new();

    for (&page_num, &page_id) in pages {
        let page_number = page_num as usize;

        // Skip pages not in the ByPage filter.
        if let RemoveMode::ByPage(page_list) = mode
            && !page_list.contains(&page_number)
        {
            continue;
        }

        // Collect annotation sources from this page's /Annots array.
        let sources = collect_page_annot_sources(doc, page_id);

        for (index, source) in sources.iter().enumerate() {
            let (is_highlight, color_hex, obj_id) = match source {
                AnnotSource::Ref(id) => {
                    if let Some(dict) = doc.get_object(*id).ok().and_then(|o| o.as_dict().ok()) {
                        let subtype = read_subtype_str(dict);
                        let color = read_color_entry(dict, b"C");
                        (subtype == "highlight", color, Some(*id))
                    } else {
                        continue;
                    }
                }
                AnnotSource::Inline(dict) => {
                    let subtype = read_subtype_str(dict);
                    let color = read_color_entry(dict, b"C");
                    (subtype == "highlight", color, None)
                }
            };

            if !is_highlight {
                continue;
            }

            let matches = match mode {
                RemoveMode::All => true,
                RemoveMode::ByPage(_) => true, // Page filtering is handled above.
                RemoveMode::ByColor(colors) => {
                    if let Some(hex) = &color_hex {
                        colors.iter().any(|c| c.eq_ignore_ascii_case(hex))
                    } else {
                        false
                    }
                }
            };

            if matches {
                targets.push(HighlightTarget {
                    page_number,
                    page_obj_id: page_id,
                    annot_obj_id: obj_id,
                    annot_index: index,
                });
            }
        }
    }

    targets
}

/// Collects AnnotSource entries from a page's /Annots array. Handles both
/// direct arrays and indirect references to arrays. All Document borrows are
/// contained within this function.
fn collect_page_annot_sources(doc: &lopdf::Document, page_id: lopdf::ObjectId) -> Vec<AnnotSource> {
    let page_dict = match doc.get_object(page_id).ok().and_then(|o| o.as_dict().ok()) {
        Some(d) => d,
        None => return Vec::new(),
    };

    let annots_obj = match page_dict.get(b"Annots").ok() {
        Some(o) => o,
        None => return Vec::new(),
    };

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

    let mut sources = Vec::with_capacity(annots_arr.len());
    for item in &annots_arr {
        match item {
            lopdf::Object::Reference(id) => {
                sources.push(AnnotSource::Ref(*id));
            }
            lopdf::Object::Dictionary(d) => {
                sources.push(AnnotSource::Inline(d.clone()));
            }
            _ => {}
        }
    }
    sources
}

/// Pass 2: Removes the targeted annotations from each page's /Annots array and
/// deletes orphaned annotation objects. Returns the number of objects cleaned up.
fn remove_targets(doc: &mut lopdf::Document, targets: &[HighlightTarget]) -> usize {
    // Group targets by page ObjectId for batch processing.
    let mut by_page: std::collections::HashMap<lopdf::ObjectId, Vec<&HighlightTarget>> =
        std::collections::HashMap::new();
    for target in targets {
        by_page.entry(target.page_obj_id).or_default().push(target);
    }

    let mut objects_cleaned = 0;

    for (page_id, page_targets) in &by_page {
        // Collect the set of annotation indices to remove from this page.
        let remove_indices: std::collections::HashSet<usize> =
            page_targets.iter().map(|t| t.annot_index).collect();

        // Read the current /Annots array for this page.
        let annots_arr = {
            let page_dict = match doc.get_object(*page_id).ok().and_then(|o| o.as_dict().ok()) {
                Some(d) => d,
                None => continue,
            };
            let annots_obj = match page_dict.get(b"Annots").ok() {
                Some(o) => o,
                None => continue,
            };
            match annots_obj {
                lopdf::Object::Array(arr) => arr.clone(),
                lopdf::Object::Reference(r) => {
                    match doc.get_object(*r).ok().and_then(|o| o.as_array().ok()) {
                        Some(arr) => arr.clone(),
                        None => continue,
                    }
                }
                _ => continue,
            }
        };

        // Build a filtered /Annots array excluding the removed annotations.
        let filtered: Vec<lopdf::Object> = annots_arr
            .iter()
            .enumerate()
            .filter(|(i, _)| !remove_indices.contains(i))
            .map(|(_, obj)| obj.clone())
            .collect();

        // Write the filtered array back to the page dictionary.
        if let Some(dict) = doc
            .get_object_mut(*page_id)
            .ok()
            .and_then(|o| o.as_dict_mut().ok())
        {
            if filtered.is_empty() {
                dict.remove(b"Annots");
            } else {
                dict.set(b"Annots", lopdf::Object::Array(filtered));
            }
        }

        // Delete orphaned annotation objects (indirect references only).
        for target in page_targets {
            if let Some(obj_id) = target.annot_obj_id {
                doc.delete_object(obj_id);
                objects_cleaned += 1;
            }
        }
    }

    objects_cleaned
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a minimal PDF document with highlight annotations using lopdf.
    /// Each entry in `highlights` specifies (page_index_0based, r, g, b) where
    /// r/g/b are 0.0..1.0 normalized PDF color values.
    fn create_test_pdf_with_highlights(
        num_pages: usize,
        highlights: &[(usize, f32, f32, f32)],
    ) -> lopdf::Document {
        let mut doc = lopdf::Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let mut page_ids = Vec::new();

        for _ in 0..num_pages {
            let page_id = doc.new_object_id();
            page_ids.push(page_id);
        }

        // Create page objects with /Annots arrays based on the highlights.
        for (page_idx, page_id) in page_ids.iter().enumerate() {
            let page_highlights: Vec<&(usize, f32, f32, f32)> =
                highlights.iter().filter(|h| h.0 == page_idx).collect();

            let mut page_dict = lopdf::Dictionary::new();
            page_dict.set("Type", lopdf::Object::Name(b"Page".to_vec()));
            page_dict.set("Parent", lopdf::Object::Reference(pages_id));
            page_dict.set(
                "MediaBox",
                lopdf::Object::Array(vec![
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(0),
                    lopdf::Object::Integer(612),
                    lopdf::Object::Integer(792),
                ]),
            );

            if !page_highlights.is_empty() {
                let mut annots = Vec::new();
                for &&(_, r, g, b) in &page_highlights {
                    // Create each annotation as an indirect object (the format
                    // produced by lopdf's promote_inline_annotations).
                    let mut annot_dict = lopdf::Dictionary::new();
                    annot_dict.set("Type", lopdf::Object::Name(b"Annot".to_vec()));
                    annot_dict.set("Subtype", lopdf::Object::Name(b"Highlight".to_vec()));
                    annot_dict.set(
                        "Rect",
                        lopdf::Object::Array(vec![
                            lopdf::Object::Real(50.0),
                            lopdf::Object::Real(700.0),
                            lopdf::Object::Real(200.0),
                            lopdf::Object::Real(720.0),
                        ]),
                    );
                    annot_dict.set(
                        "C",
                        lopdf::Object::Array(vec![
                            lopdf::Object::Real(r),
                            lopdf::Object::Real(g),
                            lopdf::Object::Real(b),
                        ]),
                    );
                    annot_dict.set(
                        "QuadPoints",
                        lopdf::Object::Array(vec![
                            lopdf::Object::Real(50.0),
                            lopdf::Object::Real(720.0),
                            lopdf::Object::Real(200.0),
                            lopdf::Object::Real(720.0),
                            lopdf::Object::Real(50.0),
                            lopdf::Object::Real(700.0),
                            lopdf::Object::Real(200.0),
                            lopdf::Object::Real(700.0),
                        ]),
                    );

                    let annot_id = doc.add_object(lopdf::Object::Dictionary(annot_dict));
                    annots.push(lopdf::Object::Reference(annot_id));
                }
                page_dict.set("Annots", lopdf::Object::Array(annots));
            }

            doc.objects
                .insert(*page_id, lopdf::Object::Dictionary(page_dict));
        }

        // Create the /Pages node.
        let mut pages_dict = lopdf::Dictionary::new();
        pages_dict.set("Type", lopdf::Object::Name(b"Pages".to_vec()));
        pages_dict.set(
            "Kids",
            lopdf::Object::Array(
                page_ids
                    .iter()
                    .map(|id| lopdf::Object::Reference(*id))
                    .collect(),
            ),
        );
        pages_dict.set("Count", lopdf::Object::Integer(num_pages as i64));
        doc.objects
            .insert(pages_id, lopdf::Object::Dictionary(pages_dict));

        // Create the /Catalog.
        let catalog_id = doc.add_object(lopdf::Object::Dictionary({
            let mut d = lopdf::Dictionary::new();
            d.set("Type", lopdf::Object::Name(b"Catalog".to_vec()));
            d.set("Pages", lopdf::Object::Reference(pages_id));
            d
        }));

        doc.trailer
            .set("Root", lopdf::Object::Reference(catalog_id));

        doc
    }

    /// Saves a lopdf Document to a temporary file and returns the path.
    fn save_temp(doc: &mut lopdf::Document) -> tempfile::NamedTempFile {
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        doc.save(tmp.path()).expect("save test PDF");
        tmp
    }

    /// T-REMOVE-001: RemoveMode::All removes all highlight annotations from
    /// a PDF with 3 highlights across 2 pages. Verifies annotations_removed=3
    /// and both pages are in pages_affected.
    #[test]
    fn t_remove_001_all_mode_removes_all_highlights() {
        let mut doc = create_test_pdf_with_highlights(
            2,
            &[
                (0, 1.0, 0.0, 0.0), // page 1: red
                (0, 0.0, 1.0, 0.0), // page 1: green
                (1, 0.0, 0.0, 1.0), // page 2: blue
            ],
        );
        let tmp_in = save_temp(&mut doc);
        let tmp_out = tempfile::NamedTempFile::new().expect("create output temp");

        let result = remove_highlights(tmp_in.path(), tmp_out.path(), &RemoveMode::All, false)
            .expect("removal must succeed");

        assert_eq!(result.annotations_removed, 3);
        assert_eq!(result.pages_affected, vec![1, 2]);
        assert_eq!(result.appearance_objects_cleaned, 3);

        // Verify the output PDF has no highlights.
        let inspected =
            crate::inspect::inspect_pdf(tmp_out.path(), None).expect("inspect must succeed");
        assert_eq!(
            inspected.highlight_count, 0,
            "all highlights must be removed from output PDF"
        );
    }

    /// T-REMOVE-002: RemoveMode::ByColor removes only highlights matching the
    /// specified color. Red highlights are removed, yellow highlights survive.
    #[test]
    fn t_remove_002_by_color_removes_matching_only() {
        let mut doc = create_test_pdf_with_highlights(
            1,
            &[
                (0, 1.0, 0.0, 0.0), // red
                (0, 1.0, 1.0, 0.0), // yellow
            ],
        );
        let tmp_in = save_temp(&mut doc);
        let tmp_out = tempfile::NamedTempFile::new().expect("create output temp");

        let result = remove_highlights(
            tmp_in.path(),
            tmp_out.path(),
            &RemoveMode::ByColor(vec!["#FF0000".to_string()]),
            false,
        )
        .expect("removal must succeed");

        assert_eq!(result.annotations_removed, 1, "only red highlight removed");
        assert_eq!(result.pages_affected, vec![1]);

        let inspected =
            crate::inspect::inspect_pdf(tmp_out.path(), None).expect("inspect must succeed");
        assert_eq!(
            inspected.highlight_count, 1,
            "one yellow highlight must survive"
        );
        assert!(
            inspected.unique_colors.contains(&"#FFFF00".to_string()),
            "surviving highlight must be yellow"
        );
    }

    /// T-REMOVE-003: RemoveMode::ByPage removes only highlights on the specified
    /// page. Highlights on page 1 are removed, page 2 highlights are untouched.
    #[test]
    fn t_remove_003_by_page_removes_correct_page() {
        let mut doc = create_test_pdf_with_highlights(
            2,
            &[
                (0, 1.0, 0.0, 0.0), // page 1: red
                (1, 0.0, 0.0, 1.0), // page 2: blue
            ],
        );
        let tmp_in = save_temp(&mut doc);
        let tmp_out = tempfile::NamedTempFile::new().expect("create output temp");

        let result = remove_highlights(
            tmp_in.path(),
            tmp_out.path(),
            &RemoveMode::ByPage(vec![1]),
            false,
        )
        .expect("removal must succeed");

        assert_eq!(result.annotations_removed, 1);
        assert_eq!(result.pages_affected, vec![1]);

        let inspected =
            crate::inspect::inspect_pdf(tmp_out.path(), None).expect("inspect must succeed");
        assert_eq!(
            inspected.highlight_count, 1,
            "page 2 highlight must survive"
        );
        assert!(
            inspected.unique_colors.contains(&"#0000FF".to_string()),
            "surviving highlight must be blue (page 2)"
        );
    }

    /// T-REMOVE-004: dry_run mode reports what would be removed but does not
    /// modify the source file. The output file is not written.
    #[test]
    fn t_remove_004_dry_run_does_not_write() {
        let mut doc = create_test_pdf_with_highlights(1, &[(0, 1.0, 0.0, 0.0)]);
        let tmp_in = save_temp(&mut doc);
        let tmp_out = tempfile::NamedTempFile::new().expect("create output temp");

        // Record the original file size.
        let original_size = std::fs::metadata(tmp_in.path()).expect("metadata").len();

        let result = remove_highlights(tmp_in.path(), tmp_out.path(), &RemoveMode::All, true)
            .expect("dry run must succeed");

        assert_eq!(
            result.annotations_removed, 1,
            "dry_run reports removal count"
        );
        assert_eq!(result.pages_affected, vec![1]);

        // Source file must be unchanged.
        let after_size = std::fs::metadata(tmp_in.path()).expect("metadata").len();
        assert_eq!(
            original_size, after_size,
            "source file must not be modified in dry_run mode"
        );

        // Source file must still have the highlight.
        let inspected =
            crate::inspect::inspect_pdf(tmp_in.path(), None).expect("inspect must succeed");
        assert_eq!(
            inspected.highlight_count, 1,
            "source file highlights must be intact after dry_run"
        );
    }

    /// T-REMOVE-005: When output_path differs from the source, the source
    /// file is preserved unchanged while the output has fewer annotations.
    #[test]
    fn t_remove_005_output_path_separate() {
        let mut doc = create_test_pdf_with_highlights(
            1,
            &[
                (0, 1.0, 0.0, 0.0), // red
                (0, 0.0, 1.0, 0.0), // green
            ],
        );
        let tmp_in = save_temp(&mut doc);
        let tmp_out = tempfile::NamedTempFile::new().expect("create output temp");

        let _ = remove_highlights(
            tmp_in.path(),
            tmp_out.path(),
            &RemoveMode::ByColor(vec!["#FF0000".to_string()]),
            false,
        )
        .expect("removal must succeed");

        // Source file must still have 2 highlights.
        let source_inspected =
            crate::inspect::inspect_pdf(tmp_in.path(), None).expect("inspect source");
        assert_eq!(source_inspected.highlight_count, 2);

        // Output file must have 1 highlight (green).
        let output_inspected =
            crate::inspect::inspect_pdf(tmp_out.path(), None).expect("inspect output");
        assert_eq!(output_inspected.highlight_count, 1);
    }

    /// T-REMOVE-006: ByColor with a non-existent color results in zero removals.
    /// The output PDF is identical to the input (except for lopdf re-serialization).
    #[test]
    fn t_remove_006_no_match_returns_zero() {
        let mut doc = create_test_pdf_with_highlights(1, &[(0, 1.0, 0.0, 0.0)]);
        let tmp_in = save_temp(&mut doc);
        let tmp_out = tempfile::NamedTempFile::new().expect("create output temp");

        let result = remove_highlights(
            tmp_in.path(),
            tmp_out.path(),
            &RemoveMode::ByColor(vec!["#00FF00".to_string()]),
            false,
        )
        .expect("removal must succeed");

        assert_eq!(
            result.annotations_removed, 0,
            "no green highlights to remove"
        );
        assert!(result.pages_affected.is_empty());
        assert_eq!(result.appearance_objects_cleaned, 0);
    }

    /// T-REMOVE-007: ByColor comparison is case-insensitive. "#ff0000" matches
    /// a highlight with color "#FF0000".
    #[test]
    fn t_remove_007_by_color_case_insensitive() {
        let mut doc = create_test_pdf_with_highlights(1, &[(0, 1.0, 0.0, 0.0)]);
        let tmp_in = save_temp(&mut doc);
        let tmp_out = tempfile::NamedTempFile::new().expect("create output temp");

        let result = remove_highlights(
            tmp_in.path(),
            tmp_out.path(),
            &RemoveMode::ByColor(vec!["#ff0000".to_string()]),
            false,
        )
        .expect("removal must succeed");

        assert_eq!(
            result.annotations_removed, 1,
            "lowercase hex must match uppercase color entry"
        );
    }

    /// T-REMOVE-008: RemoveMode::All on a PDF with zero annotations results
    /// in zero removals and an empty pages_affected list.
    #[test]
    fn t_remove_008_no_annotations_in_pdf() {
        let mut doc = create_test_pdf_with_highlights(2, &[]);
        let tmp_in = save_temp(&mut doc);
        let tmp_out = tempfile::NamedTempFile::new().expect("create output temp");

        let result = remove_highlights(tmp_in.path(), tmp_out.path(), &RemoveMode::All, false)
            .expect("removal must succeed");

        assert_eq!(result.annotations_removed, 0);
        assert!(result.pages_affected.is_empty());
    }

    /// T-REMOVE-009: ByPage with a page number that exceeds the document's
    /// page count results in zero removals (no error).
    #[test]
    fn t_remove_009_by_page_out_of_range() {
        let mut doc = create_test_pdf_with_highlights(1, &[(0, 1.0, 0.0, 0.0)]);
        let tmp_in = save_temp(&mut doc);
        let tmp_out = tempfile::NamedTempFile::new().expect("create output temp");

        let result = remove_highlights(
            tmp_in.path(),
            tmp_out.path(),
            &RemoveMode::ByPage(vec![99]),
            false,
        )
        .expect("removal must succeed");

        assert_eq!(
            result.annotations_removed, 0,
            "out-of-range page produces zero removals"
        );
    }

    /// T-REMOVE-010: Multiple colors in ByColor filter. Both red and blue
    /// highlights are removed, green survives.
    #[test]
    fn t_remove_010_by_color_multiple_colors() {
        let mut doc = create_test_pdf_with_highlights(
            1,
            &[
                (0, 1.0, 0.0, 0.0), // red
                (0, 0.0, 1.0, 0.0), // green
                (0, 0.0, 0.0, 1.0), // blue
            ],
        );
        let tmp_in = save_temp(&mut doc);
        let tmp_out = tempfile::NamedTempFile::new().expect("create output temp");

        let result = remove_highlights(
            tmp_in.path(),
            tmp_out.path(),
            &RemoveMode::ByColor(vec!["#FF0000".to_string(), "#0000FF".to_string()]),
            false,
        )
        .expect("removal must succeed");

        assert_eq!(result.annotations_removed, 2, "red and blue removed");

        let inspected = crate::inspect::inspect_pdf(tmp_out.path(), None).expect("inspect output");
        assert_eq!(inspected.highlight_count, 1, "green must survive");
        assert!(
            inspected.unique_colors.contains(&"#00FF00".to_string()),
            "surviving highlight must be green"
        );
    }
}
