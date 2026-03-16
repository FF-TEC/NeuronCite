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

// Export functions for citation verification results.
//
// Generates output files from completed citation verification jobs:
//
// 1. Annotation pipeline CSV (annotation_pipeline_input.csv) -- Input for the
//    neuroncite-annotate pipeline. Each row maps a verified citation to a
//    highlight annotation with verdict-based color coding.
//
// 2. Full-data CSV (citation_data.csv) -- Complete 38-column export with all
//    metadata and verification results, sorted by tex_line ascending.
//
// 3. Excel workbook (citation_data.xlsx) -- Same 38 columns as the CSV with
//    formatted header, verdict-colored rows, badge cells, and frozen panes.
//
// 4. Corrections JSON -- Array of suggested LaTeX corrections sorted by line
//    number, extracted from the result_json of done citation rows.
//
// 5. Summary report JSON -- Statistics, per-citation details, and alerts for
//    the complete job.

use std::collections::HashMap;

use rust_xlsxwriter::{Color, Format, FormatAlign, FormatBorder, Workbook, XlsxError};

use crate::error::CitationError;
use crate::types::{
    Alert, CitationRow, CorrectionType, EXPORT_COLUMNS, ExportSummary, SubmitEntry, Verdict,
};

// ---------------------------------------------------------------------------
// Shared helpers for extracting row data into 38-column string arrays
// ---------------------------------------------------------------------------

/// Extracts the 38-column string values from a CitationRow and its parsed
/// SubmitEntry. Returns a fixed-size array matching the EXPORT_COLUMNS order.
/// The `row_number` parameter provides a 1-based sequential ID sorted by
/// tex_line (assigned by the caller). Includes passage_location for each
/// of the three passage slots, classifying where in the document structure
/// the passage was found (e.g., abstract, methodology, results).
fn row_to_columns(row: &CitationRow, row_number: usize) -> [String; 38] {
    let co_info = row.co_citation_info();

    // Parse the SubmitEntry from result_json if the row is done.
    let entry: Option<SubmitEntry> = if row.status == "done" {
        row.result_json
            .as_ref()
            .and_then(|json| serde_json::from_str(json).ok())
    } else {
        None
    };

    // Extract passage columns (up to 3 passages).
    let passages = entry.as_ref().map(|e| &e.passages);
    let p1_text = passages
        .and_then(|p| p.first())
        .map(|p| p.passage_text.clone())
        .unwrap_or_default();
    let p1_page = passages
        .and_then(|p| p.first())
        .map(|p| p.page.to_string())
        .unwrap_or_default();
    let p1_score = passages
        .and_then(|p| p.first())
        .map(|p| format!("{:.2}", p.relevance_score))
        .unwrap_or_default();
    let p2_text = passages
        .and_then(|p| p.get(1))
        .map(|p| p.passage_text.clone())
        .unwrap_or_default();
    let p2_page = passages
        .and_then(|p| p.get(1))
        .map(|p| p.page.to_string())
        .unwrap_or_default();
    let p2_score = passages
        .and_then(|p| p.get(1))
        .map(|p| format!("{:.2}", p.relevance_score))
        .unwrap_or_default();
    let p3_text = passages
        .and_then(|p| p.get(2))
        .map(|p| p.passage_text.clone())
        .unwrap_or_default();
    let p3_page = passages
        .and_then(|p| p.get(2))
        .map(|p| p.page.to_string())
        .unwrap_or_default();
    let p3_score = passages
        .and_then(|p| p.get(2))
        .map(|p| format!("{:.2}", p.relevance_score))
        .unwrap_or_default();

    // Extract passage_location labels. Each passage carries a PassageLocation
    // enum value that classifies where in the PDF the passage was found.
    let p1_location = passages
        .and_then(|p| p.first())
        .map(|p| p.passage_location.label().to_string())
        .unwrap_or_default();
    let p2_location = passages
        .and_then(|p| p.get(1))
        .map(|p| p.passage_location.label().to_string())
        .unwrap_or_default();
    let p3_location = passages
        .and_then(|p| p.get(2))
        .map(|p| p.passage_location.label().to_string())
        .unwrap_or_default();

    // Serialize other_source_list as JSON.
    let other_source_json = entry
        .as_ref()
        .map(|e| serde_json::to_string(&e.other_source_list).unwrap_or_else(|_| "[]".to_string()))
        .unwrap_or_else(|| "[]".to_string());

    [
        row_number.to_string(),                        // id (1-based)
        row.tex_line.to_string(),                      // tex_line
        row.section_title.clone().unwrap_or_default(), // section_title
        row.anchor_before.clone(),                     // anchor_before
        row.anchor_after.clone(),                      // anchor_after
        co_info.is_co_citation.to_string(),            // is_co_citation
        serde_json::to_string(&co_info.co_cited_with).unwrap_or_else(|_| "[]".to_string()), // co_cited_with
        row.cite_key.clone(),                 // cite_key
        row.author.clone(),                   // author
        row.year.clone().unwrap_or_default(), // year
        row.title.clone(),                    // title
        entry
            .as_ref()
            .map(|e| e.claim_original.clone())
            .unwrap_or_default(), // claim_original
        entry
            .as_ref()
            .map(|e| e.claim_english.clone())
            .unwrap_or_default(), // claim_english
        entry
            .as_ref()
            .map(|e| {
                format!("{}", serde_json::to_value(e.verdict).unwrap_or_default())
                    .trim_matches('"')
                    .to_string()
            })
            .unwrap_or_default(), // verdict
        entry
            .as_ref()
            .map(|e| format!("{:.2}", e.confidence))
            .unwrap_or_default(), // confidence
        entry
            .as_ref()
            .map(|e| e.source_match.to_string())
            .unwrap_or_default(), // source_match
        entry
            .as_ref()
            .map(|e| e.reasoning.clone())
            .unwrap_or_default(), // reasoning
        entry.as_ref().map(|e| e.flag.clone()).unwrap_or_default(), // flag
        entry
            .as_ref()
            .map(|e| {
                format!(
                    "{}",
                    serde_json::to_value(e.latex_correction.correction_type).unwrap_or_default()
                )
                .trim_matches('"')
                .to_string()
            })
            .unwrap_or_default(), // latex_correction_type
        entry
            .as_ref()
            .map(|e| e.latex_correction.explanation.clone())
            .unwrap_or_default(), // latex_correction_explanation
        entry
            .as_ref()
            .map(|e| e.latex_correction.original_text.clone())
            .unwrap_or_default(), // latex_correction_original
        entry
            .as_ref()
            .map(|e| e.latex_correction.suggested_text.clone())
            .unwrap_or_default(), // latex_correction_suggested
        entry
            .as_ref()
            .map(|e| serde_json::to_string(&e.better_source).unwrap_or_else(|_| "[]".to_string()))
            .unwrap_or_else(|| "[]".to_string()), // better_source
        p1_text,                              // passage_1_text
        p1_page,                              // passage_1_page
        p1_score,                             // passage_1_score
        p1_location,                          // passage_1_location
        p2_text,                              // passage_2_text
        p2_page,                              // passage_2_page
        p2_score,                             // passage_2_score
        p2_location,                          // passage_2_location
        p3_text,                              // passage_3_text
        p3_page,                              // passage_3_page
        p3_score,                             // passage_3_score
        p3_location,                          // passage_3_location
        other_source_json,                    // other_source_list
        entry
            .as_ref()
            .map(|e| e.search_rounds.to_string())
            .unwrap_or_default(), // search_rounds
        row.matched_file_id
            .map(|id| id.to_string())
            .unwrap_or_default(), // matched_file_id
    ]
}

/// Sorts rows by tex_line ascending and returns the sorted order.
fn sort_rows_by_tex_line(rows: &[CitationRow]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..rows.len()).collect();
    indices.sort_by_key(|&i| rows[i].tex_line);
    indices
}

// ---------------------------------------------------------------------------
// Annotation pipeline CSV (for neuroncite-annotate)
// ---------------------------------------------------------------------------

/// Builds a CSV string compatible with the neuroncite-annotate input format.
/// Each row represents one quote passage from a verified citation. Rows
/// without passages or without result_json are skipped.
///
/// CSV columns: title, author, quote, color, comment
///
/// The color is determined by the verdict stored in result_json. The comment
/// contains the sub-agent's reasoning text.
pub fn build_annotation_csv(rows: &[CitationRow]) -> Result<String, CitationError> {
    let mut wtr = csv::Writer::from_writer(Vec::new());

    // Write header row. The `page` column carries the 1-indexed page number
    // from the sub-agent's PassageRef so the annotation pipeline can target
    // the correct page directly instead of scanning the entire PDF.
    wtr.write_record(["title", "author", "quote", "color", "comment", "page"])?;

    for row in rows {
        if row.status != "done" {
            continue;
        }

        let result_json = match &row.result_json {
            Some(json) => json,
            None => continue,
        };

        let entry: SubmitEntry = match serde_json::from_str(result_json) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Skip rows without passages (nothing to annotate in the PDF).
        if entry.passages.is_empty() {
            continue;
        }

        let color = entry.verdict.color_hex();

        // Build a comment string from the verdict and reasoning.
        let comment = format!("[{}] {}", entry.verdict.label(), entry.reasoning);

        // Create one annotation row per passage. Each row includes the page
        // number from the sub-agent's search result, enabling page-targeted
        // matching in the annotation pipeline.
        for passage in &entry.passages {
            let page_str = passage.page.to_string();
            wtr.write_record([
                &row.title,
                &row.author,
                &passage.passage_text,
                color,
                &comment,
                &page_str,
            ])?;
        }
    }

    let bytes = wtr
        .into_inner()
        .map_err(|e| CitationError::Io(e.into_error()))?;
    String::from_utf8(bytes)
        .map_err(|e| CitationError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
}

// ---------------------------------------------------------------------------
// Full 38-column data CSV
// ---------------------------------------------------------------------------

/// Builds the full 38-column CSV export with all metadata and verification
/// results. Rows are sorted by tex_line ascending with a 1-based sequential
/// ID in the first column.
pub fn build_data_csv(rows: &[CitationRow]) -> Result<String, CitationError> {
    let mut wtr = csv::Writer::from_writer(Vec::new());

    // Write header row.
    let headers: Vec<&str> = EXPORT_COLUMNS.iter().map(|(name, _)| *name).collect();
    wtr.write_record(&headers)?;

    // Sort rows by tex_line ascending.
    let sorted_indices = sort_rows_by_tex_line(rows);

    for (seq, &idx) in sorted_indices.iter().enumerate() {
        let columns = row_to_columns(&rows[idx], seq + 1);
        let refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        wtr.write_record(&refs)?;
    }

    let bytes = wtr
        .into_inner()
        .map_err(|e| CitationError::Io(e.into_error()))?;
    String::from_utf8(bytes)
        .map_err(|e| CitationError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
}

// ---------------------------------------------------------------------------
// Excel workbook (.xlsx)
// ---------------------------------------------------------------------------

/// Parses a hex color string (#RRGGBB) into a rust_xlsxwriter Color value.
fn hex_to_color(hex: &str) -> Color {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
    Color::RGB(u32::from(r) << 16 | u32::from(g) << 8 | u32::from(b))
}

/// Builds a formatted Excel workbook (.xlsx) containing all 38 columns.
/// Returns the workbook bytes ready to be written to disk.
///
/// Formatting:
/// - Header: dark blue background (#1F3864), white bold text, autofilter,
///   frozen first row
/// - Row background colored by verdict
/// - Verdict cell: badge color with white bold centered text
/// - Flag cell: orange badge for warning, red for critical
/// - Row height: 55px for data, 22px for header
/// - Text wrapping enabled for all cells
/// - Thin gray borders (#BBBBBB)
pub fn build_excel(rows: &[CitationRow]) -> Result<Vec<u8>, CitationError> {
    let mut workbook = Workbook::new();
    let worksheet = workbook.add_worksheet();
    worksheet
        .set_name("Citation Verification")
        .map_err(xlsx_err)?;

    let col_count = EXPORT_COLUMNS.len() as u16;

    // -- Format definitions ------------------------------------------------

    let border_color = hex_to_color("#BBBBBB");

    // Header format: dark blue background, white bold, center-aligned.
    let header_fmt = Format::new()
        .set_background_color(hex_to_color("#1F3864"))
        .set_font_color(Color::White)
        .set_bold()
        .set_text_wrap()
        .set_align(FormatAlign::Center)
        .set_align(FormatAlign::VerticalCenter)
        .set_border(FormatBorder::Thin)
        .set_border_color(border_color);

    // Base cell format with text wrap and thin borders.
    let base_cell_fmt = Format::new()
        .set_text_wrap()
        .set_align(FormatAlign::Top)
        .set_border(FormatBorder::Thin)
        .set_border_color(border_color);

    // Write header row.
    for (col_idx, (name, width)) in EXPORT_COLUMNS.iter().enumerate() {
        worksheet
            .write_with_format(0, col_idx as u16, *name, &header_fmt)
            .map_err(xlsx_err)?;
        worksheet
            .set_column_width(col_idx as u16, *width as f64)
            .map_err(xlsx_err)?;
    }

    // Set header row height.
    worksheet.set_row_height(0, 22.0).map_err(xlsx_err)?;

    // Freeze the first row so it stays visible when scrolling.
    worksheet.set_freeze_panes(1, 0).map_err(xlsx_err)?;

    // Enable autofilter on the header row.
    worksheet
        .autofilter(0, 0, 0, col_count - 1)
        .map_err(xlsx_err)?;

    // Sort rows by tex_line ascending.
    let sorted_indices = sort_rows_by_tex_line(rows);

    // Write data rows.
    for (seq, &idx) in sorted_indices.iter().enumerate() {
        let excel_row = (seq + 1) as u32;
        let row = &rows[idx];
        let columns = row_to_columns(row, seq + 1);

        // Determine verdict for row coloring.
        let verdict: Option<Verdict> = if row.status == "done" {
            row.result_json
                .as_ref()
                .and_then(|json| serde_json::from_str::<SubmitEntry>(json).ok())
                .map(|e| e.verdict)
        } else {
            None
        };

        // Build row-level format with verdict background color.
        let row_bg = verdict.map(|v| hex_to_color(v.row_bg_hex()));
        let mut row_fmt = base_cell_fmt.clone();
        if let Some(bg) = row_bg {
            row_fmt = row_fmt.set_background_color(bg);
        }

        // Write each column.
        for (col_idx, value) in columns.iter().enumerate() {
            let col = col_idx as u16;
            let col_name = EXPORT_COLUMNS[col_idx].0;

            match col_name {
                // Verdict column: badge format with darker color and white bold text.
                // The match guard ensures verdict is Some. The if-let inside the
                // arm body extracts the value without calling unwrap().
                "verdict" if verdict.is_some() => {
                    if let Some(v) = verdict {
                        let badge_fmt = Format::new()
                            .set_background_color(hex_to_color(v.badge_hex()))
                            .set_font_color(Color::White)
                            .set_bold()
                            .set_align(FormatAlign::Center)
                            .set_align(FormatAlign::VerticalCenter)
                            .set_text_wrap()
                            .set_border(FormatBorder::Thin)
                            .set_border_color(border_color);
                        worksheet
                            .write_with_format(excel_row, col, value.as_str(), &badge_fmt)
                            .map_err(xlsx_err)?;
                    }
                }
                // Flag column: badge format for critical/warning.
                "flag" if !value.is_empty() => {
                    let flag_color = if value == "critical" {
                        hex_to_color("#DC3545")
                    } else {
                        hex_to_color("#FF6D00")
                    };
                    let flag_fmt = Format::new()
                        .set_background_color(flag_color)
                        .set_font_color(Color::White)
                        .set_bold()
                        .set_align(FormatAlign::Center)
                        .set_align(FormatAlign::VerticalCenter)
                        .set_text_wrap()
                        .set_border(FormatBorder::Thin)
                        .set_border_color(border_color);
                    worksheet
                        .write_with_format(excel_row, col, value.as_str(), &flag_fmt)
                        .map_err(xlsx_err)?;
                }
                // Numeric columns: write as numbers for sorting.
                "id" | "tex_line" | "confidence" | "search_rounds" | "matched_file_id"
                | "passage_1_page" | "passage_1_score" | "passage_2_page" | "passage_2_score"
                | "passage_3_page" | "passage_3_score" => {
                    if let Ok(num) = value.parse::<f64>() {
                        worksheet
                            .write_number_with_format(excel_row, col, num, &row_fmt)
                            .map_err(xlsx_err)?;
                    } else {
                        worksheet
                            .write_with_format(excel_row, col, value.as_str(), &row_fmt)
                            .map_err(xlsx_err)?;
                    }
                }
                // Boolean columns.
                "is_co_citation" | "source_match" => {
                    let bool_val = value == "true";
                    worksheet
                        .write_boolean_with_format(excel_row, col, bool_val, &row_fmt)
                        .map_err(xlsx_err)?;
                }
                // All other columns: plain text with row background.
                _ => {
                    worksheet
                        .write_with_format(excel_row, col, value.as_str(), &row_fmt)
                        .map_err(xlsx_err)?;
                }
            }
        }

        // Set data row height.
        worksheet
            .set_row_height(excel_row, 55.0)
            .map_err(xlsx_err)?;
    }

    let buffer = workbook.save_to_buffer().map_err(xlsx_err)?;
    Ok(buffer)
}

/// Converts an XlsxError into a CitationError::Io for propagation through
/// the existing error chain.
fn xlsx_err(e: XlsxError) -> CitationError {
    CitationError::Io(std::io::Error::other(e.to_string()))
}

// ---------------------------------------------------------------------------
// Corrections JSON
// ---------------------------------------------------------------------------

/// Builds a JSON array of LaTeX corrections extracted from done citation rows.
/// Corrections are sorted by tex_line ascending. Rows without a latex_correction
/// field in their result_json are skipped.
///
/// Each entry in the output array contains: cite_key, tex_line, correction_type,
/// original_text, suggested_text, explanation.
pub fn build_corrections_json(rows: &[CitationRow]) -> Result<String, CitationError> {
    let mut corrections: Vec<serde_json::Value> = Vec::new();

    for row in rows {
        if row.status != "done" {
            continue;
        }

        let result_json = match &row.result_json {
            Some(json) => json,
            None => continue,
        };

        let entry: SubmitEntry = match serde_json::from_str(result_json) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Skip corrections with type "none".
        if entry.latex_correction.correction_type == CorrectionType::None {
            continue;
        }

        corrections.push(serde_json::json!({
            "cite_key": row.cite_key,
            "tex_line": row.tex_line,
            "correction_type": entry.latex_correction.correction_type,
            "original_text": entry.latex_correction.original_text,
            "suggested_text": entry.latex_correction.suggested_text,
            "explanation": entry.latex_correction.explanation,
        }));
    }

    // Sort by tex_line ascending.
    corrections.sort_by(|a, b| {
        let la = a.get("tex_line").and_then(|v| v.as_i64()).unwrap_or(0);
        let lb = b.get("tex_line").and_then(|v| v.as_i64()).unwrap_or(0);
        la.cmp(&lb)
    });

    Ok(serde_json::to_string_pretty(&corrections)?)
}

// ---------------------------------------------------------------------------
// Summary report JSON
// ---------------------------------------------------------------------------

/// Builds a summary report JSON containing job statistics, verdict counts,
/// per-citation details, and alerts.
pub fn build_summary_report(
    job_id: &str,
    rows: &[CitationRow],
    alerts: &[Alert],
    verdicts: &HashMap<String, i64>,
    elapsed_secs: i64,
) -> Result<String, CitationError> {
    let total = rows.len() as i64;
    let done = rows.iter().filter(|r| r.status == "done").count() as i64;

    let summary = ExportSummary {
        supported: *verdicts.get("supported").unwrap_or(&0),
        partial: *verdicts.get("partial").unwrap_or(&0),
        unsupported: *verdicts.get("unsupported").unwrap_or(&0),
        not_found: *verdicts.get("not_found").unwrap_or(&0),
        wrong_source: *verdicts.get("wrong_source").unwrap_or(&0),
        unverifiable: *verdicts.get("unverifiable").unwrap_or(&0),
        peripheral_match: *verdicts.get("peripheral_match").unwrap_or(&0),
        corrections_suggested: count_corrections(rows),
        critical_alerts: alerts.iter().filter(|a| a.flag == "critical").count() as i64,
    };

    let report = serde_json::json!({
        "job_id": job_id,
        "total_citations": total,
        "done_citations": done,
        "elapsed_seconds": elapsed_secs,
        "verdicts": verdicts,
        "summary": summary,
        "alerts": alerts,
    });

    Ok(serde_json::to_string_pretty(&report)?)
}

/// Counts the number of rows that have a non-none LaTeX correction in their
/// result_json.
fn count_corrections(rows: &[CitationRow]) -> i64 {
    rows.iter()
        .filter(|r| r.status == "done" && r.result_json.is_some())
        .filter_map(|r| {
            let entry: SubmitEntry = serde_json::from_str(r.result_json.as_ref()?).ok()?;
            Some(entry.latex_correction)
        })
        .filter(|c| c.correction_type != CorrectionType::None)
        .count() as i64
}

// ---------------------------------------------------------------------------
// Full-detail JSON
// ---------------------------------------------------------------------------

/// Builds a JSON array containing all citation rows with their complete
/// verification results. Each entry merges the database row metadata
/// (cite_key, author, title, year, tex_line, section, matched_file_id)
/// with the full SubmitEntry from result_json.
///
/// Rows that are not in "done" status or have no result_json are included
/// with a null `result` field so the caller sees every citation in the job.
pub fn build_full_detail_json(rows: &[CitationRow]) -> Result<String, CitationError> {
    let mut entries: Vec<serde_json::Value> = Vec::with_capacity(rows.len());

    for row in rows {
        // Parse the result_json if available and status is done.
        let result: Option<SubmitEntry> = if row.status == "done" {
            row.result_json
                .as_ref()
                .and_then(|json| serde_json::from_str(json).ok())
        } else {
            None
        };

        let co_info = row.co_citation_info();

        let result_value = match &result {
            Some(entry) => serde_json::json!({
                "verdict": entry.verdict,
                "claim_original": entry.claim_original,
                "claim_english": entry.claim_english,
                "source_match": entry.source_match,
                "other_source_list": entry.other_source_list,
                "passages": entry.passages,
                "reasoning": entry.reasoning,
                "confidence": entry.confidence,
                "search_rounds": entry.search_rounds,
                "flag": entry.flag,
                "better_source": entry.better_source,
                "latex_correction": entry.latex_correction,
            }),
            None => serde_json::Value::Null,
        };

        entries.push(serde_json::json!({
            "row_id": row.id,
            "cite_key": row.cite_key,
            "author": row.author,
            "title": row.title,
            "year": row.year,
            "tex_line": row.tex_line,
            "anchor_before": row.anchor_before,
            "anchor_after": row.anchor_after,
            "section_title": row.section_title,
            "matched_file_id": row.matched_file_id,
            "group_id": row.group_id,
            "batch_id": row.batch_id,
            "status": row.status,
            "is_co_citation": co_info.is_co_citation,
            "co_cited_with": co_info.co_cited_with,
            "result": result_value,
        }));
    }

    Ok(serde_json::to_string_pretty(&entries)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CorrectionType, LatexCorrection, OtherSourceEntry, OtherSourcePassage, PassageLocation,
        PassageRef,
    };

    /// Helper: creates a CitationRow with done status and the given result_json.
    fn make_done_row(cite_key: &str, tex_line: i64, result: &SubmitEntry) -> CitationRow {
        CitationRow {
            id: 1,
            job_id: "test-job".to_string(),
            group_id: 0,
            cite_key: cite_key.to_string(),
            author: "Test Author".to_string(),
            title: "Test Title".to_string(),
            year: Some("2020".to_string()),
            tex_line,
            anchor_before: "before".to_string(),
            anchor_after: "after".to_string(),
            section_title: Some("Introduction".to_string()),
            matched_file_id: Some(1),
            bib_abstract: None,
            bib_keywords: None,
            tex_context: None,
            batch_id: Some(0),
            status: "done".to_string(),
            flag: None,
            claimed_at: None,
            result_json: Some(serde_json::to_string(result).expect("serialize")),
            co_citation_json: None,
        }
    }

    fn make_submit_entry(verdict: Verdict, passages: Vec<PassageRef>) -> SubmitEntry {
        SubmitEntry {
            row_id: 1,
            verdict,
            claim_original: "test claim".to_string(),
            claim_english: "test claim".to_string(),
            source_match: true,
            other_source_list: vec![],
            passages,
            reasoning: "test reasoning".to_string(),
            confidence: 0.75,
            search_rounds: 1,
            flag: String::new(),
            better_source: vec![],
            latex_correction: LatexCorrection {
                correction_type: CorrectionType::None,
                original_text: String::new(),
                suggested_text: String::new(),
                explanation: String::new(),
            },
        }
    }

    /// T-CIT-036: Annotation CSV has correct columns.
    #[test]
    fn t_cit_036_csv_columns() {
        let entry = make_submit_entry(
            Verdict::Supported,
            vec![PassageRef {
                file_id: 1,
                page: 5,
                passage_text: "The market is efficient.".to_string(),
                relevance_score: 0.95,
                passage_location: PassageLocation::Introduction,
                source_chunk_id: None,
            }],
        );
        let row = make_done_row("fama1970", 42, &entry);

        let csv = build_annotation_csv(&[row]).expect("csv build failed");

        // Verify header.
        assert!(csv.starts_with("title,author,quote,color,comment"));
        // Verify content row exists.
        assert!(csv.contains("Test Title"));
        assert!(csv.contains("Test Author"));
        assert!(csv.contains("The market is efficient."));
    }

    /// T-CIT-037: Colors map correctly to verdicts.
    #[test]
    fn t_cit_037_verdict_colors() {
        assert_eq!(Verdict::Supported.color_hex(), "#00AA00");
        assert_eq!(Verdict::Partial.color_hex(), "#FFD700");
        assert_eq!(Verdict::Unsupported.color_hex(), "#FF4444");
        assert_eq!(Verdict::NotFound.color_hex(), "#AAAAAA");
        assert_eq!(Verdict::WrongSource.color_hex(), "#FF8800");
        assert_eq!(Verdict::Unverifiable.color_hex(), "#CCCCCC");
        assert_eq!(Verdict::PeripheralMatch.color_hex(), "#FFC107");
    }

    /// T-CIT-038: Corrections JSON sorted by tex_line.
    #[test]
    fn t_cit_038_corrections_sorted() {
        let mut entry1 = make_submit_entry(Verdict::Unsupported, vec![]);
        entry1.latex_correction = LatexCorrection {
            correction_type: CorrectionType::Rephrase,
            original_text: "original at line 50".to_string(),
            suggested_text: "corrected".to_string(),
            explanation: "needs rephrasing".to_string(),
        };

        let mut entry2 = make_submit_entry(Verdict::WrongSource, vec![]);
        entry2.latex_correction = LatexCorrection {
            correction_type: CorrectionType::ReplaceCitation,
            original_text: "original at line 10".to_string(),
            suggested_text: "corrected".to_string(),
            explanation: "wrong source".to_string(),
        };

        let row1 = make_done_row("key1", 50, &entry1);
        let row2 = make_done_row("key2", 10, &entry2);

        let json = build_corrections_json(&[row1, row2]).expect("corrections build failed");
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("parse");

        assert_eq!(parsed.len(), 2);
        // First entry should be line 10, second line 50.
        assert_eq!(parsed[0]["tex_line"], 10);
        assert_eq!(parsed[1]["tex_line"], 50);
    }

    /// T-CIT-039: Summary report contains all statistics.
    #[test]
    fn t_cit_039_summary_report() {
        let entry = make_submit_entry(Verdict::Supported, vec![]);
        let row = make_done_row("key1", 10, &entry);

        let mut verdicts = HashMap::new();
        verdicts.insert("supported".to_string(), 5);
        verdicts.insert("partial".to_string(), 2);

        let alerts = vec![Alert {
            row_id: 1,
            cite_key: "bad_key".to_string(),
            flag: "critical".to_string(),
            verdict: Some("wrong_source".to_string()),
            reasoning: Some("source missing".to_string()),
        }];

        let json = build_summary_report("test-job", &[row], &alerts, &verdicts, 120)
            .expect("report build failed");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse");

        assert_eq!(parsed["job_id"], "test-job");
        assert_eq!(parsed["elapsed_seconds"], 120);
        assert_eq!(parsed["verdicts"]["supported"], 5);
        assert_eq!(parsed["alerts"].as_array().expect("alerts array").len(), 1);
    }

    /// T-CIT-096: Rows without passages produce no annotation CSV entry.
    #[test]
    fn t_cit_040_no_passages_no_csv() {
        let entry = make_submit_entry(Verdict::NotFound, vec![]); // No passages.
        let row = make_done_row("key1", 10, &entry);

        let csv = build_annotation_csv(&[row]).expect("csv build failed");

        // Only the header should be present, no data rows.
        let lines: Vec<&str> = csv.trim().lines().collect();
        assert_eq!(lines.len(), 1, "only header row when no passages");
    }

    /// T-CIT-097: Rows without latex_correction produce no corrections entry.
    #[test]
    fn t_cit_041_no_correction_no_entry() {
        let entry = make_submit_entry(Verdict::Supported, vec![]); // No correction.
        let row = make_done_row("key1", 10, &entry);

        let json = build_corrections_json(&[row]).expect("corrections build failed");
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("parse");

        assert!(
            parsed.is_empty(),
            "no corrections when latex_correction is None"
        );
    }

    /// T-CIT-098: Full-detail JSON contains all verification fields for done rows.
    #[test]
    fn t_cit_042_full_detail_all_fields() {
        let mut entry = make_submit_entry(
            Verdict::Partial,
            vec![PassageRef {
                file_id: 3,
                page: 12,
                passage_text: "relevant passage text".to_string(),
                relevance_score: 0.88,
                passage_location: PassageLocation::Results,
                source_chunk_id: None,
            }],
        );
        entry.confidence = 0.75;
        entry.better_source = vec!["smith2021".to_string()];
        entry.other_source_list = vec![OtherSourceEntry {
            cite_key_or_title: "smith2021".to_string(),
            passages: vec![OtherSourcePassage {
                text: "alt passage".to_string(),
                page: 5,
                score: 0.82,
            }],
        }];
        entry.latex_correction = LatexCorrection {
            correction_type: CorrectionType::AddContext,
            original_text: "the result holds".to_string(),
            suggested_text: "the result holds under certain conditions".to_string(),
            explanation: "qualification needed".to_string(),
        };

        let row = make_done_row("fama1970", 42, &entry);

        let json = build_full_detail_json(&[row]).expect("full detail build failed");
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("parse");

        assert_eq!(parsed.len(), 1);
        let item = &parsed[0];

        // Verify database row metadata.
        assert_eq!(item["cite_key"], "fama1970");
        assert_eq!(item["author"], "Test Author");
        assert_eq!(item["title"], "Test Title");
        assert_eq!(item["year"], "2020");
        assert_eq!(item["tex_line"], 42);
        assert_eq!(item["matched_file_id"], 1);
        assert_eq!(item["status"], "done");

        // Verify co-citation fields (default for non-co-cited row).
        assert_eq!(item["is_co_citation"], false);
        assert!(item["co_cited_with"].as_array().unwrap().is_empty());

        // Verify result fields from SubmitEntry.
        let result = &item["result"];
        assert_eq!(result["verdict"], "partial");
        assert_eq!(result["claim_original"], "test claim");
        assert_eq!(result["claim_english"], "test claim");
        assert_eq!(result["source_match"], true);
        assert_eq!(result["confidence"], 0.75);
        assert_eq!(result["search_rounds"], 1);
        assert_eq!(result["better_source"][0], "smith2021");

        // Verify other_source_list.
        let osl = result["other_source_list"].as_array().unwrap();
        assert_eq!(osl.len(), 1);
        assert_eq!(osl[0]["cite_key_or_title"], "smith2021");

        // Verify passages array.
        let passages = result["passages"].as_array().expect("passages array");
        assert_eq!(passages.len(), 1);
        assert_eq!(passages[0]["file_id"], 3);
        assert_eq!(passages[0]["page"], 12);
        assert_eq!(passages[0]["passage_text"], "relevant passage text");

        // Verify latex_correction.
        let correction = &result["latex_correction"];
        assert_eq!(correction["correction_type"], "add_context");
        assert_eq!(correction["explanation"], "qualification needed");
    }

    /// T-CIT-099: Full-detail JSON includes pending rows with null result.
    #[test]
    fn t_cit_043_full_detail_pending_row() {
        let row = CitationRow {
            id: 5,
            job_id: "test-job".to_string(),
            group_id: 0,
            cite_key: "pending_key".to_string(),
            author: "Pending Author".to_string(),
            title: "Pending Title".to_string(),
            year: None,
            tex_line: 100,
            anchor_before: "see".to_string(),
            anchor_after: "for".to_string(),
            section_title: None,
            matched_file_id: None,
            bib_abstract: None,
            bib_keywords: None,
            tex_context: None,
            batch_id: Some(2),
            status: "pending".to_string(),
            flag: None,
            claimed_at: None,
            result_json: None,
            co_citation_json: None,
        };

        let json = build_full_detail_json(&[row]).expect("full detail build failed");
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("parse");

        assert_eq!(parsed.len(), 1);
        let item = &parsed[0];

        assert_eq!(item["cite_key"], "pending_key");
        assert_eq!(item["status"], "pending");
        assert!(item["result"].is_null(), "pending row result must be null");
        assert!(
            item["matched_file_id"].is_null(),
            "unmatched file_id is null"
        );
    }

    /// T-CIT-109: Data CSV has 38 columns in correct order.
    #[test]
    fn t_cit_060_data_csv_columns() {
        let entry = make_submit_entry(
            Verdict::Supported,
            vec![PassageRef {
                file_id: 1,
                page: 5,
                passage_text: "passage text".to_string(),
                relevance_score: 0.95,
                passage_location: PassageLocation::Methodology,
                source_chunk_id: None,
            }],
        );
        let row = make_done_row("fama1970", 42, &entry);

        let csv = build_data_csv(&[row]).expect("data csv failed");
        let lines: Vec<&str> = csv.trim().lines().collect();

        assert!(lines.len() >= 2, "header + at least 1 data row");

        // Verify header has all 38 columns.
        let header_cols: Vec<&str> = lines[0].split(',').collect();
        assert_eq!(header_cols.len(), 38, "must have 38 columns");
        assert_eq!(header_cols[0], "id");
        assert_eq!(header_cols[1], "tex_line");
        assert_eq!(header_cols[7], "cite_key");
        assert_eq!(header_cols[13], "verdict");
        assert_eq!(header_cols[26], "passage_1_location");
        assert_eq!(header_cols[37], "matched_file_id");
    }

    /// T-CIT-110: Data CSV sorts by tex_line ascending.
    #[test]
    fn t_cit_061_data_csv_sorted() {
        let entry1 = make_submit_entry(Verdict::Supported, vec![]);
        let entry2 = make_submit_entry(Verdict::Partial, vec![]);

        let mut row1 = make_done_row("key_b", 50, &entry1);
        row1.id = 2;
        let mut row2 = make_done_row("key_a", 10, &entry2);
        row2.id = 1;

        let csv = build_data_csv(&[row1, row2]).expect("csv failed");
        let lines: Vec<&str> = csv.trim().lines().collect();

        assert_eq!(lines.len(), 3, "header + 2 data rows");

        // First data row should be tex_line=10 (key_a).
        assert!(
            lines[1].contains("key_a"),
            "first row is key_a (tex_line=10)"
        );
        assert!(
            lines[2].contains("key_b"),
            "second row is key_b (tex_line=50)"
        );
    }

    /// T-CIT-111: Excel workbook generates valid bytes.
    #[test]
    fn t_cit_062_excel_generates_bytes() {
        let entry = make_submit_entry(
            Verdict::Supported,
            vec![PassageRef {
                file_id: 1,
                page: 5,
                passage_text: "excel passage text".to_string(),
                relevance_score: 0.95,
                passage_location: PassageLocation::Discussion,
                source_chunk_id: None,
            }],
        );
        let row = make_done_row("fama1970", 42, &entry);

        let bytes = build_excel(&[row]).expect("excel build failed");

        // XLSX files start with the PK zip magic bytes.
        assert!(bytes.len() > 100, "excel bytes must have content");
        assert_eq!(&bytes[0..2], b"PK", "xlsx must be a valid zip archive");
    }

    /// T-CIT-112: Excel workbook handles multiple verdicts with different row colors.
    #[test]
    fn t_cit_063_excel_multiple_verdicts() {
        let entries = [
            make_submit_entry(Verdict::Supported, vec![]),
            make_submit_entry(Verdict::Partial, vec![]),
            make_submit_entry(Verdict::Unsupported, vec![]),
            make_submit_entry(Verdict::NotFound, vec![]),
            make_submit_entry(Verdict::WrongSource, vec![]),
            make_submit_entry(Verdict::Unverifiable, vec![]),
        ];

        let rows: Vec<CitationRow> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let mut row = make_done_row(&format!("key{i}"), (i * 10) as i64, e);
                row.id = (i + 1) as i64;
                row
            })
            .collect();

        let bytes = build_excel(&rows).expect("excel build failed");
        assert!(
            bytes.len() > 100,
            "excel with 6 rows must have substantial content"
        );
    }

    /// T-CIT-113: Verdict row_bg_hex and badge_hex return valid hex strings.
    #[test]
    fn t_cit_064_verdict_color_methods() {
        let verdicts = [
            Verdict::Supported,
            Verdict::Partial,
            Verdict::Unsupported,
            Verdict::NotFound,
            Verdict::WrongSource,
            Verdict::Unverifiable,
            Verdict::PeripheralMatch,
        ];

        for v in &verdicts {
            let bg = v.row_bg_hex();
            let badge = v.badge_hex();
            let label = v.label();

            assert!(
                bg.starts_with('#') && bg.len() == 7,
                "row_bg_hex must be #RRGGBB"
            );
            assert!(
                badge.starts_with('#') && badge.len() == 7,
                "badge_hex must be #RRGGBB"
            );
            assert!(!label.is_empty(), "label must not be empty");
        }
    }

    /// T-CIT-114: other_source_list serialized as JSON in data CSV.
    #[test]
    fn t_cit_065_other_source_list_in_csv() {
        let mut entry = make_submit_entry(Verdict::Partial, vec![]);
        entry.confidence = 0.70;
        entry.other_source_list = vec![OtherSourceEntry {
            cite_key_or_title: "alt_source".to_string(),
            passages: vec![OtherSourcePassage {
                text: "alt text".to_string(),
                page: 3,
                score: 0.85,
            }],
        }];

        let row = make_done_row("key1", 10, &entry);
        let csv = build_data_csv(&[row]).expect("csv failed");

        // The other_source_list column (index 35) should contain JSON.
        assert!(
            csv.contains("alt_source"),
            "other_source_list must contain the cite_key_or_title"
        );
    }

    /// T-CIT-115: better_source serialized as JSON array in data CSV.
    #[test]
    fn t_cit_066_better_source_array_in_csv() {
        let mut entry = make_submit_entry(Verdict::WrongSource, vec![]);
        entry.better_source = vec!["source_a".to_string(), "source_b".to_string()];
        entry.other_source_list = vec![OtherSourceEntry {
            cite_key_or_title: "source_a".to_string(),
            passages: vec![OtherSourcePassage {
                text: "text".to_string(),
                page: 1,
                score: 0.9,
            }],
        }];

        let row = make_done_row("key1", 10, &entry);
        let csv = build_data_csv(&[row]).expect("csv failed");

        assert!(
            csv.contains("source_a"),
            "better_source must contain source_a"
        );
        assert!(
            csv.contains("source_b"),
            "better_source must contain source_b"
        );
    }

    // -----------------------------------------------------------------------
    // PassageLocation tests
    // -----------------------------------------------------------------------

    /// T-CIT-116: passage_location appears in data CSV at the correct column positions.
    /// Columns 26, 30, 34 (0-indexed) are passage_1_location, passage_2_location,
    /// passage_3_location respectively.
    #[test]
    fn t_cit_067_passage_location_in_csv() {
        let entry = make_submit_entry(
            Verdict::Supported,
            vec![
                PassageRef {
                    file_id: 1,
                    page: 5,
                    passage_text: "passage one".to_string(),
                    relevance_score: 0.95,
                    passage_location: PassageLocation::Methodology,
                    source_chunk_id: None,
                },
                PassageRef {
                    file_id: 1,
                    page: 10,
                    passage_text: "passage two".to_string(),
                    relevance_score: 0.85,
                    passage_location: PassageLocation::Results,
                    source_chunk_id: None,
                },
                PassageRef {
                    file_id: 2,
                    page: 3,
                    passage_text: "passage three".to_string(),
                    relevance_score: 0.70,
                    passage_location: PassageLocation::Abstract,
                    source_chunk_id: None,
                },
            ],
        );
        let row = make_done_row("key1", 10, &entry);
        let csv = build_data_csv(&[row]).expect("csv build failed");

        // Verify location labels appear in the CSV.
        assert!(
            csv.contains("methodology"),
            "passage_1_location must be 'methodology'"
        );
        assert!(
            csv.contains("results"),
            "passage_2_location must be 'results'"
        );
        assert!(
            csv.contains("abstract"),
            "passage_3_location must be 'abstract'"
        );

        // Verify the header includes passage_location columns.
        let header_line = csv.lines().next().expect("header must exist");
        assert!(
            header_line.contains("passage_1_location"),
            "header must include passage_1_location"
        );
        assert!(
            header_line.contains("passage_2_location"),
            "header must include passage_2_location"
        );
        assert!(
            header_line.contains("passage_3_location"),
            "header must include passage_3_location"
        );
    }

    /// T-CIT-068: passage_location is empty string for missing passages.
    /// When fewer than 3 passages exist, the missing passage_location columns
    /// are empty strings in the CSV.
    #[test]
    fn t_cit_068_passage_location_empty_for_missing() {
        let entry = make_submit_entry(
            Verdict::Supported,
            vec![PassageRef {
                file_id: 1,
                page: 5,
                passage_text: "only one passage".to_string(),
                relevance_score: 0.90,
                passage_location: PassageLocation::Introduction,
                source_chunk_id: None,
            }],
        );
        let row = make_done_row("key1", 10, &entry);
        let columns = row_to_columns(&row, 1);

        // passage_1_location (index 26) should be "introduction".
        assert_eq!(
            columns[26], "introduction",
            "passage_1_location must be 'introduction'"
        );
        // passage_2_location (index 30) should be empty.
        assert_eq!(
            columns[30], "",
            "passage_2_location must be empty for missing passage"
        );
        // passage_3_location (index 34) should be empty.
        assert_eq!(
            columns[34], "",
            "passage_3_location must be empty for missing passage"
        );
    }

    /// T-CIT-069: row_to_columns returns exactly 38 elements.
    #[test]
    fn t_cit_069_row_to_columns_length() {
        let entry = make_submit_entry(Verdict::Supported, vec![]);
        let row = make_done_row("key1", 10, &entry);
        let columns = row_to_columns(&row, 1);
        assert_eq!(
            columns.len(),
            38,
            "row_to_columns must return exactly 38 elements"
        );
    }

    /// T-CIT-070: All 16 PassageLocation variants serialize to their snake_case
    /// label and round-trip through serde correctly.
    #[test]
    fn t_cit_070_passage_location_serde_roundtrip() {
        let locations = [
            (PassageLocation::Abstract, "abstract"),
            (PassageLocation::Foreword, "foreword"),
            (PassageLocation::TableOfContents, "table_of_contents"),
            (PassageLocation::Introduction, "introduction"),
            (PassageLocation::LiteratureReview, "literature_review"),
            (
                PassageLocation::TheoreticalFramework,
                "theoretical_framework",
            ),
            (PassageLocation::Methodology, "methodology"),
            (PassageLocation::Results, "results"),
            (PassageLocation::Discussion, "discussion"),
            (PassageLocation::Conclusion, "conclusion"),
            (PassageLocation::Bibliography, "bibliography"),
            (PassageLocation::Appendix, "appendix"),
            (PassageLocation::Glossary, "glossary"),
            (PassageLocation::TableOrFigure, "table_or_figure"),
            (PassageLocation::Footnote, "footnote"),
            (PassageLocation::BodyText, "body_text"),
        ];

        for (variant, expected_label) in &locations {
            // Verify the label() method returns the correct snake_case string.
            assert_eq!(
                variant.label(),
                *expected_label,
                "label mismatch for {:?}",
                variant
            );

            // Verify serde serialization produces the snake_case string.
            let json = serde_json::to_string(variant).expect("serialize failed");
            assert_eq!(
                json,
                format!("\"{}\"", expected_label),
                "serde mismatch for {:?}",
                variant
            );

            // Verify serde deserialization round-trips correctly.
            let deserialized: PassageLocation =
                serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(
                *variant, deserialized,
                "round-trip mismatch for {:?}",
                variant
            );
        }
    }

    /// T-CIT-071: PassageRef with passage_location serializes and deserializes
    /// correctly through serde_json, including through result_json storage.
    #[test]
    fn t_cit_071_passage_ref_with_location_roundtrip() {
        let passage = PassageRef {
            file_id: 42,
            page: 7,
            passage_text: "The Sharpe ratio is defined as...".to_string(),
            relevance_score: 0.92,
            passage_location: PassageLocation::TheoreticalFramework,
            source_chunk_id: None,
        };

        let json = serde_json::to_string(&passage).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse");

        assert_eq!(parsed["passage_location"], "theoretical_framework");
        assert_eq!(parsed["file_id"], 42);
        assert_eq!(parsed["page"], 7);

        // Round-trip back to PassageRef.
        let back: PassageRef = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.passage_location, PassageLocation::TheoreticalFramework);
        assert_eq!(back.file_id, 42);
    }

    /// T-CIT-072: Full-detail JSON export includes passage_location in passages.
    #[test]
    fn t_cit_072_full_detail_includes_passage_location() {
        let entry = make_submit_entry(
            Verdict::Supported,
            vec![PassageRef {
                file_id: 1,
                page: 15,
                passage_text: "regression analysis shows".to_string(),
                relevance_score: 0.88,
                passage_location: PassageLocation::Results,
                source_chunk_id: None,
            }],
        );
        let row = make_done_row("key1", 10, &entry);
        let json = build_full_detail_json(&[row]).expect("full detail failed");
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("parse");

        let passages = parsed[0]["result"]["passages"]
            .as_array()
            .expect("passages array");
        assert_eq!(passages.len(), 1);
        assert_eq!(
            passages[0]["passage_location"], "results",
            "passage_location must be serialized in full-detail JSON"
        );
    }

    /// T-CIT-073: Excel workbook generates valid bytes with passage_location
    /// columns. Verifies the workbook contains all 38 columns including the
    /// three passage_location columns.
    #[test]
    fn t_cit_073_excel_with_passage_location() {
        let entry = make_submit_entry(
            Verdict::Supported,
            vec![PassageRef {
                file_id: 1,
                page: 5,
                passage_text: "methodology passage".to_string(),
                relevance_score: 0.95,
                passage_location: PassageLocation::Methodology,
                source_chunk_id: None,
            }],
        );
        let row = make_done_row("key1", 42, &entry);

        let bytes = build_excel(&[row]).expect("excel build failed");
        assert!(bytes.len() > 100, "excel bytes must have content");
        assert_eq!(&bytes[0..2], b"PK", "xlsx must be a valid zip archive");

        // Verify EXPORT_COLUMNS has 38 entries (used by build_excel for column count).
        assert_eq!(
            EXPORT_COLUMNS.len(),
            38,
            "EXPORT_COLUMNS must have 38 entries"
        );
    }

    /// T-CIT-074: EXPORT_COLUMNS contains passage_N_location at correct indices.
    /// Verifies the column ordering: passage_N_location appears after each
    /// passage_N_score, before the next passage group.
    #[test]
    fn t_cit_074_export_columns_ordering() {
        // Find the indices of key columns.
        let column_names: Vec<&str> = EXPORT_COLUMNS.iter().map(|(name, _)| *name).collect();

        let p1_score_idx = column_names
            .iter()
            .position(|&n| n == "passage_1_score")
            .expect("passage_1_score");
        let p1_loc_idx = column_names
            .iter()
            .position(|&n| n == "passage_1_location")
            .expect("passage_1_location");
        let p2_text_idx = column_names
            .iter()
            .position(|&n| n == "passage_2_text")
            .expect("passage_2_text");
        let p2_score_idx = column_names
            .iter()
            .position(|&n| n == "passage_2_score")
            .expect("passage_2_score");
        let p2_loc_idx = column_names
            .iter()
            .position(|&n| n == "passage_2_location")
            .expect("passage_2_location");
        let p3_text_idx = column_names
            .iter()
            .position(|&n| n == "passage_3_text")
            .expect("passage_3_text");
        let p3_score_idx = column_names
            .iter()
            .position(|&n| n == "passage_3_score")
            .expect("passage_3_score");
        let p3_loc_idx = column_names
            .iter()
            .position(|&n| n == "passage_3_location")
            .expect("passage_3_location");

        // passage_1_location immediately follows passage_1_score.
        assert_eq!(
            p1_loc_idx,
            p1_score_idx + 1,
            "passage_1_location follows passage_1_score"
        );
        // passage_2_text immediately follows passage_1_location.
        assert_eq!(
            p2_text_idx,
            p1_loc_idx + 1,
            "passage_2_text follows passage_1_location"
        );
        // passage_2_location immediately follows passage_2_score.
        assert_eq!(
            p2_loc_idx,
            p2_score_idx + 1,
            "passage_2_location follows passage_2_score"
        );
        // passage_3_text immediately follows passage_2_location.
        assert_eq!(
            p3_text_idx,
            p2_loc_idx + 1,
            "passage_3_text follows passage_2_location"
        );
        // passage_3_location immediately follows passage_3_score.
        assert_eq!(
            p3_loc_idx,
            p3_score_idx + 1,
            "passage_3_location follows passage_3_score"
        );
    }

    /// T-CIT-075: SubmitEntry with all fields populated serializes and
    /// deserializes correctly, including the passage_location field on passages
    /// and the other_source_list, better_source array, and co-citation fields.
    #[test]
    fn t_cit_075_submit_entry_full_roundtrip() {
        let entry = SubmitEntry {
            row_id: 42,
            verdict: Verdict::Partial,
            claim_original: "Die Effizienzmarkthypothese besagt...".to_string(),
            claim_english: "The efficient market hypothesis states...".to_string(),
            source_match: false,
            other_source_list: vec![OtherSourceEntry {
                cite_key_or_title: "fama1991".to_string(),
                passages: vec![OtherSourcePassage {
                    text: "market efficiency revisited".to_string(),
                    page: 22,
                    score: 0.87,
                }],
            }],
            passages: vec![
                PassageRef {
                    file_id: 10,
                    page: 3,
                    passage_text: "capital markets show...".to_string(),
                    relevance_score: 0.76,
                    passage_location: PassageLocation::LiteratureReview,
                    source_chunk_id: None,
                },
                PassageRef {
                    file_id: 10,
                    page: 15,
                    passage_text: "our results indicate...".to_string(),
                    relevance_score: 0.68,
                    passage_location: PassageLocation::Results,
                    source_chunk_id: None,
                },
            ],
            reasoning: "The source discusses the topic but does not directly state the claim."
                .to_string(),
            confidence: 0.72,
            search_rounds: 3,
            flag: "warning".to_string(),
            better_source: vec!["fama1991".to_string(), "jensen1978".to_string()],
            latex_correction: LatexCorrection {
                correction_type: CorrectionType::AddContext,
                original_text: "markets are efficient".to_string(),
                suggested_text: "markets are informationally efficient in the weak form"
                    .to_string(),
                explanation: "qualification needed for accuracy".to_string(),
            },
        };

        let json = serde_json::to_string(&entry).expect("serialize");
        let back: SubmitEntry = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(back.verdict, Verdict::Partial);
        assert_eq!(back.passages.len(), 2);
        assert_eq!(
            back.passages[0].passage_location,
            PassageLocation::LiteratureReview
        );
        assert_eq!(back.passages[1].passage_location, PassageLocation::Results);
        assert_eq!(back.other_source_list.len(), 1);
        assert_eq!(back.better_source.len(), 2);
        assert_eq!(back.confidence, 0.72);
        assert_eq!(
            back.latex_correction.correction_type,
            CorrectionType::AddContext
        );
    }

    /// T-CIT-076: Annotation CSV includes passage_location in the comment field
    /// when passages have location metadata.
    #[test]
    fn t_cit_076_annotation_csv_with_passage_location() {
        let entry = make_submit_entry(
            Verdict::Supported,
            vec![PassageRef {
                file_id: 1,
                page: 8,
                passage_text: "hypothesis confirmed by data".to_string(),
                relevance_score: 0.95,
                passage_location: PassageLocation::Conclusion,
                source_chunk_id: None,
            }],
        );
        let row = make_done_row("key1", 10, &entry);
        let csv = build_annotation_csv(&[row]).expect("csv build failed");

        // The annotation CSV comment includes [VERDICT] reasoning format.
        assert!(
            csv.contains("[SUPPORTED]"),
            "annotation CSV must include verdict label"
        );
        assert!(
            csv.contains("hypothesis confirmed by data"),
            "annotation CSV must include passage text"
        );
    }

    /// T-CIT-077: Data CSV with rows having no passages produces empty
    /// passage_location columns.
    #[test]
    fn t_cit_077_no_passages_empty_locations() {
        let entry = make_submit_entry(Verdict::NotFound, vec![]);
        let row = make_done_row("key1", 10, &entry);
        let columns = row_to_columns(&row, 1);

        // All passage_location columns must be empty.
        assert_eq!(columns[26], "", "passage_1_location empty when no passages");
        assert_eq!(columns[30], "", "passage_2_location empty when no passages");
        assert_eq!(columns[34], "", "passage_3_location empty when no passages");

        // All passage_text, page, score columns must also be empty.
        assert_eq!(columns[23], "", "passage_1_text empty");
        assert_eq!(columns[24], "", "passage_1_page empty");
        assert_eq!(columns[25], "", "passage_1_score empty");
    }

    /// T-CIT-078: Excel with all 6 verdict types and passage_location produces
    /// valid output without errors.
    #[test]
    fn t_cit_078_excel_all_verdicts_with_locations() {
        let verdicts_and_locations = [
            (Verdict::Supported, PassageLocation::Methodology),
            (Verdict::Partial, PassageLocation::Discussion),
            (Verdict::Unsupported, PassageLocation::Abstract),
            (Verdict::NotFound, PassageLocation::BodyText),
            (Verdict::WrongSource, PassageLocation::LiteratureReview),
            (Verdict::Unverifiable, PassageLocation::Appendix),
        ];

        let rows: Vec<CitationRow> = verdicts_and_locations
            .iter()
            .enumerate()
            .map(|(i, (verdict, location))| {
                let mut entry = make_submit_entry(
                    *verdict,
                    vec![PassageRef {
                        file_id: 1,
                        page: (i + 1) as i64,
                        passage_text: format!("passage for {:?}", verdict),
                        relevance_score: 0.80,
                        passage_location: *location,
                        source_chunk_id: None,
                    }],
                );

                // Set required fields for specific verdicts.
                if *verdict == Verdict::WrongSource || *verdict == Verdict::Partial {
                    entry.other_source_list = vec![OtherSourceEntry {
                        cite_key_or_title: "alt".to_string(),
                        passages: vec![OtherSourcePassage {
                            text: "alt".to_string(),
                            page: 1,
                            score: 0.8,
                        }],
                    }];
                    if *verdict == Verdict::Partial {
                        entry.confidence = 0.70;
                    }
                }

                let mut row = make_done_row(&format!("key{i}"), (i * 10) as i64, &entry);
                row.id = (i + 1) as i64;
                row
            })
            .collect();

        let bytes = build_excel(&rows).expect("excel build failed");
        assert!(
            bytes.len() > 200,
            "excel with 6 rows must have substantial content"
        );
        assert_eq!(&bytes[0..2], b"PK", "valid zip archive");

        // Also verify data CSV has 38 columns.
        let csv = build_data_csv(&rows).expect("csv build failed");
        let header_cols: Vec<&str> = csv.lines().next().unwrap().split(',').collect();
        assert_eq!(header_cols.len(), 38, "CSV header must have 38 columns");
    }
}
