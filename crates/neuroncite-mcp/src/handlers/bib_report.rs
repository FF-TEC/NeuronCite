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

// Handler for the `neuroncite_bib_report` MCP tool.
//
// Generates CSV and XLSX report files from a BibTeX file. Parses the .bib
// file, checks file existence in the output directory, and writes a report
// listing all entries with their link type and download status. This handler
// delegates to the same API endpoint logic used by the web frontend's
// "Export Report" button.

use std::sync::Arc;

use neuroncite_api::AppState;

/// Generates CSV and XLSX report files from a BibTeX file. For each entry
/// in the .bib file, the report includes cite_key, title, author, year,
/// link type (URL/DOI/none), download status (exists/missing/no_link),
/// URL, DOI, and the existing filename when applicable.
///
/// # Parameters (from MCP tool call)
///
/// - `bib_path` (required): Absolute path to the .bib file.
/// - `output_directory` (required): Absolute path where report files are
///   written. Also the directory where existing source files are checked
///   for the status column.
pub async fn handle(
    _state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let bib_path = params["bib_path"]
        .as_str()
        .ok_or("missing required parameter: bib_path")?;

    let output_directory = params["output_directory"]
        .as_str()
        .ok_or("missing required parameter: output_directory")?;

    // Validate bib file exists.
    let bib_file = std::path::Path::new(bib_path);
    if !bib_file.is_file() {
        return Err(format!("bib file does not exist: {bib_path}"));
    }

    let output_dir = std::path::Path::new(output_directory);
    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("failed to create output directory: {e}"))?;

    // Parse BibTeX on a blocking thread (CPU-bound text processing).
    let bib_path_owned = bib_path.to_string();
    let bib_entries = tokio::task::spawn_blocking(move || {
        let content = std::fs::read_to_string(&bib_path_owned)
            .map_err(|e| format!("failed to read bib file: {e}"))?;
        Ok::<_, String>(neuroncite_citation::bibtex::parse_bibtex(&content))
    })
    .await
    .map_err(|e| format!("blocking task panicked: {e}"))??;

    // Scan the output directory and its pdf/html subfolders for existing
    // downloaded files. Checks the flat root directory (backward compatibility
    // with sources downloaded before subfolder separation) and the
    // type-separated pdf/ and html/ subfolders.
    let mut existing_files: Vec<(String, String)> = Vec::new();
    for scan_dir in [
        output_dir.to_path_buf(),
        output_dir.join("pdf"),
        output_dir.join("html"),
    ] {
        if scan_dir.is_dir()
            && let Ok(rd) = std::fs::read_dir(&scan_dir)
        {
            for entry in rd.filter_map(|e| e.ok()) {
                let name = entry.file_name().to_string_lossy().to_string();
                let lower = name.to_lowercase();
                if lower.ends_with(".pdf") || lower.ends_with(".html") {
                    existing_files.push((lower, name));
                }
            }
        }
    }

    // Build report row data sorted by cite_key.
    struct ReportEntry {
        cite_key: String,
        title: String,
        author: String,
        year: String,
        link_type: String,
        status: String,
        url: String,
        doi: String,
        existing_file: String,
    }

    let mut rows: Vec<ReportEntry> = bib_entries
        .into_iter()
        .map(|(cite_key, entry)| {
            let link_type = if entry.url.is_some() {
                "URL"
            } else if entry.doi.is_some() {
                "DOI"
            } else {
                ""
            };

            let (file_exists, existing_file) = if !existing_files.is_empty() {
                let expected = neuroncite_html::build_source_filename(
                    &entry.title,
                    &entry.author,
                    entry.year.as_deref(),
                    &cite_key,
                )
                .to_lowercase();
                let cite_lower = cite_key.to_lowercase();

                let found = existing_files.iter().find(|(lower, _original)| {
                    let stem = lower
                        .strip_suffix(".pdf")
                        .or_else(|| lower.strip_suffix(".html"))
                        .unwrap_or(lower);
                    stem == expected || stem == cite_lower || lower.contains(&cite_lower)
                });
                match found {
                    Some((_lower, original)) => (true, original.clone()),
                    None => (false, String::new()),
                }
            } else {
                (false, String::new())
            };

            let status = if file_exists {
                "exists"
            } else if entry.url.is_some() || entry.doi.is_some() {
                "missing"
            } else {
                "no_link"
            };

            ReportEntry {
                cite_key,
                title: entry.title,
                author: entry.author,
                year: entry.year.unwrap_or_default(),
                link_type: link_type.to_string(),
                status: status.to_string(),
                url: entry.url.unwrap_or_default(),
                doi: entry.doi.unwrap_or_default(),
                existing_file,
            }
        })
        .collect();

    rows.sort_by(|a, b| a.cite_key.cmp(&b.cite_key));

    let total_entries = rows.len();
    let existing_count = rows.iter().filter(|r| r.status == "exists").count();
    let missing_count = rows.iter().filter(|r| r.status == "missing").count();
    let no_link_count = rows.iter().filter(|r| r.status == "no_link").count();

    // Column definitions: (header_name, xlsx_width).
    const COLUMNS: &[(&str, f64)] = &[
        ("cite_key", 22.0),
        ("title", 50.0),
        ("author", 35.0),
        ("year", 8.0),
        ("link_type", 10.0),
        ("status", 12.0),
        ("url", 45.0),
        ("doi", 25.0),
        ("existing_file", 45.0),
    ];

    // --- CSV report ---
    let csv_escape = |s: &str| -> String {
        if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    };

    let mut csv = String::new();
    csv.push('\u{FEFF}');
    csv.push_str(
        &COLUMNS
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(","),
    );
    csv.push('\n');

    for row in &rows {
        let fields = [
            &row.cite_key,
            &row.title,
            &row.author,
            &row.year,
            &row.link_type,
            &row.status,
            &row.url,
            &row.doi,
            &row.existing_file,
        ];
        csv.push_str(
            &fields
                .iter()
                .map(|f| csv_escape(f))
                .collect::<Vec<_>>()
                .join(","),
        );
        csv.push('\n');
    }

    let csv_path = output_dir.join("bib_report.csv");
    std::fs::write(&csv_path, &csv).map_err(|e| format!("failed to write CSV report: {e}"))?;

    // --- XLSX report ---
    let xlsx_path = output_dir.join("bib_report.xlsx");
    let xlsx_result = (|| -> Result<(), rust_xlsxwriter::XlsxError> {
        use rust_xlsxwriter::{Color, Format, FormatAlign, FormatBorder, Workbook};

        let mut workbook = Workbook::new();
        let ws = workbook.add_worksheet();
        ws.set_name("BibTeX Report")?;

        let border_color = Color::RGB(0xBBBBBB);

        let header_fmt = Format::new()
            .set_background_color(Color::RGB(0x1F3864))
            .set_font_color(Color::White)
            .set_bold()
            .set_font_size(10.0)
            .set_align(FormatAlign::Center)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        for (col, (name, width)) in COLUMNS.iter().enumerate() {
            ws.set_column_width(col as u16, *width)?;
            ws.write_with_format(0, col as u16, *name, &header_fmt)?;
        }
        ws.set_freeze_panes(1, 0)?;

        let exists_fmt = Format::new()
            .set_background_color(Color::RGB(0xE0F7FA))
            .set_font_size(10.0)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color)
            .set_text_wrap();

        let missing_fmt = Format::new()
            .set_background_color(Color::RGB(0xFCE4EC))
            .set_font_size(10.0)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color)
            .set_text_wrap();

        let no_link_fmt = Format::new()
            .set_background_color(Color::RGB(0xF5F5F5))
            .set_font_size(10.0)
            .set_font_color(Color::RGB(0x999999))
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color)
            .set_text_wrap();

        let exists_badge = Format::new()
            .set_background_color(Color::RGB(0x00897B))
            .set_font_color(Color::White)
            .set_bold()
            .set_font_size(10.0)
            .set_align(FormatAlign::Center)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        let missing_badge = Format::new()
            .set_background_color(Color::RGB(0xD81B60))
            .set_font_color(Color::White)
            .set_bold()
            .set_font_size(10.0)
            .set_align(FormatAlign::Center)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        let no_link_badge = Format::new()
            .set_background_color(Color::RGB(0xBDBDBD))
            .set_font_color(Color::White)
            .set_bold()
            .set_font_size(10.0)
            .set_align(FormatAlign::Center)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        for (row_idx, row) in rows.iter().enumerate() {
            let excel_row = (row_idx + 1) as u32;
            let row_fmt = match row.status.as_str() {
                "exists" => &exists_fmt,
                "missing" => &missing_fmt,
                _ => &no_link_fmt,
            };

            let fields = [
                &row.cite_key,
                &row.title,
                &row.author,
                &row.year,
                &row.link_type,
                &row.status,
                &row.url,
                &row.doi,
                &row.existing_file,
            ];

            for (col, value) in fields.iter().enumerate() {
                let col16 = col as u16;
                if col == 5 {
                    let badge = match row.status.as_str() {
                        "exists" => &exists_badge,
                        "missing" => &missing_badge,
                        _ => &no_link_badge,
                    };
                    ws.write_with_format(excel_row, col16, value.as_str(), badge)?;
                } else {
                    ws.write_with_format(excel_row, col16, value.as_str(), row_fmt)?;
                }
            }
            ws.set_row_height(excel_row, 30.0)?;
        }

        // Summary sheet.
        let ws2 = workbook.add_worksheet();
        ws2.set_name("Summary")?;

        let label_fmt = Format::new()
            .set_bold()
            .set_align(FormatAlign::Left)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        let value_fmt = Format::new()
            .set_align(FormatAlign::Right)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        let title_fmt = Format::new()
            .set_background_color(Color::RGB(0x1F3864))
            .set_font_color(Color::White)
            .set_bold()
            .set_font_size(14.0)
            .set_align(FormatAlign::Center)
            .set_border(FormatBorder::Thin)
            .set_border_color(border_color);

        ws2.set_column_width(0, 28.0)?;
        ws2.set_column_width(1, 16.0)?;
        ws2.merge_range(0, 0, 0, 1, "BibTeX Report Summary", &title_fmt)?;

        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

        let summary_rows: &[(&str, String)] = &[
            ("Generated", timestamp),
            ("BibTeX file", bib_path.to_string()),
            ("Output directory", output_directory.to_string()),
            ("", String::new()),
            ("Total entries", total_entries.to_string()),
            (
                "Entries with URL/DOI",
                (existing_count + missing_count).to_string(),
            ),
            ("Entries without URL/DOI", no_link_count.to_string()),
            ("Already downloaded (exists)", existing_count.to_string()),
            ("Missing (not yet downloaded)", missing_count.to_string()),
        ];

        for (idx, (label, value)) in summary_rows.iter().enumerate() {
            let r = (idx + 1) as u32;
            if label.is_empty() {
                continue;
            }
            ws2.write_with_format(r, 0, *label, &label_fmt)?;
            ws2.write_with_format(r, 1, value.as_str(), &value_fmt)?;
        }

        workbook.save(&xlsx_path)?;
        Ok(())
    })();

    if let Err(e) = xlsx_result {
        return Err(format!("failed to write XLSX report: {e}"));
    }

    tracing::info!(
        csv = %csv_path.display(),
        xlsx = %xlsx_path.display(),
        entries = total_entries,
        "bib report generated"
    );

    Ok(serde_json::json!({
        "csv_path": csv_path.display().to_string(),
        "xlsx_path": xlsx_path.display().to_string(),
        "total_entries": total_entries,
        "existing_count": existing_count,
        "missing_count": missing_count,
        "no_link_count": no_link_count,
    }))
}
