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

// CSV and JSON input parsing with automatic format detection.
//
// The annotation pipeline accepts input in two formats:
// - CSV with headers: title, author, quote, color (optional), comment (optional)
// - JSON array of objects with the same fields, or a wrapper object with
//   an "annotations" key containing the array.
//
// Format detection inspects the first non-whitespace byte: '[' or '{'
// indicates JSON, anything else is treated as CSV.

use crate::error::AnnotateError;
use crate::types::{InputFormat, InputRow};

/// Detects the input format by inspecting the first non-whitespace byte.
/// Returns `InputFormat::Json` if the content starts with '[' or '{',
/// otherwise `InputFormat::Csv`.
pub fn detect_format(data: &[u8]) -> InputFormat {
    for &byte in data {
        if byte.is_ascii_whitespace() {
            continue;
        }
        return if byte == b'[' || byte == b'{' {
            InputFormat::Json
        } else {
            InputFormat::Csv
        };
    }
    // Empty or whitespace-only input defaults to CSV (which will fail with
    // a descriptive error during parsing).
    InputFormat::Csv
}

/// Parses annotation input from raw bytes. Auto-detects the format (CSV
/// or JSON) and validates all rows. Returns the parsed rows or an error
/// describing the first validation failure.
pub fn parse_input(data: &[u8]) -> Result<Vec<InputRow>, AnnotateError> {
    let format = detect_format(data);
    let rows = match format {
        InputFormat::Json => parse_json(data)?,
        InputFormat::Csv => parse_csv(data)?,
    };

    if rows.is_empty() {
        return Err(AnnotateError::InputParse(
            "input contains no annotation rows".into(),
        ));
    }

    // Validate each row: required fields must be non-empty, color must be
    // valid hex if present.
    for (i, row) in rows.iter().enumerate() {
        let row_num = i + 1;
        if row.title.trim().is_empty() {
            return Err(AnnotateError::InputParse(format!(
                "row {row_num}: title is empty"
            )));
        }
        if row.author.trim().is_empty() {
            return Err(AnnotateError::InputParse(format!(
                "row {row_num}: author is empty"
            )));
        }
        if row.quote.trim().is_empty() {
            return Err(AnnotateError::InputParse(format!(
                "row {row_num}: quote is empty"
            )));
        }
        if let Some(ref color) = row.color
            && !color.is_empty()
            && !is_valid_hex_color(color)
        {
            return Err(AnnotateError::InputParse(format!(
                "row {row_num}: invalid color '{color}' (expected #RRGGBB)"
            )));
        }
    }

    Ok(rows)
}

/// Parses JSON input. Accepts either a bare array `[{...}]` or a wrapper
/// object `{"annotations": [{...}]}`.
fn parse_json(data: &[u8]) -> Result<Vec<InputRow>, AnnotateError> {
    // Try parsing as a direct array first.
    if let Ok(rows) = serde_json::from_slice::<Vec<InputRow>>(data) {
        return Ok(rows);
    }

    // Try parsing as a wrapper object with an "annotations" key.
    #[derive(serde::Deserialize)]
    struct Wrapper {
        annotations: Vec<InputRow>,
    }

    if let Ok(wrapper) = serde_json::from_slice::<Wrapper>(data) {
        return Ok(wrapper.annotations);
    }

    // Neither format worked; return the array parse error for clarity.
    let err = serde_json::from_slice::<Vec<InputRow>>(data).unwrap_err();
    Err(AnnotateError::InputParse(format!(
        "JSON parse failed: {err}"
    )))
}

/// Parses CSV input with headers. The CSV must have at least the columns
/// "title", "author", and "quote". Columns "color" and "comment" are
/// optional (missing columns are treated as absent).
fn parse_csv(data: &[u8]) -> Result<Vec<InputRow>, AnnotateError> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(data);

    let mut rows = Vec::new();
    for (i, record) in reader.deserialize().enumerate() {
        let row: InputRow =
            record.map_err(|e| AnnotateError::InputParse(format!("CSV row {}: {e}", i + 1)))?;
        rows.push(row);
    }

    Ok(rows)
}

/// Validates a hex color string against the #RRGGBB pattern.
fn is_valid_hex_color(s: &str) -> bool {
    let s = s.trim();
    s.len() == 7 && s.starts_with('#') && s[1..].chars().all(|c| c.is_ascii_hexdigit())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// T-ANNOTATE-001: Parses a CSV with all 5 columns present.
    #[test]
    fn t_annotate_001_parse_csv_all_fields() {
        let csv = b"title,author,quote,color,comment\n\
                     Foo,Bar,Hello world,#FF0000,Test comment";
        let rows = parse_input(csv).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].title, "Foo");
        assert_eq!(rows[0].author, "Bar");
        assert_eq!(rows[0].quote, "Hello world");
        assert_eq!(rows[0].color.as_deref(), Some("#FF0000"));
        assert_eq!(rows[0].comment.as_deref(), Some("Test comment"));
    }

    /// T-ANNOTATE-002: Parses a CSV with only required columns (title,
    /// author, quote). Optional columns color and comment are absent.
    #[test]
    fn t_annotate_002_parse_csv_optional_fields() {
        let csv = b"title,author,quote\nFoo,Bar,Hello world";
        let rows = parse_input(csv).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].color.is_none());
        assert!(rows[0].comment.is_none());
    }

    /// T-ANNOTATE-003: Parses a JSON array input.
    #[test]
    fn t_annotate_003_parse_json_array() {
        let json = br#"[{"title":"Foo","author":"Bar","quote":"Hello"}]"#;
        let rows = parse_input(json).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].title, "Foo");
    }

    /// T-ANNOTATE-004: Parses a JSON wrapper object with "annotations" key.
    #[test]
    fn t_annotate_004_parse_json_wrapper_object() {
        let json = br#"{"annotations":[{"title":"A","author":"B","quote":"C"}]}"#;
        let rows = parse_input(json).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].quote, "C");
    }

    /// T-ANNOTATE-005: Auto-detects CSV format from content.
    #[test]
    fn t_annotate_005_auto_detect_csv() {
        let csv = b"title,author,quote\nA,B,C";
        assert_eq!(detect_format(csv), InputFormat::Csv);
    }

    /// T-ANNOTATE-006: Auto-detects JSON format from content starting
    /// with '['.
    #[test]
    fn t_annotate_006_auto_detect_json() {
        let json = br#"[{"title":"A","author":"B","quote":"C"}]"#;
        assert_eq!(detect_format(json), InputFormat::Json);
    }

    /// T-ANNOTATE-007: Rejects input with an empty title field.
    #[test]
    fn t_annotate_007_reject_empty_title() {
        let csv = b"title,author,quote\n,Bar,Hello";
        let err = parse_input(csv).unwrap_err();
        assert!(err.to_string().contains("title is empty"));
    }

    /// T-ANNOTATE-008: Rejects input with an invalid hex color value.
    #[test]
    fn t_annotate_008_reject_invalid_color() {
        let csv = b"title,author,quote,color\nFoo,Bar,Hello,red";
        let err = parse_input(csv).unwrap_err();
        assert!(err.to_string().contains("invalid color"));
    }

    /// T-ANNOTATE-009: Parses CSV containing Unicode characters in the
    /// quote field (e.g., German umlauts, accented characters).
    #[test]
    fn t_annotate_009_parse_csv_unicode() {
        let csv = "title,author,quote\nFoo,Bar,\u{00FC}ber die Sch\u{00E4}tzung".as_bytes();
        let rows = parse_input(csv).unwrap();
        assert!(rows[0].quote.contains('\u{00FC}'));
    }

    /// T-ANNOTATE-010: Parses inline JSON (as would be sent by an API
    /// request body) with multiple rows.
    #[test]
    fn t_annotate_010_parse_json_inline() {
        let json = br##"[
            {"title":"A","author":"X","quote":"Q1","color":"#00FF00"},
            {"title":"B","author":"Y","quote":"Q2","comment":"Note"}
        ]"##;
        let rows = parse_input(json).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].color.as_deref(), Some("#00FF00"));
        assert!(rows[1].color.is_none());
        assert_eq!(rows[1].comment.as_deref(), Some("Note"));
    }

    /// T-ANNOTATE-011: Rejects empty input (no rows).
    #[test]
    fn t_annotate_011_reject_empty_input() {
        let csv = b"title,author,quote\n";
        let err = parse_input(csv).unwrap_err();
        assert!(err.to_string().contains("no annotation rows"));
    }

    /// T-ANNOTATE-012: JSON with leading whitespace is still detected
    /// as JSON format.
    #[test]
    fn t_annotate_012_detect_json_with_whitespace() {
        let json = b"   [{\"title\":\"A\",\"author\":\"B\",\"quote\":\"C\"}]";
        assert_eq!(detect_format(json), InputFormat::Json);
    }
}
