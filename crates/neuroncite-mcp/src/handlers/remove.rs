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

//! Handler for the `neuroncite_annotation_remove` MCP tool.
//!
//! Removes highlight annotations from a PDF file based on a filter mode:
//! all highlights, highlights matching specific hex colors, or highlights
//! on specific pages. Operates using lopdf (pure Rust) without pdfium.
//!
//! When `output_path` is omitted, the source PDF is overwritten in-place.
//! When `dry_run` is true, the function reports what would be removed
//! without modifying any file.

use std::sync::Arc;

use neuroncite_api::AppState;

/// Removes highlight annotations from a PDF file.
///
/// # Parameters (from MCP tool call)
///
/// - `pdf_path` (required): Absolute path to the PDF file to process.
/// - `mode` (required): Removal mode. One of `"all"`, `"by_color"`, or `"by_page"`.
/// - `output_path` (optional): Absolute path where the modified PDF is saved.
///   When omitted, the source file at `pdf_path` is overwritten.
/// - `colors` (optional): Array of hex color strings (e.g., `["#FF0000", "#FFFF00"]`).
///   Required when `mode` is `"by_color"`. Comparison is case-insensitive.
/// - `pages` (optional): Array of 1-indexed page numbers (e.g., `[1, 3]`).
///   Required when `mode` is `"by_page"`.
/// - `dry_run` (optional): When true, report what would be removed without
///   modifying any file. Default: false.
pub async fn handle(
    _state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let pdf_path_str = params["pdf_path"]
        .as_str()
        .ok_or("missing required parameter: pdf_path")?;

    let mode_str = params["mode"]
        .as_str()
        .ok_or("missing required parameter: mode")?;

    let pdf_path = std::path::PathBuf::from(pdf_path_str);
    if !pdf_path.is_file() {
        return Err(format!(
            "pdf_path is not a file or does not exist: {pdf_path_str}"
        ));
    }

    let output_path = match params["output_path"].as_str() {
        Some(p) => std::path::PathBuf::from(p),
        None => pdf_path.clone(),
    };

    let dry_run = params["dry_run"].as_bool().unwrap_or(false);

    // Parse the removal mode from the mode string and validate required
    // mode-specific parameters.
    let mode = match mode_str {
        "all" => neuroncite_annotate::RemoveMode::All,
        "by_color" => {
            let colors = params["colors"]
                .as_array()
                .ok_or("mode 'by_color' requires a 'colors' array parameter")?
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<String>>();
            if colors.is_empty() {
                return Err("'colors' array must contain at least one hex color string".to_string());
            }
            neuroncite_annotate::RemoveMode::ByColor(colors)
        }
        "by_page" => {
            let pages = params["pages"]
                .as_array()
                .ok_or("mode 'by_page' requires a 'pages' array parameter")?
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect::<Vec<usize>>();
            if pages.is_empty() {
                return Err(
                    "'pages' array must contain at least one page number (1-indexed)".to_string(),
                );
            }
            neuroncite_annotate::RemoveMode::ByPage(pages)
        }
        _ => {
            return Err(format!(
                "invalid mode '{mode_str}': must be 'all', 'by_color', or 'by_page'"
            ));
        }
    };

    // Run the removal in a blocking task because lopdf performs synchronous
    // file I/O that would block the tokio runtime.
    let result = tokio::task::spawn_blocking(move || {
        neuroncite_annotate::remove_highlights(&pdf_path, &output_path, &mode, dry_run)
    })
    .await
    .map_err(|e| format!("removal task panicked: {e}"))?
    .map_err(|e| format!("annotation removal failed: {e}"))?;

    Ok(serde_json::json!({
        "dry_run": dry_run,
        "annotations_removed": result.annotations_removed,
        "pages_affected": result.pages_affected,
        "appearance_objects_cleaned": result.appearance_objects_cleaned,
        "pdf_path": pdf_path_str,
        "output_path": params["output_path"].as_str().unwrap_or(pdf_path_str),
        "mode": mode_str,
    }))
}

#[cfg(test)]
mod tests {
    /// T-MCP-086: Missing pdf_path parameter returns an error string.
    #[test]
    fn t_mcp_086_remove_missing_pdf_path() {
        let params = serde_json::json!({"mode": "all"});
        let pdf_path = params["pdf_path"].as_str();
        assert!(pdf_path.is_none(), "missing pdf_path must return None");
    }

    /// T-MCP-087: Missing mode parameter returns an error string.
    #[test]
    fn t_mcp_087_remove_missing_mode() {
        let params = serde_json::json!({"pdf_path": "/tmp/test.pdf"});
        let mode = params["mode"].as_str();
        assert!(mode.is_none(), "missing mode must return None");
    }

    /// T-MCP-088: Invalid mode string is detected during parameter parsing.
    #[test]
    fn t_mcp_088_remove_invalid_mode() {
        let mode_str = "invalid_mode";
        let is_valid = ["all", "by_color", "by_page"].contains(&mode_str);
        assert!(!is_valid, "invalid_mode must not be in the valid set");
    }

    /// T-MCP-089: dry_run defaults to false when not provided.
    #[test]
    fn t_mcp_089_remove_dry_run_default() {
        let params = serde_json::json!({"pdf_path": "/tmp/test.pdf", "mode": "all"});
        let dry_run = params["dry_run"].as_bool().unwrap_or(false);
        assert!(!dry_run, "dry_run must default to false");
    }

    /// T-MCP-090: output_path defaults to pdf_path when not provided.
    #[test]
    fn t_mcp_090_remove_output_path_default() {
        let params = serde_json::json!({"pdf_path": "/tmp/test.pdf", "mode": "all"});
        let pdf_path = params["pdf_path"].as_str().unwrap();
        let output_path = params["output_path"].as_str().unwrap_or(pdf_path);
        assert_eq!(output_path, "/tmp/test.pdf");
    }

    /// T-MCP-091: by_color mode requires colors parameter.
    #[test]
    fn t_mcp_091_remove_by_color_requires_colors() {
        let params = serde_json::json!({"pdf_path": "/tmp/test.pdf", "mode": "by_color"});
        let colors = params["colors"].as_array();
        assert!(colors.is_none(), "missing colors array must return None");
    }

    /// T-MCP-092: by_page mode requires pages parameter.
    #[test]
    fn t_mcp_092_remove_by_page_requires_pages() {
        let params = serde_json::json!({"pdf_path": "/tmp/test.pdf", "mode": "by_page"});
        let pages = params["pages"].as_array();
        assert!(pages.is_none(), "missing pages array must return None");
    }
}
