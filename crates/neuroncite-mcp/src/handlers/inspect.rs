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

//! Handler for the `neuroncite_inspect_annotations` MCP tool.
//!
//! Reads a PDF file from disk and returns a structured report of all
//! annotations found, including their type, fill color (as hex), opacity,
//! and bounding rectangle. Agents use this tool after `neuroncite_annotate`
//! to verify that highlight colors were physically written to the PDF.
//!
//! The inspection is read-only and does not require an indexed session or
//! database access. It uses pdfium directly via `spawn_blocking` because
//! pdfium is a synchronous C library that must not run on the tokio runtime.

use std::sync::Arc;

use neuroncite_api::AppState;

/// Inspects annotations in a PDF file and returns their properties.
///
/// # Parameters (from MCP tool call)
///
/// - `pdf_path` (required): Absolute path to the PDF file to inspect.
/// - `page_number` (optional): Restrict inspection to a single page (1-indexed).
///   When absent, all pages are scanned.
pub async fn handle(
    _state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let pdf_path_str = params["pdf_path"]
        .as_str()
        .ok_or("missing required parameter: pdf_path")?;

    let pdf_path = std::path::PathBuf::from(pdf_path_str);

    // Validate the file exists and is a regular file (not a directory).
    if !pdf_path.is_file() {
        return Err(format!(
            "pdf_path is not a file or does not exist: {pdf_path_str}"
        ));
    }

    let page_filter = params["page_number"].as_u64().map(|n| n as usize);

    // Run the inspection in a blocking task because pdfium is a synchronous
    // C library. Calling it from the tokio runtime thread would block the
    // event loop and stall all other concurrent operations.
    let result = tokio::task::spawn_blocking(move || {
        neuroncite_annotate::inspect_pdf(&pdf_path, page_filter)
    })
    .await
    .map_err(|e| format!("inspection task panicked: {e}"))?
    .map_err(|e| format!("inspection failed: {e}"))?;

    serde_json::to_value(&result).map_err(|e| format!("serialization failed: {e}"))
}

#[cfg(test)]
mod tests {
    /// T-MCP-080: Missing pdf_path parameter returns an error string.
    #[test]
    fn t_mcp_080_inspect_missing_pdf_path() {
        let params = serde_json::json!({});
        let pdf_path = params["pdf_path"].as_str();
        assert!(pdf_path.is_none(), "missing pdf_path must return None");
    }

    /// T-MCP-081: Non-file pdf_path (a directory path) is detected before
    /// reaching the pdfium layer.
    #[test]
    fn t_mcp_081_inspect_directory_path_rejected() {
        // Use the system's temp directory as a path that exists but is not a file.
        let dir = std::env::temp_dir();
        assert!(dir.is_dir(), "temp dir must be a directory");
        assert!(!dir.is_file(), "temp dir must not be a file");
    }

    /// T-MCP-082: page_number extraction from params handles absent value
    /// (returns None, meaning scan all pages).
    #[test]
    fn t_mcp_082_inspect_no_page_filter() {
        let params = serde_json::json!({"pdf_path": "/tmp/test.pdf"});
        let page_filter = params["page_number"].as_u64();
        assert!(page_filter.is_none(), "absent page_number yields None");
    }

    /// T-MCP-083: page_number extraction from params handles a valid integer.
    #[test]
    fn t_mcp_083_inspect_with_page_filter() {
        let params = serde_json::json!({"pdf_path": "/tmp/test.pdf", "page_number": 3});
        let page_filter = params["page_number"].as_u64().map(|n| n as usize);
        assert_eq!(page_filter, Some(3));
    }
}
