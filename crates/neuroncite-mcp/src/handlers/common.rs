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

//! Shared utility functions for MCP tool handlers.
//!
//! Contains content truncation and JSON serialization helpers used by the
//! content retrieval handlers (neuroncite_content, neuroncite_batch_content).
//! Extracted from the per-handler copies to eliminate code duplication.

/// Maximum byte length of a single content part in a response. Content
/// exceeding this limit is truncated at a UTF-8 character boundary.
/// 100 KB accommodates all normal document parts while preventing extreme
/// cases where entire PDF text is collapsed into a single page record.
pub const MAX_CONTENT_BYTES: usize = 100_000;

/// Truncates `content` to at most `MAX_CONTENT_BYTES` bytes at a valid UTF-8
/// character boundary. Returns the (possibly truncated) string and a boolean
/// indicating whether truncation occurred.
pub fn truncate_content(content: &str) -> (&str, bool) {
    if content.len() <= MAX_CONTENT_BYTES {
        return (content, false);
    }
    // Find the largest valid UTF-8 boundary at or before MAX_CONTENT_BYTES.
    let mut boundary = MAX_CONTENT_BYTES;
    while boundary > 0 && !content.is_char_boundary(boundary) {
        boundary -= 1;
    }
    (&content[..boundary], true)
}

/// Builds a JSON object for a single content part, applying content truncation.
/// The `part_number` field is the 1-indexed logical unit number (page for PDFs,
/// section for HTML, paragraph block for TXT, etc.).
pub fn content_part_to_json(part_number: i64, content: &str, backend: &str) -> serde_json::Value {
    let (truncated_content, was_truncated) = truncate_content(content);
    let mut json = serde_json::json!({
        "part_number": part_number,
        "content": truncated_content,
        "extraction_backend": backend,
    });
    if was_truncated {
        json["truncated"] = serde_json::json!(true);
        json["original_bytes"] = serde_json::json!(content.len());
    }
    json
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-MCP-COMMON-001: Content within the byte limit is returned unmodified.
    #[test]
    fn t_mcp_common_001_no_truncation_within_limit() {
        let short = "hello world";
        let (result, truncated) = truncate_content(short);
        assert_eq!(result, short);
        assert!(!truncated);
    }

    /// T-MCP-COMMON-002: Content exceeding MAX_CONTENT_BYTES is truncated at a
    /// valid UTF-8 boundary and the truncation flag is set.
    #[test]
    fn t_mcp_common_002_truncation_at_boundary() {
        let large = "A".repeat(MAX_CONTENT_BYTES + 5000);
        let (result, truncated) = truncate_content(&large);
        assert!(truncated);
        assert!(result.len() <= MAX_CONTENT_BYTES);
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
    }

    /// T-MCP-COMMON-003: Truncation respects multi-byte UTF-8 characters and
    /// does not split in the middle of a codepoint.
    #[test]
    fn t_mcp_common_003_truncation_respects_utf8() {
        let emoji = "\u{1F600}"; // 4 bytes per character
        let count = MAX_CONTENT_BYTES / 4 + 10;
        let large: String = emoji.repeat(count);
        let (result, truncated) = truncate_content(&large);
        assert!(truncated);
        assert!(result.len() <= MAX_CONTENT_BYTES);
        assert_eq!(result.len() % 4, 0);
    }

    /// T-MCP-COMMON-004: content_part_to_json includes truncation metadata for
    /// large content.
    #[test]
    fn t_mcp_common_004_json_truncation_metadata() {
        let large = "B".repeat(MAX_CONTENT_BYTES + 1000);
        let json = content_part_to_json(1, &large, "pdfium");
        assert_eq!(json["truncated"], true);
        assert_eq!(json["original_bytes"], large.len());
    }

    /// T-MCP-COMMON-005: content_part_to_json does not include truncation
    /// fields when content is within the limit.
    #[test]
    fn t_mcp_common_005_json_no_truncation() {
        let json = content_part_to_json(42, "test content", "pdf-extract");
        assert!(json.get("truncated").is_none());
        assert_eq!(json["part_number"], 42);
        assert_eq!(json["content"], "test content");
        assert_eq!(json["extraction_backend"], "pdf-extract");
    }

    /// T-MCP-COMMON-006: Content exactly at MAX_CONTENT_BYTES is not truncated.
    #[test]
    fn t_mcp_common_006_exact_boundary_not_truncated() {
        let exact = "A".repeat(MAX_CONTENT_BYTES);
        let (result, truncated) = truncate_content(&exact);
        assert!(!truncated);
        assert_eq!(result.len(), MAX_CONTENT_BYTES);
    }

    /// T-MCP-COMMON-007: Empty content is handled correctly.
    #[test]
    fn t_mcp_common_007_empty_content() {
        let (result, truncated) = truncate_content("");
        assert!(!truncated);
        assert_eq!(result, "");
    }
}
