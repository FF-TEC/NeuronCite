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

//! Chunk-level byte offset tracking for chunk-to-source-text mapping.
//!
//! Provides a convenience function that delegates to the core crate's
//! `resolve_page` to map document-level byte offsets to 1-indexed page
//! numbers. This module bridges the gap between the chunk pipeline's byte
//! offset tracking and the core crate's page resolution algorithm.

use neuroncite_core::offset::resolve_page;

/// Computes the 1-indexed page range for a chunk given its document-level
/// byte offsets and the per-page byte counts.
///
/// Delegates to `neuroncite_core::offset::resolve_page` for both the
/// start and end offsets and returns the resulting page numbers.
///
/// # Arguments
///
/// * `byte_counts` - Per-page byte counts (UTF-8 byte length of each page's
///   content, excluding the inter-page newline separator).
/// * `doc_offset_start` - Document-level byte offset where the chunk begins.
/// * `doc_offset_end` - Document-level byte offset where the chunk ends
///   (exclusive). If this equals the total document length, it resolves to
///   the last page.
///
/// # Returns
///
/// A tuple `(page_start, page_end)` where both values are 1-indexed page
/// numbers. `page_start` is the page containing the first byte of the chunk,
/// `page_end` is the page containing the last byte of the chunk.
///
/// # Errors
///
/// Returns `NeuronCiteError::InvalidArgument` if `byte_counts` is empty or
/// if either offset exceeds the total document length (as defined by
/// `resolve_page`).
pub fn compute_page_range(
    byte_counts: &[usize],
    doc_offset_start: usize,
    doc_offset_end: usize,
) -> Result<(usize, usize), neuroncite_core::NeuronCiteError> {
    let (page_start, _) = resolve_page(byte_counts, doc_offset_start)?;

    // For the end offset, resolve the byte position immediately before the
    // exclusive end. If doc_offset_end equals doc_offset_start (zero-length
    // chunk), both resolve to the same page.
    let end_for_resolve = if doc_offset_end > doc_offset_start {
        doc_offset_end - 1
    } else {
        doc_offset_start
    };
    let (page_end, _) = resolve_page(byte_counts, end_for_resolve)?;

    Ok((page_start, page_end))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-CHK-016: Resolving doc_text_offset_start through cumulative page
    /// byte counts yields the correct page_start value.
    ///
    /// Test setup: three pages with byte counts [80, 120, 60].
    /// Concatenated text layout: p1(80) + '\n'(1) + p2(120) + '\n'(1) + p3(60).
    /// An offset of 100 falls within page 2 (page 2 starts at byte 81).
    #[test]
    fn t_chk_016_offset_to_page_mapping() {
        let byte_counts = [80, 120, 60];

        // Offset 0 is the first byte of page 1
        let (ps, pe) = compute_page_range(&byte_counts, 0, 50).unwrap();
        assert_eq!(ps, 1);
        assert_eq!(pe, 1);

        // Offset 100 is within page 2 (page 2 starts at 81)
        let (ps, pe) = compute_page_range(&byte_counts, 100, 150).unwrap();
        assert_eq!(ps, 2);
        assert_eq!(pe, 2);

        // Offset spanning from page 1 into page 2
        let (ps, pe) = compute_page_range(&byte_counts, 50, 100).unwrap();
        assert_eq!(ps, 1);
        assert_eq!(pe, 2);

        // Offset in the last page (page 3 starts at 202)
        let (ps, pe) = compute_page_range(&byte_counts, 210, 250).unwrap();
        assert_eq!(ps, 3);
        assert_eq!(pe, 3);
    }

    /// Verifies that a chunk spanning all three pages resolves correctly.
    #[test]
    fn full_document_span() {
        let byte_counts = [80, 120, 60];
        // Total document length: 80 + 1 + 120 + 1 + 60 = 262
        let (ps, pe) = compute_page_range(&byte_counts, 0, 262).unwrap();
        assert_eq!(ps, 1);
        assert_eq!(pe, 3);
    }
}
