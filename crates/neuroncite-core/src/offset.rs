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

// Document-offset-to-page resolution algorithm shared between the chunking
// pipeline (for computing `page_start`/`page_end` on chunks) and the search
// pipeline (for citation generation).
//
// The concatenated document text is defined as:
//   p1 + "\n" + p2 + "\n" + ... + "\n" + pn
// where each pi is the normalized text of page i. The inter-page separator
// is a single newline character (1 UTF-8 byte). There are (n - 1) separators
// for n pages.
//
// The `byte_counts` slice contains the UTF-8 byte length of each page's content
// (excluding the separator). This function maps a document-level byte offset
// back to a 1-indexed page number and a page-local byte offset.

/// Resolves a document-level UTF-8 byte offset to a 1-indexed page number
/// and a page-local byte offset.
///
/// # Arguments
///
/// * `byte_counts` - Per-page byte counts (from the `page.byte_count` column).
///   The length of this slice equals the number of pages in the document.
/// * `doc_offset` - Document-level UTF-8 byte offset into the concatenated
///   text (page1 + "\n" + page2 + "\n" + ... + "\n" + `pageN`).
///
/// # Returns
///
/// A tuple `(page_number, page_local_offset)` where `page_number` is 1-indexed
/// and `page_local_offset` is the byte offset within that page's content.
///
/// # Errors
///
/// Returns `NeuronCiteError::InvalidArgument` if `byte_counts` is empty or
/// if `doc_offset` exceeds the total document length (sum of all page byte
/// counts plus n-1 separator bytes).
///
/// # Examples
///
/// ```
/// use neuroncite_core::offset::resolve_page;
///
/// // Single page of 100 bytes: offset 50 maps to page 1, local offset 50
/// assert_eq!(resolve_page(&[100], 50).unwrap(), (1, 50));
///
/// // Three pages [80, 120, 60] with separators: total = 80+1+120+1+60 = 262
/// // Offset 81 is the first byte of page 2 (80 bytes of page 1 + 1 separator)
/// assert_eq!(resolve_page(&[80, 120, 60], 81).unwrap(), (2, 0));
/// ```
pub fn resolve_page(
    byte_counts: &[usize],
    doc_offset: usize,
) -> Result<(usize, usize), crate::NeuronCiteError> {
    if byte_counts.is_empty() {
        return Err(crate::NeuronCiteError::InvalidArgument(
            "byte_counts must contain at least one page".to_string(),
        ));
    }

    let n = byte_counts.len();

    // Build a prefix-sum array where prefix_sums[i] is the document-level byte
    // offset where page i's content begins. Between consecutive pages there is
    // a 1-byte '\n' separator, so:
    //   prefix_sums[0] = 0
    //   prefix_sums[i] = sum(byte_counts[0..i]) + i   (i separators precede page i)
    // The total document length is sum(byte_counts) + (n - 1).
    let mut prefix_sums = Vec::with_capacity(n);
    let mut acc: usize = 0;
    for (i, &count) in byte_counts.iter().enumerate() {
        prefix_sums.push(acc);
        if i < n - 1 {
            acc += count + 1; // +1 for the '\n' separator between consecutive pages
        } else {
            acc += count; // no separator after the last page
        }
    }
    // acc is the total document length (all page bytes + n-1 separator bytes).

    if doc_offset > acc {
        return Err(crate::NeuronCiteError::InvalidArgument(format!(
            "doc_offset {doc_offset} exceeds total document length {acc}"
        )));
    }

    // Binary search: partition_point returns the first index where
    // prefix_sums[index] > doc_offset. The containing page is one before that.
    let page_index = prefix_sums
        .partition_point(|&start| start <= doc_offset)
        .saturating_sub(1);

    let page_local_offset = doc_offset - prefix_sums[page_index];

    // Return 1-indexed page number and page-local byte offset.
    Ok((page_index + 1, page_local_offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-CORE-009: Single-page document of 100 bytes. Resolving offset 50
    /// returns page 1 with page-local offset 50.
    #[test]
    fn t_core_009_single_page_offset() {
        let byte_counts = [100];
        let (page, local_offset) = resolve_page(&byte_counts, 50).unwrap();
        assert_eq!(page, 1);
        assert_eq!(local_offset, 50);
    }

    /// T-CORE-010: Multi-page document with pages [80, 120, 60] bytes.
    /// The concatenated text is: p1(80) + '\n'(1) + p2(120) + '\n'(1) + p3(60)
    /// Total length = 80 + 1 + 120 + 1 + 60 = 262 bytes.
    /// Offset 81 is the first byte of page 2 (80 bytes for page 1 + 1 separator).
    /// Expected result: page 2, page-local offset 0.
    #[test]
    fn t_core_010_multi_page_boundary() {
        let byte_counts = [80, 120, 60];
        let (page, local_offset) = resolve_page(&byte_counts, 81).unwrap();
        assert_eq!(page, 2);
        assert_eq!(local_offset, 0);
    }

    /// T-CORE-011: Last byte of the document returns the last page with the
    /// correct page-local offset.
    /// Pages [80, 120, 60]: total = 80 + 1 + 120 + 1 + 60 = 262.
    /// Last byte offset = 261 (0-indexed).
    /// Accumulated before page 3: 80 + 1 + 120 + 1 = 202.
    /// Page-local offset = 261 - 202 = 59. Page 3.
    #[test]
    fn t_core_011_last_byte_of_document() {
        let byte_counts = [80, 120, 60];
        // Total document length: 80 + 1 + 120 + 1 + 60 = 262
        // Last valid byte offset: 261
        let last_offset = 80 + 1 + 120 + 1 + 60 - 1; // 261
        let (page, local_offset) = resolve_page(&byte_counts, last_offset).unwrap();
        assert_eq!(page, 3);
        assert_eq!(local_offset, 59); // 261 - 202 = 59
    }

    /// Verifies that offset 0 in a multi-page document maps to page 1,
    /// local offset 0.
    #[test]
    fn first_byte_maps_to_first_page() {
        let byte_counts = [80, 120, 60];
        let (page, local_offset) = resolve_page(&byte_counts, 0).unwrap();
        assert_eq!(page, 1);
        assert_eq!(local_offset, 0);
    }

    /// Verifies that an offset within the middle of page 2 maps correctly.
    /// Pages [80, 120, 60]: page 2 starts at offset 81. Offset 100 is
    /// 100 - 81 = 19 bytes into page 2.
    #[test]
    fn mid_page_offset() {
        let byte_counts = [80, 120, 60];
        let (page, local_offset) = resolve_page(&byte_counts, 100).unwrap();
        assert_eq!(page, 2);
        assert_eq!(local_offset, 19);
    }

    /// T-CORE-023a: Regression test for O(log n) binary search correctness.
    /// Verifies that a 200-page document resolves correctly at page boundaries
    /// and mid-page offsets. The old O(n) linear scan would produce the same
    /// results but in O(n) time; the prefix-sum + partition_point approach
    /// achieves O(log n).
    #[test]
    fn t_core_023a_large_document_binary_search() {
        // 200 pages, each 500 bytes. Separators: 199 newlines.
        // Total = 200 * 500 + 199 = 100_199 bytes.
        let byte_counts = vec![500_usize; 200];
        let total = 200 * 500 + 199;

        // First byte of document: page 1, offset 0.
        assert_eq!(resolve_page(&byte_counts, 0).unwrap(), (1, 0));

        // Last byte of document: page 200, offset 499.
        assert_eq!(resolve_page(&byte_counts, total - 1).unwrap(), (200, 499));

        // Exclusive-end boundary of entire document: page 200, offset 500.
        assert_eq!(resolve_page(&byte_counts, total).unwrap(), (200, 500));

        // First byte of page 100 (0-indexed page 99):
        // prefix_sums[99] = 99 * 501 = 49_599.
        let page_100_start = 99 * 501;
        assert_eq!(
            resolve_page(&byte_counts, page_100_start).unwrap(),
            (100, 0)
        );

        // Mid-page: page 100, offset 250.
        assert_eq!(
            resolve_page(&byte_counts, page_100_start + 250).unwrap(),
            (100, 250)
        );
    }

    /// T-CORE-057b: Separator-boundary offset maps to the end of the preceding
    /// page. The inter-page '\n' separator at position `page_end` is treated as
    /// the exclusive-end offset of the previous page. The old O(n) implementation
    /// had a latent underflow bug on separator offsets; the O(log n) version
    /// handles this correctly.
    #[test]
    fn t_core_057b_separator_boundary_maps_to_page_end() {
        let byte_counts = [80, 120, 60];
        // Byte 80 is the separator between page 1 and page 2.
        // The O(log n) implementation maps this to (page 1, offset 80),
        // which is the exclusive-end of page 1.
        let (page, local_offset) = resolve_page(&byte_counts, 80).unwrap();
        assert_eq!(page, 1);
        assert_eq!(local_offset, 80);
    }

    /// T-CORE-058c: Single-page exclusive-end boundary. For a single page of
    /// 100 bytes, offset 100 (one past end) maps to (1, 100).
    #[test]
    fn t_core_058c_single_page_exclusive_end() {
        let byte_counts = [100];
        let (page, local_offset) = resolve_page(&byte_counts, 100).unwrap();
        assert_eq!(page, 1);
        assert_eq!(local_offset, 100);
    }

    /// T-CORE-058d: Returns Err when doc_offset exceeds document length.
    #[test]
    fn t_core_058d_out_of_bounds_returns_error() {
        let byte_counts = [80, 120, 60];
        // Total = 80 + 1 + 120 + 1 + 60 = 262. Offset 263 is out of bounds.
        let result = resolve_page(&byte_counts, 263);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("exceeds total document length"),
            "error message should mention offset exceeding length: {err_msg}"
        );
    }

    /// T-CORE-058e: Returns Err when byte_counts is empty.
    #[test]
    fn t_core_058e_empty_byte_counts_returns_error() {
        let result = resolve_page(&[], 0);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("at least one page"),
            "error message should mention empty byte_counts: {err_msg}"
        );
    }
}
