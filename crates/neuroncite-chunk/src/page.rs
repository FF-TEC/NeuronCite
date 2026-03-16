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

//! Page-based chunking strategy.
//!
//! Treats each PDF page's extracted text as a single chunk. This is the
//! simplest strategy and produces no overlap between chunks. Each `PageText`
//! value becomes one `Chunk` with its full text content.
//!
//! The `chunk_index` is assigned as the sequential position (0 through n-1)
//! and the byte offsets track each page's position within the concatenated
//! document text (pages joined by "\n").

use neuroncite_core::error::NeuronCiteError;
use neuroncite_core::traits::ChunkStrategy;
use neuroncite_core::types::{Chunk, PageText};

/// Page-based chunking strategy that converts each PDF page into one chunk.
///
/// This strategy does not split or merge pages. The number of output chunks
/// equals the number of input pages. Byte offsets are computed relative to
/// the concatenated document text where pages are joined by a single newline
/// character.
pub struct PageStrategy;

impl ChunkStrategy for PageStrategy {
    /// Converts each `PageText` into a single `Chunk`.
    ///
    /// The concatenated document text is defined as:
    ///
    /// ```text
    /// page1 + "\n" + page2 + "\n" + ... + "\n" + pageN
    /// ```
    ///
    /// Each chunk's `doc_text_offset_start` and `doc_text_offset_end` correspond
    /// to the byte range of that page's content within this concatenated string.
    ///
    /// # Errors
    ///
    /// Returns `NeuronCiteError::Chunk` if the input pages slice is empty.
    fn chunk(&self, pages: &[PageText]) -> Result<Vec<Chunk>, NeuronCiteError> {
        if pages.is_empty() {
            return Err(NeuronCiteError::Chunk(
                "cannot chunk an empty page list".into(),
            ));
        }

        let source_file = pages[0].source_file.clone();
        let mut chunks = Vec::with_capacity(pages.len());

        // accumulated_offset tracks the document-level byte position where the
        // current page's content starts in the concatenated document text.
        let mut accumulated_offset: usize = 0;

        for (index, page) in pages.iter().enumerate() {
            let content = &page.content;
            let byte_len = content.len();

            let content_hash = Chunk::compute_content_hash(content);

            chunks.push(Chunk {
                source_file: source_file.clone(),
                page_start: page.page_number,
                page_end: page.page_number,
                chunk_index: index,
                doc_text_offset_start: accumulated_offset,
                doc_text_offset_end: accumulated_offset + byte_len,
                content: content.clone(),
                content_hash,
            });

            // Advance past the current page's content. If this is not the
            // last page, add 1 byte for the "\n" separator between pages.
            accumulated_offset += byte_len;
            if index < pages.len() - 1 {
                accumulated_offset += 1; // inter-page newline separator
            }
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_pages;

    /// T-CHK-001: Given n pages, page-based chunking produces exactly n chunks.
    #[test]
    fn t_chk_001_one_chunk_per_page() {
        let pages = make_pages(&["Page one text.", "Page two text.", "Page three."]);
        let strategy = PageStrategy;
        let chunks = strategy.chunk(&pages).expect("chunking failed");
        assert_eq!(chunks.len(), pages.len());
    }

    /// T-CHK-002: The `chunk_index` values form a contiguous sequence from
    /// 0 to n-1.
    #[test]
    fn t_chk_002_sequential_chunk_index() {
        let pages = make_pages(&[
            "First page content.",
            "Second page content.",
            "Third page content.",
            "Fourth page content.",
        ]);
        let strategy = PageStrategy;
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i, "chunk_index mismatch at position {i}");
        }
    }

    /// Verifies that byte offsets are computed correctly for multi-page
    /// documents with inter-page newline separators.
    #[test]
    fn byte_offsets_correct() {
        let pages = make_pages(&["AAAA", "BBBB", "CC"]);
        let strategy = PageStrategy;
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        // Page 1: "AAAA" (4 bytes), offset 0..4
        assert_eq!(chunks[0].doc_text_offset_start, 0);
        assert_eq!(chunks[0].doc_text_offset_end, 4);

        // Page 2: "BBBB" (4 bytes), starts at 4 + 1 (newline) = 5, offset 5..9
        assert_eq!(chunks[1].doc_text_offset_start, 5);
        assert_eq!(chunks[1].doc_text_offset_end, 9);

        // Page 3: "CC" (2 bytes), starts at 9 + 1 = 10, offset 10..12
        assert_eq!(chunks[2].doc_text_offset_start, 10);
        assert_eq!(chunks[2].doc_text_offset_end, 12);
    }

    /// Verifies that `page_start` and `page_end` match the original page
    /// numbers.
    #[test]
    fn page_numbers_correct() {
        let pages = make_pages(&["A", "B", "C"]);
        let strategy = PageStrategy;
        let chunks = strategy.chunk(&pages).expect("chunking failed");

        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.page_start, i + 1);
            assert_eq!(chunk.page_end, i + 1);
        }
    }

    /// Verifies that chunking an empty page list returns an error.
    #[test]
    fn empty_pages_returns_error() {
        let strategy = PageStrategy;
        let result = strategy.chunk(&[]);
        assert!(result.is_err());
    }
}
