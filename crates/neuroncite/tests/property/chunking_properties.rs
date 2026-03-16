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

// Property tests for the text chunking subsystem.
//
// These tests verify structural invariants of the chunking pipeline that must
// hold for any valid input: content preservation (no text loss during chunking)
// and non-overlapping byte offset coverage of the document text.
//
// Input generation uses a linear congruential generator (LCG) seeded from
// wall-clock nanoseconds, producing a different pseudo-random sequence on
// each test run. The seed is printed to stderr so that failures can be
// reproduced by hardcoding the value in `make_lcg`. Each property is tested
// across 200 distinct random inputs.

use std::path::PathBuf;

use neuroncite_core::types::{ExtractionBackend, PageText};

// -- Deterministic pseudo-random input generation ----------------------------

/// Deterministic pseudo-random number generator based on a 64-bit linear
/// congruential generator. The constants are from Knuth's MMIX LCG.
struct Lcg {
    state: u64,
}

impl Lcg {
    /// Creates a generator with the given seed value.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the generator state and returns the next 64-bit value.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Returns a pseudo-random value in the range [lo, hi] (inclusive).
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        assert!(hi >= lo);
        let span = hi - lo + 1;
        lo + (self.next_u64() % span)
    }
}

/// Creates an LCG seeded from the current wall-clock nanoseconds.
///
/// Using the current time as a seed means each test run exercises a different
/// sequence of random inputs. The seed value is printed to stderr so that a
/// failing run can be reproduced by replacing the `SystemTime` call with a
/// hardcoded seed of the printed value.
fn make_lcg() -> Lcg {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0xDEAD_BEEF_CAFE_1001);
    eprintln!("LCG seed: {seed:#018x}");
    Lcg::new(seed)
}

/// A fixed vocabulary of words used for generating random page content.
/// Covers a range of word lengths and character types found in academic text.
const WORD_POOL: &[&str] = &[
    "the",
    "of",
    "and",
    "in",
    "to",
    "a",
    "is",
    "that",
    "for",
    "it",
    "was",
    "on",
    "are",
    "as",
    "with",
    "his",
    "they",
    "at",
    "be",
    "this",
    "from",
    "have",
    "or",
    "by",
    "one",
    "had",
    "not",
    "but",
    "what",
    "all",
    "statistical",
    "analysis",
    "regression",
    "coefficient",
    "variance",
    "distribution",
    "hypothesis",
    "probability",
    "sampling",
    "correlation",
    "experiment",
    "observation",
    "measurement",
    "population",
    "inference",
    "Bayesian",
    "frequentist",
    "likelihood",
    "parameter",
    "estimation",
];

/// Generates a random page content string with the given number of words.
fn random_page_content(rng: &mut Lcg, word_count: usize) -> String {
    let mut words = Vec::with_capacity(word_count);
    for _ in 0..word_count {
        let idx = rng.range(0, (WORD_POOL.len() - 1) as u64) as usize;
        words.push(WORD_POOL[idx]);
    }
    words.join(" ")
}

/// Constructs a vector of `PageText` values with random content. The number
/// of pages is in [1, max_pages] and each page has [1, max_words] words.
fn random_pages(rng: &mut Lcg, max_pages: u64, max_words: u64) -> Vec<PageText> {
    let page_count = rng.range(1, max_pages) as usize;
    let mut pages = Vec::with_capacity(page_count);
    for i in 0..page_count {
        let word_count = rng.range(1, max_words) as usize;
        let content = random_page_content(rng, word_count);
        pages.push(PageText {
            source_file: PathBuf::from("/test/property_doc.pdf"),
            page_number: i + 1,
            content,
            backend: ExtractionBackend::PdfExtract,
        });
    }
    pages
}

// -- T-PROP-001 ---------------------------------------------------------------

/// T-PROP-001: For any valid pages input, chunk output preserves all text content.
///
/// This property verifies that the union of all chunk texts, when their words
/// are collected and concatenated, accounts for every word present in the
/// normalized concatenated document text. The word-window strategy splits on
/// whitespace, so the content of all chunks (accounting for overlap) must
/// contain every word from the original text at least once.
///
/// The test iterates over 200 pseudo-random inputs with varying page counts
/// and word counts, using both the "word" and "page" chunking strategies.
/// The LCG seed is printed to stderr for failure reproduction.
#[test]
fn t_prop_001_chunk_output_preserves_all_text_content() {
    let mut rng = make_lcg();

    for trial in 0..200 {
        let pages = random_pages(&mut rng, 5, 100);

        // Test with the "page" strategy: each page becomes one chunk, so
        // the chunk contents must exactly equal the page contents.
        let page_strategy = neuroncite_chunk::create_strategy("page", None, None, None, None)
            .expect("page strategy creation failed");

        let page_chunks = page_strategy.chunk(&pages).expect("page chunking failed");

        // Verify that each page's content is preserved in its corresponding chunk.
        assert_eq!(
            page_chunks.len(),
            pages.len(),
            "trial {trial}: page strategy must produce one chunk per page"
        );
        for (chunk, page) in page_chunks.iter().zip(pages.iter()) {
            assert_eq!(
                chunk.content, page.content,
                "trial {trial}: page chunk content must equal original page content"
            );
        }

        // Test with the "word" strategy: collect all words from chunks and
        // verify they cover all words from the source text.
        let window_size = rng.range(3, 20) as usize;
        let overlap = rng.range(0, (window_size - 1) as u64) as usize;

        let word_strategy =
            neuroncite_chunk::create_strategy("word", Some(window_size), Some(overlap), None, None)
                .expect("word strategy creation failed");

        let word_chunks = word_strategy.chunk(&pages).expect("word chunking failed");

        // Collect all unique words from the chunk output.
        let chunk_words: std::collections::HashSet<&str> = word_chunks
            .iter()
            .flat_map(|c| c.content.split_whitespace())
            .collect();

        // Collect all unique words from the original pages after normalization.
        // The word-window strategy applies normalize_text internally, so we
        // normalize each page and concatenate with "\n" to match the strategy's
        // behavior.
        let concatenated = pages
            .iter()
            .map(|p| p.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let normalized = neuroncite_chunk::normalize::normalize_text(&concatenated);
        let source_words: std::collections::HashSet<&str> = normalized.split_whitespace().collect();

        // Every source word must appear in at least one chunk.
        for word in &source_words {
            assert!(
                chunk_words.contains(word),
                "trial {trial}: word '{word}' from source text is missing from chunk output \
                 (window_size={window_size}, overlap={overlap})"
            );
        }
    }
}

// -- T-PROP-002 ---------------------------------------------------------------

/// T-PROP-002: Chunk byte offsets are non-overlapping and cover the entire document.
///
/// For the "page" strategy, each chunk's byte range [doc_text_offset_start,
/// doc_text_offset_end) corresponds to exactly one page's content within the
/// concatenated document text. The byte ranges must be non-overlapping (each
/// page occupies a distinct region) and the union of all byte ranges plus the
/// inter-page newline separators must cover the full concatenated text length.
///
/// For the "word" strategy with zero overlap, consecutive chunk byte ranges
/// must not overlap (though they may have gaps at whitespace boundaries).
/// With non-zero overlap, chunks share bytes in the overlap regions, but the
/// start of each chunk must advance beyond the start of the previous chunk.
///
/// The test iterates over 200 pseudo-random inputs.
/// The LCG seed is printed to stderr for failure reproduction.
#[test]
fn t_prop_002_chunk_byte_offsets_structural_invariants() {
    let mut rng = make_lcg();

    for trial in 0..200 {
        let pages = random_pages(&mut rng, 5, 80);

        // -- Page strategy: non-overlapping, complete coverage -----------------
        let page_strategy = neuroncite_chunk::create_strategy("page", None, None, None, None)
            .expect("page strategy creation failed");

        let page_chunks = page_strategy.chunk(&pages).expect("page chunking failed");

        // Verify that byte ranges are monotonically increasing (non-overlapping).
        for i in 1..page_chunks.len() {
            assert!(
                page_chunks[i].doc_text_offset_start >= page_chunks[i - 1].doc_text_offset_end,
                "trial {trial}: page chunk {i} offset start ({}) overlaps with chunk {} end ({})",
                page_chunks[i].doc_text_offset_start,
                i - 1,
                page_chunks[i - 1].doc_text_offset_end
            );
        }

        // Verify that the first chunk starts at offset 0 and the last chunk
        // ends at the total concatenated text length.
        assert_eq!(
            page_chunks[0].doc_text_offset_start, 0,
            "trial {trial}: first page chunk must start at offset 0"
        );

        // Compute expected total length: sum of page content lengths plus
        // (n-1) newline separator bytes.
        let total_len: usize =
            pages.iter().map(|p| p.content.len()).sum::<usize>() + pages.len().saturating_sub(1);
        let last_chunk = page_chunks.last().expect("no page chunks");
        assert_eq!(
            last_chunk.doc_text_offset_end, total_len,
            "trial {trial}: last page chunk must end at total text length ({total_len})"
        );

        // -- Word strategy: monotonic offset advancement ----------------------
        let window_size = rng.range(3, 15) as usize;
        let word_strategy = neuroncite_chunk::create_strategy(
            "word",
            Some(window_size),
            Some(0), // zero overlap for non-overlapping guarantee
            None,
            None,
        )
        .expect("word strategy creation failed");

        let word_chunks = word_strategy.chunk(&pages).expect("word chunking failed");

        if word_chunks.len() > 1 {
            // With zero overlap, each chunk's start must be at or beyond the
            // previous chunk's end (no overlapping byte ranges).
            for i in 1..word_chunks.len() {
                assert!(
                    word_chunks[i].doc_text_offset_start >= word_chunks[i - 1].doc_text_offset_end,
                    "trial {trial}: word chunk {i} offset start ({}) must be >= chunk {} end ({})",
                    word_chunks[i].doc_text_offset_start,
                    i - 1,
                    word_chunks[i - 1].doc_text_offset_end
                );
            }
        }

        // Verify monotonic start offset advancement for overlapping strategy.
        let overlap = rng.range(0, (window_size - 1) as u64) as usize;
        let overlap_strategy =
            neuroncite_chunk::create_strategy("word", Some(window_size), Some(overlap), None, None)
                .expect("overlapping strategy creation failed");

        let overlap_chunks = overlap_strategy
            .chunk(&pages)
            .expect("overlapping chunking failed");

        for i in 1..overlap_chunks.len() {
            assert!(
                overlap_chunks[i].doc_text_offset_start
                    > overlap_chunks[i - 1].doc_text_offset_start,
                "trial {trial}: overlapping chunk {i} start ({}) must advance past chunk {} start ({})",
                overlap_chunks[i].doc_text_offset_start,
                i - 1,
                overlap_chunks[i - 1].doc_text_offset_start
            );
        }
    }
}
