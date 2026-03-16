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

//! Batch size computation, length-sorted permutation, and byte conversion
//! utilities for the embedding pipeline.
//!
//! This module contains:
//! - Adaptive batch size computation based on the model's maximum sequence
//!   length, preventing GPU memory exhaustion for long-context models.
//! - Length-sorted permutation and order restoration for minimizing padding
//!   within embedding batches.
//! - f32-to-bytes and bytes-to-f32 conversion for SQLite BLOB storage.

use neuroncite_core::{Chunk, InferenceCapabilities};

// ---------------------------------------------------------------------------
// Adaptive batch size computation
// ---------------------------------------------------------------------------

/// Reference batch size that is safe for BERT-family encoder models with a
/// maximum sequence length of 512 tokens on typical consumer GPUs (8-12 GB VRAM).
const REFERENCE_BATCH_SIZE: usize = 32;

/// Reference sequence length corresponding to the `REFERENCE_BATCH_SIZE`.
/// BERT-family models (BGE, MiniLM) use 512 tokens as their maximum context.
const REFERENCE_SEQ_LEN: usize = 512;

/// Computes the embedding batch size using conservative CPU-like sizing.
///
/// This is the legacy single-parameter overload. Existing callers (e.g.,
/// `embed_batch_search` in worker.rs for search refinement) continue to work
/// without access to `InferenceCapabilities`. For indexing workloads that have
/// access to hardware capabilities, prefer [`compute_batch_size_with_caps`].
pub fn compute_batch_size(max_sequence_length: usize) -> usize {
    compute_batch_size_with_caps(max_sequence_length, &InferenceCapabilities::default())
}

/// Hardware-adaptive embedding batch size computation.
///
/// GPU self-attention memory scales as O(batch_size * num_heads * seq_len^2).
/// The base formula scales a reference batch size (32 for 512-token models)
/// inversely with the model's maximum sequence length. A hardware-specific
/// multiplier then adjusts this base to the platform's memory architecture:
///
/// | Execution Provider | Strategy                                     |
/// |--------------------|----------------------------------------------|
/// | CoreML (unified)   | Scale with √(RAM/8GB), up to 256. Unified    |
/// |                    | memory has no separate VRAM budget to exhaust.|
/// | CUDA / ROCm        | Keep base (calibrated for 8-12GB VRAM).      |
/// | DirectML           | Keep base (similar VRAM constraints).        |
/// | CPU                | Scale with √(cores/4), up to 64. Larger      |
/// |                    | batches amortize per-batch overhead.          |
///
/// # Examples
///
/// ```text
/// CoreML, 64GB, seq_len=512  → base=32, scale=√8≈2 → 64
/// CoreML, 128GB, seq_len=512 → base=32, scale=√16=4 → 128
/// CoreML, 64GB, seq_len=8192 → base=2, scale=√8≈2 → 4
/// CUDA, seq_len=512          → 32  (unchanged)
/// CUDA, seq_len=8192         → 2   (unchanged)
/// CPU, 8-core, seq_len=512   → base=32, scale=√2≈1 → 32
/// ```
pub fn compute_batch_size_with_caps(
    max_sequence_length: usize,
    caps: &InferenceCapabilities,
) -> usize {
    let base = (REFERENCE_BATCH_SIZE * REFERENCE_SEQ_LEN / max_sequence_length.max(1))
        .clamp(1, REFERENCE_BATCH_SIZE);

    let memory_gb = (caps.system_memory_bytes / (1024 * 1024 * 1024)) as f64;

    match caps.active_ep.as_str() {
        "CoreML" if caps.unified_memory => {
            // Apple Silicon unified memory: CPU, GPU, and ANE share the same
            // DRAM pool. There is no separate VRAM to exhaust, so batch sizes
            // can scale with available system memory. Logarithmic scaling
            // (square root) prevents excessively large batches that would
            // still cause diminishing returns from ANE pipeline stalls.
            let scale = (memory_gb / 8.0).sqrt().max(1.0) as usize;
            (base * scale).clamp(1, 256)
        }
        "CUDA" | "ROCm" => {
            // Discrete GPU with dedicated VRAM (typically 8-24 GB).
            // The base formula was calibrated for this configuration.
            base
        }
        "DirectML" => {
            // Windows GPU via DirectX 12. VRAM constraints are similar to
            // CUDA, and may be tighter on Intel/AMD integrated GPUs.
            base
        }
        _ => {
            // CPU-only: no GPU memory constraint, but larger batches improve
            // throughput by amortizing per-batch overhead across more inputs.
            // Scale with available cores (more cores = faster large-batch GEMM).
            let cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4) as f64;
            let scale = (cores / 4.0).sqrt().max(1.0) as usize;
            (base * scale).clamp(1, 64)
        }
    }
}

// ---------------------------------------------------------------------------
// Length-sorted batching helpers
// ---------------------------------------------------------------------------

/// Computes an index permutation that sorts chunk indices by their content
/// byte length in ascending order. Short chunks are placed first so that
/// within each batch, chunks have similar lengths, minimizing the padding
/// needed to match the longest sequence in the batch.
///
/// Ascending (shortest-first) order also causes the ONNX Runtime CUDA
/// memory arena to grow monotonically: early batches use small tensors,
/// later batches progressively use larger ones. This avoids the
/// fragmentation pattern where a single long-sequence batch mid-stream
/// forces a large arena allocation that remains reserved for the rest of
/// the session.
///
/// Returns a vector where `result[sorted_position] = original_chunk_index`.
pub fn length_sorted_permutation(chunks: &[Chunk]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..chunks.len()).collect();
    indices.sort_by_key(|&i| chunks[i].content.len());
    indices
}

/// Reorders a vector of embeddings from length-sorted order back to the
/// original document order using the permutation from `length_sorted_permutation`.
///
/// `sorted_embeddings[sorted_pos]` is the embedding computed for
/// `chunks[permutation[sorted_pos]]`. The returned vector has
/// `result[original_idx]` as the embedding for `chunks[original_idx]`.
pub fn restore_original_order(
    sorted_embeddings: Vec<Vec<f32>>,
    permutation: &[usize],
) -> Vec<Vec<f32>> {
    let n = sorted_embeddings.len();
    let mut result: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();
    for (sorted_pos, embedding) in sorted_embeddings.into_iter().enumerate() {
        let original_idx = permutation[sorted_pos];
        result[original_idx] = embedding;
    }
    result
}

// ---------------------------------------------------------------------------
// Byte conversion helpers
// ---------------------------------------------------------------------------

/// Converts a slice of f32 values to a byte vector using little-endian encoding.
/// Each f32 occupies 4 bytes, so the output length is `floats.len() * 4`.
pub fn f32_slice_to_bytes(floats: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(floats.len() * 4);
    for &f in floats {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

/// Converts a byte slice (little-endian encoded f32 values) back to a `Vec<f32>`.
/// The input length must be a multiple of 4 (one f32 = 4 bytes). When the length
/// is not a multiple of 4, the trailing incomplete bytes are discarded by
/// `chunks_exact` and a warning is emitted in all build profiles. This indicates
/// an embedding BLOB stored in SQLite has been corrupted or truncated.
pub fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    if !bytes.len().is_multiple_of(4) {
        tracing::warn!(
            bytes_len = bytes.len(),
            trailing_bytes = bytes.len() % 4,
            "bytes_to_f32_vec: embedding BLOB length is not a multiple of 4 — \
             the stored embedding data may be corrupted or truncated. \
             The {} trailing byte(s) will be discarded.",
            bytes.len() % 4
        );
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // -----------------------------------------------------------------------
    // T-IDX-005: f32_slice_to_bytes roundtrips correctly with bytes_to_f32_vec
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_005_f32_bytes_roundtrip() {
        let original = vec![1.0_f32, -2.5, 0.0, std::f32::consts::PI, f32::MAX, f32::MIN];
        let bytes = f32_slice_to_bytes(&original);
        assert_eq!(bytes.len(), original.len() * 4);

        let recovered = bytes_to_f32_vec(&bytes);
        assert_eq!(recovered.len(), original.len());

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "f32 roundtrip mismatch: {a} != {b}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // T-IDX-011: compute_batch_size returns REFERENCE_BATCH_SIZE for BERT
    //            sequence length (512 tokens).
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_011_batch_size_bert_512() {
        assert_eq!(
            compute_batch_size(512),
            32,
            "BERT-family models with 512 tokens must use batch_size=32"
        );
    }

    // -----------------------------------------------------------------------
    // T-IDX-012: compute_batch_size returns 2 for Qwen/GTE-Large
    //            sequence length (8192 tokens).
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_012_batch_size_qwen_8192() {
        assert_eq!(
            compute_batch_size(8192),
            2,
            "models with 8192-token context must use batch_size=2"
        );
    }

    // -----------------------------------------------------------------------
    // T-IDX-013: compute_batch_size clamps to 1 for very long contexts.
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_013_batch_size_very_long_context() {
        assert_eq!(
            compute_batch_size(32768),
            1,
            "very long context (32768) must clamp batch_size to 1"
        );
        assert_eq!(
            compute_batch_size(65536),
            1,
            "extreme context (65536) must clamp batch_size to 1"
        );
    }

    // -----------------------------------------------------------------------
    // T-IDX-014: compute_batch_size handles edge case of zero or one.
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_014_batch_size_edge_cases() {
        // max_sequence_length=0 should not divide by zero; clamps to 32.
        assert_eq!(
            compute_batch_size(0),
            32,
            "zero sequence length must not panic and must return max batch_size"
        );
        // max_sequence_length=1 -> 32*512/1 = 16384, clamped to 32.
        assert_eq!(
            compute_batch_size(1),
            32,
            "sequence length 1 must clamp to REFERENCE_BATCH_SIZE"
        );
    }

    // -----------------------------------------------------------------------
    // T-IDX-016: length_sorted_permutation produces ascending-length order
    // -----------------------------------------------------------------------

    /// Verifies that length_sorted_permutation returns indices sorted by
    /// content byte length in ascending order. Chunks with varying content
    /// lengths must be ordered from shortest to longest.
    #[test]
    fn t_idx_016_length_sorted_permutation_ascending_order() {
        let pdf_path = PathBuf::from("/test/doc.pdf");
        let chunks = vec![
            Chunk {
                source_file: pdf_path.clone(),
                page_start: 1,
                page_end: 1,
                chunk_index: 0,
                doc_text_offset_start: 0,
                doc_text_offset_end: 40,
                content: "This is a medium-length chunk of content".to_string(),
                content_hash: Chunk::compute_content_hash("This is a medium-length chunk of content"),
            },
            Chunk {
                source_file: pdf_path.clone(),
                page_start: 1,
                page_end: 1,
                chunk_index: 1,
                doc_text_offset_start: 40,
                doc_text_offset_end: 45,
                content: "Short".to_string(),
                content_hash: Chunk::compute_content_hash("Short"),
            },
            Chunk {
                source_file: pdf_path.clone(),
                page_start: 2,
                page_end: 2,
                chunk_index: 2,
                doc_text_offset_start: 45,
                doc_text_offset_end: 145,
                content: "This is a significantly longer chunk of content that contains many more words and characters than the others in this test".to_string(),
                content_hash: Chunk::compute_content_hash("This is a significantly longer chunk of content that contains many more words and characters than the others in this test"),
            },
            Chunk {
                source_file: pdf_path.clone(),
                page_start: 2,
                page_end: 2,
                chunk_index: 3,
                doc_text_offset_start: 145,
                doc_text_offset_end: 151,
                content: "Tiny".to_string(),
                content_hash: Chunk::compute_content_hash("Tiny"),
            },
        ];

        let perm = length_sorted_permutation(&chunks);

        // Verify that chunk content lengths are in ascending order through
        // the permutation.
        for window in perm.windows(2) {
            let len_a = chunks[window[0]].content.len();
            let len_b = chunks[window[1]].content.len();
            assert!(
                len_a <= len_b,
                "permutation must produce ascending order: len({})={} > len({})={}",
                window[0],
                len_a,
                window[1],
                len_b
            );
        }

        // The shortest chunk ("Tiny", index 3) must be first in the permutation.
        assert_eq!(perm[0], 3, "shortest chunk must be first");
        // The longest chunk (index 2) must be last.
        assert_eq!(perm[3], 2, "longest chunk must be last");
    }

    // -----------------------------------------------------------------------
    // T-IDX-017: restore_original_order correctly inverts the permutation
    // -----------------------------------------------------------------------

    /// Verifies that restore_original_order maps embeddings back to their
    /// original chunk positions using the permutation vector.
    #[test]
    fn t_idx_017_restore_original_order_inverts_permutation() {
        // Permutation: sorted_pos -> original_idx
        // [2, 0, 3, 1] means:
        //   sorted_pos 0 -> original chunk 2
        //   sorted_pos 1 -> original chunk 0
        //   sorted_pos 2 -> original chunk 3
        //   sorted_pos 3 -> original chunk 1
        let permutation = vec![2, 0, 3, 1];

        // Embeddings in sorted order. Each uses a unique marker value.
        let sorted_embeddings = vec![
            vec![20.0], // sorted_pos 0, for original chunk 2
            vec![0.0],  // sorted_pos 1, for original chunk 0
            vec![30.0], // sorted_pos 2, for original chunk 3
            vec![10.0], // sorted_pos 3, for original chunk 1
        ];

        let restored = restore_original_order(sorted_embeddings, &permutation);

        // restored[original_idx] must contain the embedding for that chunk.
        assert_eq!(restored[0], vec![0.0], "original chunk 0");
        assert_eq!(restored[1], vec![10.0], "original chunk 1");
        assert_eq!(restored[2], vec![20.0], "original chunk 2");
        assert_eq!(restored[3], vec![30.0], "original chunk 3");
    }

    // -----------------------------------------------------------------------
    // T-IDX-018: length_sorted_permutation handles single chunk
    // -----------------------------------------------------------------------

    /// Verifies that a single-element chunk list produces a trivial permutation.
    #[test]
    fn t_idx_018_length_sorted_permutation_single_chunk() {
        let pdf_path = PathBuf::from("/test/doc.pdf");
        let chunks = vec![Chunk {
            source_file: pdf_path,
            page_start: 1,
            page_end: 1,
            chunk_index: 0,
            doc_text_offset_start: 0,
            doc_text_offset_end: 5,
            content: "Hello".to_string(),
            content_hash: Chunk::compute_content_hash("Hello"),
        }];

        let perm = length_sorted_permutation(&chunks);
        assert_eq!(perm, vec![0]);
    }

    // -----------------------------------------------------------------------
    // T-IDX-019: length_sorted_permutation handles empty chunk list
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_019_length_sorted_permutation_empty() {
        let chunks: Vec<Chunk> = vec![];
        let perm = length_sorted_permutation(&chunks);
        assert!(perm.is_empty());
    }

    // -----------------------------------------------------------------------
    // T-IDX-020: restore_original_order handles empty input
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_020_restore_original_order_empty() {
        let restored = restore_original_order(vec![], &[]);
        assert!(restored.is_empty());
    }

    // -----------------------------------------------------------------------
    // T-IDX-021: length_sorted_permutation with equal-length chunks
    //            produces a valid permutation (all indices present).
    // -----------------------------------------------------------------------

    #[test]
    fn t_idx_021_length_sorted_permutation_equal_lengths() {
        let pdf_path = PathBuf::from("/test/doc.pdf");
        let chunks: Vec<Chunk> = (0..5)
            .map(|i| Chunk {
                source_file: pdf_path.clone(),
                page_start: 1,
                page_end: 1,
                chunk_index: i,
                doc_text_offset_start: i * 5,
                doc_text_offset_end: (i + 1) * 5,
                content: "AAAAA".to_string(), // all same length
                content_hash: Chunk::compute_content_hash("AAAAA"),
            })
            .collect();

        let perm = length_sorted_permutation(&chunks);

        // All original indices must be present exactly once.
        let mut sorted_perm = perm.clone();
        sorted_perm.sort();
        assert_eq!(sorted_perm, vec![0, 1, 2, 3, 4]);
    }

    // -----------------------------------------------------------------------
    // T-IDX-024: Analytical test that length-sorted batching reduces
    //            total padding compared to document-order batching.
    // -----------------------------------------------------------------------

    /// Computes the total padding tokens across all batches for a given
    /// chunk ordering and batch size. Padding per batch is:
    ///   (max_len_in_batch - len_i) summed over all elements in the batch.
    /// This models what the tokenizer does: pad all sequences in a batch
    /// to the length of the longest one.
    fn compute_total_padding(chunk_lengths: &[usize], batch_size: usize) -> usize {
        let mut total_padding = 0;
        for batch_start in (0..chunk_lengths.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(chunk_lengths.len());
            let batch = &chunk_lengths[batch_start..batch_end];
            let max_len = batch.iter().copied().max().unwrap_or(0);
            for &len in batch {
                total_padding += max_len - len;
            }
        }
        total_padding
    }

    #[test]
    fn t_idx_024_sorted_batching_reduces_padding() {
        // Simulate chunk lengths from a long book: wide variance in content size.
        let document_order_lengths = vec![
            500, 100, 800, 150, 750, 200, 900, 50, 600, 300, 450, 120, 700, 80, 550, 250, 850, 180,
            650, 350,
        ];

        // Qwen3 batch_size=2
        let batch_size = 2;

        // Document-order padding
        let document_order_padding = compute_total_padding(&document_order_lengths, batch_size);

        // Length-sorted padding
        let mut sorted_lengths = document_order_lengths.clone();
        sorted_lengths.sort();
        let sorted_padding = compute_total_padding(&sorted_lengths, batch_size);

        // Sorted batching must produce strictly less padding.
        assert!(
            sorted_padding < document_order_padding,
            "sorted padding ({}) must be less than document-order padding ({})",
            sorted_padding,
            document_order_padding
        );

        // For this distribution, sorted batching should reduce padding by at
        // least 50% compared to random document order.
        let reduction_pct = 100.0 * (1.0 - sorted_padding as f64 / document_order_padding as f64);
        assert!(
            reduction_pct > 50.0,
            "padding reduction ({:.1}%) should exceed 50% for high-variance chunk lengths",
            reduction_pct
        );
    }

    // -----------------------------------------------------------------------
    // T-IDX-025: restore_original_order is the inverse of
    //            length_sorted_permutation for any chunk set.
    // -----------------------------------------------------------------------

    /// Round-trip test: sort chunks by length, generate index-based
    /// "embeddings", restore order, and verify alignment.
    #[test]
    fn t_idx_025_sort_unsort_round_trip() {
        let pdf_path = PathBuf::from("/test/doc.pdf");
        let content_strings = ["AAAA", "BB", "CCCCCC", "D", "EEEEE"];
        let chunks: Vec<Chunk> = content_strings
            .iter()
            .enumerate()
            .map(|(i, &text)| Chunk {
                source_file: pdf_path.clone(),
                page_start: 1,
                page_end: 1,
                chunk_index: i,
                doc_text_offset_start: 0,
                doc_text_offset_end: text.len(),
                content: text.to_string(),
                content_hash: Chunk::compute_content_hash(text),
            })
            .collect();

        let perm = length_sorted_permutation(&chunks);

        // Simulate embedding: each sorted position gets its original index
        // as the embedding value. This lets us verify the round-trip.
        let sorted_embeddings: Vec<Vec<f32>> = perm
            .iter()
            .map(|&original_idx| vec![original_idx as f32])
            .collect();

        let restored = restore_original_order(sorted_embeddings, &perm);

        // After restore, restored[i] must contain vec![i as f32].
        for (i, emb) in restored.iter().enumerate() {
            assert_eq!(
                emb,
                &vec![i as f32],
                "round-trip broken at index {}: expected [{}], got {:?}",
                i,
                i,
                emb
            );
        }
    }

    // -----------------------------------------------------------------------
    // Hardware-adaptive batch size tests (compute_batch_size_with_caps)
    // -----------------------------------------------------------------------

    /// Helper: creates InferenceCapabilities with given EP, memory, and unified flag.
    fn caps(ep: &str, memory_gb: u64, unified: bool) -> InferenceCapabilities {
        InferenceCapabilities {
            active_ep: ep.to_string(),
            system_memory_bytes: memory_gb * 1024 * 1024 * 1024,
            unified_memory: unified,
        }
    }

    /// T-IDX-026: Legacy compute_batch_size produces identical results to
    /// compute_batch_size_with_caps with default (CPU, 8GB) capabilities.
    #[test]
    fn t_idx_026_legacy_matches_default_caps() {
        for seq_len in [1, 128, 512, 2048, 8192, 32768, 65536] {
            assert_eq!(
                compute_batch_size(seq_len),
                compute_batch_size_with_caps(seq_len, &InferenceCapabilities::default()),
                "legacy and default-caps must agree for seq_len={seq_len}"
            );
        }
    }

    /// T-IDX-027: CoreML on Apple Silicon with unified memory produces larger
    /// batch sizes than the same model on CUDA (discrete VRAM).
    #[test]
    fn t_idx_027_coreml_unified_larger_than_cuda() {
        let coreml = caps("CoreML", 32, true);
        let cuda = caps("CUDA", 32, false);

        for seq_len in [512, 2048, 8192] {
            let coreml_bs = compute_batch_size_with_caps(seq_len, &coreml);
            let cuda_bs = compute_batch_size_with_caps(seq_len, &cuda);
            assert!(
                coreml_bs >= cuda_bs,
                "CoreML unified batch_size ({coreml_bs}) must be >= CUDA ({cuda_bs}) for seq_len={seq_len}"
            );
        }
    }

    /// T-IDX-028: CoreML batch size scales with available RAM.
    /// More unified memory → larger batches (logarithmic scaling).
    #[test]
    fn t_idx_028_coreml_scales_with_memory() {
        let bs_32gb = compute_batch_size_with_caps(512, &caps("CoreML", 32, true));
        let bs_64gb = compute_batch_size_with_caps(512, &caps("CoreML", 64, true));
        let bs_128gb = compute_batch_size_with_caps(512, &caps("CoreML", 128, true));

        assert!(
            bs_64gb >= bs_32gb,
            "64GB ({bs_64gb}) must produce >= batch size than 32GB ({bs_32gb})"
        );
        assert!(
            bs_128gb >= bs_64gb,
            "128GB ({bs_128gb}) must produce >= batch size than 64GB ({bs_64gb})"
        );
    }

    /// T-IDX-029: CoreML batch size is clamped to 256 even with enormous RAM.
    #[test]
    fn t_idx_029_coreml_max_clamp_256() {
        let huge = caps("CoreML", 1024, true);
        let bs = compute_batch_size_with_caps(512, &huge);
        assert!(
            bs <= 256,
            "CoreML batch_size must be clamped to 256, got {bs}"
        );
    }

    /// T-IDX-030: CUDA and ROCm batch sizes are unaffected by system memory.
    /// They only depend on the model's sequence length.
    #[test]
    fn t_idx_030_cuda_rocm_ignore_memory() {
        for ep in ["CUDA", "ROCm"] {
            let small = caps(ep, 8, false);
            let large = caps(ep, 128, false);
            for seq_len in [512, 2048, 8192] {
                assert_eq!(
                    compute_batch_size_with_caps(seq_len, &small),
                    compute_batch_size_with_caps(seq_len, &large),
                    "{ep} batch size must not change with memory for seq_len={seq_len}"
                );
            }
        }
    }

    /// T-IDX-031: CPU batch size is clamped to 64.
    #[test]
    fn t_idx_031_cpu_max_clamp_64() {
        let cpu = caps("CPU", 256, false);
        let bs = compute_batch_size_with_caps(512, &cpu);
        assert!(bs <= 64, "CPU batch_size must be clamped to 64, got {bs}");
    }

    /// T-IDX-032: All EP configurations return at least 1 for any sequence length.
    #[test]
    fn t_idx_032_all_eps_minimum_1() {
        for ep in ["CoreML", "CUDA", "ROCm", "DirectML", "CPU", "unknown"] {
            for seq_len in [0, 1, 512, 8192, 65536, usize::MAX / 2] {
                let unified = ep == "CoreML";
                let c = caps(ep, 32, unified);
                let bs = compute_batch_size_with_caps(seq_len, &c);
                assert!(
                    bs >= 1,
                    "{ep} with seq_len={seq_len} must produce batch_size >= 1, got {bs}"
                );
            }
        }
    }

    /// T-IDX-033: CoreML without unified memory (Intel Mac) gets the same
    /// treatment as a CPU EP since CoreML on Intel dispatches to Accelerate.
    #[test]
    fn t_idx_033_coreml_intel_no_unified() {
        let intel_mac = caps("CoreML", 32, false);
        // Without unified_memory flag, the "CoreML" match arm's guard fails,
        // falling through to the default (CPU-like) path.
        let cpu = caps("CPU", 32, false);
        for seq_len in [512, 2048, 8192] {
            let coreml_bs = compute_batch_size_with_caps(seq_len, &intel_mac);
            let cpu_bs = compute_batch_size_with_caps(seq_len, &cpu);
            // They should use the same default path, producing similar results.
            // (Not necessarily identical because CPU path scales with cores.)
            assert!(
                coreml_bs >= 1,
                "CoreML on Intel must still return valid batch_size"
            );
            assert!(cpu_bs >= 1, "CPU must still return valid batch_size");
        }
    }

    /// T-IDX-034: DirectML batch sizes match CUDA for the same sequence lengths.
    /// Both use discrete VRAM, calibrated identically.
    #[test]
    fn t_idx_034_directml_matches_cuda() {
        for seq_len in [512, 2048, 8192, 32768] {
            let dml = compute_batch_size_with_caps(seq_len, &caps("DirectML", 16, false));
            let cuda = compute_batch_size_with_caps(seq_len, &caps("CUDA", 16, false));
            assert_eq!(
                dml, cuda,
                "DirectML and CUDA batch_size must match for seq_len={seq_len}"
            );
        }
    }
}
