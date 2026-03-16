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

// HNSW approximate nearest neighbor index management.
//
// Wraps the `hnsw_rs` crate to build, serialize, and deserialize an in-memory
// HNSW graph from stored embedding vectors. The index uses cosine distance
// (`DistCosine` from `anndists`), M=16 maximum connections per layer, and
// `ef_construction`=200 candidate neighbors during construction.
//
// Each vector is labeled with its `chunk.id` (i64 cast to usize as `DataId`) to
// establish a direct mapping from HNSW search results back to the chunk table.
//
// The index is serialized to disk as two files (`{basename}.hnsw.graph` and
// `{basename}.hnsw.data`) via the `hnsw_rs` `file_dump` and `load_hnsw` APIs.
//
// # Lifetime design
//
// `hnsw_rs` declares `Hnsw<'b, T, D>` where the lifetime parameter `'b` is
// meaningful only when mmap is enabled via `ReloadOptions::set_mmap`. In mmap
// mode, `'b` bounds the `Hnsw` to not outlive the memory-mapped region in
// `HnswIo`. NeuronCite uses non-mmap mode exclusively; `load_hnsw` copies all
// point data into owned `Vec<T>` allocations so no reference with lifetime `'b`
// is held by the `Hnsw` struct after loading completes.
//
// `Hnsw<'b, T, D>` is invariant over `'b` (the compiler enforces this). As a
// result, self-referential struct approaches that require lifetime covariance
// (such as `ouroboros`) cannot express the relationship. Instead, the
// `HnswStorage::Loaded` variant retains the `HnswIo` owner alongside the
// `Hnsw<'static, ...>` graph. The transmute from `Hnsw<'_, ...>` to
// `Hnsw<'static, ...>` is still required, but it is now:
//
//  1. Isolated to a single private function `load_hnsw_from_io`.
//  2. Justified by structural ownership: `HnswIo` lives inside `HnswStorage`
//     and is dropped only when the `HnswIndex` is dropped, which is after the
//     `Hnsw` graph is also dropped.
//  3. Protected by a compile-time assertion that `HnswIo` is not a zero-sized
//     type (if it were, the ownership argument would be vacuous).
//
// The previous code dropped `HnswIo` at the end of `deserialize_hnsw`,
// relying solely on comments to explain safety. The current design makes the
// lifetime dependency visible in the type: `HnswStorage::Loaded` contains both
// the owner and the graph, preventing accidental reordering or early drop.

use std::io::Read as _;
use std::path::Path;

use hnsw_rs::prelude::*;
use sha2::{Digest, Sha256};

use crate::error::StoreError;

// Compile-time assertion: HnswIo must not be a zero-sized type. If it were,
// retaining it in HnswStorage::Loaded would provide no ownership guarantee
// because zero-sized types carry no address identity. The non-mmap safety
// argument depends on HnswIo being a real heap allocation (a file reader with
// internal buffers and path state), which this assertion enforces at build time.
const _: () = assert!(
    std::mem::size_of::<HnswIo>() > 0,
    "HnswIo must not be a zero-sized type"
);

// Compile-time assertion: chunk IDs (i64) are cast to `usize` when labeling
// HNSW data points (DataId = usize). On a 64-bit platform usize is 8 bytes,
// so the cast is lossless for all valid i64 values. On a 32-bit platform
// usize is 4 bytes and the cast would silently truncate chunk IDs above
// 2^32, corrupting search results. This assertion prevents compilation on
// any target where usize is narrower than i64.
const _: () = assert!(
    std::mem::size_of::<usize>() == 8,
    "HNSW chunk-ID casting requires a 64-bit platform (usize must be 8 bytes)"
);

/// HNSW construction parameter: maximum number of connections per layer.
const HNSW_M: usize = 16;

/// HNSW construction parameter: number of candidate neighbors explored
/// during index construction. Higher values produce a more accurate graph
/// at the cost of slower indexing.
const HNSW_EF_CONSTRUCTION: usize = 200;

/// Maximum number of layers in the HNSW graph hierarchy.
const HNSW_MAX_LAYER: usize = 16;

/// Basename prefix used for HNSW dump files. The actual files are
/// `{HNSW_BASENAME}.hnsw.graph` and `{HNSW_BASENAME}.hnsw.data`.
const HNSW_BASENAME: &str = "neuroncite_hnsw";

/// Temporary basename used during atomic serialization. Files written with this
/// prefix are renamed to `HNSW_BASENAME` files once both are fully written.
const HNSW_BASENAME_TMP: &str = "neuroncite_hnsw_tmp";

/// Filename suffix for the SHA-256 checksum sidecar file that accompanies the
/// graph file. Written after the atomic rename so it is consistent with the
/// on-disk graph content. Used by `deserialize_hnsw` to detect corruption.
const HNSW_GRAPH_SHA256_SUFFIX: &str = "neuroncite_hnsw.hnsw.graph.sha256";

/// Computes the SHA-256 digest of the file at `path` by reading it in streaming
/// 64 KiB blocks. Returns the hex-encoded hash string.
fn sha256_of_file(path: &Path) -> Result<String, StoreError> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| StoreError::hnsw(format!("cannot open file for checksum: {e}")))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| StoreError::hnsw(format!("read error during checksum: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Loads the HNSW graph from a mutable `HnswIo` reference and transmutes the
/// returned lifetime to `'static`. This is the single location of the transmute
/// in this module; all other code is safe.
///
/// # Safety
///
/// This function is sound when ALL of the following hold:
///
/// 1. `set_options` has not been called on `hnsw_io` before this call.
///    Without a `ReloadOptions::set_mmap` call, `load_hnsw` operates in copy
///    mode: all point data is read into owned `Vec<T>` allocations. No
///    reference into `hnsw_io`'s memory is held by the returned `Hnsw`.
///
/// 2. The hnsw_rs crate is pinned to version =0.3.4 in the workspace
///    Cargo.toml. This invariant must be re-verified before bumping the pin.
///
/// 3. The caller stores the returned `Hnsw<'static, ...>` inside a struct that
///    also owns (and outlives) the `HnswIo` passed here. This function takes
///    `hnsw_io` by value and returns it alongside the graph, enforcing this
///    ownership constraint at the call site via the type system.
///
/// The transmute changes only the phantom lifetime parameter `'b`. No pointer
/// values, vtable entries, or object representations are altered.
/// `Hnsw<'a, T, D>` and `Hnsw<'b, T, D>` have identical memory layouts for
/// any `'a`, `'b` because `'b` appears only in lifetime-parameterized types
/// whose runtime representation is independent of the lifetime.
unsafe fn load_hnsw_static(
    hnsw_io: &mut HnswIo,
) -> Result<Hnsw<'static, f32, DistCosine>, StoreError> {
    let hnsw_with_borrow: Hnsw<'_, f32, DistCosine> = hnsw_io
        .load_hnsw()
        .map_err(|e| StoreError::hnsw(format!("deserialization failed: {e}")))?;
    // SAFETY: see the function-level comment for the three conditions. The
    // transmute changes only the phantom lifetime `'_` to `'static`. The
    // explicit type annotation is required by clippy::missing_transmute_annotations.
    Ok(unsafe {
        std::mem::transmute::<Hnsw<'_, f32, DistCosine>, Hnsw<'static, f32, DistCosine>>(
            hnsw_with_borrow,
        )
    })
}

/// Internal storage variant for an HNSW index.
enum HnswStorage {
    /// Graph constructed by `build_hnsw` from in-memory vectors. `Hnsw::new()`
    /// returns `Hnsw<'static, T, D>` directly; no lifetime management needed.
    Built(Hnsw<'static, f32, DistCosine>),

    /// Graph loaded from disk by `deserialize_hnsw`. `HnswIo` is retained here
    /// so that its lifetime structurally outlives the graph that was loaded from
    /// it. The graph is stored as `Hnsw<'static, ...>` after the transmute in
    /// `load_hnsw_static`; the `HnswIo` owner provides the structural guarantee
    /// that no actual borrow is violated (non-mmap mode, all data owned).
    ///
    /// `_owner` is boxed to equalize the enum variant sizes and suppress
    /// `clippy::large_enum_variant`. The Box does not change the ownership
    /// semantics: `Box<HnswIo>` is dropped when `HnswStorage::Loaded` is
    /// dropped, which is after the `graph` field is dropped.
    Loaded {
        /// File reader used to load the graph. Boxed to reduce the size of this
        /// enum variant. Retained to enforce ownership: the box is dropped only
        /// when `HnswStorage::Loaded` is dropped, which occurs after `graph`.
        _owner: Box<HnswIo>,
        /// The HNSW graph loaded from disk with the lifetime erased to `'static`
        /// by `load_hnsw_static`. The `_owner` field above ensures `HnswIo`
        /// outlives this graph within this struct.
        graph: Hnsw<'static, f32, DistCosine>,
    },
}

impl HnswStorage {
    /// Returns a reference to the inner `Hnsw` graph regardless of variant.
    fn hnsw(&self) -> &Hnsw<'static, f32, DistCosine> {
        match self {
            HnswStorage::Built(h) => h,
            HnswStorage::Loaded { graph, .. } => graph,
        }
    }
}

/// Wrapper around the `hnsw_rs` Hnsw type that provides search and metric
/// functionality. Uses `HnswStorage` to cover both the freshly-built and the
/// deserialized code paths. The unsafe transmute is isolated to the private
/// `load_hnsw_static` function and occurs at most once per deserialization.
pub struct HnswIndex {
    inner: HnswStorage,
}

impl HnswIndex {
    /// Returns the number of vectors stored in the index.
    pub fn len(&self) -> usize {
        self.inner.hnsw().get_nb_point()
    }

    /// Returns true if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Searches the index for the `k` nearest neighbors of the given query
    /// vector. Returns a list of `(chunk_id, distance)` pairs sorted by
    /// ascending distance (lower distance = more similar for cosine).
    ///
    /// The `ef_search` parameter controls the trade-off between recall and
    /// latency. Higher values increase recall at the cost of slower queries.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(i64, f32)> {
        self.inner
            .hnsw()
            .search(query, k, ef_search)
            .into_iter()
            // INVARIANT: labels originate from i64 primary keys that were converted
            // to usize via try_from in build_hnsw. The cast back to i64 is lossless
            // because the original value fit in i64.
            .map(|n| (n.d_id as i64, n.distance))
            .collect()
    }
}

/// Builds an HNSW index from the given vectors. Each vector is a pair of
/// `(chunk_id, embedding_slice)`. The `chunk_id` is stored as the `DataId` label
/// in the HNSW graph, establishing a direct mapping back to the chunk table.
///
/// # Arguments
///
/// * `vectors` - Slice of `(chunk_id, embedding)` pairs. The `chunk_id` is the
///   primary key from the chunk table. The embedding slice length must equal
///   `dimension` for all vectors.
/// * `dimension` - The dimensionality of each embedding vector.
///
/// # Errors
///
/// Returns `StoreError::HnswIndex` if any vector's length does not match the
/// declared `dimension`.
pub fn build_hnsw(vectors: &[(i64, &[f32])], dimension: usize) -> Result<HnswIndex, StoreError> {
    let max_elements = vectors.len().max(1);

    let mut hnsw = Hnsw::<f32, DistCosine>::new(
        HNSW_M,
        max_elements,
        HNSW_MAX_LAYER,
        HNSW_EF_CONSTRUCTION,
        DistCosine,
    );

    let mut data_for_insert: Vec<(&[f32], usize)> = Vec::with_capacity(vectors.len());
    for (id, vec) in vectors {
        if vec.len() != dimension {
            return Err(StoreError::hnsw(format!(
                "embedding dimension mismatch for chunk {id}: expected {dimension}, found {}",
                vec.len()
            )));
        }
        let label = usize::try_from(*id).map_err(|_| {
            StoreError::hnsw(format!(
                "chunk id {id} cannot be converted to usize (negative or out of range)"
            ))
        })?;
        data_for_insert.push((*vec, label));
    }

    hnsw.parallel_insert_slice(&data_for_insert);

    // Switch to searching mode after all inserts are complete. This is
    // required by hnsw_rs to allow concurrent search operations.
    hnsw.set_searching_mode(true);

    Ok(HnswIndex {
        inner: HnswStorage::Built(hnsw),
    })
}

/// Serializes an HNSW index to disk at the given directory path using an atomic
/// write-then-rename strategy to prevent partial writes from leaving the on-disk
/// index in a corrupt state.
///
/// Write sequence:
/// 1. Write graph and data to temporary files (`{HNSW_BASENAME_TMP}.hnsw.graph`
///    and `{HNSW_BASENAME_TMP}.hnsw.data`).
/// 2. Compute the SHA-256 checksum of the temporary graph file.
/// 3. Atomically rename the temporary files to the canonical names
///    (`{HNSW_BASENAME}.hnsw.graph` and `{HNSW_BASENAME}.hnsw.data`).
/// 4. Write the checksum to `{HNSW_GRAPH_SHA256_SUFFIX}` so that the next call
///    to `deserialize_hnsw` can detect any post-write corruption.
///
/// On Unix the rename step is atomic (a POSIX guarantee). On Windows, rename
/// is not atomic when the destination exists; `std::fs::rename` on Windows will
/// attempt a replace, which is best-effort but not guaranteed. A crash between
/// step 3 and step 4 leaves a valid graph file without a checksum sidecar;
/// `deserialize_hnsw` treats a missing sidecar as unchecked (not as corrupt).
///
/// # Errors
///
/// Returns `StoreError::HnswIndex` if serialization, checksum computation, the
/// atomic rename, or the sidecar write fails.
pub fn serialize_hnsw(index: &HnswIndex, path: &Path) -> Result<(), StoreError> {
    // Step 1: write to temporary files via the tmp basename.
    index
        .inner
        .hnsw()
        .file_dump(path, HNSW_BASENAME_TMP)
        .map_err(|e| StoreError::hnsw(format!("serialization to temp files failed: {e}")))?;

    let tmp_graph = path.join(format!("{HNSW_BASENAME_TMP}.hnsw.graph"));
    let tmp_data = path.join(format!("{HNSW_BASENAME_TMP}.hnsw.data"));
    let dst_graph = path.join(format!("{HNSW_BASENAME}.hnsw.graph"));
    let dst_data = path.join(format!("{HNSW_BASENAME}.hnsw.data"));
    let checksum_path = path.join(HNSW_GRAPH_SHA256_SUFFIX);

    // Step 2: compute checksum of the temporary graph file before renaming.
    let checksum = sha256_of_file(&tmp_graph)?;

    // Step 3: atomic rename — graph first, then data.
    // On failure, temporary files are left in place so the caller can retry or
    // clean up. The canonical files are not modified until both renames succeed.
    std::fs::rename(&tmp_graph, &dst_graph)
        .map_err(|e| StoreError::hnsw(format!("graph rename failed: {e}")))?;
    std::fs::rename(&tmp_data, &dst_data)
        .map_err(|e| StoreError::hnsw(format!("data rename failed: {e}")))?;

    // Step 4: write checksum sidecar. A crash between step 3 and here leaves a
    // valid graph without a checksum; `deserialize_hnsw` treats a missing
    // sidecar as unchecked rather than corrupt.
    std::fs::write(&checksum_path, checksum.as_bytes())
        .map_err(|e| StoreError::hnsw(format!("checksum write failed: {e}")))?;

    Ok(())
}

/// Deserializes an HNSW index from disk at the given directory path. Reads
/// the `{HNSW_BASENAME}.hnsw.graph` and `{HNSW_BASENAME}.hnsw.data` files.
///
/// The `HnswIo` file reader is retained inside `HnswStorage::Loaded` alongside
/// the loaded graph. This structural ownership ensures that the `HnswIo` cannot
/// be dropped before the `Hnsw` graph, making the lifetime dependency between
/// owner and graph visible in the type system rather than relying solely on
/// comments. See the module-level documentation for a full discussion of the
/// transmute safety argument.
///
/// # Errors
///
/// Returns `StoreError::HnswIndex` if the files are missing, corrupted, or
/// contain incompatible data.
pub fn deserialize_hnsw(path: &Path) -> Result<HnswIndex, StoreError> {
    // Verify the SHA-256 checksum sidecar before loading if it exists. A missing
    // sidecar (e.g., after a crash between the rename and the sidecar write) is
    // treated as unchecked rather than corrupt. A sidecar with a mismatching hash
    // indicates post-write corruption; the caller is expected to rebuild the index.
    let checksum_path = path.join(HNSW_GRAPH_SHA256_SUFFIX);
    if checksum_path.exists() {
        let expected = std::fs::read_to_string(&checksum_path)
            .map_err(|e| StoreError::hnsw(format!("cannot read checksum sidecar: {e}")))?;
        let expected = expected.trim();
        let graph_path = path.join(format!("{HNSW_BASENAME}.hnsw.graph"));
        let actual = sha256_of_file(&graph_path)?;
        if actual != expected {
            return Err(StoreError::hnsw(format!(
                "HNSW graph checksum mismatch (expected {expected}, got {actual}) — \
                 the on-disk index is corrupt; rebuild required"
            )));
        }
    }

    // `set_options` is intentionally never called on this HnswIo. Without a
    // `ReloadOptions::set_mmap` call, `load_hnsw` operates in copy mode: all
    // point data is read from disk into owned `Vec<T>` allocations. This is the
    // non-mmap invariant that the SAFETY comment in `load_hnsw_static` depends on.
    // The hnsw_rs crate is pinned to =0.3.4 in the workspace Cargo.toml; this
    // invariant must be re-verified on any version upgrade.
    let mut hnsw_io = HnswIo::new(path, HNSW_BASENAME);

    // SAFETY: mmap is not enabled (no `set_options` call above) so `load_hnsw`
    // copies all data into owned allocations. `hnsw_io` is moved into the
    // `_owner` field of `HnswStorage::Loaded` below; it is dropped only when
    // the enclosing `HnswIndex` is dropped, after `graph` is also dropped.
    let mut graph = unsafe { load_hnsw_static(&mut hnsw_io) }?;

    graph.set_searching_mode(true);
    // Note: set_searching_mode takes &mut self, which requires the graph to be mutable.
    // We do this before moving into HnswStorage::Loaded.

    Ok(HnswIndex {
        inner: HnswStorage::Loaded {
            _owner: Box::new(hnsw_io),
            graph,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Generates a deterministic pseudo-random f32 vector from a seed value.
    /// Uses a simple linear congruential generator for reproducibility.
    fn make_vector(seed: u64, dimension: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(dimension);
        let mut state = seed;
        for _ in 0..dimension {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let val = (state >> 33) as f32 / (u32::MAX as f32);
            v.push(val);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// T-STO-015: HNSW serialization roundtrip.
    #[test]
    fn t_sto_015_hnsw_serialization_roundtrip() {
        let dimension = 128;
        let num_vectors = 500;
        let ef_search = 100;
        let top_k = 10;

        let vectors: Vec<Vec<f32>> = (1..=num_vectors)
            .map(|i| make_vector(i, dimension))
            .collect();

        let labeled_vectors: Vec<(i64, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| ((i as i64) + 1, v.as_slice()))
            .collect();

        let original_index = build_hnsw(&labeled_vectors, dimension).expect("build_hnsw");
        assert_eq!(original_index.len(), num_vectors as usize);

        let query = &vectors[0];
        let original_results = original_index.search(query, top_k, ef_search);

        let tmp = TempDir::new().expect("failed to create temp dir");
        serialize_hnsw(&original_index, tmp.path()).expect("serialization failed");

        let loaded_index = deserialize_hnsw(tmp.path()).expect("deserialization failed");
        assert_eq!(loaded_index.len(), num_vectors as usize);

        let loaded_results = loaded_index.search(query, top_k, ef_search);

        let original_ids: Vec<i64> = original_results.iter().map(|(id, _)| *id).collect();
        let loaded_ids: Vec<i64> = loaded_results.iter().map(|(id, _)| *id).collect();

        assert_eq!(
            original_ids, loaded_ids,
            "serialization roundtrip must preserve search results"
        );

        assert_eq!(
            original_ids[0], 1,
            "nearest neighbor of vector #1 must be itself"
        );
    }

    /// Verifies that deserialized HNSW indexes release memory when dropped.
    /// This test creates and drops a deserialized index to confirm that the
    /// `HnswStorage::Loaded` variant drops both `graph` and `_owner` cleanly.
    #[test]
    fn deserialized_index_drops_cleanly() {
        let dimension = 32;
        let num_vectors = 50;

        let vectors: Vec<Vec<f32>> = (1..=num_vectors)
            .map(|i| make_vector(i as u64, dimension))
            .collect();

        let labeled: Vec<(i64, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| ((i as i64) + 1, v.as_slice()))
            .collect();

        let index = build_hnsw(&labeled, dimension).expect("build_hnsw");
        let tmp = TempDir::new().expect("failed to create temp dir");
        serialize_hnsw(&index, tmp.path()).expect("serialization failed");

        // Deserialize and immediately drop — validates that both the graph and
        // the retained HnswIo owner are freed without panic.
        {
            let loaded = deserialize_hnsw(tmp.path()).expect("deserialization failed");
            assert_eq!(loaded.len(), num_vectors);
            // loaded is dropped here, freeing HnswStorage::Loaded { graph, _owner }.
        }

        // Deserialize again to confirm the files are still valid and the
        // previous drop did not corrupt the on-disk state.
        let loaded2 = deserialize_hnsw(tmp.path()).expect("second deserialization failed");
        assert_eq!(loaded2.len(), num_vectors);
    }

    /// T-STO-017: `build_hnsw` returns an error when a vector's length does
    /// not match the declared dimension. Before this fix, the function used
    /// `assert_eq!` which panics in production. The error variant is
    /// `StoreError::HnswIndex` containing the mismatched chunk ID and
    /// both the expected and actual dimensions.
    #[test]
    fn t_sto_017_build_hnsw_dimension_mismatch_returns_error() {
        let correct = vec![1.0_f32, 0.0, 0.0, 0.0];
        let wrong = vec![1.0_f32, 0.0, 0.0]; // 3 dimensions instead of 4

        let vectors: Vec<(i64, &[f32])> = vec![(1, correct.as_slice()), (2, wrong.as_slice())];

        let result = build_hnsw(&vectors, 4);

        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("build_hnsw must return Err on dimension mismatch"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("dimension mismatch"),
            "error message must mention 'dimension mismatch', got: {msg}"
        );
        assert!(
            msg.contains("chunk 2"),
            "error message must identify the offending chunk ID, got: {msg}"
        );
    }

    /// T-STO-018: `build_hnsw` succeeds when all vectors match the declared
    /// dimension, returning an index that contains all inserted vectors.
    #[test]
    fn t_sto_018_build_hnsw_correct_dimensions() {
        let dim = 8;
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| make_vector(i, dim)).collect();
        let labeled: Vec<(i64, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| ((i as i64) + 1, v.as_slice()))
            .collect();

        let index = build_hnsw(&labeled, dim).expect("build_hnsw with correct dimensions");
        assert_eq!(index.len(), 10);
    }

    /// T-STO-019: `build_hnsw` with an empty vector slice returns a valid
    /// empty index without error.
    #[test]
    fn t_sto_019_build_hnsw_empty_vectors() {
        let vectors: Vec<(i64, &[f32])> = Vec::new();
        let index = build_hnsw(&vectors, 128).expect("build_hnsw with empty input");
        assert!(index.is_empty());
    }

    /// T-STO-020: `build_hnsw` rejects negative chunk IDs that cannot be
    /// represented as usize. Before the fix, negative i64 values were silently
    /// truncated via `as usize`, producing incorrect DataId labels in the
    /// HNSW graph and corrupting search results.
    #[test]
    fn t_sto_020_build_hnsw_rejects_negative_chunk_id() {
        let vec_data = vec![1.0_f32, 0.0, 0.0, 0.0];
        let vectors: Vec<(i64, &[f32])> = vec![(-1, vec_data.as_slice())];

        let result = build_hnsw(&vectors, 4);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("build_hnsw must reject negative chunk IDs"),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("cannot be converted to usize"),
            "error message must explain the conversion failure, got: {msg}"
        );
    }

    /// T-STO-021: HNSW search returns the original i64 chunk IDs that were
    /// used during construction. Verifies the usize->i64 round-trip in
    /// build_hnsw (i64->usize) and search (usize->i64) is lossless.
    ///
    /// Uses 20 vectors instead of 3 because `parallel_insert_slice` needs
    /// a sufficiently large dataset to build a well-connected HNSW graph.
    /// With fewer than ~10 points the graph connectivity is non-deterministic
    /// and the query vector may not appear in search results.
    #[test]
    fn t_sto_021_hnsw_search_returns_original_ids() {
        let dimension = 8;
        let ids: Vec<i64> = (1..=20).collect();
        let vectors: Vec<Vec<f32>> = ids
            .iter()
            .map(|&id| make_vector(id as u64, dimension))
            .collect();
        let labeled: Vec<(i64, &[f32])> = ids
            .iter()
            .zip(vectors.iter())
            .map(|(&id, v)| (id, v.as_slice()))
            .collect();

        let index = build_hnsw(&labeled, dimension).expect("build_hnsw");
        assert_eq!(index.len(), 20);

        // Search with the first vector as query; it must appear in the results
        // with its original i64 ID (1).
        let results = index.search(&vectors[0], 5, 100);
        let result_ids: Vec<i64> = results.iter().map(|(id, _)| *id).collect();
        assert!(
            result_ids.contains(&1),
            "search results must contain the original chunk ID 1, got: {:?}",
            result_ids
        );
    }

    /// T-STO-022: Serializing a deserialized index (`HnswStorage::Loaded`)
    /// produces a valid file that deserializes into an index with the same
    /// search results. Verifies that `hnsw().file_dump()` delegates correctly
    /// for both `Built` and `Loaded` variants via `HnswStorage::hnsw()`.
    #[test]
    fn t_sto_022_serialize_loaded_index_roundtrips() {
        let dimension = 16;
        let num_vectors = 30;

        let vectors: Vec<Vec<f32>> = (1..=num_vectors)
            .map(|i| make_vector(i as u64, dimension))
            .collect();
        let labeled: Vec<(i64, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| ((i as i64) + 1, v.as_slice()))
            .collect();

        let original = build_hnsw(&labeled, dimension).expect("build_hnsw");
        let tmp1 = TempDir::new().unwrap();
        serialize_hnsw(&original, tmp1.path()).expect("first serialize");

        // Load the index, then serialize it again into a second temp directory.
        let loaded = deserialize_hnsw(tmp1.path()).expect("first deserialize");
        let tmp2 = TempDir::new().unwrap();
        serialize_hnsw(&loaded, tmp2.path()).expect("re-serialize loaded index");

        // Deserialize the re-serialized index and confirm search results match.
        let reloaded = deserialize_hnsw(tmp2.path()).expect("second deserialize");
        assert_eq!(reloaded.len(), num_vectors);

        let query = &vectors[0];
        let original_results = original.search(query, 5, 50);
        let reloaded_results = reloaded.search(query, 5, 50);

        let original_ids: Vec<i64> = original_results.iter().map(|(id, _)| *id).collect();
        let reloaded_ids: Vec<i64> = reloaded_results.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            original_ids, reloaded_ids,
            "re-serialized loaded index must yield the same search results"
        );
    }
}
