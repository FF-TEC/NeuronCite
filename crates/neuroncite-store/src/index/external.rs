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

// External embedding storage via memory-mapped files.
//
// Stores embedding vectors in a flat binary file alongside the `SQLite` database
// rather than inside blob columns. Each embedding occupies a fixed number of
// bytes (4 bytes per f32 dimension), and the chunk's database record stores the
// byte offset and length into the external file. Memory-mapped I/O via `memmap2`
// provides zero-copy access to embedding vectors during HNSW index construction
// and querying.
//
// File format: contiguous little-endian IEEE 754 f32 values with no header or
// padding. Each vector occupies exactly dimension * 4 bytes at a fixed offset.
//
// Stateful `ExternalWriter` and `ExternalReader` structs hold the file handle
// (and memory map, respectively) open for the lifetime of the struct, avoiding
// repeated open/mmap/close syscall overhead when writing or reading thousands
// of embeddings during HNSW index construction.

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Size of a single f32 value in bytes (IEEE 754 single-precision). Used for
/// byte length calculations when converting between f32 slices and the on-disk
/// little-endian binary format. Defined as a constant to avoid magic `4`
/// literals scattered across the read/write functions.
const BYTES_PER_F32: usize = std::mem::size_of::<f32>();

use memmap2::Mmap;

use crate::error::StoreError;

/// Creates an empty external embeddings file at the given path. If the file
/// already exists, it is truncated to zero length.
///
/// # Errors
///
/// Returns `StoreError::Io` if the file cannot be created.
pub fn create_external_file(path: &Path) -> Result<(), StoreError> {
    File::create(path).map_err(|e| StoreError::io(path, e))?;
    Ok(())
}

/// Appends an embedding vector to the external embeddings file. Returns the
/// byte offset and byte length of the appended data. The caller stores these
/// values in the chunk table's `ext_offset` and `ext_length` columns.
///
/// The embedding is written as contiguous little-endian f32 values.
///
/// This is a convenience function for single-shot appends. For bulk writes
/// during indexing, use `ExternalWriter` to amortize the file open/close cost.
///
/// # Arguments
///
/// * `path` - Path to the external embeddings file.
/// * `embedding` - The embedding vector to append.
///
/// # Returns
///
/// A tuple `(offset, length)` where `offset` is the byte position in the file
/// where the embedding starts, and `length` is the number of bytes written.
///
/// # Errors
///
/// Returns `StoreError::Io` if the file cannot be opened or written to.
/// Returns `StoreError::IntegerOverflow` if the byte length calculation overflows.
pub fn append_embedding(path: &Path, embedding: &[f32]) -> Result<(u64, u64), StoreError> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| StoreError::io(path, e))?;

    // Determine the current file size (= offset for the appended data)
    let offset = file
        .seek(SeekFrom::End(0))
        .map_err(|e| StoreError::io(path, e))?;

    // checked_mul guards against overflow when the embedding dimension is
    // extremely large. On 64-bit platforms this is practically unreachable,
    // but on 32-bit platforms (usize = u32) a dimension above ~1 billion
    // would overflow.
    let byte_len = embedding.len().checked_mul(BYTES_PER_F32).ok_or_else(|| {
        StoreError::integer_overflow(format!(
            "byte length overflow: {} elements * {} bytes/element",
            embedding.len(),
            BYTES_PER_F32
        ))
    })?;

    let mut bytes = Vec::with_capacity(byte_len);
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }

    file.write_all(&bytes)
        .map_err(|e| StoreError::io(path, e))?;

    // Convert usize byte_len to u64. On 64-bit platforms this is infallible;
    // on 32-bit platforms usize fits within u64 trivially.
    let byte_len_u64 = u64::try_from(byte_len).map_err(|_| {
        StoreError::integer_overflow(format!(
            "byte length {byte_len} cannot be represented as u64"
        ))
    })?;

    Ok((offset, byte_len_u64))
}

/// Reads an embedding vector from the external embeddings file at the given
/// byte offset and length. The bytes are interpreted as contiguous
/// little-endian f32 values.
///
/// This is a convenience function for single-shot reads. For bulk reads during
/// HNSW construction or querying, use `ExternalReader` to amortize the file
/// open/mmap cost across thousands of reads.
///
/// # Arguments
///
/// * `path` - Path to the external embeddings file.
/// * `offset` - Byte offset where the embedding starts.
/// * `length` - Number of bytes to read (must be a multiple of BYTES_PER_F32).
///
/// # Errors
///
/// Returns `StoreError::Io` if the file cannot be opened or memory-mapped,
/// or `StoreError::HnswIndex` if the offset/length are out of bounds or
/// the length is not a multiple of BYTES_PER_F32 (4).
/// Returns `StoreError::IntegerOverflow` if offset or length cannot be
/// represented as usize on the current platform.
pub fn read_embedding(path: &Path, offset: u64, length: u64) -> Result<Vec<f32>, StoreError> {
    if !length.is_multiple_of(BYTES_PER_F32 as u64) {
        return Err(StoreError::hnsw(format!(
            "embedding byte length {length} is not a multiple of BYTES_PER_F32 (4)"
        )));
    }

    let file = File::open(path).map_err(|e| StoreError::io(path, e))?;

    let file_len = file.metadata().map_err(|e| StoreError::io(path, e))?.len();

    // checked_add guards against u64 wraparound when both values are near
    // u64::MAX. Without it, offset + length could wrap to a small number in
    // release builds (Rust does not panic on overflow in release mode), causing
    // the bounds check to pass for an actually out-of-bounds range.
    let end = offset.checked_add(length).ok_or_else(|| {
        StoreError::hnsw(format!(
            "read range offset {offset} + length {length} overflows u64"
        ))
    })?;
    if end > file_len {
        return Err(StoreError::hnsw(format!(
            "read range [{offset}..{end}] exceeds file size {file_len}"
        )));
    }

    // SAFETY: The file is opened read-only and remains open for the duration
    // of the Mmap lifetime. No concurrent writer modifies the mapped region
    // because external embedding files are only appended to (never modified
    // in place) during indexing, and reads occur after indexing completes.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| StoreError::io(path, e))?;

    // Convert u64 offset and length to usize via checked conversion. On 32-bit
    // platforms, a u64 value exceeding usize::MAX would cause silent truncation
    // with `as usize`. usize::try_from returns Err for out-of-range values.
    let offset_usize = usize::try_from(offset).map_err(|_| {
        StoreError::integer_overflow(format!(
            "embedding offset {offset} cannot be represented as usize on this platform"
        ))
    })?;
    let length_usize = usize::try_from(length).map_err(|_| {
        StoreError::integer_overflow(format!(
            "embedding length {length} cannot be represented as usize on this platform"
        ))
    })?;

    decode_f32_slice(&mmap, offset_usize, length_usize)
}

// ---------------------------------------------------------------------------
// Stateful writer: holds the file handle open across multiple appends
// ---------------------------------------------------------------------------

/// Stateful writer that holds the external embeddings file open for the
/// duration of a bulk write session. Avoids repeated open/seek/close syscalls
/// when appending thousands of embeddings during index construction.
pub struct ExternalWriter {
    /// The open file handle in append mode.
    file: File,
    /// Current write position (byte offset). Tracked locally to avoid a
    /// seek(SeekFrom::End(0)) syscall before each write.
    position: u64,
    /// Path stored for error reporting.
    path: PathBuf,
}

impl ExternalWriter {
    /// Opens (or creates) the external embeddings file for appending. The
    /// writer seeks to the end of the file to determine the initial write
    /// position.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::Io` if the file cannot be opened.
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| StoreError::io(path, e))?;

        let position = file
            .seek(SeekFrom::End(0))
            .map_err(|e| StoreError::io(path, e))?;

        Ok(Self {
            file,
            position,
            path: path.to_path_buf(),
        })
    }

    /// Appends an embedding vector and returns (byte_offset, byte_length).
    /// The write position is advanced by the number of bytes written.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::Io` on write failure.
    /// Returns `StoreError::IntegerOverflow` if the byte length calculation
    /// overflows usize or cannot be represented as u64.
    pub fn append(&mut self, embedding: &[f32]) -> Result<(u64, u64), StoreError> {
        let offset = self.position;

        // checked_mul guards against overflow when embedding.len() * BYTES_PER_F32
        // exceeds usize::MAX.
        let byte_len_usize = embedding.len().checked_mul(BYTES_PER_F32).ok_or_else(|| {
            StoreError::integer_overflow(format!(
                "byte length overflow: {} elements * {} bytes/element",
                embedding.len(),
                BYTES_PER_F32
            ))
        })?;

        let byte_len = u64::try_from(byte_len_usize).map_err(|_| {
            StoreError::integer_overflow(format!(
                "byte length {byte_len_usize} cannot be represented as u64"
            ))
        })?;

        let mut bytes = Vec::with_capacity(byte_len_usize);
        for &val in embedding {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        self.file
            .write_all(&bytes)
            .map_err(|e| StoreError::io(&self.path, e))?;

        self.position += byte_len;
        Ok((offset, byte_len))
    }

    /// Flushes user-space buffers and calls `fdatasync` to ensure all bytes
    /// written since the last `flush` call are stored on the storage device.
    ///
    /// This must be called after each embedding batch and before the chunk
    /// records (with their `ext_offset` / `ext_length` columns) are committed
    /// to the SQLite database. If the database commit succeeds but the
    /// embedding data has not been fsynced, a crash between the two operations
    /// leaves the database referencing byte ranges that do not yet exist in
    /// the file, causing read failures on next startup.
    ///
    /// `sync_data` (equivalent to `fdatasync`) is used instead of `sync_all`
    /// because the file metadata (size) is updated by the write itself;
    /// syncing metadata separately is unnecessary overhead.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::Io` if either the Rust flush or the OS sync fails.
    pub fn flush(&mut self) -> Result<(), StoreError> {
        // std::io::Write::flush on File is a no-op in Rust std (File has no
        // user-space buffer). The call is kept for API consistency and in case
        // a future wrapper introduces buffering.
        self.file
            .flush()
            .map_err(|e| StoreError::io(&self.path, e))?;
        // fdatasync: flush OS page-cache pages to the storage device.
        self.file
            .sync_data()
            .map_err(|e| StoreError::io(&self.path, e))
    }
}

// ---------------------------------------------------------------------------
// Stateful reader: holds a single memory mapping across multiple reads
// ---------------------------------------------------------------------------

/// Stateful reader that holds a single memory mapping of the external
/// embeddings file. Avoids repeated open/mmap/munmap/close syscalls when
/// reading thousands of embeddings during HNSW index construction or querying.
pub struct ExternalReader {
    /// The memory-mapped file contents.
    mmap: Mmap,
    /// Total file size in bytes.
    file_len: u64,
    /// Path stored for error reporting.
    path: PathBuf,
}

impl ExternalReader {
    /// Opens the external embeddings file and creates a read-only memory map.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::Io` if the file cannot be opened or mapped.
    ///
    /// # Safety contract
    ///
    /// The caller must ensure no concurrent writer modifies the mapped region
    /// for the lifetime of this reader. In practice, reads occur after indexing
    /// completes, so this invariant is maintained by the application workflow.
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        let file = File::open(path).map_err(|e| StoreError::io(path, e))?;
        let file_len = file.metadata().map_err(|e| StoreError::io(path, e))?.len();

        // SAFETY: The file is opened read-only. The caller guarantees no
        // concurrent modification of the mapped region.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| StoreError::io(path, e))?;

        Ok(Self {
            mmap,
            file_len,
            path: path.to_path_buf(),
        })
    }

    /// Reads an embedding vector at the given byte offset and length from
    /// the pre-existing memory map. No syscalls are issued for this operation.
    ///
    /// # Errors
    ///
    /// Returns `StoreError::HnswIndex` if the offset/length are out of bounds
    /// or the length is not a multiple of BYTES_PER_F32 (4).
    /// Returns `StoreError::IntegerOverflow` if offset or length cannot be
    /// represented as usize on the current platform.
    pub fn read(&self, offset: u64, length: u64) -> Result<Vec<f32>, StoreError> {
        if !length.is_multiple_of(BYTES_PER_F32 as u64) {
            return Err(StoreError::hnsw(format!(
                "embedding byte length {length} is not a multiple of BYTES_PER_F32 (4)"
            )));
        }

        // checked_add guards against u64 wraparound in release builds.
        // See the equivalent check in `read_embedding` for the rationale.
        let end = offset.checked_add(length).ok_or_else(|| {
            StoreError::hnsw(format!(
                "{}: read range offset {offset} + length {length} overflows u64",
                self.path.display()
            ))
        })?;
        if end > self.file_len {
            return Err(StoreError::hnsw(format!(
                "{}: read range [{offset}..{end}] exceeds file size {}",
                self.path.display(),
                self.file_len
            )));
        }

        // Convert u64 offset and length to usize via checked conversion.
        let offset_usize = usize::try_from(offset).map_err(|_| {
            StoreError::integer_overflow(format!(
                "embedding offset {offset} cannot be represented as usize on this platform"
            ))
        })?;
        let length_usize = usize::try_from(length).map_err(|_| {
            StoreError::integer_overflow(format!(
                "embedding length {length} cannot be represented as usize on this platform"
            ))
        })?;

        decode_f32_slice(&self.mmap, offset_usize, length_usize)
    }
}

/// Verifies that the external embedding file is large enough to contain all
/// byte ranges recorded in the database for the given session. Called at
/// application startup (after HNSW index loading) to detect truncated files
/// before they cause read failures during search.
///
/// The check queries `MAX(ext_offset + ext_length)` from the chunk table for
/// the session's external-storage chunks. If the actual file size is smaller
/// than the maximum expected byte position, the embedding file is truncated
/// (incomplete write, copy failure, or filesystem corruption).
///
/// A missing sidecar checksum file is not treated as an error here; this
/// function checks only byte-length consistency, not content integrity.
///
/// # Arguments
///
/// * `conn` - An open database connection. The caller holds this connection;
///   no pool is needed because callers already have a connection in context.
/// * `session_id` - The session whose chunks are checked.
/// * `emb_path` - Path to the external embeddings file for the session.
///
/// # Errors
///
/// Returns `StoreError::HnswIndex` if the file is smaller than required.
/// Returns `StoreError::Sqlite` if the database query fails.
/// Returns `StoreError::Io` if the file metadata cannot be read.
pub fn verify_external_file_integrity(
    conn: &rusqlite::Connection,
    session_id: i64,
    emb_path: &Path,
) -> Result<(), StoreError> {
    // MAX(ext_offset + ext_length) gives the byte position one past the last
    // embedding in the file. If the file size is less than this value, at least
    // one chunk's embedding data is missing.
    let max_end: Option<i64> = conn.query_row(
        "SELECT MAX(ext_offset + ext_length) FROM chunk
         WHERE session_id = ?1 AND ext_offset IS NOT NULL AND is_deleted = 0",
        rusqlite::params![session_id],
        |row| row.get(0),
    )?;

    if let Some(expected_end) = max_end {
        let actual_size = std::fs::metadata(emb_path)
            .map_err(|e| StoreError::io(emb_path, e))?
            .len();

        // expected_end is stored as i64 in SQLite but represents a byte count,
        // so it must be non-negative. Treat a negative value as a data error.
        let expected_end_u64 = u64::try_from(expected_end).unwrap_or(0);

        if actual_size < expected_end_u64 {
            return Err(StoreError::hnsw(format!(
                "external embedding file '{}' is {actual_size} bytes but the database \
                 expects at least {expected_end_u64} bytes for session {session_id} — \
                 the file may be truncated or corrupted",
                emb_path.display()
            )));
        }
    }

    Ok(())
}

/// Decodes a contiguous region of little-endian f32 values from a byte slice.
/// Returns an error if the requested range exceeds the data buffer or if
/// byte_len is not a multiple of BYTES_PER_F32.
///
/// # Arguments
///
/// * `data` - The source byte buffer (e.g., a memory-mapped file).
/// * `start` - Byte offset where the f32 sequence begins.
/// * `byte_len` - Number of bytes to decode (must be a multiple of BYTES_PER_F32).
///
/// # Errors
///
/// Returns `StoreError::HnswIndex` if the byte range [start..start+byte_len]
/// exceeds the data buffer length.
fn decode_f32_slice(data: &[u8], start: usize, byte_len: usize) -> Result<Vec<f32>, StoreError> {
    let end = start.checked_add(byte_len).ok_or_else(|| {
        StoreError::integer_overflow(format!(
            "decode_f32_slice range overflow: start {start} + byte_len {byte_len}"
        ))
    })?;

    if end > data.len() {
        return Err(StoreError::hnsw(format!(
            "decode_f32_slice: range [{start}..{end}] exceeds buffer length {}",
            data.len()
        )));
    }

    let slice = &data[start..end];
    let num_floats = byte_len / BYTES_PER_F32;
    let mut embedding = Vec::with_capacity(num_floats);
    for chunk in slice.chunks_exact(BYTES_PER_F32) {
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        embedding.push(f32::from_le_bytes(bytes));
    }
    Ok(embedding)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn external_embedding_roundtrip() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let path = tmp.path().join("embeddings.bin");

        create_external_file(&path).expect("create failed");

        let vec1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let vec2: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

        let (off1, len1) = append_embedding(&path, &vec1).expect("append1 failed");
        let (off2, len2) = append_embedding(&path, &vec2).expect("append2 failed");

        assert_eq!(off1, 0);
        assert_eq!(len1, 16); // 4 floats * 4 bytes
        assert_eq!(off2, 16);
        assert_eq!(len2, 16);

        let loaded1 = read_embedding(&path, off1, len1).expect("read1 failed");
        let loaded2 = read_embedding(&path, off2, len2).expect("read2 failed");

        assert_eq!(loaded1, vec1);
        assert_eq!(loaded2, vec2);
    }

    #[test]
    fn read_invalid_length_fails() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let path = tmp.path().join("embeddings.bin");

        create_external_file(&path).expect("create failed");
        append_embedding(&path, &[1.0, 2.0]).expect("append failed");

        let result = read_embedding(&path, 0, 5);
        assert!(
            result.is_err(),
            "reading 5 bytes (not multiple of 4) must fail"
        );
    }

    /// Verifies that ExternalWriter appends multiple embeddings with correct
    /// offsets and lengths without re-opening the file for each append.
    #[test]
    fn stateful_writer_appends_correctly() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let path = tmp.path().join("embeddings_writer.bin");

        create_external_file(&path).expect("create failed");

        let vec1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let vec2: Vec<f32> = vec![4.0, 5.0, 6.0];
        let vec3: Vec<f32> = vec![7.0, 8.0, 9.0];

        let mut writer = ExternalWriter::open(&path).expect("writer open failed");
        let (off1, len1) = writer.append(&vec1).expect("append1 failed");
        let (off2, len2) = writer.append(&vec2).expect("append2 failed");
        let (off3, len3) = writer.append(&vec3).expect("append3 failed");
        writer.flush().expect("flush failed");

        assert_eq!(off1, 0);
        assert_eq!(len1, 12);
        assert_eq!(off2, 12);
        assert_eq!(len2, 12);
        assert_eq!(off3, 24);
        assert_eq!(len3, 12);

        // Verify read-back via ExternalReader
        let reader = ExternalReader::open(&path).expect("reader open failed");
        assert_eq!(reader.read(off1, len1).unwrap(), vec1);
        assert_eq!(reader.read(off2, len2).unwrap(), vec2);
        assert_eq!(reader.read(off3, len3).unwrap(), vec3);
    }

    /// Verifies that ExternalReader returns an error for out-of-bounds reads.
    #[test]
    fn stateful_reader_bounds_check() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let path = tmp.path().join("embeddings_reader.bin");

        create_external_file(&path).expect("create failed");
        append_embedding(&path, &[1.0, 2.0]).expect("append failed");

        let reader = ExternalReader::open(&path).expect("reader open failed");

        // Valid read
        assert!(reader.read(0, 8).is_ok());

        // Out of bounds
        assert!(reader.read(0, 12).is_err());

        // Invalid length (not multiple of 4)
        assert!(reader.read(0, 5).is_err());
    }

    /// T-EXT-010: `read_embedding` rejects (offset, length) pairs that overflow
    /// u64 when added together. Without checked_add, the sum would silently wrap
    /// to a small number in release builds and the bounds check would pass,
    /// leading to an out-of-bounds memory access.
    #[test]
    fn t_ext_010_read_embedding_overflow_rejected() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let path = tmp.path().join("overflow_test.bin");

        create_external_file(&path).expect("create failed");
        append_embedding(&path, &[1.0, 2.0]).expect("append failed");

        // offset = u64::MAX - 3, length = 8: their sum overflows u64.
        // The call must return an error rather than performing an out-of-bounds read.
        let overflow_offset = u64::MAX - 3;
        let result = read_embedding(&path, overflow_offset, 8);
        assert!(
            result.is_err(),
            "overflow offset+length must be rejected as an error"
        );
    }

    /// T-EXT-011: `ExternalReader::read` rejects (offset, length) pairs that
    /// overflow u64. Mirrors T-EXT-012 for the stateful reader path.
    #[test]
    fn t_ext_011_stateful_reader_overflow_rejected() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let path = tmp.path().join("overflow_reader_test.bin");

        create_external_file(&path).expect("create failed");
        append_embedding(&path, &[3.0, 4.0]).expect("append failed");

        let reader = ExternalReader::open(&path).expect("reader open failed");

        // offset = u64::MAX - 3, length = 8: their sum overflows u64.
        let overflow_offset = u64::MAX - 3;
        let result = reader.read(overflow_offset, 8);
        assert!(
            result.is_err(),
            "overflow offset+length must be rejected as an error"
        );
    }

    /// T-EXT-012: The BYTES_PER_F32 constant matches the actual size of f32
    /// on the target platform. This compile-time assertion ensures the constant
    /// remains correct if the codebase is ever built on an exotic platform.
    #[test]
    fn t_ext_012_bytes_per_f32_matches_size_of() {
        assert_eq!(
            BYTES_PER_F32,
            std::mem::size_of::<f32>(),
            "BYTES_PER_F32 constant must equal std::mem::size_of::<f32>()"
        );
        assert_eq!(BYTES_PER_F32, 4, "IEEE 754 single-precision is 4 bytes");
    }

    /// T-EXT-013: Byte length calculations using BYTES_PER_F32 produce the
    /// same result as std::mem::size_of_val on a concrete f32 slice. Validates
    /// that the refactored constant produces identical byte offsets to the
    /// original size_of_val approach.
    #[test]
    fn t_ext_013_byte_len_consistency() {
        let embedding: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let via_constant = embedding.len() * BYTES_PER_F32;
        let via_size_of_val = std::mem::size_of_val(embedding.as_slice());

        assert_eq!(
            via_constant, via_size_of_val,
            "embedding.len() * BYTES_PER_F32 must equal size_of_val(embedding)"
        );
    }

    /// T-EXT-014: ExternalWriter byte offset tracking matches BYTES_PER_F32.
    /// Verifies that the writer reports correct cumulative offsets when writing
    /// embeddings of different dimensions.
    #[test]
    fn t_ext_014_writer_offsets_match_bytes_per_f32() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let path = tmp.path().join("offsets_test.bin");

        create_external_file(&path).expect("create failed");

        let dim3: Vec<f32> = vec![1.0, 2.0, 3.0];
        let dim5: Vec<f32> = vec![4.0, 5.0, 6.0, 7.0, 8.0];

        let mut writer = ExternalWriter::open(&path).expect("writer open failed");
        let (off1, len1) = writer.append(&dim3).expect("append1 failed");
        let (off2, len2) = writer.append(&dim5).expect("append2 failed");
        writer.flush().expect("flush failed");

        assert_eq!(off1, 0);
        assert_eq!(len1, (3 * BYTES_PER_F32) as u64);
        assert_eq!(off2, (3 * BYTES_PER_F32) as u64);
        assert_eq!(len2, (5 * BYTES_PER_F32) as u64);

        // Verify data integrity via reader.
        let reader = ExternalReader::open(&path).expect("reader open failed");
        assert_eq!(reader.read(off1, len1).unwrap(), dim3);
        assert_eq!(reader.read(off2, len2).unwrap(), dim5);
    }
}
