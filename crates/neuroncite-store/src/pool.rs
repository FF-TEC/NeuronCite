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

// `SQLite` connection pool management.
//
// Wraps the r2d2 connection pool with `r2d2_sqlite::SqliteConnectionManager`
// to provide thread-safe, pooled access to a single `SQLite` database file. The
// pool is configured with WAL journal mode and busy timeout settings appropriate
// for concurrent read/write access from the API server and background indexing
// workers.
//
// Configuration per the architecture document (Section 7.2):
// - WAL mode for concurrent readers with a single writer
// - `foreign_keys` ON for cascading deletes
// - `busy_timeout` 5000ms for write lock contention
// - max 8 connections (1 writer, 7 readers)
// - `mmap_size` 256 MB for memory-mapped database I/O
// - `cache_size` -65536 (64 MB page cache)
// - `temp_store` MEMORY for in-memory temporary tables
// - `journal_size_limit` 64 MB for WAL checkpoint threshold

use std::path::Path;

use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;

use crate::error::StoreError;

/// Creates and configures an r2d2 connection pool backed by `SQLite`.
///
/// Each connection in the pool is initialized with the PRAGMA settings
/// specified in the architecture document. The pool maintains up to
/// `pool_size` connections for concurrent read access alongside a single
/// writer.
///
/// # Arguments
///
/// * `db_path` - Filesystem path to the `SQLite` database file. The file
///   and its parent directory must exist, or `SQLite` will create them.
/// * `pool_size` - Maximum number of connections in the pool. Defaults to
///   8 when `AppConfig::db_pool_size` is used. Pass 0 or values > 64 to
///   fall back to the default of 8.
///
/// # Errors
///
/// Returns `StoreError::Pool` if the pool cannot be built or if the
/// initial PRAGMA configuration fails on any connection.
pub fn create_pool(db_path: &Path) -> Result<Pool<SqliteConnectionManager>, StoreError> {
    create_pool_with_size(db_path, 8)
}

/// Creates a connection pool with an explicit maximum size.
/// See `create_pool` for PRAGMA configuration details.
pub fn create_pool_with_size(
    db_path: &Path,
    pool_size: u32,
) -> Result<Pool<SqliteConnectionManager>, StoreError> {
    let effective_size = if pool_size == 0 || pool_size > 64 {
        8
    } else {
        pool_size
    };
    let manager = SqliteConnectionManager::file(db_path).with_init(|conn| {
        // busy_timeout MUST be set before any write-acquiring PRAGMAs. When
        // r2d2 creates multiple pool connections concurrently, journal_mode=WAL
        // requires an exclusive lock on the database file. Without busy_timeout
        // set first, concurrent connections attempting journal_mode=WAL receive
        // SQLITE_BUSY immediately (default timeout is 0ms), causing "database
        // is locked" errors during pool initialization.
        conn.execute_batch("PRAGMA busy_timeout = 5000;")?;

        // WAL mode enables concurrent readers while a single writer is active.
        // This PRAGMA requires an exclusive lock, which is why busy_timeout
        // must be set above to handle contention during concurrent pool init.
        conn.execute_batch("PRAGMA journal_mode = WAL;")?;

        // In WAL mode, NORMAL provides the same crash-safety guarantees as FULL
        // (the WAL is always consistent after a crash), but skips one fsync per
        // commit. This removes the single largest write-throughput bottleneck.
        conn.execute_batch("PRAGMA synchronous = NORMAL;")?;

        // Foreign key enforcement is required for ON DELETE CASCADE to function.
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;

        // Memory-mapped I/O for the database file (256 MB). Reduces read
        // system calls by mapping database pages directly into the process
        // address space.
        conn.execute_batch("PRAGMA mmap_size = 268435456;")?;

        // Increase the page cache to 64 MB (negative value = size in KiB).
        // Keeps frequently accessed chunk metadata in memory.
        conn.execute_batch("PRAGMA cache_size = -65536;")?;

        // Store temporary tables and indices in memory rather than on disk.
        conn.execute_batch("PRAGMA temp_store = MEMORY;")?;

        // The WAL file is checkpointed when it exceeds 64 MB (67108864 bytes).
        conn.execute_batch("PRAGMA journal_size_limit = 67108864;")?;

        // wal_autocheckpoint controls how many WAL pages accumulate before
        // SQLite automatically triggers a passive checkpoint. The default is
        // 1000 pages (4 MB at 4 KB/page), which is far below journal_size_limit
        // (64 MB). Setting both to equivalent values (16000 pages ≈ 64 MB)
        // prevents the WAL from being checkpointed excessively early, which
        // would stall write operations. A passive checkpoint does not block
        // readers or writers; it only copies WAL pages to the main file when
        // no read transaction is holding them.
        conn.execute_batch("PRAGMA wal_autocheckpoint = 16000;")?;

        Ok(())
    });

    Pool::builder()
        .max_size(effective_size)
        .build(manager)
        .map_err(|e| StoreError::pool(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Verifies that `create_pool` produces a pool whose connections have
    /// WAL mode, foreign keys enabled, and the configured busy timeout.
    #[test]
    fn pool_pragmas_are_set() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let db_path = tmp.path().join("test.db");

        let pool = create_pool(&db_path).expect("pool creation failed");
        let conn = pool.get().expect("failed to get connection from pool");

        // Check WAL mode
        let journal_mode: String = conn
            .query_row("PRAGMA journal_mode;", [], |row| row.get(0))
            .expect("failed to query journal_mode");
        assert_eq!(journal_mode, "wal");

        // Check foreign_keys
        let fk: i64 = conn
            .query_row("PRAGMA foreign_keys;", [], |row| row.get(0))
            .expect("failed to query foreign_keys");
        assert_eq!(fk, 1);

        // Check busy_timeout
        let timeout: i64 = conn
            .query_row("PRAGMA busy_timeout;", [], |row| row.get(0))
            .expect("failed to query busy_timeout");
        assert_eq!(timeout, 5000);

        // Check synchronous = NORMAL (value 1). FULL is 2, OFF is 0.
        let sync: i64 = conn
            .query_row("PRAGMA synchronous;", [], |row| row.get(0))
            .expect("failed to query synchronous");
        assert_eq!(
            sync, 1,
            "synchronous must be NORMAL (1) for WAL write throughput"
        );
    }

    /// Verifies that 8 threads can concurrently acquire pool connections and
    /// execute queries without triggering "database is locked" errors. This
    /// test validates that `busy_timeout` is set before `journal_mode = WAL`
    /// in the connection init closure, preventing SQLITE_BUSY when multiple
    /// connections initialize simultaneously.
    #[test]
    fn concurrent_pool_access_does_not_produce_database_locked() {
        let tmp = TempDir::new().expect("failed to create temp dir");
        let db_path = tmp.path().join("concurrent_test.db");

        let pool = create_pool(&db_path).expect("pool creation failed");

        // Create a table to exercise write operations.
        {
            let conn = pool.get().expect("failed to get initial connection");
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, val TEXT);",
            )
            .expect("failed to create test table");
        }

        // Spawn 8 threads (matching pool max_size) that all acquire connections
        // and perform read/write operations concurrently.
        let mut handles = Vec::with_capacity(8);
        for thread_idx in 0..8 {
            let pool = pool.clone();
            handles.push(std::thread::spawn(move || {
                let conn = pool.get().unwrap_or_else(|e| {
                    panic!("thread {thread_idx}: failed to get connection: {e}");
                });
                conn.execute(
                    "INSERT OR IGNORE INTO test_table (id, val) VALUES (?1, ?2);",
                    rusqlite::params![thread_idx, format!("thread_{thread_idx}")],
                )
                .unwrap_or_else(|e| {
                    panic!("thread {thread_idx}: INSERT failed: {e}");
                });
                let count: i64 = conn
                    .query_row("SELECT COUNT(*) FROM test_table;", [], |row| row.get(0))
                    .unwrap_or_else(|e| {
                        panic!("thread {thread_idx}: SELECT failed: {e}");
                    });
                assert!(count > 0, "thread {thread_idx}: table should have rows");
            }));
        }

        for (idx, handle) in handles.into_iter().enumerate() {
            handle
                .join()
                .unwrap_or_else(|_| panic!("thread {idx} panicked"));
        }
    }
}
