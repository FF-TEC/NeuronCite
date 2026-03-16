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

// Error types for the neuroncite-store crate.
//
// StoreError covers SQLite query failures, connection pool exhaustion,
// schema migration issues, manifest parsing errors, HNSW index corruption,
// filesystem I/O errors during external embedding storage operations,
// integer overflow conditions, and entity-not-found conditions.

use std::path::PathBuf;

/// Represents all error conditions that can occur within the storage layer.
/// Each variant maps to a distinct failure category, allowing callers to
/// pattern-match on the error source without inspecting message strings.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// A `SQLite` query or transaction failed. Wraps the underlying `rusqlite`
    /// error, providing automatic conversion via the `?` operator.
    #[error("SQLite error: {source}")]
    Sqlite {
        /// The underlying `rusqlite::Error`.
        #[from]
        source: rusqlite::Error,
    },

    /// The r2d2 connection pool could not provide a connection within the
    /// configured timeout, or pool initialization failed.
    #[error("connection pool error: {reason}")]
    Pool {
        /// A human-readable description of the pool failure.
        reason: String,
    },

    /// A schema migration step failed, leaving the database in a potentially
    /// inconsistent state. The version field identifies which migration failed.
    #[error("schema migration failed at version {version}: {reason}")]
    Migration {
        /// The schema version number that failed to apply.
        version: u32,
        /// A human-readable description of the migration failure.
        reason: String,
    },

    /// An I/O error occurred while reading or writing external embedding files,
    /// the manifest file, or HNSW index files on disk.
    #[error("I/O error at path {path}: {source}")]
    Io {
        /// The file system path where the I/O error occurred.
        path: PathBuf,
        /// The underlying `std::io::Error`.
        source: std::io::Error,
    },

    /// The HNSW index is corrupted, contains unexpected data, or could not
    /// be serialized/deserialized.
    #[error("HNSW index error: {reason}")]
    HnswIndex {
        /// A human-readable description of the index corruption or failure.
        reason: String,
    },

    /// The manifest file (`models.lock`) could not be parsed or written.
    #[error("manifest error: {reason}")]
    Manifest {
        /// A human-readable description of the manifest failure.
        reason: String,
    },

    /// The requested entity (session, file, page, or chunk) was not found in
    /// the database. The entity field identifies the type, and the id field
    /// carries the lookup key.
    #[error("{entity} not found: {id}")]
    NotFound {
        /// The type of entity that was not found (e.g., "session", "file").
        entity: String,
        /// The identifier that was looked up.
        id: String,
    },

    /// An integer value from the database could not be converted to the target
    /// type (e.g., i64 -> usize). This indicates database corruption or a
    /// platform incompatibility where the stored value exceeds the target
    /// type's range.
    #[error("integer conversion overflow: {reason}")]
    IntegerOverflow {
        /// A human-readable description of the overflow condition.
        reason: String,
    },
}

/// Convenience constructors for `StoreError` variants that take string arguments.
/// These reduce boilerplate at call sites by accepting `impl Into<String>`.
impl StoreError {
    /// Creates a `Pool` variant from any string-convertible type.
    pub fn pool(reason: impl Into<String>) -> Self {
        Self::Pool {
            reason: reason.into(),
        }
    }

    /// Creates a `Migration` variant from a version number and reason string.
    pub fn migration(version: u32, reason: impl Into<String>) -> Self {
        Self::Migration {
            version,
            reason: reason.into(),
        }
    }

    /// Creates an `Io` variant from a path and an `std::io::Error`.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Creates an `HnswIndex` variant from any string-convertible type.
    pub fn hnsw(reason: impl Into<String>) -> Self {
        Self::HnswIndex {
            reason: reason.into(),
        }
    }

    /// Creates a `Manifest` variant from any string-convertible type.
    pub fn manifest(reason: impl Into<String>) -> Self {
        Self::Manifest {
            reason: reason.into(),
        }
    }

    /// Creates a `NotFound` variant from entity type and identifier strings.
    pub fn not_found(entity: impl Into<String>, id: impl Into<String>) -> Self {
        Self::NotFound {
            entity: entity.into(),
            id: id.into(),
        }
    }

    /// Creates an `IntegerOverflow` variant from any string-convertible type.
    pub fn integer_overflow(reason: impl Into<String>) -> Self {
        Self::IntegerOverflow {
            reason: reason.into(),
        }
    }
}
