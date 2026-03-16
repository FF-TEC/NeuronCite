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

// Index layer: HNSW vector index, FTS5 keyword index, and external embedding
// storage.
//
// - `hnsw` -- Manages the in-memory HNSW graph for approximate nearest neighbor
//   search. The index is built from embeddings stored in the database or in
//   external files, and is hot-reloaded via `arc-swap` when new embeddings are
//   added.
// - `fts` -- Manages the SQLite FTS5 virtual table for BM25 keyword search.
// - `external` -- Manages memory-mapped files for storing embedding vectors
//   outside the SQLite database, reducing database size and enabling zero-copy
//   access to large embedding matrices.

pub mod external;
pub mod fts;
pub mod hnsw;
