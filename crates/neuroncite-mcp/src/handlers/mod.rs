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

//! MCP tool handler implementations.
//!
//! Each submodule contains one handler function that bridges a single MCP tool
//! call to the NeuronCite subsystems (store, search, embed). Handlers receive
//! the shared `AppState` and the deserialized parameters from the `tools/call`
//! request, execute the operation, and return a JSON value for the MCP response
//! content array.
//!
//! All handlers return `Result<serde_json::Value, String>` where the error
//! string is a human-readable message that the dispatch layer wraps into a
//! JSON-RPC error response.

pub mod annotate;
pub mod annotate_status;
pub mod batch_content;
pub mod batch_search;
pub mod bib_report;
pub mod chunks;
pub mod citation;
pub mod common;
pub mod compare;
pub mod compare_search;
pub mod content;
pub mod discover;
pub mod export;
pub mod files;
pub mod html_crawl;
pub mod html_fetch;
pub mod index;
pub mod index_add;
pub mod inspect;
pub mod jobs;
pub mod models;
pub mod multi_search;
pub mod preview_chunks;
pub mod quality;
pub mod reindex;
pub mod remove;
pub mod reranker;
pub mod search;
pub mod session_diff;
pub mod sessions;
pub mod source_fetch;
pub mod system;
pub mod text_search;
