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

//! Per-endpoint request handler functions.
//!
//! Each submodule contains the axum handler function(s) for a single API endpoint
//! group. Handlers extract request parameters from the axum `State` and `Json`
//! extractors, delegate to the appropriate service layer (search, store, worker),
//! and return structured JSON responses.

pub mod annotate;
pub mod backends;
pub mod chunks;
pub mod citation;
pub mod compare;
pub mod discover;
pub mod documents;
pub mod health;
pub mod index;
pub mod jobs;
pub mod quality;
pub mod search;
pub mod sessions;
pub mod shutdown;
pub mod verify;
