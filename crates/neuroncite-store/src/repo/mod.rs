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

// Repository layer: CRUD operations for persistent entities.
//
// Each submodule handles a single entity type (session, file, page, chunk) and
// provides functions that accept a borrowed `rusqlite::Connection`. These
// functions are transaction-agnostic -- the caller controls transaction
// boundaries at a higher level (typically in the workflow module).

pub mod aggregate;
pub mod annotation_status;
pub mod chunk;
pub mod diff;
pub mod file;
pub mod page;
pub mod session;
pub mod web_source;
