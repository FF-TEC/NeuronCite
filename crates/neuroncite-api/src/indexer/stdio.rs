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

//! Stdout suppression utilities re-exported from neuroncite-pipeline.
//!
//! Stdout suppression and MCP writer flag logic lives in
//! `neuroncite-pipeline::indexer::stdio`. This shim re-exports everything so
//! that paths such as `crate::indexer::suppress_stdout` continue to resolve
//! in callers inside neuroncite-api.

pub use neuroncite_pipeline::indexer::stdio::*;
