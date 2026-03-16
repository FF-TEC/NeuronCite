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

// Integration test entry point for the neuroncite binary crate.
//
// This file serves as the root of the integration test compilation unit.
// It declares the common test utilities module and the integration test
// submodules. Each submodule contains tests that exercise multi-crate
// interactions across the NeuronCite workspace.
//
// Cargo compiles files in the `tests/` directory as separate crate roots.
// This single entry point allows all integration test submodules to share
// the common module's helper functions and stub types.

mod common;
mod integration;
