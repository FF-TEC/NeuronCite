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

// Property test module declarations for the NeuronCite integration test suite.
//
// Each submodule contains property-based tests that verify invariants across
// many randomized inputs. These tests use manual pseudo-random input generation
// (linear congruential generators with fixed seeds) rather than an external
// property testing framework, ensuring deterministic and reproducible results.

mod chunking_properties;
mod embedding_properties;
mod search_properties;
