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

// Integration test submodule declarations for the neuroncite binary crate.
//
// Each submodule contains integration tests that exercise multi-crate
// interactions. These tests import from the workspace library crates
// directly and use shared test utilities from the common module.

pub mod cli;
pub mod pipeline;
pub mod real_pdf;
pub mod search;
pub mod server;

// Pdfium fallback integration tests: exercise the pdfium extraction backend
// against real PDFs that the default pdf-extract backend cannot parse
// (cross-reference errors, CID-range panics, Type1 font panics). Only
// compiled when the "pdfium" feature is enabled. The pdfium shared library
// (pdfium.dll / libpdfium.so) must be available at runtime; tests gracefully
// skip when the library is absent.
#[cfg(feature = "pdfium")]
pub mod ocr_fallback;
