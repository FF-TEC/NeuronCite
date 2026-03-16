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

//! ONNX Runtime backend implementation (feature-gated: `backend-ort`).
//!
//! Provides `EmbeddingBackend` and `Reranker` trait implementations using the `ort`
//! crate for ONNX model inference. The CUDA execution provider is preferred when
//! available; the CPU execution provider is used as a fallback.

pub mod embedder;
pub mod reranker;
pub mod session;
