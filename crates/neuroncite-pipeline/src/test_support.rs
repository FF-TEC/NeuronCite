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

// Shared test utilities for the neuroncite-pipeline crate.
//
// This module provides stub implementations of `EmbeddingBackend` used by
// the test modules in worker.rs and executor.rs. It is compiled only under
// `#[cfg(test)]` and is not part of the public API.
//
// The `dead_code` lint is suppressed because not every test module imports
// every helper, and unused items would trigger false positives under
// `#![deny(warnings)]`.
#![allow(dead_code)]

use neuroncite_core::{EmbeddingBackend, ModelInfo, NeuronCiteError};

// ---------------------------------------------------------------------------
// Stub embedding backend
// ---------------------------------------------------------------------------

/// Deterministic embedding backend for tests. Produces vectors of a configurable
/// dimensionality where the first element is 1.0 and the remaining elements are
/// 0.0. This makes distance calculations predictable without requiring a real
/// ONNX Runtime session.
///
/// Common configurations:
///
/// - `StubBackend::default()` -- 4-dimensional vectors, model ID "stub-model".
/// - `StubBackend::with_dimension(384)` -- 384-dimensional vectors.
pub struct StubBackend {
    /// Number of dimensions in the embedding vectors produced by this backend.
    pub dimension: usize,
    /// Hugging Face model identifier reported by `loaded_model_id()`.
    pub model_id: String,
}

impl Default for StubBackend {
    /// Creates a 4-dimensional stub backend with model ID "stub-model".
    fn default() -> Self {
        Self {
            dimension: 4,
            model_id: "stub-model".to_string(),
        }
    }
}

impl StubBackend {
    /// Creates a stub backend with the specified vector dimensionality and
    /// the default model ID "stub-model".
    #[must_use]
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            dimension,
            model_id: "stub-model".to_string(),
        }
    }
}

impl EmbeddingBackend for StubBackend {
    fn name(&self) -> &str {
        "stub"
    }

    fn vector_dimension(&self) -> usize {
        self.dimension
    }

    fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
        Ok(())
    }

    /// Returns one deterministic embedding vector per input text. Each vector
    /// has `self.dimension` elements: the first is 1.0, the rest are 0.0.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
        Ok(texts
            .iter()
            .map(|_| {
                let mut v = vec![0.0_f32; self.dimension];
                v[0] = 1.0;
                v
            })
            .collect())
    }

    fn supports_gpu(&self) -> bool {
        false
    }

    fn available_models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            id: self.model_id.clone(),
            display_name: "Stub Model".to_string(),
            vector_dimension: self.dimension,
            backend: "stub".to_string(),
        }]
    }

    fn loaded_model_id(&self) -> String {
        self.model_id.clone()
    }
}
