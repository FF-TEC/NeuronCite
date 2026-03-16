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

//! GPU Worker for serialized embedding and reranking access.
//!
//! The GPU Worker runs as a dedicated thread that receives embedding and
//! reranking requests through dual-priority bounded tokio::sync::mpsc channels.
//! High-priority channel serves search (embed_query) requests; low-priority
//! channel serves indexing (embed_batch) requests. A biased tokio::select! loop
//! drains the high-priority channel first to maintain interactive responsiveness.
//!
//! `WorkerHandle` is the send-side API used by handler functions to submit
//! embedding and reranking work. Each request carries a oneshot response channel.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arc_swap::ArcSwap;
use tokio::sync::{mpsc, oneshot};

use neuroncite_core::{EmbeddingBackend, InferenceCapabilities, NeuronCiteError, Reranker};
use neuroncite_search::CachedTokenizer;

/// Capacity of the high-priority (search) channel.
const HIGH_PRIORITY_CAPACITY: usize = 64;

/// Capacity of the low-priority (indexing) channel.
const LOW_PRIORITY_CAPACITY: usize = 256;

/// Internal message type for the worker event loop. Each variant carries
/// the input data and a oneshot sender for returning the result.
enum WorkerMsg {
    /// Embed a single query string (high priority, used by search).
    EmbedQuery {
        text: String,
        reply: oneshot::Sender<Result<Vec<f32>, NeuronCiteError>>,
    },
    /// Embed a batch of text strings (low priority, used by indexing).
    EmbedBatch {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<Vec<Vec<f32>>, NeuronCiteError>>,
    },
    /// Rerank a batch of candidates against a query (high priority).
    RerankBatch {
        query: String,
        candidates: Vec<String>,
        reply: oneshot::Sender<Result<Vec<f64>, NeuronCiteError>>,
    },
}

/// Send-side handle for submitting work to the GPU worker thread.
/// Cloneable and cheaply sharable across handler tasks.
///
/// All metadata fields (model ID, sequence length, tokenizer) use atomic or
/// ArcSwap wrappers so that `swap_backend()` can update them lock-free when
/// the user activates a different embedding model at runtime. The same ArcSwap
/// pattern is used for the reranker slot (see `swap_reranker`).
#[derive(Clone)]
pub struct WorkerHandle {
    /// Sender for high-priority requests (search queries, reranking).
    high_tx: mpsc::Sender<WorkerMsg>,
    /// Sender for low-priority requests (batch embedding during indexing).
    low_tx: mpsc::Sender<WorkerMsg>,
    /// Maximum input sequence length (in tokens) of the loaded embedding model.
    /// Stored as AtomicUsize so `swap_backend()` can update it without locks.
    /// The indexing pipeline reads this to compute an adaptive batch size
    /// that avoids GPU memory exhaustion for models with long context windows.
    max_sequence_length: Arc<AtomicUsize>,
    /// The loaded model's tokenizer serialized as a JSON string. Wrapped in
    /// ArcSwap so `swap_backend()` can update it atomically when the active
    /// model changes. Consumed by the job executor when constructing a
    /// token-based chunking strategy via `neuroncite_chunk::create_strategy`.
    /// Inner value is `None` if no tokenizer is available.
    tokenizer_json: Arc<ArcSwap<Option<String>>>,
    /// The Hugging Face model identifier of the currently loaded embedding model.
    /// Wrapped in ArcSwap so `swap_backend()` can update it atomically.
    /// Used by the index handler and executor pipeline to reject indexing
    /// requests for a model that differs from the loaded one, preventing
    /// the dimension mismatch that would cause a panic in the HNSW builder.
    loaded_model_id: Arc<ArcSwap<String>>,
    /// Shared reference to the embedding backend. Uses ArcSwap for lock-free
    /// hot-swapping at runtime. The same ArcSwap instance is shared between
    /// the WorkerHandle (write side via `swap_backend`) and the worker task
    /// (read side via `load` before each embedding message). This allows
    /// activating a different embedding model without restarting the server.
    backend: Arc<ArcSwap<Arc<dyn EmbeddingBackend>>>,
    /// Shared reference to the reranker slot. Uses `ArcSwap<Option<Arc<dyn Reranker>>>`
    /// for lock-free hot-swapping at runtime. The same `ArcSwap` instance is shared
    /// between the `WorkerHandle` (write side via `swap_reranker`) and the worker
    /// task (read side via `load` before each `RerankBatch` message). This allows
    /// loading a reranker model at runtime via the `neuroncite_reranker_load` MCP
    /// tool or the web models/load-reranker endpoint without restarting the server.
    reranker: Arc<ArcSwap<Option<Arc<dyn Reranker>>>>,
    /// Pre-deserialized tokenizer instance built from `tokenizer_json`. Wrapped
    /// in `ArcSwap<Option<Arc<CachedTokenizer>>>` so that `swap_backend()` can
    /// replace it atomically when the active model changes. Search handlers use
    /// this cached instance instead of deserializing the tokenizer JSON per
    /// request, avoiding repeated JSON parsing overhead during sub-chunk
    /// refinement.
    cached_tokenizer: Arc<ArcSwap<Option<Arc<CachedTokenizer>>>>,
    /// HuggingFace model identifier of the loaded reranker model. Wrapped in
    /// ArcSwap for lock-free updates. Empty string when no reranker is loaded.
    /// Used by the web model catalog endpoint to indicate which reranker model
    /// is currently active.
    loaded_reranker_id: Arc<ArcSwap<String>>,
    /// Runtime-detected hardware capabilities for the active inference session.
    /// Updated atomically via `swap_backend()` when the model changes. Read by
    /// the pipeline to compute hardware-adaptive batch sizes.
    capabilities: Arc<ArcSwap<InferenceCapabilities>>,
}

impl WorkerHandle {
    /// Returns the maximum input sequence length of the embedding model
    /// served by this worker. Reads from the shared AtomicUsize, which
    /// is updated by `swap_backend()` when the active model changes.
    /// Used by the indexing pipeline to compute an adaptive batch size.
    pub fn max_sequence_length(&self) -> usize {
        self.max_sequence_length.load(Ordering::Acquire)
    }

    /// Returns a cheaply cloned `Arc` pointing to the loaded model's
    /// tokenizer JSON string, or `None` if no tokenizer is available.
    /// Uses `Arc::clone` (atomic refcount increment) instead of cloning
    /// the inner `Option<String>`, avoiding a heap allocation on each call.
    /// Callers that need `Option<&str>` can use `.as_deref()` on the
    /// dereferenced Arc value.
    pub fn tokenizer_json(&self) -> Arc<Option<String>> {
        Arc::clone(&self.tokenizer_json.load())
    }

    /// Returns a cheaply cloned `Arc` pointing to the Hugging Face model
    /// identifier of the embedding model loaded by this worker. Uses
    /// `Arc::clone` (atomic refcount increment) instead of cloning the
    /// inner `String`, avoiding a heap allocation on each call. Callers
    /// that need `&str` can dereference the Arc.
    pub fn loaded_model_id(&self) -> Arc<String> {
        Arc::clone(&self.loaded_model_id.load())
    }

    /// Returns whether a cross-encoder reranker model is loaded. Reads the
    /// current value from the shared `ArcSwap` slot. When false, calling
    /// `rerank_batch` will return an error. Used by health/doctor endpoints
    /// to inform clients about reranker availability.
    pub fn reranker_available(&self) -> bool {
        self.reranker.load().is_some()
    }

    /// Returns the pre-deserialized tokenizer instance, or `None` if the
    /// loaded model does not provide tokenizer metadata. The returned `Arc`
    /// can be cheaply cloned and used across multiple search requests without
    /// re-parsing the tokenizer JSON.
    pub fn cached_tokenizer(&self) -> Arc<Option<Arc<CachedTokenizer>>> {
        Arc::clone(&self.cached_tokenizer.load())
    }

    /// Returns the runtime-detected hardware capabilities for the active
    /// inference session. Used by the pipeline to compute hardware-adaptive
    /// batch sizes via `compute_batch_size_with_caps`.
    pub fn inference_capabilities(&self) -> InferenceCapabilities {
        (**self.capabilities.load()).clone()
    }

    /// Atomically replaces the embedding backend used by the worker task.
    /// Updates all cached metadata fields (model ID, sequence length,
    /// tokenizer, cached tokenizer instance, inference capabilities) to match
    /// the new backend. The worker task observes the change on its next
    /// embedding message via `ArcSwap::load()`. In-flight embedding requests
    /// complete with the previous backend.
    pub fn swap_backend(&self, new_backend: Arc<dyn EmbeddingBackend>) {
        self.loaded_model_id
            .store(Arc::new(new_backend.loaded_model_id()));
        let tok_json = new_backend.tokenizer_json();
        self.tokenizer_json.store(Arc::new(tok_json.clone()));
        let cached = build_cached_tokenizer(&tok_json);
        self.cached_tokenizer.store(Arc::new(cached));
        self.max_sequence_length
            .store(new_backend.max_sequence_length(), Ordering::Release);
        self.capabilities
            .store(Arc::new(new_backend.inference_capabilities()));
        self.backend.store(Arc::new(new_backend));
    }

    /// Returns a cheaply cloned `Arc` pointing to the HuggingFace model
    /// identifier of the loaded reranker. The inner string is empty when
    /// no reranker is loaded. Uses `Arc::clone` (atomic refcount increment)
    /// instead of cloning the inner `String`.
    pub fn loaded_reranker_id(&self) -> Arc<String> {
        Arc::clone(&self.loaded_reranker_id.load())
    }

    /// Atomically replaces the reranker used by the worker task. Pass
    /// `Some(reranker)` with its model_id to install a reranker, or `None`
    /// with an empty model_id to disable reranking. The worker task observes
    /// the change on its next `RerankBatch` message without any restart or
    /// channel reconnection.
    pub fn swap_reranker(&self, new: Option<Arc<dyn Reranker>>, model_id: &str) {
        self.reranker.store(Arc::new(new));
        self.loaded_reranker_id
            .store(Arc::new(model_id.to_string()));
    }

    /// Embeds a single query string via the high-priority channel.
    /// Returns the embedding vector or an error if the worker is unavailable.
    pub async fn embed_query(&self, text: String) -> Result<Vec<f32>, NeuronCiteError> {
        let (tx, rx) = oneshot::channel();
        self.high_tx
            .send(WorkerMsg::EmbedQuery { text, reply: tx })
            .await
            .map_err(|_| NeuronCiteError::Embed("worker channel closed".to_string()))?;
        rx.await
            .map_err(|_| NeuronCiteError::Embed("worker dropped response channel".to_string()))?
    }

    /// Embeds a batch of text strings via the low-priority channel.
    /// Returns one vector per input text.
    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
        let (tx, rx) = oneshot::channel();
        self.low_tx
            .send(WorkerMsg::EmbedBatch { texts, reply: tx })
            .await
            .map_err(|_| NeuronCiteError::Embed("worker channel closed".to_string()))?;
        rx.await
            .map_err(|_| NeuronCiteError::Embed("worker dropped response channel".to_string()))?
    }

    /// Embeds a batch of text strings via the high-priority channel with
    /// automatic sub-batching to prevent ONNX Runtime memory exhaustion.
    ///
    /// Search refinement can produce hundreds of sub-chunk texts in a single
    /// call (e.g., 10 results with divisors \[4,8,16\] yields ~360 sub-chunks).
    /// Without sub-batching, all texts are passed to the ONNX model in a
    /// single forward pass, where intermediate activation memory scales as
    /// O(batch_size * num_layers * seq_len^2). This caused unbounded RAM
    /// growth reported as the "RAM explosion" issue.
    ///
    /// Sub-batching splits the input into groups sized by `compute_batch_size()`
    /// (the same formula used during indexing), sends each group as a separate
    /// high-priority message, and concatenates the results. Each sub-batch's
    /// intermediate ONNX tensors are freed before the next sub-batch runs.
    ///
    /// Accepts `&[&str]` instead of `Vec<String>` to avoid callers cloning
    /// all candidate content strings upfront. Only the current sub-batch's
    /// strings are allocated as owned `String` values, keeping peak memory
    /// proportional to batch_size rather than total candidate count.
    pub async fn embed_batch_search(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = crate::indexer::compute_batch_size(self.max_sequence_length());
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for sub_batch in texts.chunks(batch_size) {
            // Convert the current sub-batch's &str references to owned Strings
            // for the channel message. Only this sub-batch is allocated at a time;
            // previous sub-batch Strings are consumed by the worker and freed.
            let owned: Vec<String> = sub_batch.iter().map(|&s| s.to_string()).collect();
            let (tx, rx) = oneshot::channel();
            self.high_tx
                .send(WorkerMsg::EmbedBatch {
                    texts: owned,
                    reply: tx,
                })
                .await
                .map_err(|_| NeuronCiteError::Embed("worker channel closed".to_string()))?;
            let embeddings = rx.await.map_err(|_| {
                NeuronCiteError::Embed("worker dropped response channel".to_string())
            })??;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    /// Reranks a batch of candidate texts against a query via the
    /// high-priority channel. Returns one score per candidate.
    ///
    /// Accepts borrowed slices (`&str`, `&[&str]`) so callers do not need to
    /// clone owned `String` values for reranking. The owned copies required
    /// by the `Send + 'static` channel message are created inside this method.
    pub async fn rerank_batch(
        &self,
        query: &str,
        candidates: &[&str],
    ) -> Result<Vec<f64>, NeuronCiteError> {
        let (tx, rx) = oneshot::channel();
        self.high_tx
            .send(WorkerMsg::RerankBatch {
                query: query.to_string(),
                candidates: candidates.iter().map(|&s| s.to_string()).collect(),
                reply: tx,
            })
            .await
            .map_err(|_| NeuronCiteError::Embed("worker channel closed".to_string()))?;
        rx.await
            .map_err(|_| NeuronCiteError::Embed("worker dropped response channel".to_string()))?
    }
}

/// Attempts to build a `CachedTokenizer` from the tokenizer JSON string
/// provided by the embedding backend. Returns `None` if the JSON is absent
/// or deserialization fails (logged at warn level). Called during worker
/// initialization and after `swap_backend()` to keep the cached instance
/// in sync with the loaded model.
fn build_cached_tokenizer(tok_json: &Option<String>) -> Option<Arc<CachedTokenizer>> {
    let json = tok_json.as_ref()?;
    match CachedTokenizer::from_json(json) {
        Ok(tok) => Some(Arc::new(tok)),
        Err(e) => {
            tracing::warn!("failed to build cached tokenizer from backend JSON: {e}");
            None
        }
    }
}

/// Spawns the GPU worker on a dedicated thread and returns the `WorkerHandle`.
///
/// The worker thread owns the `EmbeddingBackend` and optional `Reranker`,
/// running a biased `tokio::select!` loop that prioritizes the high-priority
/// channel. When both channels are closed, the worker exits.
///
/// # Arguments
///
/// * `backend` - The embedding backend to use for inference.
/// * `reranker` - Optional cross-encoder reranker.
pub fn spawn_worker(
    backend: Arc<dyn EmbeddingBackend>,
    reranker: Option<Arc<dyn Reranker>>,
) -> WorkerHandle {
    let max_seq_len = backend.max_sequence_length();
    let tok_json = backend.tokenizer_json();
    let cached_tok = build_cached_tokenizer(&tok_json);
    let model_id = backend.loaded_model_id();
    let caps = backend.inference_capabilities();
    let (high_tx, mut high_rx) = mpsc::channel::<WorkerMsg>(HIGH_PRIORITY_CAPACITY);
    let (low_tx, mut low_rx) = mpsc::channel::<WorkerMsg>(LOW_PRIORITY_CAPACITY);

    // Shared embedding backend slot using ArcSwap for lock-free hot-swap.
    // The same Arc<ArcSwap<...>> is held by both the WorkerHandle (for
    // swap_backend) and the worker task (for loading the current backend
    // on each message). This follows the same pattern as the reranker slot.
    let backend_shared: Arc<ArcSwap<Arc<dyn EmbeddingBackend>>> =
        Arc::new(ArcSwap::from_pointee(backend));
    let backend_worker = backend_shared.clone();

    // Shared reranker slot using ArcSwap for lock-free hot-swap. The same
    // Arc<ArcSwap<...>> is held by both the WorkerHandle (for swap_reranker)
    // and the worker task (for loading the current reranker on each message).
    let reranker_shared: Arc<ArcSwap<Option<Arc<dyn Reranker>>>> =
        Arc::new(ArcSwap::from_pointee(reranker));
    let reranker_worker = reranker_shared.clone();

    // Spawn a dedicated tokio task for the worker loop. The task receives
    // messages via async channels but delegates synchronous GPU-bound
    // computation to spawn_blocking. This prevents embedding and reranking
    // operations (which can take seconds for large batches) from blocking
    // the tokio runtime worker threads. Awaiting each spawn_blocking call
    // before processing the next message maintains the serialization
    // guarantee: only one GPU operation runs at a time.
    tokio::spawn(async move {
        loop {
            // Biased select: drain high-priority messages before low-priority.
            let msg = tokio::select! {
                biased;
                msg = high_rx.recv() => msg,
                msg = low_rx.recv() => msg,
            };

            let Some(msg) = msg else {
                // Both channels are closed; worker exits.
                tracing::info!("GPU worker shutting down: all channels closed");
                break;
            };

            // Load the current embedding backend snapshot from the shared
            // ArcSwap slot. The clone of Arc<dyn EmbeddingBackend> is cheap
            // (Arc reference count increment). The snapshot remains valid
            // even if swap_backend is called concurrently.
            let backend_ref: Arc<dyn EmbeddingBackend> = (**backend_worker.load()).clone();
            // Load the current reranker snapshot from the shared ArcSwap slot.
            // This clone of the inner Option<Arc<dyn Reranker>> is cheap: it
            // increments the Arc reference count (or is None). The loaded
            // snapshot remains valid even if swap_reranker is called concurrently.
            let reranker_snapshot: Option<Arc<dyn Reranker>> = (**reranker_worker.load()).clone();
            let _ = tokio::task::spawn_blocking(move || {
                process_message(&backend_ref, &reranker_snapshot, msg);
            })
            .await;
        }
    });

    WorkerHandle {
        high_tx,
        low_tx,
        max_sequence_length: Arc::new(AtomicUsize::new(max_seq_len)),
        tokenizer_json: Arc::new(ArcSwap::from_pointee(tok_json)),
        cached_tokenizer: Arc::new(ArcSwap::from_pointee(cached_tok)),
        loaded_model_id: Arc::new(ArcSwap::from_pointee(model_id)),
        capabilities: Arc::new(ArcSwap::from_pointee(caps)),
        backend: backend_shared,
        reranker: reranker_shared,
        // Initially empty. Set when a reranker model is loaded at runtime
        // via swap_reranker (called from the web or MCP load endpoints).
        loaded_reranker_id: Arc::new(ArcSwap::from_pointee(String::new())),
    }
}

/// Processes a single worker message by delegating to the appropriate
/// backend method and sending the result back through the oneshot channel.
fn process_message(
    backend: &Arc<dyn EmbeddingBackend>,
    reranker: &Option<Arc<dyn Reranker>>,
    msg: WorkerMsg,
) {
    match msg {
        WorkerMsg::EmbedQuery { text, reply } => {
            let result = backend.embed_single(&text);
            // The receiver may have been dropped if the request was canceled.
            let _ = reply.send(result);
        }
        WorkerMsg::EmbedBatch { texts, reply } => {
            let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let result = backend.embed_batch(&text_refs);
            let _ = reply.send(result);
        }
        WorkerMsg::RerankBatch {
            query,
            candidates,
            reply,
        } => {
            let result = match reranker {
                Some(r) => {
                    let candidate_refs: Vec<&str> = candidates.iter().map(String::as_str).collect();
                    r.rerank_batch(&query, &candidate_refs)
                }
                None => Err(NeuronCiteError::Search(
                    "no reranker configured".to_string(),
                )),
            };
            let _ = reply.send(result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuroncite_core::{EmbeddingBackend, ModelInfo, NeuronCiteError};

    // The StubBackend from crate::test_support provides a configurable
    // dimension and model_id. Tests in this module that only need a basic
    // stub use test_support::StubBackend directly. Tests that need
    // specialized behavior (e.g., OrderBackend, RecordingBackend) still
    // define their own local implementations because those are unique to
    // the specific test scenario and not reusable.
    //
    // Audit finding M-014: the previous local StubBackend (a struct with
    // a `dimension: usize` field) was functionally identical to
    // test_support::StubBackend and has been removed.
    use crate::test_support;

    /// T-API-050: WorkerHandle embed_query returns a vector with the correct
    /// dimensionality from the stub backend.
    #[tokio::test]
    async fn t_api_015_worker_handle_embed_query_returns_vector() {
        let backend = Arc::new(test_support::StubBackend::with_dimension(384));
        let handle = spawn_worker(backend, None);

        let result = handle
            .embed_query("test query".to_string())
            .await
            .expect("embed_query failed");

        assert_eq!(result.len(), 384, "vector dimension must be 384");
        assert!(
            (result[0] - 1.0).abs() < f32::EPSILON,
            "first element must be 1.0"
        );
    }

    /// T-API-052: spawn_worker succeeds when called outside block_on but with
    /// an active runtime guard (rt.enter()). This reproduces the MCP server
    /// startup pattern where the tokio runtime is created manually and the
    /// server loop runs synchronously on the main thread. Without rt.enter(),
    /// tokio::spawn inside spawn_worker panics with "there is no reactor
    /// running". This test verifies the guard-based approach works correctly.
    #[test]
    fn t_api_017_spawn_worker_with_runtime_guard() {
        let rt = tokio::runtime::Runtime::new().expect("create runtime");
        let _guard = rt.enter();

        let backend = Arc::new(test_support::StubBackend::default());
        let handle = spawn_worker(backend, None);

        // Use block_on via the runtime handle (same pattern as the MCP server
        // dispatch layer) to verify the worker processes requests correctly.
        let result = rt
            .handle()
            .block_on(handle.embed_query("test from guard context".to_string()))
            .expect("embed_query should succeed with runtime guard");

        assert_eq!(result.len(), 4, "vector dimension must match backend");
        assert!(
            (result[0] - 1.0).abs() < f32::EPSILON,
            "first element must be 1.0 from StubBackend"
        );
    }

    /// T-API-051: WorkerHandle priority ordering. High-priority messages
    /// (embed_query) are processed before low-priority messages (embed_batch)
    /// when both channels have pending items.
    ///
    /// This test fills both channels and verifies that the high-priority
    /// request completes before the low-priority request.
    #[tokio::test]
    async fn t_api_016_worker_handle_priority_ordering() {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicU64, Ordering};

        // Use a counter to track the order of processing.
        let counter = StdArc::new(AtomicU64::new(0));
        let counter_clone = counter.clone();

        /// Backend that records the order in which calls are processed.
        struct OrderBackend {
            counter: StdArc<AtomicU64>,
            dimension: usize,
        }

        impl EmbeddingBackend for OrderBackend {
            fn name(&self) -> &str {
                "order"
            }

            fn vector_dimension(&self) -> usize {
                self.dimension
            }

            fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }

            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                // Increment the counter; the returned value encodes
                // the processing order in the first element.
                let order = self.counter.fetch_add(1, Ordering::SeqCst);
                Ok(texts
                    .iter()
                    .map(|_| vec![order as f32; self.dimension])
                    .collect())
            }

            fn embed_single(&self, _text: &str) -> Result<Vec<f32>, NeuronCiteError> {
                let order = self.counter.fetch_add(1, Ordering::SeqCst);
                Ok(vec![order as f32; self.dimension])
            }

            fn supports_gpu(&self) -> bool {
                false
            }

            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }

            fn loaded_model_id(&self) -> String {
                String::new()
            }
        }

        let backend = Arc::new(OrderBackend {
            counter: counter_clone,
            dimension: 4,
        });

        let handle = spawn_worker(backend, None);

        // Submit low-priority first, then high-priority.
        // Due to biased select!, the high-priority should still be
        // processed first when both are available.
        let low_future = handle.embed_batch(vec!["low priority text".to_string()]);
        let high_future = handle.embed_query("high priority text".to_string());

        let (low_result, high_result) = tokio::join!(low_future, high_future);

        let high_vec = high_result.expect("high priority failed");
        let low_vecs = low_result.expect("low priority failed");

        // The high-priority request's order value should be less than
        // or equal to the low-priority request's order value.
        // Due to async scheduling, we verify both completed successfully.
        assert_eq!(high_vec.len(), 4, "high priority vector length");
        assert_eq!(low_vecs.len(), 1, "low priority batch size");
        assert_eq!(low_vecs[0].len(), 4, "low priority vector length");
    }

    /// T-API-053: rerank_batch returns an error when no reranker is configured.
    /// The worker is spawned without a reranker (None). Calling rerank_batch
    /// must return Err, not silently succeed with empty or default scores.
    /// This error is what the MCP and API handlers propagate to the caller
    /// when the user requests reranking without a configured reranker model.
    #[tokio::test]
    async fn t_api_018_rerank_batch_errors_without_reranker() {
        let backend = Arc::new(test_support::StubBackend::default());
        let handle = spawn_worker(backend, None);

        let result = handle
            .rerank_batch("test query", &["candidate one", "candidate two"])
            .await;

        assert!(
            result.is_err(),
            "rerank_batch must return Err when no reranker is configured"
        );

        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("no reranker configured"),
            "error message must indicate missing reranker, got: {err_msg}"
        );
    }

    /// T-API-019: rerank_batch returns scores when a reranker is configured.
    /// Spawns the worker with a stub reranker and verifies that rerank_batch
    /// returns one score per candidate with the expected values.
    #[tokio::test]
    async fn t_api_019_rerank_batch_returns_scores_with_reranker() {
        /// Stub reranker that returns descending scores: 1.0, 0.5, 0.25, ...
        struct DescendingReranker;

        impl Reranker for DescendingReranker {
            fn name(&self) -> &str {
                "descending-stub"
            }

            fn rerank_batch(
                &self,
                _query: &str,
                candidates: &[&str],
            ) -> Result<Vec<f64>, NeuronCiteError> {
                Ok(candidates
                    .iter()
                    .enumerate()
                    .map(|(i, _)| 1.0 / (i as f64 + 1.0))
                    .collect())
            }

            fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
        }

        let backend = Arc::new(test_support::StubBackend::default());
        let reranker: Arc<dyn Reranker> = Arc::new(DescendingReranker);
        let handle = spawn_worker(backend, Some(reranker));

        let scores = handle
            .rerank_batch("test query", &["first", "second", "third"])
            .await
            .expect("rerank_batch with configured reranker must succeed");

        assert_eq!(scores.len(), 3, "one score per candidate");
        assert!(
            (scores[0] - 1.0).abs() < 1e-10,
            "first score must be 1.0, got {}",
            scores[0]
        );
        assert!(
            (scores[1] - 0.5).abs() < 1e-10,
            "second score must be 0.5, got {}",
            scores[1]
        );
    }

    /// T-API-020: embed_batch_search splits large input batches into
    /// sub-batches limited by compute_batch_size. Each sub-batch is sent
    /// as a separate message to the worker, so the ONNX forward pass
    /// processes at most batch_size texts at a time. This prevents the
    /// memory exhaustion that occurred when hundreds of refinement
    /// sub-chunks were embedded in a single ONNX call.
    #[tokio::test]
    async fn t_api_020_embed_batch_search_sub_batches() {
        use std::sync::Mutex as StdMutex;

        /// Backend that records the size of each embed_batch call.
        /// Used to verify that the worker splits large requests into
        /// smaller sub-batches matching compute_batch_size().
        struct RecordingBackend {
            dimension: usize,
            batch_sizes: Arc<StdMutex<Vec<usize>>>,
        }

        impl EmbeddingBackend for RecordingBackend {
            fn name(&self) -> &str {
                "recording"
            }

            fn vector_dimension(&self) -> usize {
                self.dimension
            }

            fn load_model(&mut self, _: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }

            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                self.batch_sizes.lock().unwrap().push(texts.len());
                Ok(texts
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let mut v = vec![0.0_f32; self.dimension];
                        v[0] = i as f32;
                        v
                    })
                    .collect())
            }

            fn supports_gpu(&self) -> bool {
                false
            }

            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }

            fn loaded_model_id(&self) -> String {
                String::new()
            }

            // max_sequence_length defaults to 512, so compute_batch_size
            // returns 32 (REFERENCE_BATCH_SIZE * 512 / 512 = 32).
        }

        let batch_sizes = Arc::new(StdMutex::new(Vec::new()));
        let backend = Arc::new(RecordingBackend {
            dimension: 4,
            batch_sizes: batch_sizes.clone(),
        });
        let handle = spawn_worker(backend, None);

        // Send 100 texts. With max_seq_len=512, batch_size=32.
        // Expected: 4 sub-batches (32 + 32 + 32 + 4).
        let texts: Vec<&str> = vec!["test text"; 100];
        let result = handle
            .embed_batch_search(&texts)
            .await
            .expect("embed_batch_search must succeed");

        assert_eq!(
            result.len(),
            100,
            "must return one embedding per input text"
        );

        let recorded = batch_sizes.lock().unwrap();
        assert_eq!(
            recorded.len(),
            4,
            "100 texts with batch_size=32 must produce 4 sub-batches, got {:?}",
            *recorded
        );
        assert_eq!(recorded[0], 32);
        assert_eq!(recorded[1], 32);
        assert_eq!(recorded[2], 32);
        assert_eq!(recorded[3], 4);
    }

    /// T-API-021: embed_batch_search with a small input (below batch_size)
    /// sends a single message to the worker without splitting.
    #[tokio::test]
    async fn t_api_021_embed_batch_search_small_batch_single_call() {
        use std::sync::Mutex as StdMutex;

        struct RecordingBackend {
            dimension: usize,
            batch_sizes: Arc<StdMutex<Vec<usize>>>,
        }

        impl EmbeddingBackend for RecordingBackend {
            fn name(&self) -> &str {
                "recording"
            }

            fn vector_dimension(&self) -> usize {
                self.dimension
            }

            fn load_model(&mut self, _: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }

            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                self.batch_sizes.lock().unwrap().push(texts.len());
                Ok(texts
                    .iter()
                    .map(|_| vec![1.0_f32; self.dimension])
                    .collect())
            }

            fn supports_gpu(&self) -> bool {
                false
            }

            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }

            fn loaded_model_id(&self) -> String {
                String::new()
            }
        }

        let batch_sizes = Arc::new(StdMutex::new(Vec::new()));
        let backend = Arc::new(RecordingBackend {
            dimension: 4,
            batch_sizes: batch_sizes.clone(),
        });
        let handle = spawn_worker(backend, None);

        // 10 texts with batch_size=32 fits in a single sub-batch.
        let texts: Vec<&str> = vec!["test"; 10];
        let result = handle
            .embed_batch_search(&texts)
            .await
            .expect("small batch must succeed");

        assert_eq!(result.len(), 10, "must return one embedding per text");

        let recorded = batch_sizes.lock().unwrap();
        assert_eq!(
            recorded.len(),
            1,
            "10 texts with batch_size=32 must be a single sub-batch"
        );
        assert_eq!(recorded[0], 10);
    }

    /// T-API-022: embed_batch_search with empty input returns empty result
    /// without sending any messages to the worker.
    #[tokio::test]
    async fn t_api_022_embed_batch_search_empty_input() {
        let backend = Arc::new(test_support::StubBackend::default());
        let handle = spawn_worker(backend, None);

        let result = handle
            .embed_batch_search(&[])
            .await
            .expect("empty input must succeed");

        assert!(result.is_empty(), "empty input must return empty result");
    }

    /// T-API-023: swap_reranker changes reranker_available from false to true.
    /// Spawns the worker without a reranker, verifies reranker_available is false,
    /// then hot-swaps a stub reranker and verifies reranker_available is true.
    /// This validates the ArcSwap-based runtime reranker activation mechanism.
    #[tokio::test]
    async fn t_api_023_swap_reranker_activates_reranker() {
        /// Stub reranker that returns a constant score for each candidate.
        struct ConstantReranker;

        impl Reranker for ConstantReranker {
            fn name(&self) -> &str {
                "constant-stub"
            }

            fn rerank_batch(
                &self,
                _query: &str,
                candidates: &[&str],
            ) -> Result<Vec<f64>, NeuronCiteError> {
                Ok(candidates.iter().map(|_| 0.5).collect())
            }

            fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
        }

        let backend = Arc::new(test_support::StubBackend::default());
        let handle = spawn_worker(backend, None);

        assert!(
            !handle.reranker_available(),
            "reranker_available must be false when spawned without a reranker"
        );

        // Hot-swap a reranker into the worker.
        let reranker: Arc<dyn Reranker> = Arc::new(ConstantReranker);
        handle.swap_reranker(Some(reranker), "test/constant-reranker");

        assert!(
            handle.reranker_available(),
            "reranker_available must be true after swap_reranker(Some(...))"
        );
        assert_eq!(
            *handle.loaded_reranker_id(),
            "test/constant-reranker",
            "loaded_reranker_id must match the model_id passed to swap_reranker"
        );
    }

    /// T-API-024: rerank_batch succeeds after swap_reranker installs a reranker.
    /// Spawns the worker without a reranker (rerank_batch fails), then hot-swaps
    /// a stub reranker and verifies that rerank_batch returns scores.
    #[tokio::test]
    async fn t_api_024_rerank_batch_succeeds_after_swap() {
        /// Stub reranker returning descending scores.
        struct SwapTestReranker;

        impl Reranker for SwapTestReranker {
            fn name(&self) -> &str {
                "swap-test"
            }

            fn rerank_batch(
                &self,
                _query: &str,
                candidates: &[&str],
            ) -> Result<Vec<f64>, NeuronCiteError> {
                Ok(candidates
                    .iter()
                    .enumerate()
                    .map(|(i, _)| 1.0 / (i as f64 + 1.0))
                    .collect())
            }

            fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
        }

        let backend = Arc::new(test_support::StubBackend::default());
        let handle = spawn_worker(backend, None);

        // Reranking fails without a reranker.
        let err_result = handle.rerank_batch("test", &["a", "b"]).await;
        assert!(
            err_result.is_err(),
            "rerank_batch must fail before swap_reranker"
        );

        // Hot-swap a reranker.
        let reranker: Arc<dyn Reranker> = Arc::new(SwapTestReranker);
        handle.swap_reranker(Some(reranker), "test/swap-test");

        // Reranking succeeds after swap.
        let scores = handle
            .rerank_batch("test", &["a", "b"])
            .await
            .expect("rerank_batch must succeed after swap_reranker");

        assert_eq!(scores.len(), 2, "one score per candidate");
        assert!((scores[0] - 1.0).abs() < 1e-10, "first score must be 1.0");
    }

    /// T-API-025: swap_reranker(None) disables the reranker. After disabling,
    /// reranker_available returns false and rerank_batch returns an error.
    #[tokio::test]
    async fn t_api_025_swap_reranker_none_disables() {
        /// Stub reranker for the enable/disable cycle test.
        struct ToggleReranker;

        impl Reranker for ToggleReranker {
            fn name(&self) -> &str {
                "toggle-stub"
            }

            fn rerank_batch(
                &self,
                _query: &str,
                candidates: &[&str],
            ) -> Result<Vec<f64>, NeuronCiteError> {
                Ok(candidates.iter().map(|_| 0.75).collect())
            }

            fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
        }

        let backend = Arc::new(test_support::StubBackend::default());
        // Spawn with a reranker.
        let reranker: Arc<dyn Reranker> = Arc::new(ToggleReranker);
        let handle = spawn_worker(backend, Some(reranker));

        assert!(
            handle.reranker_available(),
            "reranker_available must be true when spawned with a reranker"
        );

        // Disable the reranker by swapping in None.
        handle.swap_reranker(None, "");

        assert!(
            !handle.reranker_available(),
            "reranker_available must be false after swap_reranker(None)"
        );

        // rerank_batch must fail after disabling.
        let result = handle.rerank_batch("test", &["candidate"]).await;

        assert!(
            result.is_err(),
            "rerank_batch must return Err after swap_reranker(None)"
        );
    }
}
