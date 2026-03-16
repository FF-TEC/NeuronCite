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

//! Server-Sent Event (SSE) endpoints for real-time streaming to the browser.
//!
//! Provides four SSE streams that replace the per-frame `try_recv()` polling
//! used by the egui GUI:
//!
//! - `/logs` -- Structured log messages from the tracing subscriber
//! - `/progress` -- Indexing progress updates (files done, chunks created)
//! - `/jobs` -- Job state transitions (queued, running, completed, failed)
//! - `/models` -- Model download progress, switch completion, reranker load
//!
//! Each stream uses a `tokio::sync::broadcast` channel. The SSE handler
//! subscribes to the channel and converts each received message into an
//! SSE `Event`. A 15-second keep-alive interval prevents proxy timeouts.
//!
//! Connection counting: each SSE handler increments `WebState::sse_connections`
//! on entry and decrements it on stream termination (via a Drop guard). When
//! the counter exceeds MAX_SSE_CONNECTIONS, the handler returns 503 Service
//! Unavailable instead of opening a new stream.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use crate::WebState;

/// Maximum number of concurrent SSE connections across all event streams.
/// Each SSE connection holds a tokio broadcast receiver and an HTTP connection
/// open for the lifetime of the stream. Capping at 100 prevents resource
/// exhaustion from runaway browser tabs or misbehaving clients that open
/// connections without closing them.
const MAX_SSE_CONNECTIONS: usize = 100;

/// RAII guard that decrements the SSE connection counter when dropped.
/// Ensures the counter is decremented even if the stream is dropped due
/// to client disconnect, server shutdown, or panic unwinding.
struct SseConnectionGuard {
    state: Arc<WebState>,
}

impl Drop for SseConnectionGuard {
    fn drop(&mut self) {
        // Release ordering ensures that all memory writes performed while
        // holding the connection slot are visible to threads that later
        // observe a lower connection count via Acquire loads.
        self.state.sse_connections.fetch_sub(1, Ordering::Release);
    }
}

/// Attempts to acquire an SSE connection slot. Returns `Ok(SseConnectionGuard)`
/// if the current count is below MAX_SSE_CONNECTIONS, or `Err(Response)` with
/// 503 Service Unavailable if the limit is exceeded.
#[allow(clippy::result_large_err)]
fn try_acquire_sse_slot(state: &Arc<WebState>) -> Result<SseConnectionGuard, Response> {
    // AcqRel: the fetch_add acquires the current count so that we observe
    // any concurrent increments before making our limit decision, and
    // releases our own increment so subsequent threads see it.
    let prev = state.sse_connections.fetch_add(1, Ordering::AcqRel);
    if prev >= MAX_SSE_CONNECTIONS {
        // Undo the increment since we are rejecting this connection.
        // Release ordering makes the decrement visible to threads checking
        // the counter after this rejection.
        state.sse_connections.fetch_sub(1, Ordering::Release);
        return Err((StatusCode::SERVICE_UNAVAILABLE, "too many SSE connections").into_response());
    }
    Ok(SseConnectionGuard {
        state: Arc::clone(state),
    })
}

/// Filters a BroadcastStream, logging any receiver lag events at debug level.
/// Returns only the successfully received messages. When a client falls behind
/// the broadcast sender, the lagged messages are dropped and the count is logged
/// so operators can diagnose slow SSE consumers or undersized channel buffers.
fn filter_lagged(
    result: Result<String, tokio_stream::wrappers::errors::BroadcastStreamRecvError>,
    channel_name: &'static str,
) -> Option<String> {
    match result {
        Ok(msg) => Some(msg),
        Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
            tracing::debug!(
                channel = channel_name,
                skipped = n,
                "SSE client lagged behind"
            );
            None
        }
    }
}

/// Builds the SSE router with all streaming endpoints.
pub fn sse_routes(state: Arc<WebState>) -> Router {
    Router::new()
        .route("/logs", get(log_stream))
        .route("/progress", get(progress_stream))
        .route("/jobs", get(job_stream))
        .route("/models", get(model_stream))
        .route("/citation", get(citation_stream))
        .route("/sources", get(source_stream))
        .with_state(state)
}

/// SSE endpoint that streams structured log messages. Each log event from the
/// tracing subscriber is broadcast to all connected SSE clients.
async fn log_stream(
    State(state): State<Arc<WebState>>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, Response> {
    let guard = try_acquire_sse_slot(&state)?;
    let rx = state.log_tx.subscribe();
    let stream = BroadcastStream::new(rx)
        .filter_map(|result| filter_lagged(result, "logs"))
        .map(|msg| Ok(Event::default().event("log").data(msg)));

    // Move the guard into the stream's async context so it is dropped when the
    // stream terminates (client disconnect or server shutdown).
    let stream = stream.map(move |item| {
        let _keep = &guard;
        item
    });

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    ))
}

/// SSE endpoint that streams indexing progress updates and completion signals.
/// Each message is a JSON object with an "event" field indicating the SSE event
/// type ("index_progress" or "index_complete"). The event type is extracted from
/// the JSON and used as the SSE event name so the browser's EventSource
/// dispatches to the correct handler.
async fn progress_stream(
    State(state): State<Arc<WebState>>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, Response> {
    let guard = try_acquire_sse_slot(&state)?;
    let rx = state.progress_tx.subscribe();
    let stream = BroadcastStream::new(rx)
        .filter_map(|result| filter_lagged(result, "progress"))
        .map(|msg| {
            let event_type = serde_json::from_str::<serde_json::Value>(&msg)
                .ok()
                .and_then(|v| v["event"].as_str().map(String::from))
                .unwrap_or_else(|| "index_progress".to_string());
            Ok(Event::default().event(event_type).data(msg))
        });

    let stream = stream.map(move |item| {
        let _keep = &guard;
        item
    });

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    ))
}

/// SSE endpoint that streams job state transitions. Each event represents a
/// job changing state (queued -> running -> completed/failed/canceled).
async fn job_stream(
    State(state): State<Arc<WebState>>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, Response> {
    let guard = try_acquire_sse_slot(&state)?;
    let rx = state.job_tx.subscribe();
    let stream = BroadcastStream::new(rx)
        .filter_map(|result| filter_lagged(result, "jobs"))
        .map(|msg| Ok(Event::default().event("job_update").data(msg)));

    let stream = stream.map(move |item| {
        let _keep = &guard;
        item
    });

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    ))
}

/// SSE endpoint that streams citation verification events from the autonomous
/// citation agent. Three event types flow through this channel:
///
/// - `citation_row_update` -- Per-row status transitions (pending -> searching
///   -> evaluating -> done/error) with verdict, confidence, and reasoning.
/// - `citation_reasoning_token` -- Individual tokens during LLM streaming for
///   live typewriter display in the frontend.
/// - `citation_job_progress` -- Aggregate progress after each row completes
///   (rows_done, rows_total, verdict distribution).
///
/// The event type is extracted from the JSON "event" field of each message
/// and used as the SSE event name for browser EventSource dispatch.
async fn citation_stream(
    State(state): State<Arc<WebState>>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, Response> {
    let guard = try_acquire_sse_slot(&state)?;
    let rx = state.citation_tx.subscribe();
    let stream = BroadcastStream::new(rx)
        .filter_map(|result| filter_lagged(result, "citation"))
        .map(|msg| {
            let event_type = serde_json::from_str::<serde_json::Value>(&msg)
                .ok()
                .and_then(|v| v["event"].as_str().map(String::from))
                .unwrap_or_else(|| "citation_row_update".to_string());
            Ok(Event::default().event(event_type).data(msg))
        });

    let stream = stream.map(move |item| {
        let _keep = &guard;
        item
    });

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    ))
}

/// SSE endpoint that streams source fetching events from the fetch_sources
/// handler. Each event represents one BibTeX entry being processed:
///
/// - `source_entry_update` -- Per-entry status update with cite_key, URL,
///   type (pdf/html), status (downloaded/fetched/failed/blocked), and metadata.
///
/// The event type is extracted from the JSON "event" field of each message.
async fn source_stream(
    State(state): State<Arc<WebState>>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, Response> {
    let guard = try_acquire_sse_slot(&state)?;
    let rx = state.source_tx.subscribe();
    let stream = BroadcastStream::new(rx)
        .filter_map(|result| filter_lagged(result, "sources"))
        .map(|msg| {
            let event_type = serde_json::from_str::<serde_json::Value>(&msg)
                .ok()
                .and_then(|v| v["event"].as_str().map(String::from))
                .unwrap_or_else(|| "source_entry_update".to_string());
            Ok(Event::default().event(event_type).data(msg))
        });

    let stream = stream.map(move |item| {
        let _keep = &guard;
        item
    });

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    ))
}

/// SSE endpoint that streams model operation events: download progress,
/// embedding model switch completion, reranker load completion, and Ollama
/// pull progress. The event type is extracted from the JSON "event" field
/// of each message and used as the SSE event name so the browser's
/// EventSource dispatches to the correct handler (model_switched,
/// model_downloaded, reranker_loaded, ollama_pull_progress, etc.).
/// Falls back to "model_event" if no event field is present in the payload.
async fn model_stream(
    State(state): State<Arc<WebState>>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, Response> {
    let guard = try_acquire_sse_slot(&state)?;
    let rx = state.model_tx.subscribe();
    let stream = BroadcastStream::new(rx)
        .filter_map(|result| filter_lagged(result, "models"))
        .map(|msg| {
            let event_type = serde_json::from_str::<serde_json::Value>(&msg)
                .ok()
                .and_then(|v| v["event"].as_str().map(String::from))
                .unwrap_or_else(|| "model_event".to_string());
            Ok(Event::default().event(event_type).data(msg))
        });

    let stream = stream.map(move |item| {
        let _keep = &guard;
        item
    });

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("ping"),
    ))
}
