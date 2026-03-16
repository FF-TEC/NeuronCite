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

//! Custom tracing Layer that forwards formatted log events to all connected
//! SSE clients via the log_tx broadcast channel. Runs alongside the standard
//! fmt subscriber (which prints to stdout) so both console and browser Log
//! panel receive the same events.
//!
//! The layer captures ALL fields from each tracing event -- both the primary
//! "message" field and any structured key-value pairs (e.g. session_id, path,
//! job_id). Structured fields are appended to the message in the same
//! `key=value` format that the console fmt layer uses, so the browser Log
//! panel displays the exact same information as the terminal.

use tokio::sync::broadcast;
use tracing::Subscriber;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

/// Tracing Layer that broadcasts formatted log events to SSE consumers.
/// Each event is serialized as a JSON string and sent through the
/// `tokio::sync::broadcast::Sender<String>`. Lagged receivers (slow
/// browser SSE clients) are handled by the broadcast channel's built-in
/// lagging mechanism -- the sender never blocks.
pub struct BroadcastLayer {
    /// Broadcast sender for distributing log events to SSE subscribers.
    tx: broadcast::Sender<String>,
}

impl BroadcastLayer {
    /// Creates a BroadcastLayer that publishes formatted tracing events
    /// to the given broadcast sender. The sender is typically the same
    /// `log_tx` from `WebState` that the SSE `/events/logs` endpoint
    /// subscribes to.
    #[must_use]
    pub fn new(tx: broadcast::Sender<String>) -> Self {
        Self { tx }
    }
}

impl<S: Subscriber> Layer<S> for BroadcastLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        // Forward only INFO, WARN, and ERROR events to SSE clients. DEBUG and
        // TRACE events are too verbose for the browser Log panel: a typical
        // indexing run produces thousands of TRACE calls per second from the
        // HNSW and tokenizer internals, which would saturate the broadcast
        // channel and force the browser to process and render high-frequency
        // updates unnecessarily.
        //
        // Tracing's Level ordering is by verbosity: TRACE (most verbose) is
        // the "largest" PartialOrd value and ERROR (least verbose) is the
        // "smallest". Concretely: TRACE > DEBUG > INFO > WARN > ERROR.
        // Events with higher verbosity than INFO (DEBUG, TRACE) satisfy
        // `level > &Level::INFO` and are dropped here.
        if event.metadata().level() > &tracing::Level::INFO {
            return;
        }

        // Extract ALL fields from the event: the primary "message" field
        // and any additional structured key-value pairs.
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);

        let level = event.metadata().level().as_str();
        let target = event.metadata().target();

        // Append structured fields to the message in the same `key=value`
        // format that the tracing fmt layer uses for console output. This
        // ensures the browser Log panel shows the same information as the
        // terminal (e.g. "Indexing started session_id=5 files=10").
        let mut full_message = visitor.message;
        for (key, value) in &visitor.fields {
            full_message.push(' ');
            full_message.push_str(key);
            full_message.push('=');
            full_message.push_str(value);
        }

        // Format as JSON for structured consumption by the frontend Log panel.
        let json = serde_json::json!({
            "level": level,
            "target": target,
            "message": full_message,
        });

        // send() returns Err only when there are zero active receivers,
        // which is expected when no browser has the Log panel open.
        let _ = self.tx.send(json.to_string());
    }
}

/// Field visitor that extracts ALL fields from a tracing event. The "message"
/// field (set by `tracing::info!("...")` macros) is stored separately; all
/// other key-value pairs are collected into the `fields` vec for appending
/// to the output string. This matches the behavior of the tracing fmt layer
/// which displays all fields after the message text.
#[derive(Default)]
pub(crate) struct FieldVisitor {
    /// The primary human-readable message from the tracing event.
    pub(crate) message: String,
    /// Additional structured key-value pairs from the tracing event.
    /// Each tuple contains (field_name, formatted_value).
    pub(crate) fields: Vec<(String, String)>,
}

impl tracing::field::Visit for FieldVisitor {
    /// Handles Debug-formatted fields. For the "message" field, this captures
    /// the formatted text (fmt::Arguments Debug delegates to Display, so no
    /// extra quotes are added). For other fields, the Debug representation
    /// is stored (strings get quotes, numbers are plain -- matching the
    /// console fmt layer output).
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{:?}", value);
        } else {
            self.fields
                .push((field.name().to_string(), format!("{:?}", value)));
        }
    }

    /// Handles plain string fields. For the "message" field, stores the raw
    /// string. For other fields, stores the value without quotes (matching
    /// how the fmt layer displays &str-typed fields).
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            self.fields
                .push((field.name().to_string(), value.to_string()));
        }
    }

    /// Handles i64 fields (i8, i16, i32, i64 all dispatch here).
    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }

    /// Handles u64 fields (u8, u16, u32, u64 all dispatch here).
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }

    /// Handles f64 fields (f32 and f64 both dispatch here).
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }

    /// Handles boolean fields.
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.fields
            .push((field.name().to_string(), value.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast;
    use tracing_subscriber::layer::SubscriberExt;

    /// Verifies that the BroadcastLayer sends JSON with level, target, and
    /// message fields when a tracing event is emitted. The receiver should
    /// get a valid JSON string that the frontend can parse.
    #[tokio::test]
    async fn broadcast_layer_sends_json_to_channel() {
        let (tx, mut rx) = broadcast::channel::<String>(16);
        let layer = BroadcastLayer::new(tx);

        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        tracing::info!(target: "test_target", "hello from test");

        let received = rx.try_recv().expect("should receive a log message");
        let parsed: serde_json::Value =
            serde_json::from_str(&received).expect("should be valid JSON");

        assert_eq!(parsed["level"], "INFO");
        assert_eq!(parsed["target"], "test_target");
        // The message field contains the formatted text from the info! macro.
        let msg = parsed["message"]
            .as_str()
            .expect("message should be a string");
        assert!(
            msg.contains("hello from test"),
            "message should contain the log text, got: {msg}"
        );
    }

    /// Verifies that the JSON output contains exactly three fields:
    /// level, target, and message. No extra fields should be present.
    #[tokio::test]
    async fn broadcast_layer_json_has_three_fields() {
        let (tx, mut rx) = broadcast::channel::<String>(16);
        let layer = BroadcastLayer::new(tx);

        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        tracing::warn!("disk space low");

        let received = rx.try_recv().expect("should receive a log message");
        let parsed: serde_json::Value = serde_json::from_str(&received).unwrap();
        let obj = parsed.as_object().expect("should be a JSON object");

        assert_eq!(obj.len(), 3, "JSON should have exactly 3 fields");
        assert!(obj.contains_key("level"), "should have 'level' field");
        assert!(obj.contains_key("target"), "should have 'target' field");
        assert!(obj.contains_key("message"), "should have 'message' field");
    }

    /// Verifies that only INFO, WARN, and ERROR events are forwarded to SSE
    /// clients. DEBUG and TRACE events are filtered out by the level guard in
    /// `on_event` to avoid saturating the broadcast channel and the browser
    /// Log panel with verbose internal tracing calls.
    #[tokio::test]
    async fn broadcast_layer_forwards_info_and_above_only() {
        let (tx, mut rx) = broadcast::channel::<String>(16);
        let layer = BroadcastLayer::new(tx);

        // Enable TRACE at the subscriber level so all events reach on_event.
        // The BroadcastLayer's own level guard determines what gets forwarded.
        let filter = tracing_subscriber::filter::LevelFilter::TRACE;
        let subscriber = tracing_subscriber::registry().with(filter).with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        tracing::error!("error msg");
        tracing::warn!("warn msg");
        tracing::info!("info msg");
        tracing::debug!("debug msg"); // filtered out by BroadcastLayer
        tracing::trace!("trace msg"); // filtered out by BroadcastLayer

        // Only 3 messages should arrive: ERROR, WARN, INFO.
        let levels: Vec<String> = (0..3)
            .map(|_| {
                let received = rx.try_recv().expect("should receive INFO/WARN/ERROR");
                let parsed: serde_json::Value = serde_json::from_str(&received).unwrap();
                parsed["level"].as_str().unwrap().to_string()
            })
            .collect();

        assert_eq!(
            levels,
            vec!["ERROR", "WARN", "INFO"],
            "only INFO and above must be forwarded to SSE"
        );

        // DEBUG and TRACE must not appear in the channel.
        assert!(
            rx.try_recv().is_err(),
            "DEBUG and TRACE must not be forwarded to SSE subscribers"
        );
    }

    /// Verifies that send() silently succeeds when no receivers are connected
    /// (the broadcast channel drops messages when there are zero subscribers).
    #[tokio::test]
    async fn broadcast_layer_no_receivers_does_not_panic() {
        let (tx, _) = broadcast::channel::<String>(16);
        // Drop the receiver immediately -- the layer should handle this gracefully
        let layer = BroadcastLayer::new(tx);

        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        // This should not panic despite having no receivers
        tracing::info!("message with no listeners");
    }

    /// Verifies that formatted messages (with interpolation) are captured correctly.
    #[tokio::test]
    async fn broadcast_layer_captures_formatted_messages() {
        let (tx, mut rx) = broadcast::channel::<String>(16);
        let layer = BroadcastLayer::new(tx);

        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let port = 3030;
        tracing::info!("Server started on port {}", port);

        let received = rx.try_recv().expect("should receive");
        let parsed: serde_json::Value = serde_json::from_str(&received).unwrap();
        let msg = parsed["message"].as_str().unwrap();
        assert!(
            msg.contains("Server started on port 3030"),
            "formatted message should contain interpolated value, got: {msg}"
        );
    }

    /// Verifies that structured key-value fields from tracing events are
    /// appended to the message string in `key=value` format, matching the
    /// console fmt layer output. Without this, the GUI Log panel would
    /// silently drop fields like session_id, job_id, path, etc.
    #[tokio::test]
    async fn broadcast_layer_captures_structured_fields() {
        let (tx, mut rx) = broadcast::channel::<String>(16);
        let layer = BroadcastLayer::new(tx);

        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        tracing::info!(session_id = 42, files = 10, "Indexing started");

        let received = rx.try_recv().expect("should receive");
        let parsed: serde_json::Value = serde_json::from_str(&received).unwrap();
        let msg = parsed["message"].as_str().unwrap();

        assert!(
            msg.contains("Indexing started"),
            "message should contain the text, got: {msg}"
        );
        assert!(
            msg.contains("session_id=42"),
            "message should contain session_id field, got: {msg}"
        );
        assert!(
            msg.contains("files=10"),
            "message should contain files field, got: {msg}"
        );
    }

    /// Verifies that Display-formatted fields (using the % sigil in tracing
    /// macros) are captured without extra Debug quoting.
    #[tokio::test]
    async fn broadcast_layer_captures_display_formatted_fields() {
        let (tx, mut rx) = broadcast::channel::<String>(16);
        let layer = BroadcastLayer::new(tx);

        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let job_id = "abc-123";
        tracing::info!(job_id = %job_id, "Job completed");

        let received = rx.try_recv().expect("should receive");
        let parsed: serde_json::Value = serde_json::from_str(&received).unwrap();
        let msg = parsed["message"].as_str().unwrap();

        assert!(
            msg.contains("Job completed"),
            "message should contain the text, got: {msg}"
        );
        assert!(
            msg.contains("job_id=abc-123"),
            "message should contain the Display-formatted job_id field, got: {msg}"
        );
    }

    /// Verifies that boolean fields are captured correctly.
    #[tokio::test]
    async fn broadcast_layer_captures_bool_fields() {
        let (tx, mut rx) = broadcast::channel::<String>(16);
        let layer = BroadcastLayer::new(tx);

        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        tracing::info!(gpu_available = true, "System check");

        let received = rx.try_recv().expect("should receive");
        let parsed: serde_json::Value = serde_json::from_str(&received).unwrap();
        let msg = parsed["message"].as_str().unwrap();

        assert!(
            msg.contains("gpu_available=true"),
            "message should contain boolean field, got: {msg}"
        );
    }

    /// Verifies that a message-only event (no extra fields) produces a clean
    /// message string without trailing whitespace or empty field separators.
    #[tokio::test]
    async fn broadcast_layer_message_only_no_trailing_fields() {
        let (tx, mut rx) = broadcast::channel::<String>(16);
        let layer = BroadcastLayer::new(tx);

        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        tracing::info!("Simple message");

        let received = rx.try_recv().expect("should receive");
        let parsed: serde_json::Value = serde_json::from_str(&received).unwrap();
        let msg = parsed["message"].as_str().unwrap();

        assert_eq!(
            msg, "Simple message",
            "message-only event should have no trailing fields"
        );
    }
}
