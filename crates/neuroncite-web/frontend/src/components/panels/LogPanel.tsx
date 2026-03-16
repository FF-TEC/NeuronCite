import { Component, For, createEffect } from "solid-js";
import { state, actions } from "../../stores/app";
import type { LogEntry } from "../../stores/app";

/**
 * Log panel displaying real-time tracing log messages received via SSE.
 * Shows a monospace-font scrollable list of structured log entries with
 * color-coded level badges. Auto-scrolls to the bottom as messages arrive.
 * Provides a "Clear" button to empty the client-side log buffer.
 *
 * The log buffer is capped at 500 messages in the store to prevent
 * unbounded memory growth during long indexing sessions.
 */

/** Maps tracing log levels to CSS color values matching the neon tech theme. */
const LEVEL_COLORS: Record<string, string> = {
  ERROR: "var(--color-accent-magenta)",
  WARN: "var(--color-accent-amber)",
  INFO: "var(--color-accent-cyan)",
  DEBUG: "var(--color-text-muted)",
  TRACE: "var(--color-text-muted)",
};

/** Returns the CSS color for a given log level string. Falls back to the
 *  secondary text color when the level is not in the known set. */
function levelColor(level: string): string {
  return LEVEL_COLORS[level.toUpperCase()] || "var(--color-text-secondary)";
}

/** Formats a LogEntry timestamp (ISO 8601) into a short HH:MM:SS display
 *  string for the log line prefix. */
function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString("en-GB", { hour12: false });
  } catch {
    return "";
  }
}

const LogPanel: Component = () => {
  let scrollRef: HTMLDivElement | undefined;

  /** Auto-scroll to the bottom when new log messages arrive. */
  createEffect(() => {
    const _count = state.logMessages.length;
    if (scrollRef) {
      scrollRef.scrollTop = scrollRef.scrollHeight;
    }
  });

  return (
    <div>
      <div style={{ display: "flex", "justify-content": "space-between", "margin-bottom": "8px" }}>
        <span style={{ "font-size": "12px", color: "var(--color-text-muted)" }}>
          {state.logMessages.length} messages
        </span>
        <button class="btn btn-sm" onClick={() => actions.clearLogMessages()}>
          Clear
        </button>
      </div>
      <div
        ref={scrollRef}
        style={{
          height: "300px",
          "overflow-y": "auto",
          "font-family": "monospace",
          "font-size": "11px",
          "line-height": "1.5",
          background: "rgba(0, 0, 0, 0.3)",
          "border-radius": "6px",
          padding: "8px",
          color: "var(--color-text-secondary)",
        }}
      >
        <For each={state.logMessages}>
          {(entry: LogEntry) => (
            <div style={{ "white-space": "pre-wrap", "word-break": "break-all", padding: "1px 0" }}>
              <span style={{ color: "var(--color-text-muted)", "margin-right": "6px" }}>
                {formatTime(entry.timestamp)}
              </span>
              <span
                style={{
                  color: levelColor(entry.level),
                  "font-weight": entry.level === "ERROR" || entry.level === "WARN" ? "600" : "400",
                  "margin-right": "6px",
                  "min-width": "40px",
                  display: "inline-block",
                }}
              >
                {entry.level.padEnd(5)}
              </span>
              <span style={{ color: "var(--color-accent-purple)", "margin-right": "6px" }}>
                {entry.target}:
              </span>
              <span>{entry.message}</span>
            </div>
          )}
        </For>
      </div>
    </div>
  );
};

export default LogPanel;
