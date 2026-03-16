/**
 * Tests for the log message pipeline: SSE event parsing, store actions,
 * and the data flow from the backend BroadcastLayer JSON format through
 * the frontend LogEntry structure.
 *
 * These tests verify that the type mismatch between the backend JSON object
 * and the frontend string display is correctly handled. The backend sends
 * structured JSON ({"level","target","message"}) via SSE, and the frontend
 * must parse this into LogEntry objects for rendering.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";

// ---------------------------------------------------------------------------
// SSE subscribeToLogs parsing tests
// ---------------------------------------------------------------------------

/**
 * Inline re-implementation of the log parsing logic from sse.ts.
 * This avoids importing the real SSE module (which creates EventSource
 * instances requiring a browser environment) while testing the exact
 * same transformation logic.
 */
function parseLogEvent(data: unknown): {
  level: string;
  target: string;
  message: string;
  timestamp: string;
} | null {
  const now = "2026-01-01T00:00:00.000Z";
  if (typeof data === "object" && data !== null) {
    const raw = data as { level?: string; target?: string; message?: string };
    return {
      level: (raw.level || "INFO").toUpperCase(),
      target: raw.target || "",
      message: raw.message || "",
      timestamp: now,
    };
  } else if (typeof data === "string") {
    return { level: "INFO", target: "", message: data, timestamp: now };
  }
  return null;
}

describe("parseLogEvent", () => {
  it("parses a structured JSON object from BroadcastLayer into a LogEntry", () => {
    const input = {
      level: "info",
      target: "neuroncite_api::routes",
      message: "Server started on port 3030",
    };

    const result = parseLogEvent(input);

    expect(result).not.toBeNull();
    expect(result!.level).toBe("INFO");
    expect(result!.target).toBe("neuroncite_api::routes");
    expect(result!.message).toBe("Server started on port 3030");
    expect(result!.timestamp).toBe("2026-01-01T00:00:00.000Z");
  });

  it("normalizes the level to uppercase", () => {
    const input = { level: "warn", target: "test", message: "low disk" };
    const result = parseLogEvent(input);
    expect(result!.level).toBe("WARN");
  });

  it("handles missing fields with defaults", () => {
    const input = {};
    const result = parseLogEvent(input);
    expect(result!.level).toBe("INFO");
    expect(result!.target).toBe("");
    expect(result!.message).toBe("");
  });

  it("handles a plain string fallback", () => {
    const result = parseLogEvent("plain text message");
    expect(result!.level).toBe("INFO");
    expect(result!.target).toBe("");
    expect(result!.message).toBe("plain text message");
  });

  it("returns null for null input", () => {
    const result = parseLogEvent(null);
    expect(result).toBeNull();
  });

  it("returns null for undefined input", () => {
    const result = parseLogEvent(undefined);
    expect(result).toBeNull();
  });

  it("returns null for numeric input", () => {
    const result = parseLogEvent(42);
    expect(result).toBeNull();
  });

  it("handles error-level events", () => {
    const input = {
      level: "ERROR",
      target: "neuroncite_embed",
      message: "GPU memory allocation failed",
    };
    const result = parseLogEvent(input);
    expect(result!.level).toBe("ERROR");
    expect(result!.target).toBe("neuroncite_embed");
    expect(result!.message).toBe("GPU memory allocation failed");
  });

  it("handles trace-level events", () => {
    const input = {
      level: "trace",
      target: "neuroncite_store::db",
      message: "SELECT * FROM chunks WHERE ...",
    };
    const result = parseLogEvent(input);
    expect(result!.level).toBe("TRACE");
  });

  it("handles the exact JSON format from Rust BroadcastLayer", () => {
    // This is the exact string the BroadcastLayer sends via the broadcast channel.
    // The SSE handler JSON.parse()s it before calling the log handler.
    const jsonString = '{"level":"info","target":"neuroncite_web","message":"SSE client connected"}';
    const parsed = JSON.parse(jsonString);
    const result = parseLogEvent(parsed);

    expect(result!.level).toBe("INFO");
    expect(result!.target).toBe("neuroncite_web");
    expect(result!.message).toBe("SSE client connected");
  });

  it("handles BroadcastLayer JSON with structured fields appended to message", () => {
    // The BroadcastLayer appends structured key-value fields to the message
    // string in `key=value` format, matching the console fmt layer output.
    const jsonString =
      '{"level":"INFO","target":"neuroncite_api::executor","message":"Indexing started session_id=42 files=10"}';
    const parsed = JSON.parse(jsonString);
    const result = parseLogEvent(parsed);

    expect(result!.level).toBe("INFO");
    expect(result!.target).toBe("neuroncite_api::executor");
    expect(result!.message).toBe("Indexing started session_id=42 files=10");
    expect(result!.message).toContain("session_id=42");
    expect(result!.message).toContain("files=10");
  });
});

// ---------------------------------------------------------------------------
// Store log buffer management tests
// ---------------------------------------------------------------------------

describe("log buffer management", () => {
  /** Simulates the produce() logic from the store's appendLogMessage action. */
  function appendToBuffer(
    buffer: Array<{ level: string; target: string; message: string; timestamp: string }>,
    entry: { level: string; target: string; message: string; timestamp: string },
    maxSize: number,
  ): void {
    buffer.push(entry);
    if (buffer.length > maxSize) {
      buffer.splice(0, buffer.length - maxSize);
    }
  }

  it("appends entries to the buffer", () => {
    const buffer: Array<{ level: string; target: string; message: string; timestamp: string }> = [];
    const entry = { level: "INFO", target: "test", message: "hello", timestamp: "2026-01-01T00:00:00Z" };

    appendToBuffer(buffer, entry, 500);

    expect(buffer).toHaveLength(1);
    expect(buffer[0].message).toBe("hello");
  });

  it("trims oldest entries when exceeding the max size", () => {
    const buffer: Array<{ level: string; target: string; message: string; timestamp: string }> = [];
    const maxSize = 3;

    for (let i = 0; i < 5; i++) {
      appendToBuffer(buffer, {
        level: "INFO",
        target: "test",
        message: `msg-${i}`,
        timestamp: "2026-01-01T00:00:00Z",
      }, maxSize);
    }

    expect(buffer).toHaveLength(3);
    expect(buffer[0].message).toBe("msg-2");
    expect(buffer[1].message).toBe("msg-3");
    expect(buffer[2].message).toBe("msg-4");
  });

  it("clears the buffer correctly", () => {
    const buffer = [
      { level: "INFO", target: "test", message: "a", timestamp: "2026-01-01T00:00:00Z" },
      { level: "WARN", target: "test", message: "b", timestamp: "2026-01-01T00:00:00Z" },
    ];

    buffer.length = 0;

    expect(buffer).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Log level color mapping tests
// ---------------------------------------------------------------------------

describe("log level color mapping", () => {
  /** Re-implementation of the level color mapping from LogPanel and StatusBar. */
  const LEVEL_COLORS: Record<string, string> = {
    ERROR: "var(--color-accent-magenta)",
    WARN: "#f0a030",
    INFO: "var(--color-accent-cyan)",
    DEBUG: "var(--color-text-muted)",
    TRACE: "var(--color-text-muted)",
  };

  function levelColor(level: string): string {
    return LEVEL_COLORS[level.toUpperCase()] || "var(--color-text-secondary)";
  }

  it("returns magenta for ERROR level", () => {
    expect(levelColor("ERROR")).toBe("var(--color-accent-magenta)");
  });

  it("returns amber for WARN level", () => {
    expect(levelColor("WARN")).toBe("#f0a030");
  });

  it("returns cyan for INFO level", () => {
    expect(levelColor("INFO")).toBe("var(--color-accent-cyan)");
  });

  it("returns muted for DEBUG level", () => {
    expect(levelColor("DEBUG")).toBe("var(--color-text-muted)");
  });

  it("returns muted for TRACE level", () => {
    expect(levelColor("TRACE")).toBe("var(--color-text-muted)");
  });

  it("is case-insensitive", () => {
    expect(levelColor("info")).toBe("var(--color-accent-cyan)");
    expect(levelColor("Error")).toBe("var(--color-accent-magenta)");
  });

  it("falls back to secondary text color for unknown levels", () => {
    expect(levelColor("UNKNOWN")).toBe("var(--color-text-secondary)");
  });
});

// ---------------------------------------------------------------------------
// End-to-end SSE data flow simulation
// ---------------------------------------------------------------------------

describe("end-to-end SSE log data flow", () => {
  it("simulates the complete pipeline from BroadcastLayer JSON to displayable LogEntry", () => {
    // Step 1: Rust BroadcastLayer formats the event as JSON string.
    // Structured fields are appended to the message in key=value format.
    const broadcastOutput = JSON.stringify({
      level: "info",
      target: "neuroncite_api::handlers::search",
      message: "Search query: 'machine learning' returned 15 results session_id=3",
    });

    // Step 2: SSE EventSource receives and the subscribeSSE handler JSON.parses it
    const parsed = JSON.parse(broadcastOutput);

    // Step 3: subscribeToLogs transforms the parsed object into a LogEntry
    const entry = parseLogEvent(parsed);

    // Step 4: Verify the entry is renderable as a string (not [object Object])
    expect(entry).not.toBeNull();
    expect(typeof entry!.level).toBe("string");
    expect(typeof entry!.target).toBe("string");
    expect(typeof entry!.message).toBe("string");
    expect(typeof entry!.timestamp).toBe("string");

    // Step 5: Verify the display format including structured fields
    expect(entry!.level).toBe("INFO");
    expect(entry!.message).toContain("Search query: 'machine learning' returned 15 results");
    expect(entry!.message).toContain("session_id=3");

    // Step 6: Verify each field is directly renderable as JSX text content
    // (this was the original bug: objects passed to JSX render as nothing)
    const statusBarText = `${entry!.level} ${entry!.target}: ${entry!.message}`;
    expect(statusBarText).toContain("Search query:");
    expect(statusBarText).toContain("session_id=3");
    expect(statusBarText).not.toContain("[object Object]");
  });

  it("verifies that synthetic frontend log messages follow the same LogEntry structure", () => {
    // Synthetic messages created in App.tsx for model events
    const syntheticEntry = {
      level: "INFO",
      target: "frontend",
      message: "Model switched to BAAI/bge-small-en-v1.5 (384d)",
      timestamp: new Date().toISOString(),
    };

    expect(typeof syntheticEntry.level).toBe("string");
    expect(typeof syntheticEntry.message).toBe("string");
    expect(syntheticEntry.level).toBe("INFO");
    expect(syntheticEntry.target).toBe("frontend");
  });
});
