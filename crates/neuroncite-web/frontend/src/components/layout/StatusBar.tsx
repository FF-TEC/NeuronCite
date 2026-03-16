import { Component, Show, For, createSignal, createEffect } from "solid-js";
import { state, actions } from "../../stores/app";
import type { LogEntry } from "../../stores/app";

/** Maps tracing log levels to CSS color values for the log popup display.
 *  Matches the color scheme in LogPanel for visual consistency. */
const LOG_LEVEL_COLORS: Record<string, string> = {
  ERROR: "var(--color-accent-magenta)",
  WARN: "#f0a030",
  INFO: "var(--color-accent-cyan)",
  DEBUG: "var(--color-text-muted)",
  TRACE: "var(--color-text-muted)",
};

/** Returns the CSS color for a given log level string. */
function logLevelColor(level: string): string {
  return LOG_LEVEL_COLORS[level.toUpperCase()] || "var(--color-text-secondary)";
}

/**
 * Bottom status bar displaying the active embedding model, optional reranker
 * model, GPU compute mode, progress indicators, and the last log message.
 *
 * The left side shows model information (embedding model name + dimension,
 * reranker model name when loaded) and the compute device (GPU name or CPU
 * fallback). The right side shows the last log line received via SSE.
 *
 * Clicking the log line toggles a popup log viewer panel that slides up
 * from above the status bar, containing the full scrollable log buffer
 * with copy support.
 *
 * The log popup uses role="dialog" and aria-modal="true" so screen readers
 * announce it as a dialog. Pressing Escape dismisses the popup.
 *
 * Two independent progress systems are rendered:
 *
 * 1. **indexProgress** (SSE-driven, phase-aware): shown during PDF indexing
 *    with phase-specific labels (extracting, embedding, building HNSW index).
 *
 * 2. **taskProgress** (generic, any tab): shown for any long-running operation
 *    like source fetching or citation export. Any component can set this via
 *    actions.setTaskProgress({ label, done, total }).
 *
 * The progress bar overlay shows indexProgress when active (takes priority),
 * otherwise shows taskProgress. Both labels can appear simultaneously.
 */
const StatusBar: Component = () => {
  // ---- Log popup state ----

  /** Whether the log popup panel above the status bar is visible. */
  const [logPopupOpen, setLogPopupOpen] = createSignal(false);

  /** Reference to the scrollable container inside the log popup,
   *  used for auto-scrolling to the bottom when messages arrive. */
  let popupScrollRef: HTMLDivElement | undefined;

  /** Reference to the log popup container for programmatic focus
   *  management. Focus is placed here when the popup opens. */
  let popupRef: HTMLDivElement | undefined;

  /** Returns the last log entry from the store buffer, or null when no log
   *  messages have been received yet. */
  const lastLogEntry = (): LogEntry | null => {
    const msgs = state.logMessages;
    if (msgs.length === 0) return null;
    return msgs[msgs.length - 1];
  };

  /** Formats the last log entry as a compact single-line string for the
   *  status bar display and tooltip. */
  const lastLogLine = (): string => {
    const entry = lastLogEntry();
    if (!entry) return "";
    return `${entry.level} ${entry.target}: ${entry.message}`;
  };

  /** Auto-scrolls the popup log viewer to the bottom when new messages
   *  arrive and the popup is currently open. */
  createEffect(() => {
    const _count = state.logMessages.length;
    if (popupScrollRef && logPopupOpen()) {
      popupScrollRef.scrollTop = popupScrollRef.scrollHeight;
    }
  });

  /** Handles keydown events on the log popup. Escape key dismisses
   *  the popup, matching the expected dialog dismissal behavior. */
  const onPopupKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      setLogPopupOpen(false);
    }
  };

  // ---- Embedding model display ----

  /** Formats the embedding model name by extracting the short name from the
   *  HuggingFace model identifier (e.g., "BAAI/bge-small-en-v1.5" becomes
   *  "bge-small-en-v1.5") and appending the vector dimension. */
  const embeddingDisplay = () => {
    if (!state.activeModelId) return "No model loaded";
    const shortName = state.activeModelId.split("/").pop();
    return `${shortName} (${state.activeModelDimension}d)`;
  };

  // ---- Reranker display (conditional) ----

  /** Extracts the short reranker model name from the full HuggingFace
   *  identifier. Only called when a reranker is loaded. */
  const rerankerDisplay = () => {
    if (!state.rerankerModelId) return "";
    return state.rerankerModelId.split("/").pop() || state.rerankerModelId;
  };

  // ---- Compute mode display ----

  /** Formats the compute mode label. Shows the GPU device name when a GPU
   *  is detected (e.g., "NVIDIA GeForce RTX 4090"), otherwise "CPU Fallback". */
  const computeDisplay = () => {
    if (state.gpuDeviceName && state.gpuDeviceName !== "CPU") {
      return state.gpuDeviceName;
    }
    if (state.health?.gpu_available) {
      return state.health.active_backend.toUpperCase();
    }
    return "CPU Fallback";
  };

  // ---- Indexing progress (SSE-driven, phase-aware) ----

  /** Weight of the extraction phase as a fraction of the total progress bar.
   *  Extraction occupies 0-30%, embedding occupies 30-100%. This weighting
   *  reflects that embedding is the dominant cost (GPU-bound, sequential). */
  const EXTRACT_WEIGHT = 0.3;

  /** Computes a unified indexing progress percentage (0-100) across all phases.
   *  The progress bar is split into weighted segments:
   *    - extracting:     0% to 30%  (proportional to files_done / files_total)
   *    - embedding:     30% to 100% (proportional to files_done / files_total)
   *    - building_index: 100%       (file processing is complete)
   *  This prevents the confusing behavior of the bar filling twice to 100%. */
  const indexPercent = () => {
    const p = state.indexProgress;
    if (!p || p.files_total === 0) return 0;
    if (p.phase === "building_index") return 100;

    const phaseRatio = p.files_done / p.files_total;

    if (p.phase === "extracting") {
      // 0% to 30%: extraction phase
      return Math.round(phaseRatio * EXTRACT_WEIGHT * 100);
    }
    // 30% to 100%: embedding phase
    const EMBED_WEIGHT = 1 - EXTRACT_WEIGHT;
    return Math.round((EXTRACT_WEIGHT + phaseRatio * EMBED_WEIGHT) * 100);
  };

  /** Generates the phase-specific progress text for indexing operations.
   *  Each phase shows its own file counter alongside the unified percentage. */
  const indexLabel = () => {
    const p = state.indexProgress;
    if (!p) return "";

    switch (p.phase) {
      case "extracting":
        return `Extracting: ${p.files_done}/${p.files_total} files (${indexPercent()}%)`;
      case "embedding":
        return `Embedding: ${p.files_done}/${p.files_total} files, ${p.chunks_created} chunks (${indexPercent()}%)`;
      case "building_index":
        return `Building HNSW index... (${p.chunks_created} chunks)`;
      default:
        return `Indexing: ${p.files_done}/${p.files_total} files (${indexPercent()}%)`;
    }
  };

  /** Whether the indexing progress bar is active and should be displayed. */
  const indexActive = () => state.indexProgress && !state.indexProgress.complete;

  // ---- Generic task progress (any tab) ----

  /** Computes the task progress percentage (0-100). */
  const taskPercent = () => {
    const t = state.taskProgress;
    if (!t || t.total === 0) return 0;
    return Math.round((t.done / t.total) * 100);
  };

  /** Generates the label string for the generic task progress. */
  const taskLabel = () => {
    const t = state.taskProgress;
    if (!t) return "";
    return `${t.label}: ${t.done}/${t.total} (${taskPercent()}%)`;
  };

  /** The bar overlay percentage: indexing takes priority over task progress. */
  const barPercent = () => {
    if (indexActive()) return indexPercent();
    if (state.taskProgress) return taskPercent();
    return 0;
  };

  /** Whether any progress bar should be visible. */
  const barVisible = () => indexActive() || state.taskProgress !== null;

  return (
    <footer class="statusbar">
      {/* Full-width progress bar overlay. Indexing progress takes priority
       *  over generic task progress when both are active simultaneously. */}
      <Show when={barVisible()}>
        <div
          class="statusbar-progress"
          style={{ width: `${barPercent()}%` }}
        />
      </Show>

      {/* LEFT: Embedding model indicator with cyan status dot */}
      <div class="statusbar-item">
        <div
          class="statusbar-dot"
          style={{
            background: state.activeModelId
              ? "var(--color-accent-cyan)"
              : "var(--color-text-muted)",
          }}
        />
        <span style={{ color: "var(--color-text-secondary)" }}>Embed:</span>
        <span>{embeddingDisplay()}</span>
      </div>

      {/* Reranker model indicator with purple status dot. Only rendered
       *  when a cross-encoder reranker is loaded in the GPU worker. */}
      <Show when={state.rerankerAvailable && state.rerankerModelId}>
        <div class="statusbar-separator" />
        <div class="statusbar-item">
          <div
            class="statusbar-dot"
            style={{ background: "var(--color-accent-purple)" }}
          />
          <span style={{ color: "var(--color-text-secondary)" }}>Rerank:</span>
          <span>{rerankerDisplay()}</span>
        </div>
      </Show>

      <div class="statusbar-separator" />

      {/* Compute mode indicator showing GPU device name or CPU fallback */}
      <div class="statusbar-item">
        <span style={{ color: "var(--color-text-secondary)" }}>Compute:</span>
        <span>{computeDisplay()}</span>
      </div>

      {/* Phase-aware indexing progress label (extraction, embedding, HNSW build) */}
      <Show when={indexActive()}>
        <div class="statusbar-separator" />
        <div class="statusbar-item">
          <span style={{ color: "var(--color-accent-cyan)" }}>
            {indexLabel()}
          </span>
        </div>
      </Show>

      {/* Generic task progress label (source fetching, citation export, etc.) */}
      <Show when={state.taskProgress}>
        <div class="statusbar-separator" />
        <div class="statusbar-item">
          <span style={{ color: "var(--color-accent-purple)" }}>
            {taskLabel()}
          </span>
        </div>
      </Show>

      {/* Spacer pushes the last log line to the right edge of the status bar */}
      <div style={{ flex: "1" }} />

      {/* Last log message, truncated to fit the status bar. Clicking toggles
       *  the popup log viewer panel above the status bar. Shows just the
       *  message text for compactness; full detail is in the popup. */}
      <Show when={lastLogEntry()}>
        <div
          class="statusbar-log-line"
          onClick={() => setLogPopupOpen(!logPopupOpen())}
          title={lastLogLine()}
        >
          <span class="statusbar-log-text">{lastLogEntry()!.message}</span>
        </div>
      </Show>

      {/* Log popup: full-width panel anchored above the status bar with
       *  scrollable log messages. Backdrop click or close button dismisses it.
       *  Uses role="dialog" and aria-modal="true" for screen reader semantics.
       *  Pressing Escape dismisses the popup via the onPopupKeyDown handler. */}
      <Show when={logPopupOpen()}>
        <div class="log-popup-backdrop" onClick={() => setLogPopupOpen(false)} />
        <div
          class="log-popup"
          ref={(el) => {
            popupRef = el;
            // Place focus on the popup container after render so Escape
            // key handling works immediately without requiring a click.
            requestAnimationFrame(() => popupRef?.focus());
          }}
          role="dialog"
          aria-modal="true"
          aria-label="Log Viewer"
          tabIndex={-1}
          onKeyDown={onPopupKeyDown}
        >
          <div class="log-popup-header">
            <span class="log-popup-title">Log Viewer</span>
            <span style={{ "font-size": "11px", color: "var(--color-text-muted)" }}>
              {state.logMessages.length} messages
            </span>
            <div style={{ flex: "1" }} />
            <button class="btn btn-sm" onClick={() => actions.clearLogMessages()}>
              Clear
            </button>
            <button class="btn btn-sm" onClick={() => setLogPopupOpen(false)}>
              Close
            </button>
          </div>
          <div class="log-popup-body" ref={popupScrollRef}>
            <For each={state.logMessages}>
              {(entry: LogEntry) => (
                <div class="log-popup-line">
                  <span style={{ color: logLevelColor(entry.level), "margin-right": "6px" }}>
                    {entry.level.padEnd(5)}
                  </span>
                  <span style={{ color: "var(--color-accent-purple)", "margin-right": "6px" }}>
                    {entry.target}
                  </span>
                  <span>{entry.message}</span>
                </div>
              )}
            </For>
          </div>
        </div>
      </Show>
    </footer>
  );
};

export default StatusBar;
