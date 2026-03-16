import { Component, Show, createSignal } from "solid-js";
import { state, actions } from "../../stores/app";
import { api } from "../../api/client";

/**
 * Maintenance panel providing database operations (FTS5 optimization, HNSW
 * index rebuild) and a localStorage reset button. The database operations
 * require an active session. The reset button is always available because
 * it operates on browser-local state independent of any session.
 *
 * Each operation uses a button that shows a spinner while in progress and
 * displays a success/error status on completion. The reset button uses a
 * two-stage confirmation to prevent accidental data loss.
 */
const MaintenancePanel: Component = () => {
  const [optimizing, setOptimizing] = createSignal(false);
  const [rebuilding, setRebuilding] = createSignal(false);
  const [optimizeStatus, setOptimizeStatus] = createSignal("");
  const [rebuildStatus, setRebuildStatus] = createSignal("");

  /** Two-stage confirmation state for the reset action. When false, the
   *  button reads "Reset to Defaults". When true, it switches to
   *  "Confirm Reset" / "Cancel" to prevent accidental resets. */
  const [confirmReset, setConfirmReset] = createSignal(false);
  const [resetStatus, setResetStatus] = createSignal("");

  /** Triggers FTS5 optimization for the active session. */
  const runOptimize = async () => {
    if (state.activeSessionId === null) return;
    setOptimizing(true);
    setOptimizeStatus("");
    try {
      const res = await api.optimizeSession(state.activeSessionId);
      setOptimizeStatus(res.status);
    } catch (e) {
      setOptimizeStatus(`Error: ${e}`);
    } finally {
      setOptimizing(false);
    }
  };

  /** Triggers HNSW index rebuild for the active session. */
  const runRebuild = async () => {
    if (state.activeSessionId === null) return;
    setRebuilding(true);
    setRebuildStatus("");
    try {
      const res = await api.rebuildIndex(state.activeSessionId);
      setRebuildStatus(`Rebuild job created: ${res.job_id}`);
    } catch (e) {
      setRebuildStatus(`Error: ${e}`);
    } finally {
      setRebuilding(false);
    }
  };

  /** Executes the localStorage reset after the user confirms. Clears all
   *  persisted keys and resets corresponding store fields to defaults. */
  const executeReset = () => {
    actions.resetToDefaults();
    setConfirmReset(false);
    setResetStatus("All settings reset to defaults.");
  };

  const noSession = () => state.activeSessionId === null;

  return (
    <div style={{ display: "flex", "flex-direction": "column", gap: "16px" }}>
      <Show
        when={!noSession()}
        fallback={
          <div style={{ color: "var(--color-text-muted)", "font-size": "13px" }}>
            Select a session to perform maintenance operations.
          </div>
        }
      >
        {/* FTS5 Optimization */}
        <div class="glass-card" style={{ padding: "12px 16px" }}>
          <div style={{ "font-weight": "600", "font-size": "13px", "margin-bottom": "8px" }}>
            Optimize FTS5
          </div>
          <div style={{ "font-size": "12px", color: "var(--color-text-secondary)", "margin-bottom": "8px" }}>
            Merges FTS5 internal b-tree segments for faster keyword search performance.
          </div>
          <div class="row" style={{ gap: "8px", "align-items": "center" }}>
            <button class="btn btn-sm" onClick={runOptimize} disabled={optimizing()}>
              {optimizing() ? "Optimizing..." : "Run Optimize"}
            </button>
            <Show when={optimizeStatus()}>
              <span style={{ "font-size": "11px", color: "var(--color-accent-cyan)" }}>
                {optimizeStatus()}
              </span>
            </Show>
          </div>
        </div>

        {/* HNSW Rebuild */}
        <div class="glass-card" style={{ padding: "12px 16px" }}>
          <div style={{ "font-weight": "600", "font-size": "13px", "margin-bottom": "8px" }}>
            Rebuild HNSW Index
          </div>
          <div style={{ "font-size": "12px", color: "var(--color-text-secondary)", "margin-bottom": "8px" }}>
            Recreates the approximate nearest neighbor index from stored embeddings.
            Creates a background job.
          </div>
          <div class="row" style={{ gap: "8px", "align-items": "center" }}>
            <button class="btn btn-sm" onClick={runRebuild} disabled={rebuilding()}>
              {rebuilding() ? "Creating job..." : "Rebuild Index"}
            </button>
            <Show when={rebuildStatus()}>
              <span style={{ "font-size": "11px", color: "var(--color-accent-cyan)" }}>
                {rebuildStatus()}
              </span>
            </Show>
          </div>
        </div>
      </Show>

      {/* Reset to Defaults — always visible, independent of session selection */}
      <div class="glass-card" style={{ padding: "12px 16px" }}>
        <div style={{ "font-weight": "600", "font-size": "13px", "margin-bottom": "8px" }}>
          Reset to Defaults
        </div>
        <div style={{ "font-size": "12px", color: "var(--color-text-secondary)", "margin-bottom": "8px" }}>
          Clears all saved UI preferences from the browser (panel widths, Ollama
          config, file paths, search settings). Sessions and indexed data in the
          database are not affected.
        </div>
        <div class="row" style={{ gap: "8px", "align-items": "center" }}>
          <Show
            when={confirmReset()}
            fallback={
              <button
                class="btn btn-sm"
                style={{ color: "var(--color-accent-magenta)", "border-color": "var(--color-accent-magenta)" }}
                onClick={() => { setConfirmReset(true); setResetStatus(""); }}
              >
                Reset to Defaults
              </button>
            }
          >
            <button
              class="btn btn-sm"
              style={{
                color: "var(--color-accent-magenta)",
                "border-color": "var(--color-accent-magenta)",
                "font-weight": "700",
              }}
              onClick={executeReset}
            >
              Confirm Reset
            </button>
            <button class="btn btn-sm" onClick={() => setConfirmReset(false)}>
              Cancel
            </button>
          </Show>
          <Show when={resetStatus()}>
            <span style={{ "font-size": "11px", color: "var(--color-accent-cyan)" }}>
              {resetStatus()}
            </span>
          </Show>
        </div>
      </div>
    </div>
  );
};

export default MaintenancePanel;
