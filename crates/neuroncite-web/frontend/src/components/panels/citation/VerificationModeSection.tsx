/**
 * Verification mode selector section for the Citation panel left sidebar.
 * Provides three preset modes (Quick, Standard, Thorough) and a Custom mode
 * that reveals individual parameter sliders for fine-grained control over
 * search depth, reasoning length, temperature, retry behavior, and reranking.
 *
 * The component exposes the currently active verification parameters via
 * the activeParams() accessor passed through props. When a preset is
 * selected, the preset values are returned. When Custom is selected, the
 * individual slider values are returned.
 *
 * This component does not perform any API calls. It manages local signal
 * state for the custom parameter sliders and delegates the combined
 * parameter set to the parent via the onParamsChange callback.
 */

import { Component, For, Show, createSignal } from "solid-js";
import { state } from "../../../stores/app";
import Tip from "../../ui/Tip";
import { PRESETS, MODE_INFO, LABEL_STYLE } from "./constants";
import type { VerifyParams, VerifyMode } from "./constants";

/** Props for VerificationModeSection. The parent reads the active params
 *  via the accessor returned from this component, and the component
 *  notifies the parent on mode/param changes. */
interface VerificationModeSectionProps {
  /** Callback invoked whenever the active verification parameters change.
   *  The parent uses this to pass parameters to the auto-verify API call. */
  onParamsChange: (params: VerifyParams) => void;
}

/** Return type from useVerificationMode, providing reactive accessors
 *  for the current mode and computed parameters. */
export interface VerificationModeHandle {
  /** The currently selected verification mode (quick, standard, thorough, custom). */
  verifyMode: () => VerifyMode;
  /** Computed verification parameters for the current mode. For presets,
   *  returns the preset values. For custom, returns the slider values. */
  activeParams: () => VerifyParams;
  /** The JSX component to render inside the left panel. */
  VerificationModeUI: Component;
}

/**
 * Hook-style composable that creates the verification mode state and
 * returns a component plus accessors. This pattern allows the parent
 * to read activeParams() without prop drilling, while the UI rendering
 * is encapsulated in VerificationModeUI.
 */
export function useVerificationMode(): VerificationModeHandle {
  // Verification mode and custom parameter signals.
  const [verifyMode, setVerifyMode] = createSignal<VerifyMode>("standard");
  const [customTopK, setCustomTopK] = createSignal(5);
  const [customCrossCorpus, setCustomCrossCorpus] = createSignal(2);
  const [customMaxTokens, setCustomMaxTokens] = createSignal(4096);
  const [customTemperature, setCustomTemperature] = createSignal(0.1);
  const [customMaxRetry, setCustomMaxRetry] = createSignal(3);
  const [customMinScore, setCustomMinScore] = createSignal<number | undefined>(undefined);
  const [customMinScoreEnabled, setCustomMinScoreEnabled] = createSignal(false);
  const [customRerank, setCustomRerank] = createSignal(false);

  /** Returns the verification parameters for the currently selected mode.
   *  For preset modes, returns the preset values. For custom mode, reads
   *  the individual signal values. */
  const activeParams = (): VerifyParams => {
    const mode = verifyMode();
    if (mode !== "custom") return PRESETS[mode];
    return {
      top_k: customTopK(),
      cross_corpus_queries: customCrossCorpus(),
      max_tokens: customMaxTokens(),
      temperature: customTemperature(),
      max_retry_attempts: customMaxRetry(),
      min_score: customMinScoreEnabled() ? customMinScore() : undefined,
      rerank: customRerank(),
    };
  };

  /** When switching to custom mode, copies the current preset values into
   *  the custom signals so the user starts from a familiar baseline. */
  const switchMode = (mode: VerifyMode) => {
    if (mode === "custom" && verifyMode() !== "custom") {
      const current = activeParams();
      setCustomTopK(current.top_k);
      setCustomCrossCorpus(current.cross_corpus_queries);
      setCustomMaxTokens(current.max_tokens);
      setCustomTemperature(current.temperature);
      setCustomMaxRetry(current.max_retry_attempts);
      setCustomMinScore(current.min_score);
      setCustomMinScoreEnabled(current.min_score !== undefined);
      setCustomRerank(current.rerank);
    }
    setVerifyMode(mode);
  };

  /** The rendered UI component for the verification mode section. */
  const VerificationModeUI: Component = () => (
    <div class="glass-card" style={{ padding: "16px" }}>
      <div class="section-title">
        Verification Mode
        <Tip text="Controls how thoroughly each citation is verified. Presets adjust search depth, reasoning length, and retry behavior. Custom mode reveals individual parameter sliders for full control." />
      </div>

      {/* Mode selector dropdown */}
      <div style={{ display: "flex", gap: "8px", "align-items": "center", "margin-bottom": "8px" }}>
        <label style={LABEL_STYLE}>
          Mode
          <Tip text="Quick: fast with fewer searches. Standard: balanced depth and speed (default). Thorough: deep search with reranking. Custom: manual parameter control." />
        </label>
        <select
          class="select"
          style={{ flex: "1" }}
          value={verifyMode()}
          onChange={(e) => switchMode(e.target.value as VerifyMode)}
        >
          <For each={(["quick", "standard", "thorough", "custom"] as VerifyMode[])}>
            {(mode) => (
              <option value={mode}>
                {MODE_INFO[mode].label} - {MODE_INFO[mode].desc}
              </option>
            )}
          </For>
        </select>
      </div>

      {/* Parameter summary for non-custom modes: compact two-column grid
       *  showing the preset values at a glance. */}
      <Show when={verifyMode() !== "custom"}>
        <div style={{
          display: "grid",
          "grid-template-columns": "1fr 1fr",
          gap: "4px 16px",
          "font-size": "11px",
          color: "var(--color-text-muted)",
          padding: "4px 0",
        }}>
          <span>Search Depth: <span style={{ color: "var(--color-text-secondary)" }}>{activeParams().top_k}</span></span>
          <span>Cross-Corpus: <span style={{ color: "var(--color-text-secondary)" }}>{activeParams().cross_corpus_queries}</span></span>
          <span>Reasoning: <span style={{ color: "var(--color-text-secondary)" }}>{activeParams().max_tokens}</span></span>
          <span>Temperature: <span style={{ color: "var(--color-text-secondary)" }}>{activeParams().temperature}</span></span>
          <span>Retries: <span style={{ color: "var(--color-text-secondary)" }}>{activeParams().max_retry_attempts}</span></span>
          <span>Min Score: <span style={{ color: "var(--color-text-secondary)" }}>{activeParams().min_score ?? "off"}</span></span>
          <span>Reranking: <span style={{ color: "var(--color-text-secondary)" }}>{activeParams().rerank ? "on" : "off"}</span></span>
        </div>
      </Show>

      {/* Custom parameter controls (visible only in custom mode). Each
       *  parameter has a labeled range slider with the current value displayed. */}
      <Show when={verifyMode() === "custom"}>
        <div style={{ display: "flex", "flex-direction": "column", gap: "10px", "margin-top": "4px" }}>

          {/* Search Depth (top_k) */}
          <div>
            <div style={{ display: "flex", "justify-content": "space-between", "font-size": "12px", "margin-bottom": "2px" }}>
              <span style={{ color: "var(--color-text-secondary)" }}>
                Search Depth
                <Tip text="Number of search results returned per individual query. Higher values find more relevant passages but slow down verification. Range: 1-20." />
              </span>
              <span style={{ color: "var(--color-accent-cyan)", "font-family": "monospace" }}>{customTopK()}</span>
            </div>
            <input type="range" min="1" max="20" step="1" value={customTopK()} onInput={(e) => setCustomTopK(parseInt(e.target.value, 10))} style={{ width: "100%", "accent-color": "var(--color-accent-purple)" }} />
          </div>

          {/* Cross-Corpus Queries */}
          <div>
            <div style={{ display: "flex", "justify-content": "space-between", "font-size": "12px", "margin-bottom": "2px" }}>
              <span style={{ color: "var(--color-text-secondary)" }}>
                Cross-Corpus Queries
                <Tip text="Number of additional search queries executed across all sessions to find alternative sources for each citation. 0 disables cross-corpus search entirely. Range: 0-4." />
              </span>
              <span style={{ color: "var(--color-accent-cyan)", "font-family": "monospace" }}>{customCrossCorpus()}</span>
            </div>
            <input type="range" min="0" max="4" step="1" value={customCrossCorpus()} onInput={(e) => setCustomCrossCorpus(parseInt(e.target.value, 10))} style={{ width: "100%", "accent-color": "var(--color-accent-purple)" }} />
          </div>

          {/* Reasoning Length (max_tokens) */}
          <div>
            <div style={{ display: "flex", "justify-content": "space-between", "font-size": "12px", "margin-bottom": "2px" }}>
              <span style={{ color: "var(--color-text-secondary)" }}>
                Reasoning Length
                <Tip text="Maximum number of tokens the LLM generates per verdict. Longer reasoning allows more detailed analysis but increases verification time. Range: 1024-16384." />
              </span>
              <span style={{ color: "var(--color-accent-cyan)", "font-family": "monospace" }}>{customMaxTokens()}</span>
            </div>
            <input type="range" min="1024" max="16384" step="512" value={customMaxTokens()} onInput={(e) => setCustomMaxTokens(parseInt(e.target.value, 10))} style={{ width: "100%", "accent-color": "var(--color-accent-purple)" }} />
          </div>

          {/* Temperature */}
          <div>
            <div style={{ display: "flex", "justify-content": "space-between", "font-size": "12px", "margin-bottom": "2px" }}>
              <span style={{ color: "var(--color-text-secondary)" }}>
                Temperature
                <Tip text="LLM sampling temperature controlling output randomness. Lower values produce more deterministic, reproducible verdicts. 0.0 is fully deterministic, 2.0 is highly random. Range: 0.0-2.0." />
              </span>
              <span style={{ color: "var(--color-accent-cyan)", "font-family": "monospace" }}>{customTemperature().toFixed(2)}</span>
            </div>
            <input type="range" min="0" max="2" step="0.05" value={customTemperature()} onInput={(e) => setCustomTemperature(parseFloat(e.target.value))} style={{ width: "100%", "accent-color": "var(--color-accent-purple)" }} />
          </div>

          {/* Retry Attempts */}
          <div>
            <div style={{ display: "flex", "justify-content": "space-between", "font-size": "12px", "margin-bottom": "2px" }}>
              <span style={{ color: "var(--color-text-secondary)" }}>
                Retry Attempts
                <Tip text="Maximum times the agent retries when the LLM produces an unparseable verdict JSON. Higher values reduce failed rows at the cost of additional LLM calls. Range: 1-5." />
              </span>
              <span style={{ color: "var(--color-accent-cyan)", "font-family": "monospace" }}>{customMaxRetry()}</span>
            </div>
            <input type="range" min="1" max="5" step="1" value={customMaxRetry()} onInput={(e) => setCustomMaxRetry(parseInt(e.target.value, 10))} style={{ width: "100%", "accent-color": "var(--color-accent-purple)" }} />
          </div>

          {/* Min Score Threshold */}
          <div>
            <div style={{ display: "flex", "justify-content": "space-between", "font-size": "12px", "margin-bottom": "2px" }}>
              <span style={{ color: "var(--color-text-secondary)" }}>
                Min Score Threshold
                <Tip text="Minimum cosine similarity for search results. Results below this score are discarded. Disable to include all results regardless of similarity. Range: 0.0-1.0." />
              </span>
              <div style={{ display: "flex", "align-items": "center", gap: "6px" }}>
                <Show when={customMinScoreEnabled()}>
                  <span style={{ color: "var(--color-accent-cyan)", "font-family": "monospace" }}>{(customMinScore() ?? 0).toFixed(2)}</span>
                </Show>
                <label style={{ display: "flex", "align-items": "center", gap: "4px", cursor: "pointer", "font-size": "11px", color: "var(--color-text-muted)" }}>
                  <input
                    type="checkbox"
                    checked={customMinScoreEnabled()}
                    onChange={(e) => {
                      setCustomMinScoreEnabled(e.target.checked);
                      if (e.target.checked && customMinScore() === undefined) {
                        setCustomMinScore(0.5);
                      }
                    }}
                    style={{ "accent-color": "var(--color-accent-purple)" }}
                  />
                  {customMinScoreEnabled() ? "on" : "off"}
                </label>
              </div>
            </div>
            <Show when={customMinScoreEnabled()}>
              <input type="range" min="0" max="1" step="0.05" value={customMinScore() ?? 0.5} onInput={(e) => setCustomMinScore(parseFloat(e.target.value))} style={{ width: "100%", "accent-color": "var(--color-accent-purple)" }} />
            </Show>
          </div>

          {/* Reranking toggle with warning when no reranker model is loaded */}
          <label style={{
            display: "flex",
            "align-items": "center",
            gap: "8px",
            "font-size": "12px",
            color: "var(--color-text-secondary)",
            cursor: "pointer",
          }}>
            <input
              type="checkbox"
              checked={customRerank()}
              onChange={(e) => setCustomRerank(e.target.checked)}
              style={{ "accent-color": "var(--color-accent-purple)" }}
            />
            Enable Reranking
            <Tip text="Applies cross-encoder reranking to search results for higher precision. Requires a reranker model to be loaded via the Indexing tab. Increases latency per search." />
            <Show when={!state.rerankerAvailable && customRerank()}>
              <span style={{ color: "var(--color-accent-amber)", "font-size": "10px" }}>(no reranker loaded)</span>
            </Show>
          </label>

        </div>
      </Show>
    </div>
  );

  return {
    verifyMode,
    activeParams,
    VerificationModeUI,
  };
}
