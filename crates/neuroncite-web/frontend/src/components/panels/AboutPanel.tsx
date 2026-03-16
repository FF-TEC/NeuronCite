import { Component, Show, For } from "solid-js";
import { state } from "../../stores/app";

/**
 * About panel displaying the application version, build features, and
 * runtime backend status. All data is read from the HealthResponse
 * stored in the global app state, which is fetched once on startup
 * from GET /api/v1/health. No additional API calls are made.
 *
 * Renders a two-column key-value layout with labels on the left and
 * values on the right. Build features are displayed as cyan badge pills.
 * When health data has not arrived yet (state.health is null), a
 * loading placeholder is shown.
 */
const AboutPanel: Component = () => {
  return (
    <div>
      <Show when={state.health} fallback={<span style={{ color: "var(--color-text-muted)" }}>Loading...</span>}>
        <div style={{ display: "flex", "flex-direction": "column", gap: "10px" }}>

          {/* Application version from the workspace Cargo.toml */}
          <div style={{ display: "flex", "align-items": "center", gap: "10px" }}>
            <span style={{ "min-width": "140px", color: "var(--color-text-muted)" }}>Version</span>
            <span style={{ "font-family": "var(--font-mono)", color: "var(--color-accent-cyan)" }}>
              {state.health!.version}
            </span>
          </div>

          {/* Compile-time feature flags reported by the health endpoint */}
          <div style={{ display: "flex", "align-items": "flex-start", gap: "10px" }}>
            <span style={{ "min-width": "140px", color: "var(--color-text-muted)" }}>Build Features</span>
            <div style={{ display: "flex", "flex-wrap": "wrap", gap: "6px" }}>
              <For each={state.health!.build_features}>
                {(feature) => (
                  <span class="badge badge-cyan">{feature}</span>
                )}
              </For>
            </div>
          </div>

          {/* Active embedding backend name (e.g. "ort") */}
          <div style={{ display: "flex", "align-items": "center", gap: "10px" }}>
            <span style={{ "min-width": "140px", color: "var(--color-text-muted)" }}>Backend</span>
            <span>{state.health!.active_backend}</span>
          </div>

          {/* GPU availability as reported by the embedding backend */}
          <div style={{ display: "flex", "align-items": "center", gap: "10px" }}>
            <span style={{ "min-width": "140px", color: "var(--color-text-muted)" }}>GPU</span>
            <span style={{ color: state.health!.gpu_available ? "var(--color-accent-cyan)" : "var(--color-text-muted)" }}>
              {state.health!.gpu_available ? "Available" : "Not available"}
            </span>
          </div>

          {/* Pdfium compile-time availability (feature flag pdfium) */}
          <div style={{ display: "flex", "align-items": "center", gap: "10px" }}>
            <span style={{ "min-width": "140px", color: "var(--color-text-muted)" }}>Pdfium</span>
            <span>{state.health!.pdfium_available ? "Compiled in" : "Not compiled"}</span>
          </div>

          {/* Tesseract OCR compile-time availability (feature flag ocr) */}
          <div style={{ display: "flex", "align-items": "center", gap: "10px" }}>
            <span style={{ "min-width": "140px", color: "var(--color-text-muted)" }}>Tesseract OCR</span>
            <span>{state.health!.tesseract_available ? "Compiled in" : "Not compiled"}</span>
          </div>

          {/* Cross-encoder reranker runtime availability */}
          <div style={{ display: "flex", "align-items": "center", gap: "10px" }}>
            <span style={{ "min-width": "140px", color: "var(--color-text-muted)" }}>Reranker</span>
            <span>{state.health!.reranker_available ? "Loaded" : "Not loaded"}</span>
          </div>

          {/* Software license inherited from the workspace Cargo.toml */}
          <div style={{ display: "flex", "align-items": "center", gap: "10px" }}>
            <span style={{ "min-width": "140px", color: "var(--color-text-muted)" }}>License</span>
            <span>AGPL-3.0-only</span>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default AboutPanel;
