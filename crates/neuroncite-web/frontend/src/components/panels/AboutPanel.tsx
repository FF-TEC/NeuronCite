import { Component, Show, For, createSignal } from "solid-js";
import { state } from "../../stores/app";
import { api } from "../../api/client";
import type { UpdateCheckResponse } from "../../api/types";

/**
 * About panel displaying the application version, build features, and
 * runtime backend status. All data is read from the HealthResponse
 * stored in the global app state, which is fetched once on startup
 * from GET /api/v1/health.
 *
 * Includes a "Check for Updates" button that queries the GitHub Releases
 * API to determine whether a newer version is available. This is not an
 * auto-updater -- the user must download and install new versions manually.
 *
 * Renders a two-column key-value layout with labels on the left and
 * values on the right. Build features are displayed as cyan badge pills.
 * When health data has not arrived yet (state.health is null), a
 * loading placeholder is shown.
 */
const AboutPanel: Component = () => {
  const [checking, setChecking] = createSignal(false);
  const [updateResult, setUpdateResult] = createSignal<UpdateCheckResponse | null>(null);
  const [updateError, setUpdateError] = createSignal("");

  const checkForUpdates = async () => {
    setChecking(true);
    setUpdateResult(null);
    setUpdateError("");
    try {
      const res = await api.checkUpdate();
      setUpdateResult(res);
    } catch (e) {
      if (e && typeof e === "object" && "body" in e) {
        try {
          const parsed = JSON.parse((e as { body: string }).body);
          setUpdateError(parsed.error || "Update check failed");
        } catch {
          setUpdateError(`Update check failed: ${e}`);
        }
      } else {
        setUpdateError(`Network error: ${e}`);
      }
    } finally {
      setChecking(false);
    }
  };

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

          {/* Update check -- queries GitHub Releases API for newer versions.
            * Not an auto-updater: the user must download new versions manually. */}
          <div style={{
            display: "flex",
            "align-items": "flex-start",
            gap: "10px",
            "margin-top": "8px",
            "padding-top": "12px",
            "border-top": "1px solid var(--color-border)",
          }}>
            <span style={{ "min-width": "140px", color: "var(--color-text-muted)" }}>Updates</span>
            <div style={{ display: "flex", "flex-direction": "column", gap: "6px" }}>
              <div style={{ display: "flex", gap: "8px", "align-items": "center" }}>
                <button class="btn btn-sm" onClick={checkForUpdates} disabled={checking()}>
                  {checking() ? "Checking..." : "Check for Updates"}
                </button>
              </div>

              {/* Update available */}
              <Show when={updateResult()?.update_available}>
                <div style={{ "font-size": "12px" }}>
                  <span style={{ color: "var(--color-accent-cyan)" }}>
                    Version {updateResult()!.latest_version} available
                  </span>
                  {" \u{2014} "}
                  <a
                    href={updateResult()!.release_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: "var(--color-accent-cyan)", "text-decoration": "underline" }}
                  >
                    View release on GitHub
                  </a>
                </div>
              </Show>

              {/* Already up to date */}
              <Show when={updateResult() && !updateResult()!.update_available}>
                <span style={{ "font-size": "12px", color: "var(--color-accent-cyan)" }}>
                  Up to date (latest: {updateResult()!.latest_version})
                </span>
              </Show>

              {/* Error */}
              <Show when={updateError()}>
                <span style={{ "font-size": "12px", color: "var(--color-accent-magenta)" }}>
                  {updateError()}
                </span>
              </Show>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default AboutPanel;
