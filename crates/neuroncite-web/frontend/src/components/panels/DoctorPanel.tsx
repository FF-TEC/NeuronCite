import { Component, For, Show, createResource, createSignal } from "solid-js";
import { api } from "../../api/client";
import type { DependencyProbe } from "../../api/types";
import { isSafeUrl } from "../../utils/url";

/**
 * Doctor panel displaying dependency probe results. Each dependency (pdfium,
 * tesseract, ONNX Runtime, Ollama, poppler) is shown with its availability
 * status (cyan/magenta dot), version info, and an action area that adapts
 * to the dependency state:
 *
 * - Available: checkmark status text
 * - Not available + installable: "Install" button triggers auto-download
 * - Not available + not installable + has link: "Install Manually" link
 * - Not available + not installable + no link: hint text
 *
 * Probes are fetched from GET /api/v1/web/doctor/probes via the typed API client.
 * The install action uses POST /api/v1/web/doctor/install, followed by an
 * automatic re-probe.
 */

const DoctorPanel: Component = () => {
  const [probes, { refetch }] = createResource(fetchProbes);
  const [installing, setInstalling] = createSignal<string | null>(null);
  const [installError, setInstallError] = createSignal<string | null>(null);

  /** Fetches dependency probe results from the server via the typed API client. */
  async function fetchProbes(): Promise<DependencyProbe[]> {
    return api.doctorProbes();
  }

  /** Triggers installation of a dependency using its install_id (the identifier
   *  accepted by POST /api/v1/web/doctor/install). Re-probes afterward to
   *  reflect the new availability state. On failure, extracts the error message
   *  from the ApiError response body for display. */
  const install = async (installId: string) => {
    setInstalling(installId);
    setInstallError(null);
    try {
      await api.doctorInstall({ dependency: installId });
      await refetch();
    } catch (e) {
      // ApiError instances carry the response body as a string. Parse it
      // to extract the error field if possible, otherwise display as-is.
      if (e && typeof e === "object" && "body" in e) {
        try {
          const parsed = JSON.parse((e as { body: string }).body);
          setInstallError(parsed.error || `Install failed`);
        } catch {
          setInstallError(`Install failed: ${e}`);
        }
      } else {
        setInstallError(`Network error: ${e}`);
      }
      await refetch();
    } finally {
      setInstalling(null);
    }
  };

  return (
    <div>
      <div style={{ display: "flex", "justify-content": "space-between", "margin-bottom": "12px" }}>
        <span style={{ "font-size": "13px", color: "var(--color-text-secondary)" }}>
          Dependency availability probes
        </span>
        <button class="btn btn-sm" onClick={() => refetch()} disabled={probes.loading}>
          {probes.loading ? "Probing..." : "Run Probes"}
        </button>
      </div>

      <Show when={probes()}>
        <div style={{ display: "flex", "flex-direction": "column", gap: "8px" }}>
          <For each={probes()}>
            {(probe) => (
              <div
                class="glass-card"
                style={{
                  padding: "10px 16px",
                  display: "flex",
                  "align-items": "center",
                  "justify-content": "space-between",
                  gap: "12px",
                }}
              >
                {/* Left side: status dot + name + purpose + version */}
                <div style={{ display: "flex", "align-items": "center", gap: "10px", "min-width": "0", flex: "1" }}>
                  <div
                    style={{
                      width: "8px",
                      height: "8px",
                      "border-radius": "50%",
                      "flex-shrink": "0",
                      background: probe.available
                        ? "var(--color-accent-cyan)"
                        : "var(--color-accent-magenta)",
                    }}
                  />
                  <div style={{ display: "flex", "flex-direction": "column", gap: "2px", "min-width": "0" }}>
                    <div style={{ display: "flex", "align-items": "center", gap: "8px" }}>
                      <span style={{ "font-size": "13px", "font-weight": "500", "white-space": "nowrap" }}>
                        {probe.name}
                      </span>
                      <Show when={probe.version}>
                        <span style={{ "font-size": "11px", color: "var(--color-text-muted)", "white-space": "nowrap" }}>
                          v{probe.version}
                        </span>
                      </Show>
                      <span style={{ "font-size": "11px", color: probe.available ? "var(--color-accent-cyan)" : "var(--color-accent-magenta)", "white-space": "nowrap" }}>
                        {probe.available ? "Available" : "Not Found"}
                      </span>
                    </div>
                    <span style={{ "font-size": "11px", color: "var(--color-text-muted)" }}>
                      {probe.purpose}
                    </span>
                  </div>
                </div>

                {/* Right side: action area (install button, manual link, or hint) */}
                <div style={{ "flex-shrink": "0", display: "flex", "align-items": "center", gap: "8px" }}>
                  {/* Auto-install button for installable dependencies with a valid install_id */}
                  <Show when={!probe.available && probe.installable && probe.install_id}>
                    <button
                      class="btn btn-sm"
                      onClick={() => install(probe.install_id)}
                      disabled={installing() !== null}
                    >
                      {installing() === probe.install_id ? "Installing..." : "Install"}
                    </button>
                  </Show>

                  {/* Manual install link for non-installable dependencies with a URL */}
                  <Show when={!probe.available && !probe.installable && probe.link && isSafeUrl(probe.link)}>
                    <a
                      href={probe.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      class="btn btn-sm"
                      style={{
                        "text-decoration": "none",
                        "font-size": "11px",
                      }}
                    >
                      Install Manually
                    </a>
                  </Show>

                  {/* Hint text for unavailable deps without link or install capability */}
                  <Show when={!probe.available && !probe.installable && !probe.link}>
                    <span style={{ "font-size": "11px", color: "var(--color-text-muted)", "max-width": "260px" }}>
                      {probe.hint}
                    </span>
                  </Show>
                </div>
              </div>
            )}
          </For>
        </div>
      </Show>

      {/* Installation error message */}
      <Show when={installError()}>
        <div
          style={{
            color: "var(--color-accent-magenta)",
            "font-size": "12px",
            "margin-top": "8px",
            padding: "8px 12px",
            "border-radius": "6px",
            background: "rgba(255, 0, 128, 0.08)",
          }}
        >
          {installError()}
        </div>
      </Show>

      <Show when={probes.error}>
        <div style={{ color: "var(--color-accent-magenta)", "text-align": "center", padding: "20px", "font-size": "13px" }}>
          Failed to load probes. The server may not support this endpoint yet.
        </div>
      </Show>
    </div>
  );
};

export default DoctorPanel;
