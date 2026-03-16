import { Component, createResource, createSignal, For, Show } from "solid-js";
import { api } from "../../api/client";
import type { DependencyProbe } from "../../api/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type StepStatus = "pending" | "loading" | "done" | "error";

interface InstallStep {
  id: string;
  label: string;
}

// ---------------------------------------------------------------------------
// Install step definitions for the "install everything" sequential flow
// ---------------------------------------------------------------------------

/**
 * Steps run in order by runInstallAll. The ONNX Runtime step is explicit here
 * because the doctor install endpoint supports it and the user benefits from
 * seeing its status separately from the model download step.
 */
const STEPS: InstallStep[] = [
  { id: "pdfium",      label: "Installing pdfium (PDF extraction library)" },
  { id: "tesseract",   label: "Installing Tesseract OCR engine" },
  { id: "onnxruntime", label: "Installing ONNX Runtime (AI inference engine)" },
  { id: "model",       label: "Downloading embedding model (BAAI/bge-small-en-v1.5, ~90 MB)" },
  { id: "activate",    label: "Activating embedding model" },
];

// ---------------------------------------------------------------------------
// Probe display helpers
// ---------------------------------------------------------------------------

/**
 * Opens a URL using the wry IPC channel when running inside the native window,
 * or window.open when running in a plain browser during development. The type
 * of `window.ipc` is declared in `src/global.d.ts`, so no manual cast is
 * needed here.
 */
function openExternal(url: string): void {
  if (window.ipc) {
    window.ipc.postMessage(`open_url:${url}`);
  } else {
    window.open(url, "_blank", "noopener,noreferrer");
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * First-run welcome dialog shown once on a fresh installation.
 *
 * The dialog is non-dismissable via backdrop click or Escape -- the user must
 * choose one of the two footer buttons. This guarantees the setup marker file
 * is always written, suppressing the dialog on subsequent launches.
 *
 * On mount the dialog fetches live dependency probe data from the doctor
 * endpoint so the user sees the actual system state rather than a static list.
 * Each probe row shows current availability and offers an Install button for
 * auto-installable dependencies.
 *
 * The "Download and install everything" button runs five install steps
 * sequentially: pdfium, Tesseract, ONNX Runtime, model download, model
 * activation. Progress is shown per-step. A failed step is recorded but does
 * not abort later steps. After completion the setup marker is written and the
 * dialog closes.
 *
 * All API calls go through the typed client (api.doctorProbes, api.doctorInstall,
 * api.modelDownload, api.modelActivate, api.setupComplete) instead of raw fetch().
 *
 * Props:
 *   dataDir -- absolute path shown in the storage location section.
 *   onClose -- called after the marker file has been written.
 */
const WelcomeDialog: Component<{
  dataDir: string;
  onClose: () => void;
}> = (props) => {

  // Live dependency probe data fetched from the doctor endpoint on mount.
  // Provides current availability status for pdfium, tesseract, ONNX Runtime,
  // Ollama, and poppler directly from the running process.
  const [probes, { refetch: refetchProbes }] = createResource<DependencyProbe[]>(async () => {
    try {
      return await api.doctorProbes();
    } catch {
      return [];
    }
  });

  // Per-step status tracking for the install-all sequence.
  const [stepStatuses, setStepStatuses] = createSignal<Record<string, StepStatus>>(
    Object.fromEntries(STEPS.map((s) => [s.id, "pending"])),
  );
  const [stepErrors, setStepErrors] = createSignal<Record<string, string>>(
    Object.fromEntries(STEPS.map((s) => [s.id, ""])),
  );

  // True while the install-all sequence is running -- disables all buttons.
  const [installing, setInstalling] = createSignal(false);

  // True after the user clicks "install everything" -- reveals the step list.
  const [showSteps, setShowSteps] = createSignal(false);

  // Name of the probe currently being installed via its individual row button.
  const [probeInstalling, setProbeInstalling] = createSignal<string | null>(null);

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  const setStatus = (id: string, status: StepStatus) =>
    setStepStatuses((prev) => ({ ...prev, [id]: status }));

  const setError = (id: string, detail: string) =>
    setStepErrors((prev) => ({ ...prev, [id]: detail }));

  /** Writes the first-run marker file via the typed API client and closes
   *  the dialog. If the marker write fails, the dialog reappears on next launch. */
  const markComplete = async () => {
    try {
      await api.setupComplete();
    } catch {
      // If the marker write fails the dialog reappears on next launch.
    }
    props.onClose();
  };

  // ---------------------------------------------------------------------------
  // Individual probe install (row-level Install button)
  // ---------------------------------------------------------------------------

  /**
   * Installs a single dependency via the doctor endpoint and re-fetches all
   * probes afterward to show the updated status in the probe list.
   */
  const installProbe = async (installId: string) => {
    if (!installId) return;
    setProbeInstalling(installId);
    try {
      await api.doctorInstall({ dependency: installId });
    } finally {
      setProbeInstalling(null);
      refetchProbes();
    }
  };

  // ---------------------------------------------------------------------------
  // Install-all orchestration
  // ---------------------------------------------------------------------------

  /**
   * Runs a single dependency install step via the typed API client. Transitions
   * the step through loading -> done | error. Returns true on success, false
   * on failure. Error details are stored in the stepErrors signal.
   */
  const runDoctorStep = async (stepId: string, dependency: string): Promise<boolean> => {
    setStatus(stepId, "loading");
    try {
      await api.doctorInstall({ dependency });
      setStatus(stepId, "done");
      return true;
    } catch (e) {
      setStatus(stepId, "error");
      // Extract error message from ApiError body if possible.
      if (e && typeof e === "object" && "body" in e) {
        try {
          const parsed = JSON.parse((e as { body: string }).body);
          setError(stepId, parsed.error || "Install failed");
        } catch {
          setError(stepId, String(e));
        }
      } else {
        setError(stepId, String(e));
      }
      return false;
    }
  };

  /**
   * Runs all install steps sequentially: pdfium, Tesseract, ONNX Runtime,
   * embedding model download, and model activation. Each step transitions
   * through loading -> done | error. A failed step stores its error message
   * but does not abort subsequent steps, allowing partial success. After all
   * steps finish, probes are re-fetched and the setup marker is written.
   */
  const runInstallAll = async () => {
    setInstalling(true);
    setShowSteps(true);
    setStepStatuses(Object.fromEntries(STEPS.map((s) => [s.id, "pending"])));
    setStepErrors(Object.fromEntries(STEPS.map((s) => [s.id, ""])));

    // pdfium
    await runDoctorStep("pdfium", "pdfium");

    // Tesseract
    await runDoctorStep("tesseract", "tesseract");

    // ONNX Runtime
    await runDoctorStep("onnxruntime", "onnxruntime");

    // Embedding model download
    setStatus("model", "loading");
    try {
      await api.modelDownload({ model_id: "BAAI/bge-small-en-v1.5" });
      setStatus("model", "done");
    } catch (e) {
      setStatus("model", "error");
      if (e && typeof e === "object" && "body" in e) {
        try {
          const parsed = JSON.parse((e as { body: string }).body);
          setError("model", parsed.error || "Download failed");
        } catch {
          setError("model", String(e));
        }
      } else {
        setError("model", String(e));
      }
    }

    // Model activation -- only when download succeeded
    if (stepStatuses()["model"] === "done") {
      setStatus("activate", "loading");
      try {
        await api.modelActivate({ model_id: "BAAI/bge-small-en-v1.5" });
        setStatus("activate", "done");
      } catch (e) {
        setStatus("activate", "error");
        if (e && typeof e === "object" && "body" in e) {
          try {
            const parsed = JSON.parse((e as { body: string }).body);
            setError("activate", parsed.error || "Activation failed");
          } catch {
            setError("activate", String(e));
          }
        } else {
          setError("activate", String(e));
        }
      }
    } else {
      setError("activate", "Skipped -- model download did not complete");
      setStatus("activate", "error");
    }

    setInstalling(false);
    // Re-fetch probes to show which components are now available.
    refetchProbes();
    await markComplete();
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div class="welcome-backdrop">
      <div class="welcome-dialog" role="dialog" aria-modal="true" aria-labelledby="welcome-title">

        {/* Header -- centered with subtitle, main title, and friendly tagline */}
        <div class="welcome-header welcome-header--centered">
          <p class="welcome-subtitle">First launch</p>
          <h1 class="welcome-title" id="welcome-title">Welcome to Neuron<span>Cite</span></h1>
          <p class="welcome-tagline">
            Thank you for choosing a local-first approach to research.
          </p>
        </div>

        {/* Scrollable body */}
        <div class="welcome-body">

          {/* Tool description and privacy statement */}
          <div class="welcome-section">
            <p class="welcome-description">
              NeuronCite is a local semantic search engine for your documents. It indexes
              PDFs, research papers, reports, and other files using AI embedding models,
              letting you search by meaning rather than keywords. It also verifies
              citations and cross-references across your library.
            </p>
            <div class="welcome-privacy-box">
              <span class="welcome-privacy-icon" aria-hidden="true">&#128274;</span>
              <p class="welcome-privacy-text">
                Everything runs on your machine — no data is sent anywhere during normal use.
                Your documents, search queries, and results stay on your computer. When
                NeuronCite is connected as an MCP tool, an external AI agent can send search
                queries to your local server and receives only the matching result snippets —
                it does not have access to your full documents or index.
              </p>
            </div>
          </div>

          {/* Live dependency status -- probes fetched from the doctor endpoint */}
          <div class="welcome-section">
            <p class="welcome-section-title">Required components</p>
            <p class="welcome-description">
              The following components are needed for full functionality. Nothing is downloaded
              without your explicit consent. The current status on your system is shown for each.
            </p>
            <div class="welcome-probe-list">
              <Show
                when={!probes.loading}
                fallback={<div class="welcome-probe-loading">Checking system...</div>}
              >
                <For each={probes() ?? []}>
                  {(probe) => (
                      <div class="welcome-probe-row">
                        {/* Availability dot: cyan = present, magenta = missing */}
                        <div
                          class={`welcome-probe-dot ${
                            probe.available
                              ? "welcome-probe-dot--ok"
                              : "welcome-probe-dot--missing"
                          }`}
                        />
                        {/* Name and one-line purpose description from backend */}
                        <div class="welcome-probe-info">
                          <span class="welcome-probe-name">{probe.name}</span>
                          <span class="welcome-probe-purpose">{probe.purpose}</span>
                        </div>
                        {/* Action area: available label, install button, or manual install link */}
                        <div class="welcome-probe-action">
                          <Show when={probe.available}>
                            <span class="welcome-probe-status-ok">Available</span>
                          </Show>
                          <Show when={!probe.available && !!probe.install_id}>
                            <button
                              class="btn btn-sm"
                              onClick={() => installProbe(probe.install_id)}
                              disabled={installing() || probeInstalling() !== null}
                            >
                              {probeInstalling() === probe.install_id ? "Installing..." : "Install"}
                            </button>
                          </Show>
                          <Show when={!probe.available && !probe.install_id && !!probe.link}>
                            <button
                              class="welcome-link-btn"
                              onClick={() => openExternal(probe.link)}
                            >
                              Manual install
                            </button>
                          </Show>
                          <Show when={!probe.available && !probe.install_id && !probe.link}>
                            <span class="welcome-probe-status-optional">Optional</span>
                          </Show>
                        </div>
                      </div>
                  )}
                </For>
              </Show>

              {/* Embedding model entry -- not a system probe, shown as a fixed informational row */}
              <div class="welcome-probe-row">
                <div class="welcome-probe-dot welcome-probe-dot--info" />
                <div class="welcome-probe-info">
                  <span class="welcome-probe-name">BAAI/bge-small-en-v1.5</span>
                  <span class="welcome-probe-purpose">
                    Embedding model -- converts text to vectors for semantic search (~90 MB, HuggingFace)
                  </span>
                </div>
                <div class="welcome-probe-action">
                  <span class="welcome-probe-status-pending">Downloaded during setup</span>
                </div>
              </div>
            </div>
          </div>

          {/* Storage location with directory tree */}
          <div class="welcome-section">
            <p class="welcome-section-title">Storage location</p>
            <p class="welcome-description">
              All downloaded files are stored in the following directory. You can browse or delete
              them at any time:
            </p>
            <div class="welcome-path-box">
              <div>{props.dataDir}</div>
              <div class="welcome-dir-tree">
                <div class="welcome-dir-row">
                  <span class="welcome-dir-branch">├── </span>
                  <span class="welcome-dir-name">models\</span>
                  <span class="welcome-dir-desc"> embedding model files (ONNX weights + tokenizers)</span>
                </div>
                <div class="welcome-dir-row">
                  <span class="welcome-dir-branch">├── </span>
                  <span class="welcome-dir-name">runtime\</span>
                  <span class="welcome-dir-desc"> ONNX Runtime, pdfium, Tesseract binaries</span>
                </div>
                <div class="welcome-dir-row">
                  <span class="welcome-dir-branch">└── </span>
                  <span class="welcome-dir-name">indexes\</span>
                  <span class="welcome-dir-desc"> document databases and HNSW search indices</span>
                </div>
              </div>
            </div>
          </div>

          {/* Installation progress -- visible only after the user starts install-all */}
          <Show when={showSteps()}>
            <div class="welcome-section">
              <p class="welcome-section-title">Installation progress</p>
              <ul class="welcome-steps">
                <For each={STEPS}>
                  {(step) => {
                    const status = () => stepStatuses()[step.id] as StepStatus;
                    const errorDetail = () => stepErrors()[step.id];
                    return (
                      <li class={`welcome-step welcome-step--${status()}`}>
                        <span class="welcome-step-icon" />
                        <span class="welcome-step-label">
                          {step.label}
                          <Show when={status() === "error" && errorDetail()}>
                            <div class="welcome-step-error-detail">{errorDetail()}</div>
                          </Show>
                        </span>
                      </li>
                    );
                  }}
                </For>
              </ul>
            </div>
          </Show>

        </div>

        {/* Footer with two action choices */}
        <div class="welcome-footer">
          <button
            class="welcome-btn-primary"
            disabled={installing()}
            onClick={runInstallAll}
          >
            {installing() ? "Installing — please wait..." : "Download and install everything"}
          </button>
          <button
            class="welcome-btn-secondary"
            disabled={installing()}
            onClick={markComplete}
          >
            I will set it up myself
          </button>
        </div>

      </div>
    </div>
  );
};

export default WelcomeDialog;
