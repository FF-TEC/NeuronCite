import { Component, For, Show, createEffect, createResource, createSignal, on, onMount } from "solid-js";
import { state, actions } from "../../stores/app";
import { api } from "../../api/client";
import type {
  EmbedModel,
  RerankModel,
  ModelCatalogResponse,
  ModelHealthEntry,
  OllamaCatalogEntry,
  OllamaModelDto,
} from "../../api/types";

/**
 * Model manager panel displaying system information (GPU, CUDA, Ollama status),
 * three model catalogs (embedding, reranker, Ollama LLM), and action buttons
 * for download/activate/navigate. All three tables share an identical 10-column
 * grid layout enforced by a common <colgroup> and table-layout: fixed CSS.
 *
 * Embedding models: Download -> Activate -> Active
 * Reranker models:  Activate (auto-downloads if needed) -> Active
 * Ollama models:    "Select in Citations" (navigates to the Citations tab)
 *
 * The embedding and reranker catalogs are fetched from GET /api/v1/web/models/catalog.
 * Ollama model list is fetched from GET /api/v1/web/ollama/models on mount.
 */

const ModelsPanel: Component = () => {
  const [catalog, { refetch }] = createResource(fetchCatalog);
  const [downloading, setDownloading] = createSignal<string | null>(null);
  const [activating, setActivating] = createSignal<string | null>(null);
  const [statusMessage, setStatusMessage] = createSignal<string | null>(null);
  const [statusIsError, setStatusIsError] = createSignal(false);

  /** Ollama connection state, probed on mount. */
  const [ollamaConnected, setOllamaConnected] = createSignal(false);
  const [ollamaModels, setOllamaModels] = createSignal<OllamaModelDto[]>([]);
  /** Curated catalog of popular models fetched from the backend. */
  const [ollamaCatalog, setOllamaCatalog] = createSignal<OllamaCatalogEntry[]>([]);
  /** Names of models currently loaded in Ollama GPU/CPU RAM. */
  const [ollamaRunning, setOllamaRunning] = createSignal<string[]>([]);
  /** Model tag currently being deleted (for button disabled state). */
  const [deletingModel, setDeletingModel] = createSignal<string | null>(null);
  /** Controls visibility of the delete confirmation dialog. */
  const [confirmDeleteModel, setConfirmDeleteModel] = createSignal<string | null>(null);

  /** Model Doctor diagnostic state. */
  const [showDiagnostics, setShowDiagnostics] = createSignal(false);
  const [diagnostics, setDiagnostics] = createSignal<ModelHealthEntry[] | null>(null);
  const [repairing, setRepairing] = createSignal<string | null>(null);

  // Watch the store's catalogRefreshTrigger counter and refetch the catalog
  // when it changes. This ensures the ModelsPanel table stays in sync when
  // SSE events (model_downloaded, model_switched, reranker_loaded) arrive
  // from the backend, including events triggered by other browser tabs.
  createEffect(on(
    () => state.catalogRefreshTrigger,
    () => { refetch(); },
    { defer: true }
  ));

  /** Probes Ollama connection, loads installed models, running models, and
   *  catalog on component mount. Uses the stored URL from the app store. */
  onMount(async () => {
    // Fetch the static catalog regardless of Ollama connection status
    try {
      const catalogRes = await api.ollamaCatalog();
      setOllamaCatalog(catalogRes.models);
    } catch {
      // Catalog endpoint is served by NeuronCite itself, failure is unexpected
    }

    try {
      const statusRes = await api.ollamaStatus(state.ollamaUrl);
      setOllamaConnected(statusRes.connected);
      if (statusRes.connected) {
        // Fetch installed models and running models in parallel
        const [modelsRes, runningRes] = await Promise.all([
          api.ollamaModels(state.ollamaUrl),
          api.ollamaRunning(state.ollamaUrl),
        ]);
        setOllamaModels(modelsRes.models);
        const runningNames = runningRes.models.map((m) => m.name);
        setOllamaRunning(runningNames);
        actions.setOllamaRunningModels(runningNames);
        actions.setOllamaModels(modelsRes.models);
      }
    } catch {
      setOllamaConnected(false);
    }
  });

  /** Refreshes Ollama connection status, installed models, and running models.
   *  Called by the manual refresh button in the Ollama section header. */
  const refreshOllama = async () => {
    try {
      const statusRes = await api.ollamaStatus(state.ollamaUrl);
      setOllamaConnected(statusRes.connected);
      if (statusRes.connected) {
        const [modelsRes, runningRes] = await Promise.all([
          api.ollamaModels(state.ollamaUrl),
          api.ollamaRunning(state.ollamaUrl),
        ]);
        setOllamaModels(modelsRes.models);
        const runningNames = runningRes.models.map((m) => m.name);
        setOllamaRunning(runningNames);
        actions.setOllamaRunningModels(runningNames);
        actions.setOllamaModels(modelsRes.models);
      } else {
        setOllamaModels([]);
        setOllamaRunning([]);
        actions.setOllamaRunningModels([]);
      }
    } catch {
      setOllamaConnected(false);
    }
  };

  /** Triggers an Ollama model pull via POST /api/v1/web/ollama/pull.
   *  The pull runs asynchronously in the backend; progress updates arrive
   *  via SSE and are displayed as a progress bar on the row. */
  const pullModel = async (modelName: string) => {
    actions.setOllamaPullingModel(modelName);
    actions.setOllamaPullProgress(null);
    setStatusMessage(null);
    try {
      await api.ollamaPull(modelName, state.ollamaUrl);
      showStatus(`Pulling ${modelName}... Progress is shown below.`, false);
    } catch (e) {
      actions.setOllamaPullingModel(null);
      showStatus(`Pull failed for ${modelName}: ${e}`, true);
    }
  };

  /** Deletes an Ollama model after user confirmation. Refreshes the model
   *  list after successful deletion. */
  const executeDelete = async (modelName: string) => {
    setConfirmDeleteModel(null);
    setDeletingModel(modelName);
    setStatusMessage(null);
    try {
      await api.ollamaDelete(modelName, state.ollamaUrl);
      showStatus(`${modelName} deleted.`, false);
      await refreshOllama();
    } catch (e) {
      showStatus(`Delete failed for ${modelName}: ${e}`, true);
    } finally {
      setDeletingModel(null);
    }
  };

  /** Fetches the model catalog from the server via the typed API client. */
  async function fetchCatalog(): Promise<ModelCatalogResponse> {
    return api.modelCatalog();
  }

  /** Incrementing counter used as a staleness guard for auto-clear timeouts.
   *  Each call to showStatus bumps this counter; the setTimeout callback only
   *  clears the message if the counter has not changed since the timeout was set. */
  let statusGeneration = 0;

  /** Displays a status message to the user with a color indicator.
   *  Success messages auto-clear after 8 seconds, error messages after 12
   *  seconds. A staleness check prevents a newer message from being cleared
   *  by an older timeout that has not yet fired. */
  const showStatus = (message: string, isError: boolean) => {
    setStatusMessage(message);
    setStatusIsError(isError);
    const gen = ++statusGeneration;
    const delay = isError ? 12000 : 8000;
    setTimeout(() => {
      if (statusGeneration === gen) {
        setStatusMessage(null);
      }
    }, delay);
  };

  /** Fetches model health diagnostics from the Model Doctor endpoint and
   *  displays the results in the diagnostics panel below the header. */
  const runDiagnostics = async () => {
    setShowDiagnostics(true);
    setDiagnostics(null);
    try {
      const res = await api.modelDoctor();
      setDiagnostics(res.models);
    } catch (e) {
      showStatus(`Diagnostics failed: ${e}`, true);
    }
  };

  /** Repairs a broken model by purging its cache and re-downloading all
   *  files. After repair completes, re-runs diagnostics and refetches the
   *  catalog so both the diagnostic panel and model tables update. */
  const repairModel = async (modelId: string) => {
    setRepairing(modelId);
    try {
      await api.repairModel(modelId);
      showStatus(`Repair completed for ${modelId.split("/").pop()}.`, false);
      await runDiagnostics();
      await refetch();
    } catch (e) {
      showStatus(`Repair failed for ${modelId.split("/").pop()}: ${e}`, true);
    } finally {
      setRepairing(null);
    }
  };

  /** Triggers embedding model download via POST /api/v1/web/models/download.
   *  Uses the typed API client which handles JSON serialization and error
   *  extraction. The response status field distinguishes "already_cached"
   *  from "downloading" for user feedback. */
  const downloadModel = async (modelId: string) => {
    setDownloading(modelId);
    setStatusMessage(null);
    try {
      const data = await api.modelDownload({ model_id: modelId });
      if (data.status === "already_cached") {
        showStatus(`${modelId.split("/").pop()} is already cached.`, false);
      } else {
        showStatus(`Download started for ${modelId.split("/").pop()}. This may take several minutes.`, false);
      }
      await refetch();
    } catch (e) {
      showStatus(`Download failed: ${e}`, true);
    } finally {
      setDownloading(null);
    }
  };

  /** Activates a cached embedding model via POST /api/v1/web/models/activate.
   *  The backend hot-swaps the embedding model at runtime via ArcSwap
   *  and returns status "activated" with the model ID and dimension.
   *  The topbar model badge is updated immediately via the store action. */
  const activateModel = async (modelId: string) => {
    setActivating(modelId);
    setStatusMessage(null);
    try {
      const data = await api.modelActivate({ model_id: modelId });
      if (data.status === "already_active") {
        showStatus(`${modelId.split("/").pop()} is already active.`, false);
      } else if (data.status === "activated") {
        actions.setActiveModel(data.model_id, data.vector_dimension || 0);
        showStatus(`${modelId.split("/").pop()} activated (${data.vector_dimension}d).`, false);
      }
      await refetch();
    } catch (e) {
      showStatus(`Activation failed: ${e}`, true);
    } finally {
      setActivating(null);
    }
  };

  /** Activates a reranker model via POST /api/v1/web/models/load-reranker.
   *  The backend automatically downloads the model if not cached, verifies
   *  cache integrity, creates an ORT cross-encoder session, and hot-swaps
   *  it into the GPU worker. This single endpoint handles the full lifecycle
   *  (download + load) so the UI only needs one "Activate" button. */
  const activateReranker = async (modelId: string) => {
    setActivating(modelId);
    setStatusMessage(null);
    try {
      const data = await api.loadReranker({ model_id: modelId });
      if (data.status === "already_loaded") {
        showStatus(`${modelId.split("/").pop()} is already active.`, false);
      } else if (data.status === "loaded") {
        actions.setRerankerAvailable(true);
        showStatus(`${modelId.split("/").pop()} activated. Reranking is available in Search.`, false);
      }
      await refetch();
    } catch (e) {
      showStatus(`Reranker activation failed: ${e}`, true);
    } finally {
      setActivating(null);
    }
  };

  /** Maps quality rating strings to CSS color values.
   *  Cyan for top-tier ratings (Very good/high, SOTA, High),
   *  purple for mid-tier (Good, Good+, Solid). */
  const qualityColor = (rating: string) => {
    if (rating.startsWith("Very") || rating.startsWith("SOTA") || rating === "High") return "var(--color-accent-cyan)";
    if (rating.startsWith("Good") || rating === "Solid") return "var(--color-accent-purple)";
    return "var(--color-text-muted)";
  };

  /** Formats a byte or megabyte size value into a human-readable string.
   *  When isBytes is true, converts from bytes to MB/GB. Otherwise treats
   *  the input as megabytes directly. */
  const formatSize = (value: number, isBytes = false): string => {
    const mb = isBytes ? value / (1024 * 1024) : value;
    if (mb >= 1000) return `${(mb / 1000).toFixed(1)} GB`;
    return `${Math.round(mb)} MB`;
  };

  /** Shared colgroup defining identical column widths for all three model
   *  tables (embedding, reranker, Ollama). All tables use 10 columns:
   *  Name | Quality | Spec-A | Spec-B | Languages | Size | GPU | RAM | Status | Action.
   *  The percentages are tuned so the widest content per column
   *  (e.g. "Strongly recommended" in GPU) fits without overflow. */
  const TableColgroup = () => (
    <colgroup>
      <col style={{ width: "15%" }} />
      <col style={{ width: "8%" }} />
      <col style={{ width: "5%" }} />
      <col style={{ width: "6%" }} />
      <col style={{ width: "10%" }} />
      <col style={{ width: "7%" }} />
      <col style={{ width: "14%" }} />
      <col style={{ width: "8%" }} />
      <col style={{ width: "10%" }} />
      <col style={{ width: "17%" }} />
    </colgroup>
  );

  /** Muted dash placeholder for cells where a column does not apply. */
  const Dash = () => (
    <span style={{ color: "var(--color-text-muted)" }}>--</span>
  );

  return (
    <div>
      {/* Tab header with Model Doctor button */}
      <div style={{ display: "flex", "align-items": "center", gap: "10px", "margin-bottom": "16px" }}>
        <div style={{ "font-weight": "600", "font-size": "16px", color: "var(--color-text-primary)" }}>
          Model Manager
        </div>
        <button
          class="btn btn-sm btn-primary"
          style={{ "font-size": "11px", padding: "2px 10px", "margin-left": "auto" }}
          onClick={runDiagnostics}
          title="Run file-level health diagnostics on all cached models"
        >
          Diagnostics
        </button>
      </div>

      {/* Model Doctor diagnostic panel (collapsible, shown after clicking Diagnostics) */}
      <Show when={showDiagnostics()}>
        <div style={{
          padding: "10px 14px",
          "margin-bottom": "14px",
          "border-radius": "8px",
          background: "rgba(139, 92, 246, 0.08)",
          border: "1px solid rgba(139, 92, 246, 0.2)",
          "font-size": "12px",
        }}>
          <div style={{ display: "flex", "justify-content": "space-between", "align-items": "center", "margin-bottom": "8px" }}>
            <span style={{ "font-weight": "600", color: "var(--color-accent-purple)" }}>Model Doctor</span>
            <button
              class="btn btn-sm"
              style={{ "font-size": "10px", padding: "1px 6px" }}
              onClick={() => setShowDiagnostics(false)}
            >
              Close
            </button>
          </div>
          <Show when={diagnostics() === null}>
            <div style={{ color: "var(--color-text-muted)" }}>Running diagnostics...</div>
          </Show>
          <Show when={diagnostics() !== null}>
            <div style={{ display: "flex", "flex-direction": "column", gap: "6px" }}>
              <For each={diagnostics()!}>
                {(entry) => {
                  const healthColor = () => {
                    if (entry.health === "healthy") return "var(--color-accent-cyan)";
                    if (entry.health === "missing") return "var(--color-text-muted)";
                    return "var(--color-accent-magenta)";
                  };
                  const healthLabel = () => {
                    if (entry.health === "healthy") return "Healthy";
                    if (entry.health === "missing") return "Not Downloaded";
                    if (entry.health === "incomplete") return "Incomplete";
                    return "Corrupt";
                  };
                  return (
                    <div style={{
                      display: "flex",
                      "align-items": "center",
                      gap: "10px",
                      padding: "4px 8px",
                      "border-radius": "4px",
                      background: "rgba(255,255,255,0.03)",
                    }}>
                      <span style={{ flex: "1", "font-weight": "500" }}>{entry.model_id}</span>
                      <span style={{ color: healthColor(), "min-width": "90px" }}>{healthLabel()}</span>
                      <span style={{ color: "var(--color-text-muted)", "min-width": "60px" }}>
                        {entry.total_size_bytes > 0 ? formatSize(entry.total_size_bytes, true) : "--"}
                      </span>
                      <Show when={entry.files_missing.length > 0}>
                        <span style={{ color: "var(--color-accent-magenta)", "font-size": "11px" }}>
                          Missing: {entry.files_missing.join(", ")}
                        </span>
                      </Show>
                      <Show when={entry.health !== "healthy" && entry.health !== "missing" && entry.repairable}>
                        <button
                          class="btn btn-sm"
                          style={{ "font-size": "10px", padding: "1px 8px", color: "var(--color-accent-magenta)" }}
                          onClick={() => repairModel(entry.model_id)}
                          disabled={repairing() !== null}
                        >
                          {repairing() === entry.model_id ? "Repairing..." : "Repair"}
                        </button>
                      </Show>
                    </div>
                  );
                }}
              </For>
            </div>
          </Show>
        </div>
      </Show>

      {/* Status message bar */}
      <Show when={statusMessage()}>
        <div
          style={{
            padding: "8px 12px",
            "margin-bottom": "12px",
            "border-radius": "6px",
            "font-size": "12px",
            "line-height": "1.5",
            background: statusIsError()
              ? "rgba(244, 63, 94, 0.15)"
              : "rgba(34, 211, 238, 0.1)",
            color: statusIsError()
              ? "var(--color-accent-magenta)"
              : "var(--color-accent-cyan)",
            border: `1px solid ${statusIsError() ? "rgba(244, 63, 94, 0.3)" : "rgba(34, 211, 238, 0.2)"}`,
          }}
        >
          {statusMessage()}
        </div>
      </Show>

      {/* System Info */}
      <Show when={catalog()}>
        <div style={{ display: "flex", gap: "16px", "margin-bottom": "16px", "font-size": "13px", "flex-wrap": "wrap" }}>
          <div>
            <span style={{ color: "var(--color-text-muted)" }}>GPU: </span>
            <span>{catalog()!.gpu_name}</span>
          </div>
          <div>
            <span style={{ color: "var(--color-text-muted)" }}>CUDA: </span>
            <span style={{ color: catalog()!.cuda_available ? "var(--color-accent-cyan)" : "var(--color-accent-magenta)" }}>
              {catalog()!.cuda_available ? "Available" : "Not Available"}
            </span>
          </div>
          <div>
            <span style={{ color: "var(--color-text-muted)" }}>Active Model: </span>
            <span style={{ color: "var(--color-accent-cyan)" }}>
              {state.activeModelId || "None"}
            </span>
          </div>
          <div>
            <span style={{ color: "var(--color-text-muted)" }}>Ollama: </span>
            <span style={{ color: ollamaConnected() ? "var(--color-accent-cyan)" : "var(--color-accent-magenta)" }}>
              {ollamaConnected() ? "Available" : "Not Available"}
            </span>
          </div>
        </div>

        {/* ================================================================
         *  Embedding Models Table
         *  10 columns: Model | Quality | Dim | Context | Languages | Size | GPU | RAM | Status | Action
         * ================================================================ */}
        <div style={{ "margin-bottom": "20px" }}>
          <div style={{ "font-weight": "600", "font-size": "14px", "margin-bottom": "8px" }}>
            Embedding Models
          </div>
          <div style={{ "overflow-x": "auto" }}>
            <table class="model-table">
              <TableColgroup />
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Quality</th>
                  <th>Dim</th>
                  <th>Context</th>
                  <th>Languages</th>
                  <th>Size</th>
                  <th>GPU</th>
                  <th>RAM</th>
                  <th>Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                <For each={catalog()!.embedding_models}>
                  {(model: EmbedModel) => (
                    <tr>
                      <td style={{ "font-weight": "500" }}>{model.display_name}</td>
                      <td style={{ color: qualityColor(model.quality_rating) }}>{model.quality_rating}</td>
                      <td>{model.vector_dimension}</td>
                      <td>{model.max_seq_len}</td>
                      <td>{model.language_scope}</td>
                      <td>{formatSize(model.model_size_mb)}</td>
                      <td>{model.gpu_recommendation}</td>
                      <td>{model.ram_requirement}</td>
                      <td>
                        <Show when={model.active}>
                          <span class="badge badge-cyan">Active</span>
                        </Show>
                        <Show when={model.cached && !model.active}>
                          <span class="badge badge-purple">Cached</span>
                        </Show>
                        <Show when={!model.cached}>
                          <span class="badge badge-gray">Not Cached</span>
                        </Show>
                      </td>
                      <td>
                        <Show when={!model.cached}>
                          <button
                            class="btn btn-sm"
                            onClick={() => downloadModel(model.model_id)}
                            disabled={downloading() !== null}
                          >
                            {downloading() === model.model_id ? "Downloading..." : "Download"}
                          </button>
                        </Show>
                        <Show when={model.cached && !model.active}>
                          <button
                            class="btn btn-sm btn-primary"
                            onClick={() => activateModel(model.model_id)}
                            disabled={activating() !== null}
                          >
                            {activating() === model.model_id ? "Activating..." : "Activate"}
                          </button>
                        </Show>
                        <Show when={model.active}>
                          <span style={{ "font-size": "11px", color: "var(--color-text-muted)" }}>--</span>
                        </Show>
                      </td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
        </div>

        {/* ================================================================
         *  Reranker Models Table
         *  10 columns matching embedding table: Model | Quality | Layers | Params | Languages | Size | GPU | RAM | Status | Action
         *  The backend load-reranker endpoint handles download + load in one
         *  step, so only a single "Activate" button is needed.
         * ================================================================ */}
        <div style={{ "margin-bottom": "20px" }}>
          <div style={{ "font-weight": "600", "font-size": "14px", "margin-bottom": "8px" }}>
            Reranker Models
          </div>
          <div style={{ "overflow-x": "auto" }}>
            <table class="model-table">
              <TableColgroup />
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Quality</th>
                  <th>Layers</th>
                  <th>Params</th>
                  <th>Languages</th>
                  <th>Size</th>
                  <th>GPU</th>
                  <th>RAM</th>
                  <th>Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                <For each={catalog()!.reranker_models}>
                  {(model: RerankModel) => (
                    <tr>
                      <td style={{ "font-weight": "500" }}>{model.display_name}</td>
                      <td style={{ color: qualityColor(model.quality_rating) }}>{model.quality_rating}</td>
                      <td>{model.layer_count}</td>
                      <td>{model.param_count_m}M</td>
                      <td>{model.language_scope}</td>
                      <td>{formatSize(model.model_size_mb)}</td>
                      <td>{model.gpu_recommendation}</td>
                      <td>{model.ram_requirement}</td>
                      <td>
                        <Show when={model.loaded}>
                          <span class="badge badge-cyan">Active</span>
                        </Show>
                        <Show when={model.cached && !model.loaded}>
                          <span class="badge badge-purple">Cached</span>
                        </Show>
                        <Show when={!model.cached}>
                          <span class="badge badge-gray">Not Cached</span>
                        </Show>
                      </td>
                      <td>
                        <Show when={!model.cached}>
                          <button
                            class="btn btn-sm"
                            onClick={() => downloadModel(model.model_id)}
                            disabled={downloading() !== null}
                          >
                            {downloading() === model.model_id ? "Downloading..." : "Download"}
                          </button>
                        </Show>
                        <Show when={model.cached && !model.loaded}>
                          <button
                            class="btn btn-sm btn-primary"
                            onClick={() => activateReranker(model.model_id)}
                            disabled={activating() !== null}
                          >
                            {activating() === model.model_id ? "Activating..." : "Activate"}
                          </button>
                        </Show>
                        <Show when={model.loaded}>
                          <span style={{ "font-size": "11px", color: "var(--color-text-muted)" }}>--</span>
                        </Show>
                      </td>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
        </div>

        {/* ================================================================
         *  LLM Models (Ollama) Table
         *  Displays a merged view of installed models, running models, and
         *  the curated catalog. Installed models appear first (sorted by name),
         *  followed by catalog entries that are not yet installed. Each row
         *  shows status badges (Loaded/Installed/Available) and action buttons
         *  (Pull/Delete/Select in Citations) based on the model's state.
         *
         *  Pull operations run asynchronously in the backend. Progress updates
         *  arrive via SSE and are displayed as a progress bar on the row.
         * ================================================================ */}
        <div>
          <div style={{ display: "flex", "align-items": "center", gap: "8px", "margin-bottom": "8px" }}>
            <div style={{ "font-weight": "600", "font-size": "14px" }}>
              LLM Models (Ollama)
            </div>
            <button
              class="btn btn-sm"
              style={{ "font-size": "11px", padding: "2px 8px" }}
              onClick={refreshOllama}
              title="Refresh Ollama connection and model list"
            >
              Refresh
            </button>
          </div>

          {/* Delete confirmation dialog */}
          <Show when={confirmDeleteModel()}>
            <div style={{
              padding: "8px 12px",
              "margin-bottom": "8px",
              "border-radius": "6px",
              "font-size": "12px",
              background: "rgba(244, 63, 94, 0.1)",
              border: "1px solid rgba(244, 63, 94, 0.3)",
              display: "flex",
              "align-items": "center",
              gap: "12px",
            }}>
              <span>Delete <strong>{confirmDeleteModel()}</strong>? This removes the model files from disk.</span>
              <button
                class="btn btn-sm"
                style={{ background: "rgba(244, 63, 94, 0.3)", color: "var(--color-accent-magenta)" }}
                onClick={() => executeDelete(confirmDeleteModel()!)}
              >
                Confirm
              </button>
              <button
                class="btn btn-sm"
                onClick={() => setConfirmDeleteModel(null)}
              >
                Cancel
              </button>
            </div>
          </Show>

          <div style={{ "overflow-x": "auto" }}>
            <table class="model-table">
              <TableColgroup />
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Family</th>
                  <th>Params</th>
                  <th>Quant</th>
                  <th>Description</th>
                  <th>Size</th>
                  <th></th>
                  <th></th>
                  <th>Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {/* Installed models first */}
                <For each={ollamaModels()}>
                  {(model) => {
                    const isLoaded = () => ollamaRunning().includes(model.name);
                    const isPulling = () => state.ollamaPullingModel === model.name;
                    const isDeleting = () => deletingModel() === model.name;
                    // Find matching catalog entry for the description column
                    const catalogEntry = () => ollamaCatalog().find((c) => c.name === model.name);
                    return (
                      <tr>
                        <td style={{ "font-weight": "500" }}>{model.name}</td>
                        <td>{model.family || <Dash />}</td>
                        <td>{model.parameter_size || <Dash />}</td>
                        <td>{model.quantization_level || <Dash />}</td>
                        <td style={{ "font-size": "11px", color: "var(--color-text-muted)" }}>
                          {catalogEntry()?.description || <Dash />}
                        </td>
                        <td>{formatSize(model.size, true)}</td>
                        <td><Dash /></td>
                        <td><Dash /></td>
                        <td>
                          <Show when={isLoaded()}>
                            <span class="badge badge-cyan">Loaded</span>
                          </Show>
                          <Show when={!isLoaded()}>
                            <span class="badge badge-purple">Installed</span>
                          </Show>
                        </td>
                        <td style={{ display: "flex", gap: "4px", "align-items": "center" }}>
                          <button
                            class="btn btn-sm btn-primary"
                            onClick={() => {
                              actions.setOllamaSelectedModel(model.name);
                              actions.setActiveTab("citations");
                            }}
                          >
                            Select
                          </button>
                          <button
                            class="btn btn-sm"
                            style={{ "font-size": "11px", padding: "2px 6px", color: "var(--color-text-muted)" }}
                            onClick={() => setConfirmDeleteModel(model.name)}
                            disabled={isDeleting() || isPulling()}
                            title="Delete model"
                          >
                            Del
                          </button>
                        </td>
                      </tr>
                    );
                  }}
                </For>

                {/* Catalog models that are not installed */}
                <For each={ollamaCatalog().filter((c) => !ollamaModels().some((m) => m.name === c.name))}>
                  {(entry) => {
                    const isPulling = () => state.ollamaPullingModel === entry.name;
                    const pullProgress = () => state.ollamaPullProgress;
                    return (
                      <tr>
                        <td style={{ "font-weight": "500" }}>{entry.display_name}</td>
                        <td>{entry.family}</td>
                        <td>{entry.parameter_size}</td>
                        <td><Dash /></td>
                        <td style={{ "font-size": "11px", color: "var(--color-text-muted)" }}>
                          {entry.description}
                        </td>
                        <td>{formatSize(entry.size_mb)}</td>
                        <td><Dash /></td>
                        <td><Dash /></td>
                        <td>
                          <Show when={isPulling()}>
                            {/* Progress bar during pull operation */}
                            <div style={{ width: "100%", "min-width": "60px" }}>
                              <div style={{
                                height: "6px",
                                "border-radius": "3px",
                                background: "rgba(255,255,255,0.1)",
                                overflow: "hidden",
                              }}>
                                <div style={{
                                  height: "100%",
                                  "border-radius": "3px",
                                  background: "var(--color-accent-cyan)",
                                  width: pullProgress() && pullProgress()!.total > 0
                                    ? `${Math.round((pullProgress()!.completed / pullProgress()!.total) * 100)}%`
                                    : "0%",
                                  transition: "width 0.3s ease",
                                }} />
                              </div>
                              <div style={{ "font-size": "10px", color: "var(--color-text-muted)", "margin-top": "2px" }}>
                                {pullProgress() && pullProgress()!.total > 0
                                  ? `${Math.round((pullProgress()!.completed / pullProgress()!.total) * 100)}%`
                                  : "Starting..."}
                              </div>
                            </div>
                          </Show>
                          <Show when={!isPulling()}>
                            <span class="badge badge-gray">Available</span>
                          </Show>
                        </td>
                        <td>
                          <Show when={!isPulling()}>
                            <button
                              class="btn btn-sm"
                              onClick={() => pullModel(entry.name)}
                              disabled={state.ollamaPullingModel !== null}
                            >
                              Pull
                            </button>
                          </Show>
                          <Show when={isPulling()}>
                            <span style={{ "font-size": "11px", color: "var(--color-text-muted)" }}>Pulling...</span>
                          </Show>
                        </td>
                      </tr>
                    );
                  }}
                </For>

                {/* Empty state when no models and no catalog */}
                <Show when={ollamaModels().length === 0 && ollamaCatalog().length === 0 && !ollamaConnected()}>
                  <tr>
                    <td colspan="10" style={{ color: "var(--color-text-muted)", "text-align": "center" }}>
                      Ollama is not running. Start Ollama to manage LLM models.
                    </td>
                  </tr>
                </Show>
              </tbody>
            </table>
          </div>
        </div>
      </Show>

      {/* Loading state */}
      <Show when={catalog.loading}>
        <div style={{ color: "var(--color-text-muted)", "text-align": "center", padding: "20px" }}>
          Loading model catalog...
        </div>
      </Show>

      {/* Error state */}
      <Show when={catalog.error}>
        <div style={{ color: "var(--color-accent-magenta)", "text-align": "center", padding: "20px" }}>
          Failed to load model catalog. The server may not support this endpoint yet.
        </div>
      </Show>
    </div>
  );
};

export default ModelsPanel;
