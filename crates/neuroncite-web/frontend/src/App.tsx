import {
  Component,
  Switch,
  Match,
  onMount,
  onCleanup,
  createSignal,
  Show,
  lazy,
  Suspense,
  ErrorBoundary,
} from "solid-js";
import Topbar from "./components/layout/Topbar";
import StatusBar from "./components/layout/StatusBar";
import { state, actions, safeLsGet, safeLsRemove, LS_CITATION_JOB_ID } from "./stores/app";
import { api } from "./api/client";
import { subscribeToLogs, subscribeToProgress, subscribeToModels, subscribeToCitation, subscribeToJobs } from "./api/sse";
import { logError } from "./utils/logger";

// ---------------------------------------------------------------------------
// Lazy-loaded tab components. Each tab is code-split into its own chunk
// and loaded on first navigation, reducing the initial bundle size. The
// Suspense boundary in the render tree shows a loading placeholder while
// the chunk is being fetched from the server.
// ---------------------------------------------------------------------------

const SearchTab = lazy(() => import("./components/layout/SearchTab"));
const IndexingTab = lazy(() => import("./components/layout/IndexingTab"));
const SourcesTab = lazy(() => import("./components/layout/SourcesTab"));
const SettingsTab = lazy(() => import("./components/layout/SettingsTab"));
const ModelsPanel = lazy(() => import("./components/panels/ModelsPanel"));
const CitationPanel = lazy(() => import("./components/panels/CitationPanel"));
const AnnotationPanel = lazy(() => import("./components/panels/AnnotationPanel"));
const WelcomeDialog = lazy(() => import("./components/ui/WelcomeDialog"));

/**
 * Root application component. Renders the three-zone layout (topbar, body,
 * statusbar) on top of the animated gradient mesh background. The body zone
 * shows one of six tabs controlled by the activeTab state in the store:
 *
 * Workflow tabs (left group): Sources, Indexing, Search, Citations, Annotations
 * Utility tabs (right group): Settings, Models
 *
 * On mount, fetches initial data (health status, session list, config, model
 * catalog) and subscribes to SSE streams for real-time log, progress, model,
 * and citation updates. SSE subscriptions persist across tab switches because
 * they are attached at the root level. Subscriptions are cleaned up on unmount.
 *
 * Tab components are lazy-loaded via SolidJS lazy() so each tab's code is
 * fetched only when the user first navigates to it. A Suspense boundary
 * provides a loading indicator while the chunk downloads.
 *
 * An ErrorBoundary wraps the main content area to catch rendering errors
 * from any tab and display a recovery UI instead of a blank screen.
 *
 * An initializing signal gates the content area behind a loading indicator
 * until the first config/health fetch completes, preventing the user from
 * seeing an empty interface during backend handshake.
 */
const App: Component = () => {
  let cleanupLogs: (() => void) | undefined;
  let cleanupProgress: (() => void) | undefined;
  let cleanupModels: (() => void) | undefined;
  let cleanupCitation: (() => void) | undefined;
  let cleanupJobs: (() => void) | undefined;

  // First-run welcome dialog state. showWelcome becomes true when the backend
  // reports that the .setup_complete marker file does not exist. dataDir holds
  // the absolute path string shown in the dialog for the storage location.
  const [showWelcome, setShowWelcome] = createSignal(false);
  const [welcomeDataDir, setWelcomeDataDir] = createSignal("");

  // Loading indicator state. True until the first config/health fetch round
  // completes, at which point the main content area becomes visible. This
  // prevents the user from seeing an empty tab while the backend handshake
  // is still in progress.
  const [initializing, setInitializing] = createSignal(true);

  onMount(async () => {
    // Trigger all lazy tab chunk downloads immediately so every module is
    // cached in the browser before the user starts any long-running operation.
    // Without preloading, switching tabs during an active fetch-sources
    // operation requires a new HTTP connection to download the chunk. With
    // 6 HTTP/1.1 connections already occupied (5 persistent SSE streams plus
    // 1 fetch POST), the chunk download blocks indefinitely and SolidJS
    // Suspense shows "Loading..." permanently. Preloading ensures all chunks
    // arrive during the idle initialization phase when connections are free.
    void Promise.all([
      import("./components/layout/SearchTab"),
      import("./components/layout/IndexingTab"),
      import("./components/layout/SourcesTab"),
      import("./components/layout/SettingsTab"),
      import("./components/panels/ModelsPanel"),
      import("./components/panels/CitationPanel"),
      import("./components/panels/AnnotationPanel"),
      import("./components/ui/WelcomeDialog"),
    ]);

    // Check whether this is the first launch before fetching any other data.
    // The setup status endpoint is stateless and fast (filesystem stat only).
    // The dialog is shown before the rest of the UI initializes so the user
    // is not confused by an empty interface while deciding on downloads.
    try {
      const setup = await api.setupStatus();
      if (setup.is_first_run) {
        setWelcomeDataDir(setup.data_dir);
        setShowWelcome(true);
      }
    } catch (e) {
      // If the endpoint is unreachable, skip the dialog and proceed normally.
      logError("App.setupStatus", e);
    }

    // Fetch initial health status to populate system state and GPU availability
    try {
      const health = await api.health();
      actions.setHealth(health);
    } catch (e) {
      logError("App.healthCheck", e);
    }

    // Fetch initial session list to populate the session dropdown
    try {
      const sessionRes = await api.sessions();
      actions.setSessions(sessionRes.sessions);
      // Auto-select the first session for IndexingTab (single) and SearchTab (multi)
      if (sessionRes.sessions.length > 0) {
        const first = sessionRes.sessions[0];
        actions.setActiveSession(first.id);
        // Pre-select the first session for multi-search in SearchTab
        // and for citation verification in CitationPanel
        actions.setSearchSessionIds([first.id]);
        actions.setCitationSessionIds([first.id]);
      }
    } catch (e) {
      logError("App.sessionList", e);
    }

    // Fetch config and runtime model info from the server. The config
    // endpoint returns both configuration defaults and the actually loaded
    // model (loaded_model_id, loaded_model_dimension) from the running
    // worker handle, which may differ from the default_model config value.
    try {
      const config = await api.config();
      actions.setStrategy(config.default_strategy);
      actions.setChunkSize(String(config.default_chunk_size));
      actions.setChunkOverlap(String(config.default_overlap));
      actions.setActiveModel(
        config.loaded_model_id || config.default_model,
        config.loaded_model_dimension || 0,
      );
    } catch (e) {
      logError("App.configFetch", e);
    }

    // The initial config/health fetch round is complete. Remove the loading
    // indicator so the main content area becomes visible to the user.
    setInitializing(false);

    // Fetch model catalog to obtain the GPU device name for the StatusBar
    // compute mode indicator. The catalog endpoint invokes nvidia-smi to
    // detect the actual GPU device. Also extracts the loaded reranker
    // model ID from the catalog if a reranker was loaded before page load.
    try {
      const catalog = await api.modelCatalog();
      if (catalog.gpu_name && catalog.gpu_name !== "none") {
        actions.setGpuDeviceName(catalog.gpu_name);
      } else {
        actions.setGpuDeviceName("CPU");
      }
      // Extract loaded reranker model ID from catalog if one is active.
      // This handles the case where a reranker was loaded before the page
      // opened and the SSE reranker_loaded event was already sent.
      const loadedReranker = catalog.reranker_models?.find(
        (m) => m.loaded,
      );
      if (loadedReranker) {
        actions.setRerankerModelId(loadedReranker.model_id);
        actions.setRerankerAvailable(true);
      }
    } catch (e) {
      logError("App.modelCatalog", e);
    }

    // Subscribe to SSE log stream for the log panel in the Settings tab
    cleanupLogs = subscribeToLogs((entry) => {
      actions.appendLogMessage(entry);
    });

    // Subscribe to SSE progress stream for the status bar and session refresh.
    // The phase field from the backend SSE event distinguishes extraction,
    // embedding, and HNSW build stages for phase-specific display.
    cleanupProgress = subscribeToProgress(
      (data) => {
        actions.setIndexProgress({
          phase: (data.phase || "embedding") as "extracting" | "embedding" | "building_index",
          files_total: data.files_total,
          files_done: data.files_done,
          chunks_created: data.chunks_created,
          complete: false,
        });
      },
      () => {
        // Indexing complete: refresh session list and rescan file statuses
        // before clearing the progress indicator. This ensures the session
        // metadata (chunk counts, file counts) and the file status badges
        // (pending -> indexed) are updated atomically. The progress bar
        // remains visible during the refresh so the button transitions
        // directly from "Indexing..." to "All files up to date".
        api.sessions()
          .then((res) => {
            actions.setSessions(res.sessions);
            // Rescan the document file list so status badges update from
            // "pending" to "indexed". The backend checks each file's
            // hash against the session's indexed_file records.
            if (state.folderPath) {
              const sessionId = state.activeSessionId ?? undefined;
              return api.scanDocuments(state.folderPath, sessionId).then((scanRes) => {
                actions.setDocumentFiles(scanRes.files);
              });
            }
          })
          .catch((e) => logError("App.postIndexRefresh", e))
          .finally(() => actions.setIndexProgress(null));
      },
    );

    // Subscribe to SSE model events for download progress and model switches
    cleanupModels = subscribeToModels({
      onModelSwitched: (data) => {
        actions.setActiveModel(data.model_id, data.vector_dimension);
        actions.triggerCatalogRefresh();
        actions.appendLogMessage({
          level: "INFO",
          target: "frontend",
          message: `Model switched to ${data.model_id} (${data.vector_dimension}d)`,
          timestamp: new Date().toISOString(),
        });
      },
      onModelDownloaded: (data) => {
        actions.triggerCatalogRefresh();
        actions.appendLogMessage({
          level: "INFO",
          target: "frontend",
          message: `Model download complete: ${data.model_id}`,
          timestamp: new Date().toISOString(),
        });
      },
      onRerankerLoaded: (data) => {
        actions.setRerankerAvailable(true);
        actions.setRerankerModelId(data.model_id);
        actions.triggerCatalogRefresh();
        actions.appendLogMessage({
          level: "INFO",
          target: "frontend",
          message: `Reranker loaded: ${data.model_id}`,
          timestamp: new Date().toISOString(),
        });
      },
      onOllamaPullProgress: (data) => {
        // Use loose equality (!=) instead of strict (!==) to narrow away
        // both null and undefined in a single check. The SSE type guard
        // types total and completed as `number | null` because Ollama does
        // not report byte counts for all layers (e.g. manifests). Loose
        // equality `!= null` correctly narrows `number | null` to `number`,
        // while also handling zero values (total=0 and completed=0 at the
        // start of a pull are valid progress updates, not skip conditions).
        if (data.total != null && data.completed != null) {
          actions.setOllamaPullProgress({ total: data.total, completed: data.completed });
        }
      },
      onOllamaPullComplete: (data) => {
        actions.setOllamaPullingModel(null);
        actions.setOllamaPullProgress(null);
        actions.appendLogMessage({
          level: "INFO",
          target: "frontend",
          message: `Ollama model pull complete: ${data.model}`,
          timestamp: new Date().toISOString(),
        });
      },
      onOllamaPullError: (data) => {
        actions.setOllamaPullingModel(null);
        actions.setOllamaPullProgress(null);
        actions.appendLogMessage({
          level: "ERROR",
          target: "frontend",
          message: `Ollama model pull failed: ${data.model} - ${data.error}`,
          timestamp: new Date().toISOString(),
        });
      },
    });

    // Subscribe to SSE citation events for live verification progress.
    // The onJobProgress handler also triggers row recovery when the store
    // has a job_id but no rows (happens after page reload while job runs).
    cleanupCitation = subscribeToCitation({
      onRowUpdate: (data) => {
        actions.updateCitationRow(data);
      },
      onReasoningToken: (data) => {
        actions.appendCitationReasoning(data.row_id, data.token);
      },
      onJobProgress: (data) => {
        actions.setCitationProgress(data);
        // If we have progress data but the rows array is empty, the page
        // was likely reloaded. Trigger a one-time row recovery fetch.
        if (state.citationRows.length === 0 && data.rows_total > 0) {
          recoverCitationRows(data.job_id);
        }
        // When the job completes, reconcile all rows with authoritative DB
        // state. SSE events are best-effort; dropped "done" or "error"
        // events leave rows stuck in intermediate phases ("evaluating",
        // "searching"). The reconciliation overwrites stale SSE state with
        // the final phase/verdict from the database.
        if (data.rows_done >= data.rows_total && data.rows_total > 0 && data.job_id) {
          reconcileCitationRows(data.job_id);
        }
      },
    });

    // Subscribe to SSE job update stream for annotation job progress tracking.
    // The Annotations tab stores its job ID in state.annotationJobId. When SSE
    // sends job_update events matching that ID, the progress and state are
    // updated in the store for the AnnotationPanel to render.
    cleanupJobs = subscribeToJobs((data) => {
      if (state.annotationJobId && data.job_id === state.annotationJobId) {
        actions.setAnnotationJobState(data.state);
        actions.setAnnotationProgress(data.progress_done, data.progress_total);
        if (data.error_message) {
          actions.setAnnotationErrorMessage(data.error_message);
        }
      }
    });

    // Recover citation state from localStorage if a previous job exists.
    // This restores the rows table and progress counters so the user sees
    // the verification state after a page reload or panel close/reopen.
    try {
      await recoverCitationState();
    } catch (e) {
      console.error('citation state recovery failed:', e);
    }
  });

  // Recovery guard: prevents multiple concurrent recovery fetches when
  // SSE progress events arrive before the initial recovery completes.
  let citationRecoveryInFlight = false;

  /** Recovers citation job state from localStorage. Checks whether the
   *  stored job_id refers to an active or completed job, and if so,
   *  fetches all rows from the API to rebuild the live results table. */
  const recoverCitationState = async () => {
    const storedJobId = safeLsGet(LS_CITATION_JOB_ID);
    if (!storedJobId) return;

    try {
      const statusRes = await api.citationStatus(storedJobId);
      // Only recover jobs that are running or completed. Failed/queued
      // jobs are not useful to display.
      if (statusRes.job_state !== "running" && statusRes.job_state !== "completed") {
        safeLsRemove(LS_CITATION_JOB_ID);
        return;
      }

      // Set the job_id in the store. This also writes to localStorage
      // which is a no-op since we just read the same value from there.
      actions.setCitationJobId(storedJobId);

      // Restore aggregate progress counters from the status endpoint.
      const verdicts = (typeof statusRes.verdicts === "object" && statusRes.verdicts !== null)
        ? statusRes.verdicts as Record<string, number>
        : {};
      actions.setCitationProgress({
        job_id: storedJobId,
        rows_done: statusRes.done,
        rows_total: statusRes.total,
        verdicts,
      });
      if (statusRes.is_complete) {
        actions.setCitationComplete(true);
      }

      // Fetch all rows to populate the table. The API limit is 500 per
      // request; for larger jobs, multiple pages would be needed.
      await recoverCitationRows(storedJobId);
    } catch {
      // Job no longer exists or API is unreachable. Clear the stored ID
      // so we don't retry on every reload.
      safeLsRemove(LS_CITATION_JOB_ID);
    }
  };

  /** Fetches citation rows from the API and populates the store.
   *  Deduplicates concurrent calls via the citationRecoveryInFlight flag. */
  const recoverCitationRows = async (jobId: string) => {
    if (citationRecoveryInFlight) return;
    if (state.citationRows.length > 0) return;
    citationRecoveryInFlight = true;
    try {
      const rowsRes = await api.citationRows(jobId, 500);
      // rowsRes.rows is already typed as CitationRowDto[] from the API
      // client, so no type assertion is necessary here.
      actions.recoverCitationRows(rowsRes.rows);
    } catch (e) {
      logError("App.citationRowRecovery", e);
    } finally {
      citationRecoveryInFlight = false;
    }
  };

  // Guard: prevents multiple concurrent reconciliation fetches when
  // the SSE completion event fires repeatedly (e.g., one per remaining row).
  let citationReconcileInFlight = false;

  /** Fetches all citation rows from the API and overwrites the store's
   *  phase/verdict data with authoritative DB state. Called when the job
   *  completes to correct rows whose SSE events were dropped. Unlike
   *  recoverCitationRows, this runs even when rows already exist. */
  const reconcileCitationRows = async (jobId: string) => {
    if (citationReconcileInFlight) return;
    citationReconcileInFlight = true;
    try {
      const rowsRes = await api.citationRows(jobId, 500);
      actions.reconcileCitationRows(rowsRes.rows);
    } catch (e) {
      logError("App.citationRowReconcile", e);
    } finally {
      citationReconcileInFlight = false;
    }
  };

  onCleanup(() => {
    cleanupLogs?.();
    cleanupProgress?.();
    cleanupModels?.();
    cleanupCitation?.();
    cleanupJobs?.();
  });

  return (
    <>
      {/* Skip navigation link for keyboard/screen-reader users to bypass
          the topbar tab navigation and jump directly to the active tab content. */}
      <a class="skip-link" href={`#panel-${state.activeTab}`}>Skip to main content</a>

      {/* Animated gradient mesh background behind all content */}
      <div class="gradient-mesh" />

      {/* First-run welcome dialog. Rendered above the app shell. The dialog
          writes the .setup_complete marker file before calling onClose, so
          setShowWelcome(false) is the only action needed here. */}
      <Show when={showWelcome()}>
        <Suspense>
          <WelcomeDialog
            dataDir={welcomeDataDir()}
            onClose={() => setShowWelcome(false)}
          />
        </Suspense>
      </Show>

      <div class="app-shell">
        {/* Screen reader announcement region for tab navigation changes.
            When the active tab changes, assistive technologies announce
            the new tab name to the user without interrupting other content. */}
        <div role="status" aria-live="polite" class="sr-only">
          {state.activeTab + ' tab active'}
        </div>
        <Topbar />

        {/* The tabpanel ID is dynamic and matches the aria-controls value
            of the currently active tab button in Topbar.tsx. Each tab button
            uses aria-controls="panel-{tabName}" (e.g., "panel-sources",
            "panel-search"), so the tabpanel ID follows the same pattern. */}
        <div class="app-body" id={`panel-${state.activeTab}`} role="tabpanel" aria-label={state.activeTab}>
          {/* Loading indicator displayed during initial config/health fetch.
              Prevents the user from seeing an empty tab while the backend
              handshake is still in progress. */}
          <Show when={initializing()}>
            <div class="tab-loading">
              <div class="tab-loading-spinner" />
              <span>Connecting to backend...</span>
            </div>
          </Show>

          {/* Main content area gated behind the initializing signal. The
              ErrorBoundary catches rendering errors from any tab component
              and shows a recovery UI with the error message and a retry
              button. The Suspense boundary provides a loading placeholder
              while lazy-loaded tab chunks are being fetched. */}
          <Show when={!initializing()}>
            <ErrorBoundary fallback={(err, reset) => (
              <div class="error-boundary">
                <h2>Something went wrong</h2>
                <pre>{err.toString()}</pre>
                <button onClick={reset}>Retry</button>
              </div>
            )}>
              <Suspense fallback={<div class="tab-loading">Loading...</div>}>
                <Switch>
                  <Match when={state.activeTab === "sources"}>
                    <SourcesTab />
                  </Match>
                  <Match when={state.activeTab === "indexing"}>
                    <IndexingTab />
                  </Match>
                  <Match when={state.activeTab === "search"}>
                    <SearchTab />
                  </Match>
                  <Match when={state.activeTab === "citations"}>
                    <CitationPanel />
                  </Match>
                  <Match when={state.activeTab === "annotations"}>
                    <AnnotationPanel />
                  </Match>
                  <Match when={state.activeTab === "settings"}>
                    <SettingsTab />
                  </Match>
                  <Match when={state.activeTab === "models"}>
                    <div class="models-tab">
                      <ModelsPanel />
                    </div>
                  </Match>
                </Switch>
              </Suspense>
            </ErrorBoundary>
          </Show>
        </div>

        <StatusBar />
      </div>
    </>
  );
};

export default App;
