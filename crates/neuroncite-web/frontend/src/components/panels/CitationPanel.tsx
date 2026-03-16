/**
 * Citation verification panel for autonomous local LLM-based citation checking.
 *
 * Uses a resizable SplitPane layout consistent with the Search and Sources tabs:
 *   Left panel:  All configuration (file selection, fetch & index, Ollama,
 *                verification mode with presets, action buttons)
 *   Right panel: Live results table with expandable rows, verdict summary,
 *                and progress indicators
 *
 * The panel is decomposed into sub-components in the citation/ subdirectory:
 *   - constants.ts: Shared color maps, types, preset configs
 *   - CitationResultsPanel.tsx: Right-side results table with expandable rows
 *   - VerificationModeSection.tsx: Mode selector with presets and custom sliders
 *   - FetchIndexSection.tsx: Collapsible source indexing configuration
 *
 * The panel communicates with the backend via:
 *   - REST: /citation/create, /citation/{id}/auto-verify, /citation/{id}/status
 *   - REST: /web/ollama/status, /web/ollama/models
 *   - REST: /web/browse/native, /web/browse/native-file, /web/browse
 *   - SSE:  /events/citation (citation_row_update, citation_reasoning_token,
 *           citation_job_progress)
 *
 * State is stored in the central app store (citationRows, ollamaModels, etc.)
 * and updated via SSE event handlers registered in App.tsx.
 */

import { Component, For, Show, createSignal, onCleanup, onMount } from "solid-js";
import { state, actions, safeLsGet, safeLsSet, LS_TEX_PATH, LS_BIB_PATH, LS_SOURCE_PATH } from "../../stores/app";
import { api } from "../../api/client";
import { logWarn } from "../../utils/logger";
import Tip from "../ui/Tip";
import SplitPane from "../layout/SplitPane";
import { createBrowseModal } from "../ui/BrowseModal";
import type { BrowseModalConfig } from "../ui/BrowseModal";
import CitationResultsPanel from "./citation/CitationResultsPanel";
import { useVerificationMode } from "./citation/VerificationModeSection";
import { useFetchIndex } from "./citation/FetchIndexSection";
import { LABEL_STYLE } from "./citation/constants";

const CitationPanel: Component = () => {
  // Local signals for form inputs (not in global store because they are
  // panel-scoped and not needed by other components).
  const [texPath, setTexPath] = createSignal("");
  const [bibPath, setBibPath] = createSignal("");
  const [sourcePath, setSourcePath] = createSignal("");
  // Status display: message with a kind that controls the text color.
  // "info" (secondary text) for progress updates, "error" (magenta) for
  // failures, "success" (secondary text) for completed actions.
  type StatusKind = "info" | "error" | "success";
  const [status, setStatus] = createSignal<{ message: string; kind: StatusKind } | null>(null);

  /** Sets the status message and its display kind. Passing an empty string
   *  clears the status display. */
  const setStatusMessage = (message: string, kind: StatusKind = "info") =>
    setStatus(message ? { message, kind } : null);

  const [isRunning, setIsRunning] = createSignal(false);
  const [annotateAfterExport, setAnnotateAfterExport] = createSignal(true);
  const [isConnecting, setIsConnecting] = createSignal(false);

  // Tracks the completion-polling interval so it can be cleared when the
  // component unmounts. Without this, the setInterval from Step 3 of
  // runVerification would continue polling indefinitely if the user
  // navigates away from the Citation tab while verification is running.
  let completionPollInterval: ReturnType<typeof setInterval> | undefined;
  onCleanup(() => {
    if (completionPollInterval !== undefined) {
      clearInterval(completionPollInterval);
    }
    // Clear status bar progress if the component unmounts while a
    // verification workflow is still running (e.g. user switches tabs).
    if (isRunning()) {
      actions.setTaskProgress(null);
    }
  });
  const [isIndexing, setIsIndexing] = createSignal(false);
  const [expandedRow, setExpandedRow] = createSignal<number | null>(null);

  // Auto-detect notification message for .bib file detection.
  // Displayed as a brief info box when a .bib file is automatically found.
  const [bibAutoMessage, setBibAutoMessage] = createSignal("");

  // ---- Sub-component composables ----

  /** Verification mode composable providing the mode selector UI and
   *  activeParams() accessor for the auto-verify API call. */
  const verifyMode = useVerificationMode();

  /** Fetch & Index composable providing the section toggle state and
   *  collapsible UI for source directory indexing configuration. */
  const fetchIndex = useFetchIndex();

  // ---- Browse modal (shared composable) ----

  const browse = createBrowseModal({
    onSelect: (path, mode) => {
      if (mode === "tex") {
        setTexPathPersist(path);
        if (!bibPath()) tryAutoDetectBib(path);
      } else if (mode === "bib") {
        setBibPathPersist(path);
      } else {
        setSourcePathPersist(path);
      }
    },
  });

  /** Returns the BrowseModalConfig for the current browse mode. */
  const browseConfig = (): BrowseModalConfig => {
    const mode = browse.browseMode();
    if (mode === "tex") {
      return { title: "Select LaTeX File", fileExtension: ".tex", showFolderSelect: false };
    }
    if (mode === "bib") {
      return { title: "Select BibTeX File", fileExtension: ".bib", showFolderSelect: false };
    }
    return { title: "Select Source Directory", fileExtension: "", showFolderSelect: true };
  };

  // ------------------------------------------------------------------
  // Restore persisted form inputs and Ollama state on panel mount.
  // This handles both page-reload and panel-close-reopen scenarios.
  // ------------------------------------------------------------------

  onMount(() => {
    // Restore file paths from localStorage.
    const storedTex = safeLsGet(LS_TEX_PATH);
    const storedBib = safeLsGet(LS_BIB_PATH);
    const storedSource = safeLsGet(LS_SOURCE_PATH);
    if (storedTex) setTexPath(storedTex);
    if (storedBib) setBibPath(storedBib);
    if (storedSource) setSourcePath(storedSource);

    // If an Ollama URL is configured and we have a model selection but
    // the connection status is false (lost after reload), re-test the
    // connection to restore the green status indicator.
    if (state.ollamaUrl && !state.ollamaConnected) {
      testConnection();
    }
  });

  // ------------------------------------------------------------------
  // Persistence helpers: save form inputs to localStorage on change
  // ------------------------------------------------------------------

  const setTexPathPersist = (path: string) => {
    setTexPath(path);
    safeLsSet(LS_TEX_PATH, path);
  };

  const setBibPathPersist = (path: string) => {
    setBibPath(path);
    safeLsSet(LS_BIB_PATH, path);
  };

  const setSourcePathPersist = (path: string) => {
    setSourcePath(path);
    safeLsSet(LS_SOURCE_PATH, path);
  };

  // ------------------------------------------------------------------
  // Auto-detect .bib file in the same directory as the selected .tex
  // ------------------------------------------------------------------

  /** Scans the directory of the given .tex file path for .bib files.
   *  If exactly one .bib file exists, auto-fills the bib path and
   *  shows a brief notification. Does nothing when 0 or 2+ .bib files
   *  are found, leaving the user to select manually. */
  const tryAutoDetectBib = async (texFilePath: string) => {
    const texDir = texFilePath.replace(/[\\/][^\\/]*$/, "");
    if (!texDir) return;
    try {
      const res = await api.browse(texDir);
      const bibFiles = res.entries.filter(
        (e) => e.kind === "file" && e.name.toLowerCase().endsWith(".bib"),
      );
      if (bibFiles.length === 1) {
        const sep = texDir.includes("/") ? "/" : "\\";
        const fullBibPath = `${texDir}${sep}${bibFiles[0].name}`;
        setBibPathPersist(fullBibPath);
        setBibAutoMessage(`Auto-detected: ${bibFiles[0].name}`);
        setTimeout(() => setBibAutoMessage(""), 4000);
      }
    } catch {
      // Directory listing failed; user selects .bib manually.
    }
  };

  // ------------------------------------------------------------------
  // Ollama connection
  // ------------------------------------------------------------------

  /** Tests the Ollama connection and loads the model list on success. */
  const testConnection = async () => {
    setIsConnecting(true);
    setStatusMessage("");
    try {
      const statusRes = await api.ollamaStatus(state.ollamaUrl);
      actions.setOllamaConnected(statusRes.connected);

      if (statusRes.connected) {
        const modelsRes = await api.ollamaModels(state.ollamaUrl);
        actions.setOllamaModels(modelsRes.models);
        if (modelsRes.models.length > 0 && !state.ollamaSelectedModel) {
          actions.setOllamaSelectedModel(modelsRes.models[0].name);
        }
        setStatusMessage(`Connected. ${modelsRes.models.length} model(s) available.`, "success");
      } else {
        setStatusMessage("Connection failed. Is Ollama running?", "error");
      }
    } catch (e) {
      actions.setOllamaConnected(false);
      setStatusMessage(`Connection error: ${e instanceof Error ? e.message : String(e)}`, "error");
    } finally {
      setIsConnecting(false);
    }
  };

  // ------------------------------------------------------------------
  // File/folder browsing via native OS dialogs with BrowseModal fallback
  // ------------------------------------------------------------------

  /** Opens a native OS file dialog for .tex file selection. Falls back
   *  to the custom browser modal if the native dialog is not available. */
  const openTexBrowser = async () => {
    const startPath = texPath().replace(/[\\/][^\\/]*$/, "");
    const selected = await browse.openNativeFileDialog("tex", startPath, "tex");
    if (selected) {
      setTexPathPersist(selected);
      if (!bibPath()) tryAutoDetectBib(selected);
    }
  };

  /** Opens a native OS file dialog for .bib file selection. Falls back
   *  to the custom browser modal if the native dialog is not available. */
  const openBibBrowser = async () => {
    const startPath = bibPath().replace(/[\\/][^\\/]*$/, "");
    const selected = await browse.openNativeFileDialog("bib", startPath);
    if (selected) {
      setBibPathPersist(selected);
    }
  };

  /** Opens a native OS folder dialog for source directory selection.
   *  Falls back to the custom browser modal if native is not available. */
  const openDirBrowser = async () => {
    const selected = await browse.openNativeFolderDialog("folder", sourcePath());
    if (selected) {
      setSourcePathPersist(selected);
    }
  };

  // ------------------------------------------------------------------
  // Source directory indexing with conflict validation
  // ------------------------------------------------------------------

  /** Checks whether any selected session already indexes the same directory
   *  with different settings. Returns a warning message if a conflict is
   *  found, or an empty string if no conflict exists. The chunkStrategy
   *  parameter is the strategy selected in the FetchIndexSection. */
  const checkSessionConflict = (chunkStrategy: string): string => {
    if (!sourcePath()) return "";
    // Normalize the source path for comparison by stripping trailing slashes.
    const normSource = sourcePath().replace(/[\\/]+$/, "").toLowerCase();

    for (const session of state.sessions) {
      if (!state.citationSessionIds.includes(session.id)) continue;
      const normDir = session.directory_path.replace(/[\\/]+$/, "").toLowerCase();
      if (normDir !== normSource) continue;

      // Session indexes the same directory. Check if model or strategy differ.
      const currentModel = state.activeModelId || "";
      if (session.model_name !== currentModel || session.chunk_strategy !== chunkStrategy) {
        const modelShort = session.model_name.split("/").pop() || session.model_name;
        return `Session #${session.id} already indexes this directory with different settings (${modelShort}, ${session.chunk_strategy}). The new index will create an additional session.`;
      }
    }
    return "";
  };

  /** Indexes the source directory as a new session. Runs conflict validation
   *  first and displays a warning if settings differ from existing sessions.
   *  After indexing completes (detected via SSE progress events), the session
   *  list is refreshed and the new session is auto-selected.
   *  Called by the FetchIndexSection sub-component. */
  const indexSourceDirectory = async (strategy: string, chunkSize: string, chunkOverlap: string) => {
    if (!sourcePath()) {
      setStatusMessage("Source directory is required for indexing.", "error");
      return;
    }
    if (!state.activeModelId) {
      setStatusMessage("No embedding model loaded. Load a model in the Indexing tab first.", "error");
      return;
    }

    // Display conflict warning (non-blocking) if applicable.
    const conflict = checkSessionConflict(strategy);
    if (conflict) {
      setStatusMessage(conflict, "error");
    }

    setIsIndexing(true);
    if (!conflict) setStatusMessage("Indexing source directory...");
    try {
      const parsedChunkSize = parseInt(chunkSize, 10);
      const parsedChunkOverlap = parseInt(chunkOverlap, 10);
      const res = await api.startIndex({
        directory: sourcePath(),
        model_name: state.activeModelId,
        chunk_strategy: strategy,
        chunk_size: isNaN(parsedChunkSize) ? undefined : parsedChunkSize,
        chunk_overlap: isNaN(parsedChunkOverlap) ? undefined : parsedChunkOverlap,
      });
      // Auto-add the newly created session to the citation selection.
      if (!state.citationSessionIds.includes(res.session_id)) {
        actions.setCitationSessionIds([...state.citationSessionIds, res.session_id]);
      }
      actions.setActiveSession(res.session_id);
      actions.setIndexProgress({
        phase: "extracting",
        files_total: 0,
        files_done: 0,
        chunks_created: 0,
        complete: false,
      });
      setStatusMessage(`Indexing started (session #${res.session_id}). Wait for completion before creating job.`, "info");
      // Refresh sessions after a delay to pick up the new session.
      setTimeout(async () => {
        try {
          const sessionRes = await api.sessions();
          actions.setSessions(sessionRes.sessions);
        } catch {
          logWarn("CitationPanel.indexSourceDirectory", "Failed to refresh sessions after indexing start.");
        }
      }, 2000);
    } catch (e) {
      setStatusMessage(`Indexing failed: ${e instanceof Error ? e.message : String(e)}`, "error");
    } finally {
      setIsIndexing(false);
    }
  };

  // ------------------------------------------------------------------
  // Full workflow: create -> verify -> wait -> export (single button)
  // ------------------------------------------------------------------

  /** Runs the complete citation verification pipeline as a single action.
   *  Chains: (1) create job from .tex/.bib + sessions, (2) start the
   *  autonomous verification agent, (3) poll for completion via SSE-driven
   *  state.citationComplete, (4) export results with optional annotation.
   *  The isRunning signal gates the button state for the entire duration. */
  const runFullWorkflow = async () => {
    if (!texPath() || !bibPath() || state.citationSessionIds.length === 0) {
      setStatusMessage("Select .tex file, .bib file, and at least one session first.", "error");
      return;
    }
    if (!state.ollamaConnected || !state.ollamaSelectedModel) {
      setStatusMessage("Connect to Ollama and select a model first.", "error");
      return;
    }

    setIsRunning(true);

    // Step 1: Refresh sessions and create the citation job.
    try {
      const sessionRes = await api.sessions();
      actions.setSessions(sessionRes.sessions);
    } catch {
      logWarn("CitationPanel.runFullWorkflow", "Failed to refresh sessions before job creation.");
    }

    setStatusMessage("Creating citation job...");
    actions.setTaskProgress({ label: "Creating citation job...", done: 0, total: 100 });
    let jobId: string;
    try {
      const createRes = await api.citationCreate({
        tex_path: texPath(),
        bib_path: bibPath(),
        session_id: state.citationSessionIds[0],
        session_ids: state.citationSessionIds,
      });
      jobId = createRes.job_id;
      actions.setCitationJobId(jobId);
      setStatusMessage(
        `Job created: ${createRes.total_citations} citations, ${createRes.matched_pdfs} PDFs matched.`,
        "success",
      );
      const rowsRes = await api.citationRows(jobId, 500);
      actions.setCitationRows(rowsRes.rows);
    } catch (e) {
      setStatusMessage(`Job creation failed: ${e instanceof Error ? e.message : String(e)}`, "error");
      actions.setTaskProgress(null);
      setIsRunning(false);
      return;
    }

    // Step 2: Start the autonomous verification agent.
    const params = verifyMode.activeParams();
    setStatusMessage("Starting verification agent...");
    actions.setTaskProgress({
      label: `Verifying (0/${state.citationRowsTotal})`,
      done: 0,
      total: state.citationRowsTotal,
    });
    try {
      await api.citationAutoVerify(jobId, {
        ollama_url: state.ollamaUrl,
        model: state.ollamaSelectedModel,
        temperature: params.temperature,
        max_tokens: params.max_tokens,
        source_directory: fetchIndex.fetchIndexEnabled() ? (sourcePath() || undefined) : undefined,
        fetch_sources: fetchIndex.fetchIndexEnabled() ? true : undefined,
        unpaywall_email: fetchIndex.fetchIndexEnabled() ? (state.unpaywallEmail || undefined) : undefined,
        top_k: params.top_k,
        cross_corpus_queries: params.cross_corpus_queries,
        max_retry_attempts: params.max_retry_attempts,
        min_score: params.min_score,
        rerank: params.rerank,
      });
      setStatusMessage("Verification running. Waiting for completion...");
    } catch (e) {
      setStatusMessage(`Verification start failed: ${e instanceof Error ? e.message : String(e)}`, "error");
      actions.setTaskProgress(null);
      setIsRunning(false);
      return;
    }

    // Step 3: Wait for completion by polling the citationComplete store flag.
    // SSE events update citationRowsDone/citationRowsTotal in real-time.
    // The 500ms interval syncs the StatusBar progress bar with those counters.
    // The interval ID is stored at component scope so onCleanup can clear it
    // if the component unmounts during verification.
    await new Promise<void>((resolve) => {
      completionPollInterval = setInterval(() => {
        if (state.citationRowsTotal > 0) {
          actions.setTaskProgress({
            label: `Verifying (${state.citationRowsDone}/${state.citationRowsTotal})`,
            done: state.citationRowsDone,
            total: state.citationRowsTotal,
          });
        }
        if (state.citationComplete) {
          clearInterval(completionPollInterval!);
          completionPollInterval = undefined;
          resolve();
        }
      }, 500);
    });

    // Step 4: Export results with the annotation flag from the checkbox.
    const texDir = texPath().replace(/[/\\][^/\\]+$/, "");
    setStatusMessage("Exporting results...");
    actions.setTaskProgress({
      label: "Exporting results...",
      done: state.citationRowsTotal,
      total: state.citationRowsTotal,
    });
    try {
      await api.citationExport(jobId, {
        output_directory: texDir + "/neuroncite_output",
        source_directory: sourcePath() || state.folderPath || texDir,
        annotate: annotateAfterExport(),
      });
      const suffix = annotateAfterExport() ? " Annotation job started." : "";
      setStatusMessage(`Export complete. Check the neuroncite_output folder.${suffix}`, "success");
    } catch (e) {
      setStatusMessage(`Export failed: ${e instanceof Error ? e.message : String(e)}`, "error");
    } finally {
      actions.setTaskProgress(null);
      setIsRunning(false);
    }
  };

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  const toggleRow = (rowId: number) => {
    setExpandedRow(expandedRow() === rowId ? null : rowId);
  };

  /** Whether a session has a matching vector dimension with the loaded model.
   *  Sessions with a different dimension cannot be searched because the query
   *  vector would be incompatible. When no model is loaded (dimension === 0),
   *  all sessions are considered compatible. */
  const isDimensionCompatible = (session: { vector_dimension: number }) =>
    state.activeModelDimension === 0 || session.vector_dimension === state.activeModelDimension;

  /** Whether a session's directory matches the configured source path.
   *  Used to visually distinguish sessions that were created via
   *  "Fetch & Index Sources" from regular sessions. */
  const isSourceSession = (session: { directory_path: string }) => {
    if (!fetchIndex.fetchIndexEnabled() || !sourcePath()) return false;
    const normSource = sourcePath().replace(/[\\/]+$/, "").toLowerCase();
    const normDir = session.directory_path.replace(/[\\/]+$/, "").toLowerCase();
    return normDir === normSource;
  };

  // ------------------------------------------------------------------
  // Left panel: all configuration sections
  // ------------------------------------------------------------------

  const leftPanel = () => (
    <div style={{ display: "flex", "flex-direction": "column", gap: "16px", height: "100%", "overflow-y": "auto", "min-height": "0" }}>

      {/* ================================================================
       *  Section 1: File Selection
       * ================================================================ */}
      <div class="glass-card" style={{ padding: "16px" }}>
        <div class="section-title">
          File Selection
          <Tip text="Select the LaTeX manuscript (.tex) and bibliography (.bib) files for citation verification. Choose one or more index sessions containing the cited source PDFs." />
        </div>

        {/* .tex file path */}
        <div style={{ display: "flex", gap: "8px", "align-items": "center", "margin-bottom": "8px" }}>
          <label style={LABEL_STYLE}>
            .tex
            <Tip text="Absolute path to the LaTeX file containing citation commands (\cite, \citep, \citet, etc.). The file is parsed to extract all citation references and their surrounding context." />
          </label>
          <input
            class="input"
            type="text"
            placeholder="/path/to/paper.tex"
            value={texPath()}
            onInput={(e) => setTexPathPersist(e.target.value)}
            style={{ flex: "1" }}
          />
          <button class="btn btn-sm" onClick={openTexBrowser} disabled={browse.dialogPending()}>
            {browse.dialogPending() ? "..." : "Browse"}
          </button>
        </div>

        {/* .bib file path */}
        <div style={{ display: "flex", gap: "8px", "align-items": "center", "margin-bottom": "8px" }}>
          <label style={LABEL_STYLE}>
            .bib
            <Tip text="Absolute path to the BibTeX bibliography file referenced by the LaTeX document. Each entry is matched against indexed PDFs by title and author similarity." />
          </label>
          <input
            class="input"
            type="text"
            placeholder="/path/to/references.bib"
            value={bibPath()}
            onInput={(e) => setBibPathPersist(e.target.value)}
            style={{ flex: "1" }}
          />
          <button class="btn btn-sm" onClick={openBibBrowser} disabled={browse.dialogPending()}>
            {browse.dialogPending() ? "..." : "Browse"}
          </button>
        </div>

        {/* Brief info notification when .bib was auto-detected */}
        <Show when={bibAutoMessage()}>
          <div class="citation-info-toast">
            {bibAutoMessage()}
          </div>
        </Show>

        {/* Multi-select session list */}
        <div>
          <div style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "6px" }}>
            <label style={LABEL_STYLE}>
              Sessions
              <Tip text="Index sessions containing the cited source documents. Multiple sessions can be selected for cross-session search. Each session shows its directory, embedding model, vector dimension, and chunk count." />
            </label>
            {state.citationSessionIds.length > 1 && (
              <span style={{ "font-size": "10px", color: "var(--color-accent-purple)" }}>
                multi ({state.citationSessionIds.length})
              </span>
            )}
            <div style={{ flex: "1" }} />
            <button
              class="btn btn-sm"
              style={{ "font-size": "10px", padding: "2px 6px" }}
              onClick={() => actions.setCitationSessionIds(
                state.sessions.filter((s) => isDimensionCompatible(s)).map((s) => s.id)
              )}
            >
              All
            </button>
            <button
              class="btn btn-sm"
              style={{ "font-size": "10px", padding: "2px 6px" }}
              onClick={() => actions.setCitationSessionIds([])}
            >
              None
            </button>
          </div>
          <Show
            when={state.sessions.length > 0}
            fallback={
              <div style={{ "font-size": "12px", color: "var(--color-text-muted)", padding: "4px 0" }}>
                No sessions. Index a folder first.
              </div>
            }
          >
            <div style={{ "max-height": "120px", "overflow-y": "auto" }}>
              <For each={state.sessions}>
                {(session) => {
                  const compatible = () => isDimensionCompatible(session);
                  const checked = () => state.citationSessionIds.includes(session.id);
                  const isSrc = () => isSourceSession(session);
                  const dirName = () =>
                    session.directory_path.split(/[/\\]/).filter(Boolean).pop() || session.directory_path;
                  const modelShort = () => session.model_name.split("/").pop() || session.model_name;
                  return (
                    <label
                      style={{
                        display: "flex",
                        "align-items": "center",
                        gap: "6px",
                        padding: "3px 2px",
                        "font-size": "12px",
                        cursor: compatible() ? "pointer" : "not-allowed",
                        opacity: compatible() ? "1" : "0.4",
                        color: checked() ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                        "border-left": isSrc() ? "2px solid var(--color-accent-cyan)" : "2px solid transparent",
                        "padding-left": "6px",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={checked()}
                        disabled={!compatible()}
                        onChange={() => actions.toggleCitationSession(session.id)}
                        style={{ "accent-color": isSrc() ? "var(--color-accent-cyan)" : "var(--color-accent-purple)" }}
                      />
                      <span style={{
                        flex: "1",
                        "white-space": "nowrap",
                        overflow: "hidden",
                        "text-overflow": "ellipsis",
                      }}>
                        #{session.id} {dirName()} ({modelShort()}, {session.vector_dimension}d, {session.total_chunks} chunks)
                      </span>
                      <Show when={isSrc()}>
                        <span style={{ "font-size": "9px", color: "var(--color-accent-cyan)", "flex-shrink": "0" }}>
                          [src]
                        </span>
                      </Show>
                      <Show when={!compatible()}>
                        <span style={{ "font-size": "10px", color: "var(--color-accent-magenta)", "flex-shrink": "0" }}>
                          {session.vector_dimension}d
                        </span>
                      </Show>
                    </label>
                  );
                }}
              </For>
            </div>
          </Show>
        </div>
      </div>

      {/* ================================================================
       *  Section 2: Fetch & Index Sources (collapsible sub-component)
       * ================================================================ */}
      <fetchIndex.FetchIndexUI
        sourcePath={sourcePath}
        setSourcePath={setSourcePathPersist}
        onBrowseDir={openDirBrowser}
        dialogPending={browse.dialogPending}
        onIndex={indexSourceDirectory}
        isIndexing={isIndexing}
      />

      {/* ================================================================
       *  Section 3: Ollama Configuration
       * ================================================================ */}
      <div class="glass-card" style={{ padding: "16px" }}>
        <div class="section-title">
          Ollama Configuration
          <Tip text="Connection settings for the local Ollama LLM server that performs the autonomous citation verification. Ollama must be running and have at least one model pulled." />
        </div>

        <div style={{ display: "flex", gap: "8px", "align-items": "center", "margin-bottom": "8px" }}>
          <label style={LABEL_STYLE}>
            URL
            <Tip text="HTTP endpoint of the Ollama server. Default: http://localhost:11434. The server must be accessible from this machine." />
          </label>
          <input
            class="input"
            type="text"
            value={state.ollamaUrl}
            onInput={(e) => actions.setOllamaUrl(e.target.value)}
            style={{ flex: "1" }}
          />
          <button
            class="btn btn-sm"
            onClick={testConnection}
            disabled={isConnecting()}
          >
            {isConnecting() ? "Testing..." : "Connect"}
          </button>
          <div
            style={{
              width: "10px",
              height: "10px",
              "border-radius": "50%",
              background: state.ollamaConnected ? "var(--color-accent-cyan)" : "var(--color-status-error)",
              "box-shadow": state.ollamaConnected
                ? "0 0 8px rgba(34, 211, 238, 0.6)"
                : "0 0 8px rgba(239, 68, 68, 0.4)",
            }}
          />
        </div>

        <Show when={state.ollamaModels.length > 0}>
          <div style={{ display: "flex", gap: "8px", "align-items": "center" }}>
            <label style={LABEL_STYLE}>
              Model
              <Tip text="Ollama model used for citation verification reasoning. Larger models produce more accurate verdicts but are slower. Models with 7B+ parameters are recommended." />
            </label>
            <select
              class="select"
              style={{ flex: "1" }}
              value={state.ollamaSelectedModel}
              onChange={(e) => actions.setOllamaSelectedModel(e.target.value)}
            >
              <For each={state.ollamaModels}>
                {(model) => (
                  <option value={model.name}>
                    {model.name}
                    {model.parameter_size ? ` (${model.parameter_size})` : ""}
                    {" "}- {formatSize(model.size)}
                  </option>
                )}
              </For>
            </select>
          </div>
        </Show>
      </div>

      {/* ================================================================
       *  Section 4: Verification Mode (sub-component)
       * ================================================================ */}
      <verifyMode.VerificationModeUI />

      {/* ================================================================
       *  Section 5: Controls
       * ================================================================ */}
      <div class="glass-card" style={{ padding: "16px" }}>
        <div class="section-title">
          Controls
          <Tip text="Runs the complete pipeline: creates the citation job, starts the autonomous verification agent, waits for completion, and exports results. The annotation checkbox controls whether source PDFs are highlighted after export." />
        </div>

        {/* Annotation toggle above the start button. */}
        <label style={{
          display: "flex",
          "align-items": "center",
          gap: "8px",
          "margin-bottom": "10px",
          "font-size": "12px",
          color: "var(--color-text-secondary)",
          cursor: "pointer",
        }}>
          <input
            type="checkbox"
            checked={annotateAfterExport()}
            onChange={(e) => setAnnotateAfterExport(e.target.checked)}
            style={{ "accent-color": "var(--color-accent-purple)" }}
          />
          Annotate PDFs after export
          <Tip text="When checked, the export step creates an annotation job that highlights verified citation quotes in the source PDFs. Annotated PDFs are saved to the annotated_pdfs/ subfolder within the output directory." />
        </label>

        <button
          class="btn btn-primary"
          style={{ width: "100%" }}
          onClick={runFullWorkflow}
          disabled={
            isRunning()
            || !texPath()
            || !bibPath()
            || state.citationSessionIds.length === 0
            || !state.ollamaConnected
            || !state.ollamaSelectedModel
            || state.indexProgress !== null
          }
        >
          {isRunning()
            ? state.citationComplete
              ? "Exporting..."
              : state.citationJobId
                ? `Verifying... ${state.citationRowsDone}/${state.citationRowsTotal}`
                : "Creating..."
            : "Start"}
        </button>

        {/* Status message: error kind uses magenta, info/success use secondary text */}
        <Show when={status()}>
          <div style={{
            "margin-top": "8px",
            "font-size": "12px",
            color: status()?.kind === "error"
              ? "var(--color-accent-magenta)"
              : "var(--color-text-secondary)",
          }}>
            {status()?.message}
          </div>
        </Show>
      </div>
    </div>
  );

  // ------------------------------------------------------------------
  // Right panel: results sub-component
  // ------------------------------------------------------------------

  const rightPanel = () => (
    <CitationResultsPanel
      expandedRow={expandedRow}
      toggleRow={toggleRow}
    />
  );

  // ------------------------------------------------------------------
  // Main render: SplitPane + browse modal
  // ------------------------------------------------------------------

  return (
    <>
      <SplitPane
        left={leftPanel()}
        right={rightPanel()}
        leftWidth={state.citationLeftPanelWidth}
        onResize={(w) => actions.setCitationLeftPanelWidth(w)}
        minLeft={240}
        minRight={300}
      />

      {/* Shared browse modal rendered via the BrowseModal composable */}
      <browse.BrowseModalUI config={browseConfig()} />
    </>
  );
};

export default CitationPanel;
