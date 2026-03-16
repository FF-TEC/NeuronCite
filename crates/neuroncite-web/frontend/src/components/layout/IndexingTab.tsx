import { Component, For, Show, createSignal } from "solid-js";
import { state, actions } from "../../stores/app";
import { api } from "../../api/client";
import { logWarn } from "../../utils/logger";
import { normalizeWindowsPath } from "../../utils/path";
import { clickableProps } from "../../utils/a11y";
import type { SessionDto } from "../../api/types";
import SplitPane from "./SplitPane";
import Tip from "../ui/Tip";
import { createBrowseModal } from "../ui/BrowseModal";
import type { BrowseModalConfig } from "../ui/BrowseModal";

/**
 * Indexing tab for managing index sessions and document ingestion.
 * Provides session management (list with expandable detail rows, select,
 * delete), document folder selection via native OS dialog with fallback
 * directory browser, indexing configuration (model, strategy, chunk
 * parameters), and a file status overview with expandable per-file metadata.
 * Supports all indexable formats (PDF, HTML, and future types).
 *
 * Layout: resizable horizontal SplitPane. Left panel contains session list
 * and indexing controls; right panel displays the file status list for the
 * selected folder/session. The divider is draggable and the left panel
 * width is persisted to localStorage.
 */
const IndexingTab: Component = () => {
  // ---- Indexing state ----
  const [indexing, setIndexing] = createSignal(false);
  const [indexError, setIndexError] = createSignal<string | null>(null);

  // ---- Session delete confirmation ----
  const [confirmDeleteId, setConfirmDeleteId] = createSignal<number | null>(null);

  // ---- Expanded session detail: tracks the session ID of the currently
  //      expanded session row, or null when none is expanded. ----
  const [expandedSessionId, setExpandedSessionId] = createSignal<number | null>(null);

  // ---- Expanded file detail: tracks the index of the currently expanded
  //      file entry in the documentFiles list, or null when none is expanded. ----
  const [expandedFileIdx, setExpandedFileIdx] = createSignal<number | null>(null);

  // ---- Browse modal (shared composable) ----

  const browse = createBrowseModal({
    onSelect: (path) => {
      actions.setFolderPath(path);
      scanDocuments(path);
    },
  });

  /** Configuration for the browse modal. IndexingTab only uses folder mode. */
  const browseConfig: BrowseModalConfig = {
    title: "Select Document Folder",
    fileExtension: "",
    showFolderSelect: true,
  };

  /** Opens a native OS folder selection dialog via the backend endpoint.
   *  Prevents concurrent dialogs via the dialogPending guard signal.
   *  If the native dialog fails (headless environment, no display), falls
   *  back to the custom directory browser modal. */
  const openNativeDialog = async () => {
    const selected = await browse.openNativeFolderDialog("folder", state.folderPath || "");
    if (selected) {
      actions.setFolderPath(selected);
      scanDocuments(selected);
    }
  };

  /** Scans the selected folder for indexable document files and updates the
   *  store. Discovers all supported formats (PDF, HTML, and future types).
   *  When an active session exists, passes the session_id so the backend
   *  enriches each entry with its indexing status (indexed/outdated/pending). */
  const scanDocuments = async (path: string) => {
    if (!path) return;
    actions.setFileBrowserScanning(true);
    try {
      const sessionId = state.activeSessionId ?? undefined;
      const res = await api.scanDocuments(path, sessionId);
      actions.setDocumentFiles(res.files);
    } catch (e) {
      logWarn("IndexingTab.scanDocuments", `Document scan failed for path "${path}": ${e instanceof Error ? e.message : String(e)}`);
      actions.setDocumentFiles([]);
    } finally {
      actions.setFileBrowserScanning(false);
    }
  };

  /** Starts the indexing process for the selected folder and session parameters.
   *  Sets the active session from the API response so subsequent searches
   *  target the correct session. */
  const startIndexing = async () => {
    if (!state.folderPath || !state.activeModelId) return;
    setIndexing(true);
    setIndexError(null);
    try {
      const chunkSize = parseInt(state.chunkSize, 10);
      const chunkOverlap = parseInt(state.chunkOverlap, 10);
      const res = await api.startIndex({
        directory: state.folderPath,
        model_name: state.activeModelId,
        chunk_strategy: state.selectedStrategy,
        chunk_size: isNaN(chunkSize) ? undefined : chunkSize,
        chunk_overlap: isNaN(chunkOverlap) ? undefined : chunkOverlap,
      });
      actions.setActiveSession(res.session_id);
      // Auto-add the new session to the search and citation selections
      if (!state.searchSessionIds.includes(res.session_id)) {
        actions.setSearchSessionIds([...state.searchSessionIds, res.session_id]);
      }
      if (!state.citationSessionIds.includes(res.session_id)) {
        actions.setCitationSessionIds([...state.citationSessionIds, res.session_id]);
      }
      // Set initial progress state so the status bar shows "Indexing..."
      // before the first SSE progress event arrives.
      actions.setIndexProgress({
        phase: "extracting",
        files_total: 0,
        files_done: 0,
        chunks_created: 0,
        complete: false,
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setIndexError(msg);
      console.error("Indexing failed:", e);
    } finally {
      setIndexing(false);
    }
  };

  /** Deletes an index session and refreshes the session list. Clears the
   *  active session if the deleted session was selected. Also removes the
   *  session from the search selection. */
  const deleteSession = async (sessionId: number) => {
    try {
      await api.deleteSession(sessionId);
      const sessionRes = await api.sessions();
      actions.setSessions(sessionRes.sessions);
      // Remove the deleted session from search and citation selections
      actions.setSearchSessionIds(state.searchSessionIds.filter((id) => id !== sessionId));
      actions.setCitationSessionIds(state.citationSessionIds.filter((id) => id !== sessionId));
      if (state.activeSessionId === sessionId) {
        actions.setActiveSession(null);
        actions.setDocumentFiles([]);
      }
    } catch (e) {
      console.error("Session deletion failed:", e);
    }
    setConfirmDeleteId(null);
    setExpandedSessionId(null);
  };

  /** Handles clicking a session row to select it. Updates the store and
   *  triggers a document rescan for the selected session's directory. Also
   *  ensures the session is included in the search selection. */
  const selectSession = async (sessionId: number) => {
    actions.setActiveSession(sessionId);
    // Add this session to the search and citation selections if not present
    if (!state.searchSessionIds.includes(sessionId)) {
      actions.setSearchSessionIds([...state.searchSessionIds, sessionId]);
    }
    if (!state.citationSessionIds.includes(sessionId)) {
      actions.setCitationSessionIds([...state.citationSessionIds, sessionId]);
    }
    const session = state.sessions.find((s) => s.id === sessionId);
    if (session) {
      if (session.directory_path) {
        actions.setFolderPath(session.directory_path);
        scanDocuments(session.directory_path);
      }
    }
  };

  /** Computes file status counts from the backend-provided status field. */
  const fileStats = () => {
    const files = state.documentFiles;
    const total = files.length;
    let indexed = 0;
    let outdated = 0;
    let pending = 0;
    for (const f of files) {
      if (f.status === "indexed") indexed++;
      else if (f.status === "outdated") outdated++;
      else pending++;
    }
    return { total, indexed, outdated, pending };
  };

  /** Computes per-format file counts from the document list. Returns a
   *  Record mapping file type identifiers ("pdf", "html") to their count.
   *  Adding a format in the backend produces a new key automatically. */
  const typeCounts = () => {
    const counts: Record<string, number> = {};
    for (const f of state.documentFiles) {
      counts[f.file_type] = (counts[f.file_type] || 0) + 1;
    }
    return counts;
  };

  /** Maps canonical file type identifiers to CSS color variables for
   *  format badges. Adding a format requires one case line here. */
  const fileTypeColor = (ft: string) => {
    switch (ft) {
      case "pdf": return "var(--color-accent-cyan)";
      case "html": return "var(--color-accent-purple)";
      default: return "var(--color-text-muted)";
    }
  };

  /** Determines the indexing status for the current folder/session combination.
   *  Returns "up_to_date" when the active session matches the folder AND the
   *  current chunk configuration, and all scanned files are indexed. Returns
   *  "needs_update" with counts when some files are pending or outdated.
   *  Returns "no_session" when no matching session exists for the selected
   *  folder or when the chunk settings in the UI differ from the active
   *  session (indicating the user wants a new index with different parameters).
   *
   *  Path comparison uses normalizeWindowsPath to strip extended-length prefixes
   *  (\\?\, \\.\, \\?\UNC\) that the Win32 API may insert for paths exceeding
   *  MAX_PATH. This ensures the folder path from the file picker matches the
   *  session directory stored by the Rust backend.
   *
   *  Chunk setting comparison mirrors the backend find_session() logic in
   *  neuroncite-store/src/repo/session.rs which checks directory_path,
   *  model_name, chunk_strategy, chunk_size, chunk_overlap, and max_words. */
  const indexStatus = () => {
    const stats = fileStats();
    if (stats.total === 0) return { kind: "no_files" as const };

    const session = state.sessions.find((s) => s.id === state.activeSessionId);
    if (!session || !state.folderPath) return { kind: "no_session" as const };

    // Directory path comparison with Windows extended-length prefix normalization
    const pathMatch = normalizeWindowsPath(session.directory_path).toLowerCase()
      === normalizeWindowsPath(state.folderPath).toLowerCase();
    if (!pathMatch) return { kind: "no_session" as const };

    // Chunk strategy comparison: different strategies produce fundamentally
    // different chunk boundaries, so a strategy change requires a new session.
    const uiStrategy = state.selectedStrategy;
    if (session.chunk_strategy !== uiStrategy) return { kind: "no_session" as const };

    // Strategy-specific parameter comparison. Each branch checks the parameters
    // that are relevant to the active strategy, matching the backend IndexConfig
    // construction in neuroncite-api/src/handlers/index.rs.
    const uiSize = parseInt(state.chunkSize, 10);
    if (uiStrategy === "token" || uiStrategy === "word") {
      if (!isNaN(uiSize) && session.chunk_size !== null && session.chunk_size !== uiSize) {
        return { kind: "no_session" as const };
      }
      const uiOverlap = parseInt(state.chunkOverlap, 10);
      if (!isNaN(uiOverlap) && session.chunk_overlap !== null && session.chunk_overlap !== uiOverlap) {
        return { kind: "no_session" as const };
      }
    } else if (uiStrategy === "sentence") {
      // Sentence strategy maps the UI "Max Words" input to the max_words field.
      // The chunkSize state variable holds the max_words value for this strategy.
      if (!isNaN(uiSize) && session.max_words !== null && session.max_words !== uiSize) {
        return { kind: "no_session" as const };
      }
    }
    // "page" strategy has no configurable parameters; path + strategy match suffices.

    if (stats.pending === 0 && stats.outdated === 0) {
      return { kind: "up_to_date" as const };
    }
    return { kind: "needs_update" as const, pending: stats.pending, outdated: stats.outdated };
  };

  /** Whether the chunk size/overlap inputs should be visible. */
  const showChunkParams = () =>
    state.selectedStrategy === "token" || state.selectedStrategy === "word";

  /** Whether the max_words input should be shown for sentence strategy. */
  const showMaxWords = () => state.selectedStrategy === "sentence";

  /** Formats a UNIX timestamp (seconds since epoch) to a locale date string. */
  const formatDate = (ts: number) =>
    ts > 0 ? new Date(ts * 1000).toLocaleString() : "unknown";

  /** Formats byte count to a human-readable size string (KB/MB/GB). */
  const formatBytes = (bytes: number) => {
    if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(1)} GB`;
    if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
    return `${(bytes / 1024).toFixed(0)} KB`;
  };

  // ---- Left panel content: session list + indexing controls ----
  const leftPanel = () => (
    <div style={{ display: "flex", "flex-direction": "column", gap: "16px", height: "100%", "overflow-y": "auto", "min-height": "0" }}>

      {/* Session Management: clickable rows with expandable detail panels */}
      <div class="glass-card" style={{ padding: "16px" }}>
        <div class="section-title">
          Sessions
          <Tip text="Each session stores the vector index for one folder. It records which embedding model, chunking strategy, and parameters were used during indexing. Multiple sessions can coexist for different folders or configurations. Click a row to select it and expand its full metadata." />
        </div>
        <Show
          when={state.sessions.length > 0}
          fallback={
            <div style={{ "font-size": "13px", color: "var(--color-text-muted)", padding: "12px 0" }}>
              No sessions yet. Index a folder to create one.
            </div>
          }
        >
          <div style={{ "max-height": "50vh", "overflow-y": "auto" }}>
            <For each={state.sessions}>
              {(session) => {
                const isActive = () => state.activeSessionId === session.id;
                const isExpanded = () => expandedSessionId() === session.id;
                const dirName = () => session.directory_path.split(/[/\\]/).filter(Boolean).pop() || session.directory_path;
                const modelShort = () => session.model_name.split("/").pop() || session.model_name;

                /** Click handler shared between onClick and keyboard activation
                 *  via clickableProps. Selects the session and toggles expansion. */
                const handleSessionClick = () => {
                  selectSession(session.id);
                  setExpandedSessionId(isExpanded() ? null : session.id);
                };

                return (
                  <div style={{ "border-bottom": "1px solid rgba(255, 255, 255, 0.03)" }}>
                    {/* Session summary row: click to select and expand.
                     *  clickableProps adds role="button", tabIndex=0, and keyboard
                     *  handlers so the row is accessible via Tab + Enter/Space. */}
                    <div
                      style={{
                        display: "flex",
                        "align-items": "center",
                        gap: "8px",
                        padding: "8px 6px",
                        cursor: "pointer",
                        background: isActive() ? "rgba(168, 85, 247, 0.08)" : "transparent",
                        "border-radius": "4px",
                        transition: "background 0.15s",
                      }}
                      onClick={handleSessionClick}
                      {...clickableProps(handleSessionClick)}
                    >
                      {/* Active indicator dot */}
                      <div style={{
                        width: "6px",
                        height: "6px",
                        "border-radius": "50%",
                        background: isActive() ? "var(--color-accent-purple)" : "transparent",
                        "flex-shrink": "0",
                      }} />

                      <span style={{
                        "font-size": "12px",
                        color: isActive() ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                        flex: "1",
                        "min-width": "0",
                        overflow: "hidden",
                        "text-overflow": "ellipsis",
                        "white-space": "nowrap",
                      }}
                        title={session.directory_path}
                      >
                        #{session.id} {dirName()}
                      </span>

                      <span style={{
                        "font-size": "11px",
                        color: "var(--color-text-muted)",
                        "flex-shrink": "0",
                      }}>
                        {session.total_chunks} chunks
                      </span>

                      {/* Expand/collapse chevron */}
                      <span style={{
                        "font-size": "10px",
                        color: "var(--color-text-muted)",
                        "flex-shrink": "0",
                        transform: isExpanded() ? "rotate(180deg)" : "rotate(0deg)",
                        transition: "transform 0.15s",
                      }}>
                        &#9660;
                      </span>
                    </div>

                    {/* Expanded detail panel: full session metadata */}
                    <Show when={isExpanded()}>
                      <SessionDetailPanel session={session} isActive={isActive()} onDelete={deleteSession} confirmDeleteId={confirmDeleteId} setConfirmDeleteId={setConfirmDeleteId} />
                    </Show>
                  </div>
                );
              }}
            </For>
          </div>
        </Show>
      </div>

      {/* Document Folder Selection */}
      <div class="glass-card" style={{ padding: "16px" }}>
        <div class="section-title">
          Document Folder
          <Tip text="Select the local directory containing documents to index. All supported files (PDF, HTML, and other indexable formats) in the folder and its subfolders are discovered automatically. The folder path is compared against existing sessions to detect whether a matching index already exists." />
        </div>
        <div class="row">
          <input
            class="input"
            type="text"
            placeholder="Select a folder..."
            readonly
            value={state.folderPath}
            style={{ flex: "1" }}
          />
          <button class="btn btn-sm" onClick={openNativeDialog} disabled={browse.dialogPending()}>
            {browse.dialogPending() ? "..." : "Browse"}
          </button>
        </div>
      </div>

      {/* Indexing Configuration */}
      <div class="glass-card" style={{ padding: "16px" }}>
        <div class="section-title">
          Indexing
          <Tip text="Configure and start the document indexing pipeline. Text is extracted from each document, split into chunks according to the selected strategy, converted to vector embeddings by the loaded model, and stored in a searchable HNSW index." />
        </div>

        {/* Model */}
        <div class="row-between" style={{ "margin-bottom": "12px" }}>
          <span style={{ "font-size": "13px", color: "var(--color-text-secondary)", display: "inline-flex", "align-items": "center" }}>
            {state.activeModelId
              ? `${state.activeModelId.split("/").pop()} (${state.activeModelDimension}d)`
              : "No model loaded"}
            <Tip text="The embedding model converts text passages into fixed-size numerical vectors for semantic similarity search. Different models vary in vector dimensions, language coverage, and accuracy. The model must be loaded before indexing can start. Use 'Manage' to download, load, or switch models." />
          </span>
          <button class="btn btn-sm" onClick={() => actions.setActiveTab("models")}>
            Manage
          </button>
        </div>

        {/* Strategy */}
        <div class="label">
          Strategy
          <Tip text="Controls how extracted document text is divided into searchable chunks. Token: fixed-size windows counted in subword tokens, aligned to the model tokenizer. Word: fixed-size windows counted in whitespace-separated words. Sentence: groups consecutive sentences up to a word limit, preserving natural sentence boundaries. Page: one chunk per document page, no splitting." />
        </div>
        <select
          class="select"
          style={{ "margin-bottom": "12px" }}
          value={state.selectedStrategy}
          onChange={(e) => actions.setStrategy(e.target.value)}
        >
          <option value="token">Token</option>
          <option value="word">Word</option>
          <option value="sentence">Sentence</option>
          <option value="page">Page</option>
        </select>

        {/* Chunk Parameters (token/word strategies) */}
        <Show when={showChunkParams()}>
          <div class="row" style={{ gap: "8px", "margin-bottom": "12px" }}>
            <div style={{ flex: "1" }}>
              <div class="label">
                Size
                <Tip text="The number of tokens (Token strategy) or words (Word strategy) per chunk. Larger values retain more context per chunk but reduce search granularity. Smaller values improve precision but may fragment sentences. Default: 256." />
              </div>
              <input
                class="input"
                type="text"
                value={state.chunkSize}
                onInput={(e) => actions.setChunkSize(e.target.value)}
              />
            </div>
            <div style={{ flex: "1" }}>
              <div class="label">
                Overlap
                <Tip text="The number of tokens or words shared between consecutive chunks. Overlap ensures that text near chunk boundaries appears in at least two chunks, preventing information loss at split points. Higher overlap increases index size but improves recall. Default: 32." />
              </div>
              <input
                class="input"
                type="text"
                value={state.chunkOverlap}
                onInput={(e) => actions.setChunkOverlap(e.target.value)}
              />
            </div>
          </div>
        </Show>

        {/* Max Words (sentence strategy) */}
        <Show when={showMaxWords()}>
          <div style={{ "margin-bottom": "12px" }}>
            <div class="label">
              Max Words
              <Tip text="Maximum word count per chunk when using the Sentence strategy. Consecutive sentences are grouped until adding the next sentence would exceed this limit, then a new chunk starts. This preserves natural sentence boundaries while controlling chunk size." />
            </div>
            <input
              class="input"
              type="text"
              value={state.chunkSize}
              onInput={(e) => actions.setChunkSize(e.target.value)}
            />
          </div>
        </Show>

        {/* Index Button */}
        <Show
          when={indexStatus().kind !== "up_to_date"}
          fallback={
            <div style={{
              width: "100%",
              padding: "8px 12px",
              "text-align": "center",
              "font-size": "12px",
              color: "var(--color-status-indexed)",
              background: "rgba(34, 211, 238, 0.06)",
              border: "1px solid rgba(34, 211, 238, 0.15)",
              "border-radius": "var(--radius-sm)",
            }}>
              All files up to date
            </div>
          }
        >
          <button
            class="btn btn-primary"
            disabled={!state.folderPath || !state.activeModelId || indexing() || state.indexProgress !== null}
            style={{ width: "100%" }}
            onClick={startIndexing}
          >
            {indexing() || state.indexProgress !== null
              ? "Indexing..."
              : indexStatus().kind === "needs_update"
                ? (() => {
                    const s = indexStatus();
                    if (s.kind !== "needs_update") return "Start Indexing";
                    const parts: string[] = [];
                    if (s.pending > 0) parts.push(`${s.pending} new`);
                    if (s.outdated > 0) parts.push(`${s.outdated} changed`);
                    return `Index ${parts.join(" + ")} files`;
                  })()
                : "Start Indexing"
            }
          </button>
        </Show>
        <Show when={indexError()}>
          <div style={{
            "font-size": "11px",
            color: "var(--color-accent-magenta)",
            "margin-top": "4px",
            "line-height": "1.4",
          }}>
            {indexError()}
          </div>
        </Show>
      </div>
    </div>
  );

  // ---- Right panel content: file status list ----
  const rightPanel = () => (
    <div style={{ display: "flex", "flex-direction": "column", height: "100%", "min-height": "0", "min-width": "0" }}>
      <div class="glass-card" style={{ padding: "16px", flex: "1", display: "flex", "flex-direction": "column", "min-height": "0", "min-width": "0", overflow: "hidden" }}>
        <div class="section-title" style={{ "flex-shrink": "0" }}>
          Files
          <Tip text="Lists all indexable documents found in the selected folder with their indexing status relative to the active session. Indexed: file content matches the stored version, no re-indexing needed. Outdated: file has been modified since last indexing, re-indexing will update chunks and embeddings. Pending: file has not been indexed in the active session. The list refreshes automatically when indexing completes." />
        </div>

        <Show
          when={state.documentFiles.length > 0}
          fallback={
            <div style={{ "font-size": "13px", color: "var(--color-text-muted)", padding: "12px 0" }}>
              {state.folderPath ? "No documents found in selected folder" : "Select a folder to scan for documents"}
            </div>
          }
        >
          {/* Summary counts with per-status tooltips and per-type breakdown */}
          <div class="row" style={{ gap: "16px", "margin-bottom": "16px", "flex-wrap": "wrap", "flex-shrink": "0", "min-width": "0", overflow: "hidden" }}>
            <div style={{ "font-size": "13px", display: "inline-flex", "align-items": "center", gap: "6px" }}>
              <span><span style={{ "font-weight": "600" }}>{fileStats().total}</span>&nbsp;Documents</span>
              {/* Per-type breakdown: shows count per format in its assigned color */}
              <span style={{ "font-size": "11px", color: "var(--color-text-muted)" }}>
                ({Object.entries(typeCounts()).map(([ft, count], i) =>
                  <><Show when={i > 0}>, </Show><span style={{ color: fileTypeColor(ft) }}>{count} {ft.toUpperCase()}</span></>
                )})
              </span>
              <Tip text="Total number of indexable documents discovered in the selected folder and its subfolders. The per-type breakdown shows how many files of each format were found." />
            </div>
            <div style={{ "font-size": "13px", color: "var(--color-status-indexed)", display: "inline-flex", "align-items": "center" }}>
              <span style={{ "font-weight": "600" }}>{fileStats().indexed}</span>&nbsp;indexed
              <Tip text="Files whose content and modification date match the stored version in the active session. These files do not need re-indexing." />
            </div>
            <div style={{ "font-size": "13px", color: "var(--color-status-outdated)", display: "inline-flex", "align-items": "center" }}>
              <span style={{ "font-weight": "600" }}>{fileStats().outdated}</span>&nbsp;outdated
              <Tip text="Files that have been modified on disk since they were last indexed. Re-indexing will extract the updated content and replace the old chunks and embeddings." />
            </div>
            <div style={{ "font-size": "13px", color: "var(--color-status-pending)", display: "inline-flex", "align-items": "center" }}>
              <span style={{ "font-weight": "600" }}>{fileStats().pending}</span>&nbsp;pending
              <Tip text="Files that exist in the folder but have not been indexed in the active session yet. These will be processed on the next indexing run." />
            </div>
          </div>

          {/* File list: flex-grow fills remaining height, overflow-y scrolls */}
          <div style={{ flex: "1", "overflow-y": "auto", "min-height": "0" }}>
            <For each={state.documentFiles}>
              {(file, idx) => {
                const statusColor = () =>
                  file.status === "indexed" ? "var(--color-status-indexed)"
                  : file.status === "outdated" ? "var(--color-status-outdated)"
                  : "var(--color-text-secondary)";
                const statusLabel = () =>
                  file.status === "indexed" ? "indexed"
                  : file.status === "outdated" ? "outdated"
                  : "pending";
                const isExpanded = () => expandedFileIdx() === idx();
                const sizeKb = () => (file.size / 1024).toFixed(0);
                const sizeMb = () => (file.size / (1024 * 1024)).toFixed(1);
                const dateStr = () => file.mtime > 0
                  ? new Date(file.mtime * 1000).toLocaleDateString()
                  : "unknown";

                /** Click handler for expanding/collapsing the file detail panel. */
                const handleFileClick = () => setExpandedFileIdx(isExpanded() ? null : idx());

                return (
                  <div>
                    {/* File row: clickable to expand/collapse detail panel.
                     *  clickableProps adds role="button", tabIndex=0, and keyboard
                     *  handlers so the row is accessible via Tab + Enter/Space. */}
                    <div
                      style={{
                        display: "flex",
                        "align-items": "center",
                        "justify-content": "space-between",
                        padding: "6px 12px 6px 0",
                        "border-bottom": isExpanded() ? "none" : "1px solid rgba(255, 255, 255, 0.03)",
                        cursor: "pointer",
                      }}
                      title={file.path}
                      onClick={handleFileClick}
                      {...clickableProps(handleFileClick)}
                    >
                      <span
                        style={{
                          "font-size": "12px",
                          color: isExpanded() ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                          overflow: "hidden",
                          "text-overflow": "ellipsis",
                          "white-space": "nowrap",
                          flex: "1",
                          "min-width": "0",
                        }}
                      >
                        {file.subfolder ? `${file.subfolder}/` : ""}{file.name}
                      </span>
                      {/* Format type badge: shows uppercase file type in its assigned color */}
                      <span
                        class="badge"
                        style={{
                          color: fileTypeColor(file.file_type),
                          background: `color-mix(in srgb, ${fileTypeColor(file.file_type)} 15%, transparent)`,
                          "flex-shrink": "0",
                          "margin-left": "8px",
                          "font-size": "10px",
                          "text-transform": "uppercase",
                        }}
                      >
                        {file.file_type}
                      </span>
                      {/* Indexing status badge */}
                      <span
                        class="badge"
                        style={{
                          color: statusColor(),
                          background: `color-mix(in srgb, ${statusColor()} 15%, transparent)`,
                          "flex-shrink": "0",
                          "margin-left": "4px",
                        }}
                      >
                        {statusLabel()}
                      </span>

                      {/* Expand/collapse chevron matching the session row pattern */}
                      <span style={{
                        "font-size": "10px",
                        color: "var(--color-text-muted)",
                        "flex-shrink": "0",
                        transform: isExpanded() ? "rotate(180deg)" : "rotate(0deg)",
                        transition: "transform 0.15s",
                        "margin-left": "4px",
                      }}>
                        &#9660;
                      </span>
                    </div>

                    {/* Expanded detail panel: shows file metadata and index statistics */}
                    <Show when={isExpanded()}>
                      <div style={{
                        padding: "8px 12px 12px",
                        "margin-bottom": "4px",
                        background: "rgba(168, 85, 247, 0.04)",
                        "border-left": "2px solid var(--color-accent-purple)",
                        "border-bottom": "1px solid rgba(255, 255, 255, 0.03)",
                        "font-size": "11px",
                        color: "var(--color-text-secondary)",
                        display: "grid",
                        "grid-template-columns": "auto 1fr",
                        gap: "3px 12px",
                      }}>
                        <span style={{ color: "var(--color-text-muted)" }}>Path</span>
                        <span style={{ "word-break": "break-all" }}>{file.path}</span>

                        <span style={{ color: "var(--color-text-muted)" }}>Size</span>
                        <span>{file.size >= 1048576 ? `${sizeMb()} MB` : `${sizeKb()} KB`}</span>

                        <span style={{ color: "var(--color-text-muted)" }}>Modified</span>
                        <span>{dateStr()}</span>

                        <Show when={file.page_count != null}>
                          <span style={{ color: "var(--color-text-muted)" }}>Pages</span>
                          <span>{file.page_count}</span>
                        </Show>

                        <Show when={file.chunk_count != null}>
                          <span style={{ color: "var(--color-text-muted)" }}>Chunks</span>
                          <span style={{ color: "var(--color-accent-cyan)" }}>{file.chunk_count}</span>
                        </Show>

                        <Show when={file.file_id != null}>
                          <span style={{ color: "var(--color-text-muted)" }}>File ID</span>
                          <span>#{file.file_id}</span>
                        </Show>
                      </div>
                    </Show>
                  </div>
                );
              }}
            </For>
          </div>
        </Show>
      </div>
    </div>
  );

  return (
    <>
      <SplitPane
        left={leftPanel()}
        right={rightPanel()}
        leftWidth={state.sourcesLeftPanelWidth}
        onResize={(w) => actions.setSourcesLeftPanelWidth(w)}
        minLeft={240}
        minRight={300}
      />

      {/* Shared browse modal rendered via the BrowseModal composable */}
      <browse.BrowseModalUI config={browseConfig} />
    </>
  );
};

// ---------------------------------------------------------------------------
// SessionDetailPanel: extracted sub-component for expanded session metadata.
// Displays all indexing parameters and statistics in a two-column grid.
// ---------------------------------------------------------------------------

interface SessionDetailPanelProps {
  session: SessionDto;
  isActive: boolean;
  onDelete: (id: number) => void;
  confirmDeleteId: () => number | null;
  setConfirmDeleteId: (id: number | null) => void;
}

/** Expanded detail panel for a single session. Shows all stored metadata
 *  fields that were used during indexing: model, strategy, chunk parameters,
 *  vector dimensions, creation date, file/page/chunk counts, and total
 *  content size. Includes a delete button with confirmation. */
const SessionDetailPanel: Component<SessionDetailPanelProps> = (props) => {
  // Access props.session fields via accessor to preserve SolidJS fine-grained
  // reactivity. Destructuring into `const s = props.session` would capture a
  // snapshot at component creation time, and subsequent signal updates to
  // props.session would not trigger re-renders.
  const dateStr = () => props.session.created_at > 0
    ? new Date(props.session.created_at * 1000).toLocaleString()
    : "unknown";

  /** Formats byte count to human-readable size (KB/MB/GB). */
  const formatBytes = (bytes: number) => {
    if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(1)} GB`;
    if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
    return `${(bytes / 1024).toFixed(0)} KB`;
  };

  return (
    <div style={{
      padding: "8px 12px 12px 20px",
      "margin-bottom": "2px",
      background: "rgba(168, 85, 247, 0.04)",
      "border-left": "2px solid var(--color-accent-purple)",
      "font-size": "11px",
      color: "var(--color-text-secondary)",
    }}>
      {/* Metadata grid with per-field tooltips */}
      <div style={{
        display: "grid",
        "grid-template-columns": "auto 1fr",
        gap: "3px 12px",
        "margin-bottom": "8px",
      }}>
        <span style={{ color: "var(--color-text-muted)", display: "inline-flex", "align-items": "center" }}>Directory <Tip text="Absolute path of the folder that was indexed in this session. All supported documents in this folder and its subfolders were included." /></span>
        <span style={{ "word-break": "break-all" }}>{props.session.directory_path}</span>

        <span style={{ color: "var(--color-text-muted)", display: "inline-flex", "align-items": "center" }}>Model <Tip text="The embedding model used to generate vector representations for all chunks in this session. Searching requires the same model (or one with matching vector dimensions) to be loaded." /></span>
        <span>{props.session.model_name}</span>

        <span style={{ color: "var(--color-text-muted)", display: "inline-flex", "align-items": "center" }}>Strategy <Tip text="The chunking strategy that was used to split document text. Token: fixed-size subword token windows. Word: fixed-size word windows. Sentence: groups of consecutive sentences. Page: one chunk per document page." /></span>
        <span>{props.session.chunk_strategy}</span>

        <span style={{ color: "var(--color-text-muted)", display: "inline-flex", "align-items": "center" }}>Chunk Params <Tip text="The size and overlap parameters that were used during chunking. For Token/Word strategies this shows chunk size, for Sentence strategy max words, for Page strategy no parameters apply." /></span>
        <span>{
          props.session.chunk_strategy === "sentence"
            ? props.session.max_words != null ? `${props.session.max_words} max words` : "sentence (default)"
            : props.session.chunk_strategy === "page"
            ? "one chunk per page"
            : props.session.chunk_size != null
              ? `${props.session.chunk_size} ${props.session.chunk_strategy}s` + (props.session.chunk_overlap != null ? `, ${props.session.chunk_overlap} overlap` : "")
              : `${props.session.chunk_strategy} (default)`
        }</span>

        <span style={{ color: "var(--color-text-muted)", display: "inline-flex", "align-items": "center" }}>Vector Dim <Tip text="Number of dimensions in each embedding vector produced by the model. All vectors in this session share this dimensionality. A query model must output the same dimension to search this session." /></span>
        <span>{props.session.vector_dimension}</span>

        <span style={{ color: "var(--color-text-muted)", display: "inline-flex", "align-items": "center" }}>Created <Tip text="Timestamp when this session was first created. Subsequent re-indexing of modified files updates the session content but does not change this date." /></span>
        <span>{dateStr()}</span>

        <span style={{ color: "var(--color-text-muted)", display: "inline-flex", "align-items": "center" }}>Files <Tip text="Number of documents that have been indexed in this session." /></span>
        <span>{props.session.file_count}</span>

        <span style={{ color: "var(--color-text-muted)", display: "inline-flex", "align-items": "center" }}>Pages <Tip text="Total number of extracted pages across all indexed files in this session." /></span>
        <span>{props.session.total_pages}</span>

        <span style={{ color: "var(--color-text-muted)", display: "inline-flex", "align-items": "center" }}>Chunks <Tip text="Total number of text chunks stored in the vector index. Each chunk has one embedding vector. This is the searchable unit count for this session." /></span>
        <span style={{ color: "var(--color-accent-cyan)" }}>{props.session.total_chunks}</span>

        <span style={{ color: "var(--color-text-muted)", display: "inline-flex", "align-items": "center" }}>Content <Tip text="Total size of the raw extracted text content across all chunks, before embedding. Indicates how much document text this session covers." /></span>
        <span>{formatBytes(props.session.total_content_bytes)}</span>
      </div>

      {/* Delete button with confirmation */}
      <Show
        when={props.confirmDeleteId() === props.session.id}
        fallback={
          <button
            class="btn btn-sm"
            style={{ "font-size": "11px", padding: "2px 8px" }}
            onClick={(e) => {
              e.stopPropagation();
              props.setConfirmDeleteId(props.session.id);
            }}
          >
            Delete Session
          </button>
        }
      >
        <div class="row" style={{ gap: "4px" }} onClick={(e) => e.stopPropagation()}>
          <button
            class="btn btn-sm"
            style={{
              "font-size": "11px",
              padding: "2px 8px",
              color: "var(--color-accent-magenta)",
              "border-color": "var(--color-accent-magenta)",
            }}
            onClick={() => props.onDelete(props.session.id)}
          >
            Confirm Delete
          </button>
          <button
            class="btn btn-sm"
            style={{ "font-size": "11px", padding: "2px 8px" }}
            onClick={() => props.setConfirmDeleteId(null)}
          >
            Cancel
          </button>
        </div>
      </Show>
    </div>
  );
};

export default IndexingTab;
