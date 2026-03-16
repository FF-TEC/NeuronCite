/**
 * Central application store for the NeuronCite web frontend.
 * Uses SolidJS createStore for fine-grained reactivity. The store holds
 * all client-side state for sessions, settings, search, file browsing,
 * model status, and panel visibility.
 *
 * The store is initialized with default placeholder values. Components
 * call the provided setter functions (actions) to update state. SSE events
 * and API responses flow through these actions to keep the UI in sync.
 */

import { createStore, produce } from "solid-js/store";

import type {
  CitationRowDto,
  HealthResponse,
  OllamaModelDto,
  DocumentEntry,
  SearchResultDto,
  SessionDto,
} from "../api/types";

// ---------------------------------------------------------------------------
// Safe localStorage wrappers
// ---------------------------------------------------------------------------

/**
 * Reads a value from localStorage, returning null if localStorage is
 * unavailable (e.g. private browsing mode, sandboxed iframe, or
 * SecurityError). Avoids unhandled exceptions in restricted environments.
 */
function safeLsGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/**
 * Writes a value to localStorage. On QuotaExceededError (private browsing
 * or storage full), logs a warning with the key name so the caller can
 * diagnose which data could not be persisted. SecurityErrors from sandboxed
 * contexts are swallowed silently because the application still functions
 * correctly without persistence in those environments.
 */
function safeLsSet(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch (e) {
    if (e instanceof DOMException && e.name === 'QuotaExceededError') {
      console.warn(`localStorage quota exceeded for key "${key}"`);
    }
  }
}

/**
 * Removes a key from localStorage, silently swallowing errors caused by
 * SecurityError in sandboxed contexts or other browser restrictions.
 */
function safeLsRemove(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    /* private browsing or sandboxed context */
  }
}

// ---------------------------------------------------------------------------
// Store shape
// ---------------------------------------------------------------------------

/** Discriminated union for the six main application tabs in the topbar.
 *  Workflow tabs (left group): "sources" handles BibTeX-based source
 *  downloading, "indexing" manages sessions and document ingestion,
 *  "search" runs queries, "citations" runs local LLM citation verification.
 *  Utility tabs (right group): "settings" consolidates maintenance, doctor,
 *  log, and MCP panels; "models" manages embedding model downloads, loading,
 *  and reranker configuration. */
export type AppTab = "indexing" | "search" | "sources" | "citations" | "annotations" | "settings" | "models";

/** Indexing progress state received via SSE events. The `phase` field
 *  distinguishes between "extracting" (parallel PDF text extraction),
 *  "embedding" (sequential GPU embedding), and "building_index" (HNSW
 *  construction). This enables the StatusBar to show phase-specific labels. */
export interface IndexProgress {
  phase: "extracting" | "embedding" | "building_index";
  files_total: number;
  files_done: number;
  chunks_created: number;
  complete: boolean;
}

/** Generic progress tracker for any long-running operation (source fetching,
 *  citation export, etc.). Any tab can set this via actions.setTaskProgress()
 *  and the StatusBar renders a progress bar and label automatically. */
export interface TaskProgress {
  label: string;
  done: number;
  total: number;
}

/** Per-row state for the citation verification live table. Rows are
 *  initialized from the citation_create response and progressively updated
 *  via SSE events from the autonomous citation agent. */
export interface CitationRowState {
  id: number;
  cite_key: string;
  title: string;
  author: string;
  tex_line: number;
  tex_context: string;
  phase: "pending" | "searching" | "evaluating" | "reasoning" | "done" | "error";
  verdict: string | null;
  confidence: number | null;
  reasoning: string;
  flag: string;
  search_queries: string[];
}

/** Application state covering all tabs, panels, and status indicators. */
export interface AppState {
  // ---- Tab navigation ----
  /** The currently active main tab in the topbar. */
  activeTab: AppTab;
  /** Width in pixels of the left settings panel in the search tab.
   *  Persisted to localStorage to survive page reloads. */
  searchLeftPanelWidth: number;
  /** Width in pixels of the left panel in the sources (indexing) tab.
   *  Persisted to localStorage to survive page reloads. */
  sourcesLeftPanelWidth: number;
  /** Width in pixels of the left panel in the sources fetch tab.
   *  Persisted to localStorage to survive page reloads. */
  fetchLeftPanelWidth: number;
  /** Width in pixels of the left settings panel in the citations tab.
   *  Persisted to localStorage to survive page reloads. */
  citationLeftPanelWidth: number;
  /** Width in pixels of the left panel in the annotations tab.
   *  Persisted to localStorage to survive page reloads. */
  annotationLeftPanelWidth: number;

  // ---- Session management ----
  sessions: SessionDto[];
  activeSessionId: number | null;

  // ---- Settings ----
  folderPath: string;
  selectedStrategy: string;
  chunkSize: string;
  chunkOverlap: string;

  // ---- Search ----
  /** Session IDs selected for searching. When one session is selected, the
   *  standard /search endpoint is used. When two or more are selected, the
   *  /search/multi endpoint merges results from all sessions. */
  searchSessionIds: number[];
  queryText: string;
  useHybrid: boolean;
  useReranker: boolean;
  useRefine: boolean;
  refineDivisors: string;
  searchInProgress: boolean;

  // ---- Results ----
  searchResults: SearchResultDto[];
  groupByDocument: boolean;
  /** Maps result indices to their expanded/collapsed state. Uses a plain
   *  record instead of Set<number> because SolidJS store proxies do not
   *  track mutations on Set/Map objects. Record properties are individually
   *  tracked by the reactivity system. */
  expandedResults: Record<number, boolean>;

  // ---- Status bar ----
  /** Progress state for the indexing operation. Drives the progress bar
   *  overlay in the StatusBar with phase-specific labels. */
  indexProgress: IndexProgress | null;

  // ---- File browser ----
  documentFiles: DocumentEntry[];
  fileBrowserScanning: boolean;

  // ---- Health / system info ----
  health: HealthResponse | null;

  // ---- Model info ----
  activeModelId: string;
  activeModelDimension: number;
  /** Whether a cross-encoder reranker model is loaded in the GPU worker.
   *  Derived from the health endpoint on startup and updated via SSE
   *  reranker_loaded events. Controls whether the Search tab's Reranker
   *  toggle is functional. */
  rerankerAvailable: boolean;
  /** HuggingFace model identifier of the loaded cross-encoder reranker.
   *  Populated via SSE reranker_loaded events. Empty string when no reranker
   *  is loaded. The StatusBar displays this alongside the embedding model
   *  when non-empty. */
  rerankerModelId: string;

  // ---- Compute mode ----
  /** Actual GPU device name (e.g., "NVIDIA GeForce RTX 4090"). Populated
   *  from the model catalog endpoint on startup. Empty string when no GPU
   *  is detected or when the catalog has not been fetched yet. */
  gpuDeviceName: string;

  // ---- Log panel ----
  logMessages: LogEntry[];

  // ---- Citation verification ----
  /** Session IDs selected for citation verification. When one session is
   *  selected, the backend matches files from that session. When multiple
   *  sessions are selected, files from all sessions are aggregated for
   *  cite-key matching and the agent searches across all sessions. */
  citationSessionIds: number[];
  citationJobId: string | null;
  citationRows: CitationRowState[];
  citationRowsDone: number;
  citationRowsTotal: number;
  citationComplete: boolean;
  citationVerdicts: Record<string, number>;

  // ---- Ollama configuration ----
  ollamaUrl: string;
  ollamaModels: OllamaModelDto[];
  ollamaSelectedModel: string;
  ollamaConnected: boolean;
  /** Names of models currently loaded in GPU/CPU RAM (from GET /api/ps). */
  ollamaRunningModels: string[];
  /** Model tag currently being pulled from the Ollama registry, or null. */
  ollamaPullingModel: string | null;
  /** Byte-level progress for the active pull operation. Null when no pull
   *  is in progress. Updated via SSE ollama_pull_progress events. */
  ollamaPullProgress: { total: number; completed: number } | null;

  // ---- Unpaywall DOI resolution ----
  /** Email address for Unpaywall API access in the DOI resolution chain.
   *  When non-empty, the source fetch pipeline queries Unpaywall first for
   *  direct open-access PDF URLs before falling back to Semantic Scholar,
   *  OpenAlex, and doi.org. */
  unpaywallEmail: string;

  // ---- Source fetching (Sources tab) ----
  /** Whether a source fetch operation is running in the backend. */
  sourceFetchInProgress: boolean;
  /** Status message from the last source fetch operation. */
  sourceFetchMessage: string;

  // ---- Generic task progress ----
  /** Generic progress for any long-running operation. The StatusBar renders
   *  a progress bar and label when this is non-null. Independent of
   *  indexProgress which has its own SSE-driven phase system. */
  taskProgress: TaskProgress | null;

  // ---- Annotation tab ----
  /** Job ID of the annotation job started from the Annotations tab.
   *  Null when no annotation job is active. */
  annotationJobId: string | null;
  /** Current state of the annotation job ("queued", "running", "completed",
   *  "failed"). Empty string when no job is active. */
  annotationJobState: string;
  /** Number of annotation rows processed so far. */
  annotationProgressDone: number;
  /** Total number of annotation rows in the job. */
  annotationProgressTotal: number;
  /** Error message from the annotation job, if any. */
  annotationErrorMessage: string;

  // ---- Model catalog refresh ----
  /** Monotonically increasing counter that triggers catalog refetch in
   *  ModelsPanel when incremented. Updated by SSE event handlers in App.tsx
   *  when model_downloaded, model_switched, or reranker_loaded events arrive. */
  catalogRefreshTrigger: number;
}

// ---------------------------------------------------------------------------
// Default values
// ---------------------------------------------------------------------------

const initialState: AppState = {
  activeTab: "search",
  searchLeftPanelWidth: 320,
  sourcesLeftPanelWidth: 320,
  fetchLeftPanelWidth: 320,
  citationLeftPanelWidth: 320,
  annotationLeftPanelWidth: 320,

  sessions: [],
  activeSessionId: null,

  folderPath: "",
  selectedStrategy: "token",
  chunkSize: "256",
  chunkOverlap: "32",

  searchSessionIds: [],
  queryText: "",
  useHybrid: true,
  useReranker: false,
  useRefine: true,
  refineDivisors: "4,8,16",
  searchInProgress: false,

  searchResults: [],
  groupByDocument: true,
  expandedResults: {},

  indexProgress: null,

  documentFiles: [],
  fileBrowserScanning: false,

  health: null,

  activeModelId: "",
  activeModelDimension: 0,
  rerankerAvailable: false,
  rerankerModelId: "",
  gpuDeviceName: "",

  logMessages: [],

  citationSessionIds: [],
  citationJobId: null,
  citationRows: [],
  citationRowsDone: 0,
  citationRowsTotal: 0,
  citationComplete: false,
  citationVerdicts: {},

  ollamaUrl: "http://localhost:11434",
  ollamaModels: [],
  ollamaSelectedModel: "",
  ollamaConnected: false,
  ollamaRunningModels: [],
  ollamaPullingModel: null,
  ollamaPullProgress: null,

  unpaywallEmail: "",

  sourceFetchInProgress: false,
  sourceFetchMessage: "",

  taskProgress: null,

  annotationJobId: null,
  annotationJobState: "",
  annotationProgressDone: 0,
  annotationProgressTotal: 0,
  annotationErrorMessage: "",

  catalogRefreshTrigger: 0,
};

// ---------------------------------------------------------------------------
// localStorage keys for persisting citation state across page reloads
// ---------------------------------------------------------------------------

const LS_CITATION_JOB_ID = "neuroncite_citation_job_id";
const LS_OLLAMA_URL = "neuroncite_ollama_url";
const LS_OLLAMA_MODEL = "neuroncite_ollama_model";
const LS_TEX_PATH = "neuroncite_tex_path";
const LS_BIB_PATH = "neuroncite_bib_path";
const LS_SOURCE_PATH = "neuroncite_source_path";
const LS_UNPAYWALL_EMAIL = "neuroncite_unpaywall_email";
const LS_SEARCH_PANEL_WIDTH = "neuroncite_search_panel_width";
const LS_SOURCES_PANEL_WIDTH = "neuroncite_sources_panel_width";
const LS_FETCH_PANEL_WIDTH = "neuroncite_fetch_panel_width";
const LS_CITATION_PANEL_WIDTH = "neuroncite_citation_panel_width";
const LS_ANNOTATION_PANEL_WIDTH = "neuroncite_annotation_panel_width";

// ---------------------------------------------------------------------------
// URL validation helpers
// ---------------------------------------------------------------------------

/**
 * Returns true when the given string is a valid URL with an http or https
 * scheme. Used to reject file:, javascript:, and other non-HTTP schemes
 * before persisting a user-supplied Ollama server URL to localStorage or
 * the application store.
 */
function isHttpUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Store instance (singleton for the application)
// ---------------------------------------------------------------------------

// Restore persisted values from localStorage. These values survive page
// reloads so the user's configuration carries over between sessions.
const restoredOllamaUrl = safeLsGet(LS_OLLAMA_URL) || "http://localhost:11434";
const restoredOllamaModel = safeLsGet(LS_OLLAMA_MODEL) || "";
const restoredUnpaywallEmail = safeLsGet(LS_UNPAYWALL_EMAIL) || "";
const restoredPanelWidth = parseInt(safeLsGet(LS_SEARCH_PANEL_WIDTH) || "320", 10);
const restoredSourcesPanelWidth = parseInt(safeLsGet(LS_SOURCES_PANEL_WIDTH) || "320", 10);
const restoredFetchPanelWidth = parseInt(safeLsGet(LS_FETCH_PANEL_WIDTH) || "320", 10);
const restoredCitationPanelWidth = parseInt(safeLsGet(LS_CITATION_PANEL_WIDTH) || "320", 10);
const restoredAnnotationPanelWidth = parseInt(safeLsGet(LS_ANNOTATION_PANEL_WIDTH) || "320", 10);

const [state, setState] = createStore<AppState>({
  ...initialState,
  searchLeftPanelWidth: isNaN(restoredPanelWidth) ? 320 : restoredPanelWidth,
  sourcesLeftPanelWidth: isNaN(restoredSourcesPanelWidth) ? 320 : restoredSourcesPanelWidth,
  fetchLeftPanelWidth: isNaN(restoredFetchPanelWidth) ? 320 : restoredFetchPanelWidth,
  citationLeftPanelWidth: isNaN(restoredCitationPanelWidth) ? 320 : restoredCitationPanelWidth,
  annotationLeftPanelWidth: isNaN(restoredAnnotationPanelWidth) ? 320 : restoredAnnotationPanelWidth,
  ollamaUrl: restoredOllamaUrl,
  ollamaSelectedModel: restoredOllamaModel,
  unpaywallEmail: restoredUnpaywallEmail,
});

/** Maximum number of log messages retained in the client-side buffer.
 *  Set to 2000 to retain a larger log history during long indexing sessions,
 *  matching the backend broadcast channel capacity of 2048. */
const MAX_LOG_MESSAGES = 2000;

/** Structured log entry received from the backend's BroadcastLayer via SSE.
 *  Contains the tracing level, module target, and human-readable message text.
 *  The timestamp is captured client-side when the SSE event arrives. */
export interface LogEntry {
  level: string;
  target: string;
  message: string;
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Actions -- grouped by domain
// ---------------------------------------------------------------------------

const actions = {
  // ---- Sessions ----

  setSessions(sessions: SessionDto[]) {
    setState("sessions", sessions);
  },

  setActiveSession(id: number | null) {
    setState("activeSessionId", id);
  },

  // ---- Settings ----

  setFolderPath(path: string) {
    setState("folderPath", path);
  },

  setStrategy(strategy: string) {
    setState("selectedStrategy", strategy);
  },

  setChunkSize(size: string) {
    setState("chunkSize", size);
  },

  setChunkOverlap(overlap: string) {
    setState("chunkOverlap", overlap);
  },

  // ---- Search ----

  /** Toggles a session ID in the search selection. If the session is already
   *  selected, it is removed. If not, it is added. Used by the SearchTab
   *  multi-select session list. */
  toggleSearchSession(sessionId: number) {
    setState(
      produce((s) => {
        const idx = s.searchSessionIds.indexOf(sessionId);
        if (idx >= 0) {
          s.searchSessionIds.splice(idx, 1);
        } else {
          s.searchSessionIds.push(sessionId);
        }
      }),
    );
  },

  /** Replaces the entire search session selection. Used when auto-selecting
   *  sessions on startup or when the user selects a session in IndexingTab. */
  setSearchSessionIds(ids: number[]) {
    setState("searchSessionIds", ids);
  },

  setQueryText(text: string) {
    setState("queryText", text);
  },

  setUseHybrid(value: boolean) {
    setState("useHybrid", value);
  },

  setUseReranker(value: boolean) {
    setState("useReranker", value);
  },

  setUseRefine(value: boolean) {
    setState("useRefine", value);
  },

  setRefineDivisors(value: string) {
    setState("refineDivisors", value);
  },

  setSearchInProgress(value: boolean) {
    setState("searchInProgress", value);
  },

  // ---- Results ----

  /** Replaces the search results array and resets all expansion states.
   *  The expandedResults record is cleared to an empty object because
   *  previous expansion indices do not apply to a fresh result set. */
  setSearchResults(results: SearchResultDto[]) {
    setState("searchResults", results);
    setState("expandedResults", {});
  },

  setGroupByDocument(value: boolean) {
    setState("groupByDocument", value);
  },

  /** Toggles the expanded/collapsed state of a search result at the given
   *  index. Uses produce() for fine-grained property-level reactivity on
   *  the record, which SolidJS tracks per key unlike Set mutations. */
  toggleResultExpanded(index: number) {
    setState(
      produce((s) => {
        s.expandedResults[index] = !s.expandedResults[index];
      }),
    );
  },

  // ---- Status bar ----

  setIndexProgress(progress: IndexProgress | null) {
    setState("indexProgress", progress);
  },

  // ---- File browser ----

  setDocumentFiles(files: DocumentEntry[]) {
    setState("documentFiles", files);
  },

  setFileBrowserScanning(scanning: boolean) {
    setState("fileBrowserScanning", scanning);
  },

  // ---- Health ----

  setHealth(health: HealthResponse) {
    setState("health", health);
    setState("rerankerAvailable", health.reranker_available);
  },

  // ---- Model ----

  setActiveModel(modelId: string, dimension: number) {
    setState("activeModelId", modelId);
    setState("activeModelDimension", dimension);
  },

  setRerankerAvailable(available: boolean) {
    setState("rerankerAvailable", available);
  },

  /** Stores the cross-encoder reranker model identifier from the SSE
   *  reranker_loaded event for display in the StatusBar. */
  setRerankerModelId(modelId: string) {
    setState("rerankerModelId", modelId);
  },

  /** Stores the GPU device name for display in the StatusBar compute
   *  mode indicator. Populated from the model catalog endpoint. */
  setGpuDeviceName(name: string) {
    setState("gpuDeviceName", name);
  },

  // ---- Tab navigation ----

  /** Switches the active main tab in the topbar. */
  setActiveTab(tab: AppTab) {
    setState("activeTab", tab);
  },

  /** Sets the search tab left panel width and persists to localStorage. */
  setSearchLeftPanelWidth(width: number) {
    setState("searchLeftPanelWidth", width);
    safeLsSet(LS_SEARCH_PANEL_WIDTH, String(width));
  },

  /** Sets the sources (indexing) tab left panel width and persists to localStorage. */
  setSourcesLeftPanelWidth(width: number) {
    setState("sourcesLeftPanelWidth", width);
    safeLsSet(LS_SOURCES_PANEL_WIDTH, String(width));
  },

  /** Sets the sources fetch tab left panel width and persists to localStorage. */
  setFetchLeftPanelWidth(width: number) {
    setState("fetchLeftPanelWidth", width);
    safeLsSet(LS_FETCH_PANEL_WIDTH, String(width));
  },

  /** Sets the citations tab left panel width and persists to localStorage. */
  setCitationLeftPanelWidth(width: number) {
    setState("citationLeftPanelWidth", width);
    safeLsSet(LS_CITATION_PANEL_WIDTH, String(width));
  },

  /** Sets the annotations tab left panel width and persists to localStorage. */
  setAnnotationLeftPanelWidth(width: number) {
    setState("annotationLeftPanelWidth", width);
    safeLsSet(LS_ANNOTATION_PANEL_WIDTH, String(width));
  },

  // ---- Log ----

  /** Appends a structured log entry to the message buffer. Trims older
   *  entries when the buffer exceeds MAX_LOG_MESSAGES to prevent unbounded
   *  memory growth during long-running sessions. */
  appendLogMessage(entry: LogEntry) {
    setState(
      produce((s) => {
        s.logMessages.push(entry);
        if (s.logMessages.length > MAX_LOG_MESSAGES) {
          s.logMessages.splice(0, s.logMessages.length - MAX_LOG_MESSAGES);
        }
      }),
    );
  },

  clearLogMessages() {
    setState("logMessages", []);
  },

  // ---- Citation verification ----

  /** Toggles a session ID in the citation session selection. If the session
   *  is already selected, it is removed. If not, it is added. Used by the
   *  CitationPanel multi-select session list. */
  toggleCitationSession(sessionId: number) {
    setState(
      produce((s) => {
        const idx = s.citationSessionIds.indexOf(sessionId);
        if (idx >= 0) {
          s.citationSessionIds.splice(idx, 1);
        } else {
          s.citationSessionIds.push(sessionId);
        }
      }),
    );
  },

  /** Replaces the entire citation session selection. Used when auto-selecting
   *  sessions after indexing or on startup. */
  setCitationSessionIds(ids: number[]) {
    setState("citationSessionIds", ids);
  },

  /** Sets the citation job ID and persists it to localStorage. Passing null
   *  clears the stored job (e.g. when creating a fresh job or resetting). */
  setCitationJobId(jobId: string | null) {
    setState("citationJobId", jobId);
    if (jobId) {
      safeLsSet(LS_CITATION_JOB_ID, jobId);
    } else {
      safeLsRemove(LS_CITATION_JOB_ID);
    }
  },

  /** Initializes the citation rows table from the creation response.
   *  Each row starts in "pending" phase because the job was just created. */
  setCitationRows(rows: CitationRowDto[]) {
    const rowStates: CitationRowState[] = rows.map((r) => ({
      id: r.id,
      cite_key: r.cite_key,
      title: r.title,
      author: r.author,
      tex_line: r.tex_line,
      tex_context: r.tex_context || "",
      phase: "pending" as const,
      verdict: null,
      confidence: null,
      reasoning: "",
      flag: "",
      search_queries: [],
    }));
    setState("citationRows", rowStates);
    setState("citationRowsTotal", rows.length);
    setState("citationRowsDone", 0);
    setState("citationComplete", false);
    setState("citationVerdicts", {});
  },

  /** Recovers the citation rows table from API data. Unlike setCitationRows,
   *  this maps rows that may already be completed/claimed/failed by parsing
   *  the result_json field for verdict, confidence, and reasoning data. */
  recoverCitationRows(rows: CitationRowDto[]) {
    const rowStates: CitationRowState[] = rows.map((r) => {
      let phase: CitationRowState["phase"] = "pending";
      let verdict: string | null = null;
      let confidence: number | null = null;
      let reasoning = "";
      let flag = r.flag || "";

      // Map the DB status to the UI phase indicator.
      if (r.status === "done") phase = "done";
      else if (r.status === "claimed") phase = "searching";
      else if (r.status === "failed") phase = "error";

      // For completed rows, extract the verification result from the
      // serialized JSON stored in the database.
      if (r.result_json) {
        try {
          const result = JSON.parse(r.result_json);
          verdict = result.verdict || null;
          confidence = typeof result.confidence === "number" ? result.confidence : null;
          reasoning = result.reasoning || "";
          if (result.flag && !flag) flag = result.flag;
        } catch {
          // Malformed result_json: row will display without verdict data.
        }
      }

      return {
        id: r.id,
        cite_key: r.cite_key,
        title: r.title,
        author: r.author,
        tex_line: r.tex_line,
        tex_context: r.tex_context || "",
        phase,
        verdict,
        confidence,
        reasoning,
        flag,
        search_queries: [],
      };
    });
    setState("citationRows", rowStates);
  },

  /** Reconciles citation rows with authoritative database state after job
   *  completion. Unlike recoverCitationRows (which only runs when the rows
   *  array is empty), this overwrites existing row phases and verdicts to
   *  correct stale SSE state. Rows whose SSE "done" or "error" events were
   *  lost during processing will have their phase/verdict corrected from
   *  the database result_json field. */
  reconcileCitationRows(rows: CitationRowDto[]) {
    // Build a lookup map from the API rows keyed by row ID for O(1) access.
    const apiMap = new Map<number, CitationRowDto>();
    for (const r of rows) {
      apiMap.set(r.id, r);
    }

    setState(
      produce((s) => {
        for (const row of s.citationRows) {
          const apiRow = apiMap.get(row.id);
          if (!apiRow) continue;

          // Map the DB status to the UI phase indicator.
          if (apiRow.status === "done") row.phase = "done";
          else if (apiRow.status === "claimed") row.phase = "searching";
          else if (apiRow.status === "failed") row.phase = "error";
          else row.phase = "pending";

          // Extract verdict, confidence, reasoning from result_json.
          if (apiRow.result_json) {
            try {
              const result = JSON.parse(apiRow.result_json);
              row.verdict = result.verdict || null;
              row.confidence = typeof result.confidence === "number" ? result.confidence : null;
              row.reasoning = result.reasoning || "";
              if (result.flag) row.flag = result.flag;
            } catch {
              // Malformed result_json: leave existing row data unchanged.
            }
          }
          if (apiRow.flag && !row.flag) row.flag = apiRow.flag;
        }
      }),
    );
  },

  /** Updates a single citation row's phase, verdict, confidence, and other
   *  fields from an SSE citation_row_update event. */
  updateCitationRow(data: {
    row_id: number;
    phase: string;
    verdict?: string;
    confidence?: number;
    reasoning?: string;
    search_queries?: string[];
    error_message?: string;
  }) {
    setState(
      produce((s) => {
        const row = s.citationRows.find((r) => r.id === data.row_id);
        if (!row) return;
        row.phase = data.phase as CitationRowState["phase"];
        if (data.verdict !== undefined) row.verdict = data.verdict;
        if (data.confidence !== undefined) row.confidence = data.confidence;
        if (data.reasoning !== undefined) row.reasoning = data.reasoning;
        if (data.search_queries !== undefined) row.search_queries = data.search_queries;
        if (data.error_message) row.reasoning = data.error_message;
      }),
    );
  },

  /** Appends a streaming reasoning token to a specific citation row.
   *  Used by the citation_reasoning_token SSE event for live display. */
  appendCitationReasoning(rowId: number, token: string) {
    setState(
      produce((s) => {
        const row = s.citationRows.find((r) => r.id === rowId);
        if (row) {
          row.reasoning += token;
        }
      }),
    );
  },

  /** Updates the aggregate job progress counters from the
   *  citation_job_progress SSE event. */
  setCitationProgress(data: {
    job_id: string;
    rows_done: number;
    rows_total: number;
    verdicts: Record<string, number>;
  }) {
    // If we receive a progress event for a job we don't have in the store,
    // store the job_id so recovery can pick it up. This handles the case
    // where the page reloaded and the SSE reconnects before recovery runs.
    if (!state.citationJobId && data.job_id) {
      setState("citationJobId", data.job_id);
      safeLsSet(LS_CITATION_JOB_ID, data.job_id);
    }

    setState("citationRowsDone", data.rows_done);
    setState("citationRowsTotal", data.rows_total);
    setState("citationVerdicts", data.verdicts);
    if (data.rows_done >= data.rows_total && data.rows_total > 0) {
      setState("citationComplete", true);
    }
  },

  setCitationComplete(complete: boolean) {
    setState("citationComplete", complete);
  },

  // ---- Ollama ----

  /** Sets the Ollama URL and persists to localStorage. Rejects URLs that
   *  do not use the http or https scheme to prevent non-HTTP values
   *  (e.g. file:, javascript:) from being stored or sent to the backend. */
  setOllamaUrl(url: string) {
    if (!isHttpUrl(url)) {
      console.warn('rejected non-http(s) Ollama URL');
      return;
    }
    setState("ollamaUrl", url);
    safeLsSet(LS_OLLAMA_URL, url);
  },

  setOllamaModels(models: OllamaModelDto[]) {
    setState("ollamaModels", models);
  },

  /** Sets the selected Ollama model and persists to localStorage. */
  setOllamaSelectedModel(model: string) {
    setState("ollamaSelectedModel", model);
    safeLsSet(LS_OLLAMA_MODEL, model);
  },

  setOllamaConnected(connected: boolean) {
    setState("ollamaConnected", connected);
  },

  /** Replaces the list of model names currently loaded in GPU/CPU RAM.
   *  Populated from the GET /api/v1/web/ollama/running response. */
  setOllamaRunningModels(names: string[]) {
    setState("ollamaRunningModels", names);
  },

  /** Sets the model tag currently being pulled, or null when no pull
   *  is in progress. The ModelsPanel uses this to show a progress bar
   *  on the corresponding table row. */
  setOllamaPullingModel(model: string | null) {
    setState("ollamaPullingModel", model);
  },

  /** Updates the byte-level progress for the active Ollama pull operation.
   *  Set to null when no pull is in progress or when the pull completes. */
  setOllamaPullProgress(progress: { total: number; completed: number } | null) {
    setState("ollamaPullProgress", progress);
  },

  // ---- Unpaywall ----

  /** Sets the Unpaywall email and persists to localStorage. */
  setUnpaywallEmail(email: string) {
    setState("unpaywallEmail", email);
    safeLsSet(LS_UNPAYWALL_EMAIL, email);
  },

  // ---- Source fetching ----

  setSourceFetchInProgress(value: boolean) {
    setState("sourceFetchInProgress", value);
  },

  setSourceFetchMessage(message: string) {
    setState("sourceFetchMessage", message);
  },

  // ---- Generic task progress ----

  setTaskProgress(progress: TaskProgress | null) {
    setState("taskProgress", progress);
  },

  // ---- Annotation tab ----

  /** Sets the annotation job ID and state when a standalone annotation job
   *  is started from the Annotations tab. */
  setAnnotationJob(jobId: string | null, jobState: string) {
    setState("annotationJobId", jobId);
    setState("annotationJobState", jobState);
    if (!jobId) {
      setState("annotationProgressDone", 0);
      setState("annotationProgressTotal", 0);
      setState("annotationErrorMessage", "");
    }
  },

  /** Updates annotation job progress from SSE job_update events. */
  setAnnotationProgress(done: number, total: number) {
    setState("annotationProgressDone", done);
    setState("annotationProgressTotal", total);
  },

  /** Updates annotation job state from SSE job_update events. */
  setAnnotationJobState(jobState: string) {
    setState("annotationJobState", jobState);
  },

  /** Sets an error message for the annotation job. */
  setAnnotationErrorMessage(message: string) {
    setState("annotationErrorMessage", message);
  },

  /** Increments the catalog refresh trigger counter. Used by SSE handlers
   *  to signal ModelsPanel that the catalog data is stale and should be
   *  refetched from the server. */
  triggerCatalogRefresh() {
    setState("catalogRefreshTrigger", (prev: number) => prev + 1);
  },

  // ---- Reset ----

  /** Removes all NeuronCite keys from localStorage and resets the store
   *  fields that are backed by localStorage to their initial defaults.
   *  Runtime-only state (sessions, results, health, etc.) is left untouched
   *  because it is populated from the backend on every page load anyway. */
  resetToDefaults() {
    // Remove every persisted key from the browser's localStorage.
    safeLsRemove(LS_CITATION_JOB_ID);
    safeLsRemove(LS_OLLAMA_URL);
    safeLsRemove(LS_OLLAMA_MODEL);
    safeLsRemove(LS_TEX_PATH);
    safeLsRemove(LS_BIB_PATH);
    safeLsRemove(LS_SOURCE_PATH);
    safeLsRemove(LS_UNPAYWALL_EMAIL);
    safeLsRemove(LS_SEARCH_PANEL_WIDTH);
    safeLsRemove(LS_SOURCES_PANEL_WIDTH);
    safeLsRemove(LS_FETCH_PANEL_WIDTH);
    safeLsRemove(LS_CITATION_PANEL_WIDTH);
    safeLsRemove(LS_ANNOTATION_PANEL_WIDTH);

    // Reset the in-memory store fields to match the initial defaults
    // defined at the top of this module.
    setState("searchLeftPanelWidth", initialState.searchLeftPanelWidth);
    setState("sourcesLeftPanelWidth", initialState.sourcesLeftPanelWidth);
    setState("fetchLeftPanelWidth", initialState.fetchLeftPanelWidth);
    setState("citationLeftPanelWidth", initialState.citationLeftPanelWidth);
    setState("annotationLeftPanelWidth", initialState.annotationLeftPanelWidth);
    setState("annotationJobId", initialState.annotationJobId);
    setState("annotationJobState", initialState.annotationJobState);
    setState("annotationProgressDone", initialState.annotationProgressDone);
    setState("annotationProgressTotal", initialState.annotationProgressTotal);
    setState("annotationErrorMessage", initialState.annotationErrorMessage);
    setState("ollamaUrl", initialState.ollamaUrl);
    setState("ollamaSelectedModel", initialState.ollamaSelectedModel);
    setState("unpaywallEmail", initialState.unpaywallEmail);
    setState("citationJobId", initialState.citationJobId);
    setState("queryText", initialState.queryText);
    setState("useHybrid", initialState.useHybrid);
    setState("useReranker", initialState.useReranker);
    setState("useRefine", initialState.useRefine);
    setState("refineDivisors", initialState.refineDivisors);
    setState("folderPath", initialState.folderPath);
    setState("selectedStrategy", initialState.selectedStrategy);
    setState("chunkSize", initialState.chunkSize);
    setState("chunkOverlap", initialState.chunkOverlap);
  },
};

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export { state, actions, safeLsGet, safeLsSet, safeLsRemove, LS_CITATION_JOB_ID, LS_TEX_PATH, LS_BIB_PATH, LS_SOURCE_PATH, LS_UNPAYWALL_EMAIL };
