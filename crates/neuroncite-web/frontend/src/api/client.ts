/**
 * Typed HTTP client for the NeuronCite REST API. Wraps the native fetch API
 * with JSON serialization, error handling, and typed request/response pairs.
 *
 * All paths are relative to /api/v1 and are proxied to the Rust backend
 * during development via Vite's dev server proxy configuration.
 *
 * Every request goes through fetchWithTimeout which enforces a maximum
 * duration via AbortController. The default timeout is 30 seconds for
 * standard requests and 120 seconds for long-running operations (indexing,
 * citation creation, source fetching, model doctor, annotation).
 */

import type {
  AnnotateFromFileRequest,
  AnnotateFromFileResponse,
  AutoVerifyRequest,
  AutoVerifyResponse,
  BackendListResponse,
  BibReportRequest,
  BibReportResponse,
  BrowseRequest,
  BrowseResponse,
  ChunksResponse,
  ParseBibRequest,
  ParseBibResponse,
  CitationCreateRequest,
  CitationCreateResponse,
  CitationExportRequest,
  CitationRowsResponse,
  CitationStatusResponse,
  ConfigResponse,
  DependencyProbe,
  DiscoverRequest,
  DiscoverResponse,
  DoctorInstallRequest,
  DoctorInstallResponse,
  FetchSourcesRequest,
  FetchSourcesResponse,
  HealthResponse,
  IndexRequest,
  IndexResponse,
  JobCancelResponse,
  JobListResponse,
  JobResponse,
  LoadRerankerRequest,
  LoadRerankerResponse,
  McpStatusResponse,
  McpTarget,
  ModelActivateRequest,
  ModelActivateResponse,
  ModelCatalogResponse,
  ModelDoctorResponse,
  ModelDownloadRequest,
  ModelDownloadResponse,
  MultiSearchRequest,
  MultiSearchResponse,
  NativeBrowseResponse,
  OllamaCatalogResponse,
  OllamaDeleteResponse,
  OllamaModelsResponse,
  OllamaPullResponse,
  OllamaRunningResponse,
  OllamaShowResponse,
  OllamaStatusResponse,
  OptimizeResponse,
  PageResponse,
  RebuildResponse,
  ScanDocumentsRequest,
  ScanDocumentsResponse,
  SearchRequest,
  SearchResponse,
  SessionDeleteResponse,
  SessionListResponse,
  SetupCompleteResponse,
  SetupStatusResponse,
} from "./types";

/** Base path for all API requests. */
const BASE = "/api/v1";

/** Default timeout in milliseconds for standard API requests. */
const DEFAULT_TIMEOUT_MS = 30_000;

/** Timeout in milliseconds for long-running API operations such as
 *  indexing, citation verification, annotation, and model diagnostics
 *  that may take several minutes to complete. */
const LONG_TIMEOUT_MS = 120_000;

/** Timeout disabled for native OS dialog endpoints. These requests block
 *  until the user selects a file/folder or cancels, which can take an
 *  arbitrary amount of time. A finite timeout causes "Failed to fetch"
 *  errors and an unwanted fallback to the browser-based file picker. */
const NATIVE_DIALOG_TIMEOUT_MS = 0;

/** Timeout in milliseconds for the fetch-sources operation. Source fetching
 *  processes one BibTeX entry per delay_ms interval (default 1000 ms), so a
 *  file with 200 entries takes at least 200 seconds plus DOI resolution and
 *  network download latency. 30 minutes accommodates bib files up to ~1700
 *  entries at 1 s delay each. LONG_TIMEOUT_MS (2 min) is far too short for
 *  real-world bib files and causes the frontend to abort before completion,
 *  leaving results stuck at "pending" and never calling setResults. */
const FETCH_SOURCES_TIMEOUT_MS = 1_800_000;

/** Error thrown when the API returns a non-2xx status code. */
export class ApiError extends Error {
  constructor(
    public status: number,
    public body: string,
  ) {
    super(`API error ${status}: ${body}`);
    this.name = "ApiError";
  }
}

/**
 * Asserts that a parsed response object contains a required field.
 * Throws if the value is not a non-null object or if the field is absent.
 * Used to validate the shape of critical API responses before accessing
 * their properties, catching contract violations at the boundary.
 *
 * @param data - The parsed response value of unknown type
 * @param field - The field name that must be present on the object
 * @returns The field value cast to T
 */
export function assertField<T>(data: unknown, field: string): T {
  if (typeof data !== 'object' || data === null || !(field in data)) {
    throw new Error(`response missing field: ${field}`);
  }
  return (data as Record<string, unknown>)[field] as T;
}

/**
 * Wraps fetch() with an AbortController-based timeout. If the request
 * does not complete within timeoutMs, the controller aborts the request
 * and fetch() rejects with an AbortError.
 *
 * If the caller provides an existing signal in init.signal (e.g. for
 * user-initiated cancellation), the abort is forwarded to the internal
 * controller so both timeout and manual cancellation work together.
 * The forwarding listener is removed in the finally block to prevent
 * memory leaks when the caller's AbortSignal outlives this request.
 */
async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  // When timeoutMs is 0, no timeout is applied. This is used for native
  // OS dialog endpoints that block until the user interacts with the dialog.
  const timer = timeoutMs > 0
    ? setTimeout(() => controller.abort(), timeoutMs)
    : undefined;

  /** Listener reference for the caller-provided signal. Stored so it
   *  can be removed in the finally block to avoid a retained reference
   *  from the caller's AbortSignal to this controller after the request
   *  completes or times out. */
  let forwardAbort: (() => void) | undefined;
  const existing = init.signal;

  try {
    if (existing) {
      forwardAbort = () => controller.abort();
      existing.addEventListener("abort", forwardAbort);
    }
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    if (timer !== undefined) clearTimeout(timer);
    if (existing && forwardAbort) {
      existing.removeEventListener("abort", forwardAbort);
    }
  }
}

/**
 * HTTP status codes for which a request may be retried. These are transient
 * server-side conditions (timeout, rate limit, temporary unavailability) that
 * have a reasonable probability of succeeding on a subsequent attempt.
 * 408 = Request Timeout, 429 = Too Many Requests, 500/502/503/504 = server errors.
 */
const RETRYABLE_STATUSES = new Set([408, 429, 500, 502, 503, 504]);

/**
 * Wraps a single fetch attempt with automatic retry on retryable HTTP status
 * codes. On each failed attempt, waits for either the Retry-After header
 * duration (if present) or an exponentially increasing base delay before
 * the next attempt. Returns the final Response after exhausting all retries.
 *
 * This function is used only for JSON API calls (GET, POST, DELETE). SSE
 * connections handle their own reconnection logic independently.
 *
 * @param url - Full request URL
 * @param init - RequestInit passed through to fetch on each attempt
 * @param maxRetries - Number of retry attempts after the first failure (default 2)
 * @param baseDelayMs - Base delay for exponential backoff in milliseconds (default 500)
 * @returns The last Response received, which callers inspect for ok status
 */
async function fetchWithRetry(
  url: string,
  init: RequestInit,
  timeoutMs: number,
  maxRetries = 2,
  baseDelayMs = 500,
): Promise<Response> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const res = await fetchWithTimeout(url, init, timeoutMs);
    if (!RETRYABLE_STATUSES.has(res.status) || attempt === maxRetries) return res;
    const retryAfterHeader = res.headers.get('Retry-After');
    const delay = retryAfterHeader
      ? Number(retryAfterHeader) * 1000
      : baseDelayMs * Math.pow(2, attempt);
    await new Promise<void>((r) => setTimeout(r, delay));
  }
  // The loop above always returns before reaching this point because the
  // condition `attempt === maxRetries` forces an early return on the last
  // iteration. This throw satisfies the TypeScript exhaustive return check.
  throw new Error('retry loop exhausted');
}

/**
 * Parses the JSON body from a successful response and rejects null or undefined
 * bodies. A null or undefined body indicates an unexpected API contract violation
 * (the server returned HTTP 2xx but no payload). Throwing here gives callers a
 * clear error rather than silently forwarding undefined into typed code.
 */
async function parseJsonBody<T>(res: Response): Promise<T> {
  const data: unknown = await res.json();
  if (data === null || data === undefined) {
    throw new ApiError(res.status, "empty response body");
  }
  return data as T;
}

/**
 * Sends a GET request to the given API path and returns the parsed JSON response.
 * Retries on transient server errors using exponential backoff.
 * Throws ApiError for non-2xx responses or when the response body is empty.
 */
async function get<T>(path: string, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const res = await fetchWithRetry(`${BASE}${path}`, {}, timeoutMs);
  if (!res.ok) {
    const text = await res.text();
    throw new ApiError(res.status, text);
  }
  return parseJsonBody<T>(res);
}

/**
 * Sends a POST request with a JSON body to the given API path.
 * Retries on transient server errors using exponential backoff.
 * Throws ApiError for non-2xx responses or when the response body is empty.
 */
async function post<T>(path: string, body?: unknown, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const res = await fetchWithRetry(
    `${BASE}${path}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    },
    timeoutMs,
  );
  if (!res.ok) {
    const text = await res.text();
    throw new ApiError(res.status, text);
  }
  return parseJsonBody<T>(res);
}

/**
 * Sends a DELETE request to the given API path.
 * Retries on transient server errors using exponential backoff.
 * Throws ApiError for non-2xx responses or when the response body is empty.
 */
async function del<T>(path: string, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const res = await fetchWithRetry(`${BASE}${path}`, { method: "DELETE" }, timeoutMs);
  if (!res.ok) {
    const text = await res.text();
    throw new ApiError(res.status, text);
  }
  return parseJsonBody<T>(res);
}

/**
 * Typed API client with methods for every NeuronCite endpoint.
 * Methods are grouped by domain (health, sessions, search, jobs, etc.).
 *
 * Long-running endpoints (indexing, citation, annotation, model doctor,
 * source fetching, session optimize/rebuild) use the LONG_TIMEOUT_MS
 * value of 120 seconds. All other endpoints use the DEFAULT_TIMEOUT_MS
 * value of 30 seconds.
 */
export const api = {
  // ---- Health ----
  health: () => get<HealthResponse>("/health"),

  // ---- Sessions ----
  sessions: () => get<SessionListResponse>("/sessions"),
  deleteSession: (id: number) => del<SessionDeleteResponse>(`/sessions/${id}`),
  deleteSessionsByDirectory: (directory: string) =>
    post<{ api_version: string; deleted_session_ids: number[]; directory: string }>(
      "/sessions/delete-by-directory",
      { directory },
      LONG_TIMEOUT_MS,
    ),
  optimizeSession: (id: number) => post<OptimizeResponse>(`/sessions/${id}/optimize`, undefined, LONG_TIMEOUT_MS),
  rebuildIndex: (id: number) => post<RebuildResponse>(`/sessions/${id}/rebuild`, undefined, LONG_TIMEOUT_MS),

  // ---- Search ----
  search: (body: SearchRequest) => post<SearchResponse>("/search", body),
  hybridSearch: (body: SearchRequest) => post<SearchResponse>("/search/hybrid", body),
  searchMulti: (body: MultiSearchRequest) => post<MultiSearchResponse>("/search/multi", body),

  // ---- Indexing ----
  startIndex: (body: IndexRequest) => post<IndexResponse>("/index", body, LONG_TIMEOUT_MS),

  // ---- Jobs ----
  jobs: () => get<JobListResponse>("/jobs"),
  job: (id: string) => get<JobResponse>(`/jobs/${id}`),
  cancelJob: (id: string) => post<JobCancelResponse>(`/jobs/${id}/cancel`),

  // ---- Documents ----
  page: (fileId: number, pageNumber: number) =>
    get<PageResponse>(`/documents/${fileId}/pages/${pageNumber}`),

  // ---- Chunks ----
  chunks: (sessionId: number, fileId: number, params?: { page_number?: number; offset?: number; limit?: number }) => {
    const query = new URLSearchParams();
    if (params?.page_number !== undefined) query.set("page_number", String(params.page_number));
    if (params?.offset !== undefined) query.set("offset", String(params.offset));
    if (params?.limit !== undefined) query.set("limit", String(params.limit));
    const qs = query.toString();
    return get<ChunksResponse>(`/sessions/${sessionId}/files/${fileId}/chunks${qs ? `?${qs}` : ""}`);
  },

  // ---- Backends ----
  backends: () => get<BackendListResponse>("/backends"),

  // ---- Discover ----
  discover: (body: DiscoverRequest) => post<DiscoverResponse>("/discover", body),

  // ---- Web-specific endpoints ----
  browse: (path: string) => post<BrowseResponse>("/web/browse", { path } as BrowseRequest),
  scanDocuments: (path: string, sessionId?: number) =>
    post<ScanDocumentsResponse>("/web/scan-documents", { path, session_id: sessionId } as ScanDocumentsRequest, LONG_TIMEOUT_MS),
  config: () => get<ConfigResponse>("/web/config"),

  // ---- Native OS file/folder dialogs ----

  /** Opens a native OS file selection dialog. Returns the selected path and
   *  whether the user confirmed. An optional filter query parameter restricts
   *  the dialog to a specific file extension (e.g., "tex", "bib").
   *  Uses NATIVE_DIALOG_TIMEOUT_MS (no timeout) because the request blocks
   *  until the user selects a file or cancels the dialog. */
  browseNativeFile: (filter?: string) => {
    const qs = filter ? `?filter=${encodeURIComponent(filter)}` : "";
    return post<NativeBrowseResponse>(`/web/browse/native-file${qs}`, undefined, NATIVE_DIALOG_TIMEOUT_MS);
  },

  /** Opens a native OS folder selection dialog. Returns the selected path
   *  and whether the user confirmed. Uses NATIVE_DIALOG_TIMEOUT_MS (no
   *  timeout) because the request blocks until the user selects a folder
   *  or cancels the dialog. */
  browseNativeFolder: () =>
    post<NativeBrowseResponse>("/web/browse/native", undefined, NATIVE_DIALOG_TIMEOUT_MS),

  // ---- Setup (first-run welcome dialog) ----
  /** Checks whether this is the first launch by looking for the .setup_complete
   *  marker file. Returns the is_first_run flag and the data directory path. */
  setupStatus: () => get<SetupStatusResponse>("/web/setup/status"),

  /** Writes the .setup_complete marker file to suppress the welcome dialog
   *  on subsequent launches. */
  setupComplete: () => post<SetupCompleteResponse>("/web/setup/complete"),

  // ---- Model catalog and management ----
  /** Fetches the full model catalog including GPU detection, embedding models,
   *  and reranker models with their cache/active status. */
  modelCatalog: () => get<ModelCatalogResponse>("/web/models/catalog"),

  /** Triggers download of an embedding model to the local cache. The download
   *  runs asynchronously; progress updates arrive via SSE model events. */
  modelDownload: (body: ModelDownloadRequest) =>
    post<ModelDownloadResponse>("/web/models/download", body, LONG_TIMEOUT_MS),

  /** Activates a cached embedding model for use in indexing and search.
   *  The backend hot-swaps the model at runtime via ArcSwap. */
  modelActivate: (body: ModelActivateRequest) =>
    post<ModelActivateResponse>("/web/models/activate", body, LONG_TIMEOUT_MS),

  /** Loads a reranker model for cross-encoder reranking in search. The backend
   *  automatically downloads the model if not cached, verifies cache integrity,
   *  creates an ORT session, and hot-swaps it into the GPU worker. */
  loadReranker: (body: LoadRerankerRequest) =>
    post<LoadRerankerResponse>("/web/models/load-reranker", body, LONG_TIMEOUT_MS),

  // ---- Dependency Doctor (probes and auto-install) ----
  /** Fetches dependency probe results for pdfium, tesseract, ONNX Runtime,
   *  Ollama, and poppler. Each probe reports availability, version, and
   *  whether auto-installation is supported. */
  doctorProbes: () => get<DependencyProbe[]>("/web/doctor/probes"),

  /** Triggers auto-installation of a specific dependency. Supported dependency
   *  IDs: "pdfium", "tesseract", "onnxruntime". */
  doctorInstall: (body: DoctorInstallRequest) =>
    post<DoctorInstallResponse>("/web/doctor/install", body, LONG_TIMEOUT_MS),

  // ---- MCP Server (registration status and install/uninstall) ----
  /** Checks the MCP server registration status for both Claude Code and
   *  Claude Desktop App. */
  mcpStatus: () => get<McpStatusResponse>("/web/mcp/status"),

  /** Installs or uninstalls the MCP server registration for a specific target.
   *  The target is "claude-code" or "claude-desktop", the action is "install"
   *  or "uninstall". */
  mcpAction: (action: "install" | "uninstall", target: McpTarget) =>
    post<{ status: string }>(`/web/mcp/${target}/${action}`),

  // ---- Ollama (local LLM proxy and model management) ----
  ollamaStatus: (url: string) =>
    get<OllamaStatusResponse>(`/web/ollama/status?url=${encodeURIComponent(url)}`),
  ollamaModels: (url: string) =>
    get<OllamaModelsResponse>(`/web/ollama/models?url=${encodeURIComponent(url)}`),
  ollamaRunning: (url: string) =>
    get<OllamaRunningResponse>(`/web/ollama/running?url=${encodeURIComponent(url)}`),
  ollamaCatalog: () =>
    get<OllamaCatalogResponse>("/web/ollama/catalog"),
  ollamaPull: (model: string, url: string) =>
    post<OllamaPullResponse>("/web/ollama/pull", { model, url }, LONG_TIMEOUT_MS),
  ollamaDelete: (model: string, url: string) =>
    post<OllamaDeleteResponse>("/web/ollama/delete", { model, url }),
  ollamaShow: (model: string, url: string) =>
    post<OllamaShowResponse>("/web/ollama/show", { model, url }),

  // ---- Citation verification ----
  citationCreate: (body: CitationCreateRequest) =>
    post<CitationCreateResponse>("/citation/create", body, LONG_TIMEOUT_MS),
  citationAutoVerify: (jobId: string, body: AutoVerifyRequest) =>
    post<AutoVerifyResponse>(`/citation/${jobId}/auto-verify`, body, LONG_TIMEOUT_MS),
  citationStatus: (jobId: string) =>
    get<CitationStatusResponse>(`/citation/${jobId}/status`),
  citationRows: (jobId: string, limit?: number) => {
    const params = new URLSearchParams();
    if (limit !== undefined) params.set('limit', String(limit));
    const qs = params.toString();
    return get<CitationRowsResponse>(`/citation/${jobId}/rows${qs ? `?${qs}` : ''}`);
  },
  citationExport: (jobId: string, body: CitationExportRequest) =>
    post<{ api_version: string; status: string }>(`/citation/${jobId}/export`, body, LONG_TIMEOUT_MS),

  // ---- Source fetching (BibTeX-based DOI resolution and downloading) ----
  parseBib: (body: ParseBibRequest) =>
    post<ParseBibResponse>("/citation/parse-bib", body),
  bibReport: (body: BibReportRequest) =>
    post<BibReportResponse>("/citation/bib-report", body),
  fetchSources: (body: FetchSourcesRequest) =>
    post<FetchSourcesResponse>("/citation/fetch-sources", body, FETCH_SOURCES_TIMEOUT_MS),

  // ---- Annotation (standalone PDF highlighting from exported file) ----
  annotateFromFile: (body: AnnotateFromFileRequest) =>
    post<AnnotateFromFileResponse>("/annotate/from-file", body, LONG_TIMEOUT_MS),

  // ---- Model Doctor (diagnostics and repair) ----
  modelDoctor: () => get<ModelDoctorResponse>("/web/doctor/models", LONG_TIMEOUT_MS),
  repairModel: (modelId: string) =>
    post<{ status: string; model_id: string }>("/web/doctor/repair-model", { model_id: modelId }, LONG_TIMEOUT_MS),
};
