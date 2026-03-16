/**
 * SSE (Server-Sent Events) client for real-time event subscriptions.
 * Connects to the NeuronCite SSE endpoints and dispatches typed event
 * payloads to registered handlers.
 *
 * Each SSE connection uses exponential backoff reconnection: when the
 * connection drops (network error, server restart), the client waits
 * an increasing delay (1s, 2s, 4s, ..., up to 30s) before retrying.
 * A successful message reception resets the retry counter to zero.
 *
 * All incoming payloads are validated through runtime type guard functions
 * before being dispatched to handlers. Malformed events are logged as
 * warnings and discarded.
 *
 * Five SSE streams are available:
 *   - /api/v1/events/logs       -> "log" events
 *   - /api/v1/events/progress   -> "index_progress", "index_complete" events
 *   - /api/v1/events/jobs       -> "job_update" events
 *   - /api/v1/events/models     -> "model_download_progress", "model_switched",
 *                                  "model_downloaded", "reranker_loaded",
 *                                  "ollama_pull_progress", "ollama_pull_complete",
 *                                  "ollama_pull_error" events
 *   - /api/v1/events/citation   -> "citation_row_update", "citation_reasoning_token",
 *                                  "citation_job_progress" events
 */

/** Base path for SSE event endpoints. */
const SSE_BASE = "/api/v1/events";

// ---------------------------------------------------------------------------
// Module-level source entry handler for progress-stream multiplexing
// ---------------------------------------------------------------------------

/**
 * Module-level callback for source_entry_update events that arrive through
 * the progress SSE stream. The backend routes source fetch events through
 * progress_tx instead of a dedicated source_tx to stay within the browser's
 * HTTP/1.1 6-connection-per-host limit. With 5 permanent SSE streams (logs,
 * progress, jobs, models, citation) plus the fetch POST, opening a 6th SSE
 * stream for sources would exceed the limit and block the POST indefinitely.
 *
 * Null when no source fetch is in progress. Set by registerSourceEntryHandler
 * before api.fetchSources() is called, cleared by the returned cleanup
 * function when the fetch completes or the component unmounts.
 */
let _sourceEntryHandler: ((data: {
  cite_key: string;
  url: string;
  type: string;
  status: string;
  file_path?: string;
  cache_path?: string;
  title?: string;
  error?: string;
  reason?: string;
  doi_resolved_via?: string;
  doi_done?: number;
  doi_total?: number;
}) => void) | null = null;

/**
 * Registers a handler for source_entry_update events that arrive through the
 * progress SSE stream. Call this before firing api.fetchSources() to ensure
 * all per-entry events are received. Returns a cleanup function that
 * unregisters the handler; call it in the fetch finally block or onCleanup.
 *
 * @param handler - Callback for each validated source entry event
 * @returns Cleanup function that clears the handler
 */
export function registerSourceEntryHandler(
  handler: NonNullable<typeof _sourceEntryHandler>,
): () => void {
  _sourceEntryHandler = handler;
  return () => {
    _sourceEntryHandler = null;
  };
}

// ---------------------------------------------------------------------------
// Runtime type guards
// ---------------------------------------------------------------------------

/**
 * Attempts to parse a JSON string and returns the parsed value on success,
 * or null if the string is not valid JSON. Used as the first validation
 * step for every incoming SSE event payload.
 */
function parseJson(data: string): unknown | null {
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
}

/**
 * Checks whether the value is a non-null object. This is the baseline
 * predicate used by all specific type guards below.
 */
function isObject(d: unknown): d is Record<string, unknown> {
  return typeof d === "object" && d !== null;
}

/**
 * Type guard for log event payloads. The Rust BroadcastLayer sends objects
 * with `level`, `target`, and `message` string fields. All three must be
 * present for the payload to be considered valid.
 */
function isLogPayload(d: unknown): d is { level: string; target: string; message: string } {
  return isObject(d) && "level" in d && "target" in d && "message" in d;
}

/**
 * Type guard for index_progress event payloads. The backend emits progress
 * objects with `phase` (string), `files_total`, `files_done`, and
 * `chunks_created` (all numbers).
 */
function isIndexProgress(d: unknown): d is {
  phase: string;
  files_total: number;
  files_done: number;
  chunks_created: number;
} {
  return (
    isObject(d) &&
    "files_total" in d &&
    typeof d.files_total === "number" &&
    "files_done" in d &&
    typeof d.files_done === "number" &&
    "chunks_created" in d &&
    typeof d.chunks_created === "number"
  );
}

/**
 * Type guard for job_update event payloads. Each job event carries the
 * job ID, state label, and progress counters.
 */
function isJobUpdate(d: unknown): d is {
  job_id: string;
  state: string;
  progress_done: number;
  progress_total: number;
  error_message: string | null;
} {
  return (
    isObject(d) &&
    "job_id" in d &&
    typeof d.job_id === "string" &&
    "state" in d &&
    typeof d.state === "string" &&
    "progress_done" in d &&
    typeof d.progress_done === "number" &&
    "progress_total" in d &&
    typeof d.progress_total === "number"
  );
}

/**
 * Type guard for model_download_progress event payloads. Sent during
 * HuggingFace model downloads with byte counts.
 */
function isDownloadProgress(d: unknown): d is {
  model_id: string;
  bytes_downloaded: number;
  bytes_total: number;
} {
  return (
    isObject(d) &&
    "model_id" in d &&
    typeof d.model_id === "string" &&
    "bytes_downloaded" in d &&
    typeof d.bytes_downloaded === "number" &&
    "bytes_total" in d &&
    typeof d.bytes_total === "number"
  );
}

/**
 * Type guard for model_switched event payloads. Sent after the embedding
 * worker loads a different model and the vector dimension changes.
 */
function isModelSwitched(d: unknown): d is { model_id: string; vector_dimension: number } {
  return (
    isObject(d) &&
    "model_id" in d &&
    typeof d.model_id === "string" &&
    "vector_dimension" in d &&
    typeof d.vector_dimension === "number"
  );
}

/**
 * Type guard for events that carry only a `model_id` string field.
 * Used for "model_downloaded" and "reranker_loaded" events.
 */
function hasModelId(d: unknown): d is { model_id: string } {
  return isObject(d) && "model_id" in d && typeof d.model_id === "string";
}

/**
 * Type guard for ollama_pull_progress event payloads. The `total` and
 * `completed` fields are nullable because Ollama does not report byte
 * counts for all layers (e.g. manifests).
 */
function isOllamaPullProgress(d: unknown): d is {
  model: string;
  status: string;
  total: number | null;
  completed: number | null;
} {
  return (
    isObject(d) &&
    "model" in d &&
    typeof d.model === "string" &&
    "status" in d &&
    typeof d.status === "string"
  );
}

/**
 * Type guard for events that carry only a `model` string field.
 * Used for "ollama_pull_complete" events.
 */
function hasOllamaModel(d: unknown): d is { model: string } {
  return isObject(d) && "model" in d && typeof d.model === "string";
}

/**
 * Type guard for ollama_pull_error event payloads. Requires both
 * the `model` and `error` string fields.
 */
function isOllamaPullError(d: unknown): d is { model: string; error: string } {
  return (
    isObject(d) &&
    "model" in d &&
    typeof d.model === "string" &&
    "error" in d &&
    typeof d.error === "string"
  );
}

/**
 * Type guard for citation_row_update event payloads. The row update
 * carries job context, row identification, phase state, and optional
 * verdict/reasoning fields that are present only after evaluation.
 */
function isCitationRowUpdate(d: unknown): d is {
  job_id: string;
  row_id: number;
  cite_key: string;
  phase: string;
  verdict?: string;
  confidence?: number;
  reasoning?: string;
  search_queries?: string[];
  error_message?: string;
} {
  return (
    isObject(d) &&
    "job_id" in d &&
    typeof d.job_id === "string" &&
    "row_id" in d &&
    typeof d.row_id === "number" &&
    "cite_key" in d &&
    typeof d.cite_key === "string" &&
    "phase" in d &&
    typeof d.phase === "string"
  );
}

/**
 * Type guard for citation_reasoning_token event payloads. Each token
 * event carries the job context, the target row, and a single string
 * token from the LLM streaming output.
 */
function isCitationReasoningToken(d: unknown): d is {
  job_id: string;
  row_id: number;
  token: string;
} {
  return (
    isObject(d) &&
    "job_id" in d &&
    typeof d.job_id === "string" &&
    "row_id" in d &&
    typeof d.row_id === "number" &&
    "token" in d &&
    typeof d.token === "string"
  );
}

/**
 * Type guard for citation_job_progress event payloads. The progress
 * event carries aggregate counters and a verdict distribution map.
 */
function isCitationJobProgress(d: unknown): d is {
  job_id: string;
  rows_done: number;
  rows_total: number;
  verdicts: Record<string, number>;
} {
  return (
    isObject(d) &&
    "job_id" in d &&
    typeof d.job_id === "string" &&
    "rows_done" in d &&
    typeof d.rows_done === "number" &&
    "rows_total" in d &&
    typeof d.rows_total === "number" &&
    "verdicts" in d &&
    isObject(d.verdicts)
  );
}

/**
 * Type guard for source_entry_update SSE event payloads. Each entry carries
 * the cite_key, URL, type classification, and download status from the BibTeX
 * source fetching pipeline. Only the required discriminating fields are checked;
 * optional fields (file_path, title, error, etc.) are structurally present but
 * not verified to avoid duplicating the full FetchSourceResult interface here.
 */
function isSourceEntryUpdate(d: unknown): d is {
  cite_key: string;
  url: string;
  type: string;
  status: string;
  file_path?: string;
  cache_path?: string;
  title?: string;
  error?: string;
  reason?: string;
  doi_resolved_via?: string;
} {
  return (
    isObject(d) &&
    "cite_key" in d &&
    typeof d.cite_key === "string" &&
    "status" in d &&
    typeof d.status === "string"
  );
}

// ---------------------------------------------------------------------------
// Resilient SSE connection with exponential backoff
// ---------------------------------------------------------------------------

/**
 * Options for the resilient SSE connection factory.
 *
 * setupListeners is called each time a new EventSource is created. It attaches
 * named event listeners and receives the resetRetry callback so listeners can
 * reset the backoff counter when a message is successfully received.
 *
 * onDisconnect is an optional callback invoked in the onerror handler before
 * the reconnect timer is scheduled. Callers can use this to update UI state
 * or log the disconnection event independently of the automatic reconnection.
 *
 * onOpen is an optional callback invoked each time the EventSource connection
 * opens (initial connection and every reconnect after an error). The server-side
 * broadcast subscriber is created before the HTTP 200 response headers are sent,
 * so onOpen firing guarantees the subscriber is active and no subsequent events
 * will be missed. Callers that need to coordinate startup (e.g., SourcesTab
 * waiting to POST until the SSE stream is ready) use this to avoid the race
 * where events are broadcast before the subscriber connects.
 */
interface ResilientSSEOptions {
  setupListeners: (es: EventSource, resetRetry: () => void) => void;
  onDisconnect?: () => void;
  onOpen?: () => void;
}

/**
 * Creates an EventSource connection to the given URL with automatic
 * reconnection using exponential backoff. On connection error, the
 * client closes the broken EventSource and schedules a reconnection
 * attempt after an increasing delay: 1s, 2s, 4s, 8s, ..., capped at
 * 30 seconds. A successfully received message resets the retry counter
 * to zero so that transient errors do not permanently increase latency.
 *
 * Returns a cleanup function that closes the EventSource and cancels
 * any pending reconnection timer. Callers should invoke this cleanup
 * in SolidJS onCleanup or equivalent lifecycle hooks.
 *
 * @param url - Full URL of the SSE endpoint (e.g., "/api/v1/events/logs")
 * @param options - Configuration with setupListeners callback and optional onDisconnect
 * @returns Cleanup function that tears down the connection
 */
function createResilientSSE(
  url: string,
  options: ResilientSSEOptions | ((es: EventSource, resetRetry: () => void) => void),
): () => void {
  // Accept either the options object form or the legacy callback form so
  // existing internal callers do not need to be restructured. The legacy form
  // (passing a plain function) does not support onOpen or onDisconnect.
  const setupListeners = typeof options === 'function' ? options : options.setupListeners;
  const onDisconnect = typeof options === 'function' ? undefined : options.onDisconnect;
  const onOpen = typeof options === 'function' ? undefined : options.onOpen;

  let es: EventSource | null = null;
  let retryCount = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let disposed = false;

  function connect(): void {
    if (disposed) return;

    es = new EventSource(url);

    const resetRetry = (): void => {
      retryCount = 0;
    };

    // Notify the caller each time the connection opens. The server-side
    // broadcast subscriber is created during the GET request handling,
    // before the 200 response headers are sent. By the time onopen fires
    // in the browser, the subscriber is already active and will receive
    // all subsequent events without missing any.
    if (onOpen) {
      es.onopen = onOpen;
    }

    setupListeners(es, resetRetry);

    es.onerror = () => {
      es?.close();
      es = null;
      if (disposed) return;

      // Notify the caller that the connection was lost before scheduling
      // the reconnect. This allows callers to update UI or log the event.
      onDisconnect?.();

      /** Delay in milliseconds, doubling each attempt, capped at 30 seconds. */
      const delay = Math.min(1000 * Math.pow(2, retryCount), 30000);
      retryCount++;
      console.warn(
        `SSE connection to ${url} lost. Reconnecting in ${delay}ms (attempt ${retryCount}).`,
      );
      timer = setTimeout(connect, delay);
    };
  }

  connect();

  return () => {
    disposed = true;
    es?.close();
    es = null;
    if (timer !== undefined) {
      clearTimeout(timer);
      timer = undefined;
    }
  };
}

// ---------------------------------------------------------------------------
// Core subscription function
// ---------------------------------------------------------------------------

/**
 * Subscribes to an SSE stream and dispatches validated JSON payloads to
 * the provided event handlers. Uses exponential backoff reconnection
 * for resilience against network interruptions and server restarts.
 *
 * Each handler receives `unknown` data; callers are responsible for
 * narrowing the type (typically via the specific type guards above).
 * If JSON parsing fails for a message, a warning is logged and the
 * message is discarded.
 *
 * Returns a cleanup function that closes the EventSource and cancels
 * any pending reconnection (for use with SolidJS onCleanup).
 *
 * @param endpoint - SSE stream name (e.g., "logs", "progress", "jobs", "models")
 * @param handlers - Map of event type names to handler functions
 * @returns Cleanup function that closes the EventSource
 */
export function subscribeSSE(
  endpoint: string,
  handlers: Record<string, (data: unknown) => void>,
  onOpen?: () => void,
): () => void {
  const url = `${SSE_BASE}/${endpoint}`;

  return createResilientSSE(url, {
    setupListeners: (es, resetRetry) => {
      for (const [eventType, handler] of Object.entries(handlers)) {
        es.addEventListener(eventType, ((event: MessageEvent) => {
          // Reset the retry counter on any received message, including those
          // that cannot be parsed. A message arriving at all confirms the
          // connection is alive, so the backoff counter should reset regardless
          // of whether the payload is well-formed JSON.
          resetRetry();
          const data = parseJson(event.data);
          if (data === null) {
            console.warn(
              `SSE [${endpoint}/${eventType}]: received malformed JSON, discarding event.`,
            );
            return;
          }
          handler(data);
        }) as EventListener);
      }
    },
    onOpen,
  });
}

// ---------------------------------------------------------------------------
// Typed subscription helpers
// ---------------------------------------------------------------------------

/**
 * Subscribes to the log stream. Each "log" SSE event carries a JSON object
 * with `level`, `target`, and `message` fields from the Rust BroadcastLayer.
 * This function validates the payload shape using the isLogPayload type guard
 * and passes a structured LogEntry to the callback. A client-side ISO
 * timestamp is attached when the event arrives.
 *
 * Plain-string payloads are accepted as a fallback (wrapped as INFO messages)
 * to handle edge cases from non-standard loggers.
 *
 * @param onLog - Callback invoked with each validated log entry
 * @returns Cleanup function that closes the EventSource
 */
export function subscribeToLogs(
  onLog: (entry: { level: string; target: string; message: string; timestamp: string }) => void,
): () => void {
  return subscribeSSE("logs", {
    log: (data) => {
      const now = new Date().toISOString();
      if (isLogPayload(data)) {
        onLog({
          level: (data.level || "INFO").toUpperCase(),
          target: data.target || "",
          message: data.message || "",
          timestamp: now,
        });
      } else if (typeof data === "string") {
        // Fallback for plain-string payloads (should not occur with current
        // BroadcastLayer, but handles edge cases gracefully).
        onLog({ level: "INFO", target: "", message: data, timestamp: now });
      } else {
        console.warn("SSE [logs/log]: payload failed type guard, discarding.", data);
      }
    },
  });
}

/**
 * Subscribes to the indexing progress stream. Receives "index_progress"
 * events during indexing and "index_complete" when indexing finishes.
 * The index_progress payload is validated via the isIndexProgress type guard;
 * malformed payloads are logged and discarded.
 *
 * @param onProgress - Callback for progress updates
 * @param onComplete - Callback when indexing completes
 * @returns Cleanup function that closes the EventSource
 */
export function subscribeToProgress(
  onProgress: (data: {
    phase: string;
    files_total: number;
    files_done: number;
    chunks_created: number;
  }) => void,
  onComplete: () => void,
): () => void {
  return subscribeSSE("progress", {
    index_progress: (data) => {
      if (isIndexProgress(data)) {
        onProgress(data);
      } else {
        console.warn("SSE [progress/index_progress]: payload failed type guard, discarding.", data);
      }
    },
    index_complete: () => onComplete(),
    // Source fetch events are routed through progress_tx on the backend to
    // avoid opening a 6th SSE connection (which would exceed the HTTP/1.1
    // 6-connection-per-host limit and block the fetch POST). The handler
    // dispatches to _sourceEntryHandler when a source fetch is in progress.
    source_entry_update: (data) => {
      if (isSourceEntryUpdate(data) && _sourceEntryHandler) {
        _sourceEntryHandler(data);
      }
    },
  });
}

/**
 * Subscribes to the job status stream. Each "job_update" event carries
 * the job ID, state, and progress counters. Payloads are validated via
 * the isJobUpdate type guard.
 *
 * @param onUpdate - Callback for job status updates
 * @returns Cleanup function that closes the EventSource
 */
export function subscribeToJobs(
  onUpdate: (data: {
    job_id: string;
    state: string;
    progress_done: number;
    progress_total: number;
    error_message: string | null;
  }) => void,
): () => void {
  return subscribeSSE("jobs", {
    job_update: (data) => {
      if (isJobUpdate(data)) {
        onUpdate(data);
      } else {
        console.warn("SSE [jobs/job_update]: payload failed type guard, discarding.", data);
      }
    },
  });
}

/**
 * Subscribes to the model events stream. Receives download progress,
 * model switch confirmations, download completions, reranker load events,
 * and Ollama pull lifecycle events. Each event type is validated through
 * its specific type guard before being dispatched to the handler.
 *
 * @param handlers - Object with optional callbacks for each model event type
 * @returns Cleanup function that closes the EventSource
 */
export function subscribeToModels(handlers: {
  onDownloadProgress?: (data: { model_id: string; bytes_downloaded: number; bytes_total: number }) => void;
  onModelSwitched?: (data: { model_id: string; vector_dimension: number }) => void;
  onModelDownloaded?: (data: { model_id: string }) => void;
  onRerankerLoaded?: (data: { model_id: string }) => void;
  /** Ollama model pull progress with layer download byte counts. */
  onOllamaPullProgress?: (data: { model: string; status: string; total: number | null; completed: number | null }) => void;
  /** Ollama model pull completed. */
  onOllamaPullComplete?: (data: { model: string }) => void;
  /** Ollama model pull failed with an error message. */
  onOllamaPullError?: (data: { model: string; error: string }) => void;
}): () => void {
  const eventHandlers: Record<string, (data: unknown) => void> = {};

  if (handlers.onDownloadProgress) {
    const cb = handlers.onDownloadProgress;
    eventHandlers["model_download_progress"] = (data) => {
      if (isDownloadProgress(data)) {
        cb(data);
      } else {
        console.warn("SSE [models/model_download_progress]: payload failed type guard, discarding.", data);
      }
    };
  }
  if (handlers.onModelSwitched) {
    const cb = handlers.onModelSwitched;
    eventHandlers["model_switched"] = (data) => {
      if (isModelSwitched(data)) {
        cb(data);
      } else {
        console.warn("SSE [models/model_switched]: payload failed type guard, discarding.", data);
      }
    };
  }
  if (handlers.onModelDownloaded) {
    const cb = handlers.onModelDownloaded;
    eventHandlers["model_downloaded"] = (data) => {
      if (hasModelId(data)) {
        cb(data);
      } else {
        console.warn("SSE [models/model_downloaded]: payload failed type guard, discarding.", data);
      }
    };
  }
  if (handlers.onRerankerLoaded) {
    const cb = handlers.onRerankerLoaded;
    eventHandlers["reranker_loaded"] = (data) => {
      if (hasModelId(data)) {
        cb(data);
      } else {
        console.warn("SSE [models/reranker_loaded]: payload failed type guard, discarding.", data);
      }
    };
  }
  if (handlers.onOllamaPullProgress) {
    const cb = handlers.onOllamaPullProgress;
    eventHandlers["ollama_pull_progress"] = (data) => {
      if (isOllamaPullProgress(data)) {
        cb(data);
      } else {
        console.warn("SSE [models/ollama_pull_progress]: payload failed type guard, discarding.", data);
      }
    };
  }
  if (handlers.onOllamaPullComplete) {
    const cb = handlers.onOllamaPullComplete;
    eventHandlers["ollama_pull_complete"] = (data) => {
      if (hasOllamaModel(data)) {
        cb(data);
      } else {
        console.warn("SSE [models/ollama_pull_complete]: payload failed type guard, discarding.", data);
      }
    };
  }
  if (handlers.onOllamaPullError) {
    const cb = handlers.onOllamaPullError;
    eventHandlers["ollama_pull_error"] = (data) => {
      if (isOllamaPullError(data)) {
        cb(data);
      } else {
        console.warn("SSE [models/ollama_pull_error]: payload failed type guard, discarding.", data);
      }
    };
  }

  return subscribeSSE("models", eventHandlers);
}

/**
 * Subscribes to the source fetching event stream. Receives source_entry_update
 * events from the fetch_sources handler as each BibTeX entry is processed.
 * Each event carries the cite_key, URL, type classification, and download
 * status. DOI resolution progress events (status "resolving" / "resolved") and
 * download completion events flow through the same stream.
 *
 * Payloads are validated via the isSourceEntryUpdate type guard before being
 * dispatched to the handler. Invalid payloads are logged and discarded.
 *
 * @param onSourceUpdate - Callback invoked for each validated source entry event
 * @returns Cleanup function that closes the EventSource
 */
export function subscribeToSources(
  onSourceUpdate: (data: {
    cite_key: string;
    url: string;
    type: string;
    status: string;
    file_path?: string;
    cache_path?: string;
    title?: string;
    error?: string;
    reason?: string;
    doi_resolved_via?: string;
  }) => void,
  onOpen?: () => void,
): () => void {
  return subscribeSSE(
    "sources",
    {
      source_entry_update: (data) => {
        if (isSourceEntryUpdate(data)) {
          onSourceUpdate(data);
        } else {
          console.warn(
            "SSE [sources/source_entry_update]: payload failed type guard, discarding.",
            data,
          );
        }
      },
    },
    onOpen,
  );
}

/**
 * Subscribes to the citation verification event stream. Receives three
 * event types from the autonomous citation agent:
 *
 * - "citation_row_update" -- Per-row phase transitions (searching, evaluating,
 *   done, error) with verdict, confidence, and reasoning summary.
 * - "citation_reasoning_token" -- Individual tokens during LLM streaming for
 *   live typewriter display.
 * - "citation_job_progress" -- Aggregate job progress after each row completes.
 *
 * Each event type is validated through its specific type guard. Payloads
 * that fail validation are logged as warnings and discarded.
 *
 * @param handlers - Callbacks for each citation event type
 * @returns Cleanup function that closes the EventSource
 */
export function subscribeToCitation(handlers: {
  onRowUpdate: (data: {
    job_id: string;
    row_id: number;
    cite_key: string;
    phase: string;
    verdict?: string;
    confidence?: number;
    reasoning?: string;
    search_queries?: string[];
    error_message?: string;
  }) => void;
  onReasoningToken: (data: { job_id: string; row_id: number; token: string }) => void;
  onJobProgress: (data: {
    job_id: string;
    rows_done: number;
    rows_total: number;
    verdicts: Record<string, number>;
  }) => void;
}): () => void {
  return subscribeSSE("citation", {
    citation_row_update: (data) => {
      if (isCitationRowUpdate(data)) {
        handlers.onRowUpdate(data);
      } else {
        console.warn("SSE [citation/citation_row_update]: payload failed type guard, discarding.", data);
      }
    },
    citation_reasoning_token: (data) => {
      if (isCitationReasoningToken(data)) {
        handlers.onReasoningToken(data);
      } else {
        console.warn("SSE [citation/citation_reasoning_token]: payload failed type guard, discarding.", data);
      }
    },
    citation_job_progress: (data) => {
      if (isCitationJobProgress(data)) {
        handlers.onJobProgress(data);
      } else {
        console.warn("SSE [citation/citation_job_progress]: payload failed type guard, discarding.", data);
      }
    },
  });
}
