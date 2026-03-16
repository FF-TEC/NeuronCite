import { Component, For, Show, createMemo, createSignal, onMount, onCleanup } from "solid-js";
import { state, actions, safeLsGet, safeLsSet } from "../../stores/app";
import { api } from "../../api/client";
import { logWarn } from "../../utils/logger";
import { clickableProps } from "../../utils/a11y";
import type { SearchResultDto } from "../../api/types";
import SplitPane from "./SplitPane";
import ChunkViewer from "./ChunkViewer";
import Tip from "../ui/Tip";

/** localStorage key for persisting the search query textarea height. */
const LS_QUERY_HEIGHT = "neuroncite_query_height";

/**
 * Search tab containing a resizable split pane. The left panel holds the
 * session selector (multi-select checkbox list), search input, toggle chips
 * for search modes, and the search button. The right panel displays ranked
 * search results with export controls for Markdown, BibTeX, and CSL-JSON.
 *
 * Session selection supports 1-10 sessions. When one session is selected,
 * the standard /search endpoint is used. When two or more are selected,
 * the /search/multi endpoint merges results from all sessions into a
 * single ranked list with session_id tags on each result.
 *
 * Sessions with a vector dimension that differs from the loaded model
 * are grayed out and cannot be selected (dimension mismatch).
 *
 * The reranker toggle checks store.rerankerAvailable before enabling.
 * Export buttons copy formatted text to the clipboard with feedback.
 */
const SearchTab: Component = () => {
  /** Error message from the last failed search request, displayed below
   *  the search button until the next search attempt clears it. */
  const [searchError, setSearchError] = createSignal<string | null>(null);

  /** Tracks which export button was last clicked for brief copy feedback.
   *  Values: "md" | "bibtex" | "csl" | null. Resets after 1.5 seconds. */
  const [copiedLabel, setCopiedLabel] = createSignal<string | null>(null);

  /** Shown in the Search Modes card when the user tries to enable the
   *  reranker toggle but no reranker model is loaded. Cleared when the
   *  user successfully toggles reranker off or loads a model. */
  const [rerankerWarning, setRerankerWarning] = createSignal(false);

  /** When set, the right panel shows the ChunkViewer for this result
   *  instead of the results list. Cleared by the "Back to Results" button
   *  inside ChunkViewer. */
  const [openedResult, setOpenedResult] = createSignal<SearchResultDto | null>(null);

  /** Timer ID for the copiedLabel auto-clear timeout. Stored so that
   *  onCleanup can cancel a pending timer if the component unmounts
   *  while the "Copied!" feedback is still visible. */
  let copiedTimer: ReturnType<typeof setTimeout> | undefined;

  onCleanup(() => {
    if (copiedTimer !== undefined) clearTimeout(copiedTimer);
  });

  /** Whether multiple sessions are selected (triggers multi-search mode). */
  const isMultiSearch = () => state.searchSessionIds.length > 1;

  /** Whether at least one session is selected (enables search). */
  const hasSelection = () => state.searchSessionIds.length > 0;

  /** Returns a map from session_id to session directory basename for
   *  displaying session badges on multi-search results. */
  const sessionLabelMap = createMemo(() => {
    const map = new Map<number, string>();
    for (const s of state.sessions) {
      const dirName = s.directory_path.split(/[/\\]/).filter(Boolean).pop() || s.directory_path;
      map.set(s.id, `#${s.id} ${dirName}`);
    }
    return map;
  });

  /** Executes a search against the selected session(s) using the current
   *  query and search mode settings from the store. Routes to /search for
   *  single-session or /search/multi for multi-session. Displays API error
   *  messages to the user instead of silently swallowing them. */
  const executeSearch = async () => {
    if (!state.queryText.trim() || !hasSelection()) return;

    setSearchError(null);
    actions.setSearchInProgress(true);
    try {
      if (isMultiSearch()) {
        // Multi-session search: merges results from 2+ sessions
        const res = await api.searchMulti({
          query: state.queryText,
          session_ids: state.searchSessionIds,
          use_fts: state.useHybrid,
          rerank: state.useReranker,
          refine: state.useRefine,
          refine_divisors: state.useRefine ? state.refineDivisors : undefined,
        });
        actions.setSearchResults(res.results);
      } else {
        // Single-session search: uses the standard endpoint
        const res = await api.search({
          query: state.queryText,
          session_id: state.searchSessionIds[0],
          use_fts: state.useHybrid,
          rerank: state.useReranker,
          refine: state.useRefine,
          refine_divisors: state.useRefine ? state.refineDivisors : undefined,
        });
        actions.setSearchResults(res.results);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setSearchError(msg);
      actions.setSearchResults([]);
    } finally {
      actions.setSearchInProgress(false);
    }
  };

  /** Handles Enter key press in the search input. */
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      executeSearch();
    }
  };

  /** Groups results by source file for the grouped view. Returns an array of
   *  groups sorted by best score descending. */
  const groupedResults = createMemo(() => {
    if (!state.groupByDocument) return null;
    const groups = new Map<string, SearchResultDto[]>();
    for (const r of state.searchResults) {
      const key = r.source_file;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(r);
    }
    return Array.from(groups.entries())
      .map(([file, results]) => ({
        file,
        results,
        bestScore: Math.max(...results.map((r) => r.score)),
      }))
      .sort((a, b) => b.bestScore - a.bestScore);
  });

  // ---- Clipboard export with feedback ----

  /** Shows a brief "Copied!" indicator next to the export button that was
   *  clicked. The label auto-clears after 1.5 seconds. The previous timer
   *  is cancelled if a different button is clicked before the timeout. */
  const flashCopied = (label: string) => {
    if (copiedTimer !== undefined) clearTimeout(copiedTimer);
    setCopiedLabel(label);
    copiedTimer = setTimeout(() => {
      setCopiedLabel(null);
      copiedTimer = undefined;
    }, 1500);
  };

  /** Copies text to the clipboard and flashes feedback on the button.
   *  Catches clipboard API errors (e.g. missing permissions) and shows
   *  a brief "Failed" indicator instead. */
  const copyToClipboard = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      flashCopied(label);
    } catch {
      setCopiedLabel(null);
      setSearchError("Clipboard write failed. Check browser permissions.");
    }
  };

  /** Exports all results as Markdown to the clipboard. Each result is
   *  formatted as a blockquote with citation and relevance score. */
  const exportMarkdown = () => {
    const md = state.searchResults
      .map(
        (r, i) =>
          `### Result ${i + 1}\n\n> ${r.content.slice(0, 300)}${r.content.length > 300 ? "..." : ""}\n>\n> -- ${r.citation} (score: ${r.score.toFixed(3)})`,
      )
      .join("\n\n---\n\n");
    copyToClipboard(md, "md");
  };

  /** Exports all results as BibTeX entries to the clipboard. Generates
   *  safe citation keys from source file names with index suffixes. */
  const exportBibtex = () => {
    const entries = state.searchResults.map((r, i) => {
      const key = r.source_file.replace(/[^a-zA-Z0-9]/g, "_").slice(0, 30) + `_${i}`;
      return `@misc{${key},\n  title = {${r.citation}},\n  note = {Score: ${r.score.toFixed(3)}, Pages ${r.page_start}-${r.page_end}},\n  howpublished = {${r.source_file}}\n}`;
    });
    copyToClipboard(entries.join("\n\n"), "bibtex");
  };

  /** Exports all results as CSL-JSON to the clipboard. Maps citation
   *  data to standard Citation Style Language fields. */
  const exportCslJson = () => {
    const items = state.searchResults.map((r, i) => ({
      id: `result-${i}`,
      type: "article" as const,
      title: r.citation,
      source: r.source_file,
      page: `${r.page_start}-${r.page_end}`,
      note: `Score: ${r.score.toFixed(3)}`,
    }));
    copyToClipboard(JSON.stringify(items, null, 2), "csl");
  };

  /** Formats a session for display in the checkbox list. Shows the directory
   *  basename, model name shorthand, dimension, and chunk count. */
  const sessionLabel = (s: { id: number; directory_path: string; model_name: string; total_chunks: number; vector_dimension: number }) => {
    const dirName = s.directory_path.split(/[/\\]/).filter(Boolean).pop() || s.directory_path;
    const modelShort = s.model_name.split("/").pop() || s.model_name;
    return `#${s.id} ${dirName} (${modelShort}, ${s.vector_dimension}d, ${s.total_chunks} chunks)`;
  };

  /** Whether a session has a matching vector dimension with the loaded model. */
  const isDimensionCompatible = (session: { vector_dimension: number }) =>
    state.activeModelDimension === 0 || session.vector_dimension === state.activeModelDimension;

  /** Handles the Reranker toggle click. When the user tries to enable
   *  reranking but no reranker model is loaded, keeps the toggle off
   *  and shows a warning inline in the Search Modes card. When toggling
   *  off or when a reranker becomes available, clears the warning. */
  const toggleReranker = () => {
    if (!state.useReranker && !state.rerankerAvailable) {
      setRerankerWarning(true);
      return;
    }
    setRerankerWarning(false);
    actions.setUseReranker(!state.useReranker);
  };

  // ---- Horizontal resize bar for the query textarea ----

  /** Height of the query textarea in pixels. Persisted to localStorage
   *  so the user's preferred size survives page reloads. Default: 80px. */
  const restoredHeight = parseInt(safeLsGet(LS_QUERY_HEIGHT) || "80", 10);
  const [queryHeight, setQueryHeight] = createSignal(isNaN(restoredHeight) ? 80 : restoredHeight);
  const [hDragging, setHDragging] = createSignal(false);

  /** Pointer Y coordinate at drag start. Used together with dragStartHeight
   *  to compute a delta-based height change, avoiding any absolute offset
   *  calculation that would cause the textarea to jump on first click. */
  let dragStartY = 0;
  /** Textarea height at drag start. The final height is dragStartHeight
   *  plus the vertical distance the pointer has moved since drag start. */
  let dragStartHeight = 0;

  const onHPointerMove = (e: PointerEvent) => {
    if (!hDragging()) return;
    e.preventDefault();
    const delta = e.clientY - dragStartY;
    setQueryHeight(Math.max(42, Math.min(600, dragStartHeight + delta)));
  };

  const onHPointerUp = () => {
    if (!hDragging()) return;
    setHDragging(false);
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
    // Persist the final height to localStorage
    safeLsSet(LS_QUERY_HEIGHT, String(queryHeight()));
  };

  const onHDividerPointerDown = (e: PointerEvent) => {
    e.preventDefault();
    dragStartY = e.clientY;
    dragStartHeight = queryHeight();
    setHDragging(true);
    document.body.style.userSelect = "none";
    document.body.style.cursor = "row-resize";
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  onMount(() => {
    document.addEventListener("pointermove", onHPointerMove);
    document.addEventListener("pointerup", onHPointerUp);
  });

  onCleanup(() => {
    document.removeEventListener("pointermove", onHPointerMove);
    document.removeEventListener("pointerup", onHPointerUp);
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
  });

  /** Selects all compatible sessions at once. */
  const selectAllSessions = () => {
    const compatible = state.sessions
      .filter((s) => isDimensionCompatible(s))
      .map((s) => s.id);
    actions.setSearchSessionIds(compatible);
  };

  /** Deselects all sessions. */
  const deselectAllSessions = () => {
    actions.setSearchSessionIds([]);
  };

  /** Returns the session ID to use for chunk fetching when opening a result.
   *  In multi-search mode, the result carries its own session_id. In single-
   *  session mode, the sole selected session is used. */
  const sessionIdForResult = (result: SearchResultDto): number => {
    if (result.session_id !== undefined && result.session_id !== null) {
      return result.session_id;
    }
    return state.searchSessionIds[0];
  };

  // ---- Left panel: search settings (arrow function for SolidJS reactivity) ----
  const leftPanel = () => (
    <div style={{ display: "flex", "flex-direction": "column", gap: "16px", height: "100%", "overflow-y": "auto", "min-height": "0" }}>
      {/* Session selector: multi-select checkbox list */}
      <div class="glass-card" style={{ padding: "16px" }}>
        <div class="section-title">
          Sessions
          {isMultiSearch() && (
            <span style={{ "font-size": "10px", color: "var(--color-accent-purple)", "margin-left": "6px" }}>
              multi ({state.searchSessionIds.length})
            </span>
          )}
          <Tip text="Select one or more sessions to search against. When multiple sessions are selected, results are merged into a single ranked list. Each session stores the vector index for a specific folder and embedding model combination. Sessions with a different vector dimension than the loaded model are grayed out because the query vector would be incompatible." />
        </div>
        <Show
          when={state.sessions.length > 0}
          fallback={
            <div style={{ "font-size": "12px", color: "var(--color-text-muted)", padding: "8px 0" }}>
              No sessions. Index a folder first.
            </div>
          }
        >
          {/* Select all / Deselect all controls */}
          <div style={{ display: "flex", gap: "8px", "margin-bottom": "6px" }}>
            <button
              class="btn btn-sm"
              style={{ "font-size": "10px", padding: "2px 6px" }}
              onClick={selectAllSessions}
            >
              All
            </button>
            <button
              class="btn btn-sm"
              style={{ "font-size": "10px", padding: "2px 6px" }}
              onClick={deselectAllSessions}
            >
              None
            </button>
          </div>
          <div style={{ "max-height": "180px", "overflow-y": "auto" }}>
            <For each={state.sessions}>
              {(session) => {
                const compatible = () => isDimensionCompatible(session);
                const checked = () => state.searchSessionIds.includes(session.id);
                return (
                  <label
                    style={{
                      display: "flex",
                      "align-items": "center",
                      gap: "6px",
                      padding: "4px 2px",
                      "font-size": "12px",
                      cursor: compatible() ? "pointer" : "not-allowed",
                      opacity: compatible() ? "1" : "0.4",
                      color: checked() ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={checked()}
                      disabled={!compatible()}
                      onChange={() => actions.toggleSearchSession(session.id)}
                      style={{ "accent-color": "var(--color-accent-purple)" }}
                    />
                    <span style={{ flex: "1", "white-space": "nowrap", overflow: "hidden", "text-overflow": "ellipsis" }}>
                      {sessionLabel(session)}
                    </span>
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

      {/* Search input and button. The textarea height is controlled by a
       *  horizontal drag bar below this card, not by native resize. */}
      <div class="glass-card" style={{ padding: "16px" }}>
        <div class="section-title">
          Query
          <Tip text="Enter a natural language query to search your indexed documents. The query is converted to a vector embedding and matched against the stored document chunks using cosine similarity. Press Enter to execute the search." />
        </div>
        <textarea
          class="search-input"
          style={{ height: `${queryHeight()}px`, "margin-bottom": "12px" }}
          placeholder={
            hasSelection()
              ? "Search your documents..."
              : "Select a session first..."
          }
          value={state.queryText}
          onInput={(e) => actions.setQueryText(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={!hasSelection()}
        />
        <button
          class="btn btn-primary"
          style={{ width: "100%" }}
          onClick={executeSearch}
          disabled={
            state.searchInProgress ||
            !hasSelection() ||
            !state.queryText.trim()
          }
        >
          {state.searchInProgress
            ? "Searching..."
            : isMultiSearch()
              ? `Search ${state.searchSessionIds.length} Sessions`
              : "Search"}
        </button>

        {/* Search error display for API-level errors (network failures, etc.) */}
        <Show when={searchError()}>
          <div style={{
            "font-size": "11px",
            color: "var(--color-accent-magenta)",
            "margin-top": "8px",
            "line-height": "1.4",
          }}>
            {searchError()}
          </div>
        </Show>
      </div>

      {/* Horizontal drag bar between Query and Search Modes cards.
       *  Visually identical to the vertical SplitPane divider. Dragging
       *  vertically adjusts the query textarea height above. */}
      <div
        class={`h-resize-divider${hDragging() ? " dragging" : ""}`}
        onPointerDown={onHDividerPointerDown}
      />

      {/* Search mode toggles */}
      <div class="glass-card" style={{ padding: "16px" }}>
        <div class="section-title">
          Search Modes
          <Tip text="Configure which search stages to apply. Multiple modes can be combined for higher result quality at the cost of additional computation time." />
        </div>
        <div style={{ display: "flex", "flex-wrap": "wrap", gap: "8px" }}>
          <button
            class={`toggle-chip ${state.useHybrid ? "active" : ""}`}
            onClick={() => actions.setUseHybrid(!state.useHybrid)}
          >
            Hybrid
          </button>
          <button
            class={`toggle-chip ${state.useReranker ? "active" : ""}`}
            onClick={toggleReranker}
          >
            Reranker
            <Show when={state.useReranker && !state.rerankerAvailable}>
              <span style={{ color: "var(--color-accent-magenta)", "margin-left": "4px" }}>!</span>
            </Show>
          </button>
          <button
            class={`toggle-chip ${state.useRefine ? "active" : ""}`}
            onClick={() => actions.setUseRefine(!state.useRefine)}
          >
            Refine
          </button>
        </div>

        {/* Inline warning when the user tries to enable reranking without a loaded model.
           "Models tab" is a clickable text link that navigates to the Models tab. */}
        <Show when={rerankerWarning()}>
          <div style={{
            "font-size": "11px",
            color: "var(--color-accent-magenta)",
            "margin-top": "8px",
            "line-height": "1.5",
          }}>
            No reranker model loaded. Activate one in the{" "}
            <a
              href="#"
              style={{
                color: "var(--color-accent-cyan)",
                "text-decoration": "underline",
                cursor: "pointer",
              }}
              onClick={(e) => {
                e.preventDefault();
                actions.setActiveTab("models");
              }}
            >
              Models tab
            </a>{" "}
            first.
          </div>
        </Show>

        {/* Per-mode tooltip explanations */}
        <div style={{ "margin-top": "8px", "font-size": "11px", color: "var(--color-text-muted)", "line-height": "1.5" }}>
          <Show when={state.useHybrid}>
            <div style={{ display: "flex", "align-items": "flex-start", gap: "4px" }}>
              <span style={{ color: "var(--color-accent-cyan)", "flex-shrink": "0", "min-width": "62px" }}>Hybrid:</span>
              <span>Combines vector similarity with BM25 keyword search using Reciprocal Rank Fusion.</span>
              <Tip text="Hybrid search runs both a vector similarity search (semantic meaning) and a BM25 full-text keyword search (exact term matches) in parallel. Results from both are merged using Reciprocal Rank Fusion (RRF), which balances semantic relevance with keyword precision. Effective for queries that mix conceptual meaning with specific terminology." />
            </div>
          </Show>
          <div style={{ display: "flex", "align-items": "flex-start", gap: "4px" }}>
            <span style={{ color: state.useReranker ? "var(--color-accent-cyan)" : "var(--color-accent-purple)", "flex-shrink": "0", "min-width": "62px" }}>Reranker:</span>
            <span>Re-scores results using a cross-encoder model for higher precision.</span>
            <Tip text="The cross-encoder reranker takes each (query, result) pair and computes a joint relevance score, unlike the bi-encoder embedding model which encodes query and document independently. This produces more accurate ranking but is slower because every result must pass through the model. Requires a reranker model to be loaded in the Models tab." />
          </div>
          <Show when={state.useRefine}>
            <div style={{ display: "flex", "align-items": "flex-start", gap: "4px" }}>
              <span style={{ color: "var(--color-accent-cyan)", "flex-shrink": "0", "min-width": "62px" }}>Refine:</span>
              <span>Splits results into sub-chunks at multiple scales and re-embeds to find the most relevant passage.</span>
              <Tip text="Refinement takes each search result and splits it into smaller sub-chunks at multiple granularity levels (defined by divisors, e.g. 4,8,16 means 1/4, 1/8, 1/16 of the original). Each sub-chunk is embedded and scored against the query. When a sub-chunk scores higher than its parent, the result content is replaced with the more focused passage. This narrows down large chunks to the most relevant paragraph or sentence." />
            </div>
          </Show>
        </div>

        <Show when={state.useRefine}>
          <div style={{ "margin-top": "12px" }}>
            <div class="label">
              Refine Divisors
              <Tip text="Comma-separated integers (e.g. 4,8,16). Each value defines a sliding window size relative to the chunk length: a value of 4 creates windows that are the chunk length divided by 4, a value of 8 divides by 8, and so on. Higher values produce smaller windows for finer precision, lower values produce larger windows for broader context. The default 4,8,16 scans each result at three scales simultaneously." />
            </div>
            <input
              class="input"
              type="text"
              value={state.refineDivisors}
              onInput={(e) => actions.setRefineDivisors(e.target.value)}
              placeholder="4,8,16"
            />
          </div>
        </Show>
      </div>
    </div>
  );

  // ---- Right panel: results display or chunk context viewer (arrow function for SolidJS reactivity) ----
  const rightPanel = () => (
    <div style={{ display: "flex", "flex-direction": "column", height: "100%" }}>
      <div class="glass-card" style={{ flex: "1", display: "flex", "flex-direction": "column", "min-height": "0" }}>
        <Show
          when={openedResult() === null}
          fallback={
            /* ChunkViewer replaces the results list when a result is opened */
            <ChunkViewer
              result={openedResult()!}
              sessionId={sessionIdForResult(openedResult()!)}
              sessionLabel={sessionLabelMap().get(sessionIdForResult(openedResult()!)) || `Session #${sessionIdForResult(openedResult()!)}`}
              onClose={() => setOpenedResult(null)}
            />
          }
        >
          {/* Results header */}
          <div
            style={{
              padding: "12px 16px",
              "border-bottom": "1px solid var(--color-glass-border)",
              display: "flex",
              "align-items": "center",
              "justify-content": "space-between",
              "flex-shrink": "0",
            }}
          >
            <span style={{ "font-size": "14px", "font-weight": "600", display: "inline-flex", "align-items": "center" }}>
              Results
              <Show when={state.searchResults.length > 0}>
                {" "}({state.searchResults.length} hits)
              </Show>
              <Tip text="Search results ranked by relevance score. Scores above 0.8 (cyan) indicate strong matches, 0.5-0.8 (amber) moderate matches, below 0.5 (gray) weak matches. Click a result to view it in full document context with neighboring chunks. Use the Group toggle to cluster results by source document. Export buttons copy all results to the clipboard in the selected format." />
            </span>
            <div class="row" style={{ gap: "6px" }}>
              <Show when={state.searchResults.length > 0}>
                <button
                  class={`btn btn-sm ${state.groupByDocument ? "active" : ""}`}
                  onClick={() => actions.setGroupByDocument(!state.groupByDocument)}
                  title="Group results by source document"
                >
                  Group
                </button>
              </Show>
              <button
                class="btn btn-sm"
                disabled={state.searchResults.length === 0}
                onClick={exportMarkdown}
                title="Copy results as Markdown to clipboard"
              >
                {copiedLabel() === "md" ? "Copied!" : "MD"}
              </button>
              <button
                class="btn btn-sm"
                disabled={state.searchResults.length === 0}
                onClick={exportBibtex}
                title="Copy results as BibTeX entries to clipboard"
              >
                {copiedLabel() === "bibtex" ? "Copied!" : "BibTeX"}
              </button>
              <button
                class="btn btn-sm"
                disabled={state.searchResults.length === 0}
                onClick={exportCslJson}
                title="Copy results as CSL-JSON to clipboard"
              >
                {copiedLabel() === "csl" ? "Copied!" : "CSL"}
              </button>
            </div>
          </div>

          {/* Results body (scrollable) */}
          <div style={{ flex: "1", "overflow-y": "auto" }}>
            <Show
              when={state.searchResults.length > 0}
              fallback={
                <div
                  style={{
                    padding: "48px 20px",
                    "text-align": "center",
                    color: "var(--color-text-muted)",
                    "font-size": "14px",
                  }}
                >
                  <div style={{ "font-size": "32px", "margin-bottom": "12px", opacity: "0.3" }}>
                    <svg
                      width="48"
                      height="48"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      style={{ margin: "0 auto" }}
                    >
                      <circle cx="11" cy="11" r="8" />
                      <path d="M21 21l-4.35-4.35" />
                    </svg>
                  </div>
                  <div>Enter a query to search your indexed documents</div>
                  <div style={{ "font-size": "12px", "margin-top": "4px" }}>
                    Results will appear here with relevance scores
                  </div>
                </div>
              }
            >
              {/* Grouped view */}
              <Show when={state.groupByDocument && groupedResults()}>
                <For each={groupedResults()!}>
                  {(group) => (
                    <div style={{ "border-bottom": "1px solid var(--color-glass-border)" }}>
                      <div
                        style={{
                          padding: "10px 16px",
                          "font-size": "13px",
                          "font-weight": "600",
                          color: "var(--color-accent-cyan)",
                          background: "rgba(34, 211, 238, 0.05)",
                        }}
                      >
                        {group.file.split(/[/\\]/).pop()} — {group.results.length} results (best: {group.bestScore.toFixed(3)})
                      </div>
                      <For each={group.results}>
                        {(result) => <ResultCard result={result} sessionLabels={sessionLabelMap()} onOpen={setOpenedResult} />}
                      </For>
                    </div>
                  )}
                </For>
              </Show>

              {/* Flat view */}
              <Show when={!state.groupByDocument}>
                <For each={state.searchResults}>
                  {(result, i) => <ResultCard result={result} rank={i() + 1} sessionLabels={sessionLabelMap()} onOpen={setOpenedResult} />}
                </For>
              </Show>
            </Show>
          </div>
        </Show>
      </div>
    </div>
  );

  return (
    <SplitPane
      left={leftPanel()}
      right={rightPanel()}
      leftWidth={state.searchLeftPanelWidth}
      onResize={(w) => actions.setSearchLeftPanelWidth(w)}
      minLeft={240}
      minRight={300}
    />
  );
};

// ---------------------------------------------------------------------------
// ResultCard sub-component
// ---------------------------------------------------------------------------

/**
 * Displays a single search result as a card with relevance score, citation,
 * expandable content preview, copy button, context button, and page reference.
 * In multi-search mode, shows a session badge identifying which session the
 * result came from. Clicking the "Context" button opens the ChunkViewer to
 * show the matched chunk within its document context. The per-card copy
 * button uses async clipboard write with brief feedback.
 */
const ResultCard: Component<{
  result: SearchResultDto;
  rank?: number;
  sessionLabels: Map<number, string>;
  onOpen?: (result: SearchResultDto) => void;
}> = (props) => {
  const [expanded, setExpanded] = createSignal(false);
  const [copied, setCopied] = createSignal(false);

  /** Timer ID for the per-card "Copied!" feedback auto-clear. Stored so
   *  onCleanup can cancel a pending timer if the component unmounts while
   *  the feedback is still visible (e.g., results replaced by a new search). */
  let cardCopiedTimer: ReturnType<typeof setTimeout> | undefined;

  onCleanup(() => {
    if (cardCopiedTimer !== undefined) clearTimeout(cardCopiedTimer);
  });

  /** Truncated preview (first 200 characters) of the result content. */
  const preview = () => {
    const text = props.result.content;
    if (text.length <= 200) return text;
    return text.slice(0, 200) + "...";
  };

  /** Color for the score badge based on relevance score thresholds. */
  const scoreColor = () => {
    const s = props.result.score;
    if (s >= 0.8) return "var(--color-accent-cyan)";
    if (s >= 0.5) return "var(--color-status-outdated)";
    return "var(--color-text-muted)";
  };

  /** Copies the full result content to the clipboard with feedback.
   *  Logs a warning on clipboard failure instead of silently swallowing it. */
  const copyContent = async () => {
    try {
      await navigator.clipboard.writeText(props.result.content);
      setCopied(true);
      if (cardCopiedTimer !== undefined) clearTimeout(cardCopiedTimer);
      cardCopiedTimer = setTimeout(() => {
        setCopied(false);
        cardCopiedTimer = undefined;
      }, 1500);
    } catch (e) {
      logWarn("ResultCard.copyContent", `Clipboard write failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  /** Click handler for toggling the content preview expansion. */
  const handleContentClick = () => setExpanded(!expanded());

  return (
    <div class="result-card">
      <div class="result-card-header">
        <div class="row" style={{ gap: "8px", "align-items": "center" }}>
          <Show when={props.rank !== undefined}>
            <span style={{ "font-size": "12px", color: "var(--color-text-muted)" }}>
              #{props.rank}
            </span>
          </Show>
          <span
            style={{
              "font-size": "13px",
              "font-weight": "700",
              color: scoreColor(),
              "font-family": "monospace",
            }}
          >
            [{props.result.score.toFixed(3)}]
          </span>
          {/* Session badge for multi-search results */}
          <Show when={props.result.session_id !== undefined && props.result.session_id !== null}>
            <span
              style={{
                "font-size": "10px",
                padding: "1px 5px",
                "border-radius": "3px",
                background: "rgba(168, 85, 247, 0.15)",
                color: "var(--color-accent-purple)",
                "white-space": "nowrap",
              }}
            >
              {props.sessionLabels.get(props.result.session_id!) || `#${props.result.session_id}`}
            </span>
          </Show>
          <span style={{ "font-size": "12px", color: "var(--color-text-secondary)", flex: "1" }}>
            {props.result.citation}
          </span>
          <span style={{ "font-size": "11px", color: "var(--color-text-muted)" }}>
            pp.{props.result.page_start}-{props.result.page_end}
          </span>
        </div>
        <div class="row" style={{ gap: "8px", "margin-top": "4px" }}>
          <Show when={props.result.reranker_score !== null}>
            <span style={{ "font-size": "10px", color: "var(--color-accent-purple)" }}>
              rerank: {props.result.reranker_score!.toFixed(3)}
            </span>
          </Show>
          <Show when={props.result.bm25_rank !== null}>
            <span style={{ "font-size": "10px", color: "var(--color-text-muted)" }}>
              bm25: #{props.result.bm25_rank}
            </span>
          </Show>
        </div>
      </div>
      <div
        style={{ cursor: "pointer" }}
        onClick={handleContentClick}
        {...clickableProps(handleContentClick, "result-card-content")}
      >
        {expanded() ? props.result.content : preview()}
      </div>
      <div class="result-card-footer">
        <div class="row" style={{ gap: "6px" }}>
          <button
            class="btn btn-sm btn-primary"
            onClick={copyContent}
            style={{ "font-size": "11px" }}
          >
            {copied() ? "Copied!" : "Copy"}
          </button>
          <Show when={props.onOpen !== undefined}>
            <button
              class="btn btn-sm btn-primary"
              onClick={() => props.onOpen!(props.result)}
              style={{ "font-size": "11px" }}
              title="View this chunk in its full document context with neighboring chunks"
            >
              Context
            </button>
          </Show>
        </div>
        <span style={{ "font-size": "11px", color: "var(--color-text-muted)" }}>
          {props.result.source_file.split(/[/\\]/).pop()}
        </span>
      </div>
    </div>
  );
};

export default SearchTab;
