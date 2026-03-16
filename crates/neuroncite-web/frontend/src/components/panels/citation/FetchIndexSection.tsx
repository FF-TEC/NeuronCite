/**
 * Collapsible "Fetch & Index Sources" section for the Citation panel left
 * sidebar. Provides a master toggle to enable/disable the section, source
 * directory selection, Unpaywall email for DOI resolution, embedding model
 * display, chunking strategy selector, and an "Index Source Directory" button.
 *
 * When enabled, the section body reveals inputs for configuring and indexing
 * the source directory. The toggle state persists to localStorage so the
 * user's preference survives page reloads.
 *
 * This component delegates folder browsing to the parent's BrowseModal
 * instance via the onBrowseDir callback. Indexing is triggered via the
 * onIndex callback; the actual API call is made by the parent CitationPanel.
 */

import { Component, Show, createSignal } from "solid-js";
import { state, actions, safeLsGet, safeLsSet } from "../../../stores/app";
import Tip from "../../ui/Tip";
import { LABEL_STYLE } from "./constants";

/** localStorage key for the fetch-and-index section toggle state. */
const LS_FETCH_INDEX_ENABLED = "neuroncite_fetch_index_enabled";

/** Props for FetchIndexSection. Callbacks are used to delegate browsing
 *  and indexing actions to the parent CitationPanel where the browse
 *  modal and API client logic reside. */
interface FetchIndexSectionProps {
  /** Current source directory path. */
  sourcePath: () => string;
  /** Callback to update the source directory path (persisted). */
  setSourcePath: (path: string) => void;
  /** Opens the folder browser for source directory selection. */
  onBrowseDir: () => void;
  /** Whether a browse dialog is pending (disables the Browse button). */
  dialogPending: () => boolean;
  /** Triggers source directory indexing with the current settings. */
  onIndex: (strategy: string, chunkSize: string, chunkOverlap: string) => void;
  /** Whether indexing is in progress (disables the Index button). */
  isIndexing: () => boolean;
}

/** Return type from useFetchIndex, providing the enabled state accessor
 *  and the rendered UI component. */
export interface FetchIndexHandle {
  /** Whether the Fetch & Index Sources section is enabled. */
  fetchIndexEnabled: () => boolean;
  /** The JSX component to render inside the left panel. */
  FetchIndexUI: Component<FetchIndexSectionProps>;
}

/**
 * Hook-style composable that creates the fetch-index enabled state and
 * returns a component plus accessor. The enabled state is shared with
 * the parent so the auto-verify request can conditionally include
 * source_directory and fetch_sources parameters.
 */
export function useFetchIndex(): FetchIndexHandle {
  // Master toggle for the "Fetch & Index Sources" section. When disabled,
  // the entire section body is hidden. Persisted to localStorage so the
  // user's preference survives page reloads.
  const [fetchIndexEnabled, setFetchIndexEnabled] = createSignal(
    safeLsGet(LS_FETCH_INDEX_ENABLED) === "true",
  );

  /** Persists the Fetch & Index Sources toggle state and clears
   *  all selected sessions when enabling, so the subsequently
   *  indexed session becomes the sole selection. */
  const setFetchIndexEnabledPersist = (enabled: boolean) => {
    setFetchIndexEnabled(enabled);
    safeLsSet(LS_FETCH_INDEX_ENABLED, String(enabled));
    if (enabled) {
      actions.setCitationSessionIds([]);
    }
  };

  const FetchIndexUI: Component<FetchIndexSectionProps> = (props) => {
    // Indexing configuration signals. Initialized from the global store
    // defaults but maintained locally since they only apply to this section.
    const [indexStrategy, setIndexStrategy] = createSignal(state.selectedStrategy);
    const [indexChunkSize, setIndexChunkSize] = createSignal(state.chunkSize);
    const [indexChunkOverlap, setIndexChunkOverlap] = createSignal(state.chunkOverlap);

    return (
      <div class="glass-card" style={{ padding: "16px" }}>
        <label style={{
          display: "flex",
          "align-items": "center",
          gap: "8px",
          cursor: "pointer",
          "font-weight": "600",
          color: "var(--color-text-primary)",
        }}>
          <input
            type="checkbox"
            checked={fetchIndexEnabled()}
            onChange={(e) => setFetchIndexEnabledPersist(e.target.checked)}
            style={{ "accent-color": "var(--color-accent-purple)" }}
          />
          Fetch & Index Sources
          <Tip text="Downloads cited source documents from BibTeX URL/DOI fields and indexes them into a searchable session. Enable this section to automatically acquire and index source PDFs before verification." />
        </label>

        <Show when={fetchIndexEnabled()}>
          <div style={{ "margin-top": "12px", display: "flex", "flex-direction": "column", gap: "8px" }}>

            {/* Source directory path with browse button */}
            <div style={{ display: "flex", gap: "8px", "align-items": "center" }}>
              <label style={LABEL_STYLE}>
                Sources
                <Tip text="Local directory where source PDFs are stored or will be downloaded to. This path is used both as the download target for fetching and as the input directory for indexing." />
              </label>
              <input
                class="input"
                type="text"
                placeholder="/path/to/source/pdfs/"
                value={props.sourcePath()}
                onInput={(e) => props.setSourcePath(e.target.value)}
                style={{ flex: "1" }}
              />
              <button class="btn btn-sm" onClick={props.onBrowseDir} disabled={props.dialogPending()}>
                {props.dialogPending() ? "..." : "Browse"}
              </button>
            </div>

            {/* Unpaywall email for DOI resolution */}
            <div style={{ display: "flex", gap: "8px", "align-items": "center" }}>
              <label style={LABEL_STYLE}>
                Email
                <Tip text="Email address for Unpaywall API access. When provided, DOI resolution queries Unpaywall first for direct open-access PDF URLs before falling back to Semantic Scholar, OpenAlex, and doi.org." />
              </label>
              <input
                class="input"
                type="email"
                placeholder="your@email.com (for Unpaywall DOI resolution)"
                value={state.unpaywallEmail}
                onInput={(e) => actions.setUnpaywallEmail(e.target.value)}
                style={{ flex: "1" }}
              />
            </div>

            {/* Currently loaded embedding model (read-only display) */}
            <div style={{ display: "flex", gap: "8px", "align-items": "center" }}>
              <label style={LABEL_STYLE}>
                Model
                <Tip text="The embedding model loaded in the Indexing tab. Documents are converted to vectors using this model. All sessions searched together must use the same model and vector dimension." />
              </label>
              <span style={{ "font-size": "12px", color: "var(--color-accent-cyan)" }}>
                {state.activeModelId
                  ? `${state.activeModelId.split("/").pop()} (${state.activeModelDimension}d)`
                  : "No model loaded"}
              </span>
            </div>

            {/* Chunking strategy selector */}
            <div style={{ display: "flex", gap: "8px", "align-items": "center" }}>
              <label style={LABEL_STYLE}>
                Strategy
                <Tip text="Chunking strategy for splitting documents into searchable segments. 'sentence' is recommended for academic papers. 'token' and 'word' allow custom chunk size and overlap settings." />
              </label>
              <select
                class="select"
                style={{ flex: "1" }}
                value={indexStrategy()}
                onChange={(e) => setIndexStrategy(e.target.value)}
              >
                <option value="token">Token</option>
                <option value="word">Word</option>
                <option value="sentence">Sentence</option>
                <option value="page">Page</option>
              </select>
            </div>

            {/* Chunk size and overlap inputs (visible for token/word strategies) */}
            <Show when={indexStrategy() === "token" || indexStrategy() === "word"}>
              <div style={{ display: "flex", gap: "8px", "align-items": "center" }}>
                <label style={LABEL_STYLE}>
                  Chunk
                  <Tip text="Number of tokens or words per chunk. Larger chunks retain more context but reduce retrieval precision. Default: 256." />
                </label>
                <input
                  class="input"
                  type="text"
                  placeholder="256"
                  value={indexChunkSize()}
                  onInput={(e) => setIndexChunkSize(e.target.value)}
                  style={{ flex: "1" }}
                />
                <label style={{ color: "var(--color-text-secondary)", "font-size": "13px" }}>
                  Overlap
                  <Tip text="Number of tokens or words overlapping between consecutive chunks. Prevents information loss at chunk boundaries. Default: 32." />
                </label>
                <input
                  class="input"
                  type="text"
                  placeholder="32"
                  value={indexChunkOverlap()}
                  onInput={(e) => setIndexChunkOverlap(e.target.value)}
                  style={{ width: "60px" }}
                />
              </div>
            </Show>

            {/* Index button. Disabled when no source path is set, no model
             *  is loaded, or indexing is already in progress. */}
            <button
              class="btn btn-primary"
              onClick={() => props.onIndex(indexStrategy(), indexChunkSize(), indexChunkOverlap())}
              disabled={props.isIndexing() || !props.sourcePath() || !state.activeModelId || state.indexProgress !== null}
              style={{ width: "100%" }}
            >
              {props.isIndexing() || state.indexProgress !== null ? "Indexing..." : "Index Source Directory"}
            </button>
          </div>
        </Show>
      </div>
    );
  };

  return {
    fetchIndexEnabled,
    FetchIndexUI,
  };
}
