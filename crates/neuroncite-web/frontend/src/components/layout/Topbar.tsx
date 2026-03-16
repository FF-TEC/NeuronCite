import { Component, Show, createSignal, onCleanup } from "solid-js";
import { Portal } from "solid-js/web";
import { state, actions } from "../../stores/app";
import type { AppTab } from "../../stores/app";

/**
 * Sends a window control command to the native tao window via wry's IPC
 * channel. In the native GUI build (tao + wry), `window.ipc` is injected
 * by wry and `postMessage()` delivers the string to the Rust IPC handler.
 * In browser dev mode, `window.ipc` is undefined, so calls are silently
 * ignored via optional chaining.
 *
 * The type of `window.ipc` is declared in `src/global.d.ts`, which augments
 * the Window interface with the optional `ipc` property.
 */
const sendWindowCommand = (command: string): void => {
  window.ipc?.postMessage(command);
};

/**
 * Descriptive tooltip text for each tab in the topbar navigation.
 * Each entry explains what the tab does in plain language so that
 * first-time users understand the purpose without reading documentation.
 */
const TAB_TOOLTIPS: Record<AppTab, string> = {
  sources:
    "Parse a BibTeX (.bib) file and automatically download the cited " +
    "PDFs from URL and DOI fields. Previews all entries and shows which " +
    "source files already exist in the output folder.",
  indexing:
    "Create and manage index sessions for semantic document search. " +
    "Select a folder of PDFs, choose an embedding model, configure " +
    "chunk size and overlap, and start the indexing process.",
  search:
    "Query indexed documents using vector similarity search. Supports " +
    "searching across multiple sessions, hybrid search mode, reranking, " +
    "and exporting results as Markdown, BibTeX, or CSL-JSON.",
  citations:
    "Verify LaTeX citations automatically with a local LLM via Ollama. " +
    "Each citation is checked against its source document and receives " +
    "a verdict: Supported, Unsupported, or Uncertain.",
  annotations:
    "Highlight cited text passages directly in PDF files. Reads an " +
    "annotation CSV, locates matching text in source PDFs using " +
    "multi-stage matching, and creates highlighted copies.",
  models:
    "Manage embedding models, reranker models, and Ollama LLMs. Shows " +
    "GPU and CUDA system status. Download, activate, or remove models " +
    "used by the indexing, search, and citation pipelines.",
  settings:
    "Database maintenance (FTS5 optimization, HNSW rebuild), runtime " +
    "dependency checks (pdfium, tesseract, poppler), MCP server " +
    "registration, and real-time log viewer.",
};

/**
 * Top navigation bar spanning the full width of the application. Contains the
 * NeuronCite logo, tab navigation buttons, and native window control buttons.
 *
 * The entire topbar acts as a drag region for moving the frameless window,
 * except for interactive elements (buttons) which are excluded from dragging.
 *
 * Tab navigation is split into two groups separated by a flexible spacer:
 *
 * Left group (workflow tabs): Sources, Indexing, Search, Citations
 * Right group (utility tabs): Models, Settings
 *
 * Tab order in the left group follows the logical workflow: download cited
 * PDFs from BibTeX, embed and index documents, run search queries, verify
 * citations with a local LLM. The right group contains configuration and
 * model management tabs that are used less frequently during the workflow.
 *
 * Window controls (minimize, maximize/restore, close) are positioned at the
 * far right edge after the tab navigation. These buttons communicate with
 * the native tao window via wry's IPC channel.
 *
 * Each tab button shows a descriptive tooltip on hover (after a 400ms delay)
 * rendered via a SolidJS Portal into document.body. The tooltip explains
 * the tab's purpose in plain language and is positioned below the button
 * with horizontal clamping to stay within the viewport.
 *
 * The default active tab on startup is "search".
 */
const Topbar: Component = () => {
  /** Switches the active tab in the topbar navigation. */
  const switchTab = (tab: AppTab) => {
    actions.setActiveTab(tab);
  };

  /** Tab ID currently being hovered, or null when no tab is hovered.
   *  Controls visibility of the Portal-based tooltip panel. */
  const [hoverTab, setHoverTab] = createSignal<AppTab | null>(null);

  /** Viewport coordinates for the tooltip panel, calculated from the
   *  hovered button's bounding rect on each mouseenter event. */
  const [tipPos, setTipPos] = createSignal({ top: 0, left: 0 });

  /** Timer handle for the 400ms hover delay. Cleared on mouseleave
   *  so that brief mouse passes over tabs do not flash tooltips. */
  let hoverTimer: number | undefined;

  onCleanup(() => clearTimeout(hoverTimer));

  /** Measures the hovered tab button's viewport position and schedules
   *  the tooltip to appear after a 400ms delay. The horizontal position
   *  is clamped so the 280px-wide panel stays within the viewport. */
  const onTabEnter = (tab: AppTab, e: MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const left = Math.max(8, Math.min(rect.left, window.innerWidth - 292));
    setTipPos({ top: rect.bottom + 8, left });
    clearTimeout(hoverTimer);
    hoverTimer = window.setTimeout(() => setHoverTab(tab), 400);
  };

  /** Cancels any pending hover timer and hides the tooltip immediately. */
  const onTabLeave = () => {
    clearTimeout(hoverTimer);
    setHoverTab(null);
  };

  /** Initiates a native window drag via IPC on mousedown. The CSS property
   *  -webkit-app-region:drag does not work on macOS WKWebView, so we use
   *  tao's drag_window() API instead. Only triggers on primary button clicks
   *  directly on the topbar (not on child buttons/nav). */
  const onTopbarMouseDown = (e: MouseEvent) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    // Only drag when clicking on the topbar itself or passive children
    // (logo, spacer), not on interactive elements like buttons or nav.
    if (target.closest("button, a, input, select, textarea")) return;
    sendWindowCommand("drag");
  };

  return (
    <header class="topbar" onMouseDown={onTopbarMouseDown}>
      <div class="topbar-logo">
        Neuron<span>Cite</span>
      </div>

      {/* Tab navigation split into workflow tabs (left) and utility tabs (right).
       *  The topbar-tab-spacer pushes Models and Settings to the right edge. */}
      <nav class="topbar-tabs" role="tablist" aria-label="Application tabs">
        <button
          class={`topbar-tab${state.activeTab === "sources" ? " active" : ""}`}
          role="tab"
          aria-selected={state.activeTab === "sources"}
          aria-controls="panel-sources"
          onClick={() => switchTab("sources")}
          onMouseEnter={(e) => onTabEnter("sources", e)}
          onMouseLeave={onTabLeave}
        >
          Sources
        </button>
        <button
          class={`topbar-tab${state.activeTab === "indexing" ? " active" : ""}`}
          role="tab"
          aria-selected={state.activeTab === "indexing"}
          aria-controls="panel-indexing"
          onClick={() => switchTab("indexing")}
          onMouseEnter={(e) => onTabEnter("indexing", e)}
          onMouseLeave={onTabLeave}
        >
          Indexing
        </button>
        <button
          class={`topbar-tab${state.activeTab === "search" ? " active" : ""}`}
          role="tab"
          aria-selected={state.activeTab === "search"}
          aria-controls="panel-search"
          onClick={() => switchTab("search")}
          onMouseEnter={(e) => onTabEnter("search", e)}
          onMouseLeave={onTabLeave}
        >
          Search
        </button>
        <button
          class={`topbar-tab${state.activeTab === "citations" ? " active" : ""}`}
          role="tab"
          aria-selected={state.activeTab === "citations"}
          aria-controls="panel-citations"
          onClick={() => switchTab("citations")}
          onMouseEnter={(e) => onTabEnter("citations", e)}
          onMouseLeave={onTabLeave}
        >
          Citations
        </button>
        <button
          class={`topbar-tab${state.activeTab === "annotations" ? " active" : ""}`}
          role="tab"
          aria-selected={state.activeTab === "annotations"}
          aria-controls="panel-annotations"
          onClick={() => switchTab("annotations")}
          onMouseEnter={(e) => onTabEnter("annotations", e)}
          onMouseLeave={onTabLeave}
        >
          Annotations
        </button>

        {/* Flexible spacer separating workflow tabs from utility tabs */}
        <div class="topbar-tab-spacer" />

        <button
          class={`topbar-tab${state.activeTab === "models" ? " active" : ""}`}
          role="tab"
          aria-selected={state.activeTab === "models"}
          aria-controls="panel-models"
          onClick={() => switchTab("models")}
          onMouseEnter={(e) => onTabEnter("models", e)}
          onMouseLeave={onTabLeave}
        >
          Models
        </button>
        <button
          class={`topbar-tab${state.activeTab === "settings" ? " active" : ""}`}
          role="tab"
          aria-selected={state.activeTab === "settings"}
          aria-controls="panel-settings"
          onClick={() => switchTab("settings")}
          onMouseEnter={(e) => onTabEnter("settings", e)}
          onMouseLeave={onTabLeave}
        >
          Settings
        </button>
      </nav>

      {/* Tab tooltip rendered via Portal into document.body so that the
       *  topbar's backdrop-filter (which creates a containing block) does
       *  not interfere with position:fixed viewport coordinates. */}
      <Show when={hoverTab()}>
        <Portal>
          <div
            class="topbar-tooltip"
            style={{ top: `${tipPos().top}px`, left: `${tipPos().left}px` }}
          >
            {TAB_TOOLTIPS[hoverTab()!]}
          </div>
        </Portal>
      </Show>

      {/* Native window control buttons for the frameless window. Each button
       *  sends an IPC command to the Rust backend via window.ipc.postMessage().
       *  The close button has a distinct hover color (red) to match platform
       *  conventions. All three buttons are excluded from the drag region via
       *  the CSS property -webkit-app-region: no-drag. */}
      <div class="window-controls">
        <button
          class="window-control-btn"
          onClick={() => sendWindowCommand("minimize")}
          title="Minimize"
        >
          {/* Horizontal line representing the minimize icon */}
          <svg width="12" height="12" viewBox="0 0 12 12">
            <line x1="1" y1="6" x2="11" y2="6" stroke="currentColor" stroke-width="1.2" />
          </svg>
        </button>
        <button
          class="window-control-btn"
          onClick={() => sendWindowCommand("maximize")}
          title="Maximize"
        >
          {/* Square outline representing the maximize/restore icon */}
          <svg width="12" height="12" viewBox="0 0 12 12">
            <rect x="1.5" y="1.5" width="9" height="9" rx="1" fill="none" stroke="currentColor" stroke-width="1.2" />
          </svg>
        </button>
        <button
          class="window-control-btn window-control-close"
          onClick={() => sendWindowCommand("close")}
          title="Close"
        >
          {/* X shape representing the close icon */}
          <svg width="12" height="12" viewBox="0 0 12 12">
            <line x1="2" y1="2" x2="10" y2="10" stroke="currentColor" stroke-width="1.2" />
            <line x1="10" y1="2" x2="2" y2="10" stroke="currentColor" stroke-width="1.2" />
          </svg>
        </button>
      </div>
    </header>
  );
};

export default Topbar;
