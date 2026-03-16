import { Component, createSignal, createEffect, For, Show, on, onCleanup } from "solid-js";
import { state, actions, safeLsGet, safeLsSet, LS_BIB_PATH, LS_SOURCE_PATH } from "../../stores/app";
import { api } from "../../api/client";
import type { BibEntryPreview, FetchSourceResult } from "../../api/types";
import Tip from "../ui/Tip";
import SplitPane from "./SplitPane";
import { registerSourceEntryHandler } from "../../api/sse";
import { createBrowseModal } from "../ui/BrowseModal";
import type { BrowseModalConfig } from "../ui/BrowseModal";

/**
 * Sources tab content for downloading cited source documents from BibTeX
 * URL/DOI fields. Uses a resizable SplitPane layout identical to the
 * Indexing and Search tabs.
 *
 * Left panel: Configuration inputs (bib path, output dir, email, delay).
 * Right panel: Live BibTeX preview with file existence indicators, or
 * streaming download results during/after fetching.
 *
 * The BibTeX preview automatically checks which sources already exist in
 * the output directory whenever both paths are set, allowing the user to
 * see what is missing before starting a fetch.
 */
const SourcesTab: Component = () => {
  /** Path to the .bib file containing BibTeX entries with URL/DOI fields. */
  const [bibPath, setBibPath] = createSignal(
    safeLsGet(LS_BIB_PATH) || "",
  );

  /** Output directory where downloaded PDF and HTML files are stored. */
  const [outputDir, setOutputDir] = createSignal(
    safeLsGet(LS_SOURCE_PATH) || "",
  );

  /** Delay in milliseconds between consecutive HTTP requests to avoid
   *  overwhelming target servers. Default: 1000ms. */
  const [delayMs, setDelayMs] = createSignal("1000");

  /** Parsed BibTeX entries from the parse-bib endpoint, used for the
   *  live preview before fetching starts. Includes file_exists status
   *  when an output directory is configured. */
  const [bibEntries, setBibEntries] = createSignal<BibEntryPreview[]>([]);

  /** Per-entry results during/after fetch. Each entry is identified by
   *  cite_key and carries the download status and metadata. */
  const [results, setResults] = createSignal<FetchSourceResult[]>([]);

  /** Whether the bib file is being parsed (loading indicator). */
  const [bibParsing, setBibParsing] = createSignal(false);

  /** Whether fetching is in progress (entries are being downloaded). */
  const [fetching, setFetching] = createSignal(false);

  /** Aggregate download statistics from the last fetch operation. */
  const [summary, setSummary] = createSignal<{
    total_entries: number;
    entries_with_url: number;
    pdfs_downloaded: number;
    pdfs_failed: number;
    pdfs_skipped: number;
    html_fetched: number;
    html_failed: number;
    html_blocked: number;
    html_skipped: number;
  } | null>(null);

  /** Error message from the last parse or fetch attempt. */
  const [error, setError] = createSignal("");

  /** Whether a bib report export is in progress. */
  const [exporting, setExporting] = createSignal(false);

  /** Set of cite_keys whose detail view is expanded in the BibTeX preview. */
  const [expandedKeys, setExpandedKeys] = createSignal<Set<string>>(new Set());

  /** Persists the bib path to localStorage when the input changes. */
  const updateBibPath = (value: string) => {
    setBibPath(value);
    safeLsSet(LS_BIB_PATH, value);
  };

  /** Persists the output directory to localStorage when the input changes. */
  const updateOutputDir = (value: string) => {
    setOutputDir(value);
    safeLsSet(LS_SOURCE_PATH, value);
  };

  /** Persists the Unpaywall email via the store action. */
  const updateEmail = (value: string) => {
    actions.setUnpaywallEmail(value);
  };

  // ---- Browse modal (shared composable) ----

  const browse = createBrowseModal({
    onSelect: (path, mode) => {
      if (mode === "file") {
        updateBibPath(path);
      } else {
        updateOutputDir(path);
      }
    },
  });

  /** Returns the BrowseModalConfig for the current browse mode. */
  const browseConfig = (): BrowseModalConfig => {
    if (browse.browseMode() === "file") {
      return { title: "Select BibTeX File", fileExtension: ".bib", showFolderSelect: false };
    }
    return { title: "Select Output Directory", fileExtension: "", showFolderSelect: true };
  };

  /** Opens a native OS file dialog for .bib file selection. Falls back
   *  to the custom browser modal if the native dialog is not available. */
  const openBibBrowser = async () => {
    const startPath = bibPath().replace(/[\\/][^\\/]*$/, "");
    const selected = await browse.openNativeFileDialog("file", startPath);
    if (selected) {
      updateBibPath(selected);
    }
  };

  /** Opens a native OS folder dialog for output directory selection.
   *  Falls back to the custom browser modal if native is not available. */
  const openDirBrowser = async () => {
    const selected = await browse.openNativeFolderDialog("folder", outputDir());
    if (selected) {
      updateOutputDir(selected);
    }
  };

  // ---- Live BibTeX preview: parse the .bib file when the path changes ----

  /** Debounce timer ID for bib path / output dir changes. Prevents rapid-fire
   *  API calls while the user is still typing. */
  let bibDebounceTimer: ReturnType<typeof setTimeout> | null = null;

  /** Parses the .bib file at the current path and populates the preview.
   *  When output_directory is non-empty, also checks which sources already
   *  exist in the directory for the file_exists indicators. */
  const parseBibFile = async (path: string, outDir: string) => {
    const trimmed = path.trim();
    if (!trimmed || !trimmed.endsWith(".bib")) {
      setBibEntries([]);
      return;
    }

    setBibParsing(true);
    setError("");

    try {
      const req: { bib_path: string; output_directory?: string } = { bib_path: trimmed };
      const outTrimmed = outDir.trim();
      if (outTrimmed) {
        req.output_directory = outTrimmed;
      }
      const res = await api.parseBib(req);
      setBibEntries(res.entries);
    } catch (e) {
      // Parse errors are non-critical: clear entries and show error.
      setBibEntries([]);
      const msg = e instanceof Error ? e.message : String(e);
      // Only show file-not-found errors, ignore others during typing.
      if (msg.includes("does not exist")) {
        setError(msg);
      }
    } finally {
      setBibParsing(false);
    }
  };

  /** Triggers a debounced re-parse when either the bib path or output
   *  directory changes. Both values are needed for the file existence check. */
  const triggerReparse = () => {
    if (bibDebounceTimer !== null) clearTimeout(bibDebounceTimer);
    bibDebounceTimer = setTimeout(() => parseBibFile(bibPath(), outputDir()), 500);
  };

  // React to bib path changes with a 500ms debounce.
  createEffect(on(bibPath, triggerReparse));
  // React to output directory changes to refresh existence checks.
  createEffect(on(outputDir, triggerReparse));

  // Re-parse when the Sources tab becomes the active tab, and poll every
  // 5 seconds while the tab stays active. This ensures file_exists flags
  // are refreshed when files are deleted or added on disk without
  // requiring a manual refresh or tab switch.
  let pollTimer: ReturnType<typeof setInterval> | null = null;

  createEffect(
    on(
      () => state.activeTab,
      (tab) => {
        // Clear any existing poll interval when leaving the Sources tab.
        if (pollTimer !== null) {
          clearInterval(pollTimer);
          pollTimer = null;
        }

        if (tab === "sources" && bibPath().trim()) {
          // Immediate refresh on tab activation.
          parseBibFile(bibPath(), outputDir());

          // Poll every 5 seconds while the Sources tab is active and no
          // fetch operation is running. The poll silently re-parses the
          // bib file with the output directory so file_exists flags stay
          // current when files are added or removed on disk.
          pollTimer = setInterval(() => {
            if (!fetching() && bibPath().trim()) {
              parseBibFile(bibPath(), outputDir());
            }
          }, 5000);
        }
      },
    ),
  );

  onCleanup(() => {
    if (bibDebounceTimer !== null) clearTimeout(bibDebounceTimer);
    if (pollTimer !== null) clearInterval(pollTimer);
  });

  // ---- SSE subscription for live fetch results ----

  /** Cleanup function returned by subscribeSSE when connecting to the
   *  source fetch event stream. Called to close the connection when the
   *  fetch completes or the component unmounts. */
  let cleanupSourceSSE: (() => void) | null = null;

  /** Number of entries that require downloading (not skipped). Set
   *  when the fetch starts and used to compute download-phase progress. */
  let downloadTotal = 0;

  /** Registers the per-entry source update handler through the existing progress
   *  SSE stream (App.tsx subscribes to /api/v1/events/progress on mount). The
   *  backend routes source fetch events through progress_tx so that no 6th SSE
   *  connection is required from the browser. With 5 permanent SSE streams
   *  (logs, progress, jobs, models, citation) already open, adding a sources
   *  stream would exhaust the HTTP/1.1 6-connection-per-host limit and leave
   *  no slot for the fetch POST itself, causing a complete hang.
   *
   *  Progress is mapped to two phases:
   *    0-20%  = DOI resolution (entries with DOI but no URL)
   *   20-100% = downloading PDFs and HTML pages
   *
   *  The StatusBar receives total=100 so `done` equals the displayed
   *  percentage directly. */
  const connectSourceSSE = (): void => {
    if (cleanupSourceSSE) {
      cleanupSourceSSE();
    }

    // registerSourceEntryHandler sets a module-level callback in sse.ts that
    // the subscribeToProgress event listener calls when source_entry_update
    // events arrive on the progress EventSource. The returned cleanup clears
    // the callback when the fetch completes or the component unmounts.
    cleanupSourceSSE = registerSourceEntryHandler((entry) => {
      const status = entry.status;

      // Phase 1: DOI resolution events carry doi_done / doi_total fields.
      // These map to 0-20% of total progress. The fields are not part of
      // FetchSourceResult but may be present on DOI progress events.
      if (status === "resolving" || status === "resolved") {
        const raw = entry as Record<string, unknown>;
        const doiDone = typeof raw.doi_done === "number" ? raw.doi_done : 0;
        const doiTotal = typeof raw.doi_total === "number" ? raw.doi_total : 1;
        const pct = Math.round((doiDone / doiTotal) * 20);
        actions.setTaskProgress({
          label: `Resolving DOIs (${doiDone}/${doiTotal})`,
          done: pct,
          total: 100,
        });
        return;
      }

      // Phase 2: Download events update the results array. The entry is
      // already typed as FetchSourceResult (validated by isSourceEntryUpdate).
      setResults((prev) => {
        const idx = prev.findIndex((r) => r.cite_key === entry.cite_key);
        if (idx >= 0) {
          const updated = [...prev];
          updated[idx] = entry;
          return updated;
        }
        return [...prev, entry];
      });

      // Count completed download entries. Statuses other than "pending"
      // and "resolving" indicate the entry has been processed.
      if (status !== "pending") {
        const currentResults = results();
        const doneCount = currentResults.filter(
          (r) => r.status !== "pending" && r.status !== "resolving",
        ).length;
        // Map download progress to 20-100% range.
        const total = downloadTotal > 0 ? downloadTotal : currentResults.length;
        const downloadPct = total > 0 ? Math.round((doneCount / total) * 80) : 0;
        actions.setTaskProgress({
          label: `Downloading (${doneCount}/${total})`,
          done: 20 + downloadPct,
          total: 100,
        });
      }
    });
  };

  /** Disconnects the SSE source fetch event stream. */
  const disconnectSourceSSE = () => {
    if (cleanupSourceSSE) {
      cleanupSourceSSE();
      cleanupSourceSSE = null;
    }
  };

  onCleanup(disconnectSourceSSE);

  // ---- Fetch Sources ----

  /** Initiates the source fetching process. First populates the results
   *  array with "pending" entries from the bib preview, then connects SSE
   *  for live updates, and finally fires the fetch request. */
  const startFetch = async () => {
    if (!bibPath().trim()) {
      setError("BibTeX file path is required.");
      return;
    }
    if (!outputDir().trim()) {
      setError("Output directory is required.");
      return;
    }
    setError("");
    setSummary(null);
    setFetching(true);
    actions.setSourceFetchInProgress(true);
    actions.setSourceFetchMessage("Fetching sources...");

    // Pre-populate the results with "pending" entries from the bib preview
    // so the user can see what will be processed.
    const entries = bibEntries();
    const pendingResults: FetchSourceResult[] = entries
      .filter((e) => (e.has_url || e.has_doi) && !e.file_exists)
      .map((e) => ({
        cite_key: e.cite_key,
        url: e.url || (e.doi ? `https://doi.org/${e.doi}` : ""),
        type: "unknown",
        status: "pending",
      }));

    // Also add "skipped" entries for already-existing sources so they
    // appear in the results list immediately.
    const skippedResults: FetchSourceResult[] = entries
      .filter((e) => e.file_exists && (e.has_url || e.has_doi))
      .map((e) => ({
        cite_key: e.cite_key,
        url: e.url || (e.doi ? `https://doi.org/${e.doi}` : ""),
        type: "unknown",
        status: "skipped",
      }));

    setResults([...skippedResults, ...pendingResults]);

    // Store the number of entries that will be downloaded (not skipped)
    // for accurate download-phase percentage calculation.
    downloadTotal = skippedResults.length + pendingResults.length;

    // Initialize the progress bar at 0%. Phase 1 (DOI resolution) covers
    // 0-20%, phase 2 (downloading) covers 20-100%.
    actions.setTaskProgress({
      label: "Resolving DOIs...",
      done: 0,
      total: 100,
    });

    // Register the source entry update handler before firing the POST. The
    // handler is set synchronously via registerSourceEntryHandler, so it is
    // in place before the first server-side SSE event can arrive. Source
    // events flow through the already-open progress SSE stream, so no new
    // connection slot is consumed and no race condition exists.
    connectSourceSSE();

    try {
      const delay = parseInt(delayMs(), 10);
      const res = await api.fetchSources({
        bib_path: bibPath().trim(),
        output_directory: outputDir().trim(),
        delay_ms: isNaN(delay) ? 1000 : delay,
        email: state.unpaywallEmail.trim() || undefined,
      });

      // Final update from the complete response overwrites any SSE data.
      setResults(res.results);
      setSummary({
        total_entries: res.total_entries,
        entries_with_url: res.entries_with_url,
        pdfs_downloaded: res.pdfs_downloaded,
        pdfs_failed: res.pdfs_failed,
        pdfs_skipped: res.pdfs_skipped,
        html_fetched: res.html_fetched,
        html_failed: res.html_failed,
        html_blocked: res.html_blocked,
        html_skipped: res.html_skipped,
      });

      const totalSkipped = res.pdfs_skipped + res.html_skipped;
      const totalDownloaded = res.pdfs_downloaded + res.html_fetched;
      actions.setSourceFetchMessage(
        `Done: ${totalDownloaded} downloaded, ${totalSkipped} skipped (already exist).`,
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      actions.setSourceFetchMessage("");
    } finally {
      setFetching(false);
      actions.setSourceFetchInProgress(false);
      // Clear the generic task progress bar in the StatusBar.
      actions.setTaskProgress(null);
      // Disconnect SSE after the fetch completes.
      disconnectSourceSSE();
      // Refresh the preview to update file_exists flags after download.
      parseBibFile(bibPath(), outputDir());
    }
  };

  /** Returns a CSS color for the result status badge. */
  const statusColor = (status: string): string => {
    switch (status) {
      case "downloaded":
      case "fetched":
        return "var(--color-accent-cyan)";
      case "failed":
        return "var(--color-accent-magenta)";
      case "blocked":
        return "var(--color-accent-amber)";
      case "skipped":
        return "var(--color-accent-purple)";
      case "pending":
        return "var(--color-text-muted)";
      default:
        return "var(--color-text-muted)";
    }
  };

  /** Returns a color-coded badge label for link availability. */
  const linkBadge = (entry: BibEntryPreview) => {
    if (entry.has_url) return { label: "URL", color: "var(--color-accent-cyan)" };
    if (entry.has_doi) return { label: "DOI", color: "var(--color-accent-purple)" };
    return { label: "no link", color: "var(--color-text-muted)" };
  };

  /** Calls the bib-report endpoint to generate CSV and XLSX files in the
   *  output directory. Requires both bib path and output directory to be set. */
  const exportBibReport = async () => {
    if (!bibPath().trim() || !outputDir().trim()) return;
    setExporting(true);
    setError("");
    try {
      await api.bibReport({
        bib_path: bibPath().trim(),
        output_directory: outputDir().trim(),
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting(false);
    }
  };

  /** Toggles the expanded state of a BibTeX entry by cite_key. */
  const toggleExpanded = (citeKey: string) => {
    setExpandedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(citeKey)) {
        next.delete(citeKey);
      } else {
        next.add(citeKey);
      }
      return next;
    });
  };

  // ---- Computed statistics for the preview header ----
  const existingCount = () => bibEntries().filter((e) => e.file_exists).length;
  const duplicateCount = () => bibEntries().filter((e) => e.duplicate_files && e.duplicate_files.length >= 2).length;
  const missingCount = () => bibEntries().filter((e) => !e.file_exists && (e.has_url || e.has_doi)).length;

  // ---- Left panel content ----
  const leftPanel = () => (
    <div class="glass-card" style={{ padding: "20px" }}>
      <div class="section-title">
        Source Fetching
        <Tip text="Downloads cited source documents (PDFs and HTML pages) from the URLs and DOIs listed in a BibTeX file. After downloading, index the output folder in the Indexing tab to make the documents searchable." />
      </div>

      {/* BibTeX file path with browse button */}
      <div style={{ "margin-bottom": "12px" }}>
        <div class="label">
          BibTeX File (.bib)
          <Tip text="Absolute path to a .bib file on this machine. The file is parsed to extract URL and DOI fields from each entry. Entries without a URL or DOI are skipped during fetching. The preview on the right updates automatically as you type." />
        </div>
        <div class="row">
          <input
            class="input"
            type="text"
            placeholder="/path/to/references.bib"
            value={bibPath()}
            onInput={(e) => updateBibPath(e.currentTarget.value)}
            style={{ flex: "1" }}
          />
          <button class="btn btn-sm" onClick={openBibBrowser} disabled={browse.dialogPending()}>
            {browse.dialogPending() ? "..." : "Browse"}
          </button>
        </div>
      </div>

      {/* Output directory for downloaded sources with browse button */}
      <div style={{ "margin-bottom": "12px" }}>
        <div class="label">
          Output Directory
          <Tip text="Local directory where downloaded source files are saved. PDFs are stored in a pdf/ subfolder and HTML pages in an html/ subfolder. The directories are created automatically if they do not exist. File names are derived from BibTeX metadata (title, author, year) to enable citation matching later." />
        </div>
        <div class="row">
          <input
            class="input"
            type="text"
            placeholder="/path/to/downloaded_sources"
            value={outputDir()}
            onInput={(e) => updateOutputDir(e.currentTarget.value)}
            style={{ flex: "1" }}
          />
          <button class="btn btn-sm" onClick={openDirBrowser} disabled={browse.dialogPending()}>
            {browse.dialogPending() ? "..." : "Browse"}
          </button>
        </div>
      </div>

      {/* Unpaywall email for DOI resolution */}
      <div style={{ "margin-bottom": "12px" }}>
        <div class="label">
          Unpaywall Email (optional)
          <Tip text="Email address used to authenticate with the Unpaywall API. When a BibTeX entry has a DOI but no direct URL, the system queries Unpaywall for a free open-access PDF link before falling back to the DOI redirect. Unpaywall requires a valid email for API access." />
        </div>
        <input
          class="input"
          type="email"
          placeholder="user@example.com"
          value={state.unpaywallEmail}
          onInput={(e) => updateEmail(e.currentTarget.value)}
        />
        <div style={{ "font-size": "11px", color: "var(--color-text-muted)", "margin-top": "4px" }}>
          When provided, DOI resolution queries Unpaywall first for direct open-access PDF URLs.
        </div>
      </div>

      {/* Request delay */}
      <div style={{ "margin-bottom": "16px" }}>
        <div class="label">
          Request Delay (ms)
          <Tip text="Pause in milliseconds between consecutive HTTP requests. Prevents overwhelming target servers with rapid-fire downloads, which could trigger rate limiting or IP blocking. Default: 1000ms (one request per second)." />
        </div>
        <input
          class="input"
          type="number"
          min="0"
          step="100"
          value={delayMs()}
          onInput={(e) => setDelayMs(e.currentTarget.value)}
          style={{ width: "120px" }}
        />
      </div>

      {/* Error message */}
      <Show when={error()}>
        <div style={{
          padding: "8px 12px",
          background: "rgba(236, 72, 153, 0.1)",
          border: "1px solid rgba(236, 72, 153, 0.3)",
          "border-radius": "var(--radius-sm)",
          color: "var(--color-accent-magenta)",
          "font-size": "12px",
          "margin-bottom": "12px",
        }}>
          {error()}
        </div>
      </Show>

      {/* Fetch button */}
      <button
        class="btn btn-primary"
        style={{ width: "100%" }}
        disabled={state.sourceFetchInProgress}
        onClick={startFetch}
      >
        {state.sourceFetchInProgress ? "Fetching Sources..." : "Fetch Sources"}
      </button>

      {/* Status message from last operation */}
      <Show when={state.sourceFetchMessage}>
        <div style={{
          "font-size": "12px",
          color: "var(--color-text-secondary)",
          "margin-top": "8px",
        }}>
          {state.sourceFetchMessage}
        </div>
      </Show>

      {/* Aggregate summary statistics */}
      <Show when={summary()}>
        <div style={{
          "margin-top": "16px",
          padding: "12px",
          background: "var(--color-bg-card)",
          "border-radius": "var(--radius-sm)",
          "font-size": "12px",
          color: "var(--color-text-secondary)",
        }}>
          <div class="section-title" style={{ "margin-bottom": "8px" }}>
            Summary
            <Tip text="Aggregate download statistics from the completed fetch operation. Shows how many BibTeX entries were processed and how many documents were successfully downloaded, skipped, or failed." />
          </div>
          <div style={{ display: "grid", "grid-template-columns": "1fr 1fr", gap: "4px 12px" }}>
            <span>BibTeX entries:</span>
            <span style={{ color: "var(--color-text-primary)" }}>{summary()!.total_entries}</span>
            <span>With URL/DOI:</span>
            <span style={{ color: "var(--color-text-primary)" }}>{summary()!.entries_with_url}</span>
            <span>PDFs downloaded:</span>
            <span style={{ color: "var(--color-accent-cyan)" }}>{summary()!.pdfs_downloaded}</span>
            <span>PDFs skipped:</span>
            <span style={{ color: "var(--color-accent-purple)" }}>{summary()!.pdfs_skipped}</span>
            <span>PDFs failed:</span>
            <span style={{ color: summary()!.pdfs_failed > 0 ? "var(--color-accent-magenta)" : "var(--color-text-primary)" }}>
              {summary()!.pdfs_failed}
            </span>
            <span>HTML fetched:</span>
            <span style={{ color: "var(--color-accent-cyan)" }}>{summary()!.html_fetched}</span>
            <span>HTML skipped:</span>
            <span style={{ color: "var(--color-accent-purple)" }}>{summary()!.html_skipped}</span>
            <span>HTML failed:</span>
            <span style={{ color: summary()!.html_failed > 0 ? "var(--color-accent-magenta)" : "var(--color-text-primary)" }}>
              {summary()!.html_failed}
            </span>
            <span>HTML blocked:</span>
            <span style={{ color: summary()!.html_blocked > 0 ? "var(--color-accent-amber)" : "var(--color-text-primary)" }}>
              {summary()!.html_blocked}
            </span>
          </div>
        </div>
      </Show>

      {/* Info box: shown after a successful fetch with at least one downloaded file.
       *  Recommends the user to index the downloaded documents via the Indexing tab. */}
      <Show when={summary() && !fetching() && (summary()!.pdfs_downloaded > 0 || summary()!.html_fetched > 0)}>
        <div style={{
          "margin-top": "12px",
          padding: "12px 14px",
          background: "rgba(0, 224, 255, 0.06)",
          border: "1px solid rgba(0, 224, 255, 0.25)",
          "border-radius": "var(--radius-sm)",
          "font-size": "12px",
          "line-height": "1.5",
          color: "var(--color-text-secondary)",
        }}>
          Documents saved to <span style={{ color: "var(--color-text-primary)", "font-weight": "500" }}>{outputDir()}</span> (PDFs in <code style={{ "font-size": "11px" }}>pdf/</code>, HTML in <code style={{ "font-size": "11px" }}>html/</code>).
          {" "}To make them searchable, index this folder in the{" "}
          <span
            style={{
              color: "var(--color-accent-cyan)",
              "text-decoration": "underline",
              cursor: "pointer",
            }}
            onClick={() => actions.setActiveTab("indexing")}
          >
            Indexing tab
          </span>.
        </div>
      </Show>
    </div>
  );

  // ---- Right panel content ----
  const rightPanel = () => (
    <div class="glass-card" style={{ padding: "20px", display: "flex", "flex-direction": "column", height: "100%" }}>

      {/* Show fetch results when fetching is active or results exist */}
      <Show
        when={fetching() || results().length > 0}
        fallback={
          /* Show BibTeX preview when a .bib file is parsed */
          <Show
            when={bibEntries().length > 0}
            fallback={
              <div style={{
                color: "var(--color-text-muted)",
                "font-size": "13px",
                "text-align": "center",
                "margin-top": "40px",
              }}>
                {bibParsing()
                  ? "Parsing BibTeX file..."
                  : 'Enter a .bib file path to preview entries, then click "Fetch Sources" to download.'}
              </div>
            }
          >
            <div style={{ display: "flex", "align-items": "center" }}>
              <div class="section-title" style={{ flex: "1" }}>
                BibTeX Preview
                <Tip text="Live preview of all entries parsed from the .bib file. Each entry shows its cite_key, title, author, and whether a URL or DOI is available for downloading. Entries marked 'exists' already reside in the output directory and will be skipped during fetching." />
                <span style={{
                  "font-weight": "400",
                  "font-size": "12px",
                  color: "var(--color-text-muted)",
                  "margin-left": "8px",
                }}>
                  {bibEntries().length} entries
                  {" \u2022 "}
                  {bibEntries().filter((e) => e.has_url || e.has_doi).length} with link
                </span>
              </div>
              <Show when={outputDir().trim()}>
                <button
                  class="btn btn-sm"
                  style={{
                    "font-size": "11px",
                    padding: "3px 10px",
                    "white-space": "nowrap",
                    opacity: exporting() ? "0.6" : "1",
                  }}
                  disabled={exporting()}
                  onClick={exportBibReport}
                >
                  {exporting() ? "Exporting..." : "Export Report"}
                </button>
              </Show>
              {/* Manual refresh button to re-check file existence on disk. */}
              <button
                class="btn btn-sm"
                style={{
                  "font-size": "11px",
                  padding: "3px 8px",
                  "white-space": "nowrap",
                  opacity: bibParsing() ? "0.6" : "1",
                }}
                disabled={bibParsing()}
                onClick={() => parseBibFile(bibPath(), outputDir())}
                title="Refresh file existence status from disk"
              >
                {bibParsing() ? "..." : "Refresh"}
              </button>
            </div>

            {/* Status summary bar: shows counts of existing, duplicate, missing sources */}
            <Show when={outputDir().trim() && (existingCount() > 0 || missingCount() > 0)}>
              <div style={{
                display: "flex",
                gap: "12px",
                "margin-bottom": "8px",
                "font-size": "12px",
              }}>
                <span style={{ color: "var(--color-accent-cyan)" }}>
                  {existingCount()} exist
                </span>
                <Show when={duplicateCount() > 0}>
                  <span style={{ color: "var(--color-accent-amber)" }}>
                    {duplicateCount()} duplicate
                  </span>
                </Show>
                <span style={{ color: "var(--color-accent-magenta)" }}>
                  {missingCount()} missing
                </span>
                <span style={{ color: "var(--color-text-muted)" }}>
                  {bibEntries().filter((e) => !e.has_url && !e.has_doi).length} no link
                </span>
              </div>
            </Show>

            <div style={{
              display: "flex",
              "flex-direction": "column",
              gap: "1px",
              flex: "1",
              "min-height": "0",
              "overflow-y": "auto",
              "margin-top": "8px",
            }}>
              <For each={bibEntries()}>
                {(entry) => {
                  const badge = linkBadge(entry);
                  const isExpanded = () => expandedKeys().has(entry.cite_key);

                  // Collect all detail fields that have values for the
                  // expanded view. Named fields first, then extra_fields.
                  const detailFields = (): Array<{ label: string; value: string }> => {
                    const fields: Array<{ label: string; value: string }> = [];
                    if (entry.entry_type) fields.push({ label: "Type", value: entry.entry_type });
                    if (entry.year) fields.push({ label: "Year", value: entry.year });
                    if (entry.url) fields.push({ label: "URL", value: entry.url });
                    if (entry.doi) fields.push({ label: "DOI", value: entry.doi });
                    if (entry.keywords) fields.push({ label: "Keywords", value: entry.keywords });
                    if (entry.bib_abstract) fields.push({ label: "Abstract", value: entry.bib_abstract });
                    if (entry.duplicate_files && entry.duplicate_files.length >= 2) {
                      // Show all duplicate file paths so the user can identify copies.
                      for (const path of entry.duplicate_files) {
                        fields.push({ label: "Duplicate File", value: path });
                      }
                    } else if (entry.existing_file) {
                      fields.push({ label: "File", value: entry.existing_file });
                    }
                    if (entry.expected_filename) fields.push({ label: "Expected Filename", value: entry.expected_filename });
                    // Append extra BibTeX fields (journal, volume, pages, etc.)
                    if (entry.extra_fields) {
                      for (const [key, val] of Object.entries(entry.extra_fields)) {
                        fields.push({ label: key, value: val });
                      }
                    }
                    return fields;
                  };

                  return (
                    <div
                      style={{
                        padding: "8px 12px",
                        "border-bottom": "1px solid var(--color-glass-border)",
                        "font-size": "12px",
                        opacity: entry.file_exists ? "0.6" : "1",
                        cursor: "pointer",
                        transition: "background 0.15s",
                      }}
                      onClick={() => toggleExpanded(entry.cite_key)}
                      onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.03)"; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = ""; }}
                    >
                      {/* Header row: cite_key on the left, type + status badges right-aligned */}
                      <div style={{ display: "flex", "align-items": "center", "margin-bottom": "3px" }}>
                        {/* Expand/collapse indicator */}
                        <span style={{
                          color: "var(--color-text-muted)",
                          "font-size": "10px",
                          width: "14px",
                          "flex-shrink": "0",
                          "user-select": "none",
                        }}>
                          {isExpanded() ? "\u25BC" : "\u25B6"}
                        </span>
                        <span style={{ "font-weight": "600", color: "var(--color-text-primary)", "font-family": "monospace" }}>
                          {entry.cite_key}
                        </span>
                        <span style={{ "margin-left": "auto", display: "flex", "align-items": "center", gap: "6px" }}>
                          {/* Link type column: URL, DOI, or no link */}
                          <span class="badge" style={{
                            background: `color-mix(in srgb, ${badge.color} 15%, transparent)`,
                            color: badge.color,
                            "min-width": "36px",
                            "text-align": "center",
                          }}>
                            {badge.label}
                          </span>
                          {/* File status column: duplicate, exists, or missing */}
                          <Show when={entry.file_exists && entry.duplicate_files && entry.duplicate_files.length >= 2}>
                            <span class="badge" style={{
                              background: "color-mix(in srgb, var(--color-accent-amber) 15%, transparent)",
                              color: "var(--color-accent-amber)",
                              "min-width": "52px",
                              "text-align": "center",
                            }}>
                              duplicate
                            </span>
                          </Show>
                          <Show when={entry.file_exists && !(entry.duplicate_files && entry.duplicate_files.length >= 2)}>
                            <span class="badge" style={{
                              background: "color-mix(in srgb, var(--color-accent-cyan) 15%, transparent)",
                              color: "var(--color-accent-cyan)",
                              "min-width": "52px",
                              "text-align": "center",
                            }}>
                              exists
                            </span>
                          </Show>
                          <Show when={!entry.file_exists && (entry.has_url || entry.has_doi)}>
                            <span class="badge" style={{
                              background: "color-mix(in srgb, var(--color-accent-magenta) 15%, transparent)",
                              color: "var(--color-accent-magenta)",
                              "min-width": "52px",
                              "text-align": "center",
                            }}>
                              missing
                            </span>
                          </Show>
                          <Show when={!entry.file_exists && !entry.has_url && !entry.has_doi}>
                            <span style={{ "min-width": "52px" }} />
                          </Show>
                        </span>
                      </div>

                      {/* Title */}
                      <div style={{
                        color: "var(--color-text-secondary)",
                        "line-height": "1.3",
                        "padding-left": "14px",
                      }}>
                        {entry.title}
                      </div>

                      {/* Author (always visible in collapsed state) */}
                      <Show when={entry.author}>
                        <div style={{
                          color: "var(--color-text-muted)",
                          "font-size": "11px",
                          "margin-top": "2px",
                          "padding-left": "14px",
                        }}>
                          {entry.author}
                        </div>
                      </Show>

                      {/* Expanded detail view: all BibTeX fields */}
                      <Show when={isExpanded()}>
                        <div style={{
                          "margin-top": "6px",
                          "padding-left": "14px",
                          "padding-top": "6px",
                          "border-top": "1px solid rgba(255,255,255,0.06)",
                        }}>
                          <For each={detailFields()}>
                            {(field) => (
                              <div style={{
                                display: "flex",
                                gap: "8px",
                                "margin-bottom": "4px",
                                "font-size": "11px",
                                "line-height": "1.4",
                              }}>
                                <span style={{
                                  color: "var(--color-text-muted)",
                                  "min-width": "70px",
                                  "flex-shrink": "0",
                                  "text-transform": "capitalize",
                                }}>
                                  {field.label}:
                                </span>
                                <span style={{
                                  color: field.label === "URL" || field.label === "DOI"
                                    ? "var(--color-accent-cyan)"
                                    : "var(--color-text-secondary)",
                                  "word-break": "break-all",
                                }}>
                                  {field.value}
                                </span>
                              </div>
                            )}
                          </For>
                        </div>
                      </Show>
                    </div>
                  );
                }}
              </For>
            </div>
          </Show>
        }
      >
        <div style={{ display: "flex", "align-items": "center" }}>
          <div class="section-title" style={{ flex: "1" }}>
            Results
            <Tip text="Per-entry download results streamed in real-time. Status colors: cyan = downloaded/fetched successfully, magenta = failed (network error or unavailable), amber = blocked (paywall or bot detection), purple = skipped (already exists). Each entry transitions from pending through downloading to its final status." />
            <Show when={fetching()}>
              <span style={{
                "font-weight": "400",
                "font-size": "12px",
                color: "var(--color-accent-cyan)",
                "margin-left": "8px",

              }}>
                fetching...
              </span>
            </Show>
            <Show when={!fetching() && results().length > 0}>
              <span style={{
                "font-weight": "400",
                "font-size": "12px",
                color: "var(--color-text-muted)",
                "margin-left": "8px",
              }}>
                {results().length} entries
              </span>
            </Show>
          </div>
          {/* Switch back to the BibTeX preview. Clears results and triggers
              a fresh re-parse to update file existence flags from disk. */}
          <Show when={!fetching()}>
            <button
              class="btn btn-sm"
              style={{
                "font-size": "11px",
                padding: "3px 10px",
                "white-space": "nowrap",
              }}
              onClick={() => {
                setResults([]);
                setSummary(null);
                parseBibFile(bibPath(), outputDir());
              }}
            >
              Back to Preview
            </button>
          </Show>
        </div>

        <div style={{
          display: "flex",
          "flex-direction": "column",
          gap: "1px",
          flex: "1",
          "min-height": "0",
          "overflow-y": "auto",
          "margin-top": "8px",
        }}>
          <For each={results()}>
            {(r) => (
              <div style={{
                padding: "8px 12px",
                "border-bottom": "1px solid var(--color-glass-border)",
                "font-size": "12px",
              }}>
                {/* Header row: cite_key + status badge */}
                <div style={{ display: "flex", "align-items": "center", gap: "8px", "margin-bottom": "4px" }}>
                  <span style={{ "font-weight": "600", color: "var(--color-text-primary)" }}>
                    {r.cite_key}
                  </span>
                  <span class="badge" style={{
                    background: `color-mix(in srgb, ${statusColor(r.status)} 15%, transparent)`,
                    color: statusColor(r.status),
                  }}>
                    {r.status}
                  </span>
                  <Show when={r.type && r.type !== "unknown"}>
                    <span class="badge badge-gray">
                      {r.type}
                    </span>
                  </Show>
                  <Show when={r.doi_resolved_via}>
                    <span class="badge badge-purple">
                      via {r.doi_resolved_via}
                    </span>
                  </Show>
                </div>

                {/* URL (truncated) */}
                <Show when={r.url}>
                  <div style={{
                    color: "var(--color-text-muted)",
                    "white-space": "nowrap",
                    overflow: "hidden",
                    "text-overflow": "ellipsis",
                    "max-width": "100%",
                  }}>
                    {r.url}
                  </div>
                </Show>

                {/* Error message if failed */}
                <Show when={r.error}>
                  <div style={{ color: "var(--color-accent-magenta)", "margin-top": "2px" }}>
                    {r.error}
                  </div>
                </Show>

                {/* Block reason if blocked */}
                <Show when={r.reason}>
                  <div style={{ color: "var(--color-accent-amber)", "margin-top": "2px" }}>
                    {r.reason}
                  </div>
                </Show>

                {/* HTML title if available */}
                <Show when={r.title}>
                  <div style={{ color: "var(--color-text-secondary)", "margin-top": "2px" }}>
                    {r.title}
                  </div>
                </Show>
              </div>
            )}
          </For>
        </div>
      </Show>
    </div>
  );

  return (
    <>
      <SplitPane
        left={leftPanel()}
        right={rightPanel()}
        leftWidth={state.fetchLeftPanelWidth}
        onResize={(w) => actions.setFetchLeftPanelWidth(w)}
        minLeft={240}
        minRight={300}
      />

      {/* Shared browse modal rendered via the BrowseModal composable */}
      <browse.BrowseModalUI config={browseConfig()} />
    </>
  );
};

export default SourcesTab;
