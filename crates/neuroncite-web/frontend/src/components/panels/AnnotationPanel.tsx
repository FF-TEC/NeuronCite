/**
 * Annotation panel for standalone PDF highlighting from a previously
 * exported annotation CSV file. Provides a SplitPane layout consistent
 * with the Citations, Search, and Sources tabs:
 *
 *   Left panel:  File selection (annotation CSV, source PDF directory,
 *                output directory) and the start button with status message.
 *   Right panel: Job progress bar, state badge, and error display.
 *
 * This panel calls POST /api/v1/annotate/from-file to create an annotation
 * job from a file on disk. Job progress is tracked via SSE job_update events
 * that are subscribed at the App.tsx root level and routed to the store
 * fields (annotationJobId, annotationJobState, annotationProgressDone,
 * annotationProgressTotal, annotationErrorMessage).
 *
 * The annotation pipeline reads the CSV, locates matching text in the
 * source PDFs using a 5-stage strategy (exact, normalized, fuzzy, fallback,
 * OCR), and produces highlighted PDF copies in the output directory.
 *
 * Browse modal functionality is provided by the shared BrowseModal composable
 * (createBrowseModal) instead of duplicating browse signals and JSX locally.
 */

import { Component, Show, createSignal } from "solid-js";
import { state, actions } from "../../stores/app";
import { api } from "../../api/client";
import Tip from "../ui/Tip";
import SplitPane from "../layout/SplitPane";
import { createBrowseModal } from "../ui/BrowseModal";
import type { BrowseModalConfig } from "../ui/BrowseModal";

const AnnotationPanel: Component = () => {
  // Local form input signals (panel-scoped, not in the global store).
  const [inputFile, setInputFile] = createSignal("");
  const [sourceDir, setSourceDir] = createSignal("");
  const [outputDir, setOutputDir] = createSignal("");
  const [statusMessage, setStatusMessage] = createSignal("");
  const [isStarting, setIsStarting] = createSignal(false);

  // Shared inline styles for consistent form layout with CitationPanel.
  const labelStyle = { "min-width": "70px", color: "var(--color-text-secondary)", "font-size": "12px" } as const;

  // ---- Browse modal (shared composable) ----

  const browse = createBrowseModal({
    onSelect: (path, mode) => {
      if (mode === "csv") setInputFile(path);
      else if (mode === "source") setSourceDir(path);
      else setOutputDir(path);
    },
  });

  /** Returns the BrowseModalConfig for the current browse mode.
   *  "csv" mode filters for .csv files; "source" and "output" modes
   *  select folders with the "Select This Folder" button visible. */
  const browseConfig = (): BrowseModalConfig => {
    const mode = browse.browseMode();
    if (mode === "csv") {
      return { title: "Select Annotation CSV", fileExtension: ".csv", showFolderSelect: false };
    }
    if (mode === "source") {
      return { title: "Select Source PDF Directory", fileExtension: "", showFolderSelect: true };
    }
    return { title: "Select Output Directory", fileExtension: "", showFolderSelect: true };
  };

  // ------------------------------------------------------------------
  // File/folder browsing via native OS dialogs with BrowseModal fallback
  // ------------------------------------------------------------------

  /** Opens a native OS file dialog for annotation CSV file selection.
   *  Falls back to the custom browser modal if native is not available. */
  const openFileBrowser = async () => {
    const startPath = inputFile().replace(/[\\/][^\\/]*$/, "");
    const selected = await browse.openNativeFileDialog("csv", startPath, "csv");
    if (selected) {
      setInputFile(selected);
    }
  };

  /** Opens a native OS folder dialog for source or output directory.
   *  Falls back to the custom browser modal if native is not available. */
  const openDirBrowser = async (mode: "source" | "output") => {
    const startPath = mode === "source" ? sourceDir() : outputDir();
    const selected = await browse.openNativeFolderDialog(mode, startPath);
    if (selected) {
      if (mode === "source") setSourceDir(selected);
      else setOutputDir(selected);
    }
  };

  // ------------------------------------------------------------------
  // Start annotation job
  // ------------------------------------------------------------------

  /** Sends the annotation request to the backend and stores the returned
   *  job ID in the global store for SSE progress tracking. */
  const startAnnotation = async () => {
    if (!inputFile()) {
      setStatusMessage("Select an annotation CSV file first.");
      return;
    }
    if (!sourceDir()) {
      setStatusMessage("Select the source PDF directory.");
      return;
    }
    if (!outputDir()) {
      setStatusMessage("Select an output directory for annotated PDFs.");
      return;
    }

    setIsStarting(true);
    setStatusMessage("Starting annotation job...");
    try {
      const res = await api.annotateFromFile({
        input_file: inputFile(),
        source_directory: sourceDir(),
        output_directory: outputDir(),
      });
      actions.setAnnotationJob(res.job_id, "queued");
      actions.setAnnotationProgress(0, res.total_quotes);
      setStatusMessage(`Annotation job created: ${res.total_quotes} quotes to process.`);
    } catch (e) {
      setStatusMessage(`Annotation failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setIsStarting(false);
    }
  };

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  const progressPct = (): number => {
    if (state.annotationProgressTotal === 0) return 0;
    return Math.round((state.annotationProgressDone / state.annotationProgressTotal) * 100);
  };

  /** Maps job state strings to display colors matching the neon tech palette. */
  const stateColor = (s: string): string => {
    if (s === "completed") return "var(--color-status-success)";
    if (s === "running") return "var(--color-accent-cyan)";
    if (s === "queued") return "var(--color-text-muted)";
    if (s === "failed") return "var(--color-status-error)";
    return "var(--color-text-muted)";
  };

  // ------------------------------------------------------------------
  // Left panel: file selection + settings + controls
  // ------------------------------------------------------------------

  const leftPanel = () => (
    <div style={{ display: "flex", "flex-direction": "column", gap: "16px", height: "100%", "overflow-y": "auto", "min-height": "0" }}>

      {/* Section 1: File Selection */}
      <div class="glass-card" style={{ padding: "16px" }}>
        <div class="section-title">
          File Selection
          <Tip text="Select the annotation CSV file exported by the citation verification pipeline, the source PDF directory containing the original documents, and an output directory for the annotated copies." />
        </div>

        {/* Annotation CSV file path */}
        <div style={{ display: "flex", gap: "8px", "align-items": "center", "margin-bottom": "8px" }}>
          <label style={labelStyle}>
            CSV File
            <Tip text="Path to the annotation_pipeline_input.csv file produced by the citation export. Contains quote text, page numbers, and color codes for each highlight." />
          </label>
          <input
            class="input"
            type="text"
            placeholder="/path/to/annotation_pipeline_input.csv"
            value={inputFile()}
            onInput={(e) => setInputFile(e.target.value)}
            style={{ flex: "1" }}
          />
          <button class="btn btn-sm" onClick={openFileBrowser} disabled={browse.dialogPending()}>
            {browse.dialogPending() ? "..." : "Browse"}
          </button>
        </div>

        {/* Source PDF directory */}
        <div style={{ display: "flex", gap: "8px", "align-items": "center", "margin-bottom": "8px" }}>
          <label style={labelStyle}>
            Sources
            <Tip text="Directory containing the original source PDFs referenced in the annotation CSV. The pipeline matches filenames from the CSV against files in this directory." />
          </label>
          <input
            class="input"
            type="text"
            placeholder="/path/to/source/pdfs/"
            value={sourceDir()}
            onInput={(e) => setSourceDir(e.target.value)}
            style={{ flex: "1" }}
          />
          <button class="btn btn-sm" onClick={() => openDirBrowser("source")} disabled={browse.dialogPending()}>
            {browse.dialogPending() ? "..." : "Browse"}
          </button>
        </div>

        {/* Output directory */}
        <div style={{ display: "flex", gap: "8px", "align-items": "center" }}>
          <label style={labelStyle}>
            Output
            <Tip text="Directory where annotated PDF copies are saved. Each source PDF with matching quotes is copied here with highlights applied. The directory is created automatically if it does not exist." />
          </label>
          <input
            class="input"
            type="text"
            placeholder="/path/to/output/"
            value={outputDir()}
            onInput={(e) => setOutputDir(e.target.value)}
            style={{ flex: "1" }}
          />
          <button class="btn btn-sm" onClick={() => openDirBrowser("output")} disabled={browse.dialogPending()}>
            {browse.dialogPending() ? "..." : "Browse"}
          </button>
        </div>
      </div>

      {/* Section 2: Controls */}
      <div class="glass-card" style={{ padding: "16px" }}>
        <div class="section-title">
          Controls
          <Tip text="Start the annotation job. The pipeline reads the CSV, locates matching text in the source PDFs using multi-stage text matching, and produces highlighted PDF copies in the output directory." />
        </div>

        <button
          class="btn btn-primary"
          style={{ width: "100%" }}
          onClick={startAnnotation}
          disabled={
            isStarting()
            || !inputFile()
            || !sourceDir()
            || !outputDir()
            || state.annotationJobState === "running"
            || state.annotationJobState === "queued"
          }
        >
          {isStarting()
            ? "Starting..."
            : state.annotationJobState === "running" || state.annotationJobState === "queued"
              ? "Annotation in progress..."
              : "Start Annotation"}
        </button>

        <Show when={statusMessage()}>
          <div style={{
            "margin-top": "8px",
            "font-size": "12px",
            color: "var(--color-text-secondary)",
          }}>
            {statusMessage()}
          </div>
        </Show>
      </div>
    </div>
  );

  // ------------------------------------------------------------------
  // Right panel: progress and job status
  // ------------------------------------------------------------------

  const rightPanel = () => (
    <div style={{ display: "flex", "flex-direction": "column", gap: "0", height: "100%", "overflow-y": "auto" }}>

      {/* Progress bar and job state badge */}
      <Show when={state.annotationJobId}>
        <div style={{ padding: "16px", "border-bottom": "1px solid var(--color-glass-border)" }}>

          {/* Job state badge */}
          <div style={{ display: "flex", "align-items": "center", gap: "8px", "margin-bottom": "8px" }}>
            <span style={{
              display: "inline-block",
              padding: "2px 10px",
              "border-radius": "100px",
              "font-size": "12px",
              color: "#fff",
              background: stateColor(state.annotationJobState),
            }}>
              {state.annotationJobState || "unknown"}
            </span>
            <span style={{ color: "var(--color-text-secondary)", "font-size": "13px" }}>
              {state.annotationProgressDone} / {state.annotationProgressTotal} ({progressPct()}%)
            </span>
          </div>

          {/* Progress bar (visible during running/queued states) */}
          <Show when={state.annotationJobState === "running" || state.annotationJobState === "queued"}>
            <div style={{
              height: "4px",
              background: "var(--color-bg-tertiary)",
              "border-radius": "2px",
              overflow: "hidden",
            }}>
              <div style={{
                height: "100%",
                width: `${progressPct()}%`,
                background: "linear-gradient(90deg, var(--color-accent-purple), var(--color-accent-cyan))",
                transition: "width 0.3s ease",
              }} />
            </div>
          </Show>

          {/* Error message display */}
          <Show when={state.annotationErrorMessage}>
            <div style={{
              "margin-top": "8px",
              padding: "8px 12px",
              background: "rgba(239, 68, 68, 0.1)",
              "border-radius": "6px",
              "font-size": "12px",
              color: "var(--color-status-error)",
            }}>
              {state.annotationErrorMessage}
            </div>
          </Show>

          {/* Completion message */}
          <Show when={state.annotationJobState === "completed"}>
            <div style={{
              "margin-top": "8px",
              "font-size": "13px",
              color: "var(--color-status-success)",
            }}>
              Annotation complete. Check the output directory for highlighted PDFs.
            </div>
          </Show>
        </div>
      </Show>

      {/* Empty state when no job is active */}
      <Show when={!state.annotationJobId}>
        <div style={{
          display: "flex",
          "align-items": "center",
          "justify-content": "center",
          height: "100%",
          color: "var(--color-text-muted)",
          "font-size": "13px",
          padding: "32px",
          "text-align": "center",
        }}>
          Select an annotation CSV file, source PDF directory, and output directory, then click "Start Annotation" to highlight matching quotes in the PDFs.
        </div>
      </Show>
    </div>
  );

  // ------------------------------------------------------------------
  // Main render: SplitPane + browse modal
  // ------------------------------------------------------------------

  return (
    <>
      <SplitPane
        left={leftPanel()}
        right={rightPanel()}
        leftWidth={state.annotationLeftPanelWidth}
        onResize={(w) => actions.setAnnotationLeftPanelWidth(w)}
        minLeft={240}
        minRight={300}
      />

      {/* Shared browse modal rendered via the BrowseModal composable */}
      <browse.BrowseModalUI config={browseConfig()} />
    </>
  );
};

export default AnnotationPanel;
