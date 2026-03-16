/**
 * Reusable file/folder browser modal composable for NeuronCite.
 *
 * Four components (SourcesTab, CitationPanel, IndexingTab, AnnotationPanel)
 * independently implemented ~100-150 lines of identical browse modal signals,
 * navigation logic, and JSX. This module consolidates that code into a single
 * composable function and a JSX component.
 *
 * Usage:
 *   const { browseOpen, openBrowser, selectBrowsePath, BrowseModalUI } = createBrowseModal({
 *     onSelect: (path) => { ... },
 *   });
 *
 * The composable returns reactive signals and functions that the consuming
 * component integrates into its own event handlers and render tree.
 *
 * The BrowseModalUI component renders the modal backdrop, navigation controls,
 * error display, and directory entry list. It must be placed inside the
 * consuming component's JSX tree (typically at the end, after the main content).
 *
 * The browseMode signal supports arbitrary string values. The consuming
 * component decides what modes it needs (e.g., "file" | "folder", or
 * "tex" | "bib" | "folder"). The modal title, file extension filter, and
 * select button visibility are controlled via the BrowseModalConfig passed
 * to BrowseModalUI.
 *
 * Accessibility: the modal container uses role="dialog" and aria-modal="true"
 * to announce it as a dialog to screen readers. Pressing the Escape key
 * dismisses the modal. When the modal opens, initial focus is placed on
 * the modal container so keyboard navigation starts inside it.
 */

import { createSignal, Component, For, Show, onCleanup } from "solid-js";
import { api } from "../../api/client";
import { logWarn } from "../../utils/logger";
import type { DirEntry } from "../../api/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Configuration for the BrowseModalUI component, provided by the consumer
 *  to control modal behavior for the current browse mode. */
export interface BrowseModalConfig {
  /** Title displayed in the modal header (e.g., "Select BibTeX File"). */
  title: string;
  /** File extension filter (e.g., ".bib", ".tex", ".csv"). Empty string
   *  means no file selection is possible (folder-only mode). */
  fileExtension: string;
  /** Whether the "Select This Folder" button is visible. True for folder
   *  browse modes, false for file-only modes. */
  showFolderSelect: boolean;
}

/** Options passed to createBrowseModal. */
export interface CreateBrowseModalOptions {
  /** Callback invoked when the user confirms a selection (file or folder). */
  onSelect: (path: string, mode: string) => void;
}

/** Return type of createBrowseModal. */
export interface BrowseModalHandle {
  /** Signal indicating whether the browse modal is currently visible. */
  browseOpen: () => boolean;
  /** Signal indicating whether a native OS dialog is pending. */
  dialogPending: () => boolean;
  /** Opens the custom file browser modal with the given mode string.
   *  Navigates to startPath as the initial directory listing. */
  openBrowser: (mode: string, startPath: string) => Promise<void>;
  /** Opens a native OS file dialog (POST /web/browse/native-file) with an
   *  optional filter query parameter. Falls back to the custom modal with
   *  the specified mode and startPath if the native dialog is unavailable. */
  openNativeFileDialog: (
    mode: string,
    startPath: string,
    filter?: string,
  ) => Promise<string | null>;
  /** Opens a native OS folder dialog (POST /web/browse/native). Falls back
   *  to the custom modal with the specified mode and startPath if the native
   *  dialog is unavailable. */
  openNativeFolderDialog: (
    mode: string,
    startPath: string,
  ) => Promise<string | null>;
  /** Confirms the selected path in the browse modal and closes it.
   *  When called without arguments in folder mode, uses the current browsePath.
   *  When called with a specific path (file selection), uses that path. */
  selectBrowsePath: (path?: string) => void;
  /** The current browse mode string set by the last openBrowser call. */
  browseMode: () => string;
  /** The BrowseModalUI component to render inside the consuming component's JSX. */
  BrowseModalUI: Component<{ config: BrowseModalConfig }>;
}

// ---------------------------------------------------------------------------
// Composable
// ---------------------------------------------------------------------------

/**
 * Creates a browse modal composable encapsulating all state, navigation
 * logic, and the modal UI component. Each consuming component calls this
 * once and integrates the returned handle into its own logic.
 */
export function createBrowseModal(options: CreateBrowseModalOptions): BrowseModalHandle {
  const [browseOpen, setBrowseOpen] = createSignal(false);
  const [browsePath, setBrowsePath] = createSignal("");
  const [browseEntries, setBrowseEntries] = createSignal<DirEntry[]>([]);
  const [browseParent, setBrowseParent] = createSignal("");
  const [browseDrives, setBrowseDrives] = createSignal<string[]>([]);
  const [browseError, setBrowseError] = createSignal("");
  const [dialogPending, setDialogPending] = createSignal(false);
  const [browseMode, setBrowseMode] = createSignal("");

  /** Fetches the directory listing for the given path from the backend
   *  and updates all browse modal signals. Empty path returns drive roots. */
  const navigateTo = async (path: string) => {
    setBrowseError("");
    try {
      const res = await api.browse(path);
      setBrowsePath(res.path);
      setBrowseEntries(res.entries);
      setBrowseParent(res.parent);
      setBrowseDrives(res.drives);
    } catch (e) {
      setBrowseError(String(e));
    }
  };

  /** Opens the custom file browser modal with the given mode and navigates
   *  to startPath as the initial directory listing. */
  const openBrowser = async (mode: string, startPath: string) => {
    setBrowseMode(mode);
    setBrowseOpen(true);
    await navigateTo(startPath || "");
  };

  /** Opens a native OS file dialog via the typed API client. Returns the
   *  selected path on success, or null if the user cancelled. Falls back
   *  to the custom modal if the native dialog is unavailable (e.g.,
   *  headless environment). */
  const openNativeFileDialog = async (
    mode: string,
    startPath: string,
    filter?: string,
  ): Promise<string | null> => {
    if (dialogPending()) return null;
    setDialogPending(true);
    try {
      const data = await api.browseNativeFile(filter);
      if (data.selected && data.path) {
        return data.path;
      }
      // User cancelled the native dialog.
      return null;
    } catch {
      // Native dialog not available (503 from server, network error, or
      // timeout). Fall through to the custom browser modal as fallback.
      logWarn("BrowseModal.openNativeFileDialog", "Native file dialog unavailable, falling back to custom browser.");
    } finally {
      setDialogPending(false);
    }
    setBrowseError("");
    await openBrowser(mode, startPath);
    return null;
  };

  /** Opens a native OS folder dialog via the typed API client. Returns the
   *  selected path on success, or null if the user cancelled. Falls back
   *  to the custom modal if the native dialog is unavailable (non-GUI mode,
   *  headless environment, or rfd failure). The fallback opens with a clean
   *  state (browseError cleared) so no stale error messages are shown. */
  const openNativeFolderDialog = async (
    mode: string,
    startPath: string,
  ): Promise<string | null> => {
    if (dialogPending()) return null;
    setDialogPending(true);
    try {
      const data = await api.browseNativeFolder();
      if (data.selected && data.path) {
        return data.path;
      }
      // User cancelled the native dialog.
      return null;
    } catch {
      // Native dialog not available (503 from server, network error, or
      // timeout). Fall through to the custom browser modal as fallback.
      logWarn("BrowseModal.openNativeFolderDialog", "Native folder dialog unavailable, falling back to custom browser.");
    } finally {
      setDialogPending(false);
    }
    setBrowseError("");
    await openBrowser(mode, startPath);
    return null;
  };

  /** Confirms the selected path in the browse modal and closes it.
   *  Delegates to the onSelect callback with the current mode. */
  const selectBrowsePath = (path?: string) => {
    const selected = path || browsePath();
    options.onSelect(selected, browseMode());
    setBrowseOpen(false);
  };

  /** Constructs the full child path by appending an entry name to the
   *  current browse path, using the appropriate path separator. */
  const buildChildPath = (entryName: string): string => {
    const base = browsePath();
    if (!base) return entryName;
    const sep = base.includes("/") ? "/" : "\\";
    const endsWithSep = base.endsWith("\\") || base.endsWith("/");
    return `${base}${endsWithSep ? "" : sep}${entryName}`;
  };

  /** The modal UI component rendered by the consuming component.
   *  Uses role="dialog" and aria-modal="true" for screen reader semantics.
   *  Pressing Escape dismisses the modal. Initial focus is placed on the
   *  modal container when it becomes visible. */
  const BrowseModalUI: Component<{ config: BrowseModalConfig }> = (props) => {
    /** Reference to the modal container div for programmatic focus management.
     *  Focus is moved here when the modal opens so keyboard navigation starts
     *  inside the dialog instead of behind the backdrop. */
    let modalRef: HTMLDivElement | undefined;

    /** Handles keydown events on the modal container. Escape key dismisses
     *  the modal, matching the expected behavior for dialog components. */
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setBrowseOpen(false);
      }
    };

    return (
      <Show when={browseOpen()}>
        <div class="modal-backdrop" onClick={() => setBrowseOpen(false)}>
          <div
            class="modal"
            ref={(el) => {
              modalRef = el;
              // Place initial focus on the modal container after render.
              // The requestAnimationFrame ensures the DOM is settled before
              // calling focus(), which is necessary for SolidJS Show transitions.
              requestAnimationFrame(() => modalRef?.focus());
            }}
            role="dialog"
            aria-modal="true"
            aria-label={props.config.title}
            tabIndex={-1}
            onKeyDown={onKeyDown}
            onClick={(e) => e.stopPropagation()}
          >
            <div class="modal-header">
              <span style={{ "font-weight": "600" }}>
                {props.config.title}
              </span>
              <button class="btn btn-sm" onClick={() => setBrowseOpen(false)}>
                X
              </button>
            </div>

            {/* Current path breadcrumb */}
            <div style={{ padding: "8px 16px", "font-size": "12px", color: "var(--color-text-secondary)" }}>
              {browsePath() || "Root"}
            </div>

            {/* Navigation: Up button + folder select button (when applicable) */}
            <div style={{ padding: "0 16px 8px", display: "flex", gap: "8px" }}>
              <Show when={browsePath()}>
                <button class="btn btn-sm" onClick={() => navigateTo(browseParent())}>
                  Up
                </button>
              </Show>
              <Show when={props.config.showFolderSelect}>
                <button
                  class="btn btn-sm btn-primary"
                  onClick={() => selectBrowsePath()}
                  disabled={!browsePath()}
                >
                  Select This Folder
                </button>
              </Show>
            </div>

            {/* Browse error display */}
            <Show when={browseError()}>
              <div style={{ padding: "0 16px 8px", color: "var(--color-accent-magenta)", "font-size": "12px" }}>
                {browseError()}
              </div>
            </Show>

            {/* Drive letter buttons (Windows, visible at root level) */}
            <Show when={browseDrives().length > 0 && !browsePath()}>
              <div style={{ padding: "0 16px 8px", display: "flex", gap: "4px", "flex-wrap": "wrap" }}>
                <For each={browseDrives()}>
                  {(drive) => (
                    <button class="btn btn-sm" onClick={() => navigateTo(drive)}>
                      {drive}
                    </button>
                  )}
                </For>
              </div>
            </Show>

            {/* Directory listing with scrollable entries */}
            <div style={{ "max-height": "400px", "overflow-y": "auto", padding: "0 16px 16px" }}>
              <For each={browseEntries()}>
                {(entry) => {
                  const isDir = entry.kind === "directory";
                  const ext = props.config.fileExtension;
                  const isMatchingFile = !isDir && ext && entry.name.toLowerCase().endsWith(ext);
                  const isClickable = isDir || (ext && isMatchingFile);

                  return (
                    <div
                      style={{
                        padding: "6px 8px",
                        cursor: isClickable ? "pointer" : "default",
                        "font-size": "13px",
                        "border-radius": "4px",
                        display: "flex",
                        "align-items": "center",
                        gap: "8px",
                      }}
                      class={isClickable ? "browse-entry" : ""}
                      onClick={() => {
                        if (isDir) {
                          navigateTo(buildChildPath(entry.name));
                        } else if (isMatchingFile && ext) {
                          selectBrowsePath(buildChildPath(entry.name));
                        }
                      }}
                    >
                      <span style={{
                        color: isDir
                          ? "var(--color-accent-cyan)"
                          : isMatchingFile
                            ? "var(--color-accent-purple)"
                            : "var(--color-text-muted)",
                      }}>
                        {isDir ? "[DIR]" : "[FILE]"}
                      </span>
                      <span style={{
                        color: isClickable ? "var(--color-text-primary)" : "var(--color-text-muted)",
                      }}>
                        {entry.name}
                      </span>
                    </div>
                  );
                }}
              </For>
            </div>
          </div>
        </div>
      </Show>
    );
  };

  return {
    browseOpen,
    dialogPending,
    openBrowser,
    openNativeFileDialog,
    openNativeFolderDialog,
    selectBrowsePath,
    browseMode,
    BrowseModalUI,
  };
}
