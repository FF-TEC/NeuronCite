/**
 * Centralized logging utility that gates console output by the current
 * build mode. In development builds (import.meta.env.DEV === true), all
 * log levels are active. In production builds, only "warn" and "error"
 * levels produce output. This prevents verbose debug/info messages from
 * cluttering the browser console in deployed environments.
 *
 * Each function accepts a `context` string identifying the call site
 * (e.g., "App.onMount", "SearchTab.fetchResults") which is prepended
 * to the log message in square brackets for grep-ability.
 *
 * Migration note: This module is intended to replace direct console.error,
 * console.warn, and console.log calls across the frontend codebase. The
 * initial migration covers App.tsx. Remaining files (SearchTab, IndexingTab,
 * SourcesTab, CitationPanel, etc.) should adopt these functions incrementally.
 */

/** Log level threshold derived from the Vite build mode. Development
 *  builds emit all levels; production builds suppress debug and info. */
const LOG_LEVEL: "debug" | "warn" = import.meta.env.DEV ? "debug" : "warn";

/**
 * Logs an error with a context label. Active in both development and
 * production builds because errors always warrant visibility.
 *
 * @param context - Identifier for the call site (e.g., "App.onMount").
 * @param error - The error value to log. Accepts any type because catch
 *   blocks produce `unknown`.
 */
export function logError(context: string, error: unknown): void {
  if (LOG_LEVEL === "debug" || LOG_LEVEL === "warn") {
    console.error(`[${context}]`, error);
  }
}

/**
 * Logs a warning with a context label. Active in both development and
 * production builds because warnings indicate conditions that may
 * require attention.
 *
 * @param context - Identifier for the call site.
 * @param message - The warning message string.
 */
export function logWarn(context: string, message: string): void {
  if (LOG_LEVEL === "debug" || LOG_LEVEL === "warn") {
    console.warn(`[${context}]`, message);
  }
}

/**
 * Logs a debug message with a context label. Active only in development
 * builds. Suppressed entirely in production to avoid console noise.
 *
 * @param context - Identifier for the call site.
 * @param message - The debug message string.
 */
export function logDebug(context: string, message: string): void {
  if (LOG_LEVEL === "debug") {
    console.log(`[${context}]`, message);
  }
}
