/**
 * Shared constants, types, and preset configurations for the Citation
 * verification panel and its sub-components. Extracted from CitationPanel.tsx
 * to avoid circular dependencies between the parent and child components.
 */

// ---------------------------------------------------------------------------
// Color mappings
// ---------------------------------------------------------------------------

/** Color mapping for citation verdicts. Each verdict type has a distinct
 *  color in the neon tech palette for visual identification in badges,
 *  summary pills, and the results table. */
export const VERDICT_COLORS: Record<string, string> = {
  supported: "var(--color-status-success)",
  partial: "var(--color-status-warning)",
  unsupported: "var(--color-status-error)",
  not_found: "var(--color-text-muted)",
  wrong_source: "var(--color-accent-magenta)",
  unverifiable: "var(--color-accent-purple)",
  peripheral_match: "var(--color-accent-purple-light)",
};

/** Background color for phase badges in the status column. Maps the
 *  autonomous agent's processing phases to semantic colors. Active phases
 *  (searching, evaluating, reasoning) receive the pulse animation. */
export const PHASE_COLORS: Record<string, string> = {
  pending: "var(--color-text-muted)",
  searching: "var(--color-accent-cyan)",
  evaluating: "var(--color-accent-purple)",
  reasoning: "var(--color-accent-purple)",
  done: "var(--color-status-success)",
  error: "var(--color-status-error)",
};

// ---------------------------------------------------------------------------
// Verification mode presets
// ---------------------------------------------------------------------------

/** Verification parameter set applied by each preset mode. Controls search
 *  depth, LLM reasoning length, retry behavior, and reranking. */
export interface VerifyParams {
  top_k: number;
  cross_corpus_queries: number;
  max_tokens: number;
  temperature: number;
  max_retry_attempts: number;
  min_score: number | undefined;
  rerank: boolean;
}

/** Union type for the four verification modes. Three presets (quick,
 *  standard, thorough) plus a custom mode with manual sliders. */
export type VerifyMode = "quick" | "standard" | "thorough" | "custom";

/** Preset configurations for the three built-in verification modes.
 *  Custom mode does not have a preset; it uses the individual signal values
 *  from the VerificationModeSection component. */
export const PRESETS: Record<Exclude<VerifyMode, "custom">, VerifyParams> = {
  quick: {
    top_k: 3,
    cross_corpus_queries: 1,
    max_tokens: 2048,
    temperature: 0.1,
    max_retry_attempts: 2,
    min_score: undefined,
    rerank: false,
  },
  standard: {
    top_k: 5,
    cross_corpus_queries: 2,
    max_tokens: 4096,
    temperature: 0.1,
    max_retry_attempts: 3,
    min_score: undefined,
    rerank: false,
  },
  thorough: {
    top_k: 10,
    cross_corpus_queries: 3,
    max_tokens: 8192,
    temperature: 0.05,
    max_retry_attempts: 5,
    min_score: 0.5,
    rerank: true,
  },
};

/** Human-readable labels and descriptions for each verification mode.
 *  Displayed in the mode selector dropdown. */
export const MODE_INFO: Record<VerifyMode, { label: string; desc: string }> = {
  quick: { label: "Quick", desc: "Fast scan, fewer searches, lower recall" },
  standard: { label: "Standard", desc: "Balanced depth and speed (default)" },
  thorough: { label: "Thorough", desc: "Deep search, reranking, high recall" },
  custom: { label: "Custom", desc: "Configure all parameters manually" },
};

// ---------------------------------------------------------------------------
// Shared inline style constants
// ---------------------------------------------------------------------------

/** Consistent label style applied to form field labels across all Citation
 *  panel sections. Provides minimum width, secondary color, and small font
 *  for a compact two-column layout. */
export const LABEL_STYLE = {
  "min-width": "60px",
  color: "var(--color-text-secondary)",
  "font-size": "12px",
} as const;
