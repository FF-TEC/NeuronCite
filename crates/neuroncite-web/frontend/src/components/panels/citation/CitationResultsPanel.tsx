/**
 * Right-side results panel for the Citation verification tab. Displays
 * a progress bar, verdict summary badges, and a scrollable results table
 * with expandable rows showing per-citation detail (title, author, tex
 * context, search queries, and LLM reasoning).
 *
 * This component reads citation state directly from the app store
 * (citationRows, citationRowsDone, citationRowsTotal, citationVerdicts,
 * citationComplete). It does not manage any API calls or SSE subscriptions;
 * those responsibilities remain in CitationPanel and App.tsx respectively.
 *
 * Props are used only for the expandedRow signal accessor and toggle
 * callback, which are owned by CitationPanel so the parent can control
 * row expansion state (e.g., for keyboard navigation or programmatic
 * expand/collapse).
 */

import { Component, For, Show, createSignal } from "solid-js";
import { state } from "../../../stores/app";
import { VERDICT_COLORS, PHASE_COLORS } from "./constants";

/** Props for CitationResultsPanel. The parent provides the expandedRow
 *  signal and toggle function to maintain a single source of truth for
 *  which row is expanded. */
interface CitationResultsPanelProps {
  expandedRow: () => number | null;
  toggleRow: (rowId: number) => void;
}

/** Computes the overall verification progress percentage (0-100).
 *  Returns 0 when no rows exist. */
const progressPct = (): number => {
  if (state.citationRowsTotal === 0) return 0;
  return Math.round((state.citationRowsDone / state.citationRowsTotal) * 100);
};

const CitationResultsPanel: Component<CitationResultsPanelProps> = (props) => {
  return (
    <div style={{ display: "flex", "flex-direction": "column", gap: "0", height: "100%", "overflow-y": "auto" }}>

      {/* Progress bar and aggregate statistics header. Visible once the
       *  citation job has been created and the backend reports row totals. */}
      <Show when={state.citationRowsTotal > 0}>
        <div style={{ padding: "12px 16px", "border-bottom": "1px solid var(--color-glass-border)" }}>
          <div style={{ display: "flex", "align-items": "center", gap: "8px", "margin-bottom": "6px" }}>
            <span style={{ color: "var(--color-text-secondary)", "font-size": "13px" }}>
              {state.citationRowsDone} / {state.citationRowsTotal} ({progressPct()}%)
            </span>
            <Show when={state.citationComplete}>
              <span style={{ color: "var(--color-status-success)", "font-size": "12px" }}>Complete</span>
            </Show>
          </div>

          {/* Animated progress bar. Hidden after verification completes
           *  since the "Complete" label is sufficient. */}
          <Show when={!state.citationComplete}>
            <div style={{
              height: "4px",
              background: "var(--color-bg-tertiary)",
              "border-radius": "2px",
              overflow: "hidden",
              "margin-bottom": "8px",
            }}>
              <div style={{
                height: "100%",
                width: `${progressPct()}%`,
                background: "linear-gradient(90deg, var(--color-accent-purple), var(--color-accent-cyan))",
                transition: "width 0.3s ease",
              }} />
            </div>
          </Show>

          {/* Verdict summary badges. Each verdict type is shown as a colored
           *  pill with the count. The colors match VERDICT_COLORS. */}
          <Show when={Object.keys(state.citationVerdicts).length > 0}>
            <div style={{ display: "flex", gap: "6px", "flex-wrap": "wrap" }}>
              <For each={Object.entries(state.citationVerdicts)}>
                {([verdict, count]) => (
                  <span
                    class="badge"
                    style={{
                      background: VERDICT_COLORS[verdict] || "var(--color-text-muted)",
                      color: "#fff",
                      "font-size": "11px",
                      padding: "2px 8px",
                      "border-radius": "100px",
                    }}
                  >
                    {verdict}: {count as number}
                  </span>
                )}
              </For>
            </div>
          </Show>
        </div>
      </Show>

      {/* Empty state when no citation job has been created yet. */}
      <Show when={state.citationRows.length === 0}>
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
          Create a citation job to see verification results here.
        </div>
      </Show>

      {/* Live results table with sticky header and expandable detail rows. */}
      <Show when={state.citationRows.length > 0}>
        <div style={{ flex: "1", "overflow-y": "auto" }}>
          <table style={{
            width: "100%",
            "border-collapse": "collapse",
            "font-size": "12px",
          }}>
            <thead>
              <tr style={{
                background: "var(--color-bg-tertiary)",
                color: "var(--color-text-secondary)",
                "text-align": "left",
                position: "sticky",
                top: "0",
                "z-index": "1",
              }}>
                <th style={{ padding: "8px 12px", width: "40px" }}>#</th>
                <th style={{ padding: "8px 8px", width: "120px" }}>Cite Key</th>
                <th style={{ padding: "8px 8px", width: "50px" }}>Line</th>
                <th style={{ padding: "8px 8px" }}>Status</th>
                <th style={{ padding: "8px 8px", width: "100px" }}>Verdict</th>
                <th style={{ padding: "8px 8px", width: "60px" }}>Conf.</th>
              </tr>
            </thead>
            <tbody>
              <For each={state.citationRows}>
                {(row, index) => (
                  <>
                    <tr
                      tabIndex={0}
                      role="button"
                      aria-expanded={props.expandedRow() === row.id}
                      style={{
                        "border-bottom": "1px solid var(--color-glass-border)",
                        cursor: "pointer",
                        background: props.expandedRow() === row.id
                          ? "rgba(168, 85, 247, 0.08)"
                          : "transparent",
                        transition: "background 0.2s ease",
                      }}
                      onClick={() => props.toggleRow(row.id)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          props.toggleRow(row.id);
                        }
                      }}
                    >
                      <td style={{ padding: "6px 12px", color: "var(--color-text-muted)" }}>
                        {index() + 1}
                      </td>
                      <td style={{
                        padding: "6px 8px",
                        color: "var(--color-accent-cyan)",
                        "font-family": "monospace",
                        "font-size": "11px",
                        "max-width": "120px",
                        overflow: "hidden",
                        "text-overflow": "ellipsis",
                        "white-space": "nowrap",
                      }}>
                        {row.cite_key}
                      </td>
                      <td style={{ padding: "6px 8px", color: "var(--color-text-muted)" }}>
                        {row.tex_line}
                      </td>
                      <td style={{ padding: "6px 8px" }}>
                        {/* Phase badge with pulse animation for active phases */}
                        <span style={{
                          display: "inline-block",
                          padding: "1px 8px",
                          "border-radius": "100px",
                          "font-size": "11px",
                          color: "#fff",
                          background: PHASE_COLORS[row.phase] || "var(--color-text-muted)",
                          animation: (row.phase === "searching" || row.phase === "evaluating" || row.phase === "reasoning")
                            ? "pulse 1.5s ease-in-out infinite"
                            : "none",
                        }}>
                          {row.phase}
                        </span>
                      </td>
                      <td style={{ padding: "6px 8px" }}>
                        <Show when={row.verdict}>
                          <span style={{
                            display: "inline-block",
                            padding: "1px 8px",
                            "border-radius": "100px",
                            "font-size": "11px",
                            color: "#fff",
                            background: VERDICT_COLORS[row.verdict!] || "var(--color-text-muted)",
                          }}>
                            {row.verdict}
                          </span>
                        </Show>
                      </td>
                      <td style={{ padding: "6px 8px" }}>
                        {/* Confidence bar: horizontal fill with percentage label.
                         *  For positive verdicts (supported, partial, peripheral_match)
                         *  the bar uses a graduated color scheme: green >= 80%,
                         *  amber >= 50%, red < 50%. For negative verdicts (unsupported,
                         *  not_found, wrong_source, unverifiable), the bar always uses
                         *  the verdict's own color because a high meta-confidence on a
                         *  negative verdict is not a positive signal -- it means the
                         *  agent is very certain the citation has a problem. */}
                        <Show when={row.confidence !== null}>
                          <div style={{
                            display: "flex",
                            "align-items": "center",
                            gap: "4px",
                          }}>
                            <div style={{
                              flex: "1",
                              height: "4px",
                              background: "var(--color-bg-secondary)",
                              "border-radius": "2px",
                              overflow: "hidden",
                            }}>
                              <div style={{
                                height: "100%",
                                width: `${Math.round((row.confidence || 0) * 100)}%`,
                                background: (
                                  row.verdict === "unsupported" ||
                                  row.verdict === "not_found" ||
                                  row.verdict === "wrong_source" ||
                                  row.verdict === "unverifiable"
                                )
                                  ? (VERDICT_COLORS[row.verdict] || "var(--color-status-error)")
                                  : (row.confidence || 0) >= 0.8
                                    ? "var(--color-status-success)"
                                    : (row.confidence || 0) >= 0.5
                                      ? "var(--color-status-warning)"
                                      : "var(--color-status-error)",
                                transition: "width 0.3s ease",
                              }} />
                            </div>
                            <span style={{
                              "font-size": "10px",
                              color: "var(--color-text-muted)",
                              "min-width": "28px",
                              "text-align": "right",
                            }}>
                              {Math.round((row.confidence || 0) * 100)}%
                            </span>
                          </div>
                        </Show>
                      </td>
                    </tr>

                    {/* Expanded row detail: title, author, tex context, search
                     *  queries, and LLM reasoning text. Shown below the summary
                     *  row when the user clicks to expand. */}
                    <Show when={props.expandedRow() === row.id}>
                      <tr>
                        <td
                          colSpan={6}
                          style={{
                            padding: "8px 12px 12px 12px",
                            background: "rgba(168, 85, 247, 0.04)",
                            "border-bottom": "1px solid var(--color-glass-border)",
                          }}
                        >
                          <div style={{
                            "font-size": "11px",
                            color: "var(--color-text-secondary)",
                            "margin-bottom": "4px",
                          }}>
                            <strong style={{ color: "var(--color-text-primary)" }}>{row.title}</strong>
                            {" "}<span>({row.author})</span>
                          </div>

                          <Show when={row.tex_context}>
                            <div style={{
                              "font-size": "11px",
                              color: "var(--color-text-muted)",
                              "font-style": "italic",
                              "margin-bottom": "4px",
                              "max-height": "40px",
                              overflow: "hidden",
                              "text-overflow": "ellipsis",
                            }}>
                              {row.tex_context}
                            </div>
                          </Show>

                          <Show when={row.search_queries.length > 0}>
                            <div style={{ "font-size": "11px", "margin-bottom": "4px" }}>
                              <span style={{ color: "var(--color-text-muted)" }}>Queries: </span>
                              <For each={row.search_queries}>
                                {(q) => (
                                  <span style={{
                                    display: "inline-block",
                                    background: "var(--color-bg-tertiary)",
                                    padding: "1px 6px",
                                    "border-radius": "4px",
                                    "margin-right": "4px",
                                    "font-size": "10px",
                                    color: "var(--color-text-secondary)",
                                  }}>
                                    {q}
                                  </span>
                                )}
                              </For>
                            </div>
                          </Show>

                          <Show when={row.reasoning}>
                            <div style={{
                              "font-size": "11px",
                              color: "var(--color-text-primary)",
                              "white-space": "pre-wrap",
                              "max-height": "200px",
                              overflow: "auto",
                              padding: "6px 8px",
                              background: "var(--color-bg-secondary)",
                              "border-radius": "6px",
                              "font-family": "monospace",
                              "line-height": "1.5",
                            }}>
                              {row.reasoning}
                            </div>
                          </Show>
                        </td>
                      </tr>
                    </Show>
                  </>
                )}
              </For>
            </tbody>
          </table>
        </div>
      </Show>
    </div>
  );
};

export default CitationResultsPanel;
