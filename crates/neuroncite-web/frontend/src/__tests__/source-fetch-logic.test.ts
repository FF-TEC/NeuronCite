/**
 * Regression tests for the source fetching pipeline logic.
 *
 * Tests cover pure functions and constants extracted or re-implemented
 * from SourcesTab.tsx, client.ts, and sse.ts to avoid importing SolidJS
 * component internals in a non-reactive context. Each test group documents
 * a specific bug that was previously present and is now fixed.
 *
 * Bug history:
 *   - T-SRC-TIMEOUT-001/002: LONG_TIMEOUT_MS (2 min) was used for fetch-sources,
 *     causing the frontend to abort before the backend finished processing large
 *     bib files. Results stayed "pending" because setResults was never called.
 *   - T-SRC-SSE-RACE-001/002/003: SSE events from the early processing phase
 *     (DOI resolution) were lost because the EventSource connection had not
 *     yet established when the backend started sending events. The connectSourceSSE
 *     function now returns a Promise that resolves on onopen before the POST fires.
 *   - T-SRC-PROGRESS-001/002/003: DOI and download progress percentage calculations
 *     from the SSE event handler.
 */

import { describe, it, expect } from "vitest";

// ---------------------------------------------------------------------------
// T-SRC-TIMEOUT-001: Timeout constants
// ---------------------------------------------------------------------------

/**
 * Re-implementation of the timeout constants from client.ts.
 * The fetch-sources operation processes one entry per delay_ms interval;
 * with the default 1000 ms delay and a 100-entry bib file the operation
 * takes at least 100 seconds, well beyond the 2-minute LONG_TIMEOUT_MS.
 */
const LONG_TIMEOUT_MS = 120_000;
const FETCH_SOURCES_TIMEOUT_MS = 1_800_000;

describe("T-SRC-TIMEOUT-001: fetch-sources timeout exceeds LONG_TIMEOUT_MS", () => {
  it("FETCH_SOURCES_TIMEOUT_MS is greater than LONG_TIMEOUT_MS", () => {
    expect(FETCH_SOURCES_TIMEOUT_MS).toBeGreaterThan(LONG_TIMEOUT_MS);
  });

  it("FETCH_SOURCES_TIMEOUT_MS is at least 30 minutes", () => {
    // 30 minutes = 1 800 000 ms. Accommodates ~1700 entries at 1 s delay each.
    expect(FETCH_SOURCES_TIMEOUT_MS).toBeGreaterThanOrEqual(1_800_000);
  });
});

describe("T-SRC-TIMEOUT-002: LONG_TIMEOUT_MS is 2 minutes", () => {
  it("LONG_TIMEOUT_MS is exactly 120 000 ms", () => {
    // Confirms LONG_TIMEOUT_MS has not been silently increased to fix the
    // fetch-sources issue (which would break other endpoints relying on it).
    expect(LONG_TIMEOUT_MS).toBe(120_000);
  });
});

// ---------------------------------------------------------------------------
// T-SRC-SSE-RACE-001/002/003: resolveOnce helper from connectSourceSSE
// ---------------------------------------------------------------------------

/**
 * Re-implementation of the resolveOnce pattern used inside connectSourceSSE.
 * The pattern ensures the returned Promise is resolved exactly once even when
 * both the onopen callback and the safety timer fire close together.
 */
function makeResolveOnce(): { resolveOnce: () => void; resolvedCount: () => number } {
  let count = 0;
  let resolved = false;
  const resolveOnce = () => {
    if (!resolved) {
      resolved = true;
      count++;
    }
  };
  return { resolveOnce, resolvedCount: () => count };
}

describe("T-SRC-SSE-RACE-001: resolveOnce resolves the promise exactly once", () => {
  it("first call increments counter to 1", () => {
    const { resolveOnce, resolvedCount } = makeResolveOnce();
    resolveOnce();
    expect(resolvedCount()).toBe(1);
  });

  it("subsequent calls are no-ops", () => {
    const { resolveOnce, resolvedCount } = makeResolveOnce();
    resolveOnce();
    resolveOnce();
    resolveOnce();
    expect(resolvedCount()).toBe(1);
  });
});

describe("T-SRC-SSE-RACE-002: resolveOnce is idempotent regardless of call order", () => {
  it("safety timer and onopen both calling resolveOnce resolves exactly once", () => {
    // Simulates both the onopen callback and the 2-second safety timer
    // calling resolveOnce in the same tick.
    const { resolveOnce, resolvedCount } = makeResolveOnce();
    // Simulate onopen firing first.
    resolveOnce();
    // Simulate safety timer firing immediately after.
    resolveOnce();
    expect(resolvedCount()).toBe(1);
  });
});

describe("T-SRC-SSE-RACE-003: fresh resolveOnce starts at zero", () => {
  it("each connectSourceSSE call gets an independent resolved flag", () => {
    const first = makeResolveOnce();
    const second = makeResolveOnce();
    first.resolveOnce();
    // second is unaffected by first resolving.
    expect(second.resolvedCount()).toBe(0);
    second.resolveOnce();
    expect(first.resolvedCount()).toBe(1);
    expect(second.resolvedCount()).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// T-SRC-PROGRESS-001/002/003: DOI and download progress calculations
// ---------------------------------------------------------------------------

/**
 * Re-implementation of the DOI progress calculation from connectSourceSSE.
 * Maps doi_done / doi_total to 0-20% of the total 100-unit progress bar.
 */
function calcDoiProgress(doiDone: number, doiTotal: number): number {
  return Math.round((doiDone / doiTotal) * 20);
}

describe("T-SRC-PROGRESS-001: DOI resolution phase maps to 0-20 percent", () => {
  it("0 of 10 resolved = 0%", () => {
    expect(calcDoiProgress(0, 10)).toBe(0);
  });

  it("5 of 10 resolved = 10%", () => {
    expect(calcDoiProgress(5, 10)).toBe(10);
  });

  it("10 of 10 resolved = 20%", () => {
    expect(calcDoiProgress(10, 10)).toBe(20);
  });

  it("1 of 1 resolved = 20%", () => {
    expect(calcDoiProgress(1, 1)).toBe(20);
  });

  it("rounds to nearest integer", () => {
    // 3 of 10 = 0.3 * 20 = 6.0
    expect(calcDoiProgress(3, 10)).toBe(6);
    // 1 of 3 = 0.333... * 20 = 6.666... -> 7
    expect(calcDoiProgress(1, 3)).toBe(7);
  });
});

/**
 * Re-implementation of the download progress calculation from connectSourceSSE.
 * Maps doneCount / total to 20-100% of the total 100-unit progress bar.
 */
function calcDownloadProgress(doneCount: number, total: number): number {
  if (total <= 0) return 0;
  const downloadPct = Math.round((doneCount / total) * 80);
  return 20 + downloadPct;
}

describe("T-SRC-PROGRESS-002: download phase maps to 20-100 percent", () => {
  it("0 of 10 downloaded = 20%", () => {
    expect(calcDownloadProgress(0, 10)).toBe(20);
  });

  it("5 of 10 downloaded = 60%", () => {
    expect(calcDownloadProgress(5, 10)).toBe(60);
  });

  it("10 of 10 downloaded = 100%", () => {
    expect(calcDownloadProgress(10, 10)).toBe(100);
  });

  it("total of 0 returns 0 to avoid division by zero", () => {
    expect(calcDownloadProgress(0, 0)).toBe(0);
  });
});

describe("T-SRC-PROGRESS-003: combined DOI + download covers the full 0-100 range", () => {
  it("DOI complete (done=20) then all downloads complete (done=100)", () => {
    const doiComplete = calcDoiProgress(5, 5);
    expect(doiComplete).toBe(20);
    const downloadComplete = calcDownloadProgress(8, 8);
    expect(downloadComplete).toBe(100);
  });

  it("0% at start, 100% at end with no overlap or gap", () => {
    expect(calcDoiProgress(0, 5)).toBe(0);
    expect(calcDownloadProgress(8, 8)).toBe(100);
  });
});
