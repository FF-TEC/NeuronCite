/**
 * Tests for the CitationPanel logic extracted from the component:
 *   1. Dimension compatibility check for session filtering
 *   2. Source session identification (directory matching)
 *   3. .bib auto-detection from directory listings
 *   4. Verification parameter resolution from presets
 *
 * These tests verify pure functions re-implemented from CitationPanel.tsx
 * to avoid importing SolidJS component internals in a non-reactive context.
 */

import { describe, it, expect } from "vitest";

// ---------------------------------------------------------------------------
// 1. Dimension compatibility check
// ---------------------------------------------------------------------------

/**
 * Re-implementation of isDimensionCompatible from CitationPanel.tsx.
 * Returns true when the session's vector dimension matches the loaded
 * model's dimension, or when no model is loaded (dimension === 0).
 */
function isDimensionCompatible(
  sessionDimension: number,
  activeModelDimension: number,
): boolean {
  return activeModelDimension === 0 || sessionDimension === activeModelDimension;
}

describe("isDimensionCompatible", () => {
  it("returns true when no model is loaded (dimension 0)", () => {
    expect(isDimensionCompatible(384, 0)).toBe(true);
    expect(isDimensionCompatible(768, 0)).toBe(true);
    expect(isDimensionCompatible(1024, 0)).toBe(true);
  });

  it("returns true when dimensions match", () => {
    expect(isDimensionCompatible(384, 384)).toBe(true);
    expect(isDimensionCompatible(768, 768)).toBe(true);
    expect(isDimensionCompatible(1024, 1024)).toBe(true);
  });

  it("returns false when dimensions differ", () => {
    expect(isDimensionCompatible(384, 768)).toBe(false);
    expect(isDimensionCompatible(768, 384)).toBe(false);
    expect(isDimensionCompatible(1024, 384)).toBe(false);
  });

  it("returns true for zero-dimension session when no model is loaded", () => {
    expect(isDimensionCompatible(0, 0)).toBe(true);
  });

  it("returns false for zero-dimension session when a model is loaded", () => {
    expect(isDimensionCompatible(0, 384)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// 2. Source session identification
// ---------------------------------------------------------------------------

/**
 * Re-implementation of isSourceSession from CitationPanel.tsx.
 * Normalizes trailing slashes and compares directory paths
 * case-insensitively to identify sessions created from the
 * Fetch & Index Sources workflow.
 */
function isSourceSession(
  sessionDirectoryPath: string,
  sourcePath: string,
  fetchIndexEnabled: boolean,
): boolean {
  if (!fetchIndexEnabled || !sourcePath) return false;
  const normSource = sourcePath.replace(/[\\/]+$/, "").toLowerCase();
  const normDir = sessionDirectoryPath.replace(/[\\/]+$/, "").toLowerCase();
  return normDir === normSource;
}

describe("isSourceSession", () => {
  it("returns false when fetchIndexEnabled is false", () => {
    expect(isSourceSession("/data/papers", "/data/papers", false)).toBe(false);
  });

  it("returns false when sourcePath is empty", () => {
    expect(isSourceSession("/data/papers", "", true)).toBe(false);
  });

  it("returns true for matching paths", () => {
    expect(isSourceSession("/data/papers", "/data/papers", true)).toBe(true);
  });

  it("ignores trailing slashes", () => {
    expect(isSourceSession("/data/papers/", "/data/papers", true)).toBe(true);
    expect(isSourceSession("/data/papers", "/data/papers/", true)).toBe(true);
    expect(isSourceSession("/data/papers/", "/data/papers/", true)).toBe(true);
  });

  it("handles Windows backslash paths", () => {
    expect(isSourceSession("D:\\Data\\Papers\\", "D:\\Data\\Papers", true)).toBe(true);
    expect(isSourceSession("D:\\Data\\Papers", "D:\\Data\\Papers\\", true)).toBe(true);
  });

  it("is case-insensitive", () => {
    expect(isSourceSession("/Data/Papers", "/data/papers", true)).toBe(true);
    expect(isSourceSession("D:\\DATA\\PAPERS", "d:\\data\\papers", true)).toBe(true);
  });

  it("returns false for different paths", () => {
    expect(isSourceSession("/data/other", "/data/papers", true)).toBe(false);
    expect(isSourceSession("/data", "/data/papers", true)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// 3. .bib auto-detection filter logic
// ---------------------------------------------------------------------------

interface DirEntry {
  name: string;
  kind: "directory" | "file";
}

/**
 * Re-implementation of the .bib filter logic from tryAutoDetectBib
 * in CitationPanel.tsx. Returns the .bib filename if exactly one
 * .bib file exists in the directory listing, or null otherwise.
 */
function detectSingleBib(entries: DirEntry[]): string | null {
  const bibFiles = entries.filter(
    (e) => e.kind === "file" && e.name.toLowerCase().endsWith(".bib"),
  );
  if (bibFiles.length === 1) return bibFiles[0].name;
  return null;
}

describe("detectSingleBib", () => {
  it("returns the filename when exactly one .bib file exists", () => {
    const entries: DirEntry[] = [
      { name: "paper.tex", kind: "file" },
      { name: "references.bib", kind: "file" },
      { name: "figures", kind: "directory" },
    ];
    expect(detectSingleBib(entries)).toBe("references.bib");
  });

  it("returns null when no .bib files exist", () => {
    const entries: DirEntry[] = [
      { name: "paper.tex", kind: "file" },
      { name: "main.pdf", kind: "file" },
    ];
    expect(detectSingleBib(entries)).toBeNull();
  });

  it("returns null when multiple .bib files exist", () => {
    const entries: DirEntry[] = [
      { name: "refs.bib", kind: "file" },
      { name: "extra.bib", kind: "file" },
      { name: "paper.tex", kind: "file" },
    ];
    expect(detectSingleBib(entries)).toBeNull();
  });

  it("ignores directories named .bib", () => {
    const entries: DirEntry[] = [
      { name: "backup.bib", kind: "directory" },
      { name: "paper.tex", kind: "file" },
    ];
    expect(detectSingleBib(entries)).toBeNull();
  });

  it("is case-insensitive for .bib extension", () => {
    const entries: DirEntry[] = [
      { name: "References.BIB", kind: "file" },
      { name: "paper.tex", kind: "file" },
    ];
    expect(detectSingleBib(entries)).toBe("References.BIB");
  });

  it("handles an empty directory listing", () => {
    expect(detectSingleBib([])).toBeNull();
  });

  it("handles mixed case extensions correctly", () => {
    const entries: DirEntry[] = [
      { name: "refs.Bib", kind: "file" },
      { name: "notes.txt", kind: "file" },
    ];
    expect(detectSingleBib(entries)).toBe("refs.Bib");
  });
});

// ---------------------------------------------------------------------------
// 4. Verification parameter presets
// ---------------------------------------------------------------------------

interface VerifyParams {
  top_k: number;
  cross_corpus_queries: number;
  max_tokens: number;
  temperature: number;
  max_retry_attempts: number;
  min_score: number | undefined;
  rerank: boolean;
}

/**
 * Re-implementation of the preset lookup from CitationPanel.tsx.
 * Returns the parameter set for a given preset mode name.
 */
const PRESETS: Record<string, VerifyParams> = {
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

describe("verification mode presets", () => {
  it("quick preset has minimal search depth", () => {
    expect(PRESETS.quick.top_k).toBe(3);
    expect(PRESETS.quick.cross_corpus_queries).toBe(1);
    expect(PRESETS.quick.rerank).toBe(false);
  });

  it("standard preset provides balanced parameters", () => {
    expect(PRESETS.standard.top_k).toBe(5);
    expect(PRESETS.standard.cross_corpus_queries).toBe(2);
    expect(PRESETS.standard.max_tokens).toBe(4096);
    expect(PRESETS.standard.rerank).toBe(false);
  });

  it("thorough preset enables reranking and minimum score", () => {
    expect(PRESETS.thorough.top_k).toBe(10);
    expect(PRESETS.thorough.rerank).toBe(true);
    expect(PRESETS.thorough.min_score).toBe(0.5);
  });

  it("quick and standard presets have no minimum score threshold", () => {
    expect(PRESETS.quick.min_score).toBeUndefined();
    expect(PRESETS.standard.min_score).toBeUndefined();
  });

  it("thorough has lower temperature than quick and standard", () => {
    expect(PRESETS.thorough.temperature).toBeLessThan(PRESETS.standard.temperature);
  });

  it("max_tokens increases from quick to standard to thorough", () => {
    expect(PRESETS.quick.max_tokens).toBeLessThan(PRESETS.standard.max_tokens);
    expect(PRESETS.standard.max_tokens).toBeLessThan(PRESETS.thorough.max_tokens);
  });
});

// ---------------------------------------------------------------------------
// 5. Session compatibility filter for "All" button
// ---------------------------------------------------------------------------

/**
 * Re-implementation of the selectAll logic from CitationPanel.tsx.
 * Filters sessions by dimension compatibility before selecting.
 */
function selectAllCompatible(
  sessions: Array<{ id: number; vector_dimension: number }>,
  activeModelDimension: number,
): number[] {
  return sessions
    .filter((s) => isDimensionCompatible(s.vector_dimension, activeModelDimension))
    .map((s) => s.id);
}

describe("selectAllCompatible", () => {
  const sessions = [
    { id: 1, vector_dimension: 384 },
    { id: 2, vector_dimension: 768 },
    { id: 3, vector_dimension: 384 },
    { id: 4, vector_dimension: 1024 },
  ];

  it("selects all sessions when no model is loaded", () => {
    const result = selectAllCompatible(sessions, 0);
    expect(result).toEqual([1, 2, 3, 4]);
  });

  it("selects only sessions with matching dimension", () => {
    const result = selectAllCompatible(sessions, 384);
    expect(result).toEqual([1, 3]);
  });

  it("selects single session for unique dimension", () => {
    const result = selectAllCompatible(sessions, 1024);
    expect(result).toEqual([4]);
  });

  it("returns empty array when no sessions match", () => {
    const result = selectAllCompatible(sessions, 512);
    expect(result).toEqual([]);
  });

  it("handles empty session list", () => {
    const result = selectAllCompatible([], 384);
    expect(result).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// 6. Session conflict check (path normalization)
// ---------------------------------------------------------------------------

/**
 * Re-implementation of the path normalization from checkSessionConflict
 * in CitationPanel.tsx. Strips trailing slashes and lowercases for
 * comparison.
 */
function normalizePath(path: string): string {
  return path.replace(/[\\/]+$/, "").toLowerCase();
}

describe("normalizePath", () => {
  it("strips trailing forward slash", () => {
    expect(normalizePath("/data/papers/")).toBe("/data/papers");
  });

  it("strips trailing backslash", () => {
    expect(normalizePath("D:\\Data\\Papers\\")).toBe("d:\\data\\papers");
  });

  it("strips multiple trailing slashes", () => {
    expect(normalizePath("/data/papers///")).toBe("/data/papers");
  });

  it("lowercases the entire path", () => {
    expect(normalizePath("/Data/Papers")).toBe("/data/papers");
  });

  it("handles paths without trailing slash", () => {
    expect(normalizePath("/data/papers")).toBe("/data/papers");
  });

  it("handles empty string", () => {
    expect(normalizePath("")).toBe("");
  });
});

// ---------------------------------------------------------------------------
// 7. Citation row reconciliation logic
// ---------------------------------------------------------------------------

/**
 * Re-implementation of the DB-status-to-UI-phase mapping and result_json
 * extraction from stores/app.ts reconcileCitationRows. This function takes
 * a CitationRowDto (API response) and returns the UI-side phase, verdict,
 * confidence, reasoning, and flag. The same logic is used by both
 * recoverCitationRows and reconcileCitationRows in the store.
 */
interface CitationRowDto {
  id: number;
  cite_key: string;
  title: string;
  author: string;
  tex_line: number;
  tex_context: string;
  status: string;
  result_json: string | null;
  flag: string;
}

interface ReconcileResult {
  phase: "pending" | "searching" | "evaluating" | "reasoning" | "done" | "error";
  verdict: string | null;
  confidence: number | null;
  reasoning: string;
  flag: string;
}

function reconcileRow(dto: CitationRowDto): ReconcileResult {
  let phase: ReconcileResult["phase"] = "pending";
  let verdict: string | null = null;
  let confidence: number | null = null;
  let reasoning = "";
  let flag = dto.flag || "";

  if (dto.status === "done") phase = "done";
  else if (dto.status === "claimed") phase = "searching";
  else if (dto.status === "failed") phase = "error";

  if (dto.result_json) {
    try {
      const result = JSON.parse(dto.result_json);
      verdict = result.verdict || null;
      confidence = typeof result.confidence === "number" ? result.confidence : null;
      reasoning = result.reasoning || "";
      if (result.flag && !flag) flag = result.flag;
    } catch {
      // Malformed result_json: leave defaults.
    }
  }

  return { phase, verdict, confidence, reasoning, flag };
}

describe("reconcileRow (DB status to UI phase mapping)", () => {
  const baseDto: CitationRowDto = {
    id: 1,
    cite_key: "fama1970",
    title: "Efficient Capital Markets",
    author: "Fama",
    tex_line: 100,
    tex_context: "",
    status: "pending",
    result_json: null,
    flag: "",
  };

  it("maps status 'done' to phase 'done' with verdict from result_json", () => {
    const dto: CitationRowDto = {
      ...baseDto,
      status: "done",
      result_json: JSON.stringify({
        verdict: "supported",
        confidence: 0.95,
        reasoning: "The source explicitly states the claim.",
        flag: "",
      }),
    };
    const result = reconcileRow(dto);
    expect(result.phase).toBe("done");
    expect(result.verdict).toBe("supported");
    expect(result.confidence).toBe(0.95);
    expect(result.reasoning).toBe("The source explicitly states the claim.");
    expect(result.flag).toBe("");
  });

  it("maps status 'done' with 'partial' verdict correctly", () => {
    const dto: CitationRowDto = {
      ...baseDto,
      status: "done",
      result_json: JSON.stringify({
        verdict: "partial",
        confidence: 0.7,
        reasoning: "Only part A is confirmed.",
        flag: "warning",
      }),
    };
    const result = reconcileRow(dto);
    expect(result.phase).toBe("done");
    expect(result.verdict).toBe("partial");
    expect(result.confidence).toBe(0.7);
    expect(result.flag).toBe("warning");
  });

  it("maps status 'claimed' to phase 'searching'", () => {
    const dto: CitationRowDto = { ...baseDto, status: "claimed" };
    const result = reconcileRow(dto);
    expect(result.phase).toBe("searching");
    expect(result.verdict).toBeNull();
    expect(result.confidence).toBeNull();
  });

  it("maps status 'failed' to phase 'error'", () => {
    const dto: CitationRowDto = {
      ...baseDto,
      status: "failed",
      result_json: JSON.stringify({ error: "LLM timeout" }),
    };
    const result = reconcileRow(dto);
    expect(result.phase).toBe("error");
    // Failed rows have error JSON, not verdict JSON.
    expect(result.verdict).toBeNull();
  });

  it("maps status 'pending' to phase 'pending'", () => {
    const dto: CitationRowDto = { ...baseDto, status: "pending" };
    const result = reconcileRow(dto);
    expect(result.phase).toBe("pending");
    expect(result.verdict).toBeNull();
  });

  it("handles null result_json gracefully", () => {
    const dto: CitationRowDto = { ...baseDto, status: "done", result_json: null };
    const result = reconcileRow(dto);
    expect(result.phase).toBe("done");
    expect(result.verdict).toBeNull();
    expect(result.confidence).toBeNull();
    expect(result.reasoning).toBe("");
  });

  it("handles malformed result_json without crashing", () => {
    const dto: CitationRowDto = {
      ...baseDto,
      status: "done",
      result_json: "not valid json {{{",
    };
    const result = reconcileRow(dto);
    expect(result.phase).toBe("done");
    expect(result.verdict).toBeNull();
    expect(result.confidence).toBeNull();
  });

  it("preserves flag from dto when result_json has no flag", () => {
    const dto: CitationRowDto = {
      ...baseDto,
      status: "done",
      flag: "critical",
      result_json: JSON.stringify({
        verdict: "unsupported",
        confidence: 0.85,
        reasoning: "Not found.",
      }),
    };
    const result = reconcileRow(dto);
    expect(result.flag).toBe("critical");
  });

  it("uses result_json flag when dto flag is empty", () => {
    const dto: CitationRowDto = {
      ...baseDto,
      status: "done",
      flag: "",
      result_json: JSON.stringify({
        verdict: "unsupported",
        confidence: 0.85,
        reasoning: "Not found.",
        flag: "warning",
      }),
    };
    const result = reconcileRow(dto);
    expect(result.flag).toBe("warning");
  });

  it("corrects a row stuck in 'evaluating' phase to 'done' with verdict", () => {
    // This is the core bug scenario: frontend shows phase='evaluating'
    // because the SSE 'done' event was dropped. The reconciliation reads
    // the DB status='done' and result_json to fix it.
    const dto: CitationRowDto = {
      ...baseDto,
      status: "done",
      result_json: JSON.stringify({
        verdict: "partial",
        confidence: 0.65,
        reasoning: "Part of the claim is supported.",
        flag: "",
      }),
    };
    const result = reconcileRow(dto);
    // The reconciled phase must be 'done' (not 'evaluating').
    expect(result.phase).toBe("done");
    expect(result.verdict).toBe("partial");
    expect(result.confidence).toBe(0.65);
  });

  it("handles result_json with missing confidence field", () => {
    const dto: CitationRowDto = {
      ...baseDto,
      status: "done",
      result_json: JSON.stringify({
        verdict: "unsupported",
        reasoning: "No matching passages.",
      }),
    };
    const result = reconcileRow(dto);
    expect(result.verdict).toBe("unsupported");
    expect(result.confidence).toBeNull();
  });

  it("handles result_json with non-numeric confidence", () => {
    const dto: CitationRowDto = {
      ...baseDto,
      status: "done",
      result_json: JSON.stringify({
        verdict: "supported",
        confidence: "high",
        reasoning: "Strong match.",
      }),
    };
    const result = reconcileRow(dto);
    expect(result.verdict).toBe("supported");
    expect(result.confidence).toBeNull();
  });
});
