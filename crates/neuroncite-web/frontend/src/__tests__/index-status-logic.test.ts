/**
 * Tests for the indexStatus() logic in IndexingTab.tsx.
 *
 * The indexStatus function determines whether the "Start Indexing" button
 * is active or replaced by "All files up to date". It compares the active
 * session's directory path AND chunk configuration against the current
 * UI settings. A mismatch in any parameter (strategy, chunk_size,
 * chunk_overlap, max_words) results in "no_session", enabling the button
 * so the user can create a new index with different parameters.
 *
 * These tests re-implement the pure logic from IndexingTab.tsx to avoid
 * importing SolidJS component internals in a non-reactive context.
 */

import { describe, it, expect } from "vitest";

// ---------------------------------------------------------------------------
// Type definitions mirroring the component's data structures
// ---------------------------------------------------------------------------

/** Subset of SessionDto fields relevant to indexStatus comparison. */
interface SessionInfo {
  id: number;
  directory_path: string;
  chunk_strategy: string;
  chunk_size: number | null;
  chunk_overlap: number | null;
  max_words: number | null;
}

/** UI state fields that indexStatus reads. */
interface UiState {
  activeSessionId: number | null;
  folderPath: string;
  selectedStrategy: string;
  chunkSize: string;
  chunkOverlap: string;
}

/** File status from the scan endpoint. */
interface FileEntry {
  status: "indexed" | "outdated" | "pending";
}

/** Return type of indexStatus. */
type IndexStatusResult =
  | { kind: "no_files" }
  | { kind: "no_session" }
  | { kind: "up_to_date" }
  | { kind: "needs_update"; pending: number; outdated: number };

// ---------------------------------------------------------------------------
// Path normalization (simplified version of normalizeWindowsPath)
// ---------------------------------------------------------------------------

const WINDOWS_PREFIX_RE = /^\\\\[?.]\\(?!UNC\\)/;

function normalizeWindowsPath(path: string): string {
  if (path.startsWith("\\\\?\\UNC\\")) {
    return "\\\\" + path.slice(8);
  }
  return path.replace(WINDOWS_PREFIX_RE, "");
}

// ---------------------------------------------------------------------------
// Re-implementation of indexStatus from IndexingTab.tsx
// ---------------------------------------------------------------------------

/**
 * Pure function re-implementing the indexStatus() logic from IndexingTab.tsx.
 * Mirrors the exact comparison chain: path match, strategy match, then
 * strategy-specific parameter comparison.
 */
function indexStatus(
  sessions: SessionInfo[],
  files: FileEntry[],
  ui: UiState,
): IndexStatusResult {
  // File count gate
  if (files.length === 0) return { kind: "no_files" };

  const session = sessions.find((s) => s.id === ui.activeSessionId);
  if (!session || !ui.folderPath) return { kind: "no_session" };

  // Directory path comparison with Windows extended-length prefix normalization
  const pathMatch = normalizeWindowsPath(session.directory_path).toLowerCase()
    === normalizeWindowsPath(ui.folderPath).toLowerCase();
  if (!pathMatch) return { kind: "no_session" };

  // Chunk strategy comparison
  const uiStrategy = ui.selectedStrategy;
  if (session.chunk_strategy !== uiStrategy) return { kind: "no_session" };

  // Strategy-specific parameter comparison
  const uiSize = parseInt(ui.chunkSize, 10);
  if (uiStrategy === "token" || uiStrategy === "word") {
    if (!isNaN(uiSize) && session.chunk_size !== null && session.chunk_size !== uiSize) {
      return { kind: "no_session" };
    }
    const uiOverlap = parseInt(ui.chunkOverlap, 10);
    if (!isNaN(uiOverlap) && session.chunk_overlap !== null && session.chunk_overlap !== uiOverlap) {
      return { kind: "no_session" };
    }
  } else if (uiStrategy === "sentence") {
    if (!isNaN(uiSize) && session.max_words !== null && session.max_words !== uiSize) {
      return { kind: "no_session" };
    }
  }

  // File status aggregation
  let pending = 0;
  let outdated = 0;
  for (const f of files) {
    if (f.status === "outdated") outdated++;
    else if (f.status !== "indexed") pending++;
  }

  if (pending === 0 && outdated === 0) {
    return { kind: "up_to_date" };
  }
  return { kind: "needs_update", pending, outdated };
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/** Creates a session with default token/256/32 configuration. */
function makeSession(overrides?: Partial<SessionInfo>): SessionInfo {
  return {
    id: 1,
    directory_path: "D:\\Documents\\Papers",
    chunk_strategy: "token",
    chunk_size: 256,
    chunk_overlap: 32,
    max_words: null,
    ...overrides,
  };
}

/** Creates UI state matching the default session configuration. */
function makeUi(overrides?: Partial<UiState>): UiState {
  return {
    activeSessionId: 1,
    folderPath: "D:\\Documents\\Papers",
    selectedStrategy: "token",
    chunkSize: "256",
    chunkOverlap: "32",
    ...overrides,
  };
}

/** Creates an array of files with the given status. */
function makeFiles(count: number, status: "indexed" | "outdated" | "pending" = "indexed"): FileEntry[] {
  return Array.from({ length: count }, () => ({ status }));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("indexStatus", () => {
  // ---- Basic guard conditions ----

  describe("guard conditions", () => {
    it("returns no_files when the file list is empty", () => {
      const result = indexStatus([makeSession()], [], makeUi());
      expect(result.kind).toBe("no_files");
    });

    it("returns no_session when no session is active", () => {
      const result = indexStatus([makeSession()], makeFiles(3), makeUi({ activeSessionId: null }));
      expect(result.kind).toBe("no_session");
    });

    it("returns no_session when folderPath is empty", () => {
      const result = indexStatus([makeSession()], makeFiles(3), makeUi({ folderPath: "" }));
      expect(result.kind).toBe("no_session");
    });

    it("returns no_session when active session ID does not exist in session list", () => {
      const result = indexStatus([makeSession({ id: 99 })], makeFiles(3), makeUi({ activeSessionId: 1 }));
      expect(result.kind).toBe("no_session");
    });
  });

  // ---- Path comparison ----

  describe("path comparison", () => {
    it("returns up_to_date when paths match exactly", () => {
      const result = indexStatus([makeSession()], makeFiles(3), makeUi());
      expect(result.kind).toBe("up_to_date");
    });

    it("returns up_to_date when paths differ only in case", () => {
      const result = indexStatus(
        [makeSession({ directory_path: "D:\\documents\\papers" })],
        makeFiles(3),
        makeUi({ folderPath: "D:\\Documents\\Papers" }),
      );
      expect(result.kind).toBe("up_to_date");
    });

    it("normalizes Windows extended-length prefixes before comparison", () => {
      const result = indexStatus(
        [makeSession({ directory_path: "D:\\Documents\\Papers" })],
        makeFiles(3),
        makeUi({ folderPath: "\\\\?\\D:\\Documents\\Papers" }),
      );
      expect(result.kind).toBe("up_to_date");
    });

    it("returns no_session when paths differ", () => {
      const result = indexStatus(
        [makeSession()],
        makeFiles(3),
        makeUi({ folderPath: "D:\\Other\\Folder" }),
      );
      expect(result.kind).toBe("no_session");
    });
  });

  // ---- Strategy comparison ----

  describe("strategy comparison", () => {
    it("returns no_session when strategy differs (token vs word)", () => {
      const result = indexStatus(
        [makeSession({ chunk_strategy: "token" })],
        makeFiles(3),
        makeUi({ selectedStrategy: "word" }),
      );
      expect(result.kind).toBe("no_session");
    });

    it("returns no_session when strategy differs (token vs sentence)", () => {
      const result = indexStatus(
        [makeSession({ chunk_strategy: "token" })],
        makeFiles(3),
        makeUi({ selectedStrategy: "sentence" }),
      );
      expect(result.kind).toBe("no_session");
    });

    it("returns no_session when strategy differs (token vs page)", () => {
      const result = indexStatus(
        [makeSession({ chunk_strategy: "token" })],
        makeFiles(3),
        makeUi({ selectedStrategy: "page" }),
      );
      expect(result.kind).toBe("no_session");
    });
  });

  // ---- Chunk size comparison (token/word strategies) ----

  describe("chunk size comparison for token/word strategies", () => {
    it("returns no_session when chunk_size differs (token strategy)", () => {
      const result = indexStatus(
        [makeSession({ chunk_strategy: "token", chunk_size: 256 })],
        makeFiles(3),
        makeUi({ selectedStrategy: "token", chunkSize: "512" }),
      );
      expect(result.kind).toBe("no_session");
    });

    it("returns no_session when chunk_size differs (word strategy)", () => {
      const result = indexStatus(
        [makeSession({ chunk_strategy: "word", chunk_size: 300 })],
        makeFiles(3),
        makeUi({ selectedStrategy: "word", chunkSize: "200" }),
      );
      expect(result.kind).toBe("no_session");
    });

    it("returns up_to_date when chunk_size matches", () => {
      const result = indexStatus(
        [makeSession({ chunk_strategy: "token", chunk_size: 256 })],
        makeFiles(3),
        makeUi({ selectedStrategy: "token", chunkSize: "256" }),
      );
      expect(result.kind).toBe("up_to_date");
    });

    it("skips chunk_size comparison when UI value is not a valid number", () => {
      const result = indexStatus(
        [makeSession({ chunk_strategy: "token", chunk_size: 256 })],
        makeFiles(3),
        makeUi({ selectedStrategy: "token", chunkSize: "" }),
      );
      expect(result.kind).toBe("up_to_date");
    });

    it("skips chunk_size comparison when session value is null", () => {
      const result = indexStatus(
        [makeSession({ chunk_strategy: "token", chunk_size: null })],
        makeFiles(3),
        makeUi({ selectedStrategy: "token", chunkSize: "512" }),
      );
      expect(result.kind).toBe("up_to_date");
    });
  });

  // ---- Chunk overlap comparison (token/word strategies) ----

  describe("chunk overlap comparison for token/word strategies", () => {
    it("returns no_session when chunk_overlap differs", () => {
      const result = indexStatus(
        [makeSession({ chunk_strategy: "token", chunk_size: 256, chunk_overlap: 32 })],
        makeFiles(3),
        makeUi({ selectedStrategy: "token", chunkSize: "256", chunkOverlap: "64" }),
      );
      expect(result.kind).toBe("no_session");
    });

    it("returns up_to_date when chunk_overlap matches", () => {
      const result = indexStatus(
        [makeSession({ chunk_strategy: "word", chunk_size: 300, chunk_overlap: 50 })],
        makeFiles(3),
        makeUi({ selectedStrategy: "word", chunkSize: "300", chunkOverlap: "50" }),
      );
      expect(result.kind).toBe("up_to_date");
    });

    it("skips chunk_overlap comparison when UI value is not a valid number", () => {
      const result = indexStatus(
        [makeSession({ chunk_strategy: "token", chunk_size: 256, chunk_overlap: 32 })],
        makeFiles(3),
        makeUi({ selectedStrategy: "token", chunkSize: "256", chunkOverlap: "" }),
      );
      expect(result.kind).toBe("up_to_date");
    });

    it("skips chunk_overlap comparison when session value is null", () => {
      const result = indexStatus(
        [makeSession({ chunk_strategy: "token", chunk_size: 256, chunk_overlap: null })],
        makeFiles(3),
        makeUi({ selectedStrategy: "token", chunkSize: "256", chunkOverlap: "64" }),
      );
      expect(result.kind).toBe("up_to_date");
    });
  });

  // ---- Sentence strategy: max_words comparison ----

  describe("sentence strategy max_words comparison", () => {
    it("returns no_session when max_words differs", () => {
      const session = makeSession({
        chunk_strategy: "sentence",
        chunk_size: null,
        chunk_overlap: null,
        max_words: 200,
      });
      const ui = makeUi({ selectedStrategy: "sentence", chunkSize: "300" });
      const result = indexStatus([session], makeFiles(3), ui);
      expect(result.kind).toBe("no_session");
    });

    it("returns up_to_date when max_words matches", () => {
      const session = makeSession({
        chunk_strategy: "sentence",
        chunk_size: null,
        chunk_overlap: null,
        max_words: 200,
      });
      const ui = makeUi({ selectedStrategy: "sentence", chunkSize: "200" });
      const result = indexStatus([session], makeFiles(3), ui);
      expect(result.kind).toBe("up_to_date");
    });

    it("ignores chunk_overlap for sentence strategy", () => {
      const session = makeSession({
        chunk_strategy: "sentence",
        chunk_size: null,
        chunk_overlap: null,
        max_words: 200,
      });
      // Even though chunkOverlap differs from session, sentence strategy
      // does not use overlap, so the comparison should not apply.
      const ui = makeUi({ selectedStrategy: "sentence", chunkSize: "200", chunkOverlap: "999" });
      const result = indexStatus([session], makeFiles(3), ui);
      expect(result.kind).toBe("up_to_date");
    });
  });

  // ---- Page strategy: no parameter comparison ----

  describe("page strategy (no parameters)", () => {
    it("returns up_to_date when strategy matches as page", () => {
      const session = makeSession({
        chunk_strategy: "page",
        chunk_size: null,
        chunk_overlap: null,
        max_words: null,
      });
      const ui = makeUi({ selectedStrategy: "page", chunkSize: "999", chunkOverlap: "999" });
      const result = indexStatus([session], makeFiles(3), ui);
      expect(result.kind).toBe("up_to_date");
    });
  });

  // ---- File status aggregation ----

  describe("file status aggregation", () => {
    it("returns up_to_date when all files are indexed", () => {
      const result = indexStatus([makeSession()], makeFiles(5, "indexed"), makeUi());
      expect(result.kind).toBe("up_to_date");
    });

    it("returns needs_update with pending count", () => {
      const files = [
        ...makeFiles(3, "indexed"),
        ...makeFiles(2, "pending"),
      ];
      const result = indexStatus([makeSession()], files, makeUi());
      expect(result).toEqual({ kind: "needs_update", pending: 2, outdated: 0 });
    });

    it("returns needs_update with outdated count", () => {
      const files = [
        ...makeFiles(3, "indexed"),
        ...makeFiles(1, "outdated"),
      ];
      const result = indexStatus([makeSession()], files, makeUi());
      expect(result).toEqual({ kind: "needs_update", pending: 0, outdated: 1 });
    });

    it("returns needs_update with both pending and outdated counts", () => {
      const files = [
        ...makeFiles(2, "indexed"),
        ...makeFiles(1, "pending"),
        ...makeFiles(1, "outdated"),
      ];
      const result = indexStatus([makeSession()], files, makeUi());
      expect(result).toEqual({ kind: "needs_update", pending: 1, outdated: 1 });
    });
  });

  // ---- Combined scenario: the original bug ----

  describe("original bug scenario", () => {
    it("allows re-indexing when only chunk_size differs for same directory", () => {
      // Session was indexed with token/256/32
      const session = makeSession({ chunk_strategy: "token", chunk_size: 256, chunk_overlap: 32 });
      // User changes chunk_size to 512 in the UI
      const ui = makeUi({ selectedStrategy: "token", chunkSize: "512", chunkOverlap: "32" });
      const result = indexStatus([session], makeFiles(5), ui);
      // Must return no_session so the "Start Indexing" button is enabled
      expect(result.kind).toBe("no_session");
    });

    it("allows re-indexing when only chunk_overlap differs for same directory", () => {
      const session = makeSession({ chunk_strategy: "token", chunk_size: 256, chunk_overlap: 32 });
      const ui = makeUi({ selectedStrategy: "token", chunkSize: "256", chunkOverlap: "64" });
      const result = indexStatus([session], makeFiles(5), ui);
      expect(result.kind).toBe("no_session");
    });

    it("allows re-indexing when strategy changes for same directory", () => {
      const session = makeSession({ chunk_strategy: "token", chunk_size: 256, chunk_overlap: 32 });
      const ui = makeUi({ selectedStrategy: "word", chunkSize: "256", chunkOverlap: "32" });
      const result = indexStatus([session], makeFiles(5), ui);
      expect(result.kind).toBe("no_session");
    });

    it("blocks re-indexing when all settings match and all files are indexed", () => {
      const session = makeSession({ chunk_strategy: "token", chunk_size: 256, chunk_overlap: 32 });
      const ui = makeUi({ selectedStrategy: "token", chunkSize: "256", chunkOverlap: "32" });
      const result = indexStatus([session], makeFiles(5), ui);
      expect(result.kind).toBe("up_to_date");
    });
  });
});
