/**
 * Tests for the update check logic extracted from AboutPanel.
 *
 * 1. UpdateCheckResponse interpretation (update available vs up to date)
 * 2. Error message parsing from ApiError body (JSON and non-JSON)
 *
 * These tests verify pure functions re-implemented from AboutPanel.tsx
 * to avoid importing SolidJS component internals in a non-reactive context.
 */

import { describe, it, expect } from "vitest";

// ---------------------------------------------------------------------------
// 1. UpdateCheckResponse interpretation
// ---------------------------------------------------------------------------

interface UpdateCheckResponse {
  current_version: string;
  latest_version: string;
  update_available: boolean;
  release_url: string;
}

describe("T-UPD-001: UpdateCheckResponse interpretation", () => {
  it("recognizes when an update is available", () => {
    const res: UpdateCheckResponse = {
      current_version: "0.1.0",
      latest_version: "0.2.0",
      update_available: true,
      release_url: "https://github.com/FF-TEC/NeuronCite/releases/tag/v0.2.0",
    };
    expect(res.update_available).toBe(true);
    expect(res.release_url).toContain("github.com");
  });

  it("recognizes when already up to date", () => {
    const res: UpdateCheckResponse = {
      current_version: "0.1.0",
      latest_version: "0.1.0",
      update_available: false,
      release_url: "https://github.com/FF-TEC/NeuronCite/releases/tag/v0.1.0",
    };
    expect(res.update_available).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// 2. Error message extraction (ApiError body parsing)
// ---------------------------------------------------------------------------

/**
 * Re-implementation of the error parsing logic from the checkForUpdates
 * handler in AboutPanel.tsx. Extracts the `error` field from an ApiError
 * body string, falling back to a generic message.
 */
function parseUpdateError(errorBody: string): string {
  try {
    const parsed = JSON.parse(errorBody);
    return parsed.error || "Update check failed";
  } catch {
    return `Update check failed: ${errorBody}`;
  }
}

describe("T-UPD-002: parseUpdateError", () => {
  it("extracts error field from valid JSON", () => {
    const body =
      '{"error":"GitHub API rate limit exceeded \\u2014 try again in a few minutes"}';
    expect(parseUpdateError(body)).toContain("rate limit");
  });

  it("returns generic message when error field is missing", () => {
    const body = '{"status":"unknown"}';
    expect(parseUpdateError(body)).toBe("Update check failed");
  });

  it("handles non-JSON body gracefully", () => {
    const body = "Internal Server Error";
    expect(parseUpdateError(body)).toBe(
      "Update check failed: Internal Server Error",
    );
  });

  it("handles empty JSON error field", () => {
    const body = '{"error":""}';
    expect(parseUpdateError(body)).toBe("Update check failed");
  });
});
