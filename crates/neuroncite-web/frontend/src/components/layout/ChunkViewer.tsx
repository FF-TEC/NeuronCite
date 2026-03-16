import { Component, For, Show, createSignal, onMount, onCleanup } from "solid-js";
import { api } from "../../api/client";
import type { ChunkDto, SearchResultDto } from "../../api/types";

/**
 * ChunkViewer displays the matched search result in its document context.
 * The matched chunk is centered in a scrollable list, with neighboring
 * chunks loaded on demand via IntersectionObserver sentinels at both ends.
 *
 * Props:
 *  - result: The search result that was clicked (carries file_id, chunk_index,
 *    session_id, and all score metadata).
 *  - sessionId: The session to fetch chunks from. For single-session search
 *    this comes from the selected session; for multi-search from result.session_id.
 *  - onClose: Callback to return to the results list.
 *  - sessionLabel: Display label for the session (shown in the header).
 */
const ChunkViewer: Component<{
  result: SearchResultDto;
  sessionId: number;
  onClose: () => void;
  sessionLabel: string;
}> = (props) => {
  /** Loaded chunks keyed by chunk_index for deduplication and sorted rendering. */
  const [chunks, setChunks] = createSignal<Map<number, ChunkDto>>(new Map());

  /** Total number of chunks in this file, reported by the first API response. */
  const [totalChunks, setTotalChunks] = createSignal(0);

  /** Whether an upward fetch is in progress (prevents duplicate requests). */
  const [loadingUp, setLoadingUp] = createSignal(false);

  /** Whether a downward fetch is in progress (prevents duplicate requests). */
  const [loadingDown, setLoadingDown] = createSignal(false);

  /** Whether the initial load around the matched chunk has completed. */
  const [initialized, setInitialized] = createSignal(false);

  /** Error message from a failed chunk fetch. */
  const [fetchError, setFetchError] = createSignal<string | null>(null);

  /** Reference to the scrollable container for programmatic scroll positioning. */
  let scrollContainerRef: HTMLDivElement | undefined;

  /** Reference to the matched chunk element for scrollIntoView after initial load. */
  let matchedChunkRef: HTMLDivElement | undefined;

  /** Sentinel element at the top of the list for upward infinite scroll. */
  let topSentinelRef: HTMLDivElement | undefined;

  /** Sentinel element at the bottom of the list for downward infinite scroll. */
  let bottomSentinelRef: HTMLDivElement | undefined;

  /** Number of chunks to fetch per request in each direction. */
  const FETCH_SIZE = 5;

  /** Returns the sorted array of loaded chunks for rendering. */
  const sortedChunks = () => {
    const arr = Array.from(chunks().values());
    arr.sort((a, b) => a.chunk_index - b.chunk_index);
    return arr;
  };

  /** Lowest chunk_index currently loaded. Returns -1 when no chunks are loaded. */
  const minLoadedIndex = () => {
    const sorted = sortedChunks();
    return sorted.length > 0 ? sorted[0].chunk_index : -1;
  };

  /** Highest chunk_index currently loaded. Returns -1 when no chunks are loaded. */
  const maxLoadedIndex = () => {
    const sorted = sortedChunks();
    return sorted.length > 0 ? sorted[sorted.length - 1].chunk_index : -1;
  };

  /** Whether there are more chunks above the currently loaded range. */
  const canLoadUp = () => minLoadedIndex() > 0;

  /** Whether there are more chunks below the currently loaded range. */
  const canLoadDown = () => {
    const total = totalChunks();
    return total > 0 && maxLoadedIndex() < total - 1;
  };

  /** Merges fetched chunks into the existing map without duplicates. */
  const mergeChunks = (fetched: ChunkDto[]) => {
    setChunks((prev) => {
      const next = new Map(prev);
      for (const c of fetched) {
        next.set(c.chunk_index, c);
      }
      return next;
    });
  };

  /** Fetches chunks at the given offset and merges them into state.
   *  Updates totalChunks from the API response. */
  const fetchChunks = async (offset: number, limit: number): Promise<ChunkDto[]> => {
    const res = await api.chunks(props.sessionId, props.result.file_id, { offset, limit });
    setTotalChunks(res.total_chunks);
    return res.chunks;
  };

  /** Initial load: fetches a window centered around the matched chunk_index. */
  const initialLoad = async () => {
    const centerIndex = props.result.chunk_index;
    const startOffset = Math.max(0, centerIndex - Math.floor(FETCH_SIZE / 2));
    const limit = FETCH_SIZE + Math.floor(FETCH_SIZE / 2);

    try {
      const fetched = await fetchChunks(startOffset, limit);
      mergeChunks(fetched);
      setInitialized(true);

      // After the DOM updates, scroll the matched chunk into the center of the viewport.
      requestAnimationFrame(() => {
        matchedChunkRef?.scrollIntoView({ block: "center", behavior: "instant" });
      });
    } catch (e) {
      setFetchError(e instanceof Error ? e.message : String(e));
    }
  };

  /** Loads earlier chunks when the user scrolls to the top sentinel. */
  const loadUpward = async () => {
    if (loadingUp() || !canLoadUp()) return;
    setLoadingUp(true);

    const currentMin = minLoadedIndex();
    const newOffset = Math.max(0, currentMin - FETCH_SIZE);
    const limit = currentMin - newOffset;
    if (limit <= 0) { setLoadingUp(false); return; }

    // Record scroll height before prepending to preserve scroll position.
    const scrollEl = scrollContainerRef;
    const prevScrollHeight = scrollEl ? scrollEl.scrollHeight : 0;
    const prevScrollTop = scrollEl ? scrollEl.scrollTop : 0;

    try {
      const fetched = await fetchChunks(newOffset, limit);
      mergeChunks(fetched);

      // Restore scroll position after the DOM reflows with prepended content.
      requestAnimationFrame(() => {
        if (scrollEl) {
          const newScrollHeight = scrollEl.scrollHeight;
          const addedHeight = newScrollHeight - prevScrollHeight;
          scrollEl.scrollTop = prevScrollTop + addedHeight;
        }
      });
    } catch (e) {
      setFetchError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingUp(false);
    }
  };

  /** Loads later chunks when the user scrolls to the bottom sentinel. */
  const loadDownward = async () => {
    if (loadingDown() || !canLoadDown()) return;
    setLoadingDown(true);

    const currentMax = maxLoadedIndex();
    const newOffset = currentMax + 1;

    try {
      const fetched = await fetchChunks(newOffset, FETCH_SIZE);
      mergeChunks(fetched);
    } catch (e) {
      setFetchError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingDown(false);
    }
  };

  // Set up IntersectionObserver for both sentinels after mount.
  onMount(() => {
    initialLoad();

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          if (entry.target === topSentinelRef) loadUpward();
          if (entry.target === bottomSentinelRef) loadDownward();
        }
      },
      { root: scrollContainerRef, rootMargin: "100px", threshold: 0 },
    );

    // Observe sentinels after a short delay to avoid triggering on initial render.
    const timerId = setTimeout(() => {
      if (topSentinelRef) observer.observe(topSentinelRef);
      if (bottomSentinelRef) observer.observe(bottomSentinelRef);
    }, 300);

    onCleanup(() => {
      clearTimeout(timerId);
      observer.disconnect();
    });
  });

  /** File name extracted from the full source path. */
  const fileName = () => props.result.source_file.split(/[/\\]/).pop() || props.result.source_file;

  /** Score color matching the ResultCard logic. */
  const scoreColor = () => {
    const s = props.result.score;
    if (s >= 0.8) return "var(--color-accent-cyan)";
    if (s >= 0.5) return "var(--color-status-outdated)";
    return "var(--color-text-muted)";
  };

  return (
    <div style={{ display: "flex", "flex-direction": "column", height: "100%" }}>
      {/* Header bar with back button, file info, and chunk position */}
      <div
        style={{
          display: "flex",
          "align-items": "center",
          gap: "12px",
          padding: "10px 16px",
          "border-bottom": "1px solid var(--color-glass-border)",
          "flex-shrink": "0",
          background: "rgba(15, 22, 42, 0.4)",
        }}
      >
        <button
          class="btn btn-sm btn-primary"
          onClick={props.onClose}
          style={{ "flex-shrink": "0", display: "flex", "align-items": "center", gap: "4px" }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M19 12H5" />
            <path d="M12 19l-7-7 7-7" />
          </svg>
          Results
        </button>

        <div style={{ flex: "1", "min-width": "0" }}>
          <div style={{
            "font-size": "13px",
            "font-weight": "600",
            color: "var(--color-text-primary)",
            overflow: "hidden",
            "text-overflow": "ellipsis",
            "white-space": "nowrap",
          }}>
            {fileName()}
          </div>
          <div style={{ "font-size": "11px", color: "var(--color-text-muted)", "margin-top": "1px" }}>
            Chunk {props.result.chunk_index + 1}
            <Show when={totalChunks() > 0}>
              {" "}/ {totalChunks()}
            </Show>
            {" "} &middot; pp.{props.result.page_start}-{props.result.page_end}
            {" "} &middot; {props.sessionLabel}
          </div>
        </div>

        {/* Score badge */}
        <span
          style={{
            "font-size": "14px",
            "font-weight": "700",
            "font-family": "monospace",
            color: scoreColor(),
            "flex-shrink": "0",
          }}
        >
          [{props.result.score.toFixed(3)}]
        </span>
      </div>

      {/* Score detail strip below the header */}
      <div
        style={{
          display: "flex",
          "flex-wrap": "wrap",
          gap: "12px",
          padding: "6px 16px",
          "font-size": "11px",
          "border-bottom": "1px solid var(--color-glass-border)",
          "flex-shrink": "0",
          background: "rgba(15, 22, 42, 0.25)",
        }}
      >
        <span style={{ color: "var(--color-accent-cyan)" }}>
          vector: {props.result.vector_score.toFixed(3)}
        </span>
        <Show when={props.result.reranker_score !== null}>
          <span style={{ color: "var(--color-accent-purple)" }}>
            reranker: {props.result.reranker_score!.toFixed(3)}
          </span>
        </Show>
        <Show when={props.result.bm25_rank !== null}>
          <span style={{ color: "var(--color-text-muted)" }}>
            bm25: #{props.result.bm25_rank}
          </span>
        </Show>
        <span style={{ color: "var(--color-text-muted)" }}>
          file_id: {props.result.file_id}
        </span>
      </div>

      {/* Scrollable chunk list with infinite scroll sentinels */}
      <div
        ref={scrollContainerRef}
        style={{
          flex: "1",
          "overflow-y": "auto",
          "min-height": "0",
        }}
      >
        {/* Loading indicator while initial fetch is pending */}
        <Show when={!initialized() && !fetchError()}>
          <div style={{
            padding: "48px 20px",
            "text-align": "center",
            color: "var(--color-text-muted)",
            "font-size": "13px",
          }}>
            Loading document context...
          </div>
        </Show>

        {/* Error display */}
        <Show when={fetchError()}>
          <div style={{
            padding: "24px 16px",
            color: "var(--color-accent-magenta)",
            "font-size": "12px",
          }}>
            Failed to load chunks: {fetchError()}
          </div>
        </Show>

        <Show when={initialized()}>
          {/* Top sentinel: always rendered so IntersectionObserver stays attached.
              The loadUpward function returns early when canLoadUp() is false. */}
          <div ref={topSentinelRef} style={{ height: "1px" }} />

          <Show when={loadingUp()}>
            <div style={{
              padding: "8px 16px",
              "text-align": "center",
              "font-size": "11px",
              color: "var(--color-text-muted)",
            }}>
              Loading earlier chunks...
            </div>
          </Show>

          {/* Boundary indicator when at the beginning of the file */}
          <Show when={!canLoadUp() && minLoadedIndex() === 0}>
            <div style={{
              padding: "8px 16px",
              "text-align": "center",
              "font-size": "10px",
              color: "var(--color-text-muted)",
              "letter-spacing": "2px",
              "text-transform": "uppercase",
              opacity: "0.5",
            }}>
              start of document
            </div>
          </Show>

          {/* Chunk cards */}
          <For each={sortedChunks()}>
            {(chunk) => {
              const isMatch = () => chunk.chunk_index === props.result.chunk_index;
              return (
                <div
                  ref={(el) => { if (isMatch()) matchedChunkRef = el; }}
                  style={{
                    padding: "12px 16px",
                    "border-bottom": "1px solid var(--color-glass-border)",
                    background: isMatch()
                      ? "rgba(168, 85, 247, 0.08)"
                      : "transparent",
                    "border-left": isMatch()
                      ? "3px solid var(--color-accent-purple)"
                      : "3px solid transparent",
                    transition: "background var(--transition-fast)",
                  }}
                >
                  {/* Chunk header with index, page range, and word count */}
                  <div style={{
                    display: "flex",
                    "align-items": "center",
                    gap: "8px",
                    "margin-bottom": "6px",
                  }}>
                    <span style={{
                      "font-size": "11px",
                      "font-weight": "600",
                      color: isMatch()
                        ? "var(--color-accent-purple)"
                        : "var(--color-text-muted)",
                      "font-family": "monospace",
                    }}>
                      #{chunk.chunk_index}
                    </span>
                    <span style={{
                      "font-size": "10px",
                      color: "var(--color-text-muted)",
                    }}>
                      pp.{chunk.page_start}-{chunk.page_end}
                    </span>
                    <span style={{
                      "font-size": "10px",
                      color: "var(--color-text-muted)",
                    }}>
                      {chunk.word_count} words
                    </span>
                    <Show when={isMatch()}>
                      <span style={{
                        "font-size": "9px",
                        padding: "1px 6px",
                        "border-radius": "3px",
                        background: "rgba(168, 85, 247, 0.2)",
                        color: "var(--color-accent-purple-light)",
                        "font-weight": "600",
                        "letter-spacing": "0.5px",
                        "text-transform": "uppercase",
                      }}>
                        match
                      </span>
                    </Show>
                  </div>

                  {/* Chunk text content */}
                  <div style={{
                    "font-size": "13px",
                    "line-height": "1.7",
                    color: isMatch()
                      ? "var(--color-text-primary)"
                      : "var(--color-text-secondary)",
                    "white-space": "pre-wrap",
                    "word-break": "break-word",
                  }}>
                    {chunk.content}
                  </div>
                </div>
              );
            }}
          </For>

          {/* Bottom sentinel: always rendered so IntersectionObserver stays attached.
              The loadDownward function returns early when canLoadDown() is false. */}
          <Show when={loadingDown()}>
            <div style={{
              padding: "8px 16px",
              "text-align": "center",
              "font-size": "11px",
              color: "var(--color-text-muted)",
            }}>
              Loading more chunks...
            </div>
          </Show>
          <div ref={bottomSentinelRef} style={{ height: "1px" }} />

          {/* Boundary indicator when at the end of the file */}
          <Show when={!canLoadDown() && totalChunks() > 0 && maxLoadedIndex() >= totalChunks() - 1}>
            <div style={{
              padding: "8px 16px",
              "text-align": "center",
              "font-size": "10px",
              color: "var(--color-text-muted)",
              "letter-spacing": "2px",
              "text-transform": "uppercase",
              opacity: "0.5",
            }}>
              end of document
            </div>
          </Show>
        </Show>
      </div>
    </div>
  );
};

export default ChunkViewer;
