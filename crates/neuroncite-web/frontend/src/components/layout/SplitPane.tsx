import { Component, JSX, createSignal, onMount, onCleanup } from "solid-js";

/**
 * Resizable horizontal split pane component. Renders two child panels
 * separated by a draggable divider. The left panel has a controlled pixel
 * width; the right panel fills the remaining space.
 *
 * Drag interaction uses pointer events with setPointerCapture so the
 * divider keeps tracking even when the cursor moves outside its bounds.
 * During drag, text selection is suppressed and the cursor is forced to
 * col-resize across the entire document.
 *
 * Drag is disabled when the viewport is narrower than NARROW_BREAKPOINT_PX.
 * At narrow widths the two panels stack vertically and the drag handle is
 * hidden to avoid confusing interactions on small screens.
 */

/** Viewport width in pixels below which the drag divider is disabled. */
const NARROW_BREAKPOINT_PX = 1024;

interface SplitPaneProps {
  /** Content rendered in the left panel. */
  left: JSX.Element;
  /** Content rendered in the right panel. */
  right: JSX.Element;
  /** Current width of the left panel in pixels (controlled externally). */
  leftWidth: number;
  /** Callback fired during and at the end of a drag with the new width. */
  onResize: (width: number) => void;
  /** Minimum allowed width for the left panel. Defaults to 240. */
  minLeft?: number;
  /** Minimum allowed width for the right panel. Defaults to 300. */
  minRight?: number;
}

const SplitPane: Component<SplitPaneProps> = (props) => {
  const [dragging, setDragging] = createSignal(false);
  // Tracks whether the viewport is wide enough to allow drag interaction.
  // Initialized from window.innerWidth and updated via a matchMedia listener.
  const [dragEnabled, setDragEnabled] = createSignal(
    window.innerWidth >= NARROW_BREAKPOINT_PX,
  );
  let containerRef: HTMLDivElement | undefined;

  /** Computes a clamped width from the pointer x-position relative to the
   *  container's left edge. Ensures both panels respect their minimum widths. */
  const clampWidth = (clientX: number): number => {
    if (!containerRef) return props.leftWidth;
    const rect = containerRef.getBoundingClientRect();
    const minL = props.minLeft ?? 240;
    const minR = props.minRight ?? 300;
    const maxLeft = rect.width - minR - 6; // 6px divider width
    const raw = clientX - rect.left;
    return Math.max(minL, Math.min(maxLeft, raw));
  };

  /** Pointer move handler attached to the document during drag.
   *  Computes new left panel width and reports it to the parent. */
  const onPointerMove = (e: PointerEvent) => {
    if (!dragging()) return;
    e.preventDefault();
    props.onResize(clampWidth(e.clientX));
  };

  /** Pointer up handler ends the drag and restores normal text selection. */
  const onPointerUp = () => {
    if (!dragging()) return;
    setDragging(false);
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
  };

  onMount(() => {
    document.addEventListener("pointermove", onPointerMove);
    document.addEventListener("pointerup", onPointerUp);

    // Update dragEnabled whenever the viewport crosses the breakpoint.
    // matchMedia is more efficient than a resize event listener because
    // it fires only when the media condition changes, not on every resize.
    const mq = window.matchMedia(`(min-width: ${NARROW_BREAKPOINT_PX}px)`);
    const handleMediaChange = (ev: MediaQueryListEvent) => {
      setDragEnabled(ev.matches);
      // End any active drag when switching to narrow layout so the body
      // styles are restored and the component is not left in a dragging state.
      if (!ev.matches && dragging()) {
        setDragging(false);
        document.body.style.userSelect = "";
        document.body.style.cursor = "";
      }
    };
    mq.addEventListener("change", handleMediaChange);

    onCleanup(() => {
      mq.removeEventListener("change", handleMediaChange);
    });
  });

  onCleanup(() => {
    document.removeEventListener("pointermove", onPointerMove);
    document.removeEventListener("pointerup", onPointerUp);
    // Restore body styles in case component unmounts during an active drag
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
  });

  /** Pointer down on the divider starts the drag. Sets pointer capture so
   *  events continue even when the cursor leaves the divider element.
   *  No-ops when the viewport is narrower than the breakpoint. */
  const onDividerPointerDown = (e: PointerEvent) => {
    if (!dragEnabled()) return;
    e.preventDefault();
    setDragging(true);
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  return (
    <div ref={containerRef} class="split-pane">
      <div
        class="split-pane-left"
        style={{ "flex-basis": `${props.leftWidth}px` }}
      >
        {props.left}
      </div>
      <div
        class={`split-pane-divider${dragging() ? " dragging" : ""}${!dragEnabled() ? " disabled" : ""}`}
        onPointerDown={onDividerPointerDown}
        style={{ cursor: dragEnabled() ? "col-resize" : "default" }}
      />
      <div class="split-pane-right">
        {props.right}
      </div>
    </div>
  );
};

export default SplitPane;
