import { Component, Show, createSignal, JSX } from "solid-js";
import { Portal } from "solid-js/web";

// ---------------------------------------------------------------------------
// Tip: inline help icon that shows a descriptive tooltip on hover.
// Renders a small circled "i" icon; hovering reveals a position:fixed panel
// with the explanation text. The tooltip content is rendered via a SolidJS
// Portal into document.body so that ancestor CSS properties (backdrop-filter,
// transform, etc.) on glass-card containers cannot create a new containing
// block that breaks position:fixed viewport coordinates.
// ---------------------------------------------------------------------------

interface TipProps {
  /** Tooltip text displayed on hover. */
  text: string | JSX.Element;
}

const Tip: Component<TipProps> = (props) => {
  const [visible, setVisible] = createSignal(false);
  const [pos, setPos] = createSignal({ top: 0, left: 0 });
  const [flipped, setFlipped] = createSignal(false);
  let iconRef: HTMLSpanElement | undefined;
  let contentRef: HTMLSpanElement | undefined;

  /** Measures the icon's viewport position and places the tooltip panel
   *  8px below it by default. If the tooltip would overflow the bottom
   *  of the viewport, flips it to appear above the icon instead.
   *  Horizontal position is clamped to keep the 280px panel on-screen. */
  const onEnter = () => {
    if (!iconRef) return;
    const rect = iconRef.getBoundingClientRect();
    const left = Math.min(rect.left - 4, window.innerWidth - 292);
    // Place below initially
    setPos({ top: rect.bottom + 8, left: Math.max(8, left) });
    setFlipped(false);
    setVisible(true);

    // After the browser paints the tooltip, measure its actual height.
    // If it overflows the viewport bottom, reposition above the icon.
    requestAnimationFrame(() => {
      if (!contentRef) return;
      const tipRect = contentRef.getBoundingClientRect();
      if (tipRect.bottom > window.innerHeight - 8) {
        setPos({ top: rect.top - tipRect.height - 8, left: Math.max(8, left) });
        setFlipped(true);
      }
    });
  };

  return (
    <span class="tip-trigger" onMouseEnter={onEnter} onMouseLeave={() => setVisible(false)}>
      <span ref={iconRef} class="tip-icon">i</span>
      <Show when={visible()}>
        <Portal>
          <span
            ref={contentRef}
            class={`tip-content${flipped() ? " tip-above" : ""}`}
            style={{ top: `${pos().top}px`, left: `${pos().left}px` }}
          >
            {props.text}
          </span>
        </Portal>
      </Show>
    </span>
  );
};

export default Tip;
