import { Component, JSX } from "solid-js";

/**
 * Floating glass-morphism modal panel used for the Tools panels (Models,
 * Maintenance, Doctor, Log, MCP, Citation). Renders a header with title
 * and close button, plus a scrollable content area. The modal is overlaid
 * on a semi-transparent backdrop that closes the modal on click.
 *
 * When `fullscreen` is true, the modal expands to fill the browser viewport
 * with minimal margins (used by the Citation Verification panel which
 * needs the full width for its results table).
 *
 * Keyboard: pressing Escape triggers the onClose callback.
 */
const Modal: Component<{
  title: string;
  onClose: () => void;
  fullscreen?: boolean;
  children: JSX.Element;
}> = (props) => {
  /** Close on Escape key press. The handler is attached to the backdrop
   *  div which receives focus via tabIndex. */
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      props.onClose();
    }
  };

  return (
    <div class="modal-backdrop" onClick={props.onClose} onKeyDown={onKeyDown} tabIndex={-1}>
      <div
        class="modal"
        onClick={(e) => e.stopPropagation()}
        style={props.fullscreen
          ? { width: "calc(100vw - 40px)", "max-width": "calc(100vw - 40px)", height: "calc(100vh - 40px)", "max-height": "calc(100vh - 40px)" }
          : { "max-width": "700px", width: "90vw" }
        }
      >
        <div class="modal-header">
          <span style={{ "font-weight": "600", "font-size": "14px" }}>{props.title}</span>
          <button class="btn btn-sm" onClick={props.onClose}>
            X
          </button>
        </div>
        <div class="modal-body">{props.children}</div>
      </div>
    </div>
  );
};

export default Modal;
