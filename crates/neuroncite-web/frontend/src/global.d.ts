/**
 * Global type augmentations for the NeuronCite frontend.
 *
 * In the native GUI build (tao + wry), the `window.ipc` object is injected
 * by wry's JavaScript evaluation layer. It exposes a single `postMessage`
 * method that delivers a UTF-8 string to the Rust IPC handler. In browser
 * dev mode, `window.ipc` is undefined because no wry runtime is present.
 *
 * This declaration eliminates the need for `(window as any).ipc` casts
 * throughout the codebase by making `window.ipc` an optional typed property.
 */
declare global {
  interface Window {
    ipc?: {
      postMessage(msg: string): void;
    };
  }
}

export {};
