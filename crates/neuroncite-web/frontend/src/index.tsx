/* @refresh reload */
import { render } from "solid-js/web";
import App from "./App";

import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/animations.css";
import "./styles/layout.css";
import "./styles/components.css";

// Log unhandled promise rejections to the console for debugging. This catches
// async errors that are not handled by an ErrorBoundary or explicit .catch().
window.addEventListener('unhandledrejection', (ev) => {
  console.error('unhandled async error:', ev.reason);
});

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element not found. The HTML template must contain a <div id='root'>.");
}

render(() => <App />, root);
