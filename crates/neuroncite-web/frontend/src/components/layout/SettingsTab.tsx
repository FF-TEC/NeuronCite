import { Component } from "solid-js";
import MaintenancePanel from "../panels/MaintenancePanel";
import DoctorPanel from "../panels/DoctorPanel";
import LogPanel from "../panels/LogPanel";
import McpPanel from "../panels/McpPanel";
import AboutPanel from "../panels/AboutPanel";

/**
 * Settings tab consolidating the five utility panels rendered inline as
 * glass-card sections within a single scrollable column layout. The panel
 * components are imported without modification and work identically whether
 * rendered inside a modal or inline.
 *
 * Section order: Maintenance (database operations), Dependencies (runtime
 * dependency probes), MCP Server (Model Context Protocol registration and
 * management), Log Viewer (real-time SSE tracing logs), About (version and
 * build information). The tab content stretches to the full viewport width
 * with the scrollbar at the window edge.
 */
const SettingsTab: Component = () => {
  return (
    <div class="settings-tab">
      {/* Maintenance: FTS5 optimization, HNSW index rebuild, Reset to Defaults.
       *  Requires an active session for database operations. */}
      <section class="glass-card settings-section">
        <div class="section-title">Maintenance</div>
        <MaintenancePanel />
      </section>

      {/* Dependencies: pdfium, tesseract, poppler availability probes.
       *  Each dependency shows availability status and an install button
       *  when the dependency is missing and installable. */}
      <section class="glass-card settings-section">
        <div class="section-title">Dependencies</div>
        <DoctorPanel />
      </section>

      {/* MCP Server: registration status, install/uninstall buttons, and
       *  display of the config file path and executable path. */}
      <section class="glass-card settings-section">
        <div class="section-title">MCP Server</div>
        <McpPanel />
      </section>

      {/* Log Viewer: real-time tracing log messages received via SSE.
       *  Shows a monospace scrollable list with auto-scroll and a clear button.
       *  Buffer is capped at 500 messages in the store. */}
      <section class="glass-card settings-section">
        <div class="section-title">Log Viewer</div>
        <LogPanel />
      </section>

      {/* About: application version, build features, and runtime backend
       *  status read from the health endpoint response stored in the global
       *  app state. Placed last because version information is reference-only
       *  and accessed less frequently than the operational panels above. */}
      <section class="glass-card settings-section">
        <div class="section-title">About</div>
        <AboutPanel />
      </section>
    </div>
  );
};

export default SettingsTab;
