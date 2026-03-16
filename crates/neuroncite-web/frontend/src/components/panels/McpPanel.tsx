import { Component, Show, createResource, createSignal } from "solid-js";
import { api } from "../../api/client";
import type { McpStatusResponse } from "../../api/types";

/**
 * MCP (Model Context Protocol) panel showing the registration status of the
 * NeuronCite MCP server. Provides install/uninstall buttons to register or
 * remove the MCP server configuration entry in ~/.claude.json.
 *
 * All API calls go through the typed client (api.mcpStatus, api.mcpAction)
 * instead of raw fetch(). The status response carries a boolean `registered`
 * field and the binary version string.
 */

const McpPanel: Component = () => {
  const [mcpStatus, { refetch }] = createResource(fetchMcpStatus);
  const [acting, setActing] = createSignal(false);

  /** Fetches MCP registration status from the server via the typed API client. */
  async function fetchMcpStatus(): Promise<McpStatusResponse> {
    return api.mcpStatus();
  }

  /** Installs or uninstalls the MCP server registration via the typed API client. */
  const mcpAction = async (action: "install" | "uninstall") => {
    setActing(true);
    try {
      await api.mcpAction(action);
      await refetch();
    } catch (e) {
      console.error(`MCP ${action} failed:`, e);
    } finally {
      setActing(false);
    }
  };

  return (
    <div>
      <Show when={mcpStatus()}>
        <div style={{ display: "flex", "flex-direction": "column", gap: "12px" }}>
          {/* Status display */}
          <div class="glass-card" style={{ padding: "12px 16px" }}>
            <div style={{ display: "flex", "align-items": "center", gap: "10px", "margin-bottom": "12px" }}>
              <div
                style={{
                  width: "10px",
                  height: "10px",
                  "border-radius": "50%",
                  background: mcpStatus()!.registered
                    ? "var(--color-accent-cyan)"
                    : "var(--color-status-pending)",
                }}
              />
              <span style={{ "font-size": "14px", "font-weight": "600" }}>
                MCP Server: {mcpStatus()!.registered ? "Registered" : "Not Registered"}
              </span>
            </div>

            {/* Server version */}
            <div style={{ "font-size": "12px", color: "var(--color-text-muted)" }}>
              Version: {mcpStatus()!.server_version}
            </div>
          </div>

          {/* Action buttons */}
          <div class="row" style={{ gap: "8px" }}>
            <Show when={!mcpStatus()!.registered}>
              <button
                class="btn btn-primary"
                onClick={() => mcpAction("install")}
                disabled={acting()}
              >
                {acting() ? "Installing..." : "Install MCP Server"}
              </button>
            </Show>
            <Show when={mcpStatus()!.registered}>
              <button
                class="btn"
                onClick={() => mcpAction("uninstall")}
                disabled={acting()}
              >
                {acting() ? "Uninstalling..." : "Uninstall MCP Server"}
              </button>
            </Show>
            <button class="btn btn-sm" onClick={() => refetch()} disabled={mcpStatus.loading}>
              Refresh
            </button>
          </div>
        </div>
      </Show>

      <Show when={mcpStatus.loading}>
        <div style={{ color: "var(--color-text-muted)", "text-align": "center", padding: "20px" }}>
          Loading MCP status...
        </div>
      </Show>

      <Show when={mcpStatus.error}>
        <div style={{ color: "var(--color-accent-magenta)", "text-align": "center", padding: "20px", "font-size": "13px" }}>
          Failed to load MCP status. The server may not support this endpoint yet.
        </div>
      </Show>
    </div>
  );
};

export default McpPanel;
