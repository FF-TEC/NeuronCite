import { Component, Show, createResource, createSignal } from "solid-js";
import { api } from "../../api/client";
import type { McpStatusResponse, McpTarget, McpTargetStatus } from "../../api/types";

/**
 * MCP (Model Context Protocol) panel showing registration status for both
 * Claude Code (CLI) and Claude Desktop App (GUI). Each target can be
 * installed/uninstalled independently.
 *
 * All API calls go through the typed client (api.mcpStatus, api.mcpAction).
 * The status endpoint returns both targets in a single response.
 */

/** Props for a single target card within the MCP panel. */
interface McpTargetCardProps {
  name: string;
  description: string;
  status: McpTargetStatus;
  target: McpTarget;
  actingTarget: McpTarget | null;
  onAction: (action: "install" | "uninstall", target: McpTarget) => void;
}

/** Renders a single MCP target card with status, config path, and action button. */
const McpTargetCard: Component<McpTargetCardProps> = (props) => {
  const isActing = () => props.actingTarget === props.target;

  return (
    <div class="glass-card" style={{ padding: "12px 16px" }}>
      {/* Header with status dot and target name */}
      <div style={{ display: "flex", "align-items": "center", gap: "10px", "margin-bottom": "8px" }}>
        <div
          style={{
            width: "10px",
            height: "10px",
            "border-radius": "50%",
            "flex-shrink": "0",
            background: props.status.registered
              ? "var(--color-accent-cyan)"
              : "var(--color-status-pending)",
          }}
        />
        <span style={{ "font-size": "14px", "font-weight": "600" }}>
          {props.name}: {props.status.registered ? "Registered" : "Not Registered"}
        </span>
      </div>

      {/* Description */}
      <div style={{
        "font-size": "12px",
        color: "var(--color-text-secondary)",
        "margin-bottom": "8px",
        "line-height": "1.5",
      }}>
        {props.description}
      </div>

      {/* Config path */}
      <div style={{
        "font-size": "11px",
        color: "var(--color-text-muted)",
        "margin-bottom": "12px",
        "font-family": "var(--font-mono, monospace)",
        "word-break": "break-all",
      }}>
        {props.status.config_path}
      </div>

      {/* Action button */}
      <Show when={!props.status.registered}>
        <button
          class="btn btn-primary"
          onClick={() => props.onAction("install", props.target)}
          disabled={isActing()}
        >
          {isActing() ? "Installing..." : "Install"}
        </button>
      </Show>
      <Show when={props.status.registered}>
        <button
          class="btn"
          onClick={() => props.onAction("uninstall", props.target)}
          disabled={isActing()}
        >
          {isActing() ? "Uninstalling..." : "Uninstall"}
        </button>
      </Show>
    </div>
  );
};

const McpPanel: Component = () => {
  const [mcpStatus, { refetch, mutate }] = createResource(fetchMcpStatus);
  const [actingTarget, setActingTarget] = createSignal<McpTarget | null>(null);

  async function fetchMcpStatus(): Promise<McpStatusResponse> {
    return api.mcpStatus();
  }

  /** Re-fetches MCP status without clearing the current value, so the UI
   *  stays mounted and the scroll position is preserved. */
  async function silentRefetch() {
    try {
      const updated = await api.mcpStatus();
      mutate(updated);
    } catch (e) {
      console.error("MCP status refresh failed:", e);
    }
  }

  const mcpAction = async (action: "install" | "uninstall", target: McpTarget) => {
    setActingTarget(target);
    try {
      await api.mcpAction(action, target);
      await silentRefetch();
    } catch (e) {
      console.error(`MCP ${action} for ${target} failed:`, e);
    } finally {
      setActingTarget(null);
    }
  };

  return (
    <div>
      <Show when={mcpStatus()}>
        <div style={{ display: "flex", "flex-direction": "column", gap: "12px" }}>
          {/* Introductory explanation */}
          <div style={{
            "font-size": "12px",
            color: "var(--color-text-secondary)",
            "line-height": "1.6",
            "margin-bottom": "4px",
          }}>
            NeuronCite can register as an MCP server with two different Claude
            clients. Each option can be enabled independently. Claude Code (CLI)
            and the Claude Desktop App use separate configuration files, so
            registration must be done for each client separately.
          </div>

          {/* Claude Code target card */}
          <McpTargetCard
            name="Claude Code (CLI)"
            description="Registers NeuronCite as an MCP server for Claude Code, the command-line interface for Claude. Use this if you interact with Claude through the terminal or VS Code extension."
            status={mcpStatus()!.claude_code}
            target="claude-code"
            actingTarget={actingTarget()}
            onAction={mcpAction}
          />

          {/* Claude Desktop App target card */}
          <McpTargetCard
            name="Claude Desktop App"
            description="Registers NeuronCite as an MCP server for the Claude Desktop App (GUI application). Use this if you interact with Claude through the standalone desktop application."
            status={mcpStatus()!.claude_desktop}
            target="claude-desktop"
            actingTarget={actingTarget()}
            onAction={mcpAction}
          />

          {/* Footer: version and refresh */}
          <div style={{ display: "flex", "align-items": "center", "justify-content": "space-between" }}>
            <div style={{ "font-size": "12px", color: "var(--color-text-muted)" }}>
              Server Version: {mcpStatus()!.server_version}
            </div>
            <button class="btn btn-sm" onClick={silentRefetch} disabled={mcpStatus.loading}>
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
