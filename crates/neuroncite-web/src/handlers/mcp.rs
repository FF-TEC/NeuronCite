// NeuronCite -- local, privacy-preserving semantic document search engine.
// Copyright (C) 2026 NeuronCite Contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! MCP (Model Context Protocol) registration status and management handlers.
//!
//! Provides endpoints to check the MCP server registration status, install
//! (register) the NeuronCite MCP server in the client's configuration, and
//! uninstall (deregister) it. The actual registration logic reuses the
//! existing `neuroncite mcp install/uninstall` subprocess commands.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Serialize;

use crate::WebState;

/// Response body for GET /api/v1/web/mcp/status.
///
/// Field names match the TypeScript McpStatusResponse interface in
/// frontend/src/api/types.ts exactly. The `registered` boolean indicates
/// whether a NeuronCite entry is present in the Claude Code mcpServers
/// config. `server_version` carries the binary's own package version string.
#[derive(Serialize)]
pub struct McpStatusResponse {
    /// True when the `mcpServers.neuroncite` key exists in `~/.claude.json`.
    /// False when the file is absent, the key is missing, or a parse error
    /// prevents reading the file.
    pub registered: bool,
    /// The current NeuronCite binary version string from `CARGO_PKG_VERSION`.
    pub server_version: String,
}

/// Returns the current MCP server registration status by checking the
/// Claude Code configuration file (`~/.claude.json`) for a NeuronCite
/// server entry in the `mcpServers` object.
///
/// The response contains a boolean `registered` field and the binary version
/// string. On read or parse errors the file is treated as not registered
/// (registered = false) rather than returning an error status.
pub async fn mcp_status(State(_state): State<Arc<WebState>>) -> Json<McpStatusResponse> {
    let config_path = get_claude_config_path();

    // Parse the config file as JSON and check the mcpServers.neuroncite key.
    // A string search is insufficient because unrelated keys (e.g. project
    // history entries) may contain "neuroncite" as a substring.
    // On file-not-found (fresh Claude Code install) or parse errors, treat
    // registration as absent rather than propagating an error to the frontend.
    let registered = match std::fs::read_to_string(&config_path) {
        Ok(contents) => match serde_json::from_str::<serde_json::Value>(&contents) {
            Ok(json) => json
                .get("mcpServers")
                .and_then(|s| s.get("neuroncite"))
                .is_some(),
            // Config file exists but contains invalid JSON. Treat as not registered.
            Err(_) => false,
        },
        // File not found is the normal state on a fresh Claude Code installation.
        // Any other I/O error also maps to not registered.
        Err(_) => false,
    };

    Json(McpStatusResponse {
        registered,
        server_version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Registers the NeuronCite MCP server by spawning the `neuroncite mcp install`
/// subprocess. Returns 202 Accepted on success.
pub async fn mcp_install(
    State(_state): State<Arc<WebState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    match spawn_mcp_command("install") {
        Ok(output) => {
            if output.status.success() {
                (
                    StatusCode::ACCEPTED,
                    Json(serde_json::json!({ "status": "installed" })),
                )
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("MCP install failed: {}", stderr.trim())
                    })),
                )
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to spawn MCP install: {e}")
            })),
        ),
    }
}

/// Unregisters the NeuronCite MCP server by spawning the `neuroncite mcp uninstall`
/// subprocess. Returns 202 Accepted on success.
pub async fn mcp_uninstall(
    State(_state): State<Arc<WebState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    match spawn_mcp_command("uninstall") {
        Ok(output) => {
            if output.status.success() {
                (
                    StatusCode::ACCEPTED,
                    Json(serde_json::json!({ "status": "uninstalled" })),
                )
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("MCP uninstall failed: {}", stderr.trim())
                    })),
                )
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to spawn MCP uninstall: {e}")
            })),
        ),
    }
}

/// Spawns the `neuroncite mcp <action>` subprocess and captures its output.
fn spawn_mcp_command(action: &str) -> std::io::Result<std::process::Output> {
    let exe = std::env::current_exe()?;
    std::process::Command::new(exe)
        .args(["mcp", action])
        .output()
}

/// Returns the platform-appropriate path to the Claude Code MCP configuration
/// file. Claude Code reads MCP server definitions from the top-level
/// `mcpServers` key in `~/.claude.json`. This is the same path that
/// `neuroncite_mcp::registration` reads and writes during install/uninstall.
///
/// Platform resolution:
/// - Windows: `%USERPROFILE%\.claude.json`
/// - macOS/Linux: `$HOME/.claude.json`
fn get_claude_config_path() -> std::path::PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(userprofile) = std::env::var("USERPROFILE") {
            return std::path::PathBuf::from(userprofile).join(".claude.json");
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home).join(".claude.json");
        }
    }

    std::path::PathBuf::from(".claude.json")
}
