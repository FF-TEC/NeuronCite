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
//! Provides endpoints to check the MCP server registration status for both
//! Claude Code and Claude Desktop App, install (register) the NeuronCite MCP
//! server in either client's configuration, and uninstall (deregister) it.
//! The actual registration logic reuses the existing `neuroncite mcp
//! install/uninstall` subprocess commands with a `--target` flag.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::Serialize;

use crate::WebState;

/// Status for a single MCP target (Claude Code or Claude Desktop App).
#[derive(Serialize)]
pub struct McpTargetStatus {
    /// True when the `mcpServers.neuroncite` key exists in the target's config.
    pub registered: bool,
    /// Filesystem path to the config file for this target.
    pub config_path: String,
}

/// Response body for GET /api/v1/web/mcp/status.
///
/// Returns registration status for both supported MCP targets independently,
/// plus the NeuronCite binary version string.
#[derive(Serialize)]
pub struct McpStatusResponse {
    /// Registration status for Claude Code (`~/.claude.json`).
    pub claude_code: McpTargetStatus,
    /// Registration status for Claude Desktop App (platform-specific path).
    pub claude_desktop: McpTargetStatus,
    /// The current NeuronCite binary version string from `CARGO_PKG_VERSION`.
    pub server_version: String,
}

/// Checks registration status for a single config file path. Returns whether
/// the `mcpServers.neuroncite` key is present.
fn check_registered(config_path: &std::path::Path) -> bool {
    match std::fs::read_to_string(config_path) {
        Ok(contents) => match serde_json::from_str::<serde_json::Value>(&contents) {
            Ok(json) => json
                .get("mcpServers")
                .and_then(|s| s.get("neuroncite"))
                .is_some(),
            Err(_) => false,
        },
        Err(_) => false,
    }
}

/// Returns the Claude Code config path (`~/.claude.json`).
fn claude_code_config_path() -> std::path::PathBuf {
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

/// Returns the Claude Desktop App config path (platform-specific).
fn claude_desktop_config_path() -> std::path::PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home)
                .join("Library/Application Support/Claude/claude_desktop_config.json");
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            return std::path::PathBuf::from(appdata)
                .join("Claude")
                .join("claude_desktop_config.json");
        }
        if let Ok(userprofile) = std::env::var("USERPROFILE") {
            return std::path::PathBuf::from(userprofile)
                .join("AppData")
                .join("Roaming")
                .join("Claude")
                .join("claude_desktop_config.json");
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home)
                .join(".config/Claude/claude_desktop_config.json");
        }
    }

    std::path::PathBuf::from("claude_desktop_config.json")
}

/// Returns MCP registration status for both Claude Code and Claude Desktop App.
pub async fn mcp_status(State(_state): State<Arc<WebState>>) -> Json<McpStatusResponse> {
    let code_path = claude_code_config_path();
    let desktop_path = claude_desktop_config_path();

    Json(McpStatusResponse {
        claude_code: McpTargetStatus {
            registered: check_registered(&code_path),
            config_path: code_path.display().to_string(),
        },
        claude_desktop: McpTargetStatus {
            registered: check_registered(&desktop_path),
            config_path: desktop_path.display().to_string(),
        },
        server_version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Validates the target path parameter.
fn validate_target(target: &str) -> Result<&str, String> {
    match target {
        "claude-code" | "claude-desktop" => Ok(target),
        other => Err(format!(
            "unknown MCP target '{other}' (expected 'claude-code' or 'claude-desktop')"
        )),
    }
}

/// Registers the NeuronCite MCP server for the specified target by spawning
/// the `neuroncite mcp install --target <target>` subprocess.
pub async fn mcp_install(
    State(_state): State<Arc<WebState>>,
    Path(target): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(e) = validate_target(&target) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e })),
        );
    }

    match spawn_mcp_command("install", &target) {
        Ok(output) => {
            if output.status.success() {
                (
                    StatusCode::ACCEPTED,
                    Json(serde_json::json!({ "status": "installed", "target": target })),
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

/// Unregisters the NeuronCite MCP server for the specified target by spawning
/// the `neuroncite mcp uninstall --target <target>` subprocess.
pub async fn mcp_uninstall(
    State(_state): State<Arc<WebState>>,
    Path(target): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(e) = validate_target(&target) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e })),
        );
    }

    match spawn_mcp_command("uninstall", &target) {
        Ok(output) => {
            if output.status.success() {
                (
                    StatusCode::ACCEPTED,
                    Json(serde_json::json!({ "status": "uninstalled", "target": target })),
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

/// Spawns the `neuroncite mcp <action> --target <target>` subprocess and
/// captures its output.
fn spawn_mcp_command(action: &str, target: &str) -> std::io::Result<std::process::Output> {
    let exe = std::env::current_exe()?;
    std::process::Command::new(exe)
        .args(["mcp", action, "--target", target])
        .output()
}
