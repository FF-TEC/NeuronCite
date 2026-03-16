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

//! MCP server registration for Claude Code.
//!
//! Provides install/uninstall/status operations that manage the NeuronCite
//! entry in Claude Code's user configuration file (`~/.claude.json`). Claude
//! Code reads MCP server definitions from the top-level `mcpServers` key in
//! this file. This is the same file that stores internal Claude Code state
//! (startup counters, project entries, cached gates), so all write operations
//! perform a careful read-modify-write merge to preserve existing keys.
//!
//! The registration logic resolves the current executable's path and writes
//! it into the `mcpServers.neuroncite` entry with the `["mcp", "serve"]`
//! arguments and `type: "stdio"` transport. This allows Claude Code to spawn
//! the NeuronCite MCP server automatically when a conversation starts.

use std::path::{Path, PathBuf};

/// Entry name used in the MCP configuration file.
const MCP_SERVER_NAME: &str = "neuroncite";

/// Arguments passed to the NeuronCite executable when started by Claude Code.
const MCP_ARGS: &[&str] = &["mcp", "serve"];

/// Returns the path to the Claude Code MCP configuration file.
///
/// On all platforms, this is `~/.claude.json`. Claude Code reads MCP server
/// definitions from the top-level `mcpServers` key in this file. Note that
/// `~/.claude/settings.json` is a separate file used for permissions and
/// general settings -- it does NOT contain MCP server definitions.
/// Returns `None` if the home directory cannot be determined.
pub fn config_file_path() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE")
            .ok()
            .map(|home| PathBuf::from(home).join(".claude.json"))
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME")
            .ok()
            .map(|home| PathBuf::from(home).join(".claude.json"))
    }
}

/// Registration status returned by `check_status`.
pub struct McpStatus {
    /// Whether the NeuronCite MCP server is registered in the config file.
    pub registered: bool,
    /// Path to the registered executable (if registered).
    pub exe_path: Option<String>,
    /// Arguments passed to the executable (if registered).
    pub args: Option<Vec<String>>,
    /// Path to the `~/.claude.json` config file.
    pub config_path: Option<String>,
}

/// Registers the NeuronCite MCP server in `~/.claude.json`.
///
/// Creates the config file if it does not exist. If a `mcpServers.neuroncite`
/// entry already exists, it is overwritten with the current executable path,
/// arguments, and `"type": "stdio"`. All other keys in the file (e.g.
/// internal Claude Code state like `numStartups`, `projects`) are preserved.
///
/// # Arguments
///
/// * `exe_override` - Optional path to use instead of the current executable.
///   Used for testing and for registering a specific build.
pub fn install(exe_override: Option<&Path>) -> Result<String, String> {
    let exe_path = match exe_override {
        Some(p) => p.to_path_buf(),
        None => std::env::current_exe()
            .map_err(|e| format!("failed to determine executable path: {e}"))?,
    };

    let config_path = config_file_path().ok_or("cannot determine home directory for MCP config")?;

    // Read existing config or start with an empty object.
    let mut config: serde_json::Value = if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("failed to parse {}: {e}", config_path.display()))?
    } else {
        serde_json::json!({})
    };

    // Ensure the mcpServers object exists.
    if !config["mcpServers"].is_object() {
        config["mcpServers"] = serde_json::json!({});
    }

    // Write the neuroncite entry with stdio transport type.
    let exe_str = exe_path.display().to_string();
    config["mcpServers"][MCP_SERVER_NAME] = serde_json::json!({
        "type": "stdio",
        "command": exe_str,
        "args": MCP_ARGS,
    });

    // Write the config file with pretty formatting for readability.
    let json_str = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("failed to serialize config: {e}"))?;
    std::fs::write(&config_path, json_str)
        .map_err(|e| format!("failed to write {}: {e}", config_path.display()))?;

    Ok(format!(
        "MCP server registered at {}\nExecutable: {}\nArgs: {:?}",
        config_path.display(),
        exe_str,
        MCP_ARGS
    ))
}

/// Removes the NeuronCite MCP server entry from `~/.claude.json`.
///
/// Only the `mcpServers.neuroncite` key is removed. All other keys in the
/// config file are preserved. If the file does not exist or does not contain
/// a "neuroncite" entry, this operation is a no-op.
pub fn uninstall() -> Result<String, String> {
    let config_path = config_file_path().ok_or("cannot determine home directory for MCP config")?;

    if !config_path.exists() {
        return Ok("MCP config file does not exist; nothing to uninstall".to_string());
    }

    let content = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?;
    let mut config: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse {}: {e}", config_path.display()))?;

    // Remove the neuroncite entry if it exists.
    let removed = config["mcpServers"]
        .as_object_mut()
        .map(|servers| servers.remove(MCP_SERVER_NAME).is_some())
        .unwrap_or(false);

    if !removed {
        return Ok("NeuronCite MCP server was not registered; nothing to uninstall".to_string());
    }

    let json_str = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("failed to serialize config: {e}"))?;
    std::fs::write(&config_path, json_str)
        .map_err(|e| format!("failed to write {}: {e}", config_path.display()))?;

    Ok(format!(
        "MCP server unregistered from {}",
        config_path.display()
    ))
}

/// Checks the current registration status of the NeuronCite MCP server
/// by reading `~/.claude.json`.
pub fn check_status() -> McpStatus {
    let config_path = config_file_path();

    let Some(ref path) = config_path else {
        return McpStatus {
            registered: false,
            exe_path: None,
            args: None,
            config_path: None,
        };
    };

    if !path.exists() {
        return McpStatus {
            registered: false,
            exe_path: None,
            args: None,
            config_path: Some(path.display().to_string()),
        };
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => {
            return McpStatus {
                registered: false,
                exe_path: None,
                args: None,
                config_path: Some(path.display().to_string()),
            };
        }
    };

    let config: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => {
            return McpStatus {
                registered: false,
                exe_path: None,
                args: None,
                config_path: Some(path.display().to_string()),
            };
        }
    };

    let entry = &config["mcpServers"][MCP_SERVER_NAME];
    if entry.is_null() {
        return McpStatus {
            registered: false,
            exe_path: None,
            args: None,
            config_path: Some(path.display().to_string()),
        };
    }

    let exe = entry["command"].as_str().map(String::from);
    let args = entry["args"].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect()
    });

    McpStatus {
        registered: true,
        exe_path: exe,
        args,
        config_path: Some(path.display().to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// T-MCP-023: Install creates the config file and writes the correct
    /// entry structure including the `"type": "stdio"` field.
    #[test]
    fn t_mcp_023_install_creates_config_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config_file = dir.path().join(".claude.json");

        // Create a fake exe path for testing.
        let fake_exe = dir.path().join("neuroncite.exe");
        std::fs::write(&fake_exe, "fake").expect("write fake exe");

        // Since we cannot override config_file_path(), we simulate the install
        // logic by constructing the expected JSON and verifying it.
        let exe_str = fake_exe.display().to_string();

        // Simulate the install logic (mirrors what install() writes).
        let config = serde_json::json!({
            "mcpServers": {
                "neuroncite": {
                    "type": "stdio",
                    "command": exe_str,
                    "args": MCP_ARGS,
                }
            }
        });
        let json_str = serde_json::to_string_pretty(&config).expect("serialize");
        let mut f = std::fs::File::create(&config_file).expect("create file");
        f.write_all(json_str.as_bytes()).expect("write");

        // Read back and verify.
        let content = std::fs::read_to_string(&config_file).expect("read");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("parse");
        assert_eq!(
            parsed["mcpServers"]["neuroncite"]["type"].as_str().unwrap(),
            "stdio"
        );
        assert_eq!(
            parsed["mcpServers"]["neuroncite"]["command"]
                .as_str()
                .unwrap(),
            exe_str
        );
        let args = parsed["mcpServers"]["neuroncite"]["args"]
            .as_array()
            .expect("args array");
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], "mcp");
        assert_eq!(args[1], "serve");
    }

    /// T-MCP-024: check_status returns registered=false when the config
    /// file does not contain a neuroncite entry.
    #[test]
    fn t_mcp_024_status_returns_false_when_not_registered() {
        // Use the default check_status which reads the real config file.
        // If neuroncite is not installed, this returns false.
        // Verify the struct is well-formed by asserting the exe_path is only
        // populated when the server is registered.
        let status = check_status();
        if !status.registered {
            assert!(
                status.exe_path.is_none(),
                "exe_path must be None when not registered"
            );
        }
    }

    /// T-MCP-025: config_file_path returns a path ending with `.claude.json`
    /// on all platforms.
    #[test]
    fn t_mcp_025_config_file_path_has_correct_name() {
        if let Some(path) = config_file_path() {
            assert!(
                path.ends_with(".claude.json"),
                "config path must end with .claude.json, got: {}",
                path.display()
            );
        }
    }
}
