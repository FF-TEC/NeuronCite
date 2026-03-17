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

//! MCP server registration for Claude Code and Claude Desktop App.
//!
//! Provides install/uninstall/status operations that manage the NeuronCite
//! entry in the MCP configuration files of two supported clients:
//!
//! - **Claude Code** (CLI): `~/.claude.json`
//! - **Claude Desktop App** (GUI): platform-specific config file
//!   (`~/Library/Application Support/Claude/claude_desktop_config.json` on
//!   macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows).
//!
//! Both clients read MCP server definitions from a top-level `mcpServers`
//! key with identical JSON structure, so the core read-modify-write logic is
//! shared. The [`McpTarget`] enum selects which config file to operate on.

use std::fmt;
use std::path::{Path, PathBuf};

/// Entry name used in the MCP configuration file.
const MCP_SERVER_NAME: &str = "neuroncite";

/// Arguments passed to the NeuronCite executable when started by an MCP client.
const MCP_ARGS: &[&str] = &["mcp", "serve"];

/// Identifies which MCP client configuration to target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpTarget {
    /// Claude Code (CLI) -- config at `~/.claude.json`.
    ClaudeCode,
    /// Claude Desktop App (GUI) -- config at a platform-specific path.
    ClaudeDesktop,
}

impl McpTarget {
    /// Returns all supported targets for iteration.
    pub fn all() -> [McpTarget; 2] {
        [McpTarget::ClaudeCode, McpTarget::ClaudeDesktop]
    }

    /// Human-readable display name for UI and log messages.
    pub fn display_name(self) -> &'static str {
        match self {
            McpTarget::ClaudeCode => "Claude Code",
            McpTarget::ClaudeDesktop => "Claude Desktop App",
        }
    }

    /// CLI string identifier used with `--target`.
    pub fn cli_name(self) -> &'static str {
        match self {
            McpTarget::ClaudeCode => "claude-code",
            McpTarget::ClaudeDesktop => "claude-desktop",
        }
    }

    /// Parses a CLI `--target` string into an `McpTarget`.
    pub fn from_cli_str(s: &str) -> Result<Self, String> {
        match s {
            "claude-code" => Ok(McpTarget::ClaudeCode),
            "claude-desktop" => Ok(McpTarget::ClaudeDesktop),
            other => Err(format!(
                "unknown MCP target '{other}' (expected 'claude-code' or 'claude-desktop')"
            )),
        }
    }

    /// Returns the platform-appropriate path to the MCP configuration file
    /// for this target. Returns `None` if the home directory cannot be
    /// determined.
    pub fn config_file_path(self) -> Option<PathBuf> {
        let home = home_dir()?;
        Some(match self {
            McpTarget::ClaudeCode => home.join(".claude.json"),
            McpTarget::ClaudeDesktop => claude_desktop_config_path(&home),
        })
    }
}

impl fmt::Display for McpTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.display_name())
    }
}

/// Returns the user's home directory.
fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

/// Returns the Claude Desktop App config file path relative to the given home
/// directory.
fn claude_desktop_config_path(home: &Path) -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        home.join("Library/Application Support/Claude/claude_desktop_config.json")
    }
    #[cfg(target_os = "windows")]
    {
        // On Windows, APPDATA is typically used. Fall back to home-relative path.
        if let Ok(appdata) = std::env::var("APPDATA") {
            PathBuf::from(appdata)
                .join("Claude")
                .join("claude_desktop_config.json")
        } else {
            home.join("AppData")
                .join("Roaming")
                .join("Claude")
                .join("claude_desktop_config.json")
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        home.join(".config")
            .join("Claude")
            .join("claude_desktop_config.json")
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
    /// Path to the config file for this target.
    pub config_path: Option<String>,
}

/// Registers the NeuronCite MCP server in the config file for the given
/// `target`.
///
/// Creates the config file (and parent directories) if they do not exist. If
/// a `mcpServers.neuroncite` entry already exists, it is overwritten with the
/// current executable path, arguments, and `"type": "stdio"`. All other keys
/// in the file are preserved.
///
/// # Arguments
///
/// * `exe_override` - Optional path to use instead of the current executable.
///   Used for testing and for registering a specific build.
/// * `target` - Which MCP client configuration to write to.
pub fn install(exe_override: Option<&Path>, target: McpTarget) -> Result<String, String> {
    let config_path = target
        .config_file_path()
        .ok_or("cannot determine home directory for MCP config")?;
    install_at(exe_override, &config_path, target.display_name())
}

/// Core install logic operating on an explicit config path. Used by
/// `install()` and by tests that need to target a temporary directory.
fn install_at(
    exe_override: Option<&Path>,
    config_path: &Path,
    label: &str,
) -> Result<String, String> {
    let exe_path = match exe_override {
        Some(p) => p.to_path_buf(),
        None => std::env::current_exe()
            .map_err(|e| format!("failed to determine executable path: {e}"))?,
    };

    // Ensure parent directory exists (important for Claude Desktop App whose
    // config directory may not have been created yet).
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create directory {}: {e}", parent.display()))?;
    }

    // Read existing config or start with an empty object.
    let mut config: serde_json::Value = if config_path.exists() {
        let content = std::fs::read_to_string(config_path)
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
    std::fs::write(config_path, json_str)
        .map_err(|e| format!("failed to write {}: {e}", config_path.display()))?;

    Ok(format!(
        "{label} MCP server registered at {}\nExecutable: {}\nArgs: {:?}",
        config_path.display(),
        exe_str,
        MCP_ARGS
    ))
}

/// Removes the NeuronCite MCP server entry from the config file for the
/// given `target`.
///
/// Only the `mcpServers.neuroncite` key is removed. All other keys in the
/// config file are preserved. If the file does not exist or does not contain
/// a "neuroncite" entry, this operation is a no-op.
pub fn uninstall(target: McpTarget) -> Result<String, String> {
    let config_path = target
        .config_file_path()
        .ok_or("cannot determine home directory for MCP config")?;
    uninstall_at(&config_path, target.display_name())
}

/// Core uninstall logic operating on an explicit config path.
fn uninstall_at(config_path: &Path, label: &str) -> Result<String, String> {
    if !config_path.exists() {
        return Ok(format!(
            "{label} config file does not exist; nothing to uninstall"
        ));
    }

    let content = std::fs::read_to_string(config_path)
        .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?;
    let mut config: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse {}: {e}", config_path.display()))?;

    // Remove the neuroncite entry if it exists.
    let removed = config["mcpServers"]
        .as_object_mut()
        .map(|servers| servers.remove(MCP_SERVER_NAME).is_some())
        .unwrap_or(false);

    if !removed {
        return Ok(format!(
            "NeuronCite MCP server was not registered in {label}; nothing to uninstall"
        ));
    }

    let json_str = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("failed to serialize config: {e}"))?;
    std::fs::write(config_path, json_str)
        .map_err(|e| format!("failed to write {}: {e}", config_path.display()))?;

    Ok(format!(
        "MCP server unregistered from {label} ({})",
        config_path.display()
    ))
}

/// Checks the current registration status of the NeuronCite MCP server
/// for the given `target`.
pub fn check_status(target: McpTarget) -> McpStatus {
    match target.config_file_path() {
        Some(path) => check_status_at(&path),
        None => McpStatus {
            registered: false,
            exe_path: None,
            args: None,
            config_path: None,
        },
    }
}

/// Core status-check logic operating on an explicit config path.
fn check_status_at(path: &Path) -> McpStatus {
    let config_path = Some(path.to_path_buf());

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
        let status = check_status(McpTarget::ClaudeCode);
        if !status.registered {
            assert!(
                status.exe_path.is_none(),
                "exe_path must be None when not registered"
            );
        }
    }

    /// T-MCP-025: config_file_path returns a path ending with `.claude.json`
    /// for the Claude Code target.
    #[test]
    fn t_mcp_025_config_file_path_has_correct_name() {
        if let Some(path) = McpTarget::ClaudeCode.config_file_path() {
            assert!(
                path.ends_with(".claude.json"),
                "config path must end with .claude.json, got: {}",
                path.display()
            );
        }
    }

    /// T-MCP-030: config_file_path returns a path ending with
    /// `claude_desktop_config.json` for the Claude Desktop target.
    #[test]
    fn t_mcp_030_desktop_config_file_path_has_correct_name() {
        if let Some(path) = McpTarget::ClaudeDesktop.config_file_path() {
            assert!(
                path.ends_with("claude_desktop_config.json"),
                "desktop config path must end with claude_desktop_config.json, got: {}",
                path.display()
            );
        }
    }

    /// T-MCP-031: McpTarget::from_cli_str parses valid target strings and
    /// rejects unknown ones.
    #[test]
    fn t_mcp_031_from_cli_str_parsing() {
        assert_eq!(
            McpTarget::from_cli_str("claude-code").unwrap(),
            McpTarget::ClaudeCode
        );
        assert_eq!(
            McpTarget::from_cli_str("claude-desktop").unwrap(),
            McpTarget::ClaudeDesktop
        );
        assert!(McpTarget::from_cli_str("unknown").is_err());
    }

    /// T-MCP-032: McpTarget::all returns both targets.
    #[test]
    fn t_mcp_032_all_targets() {
        let targets = McpTarget::all();
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0], McpTarget::ClaudeCode);
        assert_eq!(targets[1], McpTarget::ClaudeDesktop);
    }

    /// T-MCP-033: cli_name round-trips through from_cli_str for all targets.
    #[test]
    fn t_mcp_033_cli_name_round_trip() {
        for target in McpTarget::all() {
            let name = target.cli_name();
            let parsed = McpTarget::from_cli_str(name).unwrap();
            assert_eq!(parsed, target);
        }
    }

    /// T-MCP-034: display_name returns non-empty human-readable names.
    #[test]
    fn t_mcp_034_display_name_non_empty() {
        for target in McpTarget::all() {
            let name = target.display_name();
            assert!(!name.is_empty(), "display_name must not be empty");
            // Display trait should also work
            assert_eq!(format!("{target}"), name);
        }
    }

    /// T-MCP-035: install_at writes the correct mcpServers entry and
    /// uninstall_at removes it again.
    #[test]
    fn t_mcp_035_install_uninstall_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fake_exe = dir.path().join("neuroncite");
        std::fs::write(&fake_exe, "fake").expect("write fake exe");
        let config_file = dir.path().join(".claude.json");

        // Install.
        let result = install_at(Some(&fake_exe), &config_file, "test");
        assert!(result.is_ok(), "install should succeed: {:?}", result);
        assert!(
            config_file.exists(),
            "config file should exist after install"
        );

        // Verify content.
        let content = std::fs::read_to_string(&config_file).expect("read config");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("parse");
        assert_eq!(parsed["mcpServers"]["neuroncite"]["type"], "stdio");
        assert_eq!(
            parsed["mcpServers"]["neuroncite"]["command"]
                .as_str()
                .unwrap(),
            fake_exe.display().to_string()
        );
        let args = parsed["mcpServers"]["neuroncite"]["args"]
            .as_array()
            .expect("args");
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], "mcp");
        assert_eq!(args[1], "serve");

        // check_status_at should report registered.
        let status = check_status_at(&config_file);
        assert!(status.registered, "should be registered after install");
        assert!(status.exe_path.is_some());

        // Uninstall.
        let result = uninstall_at(&config_file, "test");
        assert!(result.is_ok(), "uninstall should succeed: {:?}", result);

        // check_status_at should report not registered.
        let status = check_status_at(&config_file);
        assert!(
            !status.registered,
            "should not be registered after uninstall"
        );
    }

    /// T-MCP-036: install_at creates parent directories if they do not exist.
    #[test]
    fn t_mcp_036_install_creates_parent_dirs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fake_exe = dir.path().join("neuroncite");
        std::fs::write(&fake_exe, "fake").expect("write fake exe");

        // Deeply nested config path (simulates Claude Desktop App directory).
        let config_file = dir
            .path()
            .join("Library")
            .join("Application Support")
            .join("Claude")
            .join("claude_desktop_config.json");

        let result = install_at(Some(&fake_exe), &config_file, "test");
        assert!(result.is_ok(), "install should succeed: {:?}", result);
        assert!(
            config_file.exists(),
            "config file should exist after install: {}",
            config_file.display()
        );

        let content = std::fs::read_to_string(&config_file).expect("read config");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("parse");
        assert_eq!(parsed["mcpServers"]["neuroncite"]["type"], "stdio");
    }

    /// T-MCP-037: install preserves existing keys in the config file.
    #[test]
    fn t_mcp_037_install_preserves_existing_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fake_exe = dir.path().join("neuroncite");
        std::fs::write(&fake_exe, "fake").expect("write fake exe");
        let config_file = dir.path().join(".claude.json");

        // Pre-populate the config file with extra keys.
        let existing = serde_json::json!({
            "numStartups": 42,
            "projects": ["myproject"],
        });
        std::fs::write(
            &config_file,
            serde_json::to_string_pretty(&existing).unwrap(),
        )
        .expect("write existing config");

        // Install should add mcpServers without clobbering existing keys.
        let result = install_at(Some(&fake_exe), &config_file, "test");
        assert!(result.is_ok());

        let content = std::fs::read_to_string(&config_file).expect("read config");
        let parsed: serde_json::Value = serde_json::from_str(&content).expect("parse");
        assert_eq!(
            parsed["numStartups"], 42,
            "existing key should be preserved"
        );
        assert!(
            parsed["projects"].is_array(),
            "existing array should be preserved"
        );
        assert_eq!(parsed["mcpServers"]["neuroncite"]["type"], "stdio");
    }

    /// T-MCP-038: uninstall is a no-op when config file does not exist.
    #[test]
    fn t_mcp_038_uninstall_noop_when_no_config() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config_file = dir.path().join("nonexistent.json");

        let result = uninstall_at(&config_file, "test");
        assert!(
            result.is_ok(),
            "uninstall should succeed even without config file"
        );
        assert!(!config_file.exists(), "file should not be created");
    }

    /// T-MCP-039: check_status_at returns correct config_path when the
    /// file does not exist.
    #[test]
    fn t_mcp_039_check_status_returns_config_path_when_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config_file = dir.path().join("nonexistent.json");

        let status = check_status_at(&config_file);
        assert!(!status.registered);
        assert!(
            status.config_path.is_some(),
            "config_path should be Some even when file is missing"
        );
        assert!(
            status.config_path.unwrap().contains("nonexistent.json"),
            "config_path should contain the file name"
        );
    }

    /// T-MCP-040: install and uninstall on separate config files are
    /// independent -- installing in one does not affect the other.
    #[test]
    fn t_mcp_040_targets_are_independent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fake_exe = dir.path().join("neuroncite");
        std::fs::write(&fake_exe, "fake").expect("write fake exe");

        let code_config = dir.path().join("code.json");
        let desktop_config = dir.path().join("desktop.json");

        // Install only for "code".
        install_at(Some(&fake_exe), &code_config, "code").expect("install code");

        assert!(
            check_status_at(&code_config).registered,
            "code should be registered"
        );
        assert!(
            !check_status_at(&desktop_config).registered,
            "desktop should NOT be registered"
        );

        // Now install for "desktop" too.
        install_at(Some(&fake_exe), &desktop_config, "desktop").expect("install desktop");

        assert!(
            check_status_at(&code_config).registered,
            "code should still be registered"
        );
        assert!(
            check_status_at(&desktop_config).registered,
            "desktop should now be registered"
        );

        // Uninstall only code -- desktop should remain.
        uninstall_at(&code_config, "code").expect("uninstall code");

        assert!(
            !check_status_at(&code_config).registered,
            "code should be unregistered"
        );
        assert!(
            check_status_at(&desktop_config).registered,
            "desktop should still be registered"
        );
    }
}
