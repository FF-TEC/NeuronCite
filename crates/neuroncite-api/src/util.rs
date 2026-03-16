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

//! Shared utility functions for request validation in API handlers.
//!
//! This module provides defense-in-depth validation helpers that are used
//! across multiple handlers. The path containment check prevents handlers
//! that accept user-supplied filesystem paths (annotation, citation, export)
//! from reading or writing outside the operator's intended directory scope.

use std::path::{Path, PathBuf};

use neuroncite_core::AppConfig;

use crate::error::ApiError;

/// Validates that a user-supplied filesystem path is permitted by the server's
/// path access policy. This prevents directory traversal attacks where an
/// attacker supplies paths like `/etc/shadow` or `C:\Windows\System32` to
/// read system files, or `/root/.ssh/` to write to sensitive locations.
///
/// The validation logic mirrors the `validate_browse_path` function in the
/// web browse handler but returns `ApiError` instead of a raw status tuple.
///
/// Access policy:
///
/// - When `allowed_browse_roots` is empty AND the server is bound to a
///   loopback address (127.0.0.1, ::1, localhost), all paths are permitted.
///   This preserves backward compatibility for localhost-only deployments
///   where the operator controls the machine.
///
/// - When `allowed_browse_roots` is empty AND the server is bound to a
///   non-loopback address, the path is rejected. Without an explicit
///   allowlist, binding to a network interface would allow any network
///   client to read/write arbitrary filesystem locations.
///
/// - When `allowed_browse_roots` is non-empty, the canonical form of the
///   supplied path must be a descendant of at least one allowed root.
///   Both the path and the roots are canonicalized to resolve symlinks
///   and `..` components before the containment check.
///
/// # Arguments
///
/// * `path` - The user-supplied path to validate (file or directory).
/// * `config` - The application configuration containing `allowed_browse_roots`
///   and `bind_address`.
///
/// # Returns
///
/// On success, returns the canonicalized `PathBuf` that passed the containment
/// check. Callers that perform subsequent file I/O should use this returned
/// path instead of the original user-supplied path to prevent time-of-check
/// to time-of-use (TOCTOU) attacks where a symlink is swapped between the
/// containment check and the actual file operation.
///
/// For the loopback fast-path (empty allowlist, loopback bind), the original
/// path is returned as-is because no containment restriction applies.
///
/// # Errors
///
/// Returns `ApiError::BadRequest` when the path falls outside the allowlist
/// or when canonicalization fails (e.g., the path does not exist).
pub fn validate_path_access(path: &Path, config: &AppConfig) -> Result<PathBuf, ApiError> {
    if config.allowed_browse_roots.is_empty() {
        if neuroncite_core::config::is_loopback(&config.bind_address) {
            // Loopback bind with empty allowlist: all paths are permitted
            // (backward-compatible default for localhost-only deployments).
            // Return the original path since no canonicalization is needed.
            return Ok(path.to_path_buf());
        }
        // Non-loopback bind with empty allowlist: reject all path-based
        // operations to prevent network clients from accessing arbitrary
        // filesystem locations.
        return Err(ApiError::BadRequest {
            reason: "path-based operations are restricted when no allowed_browse_roots \
                     are configured and the server is bound to a non-loopback address"
                .to_string(),
        });
    }

    // Canonicalize the supplied path to resolve symlinks and ".." components.
    // If canonicalization fails (path does not exist), reject the request
    // because containment cannot be verified without a canonical path.
    let canonical = std::fs::canonicalize(path).map_err(|_| ApiError::BadRequest {
        reason: format!(
            "cannot resolve path for containment check: {}",
            path.display()
        ),
    })?;

    // Strip the Windows \\?\ prefix that std::fs::canonicalize adds on Windows.
    // This prefix is valid for Win32 API calls but breaks string prefix matching
    // against root paths stored without the prefix in the configuration file.
    let canonical = strip_windows_prefix(&canonical);

    for root in &config.allowed_browse_roots {
        let root_path = PathBuf::from(root);
        if let Ok(canonical_root) = std::fs::canonicalize(&root_path) {
            let canonical_root = strip_windows_prefix(&canonical_root);
            if canonical.starts_with(&canonical_root) {
                // Return the canonicalized path so callers use the validated
                // path for subsequent I/O, preventing TOCTOU symlink swaps.
                return Ok(canonical);
            }
        }
    }

    Err(ApiError::BadRequest {
        reason: format!("path is outside the allowed roots: {}", path.display()),
    })
}

/// Strips the Windows extended-length path prefix (`\\?\`) from a PathBuf.
/// On non-Windows platforms this is a no-op. On Windows, the prefix is an
/// internal API detail added by `std::fs::canonicalize` that breaks prefix
/// matching when one path has the prefix and the other does not.
fn strip_windows_prefix(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if let Some(stripped) = s.strip_prefix(r"\\?\") {
        PathBuf::from(stripped)
    } else {
        path.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-UTIL-001: Loopback bind with empty allowlist permits all paths.
    /// The returned PathBuf matches the input path (no canonicalization
    /// is performed in the loopback fast-path).
    #[test]
    fn t_util_001_loopback_empty_roots_permits_all() {
        let config = AppConfig {
            bind_address: "127.0.0.1".into(),
            allowed_browse_roots: Vec::new(),
            ..AppConfig::default()
        };
        let input = Path::new("/some/arbitrary/path");
        let result = validate_path_access(input, &config);
        assert!(
            result.is_ok(),
            "loopback bind with empty roots must permit all paths"
        );
        assert_eq!(
            result.unwrap(),
            input.to_path_buf(),
            "loopback fast-path must return the original path"
        );
    }

    /// T-UTIL-002: Non-loopback bind with empty allowlist rejects all paths.
    #[test]
    fn t_util_002_nonloopback_empty_roots_rejects() {
        let config = AppConfig {
            bind_address: "0.0.0.0".into(),
            allowed_browse_roots: Vec::new(),
            ..AppConfig::default()
        };
        let result = validate_path_access(Path::new("/some/path"), &config);
        assert!(
            result.is_err(),
            "non-loopback bind with empty roots must reject paths"
        );
    }

    /// T-UTIL-003: Path within an allowed root is permitted. The returned
    /// PathBuf is the canonicalized form of the input path.
    #[test]
    fn t_util_003_path_within_allowed_root() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let subdir = tmp.path().join("subdir");
        std::fs::create_dir_all(&subdir).expect("create subdir");

        let config = AppConfig {
            bind_address: "0.0.0.0".into(),
            allowed_browse_roots: vec![tmp.path().to_string_lossy().to_string()],
            ..AppConfig::default()
        };
        let result = validate_path_access(&subdir, &config);
        assert!(
            result.is_ok(),
            "path within allowed root must be permitted: {:?}",
            result.err()
        );
        // The returned path must be a canonicalized PathBuf (not the raw input).
        let canonical = result.unwrap();
        assert!(
            canonical.is_absolute(),
            "returned path must be absolute: {:?}",
            canonical
        );
    }

    /// T-UTIL-004: Path outside all allowed roots is rejected.
    #[test]
    fn t_util_004_path_outside_roots_rejected() {
        let allowed = tempfile::tempdir().expect("create allowed dir");
        let outside = tempfile::tempdir().expect("create outside dir");

        let config = AppConfig {
            bind_address: "0.0.0.0".into(),
            allowed_browse_roots: vec![allowed.path().to_string_lossy().to_string()],
            ..AppConfig::default()
        };
        let result = validate_path_access(outside.path(), &config);
        assert!(
            result.is_err(),
            "path outside allowed roots must be rejected"
        );
    }

    /// T-UTIL-005: Nonexistent path is rejected when roots are configured.
    #[test]
    fn t_util_005_nonexistent_path_rejected() {
        let allowed = tempfile::tempdir().expect("create allowed dir");
        let config = AppConfig {
            bind_address: "0.0.0.0".into(),
            allowed_browse_roots: vec![allowed.path().to_string_lossy().to_string()],
            ..AppConfig::default()
        };
        let result = validate_path_access(
            Path::new("/this/path/definitely/does/not/exist/anywhere"),
            &config,
        );
        assert!(
            result.is_err(),
            "nonexistent path must be rejected when roots are configured"
        );
    }
}
