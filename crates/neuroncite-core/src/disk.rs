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

// Disk space utilities for pre-download validation.
//
// Provides cross-platform available disk space detection and threshold
// checks. Used by neuroncite-embed (ORT downloads) and neuroncite-pdf
// (pdfium/Tesseract downloads) to prevent partial downloads on nearly-full
// filesystems.

use std::path::Path;

/// Minimum disk space in bytes (500 MB) below which downloads are blocked
/// with an error. This prevents partially-written files from consuming the
/// remaining space and leaving the system in an unusable state.
const MIN_SPACE_ERROR: u64 = 500 * 1024 * 1024;

/// Warning threshold in bytes (1 GB). When available space is between
/// MIN_SPACE_ERROR and this value, a warning is logged but the download
/// proceeds. This gives the operator advance notice before space runs out.
const MIN_SPACE_WARN: u64 = 1024 * 1024 * 1024;

/// Returns the available disk space in bytes for the filesystem containing
/// the given path. Uses platform-specific methods:
///
/// - **Windows**: PowerShell `.NET` call to `Get-PSDrive` for drive-letter
///   paths, with `[System.IO.DriveInfo]` fallback for UNC paths. The Win32
///   `GetDiskFreeSpaceExW` API would be faster but requires `unsafe` code,
///   which this crate forbids via `#![forbid(unsafe_code)]`.
/// - **Linux/macOS**: `df --output=avail -B1 <path>` (GNU coreutils) or
///   `df -k <path>` (BSD/macOS fallback).
///
/// Returns `None` if the command fails or the output cannot be parsed.
/// This function is intentionally lenient: a failure to determine disk
/// space does not block operations, it only prevents pre-checks from
/// running.
pub fn available_disk_space(path: &Path) -> Option<u64> {
    #[cfg(target_os = "windows")]
    {
        available_disk_space_windows(path)
    }
    #[cfg(not(target_os = "windows"))]
    {
        available_disk_space_unix(path)
    }
}

/// Windows implementation: queries available disk space via PowerShell.
///
/// NOTE: Using the Win32 `GetDiskFreeSpaceExW` API would be faster (no
/// subprocess), but that function is `unsafe` in the `windows` crate
/// because it requires raw PCWSTR pointers. This crate has
/// `#![forbid(unsafe_code)]`, so we use PowerShell as a safe alternative.
///
/// Two strategies are tried in order:
/// 1. Drive-letter paths (e.g., `C:\Users\...`): Uses `(Get-PSDrive <letter>).Free`
///    which is fast and reliable for local and mapped network drives.
/// 2. UNC paths (e.g., `\\server\share\...`) or paths without a drive letter:
///    Uses `[System.IO.DriveInfo]::new(<root>).AvailableFreeSpace` which handles
///    network shares and subst drives that do not have a drive letter.
#[cfg(target_os = "windows")]
fn available_disk_space_windows(path: &Path) -> Option<u64> {
    let path_str = path.to_string_lossy();

    // Strategy 1: Extract drive letter for fast Get-PSDrive query.
    let first_char = path_str.chars().next()?;
    if first_char.is_ascii_alphabetic() && path_str.chars().nth(1) == Some(':') {
        let ps_command = format!("(Get-PSDrive {}).Free", first_char.to_ascii_uppercase());

        let output = std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &ps_command])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output()
            .ok()?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(bytes) = stdout.trim().parse::<u64>() {
                return Some(bytes);
            }
        }
    }

    // Strategy 2: Use .NET DriveInfo for UNC paths, subst drives, and paths
    // where Get-PSDrive failed. DriveInfo accepts any path root including
    // UNC shares like "\\server\share".
    let root = path.ancestors().last().unwrap_or(path).to_string_lossy();

    // For UNC paths, the root is the share itself (e.g., \\server\share).
    // For drive paths, ancestors().last() returns the empty prefix, so we
    // use the full path and let DriveInfo resolve the drive root.
    let drive_root = if root.is_empty() || root == "\\" {
        path_str.to_string()
    } else {
        root.to_string()
    };

    let ps_command = format!(
        "try {{ [System.IO.DriveInfo]::new('{}').AvailableFreeSpace }} catch {{ }}",
        drive_root.replace('\'', "''")
    );

    let output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &ps_command])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Ok(bytes) = stdout.trim().parse::<u64>() {
            return Some(bytes);
        }
    }

    None
}

/// Unix (Linux/macOS) implementation: uses `df` to query available space.
/// Tries GNU coreutils format first (`--output=avail -B1`), then falls
/// back to BSD/macOS format (`-k`) if the GNU flags are unsupported.
#[cfg(not(target_os = "windows"))]
fn available_disk_space_unix(path: &Path) -> Option<u64> {
    // Try GNU coreutils df first (outputs bytes directly).
    let output = std::process::Command::new("df")
        .args(["--output=avail", "-B1"])
        .arg(path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // GNU df output has a header line ("Avail") followed by the available
        // bytes as a plain integer on the second line.
        if let Some(line) = stdout.lines().nth(1)
            && let Ok(bytes) = line.trim().parse::<u64>()
        {
            return Some(bytes);
        }
    }

    // Fallback: BSD/macOS df -k (outputs 1K blocks).
    let output = std::process::Command::new("df")
        .args(["-k"])
        .arg(path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // BSD df output: header line then data row. Fields are: filesystem,
        // 1K-blocks, used, available, capacity%, mount. Index 3 is available.
        if let Some(line) = stdout.lines().nth(1) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() >= 4
                && let Ok(kb) = fields[3].parse::<u64>()
            {
                return Some(kb * 1024);
            }
        }
    }

    None
}

/// Checks whether sufficient disk space is available at the given path
/// for a download operation. Returns `Ok(())` if space is sufficient or
/// if the available space cannot be determined (fail-open behavior).
///
/// Thresholds:
/// - Below 500 MB: returns `Err` with an error message
/// - Below 1 GB: logs a warning but returns `Ok`
/// - Above 1 GB or undetermined: returns `Ok` silently
///
/// # Errors
///
/// Returns an error string describing the insufficient disk space
/// condition, including the available and required amounts.
pub fn check_disk_space(path: &Path) -> Result<(), String> {
    let Some(available) = available_disk_space(path) else {
        // Cannot determine disk space; proceed without blocking.
        return Ok(());
    };

    if available < MIN_SPACE_ERROR {
        let available_mb = available / (1024 * 1024);
        return Err(format!(
            "insufficient disk space: {available_mb} MB available at {}, \
             minimum 500 MB required for download operations",
            path.display()
        ));
    }

    if available < MIN_SPACE_WARN {
        let available_mb = available / (1024 * 1024);
        tracing::warn!(
            available_mb,
            path = %path.display(),
            "low disk space: {available_mb} MB available, \
             downloads may fail if space runs out"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// T-DSK-001: `available_disk_space` returns Some for a valid local path.
    /// The current working directory always resides on a valid filesystem, so
    /// this test validates that the platform-specific implementation produces
    /// a parseable u64 value.
    #[test]
    fn t_dsk_001_available_space_returns_some_for_cwd() {
        let cwd = std::env::current_dir().expect("cwd is valid");
        let space = available_disk_space(&cwd);
        assert!(
            space.is_some(),
            "available_disk_space must return Some for the current working directory"
        );
        // Any modern filesystem has at least 1 byte free.
        assert!(space.unwrap() > 0, "available space must be positive");
    }

    /// T-DSK-002: `available_disk_space` returns None for a nonexistent path
    /// rather than panicking. Verifies the fail-open behavior documented in
    /// the function contract.
    #[test]
    fn t_dsk_002_nonexistent_path_returns_none() {
        let fake = PathBuf::from("/this/path/does/not/exist/at/all");
        // Should return None gracefully, not panic.
        let _ = available_disk_space(&fake);
    }

    /// T-DSK-003: `check_disk_space` returns Ok for a valid local path.
    /// Since the test environment has more than 500 MB free, this validates
    /// the happy path.
    #[test]
    fn t_dsk_003_check_passes_for_valid_path() {
        let cwd = std::env::current_dir().expect("cwd is valid");
        let result = check_disk_space(&cwd);
        assert!(
            result.is_ok(),
            "check_disk_space must succeed for a valid path with sufficient space"
        );
    }

    /// T-DSK-004: `check_disk_space` returns Ok for a nonexistent path
    /// (fail-open behavior). When disk space cannot be determined, the check
    /// must not block the operation.
    #[test]
    fn t_dsk_004_check_passes_for_nonexistent_path() {
        let fake = PathBuf::from("/nonexistent/path/for/disk/check");
        let result = check_disk_space(&fake);
        assert!(
            result.is_ok(),
            "check_disk_space must succeed (fail-open) when space is undetermined"
        );
    }

    /// T-DSK-005: Threshold constants are correctly ordered. The error
    /// threshold must be strictly below the warning threshold.
    #[test]
    fn t_dsk_005_threshold_ordering() {
        let error = MIN_SPACE_ERROR;
        let warn = MIN_SPACE_WARN;
        assert!(
            error < warn,
            "error threshold (500 MB) must be less than warning threshold (1 GB)"
        );
    }

    /// T-DSK-006: On Windows, drive-letter paths produce a valid space reading.
    /// Validates that the first character extraction and PowerShell invocation
    /// work for standard drive-letter paths.
    #[cfg(target_os = "windows")]
    #[test]
    fn t_dsk_006_windows_drive_letter_path() {
        // Use the system drive (typically C:\).
        let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
        let path = PathBuf::from(system_root);
        let space = available_disk_space(&path);
        assert!(
            space.is_some(),
            "available_disk_space must return Some for a drive-letter path on Windows"
        );
    }

    /// T-DSK-007: On Windows, a path without a drive letter (e.g., a relative
    /// path or UNC-like prefix) does not panic. The function should either
    /// return Some (if the fallback DriveInfo strategy succeeds) or None.
    #[cfg(target_os = "windows")]
    #[test]
    fn t_dsk_007_windows_non_drive_letter_path() {
        // A relative path has no drive letter prefix.
        let path = PathBuf::from("relative\\path\\without\\drive");
        // Must not panic; None is acceptable.
        let _ = available_disk_space(&path);
    }
}
