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

// Centralized UNIX timestamp helper for the NeuronCite workspace.
//
// This module provides a single `unix_timestamp_secs()` function that returns
// the current time as seconds since the UNIX epoch. All crates in the workspace
// import this function instead of duplicating the SystemTime::now() pattern
// with `.expect("system clock before UNIX epoch")`.
//
// The function uses `unwrap_or(0)` as a fallback for the theoretically impossible
// case where the system clock is before the UNIX epoch (1970-01-01T00:00:00Z).
// Returning 0 is safe because all callers use the timestamp for "created_at" or
// "updated_at" columns where 0 is a recognizable sentinel that does not cause
// crashes or data corruption.

use std::time::{SystemTime, UNIX_EPOCH};

/// Returns the current system time as seconds since the UNIX epoch (1970-01-01).
///
/// Falls back to 0 if the system clock is before the UNIX epoch. This is a
/// theoretical impossibility on all supported platforms (Windows, Linux, macOS),
/// but using `unwrap_or(0)` avoids a panic in the event of a misconfigured
/// system clock or virtualization artifact.
pub fn unix_timestamp_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(std::time::Duration::ZERO)
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-TIME-001: unix_timestamp_secs returns a positive value on any system
    /// with a properly configured clock.
    #[test]
    fn t_time_001_returns_positive_timestamp() {
        let ts = unix_timestamp_secs();
        // Any system clock after 2024-01-01 yields a value > 1_704_067_200.
        assert!(ts > 1_704_067_200, "timestamp should be after 2024: {ts}");
    }
}
