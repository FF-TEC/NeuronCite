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

//! Update check handler for the web frontend.
//!
//! Queries the GitHub Releases API to determine whether a newer version of
//! NeuronCite is available. The handler does **not** download or install
//! anything -- it returns the latest version information and a link to the
//! GitHub release page so the user can download the update manually.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::WebState;

/// GitHub repository owner and name used for the Releases API endpoint.
const GITHUB_REPO: &str = "FF-TEC/NeuronCite";

/// Response body for `GET /api/v1/web/check-update`.
#[derive(Debug, Serialize)]
pub struct UpdateCheckResponse {
    /// The version currently running (from `CARGO_PKG_VERSION`).
    pub current_version: String,
    /// The latest version available on GitHub (tag_name with `v` prefix stripped).
    pub latest_version: String,
    /// True when the latest version is strictly newer than the current version.
    pub update_available: bool,
    /// URL to the GitHub release page for the latest version.
    pub release_url: String,
}

/// Subset of the GitHub Releases API response body. Only the fields needed
/// for version comparison and release URL construction are deserialized.
#[derive(Deserialize)]
struct GitHubRelease {
    /// Release tag name (e.g. "v0.2.0" or "0.2.0").
    tag_name: String,
    /// URL to the release page on GitHub.
    html_url: String,
}

/// Strips an optional `v` or `V` prefix from a version tag and parses the
/// remainder as a [`semver::Version`]. Returns `None` if the tag cannot be
/// parsed after prefix stripping.
fn parse_version_tag(tag: &str) -> Option<semver::Version> {
    let stripped = tag
        .strip_prefix('v')
        .or_else(|| tag.strip_prefix('V'))
        .unwrap_or(tag);
    semver::Version::parse(stripped).ok()
}

/// `GET /api/v1/web/check-update`
///
/// Calls the GitHub Releases API to check whether a newer version of
/// NeuronCite is available. Compares the latest release tag against the
/// compile-time `CARGO_PKG_VERSION` using semver ordering.
///
/// Returns `200 OK` with an [`UpdateCheckResponse`] on success.
/// Returns `502 Bad Gateway` if the GitHub API is unreachable, rate-limited,
/// or returns an unexpected response.
pub async fn check_update(
    State(_state): State<Arc<WebState>>,
) -> Result<Json<UpdateCheckResponse>, (StatusCode, Json<serde_json::Value>)> {
    let current_version_str = env!("CARGO_PKG_VERSION");

    let client = reqwest::Client::builder()
        .user_agent(format!("NeuronCite/{current_version_str}"))
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| {
            warn!("failed to build HTTP client for update check: {e}");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to initialize HTTP client",
            )
        })?;

    let response = client
        .get(format!(
            "https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
        ))
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| {
            warn!("GitHub API request failed: {e}");
            error_response(
                StatusCode::BAD_GATEWAY,
                "unable to reach GitHub \u{2014} check your internet connection",
            )
        })?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let hint = match status {
            403 => "GitHub API rate limit exceeded \u{2014} try again in a few minutes",
            404 => "no releases found on GitHub",
            _ => "unexpected response from GitHub API",
        };
        warn!(
            github_status = status,
            "GitHub API returned non-success status"
        );
        return Err(error_response(StatusCode::BAD_GATEWAY, hint));
    }

    let release: GitHubRelease = response.json().await.map_err(|e| {
        warn!("failed to parse GitHub release response: {e}");
        error_response(StatusCode::BAD_GATEWAY, "invalid response from GitHub API")
    })?;

    let current = semver::Version::parse(current_version_str)
        .unwrap_or_else(|_| semver::Version::new(0, 0, 0));
    let latest =
        parse_version_tag(&release.tag_name).unwrap_or_else(|| semver::Version::new(0, 0, 0));

    let update_available = latest > current;

    Ok(Json(UpdateCheckResponse {
        current_version: current_version_str.to_string(),
        latest_version: latest.to_string(),
        update_available,
        release_url: release.html_url,
    }))
}

/// Builds a JSON error response tuple used by the handler.
fn error_response(status: StatusCode, message: &str) -> (StatusCode, Json<serde_json::Value>) {
    (status, Json(serde_json::json!({ "error": message })))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_web_upd_001_parse_version_strips_v_prefix() {
        let v = parse_version_tag("v0.2.0").expect("must parse");
        assert_eq!(v, semver::Version::new(0, 2, 0));
    }

    #[test]
    fn t_web_upd_002_parse_version_strips_uppercase_v() {
        let v = parse_version_tag("V1.0.0").expect("must parse");
        assert_eq!(v, semver::Version::new(1, 0, 0));
    }

    #[test]
    fn t_web_upd_003_parse_version_no_prefix() {
        let v = parse_version_tag("0.1.0").expect("must parse");
        assert_eq!(v, semver::Version::new(0, 1, 0));
    }

    #[test]
    fn t_web_upd_004_parse_version_with_prerelease() {
        let v = parse_version_tag("v1.0.0-beta.1").expect("must parse");
        assert_eq!(v.pre, semver::Prerelease::new("beta.1").unwrap());
    }

    #[test]
    fn t_web_upd_005_parse_version_invalid_returns_none() {
        assert!(parse_version_tag("not-a-version").is_none());
        assert!(parse_version_tag("").is_none());
        assert!(parse_version_tag("v").is_none());
    }

    #[test]
    fn t_web_upd_006_newer_version_detected() {
        let current = semver::Version::parse("0.1.0").unwrap();
        let latest = parse_version_tag("v0.2.0").unwrap();
        assert!(latest > current);
    }

    #[test]
    fn t_web_upd_007_same_version_not_update() {
        let current = semver::Version::parse("0.1.0").unwrap();
        let latest = parse_version_tag("v0.1.0").unwrap();
        assert!(!(latest > current));
    }

    #[test]
    fn t_web_upd_008_prerelease_less_than_release() {
        let pre = semver::Version::parse("1.0.0-beta.1").unwrap();
        let release = semver::Version::parse("1.0.0").unwrap();
        assert!(release > pre);
    }
}
