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

//! Embedded static file serving with SPA fallback routing.
//!
//! The compiled SolidJS frontend from `frontend/dist/` is embedded into the binary
//! at compile time via `rust-embed`. The fallback handler serves files with correct
//! MIME types based on file extension. When a requested path does not match any
//! embedded file, `index.html` is returned to enable client-side routing.

use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use rust_embed::Embed;

/// Embeds the Vite build output directory into the compiled binary. The folder
/// path is relative to the neuroncite-web crate root. During development, this
/// directory is populated by `npm run build` in the frontend/ subdirectory.
/// The `prefix = ""` attribute strips the directory prefix from asset paths so
/// that `index.html` is served at `/` rather than `/frontend/dist/index.html`.
#[derive(Embed)]
#[folder = "frontend/dist"]
#[prefix = ""]
struct FrontendAssets;

/// Axum fallback handler that serves embedded static files or falls back to
/// `index.html` for SPA client-side routing. Called for any request that does
/// not match an API route.
///
/// The handler strips a leading `/` from the URI path, looks up the file in the
/// embedded assets, and returns it with the appropriate Content-Type header
/// derived from the file extension. Cache-Control headers are set based on
/// whether the filename contains a content hash (Vite adds hashes to JS/CSS
/// bundles for cache-busting).
pub async fn serve_embedded(uri: axum::http::Uri) -> Response {
    let path = uri.path().trim_start_matches('/');

    // Attempt to serve the exact requested file from embedded assets.
    if let Some(file) = FrontendAssets::get(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();

        // Vite-produced assets in the assets/ directory have content hashes
        // in their filenames (e.g., assets/index-a1b2c3d4.js). These are
        // safe to cache indefinitely because a content change always produces
        // a new hash and therefore a new filename. Files outside assets/ or
        // without a Vite hash pattern (index.html, favicon.ico) must not be
        // cached immutably: they change between application versions and must
        // be revalidated.
        let cache_control = if is_vite_hashed_asset(path) {
            "public, max-age=31536000, immutable"
        } else {
            "public, max-age=60"
        };

        return (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, mime.as_ref().to_string()),
                (header::CACHE_CONTROL, cache_control.to_string()),
            ],
            file.data.to_vec(),
        )
            .into_response();
    }

    // SPA fallback: serve index.html for any unmatched path. This enables
    // client-side routing where the browser navigates to paths like /settings
    // or /models that have no corresponding server-side file.
    match FrontendAssets::get("index.html") {
        Some(index) => {
            let mime = mime_guess::from_path("index.html").first_or_octet_stream();
            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, mime.as_ref().to_string()),
                    (header::CACHE_CONTROL, "public, max-age=60".to_string()),
                ],
                index.data.to_vec(),
            )
                .into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            "Frontend not built. Run: cd crates/neuroncite-web/frontend && npm run build",
        )
            .into_response(),
    }
}

/// Returns true when `path` matches the Vite content-hashed asset pattern.
///
/// Vite places all bundled JS/CSS/font files under `assets/` with a
/// `-XXXXXXXX.ext` suffix where the X characters are exactly 8 lowercase
/// hexadecimal digits derived from the file content hash. Any path that
/// matches this pattern is safe to cache immutably (max-age=31536000) because
/// a content change always produces a different hash and therefore a different
/// URL.
///
/// Paths outside `assets/`, paths without a dash-separated segment before
/// the extension, and paths whose hash segment is not exactly 8 hex characters
/// are treated as mutable and receive a short cache TTL (max-age=60).
fn is_vite_hashed_asset(path: &str) -> bool {
    // Must be under the assets/ directory emitted by Vite.
    if !path.starts_with("assets/") {
        return false;
    }

    // Extract the filename component (last path segment after the final '/').
    let filename = match path.rfind('/') {
        Some(pos) => &path[pos + 1..],
        None => path,
    };

    // Must have an extension preceded by a Vite hash segment (-XXXXXXXX).
    // Find the last '.' which separates the extension.
    let dot_pos = match filename.rfind('.') {
        Some(pos) if pos > 0 => pos,
        _ => return false,
    };

    let stem = &filename[..dot_pos]; // e.g. "index-a1b2c3d4"

    // Find the last '-' which separates the base name from the hash.
    let dash_pos = match stem.rfind('-') {
        Some(pos) => pos,
        None => return false,
    };

    let hash_candidate = &stem[dash_pos + 1..]; // e.g. "a1b2c3d4"

    // Vite uses exactly 8 lowercase hexadecimal characters as the content hash.
    hash_candidate.len() == 8
        && hash_candidate
            .chars()
            .all(|c| matches!(c, '0'..='9' | 'a'..='f'))
}
