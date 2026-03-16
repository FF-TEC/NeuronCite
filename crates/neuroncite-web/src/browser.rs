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

//! Cross-platform browser launcher for the web UI.
//!
//! Opens the default browser to the server URL after the axum server has
//! bound to its TCP listener. Includes a short delay to ensure the server
//! is ready to accept connections before the browser connects.

/// Opens the default browser to the given URL. Prints the URL to stdout
/// as a fallback for environments where the browser cannot be launched
/// automatically (headless servers, SSH sessions).
///
/// The function spawns a background tokio task with a 200ms delay to give
/// the server time to start accepting connections before the browser sends
/// its first request.
pub fn open_browser(url: String) {
    tokio::spawn(async move {
        // Short delay to ensure the axum server is fully ready to accept
        // connections. Without this delay, the browser may connect before
        // the server's accept loop is running, causing a connection refused
        // error on the initial page load.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        match opener::open_browser(&url) {
            Ok(()) => {
                tracing::info!(url = %url, "opened default browser");
            }
            Err(e) => {
                tracing::warn!(
                    url = %url,
                    error = %e,
                    "failed to open browser automatically"
                );
                eprintln!();
                eprintln!("  Could not open browser automatically.");
                eprintln!("  Open this URL manually: {url}");
                eprintln!();
            }
        }
    });
}
