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

// Security response headers middleware for the web server.
//
// Adds standard security headers to all HTTP responses served by the web
// router. These headers instruct browsers to apply additional protections:
//
// - X-Content-Type-Options: nosniff -- prevents browsers from MIME-sniffing
//   the response body and treating it as a different content type than what
//   the server declared (mitigates content-type confusion attacks).
//
// - X-Frame-Options: DENY -- prevents any site from embedding this page in
//   an iframe (mitigates clickjacking attacks).
//
// - Referrer-Policy: strict-origin-when-cross-origin -- sends the full URL
//   as referrer for same-origin requests but only the origin (no path) for
//   cross-origin requests (reduces information leakage).

use axum::body::Body;
use axum::http::{HeaderValue, Request};
use axum::middleware::Next;
use axum::response::Response;

/// Axum middleware function that inserts security headers into every HTTP
/// response before returning it to the client.
pub async fn security_headers(req: Request<Body>, next: Next) -> Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();

    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("x-frame-options", HeaderValue::from_static("DENY"));
    headers.insert(
        "referrer-policy",
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );
    // Content-Security-Policy restricts which resources the browser may load.
    // - script-src 'self': only scripts from the same origin (no inline scripts, no eval).
    // - style-src 'self' 'unsafe-inline': SolidJS injects critical CSS at render time,
    //   requiring 'unsafe-inline'; no external stylesheets are loaded.
    // - img-src 'self' data:: favicon and any base64-encoded images served inline.
    // - connect-src 'self': fetch/SSE only to the same origin (blocks exfiltration).
    // - font-src 'self': web fonts served from the same origin only.
    // - frame-ancestors 'none': equivalent to X-Frame-Options DENY; prevents embedding.
    headers.insert(
        "content-security-policy",
        HeaderValue::from_static(
            "default-src 'self'; \
             script-src 'self'; \
             style-src 'self' 'unsafe-inline'; \
             img-src 'self' data:; \
             connect-src 'self'; \
             font-src 'self'; \
             frame-ancestors 'none'",
        ),
    );
    // X-Permitted-Cross-Domain-Policies: none prevents Flash and Acrobat from
    // treating this server as a cross-domain policy target. Flash is end-of-life
    // but the header is a low-cost defense-in-depth measure.
    headers.insert(
        "x-permitted-cross-domain-policies",
        HeaderValue::from_static("none"),
    );
    // Strict-Transport-Security instructs browsers to require HTTPS for all future
    // requests to this origin for the next two years (63072000 seconds). Per RFC 6797
    // section 7.2, compliant browsers must not honor this header when received over
    // plain HTTP, so including it unconditionally is safe. Deployments that place a
    // TLS-terminating reverse proxy (e.g., nginx, caddy) in front of this server
    // benefit from the header because the client-to-proxy leg uses HTTPS.
    headers.insert(
        "strict-transport-security",
        HeaderValue::from_static("max-age=63072000"),
    );

    response
}
