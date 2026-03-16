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

//! Security response headers middleware.
//!
//! Adds defensive HTTP response headers to every response processed by the
//! axum router. The headers instruct browsers to apply additional security
//! policies that mitigate common web vulnerabilities:
//!
//! - `X-Content-Type-Options: nosniff` -- Prevents browsers from MIME-sniffing
//!   the response content type away from the declared `Content-Type`, which
//!   stops attacks that upload polyglot files and trigger script execution.
//!
//! - `X-Frame-Options: DENY` -- Prevents the application from being embedded
//!   in `<iframe>`, `<frame>`, or `<object>` elements. Mitigates clickjacking.
//!
//! - `Content-Security-Policy` -- Restricts which origins can load scripts,
//!   stylesheets, images, and other sub-resources. The policy here is
//!   conservative: all resources must come from the same origin (`'self'`),
//!   with the exception that inline styles are permitted (`'unsafe-inline'`)
//!   because the SolidJS frontend uses them for dynamic styling.

use axum::body::Body;
use axum::http::{HeaderValue, Request, Response};
use std::task::{Context, Poll};
use tower::{Layer, Service};

/// Tower layer that wraps any inner service with security response headers.
#[derive(Clone)]
pub struct SecurityHeadersLayer;

impl<S> Layer<S> for SecurityHeadersLayer {
    type Service = SecurityHeadersMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        SecurityHeadersMiddleware { inner }
    }
}

/// Tower service that injects security headers into every response.
#[derive(Clone)]
pub struct SecurityHeadersMiddleware<S> {
    inner: S,
}

impl<S> Service<Request<Body>> for SecurityHeadersMiddleware<S>
where
    S: Service<Request<Body>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response<Body>;
    type Error = S::Error;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let mut inner = self.inner.clone();
        Box::pin(async move {
            let mut response = inner.call(req).await?;
            let headers = response.headers_mut();
            // Prevent MIME-type sniffing attacks.
            headers.insert(
                "x-content-type-options",
                HeaderValue::from_static("nosniff"),
            );
            // Prevent clickjacking via iframe embedding.
            headers.insert("x-frame-options", HeaderValue::from_static("DENY"));
            // Restrict resource loading to the same origin. Inline styles are
            // permitted because the SolidJS frontend uses them.
            headers.insert(
                "content-security-policy",
                HeaderValue::from_static(
                    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
                ),
            );
            Ok(response)
        })
    }
}
