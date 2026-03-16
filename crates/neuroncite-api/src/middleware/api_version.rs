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

// Tower middleware layer that adds the X-API-Version header to every response.
// The header value is the current API version string, allowing clients to detect
// version mismatches without parsing the response body. This replaces the per-DTO
// `api_version` field pattern: the version is conveyed at the transport layer
// via a single middleware instead of being serialized redundantly in every JSON body.

use axum::body::Body;
use axum::http::{HeaderValue, Request, Response};
use std::task::{Context, Poll};
use tower::{Layer, Service};

/// The API version string injected into every response header.
const API_VERSION: &str = "v1";

/// Tower layer that wraps any inner service with the X-API-Version response header.
/// Applied alongside SecurityHeadersLayer in the router so that every response
/// includes the version identifier regardless of which handler produced it.
#[derive(Clone)]
pub struct ApiVersionLayer;

impl<S> Layer<S> for ApiVersionLayer {
    type Service = ApiVersionMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ApiVersionMiddleware { inner }
    }
}

/// Tower service that injects X-API-Version into every response before passing
/// it upstream. Uses the same pinned-future pattern as SecurityHeadersMiddleware
/// to avoid requiring the futures_util dependency.
#[derive(Clone)]
pub struct ApiVersionMiddleware<S> {
    inner: S,
}

impl<S> Service<Request<Body>> for ApiVersionMiddleware<S>
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
            // Insert X-API-Version so clients can detect version mismatches at
            // the HTTP header level without parsing the JSON body.
            response.headers_mut().insert(
                axum::http::header::HeaderName::from_static("x-api-version"),
                HeaderValue::from_static(API_VERSION),
            );
            Ok(response)
        })
    }
}
