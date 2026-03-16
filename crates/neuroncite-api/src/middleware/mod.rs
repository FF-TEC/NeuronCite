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

//! Tower middleware layers for the axum server.
//!
//! - `api_version` -- Tower layer that appends an `X-API-Version` response
//!   header to every response. Conveys the API version at the transport layer
//!   so clients can detect version mismatches without parsing the JSON body.
//! - `auth` -- Optional bearer token authentication middleware. When configured,
//!   rejects requests from non-localhost origins without a valid
//!   `Authorization: Bearer <token>` header.
//! - `cors` -- Cross-Origin Resource Sharing (CORS) middleware. Omitted for
//!   localhost bindings, wildcard or explicit origins for LAN bindings.
//! - `security_headers` -- Tower layer that appends defensive response headers
//!   (`X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy`)
//!   to every response regardless of the handler that produced it.

pub mod api_version;
pub mod auth;
pub mod cors;
pub mod security_headers;
