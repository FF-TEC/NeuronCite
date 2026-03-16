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

//! neuroncite-mcp: Model Context Protocol server for NeuronCite.
//!
//! This crate implements the MCP specification over stdio transport. The
//! protocol version reported in the initialize handshake is derived from
//! `CARGO_PKG_VERSION` at compile time, keeping it in sync with the workspace
//! crate version. The server reads JSON-RPC 2.0 requests from stdin,
//! dispatches them to the appropriate handler, and writes JSON-RPC responses
//! to stdout. stderr is reserved for tracing log output.
//!
//! The crate is structured into the following modules:
//!
//! - `protocol` -- JSON-RPC 2.0 message types (Request, Response, Error).
//! - `transport` -- Line-based stdin/stdout reader/writer.
//! - `tools` -- Static MCP tool definitions with JSON Schema input descriptors.
//! - `dispatch` -- Routes `tools/call` requests to the correct handler function.
//! - `handlers` -- Per-tool handler implementations that bridge MCP calls to
//!   the NeuronCite search, store, and embed subsystems.
//! - `server` -- The main server loop (initialize handshake, tool discovery,
//!   request dispatch).
//! - `registration` -- Install/uninstall/status logic for writing the MCP
//!   server entry into Claude Code's configuration file.

#![forbid(unsafe_code)]
// `deny(warnings)` is inherited from [workspace.lints.rust] in the root Cargo.toml.

pub mod dispatch;
pub mod handlers;
pub mod protocol;
pub mod registration;
pub mod server;
pub mod tools;
pub mod transport;

/// Re-export of the MCP protocol version constant from `server::PROTOCOL_VERSION`.
/// Provides a single, crate-level access point for the protocol version string
/// used in initialize handshake responses.
pub use server::PROTOCOL_VERSION;
