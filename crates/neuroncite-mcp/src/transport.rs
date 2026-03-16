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

//! Line-based stdin/stdout transport for JSON-RPC 2.0 messages.
//!
//! The MCP protocol over stdio uses newline-delimited JSON: each message is a
//! single JSON object followed by `\n`. This module provides `read_message` to
//! read one JSON-RPC request from stdin and `write_message` to write one
//! JSON-RPC response to stdout. stderr remains available for tracing output.
//!
//! The transport operates synchronously on `BufRead` and `Write` trait objects
//! so that it can be tested with in-memory buffers without requiring actual
//! stdin/stdout file descriptors.

use std::io::{BufRead, Write};

use crate::protocol::{JsonRpcRequest, JsonRpcResponse};

/// Reads a single JSON-RPC request from the given buffered reader.
///
/// Blocks until a complete line is available. Returns `None` when the reader
/// reaches EOF (the MCP client closed the connection). Returns `Some(Err(...))`
/// when the line is not valid JSON or does not conform to the JsonRpcRequest
/// schema.
pub fn read_message(reader: &mut dyn BufRead) -> Option<Result<JsonRpcRequest, String>> {
    let mut line = String::new();
    match reader.read_line(&mut line) {
        Ok(0) => None, // EOF: client closed stdin.
        Ok(_) => {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                // Skip blank lines (some clients send trailing newlines).
                return Some(Err("empty line".to_string()));
            }
            match serde_json::from_str::<JsonRpcRequest>(trimmed) {
                Ok(req) => Some(Ok(req)),
                Err(e) => Some(Err(format!("JSON parse error: {e}"))),
            }
        }
        Err(e) => Some(Err(format!("stdin read error: {e}"))),
    }
}

/// Writes a single JSON-RPC response to the given writer as a newline-terminated
/// JSON line. Flushes the writer after each message to prevent buffering delays.
pub fn write_message(writer: &mut dyn Write, response: &JsonRpcResponse) -> std::io::Result<()> {
    let json = serde_json::to_string(response)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writeln!(writer, "{json}")?;
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// T-MCP-006: read_message returns None on empty input (EOF).
    #[test]
    fn t_mcp_006_read_eof_returns_none() {
        let mut reader = Cursor::new(b"");
        let result = read_message(&mut reader);
        assert!(result.is_none());
    }

    /// T-MCP-007: read_message parses a valid JSON-RPC request line.
    #[test]
    fn t_mcp_007_read_valid_request() {
        let input = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}\n";
        let mut reader = Cursor::new(input.as_slice());
        let result = read_message(&mut reader);
        let req = result.expect("should read").expect("should parse");
        assert_eq!(req.method, "tools/list");
        assert_eq!(req.id, Some(serde_json::json!(1)));
    }

    /// T-MCP-008: read_message returns an error for malformed JSON.
    #[test]
    fn t_mcp_008_read_malformed_json() {
        let input = b"not valid json\n";
        let mut reader = Cursor::new(input.as_slice());
        let result = read_message(&mut reader);
        let err = result.expect("should read").expect_err("should fail");
        assert!(err.contains("JSON parse error"));
    }

    /// T-MCP-009: write_message produces a newline-terminated JSON string.
    #[test]
    fn t_mcp_009_write_produces_newline_terminated_json() {
        let response =
            JsonRpcResponse::success(serde_json::json!(1), serde_json::json!({"tools": []}));
        let mut buf = Vec::new();
        write_message(&mut buf, &response).expect("write");
        let output = String::from_utf8(buf).expect("utf8");
        assert!(output.ends_with('\n'));
        let parsed: serde_json::Value = serde_json::from_str(output.trim()).expect("parse output");
        assert_eq!(parsed["jsonrpc"], "2.0");
    }

    /// T-MCP-010: Multiple messages can be read sequentially from the same stream.
    #[test]
    fn t_mcp_010_read_multiple_messages() {
        let input = concat!(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}\n",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}\n"
        );
        let mut reader = Cursor::new(input.as_bytes());

        let msg1 = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(msg1.method, "initialize");

        let msg2 = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(msg2.method, "notifications/initialized");

        let msg3 = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(msg3.method, "tools/list");

        // EOF after all messages.
        assert!(read_message(&mut reader).is_none());
    }
}
