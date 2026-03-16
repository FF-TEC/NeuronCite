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

//! JSON-RPC 2.0 message types for the MCP protocol.
//!
//! Defines the request, response, and error structures used by the MCP server.
//! All messages are serialized as single-line JSON objects terminated by a
//! newline character. The `id` field links responses to their originating
//! requests. Notifications (requests without an `id`) are supported for the
//! `notifications/initialized` message sent by the client after the handshake.

use serde::{Deserialize, Serialize};

/// A JSON-RPC 2.0 request message received from the MCP client (stdin).
///
/// The `id` field is optional because MCP clients send `notifications/initialized`
/// as a notification (no `id`). For regular method calls (`initialize`,
/// `tools/list`, `tools/call`), the `id` is present and must be echoed back
/// in the response.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    /// Protocol version identifier. Always "2.0" for JSON-RPC 2.0.
    /// Deserialized to validate the complete JSON-RPC 2.0 message structure,
    /// but not read in Rust code -- the server always responds with the
    /// hard-coded "2.0" literal in `JsonRpcResponse`.
    #[allow(dead_code)]
    pub jsonrpc: String,

    /// Request identifier. Present for method calls, absent for notifications.
    pub id: Option<serde_json::Value>,

    /// The method name being invoked (e.g., "initialize", "tools/list", "tools/call").
    pub method: String,

    /// Method parameters. May be absent for parameterless methods like "tools/list".
    pub params: Option<serde_json::Value>,
}

/// A JSON-RPC 2.0 response message sent to the MCP client (stdout).
///
/// Contains either a `result` (on success) or an `error` (on failure), but
/// not both. The `id` matches the request that triggered this response.
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    /// Protocol version identifier. Always "2.0".
    pub jsonrpc: &'static str,

    /// The request identifier echoed back from the originating request.
    pub id: serde_json::Value,

    /// The successful result payload. Absent when `error` is present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,

    /// The error payload. Absent when `result` is present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC 2.0 error object returned in the `error` field of a response.
///
/// The `code` field uses standard JSON-RPC error codes:
/// - `-32700` Parse error (malformed JSON)
/// - `-32600` Invalid request (missing required fields)
/// - `-32601` Method not found (unknown method name)
/// - `-32602` Invalid params (schema validation failed)
/// - `-32603` Internal error (backend failure, database error)
#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    /// Numeric error code following JSON-RPC 2.0 conventions.
    pub code: i32,

    /// Human-readable error description.
    pub message: String,

    /// Additional structured data about the error. May be null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Standard JSON-RPC 2.0 error codes.
pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;
pub const INTERNAL_ERROR: i32 = -32603;

impl JsonRpcResponse {
    /// Constructs a successful response with the given result payload.
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Constructs an error response with the given error code and message.
    pub fn error(id: serde_json::Value, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message,
                data: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-MCP-001: JsonRpcRequest deserializes a valid initialize request with
    /// all fields present (jsonrpc, id, method, params).
    #[test]
    fn t_mcp_001_deserialize_request_with_all_fields() {
        let json = r#"{
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "0.1.0"}
        }"#;
        let req: JsonRpcRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.method, "initialize");
        assert_eq!(req.id, Some(serde_json::json!(1)));
        assert!(req.params.is_some());
    }

    /// T-MCP-002: JsonRpcRequest deserializes a notification (no `id` field).
    #[test]
    fn t_mcp_002_deserialize_notification_without_id() {
        let json = r#"{"jsonrpc": "2.0", "method": "notifications/initialized"}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.method, "notifications/initialized");
        assert!(req.id.is_none());
        assert!(req.params.is_none());
    }

    /// T-MCP-003: JsonRpcResponse success serialization produces correct JSON
    /// with `result` present and `error` absent.
    #[test]
    fn t_mcp_003_serialize_success_response() {
        let resp =
            JsonRpcResponse::success(serde_json::json!(42), serde_json::json!({"tools": []}));
        let json = serde_json::to_string(&resp).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["id"], 42);
        assert!(parsed["result"].is_object());
        assert!(parsed.get("error").is_none());
    }

    /// T-MCP-004: JsonRpcResponse error serialization produces correct JSON
    /// with `error` present and `result` absent.
    #[test]
    fn t_mcp_004_serialize_error_response() {
        let resp = JsonRpcResponse::error(
            serde_json::json!(7),
            METHOD_NOT_FOUND,
            "method not found".to_string(),
        );
        let json = serde_json::to_string(&resp).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(parsed["error"]["code"], METHOD_NOT_FOUND);
        assert_eq!(parsed["error"]["message"], "method not found");
        assert!(parsed.get("result").is_none());
    }

    /// T-MCP-005: All standard error codes are negative integers conforming
    /// to the JSON-RPC 2.0 specification range (-32768 to -32000).
    #[test]
    fn t_mcp_005_error_codes_in_valid_range() {
        let codes = [
            PARSE_ERROR,
            INVALID_REQUEST,
            METHOD_NOT_FOUND,
            INVALID_PARAMS,
            INTERNAL_ERROR,
        ];
        for code in codes {
            assert!(
                (-32768..=-32000).contains(&code),
                "error code {code} outside JSON-RPC reserved range"
            );
        }
    }
}
