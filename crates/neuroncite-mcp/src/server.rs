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

//! MCP server main loop.
//!
//! Implements the MCP lifecycle: initialize handshake, tool discovery, and
//! the request dispatch loop. The server reads JSON-RPC requests from a
//! buffered reader (stdin in production), processes them, and writes responses
//! to a writer (stdout in production).
//!
//! The server loop runs inside a tokio runtime because the tool handlers use
//! async operations (GPU worker channel communication). The transport layer
//! itself is synchronous (blocking reads from stdin), but each tool call is
//! dispatched as an async operation on the runtime.

use std::io::{BufRead, Write};
use std::sync::Arc;

use neuroncite_api::AppState;

use crate::dispatch::{DispatchResult, dispatch_tool_call};
use crate::protocol::{JsonRpcResponse, METHOD_NOT_FOUND, PARSE_ERROR};
use crate::tools::all_tools;
use crate::transport::{read_message, write_message};

/// Server metadata reported during the initialize handshake.
const SERVER_NAME: &str = "neuroncite";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// MCP protocol version reported in the initialize handshake response.
/// This must match the MCP specification version that the server implements,
/// not the crate version. Claude Code and other MCP clients use this field
/// to verify protocol compatibility during the handshake.
/// Re-exported from the crate root (`neuroncite_mcp::PROTOCOL_VERSION`).
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// Runs the MCP server loop on the given reader/writer pair.
///
/// This function blocks until the reader reaches EOF (the MCP client closed
/// the connection) or an unrecoverable I/O error occurs. It is designed to
/// be called from a tokio runtime context so that async tool handlers can
/// execute GPU worker operations.
///
/// # Arguments
///
/// * `reader` - Buffered reader for incoming JSON-RPC requests (stdin).
/// * `writer` - Writer for outgoing JSON-RPC responses (stdout).
/// * `state` - Shared application state (database pool, GPU worker, config).
/// * `rt` - Handle to the tokio runtime for spawning async tool handlers.
pub fn run_server(
    reader: &mut dyn BufRead,
    writer: &mut dyn Write,
    state: Arc<AppState>,
    rt: &tokio::runtime::Handle,
) {
    tracing::info!("MCP server starting (protocol {})", PROTOCOL_VERSION);

    loop {
        let msg = match read_message(reader) {
            None => {
                // EOF: the MCP client closed stdin. Clean shutdown.
                tracing::info!("MCP server shutting down: stdin closed");
                break;
            }
            Some(Err(e)) => {
                // Parse error: send an error response with id=null.
                tracing::warn!("MCP parse error: {e}");
                let resp = JsonRpcResponse::error(
                    serde_json::Value::Null,
                    PARSE_ERROR,
                    format!("parse error: {e}"),
                );
                if write_message(writer, &resp).is_err() {
                    break;
                }
                continue;
            }
            Some(Ok(req)) => req,
        };

        // Notifications (no `id`) do not require a response.
        let Some(ref id) = msg.id else {
            tracing::debug!("received notification: {}", msg.method);
            continue;
        };

        let id = id.clone();

        let response = match msg.method.as_str() {
            "initialize" => handle_initialize(id.clone()),
            "tools/list" => handle_tools_list(id.clone()),
            "tools/call" => {
                let params = msg.params.unwrap_or(serde_json::Value::Null);
                handle_tools_call(id.clone(), &params, &state, rt)
            }
            "ping" => JsonRpcResponse::success(id.clone(), serde_json::json!({})),
            _ => JsonRpcResponse::error(
                id.clone(),
                METHOD_NOT_FOUND,
                format!("unknown method: {}", msg.method),
            ),
        };

        if write_message(writer, &response).is_err() {
            tracing::error!("failed to write response; shutting down");
            break;
        }
    }
}

/// Handles the `initialize` method.
///
/// Returns the server's capabilities (tools support) and server metadata.
/// The MCP client sends this as the first message after process startup.
fn handle_initialize(id: serde_json::Value) -> JsonRpcResponse {
    let result = serde_json::json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
        }
    });
    JsonRpcResponse::success(id, result)
}

/// Handles the `tools/list` method.
///
/// Returns the full list of tool definitions (name, description, inputSchema)
/// for all tools exposed by this MCP server.
fn handle_tools_list(id: serde_json::Value) -> JsonRpcResponse {
    let tools = all_tools();
    let tool_array: Vec<serde_json::Value> = tools
        .into_iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            })
        })
        .collect();

    JsonRpcResponse::success(id, serde_json::json!({ "tools": tool_array }))
}

/// Handles the `tools/call` method.
///
/// Dispatches the tool call to the appropriate handler via the dispatch layer.
/// Runs the async handler on the provided tokio runtime handle and blocks
/// until the result is available.
fn handle_tools_call(
    id: serde_json::Value,
    params: &serde_json::Value,
    state: &Arc<AppState>,
    rt: &tokio::runtime::Handle,
) -> JsonRpcResponse {
    // Clone state for the async block.
    let state = state.clone();
    let params = params.clone();

    // Block on the async dispatch. The transport is synchronous (stdin line
    // reads), so blocking here is expected and does not starve other tasks.
    let result = rt.block_on(async { dispatch_tool_call(&state, &params).await });

    match result {
        DispatchResult::Success(content) => JsonRpcResponse::success(id, content),
        DispatchResult::Error { code: _, message } => {
            // All tool dispatch errors are returned as MCP content with
            // isError=true, regardless of the error code. This includes
            // unknown tool names (METHOD_NOT_FOUND from dispatch), missing
            // parameters (INVALID_PARAMS), and handler errors (INTERNAL_ERROR).
            //
            // Returning protocol-level JSON-RPC errors for tool calls causes
            // MCP clients to interpret them as transport failures. When the
            // client sends parallel tool calls, a protocol-level error from
            // one call triggers cancellation of all in-flight sibling calls
            // (the "sibling error" problem). By using isError instead, each
            // call is a protocol-level success with an application-level error
            // flag, allowing siblings to complete independently.
            JsonRpcResponse::success(
                id,
                serde_json::json!({
                    "content": [{
                        "type": "text",
                        "text": message,
                    }],
                    "isError": true,
                }),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Builds the JSON-RPC initialize request payload using the server's
    /// PROTOCOL_VERSION constant, keeping test payloads in sync with the
    /// crate version. Returns a single JSON line without a trailing newline.
    fn init_request_json() -> String {
        format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{{\"protocolVersion\":\"{}\",\"capabilities\":{{}},\"clientInfo\":{{\"name\":\"test\",\"version\":\"1.0\"}}}}}}",
            PROTOCOL_VERSION,
        )
    }

    /// T-MCP-018: The initialize handshake returns the correct protocol version
    /// and server capabilities.
    #[test]
    fn t_mcp_018_initialize_returns_capabilities() {
        let resp = handle_initialize(serde_json::json!(1));
        let result = resp.result.expect("should have result");
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
        assert!(result["capabilities"]["tools"].is_object());
        assert_eq!(result["serverInfo"]["name"], SERVER_NAME);
    }

    /// T-MCP-019: tools/list returns all tool definitions with the correct
    /// structure (name, description, inputSchema for each tool).
    #[test]
    fn t_mcp_019_tools_list_returns_all_tools() {
        let resp = handle_tools_list(serde_json::json!(2));
        let result = resp.result.expect("should have result");
        let tools = result["tools"].as_array().expect("tools array");
        assert!(!tools.is_empty());

        for tool in tools {
            assert!(tool["name"].is_string(), "tool must have name");
            assert!(
                tool["description"].is_string(),
                "tool must have description"
            );
            assert!(
                tool["inputSchema"].is_object(),
                "tool must have inputSchema"
            );
        }
    }

    /// T-MCP-020: The server loop processes a full initialize -> tools/list ->
    /// EOF sequence without errors.
    #[test]
    fn t_mcp_020_server_loop_initialize_and_list() {
        let input = format!(
            "{}\n{}\n{}\n",
            init_request_json(),
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}",
        );

        let mut reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();

        // Create a tokio runtime first so spawn_worker has a reactor context.
        let rt = tokio::runtime::Runtime::new().expect("runtime");

        let state = rt.block_on(async { build_test_state() });

        run_server(&mut reader, &mut output, state, rt.handle());

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();

        // Two responses expected: one for initialize (id=1), one for tools/list (id=2).
        // The notification does not produce a response.
        assert_eq!(lines.len(), 2, "expected 2 responses, got {}", lines.len());

        // Parse and verify the initialize response.
        let init_resp: serde_json::Value = serde_json::from_str(lines[0]).expect("parse init");
        assert_eq!(init_resp["id"], 1);
        assert_eq!(init_resp["result"]["protocolVersion"], PROTOCOL_VERSION);

        // Parse and verify the tools/list response.
        let list_resp: serde_json::Value = serde_json::from_str(lines[1]).expect("parse list");
        assert_eq!(list_resp["id"], 2);
        assert!(list_resp["result"]["tools"].is_array());
    }

    /// T-MCP-021: Malformed JSON input produces a parse error response.
    #[test]
    fn t_mcp_021_malformed_json_produces_parse_error() {
        let input = "this is not json\n";
        let mut reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let state = rt.block_on(async { build_test_state() });

        run_server(&mut reader, &mut output, state, rt.handle());

        let output_str = String::from_utf8(output).expect("utf8");
        let resp: serde_json::Value = serde_json::from_str(output_str.trim()).expect("parse");
        assert_eq!(resp["error"]["code"], PARSE_ERROR);
    }

    /// T-MCP-022: Unknown method name returns METHOD_NOT_FOUND error.
    #[test]
    fn t_mcp_022_unknown_method_returns_error() {
        let input = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"unknown/method\"}\n";
        let mut reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let state = rt.block_on(async { build_test_state() });

        run_server(&mut reader, &mut output, state, rt.handle());

        let output_str = String::from_utf8(output).expect("utf8");
        let resp: serde_json::Value = serde_json::from_str(output_str.trim()).expect("parse");
        assert_eq!(resp["error"]["code"], METHOD_NOT_FOUND);
    }

    /// T-MCP-116: The MCP server starts and completes an initialize handshake
    /// when spawn_worker is called with rt.enter() guard instead of inside
    /// rt.block_on(). This reproduces the exact pattern used by `run_mcp_serve`
    /// in the main binary: create a tokio runtime, enter its context via guard,
    /// spawn the worker, then run the synchronous server loop. The test
    /// verifies that this pattern does not panic and produces valid responses.
    #[test]
    fn t_mcp_116_server_loop_with_runtime_guard_pattern() {
        let input = format!(
            "{}\n{}\n{}\n",
            init_request_json(),
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}",
        );

        let mut reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();

        // Reproduce the run_mcp_serve pattern: create runtime, enter guard,
        // build state with spawn_worker, then run server with rt.handle().
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let _guard = rt.enter();
        let state = build_test_state_sync();

        run_server(&mut reader, &mut output, state, rt.handle());

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();

        assert_eq!(lines.len(), 2, "expected 2 responses (init + tools/list)");

        let init_resp: serde_json::Value = serde_json::from_str(lines[0]).expect("parse init");
        assert_eq!(init_resp["id"], 1);
        assert_eq!(init_resp["result"]["protocolVersion"], PROTOCOL_VERSION);

        let list_resp: serde_json::Value = serde_json::from_str(lines[1]).expect("parse list");
        assert_eq!(list_resp["id"], 2);
        assert!(list_resp["result"]["tools"].is_array());
    }

    /// T-MCP-026: Every line written to the MCP stdout transport is valid
    /// JSON-RPC 2.0. If tracing, ORT warnings, or any other diagnostic output
    /// leaks into stdout, the MCP client receives unparseable data. This test
    /// sends a full initialize + tools/list sequence and asserts that every
    /// output line is valid JSON containing the "jsonrpc" field.
    #[test]
    fn t_mcp_026_stdout_contains_only_valid_jsonrpc() {
        let input = format!(
            "{}\n{}\n{}\n",
            init_request_json(),
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}",
        );

        let mut reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let state = rt.block_on(async { build_test_state() });

        run_server(&mut reader, &mut output, state, rt.handle());

        let output_str = String::from_utf8(output).expect("utf8 output");
        for (i, line) in output_str.lines().enumerate() {
            let parsed: serde_json::Value = serde_json::from_str(line).unwrap_or_else(|e| {
                panic!(
                    "stdout line {} is not valid JSON (tracing or ORT leak?): \
                     error={e}, line={line:?}",
                    i + 1
                )
            });
            assert!(
                parsed["jsonrpc"].is_string(),
                "stdout line {} missing 'jsonrpc' field: {line:?}",
                i + 1
            );
        }
    }

    /// T-MCP-027: Searching a session whose embedding dimension differs from
    /// the loaded model returns an MCP error with isError=true and a message
    /// containing "dimension mismatch". This guards against silently producing
    /// meaningless cosine similarity scores when query and document embeddings
    /// have different vector sizes (e.g., bge-small 384d vs gte-large 1024d).
    #[test]
    fn t_mcp_027_search_dimension_mismatch_returns_error() {
        use neuroncite_core::{IndexConfig, StorageMode};
        use std::path::PathBuf;

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        // AppState has vector_dimension=4 (from StubBackend).
        let state = rt.block_on(async { build_test_state() });

        // Create a session with vector_dimension=1024, mismatching the
        // AppState's 4-dimensional StubBackend.
        let session_id = {
            let conn = state.pool.get().expect("conn");
            let config = IndexConfig {
                directory: PathBuf::from("/test"),
                model_name: "gte-large-en-v1.5".to_string(),
                chunk_strategy: "word".to_string(),
                chunk_size: Some(300),
                chunk_overlap: Some(50),
                max_words: None,
                ocr_language: "eng".to_string(),
                embedding_storage_mode: StorageMode::SqliteBlob,
                vector_dimension: 1024,
            };
            neuroncite_store::create_session(&conn, &config, "test").expect("create session")
        };

        // Build a tools/call request for neuroncite_search targeting the
        // mismatched session.
        let search_call = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "neuroncite_search",
                "arguments": {
                    "session_id": session_id,
                    "query": "test query",
                    "top_k": 5
                }
            }
        });

        let input = format!(
            "{}\n{}\n{}\n",
            init_request_json(),
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}",
            serde_json::to_string(&search_call).expect("serialize search call"),
        );

        let mut reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();

        run_server(&mut reader, &mut output, state, rt.handle());

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();

        // Two responses: initialize (id=1) and tools/call (id=3).
        assert_eq!(
            lines.len(),
            2,
            "expected 2 response lines, got {}",
            lines.len()
        );

        let search_resp: serde_json::Value =
            serde_json::from_str(lines[1]).expect("parse search response");
        assert_eq!(search_resp["id"], 3);

        // The dimension mismatch should be surfaced as an MCP isError response,
        // not a protocol-level JSON-RPC error.
        let result = &search_resp["result"];
        assert_eq!(
            result["isError"], true,
            "expected isError=true for dimension mismatch"
        );
        let error_text = result["content"][0]["text"]
            .as_str()
            .expect("error text should be a string");
        assert!(
            error_text.contains("dimension mismatch"),
            "error message should mention 'dimension mismatch', got: {error_text}"
        );
    }

    /// T-MCP-028: Searching a session whose embedding dimension matches the
    /// loaded model does NOT trigger a dimension mismatch error. This is the
    /// positive counterpart to T-MCP-087, verifying that the validation does
    /// not produce false positives for correctly configured sessions.
    #[test]
    fn t_mcp_028_search_matching_dimension_no_mismatch() {
        use neuroncite_core::{IndexConfig, StorageMode};
        use std::path::PathBuf;

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        // AppState has vector_dimension=4 (from StubBackend).
        let state = rt.block_on(async { build_test_state() });

        // Create a session with vector_dimension=4, matching the StubBackend.
        let session_id = {
            let conn = state.pool.get().expect("conn");
            let config = IndexConfig {
                directory: PathBuf::from("/test"),
                model_name: "stub".to_string(),
                chunk_strategy: "word".to_string(),
                chunk_size: Some(300),
                chunk_overlap: Some(50),
                max_words: None,
                ocr_language: "eng".to_string(),
                embedding_storage_mode: StorageMode::SqliteBlob,
                vector_dimension: 4,
            };
            neuroncite_store::create_session(&conn, &config, "test").expect("create session")
        };

        // Build a tools/call request for neuroncite_search targeting the
        // matching session. The search itself will return empty results (no
        // HNSW index loaded), but it should NOT fail with a mismatch error.
        let search_call = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "neuroncite_search",
                "arguments": {
                    "session_id": session_id,
                    "query": "test query",
                    "top_k": 5
                }
            }
        });

        let input = format!(
            "{}\n{}\n{}\n",
            init_request_json(),
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}",
            serde_json::to_string(&search_call).expect("serialize search call"),
        );

        let mut reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();

        run_server(&mut reader, &mut output, state, rt.handle());

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();
        assert_eq!(lines.len(), 2, "expected 2 response lines");

        let search_resp: serde_json::Value =
            serde_json::from_str(lines[1]).expect("parse search response");
        assert_eq!(search_resp["id"], 3);

        // The response should be a successful result (no isError), even though
        // the search returns empty results due to missing HNSW index.
        let result = &search_resp["result"];
        assert!(
            result.get("isError").is_none() || result["isError"] == false,
            "matching dimensions should not trigger mismatch error"
        );
    }

    /// T-MCP-029: Calling a nonexistent tool via tools/call returns isError=true
    /// instead of a protocol-level JSON-RPC error. This prevents MCP clients
    /// from interpreting tool dispatch failures as transport errors, which would
    /// cause cancellation of sibling calls in parallel tool call batches.
    ///
    /// Regression test for the "sibling error" problem: when the MCP client
    /// sends parallel tool calls and one returns a protocol-level error, the
    /// client aborts all in-flight sibling calls.
    #[test]
    fn t_mcp_029_unknown_tool_returns_is_error_not_protocol_error() {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let state = rt.block_on(async { build_test_state() });

        // Call a tool that does not exist.
        let call = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool_xyz",
                "arguments": {}
            }
        });

        let input = format!(
            "{}\n{}\n{}\n",
            init_request_json(),
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}",
            serde_json::to_string(&call).expect("serialize"),
        );

        let mut reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();

        run_server(&mut reader, &mut output, state, rt.handle());

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();
        assert_eq!(lines.len(), 2, "expected 2 response lines");

        let resp: serde_json::Value = serde_json::from_str(lines[1]).expect("parse response");
        assert_eq!(resp["id"], 3);

        // The response must be a protocol-level success with isError=true,
        // NOT a protocol-level JSON-RPC error (which would have an "error"
        // field instead of "result").
        assert!(
            resp.get("error").is_none(),
            "must not have a protocol-level error field"
        );
        let result = &resp["result"];
        assert_eq!(
            result["isError"], true,
            "unknown tool must return isError=true"
        );
        let error_text = result["content"][0]["text"].as_str().expect("error text");
        assert!(
            error_text.contains("nonexistent_tool_xyz"),
            "error message must mention the unknown tool name, got: {error_text}"
        );
    }

    /// T-MCP-030: A tool handler error (INTERNAL_ERROR) is returned as
    /// isError=true, not as a protocol-level JSON-RPC error. This validates
    /// that the isError wrapping works for handler-level failures (e.g.,
    /// database errors, validation errors).
    #[test]
    fn t_mcp_030_handler_error_returns_is_error() {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let state = rt.block_on(async { build_test_state() });

        // Call neuroncite_search with a non-existent session_id to trigger
        // a handler error (SESSION_NOT_FOUND).
        let call = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "neuroncite_search",
                "arguments": {
                    "session_id": 999999,
                    "query": "test"
                }
            }
        });

        let input = format!(
            "{}\n{}\n{}\n",
            init_request_json(),
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}",
            serde_json::to_string(&call).expect("serialize"),
        );

        let mut reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();

        run_server(&mut reader, &mut output, state, rt.handle());

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();
        assert_eq!(lines.len(), 2, "expected 2 response lines");

        let resp: serde_json::Value = serde_json::from_str(lines[1]).expect("parse response");
        assert_eq!(resp["id"], 3);

        // Handler errors must also use isError wrapping.
        assert!(
            resp.get("error").is_none(),
            "handler error must not have protocol-level error field"
        );
        let result = &resp["result"];
        assert_eq!(
            result["isError"], true,
            "handler error must return isError=true"
        );
    }

    /// T-MCP-031: Protocol-level errors (unknown JSON-RPC method like
    /// "resources/list") still return as JSON-RPC errors with an "error" field.
    /// The isError wrapping only applies to tools/call dispatch, not to
    /// top-level method routing.
    #[test]
    fn t_mcp_031_unknown_method_still_protocol_error() {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let state = rt.block_on(async { build_test_state() });

        let call = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list"
        });

        let input = format!(
            "{}\n{}\n{}\n",
            init_request_json(),
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}",
            serde_json::to_string(&call).expect("serialize"),
        );

        let mut reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();

        run_server(&mut reader, &mut output, state, rt.handle());

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();
        assert_eq!(lines.len(), 2, "expected 2 response lines");

        let resp: serde_json::Value = serde_json::from_str(lines[1]).expect("parse response");
        assert_eq!(resp["id"], 3);

        // Unknown JSON-RPC methods (not tool names) should still be
        // protocol-level errors. The isError change only affects tools/call.
        assert!(
            resp.get("error").is_some(),
            "unknown method must return a protocol-level error"
        );
        assert_eq!(resp["error"]["code"], METHOD_NOT_FOUND);
    }

    /// T-MCP-032: Two sequential tool calls where the first fails and the
    /// second succeeds both return independent responses. This verifies that
    /// a failed tool call does not corrupt the server state or prevent
    /// subsequent calls from succeeding.
    ///
    /// The second call uses tools/list (synchronous dispatch) instead of a
    /// real tool call to avoid the Handle::block_on path that deadlocks
    /// when the tokio worker threads are contended by parallel tests.
    #[test]
    fn t_mcp_032_failed_call_does_not_affect_subsequent() {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let state = rt.block_on(async { build_test_state() });

        // First call: unknown tool (will fail with isError=true).
        let fail_call = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {}
            }
        });

        // Second call: tools/list (synchronous, always succeeds).
        let ok_call = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/list"
        });

        let input = format!(
            "{}\n{}\n{}\n{}\n",
            init_request_json(),
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}",
            serde_json::to_string(&fail_call).expect("serialize"),
            serde_json::to_string(&ok_call).expect("serialize"),
        );

        let mut reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();

        run_server(&mut reader, &mut output, state, rt.handle());

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();
        assert_eq!(
            lines.len(),
            3,
            "expected 3 response lines (init + fail + list)"
        );

        // First tool call: isError=true.
        let fail_resp: serde_json::Value =
            serde_json::from_str(lines[1]).expect("parse fail response");
        assert_eq!(fail_resp["id"], 3);
        assert_eq!(fail_resp["result"]["isError"], true);

        // Second call (tools/list): success with tool array.
        let ok_resp: serde_json::Value =
            serde_json::from_str(lines[2]).expect("parse list response");
        assert_eq!(ok_resp["id"], 4);
        assert!(
            ok_resp.get("error").is_none(),
            "tools/list must not have error after a failed call"
        );
        assert!(
            ok_resp["result"]["tools"].is_array(),
            "tools/list must return a tools array"
        );
    }

    /// Constructs a minimal in-memory AppState using the runtime guard pattern.
    /// Requires an active tokio runtime context via rt.enter() on the calling
    /// thread (spawn_worker calls tokio::spawn internally).
    fn build_test_state_sync() -> Arc<AppState> {
        use neuroncite_core::{EmbeddingBackend, ModelInfo, NeuronCiteError};

        struct StubBackend;
        impl EmbeddingBackend for StubBackend {
            fn name(&self) -> &str {
                "stub"
            }
            fn vector_dimension(&self) -> usize {
                4
            }
            fn load_model(&mut self, _: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
            }
            fn supports_gpu(&self) -> bool {
                false
            }
            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn loaded_model_id(&self) -> String {
                "stub".to_string()
            }
        }

        let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA foreign_keys = ON;")?;
            Ok(())
        });
        let pool = r2d2::Pool::builder()
            .max_size(1)
            .build(manager)
            .expect("pool build");

        {
            let conn = pool.get().expect("get conn");
            neuroncite_store::migrate(&conn).expect("migrate");
        }

        let backend: Arc<dyn EmbeddingBackend> = Arc::new(StubBackend);
        let worker_handle = neuroncite_api::spawn_worker(backend, None);
        let config = neuroncite_core::AppConfig::default();
        neuroncite_api::AppState::new(pool, worker_handle, config, true, None, 4)
            .expect("test AppState construction must succeed")
    }

    /// Constructs a minimal in-memory AppState for server loop tests. Must
    /// be called from within a tokio runtime context (spawn_worker requires
    /// a reactor for the background task).
    fn build_test_state() -> Arc<AppState> {
        use neuroncite_core::{EmbeddingBackend, ModelInfo, NeuronCiteError};

        struct StubBackend;
        impl EmbeddingBackend for StubBackend {
            fn name(&self) -> &str {
                "stub"
            }
            fn vector_dimension(&self) -> usize {
                4
            }
            fn load_model(&mut self, _: &str) -> Result<(), NeuronCiteError> {
                Ok(())
            }
            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, NeuronCiteError> {
                Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
            }
            fn supports_gpu(&self) -> bool {
                false
            }
            fn available_models(&self) -> Vec<ModelInfo> {
                vec![]
            }
            fn loaded_model_id(&self) -> String {
                "stub".to_string()
            }
        }

        let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA foreign_keys = ON;")?;
            Ok(())
        });
        let pool = r2d2::Pool::builder()
            .max_size(1)
            .build(manager)
            .expect("pool build");

        {
            let conn = pool.get().expect("get conn");
            neuroncite_store::migrate(&conn).expect("migrate");
        }

        let backend: Arc<dyn EmbeddingBackend> = Arc::new(StubBackend);
        let worker_handle = neuroncite_api::spawn_worker(backend, None);
        let config = neuroncite_core::AppConfig::default();
        neuroncite_api::AppState::new(pool, worker_handle, config, true, None, 4)
            .expect("test AppState construction must succeed")
    }
}
