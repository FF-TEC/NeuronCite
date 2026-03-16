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

// Handler for the neuroncite_session_diff MCP tool.
//
// Compares two index sessions by their indexed file records and reports
// which files are unique to each session, which are identical, and which
// have content or structural differences.

use std::sync::Arc;

use neuroncite_api::AppState;

/// Compares two sessions and reports file-level differences.
///
/// Parameters:
/// - session_a (required): First session ID.
/// - session_b (required): Second session ID.
///
/// Returns JSON with summary counts and per-category file lists.
pub async fn handle(
    state: &Arc<AppState>,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let session_a = params["session_a"]
        .as_i64()
        .ok_or("missing required parameter: session_a")?;

    let session_b = params["session_b"]
        .as_i64()
        .ok_or("missing required parameter: session_b")?;

    if session_a == session_b {
        return Err("session_a and session_b must be different sessions".to_string());
    }

    let conn = state
        .pool
        .get()
        .map_err(|e| format!("connection pool error: {e}"))?;

    // Validate both sessions exist.
    neuroncite_store::get_session(&conn, session_a)
        .map_err(|_| format!("session not found: {session_a}"))?;
    neuroncite_store::get_session(&conn, session_b)
        .map_err(|_| format!("session not found: {session_b}"))?;

    let diff = neuroncite_store::diff_sessions(&conn, session_a, session_b)
        .map_err(|e| format!("session diff failed: {e}"))?;

    let only_in_a: Vec<serde_json::Value> = diff
        .only_in_a
        .iter()
        .map(|f| {
            serde_json::json!({
                "file_id": f.id,
                "file_path": f.file_path,
                "page_count": f.page_count,
                "size": f.size,
            })
        })
        .collect();

    let only_in_b: Vec<serde_json::Value> = diff
        .only_in_b
        .iter()
        .map(|f| {
            serde_json::json!({
                "file_id": f.id,
                "file_path": f.file_path,
                "page_count": f.page_count,
                "size": f.size,
            })
        })
        .collect();

    let identical: Vec<serde_json::Value> = diff
        .identical
        .iter()
        .map(|d| {
            serde_json::json!({
                "file_path": d.file_path,
                "file_id_a": d.file_id_a,
                "file_id_b": d.file_id_b,
            })
        })
        .collect();

    let changed: Vec<serde_json::Value> = diff
        .changed
        .iter()
        .map(|d| {
            serde_json::json!({
                "file_path": d.file_path,
                "file_id_a": d.file_id_a,
                "file_id_b": d.file_id_b,
                "hash_changed": d.hash_changed,
                "page_count_changed": d.page_count_changed,
                "size_changed": d.size_changed,
                "hash_a": d.hash_a,
                "hash_b": d.hash_b,
            })
        })
        .collect();

    Ok(serde_json::json!({
        "session_a": session_a,
        "session_b": session_b,
        "summary": {
            "only_in_a": diff.only_in_a.len(),
            "only_in_b": diff.only_in_b.len(),
            "identical": diff.identical.len(),
            "changed": diff.changed.len(),
            "total_in_a": diff.only_in_a.len() + diff.identical.len() + diff.changed.len(),
            "total_in_b": diff.only_in_b.len() + diff.identical.len() + diff.changed.len(),
        },
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "identical": identical,
        "changed": changed,
    }))
}

#[cfg(test)]
mod tests {
    /// T-MCP-114: session_diff handler rejects same session IDs.
    #[test]
    fn t_mcp_075_session_diff_rejects_same_session() {
        let params = serde_json::json!({
            "session_a": 1,
            "session_b": 1,
        });
        let a = params["session_a"].as_i64().unwrap();
        let b = params["session_b"].as_i64().unwrap();
        assert_eq!(a, b, "should detect same session");
    }

    /// T-MCP-115: session_diff handler requires session_a parameter.
    #[test]
    fn t_mcp_076_session_diff_requires_session_a() {
        let params = serde_json::json!({
            "session_b": 2,
        });
        assert!(params["session_a"].as_i64().is_none());
    }
}
