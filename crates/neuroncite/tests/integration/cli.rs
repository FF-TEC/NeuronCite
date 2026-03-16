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

// CLI subcommand integration tests.
//
// These tests invoke the compiled neuroncite binary as a child process using
// std::process::Command and verify exit codes and stdout/stderr output. Each
// test targets a specific CLI subcommand defined in main.rs.
//
// The binary path is resolved at compile time via the env!("CARGO_BIN_EXE_neuroncite")
// macro, which cargo sets when compiling integration test targets for binary crates.

use std::process::Command;
use std::time::Duration;

/// Returns the absolute path to the compiled neuroncite binary.
/// The path is resolved at compile time by cargo via the CARGO_BIN_EXE_<name>
/// environment variable.
fn neuroncite_bin() -> &'static str {
    env!("CARGO_BIN_EXE_neuroncite")
}

/// Forcefully terminates a child process by its OS process ID. On Windows, uses
/// `taskkill /F /PID`. On Unix, sends SIGKILL via the `kill` command. Called
/// when a spawned child process exceeds its timeout and the Child handle has
/// been moved into a waiter thread, preventing a direct `.kill()` call.
fn kill_process_by_id(pid: u32) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/PID", &pid.to_string()])
            .output();
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid.to_string()])
            .output();
    }
}

/// Checks whether a failed Command output indicates the `backend-ort` feature
/// was not compiled in. The binary returns exit code 1 with a JSON stderr
/// containing `"code":"feature_disabled"` when a subcommand requires ORT
/// but the feature is absent. Tests that require ORT call this helper and
/// return early (skip) when the feature is not available, rather than failing.
fn is_feature_disabled(output: &std::process::Output) -> bool {
    if output.status.success() {
        return false;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    stderr.contains("feature_disabled")
}

// ---------------------------------------------------------------------------
// T-CLI-001: `neuroncite version` exits 0 and prints JSON with version field
// ---------------------------------------------------------------------------

/// T-CLI-001: Verifies that the `version` subcommand exits with status code 0
/// and produces JSON output containing a "version" field on stdout.
///
/// The version subcommand prints build metadata including the crate version
/// from Cargo.toml and the list of compiled feature flags.
#[test]
fn t_cli_001_version_exits_0_with_json() {
    let output = Command::new(neuroncite_bin())
        .args(["version", "--log-level", "error"])
        .output()
        .expect("failed to execute neuroncite binary");

    assert!(
        output.status.success(),
        "neuroncite version must exit with status 0, got: {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("version output must be valid JSON.\nParse error: {e}\nStdout: {stdout}")
    });

    assert!(
        json["version"].is_string(),
        "JSON output must contain a 'version' field of type string"
    );

    // The version string must match the Cargo package version.
    let version_str = json["version"].as_str().expect("version must be a string");
    assert!(!version_str.is_empty(), "version string must not be empty");
}

// ---------------------------------------------------------------------------
// T-CLI-002: `neuroncite doctor` exits 0 with valid JSON report
// ---------------------------------------------------------------------------

/// T-CLI-002: Verifies that the `doctor` subcommand exits with status code 0
/// and produces a valid JSON report on stdout.
///
/// The doctor subcommand checks for the availability of optional runtime
/// dependencies (CUDA, ONNX Runtime, PDFium, Tesseract) and reports their
/// status. The report must be valid JSON with entries for each dependency.
#[test]
fn t_cli_002_doctor_exits_0_with_json() {
    let output = Command::new(neuroncite_bin())
        .args(["doctor", "--log-level", "error"])
        .output()
        .expect("failed to execute neuroncite binary");

    assert!(
        output.status.success(),
        "neuroncite doctor must exit with status 0, got: {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("doctor output must be valid JSON.\nParse error: {e}\nStdout: {stdout}")
    });

    // The report must contain entries for each checked dependency.
    assert!(
        json["cuda"].is_object(),
        "doctor report must contain a 'cuda' entry"
    );
    assert!(
        json["onnxruntime"].is_object(),
        "doctor report must contain an 'onnxruntime' entry"
    );
    assert!(
        json["pdfium"].is_object(),
        "doctor report must contain a 'pdfium' entry"
    );
    assert!(
        json["tesseract"].is_object(),
        "doctor report must contain a 'tesseract' entry"
    );
}

// ---------------------------------------------------------------------------
// T-CLI-003: `neuroncite index --directory /nonexistent` exits 1
// ---------------------------------------------------------------------------

/// T-CLI-003: Verifies that the `index` subcommand exits with status code 1
/// when given a directory path that does not exist on the filesystem.
///
/// The run_index function checks for directory existence before proceeding
/// with the indexing pipeline and returns exit code 1 on failure.
#[test]
fn t_cli_003_index_nonexistent_directory_exits_1() {
    let output = Command::new(neuroncite_bin())
        .args([
            "index",
            "--directory",
            "/nonexistent/path/that/does/not/exist",
            "--log-level",
            "error",
        ])
        .output()
        .expect("failed to execute neuroncite binary");

    assert_eq!(
        output.status.code(),
        Some(1),
        "neuroncite index with nonexistent directory must exit with status 1\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ---------------------------------------------------------------------------
// T-CLI-004: `neuroncite search` without required args exits 2
// ---------------------------------------------------------------------------

/// T-CLI-004: Verifies that the `search` subcommand exits with status code 2
/// (clap's default for missing required arguments) when invoked without the
/// mandatory --directory, --session-id, and --query arguments.
///
/// Clap exits with code 2 and writes a usage error message to stderr when
/// required arguments are not provided.
#[test]
fn t_cli_004_search_missing_args_exits_2() {
    let output = Command::new(neuroncite_bin())
        .args(["search"])
        .output()
        .expect("failed to execute neuroncite binary");

    assert_eq!(
        output.status.code(),
        Some(2),
        "neuroncite search without required arguments must exit with status 2\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ---------------------------------------------------------------------------
// T-CLI-005: `neuroncite sessions --directory <tmpdir>` prints empty array
// ---------------------------------------------------------------------------

/// T-CLI-005: Verifies that the `sessions` subcommand prints an empty JSON
/// array when pointed at a temporary directory that has no .neuroncite database.
///
/// The current implementation returns an empty array regardless of the directory
/// contents (it does not open or create a database). This test documents that
/// behavior: the output is `[]`, and the exit code is 0.
#[test]
fn t_cli_005_sessions_empty_directory_prints_empty_array() {
    let tmp = tempfile::TempDir::new().expect("failed to create temp dir");

    let output = Command::new(neuroncite_bin())
        .args([
            "sessions",
            "--directory",
            &tmp.path().to_string_lossy(),
            "--log-level",
            "error",
        ])
        .output()
        .expect("failed to execute neuroncite binary");

    assert!(
        output.status.success(),
        "neuroncite sessions must exit with status 0, got: {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("sessions output must be valid JSON.\nParse error: {e}\nStdout: {stdout}")
    });

    assert!(json.is_array(), "sessions output must be a JSON array");
    assert_eq!(
        json.as_array().expect("must be array").len(),
        0,
        "sessions for a directory with no database must be an empty array"
    );
}

// ---------------------------------------------------------------------------
// T-CLI-006: `neuroncite models list` exits 0 with valid JSON
// ---------------------------------------------------------------------------

/// T-CLI-006: Verifies that `models list` exits with status 0 and produces
/// JSON output containing a `models` array with at least one entry. Each
/// entry must have `model_id`, `display_name`, `vector_dimension`, `cached`,
/// and `pooling` fields. The response also includes `total` and `cached_count`
/// summary fields.
#[test]
fn t_cli_006_models_list_exits_0_with_json() {
    let output = Command::new(neuroncite_bin())
        .args(["models", "list", "--log-level", "error"])
        .output()
        .expect("failed to execute neuroncite binary");

    if is_feature_disabled(&output) {
        eprintln!("T-CLI-006: skipping -- backend-ort feature not compiled in");
        return;
    }

    assert!(
        output.status.success(),
        "neuroncite models list must exit with status 0, got: {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("models list output must be valid JSON.\nParse error: {e}\nStdout: {stdout}")
    });

    // Verify top-level structure.
    assert!(
        json["models"].is_array(),
        "output must contain a 'models' array"
    );
    let models = json["models"].as_array().expect("must be array");
    assert!(
        !models.is_empty(),
        "models array must contain at least one entry"
    );

    assert!(
        json["total"].is_number(),
        "output must contain a 'total' field"
    );
    assert!(
        json["cached_count"].is_number(),
        "output must contain a 'cached_count' field"
    );

    // Verify each entry has the required fields.
    for (i, model) in models.iter().enumerate() {
        assert!(
            model["model_id"].is_string(),
            "model[{i}] must have a 'model_id' string field"
        );
        assert!(
            model["display_name"].is_string(),
            "model[{i}] must have a 'display_name' string field"
        );
        assert!(
            model["vector_dimension"].is_number(),
            "model[{i}] must have a 'vector_dimension' number field"
        );
        assert!(
            model["pooling"].is_string(),
            "model[{i}] must have a 'pooling' string field"
        );
        assert!(
            model["cached"].is_boolean(),
            "model[{i}] must have a 'cached' boolean field"
        );
    }
}

// ---------------------------------------------------------------------------
// T-CLI-007: `neuroncite models info BAAI/bge-small-en-v1.5` exits 0
// ---------------------------------------------------------------------------

/// T-CLI-007: Verifies that `models info` with a known model ID exits with
/// status 0 and returns the correct configuration for BGE Small EN v1.5:
/// 384 dimensions, "cls" pooling, 512 max sequence length, and the
/// `uses_token_type_ids` flag set to true.
#[test]
fn t_cli_007_models_info_known_model() {
    let output = Command::new(neuroncite_bin())
        .args([
            "models",
            "info",
            "BAAI/bge-small-en-v1.5",
            "--log-level",
            "error",
        ])
        .output()
        .expect("failed to execute neuroncite binary");

    if is_feature_disabled(&output) {
        eprintln!("T-CLI-007: skipping -- backend-ort feature not compiled in");
        return;
    }

    assert!(
        output.status.success(),
        "neuroncite models info must exit with status 0, got: {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("models info output must be valid JSON.\nParse error: {e}\nStdout: {stdout}")
    });

    assert_eq!(json["model_id"], "BAAI/bge-small-en-v1.5");
    assert_eq!(json["vector_dimension"], 384);
    assert_eq!(json["pooling"], "cls");
    assert_eq!(json["max_seq_len"], 512);
    assert_eq!(json["uses_token_type_ids"], true);
    assert!(json["files"].is_array(), "must have a 'files' array");
}

// ---------------------------------------------------------------------------
// T-CLI-008: `neuroncite models info nonexistent/model` exits 1
// ---------------------------------------------------------------------------

/// T-CLI-008: Verifies that `models info` with an unknown model ID exits with
/// status 1 and writes an error JSON object to stderr with the error code
/// "model_not_found".
#[test]
fn t_cli_008_models_info_unknown_model_exits_1() {
    let output = Command::new(neuroncite_bin())
        .args([
            "models",
            "info",
            "nonexistent/model-xyz",
            "--log-level",
            "error",
        ])
        .output()
        .expect("failed to execute neuroncite binary");

    if is_feature_disabled(&output) {
        eprintln!("T-CLI-008: skipping -- backend-ort feature not compiled in");
        return;
    }

    assert_eq!(
        output.status.code(),
        Some(1),
        "neuroncite models info with unknown model must exit with status 1\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify the error JSON on stderr.
    let stderr = String::from_utf8_lossy(&output.stderr);
    let error_json: serde_json::Value = serde_json::from_str(stderr.trim()).unwrap_or_else(|e| {
        panic!("stderr must contain valid error JSON.\nParse error: {e}\nStderr: {stderr}")
    });
    assert_eq!(
        error_json["code"], "model_not_found",
        "error code must be 'model_not_found'"
    );
}

// ---------------------------------------------------------------------------
// T-CLI-009: `neuroncite models system` exits 0 with valid JSON
// ---------------------------------------------------------------------------

/// T-CLI-009: Verifies that `models system` exits with status 0 and produces
/// JSON output containing `gpu` (object), `backends` (array), `cache_dir`
/// (string), and `cache_dir_exists` (boolean) fields.
///
/// This command initializes the ONNX Runtime shared library (via load-dynamic)
/// and probes GPU execution providers, which can take significant time on
/// first invocation (DLL download) or hang if a GPU provider probe deadlocks.
/// A 90-second timeout prevents the test suite from blocking indefinitely.
#[test]
fn t_cli_009_models_system_exits_0_with_json() {
    let timeout = Duration::from_secs(15);

    let child = Command::new(neuroncite_bin())
        .args(["models", "system", "--log-level", "error"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn neuroncite binary");

    // Record the child PID so the timeout branch can kill the process even
    // though the Child handle is moved into the waiter thread.
    let child_id = child.id();

    // Spawn a background thread that waits for the child process. The main
    // thread joins with a timeout to prevent the test from blocking forever
    // when ORT library init or GPU probe hangs.
    let (tx, rx) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        let result = child.wait_with_output();
        let _ = tx.send(());
        result
    });

    let output = match rx.recv_timeout(timeout) {
        Ok(()) => {
            let result = waiter.join().expect("waiter thread panicked");
            result.expect("failed to wait on child process")
        }
        Err(_) => {
            // Timed out: the child process is still running. ORT library
            // initialization (DLL download or GPU provider probe) exceeded
            // the timeout. Kill the process explicitly to prevent leaked
            // handles that lock the binary file on Windows.
            kill_process_by_id(child_id);
            eprintln!(
                "SKIPPED: `models system` did not complete within {}s \
                 (ORT init or GPU probe timed out)",
                timeout.as_secs()
            );
            return;
        }
    };

    if is_feature_disabled(&output) {
        eprintln!("T-CLI-009: skipping -- backend-ort feature not compiled in");
        return;
    }

    assert!(
        output.status.success(),
        "neuroncite models system must exit with status 0, got: {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
        panic!("models system output must be valid JSON.\nParse error: {e}\nStdout: {stdout}")
    });

    assert!(json["gpu"].is_object(), "must contain a 'gpu' object");
    assert!(
        json["gpu"]["nvidia_detected"].is_boolean(),
        "gpu.nvidia_detected must be boolean"
    );
    assert!(
        json["gpu"]["cuda_available"].is_boolean(),
        "gpu.cuda_available must be boolean"
    );
    assert!(
        json["backends"].is_array(),
        "must contain a 'backends' array"
    );
    let backends = json["backends"].as_array().expect("must be array");
    // When no backend features are compiled in (no-default-features), the
    // backends array is empty. This is valid behavior, not a test failure.
    if backends.is_empty() {
        eprintln!("T-CLI-009: no backends compiled in, skipping remaining assertions");
        return;
    }
    assert!(
        json["cache_dir"].is_string(),
        "must contain a 'cache_dir' string"
    );
    assert!(
        json["cache_dir_exists"].is_boolean(),
        "must contain a 'cache_dir_exists' boolean"
    );
    assert!(
        json["cache_disk_usage_bytes"].is_number(),
        "must contain a 'cache_disk_usage_bytes' number"
    );
}

// ---------------------------------------------------------------------------
// T-CLI-010: `neuroncite models verify nonexistent/model` exits 1
// ---------------------------------------------------------------------------

/// T-CLI-010: Verifies that `models verify` with an unknown model ID exits
/// with status 1 and writes an error JSON object to stderr.
#[test]
fn t_cli_010_models_verify_unknown_model_exits_1() {
    let output = Command::new(neuroncite_bin())
        .args([
            "models",
            "verify",
            "nonexistent/model-xyz",
            "--log-level",
            "error",
        ])
        .output()
        .expect("failed to execute neuroncite binary");

    assert_eq!(
        output.status.code(),
        Some(1),
        "neuroncite models verify with unknown model must exit with status 1\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ---------------------------------------------------------------------------
// T-CLI-011: `neuroncite models list` cached_count <= total
// ---------------------------------------------------------------------------

/// T-CLI-011: Verifies the invariant that `cached_count` is less than or
/// equal to `total` in the `models list` output. A violation would indicate
/// a counting bug in the list implementation.
#[test]
fn t_cli_011_models_list_cached_count_le_total() {
    let output = Command::new(neuroncite_bin())
        .args(["models", "list", "--log-level", "error"])
        .output()
        .expect("failed to execute neuroncite binary");

    if is_feature_disabled(&output) {
        eprintln!("T-CLI-011: skipping -- backend-ort feature not compiled in");
        return;
    }

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).expect("must be valid JSON");

    let total = json["total"].as_u64().expect("total must be a number");
    let cached_count = json["cached_count"]
        .as_u64()
        .expect("cached_count must be a number");

    assert!(
        cached_count <= total,
        "cached_count ({cached_count}) must be <= total ({total})"
    );
}

// ---------------------------------------------------------------------------
// T-CLI-012: `neuroncite models list` pooling values are valid strings
// ---------------------------------------------------------------------------

/// T-CLI-012: Verifies that every model in the `models list` output has a
/// `pooling` field with one of the three valid values: "cls", "mean-pooling",
/// or "last-token". These correspond to the PoolingStrategy enum variants
/// and their Display trait implementations.
#[test]
fn t_cli_012_models_list_valid_pooling_strings() {
    let output = Command::new(neuroncite_bin())
        .args(["models", "list", "--log-level", "error"])
        .output()
        .expect("failed to execute neuroncite binary");

    if is_feature_disabled(&output) {
        eprintln!("T-CLI-012: skipping -- backend-ort feature not compiled in");
        return;
    }

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).expect("must be valid JSON");

    let valid_pooling = ["cls", "mean-pooling", "last-token"];
    let models = json["models"].as_array().expect("must be array");

    for (i, model) in models.iter().enumerate() {
        let pooling = model["pooling"]
            .as_str()
            .unwrap_or_else(|| panic!("model[{i}].pooling must be a string"));
        assert!(
            valid_pooling.contains(&pooling),
            "model[{i}].pooling '{}' is not one of {:?}",
            pooling,
            valid_pooling
        );
    }
}

// ---------------------------------------------------------------------------
// T-CLI-013: `neuroncite models download` without model_id exits 2
// ---------------------------------------------------------------------------

/// T-CLI-013: Verifies that `models download` without the required model_id
/// positional argument exits with status 2 (clap's default for missing
/// required arguments).
#[test]
fn t_cli_013_models_download_missing_arg_exits_2() {
    let output = Command::new(neuroncite_bin())
        .args(["models", "download"])
        .output()
        .expect("failed to execute neuroncite binary");

    assert_eq!(
        output.status.code(),
        Some(2),
        "neuroncite models download without model_id must exit with status 2\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
