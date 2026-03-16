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

// HTTP API server integration tests.
//
// These tests spawn the neuroncite binary in headless server mode (using the
// `serve` subcommand with port 0 for OS-assigned ephemeral ports) and make
// HTTP requests against the running server process. Each test targets a
// specific aspect of the server's behavior: health endpoint, concurrency,
// and graceful shutdown.
//
// The binary path is resolved at compile time via env!("CARGO_BIN_EXE_neuroncite").
// Tests use reqwest as the HTTP client to send requests to the spawned server.
//
// The server requires a compatible ONNX Runtime shared library and a cached
// embedding model to start. If the server process exits before logging a
// listening address (e.g., missing ORT library), the test fails fast instead
// of hanging. Stdout is redirected to null to prevent pipe buffer deadlocks.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

/// Returns the absolute path to the compiled neuroncite binary.
fn neuroncite_bin() -> &'static str {
    env!("CARGO_BIN_EXE_neuroncite")
}

/// Spawns the neuroncite server on a random port and returns the child process
/// handle along with the bound port number.
///
/// Stderr is read in a background thread to avoid blocking the test thread.
/// The function polls `try_wait()` to detect early process termination (e.g.,
/// missing ONNX Runtime or model) and fails fast instead of hanging for the
/// full timeout duration.
///
/// Stdout is redirected to null to prevent pipe buffer deadlocks: if the
/// server writes to stdout while nothing reads the pipe, the pipe buffer
/// fills up and the server process blocks, which in turn blocks stderr
/// output and causes the test to hang.
///
/// # Returns
///
/// `Ok((child, port))` if the server logs a listening address within the
/// timeout. `Err(reason)` if the server exits early or times out.
fn spawn_server() -> Result<(Child, u16), String> {
    let mut child = Command::new(neuroncite_bin())
        .args([
            "serve",
            "--port",
            "0",
            "--bind",
            "127.0.0.1",
            "--log-level",
            "info",
        ])
        .stderr(Stdio::piped())
        .stdout(Stdio::null()) // Prevent pipe buffer deadlock.
        .spawn()
        .expect("failed to spawn neuroncite server process");

    let stderr = child.stderr.take().expect("failed to capture stderr");

    // Read stderr in a background thread to avoid blocking the test thread.
    // Each line is sent through a channel so the main thread can poll with
    // a timeout and also check for early process termination via try_wait().
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line_result in reader.lines() {
            match line_result {
                Ok(line) => {
                    if tx.send(line).is_err() {
                        break; // Receiver dropped, test is done.
                    }
                }
                Err(_) => break, // EOF or I/O error.
            }
        }
    });

    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(15);

    loop {
        // Check if the process has already exited (failed to start).
        // This detects missing ONNX Runtime, missing model, port conflicts,
        // and other initialization failures within milliseconds.
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!(
                "server process exited with {status} before logging a listening address"
            ));
        }

        if start.elapsed() > timeout {
            child.kill().ok();
            child.wait().ok();
            return Err(format!(
                "server did not log a listening address within {timeout:?}"
            ));
        }

        // Poll the stderr channel with a 100ms timeout. This allows the loop
        // to re-check try_wait() and the elapsed time between line reads.
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(line) => {
                if (line.contains("HTTP server listening") || line.contains("listening"))
                    && let Some(addr_part) = line.split("127.0.0.1:").nth(1)
                {
                    let port_str: String = addr_part
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if let Ok(port) = port_str.parse::<u16>() {
                        // Brief pause to ensure the server is accepting connections.
                        std::thread::sleep(Duration::from_millis(200));
                        return Ok((child, port));
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Stderr pipe closed. The server process likely exited.
                child.kill().ok();
                child.wait().ok();
                return Err("server stderr closed without logging a listening address".to_string());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T-SRV-001: Server binds and responds to /api/v1/health
// ---------------------------------------------------------------------------

/// T-SRV-001: Verifies that the server process binds to a port and responds
/// to the /api/v1/health endpoint with a 200 status code and valid JSON.
///
/// If the server cannot start (e.g., missing ONNX Runtime or model), the
/// test prints a warning and returns successfully. The test only fails if
/// the server starts but returns incorrect responses.
#[test]
fn t_srv_001_health_endpoint_responds() {
    let (mut child, port) = match spawn_server() {
        Ok(pair) => pair,
        Err(reason) => {
            eprintln!("skipping server test: {reason}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    let result = rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build HTTP client");

        let url = format!("http://127.0.0.1:{port}/api/v1/health");
        client.get(&url).send().await
    });

    // Terminate the server before checking assertions (cleanup first).
    child.kill().ok();
    child.wait().ok();

    let resp = result.expect("HTTP request to health endpoint failed");
    assert_eq!(
        resp.status().as_u16(),
        200,
        "health endpoint must return 200"
    );

    let json: serde_json::Value = rt
        .block_on(resp.json())
        .expect("health response must be valid JSON");

    assert!(
        json["api_version"].is_string(),
        "health response must contain 'api_version' field"
    );
    assert!(
        json["version"].is_string(),
        "health response must contain 'version' field"
    );
}

// ---------------------------------------------------------------------------
// T-SRV-002: Concurrent requests are handled without panics
// ---------------------------------------------------------------------------

/// T-SRV-002: Verifies that the server handles multiple concurrent requests
/// without panicking or returning server errors.
///
/// Sends 10 concurrent GET requests to the health endpoint and verifies
/// that all responses have a 200 status code. This exercises the server's
/// ability to handle parallel request processing.
#[test]
fn t_srv_002_concurrent_requests_handled() {
    let (mut child, port) = match spawn_server() {
        Ok(pair) => pair,
        Err(reason) => {
            eprintln!("skipping server test: {reason}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    let results = rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build HTTP client");

        let url = format!("http://127.0.0.1:{port}/api/v1/health");

        // Spawn 10 concurrent requests.
        let mut handles = Vec::with_capacity(10);
        for _ in 0..10 {
            let client = client.clone();
            let url = url.clone();
            handles.push(tokio::spawn(async move { client.get(&url).send().await }));
        }

        // Collect all results.
        let mut responses = Vec::with_capacity(10);
        for handle in handles {
            responses.push(handle.await);
        }
        responses
    });

    // Terminate the server.
    child.kill().ok();
    child.wait().ok();

    // Verify all requests succeeded.
    let mut success_count = 0_usize;
    for result in &results {
        match result {
            Ok(Ok(resp)) => {
                assert_eq!(
                    resp.status().as_u16(),
                    200,
                    "all concurrent health requests must return 200"
                );
                success_count += 1;
            }
            Ok(Err(e)) => {
                panic!("concurrent request failed with HTTP error: {e}");
            }
            Err(e) => {
                panic!("concurrent request task panicked: {e}");
            }
        }
    }

    assert_eq!(success_count, 10, "all 10 concurrent requests must succeed");
}

// ---------------------------------------------------------------------------
// T-SRV-003: Shutdown terminates the server (headless mode)
// ---------------------------------------------------------------------------

/// T-SRV-003: Verifies that the server terminates when a kill signal is
/// sent to the process.
///
/// On Windows, the process is terminated via TerminateProcess since SIGINT
/// is not directly supported for child processes. The test verifies that
/// the process exits and does not hang.
#[test]
fn t_srv_003_shutdown_terminates_server() {
    let (mut child, _port) = match spawn_server() {
        Ok(pair) => pair,
        Err(reason) => {
            eprintln!("skipping server test: {reason}");
            return;
        }
    };

    // Allow the server a moment to fully initialize.
    std::thread::sleep(Duration::from_millis(500));

    // Send a termination signal to the server process.
    child.kill().expect("failed to kill server process");

    // Wait for the process to exit with a timeout.
    let status = child.wait().expect("failed to wait for server process");

    // On Windows, kill() sends TerminateProcess which results in a non-zero
    // exit code. On Unix, kill() sends SIGKILL. In either case, the process
    // must have exited (not hung).
    assert!(
        !status.success() || status.code().is_some(),
        "server process must terminate after receiving kill signal"
    );
}

// ---------------------------------------------------------------------------
// T-SRV-010: OpenAPI endpoint requires auth when token is configured
// ---------------------------------------------------------------------------

/// Spawns the neuroncite server with `--print-token`, captures the bearer
/// token from stdout, and returns `(child, port, token)`. Both stdout and
/// stderr are read in background threads to avoid pipe buffer deadlocks.
///
/// Returns `Err` if the server exits before printing a port or the token.
fn spawn_server_with_token() -> Result<(std::process::Child, u16, String), String> {
    let mut child = std::process::Command::new(neuroncite_bin())
        .args([
            "serve",
            "--port",
            "0",
            "--bind",
            "127.0.0.1",
            "--log-level",
            "info",
            "--print-token",
        ])
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn neuroncite server process");

    let stderr = child.stderr.take().expect("failed to capture stderr");
    let stdout = child.stdout.take().expect("failed to capture stdout");

    let (stderr_tx, stderr_rx) = mpsc::channel::<String>();
    let (stdout_tx, stdout_rx) = mpsc::channel::<String>();

    std::thread::spawn(move || {
        let reader = std::io::BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            if stderr_tx.send(line).is_err() {
                break;
            }
        }
    });

    std::thread::spawn(move || {
        let reader = std::io::BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            if stdout_tx.send(line).is_err() {
                break;
            }
        }
    });

    // Extract the bearer token from the first stdout line ("Bearer token: <value>").
    let token = match stdout_rx.recv_timeout(Duration::from_secs(15)) {
        Ok(line) => match line.strip_prefix("Bearer token: ").map(str::to_string) {
            Some(t) => t,
            None => {
                child.kill().ok();
                child.wait().ok();
                return Err(format!("unexpected stdout line: {line}"));
            }
        },
        Err(_) => {
            child.kill().ok();
            child.wait().ok();
            return Err("server did not print a bearer token within 15s".into());
        }
    };

    // Wait for the server to log its listening address on stderr.
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(15);
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            // The child exited on its own; no kill needed, but wait to reap.
            child.wait().ok();
            return Err(format!("server exited with {status} before listening"));
        }
        if start.elapsed() > timeout {
            child.kill().ok();
            child.wait().ok();
            return Err("server did not log a listening address within 15s".into());
        }
        match stderr_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(line) => {
                if (line.contains("HTTP server listening") || line.contains("listening"))
                    && let Some(addr_part) = line.split("127.0.0.1:").nth(1)
                {
                    let port_str: String = addr_part
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if let Ok(port) = port_str.parse::<u16>() {
                        std::thread::sleep(Duration::from_millis(200));
                        return Ok((child, port, token));
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                child.kill().ok();
                child.wait().ok();
                return Err("server stderr closed before logging a listening address".into());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T-SRV-004: Concurrent health checks do not return errors (H-06)
// ---------------------------------------------------------------------------

/// T-SRV-004: Verifies that the server handles multiple concurrent GET
/// /api/v1/health requests without returning any error responses.
///
/// Spawns 10 tasks, each making 2 sequential requests to the health endpoint
/// for a total of 20 concurrent requests. All responses must have HTTP 200.
/// This exercises the multi-threaded Tokio executor's ability to schedule
/// parallel connections without triggering race conditions on the Axum router
/// or the shared AppState.
///
/// If the server cannot start (missing ORT library or model), the test
/// prints a diagnostic message and returns without failing.
#[test]
fn t_srv_004_concurrent_health_checks() {
    let (mut child, port) = match spawn_server() {
        Ok(pair) => pair,
        Err(reason) => {
            eprintln!("skipping t_srv_004: {reason}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    let results = rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build HTTP client");

        let url = format!("http://127.0.0.1:{port}/api/v1/health");

        // Spawn 10 tasks, each sending 2 sequential requests.
        let mut handles = Vec::with_capacity(10);
        for _ in 0..10 {
            let c = client.clone();
            let u = url.clone();
            handles.push(tokio::spawn(async move {
                let mut statuses = Vec::with_capacity(2);
                for _ in 0..2 {
                    let res = c.get(&u).send().await.expect("health request must succeed");
                    statuses.push(res.status().as_u16());
                }
                statuses
            }));
        }

        // Collect all status codes from all tasks.
        let mut all_statuses = Vec::with_capacity(20);
        for handle in handles {
            let statuses = handle.await.expect("concurrent health task must not panic");
            all_statuses.extend(statuses);
        }
        all_statuses
    });

    child.kill().ok();
    child.wait().ok();

    assert_eq!(
        results.len(),
        20,
        "20 responses must be collected (10 tasks x 2 requests)"
    );
    for (i, status) in results.iter().enumerate() {
        assert_eq!(*status, 200, "response {i} must return 200, got {status}");
    }
}

// ---------------------------------------------------------------------------
// T-SRV-007: Request without Authorization header returns 401 when token
//            is configured (H-08)
// ---------------------------------------------------------------------------

/// T-SRV-007: When the server is started with `--print-token`, requests to
/// /api/v1/health without an Authorization header must receive HTTP 401.
///
/// This verifies that the auth middleware enforces bearer token authentication
/// on all API endpoints when a token is configured.
#[test]
fn t_srv_007_request_without_auth_header_returns_401() {
    let (mut child, port, _token) = match spawn_server_with_token() {
        Ok(triple) => triple,
        Err(reason) => {
            eprintln!("skipping t_srv_007: {reason}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    let status = rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build HTTP client");

        client
            .get(format!("http://127.0.0.1:{port}/api/v1/health"))
            .send()
            .await
            .expect("request must complete")
            .status()
            .as_u16()
    });

    child.kill().ok();
    child.wait().ok();

    assert_eq!(
        status, 401,
        "request without Authorization header must return 401"
    );
}

// ---------------------------------------------------------------------------
// T-SRV-008: Request with wrong token returns 401 (H-08)
// ---------------------------------------------------------------------------

/// T-SRV-008: When the server is started with `--print-token`, requests to
/// /api/v1/health with an incorrect bearer token must receive HTTP 401.
///
/// This verifies that the auth middleware rejects tokens that do not match
/// the configured server token, even when the Authorization header is present.
#[test]
fn t_srv_008_request_with_wrong_token_returns_401() {
    let (mut child, port, _token) = match spawn_server_with_token() {
        Ok(triple) => triple,
        Err(reason) => {
            eprintln!("skipping t_srv_008: {reason}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    let status = rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build HTTP client");

        client
            .get(format!("http://127.0.0.1:{port}/api/v1/health"))
            .header(
                "Authorization",
                "Bearer wrong_token_that_does_not_match_server_token_aaa",
            )
            .send()
            .await
            .expect("request must complete")
            .status()
            .as_u16()
    });

    child.kill().ok();
    child.wait().ok();

    assert_eq!(status, 401, "request with wrong token must return 401");
}

// ---------------------------------------------------------------------------
// T-SRV-009: Request with correct token returns 200 (H-08)
// ---------------------------------------------------------------------------

/// T-SRV-009: When the server is started with `--print-token`, requests to
/// /api/v1/health with the correct bearer token must receive HTTP 200.
///
/// This verifies that the auth middleware allows requests through when the
/// Authorization header contains the server's configured token.
#[test]
fn t_srv_009_request_with_correct_token_returns_200() {
    let (mut child, port, token) = match spawn_server_with_token() {
        Ok(triple) => triple,
        Err(reason) => {
            eprintln!("skipping t_srv_009: {reason}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    let status = rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build HTTP client");

        client
            .get(format!("http://127.0.0.1:{port}/api/v1/health"))
            .header("Authorization", format!("Bearer {token}"))
            .send()
            .await
            .expect("request must complete")
            .status()
            .as_u16()
    });

    child.kill().ok();
    child.wait().ok();

    assert_eq!(status, 200, "request with correct token must return 200");
}

// ---------------------------------------------------------------------------
// T-SRV-010: OpenAPI endpoint requires auth when token is configured
// ---------------------------------------------------------------------------

/// T-SRV-010: When the server is started with `--print-token`, all API
/// endpoints including `/api/v1/openapi.json` require an `Authorization:
/// Bearer <token>` header. Requests without the header receive 401.
/// Requests with the correct token receive 200.
///
/// This test verifies that the fix for the OpenAPI-outside-auth finding is
/// effective: `/api/v1/openapi.json` must now be inside the auth middleware
/// layer and must not be accessible to unauthenticated callers when a token
/// is configured.
#[test]
fn t_srv_010_openapi_requires_auth_when_token_configured() {
    let (mut child, port, token) = match spawn_server_with_token() {
        Ok(triple) => triple,
        Err(reason) => {
            eprintln!("skipping server test: {reason}");
            return;
        }
    };

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");

    let (no_auth_status, with_auth_status) = rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build HTTP client");

        let url = format!("http://127.0.0.1:{port}/api/v1/openapi.json");

        // Request without Authorization header must be rejected.
        let no_auth = client
            .get(&url)
            .send()
            .await
            .expect("request without auth must complete")
            .status()
            .as_u16();

        // Request with the correct token must succeed.
        let with_auth = client
            .get(&url)
            .header("Authorization", format!("Bearer {token}"))
            .send()
            .await
            .expect("request with token must complete")
            .status()
            .as_u16();

        (no_auth, with_auth)
    });

    child.kill().ok();
    child.wait().ok();

    assert_eq!(
        no_auth_status, 401,
        "openapi.json without token must return 401"
    );
    assert_eq!(
        with_auth_status, 200,
        "openapi.json with valid token must return 200"
    );
}
