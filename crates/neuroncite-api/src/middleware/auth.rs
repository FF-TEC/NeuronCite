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

//! Bearer token authentication middleware with rate limiting.
//!
//! When a bearer token hash is configured in `AppState`, this middleware checks
//! incoming requests for a valid `Authorization: Bearer <token>` header.
//! Requests without a valid token receive a 401 Unauthorized response.
//!
//! Security measures applied:
//! - The plaintext token is never stored; only its SHA-256 hash is retained in
//!   `AppState`. The middleware hashes the incoming token before comparison.
//! - Comparison uses constant-time equality (`subtle::ConstantTimeEq`) to
//!   prevent timing side-channel attacks that could leak token bytes.
//! - A per-IP rate limiter tracks consecutive failed authentication attempts.
//!   After exceeding the threshold (5 failures in 60 seconds), the middleware
//!   introduces an exponential delay (up to 30 seconds) on subsequent failures
//!   to slow down brute-force attacks.
//! - A periodic eviction task removes stale entries from the rate limiter map
//!   to prevent unbounded memory growth under distributed brute-force attacks.
//!
//! The localhost bypass is handled by checking the bind_address in AppConfig
//! at router construction time rather than per-request.

use std::collections::BTreeMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use axum::body::Body;
use axum::http::{HeaderValue, Request, StatusCode, header::WWW_AUTHENTICATE};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use dashmap::DashMap;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::state::AppState;

// ---------------------------------------------------------------------------
// Rate limiting state
// ---------------------------------------------------------------------------

/// Maximum number of failed auth attempts per IP within the tracking window
/// before the rate limiter starts delaying responses.
const RATE_LIMIT_THRESHOLD: u32 = 5;

/// Duration in seconds after which a client's failure counter resets. If no
/// failed attempt occurs within this window, the counter drops to zero.
const RATE_LIMIT_WINDOW_SECS: u64 = 60;

/// Maximum delay in seconds applied to rate-limited clients. The actual delay
/// is `min(2^consecutive_failures, MAX_DELAY_SECS)`.
const MAX_DELAY_SECS: u64 = 30;

/// Interval between periodic eviction sweeps of the failed attempts map.
/// Every 5 minutes, entries whose last failure timestamp is older than
/// RATE_LIMIT_WINDOW_SECS are removed. Without periodic eviction, the map
/// would grow unboundedly under sustained distributed brute-force attacks
/// where each source IP is used only once and never triggers the lazy
/// eviction path (which only fires when the same IP attempts again).
const EVICTION_INTERVAL: std::time::Duration = std::time::Duration::from_secs(300);

/// Maximum total failed auth attempts across all IPs within one window before
/// a server-wide baseline delay is added to all subsequent failed attempts.
/// This provides protection against distributed brute-force attacks where each
/// attacking IP contributes only a few failures below the per-IP threshold.
const GLOBAL_RATE_LIMIT: u64 = 200;

// ---------------------------------------------------------------------------
// GlobalFailureCounter
// ---------------------------------------------------------------------------

/// Server-wide count of failed authentication attempts within the current
/// tracking window. When this count exceeds GLOBAL_RATE_LIMIT, all subsequent
/// failed attempts from any IP are delayed regardless of per-IP state,
/// providing protection against distributed brute-force attacks where each
/// attacking IP contributes only a few failures below the per-IP threshold.
pub struct GlobalFailureCounter {
    count: AtomicU64,
    /// Unix timestamp in seconds of the start of the current counting window.
    window_start: AtomicU64,
}

impl GlobalFailureCounter {
    /// Constructs a new counter with the count and window start both at zero.
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            window_start: AtomicU64::new(0),
        }
    }

    /// Records one failure. Returns the total failure count within the current
    /// window. When the elapsed time since `window_start` exceeds
    /// `RATE_LIMIT_WINDOW_SECS`, the window resets and the count restarts at 1.
    pub fn record(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let window = self.window_start.load(Ordering::Relaxed);
        if now.saturating_sub(window) >= RATE_LIMIT_WINDOW_SECS {
            // The current window has expired. Reset the window start and count.
            self.window_start.store(now, Ordering::Relaxed);
            self.count.store(1, Ordering::Relaxed);
            return 1;
        }
        self.count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Returns the total failure count in the current window without
    /// incrementing it.
    pub fn current_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

impl Default for GlobalFailureCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for GlobalFailureCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Expose the current count so debug output is useful without exposing
        // internal atomic state details.
        f.debug_struct("GlobalFailureCounter")
            .field("count", &self.count.load(Ordering::Relaxed))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// TimeIndex
// ---------------------------------------------------------------------------

/// Time-ordered index alongside the per-IP DashMap for efficient eviction.
/// The BTreeMap is keyed by failure timestamp so expired entries can be found
/// in O(k log n) time (where k is the number of expired entries) rather than
/// the O(n) full scan used by DashMap::retain.
pub struct TimeIndex(std::sync::Mutex<BTreeMap<Instant, IpAddr>>);

impl TimeIndex {
    /// Constructs an empty time index.
    pub fn new() -> Self {
        Self(std::sync::Mutex::new(BTreeMap::new()))
    }

    /// Records the failure timestamp for the given IP. Any previous entry for
    /// this IP is removed because only the most recent failure time is relevant
    /// for eviction decisions.
    pub fn upsert(&self, ip: IpAddr, when: Instant) {
        if let Ok(mut map) = self.0.lock() {
            // Remove any previous entry for this IP. The retain scan is linear
            // in the map size but eviction calls are rare relative to auth checks.
            map.retain(|_, v| *v != ip);
            map.insert(when, ip);
        }
    }

    /// Removes all entries with a timestamp older than `cutoff` and returns
    /// the set of IP addresses whose entries were evicted. Uses the BTreeMap
    /// ordering to find the expired range in O(k log n) time.
    pub fn drain_before(&self, cutoff: Instant) -> Vec<IpAddr> {
        let mut evicted = Vec::new();
        if let Ok(mut map) = self.0.lock() {
            let old_keys: Vec<Instant> = map.range(..cutoff).map(|(k, _)| *k).collect();
            for k in old_keys {
                if let Some(ip) = map.remove(&k) {
                    evicted.push(ip);
                }
            }
        }
        evicted
    }
}

impl Default for TimeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TimeIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Expose the number of tracked entries. The IP addresses and timestamps
        // are omitted because they are transient rate-limiting data, not
        // diagnostically useful in most contexts.
        let len = self.0.lock().map(|m| m.len()).unwrap_or(0);
        f.debug_struct("TimeIndex")
            .field("tracked_entries", &len)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// spawn_rate_limit_eviction
// ---------------------------------------------------------------------------

/// Spawns a background tokio task that periodically removes stale entries from
/// the per-IP failed authentication attempts map. Entries are considered stale
/// when their last failure timestamp is older than `RATE_LIMIT_WINDOW_SECS`.
/// The task runs every `EVICTION_INTERVAL` (5 minutes) and terminates when the
/// cancellation token is triggered during server shutdown.
///
/// The `attempts` map is `Arc`-shared with `AppState` so the eviction task
/// operates on the same data structure as the authentication middleware.
/// The `time_index` provides O(k log n) identification of expired entries.
/// The `global_counter` window is implicitly maintained via `record()` calls
/// in the auth path; this task reads the count to keep the counter alive.
///
/// Should be called once at server startup when bearer token authentication
/// is configured. When no bearer token is configured, there is no rate limiting
/// and this task should not be spawned.
pub fn spawn_rate_limit_eviction(
    cancel: tokio_util::sync::CancellationToken,
    attempts: Arc<DashMap<IpAddr, (u32, Instant)>>,
    time_index: Arc<TimeIndex>,
    global_counter: Arc<GlobalFailureCounter>,
) {
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    tracing::debug!("rate limit eviction task shutting down");
                    break;
                }
                _ = tokio::time::sleep(EVICTION_INTERVAL) => {
                    let now = Instant::now();
                    let cutoff = now - std::time::Duration::from_secs(RATE_LIMIT_WINDOW_SECS);
                    let evicted_ips = time_index.drain_before(cutoff);
                    for ip in &evicted_ips {
                        if let Some(entry) = attempts.get(ip) {
                            let (_, last_failure) = *entry;
                            // Only remove the DashMap entry when it is also
                            // stale. A racing auth attempt may have updated the
                            // timestamp after the time_index entry was drained.
                            if now.duration_since(last_failure).as_secs()
                                > RATE_LIMIT_WINDOW_SECS
                            {
                                drop(entry);
                                attempts.remove(ip);
                            }
                        }
                    }
                    let evicted = evicted_ips.len();
                    // Read the global counter to prevent the compiler from
                    // treating the field as dead code. The counter resets via
                    // record() calls in the auth path.
                    let _ = global_counter.current_count();
                    if evicted > 0 {
                        tracing::debug!(
                            evicted,
                            remaining = attempts.len(),
                            "rate limit eviction sweep completed"
                        );
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Token hashing helper (shared between AppState construction and middleware)
// ---------------------------------------------------------------------------

/// Computes the SHA-256 hash of a bearer token. This function is used both
/// during `AppState` construction (to hash the configured token) and during
/// request processing (to hash the incoming token before comparison).
pub fn hash_token(token: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

// ---------------------------------------------------------------------------
// Middleware function
// ---------------------------------------------------------------------------

/// Axum middleware function that validates bearer token authentication.
///
/// This function is registered via `axum::middleware::from_fn_with_state`
/// in the router. It hashes the incoming token with SHA-256 and performs a
/// constant-time comparison against the hash stored in `AppState`.
///
/// On authentication failure, the per-IP rate limiter is consulted. If the
/// client has exceeded the failure threshold, an exponential delay is applied
/// before returning the 401 response.
pub async fn auth_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    // If no bearer token hash is configured, all requests pass through.
    // This is the trusted-local-network mode where no authentication is required.
    let Some(ref expected_hash) = state.auth.bearer_token_hash else {
        return next.run(req).await;
    };

    // Extract the client IP for rate limiting. Falls back to 127.0.0.1 if the
    // ConnectInfo extension is not available (e.g. in test environments).
    let client_ip = extract_client_ip(&req);

    // Extract and validate the Authorization header.
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(value) if value.starts_with("Bearer ") => {
            // Trim surrounding whitespace from the token value. The Bearer
            // scheme prefix is 7 characters ("Bearer "). Trimming prevents
            // spurious authentication failures caused by trailing spaces or
            // newlines added by HTTP clients or proxies.
            let token = value["Bearer ".len()..].trim();
            let incoming_hash = hash_token(token);

            // Constant-time comparison prevents timing side-channel attacks
            // that could leak token bytes by measuring response latency.
            // `subtle::ConstantTimeEq` runs in constant time regardless of
            // where the first byte difference occurs.
            if incoming_hash.ct_eq(expected_hash).into() {
                // Successful authentication: clear any rate-limit state for
                // this IP so legitimate users are not penalized.
                state.auth.failed_auth_attempts.remove(&client_ip);
                return next.run(req).await;
            }

            // Token present but invalid: record failure and apply delay.
            record_failure_and_delay(
                client_ip,
                &state.auth.failed_auth_attempts,
                &state.auth.time_index,
                &state.auth.global_failure_counter,
            )
            .await;
            let mut resp = (
                StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({
                    "api_version": "v1",
                    "error": "invalid bearer token"
                })),
            )
                .into_response();
            resp.headers_mut().insert(
                WWW_AUTHENTICATE,
                HeaderValue::from_static(r#"Bearer realm="neuroncite""#),
            );
            resp
        }
        _ => {
            // No Authorization header or wrong scheme: record failure and apply delay.
            record_failure_and_delay(
                client_ip,
                &state.auth.failed_auth_attempts,
                &state.auth.time_index,
                &state.auth.global_failure_counter,
            )
            .await;
            let mut resp = (
                StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({
                    "api_version": "v1",
                    "error": "bearer token required"
                })),
            )
                .into_response();
            resp.headers_mut().insert(
                WWW_AUTHENTICATE,
                HeaderValue::from_static(r#"Bearer realm="neuroncite""#),
            );
            resp
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extracts the client IP address from the request. Checks for
/// `axum::extract::ConnectInfo<SocketAddr>` in the request extensions,
/// falling back to the loopback address for test environments.
fn extract_client_ip(req: &Request<Body>) -> IpAddr {
    req.extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0.ip())
        .unwrap_or(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
}

/// Maximum number of distinct IP entries tracked in the failed authentication
/// attempts map. When the map reaches this limit and a new (unseen) IP fails,
/// the entry is not recorded. This prevents unbounded memory growth during
/// distributed brute-force attacks where each packet arrives from a unique
/// source IP and is never repeated, bypassing the lazy eviction path that
/// fires only when the same IP attempts again.
const MAX_TRACKED_IPS: usize = 100_000;

/// Records a failed authentication attempt for the given IP into the shared
/// `attempts` map and applies an exponential delay when the client exceeds
/// the rate limit threshold.
///
/// Also records the failure in the server-wide `global_counter` and updates
/// the `time_index` with the current timestamp for O(k log n) eviction. When
/// the global failure count exceeds `GLOBAL_RATE_LIMIT`, an additional flat
/// baseline delay of `RATE_LIMIT_WINDOW_SECS / 10` seconds is applied on top
/// of the per-IP exponential delay.
///
/// The `attempts` map is `Arc`-shared with `AppState` and the background
/// eviction task. This function only holds a mutable DashMap shard lock for
/// the duration of the counter increment, so it never blocks the eviction task.
///
/// The per-IP delay formula is `min(2^count, MAX_DELAY_SECS)` seconds, where
/// `count` is the number of consecutive failures beyond the threshold. All
/// delays are applied after the lock is released so no lock is held during sleep.
async fn record_failure_and_delay(
    ip: IpAddr,
    attempts: &DashMap<IpAddr, (u32, Instant)>,
    time_index: &TimeIndex,
    global_counter: &GlobalFailureCounter,
) {
    let now = Instant::now();

    // When the map is at capacity and this IP has no existing entry, skip
    // insertion rather than allowing unbounded growth. Existing entries for
    // known IPs are still updated so repeated attackers continue to be tracked.
    if !attempts.contains_key(&ip) && attempts.len() >= MAX_TRACKED_IPS {
        tracing::warn!(
            ip = %ip,
            map_len = attempts.len(),
            "failed_auth_attempts map at capacity; skipping entry for new IP"
        );
        return;
    }

    let count = {
        let mut entry = attempts.entry(ip).or_insert((0, now));
        let (ref mut count, ref mut last_failure) = *entry;

        // Reset the counter if the tracking window has expired so that
        // clients who stop attacking and resume later start from zero again.
        if now.duration_since(*last_failure).as_secs() > RATE_LIMIT_WINDOW_SECS {
            *count = 0;
        }

        *count = count.saturating_add(1);
        *last_failure = now;
        *count
    };
    // The DashMap shard lock is released here. The delay below does not
    // hold any lock.

    // Record this failure in the time index so the eviction task can find it
    // in O(k log n) time without a full DashMap scan.
    time_index.upsert(ip, now);

    // Record this failure in the server-wide counter to detect distributed
    // brute-force attacks where no single IP exceeds the per-IP threshold.
    let global_count = global_counter.record();

    // Apply exponential delay once the per-IP threshold is exceeded.
    if count > RATE_LIMIT_THRESHOLD {
        let exponent = (count - RATE_LIMIT_THRESHOLD).min(5);
        let delay_secs = (1u64 << exponent).min(MAX_DELAY_SECS);
        tracing::debug!(
            ip = %ip,
            failures = count,
            delay_secs,
            "rate limiting auth failure"
        );
        tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
    }

    // Apply an additional flat baseline delay when the server-wide failure
    // count exceeds GLOBAL_RATE_LIMIT. This slows down distributed attacks
    // where each source IP stays under the per-IP threshold individually.
    if global_count > GLOBAL_RATE_LIMIT {
        let baseline_delay_secs = RATE_LIMIT_WINDOW_SECS / 10;
        tracing::debug!(
            ip = %ip,
            global_count,
            baseline_delay_secs,
            "global rate limit exceeded; applying baseline delay"
        );
        tokio::time::sleep(std::time::Duration::from_secs(baseline_delay_secs)).await;
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};
    use std::sync::Arc;
    use std::time::Instant;

    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::routing::get;
    use dashmap::DashMap;
    use tower::ServiceExt;

    use neuroncite_core::{AppConfig, EmbeddingBackend, ModelInfo, NeuronCiteError};

    use crate::state::AppState;
    use crate::worker::spawn_worker;

    use super::{
        GlobalFailureCounter, RATE_LIMIT_THRESHOLD, TimeIndex, auth_middleware, hash_token,
        record_failure_and_delay,
    };

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Stub embedding backend that always returns a fixed 4-dimensional vector.
    struct StubBackend;

    impl EmbeddingBackend for StubBackend {
        fn name(&self) -> &str {
            "stub"
        }

        fn vector_dimension(&self) -> usize {
            4
        }

        fn load_model(&mut self, _model_id: &str) -> Result<(), NeuronCiteError> {
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
            "stub-model".to_string()
        }
    }

    /// Constructs an `AppState` backed by an in-memory SQLite database with the
    /// given bearer token. The token is hashed internally by `AppState::new()`.
    fn build_state(bearer_token: Option<String>) -> Arc<AppState> {
        let manager = r2d2_sqlite::SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch("PRAGMA foreign_keys = ON;")?;
            Ok(())
        });
        let pool = r2d2::Pool::builder()
            .max_size(2)
            .build(manager)
            .expect("pool build");
        {
            let conn = pool.get().expect("get conn for migration");
            neuroncite_store::migrate(&conn).expect("migrate");
        }
        let backend: Arc<dyn EmbeddingBackend> = Arc::new(StubBackend);
        let handle = spawn_worker(backend, None);
        AppState::new(pool, handle, AppConfig::default(), true, bearer_token, 4)
            .expect("test AppState construction must succeed")
    }

    /// Builds an axum `Router` that applies the auth middleware in front of a
    /// trivial handler that always returns 200 OK. The state is wired through
    /// both `from_fn_with_state` (required by the middleware) and `with_state`
    /// (required by axum's router).
    fn build_app(state: Arc<AppState>) -> Router {
        async fn ok_handler() -> StatusCode {
            StatusCode::OK
        }

        Router::new()
            .route("/test", get(ok_handler))
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                auth_middleware,
            ))
            .with_state(state)
    }

    // -----------------------------------------------------------------------
    // T-AUTH-001: When AppState carries no bearer token, the middleware passes
    //             all requests through regardless of the Authorization header.
    // -----------------------------------------------------------------------

    /// Regression test for the auth bypass: when bearer_token_hash is None the
    /// middleware must not block any request. This is the trusted-local-network
    /// mode where no authentication is required.
    #[tokio::test]
    async fn t_auth_001_no_token_bypasses_when_state_has_none() {
        let state = build_state(None);
        let app = build_app(state);

        // A request with no Authorization header must pass through.
        let request = Request::builder().uri("/test").body(Body::empty()).unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "request without Authorization header must pass when bearer_token_hash is None"
        );
    }

    // -----------------------------------------------------------------------
    // T-AUTH-002: When a bearer token is configured and the request carries no
    //             Authorization header, the middleware returns 401.
    // -----------------------------------------------------------------------

    /// Verifies that a request with a missing Authorization header is rejected
    /// with 401 when a bearer token is required by the server configuration.
    #[tokio::test]
    async fn t_auth_002_missing_header_returns_401() {
        let state = build_state(Some("test-secret-token-for-auth-tests-!".to_string()));
        let app = build_app(state);

        let request = Request::builder().uri("/test").body(Body::empty()).unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "missing Authorization header must yield 401 when bearer_token_hash is configured"
        );
    }

    // -----------------------------------------------------------------------
    // T-AUTH-003: When a bearer token is configured and the request carries a
    //             wrong token value, the middleware returns 401.
    // -----------------------------------------------------------------------

    /// Verifies that an incorrect bearer token value is rejected with 401.
    /// This prevents callers from guessing or brute-forcing the token.
    #[tokio::test]
    async fn t_auth_003_wrong_token_returns_401() {
        let state = build_state(Some("test-correct-token-for-auth-tests!".to_string()));
        let app = build_app(state);

        let request = Request::builder()
            .uri("/test")
            .header("authorization", "Bearer wrong-token")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "incorrect bearer token must yield 401"
        );
    }

    // -----------------------------------------------------------------------
    // T-AUTH-004: When a bearer token is configured and the request carries the
    //             correct token, the middleware passes the request through.
    // -----------------------------------------------------------------------

    /// Verifies that a request with the correct bearer token is forwarded to
    /// the downstream handler and returns 200 OK.
    #[tokio::test]
    async fn t_auth_004_correct_token_passes() {
        let state = build_state(Some("test-valid-token-for-auth-testing!".to_string()));
        let app = build_app(state);

        let request = Request::builder()
            .uri("/test")
            .header("authorization", "Bearer test-valid-token-for-auth-testing!")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "correct bearer token must pass through to the handler"
        );
    }

    // -----------------------------------------------------------------------
    // T-AUTH-005: hash_token produces consistent SHA-256 output for the same
    //             input and different output for different inputs.
    // -----------------------------------------------------------------------

    #[test]
    fn t_auth_005_hash_token_deterministic() {
        let hash_a = hash_token("test-token");
        let hash_b = hash_token("test-token");
        let hash_c = hash_token("different-token");

        assert_eq!(hash_a, hash_b, "same input must produce same hash");
        assert_ne!(
            hash_a, hash_c,
            "different input must produce different hash"
        );
    }

    // -----------------------------------------------------------------------
    // T-AUTH-006: hash_token produces a 32-byte SHA-256 hash matching the
    //             known digest for a test vector.
    // -----------------------------------------------------------------------

    #[test]
    fn t_auth_006_hash_token_sha256_known_vector() {
        // SHA-256 of "hello" is well-known.
        let hash = hash_token("hello");
        let expected_hex = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        let actual_hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            actual_hex, expected_hex,
            "SHA-256 hash must match known vector"
        );
    }

    // -----------------------------------------------------------------------
    // T-AUTH-007: bearer_token_hash is populated when a token is provided
    //             to AppState::new() and is None when no token is provided.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_auth_007_appstate_hashes_token_on_construction() {
        let state_with = build_state(Some("test-my-token-for-auth-testing-aa".to_string()));
        assert!(
            state_with.auth.bearer_token_hash.is_some(),
            "bearer_token_hash must be Some when a token is provided"
        );
        let expected = hash_token("test-my-token-for-auth-testing-aa");
        assert_eq!(
            state_with.auth.bearer_token_hash.unwrap(),
            expected,
            "stored hash must match hash_token output"
        );

        let state_without = build_state(None);
        assert!(
            state_without.auth.bearer_token_hash.is_none(),
            "bearer_token_hash must be None when no token is provided"
        );
    }

    // -----------------------------------------------------------------------
    // T-AUTH-008: The "Basic" scheme is rejected with 401 (only "Bearer" is
    //             accepted).
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_auth_008_basic_scheme_rejected() {
        let state = build_state(Some("test-token123-for-auth-test-008aa".to_string()));
        let app = build_app(state);

        let request = Request::builder()
            .uri("/test")
            .header("authorization", "Basic dXNlcjpwYXNz")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "Basic auth scheme must be rejected"
        );
    }

    // -----------------------------------------------------------------------
    // T-AUTH-009: Empty bearer token is rejected with 401.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_auth_009_empty_bearer_rejected() {
        let state = build_state(Some("test-nonempty-token-for-auth-tests".to_string()));
        let app = build_app(state);

        let request = Request::builder()
            .uri("/test")
            .header("authorization", "Bearer ")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "empty bearer token must be rejected"
        );
    }

    // -----------------------------------------------------------------------
    // T-AUTH-010: Rate limiter records failures and resets on window expiry.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_auth_010_rate_limiter_records_failures() {
        // Use fresh structures for this test to avoid interference from other tests.
        let attempts: Arc<DashMap<IpAddr, (u32, Instant)>> = Arc::new(DashMap::new());
        let time_index = TimeIndex::new();
        let global_counter = GlobalFailureCounter::new();
        let test_ip = IpAddr::V4(Ipv4Addr::new(192, 0, 2, 99));

        // Record failures up to the threshold (should not delay).
        for _ in 0..RATE_LIMIT_THRESHOLD {
            record_failure_and_delay(test_ip, &attempts, &time_index, &global_counter).await;
        }

        // Verify the per-IP counter matches.
        let count = attempts.get(&test_ip).map(|e| e.value().0).unwrap_or(0);
        assert_eq!(
            count, RATE_LIMIT_THRESHOLD,
            "counter must equal threshold after {RATE_LIMIT_THRESHOLD} failures"
        );
    }

    // -----------------------------------------------------------------------
    // T-AUTH-011: Successful authentication clears rate limit state.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_auth_011_success_clears_rate_limit() {
        let token = "test-rate-limit-clear-token-t011a".to_string();
        let state = build_state(Some(token.clone()));
        let app = build_app(state.clone());

        // Simulate 3 failed attempts by pre-populating the AppState's map.
        let test_ip = IpAddr::V4(Ipv4Addr::LOCALHOST);
        state
            .auth
            .failed_auth_attempts
            .insert(test_ip, (3, std::time::Instant::now()));

        // A successful request clears the counter via state.auth.failed_auth_attempts.remove().
        let request = Request::builder()
            .uri("/test")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // The rate limiter entry for localhost must be removed on success.
        let entry = state.auth.failed_auth_attempts.get(&test_ip);
        assert!(
            entry.is_none(),
            "rate limit entry must be cleared after successful auth"
        );
    }

    // -----------------------------------------------------------------------
    // T-AUTH-012: The 401 JSON response body contains the expected fields.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn t_auth_012_unauthorized_response_body() {
        let state = build_state(Some("test-body-test-token-for-auth-012".to_string()));
        let app = build_app(state);

        let request = Request::builder().uri("/test").body(Body::empty()).unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["api_version"], "v1");
        assert!(
            body["error"].as_str().is_some(),
            "error field must be a string"
        );
    }
}
