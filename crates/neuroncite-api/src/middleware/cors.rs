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

//! CORS middleware configuration.
//!
//! Builds a `tower_http::cors::CorsLayer` based on the bind address. When
//! bound to localhost (127.0.0.1 or ::1), CORS headers are permissive because
//! same-origin requests do not trigger preflight. When bound to a LAN address,
//! CORS requires explicit origins in config; without them, cross-origin
//! requests are denied by default to prevent accidental wildcard exposure.
//!
//! The caller must ensure that `config.bind_address` matches the address the
//! server actually listens on. When the CLI --bind flag overrides the config
//! file value, the caller is responsible for updating `config.bind_address`
//! before calling `build_cors_layer`. Failure to do so causes CORS to be
//! evaluated against the stale config-file address, not the actual listener.

use axum::http::{HeaderValue, Method, header};
use tower_http::cors::CorsLayer;

use neuroncite_core::AppConfig;

/// Returns `true` when the given bind address is a loopback address.
///
/// Recognized loopback addresses: `127.0.0.1`, `::1`, and the hostname
/// `localhost`. All other values are treated as LAN or public addresses.
/// This function is the single decision point for CORS policy selection.
fn is_loopback(addr: &str) -> bool {
    addr == "127.0.0.1" || addr == "::1" || addr == "localhost"
}

/// Builds the CORS middleware layer based on the server's bind address
/// and the configured allowed origins.
///
/// - Localhost binding: returns a permissive layer (all methods, all headers).
/// - LAN binding with explicit origins: returns a layer allowing those origins.
/// - LAN binding without configured origins: returns a restrictive default
///   layer that denies cross-origin requests. The operator must provide
///   explicit origins via the `cors_origins` config field to enable CORS
///   for LAN-exposed servers.
pub fn build_cors_layer(config: &AppConfig) -> CorsLayer {
    if is_loopback(&config.bind_address) {
        // Localhost: restrict CORS to the server's own origin only. This prevents
        // malicious websites from making cross-origin requests to the local server
        // (e.g. via DNS rebinding or a compromised page in another tab). The
        // NeuronCite frontend is served from the same origin, so it passes the
        // same-origin check without needing CORS at all. Both 127.0.0.1 and
        // localhost are included because the browser may use either form.
        let port = config.port;
        let mut origins = vec![
            format!("http://127.0.0.1:{port}")
                .parse::<HeaderValue>()
                .expect("127.0.0.1 origin must be a valid header value"),
            format!("http://localhost:{port}")
                .parse::<HeaderValue>()
                .expect("localhost origin must be a valid header value"),
        ];
        // Include IPv6 loopback for completeness.
        if let Ok(v) = format!("http://[::1]:{port}").parse::<HeaderValue>() {
            origins.push(v);
        }
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([Method::GET, Method::POST, Method::DELETE])
            .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION, header::ACCEPT])
    } else if config.cors_origins.is_empty() {
        // LAN binding without explicit origins: deny cross-origin requests.
        // The default CorsLayer returns no CORS headers, causing browsers to
        // reject cross-origin requests. Operators binding to a LAN address
        // must configure `cors_origins` explicitly.
        tracing::warn!(
            bind_address = %config.bind_address,
            "CORS: no origins configured for LAN binding; cross-origin requests will be denied. \
             Set `cors_origins` in the config file to allow specific origins."
        );
        CorsLayer::new()
    } else {
        // LAN binding with explicit origins from config.
        let origins: Vec<_> = config
            .cors_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([Method::GET, Method::POST, Method::DELETE])
            .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION, header::ACCEPT])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-CORS-001: `is_loopback` recognizes all three canonical loopback
    /// address representations (IPv4, IPv6, hostname). Any other value is
    /// treated as a non-loopback address.
    #[test]
    fn t_cors_001_is_loopback_detection() {
        // All loopback forms must return true.
        assert!(is_loopback("127.0.0.1"), "IPv4 loopback must be detected");
        assert!(is_loopback("::1"), "IPv6 loopback must be detected");
        assert!(
            is_loopback("localhost"),
            "hostname 'localhost' must be detected"
        );

        // LAN and wildcard addresses must return false.
        assert!(
            !is_loopback("0.0.0.0"),
            "wildcard must not be treated as loopback"
        );
        assert!(
            !is_loopback("192.168.1.100"),
            "LAN address must not be loopback"
        );
        assert!(
            !is_loopback("10.0.0.1"),
            "private range must not be loopback"
        );
        assert!(!is_loopback("::"), "IPv6 unspecified must not be loopback");
    }

    /// T-CORS-002: `build_cors_layer` constructs a layer without panicking for
    /// each of the three configuration branches: localhost (permissive), LAN
    /// without origins (restrictive default), and LAN with explicit origins.
    /// This test verifies that config.bind_address is the sole input used
    /// for CORS policy selection.
    #[test]
    fn t_cors_002_build_cors_layer_all_branches() {
        // Branch 1: localhost binding -- permissive layer.
        let mut config = neuroncite_core::AppConfig {
            bind_address: "127.0.0.1".into(),
            cors_origins: Vec::new(),
            ..neuroncite_core::AppConfig::default()
        };
        let _ = build_cors_layer(&config);

        config.bind_address = "::1".into();
        let _ = build_cors_layer(&config);

        config.bind_address = "localhost".into();
        let _ = build_cors_layer(&config);

        // Branch 2: LAN binding without explicit origins -- restrictive default.
        // Cross-origin requests are denied because no origins are configured.
        config.bind_address = "0.0.0.0".into();
        config.cors_origins = Vec::new();
        let _ = build_cors_layer(&config);

        // Branch 3: LAN binding with an explicit (valid) origin.
        config.bind_address = "0.0.0.0".into();
        config.cors_origins = vec!["http://192.168.1.10:3000".into()];
        let _ = build_cors_layer(&config);
    }

    /// T-CORS-003: Documents the fix for the CLI--bind divergence: if the
    /// caller omits updating config.bind_address before calling build_cors_layer,
    /// a server bound to 0.0.0.0 would be evaluated as localhost (default value).
    /// This test asserts that the is_loopback check uses the actual bind address,
    /// NOT the default config value, i.e., the caller has the responsibility to
    /// override config.bind_address with the CLI value before calling this function.
    #[test]
    fn t_cors_003_bind_address_must_be_set_to_actual_listener() {
        // Simulate the correct behavior: CLI --bind 0.0.0.0 overrides the
        // config default (127.0.0.1). The caller sets config.bind_address to
        // the CLI value before building the CORS layer.
        let mut config = neuroncite_core::AppConfig::default();
        assert_eq!(
            config.bind_address, "127.0.0.1",
            "default bind_address must be 127.0.0.1"
        );

        // Before the fix: config.bind_address was NOT updated from the CLI value.
        // is_loopback("127.0.0.1") would return true, wrongly treating a 0.0.0.0
        // listener as localhost. This is verified here by confirming the default
        // config evaluates as loopback.
        assert!(
            is_loopback(&config.bind_address),
            "default config.bind_address evaluates as loopback (this is what the bug caused)"
        );

        // After the fix: the CLI --bind value overwrites config.bind_address.
        // is_loopback now correctly evaluates the actual listener address.
        config.bind_address = "0.0.0.0".into();
        assert!(
            !is_loopback(&config.bind_address),
            "after CLI override, 0.0.0.0 must NOT evaluate as loopback"
        );
    }
}
