# Changelog

All notable changes to NeuronCite are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-05-13

### Security

- **SSRF redirect validation:** the `neuroncite-html` HTTP client now revalidates
  every redirect target against the SSRF allow-list (RFC1918, loopback,
  link-local, multicast, and metadata IPs are blocked) instead of trusting the
  initial DNS resolution, closing a redirect-chain bypass
- **rustls-webpki 0.103.10 -> 0.103.13:** fixes RUSTSEC-2026-0098 (name
  constraints for URI names incorrectly accepted), RUSTSEC-2026-0099 (name
  constraints accepted for wildcard certificates), and RUSTSEC-2026-0104
  (reachable panic in CRL parsing)

### Changed

- **Workspace dependencies bulk update:**
  - `sha2` 0.10 -> 0.11
  - `tokio` 1.43 -> 1.52
  - `clap` 4.5 -> 4.6
  - `pdfium-render` 0.8 -> 0.9
  - `arc-swap`, `libc`, `proptest`, `semver`, `unicode-segmentation`, `uuid`
    bumped to latest compatible patch/minor via `cargo update`
- **Frontend dependencies bulk update:**
  - `@typescript-eslint/eslint-plugin` 8.57.1 -> 8.59.3
  - `@typescript-eslint/parser` 8.57.1 -> 8.59.3
  - `jsdom` 29.0.0 -> 29.1.1
  - `typescript` 6.0.0 -> 6.0.3
  - `vite` 8.0.0 -> 8.0.12
  - `vite-plugin-solid` 2.11.11 -> 2.11.12
  - `vitest` 4.1.0 -> 4.1.6
  - `eslint`, `eslint-plugin-jsx-a11y`, `eslint-plugin-solid`, `solid-js`
    bumped to latest compatible patch/minor via `npm update`
- **Docker base images:** `node:25-bookworm` -> `node:26-bookworm`,
  `rust:1.94-bookworm` -> `rust:1.95-bookworm`
- **GitHub Actions:** `softprops/action-gh-release` v2 -> v3 (Node 20 -> 24
  runtime; release input schema unchanged)
- **CI audit ignore-list:** added RUSTSEC-2026-0097 (`rand` unsound with custom
  logger; transitive via `hnsw_rs`, `lopdf`, `uuid`) and RUSTSEC-2026-0105
  (`core2` unmaintained and yanked; transitive via
  `image -> ravif -> rav1e -> bitstream-io`) - both have no upstream fix path
- **CI and pre-commit test execution:** all `cargo test` invocations now run
  with `--test-threads=1`. `pdfium-render` 0.9 loads the pdfium dynamic
  library per test invocation, and concurrent `LoadLibrary`/`FreeLibrary`
  calls abort the process on Windows and produce intermittent failures on
  Linux. Tests run serially until a shared pdfium handle is introduced

## [0.1.1] - 2026-03-17

### Added

- **MCP dual-target support:** independent registration for both Claude Code
  (`~/.claude.json`) and Claude Desktop App (`claude_desktop_config.json`) with
  `--target` CLI flag, per-target web UI status cards, and install/uninstall
  controls
- **Check for Updates button** in the Settings > About panel: queries the GitHub
  Releases API, compares semver versions, and displays a link to the latest
  release if an update is available

### Fixed

- **Linux GUI → browser mode:** skip the native wry/tao WebView on Linux
  entirely (wry 0.54 + tao 0.34 cannot reliably render under Wayland) and use
  the browser-based UI with native file dialogs via xdg-desktop-portal/zenity;
  macOS and Windows continue to use the native window
- **Stale counts and inaccurate claims** across docs and code: corrected tab
  count (6 → 7), crate count (15 → 16), CLI command count (11 → 10), Python
  client URLs, license wording, single-binary qualification, Tesseract
  auto-download claims, GUI/browser behavior, MCP tool descriptions, Linux
  runtime dependency claims, pip install instructions, and pdfium comment
- **Release notes Docker commands:** lowercased `github.repository_owner` in
  release body to prevent OCI "invalid reference format" errors
- **Duplicate MCP test IDs:** renumbered 11 colliding IDs (T-MCP-030..040 →
  T-MCP-125..135), updated `t_web_016_mcp_status_returns_fields` to match the
  current dual-target `McpStatusResponse`, and documented `handlers/update.rs`
  in the architecture module table
- **MCP protocol version:** report spec version `2024-11-05` instead of crate
  version in initialize handshake

### Changed

- Bump `@typescript-eslint/parser` and `@typescript-eslint/eslint-plugin` from
  8.57.0 to 8.57.1

## [0.1.0] - 2026-03-14

Initial release.

### Added

- Hybrid semantic search combining HNSW vector similarity, BM25 keyword matching,
  and Reciprocal Rank Fusion, with optional cross-encoder reranking
- PDF text extraction with three backends: pdf-extract (default), pdfium
  (multi-column layout), and Tesseract OCR (image-heavy pages)
- HTML web page fetching with readability-based boilerplate removal, disk caching,
  and BFS crawling with same-domain filtering
- Text chunking with four strategies: page, word-window, token-window, sentence
- Dense vector embeddings via ONNX Runtime with CUDA, DirectML, and CoreML
  execution providers
- LaTeX/BibTeX citation verification pipeline with batch claim extraction,
  two-component scoring (keyword overlap + semantic similarity), and LLM-driven
  auto-verification via Ollama
- PDF annotation pipeline with 4-stage text matching (exact, normalized, fuzzy, OCR)
- REST API server (Axum) with 21+ endpoints, OpenAPI specification, bearer token
  authentication, and Server-Sent Events for real-time progress
- SolidJS web frontend embedded in the binary via rust-embed
- Native GUI window via tao/wry (WebView2 on Windows, WebKit on macOS/Linux)
  with browser fallback mode
- MCP server with 40+ tools for Claude Code integration (JSON-RPC 2.0 over stdio)
- Python client library with typed access to all REST endpoints and subprocess
  server management
- CLI subcommands: web, serve, index, search, doctor, sessions, export, models,
  mcp, annotate, version
- Docker images for four variants: NVIDIA CUDA 12.4, AMD ROCm 6.4, CPU x86_64,
  CPU ARM64
- Multi-stage Docker build with Ollama LLM server, Tesseract, and pdfium bundled
- CI/CD pipeline with multi-platform builds, release automation, and SHA-256
  checksums
- Dependabot for Cargo, npm, Docker, and GitHub Actions dependencies
- Pre-commit hooks for formatting, linting, license auditing, and architecture
  validation

[0.1.1]: https://github.com/FF-TEC/NeuronCite/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/FF-TEC/NeuronCite/releases/tag/v0.1.0
