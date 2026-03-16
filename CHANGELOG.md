# Changelog

All notable changes to NeuronCite are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/FF-TEC/NeuronCite/releases/tag/v0.1.0
