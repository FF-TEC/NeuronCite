[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![CI](https://github.com/FF-TEC/NeuronCite/actions/workflows/ci.yml/badge.svg)](https://github.com/FF-TEC/NeuronCite/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.1.1-brightgreen.svg)](Cargo.toml)

<p align="center">
  <img src="crates/neuroncite-web/assets/splash.png" alt="NeuronCite" width="400" />
</p>

<h3 align="center">Autonomous Citation Verification & Semantic Search</h3>

<p align="center">
  Local-first. Single binary (CPU-only) or minimal bundle (GPU). No cloud dependency. GPU-accelerated.
</p>

---

NeuronCite is an enterprise-grade semantic document search engine with autonomous
citation verification, written in Rust. CPU-only builds ship as a single binary
with zero runtime dependencies; GPU and pdfium builds include additional shared
libraries (see [Installation](#installation)). Available for Windows, macOS, and
Linux. All document processing runs locally -- no user data leaves the machine,
no telemetry, no cloud services. Network access is used only for initial
dependency downloads (embedding models, pdfium) and optional features (DOI
resolution, HTML crawling). Documents are indexed into a local SQLite database
with dense vector embeddings and full-text keyword indexes. Embedding models and
LLMs run entirely on user hardware, supporting air-gapped deployments after
provisioning.

---

## Why NeuronCite

**Privacy by Design** -- All document processing runs locally. No user data
leaves the machine, no telemetry, no cloud accounts. No internet connection
required after initial setup (embedding models, pdfium, and optional
OCR/Ollama binaries). Network-dependent features (DOI resolution, HTML crawling,
citation source fetching) are unavailable in air-gapped mode. Supports
air-gapped and classified environments once all dependencies are provisioned.

**Autonomous Citation Verification** -- Feed a LaTeX paper and its bibliography.
NeuronCite indexes the cited PDFs, runs a local LLM via Ollama, and verifies
every `\cite{}` command against actual source material. Each citation receives a
verdict, confidence score, and correction suggestions.

**Enterprise-Grade Architecture** -- 16 Rust crates with clear separation of
concerns. 43 MCP tools, 34 REST API endpoints, a browser-based GUI with 7
tabs, a Python client library, and 11 CLI commands. CPU-only builds compile into
a single executable that runs without Docker, Kubernetes, or external infrastructure.
Citation verification additionally requires a running Ollama instance.

---

## Quick Start

**1.** Download the binary for your platform from the
[Releases](https://github.com/FF-TEC/NeuronCite/releases) page.

**2.** Double-click the binary. The application opens in a native window
(WebView2 on Windows, WebKit on macOS/Linux) -- no terminal, no configuration
required. Linux GUI builds require `libwebkit2gtk-4.1-0` at runtime (see
[Installation](#installation) for details).

**3.** Select a directory of PDFs in the **Indexing** tab, choose an embedding
model, and click **Start**. The first run downloads the selected model
(50 MB--1 GB depending on model size).

> **Terminal alternative:** Run `neuroncite` from the command line, or index
> directly with `neuroncite index --directory ./papers`.

> **Docker alternative:**
> ```bash
> docker run --gpus all -p 3030:3030 \
>   -v neuroncite-data:/data/Documents/NeuronCite \
>   ghcr.io/ff-tec/neuroncite:latest
> ```

---

## Features

### Hybrid Search

Combines three retrieval algorithms for high-precision document search:

- **HNSW vector similarity** -- Approximate nearest neighbor search over dense
  embeddings for semantic matching
- **BM25 keyword matching** -- SQLite FTS5 full-text index for exact term
  matching
- **Reciprocal Rank Fusion** -- Merges and normalizes scores from both retrieval
  methods
- **Cross-encoder reranking** -- Optional second-stage scoring with a
  cross-encoder model for precision-critical queries
- **Sub-chunk refinement** -- Configurable divisors for finer-grained passage
  retrieval within chunks

### Document Indexing

- **PDF extraction** with three backends: pdf-extract (pure Rust, default),
  pdfium (multi-column layout support), and Tesseract OCR (image-heavy pages)
- **HTML crawling** with BFS traversal, depth limiting, rate limiting, domain
  filtering, regex URL patterns, and sitemap parsing
- **Four chunking strategies**: page-based, word-window (fixed word count with
  overlap), token-window (subword tokens), and sentence-based (respects citation
  boundaries)
- **Eight embedding models** from 33M to 335M parameters (384 to 4096
  dimensions), downloaded from HuggingFace on first use
- **GPU acceleration** via ONNX Runtime with CUDA 12.4, DirectML, CoreML, and
  ROCm execution providers; CPU fallback on all platforms

### Citation Verification

NeuronCite parses LaTeX papers for citation commands (`\cite`, `\citep`,
`\citet`, `\autocite`, and variants), resolves them against BibTeX entries, and
verifies each claim against indexed source PDFs using a local LLM via Ollama.

**Five-stage verification pipeline:**

1. Parse LaTeX, extract citation commands, resolve cite-keys against BibTeX
2. Match bibliographic entries to PDFs via Jaro-Winkler similarity (threshold
   0.80)
3. Group citations into batches for parallel processing, preserving co-citations
   and section context
4. Verify each batch through the LLM with a minimum of 2 search queries per
   citation, including cross-corpus search for alternative sources
5. Export 6 output files: annotation CSV, citation CSV/XLSX, corrections JSON,
   citation report JSON, full detail JSON, and annotated PDFs

**Seven verdict types:**

| Verdict | Meaning |
|---------|---------|
| `supported` | Claim explicitly found in cited source with page numbers and passages |
| `partial` | Source supports part of the claim; other parts absent or unsupported |
| `unsupported` | PDF found and read but no passage supports the claim |
| `not_found` | Source PDF not in indexed corpus |
| `wrong_source` | Claim verifiable but evidence found in a different source |
| `unverifiable` | Future projections, subjective statements, or insufficiently specific claims |
| `peripheral_match` | Text found only in non-substantive sections (TOC, bibliography, appendix) |

Each verdict includes a confidence score (0.00--1.00), critical/warning flags
for contradictions and temporal mismatches, and correction suggestions with
types (rephrase, add context, replace citation).

### PDF Annotation

Highlights text passages in PDFs with color-coded annotations based on
verification verdicts. Uses a 5-stage text matching pipeline:

1. Exact byte-level match
2. Normalized match with whitespace collapsing
3. Fuzzy character-level match (string distance)
4. Fallback extraction via multi-backend PDF text pipeline
5. OCR fallback for scanned pages

Accepts annotation input as CSV or JSON. Supports comment annotations with
popup text alongside highlights.

### Web Frontend (7 Tabs)

The SolidJS single-page application is compiled into the binary via rust-embed
and served at `http://localhost:3030`.

| Tab | Function |
|-----|----------|
| **Sources** | BibTeX management, web crawling (BFS with regex patterns, sitemap parsing), DOI resolution (Unpaywall, Semantic Scholar, OpenAlex), metadata extraction |
| **Indexing** | Directory selection, embedding model and chunking strategy configuration, real-time progress tracking, session management |
| **Search** | Multi-session hybrid search with vector/BM25/reranking toggles, sub-chunk refinement, grouped and flat result views, export as Markdown/BibTeX/CSL-JSON/RIS |
| **Citations** | LaTeX file selection with auto-detection of .bib files, Ollama model selection with connection test, verification mode presets (quick, balanced, thorough), live results with expandable verdicts |
| **Annotations** | CSV/JSON annotation input, source PDF directory selection, 5-stage text location pipeline, color configuration, per-quote progress tracking |
| **Models** | Embedding model catalog with download/activate controls, cross-encoder reranker management, Ollama LLM catalog, GPU/CUDA system info, model diagnostics |
| **Settings** | FTS5 index optimization, HNSW vector index rebuild, database reset, dependency detection (pdfium, Tesseract, Poppler), MCP server registration, real-time log streaming via SSE |

### MCP Server (43 Tools)

The Model Context Protocol server exposes 43 tools for AI agent integration via
JSON-RPC 2.0 over stdio, organized in 8 categories:

| Category | Tools | Purpose |
|----------|-------|---------|
| Session Management | 5 | List, delete, update, diff, discover index sessions |
| Indexing | 4 | Index directories, add files, reindex, preview chunks |
| Search & Retrieval | 8 | Search, batch search, multi-session search, compare, text search, content retrieval, export |
| File & Chunk Inspection | 4 | List files, inspect chunks, compare files, quality reports |
| Citation Verification | 8 | Create jobs, add claims, submit, check status, fetch rows, export, retry, fetch sources |
| Annotation | 4 | Annotate PDFs, check status, inspect annotations, remove annotations |
| Web Sources | 3 | Fetch HTML, crawl websites, check crawl status |
| System | 3 | List models, health check, log streaming |

### REST API (34 Endpoints)

Axum-based HTTP server with OpenAPI specification (utoipa), bearer token
authentication with constant-time comparison, per-IP rate limiting, CORS, and
Server-Sent Events for real-time progress streaming. Endpoints cover all
functionality: sessions, indexing, search, citation, annotation, models, and
system health.

### Python Client

Typed access to all REST endpoints with subprocess server management.

```python
from neuroncite import NeuronCiteClient

client = NeuronCiteClient()  # default: http://127.0.0.1:3030
results = client.search(session_id="...", query="capital asset pricing model")
for r in results:
    print(f"  [{r.score:.2f}] {r.citation}")
```

30+ typed methods covering search, indexing, citation, annotation, and model
management. See [clients/python/README.md](clients/python/README.md) for the
full API reference.

---

## Installation

### Pre-built Binaries

Download the binary for your platform from the
[Releases](https://github.com/FF-TEC/NeuronCite/releases) page.
Each release includes SHA-256 checksums for verification.

| Platform | Architecture | GUI | Artifact |
|----------|-------------|-----|----------|
| Windows | x86_64 | Native window (WebView2) | `neuroncite-windows-x64.exe` |
| Linux | x86_64 | Native window (WebKit2GTK) | `neuroncite-linux-x64` |
| Linux | x86_64 | Browser-only (headless) | `neuroncite-linux-x64-server` |
| Linux | ARM64 | Native window (WebKit2GTK) | `neuroncite-linux-arm64` |
| Linux | ARM64 | Browser-only (headless) | `neuroncite-linux-arm64-server` |
| macOS | ARM64 (Apple Silicon) | Native window (WebKit) | `neuroncite-macos-arm64` |
| macOS | x86_64 (Intel) | Native window (WebKit) | `neuroncite-macos-x64` |

Linux GUI variants require `libwebkit2gtk-4.1-0` at runtime.
Server variants have zero runtime dependencies beyond glibc.

### Docker

Four image variants are published to the GitHub Container Registry.
See [docker/README.md](docker/README.md) for compose profiles, environment
variables, and local build instructions.

```bash
# NVIDIA GPU (default)
docker run --gpus all -p 3030:3030 \
  -v neuroncite-data:/data/Documents/NeuronCite \
  ghcr.io/ff-tec/neuroncite:latest

# AMD ROCm
docker run --device=/dev/kfd --device=/dev/dri -p 3030:3030 \
  -v neuroncite-data:/data/Documents/NeuronCite \
  ghcr.io/ff-tec/neuroncite:latest-rocm

# CPU only
docker run -p 3030:3030 \
  -v neuroncite-data:/data/Documents/NeuronCite \
  ghcr.io/ff-tec/neuroncite:latest-cpu
```

| Variant | Base Image | GPU | Tag |
|---------|-----------|-----|-----|
| NVIDIA | CUDA 12.4 + cuDNN, Ubuntu 22.04 | CUDA | `latest` / `<version>-nvidia` |
| ROCm | ROCm 6.4, Ubuntu 22.04 | ROCm | `latest-rocm` / `<version>-rocm` |
| CPU | Ubuntu 22.04, x86_64 | None | `latest-cpu` / `<version>-cpu` |
| CPU ARM64 | Ubuntu 22.04, ARM64 | None | `latest-cpu-arm64` / `<version>-cpu-arm64` |

### From Source

Prerequisites: Rust 1.88+ (stable), Node 20+, npm.

```bash
git clone https://github.com/FF-TEC/NeuronCite.git
cd neuroncite

# Build the SolidJS frontend
cd crates/neuroncite-web/frontend && npm ci && npx vite build && cd ../../..

# Build the Rust binary (all features enabled by default)
cargo build --release -p neuroncite
```

The binary is at `target/release/neuroncite` (Linux/macOS) or
`target/release/neuroncite.exe` (Windows).

For a server-only build without native GUI:

```bash
cargo build --release -p neuroncite \
  --no-default-features \
  --features backend-ort,web,mcp,pdfium,ocr
```

### Python Client

```bash
# From PyPI (once published):
pip install neuroncite

# From source:
pip install ./clients/python
```

Requires Python 3.10+. See [clients/python/README.md](clients/python/README.md)
for the full API reference.

---

## Usage

### Web UI

```bash
neuroncite
# or explicitly:
neuroncite web --port 3030
```

Starts the API server and opens the SolidJS web frontend in a native window
(WebView2 on Windows, WebKit on macOS/Linux). Falls back to the default browser
if the native window cannot be created. The frontend is served at
`http://localhost:3030`.

### Headless Server

```bash
neuroncite serve --port 3030 --bind 0.0.0.0
```

Runs the REST API without opening a browser or native window. Suitable for
remote servers, Docker, and automation.

### CLI Indexing

```bash
neuroncite index \
  --directory /path/to/pdfs \
  --model "BAAI/bge-small-en-v1.5" \
  --strategy word \
  --chunk-size 300 \
  --overlap 50
```

Indexes a directory of PDFs without running a persistent server. Embedding
models are downloaded from HuggingFace on first use.

### CLI Search

```bash
neuroncite search \
  --directory /path \
  --session-id 1 \
  --query "heteroskedasticity-consistent covariance matrix" \
  --hybrid \
  --rerank
```

Executes a single search query against an existing session. Results are printed
as JSON (default) or plain text (`--format text`).

### MCP Server (Claude Code & Claude Desktop App)

```bash
neuroncite mcp install                        # registers in Claude Code settings (default)
neuroncite mcp install --target claude-desktop # registers in Claude Desktop App settings
neuroncite mcp uninstall                      # removes registration from Claude Code settings
neuroncite mcp status                         # shows current registration status
neuroncite mcp serve                          # starts stdio JSON-RPC server
```

### All Commands

| Command | Description |
|---------|-------------|
| `neuroncite` / `neuroncite web` | Launch web UI in a native window (browser fallback) |
| `neuroncite serve` | Headless API server |
| `neuroncite index` | Index a directory of PDFs |
| `neuroncite search` | Execute search queries |
| `neuroncite annotate` | Annotate PDFs from CSV/JSON |
| `neuroncite doctor` | Check runtime dependencies (Tesseract, pdfium, GPU) |
| `neuroncite sessions` | List index sessions in a database |
| `neuroncite export` | Export results as Markdown, BibTeX, CSL-JSON, RIS, or plain text |
| `neuroncite models list\|info\|download\|verify\|system` | Manage embedding models and check system capabilities |
| `neuroncite mcp install\|uninstall\|serve\|status` | Register, remove, run, and check MCP server |
| `neuroncite version` | Print version, build features, and Git commit hash |

---

## Architecture

```
PDFs / HTML pages
       |
       v
  Extract text
  (pdf-extract / pdfium / Tesseract OCR / readability)
       |
       v
  Chunk text
  (page / word-window / token-window / sentence)
       |
       v
  Embed chunks
  (ONNX Runtime with CUDA / DirectML / CoreML / CPU)
       |
       v
  Store vectors + metadata
  (HNSW index + SQLite FTS5)
       |
       v
  Hybrid search
  (vector kNN + BM25 keyword + RRF fusion + optional reranking)
       |
       v
  Ranked results with citations, page numbers, and scores
```

### Cargo Workspace (16 Crates)

| Layer | Crate | Responsibility |
|-------|-------|---------------|
| Binary | `neuroncite` | Entry point, CLI argument parsing (clap), execution mode dispatch |
| Presentation | `neuroncite-web` | SolidJS frontend (rust-embed), native GUI (tao/wry), SSE broadcast |
| Presentation | `neuroncite-api` | REST API server (Axum), 34 endpoints, OpenAPI, bearer auth, rate limiting, SSE |
| Presentation | `neuroncite-mcp` | MCP server (43 tools, JSON-RPC 2.0 over stdio) |
| Domain | `neuroncite-pipeline` | Background job executor, GPU worker with priority channels, two-phase indexing |
| Domain | `neuroncite-search` | Hybrid search, BM25, Reciprocal Rank Fusion, deduplication, reranking |
| Domain | `neuroncite-citation` | LaTeX/BibTeX parsing, batch claim extraction, LLM-driven verification |
| Domain | `neuroncite-annotate` | PDF annotation with 5-stage text matching pipeline |
| Core | `neuroncite-store` | SQLite storage (r2d2 pool), HNSW index, FTS5 full-text search, workflow tracking |
| Core | `neuroncite-embed` | Dense embeddings via ONNX Runtime, model download and management, cross-encoder reranking |
| Core | `neuroncite-pdf` | PDF discovery and text extraction (pdf-extract, pdfium, Tesseract OCR) |
| Core | `neuroncite-html` | HTML fetching, readability extraction, caching, BFS crawling with SSRF protection |
| Core | `neuroncite-chunk` | Text chunking strategies (page, word, token, sentence) |
| Core | `neuroncite-llm` | LLM abstraction layer, Ollama HTTP client |
| Foundation | `neuroncite-core` | Shared types, trait definitions, configuration, error types (zero internal dependencies) |
| Dev | `neuroncite-testgen` | Test data generation and property-based testing utilities |

### Feature Flags

| Flag | Purpose | Default |
|------|---------|---------|
| `backend-ort` | ONNX Runtime embedding backend with CUDA support | Enabled |
| `web` | SolidJS frontend embedded via rust-embed | Enabled |
| `gui` | Native window via tao/wry (requires `web`) | Enabled |
| `mcp` | Model Context Protocol server for AI agent integration | Enabled |
| `pdfium` | Multi-column PDF extraction backend | Enabled |
| `ocr` | Tesseract OCR fallback for scanned pages (requires `pdfium`) | Enabled |

For the full architecture document, see [docs/architecture.pdf](docs/architecture.pdf).

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB+ |
| CPU | 2 cores | 4+ cores |
| GPU | Not required | NVIDIA (CUDA 12.4+) or AMD (ROCm 6.4+) |
| Disk | 200 MB (binary) + model size | 2 GB+ for large collections |

Embedding model sizes range from 50 MB (bge-small, 33M parameters) to 1 GB
(large models, 335M parameters). GPU acceleration is optional -- the CPU
execution provider works on all platforms. The application runs entirely offline
after the initial model download.

---

## Documentation

- [neuroncite.com](https://neuroncite.com) -- Project website with feature overview, FAQ, and roadmap
- [REST API and CLI Reference](https://neuroncite.com/docs/) -- Full endpoint documentation, Python client guide, MCP setup
- [Pricing and Licensing](https://neuroncite.com/pricing/) -- AGPL-3.0 (free) and Enterprise license comparison
- [Architecture Document](docs/architecture.pdf) -- Full system design (16 crates, data flow, design decisions)
- [Docker Deployment](docker/README.md) -- Image variants, compose profiles, environment variables, local builds
- [Python Client](clients/python/README.md) -- Full API reference with 30+ typed methods
- [Tools and Scripts](tools/README.md) -- CI validators, code generators, developer utilities
- [Commercial License](COMMERCIAL_LICENSE.md) -- Dual licensing details and use-case matrix

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
development setup, code style guidelines, testing instructions, and the pull
request process.

Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

---

## Security

To report a vulnerability, see [SECURITY.md](SECURITY.md) for the disclosure
process and response timeline.

---

## License

NeuronCite is dual-licensed:

- **AGPL-3.0-only** -- free copyleft license for any use, including commercial.
  Requires source disclosure when distributing or providing network access.
  See [LICENSE](LICENSE).
- **Commercial license** for proprietary products, SaaS deployments, and
  redistribution without AGPL source-disclosure obligations.
  See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

For commercial licensing inquiries: licensing@neuroncite.com

---

Copyright (C) 2026 Felix Fritz. All rights reserved.
