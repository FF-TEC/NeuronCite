# Contributing to NeuronCite

Thank you for your interest in contributing to NeuronCite. This document
describes the process for contributing code, reporting bugs, and working with
the codebase.

Please read and follow the [Code of Conduct](CODE_OF_CONDUCT.md) in all
interactions with the project.

---

## Before You Contribute

NeuronCite maintains a high standard for code quality, architecture, and
consistency. All contributions are welcome -- no prior introduction is required.

**For small changes** (typos, documentation, bug fixes with clear scope):
open a pull request directly. The PR template and CI pipeline guide you through
the quality checks.

**For larger changes** (new features, architectural modifications, new crates):
open a GitHub Discussion or Issue first to align on the approach before writing
code. This avoids wasted effort on changes that may conflict with the project's
direction.

Bug reports and feature requests do not require prior discussion -- use the
issue templates directly.

For an overview of planned features and project direction, see the
[Roadmap](https://neuroncite.com/roadmap).

---

## Development Setup

### Prerequisites

- **Rust 1.88+** (stable) -- install via [rustup](https://rustup.rs)
- **Node 20+** and **npm** -- for the SolidJS frontend build
- **Git** -- for version control and pre-commit hooks

### Clone and Build

```bash
git clone https://github.com/FF-TEC/NeuronCite.git
cd NeuronCite

# Build the SolidJS frontend (required for the web feature)
cd crates/neuroncite-web/frontend && npm ci && npx vite build && cd ../../..

# Build the Rust binary with all default features
cargo build -p neuroncite
```

### Feature Flags

The binary crate has six feature flags, all enabled by default:

| Flag | Purpose |
|------|---------|
| `backend-ort` | ONNX Runtime embedding backend (forwarded through the crate chain) |
| `web` | SolidJS frontend embedded via rust-embed |
| `gui` | Native window via tao/wry (requires `web`) |
| `mcp` | Model Context Protocol server for Claude Code and Claude Desktop App |
| `pdfium` | PDFium backend for multi-column PDF extraction |
| `ocr` | Tesseract OCR fallback (requires `pdfium`) |

For a headless server build without native GUI:

```bash
cargo build -p neuroncite --no-default-features \
  --features backend-ort,web,mcp,pdfium,ocr
```

### Frontend Development

During frontend development, run the Vite dev server and the Rust backend
separately:

```bash
# Terminal 1: Start the Rust backend
cargo run -p neuroncite -- serve --port 3030

# Terminal 2: Start the Vite dev server (proxies /api/v1 to port 3030)
cd crates/neuroncite-web/frontend
npm run dev
```

---

## Code Style

### Rust

- **Formatter:** `cargo fmt` (config in `rustfmt.toml`: 4 spaces, max width 100)
- **Linter:** `cargo clippy` with workspace-wide lint configuration
- **Comments:** English, describing the current state of the code (not aspirations
  or history). No exaggerations ("ultra", "enhanced", "optimized"), no emojis.
- **Error handling:** `thiserror` in library crates, `anyhow` in the binary crate.
  Use `expect("reason")` instead of bare `unwrap()`.
- **Unsafe code:** Forbidden in all crates (`#![forbid(unsafe_code)]` where applicable)

### TypeScript / Frontend

- **Formatter:** Prettier (via Vite)
- **Framework:** SolidJS with TypeScript
- **Styles:** CSS custom properties defined in `frontend/src/styles/tokens.css`

---

## Testing

Run the full test suite before submitting a pull request:

```bash
cargo test --workspace
```

The repository includes pre-commit hooks that run automatically on each commit.
The first `cargo build` configures git to use the tracked `.githooks/` directory,
so no manual hook installation is required. The only prerequisite is the
`pre-commit` Python package:

```bash
pip install pre-commit
```

The hooks check: formatting (`cargo fmt`), linting (`cargo clippy`), license
compliance (`cargo deny`), tests (`cargo test`), and architecture validation.

CI must pass on all target platforms before a pull request is merged.

---

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Write a descriptive branch name (e.g., `fix-search-pagination`,
   `add-csv-export`).
3. Keep commits focused -- one logical change per commit.
4. Ensure `cargo fmt`, `cargo clippy`, and `cargo test` pass locally.
5. Open a pull request with a clear title and description. Reference related
   issues if applicable.
6. The maintainer reviews the PR. Address feedback in additional commits (do
   not force-push during review).
7. Once approved, the maintainer merges the PR.

---

## Reporting Bugs

Use the [Bug Report](https://github.com/FF-TEC/NeuronCite/issues/new?template=bug_report.md)
issue template. Include:

- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, NeuronCite version, GPU, enabled features)
- Relevant log output

---

## License

By contributing to NeuronCite, you agree that your contributions are licensed
under the [AGPL-3.0-only](LICENSE) license. No Contributor License Agreement
(CLA) is required.
