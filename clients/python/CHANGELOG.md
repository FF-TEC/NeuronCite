# Changelog

All changes to the NeuronCite Python client are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.1.1 - 2026-03-17

### Fixed

- Corrected project URLs from `neuroncite/neuroncite` to `FF-TEC/NeuronCite`
  in `pyproject.toml`.
- Added from-source install alternative to README since PyPI package is not yet
  published.

## 0.1.0 - 2026-03-14

### Added

- Full REST API parity with 35 public methods covering all 33 endpoints.
- 47 frozen response dataclasses matching the Rust DTO wire format.
- `NeuronCiteServer` subprocess manager with context manager protocol.
- Bearer token authentication for LAN-mode deployments.
- PEP 561 inline type annotations (`py.typed` marker).
- Comprehensive test suite with 58+ test methods using mock HTTP responses.
