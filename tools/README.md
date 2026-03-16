# Tools

Python scripts for CI validation, artifact generation, and development utilities.
All scripts require Python 3.10+ and use only the standard library unless noted.

## Directory Structure

```
tools/
  ci/    -- Automated validators (CI pipeline + pre-commit hooks)
  gen/   -- Artifact generators (splash screen, LaTeX chapters)
  dev/   -- Manual developer utilities (metrics, license headers)
```

## ci/ -- CI Validators

| Script                    | Purpose                                           | Invocation                                                                                      |
| ------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `validate_architecture.py`| Crate structure, modules, features vs. LaTeX doc  | `python tools/ci/validate_architecture.py --tex docs/architecture.tex --root . --strict`        |
| `validate_schemas.py`     | SQL DDL in schema.rs vs. LaTeX schema tables      | `python tools/ci/validate_schemas.py --tex docs/architecture.tex --schema crates/neuroncite-store/src/schema.rs --strict` |
| `validate_tests.py`       | Rust test IDs vs. LaTeX test catalogs             | `python tools/ci/validate_tests.py --tex docs/architecture.tex --root . --strict`               |
| `validate_consistency.py` | MCP tools, REST endpoints, CLI, crate counts      | `python tools/ci/validate_consistency.py --tex docs/architecture.tex --root . --strict`         |
| `api_parity_check.py`     | REST routes, MCP dispatch, Python client parity   | `python tools/ci/api_parity_check.py --strict`                                                  |
| `check_openapi_parity.py` | router.rs route count vs. openapi.rs handlers     | `python tools/ci/check_openapi_parity.py`                                                       |

## gen/ -- Generators

| Script                     | Purpose                                           | Invocation                                                                                      | Extra Deps |
| -------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------- |
| `generate_splash.py`       | Renders the 960x360 HiDPI splash screen PNG       | `python tools/gen/generate_splash.py`                                                           | Pillow     |
| `generate_test_chapter.py` | Scans Rust test IDs, produces LaTeX test chapter   | `python tools/gen/generate_test_chapter.py --root . --output docs/tests_generated.tex`          | --         |

## dev/ -- Developer Utilities

| Script              | Purpose                                      | Invocation                                       |
| ------------------- | -------------------------------------------- | ------------------------------------------------ |
| `license_header.py` | Set or remove AGPL-3.0 headers on .rs files  | `python tools/dev/license_header.py --set`       |
| `count_lines.py`    | Repository metrics (lines, crates, safety)   | `python tools/dev/count_lines.py`                |
