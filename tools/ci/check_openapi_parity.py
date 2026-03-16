#!/usr/bin/env python3
"""Verify that every API route registered in router.rs has a corresponding
handler entry in the OpenAPI specification in openapi.rs.

The script compares two counts:
  - Router route count: the number of .route() calls in router.rs, excluding
    the /openapi.json endpoint which serves the spec itself and is not an API
    operation.
  - OpenAPI path count: the number of handlers::* entries in the paths(...)
    attribute of the #[openapi(...)] derive macro in openapi.rs.

When these counts diverge, a handler is either undocumented in the OpenAPI
spec (callers cannot discover it via the spec) or listed in the spec without
a corresponding route (the spec documents a phantom endpoint). Both cases
are CI failures.

Exit code:
  0  counts match, no discrepancy detected
  1  counts differ or a required file could not be read
"""

import re
import sys
from pathlib import Path

# Paths are relative to the workspace root, which is the working directory
# when this script is invoked from the validate-architecture CI job.
ROUTER_FILE = Path("crates/neuroncite-api/src/router.rs")
OPENAPI_FILE = Path("crates/neuroncite-api/src/openapi.rs")

# The OpenAPI spec endpoint itself is not an API operation and must be
# excluded from the route count so the two sides remain comparable.
EXCLUDED_PATHS = {"/openapi.json"}


def extract_router_paths(text: str) -> list[str]:
    """Return all path strings from .route("...", ...) calls in router.rs.

    Captures the first string argument of every .route() call.  Paths in
    EXCLUDED_PATHS are removed from the result before returning.
    """
    # Match .route( optionally followed by whitespace then a quoted path.
    pattern = re.compile(r'\.route\(\s*"(/[^"]*)"')
    paths = pattern.findall(text)
    return [p for p in paths if p not in EXCLUDED_PATHS]


def extract_openapi_handler_count(text: str) -> int:
    """Return the number of handler references inside the paths(...) macro.

    Looks for the paths(...) block in the #[openapi(...)] attribute and
    counts every occurrence of the pattern handlers::<module>::<function>.
    Each such reference corresponds to one documented API operation.
    """
    # Extract the content of the paths(...) argument, which may span multiple
    # lines. A greedy match between 'paths(' and the matching ')' is used;
    # since the block contains no nested parentheses in practice this is safe.
    paths_block_match = re.search(r'\bpaths\s*\((.*?)\)', text, re.DOTALL)
    if not paths_block_match:
        print("ERROR: could not locate paths(...) block in openapi.rs", file=sys.stderr)
        return 0
    paths_content = paths_block_match.group(1)
    # Each documented handler appears as handlers::<module>::<function>.
    handler_refs = re.findall(r'\bhandlers::\w+::\w+', paths_content)
    return len(handler_refs)


def main() -> int:
    """Run the parity check and return an exit code."""
    if not ROUTER_FILE.exists():
        print(f"ERROR: router file not found: {ROUTER_FILE}", file=sys.stderr)
        return 1
    if not OPENAPI_FILE.exists():
        print(f"ERROR: openapi file not found: {OPENAPI_FILE}", file=sys.stderr)
        return 1

    router_text = ROUTER_FILE.read_text(encoding="utf-8")
    openapi_text = OPENAPI_FILE.read_text(encoding="utf-8")

    router_paths = extract_router_paths(router_text)
    router_count = len(router_paths)
    openapi_count = extract_openapi_handler_count(openapi_text)

    print(f"Router API routes (excluding /openapi.json): {router_count}")
    print(f"OpenAPI documented handlers:                 {openapi_count}")

    if router_count != openapi_count:
        print(
            f"\nERROR: route count mismatch -- {router_count} routes in router.rs "
            f"vs {openapi_count} handler entries in openapi.rs.\n"
            "Add missing handlers to the paths(...) attribute in openapi.rs "
            "or remove stale entries.",
            file=sys.stderr,
        )
        return 1

    print("OK: route counts match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
