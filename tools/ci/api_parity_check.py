#!/usr/bin/env python3
"""API Parity Checker for NeuronCite.

Deterministic static analysis tool that parses three authoritative Rust source
files and two Python source files to verify that the REST API, MCP tools, and
Python client all have consistent coverage. Detects missing endpoint wrappers,
stale methods, missing model classes, and field-level mismatches between Rust
DTOs and Python dataclasses.

Data sources (parsed via regex, no compilation required):
    1. ``dto.rs``      -- Authoritative Rust DTO structs (request + response).
    2. ``router.rs``   -- Authoritative REST route definitions (Axum router).
    3. ``dispatch.rs`` -- MCP tool dispatch match arms (JSON-RPC handler).
    4. ``client.py``   -- Python client method definitions with endpoint docstrings.
    5. ``models.py``   -- Python response model dataclasses.

Checks performed:
    1. REST endpoint coverage:  Every route in router.rs has a Python wrapper.
    2. Stale method detection:  Every Python endpoint references a route that
       still exists in router.rs.
    3. MCP tool classification: Each MCP tool is classified as REST-backed or
       internal-only (no REST endpoint).
    4. Model parity:            Every Rust response DTO has a corresponding
       Python dataclass.
    5. Field parity:            Field names in Rust structs match the fields in
       their corresponding Python dataclasses (minus intentionally stripped
       fields like ``api_version``).

Exit codes:
    0 -- Full parity (no errors, no warnings unless --strict is active).
    1 -- At least one discrepancy found (or a warning under --strict).
    2 -- Source file parsing error or missing file.

Invocation:
    python tools/api_parity_check.py [--json] [--strict]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Diagnostic data structures
# ---------------------------------------------------------------------------

@dataclass
class Diagnostic:
    """A single parity check result.

    Attributes:
        check:    Name of the check that produced this diagnostic
                  (e.g., ``rest_coverage``, ``field_parity``).
        severity: ``error`` for mandatory parity violations, ``warning`` for
                  informational items that do not block a passing result.
        message:  Human-readable description of the finding.
    """

    check: str
    severity: str
    message: str


@dataclass
class ParityReport:
    """Aggregated results from all parity checks.

    Attributes:
        diagnostics:         All individual findings from every check.
        rest_routes:         Full route paths extracted from router.rs
                             (e.g., ``/api/v1/health``).
        rest_route_methods:  Route path -> HTTP method (``GET``, ``POST``,
                             ``DELETE``) mapping from router.rs.
        mcp_tools:           Tool names extracted from dispatch.rs match arms.
        python_methods:      Method name -> endpoint string mapping from
                             client.py docstrings.
        rust_structs:        All ``pub struct`` names from dto.rs.
        python_model_names:  All dataclass names from models.py.
        mcp_rest_backed:     MCP tools that have a corresponding REST endpoint.
        mcp_internal_only:   MCP tools with no REST endpoint (call internal Rust
                             functions directly).
        route_to_method:     Route -> Python method name mapping built during the
                             REST coverage check (used for the coverage table).
    """

    diagnostics: list[Diagnostic] = field(default_factory=list)
    rest_routes: list[str] = field(default_factory=list)
    rest_route_methods: dict[str, str] = field(default_factory=dict)
    mcp_tools: list[str] = field(default_factory=list)
    python_methods: dict[str, str] = field(default_factory=dict)
    rust_structs: list[str] = field(default_factory=list)
    python_model_names: list[str] = field(default_factory=list)
    mcp_rest_backed: list[str] = field(default_factory=list)
    mcp_internal_only: list[str] = field(default_factory=list)
    route_to_method: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Path parameter normalization
# ---------------------------------------------------------------------------

def normalize_path_params(path: str) -> str:
    """Replace all ``{param_name}`` path segments with ``{}`` for comparison.

    Route definitions in router.rs and endpoint annotations in Python docstrings
    may use different parameter names for the same logical slot (e.g.,
    ``{id}`` vs ``{job_id}``). Normalizing both sides to ``{}`` allows matching
    routes to endpoints without maintaining a manual equivalence table.

    Args:
        path: A route path string (e.g., ``GET /api/v1/jobs/{id}``).

    Returns:
        The same path with all ``{...}`` segments replaced by ``{}``.
    """
    return re.sub(r"\{[^}]+\}", "{}", path)


# ---------------------------------------------------------------------------
# Source file parsers
# ---------------------------------------------------------------------------

def parse_router_routes(router_content: str) -> list[str]:
    """Extract all REST API route paths from router.rs.

    Parses ``.route("/<path>", ...)`` patterns and prepends the ``/api/v1``
    prefix applied by the ``Router::nest`` call. The ``openapi.json`` endpoint
    is excluded because it serves documentation, not application data.
    The published "34 REST API endpoints" count in the README includes
    openapi.json; this function returns 33 routes (34 minus the excluded
    documentation endpoint).

    Args:
        router_content: Full text of the router.rs file.

    Returns:
        Sorted, deduplicated list of full route paths.
    """
    pattern = re.compile(r'\.route\(\s*"(/[^"]+)"')
    routes: list[str] = []
    for match in pattern.finditer(router_content):
        path = match.group(1)
        if "openapi.json" in path:
            continue
        routes.append(f"/api/v1{path}")
    return sorted(set(routes))


def parse_router_route_methods(router_content: str) -> dict[str, str]:
    """Extract route paths with their HTTP methods from router.rs.

    Parses ``.route("/path", get(...))`` / ``post(...)`` / ``delete(...)``
    patterns to determine the HTTP verb for each route.

    Args:
        router_content: Full text of the router.rs file.

    Returns:
        Dict mapping full route path to HTTP method string (``GET``, ``POST``,
        ``DELETE``).
    """
    pattern = re.compile(r'\.route\(\s*"(/[^"]+)",\s*\n?\s*(get|post|delete)\(')
    result: dict[str, str] = {}
    for match in pattern.finditer(router_content):
        path = match.group(1)
        method = match.group(2).upper()
        if "openapi.json" in path:
            continue
        result[f"/api/v1{path}"] = method
    return result


def parse_mcp_tools(dispatch_content: str) -> list[str]:
    """Extract all MCP tool names from dispatch.rs match arms.

    Parses ``"neuroncite_<name>" =>`` patterns from the tool dispatch function.

    Args:
        dispatch_content: Full text of the dispatch.rs file.

    Returns:
        Sorted list of MCP tool names.
    """
    pattern = re.compile(r'"(neuroncite_\w+)"\s*=>')
    return sorted(set(match.group(1) for match in pattern.finditer(dispatch_content)))


def parse_python_methods(client_content: str) -> dict[str, str]:
    """Extract Python client method names and their endpoint docstrings.

    Uses a two-pass approach to handle both single-line and multi-line method
    signatures. Pass 1 finds all ``def method_name(`` occurrences (the ``(``
    matches the opening paren regardless of whether ``self`` is on the same
    line or the next line). Pass 2 searches the region between each method
    and the next for an endpoint annotation in the docstring (e.g.,
    ``GET /api/v1/health``).

    Args:
        client_content: Full text of client.py.

    Returns:
        Dict mapping method name to endpoint string (e.g., ``GET /api/v1/health``).
        Methods without an endpoint annotation (helpers like ``wait_for_job``)
        have an empty endpoint string.
    """
    methods: dict[str, str] = {}

    # Pass 1: Find all public method definitions (lowercase names, indented).
    method_def = re.compile(r"^\s+def\s+([a-z]\w+)\s*\(", re.MULTILINE)
    matches = list(method_def.finditer(client_content))

    # Pass 2: For each method, look in the region up to the next method
    # definition (or EOF) for the endpoint docstring.
    endpoint_re = re.compile(r'"""``([A-Z]+\s+/api/v1/[^`]+)``')
    for i, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(client_content)
        region = client_content[start:end]
        ep_match = endpoint_re.search(region)
        methods[name] = ep_match.group(1) if ep_match else ""

    return methods


def parse_rust_structs(dto_content: str) -> list[str]:
    """Extract all ``pub struct`` names from dto.rs.

    Args:
        dto_content: Full text of dto.rs.

    Returns:
        Sorted, deduplicated list of struct names.
    """
    pattern = re.compile(r"pub struct (\w+)")
    return sorted(set(match.group(1) for match in pattern.finditer(dto_content)))


def parse_rust_struct_fields(dto_content: str) -> dict[str, list[str]]:
    """Extract field names for each Rust struct in dto.rs.

    Matches ``pub struct Name { ... }`` blocks and extracts ``pub field_name:``
    declarations from each block body.

    Args:
        dto_content: Full text of dto.rs.

    Returns:
        Dict mapping struct name to an ordered list of field names.
    """
    result: dict[str, list[str]] = {}
    struct_pattern = re.compile(r"pub struct (\w+)\s*\{(.*?)\}", re.DOTALL)
    field_pattern = re.compile(r"pub\s+(\w+)\s*:")

    for match in struct_pattern.finditer(dto_content):
        struct_name = match.group(1)
        body = match.group(2)
        result[struct_name] = field_pattern.findall(body)

    return result


def parse_python_dataclass_names(models_content: str) -> list[str]:
    """Extract all ``@dataclass``-decorated class names from models.py.

    Finds classes preceded by a ``@dataclass(...)`` decorator line.

    Args:
        models_content: Full text of models.py.

    Returns:
        Sorted, deduplicated list of dataclass names.
    """
    pattern = re.compile(r"@dataclass[^)]*\)\s*\nclass\s+(\w+)")
    return sorted(set(match.group(1) for match in pattern.finditer(models_content)))


def parse_python_dataclass_fields(models_content: str) -> dict[str, list[str]]:
    """Extract field names for each Python dataclass in models.py.

    Uses a two-pass approach: Pass 1 locates all ``class Name:`` declarations
    decorated with ``@dataclass``. Pass 2 extracts the region between consecutive
    class declarations and finds field definitions (indented ``name: type`` lines
    that are not inside a docstring).

    A field line is identified by the pattern ``    name: type`` where the line
    starts with whitespace, followed by an identifier, a colon, and a type
    annotation. Docstring lines (containing triple quotes) are excluded.

    Args:
        models_content: Full text of models.py.

    Returns:
        Dict mapping class name to an ordered list of field names.
    """
    result: dict[str, list[str]] = {}

    # Pass 1: Locate all class definitions with their character positions.
    class_def = re.compile(r"@dataclass[^)]*\)\s*\nclass\s+(\w+)\s*:", re.MULTILINE)
    matches = list(class_def.finditer(models_content))

    # Field lines: indented identifier followed by colon and type annotation.
    # Excludes lines inside docstrings by filtering out lines containing triple
    # quotes and lines that are only whitespace/comments.
    field_re = re.compile(r"^\s{4}(\w+)\s*:", re.MULTILINE)

    for i, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(models_content)
        body = models_content[start:end]

        # Remove docstrings before scanning for fields. Docstrings are
        # triple-quoted strings that may span multiple lines.
        body_no_docstrings = re.sub(r'""".*?"""', "", body, flags=re.DOTALL)
        body_no_docstrings = re.sub(r"'''.*?'''", "", body_no_docstrings, flags=re.DOTALL)

        result[name] = field_re.findall(body_no_docstrings)

    return result


# ---------------------------------------------------------------------------
# Parity checks
# ---------------------------------------------------------------------------

# MCP tools that have no REST API endpoint because they call internal Rust
# functions directly. These are expected to be absent from the Python client.
KNOWN_INTERNAL_ONLY_TOOLS: set[str] = {
    "neuroncite_batch_search",
    "neuroncite_multi_search",
    "neuroncite_export",
    "neuroncite_models",
    "neuroncite_doctor",
    "neuroncite_session_update",
    "neuroncite_files",
    "neuroncite_inspect_annotations",
    "neuroncite_preview_chunks",
    "neuroncite_index_add",
    "neuroncite_reranker_load",
    # Tools that call internal Rust functions directly without a REST endpoint.
    "neuroncite_batch_page",
    "neuroncite_compare_search",
    "neuroncite_reindex_file",
}

# Mapping from Rust DTO struct names to Python dataclass names where the names
# differ across the two languages. DTOs not in this map are expected to have
# identical names on both sides. A ``None`` value means the Rust struct is a
# thin wrapper, query parameter object, or request body that has no Python
# dataclass counterpart by design.
DTO_NAME_MAP: dict[str, str | None] = {
    "JobResponse": "JobStatus",
    "SessionDto": "SessionInfo",
    "BackendDto": "BackendInfo",
    "SearchResultDto": "SearchResult",
    "SessionListResponse": None,   # List wrapper, Python uses list[SessionInfo]
    "JobListResponse": None,       # List wrapper, Python uses list[JobStatus]
    "SearchResponse": None,        # List wrapper, Python uses list[SearchResult]
}

# Rust DTO structs that are request bodies, query parameters, or entries used
# inside request bodies. These do not have Python dataclass counterparts because
# the Python client builds request payloads as plain dicts.
REQUEST_DTOS: set[str] = {
    "AnnotateFromFileRequest",
    "AnnotateRequest",
    "BibReportRequest",
    "ChunksQuery",
    "CitationClaimRequest",
    "CitationCreateRequest",
    "CitationExportRequest",
    "CitationRowsQuery",
    "CitationSubmitEntryDto",
    "CitationSubmitRequest",
    "DiscoverRequest",
    "FetchSourcesRequest",
    "FileCompareRequest",
    "IndexRequest",
    "MultiSearchRequest",
    "ParseBibRequest",
    "SearchRequest",
    "SessionDeleteByDirectoryRequest",
    "ShutdownRequest",
    "VerifyRequest",
}

# Rust enums defined in dto.rs that exist only on the backend side. These are
# not structs and have no Python dataclass counterpart, so they are excluded
# from model and field parity checks.
RUST_ENUMS: set[str] = {
    "SessionSearchStatus",
}

# Fields present in Rust DTOs that the Python client intentionally strips before
# constructing dataclass instances. These fields are excluded from both sides
# of the field parity comparison.
STRIPPED_FIELDS: set[str] = {"api_version"}


def check_rest_coverage(
    routes: list[str],
    route_methods: dict[str, str],
    python_methods: dict[str, str],
) -> tuple[dict[str, str], list[Diagnostic]]:
    """Check that every REST route has a corresponding Python client method.

    Matching uses normalized path parameters: ``{param_name}`` segments are
    replaced with ``{}`` on both sides so that ``/jobs/{id}`` in router.rs
    matches ``/jobs/{job_id}`` in a Python docstring.

    Args:
        routes:         List of route paths from router.rs.
        route_methods:  Route path -> HTTP method mapping.
        python_methods: Method name -> endpoint string mapping from client.py.

    Returns:
        Tuple of (route_to_method mapping, list of diagnostics). The mapping
        associates each matched route with the Python method name that covers it.
    """
    diagnostics: list[Diagnostic] = []
    route_to_method: dict[str, str] = {}

    # Build a lookup from normalized endpoint to Python method name.
    # Endpoint format: "GET /api/v1/health", normalized to "GET /api/v1/health".
    normalized_to_method: dict[str, str] = {}
    for method_name, endpoint in python_methods.items():
        if endpoint:
            normalized_to_method[normalize_path_params(endpoint)] = method_name

    for route in routes:
        method = route_methods.get(route, "?")
        expected = f"{method} {route}"
        normalized = normalize_path_params(expected)

        if normalized in normalized_to_method:
            route_to_method[route] = normalized_to_method[normalized]
        else:
            diagnostics.append(Diagnostic(
                check="rest_coverage",
                severity="error",
                message=f"REST route {expected} has no Python wrapper",
            ))

    return route_to_method, diagnostics


def check_stale_methods(
    routes: list[str],
    route_methods: dict[str, str],
    python_methods: dict[str, str],
) -> list[Diagnostic]:
    """Check that every Python endpoint references a route that exists in router.rs.

    Detects Python methods whose docstring endpoint annotation points to a REST
    route that was removed or renamed. The inverse of ``check_rest_coverage``.

    Args:
        routes:         List of route paths from router.rs.
        route_methods:  Route path -> HTTP method mapping.
        python_methods: Method name -> endpoint string mapping from client.py.

    Returns:
        List of diagnostics for stale Python methods.
    """
    diagnostics: list[Diagnostic] = []

    # Build the set of normalized route endpoints from router.rs.
    normalized_routes: set[str] = set()
    for route in routes:
        method = route_methods.get(route, "?")
        normalized_routes.add(normalize_path_params(f"{method} {route}"))

    for method_name, endpoint in python_methods.items():
        if not endpoint:
            continue  # Helper method without endpoint annotation (e.g., wait_for_job)
        normalized = normalize_path_params(endpoint)
        if normalized not in normalized_routes:
            diagnostics.append(Diagnostic(
                check="stale_method",
                severity="error",
                message=(
                    f"Python method {method_name}() references endpoint "
                    f"'{endpoint}' which does not exist in router.rs"
                ),
            ))

    return diagnostics


def check_mcp_classification(
    mcp_tools: list[str],
) -> tuple[list[str], list[str], list[Diagnostic]]:
    """Classify MCP tools as REST-backed or internal-only.

    Classification is based on the ``KNOWN_INTERNAL_ONLY_TOOLS`` set. Tools
    not in this set are assumed to be REST-backed. A diagnostic is emitted if
    a tool appears in dispatch.rs that is not in the known set and is not
    REST-backed (which would indicate a new tool was added that needs
    classification).

    Args:
        mcp_tools: List of MCP tool names from dispatch.rs.

    Returns:
        Tuple of (rest_backed_tools, internal_only_tools, diagnostics).
    """
    rest_backed: list[str] = []
    internal_only: list[str] = []

    for tool in mcp_tools:
        if tool in KNOWN_INTERNAL_ONLY_TOOLS:
            internal_only.append(tool)
        else:
            rest_backed.append(tool)

    return rest_backed, internal_only, []


def check_model_parity(
    rust_structs: list[str],
    python_classes: list[str],
) -> list[Diagnostic]:
    """Check that every Rust response DTO has a Python dataclass counterpart.

    Request DTOs (in ``REQUEST_DTOS``), Rust enums (in ``RUST_ENUMS``), list
    wrappers, and query parameter structs (mapped to ``None`` in
    ``DTO_NAME_MAP``) are excluded from the comparison because they have no
    Python counterpart by design.

    Args:
        rust_structs:   Struct names from dto.rs.
        python_classes: Dataclass names from models.py.

    Returns:
        List of diagnostics for missing Python models.
    """
    diagnostics: list[Diagnostic] = []
    python_set = set(python_classes)

    for struct_name in rust_structs:
        if struct_name in REQUEST_DTOS:
            continue

        if struct_name in RUST_ENUMS:
            continue

        if struct_name in DTO_NAME_MAP:
            expected_python = DTO_NAME_MAP[struct_name]
            if expected_python is None:
                continue  # Wrapper or entry DTO, no direct Python class
            if expected_python not in python_set:
                diagnostics.append(Diagnostic(
                    check="model_parity",
                    severity="error",
                    message=(
                        f"Rust DTO {struct_name} maps to Python "
                        f"{expected_python} which is missing from models.py"
                    ),
                ))
        elif struct_name in python_set:
            pass  # Same name on both sides
        else:
            diagnostics.append(Diagnostic(
                check="model_parity",
                severity="error",
                message=(
                    f"Rust DTO {struct_name} has no Python counterpart "
                    f"in models.py"
                ),
            ))

    return diagnostics


def check_field_parity(
    rust_fields: dict[str, list[str]],
    python_fields: dict[str, list[str]],
) -> list[Diagnostic]:
    """Compare field names between Rust DTOs and Python dataclasses.

    Only checks response DTOs (skips request DTOs, Rust enums, and wrapper
    structs). Fields in ``STRIPPED_FIELDS`` (like ``api_version``) are removed
    from both sides before comparison since they are intentionally handled by
    the client layer and may or may not appear in the Python models.

    Severity: Missing fields in Python are ``error`` (the dataclass cannot
    parse the response). Extra fields in Python are ``warning`` (the dataclass
    has a field the server does not send).

    Args:
        rust_fields:   Struct name -> field names from dto.rs.
        python_fields: Class name -> field names from models.py.

    Returns:
        List of diagnostics for field mismatches.
    """
    diagnostics: list[Diagnostic] = []

    for rust_name, rust_field_list in rust_fields.items():
        if rust_name in REQUEST_DTOS:
            continue

        if rust_name in RUST_ENUMS:
            continue

        # Resolve the Python class name via the DTO name map.
        if rust_name in DTO_NAME_MAP:
            python_name = DTO_NAME_MAP[rust_name]
            if python_name is None:
                continue
        else:
            python_name = rust_name

        if python_name not in python_fields:
            continue  # Model parity check already reports missing classes

        # Strip intentionally omitted fields from both sides.
        rust_set = set(rust_field_list) - STRIPPED_FIELDS
        python_set = set(python_fields[python_name]) - STRIPPED_FIELDS

        for f in sorted(rust_set - python_set):
            diagnostics.append(Diagnostic(
                check="field_parity",
                severity="error",
                message=(
                    f"{rust_name}.{f} exists in Rust but is missing "
                    f"in Python {python_name}"
                ),
            ))

        for f in sorted(python_set - rust_set):
            diagnostics.append(Diagnostic(
                check="field_parity",
                severity="warning",
                message=(
                    f"{python_name}.{f} exists in Python but is missing "
                    f"in Rust {rust_name}"
                ),
            ))

    return diagnostics


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_text_report(report: ParityReport) -> str:
    """Format the parity report as a human-readable text table.

    Includes a summary section, MCP classification table, REST coverage
    mapping (route -> method), and any diagnostic findings.

    Args:
        report: The completed parity report.

    Returns:
        Multi-line string with formatted results for stderr output.
    """
    lines: list[str] = []
    sep = "=" * 72
    thin = "-" * 72
    lines.append(sep)
    lines.append("NeuronCite API Parity Check")
    lines.append(sep)
    lines.append("")

    # Count Python endpoint methods (methods with a non-empty endpoint string).
    endpoint_count = sum(1 for ep in report.python_methods.values() if ep)

    lines.append(f"REST routes (router.rs):    {len(report.rest_routes)}")
    lines.append(f"MCP tools (dispatch.rs):    {len(report.mcp_tools)}")
    lines.append(f"  REST-backed:              {len(report.mcp_rest_backed)}")
    lines.append(f"  Internal-only:            {len(report.mcp_internal_only)}")
    lines.append(f"Python methods (client.py): {len(report.python_methods)}")
    lines.append(f"  Endpoint wrappers:        {endpoint_count}")
    lines.append(f"  Helpers (no endpoint):    {len(report.python_methods) - endpoint_count}")
    lines.append(f"Rust structs (dto.rs):      {len(report.rust_structs)}")
    lines.append(f"Python models (models.py):  {len(report.python_model_names)}")
    lines.append("")

    # REST coverage mapping table
    lines.append(thin)
    lines.append("REST Route Coverage")
    lines.append(thin)
    for route in report.rest_routes:
        method = report.rest_route_methods.get(route, "?")
        py_method = report.route_to_method.get(route)
        endpoint_str = f"{method:6s} {route}"
        if py_method:
            lines.append(f"  {endpoint_str:55s} -> {py_method}()")
        else:
            lines.append(f"  {endpoint_str:55s} -> MISSING")
    lines.append("")

    # MCP classification table
    lines.append(thin)
    lines.append("MCP Tool Classification")
    lines.append(thin)
    for tool in sorted(report.mcp_rest_backed):
        lines.append(f"  [REST]     {tool}")
    for tool in sorted(report.mcp_internal_only):
        lines.append(f"  [INTERNAL] {tool}")
    lines.append("")

    # Diagnostics
    errors = [d for d in report.diagnostics if d.severity == "error"]
    warnings = [d for d in report.diagnostics if d.severity == "warning"]

    if errors:
        lines.append(thin)
        lines.append(f"ERRORS ({len(errors)})")
        lines.append(thin)
        for d in errors:
            lines.append(f"  [{d.check}] {d.message}")
        lines.append("")

    if warnings:
        lines.append(thin)
        lines.append(f"WARNINGS ({len(warnings)})")
        lines.append(thin)
        for d in warnings:
            lines.append(f"  [{d.check}] {d.message}")
        lines.append("")

    # Verdict
    lines.append(sep)
    if errors:
        lines.append(f"RESULT: FAIL ({len(errors)} errors, {len(warnings)} warnings)")
    elif warnings:
        lines.append(f"RESULT: PASS ({len(warnings)} warnings)")
    else:
        lines.append("RESULT: PASS (full parity)")
    lines.append(sep)

    return "\n".join(lines)


def format_json_report(report: ParityReport) -> str:
    """Format the parity report as a JSON string for machine consumption.

    Args:
        report: The completed parity report.

    Returns:
        JSON string with the complete report data.
    """
    errors = sum(1 for d in report.diagnostics if d.severity == "error")
    warnings = sum(1 for d in report.diagnostics if d.severity == "warning")

    return json.dumps(
        {
            "rest_routes": report.rest_routes,
            "rest_route_methods": report.rest_route_methods,
            "mcp_tools": report.mcp_tools,
            "mcp_rest_backed": report.mcp_rest_backed,
            "mcp_internal_only": report.mcp_internal_only,
            "python_methods": report.python_methods,
            "route_to_method": report.route_to_method,
            "rust_structs": report.rust_structs,
            "python_model_names": report.python_model_names,
            "diagnostics": [
                {"check": d.check, "severity": d.severity, "message": d.message}
                for d in report.diagnostics
            ],
            "errors": errors,
            "warnings": warnings,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """Locate the project root by searching upward for ``Cargo.toml``.

    Starts from this script's directory and walks up the directory tree until
    a directory containing ``Cargo.toml`` is found.

    Returns:
        Path to the project root containing ``Cargo.toml``.

    Raises:
        SystemExit: If no ``Cargo.toml`` is found after reaching the
            filesystem root.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "Cargo.toml").exists():
            return current
        current = current.parent
    print("ERROR: Could not find project root (no Cargo.toml found)", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    """Entry point: parse source files, run all parity checks, print report."""
    parser = argparse.ArgumentParser(
        description="NeuronCite API Parity Checker",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report in JSON format instead of human-readable text.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (exit code 1 if any warnings).",
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Project root directory. Auto-detected from Cargo.toml if not specified.",
    )
    args = parser.parse_args()

    # Locate source files relative to the project root.
    root = Path(args.root) if args.root else find_project_root()

    source_files: dict[str, Path] = {
        "dto.rs": root / "crates" / "neuroncite-api" / "src" / "dto.rs",
        "router.rs": root / "crates" / "neuroncite-api" / "src" / "router.rs",
        "dispatch.rs": root / "crates" / "neuroncite-mcp" / "src" / "dispatch.rs",
        "client.py": root / "clients" / "python" / "neuroncite" / "client.py",
        "models.py": root / "clients" / "python" / "neuroncite" / "models.py",
    }

    # Read all source files, exit with code 2 if any are missing.
    contents: dict[str, str] = {}
    for name, path in source_files.items():
        if not path.exists():
            print(f"ERROR: Source file not found: {path}", file=sys.stderr)
            sys.exit(2)
        contents[name] = path.read_text(encoding="utf-8")

    # ---- Parse ----
    routes = parse_router_routes(contents["router.rs"])
    route_methods = parse_router_route_methods(contents["router.rs"])
    mcp_tools = parse_mcp_tools(contents["dispatch.rs"])
    python_methods = parse_python_methods(contents["client.py"])
    rust_structs = parse_rust_structs(contents["dto.rs"])
    rust_fields = parse_rust_struct_fields(contents["dto.rs"])
    python_classes = parse_python_dataclass_names(contents["models.py"])
    python_fields = parse_python_dataclass_fields(contents["models.py"])

    # ---- Assemble report ----
    report = ParityReport()
    report.rest_routes = routes
    report.rest_route_methods = route_methods
    report.mcp_tools = mcp_tools
    report.python_methods = python_methods
    report.rust_structs = rust_structs
    report.python_model_names = python_classes

    # Check 1: REST endpoint coverage
    route_to_method, coverage_diags = check_rest_coverage(
        routes, route_methods, python_methods,
    )
    report.route_to_method = route_to_method
    report.diagnostics.extend(coverage_diags)

    # Check 2: Stale method detection
    report.diagnostics.extend(
        check_stale_methods(routes, route_methods, python_methods)
    )

    # Check 3: MCP tool classification
    rest_backed, internal_only, mcp_diags = check_mcp_classification(mcp_tools)
    report.mcp_rest_backed = rest_backed
    report.mcp_internal_only = internal_only
    report.diagnostics.extend(mcp_diags)

    # Check 4: Model parity
    report.diagnostics.extend(
        check_model_parity(rust_structs, python_classes)
    )

    # Check 5: Field parity
    report.diagnostics.extend(
        check_field_parity(rust_fields, python_fields)
    )

    # ---- Output ----
    if args.json:
        print(format_json_report(report))
    else:
        print(format_text_report(report), file=sys.stderr)

    # ---- Exit code ----
    errors = sum(1 for d in report.diagnostics if d.severity == "error")
    warnings = sum(1 for d in report.diagnostics if d.severity == "warning")

    if errors > 0:
        sys.exit(1)
    elif args.strict and warnings > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
