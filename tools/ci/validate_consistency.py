#!/usr/bin/env python3
"""Cross-Section Consistency Validator for NeuronCite.

Validates internal consistency of the LaTeX architecture document and
bidirectional consistency between the document and the Rust source code.
Checks MCP tool catalogs, REST API endpoint tables, CLI subcommand tables,
crate counts, LaTeX label/ref integrity, and numeric prose claims.

This is the third validator in the NeuronCite documentation verification
suite, alongside validate_architecture.py (module structure, feature flags,
dependencies) and validate_schemas.py (database schema tables, columns,
indexes, triggers).

Validation checks (11 categories):

  1. crate_count          (error)   Prose "N library crates" matches crate table
  2. mcp_tool_count       (error)   Prose tool-count claims match code
  3. mcp_tool_catalog     (error)   Tool catalog table matches tools.rs
  4. mcp_tool_mentions    (warning) Tool names in doc exist in code
  5. api_endpoint_table   (error)   Endpoint table matches router.rs
  6. cli_subcommand_table (error)   Subcommand table matches main.rs
  7. label_integrity      (warning) Labels/refs are bidirectionally complete
  8. numeric_claims       (warning) Countable prose claims match item lists
  9. verdict_count        (error)   Prose verdict-count claims match Verdict enum
  10. pipeline_stage_count (error)   Prose stage-count claims match locate.rs
  11. default_features     (error)   Build doc matches Cargo.toml default features

Exit codes:
  0  All checks passed (no errors; warnings allowed unless --strict).
  1  At least one error found, or a warning with --strict enabled.
  2  Script cannot run (missing files, parse failure).

Usage:
  python tools/validate_consistency.py \\
      --tex docs/architecture.tex --root . [--strict] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# LaTeX \input{} resolution
# ---------------------------------------------------------------------------

def resolve_inputs(tex_content: str, tex_dir: Path) -> str:
    """Resolve \\input{filename} directives by inlining referenced files.

    Searches for \\input{name} commands in the LaTeX content and replaces
    each with the full contents of the referenced file. The file path is
    resolved relative to tex_dir (the directory containing the main .tex
    file). The .tex extension is appended if the filename does not already
    end with it, following the standard LaTeX \\input convention.

    Unresolvable \\input commands (file not found on disk) are left in
    place without modification.

    Args:
        tex_content: The raw LaTeX document content.
        tex_dir: The directory containing the main .tex file.

    Returns:
        The LaTeX content with all resolvable \\input{} directives
        replaced by the contents of the referenced files.
    """
    def _replace_input(match: re.Match) -> str:
        filename = match.group(1)
        if not filename.endswith(".tex"):
            filename += ".tex"
        input_path = tex_dir / filename
        if input_path.is_file():
            return input_path.read_text(encoding="utf-8")
        return match.group(0)

    return re.sub(r"\\input\{([^}]+)\}", _replace_input, tex_content)


# ---------------------------------------------------------------------------
# Word-to-number mapping for prose claims like "ten library crates".
# Covers the range relevant to NeuronCite's entity counts (1-30).
# ---------------------------------------------------------------------------
WORD_TO_NUMBER: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22,
    "twenty-three": 23, "twenty-four": 24, "twenty-five": 25,
    "twenty-six": 26, "twenty-seven": 27, "twenty-eight": 28,
    "twenty-nine": 29, "thirty": 30,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Diagnostic:
    """A single validation diagnostic (error or warning).

    Attributes:
        level: Either "error" or "warning".
        category: The check category identifier (e.g., "crate_count",
            "mcp_tool_catalog").
        section: The section or entity group this diagnostic applies to
            (e.g., "crate_overview", "mcp_tools", "api_endpoints",
            "cli_subcommands", "labels").
        detail: A human-readable description of the discrepancy.
    """

    level: str
    category: str
    section: str
    detail: str


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sanitize_latex_name(raw: str) -> str:
    """Convert a LaTeX-formatted identifier to a plain Python string.

    Handles the escaping conventions used in the architecture document:
      - \\_  -> _ (LaTeX underscore escape)
      - \\allowbreak -> removed (line-break hint)
      - \\texttt{...} wrapper -> stripped to inner content
      - Surrounding whitespace -> trimmed

    Args:
        raw: The raw LaTeX text (e.g., "neuroncite\\_search" or
            "\\texttt{neuroncite\\_annotate}").

    Returns:
        The clean identifier (e.g., "neuroncite_search").
    """
    result = raw.replace("\\_", "_")
    result = result.replace("\\allowbreak", "")
    result = result.replace("\\texttt{", "").replace("}", "")
    result = re.sub(r"\s+", " ", result).strip()
    return result


def parse_number_word(word: str) -> int | None:
    """Look up a written-out number word in the WORD_TO_NUMBER dictionary.

    The lookup is case-insensitive. Returns None if the word is not a
    recognized number word (e.g., "many" or "several").

    Args:
        word: The number word to look up (e.g., "twenty", "fourteen").

    Returns:
        The integer value, or None if not found.
    """
    return WORD_TO_NUMBER.get(word.lower().strip())


def _join_multiline_rows(table_body: str) -> list[str]:
    """Join LaTeX table rows that span multiple source lines.

    LaTeX table rows end with \\\\ (double backslash). Source lines that
    do not end with \\\\ are continuations of the current row. This
    function concatenates continuation lines and returns a list of
    complete logical rows.

    Skips structural commands (toprule, midrule, bottomrule, endfirsthead,
    endhead, endfoot, endlastfoot) and empty lines.

    Args:
        table_body: The text between \\begin{longtable/tabularx} and
            \\end{longtable/tabularx}, with the column specification
            already removed.

    Returns:
        A list of complete row strings, each ending with \\\\.
    """
    structural = {
        "\\toprule", "\\midrule", "\\bottomrule",
        "\\endfirsthead", "\\endhead", "\\endfoot", "\\endlastfoot",
    }

    rows: list[str] = []
    current_parts: list[str] = []

    for line in table_body.splitlines():
        stripped = line.strip()

        # Skip empty lines and structural commands
        if not stripped:
            continue
        if stripped in structural:
            continue
        # Skip caption lines
        if stripped.startswith("\\caption"):
            continue

        current_parts.append(stripped)

        # A row is complete when it ends with \\ (possibly followed by
        # whitespace or a comment)
        if re.search(r"\\\\(\s*(%.*)?)?$", stripped):
            rows.append(" ".join(current_parts))
            current_parts = []

    # If there are leftover parts (row without trailing \\), include them
    if current_parts:
        rows.append(" ".join(current_parts))

    return rows


def _find_table_body(tex_content: str, caption_pattern: str) -> str | None:
    """Locate a LaTeX table/longtable environment by its caption text.

    Searches for the caption matching the given pattern, then finds the
    enclosing environment (tabularx or longtable). Returns the text
    between the environment begin and end markers.

    Args:
        tex_content: The full LaTeX document content.
        caption_pattern: A regex pattern to match in the \\caption{...} text.

    Returns:
        The table body text, or None if the caption is not found.
    """
    caption_match = re.search(
        r"\\caption\{" + caption_pattern + r"[^}]*\}",
        tex_content,
        re.DOTALL,
    )
    if not caption_match:
        return None

    caption_pos = caption_match.start()

    # Search backward from caption for the nearest \begin{tabularx} or
    # \begin{longtable}
    begin_pattern = r"\\begin\{(tabularx|longtable)\}"
    begin_match = None
    for m in re.finditer(begin_pattern, tex_content):
        if m.start() < caption_pos:
            begin_match = m
        else:
            break

    if not begin_match:
        return None

    env_name = begin_match.group(1)
    end_pattern = r"\\end\{" + env_name + r"\}"
    end_match = re.search(end_pattern, tex_content[begin_match.end():])
    if not end_match:
        return None

    body_start = begin_match.end()
    body_end = begin_match.end() + end_match.start()
    return tex_content[body_start:body_end]


# ---------------------------------------------------------------------------
# LaTeX parsing functions
# ---------------------------------------------------------------------------

def parse_crate_table(tex_content: str) -> tuple[list[str], list[str]]:
    """Extract crate names and types from the workspace crate overview table.

    Parses the tabularx table with caption "Workspace crates and their
    responsibilities" (Section 2). Each data row has the format:
        \\texttt{crate-name} & Library|Binary & Description \\\\

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A tuple (library_crates, binary_crates) where each element is a
        list of crate names (e.g., ["neuroncite-core", "neuroncite-pdf", ...]).
    """
    body = _find_table_body(tex_content, r"Workspace crates")
    if not body:
        return [], []

    library_crates: list[str] = []
    binary_crates: list[str] = []

    rows = _join_multiline_rows(body)
    for row in rows:
        # Skip header row
        if "\\textbf" in row:
            continue

        cells = row.split("&")
        if len(cells) < 2:
            continue

        name_cell = cells[0].strip()
        type_cell = cells[1].strip()

        # Extract crate name from the first cell. Supports two formats:
        # 1) \texttt{neuroncite-xxx} (legacy tabularx layout)
        # 2) neuroncite-xxx          (longtable with \ttfamily column)
        name_match = re.search(r"\\texttt\{([^}]+)\}", name_cell)
        if name_match:
            crate_name = sanitize_latex_name(name_match.group(1))
        else:
            bare_match = re.search(r"(neuroncite[\w-]*)", name_cell)
            if not bare_match:
                continue
            crate_name = bare_match.group(1)

        if "Library" in type_cell:
            library_crates.append(crate_name)
        elif "Binary" in type_cell:
            binary_crates.append(crate_name)

    return library_crates, binary_crates


def parse_crate_count_claim(tex_content: str) -> tuple[int | None, int | None]:
    """Extract the prose claim about crate counts from the document.

    Searches for the pattern "N library crates and M binary crate(s)" where
    N and M are written-out number words (e.g., "ten library crates and
    one binary crate" at line 238).

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A tuple (library_count, binary_count) where each is an integer
        parsed from the number word, or None if the claim is not found.
    """
    match = re.search(
        r"(\w+(?:-\w+)?)\s+library\s+crates?\s+and\s+(\w+(?:-\w+)?)\s+binary",
        tex_content,
    )
    if not match:
        return None, None

    lib_count = parse_number_word(match.group(1))
    bin_count = parse_number_word(match.group(2))
    return lib_count, bin_count


def parse_mcp_tool_catalog(tex_content: str) -> list[str]:
    """Extract tool names from the MCP tool catalog table.

    Parses the tabularx table with caption "MCP tool catalog" (Section 20,
    MCP Server). Each data row has the format:
        \\texttt{neuroncite\\_tool\\_name} & Description \\\\

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A sorted list of tool name strings (e.g., ["neuroncite_annotate",
        "neuroncite_batch_search", ...]).
    """
    body = _find_table_body(tex_content, r"MCP tool catalog")
    if not body:
        return []

    tool_names: list[str] = []

    rows = _join_multiline_rows(body)
    for row in rows:
        if "\\textbf" in row:
            continue

        # Extract tool name from the first \texttt{} in the row
        match = re.search(r"\\texttt\{(neuroncite[^}]+)\}", row)
        if match:
            name = sanitize_latex_name(match.group(1))
            tool_names.append(name)

    return sorted(tool_names)


def parse_mcp_tool_count_claims(tex_content: str) -> list[tuple[str, int]]:
    """Extract all prose claims about MCP tool counts from the document.

    Searches for patterns like "catalog of twenty tools" or "exposes
    fourteen tools" and converts the number word to an integer.

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A list of (context_phrase, claimed_count) tuples. The context
        phrase is the matched text for diagnostic reporting (e.g.,
        "catalog of twenty tools").
    """
    claims: list[tuple[str, int]] = []

    # Pattern matches phrases like "catalog of twenty tools" or
    # "exposes fourteen tools"
    pattern = r"((?:catalog\s+of|exposes|provides|lists)\s+(\w+(?:-\w+)?)\s+tools)"
    for match in re.finditer(pattern, tex_content, re.IGNORECASE):
        full_phrase = match.group(1)
        number_word = match.group(2)
        count = parse_number_word(number_word)
        if count is not None:
            claims.append((full_phrase.strip(), count))

    return claims


def parse_api_endpoint_table(
    tex_content: str,
) -> list[tuple[str, str]]:
    """Extract HTTP method + path pairs from the REST API endpoint table.

    Parses the longtable with caption "REST API v1 endpoints" (Section 10,
    REST API Design). Each data row has the format:
        \\texttt{GET} & \\texttt{/api/v1/path} & Description \\\\

    Some paths span multiple \\texttt{} blocks across continuation lines
    (e.g., "/api/v1/documents/" followed by "{id}/pages/{n}"). These are
    concatenated after row-joining.

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A list of (method, path) tuples where method is uppercase
        (e.g., "GET") and path includes the /api/v1/ prefix.
    """
    body = _find_table_body(tex_content, r"REST API v1 endpoints")
    if not body:
        return []

    endpoints: list[tuple[str, str]] = []

    rows = _join_multiline_rows(body)
    for row in rows:
        if "\\textbf" in row:
            continue

        cells = row.split("&")
        if len(cells) < 2:
            continue

        method_cell = cells[0].strip()
        path_cell = cells[1].strip()

        # Extract HTTP method from \texttt{GET}, \texttt{POST}, etc.
        method_match = re.search(
            r"\\texttt\{(GET|POST|PUT|DELETE|PATCH)\}", method_cell,
        )
        if not method_match:
            continue
        method = method_match.group(1)

        # Extract path by concatenating all \texttt{} blocks in the path cell.
        # Handles multi-line paths like:
        #   \texttt{/api/v1/sessions/} \texttt{\{id\}/optimize}
        # The inner regex handles LaTeX escapes inside \texttt{} blocks:
        #   \{ and \} (escaped braces), \_ (escaped underscore), and
        #   \\ (escaped backslash). For example:
        #   \texttt{/api/v1/citation/\{job\_id\}/export}
        # where \} is a literal brace, \_ is a literal underscore, and
        # the final } closes the \texttt{} command.
        path_parts = re.findall(
            r"\\texttt\{((?:[^}\\]|\\[{}\\_])*)\}", path_cell,
        )
        if not path_parts:
            continue

        raw_path = "".join(path_parts)

        # Sanitize LaTeX escapes in path: \{ -> {, \} -> }, \_ -> _
        path = raw_path.replace("\\{", "{").replace("\\}", "}")
        path = path.replace("\\_", "_")
        path = path.strip()

        # Ensure /api/v1/ prefix is present (some paths in the table
        # already include it)
        if not path.startswith("/api/v1"):
            path = "/api/v1" + path

        endpoints.append((method, path))

    return endpoints


def parse_cli_subcommand_table(tex_content: str) -> list[str]:
    """Extract subcommand names from the CLI subcommands table.

    Parses the tabularx table with caption "CLI subcommands" (Section 16,
    Command-Line Interface). The first column uses \\ttfamily formatting
    (from the column spec), so subcommand names appear as plain text
    without \\texttt{} wrappers:
        gui  & Description \\\\

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A sorted list of subcommand name strings (e.g., ["doctor",
        "export", "gui", ...]).
    """
    body = _find_table_body(tex_content, r"CLI subcommands")
    if not body:
        return []

    subcommands: list[str] = []

    rows = _join_multiline_rows(body)
    for row in rows:
        # Skip header row containing \textbf{Subcommand} or \normalfont
        if "\\textbf" in row or "\\normalfont" in row:
            continue

        cells = row.split("&")
        if len(cells) < 2:
            continue

        # First cell contains the subcommand name in ttfamily context.
        # Strip trailing \\ and whitespace.
        name = cells[0].strip().rstrip("\\").strip()
        if not name or name.startswith("\\"):
            continue

        subcommands.append(name)

    return sorted(subcommands)


def parse_labels_and_refs(
    tex_content: str,
) -> tuple[set[str], set[str]]:
    """Extract all \\label{} definitions and \\ref{} usages from the document.

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A tuple (labels, refs) where each is a set of identifier strings
        (e.g., {"sec:cli", "sec:search", ...}).
    """
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex_content))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex_content))
    return labels, refs


def parse_mcp_tool_mentions(tex_content: str) -> set[str]:
    """Find all neuroncite_* tool names mentioned anywhere in the document.

    Scans for \\texttt{neuroncite\\_xxx} patterns throughout the entire
    LaTeX document, not just the catalog table. This catches tool names
    referenced in prose, interface exposure sections, and module structure
    descriptions.

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A set of sanitized tool name strings.
    """
    raw_matches = re.findall(r"\\texttt\{(neuroncite[^}]+)\}", tex_content)
    names: set[str] = set()
    for raw in raw_matches:
        name = sanitize_latex_name(raw)
        # Filter to MCP tool-like names only:
        # - Must contain underscore (tool names use snake_case)
        # - Must not start with "neuroncite-" (crate names use hyphens)
        # - Must not contain "::" (Rust module paths like neuroncite_core::offset)
        # - Must not contain "/" or "." (file paths, URLs)
        if (
            "_" in name
            and not name.startswith("neuroncite-")
            and "::" not in name
            and "/" not in name
            and "." not in name
        ):
            names.add(name)
    return names


def parse_mcp_subcommand_claims(tex_content: str) -> tuple[int | None, int]:
    """Extract the MCP subcommand count claim and actual listed items.

    Searches for a prose claim like "four MCP subcommands" and counts
    the actual \\item entries in the following description environment
    that match the pattern \\texttt{neuroncite mcp ...}.

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A tuple (claimed_count, actual_count) where claimed_count is the
        number word converted to an integer (or None if not found), and
        actual_count is the number of \\item entries found.
    """
    # Find the claim "N MCP subcommands"
    claim_match = re.search(
        r"(\w+(?:-\w+)?)\s+MCP\s+subcommands",
        tex_content,
        re.IGNORECASE,
    )
    claimed_count = None
    if claim_match:
        claimed_count = parse_number_word(claim_match.group(1))

    # Count the actual \item entries for MCP subcommands
    # They follow the pattern: \item[\texttt{neuroncite mcp ...}]
    actual_count = len(
        re.findall(r"\\item\[\\texttt\{neuroncite\s+mcp\s+\w+\}\]", tex_content)
    )

    return claimed_count, actual_count


# ---------------------------------------------------------------------------
# Code parsing functions
# ---------------------------------------------------------------------------

def parse_mcp_tools_from_code(root: Path) -> list[str]:
    """Extract all MCP tool names from the tools.rs source file.

    Reads the all_tools() function in the neuroncite-mcp crate and
    extracts tool names from ToolDefinition structs. Each tool has a
    name field with a string literal like:  name: "neuroncite_search"

    Args:
        root: The Cargo workspace root directory.

    Returns:
        A sorted list of tool name strings.
    """
    tools_path = root / "crates" / "neuroncite-mcp" / "src" / "tools.rs"
    content = tools_path.read_text(encoding="utf-8")

    # Extract tool names from  name: "neuroncite_xxx"  patterns inside
    # ToolDefinition struct literals
    names = re.findall(r'name:\s*"(neuroncite_\w+)"', content)
    return sorted(names)


def parse_api_routes_from_code(root: Path) -> list[tuple[str, str]]:
    """Extract all REST API route definitions from router.rs.

    Reads the build_router() function in the neuroncite-api crate and
    extracts .route("/path", method(handler)) calls. All routes under
    the api_routes Router are nested under /api/v1. The standalone
    /api/v1/openapi.json route is also captured.

    Args:
        root: The Cargo workspace root directory.

    Returns:
        A list of (method_upper, full_path) tuples, e.g.,
        [("GET", "/api/v1/health"), ("POST", "/api/v1/index"), ...].
    """
    router_path = root / "crates" / "neuroncite-api" / "src" / "router.rs"
    content = router_path.read_text(encoding="utf-8")

    routes: list[tuple[str, str]] = []

    # Match .route("path", method(handler)) patterns. The method
    # function name (get, post, delete) indicates the HTTP method.
    # Handles both single-line and multi-line .route() calls.
    route_pattern = re.compile(
        r'\.route\(\s*"([^"]+)"\s*,\s*(get|post|put|delete|patch)\s*\(',
        re.DOTALL,
    )

    for match in route_pattern.finditer(content):
        path = match.group(1)
        method = match.group(2).upper()

        # Routes defined inside the api_routes Router are nested under
        # /api/v1. The standalone openapi.json route already includes
        # the full path.
        if not path.startswith("/api/v1"):
            path = "/api/v1" + path

        routes.append((method, path))

    return sorted(routes)


def parse_cli_subcommands_from_code(root: Path) -> list[str]:
    """Extract CLI subcommand names from the main.rs Command enum.

    Reads the clap-derived Command enum in the binary crate's main.rs.
    Each PascalCase enum variant becomes a lowercase subcommand name
    (clap's default kebab-case conversion). Feature-gated variants
    (e.g., Mcp behind #[cfg(feature = "mcp")]) are included because
    documentation should cover all features.

    Args:
        root: The Cargo workspace root directory.

    Returns:
        A sorted list of subcommand names (e.g., ["annotate", "doctor",
        "export", "gui", ...]).
    """
    main_path = root / "crates" / "neuroncite" / "src" / "main.rs"
    content = main_path.read_text(encoding="utf-8")

    # Find the enum Command block. The enum starts with "enum Command {"
    # and ends at the matching closing brace.
    enum_match = re.search(r"enum\s+Command\s*\{", content)
    if not enum_match:
        return []

    # Find the matching closing brace by tracking brace depth
    start = enum_match.end()
    depth = 1
    pos = start
    while pos < len(content) and depth > 0:
        if content[pos] == "{":
            depth += 1
        elif content[pos] == "}":
            depth -= 1
        pos += 1

    enum_body = content[start:pos - 1]

    # Extract PascalCase variant names. Each variant is an identifier
    # at the start of a line (after optional whitespace and attributes),
    # followed by { or , or (. Skip attribute lines (#[...]).
    subcommands: list[str] = []
    for line in enum_body.splitlines():
        stripped = line.strip()
        # Skip empty lines, comments, and attributes
        if not stripped or stripped.startswith("//") or stripped.startswith("#"):
            continue
        # Skip closing braces from previous variant's struct body
        if stripped.startswith("}"):
            continue

        # Match a PascalCase identifier at the start of the line
        variant_match = re.match(r"([A-Z][a-zA-Z]*)", stripped)
        if variant_match:
            variant_name = variant_match.group(1)
            # Convert PascalCase to lowercase (all current variants are
            # single words, so simple lowercasing is correct)
            subcommands.append(variant_name.lower())

    return sorted(set(subcommands))


# ---------------------------------------------------------------------------
# Validation check functions
# ---------------------------------------------------------------------------

def check_crate_count(tex_content: str) -> list[Diagnostic]:
    """Check 1: Verify that the prose crate count claim matches the table.

    Compares the "N library crates and M binary crate(s)" prose claim
    (line 238) against the actual number of crates listed in the workspace
    crate overview table.

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A list of Diagnostic instances for any mismatches.
    """
    diagnostics: list[Diagnostic] = []

    library_crates, binary_crates = parse_crate_table(tex_content)
    claimed_lib, claimed_bin = parse_crate_count_claim(tex_content)

    if claimed_lib is not None and claimed_lib != len(library_crates):
        diagnostics.append(Diagnostic(
            level="error",
            category="crate_count",
            section="crate_overview",
            detail=(
                f"prose claims {claimed_lib} library crates but the "
                f"crate table lists {len(library_crates)}: "
                f"{', '.join(library_crates)}"
            ),
        ))

    if claimed_bin is not None and claimed_bin != len(binary_crates):
        diagnostics.append(Diagnostic(
            level="error",
            category="crate_count",
            section="crate_overview",
            detail=(
                f"prose claims {claimed_bin} binary crates but the "
                f"crate table lists {len(binary_crates)}: "
                f"{', '.join(binary_crates)}"
            ),
        ))

    return diagnostics


def check_mcp_tool_count(
    tex_content: str, code_tools: list[str],
) -> list[Diagnostic]:
    """Check 2: Verify that prose tool-count claims match the code.

    Compares all prose claims about the number of MCP tools (e.g., "twenty
    tools", "fourteen tools") against the actual number of tool definitions
    in tools.rs.

    Args:
        tex_content: The full LaTeX document content.
        code_tools: The list of tool names extracted from tools.rs.

    Returns:
        A list of Diagnostic instances for any mismatches.
    """
    diagnostics: list[Diagnostic] = []
    actual_count = len(code_tools)

    claims = parse_mcp_tool_count_claims(tex_content)
    for phrase, claimed_count in claims:
        if claimed_count != actual_count:
            diagnostics.append(Diagnostic(
                level="error",
                category="mcp_tool_count",
                section="mcp_tools",
                detail=(
                    f'prose "{phrase}" claims {claimed_count} tools '
                    f"but tools.rs defines {actual_count}"
                ),
            ))

    return diagnostics


def check_mcp_tool_catalog(
    tex_content: str, code_tools: list[str],
) -> list[Diagnostic]:
    """Check 3: Verify bidirectional consistency between the MCP tool catalog
    table and the tool definitions in tools.rs.

    Every tool defined in code should appear in the catalog table, and
    every tool listed in the catalog table should exist in code.

    Args:
        tex_content: The full LaTeX document content.
        code_tools: The list of tool names extracted from tools.rs.

    Returns:
        A list of Diagnostic instances for any mismatches.
    """
    diagnostics: list[Diagnostic] = []
    catalog_tools = parse_mcp_tool_catalog(tex_content)

    code_set = set(code_tools)
    catalog_set = set(catalog_tools)

    # Tools in code but missing from the catalog table
    for tool in sorted(code_set - catalog_set):
        diagnostics.append(Diagnostic(
            level="error",
            category="mcp_tool_catalog",
            section="mcp_tools",
            detail=(
                f"tool '{tool}' defined in tools.rs but missing from "
                f"the MCP tool catalog table"
            ),
        ))

    # Tools in the catalog table but not in code
    for tool in sorted(catalog_set - code_set):
        diagnostics.append(Diagnostic(
            level="error",
            category="mcp_tool_catalog",
            section="mcp_tools",
            detail=(
                f"tool '{tool}' listed in the MCP tool catalog table "
                f"but not defined in tools.rs"
            ),
        ))

    return diagnostics


def check_mcp_tool_mentions(
    tex_content: str, code_tools: list[str],
) -> list[Diagnostic]:
    """Check 4: Verify that tool names mentioned in the document exist in code.

    Scans the entire document for \\texttt{neuroncite_xxx} mentions and
    checks whether each mentioned tool name exists in tools.rs. This
    catches stale tool references in prose sections.

    Args:
        tex_content: The full LaTeX document content.
        code_tools: The list of tool names extracted from tools.rs.

    Returns:
        A list of Diagnostic instances (warnings) for any stale mentions.
    """
    diagnostics: list[Diagnostic] = []
    mentioned = parse_mcp_tool_mentions(tex_content)
    code_set = set(code_tools)

    for name in sorted(mentioned - code_set):
        diagnostics.append(Diagnostic(
            level="warning",
            category="mcp_tool_mentions",
            section="mcp_tools",
            detail=(
                f"tool name '{name}' mentioned in the document but "
                f"not defined in tools.rs"
            ),
        ))

    return diagnostics


def check_api_endpoint_table(
    tex_content: str, code_routes: list[tuple[str, str]],
) -> list[Diagnostic]:
    """Check 5: Verify bidirectional consistency between the REST API endpoint
    table and the route definitions in router.rs.

    Every route in code should appear in the endpoint table, and every
    endpoint listed in the table should have a corresponding route in code.

    Args:
        tex_content: The full LaTeX document content.
        code_routes: The list of (method, path) tuples from router.rs.

    Returns:
        A list of Diagnostic instances for any mismatches.
    """
    diagnostics: list[Diagnostic] = []
    tex_endpoints = parse_api_endpoint_table(tex_content)

    code_set = set(code_routes)
    tex_set = set(tex_endpoints)

    # Routes in code but missing from the endpoint table
    for method, path in sorted(code_set - tex_set):
        diagnostics.append(Diagnostic(
            level="error",
            category="api_endpoint_table",
            section="api_endpoints",
            detail=(
                f"route {method} {path} exists in router.rs but is "
                f"missing from the REST API endpoint table"
            ),
        ))

    # Endpoints in the table but not in code
    for method, path in sorted(tex_set - code_set):
        diagnostics.append(Diagnostic(
            level="error",
            category="api_endpoint_table",
            section="api_endpoints",
            detail=(
                f"endpoint {method} {path} listed in the REST API "
                f"endpoint table but not defined in router.rs"
            ),
        ))

    return diagnostics


def check_cli_subcommand_table(
    tex_content: str, code_subcommands: list[str],
) -> list[Diagnostic]:
    """Check 6: Verify bidirectional consistency between the CLI subcommands
    table and the clap Command enum in main.rs.

    Every enum variant should appear in the subcommands table, and every
    subcommand listed in the table should have a corresponding variant.

    Args:
        tex_content: The full LaTeX document content.
        code_subcommands: The list of subcommand names from main.rs.

    Returns:
        A list of Diagnostic instances for any mismatches.
    """
    diagnostics: list[Diagnostic] = []
    tex_subcommands = parse_cli_subcommand_table(tex_content)

    code_set = set(code_subcommands)
    tex_set = set(tex_subcommands)

    # Subcommands in code but missing from the table
    for name in sorted(code_set - tex_set):
        diagnostics.append(Diagnostic(
            level="error",
            category="cli_subcommand_table",
            section="cli_subcommands",
            detail=(
                f"subcommand '{name}' exists in main.rs Command enum "
                f"but is missing from the CLI subcommands table"
            ),
        ))

    # Subcommands in the table but not in code
    for name in sorted(tex_set - code_set):
        diagnostics.append(Diagnostic(
            level="error",
            category="cli_subcommand_table",
            section="cli_subcommands",
            detail=(
                f"subcommand '{name}' listed in the CLI subcommands "
                f"table but not in the main.rs Command enum"
            ),
        ))

    return diagnostics


def check_label_integrity(tex_content: str) -> list[Diagnostic]:
    """Check 7: Verify that all LaTeX labels and references resolve.

    Checks two conditions:
      - Every \\ref{xyz} points to a \\label{xyz} that exists (dangling
        reference -> warning).
      - Every \\label{xyz} has at least one \\ref{xyz} (orphaned label
        -> warning).

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A list of Diagnostic instances (warnings) for any integrity issues.
    """
    diagnostics: list[Diagnostic] = []
    labels, refs = parse_labels_and_refs(tex_content)

    # References to undefined labels
    for ref_id in sorted(refs - labels):
        diagnostics.append(Diagnostic(
            level="warning",
            category="label_integrity",
            section="labels",
            detail=(
                f"\\ref{{{ref_id}}} references a label that is not "
                f"defined anywhere in the document"
            ),
        ))

    # Labels that are never referenced.
    # Unreferenced labels are downgraded to "info" (not warning) because
    # LaTeX labels serve purposes beyond \ref{}: hyperref PDF bookmarks,
    # external linking from other documents, and nameref navigation. An
    # orphaned label is not a consistency problem.
    for label_id in sorted(labels - refs):
        diagnostics.append(Diagnostic(
            level="info",
            category="label_integrity",
            section="labels",
            detail=(
                f"\\label{{{label_id}}} is defined but never referenced "
                f"by any \\ref{{}}"
            ),
        ))

    return diagnostics


def check_numeric_claims(tex_content: str) -> list[Diagnostic]:
    """Check 8: Verify specific countable numeric claims in prose.

    Checks claims where a number word is followed by a countable set
    of items that can be verified within the document itself. Currently
    checks:
      - "N MCP subcommands" vs. the actual \\item entries in the
        following description environment.

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A list of Diagnostic instances (warnings) for any mismatches.
    """
    diagnostics: list[Diagnostic] = []

    # Check "N MCP subcommands" claim (line 4913) against actual items
    claimed, actual = parse_mcp_subcommand_claims(tex_content)
    if claimed is not None and claimed != actual:
        diagnostics.append(Diagnostic(
            level="warning",
            category="numeric_claims",
            section="mcp_subcommands",
            detail=(
                f"prose claims {claimed} MCP subcommands but "
                f"{actual} \\item entries are listed"
            ),
        ))

    return diagnostics


def parse_verdict_variants_from_code(root: Path) -> list[str]:
    """Extract Verdict enum variant names from types.rs in neuroncite-citation.

    Reads the Verdict enum definition and collects PascalCase variant names.
    Variants are identified as identifiers appearing on their own line
    inside the enum body, ignoring doc comments and attributes.

    Args:
        root: The Cargo workspace root directory.

    Returns:
        A list of variant name strings (e.g., ["Supported", "Partial", ...]).
    """
    types_path = root / "crates" / "neuroncite-citation" / "src" / "types.rs"
    content = types_path.read_text(encoding="utf-8")

    enum_match = re.search(r"pub\s+enum\s+Verdict\s*\{", content)
    if not enum_match:
        return []

    start = enum_match.end()
    depth = 1
    pos = start
    while pos < len(content) and depth > 0:
        if content[pos] == "{":
            depth += 1
        elif content[pos] == "}":
            depth -= 1
        pos += 1

    enum_body = content[start:pos - 1]
    variants: list[str] = []
    for line in enum_body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("#"):
            continue
        variant_match = re.match(r"([A-Z][a-zA-Z]*)", stripped)
        if variant_match:
            variants.append(variant_match.group(1))

    return variants


def parse_pipeline_stage_count_from_code(root: Path) -> int:
    """Count pipeline stages documented in locate.rs of neuroncite-annotate.

    Scans the module-level doc comment for "Stage N" or "Stage N.N" markers
    to determine how many stages the pipeline has.

    Args:
        root: The Cargo workspace root directory.

    Returns:
        The number of distinct stages found (counting N.N as a separate stage).
    """
    locate_path = root / "crates" / "neuroncite-annotate" / "src" / "locate.rs"
    content = locate_path.read_text(encoding="utf-8")

    # Match "Stage 1", "Stage 2", "Stage 3", "Stage 3.5", "Stage 4" etc.
    stages = re.findall(r"Stage\s+(\d+(?:\.\d+)?)", content)
    return len(set(stages))


def parse_default_features_from_cargo(root: Path) -> list[str]:
    """Extract the default feature list from the binary crate's Cargo.toml.

    Reads the [features] section of crates/neuroncite/Cargo.toml and
    parses the default = [...] array.

    Args:
        root: The Cargo workspace root directory.

    Returns:
        A sorted list of default feature names.
    """
    cargo_path = root / "crates" / "neuroncite" / "Cargo.toml"
    content = cargo_path.read_text(encoding="utf-8")

    match = re.search(r'default\s*=\s*\[([^\]]+)\]', content)
    if not match:
        return []

    raw = match.group(1)
    features = [f.strip().strip('"').strip("'") for f in raw.split(",")]
    return sorted(f for f in features if f)


def check_verdict_count(tex_content: str, root: Path) -> list[Diagnostic]:
    """Check 9: Verify that prose verdict-count claims match the Verdict enum.

    Scans the LaTeX document for patterns like "six verdict types" or
    "seven verdict types" and compares the claimed count against the actual
    number of variants in the Verdict enum in types.rs.

    Args:
        tex_content: The full LaTeX document content.
        root: The Cargo workspace root directory.

    Returns:
        A list of Diagnostic instances for any mismatches.
    """
    diagnostics: list[Diagnostic] = []
    variants = parse_verdict_variants_from_code(root)
    actual_count = len(variants)

    if actual_count == 0:
        return diagnostics

    pattern = r"(\w+(?:-\w+)?)\s+verdict\s+types?"
    for match in re.finditer(pattern, tex_content, re.IGNORECASE):
        word = match.group(1)
        claimed = parse_number_word(word)
        if claimed is not None and claimed != actual_count:
            diagnostics.append(Diagnostic(
                level="error",
                category="verdict_count",
                section="citation_verdicts",
                detail=(
                    f'prose claims "{match.group(0)}" ({claimed}) '
                    f"but the Verdict enum has {actual_count} variants: "
                    f"{', '.join(variants)}"
                ),
            ))

    return diagnostics


def check_pipeline_stage_count(
    tex_content: str, root: Path,
) -> list[Diagnostic]:
    """Check 10: Verify that prose pipeline-stage claims match locate.rs.

    Scans the LaTeX document for "N-stage" references in annotation pipeline
    context and compares against the actual stage count in locate.rs.

    Args:
        tex_content: The full LaTeX document content.
        root: The Cargo workspace root directory.

    Returns:
        A list of Diagnostic instances for any mismatches.
    """
    diagnostics: list[Diagnostic] = []
    actual_stages = parse_pipeline_stage_count_from_code(root)

    if actual_stages == 0:
        return diagnostics

    # Match "N-stage" or "N-Stage" patterns (e.g., "4-stage", "5-Stage")
    pattern = r"(\d+)-[Ss]tage"
    for match in re.finditer(pattern, tex_content):
        claimed = int(match.group(1))
        if claimed != actual_stages:
            diagnostics.append(Diagnostic(
                level="error",
                category="pipeline_stage_count",
                section="annotation_pipeline",
                detail=(
                    f'prose claims "{match.group(0)}" but locate.rs '
                    f"documents {actual_stages} stages"
                ),
            ))

    return diagnostics


def check_default_features(tex_content: str, root: Path) -> list[Diagnostic]:
    """Check 11: Verify that build documentation matches Cargo.toml defaults.

    Searches the LaTeX document for the build command description of
    ``cargo build --release`` and checks whether the feature list matches
    the actual default features in Cargo.toml.

    Args:
        tex_content: The full LaTeX document content.
        root: The Cargo workspace root directory.

    Returns:
        A list of Diagnostic instances for any mismatches.
    """
    diagnostics: list[Diagnostic] = []
    actual_features = parse_default_features_from_cargo(root)

    if not actual_features:
        return diagnostics

    # Check for "default features (xxx)" pattern near "cargo build --release"
    match = re.search(
        r"default\s+features\s*\(([^)]+)\)",
        tex_content,
    )
    if match:
        claimed_text = match.group(1).strip()
        # Parse the claimed feature list
        claimed_features = sorted(
            f.strip().replace("-{}-", "--")
            for f in claimed_text.split(",")
            if f.strip()
        )

        if claimed_features != actual_features:
            diagnostics.append(Diagnostic(
                level="error",
                category="default_features",
                section="build_commands",
                detail=(
                    f'build docs claim default features "{claimed_text}" '
                    f"but Cargo.toml defines: {', '.join(actual_features)}"
                ),
            ))

    return diagnostics


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_human_readable(diagnostics: list[Diagnostic]) -> str:
    """Format a list of diagnostics as human-readable text lines.

    Each line has the format:
        [ERROR|WARN] <category>: <section> -- <detail>

    A summary line at the end reports total error and warning counts.

    Args:
        diagnostics: The list of Diagnostic instances to format.

    Returns:
        A multi-line string with one diagnostic per line plus a summary.
    """
    lines: list[str] = []
    error_count = 0
    warning_count = 0

    info_count = 0

    for diag in diagnostics:
        if diag.level == "error":
            prefix = "[ERROR]"
            error_count += 1
        elif diag.level == "info":
            prefix = "[INFO]"
            info_count += 1
        else:
            prefix = "[WARN]"
            warning_count += 1

        lines.append(f"{prefix} {diag.category}: {diag.section} -- {diag.detail}")

    summary = f"SUMMARY: {error_count} error(s), {warning_count} warning(s)"
    if info_count:
        summary += f", {info_count} info(s)"
    lines.append(summary)
    return "\n".join(lines)


def format_json(diagnostics: list[Diagnostic]) -> str:
    """Format a list of diagnostics as a JSON string.

    The JSON object has three keys: "errors" (list of error entries),
    "warnings" (list of warning entries), and "summary" (counts).
    Each entry has "category", "section", and "detail" fields.

    Args:
        diagnostics: The list of Diagnostic instances to format.

    Returns:
        A JSON string with 2-space indentation.
    """
    errors = []
    warnings = []
    infos = []

    for diag in diagnostics:
        entry = {
            "category": diag.category,
            "section": diag.section,
            "detail": diag.detail,
        }
        if diag.level == "error":
            errors.append(entry)
        elif diag.level == "info":
            infos.append(entry)
        else:
            warnings.append(entry)

    result = {
        "errors": errors,
        "warnings": warnings,
        "infos": infos,
        "summary": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "info_count": len(infos),
        },
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Validation orchestration
# ---------------------------------------------------------------------------

def run_validation(tex_path: Path, root_path: Path) -> list[Diagnostic]:
    """Run all 11 validation checks and return the combined diagnostics.

    Reads the LaTeX document and the relevant Rust source files, parses
    both sides, and executes each check function.

    Args:
        tex_path: Absolute path to architecture.tex.
        root_path: Absolute path to the Cargo workspace root.

    Returns:
        A combined list of Diagnostic instances from all checks.
    """
    raw_tex = tex_path.read_text(encoding="utf-8")
    tex_content = resolve_inputs(raw_tex, tex_path.parent)

    # Parse code sources
    code_tools = parse_mcp_tools_from_code(root_path)
    code_routes = parse_api_routes_from_code(root_path)
    code_subcommands = parse_cli_subcommands_from_code(root_path)

    # Run all checks
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(check_crate_count(tex_content))
    diagnostics.extend(check_mcp_tool_count(tex_content, code_tools))
    diagnostics.extend(check_mcp_tool_catalog(tex_content, code_tools))
    diagnostics.extend(check_mcp_tool_mentions(tex_content, code_tools))
    diagnostics.extend(check_api_endpoint_table(tex_content, code_routes))
    diagnostics.extend(check_cli_subcommand_table(tex_content, code_subcommands))
    diagnostics.extend(check_label_integrity(tex_content))
    diagnostics.extend(check_numeric_claims(tex_content))
    diagnostics.extend(check_verdict_count(tex_content, root_path))
    diagnostics.extend(check_pipeline_stage_count(tex_content, root_path))
    diagnostics.extend(check_default_features(tex_content, root_path))

    return diagnostics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse command-line arguments, run validation, and produce output.

    Returns:
        Exit code: 0 if all checks pass, 1 if discrepancies found, 2 if
        the script cannot parse the LaTeX file or locate required source
        files.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Cross-Section Consistency Validator for NeuronCite. "
            "Checks MCP tool catalogs, REST API endpoint tables, CLI "
            "subcommand tables, crate counts, label/ref integrity, and "
            "numeric prose claims for consistency between the LaTeX "
            "architecture document and the Rust source code."
        ),
    )
    parser.add_argument(
        "--tex",
        required=True,
        type=Path,
        help="Path to the LaTeX architecture document (e.g., docs/architecture.tex)",
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to the Cargo workspace root directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat warnings as errors (used in CI to block the "
            "pipeline on any discrepancy)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable text",
    )
    args = parser.parse_args()

    # Resolve paths and validate existence
    tex_path: Path = args.tex.resolve()
    root_path: Path = args.root.resolve()

    if not tex_path.is_file():
        print(f"Error: LaTeX file not found: {tex_path}", file=sys.stderr)
        return 2

    if not root_path.is_dir():
        print(
            f"Error: workspace root directory not found: {root_path}",
            file=sys.stderr,
        )
        return 2

    # Verify that the required Rust source files exist
    required_files = [
        root_path / "crates" / "neuroncite-mcp" / "src" / "tools.rs",
        root_path / "crates" / "neuroncite-api" / "src" / "router.rs",
        root_path / "crates" / "neuroncite" / "src" / "main.rs",
    ]
    for req_file in required_files:
        if not req_file.is_file():
            print(
                f"Error: required source file not found: {req_file}",
                file=sys.stderr,
            )
            return 2

    # Run validation
    try:
        diagnostics = run_validation(tex_path, root_path)
    except Exception as exc:
        print(f"Error: validation failed: {exc}", file=sys.stderr)
        return 2

    # Format and output results
    if args.json:
        print(format_json(diagnostics))
    else:
        if diagnostics:
            print(format_human_readable(diagnostics), file=sys.stderr)
        else:
            print("All checks passed.", file=sys.stderr)

    # Determine exit code
    error_count = sum(1 for d in diagnostics if d.level == "error")
    warning_count = sum(1 for d in diagnostics if d.level == "warning")

    if error_count > 0:
        return 1
    if args.strict and warning_count > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
