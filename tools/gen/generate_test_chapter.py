#!/usr/bin/env python3
"""Test Chapter Generator for NeuronCite.

Scans all Rust source files (.rs) under the crates/ directory of the NeuronCite
Cargo workspace and extracts test IDs from doc comments matching the pattern
``/// T-PREFIX-NNN: description text``. Produces a complete LaTeX file containing
longtable catalogs (unit, integration, property/regression, benchmark) and a
summary table suitable for inclusion in the architecture document.

The generated LaTeX reproduces the exact formatting conventions established in
Section 19 of architecture.tex, including longtable column widths, multicolumn
crate headers, and tabularx summary rows.

This is the companion script to validate_tests.py, which validates the consistency
between an existing LaTeX test chapter and the Rust source code. This script
generates the LaTeX from scratch, while validate_tests.py checks an existing
document.

Test ID format:
    T-PREFIX[-SUBPREFIX]-NNN[letter]
    - PREFIX: uppercase letters and digits (CORE, CLI, E2E, MCP, etc.)
    - SUBPREFIX: optional additional segment (MCP-BATCH, MCP-IDX, ANN-BUG, etc.)
    - NNN: exactly 3 digits
    - Optional lowercase letter suffix (e.g., 024a)

Category classification by file path:
    - crates/<name>/tests/integration/  -> integration
    - crates/<name>/tests/property/     -> property
    - crates/<name>/tests/regression/   -> regression
    - crates/<name>/benches/            -> benchmark
    - Everything else (src/)            -> unit

Exit codes:
    0  Generation completed without warnings.
    1  Generation completed with warnings (e.g., duplicate test IDs).
    2  Fatal error (missing crates/ directory, I/O failure).

Usage:
    python tools/generate_test_chapter.py --root . --output docs/tests_generated.tex [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Test ID regex pattern
# ---------------------------------------------------------------------------
# Matches test ID doc comment lines like:
#   /// T-CORE-001: `PageText` construction with valid fields. All fields are
#   /// T-MCP-BATCH-003: Batch processing of citation entries in parallel.
# Group 1 captures the full test ID (e.g., T-CORE-001).
# Group 2 captures the description text after the colon.
TEST_ID_LINE_PATTERN = re.compile(
    r"///\s+(T-[A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*-\d{3}[a-z]?):\s*(.*)",
)

# Matches continuation lines that are part of the same doc comment block.
# These lines start with /// followed by text that is NOT a test ID header.
DOC_CONTINUATION_PATTERN = re.compile(r"///\s?(.*)")

# Matches Rust backtick code spans like `PageText` for conversion to \texttt{}.
BACKTICK_SPAN_PATTERN = re.compile(r"`([^`]+)`")


# ---------------------------------------------------------------------------
# Crate descriptions (derived from each crate's Cargo.toml)
# ---------------------------------------------------------------------------
CRATE_DESCRIPTIONS: dict[str, str] = {
    "neuroncite-annotate": "PDF highlight annotation and text location pipeline",
    "neuroncite-api": "REST API server and GPU worker",
    "neuroncite-chunk": "Text chunking strategies",
    "neuroncite-citation": "LaTeX citation verification pipeline",
    "neuroncite-core": "Shared domain types, traits, and configuration",
    "neuroncite-embed": "Embedding backend abstraction and ML framework integrations",
    "neuroncite-html": "HTML web scraping, parsing, section splitting, and crawling",
    "neuroncite-llm": "LLM integration for text generation",
    "neuroncite-mcp": "MCP (Model Context Protocol) server",
    "neuroncite-pdf": "PDF file discovery and text extraction",
    "neuroncite-pipeline": "Background job executor, GPU worker, and indexing pipeline",
    "neuroncite-search": "Hybrid search pipeline (vector + keyword)",
    "neuroncite-store": "SQLite storage, HNSW index management, and workflow tracking",
    "neuroncite-web": "Browser-based web frontend server",
}

# Integration test source file grouping. Maps source file base names to
# human-readable sub-category labels for the integration test catalog.
INTEGRATION_FILE_GROUPS: dict[str, str] = {
    "pipeline.rs": "Full pipeline integration tests",
    "search.rs": "Full pipeline integration tests",
    "real_pdf.rs": "Full pipeline integration tests",
    "ocr_fallback.rs": "Full pipeline integration tests",
    "cli.rs": "CLI integration tests",
    "server.rs": "API server integration tests",
}

# Ordering for integration sub-categories in the generated LaTeX.
INTEGRATION_GROUP_ORDER: list[str] = [
    "Full pipeline integration tests",
    "CLI integration tests",
    "API server integration tests",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TestEntry:
    """A single test extracted from a Rust doc comment block.

    Attributes:
        test_id: The full test identifier (e.g., "T-CORE-001").
        description: The short behavior description (text after the colon
            up to the first sentence boundary).
        acceptance: The acceptance criterion (remaining text after the
            description sentence).
        source_file: The relative path of the .rs file where this test
            was found (e.g., "crates/neuroncite-core/src/types.rs").
        crate_name: The crate name derived from the file path
            (e.g., "neuroncite-core").
        category: One of "unit", "integration", "property", "regression",
            or "benchmark".
        integration_group: For integration tests, the sub-category label
            (e.g., "Full pipeline integration tests"). Empty string for
            non-integration tests.
    """

    test_id: str
    description: str
    acceptance: str
    source_file: str
    crate_name: str
    category: str
    integration_group: str = ""


@dataclass
class GenerationResult:
    """Aggregated results from scanning the Rust workspace.

    Attributes:
        entries: All unique test entries, sorted by their natural ordering.
        duplicates: List of (test_id, file_path) tuples for tests that
            appeared more than once. Only the first occurrence is kept
            in entries.
        files_scanned: Total number of .rs files that were read.
        warnings: Human-readable warning messages emitted during scanning.
    """

    entries: list[TestEntry] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    files_scanned: int = 0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LaTeX character escaping
# ---------------------------------------------------------------------------

# Mapping of characters that require LaTeX escaping. The order matters:
# backslash must be escaped first to avoid double-escaping the backslashes
# introduced by other replacements.
_LATEX_ESCAPE_MAP: list[tuple[str, str]] = [
    ("\\", "\\textbackslash{}"),
    ("_", "\\_"),
    ("&", "\\&"),
    ("%", "\\%"),
    ("#", "\\#"),
    ("$", "\\$"),
    ("{", "\\{"),
    ("}", "\\}"),
    ("~", "\\~{}"),
    ("^", "\\^{}"),
]


def escape_latex(text: str) -> str:
    r"""Escape characters that have special meaning in LaTeX.

    Handles the standard set of LaTeX special characters and converts
    Rust backtick code spans (e.g., `Foo`) to \texttt{Foo}.

    The escaping is applied before backtick conversion so that code span
    contents are also escaped. The backtick conversion then wraps the
    escaped content in \texttt{}.

    Args:
        text: Plain text that may contain LaTeX-special characters and
            Rust-style backtick code spans.

    Returns:
        The text with all special characters escaped and code spans
        converted to \texttt{} commands.
    """
    # First, extract backtick spans and protect them from escaping.
    # Collect all backtick-delimited segments so their content can be
    # escaped independently and wrapped in \texttt{}.
    segments: list[str] = []
    last_end = 0

    for match in BACKTICK_SPAN_PATTERN.finditer(text):
        # Escape the plain text before this code span
        before = text[last_end:match.start()]
        for char, replacement in _LATEX_ESCAPE_MAP:
            before = before.replace(char, replacement)
        segments.append(before)

        # Escape the code span content and wrap in \texttt{}
        code_content = match.group(1)
        for char, replacement in _LATEX_ESCAPE_MAP:
            code_content = code_content.replace(char, replacement)
        segments.append(f"\\texttt{{{code_content}}}")

        last_end = match.end()

    # Escape any remaining text after the last code span
    remaining = text[last_end:]
    for char, replacement in _LATEX_ESCAPE_MAP:
        remaining = remaining.replace(char, replacement)
    segments.append(remaining)

    return "".join(segments)


# ---------------------------------------------------------------------------
# Test ID parsing utilities
# ---------------------------------------------------------------------------

def _base_prefix(test_id: str) -> str:
    """Extract the base prefix from a test ID, excluding "T-" and the numeric suffix.

    Examples:
        T-CORE-001    -> "CORE"
        T-MCP-IDX-006 -> "MCP-IDX"
        T-CORE-024a   -> "CORE"

    Args:
        test_id: A full test ID string.

    Returns:
        The prefix portion without "T-" and the trailing "-NNN[a-z]?" suffix.
    """
    rest = test_id[2:]  # Strip "T-"
    match = re.search(r"-(\d{3}[a-z]?)$", rest)
    if match:
        return rest[:match.start()]
    return rest


def _numeric_part(test_id: str) -> int:
    """Extract the 3-digit numeric portion of a test ID as an integer.

    Ignores the optional trailing letter suffix. For example,
    T-CORE-024a returns 24.

    Args:
        test_id: A full test ID string.

    Returns:
        The integer value of the 3-digit numeric part, or 0 if no
        numeric part is found.
    """
    match = re.search(r"(\d{3})[a-z]?$", test_id)
    if match:
        return int(match.group(1))
    return 0


def _sort_key(entry: TestEntry) -> tuple[str, int, str]:
    """Produce a sort key for ordering test entries within their category.

    Sorts by (prefix, numeric_part, full_test_id) so that entries are
    grouped by their prefix chain and ordered numerically within each
    group. The full test_id as the third element ensures stable ordering
    for letter-suffixed variants (e.g., 024a before 024b).

    Args:
        entry: A TestEntry instance.

    Returns:
        A 3-tuple suitable for sorted() comparison.
    """
    return (_base_prefix(entry.test_id), _numeric_part(entry.test_id), entry.test_id)


# ---------------------------------------------------------------------------
# File path classification
# ---------------------------------------------------------------------------

def _classify_category(rel_path: str) -> str:
    """Determine the test category based on the file's relative path.

    Classification rules (checked in order):
        - paths containing /tests/integration/ -> "integration"
        - paths containing /tests/property/    -> "property"
        - paths containing /tests/regression/  -> "regression"
        - paths containing /benches/           -> "benchmark"
        - everything else                      -> "unit"

    Args:
        rel_path: The file path relative to the workspace root, using
            forward slashes (e.g., "crates/neuroncite/tests/integration/cli.rs").

    Returns:
        One of "unit", "integration", "property", "regression", or "benchmark".
    """
    if "/tests/integration/" in rel_path:
        return "integration"
    if "/tests/property/" in rel_path:
        return "property"
    if "/tests/regression/" in rel_path:
        return "regression"
    if "/benches/" in rel_path:
        return "benchmark"
    return "unit"


def _derive_crate_name(rel_path: str) -> str:
    """Extract the crate name from a file path relative to the workspace root.

    The crate name is the directory name immediately under crates/.
    For example:
        "crates/neuroncite-core/src/types.rs" -> "neuroncite-core"
        "crates/neuroncite/tests/integration/cli.rs" -> "neuroncite"

    Args:
        rel_path: A forward-slash-separated path relative to the workspace root.

    Returns:
        The crate directory name, or "unknown" if the path does not follow
        the expected crates/<name>/... structure.
    """
    parts = rel_path.split("/")
    # Expected structure: crates/<crate-name>/...
    if len(parts) >= 2 and parts[0] == "crates":
        return parts[1]
    return "unknown"


def _derive_integration_group(rel_path: str) -> str:
    """Determine the integration test sub-category from the source file name.

    Uses the INTEGRATION_FILE_GROUPS mapping to assign a human-readable
    group label based on the file's base name. Files not listed in the
    mapping are assigned to "Full pipeline integration tests" as a
    fallback.

    Args:
        rel_path: The relative file path for an integration test file.

    Returns:
        A sub-category label string.
    """
    file_name = rel_path.rsplit("/", maxsplit=1)[-1]
    if file_name in INTEGRATION_FILE_GROUPS:
        return INTEGRATION_FILE_GROUPS[file_name]
    return "Full pipeline integration tests"


# ---------------------------------------------------------------------------
# Doc comment parsing
# ---------------------------------------------------------------------------

def _split_description_and_acceptance(full_text: str) -> tuple[str, str]:
    """Split a doc comment body into description and acceptance criterion.

    The description is the text from the start up to and including the first
    sentence-ending period that is followed by a space (". ") or is at the
    end of the text. The acceptance criterion is everything after that period.

    If no sentence boundary is found, the entire text is treated as the
    description and the acceptance criterion is empty.

    Args:
        full_text: The complete doc comment text after the "T-XXX-NNN:" prefix,
            with continuation lines joined by spaces.

    Returns:
        A (description, acceptance) tuple. Both strings are stripped of
        leading/trailing whitespace.
    """
    # Search for ". " as a sentence boundary. The period must follow a
    # non-whitespace character to avoid matching abbreviations like "e.g. "
    # or ellipses.
    match = re.search(r"(?<=\S)\.\s", full_text)
    if match:
        # The description includes the period; acceptance starts after the space
        split_pos = match.start() + 1  # position after the period
        description = full_text[:split_pos].strip()
        acceptance = full_text[split_pos:].strip()
        return description, acceptance

    # No sentence boundary found; check if the text ends with a period
    stripped = full_text.strip()
    if stripped.endswith("."):
        return stripped, ""

    return stripped, ""


def parse_test_entries_from_file(
    file_path: Path,
    root: Path,
) -> list[TestEntry]:
    """Extract all test entries from a single Rust source file.

    Reads the file and scans for doc comment blocks that start with a
    test ID line (/// T-XXX-NNN: ...). Accumulates continuation lines
    (subsequent /// lines) until a non-doc-comment line is encountered.
    The accumulated text is then split into description and acceptance
    criterion.

    Args:
        file_path: Absolute path to the .rs file.
        root: Absolute path to the workspace root, used to compute
            relative paths.

    Returns:
        A list of TestEntry instances found in the file. May be empty
        if the file contains no test ID doc comments.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    rel_path = str(file_path.relative_to(root)).replace("\\", "/")
    category = _classify_category(rel_path)
    crate_name = _derive_crate_name(rel_path)

    integration_group = ""
    if category == "integration":
        integration_group = _derive_integration_group(rel_path)

    entries: list[TestEntry] = []
    lines = content.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        id_match = TEST_ID_LINE_PATTERN.match(line.strip())

        if not id_match:
            i += 1
            continue

        test_id = id_match.group(1)
        first_line_text = id_match.group(2).strip()

        # Accumulate continuation lines from the same doc comment block.
        # A continuation line starts with "///" and does NOT match a
        # test ID pattern (which would indicate the start of a different test).
        continuation_parts: list[str] = [first_line_text]
        j = i + 1
        while j < len(lines):
            cont_line = lines[j].strip()
            # Stop if this line is not a doc comment continuation
            cont_match = DOC_CONTINUATION_PATTERN.match(cont_line)
            if not cont_match:
                break
            # Stop if this continuation line starts a different test ID
            if TEST_ID_LINE_PATTERN.match(cont_line):
                break
            continuation_parts.append(cont_match.group(1).strip())
            j += 1

        full_text = " ".join(part for part in continuation_parts if part)
        description, acceptance = _split_description_and_acceptance(full_text)

        entries.append(TestEntry(
            test_id=test_id,
            description=description,
            acceptance=acceptance,
            source_file=rel_path,
            crate_name=crate_name,
            category=category,
            integration_group=integration_group,
        ))

        i = j  # Skip past the accumulated continuation lines

    return entries


# ---------------------------------------------------------------------------
# Workspace scanning
# ---------------------------------------------------------------------------

def scan_workspace(root: Path) -> GenerationResult:
    """Scan all .rs files under crates/ and collect test entries.

    Iterates over every .rs file in the crates/ directory tree, extracts
    test entries from doc comments, and deduplicates by test ID. If a
    test ID appears in multiple files, the first occurrence (by
    alphabetical file path order) is kept and subsequent occurrences are
    recorded as duplicates with a warning.

    Args:
        root: Absolute path to the Cargo workspace root.

    Returns:
        A GenerationResult containing all unique entries, duplicate
        records, file scan count, and warning messages.
    """
    result = GenerationResult()
    crates_dir = root / "crates"

    if not crates_dir.is_dir():
        return result

    # Collect all .rs files sorted by path for deterministic ordering
    rs_files = sorted(crates_dir.rglob("*.rs"))
    result.files_scanned = len(rs_files)

    seen_ids: dict[str, str] = {}  # test_id -> first source_file

    for rs_file in rs_files:
        file_entries = parse_test_entries_from_file(rs_file, root)

        for entry in file_entries:
            if entry.test_id in seen_ids:
                # Duplicate test ID: record warning, skip this occurrence
                result.duplicates.append((entry.test_id, entry.source_file))
                result.warnings.append(
                    f"duplicate test ID {entry.test_id} in "
                    f"{entry.source_file} (first seen in "
                    f"{seen_ids[entry.test_id]})",
                )
                continue

            seen_ids[entry.test_id] = entry.source_file
            result.entries.append(entry)

    return result


# ---------------------------------------------------------------------------
# LaTeX generation: longtable catalog helpers
# ---------------------------------------------------------------------------

def _longtable_header() -> str:
    r"""Produce the longtable environment preamble with column specifications.

    The table uses three columns:
        - Column 1: test ID in ttfamily footnotesize, 4.0cm wide
          (fits the longest ID: T-ANNOTATE-V3-REPORT-NNN at 24 chars;
          24 * 0.163cm/char = 3.91cm, rounded up)
        - Column 2: description, 3.8cm ragged right
        - Column 3: acceptance criterion, 7.2cm ragged right

    Total column width: 4.0 + 3.8 + 7.2 = 15.0cm, fitting within
    the 16.0cm text width of the A4 page with 2.5cm margins.

    Returns:
        A multi-line string containing the \begin{longtable} and header rows.
    """
    return (
        "\\begin{longtable}{>{\\ttfamily\\footnotesize}p{4.0cm}\n"
        "                   >{\\raggedright\\arraybackslash}p{3.8cm}\n"
        "                   >{\\raggedright\\arraybackslash}p{7.2cm}}\n"
        "    \\toprule\n"
        "    \\normalfont\\textbf{ID} & \\textbf{Description} & "
        "\\textbf{Acceptance Criterion} \\\\\n"
        "    \\midrule\n"
        "    \\endfirsthead\n"
        "    \\toprule\n"
        "    \\normalfont\\textbf{ID} & \\textbf{Description} & "
        "\\textbf{Acceptance Criterion} \\\\\n"
        "    \\midrule\n"
        "    \\endhead\n"
        "    \\bottomrule\n"
        "    \\endfoot\n"
    )


def _format_entry_row(entry: TestEntry) -> str:
    r"""Format a single test entry as a LaTeX longtable row.

    The test ID appears unescaped in the ttfamily column (underscores and
    hyphens are valid in that context because the column uses ttfamily).
    The description and acceptance columns are LaTeX-escaped.

    Args:
        entry: The TestEntry to format.

    Returns:
        A string containing the formatted row, ending with \\.
    """
    desc = escape_latex(entry.description)
    accept = escape_latex(entry.acceptance)

    # The test ID column is in \ttfamily mode, but hyphens are fine.
    # Only need to handle characters that are problematic in LaTeX.
    tid = entry.test_id

    if accept:
        return f"    {tid} & {desc}\n        & {accept} \\\\\n"
    else:
        return f"    {tid} & {desc}\n        & --- \\\\\n"


def _multicolumn_crate_header(crate_name: str) -> str:
    r"""Produce a \multicolumn row that labels a crate section in the longtable.

    The header spans all three columns and shows the crate name with its
    description from CRATE_DESCRIPTIONS.

    Args:
        crate_name: The crate name (e.g., "neuroncite-core").

    Returns:
        A LaTeX string with the multicolumn header and a following midrule.
    """
    desc = CRATE_DESCRIPTIONS.get(crate_name, "")
    if desc:
        return (
            f"\n    \\multicolumn{{3}}{{l}}"
            f"{{\\textit{{Crate: \\texttt{{{crate_name}}} -- {desc}}}}} \\\\\n"
            f"    \\midrule\n"
        )
    return (
        f"\n    \\multicolumn{{3}}{{l}}"
        f"{{\\textit{{Crate: \\texttt{{{crate_name}}}}}}} \\\\\n"
        f"    \\midrule\n"
    )


def _multicolumn_group_header(group_label: str) -> str:
    r"""Produce a \multicolumn row for a sub-category group header.

    Used in the integration test catalog to separate sub-categories
    (e.g., "Full pipeline integration tests", "CLI integration tests").

    Args:
        group_label: The human-readable group label.

    Returns:
        A LaTeX string with the multicolumn header and a following midrule.
    """
    return (
        f"\n    \\multicolumn{{3}}{{l}}"
        f"{{\\textit{{{group_label}}}}} \\\\\n"
        f"    \\midrule\n"
    )


# ---------------------------------------------------------------------------
# LaTeX generation: catalog sections
# ---------------------------------------------------------------------------

def generate_unit_catalog(entries: list[TestEntry]) -> tuple[str, dict[str, int]]:
    """Generate the unit test catalog longtable.

    Groups unit test entries by crate name (alphabetical) with multicolumn
    crate headers. Entries within each crate are sorted by test ID numeric
    order.

    Args:
        entries: All test entries with category "unit".

    Returns:
        A tuple of (latex_string, crate_counts) where crate_counts maps
        each crate name to the number of tests in that crate.
    """
    if not entries:
        return "", {}

    # Group by crate name
    by_crate: dict[str, list[TestEntry]] = {}
    for entry in entries:
        if entry.crate_name not in by_crate:
            by_crate[entry.crate_name] = []
        by_crate[entry.crate_name].append(entry)

    crate_counts: dict[str, int] = {}
    total_count = 0

    lines: list[str] = []
    lines.append("\\subsection{Unit Test Catalog}\n")
    lines.append("\\label{sec:unit-tests}\n\n")
    lines.append(
        "The following tables define all required unit tests organized by crate. Each test\n"
        "has a unique identifier, a description of the behavior under test, and the acceptance\n"
        "criterion that must hold for the test to pass.\n\n"
    )
    lines.append(_longtable_header())

    for crate_name in sorted(by_crate.keys()):
        crate_entries = sorted(by_crate[crate_name], key=_sort_key)
        crate_counts[crate_name] = len(crate_entries)
        total_count += len(crate_entries)

        lines.append(_multicolumn_crate_header(crate_name))

        for entry in crate_entries:
            lines.append(_format_entry_row(entry))

    lines.append("\n    \\bottomrule\n")
    lines.append(
        f"    \\caption{{Unit test catalog: {total_count} tests "
        f"across all library crates.}}\n"
    )
    lines.append("\\end{longtable}\n")

    return "".join(lines), crate_counts


def generate_integration_catalog(
    entries: list[TestEntry],
) -> tuple[str, dict[str, int]]:
    """Generate the integration test catalog longtable.

    Groups integration tests by their sub-category (pipeline, CLI, server)
    with multicolumn group headers. Entries within each group are sorted
    by test ID numeric order.

    Args:
        entries: All test entries with category "integration".

    Returns:
        A tuple of (latex_string, group_counts) where group_counts maps
        each sub-category label to its test count.
    """
    if not entries:
        return "", {}

    # Group by integration sub-category
    by_group: dict[str, list[TestEntry]] = {}
    for entry in entries:
        group = entry.integration_group
        if group not in by_group:
            by_group[group] = []
        by_group[group].append(entry)

    group_counts: dict[str, int] = {}
    total_count = 0

    lines: list[str] = []
    lines.append("\\subsection{Integration Test Catalog}\n")
    lines.append("\\label{sec:integration-tests}\n\n")
    lines.append(
        "The following table lists all integration tests that verify cross-crate\n"
        "interactions, end-to-end workflows, and external system integration.\n\n"
    )
    lines.append(_longtable_header())

    # Use the defined ordering; any groups not in the ordering list are appended
    # alphabetically at the end.
    ordered_groups = [
        g for g in INTEGRATION_GROUP_ORDER if g in by_group
    ]
    remaining_groups = sorted(
        g for g in by_group if g not in INTEGRATION_GROUP_ORDER
    )
    ordered_groups.extend(remaining_groups)

    for group_label in ordered_groups:
        group_entries = sorted(by_group[group_label], key=_sort_key)
        group_counts[group_label] = len(group_entries)
        total_count += len(group_entries)

        lines.append(_multicolumn_group_header(group_label))

        for entry in group_entries:
            lines.append(_format_entry_row(entry))

    lines.append("\n    \\bottomrule\n")
    lines.append(
        f"    \\caption{{Integration test catalog: {total_count} tests "
        f"covering cross-crate interactions.}}\n"
    )
    lines.append("\\end{longtable}\n")

    return "".join(lines), group_counts


def generate_property_regression_catalog(
    property_entries: list[TestEntry],
    regression_entries: list[TestEntry],
) -> tuple[str, int, int]:
    """Generate the combined property-based and regression test catalog.

    Property tests and regression tests share a single longtable with
    separate multicolumn group headers. Each group is sorted by test ID.

    Args:
        property_entries: All test entries with category "property".
        regression_entries: All test entries with category "regression".

    Returns:
        A tuple of (latex_string, property_count, regression_count).
    """
    if not property_entries and not regression_entries:
        return "", 0, 0

    prop_sorted = sorted(property_entries, key=_sort_key)
    reg_sorted = sorted(regression_entries, key=_sort_key)
    total = len(prop_sorted) + len(reg_sorted)

    lines: list[str] = []
    lines.append("\\subsection{Property-Based and Regression Test Catalog}\n")
    lines.append("\\label{sec:property-regression-tests}\n\n")
    lines.append(
        "The following table lists property-based tests (verified via randomized\n"
        "input generation) and regression tests (covering previously identified\n"
        "edge cases and bug fixes).\n\n"
    )
    lines.append(_longtable_header())

    if prop_sorted:
        lines.append(_multicolumn_group_header("Property-based tests"))
        for entry in prop_sorted:
            lines.append(_format_entry_row(entry))

    if reg_sorted:
        lines.append(_multicolumn_group_header("Regression and edge case tests"))
        for entry in reg_sorted:
            lines.append(_format_entry_row(entry))

    lines.append("\n    \\bottomrule\n")
    lines.append(
        f"    \\caption{{Property-based and regression test catalog: "
        f"{total} tests covering invariants and edge cases.}}\n"
    )
    lines.append("\\end{longtable}\n")

    return "".join(lines), len(prop_sorted), len(reg_sorted)


def generate_benchmark_catalog(entries: list[TestEntry]) -> tuple[str, int]:
    """Generate the performance benchmark catalog longtable.

    Benchmark entries are sorted by test ID without sub-grouping.

    Args:
        entries: All test entries with category "benchmark".

    Returns:
        A tuple of (latex_string, benchmark_count).
    """
    if not entries:
        return "", 0

    sorted_entries = sorted(entries, key=_sort_key)

    lines: list[str] = []
    lines.append("\\subsection{Performance Benchmark Catalog}\n")
    lines.append("\\label{sec:performance-benchmarks}\n\n")
    lines.append(
        "The following table lists all performance benchmarks that measure\n"
        "throughput, latency, and resource usage of critical code paths.\n\n"
    )
    lines.append(_longtable_header())

    for entry in sorted_entries:
        lines.append(_format_entry_row(entry))

    lines.append("\n    \\bottomrule\n")
    lines.append(
        f"    \\caption{{Performance benchmark catalog: "
        f"{len(sorted_entries)} benchmarks.}}\n"
    )
    lines.append("\\end{longtable}\n")

    return "".join(lines), len(sorted_entries)


# ---------------------------------------------------------------------------
# LaTeX generation: summary table
# ---------------------------------------------------------------------------

def generate_summary_table(
    unit_crate_counts: dict[str, int],
    integration_group_counts: dict[str, int],
    property_count: int,
    regression_count: int,
    benchmark_count: int,
) -> str:
    """Generate the test summary table in tabularx format.

    The summary table contains per-crate unit test rows, per-group
    integration test rows, and rows for property, regression, and
    benchmark categories. Subtotals are shown for unit, integration,
    and auxiliary (property + regression + benchmark) groups. A grand
    total row appears at the bottom.

    Args:
        unit_crate_counts: Dict mapping crate name to unit test count.
        integration_group_counts: Dict mapping integration sub-category
            label to test count.
        property_count: Number of property-based tests.
        regression_count: Number of regression tests.
        benchmark_count: Number of benchmark tests.

    Returns:
        A LaTeX string containing the complete summary subsection and table.
    """
    lines: list[str] = []
    lines.append("\\subsection{Test Summary}\n\n")

    lines.append("\\begin{table}[htbp]\n")
    lines.append("\\centering\n")
    lines.append("\\begin{tabularx}{\\textwidth}{Xlr}\n")
    lines.append("    \\toprule\n")
    lines.append(
        "    \\textbf{Category} & \\textbf{Count} & \\textbf{Type} \\\\\n"
    )
    lines.append("    \\midrule\n")

    # Unit test rows (sorted by crate name)
    unit_subtotal = 0
    for crate_name in sorted(unit_crate_counts.keys()):
        count = unit_crate_counts[crate_name]
        unit_subtotal += count
        lines.append(
            f"    {crate_name} unit tests"
            f"          & {count} & Unit \\\\\n"
        )

    lines.append("    \\midrule\n")
    lines.append(
        f"    \\textit{{Unit test subtotal}}"
        f"         & \\textit{{{unit_subtotal}}} & \\\\\n"
    )
    lines.append("    \\midrule\n")

    # Integration test rows (in defined order)
    integration_subtotal = 0
    ordered_groups = [
        g for g in INTEGRATION_GROUP_ORDER if g in integration_group_counts
    ]
    remaining = sorted(
        g for g in integration_group_counts if g not in INTEGRATION_GROUP_ORDER
    )
    ordered_groups.extend(remaining)

    for group_label in ordered_groups:
        count = integration_group_counts[group_label]
        integration_subtotal += count
        lines.append(
            f"    {group_label}"
            f"     & {count} & Integration \\\\\n"
        )

    lines.append("    \\midrule\n")
    lines.append(
        f"    \\textit{{Integration test subtotal}}"
        f"  & \\textit{{{integration_subtotal}}} & \\\\\n"
    )
    lines.append("    \\midrule\n")

    # Auxiliary rows: property, regression, benchmark
    auxiliary_subtotal = property_count + regression_count + benchmark_count

    if property_count > 0:
        lines.append(
            f"    Property-based tests"
            f"                & {property_count} & Property \\\\\n"
        )
    if regression_count > 0:
        lines.append(
            f"    Regression and edge case tests"
            f"      & {regression_count} & Regression \\\\\n"
        )
    if benchmark_count > 0:
        lines.append(
            f"    Performance benchmarks"
            f"              & {benchmark_count} & Benchmark \\\\\n"
        )

    lines.append("    \\midrule\n")
    lines.append(
        f"    \\textit{{Auxiliary test subtotal}}"
        f"    & \\textit{{{auxiliary_subtotal}}} & \\\\\n"
    )
    lines.append("    \\midrule\n")
    lines.append("    \\midrule\n")

    grand_total = unit_subtotal + integration_subtotal + auxiliary_subtotal
    lines.append(
        f"    \\textbf{{Total}}"
        f"                      & \\textbf{{{grand_total}}} & \\\\\n"
    )

    lines.append("    \\bottomrule\n")
    lines.append("\\end{tabularx}\n")
    lines.append(
        f"\\caption{{Test catalog summary: {grand_total} tests "
        f"across all categories.}}\n"
    )
    lines.append("\\end{table}\n")

    return "".join(lines)


# ---------------------------------------------------------------------------
# Full document assembly
# ---------------------------------------------------------------------------

def generate_latex_document(result: GenerationResult) -> str:
    """Assemble the complete LaTeX output from all catalog sections.

    Generates a preamble comment, four catalog subsections (unit,
    integration, property/regression, benchmark), and a summary table.
    Empty catalogs are omitted from the output.

    Args:
        result: The GenerationResult from scan_workspace().

    Returns:
        The complete LaTeX document content as a string.
    """
    total_ids = len(result.entries)

    parts: list[str] = []

    # Preamble comment
    parts.append(
        "% Auto-generated by tools/gen/generate_test_chapter.py.\n"
        "% DO NOT EDIT -- regenerate with: python tools/gen/generate_test_chapter.py\n"
        f"% Source: {result.files_scanned} Rust source files scanned, "
        f"{total_ids} test IDs extracted.\n\n"
    )

    # Partition entries by category
    unit_entries = [e for e in result.entries if e.category == "unit"]
    integration_entries = [e for e in result.entries if e.category == "integration"]
    property_entries = [e for e in result.entries if e.category == "property"]
    regression_entries = [e for e in result.entries if e.category == "regression"]
    benchmark_entries = [e for e in result.entries if e.category == "benchmark"]

    # Generate each catalog section
    unit_latex, unit_crate_counts = generate_unit_catalog(unit_entries)
    if unit_latex:
        parts.append(unit_latex)
        parts.append("\n")

    integration_latex, integration_group_counts = generate_integration_catalog(
        integration_entries,
    )
    if integration_latex:
        parts.append(integration_latex)
        parts.append("\n")

    prop_reg_latex, prop_count, reg_count = generate_property_regression_catalog(
        property_entries,
        regression_entries,
    )
    if prop_reg_latex:
        parts.append(prop_reg_latex)
        parts.append("\n")

    bench_latex, bench_count = generate_benchmark_catalog(benchmark_entries)
    if bench_latex:
        parts.append(bench_latex)
        parts.append("\n")

    # Generate summary table
    summary_latex = generate_summary_table(
        unit_crate_counts,
        integration_group_counts,
        prop_count,
        reg_count,
        bench_count,
    )
    parts.append(summary_latex)

    return "".join(parts)


# ---------------------------------------------------------------------------
# JSON statistics output
# ---------------------------------------------------------------------------

def generate_json_statistics(result: GenerationResult) -> str:
    """Produce a JSON representation of extraction statistics.

    The JSON object contains counts broken down by category, per-crate
    unit test counts, per-group integration test counts, and a list of
    any duplicate warnings.

    Args:
        result: The GenerationResult from scan_workspace().

    Returns:
        A JSON string with 2-space indentation.
    """
    unit_entries = [e for e in result.entries if e.category == "unit"]
    integration_entries = [e for e in result.entries if e.category == "integration"]
    property_entries = [e for e in result.entries if e.category == "property"]
    regression_entries = [e for e in result.entries if e.category == "regression"]
    benchmark_entries = [e for e in result.entries if e.category == "benchmark"]

    # Per-crate unit test counts
    unit_by_crate: dict[str, int] = {}
    for entry in unit_entries:
        if entry.crate_name not in unit_by_crate:
            unit_by_crate[entry.crate_name] = 0
        unit_by_crate[entry.crate_name] += 1

    # Per-group integration test counts
    integration_by_group: dict[str, int] = {}
    for entry in integration_entries:
        group = entry.integration_group
        if group not in integration_by_group:
            integration_by_group[group] = 0
        integration_by_group[group] += 1

    stats = {
        "files_scanned": result.files_scanned,
        "total_test_ids": len(result.entries),
        "duplicates_skipped": len(result.duplicates),
        "categories": {
            "unit": len(unit_entries),
            "integration": len(integration_entries),
            "property": len(property_entries),
            "regression": len(regression_entries),
            "benchmark": len(benchmark_entries),
        },
        "unit_by_crate": dict(sorted(unit_by_crate.items())),
        "integration_by_group": dict(sorted(integration_by_group.items())),
        "duplicates": [
            {"test_id": tid, "file": fpath}
            for tid, fpath in result.duplicates
        ],
        "warnings": result.warnings,
    }

    return json.dumps(stats, indent=2)


# ---------------------------------------------------------------------------
# Stderr summary output
# ---------------------------------------------------------------------------

def print_summary(
    result: GenerationResult,
    output_path: Path,
) -> None:
    """Print a human-readable summary of the generation to stderr.

    Shows the total file and test ID counts, per-category breakdowns,
    and the output file path.

    Args:
        result: The GenerationResult from scan_workspace().
        output_path: The path where the LaTeX file was written.
    """
    unit_count = sum(1 for e in result.entries if e.category == "unit")
    integration_count = sum(1 for e in result.entries if e.category == "integration")
    prop_reg_count = sum(
        1 for e in result.entries if e.category in ("property", "regression")
    )
    bench_count = sum(1 for e in result.entries if e.category == "benchmark")

    # Count distinct crates among unit tests
    unit_crates = {e.crate_name for e in result.entries if e.category == "unit"}

    print(
        f"generate_test_chapter: scanned {result.files_scanned} files, "
        f"extracted {len(result.entries)} test IDs "
        f"({len(result.duplicates)} duplicates skipped)",
        file=sys.stderr,
    )
    print(
        f"  Unit tests: {unit_count} across {len(unit_crates)} crates",
        file=sys.stderr,
    )
    print(f"  Integration tests: {integration_count}", file=sys.stderr)
    print(f"  Property/regression tests: {prop_reg_count}", file=sys.stderr)
    print(f"  Benchmarks: {bench_count}", file=sys.stderr)
    print(f"  Output: {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse command-line arguments, scan the workspace, and produce output.

    Returns:
        Exit code:
            0 if generation completed without warnings.
            1 if generation completed with warnings (duplicate test IDs).
            2 if the script cannot run (missing crates/ directory, I/O error).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Test Chapter Generator for NeuronCite. "
            "Scans Rust source files for test ID doc comments and generates "
            "a LaTeX file with longtable catalogs and a summary table."
        ),
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to the Cargo workspace root directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the generated LaTeX file. "
            "Defaults to {root}/docs/tests_generated.tex"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print extraction statistics as JSON to stdout instead of "
            "generating a LaTeX file"
        ),
    )
    args = parser.parse_args()

    # Resolve and validate paths
    root_path: Path = args.root.resolve()

    if not root_path.is_dir():
        print(
            f"Error: workspace root directory not found: {root_path}",
            file=sys.stderr,
        )
        return 2

    crates_dir = root_path / "crates"
    if not crates_dir.is_dir():
        print(
            f"Error: crates directory not found: {crates_dir}",
            file=sys.stderr,
        )
        return 2

    # Determine output path
    if args.output is not None:
        output_path: Path = args.output.resolve()
    else:
        output_path = root_path / "docs" / "tests_generated.tex"

    # Scan workspace for test entries
    try:
        result = scan_workspace(root_path)
    except Exception as exc:
        print(f"Error: workspace scan failed: {exc}", file=sys.stderr)
        return 2

    if not result.entries:
        print(
            "Error: no test IDs found in any .rs files under crates/",
            file=sys.stderr,
        )
        return 2

    # JSON mode: print statistics and exit
    if args.json:
        print(generate_json_statistics(result))
        # Print warnings to stderr
        for warning in result.warnings:
            print(f"[WARN] {warning}", file=sys.stderr)
        return 1 if result.warnings else 0

    # Validate test ID lengths against the longtable column width.
    # In \ttfamily\footnotesize at 10pt base, each character occupies
    # approximately 0.163cm. The ID column is 4.0cm wide, which fits
    # 4.0 / 0.163 = ~24 monospace characters. IDs exceeding this limit
    # cause overfull \hbox warnings and visible text overlap in the PDF.
    max_id_chars = 24
    overflow_ids = [
        e.test_id for e in result.entries if len(e.test_id) > max_id_chars
    ]
    if overflow_ids:
        result.warnings.append(
            f"test_id_column_overflow: {len(overflow_ids)} test ID(s) exceed "
            f"{max_id_chars} characters and will overflow the 3.8cm ID column: "
            + ", ".join(sorted(set(overflow_ids))[:5])
        )

    # LaTeX mode: generate and write the document
    latex_content = generate_latex_document(result)

    try:
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_content, encoding="utf-8")
    except OSError as exc:
        print(
            f"Error: failed to write output file {output_path}: {exc}",
            file=sys.stderr,
        )
        return 2

    # Print summary and any warnings to stderr
    print_summary(result, output_path)
    for warning in result.warnings:
        print(f"[WARN] {warning}", file=sys.stderr)

    return 1 if result.warnings else 0


if __name__ == "__main__":
    sys.exit(main())
