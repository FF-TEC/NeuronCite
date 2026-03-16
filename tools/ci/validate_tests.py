#!/usr/bin/env python3
"""Test Catalog Validator for NeuronCite.

Validates bidirectional consistency between the LaTeX architecture document's
test catalogs (unit, integration, property, regression, performance) and the
actual Rust test code. Detects missing tests, count mismatches, numbering
inconsistencies, undocumented test categories, and arithmetic errors in the
summary table.

This is the fourth validator in the NeuronCite documentation verification
suite, alongside validate_architecture.py (module structure, feature flags,
dependencies), validate_schemas.py (database schema tables, columns, indexes,
triggers), and validate_consistency.py (MCP tools, API endpoints, CLI
subcommands, crate counts).

Validation checks (10 categories):

  1.  test_id_duplicates       (error)   No duplicate test IDs in LaTeX or code
  2.  test_id_sequential       (warning) IDs within each prefix are sequential
  3.  catalog_caption_count    (error)   Catalog entry count matches caption claim
  4.  summary_arithmetic       (error)   Summary subtotals and total are correct
  5.  summary_vs_catalog       (error)   Summary counts match catalog entry counts
  6.  latex_tests_in_code      (error)   Every LaTeX test ID exists in code
  7.  code_tests_in_latex      (error)   Every code test ID exists in LaTeX
  8.  undocumented_prefixes    (error)   Every code prefix has a LaTeX catalog
  9.  unit_crate_counts        (error)   Per-crate summary counts match catalog
  10. summary_caption_match    (error)   Summary body total matches caption

Exit codes:
  0  All checks passed (no errors; warnings allowed unless --strict).
  1  At least one error found, or a warning with --strict enabled.
  2  Script cannot run (missing files, parse failure).

Usage:
  python tools/validate_tests.py \\
      --tex docs/architecture.tex --root . [--strict] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
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
# Word-to-number mapping for prose and caption claims.
# Covers the range relevant to NeuronCite's test counts (1-999).
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
# Test ID regex pattern
# ---------------------------------------------------------------------------
# Matches test IDs like:
#   T-CORE-001, T-CORE-024a, T-MCP-BATCH-001, T-MCP-IDX-006,
#   T-EMB-POOL-001, T-PATH-014, T-REF-028, T-PERF-006, T-E2E-001
# Structure: T-PREFIX[-SUBPREFIX]-NNN[letter]
# Prefix segments start with an uppercase letter and may contain digits
# (e.g., E2E). The numeric part is always exactly 3 digits.
TEST_ID_PATTERN = re.compile(r"T-[A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*-\d{3}[a-z]?")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Diagnostic:
    """A single validation diagnostic (error or warning).

    Attributes:
        level: Either "error" or "warning".
        category: The check category identifier (e.g., "test_id_duplicates",
            "catalog_caption_count").
        section: The section or entity group this diagnostic applies to
            (e.g., "unit_tests", "integration_tests", "summary_table").
        detail: A human-readable description of the discrepancy.
    """

    level: str
    category: str
    section: str
    detail: str


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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


def _extract_caption_number(tex_content: str, caption_pattern: str) -> int | None:
    """Extract the numeric count from a test catalog caption.

    Captions follow the form: "Unit test catalog: 398 tests across..."
    or "17 tests covering...". This function finds the caption matching
    the given pattern and extracts the first integer found within it.

    Args:
        tex_content: The full LaTeX document content.
        caption_pattern: A regex pattern to match within the caption text.

    Returns:
        The integer count from the caption, or None if not found.
    """
    caption_match = re.search(
        r"\\caption\{(" + caption_pattern + r"[^}]*)\}",
        tex_content,
        re.DOTALL,
    )
    if not caption_match:
        return None

    caption_text = caption_match.group(1)
    # Extract the first integer from the caption text
    num_match = re.search(r"(\d+)\s+(?:tests?|benchmarks?)", caption_text)
    if num_match:
        return int(num_match.group(1))
    return None


def _extract_test_ids_from_body(body: str) -> list[str]:
    """Extract all test IDs from a LaTeX table body.

    Scans the joined rows of a longtable for test ID patterns in the
    first column. The first column uses ttfamily formatting, so IDs
    appear as plain text (e.g., T-CORE-001) without \\texttt{} wrapping.

    Skips \\multicolumn rows (section headers) and \\textbf rows (table
    headers).

    Args:
        body: The text content of a longtable environment.

    Returns:
        A list of test ID strings in document order.
    """
    test_ids: list[str] = []

    rows = _join_multiline_rows(body)
    for row in rows:
        # Skip header rows and section divider rows
        if "\\textbf" in row or "\\normalfont" in row:
            continue
        if "\\multicolumn" in row:
            continue

        # Extract test ID from the first cell
        cells = row.split("&")
        if not cells:
            continue

        first_cell = cells[0].strip().rstrip("\\").strip()
        match = TEST_ID_PATTERN.match(first_cell)
        if match:
            test_ids.append(match.group(0))

    return test_ids


def _base_prefix(test_id: str) -> str:
    """Extract the base prefix from a test ID for grouping purposes.

    Splits the ID after "T-" and before the numeric part. For IDs with
    sub-prefixes (like T-MCP-BATCH-001), the full prefix chain is returned.

    Examples:
        T-CORE-001    -> "CORE"
        T-MCP-IDX-006 -> "MCP-IDX"
        T-CORE-024a   -> "CORE"
        T-PATH-014    -> "PATH"

    Args:
        test_id: A test ID string (e.g., "T-CORE-001").

    Returns:
        The prefix portion without the "T-" leader and numeric suffix.
    """
    # Remove the T- prefix
    rest = test_id[2:]
    # Remove the numeric suffix (and optional letter suffix)
    # Find the last "-NNN" or "-NNNx" segment
    match = re.search(r"-(\d{3}[a-z]?)$", rest)
    if match:
        return rest[:match.start()]
    return rest


def _numeric_part(test_id: str) -> int:
    """Extract the numeric portion of a test ID.

    For T-CORE-001 returns 1, for T-MCP-IDX-006 returns 6.
    Ignores letter suffixes (T-CORE-024a returns 24).

    Args:
        test_id: A test ID string.

    Returns:
        The integer value of the 3-digit numeric part.
    """
    match = re.search(r"(\d{3})[a-z]?$", test_id)
    if match:
        return int(match.group(1))
    return 0


# ---------------------------------------------------------------------------
# LaTeX parsing functions
# ---------------------------------------------------------------------------

def parse_unit_test_catalog(tex_content: str) -> tuple[list[str], int | None]:
    """Extract all test IDs from the unit test catalog table.

    The unit test catalog is a single longtable with caption containing
    "Unit test catalog". Test IDs appear in the ttfamily first column
    as plain text entries like T-CORE-001, T-MCP-IDX-006, etc.

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A tuple (test_ids, caption_count) where test_ids is a list of
        all unit test IDs in document order, and caption_count is the
        integer count claimed in the caption (or None if not parseable).
    """
    body = _find_table_body(tex_content, r"Unit test catalog")
    if not body:
        return [], None

    test_ids = _extract_test_ids_from_body(body)
    caption_count = _extract_caption_number(tex_content, r"Unit test catalog")

    return test_ids, caption_count


def parse_integration_test_catalog(
    tex_content: str,
) -> tuple[list[str], int | None]:
    """Extract all test IDs from the integration test catalog table.

    The integration test catalog is a longtable with caption containing
    "Integration test catalog". Contains tests from three subsections:
    full pipeline (T-INT-xxx), CLI (T-CLI-xxx), and API server (T-SRV-xxx).

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A tuple (test_ids, caption_count) where test_ids is a list of
        all integration test IDs, and caption_count is the caption claim.
    """
    body = _find_table_body(tex_content, r"Integration test catalog")
    if not body:
        return [], None

    test_ids = _extract_test_ids_from_body(body)
    caption_count = _extract_caption_number(
        tex_content, r"Integration test catalog",
    )

    return test_ids, caption_count


def parse_property_regression_catalog(
    tex_content: str,
) -> tuple[list[str], int | None]:
    """Extract all test IDs from the property-based and regression test catalog.

    The combined catalog has caption containing "Property-based and regression
    test catalog". Contains property tests (T-PROP-xxx) and regression tests
    (T-REG-xxx).

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A tuple (test_ids, caption_count) where test_ids is a list of
        all property and regression test IDs, and caption_count is the
        caption claim.
    """
    body = _find_table_body(
        tex_content, r"Property-based and regression test catalog",
    )
    if not body:
        return [], None

    test_ids = _extract_test_ids_from_body(body)
    caption_count = _extract_caption_number(
        tex_content, r"Property-based and regression test catalog",
    )

    return test_ids, caption_count


def parse_performance_catalog(
    tex_content: str,
) -> tuple[list[str], int | None]:
    """Extract all test IDs from the performance benchmark catalog.

    The performance catalog has caption containing "Performance benchmark
    catalog". Contains benchmark tests (T-PERF-xxx).

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A tuple (test_ids, caption_count).
    """
    body = _find_table_body(tex_content, r"Performance benchmark catalog")
    if not body:
        return [], None

    test_ids = _extract_test_ids_from_body(body)
    caption_count = _extract_caption_number(
        tex_content, r"Performance benchmark catalog",
    )

    return test_ids, caption_count


def parse_unit_test_crate_sections(
    tex_content: str,
) -> dict[str, list[str]]:
    """Parse the unit test catalog, grouping test IDs by crate section.

    The unit test catalog longtable contains \\multicolumn divider rows
    that identify the crate for the following tests. The pattern is:
        \\multicolumn{3}{l}{\\textit{Crate: \\texttt{neuroncite-xxx} ...}}

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A dict mapping crate names (e.g., "neuroncite-core") to their
        list of test IDs. The special key "_unknown" collects any IDs
        that appear before the first crate header.
    """
    body = _find_table_body(tex_content, r"Unit test catalog")
    if not body:
        return {}

    crate_sections: dict[str, list[str]] = {}
    current_crate = "_unknown"

    rows = _join_multiline_rows(body)
    for row in rows:
        # Check for crate section header
        if "\\multicolumn" in row and "\\texttt{neuroncite" in row:
            crate_match = re.search(
                r"\\texttt\{(neuroncite-[^}]+)\}", row,
            )
            if crate_match:
                # Clean up the crate name (remove LaTeX escapes)
                crate_name = crate_match.group(1).replace("\\_", "_")
                crate_name = crate_name.replace("\\allowbreak", "")
                current_crate = crate_name.strip()
                if current_crate not in crate_sections:
                    crate_sections[current_crate] = []
            continue

        # Skip structural rows
        if "\\textbf" in row or "\\normalfont" in row:
            continue
        if "\\multicolumn" in row:
            continue

        # Extract test ID
        cells = row.split("&")
        if not cells:
            continue

        first_cell = cells[0].strip().rstrip("\\").strip()
        match = TEST_ID_PATTERN.match(first_cell)
        if match:
            if current_crate not in crate_sections:
                crate_sections[current_crate] = []
            crate_sections[current_crate].append(match.group(0))

    return crate_sections


def parse_summary_table(tex_content: str) -> dict | None:
    """Parse the test summary tabularx table.

    The summary table has caption containing "Test catalog summary" and
    lists category names, counts, and test types. Subtotal rows are
    italicized, and the total row is bolded.

    Args:
        tex_content: The full LaTeX document content.

    Returns:
        A dict with keys:
            "rows": list of (category_name, count, test_type) tuples
            "subtotals": dict mapping subtotal names to counts
            "total": the total count from the table body
            "caption_total": the count claimed in the caption
        Returns None if the table is not found.
    """
    body = _find_table_body(tex_content, r"Test catalog summary")
    if not body:
        return None

    rows_data: list[tuple[str, int, str]] = []
    subtotals: dict[str, int] = {}
    total: int | None = None

    rows = _join_multiline_rows(body)
    for row in rows:
        # Skip header rows
        if "\\textbf{Category}" in row:
            continue

        cells = row.split("&")
        if len(cells) < 2:
            continue

        name_cell = cells[0].strip().rstrip("\\").strip()
        count_cell = cells[1].strip().rstrip("\\").strip()

        # Extract the numeric count (may be wrapped in \textit{} or \textbf{})
        count_str = re.sub(r"\\textit\{([^}]*)\}", r"\1", count_cell)
        count_str = re.sub(r"\\textbf\{([^}]*)\}", r"\1", count_str)
        count_str = count_str.strip()

        if not count_str:
            continue

        try:
            count = int(count_str)
        except ValueError:
            continue

        # Clean the name cell of LaTeX formatting
        clean_name = re.sub(r"\\textit\{([^}]*)\}", r"\1", name_cell)
        clean_name = re.sub(r"\\textbf\{([^}]*)\}", r"\1", clean_name)
        clean_name = clean_name.strip()

        # Determine if this is a subtotal, total, or regular row
        if "\\textbf" in name_cell and "Total" in clean_name:
            total = count
        elif "\\textit" in name_cell and "subtotal" in clean_name.lower():
            # Map subtotal key from the name
            if "Unit" in clean_name:
                subtotals["unit"] = count
            elif "Integration" in clean_name:
                subtotals["integration"] = count
            elif "Auxiliary" in clean_name or "auxiliary" in clean_name:
                subtotals["auxiliary"] = count
        else:
            # Regular data row: extract test type from third cell if present
            test_type = ""
            if len(cells) >= 3:
                test_type = cells[2].strip().rstrip("\\").strip()

            rows_data.append((clean_name, count, test_type))

    # Extract caption total
    caption_total = _extract_caption_number(
        tex_content, r"Test catalog summary",
    )

    return {
        "rows": rows_data,
        "subtotals": subtotals,
        "total": total,
        "caption_total": caption_total,
    }


# ---------------------------------------------------------------------------
# Code parsing functions
# ---------------------------------------------------------------------------

def parse_code_test_ids(root: Path) -> dict[str, list[str]]:
    """Extract all test IDs from Rust test code across the workspace.

    Scans all .rs files under crates/ for doc comments containing test
    ID patterns (/// T-XXX-NNN:). Groups the found IDs by their base
    prefix (e.g., "CORE", "CLI", "MCP-BATCH").

    Also tracks which file each ID was found in (for diagnostic messages).

    Args:
        root: The Cargo workspace root directory.

    Returns:
        A dict mapping base prefixes to their list of test IDs, e.g.,
        {"CORE": ["T-CORE-001", ...], "CLI": ["T-CLI-001", ...], ...}.
    """
    crates_dir = root / "crates"
    ids_by_prefix: dict[str, list[str]] = {}

    for rs_file in crates_dir.rglob("*.rs"):
        try:
            content = rs_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # Find all test ID doc comments: /// T-XXX-NNN:
        for match in re.finditer(r"///\s+(T-[A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*-\d{3}[a-z]?):", content):
            test_id = match.group(1)
            prefix = _base_prefix(test_id)
            if prefix not in ids_by_prefix:
                ids_by_prefix[prefix] = []
            ids_by_prefix[prefix].append(test_id)

    return ids_by_prefix


def map_code_test_files(root: Path) -> dict[str, list[str]]:
    """Map each test ID to the source file(s) where it appears in code.

    Scans all .rs files for test ID doc comments and records the relative
    file path for each ID. Used to provide helpful file locations in
    diagnostic messages.

    Args:
        root: The Cargo workspace root directory.

    Returns:
        A dict mapping test ID strings to lists of relative file paths
        where that ID appears. Multiple files indicate a duplicate.
    """
    crates_dir = root / "crates"
    id_to_files: dict[str, list[str]] = {}

    for rs_file in crates_dir.rglob("*.rs"):
        try:
            content = rs_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        rel_path = str(rs_file.relative_to(root)).replace("\\", "/")

        for match in re.finditer(r"///\s+(T-[A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*-\d{3}[a-z]?):", content):
            test_id = match.group(1)
            if test_id not in id_to_files:
                id_to_files[test_id] = []
            id_to_files[test_id].append(rel_path)

    return id_to_files


# ---------------------------------------------------------------------------
# Validation check functions
# ---------------------------------------------------------------------------

def check_test_id_duplicates(
    latex_ids: list[str],
    code_id_files: dict[str, list[str]],
) -> list[Diagnostic]:
    """Check 1: Detect duplicate test IDs within LaTeX catalogs or within code.

    A test ID appearing more than once in the LaTeX catalog tables indicates
    a copy-paste error. A test ID appearing in multiple code files indicates
    an ID collision that creates ambiguity about which test is which.

    Args:
        latex_ids: All test IDs extracted from all LaTeX catalog tables.
        code_id_files: Mapping from test ID to list of files containing it.

    Returns:
        A list of Diagnostic instances for any duplicates found.
    """
    diagnostics: list[Diagnostic] = []

    # Check for duplicates in LaTeX
    latex_counts = Counter(latex_ids)
    for test_id, count in sorted(latex_counts.items()):
        if count > 1:
            diagnostics.append(Diagnostic(
                level="error",
                category="test_id_duplicates",
                section="latex_catalogs",
                detail=(
                    f"test ID {test_id} appears {count} times in the "
                    f"LaTeX catalog tables"
                ),
            ))

    # Check for duplicates in code (same ID in multiple files)
    for test_id, files in sorted(code_id_files.items()):
        if len(files) > 1:
            file_list = ", ".join(files)
            diagnostics.append(Diagnostic(
                level="error",
                category="test_id_duplicates",
                section="code_tests",
                detail=(
                    f"test ID {test_id} appears in {len(files)} files: "
                    f"{file_list}"
                ),
            ))

    return diagnostics


def check_test_id_sequential(
    ids_by_prefix: dict[str, list[str]],
    source: str,
) -> list[Diagnostic]:
    """Check 2: Verify that test IDs within each prefix are sequential.

    For each prefix group, collects the numeric parts of all IDs (ignoring
    letter suffixes like 024a, 024b) and checks for gaps in the sequence.
    Gaps indicate deleted or forgotten tests.

    Args:
        ids_by_prefix: Dict mapping prefix strings to their test ID lists.
        source: Either "latex" or "code" (for diagnostic messages).

    Returns:
        A list of Diagnostic instances (warnings) for any gaps found.
    """
    diagnostics: list[Diagnostic] = []

    for prefix, ids in sorted(ids_by_prefix.items()):
        # Collect unique numeric parts (ignore letter suffixes)
        numbers = sorted(set(_numeric_part(tid) for tid in ids))
        if not numbers:
            continue

        # Find gaps in the sequence
        gaps: list[int] = []
        for i in range(len(numbers) - 1):
            expected_next = numbers[i] + 1
            actual_next = numbers[i + 1]
            if actual_next > expected_next:
                # Report all missing numbers in the gap
                for missing in range(expected_next, actual_next):
                    gaps.append(missing)

        if gaps:
            gap_str = ", ".join(str(n) for n in gaps[:10])
            if len(gaps) > 10:
                gap_str += f", ... ({len(gaps)} total)"
            # Numbering gaps are informational, not actionable errors.
            # Gaps arise naturally when test IDs are renumbered to resolve
            # duplicates or when tests are removed. They do not indicate
            # a consistency problem between code and documentation.
            diagnostics.append(Diagnostic(
                level="info",
                category="test_id_sequential",
                section=f"{source}_tests",
                detail=(
                    f"T-{prefix} has gaps in numbering: missing "
                    f"{gap_str} (range {numbers[0]}-{numbers[-1]})"
                ),
            ))

    return diagnostics


def check_catalog_caption_counts(
    catalogs: list[tuple[str, list[str], int | None]],
) -> list[Diagnostic]:
    """Check 3: Verify that each catalog table's entry count matches its caption.

    Each test catalog longtable has a \\caption that claims a specific number
    of tests (e.g., "398 tests across all library crates"). This check
    compares the actual number of test ID entries in the table body against
    the claimed count.

    Args:
        catalogs: A list of (catalog_name, test_ids, caption_count) tuples
            where catalog_name is a label for the catalog, test_ids is the
            list of IDs parsed from the table body, and caption_count is the
            integer from the caption (or None if not parseable).

    Returns:
        A list of Diagnostic instances for any mismatches.
    """
    diagnostics: list[Diagnostic] = []

    for name, ids, caption_count in catalogs:
        if caption_count is None:
            continue

        actual_count = len(ids)
        if actual_count != caption_count:
            diagnostics.append(Diagnostic(
                level="error",
                category="catalog_caption_count",
                section=name,
                detail=(
                    f"caption claims {caption_count} tests but the table "
                    f"body contains {actual_count} entries"
                ),
            ))

    return diagnostics


def check_summary_arithmetic(summary: dict) -> list[Diagnostic]:
    """Check 4: Verify that the summary table's arithmetic is internally correct.

    Checks that:
      - The sum of unit test rows equals the unit subtotal
      - The sum of integration test rows equals the integration subtotal
      - The sum of auxiliary test rows equals the auxiliary subtotal
      - The sum of all subtotals equals the total

    Args:
        summary: The parsed summary table dict from parse_summary_table().

    Returns:
        A list of Diagnostic instances for any arithmetic errors.
    """
    diagnostics: list[Diagnostic] = []

    rows = summary["rows"]
    subtotals = summary["subtotals"]
    total = summary["total"]

    # Group rows by their test type
    unit_rows = [(n, c) for n, c, t in rows if t == "Unit"]
    integration_rows = [(n, c) for n, c, t in rows if t == "Integration"]
    # Auxiliary rows have various types: Property, Regression, Benchmark
    auxiliary_rows = [
        (n, c) for n, c, t in rows
        if t in ("Property", "Regression", "Benchmark")
    ]

    # Check unit subtotal
    if "unit" in subtotals and unit_rows:
        row_sum = sum(c for _, c in unit_rows)
        if row_sum != subtotals["unit"]:
            diagnostics.append(Diagnostic(
                level="error",
                category="summary_arithmetic",
                section="summary_table",
                detail=(
                    f"unit test rows sum to {row_sum} but the unit "
                    f"subtotal claims {subtotals['unit']}"
                ),
            ))

    # Check integration subtotal
    if "integration" in subtotals and integration_rows:
        row_sum = sum(c for _, c in integration_rows)
        if row_sum != subtotals["integration"]:
            diagnostics.append(Diagnostic(
                level="error",
                category="summary_arithmetic",
                section="summary_table",
                detail=(
                    f"integration test rows sum to {row_sum} but the "
                    f"integration subtotal claims {subtotals['integration']}"
                ),
            ))

    # Check auxiliary subtotal
    if "auxiliary" in subtotals and auxiliary_rows:
        row_sum = sum(c for _, c in auxiliary_rows)
        if row_sum != subtotals["auxiliary"]:
            diagnostics.append(Diagnostic(
                level="error",
                category="summary_arithmetic",
                section="summary_table",
                detail=(
                    f"auxiliary test rows sum to {row_sum} but the "
                    f"auxiliary subtotal claims {subtotals['auxiliary']}"
                ),
            ))

    # Check total equals sum of subtotals
    if total is not None and subtotals:
        subtotal_sum = sum(subtotals.values())
        if subtotal_sum != total:
            diagnostics.append(Diagnostic(
                level="error",
                category="summary_arithmetic",
                section="summary_table",
                detail=(
                    f"subtotals sum to {subtotal_sum} but the total "
                    f"row claims {total}"
                ),
            ))

    return diagnostics


def check_summary_vs_catalog(
    summary: dict,
    catalog_counts: dict[str, int],
) -> list[Diagnostic]:
    """Check 5: Verify that summary table per-category counts match catalog entries.

    Compares the count claimed for each row in the summary table against the
    actual number of test entries in the corresponding catalog table section.

    The mapping between summary row names and catalog sections:
      - "neuroncite-xxx unit tests" maps to the unit catalog crate section
      - "Full pipeline integration tests" maps to all integration entries
        except T-CLI-* and T-SRV-* (includes T-INT-, T-E2E-, T-OCR-, T-REAL-)
      - "CLI integration tests" maps to T-CLI entries
      - "API server integration tests" maps to T-SRV entries
      - "Property-based tests" maps to T-PROP entries
      - "Regression and edge case tests" maps to T-REG entries
      - "Performance benchmarks" maps to T-PERF entries

    Args:
        summary: The parsed summary table dict.
        catalog_counts: A dict mapping category keys to their actual
            entry counts in the catalog tables.

    Returns:
        A list of Diagnostic instances for any mismatches.
    """
    diagnostics: list[Diagnostic] = []

    for name, count, _ in summary["rows"]:
        # Determine which catalog category this summary row maps to
        catalog_key = None

        # Unit test rows: "neuroncite-xxx unit tests"
        crate_match = re.match(r"(neuroncite-\w+)\s+unit\s+tests", name)
        if crate_match:
            catalog_key = f"unit:{crate_match.group(1)}"

        # Integration subsections
        if "Full pipeline" in name:
            catalog_key = "integration:pipeline"
        elif "CLI integration" in name:
            catalog_key = "integration:cli"
        elif "API server" in name:
            catalog_key = "integration:server"

        # Auxiliary categories
        if "Property-based" in name or "Property" in name:
            catalog_key = "auxiliary:property"
        if "Regression" in name:
            catalog_key = "auxiliary:regression"
        if "Performance" in name:
            catalog_key = "auxiliary:performance"

        if catalog_key and catalog_key in catalog_counts:
            actual = catalog_counts[catalog_key]
            if actual != count:
                diagnostics.append(Diagnostic(
                    level="error",
                    category="summary_vs_catalog",
                    section="summary_table",
                    detail=(
                        f'summary row "{name}" claims {count} tests but '
                        f"the catalog table has {actual} entries"
                    ),
                ))

    return diagnostics


def check_latex_tests_in_code(
    latex_ids: set[str],
    code_ids: set[str],
) -> list[Diagnostic]:
    """Check 6: Verify that every test ID in LaTeX catalogs exists in code.

    Test IDs documented in the architecture document should have corresponding
    test implementations in the Rust codebase. A LaTeX-only test ID means
    the documented test was either not implemented or the ID was changed.

    Args:
        latex_ids: Set of all test IDs from LaTeX catalog tables.
        code_ids: Set of all test IDs from code doc comments.

    Returns:
        A list of Diagnostic instances for any LaTeX-only IDs.
    """
    diagnostics: list[Diagnostic] = []

    for test_id in sorted(latex_ids - code_ids):
        diagnostics.append(Diagnostic(
            level="error",
            category="latex_tests_in_code",
            section="test_coverage",
            detail=(
                f"test {test_id} is documented in the LaTeX catalog but "
                f"has no matching test function in the code"
            ),
        ))

    return diagnostics


def check_code_tests_in_latex(
    code_ids: set[str],
    latex_ids: set[str],
    code_id_files: dict[str, list[str]],
) -> list[Diagnostic]:
    """Check 7: Verify that every test ID in code exists in LaTeX catalogs.

    Test functions with T-XXX-NNN doc comment IDs should be documented in
    the corresponding LaTeX catalog table. A code-only test ID means the
    test was added to code without updating the documentation.

    Args:
        code_ids: Set of all test IDs from code doc comments.
        latex_ids: Set of all test IDs from LaTeX catalog tables.
        code_id_files: Mapping from test ID to file paths (for messages).

    Returns:
        A list of Diagnostic instances for any code-only IDs.
    """
    diagnostics: list[Diagnostic] = []

    for test_id in sorted(code_ids - latex_ids):
        files = code_id_files.get(test_id, ["unknown"])
        file_str = files[0] if files else "unknown"
        diagnostics.append(Diagnostic(
            level="error",
            category="code_tests_in_latex",
            section="test_coverage",
            detail=(
                f"test {test_id} exists in code ({file_str}) but is "
                f"missing from the LaTeX catalog"
            ),
        ))

    return diagnostics


def check_undocumented_prefixes(
    code_prefixes: set[str],
    latex_prefixes: set[str],
) -> list[Diagnostic]:
    """Check 8: Verify that every test ID prefix in code has a LaTeX catalog.

    If a test prefix exists in code (e.g., T-E2E, T-REAL, T-OCR) but has
    no entries at all in any LaTeX catalog, the entire test category is
    undocumented. This is a broader issue than individual missing tests.

    Sub-prefixes (e.g., MCP-BATCH) are considered documented if their
    top-level parent prefix (e.g., MCP) has at least one entry in a LaTeX
    catalog. The individual missing tests are already reported by checks
    6 and 7; this check targets entirely new, undocumented categories.

    Args:
        code_prefixes: Set of all base prefixes found in code test IDs.
        latex_prefixes: Set of all base prefixes found in LaTeX test IDs.

    Returns:
        A list of Diagnostic instances for any undocumented prefixes.
    """
    diagnostics: list[Diagnostic] = []

    # Build a set of top-level latex prefixes (first segment only)
    # e.g., from {"CORE", "MCP", "MCP-IDX"} extract {"CORE", "MCP"}
    latex_top_level = {p.split("-")[0] for p in latex_prefixes}

    for prefix in sorted(code_prefixes - latex_prefixes):
        # Check if the top-level segment of this code prefix is documented.
        # For MCP-BATCH, the top-level segment is "MCP".
        top_level = prefix.split("-")[0]
        if top_level in latex_top_level:
            # Parent prefix is documented; individual missing tests are
            # reported by checks 6 and 7.
            continue

        diagnostics.append(Diagnostic(
            level="error",
            category="undocumented_prefixes",
            section="test_catalog_structure",
            detail=(
                f"test prefix T-{prefix} exists in code but has no "
                f"corresponding section in any LaTeX catalog table"
            ),
        ))

    return diagnostics


def check_unit_crate_counts(
    summary_rows: list[tuple[str, int, str]],
    crate_sections: dict[str, list[str]],
) -> list[Diagnostic]:
    """Check 9: Verify per-crate unit test counts match between summary and catalog.

    Each "neuroncite-xxx unit tests" row in the summary table claims a specific
    count. This check compares that count against the actual number of test
    entries in the corresponding crate section of the unit test catalog.

    Args:
        summary_rows: The rows from the summary table.
        crate_sections: Dict mapping crate names to their test ID lists
            from the unit test catalog.

    Returns:
        A list of Diagnostic instances for any mismatches.
    """
    diagnostics: list[Diagnostic] = []

    for name, count, test_type in summary_rows:
        if test_type != "Unit":
            continue

        # Extract crate name from the summary row name
        crate_match = re.match(r"(neuroncite-\w+)\s+unit\s+tests", name)
        if not crate_match:
            continue

        crate_name = crate_match.group(1)

        # Find the matching crate section in the catalog
        # The catalog keys might have LaTeX remnants, so normalize
        catalog_count = None
        for catalog_crate, ids in crate_sections.items():
            if crate_name in catalog_crate:
                catalog_count = len(ids)
                break

        if catalog_count is not None and catalog_count != count:
            diagnostics.append(Diagnostic(
                level="error",
                category="unit_crate_counts",
                section="unit_tests",
                detail=(
                    f'summary claims {count} tests for "{crate_name}" '
                    f"but the unit catalog section has {catalog_count} entries"
                ),
            ))

    return diagnostics


def check_summary_caption_match(summary: dict) -> list[Diagnostic]:
    """Check 10: Verify that the summary table body total matches the caption.

    The summary table has a bold "Total" row with a count, and a \\caption
    line that independently claims a total. These two numbers should agree.

    Args:
        summary: The parsed summary table dict.

    Returns:
        A list of Diagnostic instances if the counts differ.
    """
    diagnostics: list[Diagnostic] = []

    total = summary["total"]
    caption_total = summary["caption_total"]

    if total is not None and caption_total is not None:
        if total != caption_total:
            diagnostics.append(Diagnostic(
                level="error",
                category="summary_caption_match",
                section="summary_table",
                detail=(
                    f"summary table body total is {total} but the "
                    f"caption claims {caption_total}"
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

        lines.append(
            f"{prefix} {diag.category}: {diag.section} -- {diag.detail}",
        )

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
    """Run all 10 validation checks and return the combined diagnostics.

    Reads the LaTeX document and scans the Rust source tree for test IDs.
    Parses all four LaTeX catalog tables and the summary table, then
    executes each check function in order.

    Args:
        tex_path: Absolute path to architecture.tex.
        root_path: Absolute path to the Cargo workspace root.

    Returns:
        A combined list of Diagnostic instances from all checks.
    """
    raw_tex = tex_path.read_text(encoding="utf-8")
    tex_content = resolve_inputs(raw_tex, tex_path.parent)

    # ----- Parse LaTeX catalogs -----
    unit_ids, unit_caption = parse_unit_test_catalog(tex_content)
    integration_ids, integration_caption = parse_integration_test_catalog(
        tex_content,
    )
    prop_reg_ids, prop_reg_caption = parse_property_regression_catalog(
        tex_content,
    )
    perf_ids, perf_caption = parse_performance_catalog(tex_content)

    crate_sections = parse_unit_test_crate_sections(tex_content)
    summary = parse_summary_table(tex_content)

    # Combine all LaTeX test IDs
    all_latex_ids = unit_ids + integration_ids + prop_reg_ids + perf_ids
    latex_id_set = set(all_latex_ids)

    # Group LaTeX IDs by prefix for sequential check
    latex_by_prefix: dict[str, list[str]] = {}
    for tid in all_latex_ids:
        prefix = _base_prefix(tid)
        if prefix not in latex_by_prefix:
            latex_by_prefix[prefix] = []
        latex_by_prefix[prefix].append(tid)

    # ----- Parse code test IDs -----
    code_by_prefix = parse_code_test_ids(root_path)
    code_id_files = map_code_test_files(root_path)

    # Flatten code IDs into a set (deduplicate across files)
    code_id_set: set[str] = set()
    for ids in code_by_prefix.values():
        code_id_set.update(ids)

    # ----- Build catalog counts for summary comparison -----
    # Map summary row categories to actual catalog entry counts
    catalog_counts: dict[str, int] = {}

    # Per-crate unit test counts from the catalog
    for crate_name, ids in crate_sections.items():
        catalog_counts[f"unit:{crate_name}"] = len(ids)

    # Integration subsections by prefix.
    # CLI and server tests are identified by their T-CLI- and T-SRV- prefixes.
    # Full pipeline tests include all remaining integration IDs (T-INT-, T-E2E-,
    # T-OCR-, T-REAL-, and any other prefix that appears in the "Full pipeline
    # integration tests" sub-section of the integration catalog longtable).
    int_cli = [t for t in integration_ids if t.startswith("T-CLI-")]
    int_server = [t for t in integration_ids if t.startswith("T-SRV-")]
    int_pipeline = [
        t for t in integration_ids
        if not t.startswith("T-CLI-") and not t.startswith("T-SRV-")
    ]
    catalog_counts["integration:pipeline"] = len(int_pipeline)
    catalog_counts["integration:cli"] = len(int_cli)
    catalog_counts["integration:server"] = len(int_server)

    # Auxiliary categories
    prop_tests = [t for t in prop_reg_ids if t.startswith("T-PROP-")]
    reg_tests = [t for t in prop_reg_ids if t.startswith("T-REG-")]
    catalog_counts["auxiliary:property"] = len(prop_tests)
    catalog_counts["auxiliary:regression"] = len(reg_tests)
    catalog_counts["auxiliary:performance"] = len(perf_ids)

    # ----- Run all checks -----
    diagnostics: list[Diagnostic] = []

    # Check 1: Duplicate IDs
    diagnostics.extend(
        check_test_id_duplicates(all_latex_ids, code_id_files),
    )

    # Check 2: Sequential numbering (both LaTeX and code)
    diagnostics.extend(
        check_test_id_sequential(latex_by_prefix, "latex"),
    )
    diagnostics.extend(
        check_test_id_sequential(code_by_prefix, "code"),
    )

    # Check 3: Caption counts
    catalogs = [
        ("unit_test_catalog", unit_ids, unit_caption),
        ("integration_test_catalog", integration_ids, integration_caption),
        ("property_regression_catalog", prop_reg_ids, prop_reg_caption),
        ("performance_catalog", perf_ids, perf_caption),
    ]
    diagnostics.extend(check_catalog_caption_counts(catalogs))

    # Check 4: Summary arithmetic
    if summary:
        diagnostics.extend(check_summary_arithmetic(summary))

    # Check 5: Summary vs catalog
    if summary:
        diagnostics.extend(
            check_summary_vs_catalog(summary, catalog_counts),
        )

    # Check 6: LaTeX IDs in code
    diagnostics.extend(
        check_latex_tests_in_code(latex_id_set, code_id_set),
    )

    # Check 7: Code IDs in LaTeX
    diagnostics.extend(
        check_code_tests_in_latex(code_id_set, latex_id_set, code_id_files),
    )

    # Check 8: Undocumented prefixes
    latex_prefixes = set(latex_by_prefix.keys())
    code_prefixes = set(code_by_prefix.keys())
    diagnostics.extend(
        check_undocumented_prefixes(code_prefixes, latex_prefixes),
    )

    # Check 9: Unit crate counts
    if summary:
        diagnostics.extend(
            check_unit_crate_counts(
                summary["rows"], crate_sections,
            ),
        )

    # Check 10: Summary caption match
    if summary:
        diagnostics.extend(check_summary_caption_match(summary))

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
            "Test Catalog Validator for NeuronCite. "
            "Checks bidirectional consistency between the LaTeX test "
            "catalog documentation and the Rust test code. Detects "
            "missing tests, count mismatches, numbering inconsistencies, "
            "and arithmetic errors."
        ),
    )
    parser.add_argument(
        "--tex",
        required=True,
        type=Path,
        help="Path to the LaTeX architecture document",
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

    # Verify that the crates directory exists
    crates_dir = root_path / "crates"
    if not crates_dir.is_dir():
        print(
            f"Error: crates directory not found: {crates_dir}",
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
