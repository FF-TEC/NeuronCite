#!/usr/bin/env python3
"""Schema-Documentation Consistency Validator for NeuronCite.

This script enforces bidirectional consistency between the database schema
documented in the LaTeX architecture document (docs/architecture.tex) and
the actual SQL DDL defined in the Rust source file
(crates/neuroncite-store/src/schema.rs). It parses schema tables, index
definitions, trigger declarations, and FTS5 configuration from both
sources, then performs eight validation checks to detect discrepancies
in either direction.

The validator is described in Section 20.8 of the architecture document
and runs as the final CI pipeline stage alongside validate_architecture.py.

Exit codes:
    0 -- All checks pass (no errors, and no warnings unless --strict is active).
    1 -- At least one error-level discrepancy was found (or a warning under --strict).
    2 -- The script cannot parse the LaTeX file or the schema.rs file.

Invocation:
    python tools/validate_schemas.py --tex docs/architecture.tex --schema crates/neuroncite-store/src/schema.rs [--strict] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures for parsed schema information from both sources
# ---------------------------------------------------------------------------

@dataclass
class ColumnSpec:
    """A column definition extracted from either LaTeX documentation or SQL DDL.

    Attributes:
        name: The column name (e.g., "directory_path").
        sql_type: The normalized SQL type string (e.g., "TEXT NOT NULL").
        table: The table this column belongs to (e.g., "index_session").
    """
    name: str
    sql_type: str
    table: str


@dataclass
class IndexSpec:
    """An index definition extracted from LaTeX documentation or SQL DDL.

    Attributes:
        name: The index name (e.g., "idx_chunk_session").
        table: The table this index is defined on (e.g., "chunk").
        is_unique: Whether the index has the UNIQUE qualifier.
    """
    name: str
    table: str
    is_unique: bool


@dataclass
class TriggerSpec:
    """A trigger definition extracted from LaTeX documentation or SQL DDL.

    Attributes:
        name: The trigger name (e.g., "chunk_fts_insert").
        event: The trigger event (e.g., "AFTER INSERT").
        table: The table the trigger is defined on (e.g., "chunk").
    """
    name: str
    event: str
    table: str


@dataclass
class Diagnostic:
    """A single validation diagnostic (error or warning).

    Attributes:
        level: Either "error" or "warning".
        category: The check category identifier (e.g., "table_existence").
        table_name: The table this diagnostic applies to.
        detail: A human-readable description of the discrepancy.
    """
    level: str
    category: str
    table_name: str
    detail: str


# ---------------------------------------------------------------------------
# LaTeX text sanitization utilities
# ---------------------------------------------------------------------------

def sanitize_latex_name(raw: str) -> str:
    """Convert a LaTeX-formatted column or table name to a plain SQL identifier.

    Strips LaTeX-specific formatting: \\allowbreak commands, escaped underscores
    (\\_), \\raggedright, \\texttt{} wrappers, and leading/trailing whitespace.
    The resulting string is a plain identifier like "doc_text_offset_start".

    Args:
        raw: The raw name string extracted from a LaTeX table cell.

    Returns:
        A cleaned SQL identifier string.
    """
    cleaned = raw.replace("\\allowbreak", "")
    cleaned = cleaned.replace("\\_", "_")
    cleaned = cleaned.replace("\\raggedright", "")
    # Remove \texttt{...} wrappers, keeping inner content
    cleaned = re.sub(r"\\texttt\{([^}]*)\}", r"\1", cleaned)
    # Collapse all whitespace (including newlines from multi-line names)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.strip()


def sanitize_latex_type(raw: str) -> str:
    """Normalize a SQL type string extracted from LaTeX documentation.

    Expands abbreviations used in tabularx tables (PK -> PRIMARY KEY,
    NN -> NOT NULL), strips LaTeX formatting commands, and normalizes
    whitespace. The result is an uppercase canonical type string suitable
    for direct comparison with normalized SQL DDL types.

    Args:
        raw: The raw type string from a LaTeX table cell.

    Returns:
        A normalized, uppercase SQL type string (e.g., "INTEGER PRIMARY KEY").
    """
    cleaned = raw.replace("\\raggedright", "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Expand abbreviations used in tabularx tables (indexed_file, page)
    cleaned = re.sub(r"\bPK\b", "PRIMARY KEY", cleaned)
    cleaned = re.sub(r"\bNN\b", "NOT NULL", cleaned)
    return cleaned.upper()


# ---------------------------------------------------------------------------
# LaTeX parsing routines
# ---------------------------------------------------------------------------

def _join_multiline_rows(table_body: str) -> list[str]:
    """Join LaTeX table rows spanning multiple source lines into logical rows.

    A LaTeX table row is terminated by \\\\ (double backslash). Column names
    with \\allowbreak (e.g., doc\\_text\\_\\allowbreak offset\\_\\allowbreak
    start) and long descriptions can cause a single row to span multiple
    source lines. This function accumulates source lines until the row
    terminator is encountered, then emits the joined row as one string.

    LaTeX structural commands (\\toprule, \\midrule, \\bottomrule,
    \\endfirsthead, \\endhead, \\endfoot, \\normalfont, \\begin, \\end)
    are emitted as standalone entries without joining.

    Args:
        table_body: Raw LaTeX table body text between \\begin{} and \\caption{}.

    Returns:
        A list of joined logical row strings.
    """
    lines = table_body.split("\n")
    joined: list[str] = []
    current_parts: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Structural LaTeX commands are standalone entries
        is_structural = (
            stripped.startswith("\\toprule")
            or stripped.startswith("\\midrule")
            or stripped.startswith("\\bottomrule")
            or stripped.startswith("\\endfirsthead")
            or stripped.startswith("\\endhead")
            or stripped.startswith("\\endfoot")
            or stripped.startswith("\\normalfont")
            or stripped.startswith("\\begin{")
            or stripped.startswith("\\end{")
        )

        if is_structural:
            # Flush any accumulated parts before the structural command
            if current_parts:
                joined.append(" ".join(current_parts))
                current_parts = []
            joined.append(stripped)
            continue

        current_parts.append(stripped)

        # The LaTeX row terminator \\\\ (two backslashes) marks end of row
        if stripped.endswith("\\\\"):
            joined.append(" ".join(current_parts))
            current_parts = []

    # Flush remaining parts (e.g., {\small PK = ...} after \end{tabularx})
    if current_parts:
        joined.append(" ".join(current_parts))

    return joined


def parse_schema_tables(tex_content: str) -> dict[str, list[ColumnSpec]]:
    """Extract column definitions from all formal schema tables in the LaTeX document.

    Finds all longtable and tabularx environments whose caption matches
    "Schema of the \\texttt{<name>} table" and extracts column name and SQL
    type from each data row. The column name cell uses \\ttfamily formatting
    (from the table column spec) with \\_  for underscores and \\allowbreak
    for line-break hints. The type cell contains SQL type keywords, optionally
    abbreviated as PK (PRIMARY KEY) or NN (NOT NULL) in tabularx tables.

    Args:
        tex_content: The full LaTeX document content as a string.

    Returns:
        A dict mapping table name to a list of ColumnSpec instances.
    """
    tables: dict[str, list[ColumnSpec]] = {}

    # Find all schema table captions and extract the table name
    caption_pattern = r"\\caption\{Schema of the \\texttt\{([^}]+)\} table"
    caption_matches = list(re.finditer(caption_pattern, tex_content))

    for cap_match in caption_matches:
        table_name = sanitize_latex_name(cap_match.group(1))
        caption_pos = cap_match.start()

        # Search backward for the nearest \begin{longtable}, \begin{table},
        # or \begin{tabularx} that contains this caption
        table_start_pattern = r"\\begin\{(?:longtable|tabularx|table)\}"
        preceding = tex_content[:caption_pos]
        table_starts = list(re.finditer(table_start_pattern, preceding))
        if not table_starts:
            continue
        table_start = table_starts[-1].start()

        # The table body is between the table start and the caption
        table_body = tex_content[table_start:caption_pos]

        # Join multi-line rows into single logical lines
        logical_rows = _join_multiline_rows(table_body)

        columns: list[ColumnSpec] = []

        for row in logical_rows:
            # Skip structural LaTeX commands and lines without column separators
            if row.startswith("\\") or "&" not in row:
                continue

            # Split into at most 3 parts: name & type & description
            parts = row.split("&", 2)
            if len(parts) < 3:
                continue

            col_name_raw = parts[0].strip()
            col_type_raw = parts[1].strip()

            # Skip the header row (Column & Type & Description)
            if "textbf" in col_name_raw or "textbf" in col_type_raw:
                continue

            col_name = sanitize_latex_name(col_name_raw)
            col_type = sanitize_latex_type(col_type_raw)

            # Skip empty names (artifacts from non-data lines)
            if not col_name:
                continue

            columns.append(ColumnSpec(
                name=col_name,
                sql_type=col_type,
                table=table_name,
            ))

        tables[table_name] = columns

    return tables


def parse_latex_indexes(tex_content: str) -> list[IndexSpec]:
    """Extract named index definitions from the LaTeX document.

    Finds indexes from two structured sources:
    1. Verbatim DDL blocks (\\begin{verbatim}...\\end{verbatim}) containing
       CREATE [UNIQUE] INDEX statements. These appear in Unique constraint
       sections for expression-based indexes like idx_session_unique.
    2. \\textbf{Indexes:} itemized lists with \\texttt{idx_name} entries.
       These appear after the chunk table for its four supporting indexes.

    Inline UNIQUE constraints in CREATE TABLE bodies (e.g., UNIQUE(file_id,
    page_number)) are not tracked here -- they are implicit SQLite indexes
    without user-specified names.

    Args:
        tex_content: The full LaTeX document content as a string.

    Returns:
        A list of IndexSpec instances for all explicitly documented indexes.
    """
    indexes: list[IndexSpec] = []
    seen_names: set[str] = set()

    # Source 1: Verbatim DDL blocks containing CREATE INDEX statements
    verbatim_pattern = r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}"
    for match in re.finditer(verbatim_pattern, tex_content, re.DOTALL):
        block = match.group(1)
        create_idx_pattern = (
            r"CREATE\s+(UNIQUE\s+)?INDEX\s+"
            r"(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s+ON\s+(\w+)"
        )
        for idx_match in re.finditer(create_idx_pattern, block, re.IGNORECASE):
            is_unique = idx_match.group(1) is not None
            idx_name = idx_match.group(2)
            table = idx_match.group(3)

            if idx_name not in seen_names:
                indexes.append(IndexSpec(
                    name=idx_name, table=table, is_unique=is_unique,
                ))
                seen_names.add(idx_name)

    # Source 2: \textbf{Indexes:} sections followed by \begin{itemize}
    # Extract \texttt{idx_name} from each \item in the list
    indexes_section_pattern = (
        r"\\textbf\{Indexes:\}"
        r"(.*?)"
        r"(?=\\(?:subsection|section|textbf\{(?!Indexes))|\Z)"
    )
    for section_match in re.finditer(indexes_section_pattern, tex_content, re.DOTALL):
        section = section_match.group(1)
        # Match index names: \texttt{idx_something} (stop at space or })
        idx_name_pattern = r"\\texttt\{(idx[^}\s]+)"
        for idx_match in re.finditer(idx_name_pattern, section):
            raw_name = idx_match.group(1)
            idx_name = sanitize_latex_name(raw_name)

            if idx_name in seen_names:
                continue

            table = _infer_table_from_index_name(idx_name)
            indexes.append(IndexSpec(
                name=idx_name, table=table, is_unique=False,
            ))
            seen_names.add(idx_name)

    return indexes


def _infer_table_from_index_name(idx_name: str) -> str:
    """Infer the table name from an index name using NeuronCite naming conventions.

    Index names follow the pattern idx_<table>_<suffix>. For compound table
    names (e.g., citation_row), all segments before the final descriptor
    suffix are part of the table name.

    Known prefix-to-table mappings:
        idx_session_*      -> index_session
        idx_chunk_*        -> chunk
        idx_citation_row_* -> citation_row

    Args:
        idx_name: The index name string (e.g., "idx_chunk_session").

    Returns:
        The inferred table name string.
    """
    if idx_name.startswith("idx_session_"):
        return "index_session"
    if idx_name.startswith("idx_chunk_"):
        return "chunk"
    if idx_name.startswith("idx_citation_row_"):
        return "citation_row"
    # Fallback: use the second component of the underscore-separated name
    parts = idx_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def parse_latex_triggers(tex_content: str) -> list[TriggerSpec]:
    """Extract trigger definitions from the LaTeX document.

    Finds triggers documented in the FTS5 subsection under the "Trigger
    correctness:" heading. Each trigger entry in the itemized list has the
    format (with LaTeX escapes and potential line wrapping):
        \\texttt{chunk\\_fts\\_insert} --- \\texttt{AFTER INSERT} on
        \\texttt{chunk}

    The four FTS5 synchronization triggers handle INSERT, DELETE, soft-delete
    (UPDATE OF is_deleted 0->1), and un-delete (UPDATE OF is_deleted 1->0).

    Args:
        tex_content: The full LaTeX document content as a string.

    Returns:
        A list of TriggerSpec instances for all documented triggers.
    """
    triggers: list[TriggerSpec] = []

    # Pattern for trigger entries in the itemized list. LaTeX escapes:
    # - Trigger names use \_ for underscores (e.g., chunk\_fts\_insert)
    # - Event text uses \_ in column names (e.g., AFTER UPDATE OF is\_deleted)
    # - The "on \texttt{chunk}" part may wrap to the next source line,
    #   so \s+ handles both spaces and newlines (with re.DOTALL).
    trigger_pattern = (
        r"\\texttt\{(chunk[^}]+)\}\s*---\s*"
        r"\\texttt\{(AFTER[^}]+)\}\s+on\s+"
        r"\\texttt\{(\w+)\}"
    )

    for match in re.finditer(trigger_pattern, tex_content, re.DOTALL):
        raw_name = match.group(1)
        raw_event = match.group(2)
        raw_table = match.group(3)

        name = sanitize_latex_name(raw_name)
        # Sanitize the event string: remove \_ escapes, normalize whitespace
        event = raw_event.replace("\\_", "_").strip().upper()
        event = re.sub(r"\s+", " ", event)
        table = raw_table.replace("\\_", "_")

        triggers.append(TriggerSpec(name=name, event=event, table=table))

    return triggers


def parse_latex_fts5(tex_content: str) -> dict[str, str]:
    """Extract FTS5 virtual table configuration from the LaTeX document.

    Parses the FTS5 configuration from two locations:
    1. The "FTS5 virtual table configuration" tabularx table for the
       tokenizer setting (Tokenizer row).
    2. Inline \\texttt{} references for the virtual table name
       (\\texttt{chunk_fts} USING fts5) and content table
       (content=\\texttt{chunk}).

    Args:
        tex_content: The full LaTeX document content as a string.

    Returns:
        A dict with keys "table_name", "content_table", "tokenizer" mapping
        to their documented string values.
    """
    config: dict[str, str] = {}

    # Virtual table name: \texttt{chunk_fts} USING fts5 (or with \_ escapes)
    fts5_name_match = re.search(
        r"\\texttt\{(chunk(?:\\_)?fts)\}\s+USING\s+fts5",
        tex_content,
    )
    if fts5_name_match:
        config["table_name"] = sanitize_latex_name(fts5_name_match.group(1))

    # Content table: content=\texttt{chunk}
    content_match = re.search(r"content=\\texttt\{(\w+)\}", tex_content)
    if content_match:
        config["content_table"] = content_match.group(1)

    # Tokenizer: found in the FTS5 config table's "Tokenizer" row
    tokenizer_match = re.search(
        r"Tokenizer\s*&\s*\\texttt\{(\w+)\}",
        tex_content,
    )
    if tokenizer_match:
        config["tokenizer"] = tokenizer_match.group(1)

    return config


def parse_citation_row_prose(tex_content: str) -> Optional[int]:
    """Extract the claimed column count for citation_row from prose description.

    The citation_row table is described in prose rather than a formal schema
    table (Section 4, "Database Schema" subsection). This function extracts
    the number from text patterns like "It contains 19 columns including...".

    Args:
        tex_content: The full LaTeX document content as a string.

    Returns:
        The claimed column count as an integer, or None if not found.
    """
    count_match = re.search(
        r"(?:contains|has)\s+(\d+)\s+columns",
        tex_content,
        re.IGNORECASE,
    )
    if count_match:
        return int(count_match.group(1))
    return None


# ---------------------------------------------------------------------------
# SQL/Rust source parsing routines
# ---------------------------------------------------------------------------

def parse_sql_tables(schema_rs: str) -> dict[str, list[ColumnSpec]]:
    """Parse CREATE TABLE statements from schema.rs to extract column definitions.

    Searches the entire schema.rs file for CREATE TABLE IF NOT EXISTS patterns.
    The SQL DDL keywords only appear inside Rust string literals, so searching
    the raw file content is equivalent to parsing the embedded SQL. For each
    table, extracts the body between the opening and closing parentheses
    (respecting nesting depth) and parses column definitions from it.

    Lines that define table-level constraints (UNIQUE, FOREIGN KEY) rather
    than columns are skipped.

    Args:
        schema_rs: The full content of schema.rs as a string.

    Returns:
        A dict mapping table name to a list of ColumnSpec instances.
    """
    tables: dict[str, list[ColumnSpec]] = {}

    table_pattern = r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)\s*\("
    for match in re.finditer(table_pattern, schema_rs, re.IGNORECASE):
        table_name = match.group(1)
        body_start = match.end()

        # Find the matching closing parenthesis (handle nested parens
        # from REFERENCES, COALESCE, etc.)
        depth = 1
        pos = body_start
        while pos < len(schema_rs) and depth > 0:
            if schema_rs[pos] == "(":
                depth += 1
            elif schema_rs[pos] == ")":
                depth -= 1
            pos += 1

        body = schema_rs[body_start:pos - 1]
        columns = _parse_column_definitions(body, table_name)
        tables[table_name] = columns

    return tables


def _parse_column_definitions(body: str, table_name: str) -> list[ColumnSpec]:
    """Parse column definitions from a CREATE TABLE body string.

    Splits the body by top-level commas (not inside nested parentheses) and
    identifies column definition lines. Each column line has the format:
        column_name TYPE [NOT NULL] [DEFAULT ...] [REFERENCES ...]

    Lines starting with constraint keywords (UNIQUE, FOREIGN KEY, CHECK,
    CONSTRAINT, PRIMARY KEY) are skipped as they define table-level
    constraints rather than columns.

    Args:
        body: The text between the opening and closing parentheses of a
            CREATE TABLE statement.
        table_name: The table name for ColumnSpec attribution.

    Returns:
        A list of ColumnSpec instances, one per column.
    """
    columns: list[ColumnSpec] = []
    parts = _split_top_level_commas(body)

    for part in parts:
        line = part.strip()
        # Remove SQL comments (-- style)
        line = re.sub(r"--.*$", "", line, flags=re.MULTILINE)
        line = " ".join(line.split())

        if not line:
            continue

        # Skip table-level constraint definitions
        upper_line = line.upper().lstrip()
        if upper_line.startswith((
            "UNIQUE", "FOREIGN", "CHECK", "CONSTRAINT", "PRIMARY KEY",
        )):
            continue

        # Parse column: column_name followed by type and optional constraints
        col_match = re.match(r"(\w+)\s+(.+)", line)
        if col_match:
            col_name = col_match.group(1)
            col_type_raw = col_match.group(2).strip()
            col_type = _normalize_sql_type(col_type_raw)

            columns.append(ColumnSpec(
                name=col_name,
                sql_type=col_type,
                table=table_name,
            ))

    return columns


def _split_top_level_commas(text: str) -> list[str]:
    """Split text by commas that are not inside parentheses.

    Tracks parenthesis nesting depth to avoid splitting on commas inside
    COALESCE(), REFERENCES(), or other SQL constructs that contain commas
    within parenthesized arguments.

    Args:
        text: The SQL text to split.

    Returns:
        A list of comma-separated segments.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []

    for char in text:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)

    if current:
        parts.append("".join(current))

    return parts


def _normalize_sql_type(raw_type: str) -> str:
    """Normalize a SQL column type definition for cross-source comparison.

    Extracts the base type and key modifiers (PRIMARY KEY, NOT NULL,
    AUTOINCREMENT) while stripping DEFAULT clauses, REFERENCES foreign key
    constraints, and ON DELETE CASCADE specifications. These stripped elements
    are not relevant for schema validation because the LaTeX documentation
    describes them in the Description column rather than the Type column.

    Args:
        raw_type: The raw SQL type string from a column definition in a
            CREATE TABLE statement.

    Returns:
        A normalized, uppercase type string (e.g., "INTEGER PRIMARY KEY AUTOINCREMENT").
    """
    # Remove REFERENCES table(column) [ON DELETE CASCADE] foreign key specs
    cleaned = re.sub(
        r"REFERENCES\s+\w+\([^)]*\)(\s+ON\s+DELETE\s+\w+)?",
        "",
        raw_type,
        flags=re.IGNORECASE,
    )
    # Remove DEFAULT 'string_value' clauses (including empty string DEFAULT '')
    cleaned = re.sub(r"DEFAULT\s+'[^']*'", "", cleaned, flags=re.IGNORECASE)
    # Remove DEFAULT numeric_value clauses (DEFAULT 0, DEFAULT -1)
    cleaned = re.sub(r"DEFAULT\s+-?\d+", "", cleaned, flags=re.IGNORECASE)
    # Remove inline CHECK constraints. CHECK constraints enforce application-
    # level invariants and are not part of the base type declaration. The
    # LaTeX architecture document records the base type (e.g. "TEXT NOT NULL")
    # without the CHECK expression, so stripping it here allows the comparison
    # to succeed when a column adds a CHECK without changing its base type.
    # The CHECK clause always appears after the base type and may contain nested
    # parentheses (e.g. CHECK (col IN ('a', 'b'))), so stripping from CHECK
    # to end-of-string is the most reliable approach.
    cleaned = re.sub(r"CHECK\b.*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    # Collapse whitespace and uppercase for canonical comparison
    cleaned = re.sub(r"\s+", " ", cleaned).strip().upper()
    return cleaned


def parse_sql_indexes(schema_rs: str) -> list[IndexSpec]:
    """Extract CREATE INDEX and CREATE UNIQUE INDEX statements from schema.rs.

    Searches for all CREATE [UNIQUE] INDEX IF NOT EXISTS patterns in the raw
    file content. These SQL statements only appear inside Rust string literals.

    Args:
        schema_rs: The full content of schema.rs as a string.

    Returns:
        A list of IndexSpec instances for all named indexes in the DDL.
    """
    indexes: list[IndexSpec] = []

    idx_pattern = (
        r"CREATE\s+(UNIQUE\s+)?INDEX\s+IF\s+NOT\s+EXISTS\s+(\w+)\s+ON\s+(\w+)"
    )

    for match in re.finditer(idx_pattern, schema_rs, re.IGNORECASE):
        is_unique = match.group(1) is not None
        idx_name = match.group(2)
        table = match.group(3)

        indexes.append(IndexSpec(
            name=idx_name,
            table=table,
            is_unique=is_unique,
        ))

    return indexes


def parse_sql_triggers(schema_rs: str) -> list[TriggerSpec]:
    """Extract CREATE TRIGGER statements from schema.rs.

    Searches for all CREATE TRIGGER IF NOT EXISTS patterns in the raw file
    content. Each trigger specifies its timing (AFTER/BEFORE), event type
    (INSERT/DELETE/UPDATE OF column), and target table.

    Args:
        schema_rs: The full content of schema.rs as a string.

    Returns:
        A list of TriggerSpec instances for all triggers in the DDL.
    """
    triggers: list[TriggerSpec] = []

    trigger_pattern = (
        r"CREATE\s+TRIGGER\s+IF\s+NOT\s+EXISTS\s+(\w+)\s+"
        r"(AFTER|BEFORE)\s+(INSERT|DELETE|UPDATE(?:\s+OF\s+\w+)?)\s+ON\s+(\w+)"
    )

    for match in re.finditer(trigger_pattern, schema_rs, re.IGNORECASE):
        name = match.group(1)
        timing = match.group(2).upper()
        event = match.group(3).upper()
        table = match.group(4)

        triggers.append(TriggerSpec(
            name=name,
            event=f"{timing} {event}",
            table=table,
        ))

    return triggers


def parse_sql_fts5(schema_rs: str) -> dict[str, str]:
    """Extract FTS5 virtual table configuration from schema.rs.

    Parses the CREATE VIRTUAL TABLE ... USING fts5(...) statement to extract
    the virtual table name, the backing content table, and the tokenizer
    identifier.

    The tokenize parameter in the Rust source uses escaped quotes (\\\")
    because it is embedded inside a Rust string literal. The extraction
    regex accounts for this escaping.

    Args:
        schema_rs: The full content of schema.rs as a string.

    Returns:
        A dict with keys "table_name", "content_table", "tokenizer".
    """
    config: dict[str, str] = {}

    fts5_pattern = (
        r"CREATE\s+VIRTUAL\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)\s+"
        r"USING\s+fts5\((.*?)\)"
    )
    match = re.search(fts5_pattern, schema_rs, re.IGNORECASE | re.DOTALL)
    if match:
        config["table_name"] = match.group(1)
        fts5_body = match.group(2)

        # Extract the content= backing table parameter
        content_match = re.search(r"content=(\w+)", fts5_body)
        if content_match:
            config["content_table"] = content_match.group(1)

        # Extract the tokenizer identifier after tokenize= and any
        # escaped quotes or whitespace from the Rust string embedding
        tokenizer_match = re.search(
            r'tokenize\s*=\s*["\\\s]*(\w+)',
            fts5_body,
        )
        if tokenizer_match:
            config["tokenizer"] = tokenizer_match.group(1)

    return config


def parse_ensure_columns(schema_rs: str) -> list[ColumnSpec]:
    """Extract column additions from the ensure_columns() repair function.

    The ensure_columns() function in schema.rs uses ALTER TABLE ... ADD COLUMN
    statements to add missing columns to databases created by older code. These
    columns are part of the current schema and should also appear in the CREATE
    TABLE DDL. This function serves as a safety net: if a repair column is
    somehow missing from the CREATE TABLE, it is still captured for validation.

    Args:
        schema_rs: The full content of schema.rs as a string.

    Returns:
        A list of ColumnSpec instances for columns added by ensure_columns().
    """
    columns: list[ColumnSpec] = []

    alter_pattern = r"ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+(\w+)\s+([^;]+)"
    for match in re.finditer(alter_pattern, schema_rs, re.IGNORECASE):
        table = match.group(1)
        col_name = match.group(2)
        col_type = _normalize_sql_type(match.group(3).strip())

        columns.append(ColumnSpec(
            name=col_name,
            sql_type=col_type,
            table=table,
        ))

    return columns


# ---------------------------------------------------------------------------
# Validation check implementations
# ---------------------------------------------------------------------------

def check_table_existence(
    latex_tables: dict[str, list[ColumnSpec]],
    sql_tables: dict[str, list[ColumnSpec]],
) -> list[Diagnostic]:
    """Check 1: Table existence mismatch (error severity).

    Verifies that every table with a formal schema table in the LaTeX document
    has a corresponding CREATE TABLE in schema.rs, and vice versa. Tables that
    are only described in prose (like citation_row) are not included in the
    latex_tables dict and will be reported as missing from documentation.
    """
    diagnostics: list[Diagnostic] = []

    latex_names = set(latex_tables.keys())
    sql_names = set(sql_tables.keys())

    for name in sorted(latex_names - sql_names):
        diagnostics.append(Diagnostic(
            level="error",
            category="table_existence",
            table_name=name,
            detail=(
                "table documented in architecture.tex but no "
                "CREATE TABLE in schema.rs"
            ),
        ))

    for name in sorted(sql_names - latex_names):
        diagnostics.append(Diagnostic(
            level="error",
            category="table_existence",
            table_name=name,
            detail=(
                "table has CREATE TABLE in schema.rs but no formal "
                "schema table in architecture.tex"
            ),
        ))

    return diagnostics


def check_columns_missing_in_code(
    latex_tables: dict[str, list[ColumnSpec]],
    sql_tables: dict[str, list[ColumnSpec]],
) -> list[Diagnostic]:
    """Check 2: Column documented but missing in code (error severity).

    For each table that has a formal schema table in both the LaTeX document
    and the SQL DDL, verifies that every column listed in the documentation
    exists in the CREATE TABLE statement.
    """
    diagnostics: list[Diagnostic] = []

    for table_name in sorted(set(latex_tables.keys()) & set(sql_tables.keys())):
        latex_col_names = {c.name for c in latex_tables[table_name]}
        sql_col_names = {c.name for c in sql_tables[table_name]}

        for col in sorted(latex_col_names - sql_col_names):
            diagnostics.append(Diagnostic(
                level="error",
                category="column_missing_in_code",
                table_name=table_name,
                detail=(
                    f"column '{col}' documented in architecture.tex "
                    f"but absent from CREATE TABLE in schema.rs"
                ),
            ))

    return diagnostics


def check_columns_missing_in_docs(
    latex_tables: dict[str, list[ColumnSpec]],
    sql_tables: dict[str, list[ColumnSpec]],
) -> list[Diagnostic]:
    """Check 3: Column in code but missing from documentation (error severity).

    For each table that has a formal schema table in both sources, verifies
    that every column in the CREATE TABLE statement is documented in the
    LaTeX schema table.
    """
    diagnostics: list[Diagnostic] = []

    for table_name in sorted(set(latex_tables.keys()) & set(sql_tables.keys())):
        latex_col_names = {c.name for c in latex_tables[table_name]}
        sql_col_names = {c.name for c in sql_tables[table_name]}

        for col in sorted(sql_col_names - latex_col_names):
            diagnostics.append(Diagnostic(
                level="error",
                category="column_missing_in_docs",
                table_name=table_name,
                detail=(
                    f"column '{col}' in CREATE TABLE (schema.rs) "
                    f"but absent from architecture.tex"
                ),
            ))

    return diagnostics


def check_column_type_mismatch(
    latex_tables: dict[str, list[ColumnSpec]],
    sql_tables: dict[str, list[ColumnSpec]],
) -> list[Diagnostic]:
    """Check 4: Column type mismatch (error severity).

    For columns that exist in both the LaTeX documentation and the SQL DDL,
    verifies that the SQL type string matches after normalization. Both sides
    are uppercased, abbreviations expanded (PK -> PRIMARY KEY, NN -> NOT NULL),
    and non-type clauses (DEFAULT, REFERENCES) stripped before comparison.
    """
    diagnostics: list[Diagnostic] = []

    for table_name in sorted(set(latex_tables.keys()) & set(sql_tables.keys())):
        latex_by_name = {c.name: c for c in latex_tables[table_name]}
        sql_by_name = {c.name: c for c in sql_tables[table_name]}

        common_cols = set(latex_by_name.keys()) & set(sql_by_name.keys())

        for col_name in sorted(common_cols):
            latex_type = latex_by_name[col_name].sql_type
            sql_type = sql_by_name[col_name].sql_type

            if latex_type != sql_type:
                diagnostics.append(Diagnostic(
                    level="error",
                    category="column_type_mismatch",
                    table_name=table_name,
                    detail=(
                        f"column '{col_name}' type mismatch: "
                        f"docs='{latex_type}' code='{sql_type}'"
                    ),
                ))

    return diagnostics


def check_index_existence(
    latex_indexes: list[IndexSpec],
    sql_indexes: list[IndexSpec],
) -> list[Diagnostic]:
    """Check 5: Index existence mismatch (error severity).

    Verifies that every named index documented in the LaTeX document exists
    in schema.rs, and vice versa. Only explicitly named indexes (CREATE INDEX
    statements) are compared; implicit indexes from inline UNIQUE constraints
    are not tracked.
    """
    diagnostics: list[Diagnostic] = []

    latex_names = {idx.name for idx in latex_indexes}
    sql_names = {idx.name for idx in sql_indexes}

    for name in sorted(latex_names - sql_names):
        table = next(
            (i.table for i in latex_indexes if i.name == name), "unknown",
        )
        diagnostics.append(Diagnostic(
            level="error",
            category="index_existence",
            table_name=table,
            detail=(
                f"index '{name}' documented in architecture.tex "
                f"but absent from schema.rs"
            ),
        ))

    for name in sorted(sql_names - latex_names):
        table = next(
            (i.table for i in sql_indexes if i.name == name), "unknown",
        )
        diagnostics.append(Diagnostic(
            level="error",
            category="index_existence",
            table_name=table,
            detail=(
                f"index '{name}' in schema.rs but not documented "
                f"in architecture.tex"
            ),
        ))

    return diagnostics


def check_trigger_existence(
    latex_triggers: list[TriggerSpec],
    sql_triggers: list[TriggerSpec],
) -> list[Diagnostic]:
    """Check 6: Trigger existence mismatch (error severity).

    Verifies that every trigger documented in the LaTeX FTS5 section exists
    in schema.rs, and vice versa.
    """
    diagnostics: list[Diagnostic] = []

    latex_names = {t.name for t in latex_triggers}
    sql_names = {t.name for t in sql_triggers}

    for name in sorted(latex_names - sql_names):
        table = next(
            (t.table for t in latex_triggers if t.name == name), "unknown",
        )
        diagnostics.append(Diagnostic(
            level="error",
            category="trigger_existence",
            table_name=table,
            detail=(
                f"trigger '{name}' documented in architecture.tex "
                f"but absent from schema.rs"
            ),
        ))

    for name in sorted(sql_names - latex_names):
        table = next(
            (t.table for t in sql_triggers if t.name == name), "unknown",
        )
        diagnostics.append(Diagnostic(
            level="error",
            category="trigger_existence",
            table_name=table,
            detail=(
                f"trigger '{name}' in schema.rs but not documented "
                f"in architecture.tex"
            ),
        ))

    return diagnostics


def check_fts5_config(
    latex_fts5: dict[str, str],
    sql_fts5: dict[str, str],
) -> list[Diagnostic]:
    """Check 7: FTS5 configuration mismatch (warning severity).

    Compares the FTS5 virtual table name, backing content table, and tokenizer
    identifier between the LaTeX documentation and the SQL DDL in schema.rs.
    """
    diagnostics: list[Diagnostic] = []

    for key in ("table_name", "content_table", "tokenizer"):
        latex_val = latex_fts5.get(key)
        sql_val = sql_fts5.get(key)

        if latex_val and sql_val and latex_val != sql_val:
            diagnostics.append(Diagnostic(
                level="warning",
                category="fts5_config",
                table_name="chunk_fts",
                detail=(
                    f"FTS5 {key} mismatch: "
                    f"docs='{latex_val}' code='{sql_val}'"
                ),
            ))
        elif latex_val and not sql_val:
            diagnostics.append(Diagnostic(
                level="warning",
                category="fts5_config",
                table_name="chunk_fts",
                detail=f"FTS5 {key}='{latex_val}' in docs but not found in code",
            ))
        elif sql_val and not latex_val:
            diagnostics.append(Diagnostic(
                level="warning",
                category="fts5_config",
                table_name="chunk_fts",
                detail=f"FTS5 {key}='{sql_val}' in code but not found in docs",
            ))

    return diagnostics


def check_column_count_prose(
    prose_count: Optional[int],
    sql_tables: dict[str, list[ColumnSpec]],
) -> list[Diagnostic]:
    """Check 8: Prose column count mismatch (warning severity).

    For tables described only in prose (citation_row), verifies that the
    claimed column count in the prose text matches the actual column count
    in the CREATE TABLE DDL.
    """
    diagnostics: list[Diagnostic] = []

    if prose_count is not None and "citation_row" in sql_tables:
        actual_count = len(sql_tables["citation_row"])
        if prose_count != actual_count:
            diagnostics.append(Diagnostic(
                level="warning",
                category="column_count_mismatch",
                table_name="citation_row",
                detail=(
                    f"prose claims {prose_count} columns but "
                    f"CREATE TABLE has {actual_count}"
                ),
            ))

    return diagnostics


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_human_readable(diagnostics: list[Diagnostic]) -> str:
    """Format a list of diagnostics as human-readable text lines.

    Each line has the format:
        [ERROR|WARN] <category>: <table_name> -- <detail>

    A summary line at the end reports total error and warning counts.
    """
    lines: list[str] = []
    error_count = 0
    warning_count = 0

    for diag in diagnostics:
        if diag.level == "error":
            prefix = "[ERROR]"
            error_count += 1
        else:
            prefix = "[WARN]"
            warning_count += 1

        lines.append(
            f"{prefix} {diag.category}: {diag.table_name} -- {diag.detail}"
        )

    lines.append(f"SUMMARY: {error_count} error(s), {warning_count} warning(s)")
    return "\n".join(lines)


def format_json(diagnostics: list[Diagnostic]) -> str:
    """Format a list of diagnostics as a JSON string.

    The JSON object has three keys: "errors" (list of error entries),
    "warnings" (list of warning entries), and "summary" (counts).
    """
    errors = []
    warnings = []

    for diag in diagnostics:
        entry = {
            "category": diag.category,
            "table": diag.table_name,
            "detail": diag.detail,
        }
        if diag.level == "error":
            errors.append(entry)
        else:
            warnings.append(entry)

    result = {
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_validation(tex_path: Path, schema_path: Path) -> list[Diagnostic]:
    """Execute all eight validation checks and return the combined diagnostics.

    Parses the LaTeX document for formal schema table definitions, index
    references, trigger documentation, and FTS5 configuration. Parses
    schema.rs for SQL DDL statements. Then runs each check to detect
    discrepancies between the two sources.
    """
    tex_content = tex_path.read_text(encoding="utf-8")
    schema_rs = schema_path.read_text(encoding="utf-8")

    # Parse LaTeX documentation
    latex_tables = parse_schema_tables(tex_content)
    latex_indexes = parse_latex_indexes(tex_content)
    latex_triggers = parse_latex_triggers(tex_content)
    latex_fts5 = parse_latex_fts5(tex_content)
    prose_count = parse_citation_row_prose(tex_content)

    # Parse SQL DDL from Rust source
    sql_tables = parse_sql_tables(schema_rs)
    sql_indexes = parse_sql_indexes(schema_rs)
    sql_triggers = parse_sql_triggers(schema_rs)
    sql_fts5 = parse_sql_fts5(schema_rs)
    repair_columns = parse_ensure_columns(schema_rs)

    # Merge repair columns into sql_tables as a safety net. The
    # ensure_columns() function adds columns to databases created by older
    # code. These columns should already be present in the CREATE TABLE
    # DDL, but if not, they are still part of the current schema.
    for col in repair_columns:
        if col.table in sql_tables:
            existing_names = {c.name for c in sql_tables[col.table]}
            if col.name not in existing_names:
                sql_tables[col.table].append(col)

    # Run all eight validation checks
    all_diagnostics: list[Diagnostic] = []

    all_diagnostics.extend(check_table_existence(latex_tables, sql_tables))
    all_diagnostics.extend(
        check_columns_missing_in_code(latex_tables, sql_tables),
    )
    all_diagnostics.extend(
        check_columns_missing_in_docs(latex_tables, sql_tables),
    )
    all_diagnostics.extend(
        check_column_type_mismatch(latex_tables, sql_tables),
    )
    all_diagnostics.extend(check_index_existence(latex_indexes, sql_indexes))
    all_diagnostics.extend(
        check_trigger_existence(latex_triggers, sql_triggers),
    )
    all_diagnostics.extend(check_fts5_config(latex_fts5, sql_fts5))
    all_diagnostics.extend(check_column_count_prose(prose_count, sql_tables))

    return all_diagnostics


def main() -> int:
    """Parse command-line arguments, run validation, and produce output.

    Returns:
        Exit code: 0 if all checks pass, 1 if discrepancies found, 2 if
        the script cannot parse the LaTeX file or locate schema.rs.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Schema-Documentation Consistency Validator for NeuronCite. "
            "Enforces bidirectional consistency between the LaTeX architecture "
            "document and the SQL DDL in schema.rs."
        ),
    )
    parser.add_argument(
        "--tex",
        required=True,
        type=Path,
        help=(
            "Path to the LaTeX architecture document "
            "(e.g., docs/architecture.tex)"
        ),
    )
    parser.add_argument(
        "--schema",
        required=True,
        type=Path,
        help=(
            "Path to the Rust schema source file "
            "(e.g., crates/neuroncite-store/src/schema.rs)"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat warnings as errors "
            "(used in CI to block the pipeline on any discrepancy)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable text",
    )
    args = parser.parse_args()

    tex_path: Path = args.tex.resolve()
    schema_path: Path = args.schema.resolve()

    if not tex_path.is_file():
        print(
            f"Error: LaTeX file not found: {tex_path}",
            file=sys.stderr,
        )
        return 2

    if not schema_path.is_file():
        print(
            f"Error: schema.rs file not found: {schema_path}",
            file=sys.stderr,
        )
        return 2

    try:
        diagnostics = run_validation(tex_path, schema_path)
    except Exception as exc:
        print(f"Error: failed to run validation: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(format_json(diagnostics))
    else:
        if diagnostics:
            print(format_human_readable(diagnostics), file=sys.stderr)
        else:
            print("All checks passed.", file=sys.stderr)

    error_count = sum(1 for d in diagnostics if d.level == "error")
    warning_count = sum(1 for d in diagnostics if d.level == "warning")

    if error_count > 0:
        return 1
    if args.strict and warning_count > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
