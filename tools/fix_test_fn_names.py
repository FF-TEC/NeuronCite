#!/usr/bin/env python3
"""One-time script to rename Rust test functions whose numeric ID does not
match the test ID declared in the preceding doc comment.

Each test function follows the naming convention ``t_{prefix}_{number}_{description}``,
and each doc comment declares a canonical test ID like ``/// T-PREFIX-NNN:``.
When the number in the function name differs from the doc-comment ID, this
script renames the function to align with the doc comment.

Usage:
  python tools/fix_test_fn_names.py --root .              # apply renames
  python tools/fix_test_fn_names.py --root . --dry-run    # preview without writing
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Regex matching the first line of a test-ID doc comment: /// T-PREFIX-NNN:
DOC_ID_RE = re.compile(
    r"///\s+(T-([A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*)-(\d{3}[a-z]?)):"
)

# Regex matching a test function definition line (with optional async keyword)
FN_DEF_RE = re.compile(
    r"^(\s*(?:pub\s+)?(?:async\s+)?fn\s+)(t_\w+)(\s*\()"
)


def _doc_id_to_prefix_number(doc_id: str) -> tuple[str, str]:
    """Extract the lowercase prefix and numeric part from a doc-comment test ID.

    ``T-STO-061``      returns ``("sto", "061")``.
    ``T-ANNOTATE-229`` returns ``("annotate", "229")``.
    ``T-MCP-BATCH-001`` returns ``("mcp_batch", "001")``.
    """
    rest = doc_id[2:]  # strip leading "T-"
    m = re.match(r"(.+)-(\d{3}[a-z]?)$", rest)
    if not m:
        raise ValueError(f"cannot parse doc ID: {doc_id}")
    prefix = m.group(1).lower().replace("-", "_")
    number = m.group(2)
    return prefix, number


def _fn_name_to_prefix_number(fn_name: str) -> tuple[str, str, str] | None:
    """Extract prefix, numeric part, and description from a test function name.

    ``t_sto_061_migration_creates_all_fts5_triggers``
    returns ``("sto", "061", "migration_creates_all_fts5_triggers")``.

    ``t_annotate_229_fallback_extract_serde``
    returns ``("annotate", "229", "fallback_extract_serde")``.

    Returns ``None`` if the function name does not follow the expected pattern.
    """
    parts = fn_name.split("_")
    if not parts or parts[0] != "t":
        return None
    # Find the first segment that is exactly 3 digits (optionally followed by
    # a single lowercase letter). Everything between "t" and that segment is the
    # prefix; everything after is the description.
    for i in range(1, len(parts)):
        if re.fullmatch(r"\d{3}[a-z]?", parts[i]):
            prefix = "_".join(parts[1:i])
            number = parts[i]
            description = "_".join(parts[i + 1:])
            return prefix, number, description
    return None


def process_file(path: Path, dry_run: bool) -> list[str]:
    """Process a single .rs file: detect and fix function-name / doc-ID mismatches.

    Args:
        path:    Absolute path to the Rust source file.
        dry_run: If True, report proposed renames without modifying the file.

    Returns:
        A list of human-readable rename descriptions (one per rename).
    """
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    lines = content.splitlines(keepends=True)
    renames: list[str] = []
    # Maps old function name to new function name for replacements
    replacements: dict[str, str] = []  # list of (line_index, old_name, new_name)

    pending_doc_id: str | None = None
    pending_line: int | None = None
    countdown: int = 0

    for idx, line in enumerate(lines):
        # Check for a doc-comment test ID
        doc_match = DOC_ID_RE.search(line)
        if doc_match:
            pending_doc_id = doc_match.group(1)
            pending_line = idx + 1  # 1-based line number
            countdown = 15  # look for the fn within the next 15 lines
            continue

        # If we have a pending doc ID, look for the corresponding fn definition
        if pending_doc_id is not None:
            countdown -= 1
            if countdown <= 0:
                pending_doc_id = None
                pending_line = None
                continue

            fn_match = FN_DEF_RE.match(line)
            if fn_match:
                fn_name = fn_match.group(2)
                parsed = _fn_name_to_prefix_number(fn_name)
                if parsed is None:
                    pending_doc_id = None
                    pending_line = None
                    continue

                fn_prefix, fn_number, fn_description = parsed
                doc_prefix, doc_number = _doc_id_to_prefix_number(pending_doc_id)

                # Only rename if the prefix matches but the number differs
                if fn_prefix == doc_prefix and fn_number != doc_number:
                    new_fn_name = f"t_{fn_prefix}_{doc_number}_{fn_description}"
                    replacements.append((idx, fn_name, new_fn_name))
                    renames.append(
                        f"  {path}:{idx + 1}\n"
                        f"    {fn_name} -> {new_fn_name}\n"
                        f"    (doc comment: {pending_doc_id})"
                    )

                pending_doc_id = None
                pending_line = None

    # Apply replacements in reverse order to preserve line indices
    if replacements and not dry_run:
        for line_idx, old_name, new_name in reversed(replacements):
            lines[line_idx] = lines[line_idx].replace(old_name, new_name, 1)
        path.write_text("".join(lines), encoding="utf-8")

    return renames


def main() -> int:
    """Entry point: parse arguments, scan all .rs files, apply renames."""
    parser = argparse.ArgumentParser(
        description="Rename Rust test functions to match their doc-comment test IDs.",
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to the Cargo workspace root directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposed renames without modifying files",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    crates_dir = root / "crates"
    if not crates_dir.is_dir():
        print(f"Error: crates directory not found: {crates_dir}", file=sys.stderr)
        return 2

    total_renames: list[str] = []
    files_changed = 0

    for rs_file in sorted(crates_dir.rglob("*.rs")):
        renames = process_file(rs_file, args.dry_run)
        if renames:
            files_changed += 1
            total_renames.extend(renames)

    action = "proposed" if args.dry_run else "applied"
    for r in total_renames:
        print(f"RENAME ({action}):\n{r}")

    print(f"\n{len(total_renames)} rename(s) {action} across {files_changed} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
