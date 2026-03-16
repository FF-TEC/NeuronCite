#!/usr/bin/env python3
"""
Manages the AGPL-3.0 license header across all .rs source files in the workspace.

Detection is based on verbatim string comparison (str.startswith), not
heuristics or regex, so doc-comments (//!) and regular code comments are
never touched.

Modes:
  --remove          Strip the license header from every .rs file that contains it.
  --set [FILE]      Insert the official AGPL-3.0 header into every .rs file.
                    Without FILE, the built-in header (matching the LICENSE
                    template) is used.  With FILE, the contents of that file
                    are used instead.  Files that already have the header
                    are skipped; files without it receive it.

Safety:
  --dry-run         Print what would happen without writing any file.
                    Always run this first to verify the changeset.

Scope:
  The script walks crates/**/*.rs from the workspace root. Directories named
  "target" and "node_modules" are excluded automatically.

Examples:
  python tools/dev/license_header.py --dry-run --remove
  python tools/dev/license_header.py --remove
  python tools/dev/license_header.py --dry-run --set
  python tools/dev/license_header.py --set
  python tools/dev/license_header.py --set path/to/custom_header.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Official AGPL-3.0 header matching the template in the LICENSE file
# (section "How to Apply These Terms to Your New Programs", lines 632-646).
# This is the text that --set uses when called without a FILE argument.
# ---------------------------------------------------------------------------
OFFICIAL_HEADER = (
    "// NeuronCite -- local, privacy-preserving semantic document search engine.\n"
    "// Copyright (C) 2026 NeuronCite Contributors\n"
    "//\n"
    "// This program is free software: you can redistribute it and/or modify\n"
    "// it under the terms of the GNU Affero General Public License as published by\n"
    "// the Free Software Foundation, either version 3 of the License, or\n"
    "// (at your option) any later version.\n"
    "//\n"
    "// This program is distributed in the hope that it will be useful,\n"
    "// but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
    "// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
    "// GNU Affero General Public License for more details.\n"
    "//\n"
    "// You should have received a copy of the GNU Affero General Public License\n"
    "// along with this program.  If not, see <https://www.gnu.org/licenses/>.\n"
)

# Directories that are never traversed.
EXCLUDED_DIRS = {"target", "node_modules", ".git"}


def find_rs_files(root: Path) -> list[Path]:
    """Collect all .rs files under crates/, skipping excluded directories.

    Returns a sorted list so the output is deterministic across runs.
    """
    crates_dir = root / "crates"
    if not crates_dir.is_dir():
        print(f"ERROR: directory not found: {crates_dir}", file=sys.stderr)
        sys.exit(1)

    results: list[Path] = []
    for path in crates_dir.rglob("*.rs"):
        # Skip files whose path contains an excluded directory segment.
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        results.append(path)

    results.sort()
    return results


def remove_header(files: list[Path], root: Path, header: str, dry_run: bool) -> None:
    """Remove the AGPL header from every file that starts with it.

    After removing the header lines, one optional blank line that
    immediately follows the header block is also stripped so there is no
    double-blank before the first doc-comment or code line.
    """
    removed = 0
    skipped = 0

    for path in files:
        content = path.read_text(encoding="utf-8")

        if not content.startswith(header):
            skipped += 1
            continue

        # Strip the matched header and at most one trailing blank line.
        remaining = content[len(header):]
        if remaining.startswith("\n"):
            remaining = remaining[1:]

        removed += 1
        rel = path.relative_to(root)
        if dry_run:
            print(f"  WOULD REMOVE : {rel}")
        else:
            path.write_text(remaining, encoding="utf-8")
            print(f"  REMOVED      : {rel}")

    label = "Dry-run summary" if dry_run else "Summary"
    print(f"\n{label}: {removed} file(s) affected, {skipped} file(s) skipped.")


def set_header(files: list[Path], root: Path, header_text: str, dry_run: bool) -> None:
    """Insert the header into every .rs file that does not already have it.

    Files that already start with the header are skipped.
    """
    # Ensure the header ends with exactly one newline followed by one blank
    # line so there is a visual separator between header and code.
    header_block = header_text.rstrip("\n") + "\n\n"

    added = 0
    already_present = 0

    for path in files:
        content = path.read_text(encoding="utf-8")

        # Already has the target header -- skip.
        if content.startswith(header_block.rstrip("\n")):
            already_present += 1
            continue

        rel = path.relative_to(root)
        new_content = header_block + content
        added += 1

        if dry_run:
            print(f"  WOULD ADD    : {rel}")
        else:
            path.write_text(new_content, encoding="utf-8")
            print(f"  ADDED        : {rel}")

    label = "Dry-run summary" if dry_run else "Summary"
    print(
        f"\n{label}: {added} file(s) affected, "
        f"{already_present} already had the header."
    )


def resolve_workspace_root() -> Path:
    """Walk upward from the script location to find the workspace root.

    The workspace root is identified by the presence of a top-level Cargo.toml
    that contains [workspace].
    """
    candidate = Path(__file__).resolve().parent
    while candidate != candidate.parent:
        cargo = candidate / "Cargo.toml"
        if cargo.is_file() and "[workspace]" in cargo.read_text(encoding="utf-8"):
            return candidate
        candidate = candidate.parent

    print("ERROR: could not locate workspace root (Cargo.toml with [workspace]).", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the AGPL license header in all .rs source files.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--remove",
        action="store_true",
        help="Remove the AGPL header from all .rs files that contain it.",
    )
    mode.add_argument(
        "--set",
        metavar="FILE",
        nargs="?",
        const=None,
        default=argparse.SUPPRESS,
        type=Path,
        help=(
            "Insert a license header into all .rs files.  Without FILE, the "
            "built-in official AGPL-3.0 header is used.  With FILE, the "
            "contents of that file are used instead."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without modifying any files.",
    )

    args = parser.parse_args()

    # Determine whether --set was used (with or without FILE argument).
    is_set_mode = hasattr(args, "set")

    root = resolve_workspace_root()
    files = find_rs_files(root)

    print(f"Workspace root : {root}")
    print(f"Files scanned  : {len(files)}")
    if args.remove:
        mode_str = "--remove"
    elif is_set_mode and args.set is not None:
        mode_str = f"--set {args.set}"
    else:
        mode_str = "--set (built-in official AGPL-3.0 header)"
    print(f"Mode           : {mode_str}")
    print(f"Dry run        : {args.dry_run}")
    print()

    if args.remove:
        # Load the header to match: either from a file or the built-in.
        remove_header(files, root, OFFICIAL_HEADER, args.dry_run)
    else:
        header_path: Path | None = args.set if is_set_mode else None
        if header_path is not None:
            if not header_path.is_file():
                print(f"ERROR: header file not found: {header_path}", file=sys.stderr)
                sys.exit(1)
            header_text = header_path.read_text(encoding="utf-8")
            if not header_text.strip():
                print("ERROR: header file is empty.", file=sys.stderr)
                sys.exit(1)
        else:
            header_text = OFFICIAL_HEADER
            print("Using built-in official AGPL-3.0 header.\n")
        set_header(files, root, header_text, args.dry_run)


if __name__ == "__main__":
    main()
