#!/usr/bin/env python3
"""Repository analytics tool for the NeuronCite project.

Walks the repository tree and produces a comprehensive report covering:
  - Per-language line counts (blank, comment, code)
  - Workspace crate breakdown with per-crate code/test/comment statistics
  - Test infrastructure analysis (#[test], #[tokio::test], #[cfg(test)], benchmarks)
  - Rust-specific code structure (functions, structs, enums, traits, impls, derives)
  - Safety audit (unsafe blocks, panic! calls, unwrap()/expect() usage)
  - Async runtime metrics (async fn count)
  - Dependency graph summary (workspace + transitive from Cargo.lock)
  - Code health indicators (TODO/FIXME, comment-to-code ratio)
  - Top N largest source files
  - Per-crate size ranking

Supported languages: Rust, Python, TOML, YAML, JSON, JavaScript,
TypeScript, HTML, CSS, SQL, Shell, TeX/LaTeX, Markdown.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

# ── language definitions ────────────────────────────────────────────────

class LangDef(NamedTuple):
    """Mapping from file extension to display name and comment syntax."""
    name: str
    line_comment: str | None        # single-line comment prefix, e.g. "//"
    block_open: str | None          # block comment opening, e.g. "/*"
    block_close: str | None         # block comment closing, e.g. "*/"

LANGUAGES: dict[str, LangDef] = {
    ".rs":   LangDef("Rust",       "//",  "/*",  "*/"),
    ".py":   LangDef("Python",     "#",   '"""', '"""'),
    ".toml": LangDef("TOML",       "#",   None,  None),
    ".yaml": LangDef("YAML",       "#",   None,  None),
    ".yml":  LangDef("YAML",       "#",   None,  None),
    ".json": LangDef("JSON",       None,  None,  None),
    ".js":   LangDef("JavaScript", "//",  "/*",  "*/"),
    ".ts":   LangDef("TypeScript", "//",  "/*",  "*/"),
    ".html": LangDef("HTML",       None,  "<!--","-->"),
    ".css":  LangDef("CSS",        None,  "/*",  "*/"),
    ".sql":  LangDef("SQL",        "--",  "/*",  "*/"),
    ".sh":   LangDef("Shell",      "#",   None,  None),
    ".tex":  LangDef("LaTeX",      "%",   None,  None),
    ".md":   LangDef("Markdown",   None,  None,  None),
}

# Directories to skip during traversal
SKIP_DIRS: set[str] = {
    "target", "node_modules", ".git", "__pycache__",
    ".neuroncite", "venv", ".venv", "dist", "build",
}

# ── per-file statistics ─────────────────────────────────────────────────

class FileStats(NamedTuple):
    """Raw line counts for a single source file."""
    total: int
    blank: int
    comment: int
    code: int

def count_file(path: Path, lang: LangDef) -> FileStats:
    """Count blank, comment, and code lines in *path* using *lang* rules.

    Block-comment detection is intentionally simple (no nesting support)
    because Rust doc-comments (///) are already handled by the line-comment
    prefix check.  The block-comment state machine covers /* ... */ and
    Python triple-quote docstrings.
    """
    blank = 0
    comment = 0
    code = 0
    in_block = False

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return FileStats(0, 0, 0, 0)

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # ── blank line ──────────────────────────────────────────────
        if not line:
            blank += 1
            continue

        # ── inside a block comment ──────────────────────────────────
        if in_block:
            comment += 1
            if lang.block_close and lang.block_close in line:
                in_block = False
            continue

        # ── block comment opening on this line ──────────────────────
        if lang.block_open and lang.block_open in line:
            # If the block opens and closes on the same line, count once
            if lang.block_close and lang.block_close in line[line.index(lang.block_open) + len(lang.block_open):]:
                comment += 1
            else:
                comment += 1
                in_block = True
            continue

        # ── single-line comment ─────────────────────────────────────
        if lang.line_comment and line.startswith(lang.line_comment):
            comment += 1
            continue

        # ── everything else is code ─────────────────────────────────
        code += 1

    total = blank + comment + code
    return FileStats(total, blank, comment, code)

# ── aggregation ─────────────────────────────────────────────────────────

class LangTotals:
    """Accumulated counters for one language."""
    __slots__ = ("files", "total", "blank", "comment", "code")

    def __init__(self) -> None:
        self.files = 0
        self.total = 0
        self.blank = 0
        self.comment = 0
        self.code = 0

    def add(self, fs: FileStats) -> None:
        self.files += 1
        self.total += fs.total
        self.blank += fs.blank
        self.comment += fs.comment
        self.code += fs.code


class CrateStats:
    """Accumulated counters for a single workspace crate."""
    __slots__ = ("name", "src_files", "test_files", "bench_files",
                 "src_code", "src_comment", "src_blank", "src_total",
                 "test_code", "test_total", "bench_total")

    def __init__(self, name: str) -> None:
        self.name = name
        self.src_files = 0
        self.test_files = 0
        self.bench_files = 0
        self.src_code = 0
        self.src_comment = 0
        self.src_blank = 0
        self.src_total = 0
        self.test_code = 0
        self.test_total = 0
        self.bench_total = 0


def collect(root: Path) -> dict[str, LangTotals]:
    """Walk *root* and return per-language totals, skipping ignored dirs."""
    totals: dict[str, LangTotals] = defaultdict(LangTotals)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune directories in-place so os.walk skips them
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in LANGUAGES:
                continue
            lang = LANGUAGES[ext]
            fpath = Path(dirpath) / fname
            stats = count_file(fpath, lang)
            totals[lang.name].add(stats)

    return dict(totals)

# ── Rust-specific analysis ──────────────────────────────────────────────

# Regex patterns for Rust code structure detection.
# Each pattern targets a specific syntactic construct at the
# beginning of a stripped line (or with leading whitespace for
# indented items inside impl blocks).

RE_TEST_ATTR       = re.compile(r"#\[test\]")
RE_TOKIO_TEST      = re.compile(r"#\[tokio::test")
RE_CFG_TEST        = re.compile(r"#\[cfg\(test\)\]")
RE_UNSAFE          = re.compile(r"\bunsafe\b")
RE_PANIC           = re.compile(r"\bpanic!\b")
RE_UNWRAP          = re.compile(r"\.unwrap\(\)")
RE_EXPECT          = re.compile(r"\.expect\(")
RE_TODO            = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b")
RE_ASYNC_FN        = re.compile(r"\basync\s+fn\b")
RE_FN_DEF          = re.compile(r"^\s*(pub(\([\w:]+\))?\s+)?fn\s+\w+")
RE_STRUCT_ENUM     = re.compile(r"^\s*(pub(\([\w:]+\))?\s+)?(struct|enum)\s+\w+")
RE_TRAIT_DEF       = re.compile(r"^\s*pub(\([\w:]+\))?\s+trait\s+\w+")
RE_IMPL_BLOCK      = re.compile(r"^impl[\s<]")
RE_DERIVE          = re.compile(r"#\[derive\(")
RE_USE_STMT        = re.compile(r"^use\s+")
RE_CRITERION       = re.compile(r"criterion_group!|criterion_main!")
RE_MOD_DECL        = re.compile(r"^\s*(pub(\([\w:]+\))?\s+)?mod\s+\w+")


class RustInsights:
    """Aggregated Rust-specific metrics across all .rs files in the repo."""
    __slots__ = (
        "test_fns", "tokio_test_fns", "cfg_test_modules",
        "unsafe_blocks", "unsafe_files",
        "panic_calls", "unwrap_calls", "expect_calls",
        "todo_count", "async_fns",
        "fn_defs", "struct_enum_defs", "trait_defs",
        "impl_blocks", "derive_attrs", "use_stmts",
        "criterion_macros", "mod_decls",
        "files_with_tests", "files_with_unsafe",
    )

    def __init__(self) -> None:
        self.test_fns = 0
        self.tokio_test_fns = 0
        self.cfg_test_modules = 0
        self.unsafe_blocks = 0
        self.unsafe_files: set[str] = set()
        self.panic_calls = 0
        self.unwrap_calls = 0
        self.expect_calls = 0
        self.todo_count = 0
        self.async_fns = 0
        self.fn_defs = 0
        self.struct_enum_defs = 0
        self.trait_defs = 0
        self.impl_blocks = 0
        self.derive_attrs = 0
        self.use_stmts = 0
        self.criterion_macros = 0
        self.mod_decls = 0
        self.files_with_tests: set[str] = set()
        self.files_with_unsafe = 0


def analyze_rust_file(path: Path, insights: RustInsights) -> None:
    """Scan a single .rs file and accumulate pattern counts into *insights*."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    file_has_test = False
    file_has_unsafe = False
    rel = str(path)

    for line in text.splitlines():
        stripped = line.strip()

        # Test markers
        if RE_TEST_ATTR.search(stripped):
            insights.test_fns += 1
            file_has_test = True
        if RE_TOKIO_TEST.search(stripped):
            insights.tokio_test_fns += 1
            file_has_test = True
        if RE_CFG_TEST.search(stripped):
            insights.cfg_test_modules += 1

        # Safety audit
        if RE_UNSAFE.search(stripped):
            # Skip lines that are just "// unsafe" comments
            if not stripped.startswith("//"):
                insights.unsafe_blocks += 1
                file_has_unsafe = True
        if RE_PANIC.search(stripped) and not stripped.startswith("//"):
            insights.panic_calls += 1
        if RE_UNWRAP.search(stripped) and not stripped.startswith("//"):
            insights.unwrap_calls += 1
        if RE_EXPECT.search(stripped) and not stripped.startswith("//"):
            insights.expect_calls += 1

        # Code health
        if RE_TODO.search(stripped):
            insights.todo_count += 1

        # Structure metrics
        if RE_ASYNC_FN.search(stripped):
            insights.async_fns += 1
        if RE_FN_DEF.match(stripped):
            insights.fn_defs += 1
        if RE_STRUCT_ENUM.match(stripped):
            insights.struct_enum_defs += 1
        if RE_TRAIT_DEF.match(stripped):
            insights.trait_defs += 1
        if RE_IMPL_BLOCK.match(stripped):
            insights.impl_blocks += 1
        if RE_DERIVE.search(stripped):
            insights.derive_attrs += 1
        if RE_USE_STMT.match(stripped):
            insights.use_stmts += 1
        if RE_CRITERION.search(stripped):
            insights.criterion_macros += 1
        if RE_MOD_DECL.match(stripped):
            insights.mod_decls += 1

    if file_has_test:
        insights.files_with_tests.add(rel)
    if file_has_unsafe:
        insights.unsafe_files.add(rel)


def collect_rust_insights(root: Path) -> RustInsights:
    """Walk the repository and gather Rust-specific metrics from all .rs files."""
    insights = RustInsights()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            if fname.endswith(".rs"):
                analyze_rust_file(Path(dirpath) / fname, insights)
    return insights

# ── crate-level breakdown ───────────────────────────────────────────────

def collect_crate_stats(root: Path) -> list[CrateStats]:
    """Compute per-crate statistics for each workspace member under crates/."""
    crates_dir = root / "crates"
    if not crates_dir.is_dir():
        return []

    rust_lang = LANGUAGES[".rs"]
    results: list[CrateStats] = []

    for entry in sorted(crates_dir.iterdir()):
        if not entry.is_dir():
            continue
        cargo_toml = entry / "Cargo.toml"
        if not cargo_toml.exists():
            continue

        cs = CrateStats(entry.name)

        for dirpath, dirnames, filenames in os.walk(entry):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            rel_dir = Path(dirpath).relative_to(entry).as_posix()

            for fname in filenames:
                if not fname.endswith(".rs"):
                    continue
                fpath = Path(dirpath) / fname
                stats = count_file(fpath, rust_lang)

                is_test = "tests" in rel_dir.split("/") or "tests" in str(fpath.stem)
                is_bench = "benches" in rel_dir.split("/")

                if is_bench:
                    cs.bench_files += 1
                    cs.bench_total += stats.total
                elif is_test:
                    cs.test_files += 1
                    cs.test_code += stats.code
                    cs.test_total += stats.total
                else:
                    cs.src_files += 1
                    cs.src_code += stats.code
                    cs.src_comment += stats.comment
                    cs.src_blank += stats.blank
                    cs.src_total += stats.total

        results.append(cs)

    return results

# ── dependency analysis ─────────────────────────────────────────────────

class DepInfo(NamedTuple):
    """Summary of the dependency graph from Cargo.lock."""
    total_packages: int         # total [[package]] entries in Cargo.lock
    workspace_packages: int     # packages whose name starts with "neuroncite"
    third_party_packages: int   # total_packages - workspace_packages

def analyze_dependencies(root: Path) -> DepInfo | None:
    """Parse Cargo.lock and count packages."""
    lock_path = root / "Cargo.lock"
    if not lock_path.exists():
        return None

    try:
        text = lock_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    # Each [[package]] block starts with 'name = "..."'
    names = re.findall(r'^name = "(.+)"', text, re.MULTILINE)
    total = len(names)
    ws = sum(1 for n in names if n.startswith("neuroncite"))
    return DepInfo(total, ws, total - ws)

# ── largest files detection ─────────────────────────────────────────────

class FileSizeEntry(NamedTuple):
    """A source file with its total line count for ranking."""
    path: str       # path relative to the repo root
    lines: int
    code: int
    comment: int

def find_largest_files(root: Path, top_n: int = 15) -> list[FileSizeEntry]:
    """Return the *top_n* largest source files by total line count."""
    entries: list[FileSizeEntry] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in LANGUAGES:
                continue
            lang = LANGUAGES[ext]
            fpath = Path(dirpath) / fname
            stats = count_file(fpath, lang)
            if stats.total > 0:
                try:
                    rel = fpath.relative_to(root).as_posix()
                except ValueError:
                    rel = str(fpath)
                entries.append(FileSizeEntry(rel, stats.total, stats.code, stats.comment))

    entries.sort(key=lambda e: e.lines, reverse=True)
    return entries[:top_n]

# ── output formatting ───────────────────────────────────────────────────

SEP_CHAR = "-"
SECTION_WIDTH = 78

def section_header(title: str) -> str:
    """Format a section title with dashes on both sides."""
    pad = SECTION_WIDTH - len(title) - 4
    left = pad // 2
    right = pad - left
    return f"\n{SEP_CHAR * left}  {title}  {SEP_CHAR * right}"


def fmt_number(n: int) -> str:
    """Format integer with thousands separator."""
    return f"{n:,}"


def fmt_pct(part: int, whole: int) -> str:
    """Format a percentage, returning '-' if the denominator is zero."""
    if whole == 0:
        return "-"
    return f"{part / whole * 100:.1f}%"


def print_table(headers: tuple, rows: list[tuple], widths: list[int] | None = None) -> None:
    """Print a formatted ASCII table with right-aligned numeric columns.

    The first column is left-aligned (labels). All remaining columns are
    right-aligned (numeric values). Column widths are auto-calculated from
    the header and row contents unless explicitly provided.
    """
    if widths is None:
        widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))

    def fmt_row(vals: tuple) -> str:
        parts = []
        for i, v in enumerate(vals):
            s = str(v)
            if i == 0:
                parts.append(s.ljust(widths[i]))
            else:
                parts.append(s.rjust(widths[i]))
        return " | ".join(parts)

    sep = SEP_CHAR.join(
        SEP_CHAR * (w + 2) if i > 0 else SEP_CHAR * w
        for i, w in enumerate(widths)
    )

    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)


def print_language_table(totals: dict[str, LangTotals]) -> None:
    """Print the per-language line count table sorted by code lines."""
    print(section_header("LINES OF CODE BY LANGUAGE"))
    print()

    headers = ("Language", "Files", "Blank", "Comment", "Code", "Total")
    rows: list[tuple] = []

    for name, t in totals.items():
        rows.append((name, fmt_number(t.files), fmt_number(t.blank),
                      fmt_number(t.comment), fmt_number(t.code), fmt_number(t.total)))

    # Sort by code lines descending (parse the formatted number for sorting)
    code_idx = 4
    rows.sort(key=lambda r: int(r[code_idx].replace(",", "")), reverse=True)

    # Grand totals
    sum_files   = sum(t.files for t in totals.values())
    sum_blank   = sum(t.blank for t in totals.values())
    sum_comment = sum(t.comment for t in totals.values())
    sum_code    = sum(t.code for t in totals.values())
    sum_total   = sum(t.total for t in totals.values())

    grand = ("TOTAL", fmt_number(sum_files), fmt_number(sum_blank),
             fmt_number(sum_comment), fmt_number(sum_code), fmt_number(sum_total))

    # Compute widths including grand totals row
    widths = [len(str(h)) for h in headers]
    for row in [*rows, grand]:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    print_table(headers, rows, widths)
    print(f"{'TOTAL':<{widths[0]}} | " + " | ".join(
        str(grand[i]).rjust(widths[i]) for i in range(1, len(grand))
    ))
    print()

    # Summary ratios
    if sum_code > 0:
        print(f"  Comment-to-code ratio : {sum_comment / sum_code * 100:.1f}%")
    print(f"  Files analysed        : {fmt_number(sum_files)}")
    print(f"  Total lines           : {fmt_number(sum_total)}")
    print()


def print_crate_breakdown(crates: list[CrateStats]) -> None:
    """Print per-crate statistics sorted by source code lines."""
    print(section_header("WORKSPACE CRATE BREAKDOWN"))
    print()

    headers = ("Crate", "Src Files", "Src Code", "Src Comment", "Test Files",
               "Test Code", "Bench Files")
    rows: list[tuple] = []

    for cs in crates:
        rows.append((
            cs.name.replace("neuroncite-", "nc-").replace("neuroncite", "nc"),
            str(cs.src_files),
            fmt_number(cs.src_code),
            fmt_number(cs.src_comment),
            str(cs.test_files),
            fmt_number(cs.test_code),
            str(cs.bench_files),
        ))

    # Sort by source code descending
    rows.sort(key=lambda r: int(r[2].replace(",", "")), reverse=True)

    # Grand totals
    g_src_files  = sum(c.src_files for c in crates)
    g_src_code   = sum(c.src_code for c in crates)
    g_src_comm   = sum(c.src_comment for c in crates)
    g_test_files = sum(c.test_files for c in crates)
    g_test_code  = sum(c.test_code for c in crates)
    g_bench      = sum(c.bench_files for c in crates)

    grand = ("TOTAL", str(g_src_files), fmt_number(g_src_code), fmt_number(g_src_comm),
             str(g_test_files), fmt_number(g_test_code), str(g_bench))

    widths = [len(str(h)) for h in headers]
    for row in [*rows, grand]:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    print_table(headers, rows, widths)
    print(f"{'TOTAL':<{widths[0]}} | " + " | ".join(
        str(grand[i]).rjust(widths[i]) for i in range(1, len(grand))
    ))
    print()


def print_test_insights(ri: RustInsights, crates: list[CrateStats]) -> None:
    """Print test infrastructure summary."""
    print(section_header("TEST INFRASTRUCTURE"))
    print()

    total_test_fns = ri.test_fns + ri.tokio_test_fns
    test_files_count = len(ri.files_with_tests)
    total_test_code = sum(c.test_code for c in crates)
    total_bench_files = sum(c.bench_files for c in crates)

    # Separate #[test] and #[tokio::test] are not mutually exclusive in the count
    # since #[tokio::test] also contains "test". The total is derived from both
    # attributes minus double-counted #[tokio::test] (which also matches #[test]
    # in some regex approaches). Here we count them independently:
    # ri.test_fns counts #[test] attributes (includes #[tokio::test] lines)
    # ri.tokio_test_fns counts #[tokio::test] specifically
    sync_tests = ri.test_fns - ri.tokio_test_fns
    async_tests = ri.tokio_test_fns

    print(f"  #[test] (sync)          : {fmt_number(sync_tests)}")
    print(f"  #[tokio::test] (async)  : {fmt_number(async_tests)}")
    print(f"  Total test functions    : {fmt_number(ri.test_fns)}")
    print(f"  #[cfg(test)] modules    : {fmt_number(ri.cfg_test_modules)}")
    print(f"  Files containing tests  : {fmt_number(test_files_count)}")
    print(f"  Dedicated test files    : {fmt_number(sum(c.test_files for c in crates))}")
    print(f"  Test code lines         : {fmt_number(total_test_code)}")
    print(f"  Benchmark files         : {fmt_number(total_bench_files)}")
    print(f"  Criterion groups        : {fmt_number(ri.criterion_macros)}")
    print()

    # Test density: tests per 1000 lines of source code
    total_src_code = sum(c.src_code for c in crates)
    if total_src_code > 0:
        density = ri.test_fns / total_src_code * 1000
        print(f"  Test density            : {density:.1f} tests per 1,000 LOC")
    # Test-to-code ratio: lines of test code vs lines of source code
    if total_src_code > 0:
        ratio = total_test_code / total_src_code * 100
        print(f"  Test-to-source ratio    : {ratio:.1f}%")
    print()


def print_code_structure(ri: RustInsights) -> None:
    """Print Rust code structure breakdown."""
    print(section_header("RUST CODE STRUCTURE"))
    print()

    print(f"  Function definitions    : {fmt_number(ri.fn_defs)}")
    print(f"    of which async fn     : {fmt_number(ri.async_fns)}")
    print(f"  Struct/Enum definitions : {fmt_number(ri.struct_enum_defs)}")
    print(f"  Trait definitions       : {fmt_number(ri.trait_defs)}")
    print(f"  impl blocks             : {fmt_number(ri.impl_blocks)}")
    print(f"  #[derive(..)] attrs     : {fmt_number(ri.derive_attrs)}")
    print(f"  Module declarations     : {fmt_number(ri.mod_decls)}")
    print(f"  use statements          : {fmt_number(ri.use_stmts)}")
    print()


def print_safety_audit(ri: RustInsights) -> None:
    """Print safety and robustness indicators."""
    print(section_header("SAFETY & ROBUSTNESS AUDIT"))
    print()

    print(f"  unsafe keyword usage    : {fmt_number(ri.unsafe_blocks)} across {len(ri.unsafe_files)} files")
    print(f"  panic!() calls          : {fmt_number(ri.panic_calls)}")
    print(f"  .unwrap() calls         : {fmt_number(ri.unwrap_calls)}")
    print(f"  .expect() calls         : {fmt_number(ri.expect_calls)}")
    print(f"  TODO/FIXME/HACK/XXX     : {fmt_number(ri.todo_count)}")
    print()

    if ri.todo_count == 0:
        print("  [CLEAN] No open TODO/FIXME markers in the codebase.")
        print()


def print_dependency_summary(dep_info: DepInfo | None) -> None:
    """Print dependency graph statistics from Cargo.lock."""
    print(section_header("DEPENDENCY GRAPH"))
    print()

    if dep_info is None:
        print("  Cargo.lock not found -- skipping dependency analysis.")
        print()
        return

    print(f"  Workspace crates        : {fmt_number(dep_info.workspace_packages)}")
    print(f"  Third-party crates      : {fmt_number(dep_info.third_party_packages)}")
    print(f"  Total (incl. transitive): {fmt_number(dep_info.total_packages)}")
    print()


def print_largest_files(entries: list[FileSizeEntry]) -> None:
    """Print the top N largest source files."""
    print(section_header("LARGEST SOURCE FILES"))
    print()

    headers = ("#", "File", "Lines", "Code", "Comment")
    rows: list[tuple] = []
    for i, e in enumerate(entries, 1):
        rows.append((str(i), e.path, fmt_number(e.lines), fmt_number(e.code),
                      fmt_number(e.comment)))

    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    print_table(headers, rows, widths)
    print()


def print_project_summary(totals: dict[str, LangTotals], crates: list[CrateStats],
                          ri: RustInsights, dep_info: DepInfo | None) -> None:
    """Print a high-level project summary dashboard."""
    print(section_header("PROJECT SUMMARY"))
    print()

    sum_files = sum(t.files for t in totals.values())
    sum_code = sum(t.code for t in totals.values())
    sum_total = sum(t.total for t in totals.values())
    sum_comment = sum(t.comment for t in totals.values())

    rust_totals = totals.get("Rust")
    rust_code = rust_totals.code if rust_totals else 0
    rust_pct = fmt_pct(rust_code, sum_code)

    print(f"  Total source files      : {fmt_number(sum_files)}")
    print(f"  Total lines             : {fmt_number(sum_total)}")
    print(f"  Lines of code           : {fmt_number(sum_code)}")
    print(f"  Lines of comments       : {fmt_number(sum_comment)}")
    print(f"  Rust as % of code       : {rust_pct}")
    print(f"  Workspace crates        : {len(crates)}")
    if dep_info:
        print(f"  Third-party deps        : {fmt_number(dep_info.third_party_packages)}")
    print(f"  Test functions          : {fmt_number(ri.test_fns)}")
    print(f"  Async functions         : {fmt_number(ri.async_fns)}")
    print(f"  Structs + Enums         : {fmt_number(ri.struct_enum_defs)}")
    print(f"  Traits                  : {fmt_number(ri.trait_defs)}")
    print(f"  unsafe usages           : {fmt_number(ri.unsafe_blocks)}")
    print(f"  Open TODOs              : {fmt_number(ri.todo_count)}")
    print()


# ── entry point ─────────────────────────────────────────────────────────

def main() -> None:
    """Determine the repo root and run the full analytics suite."""
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).resolve()
    else:
        # Default: assume this script lives in <repo>/tools/
        root = Path(__file__).resolve().parent.parent

    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {root}")

    # ── collect all metrics ──────────────────────────────────────────
    totals = collect(root)
    if not totals:
        print("No source files found.")
        sys.exit(0)

    crates = collect_crate_stats(root)
    ri = collect_rust_insights(root)
    dep_info = analyze_dependencies(root)
    largest = find_largest_files(root, top_n=15)

    # ── print all sections ───────────────────────────────────────────
    print_project_summary(totals, crates, ri, dep_info)
    print_language_table(totals)
    print_crate_breakdown(crates)
    print_test_insights(ri, crates)
    print_code_structure(ri)
    print_safety_audit(ri)
    print_dependency_summary(dep_info)
    print_largest_files(largest)


if __name__ == "__main__":
    main()
