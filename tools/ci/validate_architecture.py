#!/usr/bin/env python3
"""Architecture-Code Consistency Validator for NeuronCite.

This script enforces bidirectional consistency between the LaTeX architecture
document (docs/architecture.tex) and the Rust source tree. It parses three
categories of tables from the LaTeX file (crate overview, module structure,
and feature flags), scans the Cargo workspace for actual crate directories,
source files, mod declarations, and feature definitions, then performs seven
validation checks to detect discrepancies in either direction.

The validator is described in Section 20.8 of the architecture document and
runs as the final CI pipeline stage.

Exit codes:
    0 -- All checks pass (no errors, and no warnings unless --strict is active).
    1 -- At least one error-level discrepancy was found (or a warning under --strict).
    2 -- The script cannot parse the LaTeX file or locate the workspace root.

Invocation:
    python tools/validate_architecture.py --tex docs/architecture.tex --root . [--strict] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures for parsed LaTeX content and Rust source tree information
# ---------------------------------------------------------------------------

@dataclass
class CrateSpec:
    """A crate entry extracted from the LaTeX crate overview table (Section 2.1).

    Attributes:
        name: The crate name as it appears in the workspace (e.g., "neuroncite-core").
        crate_type: Either "Library" or "Binary", indicating the crate category.
    """
    name: str
    crate_type: str


@dataclass
class ModuleEntry:
    """A module entry extracted from a LaTeX module structure table (Section 2.3).

    Attributes:
        crate_name: The crate this module belongs to (e.g., "neuroncite-pdf").
        file_path: The relative path under src/ (e.g., "src/extract/mod.rs").
        visibility: Either "public" or "private".
        feature_flag: The Cargo feature flag gating this module, if any.
    """
    crate_name: str
    file_path: str
    visibility: str
    feature_flag: Optional[str] = None


@dataclass
class Diagnostic:
    """A single validation diagnostic (error or warning).

    Attributes:
        level: Either "error" or "warning".
        category: The check category identifier (e.g., "missing_file").
        crate_name: The crate this diagnostic applies to.
        detail: A human-readable description of the discrepancy.
    """
    level: str
    category: str
    crate_name: str
    detail: str


@dataclass
class RustCrateInfo:
    """Information gathered from scanning a Rust crate directory.

    Attributes:
        name: The crate name from its Cargo.toml.
        path: The absolute path to the crate directory.
        source_files: Set of relative paths (e.g., "src/lib.rs") for all .rs files.
        mod_declarations: Dict mapping module name to its visibility ("pub" or "priv").
        feature_gates: Dict mapping module name to the feature flag gating it.
        features: Set of feature flag names defined in [features] of Cargo.toml.
        internal_deps: Set of workspace crate names this crate depends on.
    """
    name: str
    path: Path
    source_files: set[str] = field(default_factory=set)
    mod_declarations: dict[str, str] = field(default_factory=dict)
    feature_gates: dict[str, str] = field(default_factory=dict)
    features: set[str] = field(default_factory=set)
    internal_deps: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# LaTeX parsing routines
# ---------------------------------------------------------------------------

def sanitize_latex_path(raw: str) -> str:
    """Convert a LaTeX-formatted module path to a plain filesystem path.

    Strips LaTeX-specific escapes: \\allowbreak commands, escaped underscores
    (\\_), and leading/trailing whitespace. The resulting string is a plain
    path like "src/extract/pdf_extract.rs".

    Args:
        raw: The raw module path string extracted from a LaTeX table cell.

    Returns:
        A cleaned filesystem-compatible path string.
    """
    # Remove \allowbreak commands that LaTeX uses for line-break hints
    cleaned = raw.replace("\\allowbreak", "")
    # Convert escaped underscores to plain underscores
    cleaned = cleaned.replace("\\_", "_")
    # Collapse any resulting double spaces or stray whitespace
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.strip()


def _join_multiline_latex(table_body: str) -> str:
    """Join lines that belong to the same LaTeX logical construct.

    In the module structure tables, \\multicolumn feature annotations and
    module rows can span multiple lines. This function joins continuation
    lines (lines that do not start a new table row or structural command)
    with the preceding line, producing single logical lines for parsing.

    The joining rules are:
    - A \\midrule, \\toprule, or \\bottomrule line stays standalone.
    - A \\multicolumn line starts a logical line that may continue on the
      next line(s) until the row terminator (\\\\) or next structural command.
    - A src/ line starts a logical line.
    - Any other line is appended to the previous logical line.

    Args:
        table_body: Raw LaTeX table body text.

    Returns:
        The table body with multi-line constructs joined into single lines.
    """
    lines = table_body.split("\n")
    joined: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Lines that start a new logical entry
        is_new_entry = (
            stripped.startswith("\\midrule")
            or stripped.startswith("\\toprule")
            or stripped.startswith("\\bottomrule")
            or stripped.startswith("\\multicolumn")
            or stripped.startswith("src/")
            or stripped.startswith("\\endfirsthead")
            or stripped.startswith("\\endhead")
            or stripped.startswith("\\endfoot")
            or re.match(r"\\normalfont", stripped)
        )

        if is_new_entry or not joined:
            joined.append(stripped)
        else:
            # Continuation of the previous logical line
            joined[-1] = joined[-1] + " " + stripped

    return "\n".join(joined)


def parse_crate_overview(tex_content: str) -> list[CrateSpec]:
    """Extract crate definitions from the crate overview table in Section 2.1.

    The table is identified by its caption "Workspace crates and their
    responsibilities". Two row formats are supported:

    longtable with \\ttfamily column (current):
        neuroncite-xxx & Library & description \\\\

    tabularx with explicit \\texttt (legacy):
        \\texttt{neuroncite-xxx} & Library & description \\\\

    Args:
        tex_content: The full LaTeX document content as a string.

    Returns:
        A list of CrateSpec instances, one per crate listed in the table.
    """
    crates: list[CrateSpec] = []

    # Locate the table by its caption text
    caption_pattern = r"\\caption\{Workspace crates and their responsibilities\.?\}"
    caption_match = re.search(caption_pattern, tex_content)
    if caption_match is None:
        return crates

    # Extract the table environment containing this caption by searching backward
    # for the nearest \begin{table} or \begin{longtable} and forward for \end
    caption_pos = caption_match.start()
    # Search backward for table start
    table_start_pattern = r"\\begin\{(?:table|longtable|tabularx)\}"
    table_starts = list(re.finditer(table_start_pattern, tex_content[:caption_pos]))
    if not table_starts:
        return crates
    table_start = table_starts[-1].start()

    # The table content lies between the table start and the caption
    table_content = tex_content[table_start:caption_pos + 200]

    # Match rows in two formats:
    # 1) \texttt{crate-name} & Type & ... (legacy tabularx)
    # 2) neuroncite-xxx & Type & ...       (longtable with \ttfamily column)
    row_pattern = (
        r"(?:\\texttt\{(neuroncite[\w-]*)\}|(neuroncite[\w-]*))"
        r"\s*&\s*(Library|Binary)\s*&"
    )
    for match in re.finditer(row_pattern, table_content):
        crate_name = (match.group(1) or match.group(2)).strip()
        crate_type = match.group(3).strip()
        crates.append(CrateSpec(name=crate_name, crate_type=crate_type))

    return crates


def parse_module_tables(tex_content: str) -> list[ModuleEntry]:
    """Extract module entries from per-crate module structure tables in Section 2.3.

    Each table is identified by a caption matching:
        "Internal module structure of \\texttt{<crate-name>}"

    Rows have the format:
        src/<path> & <visibility> & <description> \\\\

    Feature flag annotations appear as \\multicolumn rows containing the text
    "compiled under feature \\texttt{<flag>}" (potentially spanning multiple
    LaTeX source lines). Module rows following a feature annotation inherit
    the feature flag until a \\midrule is encountered that is NOT a post-header
    separator for a feature section.

    The LaTeX table structure for feature-gated sections follows this pattern:

        \\midrule                            <- pre-header separator (resets feature)
        \\multicolumn{...feature backend-ort}  <- sets active_feature, sets post_header_pending
        \\midrule                            <- post-header separator (skipped, post_header_pending is True)
        src/ort/mod.rs & private & ...       <- inherits active_feature = "backend-ort"
        ...more module rows...
        \\midrule                            <- pre-header separator for next feature OR terminator
        (next feature multicolumn would go here if additional backends existed)
        ...
        \\midrule                            <- terminator (no multicolumn follows, resets feature)
        src/error.rs & private & ...         <- no feature flag

    The state machine uses a boolean `post_header_pending` flag. When a feature
    \\multicolumn is encountered, this flag is set to True. The immediately
    following \\midrule (post-header separator) consumes this flag and does NOT
    reset the active feature. Subsequent \\midrule lines (section terminators)
    reset the feature to None unless they are followed by another feature
    \\multicolumn, which is detected via lookahead.

    Args:
        tex_content: The full LaTeX document content as a string.

    Returns:
        A list of ModuleEntry instances for all modules across all crates.
    """
    modules: list[ModuleEntry] = []

    # Find all module structure table captions
    caption_pattern = (
        r"\\caption\{Internal module structure of "
        r"\\texttt\{([^}]+)\}"
    )
    caption_matches = list(re.finditer(caption_pattern, tex_content))

    # Pattern for feature flag section headers within \multicolumn rows.
    # Matches text like: "compiled under feature \texttt{backend-ort}"
    feature_section_pattern = (
        r"compiled under feature\s+\\texttt\{([^}]+)\}"
    )

    for cap_match in caption_matches:
        crate_name = cap_match.group(1).replace("\\_", "_").strip()
        caption_pos = cap_match.start()

        # Search backward from the caption for the nearest longtable/table start
        table_start_pattern = r"\\begin\{(?:longtable|table|tabularx)\}"
        preceding = tex_content[:caption_pos]
        table_starts = list(re.finditer(table_start_pattern, preceding))
        if not table_starts:
            continue
        table_start = table_starts[-1].start()

        # The table body is between table start and the caption
        table_body = tex_content[table_start:caption_pos]

        # Join multi-line LaTeX constructs into single logical lines so that
        # feature annotations spanning two lines (e.g., \multicolumn on one
        # line and \texttt{backend-ort} on the next) are handled correctly
        joined_body = _join_multiline_latex(table_body)
        all_lines = joined_body.split("\n")

        # State machine for feature flag tracking across table rows.
        # - active_feature: the feature flag currently in effect, or None.
        # - post_header_pending: True after a feature \multicolumn was just
        #   processed, indicating the next \midrule is the post-header
        #   separator that should be skipped without resetting the feature.
        active_feature: Optional[str] = None
        post_header_pending: bool = False

        idx = 0
        while idx < len(all_lines):
            stripped = all_lines[idx].strip()
            idx += 1

            # Handle \midrule lines. The behavior depends on context:
            # 1. If post_header_pending is True, this \midrule immediately
            #    follows a feature \multicolumn and is the post-header
            #    separator. Consume the flag and skip without resetting.
            # 2. Otherwise, look ahead past any consecutive \midrule lines
            #    to find the next meaningful line. If that line is a feature
            #    \multicolumn, this \midrule is a pre-header separator and
            #    should not reset the feature (the \multicolumn will update
            #    it). If the next line is not a feature \multicolumn, this
            #    \midrule terminates the current feature section and resets
            #    active_feature to None.
            if stripped.startswith("\\midrule"):
                if post_header_pending:
                    # This is the post-header separator for the feature
                    # section that was just declared by a \multicolumn line.
                    # The feature remains active for the module rows that follow.
                    post_header_pending = False
                    continue

                # Look ahead past any consecutive \midrule lines to find the
                # next non-midrule meaningful line
                lookahead = idx
                while lookahead < len(all_lines):
                    next_line = all_lines[lookahead].strip()
                    if next_line.startswith("\\midrule"):
                        lookahead += 1
                        continue
                    break
                else:
                    next_line = ""

                # If the next meaningful line is a feature \multicolumn, this
                # \midrule is a pre-header separator and the feature will be
                # updated by the \multicolumn processing below. If not, this
                # \midrule terminates the current feature section.
                if not re.search(feature_section_pattern, next_line):
                    active_feature = None
                continue

            # Check for feature flag section header in \multicolumn lines.
            # Sets the active feature and marks the post-header \midrule as
            # pending so it will not reset the feature.
            feat_match = re.search(feature_section_pattern, stripped)
            if feat_match:
                active_feature = feat_match.group(1).strip()
                post_header_pending = True
                continue

            # Check for non-feature \multicolumn section headers (e.g.,
            # "Request handlers", "Middleware", "GPU Worker"). These reset
            # the feature flag and are not module rows.
            if re.search(r"\\multicolumn", stripped):
                active_feature = None
                post_header_pending = False
                continue

            # Skip structural commands that are not data rows
            if stripped.startswith(("\\toprule", "\\bottomrule",
                                    "\\endfirsthead", "\\endhead",
                                    "\\endfoot", "\\normalfont")):
                continue

            # Match module path rows: src/... & visibility & description \\
            # The path column may contain LaTeX escapes (\allowbreak, \_)
            row_match = re.match(
                r"\s*(src/[^\s&]+(?:\s*[^\s&]+)*)\s*&\s*(public|private)\s*&",
                stripped,
            )
            if row_match:
                raw_path = row_match.group(1)
                visibility = row_match.group(2).strip()
                file_path = sanitize_latex_path(raw_path)

                modules.append(ModuleEntry(
                    crate_name=crate_name,
                    file_path=file_path,
                    visibility=visibility,
                    feature_flag=active_feature,
                ))

    return modules


def parse_feature_flags(tex_content: str) -> set[str]:
    """Extract documented feature flag names from the feature flag table in Section 13.

    The table is identified by its caption "Feature flag definitions and their effects"
    and contains rows where the first column has \\texttt{<feature-name>}.

    Args:
        tex_content: The full LaTeX document content as a string.

    Returns:
        A set of feature flag name strings (e.g., {"backend-ort"}).
    """
    features: set[str] = set()

    # Locate the feature flag table by caption
    caption_pattern = r"\\caption\{Feature flag definitions and their effects\.?\}"
    caption_match = re.search(caption_pattern, tex_content)
    if caption_match is None:
        return features

    # Find the table body preceding the caption
    caption_pos = caption_match.start()
    table_start_pattern = r"\\begin\{(?:table|longtable|tabularx)\}"
    table_starts = list(re.finditer(table_start_pattern, tex_content[:caption_pos]))
    if not table_starts:
        return features
    table_start = table_starts[-1].start()
    table_body = tex_content[table_start:caption_pos]

    # Extract feature names from \texttt{<name>} entries in the first column of each
    # row. Feature names use lowercase ASCII, digits, and hyphens. The optional
    # parenthetical "(default)" suffix is not part of the name and is excluded by
    # the character class.
    feature_pattern = r"\\texttt\{([\w-]+)\}"
    for match in re.finditer(feature_pattern, table_body):
        features.add(match.group(1))

    return features


def _transitive_closure(deps: dict[str, set[str]]) -> dict[str, set[str]]:
    """Compute the transitive closure of a dependency graph.

    For each crate, collects all crates reachable through any chain of
    dependency edges. Uses iterative BFS per node. The input graph maps
    each crate name to its set of direct dependencies.

    Args:
        deps: Direct dependency edges (crate -> set of dependency crates).

    Returns:
        A dict with the same keys, where each value is the full set of
        transitively reachable dependencies.
    """
    closure: dict[str, set[str]] = {}
    for crate in deps:
        reachable: set[str] = set()
        frontier = list(deps.get(crate, set()))
        while frontier:
            dep = frontier.pop()
            if dep not in reachable:
                reachable.add(dep)
                frontier.extend(deps.get(dep, set()) - reachable)
        closure[crate] = reachable
    return closure


@dataclass
class DependencyGraphInfo:
    """Parsed representation of the TikZ dependency graph.

    Separates the directly drawn edges from metadata flags so that
    the comparison logic can apply transitive closure and universal
    dependency suppression independently per comparison direction.
    """

    direct_edges: dict[str, set[str]]
    """Edges drawn as \\draw[dep] or \\draw[bindep] commands."""

    is_transitive_reduction: bool
    """True when the graph caption/body declares a transitive reduction layout."""

    universal_dep: str | None
    """Crate name declared as a dependency of all library crates via a text
    annotation band (e.g., "All library crates depend on neuroncite-core").
    None when no such annotation exists."""

    library_crates: set[str]
    """Set of crate names represented as library crate nodes in the graph
    (excludes the binary entry point)."""


def parse_dependency_graph(tex_content: str) -> DependencyGraphInfo:
    """Extract the documented crate dependency edges from the TikZ dependency graph.

    The dependency graph in Section 2.2 uses TikZ \\draw[dep] commands to
    represent edges. The convention is: \\draw[dep] (source) -- (target);
    meaning "source depends on target".

    The graph may use a transitive reduction layout where implied edges
    (A -> B -> C, so A -> C is omitted) are not drawn. When the LaTeX
    caption or body contains the phrase "transitive reduction", the
    validator computes the transitive closure of the drawn edges before
    comparing against Cargo.toml dependencies.

    The graph may also contain an annotation band declaring that all
    library crates depend on a specific crate (e.g., neuroncite-core).
    This is detected by searching for "All library crates depend on"
    followed by a \\texttt{crate-name} reference in the graph body.

    Crate names in the TikZ graph use short identifiers (e.g., "pdf", "core",
    "bin") that must be mapped to full crate names.

    Args:
        tex_content: The full LaTeX document content as a string.

    Returns:
        A DependencyGraphInfo with direct edges, layout flags, and
        annotation metadata.
    """
    deps: dict[str, set[str]] = {}

    # Locate the dependency graph TikZ picture
    graph_pattern = r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}"
    graph_match = re.search(graph_pattern, tex_content, re.DOTALL)
    if graph_match is None:
        return DependencyGraphInfo(
            direct_edges={},
            is_transitive_reduction=False,
            universal_dep=None,
            library_crates=set(),
        )

    graph_body = graph_match.group(0)

    # Detect whether the graph uses a transitive reduction layout.
    # The caption text after \end{tikzpicture} is also checked.
    caption_region = tex_content[
        graph_match.start() : graph_match.end() + 500
    ]
    is_transitive_reduction = bool(
        re.search(r"transitive\s+reduction", caption_region, re.IGNORECASE)
    )

    # Build the node-name-to-crate-name mapping from \node declarations
    # Pattern: \node[...] (id) {neuroncite-xxx};  or {neuroncite (bin)}
    node_map: dict[str, str] = {}
    node_pattern = r"\\node\[.*?\]\s*\((\w+)\)\s*(?:\{([^}]+)\}|.*?\{([^}]+)\})"
    for match in re.finditer(node_pattern, graph_body):
        node_id = match.group(1)
        label = (match.group(2) or match.group(3)).strip()
        # Convert TikZ label to crate name
        # Labels are like "neuroncite-core", "neuroncite (bin)"
        if "(bin)" in label:
            crate_name = "neuroncite"
        else:
            crate_name = label.strip()
        node_map[node_id] = crate_name

    # Identify library crate nodes (all except the binary entry point)
    library_crates = {
        name for name in node_map.values()
        if name != "neuroncite"
    }

    # Detect the "All library crates depend on <crate>" annotation.
    # Extracts the crate name from the \texttt{} command following the phrase.
    universal_dep: str | None = None
    universal_match = re.search(
        r"All\s+library\s+crates\b.*?depend\s+.*?on"
        r".*?\\texttt\{([^}]+)\}",
        graph_body,
        re.DOTALL,
    )
    if universal_match:
        universal_dep = universal_match.group(1).strip()

    # Parse \draw[dep] edges. Supported connector syntaxes:
    #   \draw[dep] (source) -- (target);         straight line
    #   \draw[dep] (source) to[...] (target);    curved/bent line
    #   \draw[dep] (source.east) -| (target.north);  orthogonal routing
    #   \draw[dep] (source.east) |- (target.north);  orthogonal routing
    # Node anchors like .east, .north are stripped to extract the node ID.
    edge_pattern = (
        r"\\draw\[(?:bin)?dep[^\]]*\]\s*"
        r"\((\w+)(?:\.\w+)?\)\s*"
        r"(?:--|-\||\|-|\s*to\[.*?\])\s*"
        r"\((\w+)(?:\.\w+)?\)"
    )
    for match in re.finditer(edge_pattern, graph_body):
        source_id = match.group(1)
        target_id = match.group(2)
        source_crate = node_map.get(source_id)
        target_crate = node_map.get(target_id)
        if source_crate and target_crate:
            if source_crate not in deps:
                deps[source_crate] = set()
            deps[source_crate].add(target_crate)

    return DependencyGraphInfo(
        direct_edges=deps,
        is_transitive_reduction=is_transitive_reduction,
        universal_dep=universal_dep,
        library_crates=library_crates,
    )


# ---------------------------------------------------------------------------
# Rust source tree scanning routines
# ---------------------------------------------------------------------------

def parse_workspace_members(cargo_toml_path: Path) -> list[str]:
    """Parse the [workspace.members] list from the root Cargo.toml.

    Uses a simple line-by-line parser since we only need the workspace members
    array and do not require a full TOML parser. Members are expected to be
    quoted strings like "crates/neuroncite-core" inside the members = [...] block.

    Args:
        cargo_toml_path: Absolute path to the root Cargo.toml file.

    Returns:
        A list of member directory paths relative to the workspace root
        (e.g., ["crates/neuroncite-core", "crates/neuroncite-pdf"]).
    """
    members: list[str] = []
    content = cargo_toml_path.read_text(encoding="utf-8")

    # Find the members array within [workspace]
    members_match = re.search(
        r"members\s*=\s*\[(.*?)\]",
        content,
        re.DOTALL,
    )
    if members_match is None:
        return members

    members_block = members_match.group(1)
    # Extract each quoted member string
    for string_match in re.finditer(r'"([^"]+)"', members_block):
        members.append(string_match.group(1))

    return members


def extract_crate_name_from_member(member_path: str) -> str:
    """Derive the crate name from a workspace member path.

    Workspace members are paths like "crates/neuroncite-core". The crate name
    is the last path component (e.g., "neuroncite-core").

    Args:
        member_path: The member path string from Cargo.toml (e.g., "crates/neuroncite-core").

    Returns:
        The crate name string.
    """
    return member_path.rstrip("/").split("/")[-1]


def collect_source_files(crate_dir: Path) -> set[str]:
    """Recursively collect all .rs files under a crate's src/ directory.

    Returns paths relative to the crate directory (e.g., "src/lib.rs",
    "src/extract/mod.rs"). Files in tests/ and benches/ directories are
    excluded, as are build.rs files.

    Args:
        crate_dir: Absolute path to the crate directory.

    Returns:
        A set of relative path strings for all .rs source files.
    """
    src_dir = crate_dir / "src"
    if not src_dir.is_dir():
        return set()

    files: set[str] = set()
    for rs_file in src_dir.rglob("*.rs"):
        rel = rs_file.relative_to(crate_dir)
        # Use forward slashes for cross-platform consistency
        rel_str = str(rel).replace("\\", "/")
        files.add(rel_str)

    return files


def _classify_rust_visibility(vis_prefix: Optional[str]) -> str:
    """Classify a Rust visibility modifier into "pub" or "priv".

    The architecture document uses "public" and "private" to describe module
    visibility. In Rust source code, visibility modifiers map as follows:

    - `pub mod` -> "pub" (publicly visible outside the crate)
    - `pub(crate) mod` -> "priv" (visible within the crate, private externally)
    - `pub(super) mod` -> "priv" (visible within the parent module only)
    - `pub(in path) mod` -> "priv" (restricted visibility)
    - `mod` (no qualifier) -> "priv" (private to the parent module)

    Only an unqualified `pub` without parenthesized restriction counts as
    "public" in the architecture document sense.

    Args:
        vis_prefix: The visibility modifier string captured by the regex,
            or None if the module has no visibility qualifier (plain `mod`).

    Returns:
        "pub" if the module is fully public, "priv" otherwise.
    """
    if vis_prefix is None:
        return "priv"

    # Strip whitespace for comparison
    trimmed = vis_prefix.strip()

    # Only bare "pub" (without parenthesized scope restriction) is "public".
    # Patterns like "pub(crate)", "pub(super)", "pub(in ...)" are private
    # from the perspective of the architecture document.
    if trimmed == "pub":
        return "pub"

    return "priv"


def parse_mod_declarations(lib_rs_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Parse mod declarations from a lib.rs or main.rs file.

    Identifies module declarations and their visibility (pub mod, pub(crate) mod,
    or plain mod), and detects #[cfg(feature = "...")] attributes preceding mod
    statements.

    Visibility classification:
    - `pub mod foo;` -> "pub" (publicly exported)
    - `pub(crate) mod foo;` -> "priv" (crate-internal, private in arch doc)
    - `pub(super) mod foo;` -> "priv"
    - `mod foo;` -> "priv"

    Args:
        lib_rs_path: Absolute path to the lib.rs or main.rs file.

    Returns:
        A tuple of two dicts:
        - mod_visibility: maps module name to "pub" or "priv"
        - feature_gates: maps module name to the feature flag string
    """
    mod_visibility: dict[str, str] = {}
    feature_gates: dict[str, str] = {}

    if not lib_rs_path.is_file():
        return mod_visibility, feature_gates

    content = lib_rs_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Track the most recent #[cfg(feature = "...")] attribute
    pending_feature: Optional[str] = None

    for line in lines:
        stripped = line.strip()

        # Detect cfg(feature) attributes
        cfg_match = re.match(
            r'#\[cfg\(feature\s*=\s*"([^"]+)"\)\]',
            stripped,
        )
        if cfg_match:
            pending_feature = cfg_match.group(1)
            continue

        # Detect module declarations with any visibility modifier.
        # This regex handles all Rust visibility forms:
        #   pub mod foo;
        #   pub(crate) mod foo;
        #   pub(super) mod foo;
        #   pub(in crate::path) mod foo;
        #   mod foo;
        # Group 1 captures the full visibility qualifier (e.g., "pub",
        # "pub(crate)", "pub(super)") or is None for plain `mod`.
        mod_match = re.match(
            r"(pub\s*(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;",
            stripped,
        )
        if mod_match:
            vis_prefix = mod_match.group(1)
            mod_name = mod_match.group(2)
            mod_visibility[mod_name] = _classify_rust_visibility(vis_prefix)
            if pending_feature:
                feature_gates[mod_name] = pending_feature
            pending_feature = None
            continue

        # Reset pending feature if the line is not a mod declaration
        # and not a blank or comment line (attributes apply to the next item)
        if stripped and not stripped.startswith("//") and not stripped.startswith("#["):
            pending_feature = None

    return mod_visibility, feature_gates


def parse_crate_cargo_toml(
    cargo_toml_path: Path,
    workspace_crate_names: set[str],
) -> tuple[set[str], set[str]]:
    """Parse a crate-level Cargo.toml for features and internal dependencies.

    Extracts feature flag names from the [features] section and identifies
    internal workspace dependencies from the [dependencies] section by
    checking against the known workspace crate names.

    Args:
        cargo_toml_path: Absolute path to the crate's Cargo.toml.
        workspace_crate_names: Set of all crate names in the workspace (used to
            distinguish internal from external dependencies).

    Returns:
        A tuple of:
        - features: set of feature flag names defined in [features]
        - internal_deps: set of workspace crate names this crate depends on
    """
    features: set[str] = set()
    internal_deps: set[str] = set()

    if not cargo_toml_path.is_file():
        return features, internal_deps

    content = cargo_toml_path.read_text(encoding="utf-8")

    # Parse [features] section. The section ends at the next [section] header
    # or end of file.
    features_match = re.search(r"\[features\]\s*\n((?:.*\n)*?)\n*(?:\[|\Z)", content)
    if features_match:
        features_block = features_match.group(1)
        # Each feature line: feature-name = [...]
        for feat_match in re.finditer(r"^(\S+)\s*=", features_block, re.MULTILINE):
            feat_name = feat_match.group(1).strip()
            # Skip "default" since it is a Cargo built-in, not a user-facing feature
            if feat_name != "default":
                features.add(feat_name)

    # Parse [dependencies] section for internal workspace crate deps
    deps_match = re.search(r"\[dependencies\]\s*\n((?:.*\n)*?)\n*(?:\[|\Z)", content)
    if deps_match:
        deps_block = deps_match.group(1)
        for dep_match in re.finditer(r"^([\w-]+)\s*=", deps_block, re.MULTILINE):
            dep_name = dep_match.group(1).strip()
            # Replace underscores with hyphens for crate name normalization
            normalized = dep_name.replace("_", "-")
            if normalized in workspace_crate_names:
                internal_deps.add(normalized)

    return features, internal_deps


def scan_rust_workspace(root: Path) -> tuple[list[str], dict[str, RustCrateInfo]]:
    """Scan the entire Rust workspace and collect information about each crate.

    Parses the root Cargo.toml for workspace members, then iterates over each
    crate directory to collect source files, mod declarations, feature gates,
    feature definitions, and internal dependencies.

    Args:
        root: Absolute path to the workspace root directory.

    Returns:
        A tuple of:
        - member_crate_names: list of crate names from workspace members
        - crate_infos: dict mapping crate name to RustCrateInfo
    """
    cargo_toml_path = root / "Cargo.toml"
    members = parse_workspace_members(cargo_toml_path)

    member_crate_names = [extract_crate_name_from_member(m) for m in members]
    workspace_crate_names = set(member_crate_names)

    crate_infos: dict[str, RustCrateInfo] = {}

    for member_path in members:
        crate_dir = root / member_path.replace("/", os.sep)
        crate_name = extract_crate_name_from_member(member_path)

        info = RustCrateInfo(name=crate_name, path=crate_dir)

        # Collect .rs source files
        info.source_files = collect_source_files(crate_dir)

        # Parse mod declarations from lib.rs or main.rs
        lib_rs = crate_dir / "src" / "lib.rs"
        main_rs = crate_dir / "src" / "main.rs"
        entry_file = lib_rs if lib_rs.is_file() else main_rs

        mod_vis, feat_gates = parse_mod_declarations(entry_file)
        info.mod_declarations = mod_vis
        info.feature_gates = feat_gates

        # Also parse mod declarations from subdirectory mod.rs files for
        # deeper module hierarchies (e.g., src/handlers/mod.rs, src/repo/mod.rs)
        for src_file in sorted(info.source_files):
            if src_file.endswith("/mod.rs") and src_file != "src/mod.rs":
                sub_mod_path = crate_dir / src_file.replace("/", os.sep)
                sub_vis, sub_gates = parse_mod_declarations(sub_mod_path)
                # Prefix sub-module names with their parent directory
                parent_dir = src_file.replace("/mod.rs", "").replace("src/", "")
                for mod_name, visibility in sub_vis.items():
                    prefixed = f"{parent_dir}/{mod_name}"
                    info.mod_declarations[prefixed] = visibility
                for mod_name, gate in sub_gates.items():
                    prefixed = f"{parent_dir}/{mod_name}"
                    info.feature_gates[prefixed] = gate

        # Parse crate Cargo.toml for features and internal dependencies
        crate_cargo = crate_dir / "Cargo.toml"
        info.features, info.internal_deps = parse_crate_cargo_toml(
            crate_cargo, workspace_crate_names,
        )

        crate_infos[crate_name] = info

    return member_crate_names, crate_infos


# ---------------------------------------------------------------------------
# Module path to mod-declaration name mapping utilities
# ---------------------------------------------------------------------------

def file_path_to_mod_name(file_path: str) -> Optional[str]:
    """Convert a source file path to its corresponding mod declaration name.

    Maps paths like "src/types.rs" -> "types", "src/ort/mod.rs" -> "ort",
    "src/ort/embedder.rs" -> "ort/embedder". Returns None for entry point
    files (lib.rs, main.rs) that do not correspond to mod declarations.

    Args:
        file_path: Relative path string (e.g., "src/extract/mod.rs").

    Returns:
        The mod declaration name, or None for lib.rs/main.rs.
    """
    if file_path.startswith("src/"):
        inner = file_path[4:]
    else:
        return None

    if inner in ("lib.rs", "main.rs"):
        return None

    if inner.endswith("/mod.rs"):
        return inner[:-len("/mod.rs")]

    if inner.endswith(".rs"):
        return inner[:-3]

    return None


def mod_name_to_top_level(mod_name: str) -> str:
    """Extract the top-level module name from a potentially nested mod path.

    For example, "ort/embedder" -> "ort", "types" -> "types",
    "handlers/health" -> "handlers".

    Args:
        mod_name: The module name, potentially containing slashes for nesting.

    Returns:
        The top-level (first component) module name.
    """
    return mod_name.split("/")[0]


# ---------------------------------------------------------------------------
# Validation check implementations
# ---------------------------------------------------------------------------

def check_workspace_members(
    latex_crates: list[CrateSpec],
    cargo_crate_names: list[str],
) -> list[Diagnostic]:
    """Check 1: Workspace member mismatch (error severity).

    Verifies that every crate in the LaTeX crate overview table exists in
    the Cargo.toml workspace members, and vice versa.
    """
    diagnostics: list[Diagnostic] = []
    latex_names = {c.name for c in latex_crates}
    cargo_names = set(cargo_crate_names)

    for name in sorted(latex_names - cargo_names):
        diagnostics.append(Diagnostic(
            level="error",
            category="workspace_mismatch",
            crate_name=name,
            detail="crate listed in architecture document but absent from Cargo.toml [workspace.members]",
        ))

    for name in sorted(cargo_names - latex_names):
        diagnostics.append(Diagnostic(
            level="error",
            category="workspace_mismatch",
            crate_name=name,
            detail="crate in Cargo.toml [workspace.members] but absent from architecture document",
        ))

    return diagnostics


def check_missing_source_files(
    latex_modules: list[ModuleEntry],
    crate_infos: dict[str, RustCrateInfo],
) -> list[Diagnostic]:
    """Check 2: Missing source file (error severity).

    Verifies that every module path listed in the LaTeX document corresponds
    to an actual .rs file on disk.
    """
    diagnostics: list[Diagnostic] = []

    for entry in latex_modules:
        crate_info = crate_infos.get(entry.crate_name)
        if crate_info is None:
            continue

        if entry.file_path not in crate_info.source_files:
            diagnostics.append(Diagnostic(
                level="error",
                category="missing_file",
                crate_name=entry.crate_name,
                detail=f"{entry.file_path} listed in docs but not found on disk",
            ))

    return diagnostics


def check_undocumented_source_files(
    latex_modules: list[ModuleEntry],
    crate_infos: dict[str, RustCrateInfo],
) -> list[Diagnostic]:
    """Check 3: Undocumented source file (error severity).

    Verifies that every .rs file in a crate's src/ directory is documented
    in the LaTeX module structure table. Files in tests/, benches/, and
    build.rs are excluded. Entry point files for binary crates without a
    module structure table are also excluded.
    """
    diagnostics: list[Diagnostic] = []

    documented: dict[str, set[str]] = {}
    for entry in latex_modules:
        if entry.crate_name not in documented:
            documented[entry.crate_name] = set()
        documented[entry.crate_name].add(entry.file_path)

    crates_with_module_table = set(documented.keys())

    for crate_name, info in sorted(crate_infos.items()):
        crate_documented = documented.get(crate_name, set())
        has_module_table = crate_name in crates_with_module_table

        for file_path in sorted(info.source_files):
            if _is_excluded_file(file_path):
                continue
            if not has_module_table and _is_entry_point(file_path):
                continue
            if file_path not in crate_documented:
                diagnostics.append(Diagnostic(
                    level="error",
                    category="undocumented_file",
                    crate_name=crate_name,
                    detail=f"{file_path} exists on disk but not in architecture document",
                ))

    return diagnostics


def _is_excluded_file(file_path: str) -> bool:
    """Determine whether a source file should be excluded from documentation checks.

    Files in tests/, benches/ directories and build.rs are excluded because
    they are not part of the documented module structure.
    """
    parts = file_path.replace("\\", "/").split("/")
    if "tests" in parts or "benches" in parts:
        return True
    if file_path.endswith("build.rs"):
        return True
    return False


def _is_entry_point(file_path: str) -> bool:
    """Determine whether a source file is a crate entry point (lib.rs or main.rs)."""
    return file_path in ("src/lib.rs", "src/main.rs")


def check_visibility_mismatch(
    latex_modules: list[ModuleEntry],
    crate_infos: dict[str, RustCrateInfo],
) -> list[Diagnostic]:
    """Check 4: Visibility mismatch (error severity).

    Verifies that the pub/private visibility declared in the LaTeX document
    matches the actual mod declaration in the crate's lib.rs or parent mod.rs.
    """
    diagnostics: list[Diagnostic] = []

    for entry in latex_modules:
        crate_info = crate_infos.get(entry.crate_name)
        if crate_info is None:
            continue

        mod_name = file_path_to_mod_name(entry.file_path)
        if mod_name is None:
            continue

        top_level = mod_name_to_top_level(mod_name)

        actual_vis = crate_info.mod_declarations.get(mod_name)
        if actual_vis is None:
            actual_vis = crate_info.mod_declarations.get(top_level)

        if actual_vis is None:
            continue

        expected_vis = "pub" if entry.visibility == "public" else "priv"

        if mod_name == top_level:
            if actual_vis != expected_vis:
                label_actual = "pub mod" if actual_vis == "pub" else "mod (private)"
                diagnostics.append(Diagnostic(
                    level="error",
                    category="visibility_mismatch",
                    crate_name=entry.crate_name,
                    detail=(
                        f"{entry.file_path} declared {entry.visibility} in docs "
                        f"but {label_actual} in lib.rs"
                    ),
                ))
        else:
            nested_vis = crate_info.mod_declarations.get(mod_name)
            if nested_vis is not None and nested_vis != expected_vis:
                label_actual = "pub mod" if nested_vis == "pub" else "mod (private)"
                parent_mod_file = f"src/{'/'.join(mod_name.split('/')[:-1])}/mod.rs"
                diagnostics.append(Diagnostic(
                    level="error",
                    category="visibility_mismatch",
                    crate_name=entry.crate_name,
                    detail=(
                        f"{entry.file_path} declared {entry.visibility} in docs "
                        f"but {label_actual} in {parent_mod_file}"
                    ),
                ))

    return diagnostics


def check_missing_feature_gates(
    latex_modules: list[ModuleEntry],
    crate_infos: dict[str, RustCrateInfo],
) -> list[Diagnostic]:
    """Check 5: Missing feature gate (error severity).

    Verifies that modules listed under a feature flag section in the LaTeX
    document have a corresponding #[cfg(feature = "...")] attribute on their
    mod declaration in lib.rs.
    """
    diagnostics: list[Diagnostic] = []

    for entry in latex_modules:
        if entry.feature_flag is None:
            continue

        crate_info = crate_infos.get(entry.crate_name)
        if crate_info is None:
            continue

        mod_name = file_path_to_mod_name(entry.file_path)
        if mod_name is None:
            continue

        top_level = mod_name_to_top_level(mod_name)
        actual_gate = crate_info.feature_gates.get(top_level)

        if actual_gate is None:
            diagnostics.append(Diagnostic(
                level="error",
                category="missing_feature_gate",
                crate_name=entry.crate_name,
                detail=(
                    f"{entry.file_path} listed under feature \"{entry.feature_flag}\" "
                    f"in docs but no #[cfg(feature)] on mod declaration in lib.rs"
                ),
            ))
        elif actual_gate != entry.feature_flag:
            diagnostics.append(Diagnostic(
                level="error",
                category="missing_feature_gate",
                crate_name=entry.crate_name,
                detail=(
                    f"{entry.file_path} listed under feature \"{entry.feature_flag}\" "
                    f"in docs but gated by \"{actual_gate}\" in lib.rs"
                ),
            ))

    return diagnostics


def check_undocumented_features(
    latex_features: set[str],
    crate_infos: dict[str, RustCrateInfo],
) -> list[Diagnostic]:
    """Check 6: Undocumented feature flag (warning severity).

    Verifies that every feature flag defined in a crate's Cargo.toml [features]
    section is described in the architecture document's feature flag table.
    """
    diagnostics: list[Diagnostic] = []

    for crate_name, info in sorted(crate_infos.items()):
        for feature in sorted(info.features):
            if feature not in latex_features:
                diagnostics.append(Diagnostic(
                    level="warning",
                    category="undocumented_feature",
                    crate_name=crate_name,
                    detail=f"feature \"{feature}\" not described in architecture document",
                ))

    return diagnostics


def check_dependency_graph(
    graph_info: DependencyGraphInfo,
    crate_infos: dict[str, RustCrateInfo],
) -> list[Diagnostic]:
    """Check 7: Dependency graph deviation (warning severity).

    Compares the crate-to-crate dependency edges documented in the LaTeX
    TikZ dependency graph against the actual dependencies declared in each
    crate's Cargo.toml. Reports mismatches in both directions using
    direction-specific logic:

    Forward check ("in document but not in Cargo.toml"):
        Uses only the directly drawn edges. A drawn edge A -> B means
        crate A should have B in its Cargo.toml [dependencies].

    Reverse check ("in Cargo.toml but not in document"):
        When the graph is a transitive reduction, computes the transitive
        closure of the drawn edges. A Cargo.toml dep A -> C is considered
        documented if there exists a path A -> B -> C in the drawn edges.
        Additionally, when a universal dependency annotation exists (e.g.,
        "All library crates depend on neuroncite-core"), that dependency
        is suppressed from the reverse check for all library crate nodes.

    Args:
        graph_info: Parsed dependency graph with direct edges and metadata.
        crate_infos: Parsed Cargo.toml metadata per workspace crate.
    """
    diagnostics: list[Diagnostic] = []
    direct = graph_info.direct_edges

    # For the reverse check, compute effective reachable deps.
    # Transitive reduction: use transitive closure so that A -> B -> C
    # covers A -> C.
    if graph_info.is_transitive_reduction:
        reachable = _transitive_closure(direct)
    else:
        reachable = direct

    # When the graph declares a universal dependency (e.g., neuroncite-core),
    # add it to the reachable set for every crate node. The annotation states
    # that all library crates depend on the universal dep, and since the
    # binary transitively depends on library crates, it reaches the universal
    # dep as well. The universal dep crate itself (neuroncite-core) is excluded
    # because it does not depend on itself.
    if graph_info.universal_dep:
        all_graph_crates = graph_info.library_crates | {
            name for name in direct
        }
        for crate_name in all_graph_crates:
            if crate_name == graph_info.universal_dep:
                continue
            if crate_name not in reachable:
                reachable[crate_name] = set()
            reachable[crate_name].add(graph_info.universal_dep)

    all_crate_names = set(crate_infos.keys())

    for crate_name in sorted(all_crate_names):
        info = crate_infos.get(crate_name)
        if info is None:
            continue

        actual_deps = info.internal_deps
        drawn_deps = direct.get(crate_name, set())
        reachable_deps = reachable.get(crate_name, set())

        # Forward check: drawn edge not in Cargo.toml
        for dep in sorted(drawn_deps - actual_deps):
            diagnostics.append(Diagnostic(
                level="warning",
                category="dependency_deviation",
                crate_name=crate_name,
                detail=(
                    f"depends on {dep} in architecture document "
                    f"but not in Cargo.toml [dependencies]"
                ),
            ))

        # Reverse check: Cargo.toml dep not reachable in graph
        for dep in sorted(actual_deps - reachable_deps):
            diagnostics.append(Diagnostic(
                level="warning",
                category="dependency_deviation",
                crate_name=crate_name,
                detail=(
                    f"depends on {dep} in Cargo.toml but not depicted "
                    f"in architecture document dependency graph"
                ),
            ))

    return diagnostics


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_human_readable(diagnostics: list[Diagnostic]) -> str:
    """Format a list of diagnostics as human-readable text lines.

    Each line has the format:
        [ERROR|WARN] <category>: <crate-name> -- <detail>

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

        lines.append(f"{prefix} {diag.category}: {diag.crate_name} -- {diag.detail}")

    lines.append(f"SUMMARY: {error_count} error(s), {warning_count} warning(s)")
    return "\n".join(lines)


def format_json(diagnostics: list[Diagnostic]) -> str:
    """Format a list of diagnostics as a JSON string.

    The JSON object has three keys: "errors", "warnings", and "summary".
    """
    errors = []
    warnings = []

    for diag in diagnostics:
        entry = {
            "category": diag.category,
            "crate": diag.crate_name,
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

def run_validation(tex_path: Path, root_path: Path) -> list[Diagnostic]:
    """Execute all seven validation checks and return the combined diagnostics.

    Parses the LaTeX document for crate specs, module entries, feature flags,
    and dependency edges. Scans the Rust workspace for actual source files,
    mod declarations, and Cargo.toml metadata. Then runs each check.
    """
    tex_content = tex_path.read_text(encoding="utf-8")

    latex_crates = parse_crate_overview(tex_content)
    latex_modules = parse_module_tables(tex_content)
    latex_features = parse_feature_flags(tex_content)
    graph_info = parse_dependency_graph(tex_content)

    cargo_crate_names, crate_infos = scan_rust_workspace(root_path)

    all_diagnostics: list[Diagnostic] = []

    all_diagnostics.extend(check_workspace_members(latex_crates, cargo_crate_names))
    all_diagnostics.extend(check_missing_source_files(latex_modules, crate_infos))
    all_diagnostics.extend(check_undocumented_source_files(latex_modules, crate_infos))
    all_diagnostics.extend(check_visibility_mismatch(latex_modules, crate_infos))
    all_diagnostics.extend(check_missing_feature_gates(latex_modules, crate_infos))
    all_diagnostics.extend(check_undocumented_features(latex_features, crate_infos))
    all_diagnostics.extend(check_dependency_graph(graph_info, crate_infos))

    return all_diagnostics


def main() -> int:
    """Parse command-line arguments, run validation, and produce output.

    Returns:
        Exit code: 0 if all checks pass, 1 if discrepancies found, 2 if
        the script cannot parse the LaTeX file or locate the workspace root.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Architecture-Code Consistency Validator for NeuronCite. "
            "Enforces bidirectional consistency between the LaTeX architecture "
            "document and the Rust source tree."
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
        help="Treat warnings as errors (used in CI to block the pipeline on any discrepancy)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable text",
    )
    args = parser.parse_args()

    tex_path: Path = args.tex.resolve()
    root_path: Path = args.root.resolve()

    if not tex_path.is_file():
        print(f"Error: LaTeX file not found: {tex_path}", file=sys.stderr)
        return 2

    if not root_path.is_dir():
        print(f"Error: workspace root directory not found: {root_path}", file=sys.stderr)
        return 2

    cargo_toml = root_path / "Cargo.toml"
    if not cargo_toml.is_file():
        print(f"Error: Cargo.toml not found in workspace root: {cargo_toml}", file=sys.stderr)
        return 2

    try:
        diagnostics = run_validation(tex_path, root_path)
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
