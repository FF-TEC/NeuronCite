"""
Generates visual architecture diagrams for the NeuronCite workspace.

The crate dependency graph (diagram 1) and module file tree (diagram 5) are
fully automatic and work with any Rust workspace without modification.
Diagrams 2-4 and 6-7 contain project-specific content for NeuronCite.

Outputs (saved to docs/diagrams/):
  1. crate_dependencies.png  - Auto-generated inter-crate dependency graph
  2. architecture_layers.png - Layered architecture overview (project-specific)
  3. type_map.png            - Domain type relationships (project-specific)
  4. api_endpoint_map.png    - REST API endpoint structure (project-specific)
  5. module_tree.png         - Auto-generated source file tree per crate
  6. error_propagation.png   - Error type wrapping chain (project-specific)
  7. request_lifecycle.png   - HTTP request flow through layers (project-specific)

CLI flags:
  --curved-lines     Use curved edges instead of the default orthogonal (right-angle) lines.
  --no-optimize      Disable extra Graphviz layout iterations (faster but messier).
  --auto-only        Skip project-specific diagrams, generate only auto-detected ones.

Requires:
  - Graphviz CLI (dot) installed and on PATH
  - matplotlib (pip install matplotlib)
"""

import subprocess
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

# Output goes to docs/diagrams/ relative to the workspace root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs" / "diagrams"

# When True, all diagrams use only straight orthogonal lines instead of curves.
# Enabled by default; pass --curved-lines to disable.
STRAIGHT_LINES = "--curved-lines" not in sys.argv

# When True, Graphviz spends more iterations minimizing edge crossings and
# optimizing node placement. Produces cleaner layouts at the cost of compute time.
# Enabled by default; pass --no-optimize to disable.
OPTIMIZE = "--no-optimize" not in sys.argv

# When True, only auto-detected diagrams (crate deps, module tree) are generated.
AUTO_ONLY = "--auto-only" in sys.argv

# Palette of fill colors cycled through for auto-generated node coloring.
COLOR_PALETTE = [
    "#E8F5E9", "#E3F2FD", "#FFF3E0", "#FCE4EC", "#F3E5F5",
    "#E0F7FA", "#FFF9C4", "#FFECB3", "#E8EAF6", "#F1F8E9",
]


# ---------------------------------------------------------------------------
# Cargo.toml parsing (no toml library required -- simple regex extraction)
# ---------------------------------------------------------------------------

def parse_workspace_members(cargo_toml_path: Path) -> list[str]:
    """
    Extract workspace member paths from a root Cargo.toml.
    Returns a list of relative directory paths like ["crates/foo", "crates/bar"].
    """
    text = cargo_toml_path.read_text(encoding="utf-8")
    members_match = re.search(r"members\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if not members_match:
        return []
    block = members_match.group(1)
    return re.findall(r'"([^"]+)"', block)


def parse_crate_name(cargo_toml_path: Path) -> Optional[str]:
    """Extract the [package] name from a crate-level Cargo.toml."""
    text = cargo_toml_path.read_text(encoding="utf-8")
    match = re.search(r'^\s*name\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def parse_internal_deps(cargo_toml_path: Path, all_crate_names: set[str]) -> list[str]:
    """
    Extract dependency names from a crate-level Cargo.toml that are also
    workspace members (internal dependencies). Handles both inline and
    table-style dependency declarations, plus workspace = true references.
    """
    text = cargo_toml_path.read_text(encoding="utf-8")
    deps = set()

    # Match lines like: some-crate = { path = "..." } or some-crate.workspace = true
    # or [dependencies.some-crate] sections
    for name in all_crate_names:
        # Pattern: name = ... (anywhere in [dependencies] or [dev-dependencies])
        escaped = re.escape(name)
        if re.search(rf"(?:^|\n)\s*{escaped}\s*=", text):
            deps.add(name)
        # Pattern: [dependencies.name] or [dev-dependencies.name]
        if re.search(rf"\[.*dependencies\.{escaped}\]", text):
            deps.add(name)

    return sorted(deps)


def discover_workspace(root: Path) -> dict[str, list[str]]:
    """
    Auto-discover all workspace crates and their internal dependencies.
    Returns a dict mapping crate_name -> [list of internal dependency names].
    """
    root_cargo = root / "Cargo.toml"
    if not root_cargo.exists():
        return {}

    member_paths = parse_workspace_members(root_cargo)
    if not member_paths:
        return {}

    # First pass: collect all crate names
    crate_info: dict[str, Path] = {}
    for member_rel in member_paths:
        member_cargo = root / member_rel / "Cargo.toml"
        if member_cargo.exists():
            name = parse_crate_name(member_cargo)
            if name:
                crate_info[name] = member_cargo

    all_names = set(crate_info.keys())

    # Second pass: resolve internal dependencies
    result: dict[str, list[str]] = {}
    for name, cargo_path in crate_info.items():
        result[name] = parse_internal_deps(cargo_path, all_names - {name})

    return result


# ---------------------------------------------------------------------------
# Source file tree discovery
# ---------------------------------------------------------------------------

def discover_source_tree(root: Path) -> dict[str, list[str]]:
    """
    For each workspace crate, collect all .rs source file paths relative to the
    crate directory. Returns crate_name -> sorted list of relative paths.
    """
    member_paths = parse_workspace_members(root / "Cargo.toml")
    result: dict[str, list[str]] = {}

    for member_rel in member_paths:
        member_dir = root / member_rel
        cargo_path = member_dir / "Cargo.toml"
        if not cargo_path.exists():
            continue
        name = parse_crate_name(cargo_path)
        if not name:
            continue

        rs_files = sorted(
            str(p.relative_to(member_dir)).replace("\\", "/")
            for p in member_dir.rglob("*.rs")
            if "target" not in p.parts
        )
        if rs_files:
            result[name] = rs_files

    return result


# ---------------------------------------------------------------------------
# DOT rendering helpers
# ---------------------------------------------------------------------------

def _apply_dot_options(dot_source: str) -> str:
    """
    Modify DOT source based on active CLI flags before rendering.

    STRAIGHT_LINES: Insert splines=ortho and convert edge label= to xlabel=
    since ortho mode does not support inline edge labels.

    OPTIMIZE: Insert layout tuning parameters that increase the number of
    iterations the dot engine uses for crossing minimization and node placement.
    - mclimit=200:     Run crossing minimization 200x longer than default.
    - nslimit=200:     Run network simplex (ranking) 200x longer.
    - nslimit1=200:    Run network simplex (positioning) 200x longer.
    - remincross=true: Re-run crossing minimization a second time.
    - searchsize=2000: Broader search in network simplex (default 30).
    """
    injections = []

    if OPTIMIZE:
        injections.append(
            "    mclimit=200;\n"
            "    nslimit=200;\n"
            "    nslimit1=200;\n"
            "    remincross=true;\n"
            "    searchsize=2000;"
        )

    if STRAIGHT_LINES:
        injections.append("    splines=ortho;")

    if injections:
        block = "\n".join(injections)
        dot_source = dot_source.replace("{\n", "{\n" + block + "\n", 1)

    if STRAIGHT_LINES:
        dot_source = dot_source.replace("[label=", "[xlabel=")
        # Restore node labels that were incorrectly converted to xlabel.
        # Node definitions contain fillcolor or shape= attributes.
        lines = dot_source.split("\n")
        fixed = []
        for line in lines:
            stripped = line.strip()
            if ("fillcolor=" in stripped or "shape=" in stripped) and "xlabel=" in stripped:
                line = line.replace("xlabel=", "label=")
            fixed.append(line)
        dot_source = "\n".join(fixed)

    return dot_source


def write_dot_and_render(dot_source: str, output_name: str,
                         override_splines: str | None = None) -> None:
    """Write a DOT file and render it to PNG via the Graphviz dot command."""
    dot_source = _apply_dot_options(dot_source)
    if override_splines:
        # Replace the injected splines=ortho with a per-diagram override
        dot_source = dot_source.replace("splines=ortho;", f"splines={override_splines};")
    dot_path = DOCS_DIR / f"{output_name}.dot"
    png_path = DOCS_DIR / f"{output_name}.png"

    dot_path.write_text(dot_source, encoding="utf-8")

    subprocess.run(
        ["dot", "-Tpng", "-Gdpi=200", str(dot_path), "-o", str(png_path)],
        check=True,
    )
    dot_path.unlink()
    print(f"  -> {png_path}")


def _sanitize_id(name: str) -> str:
    """Convert a crate name like 'neuroncite-core' to a valid DOT node id."""
    return name.replace("-", "_")


# ---------------------------------------------------------------------------
# Diagram 1: Auto-generated crate dependency graph
# ---------------------------------------------------------------------------

def generate_crate_dependency_graph() -> None:
    """
    Auto-detect workspace crates and their internal dependencies from Cargo.toml
    files, then render the dependency graph. Works with any Rust workspace.
    """
    workspace = discover_workspace(PROJECT_ROOT)
    if not workspace:
        print("  [skip] No workspace members found")
        return

    project_title = PROJECT_ROOT.name

    lines = [
        "digraph crate_dependencies {",
        '    rankdir=BT;',
        '    fontname="Helvetica";',
        '    node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11, margin="0.25,0.12"];',
        '    edge [color="#555555", arrowsize=0.7];',
        '    bgcolor="white";',
        f'    label="{project_title} — Crate Dependency Graph";',
        '    labelloc=t;',
        '    fontsize=16;',
        '    nodesep=0.6;',
        '    ranksep=0.8;',
        '',
    ]

    # Sort crates by dependency count (fewest deps first = bottom of graph)
    sorted_crates = sorted(workspace.keys(), key=lambda c: len(workspace[c]))

    # Assign colors from palette
    for i, crate_name in enumerate(sorted_crates):
        node_id = _sanitize_id(crate_name)
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        lines.append(f'    {node_id} [label="{crate_name}", fillcolor="{color}"];')

    lines.append('')

    # Edges
    for crate_name, deps in workspace.items():
        src = _sanitize_id(crate_name)
        for dep in deps:
            dst = _sanitize_id(dep)
            lines.append(f'    {src} -> {dst};')

    lines.append('}')
    write_dot_and_render("\n".join(lines), "crate_dependencies")


# ---------------------------------------------------------------------------
# Diagram 5: Auto-generated module / file tree
# ---------------------------------------------------------------------------

def generate_module_tree() -> None:
    """
    Auto-detect all .rs source files per crate and render a file tree diagram.
    Works with any Rust workspace.
    """
    tree = discover_source_tree(PROJECT_ROOT)
    if not tree:
        print("  [skip] No source files found")
        return

    project_title = PROJECT_ROOT.name

    # Use matplotlib for a grid-based file tree to avoid Graphviz layout issues
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    sorted_crates = sorted(tree.keys())
    n = len(sorted_crates)
    cols = 4
    rows = (n + cols - 1) // cols

    cell_w = 3.5
    cell_h_base = 0.8  # header height
    line_h = 0.16      # per file line

    # Compute max cell height for uniform sizing
    max_files = 15
    max_cell_h = cell_h_base + max_files * line_h + 0.3

    fig_w = cols * cell_w + 1.0
    fig_h = rows * (max_cell_h + 0.4) + 1.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(fig_w / 2, fig_h - 0.5, f"{project_title} — Source File Tree",
            ha="center", va="center", fontsize=14, fontweight="bold",
            fontfamily="Helvetica")

    for idx, crate_name in enumerate(sorted_crates):
        col = idx % cols
        row = idx // cols
        x = 0.5 + col * cell_w
        y = fig_h - 1.3 - row * (max_cell_h + 0.4)

        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        files = tree[crate_name]
        src_files = [f for f in files if f.startswith("src/")]
        if not src_files:
            src_files = files

        display_files = src_files[:max_files]
        n_files = len(src_files)
        cell_h = cell_h_base + len(display_files) * line_h + 0.3
        if n_files > max_files:
            cell_h += line_h

        rect = mpatches.FancyBboxPatch(
            (x, y - cell_h), cell_w - 0.3, cell_h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="#666666", linewidth=0.8,
        )
        ax.add_patch(rect)

        # Crate name header
        ax.text(x + 0.15, y - 0.2, f"{crate_name}  ({n_files} files)",
                fontsize=8, fontweight="bold", va="top", fontfamily="Helvetica")

        # File list
        for fi, fname in enumerate(display_files):
            ax.text(x + 0.2, y - 0.5 - fi * line_h, fname,
                    fontsize=6, va="top", fontfamily="Helvetica", color="#333333")

        if n_files > max_files:
            ax.text(x + 0.2, y - 0.5 - len(display_files) * line_h,
                    f"... +{n_files - max_files} more",
                    fontsize=6, va="top", fontfamily="Helvetica", color="#888888",
                    style="italic")

    out = DOCS_DIR / "module_tree.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Diagram 6: Error propagation chain (project-specific)
# ---------------------------------------------------------------------------

def generate_error_propagation() -> None:
    """
    Render the error type wrapping and mapping chain across crate boundaries.
    Shows how subsystem errors propagate up through NeuronCiteError.
    """
    dot = textwrap.dedent("""\
    digraph error_propagation {
        rankdir=LR;
        fontname="Helvetica";
        node [fontname="Helvetica", fontsize=10, shape=record, style=filled, margin="0.15,0.07"];
        edge [arrowsize=0.6, fontsize=9, fontname="Helvetica"];
        bgcolor="white";
        label="NeuronCite — Error Propagation Chain";
        labelloc=t;
        fontsize=14;
        nodesep=0.4;
        ranksep=1.0;

        // Source errors (external) — leftmost column
        { rank=same; IoError; RusqliteError; OrtError; }
        IoError [fillcolor="#E0E0E0",
            label="{std::io::Error\\n(external)}"];
        RusqliteError [fillcolor="#E0E0E0",
            label="{rusqlite::Error\\n(external)}"];
        OrtError [fillcolor="#E0E0E0",
            label="{ort::Error\\n(external)}"];

        // Subsystem errors — second column
        { rank=same; PdfError; ChunkError; EmbedError; StoreError; SearchError; ApiError; }
        PdfError [fillcolor="#FFCDD2",
            label="{PdfError\\n(neuroncite-pdf)|Extraction\\lPageAccess\\lOcrFailure}"];
        ChunkError [fillcolor="#FFCDD2",
            label="{ChunkError\\n(neuroncite-chunk)|StrategyValidation\\lWindowSize\\lOverlap}"];
        EmbedError [fillcolor="#BBDEFB",
            label="{EmbedError\\n(neuroncite-embed)|ModelLoad\\lTokenization\\lComputation}"];
        StoreError [fillcolor="#BBDEFB",
            label="{StoreError\\n(neuroncite-store)|Schema\\lConnection\\lTransaction}"];
        SearchError [fillcolor="#FFE0B2",
            label="{SearchError\\n(neuroncite-search)|VectorSearch\\lFusion\\lDedup\\lReranking}"];
        ApiError [fillcolor="#F8BBD0",
            label="{ApiError\\n(neuroncite-api)|Http\\lSerialization\\lHandler}"];

        // Central error — third column
        NeuronCiteError [fillcolor="#FFF9C4",
            label="{NeuronCiteError\\n(neuroncite-core)|Pdf(String)\\lChunk(String)\\lEmbed(String)\\lStore(String)\\lSearch(String)\\lApi(String)\\lGui(String)\\lConfig(String)\\lIo(String)\\lInvalidArgument(String)}"];

        // Output — rightmost column
        { rank=same; HttpResponse; JsonRpcError; }
        HttpResponse [fillcolor="#C8E6C9",
            label="{HTTP Response|status: u16\\lbody: JSON\\lerror: String}"];
        JsonRpcError [fillcolor="#C8E6C9",
            label="{JSON-RPC Error|code: i32\\lmessage: String\\ldata: Option}"];

        // Wrapping edges
        RusqliteError -> StoreError [label="wraps"];
        OrtError -> EmbedError [label="wraps"];
        IoError -> NeuronCiteError [label="From"];

        PdfError -> NeuronCiteError [label="From"];
        ChunkError -> NeuronCiteError [label="From"];
        EmbedError -> NeuronCiteError [label="From"];
        StoreError -> NeuronCiteError [label="From"];
        SearchError -> NeuronCiteError [label="From"];
        ApiError -> NeuronCiteError [label="From"];

        NeuronCiteError -> HttpResponse [label="IntoResponse"];
        NeuronCiteError -> JsonRpcError [label="to_json_rpc"];
    }
    """)
    write_dot_and_render(dot, "error_propagation")


# ---------------------------------------------------------------------------
# Diagram 7: Request lifecycle (project-specific)
# ---------------------------------------------------------------------------

def generate_request_lifecycle() -> None:
    """
    Render the lifecycle of an HTTP request through all architectural layers.
    Shows the call chain from client to database and back.
    """
    dot = textwrap.dedent("""\
    digraph request_lifecycle {
        rankdir=TB;
        fontname="Helvetica";
        node [fontname="Helvetica", fontsize=10, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, arrowsize=0.6];
        bgcolor="white";
        label="NeuronCite — Request Lifecycle";
        labelloc=t;
        fontsize=14;
        nodesep=0.8;
        ranksep=0.7;

        // Actors — top row
        { rank=same; browser; mcp_client; cli; }
        browser [label="Browser /\\nSolidJS SPA", shape=oval, style=filled, fillcolor="#E8EAF6"];
        mcp_client [label="MCP Client\\n(Claude Code)", shape=oval, style=filled, fillcolor="#F3E5F5"];
        cli [label="CLI\\n(neuroncite)", shape=oval, style=filled, fillcolor="#FFF9C4"];

        // Entry points — second row
        { rank=same; middleware; mcp_transport; }
        middleware [label="{Middleware|SecurityHeaders\\lApiVersion\\lCORS\\lAuth (bearer token)\\lRate Limiter}", shape=record, style=filled, fillcolor="#FFF9C4"];
        mcp_transport [label="{MCP Transport\\n(neuroncite-mcp)|stdio JSON-RPC 2.0\\lParse method + params\\lDispatch to tool handler\\l50+ tools}", shape=record, style=filled, fillcolor="#F3E5F5"];

        // Handler
        handler [label="{Handler\\n(neuroncite-api)|Deserialize request\\lExtract AppState\\lDispatch to service\\lSerialize response}", shape=record, style=filled, fillcolor="#FCE4EC"];

        // Processing — same row
        { rank=same; pipeline; search; sse; }
        pipeline [label="{Pipeline\\n(neuroncite-pipeline)|Job executor\\lGPU worker thread\\lExtract + Chunk (CPU)\\lEmbed + Store (GPU)}", shape=record, style=filled, fillcolor="#FFF3E0"];
        search [label="{Search Pipeline\\n(neuroncite-search)|Vector search (HNSW)\\lKeyword search (BM25)\\lReciprocal Rank Fusion\\lDedup + Reranking}", shape=record, style=filled, fillcolor="#E0F7FA"];
        sse [label="SSE Broadcast\\n(progress + citation)", shape=oval, style=filled, fillcolor="#E0F7FA"];

        // Storage
        store [label="{Store\\n(neuroncite-store)|Sessions + Files + Pages\\lChunks + Embeddings\\lHNSW index\\lFTS5 keyword index\\lJobs}", shape=record, style=filled, fillcolor="#E3F2FD"];

        // Database
        db [label="SQLite\\n(WAL mode)", shape=cylinder, style=filled, fillcolor="#E8F5E9"];

        // HTTP path
        browser -> middleware [label="HTTP request"];
        middleware -> handler [label="routed"];
        handler -> pipeline [label="index"];
        handler -> search [label="search"];
        pipeline -> store [label="persist"];
        search -> store [label="query"];
        store -> db [label="SQL"];

        // MCP path
        mcp_client -> mcp_transport [label="JSON-RPC stdio"];
        mcp_transport -> handler [label="call"];

        // CLI path
        cli -> pipeline [label="direct"];
        cli -> search [label="direct"];

        // Side effects
        handler -> sse [label="broadcast", style=dashed];
        pipeline -> sse [label="progress", style=dashed];
    }
    """)
    write_dot_and_render(dot, "request_lifecycle")


# ---------------------------------------------------------------------------
# Diagram 2: Architecture layers (project-specific, matplotlib)
# ---------------------------------------------------------------------------

def generate_architecture_layers() -> None:
    """
    Render a layered architecture overview using matplotlib.
    Shows the vertical stack from database to UI, plus the MCP sidecar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    project_title = PROJECT_ROOT.name

    # Title
    ax.text(7, 11.6, f"{project_title} — Architecture Overview",
            ha="center", va="center", fontsize=16, fontweight="bold",
            fontfamily="Helvetica")

    # Layer definitions: (y_bottom, height, color, label, details)
    layers = [
        (0.3, 1.0, "#E8F5E9", "neuroncite-core",
         "Domain types (Chunk, SearchResult, Citation, PageText)\n"
         "EmbeddingModelConfig  |  NeuronCiteError  |  Traits"),
        (1.6, 1.0, "#FFCDD2", "neuroncite-pdf / neuroncite-html",
         "PDF extraction (pdf-extract, pdfium, OCR)  |  Page discovery\n"
         "HTML fetch  |  URL caching  |  Readability extraction"),
        (2.9, 1.0, "#E3F2FD", "neuroncite-chunk / neuroncite-embed",
         "Text splitting: Page, Word, Token, Sentence strategies\n"
         "ONNX Runtime embeddings  |  Model cache  |  Tokenization"),
        (4.2, 1.0, "#BBDEFB", "neuroncite-store",
         "SQLite (WAL)  |  Sessions, Files, Pages, Chunks, Jobs\n"
         "HNSW index serialization  |  FTS5 keyword index  |  Embedding files"),
        (5.5, 1.0, "#E0F7FA", "neuroncite-search",
         "Hybrid search: Vector (HNSW) + Keyword (BM25)\n"
         "Reciprocal Rank Fusion  |  Dedup  |  Reranking  |  Sub-chunk refinement"),
        (6.8, 1.0, "#FFF3E0", "neuroncite-pipeline",
         "Job executor  |  GPU worker thread  |  Two-phase indexing\n"
         "Extract+Chunk (CPU)  |  Embed+Store (GPU)  |  Cancellation"),
        (8.1, 1.0, "#FCE4EC", "neuroncite-api",
         "Axum REST  |  30+ endpoints under /api/v1/\n"
         "AppState  |  Auth middleware  |  OpenAPI spec  |  SSE streams"),
        (9.4, 1.0, "#E0F7FA", "neuroncite-web / neuroncite (binary)",
         "Embedded SolidJS SPA  |  Native window (tao/wry)\n"
         "CLI: web, serve, index, search, doctor, export, annotate, mcp"),
    ]

    main_left = 1.0
    main_width = 8.5

    for y, h, color, label, details in layers:
        rect = mpatches.FancyBboxPatch(
            (main_left, y), main_width, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#666666", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(main_left + 0.3, y + h - 0.15, label,
                fontsize=11, fontweight="bold", va="top", fontfamily="Helvetica")
        ax.text(main_left + 0.3, y + 0.12, details,
                fontsize=8, va="bottom", fontfamily="Helvetica", color="#333333")

    # MCP sidecar
    mcp_rect = mpatches.FancyBboxPatch(
        (10.2, 0.3), 3.2, 6.5,
        boxstyle="round,pad=0.1",
        facecolor="#F3E5F5", edgecolor="#666666", linewidth=1.2,
    )
    ax.add_patch(mcp_rect)
    ax.text(11.8, 6.5, "neuroncite-mcp",
            ha="center", fontsize=11, fontweight="bold", fontfamily="Helvetica")
    ax.text(11.8, 6.0, "(JSON-RPC / stdio)",
            ha="center", fontsize=9, fontfamily="Helvetica", color="#555555")

    mcp_details = (
        "50+ tools exposed\n\n"
        "Search & Index\n"
        "Session & File mgmt\n"
        "Job management\n"
        "Annotation & Highlight\n"
        "Citation verification\n"
        "HTML fetch & crawl\n"
        "Model management\n\n"
        "Clients:\n"
        "  Claude Code\n"
        "  Claude Desktop"
    )
    ax.text(11.8, 5.4, mcp_details,
            ha="center", va="top", fontsize=8, fontfamily="Helvetica", color="#333333")

    # Annotation / Citation sidecar
    anno_rect = mpatches.FancyBboxPatch(
        (10.2, 7.3), 3.2, 2.8,
        boxstyle="round,pad=0.1",
        facecolor="#FFF9C4", edgecolor="#666666", linewidth=1.2,
    )
    ax.add_patch(anno_rect)
    ax.text(11.8, 9.8, "neuroncite-annotate",
            ha="center", fontsize=10, fontweight="bold", fontfamily="Helvetica")
    ax.text(11.8, 9.3, "PDF highlight & comment\n5-stage text location\nAppearance stream injection",
            ha="center", va="top", fontsize=8, fontfamily="Helvetica", color="#333333")
    ax.text(11.8, 8.1, "neuroncite-citation",
            ha="center", fontsize=10, fontweight="bold", fontfamily="Helvetica")
    ax.text(11.8, 7.6, "LaTeX + BibTeX parsing\nBatch verification\nExport: CSV, XLSX, JSON",
            ha="center", va="top", fontsize=8, fontfamily="Helvetica", color="#333333")

    # Arrows between layers (vertical)
    arrow_style = dict(arrowstyle="->", color="#888888", lw=1.5)
    mid_x = main_left + main_width / 2

    for i in range(len(layers) - 1):
        y_top = layers[i][0] + layers[i][1]
        y_bot = layers[i + 1][0]
        ax.annotate("", xy=(mid_x, y_bot), xytext=(mid_x, y_top),
                     arrowprops=arrow_style)

    # Side arrow style
    side_arrow_base = dict(arrowstyle="->", color="#888888", lw=1.2, ls="--")
    if STRAIGHT_LINES:
        side_arrow_base["connectionstyle"] = "angle3,angleA=0,angleB=90"
    side_arrow = side_arrow_base

    # Arrow from API to MCP (API layer at y=8.1..9.1, MCP sidecar mid ~3.5)
    ax.annotate("", xy=(10.2, 3.5), xytext=(main_left + main_width, 8.6),
                arrowprops=side_arrow)
    ax.text(9.8, 6.0, "exposes", fontsize=8, ha="center", color="#888888",
            fontfamily="Helvetica")

    # Arrow from API to annotation/citation (sidecar mid ~8.7)
    ax.annotate("", xy=(10.2, 8.7), xytext=(main_left + main_width, 8.6),
                arrowprops=side_arrow)
    ax.text(9.8, 8.9, "uses", fontsize=8, ha="center", color="#888888",
            fontfamily="Helvetica")

    out = DOCS_DIR / "architecture_layers.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Diagram 3: Domain type map (project-specific)
# ---------------------------------------------------------------------------

def generate_type_map() -> None:
    """
    Render the domain type relationship map.
    Shows structs, enums, traits and their associations within neuroncite-core.
    """
    dot = textwrap.dedent("""\
    digraph type_map {
        rankdir=TB;
        fontname="Helvetica";
        node [fontname="Helvetica", fontsize=10, margin="0.15,0.07"];
        edge [arrowsize=0.6, fontsize=8, fontname="Helvetica"];
        bgcolor="white";
        label="NeuronCite — Domain Type Map (neuroncite-core)";
        labelloc=t;
        fontsize=14;
        compound=true;
        nodesep=0.5;
        ranksep=0.6;

        // --- Document Pipeline Types ---
        subgraph cluster_pipeline {
            label="Document Pipeline";
            style="dashed,rounded";
            color="#999999";
            margin=12;

            PageText       [shape=record, style=filled, fillcolor="#E8F5E9",
                             label="{PageText|source_file: String\\lpage_number: usize\\lcontent: String\\lbackend: ExtractionBackend}"];
            Chunk          [shape=record, style=filled, fillcolor="#E3F2FD",
                             label="{Chunk|source_file: String\\lpage_start/end: usize\\lchunk_index: usize\\lcontent: String\\lcontent_hash: u64}"];
            EmbeddedChunk  [shape=record, style=filled, fillcolor="#E3F2FD",
                             label="{EmbeddedChunk|chunk: Chunk\\lembedding: Vec\\<f32\\>}"];
        }

        // --- Search & Retrieval Types ---
        subgraph cluster_search {
            label="Search & Retrieval";
            style="dashed,rounded";
            color="#999999";
            margin=12;

            SearchResult   [shape=record, style=filled, fillcolor="#FFF3E0",
                             label="{SearchResult|score: f32\\lvector_score: f32\\lbm25_rank: Option\\lreranker_score: Option\\lcontent: String\\lcitation: Citation}"];
            Citation       [shape=record, style=filled, fillcolor="#FFF3E0",
                             label="{Citation|file_id: i64\\lsource_file: String\\lpage_start/end: usize\\ldisplay_name: String}"];
        }

        // --- Model Configuration ---
        subgraph cluster_model {
            label="Embedding Model";
            style="dashed,rounded";
            color="#999999";
            margin=12;

            EmbeddingModelConfig [shape=record, style=filled, fillcolor="#E0F7FA",
                             label="{EmbeddingModelConfig|model_id: String\\lvector_dimension: usize\\lmax_seq_len: usize\\lpooling: PoolingStrategy\\lquery_prefix: Option\\ldocument_prefix: Option}"];
        }

        // --- Enums ---
        subgraph cluster_enums {
            label="Enums";
            style="dashed,rounded";
            color="#999999";
            margin=12;

            ExtractionBackend [shape=record, style=filled, fillcolor="#FFECB3",
                            label="{\\<\\<enum\\>\\> ExtractionBackend|PdfExtract | Pdfium | Ocr\\lHtmlRaw | HtmlReadability}"];
            PoolingStrategy [shape=record, style=filled, fillcolor="#FFECB3",
                            label="{\\<\\<enum\\>\\> PoolingStrategy|Cls | MeanPooling | LastToken}"];
            SourceType     [shape=record, style=filled, fillcolor="#FFECB3",
                            label="{\\<\\<enum\\>\\> SourceType|Pdf | Html}"];
            StorageMode    [shape=record, style=filled, fillcolor="#FFECB3",
                            label="{\\<\\<enum\\>\\> StorageMode|SqliteBlob | MmapFile}"];
            NeuronCiteError [shape=record, style=filled, fillcolor="#FFCDD2",
                            label="{\\<\\<enum\\>\\> NeuronCiteError|Pdf | Chunk | Embed | Store | Search\\lApi | Gui | Config | Io | InvalidArgument}"];
        }

        // --- Relationships (data flow: top to bottom) ---
        PageText       -> Chunk           [label="chunked into"];
        PageText       -> ExtractionBackend [label="uses", style=dotted];
        Chunk          -> EmbeddedChunk   [label="embedded as"];
        EmbeddedChunk  -> SearchResult    [label="ranked as"];
        SearchResult   -> Citation        [label="contains"];
        EmbeddingModelConfig -> PoolingStrategy [label="uses", style=dotted];
        EmbeddingModelConfig -> EmbeddedChunk [label="produces", style=dashed];
    }
    """)
    write_dot_and_render(dot, "type_map")


# ---------------------------------------------------------------------------
# Diagram 4: API endpoint map (project-specific)
# ---------------------------------------------------------------------------

def generate_api_endpoint_map() -> None:
    """
    Render the REST API endpoint structure grouped by resource.
    """
    dot = textwrap.dedent("""\
    digraph api_endpoints {
        rankdir=LR;
        fontname="Helvetica";
        node [fontname="Helvetica", fontsize=9, shape=record, style=filled];
        edge [arrowsize=0.5, color="#555555"];
        bgcolor="white";
        label="NeuronCite — REST API Endpoint Map (/api/v1/)";
        labelloc=t;
        fontsize=14;
        nodesep=0.25;
        ranksep=0.8;

        router [label="Router\\n/api/v1/", fillcolor="#FFF9C4", shape=oval];

        health [label="{Health & System|GET /health\\lGET /backends\\lGET /sessions\\lPOST /shutdown}", fillcolor="#E8F5E9"];
        search [label="{Search|POST /search\\lPOST /search/hybrid\\lPOST /search/multi}", fillcolor="#E3F2FD"];
        indexing [label="{Indexing|POST /index\\lPOST /annotate\\lPOST /annotate/from-file\\lPOST /discover}", fillcolor="#E3F2FD"];
        sessions [label="{Sessions|DELETE /sessions/:id\\lPOST /sessions/delete-by-directory\\lPOST /sessions/:id/optimize\\lPOST /sessions/:id/rebuild\\lGET /sessions/:id/quality}", fillcolor="#FFF3E0"];
        documents [label="{Documents & Chunks|GET /documents/:id/pages/:n\\lGET /sessions/:sid/files/:fid/chunks\\lPOST /verify\\lPOST /files/compare}", fillcolor="#FFF3E0"];
        citation [label="{Citation Verification|POST /citation/create\\lPOST /citation/claim\\lPOST /citation/submit\\lGET /citation/:id/status\\lGET /citation/:id/rows\\lPOST /citation/:id/export\\lPOST /citation/:id/auto-verify\\lPOST /citation/fetch-sources\\lPOST /citation/parse-bib\\lPOST /citation/bib-report}", fillcolor="#FCE4EC"];
        jobs [label="{Jobs|GET /jobs\\lGET /jobs/:id\\lPOST /jobs/:id/cancel}", fillcolor="#F3E5F5"];
        openapi [label="{API Spec|GET /openapi.json}", fillcolor="#FFECB3"];

        router -> health;
        router -> search;
        router -> indexing;
        router -> sessions;
        router -> documents;
        router -> citation;
        router -> jobs;
        router -> openapi;
    }
    """)
    write_dot_and_render(dot, "api_endpoint_map")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating architecture diagrams...")
    print(f"  Output directory: {DOCS_DIR}")
    if OPTIMIZE:
        print("  [optimize] mclimit=200, nslimit=200, searchsize=2000")
    if STRAIGHT_LINES:
        print("  [straight-lines] splines=ortho")
    print()

    # Auto-detected diagrams (work with any Rust workspace)
    print("[1/7] Crate dependency graph (auto-detected)")
    generate_crate_dependency_graph()

    print("[5/7] Module file tree (auto-detected)")
    generate_module_tree()

    if AUTO_ONLY:
        print()
        print("Done (auto-only mode). Skipped project-specific diagrams.")
        sys.exit(0)

    # Project-specific diagrams
    print("[2/7] Architecture layers")
    generate_architecture_layers()

    print("[3/7] Domain type map")
    generate_type_map()

    print("[4/7] API endpoint map")
    generate_api_endpoint_map()

    print("[6/7] Error propagation chain")
    generate_error_propagation()

    print("[7/7] Request lifecycle")
    generate_request_lifecycle()

    print()
    print(f"Done. All diagrams saved to {DOCS_DIR}")
