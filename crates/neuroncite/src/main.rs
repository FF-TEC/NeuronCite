// NeuronCite -- local, privacy-preserving semantic document search engine.
// Copyright (C) 2026 NeuronCite Contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! NeuronCite binary entry point.
//!
//! Parses CLI arguments via clap to determine the execution mode (web, serve,
//! index, search, doctor, sessions, export, version). Initializes the tracing
//! subscriber for structured logging, then dispatches to the appropriate
//! execution path. When invoked without a subcommand, defaults to web UI mode:
//! the axum server starts with the embedded SolidJS frontend and the default
//! browser opens at `http://localhost:3030`. In headless mode, the API server
//! runs until interrupted, or a single CLI command is executed and the process
//! exits.

// `deny(warnings)` is inherited from [workspace.lints.rust] in the root
// Cargo.toml. Repeating it here would duplicate the enforcement and would
// interact unexpectedly with `allow(...)` attributes in this file.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Parser, Subcommand, ValueEnum};

/// NeuronCite: local, privacy-preserving semantic document search engine.
///
/// When invoked without a subcommand, defaults to web UI mode (equivalent to
/// `neuroncite web`). The embedded SolidJS frontend is served by axum and the
/// default browser opens at `http://localhost:3030`. All subcommands that produce
/// output write machine-readable JSON to stdout by default.
#[derive(Parser)]
#[command(name = "neuroncite", version, about, long_about = None)]
struct Cli {
    /// Subcommand to execute. Defaults to `web` when omitted.
    #[command(subcommand)]
    command: Option<Command>,

    /// Output format for CLI subcommands.
    #[arg(long, global = true, default_value = "json")]
    format: OutputFormat,

    /// Path to the TOML configuration file. When omitted, the application
    /// searches for `neuroncite.toml` in the standard configuration directories.
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Logging verbosity level. Maps to tracing subscriber filter directives.
    #[arg(long, global = true, default_value = "info")]
    log_level: LogLevel,
}

/// CLI subcommands corresponding to the different execution modes of NeuronCite.
#[derive(Subcommand)]
enum Command {
    /// Launch the web UI: starts the API server and opens the default browser.
    /// The SolidJS frontend is embedded in the binary and served by the same
    /// axum server that handles API requests. The server runs until Ctrl+C.
    #[cfg(feature = "web")]
    Web {
        /// TCP port for the HTTP server (default: 3030).
        #[arg(long, default_value = "3030")]
        port: u16,

        /// Bind address for the HTTP server. Use "0.0.0.0" for LAN access.
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,
    },

    /// Start the API server in headless mode without the browser-based frontend.
    /// The server runs until terminated by Ctrl+C, SIGTERM, or a /api/v1/shutdown
    /// request.
    Serve {
        /// TCP port for the HTTP server (default: 3030).
        #[arg(long, default_value = "3030")]
        port: u16,

        /// Bind address for the HTTP server. Use "0.0.0.0" for LAN access.
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,

        /// Print the bearer token to stdout on startup (for scripted environments).
        #[arg(long)]
        print_token: bool,
    },

    /// Index a directory of PDF files from the command line without starting
    /// a persistent HTTP server.
    Index {
        /// Path to the directory containing PDF files to index.
        #[arg(long)]
        directory: PathBuf,

        /// Embedding model identifier (e.g., "BAAI/bge-small-en-v1.5").
        #[arg(long, default_value = "BAAI/bge-small-en-v1.5")]
        model: String,

        /// Chunking strategy name (page, word, token, sentence).
        #[arg(long, default_value = "token")]
        strategy: String,

        /// Number of words or tokens per chunk (depends on strategy).
        #[arg(long, default_value = "256")]
        chunk_size: usize,

        /// Number of words or tokens of overlap between consecutive chunks.
        #[arg(long, default_value = "32")]
        overlap: usize,

        /// OCR language code for Tesseract (default: "eng").
        #[arg(long, default_value = "eng")]
        ocr_language: String,

        /// Suppress all stderr progress output.
        #[arg(long)]
        quiet: bool,

        /// Progress output format (ndjson or none).
        #[arg(long, default_value = "ndjson")]
        progress: ProgressMode,
    },

    /// Execute search queries against an existing session. Supports single-query
    /// mode (--query) and batch mode (--batch) for amortized GPU initialization.
    /// In batch mode, the embedding backend and HNSW index are loaded once and
    /// reused across all queries, avoiding the ~2s GPU init overhead per query.
    Search {
        /// Path to the directory containing the .neuroncite database.
        #[arg(long)]
        directory: PathBuf,

        /// Session ID to search within (from a previous index operation).
        #[arg(long)]
        session_id: i64,

        /// Search query string for single-query mode. Mutually exclusive with
        /// --batch.
        #[arg(long, required_unless_present = "batch", conflicts_with = "batch")]
        query: Option<String>,

        /// Path to a JSONL file containing batch queries, or "-" for stdin.
        /// Each line is a JSON object with a required "query" field and an
        /// optional "top_k" override. The embedding backend and HNSW index
        /// are loaded once and reused for all queries. Results are written as
        /// NDJSON (one JSON object per query) to stdout.
        #[arg(long, conflicts_with = "query")]
        batch: Option<String>,

        /// Number of results to return per query.
        #[arg(long, default_value = "10")]
        top_k: usize,

        /// Enable hybrid search (vector + BM25).
        #[arg(long)]
        hybrid: bool,

        /// Enable cross-encoder reranking.
        #[arg(long)]
        rerank: bool,
    },

    /// Run the Dependency Doctor checks and report the status of all optional
    /// runtime dependencies.
    Doctor,

    /// List all index sessions in the specified directory's database.
    Sessions {
        /// Path to the directory containing the .neuroncite database.
        #[arg(long)]
        directory: PathBuf,
    },

    /// Export search results from a previous search to a file format.
    Export {
        /// Path to the directory containing the .neuroncite database.
        #[arg(long)]
        directory: PathBuf,

        /// Session ID to search within.
        #[arg(long)]
        session_id: i64,

        /// Search query string.
        #[arg(long)]
        query: String,

        /// Export format (markdown, bibtex, csl-json, ris, plain-text).
        #[arg(long, value_name = "FORMAT")]
        export_format: ExportFormat,

        /// Output file path (defaults to stdout).
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Annotate PDFs with highlights and comments from CSV/JSON input.
    /// Reads annotation instructions (title, author, quote, color, comment)
    /// from a file, matches them to PDFs in the source directory via fuzzy
    /// filename matching, locates the quoted text, creates highlight and
    /// comment annotations, and saves annotated copies to the output directory.
    Annotate {
        /// Path to CSV or JSON input file containing annotation instructions.
        #[arg(long)]
        input: PathBuf,

        /// Directory containing source PDF files.
        #[arg(long)]
        source_dir: PathBuf,

        /// Directory for output annotated PDFs and the annotation report.
        #[arg(long)]
        output_dir: PathBuf,

        /// Default highlight color in hex (#RRGGBB). Used for rows without
        /// a color field. Default: #FFFF00 (yellow).
        #[arg(long, default_value = "#FFFF00")]
        default_color: String,
    },

    /// Print the application version, build features, and Git commit hash.
    Version,

    /// Model management commands for discovering, downloading, and verifying
    /// embedding models in the local cache. Designed for programmatic use by
    /// LLM agents that need to inspect available models, check system
    /// capabilities, and ensure models are cached before indexing.
    Models {
        #[command(subcommand)]
        action: ModelsCommand,
    },

    /// MCP (Model Context Protocol) server for integration with AI assistants.
    /// Provides subcommands to start the stdio server, register/unregister
    /// with Claude Code or Claude Desktop App, and check registration status.
    #[cfg(feature = "mcp")]
    Mcp {
        #[command(subcommand)]
        action: McpCommand,
    },
}

/// Subcommands for the `mcp` command group. Manages the MCP stdio server
/// and its registration in Claude Code and Claude Desktop App configuration files.
#[cfg(feature = "mcp")]
#[derive(Subcommand)]
enum McpCommand {
    /// Start the MCP server in stdio mode. The MCP client (Claude Code or
    /// Claude Desktop App) spawns this as a child process and communicates
    /// via stdin/stdout JSON-RPC 2.0 messages.
    Serve {
        /// HuggingFace model identifier to load at startup. Overrides the
        /// config file's `default_model` setting. The model must match the
        /// one used during indexing, otherwise search returns a dimension
        /// mismatch error.
        #[arg(long)]
        model: Option<String>,
    },

    /// Register the NeuronCite MCP server in the specified client's config.
    /// Defaults to Claude Code if no target is given.
    Install {
        /// Target client: "claude-code" or "claude-desktop".
        #[arg(long, default_value = "claude-code")]
        target: String,
    },

    /// Remove the NeuronCite MCP server entry from the specified client's config.
    Uninstall {
        /// Target client: "claude-code" or "claude-desktop".
        #[arg(long, default_value = "claude-code")]
        target: String,
    },

    /// Check the current MCP registration status. Shows all targets unless
    /// a specific one is given.
    Status {
        /// Target client: "claude-code" or "claude-desktop". Omit to show all.
        #[arg(long)]
        target: Option<String>,
    },
}

/// Subcommands for the `models` command group. Each subcommand outputs
/// machine-readable JSON to stdout (when `--format json` is active, which
/// is the default) for consumption by LLM agents and automation scripts.
#[derive(Subcommand)]
enum ModelsCommand {
    /// List all available embedding models with their configuration
    /// parameters and local cache status.
    List,

    /// Show detailed configuration for a specific embedding model,
    /// including instruction prefixes, token type ID requirements,
    /// and per-file cache status.
    Info {
        /// HuggingFace model identifier (e.g., "BAAI/bge-small-en-v1.5").
        model_id: String,
    },

    /// Download a model's files to the local cache. If the model is
    /// already cached, reports the existing cache path without
    /// re-downloading.
    Download {
        /// HuggingFace model identifier (e.g., "BAAI/bge-small-en-v1.5").
        model_id: String,
    },

    /// Verify the integrity of a cached model by checking that all
    /// expected files exist and computing their SHA-256 checksums.
    Verify {
        /// HuggingFace model identifier (e.g., "BAAI/bge-small-en-v1.5").
        model_id: String,
    },

    /// Show system capabilities: GPU hardware detection, CUDA execution
    /// provider availability, compiled backends, and cache directory
    /// information with disk usage.
    System,
}

/// Output format for CLI subcommand results.
#[derive(Clone, ValueEnum)]
enum OutputFormat {
    /// Machine-readable JSON output on stdout.
    Json,
    /// Human-readable tabular output for interactive terminal use.
    Text,
}

/// Logging verbosity level mapping to tracing subscriber filter directives.
#[derive(Clone, ValueEnum)]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    /// Returns the tracing filter directive string corresponding to this level.
    fn as_filter_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

/// Progress output format for the index subcommand.
#[derive(Clone, ValueEnum)]
enum ProgressMode {
    /// Newline-delimited JSON objects on stderr, one per processed file.
    Ndjson,
    /// No progress output (equivalent to --quiet).
    None,
}

/// Export file format for the export subcommand. Supports structured
/// reference formats (BibTeX, CSL-JSON, RIS) and human-readable output
/// (Markdown, plain text).
#[derive(Clone, ValueEnum)]
enum ExportFormat {
    Markdown,
    Bibtex,
    CslJson,
    Ris,
    PlainText,
}

// ---------------------------------------------------------------------------
// Shared indexing pipeline -- re-exported from neuroncite_api::indexer.
//
// The pipeline functions (extract_and_chunk_file, run_extraction_phase,
// embed_and_store_file, f32_slice_to_bytes, bytes_to_f32_vec) and types
// (ExtractedFile, FileIndexResult) live in neuroncite_api::indexer. All
// entry points (CLI, API, MCP, web) use the same implementation.
// ---------------------------------------------------------------------------

use neuroncite_api::indexer::{ExtractedFile, embed_and_store_file, run_extraction_phase};

#[cfg(feature = "mcp")]
use neuroncite_api::indexer::{restore_stdout, suppress_stdout};

use neuroncite_api::indexer::bytes_to_f32_vec;

// ---------------------------------------------------------------------------
// Tracing initialization
// ---------------------------------------------------------------------------

/// Initializes the tracing subscriber with the configured log level filter.
/// The `ort=warn` directive suppresses the ONNX Runtime crate's verbose
/// INFO-level log messages (GraphTransformer, Reserving memory, node
/// placement, etc.) that flood the console during session initialization.
fn init_tracing(level: &LogLevel) {
    use tracing_subscriber::EnvFilter;

    let filter_str = format!("{},ort=warn", level.as_filter_str());

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter_str));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();
}

/// Initializes a tracing subscriber for the MCP stdio server.
///
/// The MCP protocol uses stdout exclusively for JSON-RPC 2.0 messages.
/// Any non-JSON output on stdout (tracing logs, ONNX Runtime diagnostics,
/// third-party crate debug prints) corrupts the protocol and causes the
/// MCP client to reject the server.
///
/// This subscriber differs from the standard `init_tracing` in two ways:
///
/// 1. **Explicit stderr writer**: The `.with_writer(std::io::stderr)` call
///    guarantees tracing output goes to stderr regardless of platform-specific
///    default behavior or future library version changes.
///
/// 2. **Suppressed ORT logging**: The `ort=off` directive silences all ONNX
///    Runtime log messages (including the WARN about node execution provider
///    assignment) that the `ort` crate bridges from the C++ runtime into the
///    Rust tracing subscriber. These messages are informational and provide
///    no value to the MCP client.
fn init_mcp_tracing(level: &LogLevel) {
    use tracing_subscriber::EnvFilter;

    let filter_str = format!("{},ort=off", level.as_filter_str());

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter_str));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();
}

// ---------------------------------------------------------------------------
// Entry point and command dispatch
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    // Default to web UI mode when no subcommand is provided (e.g. double-click).
    // Falls back to headless serve mode when the web feature is not compiled.
    #[cfg(feature = "web")]
    let command = cli.command.unwrap_or(Command::Web {
        port: 3030,
        bind: "127.0.0.1".to_string(),
    });
    #[cfg(not(feature = "web"))]
    let command = cli.command.unwrap_or(Command::Serve {
        port: 3030,
        bind: "127.0.0.1".to_string(),
        print_token: false,
    });

    // The MCP serve subcommand requires a specialized tracing subscriber that
    // explicitly writes to stderr and suppresses all ONNX Runtime log output.
    // stdout is reserved exclusively for JSON-RPC 2.0 messages in the MCP
    // protocol; any non-JSON output on stdout corrupts the protocol and causes
    // Claude Code to reject the server.
    #[cfg(feature = "mcp")]
    let is_mcp_serve = matches!(
        &command,
        Command::Mcp {
            action: McpCommand::Serve { .. }
        }
    );
    #[cfg(not(feature = "mcp"))]
    let is_mcp_serve = false;

    // The web command sets up its own layered tracing subscriber that includes
    // a BroadcastLayer for streaming log events to the browser's Log panel via
    // SSE. For all other commands, the standard fmt subscriber is used.
    #[cfg(feature = "web")]
    let is_web = matches!(&command, Command::Web { .. });
    #[cfg(not(feature = "web"))]
    let is_web = false;

    if is_mcp_serve {
        init_mcp_tracing(&cli.log_level);
    } else if !is_web {
        init_tracing(&cli.log_level);
    }

    let exit_code = match command {
        #[cfg(feature = "web")]
        Command::Web { port, bind } => run_web(cli.config, port, bind, &cli.log_level),
        Command::Serve {
            port,
            bind,
            print_token,
        } => run_serve(cli.config, port, bind, print_token),
        Command::Index {
            directory,
            model,
            strategy,
            chunk_size,
            overlap,
            ocr_language,
            quiet,
            progress,
        } => run_index(
            cli.config,
            directory,
            model,
            strategy,
            chunk_size,
            overlap,
            ocr_language,
            quiet,
            progress,
        ),
        Command::Search {
            directory,
            session_id,
            query: Some(q),
            batch: None,
            top_k,
            hybrid,
            rerank,
        } => run_search(directory, session_id, q, top_k, hybrid, rerank),
        Command::Search {
            directory,
            session_id,
            query: None,
            batch: Some(batch_path),
            top_k,
            hybrid,
            rerank,
        } => run_batch_search(directory, session_id, batch_path, top_k, hybrid, rerank),
        // Clap enforces that exactly one of --query or --batch is present.
        Command::Search { .. } => unreachable!("clap enforces --query or --batch"),
        Command::Doctor => run_doctor(),
        Command::Sessions { directory } => run_sessions(directory),
        Command::Export {
            directory,
            session_id,
            query,
            export_format,
            output,
        } => run_export(directory, session_id, query, export_format, output),
        Command::Annotate {
            input,
            source_dir,
            output_dir,
            default_color,
        } => run_annotate(input, source_dir, output_dir, default_color),
        Command::Version => run_version(cli.format),
        Command::Models { action } => run_models(action, cli.format),
        #[cfg(feature = "mcp")]
        Command::Mcp { action } => run_mcp(cli.config, action),
    };

    std::process::exit(exit_code);
}

// ---------------------------------------------------------------------------
// Shared server initialization
// ---------------------------------------------------------------------------

/// Initializes the database, embedding backend, GPU worker, AppState, HNSW
/// indexes, and job executor. Shared by `run_serve`, `run_web` (GUI), and
/// `run_web` (non-GUI) to eliminate the three-way code duplication of the
/// server setup sequence.
///
/// Returns the fully-initialized `Arc<AppState>` and the background executor
/// handle on success, or an exit code on failure.
///
/// Callers are responsible for:
/// - Creating the tokio runtime and calling this inside `block_on(async { ... })`
/// - Setting up broadcast channels (progress_tx, citation_tx) for web mode
/// - Binding the TCP listener and building the router
async fn init_server_context(
    config: neuroncite_core::AppConfig,
    headless: bool,
    bearer_token: Option<String>,
    data_dir: PathBuf,
) -> Result<(Arc<neuroncite_api::AppState>, tokio::task::JoinHandle<()>), i32> {
    // Emit tracing warnings for potentially dangerous configuration combinations.
    // Called here (after the tracing subscriber is initialized by the caller) so the
    // warnings appear in the log output regardless of which server mode is active.
    config.check_security_warnings();

    if let Err(e) = std::fs::create_dir_all(&data_dir) {
        eprintln_error("db_init", &format!("failed to create data directory: {e}"));
        return Err(1);
    }
    let db_path = data_dir.join("index.db");

    let pool = match neuroncite_store::create_pool_with_size(&db_path, config.db_pool_size) {
        Ok(p) => p,
        Err(e) => {
            eprintln_error("db_init", &format!("failed to create database pool: {e}"));
            return Err(1);
        }
    };

    {
        let conn = match pool.get() {
            Ok(c) => c,
            Err(e) => {
                eprintln_error("db_init", &format!("failed to obtain connection: {e}"));
                return Err(1);
            }
        };
        if let Err(e) = neuroncite_store::migrate(&conn) {
            eprintln_error("db_init", &format!("schema migration failed: {e}"));
            return Err(1);
        }
    }

    // Create the embedding backend and conditionally load the default model.
    // On first run (no .setup_complete marker), the server starts with an
    // unloaded backend so no downloads are triggered before the user consents
    // via the WelcomeDialog. On subsequent runs, the default model is loaded
    // immediately so search and indexing are available without manual action.
    // Backend creation failure (no compiled backend) is still fatal.
    let is_first_run = !neuroncite_core::paths::setup_complete_path().exists();
    let backend: Arc<dyn neuroncite_core::EmbeddingBackend> = {
        let backends = neuroncite_embed::list_available_backends();
        let backend_name = backends
            .first()
            .map(|b| b.name.as_str())
            .unwrap_or(neuroncite_embed::DEFAULT_BACKEND_NAME);
        let default_model = config.default_model.clone();
        match neuroncite_embed::create_backend(backend_name) {
            Ok(mut b) => {
                if is_first_run {
                    tracing::info!(
                        "first run detected — skipping model load until user completes setup"
                    );
                } else if let Err(e) = b.load_model(&default_model) {
                    eprintln_error(
                        "model_load",
                        &format!(
                            "default model '{}' could not be loaded: {e}. \
                             Server is starting without an active model. \
                             Use the Models tab to download a model.",
                            default_model
                        ),
                    );
                    tracing::warn!(
                        model_id = default_model.as_str(),
                        error = %e,
                        "default model not available; server starting without active model"
                    );
                }
                Arc::from(b)
            }
            Err(e) => {
                eprintln_error("embed_init", &format!("failed to create backend: {e}"));
                return Err(1);
            }
        }
    };

    let vector_dim = backend.vector_dimension();
    let worker_handle = neuroncite_api::spawn_worker(backend, None);

    let state = match neuroncite_api::AppState::new(
        pool,
        worker_handle,
        config,
        headless,
        bearer_token,
        vector_dim,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln_error("config_error", &e);
            return Err(1);
        }
    };

    // Load HNSW indexes for all sessions that have embeddings stored in
    // SQLite. This populates the per-session HNSW map from previously
    // indexed data so search is available immediately after server restart.
    neuroncite_api::load_all_session_hnsw(&state);

    // Spawn the background job executor for processing queued indexing
    // jobs submitted via the REST API's POST /api/v1/index endpoint.
    let executor_handle = neuroncite_api::spawn_job_executor(state.clone());

    // Spawn the rate limiter eviction task when bearer token authentication
    // is configured. The task periodically removes stale per-IP failure
    // counters from the shared DashMap to prevent unbounded memory growth
    // under sustained brute-force attacks that never retry from the same IP.
    // When no bearer token is configured, rate limiting is not active and the
    // task is not needed.
    if state.auth.bearer_token_hash.is_some() {
        neuroncite_api::middleware::auth::spawn_rate_limit_eviction(
            state.cancellation_token.clone(),
            state.auth.failed_auth_attempts.clone(),
            state.auth.time_index.clone(),
            state.auth.global_failure_counter.clone(),
        );
    }

    Ok((state, executor_handle))
}

// ---------------------------------------------------------------------------
// Headless server mode
// ---------------------------------------------------------------------------

/// Starts the API server in headless mode. Returns exit code 0 on clean
/// shutdown, 3 on shutdown timeout.
fn run_serve(config_path: Option<PathBuf>, port: u16, bind: String, print_token: bool) -> i32 {
    let mut config = load_config(config_path);

    // Synchronize config.bind_address with the CLI --bind argument. Without
    // this assignment, the CORS middleware reads config.bind_address (which
    // defaults to "127.0.0.1" from the config file) even when the user passes
    // a different address on the command line (e.g., --bind 0.0.0.0). The
    // mismatch causes CORS to be evaluated as "localhost mode" while the
    // server actually listens on all interfaces, producing incorrect headers.
    config.bind_address = bind.clone();

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln_error(
                "runtime_init",
                &format!("failed to create tokio runtime: {e}"),
            );
            return 1;
        }
    };

    rt.block_on(async {
        tracing::info!(port = port, bind = %bind, "starting NeuronCite in headless server mode");

        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let data_dir = neuroncite_core::paths::index_dir_for_path(&cwd);

        // When --print-token is supplied, a random 128-bit token is generated
        // and passed to AppState. The auth middleware then requires every
        // incoming HTTP request to carry this token in the Authorization header.
        // The token is printed to stdout so the caller can capture it.
        // Without --print-token, no token is generated and the middleware allows
        // all requests through, which is safe for trusted local or intranet use.
        let bearer_token: Option<String> = if print_token {
            let token = uuid::Uuid::new_v4().to_string().replace('-', "");
            println!("Bearer token: {token}");
            Some(token)
        } else {
            None
        };

        let (state, _executor_handle) =
            match init_server_context(config, true, bearer_token, data_dir).await {
                Ok(ctx) => ctx,
                Err(code) => return code,
            };

        // Print the one-time shutdown nonce to stdout so the operator can capture it
        // for use in POST /api/v1/shutdown requests. The nonce is generated fresh at
        // each server startup and is required by the shutdown endpoint regardless of
        // whether a bearer token is configured.
        println!("shutdown_nonce: {}", state.auth.shutdown_nonce);

        let addr: std::net::SocketAddr = match format!("{bind}:{port}").parse() {
            Ok(addr) => addr,
            Err(e) => {
                eprintln_error("bind_error", &format!("invalid bind address: {e}"));
                return 1;
            }
        };

        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                eprintln_error("bind_error", &format!("failed to bind to {addr}: {e}"));
                return 1;
            }
        };

        tracing::info!(%addr, "HTTP server listening");

        // Graceful drain timeout: after the OS shutdown signal fires, existing
        // connections have at most this many seconds to complete. This prevents
        // a hung WebSocket or long-polling connection from blocking process exit
        // indefinitely. The value is intentionally small because this is a local
        // single-user application with no long-lived connections in normal use.
        const SHUTDOWN_DRAIN_TIMEOUT_SECS: u64 = 30;

        // A oneshot channel decouples the OS signal from the axum serve future.
        // The sender is moved into a background task that fires ctrl-c or SIGTERM
        // detection, then signals axum to start graceful drain. The receiver is
        // used as the axum shutdown signal. After signal fires, the serve task is
        // awaited with a bounded timeout.
        let (signal_tx, signal_rx) = tokio::sync::oneshot::channel::<()>();

        tokio::spawn(async move {
            shutdown_signal().await;
            tracing::info!(
                timeout_secs = SHUTDOWN_DRAIN_TIMEOUT_SECS,
                "shutdown signal received; starting graceful drain"
            );
            let _ = signal_tx.send(());
        });

        let app = neuroncite_api::build_router(state);
        let serve_task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = signal_rx.await;
                })
                .await
        });

        match tokio::time::timeout(
            std::time::Duration::from_secs(SHUTDOWN_DRAIN_TIMEOUT_SECS),
            serve_task,
        )
        .await
        {
            Ok(Ok(Ok(()))) => {
                tracing::info!("server shut down cleanly");
                0
            }
            Ok(Ok(Err(e))) => {
                eprintln_error("server_error", &format!("server error: {e}"));
                1
            }
            Ok(Err(join_err)) => {
                eprintln_error(
                    "server_error",
                    &format!("server task join error: {join_err}"),
                );
                1
            }
            Err(_elapsed) => {
                tracing::error!(
                    timeout_secs = SHUTDOWN_DRAIN_TIMEOUT_SECS,
                    "graceful drain did not complete within the timeout; forcing shutdown"
                );
                3
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Web UI mode
// ---------------------------------------------------------------------------

/// Starts the axum server with the embedded SolidJS frontend and opens the
/// default browser. The server initialization follows the same pattern as
/// `run_serve` (database, embedding backend, worker, AppState, HNSW loading,
/// job executor) but uses `neuroncite_web::build_web_router` instead of the
/// plain API router. The web router merges the existing API routes with
/// web-specific endpoints (file browsing, model management, etc.), SSE streams,
/// and embedded static file serving.
///
/// When compiled with the `gui` feature, the tokio runtime runs in a background
/// thread while tao's event loop occupies the main thread (platform requirement
/// on macOS and Windows). The two threads communicate via channels:
/// - `std::sync::mpsc` carries the server URL from the background thread to the
///   main thread after port binding completes.
/// - `tokio::sync::oneshot` carries the shutdown signal from the main thread
///   (window close) to the background thread (graceful server shutdown).
///
/// When compiled without `gui`, the existing behavior is preserved: the tokio
/// runtime runs on the main thread and the default browser is opened via the
/// `opener` crate.
#[cfg(feature = "web")]
fn run_web(config_path: Option<PathBuf>, port: u16, bind: String, log_level: &LogLevel) -> i32 {
    let mut config = load_config(config_path);
    config.bind_address = bind.clone();

    // Create the log broadcast channel before initializing the tracing
    // subscriber so the BroadcastLayer and WebState share the same sender.
    // Capacity 2048 prevents message loss during burst periods (e.g. heavy
    // indexing with many files) where the backend can emit logs faster than
    // the SSE stream delivers them to slow browser clients.
    let (log_tx, _) = tokio::sync::broadcast::channel::<String>(2048);

    // Set up a layered tracing subscriber that sends events both to the
    // console (fmt layer) and to the SSE log stream (BroadcastLayer).
    {
        use tracing_subscriber::EnvFilter;
        use tracing_subscriber::layer::SubscriberExt;

        let filter_str = format!("{},ort=warn", log_level.as_filter_str());
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter_str));

        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_thread_ids(false)
            .with_file(false)
            .with_line_number(false);

        let broadcast_layer = neuroncite_web::broadcast_layer::BroadcastLayer::new(log_tx.clone());

        let subscriber = tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .with(broadcast_layer);

        tracing::subscriber::set_global_default(subscriber)
            .expect("failed to set global tracing subscriber");
    }

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln_error(
                "runtime_init",
                &format!("failed to create tokio runtime: {e}"),
            );
            return 1;
        }
    };

    // --- GUI path: tao event loop on main thread, tokio in background thread ---
    #[cfg(feature = "gui")]
    {
        // Pre-flight: verify that the native GUI can be created on this
        // system (e.g. libwebkit2gtk is installed on Linux). If the check
        // fails, fall back to the browser-based flow instead of consuming
        // the shutdown channel in run_gui_with_splash (which would cause
        // the server to shut down immediately on GUI failure).
        if let Err(reason) = neuroncite_web::native_window::preflight_gui_check() {
            // --- Browser fallback path (GUI prerequisites missing) ---
            tracing::warn!(reason = %reason, "native GUI unavailable, falling back to browser");
            eprintln!("WARN: {reason}");
            eprintln!("Falling back to default browser.\n");

            let (url_tx, url_rx) = std::sync::mpsc::channel::<String>();

            let server_thread = std::thread::Builder::new()
                .name("axum-server".into())
                .spawn(move || {
                    rt.block_on(async {
                        tracing::info!(
                            port = port, bind = %bind,
                            "starting NeuronCite in web UI mode (browser fallback)"
                        );

                        let data_dir = neuroncite_core::paths::gui_index_dir();

                        let (state, _executor_handle) =
                            match init_server_context(config, true, None, data_dir).await {
                                Ok(ctx) => ctx,
                                Err(code) => return code,
                            };

                        let (listener, actual_port) = {
                            let mut last_err = None;
                            let mut found = None;
                            for candidate in port..=port.saturating_add(19) {
                                let addr_str = format!("{bind}:{candidate}");
                                let addr: std::net::SocketAddr = match addr_str.parse() {
                                    Ok(a) => a,
                                    Err(e) => {
                                        eprintln_error("bind_error", &format!("invalid bind address: {e}"));
                                        return 1;
                                    }
                                };
                                match tokio::net::TcpListener::bind(addr).await {
                                    Ok(l) => {
                                        if candidate != port {
                                            tracing::info!(
                                                requested_port = port,
                                                actual_port = candidate,
                                                "preferred port {port} was occupied, using port {candidate} instead"
                                            );
                                        }
                                        found = Some((l, candidate));
                                        break;
                                    }
                                    Err(e) => {
                                        last_err = Some(e);
                                    }
                                }
                            }
                            match found {
                                Some(pair) => pair,
                                None => {
                                    eprintln_error(
                                        "bind_error",
                                        &format!(
                                            "failed to bind to ports {port}--{}; last error: {}",
                                            port.saturating_add(19),
                                            last_err.map(|e| e.to_string()).unwrap_or_default()
                                        ),
                                    );
                                    return 1;
                                }
                            }
                        };

                        let url = format!("http://{bind}:{actual_port}");
                        let addr = listener
                            .local_addr()
                            .expect("TCP listener address available after successful bind");
                        tracing::info!(%addr, "NeuronCite web server listening");

                        // No native dialogs in fallback mode (no tao event loop).
                        let web_state = neuroncite_web::WebState::new(state, log_tx);
                        let app = neuroncite_web::build_web_router(
                            web_state.app_state.clone(),
                            web_state,
                        );

                        let _ = url_tx.send(url);

                        // Simplified shutdown: no native window, only Ctrl+C.
                        match axum::serve(listener, app)
                            .with_graceful_shutdown(shutdown_signal())
                            .await
                        {
                            Ok(()) => {
                                tracing::info!("web server shut down cleanly");
                                0
                            }
                            Err(e) => {
                                eprintln_error("server_error", &format!("server error: {e}"));
                                1
                            }
                        }
                    })
                })
                .expect("failed to spawn server thread");

            // Wait for the server URL, then open the default browser.
            match url_rx.recv() {
                Ok(url) => {
                    eprintln!();
                    eprintln!("  NeuronCite is running at: {url}");
                    eprintln!("  Native window not available, opened in browser instead.");
                    eprintln!("  Press Ctrl+C to stop the server.");
                    eprintln!();
                    // Open browser synchronously -- no tokio runtime on main thread.
                    if let Err(e) = opener::open_browser(&url) {
                        eprintln!("  Could not open browser automatically: {e}");
                        eprintln!("  Open this URL manually: {url}");
                    }
                }
                Err(_) => {
                    eprintln_error(
                        "server_init",
                        "server failed during startup (URL not received)",
                    );
                }
            }

            // Block main thread until server thread exits (on Ctrl+C).
            return server_thread.join().unwrap_or(1);
        }

        // --- Normal GUI path (pre-flight passed) ---

        // Channel for the server thread to send the bound URL back to the main
        // thread after port scanning completes. If the server fails during
        // setup, the sender is dropped without sending, causing recv() to
        // return Err -- the main thread then joins the server thread to
        // retrieve the exit code.
        let (url_tx, url_rx) = std::sync::mpsc::channel::<String>();

        // Channel for the main thread to signal the server to shut down when
        // the native window is closed. The receiver is awaited inside
        // tokio::select! alongside the Ctrl+C signal handler.
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        // Move the tokio runtime to a background thread. tao's event loop
        // requires the main thread on macOS and Windows (platform requirement
        // for NSApplication and Win32 message pump respectively).
        let server_thread = std::thread::Builder::new()
            .name("axum-server".into())
            .spawn(move || {
                rt.block_on(async {
                    tracing::info!(port = port, bind = %bind, "starting NeuronCite in web UI mode (native window)");

                    let data_dir = neuroncite_core::paths::gui_index_dir();

                    let (state, _executor_handle) =
                        match init_server_context(config, true, None, data_dir).await {
                            Ok(ctx) => ctx,
                            Err(code) => return code,
                        };

                    // SSE broadcast channels (progress_tx, citation_tx, source_tx) are
                    // initialised unconditionally inside AppState::new(); no set() calls
                    // are needed here. WebState::new() clones the senders from AppState.

                    let (listener, actual_port) = {
                        let mut last_err = None;
                        let mut found = None;
                        for candidate in port..=port.saturating_add(19) {
                            let addr_str = format!("{bind}:{candidate}");
                            let addr: std::net::SocketAddr = match addr_str.parse() {
                                Ok(a) => a,
                                Err(e) => {
                                    eprintln_error("bind_error", &format!("invalid bind address: {e}"));
                                    return 1;
                                }
                            };
                            match tokio::net::TcpListener::bind(addr).await {
                                Ok(l) => {
                                    if candidate != port {
                                        tracing::info!(
                                            requested_port = port,
                                            actual_port = candidate,
                                            "preferred port {port} was occupied, using port {candidate} instead"
                                        );
                                    }
                                    found = Some((l, candidate));
                                    break;
                                }
                                Err(e) => {
                                    last_err = Some(e);
                                }
                            }
                        }
                        match found {
                            Some(pair) => pair,
                            None => {
                                eprintln_error(
                                    "bind_error",
                                    &format!(
                                        "failed to bind to ports {port}--{}; last error: {}",
                                        port.saturating_add(19),
                                        last_err.map(|e| e.to_string()).unwrap_or_default()
                                    ),
                                );
                                return 1;
                            }
                        }
                    };

                    let url = format!("http://{bind}:{actual_port}");
                    let addr = listener
                .local_addr()
                .expect("TCP listener address available after successful bind");
                    tracing::info!(%addr, "NeuronCite web server listening");
                    eprintln!();
                    eprintln!("  NeuronCite is running at: {url}");
                    eprintln!();

                    // GUI mode: native dialogs are available because the main
                    // thread runs tao's event loop, which processes the GCD
                    // dispatches that rfd uses to show NSOpenPanel on macOS.
                    let web_state =
                        neuroncite_web::WebState::with_native_dialogs(state, log_tx, true);
                    let app = neuroncite_web::build_web_router(web_state.app_state.clone(), web_state);

                    // --- End of server setup ---

                    // Signal the main thread that the server is ready and provide
                    // the URL with the actual bound port for the native window.
                    // If the main thread has already terminated (e.g. window
                    // creation failed), the send fails silently and the server
                    // continues running until Ctrl+C.
                    let _ = url_tx.send(url);

                    // Internal oneshot that tells axum to begin graceful
                    // shutdown. Separated from the external shutdown signals
                    // so the timeout applies only to the drain phase, not
                    // to the entire serving lifetime.
                    let (drain_tx, drain_rx) = tokio::sync::oneshot::channel::<()>();

                    // Spawn the server task. It serves requests until
                    // drain_rx fires, then begins draining in-flight
                    // connections.
                    let serve_task = tokio::spawn(async move {
                        axum::serve(listener, app)
                            .with_graceful_shutdown(async move {
                                let _ = drain_rx.await;
                            })
                            .await
                    });

                    // Wait for an external shutdown signal: either the native
                    // window is closed or Ctrl+C is pressed in the terminal.
                    // The server continues serving requests during this wait.
                    tokio::select! {
                        _ = shutdown_rx => {
                            tracing::info!("native window closed, shutting down server");
                        }
                        _ = shutdown_signal() => {
                            tracing::info!("received Ctrl+C, shutting down server");
                        }
                    }

                    // Tell axum to stop accepting connections and begin
                    // draining. Long-lived SSE connections (progress, logs,
                    // jobs, models) may never close on their own because
                    // nobody is reading them after the WebView is destroyed.
                    // The timeout below forces the server to exit if the
                    // drain phase stalls on these orphaned streams.
                    let _ = drain_tx.send(());

                    // 5 seconds is long enough for well-behaved connections
                    // to drain, short enough that the user does not perceive
                    // the app as "hanging" after clicking the close button.
                    const SHUTDOWN_DRAIN_SECS: u64 = 5;
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(SHUTDOWN_DRAIN_SECS),
                        serve_task,
                    )
                    .await
                    {
                        Ok(Ok(Ok(()))) => {
                            tracing::info!("web server shut down cleanly");
                            0
                        }
                        Ok(Ok(Err(e))) => {
                            eprintln_error(
                                "server_error",
                                &format!("server error: {e}"),
                            );
                            1
                        }
                        Ok(Err(join_err)) => {
                            eprintln_error(
                                "server_error",
                                &format!("server task join error: {join_err}"),
                            );
                            1
                        }
                        Err(_elapsed) => {
                            tracing::warn!(
                                timeout_secs = SHUTDOWN_DRAIN_SECS,
                                "server shutdown timed out after {SHUTDOWN_DRAIN_SECS}s \
                                 (likely open SSE streams), forcing exit"
                            );
                            0
                        }
                    }
                })
            })
            .expect("failed to spawn server thread");

        // Launch the splash screen on the main thread and hand it the URL
        // receiver. The splash appears immediately while the server finishes
        // startup in the background thread. Once the server sends the URL,
        // the splash transitions to the main application window.
        //
        // run_gui_with_splash blocks the main thread until the user closes
        // the main window (or the server fails during startup), then returns
        // control here. If GUI initialization fails after the pre-flight
        // check passed (rare edge case), the error is printed and the
        // server shuts down (shutdown_tx is dropped, causing shutdown_rx
        // to fire).
        match neuroncite_web::native_window::run_gui_with_splash(url_rx, shutdown_tx) {
            Ok(()) => {
                // Window was closed normally (or server failed during startup
                // and the splash exited). The shutdown signal was already sent
                // inside the close handler.
            }
            Err(e) => {
                eprintln_error(
                    "gui_init",
                    &format!("native GUI initialization failed: {e}"),
                );
                #[cfg(target_os = "linux")]
                eprintln!(
                    "Hint: try GDK_BACKEND=x11 neuroncite, or use neuroncite serve for headless mode"
                );
            }
        }

        // Wait for the server thread to complete its graceful shutdown.
        server_thread.join().unwrap_or(1)
    }

    // --- Non-GUI path: existing browser-based flow on the main thread ---
    #[cfg(not(feature = "gui"))]
    {
        rt.block_on(async {
            tracing::info!(port = port, bind = %bind, "starting NeuronCite in web UI mode");

            let data_dir = neuroncite_core::paths::gui_index_dir();

            let (state, _executor_handle) =
                match init_server_context(config, true, None, data_dir).await {
                    Ok(ctx) => ctx,
                    Err(code) => return code,
                };

            // Broadcast channels (progress_tx, citation_tx, source_tx) are
            // initialized unconditionally inside SseChannels::new(), which is
            // called from AppState::new() in init_server_context above. No
            // explicit channel creation or OnceLock::set calls are required here.

            let (listener, actual_port) = {
                let mut last_err = None;
                let mut found = None;
                for candidate in port..=port.saturating_add(19) {
                    let addr_str = format!("{bind}:{candidate}");
                    let addr: std::net::SocketAddr = match addr_str.parse() {
                        Ok(a) => a,
                        Err(e) => {
                            eprintln_error("bind_error", &format!("invalid bind address: {e}"));
                            return 1;
                        }
                    };
                    match tokio::net::TcpListener::bind(addr).await {
                        Ok(l) => {
                            if candidate != port {
                                tracing::info!(
                                    requested_port = port,
                                    actual_port = candidate,
                                    "preferred port {port} was occupied, using port {candidate} instead"
                                );
                            }
                            found = Some((l, candidate));
                            break;
                        }
                        Err(e) => {
                            last_err = Some(e);
                        }
                    }
                }
                match found {
                    Some(pair) => pair,
                    None => {
                        eprintln_error(
                            "bind_error",
                            &format!(
                                "failed to bind to ports {port}--{}; last error: {}",
                                port.saturating_add(19),
                                last_err.map(|e| e.to_string()).unwrap_or_default()
                            ),
                        );
                        return 1;
                    }
                }
            };

            let url = format!("http://{bind}:{actual_port}");
            let addr = listener
                .local_addr()
                .expect("TCP listener address available after successful bind");
            tracing::info!(%addr, "NeuronCite web server listening");
            eprintln!();
            eprintln!("  NeuronCite is running at: {url}");
            eprintln!("  Press Ctrl+C to stop the server.");
            eprintln!();

            let web_state = neuroncite_web::WebState::new(state, log_tx);
            let app = neuroncite_web::build_web_router(web_state.app_state.clone(), web_state);

            neuroncite_web::browser::open_browser(url);

            match axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await
            {
                Ok(()) => {
                    tracing::info!("web server shut down cleanly");
                    0
                }
                Err(e) => {
                    eprintln_error("server_error", &format!("server error: {e}"));
                    1
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// CLI indexing mode
// ---------------------------------------------------------------------------

/// Indexes a directory of PDF files synchronously from the CLI. Discovers PDFs,
/// extracts text, chunks, embeds, stores in SQLite, and builds an HNSW index.
/// Outputs a JSON summary with session ID, file count, chunk count, and elapsed
/// time.
#[allow(clippy::too_many_arguments)]
fn run_index(
    config_path: Option<PathBuf>,
    directory: PathBuf,
    model: String,
    strategy: String,
    chunk_size: usize,
    overlap: usize,
    ocr_language: String,
    quiet: bool,
    _progress: ProgressMode,
) -> i32 {
    let _config = load_config(config_path);

    if !directory.exists() {
        eprintln_error(
            "not_found",
            &format!("directory does not exist: {}", directory.display()),
        );
        return 1;
    }
    if !directory.is_dir() {
        eprintln_error(
            "not_directory",
            &format!("path is not a directory: {}", directory.display()),
        );
        return 1;
    }

    let start = std::time::Instant::now();

    // Discover PDF files.
    let pdfs = match neuroncite_pdf::discover_pdfs(&directory) {
        Ok(p) => p,
        Err(e) => {
            eprintln_error("discovery_error", &format!("{e}"));
            return 1;
        }
    };

    if !quiet {
        eprintln!("Discovered {} PDF files", pdfs.len());
    }

    // Initialize the database in the centralized NeuronCite index directory.
    // Each indexed PDF directory gets its own subfolder under
    // <Documents>/NeuronCite/indexes/<sanitized_path>/.
    let data_dir = neuroncite_core::paths::index_dir_for_path(&directory);
    if let Err(e) = std::fs::create_dir_all(&data_dir) {
        eprintln_error("db_init", &format!("failed to create data directory: {e}"));
        return 1;
    }
    let db_path = data_dir.join("index.db");

    let pool = match neuroncite_store::create_pool(&db_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln_error("db_init", &format!("failed to create database pool: {e}"));
            return 1;
        }
    };

    let conn = match pool.get() {
        Ok(c) => c,
        Err(e) => {
            eprintln_error("db_init", &format!("failed to obtain connection: {e}"));
            return 1;
        }
    };

    if let Err(e) = neuroncite_store::migrate(&conn) {
        eprintln_error("db_init", &format!("schema migration failed: {e}"));
        return 1;
    }

    // Create and load the embedding backend.
    let backends = neuroncite_embed::list_available_backends();
    let backend_name = backends
        .first()
        .map(|b| b.name.as_str())
        .unwrap_or(neuroncite_embed::DEFAULT_BACKEND_NAME);

    let mut backend = match neuroncite_embed::create_backend(backend_name) {
        Ok(b) => b,
        Err(e) => {
            eprintln_error("embed_init", &format!("failed to create backend: {e}"));
            return 1;
        }
    };

    if let Err(e) = backend.load_model(&model) {
        eprintln_error(
            "model_load",
            &format!("failed to load model '{model}': {e}"),
        );
        return 1;
    }

    let vector_dim = backend.vector_dimension();
    let tokenizer_json = backend.tokenizer_json();

    if !quiet {
        eprintln!(
            "Loaded model '{}' (dim={}, backend={}, GPU={})",
            model,
            vector_dim,
            backend.name(),
            backend.supports_gpu()
        );
    }

    // Create the indexing session.
    let index_config = neuroncite_core::IndexConfig {
        directory: directory.clone(),
        model_name: model.clone(),
        chunk_strategy: strategy.clone(),
        chunk_size: Some(chunk_size),
        chunk_overlap: Some(overlap),
        max_words: None,
        ocr_language,
        embedding_storage_mode: neuroncite_core::StorageMode::SqliteBlob,
        vector_dimension: vector_dim,
    };

    let session_id = match neuroncite_store::find_session(&conn, &index_config) {
        Ok(Some(id)) => {
            if !quiet {
                eprintln!("Reusing existing session {id}");
            }
            id
        }
        _ => {
            match neuroncite_store::create_session(&conn, &index_config, env!("CARGO_PKG_VERSION"))
            {
                Ok(id) => id,
                Err(e) => {
                    eprintln_error("session_error", &format!("failed to create session: {e}"));
                    return 1;
                }
            }
        }
    };

    // Create the chunking strategy. For the "token" strategy, the tokenizer
    // JSON obtained from the embedding backend is passed to the factory
    // function which deserializes it into a tokenizers::Tokenizer instance.
    let chunk_strategy = match neuroncite_chunk::create_strategy(
        &strategy,
        Some(chunk_size),
        Some(overlap),
        None,
        tokenizer_json.as_deref(),
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln_error(
                "chunk_error",
                &format!("failed to create chunking strategy: {e}"),
            );
            return 1;
        }
    };

    let mut total_chunks_created = 0_usize;

    // Phase 1 (parallel): Extract pages and chunk all PDFs using rayon.
    // PDF extraction and chunking are CPU-bound operations with no shared
    // mutable state, so they scale linearly with available CPU cores.
    if !quiet {
        eprintln!("Extracting and chunking {} PDFs in parallel...", pdfs.len());
    }

    let extracted = match run_extraction_phase(chunk_strategy.as_ref(), &pdfs, None) {
        Ok(results) => results,
        Err(e) => {
            eprintln!("Extraction phase failed: {e}");
            return 1;
        }
    };

    let mut successful: Vec<ExtractedFile> = Vec::new();
    for result in extracted {
        match result {
            Ok(ef) => {
                if !ef.chunks.is_empty() {
                    successful.push(ef);
                }
            }
            Err((path, e)) => {
                if !quiet {
                    eprintln!("  Skipping {}: {e}", path.display());
                }
            }
        }
    }

    if !quiet {
        eprintln!(
            "Extraction complete: {} of {} files produced chunks",
            successful.len(),
            pdfs.len()
        );
    }

    // Phase 2 (sequential): Embed chunks via GPU and insert into SQLite.
    // The embedding backend requires exclusive GPU access (mutex) and SQLite
    // is a single-writer database, so this phase runs sequentially.
    // Embedding vectors are persisted to SQLite and not accumulated in memory.
    let successful_count = successful.len();
    for (file_idx, ef) in successful.iter_mut().enumerate() {
        if !quiet {
            eprintln!(
                "[{}/{}] Embedding and storing: {} ({} chunks)",
                file_idx + 1,
                successful_count,
                ef.pdf_path.display(),
                ef.chunks.len()
            );
        }

        match embed_and_store_file(&conn, backend.as_ref(), ef, session_id) {
            Ok(result) => {
                total_chunks_created += result.chunks_created;
            }
            Err(e) => {
                if !quiet {
                    eprintln!("  Skipping: {e}");
                }
            }
        }

        // Release page text memory after database insertion.
        ef.pages = Vec::new();
    }

    // Build HNSW index from embeddings stored in SQLite.
    let raw_embeddings = neuroncite_store::load_embeddings_for_hnsw(&conn, session_id)
        .unwrap_or_else(|e| {
            eprintln!("Failed to load embeddings for HNSW: {e}");
            Vec::new()
        });

    if !raw_embeddings.is_empty() {
        if !quiet {
            eprintln!(
                "Building HNSW index ({} vectors, dim={})",
                raw_embeddings.len(),
                vector_dim
            );
        }

        let f32_vectors: Vec<(i64, Vec<f32>)> = raw_embeddings
            .into_iter()
            .map(|(id, bytes)| (id, bytes_to_f32_vec(&bytes)))
            .collect();
        let labeled: Vec<(i64, &[f32])> = f32_vectors
            .iter()
            .map(|(id, emb)| (*id, emb.as_slice()))
            .collect();
        let hnsw = match neuroncite_store::build_hnsw(&labeled, vector_dim) {
            Ok(idx) => idx,
            Err(e) => {
                if !quiet {
                    eprintln!("Failed to build HNSW index: {e}");
                }
                return 1;
            }
        };

        let hnsw_path = data_dir.join(format!("hnsw_{session_id}"));
        // Create the HNSW subdirectory because hnsw_rs file_dump writes
        // .hnsw.graph and .hnsw.data files inside this directory and panics
        // (via HnswIo::init) if the directory does not exist.
        if let Err(e) = std::fs::create_dir_all(&hnsw_path) {
            if !quiet {
                eprintln!("Failed to create HNSW directory: {e}");
            }
        } else {
            // Wrap in catch_unwind because hnsw_rs panics on I/O failures
            // (hnswio.rs:206) instead of returning Result errors.
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                neuroncite_store::serialize_hnsw(&hnsw, &hnsw_path)
            })) {
                Ok(Ok(())) => {
                    if !quiet {
                        eprintln!("HNSW index saved to {}", hnsw_path.display());
                    }
                }
                Ok(Err(e)) => {
                    if !quiet {
                        eprintln!("Failed to serialize HNSW index: {e}");
                    }
                }
                Err(_panic) => {
                    if !quiet {
                        eprintln!("HNSW serialization panicked (hnsw_rs internal error)");
                    }
                }
            }
        }
    }

    let elapsed = start.elapsed();

    let result = serde_json::json!({
        "session_id": session_id,
        "files_processed": pdfs.len(),
        "chunks_created": total_chunks_created,
        "elapsed_seconds": elapsed.as_secs_f64()
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&result).unwrap_or_default()
    );
    0
}

// ---------------------------------------------------------------------------
// CLI search mode
// ---------------------------------------------------------------------------

/// Holds pre-loaded search infrastructure: database pool, embedding backend,
/// and HNSW index. Created once by `init_search_infra` and reused across
/// multiple queries in batch mode. Single-query mode also uses this struct
/// to share the same initialization path.
struct SearchInfra {
    pool: r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>,
    backend: Box<dyn neuroncite_core::EmbeddingBackend>,
    hnsw_index: neuroncite_store::HnswIndex,
}

/// Initializes the search infrastructure: opens the database, migrates the
/// schema, loads the session's embedding model, and deserializes the HNSW
/// index from disk.
///
/// Returns `None` on failure (errors are printed to stderr via `eprintln_error`).
fn init_search_infra(directory: &Path, session_id: i64) -> Option<SearchInfra> {
    let data_dir = neuroncite_core::paths::index_dir_for_path(directory);
    let db_path = data_dir.join("index.db");

    if !db_path.exists() {
        eprintln_error(
            "db_not_found",
            &format!(
                "no index.db found for {} (expected at {})",
                directory.display(),
                db_path.display()
            ),
        );
        return None;
    }

    let pool = match neuroncite_store::create_pool(&db_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln_error("db_open", &format!("failed to open database: {e}"));
            return None;
        }
    };

    let conn = match pool.get() {
        Ok(c) => c,
        Err(e) => {
            eprintln_error("db_pool", &format!("failed to get connection: {e}"));
            return None;
        }
    };
    if let Err(e) = neuroncite_store::migrate(&conn) {
        eprintln_error("db_migrate", &format!("schema migration failed: {e}"));
        return None;
    }

    // Load session metadata to discover the embedding model name.
    let session = match neuroncite_store::get_session(&conn, session_id) {
        Ok(s) => s,
        Err(e) => {
            eprintln_error(
                "session_not_found",
                &format!("session {session_id} not found: {e}"),
            );
            return None;
        }
    };

    // Initialize embedding backend and load the model used during indexing.
    let backends = neuroncite_embed::list_available_backends();
    let backend_name = backends
        .first()
        .map(|b| b.name.as_str())
        .unwrap_or(neuroncite_embed::DEFAULT_BACKEND_NAME);
    let mut backend = match neuroncite_embed::create_backend(backend_name) {
        Ok(b) => b,
        Err(e) => {
            eprintln_error("backend_init", &format!("embedding backend failed: {e}"));
            return None;
        }
    };
    if let Err(e) = backend.load_model(&session.model_name) {
        eprintln_error(
            "model_load",
            &format!("failed to load model '{}': {e}", session.model_name),
        );
        return None;
    }

    // Deserialize the HNSW index from the session's directory.
    let hnsw_dir = data_dir.join(format!("hnsw_{session_id}"));
    let hnsw_index = match neuroncite_store::deserialize_hnsw(&hnsw_dir) {
        Ok(idx) => idx,
        Err(e) => {
            eprintln_error(
                "hnsw_load",
                &format!("failed to load HNSW index from {}: {e}", hnsw_dir.display()),
            );
            return None;
        }
    };

    Some(SearchInfra {
        pool,
        backend,
        hnsw_index,
    })
}

/// Executes a single search query using pre-loaded infrastructure. Embeds the
/// query, builds a SearchPipeline, and returns the results. Returns `Err` on
/// failure without aborting the process, allowing batch mode to continue with
/// subsequent queries.
fn execute_single_query(
    infra: &SearchInfra,
    query: &str,
    top_k: usize,
    hybrid: bool,
    _rerank: bool,
    session_id: i64,
) -> Result<Vec<neuroncite_core::SearchResult>, String> {
    let query_vec = infra
        .backend
        .embed_single(query)
        .map_err(|e| format!("query embedding failed: {e}"))?;

    let conn = infra
        .pool
        .get()
        .map_err(|e| format!("failed to get connection: {e}"))?;

    let keyword_limit = if hybrid { top_k * 5 } else { 0 };
    let config = neuroncite_search::SearchConfig {
        session_id,
        vector_top_k: top_k * 5,
        keyword_limit,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: false,
        simhash_threshold: 3,
        max_results: top_k,
        rerank_enabled: false,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    let pipeline = neuroncite_search::SearchPipeline::new(
        &infra.hnsw_index,
        &conn,
        &query_vec,
        query,
        None,
        config,
    );

    pipeline
        .search()
        .map(|outcome| outcome.results)
        .map_err(|e| format!("search pipeline error: {e}"))
}

/// Executes a single search query against an existing session.
///
/// Thin wrapper around `init_search_infra` + `execute_single_query` that
/// prints the results as pretty-printed JSON to stdout. This is the handler
/// for `neuroncite search --query "..."`.
fn run_search(
    directory: PathBuf,
    session_id: i64,
    query: String,
    top_k: usize,
    hybrid: bool,
    rerank: bool,
) -> i32 {
    let infra = match init_search_infra(&directory, session_id) {
        Some(i) => i,
        None => return 1,
    };

    match execute_single_query(&infra, &query, top_k, hybrid, rerank, session_id) {
        Ok(results) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&results).unwrap_or_default()
            );
            0
        }
        Err(e) => {
            eprintln_error("search_failed", &e);
            1
        }
    }
}

/// Processes a batch of search queries from a JSONL file or stdin.
///
/// Loads the embedding backend and HNSW index once (amortizing the ~2s GPU
/// init), then iterates over each input line. Each line is a JSON object with
/// a required "query" field and an optional "top_k" override. Results are
/// written as NDJSON (one JSON object per query) to stdout.
///
/// Per-query errors are reported in the output stream without aborting the
/// batch. Returns exit code 0 if at least one query succeeded, 1 otherwise.
fn run_batch_search(
    directory: PathBuf,
    session_id: i64,
    batch_path: String,
    top_k: usize,
    hybrid: bool,
    rerank: bool,
) -> i32 {
    let infra = match init_search_infra(&directory, session_id) {
        Some(i) => i,
        None => return 1,
    };

    // Open the input source: stdin when the path is "-", otherwise a file.
    let reader: Box<dyn std::io::BufRead> = if batch_path == "-" {
        Box::new(std::io::BufReader::new(std::io::stdin()))
    } else {
        let file = match std::fs::File::open(&batch_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln_error(
                    "batch_open",
                    &format!("failed to open batch file '{}': {e}", batch_path),
                );
                return 1;
            }
        };
        Box::new(std::io::BufReader::new(file))
    };

    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let mut query_index: usize = 0;
    let mut success_count: usize = 0;

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                let error_obj = serde_json::json!({
                    "query": null,
                    "query_index": query_index,
                    "error": format!("failed to read input line: {e}"),
                });
                let _ = writeln!(stdout_lock, "{error_obj}");
                query_index += 1;
                continue;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let parsed: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let error_obj = serde_json::json!({
                    "query": null,
                    "query_index": query_index,
                    "error": format!("invalid JSON on line {}: {e}", query_index),
                });
                let _ = writeln!(stdout_lock, "{error_obj}");
                query_index += 1;
                continue;
            }
        };

        let query_text = match parsed["query"].as_str() {
            Some(q) => q,
            None => {
                let error_obj = serde_json::json!({
                    "query": null,
                    "query_index": query_index,
                    "error": "missing or non-string 'query' field",
                });
                let _ = writeln!(stdout_lock, "{error_obj}");
                query_index += 1;
                continue;
            }
        };

        // Per-query top_k override; falls back to the CLI --top_k value.
        let q_top_k = parsed["top_k"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(top_k);

        match execute_single_query(&infra, query_text, q_top_k, hybrid, rerank, session_id) {
            Ok(results) => {
                let output = serde_json::json!({
                    "query": query_text,
                    "query_index": query_index,
                    "result_count": results.len(),
                    "results": results,
                });
                let _ = writeln!(stdout_lock, "{output}");
                success_count += 1;
            }
            Err(e) => {
                let error_obj = serde_json::json!({
                    "query": query_text,
                    "query_index": query_index,
                    "error": e,
                });
                let _ = writeln!(stdout_lock, "{error_obj}");
            }
        }

        query_index += 1;
    }

    if success_count > 0 { 0 } else { 1 }
}

// ---------------------------------------------------------------------------
// CLI utility subcommands
// ---------------------------------------------------------------------------

/// Runs the Dependency Doctor checks.
fn run_doctor() -> i32 {
    let report = serde_json::json!({
        "cuda": check_dependency("CUDA", &["nvcc", "nvidia-smi"]),
        "onnxruntime": check_dependency("ONNX Runtime", &["onnxruntime"]),
        "pdfium": check_dependency("PDFium", &["pdfium"]),
        "tesseract": check_dependency("Tesseract", &["tesseract"]),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_default()
    );
    0
}

/// Lists all index sessions in the specified directory's database.
///
/// Opens the .neuroncite database, queries all sessions ordered by creation
/// time (descending), and prints a JSON array to stdout. Each entry contains
/// the session id, model name, chunk strategy, vector dimension, directory
/// path, and creation timestamp.
fn run_sessions(directory: PathBuf) -> i32 {
    let data_dir = neuroncite_core::paths::index_dir_for_path(&directory);
    let db_path = data_dir.join("index.db");

    if !db_path.exists() {
        // No database means no sessions have been created yet. Print an
        // empty JSON array and exit successfully rather than treating this
        // as an error.
        println!("[]");
        return 0;
    }

    let pool = match neuroncite_store::create_pool(&db_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln_error("db_open", &format!("failed to open database: {e}"));
            return 1;
        }
    };
    let conn = match pool.get() {
        Ok(c) => c,
        Err(e) => {
            eprintln_error("db_pool", &format!("failed to get connection: {e}"));
            return 1;
        }
    };
    if let Err(e) = neuroncite_store::migrate(&conn) {
        eprintln_error("db_migrate", &format!("schema migration failed: {e}"));
        return 1;
    }

    let sessions = match neuroncite_store::list_sessions(&conn) {
        Ok(s) => s,
        Err(e) => {
            eprintln_error("list_sessions", &format!("failed to list sessions: {e}"));
            return 1;
        }
    };

    // Convert SessionRow (no Serialize) to JSON manually.
    let json_sessions: Vec<serde_json::Value> = sessions
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": s.id,
                "directory_path": s.directory_path,
                "model_name": s.model_name,
                "chunk_strategy": s.chunk_strategy,
                "chunk_size": s.chunk_size,
                "chunk_overlap": s.chunk_overlap,
                "vector_dimension": s.vector_dimension,
                "hnsw_total": s.hnsw_total,
                "created_at": s.created_at,
            })
        })
        .collect();

    println!(
        "{}",
        serde_json::to_string_pretty(&json_sessions).unwrap_or_default()
    );
    0
}

/// Exports search results to a file format.
///
/// Runs the same search pipeline as `run_search`, then formats the results
/// in the requested export format (markdown, bibtex, csl-json, ris,
/// plain-text) and writes them to either the specified output file or stdout.
fn run_export(
    directory: PathBuf,
    session_id: i64,
    query: String,
    export_format: ExportFormat,
    output: Option<PathBuf>,
) -> i32 {
    let data_dir = neuroncite_core::paths::index_dir_for_path(&directory);
    let db_path = data_dir.join("index.db");

    if !db_path.exists() {
        eprintln_error(
            "db_not_found",
            &format!(
                "no index.db found for {} (expected at {})",
                directory.display(),
                db_path.display()
            ),
        );
        return 1;
    }

    let pool = match neuroncite_store::create_pool(&db_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln_error("db_open", &format!("failed to open database: {e}"));
            return 1;
        }
    };
    let conn = match pool.get() {
        Ok(c) => c,
        Err(e) => {
            eprintln_error("db_pool", &format!("failed to get connection: {e}"));
            return 1;
        }
    };
    if let Err(e) = neuroncite_store::migrate(&conn) {
        eprintln_error("db_migrate", &format!("schema migration failed: {e}"));
        return 1;
    }

    let session = match neuroncite_store::get_session(&conn, session_id) {
        Ok(s) => s,
        Err(e) => {
            eprintln_error(
                "session_not_found",
                &format!("session {session_id} not found: {e}"),
            );
            return 1;
        }
    };

    let backends = neuroncite_embed::list_available_backends();
    let backend_name = backends
        .first()
        .map(|b| b.name.as_str())
        .unwrap_or(neuroncite_embed::DEFAULT_BACKEND_NAME);
    let mut backend = match neuroncite_embed::create_backend(backend_name) {
        Ok(b) => b,
        Err(e) => {
            eprintln_error("backend_init", &format!("embedding backend failed: {e}"));
            return 1;
        }
    };
    if let Err(e) = backend.load_model(&session.model_name) {
        eprintln_error(
            "model_load",
            &format!("failed to load model '{}': {e}", session.model_name),
        );
        return 1;
    }

    let query_vec = match backend.embed_single(&query) {
        Ok(v) => v,
        Err(e) => {
            eprintln_error("embed_query", &format!("query embedding failed: {e}"));
            return 1;
        }
    };

    let hnsw_dir = data_dir.join(format!("hnsw_{session_id}"));
    let hnsw_index = match neuroncite_store::deserialize_hnsw(&hnsw_dir) {
        Ok(idx) => idx,
        Err(e) => {
            eprintln_error(
                "hnsw_load",
                &format!("failed to load HNSW index from {}: {e}", hnsw_dir.display()),
            );
            return 1;
        }
    };

    let config = neuroncite_search::SearchConfig {
        session_id,
        vector_top_k: 50,
        keyword_limit: 50,
        ef_search: 100,
        rrf_k: 60,
        bm25_must_match: false,
        simhash_threshold: 3,
        max_results: 10,
        rerank_enabled: false,
        file_ids: None,
        min_score: None,
        page_start: None,
        page_end: None,
    };

    let pipeline = neuroncite_search::SearchPipeline::new(
        &hnsw_index,
        &conn,
        &query_vec,
        &query,
        None,
        config,
    );

    let results = match pipeline.search() {
        Ok(outcome) => outcome.results,
        Err(e) => {
            eprintln_error("search_failed", &format!("search pipeline error: {e}"));
            return 1;
        }
    };

    // Format results in the requested export format.
    let formatted = match export_format {
        ExportFormat::Markdown => format_results_markdown(&results),
        ExportFormat::Bibtex => format_results_bibtex(&results),
        ExportFormat::CslJson => match serde_json::to_string_pretty(&results) {
            Ok(s) => s,
            Err(e) => {
                eprintln_error("json_format", &format!("JSON formatting failed: {e}"));
                return 1;
            }
        },
        ExportFormat::Ris => format_results_ris(&results),
        ExportFormat::PlainText => format_results_plain_text(&results, &query),
    };

    // Write to output file or stdout.
    match output {
        Some(path) => {
            if let Some(parent) = path.parent()
                && let Err(e) = std::fs::create_dir_all(parent)
            {
                eprintln_error(
                    "output_dir",
                    &format!("failed to create output directory: {e}"),
                );
                return 1;
            }
            if let Err(e) = std::fs::write(&path, &formatted) {
                eprintln_error(
                    "output_write",
                    &format!("failed to write to {}: {e}", path.display()),
                );
                return 1;
            }
            eprintln!("Exported to {}", path.display());
        }
        None => {
            println!("{formatted}");
        }
    }
    0
}

/// Formats search results as a Markdown numbered list. Each entry contains
/// rank, score, citation string, source file, page range, and a text excerpt.
fn format_results_markdown(results: &[neuroncite_core::SearchResult]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(results.len() * 256);
    out.push_str("# Search Results\n\n");
    for (i, r) in results.iter().enumerate() {
        let excerpt: String = r.content.chars().take(200).collect();
        let _ = writeln!(
            out,
            "{}. **Score: {:.3}** -- {}\n   > {}\n",
            i + 1,
            r.score,
            r.citation.formatted,
            excerpt,
        );
    }
    out
}

/// Formats search results as BibTeX @misc entries. Each entry uses
/// "neuroncite_N" as the citation key (1-indexed). The `note` field contains
/// the score and page range; the `abstract` field contains a text excerpt.
fn format_results_bibtex(results: &[neuroncite_core::SearchResult]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(results.len() * 512);
    for (i, r) in results.iter().enumerate() {
        let key = format!("neuroncite_{}", i + 1);
        let title = r
            .citation
            .file_display_name
            .replace('{', "\\{")
            .replace('}', "\\}")
            .replace('&', "\\&");
        let excerpt: String = r.content.chars().take(500).collect();
        let abstract_text = excerpt
            .replace('{', "\\{")
            .replace('}', "\\}")
            .replace('&', "\\&");
        let _ = writeln!(out, "@misc{{{key},");
        let _ = writeln!(out, "  title = {{{title}}},");
        let _ = writeln!(
            out,
            "  note = {{Score: {:.3}, pages {}-{}}},",
            r.score, r.citation.page_start, r.citation.page_end,
        );
        let _ = writeln!(out, "  abstract = {{{abstract_text}}},");
        let _ = writeln!(out, "}}\n");
    }
    out
}

/// Formats search results as RIS (Research Information Systems) tagged records.
/// RIS is a plain-text format supported by reference managers (EndNote, Zotero,
/// Mendeley). Each record begins with TY (type tag) and ends with ER (end record).
/// The file display name is used as the title (T1 tag). Score and page range
/// are stored in the N1 (notes) tag.
fn format_results_ris(results: &[neuroncite_core::SearchResult]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(results.len() * 256);
    for r in results {
        out.push_str("TY  - GEN\n");
        let _ = writeln!(out, "T1  - {}", r.citation.file_display_name);
        let _ = writeln!(
            out,
            "SP  - {}\nEP  - {}",
            r.citation.page_start, r.citation.page_end,
        );
        let _ = writeln!(out, "N1  - Score: {:.4}", r.score);
        let _ = writeln!(out, "L1  - {}", r.citation.source_file.display());
        out.push_str("ER  - \n\n");
    }
    out
}

/// Formats search results as plain text with source citations. Each result
/// is numbered and includes the passage content, source file, page range, and
/// score. Intended for quick human-readable output without structured metadata.
fn format_results_plain_text(results: &[neuroncite_core::SearchResult], query: &str) -> String {
    use std::fmt::Write;
    let mut out = format!("Search: {query}\n");
    let _ = writeln!(out, "Results: {}", results.len());
    out.push_str(&"=".repeat(60));
    out.push('\n');

    for (i, r) in results.iter().enumerate() {
        let _ = writeln!(out, "\n[{}] Score: {:.4}", i + 1, r.score);
        let _ = writeln!(
            out,
            "Source: {} (pp. {}-{})",
            r.citation.source_file.display(),
            r.citation.page_start,
            r.citation.page_end,
        );
        let _ = writeln!(out, "{}", r.content);
    }
    out
}

// ---------------------------------------------------------------------------
// Model management CLI subcommands
// ---------------------------------------------------------------------------

/// Dispatches the `models` subcommand to the appropriate handler function.
/// The List, Info, Download, and Verify sub-commands require the `backend-ort`
/// feature because they access the ONNX model catalog. System always compiles.
fn run_models(action: ModelsCommand, format: OutputFormat) -> i32 {
    match action {
        #[cfg(feature = "backend-ort")]
        ModelsCommand::List => run_models_list(format),
        #[cfg(feature = "backend-ort")]
        ModelsCommand::Info { model_id } => run_models_info(&model_id, format),
        #[cfg(feature = "backend-ort")]
        ModelsCommand::Download { model_id } => run_models_download(&model_id),
        #[cfg(feature = "backend-ort")]
        ModelsCommand::Verify { model_id } => run_models_verify(&model_id),
        ModelsCommand::System => run_models_system(format),
        // When backend-ort is not enabled, commands that require the model
        // catalog are unavailable. The binary still compiles so that headless
        // or web-only builds remain functional.
        #[cfg(not(feature = "backend-ort"))]
        _ => {
            eprintln_error(
                "feature_disabled",
                "this command requires the backend-ort feature enabled at compile time",
            );
            1
        }
    }
}

/// Lists all available embedding models with their configuration parameters
/// and local cache status. Iterates the static model catalog from
/// `supported_model_configs()` and checks each model's cache presence via
/// `is_cached()`. The JSON output includes a `models` array, `total` count,
/// and `cached_count` for quick agent decision-making.
#[cfg(feature = "backend-ort")]
fn run_models_list(format: OutputFormat) -> i32 {
    let configs = neuroncite_embed::supported_model_configs();
    let mut models_json = Vec::new();
    let mut cached_count = 0usize;

    for config in configs {
        let cached = neuroncite_embed::is_cached(&config.model_id, "main");
        if cached {
            cached_count += 1;
        }
        let cache_path = if cached {
            Some(
                neuroncite_embed::model_dir(&config.model_id, "main")
                    .to_string_lossy()
                    .to_string(),
            )
        } else {
            None
        };

        models_json.push(serde_json::json!({
            "model_id": config.model_id,
            "display_name": config.display_name,
            "vector_dimension": config.vector_dimension,
            "max_seq_len": config.max_seq_len,
            "pooling": format!("{}", config.pooling),
            "cached": cached,
            "cache_path": cache_path,
        }));
    }

    let total = configs.len();

    match format {
        OutputFormat::Json => {
            let output = serde_json::json!({
                "models": models_json,
                "total": total,
                "cached_count": cached_count,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&output).unwrap_or_default()
            );
        }
        OutputFormat::Text => {
            println!("Available embedding models:");
            for config in configs {
                let cached = neuroncite_embed::is_cached(&config.model_id, "main");
                let marker = if cached { "cached" } else { "      " };
                println!(
                    "  [{}] {:<45} ({}d, {}, {} seq)",
                    marker,
                    config.model_id,
                    config.vector_dimension,
                    config.pooling,
                    config.max_seq_len
                );
            }
            println!("\n{cached_count} of {total} models cached");
        }
    }

    0
}

/// Shows detailed configuration for a specific embedding model. Outputs all
/// fields from `EmbeddingModelConfig` plus the per-file cache status
/// (existence and size) for the model's expected files.
#[cfg(feature = "backend-ort")]
fn run_models_info(model_id: &str, format: OutputFormat) -> i32 {
    let config = match neuroncite_embed::find_model_config(model_id) {
        Some(c) => c,
        None => {
            eprintln_error(
                "model_not_found",
                &format!("model '{}' is not in the supported model catalog", model_id),
            );
            return 1;
        }
    };

    let cached = neuroncite_embed::is_cached(model_id, "main");
    let model_dir = neuroncite_embed::model_dir(model_id, "main");
    let cache_path = if cached {
        Some(model_dir.to_string_lossy().to_string())
    } else {
        None
    };

    // Build the per-file status array. Uses model_expected_files to determine
    // which files should be present in the cache directory.
    let mut files_json = Vec::new();
    if let Some(expected_files) = neuroncite_embed::model_expected_files(model_id) {
        for file_name in expected_files {
            let file_path = model_dir.join(file_name);
            let exists = file_path.exists();
            let size_bytes = if exists {
                std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0)
            } else {
                0
            };
            files_json.push(serde_json::json!({
                "name": file_name,
                "exists": exists,
                "size_bytes": size_bytes,
            }));
        }
    }

    match format {
        OutputFormat::Json => {
            let output = serde_json::json!({
                "model_id": config.model_id,
                "display_name": config.display_name,
                "vector_dimension": config.vector_dimension,
                "max_seq_len": config.max_seq_len,
                "pooling": format!("{}", config.pooling),
                "uses_token_type_ids": config.uses_token_type_ids,
                "query_prefix": config.query_prefix,
                "document_prefix": config.document_prefix,
                "cached": cached,
                "cache_path": cache_path,
                "files": files_json,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&output).unwrap_or_default()
            );
        }
        OutputFormat::Text => {
            println!("Model: {} ({})", config.display_name, config.model_id);
            println!("  Dimension:       {}", config.vector_dimension);
            println!("  Max Seq Length:   {}", config.max_seq_len);
            println!("  Pooling:         {}", config.pooling);
            println!("  Token Type IDs:  {}", config.uses_token_type_ids);
            if !config.query_prefix.is_empty() {
                println!("  Query Prefix:    {:?}", config.query_prefix);
            }
            if !config.document_prefix.is_empty() {
                println!("  Document Prefix: {:?}", config.document_prefix);
            }
            println!("  Cached:          {}", cached);
            if let Some(ref path) = cache_path {
                println!("  Cache Path:      {}", path);
            }
            if !files_json.is_empty() {
                println!("  Files:");
                for f in &files_json {
                    let name = f["name"].as_str().unwrap_or("?");
                    let exists = f["exists"].as_bool().unwrap_or(false);
                    let size = f["size_bytes"].as_u64().unwrap_or(0);
                    if exists {
                        println!("    {} ({} bytes)", name, size);
                    } else {
                        println!("    {} (missing)", name);
                    }
                }
            }
        }
    }

    0
}

/// Downloads a model's files to the local cache. If the model is already
/// cached, reports the existing path without re-downloading. Measures
/// elapsed time for fresh downloads to help agents estimate download
/// durations for larger models.
#[cfg(feature = "backend-ort")]
fn run_models_download(model_id: &str) -> i32 {
    // Validate the model ID against the catalog.
    if neuroncite_embed::find_model_config(model_id).is_none() {
        eprintln_error(
            "model_not_found",
            &format!("model '{}' is not in the supported model catalog", model_id),
        );
        return 1;
    }

    // Check if already cached.
    if neuroncite_embed::is_cached(model_id, "main") {
        let cache_path = neuroncite_embed::model_dir(model_id, "main");
        let output = serde_json::json!({
            "model_id": model_id,
            "status": "already_cached",
            "cache_path": cache_path.to_string_lossy(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
        return 0;
    }

    // Download the model and measure elapsed time.
    let start = std::time::Instant::now();
    match neuroncite_embed::download_model(model_id, "main") {
        Ok(cache_path) => {
            let elapsed = start.elapsed().as_secs_f64();
            let file_count = neuroncite_embed::model_expected_files(model_id)
                .map(|f| f.len())
                .unwrap_or(0);
            let output = serde_json::json!({
                "model_id": model_id,
                "status": "downloaded",
                "cache_path": cache_path.to_string_lossy(),
                "files_downloaded": file_count,
                "elapsed_seconds": (elapsed * 100.0).round() / 100.0,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&output).unwrap_or_default()
            );
            0
        }
        Err(e) => {
            eprintln_error("download_failed", &format!("{e}"));
            1
        }
    }
}

/// Verifies the integrity of a cached model by checking that all expected
/// files exist, computing their SHA-256 checksums, and reporting their sizes.
/// The `valid` field is true when all expected files exist and have non-zero
/// size.
#[cfg(feature = "backend-ort")]
fn run_models_verify(model_id: &str) -> i32 {
    // Validate the model ID against the catalog.
    if neuroncite_embed::find_model_config(model_id).is_none() {
        eprintln_error(
            "model_not_found",
            &format!("model '{}' is not in the supported model catalog", model_id),
        );
        return 1;
    }

    let cached = neuroncite_embed::is_cached(model_id, "main");
    let model_dir = neuroncite_embed::model_dir(model_id, "main");

    if !cached {
        let output = serde_json::json!({
            "model_id": model_id,
            "cached": false,
            "files": [],
            "valid": false,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
        return 0;
    }

    let expected_files = neuroncite_embed::model_expected_files(model_id).unwrap_or_default();
    let mut files_json = Vec::new();
    let mut all_valid = true;

    for file_name in &expected_files {
        let file_path = model_dir.join(file_name);
        let exists = file_path.exists();
        let size_bytes = if exists {
            std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        // Compute SHA-256 for existing files.
        let sha256 = if exists {
            neuroncite_embed::compute_sha256(&file_path).unwrap_or_default()
        } else {
            String::new()
        };

        if !exists || size_bytes == 0 {
            all_valid = false;
        }

        files_json.push(serde_json::json!({
            "name": file_name,
            "exists": exists,
            "size_bytes": size_bytes,
            "sha256": sha256,
        }));
    }

    let output = serde_json::json!({
        "model_id": model_id,
        "cached": true,
        "files": files_json,
        "valid": all_valid,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );

    0
}

/// Shows system capabilities relevant to embedding model selection: GPU
/// hardware detection, CUDA execution provider availability, compiled
/// backends with their model counts, and cache directory information
/// including disk usage.
fn run_models_system(format: OutputFormat) -> i32 {
    // GPU detection uses platform-gated functions. On non-Windows or without
    // the backend-ort feature, these return false.
    let nvidia_detected = {
        #[cfg(all(not(target_os = "macos"), feature = "backend-ort"))]
        {
            neuroncite_embed::detect_nvidia_gpu()
        }
        #[cfg(any(target_os = "macos", not(feature = "backend-ort")))]
        {
            false
        }
    };

    let cuda_available = {
        #[cfg(feature = "backend-ort")]
        {
            neuroncite_embed::is_cuda_available()
        }
        #[cfg(not(feature = "backend-ort"))]
        {
            false
        }
    };

    let coreml_available = {
        #[cfg(feature = "backend-ort")]
        {
            neuroncite_embed::is_coreml_available()
        }
        #[cfg(not(feature = "backend-ort"))]
        {
            false
        }
    };

    let gpu_ort_runtime = {
        #[cfg(all(not(target_os = "macos"), feature = "backend-ort"))]
        {
            neuroncite_embed::is_gpu_ort_runtime()
        }
        #[cfg(any(target_os = "macos", not(feature = "backend-ort")))]
        {
            false
        }
    };

    let backends = neuroncite_embed::list_available_backends();
    let backends_json: Vec<serde_json::Value> = backends
        .iter()
        .map(|b| {
            serde_json::json!({
                "name": b.name,
                "model_count": b.models.len(),
                "gpu_supported": b.gpu_supported,
            })
        })
        .collect();

    let cache_dir = neuroncite_embed::cache_dir();
    let cache_dir_exists = cache_dir.exists();
    let cache_disk_usage_bytes = dir_disk_usage(&cache_dir);

    match format {
        OutputFormat::Json => {
            let output = serde_json::json!({
                "gpu": {
                    "nvidia_detected": nvidia_detected,
                    "cuda_available": cuda_available,
                    "coreml_available": coreml_available,
                    "gpu_ort_runtime": gpu_ort_runtime,
                },
                "backends": backends_json,
                "cache_dir": cache_dir.to_string_lossy(),
                "cache_dir_exists": cache_dir_exists,
                "cache_disk_usage_bytes": cache_disk_usage_bytes,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&output).unwrap_or_default()
            );
        }
        OutputFormat::Text => {
            println!(
                "GPU: NVIDIA detected={}, CUDA EP={}, CoreML={}, GPU ORT={}",
                nvidia_detected, cuda_available, coreml_available, gpu_ort_runtime
            );
            for b in &backends {
                println!(
                    "Backend: {} ({} models, GPU {})",
                    b.name,
                    b.models.len(),
                    if b.gpu_supported {
                        "supported"
                    } else {
                        "not supported"
                    }
                );
            }
            let usage_mb = cache_disk_usage_bytes as f64 / (1024.0 * 1024.0);
            println!("Cache: {} ({:.1} MB)", cache_dir.display(), usage_mb);
        }
    }

    0
}

/// Recursively computes the total size in bytes of all files under the given
/// directory path. Returns 0 if the directory does not exist or is unreadable.
/// Used by `run_models_system` to report cache disk usage.
fn dir_disk_usage(path: &std::path::Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += dir_disk_usage(&p);
            } else if let Ok(meta) = std::fs::metadata(&p) {
                total += meta.len();
            }
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Annotate mode
// ---------------------------------------------------------------------------

/// Runs the PDF annotation pipeline synchronously from CLI arguments.
/// Reads the input file, parses annotation instructions, matches PDFs,
/// locates quoted text, creates highlight and comment annotations, and
/// saves annotated copies to the output directory. Prints the JSON report
/// to stdout on completion.
fn run_annotate(
    input: PathBuf,
    source_dir: PathBuf,
    output_dir: PathBuf,
    default_color: String,
) -> i32 {
    // Read the input file.
    let input_data = match std::fs::read(&input) {
        Ok(data) => data,
        Err(e) => {
            eprintln_error(
                "input_read",
                &format!("failed to read input file {}: {e}", input.display()),
            );
            return 1;
        }
    };

    // Parse the annotation instructions (CSV or JSON auto-detected).
    let rows = match neuroncite_annotate::parse_input(&input_data) {
        Ok(r) => r,
        Err(e) => {
            eprintln_error("input_parse", &format!("failed to parse input: {e}"));
            return 1;
        }
    };

    if rows.is_empty() {
        eprintln_error("input_empty", "input file contains no annotation rows");
        return 1;
    }

    // Verify the source directory exists.
    if !source_dir.is_dir() {
        eprintln_error(
            "source_dir",
            &format!("source directory does not exist: {}", source_dir.display()),
        );
        return 1;
    }

    // Create the output directory if it does not exist.
    if let Err(e) = std::fs::create_dir_all(&output_dir) {
        eprintln_error(
            "output_dir",
            &format!("failed to create output directory: {e}"),
        );
        return 1;
    }

    let error_dir = output_dir.join("errors");
    let total_quotes = rows.len();
    eprintln!(
        "Annotating {} quotes from {} PDFs in {}",
        total_quotes,
        source_dir.display(),
        output_dir.display()
    );

    let config = neuroncite_annotate::AnnotateConfig {
        input_rows: rows,
        source_directory: source_dir,
        output_directory: output_dir,
        error_directory: error_dir,
        default_color,
        // The CLI does not have access to an indexed session's database, so
        // no cached page texts are available. The pipeline falls back to
        // live extraction via neuroncite_pdf::extract_pages().
        cached_page_texts: std::collections::HashMap::new(),
        // The CLI does not support append mode; each run is a fresh annotation.
        prior_output_directory: None,
    };

    let report = match neuroncite_annotate::annotate_pdfs(config, |done, total| {
        eprintln!("[{done}/{total}] processing PDFs...");
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln_error("annotate", &format!("annotation pipeline failed: {e}"));
            return 1;
        }
    };

    // Print the JSON report to stdout.
    match serde_json::to_writer_pretty(std::io::stdout(), &report) {
        Ok(()) => {
            // Trailing newline for clean terminal output.
            println!();
        }
        Err(e) => {
            eprintln_error("report", &format!("failed to serialize report: {e}"));
            return 1;
        }
    }

    eprintln!(
        "Annotation complete: {}/{} quotes matched across {} PDFs",
        report.summary.quotes_matched, report.summary.total_quotes, report.summary.total_pdfs
    );

    0
}

/// Prints version information.
fn run_version(format: OutputFormat) -> i32 {
    let version_info = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "build_features": built_features(),
    });

    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&version_info).unwrap_or_default()
            );
        }
        OutputFormat::Text => {
            println!(
                "NeuronCite v{} (features: {})",
                env!("CARGO_PKG_VERSION"),
                built_features().join(", ")
            );
        }
    }
    0
}

// ---------------------------------------------------------------------------
// MCP server mode
// ---------------------------------------------------------------------------

/// Dispatches the `mcp` subcommand to the appropriate action (serve, install,
/// uninstall, status). The `serve` action initializes the embedding backend,
/// database pool, and GPU worker, then runs the MCP stdio server loop. The
/// `install`/`uninstall`/`status` actions manage the registration entry in
/// Claude Code's global settings file (~/.claude/settings.json).
#[cfg(feature = "mcp")]
fn run_mcp(config_path: Option<PathBuf>, action: McpCommand) -> i32 {
    match action {
        McpCommand::Serve { model } => run_mcp_serve(config_path, model),
        McpCommand::Install { target } => run_mcp_install(&target),
        McpCommand::Uninstall { target } => run_mcp_uninstall(&target),
        McpCommand::Status { target } => run_mcp_status(target.as_deref()),
    }
}

/// Starts the MCP stdio server. Initializes all backend subsystems (database,
/// embedding backend, GPU worker) and runs the JSON-RPC 2.0 server loop on
/// stdin/stdout. All tracing output is routed to stderr to avoid interfering
/// with the JSON-RPC protocol on stdout.
///
/// The `model_override` parameter, when set, takes precedence over the
/// config file's `default_model` field. This allows Claude Code to pass
/// `--model` via the MCP server args in `settings.json`.
#[cfg(feature = "mcp")]
fn run_mcp_serve(config_path: Option<PathBuf>, model_override: Option<String>) -> i32 {
    let config = load_config(config_path);

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln_error(
                "runtime_init",
                &format!("failed to create tokio runtime: {e}"),
            );
            return 1;
        }
    };

    // Initialize the database in the centralized NeuronCite index directory
    // for the current working directory (Claude Code sets this to the project root).
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let data_dir = neuroncite_core::paths::index_dir_for_path(&cwd);
    if let Err(e) = std::fs::create_dir_all(&data_dir) {
        eprintln_error("db_init", &format!("failed to create data directory: {e}"));
        return 1;
    }
    let db_path = data_dir.join("index.db");

    let pool = match neuroncite_store::create_pool(&db_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln_error("db_init", &format!("failed to create database pool: {e}"));
            return 1;
        }
    };

    {
        let conn = match pool.get() {
            Ok(c) => c,
            Err(e) => {
                eprintln_error("db_init", &format!("failed to obtain connection: {e}"));
                return 1;
            }
        };
        if let Err(e) = neuroncite_store::migrate(&conn) {
            eprintln_error("db_init", &format!("schema migration failed: {e}"));
            return 1;
        }
    }

    // Suppress stdout during ONNX Runtime initialization. The CUDA execution
    // provider's C++ internals may write diagnostic messages (node assignment
    // warnings, memory allocation logs) directly to C-level stdout (fd 1),
    // bypassing the Rust tracing subscriber. In the MCP stdio transport,
    // stdout carries JSON-RPC messages; any foreign output corrupts the protocol.
    let saved_stdout = suppress_stdout();

    // Determine which embedding model to load. CLI --model flag takes
    // precedence over the config file's default_model field.
    let model_id = model_override.unwrap_or_else(|| config.default_model.clone());
    tracing::info!(model = %model_id, "MCP server loading embedding model");

    // Create the embedding backend and load the selected model. Model loading
    // failure is fatal because the MCP server cannot serve search or indexing
    // requests without a loaded model. Continuing with the unloaded backend
    // would silently produce dimension=384 (the OrtEmbedder::unloaded default)
    // for all sessions, corrupting the vector index.
    let backend: Arc<dyn neuroncite_core::EmbeddingBackend> = {
        let backends = neuroncite_embed::list_available_backends();
        let backend_name = backends
            .first()
            .map(|b| b.name.as_str())
            .unwrap_or(neuroncite_embed::DEFAULT_BACKEND_NAME);
        match neuroncite_embed::create_backend(backend_name) {
            Ok(mut b) => {
                if let Err(e) = b.load_model(&model_id) {
                    if let Some(fd) = saved_stdout {
                        restore_stdout(fd);
                    }
                    eprintln_error(
                        "model_load",
                        &format!("failed to load model '{}': {e}", model_id),
                    );
                    return 1;
                }
                Arc::from(b)
            }
            Err(e) => {
                if let Some(fd) = saved_stdout {
                    restore_stdout(fd);
                }
                eprintln_error("embed_init", &format!("failed to create backend: {e}"));
                return 1;
            }
        }
    };

    let vector_dim = backend.vector_dimension();

    // Keep fd 1 permanently redirected to NUL instead of restoring it.
    // The background job executor runs pdf extraction via rayon, and the
    // pdf-extract crate prints debug output ("Unicode mismatch") to C-level
    // stdout (fd 1). If fd 1 pointed to the real stdout pipe, this garbage
    // would interleave with MCP JSON-RPC messages and corrupt the transport.
    //
    // Instead, the MCP server writes responses through a Rust File created
    // from the saved fd (which still points to the original stdout pipe).
    // This separation ensures:
    //   - pdf-extract output goes to NUL (fd 1) and is silently discarded.
    //   - MCP JSON-RPC responses reach the client through the saved fd.
    //   - No race conditions between the extraction thread pool and the
    //     MCP server loop, because they write to different file descriptors.
    let mcp_writer_file = saved_stdout.map(|fd| {
        use neuroncite_api::indexer::writer_from_saved_fd;
        writer_from_saved_fd(fd)
    });

    // Enter the tokio runtime context before spawning the worker. spawn_worker
    // calls tokio::spawn internally, which requires an active reactor. The
    // runtime guard keeps the context active for the synchronous MCP server
    // loop that follows. Without this guard, tokio::spawn panics with
    // "there is no reactor running".
    let _guard = rt.enter();
    let worker_handle = neuroncite_api::spawn_worker(backend, None);

    // AppState::new returns Err if the bearer token is shorter than the minimum
    // length. In the MCP path no bearer token is passed, so this must succeed.
    let state = neuroncite_api::AppState::new(pool, worker_handle, config, true, None, vector_dim)
        .expect("AppState construction failed in MCP path (no token configured)");

    // Load HNSW indexes for all sessions that have embeddings stored in
    // SQLite. This populates the per-session HNSW map from previously
    // indexed data so search is available immediately after MCP server restart.
    neuroncite_api::load_all_session_hnsw(&state);

    // Spawn the background job executor. It polls the database for queued
    // indexing jobs and processes them through the shared pipeline, using
    // the WorkerHandle for GPU embedding (low priority channel) so that
    // interactive search queries retain responsiveness.
    let _executor_handle = neuroncite_api::spawn_job_executor(state.clone());

    // Run the MCP server loop on stdin with the MCP writer. When stdout
    // suppression succeeded, the writer uses the saved fd that bypasses
    // the NUL-redirected fd 1. When suppression was not available (non-Windows
    // or failure), fall back to regular stdout.
    let stdin = std::io::stdin();
    let mut reader = std::io::BufReader::new(stdin.lock());

    match mcp_writer_file {
        Some(ref file) => {
            // Borrow the File via Deref so BufWriter holds &File, not File.
            // writer_from_saved_fd returns ManuallyDrop<File> to prevent
            // a double-close: the CRT fd retains HANDLE ownership.
            let mut writer = std::io::BufWriter::new(&**file);
            neuroncite_mcp::server::run_server(&mut reader, &mut writer, state, rt.handle());
        }
        None => {
            let stdout = std::io::stdout();
            let mut writer = stdout.lock();
            neuroncite_mcp::server::run_server(&mut reader, &mut writer, state, rt.handle());
        }
    }

    0
}

/// Registers the NeuronCite MCP server in Claude Code's global settings file.
#[cfg(feature = "mcp")]
fn run_mcp_install(target_str: &str) -> i32 {
    let target = match neuroncite_mcp::McpTarget::from_cli_str(target_str) {
        Ok(t) => t,
        Err(e) => {
            eprintln_error("mcp_install", &e);
            return 1;
        }
    };
    match neuroncite_mcp::registration::install(None, target) {
        Ok(msg) => {
            let output = serde_json::json!({
                "status": "installed",
                "target": target.cli_name(),
                "message": msg,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&output).unwrap_or_default()
            );
            0
        }
        Err(e) => {
            eprintln_error("mcp_install", &e);
            1
        }
    }
}

/// Removes the NeuronCite MCP server entry from the specified client's config.
#[cfg(feature = "mcp")]
fn run_mcp_uninstall(target_str: &str) -> i32 {
    let target = match neuroncite_mcp::McpTarget::from_cli_str(target_str) {
        Ok(t) => t,
        Err(e) => {
            eprintln_error("mcp_uninstall", &e);
            return 1;
        }
    };
    match neuroncite_mcp::registration::uninstall(target) {
        Ok(msg) => {
            let output = serde_json::json!({
                "status": "uninstalled",
                "target": target.cli_name(),
                "message": msg,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&output).unwrap_or_default()
            );
            0
        }
        Err(e) => {
            eprintln_error("mcp_uninstall", &e);
            1
        }
    }
}

/// Displays the current MCP registration status. When `target_str` is `None`,
/// shows status for all targets.
#[cfg(feature = "mcp")]
fn run_mcp_status(target_str: Option<&str>) -> i32 {
    let targets: Vec<neuroncite_mcp::McpTarget> = match target_str {
        Some(s) => match neuroncite_mcp::McpTarget::from_cli_str(s) {
            Ok(t) => vec![t],
            Err(e) => {
                eprintln_error("mcp_status", &e);
                return 1;
            }
        },
        None => neuroncite_mcp::McpTarget::all().to_vec(),
    };

    let results: Vec<serde_json::Value> = targets
        .iter()
        .map(|t| {
            let status = neuroncite_mcp::registration::check_status(*t);
            serde_json::json!({
                "target": t.cli_name(),
                "display_name": t.display_name(),
                "registered": status.registered,
                "exe_path": status.exe_path,
                "args": status.args,
                "config_path": status.config_path,
            })
        })
        .collect();

    let output = if results.len() == 1 {
        results.into_iter().next().unwrap()
    } else {
        serde_json::json!({ "targets": results })
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
    0
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Loads the application configuration from the given path, or from the
/// default search locations if no path is specified.
fn load_config(config_path: Option<PathBuf>) -> neuroncite_core::AppConfig {
    if let Some(path) = config_path {
        match neuroncite_core::AppConfig::load_from_file(&path) {
            Ok(config) => config,
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    "failed to load configuration file, using defaults: {e}"
                );
                neuroncite_core::AppConfig::default()
            }
        }
    } else {
        neuroncite_core::AppConfig::default()
    }
}

/// Returns the list of compile-time feature flags active in this binary.
/// Each conditional push is gated on a Cargo feature flag defined in the
/// binary crate's \[features\] table. The resulting list is used by the CLI
/// `version` subcommand and the JSON output of `run_version`.
#[allow(clippy::vec_init_then_push)]
fn built_features() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut features = Vec::new();
    #[cfg(feature = "backend-ort")]
    features.push("backend-ort");
    #[cfg(feature = "web")]
    features.push("web");
    #[cfg(feature = "gui")]
    features.push("gui");
    #[cfg(feature = "mcp")]
    features.push("mcp");
    #[cfg(feature = "pdfium")]
    features.push("pdfium");
    #[cfg(feature = "ocr")]
    features.push("ocr");
    features
}

/// Writes a JSON error object to stderr. Used by CLI subcommands to report
/// application errors in a machine-readable format.
fn eprintln_error(code: &str, message: &str) {
    let error = serde_json::json!({
        "error": message,
        "code": code,
    });
    eprintln!("{}", serde_json::to_string(&error).unwrap_or_default());
}

/// Checks whether a runtime dependency is available by probing the system
/// PATH for known executable names.
fn check_dependency(name: &str, executables: &[&str]) -> serde_json::Value {
    for exe in executables {
        if which_exists(exe) {
            return serde_json::json!({
                "name": name,
                "status": "found",
                "executable": exe,
            });
        }
    }
    serde_json::json!({
        "name": name,
        "status": "missing",
    })
}

/// Returns true if the given executable name is found in the system PATH.
fn which_exists(name: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| {
                let candidate = dir.join(name);
                candidate.exists() || candidate.with_extension("exe").exists()
            })
        })
        .unwrap_or(false)
}

/// Returns a future that resolves when a termination signal is received.
///
/// On Unix (Linux, macOS), listens for both SIGINT (Ctrl+C) and SIGTERM.
/// SIGTERM is the standard shutdown signal sent by `kill`, `systemctl stop`,
/// `docker stop`, and Kubernetes pod termination. Without SIGTERM handling,
/// bare-metal Linux deployments only respond to Ctrl+C and get SIGKILL after
/// the grace period, risking in-flight SQLite WAL corruption.
///
/// On Windows, only SIGINT (Ctrl+C) is relevant because Windows uses
/// `GenerateConsoleCtrlEvent` / `TerminateProcess` instead of Unix signals.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut sigterm =
            signal(SignalKind::terminate()).expect("failed to install SIGTERM signal handler");

        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                result.expect("failed to install CTRL+C signal handler");
                tracing::info!("received SIGINT (Ctrl+C), shutting down");
            }
            _ = sigterm.recv() => {
                tracing::info!("received SIGTERM, shutting down");
            }
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install CTRL+C signal handler");
        tracing::info!("received shutdown signal");
    }
}
