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

// Autonomous citation verification agent driven by a local LLM.
//
// The `CitationAgent` claims batches from an existing citation_verify job,
// constructs search queries via the LLM, executes them through the
// NeuronCite SearchPipeline, feeds the resulting passages back to the LLM
// for verdict generation, and submits the structured results to the database.
//
// The agent is backend-agnostic: it accepts `Arc<dyn LlmBackend>` and never
// imports a concrete LLM implementation. Swapping Ollama for a native Rust
// backend requires only constructing a different
// `LlmBackend` implementation at the call site.
//
// The search pipeline integration replicates the same HNSW + FTS5 + RRF
// hybrid search used by the REST and MCP search endpoints, running the
// synchronous SearchPipeline inside `spawn_blocking` to avoid blocking
// the tokio event loop.

use std::sync::Arc;

use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use neuroncite_citation::db as cit_db;
use neuroncite_citation::types::{
    CitationRow, CorrectionType, LatexCorrection, OtherSourceEntry, OtherSourcePassage,
    PassageLocation, PassageRef, SubmitEntry, Verdict,
};
use neuroncite_core::SearchResult;
use neuroncite_llm::LlmBackend;
use neuroncite_llm::error::LlmError;
use neuroncite_llm::types::{ChatMessage, ChatRole, LlmConfig};
use neuroncite_search::{SearchConfig, SearchPipeline};
use neuroncite_store::{self as store, JobState};

use crate::state::AppState;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during autonomous citation verification.
/// Each variant maps to a distinct failure domain so callers can
/// distinguish LLM failures from search failures from DB errors.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// The LLM backend returned an error (connection, timeout, parse).
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    /// The search pipeline failed (HNSW lookup, FTS5 query, embedding).
    #[error("search error: {0}")]
    Search(String),

    /// A database operation failed (pool, query, constraint violation).
    #[error("database error: {0}")]
    Database(String),

    /// The LLM returned a response that could not be parsed as the expected
    /// JSON structure after retry attempts.
    #[error("verdict parse failure: {raw_response}")]
    VerdictParseFailure { raw_response: String },

    /// Embedding a query vector via the GPU worker failed.
    #[error("embedding error: {0}")]
    Embed(String),
}

// ---------------------------------------------------------------------------
// SSE event helpers
// ---------------------------------------------------------------------------

/// Sends a JSON-encoded SSE event through the broadcast channel if available.
/// Silently drops events when no subscribers are connected (channel full or
/// no receivers). This is intentional: SSE events are advisory, not critical.
fn emit_sse(tx: &Option<broadcast::Sender<String>>, event_json: serde_json::Value) {
    if let Some(sender) = tx {
        let _ = sender.send(event_json.to_string());
    }
}

// ---------------------------------------------------------------------------
// Active agent guard (drop-based cleanup)
// ---------------------------------------------------------------------------

/// RAII guard that removes the job_id from `AppState::active_citation_agents`
/// when dropped. Guarantees cleanup even on early returns or panics inside
/// `CitationAgent::run()`.
struct ActiveAgentGuard<'a> {
    set: &'a dashmap::DashSet<String>,
    job_id: &'a str,
}

impl Drop for ActiveAgentGuard<'_> {
    fn drop(&mut self) {
        self.set.remove(self.job_id);
    }
}

// ---------------------------------------------------------------------------
// Prompt templates
// ---------------------------------------------------------------------------

/// System prompt establishing the citation verification agent persona.
/// Matches the behavioral rules and calibration constraints defined in
/// the MCP tool schema for `neuroncite_citation_submit`. Every verdict
/// definition, confidence band, calibration constraint, and flag rule
/// specified here is enforced identically in both the local agent and
/// the MCP-based sub-agent workflow.
// ---------------------------------------------------------------------------
// Step 1: Claim extraction prompt (plain text, no JSON)
// ---------------------------------------------------------------------------
/// System prompt for claim extraction. Returns a JSON object with
/// "claim_original" and "claim_english" fields. Uses JSON mode since
/// the LLM backend is configured with json_mode: true globally.
const CLAIM_EXTRACTION_SYSTEM_PROMPT: &str = r#"You extract factual claims from LaTeX citation contexts. Respond with ONLY a JSON object. No markdown, no extra text.

Rules:
- Extract ONLY the factual claim that the citation is supposed to support.
- Remove all LaTeX commands (\cite, \ref, formatting). Output clean natural language.
- If the text is not in English, provide the original AND an English translation.
- If the text is already in English, use the same text for both fields.
- Keep it concise: 1-3 sentences maximum.
- Do NOT add information that is not in the original text."#;

/// User prompt for claim extraction. Includes a document overview for
/// topic grounding so the LLM can disambiguate domain-specific terms.
const CLAIM_EXTRACTION_PROMPT: &str = r#"Extract the factual claim from this LaTeX citation context:

Document overview:
{DOC_OVERVIEW}

Section: {SECTION}
LaTeX context (line {TEX_LINE}):
{TEX_CONTEXT}

Cited source: {AUTHOR} ({YEAR}) - "{TITLE}"

What factual claim is being made that this citation is supposed to support?

Respond with: {"claim_original": "claim in original language", "claim_english": "claim in English"}"#;

// ---------------------------------------------------------------------------
// Step 2: Search query generation prompt
// ---------------------------------------------------------------------------

/// System prompt for the search query generation LLM call. Focused solely on
/// producing good search queries from an already-extracted claim.
const QUERY_GENERATION_SYSTEM_PROMPT: &str = r#"You generate search queries for academic citation verification. Respond with ONLY a JSON array of 2-4 query strings. No markdown, no explanation.

Rules:
- Each query should be 5-15 words: specific enough to match the source, short enough for semantic search.
- Use natural language. Convert formulas to words (e.g., "$O(n \log n)$" → "time complexity n log n").
- If the claim contains specific numbers, names, dates, or definitions, make one query targeting those specifics.
- Include queries in BOTH the original language AND English if the claim is not in English.
- Do NOT include author names or publication years in queries."#;

/// User prompt for search query generation. Uses the extracted claim instead
/// of raw LaTeX context.
const QUERY_GENERATION_PROMPT: &str = r#"Generate 2-4 search queries to find evidence for this claim in the cited PDF.

Claim: {CLAIM}

Cited source: {AUTHOR} ({YEAR}) - "{TITLE}"

Respond with a JSON array of query strings. Example:
["efficient market hypothesis weak form", "stock prices follow random walk", "Fama definition of market efficiency"]"#;

// ---------------------------------------------------------------------------
// Step 3: Verdict + reasoning prompt (only 2 fields)
// ---------------------------------------------------------------------------

/// System prompt for verdict generation. Kept concise so small models can
/// follow it reliably. Only asks for verdict + reasoning — no confidence,
/// no flag, no claim extraction (those are handled separately).
const VERDICT_SYSTEM_PROMPT: &str = r#"You are a citation verification agent. Your task: determine whether a cited source supports the claim made in an academic paper.

Verdict options:
- "supported": The source explicitly states the claim. No inference required.
- "partial": The source supports part of the claim but another part is absent or deviates.
- "unsupported": Relevant passages exist in the source but do not support the specific claim.
- "not_found": The source is not in the indexed corpus.
- "wrong_source": The claim is supported by a different source, not the one cited.
- "unverifiable": The claim cannot be checked (future projections, subjective statements, unavailable data).
- "peripheral_match": The claim appears only in non-substantive sections (table of contents, bibliography, appendix).

Respond with ONLY a JSON object. No markdown, no extra text."#;

/// User prompt for verdict generation. Only asks for verdict + reasoning.
/// Placeholders: {CLAIM}, {AUTHOR}, {YEAR}, {TITLE}, {PASSAGES},
/// {CROSS_CORPUS_PASSAGES}.
const VERDICT_PROMPT: &str = r#"Does the cited source support this claim?

Claim: {CLAIM}
Cited source: {AUTHOR} ({YEAR}) - "{TITLE}"

Passages found in the cited source:
{PASSAGES}

Passages found in other sources:
{CROSS_CORPUS_PASSAGES}

You MUST respond with a JSON object with exactly two fields:
1. "verdict" — one of: supported, partial, unsupported, not_found, wrong_source, unverifiable, peripheral_match
2. "reasoning" — 2-3 sentences explaining WHY you chose that verdict. Reference specific passages.

Example response format (do NOT copy these values, write your own analysis):
{{"verdict": "partial", "reasoning": "Passage 1 on page 5 confirms the general concept of X, but the specific claim about Y is not mentioned anywhere in the source. The author discusses Z instead, which only partially aligns with the claim."}}"#;

// ---------------------------------------------------------------------------
// Step 4: Confidence prompt (single number)
// ---------------------------------------------------------------------------

/// System prompt for confidence scoring.
const CONFIDENCE_SYSTEM_PROMPT: &str = r#"You assess how confident a citation verification verdict is. Respond with ONLY a JSON object containing a single "confidence" field. No markdown, no extra text.

Confidence is meta-confidence: how certain you are that the verdict is correct, NOT how strongly the source supports the claim.

Guidelines:
- 0.90-1.00: Verbatim match or definitive evidence.
- 0.70-0.89: Clear evidence, minor interpretation needed.
- 0.50-0.69: Moderate certainty, some ambiguity.
- Below 0.50: Low certainty, significant doubt."#;

/// User prompt for confidence scoring.
const CONFIDENCE_PROMPT: &str = r#"Rate your confidence in this verdict.

Claim: {CLAIM}
Verdict: {VERDICT}
Reasoning: {REASONING}

Respond with ONLY: {{"confidence": 0.85}}
The value must be between 0.0 and 1.0."#;

/// Phase-2 metadata prompt: collects non-critical fields (passage locations,
/// latex correction suggestions, alternative source recommendations) after
/// the core verdict has been parsed. This separation allows the agent
/// to produce valid results even when the metadata call fails or the model
/// is too small to handle the full schema.
/// Placeholders: {VERDICT}, {REASONING}, {PASSAGES}, {CROSS_CORPUS_PASSAGES}.
const VERDICT_METADATA_PROMPT: &str = r#"Given this citation verification verdict:

Verdict: {VERDICT}
Reasoning: {REASONING}

Passages from the cited source:
{PASSAGES}

Cross-corpus passages:
{CROSS_CORPUS_PASSAGES}

Provide metadata as a JSON object:
{{
  "passage_locations": ["location_for_each_passage_in_order"],
  "latex_correction": {{
    "correction_type": "none|rephrase|add_context|replace_citation",
    "original_text": "",
    "suggested_text": "",
    "explanation": ""
  }},
  "better_source": []
}}

Rules for passage_locations (one entry per cited-source passage, in order):
Valid values: abstract, foreword, table_of_contents, introduction, literature_review, theoretical_framework, methodology, results, discussion, conclusion, bibliography, appendix, glossary, table_or_figure, footnote, body_text.

Rules for latex_correction:
- "none" with empty strings when no correction is needed.
- "rephrase" when the claim overstates or misrepresents the source.
- "add_context" when a qualifying sentence would resolve ambiguity.
- "replace_citation" when the cited work is wrong.

Rules for better_source:
List filenames from cross-corpus passages that better support this claim. Empty array if none."#;

// ---------------------------------------------------------------------------
// CitationAgent
// ---------------------------------------------------------------------------

/// Request parameters for starting an autonomous verification run.
/// Passed to the `auto_verify` handler and used to construct the LlmConfig.
/// Includes verification tuning parameters that control search depth,
/// cross-corpus coverage, reranking, and retry behavior.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, utoipa::ToSchema)]
pub struct AutoVerifyRequest {
    /// Ollama server URL (default: "http://localhost:11434").
    pub ollama_url: Option<String>,
    /// Model identifier as listed by Ollama (e.g., "qwen2.5:14b").
    pub model: String,
    /// Sampling temperature (default: 0.1 for deterministic verification).
    pub temperature: Option<f32>,
    /// Maximum tokens to generate per completion (default: 4096).
    pub max_tokens: Option<u32>,
    /// Directory where cited source PDFs and scraped HTML pages are stored.
    /// When `fetch_sources` is true, the agent downloads missing sources into
    /// this directory before starting verification.
    pub source_directory: Option<String>,
    /// When true, the agent downloads source PDFs/HTML from BibTeX URL/DOI
    /// fields before starting verification. Requires `source_directory`.
    pub fetch_sources: Option<bool>,
    /// Email address for Unpaywall API access in the DOI resolution chain.
    /// When provided, DOI resolution tries Unpaywall first for direct PDF URLs.
    /// When absent, Unpaywall is skipped and resolution starts with Semantic Scholar.
    pub unpaywall_email: Option<String>,

    // -- Verification tuning parameters --
    /// Number of search results returned per individual query (default: 5).
    /// Higher values increase recall at the cost of speed.
    pub top_k: Option<usize>,
    /// Maximum number of cross-corpus search queries executed per citation
    /// to find alternative sources (default: 2, range 0-4).
    pub cross_corpus_queries: Option<usize>,
    /// Maximum retry attempts for LLM verdict parsing before marking a
    /// row as failed (default: 3, range 1-5).
    pub max_retry_attempts: Option<usize>,
    /// Minimum vector similarity score threshold. Results below this cosine
    /// similarity are excluded from consideration (default: None = no filter).
    pub min_score: Option<f64>,
    /// When true, applies cross-encoder reranking to search results.
    /// Requires a reranker model to be loaded (default: false).
    pub rerank: Option<bool>,
}

/// Response returned by the `auto_verify` handler.
#[derive(Debug, Clone, serde::Serialize, utoipa::ToSchema)]
pub struct AutoVerifyResponse {
    pub api_version: String,
    pub status: String,
    pub job_id: String,
    pub message: String,
}

/// Configuration parameters for the citation verification agent that control
/// search behavior, retry policy, source fetching, and reranking. These
/// fields were previously individual parameters on `CitationAgent::new()`,
/// grouped here to reduce the constructor's argument count.
pub struct AgentConfig {
    /// Number of search results per individual query. Controls the max_results
    /// parameter in SearchConfig and the multiplier for HNSW/FTS5 candidates.
    pub top_k: usize,

    /// Maximum number of cross-corpus search queries executed per citation
    /// to find alternative sources in the broader corpus.
    pub cross_corpus_queries: usize,

    /// Maximum retry attempts for LLM verdict parsing before marking a row
    /// as failed. Controls the retry loop in generate_verdict_with_retry.
    pub max_retry_attempts: usize,

    /// Minimum vector similarity score threshold. When set, search results
    /// below this cosine similarity are excluded from the search pipeline.
    pub min_score: Option<f64>,

    /// When true, applies cross-encoder reranking to search results.
    /// Requires a reranker model to be loaded in the GPU worker.
    pub rerank: bool,

    /// Directory where cited source PDFs and scraped HTML pages are stored.
    /// Used by the source-fetching step to download missing sources.
    pub source_directory: Option<String>,

    /// Whether to fetch sources from BibTeX URL/DOI fields before starting
    /// verification. The BibTeX path is read from the job's params_json.
    pub fetch_sources: bool,

    /// Email address for Unpaywall API access in the DOI resolution chain.
    /// Passed from the AutoVerifyRequest. When None, the resolution chain
    /// skips Unpaywall and starts with Semantic Scholar.
    pub unpaywall_email: Option<String>,
}

/// Autonomous citation verification agent. Claims batches from a citation_verify
/// job, drives the LLM through search/evaluate cycles, and submits structured
/// results to the database. Communicates progress via an optional SSE broadcast
/// channel.
///
/// The agent holds `Arc<dyn LlmBackend>` for backend-agnostic LLM access,
/// `Arc<AppState>` for database/search infrastructure, and optional SSE
/// broadcast channel for live progress reporting to the web frontend.
pub struct CitationAgent {
    /// Shared application state providing database pool, HNSW indices,
    /// GPU worker handle, and server configuration.
    state: Arc<AppState>,

    /// The LLM backend used for query generation and verdict evaluation.
    /// Trait object for backend-agnostic dispatch (Ollama, etc.).
    llm: Arc<dyn LlmBackend>,

    /// The citation_verify job ID being processed.
    job_id: String,

    /// All NeuronCite index session IDs to search against. When multiple
    /// sessions are provided, the agent searches each session independently
    /// and merges results by score. The first session in the list is used
    /// as the primary session for source indexing operations.
    session_ids: Vec<i64>,

    /// SSE broadcast channel for live progress events. None when running
    /// in headless/CLI mode without a web frontend.
    citation_tx: Option<broadcast::Sender<String>>,

    /// Grouped configuration for search behavior, retry policy, source
    /// fetching, and reranking parameters.
    config: AgentConfig,
}

impl CitationAgent {
    /// Constructs a CitationAgent for the given job with the provided
    /// configuration. The `config` parameter groups all tuning knobs
    /// (search depth, retry policy, source fetching, reranking) that
    /// were previously individual constructor parameters.
    #[must_use]
    pub fn new(
        state: Arc<AppState>,
        llm: Arc<dyn LlmBackend>,
        job_id: String,
        session_ids: Vec<i64>,
        citation_tx: Option<broadcast::Sender<String>>,
        config: AgentConfig,
    ) -> Self {
        Self {
            state,
            llm,
            job_id,
            session_ids,
            citation_tx,
            config,
        }
    }

    /// Main entry point. Runs one-time setup (source fetching, indexing,
    /// tex context reading), then processes all batches sequentially.
    ///
    /// Takes `Arc<Self>` because the worker loop method requires it.
    pub async fn run(self: Arc<Self>) -> Result<(), AgentError> {
        // Register this agent in the active set so the auto_verify handler
        // rejects duplicate spawn requests for the same job. The entry is
        // removed when this method returns (success or error) via the
        // deferred cleanup below.
        self.state
            .active_citation_agents
            .insert(self.job_id.clone());

        // Deferred cleanup: remove the job_id from the active set when
        // this scope exits, regardless of how run() returns. Uses a drop
        // guard pattern to guarantee removal even on early returns or panics.
        let _active_guard = ActiveAgentGuard {
            set: &self.state.active_citation_agents,
            job_id: &self.job_id,
        };

        info!(
            job_id = %self.job_id,
            session_ids = ?self.session_ids,
            llm_backend = %self.llm.name(),
            model = %self.llm.config().model,
            "Citation agent starting"
        );

        // --- Phase A: One-time setup (runs before any workers start) ---

        // If fetch_sources is enabled, download cited source PDFs/HTML
        // from BibTeX URL/DOI fields into the source directory and index
        // them into the session before starting verification.
        if self.config.fetch_sources {
            if let Some(ref source_dir) = self.config.source_directory {
                self.fetch_and_index_sources(source_dir).await;
            } else {
                warn!(job_id = %self.job_id, "fetch_sources is true but no source_directory provided, skipping source fetch");
            }
        }

        // If a source directory is provided (regardless of fetch_sources),
        // index any PDFs in that directory that are not yet in the session.
        // Then re-match unmatched citation rows against the updated file list.
        // This handles the case where PDFs were already present in the source
        // directory before auto-verify was called, or were added by fetch above.
        if let Some(ref source_dir) = self.config.source_directory {
            self.index_source_directory_if_needed(source_dir).await;
            self.rematch_unmatched_rows().await;
        }

        // Read the .tex file for document-level context (first ~2000 chars
        // plus section headings). This context is prepended to each LLM
        // query so the model understands the document's topic and structure.
        // Computed once and shared across all workers as an immutable string.
        let tex_context = Arc::new(self.read_document_context().await);

        // --- Phase B: Process all batches sequentially ---

        self.worker_loop(0, &tex_context).await;

        // Check for job completion (all rows done or failed).
        self.check_job_completion().await;

        info!(job_id = %self.job_id, "Citation agent finished");
        Ok(())
    }

    /// Worker loop that claims and processes batches sequentially until no
    /// pending batches remain.
    ///
    /// Errors on individual rows are caught and the row is marked as failed
    /// so the worker continues with the next row instead of aborting.
    async fn worker_loop(self: &Arc<Self>, worker_id: usize, tex_context: &str) {
        loop {
            let batch = {
                let conn = match self.state.pool.get() {
                    Ok(c) => c,
                    Err(e) => {
                        error!(job_id = %self.job_id, worker_id, error = %e, "Connection pool error, worker stopping");
                        return;
                    }
                };
                match cit_db::claim_next_batch(&conn, &self.job_id) {
                    Ok(b) => b,
                    Err(e) => {
                        error!(job_id = %self.job_id, worker_id, error = %e, "Batch claim failed, worker stopping");
                        return;
                    }
                }
            };

            let batch = match batch {
                Some(b) => b,
                None => {
                    info!(job_id = %self.job_id, worker_id, "No more pending batches, worker loop complete");
                    break;
                }
            };

            info!(
                job_id = %self.job_id,
                worker_id,
                batch_id = batch.batch_id,
                row_count = batch.rows.len(),
                "Worker claimed batch for processing"
            );

            // Process each row in the batch individually. Errors on a single
            // row do not abort the batch -- the row is marked as failed and
            // the worker continues with the next row.
            for row in &batch.rows {
                match self.process_row(row, tex_context).await {
                    Ok(()) => {
                        debug!(
                            job_id = %self.job_id,
                            worker_id,
                            row_id = row.id,
                            cite_key = %row.cite_key,
                            "Row processed"
                        );
                    }
                    Err(e) => {
                        error!(
                            job_id = %self.job_id,
                            worker_id,
                            row_id = row.id,
                            cite_key = %row.cite_key,
                            error = %e,
                            "Row processing failed, marking as failed"
                        );
                        self.mark_row_failed(row.id, &row.cite_key, &e.to_string());
                    }
                }

                // Emit progress event after each row.
                self.emit_progress().await;
            }
        }
    }

    /// Processes a single citation row through the full verification pipeline:
    /// 1. Extract claim context from the row's tex_context and metadata
    /// 2. Ask the LLM to generate search queries
    /// 3. Execute searches against the matched PDF (and cross-corpus)
    /// 4. Ask the LLM to evaluate passages and produce a verdict
    /// 5. Parse the verdict JSON and submit the result
    async fn process_row(&self, row: &CitationRow, doc_context: &str) -> Result<(), AgentError> {
        // Emit SSE: row entering "searching" phase.
        emit_sse(
            &self.citation_tx,
            serde_json::json!({
                "event": "citation_row_update",
                "job_id": self.job_id,
                "row_id": row.id,
                "cite_key": row.cite_key,
                "phase": "searching",
            }),
        );

        let file_id_opt = row.matched_file_id;
        let year_str = row.year.as_deref().unwrap_or("unknown");

        // Step 1: Extract the core claim as plain text (no JSON required).
        // Passes doc_context for topic grounding so the LLM can disambiguate terms.
        let (claim_original, claim_english) = self.extract_claim(row, doc_context).await?;
        // Use the English version for search queries and verdict evaluation.
        let claim = if claim_english.is_empty() {
            &claim_original
        } else {
            &claim_english
        };

        // Step 2: Generate search queries from the extracted claim.
        let queries = self
            .generate_search_queries(claim, &row.author, &row.title, year_str)
            .await?;

        debug!(
            row_id = row.id,
            query_count = queries.len(),
            matched_file = ?file_id_opt,
            "Generated search queries"
        );

        // Step 2: Execute searches. When file_id_opt is Some, searches target
        // only the matched PDF. When None, searches span all indexed files.
        let file_ids_filter = file_id_opt.map(|id| vec![id]);
        let mut all_passages: Vec<SearchResult> = Vec::new();
        let mut search_rounds: i64 = 0;

        for query in &queries {
            let results = self
                .do_search(query, file_ids_filter.clone(), self.config.top_k)
                .await?;
            all_passages.extend(results);
            search_rounds += 1;
        }

        // Step 3: Cross-corpus search for other_source_list. Only runs when
        // a specific file was matched (file_id_opt is Some), because corpus-
        // wide search already covers all files when file_id_opt is None.
        let mut cross_results: Vec<SearchResult> = Vec::new();
        if file_id_opt.is_some() && self.config.cross_corpus_queries > 0 {
            let cross_query_count = queries.len().clamp(1, self.config.cross_corpus_queries);
            for query in queries.iter().take(cross_query_count) {
                let results = self.do_search(query, None, self.config.top_k).await?;
                cross_results.extend(results);
                search_rounds += 1;
            }
            if queries.is_empty() {
                let results = self.do_search(claim, None, self.config.top_k).await?;
                cross_results.extend(results);
                search_rounds += 1;
            }
        }
        // Deduplicate cross-corpus results by content.
        cross_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        cross_results.dedup_by(|a, b| a.content == b.content);

        // Deduplicate matched-PDF passages by content (keep highest score).
        all_passages.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_passages.dedup_by(|a, b| a.content == b.content);

        // When no file was matched deterministically AND the corpus-wide
        // search returned zero passages, submit a not_found verdict without
        // invoking the LLM. The corpus was searched but no relevant content
        // exists, so an LLM call would only produce a "not_found" verdict
        // with no additional information.
        if all_passages.is_empty() && file_id_opt.is_none() {
            let entry = build_corpus_searched_not_found_entry(
                row,
                &claim_original,
                &claim_english,
                search_rounds,
            );
            self.submit_single_result(&entry)?;
            emit_sse(
                &self.citation_tx,
                serde_json::json!({
                    "event": "citation_row_update",
                    "job_id": self.job_id,
                    "row_id": row.id,
                    "cite_key": row.cite_key,
                    "phase": "done",
                    "verdict": "not_found",
                    "confidence": 1.0,
                }),
            );
            return Ok(());
        }

        // Emit SSE: row entering "evaluating" phase.
        emit_sse(
            &self.citation_tx,
            serde_json::json!({
                "event": "citation_row_update",
                "job_id": self.job_id,
                "row_id": row.id,
                "cite_key": row.cite_key,
                "phase": "evaluating",
                "search_queries": queries,
            }),
        );

        // Step 4: Build passage text for the LLM prompt.
        let passages_text = format_passages_for_prompt(&all_passages);
        let cross_text = format_passages_for_prompt(&cross_results);

        // Step 3-6: Multi-step verdict pipeline (verdict+reasoning, confidence,
        // flag, metadata — each as a separate focused LLM call).
        let entry = self
            .generate_verdict_with_retry(
                row,
                claim,
                &claim_original,
                &claim_english,
                &passages_text,
                &cross_text,
                search_rounds,
                &all_passages,
                &cross_results,
                file_id_opt,
            )
            .await?;

        // Step 6: Submit result.
        self.submit_single_result(&entry)?;

        // Emit SSE: row completed.
        emit_sse(
            &self.citation_tx,
            serde_json::json!({
                "event": "citation_row_update",
                "job_id": self.job_id,
                "row_id": row.id,
                "cite_key": row.cite_key,
                "phase": "done",
                "verdict": format!("{:?}", entry.verdict).to_lowercase(),
                "confidence": entry.confidence,
            }),
        );

        Ok(())
    }

    /// Reads the .tex file to extract document-level context. Returns the
    /// first ~2000 characters plus any section headings found in the file.
    /// Runs on a blocking thread because std::fs::read_to_string is sync I/O.
    async fn read_document_context(&self) -> String {
        let conn = match self.state.pool.get() {
            Ok(c) => c,
            Err(_) => return String::new(),
        };

        // Read the tex_path from the job record's params_json field.
        // Using get_job() avoids inline SQL and reuses the existing store function.
        let params_json: Option<String> = neuroncite_store::get_job(&conn, &self.job_id)
            .ok()
            .and_then(|row| row.params_json);

        let tex_path = match params_json {
            Some(json) => serde_json::from_str::<serde_json::Value>(&json)
                .ok()
                .and_then(|v| v["tex_path"].as_str().map(String::from))
                .unwrap_or_default(),
            None => return String::new(),
        };

        if tex_path.is_empty() {
            return String::new();
        }

        // Read the file on a blocking thread.
        tokio::task::spawn_blocking(move || {
            match std::fs::read_to_string(&tex_path) {
                Ok(content) => {
                    // Take the first 2000 characters as document overview.
                    let preview: String = content.chars().take(2000).collect();

                    // Extract section headings for structural context.
                    let headings: Vec<String> = content
                        .lines()
                        .filter(|line| {
                            let trimmed = line.trim();
                            trimmed.starts_with("\\section")
                                || trimmed.starts_with("\\subsection")
                                || trimmed.starts_with("\\chapter")
                        })
                        .map(|line| line.trim().to_string())
                        .collect();

                    let headings_text = if headings.is_empty() {
                        String::new()
                    } else {
                        format!("\n\nDocument sections:\n{}", headings.join("\n"))
                    };

                    format!("{preview}{headings_text}")
                }
                Err(e) => {
                    warn!(error = %e, "Failed to read tex file for document context");
                    String::new()
                }
            }
        })
        .await
        .unwrap_or_default()
    }

    /// Step 1: Extracts the core factual claim from raw LaTeX context as
    /// plain text. No JSON required — the LLM just outputs the claim.
    /// If the claim is non-English, the response contains both the original
    /// and English translation separated by " ||| ".
    /// Returns (claim_original, claim_english).
    async fn extract_claim(
        &self,
        row: &CitationRow,
        doc_context: &str,
    ) -> Result<(String, String), AgentError> {
        let tex_ctx = row.tex_context.as_deref().unwrap_or("");
        let section = row.section_title.as_deref().unwrap_or("unknown section");
        let year_str = row.year.as_deref().unwrap_or("unknown");

        // Include the first 500 chars of the document overview for topic grounding.
        let doc_overview: String = doc_context.chars().take(500).collect();

        let prompt = CLAIM_EXTRACTION_PROMPT
            .replace("{DOC_OVERVIEW}", &doc_overview)
            .replace("{SECTION}", section)
            .replace("{TEX_LINE}", &row.tex_line.to_string())
            .replace("{TEX_CONTEXT}", tex_ctx)
            .replace("{AUTHOR}", &row.author)
            .replace("{YEAR}", year_str)
            .replace("{TITLE}", &row.title);

        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: CLAIM_EXTRACTION_SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: prompt,
            },
        ];

        let response = self.llm.chat_completion(&messages).await?;
        let content = strip_think_blocks(&response.content);

        // Parse the JSON response. Fall back to raw tex_context on failure.
        let cleaned = extract_json_object(content.as_ref());
        let repaired = repair_truncated_json(&cleaned);

        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&repaired) {
            let original = parsed["claim_original"]
                .as_str()
                .unwrap_or("")
                .trim()
                .to_string();
            let english = parsed["claim_english"]
                .as_str()
                .unwrap_or("")
                .trim()
                .to_string();

            if !original.is_empty() {
                let english = if english.is_empty() {
                    original.clone()
                } else {
                    english
                };
                debug!(row_id = row.id, claim_original = %original, claim_english = %english, "Extracted claim");
                return Ok((original, english));
            }
        }

        // Fallback: use the raw response as plain text claim.
        let fallback = content.trim().to_string();
        if !fallback.is_empty() {
            debug!(row_id = row.id, claim = %fallback, "Extracted claim (fallback to raw text)");
            return Ok((fallback.clone(), fallback));
        }

        // Last resort: use raw tex_context.
        let tex = tex_ctx.to_string();
        warn!(
            row_id = row.id,
            "Claim extraction returned empty, using raw tex_context"
        );
        Ok((tex.clone(), tex))
    }

    /// Step 2: Generates 2-4 search queries from the extracted claim.
    /// Uses the clean claim text instead of raw LaTeX context.
    async fn generate_search_queries(
        &self,
        claim: &str,
        author: &str,
        title: &str,
        year: &str,
    ) -> Result<Vec<String>, AgentError> {
        let prompt = QUERY_GENERATION_PROMPT
            .replace("{CLAIM}", claim)
            .replace("{AUTHOR}", author)
            .replace("{TITLE}", title)
            .replace("{YEAR}", year);

        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: QUERY_GENERATION_SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: prompt,
            },
        ];

        let response = self.llm.chat_completion(&messages).await?;

        // Remove <think>...</think> blocks emitted by deepseek-r1 style models
        // before the actual answer content.
        let content = strip_think_blocks(&response.content);

        // Attempt 1: response is already a plain JSON array of strings.
        if let Ok(queries) = serde_json::from_str::<Vec<String>>(content.as_ref())
            && !queries.is_empty()
        {
            return Ok(queries);
        }

        // Attempt 2: response contains a JSON array embedded in markdown code
        // blocks or surrounded by other text. Extract the [...] substring.
        let array_str = extract_json_array(content.as_ref());
        if let Ok(queries) = serde_json::from_str::<Vec<String>>(&array_str)
            && !queries.is_empty()
        {
            return Ok(queries);
        }

        // Attempt 3: response is a JSON object. Smaller models sometimes
        // return {"query text": "", ...} or {"query1": "text1", ...}.
        // Extract usable strings from both keys and values.
        let obj_str = extract_json_object(content.as_ref());
        if let Ok(obj) =
            serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&obj_str)
        {
            // Case A: values are non-empty strings → use values.
            let mut pairs: Vec<(&String, &serde_json::Value)> = obj.iter().collect();
            pairs.sort_by_key(|(k, _)| k.as_str());
            let values: Vec<String> = pairs
                .iter()
                .filter_map(|(_, v)| v.as_str().map(|s| s.to_string()))
                .filter(|s| !s.is_empty())
                .collect();
            if !values.is_empty() {
                return Ok(values);
            }

            // Case B: keys are the actual queries (values are empty strings
            // or numbers). Common pattern: {"query text": "", ...}.
            let keys: Vec<String> = obj
                .keys()
                .filter(|k| k.len() > 5 && !k.starts_with('['))
                .cloned()
                .collect();
            if !keys.is_empty() {
                info!(
                    row_queries = ?keys,
                    "Extracted search queries from JSON object keys"
                );
                return Ok(keys);
            }

            // Case C: a key is itself a JSON-stringified array.
            for key in obj.keys() {
                if let Ok(arr) = serde_json::from_str::<Vec<String>>(key)
                    && !arr.is_empty()
                {
                    return Ok(arr);
                }
            }
        }

        warn!(
            response = %response.content,
            "Failed to parse search queries from LLM, using claim text as fallback"
        );
        // Fallback: use the extracted claim (now clean text, not raw LaTeX).
        let fallback: String = claim.chars().take(200).collect();
        Ok(vec![fallback])
    }

    /// Executes a hybrid search (vector + FTS5 keyword) against the NeuronCite
    /// index. Mirrors the search handler's pipeline: embed query -> load HNSW
    /// -> build SearchConfig -> run SearchPipeline inside spawn_blocking.
    /// When a reranker model is loaded, applies cross-encoder reranking to
    /// the results via the GPU worker (same path as the REST search handler).
    ///
    /// `file_ids` restricts results to specific PDF files. When None, searches
    /// the entire session (used for cross-corpus search).
    async fn do_search(
        &self,
        query: &str,
        file_ids: Option<Vec<i64>>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, AgentError> {
        // Embed the query text via the GPU worker.
        let query_vec = self
            .state
            .worker_handle
            .embed_query(query.to_string())
            .await
            .map_err(|e| AgentError::Embed(format!("embedding failed: {e}")))?;

        let ef_search = self.state.config.ef_search;
        let hnsw_guard = self.state.index.hnsw_index.load();
        let rerank_flag = self.config.rerank && self.state.worker_handle.reranker_available();
        let min_score_val = self.config.min_score;

        // Search across all sessions and merge results by score. Each
        // session has its own HNSW index and FTS5 tables. Results from
        // all sessions are collected and then sorted by descending score.
        let mut all_results: Vec<SearchResult> = Vec::new();

        for &session_id in &self.session_ids {
            let query_text = query.to_string();
            let pool = self.state.pool.clone();
            let query_vec_clone = query_vec.clone();
            let file_ids_clone = file_ids.clone();

            let hnsw_ref = match hnsw_guard.get(&session_id) {
                Some(h) => Arc::clone(h),
                None => continue,
            };

            // Run the synchronous SearchPipeline inside spawn_blocking
            // because rusqlite::Connection is not Send and the HNSW
            // search is CPU-bound.
            let session_results = tokio::task::spawn_blocking(move || {
                let conn = pool
                    .get()
                    .map_err(|e| AgentError::Search(format!("connection pool error: {e}")))?;

                let config = SearchConfig {
                    session_id,
                    vector_top_k: top_k * 5,
                    keyword_limit: top_k * 5,
                    ef_search,
                    rrf_k: 60,
                    bm25_must_match: false,
                    simhash_threshold: 3,
                    max_results: top_k,
                    // Per-session reranking is disabled here. The agent collects
                    // results from all sessions into a single list and reranks
                    // the merged set once below. Enabling reranking at the
                    // per-session level would apply it multiple times to the
                    // same documents as they propagate into the merged list.
                    rerank_enabled: false,
                    file_ids: file_ids_clone,
                    min_score: min_score_val,
                    page_start: None,
                    page_end: None,
                };

                let pipeline = SearchPipeline::new(
                    &hnsw_ref,
                    &conn,
                    &query_vec_clone,
                    &query_text,
                    None,
                    config,
                );

                let output = pipeline
                    .search()
                    .map_err(|e| AgentError::Search(format!("search pipeline failed: {e}")))?;

                Ok::<Vec<SearchResult>, AgentError>(output.results)
            })
            .await
            .map_err(|e| AgentError::Search(format!("spawn_blocking panicked: {e}")))??;

            all_results.extend(session_results);
        }

        // Sort merged results by descending score and truncate to top_k.
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(top_k);

        // Apply cross-encoder reranking via the GPU worker when the caller
        // requested reranking AND a reranker model is loaded. Reranking is
        // performed once on the fully merged and truncated result list.
        // Per-session reranking was disabled in the search loop above, so
        // each result here has only its RRF score, not a reranker score.
        // Errors are logged but not propagated — the fused results are still
        // usable for verdict evaluation without the reranker refinement.
        if rerank_flag && self.state.worker_handle.reranker_available() && !all_results.is_empty() {
            let content_refs: Vec<&str> = all_results.iter().map(|r| r.content.as_str()).collect();
            match self
                .state
                .worker_handle
                .rerank_batch(query, &content_refs)
                .await
            {
                Ok(scores) => {
                    for (result, score) in all_results.iter_mut().zip(scores.iter()) {
                        result.reranker_score = Some(*score);
                        result.score = *score;
                    }
                    // Re-sort by descending reranker score.
                    all_results.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                Err(e) => {
                    warn!(error = %e, "Reranking failed for agent search, using base scores");
                }
            }
        }

        Ok(all_results)
    }

    /// Step 3: Verdict generation — asks only for verdict + reasoning (2 fields).
    /// Uses streaming on the first attempt to deliver reasoning tokens via SSE
    /// for the live frontend display. Retry attempts use non-streaming.
    ///
    /// When `error_feedback` is Some, the error is appended with the expected
    /// JSON schema repeated for clarity.
    async fn generate_verdict_core(
        &self,
        row: &CitationRow,
        claim: &str,
        passages_text: &str,
        cross_corpus_text: &str,
        error_feedback: Option<&str>,
    ) -> Result<String, AgentError> {
        let year_str = row.year.as_deref().unwrap_or("unknown");

        let mut prompt = VERDICT_PROMPT
            .replace("{CLAIM}", claim)
            .replace("{AUTHOR}", &row.author)
            .replace("{YEAR}", year_str)
            .replace("{TITLE}", &row.title)
            .replace("{PASSAGES}", passages_text)
            .replace("{CROSS_CORPUS_PASSAGES}", cross_corpus_text);

        if row.matched_file_id.is_none() {
            prompt.push_str(
                "\n\nIMPORTANT: No specific PDF was pre-identified for this \
                 citation. The passages above are from a corpus-wide search. \
                 Check the source file name in each passage header. If no \
                 passages match the cited work, use verdict 'not_found'.",
            );
        }

        if let Some(error) = error_feedback {
            prompt.push_str(&format!(
                "\n\nYour previous response was invalid: {error}\n\
                 You MUST include both \"verdict\" and \"reasoning\" fields.\n\
                 The \"reasoning\" field must contain 2-3 sentences explaining your verdict."
            ));
        }

        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: VERDICT_SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: prompt,
            },
        ];

        // Retry attempts use non-streaming for lower latency.
        if error_feedback.is_some() {
            let response = self.llm.chat_completion(&messages).await?;
            return Ok(response.content);
        }

        let job_id = self.job_id.clone();
        let row_id = row.id;
        let tx = self.citation_tx.clone();

        let on_token: Box<dyn Fn(&str) + Send + Sync> = Box::new(move |token: &str| {
            if let Some(sender) = &tx {
                let event = serde_json::json!({
                    "event": "citation_reasoning_token",
                    "job_id": job_id,
                    "row_id": row_id,
                    "token": token,
                });
                let _ = sender.send(event.to_string());
            }
        });

        let response = self
            .llm
            .chat_completion_streaming(&messages, on_token)
            .await?;

        Ok(response.content)
    }

    /// Step 4: Confidence scoring — asks the LLM for a single number.
    /// Falls back to a default confidence if parsing fails (non-fatal).
    async fn generate_confidence(&self, claim: &str, verdict_str: &str, reasoning: &str) -> f64 {
        let prompt = CONFIDENCE_PROMPT
            .replace("{CLAIM}", claim)
            .replace("{VERDICT}", verdict_str)
            .replace("{REASONING}", reasoning);

        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: CONFIDENCE_SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: prompt,
            },
        ];

        let response = match self.llm.chat_completion(&messages).await {
            Ok(r) => r,
            Err(e) => {
                debug!(error = %e, "Confidence LLM call failed, using default 0.70");
                return 0.70;
            }
        };

        let content = strip_think_blocks(&response.content);

        // Try parsing as JSON object with "confidence" field.
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(content.as_ref())
            && let Some(c) = parsed["confidence"].as_f64()
        {
            return c.clamp(0.0, 1.0);
        }

        // Try parsing as a bare number (some models just output "0.85").
        if let Ok(c) = content.trim().parse::<f64>() {
            return c.clamp(0.0, 1.0);
        }

        // Try extracting a number from the response text by finding a
        // float-like substring (e.g., "0.85" or "1.0" embedded in prose).
        for word in content.split_whitespace() {
            let trimmed = word.trim_matches(|c: char| !c.is_ascii_digit() && c != '.');
            if let Ok(c) = trimmed.parse::<f64>()
                && (0.0..=1.0).contains(&c)
            {
                return c;
            }
        }

        debug!(response = %response.content, "Could not parse confidence, using default 0.70");
        0.70
    }

    /// Phase-2 metadata generation: asks the LLM for non-critical fields
    /// (passage_locations, latex_correction, better_source) using the already
    /// established verdict and reasoning as context. This call uses
    /// non-streaming since the tokens are not displayed to the user.
    /// Returns None on any failure so the caller can fall back to defaults.
    async fn generate_verdict_metadata(
        &self,
        verdict_str: &str,
        reasoning: &str,
        passages_text: &str,
        cross_corpus_text: &str,
    ) -> Option<serde_json::Value> {
        let prompt = VERDICT_METADATA_PROMPT
            .replace("{VERDICT}", verdict_str)
            .replace("{REASONING}", reasoning)
            .replace("{PASSAGES}", passages_text)
            .replace("{CROSS_CORPUS_PASSAGES}", cross_corpus_text);

        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are a citation metadata annotator. Respond with valid JSON only."
                    .to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: prompt,
            },
        ];

        let response = match self.llm.chat_completion(&messages).await {
            Ok(r) => r,
            Err(e) => {
                debug!(error = %e, "Phase-2 metadata LLM call failed, using defaults");
                return None;
            }
        };

        let cleaned = extract_json_object(&response.content);
        let repaired = repair_truncated_json(&cleaned);
        serde_json::from_str(&repaired).ok()
    }

    /// Multi-step verdict pipeline with retry loop.
    ///
    /// Step 3: verdict + reasoning (2-field JSON, with retry)
    /// Step 4: confidence (single number, separate LLM call, non-fatal)
    /// Step 5: flag + source_match (programmatic, no LLM call)
    /// Step 6: metadata (passage_locations, latex_correction, better_source, non-fatal)
    #[allow(clippy::too_many_arguments)]
    async fn generate_verdict_with_retry(
        &self,
        row: &CitationRow,
        claim: &str,
        claim_original: &str,
        claim_english: &str,
        passages_text: &str,
        cross_corpus_text: &str,
        search_rounds: i64,
        all_passages: &[SearchResult],
        cross_results: &[SearchResult],
        file_id: Option<i64>,
    ) -> Result<SubmitEntry, AgentError> {
        let max_attempts = self.config.max_retry_attempts;
        let mut feedback: Option<String> = None;

        // Step 3: verdict + reasoning with retry loop (only 2 fields to parse).
        // If all attempts fail, fall back to "unverifiable" instead of killing the row.
        let (verdict, reasoning) = {
            let mut result: Option<(Verdict, String)> = None;
            for attempt in 0..max_attempts {
                // LLM errors (network, 500, timeout) are retried, not fatal.
                let verdict_json = match self
                    .generate_verdict_core(
                        row,
                        claim,
                        passages_text,
                        cross_corpus_text,
                        feedback.as_deref(),
                    )
                    .await
                {
                    Ok(json) => json,
                    Err(e) => {
                        warn!(
                            row_id = row.id,
                            attempt = attempt + 1,
                            error = %e,
                            "LLM call failed, retrying"
                        );
                        feedback = Some(format!("LLM error: {e}"));
                        continue;
                    }
                };

                match self.parse_verdict_reasoning(&verdict_json) {
                    Ok(r) => {
                        result = Some(r);
                        break;
                    }
                    Err(e) => {
                        let error_msg = e.to_string();
                        warn!(
                            row_id = row.id,
                            attempt = attempt + 1,
                            error = %error_msg,
                            "Verdict parse failed, retrying"
                        );
                        feedback = Some(error_msg);
                    }
                }
            }
            // If all attempts failed, use a fallback verdict instead of erroring.
            result.unwrap_or_else(|| {
                error!(
                    row_id = row.id,
                    "Verdict failed after {max_attempts} attempts, using fallback"
                );
                (
                    Verdict::Unverifiable,
                    format!(
                        "The verification agent could not produce a verdict after {} attempts. \
                         The LLM did not return a parseable response.",
                        max_attempts
                    ),
                )
            })
        };

        let verdict_str = format!("{:?}", verdict).to_lowercase();

        // Step 4: confidence (separate LLM call, non-fatal — defaults to 0.70).
        let confidence_raw = self
            .generate_confidence(claim, &verdict_str, &reasoning)
            .await;
        let confidence = enforce_confidence_caps(verdict, confidence_raw);

        // Step 5: flag and source_match — derived programmatically, no LLM.
        // source_match means "a matched PDF was searched and the verdict does
        // not indicate absence or wrong source". It does NOT guarantee the
        // matched file is truly the cited work — only that it was searched.
        let source_match =
            file_id.is_some() && !matches!(verdict, Verdict::NotFound | Verdict::WrongSource);

        let flag = derive_flag(verdict, confidence, &reasoning);

        // Build passage refs from search results.
        let passage_refs: Vec<PassageRef> = all_passages
            .iter()
            .take(5)
            .map(|r| PassageRef {
                file_id: file_id.unwrap_or(r.citation.file_id),
                page: r.citation.page_start as i64,
                passage_text: r.content.chars().take(500).collect(),
                relevance_score: r.vector_score,
                passage_location: PassageLocation::BodyText,
                source_chunk_id: None,
            })
            .collect();

        // Build other_source_list from cross-corpus results.
        let other_sources = build_other_source_list(cross_results, file_id.unwrap_or(-1));

        // Downgrade verdict if other_source_list is required but empty.
        let original_verdict = verdict;
        let verdict = if other_sources.is_empty() {
            match verdict {
                Verdict::Partial | Verdict::WrongSource | Verdict::PeripheralMatch => {
                    tracing::warn!(
                        row_id = row.id,
                        cite_key = %row.cite_key,
                        original_verdict = ?verdict,
                        "downgrading verdict to unsupported: no cross-corpus results"
                    );
                    Verdict::Unsupported
                }
                other => other,
            }
        } else {
            verdict
        };

        let confidence = if verdict != original_verdict {
            (confidence - 0.15).max(0.50)
        } else {
            confidence
        };

        let mut entry = SubmitEntry {
            row_id: row.id,
            verdict,
            claim_original: claim_original.to_string(),
            claim_english: claim_english.to_string(),
            source_match,
            other_source_list: other_sources,
            passages: passage_refs,
            reasoning,
            confidence,
            search_rounds,
            flag,
            better_source: Vec::new(),
            latex_correction: LatexCorrection {
                correction_type: CorrectionType::None,
                original_text: String::new(),
                suggested_text: String::new(),
                explanation: String::new(),
            },
        };

        // Step 6: non-critical metadata (passage_locations, latex_correction,
        // better_source). Failure is non-fatal; defaults are used instead.
        let metadata = self
            .generate_verdict_metadata(
                &verdict_str,
                &entry.reasoning,
                passages_text,
                &format_passages_for_prompt(cross_results),
            )
            .await;

        if let Some(ref meta) = metadata {
            if let Some(locations) = meta["passage_locations"].as_array() {
                let llm_locations: Vec<String> = locations
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                for (i, loc_str) in llm_locations.iter().enumerate() {
                    if let Some(p) = entry.passages.get_mut(i) {
                        p.passage_location = parse_passage_location(loc_str);
                    }
                }
            }

            if let Some(lc) = meta.get("latex_correction") {
                let ct_str = lc["correction_type"].as_str().unwrap_or("none");
                let correction_type = match ct_str {
                    "rephrase" => CorrectionType::Rephrase,
                    "add_context" => CorrectionType::AddContext,
                    "replace_citation" => CorrectionType::ReplaceCitation,
                    _ => CorrectionType::None,
                };
                entry.latex_correction = LatexCorrection {
                    correction_type,
                    original_text: lc["original_text"].as_str().unwrap_or("").to_string(),
                    suggested_text: lc["suggested_text"].as_str().unwrap_or("").to_string(),
                    explanation: lc["explanation"].as_str().unwrap_or("").to_string(),
                };
            }

            if let Some(bs) = meta["better_source"].as_array() {
                entry.better_source = bs
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
            }
        }

        Ok(entry)
    }

    /// Parses the verdict + reasoning JSON (only 2 required fields).
    /// Returns (Verdict, reasoning_string) on success.
    fn parse_verdict_reasoning(&self, verdict_json: &str) -> Result<(Verdict, String), AgentError> {
        // Guard against empty responses (e.g., streaming returned nothing).
        if verdict_json.trim().is_empty() {
            return Err(AgentError::VerdictParseFailure {
                raw_response: "LLM returned an empty response. \
                    Respond with: {\"verdict\": \"supported\", \"reasoning\": \"your explanation\"}"
                    .to_string(),
            });
        }

        let cleaned = extract_json_object(verdict_json);
        let repaired = repair_truncated_json(&cleaned);

        let parsed: serde_json::Value =
            serde_json::from_str(&repaired).map_err(|e| AgentError::VerdictParseFailure {
                raw_response: format!(
                    "JSON parse error: {e}. Respond with: \
                     {{\"verdict\": \"supported\", \"reasoning\": \"your explanation\"}}"
                ),
            })?;

        let verdict_str =
            parsed["verdict"]
                .as_str()
                .ok_or_else(|| AgentError::VerdictParseFailure {
                    raw_response: "Missing \"verdict\" field. Must be one of: \
                    supported, partial, unsupported, wrong_source, unverifiable, \
                    peripheral_match."
                        .to_string(),
                })?;
        let verdict = parse_verdict_string(verdict_str);

        // Try multiple common key names that models use for the reasoning field.
        let reasoning = parsed["reasoning"]
            .as_str()
            .or_else(|| parsed["explanation"].as_str())
            .or_else(|| parsed["justification"].as_str())
            .or_else(|| parsed["rationale"].as_str())
            .or_else(|| parsed["reason"].as_str())
            .or_else(|| parsed["analysis"].as_str())
            .or_else(|| parsed["description"].as_str())
            .map(|s| s.trim().to_string());

        // If no known key found, try to extract any string value > 20 chars
        // from the JSON object (the model might use a non-standard key).
        let reasoning = reasoning.filter(|s| !s.is_empty()).unwrap_or_else(|| {
            if let Some(obj) = parsed.as_object() {
                for (key, val) in obj {
                    if key == "verdict" {
                        continue;
                    }
                    if let Some(s) = val.as_str()
                        && s.len() > 20
                    {
                        info!(key = %key, "Using non-standard field as reasoning");
                        return s.to_string();
                    }
                }
            }
            String::new()
        });

        // If still no reasoning found, accept the verdict with a default.
        // A verdict without reasoning is still useful — don't fail the row
        // just because a small model forgot to include the explanation.
        if reasoning.trim().is_empty() {
            warn!(
                raw_json = %repaired,
                verdict = %verdict_str,
                "Verdict JSON has no reasoning field, using default"
            );
            let default_reasoning = format!(
                "Verdict '{}' was assigned by the verification agent. \
                 No detailed reasoning was provided by the model.",
                verdict_str
            );
            return Ok((verdict, default_reasoning));
        }

        Ok((verdict, reasoning))
    }

    // Kept for backwards compat — this method is no longer called but tests
    // may reference it. The new pipeline uses parse_verdict_reasoning instead.
    #[allow(dead_code)]
    fn parse_verdict_core_legacy(
        &self,
        row: &CitationRow,
        verdict_json: &str,
        passages: &[SearchResult],
        cross_results: &[SearchResult],
        file_id: Option<i64>,
        search_rounds: i64,
    ) -> Result<SubmitEntry, AgentError> {
        let cleaned = extract_json_object(verdict_json);
        let repaired = repair_truncated_json(&cleaned);

        let parsed: serde_json::Value =
            serde_json::from_str(&repaired).map_err(|e| AgentError::VerdictParseFailure {
                raw_response: format!(
                    "JSON parse error: {e}. Respond with a single valid JSON object \
                     containing: verdict, reasoning."
                ),
            })?;

        let verdict_str =
            parsed["verdict"]
                .as_str()
                .ok_or_else(|| AgentError::VerdictParseFailure {
                    raw_response: "Missing required field \"verdict\". Must be one of: \
                    supported, partial, unsupported, wrong_source, unverifiable, \
                    peripheral_match."
                        .to_string(),
                })?;
        let verdict = parse_verdict_string(verdict_str);

        let confidence_raw =
            parsed["confidence"]
                .as_f64()
                .ok_or_else(|| AgentError::VerdictParseFailure {
                    raw_response: "Missing or non-numeric \"confidence\" field. \
                    Must be a float between 0.0 and 1.0."
                        .to_string(),
                })?;

        let reasoning = parsed["reasoning"]
            .as_str()
            .ok_or_else(|| AgentError::VerdictParseFailure {
                raw_response: "Missing required field \"reasoning\". Provide 2-4 sentences \
                    justifying the verdict."
                    .to_string(),
            })?
            .to_string();

        let source_match = parsed["source_match"].as_bool().unwrap_or(false);

        let claim_original = parsed["claim_original"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| row.tex_context.as_deref().unwrap_or("").to_string());

        let claim_english = parsed["claim_english"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| claim_original.clone());

        let flag_str = parsed["flag"].as_str().unwrap_or("");
        let flag = match flag_str {
            "critical" | "warning" => flag_str.to_string(),
            _ => String::new(),
        };

        // -- Enforce confidence caps per verdict type (Rust safety net) --
        let confidence = enforce_confidence_caps(verdict, confidence_raw);

        // Build PassageRef list from actual search results. When a specific
        // file was matched deterministically, all passages share that file_id.
        // When no file was matched (corpus-wide search), each passage uses the
        // file_id from its own search result.
        let passage_refs: Vec<PassageRef> = passages
            .iter()
            .take(5)
            .map(|r| PassageRef {
                file_id: file_id.unwrap_or(r.citation.file_id),
                page: r.citation.page_start as i64,
                passage_text: r.content.chars().take(500).collect(),
                relevance_score: r.vector_score,
                passage_location: PassageLocation::BodyText,
                source_chunk_id: None,
            })
            .collect();

        // Build other_source_list from cross-corpus results, excluding the
        // matched file. When no file was matched, use -1 as the exclude_id
        // so no results are filtered out.
        let other_sources = build_other_source_list(cross_results, file_id.unwrap_or(-1));

        // The DB validation requires other_source_list to be non-empty for
        // partial, wrong_source, and peripheral_match verdicts. When the
        // cross-corpus search found no results from other files, downgrade
        // the verdict to unsupported so the submit does not fail with an
        // InvalidRowState error. This situation occurs when the corpus is
        // small or all retrieved passages belong to the same matched file.
        let original_verdict = verdict;
        let verdict = if other_sources.is_empty() {
            match verdict {
                Verdict::Partial | Verdict::WrongSource | Verdict::PeripheralMatch => {
                    tracing::warn!(
                        row_id = row.id,
                        cite_key = %row.cite_key,
                        original_verdict = ?verdict,
                        "downgrading verdict to unsupported: cross-corpus search \
                         returned no results outside the matched file, \
                         other_source_list requirement cannot be satisfied"
                    );
                    Verdict::Unsupported
                }
                other => other,
            }
        } else {
            verdict
        };

        // When the verdict was downgraded from a nuanced classification
        // (partial, wrong_source, peripheral_match) to unsupported, the
        // original confidence reflected certainty in the pre-downgrade
        // verdict which no longer applies. Reduce by 0.15 and floor at
        // 0.50 (the minimum for unsupported under the meta-confidence model).
        let confidence = if verdict != original_verdict {
            (confidence - 0.15).max(0.50)
        } else {
            confidence
        };

        Ok(SubmitEntry {
            row_id: row.id,
            verdict,
            claim_original,
            claim_english,
            source_match,
            other_source_list: other_sources,
            passages: passage_refs,
            reasoning,
            confidence,
            search_rounds,
            flag,
            better_source: Vec::new(),
            latex_correction: LatexCorrection {
                correction_type: CorrectionType::None,
                original_text: String::new(),
                suggested_text: String::new(),
                explanation: String::new(),
            },
        })
    }

    /// Submits a single verification result for one citation row.
    /// Wraps the result in a one-element slice and calls the DB submit function.
    fn submit_single_result(&self, entry: &SubmitEntry) -> Result<(), AgentError> {
        let conn = self
            .state
            .pool
            .get()
            .map_err(|e| AgentError::Database(format!("connection pool error: {e}")))?;

        cit_db::submit_batch_results(&conn, &self.job_id, std::slice::from_ref(entry))
            .map_err(|e| AgentError::Database(format!("submit failed: {e}")))?;

        Ok(())
    }

    /// Marks a row as failed in the database by updating its status directly.
    /// Called when row processing encounters an unrecoverable error. Emits
    /// an SSE "error" phase event so the frontend transitions the row out
    /// of any intermediate phase (searching, evaluating, reasoning).
    fn mark_row_failed(&self, row_id: i64, cite_key: &str, error_msg: &str) {
        let conn = match self.state.pool.get() {
            Ok(c) => c,
            Err(e) => {
                error!(row_id, error = %e, "Cannot mark row failed: pool error");
                return;
            }
        };

        let truncated_msg: String = error_msg.chars().take(500).collect();
        let result = conn.execute(
            "UPDATE citation_row SET status = 'failed', result_json = ?1 WHERE id = ?2 AND job_id = ?3",
            rusqlite::params![
                serde_json::json!({"error": truncated_msg}).to_string(),
                row_id,
                self.job_id,
            ],
        );

        if let Err(e) = result {
            error!(row_id, error = %e, "Failed to mark citation row as failed");
        }

        // Update job progress counters after marking a row as failed.
        if let Err(e) = conn.execute(
            "UPDATE job SET progress_done = (SELECT COUNT(*) FROM citation_row WHERE job_id = ?1 AND status IN ('done', 'failed')) WHERE id = ?1",
            rusqlite::params![self.job_id],
        ) {
            warn!(job_id = %self.job_id, error = %e, "failed to update job progress counters");
        }

        // Emit SSE so the frontend moves the row to the "error" phase.
        emit_sse(
            &self.citation_tx,
            serde_json::json!({
                "event": "citation_row_update",
                "job_id": self.job_id,
                "row_id": row_id,
                "cite_key": cite_key,
                "phase": "error",
                "error_message": truncated_msg,
            }),
        );
    }

    /// Emits a progress SSE event with current row counts and verdict distribution.
    async fn emit_progress(&self) {
        let conn = match self.state.pool.get() {
            Ok(c) => c,
            Err(_) => return,
        };

        let counts = match cit_db::count_by_status(&conn, &self.job_id) {
            Ok(c) => c,
            Err(_) => return,
        };

        let total = counts.pending + counts.claimed + counts.done + counts.failed;
        let rows_done = counts.done + counts.failed;

        let verdicts = cit_db::count_verdicts(&conn, &self.job_id).unwrap_or_default();

        emit_sse(
            &self.citation_tx,
            serde_json::json!({
                "event": "citation_job_progress",
                "job_id": self.job_id,
                "rows_done": rows_done,
                "rows_total": total,
                "verdicts": verdicts,
            }),
        );
    }

    /// Checks whether all rows in the job are done or failed, and if so,
    /// transitions the job to the Completed state.
    async fn check_job_completion(&self) {
        let conn = match self.state.pool.get() {
            Ok(c) => c,
            Err(_) => return,
        };

        let counts = match cit_db::count_by_status(&conn, &self.job_id) {
            Ok(c) => c,
            Err(_) => return,
        };

        let is_complete = counts.pending == 0 && counts.claimed == 0;
        if is_complete {
            if let Err(e) = store::update_job_state(&conn, &self.job_id, JobState::Completed, None)
            {
                warn!(job_id = %self.job_id, error = %e, "failed to update citation job state to completed");
            }
            info!(job_id = %self.job_id, "Citation job completed");
        }
    }

    /// Downloads cited source PDFs and HTML pages from BibTeX URL/DOI fields
    /// into the source directory and indexes them into the session. Uses the
    /// neuroncite-html crate for URL classification, PDF downloading, and
    /// HTML fetching. Errors on individual sources are logged but do not
    /// abort the verification run.
    async fn fetch_and_index_sources(&self, source_dir: &str) {
        info!(
            job_id = %self.job_id,
            source_dir = %source_dir,
            "Fetching cited sources from BibTeX URL/DOI fields"
        );

        emit_sse(
            &self.citation_tx,
            serde_json::json!({
                "event": "citation_job_progress",
                "job_id": self.job_id,
                "rows_done": 0,
                "rows_total": 0,
                "verdicts": {},
                "message": "Downloading cited sources...",
            }),
        );

        // Read the bib_path from the job's params_json.
        let bib_path = {
            let conn = match self.state.pool.get() {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "Cannot read bib_path: pool error");
                    return;
                }
            };
            // Read the bib_path from the job record's params_json field.
            // Using get_job() avoids inline SQL and reuses the existing store function.
            let params_json: Option<String> = neuroncite_store::get_job(&conn, &self.job_id)
                .ok()
                .and_then(|row| row.params_json);
            match params_json {
                Some(json) => serde_json::from_str::<serde_json::Value>(&json)
                    .ok()
                    .and_then(|v| v["bib_path"].as_str().map(String::from))
                    .unwrap_or_default(),
                None => return,
            }
        };

        if bib_path.is_empty() {
            warn!(job_id = %self.job_id, "No bib_path in job params, skipping source fetch");
            return;
        }

        // Parse the BibTeX file on a blocking thread.
        let bib_path_clone = bib_path.clone();
        let bib_entries = match tokio::task::spawn_blocking(move || {
            let content = std::fs::read_to_string(&bib_path_clone)
                .map_err(|e| format!("failed to read bib file: {e}"))?;
            Ok::<_, String>(neuroncite_citation::bibtex::parse_bibtex(&content))
        })
        .await
        {
            Ok(Ok(entries)) => entries,
            Ok(Err(e)) => {
                error!(error = %e, "BibTeX parsing failed");
                return;
            }
            Err(e) => {
                error!(error = %e, "BibTeX parsing task panicked");
                return;
            }
        };

        // Build the HTTP client before URL resolution because resolve_doi
        // needs it for API requests to Unpaywall/Semantic Scholar/OpenAlex.
        let client = match neuroncite_html::build_http_client() {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "HTTP client initialization failed");
                return;
            }
        };

        // Resolve URLs for entries: prefer the explicit `url` field, then try
        // the multi-source DOI resolution chain (Unpaywall -> Semantic Scholar
        // -> OpenAlex -> doi.org). Entries without url or doi are skipped.
        struct SourceEntry {
            cite_key: String,
            url: String,
        }
        let mut sources: Vec<SourceEntry> = Vec::new();
        for (cite_key, entry) in &bib_entries {
            if let Some(url) = &entry.url {
                sources.push(SourceEntry {
                    cite_key: cite_key.clone(),
                    url: url.clone(),
                });
            } else if let Some(doi) = &entry.doi {
                let resolved = neuroncite_html::resolve_doi(
                    &client,
                    doi,
                    self.config.unpaywall_email.as_deref(),
                )
                .await;
                info!(
                    cite_key = %cite_key,
                    doi = %doi,
                    source = %resolved.source,
                    url = %resolved.url,
                    "DOI resolved"
                );
                sources.push(SourceEntry {
                    cite_key: cite_key.clone(),
                    url: resolved.url,
                });
            }
        }

        if sources.is_empty() {
            info!(job_id = %self.job_id, "No BibTeX entries with URL/DOI fields, nothing to fetch");
            return;
        }

        // Create the output directory.
        let output_dir = std::path::Path::new(source_dir);
        if let Err(e) = std::fs::create_dir_all(output_dir) {
            error!(error = %e, dir = %source_dir, "Failed to create source directory");
            return;
        }

        let mut downloaded_pdfs: Vec<String> = Vec::new();
        let delay = std::time::Duration::from_millis(1000);

        // Fetch each source with rate limiting.
        for (i, source) in sources.iter().enumerate() {
            if i > 0 {
                tokio::time::sleep(delay).await;
            }

            let source_type = neuroncite_html::classify_url(&client, &source.url).await;

            match source_type {
                neuroncite_html::UrlSourceType::Pdf => {
                    // Build a descriptive filename from BibTeX metadata for
                    // the token-overlap matching algorithm.
                    let bib_entry = bib_entries.get(&source.cite_key);
                    let pdf_filename = neuroncite_html::build_source_filename(
                        bib_entry.map(|e| e.title.as_str()).unwrap_or(""),
                        bib_entry.map(|e| e.author.as_str()).unwrap_or(""),
                        bib_entry.and_then(|e| e.year.as_deref()),
                        &source.cite_key,
                    );
                    match neuroncite_html::download_pdf(
                        &client,
                        &source.url,
                        output_dir,
                        &pdf_filename,
                    )
                    .await
                    {
                        Ok(pdf_path) => {
                            info!(
                                cite_key = %source.cite_key,
                                path = %pdf_path.display(),
                                "PDF downloaded"
                            );
                            downloaded_pdfs.push(pdf_path.display().to_string());
                        }
                        Err(e) => {
                            warn!(cite_key = %source.cite_key, url = %source.url, error = %e, "PDF download failed");
                        }
                    }
                }
                neuroncite_html::UrlSourceType::Html => {
                    // HTML sources are not yet indexable — only PDFs can be
                    // extracted, chunked, and embedded into the search index.
                    // Log a warning so the user knows this source was skipped.
                    warn!(
                        cite_key = %source.cite_key,
                        url = %source.url,
                        "Skipping HTML source: only PDF sources can be indexed for verification"
                    );
                }
            }
        }

        info!(
            job_id = %self.job_id,
            pdfs = downloaded_pdfs.len(),
            "Source fetching complete"
        );
    }

    /// Returns the primary session ID (first in the list). Used for source
    /// indexing operations where new files need a single target session.
    fn primary_session_id(&self) -> i64 {
        self.session_ids[0]
    }

    /// Indexes downloaded PDF files into the primary session using the
    /// extraction, chunking, and embedding pipeline. Each file is processed
    /// individually so failures on one file do not block others. Called by
    /// the auto-verify agent to index source PDFs before verification begins.
    async fn index_downloaded_pdfs(&self, pdf_paths: &[String]) {
        let session = {
            let conn = match self.state.pool.get() {
                Ok(c) => c,
                Err(_) => return,
            };
            match store::get_session(&conn, self.primary_session_id()) {
                Ok(s) => s,
                Err(_) => return,
            }
        };

        let chunk_size = match session.chunk_size_usize() {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "chunk_size integer overflow");
                return;
            }
        };
        let chunk_overlap = match session.chunk_overlap_usize() {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "chunk_overlap integer overflow");
                return;
            }
        };
        let max_words = match session.max_words_usize() {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "max_words integer overflow");
                return;
            }
        };
        let tokenizer_json = self.state.worker_handle.tokenizer_json();

        for pdf_path_str in pdf_paths {
            let pdf_path = std::path::Path::new(pdf_path_str).to_path_buf();

            let strategy = match neuroncite_chunk::create_strategy(
                &session.chunk_strategy,
                chunk_size,
                chunk_overlap,
                max_words,
                tokenizer_json.as_deref(),
            ) {
                Ok(s) => s,
                Err(e) => {
                    warn!(path = %pdf_path_str, error = %e, "Chunking strategy creation failed");
                    continue;
                }
            };

            let extracted = match tokio::task::spawn_blocking(move || {
                crate::indexer::extract_and_chunk_file(strategy.as_ref(), &pdf_path)
            })
            .await
            {
                Ok(Ok(e)) => e,
                Ok(Err(e)) => {
                    warn!(path = %pdf_path_str, error = %e, "PDF extraction failed");
                    continue;
                }
                Err(e) => {
                    warn!(path = %pdf_path_str, error = %e, "PDF extraction task panicked");
                    continue;
                }
            };

            if extracted.chunks.is_empty() {
                warn!(path = %pdf_path_str, "PDF extraction produced zero chunks");
                continue;
            }

            match crate::indexer::embed_and_store_file_async(
                &self.state.pool,
                &self.state.worker_handle,
                &extracted,
                self.primary_session_id(),
            )
            .await
            {
                Ok(result) => {
                    info!(
                        path = %pdf_path_str,
                        chunks = result.chunks_created,
                        "PDF indexed into session"
                    );
                }
                Err(e) => {
                    warn!(path = %pdf_path_str, error = %e, "PDF indexing failed");
                }
            }
        }
    }

    /// Rebuilds the HNSW index for the primary session after source documents
    /// have been added. Loads all embedding blobs from SQLite, converts them
    /// to f32 vectors, builds a new HNSW index, and atomically swaps it into
    /// the AppState per-session map. Required so searches during verification
    /// can find the freshly indexed sources.
    async fn rebuild_hnsw(&self) {
        let primary_sid = self.primary_session_id();
        let session = {
            let conn = match self.state.pool.get() {
                Ok(c) => c,
                Err(_) => return,
            };
            match store::get_session(&conn, primary_sid) {
                Ok(s) => s,
                Err(_) => return,
            }
        };

        let raw_embeddings = {
            let conn = match self.state.pool.get() {
                Ok(c) => c,
                Err(_) => return,
            };
            match store::load_embeddings_for_hnsw(&conn, primary_sid) {
                Ok(e) => e,
                Err(e) => {
                    warn!(error = %e, "Failed to load embeddings for HNSW rebuild");
                    return;
                }
            }
        };

        if raw_embeddings.is_empty() {
            return;
        }

        let vector_dim = match session.vector_dimension_usize() {
            Ok(d) => d,
            Err(e) => {
                warn!(error = %e, "vector_dimension integer overflow");
                return;
            }
        };
        let session_id = primary_sid;
        let state = Arc::clone(&self.state);

        // Convert raw byte blobs to f32 vectors, then build the HNSW index
        // inside spawn_blocking (same pattern as executor.rs).
        match tokio::task::spawn_blocking(move || {
            let f32_vectors: Vec<(i64, Vec<f32>)> = raw_embeddings
                .into_iter()
                .map(|(id, bytes)| (id, crate::indexer::bytes_to_f32_vec(&bytes)))
                .collect();

            let labeled_refs: Vec<(i64, &[f32])> = f32_vectors
                .iter()
                .map(|(id, v)| (*id, v.as_slice()))
                .collect();

            store::build_hnsw(&labeled_refs, vector_dim)
        })
        .await
        {
            Ok(Ok(hnsw)) => {
                state.insert_hnsw(session_id, hnsw);
                info!(session_id, "HNSW index rebuilt after source indexing");
            }
            Ok(Err(e)) => {
                warn!(error = %e, "HNSW build failed");
            }
            Err(e) => {
                warn!(error = %e, "HNSW rebuild task panicked");
            }
        }
    }

    /// Indexes any PDFs from the source directory that are not yet in any
    /// of the selected sessions. New files are indexed into the primary
    /// session. This handles the case where PDFs were placed in the source
    /// directory before auto-verify was called (e.g., by the user manually
    /// copying them, or by a prior fetch step).
    async fn index_source_directory_if_needed(&self, source_dir: &str) {
        let source_path = std::path::Path::new(source_dir);
        if !source_path.is_dir() {
            warn!(dir = %source_dir, "Source directory does not exist, skipping auto-index");
            return;
        }

        // Discover PDFs in the source directory.
        let pdfs = match neuroncite_pdf::discover_pdfs_flat(source_path) {
            Ok(p) => p,
            Err(e) => {
                warn!(dir = %source_dir, error = %e, "PDF discovery in source directory failed");
                return;
            }
        };
        if pdfs.is_empty() {
            debug!(dir = %source_dir, "No PDFs found in source directory");
            return;
        }

        // Check which PDFs are already indexed across all selected sessions.
        let already_indexed: std::collections::HashSet<String> = {
            let conn = match self.state.pool.get() {
                Ok(c) => c,
                Err(_) => return,
            };
            let mut indexed = std::collections::HashSet::new();
            for &sid in &self.session_ids {
                if let Ok(files) = store::list_files_by_session(&conn, sid) {
                    indexed.extend(files.iter().map(|f| f.file_path.clone()));
                }
            }
            indexed
        };

        let unindexed: Vec<String> = pdfs
            .iter()
            .filter(|p| !already_indexed.contains(&p.display().to_string()))
            .map(|p| p.display().to_string())
            .collect();

        if unindexed.is_empty() {
            debug!(dir = %source_dir, "All source PDFs already indexed");
            return;
        }

        info!(
            dir = %source_dir,
            count = unindexed.len(),
            "Indexing unindexed source PDFs into session"
        );

        self.index_downloaded_pdfs(&unindexed).await;
        self.rebuild_hnsw().await;
    }

    /// Re-matches unmatched citation rows against the updated file lists
    /// from all selected sessions. After source fetching and indexing, some
    /// rows that previously had matched_file_id = None may now have a
    /// matching file. Uses the same token-overlap matching algorithm as
    /// the citation_create handler.
    async fn rematch_unmatched_rows(&self) {
        let conn = match self.state.pool.get() {
            Ok(c) => c,
            Err(_) => return,
        };

        // Load file lists from all selected sessions and aggregate them.
        let mut all_indexed_files = Vec::new();
        for &sid in &self.session_ids {
            if let Ok(files) = store::list_files_by_session(&conn, sid) {
                all_indexed_files.extend(files);
            }
        }
        let indexed_files = all_indexed_files;

        let file_lookup: Vec<(i64, String)> = indexed_files
            .iter()
            .map(|f| {
                let filename = std::path::Path::new(&f.file_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                (f.id, filename)
            })
            .collect();

        if file_lookup.is_empty() {
            return;
        }

        // Load all rows for this job and find unmatched ones.
        let rows = match cit_db::list_all_rows(&conn, &self.job_id) {
            Ok(r) => r,
            Err(_) => return,
        };

        let mut rematched = 0usize;
        for row in &rows {
            if row.matched_file_id.is_some() {
                continue;
            }

            // Run the same token-overlap matching used by citation_create.
            let match_result = crate::handlers::citation::find_best_file_match(
                &row.author,
                &row.title,
                row.year.as_deref(),
                &file_lookup,
            );

            if let Some((file_id, score)) = match_result {
                match cit_db::update_matched_file_id(&conn, row.id, file_id) {
                    Ok(updated) if updated > 0 => {
                        rematched += 1;
                        debug!(
                            row_id = row.id,
                            cite_key = %row.cite_key,
                            file_id,
                            score,
                            "Re-matched citation row to indexed file"
                        );
                    }
                    _ => {}
                }
            }
        }

        if rematched > 0 {
            info!(
                job_id = %self.job_id,
                rematched,
                "Re-matched citation rows after source indexing"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions (module-private)
// ---------------------------------------------------------------------------

/// Builds a SubmitEntry for a citation where no file was matched
/// deterministically AND the corpus-wide search returned zero passages.
/// Records the actual search_rounds so the user can distinguish between
/// "file not indexed" and "file searched but not found". Confidence is
/// 1.0 (meta-confidence) because the corpus was exhaustively searched
/// and the source is definitively absent.
fn build_corpus_searched_not_found_entry(
    row: &CitationRow,
    claim_original: &str,
    claim_english: &str,
    search_rounds: i64,
) -> SubmitEntry {
    SubmitEntry {
        row_id: row.id,
        verdict: Verdict::NotFound,
        claim_original: claim_original.to_string(),
        claim_english: claim_english.to_string(),
        source_match: false,
        other_source_list: Vec::new(),
        passages: Vec::new(),
        reasoning: format!(
            "No specific PDF was pre-matched for '{}' ({}). A corpus-wide \
             search across all indexed documents returned no relevant passages \
             after {} search round(s).",
            row.cite_key, row.title, search_rounds
        ),
        confidence: 1.0,
        search_rounds,
        flag: String::new(),
        better_source: Vec::new(),
        latex_correction: LatexCorrection {
            correction_type: CorrectionType::None,
            original_text: String::new(),
            suggested_text: String::new(),
            explanation: String::new(),
        },
    }
}

/// Formats search results into a human-readable text block for inclusion
/// in the LLM verdict prompt. Each passage includes source file name, page
/// range, relevance score, and up to 800 characters of content. The source
/// file name is essential for cross-corpus passages so the LLM can identify
/// alternative sources for the `better_source` field.
fn format_passages_for_prompt(results: &[SearchResult]) -> String {
    if results.is_empty() {
        return "(No passages found)".to_string();
    }

    results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let score = r.vector_score;
            let page_start = r.citation.page_start;
            let page_end = r.citation.page_end;
            let source = &r.citation.file_display_name;
            let content: String = r.content.chars().take(800).collect();
            let page_range = if page_start == page_end {
                format!("page {page_start}")
            } else {
                format!("pages {page_start}-{page_end}")
            };
            format!(
                "[Passage {}] (source: \"{}\", {}, score {:.3})\n{}",
                i + 1,
                source,
                page_range,
                score,
                content
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Maps a verdict string from the LLM response to the Verdict enum.
/// Accepts snake_case and various common phrasings. Defaults to Unsupported
/// when the string is not recognized.
fn parse_verdict_string(s: &str) -> Verdict {
    match s.to_lowercase().trim() {
        "supported" => Verdict::Supported,
        "partial" => Verdict::Partial,
        "unsupported" => Verdict::Unsupported,
        "not_found" | "not found" => Verdict::NotFound,
        "wrong_source" | "wrong source" => Verdict::WrongSource,
        "unverifiable" => Verdict::Unverifiable,
        "peripheral_match" | "peripheral match" => Verdict::PeripheralMatch,
        _ => Verdict::Unsupported,
    }
}

/// Maps a passage location string to the PassageLocation enum.
/// Defaults to BodyText when the string is not recognized.
fn parse_passage_location(s: &str) -> PassageLocation {
    match s.to_lowercase().trim() {
        "abstract" => PassageLocation::Abstract,
        "foreword" => PassageLocation::Foreword,
        "table_of_contents" => PassageLocation::TableOfContents,
        "introduction" => PassageLocation::Introduction,
        "literature_review" => PassageLocation::LiteratureReview,
        "theoretical_framework" => PassageLocation::TheoreticalFramework,
        "methodology" => PassageLocation::Methodology,
        "results" => PassageLocation::Results,
        "discussion" => PassageLocation::Discussion,
        "conclusion" => PassageLocation::Conclusion,
        "bibliography" => PassageLocation::Bibliography,
        "appendix" => PassageLocation::Appendix,
        "glossary" => PassageLocation::Glossary,
        "table_or_figure" => PassageLocation::TableOrFigure,
        "footnote" => PassageLocation::Footnote,
        _ => PassageLocation::BodyText,
    }
}

/// Enforces verdict-specific confidence bounds per the meta-confidence model.
/// Each verdict type has a valid [floor, ceiling] range that reflects how
/// certain the agent must be in its classification:
/// - supported:        [0.50, 1.00] -- at least moderately sure the source supports the claim.
/// - partial:          [0.00, 0.80] -- partial support caps at 0.80 by definition.
/// - unsupported:      [0.50, 1.00] -- at least moderately sure the source does not support the claim.
/// - not_found:        [0.70, 1.00] -- absence from the corpus is a near-binary determination.
/// - wrong_source:     [0.50, 1.00] -- at least moderately sure the claim belongs to another source.
/// - unverifiable:     [0.50, 1.00] -- at least moderately sure the claim cannot be checked.
/// - peripheral_match: [0.00, 0.69] -- non-substantive location caps at the thematic-overlap band.
///
/// Floor enforcement (clamping up) is preferred over rejection because the
/// LLM already produced a verdict as the primary signal. A too-low confidence
/// typically results from the LLM using the old support-strength interpretation
/// rather than meta-confidence. The floor aligns the value without requiring
/// a costly retry.
fn enforce_confidence_caps(verdict: Verdict, raw: f64) -> f64 {
    let clamped = raw.clamp(0.0, 1.0);
    match verdict {
        Verdict::Supported => clamped.max(0.50),
        Verdict::Partial => clamped.min(0.80),
        Verdict::Unsupported => clamped.max(0.50),
        Verdict::NotFound => clamped.max(0.70),
        Verdict::WrongSource => clamped.max(0.50),
        Verdict::Unverifiable => clamped.max(0.50),
        Verdict::PeripheralMatch => clamped.min(0.69),
    }
}

/// Derives the flag field programmatically from verdict, confidence, and
/// reasoning. No LLM call needed — the rules are deterministic:
/// - "critical": reasoning mentions contradiction (heuristic keyword match)
/// - "warning": confidence < 0.70 for a positive verdict (supported, partial)
/// - "": no flag needed
fn derive_flag(verdict: Verdict, confidence: f64, reasoning: &str) -> String {
    // Check for contradiction keywords in reasoning.
    let lower = reasoning.to_lowercase();
    let contradicts = lower.contains("contradict")
        || lower.contains("widerspricht")
        || lower.contains("gegenteil")
        || lower.contains("incorrect")
        || lower.contains("falsch")
        || lower.contains("wrong");

    if contradicts && matches!(verdict, Verdict::Unsupported | Verdict::WrongSource) {
        return "critical".to_string();
    }

    if confidence < 0.70 && matches!(verdict, Verdict::Supported | Verdict::Partial) {
        return "warning".to_string();
    }

    String::new()
}

/// Removes all <think>...</think> blocks from an LLM response string.
/// Models derived from deepseek-r1 output a chain-of-thought block wrapped in
/// these tags before their actual answer. When a <think> tag has no matching
/// </think>, everything from the opening tag to the end of the string is
/// treated as think content and discarded.
fn strip_think_blocks(s: &str) -> std::borrow::Cow<'_, str> {
    if !s.contains("<think>") {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut result = String::with_capacity(s.len());
    let mut remaining = s;
    while let Some(start) = remaining.find("<think>") {
        result.push_str(&remaining[..start]);
        match remaining[start..].find("</think>") {
            Some(rel_end) => {
                // rel_end is relative to remaining[start..]. Skip past the
                // closing tag. len("</think>") == 8.
                remaining = &remaining[start + rel_end + 8..];
            }
            None => {
                // No closing tag found; discard the rest as think content.
                remaining = "";
                break;
            }
        }
    }
    result.push_str(remaining);
    std::borrow::Cow::Owned(result)
}

/// Extracts a JSON array from a string that may contain markdown code blocks
/// or surrounding text. Finds the first '[' and last ']' in the string.
fn extract_json_array(s: &str) -> String {
    if let Some(start) = s.find('[')
        && let Some(end) = s.rfind(']')
        && end > start
    {
        return s[start..=end].to_string();
    }
    s.to_string()
}

/// Extracts a JSON object from a string that may contain markdown code blocks
/// or surrounding text. Finds the first '{' and last '}' in the string.
/// When the response contains a '{' but no closing '}' (truncated JSON),
/// returns everything from the first '{' to the end of the string so the
/// downstream repair_truncated_json can close the open delimiters.
fn extract_json_object(s: &str) -> String {
    let start = match s.find('{') {
        Some(pos) => pos,
        None => return s.to_string(),
    };
    if let Some(end) = s.rfind('}')
        && end > start
    {
        return s[start..=end].to_string();
    }
    // No closing brace found: return from the first '{' to end-of-string.
    // This gives repair_truncated_json a chance to close the open delimiters.
    s[start..].to_string()
}

/// Attempts to repair JSON that was truncated mid-generation (typically when
/// the LLM hits its max_tokens limit). Handles these common truncation patterns:
///
/// 1. Unclosed string literals: a trailing `"value...` without a closing quote
/// 2. Missing closing braces/brackets: the JSON object or array was not closed
/// 3. Trailing commas before closing delimiters
///
/// Uses a stack to track the nesting order of `{` and `[` delimiters so that
/// deeply interleaved structures (e.g. `{"a": [{"b": "val...`) are closed in
/// the correct LIFO order: `"}]}` rather than the wrong `]}}`.
///
/// The repair is conservative: it only appends characters to close open
/// structures. It never removes or reorders content. If the input is already
/// valid JSON, it is returned unchanged.
fn repair_truncated_json(s: &str) -> String {
    // Fast path: if it already parses, return as-is.
    if serde_json::from_str::<serde_json::Value>(s).is_ok() {
        return s.to_string();
    }

    let mut result = s.to_string();

    // Track nesting with a stack that records the opening delimiter type.
    // Escaped quotes (\") inside string literals do not toggle in_string.
    let mut in_string = false;
    let mut escape_next = false;
    let mut stack: Vec<char> = Vec::new();

    for ch in result.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            match ch {
                '{' => stack.push('{'),
                '[' => stack.push('['),
                '}' => {
                    if stack.last() == Some(&'{') {
                        stack.pop();
                    }
                }
                ']' => {
                    if stack.last() == Some(&'[') {
                        stack.pop();
                    }
                }
                _ => {}
            }
        }
    }

    // If we ended inside a string literal, close it.
    if in_string {
        result.push('"');
    }

    // Remove trailing comma before we close delimiters. The LLM sometimes
    // writes `"field": "value",` and then gets cut off before the next field.
    if let Some(stripped) = result.trim_end().strip_suffix(',') {
        result = stripped.to_string();
    }

    // Close unclosed delimiters in reverse (LIFO) order. Each opening
    // delimiter on the stack gets its matching closing delimiter appended.
    while let Some(opener) = stack.pop() {
        match opener {
            '{' => result.push('}'),
            '[' => result.push(']'),
            _ => {}
        }
    }

    result
}

/// Builds the other_source_list from cross-corpus search results. Groups
/// results by source file, excludes the matched file_id, and takes the
/// top 3 passages per source.
/// Builds the other_source_list from cross-corpus search results. Groups
/// results by source file, excludes the matched file_id, sorts passages
/// within each group by descending score, and sorts groups by their best
/// passage score. Output is fully deterministic for identical inputs.
fn build_other_source_list(
    cross_results: &[SearchResult],
    exclude_file_id: i64,
) -> Vec<OtherSourceEntry> {
    use std::collections::BTreeMap;

    // Group by file_id using BTreeMap for deterministic iteration order.
    let mut by_file: BTreeMap<i64, Vec<&SearchResult>> = BTreeMap::new();
    for r in cross_results {
        if r.citation.file_id != exclude_file_id {
            by_file.entry(r.citation.file_id).or_default().push(r);
        }
    }

    // Sort passages within each group by descending score.
    let mut entries: Vec<OtherSourceEntry> = by_file
        .into_values()
        .map(|mut results| {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let source_name = results
                .first()
                .map(|r| {
                    r.citation
                        .source_file
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string()
                })
                .unwrap_or_else(|| "unknown".to_string());

            let passages: Vec<OtherSourcePassage> = results
                .iter()
                .take(3)
                .map(|r| OtherSourcePassage {
                    text: r.content.chars().take(300).collect(),
                    page: r.citation.page_start as i64,
                    score: r.vector_score,
                })
                .collect();

            OtherSourceEntry {
                cite_key_or_title: source_name,
                passages,
            }
        })
        .collect();

    // Sort groups by best passage score (descending) for deterministic output.
    entries.sort_by(|a, b| {
        let score_a = a.passages.first().map(|p| p.score).unwrap_or(0.0);
        let score_b = b.passages.first().map(|p| p.score).unwrap_or(0.0);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    entries
}

/// Creates an `LlmConfig` from an `AutoVerifyRequest` with sensible defaults.
/// Default max_tokens is 8192 to prevent truncation of complex JSON verdicts.
/// JSON mode is always enabled because the agent exclusively expects JSON output.
pub fn llm_config_from_request(req: &AutoVerifyRequest) -> LlmConfig {
    LlmConfig {
        base_url: req
            .ollama_url
            .clone()
            .unwrap_or_else(|| "http://localhost:11434".to_string()),
        model: req.model.clone(),
        temperature: req.temperature.unwrap_or(0.1),
        max_tokens: req.max_tokens.unwrap_or(8192),
        json_mode: true,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- repair_truncated_json tests --

    #[test]
    fn repair_valid_json_is_noop() {
        let input = r#"{"verdict": "supported", "confidence": 0.9}"#;
        assert_eq!(repair_truncated_json(input), input);
    }

    #[test]
    fn repair_missing_closing_brace() {
        let input = r#"{"verdict": "supported", "confidence": 0.9"#;
        let repaired = repair_truncated_json(input);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
        let v: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(v["verdict"], "supported");
        assert_eq!(v["confidence"], 0.9);
    }

    #[test]
    fn repair_unclosed_string_and_brace() {
        let input = r#"{"verdict": "supported", "reasoning": "The passage clearly states"#;
        let repaired = repair_truncated_json(input);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
        let v: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(v["verdict"], "supported");
        assert!(v["reasoning"].as_str().unwrap().starts_with("The passage"));
    }

    #[test]
    fn repair_trailing_comma() {
        let input = r#"{"verdict": "partial", "confidence": 0.7,"#;
        let repaired = repair_truncated_json(input);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
    }

    #[test]
    fn repair_truncated_nested_array() {
        let input = r#"{"verdict": "supported", "passage_locations": ["introduction", "results""#;
        let repaired = repair_truncated_json(input);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
        let v: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        let locs = v["passage_locations"].as_array().unwrap();
        assert_eq!(locs.len(), 2);
        assert_eq!(locs[0], "introduction");
        assert_eq!(locs[1], "results");
    }

    #[test]
    fn repair_truncated_nested_object() {
        let input = r#"{"verdict": "supported", "latex_correction": {"correction_type": "none""#;
        let repaired = repair_truncated_json(input);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
        let v: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(v["latex_correction"]["correction_type"], "none");
    }

    #[test]
    fn repair_escaped_quotes_inside_string() {
        let input = r#"{"reasoning": "The author says \"hello"#;
        let repaired = repair_truncated_json(input);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
    }

    #[test]
    fn repair_empty_string() {
        let repaired = repair_truncated_json("");
        // Empty string cannot become valid JSON, but the function should not panic.
        assert_eq!(repaired, "");
    }

    #[test]
    fn repair_deeply_nested_truncation() {
        let input = r#"{"a": {"b": [{"c": "val"#;
        let repaired = repair_truncated_json(input);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
    }

    // -- extract_json_object tests --

    #[test]
    fn extract_json_from_markdown_fence() {
        let input = "```json\n{\"verdict\": \"supported\"}\n```";
        let extracted = extract_json_object(input);
        assert_eq!(extracted, "{\"verdict\": \"supported\"}");
    }

    #[test]
    fn extract_json_with_preamble() {
        let input = "Here is my analysis:\n{\"verdict\": \"partial\"}";
        let extracted = extract_json_object(input);
        assert_eq!(extracted, "{\"verdict\": \"partial\"}");
    }

    #[test]
    fn extract_json_no_braces_returns_original() {
        let input = "no json here";
        let extracted = extract_json_object(input);
        assert_eq!(extracted, input);
    }

    // -- extract_json_array tests --

    #[test]
    fn extract_array_from_markdown() {
        let input = "```json\n[\"query1\", \"query2\"]\n```";
        let extracted = extract_json_array(input);
        assert_eq!(extracted, "[\"query1\", \"query2\"]");
    }

    // -- parse_verdict_string tests --

    #[test]
    fn parse_verdict_snake_case() {
        assert!(matches!(
            parse_verdict_string("wrong_source"),
            Verdict::WrongSource
        ));
        assert!(matches!(
            parse_verdict_string("not_found"),
            Verdict::NotFound
        ));
        assert!(matches!(
            parse_verdict_string("peripheral_match"),
            Verdict::PeripheralMatch
        ));
    }

    #[test]
    fn parse_verdict_space_separated() {
        assert!(matches!(
            parse_verdict_string("wrong source"),
            Verdict::WrongSource
        ));
        assert!(matches!(
            parse_verdict_string("not found"),
            Verdict::NotFound
        ));
    }

    #[test]
    fn parse_verdict_unknown_defaults_to_unsupported() {
        assert!(matches!(
            parse_verdict_string("gibberish"),
            Verdict::Unsupported
        ));
    }

    #[test]
    fn parse_verdict_case_insensitive() {
        assert!(matches!(
            parse_verdict_string("SUPPORTED"),
            Verdict::Supported
        ));
        assert!(matches!(parse_verdict_string("Partial"), Verdict::Partial));
    }

    // -- enforce_confidence_caps tests --

    #[test]
    fn enforce_caps_partial() {
        assert_eq!(enforce_confidence_caps(Verdict::Partial, 0.95), 0.80);
        assert_eq!(enforce_confidence_caps(Verdict::Partial, 0.75), 0.75);
    }

    #[test]
    fn enforce_caps_peripheral_match() {
        assert_eq!(
            enforce_confidence_caps(Verdict::PeripheralMatch, 0.85),
            0.69
        );
        assert_eq!(
            enforce_confidence_caps(Verdict::PeripheralMatch, 0.50),
            0.50
        );
    }

    #[test]
    fn enforce_caps_clamps_to_range() {
        // Values outside [0.0, 1.0] are clamped first, then floor/ceiling
        // is applied. For Supported: clamp(1.5) = 1.0, max(0.50) = 1.0.
        // clamp(-0.3) = 0.0, max(0.50) = 0.50 (floor enforcement).
        assert_eq!(enforce_confidence_caps(Verdict::Supported, 1.5), 1.0);
        assert_eq!(enforce_confidence_caps(Verdict::Supported, -0.3), 0.50);
    }

    #[test]
    fn enforce_caps_supported_passthrough() {
        assert_eq!(enforce_confidence_caps(Verdict::Supported, 0.95), 0.95);
    }

    #[test]
    fn enforce_caps_supported_floor() {
        // Supported verdict floors at 0.50: the agent must be at least
        // moderately certain the source supports the claim.
        assert_eq!(enforce_confidence_caps(Verdict::Supported, 0.30), 0.50);
        assert_eq!(enforce_confidence_caps(Verdict::Supported, 0.50), 0.50);
        assert_eq!(enforce_confidence_caps(Verdict::Supported, 0.95), 0.95);
    }

    #[test]
    fn enforce_caps_unsupported_floor() {
        // Unsupported verdict floors at 0.50: the agent must be at least
        // moderately certain the source does not support the claim.
        assert_eq!(enforce_confidence_caps(Verdict::Unsupported, 0.30), 0.50);
        assert_eq!(enforce_confidence_caps(Verdict::Unsupported, 0.50), 0.50);
        assert_eq!(enforce_confidence_caps(Verdict::Unsupported, 0.85), 0.85);
    }

    #[test]
    fn enforce_caps_not_found_floor() {
        // NotFound verdict floors at 0.70: absence from the corpus is a
        // near-binary determination with a higher certainty threshold.
        assert_eq!(enforce_confidence_caps(Verdict::NotFound, 0.50), 0.70);
        assert_eq!(enforce_confidence_caps(Verdict::NotFound, 0.70), 0.70);
        assert_eq!(enforce_confidence_caps(Verdict::NotFound, 1.00), 1.00);
    }

    #[test]
    fn enforce_caps_wrong_source_floor() {
        // WrongSource verdict floors at 0.50: the agent must be at least
        // moderately certain the claim belongs to another source.
        assert_eq!(enforce_confidence_caps(Verdict::WrongSource, 0.30), 0.50);
        assert_eq!(enforce_confidence_caps(Verdict::WrongSource, 0.80), 0.80);
    }

    #[test]
    fn enforce_caps_unverifiable_floor() {
        // Unverifiable verdict floors at 0.50: the agent must be at least
        // moderately certain the claim cannot be checked.
        assert_eq!(enforce_confidence_caps(Verdict::Unverifiable, 0.20), 0.50);
        assert_eq!(enforce_confidence_caps(Verdict::Unverifiable, 0.90), 0.90);
    }

    // -- llm_config_from_request tests --

    #[test]
    fn config_defaults() {
        let req = AutoVerifyRequest {
            ollama_url: None,
            model: "qwen2.5:7b".to_string(),
            temperature: None,
            max_tokens: None,
            source_directory: None,
            fetch_sources: None,
            unpaywall_email: None,
            top_k: None,
            cross_corpus_queries: None,
            max_retry_attempts: None,
            min_score: None,
            rerank: None,
        };
        let config = llm_config_from_request(&req);
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.max_tokens, 8192);
        assert!(config.json_mode);
        assert!((config.temperature - 0.1).abs() < f32::EPSILON);
    }

    // -- Combined repair + extract pipeline test --

    #[test]
    fn repair_after_extract_from_markdown() {
        // Simulates a model that wraps truncated JSON in a code fence
        let input = "```json\n{\"verdict\": \"supported\", \"confidence\": 0.85, \"reasoning\": \"The source states";
        let extracted = extract_json_object(input);
        let repaired = repair_truncated_json(&extracted);
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
    }

    #[test]
    fn repair_realistic_truncated_verdict() {
        // A realistic LLM response truncated mid-generation at max_tokens
        let input = r#"{
  "verdict": "supported",
  "claim_original": "Studies show that reproducibility varies across disciplines",
  "claim_english": "Studies show that reproducibility varies across disciplines",
  "source_match": true,
  "reasoning": "The passage on page 3 of the cited source explicitly discusses how reproducibility rates differ across scientific fields, noting that psychology has lower rates than chemistry. This directly supports the claim that reproducibility varies across disciplines.",
  "confidence": 0.88,
  "flag": ""
"#;
        let extracted = extract_json_object(input);
        let repaired = repair_truncated_json(&extracted);
        let v: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(v["verdict"], "supported");
        assert_eq!(v["confidence"], 0.88);
        assert_eq!(v["flag"], "");
    }

    // -- ActiveAgentGuard tests --

    /// Verifies that ActiveAgentGuard inserts the job_id into the DashSet
    /// on creation and removes it when the guard is dropped.
    #[test]
    fn t_active_guard_insert_and_remove_on_drop() {
        let set = dashmap::DashSet::new();
        let job_id = "test-job-001".to_string();
        set.insert(job_id.clone());
        assert!(
            set.contains("test-job-001"),
            "job_id must be present after insert"
        );

        {
            let _guard = ActiveAgentGuard {
                set: &set,
                job_id: &job_id,
            };
            // Guard is alive, job_id is still present.
            assert!(
                set.contains("test-job-001"),
                "job_id must remain while guard is alive"
            );
        }
        // Guard dropped, job_id must be removed.
        assert!(
            !set.contains("test-job-001"),
            "job_id must be removed after guard is dropped"
        );
    }

    /// Verifies that ActiveAgentGuard only removes its own job_id and does
    /// not affect other entries in the DashSet.
    #[test]
    fn t_active_guard_does_not_remove_other_entries() {
        let set = dashmap::DashSet::new();
        set.insert("job-a".to_string());
        set.insert("job-b".to_string());

        let job_id = "job-a".to_string();
        {
            let _guard = ActiveAgentGuard {
                set: &set,
                job_id: &job_id,
            };
        }
        // Only job-a is removed by the guard; job-b remains.
        assert!(!set.contains("job-a"), "job-a must be removed by guard");
        assert!(set.contains("job-b"), "job-b must remain after guard drops");
    }

    /// Verifies that dropping the guard on an already-empty set does not panic.
    /// This covers the edge case where the entry was externally removed before
    /// the guard is dropped (e.g., manual cleanup or concurrent modification).
    #[test]
    fn t_active_guard_noop_when_entry_already_removed() {
        let set = dashmap::DashSet::new();
        let job_id = "phantom-job".to_string();
        // Do NOT insert the job_id -- guard should still drop without panic.
        {
            let _guard = ActiveAgentGuard {
                set: &set,
                job_id: &job_id,
            };
        }
        assert!(
            !set.contains("phantom-job"),
            "set must remain empty after guard drops"
        );
    }

    /// Verifies that the DashSet correctly reports containment before and
    /// after guard lifecycle. This tests the integration pattern used by the
    /// auto_verify handler: check contains() -> reject if true.
    #[test]
    fn t_active_guard_contains_check_for_duplicate_prevention() {
        let set = dashmap::DashSet::new();
        let job_id = "duplicate-check-job".to_string();

        // Before agent starts: set does not contain the job_id.
        assert!(
            !set.contains("duplicate-check-job"),
            "job_id must not be present before agent starts"
        );

        // Agent starts: inserts into set.
        set.insert(job_id.clone());
        assert!(
            set.contains("duplicate-check-job"),
            "job_id must be present after agent inserts it"
        );

        // Second auto_verify call checks contains() and rejects.
        let is_duplicate = set.contains("duplicate-check-job");
        assert!(
            is_duplicate,
            "duplicate check must detect the running agent"
        );

        // Agent finishes: guard drops and removes entry.
        {
            let _guard = ActiveAgentGuard {
                set: &set,
                job_id: &job_id,
            };
        }
        assert!(
            !set.contains("duplicate-check-job"),
            "job_id must be removed after agent finishes"
        );

        // Now a new auto_verify call is allowed.
        assert!(
            !set.contains("duplicate-check-job"),
            "subsequent auto_verify call must be allowed"
        );
    }
}
