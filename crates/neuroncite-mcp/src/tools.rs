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

//! Static MCP tool definitions with JSON Schema input descriptors.
//!
//! Each tool corresponds to one NeuronCite operation (search, index, sessions,
//! etc.). The definitions are returned by the `tools/list` MCP method and
//! describe the tool name, human-readable description, and input parameter
//! schema that the MCP client uses to construct valid `tools/call` requests.
//!
//! Tool schemas follow JSON Schema draft-07 conventions. Required fields are
//! listed in the `required` array; optional fields have default values
//! documented in their descriptions.

use serde_json::json;

/// Represents a single MCP tool definition with its name, description, and
/// JSON Schema for the input parameters.
pub struct ToolDefinition {
    /// Tool name used in `tools/call` requests (e.g., "neuroncite_search").
    pub name: &'static str,

    /// Human-readable description of what the tool does. Displayed to the AI
    /// assistant for tool selection decisions.
    pub description: &'static str,

    /// JSON Schema object describing the tool's input parameters.
    pub input_schema: serde_json::Value,
}

/// Returns all tool definitions exposed by the MCP server.
///
/// The returned vector contains one entry per tool. The order is stable
/// across invocations but carries no semantic meaning.
pub fn all_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "neuroncite_search",
            description: "Performs semantic and keyword hybrid search across indexed PDF documents. Returns ranked passages with citations (source file, file_id, page numbers) and relevance scores.\n\nScoring: The 'score' field is the Reciprocal Rank Fusion (RRF) score computed as sum(1/(k+rank)) across vector and BM25 rankers (k=60). 'vector_score' is cosine similarity from the HNSW approximate nearest neighbor index (range 0.0 to 1.0). 'bm25_rank' is the 1-indexed rank from FTS5 keyword search (null if the chunk was found only by vector search). 'reranker_score' is a raw cross-encoder logit (unbounded). For ms-marco-MiniLM models, typical values range approximately -10 (irrelevant) to +10 (highly relevant). These are not probabilities. Higher is always more relevant. Scores are comparable only within a single query's result set. The reranker_score replaces the RRF score for final ranking when rerank=true.\n\nRelevance interpretation: Each result includes a 'relevance' field ('high', 'medium', 'low', 'marginal', 'bm25_only') based on vector_score thresholds: >= 0.82 high, >= 0.72 medium, >= 0.60 low, < 0.60 marginal. Results found only by BM25 keyword search (no vector score) use 'bm25_only' as the relevance label and null for vector_score. The response includes 'score_summary' with min/max/mean vector_score across results that have vector scores (null when all results are bm25_only). Use 'min_score' to filter out low-relevance results.\n\nThe 'file_id' in each result can be passed directly to neuroncite_content to retrieve the full text of any page from that document. Each result also includes a 'page_numbers' array listing the page range for direct use with neuroncite_content.\n\nWorkflow: After receiving search results, use neuroncite_content with file_id and page_start/page_end to retrieve full page text for verification. Search results contain chunk excerpts; neuroncite_content provides surrounding context for accurate citation.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "Index session ID to search within (from a previous indexing operation)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (1-50, default: 10)"
                    },
                    "file_ids": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Restrict search scope to specific files. When provided, only chunks belonging to these file IDs are included in results. File IDs are obtained from neuroncite_files. Omit to search all files in the session."
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum vector_score threshold (0.0 to 1.0). Results below this cosine similarity are excluded. Recommended thresholds: 0.82 (high confidence), 0.72 (moderate), 0.60 (permissive). Default: no filtering."
                    },
                    "use_fts": {
                        "type": "boolean",
                        "description": "Enable FTS5 BM25 keyword search alongside vector search. When true, results from both vector and keyword search are merged via Reciprocal Rank Fusion (RRF). When false, only vector (semantic) search is used. (default: true)"
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Apply cross-encoder reranking using a BERT-based model that scores query-passage pairs. When true, the reranker_score replaces the RRF score for final ranking. Reranking improves precision but increases latency. Requires a reranker model to be configured -- check neuroncite_health or neuroncite_doctor for 'reranker_available' before enabling. (default: false)"
                    },
                    "refine": {
                        "type": "boolean",
                        "description": "Apply sub-chunk refinement to narrow result content to the most relevant passage within each chunk. Each result chunk is split into overlapping sub-chunks at multiple scales (controlled by refine_divisors), batch-embedded, and compared against the query. When a sub-chunk scores higher than the original chunk, the result content is replaced with that sub-chunk. This reduces noise from surrounding context. (default: true)"
                    },
                    "refine_divisors": {
                        "type": "string",
                        "description": "Comma-separated list of divisor values controlling the sub-chunk split granularity. For each divisor d, the chunk (T tokens) is split into windows of T/d tokens with 25% overlap. Multiple values enable multi-scale refinement. Example: '4,8,16' splits into 1/4, 1/8, and 1/16 of the original chunk size. Values below 2 are ignored. (default: '4,8,16')"
                    },
                    "page_start": {
                        "type": "integer",
                        "description": "Lower bound (inclusive, 1-indexed) of a page range filter. When both page_start and page_end are provided, only chunks overlapping the range [page_start, page_end] are included in results. Useful for searching within a specific section of a large document (e.g., pages 1-50 of a 711-page book). Must be used together with page_end."
                    },
                    "page_end": {
                        "type": "integer",
                        "description": "Upper bound (inclusive, 1-indexed) of the page range filter. Must be >= page_start. Must be used together with page_start."
                    }
                },
                "required": ["session_id", "query"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_batch_search",
            description: "Executes multiple search queries in a single call against the same session. Each query runs the full hybrid pipeline (vector + BM25 keyword via Reciprocal Rank Fusion, deduplication, optional cross-encoder reranking). Returns results keyed by the caller-provided query ID.\n\nScoring: 'score' is the RRF fused score computed as sum(1/(k+rank)) across vector and BM25 rankers; 'vector_score' is cosine similarity from HNSW; 'reranker_score' is a raw cross-encoder logit (unbounded). For ms-marco-MiniLM models, typical values range approximately -10 (irrelevant) to +10 (highly relevant). These are not probabilities. Scores are comparable only within a single query's result set. The reranker_score replaces the RRF score when rerank=true. The 'file_id' in each result can be passed to neuroncite_content to retrieve full page text.\n\nRelevance interpretation: Each result includes a 'relevance' field ('high', 'medium', 'low', 'marginal', 'bm25_only') based on vector_score thresholds: >= 0.82 high, >= 0.72 medium, >= 0.60 low, < 0.60 marginal. Results found only by BM25 keyword search use 'bm25_only' and null for vector_score. Each query group includes 'score_summary' with min/max/mean vector_score (null when all results are bm25_only). Use 'min_score' to filter out low-relevance results across all queries.\n\nEach result includes a 'page_numbers' array listing the page range for direct use with neuroncite_content.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "Index session ID to search within"
                    },
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Caller-provided identifier for this query, returned in the response for correlation"
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Search query text"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of results for this query (1-50, default: 5)"
                                }
                            },
                            "required": ["id", "query"]
                        },
                        "description": "Array of search queries to execute (maximum 20 per batch)"
                    },
                    "file_ids": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Restrict search scope to specific files, applied to all queries in the batch. When provided, only chunks belonging to these file IDs are included in results. File IDs are obtained from neuroncite_files. Omit to search all files in the session."
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum vector_score threshold (0.0 to 1.0), applied to all queries. Results below this cosine similarity are excluded. Recommended thresholds: 0.82 (high confidence), 0.72 (moderate), 0.60 (permissive). Default: no filtering."
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Apply cross-encoder reranking to all queries. Requires a reranker model to be configured -- check neuroncite_health for 'reranker_available'. (default: false)"
                    },
                    "use_fts": {
                        "type": "boolean",
                        "description": "Enable FTS5 keyword search alongside vector search (default: true)"
                    },
                    "refine": {
                        "type": "boolean",
                        "description": "Apply sub-chunk refinement to all query results. Splits each result chunk into overlapping sub-chunks at multiple scales, embeds them, and replaces content with the most relevant sub-section when it scores higher than the original. (default: true)"
                    },
                    "refine_divisors": {
                        "type": "string",
                        "description": "Comma-separated divisor values for sub-chunk split granularity, applied to all queries. For each divisor d, chunks are split into T/d-token windows with 25% overlap. (default: '4,8,16')"
                    }
                },
                "required": ["session_id", "queries"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_index",
            description: "Indexes documents by extracting text, chunking, computing embeddings, and storing in the vector database. Supports PDF and HTML files. Returns a job ID and session ID for tracking progress.\n\nThree input modes (mutually exclusive):\n- directory: scans for supported files (.pdf, .html, .htm). Use file_types to restrict which extensions are included.\n- urls: indexes previously fetched HTML pages from cache (requires prior neuroncite_html_fetch or neuroncite_html_crawl).\n- files: reserved for future use (indexes specific file paths by extension).\n\nAll modes create a background job. Use neuroncite_job_status to track progress.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Absolute path to a directory. Scans for all supported file types."
                    },
                    "urls": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "URLs of previously fetched HTML pages to index from cache"
                    },
                    "files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Array of absolute file paths to index (any supported type). Reserved for future use."
                    },
                    "session_id": {
                        "type": "integer",
                        "description": "Add to an existing session. If omitted, a session is created."
                    },
                    "file_types": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Restrict directory scan to these extensions (e.g., ['pdf', 'docx']). Ignored when using urls."
                    },
                    "model": {
                        "type": "string",
                        "description": "Embedding model ID. Default: BAAI/bge-small-en-v1.5"
                    },
                    "chunk_strategy": {
                        "type": "string",
                        "enum": ["page", "word", "token", "sentence"],
                        "description": "Chunking strategy. 'token' (default): subword-token windows aligned to the model. 'word': word windows with overlap. 'sentence': sentence boundaries with max word limit, recommended for academic papers. 'page': one chunk per content part."
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Tokens/words per chunk. Default: 256. Ignored by 'page'."
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "description": "Overlap between chunks. Default: 32. Used by 'token' and 'word' only."
                    },
                    "strip_boilerplate": {
                        "type": "boolean",
                        "description": "For HTML sources: readability-based boilerplate removal. Default: true"
                    }
                }
            }),
        },
        ToolDefinition {
            name: "neuroncite_sessions",
            description: "Lists all index sessions in the database with their metadata (model name, chunk strategy, directory, creation time) and aggregate statistics (file_count, total_pages, total_chunks, total_content_bytes).",
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
        ToolDefinition {
            name: "neuroncite_session_delete",
            description: "Deletes index sessions and all associated data (files, pages, chunks, embeddings). This operation is irreversible. Provide either session_id to delete a single session, or directory to delete all sessions for a specific source directory.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "Session ID to delete (mutually exclusive with 'directory')"
                    },
                    "directory": {
                        "type": "string",
                        "description": "Absolute path to the source directory whose sessions should be deleted. The path is canonicalized before comparison. (mutually exclusive with 'session_id')"
                    }
                },
                "required": []
            }),
        },
        ToolDefinition {
            name: "neuroncite_session_update",
            description: "Updates metadata on an existing index session. Currently supports setting or clearing the human-readable 'label' field for session identification.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "Session ID to update"
                    },
                    "label": {
                        "type": ["string", "null"],
                        "description": "Human-readable label for the session. Pass a string to set, or null to clear."
                    },
                    "tags": {
                        "type": ["array", "null"],
                        "items": { "type": "string" },
                        "description": "Array of tag strings for categorizing the session (e.g. [\"finance\", \"statistics\"]). Pass an array to set, or null to clear."
                    },
                    "metadata": {
                        "type": ["object", "null"],
                        "description": "Object of arbitrary key-value pairs for session metadata (e.g. {\"source\": \"manual\"}). Pass an object to set, or null to clear."
                    }
                },
                "required": ["session_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_files",
            description: "Lists indexed documents within a session with per-file statistics: chunk count (min/max/avg byte sizes), content part count, text byte counts, word estimates, source type (pdf or html), and extraction status.\n\nFor HTML files, each entry includes a web_source object with URL, title, domain, author, and language.\n\nSupports sorting by name/size/pages/chunks, single-file detail via file_id, filtering by source_type, and domain filtering for HTML files.\n\nThe file_id from the response can be used with neuroncite_content to read document content, and with neuroncite_search via the file_ids parameter to restrict search scope. File IDs are session-scoped.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "Session ID to list files from"
                    },
                    "file_id": {
                        "type": "integer",
                        "description": "Return details for a single file"
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["name", "size", "pages", "chunks"],
                        "description": "Sort order. Default: 'name' (alphabetical)"
                    },
                    "source_type": {
                        "type": "string",
                        "description": "Filter by source type (e.g., 'pdf', 'html', 'txt', 'docx')"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Filter HTML files by domain substring (case-sensitive)"
                    }
                },
                "required": ["session_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_jobs",
            description: "Lists all indexing jobs with their state (pending, running, completed, failed, canceled), progress counters, and timestamps.",
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
        ToolDefinition {
            name: "neuroncite_job_status",
            description: "Retrieves the status of a specific indexing job by its ID.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "UUID of the job to query"
                    }
                },
                "required": ["job_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_job_cancel",
            description: "Cancels a queued or running job. Only jobs in 'queued' or 'running' state can be canceled. For running jobs, cancellation is cooperative: the executor stops processing at the next checkpoint. Returns an error for jobs that have already completed, failed, or been canceled.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "UUID of the job to cancel"
                    }
                },
                "required": ["job_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_export",
            description: "Exports search results in a specified format (markdown, bibtex, csl-json, ris, or plain-text). Runs the search query and formats the results for citation use.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "Session ID to search within"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "bibtex", "csl-json", "ris", "plain-text"],
                        "description": "Export format (default: markdown)"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to export (default: 10)"
                    },
                    "deduplicate": {
                        "type": "boolean",
                        "description": "When true, BibTeX, CSL-JSON, and RIS formats produce one entry per source document with consolidated page ranges and aggregated scores. When false, each search result produces a separate entry (original behavior). Markdown and plain-text formats are unaffected. (default: true)"
                    },
                    "refine": {
                        "type": "boolean",
                        "description": "Apply sub-chunk refinement before exporting. Narrows each result chunk to the most relevant passage. (default: true)"
                    },
                    "refine_divisors": {
                        "type": "string",
                        "description": "Comma-separated divisor values for sub-chunk refinement granularity. (default: '4,8,16')"
                    }
                },
                "required": ["session_id", "query"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_models",
            description: "Lists all available embedding models with their configuration (vector dimension, max sequence length, pooling strategy) and local cache status.",
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
        ToolDefinition {
            name: "neuroncite_doctor",
            description: "Checks system capabilities: GPU hardware detection, CUDA availability, compiled embedding backends, Tesseract OCR, pdfium library. Reports what features are available on the current machine.",
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
        ToolDefinition {
            name: "neuroncite_health",
            description: "Returns server health status including API version, active backend, GPU availability, and build features.",
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        },
        ToolDefinition {
            name: "neuroncite_annotate",
            description: "Highlights text passages in PDFs and adds comment annotations. Takes CSV or JSON input with title, author, and quote fields to identify PDFs and text passages. Searches for quoted text using a 5-stage pipeline (exact, normalized, fuzzy, fallback extraction, OCR), creates highlight annotations with configurable colors, and optionally adds popup comments. Returns a job ID for progress tracking via neuroncite_job_status and neuroncite_annotate_status. Only one annotation job can run at a time; submitting a second job while one is active returns an error.\n\nDry-run mode: When dry_run is true, validates input data and checks which PDFs exist in the source directory without creating a job or writing files. Returns a per-row preview with match status.\n\nAppend mode: When append is true, reads PDFs from the output_directory (which must already exist from a previous annotation run) instead of copying from source_directory. This allows layering annotations incrementally without re-copying source files.\n\nPDF-to-title matching uses Jaro-Winkler similarity on filenames: score = 0.60 * title_sim + 0.40 * author_sim, threshold >= 0.80. The scan is non-recursive (direct children of source_directory only). Input strings are normalized: lowercase, strip non-alphanumeric/non-whitespace, collapse whitespace.\n\nAppend mode idempotency: when append=true, duplicate detection uses 90% bounding-box area overlap to skip highlights that already exist at the same position. Re-annotating the same quote in append mode is safe.\n\nSide effect: pdfium re-linearizes PDFs when saving. Multi-pass annotation may cause structural page count mismatches (pdf_page_count != page_count in neuroncite_files) without affecting annotation correctness.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "source_directory": {
                        "type": "string",
                        "description": "Absolute path to the directory containing PDF files to annotate"
                    },
                    "output_directory": {
                        "type": "string",
                        "description": "Absolute path where annotated PDFs and the report will be saved"
                    },
                    "input_data": {
                        "type": "string",
                        "description": "CSV or JSON string with annotation instructions. Required columns/fields: title, author, quote. Optional: color (#RRGGBB hex, default #FFFF00), comment (popup text). Multiple rows can reference the same PDF. Format is auto-detected."
                    },
                    "default_color": {
                        "type": "string",
                        "description": "Default highlight color in hex (#RRGGBB) for rows without color. Default: #FFFF00 (yellow)"
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "When true, validates input data and checks which PDFs exist in the source directory without creating a job or modifying files. Returns a per-row preview showing which quotes have matching PDFs. (default: false)"
                    },
                    "append": {
                        "type": "boolean",
                        "description": "When true, reads PDFs from output_directory (from a previous annotation run) instead of source_directory. The output_directory must already exist. This allows adding annotations incrementally without re-copying source files. (default: false)"
                    }
                },
                "required": ["source_directory", "output_directory", "input_data"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_content",
            description: "Retrieves text content from an indexed document by part number. A 'part' is the logical content unit of the source format: a page in PDFs or a heading-based section in HTML.\n\nTwo retrieval modes: provide 'part' for a single unit, or 'start'/'end' for a contiguous range (max 20 parts per request). Part numbers are 1-indexed.\n\nThe response includes the extraction_backend field indicating how the text was obtained (pdfium, ocr, html_readability). For HTML sources, the response includes a web_source object with URL, title, and domain.\n\nWorkflow: Run neuroncite_search to find passages, then use neuroncite_content with the file_id and part range from each result to read surrounding context and verify quotes. Search results contain chunk excerpts; neuroncite_content provides the full content unit.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "integer",
                        "description": "Database file ID (from neuroncite_files or search results)"
                    },
                    "part": {
                        "type": "integer",
                        "description": "1-based part number for single-unit retrieval. Mutually exclusive with start/end."
                    },
                    "start": {
                        "type": "integer",
                        "description": "First part of a range (1-based, inclusive). Use with 'end'."
                    },
                    "end": {
                        "type": "integer",
                        "description": "Last part of a range (1-based, inclusive). Max 20 parts. Use with 'start'."
                    }
                },
                "required": ["file_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_citation_create",
            description: "Creates a citation verification job from LaTeX (.tex) and BibTeX (.bib) files. Parses all citation commands (\\cite, \\citep, \\citet, \\autocite, \\parencite, \\textcite), resolves cite-keys against BibTeX entries, matches cited works to indexed PDFs via filename similarity, and groups citations into batches for sub-agent processing. Returns job_id, citation counts, and PDF match statistics.\n\nDry-run mode: When dry_run is true, the pipeline runs without creating a job or inserting rows. Returns a match preview with per-cite-key data (author, title, year, matched_file_id, matched_filename, overlap_score) so incorrect matches can be corrected via file_overrides before the real run.\n\nFile overrides: A map of cite_key -> file_id that overrides the automatic PDF matching. Applied after the automatic matching step. Use the dry_run response to identify incorrect matches, then pass corrections in file_overrides.\n\nSub-agent orchestration: After creating the job, spawn parallel sub-agents to process batches. Each sub-agent calls neuroncite_citation_claim to acquire a batch, performs verification using neuroncite_search / neuroncite_content / neuroncite_batch_content, and submits results via neuroncite_citation_submit. Sub-agents require MCP tool access and must use subagent_type=\"general-purpose\" when launched via the Task tool. Other agent types (research-analyst, data-analyst, etc.) lack MCP tool access and will fail. Recommended: 3-5 parallel agents, each processing batches in a loop until neuroncite_citation_claim returns null batch_id.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "tex_path": {
                        "type": "string",
                        "description": "Absolute path to the .tex file containing citation commands"
                    },
                    "bib_path": {
                        "type": "string",
                        "description": "Absolute path to the .bib file with bibliographic entries"
                    },
                    "session_id": {
                        "type": "integer",
                        "description": "NeuronCite index session ID to search against for PDF matching and verification"
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Target number of citations per batch for sub-agent processing (default: 5). Citations in the same \\cite{a,b,c} group are never split across batches."
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "When true, runs the parse/match pipeline without creating a job or inserting rows. Returns a match preview so incorrect PDF matches can be identified and corrected via file_overrides. (default: false)"
                    },
                    "file_overrides": {
                        "type": "object",
                        "description": "Map of cite_key -> file_id to override automatic PDF matching. Applied after the automatic matching step. Use the dry_run response to identify incorrect matches, then pass corrections here.",
                        "additionalProperties": { "type": "integer" }
                    }
                },
                "required": ["tex_path", "bib_path", "session_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_citation_claim",
            description: "Claims a batch of citations for sub-agent verification. Returns all citation rows in the batch with their BibTeX metadata (author, title, year, abstract, keywords), matched PDF file_id, LaTeX context (line number, anchor words, section title), co-citation metadata (is_co_citation, co_cited_with in the co_citation_json field), and job context (tex_path, bib_path, session_id). Co-cited sources (from \\cite{a,b,c} commands) must be evaluated independently. Stale batches (claimed > 5 minutes ago) are automatically reclaimed. Returns null batch_id when all batches are claimed or done.\n\nUnderstanding the document and locating the claim (mandatory):\n1. High-level document scan: Before processing individual citations, read the beginning and section headings of the .tex file at tex_path to understand what the document is about — its topic, field, and central argument. This context is required to correctly interpret each individual claim.\n2. Reading the citation context: Each row provides tex_path and tex_line. The precomputed columns anchor_before, anchor_after, and tex_context serve only to locate the citation position in the file — they are not sufficient for claim extraction on their own. You must read the original .tex file around tex_line and extract a minimum of 500 characters before and after the citation command. Reading more context is encouraged whenever the claim is ambiguous, spans multiple sentences, or contains equations.\n3. Verbatim claim extraction: Extract the exact claim as stated in the LaTeX source. Do not paraphrase or infer intent from the precomputed columns alone.\n\nTargeted claiming: When batch_id is specified, claims that specific batch instead of the next FIFO batch. This allows retrying failed batches and out-of-order processing. Failed batches are reset to pending before claiming.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Citation verification job ID returned by neuroncite_citation_create"
                    },
                    "batch_id": {
                        "type": "integer",
                        "description": "Specific batch to claim. When absent, the lowest pending batch is claimed in FIFO order. Specifying a batch_id allows retrying failed batches and out-of-order processing."
                    }
                },
                "required": ["job_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_citation_submit",
            description: "Submits verification results for a batch of citation rows. All fields are mandatory. Fields that do not apply use their type's zero value: empty string for strings, empty array for arrays, false for booleans, 0 for numbers.\n\nVerification procedure (mandatory for every row):\n1. Search the matched PDF using neuroncite_search with at least two distinct queries: one for the core claim text, one targeting specific numbers, formulas, or named definitions present in the claim. Use neuroncite_content to read the full page for each promising result.\n2. Regardless of the primary verdict, perform at least one cross-corpus search to identify the strongest corroborating or alternative source. Report the best result found in other_source_list.\n\nConfidence calibration (meta-confidence):\nConfidence measures how certain the agent is that the chosen verdict is correct, not how strongly the source supports the claim.\n\nFor 'supported' (>= 0.50): 0.95-1.00 verbatim match, 0.85-0.94 explicit statement, 0.70-0.84 logically derived, 0.50-0.69 weak support requiring interpretation.\nFor 'unsupported' (>= 0.50): 0.85-1.00 clearly no support, 0.50-0.84 moderately certain.\nFor 'not_found' (>= 0.70): 1.00 deterministic absence from corpus, 0.70-0.99 exhaustive search found nothing.\nFor 'wrong_source' (>= 0.50): 0.85-1.00 clearly different source, 0.50-0.84 substantially stronger elsewhere.\nFor 'unverifiable' (>= 0.50): 0.85-1.00 inherently uncheckable, 0.50-0.84 might be checkable with more sources.\nFor 'partial' (<= 0.80): 0.60-0.80 part supported, below 0.60 marginal relevance.\nFor 'peripheral_match' (<= 0.69): 0.50-0.69 only in non-substantive sections.\n\nCalibration constraints:\n- 'supported' verdict requires confidence >= 0.50.\n- 'partial' verdict requires confidence <= 0.80.\n- 'peripheral_match' verdict requires confidence <= 0.69 — a match found only in non-substantive sections (ToC, bibliography, foreword, appendix, glossary) cannot exceed the thematic-overlap band.\n- 'not_found' verdict requires confidence >= 0.70.\n- 'unsupported' verdict requires confidence >= 0.50.\n- 'wrong_source' verdict requires confidence >= 0.50.\n- 'unverifiable' verdict requires confidence >= 0.50.\n- For 'supported': when a specific number, formula, date, or statistic in the claim is not found verbatim in the source, reduce confidence by 0.20.\n- Thematic overlap alone does not justify a 'supported' verdict above confidence 0.69.\n- Co-cited sources (is_co_citation=true in the claimed batch) must be evaluated independently.\n\nValidation rules:\n- 'wrong_source', 'partial', and 'peripheral_match' verdicts require a non-empty other_source_list.\n- flag must be empty string, \"critical\", or \"warning\".\n\nRows must be in 'claimed' status. When all rows are done, the job is automatically completed.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Citation verification job ID"
                    },
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "row_id": { "type": "integer", "description": "citation_row.id from the claimed batch" },
                                "verdict": {
                                    "type": "string",
                                    "enum": ["supported", "partial", "unsupported", "not_found", "wrong_source", "unverifiable", "peripheral_match"],
                                    "description": "supported — the claim is explicitly stated in the cited source, no inference required. partial — the source supports part of the claim but another part is absent or deviates. unsupported — the PDF was found and read; relevant passages exist but none support the specific claim. not_found — no file_id was assigned to this cite-key; the source is not present in the indexed corpus and cannot be checked. wrong_source — the claim is verifiable but the evidence is in a different source in the corpus, not the one cited. unverifiable — the claim cannot be checked by nature: future projections, subjective statements, or data not available in any indexed source. peripheral_match — the claim text appears in the cited source exclusively in non-substantive sections (table of contents, bibliography, foreword, appendix, glossary); no supporting passage exists in the document body where the claim would carry scientific weight. Requires confidence <= 0.69 and a non-empty other_source_list."
                                },
                                "claim_original": { "type": "string", "description": "The claim as stated in the LaTeX document" },
                                "claim_english": { "type": "string", "description": "English formulation of the core claim" },
                                "source_match": { "type": "boolean", "description": "Whether the claim was found in the cited PDF" },
                                "other_source_list": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "cite_key_or_title": { "type": "string" },
                                            "passages": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "text": { "type": "string" },
                                                        "page": { "type": "integer" },
                                                        "score": { "type": "number" }
                                                    },
                                                    "required": ["text", "page", "score"]
                                                }
                                            }
                                        },
                                        "required": ["cite_key_or_title", "passages"]
                                    },
                                    "description": "Factual finding from the cross-corpus search: which other sources in the indexed corpus contain evidence relevant to this claim. Always populate with the best result found regardless of verdict — for 'supported' report the strongest corroborating source, for 'wrong_source' and 'partial' this field must be non-empty. Empty array only when no relevant result was found across all search rounds. This field records what was found, not what should be cited."
                                },
                                "passages": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "file_id": { "type": "integer" },
                                            "page": { "type": "integer" },
                                            "passage_text": { "type": "string" },
                                            "relevance_score": { "type": "number" },
                                            "passage_location": {
                                                "type": "string",
                                                "enum": [
                                                    "abstract", "foreword", "table_of_contents",
                                                    "introduction", "literature_review", "theoretical_framework",
                                                    "methodology", "results", "discussion", "conclusion",
                                                    "bibliography", "appendix", "glossary",
                                                    "table_or_figure", "footnote",
                                                    "body_text"
                                                ],
                                                "description": "Structural location within the PDF where the passage was found. Classifies the passage by its position in the standard academic document structure: front matter (abstract, foreword, table_of_contents), body sections (introduction through conclusion), back matter (bibliography, appendix, glossary), structural elements (table_or_figure, footnote), or body_text as fallback when the specific section cannot be determined."
                                            },
                                            "source_chunk_id": {
                                                "type": "integer",
                                                "description": "Optional chunk ID from the indexed session that this passage is based on. When the agent finds supporting text via neuroncite_search, the search result contains a chunk with file_id, page_start, page_end, and verbatim content. Providing the chunk ID enables more deterministic annotation matching by allowing the pipeline to retrieve the exact indexed text."
                                            }
                                        },
                                        "required": ["file_id", "page", "passage_text", "relevance_score", "passage_location"]
                                    }
                                },
                                "reasoning": { "type": "string", "description": "Justification for the verdict" },
                                "confidence": { "type": "number", "description": "Meta-confidence: how certain the agent is that the chosen verdict is correct (0.0 to 1.0). This is NOT support-strength. A high confidence on 'unsupported' means the agent is very sure the source does not support the claim. Calibrated per verdict-specific rules in the tool description." },
                                "search_rounds": { "type": "integer", "description": "Number of search queries executed. Minimum 2 per citation: one targeting the core claim text, one targeting specific numbers, formulas, or named definitions present in the claim. Report the actual total count including the cross-corpus search." },
                                "flag": { "type": "string", "description": "Escalation flag. critical — the cited source directly contradicts the claim, or a specific number, formula, or date in the claim is demonstrably wrong based on the source. warning — confidence is below 0.70, or the publication year of the cited work postdates the claimed event, or only thematic overlap was found. \"\" — clear finding in either direction, no escalation needed." },
                                "better_source": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "Editorial recommendation: cite-keys or titles from the .bib file that the author should cite instead of or in addition to the current citation for this specific claim. Only populate when a clearly more accurate source exists in the .bib file. Empty array when the current citation is appropriate or no better match is present."
                                },
                                "latex_correction": {
                                    "type": "object",
                                    "properties": {
                                        "correction_type": { "type": "string", "enum": ["rephrase", "add_context", "replace_citation", "none"] },
                                        "original_text": { "type": "string" },
                                        "suggested_text": { "type": "string" },
                                        "explanation": { "type": "string" }
                                    },
                                    "required": ["correction_type", "original_text", "suggested_text", "explanation"],
                                    "description": "LaTeX correction suggestion. correction_type values: rephrase — the claim in the LaTeX is imprecise or overstated relative to what the source actually says, the wording needs adjustment; add_context — the citation is correct but the surrounding text creates a misleading impression that a qualifying sentence would resolve; replace_citation — the cited work is wrong and should be replaced with the source identified in better_source; none — no correction needed, use empty strings for all other fields."
                                }
                            },
                            "required": ["row_id", "verdict", "claim_original", "claim_english", "source_match", "other_source_list", "passages", "reasoning", "confidence", "search_rounds", "flag", "better_source", "latex_correction"]
                        },
                        "description": "Array of verification results, one per row_id in the claimed batch"
                    }
                },
                "required": ["job_id", "results"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_citation_status",
            description: "Returns the status of a citation verification job including: row counts by status (pending, claimed, done, failed), batch counts, verdict distribution (supported, partial, unsupported, not_found, wrong_source, unverifiable), flagged alerts with critical/warning escalations, elapsed time, and completion status.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Citation verification job ID"
                    }
                },
                "required": ["job_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_citation_rows",
            description: "Returns citation rows for a job with optional status filtering and pagination. Provides direct read-only access to citation rows including their result_json field, without the file-writing overhead of the export endpoint. Useful for programmatic access to verification results, progress monitoring, and filtering rows by status.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Citation verification job ID"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "claimed", "done", "failed"],
                        "description": "Filter rows by status. When absent, returns all rows regardless of status."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of rows to skip for pagination (default: 0)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return (default: 100, max: 500)"
                    }
                },
                "required": ["job_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_citation_export",
            description: "Exports citation verification results as six files: (1) annotation_pipeline_input.csv for the neuroncite_annotate pipeline with verdict-based color coding, (2) citation_data.csv with all 38 columns sorted by tex_line (includes passage_location per passage slot), (3) citation_data.xlsx formatted Excel workbook with verdict-colored rows and badge cells, (4) corrections.json with suggested LaTeX corrections sorted by line number, (5) citation_report.json with statistics and alerts, (6) citation_full_detail.json with complete row data. Triggers the annotation pipeline to create highlighted PDFs from the annotation CSV.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Citation verification job ID"
                    },
                    "output_directory": {
                        "type": "string",
                        "description": "Absolute path where output files (CSV, corrections, report) and annotated PDFs are saved"
                    },
                    "source_directory": {
                        "type": "string",
                        "description": "Absolute path to the directory containing the source PDF files for annotation"
                    }
                },
                "required": ["job_id", "output_directory", "source_directory"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_chunks",
            description: "Browse chunks of a file directly, without search. Returns chunk content, page range, word count, and byte count with pagination support. Useful for inspecting how a document was split into chunks and verifying chunk quality.\n\nFile IDs are session-scoped: the same physical PDF has different file_id values in different sessions. The file must belong to the specified session_id.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "Session ID containing the file"
                    },
                    "file_id": {
                        "type": "integer",
                        "description": "File ID to browse chunks for"
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "Filter chunks that span this page number (1-indexed). When provided, only chunks whose page range includes this page are returned."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of chunks to skip for pagination (default: 0)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of chunks to return (default: 20, max: 100)"
                    }
                },
                "required": ["session_id", "file_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_quality_report",
            description: "Text extraction quality overview for all files in a session. Reports extraction method distribution (native text vs OCR), page count mismatches between extracted and structural counts, empty pages, and per-file quality flags (incomplete_extraction, page_count_mismatch, ocr_heavy, low_text_density, many_empty_pages). Useful for identifying problematic files that may need re-indexing with different extraction settings.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "Session ID to generate quality report for"
                    }
                },
                "required": ["session_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_file_compare",
            description: "Compares the same file across different index sessions. Finds all indexed instances of a file by path or name pattern, and reports per-session statistics (pages extracted, chunks produced, chunk strategy, average chunk size). Useful for evaluating which chunking configuration produces the best coverage for a specific document.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Exact file path to search for across sessions"
                    },
                    "file_name_pattern": {
                        "type": "string",
                        "description": "File name pattern with SQL LIKE wildcards (% for any, _ for single char). Searches the file_path column. Example: '%Fama%French%'"
                    }
                },
                "required": []
            }),
        },
        ToolDefinition {
            name: "neuroncite_discover",
            description: "Scans a directory for supported files and reports what is already indexed. Lists all files by type (pdf, html), queries existing sessions, and identifies unindexed files. Reports per-type file counts and sizes, per-session coverage statistics, and a list of unindexed files with their types. This is the recommended first call before indexing a directory.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Absolute path to the directory to discover"
                    }
                },
                "required": ["directory"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_inspect_annotations",
            description: "Inspects all annotations in a PDF file and returns their properties: annotation type, fill color (hex #RRGGBB), opacity (alpha 0-255), and bounding rectangle. Use this tool after neuroncite_annotate to verify that highlight colors were physically written to the annotated PDF.\n\nReturns per-annotation details and summary statistics: total highlight count and a deduplicated list of unique hex colors found across all highlight annotations. Non-highlight annotations (text notes, links, widgets) are included in the annotation list but excluded from the highlight-specific summary.\n\nThis tool is read-only and does not modify the PDF. It does not require an indexed session or database access.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": "Absolute path to the PDF file to inspect"
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "Restrict inspection to a single page (1-indexed). When absent, all pages are scanned."
                    }
                },
                "required": ["pdf_path"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_preview_chunks",
            description: "Previews how a file will be split into chunks by a given strategy, without writing to the database. Extracts text, applies the chunking strategy, and returns the first N chunks with content, part ranges, and byte/word counts.\n\nSupports PDF and HTML files. For HTML: provide either file_path (cache file) or url; the response includes section structure. For PDF: provide file_path to the PDF on disk.\n\nUseful for evaluating chunking parameters before committing to a full index run.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to preview chunking for"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL of a cached HTML page (alternative to file_path for HTML)"
                    },
                    "chunk_strategy": {
                        "type": "string",
                        "enum": ["page", "word", "token", "sentence"],
                        "description": "Chunking strategy to apply"
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Tokens/words per chunk. Default: 256"
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "description": "Overlap between chunks. Default: 32"
                    },
                    "max_words": {
                        "type": "integer",
                        "description": "Max words per chunk for 'sentence' strategy"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max chunks to return in preview. Default: 5, max: 50"
                    },
                    "strip_boilerplate": {
                        "type": "boolean",
                        "description": "For HTML: readability-based boilerplate removal. Default: true"
                    }
                },
                "required": ["chunk_strategy"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_index_add",
            description: "Incrementally adds or updates specific files in an existing index session. Supports PDF and HTML cache files. Files already present are checked for changes (mtime/size + SHA-256); unchanged files are skipped. Changed and new files are extracted, chunked, embedded, and stored. The session's HNSW index is rebuilt once at the end.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "The session to add files to"
                    },
                    "files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Array of absolute file paths. Each file is checked against the session's existing records and only processed if new or changed."
                    }
                },
                "required": ["session_id", "files"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_multi_search",
            description: "Searches across multiple index sessions simultaneously. The query is embedded once, then the search pipeline runs for each session. Results from all sessions are merged into a single ranked list sorted by score, with each result tagged by its source session_id. Useful for comparing chunking strategies or searching a corpus indexed across multiple sessions.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_ids": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Array of session IDs to search across (2-10 sessions required)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Total number of results to return after merging (default: 10, max: 50)"
                    },
                    "use_fts": {
                        "type": "boolean",
                        "description": "Enable FTS5 BM25 keyword search alongside vector search (default: true)"
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum vector_score threshold (0.0 to 1.0). Results below this cosine similarity are excluded."
                    }
                },
                "required": ["session_ids", "query"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_reranker_load",
            description: "Loads a cross-encoder reranker model at runtime and activates it for search reranking. If the model is not in the local cache, it is downloaded automatically from HuggingFace before loading. After loading, the 'rerank' parameter in neuroncite_search and neuroncite_batch_search starts working. The reranker is hot-swapped into the GPU worker without restarting the server. Check neuroncite_health for 'reranker_available' status after loading.\n\nCross-encoder scores are raw logits, not normalized probabilities. For ms-marco-MiniLM models, typical values range approximately -10 (irrelevant) to +10 (highly relevant). The exact range depends on the model. Scores are comparable only within a single query's result set.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Hugging Face model identifier for the cross-encoder model (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2'). If the model is absent from the local cache, it is downloaded automatically. The response field 'downloaded' indicates whether a download was performed."
                    },
                    "backend": {
                        "type": "string",
                        "description": "Backend to use for reranker inference (default: 'ort'). Must match a compiled-in backend feature."
                    }
                },
                "required": ["model_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_batch_content",
            description: "Retrieves content parts from multiple indexed documents in a single call. Each request specifies a file_id and either a single 'part' number or a 'start'/'end' range. Results are returned per-request with partial failure handling: individual errors are reported inline without failing the entire batch.\n\nLimits: max 10 requests, max 20 total parts across all requests. Content exceeding 100 KB per part is truncated at a UTF-8 boundary.\n\nFor HTML sources, each result includes a web_source object with URL and title. Use this to verify passages from multiple documents after a neuroncite_search call.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "requests": {
                        "type": "array",
                        "description": "Array of content requests. Max 10 requests, max 20 total parts.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_id": {
                                    "type": "integer",
                                    "description": "Database file ID"
                                },
                                "part": {
                                    "type": "integer",
                                    "description": "Single part number (1-indexed). Mutually exclusive with start/end."
                                },
                                "start": {
                                    "type": "integer",
                                    "description": "First part of a range (1-based). Use with 'end'."
                                },
                                "end": {
                                    "type": "integer",
                                    "description": "Last part of a range (1-based). Use with 'start'."
                                }
                            },
                            "required": ["file_id"]
                        }
                    }
                },
                "required": ["requests"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_compare_search",
            description: "Runs the same query against two different index sessions and returns side-by-side results for comparing search quality. The query embedding is computed once and reused for both sessions.\n\nComparison requires both sessions to use the same embedding model (same vector dimension). An error is returned if the dimensions differ or if session_id_a equals session_id_b.\n\nUseful for evaluating different chunking strategies (e.g., 'sentence' vs 'word') or indexing configurations on the same document corpus. Complements neuroncite_file_compare which reports chunk statistics; this tool compares actual retrieval quality.\n\nThe response contains per-session result lists with scores, total candidate counts, and average vector scores for direct quality comparison.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id_a": {
                        "type": "integer",
                        "description": "First session to search. Must use the same embedding model as session_id_b."
                    },
                    "session_id_b": {
                        "type": "integer",
                        "description": "Second session to search. Must differ from session_id_a and use the same embedding model."
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query text. The same query is run against both sessions."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results per session (1-20, default: 5)"
                    }
                },
                "required": ["session_id_a", "session_id_b", "query"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_reindex_file",
            description: "Re-indexes a single file within an existing session. The old file record and associated content is deleted, then the file is re-extracted with the current extractor, re-chunked, re-embedded, and the HNSW index is rebuilt. Supports PDF and HTML files.\n\nWhen the file_path was not previously indexed in the session, it is added as a new file. The response 'action' field reports 'reindexed' for previously known files or 'added' for new ones.\n\nThe session's vector dimension must match the currently loaded embedding model.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "The session containing the file to re-index."
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file on disk. The file must exist at this path."
                    }
                },
                "required": ["session_id", "file_path"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_text_search",
            description: "Searches the indexed page text of a specific PDF file for literal substring occurrences. Operates on stored database content without re-reading the PDF file from disk. Useful for verifying that specific text is present in an indexed document and locating which pages contain it.\n\nThe search performs exact substring matching on the text content stored during indexing. Case-insensitive search uses Unicode lowercase folding on both the query and the stored content.\n\nReturns matching page numbers with surrounding context (~100 characters on each side of the match). When a page contains multiple occurrences, each is reported as a separate match entry.\n\nThe file_id can be obtained from neuroncite_files. This tool requires the file to be indexed in a session (the text is read from the database, not from the PDF file on disk).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "integer",
                        "description": "The indexed_file.id to search within (obtained from neuroncite_files)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Literal substring to search for in the indexed page text"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "When true, performs byte-exact case-sensitive matching. When false (default), uses Unicode lowercase folding for comparison."
                    }
                },
                "required": ["file_id", "query"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_session_diff",
            description: "Compares two index sessions to report which files differ between \
                them. Useful for detecting which files changed between indexing runs, comparing \
                sessions with different chunk strategies, or verifying incremental index_add \
                operations.\n\n\
                Comparison key is file_path. Returns four categories:\n\
                - only_in_a: files in session A but not session B\n\
                - only_in_b: files in session B but not session A\n\
                - identical: files in both sessions with matching hash, page count, and size\n\
                - changed: files in both sessions where at least one attribute differs\n\n\
                For changed files, hash_changed, page_count_changed, and size_changed flags \
                indicate which attributes differ. The file hashes are SHA-256 of the raw file bytes.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "session_a": {
                        "type": "integer",
                        "description": "First session ID"
                    },
                    "session_b": {
                        "type": "integer",
                        "description": "Second session ID"
                    }
                },
                "required": ["session_a", "session_b"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_citation_retry",
            description: "Resets specific citation rows or failed batches back to 'pending' status so they can be re-verified by sub-agents. Supports three targeting modes that can be combined in a single call:\n\n- row_ids: Reset specific citation rows by their integer ID (rows in 'done' or 'failed' status). Use after inspecting rows with neuroncite_citation_rows.\n- flags: Reset all 'done' rows carrying a specific flag value ('warning', 'critical'). Targets rows where the verification result was flagged for review.\n- batch_ids: Reset all 'failed' rows in specific batches (the original behavior).\n\nWhen no targeting parameter is provided, resets all failed rows (optionally filtered by batch_ids). If the job was completed, its state is transitioned back to 'running' so sub-agents can claim the reset rows.\n\nReturns the count of rows_reset and batches_reset. Rows that are already 'pending' or 'claimed' are not affected.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Citation verification job ID returned by neuroncite_citation_create"
                    },
                    "row_ids": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Array of citation row IDs to reset. Targets specific rows in 'done' or 'failed' status."
                    },
                    "flags": {
                        "type": "array",
                        "items": { "type": "string", "enum": ["warning", "critical"] },
                        "description": "Array of flag values. Resets all 'done' rows carrying any of the specified flags."
                    },
                    "batch_ids": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Array of batch IDs. Resets all 'failed' rows in the specified batches."
                    }
                },
                "required": ["job_id"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_citation_fetch_sources",
            description: "Fetches source documents referenced in a BibTeX file and adds them to an index session. Reads the .bib file, extracts URL and DOI fields from each entry, resolves DOIs through a multi-source resolution chain, downloads PDFs and HTML pages, and indexes all acquired files into the specified session.\n\nSkip behavior: Before downloading, each entry is checked against existing files in the output_directory. Sources that already exist (matched by filename) are skipped with status 'skipped' and reported in pdfs_skipped/html_skipped counters. This makes the tool safe to call repeatedly without re-downloading.\n\nHTML saving: HTML pages are saved as .html files in the output_directory alongside PDFs (using the same naming convention: 'Title (Author, Year).html'). They are also cached internally and indexed into the session.\n\nDOI resolution chain: For entries with a DOI field but no explicit URL, the tool queries free academic APIs in sequence to find direct open-access PDF URLs: (1) Unpaywall (requires email parameter), (2) Semantic Scholar, (3) OpenAlex, (4) doi.org fallback. The chain short-circuits on the first API that returns a PDF URL. Each result includes a `doi_resolved_via` field indicating which API provided the URL.\n\nURL precedence: The explicit `url` field takes precedence over DOI resolution. Entries without either field are skipped.\n\nPDF detection: A URL is classified as PDF when the path ends in `.pdf` or the server responds with `application/pdf` Content-Type. All other URLs are treated as HTML pages.\n\nWorkflow: Call this tool after neuroncite_citation_create to populate the corpus with cited source documents. The tool downloads PDFs directly and fetches HTML pages, then indexes both into the session so they become searchable via neuroncite_search.\n\nRate limiting: Requests are spaced by delay_ms (default 1000ms) to avoid overwhelming target servers. Some publishers may block automated downloads (HTTP 403), which are reported as per-source failures without aborting the batch.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "bib_path": {
                        "type": "string",
                        "description": "Absolute path to the .bib file containing bibliographic entries with URL/DOI fields"
                    },
                    "session_id": {
                        "type": "integer",
                        "description": "Index session ID to add downloaded sources to"
                    },
                    "output_directory": {
                        "type": "string",
                        "description": "Absolute path where downloaded PDF and HTML files are stored. Directory is created if it does not exist. Existing files in this directory are checked before downloading -- sources that already exist are skipped."
                    },
                    "delay_ms": {
                        "type": "integer",
                        "description": "Delay in milliseconds between consecutive HTTP requests (default: 1000). Higher values reduce the risk of rate limiting by publishers."
                    },
                    "email": {
                        "type": "string",
                        "description": "Email address for Unpaywall API access. When provided, the DOI resolution chain starts with Unpaywall (direct open-access PDF URLs for ~30M papers). When absent, Unpaywall is skipped and resolution starts with Semantic Scholar. The email is used only as a polite-access identifier -- no account or API key is needed."
                    }
                },
                "required": ["bib_path", "session_id", "output_directory"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_bib_report",
            description: "Generates CSV and XLSX report files from a BibTeX file. Parses the .bib file, checks file existence in the output directory, and writes a formatted report listing all entries with cite_key, title, author, year, link type (URL/DOI/none), and download status (exists/missing/no_link). Existing report files are overwritten.\n\nThe CSV file includes a UTF-8 BOM for correct encoding in Excel. The XLSX file has a formatted header row with frozen pane, status-colored rows (cyan tint for exists, pink tint for missing, gray for no_link), and a Summary sheet with aggregate statistics.\n\nOutput files: bib_report.csv and bib_report.xlsx in the output_directory.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "bib_path": {
                        "type": "string",
                        "description": "Absolute path to the .bib file"
                    },
                    "output_directory": {
                        "type": "string",
                        "description": "Absolute path where report files (bib_report.csv, bib_report.xlsx) are written. Also the directory where existing source files are checked for the status column."
                    }
                },
                "required": ["bib_path", "output_directory"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_annotation_remove",
            description: "Removes highlight annotations from a PDF file based on a filter mode. Operates using lopdf (pure Rust) without pdfium. Three removal modes are supported:\n\n- 'all': Removes all highlight annotations from the PDF.\n- 'by_color': Removes highlights matching specific hex colors. Comparison is case-insensitive.\n- 'by_page': Removes highlights on specific 1-indexed page numbers.\n\nWhen output_path is omitted, the source PDF is overwritten in-place. When dry_run is true, reports what would be removed without modifying any file.\n\nThe response includes annotations_removed (count), pages_affected (count), and appearance_objects_cleaned (count of orphaned AP stream objects removed from the PDF).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": "Absolute path to the PDF file to process"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["all", "by_color", "by_page"],
                        "description": "Removal mode: 'all' removes all highlights, 'by_color' removes highlights matching specific colors, 'by_page' removes highlights on specific pages"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Absolute path where the modified PDF is saved. When omitted, the source file at pdf_path is overwritten."
                    },
                    "colors": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Array of hex color strings (e.g., ['#FF0000', '#FFFF00']). Required when mode is 'by_color'. Comparison is case-insensitive."
                    },
                    "pages": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Array of 1-indexed page numbers. Required when mode is 'by_page'."
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "When true, report what would be removed without modifying any file. (default: false)"
                    }
                },
                "required": ["pdf_path", "mode"]
            }),
        },
        ToolDefinition {
            name: "neuroncite_annotate_status",
            description: "Returns per-quote progress information for a running or completed annotation job. Provides paginated access to individual quote processing statuses that track whether each quote was matched, not found, or encountered an error during the annotation pipeline.\n\nComplements neuroncite_job_status which reports aggregate progress (done/total). This handler exposes fine-grained per-quote details including the match method used (exact, normalized, fuzzy, ocr) and the page number where each quote was located.\n\nThe status_counts object provides aggregate counts: pending (not yet processed), matched (text located in PDF), not_found (text not found), error (processing failed).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Annotation job ID returned by neuroncite_annotate"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of quote rows to return (default: 100, max: 500)"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of rows to skip for pagination (default: 0)"
                    }
                },
                "required": ["job_id"]
            }),
        },
        // -----------------------------------------------------------------
        // HTML web scraping and indexing tools
        // -----------------------------------------------------------------
        ToolDefinition {
            name: "neuroncite_html_fetch",
            description: "Fetches one or more web pages via HTTP GET, caches the raw HTML to disk, and extracts metadata (title, author, Open Graph tags, language, canonical URL, etc.). Returns the metadata and cache paths for each URL. Use this before neuroncite_index with the urls parameter to index web content. Supports rate limiting via delay_ms when fetching multiple URLs. Each fetched page is deduplicated on disk using SHA-256 hash of the URL.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of URLs to fetch. When both 'url' and 'urls' are provided, the single URL is merged into the array (deduplicated)."
                    },
                    "url": {
                        "type": "string",
                        "description": "Single URL to fetch. Can be combined with 'urls' (merged into the array). Provide url, urls, or both."
                    },
                    "strip_boilerplate": {
                        "type": "boolean",
                        "description": "Apply readability algorithm to remove navigation, ads, sidebars from extracted text. Default: true"
                    },
                    "delay_ms": {
                        "type": "integer",
                        "description": "Delay between HTTP requests in milliseconds when fetching multiple URLs. Default: 500"
                    }
                }
            }),
        },
        ToolDefinition {
            name: "neuroncite_html_crawl",
            description: "Crawls a website starting from a URL. Discovers pages via BFS link-following with configurable depth, same-domain filtering, and URL pattern matching. Alternatively uses sitemap.xml for URL discovery when use_sitemap is true. Returns metadata and cache paths for all fetched pages. Use neuroncite_index with the urls parameter afterward to index the crawled pages into a searchable session.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "start_url": {
                        "type": "string",
                        "description": "Starting URL for the crawl"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum link-following depth. 0 = only start URL, 1 = start + directly linked pages. Default: 1"
                    },
                    "same_domain_only": {
                        "type": "boolean",
                        "description": "Only follow links to the same domain as start_url. Default: true"
                    },
                    "url_pattern": {
                        "type": "string",
                        "description": "Regex pattern to filter URLs (only matching URLs are followed/fetched)"
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum total pages to fetch. Default: 50"
                    },
                    "delay_ms": {
                        "type": "integer",
                        "description": "Delay between HTTP requests in milliseconds. Default: 500"
                    },
                    "use_sitemap": {
                        "type": "boolean",
                        "description": "Fetch sitemap.xml and use it for URL discovery instead of link-following. Default: false"
                    },
                    "strip_boilerplate": {
                        "type": "boolean",
                        "description": "Apply readability algorithm to remove boilerplate content. Default: true"
                    }
                },
                "required": ["start_url"]
            }),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-MCP-011: all_tools returns a non-empty list of tool definitions.
    #[test]
    fn t_mcp_011_all_tools_returns_definitions() {
        let tools = all_tools();
        assert!(!tools.is_empty(), "tool list must not be empty");
    }

    /// T-MCP-012: Every tool definition has a non-empty name and description.
    #[test]
    fn t_mcp_012_tool_names_and_descriptions_non_empty() {
        for tool in all_tools() {
            assert!(!tool.name.is_empty(), "tool name must not be empty");
            assert!(
                !tool.description.is_empty(),
                "tool description must not be empty for {}",
                tool.name
            );
        }
    }

    /// T-MCP-013: Every tool name starts with the "neuroncite_" prefix.
    #[test]
    fn t_mcp_013_tool_names_have_prefix() {
        for tool in all_tools() {
            assert!(
                tool.name.starts_with("neuroncite_"),
                "tool name '{}' must start with 'neuroncite_'",
                tool.name
            );
        }
    }

    /// T-MCP-014: Every tool's input_schema is a valid JSON object with a "type"
    /// field set to "object".
    #[test]
    fn t_mcp_014_input_schemas_are_objects() {
        for tool in all_tools() {
            assert_eq!(
                tool.input_schema["type"], "object",
                "input_schema for '{}' must have type: object",
                tool.name
            );
        }
    }

    /// T-MCP-015: Tool names are unique (no duplicates in the list).
    #[test]
    fn t_mcp_015_tool_names_are_unique() {
        let tools = all_tools();
        let mut names: Vec<&str> = tools.iter().map(|t| t.name).collect();
        let len_before = names.len();
        names.sort();
        names.dedup();
        assert_eq!(
            names.len(),
            len_before,
            "tool names must be unique (found duplicates)"
        );
    }

    /// T-MCP-117: The `neuroncite_batch_search` tool definition exists and
    /// declares `session_id` and `queries` as required parameters.
    #[test]
    fn t_mcp_117_batch_search_tool_definition() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_batch_search")
            .expect("neuroncite_batch_search tool must exist");

        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");
        assert!(
            required.iter().any(|v| v == "session_id"),
            "session_id must be required"
        );
        assert!(
            required.iter().any(|v| v == "queries"),
            "queries must be required"
        );

        // The queries property must be typed as an array.
        let queries_prop = &tool.input_schema["properties"]["queries"];
        assert_eq!(
            queries_prop["type"], "array",
            "queries must be an array type"
        );
    }

    /// T-MCP-118: The `neuroncite_files` tool definition exists and declares
    /// `session_id` as the sole required parameter.
    #[test]
    fn t_mcp_118_files_tool_definition() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_files")
            .expect("neuroncite_files tool must exist");

        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");
        assert_eq!(required.len(), 1, "exactly one required parameter");
        assert_eq!(required[0], "session_id");
    }

    /// T-MCP-119: The `neuroncite_session_update` tool definition exists and
    /// declares `session_id` as required, with optional `label`, `tags`, and
    /// `metadata` parameters.
    #[test]
    fn t_mcp_119_session_update_tool_definition() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_session_update")
            .expect("neuroncite_session_update tool must exist");

        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");
        assert!(
            required.iter().any(|v| v == "session_id"),
            "session_id must be required"
        );

        // label must be defined in properties but not required.
        let has_label = tool.input_schema["properties"]["label"].is_object()
            || tool.input_schema["properties"]["label"].is_array();
        assert!(has_label, "label property must be defined");
        assert!(
            !required.iter().any(|v| v == "label"),
            "label must not be required"
        );

        // tags must be defined in properties but not required.
        let tags_prop = &tool.input_schema["properties"]["tags"];
        assert!(
            tags_prop.is_object(),
            "tags property must be defined in session_update schema"
        );
        assert!(
            !required.iter().any(|v| v == "tags"),
            "tags must not be required"
        );

        // metadata must be defined in properties but not required.
        let meta_prop = &tool.input_schema["properties"]["metadata"];
        assert!(
            meta_prop.is_object(),
            "metadata property must be defined in session_update schema"
        );
        assert!(
            !required.iter().any(|v| v == "metadata"),
            "metadata must not be required"
        );
    }

    /// T-MCP-120: The `neuroncite_search` tool description mentions `file_id`
    /// so AI agents know the field is available in search results.
    #[test]
    fn t_mcp_120_search_description_mentions_file_id() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_search")
            .expect("neuroncite_search tool must exist");

        assert!(
            tool.description.contains("file_id"),
            "neuroncite_search description must mention file_id"
        );
    }

    /// T-MCP-033: The tool catalog contains 42 tools. The original 46 tools
    /// were reduced to 41 by removing 5 HTML-specific duplicates, then
    /// increased to 43 by adding neuroncite_bib_report.
    #[test]
    fn t_mcp_033_tool_count() {
        let tools = all_tools();
        assert_eq!(tools.len(), 43, "tool catalog must contain 43 tools");
    }

    /// T-MCP-034: Every tool with required parameters has those parameters
    /// defined in the properties object.
    #[test]
    fn t_mcp_034_required_params_exist_in_properties() {
        for tool in all_tools() {
            if let Some(required) = tool.input_schema["required"].as_array() {
                let properties = &tool.input_schema["properties"];
                for param in required {
                    let param_name = param.as_str().unwrap_or("");
                    assert!(
                        properties[param_name].is_object(),
                        "tool '{}': required parameter '{}' must be defined in properties",
                        tool.name,
                        param_name
                    );
                }
            }
        }
    }

    /// T-MCP-100: `neuroncite_citation_rows` tool exists with job_id as the
    /// sole required parameter, and optional status/offset/limit properties.
    #[test]
    fn t_mcp_100_citation_rows_has_job_id_required() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_citation_rows")
            .expect("neuroncite_citation_rows tool must exist");

        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");
        assert_eq!(required.len(), 1, "exactly one required parameter");
        assert_eq!(required[0], "job_id");

        // Optional parameters must be defined in properties.
        let props = &tool.input_schema["properties"];
        assert!(props["status"].is_object(), "status property must exist");
        assert!(props["offset"].is_object(), "offset property must exist");
        assert!(props["limit"].is_object(), "limit property must exist");
    }

    /// T-MCP-101: `neuroncite_citation_claim` tool includes optional batch_id
    /// in its properties while keeping only job_id as required.
    #[test]
    fn t_mcp_101_citation_claim_has_optional_batch_id() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_citation_claim")
            .expect("neuroncite_citation_claim tool must exist");

        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");
        assert!(
            required.iter().any(|v| v == "job_id"),
            "job_id must be required"
        );
        assert!(
            !required.iter().any(|v| v == "batch_id"),
            "batch_id must not be required"
        );

        // batch_id must be defined as an optional integer property.
        let batch_id_prop = &tool.input_schema["properties"]["batch_id"];
        assert!(
            batch_id_prop.is_object(),
            "batch_id property must be defined"
        );
        assert_eq!(
            batch_id_prop["type"], "integer",
            "batch_id must be an integer"
        );
    }

    /// T-MCP-121: The `neuroncite_inspect_annotations` tool definition exists
    /// with `pdf_path` as the sole required parameter and an optional
    /// `page_number` property.
    #[test]
    fn t_mcp_121_inspect_annotations_tool_definition() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_inspect_annotations")
            .expect("neuroncite_inspect_annotations tool must exist");

        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");
        assert_eq!(required.len(), 1, "exactly one required parameter");
        assert_eq!(required[0], "pdf_path");

        // page_number must be defined as an optional integer property.
        let page_prop = &tool.input_schema["properties"]["page_number"];
        assert!(
            page_prop.is_object(),
            "page_number property must be defined"
        );
        assert_eq!(
            page_prop["type"], "integer",
            "page_number must be an integer"
        );

        // page_number must not be in the required array.
        assert!(
            !required.iter().any(|v| v == "page_number"),
            "page_number must not be required"
        );
    }

    /// T-MCP-112: `neuroncite_citation_create` tool includes optional dry_run
    /// and file_overrides in its properties.
    #[test]
    fn t_mcp_112_citation_create_has_dry_run_and_file_overrides() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_citation_create")
            .expect("neuroncite_citation_create tool must exist");

        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");

        // dry_run and file_overrides must not be required.
        assert!(
            !required.iter().any(|v| v == "dry_run"),
            "dry_run must not be required"
        );
        assert!(
            !required.iter().any(|v| v == "file_overrides"),
            "file_overrides must not be required"
        );

        // Both must be defined in properties.
        let props = &tool.input_schema["properties"];
        assert!(
            props["dry_run"].is_object(),
            "dry_run property must be defined"
        );
        assert_eq!(
            props["dry_run"]["type"], "boolean",
            "dry_run must be boolean"
        );
        assert!(
            props["file_overrides"].is_object(),
            "file_overrides property must be defined"
        );
        assert_eq!(
            props["file_overrides"]["type"], "object",
            "file_overrides must be an object"
        );
    }

    /// T-MCP-113: `neuroncite_session_delete` has no required parameters.
    /// The tool accepts either `session_id` or `directory`, both optional at
    /// the schema level. Validation of "at least one" is handled by the
    /// handler at runtime, not by the JSON Schema.
    #[test]
    fn t_mcp_113_session_delete_no_required_params() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_session_delete")
            .expect("neuroncite_session_delete tool must exist");

        let required = tool.input_schema.get("required");
        let is_empty = match required {
            None => true,
            Some(arr) => arr.as_array().is_none_or(|a| a.is_empty()),
        };

        assert!(
            is_empty,
            "neuroncite_session_delete must have no required parameters"
        );

        // Both session_id and directory must be defined in properties.
        let props = &tool.input_schema["properties"];
        assert!(
            props["session_id"].is_object(),
            "session_id property must be defined"
        );
        assert!(
            props["directory"].is_object(),
            "directory property must be defined"
        );
    }

    /// T-MCP-072: `neuroncite_index` tool declares a `chunk_strategy` enum
    /// property with the four valid strategy values: page, word, token, sentence.
    #[test]
    fn t_mcp_072_index_has_chunk_strategy_enum() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_index")
            .expect("neuroncite_index tool must exist");

        let props = &tool.input_schema["properties"];
        let strategy = &props["chunk_strategy"];
        assert!(
            strategy.is_object(),
            "chunk_strategy property must be defined"
        );

        let enum_values = strategy["enum"]
            .as_array()
            .expect("chunk_strategy must have an enum array");
        let values: Vec<&str> = enum_values.iter().filter_map(|v| v.as_str()).collect();

        assert!(
            values.contains(&"page"),
            "chunk_strategy enum must include 'page'"
        );
        assert!(
            values.contains(&"word"),
            "chunk_strategy enum must include 'word'"
        );
        assert!(
            values.contains(&"token"),
            "chunk_strategy enum must include 'token'"
        );
        assert!(
            values.contains(&"sentence"),
            "chunk_strategy enum must include 'sentence'"
        );
    }

    /// T-MCP-073: `neuroncite_search` declares a `min_score` property with
    /// type number. This parameter was added for Issue 8 (search scoring
    /// transparency) to allow callers to filter low-relevance results.
    #[test]
    fn t_mcp_073_search_has_min_score_property() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_search")
            .expect("neuroncite_search tool must exist");

        let props = &tool.input_schema["properties"];
        let min_score = &props["min_score"];
        assert!(min_score.is_object(), "min_score property must be defined");
        assert_eq!(
            min_score["type"], "number",
            "min_score must be of type number"
        );

        // min_score must not be required.
        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");
        assert!(
            !required.iter().any(|v| v == "min_score"),
            "min_score must not be required"
        );
    }

    /// T-MCP-074: `neuroncite_session_update` declares tags with type
    /// array-or-null and metadata with type object-or-null. Verifies the
    /// exact JSON Schema types added for Issue 21 (session metadata).
    #[test]
    fn t_mcp_074_session_update_tags_and_metadata_types() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_session_update")
            .expect("neuroncite_session_update tool must exist");

        let props = &tool.input_schema["properties"];

        // tags: type should be ["array", "null"] to allow clearing.
        let tags_type = &props["tags"]["type"];
        let tags_types: Vec<&str> = tags_type
            .as_array()
            .expect("tags type must be an array")
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(
            tags_types.contains(&"array") && tags_types.contains(&"null"),
            "tags type must include 'array' and 'null', got: {tags_types:?}"
        );

        // tags items must be string.
        assert_eq!(
            props["tags"]["items"]["type"], "string",
            "tags items must be strings"
        );

        // metadata: type should be ["object", "null"].
        let meta_type = &props["metadata"]["type"];
        let meta_types: Vec<&str> = meta_type
            .as_array()
            .expect("metadata type must be an array")
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(
            meta_types.contains(&"object") && meta_types.contains(&"null"),
            "metadata type must include 'object' and 'null', got: {meta_types:?}"
        );
    }

    /// T-MCP-102: `neuroncite_annotate` tool declares optional `dry_run` and
    /// `append` boolean properties. These parameters are not in the required
    /// array, and the description mentions both modes.
    #[test]
    fn t_mcp_102_annotate_has_dry_run_and_append() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_annotate")
            .expect("neuroncite_annotate tool must exist");

        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");

        // dry_run and append must not be required.
        assert!(
            !required.iter().any(|v| v == "dry_run"),
            "dry_run must not be required"
        );
        assert!(
            !required.iter().any(|v| v == "append"),
            "append must not be required"
        );

        // Both must be defined as boolean properties.
        let props = &tool.input_schema["properties"];
        assert!(
            props["dry_run"].is_object(),
            "dry_run property must be defined"
        );
        assert_eq!(
            props["dry_run"]["type"], "boolean",
            "dry_run must be boolean"
        );
        assert!(
            props["append"].is_object(),
            "append property must be defined"
        );
        assert_eq!(props["append"]["type"], "boolean", "append must be boolean");

        // The tool description must mention both modes.
        assert!(
            tool.description.contains("dry_run") || tool.description.contains("Dry-run"),
            "description must mention dry_run mode"
        );
        assert!(
            tool.description.contains("append") || tool.description.contains("Append"),
            "description must mention append mode"
        );
    }

    // -----------------------------------------------------------------------
    // Tool definition tests for the 4 production tools
    // -----------------------------------------------------------------------

    /// T-MCP-103: `neuroncite_preview_chunks` tool definition exists with
    /// `chunk_strategy` as the sole required parameter. file_path is optional
    /// because HTML preview uses url or file_id instead. The chunk_strategy
    /// property must have a 4-value enum.
    #[test]
    fn t_mcp_103_preview_chunks_tool_definition() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_preview_chunks")
            .expect("neuroncite_preview_chunks tool must exist");

        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");
        assert!(
            required.iter().any(|v| v == "chunk_strategy"),
            "chunk_strategy must be required"
        );

        // file_path is no longer required (HTML preview uses url or file_id).
        assert!(
            !required.iter().any(|v| v == "file_path"),
            "file_path must not be required (HTML uses url/file_id)"
        );

        // chunk_strategy must have a 4-value enum.
        let strategy = &tool.input_schema["properties"]["chunk_strategy"];
        let enum_values = strategy["enum"]
            .as_array()
            .expect("chunk_strategy must have an enum array");
        assert_eq!(
            enum_values.len(),
            4,
            "chunk_strategy enum must have 4 values"
        );

        // limit must be optional.
        assert!(
            !required.iter().any(|v| v == "limit"),
            "limit must not be required"
        );

        // All properties must be defined (including url and strip_boilerplate
        // absorbed from html_preview).
        let props = &tool.input_schema["properties"];
        assert!(props["file_path"].is_object(), "file_path must be defined");
        assert!(props["url"].is_object(), "url must be defined");
        assert!(
            props["chunk_size"].is_object(),
            "chunk_size must be defined"
        );
        assert!(
            props["chunk_overlap"].is_object(),
            "chunk_overlap must be defined"
        );
        assert!(props["limit"].is_object(), "limit must be defined");
        assert!(
            props["strip_boilerplate"].is_object(),
            "strip_boilerplate must be defined"
        );
    }

    /// T-MCP-104: `neuroncite_index_add` tool definition exists with
    /// `session_id` and `files` as required parameters. The `files` property
    /// must be typed as an array of strings.
    #[test]
    fn t_mcp_104_index_add_tool_definition() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_index_add")
            .expect("neuroncite_index_add tool must exist");

        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");
        assert!(
            required.iter().any(|v| v == "session_id"),
            "session_id must be required"
        );
        assert!(
            required.iter().any(|v| v == "files"),
            "files must be required"
        );

        // files property must be an array of strings.
        let files_prop = &tool.input_schema["properties"]["files"];
        assert_eq!(files_prop["type"], "array", "files must be an array");
        assert_eq!(
            files_prop["items"]["type"], "string",
            "files items must be strings"
        );
    }

    /// T-MCP-078: `neuroncite_multi_search` tool definition exists with
    /// `session_ids` and `query` as required parameters. The `session_ids`
    /// property must be typed as an array of integers.
    #[test]
    fn t_mcp_078_multi_search_tool_definition() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_multi_search")
            .expect("neuroncite_multi_search tool must exist");

        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");
        assert!(
            required.iter().any(|v| v == "session_ids"),
            "session_ids must be required"
        );
        assert!(
            required.iter().any(|v| v == "query"),
            "query must be required"
        );

        // session_ids property must be an array of integers.
        let sessions_prop = &tool.input_schema["properties"]["session_ids"];
        assert_eq!(
            sessions_prop["type"], "array",
            "session_ids must be an array"
        );
        assert_eq!(
            sessions_prop["items"]["type"], "integer",
            "session_ids items must be integers"
        );

        // Optional properties must be defined.
        let props = &tool.input_schema["properties"];
        assert!(props["top_k"].is_object(), "top_k must be defined");
        assert!(props["use_fts"].is_object(), "use_fts must be defined");
        assert!(props["min_score"].is_object(), "min_score must be defined");

        // top_k, use_fts, min_score must not be required.
        assert!(
            !required.iter().any(|v| v == "top_k"),
            "top_k must not be required"
        );
        assert!(
            !required.iter().any(|v| v == "use_fts"),
            "use_fts must not be required"
        );
        assert!(
            !required.iter().any(|v| v == "min_score"),
            "min_score must not be required"
        );
    }

    /// T-MCP-079: `neuroncite_reranker_load` tool definition exists with
    /// `model_id` as the sole required parameter. The `backend` parameter
    /// must be optional.
    #[test]
    fn t_mcp_079_reranker_load_tool_definition() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_reranker_load")
            .expect("neuroncite_reranker_load tool must exist");

        let required = tool.input_schema["required"]
            .as_array()
            .expect("required must be an array");
        assert_eq!(required.len(), 1, "exactly one required parameter");
        assert_eq!(required[0], "model_id");

        // backend must be optional.
        let props = &tool.input_schema["properties"];
        assert!(
            props["backend"].is_object(),
            "backend property must be defined"
        );
        assert!(
            !required.iter().any(|v| v == "backend"),
            "backend must not be required"
        );

        // model_id must be a string.
        assert_eq!(
            props["model_id"]["type"], "string",
            "model_id must be a string"
        );
    }

    /// T-MCP-105: The `neuroncite_multi_search` description mentions
    /// `session_id` to inform AI agents that each result is tagged with
    /// its source session.
    #[test]
    fn t_mcp_105_multi_search_description_mentions_session_id() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_multi_search")
            .expect("neuroncite_multi_search tool must exist");

        assert!(
            tool.description.contains("session_id"),
            "neuroncite_multi_search description must mention session_id"
        );
    }

    /// T-MCP-106: The `neuroncite_reranker_load` description mentions
    /// `reranker_available` to inform AI agents to check health after loading.
    #[test]
    fn t_mcp_106_reranker_load_description_mentions_reranker_available() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_reranker_load")
            .expect("neuroncite_reranker_load tool must exist");

        assert!(
            tool.description.contains("reranker_available"),
            "neuroncite_reranker_load description must mention reranker_available"
        );
    }

    /// T-MCP-107: The `neuroncite_index_add` description mentions change
    /// detection to inform AI agents about the skip-unchanged behavior.
    #[test]
    fn t_mcp_107_index_add_description_mentions_change_detection() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_index_add")
            .expect("neuroncite_index_add tool must exist");

        assert!(
            tool.description.contains("unchanged") || tool.description.contains("change"),
            "neuroncite_index_add description must mention change detection"
        );
    }

    /// T-MCP-108: The `neuroncite_preview_chunks` description mentions
    /// "without" database writes to clarify this is a stateless preview.
    #[test]
    fn t_mcp_108_preview_chunks_description_mentions_no_db_writes() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_preview_chunks")
            .expect("neuroncite_preview_chunks tool must exist");

        assert!(
            tool.description.contains("without"),
            "neuroncite_preview_chunks description must clarify no database writes"
        );
    }

    /// T-MCP-085: The `neuroncite_reranker_load` tool description and the
    /// model_id property description both mention auto-download. This pins the
    /// contract that the tool no longer requires pre-cached model files.
    #[test]
    fn t_mcp_085_reranker_load_describes_auto_download() {
        let tools = all_tools();
        let tool = tools
            .iter()
            .find(|t| t.name == "neuroncite_reranker_load")
            .expect("neuroncite_reranker_load tool must exist");

        // The top-level description must mention automatic download.
        assert!(
            tool.description.contains("downloaded automatically")
                || tool.description.contains("download"),
            "neuroncite_reranker_load description must mention auto-download, \
             got: {}",
            tool.description
        );

        // The model_id property description must inform callers about the
        // 'downloaded' response field.
        let model_id_desc = tool.input_schema["properties"]["model_id"]["description"]
            .as_str()
            .expect("model_id description must be a string");
        assert!(
            model_id_desc.contains("downloaded"),
            "model_id description must reference the 'downloaded' response field, \
             got: {model_id_desc}"
        );
    }

    /// T-MCP-084: All 42 tool names exist in the catalog.
    #[test]
    fn t_mcp_084_all_tools_present() {
        let tools = all_tools();
        let expected = [
            "neuroncite_search",
            "neuroncite_batch_search",
            "neuroncite_multi_search",
            "neuroncite_compare_search",
            "neuroncite_text_search",
            "neuroncite_content",
            "neuroncite_batch_content",
            "neuroncite_index",
            "neuroncite_index_add",
            "neuroncite_reindex_file",
            "neuroncite_preview_chunks",
            "neuroncite_discover",
            "neuroncite_sessions",
            "neuroncite_session_delete",
            "neuroncite_session_update",
            "neuroncite_session_diff",
            "neuroncite_files",
            "neuroncite_chunks",
            "neuroncite_quality_report",
            "neuroncite_file_compare",
            "neuroncite_jobs",
            "neuroncite_job_status",
            "neuroncite_job_cancel",
            "neuroncite_export",
            "neuroncite_models",
            "neuroncite_doctor",
            "neuroncite_health",
            "neuroncite_reranker_load",
            "neuroncite_annotate",
            "neuroncite_annotate_status",
            "neuroncite_inspect_annotations",
            "neuroncite_annotation_remove",
            "neuroncite_html_fetch",
            "neuroncite_html_crawl",
            "neuroncite_citation_create",
            "neuroncite_citation_claim",
            "neuroncite_citation_submit",
            "neuroncite_citation_status",
            "neuroncite_citation_rows",
            "neuroncite_citation_export",
            "neuroncite_citation_retry",
            "neuroncite_citation_fetch_sources",
            "neuroncite_bib_report",
        ];

        assert_eq!(expected.len(), 43, "expected list must have 43 entries");
        for name in &expected {
            assert!(
                tools.iter().any(|t| t.name == *name),
                "tool '{}' must exist in the catalog",
                name
            );
        }
    }
}
