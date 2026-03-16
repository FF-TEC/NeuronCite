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

// Domain value types for the NeuronCite application.
// These types form the shared vocabulary imported by every crate in the workspace.
// All types derive Debug, Clone, Serialize, and Deserialize to support logging,
// cloning across pipeline stages, and TOML/JSON serialization.

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Identifies which text extraction backend produced a given page's or section's
/// text. The PDF variants (`PdfExtract`, `Pdfium`, `Ocr`) handle PDF documents.
/// The HTML variants (`HtmlRaw`, `HtmlReadability`) handle web page content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtractionBackend {
    /// Pure-Rust PDF text extraction via the pdf-extract crate.
    PdfExtract,
    /// Pdfium-based extraction for multi-column layouts and complex PDFs.
    Pdfium,
    /// Tesseract OCR fallback for image-only or low-quality text pages.
    Ocr,
    /// Full visible text extracted from HTML without boilerplate removal.
    /// Includes navigation, footer, sidebar, and all other visible elements.
    HtmlRaw,
    /// Main article content extracted from HTML via a readability algorithm.
    /// Removes navigation, sidebars, footers, ads, and other boilerplate.
    HtmlReadability,
}

impl fmt::Display for ExtractionBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PdfExtract => write!(f, "pdf-extract"),
            Self::Pdfium => write!(f, "pdfium"),
            Self::Ocr => write!(f, "ocr"),
            Self::HtmlRaw => write!(f, "html-raw"),
            Self::HtmlReadability => write!(f, "html-readability"),
        }
    }
}

/// Distinguishes the original source format of an indexed document.
/// Used in the `indexed_file` table to differentiate PDF files from HTML web pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceType {
    /// A PDF document indexed from the local filesystem.
    Pdf,
    /// An HTML web page fetched from a URL and cached locally.
    Html,
}

impl fmt::Display for SourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pdf => write!(f, "pdf"),
            Self::Html => write!(f, "html"),
        }
    }
}

impl std::str::FromStr for SourceType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pdf" => Ok(Self::Pdf),
            "html" => Ok(Self::Html),
            other => Err(format!("unknown source type: {other}")),
        }
    }
}

/// Determines how embedding vectors are persisted to disk alongside the HNSW
/// index.
///
/// `SqliteBlob` stores vectors as binary BLOB columns in the chunk table.
/// All data lives in a single SQLite database file. This is the default for
/// MCP (headless) mode because it simplifies deployment: there is only one
/// file to backup or migrate, and SQLite's WAL mode provides safe concurrent
/// reads without additional coordination.
///
/// `MmapFile` stores vectors in a separate binary file alongside the SQLite
/// database. At search time, the file is memory-mapped, so the OS page cache
/// handles read-ahead without additional I/O overhead. `MmapFile` is the
/// recommended storage mode for the web UI and for large collections where
/// search latency matters: the embedding read path bypasses SQLite entirely
/// and the OS can keep hot pages in memory across sessions.
///
/// # Scalability note (#040)
///
/// Both modes currently store all sessions in a single SQLite database file.
/// At very large scale (millions of chunks across hundreds of sessions) the
/// shared connection pool becomes the concurrency bottleneck. A future
/// per-session SQLite file layout would allow independent WAL files and
/// remove the single-file serialization point, but requires a migration path
/// for existing indexes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageMode {
    /// Embeddings stored as BLOB columns in the `SQLite` chunk table.
    /// Default for MCP mode. Single-file deployment, no external dependencies.
    SqliteBlob,
    /// Embeddings stored in an external memory-mapped file beside the database.
    /// Recommended for the web UI and latency-sensitive deployments: the OS
    /// page cache keeps frequently accessed vectors in memory across requests.
    MmapFile,
}

impl fmt::Display for StorageMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SqliteBlob => write!(f, "sqlite-blob"),
            Self::MmapFile => write!(f, "mmap-file"),
        }
    }
}

/// Determines how the final embedding vector is extracted from a transformer
/// model's output tensor. Encoder models (BERT-family) use the \[CLS\] token at
/// position 0, while decoder-based models (Qwen, Mistral, LLaMA) require
/// mean pooling or last-token extraction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingStrategy {
    /// Extract the \[CLS\] token embedding at sequence position 0.
    /// Used by BERT, RoBERTa, BGE, MiniLM, and similar encoder models.
    Cls,

    /// Compute the attention-weighted mean of all token embeddings.
    /// Padding tokens (attention_mask == 0) are excluded from the average.
    /// Used by Qwen3-Embedding, GTE-Qwen2, and similar decoder models.
    MeanPooling,

    /// Extract the embedding at the last non-padding token position.
    /// Used by E5-Mistral and similar causal decoder models where the
    /// final token aggregates the full sequence context.
    LastToken,
}

impl fmt::Display for PoolingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cls => write!(f, "cls"),
            Self::MeanPooling => write!(f, "mean-pooling"),
            Self::LastToken => write!(f, "last-token"),
        }
    }
}

/// Static configuration for a single embedding model. Encapsulates all
/// model-specific parameters that differ between encoder (BERT) and decoder
/// (Qwen, Mistral) architectures, allowing the inference pipeline to handle
/// both families through the same code path. Also carries display metadata
/// (quality rating, language scope, hardware requirements) used by the GUI
/// Model Manager panel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingModelConfig {
    // -------------------------------------------------------------------
    // Inference parameters
    // -------------------------------------------------------------------
    /// HuggingFace model identifier (e.g., "BAAI/bge-small-en-v1.5").
    pub model_id: String,

    /// Human-readable display name shown in the GUI model selector.
    pub display_name: String,

    /// Dimensionality of the output embedding vectors.
    pub vector_dimension: usize,

    /// Maximum token count accepted by the model. BERT-family models use 512,
    /// while decoder-based models support 8192 or more.
    pub max_seq_len: usize,

    /// Pooling method for extracting a fixed-size vector from the variable-length
    /// transformer output sequence.
    pub pooling: PoolingStrategy,

    /// Whether the ONNX model expects a `token_type_ids` input tensor.
    /// BERT models require this (set to all zeros for single-segment input).
    /// Decoder models do not use segment IDs and omit this tensor entirely.
    pub uses_token_type_ids: bool,

    /// Whether the ONNX model expects a `position_ids` input tensor.
    /// Decoder-based models with rotary position embeddings (RoPE) require
    /// explicit position IDs as a model input. Encoder models (BERT-family)
    /// have absolute position embeddings baked into the model weights and
    /// do not need this tensor.
    pub needs_position_ids: bool,

    /// Text prefix prepended to query inputs before tokenization. Decoder-based
    /// embedding models use instruction prefixes to distinguish query embeddings
    /// from document embeddings (e.g., "Instruct: ...\nQuery: ").
    /// Empty string for models that require no prefix.
    pub query_prefix: String,

    /// Text prefix prepended to document/passage inputs before tokenization.
    /// Empty string for models that require no prefix.
    pub document_prefix: String,

    // -------------------------------------------------------------------
    // Display metadata for the GUI Model Manager panel
    // -------------------------------------------------------------------
    /// Year the model was released (e.g., 2023, 2025).
    pub release_year: u16,

    /// Qualitative retrieval quality rating (e.g., "Good", "Very high",
    /// "SOTA Open-Source"). Reflects the model's ranking on benchmarks
    /// like MTEB as of 2026.
    pub quality_rating: String,

    /// Language coverage: "EN" for English-only models, "Multilingual" for
    /// models trained on many languages including German.
    pub language_scope: String,

    /// Cross-lingual German-English retrieval capability description
    /// (e.g., "Not supported", "Very good", "SOTA").
    pub de_en_retrieval: String,

    /// CPU inference suitability (e.g., "Very good", "Slow",
    /// "Not recommended").
    pub cpu_suitability: String,

    /// GPU recommendation level (e.g., "Optional", "Recommended",
    /// "Strongly recommended", "Not required").
    pub gpu_recommendation: String,

    /// Approximate RAM requirement for loading and running the model
    /// (e.g., "1-2 GB", "16-24 GB").
    pub ram_requirement: String,

    /// Short description of the model's typical use case
    /// (e.g., "Fast EN search", "Enterprise / High Precision").
    pub typical_use_case: String,

    /// Approximate ONNX model file size in megabytes. Reflects the primary
    /// model.onnx file size from HuggingFace (FP32 for encoder models,
    /// FP16 for decoder models). Used by the GUI Model Manager to display
    /// the download/storage footprint.
    pub model_size_mb: u32,
}

// ---------------------------------------------------------------------------
// Domain structs
// ---------------------------------------------------------------------------

/// Represents one page of extracted text from a PDF file.
/// The `page_number` is 1-indexed (matching PDF page numbering conventions).
/// The backend field records which extraction method produced this text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageText {
    /// Absolute path to the source PDF file on disk.
    pub source_file: PathBuf,
    /// 1-indexed page number within the PDF document.
    pub page_number: usize,
    /// The extracted and normalized text content of this page.
    pub content: String,
    /// The extraction backend that produced this page's text.
    pub backend: ExtractionBackend,
}

/// A contiguous text segment produced by splitting the concatenated document
/// text according to a chunking strategy. Each chunk references its source
/// file, the page range it spans, its position in the chunk sequence, and
/// the document-level byte offsets of its content boundaries.
///
/// The `content_hash` is the lowercase hex-encoded SHA-256 digest of the
/// chunk's content string. It is used for content-addressed deduplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Path to the source PDF file that this chunk originates from.
    pub source_file: PathBuf,
    /// 1-indexed page number where this chunk begins.
    pub page_start: usize,
    /// 1-indexed page number where this chunk ends (inclusive).
    pub page_end: usize,
    /// 0-indexed position of this chunk in the sequence of all chunks
    /// produced from the same source file.
    pub chunk_index: usize,
    /// Document-level UTF-8 byte offset where this chunk's content starts
    /// in the concatenated document text.
    pub doc_text_offset_start: usize,
    /// Document-level UTF-8 byte offset where this chunk's content ends
    /// (exclusive) in the concatenated document text.
    pub doc_text_offset_end: usize,
    /// The actual text content of this chunk.
    pub content: String,
    /// Lowercase hex-encoded SHA-256 digest of `content`.
    pub content_hash: String,
}

impl Chunk {
    /// Computes the SHA-256 hash of the given text and returns it as a
    /// lowercase hex-encoded string. This function is the single source
    /// of truth for content hash computation across the application.
    #[must_use]
    pub fn compute_content_hash(text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let result = hasher.finalize();
        // Pre-allocate the exact capacity needed: 64 hex characters for 32 bytes
        let mut hex = String::with_capacity(64);
        for byte in &result {
            use std::fmt::Write;
            let _ = write!(hex, "{byte:02x}");
        }
        hex
    }

    /// Computes the SHA-256 hash of the raw bytes of a file on disk.
    ///
    /// The file is read in 64 KiB chunks to avoid loading the entire PDF
    /// into memory at once. This function is used by the indexing pipeline
    /// to produce a true content fingerprint for each PDF, enabling the
    /// two-stage change detection in `check_file_changed` to correctly
    /// distinguish between metadata-only changes and actual content changes.
    ///
    /// Unlike `compute_content_hash`, this function hashes the raw file
    /// bytes rather than extracted text. This ensures that renamed or
    /// touched files (different mtime/size but identical bytes) are
    /// classified as `MetadataOnly` rather than `ContentChanged`.
    ///
    /// # Errors
    ///
    /// Returns an `std::io::Error` if the file cannot be opened or read.
    pub fn compute_file_hash(path: &std::path::Path) -> Result<String, std::io::Error> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 65536]; // 64 KiB read buffer

        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }

        let result = hasher.finalize();
        let mut hex = String::with_capacity(64);
        for byte in &result {
            use std::fmt::Write;
            let _ = write!(hex, "{byte:02x}");
        }
        Ok(hex)
    }
}

/// A chunk paired with its dense vector embedding. The embedding vector
/// length equals the `vector_dimension` of the model that produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedChunk {
    /// The text chunk that was embedded.
    pub chunk: Chunk,
    /// Dense vector representation of the chunk's content. The length of
    /// this vector equals the embedding model's `vector_dimension`.
    pub embedding: Vec<f32>,
}

/// A structured citation referencing a specific passage in a source document.
/// Contains both machine-readable fields (file path, page range, byte offsets)
/// and a pre-formatted human-readable string for display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Database primary key of the `indexed_file` record. Passed directly to
    /// the `neuroncite_page` tool to retrieve the full text of any page from
    /// this document.
    pub file_id: i64,
    /// Absolute path to the source PDF file.
    pub source_file: PathBuf,
    /// Human-readable display name for the source file (typically the file
    /// name without the directory prefix).
    pub file_display_name: String,
    /// 1-indexed page number where the cited passage begins.
    pub page_start: usize,
    /// 1-indexed page number where the cited passage ends (inclusive).
    pub page_end: usize,
    /// Document-level byte offset where the cited passage starts.
    pub doc_offset_start: usize,
    /// Document-level byte offset where the cited passage ends (exclusive).
    pub doc_offset_end: usize,
    /// Pre-formatted citation string containing the file display name, page
    /// range, and a text excerpt enclosed in quotation marks.
    pub formatted: String,
}

/// A ranked search result combining vector similarity, keyword ranking, and
/// optional reranker scores into a single final score. Each result carries
/// the matched chunk's content and a structured citation.
///
/// Source file path, page_start, and page_end are accessed through the
/// `citation` field to avoid duplicating data that `Citation` already holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Final ranking score after hybrid fusion (higher is more relevant).
    pub score: f64,
    /// Cosine similarity score from the HNSW vector search.
    pub vector_score: f64,
    /// 1-indexed rank from BM25 keyword search, if the chunk appeared in
    /// FTS5 results. None if the chunk was found only by vector search.
    pub bm25_rank: Option<usize>,
    /// Score from the cross-encoder reranker, if reranking was performed.
    /// None if reranking was skipped or unavailable.
    pub reranker_score: Option<f64>,
    /// 0-indexed chunk position within the source file's chunk sequence.
    pub chunk_index: usize,
    /// The text content of the matched chunk.
    pub content: String,
    /// Structured citation for this search result. Also provides access to
    /// source_file, page_start, page_end, and doc_offset fields.
    pub citation: Citation,
}

/// Configuration parameters for a single indexing session. Defines the
/// directory to index, the embedding model, chunking strategy, and storage
/// settings. Serialized to TOML for persistence as part of session metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Root directory containing PDF files to index.
    pub directory: PathBuf,
    /// Identifier of the embedding model (e.g., "BAAI/bge-small-en-v1.5").
    pub model_name: String,
    /// Name of the chunking strategy (e.g., "token", "word", "sentence", "page").
    pub chunk_strategy: String,
    /// Target chunk size. The unit depends on the strategy: tokens for "token",
    /// words for "word". Not used by "page".
    pub chunk_size: Option<usize>,
    /// Overlap between consecutive chunks. The unit depends on the strategy:
    /// tokens for "token", words for "word". Not used by "sentence" and "page".
    pub chunk_overlap: Option<usize>,
    /// Maximum word count per sentence-based chunk. Only used by the "sentence"
    /// strategy.
    pub max_words: Option<usize>,
    /// Tesseract language code for OCR fallback (e.g., "eng", "deu").
    pub ocr_language: String,
    /// Determines how embedding vectors are stored on disk.
    pub embedding_storage_mode: StorageMode,
    /// Dimensionality of the embedding vectors produced by the model.
    pub vector_dimension: usize,
}

/// Progress tracking for an indexing operation. Updated incrementally as
/// files are processed, allowing the GUI and API to report real-time status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexProgress {
    /// Total number of PDF files discovered for indexing.
    pub files_total: usize,
    /// Number of PDF files that have been fully processed.
    pub files_done: usize,
    /// Total number of chunks created across all processed files so far.
    pub chunks_created: usize,
    /// Whether the indexing operation has finished (all files processed).
    pub complete: bool,
}

/// Describes an embedding model available in the system. Used to populate
/// model selection dropdowns in the GUI and model listing API endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique identifier for the model (e.g., "BAAI/bge-small-en-v1.5").
    pub id: String,
    /// Human-readable name for display in the GUI.
    pub display_name: String,
    /// Dimensionality of vectors produced by this model.
    pub vector_dimension: usize,
    /// Name of the backend that implements this model (e.g., "ort").
    pub backend: String,
}

/// Manifest describing a model's artifacts, checksums, and provenance.
/// Used by the model cache system to verify downloaded files and resolve
/// models for offline operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Unique identifier matching `ModelInfo::id`.
    pub model_id: String,
    /// Git revision or version tag of the model snapshot.
    pub revision: String,
    /// SHA-256 hex digest of the model weights file.
    pub sha256: String,
    /// SHA-256 hex digest of the tokenizer configuration file.
    pub tokenizer_sha256: String,
    /// URL from which the model can be downloaded.
    pub source_url: String,
    /// SPDX license identifier for the model (e.g., "MIT", "Apache-2.0").
    pub license: String,
    /// Local filesystem path to the cached model directory, if the model
    /// has been downloaded. None if the model is not cached locally.
    pub local_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Inference hardware capabilities
// ---------------------------------------------------------------------------

/// Runtime-detected hardware characteristics that influence embedding pipeline
/// tuning decisions (batch size, thread count). Constructed once during ONNX
/// session creation and propagated through the pipeline via the
/// [`EmbeddingBackend`](crate::traits::EmbeddingBackend) trait.
///
/// All tuning parameters derive from these capabilities at runtime instead of
/// compile-time `cfg!` checks, making the pipeline hardware-adaptive across
/// platforms without hardcoded values.
#[derive(Debug, Clone)]
pub struct InferenceCapabilities {
    /// Name of the active execution provider: `"CUDA"`, `"CoreML"`,
    /// `"DirectML"`, `"ROCm"`, or `"CPU"`.
    pub active_ep: String,

    /// Total physical system RAM in bytes. On unified-memory platforms (Apple
    /// Silicon), this is the shared budget for CPU and GPU. On discrete-GPU
    /// systems, this is host RAM only (VRAM is the tighter constraint and is
    /// handled by the CUDA/ROCm-specific batch size formula).
    pub system_memory_bytes: u64,

    /// Whether CPU and GPU share the same physical DRAM (Apple Silicon unified
    /// memory architecture). When `true`, embedding batch sizes are not
    /// constrained by a separate VRAM budget and can scale with available RAM.
    pub unified_memory: bool,
}

impl Default for InferenceCapabilities {
    /// Returns conservative CPU-only capabilities: 8 GB RAM, no unified memory.
    /// Used as the fallback when hardware detection is unavailable (e.g., in
    /// test stubs or non-ORT backends).
    fn default() -> Self {
        Self {
            active_ep: "CPU".into(),
            system_memory_bytes: 8 * 1024 * 1024 * 1024,
            unified_memory: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// T-CORE-001: `PageText` construction with valid fields. All fields are
    /// accessible after construction and match the input values.
    #[test]
    fn t_core_001_pagetext_construction() {
        let page = PageText {
            source_file: PathBuf::from("/docs/paper.pdf"),
            page_number: 3,
            content: "Abstract: This paper presents...".into(),
            backend: ExtractionBackend::PdfExtract,
        };

        assert_eq!(page.source_file, PathBuf::from("/docs/paper.pdf"));
        assert_eq!(page.page_number, 3);
        assert_eq!(page.content, "Abstract: This paper presents...");
        assert_eq!(page.backend, ExtractionBackend::PdfExtract);
    }

    /// T-CORE-002: Chunk `content_hash` determinism. Two chunks with identical
    /// text content produce identical SHA-256 hashes. Two chunks with different
    /// content produce different hashes.
    #[test]
    fn t_core_002_chunk_hash_determinism() {
        let text_a = "The quick brown fox jumps over the lazy dog";
        let text_b = "A different piece of text entirely";

        let hash_a1 = Chunk::compute_content_hash(text_a);
        let hash_a2 = Chunk::compute_content_hash(text_a);
        let hash_b = Chunk::compute_content_hash(text_b);

        // Same text produces the same hash
        assert_eq!(hash_a1, hash_a2);
        // Different text produces different hashes
        assert_ne!(hash_a1, hash_b);
        // Hash is 64 hex characters (256 bits / 4 bits per hex digit)
        assert_eq!(hash_a1.len(), 64);
    }

    /// T-CORE-003: `EmbeddedChunk` vector dimension equals the `vector_dimension`
    /// field of the associated `IndexConfig`.
    #[test]
    fn t_core_003_embedded_chunk_vector_dimension() {
        let config = IndexConfig {
            directory: PathBuf::from("/docs"),
            model_name: "BAAI/bge-small-en-v1.5".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::SqliteBlob,
            vector_dimension: 384,
        };

        let content = "sample chunk text";
        let embedded = EmbeddedChunk {
            chunk: Chunk {
                source_file: PathBuf::from("/docs/paper.pdf"),
                page_start: 1,
                page_end: 1,
                chunk_index: 0,
                doc_text_offset_start: 0,
                doc_text_offset_end: content.len(),
                content: content.into(),
                content_hash: Chunk::compute_content_hash(content),
            },
            embedding: vec![0.0_f32; config.vector_dimension],
        };

        assert_eq!(embedded.embedding.len(), config.vector_dimension);
    }

    /// T-CORE-004: Citation formatted string contains the file display name,
    /// page number, and a text excerpt enclosed in quotation marks.
    #[test]
    fn t_core_004_citation_formatted_string() {
        let citation = Citation {
            file_id: 1,
            source_file: PathBuf::from("/docs/statistics.pdf"),
            file_display_name: "statistics.pdf".into(),
            page_start: 5,
            page_end: 6,
            doc_offset_start: 1200,
            doc_offset_end: 1500,
            formatted: "statistics.pdf, pp. 5-6: \"regression coefficients are...\"".into(),
        };

        assert!(
            citation.formatted.contains(&citation.file_display_name),
            "formatted string must contain the file display name"
        );
        assert!(
            citation.formatted.contains("5"),
            "formatted string must contain the page number"
        );
        // Verify the formatted string contains a text excerpt in quotation marks
        assert!(
            citation.formatted.contains('"'),
            "formatted string must contain quotation marks around the excerpt"
        );
    }

    /// T-CORE-005: `SearchResult` ordering. A vector of `SearchResult` sorted
    /// by descending score produces a monotonically non-increasing sequence.
    #[test]
    fn t_core_005_search_result_ordering() {
        let make_result = |score: f64| -> SearchResult {
            SearchResult {
                score,
                vector_score: score,
                bm25_rank: None,
                reranker_score: None,
                chunk_index: 0,
                content: "test".into(),
                citation: Citation {
                    file_id: 1,
                    source_file: PathBuf::from("/docs/test.pdf"),
                    file_display_name: "test.pdf".into(),
                    page_start: 1,
                    page_end: 1,
                    doc_offset_start: 0,
                    doc_offset_end: 4,
                    formatted: "test.pdf, p. 1: \"test\"".into(),
                },
            }
        };

        let mut results = [
            make_result(0.3),
            make_result(0.9),
            make_result(0.1),
            make_result(0.7),
            make_result(0.5),
        ];

        // Sort by descending score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Verify monotonically non-increasing order
        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "scores must be non-increasing: {} >= {}",
                window[0].score,
                window[1].score
            );
        }
    }

    /// T-CORE-006: `IndexConfig` TOML roundtrip. Serializing an `IndexConfig`
    /// to TOML and deserializing it back produces a value equal to the original.
    #[test]
    fn t_core_006_index_config_toml_roundtrip() {
        let config = IndexConfig {
            directory: PathBuf::from("/home/user/papers"),
            model_name: "BAAI/bge-small-en-v1.5".into(),
            chunk_strategy: "word".into(),
            chunk_size: Some(300),
            chunk_overlap: Some(50),
            max_words: None,
            ocr_language: "eng".into(),
            embedding_storage_mode: StorageMode::MmapFile,
            vector_dimension: 384,
        };

        let toml_string =
            toml::to_string(&config).expect("IndexConfig serialization to TOML failed");
        let deserialized: IndexConfig =
            toml::from_str(&toml_string).expect("IndexConfig deserialization from TOML failed");

        assert_eq!(config, deserialized);
    }

    /// T-CORE-008: `ModelManifest` TOML roundtrip preserves all fields
    /// including the tokenizer checksum.
    #[test]
    fn t_core_008_model_manifest_toml_roundtrip() {
        let manifest = ModelManifest {
            model_id: "BAAI/bge-small-en-v1.5".into(),
            revision: "abc123def456".into(),
            sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into(),
            tokenizer_sha256: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
                .into(),
            source_url: "https://huggingface.co/BAAI/bge-small-en-v1.5".into(),
            license: "MIT".into(),
            local_path: Some(PathBuf::from("/cache/models/bge-small")),
        };

        let toml_string =
            toml::to_string(&manifest).expect("ModelManifest serialization to TOML failed");
        let deserialized: ModelManifest =
            toml::from_str(&toml_string).expect("ModelManifest deserialization from TOML failed");

        assert_eq!(manifest, deserialized);
        // Explicitly verify the tokenizer_sha256 field survived the roundtrip
        assert_eq!(manifest.tokenizer_sha256, deserialized.tokenizer_sha256);
    }

    /// T-CORE-024a: SearchResult accesses source_file, page_start, page_end
    /// exclusively through the Citation field. Verifies that the SearchResult
    /// struct does not contain its own duplicated copies of these fields,
    /// and that all values are reachable through `result.citation.*`.
    #[test]
    fn t_core_024a_search_result_uses_citation_fields() {
        let result = SearchResult {
            score: 0.85,
            vector_score: 0.80,
            bm25_rank: Some(3),
            reranker_score: None,
            chunk_index: 5,
            content: "hypothesis testing in statistics".into(),
            citation: Citation {
                file_id: 42,
                source_file: PathBuf::from("/docs/stats.pdf"),
                file_display_name: "stats.pdf".into(),
                page_start: 12,
                page_end: 14,
                doc_offset_start: 5000,
                doc_offset_end: 5500,
                formatted: "stats.pdf, pp. 12-14: \"hypothesis testing...\"".into(),
            },
        };

        // All location fields are accessible through citation.
        assert_eq!(
            result.citation.source_file,
            PathBuf::from("/docs/stats.pdf")
        );
        assert_eq!(result.citation.page_start, 12);
        assert_eq!(result.citation.page_end, 14);
        assert_eq!(result.chunk_index, 5);
        assert_eq!(result.content, "hypothesis testing in statistics");
    }

    /// T-CORE-050: `PoolingStrategy` Display trait produces stable string
    /// representations used for logging and configuration display.
    #[test]
    fn t_core_050_pooling_strategy_display() {
        assert_eq!(format!("{}", PoolingStrategy::Cls), "cls");
        assert_eq!(format!("{}", PoolingStrategy::MeanPooling), "mean-pooling");
        assert_eq!(format!("{}", PoolingStrategy::LastToken), "last-token");
    }

    /// T-CORE-051: `PoolingStrategy` equality comparison. Two identical variants
    /// compare as equal, different variants compare as unequal.
    #[test]
    fn t_core_051_pooling_strategy_equality() {
        assert_eq!(PoolingStrategy::Cls, PoolingStrategy::Cls);
        assert_eq!(PoolingStrategy::MeanPooling, PoolingStrategy::MeanPooling);
        assert_ne!(PoolingStrategy::Cls, PoolingStrategy::MeanPooling);
        assert_ne!(PoolingStrategy::LastToken, PoolingStrategy::Cls);
    }

    /// T-CORE-052: `PoolingStrategy` JSON roundtrip. Serializing to JSON and
    /// deserializing back produces the original variant.
    #[test]
    fn t_core_052_pooling_strategy_json_roundtrip() {
        for variant in &[
            PoolingStrategy::Cls,
            PoolingStrategy::MeanPooling,
            PoolingStrategy::LastToken,
        ] {
            let json = serde_json::to_string(variant).expect("PoolingStrategy JSON serialization");
            let deserialized: PoolingStrategy =
                serde_json::from_str(&json).expect("PoolingStrategy JSON deserialization");
            assert_eq!(*variant, deserialized);
        }
    }

    /// T-CORE-053: `EmbeddingModelConfig` TOML roundtrip preserves all fields
    /// including the instruction prefix strings.
    #[test]
    fn t_core_053_embedding_model_config_toml_roundtrip() {
        let config = EmbeddingModelConfig {
            model_id: "Qwen/Qwen3-Embedding-0.6B".into(),
            display_name: "Qwen3 Embedding 0.6B".into(),
            vector_dimension: 1024,
            max_seq_len: 8192,
            pooling: PoolingStrategy::MeanPooling,
            uses_token_type_ids: false,
            needs_position_ids: true,
            query_prefix: "Instruct: retrieve relevant passages\nQuery: ".into(),
            document_prefix: String::new(),
            release_year: 2025,
            quality_rating: "Very good".into(),
            language_scope: "Multilingual".into(),
            de_en_retrieval: "Good".into(),
            cpu_suitability: "Heavy".into(),
            gpu_recommendation: "Recommended".into(),
            ram_requirement: "8-12 GB".into(),
            typical_use_case: "RAG systems".into(),
            model_size_mb: 1200,
        };

        let toml_string =
            toml::to_string(&config).expect("EmbeddingModelConfig serialization to TOML failed");
        let deserialized: EmbeddingModelConfig = toml::from_str(&toml_string)
            .expect("EmbeddingModelConfig deserialization from TOML failed");

        assert_eq!(config, deserialized);
        assert_eq!(config.query_prefix, deserialized.query_prefix);
        assert_eq!(config.document_prefix, deserialized.document_prefix);
    }

    /// T-CORE-054: `EmbeddingModelConfig` JSON roundtrip preserves all fields.
    #[test]
    fn t_core_054_embedding_model_config_json_roundtrip() {
        let config = EmbeddingModelConfig {
            model_id: "BAAI/bge-small-en-v1.5".into(),
            display_name: "BGE Small EN v1.5".into(),
            vector_dimension: 384,
            max_seq_len: 512,
            pooling: PoolingStrategy::Cls,
            uses_token_type_ids: true,
            needs_position_ids: false,
            query_prefix: String::new(),
            document_prefix: String::new(),
            release_year: 2023,
            quality_rating: "Good".into(),
            language_scope: "EN".into(),
            de_en_retrieval: "Not supported".into(),
            cpu_suitability: "Very good".into(),
            gpu_recommendation: "Optional".into(),
            ram_requirement: "1-2 GB".into(),
            typical_use_case: "Fast EN search".into(),
            model_size_mb: 130,
        };

        let json = serde_json::to_string(&config).expect("EmbeddingModelConfig JSON serialization");
        let deserialized: EmbeddingModelConfig =
            serde_json::from_str(&json).expect("EmbeddingModelConfig JSON deserialization");

        assert_eq!(config, deserialized);
    }

    /// T-CORE-059b: SearchResult JSON roundtrip. Serializing to JSON and back
    /// preserves all fields, confirming that removing the duplicated fields
    /// does not break the serde contract.
    #[test]
    fn t_core_024b_search_result_json_roundtrip() {
        let result = SearchResult {
            score: 0.75,
            vector_score: 0.70,
            bm25_rank: Some(1),
            reranker_score: Some(0.82),
            chunk_index: 0,
            content: "Bayesian inference".into(),
            citation: Citation {
                file_id: 7,
                source_file: PathBuf::from("/papers/bayes.pdf"),
                file_display_name: "bayes.pdf".into(),
                page_start: 1,
                page_end: 3,
                doc_offset_start: 0,
                doc_offset_end: 200,
                formatted: "bayes.pdf, pp. 1-3: \"Bayesian inference...\"".into(),
            },
        };

        let json = serde_json::to_string(&result).expect("SearchResult JSON serialization failed");
        let deserialized: SearchResult =
            serde_json::from_str(&json).expect("SearchResult JSON deserialization failed");

        assert!((deserialized.score - 0.75).abs() < f64::EPSILON);
        assert_eq!(deserialized.citation.page_start, 1);
        assert_eq!(deserialized.citation.page_end, 3);
        assert_eq!(
            deserialized.citation.source_file,
            PathBuf::from("/papers/bayes.pdf")
        );
        assert_eq!(deserialized.chunk_index, 0);
    }

    /// T-CORE-055: `compute_file_hash` produces the correct SHA-256 digest for
    /// a known byte sequence. Two files with identical content produce the same
    /// hash; files with different content produce different hashes.
    #[test]
    fn t_core_055_compute_file_hash_determinism_and_correctness() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let content_a = b"The quick brown fox jumps over the lazy dog";
        let content_b = b"Different content that produces a different hash";

        // Write content_a to two separate temporary files to verify determinism.
        let mut file_a1 = NamedTempFile::new().expect("temp file creation must succeed");
        file_a1.write_all(content_a).expect("write must succeed");

        let mut file_a2 = NamedTempFile::new().expect("temp file creation must succeed");
        file_a2.write_all(content_a).expect("write must succeed");

        let mut file_b = NamedTempFile::new().expect("temp file creation must succeed");
        file_b.write_all(content_b).expect("write must succeed");

        let hash_a1 =
            Chunk::compute_file_hash(file_a1.path()).expect("compute_file_hash must succeed");
        let hash_a2 =
            Chunk::compute_file_hash(file_a2.path()).expect("compute_file_hash must succeed");
        let hash_b =
            Chunk::compute_file_hash(file_b.path()).expect("compute_file_hash must succeed");

        // Identical content must produce the same hash.
        assert_eq!(hash_a1, hash_a2, "same content must produce identical hash");
        // Different content must produce different hashes.
        assert_ne!(
            hash_a1, hash_b,
            "different content must produce different hash"
        );
        // SHA-256 produces a 64-character hex string.
        assert_eq!(hash_a1.len(), 64, "SHA-256 hash must be 64 hex characters");
        // The hash must differ from the text-based compute_content_hash result
        // to verify that compute_file_hash hashes raw bytes rather than a
        // path:page_count string.
        let pseudo_hash = Chunk::compute_content_hash(&format!("{}:1", file_a1.path().display()));
        assert_ne!(
            hash_a1, pseudo_hash,
            "file hash must differ from the former path:page_count pseudo-hash"
        );
    }

    /// T-CORE-056: `compute_file_hash` returns an error for a non-existent path.
    #[test]
    fn t_core_056_compute_file_hash_nonexistent_path_returns_error() {
        let result = Chunk::compute_file_hash(std::path::Path::new("/nonexistent/file.pdf"));
        assert!(
            result.is_err(),
            "compute_file_hash on a missing file must return an error"
        );
    }

    // -----------------------------------------------------------------------
    // InferenceCapabilities tests
    // -----------------------------------------------------------------------

    /// T-CORE-020: Default InferenceCapabilities returns conservative CPU values.
    #[test]
    fn t_core_020_inference_capabilities_default() {
        let caps = InferenceCapabilities::default();
        assert_eq!(caps.active_ep, "CPU", "default EP must be CPU");
        assert_eq!(
            caps.system_memory_bytes,
            8 * 1024 * 1024 * 1024,
            "default memory must be 8 GB"
        );
        assert!(
            !caps.unified_memory,
            "default must not claim unified memory"
        );
    }

    /// T-CORE-021: InferenceCapabilities Clone produces an independent copy.
    #[test]
    fn t_core_021_inference_capabilities_clone_independent() {
        let original = InferenceCapabilities {
            active_ep: "CoreML".into(),
            system_memory_bytes: 64 * 1024 * 1024 * 1024,
            unified_memory: true,
        };
        let cloned = original.clone();
        assert_eq!(cloned.active_ep, "CoreML");
        assert_eq!(cloned.system_memory_bytes, 64 * 1024 * 1024 * 1024);
        assert!(cloned.unified_memory);
    }

    /// T-CORE-022: InferenceCapabilities Debug format includes all fields.
    #[test]
    fn t_core_022_inference_capabilities_debug_format() {
        let caps = InferenceCapabilities {
            active_ep: "CUDA".into(),
            system_memory_bytes: 32 * 1024 * 1024 * 1024,
            unified_memory: false,
        };
        let debug = format!("{caps:?}");
        assert!(debug.contains("CUDA"), "Debug must contain EP name");
        assert!(
            debug.contains("system_memory_bytes"),
            "Debug must contain memory field"
        );
        assert!(
            debug.contains("unified_memory"),
            "Debug must contain unified_memory field"
        );
    }

    /// T-CORE-023: InferenceCapabilities supports all known EP names.
    #[test]
    fn t_core_023_inference_capabilities_all_eps() {
        for ep in ["CUDA", "CoreML", "DirectML", "ROCm", "CPU"] {
            let caps = InferenceCapabilities {
                active_ep: ep.to_string(),
                system_memory_bytes: 16 * 1024 * 1024 * 1024,
                unified_memory: ep == "CoreML",
            };
            assert_eq!(caps.active_ep, ep);
        }
    }

    /// T-CORE-024: InferenceCapabilities handles edge case memory values.
    #[test]
    fn t_core_024_inference_capabilities_memory_edge_cases() {
        // Zero memory (embedded systems or detection failure)
        let zero = InferenceCapabilities {
            active_ep: "CPU".into(),
            system_memory_bytes: 0,
            unified_memory: false,
        };
        assert_eq!(zero.system_memory_bytes, 0);

        // Very large memory (multi-TB workstations)
        let huge = InferenceCapabilities {
            active_ep: "CUDA".into(),
            system_memory_bytes: 2 * 1024 * 1024 * 1024 * 1024, // 2 TB
            unified_memory: false,
        };
        assert_eq!(huge.system_memory_bytes, 2 * 1024 * 1024 * 1024 * 1024);
    }
}
