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

// Data types for the citation verification pipeline.
//
// Defines the structures used across all pipeline stages: LaTeX parsing,
// BibTeX resolution, batch assignment, sub-agent claiming/submission, status
// reporting, and result export. These types serve as the data contract between
// modules and between the backend and MCP/REST transport layers.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Verdict and correction types
// ---------------------------------------------------------------------------

/// The sub-agent's assessment of whether the cited source supports the claim
/// made in the LaTeX document. Six verdict types ordered from full support
/// to unverifiable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The cited source contains a passage that supports the claim.
    Supported,
    /// The source partially supports the claim (nuance missing or overstated).
    Partial,
    /// The source exists but does not support the claim as stated.
    Unsupported,
    /// No relevant passage found in any indexed PDF.
    NotFound,
    /// The claim is correct but attributed to the wrong source.
    WrongSource,
    /// The cited source is not in the indexed corpus and cannot be checked.
    Unverifiable,
    /// The claim text was found in the cited source, but exclusively in
    /// non-substantive sections (table of contents, bibliography, index,
    /// foreword, appendix, glossary). No supporting passage exists in the
    /// body of the work where the claim would carry scientific weight.
    PeripheralMatch,
}

impl Verdict {
    /// Returns the hex color code used for annotation highlights corresponding
    /// to this verdict. Color scheme: green = supported, gold = partial,
    /// red = unsupported, gray = not_found, orange = wrong_source,
    /// light gray = unverifiable, amber = peripheral_match.
    pub fn color_hex(&self) -> &'static str {
        match self {
            Self::Supported => "#00AA00",
            Self::Partial => "#FFD700",
            Self::Unsupported => "#FF4444",
            Self::NotFound => "#AAAAAA",
            Self::WrongSource => "#FF8800",
            Self::Unverifiable => "#CCCCCC",
            Self::PeripheralMatch => "#FFC107",
        }
    }

    /// Returns the light background color for Excel row formatting.
    /// These are softer tones for row backgrounds that maintain readability.
    pub fn row_bg_hex(&self) -> &'static str {
        match self {
            Self::Supported => "#D4EDDA",
            Self::Partial => "#FFF3CD",
            Self::Unsupported => "#F8D7DA",
            Self::NotFound => "#FCE4EC",
            Self::WrongSource => "#FFE0B2",
            Self::Unverifiable => "#E2E3E5",
            Self::PeripheralMatch => "#FFF8E1",
        }
    }

    /// Returns the badge color for the verdict cell in Excel.
    /// These are darker, saturated tones used with white bold text.
    pub fn badge_hex(&self) -> &'static str {
        match self {
            Self::Supported => "#28A745",
            Self::Partial => "#D4A017",
            Self::Unsupported => "#DC3545",
            Self::NotFound => "#E91E63",
            Self::WrongSource => "#FF6D00",
            Self::Unverifiable => "#6C757D",
            Self::PeripheralMatch => "#F9A825",
        }
    }

    /// Returns a human-readable uppercase label for this verdict.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Supported => "SUPPORTED",
            Self::Partial => "PARTIAL",
            Self::Unsupported => "UNSUPPORTED",
            Self::NotFound => "NOT FOUND",
            Self::WrongSource => "WRONG SOURCE",
            Self::Unverifiable => "UNVERIFIABLE",
            Self::PeripheralMatch => "PERIPHERAL MATCH",
        }
    }
}

/// Classification of where within a PDF document a passage was found.
/// Based on the standard structure of academic publications (front matter,
/// body sections, back matter, structural elements). The sub-agent assigns
/// this label to each passage during verification so that downstream
/// consumers can assess the evidentiary weight of the passage.
///
/// Front-matter passages (abstract, table of contents) are summaries and
/// carry less evidentiary weight than body-section passages. Back-matter
/// passages (bibliography, appendix) have specific roles. Structural
/// elements (tables, figures, footnotes) can appear in any section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PassageLocation {
    // -- Front matter -------------------------------------------------------
    /// The abstract section at the beginning of the document.
    Abstract,
    /// Preface, foreword, or acknowledgments section.
    Foreword,
    /// Table of contents, list of figures, or list of tables.
    TableOfContents,

    // -- Body sections (standard academic paper structure) -------------------
    /// Introduction section: problem statement, research questions, scope.
    Introduction,
    /// Literature review, related work, background, or prior research section.
    LiteratureReview,
    /// Theoretical framework, conceptual model, or formal definitions section.
    TheoreticalFramework,
    /// Methodology, research design, data collection, or experimental setup.
    Methodology,
    /// Results, findings, empirical analysis, or data presentation section.
    Results,
    /// Discussion section: interpretation, implications, limitations.
    Discussion,
    /// Conclusion, summary, or future work section.
    Conclusion,

    // -- Back matter ---------------------------------------------------------
    /// Reference list, bibliography, or works cited section.
    Bibliography,
    /// Appendix or supplementary material with technical details.
    Appendix,
    /// Glossary, definitions, or terminology section.
    Glossary,

    // -- Structural elements (can appear within any section) -----------------
    /// Table, figure, chart, diagram, or their caption/legend.
    TableOrFigure,
    /// Footnote, endnote, or margin note.
    Footnote,

    // -- Fallback ------------------------------------------------------------
    /// Main body text when the specific section cannot be determined by the
    /// sub-agent. Use this only when none of the other categories apply.
    BodyText,
}

impl PassageLocation {
    /// Returns the snake_case string label for this passage location.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Abstract => "abstract",
            Self::Foreword => "foreword",
            Self::TableOfContents => "table_of_contents",
            Self::Introduction => "introduction",
            Self::LiteratureReview => "literature_review",
            Self::TheoreticalFramework => "theoretical_framework",
            Self::Methodology => "methodology",
            Self::Results => "results",
            Self::Discussion => "discussion",
            Self::Conclusion => "conclusion",
            Self::Bibliography => "bibliography",
            Self::Appendix => "appendix",
            Self::Glossary => "glossary",
            Self::TableOrFigure => "table_or_figure",
            Self::Footnote => "footnote",
            Self::BodyText => "body_text",
        }
    }
}

/// The type of LaTeX correction suggested by the sub-agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionType {
    /// The claim text around the citation should be rephrased.
    Rephrase,
    /// Additional context or qualification should be added.
    AddContext,
    /// The citation should reference a different source.
    ReplaceCitation,
    /// No correction needed.
    None,
}

// ---------------------------------------------------------------------------
// LaTeX parsing output
// ---------------------------------------------------------------------------

/// A single citation occurrence extracted from a LaTeX file. Represents one
/// cite-key within one citation command (e.g., `fama1970` from `\cite{fama1970}`).
/// Multi-key commands like `\cite{a,b,c}` produce three `RawCitation` entries
/// that share the same `group_id`.
#[derive(Debug, Clone)]
pub struct RawCitation {
    /// The BibTeX cite-key (e.g., "fama1970").
    pub cite_key: String,
    /// 1-indexed line number where the citation command appears in the .tex file.
    pub line: usize,
    /// Up to 15 whitespace-delimited words before the citation command on the
    /// same line. Empty string if the citation is at the start of a line.
    pub anchor_before: String,
    /// Up to 15 whitespace-delimited words after the closing brace of the
    /// citation command on the same line. Trailing punctuation is stripped
    /// from the last word. Empty string if the citation is at the end of a line.
    pub anchor_after: String,
    /// Shared identifier for all cite-keys from the same `\cite{a,b,c}` command.
    /// Assigned sequentially starting from 0 during parsing.
    pub group_id: usize,
    /// The most recent `\section{}` or `\subsection{}` heading above this
    /// citation. None if no section heading precedes the citation.
    pub section_title: Option<String>,
    /// Batch assignment index. Set by `assign_batches()` after parsing.
    /// None until batch assignment is performed.
    pub batch_id: Option<usize>,
    /// Surrounding LaTeX text (~200 characters) centered on the citation
    /// command's line. Provides context from adjacent lines for the sub-agent
    /// to understand the claim being cited.
    pub tex_context: String,
}

// ---------------------------------------------------------------------------
// BibTeX parsing output
// ---------------------------------------------------------------------------

/// A single entry extracted from a BibTeX file. Contains the metadata fields
/// relevant for citation verification and PDF matching.
#[derive(Debug, Clone)]
pub struct BibEntry {
    /// The BibTeX entry type as it appears in the .bib file (e.g., "article",
    /// "book", "inproceedings"). Normalized to lowercase during parsing.
    pub entry_type: String,
    /// Author name(s) as they appear in the BibTeX entry. May contain LaTeX
    /// accent commands (e.g., `{\"u}`, `\'{e}`) which are preserved as-is.
    pub author: String,
    /// The title of the cited work.
    pub title: String,
    /// Publication year (e.g., "1970"). None if the `year` field is absent.
    pub year: Option<String>,
    /// Abstract text from the BibTeX entry. None if the `abstract` field
    /// is absent. Provided to sub-agents for targeted search strategies.
    pub bib_abstract: Option<String>,
    /// Keywords from the BibTeX entry. None if the `keywords` field is absent.
    /// Provided to sub-agents for targeted search strategies.
    pub keywords: Option<String>,
    /// URL pointing to the source document (direct PDF link or publisher
    /// landing page). None if the `url` field is absent in the BibTeX entry.
    /// Used by the source acquisition tool to download cited works.
    pub url: Option<String>,
    /// Digital Object Identifier for the cited work. None if the `doi` field
    /// is absent. Can be resolved via `https://doi.org/{doi}` to reach the
    /// publisher page when no direct URL is available.
    pub doi: Option<String>,
    /// Additional BibTeX fields not captured in the named fields above
    /// (e.g., journal, volume, pages, booktitle, publisher, editor, month,
    /// note, series, number, institution, school, howpublished). Keyed by
    /// the lowercase field name. Excludes the seven named fields (author,
    /// title, year, abstract, keywords, url, doi) to avoid duplication.
    pub extra_fields: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Co-citation metadata
// ---------------------------------------------------------------------------

/// Co-citation metadata computed deterministically during LaTeX parsing.
/// Stored alongside each citation row to indicate whether this cite-key
/// appears in a multi-key `\cite{a,b,c}` command and which other cite-keys
/// share the same group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoCitationInfo {
    /// True when the cite-key's group_id is shared by more than one cite-key.
    /// A single `\cite{key}` produces is_co_citation = false.
    pub is_co_citation: bool,
    /// The other cite-keys in the same `\cite{a,b,c}` group, excluding this
    /// cite-key. Empty vec when is_co_citation is false.
    pub co_cited_with: Vec<String>,
}

// ---------------------------------------------------------------------------
// Database row types
// ---------------------------------------------------------------------------

/// Complete representation of a citation_row record from the database.
/// Contains all columns defined in the schema plus computed co-citation
/// metadata deserialized from the co_citation_json column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationRow {
    /// Auto-incremented primary key.
    pub id: i64,
    /// The job this citation row belongs to.
    pub job_id: String,
    /// Group identifier shared by all cite-keys from the same `\cite{}` command.
    pub group_id: i64,
    /// The BibTeX cite-key.
    pub cite_key: String,
    /// Author name(s) resolved from the BibTeX entry.
    pub author: String,
    /// Title resolved from the BibTeX entry.
    pub title: String,
    /// Publication year resolved from the BibTeX entry.
    pub year: Option<String>,
    /// 1-indexed line number in the .tex file.
    pub tex_line: i64,
    /// Word before the citation command.
    pub anchor_before: String,
    /// Word after the citation command.
    pub anchor_after: String,
    /// Section heading above this citation.
    pub section_title: Option<String>,
    /// The indexed_file.id of the matched PDF, or None if no match.
    pub matched_file_id: Option<i64>,
    /// Abstract text from the BibTeX entry.
    pub bib_abstract: Option<String>,
    /// Keywords from the BibTeX entry.
    pub bib_keywords: Option<String>,
    /// Surrounding LaTeX text (~200 characters) centered on the citation
    /// command's line. Provides context for the sub-agent to understand
    /// the claim being cited.
    pub tex_context: Option<String>,
    /// Batch assignment index for sub-agent claiming.
    pub batch_id: Option<i64>,
    /// Row status: "pending", "claimed", "done", "failed".
    pub status: String,
    /// Sub-agent flag: None, "critical", "warning".
    pub flag: Option<String>,
    /// Unix timestamp when this row's batch was claimed.
    pub claimed_at: Option<i64>,
    /// Serialized JSON containing the sub-agent's verification result.
    pub result_json: Option<String>,
    /// Serialized JSON containing co-citation metadata (CoCitationInfo).
    /// Computed deterministically during LaTeX parsing and stored at row
    /// creation time. None for rows created before this feature was added.
    pub co_citation_json: Option<String>,
}

impl CitationRow {
    /// Deserializes the co-citation metadata from the stored JSON column.
    /// Returns a default CoCitationInfo (is_co_citation=false, empty vec)
    /// when the column is NULL or contains invalid JSON.
    pub fn co_citation_info(&self) -> CoCitationInfo {
        self.co_citation_json
            .as_ref()
            .and_then(|json| serde_json::from_str(json).ok())
            .unwrap_or(CoCitationInfo {
                is_co_citation: false,
                co_cited_with: vec![],
            })
    }
}

/// Insert-only subset of `CitationRow` for creating rows in the database.
/// Excludes auto-generated fields (id, status, flag, claimed_at, result_json).
#[derive(Debug, Clone)]
pub struct CitationRowInsert {
    pub job_id: String,
    pub group_id: i64,
    pub cite_key: String,
    pub author: String,
    pub title: String,
    pub year: Option<String>,
    pub tex_line: i64,
    pub anchor_before: String,
    pub anchor_after: String,
    pub section_title: Option<String>,
    pub matched_file_id: Option<i64>,
    pub bib_abstract: Option<String>,
    pub bib_keywords: Option<String>,
    /// Surrounding LaTeX text (~200 characters) centered on the citation.
    pub tex_context: String,
    pub batch_id: Option<i64>,
    /// Serialized CoCitationInfo JSON. Computed during LaTeX parsing to
    /// record whether this cite-key is part of a multi-key citation group.
    pub co_citation_json: Option<String>,
}

// ---------------------------------------------------------------------------
// Sub-agent claim/submit types
// ---------------------------------------------------------------------------

/// The data returned to a sub-agent when it claims a batch. Contains all
/// citation rows in the batch plus the job context needed for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchClaim {
    /// The batch index that was claimed.
    pub batch_id: i64,
    /// Absolute path to the .tex file.
    pub tex_path: String,
    /// Absolute path to the .bib file.
    pub bib_path: String,
    /// The NeuronCite index session to search against.
    pub session_id: i64,
    /// All citation rows in this batch.
    pub rows: Vec<CitationRow>,
}

/// A single passage found in an indexed PDF by the sub-agent. Contains the
/// file reference, page number, extracted text, relevance assessment, and
/// the structural location within the document where the passage was found.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassageRef {
    /// The indexed_file.id where this passage was found.
    pub file_id: i64,
    /// 1-indexed page number within the PDF.
    pub page: i64,
    /// The extracted text passage.
    pub passage_text: String,
    /// Relevance score assigned by the sub-agent (0.0 to 1.0).
    pub relevance_score: f64,
    /// Structural location within the PDF where the passage was found.
    /// Classifies the passage as belonging to a specific section of the
    /// academic document (e.g., abstract, methodology, results).
    pub passage_location: PassageLocation,
    /// Optional chunk ID from the indexed session that the sub-agent based
    /// this passage on. When the agent finds a passage via neuroncite_search,
    /// the search result contains a chunk with known page_start/page_end and
    /// verbatim text. Storing the chunk ID allows the annotation pipeline to
    /// retrieve the exact indexed text for more deterministic matching.
    #[serde(default)]
    pub source_chunk_id: Option<i64>,
}

/// A LaTeX correction suggested by the sub-agent for a specific citation.
/// All fields are mandatory. When no correction is needed, use
/// correction_type = CorrectionType::None with empty strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatexCorrection {
    /// The type of correction.
    pub correction_type: CorrectionType,
    /// The original LaTeX text around the citation.
    pub original_text: String,
    /// The corrected LaTeX text suggested by the sub-agent.
    pub suggested_text: String,
    /// Explanation of why this correction is needed.
    pub explanation: String,
}

/// A passage found in an alternative source (not the cited PDF) during
/// cross-source verification. Used in the other_source_list field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtherSourcePassage {
    /// The extracted text passage from the alternative source.
    pub text: String,
    /// 1-indexed page number within the alternative source PDF.
    pub page: i64,
    /// Relevance score (0.0 to 1.0) of this passage to the claim.
    pub score: f64,
}

/// An alternative source that also contains passages relevant to the claim.
/// Each entry represents one indexed PDF (other than the cited source) where
/// relevant passages were found during cross-source verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtherSourceEntry {
    /// The cite-key or title identifying this alternative source.
    pub cite_key_or_title: String,
    /// Relevant passages found in this alternative source.
    pub passages: Vec<OtherSourcePassage>,
}

/// A single verification result submitted by a sub-agent for one citation row.
/// All fields are mandatory. Fields that do not apply use their type's zero
/// value: empty string for strings, empty vec for arrays, false for booleans,
/// CorrectionType::None for latex_correction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitEntry {
    /// The citation_row.id this result corresponds to.
    pub row_id: i64,
    /// The sub-agent's verdict on whether the citation supports the claim.
    pub verdict: Verdict,
    /// The claim as stated in the LaTeX document.
    pub claim_original: String,
    /// The sub-agent's English formulation of the core claim.
    pub claim_english: String,
    /// Whether the claim was found in the cited PDF (matched_file_id).
    pub source_match: bool,
    /// Structured list of alternative sources where the claim was found.
    /// Empty vec means no alternative sources found. Non-empty vec means the
    /// claim appears in other indexed PDFs beyond the cited source. Replaces
    /// the former boolean other_sources field.
    pub other_source_list: Vec<OtherSourceEntry>,
    /// Passages found in the cited source during verification.
    pub passages: Vec<PassageRef>,
    /// The sub-agent's justification for the verdict.
    pub reasoning: String,
    /// Meta-confidence level (0.0 to 1.0): how certain the agent is that
    /// the chosen verdict is the correct classification. This is NOT a
    /// measure of how strongly the source supports the claim. Each verdict
    /// type has enforced bounds: supported [0.50, 1.00], partial [0.00, 0.80],
    /// unsupported [0.50, 1.00], not_found [0.70, 1.00], wrong_source
    /// [0.50, 1.00], unverifiable [0.50, 1.00], peripheral_match [0.00, 0.69].
    pub confidence: f64,
    /// Number of search iterations performed.
    pub search_rounds: i64,
    /// Sub-agent flag for escalation: "critical", "warning", or empty string.
    pub flag: String,
    /// Array of cite-keys or titles of sources that fit the claim more
    /// accurately than the cited source. Empty vec when no better source
    /// exists. Relevant for wrong_source and partial verdicts.
    pub better_source: Vec<String>,
    /// LaTeX correction suggestion. Always present; uses
    /// correction_type = CorrectionType::None when no correction is needed.
    pub latex_correction: LatexCorrection,
}

// ---------------------------------------------------------------------------
// Status and reporting types
// ---------------------------------------------------------------------------

/// Count of citation rows by status within a job.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatusCounts {
    pub pending: i64,
    pub claimed: i64,
    pub done: i64,
    pub failed: i64,
}

/// Count of batches by their aggregate status within a job.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchCounts {
    /// Total distinct batch_id values in the job.
    pub total: i64,
    /// Batches where all rows are done or failed.
    pub done: i64,
    /// Batches where all rows are pending.
    pub pending: i64,
    /// Batches where at least one row is claimed.
    pub claimed: i64,
}

/// An alert entry for a flagged citation row, included in status responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub row_id: i64,
    pub cite_key: String,
    pub flag: String,
    pub verdict: Option<String>,
    pub reasoning: Option<String>,
}

/// Complete status report for a citation verification job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusReport {
    pub job_id: String,
    pub job_state: String,
    pub total: i64,
    pub status_counts: StatusCounts,
    pub batch_counts: BatchCounts,
    pub verdicts: HashMap<String, i64>,
    pub alerts: Vec<Alert>,
    pub elapsed_seconds: i64,
    pub is_complete: bool,
}

// ---------------------------------------------------------------------------
// Export types
// ---------------------------------------------------------------------------

/// Result of the export operation. Contains paths to the generated files
/// and a summary of verdict counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    /// Path to the generated annotation pipeline CSV file.
    pub csv_path: String,
    /// Path to the generated corrections JSON file.
    pub corrections_path: String,
    /// Path to the generated summary report JSON file.
    pub report_path: String,
    /// Path to the full-detail JSON file containing all citation rows with
    /// complete verification results (verdict, claims, passages, reasoning,
    /// confidence, corrections, etc.).
    pub full_detail_path: String,
    /// Path to the human-readable full-data CSV file with all 38 columns.
    pub data_csv_path: String,
    /// Path to the formatted Excel workbook with all 38 columns.
    pub excel_path: String,
    /// Summary of verdict counts and correction statistics.
    pub summary: ExportSummary,
}

/// Summary statistics included in the export result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSummary {
    pub supported: i64,
    pub partial: i64,
    pub unsupported: i64,
    pub not_found: i64,
    pub wrong_source: i64,
    pub unverifiable: i64,
    /// Citations where the claim text was found only in peripheral sections
    /// (table of contents, bibliography, foreword, appendix, glossary) of the
    /// cited source, with no supporting passage in the document body.
    pub peripheral_match: i64,
    pub corrections_suggested: i64,
    pub critical_alerts: i64,
}

/// Parameters stored in `job.params_json` for citation verification jobs.
/// Serialized when the job is created and deserialized by the claim handler
/// to provide context to sub-agents.
///
/// Supports both single-session (`session_id`) and multi-session
/// (`session_ids`) operation. When `session_ids` is present and non-empty,
/// the agent aggregates files from all listed sessions for PDF matching
/// and searches across all sessions during verification. The `session_id`
/// field is retained for backward compatibility with jobs created before
/// multi-session support was added.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationJobParams {
    /// Absolute path to the .tex file.
    pub tex_path: String,
    /// Absolute path to the .bib file.
    pub bib_path: String,
    /// Primary index session ID. Used as fallback when `session_ids` is
    /// absent (backward compatibility with jobs created before multi-session
    /// support).
    pub session_id: i64,
    /// All session IDs to search against. When present and non-empty,
    /// overrides `session_id`. Files from all listed sessions are
    /// aggregated for cite-key matching, and the agent searches across
    /// all sessions during verification.
    #[serde(default)]
    pub session_ids: Vec<i64>,
    /// Target number of citations per batch.
    pub batch_size: usize,
}

/// Column definitions for the 38-column export CSV and Excel workbook.
/// Each tuple contains (header_name, column_width_in_characters).
/// Passage columns include a `passage_N_location` field that classifies
/// the structural position within the PDF (e.g., abstract, methodology,
/// results) for each of the three passage slots.
pub const EXPORT_COLUMNS: [(&str, u16); 38] = [
    ("id", 5),
    ("tex_line", 8),
    ("section_title", 25),
    ("anchor_before", 30),
    ("anchor_after", 30),
    ("is_co_citation", 12),
    ("co_cited_with", 20),
    ("cite_key", 18),
    ("author", 25),
    ("year", 6),
    ("title", 40),
    ("claim_original", 50),
    ("claim_english", 50),
    ("verdict", 14),
    ("confidence", 10),
    ("source_match", 12),
    ("reasoning", 60),
    ("flag", 10),
    ("latex_correction_type", 18),
    ("latex_correction_explanation", 40),
    ("latex_correction_original", 40),
    ("latex_correction_suggested", 40),
    ("better_source", 30),
    ("passage_1_text", 50),
    ("passage_1_page", 10),
    ("passage_1_score", 10),
    ("passage_1_location", 22),
    ("passage_2_text", 50),
    ("passage_2_page", 10),
    ("passage_2_score", 10),
    ("passage_2_location", 22),
    ("passage_3_text", 50),
    ("passage_3_page", 10),
    ("passage_3_score", 10),
    ("passage_3_location", 22),
    ("other_source_list", 50),
    ("search_rounds", 12),
    ("matched_file_id", 12),
];
