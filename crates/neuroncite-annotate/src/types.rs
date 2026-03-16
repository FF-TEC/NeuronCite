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

// Core types for the PDF annotation pipeline.
//
// Defines the configuration, intermediate results, and report structures
// used throughout the annotation workflow. These types are independent of
// any specific PDF library and serve as the data contract between modules.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Configuration for a single annotation pipeline run. Specifies the input
/// rows (parsed from CSV/JSON), source and output directories, and the
/// default highlight color for rows that do not specify their own.
#[derive(Debug, Clone)]
pub struct AnnotateConfig {
    /// Parsed annotation instructions (one per quote to highlight).
    pub input_rows: Vec<InputRow>,
    /// Directory containing the source PDF files to annotate.
    pub source_directory: PathBuf,
    /// Directory where annotated PDFs will be saved.
    pub output_directory: PathBuf,
    /// Subdirectory for PDFs that failed annotation entirely.
    pub error_directory: PathBuf,
    /// Default highlight color (hex #RRGGBB) for rows without a color field.
    pub default_color: String,
    /// Pre-extracted page texts from an indexed session, keyed by canonical
    /// PDF path. Each value is a map of 1-indexed page numbers to their text
    /// content. When populated, the text location pipeline's fallback
    /// extraction stage (stage 3.5) uses these cached texts instead of
    /// re-extracting via `neuroncite_pdf::extract_pages()`. This avoids
    /// redundant CPU/OCR work for PDFs that were already indexed.
    ///
    /// When empty (the default), the pipeline falls back to live extraction
    /// as before. The MCP executor populates this map from the database when
    /// a session exists for the source directory.
    pub cached_page_texts: HashMap<PathBuf, HashMap<usize, String>>,

    /// Directory containing previously annotated PDFs from a prior pipeline
    /// run. When set (append mode), the pipeline reads source PDFs from
    /// `source_directory` for text extraction (preserving accurate pdfium
    /// text parsing), creates annotations, saves to `output_directory`, and
    /// then merges existing highlight annotations from the prior output PDFs
    /// into the newly annotated files using lopdf.
    ///
    /// This two-step approach is necessary because pdfium re-linearizes PDFs
    /// when saving via `save_to_file()`. The re-linearized format breaks
    /// text extraction on a second pass, making it impossible to locate
    /// quotes in PDFs that were already annotated. By always extracting text
    /// from the original source PDFs and merging prior annotations separately,
    /// the pipeline produces correct results regardless of how many append
    /// passes are run.
    ///
    /// When None (the default for non-append runs), no annotation merging
    /// occurs.
    pub prior_output_directory: Option<PathBuf>,
}

/// A single annotation instruction parsed from the input CSV or JSON.
/// Title and author identify the target PDF; quote is the text passage
/// to highlight. Color and comment are optional overrides.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputRow {
    /// Title of the paper/book (used for PDF filename matching).
    pub title: String,
    /// Author(s) of the paper/book (used for PDF filename matching).
    pub author: String,
    /// Verbatim text passage to locate and highlight in the PDF.
    pub quote: String,
    /// Highlight color in hex format (#RRGGBB). When absent, the pipeline
    /// uses the default_color from AnnotateConfig.
    #[serde(default)]
    pub color: Option<String>,
    /// Optional popup comment text attached to the highlight annotation.
    #[serde(default)]
    pub comment: Option<String>,
    /// Optional 1-indexed page number hint from the citation verification
    /// pipeline. When present, the locate pipeline searches this page first
    /// before falling back to a full-document scan. The page number comes
    /// from the sub-agent's PassageRef.page field, which identifies where
    /// the passage was found during citation verification.
    #[serde(default)]
    pub page: Option<usize>,
}

/// Detected input format based on content inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputFormat {
    Csv,
    Json,
}

/// Hex color representation. Validated to match the #RRGGBB pattern.
#[derive(Debug, Clone)]
pub struct Color {
    /// The hex string including the leading '#', e.g. "#FFFF00".
    pub hex: String,
}

impl Color {
    /// Parses a hex color string. Returns None if the format is invalid.
    /// Accepts 6-digit hex with leading '#' (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.len() != 7 || !s.starts_with('#') {
            return None;
        }
        let hex_part = &s[1..];
        if hex_part.chars().all(|c| c.is_ascii_hexdigit()) {
            Some(Color {
                hex: s.to_uppercase(),
            })
        } else {
            None
        }
    }

    /// Extracts the red component (0-255).
    pub fn red(&self) -> u8 {
        u8::from_str_radix(&self.hex[1..3], 16).unwrap_or(255)
    }

    /// Extracts the green component (0-255).
    pub fn green(&self) -> u8 {
        u8::from_str_radix(&self.hex[3..5], 16).unwrap_or(255)
    }

    /// Extracts the blue component (0-255).
    pub fn blue(&self) -> u8 {
        u8::from_str_radix(&self.hex[5..7], 16).unwrap_or(255)
    }
}

/// The method by which a quote was located in the PDF text. Ordered by
/// pipeline stage priority (exact is most reliable, fallback_extract is
/// least precise for positioning).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchMethod {
    /// Stage 1: exact string match on pdfium-extracted text.
    Exact,
    /// Stage 2: match after Unicode normalization, whitespace collapse,
    /// and hyphen removal.
    Normalized,
    /// Stage 3: Levenshtein sliding-window match (>= 90% similarity).
    Fuzzy,
    /// Stage 3+: quote spans two adjacent pages. Found by concatenating the
    /// text of consecutive pages and running exact or normalized matching on
    /// the combined text. Bounding boxes cover the content area of the first
    /// page where the quote begins; per-character coordinates are not available
    /// across page boundaries.
    CrossPage,
    /// Stage 3.5: text found via neuroncite-pdf's multi-backend extraction
    /// pipeline (pdf-extract, OCR) when pdfium text extraction returned
    /// empty pages. Bounding boxes are page-level content-area approximations
    /// (not per-character).
    FallbackExtract,
    /// Stage 4: OCR via Tesseract hOCR output with bounding boxes.
    Ocr,
    /// Full-page fallback: the quote text was not located by any pipeline
    /// stage, but the page number is known from the citation verification
    /// pipeline. A full-page highlight covers the content area of the known
    /// page. A comment annotation notes that the passage was not located
    /// precisely.
    PageLevel,
}

/// Result of locating a quote in a PDF document. Contains the page number,
/// character range, bounding box coordinates, and the method used.
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// 1-indexed page number where the quote was found.
    pub page_number: usize,
    /// Start index in the page text (character offset, 0-based).
    pub char_start: usize,
    /// End index in the page text (exclusive).
    pub char_end: usize,
    /// Bounding rectangles for the matched text region. Each rect is
    /// (left, bottom, right, top) in PDF points. For stages 1-3, one
    /// rect per character; for stage 4 (OCR), one rect per word.
    pub bounding_boxes: Vec<[f32; 4]>,
    /// Which pipeline stage produced this match.
    pub method: MatchMethod,
    /// Similarity score from the fuzzy matching stage (0.0 to 1.0). Set when
    /// `method` is `MatchMethod::Fuzzy` and represents the best normalized
    /// Levenshtein distance that met the threshold. `None` for all other
    /// match methods.
    pub fuzzy_score: Option<f64>,
}

/// Result of the pre-annotation verification check. Verifies that each
/// character in the matched range has a valid bounding box from pdfium.
#[derive(Debug, Clone, Serialize)]
pub struct PreCheckResult {
    /// Whether all characters have valid bounding boxes.
    pub passed: bool,
    /// Total characters in the matched range.
    pub total_chars: usize,
    /// Characters that have a valid bounding box.
    pub chars_with_bbox: usize,
    /// Sum of all character bounding box areas in square PDF points.
    pub total_area_sq_pts: f32,
}
