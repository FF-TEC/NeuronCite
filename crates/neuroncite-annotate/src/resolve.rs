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

// PDF filename fuzzy matching for resolving input rows to PDF files.
//
// Academic PDFs typically follow naming conventions like:
//   "Title (Author, Year).pdf"
//   "Title (Author & Author, Year).pdf"
//   "Title (Author) [Type].pdf"
//
// This module extracts title and author from filenames using regex patterns,
// normalizes both input and filename strings (lowercase, strip punctuation,
// collapse whitespace), and computes a Jaro-Winkler similarity score with
// weighted title (60%) and author (40%) components.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use strsim::jaro_winkler;

use crate::error::AnnotateError;
use crate::types::InputRow;

/// Minimum combined similarity score (title * 0.6 + author * 0.4) required
/// to accept a filename as a match for an input row.
const MATCH_THRESHOLD: f64 = 0.80;

/// Compiled regex for stripping bracketed annotations from PDF filenames.
/// Stored as a module-level static via `LazyLock` so the regex is compiled
/// exactly once across all calls to `extract_metadata`, rather than being
/// recompiled on every invocation. Matches patterns like "[Book]",
/// "[PDF preprint version]", "[arXiv PDF]", with optional surrounding
/// whitespace.
static RE_BRACKET: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s*\[[^\]]*\]\s*").expect("bracket removal regex must compile"));

/// Result of matching input rows to PDF files: a map of matched PDFs to
/// their assigned input rows, and a list of unmatched rows.
pub type MatchResult = (HashMap<PathBuf, Vec<InputRow>>, Vec<InputRow>);

/// Matches input rows to PDF files in the given directory based on fuzzy
/// filename comparison. Returns a map of matched PDF paths to their
/// assigned input rows, plus a list of input rows that could not be matched
/// to any PDF file.
///
/// Multiple input rows can match the same PDF (e.g., multiple quotes from
/// the same paper). The grouping enables opening each PDF once and applying
/// all annotations before saving.
pub fn match_pdfs_to_quotes(
    source_directory: &Path,
    rows: Vec<InputRow>,
) -> Result<MatchResult, AnnotateError> {
    let pdfs = neuroncite_pdf::discover_pdfs(source_directory)
        .map_err(|e| AnnotateError::PdfDiscovery(format!("{e}")))?;

    // Pre-extract metadata from all PDF filenames once.
    let pdf_metadata: Vec<(PathBuf, Option<(String, String)>)> = pdfs
        .iter()
        .map(|p| {
            let filename = p
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let meta = extract_metadata(&filename);
            (p.clone(), meta)
        })
        .collect();

    let mut matched: HashMap<PathBuf, Vec<InputRow>> = HashMap::new();
    let mut unmatched: Vec<InputRow> = Vec::new();

    for row in rows {
        if let Some(best_pdf) = find_best_match(&row, &pdf_metadata) {
            matched.entry(best_pdf).or_default().push(row);
        } else {
            unmatched.push(row);
        }
    }

    Ok((matched, unmatched))
}

/// Finds the best matching PDF for an input row by computing a weighted
/// Jaro-Winkler similarity score across all candidate filenames.
///
/// When the input title contains a subtitle separator (colon, em-dash, or
/// en-dash), the function also computes similarity using only the main title
/// (text before the separator). The higher of the two scores is used. This
/// handles BibTeX titles like "Quantitative Risk Management: Concepts,
/// Techniques and Tools" matching a filename that only contains the main
/// title "Quantitative Risk Management".
fn find_best_match(
    row: &InputRow,
    candidates: &[(PathBuf, Option<(String, String)>)],
) -> Option<PathBuf> {
    let norm_title = normalize(&row.title);
    let norm_author = normalize(&row.author);

    // Extract the main title (before the first subtitle separator) for an
    // additional comparison pass. Academic titles use ":" as the primary
    // separator, with " -- " and " - " as alternatives. The main title
    // variant is only used when it is meaningfully shorter than the full
    // title (at least 4 characters shorter after normalization).
    // Stored as Option<String> so downstream code can pattern-match on it
    // directly instead of using an unwrap() guarded by a separate bool.
    let main_title_normalized: Option<String> = strip_subtitle(&row.title)
        .map(|main| normalize(&main))
        .filter(|main| main.len() + 4 <= norm_title.len());

    let mut best_score = 0.0_f64;
    let mut best_pdf: Option<PathBuf> = None;

    for (pdf_path, meta) in candidates {
        let score = if let Some((file_title, file_author)) = meta {
            let norm_file_title = normalize(file_title);
            let title_sim = jaro_winkler(&norm_title, &norm_file_title);

            // When a subtitle variant exists, also compute similarity against
            // the main title and take the higher score. This prevents subtitle
            // text from diluting the Jaro-Winkler score when the filename only
            // contains the main title.
            let title_sim = if let Some(ref main) = main_title_normalized {
                let main_sim = jaro_winkler(main, &norm_file_title);
                title_sim.max(main_sim)
            } else {
                title_sim
            };

            let author_sim = jaro_winkler(&norm_author, &normalize(file_author));
            title_sim * 0.6 + author_sim * 0.4
        } else {
            // Filename without parseable metadata (no parenthesized author/year):
            // match the full filename against the title. The score uses the full
            // Jaro-Winkler similarity without the 0.6 title weight, because there
            // is no author component to combine with. A high title similarity
            // alone (>= MATCH_THRESHOLD) is sufficient to identify the correct PDF
            // when the filename consists entirely of the title.
            let filename = pdf_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let norm_filename = normalize(&filename);
            let title_sim = jaro_winkler(&norm_title, &norm_filename);

            if let Some(ref main) = main_title_normalized {
                let main_sim = jaro_winkler(main, &norm_filename);
                title_sim.max(main_sim)
            } else {
                title_sim
            }
        };

        if score > best_score && score >= MATCH_THRESHOLD {
            best_score = score;
            best_pdf = Some(pdf_path.clone());
        }
    }

    best_pdf
}

/// Strips the subtitle from a title string by splitting at the first
/// subtitle separator. Returns the main title (trimmed) if a separator
/// is found and the result is non-empty, or None otherwise.
///
/// Recognized separators (in priority order):
/// - ": " (colon followed by space, the standard academic convention)
/// - " -- " (em-dash with spaces)
/// - " - " (en-dash/hyphen with spaces, checked last to avoid matching
///   hyphenated compound words like "Bid-Ask")
fn strip_subtitle(title: &str) -> Option<String> {
    for sep in &[": ", " -- ", " - "] {
        if let Some(pos) = title.find(sep) {
            let main = title[..pos].trim();
            if !main.is_empty() {
                return Some(main.to_string());
            }
        }
    }
    None
}

/// Extracts (title, author) from a PDF filename using the common academic
/// naming convention. Handles patterns like:
///   "Title (Author, Year)"
///   "Title (Author & Author, Year)"
///   "Title (Author1, Author2 & Author3, Year)" -- comma-separated authors
///   "Title (Author)"
///   "Title (Subtitle) (Author, Year)" -- title includes "(Subtitle)"
///
/// Strips bracketed annotations like "[Book]", "[arXiv PDF]" before parsing.
/// Uses `rfind('(')` to locate the LAST parenthesized group, which contains
/// the author/year in academic filename conventions. Within that group, splits
/// on the LAST comma to separate authors from year. This correctly handles
/// multi-author names with commas (e.g., "McNeil, Frey & Embrechts") that
/// the previous regex-based approach truncated at the first comma.
fn extract_metadata(filename: &str) -> Option<(String, String)> {
    // Reference the module-level compiled regex via LazyLock. The regex is
    // compiled once on first access and reused across all subsequent calls.
    let re_bracket = &*RE_BRACKET;
    let cleaned = re_bracket.replace_all(filename, " ");
    let cleaned = cleaned.trim();

    // Find the last parenthesized group. Using rfind('(') ensures that
    // parenthesized subtitles in the title (e.g., "(Complete Samples)") are
    // treated as part of the title, not the author block.
    let paren_start = cleaned.rfind('(')?;
    let paren_end = cleaned[paren_start..].rfind(')')?;
    let paren_end = paren_start + paren_end;

    let title = cleaned[..paren_start].trim();
    if title.is_empty() {
        return None;
    }

    let inside = &cleaned[paren_start + 1..paren_end];

    // Split on the LAST comma to separate authors from year. This handles
    // multi-author names with commas: "McNeil, Frey & Embrechts, 2015"
    // splits into authors="McNeil, Frey & Embrechts" and year_candidate="2015".
    if let Some(last_comma) = inside.rfind(',') {
        let authors_part = inside[..last_comma].trim();
        let year_candidate = inside[last_comma + 1..].trim();

        // If the part after the last comma is a 4-digit year, treat it as
        // the year and everything before as authors.
        if year_candidate.len() == 4 && year_candidate.chars().all(|c| c.is_ascii_digit()) {
            return Some((title.to_string(), authors_part.to_string()));
        }
    }

    // No year found: treat the entire parenthesized content as the author.
    Some((title.to_string(), inside.trim().to_string()))
}

/// Normalizes a string for similarity comparison: lowercase, remove all
/// non-alphanumeric/non-whitespace characters, collapse whitespace.
fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// T-ANNOTATE-020: Exact title and author match against a filename
    /// following the "(Author, Year)" convention.
    #[test]
    fn t_annotate_020_match_exact_filename() {
        let meta = extract_metadata(
            "Efficient Capital Markets A Review of Theory and Empirical Work (Fama, 1970)",
        );
        let (title, author) = meta.unwrap();
        assert!(title.contains("Efficient Capital Markets"));
        assert_eq!(author, "Fama");
    }

    /// T-ANNOTATE-021: Matching is case-insensitive.
    #[test]
    fn t_annotate_021_match_case_insensitive() {
        let norm1 = normalize("EFFICIENT CAPITAL Markets");
        let norm2 = normalize("efficient capital markets");
        assert_eq!(norm1, norm2);
    }

    /// T-ANNOTATE-022: Partial author matches work (e.g., input "Fama"
    /// matches filename "Fama & French").
    #[test]
    fn t_annotate_022_match_partial_author() {
        let meta = extract_metadata(
            "Common Risk Factors in the Returns on Stocks and Bonds (Fama & French, 1993)",
        );
        let (_, author) = meta.unwrap();
        // The regex captures "Fama & French" as the first author group.
        assert!(author.contains("Fama"));
    }

    /// T-ANNOTATE-025: Filenames without year in parentheses are still parsed.
    #[test]
    fn t_annotate_025_match_without_year() {
        let meta = extract_metadata("Forecasting Principles & Practice (Hyndman)");
        assert!(meta.is_some());
        let (title, author) = meta.unwrap();
        assert!(title.contains("Forecasting"));
        assert_eq!(author, "Hyndman");
    }

    /// T-ANNOTATE-026: Bracketed annotations like "[Book]" are stripped
    /// before metadata extraction.
    #[test]
    fn t_annotate_026_strip_bracketed_annotations() {
        let meta = extract_metadata("Exploratory Data Analysis (Tukey, 1977) [Book]");
        let (title, author) = meta.unwrap();
        assert!(title.contains("Exploratory Data Analysis"));
        assert_eq!(author, "Tukey");
    }

    /// T-ANNOTATE-027: Filenames without parenthesized author return None.
    #[test]
    fn t_annotate_027_no_parens_returns_none() {
        let meta = extract_metadata("Some Random PDF Document");
        assert!(meta.is_none());
    }

    /// T-ANNOTATE-028: The normalize function collapses multiple spaces
    /// and strips punctuation.
    #[test]
    fn t_annotate_028_normalize_strips_punctuation() {
        let result = normalize("Hello,  World! (2024)");
        assert_eq!(result, "hello world 2024");
    }

    /// T-ANNOTATE-029: Filenames with parenthesized subtitles before the
    /// author group. The greedy regex must match the LAST parenthesized
    /// group as the author, not the first. This was the root cause of
    /// Shapiro & Wilk not being matched: "Complete Samples" was incorrectly
    /// captured as the author instead of "Shapiro & Wilk".
    #[test]
    fn t_annotate_029_parenthesized_subtitle_in_title() {
        let meta = extract_metadata(
            "An Analysis of Variance Test for Normality (Complete Samples) (Shapiro & Wilk, 1965)",
        );
        let (title, author) = meta.unwrap();
        assert!(
            title.contains("Complete Samples"),
            "title must include the parenthesized subtitle, got: {title}"
        );
        assert!(
            author.contains("Shapiro"),
            "author must be 'Shapiro & Wilk', got: {author}"
        );
        assert!(
            !author.contains("Complete"),
            "author must NOT contain 'Complete Samples', got: {author}"
        );
    }

    /// T-ANNOTATE-224: Filenames with multiple parenthesized groups. The
    /// regex must capture the LAST parenthesized group as the author,
    /// regardless of how many parenthesized sections appear in the title.
    #[test]
    fn t_annotate_224_multiple_parenthesized_groups() {
        let meta = extract_metadata(
            "Information Theory (IT) and an Extension (EXT) of the Maximum Likelihood Principle (Akaike, 1973)",
        );
        let (title, author) = meta.unwrap();
        assert_eq!(author, "Akaike");
        assert!(title.contains("(IT)"));
        assert!(title.contains("(EXT)"));
    }

    /// T-ANNOTATE-225: Filename with a single author group and no
    /// subtitle parentheses. Verifies that the greedy regex does not
    /// break the common case.
    #[test]
    fn t_annotate_225_single_author_group_still_works() {
        let meta =
            extract_metadata("The Variation of Certain Speculative Prices (Mandelbrot, 1963)");
        let (title, author) = meta.unwrap();
        assert!(title.contains("Speculative Prices"));
        assert_eq!(author, "Mandelbrot");
    }

    /// T-ANNOTATE-226: Filename with a bracketed annotation AND
    /// parenthesized subtitle. Both stripping and greedy matching
    /// must work together.
    #[test]
    fn t_annotate_226_brackets_and_subtitle() {
        let meta = extract_metadata(
            "Computation and Analysis of Multiple Structural Change Models (Bai & Perron, 2003) [PDF preprint version]",
        );
        let (title, author) = meta.unwrap();
        assert!(title.contains("Structural Change"));
        assert!(author.contains("Bai"));
    }

    /// T-ANNOTATE-227: Filename with ampersand in the author field.
    #[test]
    fn t_annotate_227_ampersand_in_author() {
        let meta = extract_metadata(
            "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market (Roll, 1984)",
        );
        let (title, author) = meta.unwrap();
        assert_eq!(author, "Roll");
        assert!(title.contains("Bid-Ask Spread"));
    }

    /// T-ANNOTATE-228: Multi-author names with commas are captured in full.
    /// Regression test for the bug where the regex `[^,)]+` stopped at the
    /// first comma, truncating "McNeil, Frey & Embrechts" to just "McNeil".
    /// The rfind-based approach splits on the LAST comma, correctly
    /// identifying the year and preserving the full author string.
    #[test]
    fn t_annotate_228_comma_separated_multi_authors() {
        let meta =
            extract_metadata("Quantitative Risk Management (McNeil, Frey & Embrechts, 2015)");
        let (title, author) = meta.unwrap();
        assert_eq!(
            author, "McNeil, Frey & Embrechts",
            "full multi-author string must be captured, including commas"
        );
        assert_eq!(title, "Quantitative Risk Management");
    }

    /// T-ANNOTATE-065: Authors with "Last, First" comma format are preserved.
    /// When only one comma appears and the part after it is not a 4-digit year,
    /// the entire content is treated as the author string.
    #[test]
    fn t_annotate_065_author_with_first_name() {
        let meta = extract_metadata("Some Paper (Smith, John)");
        let (title, author) = meta.unwrap();
        // "John" is not a 4-digit year, so the entire "Smith, John" is
        // treated as the author string.
        assert_eq!(author, "Smith, John");
        assert_eq!(title, "Some Paper");
    }

    /// T-ANNOTATE-066: Two authors separated by & with year correctly parsed.
    #[test]
    fn t_annotate_066_two_authors_ampersand_and_year() {
        let meta = extract_metadata(
            "Common Risk Factors in the Returns on Stocks and Bonds (Fama & French, 1993)",
        );
        let (title, author) = meta.unwrap();
        assert_eq!(author, "Fama & French");
        assert_eq!(
            title,
            "Common Risk Factors in the Returns on Stocks and Bonds"
        );
    }

    /// T-ANNOTATE-067: Three authors with commas and ampersand, plus bracketed
    /// annotation. Verifies that bracket stripping, rfind for last paren, and
    /// rfind for last comma all work together.
    #[test]
    fn t_annotate_067_three_authors_brackets_combined() {
        let meta = extract_metadata(
            "Statistical Analysis with Missing Data (Little, Rubin & Others, 2020) [Book]",
        );
        let (title, author) = meta.unwrap();
        assert_eq!(author, "Little, Rubin & Others");
        assert_eq!(title, "Statistical Analysis with Missing Data");
    }

    // -----------------------------------------------------------------------
    // Subtitle stripping tests (NEW-5 regression tests)
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-068: strip_subtitle extracts the main title before a colon
    /// separator. Colon is the standard academic subtitle separator.
    #[test]
    fn t_annotate_068_strip_subtitle_colon() {
        let result = strip_subtitle("Quantitative Risk Management: Concepts, Techniques and Tools");
        assert_eq!(result, Some("Quantitative Risk Management".to_string()));
    }

    /// T-ANNOTATE-069: strip_subtitle extracts the main title before an
    /// em-dash separator.
    #[test]
    fn t_annotate_069_strip_subtitle_em_dash() {
        let result = strip_subtitle("Statistical Learning -- An Introduction to Modern Methods");
        assert_eq!(result, Some("Statistical Learning".to_string()));
    }

    /// T-ANNOTATE-233: strip_subtitle extracts the main title before a
    /// hyphen-with-spaces separator.
    #[test]
    fn t_annotate_233_strip_subtitle_hyphen_spaces() {
        let result = strip_subtitle("Data Science - From Theory to Practice");
        assert_eq!(result, Some("Data Science".to_string()));
    }

    /// T-ANNOTATE-234: strip_subtitle returns None when no subtitle separator
    /// is present.
    #[test]
    fn t_annotate_234_strip_subtitle_no_separator() {
        let result = strip_subtitle("Efficient Capital Markets");
        assert!(result.is_none());
    }

    /// T-ANNOTATE-235: strip_subtitle does not match hyphens without spaces.
    /// This prevents false positives on compound words like "Bid-Ask".
    #[test]
    fn t_annotate_235_strip_subtitle_no_match_compound_words() {
        let result = strip_subtitle("A Simple Implicit Measure of the Effective Bid-Ask Spread");
        assert!(
            result.is_none(),
            "hyphen without spaces must not be treated as a subtitle separator"
        );
    }

    /// T-ANNOTATE-236: BibTeX title with subtitle matches a filename that
    /// contains only the main title. Regression test for NEW-5 where the
    /// citation export annotation failed because "Quantitative Risk Management:
    /// Concepts, Techniques and Tools" scored below the 0.80 Jaro-Winkler
    /// threshold against the filename "Quantitative Risk Management".
    #[test]
    fn t_annotate_236_subtitle_title_matches_filename_without_subtitle() {
        let row = InputRow {
            title: "Quantitative Risk Management: Concepts, Techniques and Tools".into(),
            author: "McNeil, Frey & Embrechts".into(),
            quote: "test".into(),
            color: None,
            comment: None,
            page: None,
        };

        let candidates = vec![(
            PathBuf::from("Quantitative Risk Management (McNeil, Frey & Embrechts, 2015).pdf"),
            Some((
                "Quantitative Risk Management".to_string(),
                "McNeil, Frey & Embrechts".to_string(),
            )),
        )];

        let result = find_best_match(&row, &candidates);
        assert!(
            result.is_some(),
            "BibTeX title with subtitle must match filename without subtitle"
        );
    }

    /// T-ANNOTATE-237: Title without subtitle still matches correctly
    /// (no regression from the subtitle stripping logic).
    #[test]
    fn t_annotate_237_title_without_subtitle_still_matches() {
        let row = InputRow {
            title: "Efficient Capital Markets".into(),
            author: "Fama".into(),
            quote: "test".into(),
            color: None,
            comment: None,
            page: None,
        };

        let candidates = vec![(
            PathBuf::from(
                "Efficient Capital Markets A Review of Theory and Empirical Work (Fama, 1970).pdf",
            ),
            Some((
                "Efficient Capital Markets A Review of Theory and Empirical Work".to_string(),
                "Fama".to_string(),
            )),
        )];

        let result = find_best_match(&row, &candidates);
        assert!(result.is_some(), "title without subtitle must still match");
    }

    /// T-ANNOTATE-238: strip_subtitle returns None for empty main title
    /// (colon at start of string).
    #[test]
    fn t_annotate_238_strip_subtitle_empty_main_title() {
        let result = strip_subtitle(": Just a Subtitle");
        assert!(
            result.is_none(),
            "empty main title before separator must return None"
        );
    }
}
