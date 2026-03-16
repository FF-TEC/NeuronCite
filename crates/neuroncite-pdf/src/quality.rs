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

//! Extraction quality assessment for PDF page text.
//!
//! After text is extracted from a PDF page, these heuristics determine whether
//! the extraction quality is sufficient or whether the page should be re-processed
//! through a different backend or OCR. The quality assessment is based on four
//! independent metrics: alphabetic character ratio, Unicode replacement character
//! ratio, line break density, and unique token ratio.

use std::collections::HashSet;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Quality threshold constants
// ---------------------------------------------------------------------------

/// Minimum acceptable ratio of alphabetic characters to total characters.
/// Below this threshold, the text is likely garbled due to encoding issues.
const ALPHABETIC_THRESHOLD: f64 = 0.3;

/// Maximum acceptable ratio of Unicode replacement characters (U+FFFD) to
/// total characters. Above this threshold, font encoding was not decoded
/// correctly by the extraction backend.
const REPLACEMENT_CHAR_THRESHOLD: f64 = 0.05;

/// Maximum acceptable ratio of line break characters to total characters.
/// Above this threshold, the layout reconstruction fragmented continuous text
/// into single-character or single-word lines.
const LINE_BREAK_THRESHOLD: f64 = 0.15;

/// Minimum acceptable ratio of unique whitespace-separated tokens to total
/// tokens. Below this threshold, the text consists of repetitive garbage output
/// rather than natural language.
const UNIQUE_TOKEN_THRESHOLD: f64 = 0.05;

/// Minimum number of whitespace-separated tokens required before the function
/// word check is applied. Short text fragments (fewer tokens than this) are
/// exempt from the function word check because a legitimate table caption or
/// equation block may contain no function words.
const FUNCTION_WORD_MIN_TOKENS: usize = 30;

/// Minimum ratio of tokens that must be common English function words (articles,
/// prepositions, conjunctions, pronouns, auxiliary verbs). Natural English prose
/// typically contains 15-25% function words. Text with zero or near-zero
/// function word ratio, despite having sufficient tokens, is likely garbled
/// (e.g., systematically shifted character encodings that produce alphabetic
/// output but no recognizable words).
const FUNCTION_WORD_THRESHOLD: f64 = 0.02;

/// Set of common English function words used to detect garbled text. These are
/// the highest-frequency closed-class words that appear in virtually all English
/// prose: articles, prepositions, conjunctions, pronouns, and auxiliary verbs.
/// The set is intentionally small and limited to words that are unambiguously
/// function words in English, avoiding content words that might not appear in
/// specialized technical text.
const FUNCTION_WORDS: &[&str] = &[
    "the", "a", "an", "of", "in", "to", "and", "is", "for", "on", "that", "with", "as", "by", "at",
    "from", "or", "are", "was", "be", "this", "it", "not", "but", "has", "have", "we", "can", "if",
    "will", "their", "which", "been", "its", "than", "each", "may", "were", "also", "more",
    "these", "between", "where", "when", "such", "into", "all", "our", "no", "he", "she", "they",
    "his", "her", "do", "does", "did", "had",
];

/// Pre-computed HashSet of FUNCTION_WORDS for O(1) lookup during quality
/// assessment. Built once on first access via LazyLock, avoiding repeated
/// allocation and insertion on every call to `assess_text_quality`.
static FUNCTION_WORD_SET: LazyLock<HashSet<&str>> =
    LazyLock::new(|| FUNCTION_WORDS.iter().copied().collect());

// ---------------------------------------------------------------------------
// Quality result type
// ---------------------------------------------------------------------------

/// Contains the computed quality metrics for a block of extracted text.
/// Each metric is a ratio in the range [0.0, 1.0]. The `passes_all` field
/// is `true` only when every individual metric meets its threshold.
#[derive(Debug, Clone)]
pub struct TextQuality {
    /// Ratio of alphabetic characters (a-z, A-Z, and Unicode alphabetic) to
    /// the total character count.
    pub alphabetic_ratio: f64,

    /// Ratio of Unicode replacement characters (U+FFFD) to the total character
    /// count.
    pub replacement_char_ratio: f64,

    /// Ratio of line break characters ('\n' and '\r') to the total character
    /// count.
    pub line_break_density: f64,

    /// Ratio of unique whitespace-separated tokens to the total token count.
    pub unique_token_ratio: f64,

    /// Ratio of tokens that are common English function words (articles,
    /// prepositions, conjunctions, pronouns, auxiliary verbs) to the total
    /// token count. Natural English prose contains 15-25% function words.
    /// Text with a very low function word ratio despite having many tokens
    /// indicates garbled text (e.g., Caesar-cipher-shifted encodings that
    /// produce alphabetic output but no recognizable English words).
    ///
    /// This metric is only evaluated when the text has at least
    /// FUNCTION_WORD_MIN_TOKENS tokens; shorter text fragments are exempt.
    pub function_word_ratio: f64,

    /// `true` if all metrics pass their respective thresholds. `false` if
    /// any single metric fails.
    pub passes_all: bool,
}

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/// Computes quality metrics for extracted text and evaluates them against
/// the built-in threshold constants.
///
/// For empty input, all ratios are set to 0.0 and `passes_all` is `false`
/// (since the alphabetic ratio of 0.0 < 0.3 fails the threshold).
///
/// # Examples
///
/// ```
/// use neuroncite_pdf::quality::check_text_quality;
///
/// let quality = check_text_quality("This is a well-extracted paragraph of text.");
/// assert!(quality.passes_all);
/// ```
pub fn check_text_quality(text: &str) -> TextQuality {
    // Single-pass character classification: counts total characters, alphabetic
    // characters, Unicode replacement characters (U+FFFD), and line break
    // characters ('\n', '\r') in one iteration over the char stream.
    let mut total_chars: usize = 0;
    let mut alphabetic_count: usize = 0;
    let mut replacement_count: usize = 0;
    let mut line_break_count: usize = 0;

    for c in text.chars() {
        total_chars += 1;
        if c.is_alphabetic() {
            alphabetic_count += 1;
        }
        if c == '\u{FFFD}' {
            replacement_count += 1;
        }
        if c == '\n' || c == '\r' {
            line_break_count += 1;
        }
    }

    if total_chars == 0 {
        return TextQuality {
            alphabetic_ratio: 0.0,
            replacement_char_ratio: 0.0,
            line_break_density: 0.0,
            unique_token_ratio: 0.0,
            function_word_ratio: 0.0,
            passes_all: false,
        };
    }

    let total_f64 = total_chars as f64;
    let alphabetic_ratio = alphabetic_count as f64 / total_f64;
    let replacement_char_ratio = replacement_count as f64 / total_f64;
    let line_break_density = line_break_count as f64 / total_f64;

    // Compute unique token ratio from whitespace-separated tokens. This second
    // pass over the string is unavoidable because split_whitespace applies
    // different logic (whitespace boundaries) than per-character classification.
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let unique_token_ratio = if tokens.is_empty() {
        0.0
    } else {
        let unique_tokens: HashSet<&str> = tokens.iter().copied().collect();
        unique_tokens.len() as f64 / tokens.len() as f64
    };

    // Compute the function word ratio: count how many tokens are common
    // English function words (case-insensitive comparison). This detects
    // garbled text that passes all other heuristics, such as systematic
    // character shifts (Caesar cipher) that produce alphabetic output with
    // high uniqueness but contain no recognizable English words.
    let function_word_count = tokens
        .iter()
        .filter(|token| {
            let lower = token.to_ascii_lowercase();
            FUNCTION_WORD_SET.contains(lower.as_str())
        })
        .count();
    let function_word_ratio = if tokens.is_empty() {
        0.0
    } else {
        function_word_count as f64 / tokens.len() as f64
    };

    // The function word check is only applied when the text has at least
    // FUNCTION_WORD_MIN_TOKENS tokens. Short fragments (table captions,
    // equation blocks, headers) may legitimately contain no function words.
    let function_word_passes =
        tokens.len() < FUNCTION_WORD_MIN_TOKENS || function_word_ratio >= FUNCTION_WORD_THRESHOLD;

    let passes_all = alphabetic_ratio >= ALPHABETIC_THRESHOLD
        && replacement_char_ratio <= REPLACEMENT_CHAR_THRESHOLD
        && line_break_density <= LINE_BREAK_THRESHOLD
        && unique_token_ratio >= UNIQUE_TOKEN_THRESHOLD
        && function_word_passes;

    TextQuality {
        alphabetic_ratio,
        replacement_char_ratio,
        line_break_density,
        unique_token_ratio,
        function_word_ratio,
        passes_all,
    }
}

/// Returns `true` if any quality heuristic fails, indicating that the text
/// should be re-extracted using a fallback backend or OCR.
///
/// This is the inverse of `TextQuality::passes_all` and exists as a
/// standalone function for clarity at call sites where the intent is to
/// decide whether to trigger fallback processing.
pub fn should_fallback(quality: &TextQuality) -> bool {
    !quality.passes_all
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// T-PDF-047: Text with a low alphabetic ratio (below 0.3) triggers fallback.
    /// Constructs a string dominated by digits and punctuation so that the
    /// alphabetic character ratio falls below the threshold.
    #[test]
    fn t_pdf_009_low_alphabetic_ratio_triggers_fallback() {
        // 100 digits, 5 alphabetic characters -> ratio = 5/105 ~= 0.048
        let text = format!("{}{}", "0123456789".repeat(10), "abcde");

        let quality = check_text_quality(&text);

        assert!(
            quality.alphabetic_ratio < ALPHABETIC_THRESHOLD,
            "alphabetic ratio {} should be below threshold {}",
            quality.alphabetic_ratio,
            ALPHABETIC_THRESHOLD
        );
        assert!(
            should_fallback(&quality),
            "should_fallback must return true for low alphabetic ratio"
        );
    }

    /// T-PDF-049: Text with a high replacement character ratio (above 0.05)
    /// triggers fallback. Constructs a string where more than 5% of characters
    /// are U+FFFD replacement characters.
    #[test]
    fn t_pdf_010_high_replacement_char_ratio_triggers_fallback() {
        // 80 alphabetic characters + 20 replacement characters = ratio 20/100 = 0.2
        let text = format!("{}{}", "a".repeat(80), "\u{FFFD}".repeat(20));

        let quality = check_text_quality(&text);

        assert!(
            quality.replacement_char_ratio > REPLACEMENT_CHAR_THRESHOLD,
            "replacement char ratio {} should be above threshold {}",
            quality.replacement_char_ratio,
            REPLACEMENT_CHAR_THRESHOLD
        );
        assert!(
            should_fallback(&quality),
            "should_fallback must return true for high replacement char ratio"
        );
    }

    /// T-PDF-050: Text with a high line break density (above 0.15) triggers
    /// fallback. Constructs a string where line break characters make up more
    /// than 15% of the total character count.
    #[test]
    fn t_pdf_011_high_line_break_density_triggers_fallback() {
        // 5 characters per line + 1 newline = 6 chars per unit.
        // With 10 units: 50 alpha + 10 newlines = 60 total.
        // Line break density = 10/60 ~= 0.167 > 0.15.
        let text = (0..10).map(|_| "abcde").collect::<Vec<_>>().join("\n");

        let quality = check_text_quality(&text);

        assert!(
            quality.line_break_density > LINE_BREAK_THRESHOLD,
            "line break density {} should be above threshold {}",
            quality.line_break_density,
            LINE_BREAK_THRESHOLD
        );
        assert!(
            should_fallback(&quality),
            "should_fallback must return true for high line break density"
        );
    }

    /// T-PDF-051: Text with a low unique token ratio (below 0.05) triggers
    /// fallback. Constructs a string of many repeated tokens so that the
    /// unique/total ratio falls below the threshold.
    #[test]
    fn t_pdf_012_low_unique_token_ratio_triggers_fallback() {
        // Single word "garbage" repeated 100 times -> unique tokens = 1,
        // total tokens = 100, ratio = 0.01 < 0.05.
        let text = vec!["garbage"; 100].join(" ");

        let quality = check_text_quality(&text);

        assert!(
            quality.unique_token_ratio < UNIQUE_TOKEN_THRESHOLD,
            "unique token ratio {} should be below threshold {}",
            quality.unique_token_ratio,
            UNIQUE_TOKEN_THRESHOLD
        );
        assert!(
            should_fallback(&quality),
            "should_fallback must return true for low unique token ratio"
        );
    }

    /// T-PDF-028a: Single-pass quality check produces identical results to the
    /// multi-pass implementation. Tests a string containing all four character
    /// categories (alphabetic, replacement, line break, other) and verifies that
    /// each ratio is computed correctly in a single iteration.
    #[test]
    fn t_pdf_028a_single_pass_correctness() {
        // 40 alphabetic + 5 replacement + 10 newlines + 45 digits = 100 chars.
        let text = format!(
            "{}{}{}{}",
            "a".repeat(40),
            "\u{FFFD}".repeat(5),
            "\n".repeat(10),
            "0".repeat(45),
        );

        let quality = check_text_quality(&text);

        // Alphabetic: 40/100 = 0.40
        assert!(
            (quality.alphabetic_ratio - 0.40).abs() < 1e-10,
            "alphabetic_ratio should be 0.40, got {}",
            quality.alphabetic_ratio
        );
        // Replacement: 5/100 = 0.05
        assert!(
            (quality.replacement_char_ratio - 0.05).abs() < 1e-10,
            "replacement_char_ratio should be 0.05, got {}",
            quality.replacement_char_ratio
        );
        // Line break: 10/100 = 0.10
        assert!(
            (quality.line_break_density - 0.10).abs() < 1e-10,
            "line_break_density should be 0.10, got {}",
            quality.line_break_density
        );
    }

    /// T-PDF-052b: Well-formed academic text passes all quality thresholds.
    /// Verifies the single-pass implementation correctly classifies
    /// representative natural language text as passing.
    #[test]
    fn t_pdf_028b_academic_text_passes_quality() {
        let text = "In this paper, we present a statistical framework \
                    for hypothesis testing in high-dimensional data. \
                    Our approach leverages Bayesian inference to compute \
                    posterior distributions over model parameters.";

        let quality = check_text_quality(text);

        assert!(
            quality.passes_all,
            "academic text must pass all quality thresholds: \
             alpha={:.3}, repl={:.3}, lb={:.3}, uniq={:.3}, fword={:.3}",
            quality.alphabetic_ratio,
            quality.replacement_char_ratio,
            quality.line_break_density,
            quality.unique_token_ratio,
            quality.function_word_ratio,
        );
    }

    /// T-PDF-053: Caesar-cipher-shifted text (character codes shifted by a
    /// constant offset) triggers fallback via the function word check.
    /// This type of garbled output passes alphabetic ratio, replacement char,
    /// line break, and unique token checks because the shifted characters are
    /// still alphabetic and form varied tokens. Only the function word check
    /// detects the absence of recognizable English words.
    #[test]
    fn t_pdf_029_caesar_shifted_text_triggers_fallback() {
        // Simulate Caesar-cipher-shifted text: shift each character in a
        // real English paragraph by +3 positions. The result is alphabetic
        // but contains no recognizable English words.
        let original = "However we have not considered this extension \
                        in the present paper and we leave it for future work \
                        because the statistical properties require additional \
                        investigation and the computational complexity is high";
        let shifted: String = original
            .chars()
            .map(|c| {
                if c.is_ascii_lowercase() {
                    (b'a' + (c as u8 - b'a' + 3) % 26) as char
                } else if c.is_ascii_uppercase() {
                    (b'A' + (c as u8 - b'A' + 3) % 26) as char
                } else {
                    c
                }
            })
            .collect();

        let quality = check_text_quality(&shifted);

        // The shifted text passes the traditional heuristics.
        assert!(
            quality.alphabetic_ratio >= ALPHABETIC_THRESHOLD,
            "shifted text should pass alphabetic ratio check: {:.3}",
            quality.alphabetic_ratio,
        );
        assert!(
            quality.unique_token_ratio >= UNIQUE_TOKEN_THRESHOLD,
            "shifted text should pass unique token ratio check: {:.3}",
            quality.unique_token_ratio,
        );

        // The function word check detects the garbled text.
        assert!(
            quality.function_word_ratio < FUNCTION_WORD_THRESHOLD,
            "shifted text must have near-zero function word ratio: {:.3}",
            quality.function_word_ratio,
        );
        assert!(
            !quality.passes_all,
            "Caesar-shifted text must fail quality check (function word ratio {:.3})",
            quality.function_word_ratio,
        );
        assert!(
            should_fallback(&quality),
            "should_fallback must return true for Caesar-shifted text"
        );
    }

    /// T-PDF-054: Short text fragments (below FUNCTION_WORD_MIN_TOKENS) are
    /// exempt from the function word check. Table captions, equation labels,
    /// and headers may legitimately contain no function words.
    #[test]
    fn t_pdf_030_short_text_exempt_from_function_word_check() {
        // 10 tokens with no function words -- should still pass because
        // the fragment is below the minimum token threshold.
        let text = "GARCH(1,1) volatility estimation results Table 5.2 \
                    Panel regression coefficients";

        let quality = check_text_quality(text);
        let token_count = text.split_whitespace().count();

        assert!(
            token_count < FUNCTION_WORD_MIN_TOKENS,
            "test text must have fewer than {} tokens, has {}",
            FUNCTION_WORD_MIN_TOKENS,
            token_count,
        );
        // The function word ratio is low but the check is not applied.
        assert!(
            quality.passes_all,
            "short fragments must be exempt from function word check: \
             tokens={}, fword_ratio={:.3}",
            token_count, quality.function_word_ratio,
        );
    }

    /// T-PDF-055: Text from real PDF extraction where pdf-extract produces
    /// garbled output with leading slashes (observed in production: the
    /// `/krzhyhu/kdyhqrw...` pattern from misinterpreted ToUnicode CMap
    /// entries). This text contains alphabetic characters and varied tokens
    /// but zero English function words.
    #[test]
    fn t_pdf_031_real_garbled_slash_prefix_text_triggers_fallback() {
        // Simulated garbled output pattern observed in production PDFs.
        // Each "word" has a leading slash and shifted characters. The text
        // has 40+ tokens but zero function words.
        let garbled = "/krzhyhu /kdyhqrw /frqvlghuhg /wklv /h{whqvlrq \
                       /lq /wkh /suhvhqw /sdshu /dqg /zh /ohdyh /lw \
                       /iru /ixwxuh /zrun /ehfdxvh /wkh /vwdwlvwlfdo \
                       /surshuwlhv /uhtxluh /dgglwlrqdo /lqyhvwljdwlrq \
                       /dqg /wkh /frpsxwdwlrqdo /frpsoh{lwb /lv /kljk \
                       /pruhryhu /wkh /uhvxowv /vxjjhvw /wkdw /ixuwkhu \
                       /uhvhdufk /lv /qhhghg /wr /hvwdeolvk /urexvw \
                       /phwkrgv /iru /wklv /fodvv /ri /sureohpv";

        let quality = check_text_quality(garbled);

        assert!(
            !quality.passes_all,
            "garbled slash-prefix text must fail quality check: fword_ratio={:.3}",
            quality.function_word_ratio,
        );
    }

    /// T-PDF-056: Function word ratio computation correctly identifies
    /// function words in mixed-case text. The comparison is case-insensitive,
    /// so "The", "THE", and "the" all count as function words.
    #[test]
    fn t_pdf_032_function_word_case_insensitive() {
        // Text with function words in various cases.
        let text = "The analysis OF data IN This study IS based ON \
                    empirical evidence FROM multiple sources AND The \
                    results ARE consistent WITH prior research BY \
                    several authors WHO have investigated These phenomena \
                    using methods THAT provide reliable estimates FOR \
                    various parameters";

        let quality = check_text_quality(text);

        assert!(
            quality.function_word_ratio > FUNCTION_WORD_THRESHOLD,
            "mixed-case function words must be counted: ratio={:.3}",
            quality.function_word_ratio,
        );
        assert!(
            quality.passes_all,
            "text with recognizable function words (mixed case) must pass"
        );
    }
}
