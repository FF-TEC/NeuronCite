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

// LaTeX citation command parser.
//
// Extracts all citation commands from a LaTeX document's text content.
// Recognized commands: \cite, \citep, \citet, \autocite, \parencite,
// \textcite. Multi-key commands like \cite{a,b,c} produce three separate
// RawCitation entries sharing the same group_id. Tracks \section{} and
// \subsection{} headings to set section_title for each citation. Extracts
// up to 25 words before and after each citation command as anchor context,
// plus ~500 characters of surrounding LaTeX text as tex_context for
// sub-agent verification. Lines starting with % (LaTeX comments) are skipped.

use regex::Regex;
use std::sync::LazyLock;

use crate::types::RawCitation;

/// Regex pattern matching LaTeX citation commands. Captures:
/// - Group 1: the command name (cite, citep, citet, autocite, parencite, textcite)
/// - Group 2: the brace-delimited argument containing one or more cite-keys
///
/// The pattern handles optional square-bracket arguments before the brace
/// argument (e.g., \cite[p.~42]{key}) by consuming them with a non-greedy match.
static CITE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\\(cite|citep|citet|autocite|parencite|textcite)(?:\[[^\]]*\])*\{([^}]+)\}")
        .expect("citation regex must compile")
});

/// Regex pattern matching LaTeX section headings. Captures the heading text
/// from \section{}, \subsection{}, and \subsubsection{} commands.
static SECTION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\\(?:sub)*section\{([^}]+)\}").expect("section regex must compile")
});

/// Maximum number of whitespace-delimited words to extract as anchor context
/// before and after each citation command. 25 words covers one full academic
/// sentence in both directions, which is sufficient to locate the claim within
/// the paragraph without reading the entire .tex file.
const MAX_ANCHOR_WORDS: usize = 25;

/// Target character count for the tex_context field. The extractor collects
/// surrounding lines until this approximate length is reached. 500 characters
/// covers roughly 3-5 lines of LaTeX source, capturing the full sentence plus
/// neighbouring sentences for disambiguation when anchor words alone are
/// insufficient.
const TARGET_CONTEXT_CHARS: usize = 500;

/// Parses all citation commands from the given LaTeX content string.
///
/// Returns a Vec of `RawCitation` entries, one per cite-key per citation
/// command. Multi-key commands produce multiple entries with the same
/// `group_id`. Lines that are LaTeX comments (trimmed line starts with `%`)
/// are skipped.
///
/// Each citation receives:
/// - `anchor_before`: up to 25 words preceding the citation on the same line
/// - `anchor_after`: up to 25 words following the citation on the same line
/// - `tex_context`: ~500 characters of surrounding LaTeX text from adjacent
///   lines, providing broader context for sub-agent verification
///
/// # Arguments
///
/// * `tex_content` - The full text content of a .tex file.
///
/// # Returns
///
/// A vector of `RawCitation` entries ordered by their appearance in the
/// document. The vector is empty if no citation commands are found.
pub fn parse_citations(tex_content: &str) -> Vec<RawCitation> {
    let lines: Vec<&str> = tex_content.lines().collect();
    let mut citations = Vec::new();
    let mut current_section: Option<String> = None;
    let mut group_counter: usize = 0;

    for (line_idx, line) in lines.iter().enumerate() {
        let line_number = line_idx + 1; // 1-indexed

        // Skip LaTeX comment lines. A comment line has % as the first
        // non-whitespace character.
        if line.trim_start().starts_with('%') {
            continue;
        }

        // Track section headings to assign section_title to subsequent citations.
        if let Some(caps) = SECTION_RE.captures(line) {
            current_section = Some(caps[1].to_string());
        }

        // Find all citation commands on this line.
        for cite_match in CITE_RE.captures_iter(line) {
            let full_match = cite_match.get(0).expect("full match must exist");
            let keys_str = &cite_match[2];

            // Extract anchor context: up to MAX_ANCHOR_WORDS words before and
            // after the citation command on the same line.
            let before_text = &line[..full_match.start()];
            let after_text = &line[full_match.end()..];

            let anchor_before = extract_trailing_words(before_text, MAX_ANCHOR_WORDS);
            let anchor_after = extract_leading_words(after_text, MAX_ANCHOR_WORDS);

            // Build surrounding LaTeX context (~200 chars) from adjacent lines.
            let tex_context = extract_tex_context(&lines, line_idx, TARGET_CONTEXT_CHARS);

            // Split comma-separated cite-keys. All keys from one command
            // share the same group_id.
            let current_group = group_counter;
            group_counter += 1;

            let keys: Vec<&str> = keys_str.split(',').map(|k| k.trim()).collect();

            for key in &keys {
                if key.is_empty() {
                    continue;
                }
                citations.push(RawCitation {
                    cite_key: (*key).to_string(),
                    line: line_number,
                    anchor_before: anchor_before.clone(),
                    anchor_after: anchor_after.clone(),
                    group_id: current_group,
                    section_title: current_section.clone(),
                    batch_id: None,
                    tex_context: tex_context.clone(),
                });
            }
        }
    }

    citations
}

/// Extracts up to `max_words` whitespace-delimited words from the end of the
/// given text. Returns an empty string if the text contains no words.
fn extract_trailing_words(text: &str, max_words: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let start = words.len().saturating_sub(max_words);
    words[start..].join(" ")
}

/// Extracts up to `max_words` whitespace-delimited words from the start of the
/// given text. Strips trailing punctuation (commas, periods, semicolons, colons,
/// closing parentheses, closing brackets) from the last extracted word. Returns
/// an empty string if the text contains no words.
fn extract_leading_words(text: &str, max_words: usize) -> String {
    let mut words: Vec<&str> = text.split_whitespace().take(max_words).collect();
    // Strip trailing punctuation from the last word to produce clean context.
    if let Some(last) = words.last_mut() {
        *last = last.trim_end_matches([',', '.', ';', ':', ')', ']']);
    }
    words.join(" ")
}

/// Extracts surrounding LaTeX text centered on the given line index. Expands
/// outward from the center line, alternating before and after, until the total
/// character count reaches `target_chars` or no more lines are available.
///
/// Returns the concatenated text of the selected lines joined by newlines.
fn extract_tex_context(lines: &[&str], center_line_idx: usize, target_chars: usize) -> String {
    if lines.is_empty() {
        return String::new();
    }

    let mut start = center_line_idx;
    let mut end = center_line_idx;
    let mut total_len = lines[center_line_idx].len();

    // Expand outward from the center line until the target character count is
    // reached. Each iteration adds one line from before and/or after the center.
    while total_len < target_chars {
        let mut grew = false;

        if start > 0 {
            start -= 1;
            total_len += lines[start].len() + 1; // +1 for the joining newline
            grew = true;
        }

        if total_len < target_chars && end + 1 < lines.len() {
            end += 1;
            total_len += lines[end].len() + 1;
            grew = true;
        }

        if !grew {
            break;
        }
    }

    lines[start..=end].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-CIT-001: Single \cite{key} extraction. Verifies that a single
    /// citation command produces one RawCitation with correct fields.
    /// The anchor_before captures all words before the citation on the same line.
    #[test]
    fn t_cit_001_single_cite() {
        let tex = r"This is shown by \cite{fama1970} in the literature.";
        let cits = parse_citations(tex);

        assert_eq!(cits.len(), 1);
        assert_eq!(cits[0].cite_key, "fama1970");
        assert_eq!(cits[0].line, 1);
        assert_eq!(cits[0].anchor_before, "This is shown by");
        assert_eq!(cits[0].anchor_after, "in the literature");
        assert_eq!(cits[0].group_id, 0);
        assert!(cits[0].section_title.is_none());
        assert!(
            !cits[0].tex_context.is_empty(),
            "tex_context must contain surrounding text"
        );
    }

    /// T-CIT-002: Multi-key \cite{a,b,c} produces 3 entries with the same
    /// group_id. Each entry has the correct cite_key.
    #[test]
    fn t_cit_002_multi_key_cite() {
        let tex = r"As shown \cite{fama1970,bollerslev1986,mandelbrot1963} here.";
        let cits = parse_citations(tex);

        assert_eq!(cits.len(), 3);
        assert_eq!(cits[0].cite_key, "fama1970");
        assert_eq!(cits[1].cite_key, "bollerslev1986");
        assert_eq!(cits[2].cite_key, "mandelbrot1963");

        // All three must share the same group_id.
        assert_eq!(cits[0].group_id, cits[1].group_id);
        assert_eq!(cits[1].group_id, cits[2].group_id);

        // All three share the same tex_context.
        assert_eq!(cits[0].tex_context, cits[1].tex_context);
    }

    /// T-CIT-003: All supported citation command variants are recognized.
    #[test]
    fn t_cit_003_all_cite_variants() {
        let tex = r"
\cite{a}
\citep{b}
\citet{c}
\autocite{d}
\parencite{e}
\textcite{f}
";
        let cits = parse_citations(tex);

        assert_eq!(cits.len(), 6);
        let keys: Vec<&str> = cits.iter().map(|c| c.cite_key.as_str()).collect();
        assert_eq!(keys, vec!["a", "b", "c", "d", "e", "f"]);

        // Each command is a separate group.
        let groups: Vec<usize> = cits.iter().map(|c| c.group_id).collect();
        assert_eq!(groups.len(), 6);
        // All group IDs are distinct.
        let unique: std::collections::HashSet<usize> = groups.into_iter().collect();
        assert_eq!(unique.len(), 6);
    }

    /// T-CIT-004: Commented lines (starting with %) are skipped.
    #[test]
    fn t_cit_004_commented_lines_skipped() {
        let tex = "Active \\cite{real}\n% \\cite{commented}\nAlso \\cite{real2}";
        let cits = parse_citations(tex);

        assert_eq!(cits.len(), 2);
        assert_eq!(cits[0].cite_key, "real");
        assert_eq!(cits[1].cite_key, "real2");
    }

    /// T-CIT-005: Section tracking sets section_title correctly. Citations
    /// after a \section{} heading inherit that section's title. A subsequent
    /// \subsection{} updates the section_title.
    #[test]
    fn t_cit_005_section_tracking() {
        let tex = r"\section{Introduction}
First citation \cite{intro1}.
\subsection{Background}
Second citation \cite{bg1}.
\section{Method}
Third citation \cite{method1}.
";
        let cits = parse_citations(tex);

        assert_eq!(cits.len(), 3);
        assert_eq!(cits[0].section_title.as_deref(), Some("Introduction"));
        assert_eq!(cits[1].section_title.as_deref(), Some("Background"));
        assert_eq!(cits[2].section_title.as_deref(), Some("Method"));
    }

    /// T-CIT-006: Anchor extraction captures up to 15 words from the same line.
    /// With the extended anchor context, all available words on the line are
    /// captured (up to the MAX_ANCHOR_WORDS limit).
    #[test]
    fn t_cit_006_anchor_extraction() {
        let tex = r"zeigt dass \cite{key1} die Maerkte efficient sind";
        let cits = parse_citations(tex);

        assert_eq!(cits.len(), 1);
        assert_eq!(cits[0].anchor_before, "zeigt dass");
        assert_eq!(cits[0].anchor_after, "die Maerkte efficient sind");
    }

    /// T-CIT-007: Citation with optional argument \cite[p.~42]{key}.
    /// The optional argument is consumed by the regex and the cite-key
    /// is correctly extracted.
    #[test]
    fn t_cit_007_optional_argument() {
        let tex = r"See \cite[p.~42]{fama1970} for details.";
        let cits = parse_citations(tex);

        assert_eq!(cits.len(), 1);
        assert_eq!(cits[0].cite_key, "fama1970");
    }

    /// T-CIT-008: Empty .tex content produces an empty vector.
    #[test]
    fn t_cit_008_empty_tex() {
        let cits = parse_citations("");
        assert!(cits.is_empty());
    }

    /// T-CIT-009: Citation at the start of a line (no anchor_before) and
    /// at the end of a line (no anchor_after).
    #[test]
    fn t_cit_009_citation_at_line_boundaries() {
        let tex = r"\cite{start_key}
text \cite{end_key}";
        let cits = parse_citations(tex);

        assert_eq!(cits.len(), 2);
        assert_eq!(cits[0].cite_key, "start_key");
        assert_eq!(cits[0].anchor_before, "");
        assert_eq!(cits[1].cite_key, "end_key");
        assert_eq!(cits[1].anchor_after, "");
    }

    /// T-CIT-010: Line numbers are 1-indexed and correct across multiple lines.
    #[test]
    fn t_cit_010_line_numbers() {
        let tex = "Line one\nLine two \\cite{a}\nLine three\nLine four \\cite{b}";
        let cits = parse_citations(tex);

        assert_eq!(cits.len(), 2);
        assert_eq!(cits[0].line, 2);
        assert_eq!(cits[1].line, 4);
    }

    /// T-CIT-089: Anchor extraction is limited to MAX_ANCHOR_WORDS (25) words.
    /// When more than 25 words precede a citation, only the last 25 are captured.
    #[test]
    fn t_cit_089_anchor_max_words_limit() {
        // Build a sequence of 30 words before the citation so the limit is exercised.
        let tex = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 \
                   w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 \
                   w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 \\cite{key1} after";
        let cits = parse_citations(tex);

        assert_eq!(cits.len(), 1);
        let before_words: Vec<&str> = cits[0].anchor_before.split_whitespace().collect();
        assert_eq!(
            before_words.len(),
            25,
            "anchor_before must contain at most 25 words"
        );
        assert_eq!(
            before_words[0], "w6",
            "anchor_before starts at the 25th-from-last word"
        );
        assert_eq!(
            before_words[24], "w30",
            "anchor_before ends with the word immediately before the citation"
        );
    }

    /// T-CIT-091: tex_context captures surrounding lines up to ~200 chars.
    /// A citation on a short line expands to include adjacent lines for context.
    #[test]
    fn t_cit_091_tex_context_captures_surrounding_lines() {
        let tex = "This is the first line of the document.\n\
                   This is the second line with more text.\n\
                   Here is the citation \\cite{key1} in context.\n\
                   This is the fourth line after the citation.\n\
                   This is the fifth line of the document.";
        let cits = parse_citations(tex);

        assert_eq!(cits.len(), 1);
        // The tex_context should include the citation's line and surrounding lines.
        assert!(
            cits[0].tex_context.contains("citation \\cite{key1}"),
            "tex_context must contain the citation line"
        );
        assert!(
            cits[0].tex_context.contains("second line"),
            "tex_context must include lines before the citation"
        );
        assert!(
            cits[0].tex_context.contains("fourth line"),
            "tex_context must include lines after the citation"
        );
    }

    /// T-CIT-092: tex_context handles a single-line document.
    #[test]
    fn t_cit_092_tex_context_single_line() {
        let tex = "Short line \\cite{key1} end.";
        let cits = parse_citations(tex);

        assert_eq!(cits.len(), 1);
        assert_eq!(
            cits[0].tex_context, tex,
            "single-line context is the line itself"
        );
    }

    /// T-CIT-093: extract_trailing_words returns empty string for empty input.
    #[test]
    fn t_cit_093_extract_trailing_words_empty() {
        assert_eq!(extract_trailing_words("", 15), "");
        assert_eq!(extract_trailing_words("   ", 15), "");
    }

    /// T-CIT-094: extract_leading_words strips trailing punctuation from the
    /// last word only.
    #[test]
    fn t_cit_094_extract_leading_words_strips_punctuation() {
        assert_eq!(
            extract_leading_words(" are significant.", 15),
            "are significant"
        );
        assert_eq!(extract_leading_words(" value, which", 15), "value, which");
    }

    /// T-CIT-095: extract_tex_context builds context from surrounding lines.
    #[test]
    fn t_cit_095_extract_tex_context_expansion() {
        let lines: Vec<&str> = vec!["line0", "line1", "line2", "line3", "line4"];
        // Center on line 2, target 20 chars -- should expand to include adjacent lines.
        let ctx = extract_tex_context(&lines, 2, 20);
        assert!(ctx.contains("line2"), "must contain the center line");
        // At least one adjacent line should be included.
        assert!(
            ctx.contains("line1") || ctx.contains("line3"),
            "must expand to adjacent lines"
        );
    }
}
