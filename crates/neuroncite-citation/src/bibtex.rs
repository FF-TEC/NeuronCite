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

// BibTeX file parser.
//
// Extracts bibliographic entries from a .bib file and returns them as a
// HashMap keyed by cite-key. Each entry contains author, title, year,
// abstract, and keywords fields. The parser handles brace-delimited
// ({...}) and quote-delimited ("...") field values, multi-line values,
// and preserves LaTeX accent commands as-is. Field names are matched
// case-insensitively.

use std::collections::HashMap;

use regex::Regex;
use std::sync::LazyLock;

use crate::types::BibEntry;

/// Regex pattern matching the start of a BibTeX entry. Captures:
/// - Group 1: the entry type (article, book, inproceedings, etc.)
/// - Group 2: the cite-key
///
/// The pattern matches @type{key, where type is any alphanumeric string
/// and key is everything up to the first comma or whitespace.
static ENTRY_START_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"@(\w+)\s*\{\s*([^,\s]+)\s*,").expect("BibTeX entry regex must compile")
});

/// Parses all entries from the given BibTeX content string.
///
/// Returns a HashMap mapping cite-keys to `BibEntry` structs. Entry types
/// like @comment, @string, and @preamble are ignored. Field names are
/// matched case-insensitively (e.g., "Author" and "author" are equivalent).
///
/// # Arguments
///
/// * `bib_content` - The full text content of a .bib file.
///
/// # Returns
///
/// A HashMap where keys are cite-keys (as they appear in the .bib file)
/// and values are `BibEntry` structs with the resolved fields.
pub fn parse_bibtex(bib_content: &str) -> HashMap<String, BibEntry> {
    let mut entries = HashMap::new();

    // Split the content into individual entry blocks by finding each
    // @type{key, ... } boundary.
    let entry_blocks = split_entries(bib_content);

    // Fields stored in named BibEntry fields. These are excluded from the
    // extra_fields HashMap to avoid duplication.
    const NAMED_FIELDS: &[&str] = &[
        "author", "title", "year", "abstract", "keywords", "url", "doi",
    ];

    for (entry_type, cite_key, body) in entry_blocks {
        let mut fields = parse_fields(&body);

        let author = fields.remove("author").unwrap_or_default();
        let title = fields.remove("title").unwrap_or_default();

        // Skip entries that have neither author nor title (likely @comment,
        // @string, or malformed entries).
        if author.is_empty() && title.is_empty() {
            continue;
        }

        let year = fields.remove("year");
        let bib_abstract = fields.remove("abstract");
        let keywords = fields.remove("keywords");
        let url = fields.remove("url");
        let doi = fields.remove("doi");

        // Remaining fields go into extra_fields. Remove named fields that
        // were already consumed above (redundant, but defensive).
        for &named in NAMED_FIELDS {
            fields.remove(named);
        }

        let entry = BibEntry {
            entry_type: entry_type.clone(),
            author,
            title,
            year,
            bib_abstract,
            keywords,
            url,
            doi,
            extra_fields: fields,
        };

        entries.insert(cite_key, entry);
    }

    entries
}

/// Splits BibTeX content into (entry_type, cite_key, body) triples. Each
/// triple contains the entry type (lowercase), cite-key, and the raw text
/// between the opening brace and the matching closing brace. Ignores
/// @comment, @string, and @preamble entries.
fn split_entries(content: &str) -> Vec<(String, String, String)> {
    let mut results = Vec::new();

    for caps in ENTRY_START_RE.captures_iter(content) {
        let entry_type = caps[1].to_lowercase();

        // Skip non-entry directives.
        if entry_type == "comment" || entry_type == "string" || entry_type == "preamble" {
            continue;
        }

        let cite_key = caps[2].to_string();

        // Find the body: everything from after the first comma to the matching
        // closing brace, tracking brace nesting depth.
        let match_end = caps.get(0).expect("full match must exist").end();
        let remaining = &content[match_end..];

        if let Some(body) = extract_brace_body(remaining) {
            results.push((entry_type, cite_key, body));
        }
    }

    results
}

/// Extracts the body text from the current position up to the matching
/// closing brace, tracking nested braces. Returns None if no matching
/// closing brace is found.
fn extract_brace_body(text: &str) -> Option<String> {
    let mut depth: i32 = 1; // The opening brace was already consumed by the regex.
    let mut end_pos = None;

    for (i, ch) in text.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end_pos = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }

    end_pos.map(|pos| text[..pos].to_string())
}

/// Parses field = value pairs from a BibTeX entry body. Field names are
/// normalized to lowercase. Values can be brace-delimited ({...}) or
/// quote-delimited ("..."), and may span multiple lines.
fn parse_fields(body: &str) -> HashMap<String, String> {
    let mut fields = HashMap::new();
    let mut pos = 0;

    while pos < body.len() {
        // Skip whitespace and commas between fields.
        while pos < body.len()
            && (body.as_bytes()[pos] == b' '
                || body.as_bytes()[pos] == b'\n'
                || body.as_bytes()[pos] == b'\r'
                || body.as_bytes()[pos] == b'\t'
                || body.as_bytes()[pos] == b',')
        {
            pos += 1;
        }

        if pos >= body.len() {
            break;
        }

        // Read field name (everything up to '=').
        let name_start = pos;
        while pos < body.len() && body.as_bytes()[pos] != b'=' {
            pos += 1;
        }

        if pos >= body.len() {
            break;
        }

        let field_name = body[name_start..pos].trim().to_lowercase();
        pos += 1; // Skip '='

        // Skip whitespace after '='.
        while pos < body.len() && body.as_bytes()[pos].is_ascii_whitespace() {
            pos += 1;
        }

        if pos >= body.len() {
            break;
        }

        // Read field value: brace-delimited or quote-delimited.
        let value;
        let delimiter = body.as_bytes()[pos];

        if delimiter == b'{' {
            pos += 1;
            let val_start = pos;
            let mut depth = 1;
            while pos < body.len() && depth > 0 {
                match body.as_bytes()[pos] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                if depth > 0 {
                    pos += 1;
                }
            }
            value = body[val_start..pos].to_string();
            pos += 1; // Skip closing '}'
        } else if delimiter == b'"' {
            pos += 1;
            let val_start = pos;
            while pos < body.len() && body.as_bytes()[pos] != b'"' {
                pos += 1;
            }
            value = body[val_start..pos].to_string();
            if pos < body.len() {
                pos += 1; // Skip closing '"'
            }
        } else {
            // Undelimited value (e.g., year = 1970): read until comma or end.
            let val_start = pos;
            while pos < body.len() && body.as_bytes()[pos] != b',' && body.as_bytes()[pos] != b'}' {
                pos += 1;
            }
            value = body[val_start..pos].trim().to_string();
        }

        // Normalize whitespace in the parsed value: collapse consecutive
        // whitespace characters into single spaces and trim.
        let normalized = normalize_whitespace(&value);

        if !field_name.is_empty() && !normalized.is_empty() {
            fields.insert(field_name, normalized);
        }
    }

    fields
}

/// Collapses consecutive whitespace characters (spaces, tabs, newlines)
/// into single spaces and trims leading/trailing whitespace.
fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-CIT-011: Single BibTeX entry with all fields populated.
    #[test]
    fn t_cit_011_single_entry_all_fields() {
        let bib = r#"
@article{fama1970,
    author = {Fama, Eugene F.},
    title = {Efficient Capital Markets: A Review of Theory and Empirical Work},
    year = {1970},
    abstract = {This paper reviews the theoretical and empirical literature on the efficient markets hypothesis.},
    keywords = {EMH, efficient markets, stock prices}
}
"#;
        let entries = parse_bibtex(bib);

        assert_eq!(entries.len(), 1);
        let entry = entries.get("fama1970").expect("fama1970 must exist");
        assert_eq!(entry.author, "Fama, Eugene F.");
        assert_eq!(
            entry.title,
            "Efficient Capital Markets: A Review of Theory and Empirical Work"
        );
        assert_eq!(entry.year.as_deref(), Some("1970"));
        assert!(entry.bib_abstract.is_some());
        assert!(entry.keywords.is_some());
        assert!(entry.keywords.as_ref().unwrap().contains("EMH"));
    }

    /// T-CIT-012: Entry without abstract or keywords fields. Those fields
    /// must be None in the parsed BibEntry.
    #[test]
    fn t_cit_012_entry_without_optional_fields() {
        let bib = r#"
@article{roll1984,
    author = {Roll, Richard},
    title = {A Simple Implicit Measure of the Effective Bid-Ask Spread},
    year = {1984}
}
"#;
        let entries = parse_bibtex(bib);
        let entry = entries.get("roll1984").expect("roll1984 must exist");
        assert!(entry.bib_abstract.is_none());
        assert!(entry.keywords.is_none());
    }

    /// T-CIT-013: Multiple entries parsed correctly.
    #[test]
    fn t_cit_013_multiple_entries() {
        let bib = r#"
@article{a1, author = {A}, title = {Title A}, year = {2000}}
@book{b2, author = {B}, title = {Title B}, year = {2010}}
@inproceedings{c3, author = {C}, title = {Title C}, year = {2020}}
"#;
        let entries = parse_bibtex(bib);
        assert_eq!(entries.len(), 3);
        assert!(entries.contains_key("a1"));
        assert!(entries.contains_key("b2"));
        assert!(entries.contains_key("c3"));
    }

    /// T-CIT-014: Brace-delimited and quote-delimited values are both
    /// correctly parsed.
    #[test]
    fn t_cit_014_brace_vs_quote_delimiters() {
        let bib = r#"
@article{test1,
    author = {Brace Author},
    title = "Quote Title",
    year = {2000}
}
"#;
        let entries = parse_bibtex(bib);
        let entry = entries.get("test1").expect("test1 must exist");
        assert_eq!(entry.author, "Brace Author");
        assert_eq!(entry.title, "Quote Title");
    }

    /// T-CIT-015: Multi-line field values are collapsed into single-line
    /// strings with normalized whitespace.
    #[test]
    fn t_cit_015_multiline_field_values() {
        let bib = r#"
@article{multi,
    author = {Author Name},
    title = {This is a very long title
             that spans multiple lines
             in the BibTeX file},
    year = {2000}
}
"#;
        let entries = parse_bibtex(bib);
        let entry = entries.get("multi").expect("multi must exist");
        assert_eq!(
            entry.title,
            "This is a very long title that spans multiple lines in the BibTeX file"
        );
    }

    /// T-CIT-016: Author names with LaTeX accent commands are preserved
    /// as-is without interpretation.
    #[test]
    fn t_cit_016_latex_accents_preserved() {
        let bib = r#"
@article{accent,
    author = {M{\"u}ller, Hans and Caf{\'{e}}, Jean},
    title = {Test},
    year = {2000}
}
"#;
        let entries = parse_bibtex(bib);
        let entry = entries.get("accent").expect("accent must exist");
        // The raw LaTeX accent commands are preserved in the author string.
        // The backslash-quote-u sequence (\"u) inside braces is a LaTeX umlaut.
        assert!(
            entry.author.contains(r#"M{\"u}ller"#),
            "author field must contain LaTeX accent command, got: {}",
            entry.author
        );
    }

    /// T-CIT-017: Empty .bib content produces an empty HashMap.
    #[test]
    fn t_cit_017_empty_bib() {
        let entries = parse_bibtex("");
        assert!(entries.is_empty());
    }

    /// T-CIT-018: Case-insensitive field names. "Author" and "author"
    /// produce the same result.
    #[test]
    fn t_cit_018_case_insensitive_fields() {
        let bib = r#"
@article{case1,
    Author = {Case Author},
    TITLE = {Case Title},
    Year = {2000}
}
"#;
        let entries = parse_bibtex(bib);
        let entry = entries.get("case1").expect("case1 must exist");
        assert_eq!(entry.author, "Case Author");
        assert_eq!(entry.title, "Case Title");
        assert_eq!(entry.year.as_deref(), Some("2000"));
    }

    /// T-CIT-100: Entry with url field. The url field is extracted and
    /// stored in the BibEntry struct for source acquisition.
    #[test]
    fn t_cit_019_url_field_extracted() {
        let bib = r#"
@article{fama1970,
    author = {Fama, Eugene F.},
    title = {Efficient Capital Markets},
    year = {1970},
    url = {https://www.jstor.org/stable/2325486}
}
"#;
        let entries = parse_bibtex(bib);
        let entry = entries.get("fama1970").expect("fama1970 must exist");
        assert_eq!(
            entry.url.as_deref(),
            Some("https://www.jstor.org/stable/2325486")
        );
    }

    /// T-CIT-101: Entry with doi field. The doi field is extracted for
    /// resolving source URLs via https://doi.org/{doi}.
    #[test]
    fn t_cit_020_doi_field_extracted() {
        let bib = r#"
@article{black1973,
    author = {Black, Fischer and Scholes, Myron},
    title = {The Pricing of Options and Corporate Liabilities},
    year = {1973},
    doi = {10.1086/260062}
}
"#;
        let entries = parse_bibtex(bib);
        let entry = entries.get("black1973").expect("black1973 must exist");
        assert_eq!(entry.doi.as_deref(), Some("10.1086/260062"));
    }

    /// T-CIT-102: Entry with both url and doi fields. Both are extracted
    /// independently and can coexist in the same BibEntry.
    #[test]
    fn t_cit_021_url_and_doi_coexist() {
        let bib = r#"
@article{mandelbrot1963,
    author = {Mandelbrot, Benoit B.},
    title = {The Variation of Certain Speculative Prices},
    year = {1963},
    url = {https://example.com/mandelbrot1963.pdf},
    doi = {10.1086/294632}
}
"#;
        let entries = parse_bibtex(bib);
        let entry = entries
            .get("mandelbrot1963")
            .expect("mandelbrot1963 must exist");
        assert_eq!(
            entry.url.as_deref(),
            Some("https://example.com/mandelbrot1963.pdf")
        );
        assert_eq!(entry.doi.as_deref(), Some("10.1086/294632"));
    }

    /// T-CIT-103: Entry without url and doi fields. Both fields are None
    /// when absent from the BibTeX source.
    #[test]
    fn t_cit_022_missing_url_and_doi() {
        let bib = r#"
@article{roll1984,
    author = {Roll, Richard},
    title = {A Simple Implicit Measure},
    year = {1984}
}
"#;
        let entries = parse_bibtex(bib);
        let entry = entries.get("roll1984").expect("roll1984 must exist");
        assert!(entry.url.is_none(), "url must be None when absent");
        assert!(entry.doi.is_none(), "doi must be None when absent");
    }

    /// T-CIT-104: URL field with quote-delimited value. The parser handles
    /// both brace-delimited and quote-delimited url values.
    #[test]
    fn t_cit_023_url_quote_delimited() {
        let bib = r#"
@article{test_url,
    author = {Test Author},
    title = {Test Title},
    year = {2020},
    url = "https://arxiv.org/abs/2001.00001"
}
"#;
        let entries = parse_bibtex(bib);
        let entry = entries.get("test_url").expect("test_url must exist");
        assert_eq!(
            entry.url.as_deref(),
            Some("https://arxiv.org/abs/2001.00001")
        );
    }

    /// T-CIT-024: Case-insensitive url and doi field names. "URL" and "DOI"
    /// in uppercase are recognized the same as "url" and "doi".
    #[test]
    fn t_cit_024_url_doi_case_insensitive() {
        let bib = r#"
@article{upper_case,
    author = {Author},
    title = {Title},
    year = {2000},
    URL = {https://example.com/paper.pdf},
    DOI = {10.1234/test}
}
"#;
        let entries = parse_bibtex(bib);
        let entry = entries.get("upper_case").expect("upper_case must exist");
        assert_eq!(
            entry.url.as_deref(),
            Some("https://example.com/paper.pdf"),
            "URL (uppercase) must be parsed as url"
        );
        assert_eq!(
            entry.doi.as_deref(),
            Some("10.1234/test"),
            "DOI (uppercase) must be parsed as doi"
        );
    }

    /// T-CIT-025: The first test (t_cit_011) with all fields now includes
    /// url and doi as None since that entry does not have them.
    #[test]
    fn t_cit_025_all_fields_entry_url_doi_none() {
        let bib = r#"
@article{fama1970,
    author = {Fama, Eugene F.},
    title = {Efficient Capital Markets},
    year = {1970},
    abstract = {Review paper.},
    keywords = {EMH}
}
"#;
        let entries = parse_bibtex(bib);
        let entry = entries.get("fama1970").expect("fama1970 must exist");
        assert!(
            entry.url.is_none(),
            "url must be None when not in bib entry"
        );
        assert!(
            entry.doi.is_none(),
            "doi must be None when not in bib entry"
        );
    }
}
