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

//! Static abbreviation whitelist for sentence boundary suppression.
//!
//! Contains a fixed set of abbreviations commonly found in scientific and
//! academic literature. When the sentence segmentation algorithm encounters
//! a period that belongs to one of these abbreviations, it suppresses the
//! sentence break and keeps the abbreviation within the current sentence.
//!
//! All matching is case-sensitive because the abbreviations have specific
//! capitalization conventions in academic writing (e.g., "Fig." is always
//! capitalized, "i.e." is always lowercase).

/// The complete list of abbreviations recognized by the sentence boundary
/// suppression logic. Each entry includes the trailing period as part of the
/// abbreviation pattern.
const ABBREVIATIONS: &[&str] = &[
    "et al.", "Fig.", "Eq.", "Dr.", "Prof.", "Inc.", "Ltd.", "vs.", "i.e.", "e.g.", "cf.",
    "approx.", "Vol.", "No.", "pp.", "Sect.",
];

/// Checks whether the given word (including any trailing period) matches
/// one of the known academic abbreviations.
///
/// The comparison is case-sensitive. The input `word` should include the
/// trailing period if present (e.g., pass "Fig." not "Fig").
///
/// # Arguments
///
/// * `word` - A whitespace-delimited token from the text, potentially
///   including trailing punctuation.
///
/// # Returns
///
/// `true` if the word exactly matches one of the entries in the
/// abbreviation whitelist, `false` otherwise.
///
/// # Examples
///
/// ```
/// use neuroncite_chunk::abbreviations::is_abbreviation;
///
/// assert!(is_abbreviation("et al."));
/// assert!(is_abbreviation("Fig."));
/// assert!(!is_abbreviation("fig."));  // case-sensitive
/// assert!(!is_abbreviation("Hello."));
/// ```
#[must_use]
pub fn is_abbreviation(word: &str) -> bool {
    ABBREVIATIONS.contains(&word)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that all listed abbreviations are recognized.
    #[test]
    fn all_abbreviations_recognized() {
        let expected = vec![
            "et al.", "Fig.", "Eq.", "Dr.", "Prof.", "Inc.", "Ltd.", "vs.", "i.e.", "e.g.", "cf.",
            "approx.", "Vol.", "No.", "pp.", "Sect.",
        ];
        for abbr in expected {
            assert!(
                is_abbreviation(abbr),
                "abbreviation '{abbr}' should be recognized"
            );
        }
    }

    /// Verifies that non-abbreviation words are rejected.
    #[test]
    fn non_abbreviations_rejected() {
        assert!(!is_abbreviation("Hello."));
        assert!(!is_abbreviation("the"));
        assert!(!is_abbreviation("end."));
    }

    /// Verifies that matching is case-sensitive (lowercase "fig." is not
    /// recognized, only "Fig." is in the whitelist).
    #[test]
    fn case_sensitive_matching() {
        assert!(is_abbreviation("Fig."));
        assert!(!is_abbreviation("fig."));
        assert!(is_abbreviation("Dr."));
        assert!(!is_abbreviation("dr."));
    }
}
