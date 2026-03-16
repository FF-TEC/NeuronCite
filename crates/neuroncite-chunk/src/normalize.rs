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

//! Deterministic text preprocessing applied before any chunking strategy.
//!
//! Performs four transformations in sequence:
//! 1. Whitespace collapsing: runs of spaces, tabs, and form feeds are reduced
//!    to a single space character.
//! 2. Line-break hyphenation repair: a hyphen immediately followed by a newline
//!    is removed, joining the two halves of a hyphenated word that was broken
//!    across lines during PDF text extraction.
//! 3. PDF diacritical mark repair: spacing diacritics (U+00A8 DIAERESIS,
//!    U+00B4 ACUTE ACCENT, U+0060 GRAVE ACCENT, U+02C6 CIRCUMFLEX,
//!    U+02DC TILDE) that follow a letter are replaced by their Unicode
//!    combining equivalents. PDF text extraction backends frequently produce
//!    these spacing variants instead of combining marks, causing German
//!    umlauts and French accents to render as separate characters (e.g.,
//!    "fu¨r" instead of "für"). Latin ligatures (ff, fi, fl, ffi, ffl) are
//!    decomposed into their constituent ASCII letters.
//! 4. Unicode NFC normalization: combining character sequences are converted
//!    to their precomposed canonical forms (e.g., e + combining acute becomes
//!    the single codepoint precomposed e-acute).
//!
//! This preprocessing step ensures that chunking strategies operate on a
//! consistent text representation regardless of the PDF extraction backend
//! that produced the input.

use unicode_normalization::UnicodeNormalization;

/// Applies deterministic text normalization to the input string.
///
/// Three transformations are applied in sequence:
/// 1. A fused pass that repairs line-break hyphenation and collapses
///    horizontal whitespace in a single character-level traversal.
/// 2. PDF diacritical mark repair: spacing diacritics are converted to
///    combining equivalents and Latin ligatures are decomposed. This fixes
///    the common PDF extraction artifact where German umlauts appear as
///    separate characters (e.g., "fu¨r" instead of "für").
/// 3. Unicode NFC normalization to convert combining character sequences
///    to their precomposed canonical forms.
///
/// # Arguments
///
/// * `text` - The raw extracted text from one or more PDF pages.
///
/// # Returns
///
/// A normalized string with collapsed whitespace, repaired hyphenation,
/// repaired diacritical marks, decomposed ligatures, and NFC-normalized
/// Unicode.
#[must_use]
pub fn normalize_text(text: &str) -> String {
    // Pass 1: Fused hyphenation repair and whitespace collapsing. Performs
    // both transformations in a single traversal with one allocation.
    let cleaned = repair_hyphenation_and_collapse_whitespace(text);

    // Pass 2+3: Convert spacing diacritics to combining equivalents, decompose
    // Latin ligatures, and apply NFC normalization in a fixpoint loop.
    //
    // PDF extraction backends (pdf-extract, pdfium, lopdf) frequently produce
    // spacing diacritics (U+00A8 DIAERESIS) instead of combining marks
    // (U+0308 COMBINING DIAERESIS) when the PDF font uses separate glyphs for
    // the base letter and the diacritical mark. Without this conversion, NFC
    // normalization cannot compose "u" + "¨" into "ü" because U+00A8 is a
    // standalone character, not a combining mark.
    //
    // The loop is necessary for idempotency: after NFC composes a base letter
    // and a combining mark into a precomposed character (e.g., E + U+0300 ->
    // U+00C8 "È"), any subsequent spacing diacritic now follows a letter and
    // must also be converted. Without the loop, a second call to normalize_text
    // on the output would produce a different result. The loop converges in at
    // most 2 iterations because each pass reduces the count of unconverted
    // spacing diacritics by at least one.
    let mut normalized: String = {
        let repaired = repair_pdf_diacritics_and_ligatures(&cleaned);
        repaired.nfc().collect()
    };

    loop {
        let repaired = repair_pdf_diacritics_and_ligatures(&normalized);
        let next: String = repaired.nfc().collect();
        if next == normalized {
            break;
        }
        normalized = next;
    }

    normalized
}

/// Fuses hyphenation repair and horizontal whitespace collapsing into a single
/// character-level traversal. This eliminates one intermediate String allocation
/// and one full traversal compared to running the two operations sequentially.
///
/// Hyphenation repair: a hyphen '-' immediately followed by '\n' is removed
/// (both characters skipped), joining word halves split across PDF lines.
///
/// Whitespace collapsing: consecutive horizontal whitespace characters (spaces,
/// tabs, form feeds) are reduced to a single space. Newlines are preserved
/// because they serve as inter-page separators in the concatenated document text.
fn repair_hyphenation_and_collapse_whitespace(text: &str) -> String {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut result = String::with_capacity(len);
    let mut in_whitespace_run = false;
    let mut i = 0;

    // Iterate byte-by-byte for the ASCII-range characters that the two
    // transformations operate on (hyphen, newline, space, tab, form feed are
    // all single-byte in UTF-8). Non-ASCII bytes are handled by decoding the
    // full multi-byte character.
    while i < len {
        let b = bytes[i];

        // Hyphenation repair: skip '-' + '\n' pairs.
        if b == b'-' && i + 1 < len && bytes[i + 1] == b'\n' {
            in_whitespace_run = false;
            i += 2;
            continue;
        }

        // Horizontal whitespace collapsing: space, tab, form feed.
        if b == b' ' || b == b'\t' || b == b'\x0C' {
            if !in_whitespace_run {
                result.push(' ');
                in_whitespace_run = true;
            }
            i += 1;
            continue;
        }

        in_whitespace_run = false;

        // For ASCII characters (single byte), push directly.
        if b.is_ascii() {
            result.push(b as char);
            i += 1;
        } else {
            // Decode the multi-byte UTF-8 character starting at position i.
            let remaining = &text[i..];
            let ch = remaining
                .chars()
                .next()
                .expect("valid UTF-8 at byte boundary");
            result.push(ch);
            i += ch.len_utf8();
        }
    }

    result
}

/// Mapping from spacing diacritical marks to their Unicode combining equivalents.
/// PDF text extraction backends produce these spacing variants when the PDF font
/// encodes diacritical marks as separate glyphs positioned above or below the
/// base letter. The spacing variants are standalone characters that do not
/// participate in Unicode canonical composition (NFC), so they must be replaced
/// by their combining counterparts before NFC can produce precomposed forms.
///
/// Each entry maps (spacing_codepoint) -> (combining_codepoint).
const SPACING_TO_COMBINING: &[(char, char)] = &[
    ('\u{00A8}', '\u{0308}'), // DIAERESIS -> COMBINING DIAERESIS (a/o/u -> ae/oe/ue)
    ('\u{00B4}', '\u{0301}'), // ACUTE ACCENT -> COMBINING ACUTE ACCENT (e -> e-acute)
    ('\u{0060}', '\u{0300}'), // GRAVE ACCENT -> COMBINING GRAVE ACCENT (a -> a-grave)
    ('\u{00B8}', '\u{0327}'), // CEDILLA -> COMBINING CEDILLA (c -> c-cedilla)
    ('\u{02C6}', '\u{0302}'), // MODIFIER LETTER CIRCUMFLEX -> COMBINING CIRCUMFLEX
    ('\u{02DC}', '\u{0303}'), // SMALL TILDE -> COMBINING TILDE (n -> n-tilde)
];

/// Mapping from Latin ligature codepoints to their decomposed ASCII letter
/// sequences. PDF fonts that use ligature glyphs (ff, fi, fl, ffi, ffl) map
/// to Unicode ligature codepoints in the Alphabetic Presentation Forms block
/// (U+FB00..U+FB04). These ligatures cause search mismatches and display
/// inconsistencies, so they are decomposed into their constituent letters.
const LIGATURE_DECOMPOSITIONS: &[(char, &str)] = &[
    ('\u{FB00}', "ff"),  // LATIN SMALL LIGATURE FF
    ('\u{FB01}', "fi"),  // LATIN SMALL LIGATURE FI
    ('\u{FB02}', "fl"),  // LATIN SMALL LIGATURE FL
    ('\u{FB03}', "ffi"), // LATIN SMALL LIGATURE FFI
    ('\u{FB04}', "ffl"), // LATIN SMALL LIGATURE FFL
    ('\u{FB05}', "st"),  // LATIN SMALL LIGATURE LONG S T
    ('\u{FB06}', "st"),  // LATIN SMALL LIGATURE ST
];

/// Converts spacing diacritical marks that follow a letter to their combining
/// equivalents, and decomposes Latin ligatures into ASCII letter sequences.
///
/// Spacing diacritics are only converted when they immediately follow an
/// alphabetic character, because standalone diacritics in non-letter contexts
/// (e.g., in mathematical notation or as quotation marks) should be preserved.
///
/// Ligatures are unconditionally decomposed because the ligature codepoints
/// U+FB00..U+FB06 have no semantic distinction from their component letters
/// in text search and storage contexts.
fn repair_pdf_diacritics_and_ligatures(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_letter = false;

    for ch in text.chars() {
        // Check if the current character is a spacing diacritical mark that
        // should be converted to its combining equivalent. The conversion
        // only applies when the preceding character was a letter (the base
        // character for the diacritic).
        if prev_was_letter
            && let Some(&(_, combining)) = SPACING_TO_COMBINING.iter().find(|&&(s, _)| s == ch)
        {
            result.push(combining);
            prev_was_letter = false;
            continue;
        }

        // Check if the current character is a Latin ligature that should be
        // decomposed into its constituent ASCII letters.
        if let Some(&(_, decomposed)) = LIGATURE_DECOMPOSITIONS.iter().find(|&&(lig, _)| lig == ch)
        {
            result.push_str(decomposed);
            // The last character of any ligature decomposition is a letter.
            prev_was_letter = true;
            continue;
        }

        prev_was_letter = ch.is_alphabetic();
        result.push(ch);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T-CHK-013: Multiple spaces, tabs, and form feeds are collapsed to a
    /// single space character.
    #[test]
    fn t_chk_013_whitespace_collapsing() {
        let input = "hello   world\t\ttab\x0C\x0Cfeed  mixed \t end";
        let result = normalize_text(input);
        assert_eq!(result, "hello world tab feed mixed end");
    }

    /// T-CHK-014: A word hyphenated at a line break ("exam-\nple") is joined
    /// to "example" in the normalized output.
    #[test]
    fn t_chk_014_hyphenation_repair() {
        let input = "exam-\nple of hyphen-\nation";
        let result = normalize_text(input);
        assert_eq!(result, "example of hyphenation");
    }

    /// T-CHK-015: The decomposed sequence "e\u{0301}" (e + combining acute
    /// accent) is normalized to the precomposed form "\u{00e9}" (e-acute).
    #[test]
    fn t_chk_015_unicode_nfc() {
        let input = "e\u{0301}";
        let result = normalize_text(input);
        assert_eq!(result, "\u{00e9}");
    }

    /// Verifies that newlines are preserved during whitespace collapsing,
    /// since they serve as page separators.
    #[test]
    fn newlines_preserved() {
        let input = "page one\npage two\npage three";
        let result = normalize_text(input);
        assert_eq!(result, "page one\npage two\npage three");
    }

    /// Verifies that a hyphen not immediately before a newline is preserved.
    #[test]
    fn non_linebreak_hyphens_preserved() {
        let input = "state-of-the-art method";
        let result = normalize_text(input);
        assert_eq!(result, "state-of-the-art method");
    }

    /// Regression test for issue #19: the fused hyphenation+whitespace pass
    /// must handle interleaved patterns correctly. This tests a sequence
    /// where a hyphenation repair immediately precedes whitespace collapsing.
    #[test]
    fn fused_pass_handles_interleaved_patterns() {
        // "re-\n    search" -> hyphenation repair joins "re" + "search",
        // whitespace collapsing is not needed because the hyphen-newline
        // consumed both characters.
        let input = "re-\nsearch  is   great";
        let result = normalize_text(input);
        assert_eq!(result, "research is great");
    }

    /// Regression test: fused pass preserves multi-byte UTF-8 characters
    /// when they appear adjacent to hyphenation or whitespace patterns.
    #[test]
    fn fused_pass_preserves_multibyte_utf8() {
        // German umlaut surrounded by whitespace and hyphenation patterns.
        let input = "Pr\u{00FC}f-\nung   der   \u{00C4}nderung";
        let result = normalize_text(input);
        assert_eq!(result, "Pr\u{00FC}fung der \u{00C4}nderung");
    }

    /// Regression test: a standalone hyphen at end-of-string is preserved.
    #[test]
    fn trailing_hyphen_without_newline_preserved() {
        let input = "trailing-";
        let result = normalize_text(input);
        assert_eq!(result, "trailing-");
    }

    // -----------------------------------------------------------------------
    // PDF diacritical mark repair tests
    // -----------------------------------------------------------------------

    /// T-CHK-030: Spacing diaeresis (U+00A8) after a vowel is converted to a
    /// combining diaeresis, then NFC composes "u" + combining diaeresis into
    /// the precomposed "ue" (U+00FC). This is the primary fix for German umlauts
    /// extracted from PDFs with separate diacritical mark glyphs.
    #[test]
    fn t_chk_030_spacing_diaeresis_to_umlaut() {
        // PDF extraction produces "fu\u{00A8}r" instead of "f\u{00FC}r"
        let input = "fu\u{00A8}r Gewinde ohne besondere Anforderungen";
        let result = normalize_text(input);
        assert_eq!(result, "f\u{00FC}r Gewinde ohne besondere Anforderungen");
    }

    /// T-CHK-031: All three German umlaut vowels (a, o, u) are correctly
    /// composed from spacing diaeresis sequences in a single text block.
    #[test]
    fn t_chk_031_all_german_umlauts_from_spacing_diaeresis() {
        let input =
            "A\u{00A8}nderung der Pru\u{00A8}fung fu\u{00A8}r ho\u{00A8}here Qualita\u{00A8}t";
        let result = normalize_text(input);
        assert_eq!(
            result,
            "\u{00C4}nderung der Pr\u{00FC}fung f\u{00FC}r h\u{00F6}here Qualit\u{00E4}t"
        );
    }

    /// T-CHK-032: Spacing diaeresis that does not follow a letter is preserved
    /// as-is, because it may be intentional punctuation or mathematical notation.
    #[test]
    fn t_chk_032_standalone_diaeresis_preserved() {
        // Diaeresis at start of string (no preceding letter)
        let input = "\u{00A8} test";
        let result = normalize_text(input);
        assert_eq!(result, "\u{00A8} test");
    }

    /// T-CHK-033: Spacing diaeresis after a digit or punctuation is preserved,
    /// since only letter + diaeresis sequences represent decomposed umlauts.
    #[test]
    fn t_chk_033_diaeresis_after_non_letter_preserved() {
        let input = "3\u{00A8} test";
        let result = normalize_text(input);
        assert_eq!(result, "3\u{00A8} test");
    }

    /// T-CHK-034: Latin ligature ff (U+FB00) is decomposed into "ff".
    #[test]
    fn t_chk_034_ligature_ff_decomposed() {
        let input = "Werksto\u{FB00}e und Festigkeiten";
        let result = normalize_text(input);
        assert_eq!(result, "Werkstoffe und Festigkeiten");
    }

    /// T-CHK-035: Latin ligatures fi (U+FB01) and fl (U+FB02) are decomposed.
    #[test]
    fn t_chk_035_ligature_fi_fl_decomposed() {
        let input = "speci\u{FB01}c classi\u{FB01}cation of \u{FB02}uid";
        let result = normalize_text(input);
        assert_eq!(result, "specific classification of fluid");
    }

    /// T-CHK-036: Latin ligatures ffi (U+FB03) and ffl (U+FB04) are decomposed.
    #[test]
    fn t_chk_036_ligature_ffi_ffl_decomposed() {
        let input = "o\u{FB03}cial a\u{FB04}uent";
        let result = normalize_text(input);
        assert_eq!(result, "official affluent");
    }

    /// T-CHK-037: Combined diacritical mark repair and ligature decomposition
    /// in a single text block. Tests the full normalization pipeline with a
    /// realistic German technical text containing both artifact types.
    #[test]
    fn t_chk_037_combined_diacritics_and_ligatures() {
        let input =
            "Pru\u{00A8}fung der Werksto\u{FB00}e fu\u{00A8}r spezi\u{FB01}sche Anwendungen";
        let result = normalize_text(input);
        assert_eq!(
            result,
            "Pr\u{00FC}fung der Werkstoffe f\u{00FC}r spezifische Anwendungen"
        );
    }

    /// T-CHK-038: Spacing acute accent (U+00B4) after a vowel is converted
    /// to combining acute, then NFC composes the precomposed form.
    #[test]
    fn t_chk_038_spacing_acute_to_composed() {
        // French e-acute from spacing acute: "e\u{00B4}" -> "e\u{0301}" -> NFC "e-acute"
        let input = "re\u{00B4}sume\u{00B4}";
        let result = normalize_text(input);
        assert_eq!(result, "r\u{00E9}sum\u{00E9}");
    }

    /// T-CHK-039: Spacing cedilla (U+00B8) after 'c' is converted to
    /// combining cedilla, then NFC composes c-cedilla (U+00E7).
    #[test]
    fn t_chk_039_spacing_cedilla_to_composed() {
        let input = "franc\u{00B8}ais";
        let result = normalize_text(input);
        assert_eq!(result, "fran\u{00E7}ais");
    }

    /// T-CHK-040: Full pipeline test with hyphenation repair, whitespace
    /// collapsing, diacritical mark repair, ligature decomposition, and NFC
    /// normalization applied to realistic German PDF extraction output.
    #[test]
    fn t_chk_040_full_pipeline_german_pdf() {
        let input = "fu\u{00A8}r   Gewinde  ohne   beson-\ndere  Anforderungen.  \
                     Die  Werksto\u{FB00}pru\u{00A8}fung   ist  \
                     fu\u{00A8}r  die  Gu\u{00A8}te  der  Schrauben   erforderlich.";
        let result = normalize_text(input);
        assert_eq!(
            result,
            "f\u{00FC}r Gewinde ohne besondere Anforderungen. \
             Die Werkstoffpr\u{00FC}fung ist \
             f\u{00FC}r die G\u{00FC}te der Schrauben erforderlich."
        );
    }
}
