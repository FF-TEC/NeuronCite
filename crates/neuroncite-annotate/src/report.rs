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

// Annotation report generation and serialization.
//
// After the pipeline processes all PDFs, this module produces a structured
// JSON report summarizing the outcome: how many PDFs were annotated, which
// quotes were found vs. not found, verification results, and any errors.
// The report is written to the output directory as `annotate_report.json`.

use serde::Serialize;

/// Top-level annotation report containing the pipeline configuration,
/// a summary of results, per-PDF details, and unmatched input rows.
#[derive(Debug, Clone, Serialize)]
pub struct AnnotationReport {
    /// ISO 8601 timestamp when the pipeline finished.
    pub timestamp: String,
    /// Absolute path to the source PDF directory.
    pub source_directory: String,
    /// Absolute path to the output directory.
    pub output_directory: String,
    /// Aggregate counts across all processed PDFs.
    pub summary: ReportSummary,
    /// Per-PDF processing results.
    pub pdfs: Vec<PdfReport>,
    /// Input rows that could not be matched to any PDF file.
    pub unmatched_inputs: Vec<UnmatchedInput>,
}

/// Aggregate counts for the annotation run.
#[derive(Debug, Clone, Serialize)]
pub struct ReportSummary {
    /// Total number of distinct PDFs that had at least one assigned quote.
    pub total_pdfs: usize,
    /// PDFs where all quotes were found and annotated.
    pub successful: usize,
    /// PDFs where some quotes were found and some were not.
    pub partial: usize,
    /// PDFs where no quotes were found or an error prevented processing.
    pub failed: usize,
    /// Total number of quotes across all input rows.
    pub total_quotes: usize,
    /// Quotes that were located and annotated in the PDFs.
    pub quotes_matched: usize,
    /// Quotes that could not be located after all pipeline stages.
    pub quotes_not_found: usize,
}

/// Processing result for a single PDF file.
#[derive(Debug, Clone, Serialize)]
pub struct PdfReport {
    /// Original filename (without directory path).
    pub filename: String,
    /// Full path to the source PDF.
    pub source_path: String,
    /// Full path to the annotated output PDF (if saved).
    pub output_path: Option<String>,
    /// Overall status: "success", "partial", or "failed".
    pub status: String,
    /// Per-quote results within this PDF.
    pub quotes: Vec<QuoteReport>,
    /// Error message if the entire PDF failed to process.
    pub error: Option<String>,
}

/// Processing result for a single quote within a PDF.
#[derive(Debug, Clone, Serialize)]
pub struct QuoteReport {
    /// First 80 characters of the quote text.
    pub quote_excerpt: String,
    /// Status: "matched", "not_found", or "error".
    pub status: String,
    /// Which pipeline stage found the match (exact, normalized, fuzzy,
    /// fallback_extract, ocr).
    pub match_method: Option<String>,
    /// 1-indexed page number where the quote was found.
    pub page: Option<usize>,
    /// Number of characters in the matched region with valid bounding boxes.
    pub chars_matched: Option<usize>,
    /// Total characters in the quote.
    pub chars_total: Option<usize>,
    /// Whether the pre-check verification passed.
    pub pre_check_passed: Option<bool>,
    /// Whether the post-check verification passed (removed in v3, always None).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub post_check_passed: Option<bool>,
    /// Diagnostic messages describing which pipeline stages were attempted
    /// and why they failed. Populated when status is "not_found". Each entry
    /// contains a stage name followed by the failure reason, e.g.
    /// "fuzzy: best=42% on page 7 < 90% threshold".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stages_tried: Option<Vec<String>>,
    /// Fuzzy match similarity score (0.0 to 1.0) when match_method is "fuzzy".
    /// Indicates how closely the located text matched the original quote.
    /// `None` for non-fuzzy match methods or when the quote was not found.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fuzzy_score: Option<f64>,
}

/// An input row that could not be matched to any PDF file in the source
/// directory.
#[derive(Debug, Clone, Serialize)]
pub struct UnmatchedInput {
    /// Title from the input row.
    pub title: String,
    /// Author from the input row.
    pub author: String,
    /// Reason the row could not be matched.
    pub error: String,
}

impl AnnotationReport {
    /// Computes the summary counts from the per-PDF reports and unmatched
    /// inputs. Call this after all PdfReport entries have been populated.
    pub fn compute_summary(&mut self) {
        let mut successful = 0_usize;
        let mut partial = 0_usize;
        let mut failed = 0_usize;
        let mut quotes_matched = 0_usize;
        let mut quotes_not_found = 0_usize;
        let mut total_quotes = 0_usize;

        for pdf in &self.pdfs {
            // Count "matched" (word-level bounding boxes),
            // "low_confidence_match" (page-level bounding boxes from
            // FallbackExtract), and "page_level_match" (full-page fallback
            // when text location failed but page number is known) as
            // annotated quotes. All three have a highlight annotation in the
            // output PDF; the distinction is in positioning precision.
            let matched = pdf
                .quotes
                .iter()
                .filter(|q| {
                    q.status == "matched"
                        || q.status == "low_confidence_match"
                        || q.status == "page_level_match"
                })
                .count();
            let not_found = pdf
                .quotes
                .iter()
                .filter(|q| {
                    q.status != "matched"
                        && q.status != "low_confidence_match"
                        && q.status != "page_level_match"
                })
                .count();

            total_quotes += pdf.quotes.len();
            quotes_matched += matched;
            quotes_not_found += not_found;

            if pdf.error.is_some() || matched == 0 {
                failed += 1;
            } else if not_found > 0 {
                partial += 1;
            } else {
                successful += 1;
            }
        }

        // Unmatched inputs count as not-found quotes.
        total_quotes += self.unmatched_inputs.len();
        quotes_not_found += self.unmatched_inputs.len();

        self.summary = ReportSummary {
            total_pdfs: self.pdfs.len(),
            successful,
            partial,
            failed,
            total_quotes,
            quotes_matched,
            quotes_not_found,
        };
    }
}

/// Truncates a string to at most `max_len` characters, appending "..." if
/// truncated. Uses character-count-based truncation to avoid panics on
/// multi-byte UTF-8 characters such as curly apostrophes (U+2019, 3 bytes).
/// Byte-index slicing (&s[..max_len]) panics when max_len falls inside a
/// multi-byte character boundary.
pub fn excerpt(s: &str, max_len: usize) -> String {
    let truncated: String = s.chars().take(max_len).collect();
    if s.len() > truncated.len() {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// T-ANNOTATE-050: Report JSON serialization produces valid JSON.
    #[test]
    fn t_annotate_050_report_json_serialization() {
        let report = AnnotationReport {
            timestamp: "2026-02-25T10:00:00Z".into(),
            source_directory: "/source".into(),
            output_directory: "/output".into(),
            summary: ReportSummary {
                total_pdfs: 1,
                successful: 1,
                partial: 0,
                failed: 0,
                total_quotes: 1,
                quotes_matched: 1,
                quotes_not_found: 0,
            },
            pdfs: vec![PdfReport {
                filename: "test.pdf".into(),
                source_path: "/source/test.pdf".into(),
                output_path: Some("/output/test.pdf".into()),
                status: "success".into(),
                quotes: vec![QuoteReport {
                    quote_excerpt: "Hello world".into(),
                    status: "matched".into(),
                    match_method: Some("exact".into()),
                    page: Some(1),
                    chars_matched: Some(11),
                    chars_total: Some(11),
                    pre_check_passed: Some(true),
                    post_check_passed: Some(true),
                    stages_tried: None,
                    fuzzy_score: None,
                }],
                error: None,
            }],
            unmatched_inputs: vec![],
        };

        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"timestamp\""));
        assert!(json.contains("test.pdf"));
    }

    /// T-ANNOTATE-051: compute_summary calculates correct aggregate counts.
    #[test]
    fn t_annotate_051_report_summary_counts() {
        let mut report = AnnotationReport {
            timestamp: String::new(),
            source_directory: String::new(),
            output_directory: String::new(),
            summary: ReportSummary {
                total_pdfs: 0,
                successful: 0,
                partial: 0,
                failed: 0,
                total_quotes: 0,
                quotes_matched: 0,
                quotes_not_found: 0,
            },
            pdfs: vec![
                PdfReport {
                    filename: "a.pdf".into(),
                    source_path: String::new(),
                    output_path: None,
                    status: "success".into(),
                    quotes: vec![
                        QuoteReport {
                            quote_excerpt: "q1".into(),
                            status: "matched".into(),
                            match_method: Some("exact".into()),
                            page: Some(1),
                            chars_matched: None,
                            chars_total: None,
                            pre_check_passed: None,
                            post_check_passed: None,
                            stages_tried: None,
                            fuzzy_score: None,
                        },
                        QuoteReport {
                            quote_excerpt: "q2".into(),
                            status: "not_found".into(),
                            match_method: None,
                            page: None,
                            chars_matched: None,
                            chars_total: None,
                            pre_check_passed: None,
                            post_check_passed: None,
                            stages_tried: Some(vec![
                                "exact: no match on 5 pages".into(),
                                "normalized: no match on 5 pages".into(),
                            ]),
                            fuzzy_score: None,
                        },
                    ],
                    error: None,
                },
                PdfReport {
                    filename: "b.pdf".into(),
                    source_path: String::new(),
                    output_path: None,
                    status: "failed".into(),
                    quotes: vec![],
                    error: Some("load failed".into()),
                },
            ],
            unmatched_inputs: vec![UnmatchedInput {
                title: "Missing".into(),
                author: "Author".into(),
                error: "no PDF matched".into(),
            }],
        };

        report.compute_summary();

        assert_eq!(report.summary.total_pdfs, 2);
        assert_eq!(report.summary.successful, 0);
        assert_eq!(report.summary.partial, 1); // a.pdf has 1 matched + 1 not_found
        assert_eq!(report.summary.failed, 1); // b.pdf has error
        assert_eq!(report.summary.total_quotes, 3); // 2 from PDFs + 1 unmatched
        assert_eq!(report.summary.quotes_matched, 1);
        assert_eq!(report.summary.quotes_not_found, 2);
    }

    /// T-ANNOTATE-052: excerpt truncates strings longer than the limit.
    #[test]
    fn t_annotate_052_excerpt_truncation() {
        let long = "a".repeat(100);
        let result = excerpt(&long, 80);
        assert_eq!(result.len(), 83); // 80 chars + "..."
        assert!(result.ends_with("..."));
    }

    /// T-ANNOTATE-053: excerpt returns the full string when it fits.
    #[test]
    fn t_annotate_053_excerpt_no_truncation() {
        let short = "hello";
        let result = excerpt(short, 80);
        assert_eq!(result, "hello");
    }

    /// T-ANNOTATE-054: QuoteReport with stages_tried serializes the array
    /// into the JSON output.
    #[test]
    fn t_annotate_054_quote_report_stages_tried_serialization() {
        let report = QuoteReport {
            quote_excerpt: "some quote".into(),
            status: "not_found".into(),
            match_method: None,
            page: None,
            chars_matched: None,
            chars_total: None,
            pre_check_passed: None,
            post_check_passed: None,
            stages_tried: Some(vec![
                "exact: no match on 10 pages".into(),
                "normalized: no match on 10 pages".into(),
                "fuzzy: best=42% on page 7 < 90% threshold".into(),
            ]),
            fuzzy_score: None,
        };

        let json = serde_json::to_string(&report).unwrap();
        assert!(
            json.contains("stages_tried"),
            "JSON must contain stages_tried field"
        );
        assert!(
            json.contains("exact: no match on 10 pages"),
            "JSON must contain stage diagnostic text"
        );
        assert!(
            json.contains("42%"),
            "JSON must contain fuzzy score percentage"
        );
    }

    /// T-ANNOTATE-055: QuoteReport with fuzzy_score serializes the score
    /// into the JSON output.
    #[test]
    fn t_annotate_055_quote_report_fuzzy_score_serialization() {
        let report = QuoteReport {
            quote_excerpt: "fuzzy matched quote".into(),
            status: "matched".into(),
            match_method: Some("fuzzy".into()),
            page: Some(3),
            chars_matched: Some(50),
            chars_total: Some(52),
            pre_check_passed: Some(true),
            post_check_passed: Some(true),
            stages_tried: None,
            fuzzy_score: Some(0.953),
        };

        let json = serde_json::to_string(&report).unwrap();
        assert!(
            json.contains("fuzzy_score"),
            "JSON must contain fuzzy_score field"
        );
        assert!(
            json.contains("0.953"),
            "JSON must contain the fuzzy score value"
        );
        // stages_tried should be absent (skip_serializing_if = None).
        assert!(
            !json.contains("stages_tried"),
            "stages_tried must be omitted when None"
        );
    }

    /// T-ANNOTATE-056: QuoteReport with both stages_tried and fuzzy_score
    /// set to None omits both fields from JSON output.
    #[test]
    fn t_annotate_056_quote_report_none_fields_omitted() {
        let report = QuoteReport {
            quote_excerpt: "exact match".into(),
            status: "matched".into(),
            match_method: Some("exact".into()),
            page: Some(1),
            chars_matched: Some(20),
            chars_total: Some(20),
            pre_check_passed: Some(true),
            post_check_passed: Some(true),
            stages_tried: None,
            fuzzy_score: None,
        };

        let json = serde_json::to_string(&report).unwrap();
        assert!(
            !json.contains("stages_tried"),
            "stages_tried must be omitted when None"
        );
        assert!(
            !json.contains("fuzzy_score"),
            "fuzzy_score must be omitted when None"
        );
    }

    // -----------------------------------------------------------------------
    // DEFECT-007: compute_summary handles "low_confidence_match" status
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-BUG-007-REPORT-001: compute_summary counts
    /// "low_confidence_match" quotes as matched (quotes_matched), because
    /// the annotation exists in the output PDF (page-level highlight).
    ///
    /// Regression test for DEFECT-007: before the fix, FallbackExtract
    /// matches reported status "matched" despite having only a page-level
    /// bounding box. After the fix, the status is "low_confidence_match"
    /// and compute_summary must count it as a successful (annotated) quote.
    #[test]
    fn t_annotate_bug_007_report_001_low_confidence_counted_as_matched() {
        let mut report = AnnotationReport {
            timestamp: String::new(),
            source_directory: String::new(),
            output_directory: String::new(),
            summary: ReportSummary {
                total_pdfs: 0,
                successful: 0,
                partial: 0,
                failed: 0,
                total_quotes: 0,
                quotes_matched: 0,
                quotes_not_found: 0,
            },
            pdfs: vec![PdfReport {
                filename: "fama1970.pdf".into(),
                source_path: String::new(),
                output_path: Some("/output/fama1970.pdf".into()),
                status: "success".into(),
                quotes: vec![
                    QuoteReport {
                        quote_excerpt: "quote with exact match".into(),
                        status: "matched".into(),
                        match_method: Some("exact".into()),
                        page: Some(1),
                        chars_matched: Some(50),
                        chars_total: Some(50),
                        pre_check_passed: Some(true),
                        post_check_passed: Some(true),
                        stages_tried: None,
                        fuzzy_score: None,
                    },
                    QuoteReport {
                        quote_excerpt: "quote with page-level match".into(),
                        status: "low_confidence_match".into(),
                        match_method: Some("fallbackextract".into()),
                        page: Some(2),
                        chars_matched: Some(1),
                        chars_total: Some(202),
                        pre_check_passed: Some(false),
                        post_check_passed: Some(true),
                        stages_tried: None,
                        fuzzy_score: None,
                    },
                ],
                error: None,
            }],
            unmatched_inputs: vec![],
        };

        report.compute_summary();

        assert_eq!(
            report.summary.quotes_matched, 2,
            "both 'matched' and 'low_confidence_match' must be counted as matched"
        );
        assert_eq!(
            report.summary.quotes_not_found, 0,
            "no quotes must be counted as not_found"
        );
        assert_eq!(report.summary.total_quotes, 2);
    }

    /// T-ANNOTATE-BUG-007-REPORT-002: compute_summary classifies a PDF with
    /// only "low_confidence_match" quotes as "successful" (all quotes were
    /// annotated, even though imprecisely).
    #[test]
    fn t_annotate_bug_007_report_002_all_low_confidence_produces_successful_pdf() {
        let mut report = AnnotationReport {
            timestamp: String::new(),
            source_directory: String::new(),
            output_directory: String::new(),
            summary: ReportSummary {
                total_pdfs: 0,
                successful: 0,
                partial: 0,
                failed: 0,
                total_quotes: 0,
                quotes_matched: 0,
                quotes_not_found: 0,
            },
            pdfs: vec![PdfReport {
                filename: "scanned.pdf".into(),
                source_path: String::new(),
                output_path: Some("/output/scanned.pdf".into()),
                status: "success".into(),
                quotes: vec![QuoteReport {
                    quote_excerpt: "page-level match only".into(),
                    status: "low_confidence_match".into(),
                    match_method: Some("fallbackextract".into()),
                    page: Some(1),
                    chars_matched: Some(1),
                    chars_total: Some(100),
                    pre_check_passed: Some(false),
                    post_check_passed: Some(true),
                    stages_tried: None,
                    fuzzy_score: None,
                }],
                error: None,
            }],
            unmatched_inputs: vec![],
        };

        report.compute_summary();

        assert_eq!(
            report.summary.successful, 1,
            "PDF with all low_confidence_match quotes must be counted as successful"
        );
        assert_eq!(report.summary.partial, 0);
        assert_eq!(report.summary.failed, 0);
    }

    /// T-ANNOTATE-BUG-007-REPORT-003: compute_summary correctly handles a
    /// mixed PDF with "matched", "low_confidence_match", and "not_found"
    /// quotes. The PDF is classified as "partial" because some quotes were
    /// not found.
    #[test]
    fn t_annotate_bug_007_report_003_mixed_statuses_produces_partial_pdf() {
        let mut report = AnnotationReport {
            timestamp: String::new(),
            source_directory: String::new(),
            output_directory: String::new(),
            summary: ReportSummary {
                total_pdfs: 0,
                successful: 0,
                partial: 0,
                failed: 0,
                total_quotes: 0,
                quotes_matched: 0,
                quotes_not_found: 0,
            },
            pdfs: vec![PdfReport {
                filename: "mixed.pdf".into(),
                source_path: String::new(),
                output_path: Some("/output/mixed.pdf".into()),
                status: "partial".into(),
                quotes: vec![
                    QuoteReport {
                        quote_excerpt: "found precisely".into(),
                        status: "matched".into(),
                        match_method: Some("exact".into()),
                        page: Some(1),
                        chars_matched: Some(30),
                        chars_total: Some(30),
                        pre_check_passed: Some(true),
                        post_check_passed: Some(true),
                        stages_tried: None,
                        fuzzy_score: None,
                    },
                    QuoteReport {
                        quote_excerpt: "found imprecisely".into(),
                        status: "low_confidence_match".into(),
                        match_method: Some("fallbackextract".into()),
                        page: Some(5),
                        chars_matched: Some(1),
                        chars_total: Some(150),
                        pre_check_passed: Some(false),
                        post_check_passed: Some(true),
                        stages_tried: None,
                        fuzzy_score: None,
                    },
                    QuoteReport {
                        quote_excerpt: "not found anywhere".into(),
                        status: "not_found".into(),
                        match_method: None,
                        page: None,
                        chars_matched: None,
                        chars_total: None,
                        pre_check_passed: None,
                        post_check_passed: None,
                        stages_tried: Some(vec!["exact: no match".into()]),
                        fuzzy_score: None,
                    },
                ],
                error: None,
            }],
            unmatched_inputs: vec![],
        };

        report.compute_summary();

        assert_eq!(report.summary.quotes_matched, 2, "matched + low_confidence");
        assert_eq!(report.summary.quotes_not_found, 1, "one not_found quote");
        assert_eq!(
            report.summary.partial, 1,
            "PDF with some matched and some not_found must be partial"
        );
    }

    /// T-ANNOTATE-BUG-007-REPORT-004: The "low_confidence_match" status
    /// serializes correctly in JSON output.
    #[test]
    fn t_annotate_bug_007_report_004_low_confidence_json_serialization() {
        let report = QuoteReport {
            quote_excerpt: "page-level match".into(),
            status: "low_confidence_match".into(),
            match_method: Some("fallbackextract".into()),
            page: Some(1),
            chars_matched: Some(1),
            chars_total: Some(202),
            pre_check_passed: Some(false),
            post_check_passed: Some(true),
            stages_tried: None,
            fuzzy_score: None,
        };

        let json = serde_json::to_string(&report).unwrap();
        assert!(
            json.contains("\"low_confidence_match\""),
            "JSON must contain the low_confidence_match status value"
        );
        assert!(
            json.contains("\"fallbackextract\""),
            "JSON must contain the match_method"
        );
    }

    // -----------------------------------------------------------------------
    // v3 tests: page_level_match counting and post_check_passed omission
    // -----------------------------------------------------------------------

    /// T-ANNOTATE-V3-REPORT-001: compute_summary counts "page_level_match"
    /// quotes as matched. These are full-page highlights created when text
    /// location failed but the page number was known.
    #[test]
    fn t_annotate_v3_report_001_page_level_match_counted_as_matched() {
        let mut report = AnnotationReport {
            timestamp: String::new(),
            source_directory: String::new(),
            output_directory: String::new(),
            summary: ReportSummary {
                total_pdfs: 0,
                successful: 0,
                partial: 0,
                failed: 0,
                total_quotes: 0,
                quotes_matched: 0,
                quotes_not_found: 0,
            },
            pdfs: vec![PdfReport {
                filename: "test.pdf".into(),
                source_path: String::new(),
                output_path: Some("/output/test.pdf".into()),
                status: "success".into(),
                quotes: vec![
                    QuoteReport {
                        quote_excerpt: "exact match".into(),
                        status: "matched".into(),
                        match_method: Some("exact".into()),
                        page: Some(1),
                        chars_matched: Some(30),
                        chars_total: Some(30),
                        pre_check_passed: Some(true),
                        post_check_passed: None,
                        stages_tried: None,
                        fuzzy_score: None,
                    },
                    QuoteReport {
                        quote_excerpt: "page-level fallback".into(),
                        status: "page_level_match".into(),
                        match_method: Some("page_level".into()),
                        page: Some(5),
                        chars_matched: Some(1),
                        chars_total: Some(200),
                        pre_check_passed: Some(false),
                        post_check_passed: None,
                        stages_tried: Some(vec!["exact: no match".into()]),
                        fuzzy_score: None,
                    },
                ],
                error: None,
            }],
            unmatched_inputs: vec![],
        };

        report.compute_summary();

        assert_eq!(
            report.summary.quotes_matched, 2,
            "'matched' and 'page_level_match' must both be counted as matched"
        );
        assert_eq!(report.summary.quotes_not_found, 0);
        assert_eq!(
            report.summary.successful, 1,
            "PDF with all quotes annotated must be 'successful'"
        );
    }

    /// T-ANNOTATE-V3-REPORT-002: compute_summary classifies a PDF with
    /// "page_level_match" and "not_found" as "partial".
    #[test]
    fn t_annotate_v3_report_002_page_level_and_not_found_is_partial() {
        let mut report = AnnotationReport {
            timestamp: String::new(),
            source_directory: String::new(),
            output_directory: String::new(),
            summary: ReportSummary {
                total_pdfs: 0,
                successful: 0,
                partial: 0,
                failed: 0,
                total_quotes: 0,
                quotes_matched: 0,
                quotes_not_found: 0,
            },
            pdfs: vec![PdfReport {
                filename: "mixed.pdf".into(),
                source_path: String::new(),
                output_path: Some("/output/mixed.pdf".into()),
                status: "partial".into(),
                quotes: vec![
                    QuoteReport {
                        quote_excerpt: "page-level".into(),
                        status: "page_level_match".into(),
                        match_method: Some("page_level".into()),
                        page: Some(3),
                        chars_matched: Some(1),
                        chars_total: Some(100),
                        pre_check_passed: Some(false),
                        post_check_passed: None,
                        stages_tried: None,
                        fuzzy_score: None,
                    },
                    QuoteReport {
                        quote_excerpt: "not found".into(),
                        status: "not_found".into(),
                        match_method: None,
                        page: None,
                        chars_matched: None,
                        chars_total: None,
                        pre_check_passed: None,
                        post_check_passed: None,
                        stages_tried: None,
                        fuzzy_score: None,
                    },
                ],
                error: None,
            }],
            unmatched_inputs: vec![],
        };

        report.compute_summary();

        assert_eq!(report.summary.quotes_matched, 1);
        assert_eq!(report.summary.quotes_not_found, 1);
        assert_eq!(report.summary.partial, 1, "PDF must be 'partial'");
    }

    /// T-ANNOTATE-V3-REPORT-003: post_check_passed=None is omitted from JSON.
    #[test]
    fn t_annotate_v3_report_003_post_check_omitted_from_json() {
        let report = QuoteReport {
            quote_excerpt: "test".into(),
            status: "matched".into(),
            match_method: Some("exact".into()),
            page: Some(1),
            chars_matched: Some(20),
            chars_total: Some(20),
            pre_check_passed: Some(true),
            post_check_passed: None,
            stages_tried: None,
            fuzzy_score: None,
        };

        let json = serde_json::to_string(&report).unwrap();
        assert!(
            !json.contains("post_check_passed"),
            "post_check_passed must be omitted when None, got: {json}"
        );
    }

    /// T-ANNOTATE-V3-REPORT-004: page_level_match serializes correctly in JSON.
    #[test]
    fn t_annotate_v3_report_004_page_level_match_json_serialization() {
        let report = QuoteReport {
            quote_excerpt: "full page highlight".into(),
            status: "page_level_match".into(),
            match_method: Some("page_level".into()),
            page: Some(5),
            chars_matched: Some(1),
            chars_total: Some(300),
            pre_check_passed: Some(false),
            post_check_passed: None,
            stages_tried: Some(vec![
                "exact: no match on 30 pages".into(),
                "normalized: no match on 30 pages".into(),
                "fuzzy: best=55% on page 5 < 80% threshold".into(),
            ]),
            fuzzy_score: None,
        };

        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"page_level_match\""));
        assert!(json.contains("\"page_level\""));
        assert!(json.contains("stages_tried"));
        assert!(!json.contains("post_check_passed"));
    }

    /// T-ANNOTATE-V3-REPORT-005: All three match statuses together produce
    /// correct summary counts.
    #[test]
    fn t_annotate_v3_report_005_all_three_match_statuses() {
        let mut report = AnnotationReport {
            timestamp: String::new(),
            source_directory: String::new(),
            output_directory: String::new(),
            summary: ReportSummary {
                total_pdfs: 0,
                successful: 0,
                partial: 0,
                failed: 0,
                total_quotes: 0,
                quotes_matched: 0,
                quotes_not_found: 0,
            },
            pdfs: vec![PdfReport {
                filename: "all_types.pdf".into(),
                source_path: String::new(),
                output_path: Some("/output/all_types.pdf".into()),
                status: "success".into(),
                quotes: vec![
                    QuoteReport {
                        quote_excerpt: "exact".into(),
                        status: "matched".into(),
                        match_method: Some("exact".into()),
                        page: Some(1),
                        chars_matched: Some(20),
                        chars_total: Some(20),
                        pre_check_passed: Some(true),
                        post_check_passed: None,
                        stages_tried: None,
                        fuzzy_score: None,
                    },
                    QuoteReport {
                        quote_excerpt: "low confidence".into(),
                        status: "low_confidence_match".into(),
                        match_method: Some("fallbackextract".into()),
                        page: Some(3),
                        chars_matched: Some(1),
                        chars_total: Some(150),
                        pre_check_passed: Some(false),
                        post_check_passed: None,
                        stages_tried: None,
                        fuzzy_score: None,
                    },
                    QuoteReport {
                        quote_excerpt: "page level".into(),
                        status: "page_level_match".into(),
                        match_method: Some("page_level".into()),
                        page: Some(7),
                        chars_matched: Some(1),
                        chars_total: Some(200),
                        pre_check_passed: Some(false),
                        post_check_passed: None,
                        stages_tried: None,
                        fuzzy_score: None,
                    },
                ],
                error: None,
            }],
            unmatched_inputs: vec![],
        };

        report.compute_summary();

        assert_eq!(
            report.summary.quotes_matched, 3,
            "all three match types counted as matched"
        );
        assert_eq!(report.summary.quotes_not_found, 0);
        assert_eq!(report.summary.successful, 1);
    }
}
