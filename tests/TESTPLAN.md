# NeuronCite - Parallel Sub-Agent Verification Protocol

Complete test coverage for the NeuronCite MCP toolset (43 tools). The main agent acts as a pure orchestrator: it creates shared sessions in a bootstrap phase, dispatches parallel sub-agents that each write their results to a dedicated .md file, and compiles the final report from those files.

---

## Test Environment

A test project directory must be placed alongside this file (at the same level
as the `tests/` directory). The test project should contain PDF files, a LaTeX
document, and a BibTeX bibliography. Use whatever files are available in your
working directory.

- **PDF directory:** The directory containing PDF files to index. Use the files
  available in the test project next to this repository.
- **LaTeX file:** A `.tex` file from the test project for citation verification.
- **BibTeX file:** A `.bib` file from the test project for citation matching.
- **Output directory:** Create a subfolder `neuroncite_tests/` inside the PDF
  directory for all test output and result files.

---

## Orchestration Protocol

### Phase 0 - Bootstrap (Main Agent Executes Directly)

The main agent runs these steps sequentially. No sub-agents are used in this phase.

1. **System checks (Section 1):** Execute cases 1.1-1.5. Record API version, GPU status, pdfium availability, reranker status, vector dimension.
2. **Discovery:** `neuroncite_discover(directory=<PDF_DIR>)` - record pdf_count, file list.
3. **Index session_A (sentence strategy):** `neuroncite_index(directory=<PDF_DIR>, chunk_strategy="sentence", chunk_size=256)` - poll `neuroncite_job_status` until `state="completed"`. Record **session_A** ID.
4. **Index session_B (word strategy):** `neuroncite_index(directory=<PDF_DIR>, chunk_strategy="word", chunk_size=300, chunk_overlap=50)` - poll until completed. Record **session_B** ID.
5. **Collect file metadata:** Call `neuroncite_files(session_id=session_A)` to get all file_ids with filenames. Call `neuroncite_files(session_id=session_B)` likewise.
6. **Indexing edge cases (Section 2 remainder):** Execute cases 2.4-2.12 (concurrent rejection, nonexistent path, URL mode, file_types, mutually exclusive modes).
7. **Write bootstrap file:** Write `neuroncite_tests/bootstrap.md` containing:
   - session_A ID, session_B ID
   - All file_ids with filenames (from both sessions)
   - System info from section 1 (API version, GPU, pdfium, backends)
   - PDF directory path, LaTeX path, BibTeX path, output directory path
   - Results for sections 1 and 2 (using the result file template below)

### Phase 1 - Parallel Sub-Agents (Launch ALL Simultaneously)

After bootstrap completes, the main agent launches all sub-agents in a single message. Each sub-agent receives the bootstrap data and its assigned test sections. Each sub-agent writes its results to the specified file in `neuroncite_tests/`.

| Sub-Agent | Sections | Session(s) | Result File | Scope |
|-----------|----------|------------|-------------|-------|
| **SA-SESSION** | 3, 13 | session_A (read-only) | `section_03_13_session.md` | Session management, job lifecycle |
| **SA-SEARCH** | 4, 5, 6 | session_A (read-only) | `section_04_05_06_search.md` | Search, input validation, regression |
| **SA-CONTENT** | 7, 21 | session_A (read-only) | `section_07_21_content.md` | Content retrieval, batch content |
| **SA-EXPORT** | 8 | session_A (read-only) | `section_08_export.md` | All export formats |
| **SA-PREVIEW** | 14, 25 | session_A (read-only) | `section_14_25_preview.md` | Chunk preview, text search |
| **SA-ANNOTATION** | 9, 10, 19, 27, 28 | session_A (for search) | `section_09_10_19_27_28_annotation.md` | Annotation pipeline, appearance, removal, status |
| **SA-CITATION** | 11, 11b, 12, 24, 31, 32 | session_A | `section_11_12_24_31_32_citation.md` | Citation verification, export, retry, fetch sources, bib report |
| **SA-RERANKER** | 17, 18 | session_A (read-only) | `section_17_18_reranker.md` | Reranker load, search integration |
| **SA-CROSSSESSION** | 16, 22, 26 | session_A + session_B (read-only) | `section_16_22_26_cross.md` | Multi-search, compare, diff |
| **SA-INCREMENTAL** | 15, 23 | session_B (modifies) | `section_15_23_incremental.md` | index_add, reindex_file |
| **SA-HTML** | 29, 30 | creates own session_D | `section_29_30_html.md` | HTML fetch, crawl |

### Phase 2 - End-to-End (After Phase 1 Completes)

The main agent waits for all Phase 1 sub-agents to finish. Then:

| Sub-Agent | Sections | Notes |
|-----------|----------|-------|
| **SA-E2E** | 20 | Creates own sessions as needed. Reads Phase 1 result files for context. |

### Phase 3 - Final Report (Main Agent Compiles)

1. Read `neuroncite_tests/bootstrap.md`
2. Read ALL files matching `neuroncite_tests/section_*.md`
3. Compile `FEHLERBERICHT.md` in the output directory with this structure:
   - **Test Environment** -- system info, GPU/CPU, PDF directory contents (file count, total pages), model used, chunk strategies, all session IDs
   - **Regression Case Results** -- one row per REF number: REF-ID, case number, status (PASS/FAIL/SKIP), observed behavior if different from expected
   - **Newly Discovered Defects** -- for each defect: unique ID, severity (critical/major/minor), tool name, reproduction steps, expected behavior, actual behavior, whether it is a regression or new finding
   - **Correctly Functioning Features** -- table: tool name | cases tested | status | remarks (include semantic quality observation for search results)
   - **Test Statistics** -- total MCP tool calls made, total cases executed, defects found by severity, cases that produced unexpected behavior
   - **Semantic Search Quality Assessment** -- for at least 3 different queries: query text, top-3 result sources, topical relevance, vector_score range

---

## Sub-Agent Result File Template

Every sub-agent writes exactly this format to its result file. No deviation.

```markdown
# <Sub-Agent Name>: Sections X, Y, Z
**Timestamp:** <ISO 8601 datetime>
**Sessions used:** session_A=<id>, session_B=<id>

## Results
| # | Case | Result | Deviation |
|---|------|--------|-----------|
| X.1 | brief description | PASS | |
| X.2 | brief description | FAIL | actual: "error 404" expected: "session not found" |
| X.3 | brief description | SKIP | pdfium not available |

## Edge Cases Discovered
| # | Description | Result | Deviation |
|---|-------------|--------|-----------|
| X.E1 | description of discovered edge case | PASS | |

## Defects
| ID | Severity | Tool | Description | Repro |
|----|----------|------|-------------|-------|
| DEF-001 | major | neuroncite_search | min_score=0.0 returns empty | search(query="test", min_score=0.0) |

## State for Downstream
session_ids_created: []
file_ids_created: []
job_ids_created: []
notes: ""
```

---

## Sub-Agent Prompt Template

The main agent sends each sub-agent a prompt structured as follows. The `TEST CASES` section contains the exact tables from the relevant sections of this document.

```
You are a test sub-agent for NeuronCite MCP tools. Execute test sections [X, Y, Z] completely.

BOOTSTRAP DATA:
- session_A: <id> (sentence strategy, chunk_size=256) -- DO NOT MODIFY
- session_B: <id> (word strategy, chunk_size=300, overlap=50) -- modify only if your assignment says so
- file_ids_session_A: {<id1>: "filename1.pdf", <id2>: "filename2.pdf", ...}
- file_ids_session_B: {<id1>: "filename1.pdf", <id2>: "filename2.pdf", ...}
- PDF directory: <path>
- Output directory: <path>
- LaTeX file: <path>
- BibTeX file: <path>
- System info: API version=<v>, GPU=<status>, pdfium=<status>, reranker=<status>

TEST CASES:
[paste the relevant section tables from TESTPLAN.md here]

RULES:
1. Execute every numbered case listed above.
2. After the numbered cases, discover and test edge cases (ID format: X.E1, X.E2, ...).
3. Write results to: neuroncite_tests/<assigned_filename>.md using the result file template.
4. For each FAIL: record exact actual vs expected behavior in the Deviation column.
5. For each defect: assign severity (critical/major/minor) and provide one-line reproduction steps.
6. DO NOT modify session_A unless your assignment explicitly permits it.
7. The result .md file is your ONLY deliverable. Everything relevant goes into it.
8. If a case requires a running job (annotation, citation), create and manage it within your scope.
```

---

## Conflict Notes

Read these before launching sub-agents. They document known interactions between concurrent sub-agents.

- **SA-INCREMENTAL modifies session_B:** SA-CROSSSESSION also reads session_B. Both run in Phase 1. This is acceptable because SA-CROSSSESSION's diff/compare operations capture state at read time. If SA-INCREMENTAL's modifications cause unexpected diff results, the sub-agent should note this in its result file rather than treating it as a defect.
- **SA-RERANKER loads a server-wide model:** The reranker is a singleton. SA-SEARCH must NOT call `rerank=true` except in case 17.7 (error case: reranker not loaded). All positive rerank tests belong to SA-RERANKER.
- **SA-CITATION creates a citation job on session_A:** This is isolated. Citation jobs do not affect search, annotation, or export operations on the same session.
- **SA-CITATION section 31 modifies session_B:** `neuroncite_citation_fetch_sources` adds downloaded source files to session_B. SA-CROSSSESSION also reads session_B. Acceptable because SA-CROSSSESSION captures state at read time and reports any unexpected diff results rather than treating them as defects.
- **SA-SESSION section 13.6 (delete + re-index):** This sub-agent must create a SEPARATE session_C for the delete/re-index test. It must NEVER delete session_A or session_B.
- **SA-ANNOTATION creates output files:** Annotation tests write to the output directory. Each annotation sub-agent test should use a unique subfolder or filename prefix to avoid collisions with SA-E2E.
- **SA-HTML creates its own sessions:** SA-HTML indexes fetched HTML into its own session_D. This does not conflict with any other sub-agent because no other sub-agent reads session_D.

---

## 1. System Infrastructure Verification

**Sub-Agent:** BOOTSTRAP (Phase 0, main agent)

| # | Case | Expected |
|---|------|----------|
| 1.1 | `neuroncite_health` | Returns API version, backend, GPU status, reranker_available status, vector dimension |
| 1.2 | `neuroncite_doctor` | Reports CUDA, Tesseract, pdfium, compiled backends, cache directory |
| 1.3 | `neuroncite_models` | Lists models with cache status, dimensions, max_seq_len |
| 1.4 | `neuroncite_sessions` | Shows baseline state (empty or pre-existing sessions) |
| 1.5 | `neuroncite_jobs` | Shows baseline state (completed/failed jobs from prior runs) |

---

## 2. Indexing and Discovery

**Sub-Agent:** BOOTSTRAP (Phase 0, main agent)

| # | Case | Expected |
|---|------|----------|
| 2.1 | `neuroncite_discover(directory=<PDF_DIR>)` | Returns pdf_count, total_size_bytes, unindexed_pdfs list |
| 2.2 | `neuroncite_index(directory=<PDF_DIR>, chunk_strategy="sentence", chunk_size=256)` | Returns session_id and job_id; job completes successfully |
| 2.3 | Poll `neuroncite_job_status` until `state="completed"` | Record total duration |
| 2.4 | `neuroncite_index` while an index job is already running | Rejection with reference to the running job |
| 2.5 | `neuroncite_index(directory="D:\\nonexistent\\path")` | Error: `"directory does not exist"` |
| 2.6 | `neuroncite_discover(directory="D:\\nonexistent\\path")` | Warning with `directory_exists: false` |
| 2.7 | `neuroncite_index(urls=["<CACHED_URL>"])` after `neuroncite_html_fetch` has cached a page | Returns session_id; job completes; session contains HTML-sourced file with `source_type="html"` |
| 2.8 | `neuroncite_index(directory=<PDF_DIR>, file_types=["pdf"])` | Only PDF files indexed; other file types in the directory are ignored |
| 2.9 | `neuroncite_index(directory=<PDF_DIR>, session_id=<EXISTING>)` | Files added to existing session instead of creating a new one |
| 2.10 | `neuroncite_index(directory=<PDF_DIR>, chunk_strategy="token", chunk_size=256, chunk_overlap=32)` | Token-based chunking (the default strategy) with specified parameters |
| 2.11 | `neuroncite_index` with no `directory`, `urls`, or `files` parameter | Error: at least one input source required |
| 2.12 | `neuroncite_index(directory=<PDF_DIR>, urls=["<URL>"])` | Error: mutually exclusive input modes (only one of directory/urls/files allowed) |

---

## 3. Session and File Management

**Sub-Agent:** SA-SESSION

| # | Case | Expected |
|---|------|----------|
| 3.1 | `neuroncite_sessions` after indexing | New session with correct file_count, total_chunks, total_pages |
| 3.2 | `neuroncite_session_update(label="Test Label")` | Label is set |
| 3.3 | `neuroncite_session_update(label=null)` | Label is cleared |
| 3.4 | `neuroncite_session_update(tags=["finance", "statistics"])` | Tags array is stored and returned |
| 3.5 | `neuroncite_session_update(tags=null)` | Tags are cleared |
| 3.6 | `neuroncite_session_update(metadata={"source": "manual", "version": "1.0"})` | Metadata object is stored and returned |
| 3.7 | `neuroncite_session_update(metadata=null)` | Metadata is cleared |
| 3.8 | `neuroncite_session_update(label="L", tags=["t"], metadata={"k":"v"})` | All three fields set in a single call |
| 3.9 | `neuroncite_files(sort_by="name")` | Alphabetical order |
| 3.10 | `neuroncite_files(sort_by="pages")` | Descending by page count |
| 3.11 | `neuroncite_files(sort_by="chunks")` | Descending by chunk count |
| 3.12 | `neuroncite_files(sort_by="size")` | Descending by file size |
| 3.13 | `neuroncite_files(file_id=<ONE_FILE_ID>)` | Single-file detail returned |
| 3.14 | `neuroncite_quality_report` | Reports OCR-heavy files, quality flags |
| 3.15 | `neuroncite_file_compare(file_name_pattern="%<PARTIAL_NAME>%")` | Cross-session comparison data |
| 3.16 | `neuroncite_file_compare(file_name_pattern="%nonexistent%")` | Empty result set |

---

## 4. Search Functionality

**Sub-Agent:** SA-SEARCH

| # | Case | Expected |
|---|------|----------|
| 4.1 | `neuroncite_search(query="<domain-specific term>", top_k=5)` | Relevant results with score_summary |
| 4.2 | `neuroncite_search(min_score=0.72)` | All results have vector_score >= 0.72 |
| 4.3 | `neuroncite_search(file_ids=[<ONE_ID>])` | Results only from that file |
| 4.4 | `neuroncite_search(use_fts=false)` | All bm25_rank fields are null |
| 4.5 | `neuroncite_batch_search(queries=[4 different queries])` | 4 result groups, error_count=0 |
| 4.6 | `neuroncite_search(refine=true)` | Results contain refined sub-chunk content (shorter, more focused passages) |
| 4.7 | `neuroncite_search(refine=false)` | Results contain full chunk content (no sub-chunk narrowing) |
| 4.8 | `neuroncite_search(refine=true, refine_divisors="4,8")` | Custom divisor values accepted; results differ from default "4,8,16" |
| 4.9 | `neuroncite_search(page_start=1, page_end=5)` | All results come from pages 1-5 only |
| 4.10 | `neuroncite_search(page_start=3, page_end=3, file_ids=[<ONE_ID>])` | Results restricted to page 3 of the specified file |
| 4.11 | `neuroncite_search(page_start=5)` without `page_end` | Error or ignored (both must be provided together) |
| 4.12 | `neuroncite_batch_search(queries=[...], refine=true)` | Refinement applied to all query results |
| 4.13 | `neuroncite_batch_search(queries=[...], refine=false)` | No refinement applied across all queries |

---

## 5. Search Input Validation

**Sub-Agent:** SA-SEARCH

| # | Case | Expected |
|---|------|----------|
| 5.1 | `neuroncite_search(query="")` | Error: `"query must not be empty"` |
| 5.2 | `neuroncite_search(top_k=0)` | Error: `"top_k must be between 1 and 50"` |
| 5.3 | `neuroncite_search(top_k=51)` | Error: `"top_k must be between 1 and 50"` |
| 5.4 | `neuroncite_search(session_id=999)` | Error: `"session not found"` |
| 5.5 | Query with special characters `!@#$%^&*()_+-=[]{}` | No crash, arbitrary results acceptable |
| 5.6 | Query with umlauts and Unicode characters | No crash |

---

## 6. Search Regression Cases

**Sub-Agent:** SA-SEARCH

| # | Ref | Case | Expected |
|---|-----|------|----------|
| 6.1 | REF-006 | Query exceeding 350 words (>512 tokens) | Response contains a `truncation_warning` field |
| 6.2 | REF-010 | `min_score=1.1` | Error: `"min_score must be between 0.0 and 1.0"` |
| 6.3 | REF-010 | `min_score=-0.5` | Error (negative value rejected) |
| 6.4 | REF-008 | `file_ids=[99999]` (non-existent) | Warning in response about non-existent file_id |
| 6.5 | REF-007 | `batch_search` with duplicate query IDs `{id:"dup",...},{id:"dup",...}` | First query executes, second gets error. Both must NOT be discarded. |

---

## 7. Content Retrieval and Chunk Browsing

**Sub-Agent:** SA-CONTENT

The `neuroncite_content` tool retrieves text content from indexed documents by "part" number. The part concept is format-aware: a page in PDFs, a heading-based section in HTML. For HTML sources, the response includes `web_source` metadata (URL, title, domain, language, author). Content exceeding 100 KB per part is truncated with a `truncated` flag and `original_bytes` count.

| # | Case | Expected |
|---|------|----------|
| 7.1 | `neuroncite_content(file_id=<PDF_FILE_ID>, part=1)` | Part content returned with `source_type="pdf"` |
| 7.2 | `neuroncite_content(file_id=<PDF_FILE_ID>, start=1, end=3)` | 3 parts returned, `part_count=3`; parts array ordered by part number |
| 7.3 | `neuroncite_chunks(limit=5)` | 5 chunks with content, page_start, page_end, word_count |
| 7.4 | `neuroncite_chunks(page_number=5)` | Only chunks covering page 5 |
| 7.5 | `neuroncite_content(file_id=<ID>, part=0)` | Error: `"part must be >= 1"` |
| 7.6 | `neuroncite_content(file_id=<ID>, part=-1)` | Error |
| 7.7 | `neuroncite_content(file_id=<ID>, part=999)` | Error: part not found |
| 7.8 | `neuroncite_content(file_id=99999, part=1)` | Error: `"part not found"` or `"file not found"` |
| 7.9 | `neuroncite_content(file_id=<ID>, start=5, end=3)` | Error: `"start must be <= end"` |
| 7.10 | `neuroncite_content(file_id=<ID>, start=1, end=25)` | Error: `"part range exceeds maximum of 20 parts"` |
| 7.11 | `neuroncite_content(file_id=<ID>, part=1, start=1, end=2)` | Error: `part` is mutually exclusive with `start`/`end` |
| 7.12 | `neuroncite_content(file_id=<ID>)` with no `part`, `start`, or `end` | Error: at least one parameter required |
| 7.13 | `neuroncite_content(file_id=<ID>, start=1)` without `end` | Error: both `start` and `end` must be provided together |
| 7.14 | `neuroncite_content(file_id=<HTML_FILE_ID>, part=1)` | Returns content with `source_type="html"` and `web_source` metadata (url, title, domain, language, author) |

---

## 8. Export

**Sub-Agent:** SA-EXPORT

| # | Case | Expected |
|---|------|----------|
| 8.1 | `neuroncite_export(format="markdown", top_k=3)` | Valid markdown output |
| 8.2 | `neuroncite_export(format="bibtex")` | Valid BibTeX entries with `@article{...}` |
| 8.3 | `neuroncite_export(format="csl-json")` | Valid JSON array with author, title, year |
| 8.4 | `neuroncite_export(format="ris")` | Valid RIS output with TY, AU, TI, PY tags |
| 8.5 | `neuroncite_export(format="plain-text")` | Plain-text formatted citations |
| 8.6 | `neuroncite_export(format="bibtex", deduplicate=true)` | One entry per source document with consolidated page ranges and aggregated scores |
| 8.7 | `neuroncite_export(format="bibtex", deduplicate=false)` | Each search result produces a separate BibTeX entry |
| 8.8 | `neuroncite_export(format="csl-json", deduplicate=true)` | Deduplicated entries with merged page ranges |
| 8.9 | `neuroncite_export(format="ris", deduplicate=true)` | Deduplicated RIS entries |
| 8.10 | `neuroncite_export(format="markdown", deduplicate=true)` | Markdown format is unaffected by deduplicate flag |
| 8.11 | **REF-003**: `neuroncite_export(query="")` | Error: `"query must not be empty"` (must NOT return results) |

---

## 9. Annotation Pipeline

**Sub-Agent:** SA-ANNOTATION

| # | Case | Expected |
|---|------|----------|
| 9.1 | JSON input with 2+ quotes from different PDFs (author present in filename) | Job completes, annotated PDFs in output, annotate_report.json shows quotes_matched > 0 |
| 9.2 | CSV input with multiple quotes and different colors | CSV parsed correctly, multiple PDFs annotated |
| 9.3 | `input_data="not valid csv or json"` | Parsing error |
| 9.4 | Invalid color value `"notacolor"` in input_data | Error: `"invalid color"` |
| 9.5 | `source_directory="D:\\nonexistent"` | Error: `"source_directory does not exist"` |
| 9.6 | Start a second annotation job while the first is running | Rejection |
| 9.7 | `neuroncite_annotate(dry_run=true, ...)` with valid input | Returns per-row preview with match status; no job created, no files written |
| 9.8 | `neuroncite_annotate(dry_run=true, ...)` with a PDF that does not exist in source_directory | Preview shows unmatched row |
| 9.9 | Run annotation job (creates output_directory) -> then `neuroncite_annotate(append=true, ...)` with additional quotes | Reads PDFs from output_directory; new highlights layered on top of existing annotations |
| 9.10 | `neuroncite_annotate(append=true, ...)` when output_directory does not exist | Error (output_directory must already exist for append mode) |
| 9.11 | `neuroncite_inspect_annotations(pdf_path=<ANNOTATED_PDF>)` | Returns annotation list with type, fill_color (#RRGGBB), opacity, bounding rect; summary with highlight_count and unique_colors |
| 9.12 | `neuroncite_inspect_annotations(pdf_path=<ANNOTATED_PDF>, page_number=1)` | Only annotations from page 1 |
| 9.13 | `neuroncite_inspect_annotations(pdf_path="D:\\nonexistent.pdf")` | Error: file not found |

---

## 10. Annotation Regression Cases

**Sub-Agent:** SA-ANNOTATION

| # | Ref | Case | Expected |
|---|-----|------|----------|
| 10.1 | REF-004 | Query `neuroncite_job_status` while an annotation job runs | `progress_total` is set immediately (= number of quotes); `progress_done` updates incrementally |
| 10.2 | REF-011 | Start annotation job -> cancel it -> start a new annotation job | The new job starts correctly and does NOT get stuck in `"queued"`. **Warning: this defect can permanently block the annotation pipeline.** |
| 10.3 | REF-012 | Annotate with a quote that does NOT exist in the PDF (`"This sentence does not exist anywhere in this document xyz123"`) | Job duration must be reasonable. For PDFs with native text, OCR fallback should be skipped or time-limited. Compare against a job with a findable quote (should take <1s). |
| 10.4 | REF-013 | Annotate a PDF whose filename contains no author (just `"Title.pdf"`) | PDF is still found via title matching |
| 10.5 | REF-014 | Annotation where no PDF is matched at all | `progress_total` in job status shows the correct value (not 0) |

---

## 11. Citation Verification Pipeline

**Sub-Agent:** SA-CITATION

| # | Case | Expected |
|---|------|----------|
| 11.1 | `neuroncite_citation_create(dry_run=true)` | Match preview with overlap_scores, cite_keys_matched/unmatched |
| 11.2 | `neuroncite_citation_create(dry_run=false, file_overrides={...})` | Job created with manual corrections for unmatched keys |
| 11.3 | `neuroncite_citation_status` | Correct pending/claimed/done counters, total_batches |
| 11.4 | `neuroncite_citation_claim` (no batch_id) | FIFO batch with rows including cite_key, matched_file_id, tex_line, section_title |
| 11.5 | `neuroncite_citation_claim(batch_id=5)` | Targeted claiming of a specific batch |
| 11.6 | `neuroncite_citation_submit(results=[...])` | rows_submitted, is_complete returned |
| 11.7 | `neuroncite_citation_rows(status="done", limit=3)` | result_json contains verdicts |
| 11.8 | Submit an already-submitted row_id | Error: `"row is done, expected claimed"` |
| 11.9 | `neuroncite_citation_status` after partial submit | Counters are consistent: done + pending = total |

---

## 11b. Citation Export (neuroncite_citation_export)

**Sub-Agent:** SA-CITATION

Run these cases after completing a full citation verification job (all batches submitted, job_status shows is_complete=true).

| # | Case | Expected |
|---|------|----------|
| 11b.1 | `neuroncite_citation_export(job_id=<DONE_JOB>, output_directory=<OUTPUT_DIR>, source_directory=<PDF_DIR>)` | Returns without error; output_directory contains all 6 output files |
| 11b.2 | Verify file list in output_directory | Files present: `annotation_pipeline_input.csv`, `citation_data.csv`, `citation_data.xlsx`, `corrections.json`, `citation_report.json`, `citation_full_detail.json` |
| 11b.3 | Inspect `citation_report.json` | Contains verdict distribution (supported/partial/unsupported/not_found counts), flagged alert list, elapsed_time |
| 11b.4 | Inspect `corrections.json` | Array sorted by line number; each entry has correction_type, original_text, suggested_text, explanation |
| 11b.5 | Inspect `citation_data.csv` | 38 columns present; rows sorted by tex_line; verdict-colored rows for supported/unsupported |
| 11b.6 | Inspect `annotation_pipeline_input.csv` | Contains title, author, quote, color columns; color values reflect verdict (green=supported, red=wrong_source, etc.) |
| 11b.7 | Inspect annotated PDFs in output_directory | PDFs created for citations with matched file_id; highlights visible |
| 11b.8 | `neuroncite_citation_export(job_id="00000000-0000-0000-0000-000000000000", ...)` | Error: `"job not found"` |
| 11b.9 | `neuroncite_citation_export` on an incomplete job (pending rows remain) | Error or warning: job not yet complete |
| 11b.10 | `neuroncite_citation_export(output_directory="D:\\nonexistent\\deep\\path", ...)` | Output directory is created, or error if creation fails |

---

## 12. Citation Regression Cases

**Sub-Agent:** SA-CITATION

| # | Ref | Case | Expected |
|---|-----|------|----------|
| 12.1 | REF-001 | Delete session while a citation job is active (`neuroncite_session_delete(session_id=...)`) | Error: `"session has active job(s)"`. Same error when deleting by directory. |
| 12.2 | REF-002 | Dry-run with a BibTeX entry whose year differs from filename (e.g. BibTeX says 2000, filename says 1997) but title is nearly identical | `matched_file_id` is NOT null; no false-positive match to a wrong PDF. Record overlap_score and whether file_overrides were needed. |

---

## 13. Session Lifecycle and Job Management

**Sub-Agent:** SA-SESSION

**Warning:** Case 13.6 requires deleting a session and re-indexing. Create a SEPARATE session_C for this test. NEVER delete session_A or session_B.

| # | Case | Expected |
|---|------|----------|
| 13.1 | `neuroncite_job_status(job_id=<EXISTING_ID>)` | Correct state information |
| 13.2 | `neuroncite_job_cancel(job_id=<RUNNING_ID>)` | Cancel succeeds |
| 13.3 | `neuroncite_jobs` | All jobs listed with correct states |
| 13.4 | `neuroncite_job_status(job_id="00000000-0000-0000-0000-000000000000")` | Error: `"job not found"` |
| 13.5 | `neuroncite_job_cancel(job_id="00000000-0000-0000-0000-000000000000")` | Error: `"job not found"` |
| 13.6 | **REF-009**: Delete session, then re-index the same directory | New session receives a new ID (not the deleted one). Check whether an old citation job still references the deleted session_id. |

---

## 14. Chunk Preview (neuroncite_preview_chunks)

**Sub-Agent:** SA-PREVIEW

| # | Case | Expected |
|---|------|----------|
| 14.1 | `neuroncite_preview_chunks(file_path=<PDF_FILE>, chunk_strategy="sentence")` | Returns first N chunks with content, page_start, page_end, word_count, byte_count; no data written to database |
| 14.2 | `neuroncite_preview_chunks(file_path=<PDF_FILE>, chunk_strategy="word", chunk_size=128, chunk_overlap=16)` | Chunks sized around 128 words with 16-word overlap |
| 14.3 | `neuroncite_preview_chunks(file_path=<PDF_FILE>, chunk_strategy="token", chunk_size=512)` | Token-based chunking with 512-token windows |
| 14.4 | `neuroncite_preview_chunks(file_path=<PDF_FILE>, chunk_strategy="page")` | One chunk per page; chunk_size/chunk_overlap ignored |
| 14.5 | `neuroncite_preview_chunks(file_path=<PDF_FILE>, chunk_strategy="sentence", limit=3)` | Exactly 3 chunks returned |
| 14.6 | `neuroncite_preview_chunks(file_path=<PDF_FILE>, chunk_strategy="sentence", limit=50)` | Up to 50 chunks returned (max limit) |
| 14.7 | `neuroncite_preview_chunks(file_path="D:\\nonexistent.pdf", chunk_strategy="sentence")` | Error: file not found |
| 14.8 | Verify database state after preview_chunks | Session count unchanged; no new files/chunks/embeddings in the database |
| 14.9 | Compare preview_chunks output with actual indexed chunks for the same file and strategy | Chunk boundaries and content match |
| 14.10 | `neuroncite_preview_chunks(file_path=<HTML_FILE>)` | Chunks from HTML content returned; heading-based sections used as pages |
| 14.11 | `neuroncite_preview_chunks(file_path=<HTML_FILE>, chunk_strategy="page")` | One chunk per HTML section; each chunk corresponds to a heading-delimited block |

---

## 15. Incremental Indexing (neuroncite_index_add)

**Sub-Agent:** SA-INCREMENTAL (uses session_B, may modify it)

| # | Case | Expected |
|---|------|----------|
| 15.1 | `neuroncite_index_add(session_id=<EXISTING>, files=[<NEW_PDF_PATH>])` | Job created; new PDF extracted, chunked, embedded, and added to session |
| 15.2 | Poll `neuroncite_job_status` until completed | Session file_count increases by 1; total_chunks and total_pages increase |
| 15.3 | `neuroncite_index_add(session_id=<EXISTING>, files=[<ALREADY_INDEXED_UNCHANGED_PDF>])` | File skipped (unchanged: mtime/size + SHA-256 match); job completes with 0 files processed |
| 15.4 | Modify a PDF file (change content/mtime) -> `neuroncite_index_add(session_id=<EXISTING>, files=[<MODIFIED_PDF>])` | File detected as changed; old chunks replaced with new chunks |
| 15.5 | `neuroncite_index_add(session_id=<EXISTING>, files=[<NEW_PDF>, <ALREADY_INDEXED_PDF>])` | Only the new PDF is processed; the unchanged one is skipped |
| 15.6 | Search the session after index_add completes | Results from the newly added file appear in search results |
| 15.7 | `neuroncite_index_add(session_id=999, files=[<PDF_PATH>])` | Error: `"session not found"` |
| 15.8 | `neuroncite_index_add(session_id=<EXISTING>, files=["D:\\nonexistent.pdf"])` | Error or warning about non-existent file |
| 15.9 | `neuroncite_index_add(session_id=<EXISTING>, files=[])` | Error or no-op (empty file list) |

---

## 16. Cross-Session Search (neuroncite_multi_search)

**Sub-Agent:** SA-CROSSSESSION (uses session_A + session_B, read-only)

| # | Case | Expected |
|---|------|----------|
| 16.1 | Index same directory twice with different chunk strategies (e.g. "sentence" and "word") -> `neuroncite_multi_search(session_ids=[<SID_1>, <SID_2>], query="<term>")` | Merged results from both sessions, sorted by score; each result tagged with source session_id |
| 16.2 | `neuroncite_multi_search(session_ids=[<SID_1>, <SID_2>], query="<term>", top_k=5)` | Exactly 5 results after merging |
| 16.3 | `neuroncite_multi_search(session_ids=[<SID_1>, <SID_2>], use_fts=false)` | All bm25_rank fields are null across both sessions |
| 16.4 | `neuroncite_multi_search(session_ids=[<SID_1>, <SID_2>], min_score=0.72)` | All results have vector_score >= 0.72 |
| 16.5 | `neuroncite_multi_search(session_ids=[<SINGLE_SID>], query="<term>")` | Error: requires 2-10 sessions |
| 16.6 | `neuroncite_multi_search(session_ids=[], query="<term>")` | Error: requires 2-10 sessions |
| 16.7 | `neuroncite_multi_search` with 11 session_ids | Error: maximum 10 sessions exceeded |
| 16.8 | `neuroncite_multi_search(session_ids=[<VALID>, 999], query="<term>")` | Error or warning about non-existent session 999 |
| 16.9 | `neuroncite_multi_search(session_ids=[<SID_1>, <SID_2>], query="")` | Error: `"query must not be empty"` |
| 16.10 | Verify that the query embedding is computed once (check response timing vs. two separate neuroncite_search calls) | multi_search latency is lower than sum of two individual searches |

---

## 17. Reranker Management (neuroncite_reranker_load)

**Sub-Agent:** SA-RERANKER

| # | Case | Expected |
|---|------|----------|
| 17.1 | `neuroncite_health` before loading reranker | `reranker_available: false` |
| 17.2 | `neuroncite_reranker_load(model_id="cross-encoder/ms-marco-MiniLM-L-6-v2")` | Reranker loaded; returns confirmation with model_id |
| 17.3 | `neuroncite_health` after loading reranker | `reranker_available: true` |
| 17.4 | `neuroncite_search(query="<term>", rerank=true)` after reranker is loaded | Results contain `reranker_score` field; ranking may differ from non-reranked search |
| 17.5 | `neuroncite_search(query="<term>", rerank=false)` after reranker is loaded | Results do NOT contain reranker_score; standard RRF ranking |
| 17.6 | `neuroncite_batch_search(queries=[...], rerank=true)` | Reranking applied to all query groups |
| 17.7 | `neuroncite_search(query="<term>", rerank=true)` before reranker is loaded | Error or warning: reranker not available |
| 17.8 | `neuroncite_reranker_load(model_id="nonexistent/model-id")` | Error: model not found in cache |
| 17.9 | `neuroncite_reranker_load(model_id="cross-encoder/ms-marco-MiniLM-L-6-v2", backend="ort")` | Explicit backend selection accepted |
| 17.10 | Load reranker -> search with rerank=true -> load different reranker -> search again | Hot-swap works; second search uses the newly loaded reranker |

---

## 18. Reranker Search Integration

**Sub-Agent:** SA-RERANKER

| # | Case | Expected |
|---|------|----------|
| 18.1 | Compare `neuroncite_search(rerank=false)` vs `neuroncite_search(rerank=true)` for same query | Both return results; reranked version may have different ordering |
| 18.2 | `neuroncite_search(rerank=true, top_k=3)` | Exactly 3 results, all with reranker_score |
| 18.3 | `neuroncite_search(rerank=true, min_score=0.82)` | min_score filtering still applied (based on vector_score, not reranker_score) |
| 18.4 | `neuroncite_search(rerank=true, refine=true)` | Both reranking and sub-chunk refinement applied simultaneously |
| 18.5 | `neuroncite_batch_search(queries=[...], rerank=true, refine=true)` | Both features applied to all queries in the batch |

---

## 19. Annotation Appearance Integrity

**Sub-Agent:** SA-ANNOTATION

These cases verify that the appearance stream injection and inline annotation promotion logic works correctly. All verification is done through `neuroncite_annotate` and `neuroncite_inspect_annotations` since the underlying Rust functions are internal and not directly callable via MCP.

| # | Case | Expected |
|---|------|----------|
| 19.1 | Annotate with an orange highlight (`#FF7F00`) -> `neuroncite_inspect_annotations` on the output PDF | `fill_color` is `#FF7F00`; annotation has non-zero bounding rect; `/AP` stream present (PDF renders visibly highlighted) |
| 19.2 | Annotate the same quote twice via `append=true` (double injection) -> inspect | `highlight_count` does NOT double; idempotent injection -- second pass adds 0 new annotations |
| 19.3 | Annotate a PDF produced by pdfium (inline /Annots dictionary format) | Annotation visible in output; `inspect_annotations` reports it correctly -- inline dicts are promoted to indirect references |
| 19.4 | Annotate with multiple quotes using different colors (`#FF0000`, `#00FF00`, `#0000FF`) on the same PDF | `unique_colors` in inspect summary contains all three hex values |
| 19.5 | Annotate with a quote on page 1 and a quote on page 3 of the same PDF | `neuroncite_inspect_annotations(page_number=1)` returns only page-1 annotations; `page_number=3` returns only page-3 annotations |
| 19.6 | Annotate with a row that has no color field (omitted from input_data) | Default yellow `#FFFF00` is applied; inspect confirms `fill_color=#FFFF00` |
| 19.7 | Annotate with a row that has no `/C` and no `/IC` color (use a PDF where the highlight annotation has neither) | Yellow default `#FFFF00` is applied; no crash |
| 19.8 | Run `neuroncite_inspect_annotations` on a PDF that has never been annotated by neuroncite | Returns `highlight_count=0`; annotation list may contain existing annotations but no neuroncite-injected ones |
| 19.9 | Annotate a PDF with a quote that spans the boundary between two pages | At least one highlight annotation appears; bounding rects are non-zero; no crash |

---

## 20. End-to-End Workflow Integration

**Sub-Agent:** SA-E2E (Phase 2, runs after all Phase 1 sub-agents complete)

These cases test the full pipeline as a connected workflow. Each case spans multiple tools. The sub-agent creates its own sessions as needed.

| # | Case | Expected |
|---|------|----------|
| 20.1 | **Full search-to-export flow:** Index PDFs -> search for a domain-specific term -> call `neuroncite_export(format="bibtex")` -> verify the exported BibTeX entries correspond to the actual top search results | BibTeX entries reference the same source files returned by search; author/title fields match PDF metadata |
| 20.2 | **Full citation workflow:** `citation_create(dry_run=true)` -> review matches -> `citation_create(dry_run=false)` -> claim batches -> submit all results -> `citation_export` -> open `citation_report.json` | All batches complete; is_complete=true; 6 output files present; verdict counts in report match the number of submitted rows |
| 20.3 | **Incremental index + search consistency:** Index directory -> record search result scores for a query -> `index_add` with a new PDF -> run same query again | New file's content appears in results; previously returned results are still present; scores for unchanged results are stable |
| 20.4 | **Annotation + inspect round-trip:** Run annotation job with 3 quotes across 2 PDFs -> `inspect_annotations` on each output PDF -> compare highlight_count against the number of quotes that matched each PDF | highlight_count per PDF matches the number of successfully matched quotes for that PDF |
| 20.5 | **Session delete impact on dependent tools:** Create session -> run citation job -> verify citation_status -> delete session -> attempt `neuroncite_search(session_id=<DELETED>)` and `neuroncite_citation_status(job_id=<JOB>)` | Search returns "session not found"; citation job status reflects that the session is gone |
| 20.6 | **Reranker + batch_search full flow:** Load reranker -> run `batch_search` with 3 queries using `rerank=true, refine=true` -> verify all result groups contain `reranker_score` and sub-chunk refined content | All 3 query groups present; each result has reranker_score and non-empty content; no query group has error_count > 0 |
| 20.7 | **HTML fetch -> index -> search -> content:** `neuroncite_html_fetch(url=<PUBLIC_URL>)` -> `neuroncite_index(urls=[<URL>])` -> `neuroncite_search(query=<TERM>)` -> `neuroncite_content(file_id=<RESULT_FILE_ID>, part=1)` | Content returned with `source_type="html"` and `web_source` metadata (url, title, domain); search finds HTML content |
| 20.8 | **BibTeX source acquisition flow:** `neuroncite_bib_report(bib_path=<BIB>, output_directory=<DIR>)` -> review missing entries -> `neuroncite_citation_fetch_sources(bib_path=<BIB>, session_id=<SID>, output_directory=<DIR>)` -> `neuroncite_bib_report` again | Second bib_report shows previously-missing entries as `status="exists"`; `missing_count` decreased |

---

## 21. Batch Content Retrieval (neuroncite_batch_content)

**Sub-Agent:** SA-CONTENT

Retrieves content parts from multiple indexed documents in a single call. Uses the same "part" concept as `neuroncite_content` (page for PDFs, section for HTML). Partial failure model: individual request errors are reported inline without failing the entire batch.

| # | Case | Expected |
|---|------|----------|
| 21.1 | `neuroncite_batch_content(requests=[{file_id:<ID>, part:1}])` | Single result with content, file_id, request_index=0 |
| 21.2 | `neuroncite_batch_content` with 3 requests mixing single parts and ranges | All 3 results returned; `total_parts_returned` equals sum of requested parts |
| 21.3 | `neuroncite_batch_content` with 10 requests (maximum allowed) | All 10 results returned; `request_count=10` |
| 21.4 | `neuroncite_batch_content` with 11 requests | Error: `"batch exceeds maximum of 10 requests"` |
| 21.5 | `neuroncite_batch_content` with total parts > 20 across all requests | Error: `"total parts across all requests ... exceeds maximum of 20"` |
| 21.6 | `neuroncite_batch_content` with one invalid file_id among valid requests | Partial failure: invalid file gets inline `"error"` field; other results are returned normally |
| 21.7 | `neuroncite_batch_content(requests=[{file_id:<ID>, start:1, end:3}])` | Returns 3 parts with `part_count=3`; parts array ordered by part number |
| 21.8 | `neuroncite_batch_content` with `part=0` in a request | Error: `"part must be >= 1"` |
| 21.9 | `neuroncite_batch_content` with `end < start` in a request | Error: `"start must be <= end"` |
| 21.10 | `neuroncite_batch_content` with both `part` and `start` set in same request | Error about mutual exclusivity |
| 21.11 | `neuroncite_batch_content(requests=[])` | Error: `"requests array must not be empty"` |
| 21.12 | `neuroncite_batch_content` with an HTML file_id in one request | Result for that request includes `web_source` metadata (url, title, domain) |

---

## 22. Search Quality Comparison (neuroncite_compare_search)

**Sub-Agent:** SA-CROSSSESSION

| # | Case | Expected |
|---|------|----------|
| 22.1 | Index same directory twice with different chunk strategies (e.g. "sentence" and "word") -> `neuroncite_compare_search(session_id_a=<SID_1>, session_id_b=<SID_2>, query="<term>")` | Returns session_a and session_b blocks each with result_count, total_candidates, avg_vector_score, and results array |
| 22.2 | `neuroncite_compare_search(top_k=3)` | Each session block returns at most 3 results |
| 22.3 | `neuroncite_compare_search(session_id_a=<ID>, session_id_b=<ID>)` with identical session IDs | Error: `"session_id_a and session_id_b must be different sessions"` |
| 22.4 | `neuroncite_compare_search` with sessions indexed by different embedding models (different vector dimensions) | Error about incompatible vector dimensions |
| 22.5 | `neuroncite_compare_search(session_id_a=999, session_id_b=<VALID>)` | Error: `"session 999 not found"` |
| 22.6 | `neuroncite_compare_search(query="")` | Error: `"query must not be empty"` |
| 22.7 | `neuroncite_compare_search(top_k=21)` | Error: `"top_k must be between 1 and 20"` |
| 22.8 | Verify query embedding is computed once: response latency for compare_search should be lower than sum of two separate neuroncite_search calls | Response time consistent with single-embedding execution |

---

## 23. Single-File Reindexing (neuroncite_reindex_file)

**Sub-Agent:** SA-INCREMENTAL (uses session_B, may modify it)

| # | Case | Expected |
|---|------|----------|
| 23.1 | `neuroncite_reindex_file(session_id=<ID>, file_path=<KNOWN_PDF>)` for a file already indexed in the session | Returns action="reindexed"; chunks_created > 0; previous_chunks is non-null; hnsw_rebuilt=true |
| 23.2 | `neuroncite_reindex_file` for a file NOT previously indexed in the session | Returns action="added"; previous_chunks=null; file appears in subsequent neuroncite_files listing |
| 23.3 | Search the session after reindex_file -> verify updated content appears in results | Queries targeting content unique to the re-indexed PDF return that file in results |
| 23.4 | `neuroncite_reindex_file(session_id=999, file_path=<VALID>)` | Error: `"session 999 not found"` |
| 23.5 | `neuroncite_reindex_file(session_id=<ID>, file_path="D:\\nonexistent.pdf")` | Error: `"file does not exist"` |
| 23.6 | `neuroncite_reindex_file` when session vector dimension differs from loaded model | Error about vector dimension mismatch |
| 23.7 | `neuroncite_reindex_file` on a PDF that produces zero chunks (empty/image-only file) | Error: `"extraction produced zero chunks"` |
| 23.8 | Verify session file_count after reindex_file on an existing file | file_count does NOT increase (file replaced, not added); total_chunks may differ if chunk count changed |
| 23.9 | Verify session file_count after reindex_file on a new file | file_count increases by 1 |

---

## 24. Citation Retry (neuroncite_citation_retry)

**Sub-Agent:** SA-CITATION

Resets specific citation rows to `pending` status so they can be re-claimed and re-verified. Supports filtering by row IDs, flag values (warning/critical), or batch IDs. Requires a completed or partially completed citation job.

| # | Case | Expected |
|---|------|----------|
| 24.1 | `neuroncite_citation_retry(job_id=<JOB>)` with no optional parameters on a job that has failed rows | All rows with status `failed` are reset to `pending`; returns `rows_reset` count and `batches_affected` count |
| 24.2 | `neuroncite_citation_retry(job_id=<JOB>, row_ids=[<ROW_1>, <ROW_2>])` | Only the specified rows are reset; other rows unchanged |
| 24.3 | `neuroncite_citation_retry(job_id=<JOB>, flags=["warning"])` | All `done` rows with `flag="warning"` are reset to `pending`; rows with `flag="critical"` or no flag are unchanged |
| 24.4 | `neuroncite_citation_retry(job_id=<JOB>, flags=["warning", "critical"])` | All `done` rows with either flag value are reset |
| 24.5 | `neuroncite_citation_retry(job_id=<JOB>, batch_ids=[<BATCH_1>])` | All `failed` rows in the specified batch are reset |
| 24.6 | `neuroncite_citation_retry(job_id=<JOB>, row_ids=[<ROW>], flags=["warning"])` | Both filters applied; totals summed across both reset operations |
| 24.7 | `neuroncite_citation_retry(job_id=<JOB>, row_ids=[])` | Returns `rows_reset=0, batches_affected=0` (empty slice is a no-op) |
| 24.8 | `neuroncite_citation_retry(job_id=<JOB>, row_ids=[99999])` | Returns `rows_reset=0` (non-existent row ID) |
| 24.9 | `neuroncite_citation_retry(job_id="00000000-0000-0000-0000-000000000000")` | Error: `"job not found"` |
| 24.10 | After retry: `neuroncite_citation_status` shows reset rows as `pending`, job `is_complete=false` | Counters reflect the reset; pending count increased by the number of reset rows |
| 24.11 | After retry: `neuroncite_citation_claim` returns a batch containing the reset rows | Reset rows are claimable again; `result_json` fields are cleared |

---

## 25. Text Search (neuroncite_text_search)

**Sub-Agent:** SA-PREVIEW

Searches for literal substring occurrences within the extracted text of an indexed PDF file. Returns matching pages with character positions and surrounding context. Does not use embeddings or FTS5.

| # | Case | Expected |
|---|------|----------|
| 25.1 | `neuroncite_text_search(file_id=<ID>, query="<known term>")` | Returns pages with matches; each match has `position` (character offset) and `context` (surrounding text) |
| 25.2 | `neuroncite_text_search(file_id=<ID>, query="<UPPERCASE_TERM>", case_sensitive=false)` | Matches regardless of case; default behavior |
| 25.3 | `neuroncite_text_search(file_id=<ID>, query="<UPPERCASE_TERM>", case_sensitive=true)` | Only matches exact case; fewer or zero results compared to case-insensitive search |
| 25.4 | `neuroncite_text_search(file_id=<ID>, query="xyznonexistent123")` | Returns `total_matches=0, pages_with_matches=0, results=[]` |
| 25.5 | `neuroncite_text_search(file_id=<ID>, query="the")` for a page with multiple occurrences | `results[].matches` array contains one entry per non-overlapping occurrence on that page |
| 25.6 | `neuroncite_text_search(file_id=99999, query="term")` | Error: `"file not found"` |
| 25.7 | `neuroncite_text_search(file_id=<ID>, query="")` | Error: `"query must not be empty"` |
| 25.8 | Verify context boundaries: search a term near the start of a page | Context string does not extend before position 0; no index-out-of-bounds |

---

## 26. Session Diff (neuroncite_session_diff)

**Sub-Agent:** SA-CROSSSESSION

Compares two index sessions and classifies files into four categories: only in session A, only in session B, identical (same path and hash), and changed (same path, different hash or page count).

| # | Case | Expected |
|---|------|----------|
| 26.1 | `neuroncite_session_diff(session_a=<SID_1>, session_b=<SID_2>)` where both index the same directory with same strategy | `only_in_a=[], only_in_b=[], changed=[]`; all files in `identical` list |
| 26.2 | `neuroncite_session_diff` where session B has one additional file (via `index_add`) | That file appears in `only_in_b`; other files in `identical` |
| 26.3 | `neuroncite_session_diff` where a file was modified between indexing sessions | File appears in `changed` with `hash_changed=true` |
| 26.4 | `neuroncite_session_diff(session_a=<SID>, session_b=<SID>)` (same session) | All files in `identical`; `only_in_a=[], only_in_b=[], changed=[]` |
| 26.5 | `neuroncite_session_diff(session_a=999, session_b=<VALID>)` | Error: `"session 999 not found"` |
| 26.6 | `neuroncite_session_diff(session_a=<VALID>, session_b=999)` | Error: `"session 999 not found"` |
| 26.7 | Verify `summary` field contains correct counts: `total_in_a`, `total_in_b`, `only_in_a`, `only_in_b`, `identical`, `changed` | All counts consistent with the detail arrays |

---

## 27. Annotation Removal (neuroncite_annotation_remove)

**Sub-Agent:** SA-ANNOTATION

Removes highlight annotations from a PDF file. Operates via lopdf (pure Rust) without pdfium. Supports three removal modes: all highlights, by hex color, or by page number.

| # | Case | Expected |
|---|------|----------|
| 27.1 | Annotate a PDF with highlights -> `neuroncite_annotation_remove(pdf_path=<ANNOTATED>, mode="all")` -> `neuroncite_inspect_annotations` | `highlight_count=0`; all highlights removed |
| 27.2 | Annotate with red (`#FF0000`) and yellow (`#FFFF00`) highlights -> `neuroncite_annotation_remove(mode="by_color", colors=["#FF0000"])` -> inspect | Only yellow highlights remain; red removed |
| 27.3 | Annotate pages 1 and 3 -> `neuroncite_annotation_remove(mode="by_page", pages=[1])` -> inspect | Page 1 highlights removed; page 3 highlights intact |
| 27.4 | `neuroncite_annotation_remove(pdf_path=<ANNOTATED>, mode="all", dry_run=true)` -> inspect original | `annotations_removed` count returned; source PDF unchanged (verify via inspect) |
| 27.5 | `neuroncite_annotation_remove(pdf_path=<SOURCE>, output_path=<SEPARATE>, mode="all")` | Source PDF unchanged; output PDF has highlights removed |
| 27.6 | `neuroncite_annotation_remove(mode="by_color", colors=["#123456"])` where no highlight has that color | `annotations_removed=0`; PDF unchanged |
| 27.7 | `neuroncite_annotation_remove(pdf_path="D:\\nonexistent.pdf", mode="all")` | Error: `"pdf_path is not a file or does not exist"` |
| 27.8 | `neuroncite_annotation_remove(pdf_path=<VALID>, mode="invalid_mode")` | Error: `"invalid mode"` |
| 27.9 | `neuroncite_annotation_remove(mode="by_color")` without `colors` parameter | Error: `"mode 'by_color' requires a 'colors' array parameter"` |
| 27.10 | `neuroncite_annotation_remove(mode="by_page")` without `pages` parameter | Error: `"mode 'by_page' requires a 'pages' array parameter"` |

---

## 28. Annotation Status (neuroncite_annotate_status)

**Sub-Agent:** SA-ANNOTATION

Returns per-quote progress during an annotation pipeline execution. Each quote from the input data has a status (pending, matched, not_found, error), match method, page number, and matched PDF filename.

| # | Case | Expected |
|---|------|----------|
| 28.1 | Start annotation job -> immediately call `neuroncite_annotate_status(job_id=<JOB>)` | Returns `status_counts` with `pending` equal to total quotes; `quotes` array lists all quotes with `status="pending"` |
| 28.2 | Poll `neuroncite_annotate_status` during a running annotation job | `status_counts` update as quotes are processed; `matched` count increases |
| 28.3 | `neuroncite_annotate_status` after job completes | `status_counts` shows final distribution; no `pending` quotes remain |
| 28.4 | `neuroncite_annotate_status(job_id=<JOB>, limit=2, offset=0)` | Returns at most 2 quote rows; pagination works |
| 28.5 | `neuroncite_annotate_status(job_id=<JOB>, limit=2, offset=2)` | Returns next 2 quote rows; different from offset=0 |
| 28.6 | `neuroncite_annotate_status(job_id="00000000-0000-0000-0000-000000000000")` | Error: `"job not found"` |
| 28.7 | `neuroncite_annotate_status` on a citation job (kind != "annotate") | Error: `"job is not an annotation job"` |
| 28.8 | Verify matched quotes have `match_method` set (e.g. "exact", "normalized", "fuzzy") | Non-null match_method for all `status="matched"` rows |
| 28.9 | Verify matched quotes have `page` and `pdf_filename` set | Non-null page and pdf_filename for all `status="matched"` rows |
| 28.10 | Delete the annotation job -> `neuroncite_annotate_status` | Error: `"job not found"`; cascade delete removes all quote status rows |

---

## 29. HTML Page Fetching (neuroncite_html_fetch)

**Sub-Agent:** SA-HTML

Fetches one or more web pages via HTTP GET, caches raw HTML to disk (SHA-256-based filename in the default HTML cache directory), and returns metadata per URL. Applies SSRF protection that rejects private networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16), loopback, link-local (169.254.0.0/16 including cloud metadata endpoints), and non-HTTP schemes (file://, ftp://).

| # | Case | Expected |
|---|------|----------|
| 29.1 | `neuroncite_html_fetch(url="<VALID_PUBLIC_URL>")` | `fetched=1`; result has `cache_path`, `http_status`, `title`, `domain`, `content_type`, `html_bytes` |
| 29.2 | `neuroncite_html_fetch(urls=["<URL_1>","<URL_2>"])` | `fetched=2`; results array has 2 entries with metadata per URL |
| 29.3 | `neuroncite_html_fetch(url="<URL>", urls=["<URL>"])` | Error: `url` and `urls` are mutually exclusive |
| 29.4 | `neuroncite_html_fetch()` with neither `url` nor `urls` | Error: at least one of `url` or `urls` required |
| 29.5 | `neuroncite_html_fetch(urls=[])` | Error: empty `urls` array |
| 29.6 | `neuroncite_html_fetch(url="http://192.168.1.1")` | Error: SSRF rejected (private IP) |
| 29.7 | `neuroncite_html_fetch(url="http://127.0.0.1")` | Error: SSRF rejected (loopback) |
| 29.8 | `neuroncite_html_fetch(url="http://169.254.169.254")` | Error: SSRF rejected (cloud metadata endpoint) |
| 29.9 | `neuroncite_html_fetch(url="file:///etc/passwd")` | Error: SSRF rejected (non-HTTP scheme) |
| 29.10 | `neuroncite_html_fetch(urls=["<VALID_URL>","http://192.168.1.1"])` | Partial batch: `fetched=1, failed=1`; valid URL has metadata, SSRF URL has inline `"error"` field |
| 29.11 | `neuroncite_html_fetch(url="<URL>", strip_boilerplate=false)` | Content fetched without readability-based boilerplate removal |
| 29.12 | `neuroncite_html_fetch(url="<URL>", delay_ms=100)` | Custom inter-request delay accepted without error |

---

## 30. Website Crawling (neuroncite_html_crawl)

**Sub-Agent:** SA-HTML

Crawls a website starting from a seed URL using either breadth-first link-following or sitemap-based URL discovery. Each discovered page is fetched and cached. Applies the same SSRF protection as `neuroncite_html_fetch` to both the start URL and every discovered URL.

| # | Case | Expected |
|---|------|----------|
| 30.1 | `neuroncite_html_crawl(start_url="<URL>", max_depth=0)` | `pages_fetched=1` (start URL only, no link following) |
| 30.2 | `neuroncite_html_crawl(start_url="<URL>", max_depth=1)` | `pages_fetched > 1` (follows links one level deep from start page) |
| 30.3 | `neuroncite_html_crawl(start_url="<URL>", max_pages=3)` | `pages_fetched <= 3` (stops after maximum page limit) |
| 30.4 | `neuroncite_html_crawl(start_url="<URL>", same_domain_only=true)` | All results have same domain as start_url |
| 30.5 | `neuroncite_html_crawl(start_url="<URL>", url_pattern=".*docs.*")` | Only URLs matching the regex pattern are fetched |
| 30.6 | `neuroncite_html_crawl(start_url="<URL>", use_sitemap=true)` | Discovery uses sitemap.xml instead of link-following; `use_sitemap=true` in response |
| 30.7 | `neuroncite_html_crawl()` without `start_url` | Error: missing required parameter `start_url` |
| 30.8 | `neuroncite_html_crawl(start_url="http://192.168.1.1")` | Error: SSRF rejected (private IP) |
| 30.9 | `neuroncite_html_crawl(start_url="not-a-valid-url")` | Error: invalid URL format |
| 30.10 | `neuroncite_html_crawl(start_url="<URL>", strip_boilerplate=false)` | Boilerplate preserved in cached pages; metadata extraction skips readability filtering |

---

## 31. Citation Source Acquisition (neuroncite_citation_fetch_sources)

**Sub-Agent:** SA-CITATION

Reads a BibTeX file, extracts URL/DOI fields from each entry, resolves DOIs through a multi-source chain (Unpaywall -> Semantic Scholar -> OpenAlex -> doi.org), classifies each URL as PDF or HTML, downloads PDFs to `output_directory/pdf/`, fetches HTML pages to `output_directory/html/`, indexes all acquired files into the specified session, and rebuilds the HNSW index. Skip logic: checks whether a file for each entry already exists in the output directory before downloading. Bot-detection: HTML pages with fewer than 50 words that match known blocked patterns (Cloudflare, access-denied, etc.) are classified as `"blocked"` and excluded from indexing.

| # | Case | Expected |
|---|------|----------|
| 31.1 | `neuroncite_citation_fetch_sources(bib_path=<BIB>, session_id=<SID>, output_directory=<DIR>)` with a valid .bib file containing URL fields | `pdfs_downloaded > 0`; `output_directory/pdf/` and `output_directory/html/` subdirectories created; `hnsw_rebuilt=true` |
| 31.2 | Missing `bib_path` parameter | Error: `"missing required parameter: bib_path"` |
| 31.3 | Missing `session_id` parameter | Error: `"missing required parameter: session_id"` |
| 31.4 | Missing `output_directory` parameter | Error: `"missing required parameter: output_directory"` |
| 31.5 | Nonexistent `bib_path` | Error: `"bib file does not exist"` |
| 31.6 | `session_id=999` (non-existent session) | Error: `"session N not found"` |
| 31.7 | BibTeX entry with explicit `url` field | Uses URL directly; `doi_resolved_via=null` in result |
| 31.8 | BibTeX entry with only `doi` field (no `url`) | Triggers DOI resolution chain; `doi_resolved_via` field populated (e.g. `"semantic_scholar"`, `"unpaywall"`) |
| 31.9 | BibTeX entry with neither `url` nor `doi` | Entry skipped (not present in results or shown with no_url status) |
| 31.10 | Run with a file that was already downloaded in a previous call | `status="skipped"`; file not re-downloaded |
| 31.11 | HTML page blocked by Cloudflare or similar bot-detection | `status="blocked"`; `reason` contains matched pattern description; file excluded from indexing |
| 31.12 | `email` parameter provided | Unpaywall resolution attempted first in the DOI chain |
| 31.13 | `delay_ms=2000` (custom value) | Accepted without error; controls inter-request delay |
| 31.14 | Session vector dimension mismatch with loaded model | Error about vector dimension conflict |
| 31.15 | `neuroncite_search` on the session after `citation_fetch_sources` completes | Newly indexed content from downloaded sources appears in search results |
| 31.16 | Verify `hnsw_rebuilt=true` in response | HNSW index rebuilt when at least one file was indexed into the session |

---

## 32. BibTeX Report (neuroncite_bib_report)

**Sub-Agent:** SA-CITATION

Parses a BibTeX file and generates two report files in the output directory: `bib_report.csv` (UTF-8 BOM, 9 columns) and `bib_report.xlsx` (formatted header row, status-colored rows, summary sheet). For each entry, checks whether a matching file already exists in the output directory and its `pdf/`/`html/` subdirectories. Status values: `"exists"`, `"missing"` (has URL/DOI but file not found), `"no_link"` (no URL or DOI). Link type values: `"URL"`, `"DOI"`, `""`.

| # | Case | Expected |
|---|------|----------|
| 32.1 | `neuroncite_bib_report(bib_path=<BIB>, output_directory=<DIR>)` | `bib_report.csv` and `bib_report.xlsx` created; response contains `csv_path`, `xlsx_path`, `total_entries`, `existing_count`, `missing_count`, `no_link_count` |
| 32.2 | Missing `bib_path` parameter | Error: `"missing required parameter: bib_path"` |
| 32.3 | Missing `output_directory` parameter | Error: `"missing required parameter: output_directory"` |
| 32.4 | Nonexistent `bib_path` | Error: `"bib file does not exist"` |
| 32.5 | Inspect `bib_report.csv` | 9 columns present: cite_key, title, author, year, link_type, status, url, doi, existing_file; rows sorted by cite_key; UTF-8 BOM at start of file |
| 32.6 | Inspect `bib_report.xlsx` | "BibTeX Report" sheet with data rows and formatted dark blue header row; "Summary" sheet with generation timestamp, total_entries, existing_count, missing_count, no_link_count |
| 32.7 | BibTeX entry with explicit `url` field | `link_type="URL"` in report row |
| 32.8 | BibTeX entry with only `doi` field | `link_type="DOI"` in report row |
| 32.9 | BibTeX entry with neither `url` nor `doi` | `link_type=""`, `status="no_link"` in report row |
| 32.10 | Run after `neuroncite_citation_fetch_sources` on the same output_directory | Previously downloaded files show `status="exists"` with filename in `existing_file` column |
| 32.11 | `output_directory` does not exist | Directory created automatically; report files written inside |
| 32.12 | Re-run `neuroncite_bib_report` on the same output_directory | Existing report files overwritten with fresh data |
