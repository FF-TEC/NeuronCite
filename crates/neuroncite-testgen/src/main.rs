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

// Entry point for the synthetic test PDF generator.
//
// Generates three PDF files into tests/PDFs/ocr/ that replace copyrighted
// academic papers with self-created fixtures. Each PDF triggers a specific
// failure mode in the pdf-extract crate while remaining parseable (or
// intentionally unparseable) by the pdfium fallback backend.
//
// Usage: cargo run -p neuroncite-testgen
//
// Output files:
//   tests/PDFs/ocr/cross-reference-error/synthetic_statistics_xref_corruption.pdf
//   tests/PDFs/ocr/cid-range-panic/synthetic_finance_cid_range.pdf
//   tests/PDFs/ocr/type1-font-panic/synthetic_forecasting_type1_encoding.pdf

mod cid_range_panic;
mod type1_font_panic;
mod xref_corruption;

use std::path::PathBuf;

/// Mapping of subfolder name to output filename for each synthetic PDF.
/// The subfolder names match the existing test infrastructure in
/// crates/neuroncite/tests/integration/ocr_fallback.rs (OCR_PDF_ENTRIES).
const FIXTURES: [(&str, &str); 3] = [
    (
        "cross-reference-error",
        "synthetic_statistics_xref_corruption.pdf",
    ),
    ("cid-range-panic", "synthetic_finance_cid_range.pdf"),
    (
        "type1-font-panic",
        "synthetic_forecasting_type1_encoding.pdf",
    ),
];

fn main() {
    // Resolve the workspace root from the crate's manifest directory.
    // CARGO_MANIFEST_DIR points to crates/neuroncite-testgen/, so the
    // workspace root is two levels up.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("failed to resolve workspace root from CARGO_MANIFEST_DIR");

    let ocr_dir = workspace_root.join("tests").join("PDFs").join("ocr");

    // Create output directories if they do not exist.
    for (subfolder, _) in &FIXTURES {
        let dir = ocr_dir.join(subfolder);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| {
            panic!("failed to create directory {}: {e}", dir.display());
        });
    }

    // Generate each synthetic PDF.
    let xref_path = ocr_dir.join(FIXTURES[0].0).join(FIXTURES[0].1);
    xref_corruption::generate(&xref_path);
    report(&xref_path, "cross-reference-error (both backends fail)");

    let cid_path = ocr_dir.join(FIXTURES[1].0).join(FIXTURES[1].1);
    cid_range_panic::generate(&cid_path);
    report(
        &cid_path,
        "cid-range-panic (pdf-extract panics, pdfium succeeds)",
    );

    let type1_path = ocr_dir.join(FIXTURES[2].0).join(FIXTURES[2].1);
    type1_font_panic::generate(&type1_path);
    report(
        &type1_path,
        "type1-font-panic (pdf-extract panics, pdfium succeeds)",
    );

    println!("\nAll 3 synthetic test PDFs generated.");
}

/// Prints the file path and size of a generated PDF.
fn report(path: &PathBuf, description: &str) {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    println!(
        "  {} ({}) -- {} bytes",
        path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default(),
        description,
        size,
    );
}
