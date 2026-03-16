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

// Benchmark for PDF file discovery and text quality scoring.
//
// Measures the performance of recursive directory traversal (discover_pdfs)
// and text quality heuristic evaluation (check_text_quality) from the
// neuroncite-pdf crate. Uses temporary directories populated with small
// placeholder PDF files for the discovery benchmark, and sample text strings
// for the quality scoring benchmark.

use criterion::{Criterion, criterion_group, criterion_main};
use neuroncite_pdf::discover_pdfs;
use neuroncite_pdf::quality::check_text_quality;
use std::fs;
use tempfile::TempDir;

/// Number of PDF files created in the temporary directory for the
/// discovery benchmark. Distributed across 5 subdirectories (10 files each).
const NUM_PDF_FILES: usize = 50;

/// Number of subdirectories in the temporary directory tree.
const NUM_SUBDIRS: usize = 5;

/// Creates a temporary directory tree with NUM_PDF_FILES placeholder PDF
/// files distributed across NUM_SUBDIRS subdirectories. Returns the TempDir
/// handle (which deletes the directory on drop) and the root path.
fn create_pdf_directory() -> TempDir {
    let tmp = TempDir::new().expect("failed to create temp directory");
    let root = tmp.path();

    // Create subdirectory structure.
    for dir_idx in 0..NUM_SUBDIRS {
        let subdir = root.join(format!("subdir_{dir_idx:02}"));
        fs::create_dir_all(&subdir).expect("failed to create subdirectory");

        // Create PDF files in each subdirectory.
        let files_per_dir = NUM_PDF_FILES / NUM_SUBDIRS;
        for file_idx in 0..files_per_dir {
            let file_name = format!("document_{dir_idx:02}_{file_idx:02}.pdf");
            let file_path = subdir.join(file_name);
            // Write a minimal placeholder. The discovery function only checks
            // file extensions, not file content validity.
            fs::write(&file_path, b"%PDF-1.4 placeholder").expect("failed to write PDF file");
        }
    }

    // Also create some non-PDF files to exercise the extension filter.
    fs::write(root.join("readme.txt"), b"not a pdf").expect("failed to write txt");
    fs::write(root.join("data.csv"), b"col1,col2").expect("failed to write csv");
    fs::write(root.join("notes.docx"), b"not a pdf either").expect("failed to write docx");

    tmp
}

/// Measures the latency of discovering all PDF files in a directory tree
/// containing 50 PDFs across 5 subdirectories and 3 non-PDF files.
fn bench_discover_pdfs(c: &mut Criterion) {
    let tmp = create_pdf_directory();

    c.bench_function("discover_pdfs_50files_5dirs", |b| {
        b.iter(|| {
            let paths = discover_pdfs(tmp.path()).expect("discovery failed");
            assert_eq!(paths.len(), NUM_PDF_FILES);
        });
    });
}

/// Measures the throughput of the text quality scoring function on a sample
/// paragraph of well-formed English text (all thresholds pass).
fn bench_quality_check_passing(c: &mut Criterion) {
    let text = "Statistical inference provides a framework for drawing conclusions \
                about population parameters from sample data. The central limit theorem \
                states that the distribution of sample means approaches a normal distribution \
                as the sample size increases, regardless of the population distribution shape. \
                Hypothesis testing evaluates whether observed data provide sufficient evidence \
                to reject a null hypothesis at a specified significance level. Confidence \
                intervals quantify the uncertainty in parameter estimates by defining a range \
                of plausible values derived from the sampling distribution.";

    c.bench_function("text_quality_check_passing", |b| {
        b.iter(|| {
            let quality = check_text_quality(text);
            assert!(quality.passes_all);
        });
    });
}

/// Measures the throughput of the text quality scoring function on garbled
/// text that fails the alphabetic ratio threshold.
fn bench_quality_check_failing(c: &mut Criterion) {
    // Construct text dominated by digits and punctuation with minimal
    // alphabetic characters, triggering a failing alphabetic ratio.
    let garbled: String = (0..500)
        .map(|i| {
            if i % 20 == 0 {
                'a'
            } else if i % 3 == 0 {
                '.'
            } else {
                char::from(b'0' + (i % 10) as u8)
            }
        })
        .collect();

    c.bench_function("text_quality_check_failing", |b| {
        b.iter(|| {
            let quality = check_text_quality(&garbled);
            assert!(!quality.passes_all);
        });
    });
}

/// Measures the throughput of the quality scoring function on a longer text
/// block (approximately 2000 characters) to observe scaling behavior. The
/// text uses multiple distinct sentences to maintain a unique token ratio
/// above the 0.05 threshold. A single repeated paragraph would drop the
/// unique ratio below the threshold because the token count grows while
/// the unique count stays constant.
fn bench_quality_check_long_text(c: &mut Criterion) {
    let paragraphs = [
        "The analysis of variance decomposes total variability into components attributable to different sources of variation. ",
        "Bayesian inference provides a principled framework for updating beliefs given observed evidence and prior distributions. ",
        "Principal component analysis reduces dimensionality by projecting data onto orthogonal axes that capture maximum variance. ",
        "Gradient descent iteratively adjusts model parameters by computing partial derivatives of the loss function with respect to each weight. ",
        "Regularization techniques such as ridge and lasso penalties prevent overfitting by constraining the magnitude of learned coefficients. ",
        "Cross-validation partitions the dataset into training and validation folds to estimate generalization performance without holdout bias. ",
        "Kernel methods implicitly map features into higher-dimensional spaces where linear separability becomes achievable for complex boundaries. ",
        "Bootstrap resampling constructs empirical sampling distributions by drawing repeated samples with replacement from the observed dataset. ",
        "Markov chain Monte Carlo algorithms generate correlated draws from posterior distributions through carefully designed transition kernels. ",
        "Expectation maximization alternates between computing expected sufficient statistics and maximizing the complete-data log-likelihood. ",
    ];
    let long_text: String = paragraphs.to_vec().join("");

    c.bench_function("text_quality_check_2000chars", |b| {
        b.iter(|| {
            let quality = check_text_quality(&long_text);
            assert!(quality.passes_all);
        });
    });
}

criterion_group!(
    benches,
    bench_discover_pdfs,
    bench_quality_check_passing,
    bench_quality_check_failing,
    bench_quality_check_long_text
);
criterion_main!(benches);
