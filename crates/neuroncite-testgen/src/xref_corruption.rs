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

// Generator for a synthetic PDF with severely corrupted cross-reference table.
//
// This PDF triggers failure in BOTH pdf-extract and pdfium:
// - pdf-extract: lopdf::Document::load_mem() returns Err because the xref
//   byte offsets point to invalid locations.
// - pdfium: load_pdf_from_file() returns FormatError because the triple
//   corruption (xref entries + startxref pointer + trailer /Size) eliminates
//   all anchor points that pdfium's xref repair algorithm relies on.
//
// The generated file replaces the copyrighted "OpenIntro Statistics" PDF
// that was previously used in tests/PDFs/ocr/cross-reference-error/.

use std::path::Path;

use lopdf::content::{Content, Operation};
use lopdf::dictionary;
use lopdf::{Document, Object, Stream};

/// Builds a valid multi-page PDF with statistics domain text, then corrupts
/// the cross-reference table so that no PDF parser can recover the document.
/// Writes the corrupted bytes to `output_path`.
pub fn generate(output_path: &Path) {
    let mut doc = Document::with_version("1.4");

    let pages_id = doc.new_object_id();

    // Helvetica font reference (standard 14 font, no embedding required).
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });

    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! {
            "F1" => font_id,
        },
    });

    // Three pages with statistics domain text. Each page contains enough
    // content to verify the PDF was a real document before corruption.
    let page_texts = [
        "Introduction to probability and statistics covering hypothesis testing \
         confidence intervals and regression analysis for observational studies",
        "Linear regression analysis with ordinary least squares estimation \
         residual diagnostics and heteroskedasticity tests for valid inference",
        "Probability distributions including normal chi-squared and t-distribution \
         form the foundation of classical hypothesis testing and sampling theory",
    ];

    let mut page_ids = Vec::new();
    for text in &page_texts {
        let content = Content {
            operations: vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 11.into()]),
                Operation::new("Td", vec![72.into(), 720.into()]),
                Operation::new("Tj", vec![Object::string_literal(*text)]),
                Operation::new("ET", vec![]),
            ],
        };

        let content_id = doc.add_object(Stream::new(
            dictionary! {},
            content.encode().expect("content encoding failed"),
        ));

        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Contents" => content_id,
            "Resources" => resources_id,
        });
        page_ids.push(page_id);
    }

    let pages_dict = dictionary! {
        "Type" => "Pages",
        "Kids" => page_ids.iter().map(|id| Object::Reference(*id)).collect::<Vec<Object>>(),
        "Count" => page_ids.len() as i64,
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    // Serialize the valid PDF to bytes.
    let mut buf = Vec::new();
    doc.save_to(&mut buf).expect("PDF serialization failed");

    // Corrupt the cross-reference table in three independent locations.
    // All three corruptions are required to defeat pdfium's xref repair.
    corrupt_xref_offsets(&mut buf);
    corrupt_startxref_pointer(&mut buf);
    corrupt_trailer_size(&mut buf);

    std::fs::write(output_path, &buf).expect("failed to write corrupted PDF");
}

/// Overwrites every 10-digit byte offset in the xref table with 9999999999.
/// After this, no object can be located via the xref table because every
/// offset points past the end of the file.
fn corrupt_xref_offsets(buf: &mut [u8]) {
    let xref_marker = b"xref";
    let Some(xref_pos) = find_subsequence(buf, xref_marker) else {
        return;
    };

    // Scan forward from the xref marker. Each xref entry is exactly 20 bytes:
    // "NNNNNNNNNN GGGGG X \n" where N=offset, G=generation, X=f|n
    let mut i = xref_pos + xref_marker.len();

    // Skip whitespace and the subsection header line ("0 N\n").
    while i < buf.len() && (buf[i] == b'\n' || buf[i] == b'\r' || buf[i] == b' ') {
        i += 1;
    }
    // Skip subsection header digits and newline.
    while i < buf.len() && buf[i] != b'\n' && buf[i] != b'\r' {
        i += 1;
    }
    while i < buf.len() && (buf[i] == b'\n' || buf[i] == b'\r') {
        i += 1;
    }

    // Overwrite each 20-byte xref entry's offset digits.
    while i + 20 <= buf.len() {
        // Check that byte 17 is 'f' or 'n' (xref entry type marker).
        if buf[i + 17] == b'f' || buf[i + 17] == b'n' {
            for j in 0..10 {
                buf[i + j] = b'9';
            }
            i += 20;
        } else {
            break;
        }
    }
}

/// Replaces the byte offset after "startxref" with an impossibly large value.
/// This prevents pdfium from locating the xref table via the standard
/// startxref pointer at the end of the PDF file.
fn corrupt_startxref_pointer(buf: &mut [u8]) {
    let marker = b"startxref";
    let Some(pos) = find_subsequence(buf, marker) else {
        return;
    };

    let mut num_start = pos + marker.len();
    // Skip whitespace between "startxref" and the number.
    while num_start < buf.len()
        && (buf[num_start] == b'\n' || buf[num_start] == b'\r' || buf[num_start] == b' ')
    {
        num_start += 1;
    }

    // Find the end of the number (next non-digit).
    let mut num_end = num_start;
    while num_end < buf.len() && buf[num_end].is_ascii_digit() {
        num_end += 1;
    }

    // Overwrite with "9999999" padded with spaces to fill the original width.
    let replacement = b"9999999";
    for (offset, byte) in buf[num_start..num_end].iter_mut().enumerate() {
        *byte = if offset < replacement.len() {
            replacement[offset]
        } else {
            b' '
        };
    }
}

/// Corrupts the /Size value in the trailer dictionary by replacing it with
/// a much larger number. This makes the trailer inconsistent with the actual
/// object count, preventing pdfium from using the trailer as a repair anchor.
fn corrupt_trailer_size(buf: &mut [u8]) {
    let marker = b"/Size ";
    let Some(pos) = find_subsequence(buf, marker) else {
        return;
    };

    let num_start = pos + marker.len();
    let mut num_end = num_start;
    while num_end < buf.len() && buf[num_end].is_ascii_digit() {
        num_end += 1;
    }

    // Replace with 99999 (padded with spaces if the original number is longer).
    let replacement = b"99999";
    for (offset, byte) in buf[num_start..num_end].iter_mut().enumerate() {
        *byte = if offset < replacement.len() {
            replacement[offset]
        } else {
            b' '
        };
    }
}

/// Finds the first occurrence of `needle` in `haystack` and returns its
/// byte offset. Returns `None` if the needle is not found.
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
