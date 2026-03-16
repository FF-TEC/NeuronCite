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

// Generator for a synthetic PDF that triggers a panic in pdf-extract's
// Type1 font encoding parser.
//
// pdf-extract dispatches Type1 fonts to PdfSimpleFont::new(), which checks
// for a /FontFile stream in the font descriptor. When present, it calls
// type1_encoding_parser::get_encoding_map(&stream_bytes).expect("encoding").
// The type1-encoding-parser crate calls parse(&input).expect("failed to parse")
// on the PostScript font program bytes. An unterminated string literal in
// the font program causes the pom parser to consume to EOF without finding
// the closing parenthesis, returning Err and triggering the panic.
//
// This PDF contains:
// - A Type1 font with a /FontFile stream containing an unterminated PostScript
//   string literal that the pom parser cannot parse.
// - A standard Helvetica font used for all visible text content.
// - Forecasting domain text ("forecast", "accuracy", "error") that pdfium
//   extracts via the Helvetica font. Pdfium does not parse /FontFile streams
//   for text extraction; it uses content stream operators and font encoding
//   dictionaries.
//
// The generated file replaces the copyrighted "Hyndman & Koehler (2006)" PDF
// that was previously used in tests/PDFs/ocr/type1-font-panic/.

use std::path::Path;

use lopdf::content::{Content, Operation};
use lopdf::dictionary;
use lopdf::{Document, Object, Stream, StringFormat};

/// Builds a multi-page PDF with a malformed Type1 font that triggers a panic
/// in pdf-extract's type1-encoding-parser, while Helvetica-rendered text
/// remains extractable by pdfium.
pub fn generate(output_path: &Path) {
    let mut doc = Document::with_version("1.4");

    let pages_id = doc.new_object_id();

    // -- Standard Helvetica font (F1) for visible text extraction --
    let helvetica_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });

    // -- Malformed Type1 font infrastructure (F2) --

    // Malformed Type1 font program stream. The PostScript program starts
    // with valid syntax (font header, name definition, encoding array) but
    // contains an unterminated string literal "(unterminated..." that the
    // pom parser cannot close. The parser consumes to EOF looking for the
    // closing ")" and returns Err, triggering .expect("failed to parse")
    // at type1-encoding-parser line 169.
    let malformed_type1_program = b"%!PS-AdobeFont-1.0: SyntheticType1 001.000\n\
        12 dict begin\n\
        /FontInfo 3 dict dup begin\n\
        /Notice (Synthetic font for pdf-extract panic testing) readonly def\n\
        end readonly def\n\
        /FontName /SyntheticType1 def\n\
        /FontType 1 def\n\
        /Encoding 256 array\n\
        0 1 255 {1 index exch /.notdef put} for\n\
        dup 32 /space put\n\
        dup 65 /A put\n\
        dup 66 /B put\n\
        (unterminated string literal that causes the pom parser to consume to EOF\n";

    let font_file_id = doc.add_object(Stream::new(
        dictionary! {
            "Length1" => Object::Integer(malformed_type1_program.len() as i64),
        },
        malformed_type1_program.to_vec(),
    ));

    // Font descriptor referencing the malformed FontFile stream.
    // The /FontFile key (not /FontFile2 or /FontFile3) signals a Type1
    // font program, causing pdf-extract to invoke the type1-encoding-parser.
    let font_descriptor_id = doc.add_object(dictionary! {
        "Type" => "FontDescriptor",
        "FontName" => "SyntheticType1",
        "Flags" => Object::Integer(32),
        "FontBBox" => vec![
            Object::Integer(0),
            Object::Integer(-200),
            Object::Integer(1000),
            Object::Integer(800),
        ],
        "ItalicAngle" => Object::Integer(0),
        "Ascent" => Object::Integer(800),
        "Descent" => Object::Integer(-200),
        "CapHeight" => Object::Integer(700),
        "StemV" => Object::Integer(80),
        "FontFile" => font_file_id,
    });

    // Type1 font dictionary WITHOUT an /Encoding key. When /Encoding is
    // absent, pdf-extract's PdfSimpleFont::new() falls through to the
    // None branch (line 503 in pdf-extract lib.rs) and uses the
    // type1_encoding parsed from the FontFile stream. Since our FontFile
    // triggers a panic during parsing, the code never reaches the
    // encoding fallback logic.
    let type1_font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "SyntheticType1",
        "FontDescriptor" => font_descriptor_id,
    });

    // Page resources with both fonts.
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! {
            "F1" => helvetica_id,
            "F2" => type1_font_id,
        },
    });

    // Forecasting domain text for each page. These terms are verified by
    // test T-OCR-012: at least 2 of ["forecast", "accuracy", "error"].
    let page_texts = [
        "Another look at measures of forecast accuracy reveals that mean absolute \
         error and mean absolute percentage error MAPE remain the most commonly used \
         accuracy measures in forecasting practice and academic research",
        "Root mean squared error RMSE provides a scale dependent measure of forecast \
         accuracy while mean absolute scaled error MASE offers a scale independent \
         alternative for comparing prediction methods across different time series",
        "Time series forecasting methods evaluated with multiple accuracy metrics \
         show that no single error measure captures all aspects of forecast quality \
         and practitioners should report several complementary accuracy statistics",
    ];

    let mut page_ids = Vec::new();
    for text in &page_texts {
        // Content stream references F2 (malformed Type1) for one invisible
        // glyph off-page, then F1 (Helvetica) for visible text. pdf-extract
        // panics when processing the F2 Tf operator.
        let content = Content {
            operations: vec![
                // Invisible glyph using malformed Type1 font (off-page).
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F2".into(), 1.into()]),
                Operation::new("Td", vec![0.into(), Object::Integer(-9999)]),
                Operation::new(
                    "Tj",
                    vec![Object::String(b"A".to_vec(), StringFormat::Literal)],
                ),
                Operation::new("ET", vec![]),
                // Visible text using Helvetica (extractable by pdfium).
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

    doc.save(output_path)
        .expect("failed to save Type1 font panic PDF");
}
