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
// CID-range font table parser.
//
// pdf-extract dispatches Type0 (CID) fonts to PdfCIDFont::new(), which
// calls adobe_cmap_parser::get_byte_mapping(&contents).unwrap() on the
// Encoding stream. The adobe-cmap-parser crate calls parse(&input) on
// the CMap bytes and panics with .expect("failed to parse") when the
// pom parser encounters invalid CMap syntax.
//
// This PDF contains:
// - A Type0 font with a deliberately malformed CMap encoding stream
//   (bare alphabetic token where the CMap grammar expects a hex string
//   or integer, causing the pom parser to fail).
// - A standard Helvetica font used for all visible text content.
// - Finance domain text ("trading", "stock", "returns") that pdfium
//   extracts via the Helvetica font after the fallback chain engages.
//
// The generated file replaces the copyrighted "Brock et al. (1992)" PDF
// that was previously used in tests/PDFs/ocr/cid-range-panic/.

use std::path::Path;

use lopdf::content::{Content, Operation};
use lopdf::dictionary;
use lopdf::{Dictionary, Document, Object, Stream, StringFormat};

/// Builds a multi-page PDF with a malformed CID font that triggers a panic
/// in pdf-extract, while Helvetica-rendered text remains extractable by pdfium.
pub fn generate(output_path: &Path) {
    let mut doc = Document::with_version("1.4");

    let pages_id = doc.new_object_id();

    // -- Standard Helvetica font (F1) for visible text extraction --
    let helvetica_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });

    // -- Malformed CID font infrastructure (F2) --

    // CIDSystemInfo dictionary required by CIDFontType2.
    let cid_system_info = dictionary! {
        "Registry" => Object::String(b"Adobe".to_vec(), StringFormat::Literal),
        "Ordering" => Object::String(b"Identity".to_vec(), StringFormat::Literal),
        "Supplement" => Object::Integer(0),
    };

    // Font descriptor for the CID font (minimal required fields).
    let font_descriptor_id = doc.add_object(dictionary! {
        "Type" => "FontDescriptor",
        "FontName" => "SyntheticCIDFont",
        "Flags" => Object::Integer(4),
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
    });

    // CIDFontType2 descendant font dictionary.
    let cid_font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "CIDFontType2",
        "BaseFont" => "SyntheticCIDFont",
        "CIDSystemInfo" => Object::Dictionary(cid_system_info),
        "FontDescriptor" => font_descriptor_id,
        "DW" => Object::Integer(1000),
    });

    // Malformed CMap encoding stream. The begincidrange section contains
    // "INVALID_TOKEN" where the CMap grammar requires a hex string or
    // integer. The pom parser in adobe-cmap-parser cannot match this token
    // against any production rule, so parse() returns Err and the
    // .expect("failed to parse") at line 279 panics.
    let malformed_cmap_bytes = b"/CIDInit /ProcSet findresource begin\n\
        12 dict begin\n\
        begincmap\n\
        /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> def\n\
        /CMapName /MalformedCIDMap def\n\
        /CMapType 1 def\n\
        1 begincodespacerange\n\
        <00> <FF>\n\
        endcodespacerange\n\
        1 begincidrange\n\
        <00> <FF> INVALID_TOKEN\n\
        endcidrange\n\
        endcmap\n\
        CMapName currentdict /CMap defineresource pop\n\
        end\n\
        end\n";

    let encoding_stream_id = doc.add_object(Stream::new(
        Dictionary::new(),
        malformed_cmap_bytes.to_vec(),
    ));

    // Type0 top-level font dictionary referencing the malformed encoding.
    // pdf-extract encounters this font when processing the page resources,
    // enters PdfCIDFont::new(), and panics on the CMap stream.
    let type0_font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type0",
        "BaseFont" => "SyntheticCIDFont",
        "DescendantFonts" => vec![Object::Reference(cid_font_id)],
        "Encoding" => encoding_stream_id,
    });

    // Page resources with both fonts: F1 (Helvetica, visible text) and
    // F2 (malformed CID font, triggers panic).
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! {
            "F1" => helvetica_id,
            "F2" => type0_font_id,
        },
    });

    // Finance domain text for each page. These terms are verified by
    // test T-OCR-011: at least 2 of ["trading", "stock", "returns"].
    let page_texts = [
        "Simple technical trading rules and the stochastic properties of stock returns \
         demonstrate that moving average strategies applied to market indices can produce \
         statistically significant bootstrap results across multiple time periods",
        "Trading strategies based on moving average crossover signals generate returns \
         that exceed transaction costs when applied to stock market data from major exchanges \
         using bootstrap methodology to assess statistical significance of results",
        "Stock return analysis using technical trading rules and momentum indicators shows \
         persistent patterns in financial market data that challenge efficient market theory \
         and support the profitability of systematic trading approaches",
    ];

    let mut page_ids = Vec::new();
    for text in &page_texts {
        // The content stream references F2 (CID font) for a single invisible
        // glyph placed far off-page, then switches to F1 (Helvetica) for all
        // visible text. pdf-extract processes fonts in the order they appear
        // in Tf operators, so it encounters F2 first and panics before reaching
        // the Helvetica text.
        let content = Content {
            operations: vec![
                // Invisible glyph using the malformed CID font (off-page).
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F2".into(), 1.into()]),
                Operation::new("Td", vec![0.into(), Object::Integer(-9999)]),
                Operation::new(
                    "Tj",
                    vec![Object::String(b"\x00".to_vec(), StringFormat::Literal)],
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
        .expect("failed to save CID-range panic PDF");
}
