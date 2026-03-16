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

// Pre-annotation verification check.
//
// Before creating annotations, verifies that the matched text region has
// valid bounding boxes from pdfium. Computes the percentage of characters
// with bounding boxes and the total highlight area.

use crate::types::PreCheckResult;

/// Verifies that the bounding boxes for a matched text region are valid.
///
/// Counts the number of non-degenerate bounding boxes (width > 0 and
/// height > 0) and computes the total area. A pre-check passes if at
/// least 95% of characters have valid bounding boxes.
pub fn pre_check(bounding_boxes: &[[f32; 4]], total_chars: usize) -> PreCheckResult {
    let mut valid_count = 0_usize;
    let mut total_area = 0.0_f32;

    for bbox in bounding_boxes {
        let width = (bbox[2] - bbox[0]).abs();
        let height = (bbox[3] - bbox[1]).abs();

        if width > 0.0 && height > 0.0 {
            valid_count += 1;
            total_area += width * height;
        }
    }

    let ratio = if total_chars > 0 {
        valid_count as f64 / total_chars as f64
    } else {
        0.0
    };

    PreCheckResult {
        passed: ratio >= 0.95,
        total_chars,
        chars_with_bbox: valid_count,
        total_area_sq_pts: total_area,
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// T-ANNOTATE-040: Pre-check passes when all characters have valid
    /// bounding boxes.
    #[test]
    fn t_annotate_040_pre_check_all_chars_have_bbox() {
        let boxes = vec![
            [10.0, 100.0, 20.0, 112.0],
            [20.0, 100.0, 30.0, 112.0],
            [30.0, 100.0, 40.0, 112.0],
        ];
        let result = pre_check(&boxes, 3);
        assert!(result.passed);
        assert_eq!(result.total_chars, 3);
        assert_eq!(result.chars_with_bbox, 3);
        assert!(result.total_area_sq_pts > 0.0);
    }

    /// T-ANNOTATE-041: Pre-check fails when too many characters lack
    /// bounding boxes (below 95% threshold).
    #[test]
    fn t_annotate_041_pre_check_missing_chars() {
        // 5 characters total, only 4 valid boxes, 1 degenerate (zero-width).
        let boxes = vec![
            [10.0, 100.0, 20.0, 112.0],
            [20.0, 100.0, 30.0, 112.0],
            [30.0, 100.0, 40.0, 112.0],
            [40.0, 100.0, 50.0, 112.0],
            [50.0, 100.0, 50.0, 112.0], // zero width -> invalid
        ];
        // With 20 total chars and 4 valid, ratio = 4/20 = 0.20 < 0.95.
        let result = pre_check(&boxes, 20);
        assert!(!result.passed);
        assert_eq!(result.chars_with_bbox, 4);
    }

    /// T-ANNOTATE-045: Pre-check with zero total chars returns passed=false.
    #[test]
    fn t_annotate_045_pre_check_zero_chars() {
        let result = pre_check(&[], 0);
        assert!(!result.passed);
        assert_eq!(result.chars_with_bbox, 0);
    }
}
