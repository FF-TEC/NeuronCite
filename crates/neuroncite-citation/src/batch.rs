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

// Batch assignment for citation verification.
//
// Groups nearby citations from the same section into batches of configurable
// size. Citations are sorted by (section_title, tex_line) to ensure spatial
// locality within batches. Cite-groups (citations from the same \cite{a,b,c}
// command sharing a group_id) are never split across batches -- if including
// a group would exceed the target batch_size, the batch is extended rather
// than splitting the group.

use crate::types::RawCitation;

/// Assigns batch_id values to a slice of RawCitation entries. Sorts the
/// citations by (section_title, tex_line), then groups them into batches
/// of approximately `batch_size` entries. Cite-groups (entries sharing the
/// same group_id) are kept together: if adding a group to the current batch
/// would exceed the target size, the group is still added to the current
/// batch to avoid splitting.
///
/// # Arguments
///
/// * `citations` - Mutable slice of citations to assign batch IDs to. Each
///   entry's `batch_id` field is set to its assigned batch index (0-based).
/// * `batch_size` - Target number of citations per batch. Batches may exceed
///   this size when a cite-group straddles the boundary.
///
/// # Panics
///
/// Panics if `batch_size` is 0.
pub fn assign_batches(citations: &mut [RawCitation], batch_size: usize) {
    assert!(batch_size > 0, "batch_size must be greater than 0");

    if citations.is_empty() {
        return;
    }

    // Sort by section_title (None sorts before Some), then by line number.
    citations.sort_by(|a, b| {
        a.section_title
            .cmp(&b.section_title)
            .then(a.line.cmp(&b.line))
    });

    // Build groups: collect consecutive ranges of citations that share
    // the same group_id. Within the sorted order, citations from the same
    // \cite{a,b,c} command are adjacent because they share the same line
    // number and section.
    let groups = build_groups(citations);

    let mut current_batch: usize = 0;
    let mut current_batch_count: usize = 0;

    for group in &groups {
        let group_size = group.end - group.start;

        // If adding this group would exceed the batch_size AND the current
        // batch is not empty, start a new batch. If the current batch is
        // empty, add the group regardless of size (a single group that
        // exceeds batch_size occupies one batch by itself).
        if current_batch_count > 0 && current_batch_count + group_size > batch_size {
            current_batch += 1;
            current_batch_count = 0;
        }

        // Assign batch_id to all citations in this group.
        for cit in &mut citations[group.start..group.end] {
            cit.batch_id = Some(current_batch);
        }
        current_batch_count += group_size;
    }
}

/// A contiguous range of indices into the citations slice that share the
/// same group_id.
struct GroupRange {
    start: usize,
    end: usize,
}

/// Identifies contiguous ranges of citations that share the same group_id.
/// The input must already be sorted by (section_title, line). Since citations
/// from the same \cite{} command have the same line and section, they remain
/// adjacent after sorting.
fn build_groups(citations: &[RawCitation]) -> Vec<GroupRange> {
    if citations.is_empty() {
        return Vec::new();
    }

    let mut groups = Vec::new();
    let mut start = 0;

    for i in 1..citations.len() {
        if citations[i].group_id != citations[start].group_id {
            groups.push(GroupRange { start, end: i });
            start = i;
        }
    }
    groups.push(GroupRange {
        start,
        end: citations.len(),
    });

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: creates a RawCitation with the given fields.
    fn make_citation(
        cite_key: &str,
        line: usize,
        group_id: usize,
        section: Option<&str>,
    ) -> RawCitation {
        RawCitation {
            cite_key: cite_key.to_string(),
            line,
            anchor_before: String::new(),
            anchor_after: String::new(),
            group_id,
            section_title: section.map(|s| s.to_string()),
            batch_id: None,
            tex_context: String::new(),
        }
    }

    /// T-CIT-019: Basic batching respects batch_size. With 7 citations and
    /// batch_size=3, the first batch has 3 entries, the second has 3, and
    /// the third has 1.
    #[test]
    fn t_cit_019_basic_batching() {
        let mut cits = vec![
            make_citation("a", 1, 0, Some("Intro")),
            make_citation("b", 5, 1, Some("Intro")),
            make_citation("c", 10, 2, Some("Intro")),
            make_citation("d", 15, 3, Some("Intro")),
            make_citation("e", 20, 4, Some("Intro")),
            make_citation("f", 25, 5, Some("Intro")),
            make_citation("g", 30, 6, Some("Intro")),
        ];

        assign_batches(&mut cits, 3);

        // Verify all batch_ids are assigned.
        for cit in &cits {
            assert!(cit.batch_id.is_some());
        }

        // Count entries per batch.
        let batch_0 = cits.iter().filter(|c| c.batch_id == Some(0)).count();
        let batch_1 = cits.iter().filter(|c| c.batch_id == Some(1)).count();
        let batch_2 = cits.iter().filter(|c| c.batch_id == Some(2)).count();

        assert_eq!(batch_0, 3);
        assert_eq!(batch_1, 3);
        assert_eq!(batch_2, 1);
    }

    /// T-CIT-020: Citations in different sections get different batches when
    /// batch_size allows. Section boundaries act as natural batch boundaries.
    #[test]
    fn t_cit_020_section_boundary() {
        let mut cits = vec![
            make_citation("a", 10, 0, Some("Intro")),
            make_citation("b", 20, 1, Some("Intro")),
            make_citation("c", 30, 2, Some("Method")),
            make_citation("d", 40, 3, Some("Method")),
        ];

        assign_batches(&mut cits, 5);

        // With batch_size=5, all 4 fit in one batch. But sorting groups
        // them by section, so they are ordered Intro, Intro, Method, Method.
        // All should be in batch 0 since 4 < 5.
        for cit in &cits {
            assert_eq!(cit.batch_id, Some(0));
        }
    }

    /// T-CIT-021: group_id members are never split across batches. A
    /// 3-member group at the batch boundary stays in one batch.
    #[test]
    fn t_cit_021_group_not_split() {
        let mut cits = vec![
            make_citation("a", 1, 0, Some("Intro")),
            make_citation("b", 2, 1, Some("Intro")),
            // This 3-member group at position 3-4-5 would straddle batch_size=3
            make_citation("c1", 10, 2, Some("Intro")),
            make_citation("c2", 10, 2, Some("Intro")),
            make_citation("c3", 10, 2, Some("Intro")),
        ];

        assign_batches(&mut cits, 3);

        // The first two are in groups 0 and 1 (batch 0).
        // The group of 3 (group_id=2) cannot fit in batch 0 (already has 2,
        // adding 3 would make 5 > 3). So it starts a new batch.
        let c1_batch = cits.iter().find(|c| c.cite_key == "c1").unwrap().batch_id;
        let c2_batch = cits.iter().find(|c| c.cite_key == "c2").unwrap().batch_id;
        let c3_batch = cits.iter().find(|c| c.cite_key == "c3").unwrap().batch_id;

        // All three must be in the same batch.
        assert_eq!(c1_batch, c2_batch);
        assert_eq!(c2_batch, c3_batch);
    }

    /// T-CIT-022: A single group that exceeds batch_size occupies one batch.
    #[test]
    fn t_cit_022_group_exceeds_batch_size() {
        let mut cits = vec![
            make_citation("a1", 10, 0, Some("Intro")),
            make_citation("a2", 10, 0, Some("Intro")),
            make_citation("a3", 10, 0, Some("Intro")),
            make_citation("a4", 10, 0, Some("Intro")),
            make_citation("a5", 10, 0, Some("Intro")),
            make_citation("b", 20, 1, Some("Intro")),
        ];

        assign_batches(&mut cits, 3);

        // The 5-member group exceeds batch_size=3 but stays in one batch.
        let group_0_batch = cits[0].batch_id;
        for cit in &cits[..5] {
            assert_eq!(cit.batch_id, group_0_batch);
        }

        // "b" is in a different batch.
        assert_ne!(cits[5].batch_id, group_0_batch);
    }

    /// T-CIT-023: A single citation gets batch_id 0.
    #[test]
    fn t_cit_023_single_citation() {
        let mut cits = vec![make_citation("only", 1, 0, None)];

        assign_batches(&mut cits, 5);

        assert_eq!(cits[0].batch_id, Some(0));
    }
}
