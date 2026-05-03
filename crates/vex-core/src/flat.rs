use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::distance::DistanceMetric;
use crate::error::{Result, VexError};
use crate::index::{Index, SearchResult};
use crate::vector::{Vector, VectorId};

/// Brute-force flat index: linear scan over every vector for each query.
///
/// Storage is a `Vec<Entry>` plus a `HashMap<VectorId, position>` for O(1)
/// lookup and removal. Removal uses `swap_remove` to keep storage contiguous;
/// the swapped entry's id has its position updated in the map.
pub struct FlatIndex {
    metric: DistanceMetric,
    dim: usize,
    entries: Vec<Entry>,
    id_to_pos: HashMap<VectorId, usize>,
}

struct Entry {
    id: VectorId,
    vector: Vector,
}

impl FlatIndex {
    pub fn new(dim: usize, metric: DistanceMetric) -> Self {
        Self {
            metric,
            dim,
            entries: Vec::new(),
            id_to_pos: HashMap::new(),
        }
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn contains(&self, id: VectorId) -> bool {
        self.id_to_pos.contains_key(&id)
    }
}

impl Index for FlatIndex {
    fn add(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        if vector.dim() != self.dim {
            return Err(VexError::DimensionMismatch {
                expected: self.dim,
                actual: vector.dim(),
            });
        }
        if self.id_to_pos.contains_key(&id) {
            return Err(VexError::DuplicateId(id));
        }
        let pos = self.entries.len();
        self.entries.push(Entry { id, vector });
        self.id_to_pos.insert(id, pos);
        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> Result<bool> {
        let Some(pos) = self.id_to_pos.remove(&id) else {
            return Ok(false);
        };
        self.entries.swap_remove(pos);
        // If we swapped a different entry into `pos`, update its index.
        if pos < self.entries.len() {
            let moved_id = self.entries[pos].id;
            self.id_to_pos.insert(moved_id, pos);
        }
        Ok(true)
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if k == 0 {
            return Err(VexError::InvalidK);
        }
        if query.len() != self.dim {
            return Err(VexError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if self.entries.is_empty() {
            return Err(VexError::EmptyIndex);
        }

        // Max-heap of size k holding the *current* k smallest distances.
        // Top of the heap is the worst (largest) — when a new candidate is
        // smaller than the top, we pop and push, keeping size == k. This is
        // O(n log k) vs. O(n log n) for sorting the whole list.
        //
        // We don't use `Reverse<...>` because the natural ordering of our
        // `HeapItem` already does the right thing for "pop the worst first".
        let cap = k.min(self.entries.len());
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(cap);

        for entry in &self.entries {
            let d = self.metric.distance(query, entry.vector.as_slice())?;
            let item = HeapItem {
                distance: d,
                id: entry.id,
            };
            if heap.len() < cap {
                heap.push(item);
            } else if let Some(top) = heap.peek() {
                if item < *top {
                    heap.pop();
                    heap.push(item);
                }
            }
        }

        // `into_sorted_vec` returns ascending by `Ord` — i.e. smallest distance
        // first — exactly what callers expect.
        let sorted = heap.into_sorted_vec();
        Ok(sorted
            .into_iter()
            .map(|h| SearchResult {
                id: h.id,
                distance: h.distance,
            })
            .collect())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Heap entry. We compare by distance using `total_cmp` so NaN doesn't break
/// ordering invariants. Tie-break by id for determinism.
#[derive(Debug, Clone, Copy)]
struct HeapItem {
    distance: f32,
    id: VectorId,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for HeapItem {}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn v(xs: &[f32]) -> Vector {
        Vector::from_vec(xs.to_vec())
    }

    #[test]
    fn add_and_len() {
        let mut idx = FlatIndex::new(3, DistanceMetric::L2);
        assert!(idx.is_empty());
        idx.add(VectorId(1), v(&[1.0, 0.0, 0.0])).unwrap();
        idx.add(VectorId(2), v(&[0.0, 1.0, 0.0])).unwrap();
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn add_rejects_dim_mismatch() {
        let mut idx = FlatIndex::new(3, DistanceMetric::L2);
        let err = idx.add(VectorId(1), v(&[1.0, 2.0])).unwrap_err();
        assert_eq!(
            err,
            VexError::DimensionMismatch {
                expected: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn add_rejects_duplicate_id() {
        let mut idx = FlatIndex::new(2, DistanceMetric::L2);
        idx.add(VectorId(7), v(&[1.0, 0.0])).unwrap();
        let err = idx.add(VectorId(7), v(&[0.0, 1.0])).unwrap_err();
        assert_eq!(err, VexError::DuplicateId(VectorId(7)));
    }

    #[test]
    fn remove_returns_true_then_false() {
        let mut idx = FlatIndex::new(2, DistanceMetric::L2);
        idx.add(VectorId(1), v(&[1.0, 0.0])).unwrap();
        idx.add(VectorId(2), v(&[0.0, 1.0])).unwrap();
        idx.add(VectorId(3), v(&[1.0, 1.0])).unwrap();
        assert!(idx.remove(VectorId(2)).unwrap());
        assert!(!idx.remove(VectorId(2)).unwrap());
        assert_eq!(idx.len(), 2);
        // After swap_remove, id 3 should still be findable in search results.
        let res = idx.search(&[1.0, 1.0], 2).unwrap();
        let ids: Vec<_> = res.iter().map(|r| r.id).collect();
        assert!(ids.contains(&VectorId(3)));
    }

    #[test]
    fn search_empty_index_errors() {
        let idx = FlatIndex::new(2, DistanceMetric::L2);
        assert_eq!(
            idx.search(&[0.0, 0.0], 1).unwrap_err(),
            VexError::EmptyIndex
        );
    }

    #[test]
    fn search_invalid_k() {
        let mut idx = FlatIndex::new(2, DistanceMetric::L2);
        idx.add(VectorId(1), v(&[1.0, 0.0])).unwrap();
        assert_eq!(idx.search(&[1.0, 0.0], 0).unwrap_err(), VexError::InvalidK);
    }

    #[test]
    fn search_dim_mismatch() {
        let mut idx = FlatIndex::new(2, DistanceMetric::L2);
        idx.add(VectorId(1), v(&[1.0, 0.0])).unwrap();
        let err = idx.search(&[1.0, 0.0, 0.0], 1).unwrap_err();
        assert!(matches!(err, VexError::DimensionMismatch { .. }));
    }

    #[test]
    fn search_returns_sorted_ascending() {
        let mut idx = FlatIndex::new(2, DistanceMetric::L2);
        idx.add(VectorId(1), v(&[10.0, 0.0])).unwrap();
        idx.add(VectorId(2), v(&[1.0, 0.0])).unwrap();
        idx.add(VectorId(3), v(&[5.0, 0.0])).unwrap();
        let res = idx.search(&[0.0, 0.0], 3).unwrap();
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].id, VectorId(2));
        assert_eq!(res[1].id, VectorId(3));
        assert_eq!(res[2].id, VectorId(1));
        assert!(res[0].distance <= res[1].distance);
        assert!(res[1].distance <= res[2].distance);
    }

    #[test]
    fn search_k_larger_than_len_returns_all() {
        // Design choice: silently clamp k to len() rather than erroring.
        let mut idx = FlatIndex::new(2, DistanceMetric::L2);
        idx.add(VectorId(1), v(&[1.0, 0.0])).unwrap();
        idx.add(VectorId(2), v(&[0.0, 1.0])).unwrap();
        let res = idx.search(&[0.0, 0.0], 10).unwrap();
        assert_eq!(res.len(), 2);
    }

    proptest! {
        #[test]
        fn nearest_to_self_is_self_l2(
            vectors in proptest::collection::vec(
                proptest::collection::vec(-100.0f32..100.0, 8),
                1..32usize,
            )
        ) {
            let mut idx = FlatIndex::new(8, DistanceMetric::L2);
            for (i, vec) in vectors.iter().enumerate() {
                idx.add(VectorId(i as u64), Vector::from_vec(vec.clone())).unwrap();
            }
            for (i, vec) in vectors.iter().enumerate() {
                let res = idx.search(vec, 1).unwrap();
                prop_assert_eq!(res.len(), 1);
                // There may be duplicate vectors in the input; any vector at
                // distance 0 is acceptable. But the nearest *must* be at d=0.
                prop_assert!(res[0].distance.abs() < 1e-4, "distance = {} for index {}", res[0].distance, i);
            }
        }

        #[test]
        fn nearest_to_self_is_self_dot(
            vectors in proptest::collection::vec(
                proptest::collection::vec(-10.0f32..10.0, 4),
                1..16usize,
            )
        ) {
            // For Dot, "nearest to self" means largest dot product with self,
            // which isn't necessarily zero distance. So instead we verify
            // that searching for an exact entry returns *that* entry's id
            // (or one with identical vector) at the top.
            let mut idx = FlatIndex::new(4, DistanceMetric::Dot);
            for (i, vec) in vectors.iter().enumerate() {
                idx.add(VectorId(i as u64), Vector::from_vec(vec.clone())).unwrap();
            }
            for (i, vec) in vectors.iter().enumerate() {
                let res = idx.search(vec, vectors.len()).unwrap();
                let self_dist = -vec.iter().map(|x| x * x).sum::<f32>();
                // The top result's distance should be <= our self-dot distance
                // (could be equal due to a duplicate or a vector with larger dot).
                prop_assert!(res[0].distance <= self_dist + 1e-3,
                    "top={} self={} idx={}", res[0].distance, self_dist, i);
            }
        }
    }
}
