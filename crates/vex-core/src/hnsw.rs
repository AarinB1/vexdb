//! HNSW (Hierarchical Navigable Small World) approximate nearest-neighbor
//! index, after Malkov & Yashunin, "Efficient and robust approximate nearest
//! neighbor search using Hierarchical Navigable Small World graphs" (2018).
//!
//! Structure: a stack of proximity graphs. Layer 0 contains every vector;
//! each higher layer contains an exponentially shrinking subset (a node's top
//! layer is sampled from a geometric-ish distribution at insert time). A
//! query greedily descends from the sparse top layers to layer 0, then runs a
//! beam search of width `ef` over the dense bottom layer.
//!
//! Storage is arena-style: nodes live in a flat `Vec`, neighbors are `u32`
//! indices into it. This keeps the graph cache-friendly and makes the
//! eventual on-disk format (Phase 3) a straightforward dump of flat arrays.
//!
//! Deletion uses tombstones: removed nodes stay in the graph as routing
//! waypoints but are filtered from results. True removal would require graph
//! repair, which the paper does not address and most implementations
//! (hnswlib, faiss) also avoid.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};

use crate::distance::DistanceMetric;
use crate::error::{Result, VexError};
use crate::index::{Index, SearchResult};
use crate::payload::{Filter, Payload};
use crate::vector::{Vector, VectorId};

/// Tunable HNSW parameters.
///
/// - `m`: max neighbors per node on layers ≥ 1 (layer 0 allows `2 * m`).
///   Higher improves recall and memory cost; 12–48 is the usual range.
/// - `ef_construction`: beam width while *building*. Higher means a better
///   graph and slower inserts.
/// - `ef_search`: default beam width while *querying* (clamped to ≥ k).
///   This is the recall-vs-latency knob; override per query with
///   [`HnswIndex::search_with_ef`].
/// - `seed`: RNG seed for level sampling, so index construction is
///   deterministic for a fixed insertion order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: 0x5eed_cafe_f00d_d00d,
        }
    }
}

pub(crate) struct Node {
    pub(crate) id: VectorId,
    pub(crate) vector: Vector,
    /// Adjacency lists, one per layer this node participates in
    /// (`neighbors.len() - 1` is the node's top layer).
    pub(crate) neighbors: Vec<Vec<u32>>,
    pub(crate) deleted: bool,
    pub(crate) payload: Option<Payload>,
}

pub struct HnswIndex {
    pub(crate) config: HnswConfig,
    pub(crate) metric: DistanceMetric,
    pub(crate) dim: usize,
    pub(crate) nodes: Vec<Node>,
    pub(crate) id_to_pos: HashMap<VectorId, u32>,
    pub(crate) entry_point: Option<u32>,
    pub(crate) max_layer: usize,
    pub(crate) live_count: usize,
    pub(crate) rng_state: u64,
}

/// (distance, node) pair ordered by distance via `total_cmp` (NaN-safe),
/// tie-broken by node index for determinism. Natural ordering gives a
/// max-heap; wrap in `Reverse` for a min-heap.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    distance: f32,
    node: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.node.cmp(&other.node))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl HnswIndex {
    pub fn new(dim: usize, metric: DistanceMetric, config: HnswConfig) -> Self {
        Self {
            rng_state: config.seed,
            config,
            metric,
            dim,
            nodes: Vec::new(),
            id_to_pos: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            live_count: 0,
        }
    }

    pub fn with_defaults(dim: usize, metric: DistanceMetric) -> Self {
        Self::new(dim, metric, HnswConfig::default())
    }

    pub fn config(&self) -> HnswConfig {
        self.config
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn contains(&self, id: VectorId) -> bool {
        self.id_to_pos
            .get(&id)
            .is_some_and(|&pos| !self.nodes[pos as usize].deleted)
    }

    /// Search with an explicit beam width, overriding `config.ef_search`.
    /// `ef` is clamped to at least `k`.
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        self.search_filtered_with_ef(query, k, ef, None)
    }

    /// Filtered search with an explicit beam width.
    ///
    /// Filtering happens *during* traversal: non-matching nodes (and
    /// tombstones) still route the beam through the graph, but only matching
    /// live nodes occupy the `ef` result slots. A selective filter therefore
    /// widens the explored region instead of returning fewer than `k`
    /// results — the worst case (nothing matches) degrades to visiting the
    /// query's connected component, like a flat scan.
    pub fn search_filtered_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        if k == 0 {
            return Err(VexError::InvalidK);
        }
        if query.len() != self.dim {
            return Err(VexError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let Some(entry) = self.entry_point else {
            return Ok(Vec::new());
        };
        if self.live_count == 0 {
            return Ok(Vec::new());
        }

        // Greedy descent through the sparse upper layers (beam width 1)...
        let mut ep = entry;
        let mut ep_dist = self.dist_to_node(query, ep);
        for layer in (1..=self.max_layer).rev() {
            (ep, ep_dist) = self.greedy_closest(query, ep, ep_dist, layer);
        }

        // ...then the real beam search on the dense bottom layer.
        let ef = ef.max(k);
        let admit = |n: u32| {
            let node = &self.nodes[n as usize];
            !node.deleted && filter.is_none_or(|f| f.matches(node.payload.as_ref()))
        };
        let found = self.search_layer(query, ep, ep_dist, ef, 0, admit);
        Ok(found
            .into_iter()
            .take(k)
            .map(|c| SearchResult {
                id: self.nodes[c.node as usize].id,
                distance: c.distance,
            })
            .collect())
    }

    /// Distance between a raw query slice and a stored node. Dimensions are
    /// validated at insert/query boundaries, so this cannot fail.
    fn dist_to_node(&self, query: &[f32], node: u32) -> f32 {
        self.metric
            .distance(query, self.nodes[node as usize].vector.as_slice())
            .expect("dimensions validated at the API boundary")
    }

    /// Sample a node's top layer: floor(-ln(u) * (1 / ln(M))), the
    /// exponentially-decaying distribution from the paper. Uses an inline
    /// splitmix64 so we stay dependency-free and deterministic per seed.
    fn sample_level(&mut self) -> usize {
        self.rng_state = self.rng_state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let u = (((z >> 11) as f64) / ((1u64 << 53) as f64)).max(f64::MIN_POSITIVE);
        let ml = 1.0 / (self.config.m as f64).ln();
        (-u.ln() * ml) as usize
    }

    /// Hill-climb on one layer: repeatedly move to the closest neighbor until
    /// no neighbor improves. This is `search_layer` with ef = 1, special-cased
    /// because the upper layers never need the heaps.
    fn greedy_closest(
        &self,
        query: &[f32],
        mut ep: u32,
        mut ep_dist: f32,
        layer: usize,
    ) -> (u32, f32) {
        loop {
            let mut improved = false;
            for &nb in &self.nodes[ep as usize].neighbors[layer] {
                let d = self.dist_to_node(query, nb);
                if d < ep_dist {
                    ep = nb;
                    ep_dist = d;
                    improved = true;
                }
            }
            if !improved {
                return (ep, ep_dist);
            }
        }
    }

    /// Beam search on a single layer (Algorithm 2 in the paper). Maintains a
    /// min-heap of frontier candidates and a bounded max-heap of the `ef`
    /// best *admitted* results; stops when the closest frontier candidate is
    /// worse than the worst kept result. Non-admitted nodes (tombstones,
    /// filter misses) are traversed — they route the beam — but never
    /// occupy result slots. Returns results sorted ascending by distance.
    fn search_layer<F: Fn(u32) -> bool>(
        &self,
        query: &[f32],
        ep: u32,
        ep_dist: f32,
        ef: usize,
        layer: usize,
        admit: F,
    ) -> Vec<Candidate> {
        let mut visited = vec![false; self.nodes.len()];
        visited[ep as usize] = true;

        let start = Candidate {
            distance: ep_dist,
            node: ep,
        };
        let mut frontier = BinaryHeap::new();
        frontier.push(Reverse(start));
        let mut best: BinaryHeap<Candidate> = BinaryHeap::new();
        if admit(ep) {
            best.push(start);
        }

        while let Some(Reverse(current)) = frontier.pop() {
            // Until `best` holds ef admitted results, worst is +inf and the
            // beam keeps expanding — this is what makes selective filters
            // widen the search instead of starving it.
            let worst = best.peek().map_or(f32::INFINITY, |w| w.distance);
            if best.len() >= ef && current.distance > worst {
                break;
            }
            for &nb in &self.nodes[current.node as usize].neighbors[layer] {
                if std::mem::replace(&mut visited[nb as usize], true) {
                    continue;
                }
                let d = self.dist_to_node(query, nb);
                let worst = best.peek().map_or(f32::INFINITY, |w| w.distance);
                if best.len() < ef || d < worst {
                    let c = Candidate {
                        distance: d,
                        node: nb,
                    };
                    frontier.push(Reverse(c));
                    if admit(nb) {
                        best.push(c);
                        if best.len() > ef {
                            best.pop();
                        }
                    }
                }
            }
        }
        best.into_sorted_vec()
    }

    /// Neighbor selection heuristic (Algorithm 4). Walk candidates closest
    /// first; keep one only if it is closer to the target than to every
    /// neighbor already kept. This spreads edges across directions instead of
    /// clumping them in one cluster, which is what lets greedy routing escape
    /// local neighborhoods — plain "closest M" measurably hurts recall on
    /// clustered data. Pruned candidates backfill any remaining slots
    /// (`keepPrunedConnections` in the paper).
    fn select_neighbors(&self, candidates: &[Candidate], m: usize) -> Vec<u32> {
        let mut selected: Vec<Candidate> = Vec::with_capacity(m);
        let mut pruned: Vec<u32> = Vec::new();
        for &c in candidates {
            if selected.len() >= m {
                break;
            }
            let c_vec = self.nodes[c.node as usize].vector.as_slice();
            let dominated = selected
                .iter()
                .any(|s| self.dist_to_node(c_vec, s.node) < c.distance);
            if dominated {
                pruned.push(c.node);
            } else {
                selected.push(c);
            }
        }
        let mut out: Vec<u32> = selected.into_iter().map(|c| c.node).collect();
        for p in pruned {
            if out.len() >= m {
                break;
            }
            out.push(p);
        }
        out
    }

    /// Re-select `node`'s neighbor list on `layer` down to `m_max` entries,
    /// using the same heuristic as insertion (distances measured from `node`
    /// itself).
    fn shrink_neighbors(&self, node: u32, layer: usize, m_max: usize) -> Vec<u32> {
        let base = self.nodes[node as usize].vector.as_slice();
        let mut cands: Vec<Candidate> = self.nodes[node as usize].neighbors[layer]
            .iter()
            .map(|&nb| Candidate {
                distance: self.dist_to_node(base, nb),
                node: nb,
            })
            .collect();
        cands.sort_unstable();
        self.select_neighbors(&cands, m_max)
    }
}

impl Index for HnswIndex {
    fn add_with_payload(
        &mut self,
        id: VectorId,
        vector: Vector,
        payload: Option<Payload>,
    ) -> Result<()> {
        if vector.dim() != self.dim {
            return Err(VexError::DimensionMismatch {
                expected: self.dim,
                actual: vector.dim(),
            });
        }
        if let Some(&pos) = self.id_to_pos.get(&id) {
            // Re-adding a tombstoned id is allowed: the new node simply
            // takes over the id; the old node stays as an anonymous waypoint.
            if !self.nodes[pos as usize].deleted {
                return Err(VexError::DuplicateId(id));
            }
        }

        let level = self.sample_level();
        let new_pos =
            u32::try_from(self.nodes.len()).expect("HnswIndex supports at most u32::MAX nodes");
        self.nodes.push(Node {
            id,
            vector,
            neighbors: vec![Vec::new(); level + 1],
            deleted: false,
            payload,
        });
        self.id_to_pos.insert(id, new_pos);
        self.live_count += 1;

        let Some(entry) = self.entry_point else {
            self.entry_point = Some(new_pos);
            self.max_layer = level;
            return Ok(());
        };

        // Clone the query out so we can call &self helpers while mutating
        // neighbor lists below. One Vec clone per insert is noise next to
        // ef_construction distance evaluations.
        let query = self.nodes[new_pos as usize].vector.as_slice().to_vec();

        // Phase 1 of insertion: greedy descent through layers above the new
        // node's level to find a good entry point.
        let mut ep = entry;
        let mut ep_dist = self.dist_to_node(&query, ep);
        for layer in ((level + 1)..=self.max_layer).rev() {
            (ep, ep_dist) = self.greedy_closest(&query, ep, ep_dist, layer);
        }

        // Phase 2: on each layer the node lives on, beam-search for
        // candidates, pick M by the heuristic, and link bidirectionally.
        for layer in (0..=level.min(self.max_layer)).rev() {
            // Insertion admits every node — tombstones stay good waypoints,
            // and linking to them keeps the graph connected.
            let cands = self.search_layer(
                &query,
                ep,
                ep_dist,
                self.config.ef_construction,
                layer,
                |_| true,
            );
            let chosen = self.select_neighbors(&cands, self.config.m);
            let m_max = if layer == 0 {
                self.config.m * 2
            } else {
                self.config.m
            };
            for &nb in &chosen {
                self.nodes[nb as usize].neighbors[layer].push(new_pos);
                if self.nodes[nb as usize].neighbors[layer].len() > m_max {
                    let shrunk = self.shrink_neighbors(nb, layer, m_max);
                    self.nodes[nb as usize].neighbors[layer] = shrunk;
                }
            }
            self.nodes[new_pos as usize].neighbors[layer] = chosen;

            // Next layer down starts from the best candidate found here.
            let closest = cands
                .first()
                .expect("search_layer returns at least the entry point");
            ep = closest.node;
            ep_dist = closest.distance;
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(new_pos);
        }
        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> Result<bool> {
        match self.id_to_pos.get(&id) {
            Some(&pos) if !self.nodes[pos as usize].deleted => {
                self.nodes[pos as usize].deleted = true;
                self.live_count -= 1;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        self.search_filtered_with_ef(query, k, self.config.ef_search, filter)
    }

    fn payload(&self, id: VectorId) -> Option<&Payload> {
        self.id_to_pos.get(&id).and_then(|&pos| {
            let node = &self.nodes[pos as usize];
            if node.deleted {
                None
            } else {
                node.payload.as_ref()
            }
        })
    }

    fn len(&self) -> usize {
        self.live_count
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatIndex;
    use proptest::prelude::*;

    fn v(xs: &[f32]) -> Vector {
        Vector::from_vec(xs.to_vec())
    }

    /// Deterministic pseudo-random vectors for recall tests (splitmix64).
    fn random_vectors(n: usize, dim: usize, mut state: u64) -> Vec<Vec<f32>> {
        let mut next = move || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            ((z >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
        };
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    #[test]
    fn empty_search_returns_no_results() {
        let idx = HnswIndex::with_defaults(4, DistanceMetric::L2);
        assert!(idx.search(&[0.0; 4], 5).unwrap().is_empty());
    }

    #[test]
    fn add_rejects_dim_mismatch_and_duplicates() {
        let mut idx = HnswIndex::with_defaults(3, DistanceMetric::L2);
        assert!(matches!(
            idx.add(VectorId(1), v(&[1.0, 2.0])).unwrap_err(),
            VexError::DimensionMismatch { .. }
        ));
        idx.add(VectorId(1), v(&[1.0, 2.0, 3.0])).unwrap();
        assert_eq!(
            idx.add(VectorId(1), v(&[4.0, 5.0, 6.0])).unwrap_err(),
            VexError::DuplicateId(VectorId(1))
        );
    }

    #[test]
    fn search_validates_k_and_dim() {
        let mut idx = HnswIndex::with_defaults(2, DistanceMetric::L2);
        idx.add(VectorId(1), v(&[1.0, 0.0])).unwrap();
        assert_eq!(idx.search(&[1.0, 0.0], 0).unwrap_err(), VexError::InvalidK);
        assert!(matches!(
            idx.search(&[1.0], 1).unwrap_err(),
            VexError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn small_index_is_exact() {
        // With fewer vectors than ef_search the beam covers everything, so
        // HNSW must agree with brute force exactly.
        let vectors = random_vectors(50, 8, 7);
        let mut hnsw = HnswIndex::with_defaults(8, DistanceMetric::L2);
        let mut flat = FlatIndex::new(8, DistanceMetric::L2);
        for (i, vec) in vectors.iter().enumerate() {
            hnsw.add(VectorId(i as u64), Vector::from_vec(vec.clone()))
                .unwrap();
            flat.add(VectorId(i as u64), Vector::from_vec(vec.clone()))
                .unwrap();
        }
        for query in random_vectors(10, 8, 99) {
            let h = hnsw.search(&query, 5).unwrap();
            let f = flat.search(&query, 5).unwrap();
            let h_ids: Vec<_> = h.iter().map(|r| r.id).collect();
            let f_ids: Vec<_> = f.iter().map(|r| r.id).collect();
            assert_eq!(h_ids, f_ids);
        }
    }

    #[test]
    fn tombstoned_ids_never_appear_in_results() {
        let vectors = random_vectors(200, 8, 21);
        let mut idx = HnswIndex::with_defaults(8, DistanceMetric::L2);
        for (i, vec) in vectors.iter().enumerate() {
            idx.add(VectorId(i as u64), Vector::from_vec(vec.clone()))
                .unwrap();
        }
        for i in 0..100u64 {
            assert!(idx.remove(VectorId(i)).unwrap());
        }
        assert!(!idx.remove(VectorId(0)).unwrap(), "remove is idempotent");
        assert_eq!(idx.len(), 100);
        for query in random_vectors(10, 8, 33) {
            for r in idx.search(&query, 10).unwrap() {
                assert!(r.id.0 >= 100, "tombstoned id {} surfaced", r.id);
            }
        }
    }

    #[test]
    fn readd_after_remove_is_allowed() {
        let mut idx = HnswIndex::with_defaults(2, DistanceMetric::L2);
        idx.add(VectorId(1), v(&[1.0, 0.0])).unwrap();
        assert!(idx.remove(VectorId(1)).unwrap());
        idx.add(VectorId(1), v(&[0.0, 1.0])).unwrap();
        assert!(idx.contains(VectorId(1)));
        let res = idx.search(&[0.0, 1.0], 1).unwrap();
        assert_eq!(res[0].id, VectorId(1));
        assert!(res[0].distance < 1e-6);
    }

    #[test]
    fn recall_at_10_against_flat_ground_truth() {
        // 1000 uniform random vectors, 30 queries. Uniform data is an easy
        // case for HNSW (the bench harness is where real datasets go), but
        // this catches any structural regression in the graph construction.
        let n = 1000;
        let dim = 16;
        let k = 10;
        let vectors = random_vectors(n, dim, 42);
        let queries = random_vectors(30, dim, 4242);

        let mut flat = FlatIndex::new(dim, DistanceMetric::L2);
        let mut hnsw = HnswIndex::new(
            dim,
            DistanceMetric::L2,
            HnswConfig {
                m: 16,
                ef_construction: 200,
                ef_search: 100,
                seed: 1,
            },
        );
        for (i, vec) in vectors.iter().enumerate() {
            flat.add(VectorId(i as u64), Vector::from_vec(vec.clone()))
                .unwrap();
            hnsw.add(VectorId(i as u64), Vector::from_vec(vec.clone()))
                .unwrap();
        }

        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let truth: std::collections::HashSet<VectorId> =
                flat.search(q, k).unwrap().iter().map(|r| r.id).collect();
            let approx = hnsw.search(q, k).unwrap();
            hits += approx.iter().filter(|r| truth.contains(&r.id)).count();
            total += truth.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall >= 0.9, "recall@{k} = {recall:.3}, expected >= 0.9");
    }

    #[test]
    fn filtered_search_survives_selective_filters() {
        // 1000 vectors, filter matches ~10%. Traversal-time filtering must
        // still produce k results with correct payload, not starve.
        use serde_json::json;
        let vectors = random_vectors(1000, 8, 5);
        let mut idx = HnswIndex::with_defaults(8, DistanceMetric::L2);
        for (i, vec) in vectors.iter().enumerate() {
            idx.add_with_payload(
                VectorId(i as u64),
                Vector::from_vec(vec.clone()),
                Some(json!({"bucket": i % 10, "n": i})),
            )
            .unwrap();
        }
        let filter = crate::payload::Filter::Eq {
            key: "bucket".into(),
            value: json!(3),
        };
        for q in random_vectors(10, 8, 77) {
            let res = idx
                .search_filtered_with_ef(&q, 10, 64, Some(&filter))
                .unwrap();
            assert_eq!(res.len(), 10, "filtered search starved");
            for r in &res {
                assert_eq!(r.id.0 % 10, 3, "filter violated for id {}", r.id);
            }
        }
        // And a filter that matches nothing returns empty, not an error.
        let none = crate::payload::Filter::Eq {
            key: "bucket".into(),
            value: json!(99),
        };
        assert!(idx
            .search_filtered_with_ef(&random_vectors(1, 8, 1)[0], 5, 32, Some(&none))
            .unwrap()
            .is_empty());
    }

    proptest! {
        /// Structural invariants that must hold for any input: results are
        /// sorted ascending, contain no duplicate ids, only contain inserted
        /// ids, and search agrees with FlatIndex when the beam covers the
        /// whole index.
        #[test]
        fn search_invariants(
            vectors in proptest::collection::vec(
                proptest::collection::vec(-100.0f32..100.0, 6),
                0..40usize,
            ),
            k in 1usize..8,
        ) {
            let mut idx = HnswIndex::with_defaults(6, DistanceMetric::L2);
            for (i, vec) in vectors.iter().enumerate() {
                idx.add(VectorId(i as u64), Vector::from_vec(vec.clone())).unwrap();
            }
            let query = vec![0.0f32; 6];
            let res = idx.search(&query, k).unwrap();
            prop_assert_eq!(res.len(), k.min(vectors.len()));
            for w in res.windows(2) {
                prop_assert!(w[0].distance <= w[1].distance);
            }
            let mut ids: Vec<_> = res.iter().map(|r| r.id).collect();
            ids.sort();
            ids.dedup();
            prop_assert_eq!(ids.len(), res.len(), "duplicate ids in results");
            for r in &res {
                prop_assert!((r.id.0 as usize) < vectors.len());
            }
        }
    }
}
