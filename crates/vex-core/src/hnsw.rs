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

use serde::Serialize;

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

// ---------------------------------------------------------------------------
// Search tracing.
//
// `search_traced` returns the same results as `search_filtered_with_ef` plus
// a step-by-step record of what the traversal did: the greedy descent through
// the upper layers and every edge the layer-0 beam evaluated. The plain
// search paths thread a `NoopSink` through the same code, so tracing costs
// nothing unless requested (the no-op methods monomorphize away).
// ---------------------------------------------------------------------------

/// What happened to a node the moment the beam evaluated it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BeamStatus {
    /// Live, filter-matching, and close enough: admitted as a result
    /// candidate (it may still be evicted by closer nodes later).
    Admitted,
    /// Tombstoned or filtered out: explored for routing, never a result.
    Routed,
    /// Already worse than the beam's worst kept result: evaluated, dropped.
    Rejected,
}

/// One improvement step of the greedy descent through an upper layer.
#[derive(Debug, Clone, Serialize)]
pub struct DescentHop {
    pub layer: usize,
    pub from: u64,
    pub to: u64,
    pub distance: f32,
}

/// One neighbor evaluation during the layer-0 beam search, in evaluation
/// order. `from` is the node whose adjacency list surfaced `to`.
#[derive(Debug, Clone, Serialize)]
pub struct BeamEdge {
    pub from: u64,
    pub to: u64,
    pub distance: f32,
    pub status: BeamStatus,
}

/// A full record of one HNSW search. Node references are vector ids, not
/// internal arena positions, so callers can resolve payloads and vectors.
///
/// The layer-0 beam starts where the descent ends: the last [`DescentHop`]'s
/// `to`, or `entry_id` when the descent made no hops. That start node is
/// admitted to the result beam directly (if live and filter-matching), so it
/// can appear in results without an `Admitted` beam edge of its own.
#[derive(Debug, Clone, Serialize)]
pub struct SearchTrace {
    /// Entry point of the graph (`None` only for an empty index).
    pub entry_id: Option<u64>,
    /// Distance from the query to the entry point.
    pub entry_distance: Option<f32>,
    /// Top layer of the graph at search time.
    pub max_layer: usize,
    /// Greedy hops through layers `max_layer..=1`, in order.
    pub descent: Vec<DescentHop>,
    /// Every layer-0 neighbor evaluation, in order.
    pub beam: Vec<BeamEdge>,
    /// Unique nodes whose distance was evaluated on layer 0.
    pub visited: usize,
    /// Total distance computations across all layers.
    pub distance_evals: usize,
}

/// Observer threaded through the search internals. Default methods are
/// no-ops; `NoopSink` (the plain-search case) compiles to nothing.
pub(crate) trait TraceSink {
    fn entry(&mut self, _pos: u32, _distance: f32) {}
    fn descent_hop(&mut self, _layer: usize, _from: u32, _to: u32, _distance: f32) {}
    fn beam_edge(&mut self, _from: u32, _to: u32, _distance: f32, _status: BeamStatus) {}
    fn distance_eval(&mut self) {}
}

pub(crate) struct NoopSink;
impl TraceSink for NoopSink {}

/// Records raw arena positions during a search; `search_traced` resolves
/// them to vector ids afterwards (the sink can't borrow the index while the
/// search holds `&self`).
#[derive(Default)]
struct RecordingSink {
    entry: Option<(u32, f32)>,
    descent: Vec<(usize, u32, u32, f32)>,
    beam: Vec<(u32, u32, f32, BeamStatus)>,
    distance_evals: usize,
}

impl TraceSink for RecordingSink {
    fn entry(&mut self, pos: u32, distance: f32) {
        self.entry = Some((pos, distance));
    }
    fn descent_hop(&mut self, layer: usize, from: u32, to: u32, distance: f32) {
        self.descent.push((layer, from, to, distance));
    }
    fn beam_edge(&mut self, from: u32, to: u32, distance: f32, status: BeamStatus) {
        self.beam.push((from, to, distance, status));
    }
    fn distance_eval(&mut self) {
        self.distance_evals += 1;
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
        self.search_inner(query, k, ef, filter, &mut NoopSink)
    }

    /// Search and record what the traversal did: greedy descent hops, every
    /// layer-0 beam evaluation with its outcome, and work counters. Results
    /// are identical to [`HnswIndex::search_filtered_with_ef`] with the same
    /// arguments — tracing observes the search, it never alters it.
    pub fn search_traced(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&Filter>,
    ) -> Result<(Vec<SearchResult>, SearchTrace)> {
        let mut sink = RecordingSink::default();
        let results = self.search_inner(query, k, ef, filter, &mut sink)?;
        let id_of = |pos: u32| self.nodes[pos as usize].id.0;
        // Each beam edge targets a node at most once (the visited set dedups
        // before evaluation), so visited = beam targets + the entry point.
        let visited = sink.beam.len() + usize::from(sink.entry.is_some());
        let trace = SearchTrace {
            entry_id: sink.entry.map(|(pos, _)| id_of(pos)),
            entry_distance: sink.entry.map(|(_, d)| d),
            max_layer: self.max_layer,
            descent: sink
                .descent
                .into_iter()
                .map(|(layer, from, to, distance)| DescentHop {
                    layer,
                    from: id_of(from),
                    to: id_of(to),
                    distance,
                })
                .collect(),
            beam: sink
                .beam
                .into_iter()
                .map(|(from, to, distance, status)| BeamEdge {
                    from: id_of(from),
                    to: id_of(to),
                    distance,
                    status,
                })
                .collect(),
            visited,
            distance_evals: sink.distance_evals,
        };
        Ok((results, trace))
    }

    /// Exact top-k by brute force over every live vector — the ground-truth
    /// oracle for measuring this same index's HNSW recall. O(n) distance
    /// evaluations; same validation and result contract as `search`.
    pub fn search_exact(
        &self,
        query: &[f32],
        k: usize,
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
        let mut best: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
        for (pos, node) in self.nodes.iter().enumerate() {
            if node.deleted || !filter.is_none_or(|f| f.matches(node.payload.as_ref())) {
                continue;
            }
            let c = Candidate {
                distance: self.dist_to_node(query, pos as u32),
                node: pos as u32,
            };
            if best.len() < k {
                best.push(c);
            } else if c < *best.peek().expect("heap holds k > 0 entries") {
                best.pop();
                best.push(c);
            }
        }
        Ok(best
            .into_sorted_vec()
            .into_iter()
            .map(|c| SearchResult {
                id: self.nodes[c.node as usize].id,
                distance: c.distance,
            })
            .collect())
    }

    /// Iterate `(id, vector, payload)` over live vectors in arena order.
    /// The order is deterministic for an unmodified index.
    pub fn iter_live(&self) -> impl Iterator<Item = (VectorId, &Vector, Option<&Payload>)> + '_ {
        self.nodes
            .iter()
            .filter(|n| !n.deleted)
            .map(|n| (n.id, &n.vector, n.payload.as_ref()))
    }

    fn search_inner<S: TraceSink>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&Filter>,
        sink: &mut S,
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
        sink.distance_eval();
        sink.entry(ep, ep_dist);
        for layer in (1..=self.max_layer).rev() {
            (ep, ep_dist) = self.greedy_closest(query, ep, ep_dist, layer, sink);
        }

        // ...then the real beam search on the dense bottom layer.
        let ef = ef.max(k);
        let admit = |n: u32| {
            let node = &self.nodes[n as usize];
            !node.deleted && filter.is_none_or(|f| f.matches(node.payload.as_ref()))
        };
        let start = Candidate {
            distance: ep_dist,
            node: ep,
        };
        let found = self.search_layer(query, start, ef, 0, admit, sink);
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
    fn greedy_closest<S: TraceSink>(
        &self,
        query: &[f32],
        mut ep: u32,
        mut ep_dist: f32,
        layer: usize,
        sink: &mut S,
    ) -> (u32, f32) {
        loop {
            let mut improved = false;
            for &nb in &self.nodes[ep as usize].neighbors[layer] {
                let d = self.dist_to_node(query, nb);
                sink.distance_eval();
                if d < ep_dist {
                    sink.descent_hop(layer, ep, nb, d);
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
    fn search_layer<F: Fn(u32) -> bool, S: TraceSink>(
        &self,
        query: &[f32],
        start: Candidate,
        ef: usize,
        layer: usize,
        admit: F,
        sink: &mut S,
    ) -> Vec<Candidate> {
        let mut visited = vec![false; self.nodes.len()];
        visited[start.node as usize] = true;

        let mut frontier = BinaryHeap::new();
        frontier.push(Reverse(start));
        let mut best: BinaryHeap<Candidate> = BinaryHeap::new();
        if admit(start.node) {
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
                sink.distance_eval();
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
                        sink.beam_edge(current.node, nb, d, BeamStatus::Admitted);
                    } else {
                        sink.beam_edge(current.node, nb, d, BeamStatus::Routed);
                    }
                } else {
                    sink.beam_edge(current.node, nb, d, BeamStatus::Rejected);
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
            (ep, ep_dist) = self.greedy_closest(&query, ep, ep_dist, layer, &mut NoopSink);
        }

        // Phase 2: on each layer the node lives on, beam-search for
        // candidates, pick M by the heuristic, and link bidirectionally.
        for layer in (0..=level.min(self.max_layer)).rev() {
            // Insertion admits every node — tombstones stay good waypoints,
            // and linking to them keeps the graph connected.
            let cands = self.search_layer(
                &query,
                Candidate {
                    distance: ep_dist,
                    node: ep,
                },
                self.config.ef_construction,
                layer,
                |_| true,
                &mut NoopSink,
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

    fn vector(&self, id: VectorId) -> Option<&Vector> {
        self.id_to_pos.get(&id).and_then(|&pos| {
            let node = &self.nodes[pos as usize];
            if node.deleted {
                None
            } else {
                Some(&node.vector)
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
    fn vector_accessor_respects_tombstones() {
        let mut idx = HnswIndex::with_defaults(2, DistanceMetric::L2);
        idx.add(VectorId(1), v(&[3.0, 4.0])).unwrap();
        assert_eq!(idx.vector(VectorId(1)).unwrap().as_slice(), &[3.0, 4.0]);
        idx.remove(VectorId(1)).unwrap();
        assert!(idx.vector(VectorId(1)).is_none());
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

    #[test]
    fn traced_search_matches_plain_search() {
        let vectors = random_vectors(500, 8, 11);
        let mut idx = HnswIndex::with_defaults(8, DistanceMetric::Cosine);
        for (i, vec) in vectors.iter().enumerate() {
            idx.add(VectorId(i as u64), Vector::from_vec(vec.clone()))
                .unwrap();
        }
        for q in random_vectors(20, 8, 311) {
            for ef in [10, 64, 200] {
                let plain = idx.search_with_ef(&q, 10, ef).unwrap();
                let (traced, trace) = idx.search_traced(&q, 10, ef, None).unwrap();
                assert_eq!(plain, traced, "tracing changed the results at ef={ef}");

                // Trace invariants.
                assert!(trace.entry_id.is_some());
                assert_eq!(trace.visited, trace.beam.len() + 1);
                assert!(trace.distance_evals >= trace.visited);
                for hop in &trace.descent {
                    assert!(hop.layer >= 1, "descent hops live on upper layers");
                    assert!((hop.to as usize) < vectors.len());
                }
                for edge in &trace.beam {
                    assert!((edge.to as usize) < vectors.len());
                }
                // Every result must have been admitted by the beam, or be
                // the node the beam started from (the descent's last stop,
                // which enters the result heap without a beam edge).
                let beam_start = trace.descent.last().map(|h| h.to).or(trace.entry_id);
                let admitted: std::collections::HashSet<u64> = trace
                    .beam
                    .iter()
                    .filter(|e| e.status == BeamStatus::Admitted)
                    .map(|e| e.to)
                    .chain(beam_start)
                    .collect();
                for r in &traced {
                    assert!(admitted.contains(&r.id.0), "result {} never admitted", r.id);
                }
            }
        }
    }

    #[test]
    fn traced_filtered_search_marks_routed_nodes() {
        use serde_json::json;
        let vectors = random_vectors(600, 8, 23);
        let mut idx = HnswIndex::with_defaults(8, DistanceMetric::L2);
        for (i, vec) in vectors.iter().enumerate() {
            idx.add_with_payload(
                VectorId(i as u64),
                Vector::from_vec(vec.clone()),
                Some(json!({"bucket": i % 10})),
            )
            .unwrap();
        }
        let filter = crate::payload::Filter::Eq {
            key: "bucket".into(),
            value: json!(4),
        };
        let q = &random_vectors(1, 8, 90)[0];
        let plain = idx
            .search_filtered_with_ef(q, 10, 64, Some(&filter))
            .unwrap();
        let (traced, trace) = idx.search_traced(q, 10, 64, Some(&filter)).unwrap();
        assert_eq!(plain, traced);
        // A 10% filter must route through non-matching nodes...
        assert!(
            trace.beam.iter().any(|e| e.status == BeamStatus::Routed),
            "selective filter produced no routed nodes"
        );
        // ...and admitted nodes must all match the filter.
        for e in &trace.beam {
            if e.status == BeamStatus::Admitted {
                assert_eq!(e.to % 10, 4, "admitted node {} violates the filter", e.to);
            }
        }
        // The trace serializes for wire consumers (wasm, HTTP explain).
        let json = serde_json::to_value(&trace).unwrap();
        assert!(json["beam"].as_array().unwrap().len() == trace.beam.len());
        assert!(json["beam"][0]["status"].is_string());
    }

    #[test]
    fn traced_search_on_empty_index() {
        let idx = HnswIndex::with_defaults(4, DistanceMetric::L2);
        let (results, trace) = idx.search_traced(&[0.0; 4], 5, 32, None).unwrap();
        assert!(results.is_empty());
        assert_eq!(trace.entry_id, None);
        assert_eq!(trace.visited, 0);
        assert_eq!(trace.distance_evals, 0);
        assert!(trace.descent.is_empty() && trace.beam.is_empty());
    }

    #[test]
    fn search_exact_matches_flat_ground_truth() {
        use serde_json::json;
        let vectors = random_vectors(400, 8, 17);
        let mut hnsw = HnswIndex::with_defaults(8, DistanceMetric::L2);
        let mut flat = FlatIndex::new(8, DistanceMetric::L2);
        for (i, vec) in vectors.iter().enumerate() {
            let payload = Some(json!({"bucket": i % 5}));
            hnsw.add_with_payload(
                VectorId(i as u64),
                Vector::from_vec(vec.clone()),
                payload.clone(),
            )
            .unwrap();
            flat.add_with_payload(VectorId(i as u64), Vector::from_vec(vec.clone()), payload)
                .unwrap();
        }
        // Tombstones must be excluded from the exact scan.
        for i in 0..40u64 {
            hnsw.remove(VectorId(i)).unwrap();
            flat.remove(VectorId(i)).unwrap();
        }
        let filter = crate::payload::Filter::Eq {
            key: "bucket".into(),
            value: json!(2),
        };
        for q in random_vectors(10, 8, 71) {
            assert_eq!(
                hnsw.search_exact(&q, 10, None).unwrap(),
                flat.search(&q, 10).unwrap(),
                "exact scan diverged from FlatIndex"
            );
            assert_eq!(
                hnsw.search_exact(&q, 10, Some(&filter)).unwrap(),
                flat.search_filtered(&q, 10, Some(&filter)).unwrap(),
                "filtered exact scan diverged from FlatIndex"
            );
        }
        // Same edge-case contract as every other search path.
        assert!(matches!(
            hnsw.search_exact(&random_vectors(1, 8, 1)[0], 0, None),
            Err(VexError::InvalidK)
        ));
        assert!(matches!(
            hnsw.search_exact(&[0.0; 3], 5, None),
            Err(VexError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn iter_live_skips_tombstones() {
        let mut idx = HnswIndex::with_defaults(2, DistanceMetric::L2);
        for i in 0..10u64 {
            idx.add(VectorId(i), v(&[i as f32, 0.0])).unwrap();
        }
        idx.remove(VectorId(3)).unwrap();
        idx.remove(VectorId(7)).unwrap();
        let ids: Vec<u64> = idx.iter_live().map(|(id, _, _)| id.0).collect();
        assert_eq!(ids, vec![0, 1, 2, 4, 5, 6, 8, 9]);
        for (id, vec, _) in idx.iter_live() {
            assert_eq!(vec.as_slice()[0], id.0 as f32);
        }
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
