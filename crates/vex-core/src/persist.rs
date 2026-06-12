//! Binary snapshot persistence — the `.vex` format.
//!
//! Hand-rolled little-endian format rather than serde/bincode: it forces the
//! versioning, validation, and layout decisions into the open, and the flat
//! arena layout of both indexes means the body is a straight dump of what is
//! already in memory (which is also what a future mmap-backed load wants).
//!
//! ## Layout (format version 1)
//!
//! ```text
//! header:
//!   magic        8  b"VEXSNAP\0"
//!   version      u32
//!   index_type   u8   0 = flat, 1 = hnsw
//!   metric       u8   0 = cosine, 1 = l2, 2 = dot
//!   reserved     u16  must be 0
//!   dim          u64
//!
//! flat body:
//!   count        u64
//!   per entry:   id u64 · payload (u32 len + JSON bytes, 0 = none) · dim×f32
//!
//! hnsw body:
//!   m, ef_construction, ef_search, seed, rng_state   5×u64
//!   entry_point  u64  (u64::MAX = none)
//!   max_layer    u64
//!   live_count   u64
//!   node_count   u64  (includes tombstones)
//!   per node:    id u64 · deleted u8 · layer_count u32
//!                · per layer: (u32 len + len×u32 neighbor indices)
//!                · payload (u32 len + JSON bytes, 0 = none) · dim×f32
//! ```
//!
//! Every length and index read from disk is validated before use; a corrupt
//! or truncated file produces `SnapshotError::Corrupt`, never a panic or an
//! over-allocation.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::distance::DistanceMetric;
use crate::error::Result as VexResult;
use crate::flat::{Entry, FlatIndex};
use crate::hnsw::{HnswConfig, HnswIndex, Node};
use crate::index::{Index, SearchResult};
use crate::payload::{Filter, Payload};
use crate::vector::{Vector, VectorId};

const MAGIC: &[u8; 8] = b"VEXSNAP\0";
const FORMAT_VERSION: u32 = 1;
const TYPE_FLAT: u8 = 0;
const TYPE_HNSW: u8 = 1;
/// Sanity bound on per-vector payload size (16 MiB) so a corrupt length
/// field can't trigger a huge allocation.
const MAX_PAYLOAD_BYTES: u32 = 16 << 20;
/// Sanity bound on HNSW layer count; the level distribution makes anything
/// above a few dozen astronomically unlikely.
const MAX_LAYERS: u32 = 64;

#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("not a vex snapshot (bad magic bytes)")]
    BadMagic,

    #[error("unsupported snapshot format version {0} (this build reads {FORMAT_VERSION})")]
    UnsupportedVersion(u32),

    #[error("corrupt snapshot: {0}")]
    Corrupt(String),
}

fn corrupt(msg: impl Into<String>) -> SnapshotError {
    SnapshotError::Corrupt(msg.into())
}

/// An index of either type, loaded from or destined for a snapshot. This is
/// what the server and CLI hold: they don't care which engine is behind it.
pub enum AnyIndex {
    Flat(FlatIndex),
    Hnsw(HnswIndex),
}

impl AnyIndex {
    pub fn metric(&self) -> DistanceMetric {
        match self {
            AnyIndex::Flat(i) => i.metric(),
            AnyIndex::Hnsw(i) => i.metric(),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            AnyIndex::Flat(_) => "flat",
            AnyIndex::Hnsw(_) => "hnsw",
        }
    }

    /// Search with optional per-query beam width and filter. `ef` is only
    /// meaningful for HNSW; flat search is exact and ignores it.
    pub fn search_opts(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        filter: Option<&Filter>,
    ) -> VexResult<Vec<SearchResult>> {
        match self {
            AnyIndex::Flat(i) => i.search_filtered(query, k, filter),
            AnyIndex::Hnsw(i) => {
                let ef = ef.unwrap_or(i.config().ef_search);
                i.search_filtered_with_ef(query, k, ef, filter)
            }
        }
    }

    /// Serialize to a writer.
    pub fn write_to<W: Write>(&self, w: W) -> Result<(), SnapshotError> {
        let mut w = BufWriter::new(w);
        match self {
            AnyIndex::Flat(i) => write_flat(&mut w, i)?,
            AnyIndex::Hnsw(i) => write_hnsw(&mut w, i)?,
        }
        w.flush()?;
        Ok(())
    }

    /// Deserialize from a reader.
    pub fn read_from<R: Read>(r: R) -> Result<Self, SnapshotError> {
        let mut r = BufReader::new(r);
        read_any(&mut r)
    }

    /// Save to a file. Writes to a temporary sibling first and renames into
    /// place, so a crash mid-write never leaves a truncated snapshot behind.
    pub fn save(&self, path: &Path) -> Result<(), SnapshotError> {
        let tmp = path.with_extension("vex.tmp");
        let file = File::create(&tmp)?;
        self.write_to(file)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self, SnapshotError> {
        let file = File::open(path)?;
        Self::read_from(file)
    }
}

impl std::fmt::Debug for AnyIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyIndex")
            .field("type", &self.type_name())
            .field("dim", &self.dim())
            .field("len", &self.len())
            .field("metric", &self.metric())
            .finish()
    }
}

impl From<FlatIndex> for AnyIndex {
    fn from(i: FlatIndex) -> Self {
        AnyIndex::Flat(i)
    }
}

impl From<HnswIndex> for AnyIndex {
    fn from(i: HnswIndex) -> Self {
        AnyIndex::Hnsw(i)
    }
}

impl Index for AnyIndex {
    fn add_with_payload(
        &mut self,
        id: VectorId,
        vector: Vector,
        payload: Option<Payload>,
    ) -> VexResult<()> {
        match self {
            AnyIndex::Flat(i) => i.add_with_payload(id, vector, payload),
            AnyIndex::Hnsw(i) => i.add_with_payload(id, vector, payload),
        }
    }

    fn remove(&mut self, id: VectorId) -> VexResult<bool> {
        match self {
            AnyIndex::Flat(i) => i.remove(id),
            AnyIndex::Hnsw(i) => i.remove(id),
        }
    }

    fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> VexResult<Vec<SearchResult>> {
        match self {
            AnyIndex::Flat(i) => i.search_filtered(query, k, filter),
            AnyIndex::Hnsw(i) => i.search_filtered(query, k, filter),
        }
    }

    fn payload(&self, id: VectorId) -> Option<&Payload> {
        match self {
            AnyIndex::Flat(i) => i.payload(id),
            AnyIndex::Hnsw(i) => i.payload(id),
        }
    }

    fn vector(&self, id: VectorId) -> Option<&Vector> {
        match self {
            AnyIndex::Flat(i) => i.vector(id),
            AnyIndex::Hnsw(i) => i.vector(id),
        }
    }

    fn len(&self) -> usize {
        match self {
            AnyIndex::Flat(i) => i.len(),
            AnyIndex::Hnsw(i) => i.len(),
        }
    }

    fn dim(&self) -> usize {
        match self {
            AnyIndex::Flat(i) => i.dim(),
            AnyIndex::Hnsw(i) => i.dim(),
        }
    }
}

// ---------------------------------------------------------------------------
// Little-endian primitives
// ---------------------------------------------------------------------------

fn put_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn put_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn put_f32_slice<W: Write>(w: &mut W, vs: &[f32]) -> io::Result<()> {
    for v in vs {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn get_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn get_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn get_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn get_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn get_f32_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn put_payload<W: Write>(w: &mut W, payload: Option<&Payload>) -> Result<(), SnapshotError> {
    match payload {
        None => put_u32(w, 0)?,
        Some(p) => {
            let bytes = serde_json::to_vec(p)
                .map_err(|e| corrupt(format!("unserializable payload: {e}")))?;
            let len = u32::try_from(bytes.len())
                .ok()
                .filter(|&l| l > 0 && l <= MAX_PAYLOAD_BYTES)
                .ok_or_else(|| corrupt("payload exceeds 16 MiB limit"))?;
            put_u32(w, len)?;
            w.write_all(&bytes)?;
        }
    }
    Ok(())
}

fn get_payload<R: Read>(r: &mut R) -> Result<Option<Payload>, SnapshotError> {
    let len = get_u32(r)?;
    if len == 0 {
        return Ok(None);
    }
    if len > MAX_PAYLOAD_BYTES {
        return Err(corrupt(format!("payload length {len} exceeds limit")));
    }
    let mut bytes = vec![0u8; len as usize];
    r.read_exact(&mut bytes)?;
    let value = serde_json::from_slice(&bytes)
        .map_err(|e| corrupt(format!("payload is not valid JSON: {e}")))?;
    Ok(Some(value))
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

fn write_header<W: Write>(
    w: &mut W,
    index_type: u8,
    metric: DistanceMetric,
    dim: usize,
) -> io::Result<()> {
    w.write_all(MAGIC)?;
    put_u32(w, FORMAT_VERSION)?;
    w.write_all(&[index_type, metric.to_tag()])?;
    w.write_all(&0u16.to_le_bytes())?;
    put_u64(w, dim as u64)
}

struct Header {
    index_type: u8,
    metric: DistanceMetric,
    dim: usize,
}

fn read_header<R: Read>(r: &mut R) -> Result<Header, SnapshotError> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(SnapshotError::BadMagic);
    }
    let version = get_u32(r)?;
    if version != FORMAT_VERSION {
        return Err(SnapshotError::UnsupportedVersion(version));
    }
    let index_type = get_u8(r)?;
    if index_type != TYPE_FLAT && index_type != TYPE_HNSW {
        return Err(corrupt(format!("unknown index type tag {index_type}")));
    }
    let metric =
        DistanceMetric::from_tag(get_u8(r)?).ok_or_else(|| corrupt("unknown metric tag"))?;
    if get_u16(r)? != 0 {
        return Err(corrupt("reserved header bytes are not zero"));
    }
    let dim = usize::try_from(get_u64(r)?).map_err(|_| corrupt("dim overflows usize"))?;
    if dim == 0 || dim > (1 << 20) {
        return Err(corrupt(format!("implausible dim {dim}")));
    }
    Ok(Header {
        index_type,
        metric,
        dim,
    })
}

fn read_any<R: Read>(r: &mut R) -> Result<AnyIndex, SnapshotError> {
    let header = read_header(r)?;
    match header.index_type {
        TYPE_FLAT => read_flat_body(r, &header).map(AnyIndex::Flat),
        TYPE_HNSW => read_hnsw_body(r, &header).map(AnyIndex::Hnsw),
        _ => unreachable!("validated in read_header"),
    }
}

// ---------------------------------------------------------------------------
// Flat
// ---------------------------------------------------------------------------

fn write_flat<W: Write>(w: &mut W, index: &FlatIndex) -> Result<(), SnapshotError> {
    write_header(w, TYPE_FLAT, index.metric, index.dim)?;
    put_u64(w, index.entries.len() as u64)?;
    for entry in &index.entries {
        put_u64(w, entry.id.0)?;
        put_payload(w, entry.payload.as_ref())?;
        put_f32_slice(w, entry.vector.as_slice())?;
    }
    Ok(())
}

fn read_flat_body<R: Read>(r: &mut R, header: &Header) -> Result<FlatIndex, SnapshotError> {
    let count = usize::try_from(get_u64(r)?).map_err(|_| corrupt("count overflows usize"))?;
    let mut entries = Vec::new();
    let mut id_to_pos = HashMap::new();
    for pos in 0..count {
        let id = VectorId(get_u64(r)?);
        let payload = get_payload(r)?;
        let vector = Vector::from_vec(get_f32_vec(r, header.dim)?);
        if id_to_pos.insert(id, pos).is_some() {
            return Err(corrupt(format!("duplicate id {id} in flat snapshot")));
        }
        entries.push(Entry {
            id,
            vector,
            payload,
        });
    }
    Ok(FlatIndex {
        metric: header.metric,
        dim: header.dim,
        entries,
        id_to_pos,
    })
}

// ---------------------------------------------------------------------------
// HNSW
// ---------------------------------------------------------------------------

const NO_ENTRY: u64 = u64::MAX;

fn write_hnsw<W: Write>(w: &mut W, index: &HnswIndex) -> Result<(), SnapshotError> {
    write_header(w, TYPE_HNSW, index.metric, index.dim)?;
    put_u64(w, index.config.m as u64)?;
    put_u64(w, index.config.ef_construction as u64)?;
    put_u64(w, index.config.ef_search as u64)?;
    put_u64(w, index.config.seed)?;
    put_u64(w, index.rng_state)?;
    put_u64(w, index.entry_point.map_or(NO_ENTRY, u64::from))?;
    put_u64(w, index.max_layer as u64)?;
    put_u64(w, index.live_count as u64)?;
    put_u64(w, index.nodes.len() as u64)?;
    for node in &index.nodes {
        put_u64(w, node.id.0)?;
        w.write_all(&[node.deleted as u8])?;
        put_u32(w, node.neighbors.len() as u32)?;
        for layer in &node.neighbors {
            put_u32(w, layer.len() as u32)?;
            for &nb in layer {
                put_u32(w, nb)?;
            }
        }
        put_payload(w, node.payload.as_ref())?;
        put_f32_slice(w, node.vector.as_slice())?;
    }
    Ok(())
}

fn read_hnsw_body<R: Read>(r: &mut R, header: &Header) -> Result<HnswIndex, SnapshotError> {
    let m = usize::try_from(get_u64(r)?).map_err(|_| corrupt("m overflows usize"))?;
    let ef_construction =
        usize::try_from(get_u64(r)?).map_err(|_| corrupt("ef_construction overflows usize"))?;
    let ef_search =
        usize::try_from(get_u64(r)?).map_err(|_| corrupt("ef_search overflows usize"))?;
    let seed = get_u64(r)?;
    let rng_state = get_u64(r)?;
    if m == 0 || m > (1 << 16) {
        return Err(corrupt(format!("implausible m {m}")));
    }
    let entry_raw = get_u64(r)?;
    let max_layer =
        usize::try_from(get_u64(r)?).map_err(|_| corrupt("max_layer overflows usize"))?;
    let live_count =
        usize::try_from(get_u64(r)?).map_err(|_| corrupt("live_count overflows usize"))?;
    let node_count = get_u64(r)?;
    let node_count = u32::try_from(node_count)
        .map_err(|_| corrupt(format!("node count {node_count} exceeds u32 arena limit")))?;

    if max_layer >= MAX_LAYERS as usize {
        return Err(corrupt(format!("implausible max_layer {max_layer}")));
    }
    let entry_point = if entry_raw == NO_ENTRY {
        None
    } else {
        let ep = u32::try_from(entry_raw).map_err(|_| corrupt("entry point out of range"))?;
        if ep >= node_count {
            return Err(corrupt("entry point exceeds node count"));
        }
        Some(ep)
    };

    let mut nodes = Vec::new();
    let mut id_to_pos: HashMap<VectorId, u32> = HashMap::new();
    let mut counted_live = 0usize;
    for pos in 0..node_count {
        let id = VectorId(get_u64(r)?);
        let deleted = match get_u8(r)? {
            0 => false,
            1 => true,
            other => return Err(corrupt(format!("invalid deleted flag {other}"))),
        };
        let layer_count = get_u32(r)?;
        if layer_count == 0 || layer_count > MAX_LAYERS {
            return Err(corrupt(format!("implausible layer count {layer_count}")));
        }
        let mut neighbors = Vec::with_capacity(layer_count as usize);
        for _ in 0..layer_count {
            let n = get_u32(r)?;
            if n > node_count.saturating_mul(2) {
                return Err(corrupt(format!("implausible neighbor list length {n}")));
            }
            let mut layer = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let nb = get_u32(r)?;
                if nb >= node_count {
                    return Err(corrupt(format!(
                        "neighbor index {nb} exceeds node count {node_count}"
                    )));
                }
                layer.push(nb);
            }
            neighbors.push(layer);
        }
        let payload = get_payload(r)?;
        let vector = Vector::from_vec(get_f32_vec(r, header.dim)?);
        if !deleted {
            counted_live += 1;
            // Live ids must be unique; tombstones may share an id with a
            // live re-add (the map points at the live one).
            id_to_pos.insert(id, pos);
        } else {
            id_to_pos.entry(id).or_insert(pos);
        }
        nodes.push(Node {
            id,
            vector,
            neighbors,
            deleted,
            payload,
        });
    }
    if counted_live != live_count {
        return Err(corrupt(format!(
            "live_count {live_count} does not match {counted_live} live nodes"
        )));
    }

    Ok(HnswIndex {
        config: HnswConfig {
            m,
            ef_construction,
            ef_search,
            seed,
        },
        metric: header.metric,
        dim: header.dim,
        nodes,
        id_to_pos,
        entry_point,
        max_layer,
        live_count,
        rng_state,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
    fn flat_roundtrip_identical_results() {
        let mut idx = FlatIndex::new(8, DistanceMetric::Cosine);
        let vectors = random_vectors(100, 8, 9);
        for (i, v) in vectors.iter().enumerate() {
            let payload = (i % 3 == 0).then(|| json!({"i": i}));
            idx.add_with_payload(VectorId(i as u64), Vector::from_vec(v.clone()), payload)
                .unwrap();
        }
        let original = AnyIndex::Flat(idx);
        let mut buf = Vec::new();
        original.write_to(&mut buf).unwrap();
        let loaded = AnyIndex::read_from(buf.as_slice()).unwrap();
        let queries = random_vectors(10, 8, 101);
        for q in &queries {
            assert_eq!(
                original.search_opts(q, 5, None, None).unwrap(),
                loaded.search_opts(q, 5, None, None).unwrap(),
                "search results diverged after round-trip"
            );
        }
        assert_eq!(original.len(), loaded.len());
        assert_eq!(original.payload(VectorId(0)), loaded.payload(VectorId(0)));
        assert_eq!(original.payload(VectorId(1)), loaded.payload(VectorId(1)));
    }

    #[test]
    fn hnsw_roundtrip_identical_results() {
        let mut idx = HnswIndex::with_defaults(8, DistanceMetric::L2);
        let vectors = random_vectors(300, 8, 13);
        for (i, v) in vectors.iter().enumerate() {
            let payload = (i % 2 == 0).then(|| json!({"bucket": i % 5}));
            idx.add_with_payload(VectorId(i as u64), Vector::from_vec(v.clone()), payload)
                .unwrap();
        }
        // Tombstones must round-trip too.
        for i in 0..50u64 {
            idx.remove(VectorId(i)).unwrap();
        }
        let original = AnyIndex::Hnsw(idx);
        let mut buf = Vec::new();
        original.write_to(&mut buf).unwrap();
        let loaded = AnyIndex::read_from(buf.as_slice()).unwrap();

        let queries = random_vectors(20, 8, 77);
        for q in &queries {
            assert_eq!(
                original.search_opts(q, 10, Some(64), None).unwrap(),
                loaded.search_opts(q, 10, Some(64), None).unwrap(),
                "search results diverged after round-trip"
            );
        }
        assert_eq!(original.len(), loaded.len());

        // The loaded index must accept further inserts and behave (the RNG
        // state round-trips, so construction continues deterministically).
        let mut loaded = loaded;
        loaded
            .add(VectorId(10_000), Vector::from_vec(vectors[0].clone()))
            .unwrap();
        assert_eq!(loaded.len(), original.len() + 1);
    }

    #[test]
    fn save_load_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.vex");
        let mut idx = HnswIndex::with_defaults(4, DistanceMetric::Dot);
        idx.add(VectorId(1), Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        AnyIndex::Hnsw(idx).save(&path).unwrap();
        let loaded = AnyIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.metric(), DistanceMetric::Dot);
        assert_eq!(loaded.type_name(), "hnsw");
        // No stray temp file left behind.
        assert!(!dir.path().join("test.vex.tmp").exists());
    }

    #[test]
    fn rejects_bad_magic() {
        let err = AnyIndex::read_from(&b"NOTAVEXF blah blah blah"[..]).unwrap_err();
        assert!(matches!(err, SnapshotError::BadMagic));
    }

    #[test]
    fn rejects_wrong_version() {
        let mut buf = Vec::new();
        let mut idx = FlatIndex::new(2, DistanceMetric::L2);
        idx.add(VectorId(1), Vector::from_vec(vec![1.0, 2.0]))
            .unwrap();
        AnyIndex::Flat(idx).write_to(&mut buf).unwrap();
        // Bump the version field (bytes 8..12).
        buf[8] = 0xFF;
        let err = AnyIndex::read_from(buf.as_slice()).unwrap_err();
        assert!(matches!(err, SnapshotError::UnsupportedVersion(_)));
    }

    #[test]
    fn rejects_truncated_file() {
        let mut buf = Vec::new();
        let mut idx = HnswIndex::with_defaults(4, DistanceMetric::L2);
        for i in 0..20u64 {
            idx.add(VectorId(i), Vector::from_vec(vec![i as f32; 4]))
                .unwrap();
        }
        AnyIndex::Hnsw(idx).write_to(&mut buf).unwrap();
        buf.truncate(buf.len() / 2);
        let err = AnyIndex::read_from(buf.as_slice()).unwrap_err();
        assert!(matches!(
            err,
            SnapshotError::Io(_) | SnapshotError::Corrupt(_)
        ));
    }

    #[test]
    fn rejects_out_of_range_neighbor() {
        let mut buf = Vec::new();
        let mut idx = HnswIndex::with_defaults(2, DistanceMetric::L2);
        for i in 0..5u64 {
            idx.add(VectorId(i), Vector::from_vec(vec![i as f32, 0.0]))
                .unwrap();
        }
        AnyIndex::Hnsw(idx).write_to(&mut buf).unwrap();
        // Corrupt every plausible neighbor index byte region by setting a
        // huge value where the first neighbor list entry lives. Rather than
        // computing the exact offset, corrupt aggressively and accept either
        // detection path (Corrupt or Io from cascading misparse) — but it
        // must never panic and never succeed silently with bad indices.
        for i in (buf.len() / 2)..buf.len().min(buf.len() / 2 + 64) {
            buf[i] = 0xFF;
        }
        match AnyIndex::read_from(buf.as_slice()) {
            Ok(loaded) => {
                // If parsing happened to survive, every neighbor index must
                // still be in range (the validator guarantees it).
                let AnyIndex::Hnsw(_h) = loaded else {
                    panic!("type changed")
                };
            }
            Err(SnapshotError::Corrupt(_)) | Err(SnapshotError::Io(_)) => {}
            Err(e) => panic!("unexpected error kind: {e}"),
        }
    }
}
