//! Shared server state: a registry of named collections, each wrapping an
//! index behind a `RwLock` (many concurrent readers, exclusive writer).
//!
//! Persistence model: every collection maps to `<data_dir>/<name>.vex`.
//! Writes mark the collection dirty; a background task (and graceful
//! shutdown) flushes dirty collections back to disk, and an explicit
//! `POST /collections/{name}/snapshot` forces it. Loads happen once at
//! startup.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use vex_core::{AnyIndex, Index as _};

use crate::error::ApiError;

pub struct AppState {
    pub collections: RwLock<HashMap<String, Arc<Collection>>>,
    pub data_dir: PathBuf,
    pub read_only: bool,
}

pub struct Collection {
    pub name: String,
    pub index: RwLock<AnyIndex>,
    /// Set on every successful write; cleared after a successful flush.
    pub dirty: AtomicBool,
}

impl Collection {
    pub fn new(name: String, index: AnyIndex) -> Self {
        Self {
            name,
            index: RwLock::new(index),
            dirty: AtomicBool::new(false),
        }
    }

    /// Persist this collection to its snapshot path if dirty (or forced).
    /// Holds the read lock during serialization — searches proceed, writes
    /// wait.
    pub fn flush(&self, data_dir: &Path, force: bool) -> Result<bool, ApiError> {
        if !force && !self.dirty.load(Ordering::Acquire) {
            return Ok(false);
        }
        let index = self
            .index
            .read()
            .map_err(|_| ApiError::Internal("index lock poisoned".into()))?;
        index.save(&snapshot_path(data_dir, &self.name))?;
        self.dirty.store(false, Ordering::Release);
        Ok(true)
    }
}

pub fn snapshot_path(data_dir: &Path, name: &str) -> PathBuf {
    data_dir.join(format!("{name}.vex"))
}

/// Collection names become file names; restrict them so they can never
/// traverse paths or collide with the snapshot extension machinery.
pub fn validate_name(name: &str) -> Result<(), ApiError> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if ok {
        Ok(())
    } else {
        Err(ApiError::BadRequest(
            "collection name must be 1-64 chars of [a-zA-Z0-9_-]".into(),
        ))
    }
}

impl AppState {
    /// Create state, loading every `*.vex` snapshot found in `data_dir`.
    pub fn load(data_dir: PathBuf, read_only: bool) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&data_dir)?;
        let mut collections = HashMap::new();
        for entry in std::fs::read_dir(&data_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("vex") {
                continue;
            }
            let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if validate_name(name).is_err() {
                tracing::warn!("skipping snapshot with invalid name: {}", path.display());
                continue;
            }
            let index = AnyIndex::load(&path)
                .map_err(|e| anyhow::anyhow!("loading {}: {e}", path.display()))?;
            tracing::info!(
                collection = name,
                vectors = index.len(),
                dim = index.dim(),
                "loaded snapshot"
            );
            collections.insert(
                name.to_string(),
                Arc::new(Collection::new(name.to_string(), index)),
            );
        }
        Ok(Self {
            collections: RwLock::new(collections),
            data_dir,
            read_only,
        })
    }

    pub fn get(&self, name: &str) -> Result<Arc<Collection>, ApiError> {
        self.collections
            .read()
            .map_err(|_| ApiError::Internal("collections lock poisoned".into()))?
            .get(name)
            .cloned()
            .ok_or_else(|| ApiError::NotFound(format!("collection '{name}' does not exist")))
    }

    pub fn ensure_writable(&self) -> Result<(), ApiError> {
        if self.read_only {
            Err(ApiError::ReadOnly)
        } else {
            Ok(())
        }
    }

    /// Flush every dirty collection. Returns the number flushed; errors on
    /// individual collections are logged, not fatal, so one bad disk write
    /// can't stop the others.
    pub fn flush_all(&self, force: bool) -> usize {
        let collections: Vec<Arc<Collection>> = match self.collections.read() {
            Ok(map) => map.values().cloned().collect(),
            Err(_) => return 0,
        };
        let mut flushed = 0;
        for c in collections {
            match c.flush(&self.data_dir, force) {
                Ok(true) => flushed += 1,
                Ok(false) => {}
                Err(e) => tracing::error!(collection = c.name, "flush failed: {e}"),
            }
        }
        flushed
    }
}
