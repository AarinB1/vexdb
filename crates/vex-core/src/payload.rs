//! Per-vector payloads and the filter language for filtered search.
//!
//! A payload is an arbitrary JSON object attached to a vector at insert
//! time. Filters evaluate against top-level payload keys. The shape
//! deliberately mirrors Qdrant's `must`/`should` flavor, condensed:
//!
//! ```json
//! {"and": [
//!   {"eq": {"key": "category", "value": "shoes"}},
//!   {"range": {"key": "price", "lte": 99.0}}
//! ]}
//! ```
//!
//! Filtering happens *during* HNSW graph traversal (pre-filtering), not as a
//! post-search cull: non-matching nodes still route the beam, but only
//! matching nodes occupy result slots, so a selective filter widens the
//! search instead of starving it of results.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Arbitrary JSON attached to a vector.
pub type Payload = Value;

/// A predicate over payloads. Keys address top-level payload fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Filter {
    /// Exact JSON equality on a field.
    Eq {
        key: String,
        value: Value,
    },
    /// Numeric range on a field; missing bounds are unbounded. Non-numeric
    /// or absent fields never match.
    Range {
        key: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gte: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lte: Option<f64>,
    },
    /// Field value is one of the listed values.
    In {
        key: String,
        values: Vec<Value>,
    },
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

impl Filter {
    /// Evaluate against a vector's payload. A vector with no payload only
    /// matches filters that are vacuously true (`And([])`) or negations.
    pub fn matches(&self, payload: Option<&Payload>) -> bool {
        match self {
            Filter::Eq { key, value } => payload.and_then(|p| p.get(key)) == Some(value),
            Filter::Range { key, gte, lte } => payload
                .and_then(|p| p.get(key))
                .and_then(Value::as_f64)
                .is_some_and(|x| gte.is_none_or(|g| x >= g) && lte.is_none_or(|l| x <= l)),
            Filter::In { key, values } => payload
                .and_then(|p| p.get(key))
                .is_some_and(|v| values.contains(v)),
            Filter::And(fs) => fs.iter().all(|f| f.matches(payload)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(payload)),
            Filter::Not(f) => !f.matches(payload),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn eq_matches() {
        let f = Filter::Eq {
            key: "category".into(),
            value: json!("shoes"),
        };
        assert!(f.matches(Some(&json!({"category": "shoes"}))));
        assert!(!f.matches(Some(&json!({"category": "hats"}))));
        assert!(!f.matches(None));
    }

    #[test]
    fn range_matches() {
        let f = Filter::Range {
            key: "price".into(),
            gte: Some(10.0),
            lte: Some(99.0),
        };
        assert!(f.matches(Some(&json!({"price": 50}))));
        assert!(f.matches(Some(&json!({"price": 10.0}))));
        assert!(!f.matches(Some(&json!({"price": 100}))));
        assert!(!f.matches(Some(&json!({"price": "fifty"}))));
        assert!(!f.matches(None));
    }

    #[test]
    fn in_and_or_not() {
        let f = Filter::And(vec![
            Filter::In {
                key: "color".into(),
                values: vec![json!("red"), json!("blue")],
            },
            Filter::Not(Box::new(Filter::Eq {
                key: "sold_out".into(),
                value: json!(true),
            })),
        ]);
        assert!(f.matches(Some(&json!({"color": "red"}))));
        assert!(!f.matches(Some(&json!({"color": "red", "sold_out": true}))));
        assert!(!f.matches(Some(&json!({"color": "green"}))));
    }

    #[test]
    fn json_wire_format() {
        let f: Filter = serde_json::from_str(
            r#"{"and":[{"eq":{"key":"category","value":"shoes"}},{"range":{"key":"price","lte":99.0}}]}"#,
        )
        .unwrap();
        assert!(f.matches(Some(&json!({"category": "shoes", "price": 49.0}))));
        assert!(!f.matches(Some(&json!({"category": "shoes", "price": 149.0}))));
    }
}
