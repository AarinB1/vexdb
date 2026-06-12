//! Error handling across the network boundary: every internal error maps to
//! an HTTP status and a JSON body of the shape `{"error": "..."}`.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use vex_core::{SnapshotError, VexError};

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    BadRequest(String),

    #[error("{0}")]
    NotFound(String),

    #[error("{0}")]
    Conflict(String),

    #[error("server is running in read-only mode")]
    ReadOnly,

    #[error("internal error: {0}")]
    Internal(String),
}

impl ApiError {
    fn status(&self) -> StatusCode {
        match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::Conflict(_) => StatusCode::CONFLICT,
            ApiError::ReadOnly => StatusCode::FORBIDDEN,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if matches!(self, ApiError::Internal(_)) {
            tracing::error!("{self}");
        }
        (self.status(), Json(json!({ "error": self.to_string() }))).into_response()
    }
}

impl From<VexError> for ApiError {
    fn from(e: VexError) -> Self {
        match e {
            // All engine validation errors are the caller's fault.
            VexError::DimensionMismatch { .. } | VexError::InvalidK => {
                ApiError::BadRequest(e.to_string())
            }
            VexError::DuplicateId(_) => ApiError::Conflict(e.to_string()),
        }
    }
}

impl From<SnapshotError> for ApiError {
    fn from(e: SnapshotError) -> Self {
        ApiError::Internal(e.to_string())
    }
}
