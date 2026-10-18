use serde::{Deserialize, Serialize};

/// Jejak pra-, pascasinaptik, dan eligibility untuk satu sinaps.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct StdpTrace {
    pub(crate) pre: f64,
    pub(crate) post: f64,
    pub(crate) eligibility: f64,
}
