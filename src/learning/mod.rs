//! Aturan pembelajaran sinaptik dan state trace.
mod eligibility_trace;
mod long_term;
mod reward_modulated;
mod stdp;

pub(crate) use eligibility_trace::StdpTrace;
pub use long_term::{
    ConsolidationReport, ForgettingReport, LearningMemory, MemoryConflict, MemoryFeedbackWrite,
    MemoryInference, MemoryRecall, ReplayReport,
};
pub(crate) use long_term::{nanos_per_second, unix_timestamp_nanos};
pub use reward_modulated::RewardModulatedStdpConfig;
pub use stdp::StdpConfig;
