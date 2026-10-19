//! Aturan pembelajaran sinaptik dan state trace.
mod eligibility_trace;
mod long_term;
mod reward_modulated;
mod stdp;

pub(crate) use eligibility_trace::StdpTrace;
pub use long_term::{
	ForgettingReport, LearningMemory, MemoryConflict, MemoryFeedbackWrite, MemoryInference,
	MemoryRecall, ReplayReport,
};
pub use reward_modulated::RewardModulatedStdpConfig;
pub use stdp::StdpConfig;
