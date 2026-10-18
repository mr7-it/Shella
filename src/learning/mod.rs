//! Aturan pembelajaran sinaptik dan state trace.
mod eligibility_trace;
mod reward_modulated;
mod stdp;

pub(crate) use eligibility_trace::StdpTrace;
pub use reward_modulated::RewardModulatedStdpConfig;
pub use stdp::StdpConfig;
