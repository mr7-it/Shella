//! Reasoning engine modules combining LIF inference and persistent experience memory.
mod analysis;
mod calibration;
mod checkpoint;
mod inference;
mod learning;
mod memory;
mod temporal;
mod types;
mod validation;

pub use types::{
    LearningOutcome, MemoryAttention, ReasoningConfig, ReasoningResult, ReasoningStep,
};

use crate::reasoning::{CalibrationState, ContextFrame};
use crate::{LearningMemory, LifNetwork};
use std::collections::VecDeque;

/// Engine penalaran yang memiliki jaringan LIF dan memori pembelajaran persisten.
#[derive(Debug)]
pub struct ReasoningEngine {
    network: LifNetwork,
    memory: LearningMemory,
    config: ReasoningConfig,
    calibration: Option<CalibrationState>,
    context_frames: VecDeque<ContextFrame>,
    active_context_id: Option<String>,
}

impl ReasoningEngine {
    #[must_use]
    pub const fn new(network: LifNetwork, memory: LearningMemory, config: ReasoningConfig) -> Self {
        Self {
            network,
            memory,
            config,
            calibration: None,
            context_frames: VecDeque::new(),
            active_context_id: None,
        }
    }

    #[must_use]
    pub const fn network(&self) -> &LifNetwork {
        &self.network
    }

    #[must_use]
    pub const fn memory(&self) -> &LearningMemory {
        &self.memory
    }

    #[must_use]
    pub const fn config(&self) -> ReasoningConfig {
        self.config
    }

    pub fn context_window(&self) -> impl Iterator<Item = &ContextFrame> {
        self.context_frames.iter()
    }
}
