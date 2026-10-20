use super::{ReasoningConfig, ReasoningEngine};
use crate::reasoning::ReasoningCheckpoint;
use crate::{Checkpoint, LearningMemory, SimError};
use std::path::Path;

impl ReasoningEngine {
    /// Snapshot reasoning state; pengalaman disimpan pada LearningMemory terpisah.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<(), SimError> {
        let checkpoint = self.checkpoint_state()?;
        Checkpoint::save_reasoning(path, &checkpoint)
    }

    /// Memulihkan engine dari checkpoint dan LearningMemory persisten yang diberikan.
    pub fn restore_checkpoint(
        path: impl AsRef<Path>,
        memory: LearningMemory,
    ) -> Result<Self, SimError> {
        let checkpoint = Checkpoint::restore_reasoning(path)?;
        if memory.checkpoint_fingerprint()? != checkpoint.memory_fingerprint {
            return Err(SimError::InvalidCheckpoint {
                reason: "LearningMemory tidak cocok dengan snapshot reasoning".to_owned(),
            });
        }
        let config = ReasoningConfig {
            iterations: checkpoint.iterations,
            recall_limit: checkpoint.recall_limit,
            memory_current_na: checkpoint.memory_current_na,
            temporal_decay: checkpoint.temporal_decay,
            minimum_confidence: checkpoint.minimum_confidence,
            context_window_capacity: checkpoint.context_window_capacity,
            adaptive_compute: checkpoint.adaptive_compute,
        };
        let mut engine = Self::new(checkpoint.network, memory, config);
        engine.context_frames = checkpoint.context_frames.into();
        engine.active_context_id = checkpoint.active_context_id;
        engine.calibration = checkpoint.calibration;
        Ok(engine)
    }

    fn checkpoint_state(&self) -> Result<ReasoningCheckpoint, SimError> {
        Ok(ReasoningCheckpoint {
            version: 1,
            network: self.network.clone(),
            iterations: self.config.iterations,
            recall_limit: self.config.recall_limit,
            memory_current_na: self.config.memory_current_na,
            temporal_decay: self.config.temporal_decay,
            minimum_confidence: self.config.minimum_confidence,
            context_window_capacity: self.config.context_window_capacity,
            adaptive_compute: self.config.adaptive_compute,
            active_context_id: self.active_context_id.clone(),
            context_frames: self.context_frames.iter().cloned().collect(),
            calibration: self.calibration.clone(),
            memory_fingerprint: self.memory.checkpoint_fingerprint()?,
        })
    }
}
