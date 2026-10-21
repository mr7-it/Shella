use super::ReasoningEngine;
use crate::{CausalAssociation, ReplayReport, SimError};

impl ReasoningEngine {
    /// Menyelesaikan penghapusan task neural yang dijurnal oleh SemanticMemory.
    pub fn synchronize_persistent_memories(&mut self) -> Result<usize, SimError> {
        let Some(semantic_memory) = self.semantic_memory.as_ref() else {
            return Ok(0);
        };
        let pending = semantic_memory.pending_learning_task_deletions();
        let mut removed = 0;
        for task_id in pending {
            self.context_frames
                .retain(|frame| frame.context_id != task_id);
            if self.active_context_id.as_deref() == Some(task_id.as_str()) {
                self.active_context_id = None;
            }
            removed += self.memory.discard_task(&task_id)?;
            self.semantic_memory
                .as_mut()
                .ok_or(SimError::InvalidInput {
                    parameter: "semantic_memory",
                    reason: "semantic memory dilepas saat sinkronisasi",
                })?
                .complete_learning_task_deletion(&task_id)?;
        }
        Ok(removed)
    }

    pub fn remove_semantic_concept(&mut self, word_id: crate::WordID) -> Result<bool, SimError> {
        self.synchronize_persistent_memories()?;
        let removed = self
            .semantic_memory
            .as_mut()
            .ok_or(SimError::InvalidInput {
                parameter: "semantic_memory",
                reason: "ReasoningEngine belum memasang SemanticMemory",
            })?
            .remove_concept(word_id)?;
        self.synchronize_persistent_memories()?;
        Ok(removed)
    }

    pub fn forget_semantic_context(&mut self, context_id: &str) -> Result<bool, SimError> {
        self.synchronize_persistent_memories()?;
        let removed = self
            .semantic_memory
            .as_mut()
            .ok_or(SimError::InvalidInput {
                parameter: "semantic_memory",
                reason: "ReasoningEngine belum memasang SemanticMemory",
            })?
            .forget_context(context_id)?;
        self.synchronize_persistent_memories()?;
        Ok(removed)
    }

    /// Menghitung asosiasi fitur-label dari pengalaman berulang dalam task.
    pub fn causal_associations(
        &self,
        task_id: &str,
        minimum_support: usize,
    ) -> Result<Vec<CausalAssociation>, SimError> {
        self.memory.causal_associations(task_id, minimum_support)
    }

    /// Menggabungkan record fitur-mirip dengan label sama dan menghapus duplikat.
    pub fn consolidate_memories(
        &mut self,
        task_id: &str,
        minimum_similarity: f64,
        importance_increase: f64,
    ) -> Result<crate::ConsolidationReport, SimError> {
        self.memory
            .consolidate_similar(task_id, minimum_similarity, importance_increase)
    }

    /// Meluruhkan importance pengalaman dari waktu decay terakhir; penggunaan melindungi.
    pub fn decay_memories(
        &mut self,
        half_life_seconds: f64,
        usage_protection: f64,
        minimum_importance: f64,
    ) -> Result<crate::ForgettingReport, SimError> {
        self.memory
            .decay_by_time(half_life_seconds, usage_protection, minimum_importance)
    }

    /// Replay pengalaman relevan melalui LIF/STDP yang dikonfigurasi.
    pub fn replay_experiences(
        &mut self,
        task_id: &str,
        query: &[f64],
        action_neurons: &[usize],
        replay_limit: usize,
        teacher_current_na: f64,
    ) -> Result<ReplayReport, SimError> {
        self.memory.replay_task_relevant(
            task_id,
            &mut self.network,
            query,
            action_neurons,
            replay_limit,
            teacher_current_na,
        )
    }
}
