use super::Session;
use crate::{
    LearningMemory, LifNetwork, MemoryStore, SemanticMemory, SimError, decision::DecisionPolicy,
    language::TextEncoder,
};

/// Pemilik runtime dan subsistem opsional yang digunakan aplikasi.
#[derive(Debug)]
pub struct Engine {
    pub(crate) network: LifNetwork,
    pub(crate) text_encoder: Option<TextEncoder>,
    pub(crate) decision_policy: Option<DecisionPolicy>,
    pub(crate) memory: Option<MemoryStore>,
    pub(crate) learning_memory: Option<LearningMemory>,
    pub(crate) semantic_memory: Option<SemanticMemory>,
}
impl Engine {
    /// Membuat engine di atas jaringan LIF aktual.
    #[must_use]
    pub const fn new(network: LifNetwork) -> Self {
        Self {
            network,
            text_encoder: None,
            decision_policy: None,
            memory: None,
            learning_memory: None,
            semantic_memory: None,
        }
    }
    /// Memasang encoder teks berbasis vocabulary yang dilatih pemanggil.
    #[must_use]
    pub fn with_text_encoder(mut self, encoder: TextEncoder) -> Self {
        self.text_encoder = Some(encoder);
        self
    }
    /// Memasang kebijakan keputusan terstruktur.
    #[must_use]
    pub fn with_decision_policy(mut self, policy: DecisionPolicy) -> Self {
        self.decision_policy = Some(policy);
        self
    }
    /// Memasang penyimpanan persisten file-backed.
    #[must_use]
    pub fn with_memory(mut self, memory: MemoryStore) -> Self {
        self.memory = Some(memory);
        self
    }
    /// Memasang semantic memory persisten yang memakai vocabulary dan ID konsep stabil.
    #[must_use]
    pub fn with_semantic_memory(mut self, memory: SemanticMemory) -> Self {
        self.semantic_memory = Some(memory);
        self
    }
    /// Memasang persistent store pengalaman neural yang sama dengan ReasoningEngine.
    #[must_use]
    pub fn with_learning_memory(mut self, memory: LearningMemory) -> Self {
        self.learning_memory = Some(memory);
        self
    }
    /// Indeks langkah jaringan.
    #[must_use]
    pub const fn completed_steps(&self) -> u64 {
        self.network.completed_steps()
    }
    /// Akses jaringan baca-saja.
    #[must_use]
    pub const fn network(&self) -> &LifNetwork {
        &self.network
    }
    /// Akses memori bila dipasang.
    #[must_use]
    pub const fn memory(&self) -> Option<&MemoryStore> {
        self.memory.as_ref()
    }
    /// Akses semantic memory bila dipasang.
    #[must_use]
    pub const fn semantic_memory(&self) -> Option<&SemanticMemory> {
        self.semantic_memory.as_ref()
    }
    /// Akses store pengalaman bila dipasang.
    #[must_use]
    pub const fn learning_memory(&self) -> Option<&LearningMemory> {
        self.learning_memory.as_ref()
    }

    pub(crate) fn checkpoint_fingerprints(
        &self,
    ) -> Result<crate::CheckpointFingerprints, SimError> {
        if self
            .semantic_memory
            .as_ref()
            .is_some_and(|memory| !memory.pending_learning_task_deletions().is_empty())
        {
            return Err(SimError::InvalidInput {
                parameter: "semantic_memory",
                reason: "penghapusan task neural belum disinkronkan",
            });
        }
        Ok(crate::CheckpointFingerprints {
            memory_store: self
                .memory
                .as_ref()
                .map(MemoryStore::checkpoint_fingerprint)
                .transpose()?,
            learning_memory: self
                .learning_memory
                .as_ref()
                .map(LearningMemory::checkpoint_fingerprint)
                .transpose()?,
            semantic_memory: self
                .semantic_memory
                .as_ref()
                .map(SemanticMemory::checkpoint_fingerprint)
                .transpose()?,
        })
    }

    pub(crate) fn synchronize_persistent_memories(&mut self) -> Result<usize, SimError> {
        let Some(semantic_memory) = self.semantic_memory.as_ref() else {
            return Ok(0);
        };
        let pending = semantic_memory.pending_learning_task_deletions();
        if pending.is_empty() {
            return Ok(0);
        }
        if self.learning_memory.is_none() {
            return Err(SimError::InvalidInput {
                parameter: "learning_memory",
                reason: "penghapusan semantic task membutuhkan LearningMemory terpasang",
            });
        }
        let mut removed = 0;
        for task_id in pending {
            removed += self
                .learning_memory
                .as_mut()
                .ok_or(SimError::InvalidInput {
                    parameter: "learning_memory",
                    reason: "LearningMemory dilepas saat sinkronisasi",
                })?
                .discard_task(&task_id)?;
            self.semantic_memory
                .as_mut()
                .ok_or(SimError::InvalidInput {
                    parameter: "semantic_memory",
                    reason: "SemanticMemory dilepas saat sinkronisasi",
                })?
                .complete_learning_task_deletion(&task_id)?;
        }
        Ok(removed)
    }
    /// Akses encoder teks bila dipasang.
    #[must_use]
    pub const fn text_encoder(&self) -> Option<&TextEncoder> {
        self.text_encoder.as_ref()
    }
    /// Membuat handle sesi eksklusif untuk mengirim input ke engine.
    pub fn session(&mut self) -> Session<'_> {
        Session { engine: self }
    }
    /// Menyimpan state jaringan dan evaluator; konfigurasi encoder/policy dapat dipasang kembali saat startup.
    pub fn save_checkpoint(
        &self,
        path: impl AsRef<std::path::Path>,
        evaluator: Option<&crate::RewardEvaluator>,
    ) -> Result<(), SimError> {
        super::persistence::save(path, &self.network, evaluator)
    }

    /// Menyimpan neural state sambil mengikat snapshot ke fingerprint semua store terpasang.
    pub fn save_integrated_checkpoint(
        &self,
        path: impl AsRef<std::path::Path>,
        evaluator: Option<&crate::RewardEvaluator>,
    ) -> Result<(), SimError> {
        crate::Checkpoint::save_integrated_engine(
            path,
            &self.network,
            evaluator,
            &self.checkpoint_fingerprints()?,
        )
    }
}
