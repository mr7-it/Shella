use super::Session;
use crate::{LifNetwork, MemoryStore, SimError, decision::DecisionPolicy, language::TextEncoder};

/// Pemilik runtime dan subsistem opsional yang digunakan aplikasi.
#[derive(Debug)]
pub struct Engine {
    pub(crate) network: LifNetwork,
    pub(crate) text_encoder: Option<TextEncoder>,
    pub(crate) decision_policy: Option<DecisionPolicy>,
    pub(crate) memory: Option<MemoryStore>,
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
}
