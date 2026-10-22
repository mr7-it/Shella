use super::{Engine, Input, Output};
use crate::{
    GeneratedText, GenerationConfig, GenerativeError, GenerativeLanguageModel, LanguageNetwork,
    ReasoningEngine, ReasoningResult, SemanticLearningOutcome, SimError,
    decision::{Candidate, Decision},
    evaluator::{RewardEvaluation, RewardEvaluator},
    language::SemanticResponse,
    language::{EncodedText, SemanticAnalysis, SemanticResonance},
};
use serde_json::Value;
use std::path::Path;

/// Sesi eksklusif yang mengirim input ke satu engine.
pub struct Session<'a> {
    pub(crate) engine: &'a mut Engine,
}
impl Session<'_> {
    /// Menghasilkan kelanjutan neural dari model decoder yang diberikan pemanggil.
    /// Jalur ini tidak memakai SemanticMemory dan hasilnya tidak terverifikasi faktual.
    pub fn generate_text(
        &self,
        model: &GenerativeLanguageModel,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<GeneratedText, GenerativeError> {
        model.generate(prompt, config)
    }

    /// Menjalankan input bahasa melalui semantic response dan ReasoningEngine.
    /// `ReasoningEngine` tetap menjadi pemilik tunggal SemanticMemory/LearningMemory.
    #[allow(clippy::too_many_arguments)]
    pub fn respond_with_reasoning(
        &mut self,
        reasoning: &mut ReasoningEngine,
        context_id: &str,
        text: &str,
        language_network: &LanguageNetwork,
        current_per_word_na: f64,
        action_neurons: &[usize],
        retrieval_limit: usize,
    ) -> Result<(SemanticResponse, ReasoningResult), SimError> {
        reasoning.respond_text(
            context_id,
            text,
            language_network,
            current_per_word_na,
            action_neurons,
            retrieval_limit,
        )
    }

    /// Menerapkan supervised feedback melalui jalur semantik dan LearningMemory.
    #[allow(clippy::too_many_arguments)]
    pub fn learn_from_semantic_feedback(
        &mut self,
        reasoning: &mut ReasoningEngine,
        context_id: &str,
        text: &str,
        language_network: &LanguageNetwork,
        current_per_word_na: f64,
        expected_action: Option<usize>,
        predicted_action: Option<usize>,
        action_neurons: &[usize],
        learning_rate: f64,
    ) -> Result<SemanticLearningOutcome, SimError> {
        reasoning.learn_from_semantic_feedback(
            context_id,
            text,
            language_network,
            current_per_word_na,
            expected_action,
            predicted_action,
            action_neurons,
            learning_rate,
        )
    }

    /// Memproses arus/teks melalui jaringan LIF dan mengembalikan spike yang dihasilkan.
    pub fn step(&mut self, input: Input) -> Result<Output, SimError> {
        let (kind, currents, encoded): (_, Vec<f64>, Option<EncodedText>) = match input {
            Input::Currents(currents) => (super::InputKind::Currents, currents, None),
            Input::Text {
                text,
                current_per_token_na,
            } => {
                let encoder = self
                    .engine
                    .text_encoder
                    .as_ref()
                    .ok_or(SimError::InvalidInput {
                        parameter: "text_encoder",
                        reason: "engine belum memasang TextEncoder",
                    })?;
                let encoded = encoder.encode(
                    &text,
                    self.engine.network.neuron_count(),
                    current_per_token_na,
                )?;
                (
                    super::InputKind::Text,
                    encoded.currents_na().to_vec(),
                    Some(encoded),
                )
            }
        };
        let step = self.engine.network.step(&currents)?;
        Ok(Output::new(kind, step, encoded))
    }

    /// Menilai kandidat dengan kebijakan terpasang; skor bukan confidence probabilistik.
    pub fn decide(&self, candidates: &[Candidate]) -> Result<Decision, SimError> {
        let policy = self
            .engine
            .decision_policy
            .as_ref()
            .ok_or(SimError::InvalidInput {
                parameter: "decision_policy",
                reason: "engine belum memasang DecisionPolicy",
            })?;
        policy.select(candidates)
    }

    /// Menilai kanal aksi dari neuron output pada hasil jaringan yang diberikan.
    pub fn evaluate_actions(
        &self,
        output: &Output,
        expected_action: Option<usize>,
        action_neurons: &[usize],
        evaluator: &mut RewardEvaluator,
    ) -> Result<RewardEvaluation, SimError> {
        let observed = self.action_observations(output, action_neurons)?;
        evaluator.evaluate(expected_action, &observed)
    }

    /// Menilai output dan menerapkan reward ke eligibility jaringan secara atomik.
    pub fn evaluate_actions_and_apply(
        &mut self,
        output: &Output,
        expected_action: Option<usize>,
        action_neurons: &[usize],
        evaluator: &mut RewardEvaluator,
    ) -> Result<RewardEvaluation, SimError> {
        let observed = self.action_observations(output, action_neurons)?;
        evaluator.evaluate_and_apply(expected_action, &observed, &mut self.engine.network)
    }

    /// Menulis/memperbarui record ke penyimpanan persisten yang terpasang.
    pub fn remember(
        &mut self,
        key: impl Into<String>,
        value: Value,
        tags: &[String],
    ) -> Result<String, SimError> {
        self.engine
            .memory
            .as_mut()
            .ok_or(SimError::InvalidInput {
                parameter: "memory_store",
                reason: "engine belum memasang MemoryStore",
            })?
            .put(key, value, tags)
    }

    /// Akses pengalaman persistent untuk pembelajaran terkontrol di aplikasi.
    pub fn learning_memory_mut(&mut self) -> Result<&mut crate::LearningMemory, SimError> {
        self.engine
            .learning_memory
            .as_mut()
            .ok_or(SimError::InvalidInput {
                parameter: "learning_memory",
                reason: "engine belum memasang LearningMemory",
            })
    }

    /// Menganalisis teks melalui semantic memory yang dipasang pada engine.
    pub fn analyze_semantics(&self, text: &str) -> Result<SemanticAnalysis, SimError> {
        self.engine
            .semantic_memory
            .as_ref()
            .ok_or(SimError::InvalidInput {
                parameter: "semantic_memory",
                reason: "engine belum memasang SemanticMemory",
            })?
            .analyze(text)
    }

    /// Menganalisis dan menyimpan kalimat pada konteks percakapan persisten.
    pub fn analyze_semantics_in_context(
        &mut self,
        context_id: &str,
        text: &str,
    ) -> Result<SemanticAnalysis, SimError> {
        self.engine.synchronize_persistent_memories()?;
        self.engine
            .semantic_memory
            .as_mut()
            .ok_or(SimError::InvalidInput {
                parameter: "semantic_memory",
                reason: "engine belum memasang SemanticMemory",
            })?
            .analyze_in_context(context_id, text)
    }

    /// Menjalankan relasi semantik sebagai arus pada LanguageNetwork LIF.
    pub fn resonate_semantics(
        &mut self,
        network: &mut LanguageNetwork,
        text: &str,
        current_per_word_na: f64,
    ) -> Result<SemanticResonance, SimError> {
        self.engine.synchronize_persistent_memories()?;
        self.engine
            .semantic_memory
            .as_ref()
            .ok_or(SimError::InvalidInput {
                parameter: "semantic_memory",
                reason: "engine belum memasang SemanticMemory",
            })?
            .resonate(network, text, current_per_word_na)
    }

    pub fn remove_semantic_concept(&mut self, word_id: crate::WordID) -> Result<bool, SimError> {
        if self.engine.semantic_memory.is_none() || self.engine.learning_memory.is_none() {
            return Err(SimError::InvalidInput {
                parameter: "integrated_memory",
                reason: "penghapusan konsep membutuhkan SemanticMemory dan LearningMemory",
            });
        }
        self.engine.synchronize_persistent_memories()?;
        let removed = self
            .engine
            .semantic_memory
            .as_mut()
            .ok_or(SimError::InvalidInput {
                parameter: "semantic_memory",
                reason: "semantic memory tidak tersedia",
            })?
            .remove_concept(word_id)?;
        self.engine.synchronize_persistent_memories()?;
        Ok(removed)
    }

    pub fn forget_semantic_context(&mut self, context_id: &str) -> Result<bool, SimError> {
        if self.engine.semantic_memory.is_none() || self.engine.learning_memory.is_none() {
            return Err(SimError::InvalidInput {
                parameter: "integrated_memory",
                reason: "penghapusan konteks membutuhkan SemanticMemory dan LearningMemory",
            });
        }
        self.engine.synchronize_persistent_memories()?;
        let removed = self
            .engine
            .semantic_memory
            .as_mut()
            .ok_or(SimError::InvalidInput {
                parameter: "semantic_memory",
                reason: "semantic memory tidak tersedia",
            })?
            .forget_context(context_id)?;
        self.engine.synchronize_persistent_memories()?;
        Ok(removed)
    }

    /// Menyimpan snapshot penuh jaringan dan metrik evaluator.
    pub fn save_checkpoint(
        &self,
        path: impl AsRef<Path>,
        evaluator: Option<&RewardEvaluator>,
    ) -> Result<(), SimError> {
        self.engine.save_checkpoint(path, evaluator)
    }

    /// Menyimpan neural checkpoint yang diikat ke fingerprint store terpasang.
    pub fn save_integrated_checkpoint(
        &self,
        path: impl AsRef<Path>,
        evaluator: Option<&RewardEvaluator>,
    ) -> Result<(), SimError> {
        self.engine.save_integrated_checkpoint(path, evaluator)
    }

    /// Mengganti state jaringan dari checkpoint setelah validasi penuh; evaluator dikembalikan.
    pub fn restore_checkpoint(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<Option<RewardEvaluator>, SimError> {
        let (network, evaluator) = super::persistence::restore(path)?;
        self.engine.network = network;
        Ok(evaluator)
    }

    /// Memulihkan hanya jika seluruh store eksternal cocok dengan snapshot.
    pub fn restore_integrated_checkpoint(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<Option<RewardEvaluator>, SimError> {
        let restored = crate::Checkpoint::restore_integrated_engine(path)?;
        if self.engine.checkpoint_fingerprints()? != restored.fingerprints {
            return Err(SimError::InvalidCheckpoint {
                reason: "store Engine berubah sejak integrated checkpoint disimpan".to_owned(),
            });
        }
        self.engine.network = restored.network;
        Ok(restored.evaluator)
    }

    fn action_observations(
        &self,
        output: &Output,
        action_neurons: &[usize],
    ) -> Result<Vec<bool>, SimError> {
        if action_neurons.is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "action_neurons",
                reason: "harus memuat setidaknya satu kanal",
            });
        }
        let mut unique = std::collections::BTreeSet::new();
        if action_neurons
            .iter()
            .any(|index| *index >= self.engine.network.neuron_count() || !unique.insert(*index))
        {
            return Err(SimError::InvalidInput {
                parameter: "action_neurons",
                reason: "indeks harus valid dan unik",
            });
        }
        Ok(action_neurons
            .iter()
            .map(|index| output.spiking_neurons().contains(index))
            .collect())
    }
}
