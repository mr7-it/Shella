use super::{Engine, Input, Output};
use crate::{
    SimError,
    decision::{Candidate, Decision},
    evaluator::{RewardEvaluation, RewardEvaluator},
    language::EncodedText,
};
use serde_json::Value;
use std::path::Path;

/// Sesi eksklusif yang mengirim input ke satu engine.
pub struct Session<'a> {
    pub(crate) engine: &'a mut Engine,
}
impl Session<'_> {
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

    /// Menyimpan snapshot penuh jaringan dan metrik evaluator.
    pub fn save_checkpoint(
        &self,
        path: impl AsRef<Path>,
        evaluator: Option<&RewardEvaluator>,
    ) -> Result<(), SimError> {
        self.engine.save_checkpoint(path, evaluator)
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
