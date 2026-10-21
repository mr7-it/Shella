use super::validation::{validate_action_label, validate_readout_update, validate_reasoning_input};
use super::{ReasoningEngine, SemanticLearningOutcome};
use crate::{LanguageNetwork, LearningOutcome, LifNetwork, SimError};

impl ReasoningEngine {
    /// Mengubah teks melalui semantic graph, menyimpan konteks/pengalaman, dan memperbarui LIF.
    #[allow(clippy::too_many_arguments)]
    pub fn learn_from_semantic_feedback(
        &mut self,
        context_id: &str,
        text: &str,
        language_network: &LanguageNetwork,
        current_per_word_na: f64,
        expected_action: Option<usize>,
        predicted_action: Option<usize>,
        action_neurons: &[usize],
        learning_rate: f64,
    ) -> Result<SemanticLearningOutcome, SimError> {
        self.synchronize_persistent_memories()?;
        if self.network.neuron_count() != language_network.network().neuron_count()
            || (0..self.network.neuron_count()).any(|index| {
                self.network.neuron_id(index) != language_network.network().neuron_id(index)
            })
        {
            return Err(SimError::InvalidInput {
                parameter: "language_network",
                reason: "identitas/populasi neuron tidak cocok dengan reasoning engine",
            });
        }
        let (analysis, gaps, currents) = {
            let semantic_memory = self
                .semantic_memory
                .as_mut()
                .ok_or(SimError::InvalidInput {
                    parameter: "semantic_memory",
                    reason: "ReasoningEngine belum memasang SemanticMemory",
                })?;
            let analysis = semantic_memory.analyze_in_context(context_id, text)?;
            let gaps = semantic_memory.detect_gaps(text)?;
            if !gaps.unknown_tokens.is_empty()
                || !gaps.ambiguous_tokens.is_empty()
                || !gaps.conflicting_tokens.is_empty()
            {
                return Ok(SemanticLearningOutcome::new(analysis, gaps, None));
            }
            let currents = semantic_memory
                .encode_currents(language_network, text, current_per_word_na)?
                .currents_na;
            (analysis, gaps, currents)
        };
        let learning = self.learn_from_feedback(
            context_id,
            &currents,
            expected_action,
            predicted_action,
            action_neurons,
            learning_rate,
        )?;
        Ok(SemanticLearningOutcome::new(analysis, gaps, Some(learning)))
    }

    /// Menyimpan feedback pengalaman dan memperbarui sinaps readout yang sesuai.
    pub fn learn_from_feedback(
        &mut self,
        context_id: &str,
        features: &[f64],
        expected_action: Option<usize>,
        predicted_action: Option<usize>,
        action_neurons: &[usize],
        learning_rate: f64,
    ) -> Result<LearningOutcome, SimError> {
        self.synchronize_persistent_memories()?;
        validate_reasoning_input(&self.network, features, action_neurons)?;
        validate_action_label(expected_action, action_neurons.len(), "expected_action")?;
        validate_action_label(predicted_action, action_neurons.len(), "predicted_action")?;
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(SimError::InvalidInput {
                parameter: "learning_rate",
                reason: "harus finite dan positif",
            });
        }
        let mut next_network = self.network.clone();
        let updated_synapses = if next_network.stdp_config().is_some() {
            train_with_stdp(
                &mut next_network,
                features,
                expected_action.map(|action| action_neurons[action]),
                expected_action.is_some_and(|action| predicted_action == Some(action)),
            )?
        } else if let Some(target_action) = expected_action {
            validate_readout_update(
                &self.network,
                features,
                target_action,
                action_neurons,
                learning_rate,
            )?;
            next_network.apply_rewarded_readout(
                features,
                target_action,
                action_neurons,
                learning_rate,
            )?
        } else {
            0
        };
        let old_weights = self
            .network
            .synapses()
            .iter()
            .map(|synapse| synapse.weight_na())
            .collect::<Vec<_>>();
        let updated_synapses = updated_synapses.max(
            next_network
                .synapses()
                .iter()
                .zip(old_weights)
                .filter(|(synapse, old_weight)| synapse.weight_na() != *old_weight)
                .count(),
        );
        let memory_write =
            self.memory
                .record_feedback(context_id, features, expected_action, predicted_action)?;
        self.network = next_network;
        Ok(LearningOutcome {
            memory_write,
            updated_synapses,
        })
    }
}

fn train_with_stdp(
    network: &mut LifNetwork,
    features: &[f64],
    teacher_neuron: Option<usize>,
    prediction_correct: bool,
) -> Result<usize, SimError> {
    let previous_weights = network
        .synapses()
        .iter()
        .map(|synapse| synapse.weight_na())
        .collect::<Vec<_>>();

    network.step(features)?;
    if let Some(target) = teacher_neuron {
        let refractory = network
            .neuron(target)
            .ok_or(SimError::InvalidTopology {
                reason: "neuron guru berada di luar jaringan",
            })?
            .refractory_remaining();
        if refractory > 1024 {
            return Err(SimError::InvalidInput {
                parameter: "teacher_neuron",
                reason: "masa refraktori melebihi batas 1024 langkah untuk satu sesi belajar",
            });
        }
        for _ in 0..refractory {
            network.step(&vec![0.0; network.neuron_count()])?;
        }

        let neuron = network.neuron(target).ok_or(SimError::InvalidTopology {
            reason: "neuron guru berada di luar jaringan",
        })?;
        let config = neuron.config();
        let ratio = config.dt_ms() / config.tau_m_ms();
        let decay = (-ratio).exp();
        let integrated_fraction = -(-ratio).exp_m1();
        let leaked_voltage = config.rest_potential_mv()
            + (neuron.membrane_potential_mv() - config.rest_potential_mv()) * decay;
        let denominator = integrated_fraction * config.resistance_mohm();
        let teacher_current = (config.threshold_mv() - leaked_voltage + 1e-6) / denominator;
        if !denominator.is_finite()
            || denominator <= 0.0
            || !teacher_current.is_finite()
            || teacher_current < 0.0
        {
            return Err(SimError::NumericalOverflow {
                operation: "arus pengajar STDP",
            });
        }
        let mut teacher_currents = vec![0.0; network.neuron_count()];
        teacher_currents[target] = teacher_current;
        let teacher_step = network.step(&teacher_currents)?;
        if !teacher_step.spiking_neurons().contains(&target) {
            return Err(SimError::InvalidInput {
                parameter: "teacher_neuron",
                reason: "arus pengajar tidak menghasilkan spike target",
            });
        }
    }
    if network.reward_modulated_stdp_config().is_some() {
        network.apply_reward(if prediction_correct { 1.0 } else { -1.0 })?;
    }
    Ok(network
        .synapses()
        .iter()
        .zip(previous_weights)
        .filter(|(synapse, previous)| synapse.weight_na() != *previous)
        .count())
}
