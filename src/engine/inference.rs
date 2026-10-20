use super::analysis::{detect_contradictions, normalize, select_compute_budget, support_metrics};
use super::temporal::calculate_attention;
use super::validation::validate_reasoning_input;
use super::{ReasoningEngine, ReasoningResult, ReasoningStep};
use crate::reasoning::{ContextFrame, UncertaintyDecomposition};
use crate::{LanguageNetwork, SimError};
use std::collections::BTreeSet;

impl ReasoningEngine {
    /// Mengambil pengalaman sesuai konteks, menyuntikkan dukungan memori, lalu
    /// menjalankan jaringan selama konfigurasi iterasi yang diberikan.
    pub fn reason(
        &mut self,
        context_id: &str,
        input_currents_na: &[f64],
        action_neurons: &[usize],
    ) -> Result<ReasoningResult, SimError> {
        self.reason_internal(
            context_id,
            input_currents_na,
            input_currents_na,
            action_neurons,
        )
    }

    /// Mengubah teks melalui representasi kata-neuron lalu menjalankan retrieval,
    /// memori, dan inferensi pada jaringan engine yang identitas neuronnya cocok.
    pub fn reason_text(
        &mut self,
        context_id: &str,
        text: &str,
        language_network: &LanguageNetwork,
        current_per_word_na: f64,
        action_neurons: &[usize],
    ) -> Result<ReasoningResult, SimError> {
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
        let currents = language_network.encode_currents(text, current_per_word_na)?;
        self.reason(context_id, &currents, action_neurons)
    }

    /// Retrieval diarahkan oleh gabungan input dan fitur tujuan aktif.
    pub fn reason_for_goal(
        &mut self,
        context_id: &str,
        goal_features: &[f64],
        input_currents_na: &[f64],
        action_neurons: &[usize],
    ) -> Result<ReasoningResult, SimError> {
        validate_reasoning_input(&self.network, goal_features, action_neurons)?;
        validate_reasoning_input(&self.network, input_currents_na, action_neurons)?;
        let retrieval_query = input_currents_na
            .iter()
            .zip(goal_features)
            .map(|(input, goal)| (input + goal) / 2.0)
            .collect::<Vec<_>>();
        self.reason_internal(
            context_id,
            input_currents_na,
            &retrieval_query,
            action_neurons,
        )
    }

    fn reason_internal(
        &mut self,
        context_id: &str,
        input_currents_na: &[f64],
        retrieval_query: &[f64],
        action_neurons: &[usize],
    ) -> Result<ReasoningResult, SimError> {
        validate_reasoning_input(&self.network, input_currents_na, action_neurons)?;
        let recalled = self.memory.recall_task_similar(
            context_id,
            retrieval_query,
            self.config.recall_limit,
        )?;
        let attention = calculate_attention(&recalled, self.config.temporal_decay)?;
        let contradictions = detect_contradictions(&recalled)?;
        let compute_budget =
            select_compute_budget(self.config, input_currents_na, &recalled, &contradictions);
        let mut memory_support = vec![0.0; action_neurons.len()];
        for (recall, item_attention) in recalled.iter().zip(&attention) {
            if let Some(action) = recall.expected_action() {
                let support = memory_support
                    .get_mut(action)
                    .ok_or(SimError::InvalidInput {
                        parameter: "memory_action",
                        reason: "aksi tersimpan di luar mapping action neuron",
                    })?;
                *support += item_attention.weight;
                if !support.is_finite() {
                    return Err(SimError::NumericalOverflow {
                        operation: "agregasi dukungan memori",
                    });
                }
            }
        }
        normalize(&mut memory_support);

        let mut injected_currents_na = vec![0.0; self.network.neuron_count()];
        for (action, support) in memory_support.iter().enumerate() {
            let neuron = action_neurons[action];
            injected_currents_na[neuron] = self.config.memory_current_na * support;
            if !injected_currents_na[neuron].is_finite() {
                return Err(SimError::NumericalOverflow {
                    operation: "arus injeksi memori",
                });
            }
        }
        let mut first_currents = input_currents_na.to_vec();
        if self.config.context_window_capacity > 0
            && self.active_context_id.as_deref() == Some(context_id)
        {
            let context_frames = self.context_frames.iter().collect::<Vec<_>>();
            let denominator = context_frames.len().max(1) as f64;
            for frame in context_frames {
                for (current, previous) in first_currents.iter_mut().zip(&frame.currents_na) {
                    *current += previous / denominator;
                    if !current.is_finite() {
                        return Err(SimError::NumericalOverflow {
                            operation: "agregasi context window",
                        });
                    }
                }
            }
        }
        for (current, memory_current) in first_currents.iter_mut().zip(&injected_currents_na) {
            *current += memory_current;
            if !current.is_finite() {
                return Err(SimError::NumericalOverflow {
                    operation: "penggabungan arus input dan memori",
                });
            }
        }

        let mut steps = Vec::new();
        steps
            .try_reserve_exact(compute_budget)
            .map_err(|_| SimError::AllocationFailed)?;
        let mut neural_support = vec![0.0; action_neurons.len()];
        let mut observed_spikes = BTreeSet::new();
        for iteration in 0..compute_budget {
            let currents_na = if iteration == 0 {
                first_currents.clone()
            } else {
                vec![0.0; self.network.neuron_count()]
            };
            let network_step = self.network.step(&currents_na)?;
            let temporal_weight =
                (-self.config.temporal_decay * (compute_budget - iteration - 1) as f64).exp();
            observed_spikes.extend(network_step.spiking_neurons().iter().copied());
            for (action, neuron) in action_neurons.iter().enumerate() {
                if network_step.spiking_neurons().contains(neuron) {
                    neural_support[action] += temporal_weight;
                    if !neural_support[action].is_finite() {
                        return Err(SimError::NumericalOverflow {
                            operation: "akumulasi dukungan spike",
                        });
                    }
                }
            }
            steps.push(ReasoningStep {
                iteration: iteration + 1,
                currents_na,
                network_step,
            });
        }
        let memory_evidence = memory_support.iter().any(|support| *support > 0.0);
        let neural_total = neural_support.iter().sum::<f64>();
        let neural_evidence = neural_total > 0.0;
        normalize(&mut neural_support);
        let mut action_support = vec![0.0; action_neurons.len()];
        let streams = usize::from(memory_evidence) + usize::from(neural_evidence);
        if streams > 0 {
            for action in 0..action_support.len() {
                action_support[action] =
                    (memory_support[action] + neural_support[action]) / streams as f64;
            }
        }

        let (uncertainty, consistency) = support_metrics(&action_support);
        let memory_strength = if memory_evidence {
            recalled
                .iter()
                .zip(&attention)
                .filter(|(recall, _)| recall.expected_action().is_some())
                .map(|(recall, item)| recall.similarity() * item.weight)
                .sum::<f64>()
                / memory_support.iter().sum::<f64>().max(f64::MIN_POSITIVE)
        } else {
            0.0
        };
        let neural_strength = if neural_evidence {
            1.0 - (-neural_total).exp()
        } else {
            0.0
        };
        let evidence_streams = usize::from(memory_evidence) + usize::from(neural_evidence);
        let evidence_strength = if evidence_streams == 0 {
            0.0
        } else {
            (memory_strength + neural_strength) / evidence_streams as f64
        }
        .clamp(0.0, 1.0);
        let confidence = (evidence_strength * consistency * (1.0 - uncertainty)).clamp(0.0, 1.0);
        let calibrated_confidence = self
            .calibration
            .as_ref()
            .map(|state| state.calibrator().predict(confidence));
        let winner = action_support
            .iter()
            .enumerate()
            .filter(|(_, support)| **support > 0.0)
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index);
        let decision_confidence = calibrated_confidence.unwrap_or(confidence);
        let selected_action = (decision_confidence >= self.config.minimum_confidence)
            .then_some(winner)
            .flatten();

        if self.config.context_window_capacity > 0 {
            if self.active_context_id.as_deref() != Some(context_id) {
                self.context_frames.clear();
                self.active_context_id = Some(context_id.to_owned());
            }
            self.context_frames.push_back(ContextFrame {
                context_id: context_id.to_owned(),
                currents_na: first_currents,
                spiking_neurons: observed_spikes.into_iter().collect(),
                completed_step: self.network.completed_steps(),
            });
            while self.context_frames.len() > self.config.context_window_capacity {
                self.context_frames.pop_front();
            }
        }
        self.memory.record_recall_usage(
            &recalled
                .iter()
                .map(|memory| memory.record_id().to_owned())
                .collect::<Vec<_>>(),
        )?;

        Ok(ReasoningResult {
            selected_action,
            confidence,
            calibrated_confidence,
            uncertainty,
            evidence_strength,
            uncertainty_decomposition: UncertaintyDecomposition {
                prediction: uncertainty,
                evidence_gap: 1.0 - evidence_strength,
            },
            compute_budget,
            action_support,
            steps,
            recalled,
            attention,
            contradictions,
            injected_currents_na,
        })
    }
}
