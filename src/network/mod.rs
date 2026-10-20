use crate::{
    config::LifConfig,
    error::SimError,
    identity::{NeuronID, SynapseID},
    learning::{RewardModulatedStdpConfig, StdpConfig, StdpTrace},
    neuron::lif::{LifNeuron, LifStepPlan, StepOutcome},
};
mod neuron_state;
mod propagation;
mod scheduler;
mod spike_event;
mod synapse;

use self::spike_event::DelayedInput;
pub use self::synapse::Synapse;
use self::{
    neuron_state::validate_population, propagation::aggregate_currents, scheduler::schedule_next,
};
use serde::{Deserialize, Serialize};

/// Hasil satu langkah jaringan; posisi `outcomes` sesuai indeks neuron.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkStep {
    completed_step: u64,
    outcomes: Vec<StepOutcome>,
    spiking_neurons: Vec<usize>,
}

impl NetworkStep {
    /// Nomor langkah jaringan yang baru selesai, dimulai dari 1.
    #[must_use]
    pub const fn completed_step(&self) -> u64 {
        self.completed_step
    }

    /// Hasil langkah untuk setiap neuron, berdasarkan urutan indeks.
    #[must_use]
    pub fn outcomes(&self) -> &[StepOutcome] {
        &self.outcomes
    }

    /// Indeks neuron yang mengeluarkan spike pada langkah ini.
    #[must_use]
    pub fn spiking_neurons(&self) -> &[usize] {
        &self.spiking_neurons
    }
}

#[derive(Debug, Clone, PartialEq)]
struct PreparedLearning {
    weights: Vec<f64>,
    traces: Vec<StdpTrace>,
}

/// Jaringan LIF sinkron dengan urutan neuron dan koneksi yang stabil.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LifNetwork {
    neurons: Vec<LifNeuron>,
    #[serde(default)]
    neuron_ids: Vec<NeuronID>,
    synapses: Vec<Synapse>,
    #[serde(default)]
    synapse_ids: Vec<SynapseID>,
    #[serde(default = "first_synapse_id")]
    next_synapse_id: u64,
    pending_inputs: Vec<DelayedInput>,
    completed_steps: u64,
    stdp_config: Option<StdpConfig>,
    reward_config: Option<RewardModulatedStdpConfig>,
    stdp_traces: Vec<StdpTrace>,
}

const fn first_synapse_id() -> u64 {
    1
}

impl LifNetwork {
    /// Membuat jaringan homogen dengan jumlah neuron positif.
    pub fn new(config: LifConfig, neuron_count: usize) -> Result<Self, SimError> {
        config.validate()?;
        if neuron_count == 0 {
            return Err(SimError::InvalidTopology {
                reason: "jaringan harus memiliki sekurang-kurangnya satu neuron",
            });
        }
        let mut neurons = Vec::new();
        neurons
            .try_reserve_exact(neuron_count)
            .map_err(|_| SimError::AllocationFailed)?;
        for _ in 0..neuron_count {
            neurons.push(LifNeuron::new(config)?);
        }
        Self::from_neurons(neurons)
    }

    /// Membuat jaringan dengan parameter membran berbeda untuk tiap neuron.
    ///
    /// Semua konfigurasi harus memakai `dt_ms` identik sehingga langkah jaringan,
    /// delay sinaps, dan pembelajaran memiliki acuan waktu bersama.
    pub fn from_configs(configs: &[LifConfig]) -> Result<Self, SimError> {
        let Some(first) = configs.first() else {
            return Err(SimError::InvalidTopology {
                reason: "jaringan harus memiliki sekurang-kurangnya satu neuron",
            });
        };
        first.validate()?;
        let shared_dt = first.dt_ms();
        let mut neurons = Vec::new();
        neurons
            .try_reserve_exact(configs.len())
            .map_err(|_| SimError::AllocationFailed)?;
        for config in configs {
            config.validate()?;
            if config.dt_ms() != shared_dt {
                return Err(SimError::InvalidTopology {
                    reason: "semua neuron harus memakai dt_ms yang sama",
                });
            }
            neurons.push(LifNeuron::new(*config)?);
        }
        Self::from_neurons(neurons)
    }

    fn from_neurons(neurons: Vec<LifNeuron>) -> Result<Self, SimError> {
        let mut neuron_ids = Vec::new();
        neuron_ids
            .try_reserve_exact(neurons.len())
            .map_err(|_| SimError::AllocationFailed)?;
        for index in 0..neurons.len() {
            let value = u64::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(1))
                .ok_or(SimError::CounterOverflow {
                    counter: "ID neuron",
                })?;
            neuron_ids.push(NeuronID::new(value));
        }
        Ok(Self {
            neurons,
            neuron_ids,
            synapses: Vec::new(),
            synapse_ids: Vec::new(),
            next_synapse_id: first_synapse_id(),
            pending_inputs: Vec::new(),
            completed_steps: 0,
            stdp_config: None,
            reward_config: None,
            stdp_traces: Vec::new(),
        })
    }

    /// Menambahkan koneksi berarah jika endpoint sah dan pasangan unik.
    pub fn add_synapse(&mut self, synapse: Synapse) -> Result<(), SimError> {
        self.ensure_identity_state()?;
        if synapse.source >= self.neurons.len() || synapse.target >= self.neurons.len() {
            return Err(SimError::InvalidTopology {
                reason: "indeks sumber atau tujuan berada di luar jumlah neuron",
            });
        }
        if self
            .synapses
            .iter()
            .any(|existing| existing.source == synapse.source && existing.target == synapse.target)
        {
            return Err(SimError::DuplicateSynapse {
                source: synapse.source,
                target: synapse.target,
            });
        }
        self.synapses
            .try_reserve(1)
            .map_err(|_| SimError::AllocationFailed)?;
        self.synapse_ids
            .try_reserve(1)
            .map_err(|_| SimError::AllocationFailed)?;
        let next_synapse_id =
            self.next_synapse_id
                .checked_add(1)
                .ok_or(SimError::CounterOverflow {
                    counter: "ID sinaps",
                })?;
        if self.stdp_config.is_some() {
            self.stdp_traces
                .try_reserve(1)
                .map_err(|_| SimError::AllocationFailed)?;
        }
        self.synapses.push(synapse);
        self.synapse_ids.push(SynapseID::new(self.next_synapse_id));
        self.next_synapse_id = next_synapse_id;
        if self.stdp_config.is_some() {
            self.stdp_traces.push(StdpTrace::default());
        }
        Ok(())
    }

    /// Mengaktifkan STDP pair-based; trace baru dimulai dari nol.
    pub fn enable_stdp(&mut self, config: StdpConfig) -> Result<(), SimError> {
        config.validate()?;
        let mut traces = Vec::new();
        traces
            .try_reserve_exact(self.synapses.len())
            .map_err(|_| SimError::AllocationFailed)?;
        traces.resize(self.synapses.len(), StdpTrace::default());
        self.stdp_traces = traces;
        self.stdp_config = Some(config);
        self.reward_config = None;
        Ok(())
    }

    /// Mengaktifkan STDP termodulasi reward dengan eligibility trace yang meluruh.
    pub fn enable_reward_modulated_stdp(
        &mut self,
        config: RewardModulatedStdpConfig,
    ) -> Result<(), SimError> {
        config.validate()?;
        let mut traces = Vec::new();
        traces
            .try_reserve_exact(self.synapses.len())
            .map_err(|_| SimError::AllocationFailed)?;
        traces.resize(self.synapses.len(), StdpTrace::default());
        self.stdp_traces = traces;
        self.stdp_config = Some(config.stdp());
        self.reward_config = Some(config);
        Ok(())
    }

    /// Mematikan STDP dan membuang trace pembelajaran, tanpa mengubah bobot.
    pub fn disable_stdp(&mut self) {
        self.stdp_config = None;
        self.reward_config = None;
        self.stdp_traces.clear();
    }

    /// Memproses satu langkah dengan arus eksternal per neuron.
    ///
    /// Fase pertama menghitung semua neuron dari keadaan awal langkah dan arus
    /// sinaptik yang jatuh tempo. Fase kedua menerapkan semua transisi, menjadwalkan
    /// peristiwa spike, dan menerapkan pembaruan STDP. Jika perhitungan gagal,
    /// keadaan neuron, bobot, trace, antrean peristiwa, dan indeks langkah tidak
    /// berubah. Bobot sinaps yang memicu transmisi diambil sebelum STDP langkah itu.
    pub fn step(&mut self, external_currents_na: &[f64]) -> Result<NetworkStep, SimError> {
        let next_step = self
            .completed_steps
            .checked_add(1)
            .ok_or(SimError::StepCounterOverflow)?;
        if external_currents_na.len() != self.neurons.len() {
            return Err(SimError::InvalidInput {
                parameter: "external_currents_na",
                reason: "panjang irisan harus sama dengan jumlah neuron",
            });
        }
        if external_currents_na
            .iter()
            .any(|current| !current.is_finite())
        {
            return Err(SimError::InvalidInput {
                parameter: "external_currents_na",
                reason: "semua arus harus finite",
            });
        }

        let total_currents =
            aggregate_currents(external_currents_na, &self.pending_inputs, next_step)?;

        let mut plans = Vec::new();
        plans
            .try_reserve_exact(self.neurons.len())
            .map_err(|_| SimError::AllocationFailed)?;
        for (neuron, current) in self.neurons.iter_mut().zip(total_currents) {
            plans.push(neuron.prepare_advance(current)?);
        }

        let next_pending = schedule_next(&self.pending_inputs, &self.synapses, &plans, next_step)?;

        let next_learning = self.prepare_learning(&plans)?;
        let mut outcomes = Vec::new();
        let mut spiking_neurons = Vec::new();
        outcomes
            .try_reserve_exact(plans.len())
            .map_err(|_| SimError::AllocationFailed)?;
        let spike_count = plans.iter().filter(|plan| plan.outcome.spiked).count();
        spiking_neurons
            .try_reserve_exact(spike_count)
            .map_err(|_| SimError::AllocationFailed)?;
        for (index, plan) in plans.iter().enumerate() {
            outcomes.push(plan.outcome);
            if plan.outcome.spiked {
                spiking_neurons.push(index);
            }
        }

        for (neuron, plan) in self.neurons.iter_mut().zip(plans) {
            neuron.commit_advance(plan);
        }
        if let Some(learning) = next_learning {
            for (synapse, weight) in self.synapses.iter_mut().zip(learning.weights) {
                synapse.weight_na = weight;
            }
            self.stdp_traces = learning.traces;
        }
        self.pending_inputs = next_pending;
        self.completed_steps = next_step;

        Ok(NetworkStep {
            completed_step: next_step,
            outcomes,
            spiking_neurons,
        })
    }

    /// Jumlah neuron.
    #[must_use]
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Jumlah sinaps.
    #[must_use]
    pub fn synapse_count(&self) -> usize {
        self.synapses.len()
    }

    /// ID stabil neuron berdasarkan indeks jaringan.
    #[must_use]
    pub fn neuron_id(&self, index: usize) -> Option<NeuronID> {
        self.neuron_ids.get(index).copied().or_else(|| {
            self.neurons
                .get(index)
                .and_then(|_| u64::try_from(index).ok())
                .and_then(|index| index.checked_add(1))
                .map(NeuronID::new)
        })
    }

    /// Indeks neuron yang memiliki identitas tersebut.
    #[must_use]
    pub fn neuron_index(&self, id: NeuronID) -> Option<usize> {
        self.neuron_ids
            .iter()
            .position(|candidate| *candidate == id)
            .or_else(|| {
                let index = id.get().checked_sub(1)?;
                let index = usize::try_from(index).ok()?;
                (self.neuron_ids.is_empty() && index < self.neurons.len()).then_some(index)
            })
    }

    /// Indeks langkah jaringan yang telah berhasil diselesaikan.
    #[must_use]
    pub const fn completed_steps(&self) -> u64 {
        self.completed_steps
    }

    /// Akses neuron berdasarkan indeks.
    #[must_use]
    pub fn neuron(&self, index: usize) -> Option<&LifNeuron> {
        self.neurons.get(index)
    }

    /// Daftar sinaps sesuai urutan penambahan yang deterministik.
    #[must_use]
    pub fn synapses(&self) -> &[Synapse] {
        &self.synapses
    }

    /// ID stabil sinaps berdasarkan urutan penambahan.
    #[must_use]
    pub fn synapse_id(&self, index: usize) -> Option<SynapseID> {
        self.synapse_ids.get(index).copied().or_else(|| {
            self.synapses
                .get(index)
                .and_then(|_| u64::try_from(index).ok())
                .and_then(|index| index.checked_add(1))
                .map(SynapseID::new)
        })
    }

    /// Akses sinaps berdasarkan identitas stabil.
    #[must_use]
    pub fn synapse_by_id(&self, id: SynapseID) -> Option<&Synapse> {
        self.synapse_ids
            .iter()
            .position(|candidate| *candidate == id)
            .or_else(|| {
                let index = id.get().checked_sub(1)?;
                let index = usize::try_from(index).ok()?;
                (self.synapse_ids.is_empty() && index < self.synapses.len()).then_some(index)
            })
            .and_then(|index| self.synapses.get(index))
    }

    /// Jumlah peristiwa sinaptik yang masih menunggu waktu tiba.
    #[must_use]
    pub fn pending_input_count(&self) -> usize {
        self.pending_inputs.len()
    }

    /// Nilai konfigurasi STDP jika pembelajaran aktif.
    #[must_use]
    pub const fn stdp_config(&self) -> Option<StdpConfig> {
        self.stdp_config
    }

    /// Konfigurasi reward-modulated STDP jika mode tiga faktor aktif.
    #[must_use]
    pub const fn reward_modulated_stdp_config(&self) -> Option<RewardModulatedStdpConfig> {
        self.reward_config
    }

    /// Mengirimkan reward signed ke eligibility yang telah terkumpul.
    ///
    /// Reward tidak memajukan waktu jaringan dan tidak menghapus eligibility;
    /// trace terus meluruh pada langkah jaringan berikutnya.
    pub fn apply_reward(&mut self, reward: f64) -> Result<(), SimError> {
        let Some(config) = self.reward_config else {
            return Err(SimError::InvalidInput {
                parameter: "reward",
                reason: "reward-modulated STDP tidak aktif",
            });
        };
        if !reward.is_finite() {
            return Err(SimError::InvalidInput {
                parameter: "reward",
                reason: "harus berupa bilangan hingga",
            });
        }
        if self.stdp_traces.len() != self.synapses.len() {
            return Err(SimError::InvalidTopology {
                reason: "jumlah trace STDP tidak cocok dengan jumlah sinaps",
            });
        }

        let scaled_reward = config.learning_rate() * reward;
        if !scaled_reward.is_finite() {
            return Err(SimError::NumericalOverflow {
                operation: "skala reward STDP",
            });
        }
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(self.synapses.len())
            .map_err(|_| SimError::AllocationFailed)?;
        for (synapse, trace) in self.synapses.iter().zip(&self.stdp_traces) {
            let weight_delta = scaled_reward * trace.eligibility;
            let candidate_weight = synapse.weight_na + weight_delta;
            if !weight_delta.is_finite() {
                return Err(SimError::NumericalOverflow {
                    operation: "modulasi eligibility oleh reward",
                });
            }
            if !candidate_weight.is_finite() {
                return Err(SimError::NumericalOverflow {
                    operation: "pembaruan bobot oleh reward",
                });
            }
            weights.push(candidate_weight.clamp(synapse.min_weight_na, synapse.max_weight_na));
        }
        for (synapse, weight) in self.synapses.iter_mut().zip(weights) {
            synapse.weight_na = weight;
        }
        Ok(())
    }

    /// Memperbarui readout berbasis reward bertanda dengan batas bobot sinaps.
    /// Sinaps menuju action target diperkuat proporsional fitur aktif,
    /// sedangkan sinaps action lain ditekan pada jaringan LIF aktual.
    pub fn apply_rewarded_readout(
        &mut self,
        features: &[f64],
        target_action: usize,
        action_neurons: &[usize],
        learning_rate: f64,
    ) -> Result<usize, SimError> {
        if features.is_empty() || features.iter().any(|value| !value.is_finite()) {
            return Err(SimError::InvalidInput {
                parameter: "features",
                reason: "fitur harus finite",
            });
        }
        if action_neurons.is_empty() || target_action >= action_neurons.len() {
            return Err(SimError::InvalidInput {
                parameter: "target_action",
                reason: "target berada di luar mapping action",
            });
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(SimError::InvalidInput {
                parameter: "learning_rate",
                reason: "harus finite dan positif",
            });
        }
        if action_neurons
            .iter()
            .any(|neuron| *neuron >= self.neurons.len())
        {
            return Err(SimError::InvalidInput {
                parameter: "action_neurons",
                reason: "neuron action di luar network",
            });
        }
        let target = action_neurons[target_action];
        let mut updates = 0;
        for synapse in &mut self.synapses {
            if synapse.source >= features.len() || !action_neurons.contains(&synapse.target) {
                continue;
            }
            let signed = if synapse.target == target { 1.0 } else { -1.0 };
            let old = synapse.weight_na;
            synapse.weight_na = (old + signed * learning_rate * features[synapse.source])
                .clamp(synapse.min_weight_na, synapse.max_weight_na);
            updates += usize::from(old != synapse.weight_na);
        }
        Ok(updates)
    }

    pub(crate) fn validate_restored_state(&self) -> Result<(), SimError> {
        validate_population(&self.neurons, self.completed_steps)?;
        if (!self.neuron_ids.is_empty() && self.neuron_ids.len() != self.neurons.len())
            || (!self.synapse_ids.is_empty() && self.synapse_ids.len() != self.synapses.len())
        {
            return Err(SimError::InvalidCheckpoint {
                reason: "jumlah ID tidak cocok dengan populasi atau sinaps".to_owned(),
            });
        }
        if self.neuron_ids.iter().any(|id| id.get() == 0)
            || self
                .neuron_ids
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.neuron_ids.len()
            || self.synapse_ids.iter().any(|id| id.get() == 0)
            || self
                .synapse_ids
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.synapse_ids.len()
            || self
                .synapse_ids
                .iter()
                .any(|id| id.get() >= self.next_synapse_id)
        {
            return Err(SimError::InvalidCheckpoint {
                reason: "identitas neuron/sinaps duplikat atau penghitung sinaps invalid"
                    .to_owned(),
            });
        }
        for (index, synapse) in self.synapses.iter().enumerate() {
            Synapse::new(
                synapse.source,
                synapse.target,
                synapse.weight_na,
                synapse.min_weight_na,
                synapse.max_weight_na,
                synapse.delay_steps,
            )
            .map_err(|error| SimError::InvalidCheckpoint {
                reason: error.to_string(),
            })?;
            if synapse.source >= self.neurons.len()
                || synapse.target >= self.neurons.len()
                || self.synapses[..index]
                    .iter()
                    .any(|prior| prior.source == synapse.source && prior.target == synapse.target)
            {
                return Err(SimError::InvalidCheckpoint {
                    reason: "endpoint sinaps invalid atau pasangan duplikat".to_owned(),
                });
            }
        }
        for event in &self.pending_inputs {
            if event.due_step <= self.completed_steps
                || event.target >= self.neurons.len()
                || !event.current_na.is_finite()
            {
                return Err(SimError::InvalidCheckpoint {
                    reason: "antrean input tertunda tidak konsisten".to_owned(),
                });
            }
        }
        match (self.stdp_config, self.reward_config) {
            (None, None) if self.stdp_traces.is_empty() => {}
            (Some(stdp), reward) if self.stdp_traces.len() == self.synapses.len() => {
                stdp.validate()
                    .map_err(|error| SimError::InvalidCheckpoint {
                        reason: error.to_string(),
                    })?;
                if let Some(reward) = reward {
                    reward
                        .validate()
                        .map_err(|error| SimError::InvalidCheckpoint {
                            reason: error.to_string(),
                        })?;
                    if reward.stdp() != stdp {
                        return Err(SimError::InvalidCheckpoint {
                            reason: "konfigurasi STDP dan reward tidak cocok".to_owned(),
                        });
                    }
                }
                for trace in &self.stdp_traces {
                    if !trace.pre.is_finite()
                        || !trace.post.is_finite()
                        || !trace.eligibility.is_finite()
                        || (reward.is_none() && trace.eligibility != 0.0)
                    {
                        return Err(SimError::InvalidCheckpoint {
                            reason: "trace STDP tidak finite atau tidak sesuai mode".to_owned(),
                        });
                    }
                }
            }
            _ => {
                return Err(SimError::InvalidCheckpoint {
                    reason: "mode STDP, trace, dan sinaps tidak konsisten".to_owned(),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn migrate_restored_identity_state(&mut self) -> Result<(), SimError> {
        self.ensure_identity_state()?;
        self.validate_restored_state()
    }

    fn ensure_identity_state(&mut self) -> Result<(), SimError> {
        if self.neuron_ids.is_empty() {
            let mut ids = Vec::new();
            ids.try_reserve_exact(self.neurons.len())
                .map_err(|_| SimError::AllocationFailed)?;
            for index in 0..self.neurons.len() {
                let id = u64::try_from(index)
                    .ok()
                    .and_then(|index| index.checked_add(1))
                    .ok_or(SimError::CounterOverflow {
                        counter: "ID neuron",
                    })?;
                ids.push(NeuronID::new(id));
            }
            self.neuron_ids = ids;
        }
        if self.neuron_ids.len() != self.neurons.len() {
            return Err(SimError::InvalidTopology {
                reason: "jumlah ID neuron tidak cocok dengan populasi",
            });
        }
        if self.synapse_ids.is_empty() {
            let mut ids = Vec::new();
            ids.try_reserve_exact(self.synapses.len())
                .map_err(|_| SimError::AllocationFailed)?;
            for index in 0..self.synapses.len() {
                let id = u64::try_from(index)
                    .ok()
                    .and_then(|index| index.checked_add(1))
                    .ok_or(SimError::CounterOverflow {
                        counter: "ID sinaps",
                    })?;
                ids.push(SynapseID::new(id));
            }
            self.synapse_ids = ids;
            self.next_synapse_id = u64::try_from(self.synapses.len())
                .ok()
                .and_then(|count| count.checked_add(1))
                .ok_or(SimError::CounterOverflow {
                    counter: "ID sinaps",
                })?;
        }
        if self.synapse_ids.len() != self.synapses.len() {
            return Err(SimError::InvalidTopology {
                reason: "jumlah ID sinaps tidak cocok dengan koneksi",
            });
        }
        Ok(())
    }

    fn prepare_learning(
        &self,
        plans: &[LifStepPlan],
    ) -> Result<Option<PreparedLearning>, SimError> {
        let Some(config) = self.stdp_config else {
            return Ok(None);
        };
        if self.stdp_traces.len() != self.synapses.len() {
            return Err(SimError::InvalidTopology {
                reason: "jumlah trace STDP tidak cocok dengan jumlah sinaps",
            });
        }
        let decay_plus = (-1.0 / config.tau_plus_steps()).exp();
        let decay_minus = (-1.0 / config.tau_minus_steps()).exp();
        let eligibility_decay = self
            .reward_config
            .map(|reward| (-1.0 / reward.eligibility_tau_steps()).exp());
        let mut weights = Vec::new();
        let mut traces = Vec::new();
        weights
            .try_reserve_exact(self.synapses.len())
            .map_err(|_| SimError::AllocationFailed)?;
        traces
            .try_reserve_exact(self.synapses.len())
            .map_err(|_| SimError::AllocationFailed)?;

        for (synapse, trace) in self.synapses.iter().zip(&self.stdp_traces) {
            let pre_spiked = plans[synapse.source].outcome.spiked;
            let post_spiked = plans[synapse.target].outcome.spiked;
            let decayed_pre = trace.pre * decay_plus;
            let decayed_post = trace.post * decay_minus;
            let potentiation = if post_spiked {
                config.a_plus() * decayed_pre
            } else {
                0.0
            };
            let depression = if pre_spiked {
                config.a_minus() * decayed_post
            } else {
                0.0
            };
            let delta = potentiation - depression;
            let decayed_eligibility = eligibility_decay
                .map(|decay| trace.eligibility * decay)
                .unwrap_or(0.0);
            let next_eligibility = decayed_eligibility
                + if eligibility_decay.is_some() {
                    delta
                } else {
                    0.0
                };
            let applied_delta = if eligibility_decay.is_some() {
                0.0
            } else {
                delta
            };
            let candidate_weight = synapse.weight_na + applied_delta;
            let next_pre = decayed_pre + if pre_spiked { 1.0 } else { 0.0 };
            let next_post = decayed_post + if post_spiked { 1.0 } else { 0.0 };
            for (operation, value) in [
                ("peluruhan trace pra-STDP", decayed_pre),
                ("peluruhan trace pasca-STDP", decayed_post),
                ("potensiasi STDP", potentiation),
                ("depresi STDP", depression),
                ("delta bobot STDP", delta),
                ("peluruhan eligibility", decayed_eligibility),
                ("akumulasi eligibility", next_eligibility),
                ("bobot kandidat STDP", candidate_weight),
                ("akumulasi trace pra-STDP", next_pre),
                ("akumulasi trace pasca-STDP", next_post),
            ] {
                if !value.is_finite() {
                    return Err(SimError::NumericalOverflow { operation });
                }
            }
            weights.push(candidate_weight.clamp(synapse.min_weight_na, synapse.max_weight_na));
            traces.push(StdpTrace {
                pre: next_pre,
                post: next_post,
                eligibility: next_eligibility,
            });
        }
        Ok(Some(PreparedLearning { weights, traces }))
    }
}
