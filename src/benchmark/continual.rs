use crate::{
    LifConfig, LifNetwork, RewardEvaluator, RewardEvaluatorConfig, RewardModulatedStdpConfig,
    SimError, StdpConfig, Synapse,
};
use std::collections::BTreeSet;

const PHASE_COUNT: usize = 2;
const ASSOCIATIONS_PER_PHASE: usize = 2;
const ASSOCIATION_COUNT: usize = PHASE_COUNT * ASSOCIATIONS_PER_PHASE;
const CUE_COUNT: usize = ASSOCIATION_COUNT + 1;
const ACTION_COUNT: usize = ASSOCIATION_COUNT;
const NETWORK_SIZE: usize = CUE_COUNT + ACTION_COUNT;
const WASHOUT_STEPS_PER_TRAINING_EXAMPLE: usize = 100;
const TRAIN_AMPLITUDE_NA: f64 = 100.0;
const VALIDATION_AMPLITUDES_NA: [f64; 4] = [80.0, 90.0, 110.0, 120.0];
const TEST_AMPLITUDES_NA: [f64; 2] = [97.0, 103.0];
const TEACHER_SPIKE_CURRENT_NA: f64 = 200.0;
const SHARED_CUE_MAX_WEIGHT_NA: f64 = 50.0;
const LEARNING_RATE: f64 = 2.5;

/// Konfigurasi eksperimen asosiasi bertahap multi-seed.
///
/// Tiap asosiasi memakai cue umum yang berbagi lintas kelas dan satu cue pembeda.
/// Bobot cue umum dibatasi eksplisit pada 50 nA; bobot cue pembeda pada 150 nA.
/// Training hanya memakai amplitudo cue 100 nA. Validasi memakai amplitudo
/// 80/90/110/120 nA dan test memakai 97/103 nA, sehingga kedua split perturbasi
/// terpisah dan tidak berbagi nilai amplitudo dengan training atau satu sama lain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinualBenchmarkConfig {
    seeds: Vec<u64>,
    epochs_per_phase: usize,
    training_examples_per_network: u64,
    training_steps_per_network: u64,
}

impl ContinualBenchmarkConfig {
    /// Membuat konfigurasi dengan seed unik dan jumlah epoch positif.
    pub fn new(seeds: Vec<u64>, epochs_per_phase: usize) -> Result<Self, SimError> {
        if seeds.is_empty() {
            return Err(SimError::InvalidConfiguration {
                parameter: "continual_seeds",
                reason: "harus memuat sekurang-kurangnya satu seed",
            });
        }
        if epochs_per_phase == 0 {
            return Err(SimError::InvalidConfiguration {
                parameter: "epochs_per_phase",
                reason: "harus lebih besar dari nol",
            });
        }
        let unique_seeds: BTreeSet<_> = seeds.iter().copied().collect();
        if unique_seeds.len() != seeds.len() {
            return Err(SimError::InvalidConfiguration {
                parameter: "continual_seeds",
                reason: "seed harus unik agar setiap run dapat diidentifikasi",
            });
        }
        let examples = epochs_per_phase
            .checked_mul(PHASE_COUNT)
            .and_then(|count| count.checked_mul(ASSOCIATIONS_PER_PHASE))
            .and_then(|count| u64::try_from(count).ok())
            .ok_or(SimError::CounterOverflow {
                counter: "contoh training continual benchmark",
            })?;
        let steps_per_example = 2_u64
            .checked_add(WASHOUT_STEPS_PER_TRAINING_EXAMPLE as u64)
            .ok_or(SimError::CounterOverflow {
                counter: "langkah per contoh continual benchmark",
            })?;
        let training_steps_per_network =
            examples
                .checked_mul(steps_per_example)
                .ok_or(SimError::CounterOverflow {
                    counter: "langkah continual benchmark",
                })?;
        Ok(Self {
            seeds,
            epochs_per_phase,
            training_examples_per_network: examples,
            training_steps_per_network,
        })
    }

    /// Protokol standar: lima seed eksplisit dan 32 epoch per fase.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            seeds: vec![1, 7, 19, 42, 99],
            epochs_per_phase: 32,
            training_examples_per_network: 128,
            training_steps_per_network: 13_056,
        }
    }

    /// Seed yang akan dijalankan berurutan.
    #[must_use]
    pub fn seeds(&self) -> &[u64] {
        &self.seeds
    }

    /// Jumlah epoch pada tiap fase akuisisi asosiasi.
    #[must_use]
    pub const fn epochs_per_phase(&self) -> usize {
        self.epochs_per_phase
    }

    /// Jumlah contoh teacher-forced per jaringan untuk seluruh fase.
    #[must_use]
    pub const fn training_examples_per_network(&self) -> u64 {
        self.training_examples_per_network
    }

    /// Jumlah langkah jaringan termasuk cue, teacher, dan washout per contoh.
    #[must_use]
    pub const fn training_steps_per_network(&self) -> u64 {
        self.training_steps_per_network
    }

    /// Menjalankan kontrol, pembelajaran bertahap, validasi, dan test held-out.
    pub fn run(&self) -> Result<ContinualBenchmarkResult, SimError> {
        let mut seed_results = Vec::with_capacity(self.seeds.len());
        for seed in &self.seeds {
            seed_results.push(run_seed(
                *seed,
                self.epochs_per_phase,
                self.training_examples_per_network,
                self.training_steps_per_network,
            )?);
        }
        Ok(ContinualBenchmarkResult {
            seed_results,
            training_examples_per_network: self.training_examples_per_network,
            training_steps_per_network: self.training_steps_per_network,
        })
    }
}

impl Default for ContinualBenchmarkConfig {
    fn default() -> Self {
        Self::standard()
    }
}

/// Hasil satu seed, termasuk akurasi task lama sebelum dan sesudah fase berikutnya.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContinualSeedResult {
    seed: u64,
    baseline_test_accuracy: f64,
    task_a_test_after_phase_one: f64,
    task_a_test_after_phase_two: f64,
    task_b_test_after_phase_two: f64,
    final_test_accuracy: f64,
    final_validation_accuracy: f64,
    validation_accuracy_by_amplitude: [f64; 4],
    training_reward: f64,
}

impl ContinualSeedResult {
    /// Seed yang mengendalikan urutan asosiasi training.
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }
    /// Akurasi kontrol tanpa training pada split test.
    #[must_use]
    pub const fn baseline_test_accuracy(self) -> f64 {
        self.baseline_test_accuracy
    }
    /// Akurasi task A setelah asosiasi task A dipelajari.
    #[must_use]
    pub const fn task_a_test_after_phase_one(self) -> f64 {
        self.task_a_test_after_phase_one
    }
    /// Akurasi task A setelah fase belajar task B selesai.
    #[must_use]
    pub const fn task_a_test_after_phase_two(self) -> f64 {
        self.task_a_test_after_phase_two
    }
    /// Akurasi task B pada akhir eksperimen.
    #[must_use]
    pub const fn task_b_test_after_phase_two(self) -> f64 {
        self.task_b_test_after_phase_two
    }
    /// Akurasi seluruh empat asosiasi pada split test.
    #[must_use]
    pub const fn final_test_accuracy(self) -> f64 {
        self.final_test_accuracy
    }
    /// Akurasi seluruh asosiasi pada split validasi.
    #[must_use]
    pub const fn final_validation_accuracy(self) -> f64 {
        self.final_validation_accuracy
    }
    /// Akurasi validasi untuk amplitudo `[80, 90, 110, 120]` nA.
    #[must_use]
    pub const fn validation_accuracy_by_amplitude(self) -> [f64; 4] {
        self.validation_accuracy_by_amplitude
    }
    /// Akumulasi reward teacher-forced; bukan ukuran recall mandiri.
    #[must_use]
    pub const fn training_reward(self) -> f64 {
        self.training_reward
    }
    /// Penurunan akurasi task A; nilai negatif (backward transfer positif) dijepit ke nol.
    #[must_use]
    pub fn forgetting_task_a(self) -> f64 {
        (self.task_a_test_after_phase_one - self.task_a_test_after_phase_two).max(0.0)
    }
    /// Gain final terhadap kontrol dalam poin persentase.
    #[must_use]
    pub fn accuracy_gain_percentage_points(self) -> f64 {
        (self.final_test_accuracy - self.baseline_test_accuracy) * 100.0
    }
}

/// Ringkasan run multi-seed, mempertahankan setiap hasil per seed untuk audit.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinualBenchmarkResult {
    seed_results: Vec<ContinualSeedResult>,
    training_examples_per_network: u64,
    training_steps_per_network: u64,
}

impl ContinualBenchmarkResult {
    /// Hasil individual, dalam urutan seed konfigurasi.
    #[must_use]
    pub fn seed_results(&self) -> &[ContinualSeedResult] {
        &self.seed_results
    }
    /// Contoh teacher-forced yang dijalankan per seed.
    #[must_use]
    pub const fn training_examples_per_network(&self) -> u64 {
        self.training_examples_per_network
    }

    /// Langkah jaringan training per seed termasuk interval washout.
    #[must_use]
    pub const fn training_steps_per_network(&self) -> u64 {
        self.training_steps_per_network
    }

    /// Mean akurasi test final di seluruh seed.
    #[must_use]
    pub fn mean_final_test_accuracy(&self) -> f64 {
        mean(
            self.seed_results
                .iter()
                .map(|result| result.final_test_accuracy),
        )
    }
    /// Akurasi test final terendah di seluruh seed.
    #[must_use]
    pub fn minimum_final_test_accuracy(&self) -> f64 {
        self.seed_results
            .iter()
            .map(|result| result.final_test_accuracy)
            .fold(f64::INFINITY, f64::min)
    }
    /// Mean penurunan akurasi task A setelah fase task B.
    #[must_use]
    pub fn mean_forgetting_task_a(&self) -> f64 {
        mean(
            self.seed_results
                .iter()
                .map(|result| result.forgetting_task_a()),
        )
    }
    /// Gain mean test final terhadap baseline dalam poin persentase.
    #[must_use]
    pub fn mean_accuracy_gain_percentage_points(&self) -> f64 {
        mean(
            self.seed_results
                .iter()
                .map(|result| result.accuracy_gain_percentage_points()),
        )
    }
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let collected: Vec<_> = values.collect();
    collected.iter().sum::<f64>() / collected.len() as f64
}

fn run_seed(
    seed: u64,
    epochs_per_phase: usize,
    training_examples_per_network: u64,
    training_steps_per_network: u64,
) -> Result<ContinualSeedResult, SimError> {
    let evaluator_config = RewardEvaluatorConfig::new(1.0, 1.0);
    let baseline = make_network()?;
    let baseline_test_accuracy = evaluate_all(&baseline, &TEST_AMPLITUDES_NA, evaluator_config)?;

    let mut learned = make_network()?;
    let mut training_evaluator = RewardEvaluator::new(evaluator_config)?;
    let mut random_state = if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
    };
    let mut task_a_test_after_phase_one = None;

    for phase in 0..PHASE_COUNT {
        for _ in 0..epochs_per_phase {
            let mut association_order = [0, 1];
            deterministic_shuffle(&mut association_order, &mut random_state);
            for local_association in association_order {
                let cue = phase * ASSOCIATIONS_PER_PHASE + local_association;
                train_association(&mut learned, cue, &mut training_evaluator)?;
            }
        }
        if phase == 0 {
            // Probe dilakukan di clone yang dibuang agar eligibility held-out tidak
            // terbawa ke update reward pada fase training berikutnya.
            task_a_test_after_phase_one = Some(evaluate_task(
                &learned,
                0,
                &TEST_AMPLITUDES_NA,
                evaluator_config,
            )?);
        }
    }

    let task_a_test_after_phase_one =
        task_a_test_after_phase_one.ok_or(SimError::InvalidTopology {
            reason: "fase pertama benchmark continual tidak dijalankan",
        })?;
    let task_a_test_after_phase_two =
        evaluate_task(&learned, 0, &TEST_AMPLITUDES_NA, evaluator_config)?;
    let task_b_test_after_phase_two =
        evaluate_task(&learned, 1, &TEST_AMPLITUDES_NA, evaluator_config)?;
    let final_test_accuracy = evaluate_all(&learned, &TEST_AMPLITUDES_NA, evaluator_config)?;
    let mut validation_accuracy_by_amplitude = [0.0; VALIDATION_AMPLITUDES_NA.len()];
    for (index, amplitude) in VALIDATION_AMPLITUDES_NA.iter().enumerate() {
        validation_accuracy_by_amplitude[index] =
            evaluate_all(&learned, std::slice::from_ref(amplitude), evaluator_config)?;
    }
    let final_validation_accuracy = mean(validation_accuracy_by_amplitude.into_iter());

    let expected_training_samples = training_examples_per_network;
    if training_evaluator.metrics().samples() != expected_training_samples {
        return Err(SimError::InvalidTopology {
            reason: "jumlah sampel reward benchmark tidak sesuai konfigurasi",
        });
    }
    if learned.completed_steps() != training_steps_per_network {
        return Err(SimError::InvalidTopology {
            reason: "jumlah langkah jaringan benchmark tidak sesuai konfigurasi",
        });
    }
    Ok(ContinualSeedResult {
        seed,
        baseline_test_accuracy,
        task_a_test_after_phase_one,
        task_a_test_after_phase_two,
        task_b_test_after_phase_two,
        final_test_accuracy,
        final_validation_accuracy,
        validation_accuracy_by_amplitude,
        training_reward: training_evaluator.metrics().total_reward(),
    })
}

fn make_network() -> Result<LifNetwork, SimError> {
    let neuron_config = LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1);
    let mut network = LifNetwork::new(neuron_config, NETWORK_SIZE)?;
    for cue in 0..CUE_COUNT {
        for action in 0..ACTION_COUNT {
            let max_weight = if cue == 0 {
                SHARED_CUE_MAX_WEIGHT_NA
            } else {
                150.0
            };
            network.add_synapse(Synapse::new(
                cue,
                CUE_COUNT + action,
                0.0,
                0.0,
                max_weight,
                1,
            )?)?;
        }
    }
    network.enable_reward_modulated_stdp(RewardModulatedStdpConfig::new(
        StdpConfig::new(1.0, 1.0, 10.0, 10.0),
        20.0,
        LEARNING_RATE,
    ))?;
    Ok(network)
}

fn train_association(
    network: &mut LifNetwork,
    cue: usize,
    evaluator: &mut RewardEvaluator,
) -> Result<(), SimError> {
    let mut cue_currents = [0.0; NETWORK_SIZE];
    let distinguishing_cue = cue + 1;
    cue_currents[0] = TRAIN_AMPLITUDE_NA;
    cue_currents[distinguishing_cue] = TRAIN_AMPLITUDE_NA;
    let cue_response = network.step(&cue_currents)?;
    if !cue_response.spiking_neurons().contains(&0)
        || !cue_response.spiking_neurons().contains(&distinguishing_cue)
    {
        return Err(SimError::InvalidTopology {
            reason: "cue training benchmark tidak menghasilkan spike",
        });
    }

    let target = CUE_COUNT + cue;
    let mut teacher_currents = [0.0; NETWORK_SIZE];
    teacher_currents[target] = TEACHER_SPIKE_CURRENT_NA;
    let teacher_response = network.step(&teacher_currents)?;
    if !teacher_response.spiking_neurons().contains(&target) {
        return Err(SimError::InvalidTopology {
            reason: "teacher training benchmark tidak menghasilkan spike target",
        });
    }
    let observed: Vec<_> = (0..ACTION_COUNT)
        .map(|action| {
            teacher_response
                .spiking_neurons()
                .contains(&(CUE_COUNT + action))
        })
        .collect();
    evaluator.evaluate_and_apply(Some(cue), &observed, network)?;
    for _ in 0..WASHOUT_STEPS_PER_TRAINING_EXAMPLE {
        network.step(&[0.0; NETWORK_SIZE])?;
    }
    Ok(())
}

fn evaluate_all(
    network: &LifNetwork,
    amplitudes: &[f64],
    evaluator_config: RewardEvaluatorConfig,
) -> Result<f64, SimError> {
    evaluate_task_range(network, 0, PHASE_COUNT, amplitudes, evaluator_config)
}

fn evaluate_task(
    network: &LifNetwork,
    phase: usize,
    amplitudes: &[f64],
    evaluator_config: RewardEvaluatorConfig,
) -> Result<f64, SimError> {
    evaluate_task_range(network, phase, phase + 1, amplitudes, evaluator_config)
}

fn evaluate_task_range(
    source: &LifNetwork,
    first_phase: usize,
    end_phase: usize,
    amplitudes: &[f64],
    evaluator_config: RewardEvaluatorConfig,
) -> Result<f64, SimError> {
    let mut evaluator = RewardEvaluator::new(evaluator_config)?;
    for phase in first_phase..end_phase {
        for cue in phase * ASSOCIATIONS_PER_PHASE..(phase + 1) * ASSOCIATIONS_PER_PHASE {
            for amplitude in amplitudes {
                let mut probe = source.clone();
                let observed = run_recall_trial(&mut probe, cue, *amplitude)?;
                evaluator.evaluate(Some(cue), &observed)?;
            }
        }
    }
    evaluator
        .metrics()
        .accuracy()
        .ok_or(SimError::InvalidTopology {
            reason: "split evaluasi benchmark tidak memiliki sampel",
        })
}

fn run_recall_trial(
    network: &mut LifNetwork,
    cue: usize,
    amplitude: f64,
) -> Result<Vec<bool>, SimError> {
    let mut cue_currents = [0.0; NETWORK_SIZE];
    cue_currents[0] = amplitude;
    cue_currents[cue + 1] = amplitude;
    network.step(&cue_currents)?;
    let response = network.step(&[0.0; NETWORK_SIZE])?;
    Ok((0..ACTION_COUNT)
        .map(|action| response.spiking_neurons().contains(&(CUE_COUNT + action)))
        .collect())
}

fn deterministic_shuffle(values: &mut [usize], state: &mut u64) {
    for index in (1..values.len()).rev() {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let other = (*state % (index as u64 + 1)) as usize;
        values.swap(index, other);
    }
}

#[cfg(test)]
mod tests {
    use super::ContinualBenchmarkConfig;
    use crate::SimError;

    #[test]
    fn konfigurasi_menolak_seed_kosong_duplikat_dan_epoch_nol() {
        assert!(matches!(
            ContinualBenchmarkConfig::new(vec![], 1),
            Err(SimError::InvalidConfiguration { .. })
        ));
        assert!(matches!(
            ContinualBenchmarkConfig::new(vec![1, 1], 1),
            Err(SimError::InvalidConfiguration { .. })
        ));
        assert!(matches!(
            ContinualBenchmarkConfig::new(vec![1], 0),
            Err(SimError::InvalidConfiguration { .. })
        ));
    }
}
