use crate::{
    DecisionEvaluator, DecisionMetrics, LifConfig, LifNetwork, RewardEvaluator,
    RewardEvaluatorConfig, RewardModulatedStdpConfig, SimError, StdpConfig, Synapse,
    decision::AbstentionReason,
};
use std::collections::BTreeSet;

const FEATURE_COUNT: usize = 4;
const ACTION_COUNT: usize = 2;
const ACTION_OFFSET: usize = FEATURE_COUNT;
const NETWORK_SIZE: usize = FEATURE_COUNT + ACTION_COUNT;
const TASK_VARIANT_COUNT: usize = 4;
const TRAIN_AMPLITUDES_NA: [f64; TASK_VARIANT_COUNT] = [90.0, 100.0, 110.0, 120.0];
const TEST_AMPLITUDE_OFFSETS_NA: [f64; 2] = [-10.0, 10.0];
const NOISE_LEVELS_NA: [f64; 5] = [0.0, 10.0, 20.0, 40.0, 60.0];
const TRAINING_WASHOUT_STEPS: usize = 100;

/// Konfigurasi eksperimen pola komposisional held-out dengan noise input terukur.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneralizationBenchmarkConfig {
    seeds: Vec<u64>,
    task_variants: usize,
    training_epochs: usize,
    repetitions_per_pattern: usize,
}

impl GeneralizationBenchmarkConfig {
    pub fn new(
        seeds: Vec<u64>,
        task_variants: usize,
        training_epochs: usize,
        repetitions_per_pattern: usize,
    ) -> Result<Self, SimError> {
        if seeds.is_empty()
            || task_variants == 0
            || task_variants > TASK_VARIANT_COUNT
            || training_epochs == 0
            || repetitions_per_pattern == 0
        {
            return Err(SimError::InvalidConfiguration {
                parameter: "generalization_benchmark",
                reason: "seed, epoch, dan repetisi harus positif; variasi tugas maksimum empat",
            });
        }
        if seeds.iter().copied().collect::<BTreeSet<_>>().len() != seeds.len() {
            return Err(SimError::InvalidConfiguration {
                parameter: "generalization_seeds",
                reason: "seed harus unik",
            });
        }
        seeds
            .len()
            .checked_mul(task_variants)
            .and_then(|count| count.checked_mul(training_epochs))
            .and_then(|count| count.checked_mul(repetitions_per_pattern))
            .ok_or(SimError::CounterOverflow {
                counter: "ukuran generalization benchmark",
            })?;
        Ok(Self {
            seeds,
            task_variants,
            training_epochs,
            repetitions_per_pattern,
        })
    }

    /// Delapan inisialisasi × empat pemetaan fitur/kelas × 32 epoch.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            seeds: vec![17, 31, 53, 79, 109, 151, 211, 307],
            task_variants: TASK_VARIANT_COUNT,
            training_epochs: 32,
            repetitions_per_pattern: 24,
        }
    }

    #[must_use]
    pub fn seeds(&self) -> &[u64] {
        &self.seeds
    }
    #[must_use]
    pub const fn task_variants(&self) -> usize {
        self.task_variants
    }
    #[must_use]
    pub const fn training_epochs(&self) -> usize {
        self.training_epochs
    }
    #[must_use]
    pub const fn repetitions_per_pattern(&self) -> usize {
        self.repetitions_per_pattern
    }
    #[must_use]
    pub const fn noise_levels_na(&self) -> &'static [f64] {
        &NOISE_LEVELS_NA
    }

    pub fn run(&self) -> Result<GeneralizationBenchmarkResult, SimError> {
        let run_count =
            self.seeds
                .len()
                .checked_mul(self.task_variants)
                .ok_or(SimError::CounterOverflow {
                    counter: "jumlah run generalization benchmark",
                })?;
        let mut runs = Vec::new();
        runs.try_reserve_exact(run_count)
            .map_err(|_| SimError::AllocationFailed)?;
        for &seed in &self.seeds {
            for variant in 0..self.task_variants {
                runs.push(run_task(
                    seed,
                    variant,
                    self.training_epochs,
                    self.repetitions_per_pattern,
                )?);
            }
        }
        Ok(GeneralizationBenchmarkResult { runs })
    }
}

impl Default for GeneralizationBenchmarkConfig {
    fn default() -> Self {
        Self::standard()
    }
}

/// Metrik test untuk satu tingkat noise input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizationNoiseResult {
    noise_na: f64,
    metrics: DecisionMetrics,
}

impl GeneralizationNoiseResult {
    #[must_use]
    pub const fn noise_na(self) -> f64 {
        self.noise_na
    }
    #[must_use]
    pub const fn metrics(self) -> DecisionMetrics {
        self.metrics
    }
}

/// Hasil baseline dan jaringan belajar untuk satu inisialisasi/tugas.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralizationRunResult {
    seed: u64,
    task_variant: usize,
    initialization_fingerprint: u64,
    baseline: Vec<GeneralizationNoiseResult>,
    trained: Vec<GeneralizationNoiseResult>,
}

impl GeneralizationRunResult {
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }
    #[must_use]
    pub const fn task_variant(&self) -> usize {
        self.task_variant
    }
    #[must_use]
    pub const fn initialization_fingerprint(&self) -> u64 {
        self.initialization_fingerprint
    }
    #[must_use]
    pub fn baseline(&self) -> &[GeneralizationNoiseResult] {
        &self.baseline
    }
    #[must_use]
    pub fn trained(&self) -> &[GeneralizationNoiseResult] {
        &self.trained
    }
}

/// Agregat seluruh pasangan inisialisasi dan variasi tugas.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralizationBenchmarkResult {
    runs: Vec<GeneralizationRunResult>,
}

impl GeneralizationBenchmarkResult {
    #[must_use]
    pub fn runs(&self) -> &[GeneralizationRunResult] {
        &self.runs
    }
    #[must_use]
    pub fn mean_baseline_accuracy(&self, noise_na: f64) -> Option<f64> {
        mean_metric(
            self.runs
                .iter()
                .filter_map(|run| find_condition(&run.baseline, noise_na))
                .map(DecisionMetrics::accuracy),
        )
    }
    #[must_use]
    pub fn mean_trained_accuracy(&self, noise_na: f64) -> Option<f64> {
        mean_metric(
            self.runs
                .iter()
                .filter_map(|run| find_condition(&run.trained, noise_na))
                .map(DecisionMetrics::accuracy),
        )
    }
    #[must_use]
    pub fn mean_false_positive_rate(&self, noise_na: f64) -> Option<f64> {
        mean_metric(self.runs.iter().map(|run| {
            find_condition(&run.trained, noise_na).and_then(DecisionMetrics::false_positive_rate)
        }))
    }
    #[must_use]
    pub fn mean_false_negative_rate(&self, noise_na: f64) -> Option<f64> {
        mean_metric(self.runs.iter().map(|run| {
            find_condition(&run.trained, noise_na).and_then(DecisionMetrics::false_negative_rate)
        }))
    }
    #[must_use]
    pub fn mean_abstention_rate(&self, noise_na: f64) -> Option<f64> {
        mean_metric(self.runs.iter().map(|run| {
            find_condition(&run.trained, noise_na).and_then(DecisionMetrics::abstention_rate)
        }))
    }
}

fn find_condition(
    conditions: &[GeneralizationNoiseResult],
    noise_na: f64,
) -> Option<DecisionMetrics> {
    conditions
        .iter()
        .find(|condition| condition.noise_na == noise_na)
        .map(|condition| condition.metrics)
}

fn mean_metric(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let values = values.collect::<Option<Vec<_>>>()?;
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn run_task(
    seed: u64,
    task_variant: usize,
    training_epochs: usize,
    repetitions: usize,
) -> Result<GeneralizationRunResult, SimError> {
    let init_seed = seed ^ (task_variant as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let initial = make_network(init_seed)?;
    let initialization_fingerprint = initial
        .synapses()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, synapse| {
            (hash ^ synapse.weight_na().to_bits()).wrapping_mul(0x0000_0100_0000_01b3)
        });
    let mut trained = initial.clone();
    train_features(&mut trained, task_variant, training_epochs)?;

    let mut baseline = Vec::with_capacity(NOISE_LEVELS_NA.len());
    let mut trained_conditions = Vec::with_capacity(NOISE_LEVELS_NA.len());
    for (noise_index, noise_na) in NOISE_LEVELS_NA.iter().copied().enumerate() {
        let noise_seed = seed
            ^ (task_variant as u64).wrapping_mul(0xd134_2543_de82_ef95)
            ^ (noise_index as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
        baseline.push(evaluate_noise(
            &initial,
            task_variant,
            noise_na,
            repetitions,
            noise_seed,
        )?);
        trained_conditions.push(evaluate_noise(
            &trained,
            task_variant,
            noise_na,
            repetitions,
            noise_seed,
        )?);
    }
    Ok(GeneralizationRunResult {
        seed,
        task_variant,
        initialization_fingerprint,
        baseline,
        trained: trained_conditions,
    })
}

fn make_network(seed: u64) -> Result<LifNetwork, SimError> {
    let config = LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1);
    let mut network = LifNetwork::new(config, NETWORK_SIZE)?;
    let mut state = if seed == 0 {
        0xa076_1d64_78bd_642f
    } else {
        seed
    };
    for feature in 0..FEATURE_COUNT {
        for action in 0..ACTION_COUNT {
            let initial_weight = next_unit(&mut state) * 4.0;
            network.add_synapse(Synapse::new(
                feature,
                ACTION_OFFSET + action,
                initial_weight,
                0.0,
                120.0,
                1,
            )?)?;
        }
    }
    network.enable_reward_modulated_stdp(RewardModulatedStdpConfig::new(
        StdpConfig::new(1.0, 1.0, 10.0, 10.0),
        20.0,
        2.5,
    ))?;
    Ok(network)
}

fn train_features(
    network: &mut LifNetwork,
    task_variant: usize,
    epochs: usize,
) -> Result<(), SimError> {
    let mut evaluator = RewardEvaluator::new(RewardEvaluatorConfig::new(1.0, 1.0))?;
    for _ in 0..epochs {
        for feature in 0..FEATURE_COUNT {
            let action = action_for_feature(feature, task_variant);
            let mut cue = [0.0; NETWORK_SIZE];
            cue[feature] = TRAIN_AMPLITUDES_NA[task_variant];
            network.step(&cue)?;

            let mut teacher = [0.0; NETWORK_SIZE];
            teacher[ACTION_OFFSET + action] = 200.0;
            let response = network.step(&teacher)?;
            let observed = (0..ACTION_COUNT)
                .map(|index| {
                    response
                        .spiking_neurons()
                        .contains(&(ACTION_OFFSET + index))
                })
                .collect::<Vec<_>>();
            evaluator.evaluate_and_apply(Some(action), &observed, network)?;
            idle(network, TRAINING_WASHOUT_STEPS)?;
        }
    }
    Ok(())
}

fn evaluate_noise(
    network: &LifNetwork,
    task_variant: usize,
    noise_na: f64,
    repetitions: usize,
    mut noise_state: u64,
) -> Result<GeneralizationNoiseResult, SimError> {
    if noise_state == 0 {
        noise_state = 0xe703_7ed1_a0b4_28db;
    }
    let mut evaluator = DecisionEvaluator::new();
    for action in 0..ACTION_COUNT {
        let features = features_for_action(action, task_variant);
        for repetition in 0..repetitions {
            let amplitude = TRAIN_AMPLITUDES_NA[task_variant]
                + TEST_AMPLITUDE_OFFSETS_NA[repetition % TEST_AMPLITUDE_OFFSETS_NA.len()];
            let observed =
                present_pattern(network, &features, amplitude, noise_na, &mut noise_state)?;
            evaluator.record(
                Some(action_name(action)),
                observed,
                abstention_reason(observed),
            )?;
        }
    }
    for _ in 0..repetitions {
        let observed = present_pattern(network, &[], 0.0, noise_na, &mut noise_state)?;
        evaluator.record(None, observed, abstention_reason(observed))?;
    }
    Ok(GeneralizationNoiseResult {
        noise_na,
        metrics: evaluator.metrics(),
    })
}

fn present_pattern(
    network: &LifNetwork,
    active_features: &[usize],
    amplitude: f64,
    noise_na: f64,
    noise_state: &mut u64,
) -> Result<Option<&'static str>, SimError> {
    let mut input = [0.0; NETWORK_SIZE];
    for (feature, current) in input.iter_mut().take(FEATURE_COUNT).enumerate() {
        let base = if active_features.contains(&feature) {
            amplitude
        } else {
            0.0
        };
        *current = (base + signed_noise(noise_state, noise_na)).max(0.0);
    }
    let mut probe = network.clone();
    probe.step(&input)?;
    let response = probe.step(&[0.0; NETWORK_SIZE])?;
    let action_zero = response.spiking_neurons().contains(&ACTION_OFFSET);
    let action_one = response.spiking_neurons().contains(&(ACTION_OFFSET + 1));
    Ok(if action_zero {
        Some(action_name(0))
    } else if action_one {
        Some(action_name(1))
    } else {
        None
    })
}

fn action_for_feature(feature: usize, task_variant: usize) -> usize {
    let group_zero = if task_variant % 2 == 0 {
        feature < 2
    } else {
        feature % 2 == 0
    };
    let action_zero = if task_variant < 2 {
        group_zero
    } else {
        !group_zero
    };
    if action_zero { 0 } else { 1 }
}

fn features_for_action(action: usize, task_variant: usize) -> [usize; 2] {
    let mut matching =
        (0..FEATURE_COUNT).filter(|feature| action_for_feature(*feature, task_variant) == action);
    [
        matching.next().expect("setiap kelas punya fitur"),
        matching.next().expect("setiap kelas punya dua fitur"),
    ]
}

fn action_name(action: usize) -> &'static str {
    if action == 0 { "kelas_0" } else { "kelas_1" }
}

fn abstention_reason(predicted: Option<&str>) -> Option<AbstentionReason> {
    predicted
        .is_none()
        .then_some(AbstentionReason::NoActionSignal)
}

fn idle(network: &mut LifNetwork, steps: usize) -> Result<(), SimError> {
    for _ in 0..steps {
        network.step(&[0.0; NETWORK_SIZE])?;
    }
    Ok(())
}

fn signed_noise(state: &mut u64, amplitude: f64) -> f64 {
    (next_unit(state) * 2.0 - 1.0) * amplitude
}

fn next_unit(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / ((1_u64 << 53) as f64)
}
