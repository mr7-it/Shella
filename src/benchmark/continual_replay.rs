use crate::{
    DecisionEvaluator, DecisionMetrics, LifConfig, LifNetwork, RewardEvaluator,
    RewardEvaluatorConfig, RewardModulatedStdpConfig, SimError, StdpConfig, Synapse,
};
use std::collections::BTreeSet;

const FEATURE_COUNT: usize = 4;
const CONTEXT_OFFSET: usize = FEATURE_COUNT;
const HIDDEN_OFFSET: usize = FEATURE_COUNT + 2;
const HIDDEN_COUNT: usize = FEATURE_COUNT + TASK_COUNT;
pub(super) const ACTION_OFFSET: usize = HIDDEN_OFFSET + HIDDEN_COUNT;
const ACTION_COUNT: usize = 2;
pub(super) const NETWORK_SIZE: usize = ACTION_OFFSET + ACTION_COUNT;
const TASK_COUNT: usize = 2;
const TRAIN_AMPLITUDES_NA: [f64; 4] = [90.0, 100.0, 110.0, 120.0];
const VALIDATION_OFFSETS_NA: [f64; 2] = [-5.0, 5.0];
const TEST_OFFSETS_NA: [f64; 2] = [-17.0, 17.0];
const TEST_NOISE_LEVELS_NA: [f64; 3] = [30.0, 50.0, 70.0];
const VALIDATION_NOISE_NA: f64 = 10.0;
const TRAINING_GAP_STEPS: usize = 20;
const PHASE_B_PRESENTATION_SETS: usize = 3;
const REPLAY_PRESENTATION_SETS: usize = 1;
const CATASTROPHIC_FORGETTING_THRESHOLD: f64 = 0.5;
const TASK_A_LABELS: [[usize; FEATURE_COUNT]; 4] =
    [[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]];

/// Protokol dua tugas berkonflik untuk membandingkan replay dengan anggaran sama.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinualReplayBenchmarkConfig {
    seeds: Vec<u64>,
    task_variants: usize,
    epochs_per_task: usize,
    repetitions_per_pattern: usize,
}

impl ContinualReplayBenchmarkConfig {
    pub fn new(
        seeds: Vec<u64>,
        task_variants: usize,
        epochs_per_task: usize,
        repetitions_per_pattern: usize,
    ) -> Result<Self, SimError> {
        if seeds.is_empty()
            || task_variants == 0
            || task_variants > 4
            || epochs_per_task == 0
            || repetitions_per_pattern == 0
        {
            return Err(SimError::InvalidConfiguration {
                parameter: "continual_replay_benchmark",
                reason: "seed, variasi, epoch, dan repetisi harus positif; maksimum empat variasi",
            });
        }
        if seeds.iter().copied().collect::<BTreeSet<_>>().len() != seeds.len() {
            return Err(SimError::InvalidConfiguration {
                parameter: "continual_replay_seeds",
                reason: "seed harus unik",
            });
        }
        seeds
            .len()
            .checked_mul(task_variants)
            .and_then(|count| count.checked_mul(epochs_per_task))
            .and_then(|count| count.checked_mul(repetitions_per_pattern))
            .ok_or(SimError::CounterOverflow {
                counter: "ukuran continual replay benchmark",
            })?;
        Ok(Self {
            seeds,
            task_variants,
            epochs_per_task,
            repetitions_per_pattern,
        })
    }

    /// Delapan inisialisasi independen × empat variasi asosiasi.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            seeds: vec![23, 41, 67, 97, 139, 181, 251, 349],
            task_variants: 4,
            epochs_per_task: 64,
            repetitions_per_pattern: 8,
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
    pub const fn epochs_per_task(&self) -> usize {
        self.epochs_per_task
    }
    #[must_use]
    pub const fn repetitions_per_pattern(&self) -> usize {
        self.repetitions_per_pattern
    }
    #[must_use]
    pub const fn test_noise_levels_na(&self) -> &'static [f64] {
        &TEST_NOISE_LEVELS_NA
    }

    pub fn run(&self) -> Result<ContinualReplayBenchmarkResult, SimError> {
        let count =
            self.seeds
                .len()
                .checked_mul(self.task_variants)
                .ok_or(SimError::CounterOverflow {
                    counter: "jumlah run continual replay benchmark",
                })?;
        let mut runs = Vec::new();
        runs.try_reserve_exact(count)
            .map_err(|_| SimError::AllocationFailed)?;
        for &seed in &self.seeds {
            for variant in 0..self.task_variants {
                runs.push(run_pair(
                    seed,
                    variant,
                    self.epochs_per_task,
                    self.repetitions_per_pattern,
                )?);
            }
        }
        Ok(ContinualReplayBenchmarkResult { runs })
    }
}

impl Default for ContinualReplayBenchmarkConfig {
    fn default() -> Self {
        Self::standard()
    }
}

/// Metrik dua tugas pada satu tahap evaluasi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContinualReplayTaskMetrics {
    task_a: DecisionMetrics,
    task_b: DecisionMetrics,
}

impl ContinualReplayTaskMetrics {
    #[must_use]
    pub const fn task_a(self) -> DecisionMetrics {
        self.task_a
    }
    #[must_use]
    pub const fn task_b(self) -> DecisionMetrics {
        self.task_b
    }
}

/// Snapshot validasi yang tidak digunakan untuk pemilihan parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContinualReplayValidation {
    after_task_a: ContinualReplayTaskMetrics,
    without_replay: ContinualReplayTaskMetrics,
    with_replay: ContinualReplayTaskMetrics,
}

impl ContinualReplayValidation {
    #[must_use]
    pub const fn after_task_a(self) -> ContinualReplayTaskMetrics {
        self.after_task_a
    }
    #[must_use]
    pub const fn without_replay(self) -> ContinualReplayTaskMetrics {
        self.without_replay
    }
    #[must_use]
    pub const fn with_replay(self) -> ContinualReplayTaskMetrics {
        self.with_replay
    }
}

/// Metrik test pada satu tingkat noise, untuk tiap metode dan tahap belajar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContinualReplayNoiseResult {
    noise_na: u64,
    before_learning: ContinualReplayTaskMetrics,
    after_task_a: ContinualReplayTaskMetrics,
    without_replay: ContinualReplayTaskMetrics,
    with_replay: ContinualReplayTaskMetrics,
}

impl ContinualReplayNoiseResult {
    #[must_use]
    pub const fn noise_na(self) -> u64 {
        self.noise_na
    }
    #[must_use]
    pub const fn before_learning(self) -> ContinualReplayTaskMetrics {
        self.before_learning
    }
    #[must_use]
    pub const fn after_task_a(self) -> ContinualReplayTaskMetrics {
        self.after_task_a
    }
    #[must_use]
    pub const fn without_replay(self) -> ContinualReplayTaskMetrics {
        self.without_replay
    }
    #[must_use]
    pub const fn with_replay(self) -> ContinualReplayTaskMetrics {
        self.with_replay
    }
}

/// Hasil per inisialisasi/tugas, termasuk anggaran dan split validasi/test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinualReplayRunResult {
    seed: u64,
    task_variant: usize,
    initialization_fingerprint: u64,
    training_presentations_per_method: u64,
    training_steps_per_method: u64,
    validation: ContinualReplayValidation,
    test: Vec<ContinualReplayNoiseResult>,
}

impl ContinualReplayRunResult {
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
    pub const fn training_presentations_per_method(&self) -> u64 {
        self.training_presentations_per_method
    }
    #[must_use]
    pub const fn training_steps_per_method(&self) -> u64 {
        self.training_steps_per_method
    }
    #[must_use]
    pub const fn validation(&self) -> ContinualReplayValidation {
        self.validation
    }
    #[must_use]
    pub fn test(&self) -> &[ContinualReplayNoiseResult] {
        &self.test
    }
    #[must_use]
    pub fn forgetting_without_replay(&self, noise_na: u64) -> Option<f64> {
        let test = self.test_at_noise(noise_na)?;
        Some((accuracy(test.after_task_a.task_a)? - accuracy(test.without_replay.task_a)?).max(0.0))
    }
    #[must_use]
    pub fn forgetting_with_replay(&self, noise_na: u64) -> Option<f64> {
        let test = self.test_at_noise(noise_na)?;
        Some((accuracy(test.after_task_a.task_a)? - accuracy(test.with_replay.task_a)?).max(0.0))
    }
    fn test_at_noise(&self, noise_na: u64) -> Option<ContinualReplayNoiseResult> {
        self.test
            .iter()
            .find(|result| result.noise_na == noise_na)
            .copied()
    }
}

/// Hasil seluruh run dengan agregat forgetting dan variasi antar-run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinualReplayBenchmarkResult {
    runs: Vec<ContinualReplayRunResult>,
}

impl ContinualReplayBenchmarkResult {
    #[must_use]
    pub fn runs(&self) -> &[ContinualReplayRunResult] {
        &self.runs
    }
    #[must_use]
    pub fn mean_forgetting_without_replay(&self, noise_na: u64) -> Option<f64> {
        mean(
            self.runs
                .iter()
                .map(|run| run.forgetting_without_replay(noise_na)),
        )
    }
    #[must_use]
    pub fn mean_forgetting_with_replay(&self, noise_na: u64) -> Option<f64> {
        mean(
            self.runs
                .iter()
                .map(|run| run.forgetting_with_replay(noise_na)),
        )
    }
    #[must_use]
    pub fn catastrophic_forgetting_rate_without_replay(&self, noise_na: u64) -> Option<f64> {
        mean(self.runs.iter().map(|run| {
            run.forgetting_without_replay(noise_na)
                .map(|forgetting| f64::from(forgetting >= CATASTROPHIC_FORGETTING_THRESHOLD))
        }))
    }
    #[must_use]
    pub fn catastrophic_forgetting_rate_with_replay(&self, noise_na: u64) -> Option<f64> {
        mean(self.runs.iter().map(|run| {
            run.forgetting_with_replay(noise_na)
                .map(|forgetting| f64::from(forgetting >= CATASTROPHIC_FORGETTING_THRESHOLD))
        }))
    }
    #[must_use]
    pub fn forgetting_stddev_without_replay(&self, noise_na: u64) -> Option<f64> {
        stddev(
            self.runs
                .iter()
                .map(|run| run.forgetting_without_replay(noise_na)),
        )
    }
    #[must_use]
    pub fn forgetting_stddev_with_replay(&self, noise_na: u64) -> Option<f64> {
        stddev(
            self.runs
                .iter()
                .map(|run| run.forgetting_with_replay(noise_na)),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Experience {
    pub(super) context: Option<usize>,
    pub(super) feature: usize,
    pub(super) action: Option<usize>,
    pub(super) amplitude_na: f64,
}

#[derive(Debug, Clone)]
struct ReplayMemory {
    experiences: Vec<Experience>,
}

impl ReplayMemory {
    fn from_training_data(experiences: &[Experience]) -> Self {
        Self {
            experiences: experiences.to_vec(),
        }
    }

    fn samples(&self) -> &[Experience] {
        &self.experiences
    }
}

fn run_pair(
    seed: u64,
    task_variant: usize,
    epochs: usize,
    repetitions: usize,
) -> Result<ContinualReplayRunResult, SimError> {
    let initial = make_network(seed ^ (task_variant as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))?;
    let fingerprint = initial
        .synapses()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, synapse| {
            (hash ^ synapse.weight_na().to_bits()).wrapping_mul(0x0000_0100_0000_01b3)
        });
    let old_examples = training_examples(0, task_variant);
    let new_examples = training_examples(1, task_variant);
    let replay_memory = ReplayMemory::from_training_data(&old_examples);
    let evaluator_config = RewardEvaluatorConfig::new(1.0, 1.0);

    let mut after_task_a = initial.clone();
    train_phase_a(&mut after_task_a, &old_examples, epochs, evaluator_config)?;
    let before_task_b_test = evaluate_test_splits(
        &initial,
        task_variant,
        repetitions,
        seed ^ 0x243f_6a88_85a3_08d3,
    )?;
    let after_task_a_test = evaluate_test_splits(
        &after_task_a,
        task_variant,
        repetitions,
        seed ^ 0x243f_6a88_85a3_08d3,
    )?;
    let after_task_a_validation = evaluate_validation(
        &after_task_a,
        task_variant,
        repetitions,
        seed ^ 0x082e_fa98_ec4e_6c89,
    )?;

    let mut without_replay = after_task_a.clone();
    let mut with_replay = after_task_a;
    train_phase_b_without_replay(&mut without_replay, &new_examples, epochs, evaluator_config)?;
    train_phase_b_with_replay(
        &mut with_replay,
        &new_examples,
        replay_memory.samples(),
        epochs,
        evaluator_config,
    )?;

    let validation_without_replay = evaluate_validation(
        &without_replay,
        task_variant,
        repetitions,
        seed ^ 0x082e_fa98_ec4e_6c89,
    )?;
    let validation_with_replay = evaluate_validation(
        &with_replay,
        task_variant,
        repetitions,
        seed ^ 0x082e_fa98_ec4e_6c89,
    )?;
    let validation = ContinualReplayValidation {
        after_task_a: after_task_a_validation,
        without_replay: validation_without_replay,
        with_replay: validation_with_replay,
    };

    let mut test = Vec::with_capacity(TEST_NOISE_LEVELS_NA.len());
    for (index, noise_na) in TEST_NOISE_LEVELS_NA.iter().copied().enumerate() {
        let split_seed = seed
            ^ (task_variant as u64).wrapping_mul(0xd134_2543_de82_ef95)
            ^ (index as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
        let before_learning = before_task_b_test[index];
        let after_phase_a = after_task_a_test[index];
        let without = evaluate_noise_pair(
            &without_replay,
            task_variant,
            noise_na,
            repetitions,
            split_seed,
        )?;
        let with = evaluate_noise_pair(
            &with_replay,
            task_variant,
            noise_na,
            repetitions,
            split_seed,
        )?;
        test.push(ContinualReplayNoiseResult {
            noise_na: noise_na as u64,
            before_learning,
            after_task_a: after_phase_a,
            without_replay: without,
            with_replay: with,
        });
    }

    let presentations_per_epoch = old_examples
        .len()
        .checked_add(
            new_examples
                .len()
                .checked_mul(PHASE_B_PRESENTATION_SETS)
                .ok_or(SimError::CounterOverflow {
                    counter: "presentasi task B continual replay",
                })?,
        )
        .ok_or(SimError::CounterOverflow {
            counter: "presentasi continual replay per epoch",
        })?;
    let total_presentations = u64::try_from(epochs)
        .ok()
        .and_then(|epochs| {
            u64::try_from(presentations_per_epoch)
                .ok()?
                .checked_mul(epochs)
        })
        .ok_or(SimError::CounterOverflow {
            counter: "presentasi training continual replay",
        })?;
    let training_steps = total_presentations
        .checked_mul((3 + TRAINING_GAP_STEPS) as u64)
        .ok_or(SimError::CounterOverflow {
            counter: "langkah training continual replay",
        })?;
    Ok(ContinualReplayRunResult {
        seed,
        task_variant,
        initialization_fingerprint: fingerprint,
        training_presentations_per_method: total_presentations,
        training_steps_per_method: training_steps,
        validation,
        test,
    })
}

pub(super) fn training_examples(task: usize, variant: usize) -> Vec<Experience> {
    let mut examples = (0..FEATURE_COUNT)
        .map(|feature| Experience {
            context: Some(CONTEXT_OFFSET + task),
            feature,
            action: Some(action_for(task, feature, variant)),
            amplitude_na: TRAIN_AMPLITUDES_NA[variant],
        })
        .collect::<Vec<_>>();
    examples.push(Experience {
        context: Some(CONTEXT_OFFSET + task),
        feature: 0,
        action: None,
        amplitude_na: TRAIN_AMPLITUDES_NA[variant],
    });
    examples
}

pub(super) fn experience_currents(example: &Experience) -> Vec<f64> {
    let mut currents = vec![0.0; NETWORK_SIZE];
    if let Some(context) = example.context {
        currents[context] = example.amplitude_na;
    }
    if example.action.is_some() {
        currents[example.feature] = example.amplitude_na;
    }
    currents
}

pub(super) fn action_for(task: usize, feature: usize, variant: usize) -> usize {
    let task_a_action = TASK_A_LABELS[variant][feature];
    if task == 0 {
        task_a_action
    } else {
        1 - task_a_action
    }
}

pub(super) fn make_network(seed: u64) -> Result<LifNetwork, SimError> {
    let config = LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1);
    let mut network = LifNetwork::new(config, NETWORK_SIZE)?;
    let mut state = if seed == 0 {
        0xa076_1d64_78bd_642f
    } else {
        seed
    };
    for feature in 0..FEATURE_COUNT {
        let shared = shared_hidden_index(feature);
        network.add_synapse(Synapse::new(feature, shared, 30.0, 30.0, 30.0, 1)?)?;
        for task in 0..TASK_COUNT {
            network.add_synapse(Synapse::new(
                CONTEXT_OFFSET + task,
                shared,
                30.0,
                30.0,
                30.0,
                1,
            )?)?;
        }
    }
    for task in 0..TASK_COUNT {
        network.add_synapse(Synapse::new(
            CONTEXT_OFFSET + task,
            context_hidden_index(task),
            60.0,
            60.0,
            60.0,
            1,
        )?)?;
    }
    for source in HIDDEN_OFFSET..ACTION_OFFSET {
        for action in 0..ACTION_COUNT {
            let weight = next_unit(&mut state) * 8.0 - 4.0;
            network.add_synapse(Synapse::new(
                source,
                ACTION_OFFSET + action,
                weight,
                -120.0,
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

fn train_phase_a(
    network: &mut LifNetwork,
    examples: &[Experience],
    epochs: usize,
    evaluator_config: RewardEvaluatorConfig,
) -> Result<(), SimError> {
    let mut evaluator = RewardEvaluator::new(evaluator_config)?;
    for _ in 0..epochs {
        for example in examples {
            train_experience(network, *example, &mut evaluator)?;
        }
    }
    Ok(())
}

fn train_phase_b_without_replay(
    network: &mut LifNetwork,
    new_examples: &[Experience],
    epochs: usize,
    evaluator_config: RewardEvaluatorConfig,
) -> Result<(), SimError> {
    let mut evaluator = RewardEvaluator::new(evaluator_config)?;
    for _ in 0..epochs {
        for _ in 0..PHASE_B_PRESENTATION_SETS {
            for example in new_examples {
                train_experience(network, *example, &mut evaluator)?;
            }
        }
    }
    Ok(())
}

fn train_phase_b_with_replay(
    network: &mut LifNetwork,
    new_examples: &[Experience],
    replay_examples: &[Experience],
    epochs: usize,
    evaluator_config: RewardEvaluatorConfig,
) -> Result<(), SimError> {
    let mut evaluator = RewardEvaluator::new(evaluator_config)?;
    for _ in 0..epochs {
        for _ in 0..(PHASE_B_PRESENTATION_SETS - REPLAY_PRESENTATION_SETS) {
            for example in new_examples {
                train_experience(network, *example, &mut evaluator)?;
            }
        }
        for _ in 0..REPLAY_PRESENTATION_SETS {
            for example in replay_examples {
                train_experience(network, *example, &mut evaluator)?;
            }
        }
    }
    Ok(())
}

pub(super) fn train_experience(
    network: &mut LifNetwork,
    example: Experience,
    evaluator: &mut RewardEvaluator,
) -> Result<(), SimError> {
    let cue = experience_currents(&example);
    network.step(&cue)?;
    network.step(&[0.0; NETWORK_SIZE])?;

    let mut teacher = [0.0; NETWORK_SIZE];
    if let Some(action) = example.action {
        teacher[ACTION_OFFSET + action] = 200.0;
    }
    let response = network.step(&teacher)?;
    let observed = (0..ACTION_COUNT)
        .map(|action| {
            response
                .spiking_neurons()
                .contains(&(ACTION_OFFSET + action))
        })
        .collect::<Vec<_>>();
    evaluator.evaluate_and_apply(example.action, &observed, network)?;
    idle(network, TRAINING_GAP_STEPS)
}

fn evaluate_validation(
    network: &LifNetwork,
    variant: usize,
    repetitions: usize,
    seed: u64,
) -> Result<ContinualReplayTaskMetrics, SimError> {
    evaluate_both_tasks(
        network,
        variant,
        VALIDATION_OFFSETS_NA,
        VALIDATION_NOISE_NA,
        repetitions,
        seed,
    )
}

fn evaluate_test_splits(
    network: &LifNetwork,
    variant: usize,
    repetitions: usize,
    seed: u64,
) -> Result<Vec<ContinualReplayTaskMetrics>, SimError> {
    let mut results = Vec::with_capacity(TEST_NOISE_LEVELS_NA.len());
    for (index, noise_na) in TEST_NOISE_LEVELS_NA.iter().copied().enumerate() {
        results.push(evaluate_both_tasks(
            network,
            variant,
            TEST_OFFSETS_NA,
            noise_na,
            repetitions,
            seed ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        )?);
    }
    Ok(results)
}

fn evaluate_noise_pair(
    network: &LifNetwork,
    variant: usize,
    noise_na: f64,
    repetitions: usize,
    seed: u64,
) -> Result<ContinualReplayTaskMetrics, SimError> {
    evaluate_both_tasks(
        network,
        variant,
        TEST_OFFSETS_NA,
        noise_na,
        repetitions,
        seed,
    )
}

fn evaluate_both_tasks(
    network: &LifNetwork,
    variant: usize,
    offsets: [f64; 2],
    noise_na: f64,
    repetitions: usize,
    seed: u64,
) -> Result<ContinualReplayTaskMetrics, SimError> {
    Ok(ContinualReplayTaskMetrics {
        task_a: evaluate_task(network, 0, variant, offsets, noise_na, repetitions, seed)?,
        task_b: evaluate_task(
            network,
            1,
            variant,
            offsets,
            noise_na,
            repetitions,
            seed ^ 0xa409_3822_299f_31d0,
        )?,
    })
}

fn evaluate_task(
    network: &LifNetwork,
    task: usize,
    variant: usize,
    offsets: [f64; 2],
    noise_na: f64,
    repetitions: usize,
    mut noise_state: u64,
) -> Result<DecisionMetrics, SimError> {
    if noise_state == 0 {
        noise_state = 0xe703_7ed1_a0b4_28db;
    }
    let mut evaluator = DecisionEvaluator::new();
    for feature in 0..FEATURE_COUNT {
        let action = action_for(task, feature, variant);
        for repetition in 0..repetitions {
            let amplitude = TRAIN_AMPLITUDES_NA[variant] + offsets[repetition % offsets.len()];
            let predicted = present(
                network,
                Some(task),
                Some(feature),
                amplitude,
                noise_na,
                &mut noise_state,
            )?;
            evaluator.record(Some(action_name(action)), predicted, abstention(predicted))?;
        }
    }
    for _ in 0..repetitions {
        let predicted = present(
            network,
            Some(task),
            None,
            TRAIN_AMPLITUDES_NA[variant],
            noise_na,
            &mut noise_state,
        )?;
        evaluator.record(None, predicted, abstention(predicted))?;
    }
    for feature in 0..FEATURE_COUNT {
        for _ in 0..repetitions {
            let predicted = present(
                network,
                None,
                Some(feature),
                TRAIN_AMPLITUDES_NA[variant],
                noise_na,
                &mut noise_state,
            )?;
            evaluator.record(None, predicted, abstention(predicted))?;
        }
    }
    Ok(evaluator.metrics())
}

fn present(
    network: &LifNetwork,
    context: Option<usize>,
    feature: Option<usize>,
    amplitude: f64,
    noise_na: f64,
    noise_state: &mut u64,
) -> Result<Option<&'static str>, SimError> {
    let mut input = [0.0; NETWORK_SIZE];
    if let Some(task) = context {
        input[CONTEXT_OFFSET + task] = amplitude;
    }
    if let Some(feature) = feature {
        input[feature] = amplitude;
    }
    for current in input.iter_mut().take(ACTION_OFFSET) {
        *current = (*current + signed_noise(noise_state, noise_na)).max(0.0);
    }
    let mut probe = network.clone();
    probe.step(&input)?;
    probe.step(&[0.0; NETWORK_SIZE])?;
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

fn shared_hidden_index(feature: usize) -> usize {
    HIDDEN_OFFSET + feature
}

fn context_hidden_index(task: usize) -> usize {
    HIDDEN_OFFSET + FEATURE_COUNT + task
}

pub(super) fn action_name(action: usize) -> &'static str {
    if action == 0 { "kelas_0" } else { "kelas_1" }
}

fn abstention(predicted: Option<&str>) -> Option<crate::decision::AbstentionReason> {
    predicted
        .is_none()
        .then_some(crate::decision::AbstentionReason::NoActionSignal)
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

fn accuracy(metrics: DecisionMetrics) -> Option<f64> {
    metrics.accuracy()
}

fn mean(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let values = values.collect::<Option<Vec<_>>>()?;
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn stddev(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let values = values.collect::<Option<Vec<_>>>()?;
    if values.is_empty() {
        return None;
    }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    Some(
        (values
            .iter()
            .map(|value| (value - average).powi(2))
            .sum::<f64>()
            / values.len() as f64)
            .sqrt(),
    )
}
