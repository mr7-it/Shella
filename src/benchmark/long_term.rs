use super::continual_replay::{
    ACTION_OFFSET, Experience, NETWORK_SIZE, experience_currents, make_network,
    train_experience, training_examples,
};
use crate::{
    Checkpoint, DecisionEvaluator, DecisionMetrics, LearningMemory, LifNetwork,
    RewardEvaluator, RewardEvaluatorConfig, SimError,
};
use std::{
    collections::BTreeSet,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

const TASK_NAMES: [&str; 2] = ["task-a", "task-b"];
const TRAIN_EPOCHS: usize = 64;
const GAP_STEPS: usize = 20;
const TASK_B_EXTRA_PASSES: usize = 1;
const MEMORY_CURRENT_NA: f64 = 80.0;
const REPLAY_LIMIT: usize = 1;
const TEACHER_CURRENT_NA: f64 = 200.0;
const VALIDATION_OFFSETS_NA: [f64; 2] = [-5.0, 5.0];
const TEST_OFFSETS_NA: [f64; 2] = [-17.0, 17.0];
const VALIDATION_NOISE_NA: f64 = 10.0;
const TEST_NOISE_LEVELS_NA: [u64; 3] = [30, 50, 70];
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Benchmark berpasangan untuk memori persisten, replay, dan pemulihan state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongTermLearningBenchmarkConfig {
    seeds: Vec<u64>,
    task_variants: usize,
    repetitions_per_pattern: usize,
}

impl LongTermLearningBenchmarkConfig {
    pub fn new(seeds: Vec<u64>, task_variants: usize, repetitions_per_pattern: usize) -> Result<Self, SimError> {
        if seeds.is_empty() || task_variants == 0 || task_variants > 4 || repetitions_per_pattern == 0 {
            return Err(SimError::InvalidConfiguration {
                parameter: "long_term_learning_benchmark",
                reason: "seed, variasi, dan repetisi harus positif; maksimum empat variasi",
            });
        }
        if seeds.iter().copied().collect::<BTreeSet<_>>().len() != seeds.len() {
            return Err(SimError::InvalidConfiguration {
                parameter: "long_term_learning_seeds",
                reason: "seed harus unik",
            });
        }
        seeds.len().checked_mul(task_variants).and_then(|n| n.checked_mul(repetitions_per_pattern))
            .ok_or(SimError::CounterOverflow { counter: "ukuran benchmark memori jangka panjang" })?;
        Ok(Self { seeds, task_variants, repetitions_per_pattern })
    }

    /// Delapan inisialisasi independen × empat pemetaan asosiasi.
    #[must_use]
    pub fn standard() -> Self {
        Self { seeds: vec![23, 41, 67, 97, 139, 181, 251, 349], task_variants: 4, repetitions_per_pattern: 8 }
    }
    #[must_use]
    pub fn seeds(&self) -> &[u64] { &self.seeds }
    #[must_use]
    pub const fn task_variants(&self) -> usize { self.task_variants }
    #[must_use]
    pub const fn repetitions_per_pattern(&self) -> usize { self.repetitions_per_pattern }
    #[must_use]
    pub const fn test_noise_levels_na(&self) -> &'static [u64] { &TEST_NOISE_LEVELS_NA }

    pub fn run(&self) -> Result<LongTermLearningBenchmarkResult, SimError> {
        let count = self.seeds.len().checked_mul(self.task_variants)
            .ok_or(SimError::CounterOverflow { counter: "jumlah run M10" })?;
        let mut runs = Vec::new();
        runs.try_reserve_exact(count).map_err(|_| SimError::AllocationFailed)?;
        for &seed in &self.seeds {
            for variant in 0..self.task_variants {
                runs.push(run_pair(seed, variant, self.repetitions_per_pattern)?);
            }
        }
        Ok(LongTermLearningBenchmarkResult { runs })
    }
}

impl Default for LongTermLearningBenchmarkConfig {
    fn default() -> Self { Self::standard() }
}

/// Confusion metrics untuk dua task pada satu tahap evaluasi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LongTermLearningTaskMetrics {
    task_a: DecisionMetrics,
    task_b: DecisionMetrics,
}
impl LongTermLearningTaskMetrics {
    #[must_use]
    pub const fn task_a(self) -> DecisionMetrics { self.task_a }
    #[must_use]
    pub const fn task_b(self) -> DecisionMetrics { self.task_b }
}

/// Evaluasi sebelum belajar, sesudah Task A, baseline, dan metode memori.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LongTermLearningStage {
    before_learning: LongTermLearningTaskMetrics,
    after_task_a: LongTermLearningTaskMetrics,
    without_memory: LongTermLearningTaskMetrics,
    with_memory: LongTermLearningTaskMetrics,
}
impl LongTermLearningStage {
    #[must_use]
    pub const fn before_learning(self) -> LongTermLearningTaskMetrics { self.before_learning }
    #[must_use]
    pub const fn after_task_a(self) -> LongTermLearningTaskMetrics { self.after_task_a }
    #[must_use]
    pub const fn without_memory(self) -> LongTermLearningTaskMetrics { self.without_memory }
    #[must_use]
    pub const fn with_memory(self) -> LongTermLearningTaskMetrics { self.with_memory }
}

/// Metrik test pada satu tingkat noise holdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LongTermLearningNoiseResult {
    noise_na: u64,
    stage: LongTermLearningStage,
}
impl LongTermLearningNoiseResult {
    #[must_use]
    pub const fn noise_na(self) -> u64 { self.noise_na }
    #[must_use]
    pub const fn stage(self) -> LongTermLearningStage { self.stage }
}

/// Hasil individual dengan bukti checkpoint, persistence, konflik, feedback, dan budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongTermLearningRunResult {
    seed: u64,
    task_variant: usize,
    initialization_fingerprint: u64,
    checkpoint_restored_identically: bool,
    memory_records_after_restart: usize,
    memory_records_after_task_b: usize,
    feedback_errors: usize,
    detected_conflicts: usize,
    replayed_experiences: usize,
    replayed_error_experiences: usize,
    training_presentations_per_method: u64,
    training_steps_per_method: u64,
    memory_decay_deleted: usize,
    validation: LongTermLearningStage,
    test: Vec<LongTermLearningNoiseResult>,
}
impl LongTermLearningRunResult {
    #[must_use]
    pub const fn seed(&self) -> u64 { self.seed }
    #[must_use]
    pub const fn task_variant(&self) -> usize { self.task_variant }
    #[must_use]
    pub const fn initialization_fingerprint(&self) -> u64 { self.initialization_fingerprint }
    #[must_use]
    pub const fn checkpoint_restored_identically(&self) -> bool { self.checkpoint_restored_identically }
    #[must_use]
    pub const fn memory_records_after_restart(&self) -> usize { self.memory_records_after_restart }
    #[must_use]
    pub const fn memory_records_after_task_b(&self) -> usize { self.memory_records_after_task_b }
    #[must_use]
    pub const fn feedback_errors(&self) -> usize { self.feedback_errors }
    #[must_use]
    pub const fn detected_conflicts(&self) -> usize { self.detected_conflicts }
    #[must_use]
    pub const fn replayed_experiences(&self) -> usize { self.replayed_experiences }
    #[must_use]
    pub const fn replayed_error_experiences(&self) -> usize { self.replayed_error_experiences }
    #[must_use]
    pub const fn training_presentations_per_method(&self) -> u64 { self.training_presentations_per_method }
    #[must_use]
    pub const fn training_steps_per_method(&self) -> u64 { self.training_steps_per_method }
    #[must_use]
    pub const fn memory_decay_deleted(&self) -> usize { self.memory_decay_deleted }
    #[must_use]
    pub const fn validation(&self) -> LongTermLearningStage { self.validation }
    #[must_use]
    pub fn test(&self) -> &[LongTermLearningNoiseResult] { &self.test }
    #[must_use]
    pub fn forgetting_without_memory(&self, noise_na: u64) -> Option<f64> {
        let stage = self.test_at_noise(noise_na)?.stage;
        Some((stage.after_task_a.task_a.accuracy()? - stage.without_memory.task_a.accuracy()?).max(0.0))
    }
    #[must_use]
    pub fn forgetting_with_memory(&self, noise_na: u64) -> Option<f64> {
        let stage = self.test_at_noise(noise_na)?.stage;
        Some((stage.after_task_a.task_a.accuracy()? - stage.with_memory.task_a.accuracy()?).max(0.0))
    }
    fn test_at_noise(&self, noise_na: u64) -> Option<LongTermLearningNoiseResult> {
        self.test.iter().find(|result| result.noise_na == noise_na).copied()
    }
}

/// Agregat benchmark M10.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongTermLearningBenchmarkResult {
    runs: Vec<LongTermLearningRunResult>,
}
impl LongTermLearningBenchmarkResult {
    #[must_use]
    pub fn runs(&self) -> &[LongTermLearningRunResult] { &self.runs }
    #[must_use]
    pub fn mean_forgetting_without_memory(&self, noise_na: u64) -> Option<f64> {
        mean(self.runs.iter().map(|run| run.forgetting_without_memory(noise_na)))
    }
    #[must_use]
    pub fn mean_forgetting_with_memory(&self, noise_na: u64) -> Option<f64> {
        mean(self.runs.iter().map(|run| run.forgetting_with_memory(noise_na)))
    }
    #[must_use]
    pub fn catastrophic_rate_without_memory(&self, noise_na: u64) -> Option<f64> {
        mean(self.runs.iter().map(|run| run.forgetting_without_memory(noise_na).map(|v| f64::from(v >= 0.5))))
    }
    #[must_use]
    pub fn catastrophic_rate_with_memory(&self, noise_na: u64) -> Option<f64> {
        mean(self.runs.iter().map(|run| run.forgetting_with_memory(noise_na).map(|v| f64::from(v >= 0.5))))
    }
    #[must_use]
    pub fn forgetting_sd_without_memory(&self, noise_na: u64) -> Option<f64> {
        stddev(self.runs.iter().map(|run| run.forgetting_without_memory(noise_na)))
    }
    #[must_use]
    pub fn forgetting_sd_with_memory(&self, noise_na: u64) -> Option<f64> {
        stddev(self.runs.iter().map(|run| run.forgetting_with_memory(noise_na)))
    }
}

struct RunFiles { memory: PathBuf, checkpoint: PathBuf }
impl RunFiles {
    fn new(seed: u64, variant: usize) -> Self {
        let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir();
        Self {
            memory: root.join(format!("shella-m10-{}-{seed}-{variant}-{id}.json", std::process::id())),
            checkpoint: root.join(format!("shella-m10-{}-{seed}-{variant}-{id}.checkpoint", std::process::id())),
        }
    }
}
impl Drop for RunFiles {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.memory);
        let mut lock = self.memory.as_os_str().to_os_string();
        lock.push(".lock");
        let _ = fs::remove_file(PathBuf::from(lock));
        let _ = fs::remove_file(&self.checkpoint);
    }
}

fn run_pair(seed: u64, variant: usize, repetitions: usize) -> Result<LongTermLearningRunResult, SimError> {
    let files = RunFiles::new(seed, variant);
    let mut network = make_network(seed ^ (variant as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))?;
    let fingerprint = network.synapses().iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, synapse| {
        (hash ^ synapse.weight_na().to_bits()).wrapping_mul(0x0000_0100_0000_01b3)
    });
    let task_a = training_examples(0, variant);
    let task_b = training_examples(1, variant);
    let reward_config = RewardEvaluatorConfig::new(1.0, 1.0);
    let mut memory = LearningMemory::open(&files.memory)?;
    let mut feedback_errors = 0;
    let mut detected_conflicts = 0;
    let mut evaluator_a = RewardEvaluator::new(reward_config)?;
    for experience in &task_a {
        let input = experience_currents(experience);
        let predicted = predict_without_memory(&network, &input)?;
        let write = memory.record_feedback(TASK_NAMES[0], &input, experience.action, predicted)?;
        feedback_errors += usize::from(!write.prediction_correct());
    }
    for _ in 0..TRAIN_EPOCHS {
        for experience in &task_a {
            train_experience(&mut network, *experience, &mut evaluator_a)?;
        }
    }
    let entries_after_task_a = memory.len();
    let expected_state = network.clone();
    Checkpoint::save(&files.checkpoint, &network, None)?;
    drop(memory);
    drop(network);
    let restored = Checkpoint::restore(&files.checkpoint)?.network;
    let checkpoint_restored_identically = restored == expected_state;
    let mut memory = LearningMemory::open(&files.memory)?;
    if memory.len() != entries_after_task_a {
        return Err(SimError::CorruptData {
            subsystem: "benchmark memori jangka panjang",
            reason: "jumlah pengalaman berubah setelah restart".to_owned(),
        });
    }

    let initial = make_network(seed ^ (variant as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))?;
    let mut without_memory = restored.clone();
    let mut with_memory = restored.clone();
    let validation_seed = seed ^ 0x082e_fa98_ec4e_6c89;
    let before_learning = evaluate_pair(&initial, None, variant, VALIDATION_OFFSETS_NA, VALIDATION_NOISE_NA, repetitions, validation_seed)?;
    let after_task_a = evaluate_pair(&restored, Some(&memory), variant, VALIDATION_OFFSETS_NA, VALIDATION_NOISE_NA, repetitions, validation_seed)?;

    for experience in &task_b {
        let input = experience_currents(experience);
        let predicted = predict_without_memory(&with_memory, &input)?;
        let write = memory.record_feedback(TASK_NAMES[1], &input, experience.action, predicted)?;
        feedback_errors += usize::from(!write.prediction_correct());
        detected_conflicts += write.conflicts().len();
    }

    let mut evaluator_without = RewardEvaluator::new(reward_config)?;
    let mut evaluator_with = RewardEvaluator::new(reward_config)?;
    let mut replayed_experiences = 0;
    let mut replayed_error_experiences = 0;
    let mut phase_b_presentations = 0_usize;
    for _ in 0..TRAIN_EPOCHS {
        for experience in &task_b {
            let query = experience_currents(experience);
            let replay_available = !memory.recall_task_similar(TASK_NAMES[0], &query, REPLAY_LIMIT)?.is_empty();

            train_experience(&mut without_memory, *experience, &mut evaluator_without)?;
            phase_b_presentations += 1;
            for _ in 0..TASK_B_EXTRA_PASSES {
                train_experience(&mut without_memory, *experience, &mut evaluator_without)?;
                phase_b_presentations += 1;
            }

            train_experience(&mut with_memory, *experience, &mut evaluator_with)?;
            let report = memory.replay_task_relevant(
                TASK_NAMES[0],
                &mut with_memory,
                &query,
                &[ACTION_OFFSET, ACTION_OFFSET + 1],
                REPLAY_LIMIT,
                TEACHER_CURRENT_NA,
            )?;
            if report.presentations() > 1 {
                return Err(SimError::InvalidTopology {
                    reason: "jumlah replay melampaui slot budget paired",
                });
            }
            if replay_available && report.presentations() != 1 {
                return Err(SimError::InvalidTopology {
                    reason: "retrieval tersedia tetapi replay tidak dijalankan",
                });
            }
            if report.presentations() == 1 {
                idle(&mut with_memory, GAP_STEPS)?;
                replayed_experiences += report.presentations();
                replayed_error_experiences += report.error_records_replayed();
            } else {
                idle(&mut with_memory, 3 + GAP_STEPS)?;
            }
        }
    }

    let validation_without = evaluate_pair(&without_memory, None, variant, VALIDATION_OFFSETS_NA, VALIDATION_NOISE_NA, repetitions, validation_seed)?;
    let validation_with = evaluate_pair(&with_memory, Some(&memory), variant, VALIDATION_OFFSETS_NA, VALIDATION_NOISE_NA, repetitions, validation_seed)?;
    let validation = LongTermLearningStage { before_learning, after_task_a, without_memory: validation_without, with_memory: validation_with };

    let mut test = Vec::with_capacity(TEST_NOISE_LEVELS_NA.len());
    for (index, noise) in TEST_NOISE_LEVELS_NA.iter().copied().enumerate() {
        let test_seed = seed ^ (variant as u64).wrapping_mul(0xd134_2543_de82_ef95)
            ^ (index as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
        let stage = LongTermLearningStage {
            before_learning: evaluate_pair(&initial, None, variant, TEST_OFFSETS_NA, noise as f64, repetitions, test_seed)?,
            after_task_a: evaluate_pair(&restored, Some(&memory), variant, TEST_OFFSETS_NA, noise as f64, repetitions, test_seed)?,
            without_memory: evaluate_pair(&without_memory, None, variant, TEST_OFFSETS_NA, noise as f64, repetitions, test_seed)?,
            with_memory: evaluate_pair(&with_memory, Some(&memory), variant, TEST_OFFSETS_NA, noise as f64, repetitions, test_seed)?,
        };
        test.push(LongTermLearningNoiseResult { noise_na: noise, stage });
    }

    let task_a_presentations = task_a.len().checked_mul(TRAIN_EPOCHS)
        .ok_or(SimError::CounterOverflow { counter: "presentasi Task A M10" })?;
    let total_presentations = u64::try_from(task_a_presentations.checked_add(phase_b_presentations)
        .ok_or(SimError::CounterOverflow { counter: "presentasi M10" })?)
        .map_err(|_| SimError::CounterOverflow { counter: "presentasi M10" })?;
    let training_steps = total_presentations.checked_mul((3 + GAP_STEPS) as u64)
        .ok_or(SimError::CounterOverflow { counter: "langkah M10" })?;
    let decay = memory.decay_irrelevant(&experience_currents(&task_b[0]), 0.75, 0.5, 0.2, 2.0)?;
    Ok(LongTermLearningRunResult {
        seed,
        task_variant: variant,
        initialization_fingerprint: fingerprint,
        checkpoint_restored_identically,
        memory_records_after_restart: entries_after_task_a,
        memory_records_after_task_b: memory.len(),
        feedback_errors,
        detected_conflicts,
        replayed_experiences,
        replayed_error_experiences,
        training_presentations_per_method: total_presentations,
        training_steps_per_method: training_steps,
        memory_decay_deleted: decay.deleted(),
        validation,
        test,
    })
}

fn predict_without_memory(network: &LifNetwork, input: &[f64]) -> Result<Option<usize>, SimError> {
    let mut probe = network.clone();
    probe.step(input)?;
    probe.step(&vec![0.0; probe.neuron_count()])?;
    let response = probe.step(&vec![0.0; probe.neuron_count()])?;
    Ok((0..2).find(|action| response.spiking_neurons().contains(&(ACTION_OFFSET + action))))
}

fn evaluate_pair(
    network: &LifNetwork,
    memory: Option<&LearningMemory>,
    variant: usize,
    offsets: [f64; 2],
    noise_na: f64,
    repetitions: usize,
    seed: u64,
) -> Result<LongTermLearningTaskMetrics, SimError> {
    Ok(LongTermLearningTaskMetrics {
        task_a: evaluate_task(network, memory, 0, variant, offsets, noise_na, repetitions, seed)?,
        task_b: evaluate_task(network, memory, 1, variant, offsets, noise_na, repetitions, seed ^ 0xa409_3822_299f_31d0)?,
    })
}

fn evaluate_task(
    network: &LifNetwork,
    memory: Option<&LearningMemory>,
    task: usize,
    variant: usize,
    offsets: [f64; 2],
    noise_na: f64,
    repetitions: usize,
    mut noise_state: u64,
) -> Result<DecisionMetrics, SimError> {
    let mut evaluator = DecisionEvaluator::new();
    let action_neurons = [ACTION_OFFSET, ACTION_OFFSET + 1];
    let examples = training_examples(task, variant);
    for example in examples.iter().filter(|experience| experience.action.is_some()) {
        for repetition in 0..repetitions {
            let input = noisy_currents(example, example.amplitude_na + offsets[repetition % offsets.len()], noise_na, &mut noise_state);
            let predicted = infer_action(network, memory, task, &input, &action_neurons)?;
            evaluator.record(
                example.action.map(super::continual_replay::action_name),
                predicted.map(super::continual_replay::action_name),
                abstention(predicted),
            )?;
        }
    }
    for experience in examples.iter().filter(|experience| experience.action.is_none()) {
        for _ in 0..repetitions {
            let input = noisy_currents(experience, experience.amplitude_na, noise_na, &mut noise_state);
            let predicted = infer_action(network, memory, task, &input, &action_neurons)?;
            evaluator.record(None, predicted.map(super::continual_replay::action_name), abstention(predicted))?;
        }
    }
    for feature in 0..4 {
        for _ in 0..repetitions {
            let mut input = vec![0.0; NETWORK_SIZE];
            input[feature] = examples[0].amplitude_na;
            perturb(&mut input, noise_na, &mut noise_state);
            let predicted = infer_action(network, memory, task, &input, &action_neurons)?;
            evaluator.record(None, predicted.map(super::continual_replay::action_name), abstention(predicted))?;
        }
    }
    Ok(evaluator.metrics())
}

fn noisy_currents(example: &Experience, amplitude: f64, noise: f64, state: &mut u64) -> Vec<f64> {
    let mut input = experience_currents(example);
    for value in input.iter_mut().take(ACTION_OFFSET) { *value = (*value + signed_noise(state, noise)).max(0.0); }
    if let Some(context) = example.context { input[context] += amplitude - example.amplitude_na; }
    if example.action.is_some() { input[example.feature] += amplitude - example.amplitude_na; }
    input
}

fn infer_action(
    network: &LifNetwork,
    memory: Option<&LearningMemory>,
    task: usize,
    input: &[f64],
    action_neurons: &[usize],
) -> Result<Option<usize>, SimError> {
    if let Some(memory) = memory {
        let mut probe = network.clone();
        let spikes = memory
            .infer_task_with_memory(TASK_NAMES[task], &mut probe, input, action_neurons, 1, MEMORY_CURRENT_NA)?
            .spiking_neurons();
        return Ok((0..action_neurons.len()).find(|action| spikes.contains(&action_neurons[*action])));
    }
    predict_without_memory(network, input)
}

fn perturb(input: &mut [f64], noise: f64, state: &mut u64) {
    for current in input.iter_mut().take(ACTION_OFFSET) { *current = (*current + signed_noise(state, noise)).max(0.0); }
}

fn abstention(predicted: Option<usize>) -> Option<crate::decision::AbstentionReason> {
    predicted.is_none().then_some(crate::decision::AbstentionReason::NoActionSignal)
}

fn idle(network: &mut LifNetwork, steps: usize) -> Result<(), SimError> {
    for _ in 0..steps { network.step(&vec![0.0; network.neuron_count()])?; }
    Ok(())
}

fn signed_noise(state: &mut u64, amplitude: f64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f64 / ((1_u64 << 53) as f64) * 2.0 - 1.0) * amplitude
}

fn mean(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let values = values.collect::<Option<Vec<_>>>()?;
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn stddev(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let values = values.collect::<Option<Vec<_>>>()?;
    if values.is_empty() { return None; }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    Some((values.iter().map(|value| (value - average).powi(2)).sum::<f64>() / values.len() as f64).sqrt())
}