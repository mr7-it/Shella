use crate::{
    LifConfig, LifNetwork, RewardEvaluator, RewardEvaluatorConfig, RewardModulatedStdpConfig,
    SimError, StdpConfig, Synapse,
};

const INPUT_COUNT: usize = 4;
const ACTION_COUNT: usize = 2;
const NETWORK_SIZE: usize = INPUT_COUNT + ACTION_COUNT;
const TEST_AMPLITUDE_OFFSETS_NA: [f64; 2] = [-5.0, 5.0];
const REPLAY_SLOT_STEPS: usize = 2;
const SHARED_CUE_MAX_WEIGHT_NA: f64 = 35.0;

/// Protokol paired untuk interferensi tugas dan replay konsolidasi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterferenceBenchmarkConfig {
    seeds: Vec<u64>,
    task_variants: usize,
    epochs_per_phase: usize,
}

impl InterferenceBenchmarkConfig {
    pub fn new(
        seeds: Vec<u64>,
        task_variants: usize,
        epochs_per_phase: usize,
    ) -> Result<Self, SimError> {
        if seeds.is_empty() || task_variants == 0 || epochs_per_phase == 0 {
            return Err(SimError::InvalidConfiguration {
                parameter: "interference_benchmark",
                reason: "seed, variasi tugas, dan epoch harus positif",
            });
        }
        if seeds
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != seeds.len()
        {
            return Err(SimError::InvalidConfiguration {
                parameter: "interference_seeds",
                reason: "seed harus unik",
            });
        }
        seeds
            .len()
            .checked_mul(task_variants)
            .and_then(|count| count.checked_mul(epochs_per_phase))
            .ok_or(SimError::CounterOverflow {
                counter: "ukuran interference benchmark",
            })?;
        Ok(Self {
            seeds,
            task_variants,
            epochs_per_phase,
        })
    }

    /// Delapan inisialisasi independen dan empat variasi tugas yang dipasangkan.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            seeds: vec![11, 29, 47, 83, 131, 197, 263, 337],
            task_variants: 4,
            epochs_per_phase: 32,
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
    pub const fn epochs_per_phase(&self) -> usize {
        self.epochs_per_phase
    }

    pub fn run(&self) -> Result<InterferenceBenchmarkResult, SimError> {
        let mut runs = Vec::with_capacity(self.seeds.len() * self.task_variants);
        for &seed in &self.seeds {
            for variant in 0..self.task_variants {
                runs.push(run_pair(seed, variant, self.epochs_per_phase)?);
            }
        }
        Ok(InterferenceBenchmarkResult { runs })
    }
}

impl Default for InterferenceBenchmarkConfig {
    fn default() -> Self {
        Self::standard()
    }
}

/// Hasil paired untuk satu seed dan satu variasi tugas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterferenceSeedResult {
    seed: u64,
    task_variant: usize,
    initialization_fingerprint: u64,
    old_task_after_phase_one: f64,
    old_task_without_replay: f64,
    new_task_without_replay: f64,
    old_task_with_replay: f64,
    new_task_with_replay: f64,
}

impl InterferenceSeedResult {
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }
    #[must_use]
    pub const fn task_variant(self) -> usize {
        self.task_variant
    }
    #[must_use]
    pub const fn initialization_fingerprint(self) -> u64 {
        self.initialization_fingerprint
    }
    #[must_use]
    pub const fn old_task_after_phase_one(self) -> f64 {
        self.old_task_after_phase_one
    }
    #[must_use]
    pub const fn old_task_without_replay(self) -> f64 {
        self.old_task_without_replay
    }
    #[must_use]
    pub const fn new_task_without_replay(self) -> f64 {
        self.new_task_without_replay
    }
    #[must_use]
    pub const fn old_task_with_replay(self) -> f64 {
        self.old_task_with_replay
    }
    #[must_use]
    pub const fn new_task_with_replay(self) -> f64 {
        self.new_task_with_replay
    }
    #[must_use]
    pub fn forgetting_without_replay(self) -> f64 {
        (self.old_task_after_phase_one - self.old_task_without_replay).max(0.0)
    }
    #[must_use]
    pub fn forgetting_with_replay(self) -> f64 {
        (self.old_task_after_phase_one - self.old_task_with_replay).max(0.0)
    }
}

/// Hasil seluruh run beserta agregat untuk membandingkan replay dan kontrol.
#[derive(Debug, Clone, PartialEq)]
pub struct InterferenceBenchmarkResult {
    runs: Vec<InterferenceSeedResult>,
}

impl InterferenceBenchmarkResult {
    #[must_use]
    pub fn runs(&self) -> &[InterferenceSeedResult] {
        &self.runs
    }
    #[must_use]
    pub fn mean_forgetting_without_replay(&self) -> f64 {
        mean(self.runs.iter().map(|run| run.forgetting_without_replay()))
    }
    #[must_use]
    pub fn mean_forgetting_with_replay(&self) -> f64 {
        mean(self.runs.iter().map(|run| run.forgetting_with_replay()))
    }
    #[must_use]
    pub fn mean_old_task_without_replay(&self) -> f64 {
        mean(self.runs.iter().map(|run| run.old_task_without_replay))
    }
    #[must_use]
    pub fn mean_old_task_with_replay(&self) -> f64 {
        mean(self.runs.iter().map(|run| run.old_task_with_replay))
    }
    #[must_use]
    pub fn mean_new_task_without_replay(&self) -> f64 {
        mean(self.runs.iter().map(|run| run.new_task_without_replay))
    }
    #[must_use]
    pub fn mean_new_task_with_replay(&self) -> f64 {
        mean(self.runs.iter().map(|run| run.new_task_with_replay))
    }
}

fn run_pair(seed: u64, variant: usize, epochs: usize) -> Result<InterferenceSeedResult, SimError> {
    let old_context = 1 + variant % (INPUT_COUNT - 1);
    let new_context = if variant % 2 == 0 {
        old_context
    } else {
        1 + (variant + 1) % (INPUT_COUNT - 1)
    };
    let old_action = variant % ACTION_COUNT;
    let new_action = 1 - old_action;
    let train_amplitude = [90.0, 100.0, 110.0, 120.0][variant % 4];
    let test_amplitudes = TEST_AMPLITUDE_OFFSETS_NA.map(|offset| train_amplitude + offset);

    let initial = make_network(seed ^ (variant as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))?;
    let initialization_fingerprint = initial
        .synapses()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, synapse| {
            (hash ^ synapse.weight_na().to_bits()).wrapping_mul(0x0000_0100_0000_01b3)
        });
    let evaluator_config = RewardEvaluatorConfig::new(1.0, 1.0);
    let mut phase_one = initial;
    let mut evaluator = RewardEvaluator::new(evaluator_config)?;
    for _ in 0..epochs {
        train_association(
            &mut phase_one,
            old_context,
            old_action,
            train_amplitude,
            &mut evaluator,
        )?;
    }
    let old_task_after_phase_one = evaluate_task(
        &phase_one,
        old_context,
        old_action,
        test_amplitudes,
        evaluator_config,
    )?;

    let mut without_replay = phase_one.clone();
    let mut with_replay = phase_one;
    let mut no_replay_evaluator = RewardEvaluator::new(evaluator_config)?;
    let mut replay_evaluator = RewardEvaluator::new(evaluator_config)?;
    for _ in 0..epochs {
        train_association(
            &mut without_replay,
            new_context,
            new_action,
            train_amplitude,
            &mut no_replay_evaluator,
        )?;
        idle(&mut without_replay, REPLAY_SLOT_STEPS + 2)?;

        train_association(
            &mut with_replay,
            new_context,
            new_action,
            train_amplitude,
            &mut replay_evaluator,
        )?;
        train_association(
            &mut with_replay,
            old_context,
            old_action,
            train_amplitude,
            &mut replay_evaluator,
        )?;
    }

    Ok(InterferenceSeedResult {
        seed,
        task_variant: variant,
        initialization_fingerprint,
        old_task_after_phase_one,
        old_task_without_replay: evaluate_task(
            &without_replay,
            old_context,
            old_action,
            test_amplitudes,
            evaluator_config,
        )?,
        new_task_without_replay: evaluate_task(
            &without_replay,
            new_context,
            new_action,
            test_amplitudes,
            evaluator_config,
        )?,
        old_task_with_replay: evaluate_task(
            &with_replay,
            old_context,
            old_action,
            test_amplitudes,
            evaluator_config,
        )?,
        new_task_with_replay: evaluate_task(
            &with_replay,
            new_context,
            new_action,
            test_amplitudes,
            evaluator_config,
        )?,
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
    for input in 0..INPUT_COUNT {
        for action in 0..ACTION_COUNT {
            let initial_weight = next_random(&mut state) * 8.0;
            let max_weight = if input == 0 {
                SHARED_CUE_MAX_WEIGHT_NA
            } else {
                120.0
            };
            network.add_synapse(Synapse::new(
                input,
                INPUT_COUNT + action,
                initial_weight,
                0.0,
                max_weight,
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

fn train_association(
    network: &mut LifNetwork,
    context: usize,
    action: usize,
    amplitude: f64,
    evaluator: &mut RewardEvaluator,
) -> Result<(), SimError> {
    let mut cue = [0.0; NETWORK_SIZE];
    cue[0] = amplitude;
    cue[context] = amplitude;
    network.step(&cue)?;

    let mut teacher = [0.0; NETWORK_SIZE];
    teacher[INPUT_COUNT + action] = 200.0;
    let response = network.step(&teacher)?;
    let observed = (0..ACTION_COUNT)
        .map(|index| response.spiking_neurons().contains(&(INPUT_COUNT + index)))
        .collect::<Vec<_>>();
    evaluator.evaluate_and_apply(Some(action), &observed, network)?;
    idle(network, REPLAY_SLOT_STEPS)
}

fn evaluate_task(
    source: &LifNetwork,
    context: usize,
    action: usize,
    amplitudes: [f64; 2],
    config: RewardEvaluatorConfig,
) -> Result<f64, SimError> {
    let mut evaluator = RewardEvaluator::new(config)?;
    for amplitude in amplitudes {
        let mut probe = source.clone();
        let mut cue = [0.0; NETWORK_SIZE];
        cue[0] = amplitude;
        cue[context] = amplitude;
        probe.step(&cue)?;
        let response = probe.step(&[0.0; NETWORK_SIZE])?;
        let observed = (0..ACTION_COUNT)
            .map(|index| response.spiking_neurons().contains(&(INPUT_COUNT + index)))
            .collect::<Vec<_>>();
        evaluator.evaluate(Some(action), &observed)?;
    }
    evaluator
        .metrics()
        .accuracy()
        .ok_or(SimError::InvalidTopology {
            reason: "evaluasi interference tidak memiliki sampel",
        })
}

fn idle(network: &mut LifNetwork, steps: usize) -> Result<(), SimError> {
    for _ in 0..steps {
        network.step(&[0.0; NETWORK_SIZE])?;
    }
    Ok(())
}

fn next_random(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / ((1_u64 << 53) as f64)
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<_> = values.collect();
    values.iter().sum::<f64>() / values.len() as f64
}
