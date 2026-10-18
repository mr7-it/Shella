use crate::{
    LifConfig, LifNetwork, RewardEvaluator, RewardEvaluatorConfig, RewardMetrics,
    RewardModulatedStdpConfig, SimError, StdpConfig, Synapse,
};

const CUE_APPROACH: usize = 0;
const CUE_WITHHOLD: usize = 1;
const ACTION_NEURON: usize = 2;
const NETWORK_SIZE: usize = 3;
const TRAINING_SAMPLES_PER_EPOCH: usize = 2;
const STEPS_PER_TRAINING_SAMPLE: usize = 3;
const WASHOUT_STEPS: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConditioningSample {
    cue_neuron: usize,
    expected_action: Option<usize>,
    teacher_spike: bool,
}

const CONDITIONING_SAMPLES: [ConditioningSample; TRAINING_SAMPLES_PER_EPOCH] = [
    ConditioningSample {
        cue_neuron: CUE_APPROACH,
        expected_action: Some(0),
        teacher_spike: true,
    },
    ConditioningSample {
        cue_neuron: CUE_WITHHOLD,
        expected_action: None,
        teacher_spike: false,
    },
];

/// Ukuran protokol conditioning deterministik untuk evaluasi recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditioningBenchmarkConfig {
    training_epochs: usize,
    evaluation_repetitions: usize,
}

impl ConditioningBenchmarkConfig {
    /// Membuat protokol; jumlah epoch dan pengulangan uji harus positif.
    pub fn new(training_epochs: usize, evaluation_repetitions: usize) -> Result<Self, SimError> {
        if training_epochs == 0 {
            return Err(SimError::InvalidConfiguration {
                parameter: "training_epochs",
                reason: "harus lebih besar dari nol",
            });
        }
        if evaluation_repetitions == 0 {
            return Err(SimError::InvalidConfiguration {
                parameter: "evaluation_repetitions",
                reason: "harus lebih besar dari nol",
            });
        }
        let training_steps = training_epochs
            .checked_mul(TRAINING_SAMPLES_PER_EPOCH)
            .and_then(|count| count.checked_mul(STEPS_PER_TRAINING_SAMPLE))
            .ok_or(SimError::CounterOverflow {
                counter: "langkah pelatihan benchmark",
            })?;
        let evaluation_steps = evaluation_repetitions
            .checked_mul(TRAINING_SAMPLES_PER_EPOCH)
            .and_then(|count| count.checked_mul(STEPS_PER_TRAINING_SAMPLE))
            .and_then(|count| count.checked_add(WASHOUT_STEPS))
            .ok_or(SimError::CounterOverflow {
                counter: "langkah evaluasi benchmark",
            })?;
        training_steps
            .checked_add(evaluation_steps)
            .and_then(|steps| u64::try_from(steps).ok())
            .ok_or(SimError::CounterOverflow {
                counter: "langkah total benchmark",
            })?;
        Ok(Self {
            training_epochs,
            evaluation_repetitions,
        })
    }

    /// Konfigurasi benchmark baku: 32 epoch pelatihan dan 8 pengulangan per pola.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            training_epochs: 32,
            evaluation_repetitions: 8,
        }
    }

    /// Jumlah epoch dengan urutan pola tetap.
    #[must_use]
    pub const fn training_epochs(self) -> usize {
        self.training_epochs
    }

    /// Jumlah pengulangan tiap pola pada evaluasi tanpa guru.
    #[must_use]
    pub const fn evaluation_repetitions(self) -> usize {
        self.evaluation_repetitions
    }

    /// Menjalankan kontrol tanpa pembaruan dan perlakuan reward-modulated.
    pub fn run(self) -> Result<ConditioningBenchmarkResult, SimError> {
        let mut baseline = make_network()?;
        let mut learned = make_network()?;
        let initial_weights = weights(&learned)?;
        let evaluator_config = RewardEvaluatorConfig::new(1.0, 1.0);
        let mut baseline_feedback = RewardEvaluator::new(evaluator_config)?;
        let mut training_feedback = RewardEvaluator::new(evaluator_config)?;

        for _ in 0..self.training_epochs {
            for sample in CONDITIONING_SAMPLES {
                let baseline_observed = run_teacher_trial(&mut baseline, sample)?;
                baseline_feedback.evaluate(sample.expected_action, &baseline_observed)?;
                baseline.step(&[0.0; NETWORK_SIZE])?;

                let observed = run_teacher_trial(&mut learned, sample)?;
                training_feedback.evaluate_and_apply(
                    sample.expected_action,
                    &observed,
                    &mut learned,
                )?;
                learned.step(&[0.0; NETWORK_SIZE])?;
            }
        }

        for _ in 0..WASHOUT_STEPS {
            baseline.step(&[0.0; NETWORK_SIZE])?;
            learned.step(&[0.0; NETWORK_SIZE])?;
        }

        let mut baseline_evaluator = RewardEvaluator::new(evaluator_config)?;
        let mut learned_evaluator = RewardEvaluator::new(evaluator_config)?;
        for _ in 0..self.evaluation_repetitions {
            for sample in CONDITIONING_SAMPLES {
                let baseline_observed = run_recall_trial(&mut baseline, sample.cue_neuron)?;
                baseline_evaluator.evaluate(sample.expected_action, &baseline_observed)?;
                baseline.step(&[0.0; NETWORK_SIZE])?;

                let learned_observed = run_recall_trial(&mut learned, sample.cue_neuron)?;
                learned_evaluator.evaluate(sample.expected_action, &learned_observed)?;
                learned.step(&[0.0; NETWORK_SIZE])?;
            }
        }

        let training_examples = self
            .training_epochs
            .checked_mul(TRAINING_SAMPLES_PER_EPOCH)
            .and_then(|count| u64::try_from(count).ok())
            .ok_or(SimError::CounterOverflow {
                counter: "sampel pelatihan benchmark",
            })?;
        let training_steps = training_examples
            .checked_mul(STEPS_PER_TRAINING_SAMPLE as u64)
            .ok_or(SimError::CounterOverflow {
                counter: "langkah pelatihan benchmark",
            })?;

        Ok(ConditioningBenchmarkResult {
            baseline_metrics: baseline_evaluator.metrics(),
            learned_metrics: learned_evaluator.metrics(),
            training_feedback_samples: training_feedback.metrics().samples(),
            training_feedback_reward: training_feedback.metrics().total_reward(),
            initial_weights,
            learned_weights: weights(&learned)?,
            training_epochs: self.training_epochs,
            evaluation_repetitions: self.evaluation_repetitions,
            training_steps,
        })
    }
}

impl Default for ConditioningBenchmarkConfig {
    fn default() -> Self {
        Self::standard()
    }
}

/// Ringkasan recall kontrol dan jaringan yang dilatih pada benchmark conditioning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConditioningBenchmarkResult {
    baseline_metrics: RewardMetrics,
    learned_metrics: RewardMetrics,
    training_feedback_samples: u64,
    training_feedback_reward: f64,
    initial_weights: [f64; 2],
    learned_weights: [f64; 2],
    training_epochs: usize,
    evaluation_repetitions: usize,
    training_steps: u64,
}

impl ConditioningBenchmarkResult {
    /// Metrik recall kontrol tanpa menerapkan reward saat pelatihan.
    #[must_use]
    pub const fn baseline_metrics(self) -> RewardMetrics {
        self.baseline_metrics
    }

    /// Metrik recall jaringan setelah pembelajaran reward-modulated.
    #[must_use]
    pub const fn learned_metrics(self) -> RewardMetrics {
        self.learned_metrics
    }

    /// Selisih akurasi learned minus baseline, bila kedua evaluasi memiliki sampel.
    #[must_use]
    pub fn accuracy_gain(self) -> Option<f64> {
        super::statistics::accuracy_gain(self.baseline_metrics, self.learned_metrics)
    }

    /// Jumlah respons yang dinilai evaluator selama umpan balik pelatihan.
    #[must_use]
    pub const fn training_feedback_samples(self) -> u64 {
        self.training_feedback_samples
    }

    /// Akumulasi reward signed selama pelatihan, bukan skor recall.
    #[must_use]
    pub const fn training_feedback_reward(self) -> f64 {
        self.training_feedback_reward
    }

    /// Bobot awal [isyarat mendekat, isyarat menahan].
    #[must_use]
    pub const fn initial_weights(self) -> [f64; 2] {
        self.initial_weights
    }

    /// Bobot akhir [isyarat mendekat, isyarat menahan].
    #[must_use]
    pub const fn learned_weights(self) -> [f64; 2] {
        self.learned_weights
    }

    /// Epoch pelatihan.
    #[must_use]
    pub const fn training_epochs(self) -> usize {
        self.training_epochs
    }

    /// Pengulangan tiap pola pada recall.
    #[must_use]
    pub const fn evaluation_repetitions(self) -> usize {
        self.evaluation_repetitions
    }

    /// Jumlah pemanggilan langkah jaringan untuk data pelatihan per jaringan.
    #[must_use]
    pub const fn training_steps(self) -> u64 {
        self.training_steps
    }
}

fn make_network() -> Result<LifNetwork, SimError> {
    let neuron_config = LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1);
    let mut network = LifNetwork::new(neuron_config, NETWORK_SIZE)?;
    network.add_synapse(Synapse::new(0, ACTION_NEURON, 0.0, 0.0, 150.0, 1)?)?;
    network.add_synapse(Synapse::new(1, ACTION_NEURON, 0.0, 0.0, 150.0, 1)?)?;
    let learning =
        RewardModulatedStdpConfig::new(StdpConfig::new(1.0, 1.0, 10.0, 10.0), 20.0, 0.75);
    network.enable_reward_modulated_stdp(learning)?;
    Ok(network)
}

fn run_teacher_trial(
    network: &mut LifNetwork,
    sample: ConditioningSample,
) -> Result<[bool; 1], SimError> {
    let mut cue_currents = [0.0; NETWORK_SIZE];
    cue_currents[sample.cue_neuron] = 100.0;
    network.step(&cue_currents)?;

    let mut teacher_currents = [0.0; NETWORK_SIZE];
    if sample.teacher_spike {
        teacher_currents[ACTION_NEURON] = 100.0;
    }
    let response = network.step(&teacher_currents)?;
    Ok([response.spiking_neurons().contains(&ACTION_NEURON)])
}

fn run_recall_trial(network: &mut LifNetwork, cue_neuron: usize) -> Result<[bool; 1], SimError> {
    let mut cue_currents = [0.0; NETWORK_SIZE];
    cue_currents[cue_neuron] = 100.0;
    network.step(&cue_currents)?;
    let response = network.step(&[0.0; NETWORK_SIZE])?;
    Ok([response.spiking_neurons().contains(&ACTION_NEURON)])
}

fn weights(network: &LifNetwork) -> Result<[f64; 2], SimError> {
    let synapses = network.synapses();
    if synapses.len() != 2 {
        return Err(SimError::InvalidTopology {
            reason: "benchmark conditioning mengharapkan tepat dua sinaps",
        });
    }
    Ok([synapses[0].weight_na(), synapses[1].weight_na()])
}
