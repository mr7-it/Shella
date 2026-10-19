use shella::{
    GeneralizationBenchmarkConfig, LifConfig, LifNetwork, RewardEvaluator, RewardEvaluatorConfig,
    RewardModulatedStdpConfig, StdpConfig, Synapse,
};

const ACTION: usize = 3;
const NEURON_COUNT: usize = 4;
const TRAINING_TRIALS: usize = 32;

fn network() -> LifNetwork {
    let config = LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1);
    let mut network = LifNetwork::new(config, NEURON_COUNT).expect("network");
    for cue in 0..3 {
        network
            .add_synapse(Synapse::new(cue, ACTION, 0.0, 0.0, 150.0, 1).expect("synapse"))
            .expect("add synapse");
    }
    network
        .enable_reward_modulated_stdp(RewardModulatedStdpConfig::new(
            StdpConfig::new(1.0, 1.0, 10.0, 10.0),
            20.0,
            0.75,
        ))
        .expect("reward STDP");
    network
}

fn train_cue(network: &mut LifNetwork, cue_neuron: usize, cue_current_na: f64) {
    let mut evaluator =
        RewardEvaluator::new(RewardEvaluatorConfig::new(1.0, 1.0)).expect("evaluator");
    for _ in 0..TRAINING_TRIALS {
        let mut cue_currents = [0.0; NEURON_COUNT];
        cue_currents[cue_neuron] = cue_current_na;
        let cue_output = network.step(&cue_currents).expect("training cue");
        assert!(cue_output.spiking_neurons().contains(&cue_neuron));
        let teacher_output = network.step(&[0.0, 0.0, 0.0, 100.0]).expect("teacher");
        evaluator
            .evaluate_and_apply(
                Some(0),
                &[teacher_output.spiking_neurons().contains(&ACTION)],
                network,
            )
            .expect("reward update");
        network.step(&[0.0; NEURON_COUNT]).expect("trial gap");
    }
}

fn washout(network: &mut LifNetwork) {
    for _ in 0..50 {
        network.step(&[0.0; NEURON_COUNT]).expect("washout");
    }
}

fn train_only_cue_zero() -> LifNetwork {
    let mut network = network();
    train_cue(&mut network, 0, 100.0);
    washout(&mut network);
    network
}

fn present_cue_pattern(network: &mut LifNetwork, currents: [f64; NEURON_COUNT]) -> bool {
    network.step(&currents).expect("cue pattern");
    network
        .step(&[0.0; NEURON_COUNT])
        .expect("delayed synaptic response")
        .spiking_neurons()
        .contains(&ACTION)
}

#[test]
fn pembelajaran_berhasil_pada_holdout_amplitudo_dan_kombinasi_distractor() {
    let mut learned = train_only_cue_zero();
    let mut untrained = network();

    // Kombinasi ini tidak muncul saat training: cue yang dipelajari memakai amplitudo
    // berbeda dan ditemani cue kedua yang tidak pernah dilatih.
    let held_out_pattern = [80.0, 80.0, 0.0, 0.0];
    let learned_response = present_cue_pattern(&mut learned, held_out_pattern);
    let baseline_response = present_cue_pattern(&mut untrained, held_out_pattern);

    assert!(
        learned_response,
        "asosiasi cue-0 yang dipelajari harus bertahan pada variasi held-out"
    );
    assert!(
        !baseline_response,
        "kontrol bobot nol tidak boleh menghasilkan aksi"
    );
    assert!(learned.synapses()[0].weight_na() > 0.0);
    assert_eq!(learned.synapses()[1].weight_na(), 0.0);
    assert_eq!(learned.synapses()[2].weight_na(), 0.0);
}

#[test]
fn cue_unseen_tidak_dianggap_dipahami_oleh_model() {
    let mut learned = train_only_cue_zero();
    let mut untrained = network();

    // Cue-2 sama sekali tidak ikut training. Tidak ada representasi semantik yang
    // menghubungkannya dengan cue-0, sehingga target yang benar di probe ini adalah diam.
    let unseen_cue = [0.0, 0.0, 80.0, 0.0];
    assert!(!present_cue_pattern(&mut learned, unseen_cue));
    assert!(!present_cue_pattern(&mut untrained, unseen_cue));
}

#[test]
fn dapat_mempelajari_asosiasi_baru_setelah_pola_awal() {
    let mut online_learning = train_only_cue_zero();
    let mut untrained_control = network();

    // Cue-2 tidak ikut asosiasi pertama. Pelajari asosiasi baru cue-2 -> aksi dengan
    // umpan balik eksplisit, lalu uji tanpa guru pada amplitudo yang berbeda.
    train_cue(&mut online_learning, 2, 100.0);
    washout(&mut online_learning);
    let new_pattern = [0.0, 0.0, 80.0, 0.0];

    assert!(present_cue_pattern(&mut online_learning, new_pattern));
    assert!(!present_cue_pattern(&mut untrained_control, new_pattern));
    assert!(online_learning.synapses()[0].weight_na() > 0.0);
    assert_eq!(online_learning.synapses()[1].weight_na(), 0.0);
    assert!(online_learning.synapses()[2].weight_na() > 0.0);
}

#[test]
fn pola_gabungan_held_out_bertahan_pada_noise_rendah_dan_mengungguli_baseline() {
    let config = GeneralizationBenchmarkConfig::new(vec![17, 31], 2, 32, 12)
        .expect("konfigurasi generalization valid");
    let result = config.run().expect("benchmark generalization berjalan");

    for noise in config.noise_levels_na() {
        println!(
            "noise={noise:.0} nA accuracy={:?} fp_rate={:?} fn_rate={:?} abstain={:?}",
            result.mean_trained_accuracy(*noise),
            result.mean_false_positive_rate(*noise),
            result.mean_false_negative_rate(*noise),
            result.mean_abstention_rate(*noise),
        );
    }
    assert_eq!(result.runs().len(), 4);
    assert!(result.mean_baseline_accuracy(0.0).unwrap() < 1.0);
    assert!(
        result.mean_baseline_accuracy(0.0).unwrap() < result.mean_trained_accuracy(0.0).unwrap()
    );
    let fingerprints: std::collections::BTreeSet<_> = result
        .runs()
        .iter()
        .filter(|run| run.task_variant() == 0)
        .map(|run| run.initialization_fingerprint())
        .collect();
    assert_eq!(fingerprints.len(), config.seeds().len());
    for run in result.runs() {
        for condition in run.trained() {
            assert_eq!(condition.metrics().samples(), 36);
            assert_eq!(condition.metrics().positive_samples(), 24);
            assert_eq!(condition.metrics().negative_samples(), 12);
        }
    }
    assert!(result.mean_trained_accuracy(0.0).unwrap() >= 0.9);
    assert!(result.mean_trained_accuracy(20.0).unwrap() >= 0.75);
    assert!(result.mean_trained_accuracy(60.0).unwrap() < 1.0);
    assert!(result.mean_false_positive_rate(60.0).unwrap() > 0.0);
}

#[test]
fn benchmark_generalization_multi_seed_deterministik() {
    let config =
        GeneralizationBenchmarkConfig::new(vec![17, 31], 2, 8, 6).expect("konfigurasi valid");
    assert_eq!(
        config.run().expect("run pertama"),
        config.run().expect("run kedua")
    );
}

#[test]
fn benchmark_generalization_menolak_konfigurasi_tidak_valid() {
    assert!(GeneralizationBenchmarkConfig::new(Vec::new(), 1, 1, 1).is_err());
    assert!(GeneralizationBenchmarkConfig::new(vec![1], 0, 1, 1).is_err());
    assert!(GeneralizationBenchmarkConfig::new(vec![1], 5, 1, 1).is_err());
    assert!(GeneralizationBenchmarkConfig::new(vec![1], 1, 0, 1).is_err());
    assert!(GeneralizationBenchmarkConfig::new(vec![1], 1, 1, 0).is_err());
    assert!(GeneralizationBenchmarkConfig::new(vec![1, 1], 1, 1, 1).is_err());
}
