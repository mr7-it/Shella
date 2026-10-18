use shella::{
    ConditioningBenchmarkConfig, LifConfig, LifNetwork, RewardEvaluator, RewardEvaluatorConfig,
    RewardModulatedStdpConfig, SimError, StdpConfig, Synapse,
};

fn evaluator() -> RewardEvaluator {
    RewardEvaluator::new(RewardEvaluatorConfig::new(1.0, 0.5)).expect("evaluator valid")
}

fn reward_network() -> LifNetwork {
    let config = LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 0);
    let mut network = LifNetwork::new(config, 2).expect("jaringan valid");
    network
        .add_synapse(Synapse::new(0, 1, 0.0, 0.0, 10.0, 1).expect("sinaps valid"))
        .expect("sinaps ditambahkan");
    network
        .enable_reward_modulated_stdp(RewardModulatedStdpConfig::new(
            StdpConfig::new(1.0, 1.0, 10.0, 10.0),
            20.0,
            1.0,
        ))
        .expect("reward STDP aktif");
    network
}

fn causal_pair(network: &mut LifNetwork) {
    network.step(&[100.0, 0.0]).expect("spike pra");
    network.step(&[0.0, 100.0]).expect("spike pasca");
}

#[test]
fn evaluator_memberi_reward_atau_penalti_berdasarkan_aksi_teramati() {
    let mut evaluator = evaluator();
    let correct = evaluator
        .evaluate(Some(1), &[false, true])
        .expect("respons dapat dinilai");
    assert_eq!(correct.expected_action(), Some(1));
    assert_eq!(correct.predicted_action(), Some(1));
    assert!(correct.correct());
    assert_eq!(correct.reward(), 1.0);

    let incorrect = evaluator
        .evaluate(Some(1), &[true, false])
        .expect("respons salah tetap menghasilkan reward");
    assert_eq!(incorrect.predicted_action(), Some(0));
    assert!(!incorrect.correct());
    assert_eq!(incorrect.reward(), -0.5);
}

#[test]
fn evaluator_mendukung_target_diam_dan_menghitung_tanpa_respons() {
    let mut evaluator = evaluator();
    let correct_silence = evaluator
        .evaluate(None, &[false, false])
        .expect("diam adalah target yang sah");
    assert_eq!(correct_silence.predicted_action(), None);
    assert!(correct_silence.correct());
    assert_eq!(correct_silence.reward(), 1.0);

    let missed_action = evaluator
        .evaluate(Some(0), &[false, false])
        .expect("tidak merespons dihitung sebagai salah");
    assert_eq!(missed_action.predicted_action(), None);
    assert!(!missed_action.correct());
    assert_eq!(missed_action.reward(), -0.5);

    let metrics = evaluator.metrics();
    assert_eq!(metrics.samples(), 2);
    assert_eq!(metrics.correct(), 1);
    assert_eq!(metrics.no_response(), 2);
    assert_eq!(metrics.accuracy(), Some(0.5));
    assert_eq!(metrics.total_reward(), 0.5);
    assert_eq!(metrics.mean_reward(), Some(0.25));
}

#[test]
fn evaluator_memilih_indeks_aktif_terendah_secara_deterministik() {
    let mut evaluator = evaluator();
    let result = evaluator
        .evaluate(Some(0), &[true, true, true])
        .expect("beberapa aksi tetap deterministik");
    assert_eq!(result.predicted_action(), Some(0));
    assert!(result.correct());
}

#[test]
fn evaluator_menolak_config_target_dan_observasi_tidak_sah_tanpa_mengubah_metrik() {
    assert!(RewardEvaluator::new(RewardEvaluatorConfig::new(0.0, 1.0)).is_err());
    assert!(RewardEvaluator::new(RewardEvaluatorConfig::new(1.0, f64::INFINITY)).is_err());

    let mut evaluator = evaluator();
    assert!(matches!(
        evaluator.evaluate(None, &[]),
        Err(SimError::InvalidInput { .. })
    ));
    assert!(matches!(
        evaluator.evaluate(Some(2), &[false, false]),
        Err(SimError::InvalidInput { .. })
    ));
    assert_eq!(evaluator.metrics().samples(), 0);
}

#[test]
fn evaluator_mengirim_reward_ke_jaringan_dan_metrik_mencerminkan_hasilnya() {
    let mut network = reward_network();
    causal_pair(&mut network);
    assert_eq!(network.synapses()[0].weight_na(), 0.0);

    let mut evaluator = evaluator();
    let result = evaluator
        .evaluate_and_apply(Some(0), &[true], &mut network)
        .expect("reward evaluator diterapkan");
    assert!(result.correct());
    assert!(network.synapses()[0].weight_na() > 0.0);
    assert_eq!(evaluator.metrics().samples(), 1);
    assert_eq!(evaluator.metrics().correct(), 1);
}

#[test]
fn evaluator_tidak_mencatat_sampel_jika_penerapan_reward_gagal() {
    let mut network = LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 0), 1)
        .expect("jaringan valid tanpa reward mode");
    let mut evaluator = evaluator();
    assert!(matches!(
        evaluator.evaluate_and_apply(Some(0), &[true], &mut network),
        Err(SimError::InvalidInput { .. })
    ));
    assert_eq!(evaluator.metrics().samples(), 0);
}

#[test]
fn benchmark_conditioning_mengungguli_baseline_tanpa_update_bobot() {
    let result = ConditioningBenchmarkConfig::standard()
        .run()
        .expect("benchmark conditioning berjalan");
    assert_eq!(result.baseline_metrics().samples(), 16);
    assert_eq!(result.learned_metrics().samples(), 16);
    assert_eq!(result.baseline_metrics().accuracy(), Some(0.5));
    assert_eq!(result.learned_metrics().accuracy(), Some(1.0));
    assert_eq!(result.accuracy_gain(), Some(0.5));
    assert_eq!(result.initial_weights(), [0.0, 0.0]);
    assert!(result.learned_weights()[0] > result.learned_weights()[1]);
    assert!(result.training_feedback_reward() > 0.0);
    assert_eq!(result.training_feedback_samples(), 64);
    assert_eq!(result.training_steps(), 192);
}

#[test]
fn benchmark_conditioning_dapat_diulang_dengan_hasil_identik() {
    let config = ConditioningBenchmarkConfig::new(8, 3).expect("protokol valid");
    assert_eq!(
        config.run().expect("run pertama"),
        config.run().expect("run kedua")
    );
}

#[test]
fn benchmark_menolak_ukuran_nol_atau_perhitungan_sampel_yang_meluap() {
    assert!(ConditioningBenchmarkConfig::new(0, 1).is_err());
    assert!(ConditioningBenchmarkConfig::new(1, 0).is_err());
    assert!(matches!(
        ConditioningBenchmarkConfig::new(usize::MAX, 1),
        Err(SimError::CounterOverflow { .. })
    ));
    assert!(matches!(
        ConditioningBenchmarkConfig::new(1, usize::MAX),
        Err(SimError::CounterOverflow { .. })
    ));
}
