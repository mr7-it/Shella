use shella::{
    ContinualBenchmarkConfig, ContinualReplayBenchmarkConfig, InterferenceBenchmarkConfig, SimError,
};

#[test]
fn banyak_asosiasi_terpelajari_dan_task_lama_bertahan_setelah_fase_berikutnya() {
    let config = ContinualBenchmarkConfig::standard();
    assert_eq!(config.seeds(), &[1, 7, 19, 42, 99]);
    assert_eq!(config.epochs_per_phase(), 32);
    assert_eq!(config.training_examples_per_network(), 128);
    assert_eq!(config.training_steps_per_network(), 13_056);

    let result = config.run().expect("eksperimen continual valid");
    assert_eq!(result.seed_results().len(), 5);
    assert_eq!(result.training_examples_per_network(), 128);
    assert_eq!(result.training_steps_per_network(), 13_056);

    for seed in result.seed_results() {
        assert_eq!(seed.baseline_test_accuracy(), 0.0);
        assert_eq!(seed.task_a_test_after_phase_one(), 1.0);
        assert_eq!(seed.task_a_test_after_phase_two(), 1.0);
        assert_eq!(seed.task_b_test_after_phase_two(), 1.0);
        assert_eq!(seed.final_test_accuracy(), 1.0);
        assert_eq!(seed.final_validation_accuracy(), 1.0);
        assert_eq!(seed.validation_accuracy_by_amplitude(), [1.0; 4]);
        assert_eq!(seed.forgetting_task_a(), 0.0);
        assert_eq!(seed.accuracy_gain_percentage_points(), 100.0);
        assert!(seed.training_reward() > 0.0);
    }
    assert_eq!(result.mean_final_test_accuracy(), 1.0);
    assert_eq!(result.minimum_final_test_accuracy(), 1.0);
    assert_eq!(result.mean_forgetting_task_a(), 0.0);
    assert_eq!(result.mean_accuracy_gain_percentage_points(), 100.0);
}

#[test]
fn benchmark_multi_seed_deterministik_diulang() {
    let config = ContinualBenchmarkConfig::standard();
    let first = config.run().expect("run pertama");
    let second = config.run().expect("run kedua");
    assert_eq!(first, second);
}

#[test]
fn benchmark_menolak_konfigurasi_kosong_duplikat_nol_dan_overflow() {
    assert!(matches!(
        ContinualBenchmarkConfig::new(Vec::new(), 1),
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
    assert!(matches!(
        ContinualBenchmarkConfig::new(vec![1], usize::MAX),
        Err(SimError::CounterOverflow { .. })
    ));
}

#[test]
fn interference_membandingkan_replay_dan_kontrol_pada_seed_dan_tugas_bervariasi() {
    let config = InterferenceBenchmarkConfig::standard();
    let result = config.run().expect("eksperimen konflik berjalan");

    assert_eq!(result.runs().len(), 32);
    assert_eq!(config.seeds().len(), 8);
    assert_eq!(config.task_variants(), 4);
    assert!(
        result
            .runs()
            .iter()
            .all(|run| run.old_task_after_phase_one() == 1.0)
    );
    let initializations: std::collections::BTreeSet<_> = result
        .runs()
        .iter()
        .filter(|run| run.task_variant() == 0)
        .map(|run| run.initialization_fingerprint())
        .collect();
    assert_eq!(initializations.len(), config.seeds().len());
    assert!(result.runs().iter().all(|run| {
        [
            run.old_task_after_phase_one(),
            run.old_task_without_replay(),
            run.new_task_without_replay(),
            run.old_task_with_replay(),
            run.new_task_with_replay(),
        ]
        .iter()
        .all(|accuracy| (0.0..=1.0).contains(accuracy))
    }));
    assert_eq!(result.mean_forgetting_without_replay(), 0.5);
    assert_eq!(result.mean_forgetting_with_replay(), 0.0);
    assert_eq!(result.mean_old_task_without_replay(), 0.5);
    assert_eq!(result.mean_old_task_with_replay(), 1.0);
    assert_eq!(result.mean_new_task_without_replay(), 0.5);
    assert_eq!(result.mean_new_task_with_replay(), 0.5);
    assert!(
        InterferenceBenchmarkConfig::standard()
            .run()
            .expect("eksperimen deterministik kedua")
            == result
    );
}

#[test]
fn interference_menolak_seed_atau_variasi_tugas_tidak_valid() {
    assert!(InterferenceBenchmarkConfig::new(Vec::new(), 1, 1).is_err());
    assert!(InterferenceBenchmarkConfig::new(vec![1], 0, 1).is_err());
    assert!(InterferenceBenchmarkConfig::new(vec![1], 1, 0).is_err());
    assert!(InterferenceBenchmarkConfig::new(vec![1, 1], 1, 1).is_err());
}

#[test]
fn continual_replay_menggunakan_budget_seimbang_split_terpisah_dan_seed_independen() {
    let config = ContinualReplayBenchmarkConfig::standard();
    let result = config.run().expect("benchmark continual replay berjalan");
    assert_eq!(result.runs().len(), 32);

    let fingerprints: std::collections::BTreeSet<_> = result
        .runs()
        .iter()
        .map(|run| run.initialization_fingerprint())
        .collect();
    assert_eq!(fingerprints.len(), result.runs().len());

    for run in result.runs() {
        assert_eq!(run.training_presentations_per_method(), 1280);
        assert_eq!(run.training_steps_per_method(), 1280 * 23);
        assert_eq!(run.test().len(), config.test_noise_levels_na().len());
        assert_eq!(run.validation().after_task_a().task_a().samples(), 72);
        assert_eq!(run.validation().after_task_a().task_b().samples(), 72);
        for split in run.test() {
            assert_eq!(split.before_learning().task_a().samples(), 72);
            assert_eq!(split.before_learning().task_b().samples(), 72);
            assert_eq!(split.after_task_a().task_a().samples(), 72);
            assert_eq!(split.without_replay().task_b().samples(), 72);
            assert_eq!(split.with_replay().task_a().samples(), 72);
        }
    }
    let validation_without_a = result
        .runs()
        .iter()
        .map(|run| {
            run.validation()
                .without_replay()
                .task_a()
                .accuracy()
                .unwrap()
        })
        .sum::<f64>()
        / result.runs().len() as f64;
    let validation_with_a = result
        .runs()
        .iter()
        .map(|run| run.validation().with_replay().task_a().accuracy().unwrap())
        .sum::<f64>()
        / result.runs().len() as f64;
    let validation_after_a = result
        .runs()
        .iter()
        .map(|run| run.validation().after_task_a().task_a().accuracy().unwrap())
        .sum::<f64>()
        / result.runs().len() as f64;
    assert!(validation_after_a >= 0.8);
    assert!(validation_with_a > validation_without_a);
    let forgetting_without = (validation_after_a - validation_without_a).max(0.0);
    let forgetting_with = (validation_after_a - validation_with_a).max(0.0);
    assert!(forgetting_with < forgetting_without);
    let validation_without_mean = result
        .runs()
        .iter()
        .map(|run| {
            (run.validation()
                .without_replay()
                .task_a()
                .accuracy()
                .unwrap()
                + run
                    .validation()
                    .without_replay()
                    .task_b()
                    .accuracy()
                    .unwrap())
                / 2.0
        })
        .sum::<f64>()
        / result.runs().len() as f64;
    let validation_with_mean = result
        .runs()
        .iter()
        .map(|run| {
            (run.validation().with_replay().task_a().accuracy().unwrap()
                + run.validation().with_replay().task_b().accuracy().unwrap())
                / 2.0
        })
        .sum::<f64>()
        / result.runs().len() as f64;
    assert!(validation_with_mean > validation_without_mean);
    assert!(result.mean_forgetting_without_replay(30).is_some());
    assert!(result.mean_forgetting_with_replay(30).is_some());
    assert_eq!(
        result.catastrophic_forgetting_rate_without_replay(30),
        Some(0.0)
    );
    assert_eq!(
        result.catastrophic_forgetting_rate_with_replay(30),
        Some(0.0)
    );
    assert!(result.forgetting_stddev_with_replay(30).is_some());
    assert_eq!(config.run().expect("benchmark deterministik kedua"), result);
}

#[test]
fn continual_replay_menolak_konfigurasi_tidak_valid() {
    assert!(ContinualReplayBenchmarkConfig::new(Vec::new(), 1, 1, 1).is_err());
    assert!(ContinualReplayBenchmarkConfig::new(vec![1], 0, 1, 1).is_err());
    assert!(ContinualReplayBenchmarkConfig::new(vec![1], 5, 1, 1).is_err());
    assert!(ContinualReplayBenchmarkConfig::new(vec![1], 1, 0, 1).is_err());
    assert!(ContinualReplayBenchmarkConfig::new(vec![1], 1, 1, 0).is_err());
    assert!(ContinualReplayBenchmarkConfig::new(vec![1, 1], 1, 1, 1).is_err());
}
