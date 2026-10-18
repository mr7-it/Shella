use shella::{ContinualBenchmarkConfig, SimError};

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
