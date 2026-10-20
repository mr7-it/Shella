use shella::{
    Checkpoint, ClassKnowledgeModel, ExemplarPolicy, FeatureScaler, IrisFeatures, IrisSample,
    LearningMemory, LegacyPrototypeSnapshot, LifConfig, LifNetwork, SecureMemoryStore, TrainScaler,
    evaluate_predictions, feature_definitions, stratified_folds, stratified_split,
};

fn samples() -> Vec<IrisSample> {
    (0..30)
        .map(|index| {
            let label = index / 10;
            IrisSample::new(
                [
                    1.0 + label as f64 + index as f64 * 0.01,
                    2.0 + label as f64,
                    3.0 + label as f64,
                    4.0 + label as f64,
                ],
                label,
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn split_stratified_disjoint_dan_mencakup_semua_sampel() {
    let data = samples();
    let split = stratified_split(&data, 17).unwrap();
    assert_eq!(
        split.train().len() + split.validation().len() + split.test().len(),
        data.len()
    );
    assert!(
        split
            .train()
            .iter()
            .all(|index| !split.validation().contains(index) && !split.test().contains(index))
    );
    assert!(
        split
            .validation()
            .iter()
            .all(|index| !split.test().contains(index))
    );
    for class in 0..3 {
        assert!(
            split
                .train()
                .iter()
                .any(|index| data[*index].label() == class)
        );
        assert!(
            split
                .validation()
                .iter()
                .any(|index| data[*index].label() == class)
        );
        assert!(
            split
                .test()
                .iter()
                .any(|index| data[*index].label() == class)
        );
    }
}

#[test]
fn scaler_hanya_fit_pada_train_dan_cv_stratified() {
    let data = samples();
    let split = stratified_split(&data, 31).unwrap();
    let scaler = TrainScaler::fit(&data, split.train()).unwrap();
    let transformed = scaler.transform(data[split.test()[0]].features());
    assert!(transformed.iter().all(|value| value.is_finite()));
    let mut pool = split.train().to_vec();
    pool.extend_from_slice(split.validation());
    let folds = stratified_folds(&data, &pool, 5).unwrap();
    assert_eq!(folds.len(), 5);
    for (train, validation) in folds {
        assert!(train.iter().all(|index| !validation.contains(index)));
        assert_eq!(train.len() + validation.len(), pool.len());
    }
}

#[test]
fn self_evaluation_menghitung_kesalahan_dan_ketidakpastian() {
    let evaluation =
        evaluate_predictions(&[(0, Some(0), 0.8), (1, Some(0), 0.1), (2, None, 0.05)]).unwrap();
    assert_eq!(evaluation.weak_samples(), 2);
    assert_eq!(evaluation.uncertainty_samples(), 2);
    assert_eq!(evaluation.metrics().samples(), 3);
    assert_eq!(evaluation.metrics().false_negative_rate(), Some(2.0 / 3.0));
}

fn iris_uci_samples() -> Vec<IrisSample> {
    include_str!("../data/iris_uci.csv")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let columns = line.split(',').collect::<Vec<_>>();
            let features = [
                columns[0].parse().unwrap(),
                columns[1].parse().unwrap(),
                columns[2].parse().unwrap(),
                columns[3].parse().unwrap(),
            ];
            let label = match columns[4] {
                "Iris-setosa" => 0,
                "Iris-versicolor" => 1,
                "Iris-virginica" => 2,
                other => panic!("label Iris UCI tidak dikenal: {other}"),
            };
            IrisSample::new(features, label).unwrap()
        })
        .collect()
}

#[test]
fn feature_schema_menghasilkan_fitur_turunan_finite_dan_terversi() {
    let features = IrisFeatures::from_raw([5.1, 3.5, 1.4, 0.2]).unwrap();
    assert!((features.values()[4] - 17.85).abs() < 1e-12);
    assert!((features.values()[5] - 0.28).abs() < 1e-12);
    assert_eq!(feature_definitions().len(), 10);
    assert!(
        feature_definitions()
            .iter()
            .all(|feature| feature.active && feature.schema_version == 2)
    );
}

#[test]
fn class_knowledge_memilih_multi_prototype_deterministik_dari_iris_uci() {
    let samples = iris_uci_samples();
    let split = stratified_split(&samples, 17).unwrap();
    let training = split
        .train()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let validation = split
        .validation()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let model =
        ClassKnowledgeModel::fit(&training, &validation, "iris-uci-v1", "test-run", 17).unwrap();
    let repeated =
        ClassKnowledgeModel::fit(&training, &validation, "iris-uci-v1", "test-run", 17).unwrap();

    model.validate().unwrap();
    assert_eq!(
        model.selected_prototypes_per_class,
        repeated.selected_prototypes_per_class
    );
    assert!((2..=8).contains(&model.selected_prototypes_per_class));
    assert_eq!(model.classes, repeated.classes);
    assert_eq!(
        model
            .prototype_candidate_scores
            .iter()
            .map(|candidate| candidate.prototypes_per_class)
            .collect::<Vec<_>>(),
        vec![2, 3, 4, 5, 6, 8]
    );
    assert_eq!(model.classes.len(), 3);
    assert!(model.classes.iter().all(|class| {
        class.sample_count > 0
            && class.prototypes.len() == model.selected_prototypes_per_class
            && class.exemplars.len() == 2
    }));
    assert!(model.validation_accuracy.is_finite());
    assert_eq!(
        model.predict([5.1, 3.5, 1.4, 0.2]).unwrap(),
        model.predict([5.1, 3.5, 1.4, 0.2]).unwrap()
    );
    assert!(
        model
            .predict([5.1, 3.5, 1.4, 0.2])
            .unwrap()
            .calibrated_confidence
            .is_some_and(|confidence| (0.0..=1.0).contains(&confidence))
    );
}

#[test]
fn class_knowledge_scaler_hanya_memakai_batas_training_iris() {
    let samples = iris_uci_samples();
    let split = stratified_split(&samples, 31).unwrap();
    let training = split
        .train()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let scaler = FeatureScaler::fit(&training).unwrap();
    let expected_minimum = (0..10)
        .map(|feature| {
            training
                .iter()
                .map(|sample| IrisFeatures::from_raw(*sample.features()).unwrap().values()[feature])
                .fold(f64::INFINITY, f64::min)
        })
        .collect::<Vec<_>>();
    assert_eq!(scaler.minimum().as_slice(), expected_minimum.as_slice());
    assert!(
        scaler
            .transform([5.1, 3.5, 1.4, 0.2])
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn migrasi_legacy_menyimpan_pusat_lama_dan_model_pulih_setelah_restart() {
    static TEMP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let samples = iris_uci_samples();
    let split = stratified_split(&samples, 47).unwrap();
    let training = split
        .train()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let validation = split
        .validation()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let path = std::env::temp_dir().join(format!(
        "shella-class-knowledge-{}-{}.json",
        std::process::id(),
        TEMP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let legacy_centers = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ];
    let mut memory = LearningMemory::open(&path).unwrap();
    for (class, center) in legacy_centers.iter().enumerate() {
        memory
            .record_prototype("legacy-iris", class, center)
            .unwrap();
    }
    let migrated = memory
        .migrate_legacy_prototypes(
            "legacy-iris",
            &training,
            &validation,
            "iris-uci-v1",
            "migration-run",
            47,
        )
        .unwrap();
    assert_eq!(
        migrated.legacy_snapshot,
        Some(LegacyPrototypeSnapshot::new(legacy_centers).unwrap())
    );
    let before_restart = migrated.predict([6.2, 3.4, 5.4, 2.3]).unwrap();
    drop(memory);

    let restored = LearningMemory::open(&path).unwrap();
    let restored_model = restored
        .load_class_knowledge("migration-run")
        .unwrap()
        .unwrap();
    assert_eq!(restored_model, migrated);
    assert_eq!(
        restored_model.predict([6.2, 3.4, 5.4, 2.3]).unwrap(),
        before_restart
    );
    drop(restored);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}.lock", path.display()));
}

#[test]
fn class_knowledge_mengubah_state_jaringan_lif_dengan_injeksi_prototype() {
    let samples = iris_uci_samples();
    let split = stratified_split(&samples, 73).unwrap();
    let training = split
        .train()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let validation = split
        .validation()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let model =
        ClassKnowledgeModel::fit(&training, &validation, "iris-uci-v1", "lif-run", 73).unwrap();
    let mut network =
        LifNetwork::new(LifConfig::new(1.0, 20.0, 1.0, -65.0, -65.0, -50.0, 1), 7).unwrap();
    let inference = model
        .infer_with_lif(&mut network, [6.2, 3.4, 5.4, 2.3], &[4, 5, 6])
        .unwrap();

    assert_eq!(network.completed_steps(), 4);
    assert!(inference.prediction_from_knowledge < 3);
    assert!(inference.prediction_from_network.is_some());
    assert_eq!(inference.injected_currents_na.len(), network.neuron_count());
    assert!(
        inference
            .injected_currents_na
            .iter()
            .any(|current| *current > 0.0)
    );
    assert_eq!(inference.completed_step, network.completed_steps());
}

#[test]
fn checkpoint_composite_memulihkan_prediction_class_knowledge_deterministik() {
    static TEMP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let samples = iris_uci_samples();
    let split = stratified_split(&samples, 47).unwrap();
    let training = split
        .train()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let validation = split
        .validation()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let knowledge =
        ClassKnowledgeModel::fit(&training, &validation, "iris-uci-v1", "checkpoint-run", 47)
            .unwrap();
    let network =
        LifNetwork::new(LifConfig::new(1.0, 20.0, 1.0, -65.0, -65.0, -50.0, 1), 7).unwrap();
    let path = std::env::temp_dir().join(format!(
        "shella-class-knowledge-checkpoint-{}-{}.json",
        std::process::id(),
        TEMP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    Checkpoint::save_knowledge(&path, &network, &knowledge, &[4, 5, 6], None).unwrap();

    let mut before_network = network.clone();
    let before = knowledge
        .infer_with_lif(&mut before_network, [6.2, 3.4, 5.4, 2.3], &[4, 5, 6])
        .unwrap();
    let restored = Checkpoint::restore_knowledge(&path).unwrap();
    let mut after_network = restored.network;
    let after = restored
        .class_knowledge
        .infer_with_lif(
            &mut after_network,
            [6.2, 3.4, 5.4, 2.3],
            &restored.action_neurons,
        )
        .unwrap();
    assert_eq!(after, before);
    assert_eq!(after_network, before_network);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn feedback_candidate_menaikkan_versi_dan_refit_calibration_validation() {
    let samples = iris_uci_samples();
    let split = stratified_split(&samples, 73).unwrap();
    let training = split
        .train()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let validation = split
        .validation()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let original =
        ClassKnowledgeModel::fit(&training, &validation, "iris-uci-v1", "feedback-run", 73)
            .unwrap();
    let mut candidate = original.clone();
    candidate.apply_validation_feedback(2, 1, 0.1).unwrap();
    candidate.refit_calibration(&validation).unwrap();

    assert_eq!(candidate.model_version, original.model_version + 1);
    candidate.validate().unwrap();
    assert!(
        candidate
            .predict([6.2, 3.4, 5.4, 2.3])
            .unwrap()
            .calibrated_confidence
            .is_some_and(|confidence| (0.0..=1.0).contains(&confidence))
    );
}

#[test]
fn exemplar_policy_membatasi_budget_per_kelas() {
    let samples = iris_uci_samples();
    let split = stratified_split(&samples, 17).unwrap();
    let training = split
        .train()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let validation = split
        .validation()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let model = ClassKnowledgeModel::fit_with_exemplar_policy(
        &training,
        &validation,
        "iris-uci-v1",
        "exemplar-run",
        17,
        ExemplarPolicy::new(4).unwrap(),
    )
    .unwrap();
    assert!(model.classes.iter().all(|class| class.exemplars.len() <= 4));
    assert!(model.classes.iter().all(|class| {
        class
            .exemplars
            .iter()
            .all(|exemplar| exemplar.calibrated_confidence.is_some())
    }));
    assert!(ExemplarPolicy::new(0).is_err());
}

#[test]
fn class_knowledge_menolak_schema_dan_prototype_corrupt() {
    let samples = iris_uci_samples();
    let split = stratified_split(&samples, 31).unwrap();
    let training = split
        .train()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let validation = split
        .validation()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let model =
        ClassKnowledgeModel::fit(&training, &validation, "iris-uci-v1", "corrupt-run", 31).unwrap();

    let mut corrupt_prototype = model.clone();
    corrupt_prototype.classes[0].prototypes[0].feature_vector[0] = f64::NAN;
    assert!(corrupt_prototype.validate().is_err());

    let mut unsupported_schema = model;
    unsupported_schema.schema_version += 1;
    assert!(unsupported_schema.validate().is_err());
}

#[test]
fn secure_store_memulihkan_class_knowledge_terenkripsi() {
    static TEMP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let samples = iris_uci_samples();
    let split = stratified_split(&samples, 101).unwrap();
    let training = split
        .train()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let validation = split
        .validation()
        .iter()
        .map(|index| samples[*index].clone())
        .collect::<Vec<_>>();
    let model =
        ClassKnowledgeModel::fit(&training, &validation, "iris-uci-v1", "secure-run", 101).unwrap();
    let path = std::env::temp_dir().join(format!(
        "shella-secure-class-knowledge-{}-{}.json",
        std::process::id(),
        TEMP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let encryption_key = [0x5a; 32];
    let signing_key = [0x3c; 32];
    let mut secure = SecureMemoryStore::open(&path, encryption_key, signing_key).unwrap();
    model.save_secure(&mut secure).unwrap();
    drop(secure);
    let restored = SecureMemoryStore::open(&path, encryption_key, signing_key).unwrap();
    let restored_model = ClassKnowledgeModel::load_secure(&restored, "secure-run")
        .unwrap()
        .unwrap();
    assert_eq!(restored_model, model);
    assert_eq!(
        restored_model.predict([5.1, 3.5, 1.4, 0.2]).unwrap(),
        model.predict([5.1, 3.5, 1.4, 0.2]).unwrap()
    );
    drop(restored);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}.secure.lock", path.display()));
}
