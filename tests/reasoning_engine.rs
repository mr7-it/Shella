use serde_json::json;
use shella::{
    CalibrationSample, LearningMemory, LifConfig, LifNetwork, MemoryStore, ReasoningConfig,
    ReasoningEngine, RewardModulatedStdpConfig, StdpConfig, Synapse,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

static ID: AtomicU64 = AtomicU64::new(0);

fn memory_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "shella-reasoning-{}-{}.json",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ))
}

fn cleanup(path: &PathBuf) {
    fs::remove_file(path).expect("database dihapus");
    let mut lock_path = path.as_os_str().to_os_string();
    lock_path.push(".lock");
    fs::remove_file(PathBuf::from(lock_path)).expect("sidecar lock dihapus");
}

fn network(count: usize) -> LifNetwork {
    LifNetwork::new(
        LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 0),
        count,
    )
    .expect("network valid")
}

fn config(decay: f64, minimum_confidence: f64) -> ReasoningConfig {
    ReasoningConfig::new(3, 8, 100.0, decay, minimum_confidence).expect("config valid")
}

fn engine(
    path: &PathBuf,
    network: LifNetwork,
    reasoning_config: ReasoningConfig,
) -> ReasoningEngine {
    ReasoningEngine::new(
        network,
        LearningMemory::open(path).expect("memory terbuka"),
        reasoning_config,
    )
}

#[test]
fn reasoning_melakukan_iterasi_lif_dan_propagasi_spike() {
    let path = memory_path();
    let mut network = network(2);
    network
        .add_synapse(Synapse::new(0, 1, 100.0, 0.0, 100.0, 1).expect("sinaps valid"))
        .expect("sinaps ditambahkan");
    let mut engine = engine(&path, network, config(0.0, 0.0));

    let result = engine
        .reason("context", &[100.0, 0.0], &[1])
        .expect("reason berhasil");
    assert_eq!(result.steps().len(), 3);
    assert!(
        result.steps()[0]
            .network_step()
            .spiking_neurons()
            .contains(&0)
    );
    assert!(
        result.steps()[1]
            .network_step()
            .spiking_neurons()
            .contains(&1)
    );
    assert_eq!(engine.network().completed_steps(), 3);

    drop(engine);
    cleanup(&path);
}

#[test]
fn retrieval_dibatasi_konteks_dan_menyuntikkan_perhatian_terbaru() {
    let path = memory_path();
    {
        let mut memory = LearningMemory::open(&path).expect("memory terbuka");
        for _ in 0..2 {
            memory
                .record_feedback("context-a", &[1.0, 0.0, 0.0, 0.0], Some(0), Some(0))
                .expect("pengalaman disimpan");
        }
        memory
            .record_feedback("context-b", &[1.0, 0.0, 0.0, 0.0], Some(1), Some(1))
            .expect("pengalaman konteks lain disimpan");
    }
    let mut engine = engine(&path, network(4), config(1.0, 0.0));

    let result = engine
        .reason("context-a", &[1.0, 0.0, 0.0, 0.0], &[2, 3])
        .expect("reason berhasil");
    assert_eq!(result.recalled().len(), 2);
    assert!(
        result
            .recalled()
            .iter()
            .all(|item| item.task_id() == "context-a")
    );
    assert_eq!(result.attention().len(), 2);
    assert!(
        result.attention()[1].weight() > result.attention()[0].weight(),
        "record yang lebih baru mendapat perhatian lebih besar"
    );
    assert!(
        (result
            .attention()
            .iter()
            .map(|item| item.weight())
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1e-12
    );
    assert!(result.injected_currents_na()[2] > 0.0);
    assert_eq!(result.selected_action(), Some(0));
    let timestamps = result
        .recalled()
        .iter()
        .map(|memory| memory.created_at_unix_nanos())
        .collect::<Vec<_>>();
    assert!(timestamps[0] < timestamps[1]);

    drop(engine);
    cleanup(&path);
}

#[test]
fn confidence_mengukur_konsistensi_dan_abstain_pada_bukti_bertentangan() {
    let path = memory_path();
    {
        let mut memory = LearningMemory::open(&path).expect("memory terbuka");
        memory
            .record_feedback("context", &[1.0, 0.0, 0.0, 0.0], Some(0), Some(0))
            .expect("pengalaman pertama disimpan");
        memory
            .record_feedback("context", &[1.0, 0.0, 0.0, 0.0], Some(1), Some(1))
            .expect("pengalaman konflik disimpan");
    }
    let mut engine = engine(&path, network(4), config(0.0, 0.6));

    let result = engine
        .reason("context", &[1.0, 0.0, 0.0, 0.0], &[2, 3])
        .expect("reason berhasil");
    assert_eq!(result.selected_action(), None);
    assert!(result.abstained());
    assert!(result.uncertainty() > 0.9);
    assert!(result.confidence() < 0.6);
    assert!(result.evidence_strength() > 0.0);

    drop(engine);
    cleanup(&path);
}

#[test]
fn input_invalid_tidak_memajukan_jaringan_dan_tanpa_evidence_abstain() {
    let path = memory_path();
    let mut engine = engine(&path, network(3), config(0.0, 0.0));
    assert!(
        engine
            .reason("context", &[f64::NAN, 0.0, 0.0], &[1, 2])
            .is_err()
    );
    assert_eq!(engine.network().completed_steps(), 0);
    let result = engine
        .reason("context", &[0.0; 3], &[1, 2])
        .expect("input valid");
    assert!(result.abstained());
    assert_eq!(result.confidence(), 0.0);
    assert_eq!(result.steps().len(), 3);
    drop(engine);
    cleanup(&path);
}

#[test]
fn feedback_menyimpan_pengalaman_dan_memperbarui_readout_jaringan() {
    let path = memory_path();
    let mut network = network(3);
    network
        .add_synapse(Synapse::new(0, 1, 0.0, -10.0, 10.0, 1).expect("sinaps valid"))
        .expect("sinaps ditambahkan");
    let mut engine = engine(&path, network, config(0.0, 0.0));
    let outcome = engine
        .learn_from_feedback("context", &[1.0, 0.0, 0.0], Some(0), Some(1), &[1, 2], 0.5)
        .expect("feedback dipelajari");
    assert!(!outcome.memory_write().prediction_correct());
    assert_eq!(outcome.updated_synapses(), 1);
    assert_eq!(engine.network().synapses()[0].weight_na(), 0.5);
    assert_eq!(engine.memory().len(), 1);

    drop(engine);
    cleanup(&path);
}

#[test]
fn reasoning_deterministik_dengan_jaringan_dan_memori_identik() {
    let path = memory_path();
    let second_path = memory_path();
    {
        let mut memory = LearningMemory::open(&path).expect("memory terbuka");
        memory
            .record_feedback("context", &[1.0, 0.0, 0.0], Some(0), Some(0))
            .expect("pengalaman disimpan");
    }
    {
        let memory = MemoryStore::open(&path).expect("source store");
        let value = memory
            .get("learning/experience/0000000000000001")
            .expect("experience source")
            .value()
            .clone();
        let mut mirror = MemoryStore::open(&second_path).expect("mirror store");
        mirror
            .put(
                "learning/experience/0000000000000001",
                value,
                &["pengalaman".to_owned()],
            )
            .expect("snapshot identik");
    }
    let first_result = {
        let mut first = engine(&path, network(3), config(0.25, 0.0));
        first
            .reason("context", &[1.0, 0.0, 0.0], &[1, 2])
            .expect("first reason")
    };
    let mut second = engine(&second_path, network(3), config(0.25, 0.0));
    let second_result = second
        .reason("context", &[1.0, 0.0, 0.0], &[1, 2])
        .expect("second reason");
    assert_eq!(first_result, second_result);

    drop(second);
    cleanup(&path);
    cleanup(&second_path);
}

#[test]
fn konfigurasi_menolak_parameter_ambigu_dan_non_finite() {
    assert!(ReasoningConfig::new(0, 1, 1.0, 0.0, 0.5).is_err());
    assert!(ReasoningConfig::new(1, 1, f64::NAN, 0.0, 0.5).is_err());
    assert!(ReasoningConfig::new(1, 1, 1.0, -0.1, 0.5).is_err());
    assert!(ReasoningConfig::new(1, 1, 1.0, 0.0, 1.1).is_err());
}

#[test]
fn calibration_fit_di_validation_dan_metrics_dihitung_di_split_terpisah() {
    let path = memory_path();
    let mut engine = engine(&path, network(3), config(0.0, 0.0));
    let validation = [(0.05, false), (0.2, false), (0.8, true), (0.95, true)];
    let evaluation = [(0.1, false), (0.9, true)];
    let metrics = engine
        .fit_confidence_calibration(&validation, &evaluation, 3_000, 0.1)
        .expect("kalibrasi terpisah berhasil");
    assert_eq!(metrics.samples(), 2);
    assert!(metrics.brier_score().expect("brier ada") < 0.1);
    assert!(metrics.mean_absolute_calibration_error().expect("mae ada") < 0.2);
    assert_eq!(engine.calibration_metrics(), Some(metrics));
    let result = engine
        .reason("context", &[0.0; 3], &[1, 2])
        .expect("reason berhasil");
    assert!(result.calibrated_confidence().is_some());
    assert!((0.0..=1.0).contains(&result.calibrated_confidence().unwrap()));

    drop(engine);
    cleanup(&path);
}

#[test]
fn calibration_menolak_split_fit_tanpa_dua_outcome_dan_input_invalid() {
    let path = memory_path();
    let mut engine = engine(&path, network(3), config(0.0, 0.0));
    assert!(
        engine
            .fit_confidence_calibration(&[(0.2, false), (0.4, false)], &[(0.5, true)], 20, 0.1)
            .is_err()
    );
    assert!(
        engine
            .fit_confidence_calibration(&[(0.2, false), (0.8, true)], &[(f64::NAN, true)], 20, 0.1)
            .is_err()
    );
    assert_eq!(engine.calibration_metrics(), None);

    drop(engine);
    cleanup(&path);
}

#[test]
fn timestamp_pengalaman_bertahan_setelah_restart() {
    let path = memory_path();
    let timestamp;
    {
        let mut memory = LearningMemory::open(&path).expect("memory terbuka");
        memory
            .record_feedback("context", &[1.0, 0.0], Some(0), Some(0))
            .expect("pengalaman direkam");
        timestamp = memory
            .recall_similar(&[1.0, 0.0], 1)
            .expect("retrieval")
            .remove(0)
            .created_at_unix_nanos();
        assert_ne!(timestamp, 0);
    }
    let memory = LearningMemory::open(&path).expect("restart memory");
    let restored = memory
        .recall_similar(&[1.0, 0.0], 1)
        .expect("retrieval setelah restart");
    assert_eq!(restored[0].created_at_unix_nanos(), timestamp);
    drop(memory);
    cleanup(&path);
}

#[test]
fn timestamp_invalid_ditolak_saat_membuka_learning_memory() {
    let path = memory_path();
    {
        let mut store = MemoryStore::open(&path).expect("store dibuka");
        store
            .put(
                "learning/experience/0000000000000001",
                json!({
                    "version": 1,
                    "task_id": "context",
                    "features": [1.0, 0.0],
                    "expected_action": 0,
                    "feedback": 1.0,
                    "importance": 1.0,
                    "created_at_unix_nanos": u64::MAX
                }),
                &[],
            )
            .expect("record invalid ditulis untuk uji validasi");
    }
    assert!(LearningMemory::open(&path).is_err());
    cleanup(&path);
}

#[test]
fn experience_legacy_dimigrasikan_dan_timestamp_migrasi_dipersistenkan() {
    let path = memory_path();
    {
        let mut store = MemoryStore::open(&path).expect("store dibuka");
        store
            .put(
                "learning/experience/0000000000000001",
                json!({
                    "version": 1,
                    "task_id": "context",
                    "features": [1.0, 0.0],
                    "expected_action": 0,
                    "feedback": 1.0,
                    "importance": 1.0
                }),
                &["pengalaman".to_owned()],
            )
            .expect("experience legacy ditulis");
    }
    let memory = LearningMemory::open(&path).expect("legacy dimigrasikan");
    let timestamp = memory
        .recall_similar(&[1.0, 0.0], 1)
        .expect("legacy diretrieval")[0]
        .created_at_unix_nanos();
    assert_ne!(timestamp, 0);
    drop(memory);
    let restored = LearningMemory::open(&path).expect("hasil migrasi tersimpan");
    assert_eq!(
        restored.recall_similar(&[1.0, 0.0], 1).expect("retrieval")[0].created_at_unix_nanos(),
        timestamp
    );
    drop(restored);
    cleanup(&path);
}

#[test]
fn feedback_melatih_eligibility_reward_modulated_stdp_aktual() {
    let path = memory_path();
    let mut network = network(4);
    network
        .add_synapse(Synapse::new(0, 2, 0.0, -10.0, 10.0, 1).expect("sinaps valid"))
        .expect("sinaps ditambahkan");
    network
        .enable_reward_modulated_stdp(RewardModulatedStdpConfig::new(
            StdpConfig::new(1.0, 1.0, 10.0, 10.0),
            10.0,
            1.0,
        ))
        .expect("reward STDP aktif");
    let mut engine = engine(&path, network, config(0.0, 0.0));

    let outcome = engine
        .learn_from_feedback(
            "context",
            &[100.0, 0.0, 0.0, 0.0],
            Some(0),
            Some(0),
            &[2, 3],
            0.1,
        )
        .expect("feedback STDP berhasil");
    assert_eq!(outcome.updated_synapses(), 1);
    assert!(engine.network().synapses()[0].weight_na() > 0.0);
    assert_eq!(engine.network().completed_steps(), 2);

    drop(engine);
    cleanup(&path);
}

#[test]
fn feedback_salah_menghasilkan_depresi_reward_modulated_stdp() {
    let path = memory_path();
    let mut network = network(4);
    network
        .add_synapse(Synapse::new(0, 2, 0.0, -10.0, 10.0, 1).expect("sinaps valid"))
        .expect("sinaps ditambahkan");
    network
        .enable_reward_modulated_stdp(RewardModulatedStdpConfig::new(
            StdpConfig::new(1.0, 1.0, 10.0, 10.0),
            10.0,
            1.0,
        ))
        .expect("reward STDP aktif");
    let mut engine = engine(&path, network, config(0.0, 0.0));
    engine
        .learn_from_feedback(
            "context",
            &[100.0, 0.0, 0.0, 0.0],
            Some(0),
            Some(1),
            &[2, 3],
            0.1,
        )
        .expect("feedback salah diproses");
    assert!(engine.network().synapses()[0].weight_na() < 0.0);

    drop(engine);
    cleanup(&path);
}

#[test]
fn kalibrasi_menolak_overlap_id_dan_id_duplikat() {
    let path = memory_path();
    let mut engine = engine(&path, network(2), config(0.0, 0.0));
    let validation = [
        CalibrationSample::new("same", 0.1, false),
        CalibrationSample::new("valid", 0.9, true),
    ];
    let overlapping = [CalibrationSample::new("same", 0.8, true)];
    assert!(
        engine
            .fit_confidence_calibration_with_ids(&validation, &overlapping, 10, 0.1)
            .is_err()
    );
    let duplicate_validation = [
        CalibrationSample::new("duplicate", 0.1, false),
        CalibrationSample::new("duplicate", 0.9, true),
    ];
    assert!(
        engine
            .fit_confidence_calibration_with_ids(
                &duplicate_validation,
                &[CalibrationSample::new("evaluation", 0.8, true)],
                10,
                0.1
            )
            .is_err()
    );
    assert_eq!(engine.calibration_state(), None);
    drop(engine);
    cleanup(&path);
}

#[test]
fn reasoning_checkpoint_memulihkan_kalibrator_config_dan_context() {
    let path = memory_path();
    let checkpoint_path = memory_path();
    let reasoning_config = config(0.25, 0.0)
        .with_context_window(2)
        .expect("kapasitas konteks valid");
    let mut engine = engine(&path, network(3), reasoning_config);
    engine
        .fit_confidence_calibration_with_ids(
            &[
                CalibrationSample::new("v-negative", 0.1, false),
                CalibrationSample::new("v-positive", 0.9, true),
            ],
            &[
                CalibrationSample::new("e-negative", 0.2, false),
                CalibrationSample::new("e-positive", 0.8, true),
            ],
            100,
            0.1,
        )
        .expect("kalibrasi berhasil");
    engine
        .reason("task", &[1.0, 0.0, 0.0], &[1, 2])
        .expect("reason mengisi context");
    let calibration_state = engine.calibration_state().cloned();
    let frames = engine.context_window().cloned().collect::<Vec<_>>();
    engine
        .save_checkpoint(&checkpoint_path)
        .expect("checkpoint tersimpan");
    drop(engine);

    let memory = LearningMemory::open(&path).expect("memory dibuka ulang");
    let restored =
        ReasoningEngine::restore_checkpoint(&checkpoint_path, memory).expect("checkpoint pulih");
    assert_eq!(restored.calibration_state(), calibration_state.as_ref());
    assert_eq!(
        restored.context_window().cloned().collect::<Vec<_>>(),
        frames
    );
    assert_eq!(restored.network().completed_steps(), 3);
    assert_eq!(restored.config().context_window_capacity(), 2);
    assert_eq!(
        restored
            .calibration_state()
            .expect("calibration restored")
            .validation_sample_ids(),
        ["v-negative", "v-positive"]
    );
    drop(restored);
    {
        let mut store = MemoryStore::open(&path).expect("memory dapat dibuka");
        store
            .put("later-write", json!({"not_in_checkpoint":true}), &[])
            .expect("mutasi setelah checkpoint");
    }
    let changed_memory = LearningMemory::open(&path).expect("memory setelah mutasi");
    assert!(
        ReasoningEngine::restore_checkpoint(&checkpoint_path, changed_memory).is_err(),
        "restore harus menolak memory yang tidak sesuai fingerprint checkpoint"
    );
    fs::remove_file(checkpoint_path).expect("checkpoint dihapus");
    cleanup(&path);
}

#[test]
fn context_window_membawa_konteks_beberapa_langkah_dan_diisolasi_per_context() {
    let path = memory_path();
    let reasoning_config = config(0.0, 0.0)
        .with_context_window(2)
        .expect("kapasitas valid");
    let mut engine = engine(&path, network(2), reasoning_config);
    engine
        .reason("task-a", &[100.0, 0.0], &[1])
        .expect("reason pertama");
    let second = engine
        .reason("task-a", &[0.0, 0.0], &[1])
        .expect("reason kedua");
    assert_eq!(engine.context_window().count(), 2);
    assert!(second.steps()[0].currents_na()[0] > 0.0);
    let switched = engine
        .reason("task-b", &[0.0, 0.0], &[1])
        .expect("context baru");
    assert_eq!(engine.context_window().count(), 1);
    assert_eq!(
        engine.context_window().next().unwrap().context_id(),
        "task-b"
    );
    assert_eq!(switched.steps()[0].currents_na()[0], 0.0);
    drop(engine);
    cleanup(&path);
}

#[test]
fn consolidation_menggabungkan_duplikat_dengan_label_sama_dan_menjaga_konflik() {
    let path = memory_path();
    let mut memory = LearningMemory::open(&path).expect("memory");
    memory
        .record_feedback("task", &[1.0, 0.0], Some(0), Some(0))
        .expect("record pertama");
    memory
        .record_feedback("task", &[0.99, 0.01], Some(0), Some(0))
        .expect("record duplikat");
    memory
        .record_feedback("task", &[1.0, 0.0], Some(1), Some(1))
        .expect("record konflik");
    let report = memory
        .consolidate_similar("task", 0.99, 0.1)
        .expect("konsolidasi berhasil");
    assert_eq!(report.merged_records(), 1);
    assert_eq!(report.retained_record_ids().len(), 1);
    assert_eq!(memory.len(), 2);
    let consolidated = memory.recall_similar(&[1.0, 0.0], 2).expect("retrieval");
    assert!(consolidated.iter().any(|item| item.evidence_count() == 2));
    drop(memory);
    let restored = LearningMemory::open(&path).expect("hasil persist");
    assert_eq!(restored.len(), 2);
    drop(restored);
    cleanup(&path);
}

#[test]
fn time_decay_persisten_dan_usage_protection_mencegah_penghapusan_bukti_terpakai() {
    let path = memory_path();
    {
        let mut memory = LearningMemory::open(&path).expect("memory");
        memory
            .record_feedback("task", &[1.0, 0.0], Some(0), Some(0))
            .expect("bukti tak terpakai");
        memory
            .record_feedback("task", &[0.0, 1.0], Some(1), Some(1))
            .expect("bukti terpakai");
    }
    {
        let mut store = MemoryStore::open(&path).expect("store dibuka");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("waktu setelah epoch")
            .as_nanos() as u64;
        for key in [
            "learning/experience/0000000000000001",
            "learning/experience/0000000000000002",
        ] {
            let mut value = store.get(key).expect("record ada").value().clone();
            let object = value.as_object_mut().expect("object json");
            object.insert(
                "created_at_unix_nanos".to_owned(),
                (now - 10_000_000_000).into(),
            );
            object.insert(
                "last_decay_unix_nanos".to_owned(),
                (now - 10_000_000_000).into(),
            );
            if key.ends_with('2') {
                object.insert("usage_count".to_owned(), 1_000_u64.into());
                object.insert("last_used_unix_nanos".to_owned(), (now - 1_000_000).into());
            }
            store.put(key, value, &[]).expect("timestamp uji disimpan");
        }
    }
    let mut memory = LearningMemory::open(&path).expect("memory dengan umur dibuka");
    let report = memory
        .decay_by_time(10.0, 10.0, 0.75)
        .expect("time decay berhasil");
    assert_eq!(report.examined(), 2);
    assert_eq!(report.decayed(), 1);
    assert_eq!(report.deleted(), 1);
    assert_eq!(memory.len(), 1);
    drop(memory);
    let restored = LearningMemory::open(&path).expect("hasil decay persisted");
    assert_eq!(restored.len(), 1);
    assert_eq!(
        restored
            .recall_similar(&[0.0, 1.0], 1)
            .expect("bukti penggunaan bertahan")[0]
            .usage_count(),
        1_000
    );
    drop(restored);
    cleanup(&path);
}

#[test]
fn asosiasi_label_menghitung_bukti_pendukung_dan_bertentangan() {
    let path = memory_path();
    let mut memory = LearningMemory::open(&path).expect("memory");
    for _ in 0..2 {
        memory
            .record_feedback("task", &[1.0, 0.0], Some(0), Some(0))
            .expect("pengalaman support");
    }
    memory
        .record_feedback("task", &[1.0, 0.0], Some(1), Some(1))
        .expect("pengalaman kontradiksi");
    let associations = memory
        .causal_associations("task", 2)
        .expect("asosiasi terhitung");
    assert_eq!(associations.len(), 1);
    assert_eq!(associations[0].feature_index(), 0);
    assert_eq!(associations[0].action(), 0);
    assert_eq!(associations[0].supporting_experiences(), 2);
    assert_eq!(associations[0].contradicting_experiences(), 1);
    drop(memory);
    cleanup(&path);
}

#[test]
fn contradiction_dan_uncertainty_didekomposisi_dari_evidence() {
    let path = memory_path();
    {
        let mut memory = LearningMemory::open(&path).expect("memory");
        memory
            .record_feedback("task", &[1.0, 0.0, 0.0], Some(0), Some(0))
            .expect("action 0");
        memory
            .record_feedback("task", &[1.0, 0.0, 0.0], Some(1), Some(1))
            .expect("action 1");
    }
    let mut engine = engine(&path, network(3), config(0.0, 0.0));
    let result = engine
        .reason("task", &[1.0, 0.0, 0.0], &[1, 2])
        .expect("reason");
    assert_eq!(result.contradictions().len(), 1);
    let decomposition = result.uncertainty_decomposition();
    assert_eq!(decomposition.prediction(), result.uncertainty());
    assert!((0.0..=1.0).contains(&decomposition.evidence_gap()));
    drop(engine);
    cleanup(&path);
}

#[test]
fn adaptive_compute_memakai_budget_terukur_dan_terikat() {
    let path = memory_path();
    let reasoning_config = config(0.0, 0.0).with_adaptive_compute(true);
    let mut engine = engine(&path, network(4), reasoning_config);
    let sparse = engine
        .reason("task", &[1.0, 0.0, 0.0, 0.0], &[2, 3])
        .expect("reason sparse");
    assert_eq!(sparse.compute_budget(), 2);
    assert_eq!(sparse.steps().len(), sparse.compute_budget());
    let dense = engine
        .reason("task", &[1.0, 1.0, 1.0, 1.0], &[2, 3])
        .expect("reason dense");
    assert_eq!(dense.compute_budget(), 3);
    assert!(dense.compute_budget() <= engine.config().iterations());
    drop(engine);
    cleanup(&path);
}

#[test]
fn goal_directed_retrieval_memprioritaskan_fitur_tujuan() {
    let path = memory_path();
    {
        let mut memory = LearningMemory::open(&path).expect("memory");
        memory
            .record_feedback("task", &[1.0, 0.0, 0.0], Some(0), Some(0))
            .expect("experience input");
        memory
            .record_feedback("task", &[0.0, 1.0, 0.0], Some(1), Some(1))
            .expect("experience goal");
    }
    let mut engine = engine(&path, network(3), config(0.0, 0.0));
    let result = engine
        .reason_for_goal("task", &[0.0, 1.0, 0.0], &[0.0; 3], &[1, 2])
        .expect("goal retrieval");
    assert_eq!(result.recalled()[0].expected_action(), Some(1));
    assert_eq!(result.selected_action(), Some(1));
    drop(engine);
    cleanup(&path);
}

#[test]
fn engine_experience_replay_memakai_reward_modulated_stdp() {
    let path = memory_path();
    let mut memory = LearningMemory::open(&path).expect("memory");
    memory
        .record_feedback("task", &[100.0, 0.0, 0.0, 0.0], Some(0), Some(1))
        .expect("pengalaman salah");
    let mut network = network(4);
    network
        .add_synapse(Synapse::new(0, 2, 0.0, -10.0, 10.0, 1).expect("sinaps"))
        .expect("sinaps ditambah");
    network
        .enable_reward_modulated_stdp(RewardModulatedStdpConfig::new(
            StdpConfig::new(1.0, 1.0, 10.0, 10.0),
            10.0,
            1.0,
        ))
        .expect("STDP aktif");
    let mut engine = ReasoningEngine::new(network, memory, config(0.0, 0.0));
    let replay = engine
        .replay_experiences("task", &[100.0, 0.0, 0.0, 0.0], &[2], 1, 100.0)
        .expect("replay engine");
    assert_eq!(replay.replayed_record_ids().len(), 1);
    assert_eq!(replay.error_records_replayed(), 1);
    assert_ne!(engine.network().synapses()[0].weight_na(), 0.0);
    drop(engine);
    cleanup(&path);
}

#[test]
fn reasoning_mencatat_penggunaan_memori_yang_mempengaruhi_retrieval() {
    let path = memory_path();
    {
        let mut memory = LearningMemory::open(&path).expect("memory");
        memory
            .record_feedback("task", &[1.0, 0.0, 0.0], Some(0), Some(0))
            .expect("experience");
    }
    {
        let mut engine = engine(&path, network(3), config(0.0, 0.0));
        engine
            .reason("task", &[1.0, 0.0, 0.0], &[1, 2])
            .expect("retrieval pertama");
    }
    let memory = LearningMemory::open(&path).expect("restart");
    let recall = memory
        .recall_similar(&[1.0, 0.0, 0.0], 1)
        .expect("retrieval sesudah penggunaan");
    assert_eq!(recall[0].usage_count(), 1);
    assert!(recall[0].retrieval_score() > recall[0].similarity() * 2.0);
    drop(memory);
    cleanup(&path);
}

#[test]
fn standard_stdp_mengubah_bobot_dengan_spike_pengajar_tanpa_reward() {
    let path = memory_path();
    let mut network = network(4);
    network
        .add_synapse(Synapse::new(0, 2, 0.0, -10.0, 10.0, 1).expect("sinaps"))
        .expect("sinaps ditambah");
    network
        .enable_stdp(StdpConfig::new(1.0, 1.0, 10.0, 10.0))
        .expect("STDP aktif");
    let mut engine = engine(&path, network, config(0.0, 0.0));
    let outcome = engine
        .learn_from_feedback(
            "task",
            &[100.0, 0.0, 0.0, 0.0],
            Some(0),
            Some(0),
            &[2, 3],
            0.1,
        )
        .expect("training STDP");
    assert_eq!(outcome.updated_synapses(), 1);
    assert!(engine.network().synapses()[0].weight_na() > 0.0);
    assert_eq!(engine.network().reward_modulated_stdp_config(), None);
    drop(engine);
    cleanup(&path);
}

#[test]
fn attention_temporal_menggunakan_umur_absolut_persisten() {
    let path = memory_path();
    {
        let mut memory = LearningMemory::open(&path).expect("memory");
        memory
            .record_feedback("task", &[1.0, 0.0, 0.0], Some(0), Some(0))
            .expect("record lama");
        memory
            .record_feedback("task", &[1.0, 0.0, 0.0], Some(0), Some(0))
            .expect("record baru");
    }
    {
        let mut store = MemoryStore::open(&path).expect("store");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("waktu unix")
            .as_nanos() as u64;
        let times = [now - 8_000_000_000, now - 6_000_000_000];
        for (index, timestamp) in times.iter().enumerate() {
            let key = format!("learning/experience/{:016x}", index + 1);
            let mut value = store.get(&key).expect("record").value().clone();
            let object = value.as_object_mut().expect("json object");
            object.insert("created_at_unix_nanos".to_owned(), (*timestamp).into());
            object.insert("last_decay_unix_nanos".to_owned(), (*timestamp).into());
            store.put(key, value, &[]).expect("umur persist");
        }
    }
    let mut engine = engine(&path, network(3), config(0.5, 0.0));
    let result = engine
        .reason("task", &[1.0, 0.0, 0.0], &[1, 2])
        .expect("reason");
    assert_eq!(result.recalled().len(), 2);
    assert!(result.attention()[0].weight() < result.attention()[1].weight());
    assert!(result.attention()[0].weight() > 0.0);
    drop(engine);
    cleanup(&path);
}

#[test]
fn context_window_capacity_memiliki_batas_sumber_daya() {
    assert!(config(0.0, 0.0).with_context_window(4097).is_err());
}
