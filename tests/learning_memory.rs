use shella::{
    LearningMemory, LifConfig, LifNetwork, RewardModulatedStdpConfig, StdpConfig, Synapse,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static ID: AtomicU64 = AtomicU64::new(0);

fn memory_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "shella-learning-memory-{}-{}.json",
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

fn network(neuron_count: usize) -> LifNetwork {
    let config = LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1);
    LifNetwork::new(config, neuron_count).expect("jaringan valid")
}

#[test]
fn memori_persisten_pulih_setelah_restart_dan_retrieval_mengutamakan_kesalahan_relevan() {
    let path = memory_path();
    let error_id;
    {
        let mut memory = LearningMemory::open(&path).expect("store dibuka");
        let error = memory
            .record_feedback("task-a", &[1.0, 0.0, 0.0], Some(0), Some(1))
            .expect("feedback salah disimpan");
        assert!(!error.prediction_correct());
        error_id = error.record_id().to_owned();
        let conflict = memory
            .record_feedback("task-b", &[1.0, 0.0, 0.0], Some(1), Some(1))
            .expect("asosiasi berlawanan disimpan");
        assert_eq!(conflict.conflicts().len(), 1);
        assert_eq!(conflict.conflicts()[0].previous_action(), Some(0));
        assert_eq!(conflict.conflicts()[0].new_action(), Some(1));
        memory
            .record_feedback("task-a", &[0.0, 1.0, 0.0], Some(0), Some(0))
            .expect("pengalaman jauh disimpan");
        assert_eq!(memory.len(), 3);
    }

    let memory = LearningMemory::open(&path).expect("store pulih setelah restart");
    let recalled = memory
        .recall_similar(&[0.9, 0.1, 0.0], 3)
        .expect("retrieval similarity");
    assert_eq!(recalled.len(), 3);
    assert_eq!(recalled[0].record_id(), error_id);
    assert_eq!(recalled[0].expected_action(), Some(0));
    assert!(recalled[0].similarity() > recalled[2].similarity());
    drop(memory);
    cleanup(&path);
}

#[test]
fn retrieval_menginjeksi_arus_ke_neuron_lif_dan_mengubah_inferensi() {
    let path = memory_path();
    let mut memory = LearningMemory::open(&path).expect("store dibuka");
    memory
        .record_feedback("cue", &[0.0; 4], Some(0), Some(0))
        .expect("pengalaman disimpan");

    let mut without_recall = network(4);
    let plain = without_recall
        .step(&[0.0; 4])
        .expect("inferensi tanpa retrieval");
    assert!(!plain.spiking_neurons().contains(&2));

    let mut with_recall = network(4);
    let inference = memory
        .infer_with_memory(&mut with_recall, &[0.0; 4], &[2, 3], 4, 100.0)
        .expect("inferensi dengan memory bias");
    assert!(inference.injected_currents_na()[2] > 90.0);
    assert!(inference.spiking_neurons().contains(&2));
    assert!(!inference.spiking_neurons().contains(&3));

    drop(memory);
    cleanup(&path);
}

#[test]
fn retrieval_task_filter_memisahkan_association_yang_bertentangan() {
    let path = memory_path();
    let mut memory = LearningMemory::open(&path).expect("store dibuka");
    let features = [0.0, 0.0, 0.0, 0.0];
    memory
        .record_feedback("task-a", &features, Some(0), Some(0))
        .expect("asosiasi A disimpan");
    memory
        .record_feedback("task-b", &features, Some(1), Some(1))
        .expect("asosiasi B disimpan");

    assert_eq!(
        memory
            .find_conflicts(&features, Some(1), 0.95)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        memory
            .recall_task_similar("task-a", &features, 4)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        memory
            .recall_task_similar("task-b", &features, 4)
            .unwrap()
            .len(),
        1
    );

    let mut task_a_network = network(4);
    let task_a = memory
        .infer_task_with_memory("task-a", &mut task_a_network, &features, &[2, 3], 4, 100.0)
        .expect("inferensi task A");
    assert!(task_a.spiking_neurons().contains(&2));
    assert!(!task_a.spiking_neurons().contains(&3));

    let mut task_b_network = network(4);
    let task_b = memory
        .infer_task_with_memory("task-b", &mut task_b_network, &features, &[2, 3], 4, 100.0)
        .expect("inferensi task B");
    assert!(task_b.spiking_neurons().contains(&3));
    assert!(!task_b.spiking_neurons().contains(&2));

    drop(memory);
    cleanup(&path);
}

#[test]
fn replay_pengalaman_error_mengubah_bobot_stdp_network_aktual() {
    let path = memory_path();
    let mut memory = LearningMemory::open(&path).expect("store dibuka");
    memory
        .record_feedback("cue", &[100.0, 0.0, 0.0], Some(0), Some(1))
        .expect("pengalaman salah disimpan");

    let mut network = network(3);
    network
        .add_synapse(Synapse::new(0, 2, 0.0, -100.0, 100.0, 1).expect("sinaps valid"))
        .expect("sinaps ditambahkan");
    network
        .enable_reward_modulated_stdp(RewardModulatedStdpConfig::new(
            StdpConfig::new(1.0, 1.0, 10.0, 10.0),
            20.0,
            1.0,
        ))
        .expect("STDP reward aktif");
    let initial_weight = network.synapses()[0].weight_na();
    let report = memory
        .replay_relevant(&mut network, &[100.0, 0.0, 0.0], &[2], 1, 100.0)
        .expect("replay memori");
    assert_eq!(report.replayed_record_ids().len(), 1);
    assert_eq!(report.error_records_replayed(), 1);
    assert!(report.weight_updates() > 0);
    assert_ne!(network.synapses()[0].weight_na(), initial_weight);

    drop(memory);
    cleanup(&path);
}

#[test]
fn pelupaan_decay_menghapus_yang_irrelevan_dan_melindungi_memori_terkonsolidasi() {
    let path = memory_path();
    let mut memory = LearningMemory::open(&path).expect("store dibuka");
    let important = memory
        .record_feedback("stable", &[1.0, 0.0], Some(0), Some(0))
        .expect("pengalaman stabil");
    memory
        .consolidate(important.record_id(), 4.0)
        .expect("pengalaman dikonsolidasi");
    memory
        .record_feedback("ephemeral", &[0.0, 1.0], Some(1), Some(1))
        .expect("pengalaman sementara");

    let report = memory
        .decay_irrelevant(&[1.0, 0.0], 0.8, 0.1, 0.2, 2.0)
        .expect("pelupaan terkontrol");
    assert_eq!(report.examined(), 2);
    assert_eq!(report.protected(), 1);
    assert_eq!(report.deleted(), 1);
    assert_eq!(memory.len(), 1);
    let protected = memory
        .recall_similar(&[1.0, 0.0], 1)
        .expect("memori penting tetap ada");
    assert_eq!(protected[0].record_id(), important.record_id());
    assert!(protected[0].importance() >= 2.0);

    drop(memory);
    cleanup(&path);
}

#[test]
fn simpanan_memori_menolak_fitur_non_finite_dan_feedback_invalid() {
    let path = memory_path();
    let mut memory = LearningMemory::open(&path).expect("store dibuka");
    assert!(
        memory
            .record_feedback("task", &[f64::NAN], Some(0), Some(0))
            .is_err()
    );
    assert!(
        memory
            .record_with_feedback("task", &[1.0], Some(0), 2.0, 1.0)
            .is_err()
    );
    assert_eq!(memory.len(), 0);
    drop(memory);
    cleanup(&path);
}
