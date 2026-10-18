use shella::{
    Checkpoint, LifConfig, LifNetwork, LifRuntime, RewardEvaluator, RewardEvaluatorConfig,
    RewardModulatedStdpConfig, SimError, StdpConfig, Synapse,
};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

static TEST_ID: AtomicU64 = AtomicU64::new(0);

fn temp_path(label: &str) -> PathBuf {
    let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("shella-{label}-{}-{id}.json", std::process::id()))
}

fn setup() -> LifNetwork {
    let config = LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 2);
    let mut network = LifNetwork::new(config, 2).expect("jaringan valid");
    network
        .add_synapse(Synapse::new(0, 1, 5.0, 0.0, 20.0, 5).expect("sinaps valid"))
        .expect("sinaps ditambahkan");
    network
        .enable_reward_modulated_stdp(RewardModulatedStdpConfig::new(
            StdpConfig::new(0.5, 0.4, 10.0, 10.0),
            20.0,
            0.1,
        ))
        .expect("reward STDP aktif");
    network
}

#[test]
fn pemulihan_mempertahankan_semua_state_dan_eksekusi_identik() {
    let mut uninterrupted = setup();
    let mut evaluator =
        RewardEvaluator::new(RewardEvaluatorConfig::new(1.0, 0.5)).expect("evaluator sah");
    uninterrupted.step(&[100.0, 0.0]).expect("spike pra");
    let output = uninterrupted.step(&[0.0, 100.0]).expect("spike pasca");
    evaluator
        .evaluate_and_apply(
            Some(0),
            &[output.spiking_neurons().contains(&1)],
            &mut uninterrupted,
        )
        .expect("reward diterapkan");
    assert_eq!(uninterrupted.pending_input_count(), 1);
    assert_eq!(evaluator.metrics().samples(), 1);

    let path = temp_path("roundtrip");
    Checkpoint::save(&path, &uninterrupted, Some(&evaluator)).expect("checkpoint tersimpan");
    let restored = Checkpoint::restore(&path).expect("checkpoint dipulihkan");
    fs::remove_file(&path).expect("file sementara dihapus");

    assert_eq!(restored.network, uninterrupted);
    assert_eq!(restored.evaluator, Some(evaluator.clone()));

    let mut resumed = restored.network;
    let mut resumed_evaluator = restored.evaluator.expect("evaluator dipulihkan");
    for input in [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]] {
        let expected = uninterrupted.step(&input).expect("langkah tanpa interupsi");
        let actual = resumed.step(&input).expect("langkah setelah restore");
        assert_eq!(actual, expected);
        assert_eq!(resumed, uninterrupted);
        let target = expected.spiking_neurons().contains(&1);
        let expected_eval = evaluator
            .evaluate(Some(0), &[target])
            .expect("metrik kontinu");
        let actual_eval = resumed_evaluator
            .evaluate(Some(0), &[target])
            .expect("metrik resumed");
        assert_eq!(actual_eval, expected_eval);
        assert_eq!(resumed_evaluator.metrics(), evaluator.metrics());
    }
}

#[test]
fn checkpoint_mendeteksi_payload_rusak_terpotong_dan_versi_tak_didukung() {
    let network = setup();
    let path = temp_path("corrupt");
    Checkpoint::save(&path, &network, None).expect("checkpoint tersimpan");

    let original = fs::read(&path).expect("checkpoint dibaca");
    let mut damaged: serde_json::Value = serde_json::from_slice(&original).expect("envelope JSON");
    damaged["payload_json"] = serde_json::Value::String("{}".to_owned());
    fs::write(
        &path,
        serde_json::to_vec(&damaged).expect("JSON rusak diserialisasi"),
    )
    .expect("payload diubah");
    assert!(matches!(
        Checkpoint::restore(&path),
        Err(SimError::InvalidCheckpoint { .. })
    ));

    fs::write(&path, b"{\"format_version\":").expect("checkpoint dipotong");
    assert!(matches!(
        Checkpoint::restore(&path),
        Err(SimError::InvalidCheckpoint { .. })
    ));

    fs::write(&path, original).expect("checkpoint dipulihkan untuk tes versi");
    let mut unsupported: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap()).expect("envelope valid");
    unsupported["format_version"] = serde_json::Value::from(999);
    fs::write(&path, serde_json::to_vec(&unsupported).unwrap()).expect("versi diganti");
    assert!(matches!(
        Checkpoint::restore(&path),
        Err(SimError::UnsupportedCheckpointVersion { version: 999 })
    ));
    fs::remove_file(path).expect("file dibersihkan");
}

#[test]
fn checkpoint_menolak_io_tanpa_menyembunyikan_kategori_error() {
    let path = temp_path("missing-directory").join("state.json");
    let error = Checkpoint::restore(&path).expect_err("direktori tidak ada");
    assert!(matches!(
        error,
        SimError::Io {
            operation: "buka checkpoint",
            ..
        }
    ));
}

#[test]
fn runtime_satu_neuron_dapat_disimpan_dan_dilanjutkan() {
    let path = temp_path("runtime");
    let config = LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 2);
    let mut uninterrupted = LifRuntime::new(config).expect("runtime");
    uninterrupted.step(100.0).expect("spike");
    Checkpoint::save_runtime(&path, &uninterrupted).expect("runtime tersimpan");
    let mut resumed = Checkpoint::restore_runtime(&path).expect("runtime pulih");
    assert_eq!(resumed, uninterrupted);
    for current in [0.0, 0.0, 10.0, 10.0] {
        assert_eq!(
            resumed.step(current).expect("resume"),
            uninterrupted.step(current).expect("kontinu")
        );
        assert_eq!(resumed, uninterrupted);
    }
    fs::remove_file(path).expect("fixture dihapus");
}

#[test]
fn checkpoint_child_process_restore() {
    let Ok(path) = std::env::var("SHELLA_CHECKPOINT_RESTORE_TEST") else {
        return;
    };
    let restored = Checkpoint::restore(path).expect("checkpoint pada proses baru");
    let mut network = restored.network;
    let step = network
        .step(&[0.0, 100.0])
        .expect("langkah setelah restart");
    println!("RESTORE_OBSERVATION={step:?}\nRESTORE_NETWORK={network:?}");
}

#[test]
fn checkpoint_dapat_dilanjutkan_dari_proses_baru_dengan_trajectory_identik() {
    let mut saved = setup();
    saved.step(&[100.0, 0.0]).expect("spike pra");
    saved.step(&[0.0, 100.0]).expect("spike pasca");
    let path = temp_path("process-restart");
    Checkpoint::save(&path, &saved, None).expect("checkpoint tersimpan");

    let mut uninterrupted = saved;
    let expected_step = uninterrupted
        .step(&[0.0, 100.0])
        .expect("langkah uninterrupted");
    let expected =
        format!("RESTORE_OBSERVATION={expected_step:?}\nRESTORE_NETWORK={uninterrupted:?}");
    let child = Command::new(std::env::current_exe().expect("test executable"))
        .args(["--exact", "checkpoint_child_process_restore", "--nocapture"])
        .env("SHELLA_CHECKPOINT_RESTORE_TEST", &path)
        .output()
        .expect("proses pemulihan dijalankan");
    assert!(
        child.status.success(),
        "child gagal: {}",
        String::from_utf8_lossy(&child.stderr)
    );
    let stdout = String::from_utf8(child.stdout).expect("stdout UTF-8");
    assert!(
        stdout.contains(&expected),
        "output child tidak cocok: {stdout}"
    );
    fs::remove_file(path).expect("hapus fixture");
}
