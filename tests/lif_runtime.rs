use shella::{LifConfig, LifRuntime, SimError};

fn config() -> LifConfig {
    LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 2)
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "nilai aktual {actual:.16e}, diharapkan {expected:.16e}, toleransi {tolerance:.3e}"
    );
}

#[test]
fn potensial_istirahat_stabil_tanpa_arus() {
    let mut runtime = LifRuntime::new(config()).expect("konfigurasi uji sah");
    for _ in 0..1_000 {
        let result = runtime.step(0.0).expect("langkah tanpa arus berhasil");
        close(result.membrane_potential_mv, -65.0, 0.0);
        assert!(!result.spiked);
    }
}

#[test]
fn peluruhan_tegangan_mengikuti_solusi_eksponensial() {
    let cfg = config();
    let mut runtime =
        LifRuntime::with_initial_potential(cfg, -62.0).expect("keadaan awal di bawah ambang sah");
    let mut expected = cfg.rest_potential_mv() + 3.0;
    for _ in 1..=10 {
        expected = cfg.rest_potential_mv()
            + (expected - cfg.rest_potential_mv()) * (-cfg.dt_ms() / cfg.tau_m_ms()).exp();
        let result = runtime.step(0.0).expect("langkah tanpa arus berhasil");
        close(result.membrane_potential_mv, expected, 1.0e-12);
    }
}

#[test]
fn trajektori_arus_konstan_cocok_dengan_solusi_analitis() {
    let cfg = config();
    let current = 4.0;
    let equilibrium = cfg.rest_potential_mv() + cfg.resistance_mohm() * current;
    let mut runtime = LifRuntime::new(cfg).expect("konfigurasi uji sah");
    for step in 1..=20 {
        let result = runtime
            .step(current)
            .expect("langkah arus konstan berhasil");
        let expected = equilibrium
            + (cfg.rest_potential_mv() - equilibrium)
                * (-(step as f64) * cfg.dt_ms() / cfg.tau_m_ms()).exp();
        close(result.membrane_potential_mv, expected, 2.0e-12);
        assert!(!result.spiked);
    }
}

#[test]
fn arus_subambang_tidak_menghasilkan_spike() {
    let mut runtime = LifRuntime::new(config()).expect("konfigurasi uji sah");
    for _ in 0..500 {
        assert!(!runtime.step(4.0).expect("langkah berhasil").spiked);
    }
    assert_eq!(runtime.neuron().spike_count(), 0);
}

#[test]
fn waktu_spike_ditetapkan_oleh_persamaan_diskret() {
    let mut runtime = LifRuntime::new(config()).expect("konfigurasi uji sah");
    let mut spikes = Vec::new();
    for _ in 0..50 {
        let outcome = runtime.step(10.0).expect("langkah berhasil");
        if outcome.spiked {
            spikes.push(outcome.completed_step);
        }
    }
    assert_eq!(spikes, vec![7, 20, 33, 46]);
}

#[test]
fn spike_mereset_potensial_ke_nilai_konfigurasi() {
    let mut runtime = LifRuntime::new(config()).expect("konfigurasi uji sah");
    let mut spike = None;
    for _ in 0..10 {
        let outcome = runtime.step(10.0).expect("langkah berhasil");
        if outcome.spiked {
            spike = Some(outcome);
            break;
        }
    }
    let spike = spike.expect("arus suprambang menghasilkan spike");
    assert_eq!(spike.membrane_potential_mv, config().reset_potential_mv());
    assert_eq!(runtime.neuron().spike_steps(), &[7]);
}

#[test]
fn periode_refraktori_menahan_integrasi_selama_jumlah_langkah_tepat() {
    let mut runtime = LifRuntime::new(config()).expect("konfigurasi uji sah");
    for _ in 0..6 {
        runtime.step(10.0).expect("langkah berhasil");
    }
    assert!(runtime.step(10.0).expect("langkah spike").spiked);
    for expected_remaining in [1, 0] {
        let result = runtime.step(10.0).expect("langkah refraktori berhasil");
        assert!(!result.spiked);
        assert_eq!(result.membrane_potential_mv, config().reset_potential_mv());
        assert_eq!(runtime.neuron().refractory_remaining(), expected_remaining);
    }
    let resumed = runtime
        .step(10.0)
        .expect("integrasi dilanjutkan setelah refraktori");
    assert!(!resumed.spiked);
    assert!(resumed.membrane_potential_mv > config().reset_potential_mv());
}

#[test]
fn penghitung_langkah_dan_spike_konsisten() {
    let mut runtime = LifRuntime::new(config()).expect("konfigurasi uji sah");
    let outcomes = runtime.run(&[10.0; 50]).expect("run berhasil");
    assert_eq!(runtime.neuron().completed_steps(), 50);
    assert_eq!(runtime.neuron().spike_count(), 4);
    assert_eq!(outcomes.iter().filter(|outcome| outcome.spiked).count(), 4);
    assert_eq!(runtime.neuron().spike_steps(), &[7, 20, 33, 46]);
}

#[test]
fn runtime_terpisah_dan_run_berulang_memberi_hasil_yang_sama() {
    let inputs = [10.0, 0.0, 10.0, 4.0, 10.0, 0.0, 0.0, 10.0];
    let mut whole = LifRuntime::new(config()).expect("konfigurasi uji sah");
    let whole_results = whole.run(&inputs).expect("run utuh berhasil");

    let mut split = LifRuntime::new(config()).expect("konfigurasi uji sah");
    let first = split.run(&inputs[..3]).expect("bagian pertama berhasil");
    let second = split.run(&inputs[3..]).expect("bagian kedua berhasil");
    assert_eq!(whole_results, [first, second].concat());
    assert_eq!(whole.neuron(), split.neuron());
}

#[test]
fn urutan_spike_dapat_diulang_secara_deterministik() {
    let input = [10.0; 100];
    let mut first = LifRuntime::new(config()).expect("konfigurasi uji sah");
    let mut second = LifRuntime::new(config()).expect("konfigurasi uji sah");
    assert_eq!(
        first.run(&input).expect("run pertama berhasil"),
        second.run(&input).expect("run kedua berhasil")
    );
    assert_eq!(first.neuron().spike_steps(), second.neuron().spike_steps());
}

#[test]
fn konfigurasi_menolak_dt_tau_resistansi_dan_relasi_ambang_tidak_sah() {
    let invalid = [
        LifConfig::new(0.0, 10.0, 1.0, -65.0, -70.0, -50.0, 0),
        LifConfig::new(1.0, f64::NAN, 1.0, -65.0, -70.0, -50.0, 0),
        LifConfig::new(1.0, 10.0, 0.0, -65.0, -70.0, -50.0, 0),
        LifConfig::new(1.0, 10.0, 1.0, -50.0, -70.0, -50.0, 0),
        LifConfig::new(1.0, 10.0, 1.0, -65.0, -50.0, -50.0, 0),
    ];
    for bad in invalid {
        assert!(matches!(
            bad.validate(),
            Err(SimError::InvalidConfiguration { .. })
        ));
        assert!(LifRuntime::new(bad).is_err());
    }
}

#[test]
fn konfigurasi_menolak_parameter_tegangan_non_finite() {
    for bad in [
        LifConfig::new(1.0, 10.0, 1.0, f64::INFINITY, -70.0, 0.0, 0),
        LifConfig::new(1.0, 10.0, 1.0, -65.0, f64::NEG_INFINITY, -50.0, 0),
        LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, f64::NAN, 0),
    ] {
        assert!(matches!(
            bad.validate(),
            Err(SimError::InvalidConfiguration { .. })
        ));
    }
}

#[test]
fn masukan_nan_ditolak_tanpa_mengubah_keadaan() {
    let mut runtime = LifRuntime::new(config()).expect("konfigurasi uji sah");
    runtime.step(1.0).expect("langkah awal berhasil");
    let before = runtime.neuron().clone();
    assert!(matches!(
        runtime.step(f64::NAN),
        Err(SimError::InvalidInput { .. })
    ));
    assert_eq!(runtime.neuron(), &before);
}

#[test]
fn masukan_infinity_positif_dan_negatif_ditolak() {
    for current in [f64::INFINITY, f64::NEG_INFINITY] {
        let mut runtime = LifRuntime::new(config()).expect("konfigurasi uji sah");
        assert!(matches!(
            runtime.step(current),
            Err(SimError::InvalidInput { .. })
        ));
        assert_eq!(runtime.neuron().completed_steps(), 0);
    }
}

#[test]
fn overflow_numerik_ditolak_dan_keadaan_tetap_atomik() {
    let cfg = LifConfig::new(1.0, 10.0, f64::MAX, -65.0, -70.0, -50.0, 0);
    let mut runtime = LifRuntime::new(cfg).expect("konfigurasi hingga diterima");
    runtime.step(0.0).expect("langkah tanpa arus berhasil");
    let before = runtime.neuron().clone();
    assert!(matches!(
        runtime.step(f64::MAX),
        Err(SimError::NumericalOverflow { .. })
    ));
    assert_eq!(runtime.neuron(), &before);
}

#[test]
fn konfigurasi_pada_batas_positif_finite_valid() {
    let cfg = LifConfig::new(
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN_POSITIVE,
        -3.0,
        -2.0,
        -1.0,
        0,
    );
    let mut runtime = LifRuntime::new(cfg).expect("batas positif hingga valid");
    let result = runtime.step(0.0).expect("langkah pada batas valid");
    assert_eq!(result.membrane_potential_mv, -3.0);
}

#[test]
fn batch_gagal_mempertahankan_hanya_langkah_sukses_sebelumnya() {
    let mut runtime = LifRuntime::new(config()).expect("konfigurasi uji sah");
    assert!(matches!(
        runtime.run(&[1.0, f64::NAN, 2.0]),
        Err(SimError::InvalidInput { .. })
    ));
    assert_eq!(runtime.neuron().completed_steps(), 1);
    assert!(runtime.neuron().membrane_potential_mv() > config().rest_potential_mv());
}

#[test]
fn potensial_awal_di_ambang_ditolak() {
    assert!(matches!(
        shella::LifNeuron::with_initial_potential(config(), -60.0),
        Err(SimError::InvalidInput { .. })
    ));
}
