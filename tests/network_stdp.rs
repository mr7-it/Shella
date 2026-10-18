use shella::{LifConfig, LifNetwork, SimError, StdpConfig, Synapse};

fn lif_config() -> LifConfig {
    LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 0)
}

fn synapse(source: usize, target: usize, weight: f64, min: f64, max: f64, delay: usize) -> Synapse {
    Synapse::new(source, target, weight, min, max, delay).expect("fixture sinaps valid")
}

fn network() -> LifNetwork {
    LifNetwork::new(lif_config(), 2).expect("fixture jaringan valid")
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "nilai aktual {actual:.16e}, diharapkan {expected:.16e}, toleransi {tolerance:.3e}"
    );
}

#[test]
fn network_menolak_topologi_kosong_dan_endpoint_tidak_valid() {
    assert!(matches!(
        LifNetwork::new(lif_config(), 0),
        Err(SimError::InvalidTopology { .. })
    ));
    let mut net = network();
    assert!(matches!(
        net.add_synapse(synapse(0, 2, 1.0, 0.0, 2.0, 1)),
        Err(SimError::InvalidTopology { .. })
    ));
    assert!(matches!(
        net.add_synapse(synapse(2, 0, 1.0, 0.0, 2.0, 1)),
        Err(SimError::InvalidTopology { .. })
    ));
    assert_eq!(net.synapse_count(), 0);
}

#[test]
fn network_menolak_koneksi_duplikat_tanpa_mengubah_topologi() {
    let mut net = network();
    net.add_synapse(synapse(0, 1, 1.0, 0.0, 2.0, 1))
        .expect("koneksi pertama ditambahkan");
    assert!(matches!(
        net.add_synapse(synapse(0, 1, 1.5, 0.0, 2.0, 2)),
        Err(SimError::DuplicateSynapse {
            source: 0,
            target: 1
        })
    ));
    assert_eq!(net.synapse_count(), 1);
}

#[test]
fn network_menolak_bobot_atau_penundaan_tidak_sah() {
    assert!(Synapse::new(0, 1, 1.0, 0.0, 2.0, 0).is_err());
    assert!(Synapse::new(0, 1, f64::NAN, 0.0, 2.0, 1).is_err());
    assert!(Synapse::new(0, 1, 1.0, 2.0, 0.0, 1).is_err());
    assert!(Synapse::new(0, 1, 3.0, 0.0, 2.0, 1).is_err());
}

#[test]
fn network_menolak_panjang_atau_nilai_arus_eksternal_tidak_sah() {
    let mut net = network();
    assert!(matches!(
        net.step(&[1.0]),
        Err(SimError::InvalidInput { .. })
    ));
    assert!(matches!(
        net.step(&[1.0, f64::INFINITY]),
        Err(SimError::InvalidInput { .. })
    ));
    assert_eq!(net.completed_steps(), 0);
}

#[test]
fn network_spike_menyebar_ke_tujuan_setelah_delay_dua_langkah() {
    let mut net = network();
    net.add_synapse(synapse(0, 1, 100.0, 0.0, 100.0, 2))
        .expect("sinaps ditambahkan");

    let first = net.step(&[100.0, 0.0]).expect("langkah sumber");
    assert_eq!(first.spiking_neurons(), &[0]);
    assert_eq!(net.pending_input_count(), 1);

    let second = net.step(&[0.0, 0.0]).expect("langkah jeda");
    assert!(second.spiking_neurons().is_empty());
    close(
        second.outcomes()[1].membrane_potential_mv,
        lif_config().rest_potential_mv(),
        0.0,
    );

    let third = net.step(&[0.0, 0.0]).expect("arus tertunda tiba");
    assert_eq!(third.spiking_neurons(), &[1]);
    assert_eq!(net.neuron(1).expect("neuron tujuan").spike_steps(), &[3]);
    assert_eq!(net.pending_input_count(), 0);
}

#[test]
fn network_delay_satu_mengirimkan_arus_pada_langkah_berikutnya() {
    let mut net = network();
    net.add_synapse(synapse(0, 1, 100.0, 0.0, 100.0, 1))
        .expect("sinaps ditambahkan");
    net.step(&[100.0, 0.0]).expect("sumber spike");
    let arrived = net.step(&[0.0, 0.0]).expect("event delay satu tiba");
    assert_eq!(arrived.spiking_neurons(), &[1]);
}

#[test]
fn network_bobot_negatif_menghasilkan_inhibisi_hiperpolarisasi() {
    let mut net = network();
    net.add_synapse(synapse(0, 1, -100.0, -100.0, 0.0, 1))
        .expect("sinaps inhibitorik ditambahkan");
    net.step(&[100.0, 0.0]).expect("sumber spike");
    let arrival = net.step(&[0.0, 0.0]).expect("arus inhibitorik tiba");
    assert!(arrival.outcomes()[1].membrane_potential_mv < -65.0);
    assert!(!arrival.outcomes()[1].spiked);
}

#[test]
fn network_menghormati_arah_koneksi() {
    let mut net = network();
    net.add_synapse(synapse(0, 1, 100.0, 0.0, 100.0, 1))
        .expect("sinaps berarah ditambahkan");
    let result = net.step(&[0.0, 100.0]).expect("hanya tujuan diberi input");
    assert_eq!(result.spiking_neurons(), &[1]);
    assert_eq!(net.pending_input_count(), 0);
}

#[test]
fn network_menerapkan_pembaruan_dua_fase_pada_rantai_rekuren() {
    let mut net = LifNetwork::new(lif_config(), 3).expect("jaringan tiga neuron");
    net.add_synapse(synapse(0, 1, 100.0, 0.0, 100.0, 1))
        .expect("sinaps pertama");
    net.add_synapse(synapse(1, 2, 100.0, 0.0, 100.0, 1))
        .expect("sinaps kedua");

    assert_eq!(
        net.step(&[100.0, 0.0, 0.0])
            .expect("langkah 1")
            .spiking_neurons(),
        &[0]
    );
    let second = net.step(&[0.0, 0.0, 0.0]).expect("langkah 2");
    assert_eq!(second.spiking_neurons(), &[1]);
    assert_eq!(second.outcomes()[2].membrane_potential_mv, -65.0);
    let third = net.step(&[0.0, 0.0, 0.0]).expect("langkah 3");
    assert_eq!(third.spiking_neurons(), &[2]);
}

#[test]
fn network_gagal_atomik_jika_satu_neuron_mengalami_overflow() {
    let config = LifConfig::new(1.0, 10.0, f64::MAX, -65.0, -70.0, -60.0, 0);
    let mut net = LifNetwork::new(config, 2).expect("jaringan valid");
    let before = net.clone();
    assert!(matches!(
        net.step(&[1.0, f64::MAX]),
        Err(SimError::NumericalOverflow { .. })
    ));
    assert_eq!(net, before);
    assert_eq!(net.completed_steps(), 0);
    assert_eq!(net.neuron(0).expect("neuron 0").spike_count(), 0);
}

#[test]
fn network_gagal_atomik_jika_penjumlahan_event_sinaptik_overflow() {
    let mut net = LifNetwork::new(lif_config(), 3).expect("jaringan tiga neuron");
    net.add_synapse(synapse(0, 2, f64::MAX, 0.0, f64::MAX, 1))
        .expect("sinaps pertama");
    net.add_synapse(synapse(1, 2, f64::MAX, 0.0, f64::MAX, 1))
        .expect("sinaps kedua");
    net.step(&[200.0, 200.0, 0.0])
        .expect("kedua sumber memicu event tertunda");
    let before = net.clone();
    assert!(matches!(
        net.step(&[0.0, 0.0, 0.0]),
        Err(SimError::NumericalOverflow { .. })
    ));
    assert_eq!(net, before);
    assert_eq!(net.pending_input_count(), 2);
}

#[test]
fn stdp_validasi_parameter_dan_dapat_diaktifkan_pada_jaringan_kosong() {
    assert!(StdpConfig::new(0.0, 1.0, 10.0, 10.0).validate().is_err());
    assert!(
        StdpConfig::new(1.0, 1.0, f64::NAN, 10.0)
            .validate()
            .is_err()
    );
    let mut net = network();
    net.enable_stdp(StdpConfig::new(0.1, 0.1, 20.0, 20.0))
        .expect("parameter STDP sah");
    assert!(net.stdp_config().is_some());
}

#[test]
fn stdp_potensiasi_terjadi_saat_spike_pra_mendahului_pasca() {
    let mut net = network();
    net.add_synapse(synapse(0, 1, 1.0, 0.0, 5.0, 1))
        .expect("sinaps ditambahkan");
    net.enable_stdp(StdpConfig::new(1.0, 1.0, 10.0, 10.0))
        .expect("STDP diaktifkan");
    net.step(&[100.0, 0.0]).expect("spike pra pada langkah 1");
    net.step(&[0.0, 100.0]).expect("spike pasca pada langkah 2");
    close(
        net.synapses()[0].weight_na(),
        1.0 + (-0.1_f64).exp(),
        1.0e-14,
    );
}

#[test]
fn stdp_depresi_terjadi_saat_spike_pasca_mendahului_pra() {
    let mut net = network();
    net.add_synapse(synapse(0, 1, 1.0, 0.0, 5.0, 1))
        .expect("sinaps ditambahkan");
    net.enable_stdp(StdpConfig::new(1.0, 1.0, 10.0, 10.0))
        .expect("STDP diaktifkan");
    net.step(&[0.0, 100.0]).expect("spike pasca pada langkah 1");
    net.step(&[100.0, 0.0]).expect("spike pra pada langkah 2");
    close(
        net.synapses()[0].weight_na(),
        1.0 - (-0.1_f64).exp(),
        1.0e-14,
    );
}

#[test]
fn stdp_spike_serentak_tidak_membentuk_pasangan_waktu_nol() {
    let mut net = network();
    net.add_synapse(synapse(0, 1, 1.0, 0.0, 5.0, 1))
        .expect("sinaps ditambahkan");
    net.enable_stdp(StdpConfig::new(1.0, 1.0, 10.0, 10.0))
        .expect("STDP diaktifkan");
    let simultaneous = net.step(&[100.0, 100.0]).expect("kedua neuron spike");
    assert_eq!(simultaneous.spiking_neurons(), &[0, 1]);
    assert_eq!(net.synapses()[0].weight_na(), 1.0);
}

#[test]
fn stdp_trace_mengakumulasi_spike_berulang_dengan_peluruhan() {
    let mut net = network();
    net.add_synapse(synapse(0, 1, 1.0, 0.0, 10.0, 1))
        .expect("sinaps ditambahkan");
    net.enable_stdp(StdpConfig::new(0.5, 0.5, 10.0, 10.0))
        .expect("STDP diaktifkan");
    net.step(&[200.0, 0.0]).expect("spike pra pertama");
    net.step(&[200.0, 0.0]).expect("spike pra kedua");
    net.step(&[0.0, 200.0]).expect("spike pasca");
    let decay = (-0.1_f64).exp();
    close(
        net.synapses()[0].weight_na(),
        1.0 + 0.5 * (decay + decay * decay),
        1.0e-14,
    );
}

#[test]
fn stdp_bobot_dijepit_pada_batas_atas_dan_bawah() {
    let learning = StdpConfig::new(10.0, 10.0, 10.0, 10.0);
    let mut potentiation = network();
    potentiation
        .add_synapse(synapse(0, 1, 1.9, 0.0, 2.0, 1))
        .expect("sinaps ditambahkan");
    potentiation.enable_stdp(learning).expect("STDP diaktifkan");
    potentiation.step(&[100.0, 0.0]).expect("spike pra");
    potentiation.step(&[0.0, 100.0]).expect("spike pasca");
    assert_eq!(potentiation.synapses()[0].weight_na(), 2.0);

    let mut depression = network();
    depression
        .add_synapse(synapse(0, 1, 0.1, 0.0, 1.0, 1))
        .expect("sinaps ditambahkan");
    depression.enable_stdp(learning).expect("STDP diaktifkan");
    depression.step(&[0.0, 100.0]).expect("spike pasca");
    depression.step(&[100.0, 0.0]).expect("spike pra");
    assert_eq!(depression.synapses()[0].weight_na(), 0.0);
}

#[test]
fn stdp_dapat_dimatikan_tanpa_mengubah_bobot() {
    let mut net = network();
    net.add_synapse(synapse(0, 1, 1.0, 0.0, 5.0, 1))
        .expect("sinaps ditambahkan");
    net.enable_stdp(StdpConfig::new(1.0, 1.0, 10.0, 10.0))
        .expect("STDP diaktifkan");
    net.disable_stdp();
    assert!(net.stdp_config().is_none());
    assert_eq!(net.synapses()[0].weight_na(), 1.0);
}

#[test]
fn stdp_overflow_membatalkan_keadaan_jaringan_bobot_dan_trace() {
    let mut net = network();
    net.add_synapse(synapse(0, 1, 0.0, 0.0, f64::MAX, 1))
        .expect("sinaps ditambahkan");
    net.enable_stdp(StdpConfig::new(f64::MAX, 1.0, 10.0, 10.0))
        .expect("amplitudo finite sah");
    net.step(&[200.0, 0.0]).expect("spike pra pertama");
    net.step(&[200.0, 0.0]).expect("spike pra kedua");
    let before = net.clone();
    assert!(matches!(
        net.step(&[0.0, 200.0]),
        Err(SimError::NumericalOverflow { .. })
    ));
    assert_eq!(net, before);
}
