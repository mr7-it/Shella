use shella::{LifConfig, LifNetwork, RewardModulatedStdpConfig, SimError, StdpConfig, Synapse};

fn config(dt: f64, tau: f64, resistance: f64, threshold: f64, refractory: usize) -> LifConfig {
    LifConfig::new(dt, tau, resistance, -65.0, -70.0, threshold, refractory)
}

fn reward_network(
    weight: f64,
    min_weight: f64,
    max_weight: f64,
    learning_rate: f64,
    eligibility_tau: f64,
) -> LifNetwork {
    let mut network =
        LifNetwork::new(config(1.0, 10.0, 1.0, -60.0, 0), 2).expect("jaringan reward valid");
    network
        .add_synapse(
            Synapse::new(0, 1, weight, min_weight, max_weight, 1).expect("sinaps reward valid"),
        )
        .expect("sinaps ditambahkan");
    let reward_config = RewardModulatedStdpConfig::new(
        StdpConfig::new(1.0, 1.0, 10.0, 10.0),
        eligibility_tau,
        learning_rate,
    );
    network
        .enable_reward_modulated_stdp(reward_config)
        .expect("STDP reward aktif");
    network
}

fn causal_pair(network: &mut LifNetwork) {
    network
        .step(&[100.0, 0.0])
        .expect("spike pra pada langkah 1");
    network
        .step(&[0.0, 100.0])
        .expect("spike pasca pada langkah 2");
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "nilai aktual {actual:.16e}, diharapkan {expected:.16e}, toleransi {tolerance:.3e}"
    );
}

#[test]
fn heterogenitas_tau_menghasilkan_respons_numerik_berbeda_pada_arus_sama() {
    let configs = [
        config(1.0, 1.0, 1.0, -40.0, 0),
        config(1.0, 10.0, 1.0, -40.0, 0),
    ];
    let mut network = LifNetwork::from_configs(&configs).expect("konfigurasi heterogen valid");
    assert_eq!(network.neuron_count(), 2);

    let output = network.step(&[10.0, 10.0]).expect("langkah heterogen");
    let fast_expected = -65.0 + 10.0 * (1.0 - (-1.0_f64).exp());
    let slow_expected = -65.0 + 10.0 * (1.0 - (-0.1_f64).exp());
    assert!((output.outcomes()[0].membrane_potential_mv - fast_expected).abs() < 1.0e-13);
    assert!((output.outcomes()[1].membrane_potential_mv - slow_expected).abs() < 1.0e-13);
    assert_ne!(
        output.outcomes()[0].membrane_potential_mv,
        output.outcomes()[1].membrane_potential_mv
    );
}

#[test]
fn heterogenitas_ambang_menghasilkan_keputusan_spike_per_neuron() {
    let configs = [
        config(1.0, 10.0, 1.0, -50.0, 0),
        config(1.0, 10.0, 1.0, -40.0, 0),
    ];
    let mut network = LifNetwork::from_configs(&configs).expect("konfigurasi sah");
    let output = network.step(&[200.0, 200.0]).expect("arus sama diterapkan");
    assert_eq!(output.spiking_neurons(), &[0]);
    assert_eq!(network.neuron(0).expect("neuron 0").spike_count(), 1);
    assert_eq!(network.neuron(1).expect("neuron 1").spike_count(), 0);
}

#[test]
fn heterogenitas_refraktori_dipertahankan_per_neuron() {
    let configs = [
        config(1.0, 10.0, 1.0, -60.0, 0),
        config(1.0, 10.0, 1.0, -60.0, 2),
    ];
    let mut network = LifNetwork::from_configs(&configs).expect("konfigurasi sah");
    assert_eq!(
        network
            .neuron(0)
            .expect("neuron 0")
            .config()
            .refractory_steps(),
        0
    );
    assert_eq!(
        network
            .neuron(1)
            .expect("neuron 1")
            .config()
            .refractory_steps(),
        2
    );
    let spikes = network.step(&[100.0, 100.0]).expect("kedua neuron spike");
    assert_eq!(spikes.spiking_neurons(), &[0, 1]);
    assert_eq!(
        network.neuron(1).expect("neuron 1").refractory_remaining(),
        2
    );
}

#[test]
fn heterogenitas_menolak_dt_berbeda_agar_waktu_tetap_sinkron() {
    let configs = [
        config(1.0, 10.0, 1.0, -60.0, 0),
        config(0.5, 10.0, 1.0, -60.0, 0),
    ];
    assert!(matches!(
        LifNetwork::from_configs(&configs),
        Err(SimError::InvalidTopology { .. })
    ));
}

#[test]
fn heterogenitas_menolak_daftar_kosong_atau_konfigurasi_tidak_sah() {
    assert!(matches!(
        LifNetwork::from_configs(&[]),
        Err(SimError::InvalidTopology { .. })
    ));
    let invalid = [
        config(1.0, 10.0, 1.0, -60.0, 0),
        config(1.0, 0.0, 1.0, -60.0, 0),
    ];
    assert!(matches!(
        LifNetwork::from_configs(&invalid),
        Err(SimError::InvalidConfiguration { .. })
    ));
}

#[test]
fn heterogenitas_dapat_dikombinasikan_dengan_jaringan_dan_stdp() {
    let configs = [
        config(1.0, 1.0, 1.0, -40.0, 0),
        config(1.0, 10.0, 2.0, -60.0, 1),
    ];
    let mut network = LifNetwork::from_configs(&configs).expect("jaringan heterogen");
    network
        .add_synapse(Synapse::new(0, 1, 1.0, 0.0, 2.0, 1).expect("sinaps valid"))
        .expect("sinaps ditambahkan");
    network
        .enable_stdp(StdpConfig::new(0.1, 0.1, 10.0, 10.0))
        .expect("STDP aktif");
    assert_eq!(network.neuron_count(), 2);
    assert_eq!(network.synapse_count(), 1);
    assert_eq!(
        network
            .step(&[200.0, 0.0])
            .expect("langkah valid")
            .completed_step(),
        1
    );
}

#[test]
fn reward_config_menolak_tau_eligibility_atau_learning_rate_tidak_sah() {
    let stdp = StdpConfig::new(1.0, 1.0, 10.0, 10.0);
    assert!(
        RewardModulatedStdpConfig::new(stdp, 0.0, 0.1)
            .validate()
            .is_err()
    );
    assert!(
        RewardModulatedStdpConfig::new(stdp, 10.0, f64::NAN)
            .validate()
            .is_err()
    );
}

#[test]
fn reward_modulated_stdp_mengumpulkan_eligibility_tanpa_update_bobot_instans() {
    let mut network = reward_network(1.0, 0.0, 5.0, 0.5, 10.0);
    causal_pair(&mut network);
    assert_eq!(network.synapses()[0].weight_na(), 1.0);
    network.apply_reward(2.0).expect("reward valid");
    close(
        network.synapses()[0].weight_na(),
        1.0 + (-0.1_f64).exp(),
        1.0e-14,
    );
}

#[test]
fn reward_modulated_stdp_mendiskonkan_reward_tertunda_dengan_decay_eligibility() {
    let mut network = reward_network(1.0, 0.0, 5.0, 0.25, 10.0);
    causal_pair(&mut network);
    network
        .step(&[0.0, 0.0])
        .expect("eligibility meluruh satu langkah");
    assert_eq!(network.synapses()[0].weight_na(), 1.0);
    network.apply_reward(2.0).expect("reward tertunda valid");
    close(
        network.synapses()[0].weight_na(),
        1.0 + 0.5 * (-0.2_f64).exp(),
        1.0e-14,
    );
}

#[test]
fn reward_negatif_mengubah_eligibility_potensiasi_ke_arah_sebaliknya() {
    let mut network = reward_network(1.0, 0.0, 5.0, 0.25, 10.0);
    causal_pair(&mut network);
    network.apply_reward(-2.0).expect("reward negatif sah");
    close(
        network.synapses()[0].weight_na(),
        1.0 - 0.5 * (-0.1_f64).exp(),
        1.0e-14,
    );
}

#[test]
fn reward_positif_memperkuat_eligibility_negatif_dari_urutan_ltd() {
    let mut network = reward_network(1.0, 0.0, 5.0, 0.25, 10.0);
    network
        .step(&[0.0, 100.0])
        .expect("spike pasca pada langkah 1");
    network
        .step(&[100.0, 0.0])
        .expect("spike pra pada langkah 2");
    network.apply_reward(2.0).expect("reward positif sah");
    close(
        network.synapses()[0].weight_na(),
        1.0 - 0.5 * (-0.1_f64).exp(),
        1.0e-14,
    );
}

#[test]
fn reward_nol_tidak_mengubah_bobot_dan_eligibility_tetap_tersedia() {
    let mut network = reward_network(1.0, 0.0, 5.0, 0.5, 10.0);
    causal_pair(&mut network);
    network.apply_reward(0.0).expect("reward nol valid");
    network.apply_reward(1.0).expect("eligibility tetap ada");
    close(
        network.synapses()[0].weight_na(),
        1.0 + 0.5 * (-0.1_f64).exp(),
        1.0e-14,
    );
}

#[test]
fn reward_dijepit_pada_batas_bobot_sinaps() {
    let mut network = reward_network(1.0, 0.0, 1.1, 10.0, 10.0);
    causal_pair(&mut network);
    network.apply_reward(100.0).expect("reward besar finite");
    assert_eq!(network.synapses()[0].weight_na(), 1.1);
}

#[test]
fn reward_menolak_nilai_non_finite_dan_mode_stdp_biasa() {
    let mut network = reward_network(1.0, 0.0, 5.0, 0.5, 10.0);
    let before = network.clone();
    assert!(matches!(
        network.apply_reward(f64::NAN),
        Err(SimError::InvalidInput { .. })
    ));
    assert_eq!(network, before);
    assert!(matches!(
        network.apply_reward(f64::INFINITY),
        Err(SimError::InvalidInput { .. })
    ));

    let mut local_only =
        LifNetwork::new(config(1.0, 10.0, 1.0, -60.0, 0), 2).expect("jaringan lokal STDP");
    local_only
        .enable_stdp(StdpConfig::new(1.0, 1.0, 10.0, 10.0))
        .expect("STDP lokal aktif");
    assert!(matches!(
        local_only.apply_reward(1.0),
        Err(SimError::InvalidInput { .. })
    ));
}

#[test]
fn reward_overflow_membatalkan_semua_perubahan_bobot() {
    let mut network =
        LifNetwork::new(config(1.0, 10.0, 1.0, -60.0, 0), 4).expect("jaringan overflow valid");
    network
        .add_synapse(Synapse::new(0, 1, 0.0, 0.0, f64::MAX, 1_000).expect("sinaps pertama"))
        .expect("sinaps pertama ditambahkan");
    network
        .add_synapse(Synapse::new(2, 3, f64::MAX, 0.0, f64::MAX, 1_000).expect("sinaps kedua"))
        .expect("sinaps kedua ditambahkan");
    let reward_config =
        RewardModulatedStdpConfig::new(StdpConfig::new(1.0, 1.0, 10.0, 10.0), 10.0, 1.0e308);
    network
        .enable_reward_modulated_stdp(reward_config)
        .expect("reward STDP aktif");
    network
        .step(&[100.0, 0.0, 100.0, 0.0])
        .expect("spike pra pada dua sinaps");
    network
        .step(&[0.0, 100.0, 0.0, 100.0])
        .expect("spike pasca pada dua sinaps");
    let before = network.clone();
    assert!(matches!(
        network.apply_reward(1.0),
        Err(SimError::NumericalOverflow { .. })
    ));
    assert_eq!(network, before);
}
