use shella::{
    Checkpoint, LanguageNetwork, LearningMemory, LifConfig, LifNetwork, ReasoningConfig,
    ReasoningEngine, StdpConfig, Synapse, Vocabulary,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static TEST_ID: AtomicU64 = AtomicU64::new(0);

fn temp_path(label: &str) -> PathBuf {
    let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "shella-language-network-{label}-{}-{id}.json",
        std::process::id()
    ))
}

fn network(count: usize) -> LifNetwork {
    LifNetwork::new(
        LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 0),
        count,
    )
    .expect("jaringan valid")
}

#[test]
fn vocabulary_menyimpan_word_id_dan_migrasi_schema_legacy() {
    let mut vocabulary = Vocabulary::fit(["alpha beta beta"], 1, 8).expect("vocabulary valid");
    let alpha_id = vocabulary.word_id("alpha");
    assert_eq!(vocabulary.word(alpha_id), Some("alpha"));
    assert_ne!(alpha_id, vocabulary.word_id("beta"));

    let encoded = serde_json::to_string(&vocabulary).expect("vocabulary terserialisasi");
    let restored: Vocabulary = serde_json::from_str(&encoded).expect("vocabulary dipulihkan");
    assert_eq!(restored, vocabulary);
    assert_eq!(restored.word_id("alpha"), alpha_id);

    let gamma_id = vocabulary.intern_word("gamma").expect("kata baru diintern");
    assert_eq!(
        vocabulary.intern_word("gamma").expect("kata yang sama"),
        gamma_id
    );
    assert_eq!(vocabulary.word_id("alpha"), alpha_id);

    let legacy = r#"{"tokens":["<unk>","alpha"],"token_to_id":{"<unk>":0,"alpha":1}}"#;
    let migrated: Vocabulary = serde_json::from_str(legacy).expect("legacy dimigrasikan");
    assert_eq!(migrated.word_id("alpha").get(), 1);
    assert_eq!(migrated.id("alpha"), 1);
}

#[test]
fn identitas_neuron_sinaps_dan_representasi_word_terdistribusi() {
    let vocabulary = Vocabulary::fit(["alpha beta"], 1, 8).expect("vocabulary valid");
    let mut language = LanguageNetwork::new(network(3), vocabulary).expect("language network");
    let alpha = language.vocabulary().word_id("alpha");
    let beta = language.vocabulary().word_id("beta");
    let neuron_ids = (0..3)
        .map(|index| language.network().neuron_id(index).expect("neuron ID"))
        .collect::<Vec<_>>();
    assert_ne!(neuron_ids[0], neuron_ids[1]);

    language
        .bind_word(alpha, &[neuron_ids[0], neuron_ids[1]])
        .expect("representasi alpha terdistribusi");
    language
        .bind_word(beta, &[neuron_ids[2]])
        .expect("representasi beta");
    assert_eq!(
        language.encode_currents("alpha", 80.0).expect("arus alpha"),
        [40.0, 40.0, 0.0]
    );
    assert!(
        !language
            .word_subwords(alpha)
            .expect("subword alpha")
            .is_empty()
    );

    let connection_ids = language
        .connect_words(alpha, beta, 2.0, -10.0, 10.0, 1)
        .expect("koneksi all-to-all");
    assert_eq!(connection_ids.len(), 2);
    assert_ne!(connection_ids[0], connection_ids[1]);
    assert_eq!(language.network().synapse_count(), 2);
    for connection in language.connections() {
        assert_eq!(language.connection_weight_na(connection), Some(2.0));
    }
}

#[test]
fn koneksi_kata_menggunakan_stdp_jaringan_dan_checkpoint_memulihkan_semua_id() {
    let path = temp_path("checkpoint");
    let vocabulary = Vocabulary::fit(["alpha beta"], 1, 8).expect("vocabulary valid");
    let mut neural = network(2);
    neural
        .enable_stdp(StdpConfig::new(0.5, 0.5, 10.0, 10.0))
        .expect("STDP aktif");
    let mut language = LanguageNetwork::new(neural, vocabulary).expect("language network");
    let alpha = language.vocabulary().word_id("alpha");
    let beta = language.vocabulary().word_id("beta");
    let source = language.network().neuron_id(0).expect("neuron sumber");
    let target = language.network().neuron_id(1).expect("neuron tujuan");
    language.bind_word(alpha, &[source]).expect("alpha terikat");
    language.bind_word(beta, &[target]).expect("beta terikat");
    let gamma = language
        .intern_word("gamma")
        .expect("kata baru ditambahkan");
    language.bind_word(gamma, &[source]).expect("gamma terikat");
    let synapse_id = language
        .connect_words(alpha, beta, 1.0, 0.0, 10.0, 1)
        .expect("sinaps kata")[0];

    language.step_text("alpha", 200.0).expect("spike pre");
    language.step_text("beta", 200.0).expect("spike post");
    let connection = &language.connections()[0];
    let learned_weight = language
        .connection_weight_na(connection)
        .expect("bobot sinaps");
    assert!(learned_weight > 1.0, "STDP mengubah bobot sinaps aktual");
    assert_eq!(connection.synapse_id(), synapse_id);

    Checkpoint::save_language_network(&path, &language).expect("language network disimpan");
    let mut restored =
        Checkpoint::restore_language_network(&path).expect("language network dipulihkan");
    assert_eq!(restored, language);
    assert_eq!(restored.vocabulary().word(alpha), Some("alpha"));
    assert_eq!(restored.vocabulary().word(gamma), Some("gamma"));
    assert_eq!(restored.network().neuron_id(0), Some(source));
    assert_eq!(restored.network().synapse_id(0), Some(synapse_id));
    assert_eq!(
        restored.connection_weight_na(&restored.connections()[0]),
        Some(learned_weight)
    );
    assert_eq!(
        restored
            .step_text("alpha", 1.0)
            .expect("lanjut setelah restart"),
        language
            .step_text("alpha", 1.0)
            .expect("langkah berkelanjutan")
    );
    let next_ids = restored
        .connect_words(beta, alpha, 1.0, 0.0, 10.0, 1)
        .expect("koneksi baru setelah restart");
    assert!(next_ids[0].get() > synapse_id.get());
    fs::remove_file(path).expect("checkpoint dihapus");
}

#[test]
fn reasoning_engine_menerima_teks_dari_representasi_neuron_terikat() {
    let path = temp_path("memory");
    let network = network(2);
    let vocabulary = Vocabulary::fit(["alpha"], 1, 4).expect("vocabulary valid");
    let mut language = LanguageNetwork::new(network.clone(), vocabulary).expect("language network");
    let alpha = language.vocabulary().word_id("alpha");
    let ids = [
        network.neuron_id(0).expect("neuron 0"),
        network.neuron_id(1).expect("neuron 1"),
    ];
    language.bind_word(alpha, &ids).expect("alpha terikat");
    let memory = LearningMemory::open(&path).expect("memori dibuka");
    let config = ReasoningConfig::new(1, 4, 0.0, 0.0, 0.0).expect("konfigurasi reasoning");
    let mut engine = ReasoningEngine::new(network, memory, config);

    let result = engine
        .reason_text("language", "alpha", &language, 100.0, &[1])
        .expect("reason teks");
    assert_eq!(result.steps()[0].currents_na(), &[50.0, 50.0]);
    assert!(
        engine
            .reason_text("language", "unknown", &language, 100.0, &[1])
            .is_err()
    );

    drop(engine);
    fs::remove_file(&path).expect("database memori dihapus");
    let mut lock_path = path.as_os_str().to_os_string();
    lock_path.push(".lock");
    fs::remove_file(PathBuf::from(lock_path)).expect("lock memori dihapus");
}

#[test]
fn identitas_neuron_tetap_dan_sinaps_mendapat_id_monotonik() {
    let mut network = network(2);
    let neuron_id = network.neuron_id(0).expect("neuron ID stabil");
    network
        .add_synapse(Synapse::new(0, 1, 1.0, 0.0, 2.0, 1).expect("sinaps valid"))
        .expect("sinaps pertama");
    let first = network.synapse_id(0).expect("ID sinaps pertama");
    network
        .add_synapse(Synapse::new(1, 0, 1.0, 0.0, 2.0, 1).expect("sinaps valid"))
        .expect("sinaps kedua");
    let second = network.synapse_id(1).expect("ID sinaps kedua");
    assert_eq!(network.neuron_id(0), Some(neuron_id));
    assert!(second.get() > first.get());
}

#[test]
fn checkpoint_memigrasikan_network_lama_tanpa_field_identitas() {
    let path = temp_path("legacy-network");
    let mut legacy_network = network(2);
    legacy_network
        .add_synapse(Synapse::new(0, 1, 3.0, 0.0, 5.0, 1).expect("sinaps valid"))
        .expect("sinaps ditambahkan");
    let mut stored = serde_json::to_value(&legacy_network).expect("network diserialisasi");
    let state = stored.as_object_mut().expect("object network");
    state.remove("neuron_ids");
    state.remove("synapse_ids");
    state.remove("next_synapse_id");
    let legacy: LifNetwork =
        serde_json::from_value(stored).expect("schema network lama dimigrasikan");

    Checkpoint::save(&path, &legacy, None).expect("network lama disimpan");
    let restored = Checkpoint::restore(&path).expect("network lama dipulihkan");
    assert_eq!(
        restored.network.neuron_id(0).expect("ID neuron"),
        legacy.neuron_id(0).expect("ID neuron legacy")
    );
    assert_eq!(restored.network.synapse_id(0).expect("ID sinaps").get(), 1);
    assert_eq!(restored.network.synapses()[0].weight_na(), 3.0);
    fs::remove_file(path).expect("checkpoint dihapus");
}
