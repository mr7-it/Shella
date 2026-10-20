use shella::{
    Checkpoint, IndonesianVocabulary, LexiconEntryInput, LexiconMetadata, LifConfig, LifNetwork,
    LinguisticMetadata, SourceProvenance,
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
        "shella-indonesian-lexicon-{label}-{}-{id}.json",
        std::process::id()
    ))
}

fn source(entry_id: &str) -> SourceProvenance {
    SourceProvenance {
        source_name: "Fixture metadata for tests only".to_owned(),
        source_uri: format!("https://example.org/test-entries/{entry_id}"),
        license: "CC BY-SA 4.0 (test metadata only)".to_owned(),
        license_uri: "https://creativecommons.org/licenses/by-sa/4.0/".to_owned(),
        source_entry_id: entry_id.to_owned(),
    }
}

fn lexicon() -> IndonesianVocabulary {
    IndonesianVocabulary::new(LexiconMetadata::indonesian("test-only", "1"))
        .expect("metadata tes valid")
}

fn entry(headword: &str, source_id: &str) -> LexiconEntryInput {
    LexiconEntryInput {
        headword: headword.to_owned(),
        sources: vec![source(source_id)],
        linguistic: LinguisticMetadata::default(),
    }
}

#[test]
fn normalized_lookup_keeps_persistent_word_ids_and_variants() {
    let mut lexicon = lexicon();
    let mut cafe = entry("  CAFÉ ", "fixture-1");
    cafe.linguistic.spelling_variants = vec!["Cafe\u{0301}".to_owned()];
    let cafe_id = lexicon.upsert(cafe).expect("entry valid");

    let found = lexicon
        .lookup("  café, ")
        .expect("input valid")
        .expect("headword ditemukan");
    assert_eq!(found.word_id, cafe_id);
    assert_eq!(
        lexicon
            .lookup("Cafe\u{0301}")
            .expect("alias valid")
            .expect("variasi ditemukan")
            .word_id,
        cafe_id
    );
    assert_eq!(lexicon.vocabulary().word(cafe_id), Some("café"));
    assert!(lexicon.lookup("dua kata").is_err());
}

#[test]
fn upsert_preserves_ids_and_batch_import_is_transactional() {
    let mut lexicon = lexicon();
    let first_id = lexicon.upsert(entry("kata", "fixture-1")).expect("entry");
    let updated_id = lexicon
        .upsert(entry("KATA", "fixture-2"))
        .expect("entry diperbarui");
    assert_eq!(updated_id, first_id);
    assert_eq!(lexicon.entries()[0].sources.len(), 2);

    let invalid_batch = r#"{
        "schema_version": 1,
        "metadata": {"name":"test-only","version":"1","language_tag":"id"},
        "entries": [
          {"headword":"baru","sources":[{"source_name":"Fixture metadata for tests only","source_uri":"https://example.org/test-entries/fixture-3","license":"CC BY-SA 4.0 (test metadata only)","license_uri":"https://creativecommons.org/licenses/by-sa/4.0/","source_entry_id":"fixture-3"}]},
          {"headword":"konflik","sources":[]}
        ]
    }"#;
    assert!(lexicon.import_json(invalid_batch).is_err());
    assert_eq!(lexicon.entries().len(), 1);
    assert!(lexicon.lookup("baru").expect("lookup").is_none());
    assert_eq!(
        lexicon.lookup("kata").expect("lookup").unwrap().word_id,
        first_id
    );

    let mut conflicting_source = entry("kata", "fixture-1");
    conflicting_source.sources[0].license = "conflicting license claim".to_owned();
    assert!(lexicon.upsert(conflicting_source).is_err());
    assert_eq!(
        lexicon.entries()[0].sources[0].license,
        source("fixture-1").license
    );
}

#[test]
fn batch_import_assigns_reproducible_ids_for_the_same_ordered_input() {
    let document = r#"{
        "schema_version": 1,
        "metadata": {"name":"test-only","version":"1","language_tag":"id"},
        "entries": [
          {"headword":"kedua","sources":[{"source_name":"Fixture metadata for tests only","source_uri":"https://example.org/test-entries/a","license":"CC BY-SA 4.0 (test metadata only)","license_uri":"https://creativecommons.org/licenses/by-sa/4.0/","source_entry_id":"a"}]},
          {"headword":"pertama","sources":[{"source_name":"Fixture metadata for tests only","source_uri":"https://example.org/test-entries/b","license":"CC BY-SA 4.0 (test metadata only)","license_uri":"https://creativecommons.org/licenses/by-sa/4.0/","source_entry_id":"b"}]}
        ]
    }"#;
    let mut first = lexicon();
    let mut second = lexicon();
    first.import_json(document).expect("batch valid");
    second.import_json(document).expect("batch valid");
    assert_eq!(first, second);
    assert_eq!(first.lookup("kedua").unwrap().unwrap().word_id.get(), 1);
    assert_eq!(first.lookup("pertama").unwrap().unwrap().word_id.get(), 2);
}

#[test]
fn validates_provenance_definitions_versions_and_alias_collisions() {
    let mut lexicon = lexicon();
    let mut no_source = entry("kata", "fixture-1");
    no_source.sources.clear();
    assert!(lexicon.upsert(no_source).is_err());
    let mut invalid_uri = entry("kata", "fixture-1");
    invalid_uri.sources[0].source_uri = "https:///missing-host".to_owned();
    assert!(lexicon.upsert(invalid_uri).is_err());

    let mut definition = entry("kata", "fixture-1");
    definition.linguistic.definition = Some("Definition tied to test provenance".to_owned());
    assert!(lexicon.upsert(definition).is_err());

    let mut cited_definition = entry("kata", "fixture-1");
    cited_definition.linguistic.definition = Some("Definition tied to test provenance".to_owned());
    cited_definition.linguistic.definition_source_entry_id = Some("fixture-1".to_owned());
    lexicon.upsert(cited_definition).expect("citation present");

    let mut collision = entry("lain", "fixture-2");
    collision.linguistic.spelling_variants = vec!["KATA".to_owned()];
    assert!(lexicon.upsert(collision).is_err());
    assert_eq!(lexicon.entries().len(), 1);

    let unsupported = lexicon.clone();
    let json = unsupported.to_json().expect("json");
    let mut value: serde_json::Value = serde_json::from_str(&json).expect("json value");
    value["schema_version"] = serde_json::json!(99);
    assert!(serde_json::from_value::<IndonesianVocabulary>(value).is_err());
    assert_eq!(unsupported.entries().len(), 1);
}

#[test]
fn checkpoint_restores_lexicon_ids_metadata_and_neural_vocabulary() {
    let path = temp_path("restart");
    let mut lexicon = lexicon();
    let known_id = lexicon.upsert(entry("bahasa", "fixture-1")).expect("entry");
    let serialized = serde_json::to_string(&lexicon).expect("serialize");
    let restored_json = IndonesianVocabulary::from_json(&serialized).expect("restore JSON");
    assert_eq!(restored_json, lexicon);

    Checkpoint::save_indonesian_vocabulary(&path, &lexicon).expect("checkpoint disimpan");
    let mut restored =
        Checkpoint::restore_indonesian_vocabulary(&path).expect("checkpoint dipulihkan");
    assert_eq!(restored.metadata(), lexicon.metadata());
    assert_eq!(
        restored.lookup("BAHASA").expect("lookup").unwrap().word_id,
        known_id
    );
    let next_id = restored
        .upsert(entry("neural", "fixture-2"))
        .expect("ID berikutnya");
    assert!(next_id.get() > known_id.get());

    let network = LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 0), 2)
        .expect("network valid");
    let language = restored
        .to_language_network(network)
        .expect("integrasi language network");
    assert_eq!(language.vocabulary().word(known_id), Some("bahasa"));
    let neuron = language.network().neuron_id(0).expect("neuron tersedia");
    let mut language = language;
    language
        .bind_word(known_id, &[neuron])
        .expect("WordID lexicon terhubung ke neuron");
    assert_eq!(
        language.neurons_for_word(known_id),
        Some([neuron].as_slice())
    );
    fs::remove_file(path).expect("checkpoint dibersihkan");
}

#[test]
fn unlisted_terms_are_segmented_without_inferred_meaning() {
    let lexicon = lexicon();
    assert!(lexicon.lookup("belum terdaftar").is_err());
    assert_eq!(
        lexicon
            .segment_unlisted("  MENYALA ")
            .expect("segmentation"),
        ["m", "e", "n", "y", "a", "l", "a"]
    );
}
