use shella::{DatasetId, DatasetIntegrityManifest, SimError, Vocabulary, WordID};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn temporary_directory(label: &str) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("shella-{label}-{}-{sequence}", std::process::id()));
    fs::create_dir_all(&path).expect("temporary directory");
    path
}

#[test]
fn bundled_integrity_manifest_verifies_every_dataset_file_and_format() {
    let manifest = DatasetIntegrityManifest::bundled().expect("bundled manifest parses");
    let on_disk = DatasetIntegrityManifest::load("data/language-integrity.json")
        .expect("manifest survives reload");
    assert_eq!(manifest, on_disk);
    let report = manifest
        .verify_directory("data/language")
        .expect("all pinned files verify");
    assert_eq!(manifest.files.len(), 63);
    assert_eq!(report.files_verified, 63);
    assert!(report.total_bytes > 90_000_000);
}

#[test]
fn sha256_checks_detect_same_size_dataset_corruption() {
    let manifest = DatasetIntegrityManifest::bundled().expect("manifest");
    let original = include_bytes!("../data/language/stopwords-id.json");
    manifest
        .verify_source_bytes(DatasetId::StopwordsIsoId, "stopwords-id.json", original)
        .expect("the pinned bytes pass SHA-256");

    let mut altered = original.to_vec();
    let last = altered.len() - 2;
    altered[last] ^= 1;
    assert_eq!(altered.len(), original.len());
    assert!(matches!(
        manifest.verify_source_bytes(DatasetId::StopwordsIsoId, "stopwords-id.json", &altered),
        Err(SimError::CorruptData { .. })
    ));
    assert!(matches!(
        manifest.verify_source_bytes(DatasetId::HunspellDictionaries, "Indonesia.dic", original),
        Err(SimError::CorruptData { .. })
    ));
}

#[test]
fn external_hunspell_hashes_are_pinned_and_checked_by_the_same_manifest() {
    let entry = shella::DatasetFileIntegrity {
        size_bytes: 3,
        sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_owned(),
        format: shella::DatasetFileFormat::Text,
        encoding: shella::DatasetEncoding::Utf8,
    };
    let manifest = DatasetIntegrityManifest {
        format_version: 1,
        files: BTreeMap::from([("fixture.txt".to_owned(), entry.clone())]),
        external_files: BTreeMap::from([("hunspell/Indonesia.dic".to_owned(), entry)]),
    };
    manifest
        .verify_source_bytes(DatasetId::HunspellDictionaries, "Indonesia.dic", b"abc")
        .expect("external dictionary matches registered digest");
    assert!(matches!(
        manifest.verify_source_bytes(DatasetId::HunspellDictionaries, "Indonesia.dic", b"abd"),
        Err(SimError::CorruptData { .. })
    ));
}

#[test]
fn manifest_rejects_untracked_files_and_invalid_utf8() {
    let directory = temporary_directory("integrity");
    fs::write(directory.join("notes.txt"), "valid UTF-8").expect("write text");
    let manifest = DatasetIntegrityManifest::create(&directory).expect("build manifest");
    manifest
        .verify_directory(&directory)
        .expect("initial directory matches");

    fs::write(directory.join("extra.txt"), "unexpected").expect("write extra file");
    assert!(matches!(
        manifest.verify_directory(&directory),
        Err(SimError::CorruptData { .. })
    ));

    let invalid = temporary_directory("encoding");
    fs::write(invalid.join("bad.txt"), [0xff, 0xfe]).expect("write invalid UTF-8");
    assert!(matches!(
        DatasetIntegrityManifest::create(&invalid),
        Err(SimError::CorruptData { .. })
    ));
    fs::remove_dir_all(directory).expect("remove integrity temp");
    fs::remove_dir_all(invalid).expect("remove encoding temp");
}

#[test]
fn word_id_round_trip_is_normalized_unicode_and_explicit_for_unknowns() {
    let vocabulary = Vocabulary::fit(["Café 猫 café belum terdaftar"], 1, 16).expect("vocabulary");
    let ids = vocabulary
        .encode_word_ids("CAFÉ cafe\u{301} 猫 belum_terdaftar")
        .expect("encode stable IDs");
    assert_eq!(ids[0], ids[1]);
    assert_eq!(
        vocabulary.lookup_word("CAFÉ").expect("normalized lookup"),
        Some(ids[0])
    );
    assert_eq!(
        vocabulary.lookup_word("dua kata").expect("phrase lookup"),
        None
    );
    assert_eq!(
        vocabulary.decode_word_ids(&ids).expect("decode"),
        ["café", "café", "猫", "belum", "terdaftar"]
    );

    let unknown = vocabulary.encode_word_ids("kataasing").expect("unknown");
    assert_eq!(unknown, [WordID::new(0)]);
    assert_eq!(
        vocabulary
            .decode_word_ids(&[WordID::new(0)])
            .expect("unknown token"),
        ["<unk>"]
    );
    assert!(matches!(
        vocabulary.decode_word_ids(&[WordID::new(u64::MAX)]),
        Err(SimError::InvalidInput {
            parameter: "word_id",
            ..
        })
    ));
}

#[test]
fn vocabulary_save_restore_and_extension_preserve_ids_and_both_indexes() {
    let directory = temporary_directory("vocabulary");
    let path = directory.join("vocabulary.json");
    let vocabulary = Vocabulary::fit(["apel jeruk apel café"], 1, 16).expect("fit vocabulary");
    let old_id = vocabulary.word_id("apel");
    vocabulary.save(&path).expect("atomic save");

    let mut restored = Vocabulary::load(&path).expect("restart restore");
    assert_eq!(restored.word_id("apel"), old_id);
    assert_eq!(restored.word(old_id), Some("apel"));
    let new_id = restored.intern_word("Nanas").expect("intern new word");
    assert!(new_id.get() > old_id.get());
    restored.save(&path).expect("persist extension");

    let restarted = Vocabulary::load(&path).expect("second restart restore");
    assert_eq!(restarted.word_id("apel"), old_id);
    assert_eq!(restarted.word(old_id), Some("apel"));
    assert_eq!(restarted.word(new_id), Some("nanas"));
    assert_eq!(restarted, restored);

    fs::write(&path, b"{corrupt").expect("corrupt persisted snapshot");
    assert!(matches!(
        Vocabulary::load(&path),
        Err(SimError::CorruptData { .. })
    ));
    fs::remove_dir_all(directory).expect("remove vocabulary temp");
}

#[test]
fn legacy_and_sparse_word_ids_restore_bidirectional_index() {
    let legacy =
        r#"{"tokens":["<unk>","alpha","beta"],"token_to_id":{"<unk>":0,"alpha":1,"beta":2}}"#;
    let migrated: Vocabulary = serde_json::from_str(legacy).expect("legacy vocabulary");
    assert_eq!(migrated.word(WordID::new(2)), Some("beta"));

    let sparse = r#"{
        "tokens":["<unk>","alpha","beta"],
        "word_ids":[0,40,900],
        "next_word_id":901,
        "token_to_id":{"<unk>":0,"alpha":1,"beta":2}
    }"#;
    let restored: Vocabulary = serde_json::from_str(sparse).expect("sparse stable IDs");
    assert_eq!(restored.word(WordID::new(40)), Some("alpha"));
    assert_eq!(restored.word(WordID::new(900)), Some("beta"));
    assert_eq!(
        restored.encode_word_ids("beta alpha").expect("encode"),
        [WordID::new(900), WordID::new(40)]
    );
}
