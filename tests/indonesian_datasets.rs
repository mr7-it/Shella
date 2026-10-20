use shella::{DatasetId, DatasetLoader, IndonesianVocabulary, LexiconMetadata, LicenseRegistry};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static TEST_ID: AtomicU64 = AtomicU64::new(0);

fn temp_path(label: &str, filename: &str) -> PathBuf {
    let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "shella-indonesian-dataset-{label}-{}-{id}",
        std::process::id()
    ));
    fs::create_dir_all(&directory).expect("direktori fixture dibuat");
    directory.join(filename)
}

fn cleanup(path: PathBuf) {
    let directory = path.parent().expect("direktori fixture").to_path_buf();
    fs::remove_file(path).expect("fixture dibersihkan");
    fs::remove_dir(directory).expect("direktori fixture dibersihkan");
}

fn empty_lexicon() -> IndonesianVocabulary {
    IndonesianVocabulary::new(LexiconMetadata::indonesian("loader-test", "1"))
        .expect("metadata valid")
}

#[test]
fn ud_loader_rejects_unpinned_fixture_without_partial_mutation() {
    let path = temp_path("ud", "id_csui-ud-train.conllu");
    fs::write(
        &path,
        "# sent_id = test-only\n\
         1\tmemakan\tmakan\tVERB\t_\tVoice=Act|VerbForm=Fin\t0\troot\t_\t_\n\
         2\tnasi\tnasi\tNOUN\t_\tNumber=Sing\t1\tobj\t_\t_\n\n",
    )
    .expect("fixture CoNLL-U ditulis");
    let registry = LicenseRegistry::verified_sources();
    let mut lexicon = empty_lexicon();
    assert!(
        DatasetLoader::load_ud_conllu(&path, DatasetId::UdIndonesianCsui, &mut lexicon, &registry)
            .is_err()
    );
    assert!(lexicon.entries().is_empty());
    assert!(lexicon.datasets().is_empty());
    cleanup(path);
}

#[test]
fn malformed_or_unpinned_ud_is_rejected_without_partial_lexicon_mutation() {
    let path = temp_path("invalid-ud", "id_gsd-ud-train.conllu");
    fs::write(&path, "not the pinned upstream treebank").expect("fixture ditulis");
    let registry = LicenseRegistry::verified_sources();
    let mut lexicon = empty_lexicon();
    assert!(
        DatasetLoader::load_ud_conllu(&path, DatasetId::UdIndonesianGsd, &mut lexicon, &registry)
            .is_err()
    );
    assert!(lexicon.entries().is_empty());
    assert!(lexicon.datasets().is_empty());
    cleanup(path);
}

#[test]
fn stopwords_loader_rejects_unpinned_json_and_unknown_sources() {
    let registry = LicenseRegistry::verified_sources();
    let path = temp_path("stopwords", "stopwords-id.json");
    fs::write(&path, r#"["dan","yang"]"#).expect("fixture JSON ditulis");
    assert!(DatasetLoader::load_stopwords_json(&path, &registry).is_err());
    cleanup(path);

    assert!(registry.provenance(DatasetId::WordNetBahasa).is_err());
    assert!(registry.provenance(DatasetId::IndoNli).is_err());
}

#[test]
fn hunspell_loader_rejects_unpinned_dictionary_atomically() {
    let dictionary = temp_path("hunspell", "Indonesia.dic");
    let affix = dictionary.with_file_name("Indonesia.aff");
    fs::write(&dictionary, "1\nmakan\n").expect("fixture dictionary ditulis");
    fs::write(&affix, "SET UTF-8\nFLAG long\n").expect("fixture affix ditulis");
    let mut lexicon = empty_lexicon();
    assert!(
        DatasetLoader::load_hunspell_files(
            &dictionary,
            &affix,
            &mut lexicon,
            &LicenseRegistry::verified_sources()
        )
        .is_err()
    );
    assert!(lexicon.entries().is_empty());
    assert!(lexicon.datasets().is_empty());
    fs::remove_file(dictionary).expect("dictionary fixture dibersihkan");
    cleanup(affix);
}

#[test]
fn license_registry_pins_supported_sources_and_rejects_unverified_versions() {
    let registry = LicenseRegistry::verified_sources();
    assert!(registry.provenance(DatasetId::UdIndonesianGsd).is_err());
    assert!(registry.provenance(DatasetId::UdIndonesianCsui).is_err());
    assert_eq!(
        registry
            .provenance(DatasetId::StopwordsIsoId)
            .expect("stopword list is registered")
            .license_id,
        "MIT"
    );
    assert_eq!(
        registry
            .provenance(DatasetId::HunspellDictionaries)
            .expect("Hunspell dictionary is registered")
            .license_id,
        "MPL-2.0 OR LGPL-3.0-or-later"
    );

    let mut unverified = registry
        .provenance(DatasetId::HunspellDictionaries)
        .expect("Hunspell dictionary is registered");
    unverified.revision = "unverified-branch".to_owned();
    assert!(registry.verify(&unverified).is_err());
}
