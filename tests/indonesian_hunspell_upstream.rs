use shella::{DatasetLoader, IndonesianVocabulary, LexiconMetadata, LicenseRegistry};

#[test]
#[ignore = "requires the exact upstream dictionary files supplied through environment variables"]
fn verifies_and_loads_pinned_hunspell_distribution() {
    let dictionary = std::env::var("SHELLA_HUNSPELL_DIC").expect("dictionary path supplied");
    let affix = std::env::var("SHELLA_HUNSPELL_AFF").expect("affix path supplied");
    let mut vocabulary =
        IndonesianVocabulary::new(LexiconMetadata::indonesian("source-validation", "1"))
            .expect("metadata valid");
    let report = DatasetLoader::load_hunspell_files(
        dictionary,
        affix,
        &mut vocabulary,
        &LicenseRegistry::verified_sources(),
    )
    .expect("pinned upstream files load");
    assert_eq!(report.tokens + report.skipped_forms, 31_128);
    assert_eq!(vocabulary.entries().len(), report.entries_added);
    assert!(!vocabulary.hunspell_rules().is_empty());
    let makan = vocabulary
        .lookup("makan")
        .expect("normalized")
        .expect("root exists");
    assert!(
        vocabulary
            .analyze_hunspell("makan")
            .expect("analysis query valid")
            .iter()
            .any(|analysis| analysis.lemma_word_id == makan.word_id)
    );
    assert!(
        !vocabulary
            .analyze_hunspell("makanan")
            .expect("derived-form query valid")
            .is_empty()
    );
    let restored = IndonesianVocabulary::from_json(&vocabulary.to_json().expect("serialize"))
        .expect("restore roots, rules, IDs, and provenance");
    assert_eq!(restored, vocabulary);
    assert!(
        !restored
            .analyze_hunspell("makanan")
            .expect("restored derived-form query valid")
            .is_empty()
    );
}
