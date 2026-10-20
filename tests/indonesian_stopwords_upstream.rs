use shella::{DatasetLoader, LicenseRegistry};

#[test]
#[ignore = "requires the pinned stopwords-id JSON supplied through SHELLA_STOPWORDS_ID"]
fn loads_actual_pinned_stopwords_and_preserves_source_revision() {
    let path = std::env::var("SHELLA_STOPWORDS_ID").expect("stopword path supplied");
    let registry = LicenseRegistry::verified_sources();
    let stopwords =
        DatasetLoader::load_stopwords_json(path, &registry).expect("pinned stopwords load");
    assert_eq!(
        stopwords.provenance().revision,
        "6109494eb4cfebd75c59bd832848374d02f4e907"
    );
    assert!(stopwords.contains("yang").expect("query normalizes"));
    assert!(
        stopwords
            .filter_text("yang saya makan nasi")
            .expect("text filters")
            .len()
            < 4
    );
}
