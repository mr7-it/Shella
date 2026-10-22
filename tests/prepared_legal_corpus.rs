use shella::{CorpusSplit, GenerativeLanguageModel, TransformerConfig};

#[test]
fn korpus_hukum_aktual_memvalidasi_provenance_split_dan_dapat_dimuat_model() {
    let corpus_path = format!(
        "{}/data/indonesian_legal_corpus/prepared_corpus.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let corpus = CorpusSplit::load_prepared_json(&corpus_path)
        .expect("korpus hasil OCR harus lolos validasi provenance dan split");

    assert_eq!(corpus.manifest.source.dataset_id, "shella-id-public-laws");
    assert_eq!(corpus.manifest.split.seed, 42);
    assert_eq!(corpus.manifest.split.source_records, 5);
    assert_eq!(corpus.manifest.split.exact_duplicates_removed, 0);
    assert_eq!(corpus.manifest.split.near_duplicates_removed, 0);
    assert_eq!(corpus.training.len(), 3);
    assert_eq!(corpus.validation.len(), 1);
    assert_eq!(corpus.test.len(), 1);
    corpus
        .validate_disjoint_ids()
        .expect("ID antardokumen dan split harus disjoint");

    let model = GenerativeLanguageModel::from_corpus(TransformerConfig::default(), &corpus)
        .expect("model Shella harus dapat dibuat dari korpus hukum aktual");
    assert!(model.parameter_count() > 0);
    assert!(model.parameter_count() <= 4_000_000);
}

#[test]
#[ignore = "membaca JSON korpus ensiklopedis besar; jalankan eksplisit setelah build korpus"]
fn korpus_koheren_indonesia_aktual_lolos_loader_dan_dapat_membentuk_model() {
    let corpus_path = format!(
        "{}/data/indonesian_coherent_corpus/prepared_corpus.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let corpus = CorpusSplit::load_prepared_json(&corpus_path)
        .expect("korpus koheren harus lolos validasi checksum, provenance, dan split");

    assert_eq!(
        corpus.manifest.source.dataset_id,
        "shella-indonesian-coherent-2026-10-22"
    );
    assert_eq!(corpus.manifest.split.seed, 42);
    assert!(corpus.manifest.split.source_records > 100);
    assert!(corpus.manifest.source.license_id.contains("CC BY-SA 4.0"));
    assert_eq!(corpus.manifest.source.source_file_path, "sources.json");
    assert!(!corpus.training.is_empty());
    assert!(!corpus.validation.is_empty());
    assert!(!corpus.test.is_empty());
    corpus
        .validate_disjoint_ids()
        .expect("ID dokumen harus tetap terpisah antarsplit");

    let model = GenerativeLanguageModel::from_corpus(TransformerConfig::default(), &corpus)
        .expect("tokenizer dan Transformer Shella harus dapat dibentuk dari korpus baru");
    assert!(model.parameter_count() > 0);
    assert!(model.parameter_count() <= 4_000_000);
}
