use shella::{
    DatasetId, DatasetLoader, IndoSumPartition, IndonesianVocabulary, LanguageNetwork,
    LexiconMetadata, LicenseRegistry, LifConfig, LifNetwork, NusaXSentimentLabel, NusaXSplit,
    Vocabulary, tokenize,
};
use std::{fs, path::PathBuf};

fn data_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data/language")
        .join(relative)
}

fn empty_lexicon() -> IndonesianVocabulary {
    IndonesianVocabulary::new(LexiconMetadata::indonesian("upstream-test", "1"))
        .expect("metadata valid")
}

#[test]
fn bundled_stopwords_are_hash_verified_and_loadable() {
    let registry = LicenseRegistry::verified_sources();
    let stopwords = DatasetLoader::load_stopwords_json(data_path("stopwords-id.json"), &registry)
        .expect("pinned stopword file loads");

    assert!(stopwords.len() > 100);
    assert!(stopwords.contains(" DAN ").expect("query normalizes"));
    assert_eq!(
        stopwords
            .filter_text("Shella dan Shella")
            .expect("pipeline accepts stopwords"),
        ["shella", "shella"]
    );
}

#[test]
fn pud_is_imported_as_a_verified_ud_evaluation_treebank_and_roundtrips() {
    let registry = LicenseRegistry::verified_sources();
    let mut lexicon = empty_lexicon();
    let report = DatasetLoader::load_ud_conllu(
        data_path("ud-indonesian-pud/id_pud-ud-test.conllu"),
        DatasetId::UdIndonesianPud,
        &mut lexicon,
        &registry,
    )
    .expect("pinned PUD corpus loads");

    assert_eq!(report.sentences, 1000);
    assert!(report.tokens > 10_000);
    assert!(report.analyses_added > 0);
    assert!(report.dependencies_added > 0);
    assert_eq!(lexicon.datasets().len(), 1);
    assert_eq!(lexicon.datasets()[0].license_id, "CC BY-SA 3.0");

    let restored = IndonesianVocabulary::from_json(&lexicon.to_json().expect("serialize"))
        .expect("restore imported PUD lexicon");
    assert_eq!(restored, lexicon);
}

#[test]
fn nusax_mt_is_parsed_deterministically_with_multilingual_alignment() {
    let registry = LicenseRegistry::verified_sources();
    let path = data_path("nusax/mt/train.csv");
    let first = DatasetLoader::load_nusax_mt_csv(&path, NusaXSplit::Train, &registry)
        .expect("pinned MT train CSV loads");
    let second = DatasetLoader::load_nusax_mt_csv(&path, NusaXSplit::Train, &registry)
        .expect("pinned MT train CSV reloads");

    assert_eq!(first, second);
    assert!(!first.sentences.is_empty());
    let (indonesian, english) = first
        .translation(0, "indonesian", "english")
        .expect("aligned language columns");
    assert!(indonesian.contains("Nikmati cicilan"));
    assert!(english.contains("Enjoy 0% instalment"));
    assert!(
        first
            .translation(usize::MAX, "indonesian", "english")
            .is_err()
    );
}

#[test]
fn nusax_indonesian_sentiment_pipeline_and_evaluation_use_held_out_data() {
    let registry = LicenseRegistry::verified_sources();
    let corpus = DatasetLoader::load_nusax_sentiment_csv(
        data_path("nusax/sentiment/indonesian/test.csv"),
        "indonesian",
        NusaXSplit::Test,
        &registry,
    )
    .expect("pinned sentiment test CSV loads");

    assert_eq!(corpus.source.license_id, "CC BY-SA 4.0");
    assert!(!corpus.tokenize(0).expect("text tokenizes").is_empty());
    let labels = corpus
        .examples
        .iter()
        .map(|example| example.label)
        .collect::<Vec<_>>();
    let metrics = corpus
        .evaluate_predictions(&labels)
        .expect("test predictions can be measured");
    assert_eq!(metrics.samples, corpus.examples.len());
    assert_eq!(metrics.accuracy, 1.0);
    assert!(corpus.evaluate_predictions(&[]).is_err());

    let training = DatasetLoader::load_nusax_sentiment_csv(
        data_path("nusax/sentiment/indonesian/train.csv"),
        "indonesian",
        NusaXSplit::Train,
        &registry,
    )
    .expect("pinned training split loads separately");
    assert!(
        training
            .evaluate_predictions(
                &training
                    .examples
                    .iter()
                    .map(|example| example.label)
                    .collect::<Vec<_>>()
            )
            .is_err()
    );
    assert!(
        corpus
            .evaluate_predictions(&[NusaXSentimentLabel::Positive])
            .is_err()
    );
}

#[test]
fn nusax_lexicon_lookup_and_persisted_restore_keep_source_metadata() {
    let registry = LicenseRegistry::verified_sources();
    let lexicon = DatasetLoader::load_nusax_lexicon_csv(
        data_path("nusax/lexicon/acehnese.csv"),
        "acehnese",
        &registry,
    )
    .expect("pinned Indonesian-Acehnese lexicon loads");
    let translations = lexicon
        .lookup_indonesian(" ABU ")
        .expect("normalized fast lookup");
    assert!(translations.contains(&"abee"));

    let serialized = lexicon.to_json().expect("lexicon serializes");
    let restored = shella::NusaXLexicon::from_json(&serialized, &registry)
        .expect("persisted lexicon restores and rebuilds its index");
    assert_eq!(
        restored.lookup_indonesian("abu").expect("restored lookup"),
        translations
    );
    assert_eq!(restored.source, lexicon.source);
}

#[test]
fn modified_source_bytes_and_unlicensed_sources_are_rejected() {
    let registry = LicenseRegistry::verified_sources();
    let temp =
        std::env::temp_dir().join(format!("shella-nusax-corrupt-{}.csv", std::process::id()));
    fs::write(&temp, b"id,text,label\n1,teks,positive\n").expect("corrupt file written");
    assert!(
        DatasetLoader::load_nusax_sentiment_csv(&temp, "indonesian", NusaXSplit::Test, &registry,)
            .is_err()
    );
    fs::remove_file(temp).expect("temporary input removed");

    let indosum_temp = std::env::temp_dir().join(format!(
        "shella-indosum-corrupt-{}.tar.gz",
        std::process::id()
    ));
    fs::write(&indosum_temp, b"not the pinned archive").expect("corrupt archive written");
    assert!(
        DatasetLoader::load_indosum_archive(&indosum_temp, 1, IndoSumPartition::Test, &registry,)
            .is_err()
    );
    fs::remove_file(indosum_temp).expect("temporary archive removed");

    assert!(registry.provenance(DatasetId::IndoNli).is_err());
    assert!(registry.provenance(DatasetId::UdIndonesianGsd).is_err());
    assert!(registry.provenance(DatasetId::UdIndonesianCsui).is_err());
}

#[test]
fn every_bundled_nusax_csv_and_license_matches_the_pinned_git_blob() {
    let registry = LicenseRegistry::verified_sources();
    let provenance = registry
        .provenance(DatasetId::NusaX)
        .expect("NusaX source is approved");
    assert_eq!(provenance.file_git_blob_sha1.len(), 51);

    for (upstream_path, _) in provenance.file_git_blob_sha1 {
        let local_path = upstream_path
            .strip_prefix("datasets/")
            .expect("manifest path is rooted at datasets/");
        registry
            .verify_file(
                DatasetId::NusaX,
                &upstream_path,
                &fs::read(data_path(&format!("nusax/{local_path}")))
                    .expect("bundled source file is readable"),
            )
            .expect("bundled source file matches pinned revision");
    }
}

#[test]
fn nusax_lexicon_terms_are_interned_into_the_real_language_network() {
    let registry = LicenseRegistry::verified_sources();
    let lexicon = DatasetLoader::load_nusax_lexicon_csv(
        data_path("nusax/lexicon/acehnese.csv"),
        "acehnese",
        &registry,
    )
    .expect("pinned lexicon loads");
    let vocabulary = Vocabulary::fit(std::iter::empty::<&str>(), 1, 1)
        .expect("empty initial vocabulary is valid");
    let network = LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 0), 16)
        .expect("LIF network config is valid");
    let mut language =
        LanguageNetwork::new(network, vocabulary).expect("language network initializes");

    let pairs = lexicon
        .intern_terms(&mut language)
        .expect("lexicon terms are interned atomically");
    assert_eq!(pairs.len(), lexicon.entries.len());
    for (pair, entry) in pairs.iter().zip(&lexicon.entries) {
        let indonesian_tokens = tokenize(&entry.indonesian).expect("Indonesian terms tokenize");
        let local_tokens = tokenize(&entry.translation).expect("local terms tokenize");
        assert_eq!(pair.indonesian.len(), indonesian_tokens.len());
        assert_eq!(pair.translation.len(), local_tokens.len());
        for (word_id, token) in pair.indonesian.iter().zip(indonesian_tokens) {
            assert_eq!(language.vocabulary().word(*word_id), Some(token.as_str()));
            assert!(
                !language
                    .word_subwords(*word_id)
                    .unwrap_or_default()
                    .is_empty()
            );
        }
        for (word_id, token) in pair.translation.iter().zip(local_tokens) {
            assert_eq!(language.vocabulary().word(*word_id), Some(token.as_str()));
        }
    }
}

#[test]
fn indosum_stream_loads_licensed_jsonl_and_measures_extractive_summaries() {
    let registry = LicenseRegistry::verified_sources();
    let path = data_path("indosum/indosum.tar.gz");
    registry
        .verify_file(
            DatasetId::IndoSum,
            "README.txt",
            &fs::read(data_path("indosum/README.txt")).expect("license notice is readable"),
        )
        .expect("bundled data license notice matches the pinned copy");
    let corpus = DatasetLoader::load_indosum_archive(&path, 2, IndoSumPartition::Test, &registry)
        .expect("verified IndoSum test fold loads from the original archive");

    assert_eq!(corpus.fold, 2);
    assert_eq!(corpus.partition, IndoSumPartition::Test);
    assert!(!corpus.articles.is_empty());
    assert_eq!(corpus.source.license_id, "CC BY-SA 4.0");
    let article = &corpus.articles[0];
    let oracle = article.gold_extractive_summary();
    assert!(!oracle.is_empty());
    let metrics = article
        .evaluate_summary(&oracle)
        .expect("token-level ROUGE-1 can score a provided summary");
    assert!((0.0..=1.0).contains(&metrics.f1));
    let reference_score = article
        .evaluate_summary(&article.summary)
        .expect("reference summary can be scored");
    assert_eq!(reference_score.precision, 1.0);
    assert_eq!(reference_score.recall, 1.0);
    assert_eq!(reference_score.f1, 1.0);
    assert!(article.evaluate_summary(&[]).is_err());

    let restored = DatasetLoader::load_indosum_archive(&path, 2, IndoSumPartition::Test, &registry)
        .expect("bundled archive reloads after restart");
    assert_eq!(restored.articles, corpus.articles);
    assert!(
        DatasetLoader::load_indosum_archive(&path, 0, IndoSumPartition::Test, &registry,).is_err()
    );
}
