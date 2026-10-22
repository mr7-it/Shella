use shella::{DatasetSplitConfig, GenerativeLanguageModel};

const GENERATION_HOLDOUT_PROMPTS: &[&str] = &[
    "Bagaimana cara merawat tanaman di rumah?",
    "Apa saja manfaat berjalan kaki setiap pagi?",
    "Mengapa kita harus mematikan lampu yang tidak dipakai?",
];

#[test]
fn prompt_generalisasi_tidak_muncul_di_split_nusax_internal() {
    let corpus = GenerativeLanguageModel::load_nusax_training_corpus(
        "data/language/nusax/sentiment/indonesian/train.csv",
        DatasetSplitConfig::default(),
    )
    .expect("korpus NusaX train resmi");
    corpus.validate_disjoint_ids().expect("split disjoint");
    let all_records = corpus
        .training
        .iter()
        .chain(&corpus.validation)
        .chain(&corpus.test)
        .collect::<Vec<_>>();
    for prompt in GENERATION_HOLDOUT_PROMPTS {
        let normalized = prompt.to_lowercase();
        assert!(
            all_records
                .iter()
                .all(|record| !record.text.contains(&normalized)),
            "prompt holdout ada pada corpus: {prompt}"
        );
    }
}
