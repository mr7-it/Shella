use serde_json::json;
use shella::{
    AssertionStatus, ConceptRelation, DatasetId, Engine, GroupedSemanticExample,
    IndonesianVocabulary, LearningMemory, LexicalAnalysis, LexiconEntryInput, LexiconMetadata,
    LicenseRegistry, LifConfig, LifNetwork, LinguisticMetadata, MemoryStore, PhraseKind,
    ReasoningConfig, ReasoningEngine, SemanticAnswerEvidence, SemanticConfidenceBand,
    SemanticEvidence, SemanticExample, SemanticGapReport, SemanticMemory, SemanticQuestionIntent,
    SemanticRelationKind, SemanticRoleKind, SourceProvenance, Synapse,
};
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

fn memory_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "shella-semantic-{}-{}.json",
        std::process::id(),
        NEXT_PATH.fetch_add(1, Ordering::Relaxed)
    ))
}

fn cleanup(path: &PathBuf) {
    fs::remove_file(path).expect("hapus store semantic");
    let mut lock_path = path.as_os_str().to_os_string();
    lock_path.push(".lock");
    fs::remove_file(PathBuf::from(lock_path)).expect("hapus lock semantic");
}

fn entry(word: &str, part_of_speech: Option<&str>) -> LexiconEntryInput {
    LexiconEntryInput {
        headword: word.to_owned(),
        sources: vec![SourceProvenance {
            source_name: "semantic integration fixture".to_owned(),
            source_uri: "https://example.org/shella-semantic-fixture".to_owned(),
            license: "CC0-1.0".to_owned(),
            license_uri: "https://creativecommons.org/publicdomain/zero/1.0/".to_owned(),
            source_entry_id: format!("fixture:{word}"),
        }],
        linguistic: LinguisticMetadata {
            part_of_speech: part_of_speech.map(str::to_owned),
            lemma: Some(word.to_owned()),
            ..LinguisticMetadata::default()
        },
    }
}

fn lexicon() -> IndonesianVocabulary {
    let mut lexicon = IndonesianVocabulary::new(LexiconMetadata::indonesian(
        "semantic integration fixture",
        "1",
    ))
    .expect("buat lexicon fixture");
    for (word, pos) in [
        ("kucing", Some("NOUN")),
        ("meong", Some("NOUN")),
        ("anjing", Some("NOUN")),
        ("mamalia", Some("NOUN")),
        ("hewan", Some("NOUN")),
        ("makan", Some("VERB")),
        ("tidur", Some("VERB")),
        ("tidak", Some("PART")),
        ("nasi", Some("NOUN")),
        ("ikan", Some("NOUN")),
        ("saya", Some("PRON")),
        ("tinggal", Some("VERB")),
        ("di", Some("ADP")),
        ("ibu", Some("NOUN")),
        ("kota", Some("NOUN")),
        ("bisa", Some("AUX")),
        ("dia", Some("PRON")),
    ] {
        lexicon
            .upsert(entry(word, pos))
            .expect("tambah entri fixture");
    }
    let ambiguous_id = lexicon
        .lookup("bisa")
        .expect("lookup")
        .expect("bisa")
        .word_id;
    let registry = LicenseRegistry::verified_sources();
    let pud = registry
        .provenance(DatasetId::UdIndonesianPud)
        .expect("provenance PUD");
    for lemma in ["bisa", "dapat"] {
        lexicon
            .add_analysis(
                LexicalAnalysis {
                    word_id: ambiguous_id,
                    lemma: lemma.to_owned(),
                    upos: "AUX".to_owned(),
                    features: BTreeMap::new(),
                    dataset: pud.clone(),
                    source_entry_id: format!("pud:{}:{lemma}", ambiguous_id.get()),
                },
                &registry,
            )
            .expect("tambah analisis morphology");
    }
    lexicon
}

fn evidence(source_entry_id: &str, status: AssertionStatus) -> SemanticEvidence {
    SemanticEvidence {
        source: "curated semantic integration fixture".to_owned(),
        source_entry_id: source_entry_id.to_owned(),
        confidence: Some(0.9),
        status,
    }
}

fn concept_relation(
    source: shella::WordID,
    target: shella::WordID,
    kind: SemanticRelationKind,
    evidence_id: &str,
) -> ConceptRelation {
    ConceptRelation {
        source,
        target,
        kind,
        evidence: evidence(evidence_id, AssertionStatus::Hypothesis),
        recorded_at_unix: None,
    }
}

#[test]
fn analisis_memakai_wordid_pos_negasi_batas_dan_frasa_terdaftar() {
    let path = memory_path();
    let mut memory = SemanticMemory::open(&path, lexicon()).expect("semantic memory");
    memory
        .add_phrase(
            "ibu kota",
            PhraseKind::MultiwordTerm,
            "pusat pemerintahan yang dinyatakan oleh sumber",
            evidence("term:ibu-kota", AssertionStatus::SourceAssertion),
        )
        .expect("tambah istilah multi-kata");

    let analysis = memory
        .analyze("Kucing tidak makan ikan di ibu kota.")
        .expect("analisis kalimat");
    assert_eq!(
        analysis.normalized_text,
        "kucing tidak makan ikan di ibu kota."
    );
    assert!(analysis.tokens.iter().all(|token| token.known));
    assert_eq!(
        analysis.tokens[0].word_id,
        memory.lexicon().lookup("kucing").unwrap().unwrap().word_id
    );
    assert_eq!(analysis.structure.negated, Some(true));
    assert_eq!(analysis.structure.predicate_token, Some(2));
    assert_eq!(analysis.phrases.len(), 1);
    assert_eq!(analysis.phrases[0].normalized, "ibu kota");
    assert_eq!(
        analysis.phrases[0].meaning,
        "pusat pemerintahan yang dinyatakan oleh sumber"
    );
    assert_eq!(analysis.phrases[0].token_start, 5);
    assert!(
        analysis
            .structure
            .roles
            .iter()
            .any(|role| role.kind == SemanticRoleKind::Subject && role.span.start_byte == 0)
    );
    drop(memory);
    let mut restored = SemanticMemory::open(&path, lexicon()).expect("pulihkan phrase tersimpan");
    let restored_phrase = restored
        .analyze("ibu kota")
        .expect("analisis phrase setelah restart");
    assert_eq!(
        restored_phrase.phrases[0].meaning,
        "pusat pemerintahan yang dinyatakan oleh sumber"
    );
    restored
        .analyze_in_context("phrase-context", "ibu kota")
        .expect("simpan phrase ke konteks");
    assert!(restored.remove_phrase("ibu kota").expect("hapus phrase"));
    drop(restored);
    let restored = SemanticMemory::open(&path, lexicon()).expect("pulihkan penghapusan phrase");
    assert!(restored.analyze("ibu kota").unwrap().phrases.is_empty());
    drop(restored);
    cleanup(&path);
}

#[test]
fn relasi_sinonim_kontradiksi_inferensi_dan_retrieval_memakai_bukti() {
    let path = memory_path();
    let mut memory = SemanticMemory::open(&path, lexicon()).expect("semantic memory");
    let cat = memory.lexicon().lookup("kucing").unwrap().unwrap().word_id;
    let meong = memory.lexicon().lookup("meong").unwrap().unwrap().word_id;
    assert!(
        !memory
            .add_relation(concept_relation(
                cat,
                meong,
                SemanticRelationKind::Synonym,
                "relation:synonym:1",
            ))
            .expect("tambah sinonim")
    );
    assert!(
        !memory
            .add_relation(concept_relation(
                cat,
                meong,
                SemanticRelationKind::Synonym,
                "relation:synonym:1",
            ))
            .expect("upsert sinonim idempoten")
    );
    assert_eq!(memory.relations().len(), 1);
    memory
        .analyze_in_context("pet-care", "kucing makan ikan")
        .expect("simpan pengalaman");
    assert!(
        memory
            .add_relation(concept_relation(
                cat,
                meong,
                SemanticRelationKind::Antonym,
                "relation:antonym:1",
            ))
            .expect("catat konflik")
    );

    let inference = memory
        .infer(cat, SemanticRelationKind::Synonym)
        .expect("inferensi");
    let trace = &inference
        .iter()
        .find(|(word_id, _)| *word_id == meong)
        .expect("simpulan sinonim")
        .1;
    assert!(!trace.contradicted);
    assert_eq!(trace.evidence_ids, ["relation:synonym:1"]);
    assert_eq!(trace.status, AssertionStatus::Hypothesis);
    assert_eq!(
        memory.detect_gaps("kucing").unwrap().conflicting_tokens,
        ["kucing"]
    );

    let mut verified_antonym = concept_relation(
        cat,
        meong,
        SemanticRelationKind::Antonym,
        "relation:verified-antonym",
    );
    verified_antonym.evidence.status = AssertionStatus::Fact;
    assert!(memory.add_relation(verified_antonym).unwrap());
    let verified_inference = memory.infer(cat, SemanticRelationKind::Synonym).unwrap();
    assert!(
        verified_inference
            .iter()
            .find(|(word_id, _)| *word_id == meong)
            .unwrap()
            .1
            .contradicted
    );

    let recall = memory
        .retrieve("meong tidur", Some("pet-care"), 4)
        .expect("retrieval berbasis sinonim/konteks");
    assert_eq!(recall.len(), 1);
    assert!(recall[0].relevance > 0.0);
    assert!(
        recall[0]
            .reasons
            .iter()
            .any(|reason| reason.contains("sinonim"))
    );

    let mammal = memory.lexicon().lookup("mamalia").unwrap().unwrap().word_id;
    let animal = memory.lexicon().lookup("hewan").unwrap().unwrap().word_id;
    memory
        .add_relation(concept_relation(
            cat,
            mammal,
            SemanticRelationKind::IsA,
            "relation:is-a:1",
        ))
        .expect("jenis dari pertama");
    memory
        .add_relation(concept_relation(
            mammal,
            animal,
            SemanticRelationKind::IsA,
            "relation:is-a:2",
        ))
        .expect("jenis dari kedua");
    let inference_path = memory
        .infer(cat, SemanticRelationKind::IsA)
        .expect("inferensi jenis-dari");
    let trace = &inference_path
        .iter()
        .find(|(word_id, _)| *word_id == animal)
        .expect("simpulan hewan")
        .1;
    assert_eq!(trace.path, [cat, mammal, animal]);
    assert_eq!(trace.evidence_ids, ["relation:is-a:1", "relation:is-a:2"]);
    drop(memory);
    let restored = SemanticMemory::open(&path, lexicon()).expect("pulihkan graph tersimpan");
    assert!(restored.relations().iter().any(|relation| {
        relation.source == cat
            && relation.target == mammal
            && relation.kind == SemanticRelationKind::IsA
    }));
    drop(restored);
    cleanup(&path);
}

#[test]
fn konteks_pulih_setelah_restart_dan_penghapusan_konsisten() {
    let path = memory_path();
    let cat_id;
    {
        let mut memory = SemanticMemory::open(&path, lexicon()).expect("semantic memory");
        cat_id = memory.lexicon().lookup("kucing").unwrap().unwrap().word_id;
        memory
            .analyze_in_context("dialog-1", "kucing makan nasi")
            .expect("kalimat pertama");
        let pronoun = memory
            .analyze_in_context("dialog-1", "dia tidur")
            .expect("anafora konteks");
        assert_eq!(pronoun.context_resolutions.len(), 1);
        assert_eq!(
            pronoun.context_resolutions[0].resolved_word_id,
            Some(cat_id)
        );
        assert_eq!(
            pronoun.context_resolutions[0].evidence,
            shella::EvidenceLevel::RuleHeuristic
        );
    }
    {
        let mut restored =
            SemanticMemory::open(&path, lexicon()).expect("pulihkan semantic memory");
        let recall = restored
            .retrieve("kucing makan", Some("dialog-1"), 2)
            .expect("retrieval setelah restart");
        assert!(!recall.is_empty());
        assert_eq!(
            restored
                .lexicon()
                .lookup("kucing")
                .unwrap()
                .unwrap()
                .word_id,
            cat_id
        );
        assert!(restored.remove_concept(cat_id).expect("hapus konsep"));
        assert!(
            !restored
                .analyze("kucing makan nasi")
                .expect("analisis konsep terhapus sebagai unknown")
                .tokens[0]
                .known
        );
        assert!(
            restored
                .retrieve("kucing makan", Some("dialog-1"), 3)
                .expect("hapus konteks turunan secara konsisten")
                .is_empty()
        );
        assert!(
            !restored
                .retrieve("dia tidur", Some("dialog-1"), 3)
                .expect("kalimat tanpa konsep terhapus tetap ada")
                .is_empty()
        );
        assert!(restored.forget_context("dialog-1").unwrap());
    }
    drop(SemanticMemory::open(&path, lexicon()).expect("restart pascapenghapusan"));
    cleanup(&path);
}

#[test]
fn gap_ambiguity_incremental_learning_dan_validasi_input() {
    let path = memory_path();
    let mut memory = SemanticMemory::open(&path, lexicon()).expect("semantic memory");
    let ambiguity = memory.analyze("saya bisa makan nasi").expect("ambiguity");
    assert_eq!(ambiguity.ambiguities.len(), 1);
    assert_eq!(ambiguity.ambiguities[0].candidate_lemmas, ["bisa", "dapat"]);

    let gaps = memory
        .detect_gaps("burung melahap nasi")
        .expect("deteksi gap");
    assert_eq!(gaps.unknown_tokens, ["burung", "melahap"]);
    assert_eq!(gaps.predicates_without_known_roles, ["melahap"]);
    let old_id = memory.lexicon().lookup("kucing").unwrap().unwrap().word_id;
    let learned_id = memory
        .learn_entry(entry("burung", Some("NOUN")))
        .expect("belajar entri");
    assert_ne!(learned_id, old_id);
    assert!(memory.analyze("burung makan nasi").unwrap().tokens[0].known);
    assert!(
        memory
            .add_relation(concept_relation(
                learned_id,
                learned_id,
                SemanticRelationKind::RelatedTo,
                "bad:self-loop",
            ))
            .is_err()
    );
    drop(memory);
    let restored = SemanticMemory::open(&path, lexicon()).expect("pulihkan pembelajaran");
    assert_eq!(
        restored
            .lexicon()
            .lookup("burung")
            .unwrap()
            .unwrap()
            .word_id,
        learned_id
    );
    drop(restored);
    cleanup(&path);
}

#[test]
fn split_evaluasi_memakai_kalimat_baru_dan_menolak_kebocoran() {
    let path = memory_path();
    let memory = SemanticMemory::open(&path, lexicon()).expect("semantic memory");
    let example = |id: &str, text: &str, subject: &str, object: &str| SemanticExample {
        id: id.to_owned(),
        text: text.to_owned(),
        expected_subject: Some(subject.to_owned()),
        expected_predicate: Some("makan".to_owned()),
        expected_object: Some(object.to_owned()),
        expected_negated: Some(false),
    };
    let training = [example("train-1", "kucing makan ikan", "kucing", "ikan")];
    let validation = [example("valid-1", "anjing makan nasi", "anjing", "nasi")];
    let test = [
        example("test-1", "saya makan nasi", "saya", "nasi"),
        SemanticExample {
            id: "test-2".to_owned(),
            text: "kucing tidak makan ikan".to_owned(),
            expected_subject: Some("kucing".to_owned()),
            expected_predicate: Some("makan".to_owned()),
            expected_object: Some("ikan".to_owned()),
            expected_negated: Some(true),
        },
    ];
    let metrics = memory
        .evaluate_splits(&training, &validation, &test)
        .expect("evaluasi split terpisah");
    assert_eq!(metrics.exact_text_overlap, 0);
    assert_eq!(metrics.test.examples, 2);
    assert_eq!(metrics.test.exact_frames, 2);
    assert_eq!(metrics.test.frame_accuracy, Some(1.0));

    let grouped_training = [GroupedSemanticExample {
        group_id: "source-train".to_owned(),
        example: training[0].clone(),
    }];
    let grouped_validation = [GroupedSemanticExample {
        group_id: "source-validation".to_owned(),
        example: validation[0].clone(),
    }];
    let grouped_test = test
        .iter()
        .cloned()
        .map(|example| GroupedSemanticExample {
            group_id: format!("test:{}", example.id),
            example,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        memory
            .evaluate_grouped_splits(&grouped_training, &grouped_validation, &grouped_test)
            .expect("split per sumber/group")
            .test
            .frame_accuracy,
        Some(1.0)
    );
    let leaked_group = [GroupedSemanticExample {
        group_id: "source-train".to_owned(),
        example: test[0].clone(),
    }];
    assert!(
        memory
            .evaluate_grouped_splits(&grouped_training, &grouped_validation, &leaked_group)
            .is_err()
    );

    let duplicate_test = [example("test-2", "KUCING MAKAN IKAN", "kucing", "ikan")];
    assert!(
        memory
            .evaluate_splits(&training, &validation, &duplicate_test)
            .is_err()
    );
    drop(memory);
    cleanup(&path);
}

#[test]
fn sdk_meneruskan_analisis_ke_semantic_memory_yang_dipasang() {
    let path = memory_path();
    let semantic = SemanticMemory::open(&path, lexicon()).expect("semantic memory");
    let network = LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 2)
        .expect("network");
    let mut engine = Engine::new(network).with_semantic_memory(semantic);
    {
        let mut session = engine.session();
        assert!(session.analyze_semantics("kucing makan nasi").is_ok());
        assert!(
            session
                .analyze_semantics_in_context("sdk-context", "kucing makan nasi")
                .is_ok()
        );
    }
    drop(engine);
    cleanup(&path);
}

#[test]
fn feedback_semantik_mengubah_lif_menyimpan_pengalaman_dan_meningkatkan_recall_parafrasa() {
    let semantic_path = memory_path();
    let learning_path = memory_path();
    let reasoning_checkpoint_path = memory_path();
    let vocabulary = lexicon();
    let cat = vocabulary.lookup("kucing").unwrap().unwrap().word_id;
    let synonym = vocabulary.lookup("meong").unwrap().unwrap().word_id;
    let mut base = LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 4)
        .expect("network");
    for (source, target) in [(0, 2), (1, 2), (0, 3), (1, 3)] {
        base.add_synapse(
            Synapse::new(source, target, 0.0, -100.0, 100.0, 1).expect("synapse valid"),
        )
        .expect("add readout synapse");
    }
    let mut language = vocabulary
        .to_language_network(base)
        .expect("language network dengan ID lexicon");
    let cat_neuron = language.network().neuron_id(0).unwrap();
    let synonym_neuron = language.network().neuron_id(1).unwrap();
    language.bind_word(cat, &[cat_neuron]).unwrap();
    language.bind_word(synonym, &[synonym_neuron]).unwrap();
    let mut semantic = SemanticMemory::open(&semantic_path, vocabulary).unwrap();
    let mut synonym_edge = concept_relation(
        synonym,
        cat,
        SemanticRelationKind::Synonym,
        "source:fixture:synonym",
    );
    synonym_edge.evidence.status = AssertionStatus::SourceAssertion;
    semantic.add_relation(synonym_edge).unwrap();

    let mut learning = LearningMemory::open(&learning_path).unwrap();
    let old = learning
        .record_feedback("unrelated-task", &[0.0, 0.0, 1.0, 0.0], Some(1), Some(1))
        .unwrap();
    let config = ReasoningConfig::new(1, 4, 200.0, 0.0, 0.0).unwrap();
    let mut engine = ReasoningEngine::new(language.network().clone(), learning, config)
        .with_semantic_memory(semantic);

    let baseline = engine
        .reason_text(
            "semantic-task",
            "kucing makan ikan",
            &language,
            20.0,
            &[2, 3],
        )
        .expect("baseline parafrasa tanpa pengalaman");
    assert_eq!(baseline.selected_action(), None);
    let baseline_correct = usize::from(baseline.selected_action() == Some(0));

    let outcome = engine
        .learn_from_semantic_feedback(
            "semantic-task",
            "meong makan ikan",
            &language,
            20.0,
            Some(0),
            Some(1),
            &[2, 3],
            0.01,
        )
        .expect("feedback semantic dipelajari");
    assert!(outcome.analysis().tokens.iter().all(|token| token.known));
    assert!(outcome.gaps().unknown_tokens.is_empty());
    assert!(
        !outcome
            .learning()
            .expect("input known dan tidak ambigu dipelajari")
            .memory_write()
            .prediction_correct()
    );
    assert!(outcome.learning().unwrap().updated_synapses() > 0);
    assert_eq!(engine.memory().len(), 2);
    assert_eq!(
        engine
            .memory()
            .recall_task_similar("unrelated-task", &[0.0, 0.0, 1.0, 0.0], 1)
            .unwrap()[0]
            .record_id(),
        old.record_id()
    );

    let after_learning = engine
        .reason_text(
            "semantic-task",
            "kucing makan ikan",
            &language,
            20.0,
            &[2, 3],
        )
        .expect("uji parafrasa pascapembelajaran");
    assert_eq!(after_learning.selected_action(), Some(0));
    assert!(!after_learning.recalled().is_empty());
    let after_learning_correct = usize::from(after_learning.selected_action() == Some(0));
    assert_eq!((baseline_correct, after_learning_correct), (0, 1));
    engine
        .save_checkpoint(&reasoning_checkpoint_path)
        .expect("reasoning checkpoint semantic");

    drop(engine);
    drop(language);

    let learning = LearningMemory::open(&learning_path).unwrap();
    let semantic = SemanticMemory::open(&semantic_path, lexicon()).unwrap();
    let restored = ReasoningEngine::restore_checkpoint_with_semantic_memory(
        &reasoning_checkpoint_path,
        learning,
        Some(semantic),
    )
    .expect("restore reasoning dengan stores yang cocok");
    assert!(restored.semantic_memory().is_some());
    drop(restored);

    let mut changed_semantic = SemanticMemory::open(&semantic_path, lexicon()).unwrap();
    changed_semantic
        .learn_entry(entry("harimau", Some("NOUN")))
        .unwrap();
    drop(changed_semantic);
    let learning = LearningMemory::open(&learning_path).unwrap();
    let changed_semantic = SemanticMemory::open(&semantic_path, lexicon()).unwrap();
    assert!(
        ReasoningEngine::restore_checkpoint_with_semantic_memory(
            &reasoning_checkpoint_path,
            learning,
            Some(changed_semantic),
        )
        .is_err()
    );
    fs::remove_file(&reasoning_checkpoint_path).unwrap();
    cleanup(&semantic_path);
    cleanup(&learning_path);
}

#[test]
fn semantic_snapshots_lama_deserialisasi_tanpa_makna_atau_conflict_field() {
    let old_phrase = serde_json::json!({
        "normalized": "ibu kota",
        "word_ids": [1, 2],
        "kind": "multiword_term",
        "evidence": {
            "source": "legacy-source",
            "source_entry_id": "legacy-entry",
            "confidence": null,
            "status": "source_assertion"
        }
    });
    let phrase: shella::PhraseEntry = serde_json::from_value(old_phrase).unwrap();
    assert!(phrase.meaning.is_empty());

    let old_gap = serde_json::json!({
        "unknown_tokens": [],
        "ambiguous_tokens": [],
        "predicates_without_known_roles": []
    });
    let gap: SemanticGapReport = serde_json::from_value(old_gap).unwrap();
    assert!(gap.conflicting_tokens.is_empty());
}

#[test]
fn feedback_tidak_menyimpan_tebakan_oov_ambigu_atau_kontradiktif() {
    let semantic_path = memory_path();
    let learning_path = memory_path();
    let vocabulary = lexicon();
    let mut network =
        LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 4).unwrap();
    for (source, target) in [(0, 2), (1, 2)] {
        network
            .add_synapse(Synapse::new(source, target, 0.0, -100.0, 100.0, 1).unwrap())
            .unwrap();
    }
    let language = vocabulary.to_language_network(network.clone()).unwrap();
    let mut semantic = SemanticMemory::open(&semantic_path, vocabulary.clone()).unwrap();
    let cat = vocabulary.lookup("kucing").unwrap().unwrap().word_id;
    let synonym = vocabulary.lookup("meong").unwrap().unwrap().word_id;
    semantic
        .add_relation(concept_relation(
            cat,
            synonym,
            SemanticRelationKind::Synonym,
            "relation:known-synonym",
        ))
        .unwrap();
    semantic
        .add_relation(concept_relation(
            cat,
            synonym,
            SemanticRelationKind::Antonym,
            "relation:conflicting-antonym",
        ))
        .unwrap();
    let learning = LearningMemory::open(&learning_path).unwrap();
    let mut engine = ReasoningEngine::new(
        network,
        learning,
        ReasoningConfig::new(1, 2, 0.0, 0.0, 0.0).unwrap(),
    )
    .with_semantic_memory(semantic);

    for (context, text, expected_gap) in [
        ("oov", "harimau makan ikan", "oov"),
        ("ambiguous", "saya bisa makan nasi", "ambiguous"),
        ("conflict", "kucing makan ikan", "conflict"),
    ] {
        let outcome = engine
            .learn_from_semantic_feedback(
                context,
                text,
                &language,
                20.0,
                Some(0),
                Some(1),
                &[2],
                0.1,
            )
            .unwrap();
        assert!(outcome.learning().is_none());
        match expected_gap {
            "oov" => assert!(!outcome.gaps().unknown_tokens.is_empty()),
            "ambiguous" => assert!(!outcome.gaps().ambiguous_tokens.is_empty()),
            "conflict" => assert!(!outcome.gaps().conflicting_tokens.is_empty()),
            _ => unreachable!(),
        }
        assert_eq!(engine.memory().len(), 0);
        assert_eq!(engine.network().completed_steps(), 0);
    }
    drop(engine);
    drop(language);
    cleanup(&semantic_path);
    cleanup(&learning_path);
}

#[test]
fn sdk_checkpoint_terpadu_memulihkan_store_yang_cocok_dan_menolak_semantic_drift() {
    let semantic_path = memory_path();
    let learning_path = memory_path();
    let general_memory_path = memory_path();
    let checkpoint_path = memory_path();
    let semantic = SemanticMemory::open(&semantic_path, lexicon()).unwrap();
    let learning = LearningMemory::open(&learning_path).unwrap();
    let general_memory = MemoryStore::open(&general_memory_path).unwrap();
    let network =
        LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 2).unwrap();
    let mut engine = Engine::new(network)
        .with_semantic_memory(semantic)
        .with_learning_memory(learning)
        .with_memory(general_memory);
    {
        let mut session = engine.session();
        session
            .learning_memory_mut()
            .unwrap()
            .record_feedback("checkpoint-task", &[100.0, 0.0], Some(0), Some(0))
            .unwrap();
        session
            .remember(
                "checkpoint-record",
                json!({"kept": true}),
                &["semantic".to_owned()],
            )
            .unwrap();
        session
            .step(shella::SdkInput::Currents(vec![100.0, 0.0]))
            .unwrap();
        session
            .save_integrated_checkpoint(&checkpoint_path, None)
            .unwrap();
    }
    drop(engine);

    let semantic = SemanticMemory::open(&semantic_path, lexicon()).unwrap();
    let learning = LearningMemory::open(&learning_path).unwrap();
    let general_memory = MemoryStore::open(&general_memory_path).unwrap();
    let network =
        LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 2).unwrap();
    let mut restored = Engine::new(network)
        .with_semantic_memory(semantic)
        .with_learning_memory(learning)
        .with_memory(general_memory);
    restored
        .session()
        .restore_integrated_checkpoint(&checkpoint_path)
        .expect("restore seluruh store yang cocok");
    assert_eq!(restored.completed_steps(), 1);
    assert_eq!(restored.learning_memory().unwrap().len(), 1);
    assert_eq!(
        restored
            .memory()
            .unwrap()
            .get("checkpoint-record")
            .unwrap()
            .value()["kept"],
        true
    );

    {
        let mut session = restored.session();
        session
            .analyze_semantics_in_context("checkpoint-context", "kucing makan ikan")
            .unwrap();
        assert!(
            session
                .restore_integrated_checkpoint(&checkpoint_path)
                .is_err()
        );
    }
    assert_eq!(restored.completed_steps(), 1);
    restored
        .save_integrated_checkpoint(&checkpoint_path, None)
        .unwrap();
    {
        let mut session = restored.session();
        session
            .remember("post-checkpoint", json!({"changed": true}), &[])
            .unwrap();
        assert!(
            session
                .restore_integrated_checkpoint(&checkpoint_path)
                .is_err()
        );
    }
    assert_eq!(restored.completed_steps(), 1);
    drop(restored);
    fs::remove_file(&checkpoint_path).unwrap();
    cleanup(&semantic_path);
    cleanup(&learning_path);
    cleanup(&general_memory_path);
}

#[test]
fn penghapusan_konsep_menyinkronkan_pengalaman_neural_dan_context_window() {
    let semantic_path = memory_path();
    let learning_path = memory_path();
    let vocabulary = lexicon();
    let cat = vocabulary.lookup("kucing").unwrap().unwrap().word_id;
    let mut semantic = SemanticMemory::open(&semantic_path, vocabulary.clone()).unwrap();
    semantic
        .analyze_in_context("delete-task", "kucing makan ikan")
        .unwrap();
    let mut learning = LearningMemory::open(&learning_path).unwrap();
    learning
        .record_feedback("delete-task", &[1.0, 0.0], Some(0), Some(0))
        .unwrap();
    let network =
        LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 2).unwrap();
    let config = ReasoningConfig::new(1, 2, 0.0, 0.0, 0.0)
        .unwrap()
        .with_context_window(1)
        .unwrap();
    let mut engine = ReasoningEngine::new(network, learning, config).with_semantic_memory(semantic);
    engine.reason("delete-task", &[0.0, 0.0], &[1]).unwrap();
    assert_eq!(engine.context_window().count(), 1);
    assert!(engine.remove_semantic_concept(cat).unwrap());
    assert_eq!(engine.memory().len(), 0);
    assert_eq!(engine.context_window().count(), 0);
    assert!(
        engine
            .semantic_memory()
            .unwrap()
            .pending_learning_task_deletions()
            .is_empty()
    );
    drop(engine);

    let learning = LearningMemory::open(&learning_path).unwrap();
    assert_eq!(learning.len(), 0);
    let semantic = SemanticMemory::open(&semantic_path, vocabulary).unwrap();
    assert!(semantic.pending_learning_task_deletions().is_empty());
    drop(semantic);
    drop(learning);
    cleanup(&semantic_path);
    cleanup(&learning_path);
}

#[test]
fn sdk_response_jalankan_reasoning_dan_semantic_feedback_ke_learning_memory() {
    let semantic_path = memory_path();
    let learning_path = memory_path();
    let vocabulary = lexicon();
    let network = LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 64)
        .expect("LIF network");
    let mut language = vocabulary
        .to_language_network(network.clone())
        .expect("same Indonesian vocabulary");
    for entry in vocabulary.entries() {
        let index = usize::try_from(entry.word_id.get()).expect("WordID fits index");
        let neuron = language
            .network()
            .neuron_id(index)
            .expect("neuron for WordID");
        language
            .bind_word(entry.word_id, &[neuron])
            .expect("bind word");
    }
    let mut semantic =
        SemanticMemory::open(&semantic_path, vocabulary.clone()).expect("semantic store");
    semantic
        .analyze_in_context("dialog", "Kucing makan ikan.")
        .expect("declarative context");
    let learning = LearningMemory::open(&learning_path).expect("learning memory");
    let mut reasoning = ReasoningEngine::new(
        network.clone(),
        learning,
        ReasoningConfig::new(2, 4, 0.0, 0.0, 0.0).unwrap(),
    )
    .with_semantic_memory(semantic);
    let mut sdk = Engine::new(network);
    {
        let mut session = sdk.session();
        let (answer, inference) = session
            .respond_with_reasoning(
                &mut reasoning,
                "dialog",
                "Apakah kucing makan ikan?",
                &language,
                50.0,
                &[63],
                4,
            )
            .expect("SDK delegates to evidence response and reasoning inference");
        assert_eq!(
            answer.evidence_level,
            SemanticAnswerEvidence::ContextStatement
        );
        assert!(answer.answer.starts_with("Ya."));
        assert_eq!(inference.steps().len(), 2);
        let learning = session
            .learn_from_semantic_feedback(
                &mut reasoning,
                "polaritas",
                "Kucing makan ikan.",
                &language,
                50.0,
                Some(0),
                None,
                &[63],
                0.001,
            )
            .expect("explicit teacher feedback only");
        assert!(learning.learning().is_some());
    }
    assert_eq!(reasoning.memory().len(), 1);
    drop(reasoning);
    drop(language);
    cleanup(&semantic_path);
    cleanup(&learning_path);
}

#[test]
fn semantic_responses_evaluated_separately_for_intent_facts_reasoning_fluency_and_holdout() {
    let path = memory_path();
    let vocabulary = lexicon();
    let cat = vocabulary.lookup("kucing").unwrap().unwrap().word_id;
    let mammal = vocabulary.lookup("mamalia").unwrap().unwrap().word_id;
    let mut semantic = SemanticMemory::open(&path, vocabulary).unwrap();
    let mut is_a = concept_relation(
        cat,
        mammal,
        SemanticRelationKind::IsA,
        "fixture:cat-is-mammal",
    );
    is_a.evidence.status = AssertionStatus::Fact;
    semantic.add_relation(is_a).expect("relation edge accepted");
    semantic
        .analyze_in_context("dialog-a", "Kucing makan ikan.")
        .expect("declarative training context");
    semantic
        .analyze_in_context("dialog-a", "Kucing tidak makan nasi.")
        .expect("negated declarative context");

    let relation = semantic
        .respond_in_context("dialog-a", "Apakah kucing termasuk mamalia?", 8)
        .expect("relation query");
    let positive = semantic
        .respond_in_context("dialog-a", "Apakah kucing makan ikan?", 8)
        .expect("positive context query");
    let negated = semantic
        .respond_in_context("dialog-a", "Benarkah kucing tidak makan nasi?", 8)
        .expect("negated context query");
    let unsupported = semantic
        .respond_in_context("dialog-a", "Apakah kucing berenang?", 8)
        .expect("unsupported question abstains");
    let ambiguous = semantic
        .respond_in_context("dialog-a", "Apa arti bisa?", 8)
        .expect("ambiguous lemma abstains");

    let understanding_cases = [
        relation.intent == SemanticQuestionIntent::Relation,
        positive.intent == SemanticQuestionIntent::YesNo,
        negated.intent == SemanticQuestionIntent::YesNo,
        ambiguous.intent == SemanticQuestionIntent::Definition,
        unsupported.intent == SemanticQuestionIntent::YesNo,
    ];
    let understanding_correct = understanding_cases.iter().filter(|case| **case).count();
    assert_eq!(understanding_correct, understanding_cases.len());

    let factual_cases = [
        relation.evidence_level == SemanticAnswerEvidence::DirectSource
            && relation
                .evidence_ids
                .contains(&"fixture:cat-is-mammal".to_owned()),
        positive.evidence_level == SemanticAnswerEvidence::ContextStatement
            && positive.answer.starts_with("Ya."),
        negated.evidence_level == SemanticAnswerEvidence::ContextStatement
            && negated.answer.starts_with("Ya."),
        unsupported.evidence_level == SemanticAnswerEvidence::Insufficient,
        ambiguous.evidence_level == SemanticAnswerEvidence::Insufficient,
    ];
    let factual_correct = factual_cases.iter().filter(|case| **case).count();
    assert_eq!(factual_correct, factual_cases.len());

    semantic
        .analyze_in_context("conflict", "Kucing makan ikan.")
        .unwrap();
    semantic
        .analyze_in_context("conflict", "Kucing tidak makan ikan.")
        .unwrap();
    let contradiction = semantic
        .respond_in_context("conflict", "Apakah kucing makan ikan?", 8)
        .unwrap();
    let reasoning_cases = [
        positive.answer.starts_with("Ya."),
        negated.answer.starts_with("Ya."),
        contradiction.evidence_level == SemanticAnswerEvidence::ConflictingStatements,
        unsupported.evidence_level == SemanticAnswerEvidence::Insufficient,
        ambiguous.confidence_band == SemanticConfidenceBand::Insufficient,
    ];
    let reasoning_correct = reasoning_cases.iter().filter(|case| **case).count();
    assert_eq!(reasoning_correct, reasoning_cases.len());

    semantic
        .analyze_in_context("question-only", "Kucing makan ikan?")
        .unwrap();
    let question_only = semantic
        .respond_in_context("question-only", "Apakah kucing makan ikan?", 8)
        .unwrap();
    assert!(question_only.retrieved_context.is_empty());
    assert_eq!(
        question_only.evidence_level,
        SemanticAnswerEvidence::Insufficient
    );

    let responses = [
        &relation,
        &positive,
        &negated,
        &unsupported,
        &ambiguous,
        &contradiction,
    ];
    let fluent = responses
        .iter()
        .filter(|response| {
            !response.answer.trim().is_empty()
                && response.answer.chars().count() >= 20
                && (response.answer.ends_with(['.', '?']) || response.answer.ends_with('”'))
        })
        .count();
    assert_eq!(fluent, responses.len());

    // New phrasing is evaluated after training statements and never inserted as evidence first.
    let holdout_query = semantic
        .respond_in_context("dialog-a", "Benarkah kucing tidur?", 8)
        .unwrap();
    let generalization_correct = usize::from(
        holdout_query.intent == SemanticQuestionIntent::YesNo
            && holdout_query.evidence_level == SemanticAnswerEvidence::Insufficient,
    );
    assert_eq!(generalization_correct, 1);
    println!(
        "semantic response holdout: understanding={understanding_correct}/{}, factuality={factual_correct}/{}, reasoning={reasoning_correct}/{}, sentence_quality={fluent}/{}, generalization={generalization_correct}/1",
        understanding_cases.len(),
        factual_cases.len(),
        reasoning_cases.len(),
        responses.len(),
    );

    drop(semantic);
    cleanup(&path);
}

#[test]
fn respons_mencocokkan_frame_semantik_sinonim_dan_referen_konteks() {
    let path = memory_path();
    let vocabulary = lexicon();
    let cat = vocabulary.lookup("kucing").unwrap().unwrap().word_id;
    let synonym = vocabulary.lookup("meong").unwrap().unwrap().word_id;
    let mut semantic = SemanticMemory::open(&path, vocabulary).unwrap();

    let mut synonym_edge = concept_relation(
        cat,
        synonym,
        SemanticRelationKind::Synonym,
        "fixture:supported-synonym",
    );
    synonym_edge.evidence.status = AssertionStatus::SourceAssertion;
    synonym_edge.evidence.confidence = None;
    semantic.add_relation(synonym_edge).unwrap();

    let mammal = semantic
        .lexicon()
        .lookup("mamalia")
        .unwrap()
        .unwrap()
        .word_id;
    semantic
        .add_relation(concept_relation(
            cat,
            mammal,
            SemanticRelationKind::IsA,
            "fixture:unverified-hypothesis",
        ))
        .unwrap();
    let hypothesis = semantic
        .respond_in_context("hypothesis", "Apakah kucing termasuk mamalia?", 4)
        .unwrap();
    assert_eq!(
        hypothesis.evidence_level,
        SemanticAnswerEvidence::Insufficient
    );

    semantic
        .analyze_in_context("role-swap", "Kucing makan ikan.")
        .unwrap();
    let swapped = semantic
        .respond_in_context("role-swap", "Apakah ikan makan kucing?", 8)
        .unwrap();
    assert_eq!(swapped.evidence_level, SemanticAnswerEvidence::Insufficient);

    semantic
        .analyze_in_context("verb-object", "Kucing makan ikan.")
        .unwrap();
    let wrong_predicate = semantic
        .respond_in_context("verb-object", "Apakah kucing tidur ikan?", 8)
        .unwrap();
    assert_eq!(
        wrong_predicate.evidence_level,
        SemanticAnswerEvidence::Insufficient
    );
    let synonym_query = semantic
        .respond_in_context("verb-object", "Apakah meong makan ikan?", 8)
        .unwrap();
    assert_eq!(
        synonym_query.evidence_level,
        SemanticAnswerEvidence::ContextStatement
    );
    assert!(synonym_query.answer.starts_with("Ya."));

    semantic
        .analyze_in_context("pronoun", "Kucing makan nasi.")
        .unwrap();
    let pronoun_analysis = semantic
        .analyze_in_context("pronoun", "Dia makan ikan.")
        .unwrap();
    assert_eq!(
        pronoun_analysis.context_resolutions[0].resolved_word_id,
        Some(cat)
    );
    let resolved_query = semantic
        .respond_in_context("pronoun", "Apakah kucing makan ikan?", 8)
        .unwrap();
    assert_eq!(
        resolved_query.evidence_level,
        SemanticAnswerEvidence::ContextStatement
    );

    let heldout_question = "Siapa yang makan ikan?";
    let wh_baseline = semantic
        .respond_in_context("wh-baseline", heldout_question, 8)
        .unwrap();
    let wh_holdout_baseline = usize::from(
        wh_baseline.evidence_level == SemanticAnswerEvidence::ContextStatement
            && wh_baseline.answer.contains("kucing"),
    );
    assert_eq!(wh_holdout_baseline, 0);
    assert_eq!(
        wh_baseline.evidence_level,
        SemanticAnswerEvidence::Insufficient
    );
    semantic
        .analyze_in_context("wh-question", "Kucing makan ikan.")
        .unwrap();
    let who = semantic
        .respond_in_context("wh-question", heldout_question, 8)
        .unwrap();
    assert_eq!(who.intent, SemanticQuestionIntent::OpenQuestion);
    assert_eq!(who.evidence_level, SemanticAnswerEvidence::ContextStatement);
    assert!(who.answer.contains("kucing"));
    assert!(!who.evidence_ids.is_empty());
    let wh_holdout_after_context_learning = usize::from(
        who.evidence_level == SemanticAnswerEvidence::ContextStatement
            && who.answer.contains("kucing"),
    );
    assert_eq!(wh_holdout_after_context_learning, 1);
    println!(
        "natural wh holdout (question absent from stored statements): baseline={wh_holdout_baseline}/1, after semantic context input={wh_holdout_after_context_learning}/1"
    );

    semantic
        .analyze_in_context("opposing", "Kucing makan ikan.")
        .unwrap();
    semantic
        .analyze_in_context("opposing", "Kucing tidak makan ikan.")
        .unwrap();
    let conflict = semantic
        .respond_in_context("opposing", "Apakah kucing makan ikan?", 1)
        .unwrap();
    assert_eq!(
        conflict.evidence_level,
        SemanticAnswerEvidence::ConflictingStatements
    );
    assert_eq!(conflict.supporting_statements.len(), 2);

    drop(semantic);
    cleanup(&path);
}

#[test]
fn sinonim_berbukti_menginjeksikan_arus_ke_language_network_lif() {
    let path = memory_path();
    let lexicon = lexicon();
    let cat = lexicon.lookup("kucing").unwrap().unwrap().word_id;
    let synonym = lexicon.lookup("meong").unwrap().unwrap().word_id;
    let network = LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 2)
        .expect("network");
    let mut language = lexicon
        .to_language_network(network)
        .expect("network dengan vocabulary lexicon yang sama");
    let cat_neuron = language.network().neuron_id(0).expect("neuron kucing");
    let synonym_neuron = language.network().neuron_id(1).expect("neuron sinonim");
    language
        .bind_word(cat, &[cat_neuron])
        .expect("binding kucing");
    language
        .bind_word(synonym, &[synonym_neuron])
        .expect("binding sinonim");
    let mut semantic = SemanticMemory::open(&path, lexicon).expect("semantic memory");
    let mut relation = concept_relation(
        synonym,
        cat,
        SemanticRelationKind::Synonym,
        "relation:verified-synonym",
    );
    relation.evidence.status = AssertionStatus::SourceAssertion;
    semantic.add_relation(relation).expect("relasi ditambahkan");
    let sdk_network = LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 2)
        .expect("network SDK");
    let mut engine = Engine::new(sdk_network).with_semantic_memory(semantic);
    let result;
    {
        let mut session = engine.session();
        result = session
            .resonate_semantics(&mut language, "meong", 20.0)
            .expect("resonansi leksikal ke LIF");
    }
    let cat_index = language.network().neuron_index(cat_neuron).unwrap();
    assert_eq!(result.injected_currents_na[cat_index], 18.0);
    assert_eq!(result.relation_evidence_ids, ["relation:verified-synonym"]);
    assert_eq!(language.network().completed_steps(), 1);
    drop(engine);
    drop(language);
    cleanup(&path);
}

#[test]
fn benchmark_polaritas_100_test_group_held_out_membandingkan_baseline_dan_feedback() {
    #[derive(Clone)]
    struct Sample {
        group_id: String,
        text: String,
        expected_action: usize,
        semantic_example: SemanticExample,
    }

    let training_subjects = ["kucing", "anjing", "saya", "dia", "meong"];
    let training_objects = ["nasi", "ikan", "roti", "buah", "sup"];
    let validation_subjects = ["ibu", "murid", "dokter", "petani", "anak"];
    let validation_objects = ["daging", "telur", "sayur", "jagung", "susu"];
    let test_subjects = [
        "guru", "perawat", "polisi", "nelayan", "penjual", "koki", "sopir", "penari", "pemain",
        "petugas",
    ];
    let test_objects = ["apel", "pisang", "ubi", "ayam", "kacang"];
    let mut vocabulary = IndonesianVocabulary::new(LexiconMetadata::indonesian(
        "train-only polarity benchmark",
        "1",
    ))
    .unwrap();
    for word in training_subjects {
        let part_of_speech = if matches!(word, "saya" | "dia") {
            "PRON"
        } else {
            "NOUN"
        };
        vocabulary
            .upsert(entry(word, Some(part_of_speech)))
            .unwrap();
    }
    for word in training_objects {
        vocabulary.upsert(entry(word, Some("NOUN"))).unwrap();
    }
    vocabulary.upsert(entry("makan", Some("VERB"))).unwrap();
    vocabulary.upsert(entry("tidak", Some("PART"))).unwrap();
    assert!(
        validation_subjects
            .into_iter()
            .chain(validation_objects)
            .chain(test_subjects)
            .chain(test_objects)
            .all(|word| vocabulary.lookup(word).unwrap().is_none())
    );

    let build_samples = |subjects: &[&str], objects: &[&str]| {
        let mut samples = Vec::new();
        for subject in subjects {
            for object in objects {
                for negated in [false, true] {
                    let text = if negated {
                        format!("{subject} tidak makan {object}")
                    } else {
                        format!("{subject} makan {object}")
                    };
                    let group_id = format!("pair:{subject}:{object}");
                    samples.push(Sample {
                        group_id: group_id.clone(),
                        text: text.clone(),
                        expected_action: usize::from(negated),
                        semantic_example: SemanticExample {
                            id: format!("{group_id}:{}", usize::from(negated)),
                            text,
                            expected_subject: Some((*subject).to_owned()),
                            expected_predicate: Some("makan".to_owned()),
                            expected_object: Some((*object).to_owned()),
                            expected_negated: Some(negated),
                        },
                    });
                }
            }
        }
        samples
    };
    let training = build_samples(&training_subjects, &training_objects);
    let validation = build_samples(&validation_subjects, &validation_objects);
    let test = build_samples(&test_subjects, &test_objects);
    assert_eq!(
        (training.len(), validation.len(), test.len()),
        (50, 50, 100)
    );

    let grouped = |samples: &[Sample]| {
        samples
            .iter()
            .map(|sample| GroupedSemanticExample {
                group_id: sample.group_id.clone(),
                example: sample.semantic_example.clone(),
            })
            .collect::<Vec<_>>()
    };
    let grouped_training = grouped(&training);
    let grouped_validation = grouped(&validation);
    let grouped_test = grouped(&test);

    let semantic_path = memory_path();
    let baseline_learning_path = memory_path();
    let training_learning_path = memory_path();
    let checkpointed_vocabulary = vocabulary.clone();
    let base_network =
        LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 64).unwrap();
    let mut neural_network = base_network;
    for word_id in vocabulary.vocabulary().word_ids().iter().copied().skip(1) {
        let source = usize::try_from(word_id.get()).unwrap();
        assert!(source < 62);
        for action_neuron in [62, 63] {
            neural_network
                .add_synapse(Synapse::new(source, action_neuron, 0.0, -20.0, 20.0, 1).unwrap())
                .unwrap();
        }
    }
    let baseline_network = neural_network.clone();
    let mut language_network = vocabulary
        .to_language_network(neural_network)
        .expect("LanguageNetwork dibangun dari lexicon yang sama");
    for entry in vocabulary.entries() {
        let neuron_index = usize::try_from(entry.word_id.get()).unwrap();
        let neuron = language_network.network().neuron_id(neuron_index).unwrap();
        language_network
            .bind_word(entry.word_id, &[neuron])
            .unwrap();
    }
    let config = ReasoningConfig::new(3, 1, 200.0, 0.0, 0.0).unwrap();
    SemanticMemory::validate_grouped_splits(&grouped_training, &grouped_validation, &grouped_test)
        .unwrap();
    drop(LearningMemory::open(&baseline_learning_path).unwrap());
    drop(LearningMemory::open(&training_learning_path).unwrap());

    let mut learned = ReasoningEngine::new(
        language_network.network().clone(),
        LearningMemory::open(&training_learning_path).unwrap(),
        config,
    )
    .with_semantic_memory(
        SemanticMemory::open(&semantic_path, checkpointed_vocabulary.clone()).unwrap(),
    );
    for sample in &training {
        let predicted_action = learned
            .reason_text(
                "polarity-task",
                &sample.text,
                &language_network,
                120.0,
                &[62, 63],
            )
            .expect("prediksi sebelum feedback");
        let outcome = learned
            .learn_from_semantic_feedback(
                "polarity-task",
                &sample.text,
                &language_network,
                120.0,
                Some(sample.expected_action),
                predicted_action.selected_action(),
                &[62, 63],
                0.001,
            )
            .expect("feedback train dipelajari secara incremental");
        assert!(
            outcome.learning().is_some(),
            "feedback tidak diterima untuk sampel {:?}; gap={:?}",
            sample.text,
            outcome.gaps()
        );
    }
    assert_eq!(learned.memory().len(), training.len());
    let test_gap = learned
        .semantic_memory()
        .unwrap()
        .detect_gaps(&test[0].text)
        .unwrap();
    assert_eq!(test_gap.unknown_tokens.len(), 2);
    let trained_network = learned.network().clone();
    drop(learned);

    let evaluate = |network: &LifNetwork, source_memory: &PathBuf, samples: &[Sample]| {
        let mut correct = 0;
        for sample in samples {
            let sample_memory_path = memory_path();
            fs::copy(source_memory, &sample_memory_path).unwrap();
            let mut engine = ReasoningEngine::new(
                network.clone(),
                LearningMemory::open(&sample_memory_path).unwrap(),
                config,
            )
            .with_semantic_memory(
                SemanticMemory::open(&semantic_path, checkpointed_vocabulary.clone()).unwrap(),
            );
            let result = engine.reason_text(
                "polarity-task",
                &sample.text,
                &language_network,
                120.0,
                &[62, 63],
            );
            correct += usize::from(
                result.is_ok_and(|result| result.selected_action() == Some(sample.expected_action)),
            );
            drop(engine);
            cleanup(&sample_memory_path);
        }
        correct
    };
    let baseline_correct = evaluate(&baseline_network, &baseline_learning_path, &test);
    let validation_correct = evaluate(&trained_network, &training_learning_path, &validation);
    let test_correct = evaluate(&trained_network, &training_learning_path, &test);
    println!(
        "polaritas 100 group-held-out: baseline={baseline_correct}/{}; validation={validation_correct}/{}; after_feedback={test_correct}/{}",
        test.len(),
        validation.len(),
        test.len()
    );
    assert_eq!(baseline_correct, 0);
    assert_eq!(validation_correct, 50);
    assert_eq!(test_correct, 100);

    drop(language_network);
    cleanup(&semantic_path);
    cleanup(&baseline_learning_path);
    cleanup(&training_learning_path);
}
