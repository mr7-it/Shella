use shella::{AbstentionReason, Candidate, DecisionEvaluator, DecisionPolicy, SimError};
use std::collections::BTreeMap;

fn candidate(id: &str, values: &[(&str, f64)]) -> Candidate {
    Candidate::new(
        id,
        values
            .iter()
            .map(|(key, value)| ((*key).to_owned(), *value))
            .collect(),
    )
}
fn policy(minimum_score: f64, margin: f64) -> DecisionPolicy {
    DecisionPolicy::new(
        BTreeMap::from([("spike".to_owned(), 2.0), ("reward".to_owned(), 0.5)]),
        minimum_score,
        margin,
    )
    .expect("policy sah")
}

#[test]
fn decision_menghitung_kontribusi_aktual_dan_memilih_skor_tertinggi() {
    let chosen = policy(0.0, 0.0)
        .select(&[
            candidate("move", &[("spike", 1.0), ("reward", 2.0)]),
            candidate("wait", &[("spike", 0.5), ("reward", 0.0)]),
        ])
        .expect("decision valid");
    assert_eq!(chosen.selected_id(), Some("move"));
    assert_eq!(chosen.best_score(), Some(3.0));
    assert_eq!(chosen.runner_up_score(), Some(1.0));
    assert_eq!(chosen.score_margin(), Some(2.0));
    assert_eq!(
        chosen
            .contributions()
            .iter()
            .map(|item| item.contribution)
            .sum::<f64>(),
        3.0
    );
    assert!(chosen.reason().contains("skor linear"));
}

#[test]
fn decision_tie_break_berdasarkan_id_dan_dapat_abstain() {
    let tied = policy(-1.0, 0.0)
        .select(&[
            candidate("z", &[("spike", 1.0)]),
            candidate("a", &[("spike", 1.0)]),
        ])
        .expect("seri valid");
    assert_eq!(tied.selected_id(), Some("a"));
    let abstain_score = policy(5.0, 0.0)
        .select(&[candidate("a", &[("spike", 1.0)])])
        .expect("seleksi");
    assert_eq!(abstain_score.selected_id(), None);
    let abstain_margin = policy(-10.0, 3.0)
        .select(&[
            candidate("a", &[("spike", 2.0)]),
            candidate("b", &[("spike", 1.5)]),
        ])
        .expect("seleksi");
    assert_eq!(abstain_margin.selected_id(), None);
    assert!(
        policy(-10.0, 0.0)
            .select(&[])
            .expect("kandidat kosong")
            .selected_id()
            .is_none()
    );
}

#[test]
fn decision_menolak_candidate_invalid_unknown_feature_duplicate_id_dan_overflow() {
    assert!(DecisionPolicy::new(BTreeMap::new(), 0.0, 0.0).is_err());
    assert!(DecisionPolicy::new(BTreeMap::from([("x".into(), f64::NAN)]), 0.0, 0.0).is_err());
    assert!(matches!(
        policy(0.0, 0.0).select(&[candidate("", &[])]),
        Err(SimError::InvalidInput { .. })
    ));
    assert!(matches!(
        policy(0.0, 0.0).select(&[candidate("a", &[("unknown", 1.0)])]),
        Err(SimError::InvalidInput { .. })
    ));
    assert!(matches!(
        policy(0.0, 0.0).select(&[candidate("a", &[("spike", f64::INFINITY)])]),
        Err(SimError::InvalidInput { .. })
    ));
    assert!(matches!(
        policy(0.0, 0.0).select(&[candidate("same", &[]), candidate("same", &[])]),
        Err(SimError::InvalidInput { .. })
    ));
    let huge = DecisionPolicy::new(BTreeMap::from([("x".into(), f64::MAX)]), 0.0, 0.0)
        .expect("finite weight");
    assert!(matches!(
        huge.select(&[candidate("a", &[("x", 2.0)])]),
        Err(SimError::NumericalOverflow { .. })
    ));
    assert!(matches!(
        huge.select(&[
            candidate("positive", &[("x", 1.0)]),
            candidate("negative", &[("x", -1.0)]),
        ]),
        Err(SimError::NumericalOverflow { .. })
    ));
}

#[test]
fn metrik_keputusan_mengukur_false_positive_false_negative_dan_abstain() {
    let mut evaluator = DecisionEvaluator::new();
    evaluator
        .record(Some("move"), Some("move"), None)
        .expect("TP");
    evaluator
        .record(Some("move"), Some("wait"), None)
        .expect("FP dan FN");
    evaluator
        .record(
            Some("move"),
            None,
            Some(AbstentionReason::BelowMinimumMargin),
        )
        .expect("FN karena abstain");
    evaluator
        .record(None, None, Some(AbstentionReason::NoCandidates))
        .expect("abstain benar untuk target diam");
    evaluator
        .record(None, Some("wait"), None)
        .expect("false positive");

    let metrics = evaluator.metrics();
    assert_eq!(metrics.samples(), 5);
    assert_eq!(metrics.true_positives(), 1);
    assert_eq!(metrics.false_positives(), 2);
    assert_eq!(metrics.false_negatives(), 2);
    assert_eq!(metrics.abstentions(), 2);
    assert_eq!(metrics.correct_abstentions(), 1);
    assert_eq!(metrics.false_abstentions(), 1);
    assert_eq!(metrics.below_minimum_margin(), 1);
    assert_eq!(metrics.no_candidates(), 1);
    assert_eq!(metrics.precision(), Some(1.0 / 3.0));
    assert_eq!(metrics.recall(), Some(1.0 / 3.0));
    assert_eq!(metrics.false_positive_rate(), Some(0.5));
    assert_eq!(metrics.false_negative_rate(), Some(2.0 / 3.0));
}

#[test]
fn decision_menyediakan_alasan_abstain_yang_terstruktur() {
    let no_candidates = policy(0.0, 0.0).select(&[]).expect("kosong");
    assert_eq!(
        no_candidates.abstention_reason(),
        Some(AbstentionReason::NoCandidates)
    );
    let low_score = policy(5.0, 0.0)
        .select(&[candidate("move", &[("spike", 1.0)])])
        .expect("skor rendah");
    assert_eq!(
        low_score.abstention_reason(),
        Some(AbstentionReason::BelowMinimumScore)
    );
    let low_margin = policy(-10.0, 3.0)
        .select(&[
            candidate("move", &[("spike", 2.0)]),
            candidate("wait", &[("spike", 1.5)]),
        ])
        .expect("margin rendah");
    assert_eq!(
        low_margin.abstention_reason(),
        Some(AbstentionReason::BelowMinimumMargin)
    );
}

#[test]
fn evaluator_menghitung_abstain_sesuai_label_dan_alasan_policy() {
    let mut evaluator = DecisionEvaluator::new();
    let low_score = policy(5.0, 0.0)
        .select(&[candidate("move", &[("spike", 1.0)])])
        .expect("seleksi skor rendah");
    evaluator
        .evaluate_decision(Some("move"), &low_score)
        .expect("abstain pada target aksi");

    let no_candidates = policy(0.0, 0.0).select(&[]).expect("tanpa kandidat");
    evaluator
        .evaluate_decision(None, &no_candidates)
        .expect("diam memang target");

    let metrics = evaluator.metrics();
    assert_eq!(metrics.false_negatives(), 1);
    assert_eq!(metrics.false_abstentions(), 1);
    assert_eq!(metrics.correct_abstentions(), 1);
    assert_eq!(metrics.below_minimum_score(), 1);
    assert_eq!(metrics.no_candidates(), 1);
}
