use shella::{Candidate, DecisionPolicy, SimError};
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
