use shella::{IrisSample, TrainScaler, evaluate_predictions, stratified_folds, stratified_split};

fn samples() -> Vec<IrisSample> {
    (0..30)
        .map(|index| {
            let label = index / 10;
            IrisSample::new(
                [
                    1.0 + label as f64 + index as f64 * 0.01,
                    2.0 + label as f64,
                    3.0 + label as f64,
                    4.0 + label as f64,
                ],
                label,
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn split_stratified_disjoint_dan_mencakup_semua_sampel() {
    let data = samples();
    let split = stratified_split(&data, 17).unwrap();
    assert_eq!(
        split.train().len() + split.validation().len() + split.test().len(),
        data.len()
    );
    assert!(
        split
            .train()
            .iter()
            .all(|index| !split.validation().contains(index) && !split.test().contains(index))
    );
    assert!(
        split
            .validation()
            .iter()
            .all(|index| !split.test().contains(index))
    );
    for class in 0..3 {
        assert!(
            split
                .train()
                .iter()
                .any(|index| data[*index].label() == class)
        );
        assert!(
            split
                .validation()
                .iter()
                .any(|index| data[*index].label() == class)
        );
        assert!(
            split
                .test()
                .iter()
                .any(|index| data[*index].label() == class)
        );
    }
}

#[test]
fn scaler_hanya_fit_pada_train_dan_cv_stratified() {
    let data = samples();
    let split = stratified_split(&data, 31).unwrap();
    let scaler = TrainScaler::fit(&data, split.train()).unwrap();
    let transformed = scaler.transform(data[split.test()[0]].features());
    assert!(transformed.iter().all(|value| value.is_finite()));
    let mut pool = split.train().to_vec();
    pool.extend_from_slice(split.validation());
    let folds = stratified_folds(&data, &pool, 5).unwrap();
    assert_eq!(folds.len(), 5);
    for (train, validation) in folds {
        assert!(train.iter().all(|index| !validation.contains(index)));
        assert_eq!(train.len() + validation.len(), pool.len());
    }
}

#[test]
fn self_evaluation_menghitung_kesalahan_dan_ketidakpastian() {
    let evaluation =
        evaluate_predictions(&[(0, Some(0), 0.8), (1, Some(0), 0.1), (2, None, 0.05)]).unwrap();
    assert_eq!(evaluation.weak_samples(), 2);
    assert_eq!(evaluation.uncertainty_samples(), 2);
    assert_eq!(evaluation.metrics().samples(), 3);
    assert_eq!(evaluation.metrics().false_negative_rate(), Some(2.0 / 3.0));
}
