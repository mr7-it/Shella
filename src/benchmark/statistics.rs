use crate::RewardMetrics;

/// Selisih akurasi jaringan learned terhadap baseline, jika kedua set memiliki sampel.
pub(super) fn accuracy_gain(baseline: RewardMetrics, learned: RewardMetrics) -> Option<f64> {
    Some(learned.accuracy()? - baseline.accuracy()?)
}
