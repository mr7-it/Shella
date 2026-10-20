use super::{FEATURE_COUNT, FeatureStatistics};
use crate::SimError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prototype {
    pub prototype_id: String,
    pub class_id: usize,
    pub feature_vector: [f64; FEATURE_COUNT],
    pub member_count: usize,
    pub statistics: FeatureStatistics,
    pub embedding: [f64; FEATURE_COUNT],
    pub creation_version: u32,
    pub update_version: u32,
    pub importance: f64,
    pub usage_count: u64,
    pub last_retrieved_at_unix: Option<u64>,
}

pub(super) fn fit_kmeans(
    rows: &[[f64; FEATURE_COUNT]],
    cluster_count: usize,
    seed: u64,
    class_id: usize,
) -> Result<Vec<Prototype>, SimError> {
    if cluster_count == 0 || rows.len() < cluster_count {
        return Err(SimError::InvalidInput {
            parameter: "prototype_count",
            reason: "jumlah prototype harus positif dan tidak melebihi data training kelas",
        });
    }
    let mut centers = initialize_centers(rows, cluster_count, seed)?;
    let mut assignments = vec![0; rows.len()];
    for _ in 0..100 {
        let mut changed = false;
        let mut members = vec![Vec::<usize>::new(); cluster_count];
        for (index, row) in rows.iter().enumerate() {
            let assignment = nearest_center(row, &centers);
            changed |= assignments[index] != assignment;
            assignments[index] = assignment;
            members[assignment].push(index);
        }
        let mut next_centers = centers.clone();
        for (cluster, indices) in members.iter().enumerate() {
            if indices.is_empty() {
                continue;
            }
            for feature in 0..FEATURE_COUNT {
                next_centers[cluster][feature] = indices
                    .iter()
                    .map(|index| rows[*index][feature])
                    .sum::<f64>()
                    / indices.len() as f64;
            }
        }
        centers = next_centers;
        if !changed {
            break;
        }
    }
    let mut members = vec![Vec::<usize>::new(); cluster_count];
    for (index, row) in rows.iter().enumerate() {
        members[nearest_center(row, &centers)].push(index);
    }
    let mut prototypes = Vec::with_capacity(cluster_count);
    for (cluster, indices) in members.into_iter().enumerate() {
        if indices.is_empty() {
            return Err(SimError::InvalidInput {
                parameter: "prototype_count",
                reason: "clustering menghasilkan cluster kosong",
            });
        }
        let cluster_rows = indices.iter().map(|index| rows[*index]).collect::<Vec<_>>();
        let importance = indices.len() as f64 / rows.len() as f64;
        prototypes.push(Prototype {
            prototype_id: format!("class-{class_id}-prototype-{cluster}"),
            class_id,
            feature_vector: centers[cluster],
            member_count: indices.len(),
            statistics: FeatureStatistics::compute(&cluster_rows),
            embedding: centers[cluster],
            creation_version: 1,
            update_version: 1,
            importance,
            usage_count: 0,
            last_retrieved_at_unix: None,
        });
    }
    Ok(prototypes)
}

pub(super) fn squared_distance(left: &[f64; FEATURE_COUNT], right: &[f64; FEATURE_COUNT]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum()
}

fn initialize_centers(
    rows: &[[f64; FEATURE_COUNT]],
    cluster_count: usize,
    seed: u64,
) -> Result<Vec<[f64; FEATURE_COUNT]>, SimError> {
    let first = (seed as usize) % rows.len();
    let mut centers = vec![rows[first]];
    while centers.len() < cluster_count {
        let candidate = rows
            .iter()
            .enumerate()
            .filter(|(_, row)| !centers.contains(row))
            .map(|(index, row)| {
                let distance = centers
                    .iter()
                    .map(|center| squared_distance(row, center))
                    .fold(f64::INFINITY, f64::min);
                (index, distance)
            })
            .max_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map(|(index, _)| index)
            .ok_or(SimError::InvalidInput {
                parameter: "prototype_count",
                reason: "data unik tidak cukup untuk jumlah prototype yang diminta",
            })?;
        centers.push(rows[candidate]);
    }
    Ok(centers)
}

fn nearest_center(row: &[f64; FEATURE_COUNT], centers: &[[f64; FEATURE_COUNT]]) -> usize {
    centers
        .iter()
        .enumerate()
        .min_by(|left, right| {
            squared_distance(row, left.1)
                .total_cmp(&squared_distance(row, right.1))
                .then_with(|| left.0.cmp(&right.0))
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}
