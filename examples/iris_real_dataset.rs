use shella::{DecisionEvaluator, LearningMemory};
use std::{
    error::Error,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static ID: AtomicU64 = AtomicU64::new(0);
const FEATURES: usize = 4;

fn main() -> Result<(), Box<dyn Error>> {
    let rows = fs::read_to_string("data/iris_uci.csv")?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(parse_row)
        .collect::<Result<Vec<_>, _>>()?;
    if rows.len() != 150 {
        return Err("dataset UCI Iris harus berisi 150 baris".into());
    }
    let (minima, maxima) = fit_scaler(&rows);
    let path = std::env::temp_dir().join(format!(
        "shella-iris-{}-{}.json",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let mut memory = LearningMemory::open(&path)?;
    let mut train = 0;
    let mut test = 0;
    for (index, (features, label)) in rows.iter().enumerate() {
        let scaled = scale(features, &minima, &maxima);
        if index % 5 == 0 {
            test += 1;
        } else {
            memory.record_with_feedback("uci-iris", &scaled, Some(*label), 1.0, 1.0)?;
            train += 1;
        }
    }
    let clean = evaluate(&memory, &rows, &minima, &maxima, 0.0)?;
    let noisy = evaluate(&memory, &rows, &minima, &maxima, 0.35)?;
    println!("dataset,train,test,noise,accuracy,precision,recall,fpr,fnr,abstain");
    println!(
        "uci-iris,{train},{test},0,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        clean.accuracy().unwrap_or(0.0),
        clean.precision().unwrap_or(0.0),
        clean.recall().unwrap_or(0.0),
        clean.false_positive_rate().unwrap_or(0.0),
        clean.false_negative_rate().unwrap_or(0.0),
        clean.abstention_rate().unwrap_or(0.0)
    );
    println!(
        "uci-iris,{train},{test},0.35,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        noisy.accuracy().unwrap_or(0.0),
        noisy.precision().unwrap_or(0.0),
        noisy.recall().unwrap_or(0.0),
        noisy.false_positive_rate().unwrap_or(0.0),
        noisy.false_negative_rate().unwrap_or(0.0),
        noisy.abstention_rate().unwrap_or(0.0)
    );
    drop(memory);
    cleanup(&path);
    Ok(())
}

fn evaluate(
    memory: &LearningMemory,
    rows: &[(Vec<f64>, usize)],
    minima: &[f64],
    maxima: &[f64],
    noise: f64,
) -> Result<shella::DecisionMetrics, Box<dyn Error>> {
    let mut evaluator = DecisionEvaluator::new();
    for (index, (features, expected)) in rows.iter().enumerate().filter(|(index, _)| index % 5 == 0)
    {
        let mut query = scale(features, minima, maxima);
        for (position, value) in query.iter_mut().enumerate() {
            *value = (*value + signed_noise(index as u64 ^ 0x9e37_79b9, position as u64, noise))
                .clamp(0.0, 1.0);
        }
        let recalls = memory.recall_similar(&query, 5)?;
        let mut votes = [0.0_f64; 3];
        for recall in recalls {
            if let Some(action) = recall.expected_action() {
                votes[action] += recall.retrieval_score().max(0.0);
            }
        }
        let predicted = votes
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .and_then(|(action, score)| (*score > 0.0).then_some(format!("kelas_{action}")));
        let expected_name = format!("kelas_{expected}");
        evaluator.record(Some(&expected_name), predicted.as_deref(), None)?;
    }
    Ok(evaluator.metrics())
}

fn parse_row(line: &str) -> Result<(Vec<f64>, usize), Box<dyn Error>> {
    let fields = line.split(',').collect::<Vec<_>>();
    if fields.len() != 5 {
        return Err("baris Iris memiliki 4 fitur dan 1 label".into());
    }
    let features = fields[..FEATURES]
        .iter()
        .map(|value| value.parse::<f64>().map_err(|error| error.into()))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let label = match fields[4] {
        "Iris-setosa" => 0,
        "Iris-versicolor" => 1,
        "Iris-virginica" => 2,
        other => return Err(format!("label tidak dikenal: {other}").into()),
    };
    Ok((features, label))
}
fn fit_scaler(rows: &[(Vec<f64>, usize)]) -> (Vec<f64>, Vec<f64>) {
    let mut min = vec![f64::INFINITY; FEATURES];
    let mut max = vec![f64::NEG_INFINITY; FEATURES];
    for (row, _) in rows.iter().filter(|(row, _)| row.len() == FEATURES) {
        for (i, value) in row.iter().enumerate() {
            min[i] = min[i].min(*value);
            max[i] = max[i].max(*value);
        }
    }
    (min, max)
}
fn scale(row: &[f64], min: &[f64], max: &[f64]) -> Vec<f64> {
    row.iter()
        .enumerate()
        .map(|(i, value)| (value - min[i]) / (max[i] - min[i]))
        .collect()
}
fn signed_noise(index: u64, feature: u64, amplitude: f64) -> f64 {
    let mut state = index.wrapping_add(feature.wrapping_mul(0x9e37_79b9));
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    ((state >> 11) as f64 / ((1_u64 << 53) as f64) * 2.0 - 1.0) * amplitude
}
fn cleanup(path: &PathBuf) {
    let _ = fs::remove_file(path);
    let mut lock = path.as_os_str().to_os_string();
    lock.push(".lock");
    let _ = fs::remove_file(lock);
}
