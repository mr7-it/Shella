use shella::ConditioningBenchmarkConfig;
use std::{
    error::Error,
    time::{Duration, Instant},
};

const BATCHES: usize = 7;
const ITERATIONS_PER_BATCH: usize = 100;

fn main() -> Result<(), Box<dyn Error>> {
    let config = ConditioningBenchmarkConfig::standard();
    let expected = config.run()?;
    let baseline_accuracy = expected
        .baseline_metrics()
        .accuracy()
        .ok_or("evaluasi baseline tidak memiliki sampel")?;
    let learned_accuracy = expected
        .learned_metrics()
        .accuracy()
        .ok_or("evaluasi learned tidak memiliki sampel")?;
    if learned_accuracy <= baseline_accuracy {
        return Err(std::io::Error::other(format!(
            "reward learning tidak mengungguli baseline: learned={learned_accuracy}, baseline={baseline_accuracy}"
        ))
        .into());
    }

    let mut batch_times_ns = Vec::with_capacity(BATCHES);
    let mut total_elapsed = Duration::ZERO;
    for _ in 0..BATCHES {
        let start = Instant::now();
        for _ in 0..ITERATIONS_PER_BATCH {
            let result = config.run()?;
            if result.baseline_metrics() != expected.baseline_metrics()
                || result.learned_metrics() != expected.learned_metrics()
                || result.learned_weights() != expected.learned_weights()
            {
                return Err(std::io::Error::other(
                    "hasil benchmark berubah antar-run pada protokol deterministik",
                )
                .into());
            }
        }
        let elapsed = start.elapsed();
        total_elapsed += elapsed;
        batch_times_ns.push(elapsed.as_nanos() / ITERATIONS_PER_BATCH as u128);
    }
    batch_times_ns.sort_unstable();
    let median_ns_per_run = batch_times_ns[BATCHES / 2];
    let gain_percentage_points = expected
        .accuracy_gain()
        .ok_or("hasil tidak memiliki sampel untuk menghitung gain")?
        * 100.0;
    let baseline = expected.baseline_metrics();
    let learned = expected.learned_metrics();
    let weights = expected.learned_weights();

    println!(
        "Benchmark conditioning LIF/STDP; pola tugas eksplisit dan deterministik, tanpa RNG atau dataset eksternal."
    );
    println!(
        "benchmark,batches,iterations_per_batch,epochs,eval_repetitions,baseline_correct,baseline_samples,baseline_accuracy,learned_correct,learned_samples,learned_accuracy,gain_pp,weight_cue_approach_na,weight_cue_withhold_na,training_steps_per_network,total_measured_ms,median_ns_per_run"
    );
    println!(
        "conditioning,{BATCHES},{ITERATIONS_PER_BATCH},{},{},{},{},{baseline_accuracy:.6},{},{},{learned_accuracy:.6},{gain_percentage_points:.3},{:.6},{:.6},{},{:.3},{median_ns_per_run}",
        config.training_epochs(),
        config.evaluation_repetitions(),
        baseline.correct(),
        baseline.samples(),
        learned.correct(),
        learned.samples(),
        weights[0],
        weights[1],
        expected.training_steps(),
        total_elapsed.as_secs_f64() * 1_000.0,
    );
    Ok(())
}
