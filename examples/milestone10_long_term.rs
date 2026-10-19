use shella::{CalibrationEvaluator, LongTermLearningBenchmarkConfig};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Benchmark release diperbesar: 12 seed × 4 varian × 16 repetisi/pola.
    let config = LongTermLearningBenchmarkConfig::standard();
    let result = config.run()?;
    println!(
        "seed,variant,restart_ok,memory_after_restart,memory_after_task_b,feedback_errors,conflicts,replayed,replayed_errors,decay_deleted,noise,without_memory_forgetting,with_memory_forgetting,without_accuracy,with_accuracy,with_precision,with_recall,with_fpr,with_fnr,with_abstain"
    );
    for run in result.runs() {
        for split in run.test() {
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                run.seed(),
                run.task_variant(),
                run.checkpoint_restored_identically(),
                run.memory_records_after_restart(),
                run.memory_records_after_task_b(),
                run.feedback_errors(),
                run.detected_conflicts(),
                run.replayed_experiences(),
                run.replayed_error_experiences(),
                run.memory_decay_deleted(),
                split.noise_na(),
                run.forgetting_without_memory(split.noise_na())
                    .unwrap_or(0.0),
                run.forgetting_with_memory(split.noise_na()).unwrap_or(0.0),
                split
                    .stage()
                    .without_memory()
                    .task_a()
                    .accuracy()
                    .unwrap_or(0.0),
                split
                    .stage()
                    .with_memory()
                    .task_a()
                    .accuracy()
                    .unwrap_or(0.0),
                split
                    .stage()
                    .with_memory()
                    .task_a()
                    .precision()
                    .unwrap_or(0.0),
                split.stage().with_memory().task_a().recall().unwrap_or(0.0),
                split
                    .stage()
                    .with_memory()
                    .task_a()
                    .false_positive_rate()
                    .unwrap_or(0.0),
                split
                    .stage()
                    .with_memory()
                    .task_a()
                    .false_negative_rate()
                    .unwrap_or(0.0),
                split
                    .stage()
                    .with_memory()
                    .task_a()
                    .abstention_rate()
                    .unwrap_or(0.0),
            );
        }
    }

    // Confidence hanya dievaluasi melalui API khusus; skor/margin decision tidak dipakai.
    let mut calibration = CalibrationEvaluator::new();
    calibration.record(0.9, true)?;
    calibration.record(0.2, false)?;
    let metrics = calibration.metrics();
    println!(
        "calibration,samples,brier_score,mean_absolute_calibration_error\ncalibration,{},{:.6},{:.6}",
        metrics.samples(),
        metrics.brier_score().unwrap_or(0.0),
        metrics.mean_absolute_calibration_error().unwrap_or(0.0),
    );
    Ok(())
}
