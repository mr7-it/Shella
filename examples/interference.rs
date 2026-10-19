use shella::InterferenceBenchmarkConfig;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let result = InterferenceBenchmarkConfig::standard().run()?;
    println!(
        "seed,variant,initialization_fingerprint,old_after_phase_one,old_without_replay,new_without_replay,old_with_replay,new_with_replay,forgetting_without_replay,forgetting_with_replay"
    );
    for run in result.runs() {
        println!(
            "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            run.seed(),
            run.task_variant(),
            run.initialization_fingerprint(),
            run.old_task_after_phase_one(),
            run.old_task_without_replay(),
            run.new_task_without_replay(),
            run.old_task_with_replay(),
            run.new_task_with_replay(),
            run.forgetting_without_replay(),
            run.forgetting_with_replay(),
        );
    }
    println!(
        "mean,{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        result.mean_forgetting_without_replay(),
        result.mean_forgetting_with_replay(),
        result.mean_old_task_without_replay(),
        result.mean_new_task_without_replay(),
        result.mean_old_task_with_replay(),
        result.mean_new_task_with_replay(),
    );
    Ok(())
}
