use shella::{ContinualReplayBenchmarkConfig, DecisionMetrics};
use std::error::Error;

fn print_metrics(
    seed: u64,
    variant: usize,
    split: &str,
    method: &str,
    task: &str,
    noise_na: u64,
    metrics: DecisionMetrics,
) {
    println!(
        "{seed},{variant},{split},{method},{task},{noise_na},{},{:.6},{},{},{},{:.6},{:.6},{:.6},{},{}",
        metrics.samples(),
        metrics.accuracy().unwrap_or(0.0),
        metrics.false_positives(),
        metrics.false_negatives(),
        metrics.true_positives(),
        metrics.false_positive_rate().unwrap_or(0.0),
        metrics.false_negative_rate().unwrap_or(0.0),
        metrics.abstention_rate().unwrap_or(0.0),
        metrics.abstentions(),
        metrics.correct_abstentions(),
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = ContinualReplayBenchmarkConfig::standard();
    let result = config.run()?;
    println!(
        "seed,variant,split,method,task,noise_na,samples,accuracy,fp,fn,tp,fpr,fnr,abstain_rate,abstain_count,correct_abstain"
    );
    for run in result.runs() {
        let validation = run.validation();
        for (method, task_metrics) in [
            ("after_task_a", validation.after_task_a()),
            ("without_replay", validation.without_replay()),
            ("with_replay", validation.with_replay()),
        ] {
            print_metrics(
                run.seed(),
                run.task_variant(),
                "validation",
                method,
                "task_a",
                10,
                task_metrics.task_a(),
            );
            print_metrics(
                run.seed(),
                run.task_variant(),
                "validation",
                method,
                "task_b",
                10,
                task_metrics.task_b(),
            );
        }
        for split in run.test() {
            let noise = split.noise_na();
            for (method, task_metrics) in [
                ("before_learning", split.before_learning()),
                ("after_task_a", split.after_task_a()),
                ("without_replay", split.without_replay()),
                ("with_replay", split.with_replay()),
            ] {
                print_metrics(
                    run.seed(),
                    run.task_variant(),
                    "test",
                    method,
                    "task_a",
                    noise,
                    task_metrics.task_a(),
                );
                print_metrics(
                    run.seed(),
                    run.task_variant(),
                    "test",
                    method,
                    "task_b",
                    noise,
                    task_metrics.task_b(),
                );
            }
        }
        println!(
            "budget,{},{},{},presentations_per_method,{},training_steps_per_method,{}",
            run.seed(),
            run.task_variant(),
            run.initialization_fingerprint(),
            run.training_presentations_per_method(),
            run.training_steps_per_method(),
        );
    }
    println!(
        "aggregate,noise_na,forgetting_without,forgetting_with,catastrophic_rate_without,catastrophic_rate_with,forgetting_sd_without,forgetting_sd_with"
    );
    for noise in config.test_noise_levels_na() {
        println!(
            "aggregate,{noise:.0},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            result
                .mean_forgetting_without_replay(*noise as u64)
                .unwrap_or(0.0),
            result
                .mean_forgetting_with_replay(*noise as u64)
                .unwrap_or(0.0),
            result
                .catastrophic_forgetting_rate_without_replay(*noise as u64)
                .unwrap_or(0.0),
            result
                .catastrophic_forgetting_rate_with_replay(*noise as u64)
                .unwrap_or(0.0),
            result
                .forgetting_stddev_without_replay(*noise as u64)
                .unwrap_or(0.0),
            result
                .forgetting_stddev_with_replay(*noise as u64)
                .unwrap_or(0.0),
        );
    }
    Ok(())
}
