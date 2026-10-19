use shella::GeneralizationBenchmarkConfig;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let config = GeneralizationBenchmarkConfig::standard();
    let result = config.run()?;
    println!(
        "seed,variant,initialization_fingerprint,accuracy_noise_0,accuracy_noise_60,fp_rate_noise_60,fn_rate_noise_60,abstain_noise_60"
    );
    for run in result.runs() {
        let clean = run
            .trained()
            .iter()
            .find(|condition| condition.noise_na() == 0.0)
            .expect("noise nol tersedia")
            .metrics();
        let noisy = run
            .trained()
            .iter()
            .find(|condition| condition.noise_na() == 60.0)
            .expect("noise 60 tersedia")
            .metrics();
        println!(
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            run.seed(),
            run.task_variant(),
            run.initialization_fingerprint(),
            clean.accuracy().unwrap_or(0.0),
            noisy.accuracy().unwrap_or(0.0),
            noisy.false_positive_rate().unwrap_or(0.0),
            noisy.false_negative_rate().unwrap_or(0.0),
            noisy.abstention_rate().unwrap_or(0.0),
        );
    }
    println!(
        "noise_na,baseline_accuracy,trained_accuracy,trained_fp_rate,trained_fn_rate,trained_abstain_rate"
    );
    for noise in config.noise_levels_na() {
        println!(
            "{noise:.0},{:.6},{:.6},{:.6},{:.6},{:.6}",
            result.mean_baseline_accuracy(*noise).unwrap_or(0.0),
            result.mean_trained_accuracy(*noise).unwrap_or(0.0),
            result.mean_false_positive_rate(*noise).unwrap_or(0.0),
            result.mean_false_negative_rate(*noise).unwrap_or(0.0),
            result.mean_abstention_rate(*noise).unwrap_or(0.0),
        );
    }
    Ok(())
}
