use shella::ContinualBenchmarkConfig;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let config = ContinualBenchmarkConfig::standard();
    let result = config.run()?;
    println!(
        "seed,baseline_test_accuracy,task_a_after_phase_one,task_a_after_phase_two,task_b_final,final_test_accuracy,val_80,val_90,val_110,val_120,final_validation_accuracy,forgetting_task_a,accuracy_gain_pp,training_reward"
    );
    for seed in result.seed_results() {
        let validation = seed.validation_accuracy_by_amplitude();
        println!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{:.6}",
            seed.seed(),
            seed.baseline_test_accuracy(),
            seed.task_a_test_after_phase_one(),
            seed.task_a_test_after_phase_two(),
            seed.task_b_test_after_phase_two(),
            seed.final_test_accuracy(),
            validation[0],
            validation[1],
            validation[2],
            validation[3],
            seed.final_validation_accuracy(),
            seed.forgetting_task_a(),
            seed.accuracy_gain_percentage_points(),
            seed.training_reward(),
        );
    }
    Ok(())
}
