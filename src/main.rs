use shella::{
    LifConfig, LifNetwork, LifRuntime, RewardEvaluator, RewardEvaluatorConfig,
    RewardModulatedStdpConfig, StdpConfig, Synapse,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let config = LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 2);
    let mut runtime = LifRuntime::new(config)?;

    println!("Demo neuron LIF tunggal: arus 10 nA, dt 1 ms");
    println!("langkah\tpotensial (mV)\tspike");
    for _ in 0..12 {
        let outcome = runtime.step(10.0)?;
        println!(
            "{}\t{:.6}\t{}",
            outcome.completed_step, outcome.membrane_potential_mv, outcome.spiked
        );
    }
    println!(
        "indeks spike neuron tunggal: {:?}",
        runtime.neuron().spike_steps()
    );

    let neuron_configs = [
        config,
        LifConfig::new(1.0, 5.0, 1.0, -65.0, -70.0, -55.0, 0),
    ];
    let mut network = LifNetwork::from_configs(&neuron_configs)?;
    network.add_synapse(Synapse::new(0, 1, 20.0, 0.0, 40.0, 1)?)?;
    let learning = RewardModulatedStdpConfig::new(StdpConfig::new(1.0, 1.0, 20.0, 20.0), 50.0, 0.1);
    network.enable_reward_modulated_stdp(learning)?;
    let mut evaluator = RewardEvaluator::new(RewardEvaluatorConfig::new(1.0, 1.0))?;

    println!("\nDemo jaringan heterogen dengan evaluator reward otomatis");
    for external in [[100.0, 0.0], [0.0, 100.0]] {
        let result = network.step(&external)?;
        println!(
            "langkah {}: spike {:?}, bobot sebelum evaluasi {:.6} nA",
            result.completed_step(),
            result.spiking_neurons(),
            network.synapses()[0].weight_na()
        );
        if result.completed_step() == 2 {
            let observed_action = [result.spiking_neurons().contains(&1)];
            let evaluation =
                evaluator.evaluate_and_apply(Some(0), &observed_action, &mut network)?;
            println!(
                "target {:?}, prediksi {:?}, benar {}, reward {:+.1}",
                evaluation.expected_action(),
                evaluation.predicted_action(),
                evaluation.correct(),
                evaluation.reward()
            );
        }
    }
    println!(
        "setelah evaluasi otomatis: bobot {:.6} nA, akurasi {:.1}%",
        network.synapses()[0].weight_na(),
        evaluator.metrics().accuracy().unwrap_or(0.0) * 100.0,
    );
    Ok(())
}
