use serde_json::json;
use shella::{
    Checkpoint, LearningMemory, LifConfig, LifNetwork, MemoryStore, ReasoningConfig,
    ReasoningEngine, Synapse,
};
use std::{
    error::Error,
    fs,
    time::{Duration, Instant},
};

const BATCHES: usize = 5;
const NETWORK_STEPS_PER_BATCH: usize = 500;
const IO_OPS_PER_BATCH: usize = 20;

fn main() -> Result<(), Box<dyn Error>> {
    println!(
        "benchmark Shella: median {} batch; sinkron, tanpa RNG; bukan tes kebenaran",
        BATCHES
    );
    println!("benchmark,neurons,synapses,ops_per_batch,median_ns_per_op,throughput_ops_per_sec");
    for neuron_count in [16, 64, 256] {
        let mut network = spiking_ring(neuron_count)?;
        let currents = vec![100.0; neuron_count];
        let time = median_per_operation(BATCHES, NETWORK_STEPS_PER_BATCH, || {
            network.step(&currents)?;
            Ok::<_, shella::SimError>(())
        })?;
        println!(
            "network_step,{neuron_count},{neuron_count},{NETWORK_STEPS_PER_BATCH},{time:.0},{:.0}",
            1.0e9 / time
        );
    }

    let directory = std::env::temp_dir().join(format!("shella-bench-{}", std::process::id()));
    fs::create_dir_all(&directory)?;
    let checkpoint_path = directory.join("state.json");
    let network = spiking_ring(64)?;
    let save_time = median_per_operation(BATCHES, IO_OPS_PER_BATCH, || {
        Checkpoint::save(&checkpoint_path, &network, None)?;
        Ok::<_, shella::SimError>(())
    })?;
    let restore_time = median_per_operation(BATCHES, IO_OPS_PER_BATCH, || {
        std::hint::black_box(Checkpoint::restore(&checkpoint_path)?);
        Ok::<_, shella::SimError>(())
    })?;
    let checkpoint_bytes = fs::metadata(&checkpoint_path)?.len();
    println!(
        "checkpoint_save,64,64,{IO_OPS_PER_BATCH},{save_time:.0},{:.0}",
        1.0e9 / save_time
    );
    println!(
        "checkpoint_restore,64,64,{IO_OPS_PER_BATCH},{restore_time:.0},{:.0}",
        1.0e9 / restore_time
    );
    println!("# checkpoint_bytes={checkpoint_bytes}");

    let memory_path = directory.join("memory.json");
    let mut memory = MemoryStore::open(&memory_path)?;
    let tags = vec!["benchmark".to_owned(), "topic".to_owned()];
    for index in 0..256 {
        memory.put(
            format!("topic/{index:04}"),
            json!({"index":index,"active":index % 2 == 0}),
            &tags,
        )?;
    }
    let records = memory.len();
    let retrieval_time = median_per_operation(BATCHES, NETWORK_STEPS_PER_BATCH, || {
        let found = memory.search_prefix("topic/");
        std::hint::black_box(found.len());
        Ok::<_, shella::SimError>(())
    })?;
    println!(
        "memory_prefix_search,{records},0,{NETWORK_STEPS_PER_BATCH},{retrieval_time:.0},{:.0}",
        1.0e9 / retrieval_time
    );
    let tag_time = median_per_operation(BATCHES, NETWORK_STEPS_PER_BATCH, || {
        let found = memory.search_tag("benchmark");
        std::hint::black_box(found.len());
        Ok::<_, shella::SimError>(())
    })?;
    println!(
        "memory_tag_search,{records},0,{NETWORK_STEPS_PER_BATCH},{tag_time:.0},{:.0}",
        1.0e9 / tag_time
    );

    let learning_path = directory.join("learning.json");
    let mut learning_memory = LearningMemory::open(&learning_path)?;
    let reasoning_network =
        LifNetwork::new(LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1), 16)?;
    for index in 0..16 {
        let mut features = vec![0.0; 16];
        features[index] = 1.0;
        learning_memory.record_feedback(
            "benchmark",
            &features,
            Some(index % 4),
            Some(index % 4),
        )?;
    }
    let mut reasoning = ReasoningEngine::new(
        reasoning_network,
        learning_memory,
        ReasoningConfig::new(3, 8, 100.0, 0.05, 0.0)?,
    );
    let query = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let action_neurons = [12, 13, 14, 15];
    let reasoning_time = median_per_operation(BATCHES, NETWORK_STEPS_PER_BATCH, || {
        let result = reasoning.reason("benchmark", &query, &action_neurons)?;
        std::hint::black_box(result.confidence());
        Ok::<_, shella::SimError>(())
    })?;
    println!(
        "reasoning_engine,16,0,{NETWORK_STEPS_PER_BATCH},{reasoning_time:.0},{:.0}",
        1.0e9 / reasoning_time
    );
    fs::remove_dir_all(directory)?;
    Ok(())
}

fn spiking_ring(neuron_count: usize) -> Result<LifNetwork, shella::SimError> {
    let mut network = LifNetwork::new(
        LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 0),
        neuron_count,
    )?;
    for source in 0..neuron_count {
        let target = (source + 1) % neuron_count;
        network.add_synapse(Synapse::new(source, target, 0.1, 0.0, 1.0, 1)?)?;
    }
    Ok(network)
}

fn median_per_operation<E>(
    batches: usize,
    operations: usize,
    mut operation: impl FnMut() -> Result<(), E>,
) -> Result<f64, E> {
    let mut times = Vec::with_capacity(batches);
    for _ in 0..batches {
        let start = Instant::now();
        for _ in 0..operations {
            operation()?;
        }
        times.push(start.elapsed());
    }
    times.sort_unstable();
    let median: Duration = times[times.len() / 2];
    Ok(median.as_nanos() as f64 / operations as f64)
}
