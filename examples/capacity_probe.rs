use shella::{Checkpoint, LifConfig, LifNetwork, Synapse};
use std::{error::Error, fs, path::PathBuf, time::Instant};

const BATCHES: usize = 5;
const DEFAULT_STEPS_PER_BATCH: usize = 20;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() < 3 || arguments.len() > 4 {
        return Err("usage: capacity_probe <neurons> <synapses> [steps_per_batch]".into());
    }
    let neuron_count = arguments[1].parse::<usize>()?;
    let synapse_count = arguments[2].parse::<usize>()?;
    let steps_per_batch = arguments
        .get(3)
        .map_or(Ok(DEFAULT_STEPS_PER_BATCH), |value| value.parse::<usize>())?;
    if neuron_count == 0 || steps_per_batch == 0 {
        return Err("neurons dan steps_per_batch harus positif".into());
    }
    let maximum_synapses = neuron_count
        .checked_mul(neuron_count.saturating_sub(1))
        .ok_or("jumlah pasangan sinaps overflow")?;
    if synapse_count > maximum_synapses {
        return Err(format!(
            "synapse count {synapse_count} melebihi pasangan berarah unik tanpa self-edge {maximum_synapses}"
        )
        .into());
    }

    let build_start = Instant::now();
    let config = LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 0);
    let mut network = LifNetwork::new(config, neuron_count)?;
    if neuron_count > 1 {
        for index in 0..synapse_count {
            let source = index / (neuron_count - 1);
            let target_offset = index % (neuron_count - 1) + 1;
            let target = (source + target_offset) % neuron_count;
            network.add_synapse(Synapse::new(source, target, 0.1, 0.0, 1.0, 1)?)?;
        }
    }
    let build_ns = build_start.elapsed().as_nanos();
    let rss_after_build_kib = process_memory_kib("VmRSS");

    let currents = vec![100.0; neuron_count];
    let mut batch_times = Vec::with_capacity(BATCHES);
    for _ in 0..BATCHES {
        let start = Instant::now();
        for _ in 0..steps_per_batch {
            network.step(&currents)?;
        }
        batch_times.push(start.elapsed());
    }
    batch_times.sort_unstable();
    let median_step_ns = batch_times[BATCHES / 2].as_nanos() as f64 / steps_per_batch as f64;
    let rss_after_step_kib = process_memory_kib("VmRSS");
    let completed_steps = u64::try_from(BATCHES * steps_per_batch)?;

    let checkpoint_path = std::env::temp_dir().join(format!(
        "shella-capacity-{}-{}-{}.json",
        std::process::id(),
        neuron_count,
        synapse_count
    ));
    let save_start = Instant::now();
    Checkpoint::save(&checkpoint_path, &network, None)?;
    let checkpoint_save_ns = save_start.elapsed().as_nanos();
    let checkpoint_bytes = fs::metadata(&checkpoint_path)?.len();
    let rss_after_checkpoint_kib = process_memory_kib("VmRSS");
    let peak_rss_kib = process_memory_kib("VmHWM");
    fs::remove_file(checkpoint_path)?;

    println!(
        "{}",
        serde_json::json!({
            "neurons": neuron_count,
            "synapses": synapse_count,
            "build_ns": build_ns,
            "steps_measured": completed_steps,
            "steps_per_batch": steps_per_batch,
            "batches": BATCHES,
            "median_step_ns": median_step_ns,
            "checkpoint_save_ns": checkpoint_save_ns,
            "checkpoint_bytes": checkpoint_bytes,
            "rss_after_build_kib": rss_after_build_kib,
            "rss_after_step_kib": rss_after_step_kib,
            "rss_after_checkpoint_kib": rss_after_checkpoint_kib,
            "peak_rss_kib": peak_rss_kib,
        })
    );
    Ok(())
}

fn process_memory_kib(field: &str) -> Option<u64> {
    let status = fs::read_to_string(PathBuf::from("/proc/self/status")).ok()?;
    status.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name != field {
            return None;
        }
        value.split_whitespace().next()?.parse().ok()
    })
}
