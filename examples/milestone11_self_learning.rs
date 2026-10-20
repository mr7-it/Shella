use shella::{
    CalibrationEvaluator, ClassKnowledgeModel, DecisionMetrics, IrisSample, LearningMemory,
    LifConfig, LifNetwork, RewardModulatedStdpConfig, SimError, StdpConfig, Synapse,
    evaluate_predictions, stratified_folds, stratified_split,
};
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const INPUTS: usize = 4;
const ACTIONS: [usize; 3] = [4, 5, 6];
const NEURONS: usize = 7;
const SEEDS: [u64; 5] = [17, 31, 47, 73, 101];
const MEMORY_CURRENTS: [f64; 3] = [80.0, 140.0, 220.0];
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy)]
struct Scale {
    min: [f64; INPUTS],
    max: [f64; INPUTS],
}
impl Scale {
    fn apply(&self, x: &[f64; INPUTS]) -> [f64; INPUTS] {
        std::array::from_fn(|i| (x[i] - self.min[i]) / (self.max[i] - self.min[i]))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let samples = load_iris("data/iris_uci.csv")?;
    let mut prototype_validation_csv =
        String::from("seed,prototypes_per_class,validation_accuracy,selected\n");
    let mut calibration_csv = String::from("seed,split,samples,brier_score,mean_absolute_error\n");
    let mut class_metrics_csv = String::from(
        "seed,class_id,class_name,tp,fp,fn,tn,accuracy,precision,recall,f1,fpr,fnr,predicted_0,predicted_1,predicted_2\n",
    );
    println!(
        "milestone,seed,mode,split,noise,accuracy,precision,recall,fpr,fnr,abstain,forgetting,weak,uncertainty,memory_records,cv_accuracy,replay_presentations"
    );
    for seed in SEEDS {
        let split = stratified_split(&samples, seed)?;
        let scaler = fit_scale(&samples, split.train())?;
        let training = split
            .train()
            .iter()
            .map(|index| samples[*index].clone())
            .collect::<Vec<_>>();
        let validation = split
            .validation()
            .iter()
            .map(|index| samples[*index].clone())
            .collect::<Vec<_>>();
        let mut class_knowledge = ClassKnowledgeModel::fit(
            &training,
            &validation,
            "iris-uci-v1",
            format!("m11-{seed}"),
            seed,
        )?;
        for candidate in &class_knowledge.prototype_candidate_scores {
            prototype_validation_csv.push_str(&format!(
                "{seed},{},{:.6},{}\n",
                candidate.prototypes_per_class,
                candidate.validation_accuracy,
                candidate.prototypes_per_class == class_knowledge.selected_prototypes_per_class
            ));
        }
        let cv = run_cv(&samples, &split, seed)?;
        let path = std::env::temp_dir().join(format!(
            "shella-m11-{seed}-{}.json",
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let mut memory = LearningMemory::open(&path)?;
        memory.save_class_knowledge(&class_knowledge)?;
        let mut network = build_network(seed)?;
        let mut formed = 0;
        for &index in split.train() {
            let scaled = scaler.apply(samples[index].features());
            train_one(&mut network, scaled, samples[index].label())?;
            let centered = scaled.map(|value| value * 2.0 - 1.0);
            network.apply_rewarded_readout(&centered, samples[index].label(), &ACTIONS, 0.8)?;
            let predicted = predict_neural(&network, scaled);
            if predicted != Some(samples[index].label()) || predicted.is_none() || index % 4 == 0 {
                memory.record_feedback(
                    "iris-train",
                    &memory_input(scaled),
                    Some(samples[index].label()),
                    predicted,
                )?;
                formed += 1;
            }
        }
        let selected_current = select_memory_current(
            &network,
            &memory,
            split.validation(),
            &samples,
            &scaler,
            seed,
        )?;
        let test_without = evaluate_split(
            &network,
            None,
            split.test(),
            &samples,
            &scaler,
            seed,
            0.0,
            selected_current,
        )?;
        let test_memory = evaluate_split(
            &network,
            Some(&memory),
            split.test(),
            &samples,
            &scaler,
            seed,
            0.0,
            selected_current,
        )?;
        let test_external = evaluate_external(&memory, split.test(), &samples, &scaler, seed, 0.0)?;
        let test_class_predictions = collect_class_knowledge_predictions(
            &network,
            &class_knowledge,
            split.test(),
            &samples,
            &scaler,
            seed,
            0.0,
        )?;
        let test_class_knowledge = metrics_from_pairs(test_class_predictions.clone())?;
        append_class_metrics(&mut class_metrics_csv, seed, &test_class_predictions);
        let (brier_score, mean_absolute_error) = evaluate_class_knowledge_calibration(
            &class_knowledge,
            split.test(),
            &samples,
            &scaler,
        )?;
        calibration_csv.push_str(&format!(
            "{seed},test,{},{brier_score:.6},{mean_absolute_error:.6}\n",
            split.test().len()
        ));
        print_metrics(
            seed,
            "multi_prototype_lif",
            "test",
            0.0,
            test_class_knowledge,
            0.0,
            class_knowledge.selected_prototypes_per_class * 3,
            cv,
            0,
        );
        let test_noisy = evaluate_split(
            &network,
            Some(&memory),
            split.test(),
            &samples,
            &scaler,
            seed,
            0.25,
            selected_current,
        )?;
        print_metrics(
            seed,
            "lif_stdp_without_memory",
            "test",
            0.0,
            test_without,
            0.0,
            0,
            cv,
            0,
        );
        print_metrics(
            seed,
            "lif_stdp_with_memory",
            "test",
            0.0,
            test_memory,
            0.0,
            formed,
            cv,
            0,
        );
        print_metrics(
            seed,
            "external_retrieval",
            "test",
            0.0,
            test_external,
            0.0,
            formed,
            cv,
            0,
        );
        print_metrics(
            seed,
            "lif_stdp_with_memory",
            "test",
            0.25,
            test_noisy,
            0.0,
            formed,
            cv,
            0,
        );

        let phase_a = split
            .train()
            .iter()
            .copied()
            .filter(|i| samples[*i].label() < 2)
            .collect::<Vec<_>>();
        let old_before = evaluate_split(
            &network,
            Some(&memory),
            &phase_a,
            &samples,
            &scaler,
            seed,
            0.0,
            selected_current,
        )?;
        for &index in split.train().iter().filter(|i| samples[**i].label() == 2) {
            let scaled = scaler.apply(samples[index].features());
            train_one(&mut network, scaled, samples[index].label())?;
            let centered = scaled.map(|value| value * 2.0 - 1.0);
            network.apply_rewarded_readout(&centered, samples[index].label(), &ACTIONS, 0.8)?;
        }
        let old_after = evaluate_split(
            &network,
            Some(&memory),
            &phase_a,
            &samples,
            &scaler,
            seed,
            0.0,
            selected_current,
        )?;
        let forgetting = old_before.accuracy().unwrap_or(0.0) - old_after.accuracy().unwrap_or(0.0);
        let mut replay_presentations = 0;
        for &index in &phase_a {
            let report = memory.replay_task_relevant(
                "iris-train",
                &mut network,
                &memory_input(scaler.apply(samples[index].features())),
                &ACTIONS,
                1,
                500.0,
            )?;
            replay_presentations += report.presentations();
        }
        let old_replayed = evaluate_split(
            &network,
            Some(&memory),
            &phase_a,
            &samples,
            &scaler,
            seed,
            0.0,
            selected_current,
        )?;
        print_metrics(
            seed,
            "continual_old_after_replay",
            "old_task",
            0.0,
            old_replayed,
            forgetting,
            formed,
            cv,
            replay_presentations,
        );

        let validation_before_feedback = evaluate_class_knowledge(
            &network,
            &class_knowledge,
            split.validation(),
            &samples,
            &scaler,
            seed,
            0.0,
        )?;
        let weak_records = collect_class_knowledge_records(
            &network,
            &class_knowledge,
            split.validation(),
            &samples,
            &scaler,
        )?;
        let self_eval = evaluate_predictions(&weak_records)?;
        let mut candidate_knowledge = class_knowledge.clone();
        let mut candidate = network.clone();
        let mut feedback_replays = 0;
        for (index, (expected, predicted, margin)) in weak_records.iter().enumerate() {
            if predicted != &Some(*expected) || *margin < 0.15 {
                let sample_index = split.validation()[index];
                candidate_knowledge.apply_validation_feedback(
                    *expected,
                    predicted.unwrap_or((*expected + 1) % 3),
                    0.1,
                )?;
                memory.record_feedback(
                    "iris-validation-feedback",
                    &memory_input(scaler.apply(samples[sample_index].features())),
                    Some(*expected),
                    *predicted,
                )?;
                let report = memory.replay_task_relevant(
                    "iris-validation-feedback",
                    &mut candidate,
                    &memory_input(scaler.apply(samples[sample_index].features())),
                    &ACTIONS,
                    1,
                    500.0,
                )?;
                feedback_replays += report.presentations();
            }
        }
        candidate_knowledge.refit_calibration(&validation)?;
        let candidate_validation = evaluate_class_knowledge(
            &candidate,
            &candidate_knowledge,
            split.validation(),
            &samples,
            &scaler,
            seed,
            0.0,
        )?;
        let validation_after = if candidate_validation.accuracy().unwrap_or(0.0)
            >= validation_before_feedback.accuracy().unwrap_or(0.0)
        {
            network = candidate;
            class_knowledge = candidate_knowledge;
            memory.save_class_knowledge(&class_knowledge)?;
            candidate_validation
        } else {
            memory.discard_task("iris-validation-feedback")?;
            validation_before_feedback
        };
        let improvement = validation_after.accuracy().unwrap_or(0.0)
            - validation_before_feedback.accuracy().unwrap_or(0.0);
        println!(
            "self_loop,{seed},validation_before_after,validation,0,{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.6},{}",
            validation_after.accuracy().unwrap_or(0.0),
            validation_after.precision().unwrap_or(0.0),
            validation_after.recall().unwrap_or(0.0),
            validation_after.false_positive_rate().unwrap_or(0.0),
            validation_after.false_negative_rate().unwrap_or(0.0),
            validation_after.abstention_rate().unwrap_or(0.0),
            improvement,
            self_eval.weak_samples(),
            self_eval.uncertainty_samples(),
            memory.len(),
            cv,
            replay_presentations + feedback_replays
        );
        let persistent_before = evaluate_class_knowledge(
            &network,
            &class_knowledge,
            split.test(),
            &samples,
            &scaler,
            seed,
            0.0,
        )?;
        let training_run_id = format!("m11-{seed}");
        drop(memory);
        let restored = LearningMemory::open(&path)?;
        let restored_knowledge =
            restored
                .load_class_knowledge(&training_run_id)?
                .ok_or(SimError::CorruptData {
                    subsystem: "class knowledge",
                    reason: "model tidak tersedia setelah restart".to_owned(),
                })?;
        let restored_test = evaluate_class_knowledge(
            &network,
            &restored_knowledge,
            split.test(),
            &samples,
            &scaler,
            seed,
            0.0,
        )?;
        print_metrics(
            seed,
            "persistent_after_restart",
            "test",
            0.0,
            restored_test,
            restored_test.accuracy().unwrap_or(0.0) - persistent_before.accuracy().unwrap_or(0.0),
            restored.len(),
            cv,
            0,
        );
        cleanup(&path);
    }
    fs::write(
        "results/class_knowledge_validation_2026-10-19.csv",
        prototype_validation_csv,
    )?;
    fs::write(
        "results/class_knowledge_calibration_2026-10-19.csv",
        calibration_csv,
    )?;
    fs::write(
        "results/class_knowledge_per_class_2026-10-19.csv",
        class_metrics_csv,
    )?;
    Ok(())
}

fn build_network(seed: u64) -> Result<LifNetwork, SimError> {
    let config = LifConfig::new(1.0, 20.0, 1.0, -65.0, -65.0, -50.0, 1);
    let mut network = LifNetwork::new(config, NEURONS)?;
    for source in 0..INPUTS {
        for (class, &target) in ACTIONS.iter().enumerate() {
            let variation =
                ((seed.wrapping_add(source as u64 * 11 + class as u64 * 7) % 9) as f64) - 4.0;
            network.add_synapse(Synapse::new(
                source,
                target,
                18.0 + variation,
                0.0,
                80.0,
                1,
            )?)?;
        }
    }
    network.enable_reward_modulated_stdp(RewardModulatedStdpConfig::new(
        StdpConfig::new(0.02, 0.015, 20.0, 20.0),
        30.0,
        0.05,
    ))?;
    Ok(network)
}

fn train_one(
    network: &mut LifNetwork,
    features: [f64; INPUTS],
    label: usize,
) -> Result<(), SimError> {
    let input = memory_input(features);
    network.step(&input)?;
    network.step(&[0.0; NEURONS])?;
    let mut teacher = vec![0.0; NEURONS];
    teacher[ACTIONS[label]] = 500.0;
    network.step(&teacher)?;
    network.apply_reward(1.0)?;
    network.step(&[0.0; NEURONS])?;
    Ok(())
}

fn predict_neural(network: &LifNetwork, features: [f64; INPUTS]) -> Option<usize> {
    let mut probe = network.clone();
    let input = memory_input(features);
    let mut spikes = Vec::new();
    for current in [
        input,
        vec![0.0; NEURONS],
        vec![0.0; NEURONS],
        vec![0.0; NEURONS],
    ] {
        if let Ok(step) = probe.step(&current) {
            spikes.extend(step.spiking_neurons().iter().copied());
        }
    }
    ACTIONS
        .iter()
        .position(|neuron| spikes.contains(neuron))
        .or_else(|| {
            ACTIONS
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    probe
                        .neuron(**left)
                        .map(|value| value.membrane_potential_mv())
                        .unwrap_or(f64::NEG_INFINITY)
                        .total_cmp(
                            &probe
                                .neuron(**right)
                                .map(|value| value.membrane_potential_mv())
                                .unwrap_or(f64::NEG_INFINITY),
                        )
                })
                .map(|(action, _)| action)
        })
}

fn predict_memory(
    network: &LifNetwork,
    memory: &LearningMemory,
    features: [f64; INPUTS],
    current: f64,
) -> Result<(Option<usize>, f64), SimError> {
    let mut probe = network.clone();
    let result = memory.infer_task_with_memory_adaptive(
        "iris-train",
        &mut probe,
        &memory_input(features),
        &ACTIONS,
        5,
        current,
        0.35,
    )?;
    let spikes = result.spiking_neurons();
    let predicted = ACTIONS
        .iter()
        .position(|neuron| spikes.contains(neuron))
        .or_else(|| predict_neural(network, features));
    let margin = result
        .recalled()
        .first()
        .map(|record| record.similarity())
        .unwrap_or(0.0);
    Ok((predicted, margin))
}

fn predict_external(
    memory: &LearningMemory,
    features: [f64; INPUTS],
) -> Result<(Option<usize>, f64), SimError> {
    let recalls = memory.recall_similar(&memory_input(features), 5)?;
    let mut scores = [0.0; 3];
    for record in recalls {
        if let Some(action) = record.expected_action() {
            scores[action] += record.retrieval_score();
        }
    }
    let mut order = (0..3).collect::<Vec<_>>();
    order.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then_with(|| a.cmp(b)));
    let margin = scores[order[0]] - scores[order[1]];
    Ok(((scores[order[0]] > 0.0).then_some(order[0]), margin))
}

fn evaluate_class_knowledge(
    network: &LifNetwork,
    knowledge: &ClassKnowledgeModel,
    indices: &[usize],
    samples: &[IrisSample],
    scaler: &Scale,
    seed: u64,
    noise: f64,
) -> Result<DecisionMetrics, SimError> {
    metrics_from_pairs(collect_class_knowledge_predictions(
        network, knowledge, indices, samples, scaler, seed, noise,
    )?)
}

fn collect_class_knowledge_predictions(
    network: &LifNetwork,
    knowledge: &ClassKnowledgeModel,
    indices: &[usize],
    samples: &[IrisSample],
    scaler: &Scale,
    seed: u64,
    noise: f64,
) -> Result<Vec<(usize, Option<usize>)>, SimError> {
    let mut pairs = Vec::with_capacity(indices.len());
    for (position, &index) in indices.iter().enumerate() {
        let mut features = scaler.apply(samples[index].features());
        for (feature, value) in features.iter_mut().enumerate() {
            *value =
                (*value + shella::noise(seed, index + position, feature, noise)).clamp(0.0, 1.0);
        }
        let mut probe = network.clone();
        let inference =
            knowledge.infer_normalized_primary_with_lif(&mut probe, features, &ACTIONS)?;
        let predicted = inference
            .prediction_from_network
            .unwrap_or(inference.prediction_from_knowledge);
        pairs.push((samples[index].label(), Some(predicted)));
    }
    Ok(pairs)
}

fn append_class_metrics(output: &mut String, seed: u64, pairs: &[(usize, Option<usize>)]) {
    let class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
    for (class_id, class_name) in class_names.iter().enumerate() {
        let mut tp = 0;
        let mut fp = 0;
        let mut false_negative = 0;
        let mut tn = 0;
        let mut confusion = [0; 3];
        for (expected, predicted) in pairs {
            let predicted = predicted.unwrap_or(usize::MAX);
            if *expected == class_id && predicted < confusion.len() {
                confusion[predicted] += 1;
            }
            match (*expected == class_id, predicted == class_id) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => false_negative += 1,
                (false, false) => tn += 1,
            }
        }
        let accuracy = if pairs.is_empty() {
            0.0
        } else {
            (tp + tn) as f64 / pairs.len() as f64
        };
        let precision = ratio(tp, tp + fp);
        let recall = ratio(tp, tp + false_negative);
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let fpr = ratio(fp, fp + tn);
        let fnr = ratio(false_negative, false_negative + tp);
        output.push_str(&format!(
            "{seed},{class_id},{},{tp},{fp},{false_negative},{tn},{accuracy:.6},{precision:.6},{recall:.6},{f1:.6},{fpr:.6},{fnr:.6},{},{},{}\n",
            class_name, confusion[0], confusion[1], confusion[2]
        ));
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn evaluate_class_knowledge_calibration(
    knowledge: &ClassKnowledgeModel,
    indices: &[usize],
    samples: &[IrisSample],
    scaler: &Scale,
) -> Result<(f64, f64), SimError> {
    let mut evaluator = CalibrationEvaluator::new();
    for index in indices {
        let normalized = scaler.apply(samples[*index].features());
        let prediction = knowledge.predict_normalized_primary(normalized)?;
        let confidence = prediction
            .calibrated_confidence
            .ok_or(SimError::CorruptData {
                subsystem: "class calibration",
                reason: "model tidak memiliki confidence Platt".to_owned(),
            })?;
        evaluator.record(confidence, prediction.class_id == samples[*index].label())?;
    }
    let metrics = evaluator.metrics();
    Ok((
        metrics.brier_score().ok_or(SimError::InvalidInput {
            parameter: "calibration_samples",
            reason: "split tidak memiliki sampel",
        })?,
        metrics
            .mean_absolute_calibration_error()
            .ok_or(SimError::InvalidInput {
                parameter: "calibration_samples",
                reason: "split tidak memiliki sampel",
            })?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_split(
    network: &LifNetwork,
    memory: Option<&LearningMemory>,
    indices: &[usize],
    samples: &[IrisSample],
    scaler: &Scale,
    seed: u64,
    noise: f64,
    current: f64,
) -> Result<DecisionMetrics, SimError> {
    let records = indices
        .iter()
        .enumerate()
        .map(|(position, &index)| {
            let mut x = scaler.apply(samples[index].features());
            for (feature, value) in x.iter_mut().enumerate().take(INPUTS) {
                *value = (*value + shella::noise(seed, index + position, feature, noise))
                    .clamp(0.0, 1.0);
            }
            let predicted = memory.map_or(Ok(predict_neural(network, x)), |store| {
                predict_memory(network, store, x, current).map(|value| value.0)
            });
            predicted.map(|prediction| (samples[index].label(), prediction))
        })
        .collect::<Result<Vec<_>, SimError>>()?;
    metrics_from_pairs(records)
}

fn evaluate_external(
    memory: &LearningMemory,
    indices: &[usize],
    samples: &[IrisSample],
    scaler: &Scale,
    seed: u64,
    noise: f64,
) -> Result<DecisionMetrics, SimError> {
    let records = indices
        .iter()
        .enumerate()
        .map(|(position, &index)| {
            let mut x = scaler.apply(samples[index].features());
            for (feature, value) in x.iter_mut().enumerate().take(INPUTS) {
                *value = (*value + shella::noise(seed, index + position, feature, noise))
                    .clamp(0.0, 1.0);
            }
            predict_external(memory, x).map(|value| (samples[index].label(), value.0))
        })
        .collect::<Result<Vec<_>, SimError>>()?;
    metrics_from_pairs(records)
}

fn collect_class_knowledge_records(
    network: &LifNetwork,
    knowledge: &ClassKnowledgeModel,
    indices: &[usize],
    samples: &[IrisSample],
    scaler: &Scale,
) -> Result<Vec<(usize, Option<usize>, f64)>, SimError> {
    indices
        .iter()
        .map(|index| {
            let features = scaler.apply(samples[*index].features());
            let prediction = knowledge.predict_normalized_primary(features)?;
            let mut probe = network.clone();
            let inference =
                knowledge.infer_normalized_primary_with_lif(&mut probe, features, &ACTIONS)?;
            Ok((
                samples[*index].label(),
                inference
                    .prediction_from_network
                    .or(Some(inference.prediction_from_knowledge)),
                prediction.margin,
            ))
        })
        .collect()
}

fn metrics_from_pairs(pairs: Vec<(usize, Option<usize>)>) -> Result<DecisionMetrics, SimError> {
    let records = pairs
        .into_iter()
        .map(|(expected, predicted)| (expected, predicted, 1.0))
        .collect::<Vec<_>>();
    Ok(evaluate_predictions(&records)?.metrics())
}

fn select_memory_current(
    network: &LifNetwork,
    memory: &LearningMemory,
    indices: &[usize],
    samples: &[IrisSample],
    scaler: &Scale,
    seed: u64,
) -> Result<f64, SimError> {
    let mut best = (f64::NEG_INFINITY, MEMORY_CURRENTS[0]);
    for current in MEMORY_CURRENTS {
        let score = evaluate_split(
            network,
            Some(memory),
            indices,
            samples,
            scaler,
            seed,
            0.0,
            current,
        )?
        .accuracy()
        .unwrap_or(0.0);
        if score > best.0 {
            best = (score, current);
        }
    }
    Ok(best.1)
}

fn run_cv(
    samples: &[IrisSample],
    split: &shella::StratifiedSplit,
    seed: u64,
) -> Result<f64, SimError> {
    let mut pool = split.train().to_vec();
    pool.extend_from_slice(split.validation());
    let folds = stratified_folds(samples, &pool, 5)?;
    let mut scores = Vec::new();
    for (train, validation) in folds {
        let scaler = fit_scale(samples, &train)?;
        let mut memory = LearningMemory::open(std::env::temp_dir().join(format!(
            "shella-cv-{seed}-{}.json",
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        )))?;
        for index in train {
            memory.record_with_feedback(
                "cv",
                &memory_input(scaler.apply(samples[index].features())),
                Some(samples[index].label()),
                1.0,
                1.0,
            )?;
        }
        scores.push(
            evaluate_external(&memory, &validation, samples, &scaler, seed, 0.0)?
                .accuracy()
                .unwrap_or(0.0),
        );
    }
    Ok(scores.iter().sum::<f64>() / scores.len() as f64)
}

#[allow(clippy::too_many_arguments)]
fn print_metrics(
    seed: u64,
    mode: &str,
    split: &str,
    noise: f64,
    metrics: DecisionMetrics,
    forgetting: f64,
    memory: usize,
    cv: f64,
    replay_presentations: usize,
) {
    println!(
        "m11,{seed},{mode},{split},{noise:.2},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{forgetting:.6},0,0,{memory},{cv:.6},{replay_presentations}",
        metrics.accuracy().unwrap_or(0.0),
        metrics.precision().unwrap_or(0.0),
        metrics.recall().unwrap_or(0.0),
        metrics.false_positive_rate().unwrap_or(0.0),
        metrics.false_negative_rate().unwrap_or(0.0),
        metrics.abstention_rate().unwrap_or(0.0)
    );
}

fn load_iris(path: impl AsRef<Path>) -> Result<Vec<IrisSample>, Box<dyn Error>> {
    fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields = line.split(',').collect::<Vec<_>>();
            if fields.len() != 5 {
                return Err("format Iris tidak valid".into());
            }
            let mut x = [0.0; 4];
            for (i, value) in fields[..4].iter().enumerate() {
                x[i] = value.parse()?;
            }
            let label = match fields[4] {
                "Iris-setosa" => 0,
                "Iris-versicolor" => 1,
                "Iris-virginica" => 2,
                _ => return Err("label Iris tidak dikenal".into()),
            };
            IrisSample::new(x, label).map_err(|error| error.to_string().into())
        })
        .collect()
}
fn fit_scale(samples: &[IrisSample], indices: &[usize]) -> Result<Scale, SimError> {
    if indices.is_empty() {
        return Err(SimError::InvalidInput {
            parameter: "train_indices",
            reason: "tidak boleh kosong",
        });
    }
    let mut min = [f64::INFINITY; 4];
    let mut max = [f64::NEG_INFINITY; 4];
    for &index in indices {
        let sample = samples.get(index).ok_or(SimError::InvalidInput {
            parameter: "train_indices",
            reason: "indeks di luar dataset",
        })?;
        for feature in 0..4 {
            min[feature] = min[feature].min(sample.features()[feature]);
            max[feature] = max[feature].max(sample.features()[feature]);
        }
    }
    if (0..4).any(|feature| min[feature] == max[feature]) {
        return Err(SimError::InvalidInput {
            parameter: "train_scaler",
            reason: "rentang fitur train harus positif",
        });
    }
    Ok(Scale { min, max })
}
fn memory_input(features: [f64; INPUTS]) -> Vec<f64> {
    let mut input = vec![0.0; NEURONS];
    for (slot, feature) in input.iter_mut().take(INPUTS).zip(features) {
        *slot = feature * 2.0 - 1.0;
    }
    input
}
fn cleanup(path: &PathBuf) {
    let _ = fs::remove_file(path);
    let mut lock = path.as_os_str().to_os_string();
    lock.push(".lock");
    let _ = fs::remove_file(lock);
}
