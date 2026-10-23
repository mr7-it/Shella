#[path = "shella_lm/terminal_ui.rs"]
mod terminal_ui;

use serde::Serialize;
use shella::{
    CorpusSplit, DatasetSplitConfig, GeneratedText, GenerationConfig, GenerationRepetitionMetrics,
    GenerativeLanguageModel, TrainingConfig, TransformerConfig,
};
use std::{collections::BTreeMap, error::Error, fs, io::IsTerminal, path::PathBuf, time::Instant};
use terminal_ui::{TerminalUi, TrainingProgressUi};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Text,
    Json,
}

fn main() {
    if let Err(error) = run() {
        TerminalUi::detect().error(&format!("shella-lm: {error}"));
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    if raw_args.is_empty() || raw_args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_help(TerminalUi::detect());
        return Ok(());
    }
    if raw_args.len() == 1 && raw_args[0] == "--version" {
        println!("shella-lm {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let mut args = raw_args.into_iter();
    let command = args.next().ok_or("perintah belum diberikan")?;
    let mut options = BTreeMap::new();
    while let Some(key) = args.next() {
        if !key.starts_with("--") {
            return Err(format!("opsi harus diawali --: {key}").into());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("nilai untuk {key} belum diberikan"))?;
        if options.insert(key.clone(), value).is_some() {
            return Err(format!("opsi duplikat: {key}").into());
        }
    }
    match command.as_str() {
        "help" => {
            print_help(TerminalUi::detect());
            Ok(())
        }
        "train" => train_command(options),
        "generate" => generate_command(options),
        "evaluate" => evaluate_command(options),
        "metrics" => metrics_command(options),
        _ => Err("perintah harus train, generate, evaluate, atau metrics".into()),
    }
}

fn print_help(ui: TerminalUi) {
    ui.header(
        "SHELLA-LM · TERMINAL",
        "antarmuka bahasa Indonesia untuk model generatif Shella",
    );
    println!(
        r#"Penggunaan:
  shella-lm <perintah> [opsi]

Perintah:
  train      Latih model; tampilkan progres aktual dan simpan checkpoint
  generate   Buat kelanjutan teks dari checkpoint
  evaluate   Ukur split test dari checkpoint dan corpus yang cocok
  metrics    Tampilkan diagnostik pengulangan hasil generasi
  help       Tampilkan bantuan ini

Format:
  --format text   Tampilan ramah terminal
  --format json   Output JSON untuk integrasi/pipeline
  Default: text saat stdout terminal, JSON saat stdout diarahkan/di-pipe.
  Warna mati otomatis jika NO_COLOR disetel atau terminal tidak mendukung ANSI.

Contoh:
  shella-lm train --dataset data/indonesian_coherent_corpus/prepared_corpus.json \
    --checkpoint target/shella-id.sllm --epochs 1 --batch-size 64
  shella-lm generate --checkpoint target/shella-id.sllm --prompt "Apa itu neuron?"
  shella-lm evaluate --checkpoint target/shella-id.sllm \
    --dataset data/indonesian_coherent_corpus/prepared_corpus.json

Opsi train: --dataset --checkpoint --epochs --batch-size --learning-rate
            --weight-decay --gradient-clip --seed --split-seed
            --hidden-size --layers --heads --context --format
Opsi generate: --checkpoint --prompt --max-new-tokens --temperature
               --top-k --top-p --seed --format"#
    );
}

fn output_format(options: &BTreeMap<String, String>) -> Result<OutputFormat, Box<dyn Error>> {
    match options.get("--format").map(String::as_str) {
        Some("text") => Ok(OutputFormat::Text),
        Some("json") => Ok(OutputFormat::Json),
        Some(other) => Err(format!("--format harus text atau json, bukan {other}").into()),
        None if std::io::stdout().is_terminal() => Ok(OutputFormat::Text),
        None => Ok(OutputFormat::Json),
    }
}

fn load_training_corpus(
    dataset_path: &str,
    split_config: DatasetSplitConfig,
) -> Result<CorpusSplit, Box<dyn Error>> {
    if dataset_path.ends_with(".json") {
        let prepared = CorpusSplit::load_prepared_json(dataset_path)?;
        if prepared.manifest.split.seed != split_config.seed {
            return Err("--split-seed harus sama dengan seed pada manifest korpus JSON".into());
        }
        Ok(prepared)
    } else {
        Ok(GenerativeLanguageModel::load_nusax_training_corpus(
            dataset_path,
            split_config,
        )?)
    }
}

fn print_training_summary(report: &TrainRunReport<'_>) {
    println!("\n╭─ TRAINING SELESAI ──────────────────────────────────────────");
    println!("│ sumber   : {}", report.source.dataset_id);
    println!("│ lisensi  : {}", report.source_license);
    println!(
        "│ split    : {} train · {} validasi · {} test dokumen",
        report.training.training_documents,
        report.training.validation_documents,
        report.final_test.documents
    );
    println!(
        "│ token    : {} train · {} validasi · {} test",
        report.training_model_tokens, report.validation_model_tokens, report.test_model_tokens
    );
    println!("├─ METRIK ────────────────────────────────────────────────────");
    println!(
        "│ validasi : loss {:.4} → {:.4} · perplexity {:.3} → {:.3}",
        report.baseline_validation.mean_loss,
        report.final_validation.mean_loss,
        report.baseline_validation.perplexity,
        report.final_validation.perplexity
    );
    for epoch in &report.training.epochs {
        println!(
            "│ epoch {:>2} : train loss {:.4} · val loss {:.4} · {} optimizer steps",
            epoch.epoch, epoch.training_loss, epoch.validation_loss, epoch.optimizer_steps
        );
    }
    println!(
        "│ test     : loss {:.4} · perplexity {:.3}",
        report.final_test.mean_loss, report.final_test.perplexity
    );
    println!("├─ CHECKPOINT ────────────────────────────────────────────────");
    println!("│ file     : {}", report.checkpoint_path.display());
    println!("│ sha256   : {}", report.checkpoint_sha256);
    println!(
        "│ model    : {} parameter · {:.1} detik total",
        report.parameter_count, report.training_and_evaluation_seconds
    );
    println!("╰────────────────────────────────────────────────────────────");
}

fn print_generated_text(generated: &GeneratedText) {
    println!("\n╭─ PROMPT ────────────────────────────────────────────────────");
    println!("{}", terminal_safe_text(&generated.prompt));
    println!("├─ KELANJUTAN · BELUM DIVERIFIKASI ───────────────────────────");
    println!("{}", terminal_safe_text(&generated.continuation));
    println!("╰────────────────────────────────────────────────────────────");
    println!(
        "{} token karakter baru · strategi {:?} · prompt terpotong: {}",
        generated.generated_tokens, generated.strategy, generated.prompt_truncated
    );
    println!("Catatan: generasi ini tidak otomatis memiliki grounding fakta.");
}

fn print_evaluation_metrics(metrics: &shella::EvaluationMetrics) {
    println!("\n╭─ HASIL EVALUASI TEST ───────────────────────────────────────");
    println!("│ dokumen   : {}", metrics.documents);
    println!("│ windows   : {}", metrics.windows);
    println!("│ token     : {}", metrics.tokens);
    println!("│ mean loss : {:.5}", metrics.mean_loss);
    println!("│ perplexity: {:.5}", metrics.perplexity);
    println!("╰────────────────────────────────────────────────────────────");
}

fn print_repetition_metrics(metrics: &GenerationRepetitionMetrics) {
    println!("\n╭─ DIAGNOSTIK PENGULANGAN KARAKTER ───────────────────────────");
    println!("│ token        : {}", metrics.emitted_tokens);
    println!(
        "│ karakter unik: {}",
        format_rate(metrics.distinct_character_ratio)
    );
    println!(
        "│ ulang 2-gram : {}",
        format_rate(metrics.repeated_bigram_rate)
    );
    println!(
        "│ ulang 3-gram : {}",
        format_rate(metrics.repeated_trigram_rate)
    );
    println!(
        "│ ulang 4-gram : {}",
        format_rate(metrics.repeated_fourgram_rate)
    );
    println!(
        "│ ulang dekat  : {}",
        format_rate(metrics.adjacent_repeat_rate)
    );
    println!("│ selesai EOS  : {}", metrics.eos_terminated);
    println!("│ prompt cut   : {}", metrics.prompt_truncated);
    println!("╰────────────────────────────────────────────────────────────");
    println!("Catatan: diagnostik permukaan ini bukan skor kelancaran atau koherensi.");
}

fn format_rate(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_owned(), |rate| format!("{:.1}%", rate * 100.0))
}

fn terminal_safe_text(text: &str) -> String {
    let mut safe = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\n' => safe.push('\n'),
            '\t' => safe.push(' '),
            '\r' => safe.push_str("\\r"),
            control if control.is_control() => {
                safe.push_str(&format!("\\u{{{:04x}}}", control as u32));
            }
            printable => safe.push(printable),
        }
    }
    safe
}

fn train_command(options: BTreeMap<String, String>) -> Result<(), Box<dyn Error>> {
    reject_unknown(
        &options,
        &[
            "--dataset",
            "--checkpoint",
            "--epochs",
            "--batch-size",
            "--learning-rate",
            "--weight-decay",
            "--gradient-clip",
            "--seed",
            "--split-seed",
            "--hidden-size",
            "--layers",
            "--heads",
            "--context",
            "--format",
        ],
    )?;
    let format = output_format(&options)?;
    let ui = TerminalUi::detect();
    ui.header(
        "TRAINING · DECODER KARAKTER",
        "Candle CPU  /  Transformer + head LIF  /  progress aktual",
    );
    let dataset_path = value_or(
        &options,
        "--dataset",
        "data/language/nusax/sentiment/indonesian/train.csv",
    );
    let checkpoint_path = required(&options, "--checkpoint")?;
    let checkpoint_path = checkpoint_path.to_owned();
    let seed = parse_or(&options, "--seed", 2026_u64)?;
    let split_config = DatasetSplitConfig {
        seed: parse_or(&options, "--split-seed", 42_u64)?,
        ..DatasetSplitConfig::default()
    };
    let started = Instant::now();
    let corpus = ui.run_stage("memuat korpus + memeriksa provenance", || {
        load_training_corpus(&dataset_path, split_config)
    })?;
    let config = TransformerConfig {
        hidden_size: parse_or(&options, "--hidden-size", 32_usize)?,
        layer_count: parse_or(&options, "--layers", 1_usize)?,
        head_count: parse_or(&options, "--heads", 2_usize)?,
        max_context: parse_or(&options, "--context", 64_usize)?,
        seed,
        ..TransformerConfig::default()
    };
    let mut model = ui.run_stage("membangun tokenizer dan model", || {
        GenerativeLanguageModel::from_corpus(config, &corpus)
    })?;
    let training_config = TrainingConfig {
        epochs: parse_or(&options, "--epochs", 6_usize)?,
        batch_size: parse_or(&options, "--batch-size", 8_usize)?,
        learning_rate: parse_or(&options, "--learning-rate", 0.002_f64)?,
        weight_decay: parse_or(&options, "--weight-decay", 0.01_f64)?,
        gradient_clip_norm: parse_or(&options, "--gradient-clip", 1.0_f64)?,
        seed,
    };
    // Validation can be compared before/after training; held-out test remains untouched
    // until all optimizer updates and checkpoint serialization are complete.
    let baseline_validation = ui.run_stage("mengukur baseline validasi", || {
        model.evaluate(&corpus.validation, training_config.batch_size)
    })?;
    ui.info(&format!(
        "{} dokumen train · {} validasi · {} test · {} parameter",
        corpus.training.len(),
        corpus.validation.len(),
        corpus.test.len(),
        model.parameter_count()
    ));
    let mut progress_ui = TrainingProgressUi::new(ui);
    let report = model.train_with_progress(&corpus, training_config, |progress| {
        progress_ui.update(progress);
    })?;
    progress_ui.finish();
    let final_validation = ui.run_stage("mengukur validasi akhir", || {
        model.evaluate(&corpus.validation, report.config.batch_size)
    })?;
    let checkpoint_sha256 = ui.run_stage("menyimpan checkpoint model", || {
        model.save_checkpoint(&checkpoint_path)
    })?;

    // Test split baru dipakai sesudah semua optimizer step selesai dan checkpoint ditulis.
    let restored = ui.run_stage("memuat ulang checkpoint untuk verifikasi", || {
        GenerativeLanguageModel::load_checkpoint(&checkpoint_path)
    })?;
    let final_test = ui.run_stage("mengukur test holdout", || {
        restored.evaluate(&corpus.test, report.config.batch_size)
    })?;
    let training_model_tokens = encoded_token_count(&restored, &corpus.training)?;
    let validation_model_tokens = encoded_token_count(&restored, &corpus.validation)?;
    let test_model_tokens = encoded_token_count(&restored, &corpus.test)?;
    let output = TrainRunReport {
        source: &corpus.manifest.source,
        split: &corpus.manifest.split,
        source_file_sha256: &corpus.manifest.source.source_file_sha256,
        source_license: &corpus.manifest.source.license_id,
        training_characters: corpus.manifest.training_characters,
        validation_characters: corpus.manifest.validation_characters,
        test_characters: corpus.manifest.test_characters,
        training_model_tokens,
        validation_model_tokens,
        test_model_tokens,
        baseline_validation,
        final_validation,
        training: report,
        final_test,
        parameter_count: restored.parameter_count(),
        checkpoint_path: PathBuf::from(checkpoint_path),
        checkpoint_sha256,
        training_and_evaluation_seconds: started.elapsed().as_secs_f64(),
        notes: [
            "Source, license basis, checksums, and document split are recorded in the corpus manifest.",
            "Tokenizer is fit on the model training split only.",
            "Model-token counts use CharacterTokenizer encoding and include BOS/EOS per document.",
            "Test split is evaluated only after training; validation baseline/final metrics are paired.",
            "Checkpoint contains model weights and provenance/metrics, not AdamW optimizer state.",
        ],
    };
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&output)?),
        OutputFormat::Text => print_training_summary(&output),
    }
    Ok(())
}

fn generate_command(options: BTreeMap<String, String>) -> Result<(), Box<dyn Error>> {
    reject_unknown(
        &options,
        &[
            "--checkpoint",
            "--prompt",
            "--max-new-tokens",
            "--temperature",
            "--top-k",
            "--top-p",
            "--seed",
            "--format",
        ],
    )?;
    let format = output_format(&options)?;
    let ui = TerminalUi::detect();
    ui.header(
        "GENERATE · INFERENCE",
        "decoder autoregresif  /  output belum diverifikasi",
    );
    let checkpoint = required(&options, "--checkpoint")?;
    let prompt = required(&options, "--prompt")?;
    let temperature = parse_or(&options, "--temperature", 0.8_f64)?;
    let top_k_value: usize = parse_or(&options, "--top-k", 8_usize)?;
    let top_p_value: f64 = parse_or(&options, "--top-p", 0.9_f64)?;
    let config = GenerationConfig {
        max_new_tokens: parse_or(&options, "--max-new-tokens", 80_usize)?,
        temperature,
        top_k: if temperature == 0.0 || top_k_value == 0 {
            None
        } else {
            Some(top_k_value)
        },
        top_p: if temperature == 0.0 || top_p_value == 0.0 {
            None
        } else {
            Some(top_p_value)
        },
        seed: parse_or(&options, "--seed", 2026_u64)?,
    };
    let model = ui.run_stage("memuat checkpoint", || {
        GenerativeLanguageModel::load_checkpoint(checkpoint)
    })?;
    let generated = ui.run_stage("menyusun teks autoregresif", || {
        model.generate(prompt, config)
    })?;
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&generated)?),
        OutputFormat::Text => print_generated_text(&generated),
    }
    Ok(())
}

#[derive(Serialize)]
struct GenerationMetricsReport {
    repetition: GenerationRepetitionMetrics,
}

fn metrics_command(options: BTreeMap<String, String>) -> Result<(), Box<dyn Error>> {
    reject_unknown(&options, &["--input-json", "--format"])?;
    let format = output_format(&options)?;
    let ui = TerminalUi::detect();
    ui.header(
        "METRICS · SURFACE DIAGNOSTICS",
        "pengulangan karakter  /  bukan skor koherensi",
    );
    let input = required(&options, "--input-json")?;
    if fs::metadata(input)?.len() > 1_000_000 {
        return Err("berkas JSON generasi melebihi batas 1 MiB".into());
    }
    let generated: GeneratedText = ui.run_stage("membaca hasil generasi", || {
        Ok::<_, Box<dyn Error>>(serde_json::from_slice(&fs::read(input)?)?)
    })?;
    let report = GenerationMetricsReport {
        repetition: generated.repetition_metrics(),
    };
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        OutputFormat::Text => print_repetition_metrics(&report.repetition),
    }
    Ok(())
}

fn evaluate_command(options: BTreeMap<String, String>) -> Result<(), Box<dyn Error>> {
    reject_unknown(
        &options,
        &["--checkpoint", "--dataset", "--batch-size", "--format"],
    )?;
    let format = output_format(&options)?;
    let ui = TerminalUi::detect();
    ui.header(
        "EVALUATE · HOLDOUT",
        "evaluasi split test  /  tidak melakukan update bobot",
    );
    let checkpoint_path = required(&options, "--checkpoint")?;
    let dataset_path = value_or(
        &options,
        "--dataset",
        "data/language/nusax/sentiment/indonesian/train.csv",
    );
    let model = ui.run_stage("memuat model checkpoint", || {
        GenerativeLanguageModel::load_checkpoint(checkpoint_path)
    })?;
    let manifest = model
        .corpus_manifest()
        .ok_or("checkpoint tidak memiliki manifest corpus")?;
    let split = DatasetSplitConfig {
        seed: manifest.split.seed,
        training_fraction: manifest.split.training_fraction_ppm as f64 / 1_000_000.0,
        validation_fraction: manifest.split.validation_fraction_ppm as f64 / 1_000_000.0,
        test_fraction: manifest.split.test_fraction_ppm as f64 / 1_000_000.0,
    };
    let corpus = ui.run_stage("memuat dataset + memeriksa manifest", || {
        let corpus = if dataset_path.ends_with(".json") {
            CorpusSplit::load_prepared_json(&dataset_path)?
        } else {
            CorpusSplit::load_nusax_training(
                dataset_path,
                &shella::language::LicenseRegistry::verified_sources(),
                split,
            )?
        };
        if corpus.manifest != *manifest {
            return Err("dataset/provenance/split tidak sama dengan checkpoint".into());
        }
        Ok::<_, Box<dyn Error>>(corpus)
    })?;
    let batch_size = parse_or(&options, "--batch-size", 8_usize)?;
    let metrics = ui.run_stage("menghitung metrik test", || {
        model.evaluate(&corpus.test, batch_size)
    })?;
    match format {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&metrics)?),
        OutputFormat::Text => print_evaluation_metrics(&metrics),
    }
    Ok(())
}

#[derive(Serialize)]
struct TrainRunReport<'a> {
    source: &'a shella::CorpusSourceManifest,
    split: &'a shella::SplitManifest,
    source_file_sha256: &'a str,
    source_license: &'a str,
    training_characters: usize,
    validation_characters: usize,
    test_characters: usize,
    training_model_tokens: usize,
    validation_model_tokens: usize,
    test_model_tokens: usize,
    baseline_validation: shella::EvaluationMetrics,
    final_validation: shella::EvaluationMetrics,
    training: shella::TrainingReport,
    final_test: shella::EvaluationMetrics,
    parameter_count: usize,
    checkpoint_path: PathBuf,
    checkpoint_sha256: String,
    training_and_evaluation_seconds: f64,
    notes: [&'static str; 5],
}

fn encoded_token_count(
    model: &GenerativeLanguageModel,
    records: &[shella::CorpusRecord],
) -> Result<usize, Box<dyn Error>> {
    let mut total = 0_usize;
    for record in records {
        let encoded = model.tokenizer().encode_document(&record.text)?.len();
        total = total
            .checked_add(encoded)
            .ok_or("jumlah token melampaui kapasitas integer")?;
    }
    Ok(total)
}

fn required<'a>(
    options: &'a BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, Box<dyn Error>> {
    options
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("opsi wajib {key}").into())
}

fn value_or(options: &BTreeMap<String, String>, key: &str, default: &str) -> String {
    options
        .get(key)
        .cloned()
        .unwrap_or_else(|| default.to_owned())
}

fn parse_or<T>(
    options: &BTreeMap<String, String>,
    key: &str,
    default: T,
) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    match options.get(key) {
        Some(value) => Ok(value.parse::<T>()?),
        None => Ok(default),
    }
}

fn reject_unknown(
    options: &BTreeMap<String, String>,
    allowed: &[&str],
) -> Result<(), Box<dyn Error>> {
    if let Some(unknown) = options.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("opsi tidak dikenal: {unknown}").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_output_formats_are_supported() {
        let text = BTreeMap::from([("--format".to_owned(), "text".to_owned())]);
        let json = BTreeMap::from([("--format".to_owned(), "json".to_owned())]);
        let invalid = BTreeMap::from([("--format".to_owned(), "yaml".to_owned())]);
        assert_eq!(output_format(&text).unwrap(), OutputFormat::Text);
        assert_eq!(output_format(&json).unwrap(), OutputFormat::Json);
        assert!(output_format(&invalid).is_err());
    }

    #[test]
    fn terminal_text_does_not_pass_ansi_control_sequences_through() {
        assert_eq!(
            terminal_safe_text("teks\u{1b}[31m merah\r\nbaru"),
            "teks\\u{001b}[31m merah\\r\nbaru"
        );
    }
}
