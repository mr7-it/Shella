use shella::{DatasetLoader, LicenseRegistry, NusaXSplit, Vocabulary};
use std::{
    error::Error,
    hint::black_box,
    time::{Duration, Instant},
};

const BATCHES: usize = 7;
const LOOKUPS_PER_BATCH: usize = 20_000;
const ENCODE_PER_BATCH: usize = 2_000;

fn main() -> Result<(), Box<dyn Error>> {
    let registry = LicenseRegistry::verified_sources();
    let corpus = DatasetLoader::load_nusax_mt_csv(
        "data/language/nusax/mt/train.csv",
        NusaXSplit::Train,
        &registry,
    )?;
    let training = corpus
        .sentences
        .iter()
        .filter_map(|sentence| sentence.translations.get("indonesian"))
        .collect::<Vec<_>>();
    let vocabulary = Vocabulary::fit(training, 1, 50_000)?;
    let sample = vocabulary
        .tokens()
        .iter()
        .skip(1)
        .take(LOOKUPS_PER_BATCH)
        .cloned()
        .collect::<Vec<_>>();
    let text = corpus.sentences[0]
        .translations
        .get("indonesian")
        .expect("NusaX Indonesian training text");

    println!("dataset=nusax-mt-train-indonesian");
    println!("sentences={}", corpus.sentences.len());
    println!("vocabulary_tokens={}", vocabulary.len());
    println!("benchmark,operations_per_batch,median_ns_per_op,throughput_ops_per_sec");
    measure("word_lookup", LOOKUPS_PER_BATCH, || {
        for index in 0..LOOKUPS_PER_BATCH {
            let token = &sample[index % sample.len()];
            black_box(vocabulary.id(black_box(token)));
        }
        Ok::<_, std::convert::Infallible>(())
    })?;
    let word_ids = sample
        .iter()
        .map(|token| vocabulary.word_id(token))
        .collect::<Vec<_>>();
    measure("id_to_word_linear_baseline", LOOKUPS_PER_BATCH, || {
        for index in 0..LOOKUPS_PER_BATCH {
            let id = word_ids[index % word_ids.len()];
            let index = vocabulary
                .word_ids()
                .iter()
                .position(|candidate| *candidate == id);
            black_box(index.and_then(|index| vocabulary.token(index as u32)));
        }
        Ok::<_, std::convert::Infallible>(())
    })?;
    measure("id_to_word_hashbrown_index", LOOKUPS_PER_BATCH, || {
        for index in 0..LOOKUPS_PER_BATCH {
            let id = word_ids[index % word_ids.len()];
            black_box(vocabulary.word(black_box(id)));
        }
        Ok::<_, std::convert::Infallible>(())
    })?;
    measure("text_encode", ENCODE_PER_BATCH, || {
        for _ in 0..ENCODE_PER_BATCH {
            black_box(vocabulary.encode(black_box(text))?);
        }
        Ok::<_, shella::SimError>(())
    })?;
    measure("stable_word_id_encode", ENCODE_PER_BATCH, || {
        for _ in 0..ENCODE_PER_BATCH {
            black_box(vocabulary.encode_word_ids(black_box(text))?);
        }
        Ok::<_, shella::SimError>(())
    })?;
    Ok(())
}

fn measure<E>(
    name: &str,
    operations: usize,
    mut operation: impl FnMut() -> Result<(), E>,
) -> Result<(), E> {
    let mut batches = Vec::with_capacity(BATCHES);
    for _ in 0..BATCHES {
        let start = Instant::now();
        operation()?;
        batches.push(start.elapsed());
    }
    batches.sort_unstable();
    let median: Duration = batches[BATCHES / 2];
    let ns_per_op = median.as_nanos() as f64 / operations as f64;
    println!(
        "{name},{operations},{ns_per_op:.1},{:.0}",
        1.0e9 / ns_per_op
    );
    Ok(())
}
