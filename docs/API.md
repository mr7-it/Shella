# Referensi API Rust

## Identitas crate

Paket, library, dan binari Cargo menggunakan nama `shella`. Aplikasi dapat memakai dependensi registry atau path, lalu mengimpor API dari crate library:

```toml
[dependencies]
shella = { path = "../shella" }
```

Versi minimum yang dinyatakan adalah Rust 1.85, edition 2024. Tipe error umum adalah `shella::SimError`; tidak ada panic yang dipakai untuk menolak input normal.

## Class Knowledge Iris

`ClassKnowledgeModel::fit(train, validation, dataset_version, training_run_id, seed)` membuat scaler dari train saja dan menguji `k = 2, 3, 4, 5, 6, 8` prototype per kelas di validation. Semua validation score disimpan di model; jika hasil seri, k yang lebih kecil dipilih. Test split tidak diterima pada fungsi fit.

`fit_with_exemplar_policy(..., ExemplarPolicy::new(max_per_class)?)` mengatur budget exemplar per kelas (`1..=64`); `fit` menggunakan default dua. Urutan exemplar: representatif kelas, boundary margin rendah, lalu representatif cluster lain. Model memuat ID, versi, parent version dan status awal `Candidate`; ini bukan formal model registry atau approval workflow.

Untuk resume terpadu, `Checkpoint::save_knowledge` menyimpan network, ClassKnowledge, mapping action neurons, dan evaluator opsional; `Checkpoint::restore_knowledge` mengembalikan `RestoredKnowledgeState` setelah validasi integrity.

`predict([sepal_length, sepal_width, petal_length, petal_width])` mengembalikan class/prototype ID, score, margin, dan confidence Platt one-vs-rest yang di-fit dari validation. Score dan margin bukan probabilitas; confidence kalibrasi tidak otomatis menjamin kualitas di luar benchmark Iris. `infer_with_lif` mengukur probe baseline serta probe dengan arus knowledge pada LIF aktual. Persistence checksummed tersedia lewat `LearningMemory::save_class_knowledge`/`load_class_knowledge`; persistence terenkripsi tersedia lewat `ClassKnowledgeModel::save_secure`/`load_secure` dengan `SecureMemoryStore`.

`LearningMemory::migrate_legacy_prototypes` secara eksplisit membaca satu pusat lama untuk tiap kelas, mempertahankan nilainya di `legacy_snapshot`, lalu membentuk model baru dari train/validation. Lihat [schema Class Knowledge](KNOWLEDGE_SCHEMA.md) untuk struktur dan batasan.

## Membuat jaringan dan mengirim arus

```rust
use shella::{Engine, LifConfig, LifNetwork, SdkInput};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let neuron = LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 1);
    let network = LifNetwork::new(neuron, 2)?;
    let mut engine = Engine::new(network);
    let output = engine
        .session()
        .step(SdkInput::Currents(vec![100.0, 0.0]))?;
    println!("neuron spike: {:?}", output.spiking_neurons());
    Ok(())
}
```

`Session::step` menjalankan satu langkah jaringan LIF yang sebenarnya. Vektor arus harus memiliki satu nilai finite untuk setiap neuron. Besaran arus memakai nA dan dinamika waktu/delay mengikuti `dt_ms` konfigurasi.

## Input teks, keputusan, dan memori

Tambahkan `TextEncoder` yang dibangun dari `Vocabulary::fit(corpus, min_frequency, max_tokens)`, `DecisionPolicy`, atau `MemoryStore` dengan builder `with_text_encoder`, `with_decision_policy`, dan `with_memory`. Input teks hanya diproses jika encoder dipasang:

```rust
SdkInput::Text {
    text: "contoh input".to_owned(),
    current_per_token_na: 20.0,
}
```

`Session::decide(&[Candidate])` mengembalikan skor linear, kontribusi fitur, margin, ID terpilih, dan alasan atau keputusan abstain. Skor/margin bukan confidence probabilistik. `Session::remember(key, serde_json::Value, tags)` melakukan upsert pada `MemoryStore` file-backed.

Untuk identitas dan koneksi bahasa-neuron yang persisten, `Vocabulary::word_id`
mengembalikan `WordID` yang disimpan di vocabulary, sementara lookup token memakai
hashbrown SwissTable (hash hanya untuk lookup, bukan nilai ID). `LanguageNetwork`
mengambil ownership `LifNetwork` dan vocabulary, membagi kata ke subword ber-ID,
serta mengikat kata ke satu atau lebih `NeuronID`:

```rust
use shella::{LanguageNetwork, LifConfig, LifNetwork, Vocabulary};

let vocabulary = Vocabulary::fit(["alpha beta"], 1, 16)?;
let network = LifNetwork::new(
    LifConfig::new(1.0, 10.0, 1.0, -65.0, -70.0, -60.0, 0),
    4,
)?;
let mut language = LanguageNetwork::new(network, vocabulary)?;
let alpha = language.vocabulary().word_id("alpha");
let representation = [
    language.network().neuron_id(0).expect("neuron exists"),
    language.network().neuron_id(1).expect("neuron exists"),
];
language.bind_word(alpha, &representation)?;
let currents = language.encode_currents("alpha", 20.0)?;
assert_eq!(currents, [10.0, 10.0, 0.0, 0.0]);
# Ok::<(), shella::SimError>(())
```

`connect_words` membuat koneksi all-to-all antar representasi dengan `SynapseID`;
sinaps tersebut ikut propagasi dan STDP `LifNetwork`. `step_text` memproyeksikan
teks dan menjalankan satu langkah jaringan. `Checkpoint::save_language_network` /
`restore_language_network` menyimpan dan memulihkan vocabulary, ID kata/subword,
binding neuron, relasi sinaps, bobot, trace learning, dan state jaringan dalam satu
checkpoint checksummed. `ReasoningEngine::reason_text` mengubah teks menjadi arus
pada representasi neuron yang kompatibel, lalu melewati reasoning dan retrieval
memory biasa. Ini adalah representasi terikat eksplisit, bukan embedding semantik.

Untuk serialisasi vocabulary mandiri, `Vocabulary::save(path)` melakukan atomic
replace dan `Vocabulary::load(path)` memvalidasi ulang ID serta indeks dua arah.
`encode_word_ids`/`decode_word_ids` memakai ID bertipe stabil; `WordID(0)` secara
eksplisit merepresentasikan `<unk>`, sedangkan ID lain yang tidak dikenal ditolak.
`TextEncoder::encode` tetap mempertahankan field `token_ids` positional demi
kompatibilitas dan menyediakan `word_ids` bertipe untuk penggunaan baru.

Integritas semua file data yang dibundel dapat diverifikasi ulang:

```rust
use shella::DatasetIntegrityManifest;

let manifest = DatasetIntegrityManifest::bundled()?;
let report = manifest.verify_directory("data/language")?;
assert_eq!(report.files_verified, 63);
# Ok::<(), shella::SimError>(())
```

File Hunspell eksternal tidak ikut diverifikasi oleh `verify_directory`; dua
checksumnya dicatat pada manifest yang sama dan diverifikasi saat `load_hunspell_files`
membaca file. CLI `dataset-integrity verify-source hunspell_dictionaries
Indonesia.dic /path/to/Indonesia.dic` melakukan pemeriksaan terpisah.

Untuk pembelajaran jangka panjang, `LearningMemory::open(path)` memulihkan pengalaman persisten. `record_feedback` menyimpan hasil evaluasi, `recall_similar` melakukan cosine retrieval, `infer_with_memory` menginjeksikan support ke neuron aksi LIF, `replay_relevant` menjalankan ulang pengalaman melalui reward-modulated STDP, `find_conflicts` mendeteksi label berbeda pada fitur mirip, `consolidate` melindungi asosiasi penting, dan `decay_irrelevant` melakukan pelupaan terkontrol. Retrieval score bukan probabilitas.

`TextEmbedding::from_text` membuat embedding hashing token/character n-gram deterministik. `MemoryStore::put_with_embedding` dan `search_embedding` menyediakan retrieval embedding; `LearningMemory::record_text_feedback` dan `recall_text_similar` menghubungkan embedding tersebut ke pengalaman pembelajaran. `MemoryStore::put_with_ttl` serta `purge_expired` menyediakan TTL otomatis.

## Library kosakata Indonesia

`IndonesianVocabulary` menyediakan wadah lexicon terversi dan tervalidasi.
`LicenseRegistry::verified_sources()` mengizinkan stopwords-id, UD Indonesian-PUD,
WordNet Bahasa, NusaX, IndoSum, dan Hunspell hanya pada revisi/hash yang tercatat.
WordNet Bahasa r24 menggunakan MIT. Impornya hanya menerima label mutu Y/O dan
menghubungkan pasangan lema dengan synset yang sama ketika pemetaan lema tidak
ambigu; lihat [`docs/DATASET_BAHASA_INDONESIA.md`](./DATASET_BAHASA_INDONESIA.md)
untuk identitas, jumlah, dan batas cakupan data. GSD/CSUI tetap tidak aktif karena
hak teks sumber belum dipastikan; IndoNLI tidak memiliki lisensi dataset yang
terverifikasi.

```rust
use shella::{
    IndonesianVocabulary, LexiconEntryInput, LexiconMetadata, LinguisticMetadata,
    SourceProvenance,
};

let mut lexicon =
    IndonesianVocabulary::new(LexiconMetadata::indonesian("my-import", "1"))?;
let entry = LexiconEntryInput {
    headword: "istilah".to_owned(),
    sources: vec![SourceProvenance {
        source_name: "Nama sumber yang diverifikasi aplikasi".to_owned(),
        source_uri: "https://source.example/entry/identifier".to_owned(),
        license: "Lisensi dan izin yang berlaku".to_owned(),
        license_uri: "https://source.example/license".to_owned(),
        source_entry_id: "identifier".to_owned(),
    }],
    linguistic: LinguisticMetadata::default(),
};
let word_id = lexicon.upsert(entry)?;
assert_eq!(lexicon.lookup("ISTILAH")?.unwrap().word_id, word_id);
# Ok::<(), shella::SimError>(())
```

Contoh di atas hanya menunjukkan bentuk API dan bukan sumber atau kosakata nyata.
`upsert` mempertahankan ID saat kata diperbarui; `import_json` memvalidasi versi,
metadata dataset, dan seluruh batch secara atomik. `to_json`/`from_json` mengelola
representation JSON berversi, sedangkan `Checkpoint::save_indonesian_vocabulary`
dan `restore_indonesian_vocabulary` menyimpan/memulihkan state dengan checksum
serta atomic replace. Skema interoperabilitas JSON tersedia di
[`schemas/indonesian-vocabulary-v1.schema.json`](../schemas/indonesian-vocabulary-v1.schema.json).
Variasi ejaan merupakan alias lookup setelah normalisasi NFC.
Definisi hanya diterima bila merujuk `source_entry_id` yang tercatat pada entri.
`segment_unlisted` melakukan segmentasi karakter ortografis saja dan tidak
menyatakan kata baru memiliki arti atau analisis morfologi tertentu.

`into_language_network(network)` membawa vocabulary dan `WordID` ke
`LanguageNetwork`; aplikasi mengikat ID kata ke neuron memakai `bind_word`.
Tidak ada relasi neural otomatis dari definisi atau kelas kata.

`DatasetLoader::load_ud_conllu`, `load_stopwords_json`,
`load_nusax_mt_csv`, `load_nusax_sentiment_csv`, `load_nusax_lexicon_csv`,
`load_indosum_archive`, dan `load_hunspell_files` adalah batas impor berkas lokal.
Loader memeriksa hash sebelum parse. Stopwords dapat diterapkan ke input aktual
melalui `LanguageNetwork`; PUD mempertahankan anotasi token/lemma/dependency;
NusaX sentiment menyediakan tokenisasi dan evaluator caller-prediction pada split
held-out; NusaX lexicon dapat mengintern token ke vocabulary neural; IndoSum
memuat satu split JSONL dan menghitung token-level ROUGE-1. Tidak satu pun
menyediakan model translation, sentiment, NLI, atau summarization terlatih.
GSD/CSUI tetap ditolak registry. Hunspell menyediakan root dan analisis langsung
satu aturan afiks, bukan seluruh perilaku mesin Hunspell. Anotasi dependency UD
bukan relasi semantik.

`infer_task_with_memory_adaptive` hanya menginjeksikan memory ketika similarity teratas melewati ambang validasi; bila tidak, jaringan berjalan tanpa injeksi. Benchmark M10 menggabungkan kebijakan ini dengan temporal voting lima observasi sensor noisy dan fallback pada noise ekstrem. Kebijakan fallback dipakai karena hasil eksperimen menunjukkan memori yang dipaksakan pada sinyal sangat bising dapat menurunkan akurasi.

`ReasoningEngine::new(network, learning_memory, config)` menyediakan jalur reasoning
iteratif terpisah dari SDK. `ReasoningConfig` menetapkan jumlah langkah (1..=1024),
batas retrieval, arus memori, laju peluruhan temporal per detik, dan confidence minimum.
`reason(context_id, currents, action_neurons)` menggunakan cosine retrieval per
context, memberi bobot lebih pada record relevan yang lebih baru, menginjeksikan
dukungan memori sebagai arus LIF, lalu melanjutkan propagasi jaringan selama beberapa
langkah. `ReasoningResult` berisi trace tiap transisi, recall, bobot perhatian,
dukungan aksi, uncertainty entropy, evidence strength, skor heuristik, dan aksi
terpilih atau abstention. `ReasoningResult::confidence()` bukan probabilitas.
Untuk confidence terkalibrasi, panggil
`fit_confidence_calibration(validation_samples, evaluation_samples, iterations,
learning_rate)`: Platt dilatih pada validation samples dengan outcome benar dan
salah, lalu Brier/MAE dihitung hanya pada evaluation samples terpisah. Pemanggil
wajib menjaga kedua split berbeda satu sama lain dan dari data pelatihan model utama.
`calibrated_confidence()` bernilai `None` sebelum kalibrasi.

`learn_from_feedback` menyimpan pengalaman melalui `LearningMemory::record_feedback`.
Jika STDP aktif, input fitur dan spike pengajar dipresentasikan ke network; bobot
diperbarui oleh STDP, dan reward signed diterapkan jika reward-modulated STDP aktif.
Tanpa STDP, fungsi memakai update readout yang tersedia. Pembelajaran STDP berjalan
pada salinan jaringan dan hasilnya dikomit hanya setelah penyimpanan feedback berhasil.
Timestamp pengalaman Unix nanodetik dipersistenkan; record lama tanpa timestamp
dimigrasikan saat dibuka, sedangkan timestamp kosong atau lebih dari lima detik di
masa depan ditolak sebagai data korup. Setiap transisi LIF transactional, tetapi
panggilan reasoning multi-langkah tidak membatalkan langkah terdahulu bila langkah
selanjutnya gagal.

Fitur reasoning tambahan:

- `with_context_window(capacity)` mengaktifkan arus input dari beberapa pemrosesan
  terakhir (maksimum 4096); context yang berganti mengosongkan jendela sebelumnya.
- `reason_for_goal` menggabungkan input dan fitur tujuan untuk mengarahkan query
  retrieval, tanpa mengubah input arus yang masuk ke jaringan.
- `consolidate_memories` menggabungkan record fitur-mirip dengan task dan label sama;
  label konflik tetap dipisah. `decay_memories` meluruhkan importance dari waktu
  decay terakhir, dengan half-life yang diperpanjang hitungan penggunaan.
- `causal_associations` menghitung ko-occurence fitur-label yang berulang; ini
  bukan bukti sebab-akibat intervensional. `ReasoningResult::contradictions()` berisi
  pasangan recall mirip yang menyimpan label berbeda.
- `uncertainty_decomposition()` memisahkan entropy dukungan prediksi dari gap bukti.
  `with_adaptive_compute(true)` memilih langkah 1..=batas `iterations` dari input,
  entropy recall, dan rasio kontradiksi.
- `replay_experiences` menjalankan pengalaman relevan melalui replay LIF/STDP.
- `save_checkpoint`/`restore_checkpoint(path, learning_memory)` menyimpan state
  reasoning, context, konfigurasi, kalibrator beserta provenance dan metrik. Memory
  tetap file terpisah; restore memeriksa fingerprint record agar mismatch ditolak.

Untuk input bahasa, `ReasoningEngine::respond_text(context_id, text,
language_network, current_per_word_na, action_neurons, retrieval_limit)` menjalankan
respons deterministik dari `SemanticMemory` dan inference LIF/retrieval melalui
engine yang sama. Response menyertakan `SemanticQuestionIntent`, evidence class,
gap report, statement yang mendukung, ID sumber, dan provenance/license bila edge
atau gloss berasal dari dataset. Definisi memakai gloss yang dikaitkan ke source
record; relasi memakai trace `infer` dan menolak trace berstatus `Hypothesis` atau
confidence eksplisit nol sebagai fakta. Pertanyaan ya/tidak dicocokkan dengan frame
subjek/predikat/objek dan negasi pada konteks yang sama; sinonim source assertion
non-hipotesis dapat dipakai sebagai kecocokan konsep tanpa diperlakukan sebagai
confidence terkalibrasi. `Siapa yang ...?` menjawab dari frame positif yang cocok;
bentuk WH lain abstain. Pertanyaan tersimpan tidak boleh menjadi evidence. Bila
konsep ambigu/tidak dikenal, bukti tidak ada, atau sumber bertentangan, response
abstain atau melaporkan konflik. `SemanticConfidenceBand` bukan probabilitas.
Perakitan kalimat masih deterministik dan terbatas, bukan generator neural/LLM.
`ReasoningResult` tetap prediction/action-level; tanpa mapping eksplisit action-label
ke proposition, ia tidak digunakan sebagai sumber fakta pada jawaban. Detail intent
dan batas teruji ada di [`docs/BAHASA.md`](./BAHASA.md).

SDK memaparkan delegasi tipis pada `Session::respond_with_reasoning`:

```rust
let (response, reasoning_result) = session.respond_with_reasoning(
    &mut reasoning_engine,
    "dialog-1",
    "Apakah kucing makan ikan?",
    &language_network,
    20.0,
    &[action_neuron],
    8,
)?;
```

`ReasoningEngine` tetap pemilik authoritative `SemanticMemory` dan `LearningMemory`;
caller juga memberi `LanguageNetwork` dengan `WordID`, vocabulary, serta identitas
neuron yang cocok. API lama `Session::step` tidak berubah. Pembelajaran berjalan
hanya melalui feedback eksplisit, misalnya
`Session::learn_from_semantic_feedback(reasoning_engine, context_id, text,
language_network, current_per_word_na, expected_action, predicted_action,
action_neurons, learning_rate)`. Input dengan token unknown, ambiguitas, atau
konflik tidak memperbarui readout/STDP atau menulis pengalaman learning.

## Model generatif autoregresif

`GenerativeLanguageModel` menggunakan decoder-only Transformer Candle 0.8.4,
tokenizer karakter Unicode, dan CPU. Tokenizer hanya di-fit pada train split
NusaX-Senti Indonesia dan model tidak memakai `WordID` sebagai token neural.
`Session::generate_text` mendelegasi ke model yang caller berikan tanpa membaca
`SemanticMemory`; method baru `generate_text_with_semantic_memory` di bagian bawah
menyisipkan recall sebagai teks context, tetapi tidak memverifikasi klaim:

```rust
let model = GenerativeLanguageModel::load_checkpoint("shella-nusax.sllm")?;
let generated = session.generate_text(
    &model,
    "bahasa Indonesia",
    GenerationConfig::default(),
)?;
println!("{}", generated.full_text);
// generated.factual_grounding menjelaskan bahwa teks belum diverifikasi.
```

Perintah lokal dari root repository:

```sh
cargo run --locked --bin shella-lm -- train --checkpoint shella-nusax.sllm
cargo run --locked --bin shella-lm -- evaluate --checkpoint shella-nusax.sllm
cargo run --locked --bin shella-lm -- generate --checkpoint shella-nusax.sllm --prompt "bahasa Indonesia"
```

`train` membandingkan baseline/final validation; test internal dievaluasi hanya
setelah training selesai, dan label sentiment bukan target generasi. JSON report
memuat source hash/license, split, deduplikasi, konfigurasi serta metrik aktual.
`evaluate` menolak dataset/split/hash yang tidak sama dengan checkpoint. Teks
generatif belum memiliki evidence/provenance faktual dan tidak menggantikan
`SemanticMemory` untuk menjawab pertanyaan fakta. Perplexity tidak mengukur
factuality, koherensi, atau generalisasi populasi.

Gunakan `fit_confidence_calibration_with_ids` dengan `CalibrationSample` untuk
memastikan ID unik dan split validation/evaluation tidak overlap. API lama tanpa ID
dipertahankan demi kompatibilitas; pemanggilnya tetap bertanggung jawab atas split.

Untuk data sensitif, gunakan `SecureMemoryStore::open(path, encryption_key, signing_key)`. Payload dienkripsi AES-256-GCM dan ditandatangani Ed25519. Setiap `put` memperoleh lock transaksi, membaca snapshot terbaru, lalu melakukan atomic commit sehingga beberapa writer dapat bergantian tanpa lock sepanjang umur object. Kunci 32-byte harus dikelola di luar aplikasi dan tidak boleh ditulis ke repository.

Untuk evaluasi label keputusan secara terpisah dari reward jaringan, gunakan `DecisionEvaluator::record(expected, predicted, abstention_reason)` atau `evaluate_decision(expected, &decision)`. `DecisionMetrics` menyediakan jumlah TP/FP/FN, abstain dan alasannya, akurasi, precision, recall, false-positive rate pada target diam, serta false-negative rate pada target aksi. Rasio dengan denominator nol menghasilkan `None`.

`CalibrationEvaluator` adalah jalur terpisah untuk confidence probabilistik yang sudah dikalibrasi pada split validasi. Ia menolak nilai di luar `0..1` dan menyediakan Brier score serta mean absolute calibration error. Score linear dan margin decision tidak boleh dimasukkan ke API ini.

`PlattCalibrator::fit(samples, iterations, learning_rate)` melatih sigmoid dari score dan label validasi. `IsotonicCalibrator::fit(samples)` memakai pool-adjacent-violators untuk kalibrasi monoton. Keduanya hanya boleh dilatih pada split validasi, bukan test, dan `predict` menghasilkan confidence pada rentang `0..1`.

## Evaluasi reward dan persistence

`Session::evaluate_actions(output, target, action_neurons, evaluator)` memetakan indeks spike neuron ke kanal aksi sesuai urutan slice. Metode `evaluate_actions_and_apply` menerapkan reward signed ke jaringan; metode itu memerlukan konfigurasi reward-modulated STDP pada jaringan. Evaluasi/metric transactional: kegagalan update tidak mencatat metrik parsial.

Gunakan `Engine::save_checkpoint(path, evaluator)` atau `Session::save_checkpoint(path, evaluator)` untuk state jaringan/evaluator. `Session::restore_checkpoint(path)` mengganti state jaringan sesi setelah validasi. Checkpoint tidak menyertakan encoder/policy statis atau file memori; pasang ulang konfigurasi itu saat startup dan pulihkan memori dari path store terpisah.

## Kontrak error dan konkurensi

Semua operasi simulasi/storage/SDK mengembalikan `Result<_, SimError>`. Varian mencakup input/konfigurasi/topologi invalid, overflow, alokasi, I/O, checkpoint invalid/versi tak didukung, data memori corrupt/versi data tidak didukung, dan kegagalan serialisasi. `MemoryStore` satu writer pada satu path; `Engine::session()` meminjam engine secara mutable, sehingga satu engine tidak dapat dipakai serentak tanpa koordinasi aplikasi.

API ini sinkron dalam proses. Tidak ada timeout, cancellation, transport, server, RPC, autentikasi, atau concurrency runtime bawaan karena tidak ada operasi jaringan/asinkron yang ditawarkan.

## Analisis semantik dan semantic memory

`SemanticMemory::open(path, IndonesianVocabulary)` memakai `MemoryStore` yang
sudah ada untuk menyimpan lexicon, graph, phrase, dan konteks dalam satu snapshot
JSON atomik. Tambahkan entry dengan `learn_entry`; WordID lama dipertahankan.
`add_phrase(phrase, kind, meaning, evidence)` mewajibkan seluruh token terdaftar.
`add_relation(ConceptRelation)` mengembalikan `true` bila edge baru berkonflik
dengan sinonim/antonim yang sudah ada; konflik tetap tersimpan dan terlihat.

`analyze(text)` mengembalikan token ber-WordID, span byte pada teks ternormalisasi,
phrase match, lemma ambigu, struktur role, dan negasi opsional. `analyze_in_context`
menyimpan percakapan dan memberi kandidat resolusi pronomina satu kalimat.
`retrieve(query, context_id, limit)` memberikan skor relevansi serta alasan;
`infer(word_id, relation_kind)` menghasilkan simpulan dengan path dan ID bukti.
Semua skor confidence/relevansi adalah input atau skor heuristic, bukan probabilitas
yang dikalibrasi.

Pasang instance dengan `Engine::with_semantic_memory`. `Session::analyze_semantics`
dan `analyze_semantics_in_context` mengaksesnya. `Session::resonate_semantics`
menerima `LanguageNetwork` terpisah yang `WordID`-nya harus cocok dengan lexicon;
sinonim ber-confidence dan non-hipotesis dapat menambah arus ke binding neuron
sebelum satu langkah LIF/STDP. Ini tidak otomatis menyambungkan semantic store ke
`LearningMemory` pada `Engine` SDK. Untuk jalur pembelajaran terintegrasi, pasang
`SemanticMemory` pada `ReasoningEngine::with_semantic_memory`, lalu gunakan
`learn_from_semantic_feedback(context_id, text, language_network, current_per_word,
expected_action, predicted_action, action_neurons, learning_rate)`. Jalur ini
menyimpan konteks, mengembalikan gap, memakai semantic currents, merekam pengalaman
di `LearningMemory`, dan mengupdate jaringan aktual dengan aturan readout/STDP yang
sudah ada. Pengalaman tetap berupa fitur neural, tidak menyimpan teks mentah.

`ReasoningEngine::remove_semantic_concept` menyinkronkan deletion journal semantic
ke `LearningMemory::discard_task` dan menghapus context frame terkait. Jurnal
disimpan persisten serta diproses ulang setelah kegagalan. Checkpoint reasoning
menyimpan fingerprint semantic opsional; restore lama tanpa fingerprint tetap
berlaku untuk checkpoint versi sebelumnya. `Engine` SDK menyediakan
`with_learning_memory`, `save_integrated_checkpoint`, dan
`restore_integrated_checkpoint`; restore membandingkan fingerprint tiga store
sebelum memodifikasi LIF network. Snapshot terintegrasi tetap mengacu pada file
store eksternal, bukan menggandakan kontennya, sehingga path dan file store harus
dipulihkan lebih dahulu. Cross-file update bukan transaksi atomik umum.

`evaluate_splits(training, validation, test)` menolak ID contoh atau teks
ternormalisasi berulang antar split, lalu melaporkan exact-frame accuracy, slot
accuracy, dan token coverage. Fungsi ini tidak melatih atau memilih model; pemanggil
tetap bertanggung jawab atas pemisahan sumber data sebelum belajar. Lihat
[batas parser, sumber, dan interpretasi metrik](BAHASA.md#semantic-memory-dan-analisis-berbukti).
`evaluate_grouped_splits` juga menolak `group_id` sumber/semantik lintas split.

## API benchmark M11

`IrisSample` merepresentasikan satu baris data tanpa mengubah nilai asli. `stratified_split` membentuk train/validation/test per kelas, sedangkan `stratified_folds` membentuk k-fold hanya pada indeks non-test. `TrainScaler` fit pada indeks train dan `transform` menerapkan parameter tersebut ke validation/test. `evaluate_predictions` menghasilkan `SelfEvaluation` yang memuat `DecisionMetrics`, jumlah sampel lemah, dan jumlah sampel uncertainty.

Runner M11 memakai `LearningMemory::record_feedback`, `recall_similar`, `infer_task_with_memory_adaptive`, dan `replay_task_relevant` bersama `LifNetwork` reward-modulated STDP. Retrieval eksternal dilaporkan terpisah dari output neural sehingga kontribusi memory terhadap jalur LIF dapat dibandingkan secara eksplisit.

## Perbaikan M11

`LifNetwork::apply_rewarded_readout` melakukan bounded signed update pada sinaps readout aktual dan melengkapi eligibility reward-modulated STDP. `LearningMemory::record_prototype` dan `load_prototypes` menyimpan serta memulihkan prototype neural terkonsolidasi. `LearningMemory::discard_task` menjadi rollback untuk feedback task yang tidak meningkatkan validation. Runner memakai prototype yang fit dari train saja, bukan test atau retrieval eksternal.


## Generasi neural dan SemanticMemory

API kompatibel `Session::generate_text(&model, prompt, config)` tetap menghasilkan
kelanjutan autoregresif tanpa retrieval. Untuk memasukkan memory secara aktual, pasang
`SemanticMemory` yang sudah ada pada `Engine`, lalu panggil method tambahan:

```rust
use shella::{
    GenerationConfig, GenerativeLanguageModel, MemoryGenerationStatus,
};

let result = engine.session().generate_text_with_semantic_memory(
    &model,
    "kucing makan apa?",
    Some("pet-care"),
    4,
    GenerationConfig {
        max_new_tokens: 32,
        temperature: 0.0,
        top_k: None,
        top_p: None,
        seed: 7,
    },
)?;
if result.status == MemoryGenerationStatus::Generated {
    let generated = result.generated.as_ref().expect("status generated");
    println!("{}", generated.continuation);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Method baru membaca `SemanticMemory` dari Engine yang sama; ia tidak membuat salinan
lexicon atau memory. `context_id` dapat `None` untuk retrieval lintas konteks.
`retrieval_limit` maksimum 32. Hasil `MemoryConditionedGeneration` menyediakan
`retrieved_memory` dan `included_memory` secara terpisah supaya pemanggil dapat melihat
recall mana yang benar-benar masuk ke prompt. Status `NoRelevantMemory` dan
`EvidenceOutsideContextWindow` menahan generasi (`generated == None`). Output yang
terjadi tetap `unverified`; memory disisipkan sebagai konteks teks, bukan verifikasi
entailment, provenance klaim, atau instruction tuning. Untuk jawaban faktual konservatif
gunakan `SemanticMemory::respond_in_context` / `Session::respond_with_reasoning`.

`GeneratedText::repetition_metrics()` mengembalikan diagnostik token-karakter:
rasio karakter unik, repeated bigram/trigram/4-gram, adjacent-repeat rate, apakah EOS
menghentikan output, serta apakah prompt dipotong. Ini bukan metrik koherensi/factuality.
Decoding menghindari karakter 4-gram yang pernah muncul pada kelanjutan bila tersedia
alternatif; ia tidak mengubah bobot atau schema checkpoint.

CLI:

```sh
cargo run --locked --bin shella-lm -- train \
  --dataset data/language/nusax/sentiment/indonesian/train.csv \
  --checkpoint target/model-baru.sllm --context 128
cargo run --locked --bin shella-lm -- evaluate --checkpoint target/model-baru.sllm
cargo run --locked --bin shella-lm -- generate \
  --checkpoint target/model-baru.sllm --prompt "Bagaimana cara merawat tanaman di rumah?" \
  --temperature 0 --max-new-tokens 80
cargo run --locked --bin shella-lm -- metrics --input-json target/hasil-generate.json
```

Training hanya memakai NusaX-Senti train resmi yang dibagi internal train/validation/test;
validation dipakai selama training dan test dievaluasi setelah training selesai. `metrics`
hanya mengukur pengulangan permukaan dari berkas hasil generasi dan tidak menilai
koherensi atau kebenaran bahasa.
