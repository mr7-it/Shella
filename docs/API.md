# Referensi API Rust

## Identitas crate

Paket, library, dan binari Cargo menggunakan nama `shella`. Aplikasi dapat memakai dependensi registry atau path, lalu mengimpor API dari crate library:

```toml
[dependencies]
shella = { path = "../shella" }
```

Versi minimum yang dinyatakan adalah Rust 1.85, edition 2024. Tipe error umum adalah `shella::SimError`; tidak ada panic yang dipakai untuk menolak input normal.

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

Untuk pembelajaran jangka panjang, `LearningMemory::open(path)` memulihkan pengalaman persisten. `record_feedback` menyimpan hasil evaluasi, `recall_similar` melakukan cosine retrieval, `infer_with_memory` menginjeksikan support ke neuron aksi LIF, `replay_relevant` menjalankan ulang pengalaman melalui reward-modulated STDP, `find_conflicts` mendeteksi label berbeda pada fitur mirip, `consolidate` melindungi asosiasi penting, dan `decay_irrelevant` melakukan pelupaan terkontrol. Retrieval score bukan probabilitas.

`TextEmbedding::from_text` membuat embedding hashing token/character n-gram deterministik. `MemoryStore::put_with_embedding` dan `search_embedding` menyediakan retrieval embedding; `LearningMemory::record_text_feedback` dan `recall_text_similar` menghubungkan embedding tersebut ke pengalaman pembelajaran. `MemoryStore::put_with_ttl` serta `purge_expired` menyediakan TTL otomatis.

`infer_task_with_memory_adaptive` hanya menginjeksikan memory ketika similarity teratas melewati ambang validasi; bila tidak, jaringan berjalan tanpa injeksi. Benchmark M10 menggabungkan kebijakan ini dengan temporal voting lima observasi sensor noisy dan fallback pada noise ekstrem. Kebijakan fallback dipakai karena hasil eksperimen menunjukkan memori yang dipaksakan pada sinyal sangat bising dapat menurunkan akurasi.

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

## API benchmark M11

`IrisSample` merepresentasikan satu baris data tanpa mengubah nilai asli. `stratified_split` membentuk train/validation/test per kelas, sedangkan `stratified_folds` membentuk k-fold hanya pada indeks non-test. `TrainScaler` fit pada indeks train dan `transform` menerapkan parameter tersebut ke validation/test. `evaluate_predictions` menghasilkan `SelfEvaluation` yang memuat `DecisionMetrics`, jumlah sampel lemah, dan jumlah sampel uncertainty.

Runner M11 memakai `LearningMemory::record_feedback`, `recall_similar`, `infer_task_with_memory_adaptive`, dan `replay_task_relevant` bersama `LifNetwork` reward-modulated STDP. Retrieval eksternal dilaporkan terpisah dari output neural sehingga kontribusi memory terhadap jalur LIF dapat dibandingkan secara eksplisit.
