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

## Evaluasi reward dan persistence

`Session::evaluate_actions(output, target, action_neurons, evaluator)` memetakan indeks spike neuron ke kanal aksi sesuai urutan slice. Metode `evaluate_actions_and_apply` menerapkan reward signed ke jaringan; metode itu memerlukan konfigurasi reward-modulated STDP pada jaringan. Evaluasi/metric transactional: kegagalan update tidak mencatat metrik parsial.

Gunakan `Engine::save_checkpoint(path, evaluator)` atau `Session::save_checkpoint(path, evaluator)` untuk state jaringan/evaluator. `Session::restore_checkpoint(path)` mengganti state jaringan sesi setelah validasi. Checkpoint tidak menyertakan encoder/policy statis atau file memori; pasang ulang konfigurasi itu saat startup dan pulihkan memori dari path store terpisah.

## Kontrak error dan konkurensi

Semua operasi simulasi/storage/SDK mengembalikan `Result<_, SimError>`. Varian mencakup input/konfigurasi/topologi invalid, overflow, alokasi, I/O, checkpoint invalid/versi tak didukung, data memori corrupt/versi data tidak didukung, dan kegagalan serialisasi. `MemoryStore` satu writer pada satu path; `Engine::session()` meminjam engine secara mutable, sehingga satu engine tidak dapat dipakai serentak tanpa koordinasi aplikasi.

API ini sinkron dalam proses. Tidak ada timeout, cancellation, transport, server, RPC, autentikasi, atau concurrency runtime bawaan karena tidak ada operasi jaringan/asinkron yang ditawarkan.
