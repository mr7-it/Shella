# Shella

**Shella** adalah library dan runtime neuromorfik dalam Rust. Implementasi mencakup neuron LIF, jaringan multi-neuron sinkron, STDP dan reward-modulated learning, checkpoint, memori JSON file-backed, keputusan berbasis skor yang dapat diaudit, preprocessing teks deterministik, serta SDK Rust bertipe. Paket Cargo, crate library, dan binari utama semuanya bernama `shella`.

## Kemampuan yang tersedia

- **Simulasi:** integrasi LIF eksak per interval, refraktori, catatan spike, parameter heterogen per neuron, sinaps berarah, delay, agregasi arus, serta commit jaringan dua fase.
- **Pembelajaran:** STDP pair-based, eligibility trace, reward/penalti signed, batas bobot, dan evaluator otomatis yang membaca spike output aktual.
- **Checkpoint:** format JSON berversi untuk runtime tunggal atau state jaringan lengkap, evaluator opsional, checksum integritas, validasi invariants, dan penulisan melalui file sementara/rename.
- **Memori persisten:** upsert dan CRUD, ID stabil, baca berdasarkan ID/kunci, pencarian prefix/tag, validasi ukuran/schema/checksum, serta pemulihan setelah proses dimulai ulang.
- **Keputusan:** skor fitur linear, jejak kontribusi, tie-break stabil, dan abstain berdasarkan ambang. Skor/margin bukan probabilitas atau confidence terkalibrasi.
- **Pipeline teks dasar:** normalisasi Unicode lowercase, tokenisasi berbasis aturan, vocabulary deterministik dari korpus, ID token, dan proyeksi bag-of-token ke arus jaringan.
- **SDK:** `Engine`/`Session` untuk input arus atau teks, output bertipe, evaluator aksi, decision policy, memory, dan checkpoint jaringan.
- **Evaluasi/benchmark:** conditioning dan continual-learning multi-seed dengan kontrol tanpa update bobot, ditambah timing jaringan multi-skala, checkpoint, dan pencarian memori.

## Keterbatasan yang disengaja

Pipeline teks **bukan** model semantik, pemahaman bahasa, generator, decoder, subword tokenizer, atau embedding. Proyeksi `token_id % jumlah_neuron` dapat mengalami collision. Decision engine hanya menghitung fitur yang diberikan aplikasi; tidak ada probabilitas/kalibrasi. Memori adalah store lokal satu-penulis tanpa lock lintas proses, enkripsi, TTL, atau pencarian semantik. Checksum FNV-1a hanya mendeteksi kerusakan tak disengaja, bukan autentikasi. SDK sinkron dalam proses, bukan HTTP/RPC server. Benchmark mengukur perilaku protokol tetap pada mesin pengujian dan tidak membuktikan generalisasi, performa aplikasi, ataupun efisiensi energi.

## Memulai

Prasyarat: Rust 1.85 atau lebih baru (MSRV di `Cargo.toml` adalah 1.85; edition 2024).

```sh
cargo run --release --locked
cargo build --release --lib --locked
cargo test --all-targets --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo bench --bench learning --locked
cargo bench --bench system --locked
```

Dalam proyek Rust lain, tambahkan dependensi bernama `shella`, lalu impor API dari crate `shella`. Contoh integrasi ada di [Panduan API](docs/API.md) dan [SDK](docs/SDK.md). Dependency langsung runtime hanya `serde` dan `serde_json`; versi serta dependency transitif dikunci di `Cargo.lock`.

## Peta dokumentasi

- [Arsitektur](docs/ARSITEKTUR.md) dan [referensi API](docs/API.md)
- [Model matematis](docs/MODEL_MATEMATIS.md), [jaringan](docs/NETWORK.md), [STDP dan reward](docs/STDP.md)
- [Checkpoint](docs/CHECKPOINT.md), [memori](docs/MEMORI.md), [keputusan](docs/KEPUTUSAN.md)
- [Pipeline teks](docs/BAHASA.md), [SDK](docs/SDK.md), [keamanan](docs/KEAMANAN.md)
- [Evaluator dan benchmark](docs/EVALUATOR_BENCHMARK.md), [pengujian](docs/PENGUJIAN.md)
- [Audit Milestone 5](LAPORAN_MILESTONE_5.md), [progres Milestone 6](LAPORAN_MILESTONE_6.md), dan [hasil validasi Milestone 5](LAPORAN_VALIDASI.md)

Laporan Milestone 1–4 adalah snapshot historis. `LAPORAN_VALIDASI.md` mencatat pengujian kapasitas/SDK sebelum eksperimen M6; hasil continual-learning dan gerbang mutu terbaru ada di `LAPORAN_MILESTONE_6.md`.


Catatan: `LAPORAN_VALIDASI.md` adalah snapshot validasi sebelum eksperimen continual-learning; hasil dan gerbang mutu Milestone 6 dicatat di `LAPORAN_MILESTONE_6.md`.
