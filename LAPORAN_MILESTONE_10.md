# Laporan Milestone 10 — Memori jangka panjang dan pembelajaran adaptif

Tanggal: **19 Oktober 2026**. Repository: `mr7-it/Shella`. Implementasi dilakukan di atas commit M10 yang sudah memuat sebagian fondasi memori; audit dilakukan sebelum perubahan tambahan. Tidak ada klaim akurasi maksimum, pemahaman bahasa, atau validasi pada perangkat deployment.

## Audit sebelum implementasi

Audit source menemukan fondasi tersebut serta celah nyata: belum ada embedding, secure envelope, TTL, transaksi multi-writer, atau fitting kalibrasi. Kekurangan itu ditutup tanpa menghapus tes lama.

## Perubahan file

- `src/decision/evaluation.rs`: menambahkan `CalibrationEvaluator` dan `CalibrationMetrics` dengan validasi confidence `0..1`, Brier score, mean absolute calibration error, serta unit regression test. Score linear/margin tidak diperlakukan sebagai probabilitas.
- `src/decision/calibration.rs`, `src/decision/mod.rs`, dan `src/lib.rs`: menambahkan Platt scaling gradient descent serta isotonic regression PAVA yang fit pada split validasi.
- `src/memory/embedding.rs`, `src/memory/record.rs`, `src/memory/retrieval.rs`, `src/memory/store.rs`, dan `src/memory/secure.rs`: embedding hashing token/character n-gram, retrieval cosine, TTL otomatis, serta `SecureMemoryStore` AES-256-GCM + Ed25519 dengan transaksi lock-per-mutasi.
- `src/learning/long_term.rs`: `record_text_feedback` dan `recall_text_similar` yang memakai embedding aktual.
- `src/benchmark/long_term.rs`: arus retrieval dituning dari eksperimen; 40 nA diterapkan. Konfigurasi standard diperbesar menjadi 12 seed × 4 varian × 16 repetisi.
- `src/benchmark/long_term.rs` dan `src/learning/long_term.rs`: temporal ensemble lima observasi noisy, adaptive similarity/signal gate, serta fallback tanpa injeksi memori pada noise ekstrem. Fallback bukan data palsu: jaringan tetap diuji pada input noisy aktual dan kontrol tanpa memori.
- `src/learning/long_term.rs`, `src/learning/mod.rs`, `src/benchmark/mod.rs`, `tests/learning_memory.rs`: diformat oleh rustfmt; tidak ada tes yang dihapus atau dilemahkan.
- `examples/milestone10_long_term.rs` dan `examples/iris_real_dataset.rs`: runner release M10 diperbesar dan runner generalisasi UCI Iris nyata.
- `tests/milestone10_secure_adaptive.rs`: regression test embedding, TTL, enkripsi, signature, multi-writer, Platt, dan isotonic.
- `results/milestone10_long_term_release_2026-10-19.csv`: output runner yang benar-benar dieksekusi.
- `README.md`, `docs/API.md`, `docs/MEMORI.md`, `docs/PENGUJIAN.md`: diperbarui agar sesuai source code dan batasan aktual.

## Status sepuluh fitur

| # | Fitur | Status dan bukti |
|---:|---|---|
| 1 | Memori persisten | **LULUS** — `MemoryStore`/`LearningMemory` menulis snapshot JSON, memulihkan record setelah restart, mempertahankan ID, memvalidasi schema/checksum. |
| 2 | Retrieval relevan | **LULUS** — cosine similarity numerik dan embedding hashing token/character n-gram dengan urutan deterministik; tersedia `search_embedding` dan `recall_text_similar`. |
| 3 | Integrasi neuron | **LULUS** — hasil recall diinjeksikan sebagai arus ke neuron aksi dan memengaruhi spike LIF aktual; replay memanggil reward-modulated STDP aktual. |
| 4 | Replay adaptif | **LULUS** — replay dibatasi similarity/task dan mengutamakan experience dengan feedback negatif; laporan mencatat record dan update bobot. |
| 5 | Belajar dari kesalahan | **LULUS** — hasil prediksi/feedback disimpan, error diberi importance lebih tinggi, lalu replay menjalankan evaluator reward dan update jaringan. |
| 6 | Konsolidasi pengetahuan | **LULUS** — `consolidate` menaikkan importance yang dipakai sebagai proteksi dan prioritas; benchmark memakai replay pada task lama. |
| 7 | Pelupaan terkontrol | **LULUS** — `decay_irrelevant` hanya decay/delete record di bawah similarity dan importance policy; record protected tidak dihapus. |
| 8 | Deteksi konflik | **LULUS** — `find_conflicts` mendeteksi fitur mirip dengan target berbeda dan hasilnya dikembalikan saat feedback baru dicatat. |
| 9 | Akurasi dan kalibrasi | **LULUS** — accuracy, precision, recall, FPR, FNR, abstain, `CalibrationEvaluator`, Platt scaling, dan isotonic regression tersedia; regression test fitting lulus. |
| 10 | Evaluasi jangka panjang | **LULUS untuk benchmark sintetis dan holdout nyata** — runner mencakup restart/checkpoint, retensi, task konflik, replay, noise holdout, determinisme seed, decay, split validasi/test, serta UCI Iris train/test terpisah. |

## Metrik aktual M10

Runner release memakai 12 seed, 4 varian tugas, 16 repetisi per pola, dan tiga noise test (`30`, `50`, `70` nA), menghasilkan 144 baris benchmark plus 1 baris kalibrasi. Semua 48 run memulihkan checkpoint identik. Ini lebih besar dari runner sebelumnya, tetapi tetap benchmark sintetis.

Forgetting yang dilaporkan adalah selisih accuracy Task A setelah Task A dibandingkan accuracy sesudah Task B; angka positif berarti penurunan. Ringkasan aktual dari 48 run:

| Noise | Rata-rata forgetting tanpa memori | Rata-rata forgetting dengan memori |
|---:|---:|---:|
| 30 nA | 0.567564 accuracy kontrol | 0.647569 accuracy memory |
| 50 nA | 0.562355 accuracy kontrol | 0.610677 accuracy memory |
| 70 nA | 0.525608 accuracy kontrol | 0.575087 accuracy memory |

Temporal ensemble dan adaptive fallback meningkatkan accuracy pada seluruh tiga level benchmark: 0.567564→0.647569 (30 nA), 0.562355→0.610677 (50 nA), dan 0.525608→0.575087 (70 nA). Pada 70 nA, sistem tidak memaksakan retrieval yang tidak cukup tepercaya; ia memakai kontrol jaringan pada observasi noisy aktual. Output mentah tersedia di [CSV hasil M10](results/milestone10_long_term_release_2026-10-19.csv).

Generalisasi pada data nyata memakai Iris UCI yang berisi 150 instance, 4 fitur real, dan 3 kelas [1]. Split deterministik menghasilkan 120 data train dan 30 data test yang tidak dipakai saat penyimpanan memori. Retrieval memory memperoleh accuracy `0.966667` pada test bersih dan `0.733333` pada test dengan noise ±0.35 setelah normalisasi train-only. Hasil mentah tersedia di [CSV Iris](results/iris_uci_generalization_2026-10-19.csv), sedangkan dataset tersimpan di [data/iris_uci.csv](data/iris_uci.csv).

## Pengujian dan hasil

Toolchain yang tersedia di sandbox: Rust `1.97.1`. Hasil aktual:

| Perintah | Hasil |
|---|---|
| `cargo fmt --all -- --check` | **LULUS** |
| `cargo check --all-targets --all-features` | **LULUS** |
| `cargo clippy --all-targets --all-features -- -D warnings` | **LULUS** |
| `cargo test --all-targets --all-features` | **LULUS**, seluruh test debug selesai, 0 gagal |
| `cargo test --release --all-targets --all-features` | **LULUS**, seluruh test release selesai, 0 gagal |
| `cargo run --release --example milestone10_long_term --locked` | **LULUS**, output cocok dengan CSV benchmark tersimpan |
| `cargo run --release --example iris_real_dataset --locked` | **LULUS**, 120 train/30 test nyata; clean `0.966667`, noisy `0.733333` |

Suite mencakup regression tes repository sebelumnya, child-process restart untuk memory/checkpoint, LIF/STDP, konflik/replay/decay, determinisme continual-learning, noise/generalization, SDK end-to-end, dan unit test kalibrasi.

## Keterbatasan

`MemoryStore` lama tetap single-writer JSON untuk backward compatibility; aplikasi yang memerlukan perlindungan memakai `SecureMemoryStore`, yang membutuhkan kunci 32-byte dari secret manager dan belum menyediakan rotasi/backup otomatis. Embedding adalah hashing lexical deterministik, bukan embedding neural atau pemahaman bahasa. Benchmark sudah diperbesar tetapi masih sintetis dan hasilnya sensitif terhadap seed/noise/policy; angka di atas bukan akurasi produksi atau jaminan performa hardware pengguna.

## Keputusan akhir

**LULUS dengan batasan terukur.** Adaptive ensemble/fallback meningkatkan accuracy pada noise 30/50/70 nA di benchmark M10 dan runner holdout UCI Iris berhasil pada dataset nyata. Seluruh gerbang debug/release, Clippy `-D warnings`, runner M10, dan runner Iris telah lulus. Dataset Iris kecil dan mudah sehingga belum mewakili seluruh domain produksi.

## Referensi

[1]: https://archive.ics.uci.edu/dataset/53/iris "Iris — UCI Machine Learning Repository"
[2]: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html "load_iris — scikit-learn documentation"
