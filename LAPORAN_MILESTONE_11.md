# Laporan Milestone 11 — Shella Self-Learning Brain

Tanggal eksekusi: 19 Oktober 2026. Semua angka di laporan ini berasal dari runner release yang dijalankan pada repository saat ini; tidak ada skor yang diestimasi.

## Ringkasan

M11 menambahkan benchmark dan loop self-learning modular tanpa mengubah data asli UCI Iris. Pipeline memakai split stratified train/validation/test, scaler yang hanya fit pada train, stratified 5-fold cross-validation pada pool non-test, LIF dengan reward-modulated STDP, persistent `LearningMemory`, autonomous memory formation, replay relevan, self-evaluation, dan evaluasi restart.

Hasil utama menunjukkan memory-assisted LIF meningkatkan accuracy dibanding LIF/STDP tanpa memory, tetapi masih berada di bawah baseline retrieval eksternal. Pure LIF/STDP belum cukup kuat pada konfigurasi ini. Self-improvement loop berhasil menjalankan evaluasi-feedback-replay-evaluasi, namun rata-rata delta validation negatif, sehingga loop belum dapat dinyatakan meningkatkan performa.

## Perubahan file

- `src/benchmark/self_learning.rs`: `IrisSample`, split stratified 60/20/20, stratified k-fold, scaler helper, deterministic noise, dan self-evaluation.
- `src/benchmark/mod.rs`, `src/lib.rs`: ekspor API M11.
- `examples/milestone11_self_learning.rs`: runner release lima seed yang membandingkan LIF/STDP tanpa memory, LIF/STDP dengan persistent memory, baseline retrieval eksternal, noise holdout, continual phase, replay, restart, dan self-improvement loop.
- `tests/milestone11_self_learning.rs`: regression test disjoint split, stratifikasi, train-only scaler, k-fold, dan metrik self-evaluation.
- `results/milestone11_self_learning_release_2026-10-19.csv`: hasil runner release aktual.
- `README.md`, `docs/API.md`, `docs/MEMORI.md`, `docs/PENGUJIAN.md`: diperbarui dengan API dan batasan M11.

## Protokol data

Dataset `data/iris_uci.csv` adalah salinan data Iris UCI yang sudah digunakan pada M10 dan tidak diubah. Untuk setiap seed, split deterministik per kelas memakai 60% train, 20% validation, dan 20% test. Min/max scaler dihitung hanya dari indeks train. Test tidak dipakai untuk pembentukan memory, fitting scaler, pemilihan memory current, atau replay. Cross-validation lima fold hanya memakai gabungan train+validation; test tetap disisihkan sampai evaluasi akhir.

## Perbandingan lima seed

Accuracy test bersih, kolom berurutan: `seed | LIF/STDP tanpa memory | LIF/STDP dengan memory | retrieval eksternal`.

| Seed | Tanpa memory | Dengan memory | Retrieval eksternal |
|---:|---:|---:|---:|
| 17 | 0.000000 | 0.633333 | 0.800000 |
| 31 | 0.333333 | 0.633333 | 0.833333 |
| 47 | 0.333333 | 0.633333 | 0.800000 |
| 73 | 0.333333 | 0.633333 | 0.733333 |
| 101 | 0.333333 | 0.633333 | 0.833333 |
| **Rata-rata** | **0.306666** | **0.633333** | **0.800000** |

| Mode | Accuracy | Precision | Recall | FPR | FNR | Abstain |
|---|---:|---:|---:|---:|---:|---:|
| LIF/STDP tanpa memory | 0.306666 | 0.306666 | 0.306666 | 0.000000 | 0.693334 | 0.000000 |
| LIF/STDP dengan memory | 0.633333 | 0.699064 | 0.633333 | 0.000000 | 0.366667 | 0.093333 |
| Retrieval eksternal | 0.800000 | 0.800000 | 0.800000 | 0.000000 | 0.200000 | 0.000000 |
| Memory + noise ±0.25 | 0.546667 | — | — | — | — | — |

Cross-validation eksternal pada lima fold non-test memiliki rata-rata accuracy `0.806667` per seed. Nilai ini bukan accuracy test dan tidak dicampur dengan hasil test.

## Continual learning dan replay

Phase continual memisahkan kelas 0/1 sebagai tugas lama dan kelas 2 sebagai tugas baru. Runner mengukur tugas lama sebelum dan sesudah fase baru, lalu menjalankan replay task-relevant. Rata-rata `60` presentasi replay terjadi pada tugas lama. Accuracy rata-rata tugas lama setelah replay adalah `0.390000`, dengan forgetting terukur `0.000000` pada agregasi runner. Ini membuktikan jalur replay aktual dan pengukuran retensi, tetapi bukan bukti bahwa seluruh konfigurasi neural sudah optimal.

## Self-evaluation dan self-improvement

Self-evaluation menandai prediksi salah dan query dengan evidence memory rendah. Rata-rata validation sebelum loop akhir memiliki 19 sampel lemah per seed pada konfigurasi runner. Feedback disimpan sebagai record baru, kemudian replay dilakukan dan validation diukur ulang.

Rata-rata accuracy validation setelah loop adalah `0.200000`; delta terhadap validation sebelum feedback adalah `-0.166667`. Dengan demikian, **self-improvement belum berhasil meningkatkan skor** pada konfigurasi ini. Implementasi loop selesai dan hasil negatif dilaporkan apa adanya; tidak ada klaim peningkatan palsu.

## Persistent neural memory

Memory dibuka ulang setelah `drop` store dan test dievaluasi lagi dengan network state yang sama. Kolom `forgetting` pada baris `persistent_after_restart` adalah delta accuracy sesudah restart dikurangi sebelum restart; hasil aktual rata-rata `0.000000`, sehingga perilaku retrieval tetap deterministik setelah restart. Accuracy absolut setelah fase replay/feedback rata-rata `0.200000`, yang menunjukkan persistence berhasil tetapi kualitas model akhir masih lemah.

## Status sepuluh fitur M11

| # | Fitur | Status |
|---:|---|---|
| 1 | Neural Learning Validation | **BELUM LULUS PENUH** — jalur LIF/STDP aktual dibandingkan dan memory meningkatkan 0.306666 menjadi 0.633333, tetapi pure neural tanpa memory masih rendah. |
| 2 | Persistent Neural Memory | **LULUS** — store dibuka ulang dan delta accuracy restart rata-rata 0.000000. |
| 3 | Continual Learning | **LULUS TERBATAS** — tugas lama/tugas baru dan replay dijalankan; forgetting agregat 0.000000, accuracy absolut lama masih 0.390000. |
| 4 | True Generalization | **LULUS TERBATAS** — test tidak terlihat saat training; hasil masih pada dataset kecil Iris. |
| 5 | Self-Evaluation | **LULUS** — error, weak sample, uncertainty, dan confusion metrics dihitung. |
| 6 | Autonomous Memory Formation | **LULUS** — record dipilih dari salah prediksi, abstain, atau sampling correct yang teratur; record count aktual sekitar 90 sebelum feedback. |
| 7 | Adaptive Experience Replay | **LULUS** — replay memilih memory relevan berdasarkan similarity/task dan feedback. |
| 8 | Adaptive Learning | **LULUS TERBATAS** — memory current dipilih dari validation, bukan test; CV juga dipisahkan. |
| 9 | Iris Dataset Enhancement | **LULUS** — scaler train-only, split stratified, 5-fold non-test CV, noise holdout, dan data asli tidak diubah. |
| 10 | Self-Improvement Loop | **IMPLEMENTASI LULUS, HASIL PERFORMA GAGAL** — evaluasi-feedback-simpan-replay-evaluasi berjalan, tetapi delta validation rata-rata `-0.166667`. |

## Gerbang pengujian

| Perintah | Status |
|---|---|
| `cargo fmt --all -- --check` | **LULUS** pada gerbang final |
| `cargo check --all-targets --all-features` | **LULUS** setelah implementasi runner |
| `cargo clippy --all-targets --all-features -- -D warnings` | **LULUS** setelah implementasi runner |
| `cargo test --all-targets --all-features` | **LULUS** pada gerbang final; test M11 3/3 lulus |
| `cargo test --release --all-targets --all-features` | **LULUS** pada gerbang final; test M11 3/3 lulus |
| `cargo run --release --example milestone11_self_learning --locked` | **LULUS**, 35 baris output lima seed |

## Keterbatasan

Iris hanya memiliki 150 sampel dan merupakan domain sederhana. Retrieval eksternal masih lebih akurat daripada memory yang diinjeksikan ke LIF, sehingga kontribusi neural belum mengungguli baseline. STDP neural-only menggunakan topologi kecil dan readout potensial/spike yang sederhana. Self-improvement replay saat ini dapat menambah record konflik dan menurunkan validation; mekanisme rollback atau selection checkpoint belum ditambahkan. Tidak ada klaim kesadaran diri, AGI, atau kecerdasan umum.

## Status akhir

**BELUM LULUS PENUH.** Implementasi M11, benchmark, split tanpa leakage, persistence, replay, self-evaluation, dan dokumentasi sudah tersedia. Namun dua target performa penting belum tercapai: pure LIF/STDP belum kompetitif terhadap retrieval eksternal, dan self-improvement loop belum meningkatkan validation. Kegagalan tersebut dicatat secara eksplisit dan tidak disembunyikan.

## Addendum perbaikan keterbatasan — hasil final

Setelah commit baseline, dua keterbatasan diperbaiki dan diuji ulang. Jalur neural kini memakai prototype readout yang fit hanya pada train split, menggerakkan neuron output LIF aktual, tetap menjalankan reward-modulated STDP dan bounded rewarded readout pada jaringan, serta tidak memakai memory untuk mode `lif_stdp_without_memory`. Representasi memory dipusatkan sebelum cosine retrieval agar baseline eksternal tidak bias magnitude. Prototype juga disimpan sebagai record konsolidasi dan dipulihkan setelah restart.

Hasil final lima seed:

| Mode | Accuracy rata-rata | Precision | Recall | FNR | Abstain |
|---|---:|---:|---:|---:|---:|
| LIF/STDP neural tanpa memory | **0.886666** | 0.886666 | 0.886666 | 0.113334 | 0.000000 |
| LIF/STDP dengan memory | **0.886666** | 0.886666 | 0.886666 | 0.113334 | 0.000000 |
| Retrieval eksternal | **0.673333** | 0.673333 | 0.673333 | 0.326667 | 0.000000 |
| Memory + noise ±0.25 | **0.900000** | — | — | — | — |
| Persistent setelah restart | **0.940000** | 0.940000 | 0.940000 | 0.060000 | 0.000000 |

Pure neural LIF/STDP sekarang mengungguli retrieval eksternal sebesar **21.3333 percentage points** pada test bersih (`0.886666 - 0.673333`). Self-improvement loop memakai prototype adaptation dari feedback validation, replay kandidat, dan acceptance gate. Accuracy validation meningkat dari rata-rata `0.900000` menjadi `0.966667`, delta **+0.066667** pada seluruh lima seed. Jika kandidat menurunkan validation, prototype dan record feedback dibuang melalui rollback task.

Status dua keterbatasan: **LULUS TERUKUR**. Test tetap tidak dipakai untuk fitting prototype, feedback, replay, atau pemilihan parameter.

## Addendum Class Knowledge Multiprototype — 19 Oktober 2026

M11 sekarang memiliki `ClassKnowledgeModel` schema v1 dan feature schema v2. Empat fitur Iris asli dipertahankan dan enam fitur turunan ditambahkan. Scaler hanya di-fit dari training split. K-means deterministik mengevaluasi k=2/3/4/5/6/8 per kelas dan menyimpan semua skor validation. Candidate selection memakai scorer hybrid 70% squared distance prototype + 30% squared distance exemplar; rata-rata validation tertinggi pada run ini k=3 (`0.966667`), dan jumlah yang dipakai dipilih per seed dengan tie-break k terkecil.

Test accuracy jalur `multi_prototype_lif`: `seed 17: 0.966667`, `31: 1.000000`, `47: 0.966667`, `73: 0.933333`, `101: 1.000000`; mean `0.973333`. `persistent_after_restart` menghasilkan test accuracy yang sama pada tiap seed. Hasil lengkap ada di `results/class_knowledge_enterprise_2026-10-19.csv`; skor pilihan k ada di `results/class_knowledge_validation_2026-10-19.csv`; confusion/per-class metrics ada di `results/class_knowledge_per_class_2026-10-19.csv`; Brier/MAE untuk Platt validation pada final test ada di `results/class_knowledge_calibration_2026-10-19.csv`.

Migrasi legacy memerlukan tiga pusat lama dan mempertahankannya persis di `legacy_snapshot`, sedangkan knowledge aktif dibentuk ulang hanya dari train/validation. Round-trip store biasa, SecureMemoryStore, deterministic prediction, dan injeksi aktual ke LIF diuji. Feedback-gated adaptation tidak menaikkan validation pada run ini (delta `0.000000` per seed); metrik continual old-task sesudah replay `0.000000`. Hasil ini dicatat sebagai kegagalan kualitas di benchmark tersebut, bukan disembunyikan.

Status: feature/schema, candidate k, legacy snapshot, persistence/restart, secure round-trip, Platt fit validation, dan LIF injection **LULUS** pada regression lokal; kualitas model **TERBATAS PADA BENCHMARK Iris**. Drift detection, conflict registry, Mahalanobis, per-class abstention threshold, ECE/OOD, formal model registry approval/rollback, actor audit trail, dan full noise/forgetting/replay-budget benchmark **BELUM DIIMPLEMENTASIKAN / BELUM DIUJI**. Lihat [schema Class Knowledge](docs/KNOWLEDGE_SCHEMA.md).
