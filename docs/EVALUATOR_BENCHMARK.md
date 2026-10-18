# Evaluator reward otomatis dan benchmark

## Evaluator

`RewardEvaluatorConfig::new(reward_benar, penalti_salah)` menerima dua nilai finite positif. `RewardEvaluator::evaluate(target, observed)` memilih aksi dari kanal boolean yang teramati; jika beberapa kanal aktif, tie-break adalah indeks aktif terendah. Target `Some(i)` berarti kanal aksi `i` diharapkan, sedangkan `None` berarti tidak boleh ada kanal spike.

Respons cocok menghasilkan `+correct_reward`; respons tidak cocok menghasilkan `-incorrect_penalty`. `RewardMetrics` mencatat jumlah sampel/benar/no-response, akurasi, total reward, serta mean reward. `evaluate_and_apply` menerapkan reward pada eligibility `LifNetwork` sebelum commit metrik; jika network gagal, evaluator tidak mencatat sampel parsial.

SDK menambahkan adapter `Session::evaluate_actions`/`evaluate_actions_and_apply`: daftar neuron output dipetakan ke kanal aksi sesuai urutan slice, lalu dinilai berdasarkan spike dalam output langkah tersebut. Test integrasi `tests/sdk_integration.rs` memeriksa reward dari spike dan rollback metrik saat apply ditolak.

## Conditioning reward-learning

`ConditioningBenchmarkConfig::standard()` menggunakan 32 epoch dan delapan pengulangan uji per pola. Tugas mempunyai dua neuron cue, satu neuron aksi, dan dua sinaps dengan bobot awal nol.

- Cue “mendekat” dipasangkan dengan teacher spike dan target aksi kanal 0.
- Cue “menahan” mendapat teacher tanpa spike output; evaluator menargetkan diam.
- Kontrol menerima urutan cue/teacher yang sama tetapi tidak menerapkan reward ke bobot.
- Kedua jaringan menjalani washout 50 langkah, kemudian pola yang sama diuji tanpa teacher.

Putaran Release terakhir mengukur kontrol **8/16 (50%)** dan jaringan learned **16/16 (100%)**, selisih **+50 poin persentase**; bobot cue mendekat berakhir 116.155578 nA dan cue menahan 0 nA. Harness melakukan tujuh batch × 100 run; median waktu per eksperimen **230 128 ns** (≈230 µs) pada mesin sesi ini. Ini adalah recall dua pola yang dilatih, bukan generalisasi atau benchmark kecerdasan.

## Benchmark skala, persistence, retrieval

`benches/system.rs` mengukur median lima batch, tanpa RNG. Tiap batch menjalankan 500 langkah jaringan; checkpoint memakai 20 operasi/batch; memory retrieval menyiapkan 256 record lebih dahulu lalu mengukur 500 query/batch. Benchmark system tidak menjalankan fase warm-up terpisah. Hasil Release terakhir:

| Operasi | Skala | Median latency/op | Throughput |
|---|---:|---:|---:|
| Langkah network + delayed event | 16 neuron / 16 sinaps | 2,386 ns | 419,142 langkah/s |
| Langkah network + delayed event | 64 neuron / 64 sinaps | 8,393 ns | 119,154 langkah/s |
| Langkah network + delayed event | 256 neuron / 256 sinaps | 22,958 ns | 43,558 langkah/s |
| Checkpoint save | 64 neuron / 64 sinaps | 762,385 ns | 1,312 save/s |
| Checkpoint restore | 64 neuron / 64 sinaps | 265,567 ns | 3,766 restore/s |
| Memory prefix search | 256 record | 1,901 ns | 526,057 query/s |
| Memory tag search | 256 record | 2,620 ns | 381,646 query/s |

Checkpoint contoh berukuran 25,985 byte. Seluruh waktu sensitif terhadap beban host/filesystem; angka mesin sesi ini bukan jaminan kapasitas produksi. Benchmark network ring membangkitkan spike dan satu delayed event per neuron/step, bukan variasi topologi arbitrer. Benchmark tidak mengukur peak memory.

## Menjalankan ulang

```sh
cargo test --test milestone4 --locked
cargo bench --bench learning --locked
cargo bench --bench system --locked
```

Hasil merupakan output yang diukur dari kode aktual. Ulangi perintah pada target hardware/beban kerja yang relevan untuk evaluasi lokal.


## Continual learning Milestone 6

Runner `examples/milestone6_continual.rs` menjalankan empat asosiasi dalam dua fase. Setiap pola input memuat cue umum yang dipakai semua kelas dan satu cue pembeda. Jaringan memiliki 9 neuron dan 20 sinaps, learning rate 2.5, bobot maksimum cue umum 50 nA dan cue pembeda 150 nA. Tiap contoh diberi teacher spike, dinilai dengan evaluator, lalu diikuti 100 langkah tanpa input agar eligibility dari contoh terdahulu meluruh.

Training memakai cue 100 nA, 32 epoch per fase, dan 128 contoh per jaringan. Validasi memakai amplitudo 80/90/110/120 nA; test final memakai 97/103 nA. Setiap contoh evaluasi memakai clone state awal yang sama dan tidak menerima teacher/reward. Lima seed tetap `[1, 7, 19, 42, 99]` hanya mengacak urutan asosiasi secara deterministik; bobot awal, kelas, serta amplitudo tidak diacak.

Hasil final pada kelima seed: baseline test **0/8 (0%)**, test akhir **8/8 (100%)** untuk empat asosiasi × dua amplitudo, validasi **16/16 (100%)** untuk empat asosiasi × empat amplitudo, gain **+100 poin persentase**, reward teacher-forced **128** untuk 128 contoh, task A **100%** setelah fase pertama dan tetap **100%** setelah fase kedua, sehingga forgetting terukur **0 poin persentase**. CSV per seed: `results/milestone6_continual_2026-10-19.csv`.

Tugas ini sintetis, deterministik, dan kecil. Split hanya menahan amplitudo cue, bukan kelas/struktur semantik yang berbeda. Seed hanya mengubah urutan training; bukan variasi inisialisasi, data, atau noise. Forgetting nol berlaku untuk empat pola dengan cue pembeda dan bobot cue umum yang dibatasi; hasil ini tidak membuktikan pencegahan catastrophic forgetting untuk pemetaan konflik atau distribusi lain. `training_reward` teacher-forced tidak menggantikan recall tanpa teacher.

Jalankan ulang dengan `cargo run --example milestone6_continual --locked`. Hasil tersebut merupakan metrik perilaku jaringan, bukan klaim pemahaman bahasa atau benchmark kecerdasan.
