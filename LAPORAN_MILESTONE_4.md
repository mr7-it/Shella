# Laporan Milestone 4 — evaluator reward dan benchmark belajar

> **Catatan status:** laporan ini adalah snapshot historis saat milestone terkait dikerjakan. Fitur/batas yang tertulis sebagai “belum tersedia” merujuk keadaan pada tahap tersebut; untuk kondisi sekarang gunakan README, dokumentasi `docs/`, dan LAPORAN_MILESTONE_5.md.

## Ringkasan

Milestone 4 menambahkan evaluator reward otomatis, menghubungkannya langsung ke `LifNetwork`, serta menyediakan benchmark Release deterministik untuk mengukur recall sebelum dan sesudah reward-modulated STDP. Tidak ada dependency eksternal baru. Benchmark mengevaluasi dinamika jaringan aktual; akurasi tidak dipraisi atau di-hardcode.

## Implementasi

- `src/evaluator/mod.rs`: `RewardEvaluatorConfig`, `RewardEvaluator`, `RewardEvaluation`, dan `RewardMetrics`. Target aksi dibandingkan dengan kanal spike aktual; `None` berarti target diam. Bila beberapa kanal aktif, indeks aktif terendah menjadi prediksi. Respons benar memberi reward positif, respons salah penalti negatif.
- `RewardEvaluator::evaluate_and_apply` mengirim nilai reward ke `LifNetwork::apply_reward`. Metrik baru dicatat hanya setelah update jaringan berhasil; validasi dan overflow tidak menyebabkan pencatatan setengah jadi.
- `src/benchmark/mod.rs`: protokol conditioning dua cue. Isyarat mendekat dilatih dengan teacher spike, isyarat menahan dilatih untuk diam. Kontrol menjalani input/teacher yang sama tetapi tidak menerima update reward. Setelah washout, kedua jaringan dievaluasi tanpa teacher.
- `benches/learning.rs`: warm-up, tujuh batch × 100 run, cek hasil deterministik dan cek gain akurasi, lalu cetak CSV termasuk median waktu per run. `src/main.rs` juga menunjukkan evaluasi otomatis atas spike keluaran.
- Dokumentasi evaluator, model, arsitektur, determinisme, dan pengujian diperbarui.

## Hasil benchmark yang diamati

Konfigurasi baku memakai 32 epoch, 8 pengulangan per pola uji, dan 16 respons recall total:

| Ukuran | Kontrol tanpa update reward | Reward-trained |
|---|---:|---:|
| Benar | 8/16 | 16/16 |
| Akurasi | 50% | 100% |
| Gain | — | +50 poin persentase |
| Bobot akhir cue mendekat | 0 nA (awal) | 116.155578 nA |
| Bobot akhir cue menahan | 0 nA (awal) | 0 nA |

Pada run Release stable terakhir, median dari 7 batch adalah 217 620 ns (sekitar 218 mikrodetik) per eksperimen. Pengukuran waktu berubah antar-run dan sensitif terhadap beban mesin; angka tersebut hanya catatan lingkungan ini, bukan jaminan kinerja.

**Batas interpretasi:** dua pola yang sama dipakai untuk pelatihan dan recall. Hasil ini menunjukkan mekanisme conditioning bekerja pada tugas kecil yang didefinisikan eksplisit, bukan uji generalisasi, dataset dunia nyata, atau bukti manfaat pada tugas neuromorfik umum. Training cue “mendekat” menggunakan teacher signal; recall terukur tidak menggunakan teacher.

## Verifikasi final

Stable `rustc/cargo 1.99.1` dan MSRV `rustc/cargo 1.85.0`:

- `cargo fmt --all -- --check` — lulus.
- `cargo check --all-targets --all-features --locked` — lulus.
- `cargo clippy --all-targets --all-features --locked -- -D warnings` — lulus.
- `cargo test --all-targets --locked` — lulus; 63 tes unit/integrasi, plus target benchmark custom dijalankan.
- `cargo test --release --locked` — lulus.
- `cargo run --release --locked` — lulus; demo menampilkan target, prediksi, reward `+1`, dan bobot setelah evaluasi.
- `cargo bench --bench learning --locked` — lulus; reward-trained mengungguli kontrol dan hasil antar-run identik.

Tidak ada kegagalan gerbang yang tersisa. Hasil pemeriksaan lebih rinci berada di `docs/EVALUATOR_BENCHMARK.md` dan `docs/PENGUJIAN.md`.
