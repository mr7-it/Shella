# Pengujian perilaku dan benchmark

## Gerbang mutu

Perintah utama yang digunakan dari root repository:

```sh
cargo fmt --all -- --check
cargo check --all-targets --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --locked
cargo test --release --locked
cargo run --release --locked
cargo run --release --example milestone10_long_term --locked
cargo bench --bench learning --locked
cargo bench --bench system --locked
```

MSRV di manifest adalah Rust 1.85. Jalankan gerbang di stable dan MSRV jika perubahan menyentuh code/dependency. `cargo test --all-targets` membangun/menjalankan target benchmark dengan konfigurasi test; perintah `cargo bench` terpisah dipakai untuk angka performa Release. Jangan menyamakan hasil tes kebenaran dengan timing benchmark.

## Peta tes

- `tests/lif_runtime.rs`: persamaan LIF, input invalid, spike/reset/refractory, atomisitas, overflow, dan determinisme.
- `tests/network_stdp.rs`, `tests/milestone3.rs`: topologi, update dua fase, delay, propagasi/inhibisi, konfigurasi heterogen, STDP/eligibility/reward dan rollback.
- `tests/milestone4.rs`: evaluator, reward berdasarkan output aktual, metrik atomik, serta recall dibanding kontrol tanpa update.
- `tests/generalization.rs`: asosiasi cue-to-action baru, held-out amplitudo+distractor, kontrol tanpa learning, dan cue yang tak dilatih.
- `tests/continual_learning.rs`: empat asosiasi dengan fitur cue umum yang berbagi lintas kelas, dua fase belajar, lima seed, split amplitudo terpisah, determinisme, dan forgetting task lama.
- `tests/checkpoint_recovery.rs`: round-trip jaringan/runtime, evaluator, pending event, trace STDP, trajectory resume pada proses baru, file corrupt/truncated, versi tidak didukung, dan I/O.
- `tests/persistent_memory.rs`: CRUD/upsert, ID stabil, prefix/tag retrieval, restart proses nyata, input invalid dan data rusak.
- `tests/decision_engine.rs`: skor/kontribusi terhitung, tie-break, abstain, unknown features, invalid input, overflow skor, dan overflow margin.
- `tests/language_pipeline.rs`: normalisasi, aturan tokenizer Unicode, vocabulary deterministik/unknown, pemisahan train/eval, arus encoder, serta penolakan input tanpa token.
- `tests/sdk_integration.rs`: teks -> input neuron -> network terlatih -> reward evaluator -> held-out action -> decision -> memory -> checkpoint.
- Unit test evaluator kalibrasi: confidence di luar `0..1` ditolak dan Brier/MAE dihitung dari confidence probabilistik yang diberikan.
- `tests/milestone10_secure_adaptive.rs`: embedding semantic retrieval, TTL restart, enkripsi AES-GCM, signature Ed25519, writer bergantian, Platt scaling, dan isotonic regression.
- `examples/milestone10_long_term.rs`: runner release 12 seed × 4 varian × 16 repetisi untuk restart, retensi, konflik, replay error, pelupaan, noise holdout, confusion metrics, dan kalibrasi; runner memakai jaringan dan memori aktual.
- `examples/iris_real_dataset.rs`: runner data nyata UCI Iris dengan 120 train/30 test, normalisasi train-only, retrieval memory, dan evaluasi noise holdout.

Pada verifikasi M10 di sandbox dengan Rust 1.97.1, `cargo fmt --all -- --check`, `cargo check --all-targets --all-features`, Clippy `-D warnings`, test debug, dan test release lulus setelah adaptive ensemble/gate. Runner M10 dan runner Iris juga lulus serta outputnya cocok dengan CSV tersimpan. Klaim ini tidak menggantikan verifikasi pada MSRV atau perangkat deployment.

## Benchmark

`benches/learning.rs` mengukur tugas conditioning 32 epoch / dua pola dan evaluator reward, membandingkan recall learned dengan kontrol tanpa weight update. Harness melakukan warm-up dan tujuh batch × 100 run; hasil spesifik mesin tercatat di `docs/EVALUATOR_BENCHMARK.md`.

`examples/milestone6_continual.rs` menjalankan 5 seed × 32 epoch per fase untuk 4 asosiasi yang berbagi cue; ini benchmark akurasi/retensi, bukan timing. Runner menulis hasil per seed ke `results/milestone6_continual_2026-10-19.csv`.

`benches/system.rs` mengukur median lima batch untuk network 16/64/256 neuron dengan ring synapses dan delayed event, checkpoint save/restore 64 neuron, serta prefix/tag retrieval store berisi 256 record. Tidak ada warm-up terpisah pada benchmark system. Hasil bukan peak memory, beban multi-proses, atau jaminan performa deployment.

`examples/capacity_probe.rs` dan `scripts/run_capacity_probe.sh` menjalankan workload terpisah untuk RSS/peak RSS Linux, build, median step, checkpoint save dan byte size sampai 16,384 neuron/65,536 sinaps. Hasil aktual Sandbox disimpan pada `results/capacity_sandbox_2026-10-19.csv` dan raw NDJSON; hasil tersebut bukan pengukuran perangkat pengguna.

## Pengujian Milestone 11

`examples/milestone11_self_learning.rs` menjalankan lima seed dan menghasilkan 35 baris: tiga jalur test bersih, memory pada noise ±0.25, continual old-task setelah replay, self-loop validation, dan persistence setelah restart. Test Iris tidak dimasukkan ke memory sebelum evaluasi test. Scaler dan pemilihan memory current memakai train/validation saja; test hanya dievaluasi di akhir.

Regression test `tests/milestone11_self_learning.rs` memverifikasi split disjoint, label stratified, train-only scaler, k-fold tanpa overlap, dan self-evaluation. Hasil M11 lengkap ada pada `LAPORAN_MILESTONE_11.md`.
