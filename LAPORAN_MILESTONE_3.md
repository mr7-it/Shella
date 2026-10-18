# Laporan Milestone 3 — heterogenitas neuron dan reward-modulated STDP

> **Catatan status:** laporan ini adalah snapshot historis saat milestone terkait dikerjakan. Fitur/batas yang tertulis sebagai “belum tersedia” merujuk keadaan pada tahap tersebut; untuk kondisi sekarang gunakan README, dokumentasi `docs/`, dan LAPORAN_MILESTONE_5.md.

## Status

**Milestone 3: LULUS.** Heterogenitas parameter LIF per neuron dan reward-modulated STDP telah ditambahkan pada jaringan multi-neuron Milestone 2. Seluruh 52 tes lulus pada Rust stable 1.99.1 dan MSRV Rust 1.85.0; gerbang format, build, Clippy, release, dan demo juga lulus pada keduanya.

## Baseline

Sebelum perubahan, tes milestone sebelumnya lulus 37/37. Jaringan memakai satu konfigurasi LIF homogen; STDP lokal langsung menerapkan update bobot.

## Implementasi yang selesai

`LifNetwork::from_configs(&[LifConfig])` menerima konfigurasi per indeks neuron. Konstanta waktu membran, resistansi, potensial istirahat/reset, ambang, dan periode refraktori boleh berbeda. Semua `dt_ms` wajib sama agar langkah simulasi, delay sinaps, dan trace pembelajaran merujuk pada basis waktu yang sama. Constructor homogen `LifNetwork::new(config, count)` tetap tersedia.

`RewardModulatedStdpConfig` menggabungkan konfigurasi STDP lokal, konstanta waktu eligibility, dan learning rate tervalidasi. `enable_reward_modulated_stdp` mengaktifkan trace pre/post dan eligibility. Spike membentuk delta STDP lokal, tetapi dalam mode reward delta masuk ke eligibility yang meluruh; bobot tidak berubah sebelum reward tiba. `apply_reward(r)` menerapkan `learning_rate × r × eligibility`, mendukung reward finite positif/negatif, menjepit bobot dalam batas sinaps, dan tidak memajukan indeks waktu. Eligibility tidak dihapus oleh reward; ia terus meluruh pada langkah jaringan berikutnya. Seluruh update bobot dihitung sebelum commit agar overflow tidak menghasilkan update parsial.

Mode STDP lokal sebelumnya tetap tersedia. Mengaktifkan mode pembelajaran mereset trace; menonaktifkan STDP mempertahankan bobot yang sudah dipelajari dan menghapus trace.

## Pengujian dan gerbang

Hasil final setiap run: **52 tes lulus, 0 gagal** (18 LIF, 19 jaringan/STDP, 15 Milestone 3) dalam mode debug dan release. Kumpulan tes dijalankan pada stable 1.99.1 serta Rust 1.85.0.

Perintah final yang dijalankan:

```text
cargo +stable fmt --all -- --check
cargo +stable check --all-targets --all-features --locked
cargo +stable clippy --all-targets --all-features --locked -- -D warnings
cargo +stable test --all-targets --locked
cargo +stable test --release --locked
cargo +stable run --release --locked

cargo +1.85.0 fmt --all -- --check
cargo +1.85.0 check --all-targets --all-features --locked
cargo +1.85.0 clippy --all-targets --all-features --locked -- -D warnings
cargo +1.85.0 test --all-targets --locked
cargo +1.85.0 test --release --locked
cargo +1.85.0 run --release --locked
```

Semua gerbang di atas **LULUS**. Demo release berjalan pada kedua toolchain: neuron sumber spike pada langkah 1, neuron tujuan heterogen spike pada langkah 2, bobot tetap 20 nA sebelum reward, lalu naik menjadi 20.095123 nA setelah `apply_reward(1.0)`.

Tes Milestone 3 memeriksa respons numerik dari `tau_m` berbeda, ambang/refraktori berbeda, penolakan `dt` yang berbeda, input konfigurasi invalid, integrasi heterogen dengan sinaps/STDP, pembentukan eligibility tanpa update langsung, decay untuk reward tertunda, reward positif/negatif/nol, respons terhadap eligibility positif dan negatif, batas bobot, reward non-finite, penolakan reward dalam mode STDP lokal, serta rollback overflow.

`cargo bench` **BELUM DIJALANKAN** karena benchmark belum diimplementasikan. Tidak ada tes lintas platform atau jaminan hasil bit-identik lintas compiler/OS.

## Berkas yang dibuat atau diubah

API jaringan di `src/network/mod.rs` ditambah constructor heterogen dan mode reward tiga faktor. `src/learning/mod.rs` menambah `RewardModulatedStdpConfig` dan eligibility trace. API publik diekspor melalui `src/lib.rs`; `src/main.rs` kini menjalankan demo heterogen dengan reward tertunda. `tests/milestone3.rs` menambahkan 15 tes. README, dokumentasi model, jaringan, STDP, arsitektur, determinisme, pengujian, dan laporan milestone diperbarui. Tidak ada dependency eksternal yang ditambahkan.

## Batasan dan risiko

Semua neuron dalam satu jaringan harus menggunakan `dt` yang sama. Reward masih disediakan langsung oleh pemanggil; tidak ada evaluator reward, lingkungan/tugas belajar, atau benchmark yang membuktikan peningkatan perilaku. Aturan ini adalah STDP tiga faktor pair-based, bukan triplet/reward shaping yang lebih luas. Eligibility, bobot, dan event belum memiliki checkpoint/pemulihan state. Heterogenitas tidak mendukung model neuron lain selain LIF. Tidak ada commit Git yang dibuat.

Fitur memori pemulihan, mekanisme keputusan yang dapat dievaluasi, dan API integrasi aplikasi masih belum tersedia dan tidak diklaim selesai.
