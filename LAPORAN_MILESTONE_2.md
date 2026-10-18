# Laporan Milestone 2 — jaringan multi-neuron dan STDP

> **Catatan status:** laporan ini adalah snapshot historis saat milestone terkait dikerjakan. Fitur/batas yang tertulis sebagai “belum tersedia” merujuk keadaan pada tahap tersebut; untuk kondisi sekarang gunakan README, dokumentasi `docs/`, dan LAPORAN_MILESTONE_5.md.

## Status

**Milestone 2: LULUS.** Jaringan multi-neuron, sinaps tertunda, pembaruan sinkron dua fase, dan aturan STDP pair-based telah diimplementasikan serta diuji. STDP diintegrasikan sesudah tes jaringan dasar lulus, sesuai urutan dependensi. Ini belum menyelesaikan seluruh visi platform; reward learning, checkpoint, memori asosiatif, sistem keputusan, dan pemrosesan bahasa tidak termasuk.

## Kondisi awal

Proyek milestone 1 ada di `/home/ubuntu/shella`; tes awal lulus 18/18. Tidak ada dependency pihak ketiga. Kode LIF sebelumnya diekstensi dengan API prepare/commit agar jaringan dapat menghitung transisi semua neuron sebelum menerapkan satu langkah.

## Implementasi

`LifNetwork` membuat jaringan homogen dengan satu `LifConfig` untuk semua neuron. Setiap langkah menerima satu arus eksternal finite per neuron. Semua neuron menghitung transisi berdasarkan keadaan awal langkah; spike baru tidak dapat menyebar pada langkah yang sama.

`Synapse` adalah koneksi berarah unik per pasangan sumber-tujuan dengan bobot arus signed, batas inklusif, dan delay sekurang-kurangnya satu langkah. Bobot positif bersifat eksitatorik dan negatif inhibitorik. Spike pada langkah `k` menjadwalkan event untuk `k + delay`; event menjadi arus satu interval dan dijumlahkan dengan input eksternal.

Jika satu perhitungan neuron, penjumlahan arus, penjadwalan event, atau update STDP gagal, seluruh langkah jaringan dibatalkan sebelum commit keadaan simulasi. Urutan penjumlahan mengikuti urutan penambahan sinaps. Event yang dibuat pada langkah dengan STDP menggunakan bobot sebelum pembaruan pembelajaran pada langkah itu.

STDP memakai trace eksponensial all-to-all pair-based. `A_plus`, `A_minus`, `tau_plus_steps`, dan `tau_minus_steps` divalidasi positif finite. Spike post yang terlambat mempotensiasi; spike pre yang terlambat mendepresiasi; pasangan waktu sama tidak memperbarui bobot. Bobot dijepit pada batas sinaps. STDP dapat dinyalakan/dimatikan; mematikannya mempertahankan bobot dan menghapus trace.

## Pengujian

Ada 18 tes LIF yang dipertahankan dan 19 tes jaringan/STDP baru. Total **37 tes lulus** pada setiap konfigurasi final:

- `cargo +stable test --all-targets --locked`
- `cargo +stable test --release --locked`
- `cargo +1.85.0 test --all-targets --locked`
- `cargo +1.85.0 test --release --locked`

Format, `cargo check --all-targets --all-features --locked`, Clippy dengan `-D warnings`, dan demo juga lulus pada kedua toolchain. Clippy sempat menemukan satu tuple kompleks; kode diubah menjadi struct hasil bernama dan lint tidak dinonaktifkan.

Tes jaringan mencakup validasi topologi, koneksi duplikat, input, tanda bobot, arah, delay 1/2 langkah, rantai tiga neuron, serta rollback pada overflow neuron dan penjumlahan event sinaptik. Tes STDP memeriksa potensiasi, depresi, spike serentak, akumulasi trace berulang, validasi parameter, saturasi kedua batas, disable, serta rollback saat aritmetika trace overflow.

Demo aktual menunjukkan neuron sumber spike pada langkah 1, neuron tujuan menerima transmisi delay-1 dan spike pada langkah 2, lalu bobot STDP naik dari 20 menjadi sekitar 20.951229 nA.

Perintah gerbang utama:

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

`cargo bench` **BELUM DIJALANKAN** karena benchmark belum diimplementasikan. Tidak ada tes lintas-platform atau jaminan hasil bit-demi-bit lintas compiler/OS.

## Audit anti-implementasi palsu

Jalur source, tes, dan demo diperiksa untuk placeholder, stub, `todo!`, `unimplemented!`, `unreachable!`, `panic!`, `unwrap(`, dan fungsi kosong yang dicurigai; tidak ditemukan. Demo menjalankan API jaringan dan STDP aktual.

## Batasan dan risiko tersisa

Konfigurasi semua neuron dalam satu `LifNetwork` homogen. Satu pasangan berarah sumber-tujuan hanya dapat memiliki satu sinaps. Delay minimal satu langkah dan event sinaptik hanya aktif pada satu interval. Penggunaan bobot bertanda dan rentang bobot memberi kendali jenis sinaps, tetapi pengguna harus memilih rentang untuk mencegah STDP mengubah tanda bila hal itu tidak diinginkan.

STDP yang tersedia adalah aturan pair-based lokal, bukan triplet STDP, reward-modulated STDP, atau pembelajaran tervalidasi untuk tugas. Trace dan event antrean hanya ada di memori dan belum memiliki kebijakan batas/persistence. Benchmark skala jaringan belum tersedia; performa jaringan besar belum diukur. Tidak ada commit Git yang dibuat.

## Langkah yang masuk akal berikutnya

Tambahkan konfigurasi heterogen, profil sinaps dan event yang efisien, benchmark jaringan bertingkat, serta serialisasi snapshot—masing-masing dengan tes batas sebelum menambahkan reward-modulated learning atau memori. Hindari mengklaim kemampuan keputusan/bahasa sampai sub-sistem itu benar-benar dibangun dan diuji.
