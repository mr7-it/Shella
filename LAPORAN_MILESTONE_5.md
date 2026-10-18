# Laporan akhir — Milestone 5 Shella

Tanggal verifikasi: **19 Oktober 2026**. Repository berisi implementasi modular beridentitas publik `shella`; kode referensi tidak disalin. Laporan ini menggambarkan source yang telah melalui gerbang mutu pada stable dan MSRV yang dinyatakan.

Catatan status: hitungan tes dan hasil benchmark dalam dokumen ini adalah snapshot sebelum putaran validasi held-out/checkpoint/SDK terbaru. Gunakan [LAPORAN_VALIDASI.md](LAPORAN_VALIDASI.md) untuk jumlah tes dan hasil kapasitas terkini.

## 1. Identitas paket dan struktur

Package Cargo, library, dan executable dideklarasikan eksplisit bernama `shella`, edition 2024, MSRV Rust 1.85 (`Cargo.toml`). Dependency langsung runtime: `serde 1.0.229` dan `serde_json 1.0.154`; versi transitif dicatat `Cargo.lock`. Source melarang `unsafe`.

Struktur inti di bawah `src/`:

- `config/{mod,validation}.rs`, `error/mod.rs`: konfigurasi/domain, validasi, dan error bertipe.
- `neuron/{mod,lif}.rs`, `runtime/{mod,single_neuron,clock}.rs`: persamaan/transisi LIF dan runtime satu neuron.
- `network/{mod,neuron_state,propagation,scheduler,spike_event,synapse}.rs`: state jaringan, validasi pemulihan, agregasi input, scheduler, event, dan koneksi.
- `learning/{mod,stdp,reward_modulated,eligibility_trace}.rs`: STDP lokal dan reward-modulated.
- `evaluator/{mod,config,metrics,reward}.rs`, `benchmark/{mod,experiments,statistics}.rs`: evaluator reward serta conditioning/recall benchmark.
- `checkpoint/{mod,schema,integrity,save,restore}.rs`: format, checksum, persistence, dan pemulihan.
- `memory/{mod,record,store,retrieval,persistence}.rs`: record, mutasi, query, dan penyimpanan file-backed.
- `decision/{mod,policy,selection}.rs`: kandidat/kebijakan dan seleksi skor/abstain.
- `language/{mod,normalization,tokenizer,vocabulary,encoder}.rs`: pipeline teks deterministik.
- `sdk/{mod,engine,input,output,session,persistence}.rs`: adapter aplikasi bertipe.

Benchmark berada di `benches/learning.rs` dan `benches/system.rs`. Test suite meliputi LIF, jaringan/STDP, Milestone 3–4, checkpoint recovery, persistent memory, decision, language pipeline, dan SDK integration.

## 2. Perilaku yang diimplementasikan

- Neuron LIF memakai solusi eksponensial eksak untuk arus konstan per interval, validasi domain/angka finite, refraktori, reset, counter dan riwayat spike.
- Jaringan multi-neuron mendukung parameter heterogen dengan `dt` bersama, sinaps berarah, bobot eksitatori/inhibitori, event tertunda yang menyimpan snapshot bobot, update dua fase, STDP, dan eligibility yang dimodulasi reward signed.
- Evaluator memetakan target aksi/target diam ke output spike teramati, memberi reward/penalti, mengakumulasi metrik, dan commit metrik hanya setelah apply jaringan berhasil.
- Checkpoint JSON versi 1 mencakup runtime LIF atau seluruh state jaringan serta evaluator opsional. Restore memverifikasi batas, versi, checksum, dan invariants; float round-trip mempertahankan state trajectory.
- `MemoryStore` menyediakan upsert/CRUD, ID stabil, pencarian key/prefix/tag, deteksi korupsi/versi, dan persistence restart.
- Decision policy menggunakan skor fitur linear terurai, tie-break deterministik, ambang abstain, dan bukan probabilitas.
- Pipeline bahasa menormalisasi lowercase Unicode, memakai tokenizer aturan, vocabulary dari korpus, serta memproyeksikan ID token ke arus jaringan. Bukan model bahasa semantik/generatif.
- SDK `Engine`/`Session` menghubungkan input arus/teks ke network, membaca spike aktual, evaluasi aksi/reward, keputusan, memori, dan checkpoint.

## 3. Gerbang verifikasi final

| Gerbang | Rust stable 1.99.1 | Rust MSRV 1.85.0 |
|---|---|---|
| `cargo fmt --all -- --check` | Lulus | Lulus |
| `cargo check --all-targets --all-features --locked` | Lulus | Lulus |
| `cargo clippy --all-targets --all-features --locked -- -D warnings` | Lulus | Lulus |
| `cargo test --all-targets --locked` | Lulus | Lulus |
| `cargo test --release --locked` | Lulus | Lulus |
| `cargo run --release --locked` | Lulus; demo neuron, jaringan, evaluasi/reward aktif | Lulus; output demo yang sama |
| `cargo doc --no-deps --all-features --locked` | Lulus | — |

Terdapat **78 test** terhitung: 2 unit + 76 integrasi; semuanya lulus pada debug dan release untuk kedua toolchain. `cargo test --all-targets` juga membangun target bench pada test profile. Pemeriksaan akhir `git diff --check` lulus. Source audit tidak menemukan `todo!`, `unimplemented!`, `TODO`, `FIXME`, atau placeholder pada source, tests, dan dokumentasi aktif.

## 4. Benchmark Release

Hasil putaran final stable pada sandbox, bukan hasil universal:

- Conditioning 32 epoch, 7 batch × 100 run: kontrol **8/16 (50%)**, reward-trained **16/16 (100%)**, gain **+50 poin persentase**; median eksperimen **230,128 ns** (sekitar 230 µs). Bobot cue mendekat 116.155578 nA, cue menahan 0 nA.
- Benchmark system median lima batch: network step 16/16 neuron/sinaps **2,386 ns** (419,142 langkah/s); 64/64 **8,393 ns** (119,154 langkah/s); 256/256 **22,958 ns** (43,558 langkah/s).
- Checkpoint 64 neuron/64 sinaps: save **762,385 ns**, restore **265,567 ns**, fixture **25,985 byte**.
- Retrieval pada 256 record: prefix **1,901 ns/query**, tag **2,620 ns/query**.

Pengukuran system menjalankan 500 operasi per batch (checkpoint 20), tanpa warm-up terpisah. Waktu yang diukur berubah nyata antar-putaran akibat host/filesystem bersama; angka ini dicatat untuk keterulangan saja dan bukan target SLO atau kapasitas produksi. Benchmark tidak mengukur peak memory, banyak writer/proses, topologi arbitrer, atau generalisasi pembelajaran. Protokol conditioning menguji ulang pola yang dilatih, bukan generalisasi.

## 5. Dependency, lisensi, dan pemeriksaan keamanan

`serde` dan `serde_json` dipilih untuk serialisasi bertipe dan JSON checkpoint/memori. Metadata package terkunci menyatakan keduanya berlisensi `MIT OR Apache-2.0`; MSRV upstream yang dideklarasikan masing-masing 1.56 dan 1.71, di bawah MSRV proyek. Lisensi/MSRV transitif yang diperiksa juga cocok dengan MSRV 1.85. Tidak ada source repository referensi yang dijadikan dependency atau disalin.

`cargo audit` (RustSec, advisory DB memuat 1.338 advisory pada saat pemeriksaan) memindai 12 crate dependency dalam `Cargo.lock` dan tidak melaporkan advisori yang cocok. Ini hanya berarti tidak ditemukan advisory tercatat pada database yang diunduh saat audit; bukan jaminan terhadap masalah baru, supply-chain, atau advisori yang belum diterbitkan. Lihat `docs/REFERENSI.md` dan `docs/KEAMANAN.md`.

Tidak ada lisensi root proyek yang dipilih atau berkas LICENSE yang dibuat dalam milestone ini; pemilihan lisensi proyek tetap keputusan pemilik.

## 6. Masalah yang ditemukan dan perbaikan

- Refaktor jaringan menyisakan import `TryFrom` yang tidak dipakai; dihapus, lalu check/Clippy/test diulang.
- Pemecahan modul decision sempat memberi visibility modifier pada parameter constructor secara tidak sengaja; signature dikoreksi dan `cargo check` lulus.
- Test versi schema memory baru mula-mula belum rustfmt; diformat dan kemudian lulus stable/MSRV.
- `cargo audit` tidak menerima opsi `--locked`; opsi itu dihapus dan command standar membaca `Cargo.lock` serta selesai tanpa advisori.

Tidak ada kegagalan verifikasi yang tersisa pada source akhir.

## 7. Batas dan risiko operasional

- Memory hanya satu writer/proses per path, key/prefix/tag retrieval, tanpa associative retrieval, memory consolidation, decay, atau penghapusan otomatis/TTL.
- Decision score/margin bukan probabilitas atau confidence terkalibrasi.
- Pipeline teks bukan NLP/model bahasa semantik, decoder, generator, embedding, atau pembelajaran teks.
- SDK sinkron dalam proses; tidak ada HTTP/RPC server, autentikasi, cancellation, atau job background.
- Checkpoint dan file memory dipisah; checkpoint SDK tidak menyimpan encoder/policy statis. Aplikasi memasang ulang konfigurasi tersebut.
- FNV-1a hanya checksum integritas kerusakan tak sengaja, bukan cryptographic authentication. File tidak dienkripsi. Rename mengikuti semantik OS/filesystem; directory sync best-effort.
- Batas file checkpoint 256 MiB, store memori 64 MiB, teks 1 MiB; jumlah neuron/sinaps tetap perlu dibatasi aplikasi sesuai kapasitas host.
- Referensi neural-network tidak menyediakan lisensi yang lengkap pada checkout `rusty-snn`; kodenya tidak disalin.

## 8. Penggunaan dan reproduksi

Dari root proyek:

```sh
cargo run --release --locked
cargo test --all-targets --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo bench --bench learning --locked
cargo bench --bench system --locked
cargo audit
```

Dokumentasi tambahan: [README](README.md), [API](docs/API.md), [arsitektur](docs/ARSITEKTUR.md), [keamanan](docs/KEAMANAN.md), dan [pengujian](docs/PENGUJIAN.md).
