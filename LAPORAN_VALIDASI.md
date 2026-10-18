# Laporan validasi Shella

Tanggal pengujian: **19 Oktober 2026**. Ini adalah putaran uji lanjutan terhadap project Shella, bukan klaim validasi klinis/ilmiah atau hasil pada hardware pengguna.

## Ringkasan hasil

| Area | Hasil |
|---|---|
| Belajar dan pola held-out | 3 tes baru lulus; model mempelajari asosiasi cue baru dengan reward pada tugas kecil, merespons satu variasi held-out, dan tidak mengeluarkan aksi untuk cue tanpa asosiasi. |
| Checkpoint | 5 skenario + 1 helper subprocess; restore/step pada proses OS baru konsisten; payload rusak, file terpotong, versi asing, dan I/O invalid ditolak. |
| Memori | 4 skenario + 1 helper subprocess; CRUD/reopen konsisten dan child-process mengubah record persisten; korupsi/schema asing ditolak. |
| SDK | 3 tes lulus; teks → neuron → asosiasi terlatih → evaluasi reward → keputusan → memori → checkpoint dijalankan end-to-end. |
| Kapasitas | Grid aktual Sandbox tertinggi 16,384 neuron / 65,536 sinaps; peak RSS proses 65.418 MiB, median step 3,363.307 µs, checkpoint 20,473,318 byte. Bukan ukuran maksimum sistem dan bukan hasil perangkat pengguna. |

## 1. Validasi belajar: pola baru dan generalisasi terbatas

`tests/generalization.rs` memakai jaringan empat neuron: tiga cue dan satu neuron aksi, sinaps cue→aksi, reward-modulated STDP, serta evaluator target. Training adalah **32 trial feedback** per asosiasi: cue 100 nA, teacher spike di neuron aksi, lalu reward. Uji recall berjalan tanpa teacher.

| Probe | Data training | Data uji yang tak muncul saat training | Hasil |
|---|---|---|---|
| Variasi held-out | cue-0 → aksi | cue-0 80 nA + cue-1 80 nA sebagai distractor yang tidak dilatih | Jaringan learned benar **1/1**; kontrol bobot nol tidak menghasilkan aksi (**0/1**). Bobot cue-0 positif; cue-1 dan cue-2 tetap 0. |
| Akuisisi asosiasi baru | cue-0 → aksi lebih dahulu, lalu cue-2 → aksi dalam sesi pembelajaran yang sama | cue-2 80 nA tanpa teacher | Jaringan learned benar **1/1**; kontrol fresh yang tidak menerima reward tidak menghasilkan aksi. Bobot cue-2 menjadi positif. |
| Cue tanpa asosiasi | hanya cue-0 → aksi | cue-2 sendiri, tanpa pernah dilatih | Jaringan tidak mengeluarkan aksi; kontrol juga diam. Ini menunjukkan tidak ada transfer otomatis ke cue arbitrer. |

Kesimpulan yang didukung: pada tugas sintetis dan terkontrol ini Shella **dapat membentuk asosiasi spike baru dari feedback/reward**, mempertahankan respons pada satu variasi amplitudo, dan mengabaikan satu distractor yang belum dilatih. Sampel held-out sangat kecil (satu pola per probe), deterministik, tanpa noise/data bahasa; hasil ini **bukan** generalisasi semantik atau generalisasi luas. Cue-2 tanpa training justru menunjukkan batas transfer.

## 2. Validasi memori dan checkpoint

- Tes checkpoint membandingkan eksekusi uninterrupted dengan eksekusi yang menyimpan state, memulai **proses OS anak baru**, memulihkan file, lalu menjalankan langkah berikutnya. Debug output step dan seluruh state network sesudah step dibandingkan persis.
- Tes round-trip lain membandingkan state jaringan, pending delayed event, eligibility trace, evaluator/metrics, spike history, refractory state, serta kelanjutan trajectory setelah restore.
- Checkpoint diuji terhadap payload yang checksum-nya tidak cocok, JSON terpotong, `format_version` tak didukung (`999`), dan path I/O invalid; error bertipe yang sesuai dipastikan muncul.
- Memori diuji dengan menutup store pada proses induk, membuka dan memutakhirkan record di proses baru, lalu membukanya lagi dari induk. ID stabil, nilai, dan tag hasil proses anak harus persis bertahan.
- File memory rusak dan versi schema asing (`99`) ditolak; input invalid tidak meninggalkan record parsial.

Hasil harness: checkpoint **6 entri** (5 skenario mandiri + 1 helper) dan memory **5 entri** (4 skenario mandiri + 1 helper) lulus pada debug dan release. Helper sengaja kembali langsung jika environment variable pemanggil tidak ada, lalu dijalankan dengan input yang diperlukan oleh parent test di proses anak; assertion hasil lintas proses berada pada parent test. Uji proses adalah restart berurutan untuk pemulihan—bukan uji writer bersamaan.

## 3. Kapasitas, RAM, waktu, dan ukuran checkpoint

Pengukuran dilakukan pada **Manus Sandbox**, VM Linux amd64/KVM, bukan PC/perangkat fisik pengguna. MCP hanya menunjukkan Sandbox sebagai perangkat yang tersedia; tidak ada desktop atau Cloud Computer pengguna untuk dialihkan. Host yang terlihat: **6 vCPU Intel Xeon @ 2.50 GHz**. Snapshot `/proc/meminfo` saat inspeksi menunjukkan 14,387,492 kB RAM terlihat dan 12,455,436 kB available; provider tidak menyediakan telemetri RAM dan limit cgroup tidak terlihat. Detail/freshness ada di [metadata lingkungan](results/capacity_environment_2026-10-19.md).

Setiap ukuran dijalankan di proses terpisah dalam Release. Grid memakai graph sinaps berarah deterministik tanpa self-edge; seluruh neuron menerima 100 nA. Median `LifNetwork::step` dihitung dari **5 batch × 20 langkah** per ukuran (100 langkah total). `VmHWM` adalah peak RSS proses, sehingga mencakup overhead proses/allocator dan buffer checkpoint, bukan RAM marginal object jaringan.

| Neuron / sinaps | Build (ms) | Median step (µs) | Step/s | Peak RSS (`VmHWM`, MiB) | Checkpoint (byte) | Simpan checkpoint (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 / 64 | 0.018 | 4.615 | 216,664 | 2.953 | 35,801 | 1.585 |
| 256 / 1,024 | 0.510 | 37.053 | 26,988 | 3.777 | 316,582 | 5.261 |
| 1,024 / 4,096 | 8.314 | 128.208 | 7,800 | 6.516 | 1,268,326 | 17.683 |
| 4,096 / 16,384 | 91.557 | 604.856 | 1,653 | 18.312 | 5,099,110 | 142.649 |
| 8,192 / 32,768 | 600.569 | 934.296 | 1,070 | 34.090 | 10,206,822 | 620.373 |
| 16,384 / 65,536 | 3,684.197 | 3,363.307 | 297 | 65.418 | 20,473,318 | 3,915.211 |

Data mentah: [CSV](results/capacity_sandbox_2026-10-19.csv) dan [NDJSON](results/capacity_sandbox_2026-10-19.ndjson). Kode probe: [capacity_probe.rs](examples/capacity_probe.rs), skrip reproduksi: [run_capacity_probe.sh](scripts/run_capacity_probe.sh). Jalankan dari root dengan `STEPS_PER_BATCH=20 bash scripts/run_capacity_probe.sh`.

Angka menunjukkan biaya step, build, dan serialisasi meningkat dengan skala pada workload ini; angka timing sensitif terhadap host bersama. **16,384/65,536 adalah skala terbesar yang dicoba, bukan batas maksimum atau jaminan performa.** Belum diuji pemakaian jangka panjang, beban banyak pengguna/writer, graph padat/berbeda, peak memory deployment, throttling, maupun perangkat target.

## 4. Validasi SDK

Tes `sdk_teks_ke_jaringan_terlatih_ke_evaluasi_decision_dan_memory` menjalankan keseluruhan alur berikut:

1. Encoder mengubah input teks `alpha beta` menjadi arus dua neuron cue.
2. Jaringan memproses cue, menerima teacher spike, dan reward evaluator menerapkan feedback selama 32 trial.
3. Setelah washout, teks yang sama tanpa teacher memicu neuron aksi melalui koneksi terlatih.
4. Evaluator menilai output aksi yang aktual; decision policy memilih kandidat `active` berdasarkan spike aktual.
5. Output dan aksi disimpan ke persistent memory; checkpoint state terlatih disimpan.

Assertion pada tes memastikan cue spikes, respons aksi hasil pembelajaran, evaluasi benar, keputusan `active`, record memori aktual, serta checkpoint tersimpan. Suite SDK **3/3** lulus. Tes SDK ini memvalidasi integrasi dalam proses; restore lintas proses diuji khusus pada suite checkpoint/memory.

## 5. Keterbatasan yang tersisa

- **Pemahaman bahasa:** tokenizer aturan dan vocabulary hanya memberi ID deterministik; tidak ada semantik, konteks, decoder/generator, embedding, atau inferensi bahasa alami. Uji `alpha beta` membuktikan jalur input, bukan pemahaman kalimat.
- **Generalisasi:** probe hanya satu asosiasi spike dan sedikit pola held-out. Tidak ada dataset independen, validasi statistik, cross-seed, transfer antar-neuron/representasi, komposisi semantik, maupun benchmark adversarial. Cue baru perlu feedback tersendiri.
- **Keamanan storage:** memory/checkpoint berupa JSON lokal plaintext, tanpa enkripsi atau autentikasi. FNV-1a/checksum checkpoint mendeteksi korupsi tak sengaja tetapi bukan signature dan tidak mencegah penyerang mengubah data beserta checksum. Permission owner-only di Unix membantu membatasi akses lokal, bukan melindungi data bila akun/host dikompromikan.
- **Konsistensi/concurrency:** memory file-backed mengasumsikan satu writer; tidak ada lock lintas proses, transaksi database multi-writer, backup otomatis, retensi/TTL, atau pemulihan filesystem universal. Test restart berurutan tidak menjamin keselamatan write concurrent.
- **Perangkat:** angka kapasitas berasal dari VM KVM Sandbox, bukan perangkat deployment/pengguna. Untuk klaim performa target, skrip benchmark perlu dijalankan pada perangkat target dengan toolchain, batas CPU/RAM, filesystem, dan beban aplikasi yang sebenarnya.

## Verifikasi akhir

Gerbang `fmt`, `check --all-targets --all-features`, Clippy dengan `-D warnings`, seluruh test debug/release, demo release, dan `cargo audit` lulus pada stable Rust 1.99.1 serta MSRV Rust 1.85.0. Jumlah final **86 entri test harness (2 unit + 84 integrasi)**; dua entri integrasi adalah helper child-process yang dipanggil dan diverifikasi oleh parent test. RustSec audit tidak menemukan advisori dikenal yang cocok pada 12 dependency di `Cargo.lock`; hasil itu bukan jaminan terhadap advisori baru atau risiko keamanan penyimpanan aplikasi.
