# Laporan Milestone 6 Shella

Tanggal: **19 Oktober 2026**. Ini adalah iterasi M6 yang terukur dan tervalidasi pada Sandbox, bukan klaim bahwa semua pekerjaan Milestone 6 telah selesai.

## Lingkup yang dikerjakan

Saya menambahkan harness continual learning deterministik di `src/benchmark/continual.rs`, mengintegrasikannya ke API benchmark, serta menyediakan runner `examples/milestone6_continual.rs`. Eksperimen menjalankan empat asosiasi dalam dua fase. Setiap pola memakai satu cue bersama lintas kelas dan satu cue pembeda. Jaringan berisi 9 neuron dan 20 sinaps; bobot awal nol, batas bobot cue bersama 50 nA, batas cue pembeda 150 nA, learning rate 2.5, STDP pair amplitude 1.0/1.0, dan eligibility tau 20 langkah.

Setiap contoh memakai cue 100 nA dan teacher spike 200 nA. Reward evaluator membaca spike aksi aktual; sesudah contoh, jaringan menjalani 100 langkah tanpa input agar eligibility dari trial sebelumnya meluruh. Protokol standar menjalankan lima seed `[1, 7, 19, 42, 99]`, 32 epoch per fase, 128 contoh teacher-forced, dan 13.056 langkah jaringan per seed. Seed hanya mengacak urutan asosiasi secara deterministik; bobot awal, isi kelas, dan nilai amplitudo tidak diacak.

Train memakai amplitudo 100 nA. Validasi memakai 80/90/110/120 nA. Test akhir memakai 97/103 nA. Keempat nilai validation dan kedua nilai test berbeda dari training dan satu sama lain. Tiap stimulus evaluasi dimulai dari clone state hasil training yang sama, tanpa teacher atau reward; dengan demikian hasil sebuah stimulus tidak dipengaruhi sampel evaluasi sebelumnya.

## Hasil aktual

| Metrik | Hasil pada 5 seed |
|---|---:|
| Baseline test tanpa training | 0/8 (0%) |
| Test setelah training | 8/8 (100%) pada setiap seed |
| Validasi | 16/16 (100%) pada setiap seed |
| Gain test terhadap kontrol | +100 poin persentase |
| Task A setelah fase pertama | 100% |
| Task A setelah fase kedua | 100% |
| Forgetting Task A yang terukur | 0 poin persentase |
| Reward training teacher-forced | 128 untuk 128 contoh |

Hasil per seed tersedia di [CSV continual learning](results/milestone6_continual_2026-10-19.csv). Eksperimen ini memvalidasi pembentukan asosiasi baru di jaringan LIF/STDP pada tugas sintetis berfitur bersama dan cue pembeda. Nilai reward teacher-forced bukan ukuran recall mandiri; ukuran recall adalah evaluasi tanpa teacher pada split validation/test.

Batas interpretasi penting: seed hanya mengubah urutan asosiasi, jadi lima hasil identik bukan lima dataset atau lima inisialisasi independen. Test menahan amplitudo cue, bukan kelas semantik baru. Forgetting nol adalah hasil pada tugas kecil ini, ketika task berbagi cue umum tetapi tetap mempunyai cue pembeda dan bobot cue umum dibatasi. Ini **belum membuktikan pencegahan catastrophic forgetting** pada pemetaan konflik, replay jangka panjang, noisy input, atau distribusi yang berbeda. Eksplorasi awal juga menghasilkan skor lebih rendah; protokol final memakai parameter terpilih lewat validasi dan pasangan amplitudo test baru 97/103 nA.

## Perubahan robustness dan tes regresi

`DecisionPolicy::select` sekarang menolak margin yang overflow walaupun skor kandidat satu per satu finite. Tes regresi menggunakan skor ekstrem `+f64::MAX` dan `-f64::MAX`.

`TextEncoder::encode` sekarang menolak input yang tidak menghasilkan token, termasuk teks kosong, whitespace-only, dan punctuation-only. Ini mencegah pipeline menghasilkan vektor nol seolah-olah sebuah input teks valid. Token OOV masih dipetakan ke `<unk>`/ID 0 dan belum dilengkapi rasio OOV atau status unknown.

Tes M6 mencakup konfigurasi seed/epoch yang invalid, determinisme antar-run, akurasi held-out multi-seed, gain, retensi task A, dan nilai validation per amplitudo. Peta tes serta dokumentasi evaluator, bahasa, keputusan, dan README telah diperbarui.

## Audit keadaan subsistem lain

- **MemoryStore:** source mendukung persist atomik per snapshot dan tes restart berurutan. Kontraknya tetap satu writer; tidak ada lock lintas proses atau jaminan mencegah lost update ketika dua writer membuka path yang sama. Uji concurrent multi-writer belum ditambahkan.
- **Checkpoint:** checksum dan validasi menolak berbagai data korup/tidak konsisten. Implementasi tidak menyediakan backup generation atau fallback otomatis setelah checkpoint rusak; “restore” berarti pemulihan state yang valid setelah restart, bukan corruption recovery. FNV-1a bukan autentikasi.
- **Decision engine:** skor linear transparan dan abstain berbasis ambang/margin tersedia. FP/FN, precision/recall, coverage-risk, dan kalibrasi probabilitas belum ada; skor/margin bukan confidence.
- **Pipeline bahasa:** tokenisasi Unicode dan vocabulary deterministik tetap preprocessing bag-of-token, bukan pemahaman semantik. Semua token OOV masih menyatu pada ID 0; `tokenize` dapat mengembalikan daftar kosong meskipun encoder kini menolaknya.
- **SDK:** alur input/evaluasi/decision/memory tersedia tetapi komponennya dirangkai pemanggil; log audit otomatis atau provenance output lintas session belum ditambahkan.

## Gerbang mutu dan status Git

Pada toolchain stable 1.99.1, lulus `cargo fmt --all -- --check`, `cargo check --all-targets --locked`, Clippy dengan `-D warnings`, `cargo test --all-targets --locked`, `cargo test --release --locked`, serta kedua benchmark `learning` dan `system`. Pada MSRV Rust 1.85.0, lulus `cargo test --all-targets --locked` dan Clippy ketat. Terdapat **91 entri tes per suite lengkap** dan nol kegagalan.

Benchmark learning lama tetap melaporkan conditioning dua pola: kontrol 8/16 (50%) dan learned 16/16 (100%); itu terpisah dari eksperimen M6 empat asosiasi di atas. Rerun benchmark system Release mengukur median step 1.222/4.504/18.997 ns pada jaringan 16/64/256 neuron; checkpoint save/restore 550.240/133.361 ns untuk 64 neuron, ukuran checkpoint 25.985 byte, dan prefix/tag search 1.885/2.468 ns per query pada 256 record. Angka ini sensitif terhadap host dan bukan batas kapasitas baru.

Repository berada pada branch `master` tanpa commit awal. Karena itu `git status` menampilkan seluruh struktur proyek sebagai untracked; tidak ada file yang di-stage atau di-commit selama iterasi ini. Perubahan dibuat di struktur existing Shella dan tidak memulai ulang proyek dari nol.

## Langkah berikutnya

Pekerjaan lanjutan yang masih relevan adalah mengukur forgetting pada tugas yang benar-benar saling mengganggu, membandingkan replay/consolidation dengan kontrol tanpa replay, memperluas multi-seed melampaui perubahan urutan, menambah evaluasi FP/FN dan abstain pada labeled validation/test, serta menguji kontrak multi-writer atau menegaskan penegakan single-writer pada level API. Benchmark semantik harus memakai dataset independen yang sesuai sebelum klaim generalisasi bahasa dibuat.
