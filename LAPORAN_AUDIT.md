# Laporan audit dan milestone 1

> **Catatan status:** laporan ini adalah snapshot historis saat milestone terkait dikerjakan. Fitur/batas yang tertulis sebagai “belum tersedia” merujuk keadaan pada tahap tersebut; untuk kondisi sekarang gunakan README, dokumentasi `docs/`, dan LAPORAN_MILESTONE_5.md.

## A. Kondisi awal repository

Pada pemeriksaan awal, `/home/ubuntu` bukan repository Git dan tidak memuat `Cargo.toml`, `Cargo.lock`, atau kode proyek. Tidak ditemukan workspace proyek Rust pada lokasi umum yang diperiksa. `rustc` dan `cargo` juga belum tersedia. Karena tidak ada kode pengguna yang dapat diubah, proyek baru dibuat secara terpisah di `/home/ubuntu/shella`; tidak ada repository lama yang ditimpa.

## B. Tujuan pekerjaan

Menyelesaikan tahap pertama spesifikasi: runtime deterministik dan model neuron LIF yang benar-benar menghitung dinamika tegangan, mencatat spike, mengelola refraktori, menangani input tidak sah dan overflow secara bermakna, menyediakan demonstrasi, dan memiliki tes perilaku.

## C. Keputusan teknis

Model menggunakan persamaan integrasi eksponensial analitis untuk arus konstan pada setiap interval. Satuan ditetapkan sebagai ms, mV, MΩ, dan nA. Tegangan ambang diperiksa setelah integrasi; spike dicatat dengan nomor langkah selesai basis satu, kemudian potensial di-reset.

Runtime menjalankan satu neuron secara serial. Ini cukup untuk membuktikan kontrak waktu dan perilaku tahap pertama tanpa mengklaim jaringan multi-neuron. Langkah yang gagal bersifat atomik terhadap keadaan neuron. Batch `run` tidak melakukan rollback atas langkah yang telah berhasil sebelum masukan berikutnya gagal.

Tidak ada dependency eksternal. Aritmetika, validasi, error, penyimpanan vektor hasil, dan pengujian ditangani oleh standard library. Deklarasi lisensi proyek sengaja tidak dipilih karena pengguna belum menetapkan lisensi dan berkas lisensi belum tersedia.

## D. Dependency yang berubah

Tidak ada dependency eksternal yang ditambahkan. `Cargo.lock` hanya mencatat crate proyek itu sendiri.

## E. Berkas yang dibuat atau diubah

Dibuat manifest dan lockfile crate, README, library API, `main.rs` demo, konfigurasi, error, model `LifNeuron`, runtime, modul neuron, 18 tes integrasi, serta dokumentasi arsitektur, model matematis, determinisme, pengujian, audit referensi, dan laporan ini. `.gitignore` mengecualikan `target/` serta clone referensi lokal. Tidak ada berkas yang dihapus.

## F. Implementasi nyata yang selesai

`LifConfig` memvalidasi parameter finite dan positif serta relasi potensial istirahat/reset terhadap ambang. `LifNeuron` melakukan integrasi eksponensial memakai `expm1` untuk mengurangi kehilangan presisi ketika langkah waktu kecil. Runtime memvalidasi input, mendeteksi spike, mengatur reset dan refraktori, memelihara penghitung langkah/spike, dan menyimpan indeks spike.

Overflow hasil numerik, kegagalan penghitung, alokasi catatan spike, NaN, dan infinity dikembalikan sebagai kategori error yang berbeda. Kegagalan satu langkah tidak mengubah keadaan neuron pada langkah tersebut. Demo menjalankan model yang sama melalui API library.

## G. Pengujian yang ditambahkan

Terdapat 18 tes integrasi perilaku yang mencakup kestabilan potensial istirahat, peluruhan eksponensial, trajektori arus konstan terhadap solusi analitis, arus subambang, waktu dan urutan spike, reset, refraktori, penghitung, reproduksibilitas, kesetaraan run penuh dengan run terpisah, konfigurasi tidak sah, batas parameter, NaN, infinity bertanda positif dan negatif, overflow numerik, keadaan pascakegagalan, serta kegagalan di tengah batch.

## H. Perintah yang dijalankan

Gerbang berikut dijalankan pada Rust 1.99.1 dan juga pada MSRV yang dideklarasikan, Rust 1.85.0. Untuk pemeriksaan MSRV, perintah Cargo menggunakan `--locked`:

```text
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
cargo test --release
cargo run --release

cargo +1.85.0 fmt --all -- --check
cargo +1.85.0 check --all-targets --all-features --locked
cargo +1.85.0 clippy --all-targets --all-features --locked -- -D warnings
cargo +1.85.0 test --all-targets --locked
cargo +1.85.0 test --release --locked
cargo +1.85.0 run --release --locked
```

`cargo bench` tidak dijalankan karena proyek belum memiliki benchmark; ini sesuai instruksi untuk menjalankan benchmark hanya jika sudah diimplementasikan.

## I. Status keluar dan ringkasan hasil

Format, check, Clippy, tes debug, tes release, dan demo berstatus **LULUS** pada kedua toolchain yang diuji. Tes integrasi: 18 lulus, 0 gagal. Demo release berjalan dan mencatat spike pertama pada langkah ke-7, mereset potensial ke −70 mV, lalu menahan dua langkah refraktori.

Pada putaran pemeriksaan awal setelah penulisan kode, deklarasi modul neuron belum lengkap sehingga format dan kompilasi gagal. Setelah modul diperbaiki, dua tes gagal karena oracle analitis memakai tanda eksponensial yang keliru dan fixture overflow memasuki periode refraktori sebelum memicu perhitungan yang ingin diuji. Kedua penyebab diperbaiki pada dokumentasi/fixture tes; tes tidak dihapus atau dilemahkan. Semua gerbang terkait kemudian dijalankan ulang dan lulus.

## J. Audit anti-implementasi palsu

Pencarian pada `src`, `tests`, dan `examples` tidak menemukan `todo!`, `unimplemented!`, `unreachable!`, placeholder, skeleton, dummy, mock, FIXME/TODO, `panic!`, atau `unwrap(`. Pemeriksaan fungsi kosong yang dicurigai juga tidak menemukan hasil. Demo menjalankan API inti dan bukan keluaran tetap.

## K. Kriteria penerimaan yang lulus

Implementasi runtime dan LIF nyata tersedia; kode terkompilasi; 18 tes perilaku dan tes integrasi lulus; Clippy tidak menghasilkan warning; format sesuai; demo aktual berhasil; dokumentasi menjelaskan persamaan dan kontrak kegagalan; dan pengujian reproduksibilitas berhasil pada lingkungan yang diuji. MSRV 1.85.0 telah diuji langsung.

**Status milestone 1: LULUS.** Status ini hanya berlaku untuk runtime satu neuron LIF pada tahap pertama, bukan keseluruhan platform AI neuromorfik.

## L. Kriteria gagal atau belum dijalankan

Tidak ada gerbang yang gagal pada verifikasi final. Benchmark **BELUM DIJALANKAN** karena belum ada benchmark yang diimplementasikan. Belum ada tes lintas platform atau jaminan bit-identik lintas compiler dan sistem operasi.

## M. Masalah yang masih tersisa

Runtime hanya mengelola satu neuron. Riwayat indeks spike tumbuh tanpa batas selama simulasi berjalan; kegagalan reservasi ditangani, tetapi belum ada kebijakan batas atau pembuangan riwayat. Tidak ada interpolasi waktu spike di dalam interval, sehingga resolusi waktu spike dibatasi oleh `dt`.

## N. Risiko dan keterbatasan

Model menerima satu arus konstan per langkah dan memakai floating-point `f64`. Reproduksibilitas diuji untuk konfigurasi dan urutan input yang sama pada toolchain/lingkungan ini, bukan sebagai jaminan hasil bit-demi-bit lintas platform. Proyek belum memiliki sinaps, topologi jaringan, pembelajaran, checkpoint, memori, pengambilan keputusan, atau pemrosesan bahasa. Cargo manifest mendeklarasikan `rust-version = 1.85`; tidak ada pilihan lisensi proyek yang dibuat.

## O. Langkah berikutnya

Tahap berikut yang konsisten dengan spesifikasi adalah jaringan multi-neuron dan sinaps dengan definisi arah koneksi, penundaan, serta aturan pembaruan dua fase yang deterministik. Jaringan kecil perlu diuji terhadap hasil yang dapat dihitung manual sebelum STDP dimulai.
