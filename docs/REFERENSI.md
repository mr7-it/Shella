# Audit referensi dan dependency

## Repository referensi

Audit dilakukan pada shallow clone lokal di `references/`. Kode sumber tidak disalin dan tidak ada repository referensi yang ditambahkan sebagai dependency.

### CogSpike

Commit yang diaudit: `42d3d9e82b7d0bd10cc1ac541b910a882ed57f28`.

Repository [QuietRocket/CogSpike](https://github.com/QuietRocket/CogSpike) menyediakan core simulasi terpisah dari UI dan tes perilaku jaringan. Workspace menyatakan edition 2024/Rust 1.88 dan lisensi MIT/Apache-2.0. Model firing diskret/probabilistik berbeda dari integrasi LIF eksak proyek ini. Tidak ada source yang disalin.

### neuromod

Commit yang diaudit: `1aaf21bda102f7ca4eee5125fb8d3c6d8400c890`.

Repository [Limen-Neural/neuromod](https://github.com/Limen-Neural/neuromod) menyatakan edition 2024, MSRV Rust 1.98.1, dan MIT OR Apache-2.0 dengan berkas lisensi tersedia. Model yang ditinjau memakai keadaan f32/decay. Tidak ada source yang disalin.

### rusty-snn

Commit yang diaudit: `e36085c0be231c6edb4dbba8468729426cb1f141`.

Repository [haguettaz/rusty-snn](https://github.com/haguettaz/rusty-snn) menyatakan edition 2021 dan `MIT OR Apache-2.0` pada manifest, tetapi shallow clone tidak memuat file license root. Project berfokus pada SNN continuous-time dan beberapa dependency numerik/solver. Tidak ditemukan `rust-version` pada manifest yang diperiksa. Karena teks lisensi tidak tersedia pada checkout itu, kodenya tidak dipakai maupun disalin.

## Dependency runtime Shella

Serde/serde_json ditambahkan untuk manfaat teknis yang spesifik: derive serialisasi bertipe dan format JSON berversi bagi checkpoint/memori. Versi langsung di `Cargo.lock`:

| Crate | Versi | Lisensi deklaratif | MSRV deklaratif |
|---|---:|---|---:|
| `serde` | 1.0.229 | MIT OR Apache-2.0 | 1.56 |
| `serde_json` | 1.0.154 | MIT OR Apache-2.0 | 1.71 |

Transitif terkunci antara lain `serde_core`, `serde_derive`, `proc-macro2`, `quote`, `syn`, `unicode-ident`, `itoa`, `memchr`, dan `zmij`; metadata Cargo lokal mencatat lisensi permisif MIT/Apache-2.0, Unicode-3.0, Unlicense OR MIT, atau MIT sesuai crate. Rust MSRV tiap transitif yang diperiksa tidak lebih tinggi daripada 1.71, di bawah MSRV proyek 1.85. Tidak ada kode dari dependency yang disalin. Versi transitif aktual dapat dilihat melalui `cargo tree --locked`.

Pemeriksaan lisensi/MSRV dilakukan dari metadata manifest crate yang terkunci. Pemeriksaan `cargo audit` terhadap advisory database dilakukan terpisah dan hasilnya tercatat pada laporan Milestone 5. Pemeriksaan tersebut bukan jaminan tidak ada kerentanan yang belum terdaftar; update dependency serta audit berkala tetap tanggung jawab pemelihara.
