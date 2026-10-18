# Keamanan dan batas sumber daya

## File dan integritas

Checkpoint dan `MemoryStore` ditulis ke file sementara eksklusif pada direktori tujuan, isi file disinkronkan, lalu dipasang melalui `rename`. Keatomikan penggantian mengikuti semantik filesystem/OS. Sinkronisasi direktori setelah rename bersifat best-effort; API tidak menjanjikan durability terhadap seluruh skenario mati listrik. Pada Unix file sementara dibuat dengan permission pemilik `0600`. Penggantian file tujuan lama melalui `rename` telah diverifikasi pada lingkungan Linux yang digunakan.

Checkpoint dibatasi 256 MiB dan store memori 64 MiB. Pembacaan membatasi jumlah byte; deserialisasi memeriksa versi schema, checksum, domain nilai, topologi, counter, dan invariants state sebelum data dipakai. Checksum FNV-1a mendeteksi kerusakan tak sengaja saja. Algoritmenya bukan kriptografis dan bukan autentikasi: penyerang yang dapat mengganti konten juga dapat menghitung checksum baru.

Store JSON memori hanya mendukung satu proses penulis per path. Tidak ada file lock, transaksi lintas proses, enkripsi at-rest, penghapusan aman dari media, retensi otomatis, atau proteksi kerahasiaan selain permission filesystem. Jangan membuka path yang sama dengan beberapa writer. Jangan mempercayai checkpoint/store dari sumber tidak tepercaya sebagai batas keamanan aplikasi.

## Validasi input dan komputasi

Parameter numerik konfigurasi dan arus diperiksa finite/domain-nya; ukuran topologi kosong, indeks, delay, target aksi, dan skor invalid ditolak. Jaringan menggunakan `try_reserve` pada alokasi neuron/event kritis; allocation errors yang terdeteksi menjadi `SimError`. Teks dinormalisasi dengan batas 1 MiB dan menolak NUL. Key/tag memory dibatasi 16 KiB tanpa karakter kontrol. File output storage dibatasi sesuai nilai di atas.

Tidak ada batas global jumlah neuron/sinaps yang aman untuk setiap deployment: alokasi perlu disesuaikan kapasitas host dan beban kerja. SDK sinkron; operasi panjang tidak dapat dibatalkan melalui API SDK. Aplikasi server/layanan harus menerapkan kuota, timeout, autentikasi, izin path, dan isolasi proses sendiri sebelum mengekspos API ke jaringan.

## Dependency dan kode

Dependency langsung runtime adalah `serde` dan `serde_json`, terkunci di `Cargo.lock`. Source melarang unsafe code (`unsafe_code = "forbid"`). Audit versi/lisensi yang tercatat ada pada `docs/REFERENSI.md`; status kerentanan eksternal tidak dijamin oleh pemeriksaan compile/test dan perlu dipantau melalui toolchain/dependency audit organisasi.
