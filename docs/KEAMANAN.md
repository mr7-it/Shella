# Keamanan dan batas sumber daya

## File dan integritas

Checkpoint dan store ditulis ke file sementara eksklusif, disinkronkan, lalu dipasang melalui `rename`. Sinkronisasi direktori setelah rename bersifat best-effort; API tidak menjanjikan durability terhadap seluruh skenario mati listrik. Pada Unix file sementara dibuat dengan permission pemilik `0600`. Store memori dibatasi 64 MiB dan checkpoint 256 MiB.

`MemoryStore` legacy memakai JSON dan checksum FNV-1a. Checksum hanya mendeteksi korupsi tak sengaja, bukan autentikasi atau kerahasiaan. Format ini single-writer dengan advisory lock yang ditahan selama object terbuka.

`SecureMemoryStore` menyediakan jalur aman: payload dienkripsi AES-256-GCM dengan nonce acak dari OS, ciphertext ditandatangani Ed25519, lalu envelope JSON ditulis secara atomic. Signature diverifikasi sebelum dekripsi. Setiap mutasi mengambil sidecar lock hanya selama read-modify-write sehingga beberapa object/writer dapat bergantian secara serializable per mutasi. Ini bukan transaksi batch lintas beberapa mutasi.

`ClassKnowledgeModel::save_secure`/`load_secure` menyimpan model sebagai record di `SecureMemoryStore`, sehingga mengikuti enkripsi, signature, verifikasi integritas, dan lock per mutasi. Test memakai key fixture tetap hanya untuk round-trip; rotasi kunci, backup, tenant isolation, approval governance, dan audit actor belum disediakan.

Kunci enkripsi dan signing key harus 32-byte, dikelola secret manager, tidak dicatat ke log, tidak disimpan di repository, dan memiliki prosedur rotasi aplikasi sendiri. SecureMemoryStore belum menyediakan key rotation, backup terenkripsi, escrow, atau migrasi otomatis dari JSON legacy.

## Retensi dan embedding

`MemoryStore::put_with_ttl` menyimpan expiry Unix dan open/purge berikutnya menghapus record expired. TTL bukan secure deletion dari media. `TextEmbedding` adalah hashing lexical deterministik dari token dan character n-gram; ia tidak memberi kerahasiaan dan bukan model semantik neural.

## Validasi input dan komputasi

Parameter numerik, arus, key/tag, embedding, topologi, dan confidence kalibrasi divalidasi finite/domain-nya. Alokasi kritis menggunakan `try_reserve`; allocation errors yang terdeteksi menjadi `SimError`. Aplikasi server tetap harus menerapkan kuota, timeout, autentikasi, izin path, dan isolasi proses.

## Dependency dan kode

Dependency runtime meliputi `serde`, `serde_json`, `fs2`, `aes-gcm`, `ed25519-dalek`, dan `getrandom`, terkunci di `Cargo.lock`. Source melarang unsafe code (`unsafe_code = "forbid"`). Pemeriksaan compile/test tidak menggantikan audit kerentanan dependency berkala.
