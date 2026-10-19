# Memori persisten

`MemoryStore::open(path)` membuka file yang ada atau membuat database kosong. `put(key, serde_json::Value, tags)` melakukan insert/upsert: key unik dan ID `rec-…` tetap sama saat key diperbarui. Retrieval berada pada `memory/retrieval.rs`: `get`, `get_by_id`, `search_prefix`, `search_tag`, `records`, `len`, dan `is_empty`. `delete` eksplisit. Tag dinormalisasi unik/terurut, hasil pencarian deterministik.

`memory/record.rs` memuat schema catatan; `memory/store.rs` menangani operasi mutasi; `memory/persistence.rs` memvalidasi schema, checksum, ukuran, dan menulis snapshot. Mutasi menyiapkan snapshot kandidat, menserialisasi, menulis file sementara eksklusif, sinkronisasi isi, lalu `rename`; state dalam memori diubah sesudah persist berhasil. Sinkronisasi direktori setelah rename best-effort.

`MemoryStore::open` mengambil advisory exclusive lock pada sidecar `<path>.lock` sebelum membaca atau membuat database. Path dinormalisasi sehingga alias relatif menuju file yang sama memakai lock yang sama. Hanya satu instance/proses dapat memegang lock; pembukaan berikutnya gagal dengan `SimError::Io { operation: "kunci memori", kind: WouldBlock }`. Lock dilepas otomatis saat `MemoryStore` ditutup atau proses berakhir. File sidecar sengaja tidak dihapus otomatis; jangan menghapusnya saat ada proses yang mungkin masih membuka store, karena lock OS melekat pada file yang terbuka.

Format v1 menggunakan JSON dan checksum FNV-1a; maximum file 64 MiB, key/tag masing-masing dibatasi 16 KiB. File sementara Unix dibuat dengan mode `0600`. Error terstruktur mencakup `CorruptData { subsystem: "memori", .. }`, `UnsupportedDataVersion`, `SerializationFailed`, `Io`, validasi input, dan allocation/error lain yang relevan.

Penyimpanan menegakkan satu penulis lintas proses per path melalui advisory lock. Lock ini bukan transaksi multi-writer dan bergantung pada dukungan locking filesystem; jangan menghapus atau mengganti sidecar lock secara manual saat aplikasi berjalan. Tidak ada enkripsi, pencarian semantik/asosiatif, TTL, consolidation, atau garbage collection otomatis. Penghapusan logis melalui `delete`; retensi media ditangani aplikasi/host.
