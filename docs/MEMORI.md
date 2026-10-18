# Memori persisten

`MemoryStore::open(path)` membuka file yang ada atau membuat database kosong. `put(key, serde_json::Value, tags)` melakukan insert/upsert: key unik dan ID `rec-…` tetap sama saat key diperbarui. Retrieval berada pada `memory/retrieval.rs`: `get`, `get_by_id`, `search_prefix`, `search_tag`, `records`, `len`, dan `is_empty`. `delete` eksplisit. Tag dinormalisasi unik/terurut, hasil pencarian deterministik.

`memory/record.rs` memuat schema catatan; `memory/store.rs` menangani operasi mutasi; `memory/persistence.rs` memvalidasi schema, checksum, ukuran, dan menulis snapshot. Mutasi menyiapkan snapshot kandidat, menserialisasi, menulis file sementara eksklusif, sinkronisasi isi, lalu `rename`; state dalam memori diubah sesudah persist berhasil. Sinkronisasi direktori setelah rename best-effort.

Format v1 menggunakan JSON dan checksum FNV-1a; maximum file 64 MiB, key/tag masing-masing dibatasi 16 KiB. File sementara Unix dibuat dengan mode `0600`. Error terstruktur mencakup `CorruptData { subsystem: "memori", .. }`, `UnsupportedDataVersion`, `SerializationFailed`, `Io`, validasi input, dan allocation/error lain yang relevan.

Penyimpanan adalah satu penulis dalam satu proses untuk satu path. Tidak ada file lock, transaksi lintas proses, enkripsi, pencarian semantik/asosiatif, TTL, consolidation, atau garbage collection otomatis. Penghapusan logis melalui `delete`; retensi media ditangani aplikasi/host. Hindari berbagi path yang sama antara beberapa writer.
