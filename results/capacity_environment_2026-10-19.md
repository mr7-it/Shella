# Lingkungan pengukuran capacity probe

- Eksekusi: Manus Sandbox di VM Linux amd64/KVM (virtualisasi penuh), bukan perangkat fisik pengguna.
- Daftar perangkat yang dapat dipilih pada sesi hanya berisi Sandbox; tidak ada komputer desktop/Cloud Computer pengguna yang tersedia untuk dialihkan.
- CPU yang terlihat: 6 vCPU; `/proc/cpuinfo` mengidentifikasi Intel Xeon Processor @ 2.50 GHz. Utilisasi sampel cgroup 2 detik saat pemeriksaan 0,07% (sebelum benchmark, bukan utilisasi sepanjang benchmark).
- RAM terlihat oleh `/proc/meminfo` saat pemeriksaan: `MemTotal` 14,387,492 kB dan `MemAvailable` 12,455,436 kB. Telemetri RAM dari provider tidak tersedia dan batas cgroup tidak terlihat, jadi angka ini merupakan snapshot guest, bukan jaminan kuota.
- Filesystem `/home/ubuntu`: 49 GB total, sekitar 37 GB tersedia saat pemeriksaan.
- Toolchain benchmark: rustc/cargo stable 1.99.1; target optimized Release.
- Waktu pengambilan spesifikasi: 19 Oktober 2026 sekitar 02:27 WIB. Grid capacity dijalankan sekitar 02:29 WIB.

Setiap ukuran diproses dalam child executable baru untuk memisahkan `VmHWM`. Tiap proses membangun jaringan berarah deterministik, mengukur waktu build, lima batch × 20 `LifNetwork::step` (median waktu per step), menyimpan checkpoint dan menghitung ukurannya, lalu membaca `VmRSS`/`VmHWM` Linux. Semua neuron mendapat arus konstan 100 nA pada step benchmark; synapse delay 1. Pengukuran RSS mencakup runtime proses, struktur jaringan, event aktif, buffer checkpoint, dan overhead allocator; bukan pengukuran RAM per objek/working set aplikasi.

Grid tertinggi yang diuji: 16,384 neuron dan 65,536 sinaps unik tanpa self-edge. Tidak dilakukan pencarian OOM/kapasitas maksimum. Waktu dalam satuan benchmark sandbox bersifat host-load-sensitive dan tidak dapat dipakai sebagai janji throughput atau proyeksi mesin lain.
