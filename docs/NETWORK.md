# Kontrak jaringan LIF

`LifNetwork::new(config, neuron_count)` membuat jaringan homogen dengan sekurang-kurangnya satu neuron. `LifNetwork::from_configs(&configs)` membuat neuron dengan parameter membran berbeda per indeks. Setiap konfigurasi divalidasi dan `dt_ms` harus identik agar satu langkah jaringan, delay sinaps, serta waktu STDP memiliki acuan bersama. Konstanta waktu, resistansi, potensial rest/reset, ambang, dan periode refraktori dapat berbeda.

Arus eksternal pada setiap langkah harus memiliki tepat satu elemen finite per neuron. `Synapse::new(source, target, weight, min_weight, max_weight, delay)` memvalidasi bobot finite dan berada dalam batas inklusif serta delay positif. `add_synapse` memvalidasi endpoint dan menolak pasangan berarah duplikat. Self-connection diperbolehkan. Bobot positif eksitatorik, bobot negatif inhibitorik; pengguna menentukan rentang untuk mempertahankan kelas tanda bila diperlukan.

Semua neuron melangkah sinkron dari keadaan awal. Spike langkah `k` menjadwalkan arus pada langkah `k+d`, dengan `d >= 1`; event dijumlahkan dengan input eksternal dan dikonsumsi setelah satu interval. Urutan penjumlahan mengikuti urutan penambahan sinaps.

`step` menghitung neuron, event, dan update learning sebelum commit. Input, overflow, atau kegagalan alokasi yang terdeteksi membatalkan langkah jaringan secara logis. Dalam mode reward STDP, `apply_reward(r)` merupakan operasi terpisah dari waktu jaringan: ia memodulasi trace eligibility yang tersedia, menjepit bobot ke batas, dan tidak menghapus eligibility.
