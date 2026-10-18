# Determinisme

Runtime memakai `f64`, tidak menggunakan RNG atau thread, dan menjalankan operasi serial. Konfigurasi membran dapat berbeda per neuron, tetapi semua neuron di satu jaringan wajib memiliki `dt_ms` identik. Arus eksternal diindeks menurut neuron; event dijumlahkan menurut urutan penambahan sinaps. Transisi semua neuron dihitung sinkron dari keadaan awal langkah.

Spike pada langkah `k` dengan delay `d` tiba pada awal langkah `k+d`. Event memakai snapshot bobot ketika dijadwalkan. STDP lokal mengubah bobot pada commit langkah; reward-modulated STDP membangun atau meluruhkan eligibility pada langkah simulasi. `apply_reward(r)` memproses sinaps secara serial tanpa menaikkan indeks langkah dan menerapkan hasil hanya setelah semua perhitungan finite berhasil.

`RewardEvaluator` memprediksi kanal aktif berindeks paling rendah; jika tak ada kanal aktif, prediksi adalah `None`. Benchmark memakai urutan dua pola tetap, parameter tetap, dan tidak menggunakan RNG. Karena itu akurasi hasil dapat diulang persis; durasi eksekusi tentu bergantung pada mesin dan beban sistem.

Kegagalan persiapan langkah tidak mengubah keadaan simulasi logis; kegagalan `apply_reward` tidak menerapkan update parsial. Kegagalan `evaluate_and_apply` juga tidak mencatat metrik. Kapasitas internal vektor dapat berubah selama reservasi memori dan tidak termasuk state model. Tidak ada jaminan bit-identik lintas OS, compiler, atau pustaka matematika.
