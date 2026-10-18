# Model matematis LIF, jaringan, dan pembelajaran

## Neuron LIF

Untuk satu interval dengan arus konstan, neuron mengikuti

```text
τₘ dV/dt = −(V − V_rest) + R I
```

Integrasi analitis selama satu langkah `dt` menghasilkan

```text
V_next = V_rest
       + (V − V_rest) exp(−dt/τₘ)
       + R I (1 − exp(−dt/τₘ))
```

`V` dan potensial lain memakai milivolt, `dt` dan `τₘ` memakai milidetik, `R` memakai megaohm, dan `I` memakai nanoampere. Parameter `dt`, `τₘ`, dan `R` harus positif dan finite; rest/reset harus finite dan di bawah ambang.

Setiap neuron pada jaringan boleh memiliki `τₘ`, `R`, rest/reset, ambang, dan refraktori sendiri. Semua neuron harus berbagi `dt` yang sama supaya indeks waktu dan delay sinkron. Jika `V_next >= threshold`, spike dicatat pada indeks langkah selesai basis satu dan potensial di-reset. Tidak ada interpolasi waktu ambang. `1 − exp(−x)` dihitung sebagai `−expm1(−x)`.

## Sinaps dan propagasi

Sinaps menyimpan bobot arus signed `w` dalam nA, delay integer `d >= 1`, dan batas inklusif `[w_min,w_max]`. `w > 0` eksitatorik; `w < 0` inhibitorik. Spike pada langkah selesai `k` menjadwalkan event pada awal langkah `k+d`; event menjadi arus satu interval.

## STDP lokal dan reward

Dengan `Δt = t_post − t_pre` dalam langkah:

```text
Δw = A_plus  exp(−Δt / τ_plus)       bila Δt > 0
Δw = −A_minus exp( Δt / τ_minus)     bila Δt < 0
Δw = 0                               bila Δt = 0
```

Pada mode tiga faktor, delta STDP lokal masuk ke eligibility, bukan langsung ke bobot:

```text
e_t = e_(t−1) exp(−1 / τ_e) + Δw_STDP,t
w ← clamp(w + η r e_t, w_min, w_max) saat reward r diterima
```

`RewardEvaluator` memetakan target `a*` dan prediksi `â` (kanal spike aktif berindeks terendah, atau `None` untuk diam) menjadi

```text
r = +R_correct       bila â = a*
r = −P_incorrect     bila â ≠ a*
```

`R_correct` dan `P_incorrect` positif finite. `evaluate_and_apply` menerapkan `r` ke eligibility jaringan dan mencatat metrik hanya setelah update berhasil. Akurasi adalah `jumlah target benar / jumlah sampel`.

## Protokol benchmark

Benchmark conditioning memiliki satu neuron output dan dua neuron cue. `cue mendekat → spike` dipasangkan dengan teacher spike saat pelatihan; `cue menahan → diam` tidak diberi teacher spike. Kontrol menerima urutan input sama tanpa memanggil update reward. Setelah washout 50 langkah, kedua jaringan diuji tanpa teacher; setiap pola diulang sebanyak `evaluation_repetitions`. Akurasi uji adalah recall pada pola sama, bukan generalisasi pada cue yang tidak dilatih.
