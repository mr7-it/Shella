# STDP lokal dan modulasi reward

## Aturan STDP pair-based

`StdpConfig` memuat `A_plus`, `A_minus`, `tau_plus_steps`, dan `tau_minus_steps`; semuanya harus finite dan positif. Konvensi waktu `Δt = t_post − t_pre` memakai indeks langkah jaringan:

```text
Δw =  A_plus  exp(−Δt / tau_plus_steps)   jika Δt > 0
Δw = −A_minus exp( Δt / tau_minus_steps)  jika Δt < 0
Δw = 0                                    jika Δt = 0
```

Trace pra/pasca meluruh tiap langkah dan memuat kontribusi semua spike terdahulu. Trace lama di-decay sebelum pembaruan; spike sekarang ditambahkan setelah kalkulasi agar pasangan serentak tidak menyumbang update. Mode `enable_stdp` langsung menerapkan delta lokal dan membatasi bobot ke rentang inklusif sinaps.

## Reward-modulated STDP

`RewardModulatedStdpConfig` menggabungkan aturan STDP lokal, `eligibility_tau_steps`, dan `learning_rate`; waktu serta laju harus finite dan positif. Mode ini menerapkan aturan tiga faktor: aktivitas pra/pascasinaptik membentuk jejak eligibility `e`, reward eksternal bertanda `r` mengubah bobot, dan eligibility meluruh:

```text
e[t] = e[t−1] exp(−1 / tau_eligibility) + Δw_STDP[t]
Δw_reward = learning_rate × r × e[t]
w ← clamp(w + Δw_reward, w_min, w_max)
```

Dalam mode ini pasangan spike tidak langsung mengubah bobot; ia menambah atau mengurangi eligibility melalui STDP lokal. Pemanggil menyampaikan reward dengan `LifNetwork::apply_reward(r)`, termasuk reward tertunda setelah simulasi bergerak beberapa langkah. Reward tidak memajukan waktu jaringan dan eligibility tidak dihapus setelah reward, sehingga reward berikutnya dapat tetap memodulasi trace yang tersisa; peluruhan terjadi pada langkah jaringan.

Reward positif memperkuat eligibility positif dan melemahkan eligibility negatif; reward negatif membalikkan efek itu. Reward non-finite ditolak. Perhitungan semua bobot disiapkan sebelum commit, sehingga overflow pada satu sinaps tidak menerapkan update parsial.

Reward dapat dihitung oleh `RewardEvaluator` dari kecocokan kanal spike dengan target, lalu diterapkan melalui `evaluate_and_apply`. Benchmark terukur ada untuk satu tugas conditioning yang dijelaskan di `docs/EVALUATOR_BENCHMARK.md`; hasil itu tidak membuktikan generalisasi atau kinerja pada tugas eksternal.
