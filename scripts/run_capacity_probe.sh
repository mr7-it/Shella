#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
cargo build --release --locked --example capacity_probe >&2
probe="$ROOT/target/release/examples/capacity_probe"
steps_per_batch="${STEPS_PER_BATCH:-20}"

# Setiap ukuran dijalankan dalam proses baru sehingga VmHWM dapat dibandingkan.
# Maksimum jaringan pada grid ini adalah 16,384 neuron dan 65,536 sinaps.
printf '%s\n' '{"benchmark":"shella_capacity_probe","format":"one JSON object per workload"}'
for pair in \
  '64 64' \
  '256 1024' \
  '1024 4096' \
  '4096 16384' \
  '8192 32768' \
  '16384 65536'
do
  read -r neurons synapses <<<"$pair"
  "$probe" "$neurons" "$synapses" "$steps_per_batch"
done
