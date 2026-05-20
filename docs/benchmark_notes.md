# Benchmark Notes

## 2026-05-19: Single Relay Pooling Smoke Result

Environment summary from the first server run:

- 8 NVIDIA GPUs visible
- GPU0 was unavailable for the direct H2D test because `cudaSetDevice(0)`
  returned out of memory
- CUDA P2P access was only available between GPU5 and GPU6
- Tested pair:
  - target GPU: 6
  - relay GPU: 5

CUDA P2P matrix summary:

```text
row = relay/source, col = target/destination

GPU5 -> GPU6: enabled
GPU6 -> GPU5: enabled
all other tested off-diagonal pairs: disabled
```

Correctness checks passed:

```text
TURBOBUS_TARGET_GPU=6 TURBOBUS_TEST_BYTES=16777216 ./build-test/test_direct_h2d
TURBOBUS_TARGET_GPU=6 TURBOBUS_RELAY_GPU=5 TURBOBUS_TEST_BYTES=33554432 ./build-test/test_relay_h2d_p2p
```

Profiler result:

```text
TURBOBUS_TARGET_GPU=6 TURBOBUS_RELAY_GPU=5 TURBOBUS_PROFILE_BYTES=16777216 ./build-test/test_profiler

direct_h2d_bw_gbps=7.63333
relay=5 h2d=7.56767 p2p=6.76112 effective=6.76112 p2p_enabled=1
```

Pooling benchmark:

```text
TURBOBUS_TARGET_GPU=6 TURBOBUS_RELAY_GPU=5 TURBOBUS_BENCH_BYTES=268435456 ./build-test/bench_pool_bandwidth

target_gpu=6
relay_gpu=5
bytes=268435456
chunk_bytes=16777216
direct_milliseconds=34.164
direct_gib_per_second=7.31764
relay_milliseconds=34.465
relay_gib_per_second=7.25374
pool_milliseconds=17.472
pool_gib_per_second=14.3086
pool_over_direct=1.95536
pool_over_relay=1.97258
```

Conclusion:

The single-relay MVP validates the core TurboBus mechanism on this machine:

```text
CPU pinned memory -> target GPU
CPU pinned memory -> relay GPU -> target GPU
direct + relay chunk-level concurrent transfer
```

The pooled path reached 14.31 GiB/s, about 1.96x faster than direct-only H2D.

## 2026-05-19: Planner-Driven Multi-Run Pooling Result

This run validates the formal path:

```text
profiler -> ChunkPlanner -> CUDA executor
```

The benchmark used repeated runs and median reporting:

```text
TURBOBUS_TARGET_GPU=6 TURBOBUS_RELAY_GPUS=5 TURBOBUS_BENCH_ITERS=5 ./build-test/bench_pool_bandwidth
```

Run 1:

```text
target_gpu=6
relay_gpus=5,
bytes=268435456
chunk_bytes=16777216
iterations=5
direct_median_gib_per_second=7.3215
relay_median_gib_per_second=7.24071
pool_median_gib_per_second=14.3062
pool_over_direct_median=1.95399
pool_over_relay_median=1.97579
```

Run 2:

```text
target_gpu=6
relay_gpus=5,
bytes=268435456
chunk_bytes=16777216
iterations=5
direct_median_gib_per_second=7.32365
relay_median_gib_per_second=7.24344
pool_median_gib_per_second=14.2759
pool_over_direct_median=1.94929
pool_over_relay_median=1.97088
```

Conclusion:

The planner-driven pool transfer is stable on the tested GPU6 target + GPU5
relay pair. Across two 5-iteration runs, direct-only H2D stayed around
7.32 GiB/s, relay-only stayed around 7.24 GiB/s, and pooled transfer stayed
around 14.28-14.31 GiB/s. This is about 1.95x faster than direct-only H2D.

## 2026-05-19: Python API End-to-End Benchmark Result

The Python extension and PyTorch-facing API were validated after fixing the
native extension CUDA link issue.

Native extension import check:

```text
python - <<'PY'
import turbobus._turbobus as tb
print(tb)
print(tb.RuntimeOptions)
PY
```

Result:

```text
<module 'turbobus._turbobus' from '/home/sdu/fengbin/TurboBus/turbobus/_turbobus.cpython-313-x86_64-linux-gnu.so'>
<class 'turbobus._turbobus.RuntimeOptions'>
```

Short Python pooling benchmark:

```text
python benchmarks/bandwidth_pool.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --bytes 268435456 \
  --chunk-bytes 16777216 \
  --profile-bytes 16777216 \
  --warmup 1 \
  --iterations 5 \
  --verify
```

Result:

```text
direct_h2d_bw_gbps 7.633091670011347
relay 5 h2d 7.573304563650957 p2p 39.93140749818256 effective 7.573304563650957
median_gib_per_second 14.400921658986174
direct_chunks 8
relay_chunks 8
match True
```

20-iteration Python stability run:

```text
python benchmarks/bandwidth_pool.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --bytes 268435456 \
  --chunk-bytes 16777216 \
  --profile-bytes 16777216 \
  --warmup 3 \
  --iterations 20 \
  --verify
```

Result:

```text
direct_h2d_bw_gbps 7.433000521612081
relay 5 h2d 7.536251016403788 p2p 39.24143954936054 effective 7.536251016403788
median_gib_per_second 13.650014301100978
direct_chunks 8
relay_chunks 8
match True
```

1 GiB Python transfer run:

```text
python benchmarks/bandwidth_pool.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --bytes 1073741824 \
  --chunk-bytes 16777216 \
  --profile-bytes 16777216 \
  --warmup 2 \
  --iterations 10 \
  --verify
```

Result:

```text
direct_h2d_bw_gbps 7.543586322875762
relay 5 h2d 7.52428981879867 p2p 40.966628272433766 effective 7.52428981879867
median_gib_per_second 14.624586933618467
direct_chunks 32
relay_chunks 32
match True
```

100-iteration Python stability run:

```text
python benchmarks/bandwidth_pool.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --bytes 268435456 \
  --chunk-bytes 16777216 \
  --profile-bytes 16777216 \
  --warmup 5 \
  --iterations 100
```

Result:

```text
direct_h2d_bw_gbps 7.565325330576065
relay 5 h2d 7.560288734366875 p2p 40.85010046409519 effective 7.560288734366875
median_gib_per_second 14.42169022209403
direct_chunks 8
relay_chunks 8
```

Conclusion:

The Python API drives the same planner-backed pooled transfer path as the C++
benchmark. Correctness passed with `match True`, all measured Python pooling
runs used both direct and relay chunks, and the 100-iteration run stayed around
14.42 GiB/s median without transfer errors.

## 2026-05-19: Python Direct/Relay/Pool Mode Comparison

This run validates the Python benchmark's explicit transfer modes and CUDA-event
based timing stats.

Command:

```text
python benchmarks/bandwidth_pool.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --bytes 268435456 \
  --chunk-bytes 16777216 \
  --profile-bytes 16777216 \
  --warmup 1 \
  --iterations 5 \
  --mode all \
  --verify
```

Profile result:

```text
direct_h2d_bw_gbps 7.499789148536587
relay 5 h2d 7.468015139322194 p2p 40.792083213708565 effective 7.468015139322194
```

Median transfer results:

```text
mode direct median_gib_per_second 7.349199332453709
mode relay median_gib_per_second 7.2530436365708955
mode pool median_gib_per_second 14.358309897048315
pool_over_direct_median 1.953724378333666
pool_over_relay_median 1.979625467114473
match True
```

Chunk assignment checks:

```text
direct mode: direct_chunks 16, relay_chunks 0
relay mode:  direct_chunks 0, relay_chunks 16
pool mode:   direct_chunks 8, relay_chunks 8
```

Conclusion:

The explicit Python direct/relay/pool modes match the expected planner behavior.
The pooled path remained about 1.95x faster than direct-only H2D and about 1.98x
faster than relay-only transfer on the tested GPU6 target + GPU5 relay pair.

## 2026-05-19: JSON Benchmark Output and Tuner Result

This run validates `--json-output` for the Python benchmark and the first
chunk/staging sweep tuner.

JSON benchmark command:

```text
python benchmarks/bandwidth_pool.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --bytes 268435456 \
  --chunk-bytes 16777216 \
  --profile-bytes 16777216 \
  --warmup 1 \
  --iterations 5 \
  --mode all \
  --verify \
  --json-output benchmarks/results/gpu6_relay5.json
```

Result:

```text
direct_h2d_bw_gbps 7.518844436581721
relay 5 h2d 7.533576697043448 p2p 39.88248432714853 effective 7.533576697043448
mode direct median_gib_per_second 7.333724996503007
mode relay median_gib_per_second 6.841889157344973
mode pool median_gib_per_second 13.53812529727491
pool_over_direct_median 1.8460094022792497
pool_over_relay_median 1.9787115789125767
match True
json_output benchmarks/results/gpu6_relay5.json
```

Tuner command:

```text
python benchmarks/tune_transfer.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --bytes 268435456 \
  --profile-bytes 16777216 \
  --chunk-mib 4,8,16,32,64 \
  --staging-slots 2,3,4 \
  --warmup 1 \
  --iterations 5 \
  --json-output benchmarks/results/tune_gpu6_relay5.json
```

Sweep result:

```text
4 MiB, slots 2: 14.473221746095176 GiB/s
4 MiB, slots 3: 14.482986700540451 GiB/s
4 MiB, slots 4: 14.459291213205049 GiB/s
8 MiB, slots 2: 14.45057294815083 GiB/s
8 MiB, slots 3: 14.45271128925105 GiB/s
8 MiB, slots 4: 14.446404843213655 GiB/s
16 MiB, slots 2: 14.34523384771235 GiB/s
16 MiB, slots 3: 14.339229463335695 GiB/s
16 MiB, slots 4: 14.342309495685173 GiB/s
32 MiB, slots 2: 14.105244806794348 GiB/s
32 MiB, slots 3: 14.103971379715793 GiB/s
32 MiB, slots 4: 14.094633367693877 GiB/s
64 MiB, slots 2: 13.6278875121193 GiB/s
64 MiB, slots 3: 13.628458557204235 GiB/s
64 MiB, slots 4: 13.61617818710527 GiB/s
```

Best candidate:

```text
chunk_bytes 4194304
chunk_size 4 MiB
staging_slots 3
median_gib_per_second 14.482986700540451
json_output benchmarks/results/tune_gpu6_relay5.json
```

Conclusion:

The tuner found `4 MiB` chunks with `3` staging slots as the best candidate in
this sweep. The result suggests smaller chunks improve pooling on this GPU6
target + GPU5 relay pair by exposing more chunk-level parallelism. The default
chunk size remains `16 MiB` for now until repeated sweeps confirm that `4 MiB`
is consistently better under different machine load.

## 2026-05-20: 4 MiB Trace Benchmark Result

This run validates the expanded JSON trace output: direct/relay byte counts,
per-relay stats, and `last_plan` chunk assignments.

Command:

```text
python benchmarks/bandwidth_pool.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --bytes 268435456 \
  --chunk-bytes 4194304 \
  --profile-bytes 16777216 \
  --warmup 1 \
  --iterations 5 \
  --mode all \
  --verify \
  --json-output benchmarks/results/gpu6_relay5_trace.json
```

Profile result:

```text
direct_h2d_bw_gbps 7.563802560469968
relay 5 h2d 7.595924979113988 p2p 40.9563203305953 effective 7.595924979113988
```

Median transfer results:

```text
mode direct median_gib_per_second 7.317287762256032
mode relay median_gib_per_second 7.275725990581611
mode pool median_gib_per_second 14.450813519766232
pool_over_direct_median 1.9748865958649717
pool_over_relay_median 1.9861679148545086
match True
```

Trace checks:

```text
pool direct_bytes 134217728
pool relay_bytes 134217728
pool direct_chunks 32
pool relay_chunks 32
relay 5 bytes 134217728
relay 5 chunks 32
last_plan direct assignment: 32 chunks
last_plan relay5 assignment: 32 chunks
```

Conclusion:

The `4 MiB` trace run confirms that the JSON output records enough information
to explain planner behavior. The pooled path split the 256 MiB transfer evenly
between direct H2D and relay GPU5, and reached about 1.97x direct-only bandwidth.

## 2026-05-20: Per-Path Timing Stats Benchmark Result

This run validates `TransferStats.path_stats` in direct, relay-only, and pooled
transfer modes. The benchmark command was:

```text
python benchmarks/bandwidth_pool.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --bytes 268435456 \
  --chunk-bytes 4194304 \
  --profile-bytes 16777216 \
  --warmup 1 \
  --iterations 5 \
  --mode all \
  --verify \
  --json-output benchmarks/results/gpu6_relay5_path_stats.json
```

Profile result:

```text
direct_h2d_bw_gbps 7.5371818856916715
relay 5 h2d 7.492768305553483 p2p 38.33565487615392 effective 7.492768305553483
```

Median transfer results:

```text
mode direct median_gib_per_second 7.274012365465845
mode relay median_gib_per_second 7.224097373636718
mode pool median_gib_per_second 14.298448030619832
pool_over_direct_median 1.9656892664223737
pool_over_relay_median 1.9792712211770453
match True
```

Representative pooled `path_stats`:

```text
direct path bytes 134217728 chunks 32 cuda_elapsed_ms 17.16-17.22 gib_per_second 7.26-7.28
relay5 path bytes 134217728 chunks 32 cuda_elapsed_ms 17.30-17.36 gib_per_second 7.20-7.22
```

Conclusion:

The per-path stats are exposed through Python and included in benchmark JSON.
For the pooled transfer, total elapsed time tracks the slower relay path, which
is the signal needed for a future dynamic weighting pass.

## Next Implementation Steps

1. Replace the benchmark-only even/odd chunk split with the production
   `ChunkPlanner` bandwidth-proportional assignment. Done after the initial
   smoke result.
2. Add a repeated benchmark mode and report median bandwidth. Done after the
   initial smoke result.
3. Support multiple relay GPUs in the benchmark and executor validation path.
   The benchmark now accepts `TURBOBUS_RELAY_GPUS=5,6` and filters by CUDA P2P
   capability.
4. Add a topology-aware relay picker that filters by CUDA P2P capability and
   measured effective bandwidth. The current runtime scans P2P-capable relays
   when no relay list is provided; the benchmark profiles the enabled relay list
   and uses `ChunkPlanner`.
5. Add Python extension build packaging so `benchmarks/bandwidth_pool.py` can
   run without manual CMake module handling. A minimal `pip install -e .`
   CMake-backed package build has been added.
6. Add profiler result caching so runtime does not profile on every first
   transfer. The runtime now caches profile results.
7. Add basic transfer metrics to `TransferHandle` or a returned stats object.
   `TransferStats` now records bytes, submit-to-complete time, effective GiB/s,
   and direct/relay chunk counts.
