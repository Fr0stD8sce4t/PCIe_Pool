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

## 2026-05-20: Dynamic Weights and D2H Smoke Result

This run validates that dynamic H2D weights do not break the pooled benchmark
and that the new D2H offload path can round-trip data through direct + relay
paths.

H2D dynamic-weights benchmark:

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
  --json-output benchmarks/results/gpu6_relay5_dynamic_weights.json \
  --dynamic-weights
```

Median transfer results:

```text
mode direct median_gib_per_second 7.304883937325309
mode relay median_gib_per_second 7.272705481728514
mode pool median_gib_per_second 14.437940659173856
pool_over_direct_median 1.976477762418813
pool_over_relay_median 1.9852227888846627
match True
```

Representative pooled `path_stats` from the dynamic run:

```text
direct path bytes 134217728 chunks 32 gib_per_second 7.29-7.31
relay5 path bytes 134217728 chunks 32 gib_per_second 7.25-7.26
```

The final pooled plan still split the transfer evenly at this data size, but
the planner profile values moved from the initial profile toward observed
path-local bandwidths:

```text
direct effective_bw_gbps 7.312435897427332
relay5 effective_bw_gbps 7.273332880031879
```

D2H round-trip smoke result:

```text
match True
stats 15.335706683081288 direct_chunks 8 relay_chunks 8
path direct d2h relay -1 bytes 33554432 chunks 8 gib_per_second 7.911232341990831
path relay d2h relay 5 bytes 33554432 chunks 8 gib_per_second 7.733433494135791
```

Conclusion:

The dynamic-weight path is functional and conservative for this target/relay
pair. The D2H offload API correctly copies data back to pinned CPU memory and
uses both direct and relay paths in pooled mode.

## KV Block Offload Benchmark

Command shape:

```text
python benchmarks/kv_offload.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --num-blocks 8 \
  --active-blocks 4 \
  --block-bytes 16777216 \
  --chunk-bytes 4194304 \
  --profile-bytes 16777216 \
  --warmup 1 \
  --iterations 5 \
  --mode all \
  --verify \
  --dynamic-weights \
  --json-output benchmarks/results/kv_gpu6_relay5.json \
  --summary-output benchmarks/results/kv_gpu6_relay5_summary.txt
```

Copy summary:

```text
COPY_SUMMARY_BEGIN
kv_config target=6 relays=[5] num_blocks=8 active_blocks=4 block_bytes=16777216 chunk_bytes=4194304 iterations=5 mode=all dynamic_weights=True
profile direct_h2d_bw_gbps=7.556
profile_relay relay=5 h2d=7.577 p2p=39.736 effective=7.577 p2p_enabled=True
kv_op mode=direct op=prefetch count=20 batch_gib_s=7.350 batch_p50_ms=8.432 batch_p95_ms=8.636 block_gib_s=3.008 block_p50_ms=5.196 block_p95_ms=8.392 direct_chunks=80 relay_chunks=0 direct_bytes=335544320 relay_bytes=0
kv_path mode=direct op=prefetch direction=h2d kind=direct relay=-1 median_gib_s=7.499 median_ms=2.084 bytes=335544320 chunks=80
kv_op mode=direct op=evict count=20 batch_gib_s=7.900 batch_p50_ms=7.913 batch_p95_ms=7.918 block_gib_s=3.249 block_p50_ms=4.803 block_p95_ms=7.697 direct_chunks=80 relay_chunks=0 direct_bytes=335544320 relay_bytes=0
kv_path mode=direct op=evict direction=d2h kind=direct relay=-1 median_gib_s=8.008 median_ms=1.951 bytes=335544320 chunks=80
kv_verify mode=direct match=True
kv_op mode=relay op=prefetch count=20 batch_gib_s=7.288 batch_p50_ms=8.558 batch_p95_ms=8.673 block_gib_s=3.002 block_p50_ms=5.236 block_p95_ms=8.222 direct_chunks=0 relay_chunks=80 direct_bytes=0 relay_bytes=335544320
kv_path mode=relay op=prefetch direction=h2d kind=relay relay=5 median_gib_s=7.182 median_ms=2.176 bytes=335544320 chunks=80
kv_op mode=relay op=evict count=20 batch_gib_s=7.789 batch_p50_ms=8.035 batch_p95_ms=8.063 block_gib_s=3.232 block_p50_ms=4.815 block_p95_ms=7.697 direct_chunks=0 relay_chunks=80 direct_bytes=0 relay_bytes=335544320
kv_path mode=relay op=evict direction=d2h kind=relay relay=5 median_gib_s=5.497 median_ms=2.842 bytes=335544320 chunks=80
kv_verify mode=relay match=True
kv_op mode=pool op=prefetch count=20 batch_gib_s=14.116 batch_p50_ms=4.422 batch_p95_ms=4.473 block_gib_s=5.986 block_p50_ms=2.633 block_p95_ms=4.061 direct_chunks=40 relay_chunks=40 direct_bytes=167772160 relay_bytes=167772160
kv_path mode=pool op=prefetch direction=h2d kind=direct relay=-1 median_gib_s=7.517 median_ms=1.039 bytes=167772160 chunks=40
kv_path mode=pool op=prefetch direction=h2d kind=relay relay=5 median_gib_s=6.946 median_ms=1.125 bytes=167772160 chunks=40
kv_op mode=pool op=evict count=20 batch_gib_s=14.882 batch_p50_ms=4.196 batch_p95_ms=4.221 block_gib_s=6.362 block_p50_ms=2.442 block_p95_ms=3.887 direct_chunks=40 relay_chunks=40 direct_bytes=167772160 relay_bytes=167772160
kv_path mode=pool op=evict direction=d2h kind=direct relay=-1 median_gib_s=7.894 median_ms=0.990 bytes=167772160 chunks=40
kv_path mode=pool op=evict direction=d2h kind=relay relay=5 median_gib_s=4.121 median_ms=1.896 bytes=167772160 chunks=40
kv_verify mode=pool match=True
kv_speedup pool_over_direct_prefetch=1.921
kv_speedup pool_over_relay_prefetch=1.937
kv_speedup pool_over_direct_evict=1.884
kv_speedup pool_over_relay_evict=1.911
COPY_SUMMARY_END
```

Analysis:

The connector-ready KV offload benchmark validates the `OffloadStore`
`prefetch_many` and `evict_many` paths. The primary metric is `batch_gib_s`,
which measures a decode-step-style batch of active KV blocks from submission to
all blocks complete. Pooled transfer splits the work evenly between direct and
relay paths and improves batch throughput from 7.350 to 14.116 GiB/s for
prefetch, and from 7.900 to 14.882 GiB/s for evict. This corresponds to 1.921x
prefetch speedup and 1.884x evict speedup over direct-only transfer. All direct,
relay, and pooled verification checks passed. The lower `block_gib_s` values
are retained as a per-block submit-to-complete view and include queueing time
when several blocks are submitted together.

## KV Block Offload Packed Storage

Scenario:

This run uses `benchmarks/kv_offload.py` with `--storage-layout packed`. The KV
blocks share one pinned CPU backing tensor and one target-GPU backing tensor,
with each block addressed by offset and byte count. This exercises the
range-batched `OffloadStore.prefetch_many` and `evict_many` path without the
full decode-step simulator.

Copy summary:

```text
COPY_SUMMARY_BEGIN
kv_config target=6 relays=[5] num_blocks=8 active_blocks=4 storage_layout=packed block_bytes=16777216 chunk_bytes=4194304 iterations=5 mode=all dynamic_weights=True
profile direct_h2d_bw_gbps=7.611
profile_relay relay=5 h2d=7.540 p2p=39.502 effective=7.540 p2p_enabled=True
kv_op mode=direct op=prefetch count=20 batch_gib_s=7.261 batch_p50_ms=8.562 batch_p95_ms=8.787 block_gib_s=1.871 block_p50_ms=8.319 block_p95_ms=8.508 direct_chunks=80 relay_chunks=0 direct_bytes=335544320 relay_bytes=0
kv_path mode=direct op=prefetch direction=h2d kind=direct relay=-1 median_gib_s=7.466 median_ms=8.371 bytes=335544320 chunks=80
kv_op mode=direct op=evict count=20 batch_gib_s=7.782 batch_p50_ms=8.018 batch_p95_ms=8.068 block_gib_s=2.007 block_p50_ms=7.781 block_p95_ms=7.791 direct_chunks=80 relay_chunks=0 direct_bytes=335544320 relay_bytes=0
kv_path mode=direct op=evict direction=d2h kind=direct relay=-1 median_gib_s=7.981 median_ms=7.831 bytes=335544320 chunks=80
kv_verify mode=direct match=True
kv_op mode=relay op=prefetch count=20 batch_gib_s=7.209 batch_p50_ms=8.643 batch_p95_ms=8.793 block_gib_s=1.868 block_p50_ms=8.320 block_p95_ms=8.549 direct_chunks=0 relay_chunks=80 direct_bytes=0 relay_bytes=335544320
kv_path mode=relay op=prefetch direction=h2d kind=relay relay=5 median_gib_s=7.395 median_ms=8.452 bytes=335544320 chunks=80
kv_op mode=relay op=evict count=20 batch_gib_s=7.711 batch_p50_ms=8.108 batch_p95_ms=8.121 block_gib_s=2.007 block_p50_ms=7.784 block_p95_ms=7.804 direct_chunks=0 relay_chunks=80 direct_bytes=0 relay_bytes=335544320
kv_path mode=relay op=evict direction=d2h kind=relay relay=5 median_gib_s=7.883 median_ms=7.928 bytes=335544320 chunks=80
kv_verify mode=relay match=True
kv_op mode=pool op=prefetch count=20 batch_gib_s=13.813 batch_p50_ms=4.521 batch_p95_ms=4.535 block_gib_s=3.703 block_p50_ms=4.199 block_p95_ms=4.322 direct_chunks=40 relay_chunks=40 direct_bytes=167772160 relay_bytes=167772160
kv_path mode=pool op=prefetch direction=h2d kind=direct relay=-1 median_gib_s=7.490 median_ms=4.172 bytes=167772160 chunks=40
kv_path mode=pool op=prefetch direction=h2d kind=relay relay=5 median_gib_s=7.321 median_ms=4.268 bytes=167772160 chunks=40
kv_op mode=pool op=evict count=20 batch_gib_s=14.648 batch_p50_ms=4.268 batch_p95_ms=4.274 block_gib_s=3.953 block_p50_ms=3.956 block_p95_ms=3.979 direct_chunks=40 relay_chunks=40 direct_bytes=167772160 relay_bytes=167772160
kv_path mode=pool op=evict direction=d2h kind=direct relay=-1 median_gib_s=7.943 median_ms=3.934 bytes=167772160 chunks=40
kv_path mode=pool op=evict direction=d2h kind=relay relay=5 median_gib_s=7.770 median_ms=4.022 bytes=167772160 chunks=40
kv_verify mode=pool match=True
kv_speedup pool_over_direct_prefetch=1.902
kv_speedup pool_over_relay_prefetch=1.916
kv_speedup pool_over_direct_evict=1.882
kv_speedup pool_over_relay_evict=1.900
COPY_SUMMARY_END
```

Analysis:

Packed KV offload validates the shared backing-buffer path used by future
connector-style KV stores. The primary `batch_gib_s` metric stays in the
expected range: direct prefetch is 7.261 GiB/s, pooled prefetch is 13.813 GiB/s,
direct evict is 7.782 GiB/s, and pooled evict is 14.648 GiB/s. The pool/direct
speedups are 1.902x for prefetch and 1.882x for evict. The chunk counts also
match the intended physical transfer work: direct uses 80 chunks, relay uses 80
chunks, and pooled mode splits the same work into 40 direct and 40 relay chunks.
All verification checks passed.

In packed mode, `block_gib_s` is a per-block view of a shared batch transfer, so
it is lower than `batch_gib_s` and should not be used as the main bandwidth
metric. Use `batch_gib_s` and the path-level chunk counts for packed-storage
results.

## Inference Offload Simulator Capacity Pressure

Command shape:

```text
python benchmarks/inference_offload_sim.py \
  --target-gpu 6 \
  --relay-gpus 5 \
  --requests 4 \
  --blocks-per-request 8 \
  --blocks-per-step 4 \
  --gpu-block-capacity 4 \
  --access-pattern round_robin \
  --working-set-blocks 8 \
  --seed 1 \
  --block-bytes 16777216 \
  --decode-steps 32 \
  --chunk-bytes 4194304 \
  --profile-bytes 16777216 \
  --mode all \
  --dynamic-weights \
  --json-output benchmarks/results/infer_sim_gpu6_relay5_pressure.json \
  --summary-output benchmarks/results/infer_sim_gpu6_relay5_pressure_summary.txt
```

Round-robin copy summary:

```text
COPY_SUMMARY_BEGIN
sim_config target=6 relays=[5] requests=4 blocks_per_request=8 blocks_per_step=4 gpu_block_capacity=4 access_pattern=round_robin working_set_blocks=8 seed=1 block_bytes=16777216 decode_steps=32 compute_ms=0.0 mode=all dynamic_weights=True
profile direct_h2d_bw_gbps=7.547
profile_relay relay=5 h2d=7.524 p2p=38.789 effective=7.524 p2p_enabled=True
sim_mode mode=direct tokens_s=61.926 step_p50_ms=16.507 step_p95_ms=16.795 transfer_p50_ms=16.491 transfer_p95_ms=16.772 prefetch_gib_s=7.239 evict_gib_s=7.883 prefetch_blocks=128 evict_blocks=121 direct_chunks=996 relay_chunks=0
sim_mode mode=relay tokens_s=61.365 step_p50_ms=16.672 step_p95_ms=16.886 transfer_p50_ms=16.660 transfer_p95_ms=16.871 prefetch_gib_s=7.166 evict_gib_s=7.816 prefetch_blocks=128 evict_blocks=121 direct_chunks=0 relay_chunks=996
sim_mode mode=pool tokens_s=118.870 step_p50_ms=8.587 step_p95_ms=8.737 transfer_p50_ms=8.573 transfer_p95_ms=8.717 prefetch_gib_s=13.925 evict_gib_s=15.115 prefetch_blocks=128 evict_blocks=121 direct_chunks=498 relay_chunks=498
sim_speedup pool_over_direct_tokens_per_second=1.920
sim_speedup pool_over_relay_tokens_per_second=1.937
COPY_SUMMARY_END
```

Random copy summary:

```text
COPY_SUMMARY_BEGIN
sim_config target=6 relays=[5] requests=4 blocks_per_request=8 blocks_per_step=4 gpu_block_capacity=4 access_pattern=random working_set_blocks=8 seed=1 block_bytes=16777216 decode_steps=32 compute_ms=0.0 mode=all dynamic_weights=True
profile direct_h2d_bw_gbps=7.561
profile_relay relay=5 h2d=7.662 p2p=39.406 effective=7.662 p2p_enabled=True
sim_mode mode=direct tokens_s=61.945 step_p50_ms=16.546 step_p95_ms=16.664 transfer_p50_ms=16.529 transfer_p95_ms=16.636 prefetch_gib_s=7.230 evict_gib_s=7.900 prefetch_blocks=128 evict_blocks=121 direct_chunks=996 relay_chunks=0
sim_mode mode=relay tokens_s=61.351 step_p50_ms=16.700 step_p95_ms=16.808 transfer_p50_ms=16.685 transfer_p95_ms=16.783 prefetch_gib_s=7.158 evict_gib_s=7.826 prefetch_blocks=128 evict_blocks=121 direct_chunks=0 relay_chunks=996
sim_mode mode=pool tokens_s=118.746 step_p50_ms=8.608 step_p95_ms=8.813 transfer_p50_ms=8.594 transfer_p95_ms=8.786 prefetch_gib_s=13.883 evict_gib_s=15.146 prefetch_blocks=128 evict_blocks=121 direct_chunks=498 relay_chunks=498
sim_speedup pool_over_direct_tokens_per_second=1.917
sim_speedup pool_over_relay_tokens_per_second=1.936
COPY_SUMMARY_END
```

Analysis:

The simulator now exercises sustained prefetch and eviction under GPU block
capacity pressure: both access patterns transfer 128 prefetched blocks and 121
evicted blocks. In the round-robin run, pooled transfer improves simulated
decode throughput from 61.926 to 118.870 tokens/s and cuts median transfer
stall from 16.491 ms to 8.573 ms. The random run shows nearly identical
behavior, improving from 61.945 to 118.746 tokens/s. This indicates the
capacity-pressure simulator path is stable and that PCIe bandwidth pooling
reduces transfer stall in an inference-shaped workload.

## Inference Offload Simulator Dummy Compute Overlap

Scenario:

This run uses the same capacity-pressure setup as above, but adds 5 ms of dummy
decode compute per step. The non-overlap run waits for eviction and prefetch
before compute. The overlap run starts dummy compute concurrently with transfer
using a Python thread and `time.sleep`; it is a scheduling model, not a CUDA
kernel overlap test.

No-overlap copy summary:

```text
COPY_SUMMARY_BEGIN
sim_config target=6 relays=[5] requests=4 blocks_per_request=8 blocks_per_step=4 gpu_block_capacity=4 access_pattern=round_robin working_set_blocks=8 seed=1 block_bytes=16777216 decode_steps=32 compute_ms=5.0 overlap_compute=False mode=all dynamic_weights=True
profile direct_h2d_bw_gbps=7.611
profile_relay relay=5 h2d=7.598 p2p=39.223 effective=7.598 p2p_enabled=True
sim_mode mode=direct tokens_s=46.852 step_p50_ms=21.678 step_p95_ms=22.081 transfer_p50_ms=16.541 transfer_p95_ms=16.940 prefetch_gib_s=7.235 evict_gib_s=7.813 prefetch_blocks=128 evict_blocks=121 direct_chunks=996 relay_chunks=0
sim_mode mode=relay tokens_s=46.374 step_p50_ms=21.917 step_p95_ms=22.286 transfer_p50_ms=16.767 transfer_p95_ms=17.135 prefetch_gib_s=7.151 evict_gib_s=7.692 prefetch_blocks=128 evict_blocks=121 direct_chunks=0 relay_chunks=996
sim_mode mode=pool tokens_s=73.253 step_p50_ms=13.830 step_p95_ms=14.106 transfer_p50_ms=8.674 transfer_p95_ms=8.942 prefetch_gib_s=13.876 evict_gib_s=14.760 prefetch_blocks=128 evict_blocks=121 direct_chunks=498 relay_chunks=498
sim_speedup pool_over_direct_tokens_per_second=1.563
sim_speedup pool_over_relay_tokens_per_second=1.580
COPY_SUMMARY_END
```

Overlap copy summary:

```text
COPY_SUMMARY_BEGIN
sim_config target=6 relays=[5] requests=4 blocks_per_request=8 blocks_per_step=4 gpu_block_capacity=4 access_pattern=round_robin working_set_blocks=8 seed=1 block_bytes=16777216 decode_steps=32 compute_ms=5.0 overlap_compute=True mode=all dynamic_weights=True
profile direct_h2d_bw_gbps=7.549
profile_relay relay=5 h2d=7.592 p2p=40.761 effective=7.592 p2p_enabled=True
sim_mode mode=direct tokens_s=60.479 step_p50_ms=16.915 step_p95_ms=17.128 transfer_p50_ms=16.596 transfer_p95_ms=16.724 prefetch_gib_s=7.225 evict_gib_s=7.822 prefetch_blocks=128 evict_blocks=121 direct_chunks=996 relay_chunks=0
sim_mode mode=relay tokens_s=60.028 step_p50_ms=17.067 step_p95_ms=17.246 transfer_p50_ms=16.769 transfer_p95_ms=16.903 prefetch_gib_s=7.150 evict_gib_s=7.748 prefetch_blocks=128 evict_blocks=121 direct_chunks=0 relay_chunks=996
sim_mode mode=pool tokens_s=112.934 step_p50_ms=9.011 step_p95_ms=9.244 transfer_p50_ms=8.668 transfer_p95_ms=8.848 prefetch_gib_s=13.902 evict_gib_s=14.799 prefetch_blocks=128 evict_blocks=121 direct_chunks=498 relay_chunks=498
sim_speedup pool_over_direct_tokens_per_second=1.867
sim_speedup pool_over_relay_tokens_per_second=1.881
COPY_SUMMARY_END
```

Analysis:

With 5 ms of dummy compute serialized after transfer, pooled transfer improves
simulated throughput from 46.852 to 73.253 tokens/s. With the dummy compute
overlapped with transfer, pooled transfer improves from 60.479 to 112.934
tokens/s. The overlap run reduces step p50 from 13.830 ms to 9.011 ms in pooled
mode because part of the compute time is hidden behind the transfer stall. The
pool/direct speedup is 1.563x without overlap and 1.867x with overlap.

## Inference Offload Simulator Packed Storage

Scenario:

This run uses the capacity-pressure simulator with `--storage-layout packed`.
All simulated KV blocks share one pinned CPU backing tensor and one target-GPU
backing tensor. Each block is represented by an offset and byte count, so
`OffloadManager.prefetch_many` and `evict_many` use range-batched transfers.
The setup still models decode-step capacity pressure with LRU eviction and no
dummy compute.

Copy summary:

```text
COPY_SUMMARY_BEGIN
sim_config target=6 relays=[5] requests=4 blocks_per_request=8 blocks_per_step=4 gpu_block_capacity=4 access_pattern=round_robin working_set_blocks=8 seed=1 storage_layout=packed block_bytes=16777216 decode_steps=32 compute_ms=0.0 overlap_compute=False mode=all dynamic_weights=True
sim_scenario type=capacity_pressure unit=decode_step policy=lru_eviction transfer=evict_then_prefetch access_pattern=round_robin working_set_blocks=8 gpu_block_capacity=4 blocks_per_step=4 dummy_compute_ms=0.0 overlap_compute=False compute_impl=python_sleep note=not_cuda_kernel_overlap
profile direct_h2d_bw_gbps=7.507
profile_relay relay=5 h2d=7.626 p2p=40.751 effective=7.626 p2p_enabled=True
sim_mode mode=direct tokens_s=62.025 step_p50_ms=16.520 step_p95_ms=16.686 transfer_p50_ms=16.509 transfer_p95_ms=16.665 prefetch_gib_s=7.225 evict_gib_s=7.923 prefetch_blocks=128 evict_blocks=121 direct_chunks=996 relay_chunks=0
sim_mode mode=relay tokens_s=61.248 step_p50_ms=16.715 step_p95_ms=16.986 transfer_p50_ms=16.700 transfer_p95_ms=16.968 prefetch_gib_s=7.138 evict_gib_s=7.821 prefetch_blocks=128 evict_blocks=121 direct_chunks=0 relay_chunks=996
sim_mode mode=pool tokens_s=119.213 step_p50_ms=8.582 step_p95_ms=8.653 transfer_p50_ms=8.569 transfer_p95_ms=8.636 prefetch_gib_s=13.931 evict_gib_s=15.197 prefetch_blocks=128 evict_blocks=121 direct_chunks=498 relay_chunks=498
sim_speedup pool_over_direct_tokens_per_second=1.922
sim_speedup pool_over_relay_tokens_per_second=1.946
COPY_SUMMARY_END
```

Analysis:

Packed storage exercises the connector-ready range-batch path instead of one
independent tensor per block. The corrected stats now report physical transfer
bandwidth rather than repeated shared-handle bandwidth: direct prefetch is
7.225 GiB/s and pooled prefetch is 13.931 GiB/s. Pooled mode improves simulated
decode throughput from 62.025 to 119.213 tokens/s while keeping the expected
chunk split, 498 direct chunks and 498 relay chunks.

## Inference Offload Simulator Native CUDA Compute

Scenario:

This run uses the capacity-pressure simulator with packed KV storage and
`--compute-impl cuda`. The dummy compute is a native CUDA kernel launched on a
preallocated target-GPU tensor. This checks the CUDA-kernel path and the Python
thread overlap path, rather than the earlier `time.sleep` scheduling model.

Calibration copy summary (`cuda_compute_iterations=64`, no overlap):

```text
COPY_SUMMARY_BEGIN
sim_config target=6 relays=[5] requests=4 blocks_per_request=8 blocks_per_step=4 gpu_block_capacity=4 access_pattern=round_robin working_set_blocks=8 seed=1 storage_layout=packed block_bytes=16777216 decode_steps=32 compute_ms=0.0 overlap_compute=False compute_impl=cuda cuda_compute_elements=16777216 cuda_compute_iterations=64 mode=all dynamic_weights=True
sim_scenario type=capacity_pressure unit=decode_step policy=lru_eviction transfer=evict_then_prefetch access_pattern=round_robin working_set_blocks=8 gpu_block_capacity=4 blocks_per_step=4 dummy_compute_ms=0.0 overlap_compute=False compute_impl=cuda cuda_compute_elements=16777216 cuda_compute_iterations=64 note=cuda_kernel_overlap_model
profile direct_h2d_bw_gbps=7.631
profile_relay relay=5 h2d=7.575 p2p=40.721 effective=7.575 p2p_enabled=True
sim_mode mode=direct tokens_s=58.181 step_p50_ms=16.782 step_p95_ms=16.892 transfer_p50_ms=16.513 transfer_p95_ms=16.574 prefetch_gib_s=7.201 evict_gib_s=7.922 prefetch_blocks=128 evict_blocks=121 direct_chunks=996 relay_chunks=0
sim_mode mode=relay tokens_s=60.330 step_p50_ms=16.972 step_p95_ms=17.145 transfer_p50_ms=16.701 transfer_p95_ms=16.821 prefetch_gib_s=7.143 evict_gib_s=7.827 prefetch_blocks=128 evict_blocks=121 direct_chunks=0 relay_chunks=996
sim_mode mode=pool tokens_s=115.462 step_p50_ms=8.868 step_p95_ms=8.943 transfer_p50_ms=8.594 transfer_p95_ms=8.652 prefetch_gib_s=13.936 evict_gib_s=15.154 prefetch_blocks=128 evict_blocks=121 direct_chunks=498 relay_chunks=498
sim_speedup pool_over_direct_tokens_per_second=1.985
sim_speedup pool_over_relay_tokens_per_second=1.914
COPY_SUMMARY_END
```

Calibration copy summary (`cuda_compute_iterations=64`, overlap):

```text
COPY_SUMMARY_BEGIN
sim_config target=6 relays=[5] requests=4 blocks_per_request=8 blocks_per_step=4 gpu_block_capacity=4 access_pattern=round_robin working_set_blocks=8 seed=1 storage_layout=packed block_bytes=16777216 decode_steps=32 compute_ms=0.0 overlap_compute=True compute_impl=cuda cuda_compute_elements=16777216 cuda_compute_iterations=64 mode=all dynamic_weights=True
sim_scenario type=capacity_pressure unit=decode_step policy=lru_eviction transfer=evict_then_prefetch access_pattern=round_robin working_set_blocks=8 gpu_block_capacity=4 blocks_per_step=4 dummy_compute_ms=0.0 overlap_compute=True compute_impl=cuda cuda_compute_elements=16777216 cuda_compute_iterations=64 note=cuda_kernel_overlap_model
profile direct_h2d_bw_gbps=7.542
profile_relay relay=5 h2d=7.617 p2p=40.727 effective=7.617 p2p_enabled=True
sim_mode mode=direct tokens_s=60.037 step_p50_ms=17.036 step_p95_ms=17.237 transfer_p50_ms=16.711 transfer_p95_ms=16.862 prefetch_gib_s=7.182 evict_gib_s=7.750 prefetch_blocks=128 evict_blocks=121 direct_chunks=996 relay_chunks=0
sim_mode mode=relay tokens_s=59.328 step_p50_ms=17.281 step_p95_ms=17.399 transfer_p50_ms=16.938 transfer_p95_ms=17.071 prefetch_gib_s=7.109 evict_gib_s=7.642 prefetch_blocks=128 evict_blocks=121 direct_chunks=0 relay_chunks=996
sim_mode mode=pool tokens_s=113.286 step_p50_ms=8.965 step_p95_ms=9.281 transfer_p50_ms=8.756 transfer_p95_ms=8.949 prefetch_gib_s=13.808 evict_gib_s=14.562 prefetch_blocks=128 evict_blocks=121 direct_chunks=498 relay_chunks=498
sim_speedup pool_over_direct_tokens_per_second=1.887
sim_speedup pool_over_relay_tokens_per_second=1.909
COPY_SUMMARY_END
```

No-overlap copy summary (`cuda_compute_iterations=512`):

```text
COPY_SUMMARY_BEGIN
sim_config target=6 relays=[5] requests=4 blocks_per_request=8 blocks_per_step=4 gpu_block_capacity=4 access_pattern=round_robin working_set_blocks=8 seed=1 storage_layout=packed block_bytes=16777216 decode_steps=32 compute_ms=0.0 overlap_compute=False compute_impl=cuda cuda_compute_elements=16777216 cuda_compute_iterations=512 mode=all dynamic_weights=True
sim_scenario type=capacity_pressure unit=decode_step policy=lru_eviction transfer=evict_then_prefetch access_pattern=round_robin working_set_blocks=8 gpu_block_capacity=4 blocks_per_step=4 dummy_compute_ms=0.0 overlap_compute=False compute_impl=cuda cuda_compute_elements=16777216 cuda_compute_iterations=512 note=cuda_kernel_overlap_model
profile direct_h2d_bw_gbps=7.585
profile_relay relay=5 h2d=7.559 p2p=39.591 effective=7.559 p2p_enabled=True
sim_mode mode=direct tokens_s=57.672 step_p50_ms=17.633 step_p95_ms=17.929 transfer_p50_ms=16.503 transfer_p95_ms=16.741 compute_p50_ms=1.121 compute_p95_ms=1.208 prefetch_gib_s=7.193 evict_gib_s=7.884 prefetch_blocks=128 evict_blocks=121 direct_chunks=996 relay_chunks=0
sim_mode mode=relay tokens_s=56.953 step_p50_ms=17.925 step_p95_ms=18.252 transfer_p50_ms=16.760 transfer_p95_ms=17.061 compute_p50_ms=1.143 compute_p95_ms=1.194 prefetch_gib_s=7.115 evict_gib_s=7.771 prefetch_blocks=128 evict_blocks=121 direct_chunks=0 relay_chunks=996
sim_mode mode=pool tokens_s=104.858 step_p50_ms=9.747 step_p95_ms=9.816 transfer_p50_ms=8.608 transfer_p95_ms=8.657 compute_p50_ms=1.127 compute_p95_ms=1.144 prefetch_gib_s=13.934 evict_gib_s=15.122 prefetch_blocks=128 evict_blocks=121 direct_chunks=498 relay_chunks=498
sim_speedup pool_over_direct_tokens_per_second=1.818
sim_speedup pool_over_relay_tokens_per_second=1.841
COPY_SUMMARY_END
```

Overlap copy summary (`cuda_compute_iterations=512`):

```text
COPY_SUMMARY_BEGIN
sim_config target=6 relays=[5] requests=4 blocks_per_request=8 blocks_per_step=4 gpu_block_capacity=4 access_pattern=round_robin working_set_blocks=8 seed=1 storage_layout=packed block_bytes=16777216 decode_steps=32 compute_ms=0.0 overlap_compute=True compute_impl=cuda cuda_compute_elements=16777216 cuda_compute_iterations=512 mode=all dynamic_weights=True
sim_scenario type=capacity_pressure unit=decode_step policy=lru_eviction transfer=evict_then_prefetch access_pattern=round_robin working_set_blocks=8 gpu_block_capacity=4 blocks_per_step=4 dummy_compute_ms=0.0 overlap_compute=True compute_impl=cuda cuda_compute_elements=16777216 cuda_compute_iterations=512 note=cuda_kernel_overlap_model
profile direct_h2d_bw_gbps=7.567
profile_relay relay=5 h2d=7.507 p2p=39.473 effective=7.507 p2p_enabled=True
sim_mode mode=direct tokens_s=59.565 step_p50_ms=17.156 step_p95_ms=17.375 transfer_p50_ms=16.792 transfer_p95_ms=16.995 compute_p50_ms=1.243 compute_p95_ms=1.306 prefetch_gib_s=7.156 evict_gib_s=7.694 prefetch_blocks=128 evict_blocks=121 direct_chunks=996 relay_chunks=0
sim_mode mode=relay tokens_s=59.330 step_p50_ms=17.250 step_p95_ms=17.521 transfer_p50_ms=16.933 transfer_p95_ms=17.110 compute_p50_ms=1.216 compute_p95_ms=1.262 prefetch_gib_s=7.093 evict_gib_s=7.653 prefetch_blocks=128 evict_blocks=121 direct_chunks=0 relay_chunks=996
sim_mode mode=pool tokens_s=112.513 step_p50_ms=9.086 step_p95_ms=9.253 transfer_p50_ms=8.846 transfer_p95_ms=8.917 compute_p50_ms=1.188 compute_p95_ms=1.214 prefetch_gib_s=13.805 evict_gib_s=14.479 prefetch_blocks=128 evict_blocks=121 direct_chunks=498 relay_chunks=498
sim_speedup pool_over_direct_tokens_per_second=1.889
sim_speedup pool_over_relay_tokens_per_second=1.896
COPY_SUMMARY_END
```

No-overlap copy summary (`cuda_compute_iterations=2048`):

```text
COPY_SUMMARY_BEGIN
sim_config target=6 relays=[5] requests=4 blocks_per_request=8 blocks_per_step=4 gpu_block_capacity=4 access_pattern=round_robin working_set_blocks=8 seed=1 storage_layout=packed block_bytes=16777216 decode_steps=32 compute_ms=0.0 overlap_compute=False compute_impl=cuda cuda_compute_elements=16777216 cuda_compute_iterations=2048 mode=all dynamic_weights=True
sim_scenario type=capacity_pressure unit=decode_step policy=lru_eviction transfer=evict_then_prefetch access_pattern=round_robin working_set_blocks=8 gpu_block_capacity=4 blocks_per_step=4 dummy_compute_ms=0.0 overlap_compute=False compute_impl=cuda cuda_compute_elements=16777216 cuda_compute_iterations=2048 note=cuda_kernel_overlap_model
profile direct_h2d_bw_gbps=7.518
profile_relay relay=5 h2d=7.502 p2p=39.521 effective=7.502 p2p_enabled=True
sim_mode mode=direct tokens_s=48.977 step_p50_ms=20.763 step_p95_ms=20.999 transfer_p50_ms=16.509 transfer_p95_ms=16.707 compute_p50_ms=4.239 compute_p95_ms=4.265 prefetch_gib_s=7.195 evict_gib_s=7.918 prefetch_blocks=128 evict_blocks=121 direct_chunks=996 relay_chunks=0
sim_mode mode=relay tokens_s=48.528 step_p50_ms=21.006 step_p95_ms=21.128 transfer_p50_ms=16.739 transfer_p95_ms=16.849 compute_p50_ms=4.246 compute_p95_ms=4.281 prefetch_gib_s=7.130 evict_gib_s=7.804 prefetch_blocks=128 evict_blocks=121 direct_chunks=0 relay_chunks=996
sim_mode mode=pool tokens_s=78.467 step_p50_ms=12.902 step_p95_ms=13.038 transfer_p50_ms=8.640 transfer_p95_ms=8.752 compute_p50_ms=4.240 compute_p95_ms=4.288 prefetch_gib_s=13.724 evict_gib_s=15.054 prefetch_blocks=128 evict_blocks=121 direct_chunks=498 relay_chunks=498
sim_speedup pool_over_direct_tokens_per_second=1.602
sim_speedup pool_over_relay_tokens_per_second=1.617
COPY_SUMMARY_END
```

Overlap copy summary (`cuda_compute_iterations=2048`):

```text
COPY_SUMMARY_BEGIN
sim_config target=6 relays=[5] requests=4 blocks_per_request=8 blocks_per_step=4 gpu_block_capacity=4 access_pattern=round_robin working_set_blocks=8 seed=1 storage_layout=packed block_bytes=16777216 decode_steps=32 compute_ms=0.0 overlap_compute=True compute_impl=cuda cuda_compute_elements=16777216 cuda_compute_iterations=2048 mode=all dynamic_weights=True
sim_scenario type=capacity_pressure unit=decode_step policy=lru_eviction transfer=evict_then_prefetch access_pattern=round_robin working_set_blocks=8 gpu_block_capacity=4 blocks_per_step=4 dummy_compute_ms=0.0 overlap_compute=True compute_impl=cuda cuda_compute_elements=16777216 cuda_compute_iterations=2048 note=cuda_kernel_overlap_model
profile direct_h2d_bw_gbps=7.625
profile_relay relay=5 h2d=7.619 p2p=40.471 effective=7.619 p2p_enabled=True
sim_mode mode=direct tokens_s=59.556 step_p50_ms=17.042 step_p95_ms=17.212 transfer_p50_ms=16.703 transfer_p95_ms=16.801 compute_p50_ms=4.346 compute_p95_ms=4.376 prefetch_gib_s=7.184 evict_gib_s=7.750 prefetch_blocks=128 evict_blocks=121 direct_chunks=996 relay_chunks=0
sim_mode mode=relay tokens_s=59.584 step_p50_ms=17.160 step_p95_ms=17.425 transfer_p50_ms=16.887 transfer_p95_ms=17.106 compute_p50_ms=4.290 compute_p95_ms=4.406 prefetch_gib_s=7.108 evict_gib_s=7.659 prefetch_blocks=128 evict_blocks=121 direct_chunks=0 relay_chunks=996
sim_mode mode=pool tokens_s=114.267 step_p50_ms=8.895 step_p95_ms=9.168 transfer_p50_ms=8.735 transfer_p95_ms=8.896 compute_p50_ms=4.368 compute_p95_ms=4.402 prefetch_gib_s=13.803 evict_gib_s=14.689 prefetch_blocks=128 evict_blocks=121 direct_chunks=498 relay_chunks=498
sim_speedup pool_over_direct_tokens_per_second=1.919
sim_speedup pool_over_relay_tokens_per_second=1.918
COPY_SUMMARY_END
```

Analysis:

The native CUDA dummy compute path works, and pooled transfer remains around
1.8-1.9x faster than direct/relay under capacity pressure. With
`cuda_compute_iterations=64`, the kernel is too light to show useful overlap
behavior. With `cuda_compute_iterations=512`, compute becomes visible at about
1.1-1.2 ms per step. Overlap reduces direct step p50 from 17.633 ms to
17.156 ms and pool step p50 from 9.747 ms to 9.086 ms, so some compute is
hidden behind transfer.

With `cuda_compute_iterations=2048`, compute rises to about 4.3 ms. The
no-overlap step p50 is roughly transfer plus compute: direct is 20.763 ms
versus 16.509 ms transfer and 4.239 ms compute, while pool is 12.902 ms versus
8.640 ms transfer and 4.240 ms compute. With overlap enabled, direct step p50
drops to 17.042 ms and pool step p50 drops to 8.895 ms, close to the transfer
stall alone. This indicates most of the native CUDA dummy compute is hidden
behind transfer in this simulator. Under overlap, pooled transfer reaches
114.267 tokens/s versus 59.556 direct, a 1.919x speedup.

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
