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
