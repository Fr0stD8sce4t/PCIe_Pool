# TurboBus MVP Design

This repository currently implements the first single-node TurboBus boundary:

- pinned host memory to target GPU direct copy
- pinned host memory to relay GPU staging buffer
- relay GPU staging buffer to target GPU P2P copy
- chunk-level scheduling
- CUDA stream/event based relay flow
- simple bandwidth profiler
- thin Python/PyTorch API
- daemon state skeleton for session and relay quota control

It intentionally does not implement RDMA, cross-node transfer, HMC integration,
vLLM/SGLang patching, or a full KV cache state machine.

## Data Path

```text
direct:
  CPU pinned memory -> target GPU

relay:
  CPU pinned memory -> relay GPU staging slot
  relay GPU staging slot -> target GPU
```

Each relay owns:

- one H2D stream
- one P2P stream
- a ring of staging slots
- one `h2d_done` event per staging slot
- one `p2p_done` event per staging slot

Before reusing a staging slot, the H2D stream waits for the previous `p2p_done`
event for that slot. This prevents the next host-to-relay copy from overwriting
data that is still being forwarded to the target GPU.

## Planner

The first planner is deliberately simple. It builds paths from profiler output:

- direct path bandwidth is `CPU -> target GPU`
- relay path bandwidth is `min(CPU -> relay GPU, relay GPU -> target GPU)`

Chunks are assigned by the current normalized assigned bytes over path bandwidth.
This approximates bandwidth-proportional distribution without adding a complex
runtime scheduler.

The planner supports three transfer modes:

- `pool`: use direct and relay paths together
- `direct`: use only `CPU -> target GPU`
- `relay`: use only `CPU -> relay GPU -> target GPU`

In `pool` mode, requests with fewer than `min_chunks_for_relay` chunks fall back
to direct-only transfer. The default threshold is 2 chunks. This keeps small
requests from paying relay overhead when there is not enough work to split.

Relay paths can also be filtered conservatively before planning:

- `relay_min_effective_bw_gbps` skips relays below an absolute effective
  bandwidth
- `relay_min_direct_ratio` skips relays whose effective bandwidth is below a
  fraction of the direct H2D bandwidth

Both filters default to 0, so the default planner behavior is unchanged.

The pool benchmark now uses this production planner for pooled transfers instead
of a hand-written even/odd chunk split.

`benchmarks/bandwidth_pool.py` can emit a JSON report with the run config,
profile result, per-mode samples, medians, speedups, last transfer plan, and
optional correctness check. `benchmarks/tune_transfer.py` sweeps chunk sizes and
staging slot counts and reports the best median pooled bandwidth for the tested
target/relay pair.

## Python API

The Python wrapper only accepts contiguous PyTorch tensors:

- source tensor must be CPU pinned memory
- destination tensor must be CUDA memory on the runtime target GPU
- copy size is derived from the source tensor byte size

The runtime also exposes `offload_to_cpu(gpu_tensor, cpu_tensor)` for D2H
offload into pinned CPU memory. The first D2H implementation mirrors the H2D
planner shape: direct copies use `target GPU -> CPU pinned`, and relay copies
use `target GPU -> relay GPU staging -> CPU pinned`.

The native extension receives raw tensor pointers and byte counts.

`TransferHandle.wait()` populates a lightweight stats object with total bytes,
direct/relay bytes, per-relay bytes and chunk counts, CUDA event elapsed time,
submit-to-complete wall-clock time, per-path CUDA timing, GiB/s, and
direct/relay chunk counts. `gib_per_second` is based on CUDA event timing;
`submit_gib_per_second` is based on the wall-clock time between submit and wait
completion.

`TransferStats.path_stats` records one entry per planned path assignment. Each
entry includes the path kind, transfer direction, relay device, bytes, chunks,
CUDA elapsed time, and path-local GiB/s. For a pooled direct + relay transfer
this makes it possible to see which path is the bottleneck without changing the
transfer schedule.

Dynamic weights can be enabled through `RuntimeOptions.enable_dynamic_weights`.
When enabled, completed H2D `path_stats` update a per-runtime planner profile
using an exponential moving average controlled by `dynamic_weight_alpha`. The
default is disabled, so profile-based planning and transfer behavior stay
unchanged unless requested.

The runtime keeps the last generated `TransferPlan`. Python callers can inspect
it through `Runtime.last_plan_dict()` to see which chunks used the direct path
and which chunks used each relay GPU.

The runtime caches the first profile result and reuses it for subsequent
transfers. By default, profiling runs on the first transfer. This can be disabled
through `RuntimeOptions.profile_on_first_transfer`, in which case the runtime
falls back to equal path weights until `profile()` is called explicitly.
Calling `profile(force=True)` refreshes the cached measurement.

`RuntimeOptions.from_tuning_json(path)` reads the best chunk size and staging
slot count from a tuner JSON output. `RuntimeOptions.from_profile_json(path)`
reads the benchmark profile config fields that are useful for recreating a
runtime configuration.

## Daemon Boundary

The first daemon version is a resource-control skeleton served over a local
Unix socket. It tracks:

- sessions
- relay GPU quotas
- max sessions per relay
- max inflight chunks per relay

It does not transfer GPU pointers across processes. Client processes still run
the CUDA transfer locally after reserving relay resources. This avoids taking on
CUDA IPC and cross-process pointer lifetime issues before the core relay transfer
path is validated.

Supported requests in this MVP:

- `REGISTER_SESSION`
- `PROFILE`
- `CLOSE_SESSION`

`FETCH_TO_GPU` is reserved in the protocol enum for later expansion, but it is
not wired to cross-process execution in this version.

## Validation Status

No build or runtime tests have been executed in this environment. The code needs
to be built and validated on a CUDA machine with at least two P2P-capable GPUs
before claiming functional correctness or performance.
