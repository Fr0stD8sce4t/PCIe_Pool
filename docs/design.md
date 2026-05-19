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

The pool benchmark now uses this production planner for pooled transfers instead
of a hand-written even/odd chunk split.

`benchmarks/bandwidth_pool.py` can emit a JSON report with the run config,
profile result, per-mode samples, medians, speedups, and optional correctness
check. `benchmarks/tune_transfer.py` sweeps chunk sizes and staging slot counts
and reports the best median pooled bandwidth for the tested target/relay pair.

## Python API

The Python wrapper only accepts contiguous PyTorch tensors:

- source tensor must be CPU pinned memory
- destination tensor must be CUDA memory on the runtime target GPU
- copy size is derived from the source tensor byte size

The native extension receives raw tensor pointers and byte counts.

`TransferHandle.wait()` populates a lightweight stats object with total bytes,
CUDA event elapsed time, submit-to-complete wall-clock time, GiB/s, and
direct/relay chunk counts. `gib_per_second` is based on CUDA event timing;
`submit_gib_per_second` is based on the wall-clock time between submit and wait
completion.

The runtime caches the first profile result and reuses it for subsequent
transfers. By default, profiling runs on the first transfer. This can be disabled
through `RuntimeOptions.profile_on_first_transfer`, in which case the runtime
falls back to equal path weights until `profile()` is called explicitly.
Calling `profile(force=True)` refreshes the cached measurement.

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
