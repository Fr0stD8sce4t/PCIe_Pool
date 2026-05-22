# TurboBus Next Steps

Future coding sessions should start here and take the task under `## Current`.
After that task is complete, move it to `## Completed`, promote the first item
from `## Upcoming` to `## Current`, then update `docs/PROGRESS.md`.

## Current

### 14. Build the real vLLM KV offload integration path

Move vLLM prefix save and restore farther into the real connector lifecycle.
Keep using vLLM-owned KV cache tensors and block ids, and keep the example from
driving save through private prefix-store calls.

Acceptance:

- Restore and save continue to flow through `KVConnectorBase_V1` metadata.
- The example reports connector events without mutating connector internals.
- Connector changes stay version-aware for the current server vLLM build.

Progress:

- 2026-05-22: Decoupled the example's save request from
  `--restore-enabled`. The first request now asks the connector to save by
  default, while restore remains an explicit opt-in for the second request.

## Completed

- 2026-05-22: Add batch block operations on top of the existing transfer APIs.
  - Added `OffloadBatch` plus `submit_prefetch_many()` and
    `submit_evict_many()` to the reusable `OffloadStore` API.
  - Kept `prefetch_many(names)` and `evict_many(names)` returning handle lists
    for existing benchmarks and adapters.
  - Packed blocks still submit one range-batched Runtime transfer, while
    non-packed blocks keep the per-block fallback path.
  - Added focused tests for batch wait, stats, block snapshots, empty batches,
    and range-batched handle reuse.

- 2026-05-22: Make the Python offload object layer reusable by future
  connectors.
  - Added block-store style aliases and helpers to `OffloadStore` for
    `add_block`, `get_block`, `block_ids`, and block state cleanup.
  - `ModelWeightLoader`, `TrainingOffloadManager`, and
    `InferenceKVSlotAdapter` now inherit the common block-store layer instead
    of wrapping it with a second internal store object.
  - Added focused tests for the reusable block-store aliases and for the
    loader, training, and inference adapters exposing shared block ids.

- 2026-05-22: Add daemon multi-process benchmark smoke.
  - Added `benchmarks/daemon_smoke.py` to start a local daemon, run two
    benchmark clients, and report shared profile cache and reservation status.
  - The wrapper keeps CUDA transfer work inside the benchmark clients and only
    coordinates daemon lifecycle from the outside.
  - Added focused unit tests for the daemon smoke command construction and
    summary parsing helpers.

- 2026-05-22: Add daemon-aware benchmark options.
  - Added shared daemon benchmark option helpers.
  - `bandwidth_pool.py`, `model_loading.py`, and `training_offload.py` now
    accept daemon socket and daemon inflight chunk options.
  - Benchmark JSON and compact summary output report daemon profile status and
    daemon transfer reservation status when enabled.
  - Benchmarks now default to reusable profile cache behavior and expose
    `--force-profile` for explicit refreshes.

- 2026-05-22: Add daemon shared profile cache.
  - Added daemon `GET_PROFILE` and `PUT_PROFILE` requests with validation and
    per target/relay cache keys.
  - Runtime now reads daemon profile data during daemon session setup and
    injects cache hits into the native Runtime profile cache.
  - Runtime publishes freshly measured local profiles back to the daemon.
  - Added Runtime and daemon tests for cache hit, miss, stale, invalid, and
    socket client paths.

- 2026-05-22: Add training offload bucket API and benchmark.
  - Added `TrainingOffloadManager` / `TrainingOffloadStore` for parameter or
    optimizer bucket movement through Runtime.
  - The API supports prefetch from CPU pinned memory to the target GPU and
    offload back to CPU pinned memory through the shared Runtime path.
  - Packed CPU/GPU backing buffers use range-batched Runtime transfers in both
    H2D and D2H directions.
  - Added `benchmarks/training_offload.py` to report iteration proxy time,
    transfer time, compute proxy time, H2D/D2H path split, and speedup summary
    lines.

- 2026-05-22: Add model loading workload API and benchmark.
  - Added `ModelWeightLoader` / `ModelLoader` for model-weight bucket loading
    through the shared Runtime H2D path.
  - The API supports separate buckets and packed CPU/GPU backing buffers, using
    range-batched Runtime transfers for packed buckets.
  - Added `benchmarks/model_loading.py` to compare direct, relay, pool, and
    auto load modes with load latency, bandwidth, path split, and speedup
    summary lines.
  - Added focused unit tests for the model-loading API with a fake Runtime.

- 2026-05-22: Continue vLLM connector lifecycle cleanup.
  - Added a connector event stream API so examples can report connector
    lifecycle outcomes without reading internal prefix-store objects.
  - Updated `examples/vllm_turbobus_kv_connector.py` to derive save/restore
    summaries from connector events emitted by `TurboBusConnector`.
  - The official connector example now reports restore bytes, layers, ranges,
    chunks, timing, and auto-selection fields from connector events.
  - Unit tests now construct `TurboBusConnector` with `kv_cache_config`,
    matching the current vLLM base-class lifecycle.

- 2026-05-22: Wire daemon transfer reservations into Runtime planning.
  - Added a daemon socket client and Runtime daemon session registration.
  - Runtime now reserves relay chunks before relay/pool transfers, releases
    reservations after `TransferHandle.wait()`, and falls back to direct when
    daemon quota denies relay use.
  - Runtime stats and vLLM restore events expose daemon session/reservation
    fields.
  - vLLM connector config can pass `turbobus.daemon_socket_path` and
    `turbobus.daemon_max_inflight_chunks` into Runtime.

- 2026-05-22: Improve multi-relay executor behavior.
  - Added a CUDA correctness test for a pool transfer that uses direct plus
    two relay paths.
  - The test verifies per-relay bytes/chunks and path stats for all used paths.
  - Native benchmark output now prints direct/relay chunk counts, per-relay
    bytes/chunks, and path stats.

- 2026-05-22: Add multi-relay planner coverage.
  - Extended `test/cpp/test_planner.cpp` with two relay GPUs.
  - Covered H2D and D2H direction-specific effective bandwidth fields.
  - Verified expected chunk assignment against the planner's
    bandwidth-weighted greedy selection.

- 2026-05-22: Split Runtime plan trace helpers out of `turbobus/runtime.py`.
  - Added `turbobus/plan_trace.py` for `transfer_plan_to_dict`.
  - Kept `turbobus.runtime.transfer_plan_to_dict` working.
  - Did not add `profile_cache.py`; there was no real Runtime complexity to
    remove yet.

- 2026-05-22: Split transfer selection out of `turbobus/runtime.py`.
  - Added `turbobus/transfer_selector.py`.
  - Kept `turbobus.runtime` re-export imports working.

- 2026-05-22: Document TurboBus roadmap workflow (`830d137`).

## Upcoming

- Add production-shaped offload clients for the three paper workloads.
- Expand daemon work toward the paper architecture.

## Working Rules

- Do not skip ahead unless the current task is blocked.
- Do not spend a turn only improving tests or docs unless it directly supports
  the current task.
- Prefer small commits that each move one roadmap item.
- Keep `references/` read-only.
- If C++/CUDA/pybind files change, remind the user to reinstall before server
  testing.

## Verification Policy

Do not run every available test after every change. Choose the smallest useful
check set for the touched code path.

Use this default split:

- Documentation or roadmap-only edits: `git diff --check` is enough.
- Narrow Python helper/API edits: run the directly related Python unit tests;
  add `python -m compileall <touched package>` when import syntax could be
  affected.
- Runtime, planner, selector, stats, or daemon Python edits: run the related
  Python test files for that area. Full `unittest discover` is optional and
  should be reserved for cross-module changes.
- C++/CUDA/pybind edits: run the relevant native build and native correctness
  tests on the server.
- vLLM connector or vLLM benchmark edits: run a small vLLM smoke or sweep on
  the server.
- Full native CUDA checks, profiler checks, and long vLLM sweeps are milestone
  checks, not per-update defaults.

Use target GPU 6 and relay GPU 5 for server checks unless the user gives
another topology.

## After Each Code Step

When a roadmap item is advanced, the final response must include related test
commands for that exact change.

Include:

- commands that were actually run;
- commands the user can run next on the server, when server validation is
  relevant;
- a short note when no CUDA/vLLM/server check is needed.

Do not paste the full benchmark suite by default. Keep the commands targeted to
the code touched by the current step.
