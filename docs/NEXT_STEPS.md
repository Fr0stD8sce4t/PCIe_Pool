# TurboBus Next Steps

Future coding sessions should start here and take the task under `## Current`.
After that task is complete, move it to `## Completed`, promote the first item
from `## Upcoming` to `## Current`, then update `docs/PROGRESS.md`.

## Current

### 9. Add Daemon Shared Profile Cache

Let Runtime read and publish measured direct/relay profile data through the
daemon so multiple processes do not each need to profile the same target/relay
pair.

Acceptance:

- Runtime can use daemon-provided profile data before local profiling.
- Runtime publishes a fresh profile back to the daemon after local profiling.
- Unit tests cover cache hit, cache miss, and stale or invalid daemon data.

## Completed

- 2026-05-22: Add training offload bucket API and benchmark.
  - Added `TrainingOffloadManager` / `TrainingOffloadStore` for parameter or
    optimizer bucket prefetch and offload through Runtime.
  - Added packed CPU/GPU bucket registration so training offload can use
    range-batched H2D and D2H transfers.
  - Added `benchmarks/training_offload.py` to measure iteration proxy time,
    transfer time, compute proxy time, and H2D/D2H path split.
  - Added focused unit tests for the training offload API with a fake Runtime.

- 2026-05-22: Add model loading workload API and benchmark.
  - Added `ModelWeightLoader` / `ModelLoader` as a model-weight bucket API over
    Runtime H2D transfers.
  - Added packed CPU pinned weight bucket registration so model-loading
    benchmarks can use one range-batched Runtime transfer.
  - Added `benchmarks/model_loading.py` to measure direct, relay, pool, and
    auto model-weight load latency, path split, and speedup.
  - Added focused unit tests for the model-loading API with a fake Runtime.

- 2026-05-22: Continue vLLM connector lifecycle cleanup.
  - Added a connector event stream API so examples can report connector
    lifecycle outcomes without reading internal prefix-store objects.
  - Updated `examples/vllm_turbobus_kv_connector.py` to derive save/restore
    summaries from connector events emitted by `TurboBusConnector`.
  - The official connector example now reports restore bytes, layers, ranges,
    chunks, timing, and auto-selection fields from connector events.
  - Unit tests now construct `TurboBusConnector` with `kv_cache_config`, matching
    the current vLLM base-class lifecycle.

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
  - Added a CUDA correctness test for a pool transfer that uses direct plus two
    relay paths.
  - The test verifies per-relay bytes/chunks and path stats for all used paths.
  - Native benchmark output now prints direct/relay chunk counts, per-relay
    bytes/chunks, and path stats.

- 2026-05-22: Add multi-relay planner coverage.
  - Extended `test/cpp/test_planner.cpp` with two relay GPUs.
  - Covered H2D and D2H direction-specific effective bandwidth fields.
  - Verified expected chunk assignment against the planner's bandwidth-weighted
    greedy selection.

- 2026-05-22: Split Runtime plan trace helpers out of `turbobus/runtime.py`.
  - Added `turbobus/plan_trace.py` for `transfer_plan_to_dict`.
  - Kept `turbobus.runtime.transfer_plan_to_dict` working.
  - Did not add `profile_cache.py`; there was no real Runtime complexity to
    remove yet.

- 2026-05-22: Split transfer selection out of `turbobus/runtime.py`.
  - Added `turbobus/transfer_selector.py`.
  - Kept `turbobus.runtime` re-export imports working.

- 2026-05-22: Document TurboBus roadmap workflow (`830d137`).

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
