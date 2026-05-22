# TurboBus Next Steps

Future coding sessions should start here and take the task under `## Current`.
After that task is complete, move it to `## Completed`, promote the first item
from `## Upcoming` to `## Current`, then update `docs/PROGRESS.md`.

## Current

### 6. Continue vLLM Connector Lifecycle Cleanup

Keep save intent, block ids, metadata, worker save, prefix registration, and
completion inside `TurboBusConnector`. Example scripts should send
`kv_transfer_params` and report events only.

Acceptance:

- No example-side manual prefix registration in the real connector path.
- Save and restore events report bytes, layers, ranges, chunks, and timing.

## Upcoming

### 7. Add Model Loading Workload API And Benchmark

Add CPU pinned weight bucket movement through Runtime. Measure direct, relay,
pool, and auto.

Acceptance:

- Benchmark reports load latency, path split, and speedup.

### 8. Add Training Offload Bucket API And Benchmark

Add PyTorch tensor bucket movement suitable for parameter or optimizer state
offload.

Acceptance:

- Benchmark reports iteration proxy time, transfer time, and path split.

## Completed

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
