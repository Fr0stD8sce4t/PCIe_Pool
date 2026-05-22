# TurboBus Next Steps

Future coding sessions should start here and take the task under `## Current`.
After that task is complete, move it to `## Completed`, promote the first item
from `## Upcoming` to `## Current`, then update `docs/PROGRESS.md`.

## Current

### 2. Split Runtime Plan/Profile Helpers

Create `turbobus/plan_trace.py` for `transfer_plan_to_dict` and related
conversion helpers. Create `turbobus/profile_cache.py` only if it removes real
Runtime complexity; do not add a speculative cache layer.

Acceptance:

- Runtime public API stays stable.
- Plan dict output is unchanged.
- Tests cover old import paths.

## Upcoming

### 3. Add Multi-Relay Planner Coverage

Extend C++ planner tests for two relay GPUs with different effective
bandwidths. Verify chunk assignment roughly follows path bandwidth. Verify H2D
and D2H both use direction-specific bandwidth fields.

Acceptance:

- Planner test passes.
- No vLLM or Python policy enters the native planner.

### 4. Improve Multi-Relay Executor Behavior

Ensure relay staging slots and streams are isolated per relay. Verify pool
transfer can use direct plus more than one relay path. Report per-relay
bytes/chunks in stats.

Acceptance:

- CUDA correctness tests pass on a multi-GPU server.
- Path stats show all used relays.

### 5. Wire Daemon Transfer Reservations Into Runtime Planning

Add or stabilize daemon messages for transfer reservation and release. Runtime
should request relay permission before planning relay chunks. If reservation
denies relay use, Runtime should reduce relay paths or fall back to direct.

Acceptance:

- Daemon state tests cover quota and reservation release.
- Runtime stats expose reservation/session information.

### 6. Continue vLLM Connector Lifecycle Cleanup

Keep save intent, block ids, metadata, worker save, prefix registration, and
completion inside `TurboBusConnector`. Example scripts should send
`kv_transfer_params` and report events only.

Acceptance:

- No example-side manual prefix registration in the real connector path.
- Save and restore events report bytes, layers, ranges, chunks, and timing.

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

## Suggested Server Checks

Use target GPU 6 and relay GPU 5 unless the user gives another topology.

```bash
python -m unittest discover -s test/python -p "test_*.py" -v
python -m compileall turbobus benchmarks examples test/python -q
cmake -S test/cpp -B build-test
cmake --build build-test --config Release
TURBOBUS_TARGET_GPU=6 TURBOBUS_RELAY_GPU=5 TURBOBUS_PROFILE_BYTES=16777216 ./build-test/test_profiler
TURBOBUS_TARGET_GPU=6 TURBOBUS_RELAY_GPU=5 TURBOBUS_TEST_BYTES=33554432 ./build-test/test_relay_h2d_p2p
```
