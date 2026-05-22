# TurboBus Next Steps

Future coding sessions should start here and take the first unfinished task.
After each task, update this file and `docs/PROGRESS.md`.

## Active Sequence

1. Split transfer selection out of `turbobus/runtime.py`.
   - Create `turbobus/transfer_selector.py`.
   - Move `TransferMode`, `AutoTransferDecision`, and `AutoTransferSelector`
     without changing public imports from `turbobus.runtime`.
   - Keep existing Runtime behavior unchanged.
   - Acceptance:
     - existing Python tests pass;
     - `from turbobus.runtime import AutoTransferSelector, TransferMode` still works;
     - benchmarks can still import and use `turbobus.Runtime`.

2. Split Runtime plan/profile helpers.
   - Create `turbobus/plan_trace.py` for `transfer_plan_to_dict` and related
     conversion helpers.
   - Create `turbobus/profile_cache.py` only if it removes real Runtime
     complexity; do not add a speculative cache layer.
   - Acceptance:
     - Runtime public API stays stable;
     - plan dict output is unchanged;
     - tests cover old import paths.

3. Add multi-relay planner coverage.
   - Extend C++ planner tests for two relay GPUs with different effective
     bandwidths.
   - Verify chunk assignment roughly follows path bandwidth.
   - Verify H2D and D2H both use direction-specific bandwidth fields.
   - Acceptance:
     - planner test passes;
     - no vLLM or Python policy enters the native planner.

4. Improve multi-relay executor behavior.
   - Ensure relay staging slots and streams are isolated per relay.
   - Verify pool transfer can use direct plus more than one relay path.
   - Report per-relay bytes/chunks in stats.
   - Acceptance:
     - CUDA correctness tests pass on a multi-GPU server;
     - path stats show all used relays.

5. Wire daemon transfer reservations into Runtime planning.
   - Add or stabilize daemon messages for transfer reservation and release.
   - Runtime should request relay permission before planning relay chunks.
   - If reservation denies relay use, Runtime should reduce relay paths or
     fall back to direct.
   - Acceptance:
     - daemon state tests cover quota and reservation release;
     - Runtime stats expose reservation/session information.

6. Continue vLLM connector lifecycle cleanup.
   - Keep save intent, block ids, metadata, worker save, prefix registration,
     and completion inside `TurboBusConnector`.
   - Example scripts should send `kv_transfer_params` and report events only.
   - Acceptance:
     - no example-side manual prefix registration in the real connector path;
     - save and restore events report bytes, layers, ranges, chunks, and timing.

7. Add model loading workload API and benchmark.
   - Add CPU pinned weight bucket movement through Runtime.
   - Measure direct, relay, pool, and auto.
   - Acceptance:
     - benchmark reports load latency, path split, and speedup.

8. Add training offload bucket API and benchmark.
   - Add PyTorch tensor bucket movement suitable for parameter or optimizer
     state offload.
   - Acceptance:
     - benchmark reports iteration proxy time, transfer time, and path split.

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
