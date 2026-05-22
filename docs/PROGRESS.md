# TurboBus Progress

Update this file after every coding turn that changes the project.

## Status As Of 2026-05-22

The active goal is to turn TurboBus from a working prototype into a paper
reproduction system for PCIe bandwidth pooling via relay GPUs.

## Recent Mainline Commits

- Add batch client APIs to the offload workload managers
  - `ModelWeightLoader`, `TrainingOffloadManager`, and
    `InferenceKVSlotAdapter` now expose batch submit helpers that return
    `OffloadBatch` objects with `wait()` and `transfer_stats()`.
  - Existing list-based helper methods remain in place for benchmarks and
    examples.
  - Added focused tests to cover the new batch client entry points across the
    three workload adapters.

- Broaden vLLM block-id extraction for version differences
  - `extract_vllm_block_ids()` now accepts `block_ids` attributes and raw
    list/tuple shapes in addition to `get_block_ids()`.
  - Added regression coverage for raw block-id lists and simple namespace
    shapes that mirror the current vLLM integration probes.

- Keep vLLM connector save intent independent from restore
  - `examples/vllm_turbobus_kv_connector.py` now asks the first request to
    save through connector `kv_transfer_params` by default.
  - `--restore-enabled` now only controls whether the second request restores
    saved backing into vLLM KV slots.
  - Added a focused parser test so `--no-save` can explicitly disable the save
    request without changing restore behavior.

- Add batch block operations on top of Runtime transfers
  - Added `OffloadBatch` to carry batch operation names, handles, block
    snapshots, and transfer stats for future connector use.
  - Added `submit_prefetch_many()` and `submit_evict_many()` while preserving
    the existing `prefetch_many()` and `evict_many()` handle-list API.
  - Packed block batches still use one range-batched Runtime transfer; separate
    tensors keep the simple per-block fallback path.

- Make the Python offload object layer reusable by future connectors
  - Added block-store style aliases and helpers to `OffloadStore` for
    `add_block`, `get_block`, `block_ids`, and block state cleanup.
  - `ModelWeightLoader`, `TrainingOffloadManager`, and
    `InferenceKVSlotAdapter` now inherit the shared block-store layer instead
    of wrapping a second internal store object.
  - Added focused tests for the reusable block-store aliases and for the
    loader, training, and inference adapters exposing shared block ids.

- Extend daemon smoke to training offload
  - Added `training-offload` as a supported daemon smoke workload so the same
    wrapper can exercise the training offload benchmark through a local daemon.
  - Added regression coverage for the training-offload client command builder.

- Fix daemon smoke parser conflict
  - Split the daemon smoke parser construction into a helper so the wrapper can
    accept both daemon-launch quotas and client daemon options without
    duplicate argparse flags.
  - Added a regression test that parses daemon socket and inflight chunk
    options successfully.

- Add daemon multi-process benchmark smoke
  - Added `benchmarks/daemon_smoke.py` to start a local daemon, run two
    benchmark clients, and surface shared profile cache and reservation state.
  - The smoke wrapper keeps CUDA movement inside the benchmark clients and
    only orchestrates daemon lifecycle around them.
  - Added focused unit tests for daemon smoke command construction and summary
    parsing helpers.

- Add daemon-aware benchmark options
  - Added `benchmarks/daemon_support.py` for shared daemon benchmark argument
    and summary helpers.
  - `bandwidth_pool.py`, `model_loading.py`, and `training_offload.py` now pass
    daemon socket settings into `RuntimeOptions`.
  - Compact summaries now include daemon profile cache status and daemon
    reservation status when daemon mode is enabled.
  - Benchmarks now default to reusable profile cache behavior and expose
    `--force-profile` when a fresh profile measurement is needed.

- Add daemon shared profile cache
  - Added daemon profile cache requests for reading and publishing measured
    direct/relay profile data by target GPU and relay GPU set.
  - Runtime now checks the daemon profile cache during daemon session setup,
    ignores stale or invalid entries, and records the cache status.
  - Daemon profile cache hits are injected into the native Runtime cache so
    transfers can use the shared profile without repeating local profiling.
  - Runtime publishes fresh local profile results back to the daemon after
    `Runtime.profile()`.

- Add training offload bucket API and benchmark
  - Added `turbobus.training_offload.TrainingOffloadManager` and
    `TrainingOffloadStore` for PyTorch parameter or optimizer bucket movement.
  - The API supports prefetch from CPU pinned memory to the target GPU and
    offload back to CPU pinned memory through the shared Runtime path.
  - Packed CPU/GPU backing buffers use range-batched Runtime transfers in both
    H2D and D2H directions.
  - Added `benchmarks/training_offload.py` to report iteration proxy time,
    transfer time, compute proxy time, H2D/D2H path split, and speedup summary
    lines.

- Add model loading workload API and benchmark
  - Added `turbobus.model_loading.ModelWeightLoader` and `ModelLoader` for
    model-weight bucket loading through the shared Runtime H2D path.
  - The API supports separate buckets and packed CPU/GPU backing buffers, using
    range-batched Runtime transfers for packed buckets.
  - Added `benchmarks/model_loading.py` to compare direct, relay, pool, and
    auto load modes with load latency, bandwidth, path split, and speedup
    summary lines.
  - Added focused unit tests for the model-loading API with a fake Runtime.

- Continue vLLM connector lifecycle cleanup
  - Added `clear_connector_events()` and `get_connector_events()` so the real
    connector example can report lifecycle events without reading internal
    prefix-store objects.
  - `examples/vllm_turbobus_kv_connector.py` now sends save/restore intent
    through `kv_transfer_params` and builds save/restore summaries from
    `TurboBusConnector` events.
  - Restore summaries now include bytes, layers, ranges, direct/relay chunks,
    timing, and auto-selection fields from connector events.
  - vLLM connector unit tests instantiate `TurboBusConnector` with
    `kv_cache_config`, matching the current vLLM base-class lifecycle.

- Wire daemon transfer reservations into Runtime planning
  - Added `TurboBusDaemonClient` for the local daemon socket protocol.
  - `RuntimeOptions.daemon_socket_path` now makes Runtime register a daemon
    session directly, reserve relay chunks before relay/pool transfers, release
    reservations after wait, and fall back to direct when daemon quota denies
    relay use.
  - `TransferHandle.stats`, `last_daemon_reservation_dict()`, and vLLM restore
    events expose daemon session and reservation fields.
  - vLLM connector config now accepts `turbobus.daemon_socket_path` and
    `turbobus.daemon_max_inflight_chunks`.

- Multi-relay executor behavior
  - Added `test_multi_relay_pool` for direct plus two relay H2D correctness.
  - The test checks returned data, per-relay bytes/chunks, and path stats.
  - `bench_pool_bandwidth` now prints per-relay and per-path stats so server
    benchmark output shows which relay paths were used.

- `6ba90ed Add multi-relay planner coverage`
  - Extended the C++ planner test with two relay paths.
  - H2D asserts direct/relay chunk assignment follows 20/40/10 effective
    bandwidth.
  - D2H asserts direct/relay chunk assignment follows 10/20/30
    direction-specific effective bandwidth.

- Split Runtime plan trace helper from Runtime
  - Added `turbobus/plan_trace.py`.
  - Moved `transfer_plan_to_dict()` out of `turbobus/runtime.py`.
  - Preserved `from turbobus.runtime import transfer_plan_to_dict`.
  - Did not create `profile_cache.py`; the current Runtime profile helpers do
    not justify a new layer yet.

- `cbfdcc2 Document targeted verification policy`
  - Added the verification policy: future code updates should use targeted
    checks based on the files changed instead of running full test, CUDA, and
    vLLM suites by default.

- `Split transfer selector from Runtime`
  - Added `turbobus/transfer_selector.py`.
  - Moved `TransferMode`, `AutoTransferDecision`, and
    `AutoTransferSelector` out of `turbobus/runtime.py`.
  - Preserved public imports from `turbobus.runtime` and `turbobus`.

- `830d137 Document TurboBus roadmap workflow`
  - Added repository roadmap files and AGENTS rules for keeping future coding
    turns on the TurboBus paper reproduction main line.

- `c9cb837 Persist vLLM save intent until block allocation`
  - The vLLM connector now keeps save parameters by request id until enough
    allocated block ids arrive.
  - This makes save metadata less dependent on one ideal scheduler-output
    shape.

- `c9ebc1f Add direction-aware transfer profiling`
  - Native profiling now measures D2H bandwidth.
  - Planner, Runtime, pybind, Python auto selection, benchmarks, and tests now
    carry direction-specific D2H profile fields.

- `86119ce Write structured vLLM sweep case outputs`
  - vLLM connector sweep output is structured for later comparison.

- `2b1597b Enforce daemon session chunk quota`
  - Daemon session quota checks reject excess transfer chunks.

- `ba8c1d0 Share vLLM example CUDA mapping`
  - vLLM examples share CUDA device mapping helpers.

- `7507ca6 Consolidate vLLM connector configuration`
  - Connector configuration was consolidated around shared keys.

## Last Verified Checks

For workload batch client APIs:

```text
python -m unittest discover -s test\python -p "test_model_loading.py" -v
python -m unittest discover -s test\python -p "test_training_offload.py" -v
python -m unittest discover -s test\python -p "test_inference_adapters.py" -v
python -m compileall turbobus\model_loading.py turbobus\training_offload.py turbobus\inference.py test\python\test_model_loading.py test\python\test_training_offload.py test\python\test_inference_adapters.py -q
```

Result: passed.

For vLLM block-id normalization:

```text
python -m unittest discover -s test\python -p "test_vllm_integration.py" -v
python -m unittest discover -s test\python -p "test_vllm_kv_connector.py" -v
python -m unittest discover -s test\python -p "test_vllm_kv_connector_example.py" -v
python -m compileall turbobus\vllm_integration.py test\python\test_vllm_integration.py -q
```

Result: passed.

For vLLM connector example save intent:

```text
python -m unittest discover -s test\python -p "test_vllm_kv_connector_example.py" -v
python -m unittest discover -s test\python -p "test_vllm_kv_connector.py" -v
python -m compileall examples\vllm_turbobus_kv_connector.py test\python\test_vllm_kv_connector_example.py -q
```

Result: passed.

For batch block operations:

```text
python -m unittest discover -s test\python -p "test_offload_store.py" -v
```

Result: 18 tests passed.

```text
python -m unittest discover -s test\python -p "test_model_loading.py" -v
python -m unittest discover -s test\python -p "test_training_offload.py" -v
python -m unittest discover -s test\python -p "test_inference_adapters.py" -v
python -m compileall turbobus\offload_store.py turbobus\__init__.py test\python\test_offload_store.py -q
```

Result: passed.

For reusable offload block-store aliases:

```text
python -m unittest discover -s test\python -p "test_offload_store.py" -v
```

Result: 16 tests passed.

```text
python -m unittest discover -s test\python -p "test_model_loading.py" -v
python -m unittest discover -s test\python -p "test_training_offload.py" -v
python -m unittest discover -s test\python -p "test_inference_adapters.py" -v
python -m compileall turbobus\offload_store.py turbobus\model_loading.py turbobus\training_offload.py turbobus\inference.py test\python\test_offload_store.py test\python\test_model_loading.py test\python\test_training_offload.py test\python\test_inference_adapters.py -q
git diff --check
```

Result: passed.

Additional local checks on the Windows development environment:

```text
python -m unittest discover -s test\python -p "test_daemon_smoke.py" -v
```

Result: 6 tests passed.

```text
python -m compileall benchmarks\daemon_smoke.py test\python\test_daemon_smoke.py -q
git diff --check
```

Result: passed.

Further local checks on the Windows development environment:

```text
python -m unittest discover -s test\python -p "test_daemon_smoke.py" -v
```

Result: 4 tests passed.

```text
python -m compileall benchmarks\daemon_smoke.py test\python\test_daemon_smoke.py -q
git diff --check
```

Result: passed.

On the local Windows development environment:

```text
python -m unittest discover -s test\python -p "test_*.py" -v
```

Result: 96 tests passed, 3 skipped.

```text
python -m compileall turbobus benchmarks examples test\python -q
git diff --check
```

Result: passed.

Additional import check:

```text
from turbobus.runtime import AutoTransferSelector, TransferMode
from turbobus.transfer_selector import AutoTransferSelector, TransferMode
```

Result: passed.

For the plan trace split:

```text
python -m unittest discover -s test\python -p "test_runtime_handle.py" -v
```

Result: 21 tests passed, 2 skipped because PyTorch is not installed locally.

```text
python -m compileall turbobus test\python\test_runtime_handle.py -q
git diff --check
```

Result: passed.

For the multi-relay planner coverage:

```text
git diff --check
```

Result: passed.

```text
cmake -S test/cpp -B build-test
```

Result: not run locally because `cmake` is not installed in this Windows
environment.

```text
python - <<'PY'
def counts(bws, chunks):
    scores = [0 for _ in bws]
    out = [0 for _ in bws]
    for _ in range(chunks):
        i = min(range(len(bws)), key=lambda j: scores[j] / bws[j])
        out[i] += 1
        scores[i] += 1
    return out
print("h2d", counts([20, 40, 10], 14))
print("d2h", counts([10, 20, 30], 12))
PY
```

Result: `h2d [4, 8, 2]`, `d2h [2, 4, 6]`.

For the multi-relay executor behavior:

```text
git diff --check
```

Result: passed.

```text
cmake -S test/cpp -B build-test
```

Result: not run locally because `cmake` is not installed in this Windows
environment.

Local C++/CUDA checks were not run because `cmake` is not installed in this
environment.

For daemon reservation Runtime wiring:

```text
python -m unittest discover -s test\python -p "test_runtime_handle.py" -v
```

Result: 25 tests passed, 2 skipped because PyTorch is not installed locally.

```text
python -m unittest discover -s test\python -p "test_daemon_*.py" -v
```

Result: 5 tests passed, 2 skipped because Windows does not expose Unix domain
socket support in this environment.

```text
python -m unittest discover -s test\python -p "test_vllm_kv_connector.py" -v
```

Result: 31 tests passed.

```text
python -m compileall turbobus test\python\test_runtime_handle.py test\python\test_daemon_socket.py test\python\test_vllm_kv_connector.py -q
git diff --check
```

Result: passed.

For vLLM connector lifecycle cleanup:

```text
python -m unittest discover -s test\python -p "test_vllm_kv_connector.py" -v
```

Result: 32 tests passed.

```text
python -m unittest discover -s test\python -p "test_vllm_kv_connector_sweep.py" -v
```

Result: 6 tests passed.

```text
python -m compileall turbobus\vllm_kv_connector.py examples\vllm_turbobus_kv_connector.py test\python\test_vllm_kv_connector.py -q
git diff --check
```

Result: passed.

For model loading workload API and benchmark:

```text
python -m unittest discover -s test\python -p "test_model_loading.py" -v
```

Result: 5 tests passed.

```text
python -m unittest discover -s test\python -p "test_offload_store.py" -v
```

Result: 15 tests passed.

```text
python -m compileall turbobus\model_loading.py benchmarks\model_loading.py test\python\test_model_loading.py -q
```

Result: passed.

For training offload bucket API and benchmark:

```text
python -m unittest discover -s test\python -p "test_training_offload.py" -v
```

Result: 5 tests passed.

```text
python -m unittest discover -s test\python -p "test_offload_store.py" -v
```

Result: 15 tests passed.

```text
python -m compileall turbobus\training_offload.py benchmarks\training_offload.py test\python\test_training_offload.py -q
```

Result: passed.

For daemon shared profile cache:

```text
python -m unittest discover -s test\python -p "test_runtime_handle.py" -v
```

Result: 29 tests passed, 2 skipped because PyTorch is not installed locally.

```text
python -m unittest discover -s test\python -p "test_daemon_*.py" -v
```

Result: 7 tests passed, 3 skipped because Windows does not expose Unix domain
socket support in this environment.

```text
python -m compileall turbobus\runtime.py turbobus\daemon test\python\test_runtime_handle.py test\python\test_daemon_state.py test\python\test_daemon_socket.py -q
git diff --check
```

Result: passed.

For daemon-aware benchmark options:

```text
python -m unittest discover -s test\python -p "test_benchmark_daemon_support.py" -v
```

Result: 3 tests passed.

```text
python -m compileall benchmarks\daemon_support.py benchmarks\bandwidth_pool.py benchmarks\model_loading.py benchmarks\training_offload.py test\python\test_benchmark_daemon_support.py -q
git diff --check
```

Result: passed.

## Known Server Follow-Up

After C++/CUDA/pybind edits, reinstall before server tests:

```bash
pip uninstall -y turbobus
rm -rf build build-test build-temp *.egg-info turbobus/_turbobus*.so
pip install -e .
```

Then run native and vLLM checks on target GPU 6 with relay GPU 5.

## Next Task

Start with the task under `## Current` in `docs/NEXT_STEPS.md`: add
production-shaped offload clients for the three paper workloads.
