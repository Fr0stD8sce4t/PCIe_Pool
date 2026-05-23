# TurboBus Progress

## Current State

The project direction is still the paper-reproduction rewrite, and the code
now has its first structural cut:

- shared protocol types for transfer mode and daemon requests live in
  `turbobus/schema.py`;
- native runtime support moved into `turbobus/runtime_engine.py`;
- `turbobus/runtime.py` is thinner and now acts more like a facade over the
  runtime engine and daemon control path;
- daemon protocol definitions are no longer duplicated in the daemon package.
- planner model types now live in `turbobus/planner_types.py`, and
  `transfer_plan_to_dict` accepts them directly.
- `turbobus/planner_engine.py` can now build direct, relay, and pooled chunk
  plans without depending on CUDA-specific native objects.

## What Was Updated

- `turbobus/schema.py` now owns the shared transfer and daemon protocol types.
- `turbobus/runtime_engine.py` now owns runtime options, transfer handles, and
  helper logic for native transfer validation and daemon profile conversion.
- `turbobus/runtime.py`, `turbobus/transfer_selector.py`, and
  `turbobus/daemon/protocol.py` now import those shared types instead of
  duplicating them.
- Added a focused protocol serialization test at
  `test/python/test_schema.py`.
- Added planner model tests at `test/python/test_planner_types.py`.
- Added planner engine tests at `test/python/test_planner_engine.py`.

## Immediate Goal

Start the planner and scheduler model on top of the new shared types:

1. make scheduler policy consume `PlannerTransferPlan` and `PlannerLease`;
2. move relay fallback and denial reasons out of the runtime hot path;
3. keep the runtime facade thin so later daemon-managed execution can replace
   the old single-process assumptions.

## Verification

This turn exercised the new shared types and the runtime facade. The relevant
checks were:

```text
$env:PYTHONPATH='.'; python test/python/test_schema.py
$env:PYTHONPATH='.'; python test/python/test_daemon_state.py
$env:PYTHONPATH='.'; python test/python/test_daemon_socket.py
$env:PYTHONPATH='.'; python test/python/test_runtime_handle.py
$env:PYTHONPATH='.'; python test/python/test_offload_store.py
$env:PYTHONPATH='.'; python test/python/test_inference_adapters.py
$env:PYTHONPATH='.'; python test/python/test_model_loading.py
$env:PYTHONPATH='.'; python test/python/test_training_offload.py
$env:PYTHONPATH='.'; python test/python/test_vllm_integration.py
$env:PYTHONPATH='.'; python test/python/test_vllm_connector.py
$env:PYTHONPATH='.'; python test/python/test_vllm_kv_connector.py
$env:PYTHONPATH='.'; python test/python/test_planner_types.py
$env:PYTHONPATH='.'; python test/python/test_planner_engine.py
```

## Remaining Work

- make daemon-side scheduling consume the new planner model;
- keep the new docs as the source of truth for the rewrite plan;
- avoid reintroducing the old single-process assumptions in future code.
