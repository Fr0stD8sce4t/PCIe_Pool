# TurboBus Progress

## Current State

The project direction is still the paper-reproduction rewrite, and the code
now has its first daemon-managed planning cut:

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
- `turbobus/daemon/scheduler.py` now converts daemon session/profile/quota
  state into `PlannerTransferPlan`, `PlannerLease`, and `PlannerStats`.
- the daemon now accepts `PLAN_TRANSFER` requests, commits scheduler leases as
  releasable reservations, and returns direct fallback plans when relay planning
  cannot be approved.
- `Runtime` now prefers daemon-issued plans when the connected daemon supports
  `PLAN_TRANSFER`, while keeping the older `RESERVE_TRANSFER` path as a
  compatibility fallback.
- `turbobus.backends` now exists as the first backend-facing package, and the
  current CUDA native runtime is reached through `CudaNativeBackend` instead of
  direct `_turbobus.Runtime(...)` calls in `Runtime`.
- `turbobus.adapters` now exists as the framework-facing package boundary for
  inference slots, vLLM, vLLM connector entry points, model loading, and
  training offload while old root-level imports remain compatible.

## What Was Updated

- `turbobus/schema.py` now owns the shared transfer and daemon protocol types.
- `turbobus/runtime_engine.py` now owns runtime options, transfer handles, and
  helper logic for native transfer validation and daemon profile conversion.
- `turbobus/runtime.py`, `turbobus/transfer_selector.py`, and
  `turbobus/daemon/protocol.py` now import those shared types instead of
  duplicating them.
- `turbobus/daemon/scheduler.py` was added as the first daemon-side scheduling
  policy module.
- `turbobus/daemon/server.py` and `turbobus/daemon/client.py` now support
  daemon-owned plan issuance through `PLAN_TRANSFER`.
- `turbobus/backends/base.py` and `turbobus/backends/cuda.py` now define the
  backend facade for the current native CUDA runtime.
- `turbobus/runtime.py` now binds the native extension through that backend and
  asks it to create the native runtime, translate transfer modes, and build
  native ranges.
- `pyproject.toml` now packages `turbobus.backends`.
- `turbobus/adapters/*.py` now provides the adapter-facing import paths.
- `turbobus/__init__.py` now re-exports framework-facing objects through
  `turbobus.adapters` instead of importing each old flat module directly.
- `pyproject.toml` now packages `turbobus.adapters`.
- Added a focused protocol serialization test at
  `test/python/test_schema.py`.
- Added planner model tests at `test/python/test_planner_types.py`.
- Added planner engine tests at `test/python/test_planner_engine.py`.
- Added daemon scheduler tests at `test/python/test_daemon_scheduler.py`.
- Added backend facade tests at `test/python/test_backend_cuda.py`.
- Added adapter package boundary tests at `test/python/test_adapters_package.py`.
- Extended daemon state and runtime handle tests for daemon-issued plans and
  direct fallback, plus runtime construction through the backend facade.

## Immediate Goal

Finish the refactor layer that separates planning/control from execution:

1. keep runtime as an execution facade that consumes daemon plans instead of
   owning relay choice;
2. keep native CUDA execution behind the backend facade while preserving direct,
   relay, and pooled behavior;
3. keep framework adapter entry points behind adapter-facing modules without
   breaking the current public imports;
4. introduce explicit client/runtime transfer request objects so later worker
   execution can consume the same transfer shape;
5. keep framework adapter code outside daemon scheduling and backend execution
   paths.

## Verification

The current refactor checks are:

```text
$env:PYTHONPATH='.'; python test/python/test_schema.py
$env:PYTHONPATH='.'; python test/python/test_adapters_package.py
$env:PYTHONPATH='.'; python test/python/test_backend_cuda.py
$env:PYTHONPATH='.'; python test/python/test_daemon_scheduler.py
$env:PYTHONPATH='.'; python test/python/test_daemon_state.py
$env:PYTHONPATH='.'; python test/python/test_daemon_socket.py
$env:PYTHONPATH='.'; python test/python/test_runtime_handle.py
$env:PYTHONPATH='.'; python test/python/test_planner_types.py
$env:PYTHONPATH='.'; python test/python/test_planner_engine.py
```

## Remaining Work

- split the current native CUDA execution path behind backend-facing Python
  interfaces further as worker execution grows;
- keep the daemon plan path as the control-plane entry point for future worker
  execution;
- separate framework adapters from the flat package root once their imports can
  move from compatibility wrappers into owned adapter modules;
- introduce client/runtime transfer request objects that can be shared with the
  future worker path;
- avoid reintroducing local relay selection in runtime or framework adapters.
