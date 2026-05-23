# TurboBus Progress

## Current State

The project direction is still the paper-reproduction rewrite. The current
code refactor layer is now organized around daemon-managed planning,
backend-facing CUDA execution, adapter-owned framework logic, and explicit
transfer request objects:

- shared protocol types for transfer mode and daemon requests live in
  `turbobus/schema.py`;
- native runtime support lives in `turbobus/runtime_engine.py`;
- `turbobus/runtime.py` is a Python execution facade over the runtime engine,
  CUDA backend facade, and daemon control path;
- planner model types live in `turbobus/planner_types.py`, and
  `transfer_plan_to_dict` accepts them directly;
- `turbobus/planner_engine.py` builds direct, relay, and pooled chunk plans
  without depending on CUDA-specific native objects;
- `turbobus/daemon/scheduler.py` converts daemon session/profile/quota state
  into `PlannerTransferPlan`, `PlannerLease`, and `PlannerStats`;
- the daemon accepts `PLAN_TRANSFER` requests, commits scheduler leases as
  releasable reservations, and returns direct fallback plans when relay planning
  cannot be approved;
- `Runtime` prefers daemon-issued transfer plans, but still falls back to the
  older reserve-only daemon path for compatibility;
- `turbobus.backends.cuda.CudaNativeBackend` owns current CUDA native runtime
  creation, transfer-mode translation, and native range construction;
- `turbobus.transfer` defines `TransferRequest`, `TransferRange`, and
  `TransferDirection` as the shared request shape for runtime, daemon, adapter,
  and future worker code;
- `turbobus.schema` now also defines the daemon protocol baseline message
  shapes for job identity, buffer registration, lease tokens, transfer status,
  and cleanup requests;
- `turbobus.adapters` owns the framework-facing implementations for inference
  slots, vLLM, vLLM connector entry points, model loading, and training offload;
- old root-level framework modules remain as compatibility aliases to the
  adapter modules.

## What Was Updated

- `turbobus/schema.py` owns the shared transfer and daemon protocol types.
- `turbobus/runtime_engine.py` owns runtime options, transfer handles, native
  transfer validation helpers, and daemon profile conversion helpers.
- `turbobus/runtime.py`, `turbobus/transfer_selector.py`, and
  `turbobus/daemon/protocol.py` import shared types instead of duplicating
  them.
- `turbobus/daemon/scheduler.py` is the daemon-side scheduling policy module.
- `turbobus/daemon/server.py` and `turbobus/daemon/client.py` support
  daemon-owned plan issuance through `PLAN_TRANSFER`.
- `turbobus/backends/base.py` and `turbobus/backends/cuda.py` define the
  backend facade for the current native CUDA runtime.
- `turbobus/runtime.py` submits request-shaped daemon plans and uses transfer
  requests for contiguous and range transfers.
- `turbobus/transfer.py` was added for transfer request/range/direction
  validation and daemon payload serialization.
- `turbobus/daemon/client.py` exposes `plan_transfer_request`; the older
  `plan_transfer` helper now builds the same request object.
- Added daemon protocol baseline message types and validation tests for job,
  buffer, lease, transfer status, and cleanup records.
- `TurboBusDaemon` now tracks registered jobs and buffers and can clean up job,
  buffer, session, and reservation state through the new cleanup request shape.
- `TurboBusDaemon` now creates transfer status records for daemon-issued plans,
  exposes `TRANSFER_STATUS` queries and updates, and marks relay-backed
  transfers complete when their reservations are released.
- `TurboBusDaemonClient` now exposes the same transfer-status request path over
  the daemon socket, and the socket round-trip test checks submitted-to-complete
  status transitions after releasing a relay reservation.
- `turbobus/adapters/*.py` now owns framework-facing implementation code.
- `turbobus/inference.py`, `turbobus/vllm.py`, `turbobus/vllm_connector.py`,
  `turbobus/vllm_integration.py`, `turbobus/vllm_kv_connector.py`,
  `turbobus/model_loading.py`, and `turbobus/training_offload.py` are
  compatibility aliases to the adapter modules.
- `turbobus/__init__.py` re-exports framework-facing objects through
  `turbobus.adapters`.
- `pyproject.toml` packages `turbobus.backends` and `turbobus.adapters`.
- Added focused protocol, planner, scheduler, backend, adapter, and transfer
  request tests.
- Extended daemon state and runtime handle tests for daemon-issued plans, direct
  fallback, backend facade construction, and request-shaped daemon planning.

## Immediate Goal

The current refactor layer has reached the intended boundary for the next
phase:

1. runtime consumes daemon plans instead of owning relay choice;
2. native CUDA execution is behind the backend facade while preserving direct,
   relay, and pooled behavior;
3. framework adapter implementations are under `turbobus.adapters`;
4. root-level framework modules remain compatible through aliases;
5. explicit transfer request objects are available for runtime, daemon, adapter,
   and future worker paths.
6. daemon protocol baseline message shapes now exist for job identity, buffer
   registration, lease tokens, transfer status, and cleanup.
7. daemon state now tracks jobs and buffers and can clean them up through the
   new protocol path.
8. daemon-issued plans now have transfer ids and status records that can be
   queried, updated, and completed when relay reservations are released.

The next immediate goal is the daemon protocol baseline: job registration,
buffer registration, transfer status, lease-token records, and cleanup
messages. This starts the privileged-daemon work, but does not add worker
execution yet.

## Verification

The current refactor checks are:

```text
$env:PYTHONPATH='.'; python test/python/test_schema.py
$env:PYTHONPATH='.'; python test/python/test_transfer.py
$env:PYTHONPATH='.'; python test/python/test_adapters_package.py
$env:PYTHONPATH='.'; python test/python/test_backend_cuda.py
$env:PYTHONPATH='.'; python test/python/test_daemon_scheduler.py
$env:PYTHONPATH='.'; python test/python/test_daemon_state.py
$env:PYTHONPATH='.'; python test/python/test_daemon_socket.py
$env:PYTHONPATH='.'; python test/python/test_runtime_handle.py
$env:PYTHONPATH='.'; python test/python/test_planner_types.py
$env:PYTHONPATH='.'; python test/python/test_planner_engine.py
$env:PYTHONPATH='.'; python test/python/test_inference_adapters.py
$env:PYTHONPATH='.'; python test/python/test_offload_store.py
$env:PYTHONPATH='.'; python test/python/test_model_loading.py
$env:PYTHONPATH='.'; python test/python/test_training_offload.py
$env:PYTHONPATH='.'; python test/python/test_vllm_connector.py
$env:PYTHONPATH='.'; python test/python/test_vllm_integration.py
$env:PYTHONPATH='.'; python test/python/test_vllm_kv_connector.py
$env:PYTHONPATH='.'; python test/python/test_vllm_kv_connector_sweep.py
```

## Remaining Work

- define daemon protocol records for job identity, buffer registration,
  lease tokens, and worker-facing cleanup;
- add daemon state and validation tests for lease token handling;
- keep the daemon plan path as the control-plane entry point for future worker
  execution;
- split the current native CUDA execution path further only when worker/helper
  execution needs it;
- avoid reintroducing local relay selection in runtime or framework adapters.
