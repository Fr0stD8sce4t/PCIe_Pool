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
- `TurboBusDaemon` now issues lease tokens for relay reservations and planned
  relay leases, stores them internally, invalidates them on release or cleanup,
  and validates them through a `VALIDATE_LEASE` request for future worker/helper
  use.
- `TransferRequest` can now send registered buffer ids to the daemon, and
  `VALIDATE_LEASE` can check those buffer ids against the lease token plus the
  owning job and session.
- Worker-facing authorization messages now exist in the daemon protocol, and
  `AUTHORIZE_WORKER_TRANSFER` returns a checked transfer context with source
  and destination buffer registrations, direction, relay GPU, and ranges.
- `turbobus.worker` now has a helper skeleton that parses daemon authorization
  payloads into worker transfer requests and reports unsupported execution
  without pretending data movement happened.
- `turbobus.worker` now also has a client-side authorization helper that calls
  `authorize_worker_transfer` on the daemon client, builds a
  `WorkerTransferRequest` from the daemon response, and keeps execution on the
  explicit unsupported path until worker IPC movement exists.
- `turbobus.worker` now has a status reporter that maps worker unsupported,
  failed, and completed outcomes into daemon `TRANSFER_STATUS` updates, keeping
  transfer state owned by the daemon even before real worker movement exists.
- `turbobus.daemon.topology` now defines backend-neutral GPU, PCIe path, and
  scale-up fabric link inventory records, plus an injectable static topology
  provider.
- `TurboBusDaemon` now owns a topology provider and exposes its resource
  inventory through `GET_INVENTORY`; `TurboBusDaemonClient.get_inventory()`
  can query the same control-plane path.
- `TurboBusDaemon.plan_transfer` now uses daemon-owned inventory to filter
  relay eligibility before profile lookup and scheduler input, while preserving
  cached profile lookup and direct fallback behavior.
- daemon plan responses now include a `planning` block with the profile key,
  requested relays, eligible relays, and filtered relays with inventory-derived
  reasons.
- daemon profile/describe payloads now include `system_cleanup_events` for
  stale sessions, closed sessions, and canceled reservations, and planned
  transfers canceled by session cleanup are marked `canceled`.
- `TurboBusDaemonClient.describe()` now queries the daemon profile/describe
  path, making cleanup observability available through the socket client used
  by runtime and future worker code.
- `TurboBusDaemonClient.cleanup()` now wraps the existing daemon `CLEANUP`
  request, and socket coverage checks that client-driven session cleanup records
  both the client cleanup event and daemon-generated session/reservation cleanup
  events.
- `WorkerTransferCleanupCoordinator` now lets worker helper code request daemon
  cleanup after authorization or execution failure, and
  `WorkerTransferClient.submit_report_and_cleanup()` keeps unsupported worker
  execution on the daemon-owned status and cleanup paths.
- `WorkerTransferLifecycleRecord` now serializes the worker authorization
  request, daemon-approved transfer context, status update and response,
  cleanup target and response, final outcome, and error for future helper
  process handoff.
- `WorkerTransferService` now exposes an in-process worker helper service
  boundary that accepts a worker authorization request and returns serialized
  lifecycle records across unsupported execution, authorization denial, status
  failure, and cleanup failure paths.
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
9. daemon-issued relay reservations now have lease tokens that can be validated
   without exposing the daemon's internal lease-token table through profile
   snapshots.
10. lease validation now ties relay permission to registered transfer buffers,
    job ownership, and session ownership.
11. worker/helper authorization can be requested through the daemon client
    without adding worker execution, so the next data-plane layer has a
    daemon-approved input shape.
12. worker/helper code now has a package boundary and a no-op unsupported
    executor for daemon-authorized transfer contexts.
13. worker/helper outcomes can be reported back through the daemon status path,
    so future worker execution can update daemon-owned transfer status instead
    of maintaining local-only state.
14. daemon-owned resource inventory now has a backend-neutral shape for GPUs,
    PCIe paths, and scale-up fabric links, with static injection for tests and
    a daemon control-plane query path.
15. daemon planning now starts relay eligibility from the daemon-owned
    inventory, so disabled or missing fabric paths are not handed to the
    scheduler as usable relay paths.
16. daemon plan responses now surface inventory-derived planning metadata, so
    clients and tests can see which relays were eligible or filtered without
    changing transfer execution.
17. daemon-generated cleanup outcomes are now visible through
    `system_cleanup_events`, including stale session cleanup and reservation
    cancellation caused by session cleanup.
18. daemon profile/describe reporting is now reachable from
    `TurboBusDaemonClient.describe()`, with socket coverage for cleanup
    observability.
19. client-driven cleanup is now reachable from `TurboBusDaemonClient.cleanup()`
    and covered through the daemon socket path.
20. worker-side cleanup coordination can now report failed or unsupported helper
    transfers and ask the daemon to reclaim the matching reservation or session.
21. worker request lifecycle records now make status and cleanup decisions
    explicit and serializable for future helper processes.
22. an in-process worker helper service skeleton now returns serialized
    lifecycle records without adding sockets, IPC, or real data movement.

The next immediate goal is to add worker service payload parsing helpers so a
future worker process can accept plain daemon JSON dictionaries and convert them
into `WorkerTransferAuthorizationRequest` objects before calling the service.
This should stay inside in-process payload validation and should not add CUDA
IPC, sockets, real data movement, or hardware discovery.

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
$env:PYTHONPATH='.'; python test/python/test_worker_helper.py
```

## Remaining Work

- define daemon protocol records for job identity, buffer registration,
  lease tokens, and worker-facing cleanup;
- add a daemon-owned resource and topology inventory skeleton that later
  scheduling can consume; done as the first inventory cut;
- connect daemon scheduling inputs to the inventory skeleton without changing
  direct fallback behavior; done as the first scheduling-input cut;
- surface inventory-derived planning metadata in daemon responses; done as the
  first observability cut;
- add daemon cleanup observability for stale sessions and canceled
  reservations; done through `system_cleanup_events`;
- add a daemon client helper for profile/describe reporting; done through
  `TurboBusDaemonClient.describe()`;
- add a daemon client helper for `CLEANUP` requests; done through
  `TurboBusDaemonClient.cleanup()`;
- add worker-side cleanup coordination for failed or unsupported helper
  transfers; done through `WorkerTransferCleanupCoordinator`;
- add worker request lifecycle records for future helper processes; done
  through `WorkerTransferLifecycleRecord`;
- add an in-process worker helper service skeleton that returns lifecycle
  records; done through `WorkerTransferService`;
- add worker service payload parsing helpers for future worker process
  boundaries;
- keep the daemon plan path as the control-plane entry point for future worker
  execution;
- split the current native CUDA execution path further only when worker/helper
  execution needs it;
- avoid reintroducing local relay selection in runtime or framework adapters.
