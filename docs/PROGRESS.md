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
- worker endpoint observability requests now record a separate event stream
  with request and response byte counts while keeping the returned snapshot
  payload stable.
- `turbobus.worker.transport` now defines a transport protocol and loopback
  adapter for the in-process worker endpoint without changing request or
  observability semantics.
- `turbobus.worker.transport` now also provides a Unix socket transport shell
  that forwards worker and observability messages through the same endpoint
  behavior.
- `turbobus.worker.process` now provides the worker helper-process entrypoint,
  and `python -m turbobus.worker` can serve the in-process worker endpoint over
  the Unix socket transport.
- worker process coverage now includes a subprocess smoke path that launches
  `python -m turbobus.worker` and exercises worker request plus observability
  round-trips over the Unix socket transport.
- daemon control plane now exposes backend-neutral relay discovery snapshots
  with per-relay eligibility, quota, active sessions, active reservations, and
  redacted lease bookkeeping across jobs.
- daemon-side expired relay leases now reap active reservations and relay quota
  before relay discovery, planning, validation, and description readouts.
- explicit expired-lease reaping now reaches the daemon through
  `TurboBusDaemonClient.reap_expired_leases()` and the socket control path.

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
- worker service payload parsing helpers now convert plain daemon JSON
  dictionaries into `WorkerTransferAuthorizationRequest` objects and preserve
  serialized lifecycle output from parsed payloads.
- `WorkerServiceRequestEnvelope` and `WorkerServiceResponseEnvelope` now wrap
  successful lifecycle payloads, malformed payload errors, status failures, and
  cleanup failures in one stable in-process response shape.
- `run_worker_service_control_plane_smoke` now wires daemon-owned planning,
  worker service envelope handling, daemon status reporting, and daemon
  reservation cleanup together in process, proving the service boundary can
  fail unsupported execution and reclaim the relay lease without sockets, IPC,
  or real data movement.
- `WorkerBufferHandle`, `WorkerStagingBufferRequirement`,
  `WorkerDataPlaneRequest`, and `WorkerDataPlaneCompletion` now define the
  first worker data-plane request and completion shapes for daemon-approved
  relay execution.
- `WorkerTransferRequest` now derives its data-plane request from the daemon
  worker authorization result and rejects mismatched transfer, lease, relay,
  direction, buffer, or range authority.
- `WorkerStagingPool` and `WorkerStagingSlot` now provide an in-memory worker
  staging-pool skeleton for daemon-approved relay staging slots. The pool can
  allocate, describe, validate, and release slots from `WorkerDataPlaneRequest`
  records while rejecting double-release and mismatched transfer, lease,
  session, job, or relay use.
- worker request lifecycle records now include staging slot allocation and
  release metadata, and `WorkerTransferClient` releases staging slots after
  unsupported execution, daemon status failure, and daemon cleanup failure.
- worker data-plane executors now receive both the daemon-authorized
  `WorkerTransferRequest` and its allocated `WorkerStagingSlot`. The default
  executor still returns explicit unsupported execution without CUDA IPC,
  sockets, real data movement, or hardware discovery.
- `WorkerDataPlaneCompletionEnvelope` now gives the future helper-process
  boundary a completion-specific serialized shape for allocated staging slots,
  worker results, daemon status updates and responses, daemon cleanup responses,
  and staging release records.
- `WorkerServiceResponseEnvelope` now includes that completion envelope
  alongside the full lifecycle payload, so future helper process boundaries can
  read completion, status, cleanup, and staging-release output directly.
- `turbobus.worker.codec` now provides JSON-safe request and response envelope
  encoders/decoders for the future worker helper process boundary while
  staying in process.
- `handle_worker_service_message` now connects the worker message codec to
  `WorkerTransferService.handle_envelope`, returning encoded response messages
  for successful worker requests and parse failures without adding a transport.
- `WorkerServiceEndpoint` now provides a transport-neutral worker service
  endpoint with a single `handle_message` entry point for future socket or IPC
  transports.
- `WorkerEndpointEvent` now records request size, response size, ok/error
  status, final state, and completion presence for each endpoint
  `handle_message` call.
- `WorkerServiceEndpoint.describe()` now summarizes recorded endpoint events
  with total request count, last event, final-state counts, error count, and
  completion count.
- `WorkerServiceEndpoint.clear_events()` now returns the current describe
  snapshot, clears recorded endpoint events, and resets `last_event`.
- `WorkerServiceEndpoint` now accepts an optional `max_events` history limit so
  long-running helper processes can bound retained endpoint event records.
- `WorkerServiceEndpoint.describe()` now reports endpoint configuration fields
  for `max_events`, retained event count, and whether event history is bounded.
- `WorkerServiceEndpoint.event_snapshot()` now returns retained endpoint event
  records as copied dictionaries for future transport observability.
- `WorkerServiceEndpoint.describe()` now includes retained event records under
  the stable `events` field for future transport observability clients.
- `WorkerServiceEndpoint.health_snapshot()` now reports in-process endpoint
  readiness from retained events, and `describe()` includes it under `health`.
- `WorkerServiceEndpoint.metrics_snapshot()` now reports retained request and
  response byte counts, and `describe()` includes it under `metrics`.
- `WorkerServiceEndpoint.observability_snapshot()` now combines `describe()`,
  retained events, health, and metrics under one stable in-process payload.
- `WorkerServiceEndpoint.handle_observability_message()` now records a
  separate observability event stream and `clear_events()` clears both worker
  and observability histories.
- `turbobus.worker.transport` now provides a loopback transport wrapper and a
  transport protocol for the worker service boundary.
- `turbobus.worker.transport` now also provides a Unix socket transport shell
  for the worker service boundary.
- `turbobus.worker.process` now builds the daemon client, worker endpoint, and
  Unix socket transport for a helper-process entrypoint.
- `test/python/test_worker_process.py` now covers the helper-process builder,
  CLI argument parsing, bounded request serving, and a subprocess worker socket
  smoke path.
- worker endpoint observability snapshots now have JSON-safe encode/decode
  helpers for future transport observability clients.
- `TurboBusDaemon.discover_relays()` now reports cross-job relay occupancy,
  target-specific inventory eligibility, quota availability, active
  reservations, and redacted active lease records without exposing lease
  tokens.
- `TurboBusDaemon.reap_expired_leases()` now clears expired relay reservations,
  marks their transfer status canceled, and records lease-expiry cleanup
  events.
- `TurboBusDaemonClient.discover_relays()` now wraps the same daemon request
  shape for future control-plane callers.
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
23. worker service payload parsing now validates plain dictionaries into
    worker authorization requests before they enter the service path.
24. worker service request/response envelopes now provide a stable shape for
    future helper process boundaries without adding sockets or IPC.
25. worker service control-plane smoke coverage now proves a daemon-owned
    planned relay transfer can pass through the service envelope path, report
    unsupported execution as daemon `failed`, and reclaim the daemon
    reservation.
26. worker data-plane request records now capture daemon-approved relay
    execution inputs and completion reports without adding CUDA IPC, sockets,
    real data movement, or hardware discovery.
27. the worker layer now has an in-memory staging-pool skeleton for relay
    staging slots, with validation and release checks but no CUDA IPC, sockets,
    real data movement, or hardware discovery.
28. worker service lifecycle now reserves a staging slot after daemon
    authorization and releases it across unsupported, status-failed, and
    cleanup-failed paths without adding CUDA IPC, sockets, real data movement,
    or hardware discovery.
29. worker data-plane executor calls now receive the allocated staging slot
    alongside the worker request, while preserving the unsupported default
    executor path.
30. worker data-plane completion envelopes now serialize lifecycle completion
    output without losing staging release information on unsupported,
    status-failed, or cleanup-failed paths.
31. worker service response envelopes now carry the completion envelope without
    dropping the existing lifecycle payload.
32. worker process message codecs now round-trip request and response
    envelopes as JSON-safe strings while preserving completion envelopes and
    malformed-message errors.
33. the in-process worker service message handler now decodes encoded requests,
    runs the existing worker service envelope path, and returns encoded
    responses while preserving completion envelopes.
34. a transport-neutral worker service endpoint now wraps the encoded message
    handler behind one `handle_message` entry point without adding sockets or
    IPC.
35. worker endpoint request/response events now record message sizes, final
    states, ok/error status, and completion presence while preserving encoded
    responses.
36. worker endpoint describe snapshots now summarize recorded message events
    without changing encoded responses.
37. worker endpoint event reset now clears recorded events after returning the
    current snapshot.
38. worker endpoint event history limits now keep only the newest retained
    events while preserving `last_event` and encoded responses.
39. worker endpoint describe snapshots now report `max_events`,
    `retained_event_count`, and `history_bounded` without changing encoded
    worker response payloads.
40. worker endpoint event snapshots now expose retained events as copied
    dictionaries without giving callers direct access to the mutable event
    list.
41. worker endpoint describe snapshots now include retained event records under
    a stable `events` field without changing encoded worker response payloads.
42. worker endpoint health snapshots now summarize retained event readiness and
    surface the same health block through `describe()`.
43. worker endpoint metrics snapshots now summarize retained request and
    response byte counts and surface the same metrics block through
    `describe()`.
44. worker endpoint observability snapshots now combine `describe()`, retained
    events, health, and metrics under one stable in-process payload.
45. worker endpoint observability snapshots now round-trip through JSON-safe
    encode/decode helpers without changing encoded worker response payloads.
46. an in-process worker endpoint observability message handler now returns an
    encoded observability snapshot for future socket or IPC observability
    requests without changing worker response payloads.
47. an in-process worker observability request envelope and codec helper now
    let future transports explicitly trigger the endpoint observability
    handler without changing worker response payloads.
48. worker endpoint observability request/response event tracking now records
    observability message sizes separately from the normal worker event stream
    while keeping the returned snapshot payload stable.
49. daemon relay discovery now reports backend-neutral relay eligibility,
    quota availability, cross-job session ownership, active reservations, and
    redacted lease records without changing direct fallback behavior.
50. make the worker service boundary transport-neutral so future socket or IPC
    transports can reuse the in-process helper contract without changing
    authorization, lifecycle, or observability handling.

The next immediate goal is to connect daemon-issued planned transfer metadata
to the worker helper process over the worker socket transport while keeping
execution on the explicit unsupported path until CUDA IPC or another real data
movement path exists.

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
  boundaries; done through `parse_worker_authorization_request_payload`;
- add worker service request/response envelope records; done through
  `WorkerServiceRequestEnvelope` and `WorkerServiceResponseEnvelope`;
- add worker service control-plane smoke coverage for daemon-owned planned
  transfers; done through `run_worker_service_control_plane_smoke`;
- define the first worker data-plane request shape for daemon-approved relay
  execution; done through `WorkerDataPlaneRequest` and related worker schema
  records;
- add an in-memory worker staging-pool skeleton for relay staging slots; done
  through `WorkerStagingPool`;
- wire the in-memory staging pool into the worker service lifecycle; done by
  allocating and releasing staging slots inside `WorkerTransferClient`;
- define the worker data-plane executor interface that receives both the worker
  request and allocated staging slot; done in `WorkerTransferUnsupportedExecutor`
  and the worker lifecycle call path;
- add a worker data-plane completion envelope for future helper process
  boundaries; done through `WorkerDataPlaneCompletionEnvelope`;
- thread the worker data-plane completion envelope through worker service
  responses; done through `WorkerServiceResponseEnvelope.completion`;
- add a minimal worker process message codec for request and response
  envelopes; done through `turbobus.worker.codec`;
- add an in-process worker service message handler that uses the codec before
  any socket or IPC transport exists; done through
  `handle_worker_service_message`;
- add a transport-neutral worker service endpoint object for future socket or
  IPC transports; done through `WorkerServiceEndpoint`;
- add worker endpoint request/response event records for observability around
  encoded message handling; done through `WorkerEndpointEvent`;
- add a worker endpoint describe snapshot for recorded message events; done
  through `WorkerServiceEndpoint.describe()`;
- add a worker endpoint event reset helper; done through
  `WorkerServiceEndpoint.clear_events()`;
- add an optional worker endpoint event history limit; done through
  `WorkerServiceEndpoint(max_events=...)`;
- add endpoint configuration fields to worker endpoint describe snapshots; done
  through `WorkerServiceEndpoint.describe()`;
- add an in-process endpoint event snapshot helper that exposes retained events
  without giving callers direct access to the mutable event list; done through
  `WorkerServiceEndpoint.event_snapshot()`;
- include retained endpoint event snapshots in `WorkerServiceEndpoint.describe()`
  under a stable field for future transport observability clients; done through
  the `events` field;
- add an in-process worker endpoint health snapshot for future transport
  observability clients; done through `WorkerServiceEndpoint.health_snapshot()`;
- add an in-process worker endpoint metrics snapshot for retained request and
  response byte counts; done through `WorkerServiceEndpoint.metrics_snapshot()`;
- add a combined worker endpoint observability snapshot for future transport
  observability clients; done through
  `WorkerServiceEndpoint.observability_snapshot()`;
- add a JSON-safe worker endpoint observability payload codec for future
  transport observability clients; done through
  `encode_worker_observability_snapshot` and
  `decode_worker_observability_snapshot`;
- add an in-process worker endpoint observability message handler for future
  transport observability clients; done through
  `WorkerServiceEndpoint.handle_observability_message()`;
- add a tiny worker observability request envelope and codec helper so future
  socket or IPC transports can trigger the new endpoint handler explicitly;
  done through `WorkerServiceObservabilityRequestEnvelope` and the worker
  observability codec helpers;
- expose expired relay lease reaping through the daemon client/socket control
  path so external callers can trigger the same cleanup without relying on
  client release or session close;
- keep the daemon plan path as the control-plane entry point for future worker
  execution;
- split the current native CUDA execution path further only when worker/helper
  execution needs it;
- avoid reintroducing local relay selection in runtime or framework adapters.
