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
- `turbobus.backends.cuda.CudaNativeBackend` can now convert daemon-issued
  planner payloads into native `TransferPlan` objects and submit them through
  exact-plan runtime methods;
- `turbobus.transfer` defines `TransferRequest`, `TransferRange`, and
  `TransferDirection` as the shared request shape for runtime, daemon, adapter,
  and future worker code;
- `turbobus.schema` now also defines the daemon protocol baseline message
  shapes for job identity, buffer registration, lease tokens, transfer status,
  and cleanup requests;
- buffer registration now carries worker-visible handle metadata for shared
  pinned CPU buffers and CUDA IPC target GPU buffers;
- `turbobus.client` now provides a TurboBus-owned shared CPU buffer allocator
  that creates cross-process shared-memory source buffers and emits
  daemon-ready `shared_pinned_cpu` registrations;
- `turbobus.backends.cuda.CudaNativeBackend` now exposes CUDA
  host-register/unregister calls for registering opened shared-memory views in
  the process that will issue CUDA copies;
- `turbobus.worker.resources` now opens daemon-authorized shared CPU source
  handles inside the worker/helper data-plane lifecycle and registers the
  opened mapping with CUDA before executor invocation;
- client-side CUDA IPC target buffers can now export a target device pointer
  into daemon-ready `cuda_ipc_device` metadata, and worker resources can open
  and close that CUDA IPC target pointer before bound executor invocation;
- `turbobus.worker.cuda_executor.CudaWorkerExecutor` now runs the first narrow
  daemon-authorized CUDA worker path: shared CPU source to worker-owned relay
  staging to CUDA IPC target GPU for H2D relay chunks;
- the worker helper process now uses the CUDA worker executor and resource
  binder by default, so the helper-process boundary no longer defaults to the
  unsupported executor when serving real requests;
- `turbobus.client_transfer.WorkerManagedTransferClient` now connects the
  first client-to-daemon-to-worker call for H2D relay transfers using shared
  CPU source buffers, CUDA IPC target buffers, daemon-issued relay leases,
  worker execution, daemon status reporting, and relay reservation release on
  completion;
- the worker-managed client call can now cross the helper-process request
  boundary through a completion-only worker service envelope, so a socket
  helper can execute the daemon-authorized request without returning an
  in-process lifecycle record;
- `turbobus.verification` now provides the CUDA-server verification entry
  point for the first real worker-managed H2D relay path. It starts daemon and
  worker helper sockets, runs the worker-managed relay transfer with shared CPU
  source memory and a CUDA IPC target tensor, verifies the target bytes, and
  asserts daemon reservation release;
- `WorkerManagedTransferClient` now submits the daemon-issued relay chunks to
  the worker/helper instead of reusing the original client request range. The
  current H2D worker path rejects direct or mixed pooled daemon plans and
  releases the relay reservation before returning the error;
- daemon worker authorization now stores and returns the exact daemon transfer
  plan, derives worker relay chunk ranges from that stored plan, and rejects
  worker requests whose supplied ranges do not match the daemon plan;
- `CudaWorkerExecutor` now requires a daemon-issued plan in the worker
  data-plane request and builds its native H2D relay plan from the authorized
  daemon-plan chunks instead of reconstructing a relay plan from worker ranges;
- the worker-managed H2D path now accepts a daemon-issued single-relay pool
  plan. The daemon still authorizes only the relay chunks for the lease, while
  the worker CUDA executor submits the complete daemon plan so direct and relay
  H2D chunks can execute together through the native exact-plan path;
- `turbobus.adapters` owns the framework-facing implementations for inference
  slots, vLLM, vLLM connector entry points, model loading, and training offload;
- old root-level framework modules remain as compatibility aliases to the
  adapter modules.
- `turbobus.worker.transport` now keeps only the Unix socket helper path that
  forwards worker messages through the endpoint behavior.
- `turbobus.worker.process` now provides the worker helper-process entrypoint,
  and `python -m turbobus.worker` can serve the in-process worker endpoint over
  the Unix socket transport.
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
  boundary that accepts a worker authorization request and keeps lifecycle
  records internal for status, cleanup, and staging decisions.
- worker service payload parsing helpers now convert plain daemon JSON
  dictionaries into `WorkerTransferAuthorizationRequest` objects before
  helper execution.
- `WorkerServiceRequestEnvelope` and `WorkerServiceResponseEnvelope` now wrap
  successful completion output, malformed payload errors, status failures, and
  cleanup failures in one stable helper response shape.
- Removed the smoke-only `run_worker_service_control_plane_smoke` helper and
  its tests so the worker package no longer preserves an unsupported
  control-plane round trip as a product path.
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
- `WorkerServiceResponseEnvelope` now carries only the completion envelope,
  final state, and error, so the helper process boundary no longer serializes
  the full unsupported lifecycle record.
- `turbobus.worker.codec` now provides JSON-safe request and response envelope
  encoders/decoders for the future worker helper process boundary while
  staying in process.
- `handle_worker_service_message` now connects the worker message codec to
  `WorkerTransferService.handle_envelope`, returning encoded response messages
  for successful worker requests and parse failures without adding a transport.
- `WorkerServiceEndpoint` now provides a transport-neutral worker service
  endpoint with a single `handle_message` entry point for future socket or IPC
  transports.
- `turbobus.worker.transport` now provides the Unix socket helper-process
  transport for the worker service boundary.
- `turbobus.worker.process` now builds the daemon client, worker endpoint, and
  Unix socket transport for a helper-process entrypoint.
- `test/python/test_worker_process.py` now covers the helper-process builder,
  CLI argument parsing, and bounded request serving.
- Removed worker endpoint observability/event-history scaffolding from
  `turbobus.worker.endpoint`, `turbobus.worker.codec`, and
  `turbobus.worker.transport`; the worker helper boundary is now a single
  request/response `handle_message` path for daemon-approved transfer
  execution.
- Removed the worker loopback transport and transport protocol wrapper so
  `turbobus.worker.transport` no longer carries an extra in-process transport
  abstraction that is not needed for the real helper-process data path.
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
- Added exact-plan runtime/backend coverage showing that daemon plans are
  submitted directly instead of falling back to native runtime replanning.
- Added schema, daemon, and worker coverage for carrying shared pinned CPU and
  CUDA IPC buffer-handle metadata into worker data-plane requests.
- Added client-buffer and backend-facade coverage for shared-memory CPU buffer
  allocation, daemon registration payloads, cross-process reopening, and CUDA
  host-register hook delegation.
- Added worker-helper coverage for binding a real shared CPU source before
  bound executor invocation, unregistering it after execution, and reporting
  daemon-owned failure when resource binding cannot open the authorized handle.
- Added backend, client, and worker coverage for exporting CUDA IPC target
  handles, registering them with the daemon, opening them in the worker, and
  closing the opened device pointer after executor invocation.
- Added worker CUDA executor coverage for converting authorized H2D relay
  chunks into an exact native relay plan, initializing a worker-local CUDA
  runtime, waiting for transfer completion, and returning daemon status
  metadata.
- Added worker-managed client coverage showing job and buffer registration,
  daemon planning, worker authorization, worker completion reporting, final
  daemon status lookup, and relay reservation release in one call.
- Added worker-managed client coverage for an envelope-style worker helper
  client and Unix socket helper boundary. The Windows local run skips the
  Unix socket case, but the envelope path verifies completion-only helper
  output and daemon reservation release.
- Added `test/python/test_verification.py` coverage for the verification
  daemon setup. It confirms the server verifier seeds a relay-capable topology
  and profile that produce a relay plan and lease before CUDA hardware is
  required.
- Added worker-managed client coverage proving that worker authorization uses
  daemon plan chunks, and that an unsupported mixed pool plan releases its
  daemon relay reservation.
- Added daemon, worker-managed client, and CUDA worker executor coverage for
  daemon-stored transfer plans flowing through worker authorization into the
  data-plane executor, with client-supplied worker ranges no longer acting as
  authority.
- Added worker CUDA executor and worker-managed client coverage for
  daemon-issued H2D pool plans that contain both direct chunks and the leased
  relay chunks.

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
22. an in-process worker helper service skeleton keeps lifecycle records
    internal and returns completion envelopes without adding sockets, IPC, or
    real data movement.
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
31. worker service response envelopes now carry the completion envelope,
    final state, and error without serializing the full lifecycle payload.
32. worker process message codecs now round-trip request and response
    envelopes as JSON-safe strings while preserving completion envelopes and
    malformed-message errors.
33. the in-process worker service message handler now decodes encoded requests,
    runs the existing worker service envelope path, and returns encoded
    responses while preserving completion envelopes.
34. a transport-neutral worker service endpoint now wraps the encoded message
    handler behind one `handle_message` entry point without adding sockets or
    IPC.
35. worker endpoint observability/event-history scaffolding has been removed
    from the endpoint, codec, transport, exports, and tests. The worker helper
    boundary now keeps only the request/response path needed for future real
    transfer execution.
36. daemon relay discovery now reports backend-neutral relay eligibility,
    quota availability, cross-job session ownership, active reservations, and
    redacted lease records without changing direct fallback behavior.
37. worker loopback transport and transport protocol wrapper have been removed;
    the remaining worker transport is the Unix socket helper-process boundary
    needed for future daemon-approved execution.
38. worker service response envelopes and legacy service dict output no longer
    expose full lifecycle payloads; completion, status, cleanup, and staging
    release data now flow through the completion envelope only.
39. backend/runtime now has an exact daemon-plan execution entry point:
    runtime stores the daemon plan payload, the CUDA backend converts it to a
    native `TransferPlan`, and native runtime submits that exact plan directly
    to the existing CUDA executor.
40. daemon buffer registration, worker authorization, and worker data-plane
    request construction now preserve concrete buffer-handle metadata for
    `shared_pinned_cpu` source buffers and `cuda_ipc_device` target buffers.
41. client-side TurboBus shared CPU buffers now have an owned allocator,
    daemon registration helper, cross-process reopen support, and CUDA backend
    host-register/unregister hooks.
42. worker/helper execution can now bind the daemon-authorized shared CPU
    source before executor invocation, register that opened mapping with CUDA,
    and release it across success and binding-failure paths.
43. target GPU buffers now have a CUDA IPC producer/consumer path: client code
    exports a device pointer into `cuda_ipc_device` metadata, and worker/helper
    resources open and close that target pointer around bound executor calls.
44. the worker/helper path now has a first CUDA executor for H2D relay
    transfers. It uses bound shared CPU and CUDA IPC target resources, creates
    the native runtime inside the worker process, initializes the daemon-
    authorized relay GPU, submits the relay-only exact chunk plan, waits for
    completion, and returns daemon-owned completion metadata.
45. the client layer now has a worker-managed transfer call for the first H2D
    relay path. It registers the job and real buffer handles, obtains the
    daemon plan and lease token, submits the worker authorization request,
    observes worker completion, releases the completed relay reservation, and
    returns the daemon-owned final transfer status.
46. the same worker-managed call can now target a helper/socket-style worker
    client. The client submits a worker service envelope, consumes the returned
    data-plane completion envelope, and no longer requires the worker helper to
    return local lifecycle state.
47. a CUDA-server verification entry point now exists for the real
    helper-socket H2D relay path. It uses spawned daemon and worker helper
    processes, shared CPU memory, a CUDA IPC target tensor, daemon-issued relay
    lease, worker CUDA executor, byte comparison, and daemon reservation-release
    assertions.
48. worker-managed H2D relay execution now honors the exact daemon-issued
    relay chunk list for the leased relay. The narrow worker path refuses mixed
    direct-plus-relay plans until pooled worker execution exists, and it asks
    the daemon to release the relay reservation before surfacing that error.
49. worker authorization is now anchored to the daemon-stored transfer plan.
    The client no longer sends locally derived worker ranges, the daemon
    derives the authorized relay chunks from its own plan, returns that plan to
    the worker data-plane request, and the CUDA worker executor requires that
    daemon-issued plan before submitting native H2D relay work.
50. the worker-managed H2D path now handles a narrow pooled plan. A single
    daemon lease still scopes the relay chunks, but the worker CUDA executor
    consumes the complete daemon-issued direct-plus-relay plan and submits it
    as one native exact-plan H2D transfer.

The next immediate goal has changed: stop extending the unsupported
control-plane path and prepare the codebase for the first real
daemon-managed data movement slice. The worker control-plane smoke helper,
worker endpoint observability/event-history plumbing, loopback transport
wrapper, and full worker response lifecycle serialization have been removed.
The next code should verify the worker-managed H2D relay call on a CUDA server
through the helper socket by running `python -m turbobus.verification`. If it
fails, the next code should fix the failing real data-path layer before adding
new functionality. If it passes, the next functional expansion should be
deeper direct-plus-relay verification and then D2H support through the same
daemon plan path.

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
$env:PYTHONPATH='.'; python test/python/test_client_shared_buffer.py
$env:PYTHONPATH='.'; python test/python/test_client_worker_transfer.py
$env:PYTHONPATH='.'; python test/python/test_worker_transport.py
$env:PYTHONPATH='.'; python test/python/test_worker_process.py
$env:PYTHONPATH='.'; python test/python/test_verification.py
$env:PYTHONPATH='.'; python -m turbobus.verification --help
$env:PYTHONPATH='.'; python -m compileall turbobus
```

## Remaining Work

- keep the minimum daemon/client/worker spine for job and buffer registration,
  transfer requests, exact plans, leases, lease validation, worker
  authorization, staging ownership, completion, cleanup, and direct fallback;
- rebuild the native extension on a CUDA server and verify the worker-managed
  H2D relay call against real shared CPU, relay GPU, and CUDA IPC target
  buffers;
- keep direct fallback available when relay lease or worker execution fails;
- add cleanup and staging-buffer protection required by the real data path;
- defer more protocol, socket, observability, and smoke-test work unless it
  directly unblocks the functional data path;
- reconnect vLLM, model loading, and training offload only after the
  daemon/helper transfer path works end to end;
- avoid reintroducing local relay selection in runtime or framework adapters.
