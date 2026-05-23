# TurboBus Next Steps

## Current

Start the next phase after the current refactor layer: build out the daemon
control plane that will later own cross-job relay discovery, leases, and worker
execution.

Completed in the refactor layer:

- generic planner types for devices, links, paths, chunks, plans, leases, and
  stats;
- direct, relay, and pooled path selection;
- relay lease requirements and fallback rules;
- serialization and validation tests for planner objects and scheduler requests.
- daemon `PLAN_TRANSFER` requests that return a `PlannerTransferPlan`,
  `PlannerLease` objects, reservation ids, and `PlannerStats`.
- runtime preference for daemon-issued plans when the daemon supports the new
  request, with the old reserve-only request kept for compatibility.
- backend facade for the current native CUDA runtime.
- adapter package boundary for current framework integrations.
- explicit `TransferRequest`, `TransferRange`, and `TransferDirection` objects
  for runtime/client transfer submission.
- runtime daemon planning now prefers request-shaped `plan_transfer_request`
  calls before falling back to the old `plan_transfer`/`reserve_transfer`
  compatibility paths.
- framework adapter logic now lives under `turbobus.adapters`; old flat modules
  such as `turbobus.vllm_kv_connector` are compatibility aliases to the adapter
  modules.
- daemon protocol baseline message shapes for job identity, buffer
  registration, lease tokens, transfer status, and cleanup now exist, with
  validation coverage.
- daemon state now handles `REGISTER_JOB`, `REGISTER_BUFFER`, and `CLEANUP`
  requests for jobs, buffers, sessions, and reservations.

Current status:

- the shared planner model types now exist in `turbobus/planner_types.py`;
- a backend-neutral Python planner engine now exists in
  `turbobus/planner_engine.py`;
- `transfer_plan_to_dict` accepts the new planner plan model and the old native
  plan shape;
- daemon-side scheduler policy now lives in `turbobus/daemon/scheduler.py`;
- `TurboBusDaemon.plan_transfer` issues daemon-approved plans and records relay
  leases as releasable reservations;
- `Runtime` consumes daemon plan responses before falling back to the legacy
  `reserve_transfer` request.
- `turbobus.backends.cuda.CudaNativeBackend` owns native runtime creation,
  transfer-mode translation, and native range construction for the current CUDA
  path.
- `turbobus.transfer` now owns request-shaped transfer metadata that adapters,
  runtime, daemon, and future worker code can share.
- `turbobus.adapters` now owns framework-facing logic for inference slot
  adapters, vLLM helpers/connectors, model loading, and training offload while
  preserving old root-level import paths as module aliases.
- daemon protocol baseline message shapes now exist in `turbobus.schema` and
  are re-exported through `turbobus.daemon.protocol`.
- `TurboBusDaemon` now tracks registered jobs and buffers and exposes cleanup
  for job, buffer, session, and reservation records.
- daemon-issued plans now include a transfer id and status record; status can
  be queried or updated through `TRANSFER_STATUS`, and relay-backed transfers
  complete when reservations are released.
- socket clients can query daemon transfer status, and the socket round-trip
  coverage now verifies a planned relay-backed transfer through release and
  completion.
- daemon-issued relay reservations now carry unguessable lease tokens, and the
  daemon exposes a `VALIDATE_LEASE` request that future worker/helper code can
  use before touching relay resources.
- transfer requests can now carry registered buffer ids, and daemon lease
  validation checks the lease token, job, session, relay, and authorized buffer
  ids together.
- worker-facing transfer authorization messages now exist, and the daemon can
  authorize a future helper transfer by checking transfer id, lease token,
  job/session ownership, relay id, buffer ids, direction, and ranges.
- a `turbobus.worker` helper package now consumes daemon-authorized transfer
  contexts and reports unsupported execution explicitly, without CUDA IPC or
  real data movement.
- worker helper code can now ask a daemon client for worker-transfer
  authorization and convert the daemon response into a `WorkerTransferRequest`;
  execution still reports unsupported until the IPC-backed worker data path is
  added.
- worker helper code can now report unsupported, failed, and completed worker
  outcomes back to the daemon through `TRANSFER_STATUS`, so worker-side control
  flow can update daemon-owned transfer status before real IPC movement exists.
- daemon now has a backend-neutral, injectable resource inventory skeleton for
  GPUs, PCIe paths, and scale-up fabric links, exposed through `GET_INVENTORY`
  on the daemon control path.
- daemon planning now derives relay eligibility from the daemon-owned inventory
  before profile lookup and scheduling, while keeping the cached profile path
  and direct fallback behavior intact.
- daemon plan responses now include inventory-derived planning metadata,
  including requested relay ids, eligible relays, filtered relays with reasons,
  and the profile key used for planning.
- daemon state now reports system cleanup outcomes for stale sessions, closed
  sessions, and canceled reservations through `system_cleanup_events` in the
  daemon profile/describe payload.
- `TurboBusDaemonClient.describe()` now exposes the daemon profile/describe
  path, and socket coverage checks that cleanup observability is reachable
  through the same control-plane client.
- `TurboBusDaemonClient.cleanup()` now wraps the existing `CLEANUP` request so
  clients can request job, buffer, session, or reservation cleanup through the
  daemon control path; socket coverage checks client-driven session cleanup and
  the resulting user/system cleanup events.
- `WorkerTransferCleanupCoordinator` now lets worker-side helper code request
  daemon cleanup after authorization or execution failure, and
  `WorkerTransferClient.submit_report_and_cleanup()` ties authorization,
  unsupported execution, daemon status reporting, and reservation/session
  cleanup together without adding real data movement.
- `WorkerTransferLifecycleRecord` now captures the worker authorization
  request, daemon-approved worker context, status update, status response,
  cleanup target, cleanup response, final outcome, and error for future helper
  processes.
- `WorkerTransferService` now provides an in-process worker helper service
  boundary that accepts a `WorkerTransferAuthorizationRequest`, runs the
  existing worker control path, and returns a serialized lifecycle record for
  unsupported execution, authorization denial, status failure, and cleanup
  failure.
- worker service payload parsing helpers now accept plain daemon JSON
  dictionaries, validate them into `WorkerTransferAuthorizationRequest`
  objects, and feed parsed payloads through the same serialized lifecycle
  output path.
- `WorkerServiceRequestEnvelope` and `WorkerServiceResponseEnvelope` now give
  future helper process boundaries one stable in-process shape for parsed
  lifecycle payloads, malformed payload errors, status failures, and cleanup
  failures.
- an in-process worker service smoke helper now wires daemon-owned planning,
  worker service envelope handling, daemon status reporting, and daemon
  reservation cleanup together without sockets, IPC, or real data movement.
- worker data-plane request records now describe daemon-approved relay
  execution inputs, including source/destination buffer handles, relay staging
  needs, relay device, direction, chunk ranges, and completion/error reporting
  fields.
- worker helper requests now derive that data-plane shape from daemon
  authorization results and reject mismatched local data-plane authority.
- an in-memory worker staging pool now allocates, describes, validates, and
  releases relay staging slots from daemon-approved `WorkerDataPlaneRequest`
  records without CUDA IPC, sockets, real data movement, or hardware discovery.
- worker service lifecycle now allocates an in-memory staging slot after daemon
  authorization and releases it on unsupported execution, status failure, and
  cleanup failure paths without leaking staging slots.
- worker data-plane executors now receive both the daemon-authorized
  `WorkerTransferRequest` and its allocated `WorkerStagingSlot`, while the
  default executor remains on the explicit unsupported path.
- worker data-plane completion envelopes now serialize the allocated staging
  slot, worker result, daemon status update, daemon status response, daemon
  cleanup response, and staging release from lifecycle records without adding
  CUDA IPC, sockets, real data movement, or hardware discovery.
- worker service responses now carry both the full lifecycle record and the
  completion envelope, so future helper process callers can consume completion,
  status, cleanup, and staging-release output without parsing the full
  lifecycle payload.
- worker process message codecs now encode and decode JSON-safe
  `WorkerServiceRequestEnvelope` and `WorkerServiceResponseEnvelope` payloads
  in process, preserving completion envelopes and malformed-message errors
  without adding sockets, IPC, CUDA IPC, real data movement, or hardware
  discovery.
- an in-process worker service message handler now accepts encoded worker
  request messages, decodes them through the worker message codec, runs the
  existing `WorkerTransferService.handle_envelope` path, and returns encoded
  worker response messages without adding sockets, IPC, CUDA IPC, real data
  movement, or hardware discovery.
- a transport-neutral worker service endpoint now owns a
  `WorkerTransferService`, accepts encoded messages through the existing
  in-process message handler, and exposes a single `handle_message` entry point
  for future socket or IPC transports without adding those transports yet.
- worker endpoint request/response events now record message size, response
  size, ok/error status, final state, and whether completion data was present
  for each `handle_message` call without changing the encoded response payload.
- worker endpoint `describe()` snapshots now summarize recorded message events,
  including total request count, last event, final-state counts, error count,
  and completion count while staying in process.
- worker endpoint `clear_events()` now returns the current event snapshot,
  clears recorded request/response events, and resets `last_event` while
  leaving future encoded responses unchanged.

Next code cut:

- add an optional worker endpoint event history limit so long-running helper
  processes can bound in-memory observability state while preserving
  `last_event` and `describe()` summaries for retained events;
- keep it in process only, with no CUDA IPC, sockets, real data movement, or
  hardware discovery yet;
- add focused tests that the limit drops the oldest retained events, keeps the
  newest event visible, and does not change encoded response payloads.

## Upcoming

1. Daemon protocol baseline.
   - job/session identity;
   - buffer registration records;
   - transfer request/status messages;
   - lease-token records;
   - cleanup and stale-session messages.

2. Daemon control plane.
   - topology discovery;
   - session tracking;
   - relay quota;
   - lease issuance;
   - stale cleanup.

3. Worker or helper execution.
   - IPC or equivalent handle exchange;
   - relay GPU ownership outside the client process;
   - safe data movement under daemon-approved leases.

4. Isolation and policy.
   - job identity;
   - quota enforcement;
   - authorization checks;
   - fallback when relay access is denied.

5. Framework adapters.
   - adapter package boundary with compatibility imports; done as the first cut;
   - vLLM KV prefix save/restore;
   - model loading;
   - training offload.

6. ROCm backend.
   - HIP execution;
   - AMD scale-up fabric support;
   - backend-neutral planner reuse.

7. Evaluation.
   - single-job and multi-job benchmarks;
   - fairness and isolation checks;
   - direct vs relay vs pool comparisons.

## Done In The Planning Layer

- The repository direction has been reset away from the old prototype-centered
  plan.
- The new docs now point at a rewrite-first architecture.
- Shared protocol and runtime-support modules now back the Python entry points.
- The daemon can now issue a plan and relay leases instead of only accepting
  per-relay reserve calls.
- Runtime now reaches the current CUDA native extension through a backend
  facade instead of creating `_turbobus.Runtime` directly.
- Framework-facing imports now have a `turbobus.adapters` package boundary.
- Framework adapter implementations now live in `turbobus.adapters`; root-level
  adapter module names are compatibility aliases.
- Runtime and daemon planning now use explicit transfer request objects as the
  shared request shape for later worker execution.
