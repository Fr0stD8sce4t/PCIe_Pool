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

Next code cut:

- connect the worker helper skeleton to daemon-authorized transfer requests
  through a small client-side helper, still returning unsupported execution
  until IPC-backed movement exists;
- keep the daemon as the owner of planning/control decisions;
- do not add worker execution yet, but shape the protocol so worker/helper
  execution can consume the same `TransferRequest` and lease records later.

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
