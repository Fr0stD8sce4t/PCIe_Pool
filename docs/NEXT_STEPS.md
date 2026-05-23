# TurboBus Next Steps

## Current

Finish the current code refactor by separating daemon planning/control from
local execution and framework adapters.

Completed in the planner/scheduler cut:

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
- `turbobus.adapters` now provides framework-facing imports for inference slot
  adapters, vLLM helpers/connectors, model loading, and training offload while
  preserving old root-level import paths.

Next code cut:

- introduce client/runtime transfer request objects that adapters can submit and
  daemon/worker code can consume later;
- keep direct, relay, and pooled behavior unchanged;
- keep adapters as clients of runtime/daemon APIs, not owners of scheduler,
  backend, or worker policy.

## Upcoming

1. CUDA backend baseline.
   - backend facade for the current native runtime; done as the first cut;
   - local transfer execution on the new interfaces;
   - staging buffer pool;
   - timing and stats parity with the old prototype.

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
