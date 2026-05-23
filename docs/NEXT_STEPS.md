# TurboBus Next Steps

## Current

Build the planner and scheduler model on top of the new shared protocol/types.

Required first deliverables:

- generic planner types for devices, links, paths, chunks, plans, leases, and
  stats;
- direct, relay, and pooled path selection;
- relay lease requirements and fallback rules;
- serialization and validation tests for planner objects and scheduler requests.

Current status:

- the shared planner model types now exist in `turbobus/planner_types.py`;
- a backend-neutral Python planner engine now exists in
  `turbobus/planner_engine.py`;
- `transfer_plan_to_dict` accepts the new planner plan model and the old native
  plan shape;
- the next code cut is scheduler policy: relay lease requirements, fallback
  decisions, and daemon-owned plan issuance.

## Upcoming

1. Planner and scheduler model.
   - chunk assignment tests.

2. CUDA backend baseline.
   - local transfer execution on the new interfaces;
   - staging buffer pool;
   - timing and stats parity with the old prototype.

3. Daemon control plane.
   - topology discovery;
   - session tracking;
   - relay quota;
   - lease issuance;
   - stale cleanup.

4. Worker or helper execution.
   - IPC or equivalent handle exchange;
   - relay GPU ownership outside the client process;
   - safe data movement under daemon-approved leases.

5. Isolation and policy.
   - job identity;
   - quota enforcement;
   - authorization checks;
   - fallback when relay access is denied.

6. Framework adapters.
   - vLLM KV prefix save/restore;
   - model loading;
   - training offload.

7. ROCm backend.
   - HIP execution;
   - AMD scale-up fabric support;
   - backend-neutral planner reuse.

8. Evaluation.
   - single-job and multi-job benchmarks;
   - fairness and isolation checks;
   - direct vs relay vs pool comparisons.

## Done In The Planning Layer

- The repository direction has been reset away from the old prototype-centered
  plan.
- The new docs now point at a rewrite-first architecture.
- Shared protocol and runtime-support modules now back the Python entry points.
