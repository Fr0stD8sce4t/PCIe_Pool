# TurboBus Roadmap

This roadmap follows a rewrite-first approach.

## Phase 0: Architecture Reset

- Define the system as client, daemon, worker, backend, planner, and adapter
  layers.
- Remove prototype assumptions from the public design.
- Lock the first implementation target to a daemon-managed transfer system.

## Phase 1: Protocol And Types

- Define daemon/client/worker messages.
- Define backend-neutral transfer objects.
- Define job, buffer, lease, path, chunk, and stats types.
- Stop once the first real data path has the metadata it needs.

## Phase 2: Planner And Backend Baseline

- Implement a backend-neutral planner.
- Implement a CUDA backend on top of the new interfaces.
- Reproduce direct, relay, and pooled behavior on the new types.
- Keep scheduling policy out of backend execution code.

## Phase 3: Privileged Daemon

- Discover topology and fabric links.
- Track jobs and sessions.
- Track relay ownership and relay quota.
- Issue relay leases.
- Reclaim stale resources.

## Phase 4: Remove Non-Functional Scaffold

- Remove smoke-only helpers and tests that do not exercise real data movement.
- Remove endpoint observability/event-history code that is not needed for the
  first working transfer path.
- Remove extra socket/transport wrappers that only preserve unsupported
  lifecycle behavior.
- Keep the smallest daemon/client/worker surface needed for plans, leases,
  worker authorization, execution, completion, cleanup, and direct fallback.

## Phase 5: Whole-System CUDA Data Path

- Execute exact daemon-issued chunk plans.
- Register real client buffers with daemon-approved handles.
- Implement the first shared pinned CPU buffer strategy.
- Move bytes through direct, relay, and pooled CUDA paths from a daemon plan.
- Keep the first cut narrow if needed: one relay, H2D, static topology.

## Phase 6: Worker Execution And Isolation

- Add worker or helper processes for safe relay execution.
- Use IPC or equivalent handles for cross-process buffer access.
- Enforce job boundaries and lease checks.
- Prevent relay reuse across unauthorized jobs.

## Phase 7: Framework Adapters

- vLLM KV prefix save/restore.
- model-loading bucket transfer.
- training offload bucket transfer.

## Phase 8: ROCm Support

- Add a ROCm backend.
- Discover AMD peer and fabric capabilities.
- Reuse the same planner and daemon protocol.

## Phase 9: Evaluation

- single-job benchmarks;
- multi-job contention benchmarks;
- fairness and isolation checks;
- framework latency and throughput evaluation;
- direct vs relay vs pool comparisons.

## Exit Criteria

TurboBus is considered on track only when the new system can:

- accept daemon-approved transfer requests;
- move real bytes through a daemon-issued direct, relay, or pooled plan;
- schedule relay use across jobs;
- keep clients out of unauthorized relay control;
- support at least one real LLM framework path end to end;
- report clear metrics for both performance and isolation.
