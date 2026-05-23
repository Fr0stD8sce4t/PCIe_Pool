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
- Add validation and serialization tests.

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

## Phase 4: Worker Execution And Isolation

- Add worker or helper processes for safe relay execution.
- Use IPC or equivalent handles for cross-process buffer access.
- Enforce job boundaries and lease checks.
- Prevent relay reuse across unauthorized jobs.

## Phase 5: Framework Adapters

- vLLM KV prefix save/restore.
- model-loading bucket transfer.
- training offload bucket transfer.

## Phase 6: ROCm Support

- Add a ROCm backend.
- Discover AMD peer and fabric capabilities.
- Reuse the same planner and daemon protocol.

## Phase 7: Evaluation

- single-job benchmarks;
- multi-job contention benchmarks;
- fairness and isolation tests;
- framework latency and throughput evaluation;
- direct vs relay vs pool comparisons.

## Exit Criteria

TurboBus is considered on track only when the new system can:

- accept daemon-approved transfer requests;
- schedule relay use across jobs;
- keep clients out of unauthorized relay control;
- support at least one real LLM framework path end to end;
- report clear metrics for both performance and isolation.
