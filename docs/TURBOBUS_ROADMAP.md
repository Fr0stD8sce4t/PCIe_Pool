# TurboBus Roadmap

This roadmap is the active implementation sequence for the paper-parity
reproduction.

## Phase 0: Paper-Parity Realignment

- Define shared contracts: JobIdentity, BufferHandle, TransferIntent,
  TopologySnapshot, SchedulingDecision, ExecutionTicket, and TransferReceipt.
- Route public transfer calls through daemon scheduling.
- Make scheduler decisions the only production transfer plans.
- Make worker execution require ExecutionTicket validation.
- Move synthetic topology into explicit test fixtures.
- Reorganize tests into unit, integration, e2e, and fixtures.
- Rewrite benchmarks and examples to use the public client API.
- Thin adapters so they submit intent and consume receipts.

Exit criteria:

- main transfer calls require daemon scheduling;
- default tests protect daemon-first contracts;
- benchmarks and examples report daemon decisions and receipts;
- `python -m compileall` and non-GPU tests pass.

## Phase 1: Automatic Topology Discovery

- Implement daemon-owned topology providers.
- Discover GPU id, UUID, PCI bus id, NUMA node, memory size, and visibility.
- Discover PCIe hierarchy, link generation, link width, negotiated speed, and
  estimated bandwidth.
- Discover CUDA P2P reachability and NVLink or NVSwitch information where
  available.
- Add topology snapshot ids, versioning, and invalidation.

Exit criteria:

- daemon can discover relay candidates for a target GPU;
- daemon reports filtered candidates and reasons;
- production startup fails clearly when topology discovery cannot satisfy the
  configured policy.

## Phase 2: Privileged Daemon Control Plane

- Add Unix socket peer credential checks.
- Bind user, process, container, job, and session identity.
- Register buffers with ownership checks.
- Manage shared CPU buffer and CUDA IPC or HIP IPC handle lifecycle.
- Track transfer states: submitted, running, complete, failed, canceled.
- Reclaim leases, reservations, and staging resources after failure or timeout.
- Emit audit records for transfer ownership and resource use.

Exit criteria:

- jobs cannot access each other's buffers;
- invalid tickets, buffers, leases, or sessions are rejected;
- stale sessions and failed workers are cleaned up.

## Phase 3: Cross-Job Dynamic Scheduling

- Add a global daemon transfer queue.
- Track active H2D, D2H, P2P, staging, and transfer state.
- Schedule from topology, measured bandwidth, current load, request size,
  workload kind, job weight, and fairness policy.
- Implement weighted fair sharing across jobs.
- Add admission control, delayed lease grants, plan expiration, and rescheduling.
- Keep direct fallback available as a scheduler outcome.

Exit criteria:

- jobs can use idle PCIe bandwidth from eligible relay devices without naming
  those devices;
- busy resources are avoided by new decisions;
- concurrent jobs receive explainable and fair decisions.

## Phase 4: Daemon-Plan Data Plane

- Use exact daemon-issued plans as data-plane input.
- Keep `fetch_plan_to_gpu` and `offload_plan_to_cpu` as backend primitives.
- Extend worker-managed execution to multiple relay paths.
- Track staging buffers through daemon or worker lifecycle.
- Share ticket and receipt semantics across H2D, D2H, and range transfers.
- Add correctness tests for direct, relay, pooled, and failure paths.

Exit criteria:

- application code does not decide transfer paths;
- workers complete daemon-ticketed direct and pooled transfers;
- repeated submission, lease expiration, and partial failure are deterministic.

## Phase 5: vLLM KV End-To-End Workload

- Convert vLLM KV save and restore into TransferIntent.
- Restore prefix KV blocks through daemon scheduling.
- Record daemon decision, topology snapshot, receipt, bytes, path split, and
  timing.
- Test single-job and multi-job vLLM scenarios.

Exit criteria:

- real vLLM requests save and restore KV cache through the daemon-first path;
- performance and path split are traceable through daemon and data-plane stats.

## Phase 6: Model Loading And Training Offload

- Convert model weight loading into TransferIntent.
- Convert training or optimizer state offload into TransferIntent.
- Include workload kind in scheduler policy.
- Unify correctness and performance reporting across workloads.

Exit criteria:

- vLLM KV, model loading, and training offload share the same public API;
- adapters carry framework mapping logic only.

## Phase 7: Paper Evaluation And Hardening

- Run experiments on 2, 4, and 8 GPU systems when available.
- Compare baseline policy and daemon-scheduled TurboBus.
- Measure single-job, multi-job, fairness, contention, and interference cases.
- Report p50 and p99 latency, throughput, PCIe utilization, relay impact,
  bytes moved, path split, failure recovery, and isolation.
- Make experiment output auditable from workload request to receipt.

Exit criteria:

- evaluation uses formal public APIs;
- results include daemon decisions, topology snapshots, execution tickets, and
  data-plane stats;
- the system supports at least one real LLM framework path end to end.

## Global Exit Criteria

TurboBus is on track when it can:

- accept daemon-scheduled transfer intent;
- discover machine topology automatically;
- schedule relay use across jobs;
- keep clients outside physical path control;
- execute exact daemon-issued plans;
- support vLLM KV cache save/restore end to end;
- report performance and isolation metrics with full traceability.
