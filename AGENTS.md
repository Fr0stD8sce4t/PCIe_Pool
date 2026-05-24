# TurboBus Agent Instructions

TurboBus is a paper-reproduction system project for:

TurboBus: Pooling PCIe Bandwidth for LLM Workloads via Scale-Up Fabrics.

The target system pools idle PCIe bandwidth in a multi-GPU server for large
model memory movement. Applications submit transfer intent. A privileged
per-node daemon discovers machine topology, schedules cross-job transfers,
issues execution tickets, and records completion. Workers and backend data
planes execute exact daemon-issued plans.

## Active Direction

Build directly toward paper parity.

The active architecture is:

- application adapters submit TransferIntent objects;
- daemon-owned topology discovery produces TopologySnapshot objects;
- daemon scheduler produces SchedulingDecision objects;
- daemon control plane issues ExecutionTicket objects and leases;
- worker or data-plane backend executes the exact ticketed plan;
- TransferReceipt objects report correctness, bytes, timing, path split, and
  failure state;
- real LLM workloads validate the system end to end.

Every code change should move the project closer to daemon-controlled,
topology-aware, cross-job execution.

## System Contract

The production path must satisfy these contracts:

- Applications describe what data must move, not which physical route to use.
- The daemon is the production scheduling authority.
- The scheduler is the only component that creates production transfer plans.
- Topology is discovered by daemon-owned providers.
- Synthetic topology is used only by explicit tests and fixtures.
- Workers execute only daemon-issued ExecutionTickets.
- Direct, relay, and pooled paths are scheduling outcomes.
- Adapters depend on the public client API and shared schema objects.
- Benchmarks and examples call the public client API.

## Core Objects

Use these shared objects across the control plane, scheduler, data plane,
adapters, tests, and experiments:

- JobIdentity: user, job, process, container, and session identity.
- BufferHandle: registered CPU or GPU buffer owned by a job.
- TransferIntent: requested movement, direction, byte ranges, workload kind,
  priority, and policy hints.
- TopologySnapshot: daemon-discovered GPU, PCIe, NUMA, and fabric state.
- SchedulingDecision: daemon-selected chunk-level plan and fallback reason.
- ExecutionTicket: daemon authorization for a worker or backend to execute one
  exact plan.
- TransferReceipt: completion state, bytes, timing, path split, and errors.

Tests may construct invalid variants of these objects only when the purpose is
to validate rejection behavior.

## Architecture Layers

Build TurboBus around these layers:

1. Client API.
   - Own user-facing transfer intent.
   - Register CPU pinned buffers and target GPU buffers.
   - Submit transfer intent to the daemon.
   - Wait for transfer receipts and expose stats.

2. Privileged daemon.
   - Own global machine state.
   - Discover GPUs, PCIe topology, NUMA topology, and scale-up fabric links.
   - Track jobs, sessions, users, containers, buffers, leases, and transfers.
   - Measure and cache path profiles.
   - Observe current PCIe and fabric utilization.
   - Schedule direct and relay paths across jobs.
   - Issue execution tickets and relay leases.
   - Reclaim resources after failure, timeout, or client exit.

3. Scheduler.
   - Convert topology, load, policy, and TransferIntent into a chunk-level
     SchedulingDecision.
   - Account for link bandwidth, fabric bandwidth, relay quotas, fairness,
     request size, workload kind, and fallback rules.
   - Explain every scheduling decision in machine-readable output.

4. Worker or helper process.
   - Own privileged data movement when the client cannot access all required
     devices.
   - Validate ExecutionTickets before touching buffers.
   - Hold relay device access and manage staging buffers.
   - Use CUDA IPC, HIP IPC, or equivalent safe handles where required.
   - Report completion through daemon status updates.

5. Data-plane backend layer.
   - Execute exact daemon-issued plans.
   - Provide CUDA/NVIDIA support first.
   - Add ROCm/AMD support as a separate backend.
   - Keep framework-specific policy out of native transfer execution.

6. LLM framework adapters.
   - Convert framework-owned tensors or blocks into TransferIntent.
   - Support vLLM KV cache prefix save/restore first.
   - Add model weight loading and training state offload after the vLLM path.

## Phase 0: Paper-Parity Realignment

Phase 0 is mandatory before broad feature expansion. It realigns main code,
tests, benchmarks, examples, exports, and adapters around the paper architecture.

### 0.1 Shared Contracts

- Define or rewrite schema objects for JobIdentity, BufferHandle,
  TransferIntent, TopologySnapshot, SchedulingDecision, ExecutionTicket, and
  TransferReceipt.
- Route the public API through daemon-first transfer intent.
- Make worker execution require ExecutionTicket.
- Make synthetic topology an explicit test fixture.

### 0.2 Package Structure

Prefer this package layout:

```text
turbobus/
  api/
  control/
  topology/
  scheduler/
  data_plane/
  adapters/
```

Experiments and paper validation code should live outside the core package path
and call the public API.

### 0.3 Public API And Exports

- Export the daemon-first API from `turbobus.__init__`.
- Keep low-level planner and backend objects internal unless they are shared
  schema objects or test fixtures.
- Express baseline behavior through scheduler policy and experiment
  configuration, not through application-side physical path choices.

### 0.4 Tests

Use this structure:

```text
test/python/
  unit/
  integration/
  e2e/
  fixtures/
```

Unit tests validate schemas, topology snapshots, scheduler decisions,
execution tickets, and plan validation. Integration tests validate daemon
session lifecycle, topology discovery, cross-job scheduling, worker ticket
execution, and client-daemon API behavior. End-to-end tests cover vLLM KV
restore, model loading, and training offload through daemon-first APIs.

### 0.5 Benchmarks And Examples

- Benchmarks submit workload intent through the public client API.
- Experiment configuration may request policy variants, but physical paths are
  recorded as daemon decisions.
- Outputs must include daemon decision id, topology snapshot id, execution
  ticket id, actual path stats, bytes, timing, and fallback reason.

### 0.6 Adapters

- Adapters submit TransferIntent and consume TransferReceipt.
- vLLM, model loading, and training adapters must not contain path selection
  policy.
- Framework-specific logic stays outside native transfer execution.

### 0.7 Acceptance Criteria

- Main transfer calls require daemon scheduling.
- The scheduler is the only source of production transfer plans.
- Workers reject requests without valid ExecutionTickets.
- Synthetic topology is available only through explicit fixtures.
- Default tests protect the daemon-first contract.
- Benchmarks and examples call public daemon-first APIs.
- `python -m compileall` and non-GPU tests pass.

## Phase 1: Automatic Topology Discovery

Implement daemon-owned topology discovery.

Required capabilities:

- GPU device id, UUID, PCI bus id, NUMA node, memory size, and visibility.
- PCIe root complex, switch hierarchy, link generation, link width, negotiated
  speed, and estimated bandwidth.
- CUDA P2P reachability and NVLink or NVSwitch information where available.
- Topology snapshot versioning and cache invalidation.
- CUDA/NVML provider as the first production provider.
- ROCm/HIP provider as a backend-specific later provider.

Acceptance criteria:

- The daemon can discover usable relay candidates from the local machine.
- Given a target GPU, the daemon reports eligible relays, filtered relays,
  filtering reasons, and path capabilities.
- Production startup fails clearly if topology discovery cannot satisfy the
  configured policy.

## Phase 2: Privileged Daemon Control Plane

Make the daemon the resource authority.

Required capabilities:

- Unix socket peer credential checks.
- User, process, container, job, and session binding.
- Buffer registration with ownership checks.
- Safe lifecycle for shared CPU buffers and CUDA IPC or HIP IPC handles.
- Transfer state machine: submitted, running, complete, failed, canceled.
- Lease and reservation cleanup on timeout, socket close, client exit, worker
  failure, or daemon-detected mismatch.
- Audit records for relay use, bytes moved, duration, owner, and failure reason.

Acceptance criteria:

- Jobs cannot access each other's buffers.
- Unauthorized buffer, device, lease, or ticket requests are rejected.
- Stale sessions and failed workers release reservations and staging resources.

## Phase 3: Cross-Job Dynamic Scheduling

Implement the shared PCIe bandwidth pool.

Required capabilities:

- Global daemon transfer queue.
- Runtime state for H2D, D2H, P2P, relay staging, and active transfers.
- Scheduling based on topology, measured bandwidth, current load, request size,
  workload kind, job weight, and fairness policy.
- Weighted fair sharing across jobs.
- Relay admission control and delayed lease grants.
- Plan expiration and rescheduling when topology, load, or leases change.
- Direct fallback when relay scheduling is unavailable or not beneficial.

Acceptance criteria:

- A job can borrow idle PCIe bandwidth from eligible devices without naming
  those devices.
- New requests avoid busy resources.
- Concurrent jobs receive fair and explainable scheduling decisions.

## Phase 4: Daemon-Plan Data Plane

Make exact daemon-issued plans the only production data-plane input.

Required capabilities:

- `fetch_plan_to_gpu` and `offload_plan_to_cpu` remain backend execution
  primitives.
- Worker-managed execution supports multiple relay paths in one plan.
- Staging buffers are tracked and cleaned by daemon or worker lifecycle.
- H2D, D2H, and range transfers share ticket and receipt semantics.
- Data correctness checks cover direct, relay, pooled, and failure cases.

Acceptance criteria:

- Application code does not decide transfer paths.
- Workers complete daemon-ticketed direct plus multi-relay pooled transfers.
- Lease expiration, repeated submission, and partial failure are deterministic.

## Phase 5: vLLM KV End-To-End Workload

Make vLLM KV cache save and restore the first full workload.

Required capabilities:

- vLLM connector submits KV save and restore TransferIntent.
- Prefix matches restore GPU KV blocks through daemon scheduling.
- Save and restore record daemon decision, topology snapshot, receipt, bytes,
  path split, and timing.
- Single-job and multi-job scenarios are covered.

Acceptance criteria:

- Real vLLM requests save and restore KV cache through the daemon-first path.
- Path split and performance results are traceable through daemon and
  data-plane stats.

## Phase 6: Model Loading And Training Offload

Extend the daemon-first path to more LLM memory movement.

Required capabilities:

- Model weight bucket loading through TransferIntent.
- Training state or optimizer state offload in both H2D and D2H directions.
- Workload kind included in scheduler policy.
- Unified correctness and performance reporting.

Acceptance criteria:

- vLLM KV, model loading, and training offload share the same public API.
- Adapters carry no physical path policy.

## Phase 7: Paper Evaluation And Hardening

Prove paper parity through full-system experiments.

Required capabilities:

- Experiments on 2, 4, and 8 GPU systems when hardware is available.
- Baseline policy versus daemon-scheduled TurboBus.
- Single-job, multi-job, fairness, relay contention, and interference studies.
- Metrics: p50 and p99 latency, throughput, PCIe utilization, relay impact,
  bytes moved, path split, failure recovery, and isolation.
- Experiment output traceable to daemon decisions, topology snapshots,
  execution tickets, and data-plane stats.

Acceptance criteria:

- Experiments use the formal public API.
- Results can be audited end to end from workload request to transfer receipt.

## Anti-Drift Rules

Rewrite changes that:

- put physical route selection into application code;
- bypass daemon scheduling to create production plans;
- make synthetic topology a production fallback;
- let adapters choose path policy;
- add benchmark-only APIs to core modules;
- add tests that do not protect a contract, integration path, or real workload;
- place framework policy inside CUDA or HIP execution code.

Prefer breaking changes when the existing code encodes the wrong architecture.

## Coding Rules

- Prefer simple, testable interfaces over compatibility shims.
- Keep client API, daemon control plane, scheduler, topology discovery, worker
  data plane, backend execution, and framework adapters separate.
- Keep native data movement framework-agnostic.
- Do not let benchmark scripts become the system.
- Keep direct transfer fallback available as a scheduler result or explicit
  failure fallback.
- Add focused tests that protect the daemon-first contract.
- For documentation-only changes, `git diff --check` is sufficient.
