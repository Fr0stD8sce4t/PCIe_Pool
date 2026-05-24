# TurboBus Next Steps

## Current Direction

TurboBus is being advanced directly toward paper parity. The immediate work is
to realign the codebase around daemon-first scheduling, automatic topology
discovery, execution tickets, and public TransferIntent APIs.

The next implementation work must update main code, tests, benchmarks,
examples, exports, and adapters together. Tests and examples should protect the
new architecture rather than preserve application-side route selection.

## Immediate Functional Target

Begin Phase 3: Cross-Job Dynamic Scheduling.

Phase 0 is complete. The public examples and benchmarks now use daemon-first
client APIs, and the old Runtime-shaped benchmark/example entry points have
been removed instead of preserved as compatibility paths.

Phase 1 is complete. The daemon now owns production topology discovery,
normalizes GPU, PCIe, and fabric capabilities, exposes versioned topology
snapshots, supports explicit topology invalidation, reports relay eligibility
with path capabilities, and fails production startup clearly when discovery
cannot satisfy policy.

Phase 2 daemon-owned resource authority is complete. The next target is
Phase 3 cross-job dynamic scheduling:

1. Add a global daemon transfer queue.
2. Track active H2D, D2H, P2P, staging, and transfer state.
3. Schedule from topology, measured bandwidth, current load, request size,
   workload kind, job weight, and fairness policy.
4. Add weighted fair sharing across jobs.
5. Add relay admission control, delayed lease grants, plan expiration, and
   rescheduling while keeping direct fallback as a scheduler outcome.

## Current

Phase 3: Cross-Job Dynamic Scheduling.

Phase 3 is now the active implementation target. Cut 1 is complete. The
daemon-owned transfer queue and scheduler-readable runtime resource state now
let later fairness, admission control, and delayed lease decisions reason
about queued and active work across jobs.

Current item: Phase 3 Cut 2, cross-job scheduling policy and fairness inputs.

### Phase 3 Cut 1

Status: complete.

- Add a daemon-owned transfer queue for submitted `TransferIntent` work.
- Track queued, running, active, relay staging, lease, reservation, and path
  state in one scheduler-readable runtime snapshot.
- Expose queue and runtime-state summaries through daemon profile data and
  pass runtime state metadata into scheduler planning.
- Keep applications and adapters on `TransferIntent` and `TransferReceipt`
  only; they do not choose direct, relay, or pooled paths.
- Preserve scheduler ownership of production transfer plans and worker
  `ExecutionTicket` enforcement.
- Add focused tests for queued requests, active-resource accounting, relay
  staging, and scheduler visibility without introducing app-side path controls.

Expected output:

- daemon scheduling can reason over queued and active work across jobs;
- Phase 3 can add fairness and admission control on top of explicit runtime
  resource state;
- no Phase 2 control-plane isolation or cleanup guarantee regresses.

### Phase 3 Cut 2

Status: current.

- Feed scheduler decisions from daemon runtime state, measured load, request
  size, workload kind, and job weight.
- Add weighted fair sharing inputs that can observe queue depth and active
  resource usage across jobs.
- Keep relay admission control, delayed lease grants, plan expiration, and
  rescheduling as daemon-owned outcomes instead of app-side controls.
- Add focused tests that prove scheduling decisions remain explainable and do
  not reintroduce application-side physical path selection.

Expected output:

- queue depth and active load influence scheduler decisions;
- fairness policy can be layered on top of the daemon-owned runtime snapshot;
- direct fallback remains a scheduler outcome, not an adapter choice.

## Phase 0 Code Cuts

### Cut 1: Contract Inventory

Status: complete.

- List all current public entry points that submit transfers.
- List all tests that construct runtimes, daemon sessions, scheduler plans, or
  worker requests directly.
- List examples and benchmarks that call low-level transfer controls.
- Mark each entry point as public API, internal API, test fixture, or experiment
  code.

Expected output:

- `docs/PHASE0_CONTRACT_INVENTORY.md`;
- no implementation behavior change.

### Cut 2: Shared Schema Layer

Status: complete.

- Add the new schema objects.
- Add validation rules for ownership, byte ranges, directions, workload kind,
  decision ids, topology snapshot ids, and ticket ids.
- Add unit tests for valid and invalid schema objects.

Expected output:

- schema unit tests under `test/python/unit/`;
- no framework adapter changes yet.

### Cut 3: Package Boundary Setup

Status: complete.

Introduce the new package layout:

```text
turbobus/
  api/
  control/
  topology/
  scheduler/
  data_plane/
  adapters/
```

Move or wrap code only when it clarifies ownership. Do not preserve module
locations only because existing tests import them.

Expected output:

- imports make layer ownership clear;
- test fixtures are separated from production topology code.

### Cut 4: Daemon-First Client API

Status: complete.

- Add a public client API that submits TransferIntent.
- Make receipt-oriented waiting the public completion path.
- Keep low-level backend execution behind scheduler decisions and tickets.
- Update `turbobus.__init__` to export the new public API.

Expected output:

- client API tests can submit an intent to a fake daemon and receive a receipt;
- public exports match the daemon-first contract.

### Cut 5: Scheduler And Ticket Contract

Status: complete.

- Make SchedulingDecision the scheduler output.
- Make ExecutionTicket the worker input.
- Validate that tickets bind job, session, buffers, byte ranges, topology
  snapshot, and scheduling decision.
- Add rejection tests for mismatched tickets.

Expected output:

- worker unit tests reject modified tickets;
- scheduler tests assert decisions are explainable and traceable.

### Cut 6: Test Tree Rewrite

Status: complete.

Restructure tests:

```text
test/python/
  unit/
  integration/
  e2e/
  fixtures/
```

Move or rewrite tests according to the layer they protect.

Expected output:

- default tests protect schema, scheduler, daemon, ticket, and public API
  contracts;
- GPU tests are clearly marked as requiring CUDA hardware.

### Cut 7: Benchmarks And Examples Rewrite

Status: complete.

Cut 7 is intentionally split so benchmark work does not drift back into the
old Runtime/direct/relay/pool path controls.

Substage 7.1: public intent and receipt reporting contract.

Status: complete.

- Daemon accepts public `TransferIntent` submission and returns
  `TransferReceipt`.
- Benchmark support helpers build workload intent without physical path hints.
- Receipt trace helpers report decision id, topology snapshot id, ticket id,
  bytes, path split, and fallback reason.

Substage 7.2: model-loading benchmark rewrite.

Status: complete.

- Replace Runtime-owned physical mode selection with public client
  `TransferIntent` submission.
- Treat baseline policy as benchmark configuration metadata only.
- Read actual path split from daemon receipt and decision data.

Substage 7.3: training-offload benchmark rewrite.

Status: complete.

- Submit H2D prefetch and D2H offload intent through the public client API.
- Report separate prefetch/offload receipt ids, decision ids, ticket ids, bytes,
  path split, timing, and fallback reason.

Substage 7.4: examples and paper-validation command rewrite.

Status: complete.

- Update examples to demonstrate daemon-first submission.
- Update paper-validation command construction and output validation so it no
  longer expects applications to choose physical transfer paths.

Expected output:

- examples demonstrate daemon-first submission;
- benchmarks no longer encode transfer path policy in application code.

### Cut 8: Adapter Thinning

Status: complete.

Cut 8 is split so adapter work does not drift back into Runtime-owned physical
path selection.

Substage 8.1: shared adapter intent and receipt path.

Status: complete.

- Add an adapter transfer context for job id, session id, registered CPU/GPU
  buffer ids, workload kind, priority, metadata, and receipt wait timeout.
- Make the shared named-block adapter layer submit `TransferIntent` through
  the client API.
- Make adapter handles wait for `TransferReceipt` and derive bytes, chunks,
  block state, and failure state from receipts.
- Rewrite model-loading, training-offload, inference KV, vLLM slot, and vLLM
  integration adapter tests around intent construction and receipt handling.

Substage 8.2: vLLM KV connector daemon-first rewrite.

Status: complete.

- Remove `Runtime`, `RuntimeOptions`, target GPU, relay GPU, transfer mode, and
  min-pool-byte configuration from `vllm_kv_connector`.
- Make connector configuration accept daemon socket/session/job/buffer
  identity and registered backing metadata.
- Route save/restore through `AdapterTransferContext`, `TurboBusClient`, and
  `VllmKVSlotAdapter`.
- Replace connector events that report requested physical mode with receipt
  ids, decision ids, topology snapshot ids, ticket ids, bytes, path split, and
  fallback reason.
- Update connector, sweep, and example tests so they no longer encode
  application-side direct/relay/pool choices.

Substage 8.3: adapter exports and old Runtime test cleanup.

Status: complete.

- Remove or demote adapter-facing tests that only protect the old Runtime
  direct/relay/pool route-selection surface.
- Keep only adapter tests that protect framework mapping, intent fields,
  receipt handling, and public package boundaries.
- Confirm no adapter module imports `Runtime` or calls Runtime transfer
  methods.

Expected output:

- adapter tests assert TransferIntent construction and receipt handling;
- no adapter owns path selection policy.

### Cut 9: Remaining Legacy Benchmark And Example Cleanup

Status: complete.

Phase 0 is not complete until remaining examples and benchmarks stop presenting
old Runtime, target GPU, relay GPU, and physical mode controls as active
application-facing workflows.

- Remove or rewrite legacy examples that still construct `Runtime` or expose
  target GPU, relay GPU, or physical mode selection.
- Remove or rewrite legacy benchmarks that still construct `Runtime` or sweep
  direct, relay, or pool as application-side modes.
- Keep benchmark policy labels only as experiment metadata; physical path
  outcomes must come from daemon decisions and receipts.
- Keep any exact-plan direct, relay, or pooled coverage inside scheduler,
  worker, or data-plane tests where daemon decisions and tickets are the
  authority.

Resolved files:

- removed `examples/vllm_turbobus_restore.py`;
- removed `benchmarks/bandwidth_pool.py`;
- removed `benchmarks/kv_offload.py`;
- removed `benchmarks/tune_transfer.py`;
- rewrote `benchmarks/summarize_result.py` so it rejects old route-shaped JSON
  and only summarizes daemon-first benchmark outputs.

Expected output:

- public examples and benchmarks call daemon-first APIs;
- no active example or benchmark constructs `Runtime` for workload submission;
- route-shaped controls remain only in scheduler, worker, data-plane, or
  explicit legacy-internal tests.

## Phase 0 Done Criteria

Phase 0 is complete.

- the main transfer path requires daemon scheduling;
- scheduler decisions are the only production transfer plans;
- workers require ExecutionTickets;
- synthetic topology is explicit test fixture data;
- default tests are organized by unit, integration, e2e, and fixtures;
- examples and benchmarks call the public client API;
- `python -m compileall` passes;
- non-GPU tests pass;
- GPU tests are clearly marked and runnable on CUDA hardware.

## Phase 1 Current Work

Phase 1 is complete.

Cut 1: topology provider boundary and production startup contract.

Status: complete.

- Add a daemon-owned topology provider interface if the current provider shape
  is not sufficient.
- Keep synthetic topology under `test/python/fixtures/`; do not add synthetic
  fallback to production startup.
- Add a production provider selection path for daemon startup.
- Return or cache `TopologySnapshot` objects with stable ids and version fields.
- Add focused tests for provider selection, snapshot shape, and startup failure
  when no production provider can satisfy policy.

Expected output:

- daemon topology code has a clear production provider boundary;
- tests protect that production startup cannot silently use fixture topology;
- no application or benchmark code chooses relays or physical paths.

Cut 2: complete GPU, PCIe, and fabric capability normalization.

Status: complete.

- Normalize provider output for GPU UUID, PCI bus id, NUMA node, memory size,
  visibility, and backend/vendor fields.
- Add PCIe link generation, width, root complex, and bandwidth estimation where
  the production provider can discover them.
- Add CUDA P2P or NVLink/NVSwitch capability parsing with relay filtering
  reasons preserved in daemon discovery output.
- Keep missing capability fields explicit rather than inventing synthetic
  defaults.
- Add focused tests for normalized records and filtered relay explanations.

Expected output:

- `GET_INVENTORY` and relay discovery expose normalized GPU, PCIe, and fabric
  fields with stable topology snapshot ids;
- the daemon can explain why each candidate relay is eligible or filtered;
- production startup still fails clearly when policy requirements are not met.

Cut 3: topology refresh and relay discovery completion.

Status: complete.

- Add an explicit daemon-owned topology refresh or invalidation path for
  providers that cache discovery results.
- Keep synthetic topology confined to explicit fixtures; do not add a
  production fallback when refresh fails.
- Ensure `GET_INVENTORY` and `DISCOVER_RELAYS` report stable snapshot ids,
  version changes after invalidation, eligible relays, filtered relays,
  filtering reasons, and per-relay path capabilities.
- Add focused tests for cache invalidation, provider refresh behavior, and
  relay discovery output after topology changes.
- Preserve clear production startup failure when discovery or relay policy
  cannot be satisfied.

Expected output:

- daemon topology refresh changes snapshot id/version when the provider
  discovers new state;
- relay discovery can be audited from snapshot id to candidate path
  capabilities;
- Phase 1 exit criteria are ready to validate on a real multi-GPU server.

## Phase 1 Done Criteria

Phase 1 is complete.

- daemon-owned CUDA/NVML topology discovery is the production provider path;
- topology inventories include GPU identity, PCI bus id, NUMA, memory,
  visibility, PCIe link capability, and fabric capability fields;
- `GET_INVENTORY` returns versioned `TopologySnapshot` data;
- `DISCOVER_RELAYS` reports eligible relays, filtered relays, filtering
  reasons, and per-relay path capabilities;
- explicit topology invalidation refreshes provider state and changes
  snapshot id/version when new topology is discovered;
- production startup rejects synthetic fixtures and fails clearly when relay
  policy cannot be satisfied.

## Phase 2 Current Work

Phase 2 is complete.

Cut 1: peer identity and socket credential foundation.

Status: complete.

- Add a daemon-side peer identity record for socket-connected clients.
- Capture platform socket credentials where available, keeping unsupported
  platforms explicit instead of pretending credentials were checked.
- Bind peer identity to job and session registration paths without allowing
  clients to spoof another job owner.
- Preserve existing daemon-first TransferIntent and ExecutionTicket contracts.
- Add focused tests for accepted same-owner registration, rejected mismatched
  owner registration, and unsupported credential behavior.

Expected output:

- daemon request handling can attach an authenticated or explicitly
  unauthenticated peer identity to control-plane actions;
- job/session registration has a clear ownership boundary for Phase 2 buffer
  checks;
- no application or adapter gains physical path selection.

Cut 2: buffer ownership checks.

Status: complete.

- Bind registered buffers to the job owner identity recorded in Cut 1.
- Reject buffer registration when the authenticated peer does not own the job.
- Reject transfer planning, worker authorization, and lease validation when
  requested buffers belong to a different job or session owner.
- Keep direct, relay, and pooled scheduling as daemon outcomes only.
- Add focused tests for same-owner buffer registration, cross-owner rejection,
  and transfer submission with mismatched buffer ownership.

Expected output:

- jobs cannot register or use another peer's buffers;
- buffer ownership errors are explicit and machine-readable through daemon
  responses;
- ExecutionTicket and TransferIntent paths continue to use daemon-side
  ownership checks instead of adapter-side path policy.

Cut 3: lifecycle cleanup for stale resources.

Status: complete.

- Make session close, timeout, socket disconnect, worker failure, and detected
  mismatch release all related leases, reservations, transfer state, staging
  records, and buffers owned by the affected job or session.
- Keep cleanup decisions inside the daemon; clients and adapters must not own
  cleanup policy or physical path choices.
- Make repeated cleanup idempotent for already terminal transfers and already
  released reservations.
- Add focused tests for socket disconnect cleanup, timeout cleanup, worker
  failure cleanup, and mismatched owner cleanup.
- Preserve existing explicit cleanup APIs while making daemon-triggered cleanup
  visible through machine-readable response and profile state.

Expected output:

- stale sessions and failed workers release reservations and staging resources;
- transfer state reaches complete, failed, or canceled deterministically;
- cleanup records identify the owner, resource ids, reason, and released
  resource counts.

Cut 4: audit records for relay use and failures.

Status: complete.

- Add daemon-owned audit records for relay use, bytes moved, duration, owner,
  transfer id, decision id, ticket id, lease id, session id, job id, and
  failure reason.
- Emit audit records for worker completion, worker failure, cleanup, timeout,
  socket disconnect, lease expiration, and detected mismatch.
- Keep audit recording inside the daemon control plane; clients, adapters,
  benchmarks, and workers may report status but must not author audit truth.
- Expose audit records through machine-readable daemon profile or a dedicated
  control-plane response without widening application-side path control.
- Add focused tests that prove audit records exist for successful relay use,
  failed worker cleanup, mismatch cleanup, and stale session cleanup.

Expected output:

- transfer ownership and relay resource use can be traced from daemon state;
- failure and cleanup records carry owner, reason, bytes, duration, and related
  resource ids;
- Phase 2 exit criteria can be checked before entering cross-job scheduling.

## Phase 2 Done Criteria

Phase 2 is complete.

- peer credentials are captured where the platform supports them;
- job and session identity bind to daemon-observed peer identity;
- buffer registration and transfer use enforce ownership;
- invalid tickets, buffers, leases, sessions, and cross-owner requests are
  rejected;
- stale sessions and failed workers clean reservations and staging records;
- daemon profile exposes audit records for relay use, worker completion, worker
  failure, cleanup, timeout, lease expiration, socket disconnect, and mismatch.

## Phase 3 Current Work

Current item: Phase 3 Cut 1, global daemon transfer queue and runtime resource
state.

Cut 1: global queue and runtime resource state.

Status: current.

- Add a daemon-owned transfer queue for submitted TransferIntent work.
- Track active H2D, D2H, P2P, relay staging, leases, and running transfer
  state in one scheduler-readable snapshot.
- Keep applications and adapters on TransferIntent and TransferReceipt only;
  they must not choose direct, relay, or pooled paths.
- Preserve scheduler ownership of production transfer plans and worker
  ExecutionTicket enforcement.
- Add focused tests for queued requests, active-resource accounting, and
  scheduler visibility without implementing app-side path controls.

Expected output:

- daemon scheduling can reason over queued and active work across jobs;
- Phase 3 can add fairness and admission control on top of explicit runtime
  resource state;
- no Phase 2 control-plane isolation or cleanup guarantee regresses.

## After Phase 0

Proceed in this order:

1. Automatic topology discovery.
2. Privileged daemon control plane. Complete.
3. Cross-job dynamic scheduling. Current.
4. Daemon-plan data plane.
5. vLLM KV end-to-end workload.
6. Model loading and training offload.
7. Paper evaluation and hardening.
