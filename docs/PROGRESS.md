# TurboBus Progress

## Current State

The active project plan has been reset to paper-parity execution.

Phase 0 realignment, Phase 1 automatic topology discovery, Phase 2
privileged daemon control plane, and Phase 3 cross-job dynamic scheduling are
complete. The next work is Phase 4, daemon-plan data plane, which makes exact
daemon-issued execution tickets the only production input to workers and
backend data movement.

The active target architecture is:

- public APIs submit TransferIntent;
- daemon topology providers produce TopologySnapshot;
- daemon scheduler produces SchedulingDecision;
- daemon control plane issues ExecutionTicket;
- workers and data-plane backends execute exact ticketed plans;
- completion returns TransferReceipt;
- vLLM KV cache save/restore is the first full workload target.

## Completed Planning Work

- `AGENTS.md` now defines the paper-parity architecture, contracts, phases, and
  anti-drift rules.
- `docs/NEXT_STEPS.md` now describes the Phase 0 execution cuts.
- `docs/TURBOBUS_ROADMAP.md` now describes the active phased roadmap.
- This progress file now tracks the same daemon-first plan.
- Phase 0 Cut 1 is complete. `docs/PHASE0_CONTRACT_INVENTORY.md` records the
  current transfer entry points, tests, examples, benchmarks, and the target
  classification for the daemon-first architecture.
- Phase 0 Cut 2 is complete. `turbobus/schema.py` now defines the shared
  daemon-first schema objects for BufferHandle, TransferIntent,
  TopologySnapshot, SchedulingDecision, ExecutionTicket, and TransferReceipt.
  `test/python/unit/test_contract_schema.py` validates serialization and
  rejection behavior for the new contracts.
- Phase 0 Cut 3 is complete. The package now has explicit `api`, `control`,
  `topology`, `scheduler`, and `data_plane` boundaries. Production topology
  records live under `turbobus/topology`; static topology has been moved to
  `test/python/fixtures/topology.py`, and the daemon no longer creates a
  synthetic topology provider by default.
- Phase 0 Cut 4 is complete. `turbobus.api.TurboBusClient` is the public
  daemon-first client facade. It submits `TransferIntent` objects through the
  daemon client and returns `TransferReceipt` objects as the public completion
  result. The root `turbobus` package now exports the daemon-first API and
  shared contract objects instead of top-level planner/runtime transfer
  controls.
- Phase 0 Cut 5 is complete. `DaemonScheduler.plan_transfer` now returns the
  shared `SchedulingDecision` contract. The daemon stores decisions, exposes
  decision ids and topology snapshot ids in plan responses, and issues
  `ExecutionTicket` objects during worker authorization. Worker request
  construction can now use tickets as input and rejects mismatched decision,
  buffer, lease, range, and daemon-plan bindings before data-plane execution.
- Phase 0 Cut 6 is complete. The Python tests are organized under
  `test/python/unit/`, `test/python/integration/`, `test/python/e2e/`, and
  `test/python/fixtures/`. Moved tests now use explicit fixture and internal
  module imports, so the root package remains focused on daemon-first public
  API exports instead of old planner, transfer, and offload internals.
- Phase 0 Cut 7 Substage 7.1 is complete. Benchmark helper code can construct
  workload `TransferIntent` objects, submit them through the public client API,
  and format `TransferReceipt` traces with decision id, topology snapshot id,
  ticket id, bytes, path split, and fallback reason.
- Phase 0 Cut 7 Substage 7.2 is complete. `benchmarks/model_loading.py` no
  longer builds torch tensors, constructs a `Runtime`, or accepts
  application-side `direct`, `relay`, or `pool` controls. It now submits
  model-weight `TransferIntent` objects through `TurboBusClient`, stores
  benchmark policy as metadata, and reads actual path split and ids from daemon
  receipts. Focused validation covered the model-loading benchmark contract and
  JSON-safe receipt output.
- Phase 0 Cut 7 Substage 7.3 is complete. `benchmarks/training_offload.py` no
  longer builds torch tensors, constructs a `Runtime`, or accepts
  application-side target GPU, relay GPU, or physical transfer mode controls.
  It now submits paired H2D prefetch and D2H offload `TransferIntent` objects
  through `TurboBusClient`, stores benchmark policy as metadata, and reads
  separate receipt ids, decision ids, topology snapshot ids, execution ticket
  ids, bytes, timing, path split, and fallback reason from daemon receipts.
- Phase 0 Cut 7 Substage 7.4 is complete. `benchmarks/paper_validation.py` now
  builds daemon-first commands for model-loading and training-offload using
  session ids and registered buffer ids instead of target GPU, relay GPU, or
  physical transfer mode arguments. Paper validation metrics now come from
  daemon receipt trace ids and path split fields. `examples/torch_tensor_fetch.py`
  demonstrates public `TransferIntent` submission and `TransferReceipt`
  reporting instead of constructing a `Runtime`.
- Phase 0 Cut 8 Substage 8.1 is complete. `AdapterTransferContext` now carries
  adapter job, session, registered CPU/GPU buffer, workload, metadata, and wait
  settings. `OffloadStore` no longer calls Runtime transfer methods; it submits
  `TransferIntent` objects through the client API, waits for
  `TransferReceipt` objects, and derives transfer stats and block state from
  receipts. The model-loading, training-offload, inference KV, vLLM slot, and
  vLLM integration adapter tests now protect intent construction and receipt
  consumption.
- Phase 0 Cut 8 Substage 8.2 is complete. `vllm_kv_connector.py` no longer
  imports `Runtime` or exposes target GPU, relay GPU, transfer mode, or
  min-pool-byte configuration. The connector now accepts daemon socket, job,
  session, and registered CPU/GPU buffer identity, routes vLLM KV save and
  restore through `AdapterTransferContext`, `TurboBusClient`, and
  `VllmKVSlotAdapter`, and reports receipt ids, decision ids, topology
  snapshot ids, ticket ids, path split, bytes, and fallback reason. The vLLM
  connector example and sweep now use public daemon-first identity arguments
  and daemon receipt output instead of application-side direct, relay, or pool
  choices.
- Phase 0 Cut 8 Substage 8.3 is complete. The old non-KV
  `vllm_connector` experiment, root wrapper, adapter re-export, route-shaped
  example, and old test have been removed. Adapter-facing exports now protect
  the daemon-first vLLM KV connector, vLLM mapping, vLLM integration, intent
  fields, receipt handling, and public package boundaries instead of the old
  Runtime route-selection surface.
- Phase 0 Cut 9 is complete. The remaining legacy workload entry points
  `examples/vllm_turbobus_restore.py`, `benchmarks/bandwidth_pool.py`,
  `benchmarks/kv_offload.py`, and `benchmarks/tune_transfer.py` have been
  removed because they encoded application-side target GPU, relay GPU, and
  physical mode control. `benchmarks/summarize_result.py` now summarizes only
  daemon-first model-loading, training-offload, and paper-validation JSON
  shapes and rejects old route-shaped benchmark JSON.
- Phase 1 Cut 1 is complete. `turbobus.topology.cuda_nvml` now provides a
  production CUDA/NVML topology provider using `nvidia-smi` probes. Topology
  inventories now carry GPU UUIDs, versioned snapshot ids, and a conversion to
  shared `TopologySnapshot`. Daemon startup now uses a production provider
  through `create_production_daemon`, rejects synthetic fixture topology for
  production startup, validates relay policy against discovered topology, and
  reports topology snapshot ids in relay discovery.
- Phase 1 Cut 2 is complete. Topology records now carry normalized PCIe
  negotiated speed, switch hierarchy, bandwidth source, and fabric
  raw-link/capability/link-count fields. The CUDA/NVML provider parses richer
  `nvidia-smi` GPU query output when available, falls back to identity-only
  query output when required, keeps missing capabilities explicit, estimates
  PCIe bandwidth from generation and width, and normalizes NVLink/NVSwitch and
  CUDA P2P topology-matrix tokens. Relay discovery now reports
  `path_capabilities` for each candidate relay beside eligibility and filtered
  reasons.
- Phase 1 Cut 3 is complete. The daemon now has an explicit topology
  invalidation control-plane request and client method. Invalidation calls the
  provider refresh path, returns the refreshed topology snapshot id/version,
  and subsequent `GET_INVENTORY` and `DISCOVER_RELAYS` responses use the new
  snapshot. Integration tests cover refresh changing relay discovery from a
  filtered candidate to an eligible relay with path capabilities.
- Phase 2 Cut 1 is complete. `PeerIdentity` now records authenticated or
  explicitly unsupported peer credential state. Socket handling attaches
  daemon-captured peer identity to requests, ignores client-supplied JSON peer
  identity, and uses Unix `SO_PEERCRED` where available. Session and job
  registration now record peer identity, bind authenticated user/process
  fields to the daemon-observed peer, and reject spoofed user ids or
  cross-peer session ownership.
- Phase 2 Cut 2 is complete. Buffer registration, transfer planning,
  `TransferIntent` submission, lease validation, and worker transfer
  authorization now enforce daemon-side authenticated job ownership. Cross-peer
  attempts to register or use another job's buffers are rejected before
  scheduling, lease use, or worker ticket construction.
- Phase 2 Cut 3 is complete. Session close, stale session timeout, socket
  disconnect for explicitly connection-scoped sessions, worker failure, and
  detected completion mismatch now release daemon-owned reservations, lease
  tokens, staging records, buffers, relay quota, and transfer state. Cleanup
  responses report released resource counts, failed workers release staging
  resources without client policy, repeated cleanup is idempotent, and direct
  transfers without relay reservations are canceled when their owning job or
  buffer is cleaned.
- Phase 2 Cut 4 is complete. The daemon now owns audit records for relay
  authorization, worker completion, worker failure, cleanup, stale session
  timeout, lease expiration, socket disconnect through session cleanup, and
  detected mismatch. Audit records are exposed through daemon profile state
  and include owner identity, transfer id, decision id, ticket id, topology
  snapshot id, lease id, session id, job id, relay GPU, direction, bytes,
  duration, buffers, staging record id, cleanup target, reason, and failure
  reason without letting clients or adapters author audit truth.
- Phase 3 Cut 1 is complete. The daemon now keeps a global transfer queue and
  scheduler-readable runtime resource snapshot in profile state. Transfer
  submission records queued work by intent, job, session, workload kind, and
  priority; runtime snapshots now surface queued, running, active, relay
  staging, reservation, lease, H2D, D2H, and relay-path usage; scheduler plan
  metadata now receives a runtime-state summary so later fairness and
  admission control can reason over queued and active work without returning
  to adapter-side path selection.
- Phase 3 Cut 2 is complete. `JobIdentity` now carries a daemon-owned positive
  weight, job registration and daemon profile state preserve that weight, and
  runtime snapshots expose per-job queued, running, active-byte, and weight
  records. The daemon now passes workload kind and priority into scheduler
  planning. Scheduler decisions now consume runtime state to avoid busy relay
  paths, compute weighted fair-share policy metadata, and prefer direct
  fallback when the requesting job already exceeds its weighted share.
- Phase 3 Cut 3 is complete. The daemon now separates scheduler planning from
  relay admission, tracks per-transfer admission state, plan generation, and
  plan expiration, delays relay lease grants when active relay paths or quotas
  make admission unsafe, and can reschedule queued work after runtime state
  changes. Workers and lease validation now reject delayed, expired, or stale
  plans before data movement, while applications and adapters still submit
  only `TransferIntent` and consume `TransferReceipt`.

## Active Phase

Phase 4: Daemon-Plan Data Plane.

Phase 1 is complete:

- daemon-owned topology provider boundaries;
- production GPU identity and PCI bus discovery;
- PCIe hierarchy and link capability discovery;
- CUDA P2P and scale-up fabric discovery where available;
- versioned `TopologySnapshot` creation and invalidation;
- startup failure when production topology cannot satisfy policy.

Phase 2 is complete:

- socket peer credential checks;
- user, process, container, job, and session ownership;
- buffer registration with ownership checks;
- transfer state, lease, reservation, and staging lifecycle cleanup;
- audit records for relay use and failures.

Phase 3 is complete:

- global daemon transfer queue;
- runtime state for H2D, D2H, P2P, relay staging, and active transfers;
- scheduling from topology, measured bandwidth, current load, request size,
  workload kind, job weight, and fairness policy;
- weighted fair sharing across jobs;
- relay admission control, delayed lease grants, plan expiration, and
  rescheduling;
- direct fallback as a scheduler outcome.

Phase 4 covers:

- exact daemon-issued plans as the only production data-plane input;
- worker-managed direct, relay, and pooled execution from `ExecutionTicket`;
- staging buffer lifecycle through daemon or worker cleanup;
- shared ticket and receipt semantics for H2D, D2H, and range transfers;
- correctness tests for direct, relay, pooled, and failure paths.

Phase 4 Cut 1 is complete. Worker request construction now rejects daemon
responses without `ExecutionTicket` payloads, non-daemon-issued tickets,
missing plan generations, stale plan generations, and mismatched ticket,
decision, buffer, lease, range, or plan fields before any worker staging or
backend execution. Worker-managed direct fallback now also consumes the
daemon-issued ticket plan instead of executing a bare daemon plan payload.
Daemon worker authorization returns ticket, decision, source buffer,
destination buffer, lease, transfer id, plan generation, and staging record
data without exposing a separate executable authorization plan.

Phase 4 Cut 2 is complete. Worker request construction can now derive one
worker data-plane request from a daemon-issued `ExecutionTicket` that contains
multiple relay leases and relay GPU paths. The CUDA worker initializes every
relay GPU named by the ticketed plan, preserves direct plus multi-relay path
stats, and still feeds the exact daemon plan into `fetch_plan_to_gpu` or
`offload_plan_to_cpu`. Daemon worker authorization now returns the full relay
lease set and per-relay staging records for one ticketed pooled transfer, while
applications and adapters still submit only transfer intent and consume
receipts.

Phase 4 Cut 3 is complete. Worker cleanup now releases every relay lease for
completed multi-relay ticketed transfers and cleans every relay lease on
status-report or worker failure paths. Completion envelopes preserve the full
lease set, client-side release validation accepts aggregate multi-lease
responses only when all lease responses are ok, and daemon state tests confirm
multi-relay staging records, reservations, relay quotas, and runtime state are
removed after completion.

## Next Work Items

Current item: Phase 4 Cut 4, legacy Runtime data-plane cleanup and GPU
correctness gate.

1. Shared schema layer.
   - Status: complete.
   - Add JobIdentity, BufferHandle, TransferIntent, TopologySnapshot,
     SchedulingDecision, ExecutionTicket, and TransferReceipt.
   - Add validation tests under `test/python/unit/`.

2. Package boundary setup.
   - Status: complete.
   - Introduce `api`, `control`, `topology`, `scheduler`, and `data_plane`
     package boundaries.
   - Keep adapters under `adapters`.
   - Move synthetic topology support into fixtures.

3. Daemon-first client API.
   - Status: complete.
   - Add intent submission.
   - Add receipt-oriented completion.
   - Update package exports.

4. Scheduler and ticket contract.
   - Status: complete.
   - Make SchedulingDecision the scheduler output.
   - Make ExecutionTicket the worker input.
   - Add rejection tests for ticket mismatches.

5. Test tree rewrite.
   - Status: complete.
   - Organize tests into unit, integration, e2e, and fixtures.
   - Mark GPU-required tests clearly.

6. Benchmark and example rewrite.
   - Status: complete.
   - Substage 7.1 complete: daemon `TransferIntent` submission now produces
     `TransferReceipt` records with decision id, topology snapshot id,
     execution ticket id, bytes, path split, and fallback reason. Benchmark
     helper code can construct workload intent and format receipt traces
     without physical path hints.
   - Substage 7.2 complete: model-loading benchmark submission now goes through
     the public client API and consumes daemon receipts. It no longer exposes
     target GPU, relay GPU, or physical transfer mode CLI controls.
   - Substage 7.3 complete: training-offload benchmark submission now goes
     through the public client API for paired H2D prefetch and D2H offload
     intent and consumes daemon receipts for both directions.
   - Substage 7.4 complete: examples and paper-validation command/output
     handling no longer expect applications to choose physical transfer paths
     for the rewritten model-loading and training-offload benchmarks.

7. Adapter thinning.
   - Status: complete.
   - Substage 8.1 complete: the shared adapter named-block path now submits
     `TransferIntent` through the client API and consumes `TransferReceipt`
     for stats and block state. Model-loading, training-offload, inference KV,
     vLLM slot, and vLLM integration adapters now use that path.
   - Substage 8.2 complete: rewrite `vllm_kv_connector` so it removes Runtime,
     target GPU, relay GPU, transfer mode, and min-pool-byte configuration from
     the adapter surface and reports daemon receipt ids and path split instead.
   - Substage 8.3 complete: clean adapter-facing exports/tests that still only
     protect old Runtime route-selection behavior.

8. Remaining legacy benchmark and example cleanup.
   - Status: complete.
   - Removed active examples and benchmarks that still constructed `Runtime`
     or exposed target GPU, relay GPU, or physical mode selection for workload
     submission.
   - Removed files: `examples/vllm_turbobus_restore.py`,
     `benchmarks/bandwidth_pool.py`, `benchmarks/kv_offload.py`, and
     `benchmarks/tune_transfer.py`.
   - Reworked `benchmarks/summarize_result.py` to dispatch only current
     daemon-first benchmark JSON shapes.
   - Keep direct, relay, and pooled path coverage inside scheduler, worker, or
     data-plane tests where daemon decisions and tickets are authoritative.

9. Automatic topology discovery.
   - Status: complete.
   - Cut 1 complete: daemon-owned topology provider boundary, CUDA/NVML
     provider, versioned topology snapshot ids, production startup provider
     selection, fixture rejection, and startup relay policy validation.
   - Cut 2 complete: normalize GPU UUID, PCI bus id, NUMA, memory, PCIe link,
     and fabric capability fields from production discovery; expose per-relay
     path capabilities in daemon discovery output.
   - Cut 3 complete: add daemon topology refresh/invalidation behavior and
     complete relay discovery reporting so Phase 1 can be validated on a real
     multi-GPU server.

10. Privileged daemon control plane.
   - Status: complete.
   - Cut 1 complete: add peer identity and socket credential foundation.
   - Cut 2 complete: enforce buffer ownership for registration, transfer,
     lease, and worker authorization paths.
   - Cut 3 complete: clean stale resources after disconnects, timeouts, worker
     failures, and detected mismatches.
   - Cut 4 complete: emit daemon-owned audit records for relay use, ownership,
     bytes, duration, cleanup, and failures.

11. Cross-job dynamic scheduling.
   - Status: complete.
   - Cut 1 complete: add a global daemon transfer queue and scheduler-readable
     runtime resource state for queued and active work.
   - Cut 2 complete: feed scheduler decisions from daemon runtime state,
     workload kind, priority, job weight, queue depth, and active resource
     usage; avoid busy relays and explain weighted fairness fallback.
   - Cut 3 complete: add relay admission control, delayed lease grants, plan
     expiration, and rescheduling.

12. Daemon-plan data plane.
   - Status: current.
   - Cut 1 complete: audit and tighten worker/data-plane execution entry points
     so production paths execute only daemon-issued `ExecutionTicket` plans.
   - Cut 2 complete: extend worker-managed execution from one relay lease to
     daemon-ticketed direct plus multi-relay pooled plans.
   - Cut 3 complete: make staging cleanup and receipt semantics deterministic
     across H2D, D2H, range, direct, relay, pooled, and failure paths.
   - Cut 4 current: remove, demote, or reroute the legacy `turbobus.runtime`
     Runtime-local execution path and add the GPU correctness validation gate.
   - Keep direct, relay, and pooled paths as scheduler outcomes and data-plane
     behaviors, not app-side controls.

## Phase 0 Acceptance Criteria

Phase 0 is complete:

- main transfer calls require daemon scheduling;
- scheduler decisions are the only production transfer plans;
- workers require ExecutionTickets;
- synthetic topology is explicit fixture data;
- default tests are organized by unit, integration, e2e, and fixtures;
- examples and benchmarks call the public client API;
- `python -m compileall` passes;
- non-GPU tests pass;
- GPU tests are clearly marked and runnable on CUDA hardware.

## Latest Validation

Phase 4 Cut 3 validation:

- `python -m unittest test.python.integration.test_worker_helper test.python.integration.test_client_worker_transfer test.python.integration.test_daemon_state`
- `python -m py_compile turbobus\worker\helper.py turbobus\client_transfer.py turbobus\daemon\server.py test\python\integration\test_worker_helper.py test\python\integration\test_client_worker_transfer.py test\python\integration\test_daemon_state.py`
- `git diff --check`

Remaining risk:

- Phase 1 implementation is complete, but production topology behavior still
  needs to be exercised on a real multi-GPU CUDA server with `nvidia-smi topo
  -m` available.
- Phase 3 now covers queue state, weighted scheduling inputs, daemon-owned
  relay admission, delayed lease grants, plan expiration, and rescheduling in
  non-GPU control-plane tests. Production behavior still needs to be exercised
  on a real multi-GPU CUDA server under concurrent workload pressure.
- Phase 4 Cut 3 added deterministic multi-relay lease and staging cleanup in
  non-GPU tests. It still needs to be exercised on a real multi-GPU CUDA
  server.
- The legacy `turbobus.runtime` module still contains importable Runtime-local
  execution code and must be removed, demoted to explicit legacy-internal
  tests, or routed through daemon-issued tickets before Phase 4 is considered
  complete.
- Phase 4 still needs legacy `turbobus.runtime` removal, demotion, or
  ticket-routing, plus GPU correctness coverage.

Latest validation:

- `python -m unittest test.python.integration.test_worker_helper test.python.integration.test_client_worker_transfer test.python.integration.test_daemon_state`
- `python -m py_compile turbobus\\worker\\helper.py turbobus\\client_transfer.py turbobus\\daemon\\server.py test\\python\\integration\\test_worker_helper.py test\\python\\integration\\test_client_worker_transfer.py test\\python\\integration\\test_daemon_state.py`
- `git diff --check`

## Upcoming Phases

After Phase 0, proceed in order:

1. Automatic topology discovery.
2. Privileged daemon control plane. Complete.
3. Cross-job dynamic scheduling. Complete.
4. Daemon-plan data plane. Current.
5. vLLM KV end-to-end workload.
6. Model loading and training offload.
7. Paper evaluation and hardening.
