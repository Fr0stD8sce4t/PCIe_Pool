# TurboBus Progress

## Current State

The active project plan has been reset to paper-parity execution.

Phase 0 realignment, Phase 1 automatic topology discovery, Phase 2
privileged daemon control plane, Phase 3 cross-job dynamic scheduling, Phase 4
daemon-plan data plane, Phase 5 vLLM KV end-to-end workload, and Phase 6 model
loading and training offload are complete in the local daemon-first code path.
The next work is Phase 7, paper evaluation and hardening.

The active target architecture is:

- public APIs submit TransferIntent;
- daemon topology providers produce TopologySnapshot;
- daemon scheduler produces SchedulingDecision;
- daemon control plane issues ExecutionTicket;
- workers and data-plane backends execute exact ticketed plans;
- completion returns TransferReceipt;
- vLLM KV cache save/restore is the first completed full workload target;
- model weight loading and training or optimizer state offload now share the
  same public transfer and report path;
- paper evaluation and hardening are the next workload validation targets.

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

Phase 7: Paper Evaluation And Hardening.

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

Phase 4 Cut 4 is complete. The importable `turbobus.runtime` wrapper and
`transfer_selector` application-side route selector have been removed. The
native Python extension no longer exposes local-planning methods
`set_transfer_mode`, `fetch_to_gpu`, `offload_to_cpu`,
`fetch_ranges_to_gpu`, or `offload_ranges_to_cpu`; only exact-plan backend
primitives remain available for daemon-ticketed execution. Runtime tests now
protect `runtime_engine` exact-plan conversion helpers and public package
boundaries instead of old direct/relay/pool route selection.

Phase 5 is complete. The vLLM KV connector now submits save and restore
`TransferIntent` objects from real vLLM lifecycle points, consumes
`TransferReceipt` objects, and records daemon receipt ids, decision ids,
topology snapshot ids, ticket ids, bytes, timing, fallback reason, and
direct/relay path split. Paper validation can run both a single vLLM KV
save/restore job and concurrent multi-job vLLM KV save/restore jobs through
the daemon-first connector path. Multi-job validation assigns distinct
job/session/buffer/prefix identities to each job, aggregates per-job daemon
trace output for fairness audits, and rejects missing per-job daemon trace
without adding target GPU, relay GPU, mode, or direct/relay/pool controls.

Phase 6 is complete. Model-loading, training-state offload, optimizer-state
offload, and vLLM KV validation now share the same public client API and the
same `phase6_unified_v1` correctness/performance report shape. Paper
validation requires receipt ids, decision ids, topology snapshot ids, ticket
ids, bytes, completion status, timing, path split, fallback reason, workload
kind, job/session identity, and registered buffer identity for every workload.

## Next Work Items

Current item: Phase 7 Cut 7, real-server execution and artifact ingestion.

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
   - Status: complete.
   - Cut 1 complete: audit and tighten worker/data-plane execution entry points
     so production paths execute only daemon-issued `ExecutionTicket` plans.
   - Cut 2 complete: extend worker-managed execution from one relay lease to
     daemon-ticketed direct plus multi-relay pooled plans.
   - Cut 3 complete: make staging cleanup and receipt semantics deterministic
     across H2D, D2H, range, direct, relay, pooled, and failure paths.
   - Cut 4 complete: remove the legacy `turbobus.runtime` Runtime-local
     execution path, remove old application-side route selector code, close
     native extension local-planning bindings, and document the GPU correctness
     validation gate.
   - Keep direct, relay, and pooled paths as scheduler outcomes and data-plane
     behaviors, not app-side controls.

13. vLLM KV end-to-end workload.
   - Status: complete.
   - Cut 1 complete: tighten the vLLM KV connector around the real
     `KVConnectorBase_V1` save/restore lifecycle, remove the public fake saved
     prefix injection path, remove the non-layer `wait_for_save` fallback, and
     protect daemon-first intent and receipt trace fields for save and restore.
   - Cut 2 complete: add a paper-validation vLLM KV workload that runs the
     real connector example on a CUDA server, requires lifecycle events, parses
     save/restore daemon trace fields, writes JSON, and reports vLLM KV paper
     metrics without application-side path controls.
   - Cut 3 complete: add concurrent multi-job vLLM KV validation through
     paper validation. Each job gets distinct job/session/buffer/prefix
     identity, runs the daemon-first connector example, and contributes
     per-job save/restore receipt ids, decision ids, topology snapshot ids,
     ticket ids, bytes, direct/relay path split, fallback reason, timing, and
     log path to the paper-validation JSON and summary.

14. Model loading and training offload.
   - Status: complete.
   - Cut 1 complete: added `docs/PHASE6_WORKLOAD_BOUNDARY_INVENTORY.md`,
     confirmed model-loading and training-offload benchmarks and adapters use
     public `TransferIntent` and `TransferReceipt` paths, added explicit
     `workload_kind=model_weights` to model-loading benchmark config, and made
     paper validation report and validate Phase 6 workload identity,
     registered buffer identity, and workload kind without adding physical path
     controls.
   - Cut 2 complete: added first-class `optimizer-offload` paper-validation
     coverage that runs through the public training-offload benchmark path with
     fixed `workload_kind=optimizer_state`; fixed `training-offload` paper
     validation to represent `workload_kind=training_state`; added focused
     scheduler tests proving `model_weights`, `training_state`, and
     `optimizer_state` reach policy metadata and request charge accounting.
   - Cut 3 complete: added the `phase6_unified_v1` shared paper-validation
     report schema across vLLM KV, model-loading, training-offload, and
     optimizer-offload; added receipt-id summaries to model-loading and
     training-offload benchmark outputs; and made validation reject missing
     unified report fields, incomplete byte completion, or invalid correctness
     status without adding physical path controls.

15. Paper evaluation and hardening.
   - Status: current.
   - Cut 1 complete: added `docs/PHASE7_EVALUATION_MATRIX.md` with daemon
     startup, 2 GPU/4 GPU/8 GPU experiment axes, single-job baseline and
     TurboBus paper-validation commands, multi-job vLLM KV fairness commands,
     correctness-gate commands, required trace fields, and pass/fail criteria.
     The commands use registered buffer ids and benchmark policy labels; they
     do not add application-side target GPU, relay GPU, mode, or path controls.
     Benchmark CLI entrypoints now run from the repository root without manual
     `PYTHONPATH` setup.
   - Cut 2 complete: added `benchmarks/phase7_result_check.py`, a standalone
     experiment-facing checker that consumes paper-validation JSON or compact
     summary output, validates the `phase6_unified_v1` trace contract, reports
     machine-readable errors for missing ids, path split, completion mismatch,
     fallback/failure state, and multi-job identity problems, and exits nonzero
     on failure without touching scheduler or data-plane modules.
   - Cut 3 complete: added `benchmarks/phase7_compare.py`, a standalone
     experiment-facing comparison tool for checker-approved baseline-label and
     `turbobus-daemon` paper-validation result files. The report preserves
     receipt ids, decision ids, topology snapshot ids, ticket ids, workload
     identity, transfer time, throughput, bytes moved, direct/relay path
     split, fallback reason, and repeated-run p50/p99 fields when raw samples
     are available. It states that path split comes from daemon transfer
     receipts and does not imply application-side path selection.
   - Cut 4 complete: added `benchmarks/phase7_evidence.py`, a standalone
     experiment-facing evidence tool that consumes accepted paper-validation
     result files, daemon `PROFILE` payloads or a live daemon socket, and
     optional Phase 7 comparison JSON. The report reruns the Phase 7 result
     checker and attaches runtime transfer records, active resource usage,
     relay quota state, relay staging records, audit records, job runtime
     state, fallback reasons, and failure reasons to each metric when those
     daemon-owned records are present. Missing daemon trace evidence is
     reported as a machine-readable error, and the tool does not create
     scheduler plans, issue execution tickets, or select physical paths.
   - Cut 5 complete: added `benchmarks/phase7_bundle_gate.py`, a standalone
     experiment-facing run-bundle gate that consumes baseline and
     `turbobus-daemon` paper-validation results, optional saved checker
     reports, comparison JSON, one or more daemon evidence JSON files, and
     optional correctness-gate JSON for one server class. The gate recomputes
     result checks from source result files, validates provided checker reports
     when present, enforces required workload membership, policy labels,
     comparison coverage, daemon evidence coverage, and vLLM KV real-workload
     presence. It reports omitted correctness artifacts as warnings and does
     not create scheduler plans, issue execution tickets, run worker data
     movement, or select physical paths.
   - Cut 6 complete: added `benchmarks/phase7_acceptance_inventory.py`, a
     standalone experiment-facing acceptance inventory that consumes a
     server-class manifest and existing bundle-gate outputs for 2 GPU, 4 GPU,
     and 8 GPU systems. It records accepted real-server bundles, explicit
     hardware/environment gaps, next operator commands, and remaining
     server-only risks. It fails when no accepted real-artifact bundle contains
     vLLM KV, when accepted entries are not marked as real artifacts, when
     bundle gates are missing or failed, or when missing server classes lack
     explicit gaps and next commands.
   - Current item: Phase 7 Cut 7, real-server execution and artifact
     ingestion.
   - Cut 7 Substage 7.1 complete: added
     `benchmarks/phase7_ingest_artifacts.py`, a standalone experiment-facing
     manifest ingestion command. It updates one server-class entry at a time,
     requires accepted bundle-gate entries to be explicit real server
     artifacts, requires blocked or missing entries to carry a
     hardware/environment gap and next command, reruns the acceptance inventory
     after writing the manifest, and can write `acceptance-inventory.json`
     without creating scheduler plans, issuing execution tickets, running
     worker data movement, or selecting physical paths.
   - Cut 7 Substage 7.2 complete: added
     `benchmarks/phase7_server_run.py`, a server-side Phase 7 artifact-chain
     command that runs baseline-label paper validation, TurboBus paper
     validation, result checks, comparison, daemon evidence, bundle gate, and
     acceptance manifest ingestion for one server class. Its dry-run mode
     emits the full command plan and expected artifact paths for local
     validation, and the generated workload commands do not include
     application-side target GPU, relay GPU, direct/relay/pool, or mode
     controls.
   - Next, run or ingest real Phase 7 server artifacts, update the acceptance
     manifest for each 2 GPU, 4 GPU, and 8 GPU server class, and produce
     `benchmarks/results/phase7/acceptance-inventory.json`.

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

Phase 7 Cut 7 Substage 7.1 validation:

- `python -m unittest test.python.e2e.test_phase7_ingest_artifacts`
- `python -m py_compile benchmarks\phase7_ingest_artifacts.py test\python\e2e\test_phase7_ingest_artifacts.py`
- `python benchmarks\phase7_ingest_artifacts.py --help`
- `git diff --check`

Phase 7 Cut 7 Substage 7.2 validation:

- `python -m unittest test.python.e2e.test_phase7_server_run`
- `python -m py_compile benchmarks\phase7_server_run.py test\python\e2e\test_phase7_server_run.py`
- `python benchmarks\phase7_server_run.py --help`
- `git diff --check`

Phase 7 Cut 6 validation:

- `python -m unittest test.python.e2e.test_phase7_acceptance_inventory`
- `python -m py_compile benchmarks\phase7_acceptance_inventory.py test\python\e2e\test_phase7_acceptance_inventory.py`
- `python benchmarks\phase7_acceptance_inventory.py --help`
- `git diff --check`

Phase 7 Cut 5 validation:

- `python -m unittest test.python.e2e.test_phase7_bundle_gate`
- `python -m py_compile benchmarks\phase7_bundle_gate.py test\python\e2e\test_phase7_bundle_gate.py`
- `python benchmarks\phase7_bundle_gate.py --help`
- `git diff --check`

Phase 7 Cut 4 validation:

- `python -m unittest test.python.e2e.test_phase7_evidence`
- `python -m py_compile benchmarks\phase7_evidence.py test\python\e2e\test_phase7_evidence.py`
- `python benchmarks\phase7_evidence.py --help`
- `git diff --check`

Phase 7 Cut 3 validation:

- `python -m unittest test.python.e2e.test_phase7_compare`
- `python -m py_compile benchmarks\phase7_compare.py test\python\e2e\test_phase7_compare.py`
- `python benchmarks\phase7_compare.py --help`
- `git diff --check`

Phase 7 Cut 2 validation:

- `python -m unittest test.python.e2e.test_phase7_result_check`
- `python -m py_compile benchmarks\phase7_result_check.py test\python\e2e\test_phase7_result_check.py`
- `python benchmarks\phase7_result_check.py --help`
- `git diff --check`

Phase 7 Cut 1 validation:

- `python -m unittest test.python.e2e.test_benchmark_cli_entrypoints`
- `python -m py_compile benchmarks\paper_validation.py benchmarks\model_loading.py benchmarks\training_offload.py test\python\e2e\test_benchmark_cli_entrypoints.py`
- `python benchmarks\paper_validation.py --help`
- `python benchmarks\model_loading.py --help`
- `python benchmarks\training_offload.py --help`
- `git diff --check`

Phase 6 Cut 3 validation:

- `python -m unittest test.python.e2e.test_paper_validation test.python.e2e.test_model_loading_benchmark test.python.e2e.test_training_offload_benchmark test.python.unit.test_daemon_scheduler`
- `python -m py_compile benchmarks\paper_validation.py benchmarks\model_loading.py benchmarks\training_offload.py test\python\e2e\test_paper_validation.py test\python\e2e\test_model_loading_benchmark.py test\python\e2e\test_training_offload_benchmark.py`
- `git diff --check`

Phase 6 Cut 2 validation:

- `python -m unittest test.python.e2e.test_paper_validation test.python.e2e.test_training_offload_benchmark test.python.unit.test_daemon_scheduler`
- `python -m py_compile benchmarks\paper_validation.py benchmarks\training_offload.py test\python\e2e\test_paper_validation.py test\python\e2e\test_training_offload_benchmark.py test\python\unit\test_daemon_scheduler.py turbobus\scheduler\daemon.py`
- `git diff --check`

Phase 6 Cut 1 validation:

- `python -m unittest test.python.e2e.test_paper_validation test.python.e2e.test_model_loading_benchmark test.python.e2e.test_training_offload_benchmark test.python.e2e.test_model_loading test.python.e2e.test_training_offload test.python.unit.test_offload_store test.python.unit.test_daemon_scheduler`
- `python -m py_compile benchmarks\model_loading.py benchmarks\training_offload.py benchmarks\paper_validation.py benchmarks\daemon_support.py turbobus\adapters\model_loading.py turbobus\adapters\training_offload.py turbobus\offload_store.py test\python\e2e\test_paper_validation.py test\python\e2e\test_model_loading_benchmark.py test\python\e2e\test_training_offload_benchmark.py test\python\e2e\test_model_loading.py test\python\e2e\test_training_offload.py test\python\unit\test_offload_store.py test\python\unit\test_daemon_scheduler.py`
- `git diff --check`

Phase 5 Cut 3 validation:

- `python -m unittest test.python.e2e.test_paper_validation`
- `python -m unittest test.python.e2e.test_paper_validation test.python.e2e.test_vllm_kv_connector_example test.python.e2e.test_vllm_kv_connector_sweep test.python.e2e.test_vllm_kv_connector`
- `python -m py_compile benchmarks\paper_validation.py examples\vllm_turbobus_kv_connector.py examples\vllm_turbobus_kv_connector_sweep.py test\python\e2e\test_paper_validation.py test\python\e2e\test_vllm_kv_connector.py test\python\e2e\test_vllm_kv_connector_example.py test\python\e2e\test_vllm_kv_connector_sweep.py turbobus\adapters\vllm_kv_connector.py`
- `git diff --check`

Earlier Phase 4 Cut 4 validation:

- `python -m unittest test.python.unit.test_public_client_api test.python.unit.test_runtime_engine test.python.unit.test_backend_cuda test.python.unit.test_worker_cuda_executor test.python.e2e.test_verification test.python.integration.test_client_worker_transfer test.python.integration.test_worker_helper test.python.integration.test_daemon_state`
- `python -m py_compile turbobus\runtime_engine.py turbobus\backends\cuda.py turbobus\client_transfer.py turbobus\worker\cuda_executor.py turbobus\verification.py test\python\unit\test_public_client_api.py test\python\unit\test_runtime_engine.py test\python\unit\test_backend_cuda.py test\python\unit\test_worker_cuda_executor.py test\python\e2e\test_verification.py test\python\integration\test_client_worker_transfer.py test\python\integration\test_worker_helper.py test\python\integration\test_daemon_state.py`
- `git diff --check`

Remaining risk:

- Phase 1 implementation is complete, but production topology behavior still
  needs to be exercised on a real multi-GPU CUDA server with `nvidia-smi topo
  -m` available.
- Phase 3 now covers queue state, weighted scheduling inputs, daemon-owned
  relay admission, delayed lease grants, plan expiration, and rescheduling in
  non-GPU control-plane tests. Production behavior still needs to be exercised
  on a real multi-GPU CUDA server under concurrent workload pressure.
- Phase 4 is complete in code and non-GPU tests, but daemon-ticketed direct,
  relay, pooled, H2D, D2H, and range-offset correctness still need to be run
  on a real multi-GPU CUDA server with the documented
  `python -m turbobus.verification` commands.
- Native C++/CUDA build checks were not run in the local Windows environment
  because `cmake` and `nvcc` are not installed there.
- Phase 5 is complete in code and local non-GPU validation, but the single-job
  and multi-job vLLM KV paper-validation commands still need to be exercised on
  a CUDA server with vLLM installed and a TurboBus daemon running.
- Phase 6 is complete in the local code and non-GPU validation path, but
  model-loading, training-state offload, optimizer-state offload, and unified
  paper-validation output still need to be exercised on a CUDA server with a
  running TurboBus daemon.
- Phase 7 Cut 4 evidence tooling is covered by local synthetic profile tests,
  but real daemon profile and audit evidence still needs to be captured on the
  2 GPU, 4 GPU, and 8 GPU CUDA servers.
- Phase 7 Cut 5 bundle gating is covered by local synthetic artifact tests,
  but actual accepted bundle reports still need to be produced from real
  paper-validation, comparison, evidence, and correctness artifacts on CUDA
  servers.
- Phase 7 Cut 6 acceptance inventory is covered by local synthetic artifact
  tests, but the actual `acceptance-inventory.json` still needs to be produced
  from real 2 GPU, 4 GPU, and 8 GPU server manifests and bundle-gate outputs.
- Phase 7 Cut 7 Substage 7.1 manifest ingestion is covered by local synthetic
  artifact tests, but real accepted entries still need to be ingested from
  actual server bundle-gate outputs.
- Phase 7 Cut 7 Substage 7.2 server-run chain planning is covered by local
  dry-run tests, but the command still needs to be executed on real 2 GPU,
  4 GPU, and 8 GPU CUDA/vLLM servers.

## Upcoming Phases

After Phase 0, proceed in order:

1. Automatic topology discovery.
2. Privileged daemon control plane. Complete.
3. Cross-job dynamic scheduling. Complete.
4. Daemon-plan data plane. Complete.
5. vLLM KV end-to-end workload. Complete.
6. Model loading and training offload. Complete.
7. Paper evaluation and hardening. Current.
