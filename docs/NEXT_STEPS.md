# TurboBus Next Steps

## Current Direction

TurboBus is being advanced directly toward paper parity. The immediate work is
to realign the codebase around daemon-first scheduling, automatic topology
discovery, execution tickets, and public TransferIntent APIs.

The next implementation work must update main code, tests, benchmarks,
examples, exports, and adapters together. Tests and examples should protect the
new architecture rather than preserve application-side route selection.

## Immediate Functional Target

Complete Phase 0: Paper-Parity Realignment.

The first target is not a new transfer optimization. It is a structural cut that
creates the contracts needed for the rest of the system:

1. Define the shared schema objects:
   - JobIdentity;
   - BufferHandle;
   - TransferIntent;
   - TopologySnapshot;
   - SchedulingDecision;
   - ExecutionTicket;
   - TransferReceipt.
2. Route the public transfer API through daemon scheduling.
3. Make scheduler output the only production source of transfer plans.
4. Make worker execution require ExecutionTicket validation.
5. Move synthetic topology into explicit test fixtures.
6. Rewrite tests, examples, and benchmarks around the public client API.

## Current

Cut 8: Adapter Thinning.

Cut 1 is complete. The contract inventory is recorded in
`docs/PHASE0_CONTRACT_INVENTORY.md`.

Cut 2 is complete. The shared daemon-first schema objects now live in
`turbobus/schema.py`, with contract tests under `test/python/unit/`.

Cut 3 is complete. The package now exposes explicit `api`, `control`,
`topology`, `scheduler`, and `data_plane` boundaries; synthetic topology lives
under `test/python/fixtures/`.

Cut 4 is complete. The public client API now submits `TransferIntent` objects
and waits for `TransferReceipt` objects. Root package exports now emphasize the
daemon-first public API and shared contract objects.

Cut 5 is complete. The daemon scheduler now returns schema
`SchedulingDecision` objects, daemon worker authorization issues
`ExecutionTicket` objects, and worker request construction can validate ticket
bindings against decisions, buffers, byte ranges, leases, and daemon plans.

Cut 6 is complete. Tests now live under `test/python/unit/`,
`test/python/integration/`, `test/python/e2e/`, and
`test/python/fixtures/`. Moved tests import internal modules explicitly instead
of widening the root public API, and synthetic topology remains an explicit
fixture.

Cut 7 Substage 7.1 is complete. The daemon can accept public
`TransferIntent` submission and return `TransferReceipt` records for benchmark
code.

Cut 7 Substage 7.2 is complete. The model-loading benchmark now submits
model-weight `TransferIntent` objects through the public client API, treats
benchmark policy as metadata instead of physical path selection, and reports
daemon receipt ids, path split, bytes, timing, and fallback reason.

Cut 7 Substage 7.3 is complete. The training-offload benchmark now submits H2D
prefetch and D2H offload `TransferIntent` objects through the public client
API, treats benchmark policy as metadata instead of physical path selection,
and reports separate receipt ids, decision ids, topology snapshot ids,
execution ticket ids, bytes, timing, path split, and fallback reason.

Cut 7 Substage 7.4 is complete. Paper validation now builds daemon-first
commands for the rewritten model-loading and training-offload benchmarks using
session ids and registered buffer ids instead of target GPU, relay GPU, or
physical transfer mode arguments. Paper metrics now come from daemon receipt
trace ids and path split fields. `examples/torch_tensor_fetch.py` now
demonstrates public `TransferIntent` submission and receipt reporting.

Current cut: Cut 8, Adapter Thinning.

Cut 8 Substage 8.1 is complete. The shared adapter transfer path now uses
`AdapterTransferContext` to describe job, session, buffer, workload, and
receipt-wait settings. `OffloadStore` submits `TransferIntent` objects through
the client API, waits for `TransferReceipt` objects, and derives stats and
state transitions from receipts. The model-loading, training-offload,
inference KV, vLLM slot, and vLLM integration adapters now use this shared
intent/receipt path instead of calling Runtime-owned transfer methods.

Cut 8 Substage 8.2 is complete. The vLLM KV connector now accepts daemon
socket, job, session, and registered CPU/GPU buffer identity instead of
target GPU, relay GPU, physical mode, or min-pool settings. Save and restore
flow through `AdapterTransferContext`, `TurboBusClient`, and
`VllmKVSlotAdapter`, and connector/example/sweep output reports daemon
receipt ids, decision ids, topology snapshot ids, ticket ids, bytes, path
split, and fallback reason.

Cut 8 Substage 8.3 is complete. The old non-KV vLLM connector experiment,
its root wrapper, its adapter re-export, its route-shaped example, and its
test have been removed. Adapter-facing exports now protect the daemon-first
KV connector, vLLM mapping, vLLM integration, intent fields, receipt handling,
and package boundaries.

Current cut: Cut 9, remaining legacy benchmark and example cleanup.

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

Status: current.

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

Status: current.

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

Current known files to resolve:

- `examples/vllm_turbobus_restore.py`;
- `benchmarks/bandwidth_pool.py`;
- `benchmarks/kv_offload.py`;
- `benchmarks/tune_transfer.py`;
- tests that only guard rejection of removed benchmark/example route-shaped
  arguments should either move to Cut 9 validation or be removed with the
  old entry points.

Expected output:

- public examples and benchmarks call daemon-first APIs;
- no active example or benchmark constructs `Runtime` for workload submission;
- route-shaped controls remain only in scheduler, worker, data-plane, or
  explicit legacy-internal tests.

## Phase 0 Done Criteria

Phase 0 is complete when:

- the main transfer path requires daemon scheduling;
- scheduler decisions are the only production transfer plans;
- workers require ExecutionTickets;
- synthetic topology is explicit test fixture data;
- default tests are organized by unit, integration, e2e, and fixtures;
- examples and benchmarks call the public client API;
- `python -m compileall` passes;
- non-GPU tests pass;
- GPU tests are clearly marked and runnable on CUDA hardware.

## After Phase 0

Proceed in this order:

1. Automatic topology discovery.
2. Privileged daemon control plane.
3. Cross-job dynamic scheduling.
4. Daemon-plan data plane.
5. vLLM KV end-to-end workload.
6. Model loading and training offload.
7. Paper evaluation and hardening.
