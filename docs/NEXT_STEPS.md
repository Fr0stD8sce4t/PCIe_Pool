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

Cut 5: Scheduler And Ticket Contract.

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

Status: current.

- Make SchedulingDecision the scheduler output.
- Make ExecutionTicket the worker input.
- Validate that tickets bind job, session, buffers, byte ranges, topology
  snapshot, and scheduling decision.
- Add rejection tests for mismatched tickets.

Expected output:

- worker unit tests reject modified tickets;
- scheduler tests assert decisions are explainable and traceable.

### Cut 6: Test Tree Rewrite

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

- Update examples to call the public client API.
- Update benchmark configuration to request workload and policy, then read
  actual path information from daemon decisions and receipts.
- Require benchmark output to include decision id, topology snapshot id, ticket
  id, bytes, timing, path split, and fallback reason.

Expected output:

- examples demonstrate daemon-first submission;
- benchmarks no longer encode transfer path policy in application code.

### Cut 8: Adapter Thinning

- Update vLLM, model loading, and training adapters to submit TransferIntent.
- Make adapters consume TransferReceipt for stats and state transitions.
- Keep framework-specific mapping logic in adapters and physical path policy in
  scheduler.

Expected output:

- adapter tests assert TransferIntent construction and receipt handling;
- no adapter owns path selection policy.

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
