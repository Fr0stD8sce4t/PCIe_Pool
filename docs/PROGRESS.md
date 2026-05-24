# TurboBus Progress

## Current State

The active project plan has been reset to paper-parity execution.

The current repository still contains implementation code, tests, benchmarks,
and examples that need to be realigned around daemon-first scheduling. The next
work is Phase 0, which updates the structure and contracts before adding broad
new features.

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

## Active Phase

Phase 0: Paper-Parity Realignment.

Phase 0 covers:

- shared schema contracts;
- package boundary setup;
- daemon-first public API;
- scheduler and ticket contract;
- test tree rewrite;
- benchmark and example rewrite;
- adapter thinning.

## Next Work Items

Current item: Cut 4, Daemon-first client API.

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
   - Status: current.
   - Add intent submission.
   - Add receipt-oriented completion.
   - Update package exports.

4. Scheduler and ticket contract.
   - Make SchedulingDecision the scheduler output.
   - Make ExecutionTicket the worker input.
   - Add rejection tests for ticket mismatches.

5. Test tree rewrite.
   - Organize tests into unit, integration, e2e, and fixtures.
   - Mark GPU-required tests clearly.

6. Benchmark and example rewrite.
   - Use public client APIs.
   - Emit decision id, topology snapshot id, ticket id, bytes, timing, path
     split, and fallback reason.

7. Adapter thinning.
   - Update vLLM, model loading, and training adapters to submit TransferIntent.
   - Consume TransferReceipt for stats and state transitions.

## Phase 0 Acceptance Criteria

Phase 0 is done when:

- main transfer calls require daemon scheduling;
- scheduler decisions are the only production transfer plans;
- workers require ExecutionTickets;
- synthetic topology is explicit fixture data;
- default tests are organized by unit, integration, e2e, and fixtures;
- examples and benchmarks call the public client API;
- `python -m compileall` passes;
- non-GPU tests pass;
- GPU tests are clearly marked and runnable on CUDA hardware.

## Upcoming Phases

After Phase 0, proceed in order:

1. Automatic topology discovery.
2. Privileged daemon control plane.
3. Cross-job dynamic scheduling.
4. Daemon-plan data plane.
5. vLLM KV end-to-end workload.
6. Model loading and training offload.
7. Paper evaluation and hardening.
