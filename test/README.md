# Test Strategy

TurboBus tests are organized around the new system plan.

## What The Tests Should Cover

- protocol validation for daemon/client/worker messages;
- planner behavior for direct, relay, and pooled chunk assignment;
- backend behavior for CUDA and later ROCm implementations;
- daemon session, lease, and quota handling;
- transfer stats and fallback behavior;
- framework adapters for vLLM, model loading, and training offload;
- benchmark-friendly summary output.

## Test Layers

1. Unit tests
   - Fast checks for request validation, path selection, stats shaping, and
     adapter state handling.

2. Integration tests
   - Daemon/client interactions.
   - Multi-process or worker-assisted transfer flow.
   - Framework adapter lifecycle behavior.

3. Native checks
   - CUDA path execution.
   - planner and executor parity.
   - profile and path timing behavior.

## Rules

- Keep tests aligned with the current architecture plan.
- Do not encode old single-process assumptions into new tests.
- Prefer focused tests over broad discovery when a change only touches one
  layer.
