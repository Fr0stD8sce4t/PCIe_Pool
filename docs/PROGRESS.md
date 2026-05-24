# TurboBus Progress

## Current State

The active project plan has been reset to paper-parity execution.

Phase 0 realignment is complete. The next work is Phase 1, automatic topology
discovery, which moves production startup away from manual or fixture topology
and toward daemon-owned machine discovery.

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

## Active Phase

Phase 1: Automatic Topology Discovery.

Phase 1 covers:

- daemon-owned topology provider boundaries;
- production GPU identity and PCI bus discovery;
- PCIe hierarchy and link capability discovery;
- CUDA P2P and scale-up fabric discovery where available;
- versioned `TopologySnapshot` creation and invalidation;
- startup failure when production topology cannot satisfy policy.

## Next Work Items

Current item: Phase 1 Cut 2, complete GPU, PCIe, and fabric capability
normalization.

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
   - Status: current.
   - Cut 1 complete: daemon-owned topology provider boundary, CUDA/NVML
     provider, versioned topology snapshot ids, production startup provider
     selection, fixture rejection, and startup relay policy validation.
   - Cut 2 current: normalize GPU UUID, PCI bus id, NUMA, memory, PCIe link,
     and fabric capability fields from production discovery.
   - Later Phase 1 work: cache invalidation policy, daemon reporting of
     eligible and filtered relay candidates for real machines, and clear
     production failure paths when discovery cannot satisfy policy.

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

Phase 1 Cut 1 validation:

- `python -m unittest test.python.unit.test_topology_provider test.python.unit.test_package_boundaries test.python.integration.test_daemon_state test.python.integration.test_daemon_socket`
- `python -m compileall -q turbobus\topology turbobus\daemon test\python\unit\test_topology_provider.py test\python\unit\test_package_boundaries.py test\python\integration\test_daemon_state.py test\python\integration\test_daemon_socket.py`
- `rg -n -g "*.py" -- "--relay-gpus|test_fixture_static|StaticTopologyProvider|synthetic topology|topology provider is required|topology_snapshot_id|CudaNvmlTopologyProvider|DaemonStartupConfig" turbobus\topology turbobus\daemon test\python\unit\test_topology_provider.py test\python\unit\test_package_boundaries.py`
- `git diff --check`

Remaining Phase 1 risk:

- the CUDA/NVML provider currently captures GPU identity, PCI bus id, memory,
  versioned snapshots, and topology-matrix fabric reachability; PCIe generation,
  negotiated width, NUMA node, measured bandwidth, and NVSwitch-specific fields
  still need richer normalization in Cut 2.
- Phase 1 is not complete until daemon discovery can report usable relay
  candidates and filtered reasons from the local production machine.

## Upcoming Phases

After Phase 0, proceed in order:

1. Automatic topology discovery.
2. Privileged daemon control plane.
3. Cross-job dynamic scheduling.
4. Daemon-plan data plane.
5. vLLM KV end-to-end workload.
6. Model loading and training offload.
7. Paper evaluation and hardening.
