# Phase 0 Contract Inventory

This inventory completes Phase 0 Cut 1. It records the current transfer-facing
entry points and classifies them against the paper-parity architecture.

## Scan Scope

The inventory was built from these source areas:

- `turbobus/`
- `test/python/`
- `examples/`
- `benchmarks/`

The scan looked for transfer submission, runtime construction, daemon planning,
worker authorization, path mode selection, relay selection, and benchmark or
example entry points.

## Production Entry Points

| Current area | Current entry points | Current role | Phase 0 classification | Required action |
| --- | --- | --- | --- | --- |
| `turbobus/runtime.py` | `Runtime`, `fetch_to_gpu`, `offload_to_cpu`, `fetch_ranges_to_gpu`, `offload_ranges_to_cpu` | Public tensor transfer facade with local runtime and daemon fallback behavior | Legacy public API and future data-plane bridge | Replace as main public route with daemon-first API. Keep exact-plan execution behind scheduler decisions and tickets. |
| `turbobus/offload_store.py` | `prefetch`, `evict`, `prefetch_many`, `evict_many` | Named block wrapper over runtime H2D/D2H | Adapter support layer | Convert callers to submit `TransferIntent` and consume `TransferReceipt`. |
| `turbobus/adapters/model_loading.py` | `load_bucket`, `load_buckets`, `load_batch`, `load_all` | Model weight loading adapter over runtime transfer | Adapter | Convert bucket loading into workload-kind `TransferIntent`. |
| `turbobus/adapters/training_offload.py` | `prefetch_bucket`, `offload_bucket`, batch variants | Training offload adapter over runtime transfer | Adapter | Convert prefetch/offload into H2D/D2H `TransferIntent`. |
| `turbobus/adapters/inference.py` | `restore_prefix`, `save_prefix` | Inference KV slot adapter over offload store | Adapter | Convert prefix save/restore into transfer intent and receipts. |
| `turbobus/adapters/vllm.py` | `restore_prefix`, `save_prefix` | vLLM-shaped wrapper around inference KV slots | Adapter | Keep framework mapping; remove direct path policy ownership. |
| `turbobus/adapters/vllm_kv_connector.py` | vLLM connector lifecycle methods | Framework connector that currently creates runtime and uses connector config | Adapter | Submit `TransferIntent`; consume `TransferReceipt`; stop carrying path policy. |
| `turbobus/client_transfer.py` | `WorkerManagedTransferClient`, `fetch_shared_cpu_to_cuda_ipc`, `offload_cuda_ipc_to_shared_cpu` | Closest current client/daemon/worker transfer path | Control/data-plane bridge | Rework around `TransferIntent`, `SchedulingDecision`, `ExecutionTicket`, and `TransferReceipt`. |
| `turbobus/daemon/client.py` | `register_session`, `reserve_transfer`, `plan_transfer`, `plan_transfer_request`, `authorize_worker_transfer` | Daemon protocol client | Control plane | Move from relay/mode shaped requests to intent/decision/ticket shaped requests. |
| `turbobus/daemon/server.py` | `register_session`, `reserve_transfer`, `plan_transfer`, `authorize_worker_transfer` | Daemon state and planning authority | Control plane | Make this the production scheduling authority and remove application route choice from protocol. |
| `turbobus/daemon/scheduler.py` | `DaemonScheduler.plan_transfer` | Daemon planner wrapper over profile/quota state | Scheduler | Make output a `SchedulingDecision`; keep plan creation here. |
| `turbobus/planner_engine.py` | `plan_transfer`, `plan_transfer_ranges` | Chunk planner with direct/relay/pool mode inputs | Scheduler internal | Keep as internal scheduling primitive; do not expose as application route selection. |
| `turbobus/worker/helper.py` | `WorkerTransferClient`, `submit_and_report`, `submit_report_and_cleanup`, `submit_report_cleanup_lifecycle` | Worker lifecycle helper around daemon authorization | Data plane/control bridge | Make worker input an `ExecutionTicket` and validate ticket binding. |
| `turbobus/worker/cuda_executor.py` | `execute_bound` | CUDA worker executor for daemon-authorized request | Data plane | Execute exact ticketed plans only. |

## Tests Requiring Realignment

| Current tests | What they exercise today | Phase 0 target |
| --- | --- | --- |
| `test/python/test_runtime_handle.py` | Local runtime mode resolution, daemon fallback, runtime facade details | Replace public-path tests with daemon-first client API tests; keep exact-plan backend behavior as internal data-plane tests. |
| `test/python/test_transfer.py` | `TransferRequest` and `TransferMode` shaped request validation | Replace or extend with `TransferIntent` validation. |
| `test/python/test_schema.py` | Existing daemon, buffer, lease, and worker request shapes | Add schema tests for `BufferHandle`, `TransferIntent`, `TopologySnapshot`, `SchedulingDecision`, `ExecutionTicket`, and `TransferReceipt`. |
| `test/python/test_daemon_state.py` | Daemon sessions, static relay configuration, profiles, reservations, leases, worker authorization | Split into control-plane integration tests and scheduler/ticket unit tests. Synthetic topology must be explicit fixture data. |
| `test/python/test_daemon_socket.py` | Socket protocol using relay-shaped session and profile calls | Rewrite around intent/decision/ticket protocol. |
| `test/python/test_daemon_scheduler.py` | Scheduler direct/relay/pool behavior through mode inputs | Keep scheduler coverage but assert `SchedulingDecision` output and explainability. |
| `test/python/test_client_worker_transfer.py` | Worker-managed transfer client with explicit target and relay parameters | Rewrite around client intent submission and daemon-issued ticket execution. |
| `test/python/test_worker_helper.py` | Worker request, authorization, completion, and cleanup lifecycle | Rewrite worker entry to require `ExecutionTicket`. |
| `test/python/test_worker_cuda_executor.py` | CUDA worker executor plan validation and execution metadata | Keep as data-plane unit tests with ticket-bound exact plans. |
| `test/python/test_backend_cuda.py` | CUDA backend transfer mode and exact-plan facade | Keep exact-plan backend tests; avoid public route selection assumptions. |
| `test/python/test_planner_engine.py`, `test/python/test_planner_types.py` | Low-level planner types and mode-shaped planning | Keep only as scheduler-internal tests after wrapping decisions. |
| `test/python/test_offload_store.py`, `test/python/test_inference_adapters.py`, `test/python/test_model_loading.py`, `test/python/test_training_offload.py` | Adapter behavior against `FakeRuntime` | Rewrite adapters to assert intent construction and receipt handling. |
| `test/python/test_vllm_kv_connector.py`, `test/python/test_vllm_connector.py`, `test/python/test_vllm_integration.py` | vLLM connector integration and event logging with runtime creation or config path policy | Convert to vLLM intent/receipt contract tests. |
| `test/python/test_paper_validation.py`, `test/python/test_example_config.py`, `test/python/test_vllm_kv_connector_sweep.py` | Benchmark/example command construction with relay-shaped arguments | Rewrite once benchmark/example public API is daemon-first. |

## Examples And Benchmarks Requiring Realignment

| Current files | Current behavior | Phase 0 target |
| --- | --- | --- |
| `examples/torch_tensor_fetch.py` | Constructs runtime and calls direct tensor fetch | Replace with minimal daemon-first client example. |
| `examples/vllm_turbobus_connector.py` | Builds runtime and configures relay-related connector options | Convert to vLLM connector using public intent API. |
| `examples/vllm_turbobus_kv_connector.py` | Passes connector config and emits path/mode events | Convert event output to decisions and receipts. |
| `examples/vllm_turbobus_kv_connector_sweep.py` | Sweeps explicit relay/mode style inputs | Convert to workload/policy sweep. |
| `examples/vllm_turbobus_restore.py` | Constructs runtime with physical/runtime relay mapping | Convert restore workflow to daemon-first path. |
| `benchmarks/bandwidth_pool.py` | Benchmarks path modes through runtime calls | Convert to policy-based daemon decision benchmark. |
| `benchmarks/model_loading.py` | Loads model buckets through runtime mode and relays | Convert to model-loading workload intent benchmark. |
| `benchmarks/training_offload.py` | Prefetch/offload benchmark through runtime mode and relays | Convert to training workload intent benchmark. |
| `benchmarks/kv_offload.py` | KV offload benchmark through runtime mode and relays | Convert to KV workload intent benchmark. |
| `benchmarks/tune_transfer.py` | Tunes chunk/staging against runtime and relay inputs | Reframe as scheduler/backend tuning using daemon decisions. |
| `benchmarks/paper_validation.py` | Builds command lines with relay-shaped benchmark inputs | Rebuild around public API experiments and receipt output. |

## Target Test Layout

Phase 0 should migrate tests toward:

```text
test/python/
  unit/
    test_contract_schema.py
    test_topology_snapshot.py
    test_scheduling_decision.py
    test_execution_ticket.py
    test_transfer_receipt.py
  integration/
    test_client_daemon_api.py
    test_daemon_scheduler_contract.py
    test_worker_ticket_execution.py
  e2e/
    test_vllm_kv_intent_flow.py
    test_model_loading_intent_flow.py
    test_training_offload_intent_flow.py
  fixtures/
    topology.py
    daemon.py
    backend.py
```

GPU-required tests should be clearly marked and should exercise exact
daemon-issued plans or ticketed worker execution.

## Cut 2 Inputs

The next Cut should add the shared schema layer and unit tests for:

- `BufferHandle`
- `TransferIntent`
- `TopologySnapshot`
- `SchedulingDecision`
- `ExecutionTicket`
- `TransferReceipt`

Existing `JobIdentity` can be reused or reshaped if it satisfies the new
contract. If it does not, update it in the same schema layer rather than
creating a parallel identity type.
