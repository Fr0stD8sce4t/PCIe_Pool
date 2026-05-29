# Phase 6 Workload Boundary Inventory

Phase 6 extends the daemon-first workload path from vLLM KV cache movement to
model weight loading and training or optimizer state offload.

## Scope

This inventory covers the current model-loading and training-offload code
paths that participate in Phase 6:

- `benchmarks/model_loading.py`
- `benchmarks/training_offload.py`
- `benchmarks/paper_validation.py`
- `benchmarks/daemon_support.py`
- `turbobus/adapters/model_loading.py`
- `turbobus/adapters/training_offload.py`
- `turbobus/offload_store.py`
- scheduler workload-kind handling in `turbobus/scheduler/daemon.py`
- related e2e and unit tests under `test/python/`

## Current Daemon-First Boundaries

Model weight loading:

- `benchmarks/model_loading.py` submits `TransferIntent` through
  `TurboBusClient`.
- The workload kind is `model_weights`.
- The transfer direction is H2D.
- Registered buffers are named by `source_buffer_id` and
  `destination_buffer_id`.
- Physical path selection is not exposed in the CLI, intent metadata, or
  policy hints.
- Output records receipt id, decision id, topology snapshot id, ticket id,
  bytes, direct/relay path split, fallback reason, and timing.

Training and optimizer state offload:

- `benchmarks/training_offload.py` submits paired `TransferIntent` objects
  through `TurboBusClient`.
- The workload kind is either `training_state` or `optimizer_state`.
- Prefetch uses H2D from CPU buffer to GPU buffer.
- Offload uses D2H from GPU buffer to CPU buffer.
- Physical path selection is not exposed in the CLI, intent metadata, or
  policy hints.
- Output records prefetch and offload receipt ids, decision ids, topology
  snapshot ids, ticket ids, bytes, direct/relay path split, fallback reason,
  timing, and compute-delay timing.

Shared adapter path:

- `ModelWeightLoader` and `TrainingOffloadManager` both use
  `AdapterTransferContext` and `OffloadStore`.
- `OffloadStore` constructs `TransferIntent` from block ranges and registered
  CPU/GPU buffer ids.
- `AdapterTransferContext` rejects physical path hints such as mode, path,
  relay GPU, and target GPU.
- Adapters consume `TransferReceipt` through `ReceiptTransferHandle`.

Scheduler policy input:

- `TransferIntent.workload_kind` reaches scheduler runtime metadata.
- The scheduler distinguishes `kv_cache`, `model_weights`, `training_state`,
  and `optimizer_state`.
- Training and optimizer state carry a higher fairness charge multiplier than
  generic/model-weight transfers.

## Completed In Cut 1

- Confirmed model-loading and training-offload benchmarks already submit
  daemon-first `TransferIntent` objects through the public client API.
- Confirmed adapters use `TransferIntent` and `TransferReceipt` through
  shared `OffloadStore` instead of choosing direct, relay, or pooled paths.
- Confirmed tests reject old benchmark CLI options such as target GPU, relay
  GPU, and mode.
- Added explicit `workload_kind=model_weights` to model-loading benchmark
  config output.
- Made paper validation surface model-loading and training-offload
  job/session/buffer identity and workload kind in unified `paper_metric`
  lines.
- Made paper validation reject missing Phase 6 workload identity or workload
  kind fields for model-loading and training-offload.

## Completed In Cut 2

- Added `optimizer-offload` as a first-class paper-validation workload that
  invokes `benchmarks/training_offload.py` through the public client API with
  fixed `workload_kind=optimizer_state`.
- Fixed `training-offload` paper validation to represent
  `workload_kind=training_state` instead of letting optimizer state appear as a
  training-offload alias.
- Added distinct intent prefixes and output files for training-state and
  optimizer-state validation runs.
- Added focused scheduler coverage showing `model_weights`, `training_state`,
  and `optimizer_state` reach policy metadata and request charge accounting.
- Added benchmark and paper-validation tests showing optimizer-state intent,
  config, metrics, and summary output are preserved without target GPU, relay
  GPU, mode, or pool controls.

## Remaining Phase 6 Cuts

Cut 3: unified correctness and performance report.

- Add a shared paper-validation report shape across vLLM KV, model loading,
  training state, and optimizer state.
- Require receipt ids, decision ids, topology snapshot ids, ticket ids, bytes,
  timing, path split, fallback reason, workload kind, job/session identity, and
  registered buffer identity for every workload.
- Keep paper validation as a consumer of public benchmark outputs, not a core
  scheduling API.

Cut 4: Phase 6 server validation gate.

- Document and test the CUDA-server command set for model loading,
  training-state offload, and optimizer-state offload.
- The commands must run through the public client API and a running TurboBus
  daemon.
- The validation output must be auditable from workload request to daemon
  receipt.
