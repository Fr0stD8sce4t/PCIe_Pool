# TurboBus Next Steps

## Current Direction

Stop extending the control-plane skeleton as the main line of work. The daemon,
worker service boundary, socket shell, request envelopes, and staging-pool
records are enough scaffolding for now. Worker endpoint observability and event
history have been removed and should not be rebuilt before real data movement
needs them.

The next phase is to make TurboBus work as a whole PCIe-pooling system, even if
the first version is narrow. The priority is real data movement through a
daemon-approved worker/helper path.

## Immediate Functional Target

Build the smallest end-to-end system that moves real bytes:

1. Client registers a job and buffers with the daemon.
2. Daemon returns an exact chunk-level transfer plan and relay lease.
3. Worker/helper validates the lease and owns relay staging buffers.
4. Backend executes the daemon-issued plan without local relay replanning.
5. Transfer completion is reported to the daemon.
6. Client waits for completion and can verify bytes landed correctly.

The first cut may be limited to one target GPU, one relay GPU, one H2D transfer,
CUDA only, static topology, and a simple shared pinned CPU buffer scheme. That
is acceptable if it exercises the real `CPU -> relay GPU -> target GPU` path
outside the client-owned relay model.

## Pre-Implementation Cleanup

Before implementing the first real data path, remove scaffolding that no longer
serves the system goal.

Delete or fold down code that only supports the old unsupported path:

- standalone smoke helpers and smoke-only tests; done for the worker
  control-plane smoke helper;
- endpoint observability snapshots, metrics, event history, and reset plumbing;
  done for the worker endpoint, codec, transport, and process path;
- extra socket/transport wrappers that do not carry real transfer execution;
  done for the worker loopback transport and transport protocol wrapper; keep
  the Unix socket helper path because it can carry the real helper-process
  request boundary;
- response envelope fields that only serialize unsupported lifecycle details;
  done for the worker service response envelope and legacy service dict output;
- protocol fields that are not consumed by daemon planning, lease validation,
  worker authorization, execution, completion, or cleanup.

Keep the minimum functional spine:

- job and buffer registration;
- transfer request objects;
- daemon-issued exact plans;
- relay leases and lease validation;
- worker authorization;
- worker staging ownership;
- direct fallback;
- status completion and cleanup.

The cleanup pass should make the next implementation simpler, not rewrite the
whole project again. If a piece will be needed by the real worker executor,
keep it and tighten it around that use.

## Next Code Cuts

Completed current code cut:

- Execute daemon-issued plans exactly in the runtime/backend/native entry
  point. Daemon plan payloads are converted into native `TransferPlan` objects
  and submitted through exact-plan methods instead of local relay replanning.
- Remaining server verification: rebuild the native extension on a CUDA
  machine and run a direct/relay/pool transfer against a daemon-issued plan.
- Define the first registered buffer-handle metadata shape. Buffer registration
  can now carry shared pinned CPU handles and CUDA IPC device handles through
  daemon authorization into worker data-plane requests.
- Add the first TurboBus-owned shared CPU buffer allocator. It creates a
  cross-process shared-memory backing, emits daemon-ready `shared_pinned_cpu`
  registration metadata, can reopen the same backing from that metadata, and
  exposes CUDA host-register/unregister hooks through the CUDA backend.
- Open shared pinned CPU handles inside the worker/helper execution lifecycle.
  A worker data-plane resource binder now reopens the daemon-authorized shared
  CPU source, registers it with CUDA before executor invocation, and closes it
  after execution or binding failure.
- Add the first CUDA IPC target GPU handle producer/consumer path. Client-side
  code can export a target device pointer as daemon registration metadata, and
  worker resources can open and close that target device pointer before
  invoking a bound executor.

1. Define the first registered buffer handles.
   - Pass bound source and target resources into the future CUDA worker
     executor.

2. Implement a CUDA worker executor.
   - Replace the default unsupported executor for one narrow path.
   - Allocate relay staging buffers in the worker/helper process.
   - Use CUDA IPC or the first accepted equivalent for target GPU access.
   - Run H2D relay transfer from shared CPU memory through relay staging to the
     target GPU.

3. Connect client, daemon, and worker into one functional call.
   - Client submits a transfer request to the daemon.
   - Daemon authorizes relay use and returns the plan/lease.
   - Worker executes and reports status.
   - Client waits on daemon-owned completion.

4. Add cleanup and isolation only where the real path needs it.
   - Validate lease tokens before touching relay resources.
   - Clear or protect reused relay staging buffers.
   - Release reservations on failure or completion.

## Defer For Now

Do not make these the next main work items:

- more socket wrappers;
- more endpoint observability;
- standalone smoke tests;
- extra protocol fields not needed by the first real transfer;
- vLLM feature expansion before the daemon/helper data path works;
- ROCm support before CUDA cross-process H2D relay works;
- benchmark polish before the system can move data end to end.

These are still important, but they should follow the first working system
slice.

## After The First Working Slice

Once the daemon/helper H2D relay path moves real bytes:

1. Add D2H support.
2. Add pooled direct-plus-relay execution through the same daemon plan path.
3. Add multiple relay GPUs.
4. Add current-load-aware scheduling using NVML/CUDA topology data.
5. Reconnect vLLM KV prefix save/restore to the daemon/helper data path.
6. Add model loading and training offload workloads.
7. Expand to multi-job fairness and isolation.
8. Add ROCm/Infinity Fabric support.

## Existing Completed Scaffolding

The current code already has useful pieces:

- backend-neutral planner objects;
- direct, relay, and pooled chunk planning;
- daemon session, job, buffer, lease, quota, cleanup, and plan records;
- Python runtime facade and CUDA native backend facade;
- worker request, completion, staging-pool, codec, endpoint, and process
  skeletons;
- framework adapter prototypes for vLLM, inference KV slots, model loading,
  and training offload.

Treat these as support code for the real system path, not as the product by
themselves.
