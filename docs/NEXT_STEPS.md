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
- Add the first CUDA worker executor for the daemon-approved H2D relay path.
  The worker/helper default process now binds shared CPU and CUDA IPC target
  handles, creates a worker-local native CUDA runtime, allocates relay staging
  inside that worker runtime, executes the authorized relay chunks, waits for
  completion, and reports daemon-owned completion metadata.
- Connect client, daemon, and worker into one functional call. The new
  worker-managed client path registers a job and shared CPU/CUDA IPC buffers,
  requests a daemon plan and relay lease, submits the worker authorization
  request, lets the worker report completion, releases the completed relay
  reservation, and returns the daemon-owned final transfer status.
- Carry the same worker-managed call across the helper-process request
  boundary. `WorkerManagedTransferClient` can now submit worker authorization
  through a completion-only worker service envelope, and
  `WorkerServiceSocketClient` can send that envelope to the Unix socket helper
  path without requiring an in-process lifecycle record.
- Add a CUDA-server verification entry point for the worker-managed H2D relay
  path. `python -m turbobus.verification` starts a daemon socket, starts a
  worker helper socket process, allocates shared CPU source memory and a CUDA
  IPC target tensor, runs the relay transfer, checks target bytes, and asserts
  that the daemon released relay reservations.
- Make worker-managed H2D relay execution consume the exact daemon-issued
  relay chunks. The worker authorization ranges now come from the daemon plan
  assignment for the leased relay instead of the original client request range.
  Plans that include direct chunks or another relay are rejected for this
  narrow worker path and their relay reservation is released.
- Anchor worker authorization to the daemon-stored transfer plan. The daemon
  now keeps the exact plan for each planned transfer, derives worker relay
  ranges from that plan during authorization, returns the plan with the worker
  context, and the CUDA worker executor refuses to execute without a
  daemon-issued plan.
- Extend the worker-managed H2D path to daemon-issued pool plans. The client
  accepts a single-relay `direct + relay` daemon plan, the daemon still derives
  only relay authorization ranges from its stored plan, and the CUDA worker
  executor submits the complete daemon plan so direct and relay chunks run in
  one native exact-plan transfer.
- Add the first D2H worker executor path. Worker resource binding now treats
  the CUDA IPC handle as the source and the shared pinned CPU handle as the
  destination for D2H requests, and the CUDA worker executor submits
  daemon-issued D2H plans through the native exact-plan offload entry point.
- Add a client-facing D2H worker-managed call. The client can now register a
  CUDA IPC GPU source and shared pinned CPU destination, request a daemon
  `d2h` plan and relay lease, submit the worker authorization, wait for worker
  completion, release the relay reservation, and return the daemon-owned final
  status.
- Extend the CUDA-server helper-socket verifier to D2H. `python -m
  turbobus.verification --direction d2h` now allocates a real CUDA IPC source,
  offloads it into a shared pinned CPU destination through the daemon-approved
  worker/helper path, checks destination bytes, and asserts reservation release.
- Protect worker-owned relay staging memory in the native CUDA data path. Relay
  staging slots are zeroed after H2D and D2H relay use, initialized clear when
  allocated, and cleared again before release so reused relay buffers do not
  retain another transfer's bytes.
- Extend the CUDA-server helper-socket verifier to daemon-issued pool plans.
  `python -m turbobus.verification --mode pool` now requires a multi-chunk
  transfer, runs the worker-managed call with a daemon `direct + relay` plan,
  and checks that worker completion reports both direct and relay bytes.
- Release daemon relay reservations when worker-managed execution fails before
  normal completion. Worker executor exceptions are now reported as daemon
  `failed` transfer status with staging release and reservation cleanup, and
  client-side worker boundary exceptions perform best-effort reservation
  cleanup before surfacing the original error.
- Require a daemon-issued exact plan before worker staging allocation. Worker
  authorization now rejects responses without a matching relay plan before the
  worker touches relay staging resources, and the failure path cleans up the
  daemon reservation.
- Reject worker-authorized ranges that exceed registered buffer sizes before
  staging allocation or resource binding, so a bad daemon/helper request cannot
  reach native CUDA copy with out-of-bounds source or destination offsets.

1. Verify the worker-managed H2D relay path on a CUDA server.
   - Rebuild the native extension with CUDA.
   - Run `python -m turbobus.verification --direction h2d --target-gpu 0 --relay-gpu 1`.
   - If it fails, fix the failing real data-path layer first: shared CPU
     binding, CUDA IPC target opening, relay runtime execution, daemon status,
     or reservation release.
   - Treat the current task as complete only after bytes land on the target GPU
     and the daemon releases the relay reservation on the CUDA server.

2. Add cleanup and isolation only where the real path needs it.
   - Validate lease tokens before touching relay resources; done for daemon
     worker authorization and exact-plan presence before staging allocation.
   - Validate authorized ranges against registered source and destination
     buffer sizes before worker staging allocation.
   - Clear or protect reused relay staging buffers; done for the native CUDA
     relay staging slots, pending CUDA-server verification.
   - Release reservations on failure or completion; done for worker executor
     exceptions and client-side worker boundary exceptions, pending CUDA-server
     verification against helper-process failures.

3. Extend the worker executor only after the functional call works.
   - Add D2H through the same resource binding path; done for worker resource
     binding, CUDA executor exact-plan submission, and the client-facing
     worker-managed call, pending CUDA-server byte verification through
     `python -m turbobus.verification --direction d2h`.
   - Add pooled direct-plus-relay execution through daemon-issued chunks; done
     for the narrow single-relay worker-managed path, pending CUDA-server byte
     verification through `python -m turbobus.verification --mode pool`.
   - Keep the executor limited to daemon-authorized plans and handles.

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
