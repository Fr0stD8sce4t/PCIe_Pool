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
  control-plane smoke helper and the old daemon benchmark smoke wrapper;
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
- Reject worker data-plane requests whose buffer handles do not match the
  transfer direction before staging allocation: H2D must bind shared pinned CPU
  to CUDA IPC device memory, and D2H must bind CUDA IPC device memory to shared
  pinned CPU.
- Reject worker `complete` results whose byte count does not match the
  daemon-issued plan before reporting completion or releasing the relay
  reservation, so partial CUDA copies cannot be masked as successful transfers.
- Reject daemon-issued worker plans whose direct or relay chunks exceed the
  registered source or destination buffer sizes before staging allocation, so
  pooled direct-plus-relay plans cannot pass an out-of-bounds direct chunk into
  native CUDA execution.
- Reject daemon transfer-status updates that try to mark a transfer complete
  before the full daemon-planned byte count has been reported, so an incomplete
  worker/helper copy cannot become a daemon-owned successful completion.
- Require the client-facing worker-managed call to observe daemon-owned
  `complete` status and the expected byte count after the worker/helper
  returns complete; otherwise it cleans the relay reservation and reports an
  error instead of returning a false success.
- Clean daemon relay reservations when worker status reporting fails after
  execution. The helper lifecycle now releases its local staging slot and then
  force-cleans the daemon reservation with `worker_status_report_failed`, even
  if the worker executor had already completed the byte movement.
- Remove the old daemon benchmark smoke wrapper. The deleted helper only
  launched the legacy daemon plus benchmark clients and parsed status lines; it
  did not exercise the daemon-approved worker/helper data path.
- Clean daemon relay reservations when the helper boundary returns a
  non-complete worker completion envelope without raising an exception. The
  client now treats that as a failed worker-managed transfer and best-effort
  cleans the daemon reservation with `worker_completion_not_complete`.
- Clean daemon relay reservations when the client cannot query the daemon-owned
  final transfer status after worker/helper execution. The client now
  best-effort cleans the relay reservation with `daemon_status_query_failed`
  before surfacing the status error.
- Derive worker completion direct/relay byte metadata from the daemon-issued
  exact plan when the native stats object does not expose per-path byte
  fields. Pool verification now keeps the direct-plus-relay split tied to the
  daemon plan instead of defaulting all completed bytes to the relay path.
- Select the CUDA device around CUDA IPC handle export, open, and close.
  Client-side target/source handle export and worker-side device binding now
  switch to the registered buffer's `device_index`, so helper-socket
  verification is not implicitly limited to target GPU 0.
- Preserve client-requested transfer range offsets in daemon-issued plans.
  Worker-managed transfers now pass `TransferRequest.ranges` into the daemon
  planner, so helper execution receives exact source and destination offsets
  instead of replanning every request from offset 0. The daemon socket
  round-trip also preserves those offsets through worker authorization.
- Extend the CUDA-server helper-socket verifier to nonzero transfer offsets.
  `python -m turbobus.verification` now accepts `--src-offset`,
  `--dst-offset`, `--source-buffer-bytes`, and
  `--destination-buffer-bytes`, initializes larger real source/destination
  buffers, submits the offset range through the worker-managed path, and checks
  that bytes land at the daemon-approved destination offset.
- Execute daemon-issued direct fallback plans in the worker-managed client.
  If daemon planning resolves to a direct-only plan without a relay lease, the
  client now runs that exact direct plan through the CUDA backend, reports
  daemon-owned completion, and returns a complete transfer instead of treating
  missing relay lease tokens as an error. The verifier accepts `--mode direct`
  and also handles pool requests that resolve to direct fallback.
- Keep worker-managed backend configuration consistent across direct fallback
  and worker/helper execution. The factory now passes the selected CUDA
  backend and runtime options into the default worker executor and resource
  binder instead of using separate default instances for relay/pool paths.
- Keep direct fallback verification independent from the worker helper socket.
  `python -m turbobus.verification --mode direct` now starts only the daemon
  and the direct data path; relay and pool modes still start the worker helper
  because they need daemon-authorized relay execution.
- Require worker-opened shared pinned CPU handles to carry explicit logical
  backing size metadata. `SharedPinnedCpuBuffer.open_from_registration` now
  rejects daemon-authorized shared CPU handles that omit
  `shared_memory_size_bytes` before the worker can host-register the mapping
  or pass it into CUDA execution.
- Reject shared pinned CPU registrations that omit logical backing size
  metadata before daemon authorization. `BufferRegistration` and
  `WorkerBufferHandle` now require `shared_memory_size_bytes` for
  `shared_pinned_cpu` handles, so malformed shared CPU handles are stopped
  before worker resource binding.
- Reject malformed CUDA IPC device registrations before worker binding.
  `BufferRegistration` and `WorkerBufferHandle` now require
  `cuda_ipc_device` handles to carry a 64-byte hex-encoded CUDA IPC memory
  handle, preventing bad device handles from reaching native CUDA IPC open.
- Validate CUDA IPC handle size at the CUDA backend native boundary.
  `CudaNativeBackend` now rejects malformed exported or opened IPC handles
  unless they are exactly 64 bytes, so direct backend calls cannot bypass the
  daemon/worker metadata checks and pass bad handles into native CUDA IPC.
- Keep borrowed shared pinned CPU handles owned by the client process. Worker
  opens now avoid POSIX resource-tracker ownership for shared-memory backing
  they did not create, while same-process owner opens stay tracked until the
  owner unlinks them.
- Wire that borrowed-open behavior into the actual registration reopen path.
  `SharedPinnedCpuBuffer.open_from_registration`, which is what worker resource
  binding calls before CUDA host registration, now opens through the borrowed
  shared-memory helper.
- Keep daemon-owned transfer status terminal once a worker/helper outcome has
  been recorded. Terminal `complete`, `failed`, and `canceled` states now allow
  only idempotent repeats and reject later conflicting status updates.
- Keep direct fallback verification scoped to the direct CUDA device. Direct
  mode no longer requires the configured relay GPU to be visible, while relay
  and pool verification still require both target and relay GPUs.
- Reclaim session-scoped job and buffer registrations when a session is
  closed. Long-lived daemons no longer keep stale shared CPU or CUDA IPC handle
  metadata after a worker-managed verification or client session ends.
- Reject job registrations that name an unknown daemon session. Buffer handles
  used by worker-managed transfers must now be anchored to an existing session
  before they can be registered and authorized.
- Reject transfer planning with buffer handles owned by jobs outside the
  transfer session, including detached legacy jobs, so worker-managed handles
  are session-anchored before lease issuance.
- Infer daemon plan ownership from registered buffer owners when a transfer
  request omits `job_id`, and carry that job identity into scheduler leases,
  transfer status, and worker authorization instead of falling back to the
  session id.
- Keep daemon-authorized buffer handles stable while a transfer lease is
  active. Buffer registration now rejects overwriting a `buffer_id` that is
  still named by an active lease, so worker authorization cannot open a handle
  that was swapped after planning.
- Require lease validation and worker authorization to match the complete
  daemon-issued source/destination buffer pair. Partial buffer validation and
  swapped source/destination worker requests are rejected before helper
  execution.
- Reject worker authorization after a daemon transfer has reached a terminal
  state. Failed, canceled, or completed transfers can no longer receive a
  helper execution context even if the relay lease has not been cleaned yet.
- Reject lease validation after a daemon transfer has reached a terminal
  state. A failed, canceled, or completed transfer can no longer present its
  still-active lease as valid while reservation cleanup is pending.
- Require daemon-planned transfers to report complete status before normal
  reservation release. `release_transfer` is now the successful completion
  release path; failed, canceled, or incomplete planned transfers must use
  cleanup to reclaim the reservation.
- Report daemon-plan completion from the runtime exact-plan baseline before
  releasing its relay reservation, keeping daemon status ownership aligned
  with worker/helper completion semantics.
- Clean daemon-planned runtime baseline reservations when native wait or
  daemon completion status reporting fails. These failure paths now use daemon
  cleanup instead of normal release, so stricter planned-release semantics do
  not leave relay reservations active.
- Validate worker/helper completion envelopes against the daemon-authorized
  transfer and lease before accepting a worker-managed result. Mismatched
  completion envelopes now clean the relay reservation instead of entering the
  normal completion path.
- Validate the nested worker result and daemon status records inside
  worker/helper completion envelopes before accepting a worker-managed result.
  A helper response whose outer envelope is complete but whose worker result,
  daemon status update, or daemon status response names another transfer now
  cleans the relay reservation instead of entering the normal completion path.
- Validate the completed byte counts inside worker/helper completion
  envelopes against the daemon-requested transfer byte count. A helper
  response whose outer envelope is complete but whose worker result or daemon
  status record reports a partial byte count now cleans the relay reservation
  instead of entering the normal completion path.
- Require complete worker/helper completion envelopes to include both the
  daemon status update and the daemon status response produced by helper
  execution. A helper response that only reports local worker completion is
  rejected and cleans the relay reservation instead of being accepted as an
  end-to-end completed transfer.
- Require complete worker/helper completion envelopes to include a successful
  daemon release response for the relay reservation. A helper response that
  reports copied bytes and daemon status but omits reservation release evidence
  is rejected before the client accepts the transfer as complete.
- Require complete worker/helper completion envelopes to include a matching
  inactive staging release record. A helper response that reports daemon
  completion and reservation release but does not prove the worker-local relay
  staging slot was released is rejected before client completion.
- Bind complete worker/helper completion envelopes to a concrete worker-local
  staging slot lifecycle. The client now requires a matching active
  `staging_slot` record and a matching inactive `staging_release` record for
  that same slot before accepting helper completion.
- Reject daemon-issued exact plans whose declared `total_bytes` does not match
  the sum of assigned chunk bytes before converting them to native CUDA plans,
  so malformed direct, relay, or pooled plans cannot reach native copy
  submission.
- Reject the same `total_bytes` mismatch inside the worker CUDA executor before
  rebuilding a relay-scoped native plan, so helper execution cannot bypass the
  shared exact-plan conversion guard.
- Bind worker resource setup to the daemon-authorized CUDA device before CUDA
  host registration and keep that device selected through host unregister and
  CUDA IPC close, so helper-process verification is not implicitly tied to the
  default CUDA context.

1. Verify the worker-managed H2D relay path on a CUDA server.
   - Rebuild the native extension with CUDA.
   - Run `python -m turbobus.verification --direction h2d --target-gpu 0 --relay-gpu 1`.
   - Also run an offset-range verifier, for example
     `python -m turbobus.verification --direction h2d --target-gpu 0 --relay-gpu 1 --bytes 1048576 --chunk-bytes 262144 --src-offset 4096 --dst-offset 8192`.
   - Verify direct fallback with
     `python -m turbobus.verification --direction h2d --mode direct --target-gpu 0 --relay-gpu 1 --bytes 1048576 --chunk-bytes 262144`.
   - Verify quota-triggered fallback with a pool request whose relay quota is
     too small, for example
     `python -m turbobus.verification --direction h2d --mode pool --target-gpu 0 --relay-gpu 1 --bytes 1048576 --chunk-bytes 262144 --max-inflight-chunks 1`.
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
   - Validate direction-specific worker handle types before worker staging
     allocation.
   - Validate completed byte counts against the daemon-issued plan before
     releasing relay reservations.
   - Validate all daemon-plan chunks against registered buffer sizes before
     worker staging allocation.
   - Reject incomplete daemon transfer-status completion updates.
   - Require client-side final status checks after worker/helper completion.
   - Clean daemon relay reservations when worker status reporting fails after
     helper execution.
   - Clean daemon relay reservations when helper completion envelopes report a
     non-complete final state.
   - Clean daemon relay reservations when the final daemon status query fails
     after worker/helper execution.
   - Require explicit shared pinned CPU backing-size metadata before worker
     resource binding opens a shared CPU handle.
   - Reject missing shared pinned CPU backing-size metadata at buffer
     registration and worker-handle construction time.
   - Reject malformed CUDA IPC handle metadata at buffer registration and
     worker-handle construction time.
   - Reject malformed CUDA IPC handles again at backend export/open time before
     native CUDA IPC calls.
   - Reject lease validation after the daemon-owned transfer status is terminal.
   - Reject normal reservation release for daemon-planned transfers until the
     daemon-owned transfer status is complete.
   - Clean daemon-planned runtime baseline reservations when native wait or
     daemon completion status reporting fails.
   - Reject worker/helper completion envelopes whose transfer id, lease id,
     nested worker result, daemon status update, or daemon status response does
     not match the daemon authorization.
   - Reject worker/helper completion envelopes whose nested worker result,
     daemon status update, or daemon status response reports a completed byte
     count that does not match the daemon-requested transfer.
   - Reject complete worker/helper completion envelopes that do not include
     both the daemon status update and daemon status response from helper
     execution.
   - Reject complete worker/helper completion envelopes that do not include a
     successful daemon release response for the relay reservation.
   - Reject complete worker/helper completion envelopes that do not include a
     matching inactive worker staging release record.
   - Reject complete worker/helper completion envelopes that do not prove the
     active worker staging slot and inactive staging release refer to the same
     slot for the daemon-authorized transfer and lease.
   - Reject daemon-issued exact plans whose declared total byte count does not
     match their assigned chunks before native CUDA execution.
   - Reject worker/helper CUDA executor plans whose daemon-declared total byte
     count does not match the rebuilt direct-plus-relay chunk set before native
     submission.
   - Bind shared pinned CPU host registration, host unregister, CUDA IPC open,
     and CUDA IPC close to the daemon-authorized CUDA device in the worker
     resource lifecycle.
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
