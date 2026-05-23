# TurboBus Agent Instructions

TurboBus should be treated as a paper-reproduction system project.

The target paper system is:

TurboBus: Pooling PCIe Bandwidth for LLM Workloads via Scale-Up Fabrics.

The system goal is to pool otherwise idle PCIe bandwidth in a multi-GPU server
for large-model memory movement. When a target GPU needs data from CPU memory,
TurboBus should support both:

- direct path: CPU pinned memory -> target GPU;
- relay path: CPU pinned memory -> relay GPU -> target GPU.

The relay path uses the relay GPU's PCIe link for the CPU-to-relay stage, then
uses a scale-up GPU-GPU fabric such as NVLink, NVSwitch, or Infinity Fabric for
the relay-to-target stage.

## Project Direction

The next architecture should be designed around a privileged per-node daemon,
cross-job resource discovery and scheduling, application isolation, real shared
relay PCIe use, scale-up fabric backends, full LLM framework integration, and
multi-tenant evaluation.

Prefer rewriting major components when the existing code assumes a single
process owns both target and relay GPUs.

Do not preserve old module boundaries just because they already exist.

## Current Implementation Priority

The next phase must prioritize whole-system functionality over additional
control-plane polish. Do not spend the next code cuts on standalone protocol
expansion, socket wrappers, observability, smoke tests, or test-only
infrastructure unless they directly unblock real data movement.

The immediate target is an end-to-end daemon-managed transfer:

1. the client registers a job and real buffers;
2. the daemon issues an exact chunk-level transfer plan and relay lease;
3. a worker/helper validates the lease and owns relay staging buffers;
4. the worker/helper performs real direct, relay, or pooled CUDA movement;
5. the daemon records completion and releases relay resources.

The first functional slice may be narrow: one node, one target GPU, one relay
GPU, H2D only, CUDA only, static topology, and a simple shared pinned CPU buffer
scheme. It must still move real bytes through the daemon-approved path.

Before starting that slice, do one cleanup pass that removes code whose only
purpose is to keep building the old unsupported control-plane scaffold:
standalone smoke helpers, endpoint observability plumbing, excess socket
wrappers, and protocol fields that are not needed by daemon-issued real
transfer execution. Keep only the minimum daemon/client/worker interfaces
needed to authorize, execute, complete, and clean up a real transfer.

Do not delete core pieces that the functional path needs: job and buffer
registration, transfer requests, daemon-issued plans, lease validation, worker
authorization, staging ownership, cleanup on failure, and direct fallback.

## New Architecture

Build TurboBus around these layers:

1. Client API.
   - Own user-facing transfer requests.
   - Register CPU pinned buffers and target GPU buffers.
   - Submit transfer requests to the daemon.
   - Wait for transfer completion and expose stats.
   - Do not choose unauthorized relay GPUs locally.

2. Privileged daemon.
   - Own global machine state.
   - Discover GPUs, PCIe topology, NUMA topology, and scale-up fabric links.
   - Track jobs, sessions, users, containers, and relay permissions.
   - Measure and cache path profiles.
   - Observe current PCIe and fabric utilization.
   - Schedule direct and relay paths across jobs.
   - Issue relay leases and enforce quotas.
   - Reclaim resources after failures or timeout.

3. Worker or helper process.
   - Own privileged data movement when relay GPUs are not visible to clients.
   - Hold relay GPU access.
   - Manage relay staging buffers.
   - Use CUDA IPC, HIP IPC, or equivalent handles where required.
   - Execute daemon-approved transfer plans.

4. Fabric backend layer.
   - Provide a common backend interface for CUDA/NVIDIA and ROCm/AMD.
   - CUDA backend should cover PCIe, P2P, NVLink, and NVSwitch through CUDA
     runtime/NVML where available.
   - ROCm backend should cover HIP and Infinity Fabric through ROCm SMI or
     equivalent APIs.
   - Planner code must consume generic path capabilities, not CUDA-specific
     objects.

5. Planner and scheduler.
   - Convert daemon resource state and request metadata into a chunk-level plan.
   - Split work across direct and relay paths.
   - Account for current load, link bandwidth, fabric bandwidth, relay quotas,
     job policy, request size, and fallback rules.

6. LLM framework adapters.
   - Keep framework-specific logic outside the native data path.
   - Support vLLM KV cache prefix save/restore first.
   - Add model weight loading and training state offload adapters.
   - Later targets may include DeepSpeed/FSDP, TensorRT-LLM, or SGLang.

## Required System Capabilities

The reproduction target requires these capabilities:

- privileged per-node daemon;
- cross-job idle PCIe discovery;
- daemon-managed relay leases;
- full-machine transfer scheduling;
- application isolation;
- client operation without direct relay GPU visibility;
- daemon/helper data path using IPC or equivalent safe handles;
- direct, relay, and pooled transfer execution;
- block-level pipelining;
- fine-grained chunk placement;
- concurrent multi-request scheduling;
- NVIDIA CUDA/NVLink/NVSwitch backend;
- AMD ROCm/Infinity Fabric backend;
- vLLM KV cache connector that works through the real framework lifecycle;
- model weight loading workload;
- training offload workload;
- multi-tenant benchmark suite.

## Milestones

M1: Define the new daemon/client/worker protocol.

- Specify job registration, buffer registration, transfer request, transfer
  planning, relay lease, transfer status, and cleanup messages.
- Add tests for protocol validation.

M2: Implement the new planner data model.

- Define generic devices, links, path capabilities, chunks, plans, leases, and
  stats.
- Support direct-only, relay-only, and pooled plans.
- Keep CUDA out of planner types.

M3: Rebuild single-process CUDA execution on the new interfaces.

- Reproduce current direct, relay, and pooled behavior.
- Use this only as a compatibility baseline.

M4: Add daemon-issued plans with client-side execution.

- The daemon chooses relay paths and returns a plan.
- The backend executes the exact daemon-issued chunk plan; local replanning is
  only allowed for explicit direct fallback.
- The client executes only daemon-approved paths during this temporary
  milestone.
- This is an intermediate milestone, not the final isolation model.

M5: Add daemon/helper execution with CUDA IPC.

- Client should not need direct visibility of relay GPUs.
- Worker/helper should own relay staging buffers.
- Define and implement the first cross-process CPU pinned buffer strategy.
- Add CUDA IPC or equivalent device-buffer handle exchange for the target GPU.
- Move real bytes through the worker/helper path before expanding observability
  or transports.
- Add ownership checks and lease-token validation.

M6: Add isolation and policy.

- Track job/session/user/container identity.
- Enforce relay access through leases.
- Prevent unauthorized buffer or relay access.
- Reclaim stale sessions and clear or protect reused staging buffers.

M7: Build full vLLM KV cache integration.

- Save prefixes from real vLLM-owned KV tensors.
- Restore prefixes through the official vLLM connector lifecycle.
- Report save/restore timing, bytes, chunks, path split, and fallback reason.

M8: Add model loading and training offload adapters.

- Model loading should move CPU-backed weight buckets into GPU memory.
- Training offload should move parameter or optimizer buckets both directions.

M9: Add ROCm/Infinity Fabric backend.

- Implement HIP transfer operations.
- Discover AMD peer/fabric capabilities.
- Run equivalent direct, relay, and pooled tests.

M10: Complete multi-tenant evaluation.

- Measure single-job and multi-job behavior.
- Measure direct vs relay vs pool.
- Measure idle-relay benefit and relay contention.
- Include vLLM KV restore latency, model load time, training step time,
  throughput, fairness, and isolation tests.

## Coding Rules

- Prefer simple, testable interfaces over patching around current assumptions.
- Keep daemon control plane, worker data plane, planner, backend, and framework
  adapters separate.
- Do not let benchmark scripts become the system.
- Do not place vLLM request or scheduler policy inside CUDA execution code.
- Do not require application code to control another job's relay GPU in the
  final design.
- Keep direct transfer fallback available whenever relay scheduling or lease
  acquisition fails.
- Add focused tests after functional code paths exist; do not let test or
  observability work displace the real daemon/helper data path.
- For documentation-only changes, `git diff --check` is sufficient.
