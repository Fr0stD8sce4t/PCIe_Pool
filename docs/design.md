# TurboBus System Design

## Design Goal

Build a daemon-managed PCIe bandwidth pooling system for LLM workloads.

The design should support:

- direct CPU-to-target GPU transfer;
- relay CPU-to-relay-GPU-to-target-GPU transfer;
- chunked and pipelined transfer execution;
- application isolation;
- cross-job relay sharing;
- CUDA and ROCm scale-up fabrics;
- framework adapters for vLLM, model loading, and training offload.

## Core Layers

### Client API

The client API is a thin submission layer.

Responsibilities:

- register pinned CPU buffers and destination GPU buffers;
- submit transfer requests;
- wait for transfer completion;
- fetch transfer stats;
- expose framework-facing adapters.

The client must not decide unauthorized relay access on its own.

### Privileged Daemon

The daemon is the authority for relay sharing.

Responsibilities:

- discover machine topology;
- observe current utilization;
- track jobs, sessions, and users;
- choose relay GPUs;
- issue relay leases;
- enforce quota and isolation;
- reclaim stale resources;
- publish cached profiles.

### Worker Or Helper

The worker/helper performs relay-side data movement when the client should not
directly see relay GPUs.

Responsibilities:

- own relay GPU access;
- manage staging buffers;
- execute daemon-approved plans;
- validate lease tokens;
- clean up staging buffers.

### Backend Layer

Backends implement the actual copy operations.

Required backend capabilities:

- topology discovery;
- peer capability discovery;
- H2D, D2H, and P2P copy;
- staging buffer allocation;
- timing and stats collection;
- handle export and import for safe cross-process use when needed.

## Planner Model

Planner inputs:

- request bytes;
- chunk size;
- direction;
- direct bandwidth estimates;
- relay PCIe estimates;
- relay fabric estimates;
- current utilization;
- relay permissions;
- fallback policy.

Planner outputs:

- direct path chunk ranges;
- relay path chunk ranges;
- lease requirements;
- estimated completion time;
- fallback mode.

The planner must be backend-neutral.

## Data Path

1. Client registers job and buffers.
2. Daemon validates identity and topology.
3. Daemon grants a relay lease or falls back to direct.
4. Client or worker submits the approved plan.
5. Backend executes chunked copy on the selected paths.
6. Client waits for completion and receives stats.

## Isolation Rules

- A job cannot borrow another job's relay GPU without daemon approval.
- Relay staging buffers must not leak data between jobs.
- Lease expiry must trigger cleanup.
- Unauthorized requests must fail cleanly or fall back.

## Implementation Rule

If a feature requires the client to own both target and relay GPUs, it is not a
final-system feature. It may be a temporary development check, but not the
production architecture.
