# Phase 7 Evaluation Matrix

Phase 7 proves paper-parity behavior with full-system experiments. The
commands in this file are server-run commands, not new application APIs. They
must be run from the repository root against a running TurboBus daemon and must
keep physical path selection inside daemon scheduling.

## Preconditions

- Build the native extension on the CUDA server before running data-plane
  correctness or paper-validation commands.
- Run on servers with 2, 4, and 8 visible GPUs when those machines are
  available.
- Start the daemon with the production CUDA/NVML topology provider. Do not use
  synthetic topology outside explicit tests.
- Register the CPU and GPU buffers used by each workload before invoking
  paper-validation commands. The commands below name registered buffer ids,
  not device routes.
- Install vLLM for vLLM KV runs and choose a model compatible with that server.

## Daemon Startup

Use the production daemon on each server. Pick the target GPU for the daemon
policy, but do not pass target or relay GPU controls to workload commands.

```bash
python -m turbobus.daemon \
  --target-gpu 0 \
  --min-relays 1 \
  --max-sessions-per-relay 1 \
  --max-inflight-chunks-per-relay 8 \
  --profile-max-age-seconds 45 \
  --socket-path /tmp/turbobusd.sock
```

If a server cannot report PCIe or fabric data, record that as an environment
failure. Do not hide it with permissive startup flags unless the run is a
separate diagnostic, not a paper evaluation result.

## Evaluation Axes

Run the following matrix for every available server size:

| Axis | Values |
| --- | --- |
| GPU count | 2 GPU, 4 GPU, 8 GPU |
| Workload | `vllm-kv`, `model-loading`, `training-offload`, `optimizer-offload`, `all` |
| Job shape | single job, 2 concurrent vLLM KV jobs |
| Policy label | `paper-baseline`, `turbobus-daemon` |
| Transfer size | 32 MiB buckets for smoke, larger buckets for paper runs |
| Output schema | `phase6_unified_v1` |

The policy label is experiment metadata consumed by the benchmark output. It
must not encode direct, relay, pooled, target GPU, relay GPU, or route choices.
The actual path split must come from daemon decisions and transfer receipts.

## Single-Job Commands

Run the baseline label first, then the daemon-scheduled TurboBus label. These
commands cover vLLM KV, model loading, training-state offload, and
optimizer-state offload through one shared paper-validation report shape.

```bash
python benchmarks/paper_validation.py \
  --workloads all \
  --session-id phase7-2gpu-baseline \
  --job-id phase7-2gpu-baseline \
  --cpu-buffer-id phase7-2gpu-baseline-cpu \
  --gpu-buffer-id phase7-2gpu-baseline-gpu \
  --daemon-socket-path /tmp/turbobusd.sock \
  --policy paper-baseline \
  --run-id phase7-2gpu-baseline \
  --bucket-count 8 \
  --bucket-bytes 33554432 \
  --chunk-bytes 4194304 \
  --warmup 1 \
  --iterations 5 \
  --vllm-model <vllm-compatible-model> \
  --vllm-restore-blocks 8 \
  --vllm-matched-tokens 128 \
  --vllm-prompt-repeat 64 \
  --vllm-enforce-eager \
  --output-dir benchmarks/results/phase7/2gpu/paper-baseline \
  --json-output benchmarks/results/phase7/2gpu/paper-baseline/result.json \
  --summary-output benchmarks/results/phase7/2gpu/paper-baseline/summary.txt
```

```bash
python benchmarks/paper_validation.py \
  --workloads all \
  --session-id phase7-2gpu-turbobus \
  --job-id phase7-2gpu-turbobus \
  --cpu-buffer-id phase7-2gpu-turbobus-cpu \
  --gpu-buffer-id phase7-2gpu-turbobus-gpu \
  --daemon-socket-path /tmp/turbobusd.sock \
  --policy turbobus-daemon \
  --run-id phase7-2gpu-turbobus \
  --bucket-count 8 \
  --bucket-bytes 33554432 \
  --chunk-bytes 4194304 \
  --warmup 1 \
  --iterations 5 \
  --vllm-model <vllm-compatible-model> \
  --vllm-restore-blocks 8 \
  --vllm-matched-tokens 128 \
  --vllm-prompt-repeat 64 \
  --vllm-enforce-eager \
  --output-dir benchmarks/results/phase7/2gpu/turbobus-daemon \
  --json-output benchmarks/results/phase7/2gpu/turbobus-daemon/result.json \
  --summary-output benchmarks/results/phase7/2gpu/turbobus-daemon/summary.txt
```

Repeat the same command shape for `4gpu` and `8gpu` by changing the
`phase7-2gpu-*` ids and output directories to `phase7-4gpu-*` or
`phase7-8gpu-*`.

## Multi-Job Fairness Command

Use this command to validate concurrent vLLM KV trace output and fairness
evidence. It creates distinct job, session, CPU buffer, GPU buffer, prefix, and
log identities for each vLLM job while still using the public workload API.

```bash
python benchmarks/paper_validation.py \
  --workloads vllm-kv \
  --session-id phase7-2gpu-vllm-multijob \
  --job-id phase7-2gpu-vllm-multijob \
  --cpu-buffer-id phase7-2gpu-vllm-multijob-cpu \
  --gpu-buffer-id phase7-2gpu-vllm-multijob-gpu \
  --daemon-socket-path /tmp/turbobusd.sock \
  --policy turbobus-daemon \
  --run-id phase7-2gpu-vllm-multijob \
  --chunk-bytes 4194304 \
  --vllm-model <vllm-compatible-model> \
  --vllm-job-count 2 \
  --vllm-restore-blocks 8 \
  --vllm-matched-tokens 128 \
  --vllm-prompt-repeat 64 \
  --vllm-enforce-eager \
  --output-dir benchmarks/results/phase7/2gpu/vllm-kv-multijob \
  --json-output benchmarks/results/phase7/2gpu/vllm-kv-multijob/result.json \
  --summary-output benchmarks/results/phase7/2gpu/vllm-kv-multijob/summary.txt
```

Repeat on 4 GPU and 8 GPU servers with the same identity and output-directory
renaming pattern.

## Correctness Gate

Before treating paper-validation results as valid, run the existing
daemon-ticketed data-plane verifier on the same server class:

```bash
python -m turbobus.verification --direction h2d --mode direct --target-gpu 0 --relay-gpu 1 --bytes 33554432 --chunk-bytes 8388608
python -m turbobus.verification --direction h2d --mode relay --target-gpu 0 --relay-gpu 1 --bytes 33554432 --chunk-bytes 8388608
python -m turbobus.verification --direction h2d --mode pool --target-gpu 0 --relay-gpu 1 --bytes 33554432 --chunk-bytes 8388608
python -m turbobus.verification --direction d2h --mode direct --target-gpu 0 --relay-gpu 1 --bytes 33554432 --chunk-bytes 8388608
python -m turbobus.verification --direction d2h --mode relay --target-gpu 0 --relay-gpu 1 --bytes 33554432 --chunk-bytes 8388608
python -m turbobus.verification --direction d2h --mode pool --target-gpu 0 --relay-gpu 1 --bytes 33554432 --chunk-bytes 8388608
```

These verifier options are allowed here because the verifier is a
daemon-ticketed data-plane correctness gate, not an application workload API.

## Required Evidence

Every accepted paper-validation result must include:

- `report_schema=phase6_unified_v1`;
- workload kind for each metric: `kv_cache`, `model_weights`,
  `training_state`, or `optimizer_state`;
- job id, session id, CPU buffer id, and GPU buffer id;
- receipt ids, decision ids, topology snapshot ids, and ticket ids;
- transfer bytes, bytes completed, direct bytes, relay bytes, direct chunks,
  relay chunks, transfer time, performance time, and fallback reason;
- `correctness_status=complete`;
- vLLM multi-job runs with one metric per job and distinct job/session/buffer
  identities.

## Result Checker

After each paper-validation run, validate the result JSON or compact summary
with the Phase 7 checker:

```bash
python benchmarks/phase7_result_check.py \
  benchmarks/results/phase7/2gpu/turbobus-daemon/result.json \
  --json-output benchmarks/results/phase7/2gpu/turbobus-daemon/check.json
```

The checker reads existing paper-validation output and reports machine-readable
errors for missing receipt ids, decision ids, topology snapshot ids, ticket
ids, path split, byte-completion mismatches, fallback or failure state, and
multi-job identity problems. It does not create transfer plans and does not
touch scheduler or data-plane modules.

## Comparison Summary

After both the baseline-label run and the `turbobus-daemon` run pass the
Phase 7 checker, compare their result JSON files:

```bash
python benchmarks/phase7_compare.py \
  --baseline benchmarks/results/phase7/2gpu/paper-baseline/result.json \
  --turbobus benchmarks/results/phase7/2gpu/turbobus-daemon/result.json \
  --json-output benchmarks/results/phase7/2gpu/comparison.json
```

Repeat the same comparison for the 4 GPU and 8 GPU output directories. The
comparison reads existing paper-validation output, reruns the Phase 7 checker
for both inputs, and reports transfer time, throughput, bytes moved,
direct/relay path split, fallback reason, trace ids, and p50/p99 fields when
the workload result contains repeated-run samples. It does not select direct,
relay, or pooled paths.

## Daemon Evidence

Attach daemon-side profile, audit, resource-state, relay-impact, and
interference evidence to every accepted result:

```bash
python benchmarks/phase7_evidence.py \
  --result benchmarks/results/phase7/2gpu/turbobus-daemon/result.json \
  --comparison benchmarks/results/phase7/2gpu/comparison.json \
  --daemon-socket-path /tmp/turbobusd.sock \
  --json-output benchmarks/results/phase7/2gpu/turbobus-daemon/evidence.json
```

If the daemon profile was captured separately, pass it with `--profile` instead
of `--daemon-socket-path`. The evidence report reruns the Phase 7 result
checker, reads daemon `PROFILE` state, and connects workload metrics to runtime
transfer records, audit records, relay quotas, active resource usage, staging
records, job runtime state, fallback reasons, and failure reasons. It does not
create transfer plans and does not select physical paths.

## Bundle Gate

After the checker, comparison, evidence, and correctness artifacts are
available for one server class, run the bundle gate:

```bash
python benchmarks/phase7_bundle_gate.py \
  --server-class 2gpu \
  --baseline-result benchmarks/results/phase7/2gpu/paper-baseline/result.json \
  --turbobus-result benchmarks/results/phase7/2gpu/turbobus-daemon/result.json \
  --baseline-check benchmarks/results/phase7/2gpu/paper-baseline/check.json \
  --turbobus-check benchmarks/results/phase7/2gpu/turbobus-daemon/check.json \
  --comparison benchmarks/results/phase7/2gpu/comparison.json \
  --evidence benchmarks/results/phase7/2gpu/turbobus-daemon/evidence.json \
  --correctness benchmarks/results/phase7/2gpu/correctness.json \
  --json-output benchmarks/results/phase7/2gpu/bundle-gate.json
```

The `--correctness` artifact is optional when the server correctness commands
have not been converted into JSON yet; omitted correctness is reported as a
warning, not silently accepted. Repeat the same command shape for 4 GPU and
8 GPU runs by changing paths and `--server-class`.

## Server Run Chain

After daemon startup and buffer registration, a server operator can run the
full artifact chain for one server class with:

```bash
python benchmarks/phase7_server_run.py \
  --server-class 2gpu \
  --daemon-socket-path /tmp/turbobusd.sock \
  --vllm-model <vllm-compatible-model> \
  --vllm-enforce-eager
```

The command runs baseline-label paper validation, `turbobus-daemon` paper
validation, result checks, comparison, daemon evidence, bundle gate, and
acceptance manifest ingestion in order. Use `--dry-run --plan-output
benchmarks/results/phase7/2gpu/run-plan.json` to inspect the command plan
without running workloads.

Repeat the same command shape for 4 GPU and 8 GPU systems by changing
`--server-class`. The generated workload commands use registered buffer ids,
policy labels, and the daemon socket; they do not include workload-side target
GPU, relay GPU, direct, relay, pooled, or mode controls.

While collecting one server class before the other required classes are
recorded, pass `--allow-incomplete-inventory`. Omit that flag for the final
Phase 7 acceptance run so missing server classes still fail the inventory.

## Artifact Ingestion

Use the ingestion helper to update the acceptance manifest from each real
server bundle. Accepted entries must be marked as real server artifacts:

```bash
python benchmarks/phase7_ingest_artifacts.py \
  --manifest benchmarks/results/phase7/acceptance-manifest.json \
  --server-class 2gpu \
  --status accepted \
  --bundle-gate benchmarks/results/phase7/2gpu/bundle-gate.json \
  --real-artifacts \
  --inventory-output benchmarks/results/phase7/acceptance-inventory.json
```

For an unavailable server class or missing prerequisite, record the explicit
gap and the command needed to close it:

```bash
python benchmarks/phase7_ingest_artifacts.py \
  --manifest benchmarks/results/phase7/acceptance-manifest.json \
  --server-class 4gpu \
  --status blocked \
  --block-reason "4 GPU CUDA server is not currently available" \
  --environment-gap hardware_unavailable \
  --next-command "run the Phase 7 matrix, bundle gate, and artifact ingestion on a 4 GPU CUDA server" \
  --allow-incomplete-inventory
```

The ingestion helper writes the manifest and reruns the acceptance inventory
over existing artifacts. It does not run workloads, create transfer plans,
issue execution tickets, or select physical paths.

## Acceptance Inventory

After each server class has either an accepted bundle gate or a recorded
hardware/environment gap, write an acceptance manifest and build the final
Phase 7 inventory:

```json
{
  "server_classes": [
    {
      "server_class": "2gpu",
      "status": "accepted",
      "real_artifacts": true,
      "bundle_gate": "2gpu/bundle-gate.json"
    },
    {
      "server_class": "4gpu",
      "status": "blocked",
      "block_reason": "4 GPU CUDA server is not available",
      "environment_gaps": ["hardware_unavailable"],
      "next_commands": [
        "run paper_validation, phase7_result_check, phase7_compare, phase7_evidence, and phase7_bundle_gate on a 4 GPU CUDA server"
      ]
    },
    {
      "server_class": "8gpu",
      "status": "blocked",
      "block_reason": "8 GPU server is missing native CUDA/vLLM prerequisites",
      "environment_gaps": ["native_cuda_or_vllm_missing"],
      "next_commands": [
        "install native CUDA extension and vLLM, then run the 8 GPU Phase 7 bundle gate sequence"
      ]
    }
  ]
}
```

Store the manifest beside the server artifacts, for example
`benchmarks/results/phase7/acceptance-manifest.json`, then run:

```bash
python benchmarks/phase7_acceptance_inventory.py \
  --manifest benchmarks/results/phase7/acceptance-manifest.json \
  --json-output benchmarks/results/phase7/acceptance-inventory.json
```

The inventory reads existing result-check, comparison, daemon-evidence,
bundle-gate, and correctness artifacts only. It fails if no accepted
real-server bundle contains the vLLM KV workload, or if any required server
class is missing without an explicit hardware/environment gap and next
commands.

## Pass/Fail Criteria

A run passes only when:

- every selected workload exits with status `ok`;
- every workload has at least one `paper_metric` line;
- no validation errors are present in the summary;
- `benchmarks/phase7_result_check.py` returns exit code 0;
- baseline and `turbobus-daemon` pairs compare successfully with
  `benchmarks/phase7_compare.py`;
- daemon-side profile evidence is attached successfully with
  `benchmarks/phase7_evidence.py`;
- the server-class artifact set passes `benchmarks/phase7_bundle_gate.py`;
- `benchmarks/phase7_acceptance_inventory.py` reports at least one accepted
  real-server vLLM KV bundle and explicit gaps for missing server classes;
- `bytes_completed` equals `transfer_bytes` for every metric;
- every metric is traceable from workload request to receipt, scheduler
  decision, topology snapshot, execution ticket, and path split;
- the commands do not include workload-side `--target-gpu`, `--relay-gpu`,
  `--relay-gpus`, `--mode`, or `--modes` arguments.

A run fails when daemon startup cannot satisfy topology policy, a workload
cannot find a registered buffer, a vLLM lifecycle event is missing, a receipt
trace field is absent, or correctness status is not complete.
