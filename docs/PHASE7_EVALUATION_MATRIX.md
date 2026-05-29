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

## Pass/Fail Criteria

A run passes only when:

- every selected workload exits with status `ok`;
- every workload has at least one `paper_metric` line;
- no validation errors are present in the summary;
- `benchmarks/phase7_result_check.py` returns exit code 0;
- baseline and `turbobus-daemon` pairs compare successfully with
  `benchmarks/phase7_compare.py`;
- `bytes_completed` equals `transfer_bytes` for every metric;
- every metric is traceable from workload request to receipt, scheduler
  decision, topology snapshot, execution ticket, and path split;
- the commands do not include workload-side `--target-gpu`, `--relay-gpu`,
  `--relay-gpus`, `--mode`, or `--modes` arguments.

A run fails when daemon startup cannot satisfy topology policy, a workload
cannot find a registered buffer, a vLLM lifecycle event is missing, a receipt
trace field is absent, or correctness status is not complete.
