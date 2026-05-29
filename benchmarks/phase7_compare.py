from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import phase7_result_check


TRACE_ID_FIELDS = (
    "receipt_ids",
    "decision_ids",
    "topology_snapshot_ids",
    "ticket_ids",
    "prefetch_receipt_ids",
    "prefetch_decision_ids",
    "offload_receipt_ids",
    "offload_decision_ids",
    "save_decision_ids",
    "save_topology_snapshot_ids",
    "save_ticket_ids",
)

IDENTITY_FIELDS = (
    "job_id",
    "session_id",
    "cpu_buffer_id",
    "gpu_buffer_id",
    "source_buffer_id",
    "destination_buffer_id",
)


def as_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, "", "NA"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        if value in (None, "", "NA"):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_number_or_string(value: object) -> float | str:
    try:
        if value in (None, "", "NA"):
            return ""
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (float(percent) / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def numeric_values(samples: list[dict], field: str) -> list[float]:
    values = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        value = sample.get(field)
        if value in (None, "", "NA"):
            continue
        values.append(as_float(value))
    return values


def sample_throughput_values(samples: list[dict]) -> list[float]:
    values = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        if "load_gib_per_second" in sample:
            values.append(as_float(sample.get("load_gib_per_second")))
            continue
        transfer_ms = as_float(sample.get("transfer_ms"))
        prefetch = sample.get("prefetch", {}) or {}
        offload = sample.get("offload", {}) or {}
        byte_count = as_int(prefetch.get("bytes")) + as_int(offload.get("bytes"))
        if byte_count > 0 and transfer_ms > 0.0:
            values.append((byte_count / (1024.0**3)) / (transfer_ms / 1000.0))
    return values


def sample_percentiles(workload_data: object) -> dict[str, float]:
    if not isinstance(workload_data, dict):
        return {}
    samples = workload_data.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        return {}

    fields = {
        "load_ms": "transfer_ms",
        "transfer_ms": "transfer_ms",
        "iteration_ms": "iteration_ms",
    }
    result: dict[str, float] = {}
    for sample_field, output_name in fields.items():
        values = numeric_values(samples, sample_field)
        if len(values) < 2:
            continue
        result[f"{output_name}_p50"] = percentile(values, 50.0)
        result[f"{output_name}_p99"] = percentile(values, 99.0)

    throughput = sample_throughput_values(samples)
    if len(throughput) >= 2:
        result["throughput_gib_s_p50"] = percentile(throughput, 50.0)
        result["throughput_gib_s_p99"] = percentile(throughput, 99.0)
    return result


def metric_percentiles(metric: dict, workload_data: object) -> dict[str, float | str]:
    result: dict[str, float | str] = {}
    for key, value in metric.items():
        lowered = str(key).lower()
        if "p50" not in lowered and "p99" not in lowered:
            continue
        if value in (None, ""):
            continue
        result[str(key)] = as_number_or_string(value)
    result.update(sample_percentiles(workload_data))
    return result


def policy_label(data: dict) -> str:
    config = data.get("config", {}) or {}
    return str(config.get("policy", "") or "")


def metric_key(metric: dict, workload_name: str) -> tuple[str, str]:
    workload = str(metric.get("workload", workload_name))
    job_index = metric.get("job_index")
    if job_index not in (None, ""):
        return workload, f"job_index={job_index}"
    return workload, ""


def key_name(key: tuple[str, str]) -> str:
    workload, suffix = key
    if suffix:
        return f"{workload}[{suffix}]"
    return workload


def metric_index(data: dict, side: str) -> tuple[dict[tuple[str, str], dict], list[str]]:
    index: dict[tuple[str, str], dict] = {}
    errors: list[str] = []
    workloads = data.get("workloads", []) or []
    for workload_result in workloads:
        if not isinstance(workload_result, dict):
            continue
        workload_name = str(workload_result.get("workload", ""))
        workload_data = workload_result.get("data", {}) or {}
        metrics = workload_result.get("metrics", []) or []
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            key = metric_key(metric, workload_name)
            if key in index:
                errors.append(f"{side}:duplicate_metric:{key_name(key)}")
                continue
            index[key] = {
                "metric": metric,
                "workload_data": workload_data,
            }
    return index, errors


def trace_ids(metric: dict) -> dict[str, object]:
    return {
        field: metric[field]
        for field in TRACE_ID_FIELDS
        if metric.get(field) not in (None, "")
    }


def identity_fields(metric: dict) -> dict[str, object]:
    return {
        field: metric[field]
        for field in IDENTITY_FIELDS
        if metric.get(field) not in (None, "")
    }


def metric_summary(record: dict) -> dict[str, object]:
    metric = record["metric"]
    transfer_bytes = as_int(metric.get("transfer_bytes"))
    bytes_completed = as_int(metric.get("bytes_completed"))
    direct_bytes = as_int(metric.get("direct_bytes"))
    relay_bytes = as_int(metric.get("relay_bytes"))
    path_bytes = direct_bytes + relay_bytes
    return {
        "workload": str(metric.get("workload", "")),
        "job_index": metric.get("job_index"),
        "policy": str(metric.get("policy", "")),
        "workload_kind": str(metric.get("workload_kind", "")),
        "identity": identity_fields(metric),
        "trace_ids": trace_ids(metric),
        "timing": {
            "transfer_ms": as_float(metric.get("transfer_ms")),
            "performance_ms": as_float(metric.get("performance_ms")),
            "percentiles": metric_percentiles(metric, record.get("workload_data")),
        },
        "throughput_gib_s": as_float(metric.get("throughput_gib_s")),
        "bytes": {
            "transfer_bytes": transfer_bytes,
            "bytes_completed": bytes_completed,
        },
        "path_split": {
            "direct_bytes": direct_bytes,
            "relay_bytes": relay_bytes,
            "direct_chunks": as_int(metric.get("direct_chunks")),
            "relay_chunks": as_int(metric.get("relay_chunks")),
            "relay_byte_fraction": ratio(float(relay_bytes), float(path_bytes)),
            "source": "daemon_transfer_receipt",
        },
        "fallback_reason": str(metric.get("fallback_reason", "")),
        "correctness_status": str(metric.get("correctness_status", "")),
    }


def compare_metric_pair(key: tuple[str, str], baseline: dict, turbobus: dict) -> dict:
    baseline_metric = metric_summary(baseline)
    turbobus_metric = metric_summary(turbobus)
    baseline_transfer = baseline_metric["timing"]["transfer_ms"]
    turbobus_transfer = turbobus_metric["timing"]["transfer_ms"]
    baseline_performance = baseline_metric["timing"]["performance_ms"]
    turbobus_performance = turbobus_metric["timing"]["performance_ms"]
    baseline_throughput = baseline_metric["throughput_gib_s"]
    turbobus_throughput = turbobus_metric["throughput_gib_s"]
    baseline_path = baseline_metric["path_split"]
    turbobus_path = turbobus_metric["path_split"]
    baseline_bytes = baseline_metric["bytes"]["transfer_bytes"]
    turbobus_bytes = turbobus_metric["bytes"]["transfer_bytes"]
    return {
        "key": key_name(key),
        "workload": key[0],
        "match": {
            "bytes_match": baseline_bytes == turbobus_bytes,
            "workload_kind_match": (
                baseline_metric["workload_kind"] == turbobus_metric["workload_kind"]
            ),
        },
        "baseline": baseline_metric,
        "turbobus": turbobus_metric,
        "comparison": {
            "transfer_ms_delta": turbobus_transfer - baseline_transfer,
            "transfer_ms_speedup": ratio(baseline_transfer, turbobus_transfer),
            "performance_ms_delta": turbobus_performance - baseline_performance,
            "performance_ms_speedup": ratio(baseline_performance, turbobus_performance),
            "throughput_gib_s_delta": turbobus_throughput - baseline_throughput,
            "throughput_gib_s_ratio": ratio(turbobus_throughput, baseline_throughput),
            "direct_bytes_delta": (
                turbobus_path["direct_bytes"] - baseline_path["direct_bytes"]
            ),
            "relay_bytes_delta": (
                turbobus_path["relay_bytes"] - baseline_path["relay_bytes"]
            ),
        },
    }


def compare_result_data(
    baseline_data: dict,
    turbobus_data: dict,
    *,
    baseline_label: str = "paper-baseline",
    turbobus_label: str = "turbobus-daemon",
) -> dict:
    baseline_check = phase7_result_check.check_phase7_result(baseline_data)
    turbobus_check = phase7_result_check.check_phase7_result(turbobus_data)
    errors: list[str] = []
    if not baseline_check["ok"]:
        errors.append("baseline_checker_failed")
    if not turbobus_check["ok"]:
        errors.append("turbobus_checker_failed")

    baseline_policy = policy_label(baseline_data)
    turbobus_policy = policy_label(turbobus_data)
    if baseline_policy and baseline_policy != baseline_label:
        errors.append(f"baseline_policy_mismatch:{baseline_policy}")
    if turbobus_policy and turbobus_policy != turbobus_label:
        errors.append(f"turbobus_policy_mismatch:{turbobus_policy}")

    comparisons = []
    if baseline_check["ok"] and turbobus_check["ok"]:
        baseline_metrics, index_errors = metric_index(baseline_data, "baseline")
        errors.extend(index_errors)
        turbobus_metrics, index_errors = metric_index(turbobus_data, "turbobus")
        errors.extend(index_errors)
        baseline_keys = set(baseline_metrics)
        turbobus_keys = set(turbobus_metrics)
        for key in sorted(baseline_keys - turbobus_keys):
            errors.append(f"missing_turbobus_metric:{key_name(key)}")
        for key in sorted(turbobus_keys - baseline_keys):
            errors.append(f"missing_baseline_metric:{key_name(key)}")
        for key in sorted(baseline_keys & turbobus_keys):
            comparisons.append(
                compare_metric_pair(
                    key,
                    baseline_metrics[key],
                    turbobus_metrics[key],
                )
            )

    return {
        "ok": not errors,
        "errors": errors,
        "baseline_label": baseline_label,
        "turbobus_label": turbobus_label,
        "baseline_policy": baseline_policy,
        "turbobus_policy": turbobus_policy,
        "path_selection_note": (
            "direct, relay, and pooled path split is read from daemon decisions "
            "and transfer receipts; this comparison does not select paths"
        ),
        "baseline_check": baseline_check,
        "turbobus_check": turbobus_check,
        "comparisons": comparisons,
    }


def compare_result_files(
    baseline_path: str | Path,
    turbobus_path: str | Path,
    *,
    baseline_label: str = "paper-baseline",
    turbobus_label: str = "turbobus-daemon",
) -> dict:
    report = compare_result_data(
        phase7_result_check.read_result(baseline_path),
        phase7_result_check.read_result(turbobus_path),
        baseline_label=baseline_label,
        turbobus_label=turbobus_label,
    )
    report["baseline_path"] = str(baseline_path)
    report["turbobus_path"] = str(turbobus_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare checker-approved Phase 7 baseline and TurboBus results",
    )
    parser.add_argument("--baseline", required=True, help="paper-baseline result JSON path")
    parser.add_argument("--turbobus", required=True, help="turbobus-daemon result JSON path")
    parser.add_argument("--baseline-label", default="paper-baseline")
    parser.add_argument("--turbobus-label", default="turbobus-daemon")
    parser.add_argument("--json-output", help="optional machine-readable comparison path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = compare_result_files(
        args.baseline,
        args.turbobus,
        baseline_label=args.baseline_label,
        turbobus_label=args.turbobus_label,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        Path(args.json_output).write_text(payload + "\n", encoding="utf-8")
    sys.stdout.write(payload)
    sys.stdout.write("\n")
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
