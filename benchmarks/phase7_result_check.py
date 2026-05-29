from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import paper_validation


REQUIRED_PHASE7_METRIC_FIELDS = tuple(paper_validation.REQUIRED_UNIFIED_METRIC_FIELDS)
EXPECTED_WORKLOAD_KINDS = {
    "model-loading": "model_weights",
    "training-offload": "training_state",
    "optimizer-offload": "optimizer_state",
    "vllm-kv": "kv_cache",
}


def read_result(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return read_json_text(text)
    return parse_summary_text(text)


def read_json_text(text: str) -> dict:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("phase7 result JSON must be an object")
    return data


def parse_summary_text(text: str) -> dict:
    workloads: dict[str, dict] = {}
    order: list[str] = []
    for line in text.splitlines():
        if line.startswith("paper_workload "):
            fields = parse_fields(line)
            workload = str(fields.get("workload", ""))
            if not workload:
                continue
            if workload not in workloads:
                order.append(workload)
            workloads[workload] = {
                "workload": workload,
                "status": fields.get("status", ""),
                "validation_errors": split_values(fields.get("validation_errors", "")),
                "metrics": workloads.get(workload, {}).get("metrics", []),
            }
        if line.startswith("paper_metric "):
            metric = parse_fields(line)
            workload = str(metric.get("workload", ""))
            if not workload:
                continue
            if workload not in workloads:
                order.append(workload)
                workloads[workload] = {
                    "workload": workload,
                    "status": "",
                    "validation_errors": [],
                    "metrics": [],
                }
            workloads[workload]["metrics"].append(metric)
    if not workloads:
        raise ValueError("phase7 summary must contain paper_workload or paper_metric lines")
    return {"workloads": [workloads[workload] for workload in order]}


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def split_values(value: object) -> list[str]:
    return [item for item in str(value).split(",") if item]


def metric_errors(metric: dict) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_PHASE7_METRIC_FIELDS:
        if metric.get(field) in (None, ""):
            errors.append(f"missing_{field}")

    workload = str(metric.get("workload", ""))
    expected_kind = EXPECTED_WORKLOAD_KINDS.get(workload)
    if expected_kind is None:
        errors.append("unknown_workload")
    elif metric.get("workload_kind") != expected_kind:
        errors.append("invalid_workload_kind")

    if metric.get("report_schema") != paper_validation.PHASE6_REPORT_SCHEMA:
        errors.append("invalid_report_schema")
    if int(metric.get("bytes_completed", 0) or 0) != int(metric.get("transfer_bytes", 0) or 0):
        errors.append("bytes_not_fully_completed")
    if metric.get("correctness_status") != "complete":
        errors.append("invalid_correctness_status")

    direct_bytes = int(metric.get("direct_bytes", 0) or 0)
    relay_bytes = int(metric.get("relay_bytes", 0) or 0)
    direct_chunks = int(metric.get("direct_chunks", 0) or 0)
    relay_chunks = int(metric.get("relay_chunks", 0) or 0)
    if direct_bytes + relay_bytes <= 0:
        errors.append("missing_path_bytes")
    if direct_chunks + relay_chunks <= 0:
        errors.append("missing_path_chunks")
    if float(metric.get("performance_ms", 0.0) or 0.0) <= 0.0:
        errors.append("missing_performance_ms")
    if float(metric.get("transfer_ms", 0.0) or 0.0) <= 0.0:
        errors.append("missing_transfer_ms")

    return sorted(set(errors), key=errors.index)


def workload_errors(workload_result: dict) -> list[str]:
    errors: list[str] = []
    if workload_result.get("status") != "ok":
        errors.append("workload_not_ok")
    if workload_result.get("validation_errors"):
        errors.append("workload_validation_errors")
    metrics = workload_result.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        errors.append("missing_paper_metrics")
    return errors


def multi_job_errors(workload_result: dict) -> list[str]:
    if workload_result.get("workload") != "vllm-kv":
        return []
    data = workload_result.get("data", {}) or {}
    multi_job = data.get("vllm_kv_multi_job", {}) or {}
    job_count = int(multi_job.get("job_count", 0) or 0)
    if job_count <= 1:
        return []

    metrics = workload_result.get("metrics", []) or []
    errors: list[str] = []
    if len(metrics) != job_count:
        errors.append("multi_job_metric_count_mismatch")
    for field in ("job_id", "session_id", "cpu_buffer_id", "gpu_buffer_id"):
        values = {str(metric.get(field, "")) for metric in metrics}
        if "" in values or len(values) != len(metrics):
            errors.append(f"multi_job_{field}_not_distinct")
    return errors


def check_phase7_result(data: dict) -> dict:
    workloads = data.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        return {
            "ok": False,
            "errors": ["missing_workloads"],
            "workloads": [],
        }

    checked_workloads = []
    all_errors: list[str] = []
    for index, workload_result in enumerate(workloads):
        workload_name = str(workload_result.get("workload", ""))
        errors = workload_errors(workload_result)
        errors.extend(multi_job_errors(workload_result))
        checked_metrics = []
        metrics = workload_result.get("metrics", [])
        if not isinstance(metrics, list):
            metrics = []
        for metric_index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                errors.append(f"metric_{metric_index}_not_object")
                checked_metrics.append(
                    {
                        "index": metric_index,
                        "workload": workload_name,
                        "errors": ["not_object"],
                    }
                )
                continue
            metric_error_list = metric_errors(metric)
            checked_metrics.append(
                {
                    "index": metric_index,
                    "workload": str(metric.get("workload", workload_name)),
                    "errors": metric_error_list,
                }
            )
            errors.extend(f"metric_{metric_index}_{error}" for error in metric_error_list)
        errors = sorted(set(errors), key=errors.index)
        checked_workloads.append(
            {
                "index": index,
                "workload": workload_name,
                "status": str(workload_result.get("status", "")),
                "errors": errors,
                "metrics": checked_metrics,
            }
        )
        all_errors.extend(f"{workload_name}:{error}" for error in errors)

    all_errors = sorted(set(all_errors), key=all_errors.index)
    return {
        "ok": not all_errors,
        "errors": all_errors,
        "workloads": checked_workloads,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Phase 7 paper-validation result traceability",
    )
    parser.add_argument("result_path", help="paper-validation JSON or summary result path")
    parser.add_argument("--json-output", help="optional machine-readable check report path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = check_phase7_result(read_result(args.result_path))
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        Path(args.json_output).write_text(payload + "\n", encoding="utf-8")
    sys.stdout.write(payload)
    sys.stdout.write("\n")
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
