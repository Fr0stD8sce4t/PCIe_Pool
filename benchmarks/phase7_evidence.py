from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import phase7_result_check


TRACE_ID_FIELDS = (
    "receipt_ids",
    "decision_ids",
    "topology_snapshot_ids",
    "ticket_ids",
    "prefetch_decision_ids",
    "prefetch_receipt_ids",
    "offload_decision_ids",
    "offload_receipt_ids",
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


def read_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Phase 7 evidence input JSON must be an object")
    return data


def profile_from_daemon(socket_path: str) -> dict:
    from turbobus import TurboBusClient

    response = TurboBusClient(socket_path=socket_path).describe()
    if not response.ok:
        raise RuntimeError(response.error or "daemon profile request failed")
    return dict(response.payload or {})


def normalize_profile(data: dict) -> dict:
    if "payload" in data and isinstance(data.get("payload"), dict):
        if data.get("ok") is False:
            raise ValueError(data.get("error") or "daemon profile response was not ok")
        return dict(data["payload"])
    return dict(data)


def split_values(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item not in (None, "")]
    return [item for item in str(value).split(",") if item]


def trace_ids(metric: dict) -> dict[str, list[str]]:
    return {
        field: split_values(metric.get(field))
        for field in TRACE_ID_FIELDS
        if split_values(metric.get(field))
    }


def metric_key(metric: dict, workload_name: str) -> str:
    workload = str(metric.get("workload", workload_name))
    job_index = metric.get("job_index")
    if job_index not in (None, ""):
        return f"{workload}[job_index={job_index}]"
    return workload


def iter_metric_records(result: dict) -> list[dict]:
    records = []
    for workload_result in result.get("workloads", []) or []:
        if not isinstance(workload_result, dict):
            continue
        workload_name = str(workload_result.get("workload", ""))
        workload_data = workload_result.get("data", {}) or {}
        for metric in workload_result.get("metrics", []) or []:
            if not isinstance(metric, dict):
                continue
            records.append(
                {
                    "key": metric_key(metric, workload_name),
                    "workload": str(metric.get("workload", workload_name)),
                    "metric": metric,
                    "workload_data": workload_data,
                }
            )
    return records


def metric_identity(metric: dict) -> dict[str, object]:
    return {
        field: metric[field]
        for field in IDENTITY_FIELDS
        if metric.get(field) not in (None, "")
    }


def id_set(metric: dict, *fields: str) -> set[str]:
    ids: set[str] = set()
    for field in fields:
        ids.update(split_values(metric.get(field)))
    return ids


def matches_trace_ids(record: dict, metric: dict) -> bool:
    decision_ids = id_set(
        metric,
        "decision_ids",
        "prefetch_decision_ids",
        "offload_decision_ids",
        "save_decision_ids",
    )
    ticket_ids = id_set(metric, "ticket_ids", "save_ticket_ids")
    topology_ids = id_set(
        metric,
        "topology_snapshot_ids",
        "save_topology_snapshot_ids",
    )
    if record.get("decision_id") in decision_ids:
        return True
    if record.get("ticket_id") in ticket_ids:
        return True
    if record.get("topology_snapshot_id") in topology_ids:
        return True
    return False


def matches_identity(record: dict, metric: dict) -> bool:
    job_id = metric.get("job_id")
    session_id = metric.get("session_id")
    if job_id and record.get("job_id") == job_id:
        return True
    if session_id and record.get("session_id") == session_id:
        return True
    metric_buffers = {
        value
        for value in (
            metric.get("cpu_buffer_id"),
            metric.get("gpu_buffer_id"),
            metric.get("source_buffer_id"),
            metric.get("destination_buffer_id"),
        )
        if value not in (None, "")
    }
    record_buffers = set(split_values(record.get("buffer_ids")))
    for field in ("source_buffer_id", "destination_buffer_id"):
        if record.get(field) not in (None, ""):
            record_buffers.add(str(record[field]))
    return bool(metric_buffers & record_buffers)


def matching_records(records: Iterable[dict], metric: dict) -> list[dict]:
    matched = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if matches_trace_ids(record, metric) or matches_identity(record, metric):
            matched.append(dict(record))
    return matched


def runtime_state(profile: dict) -> dict:
    runtime = profile.get("runtime_resource_state", {}) or {}
    return dict(runtime) if isinstance(runtime, dict) else {}


def profile_summary(profile: dict) -> dict[str, object]:
    runtime = runtime_state(profile)
    summary = runtime.get("summary", {}) or {}
    active_usage = summary.get("active_resource_usage", {}) or {}
    p2p = active_usage.get("p2p", {}) or {}
    relay_staging = active_usage.get("relay_staging", {}) or {}
    relay_quotas = profile.get("relay_quotas", {}) or {}
    quota_values = relay_quotas.values() if isinstance(relay_quotas, dict) else []
    return {
        "runtime_state_version": int(runtime.get("version", 0) or 0),
        "queued_transfer_count": int(summary.get("queued_transfer_count", 0) or 0),
        "running_transfer_count": int(summary.get("running_transfer_count", 0) or 0),
        "active_transfer_count": int(summary.get("active_transfer_count", 0) or 0),
        "terminal_transfer_count": int(summary.get("terminal_transfer_count", 0) or 0),
        "active_reservation_count": int(summary.get("active_reservation_count", 0) or 0),
        "active_lease_count": int(summary.get("active_lease_count", 0) or 0),
        "relay_staging_count": int(summary.get("relay_staging_count", 0) or 0),
        "relay_path_count": int(summary.get("relay_path_count", 0) or 0),
        "relay_path_bytes_total": int(summary.get("relay_path_bytes_total", 0) or 0),
        "p2p_bytes_total": int(p2p.get("bytes_total", 0) or 0),
        "relay_staging_active_leases": int(relay_staging.get("active_lease_count", 0) or 0),
        "relay_quota_count": len(list(quota_values)),
        "relay_quota_active_chunks": sum(
            int(quota.get("active_chunks", 0) or 0)
            for quota in quota_values
            if isinstance(quota, dict)
        ),
        "audit_record_count": len(profile.get("audit_records", []) or []),
        "cleanup_event_count": len(profile.get("cleanup_events", []) or []),
        "system_cleanup_event_count": len(profile.get("system_cleanup_events", []) or []),
    }


def matched_failure_and_fallback(
    metric: dict,
    transfer_records: list[dict],
    audit_records: list[dict],
) -> dict[str, object]:
    reasons = []
    metric_reason = str(metric.get("fallback_reason", "") or "")
    if metric_reason and metric_reason != "none":
        reasons.append(metric_reason)
    for record in [*transfer_records, *audit_records]:
        for field in ("fallback_reason", "failure_reason", "reason", "error"):
            value = record.get(field)
            if value not in (None, "", "none"):
                reasons.append(str(value))
    return {
        "reasons": sorted(set(reasons), key=reasons.index),
        "has_failure_or_fallback": bool(reasons),
    }


def relay_quota_summary(profile: dict) -> list[dict]:
    relay_quotas = profile.get("relay_quotas", {}) or {}
    if not isinstance(relay_quotas, dict):
        return []
    return [
        dict(quota)
        for _, quota in sorted(relay_quotas.items(), key=lambda item: str(item[0]))
        if isinstance(quota, dict)
    ]


def metric_evidence(record: dict, profile: dict) -> dict:
    metric = record["metric"]
    runtime = runtime_state(profile)
    transfers = runtime.get("transfers", []) or []
    active_transfers = runtime.get("active_transfers", []) or []
    active_paths = runtime.get("active_paths", []) or []
    audit_records = profile.get("audit_records", []) or []
    staging_records = list((profile.get("staging_records", {}) or {}).values())
    staging_records.extend(runtime.get("relay_staging", []) or [])

    matched_transfers = matching_records(transfers, metric)
    matched_active_transfers = matching_records(active_transfers, metric)
    matched_transfer_ids = {
        str(item.get("transfer_id"))
        for item in [*matched_transfers, *matched_active_transfers]
        if item.get("transfer_id") not in (None, "")
    }
    matched_paths = [
        dict(item)
        for item in active_paths
        if isinstance(item, dict)
        and (
            str(item.get("transfer_id")) in matched_transfer_ids
            or matches_trace_ids(item, metric)
            or matches_identity(item, metric)
        )
    ]
    matched_audit = matching_records(audit_records, metric)
    matched_staging = matching_records(staging_records, metric)
    job_id = metric.get("job_id")
    job_runtime = (runtime.get("job_runtime_state", {}) or {}).get(str(job_id), {})
    fallback = matched_failure_and_fallback(metric, matched_transfers, matched_audit)
    relay_evidence = [
        item
        for item in [*matched_paths, *matched_audit, *matched_staging]
        if item.get("relay_gpu") not in (None, "")
        or item.get("relay_device") not in (None, "")
        or str(item.get("kind", "")).lower() == "relay"
    ]
    flags = {
        "has_transfer_record": bool(matched_transfers),
        "has_audit_record": bool(matched_audit),
        "has_runtime_job_state": bool(job_runtime),
        "has_relay_evidence": bool(relay_evidence),
        "has_active_contention_state": bool(matched_active_transfers)
        or int((runtime.get("summary", {}) or {}).get("active_transfer_count", 0) or 0) > 1,
        "has_failure_or_fallback_evidence": fallback["has_failure_or_fallback"],
    }
    return {
        "key": record["key"],
        "workload": record["workload"],
        "identity": metric_identity(metric),
        "trace_ids": trace_ids(metric),
        "transfer_records": matched_transfers,
        "active_transfer_records": matched_active_transfers,
        "active_path_records": matched_paths,
        "audit_records": matched_audit,
        "staging_records": matched_staging,
        "job_runtime_state": dict(job_runtime) if isinstance(job_runtime, dict) else {},
        "relay_quotas": relay_quota_summary(profile),
        "failure_or_fallback": fallback,
        "flags": flags,
    }


def comparison_summary(comparison: dict | None) -> dict[str, object]:
    if not comparison:
        return {}
    return {
        "ok": bool(comparison.get("ok")),
        "errors": list(comparison.get("errors", []) or []),
        "comparison_count": len(comparison.get("comparisons", []) or []),
        "path_selection_note": comparison.get("path_selection_note", ""),
    }


def build_evidence_report(
    result: dict,
    profile: dict,
    *,
    comparison: dict | None = None,
) -> dict:
    result_check = phase7_result_check.check_phase7_result(result)
    errors: list[str] = []
    if not result_check["ok"]:
        errors.append("result_checker_failed")
    normalized_profile = normalize_profile(profile)
    runtime = runtime_state(normalized_profile)
    if not runtime:
        errors.append("missing_runtime_resource_state")
    if not isinstance(normalized_profile.get("audit_records", []), list):
        errors.append("missing_audit_records")
    comparison_info = comparison_summary(comparison)
    if comparison_info and not comparison_info["ok"]:
        errors.append("comparison_not_ok")

    workload_evidence = []
    if result_check["ok"]:
        for record in iter_metric_records(result):
            evidence = metric_evidence(record, normalized_profile)
            workload_evidence.append(evidence)
            flags = evidence["flags"]
            if not flags["has_transfer_record"] and not flags["has_audit_record"]:
                errors.append(f"{record['key']}:missing_daemon_trace_evidence")

    return {
        "ok": not errors,
        "errors": errors,
        "result_check": result_check,
        "profile_summary": profile_summary(normalized_profile),
        "comparison": comparison_info,
        "workloads": workload_evidence,
        "evidence_note": (
            "This report reads paper-validation output, comparison output, and "
            "daemon profile state. It does not create transfer plans or select "
            "direct, relay, or pooled paths."
        ),
    }


def build_evidence_files(
    result_path: str | Path,
    *,
    profile_path: str | Path | None = None,
    daemon_socket_path: str | None = None,
    comparison_path: str | Path | None = None,
) -> dict:
    result = phase7_result_check.read_result(result_path)
    if profile_path is not None:
        profile = read_json(profile_path)
        profile_source = str(profile_path)
    elif daemon_socket_path:
        profile = profile_from_daemon(daemon_socket_path)
        profile_source = f"daemon:{daemon_socket_path}"
    else:
        raise ValueError("either --profile or --daemon-socket-path is required")
    comparison = read_json(comparison_path) if comparison_path is not None else None
    report = build_evidence_report(result, profile, comparison=comparison)
    report["result_path"] = str(result_path)
    report["profile_source"] = profile_source
    if comparison_path is not None:
        report["comparison_path"] = str(comparison_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach daemon-side evidence to accepted Phase 7 results",
    )
    parser.add_argument("--result", required=True, help="paper-validation JSON or summary path")
    parser.add_argument("--profile", help="daemon PROFILE payload or response JSON path")
    parser.add_argument("--daemon-socket-path", help="fetch daemon PROFILE from this socket")
    parser.add_argument("--comparison", help="optional phase7_compare JSON path")
    parser.add_argument("--json-output", help="optional machine-readable evidence path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_evidence_files(
        args.result,
        profile_path=args.profile,
        daemon_socket_path=args.daemon_socket_path,
        comparison_path=args.comparison,
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
