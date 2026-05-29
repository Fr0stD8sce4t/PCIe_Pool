from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import phase7_result_check


DEFAULT_REQUIRED_WORKLOADS = (
    "model-loading",
    "training-offload",
    "optimizer-offload",
    "vllm-kv",
)


def read_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Phase 7 bundle artifact JSON must be an object")
    return data


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def result_policy(result: dict) -> str:
    config = result.get("config", {}) or {}
    return str(config.get("policy", "") or "")


def workload_names(result: dict) -> set[str]:
    names = set()
    for workload in result.get("workloads", []) or []:
        if isinstance(workload, dict) and workload.get("workload"):
            names.add(str(workload["workload"]))
    return names


def base_workload_name(key: object) -> str:
    text = str(key)
    if "[" in text:
        return text.split("[", 1)[0]
    return text


def check_report_summary(result: dict, provided_check: dict | None) -> dict:
    computed = phase7_result_check.check_phase7_result(result)
    provided = provided_check or {}
    return {
        "ok": bool(computed.get("ok")) and (
            True if not provided else bool(provided.get("ok"))
        ),
        "computed": computed,
        "provided": provided,
        "source": "provided_and_computed" if provided_check is not None else "computed",
    }


def side_summary(
    name: str,
    result: dict,
    provided_check: dict | None,
    *,
    expected_policy: str,
    required_workloads: tuple[str, ...],
) -> tuple[dict, list[str]]:
    errors: list[str] = []
    policy = result_policy(result)
    if policy != expected_policy:
        errors.append(f"{name}:policy_mismatch:{policy or 'missing'}")
    workloads = workload_names(result)
    for workload in required_workloads:
        if workload not in workloads:
            errors.append(f"{name}:missing_workload:{workload}")
    check = check_report_summary(result, provided_check)
    if not check["computed"]["ok"]:
        errors.append(f"{name}:computed_check_failed")
    if provided_check is not None and not provided_check.get("ok"):
        errors.append(f"{name}:provided_check_failed")
    return (
        {
            "policy": policy,
            "workloads": sorted(workloads),
            "missing_workloads": [
                workload for workload in required_workloads if workload not in workloads
            ],
            "check": check,
        },
        errors,
    )


def comparison_summary(comparison: dict, required_workloads: tuple[str, ...]) -> tuple[dict, list[str]]:
    errors: list[str] = []
    if not comparison.get("ok"):
        errors.append("comparison:not_ok")
    comparison_workloads = {
        base_workload_name(item.get("workload", item.get("key", "")))
        for item in comparison.get("comparisons", []) or []
        if isinstance(item, dict)
    }
    for workload in required_workloads:
        if workload not in comparison_workloads:
            errors.append(f"comparison:missing_workload:{workload}")
    return (
        {
            "ok": bool(comparison.get("ok")),
            "errors": list(comparison.get("errors", []) or []),
            "workloads": sorted(comparison_workloads),
            "comparison_count": len(comparison.get("comparisons", []) or []),
        },
        errors,
    )


def evidence_summary(evidence_reports: list[dict], required_workloads: tuple[str, ...]) -> tuple[dict, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    evidence_workloads: set[str] = set()
    reports = []
    if not evidence_reports:
        errors.append("evidence:missing")
    for index, evidence in enumerate(evidence_reports):
        if not evidence.get("ok"):
            errors.append(f"evidence:{index}:not_ok")
        workloads = {
            base_workload_name(item.get("workload", item.get("key", "")))
            for item in evidence.get("workloads", []) or []
            if isinstance(item, dict)
        }
        evidence_workloads.update(workloads)
        missing_trace = [
            item.get("key", item.get("workload", "unknown"))
            for item in evidence.get("workloads", []) or []
            if isinstance(item, dict)
            and not (
                (item.get("flags", {}) or {}).get("has_transfer_record")
                or (item.get("flags", {}) or {}).get("has_audit_record")
            )
        ]
        if missing_trace:
            errors.append(f"evidence:{index}:missing_daemon_trace")
        if not evidence.get("comparison"):
            warnings.append(f"evidence:{index}:comparison_not_attached")
        reports.append(
            {
                "ok": bool(evidence.get("ok")),
                "errors": list(evidence.get("errors", []) or []),
                "workloads": sorted(workloads),
                "profile_summary": dict(evidence.get("profile_summary", {}) or {}),
                "missing_trace": missing_trace,
            }
        )
    for workload in required_workloads:
        if workload not in evidence_workloads:
            errors.append(f"evidence:missing_workload:{workload}")
    return (
        {
            "ok": not any(not item["ok"] for item in reports) and not errors,
            "workloads": sorted(evidence_workloads),
            "reports": reports,
        },
        [*errors, *warnings],
    )


def correctness_item_status(data: object) -> tuple[bool | None, list[str]]:
    if isinstance(data, list):
        states = [correctness_item_status(item) for item in data]
        child_errors = [error for _, errors in states for error in errors]
        child_ok = [state for state, _ in states if state is not None]
        if not child_ok:
            return None, child_errors or ["correctness_status_unknown"]
        return all(child_ok), child_errors
    if not isinstance(data, dict):
        return None, ["correctness_status_unknown"]
    if "ok" in data:
        return bool(data.get("ok")), list(data.get("errors", []) or [])
    if "returncode" in data:
        return int(data.get("returncode", 1) or 0) == 0, []
    if "commands" in data:
        return correctness_item_status(data.get("commands"))
    status = str(data.get("status", "")).lower()
    if status in {"ok", "pass", "passed", "complete", "success"}:
        return True, []
    if status in {"failed", "fail", "error"}:
        return False, [str(data.get("error", "correctness_failed"))]
    return None, ["correctness_status_unknown"]


def correctness_summary(reports: list[dict]) -> tuple[dict, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not reports:
        warnings.append("correctness:not_provided")
        return {"provided": False, "ok": None, "reports": []}, errors, warnings
    normalized = []
    all_ok = True
    for index, report in enumerate(reports):
        ok, item_errors = correctness_item_status(report)
        if ok is False:
            errors.append(f"correctness:{index}:failed")
            all_ok = False
        elif ok is None:
            warnings.append(f"correctness:{index}:unknown")
            all_ok = False
        normalized.append(
            {
                "ok": ok,
                "errors": item_errors,
                "summary": report.get("summary", {}) if isinstance(report, dict) else {},
            }
        )
    return {"provided": True, "ok": all_ok, "reports": normalized}, errors, warnings


def build_bundle_report(
    *,
    baseline_result: dict,
    turbobus_result: dict,
    comparison: dict,
    evidence_reports: list[dict],
    baseline_check: dict | None = None,
    turbobus_check: dict | None = None,
    correctness_reports: list[dict] | None = None,
    server_class: str = "unknown",
    required_workloads: tuple[str, ...] = DEFAULT_REQUIRED_WORKLOADS,
    baseline_label: str = "paper-baseline",
    turbobus_label: str = "turbobus-daemon",
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    baseline, side_errors = side_summary(
        "baseline",
        baseline_result,
        baseline_check,
        expected_policy=baseline_label,
        required_workloads=required_workloads,
    )
    errors.extend(side_errors)
    turbobus, side_errors = side_summary(
        "turbobus",
        turbobus_result,
        turbobus_check,
        expected_policy=turbobus_label,
        required_workloads=required_workloads,
    )
    errors.extend(side_errors)
    comparison_info, comparison_errors = comparison_summary(comparison, required_workloads)
    errors.extend(comparison_errors)
    evidence_info, evidence_messages = evidence_summary(evidence_reports, required_workloads)
    for message in evidence_messages:
        if message.endswith("comparison_not_attached"):
            warnings.append(message)
        else:
            errors.append(message)
    correctness_info, correctness_errors, correctness_warnings = correctness_summary(
        correctness_reports or []
    )
    errors.extend(correctness_errors)
    warnings.extend(correctness_warnings)
    if "vllm-kv" not in turbobus["workloads"]:
        errors.append("real_workload:vllm_kv_missing")

    return {
        "ok": not errors,
        "errors": sorted(set(errors), key=errors.index),
        "warnings": sorted(set(warnings), key=warnings.index),
        "server_class": server_class,
        "required_workloads": list(required_workloads),
        "baseline_label": baseline_label,
        "turbobus_label": turbobus_label,
        "baseline": baseline,
        "turbobus": turbobus,
        "comparison": comparison_info,
        "evidence": evidence_info,
        "correctness": correctness_info,
        "gate_note": (
            "This gate reads existing Phase 7 artifacts only. It does not create "
            "scheduler plans, issue execution tickets, run worker data movement, "
            "or select physical transfer paths."
        ),
    }


def build_bundle_files(
    *,
    baseline_result_path: str | Path,
    turbobus_result_path: str | Path,
    comparison_path: str | Path,
    evidence_paths: list[str | Path],
    baseline_check_path: str | Path | None = None,
    turbobus_check_path: str | Path | None = None,
    correctness_paths: list[str | Path] | None = None,
    server_class: str = "unknown",
    required_workloads: tuple[str, ...] = DEFAULT_REQUIRED_WORKLOADS,
    baseline_label: str = "paper-baseline",
    turbobus_label: str = "turbobus-daemon",
) -> dict:
    report = build_bundle_report(
        baseline_result=phase7_result_check.read_result(baseline_result_path),
        turbobus_result=phase7_result_check.read_result(turbobus_result_path),
        baseline_check=read_json(baseline_check_path) if baseline_check_path else None,
        turbobus_check=read_json(turbobus_check_path) if turbobus_check_path else None,
        comparison=read_json(comparison_path),
        evidence_reports=[read_json(path) for path in evidence_paths],
        correctness_reports=[
            read_json(path) for path in (correctness_paths or [])
        ],
        server_class=server_class,
        required_workloads=required_workloads,
        baseline_label=baseline_label,
        turbobus_label=turbobus_label,
    )
    report["artifacts"] = {
        "baseline_result": str(baseline_result_path),
        "turbobus_result": str(turbobus_result_path),
        "baseline_check": None if baseline_check_path is None else str(baseline_check_path),
        "turbobus_check": None if turbobus_check_path is None else str(turbobus_check_path),
        "comparison": str(comparison_path),
        "evidence": [str(path) for path in evidence_paths],
        "correctness": [str(path) for path in (correctness_paths or [])],
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate a Phase 7 server run bundle from existing artifacts",
    )
    parser.add_argument("--server-class", default="unknown")
    parser.add_argument("--baseline-result", required=True)
    parser.add_argument("--turbobus-result", required=True)
    parser.add_argument("--baseline-check")
    parser.add_argument("--turbobus-check")
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--evidence", action="append", default=[], help="repeatable evidence JSON path")
    parser.add_argument("--correctness", action="append", default=[], help="optional correctness gate JSON path")
    parser.add_argument("--required-workloads", default=",".join(DEFAULT_REQUIRED_WORKLOADS))
    parser.add_argument("--baseline-label", default="paper-baseline")
    parser.add_argument("--turbobus-label", default="turbobus-daemon")
    parser.add_argument("--json-output", help="optional machine-readable bundle gate path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_bundle_files(
        baseline_result_path=args.baseline_result,
        turbobus_result_path=args.turbobus_result,
        baseline_check_path=args.baseline_check,
        turbobus_check_path=args.turbobus_check,
        comparison_path=args.comparison,
        evidence_paths=args.evidence,
        correctness_paths=args.correctness,
        server_class=args.server_class,
        required_workloads=tuple(parse_csv(args.required_workloads)),
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
