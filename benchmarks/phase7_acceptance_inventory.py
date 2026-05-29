from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


DEFAULT_SERVER_CLASSES = ("2gpu", "4gpu", "8gpu")
VALID_STATUSES = ("accepted", "blocked", "missing")
REAL_WORKLOAD = "vllm-kv"


def read_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Phase 7 acceptance inventory JSON must be an object")
    return data


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def as_list(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def resolve_manifest_path(path: object, base_dir: Path | None) -> Path | None:
    if path in (None, ""):
        return None
    candidate = Path(str(path))
    if candidate.is_absolute() or base_dir is None:
        return candidate
    return base_dir / candidate


def server_entries(manifest: dict) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    raw = manifest.get("server_classes", manifest.get("servers", []))
    if isinstance(raw, dict):
        entries = []
        for server_class, entry in raw.items():
            if not isinstance(entry, dict):
                errors.append(f"{server_class}:entry_not_object")
                continue
            normalized = dict(entry)
            normalized.setdefault("server_class", server_class)
            entries.append(normalized)
        return entries, errors
    if isinstance(raw, list):
        entries = []
        for index, entry in enumerate(raw):
            if not isinstance(entry, dict):
                errors.append(f"entry_{index}:not_object")
                continue
            entries.append(dict(entry))
        return entries, errors
    return [], ["manifest:server_classes_not_list_or_object"]


def bundle_workloads(bundle: dict) -> list[str]:
    workloads: set[str] = set()
    for section in ("turbobus", "baseline", "comparison", "evidence"):
        value = bundle.get(section, {}) or {}
        if not isinstance(value, dict):
            continue
        for workload in value.get("workloads", []) or []:
            workloads.add(str(workload))
        for item in value.get("comparisons", []) or []:
            if isinstance(item, dict):
                workload = item.get("workload", item.get("key"))
                if workload not in (None, ""):
                    workloads.add(str(workload).split("[", 1)[0])
    return sorted(workloads)


def bundle_summary(bundle_path: Path | None) -> tuple[dict, list[str], dict | None]:
    errors: list[str] = []
    if bundle_path is None:
        return {"provided": False}, errors, None
    if not bundle_path.exists():
        return {
            "provided": True,
            "path": str(bundle_path),
            "exists": False,
            "ok": False,
            "workloads": [],
            "has_real_workload": False,
        }, [f"bundle_gate_missing:{bundle_path}"], None

    bundle = read_json(bundle_path)
    workloads = bundle_workloads(bundle)
    summary = {
        "provided": True,
        "path": str(bundle_path),
        "exists": True,
        "ok": bool(bundle.get("ok")),
        "server_class": str(bundle.get("server_class", "")),
        "errors": list(bundle.get("errors", []) or []),
        "warnings": list(bundle.get("warnings", []) or []),
        "workloads": workloads,
        "has_real_workload": REAL_WORKLOAD in workloads,
        "artifacts": dict(bundle.get("artifacts", {}) or {}),
    }
    if not summary["ok"]:
        errors.append("bundle_gate_not_ok")
    return summary, errors, bundle


def status_for_entry(entry: dict) -> str:
    status = str(entry.get("status", "") or "").strip().lower()
    if status:
        return status
    if entry.get("bundle_gate") or entry.get("bundle_gate_path"):
        return "accepted"
    return "missing"


def real_artifacts_flag(entry: dict) -> bool:
    if "real_artifacts" in entry:
        return bool(entry.get("real_artifacts"))
    return str(entry.get("artifact_source", "")).lower() in {"server", "real-server"}


def entry_gap(entry: dict) -> dict:
    block_reason = str(
        entry.get("block_reason")
        or entry.get("missing_reason")
        or entry.get("environment_gap")
        or ""
    )
    gaps = as_list(entry.get("environment_gaps"))
    if entry.get("environment_gap") not in (None, ""):
        gaps.append(str(entry["environment_gap"]))
    return {
        "block_reason": block_reason,
        "environment_gaps": sorted(set(gaps), key=gaps.index),
    }


def summarize_entry(
    entry: dict,
    *,
    base_dir: Path | None,
) -> tuple[dict, list[str], list[str]]:
    server_class = str(entry.get("server_class", "") or "")
    errors: list[str] = []
    warnings: list[str] = []
    if not server_class:
        server_class = "unknown"
        errors.append("server_class_missing")

    status = status_for_entry(entry)
    if status not in VALID_STATUSES:
        errors.append(f"{server_class}:invalid_status:{status or 'missing'}")

    bundle_path = resolve_manifest_path(
        entry.get("bundle_gate", entry.get("bundle_gate_path")),
        base_dir,
    )
    bundle, bundle_errors, _ = bundle_summary(bundle_path)
    real_artifacts = real_artifacts_flag(entry)
    gap = entry_gap(entry)
    next_commands = as_list(entry.get("next_commands", entry.get("commands")))
    remaining_risks = as_list(entry.get("remaining_risks", entry.get("risks")))
    notes = as_list(entry.get("notes"))

    if status == "accepted":
        if not bundle.get("provided"):
            errors.append(f"{server_class}:accepted_missing_bundle_gate")
        errors.extend(f"{server_class}:{error}" for error in bundle_errors)
        if not real_artifacts:
            errors.append(f"{server_class}:accepted_requires_real_artifacts")
        if not bundle.get("has_real_workload"):
            errors.append(f"{server_class}:accepted_missing_vllm_kv_bundle")
    elif status in {"blocked", "missing"}:
        if not gap["block_reason"] and not gap["environment_gaps"]:
            errors.append(f"{server_class}:{status}_missing_environment_gap")
        if not next_commands:
            errors.append(f"{server_class}:{status}_missing_next_commands")
        if bundle.get("provided") and not bundle.get("ok"):
            warnings.extend(f"{server_class}:{error}" for error in bundle_errors)

    summary = {
        "server_class": server_class,
        "status": status,
        "ok": not errors,
        "real_artifacts": real_artifacts,
        "bundle_gate": bundle,
        "real_workload": {
            "name": REAL_WORKLOAD,
            "accepted": (
                status == "accepted"
                and real_artifacts
                and bool(bundle.get("ok"))
                and bool(bundle.get("has_real_workload"))
            ),
        },
        "block_reason": gap["block_reason"],
        "environment_gaps": gap["environment_gaps"],
        "remaining_risks": remaining_risks,
        "next_commands": next_commands,
        "notes": notes,
    }
    return summary, errors, warnings


def build_inventory(
    manifest: dict,
    *,
    base_dir: Path | None = None,
    required_server_classes: tuple[str, ...] = DEFAULT_SERVER_CLASSES,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    entries, entry_errors = server_entries(manifest)
    errors.extend(entry_errors)

    by_class: dict[str, dict] = {}
    for entry in entries:
        server_class = str(entry.get("server_class", "") or "")
        if not server_class:
            errors.append("manifest:entry_missing_server_class")
            continue
        if server_class in by_class:
            errors.append(f"{server_class}:duplicate_entry")
            continue
        by_class[server_class] = entry

    summaries = []
    ordered = list(required_server_classes)
    ordered.extend(sorted(server for server in by_class if server not in ordered))
    for server_class in ordered:
        entry = by_class.get(server_class)
        if entry is None:
            errors.append(f"{server_class}:missing_manifest_entry")
            summaries.append(
                {
                    "server_class": server_class,
                    "status": "missing",
                    "ok": False,
                    "real_artifacts": False,
                    "bundle_gate": {"provided": False},
                    "real_workload": {"name": REAL_WORKLOAD, "accepted": False},
                    "block_reason": "",
                    "environment_gaps": [],
                    "remaining_risks": [],
                    "next_commands": [],
                    "notes": [],
                }
            )
            continue
        summary, summary_errors, summary_warnings = summarize_entry(
            entry,
            base_dir=base_dir,
        )
        summaries.append(summary)
        errors.extend(summary_errors)
        warnings.extend(summary_warnings)

    real_workload_accepted = any(
        summary.get("real_workload", {}).get("accepted") for summary in summaries
    )
    if not real_workload_accepted:
        errors.append("phase7:no_accepted_real_llm_framework_bundle")

    accepted = [item for item in summaries if item["status"] == "accepted" and item["ok"]]
    blocked = [item for item in summaries if item["status"] == "blocked"]
    missing = [item for item in summaries if item["status"] == "missing"]
    next_commands = [
        {"server_class": item["server_class"], "commands": item["next_commands"]}
        for item in summaries
        if item.get("next_commands")
    ]
    remaining_risks = []
    for item in summaries:
        for risk in item.get("remaining_risks", []):
            remaining_risks.append(
                {"server_class": item["server_class"], "risk": risk}
            )
        for gap in item.get("environment_gaps", []):
            remaining_risks.append(
                {"server_class": item["server_class"], "risk": gap}
            )

    return {
        "ok": not errors,
        "errors": sorted(set(errors), key=errors.index),
        "warnings": sorted(set(warnings), key=warnings.index),
        "schema": "phase7_acceptance_inventory_v1",
        "accepted_count": len(accepted),
        "blocked_count": len(blocked),
        "missing_count": len(missing),
        "real_workload_accepted": real_workload_accepted,
        "server_classes": summaries,
        "next_commands": next_commands,
        "remaining_server_only_risks": remaining_risks,
        "inventory_note": (
            "This inventory reads existing Phase 7 result-check, comparison, "
            "daemon-evidence, bundle-gate, and correctness artifacts only. It "
            "does not create scheduler plans, issue execution tickets, run "
            "worker data movement, or select physical transfer paths."
        ),
    }


def build_inventory_file(
    manifest_path: str | Path,
    *,
    required_server_classes: tuple[str, ...] = DEFAULT_SERVER_CLASSES,
) -> dict:
    path = Path(manifest_path)
    report = build_inventory(
        read_json(path),
        base_dir=path.parent,
        required_server_classes=required_server_classes,
    )
    report["manifest_path"] = str(path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the final Phase 7 server-run acceptance inventory",
    )
    parser.add_argument("--manifest", required=True, help="acceptance manifest JSON path")
    parser.add_argument(
        "--server-classes",
        default=",".join(DEFAULT_SERVER_CLASSES),
        help="comma-separated required server classes",
    )
    parser.add_argument("--json-output", help="optional machine-readable inventory path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_inventory_file(
        args.manifest,
        required_server_classes=tuple(parse_csv(args.server_classes)),
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
