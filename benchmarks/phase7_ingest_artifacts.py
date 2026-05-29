from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import phase7_acceptance_inventory


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def read_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {"server_classes": []}
    data = phase7_acceptance_inventory.read_json(manifest_path)
    entries, errors = phase7_acceptance_inventory.server_entries(data)
    if errors:
        raise ValueError(";".join(errors))
    return {"server_classes": entries}


def write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_artifact_path(path: str | Path, manifest_path: Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return cwd_candidate
    return manifest_path.parent / raw


def stored_artifact_path(path: str | Path, manifest_path: Path) -> str:
    raw = Path(path)
    resolved = resolve_artifact_path(raw, manifest_path)
    try:
        return resolved.resolve().relative_to(manifest_path.parent.resolve()).as_posix()
    except ValueError:
        return raw.as_posix()


def validate_entry(
    entry: dict,
    *,
    manifest_path: Path,
    required_server_classes: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    server_class = str(entry.get("server_class", "") or "")
    status = str(entry.get("status", "") or "")
    if server_class not in required_server_classes:
        errors.append(f"{server_class or 'missing'}:unknown_server_class")
    if status not in phase7_acceptance_inventory.VALID_STATUSES:
        errors.append(f"{server_class}:invalid_status:{status or 'missing'}")

    bundle_gate = entry.get("bundle_gate")
    if status == "accepted":
        if not bundle_gate:
            errors.append(f"{server_class}:accepted_missing_bundle_gate")
        if not entry.get("real_artifacts"):
            errors.append(f"{server_class}:accepted_requires_real_artifacts")
        if bundle_gate:
            bundle_path = resolve_artifact_path(str(bundle_gate), manifest_path)
            if not bundle_path.exists():
                errors.append(f"{server_class}:bundle_gate_missing:{bundle_path}")
            else:
                bundle = phase7_acceptance_inventory.read_json(bundle_path)
                if not bundle.get("ok"):
                    errors.append(f"{server_class}:bundle_gate_not_ok")
    elif status in {"blocked", "missing"}:
        if not entry.get("block_reason") and not entry.get("environment_gaps"):
            errors.append(f"{server_class}:{status}_missing_environment_gap")
        if not entry.get("next_commands"):
            errors.append(f"{server_class}:{status}_missing_next_commands")
    return errors


def build_entry(
    *,
    server_class: str,
    status: str,
    manifest_path: Path,
    bundle_gate: str | None = None,
    real_artifacts: bool = False,
    block_reason: str = "",
    environment_gaps: list[str] | None = None,
    next_commands: list[str] | None = None,
    remaining_risks: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict:
    entry = {
        "server_class": server_class,
        "status": status,
    }
    if bundle_gate:
        entry["bundle_gate"] = stored_artifact_path(bundle_gate, manifest_path)
    if real_artifacts:
        entry["real_artifacts"] = True
    if block_reason:
        entry["block_reason"] = block_reason
    if environment_gaps:
        entry["environment_gaps"] = list(environment_gaps)
    if next_commands:
        entry["next_commands"] = list(next_commands)
    if remaining_risks:
        entry["remaining_risks"] = list(remaining_risks)
    if notes:
        entry["notes"] = list(notes)
    return entry


def upsert_entry(manifest: dict, entry: dict) -> dict:
    entries, errors = phase7_acceptance_inventory.server_entries(manifest)
    if errors:
        raise ValueError(";".join(errors))
    server_class = str(entry["server_class"])
    updated = False
    output = []
    for existing in entries:
        if str(existing.get("server_class", "")) == server_class:
            output.append(dict(entry))
            updated = True
        else:
            output.append(dict(existing))
    if not updated:
        output.append(dict(entry))

    order = {
        server: index
        for index, server in enumerate(phase7_acceptance_inventory.DEFAULT_SERVER_CLASSES)
    }
    output.sort(key=lambda item: (order.get(str(item.get("server_class", "")), 99), str(item.get("server_class", ""))))
    return {"server_classes": output}


def ingest_entry_file(
    *,
    manifest_path: str | Path,
    entry: dict,
    required_server_classes: tuple[str, ...] = phase7_acceptance_inventory.DEFAULT_SERVER_CLASSES,
    inventory_output_path: str | Path | None = None,
    allow_incomplete_inventory: bool = False,
) -> dict:
    manifest_file = Path(manifest_path)
    errors = validate_entry(
        entry,
        manifest_path=manifest_file,
        required_server_classes=required_server_classes,
    )
    if errors:
        return {
            "ok": False,
            "errors": errors,
            "manifest_path": str(manifest_file),
            "updated_entry": entry,
            "written": False,
        }

    manifest = upsert_entry(read_manifest(manifest_file), entry)
    write_json(manifest_file, manifest)
    inventory = phase7_acceptance_inventory.build_inventory_file(
        manifest_file,
        required_server_classes=required_server_classes,
    )
    if inventory_output_path is not None:
        write_json(inventory_output_path, inventory)

    inventory_ok = bool(inventory.get("ok"))
    return {
        "ok": inventory_ok or allow_incomplete_inventory,
        "errors": [] if inventory_ok or allow_incomplete_inventory else list(inventory.get("errors", []) or []),
        "manifest_path": str(manifest_file),
        "inventory_output_path": None if inventory_output_path is None else str(inventory_output_path),
        "updated_entry": entry,
        "written": True,
        "inventory_ok": inventory_ok,
        "inventory": inventory,
        "ingest_note": (
            "This command updates the Phase 7 acceptance manifest and runs the "
            "acceptance inventory over existing artifacts only. It does not "
            "create scheduler plans, issue execution tickets, run worker data "
            "movement, or select physical transfer paths."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest Phase 7 server artifacts into the acceptance manifest",
    )
    parser.add_argument("--manifest", required=True, help="acceptance manifest JSON path")
    parser.add_argument("--server-class", required=True, help="2gpu, 4gpu, or 8gpu")
    parser.add_argument("--status", required=True, choices=phase7_acceptance_inventory.VALID_STATUSES)
    parser.add_argument("--bundle-gate", help="bundle-gate JSON path for an accepted server class")
    parser.add_argument("--real-artifacts", action="store_true", help="mark accepted artifacts as real server output")
    parser.add_argument("--block-reason", default="", help="hardware or environment block reason")
    parser.add_argument("--environment-gap", action="append", default=[], help="repeatable hardware or environment gap")
    parser.add_argument("--next-command", action="append", default=[], help="repeatable next server command")
    parser.add_argument("--remaining-risk", action="append", default=[], help="repeatable remaining server risk")
    parser.add_argument("--note", action="append", default=[], help="repeatable operator note")
    parser.add_argument(
        "--server-classes",
        default=",".join(phase7_acceptance_inventory.DEFAULT_SERVER_CLASSES),
        help="comma-separated required server classes",
    )
    parser.add_argument("--inventory-output", help="optional acceptance inventory JSON path")
    parser.add_argument(
        "--allow-incomplete-inventory",
        action="store_true",
        help="write the manifest even if other server classes are still missing",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    entry = build_entry(
        server_class=args.server_class,
        status=args.status,
        manifest_path=manifest_path,
        bundle_gate=args.bundle_gate,
        real_artifacts=args.real_artifacts,
        block_reason=args.block_reason,
        environment_gaps=args.environment_gap,
        next_commands=args.next_command,
        remaining_risks=args.remaining_risk,
        notes=args.note,
    )
    report = ingest_entry_file(
        manifest_path=manifest_path,
        entry=entry,
        required_server_classes=tuple(parse_csv(args.server_classes)),
        inventory_output_path=args.inventory_output,
        allow_incomplete_inventory=args.allow_incomplete_inventory,
    )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
