from __future__ import annotations

import json
from typing import Mapping

from .helper import WorkerServiceRequestEnvelope, WorkerServiceResponseEnvelope


class WorkerMessageCodecError(ValueError):
    pass


def encode_worker_request_envelope(
    envelope: WorkerServiceRequestEnvelope,
) -> str:
    if not isinstance(envelope, WorkerServiceRequestEnvelope):
        raise TypeError("envelope must be a WorkerServiceRequestEnvelope")
    return _encode_json(envelope.as_dict())


def decode_worker_request_envelope(
    message: str | bytes,
) -> WorkerServiceRequestEnvelope:
    payload = _decode_json_mapping(message)
    try:
        return WorkerServiceRequestEnvelope(
            payload=_required_mapping(payload, "payload"),
            cleanup_target_kind=str(payload.get("cleanup_target_kind", "reservation")),
        )
    except (TypeError, ValueError) as exc:
        raise WorkerMessageCodecError(str(exc)) from exc


def encode_worker_response_envelope(
    envelope: WorkerServiceResponseEnvelope,
) -> str:
    if not isinstance(envelope, WorkerServiceResponseEnvelope):
        raise TypeError("envelope must be a WorkerServiceResponseEnvelope")
    return _encode_json(envelope.as_dict())


def decode_worker_response_envelope(
    message: str | bytes,
) -> WorkerServiceResponseEnvelope:
    payload = _decode_json_mapping(message)
    try:
        return WorkerServiceResponseEnvelope(
            ok=bool(payload["ok"]),
            lifecycle=_optional_mapping(payload, "lifecycle"),
            completion=_optional_mapping(payload, "completion"),
            error=payload.get("error"),
            final_state=payload.get("final_state"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerMessageCodecError(str(exc)) from exc


def _encode_json(payload: Mapping[str, object]) -> str:
    try:
        return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise WorkerMessageCodecError(str(exc)) from exc


def _decode_json_mapping(message: str | bytes) -> Mapping[str, object]:
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    if not isinstance(message, str):
        raise TypeError("message must be str or bytes")
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise WorkerMessageCodecError(str(exc)) from exc
    if not isinstance(payload, Mapping):
        raise WorkerMessageCodecError("worker message must decode to a mapping")
    return payload


def _required_mapping(
    payload: Mapping[str, object],
    field_name: str,
) -> Mapping[str, object]:
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        raise WorkerMessageCodecError(f"{field_name} must be a mapping")
    return value


def _optional_mapping(
    payload: Mapping[str, object],
    field_name: str,
) -> Mapping[str, object] | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise WorkerMessageCodecError(f"{field_name} must be a mapping")
    return value


__all__ = [
    "WorkerMessageCodecError",
    "decode_worker_request_envelope",
    "decode_worker_response_envelope",
    "encode_worker_request_envelope",
    "encode_worker_response_envelope",
]
