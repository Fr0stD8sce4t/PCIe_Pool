from __future__ import annotations

from dataclasses import fields
from typing import Protocol, runtime_checkable

from ..daemon.client import TurboBusDaemonClient
from ..schema import DaemonResponse, TransferIntent, TransferReceipt


@runtime_checkable
class DaemonIntentClient(Protocol):
    def submit_transfer_intent(self, intent: TransferIntent) -> DaemonResponse:
        ...

    def wait_transfer_receipt(
        self,
        intent_id: str,
        timeout_seconds: float | None = None,
    ) -> DaemonResponse:
        ...


class TurboBusClient:
    """Public daemon-first transfer client."""

    def __init__(
        self,
        daemon: DaemonIntentClient | None = None,
        *,
        socket_path: str | None = None,
    ) -> None:
        if daemon is None and socket_path is None:
            raise ValueError("daemon or socket_path is required")
        if daemon is not None and socket_path is not None:
            raise ValueError("provide daemon or socket_path, not both")
        self._daemon = daemon if daemon is not None else TurboBusDaemonClient(socket_path)

    def submit(self, intent: TransferIntent) -> TransferReceipt:
        return self.submit_transfer_intent(intent)

    def submit_transfer(self, intent: TransferIntent) -> TransferReceipt:
        return self.submit_transfer_intent(intent)

    def submit_transfer_intent(self, intent: TransferIntent) -> TransferReceipt:
        if not isinstance(intent, TransferIntent):
            raise TypeError("intent must be a TransferIntent")
        response = self._daemon.submit_transfer_intent(intent)
        return _receipt_from_response(response, expected_intent_id=intent.intent_id)

    def wait(
        self,
        intent: TransferIntent | str,
        timeout_seconds: float | None = None,
    ) -> TransferReceipt:
        intent_id = intent.intent_id if isinstance(intent, TransferIntent) else str(intent)
        response = self._daemon.wait_transfer_receipt(
            intent_id,
            timeout_seconds=timeout_seconds,
        )
        return _receipt_from_response(response, expected_intent_id=intent_id)

    def wait_transfer_receipt(
        self,
        intent_id: str,
        timeout_seconds: float | None = None,
    ) -> TransferReceipt:
        return self.wait(intent_id, timeout_seconds=timeout_seconds)


def _receipt_from_response(
    response: DaemonResponse,
    *,
    expected_intent_id: str,
) -> TransferReceipt:
    if not isinstance(response, DaemonResponse):
        raise TypeError("daemon response must be a DaemonResponse")
    if not response.ok:
        raise RuntimeError(response.error or "daemon request failed")
    receipt_payload = response.payload.get("receipt")
    if not isinstance(receipt_payload, dict):
        raise ValueError("daemon response missing receipt")
    receipt = _transfer_receipt_from_payload(receipt_payload)
    if receipt.intent_id != str(expected_intent_id):
        raise ValueError("daemon receipt intent_id does not match request")
    return receipt


def _transfer_receipt_from_payload(payload: dict[str, object]) -> TransferReceipt:
    names = {field.name for field in fields(TransferReceipt)}
    unknown = sorted(key for key in payload if key not in names)
    if unknown:
        raise ValueError("daemon receipt contains unknown fields: " + ", ".join(unknown))
    return TransferReceipt(**payload)


__all__ = [
    "DaemonIntentClient",
    "TurboBusClient",
]
