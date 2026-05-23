from __future__ import annotations

import math
from dataclasses import dataclass

from .schema import AutoTransferDecision, TransferMode


@dataclass(frozen=True)
class AutoTransferSelector:
    min_chunks_for_relay: int = 2
    min_pool_bytes: int = 12 * 1024 * 1024
    relay_min_effective_bw_gbps: float = 0.0
    relay_min_direct_ratio: float = 0.0
    min_pool_speedup: float = 1.15
    min_relay_speedup: float = 1.05

    def choose(
        self,
        profile,
        request_bytes: int,
        chunk_bytes: int,
        request_chunks: int | None = None,
        direction: str = "h2d",
    ) -> AutoTransferDecision:
        request_bytes = max(0, int(request_bytes))
        chunk_bytes = max(1, int(chunk_bytes))
        if request_chunks is None:
            request_chunks = max(1, math.ceil(request_bytes / chunk_bytes)) if request_bytes else 0
        else:
            request_chunks = max(0, int(request_chunks))
        direct_attr = "direct_h2d_bw_gbps" if direction == "h2d" else "direct_d2h_bw_gbps"
        direct_bw = max(0.0, float(getattr(profile, direct_attr, 0.0) or 0.0))
        if direction != "h2d" and direct_bw <= 0.0:
            direct_bw = max(0.0, float(getattr(profile, "direct_h2d_bw_gbps", 0.0) or 0.0))
        eligible_relays = self._eligible_relays(profile, direct_bw, direction)
        relay_attr = "effective_bw_gbps" if direction == "h2d" else "effective_d2h_bw_gbps"
        relay_bw = 0.0
        for relay in eligible_relays:
            effective_bw = max(0.0, float(getattr(relay, relay_attr, 0.0) or 0.0))
            if direction != "h2d" and effective_bw <= 0.0:
                effective_bw = max(0.0, float(getattr(relay, "effective_bw_gbps", 0.0) or 0.0))
            relay_bw += effective_bw

        if request_bytes == 0:
            return self._decision(
                TransferMode.AUTO,
                TransferMode.DIRECT,
                request_bytes,
                request_chunks,
                direct_bw,
                relay_bw,
                eligible_relays,
                f"{direction} request has no bytes",
            )
        if request_chunks < self.min_chunks_for_relay:
            return self._decision(
                TransferMode.AUTO,
                TransferMode.DIRECT,
                request_bytes,
                request_chunks,
                direct_bw,
                relay_bw,
                eligible_relays,
                f"{direction} request has only {request_chunks} chunk(s)",
            )
        if direct_bw <= 0.0 and relay_bw > 0.0:
            return self._decision(
                TransferMode.AUTO,
                TransferMode.RELAY,
                request_bytes,
                request_chunks,
                direct_bw,
                relay_bw,
                eligible_relays,
                f"{direction} direct bandwidth is unavailable",
            )
        if relay_bw <= 0.0:
            return self._decision(
                TransferMode.AUTO,
                TransferMode.DIRECT,
                request_bytes,
                request_chunks,
                direct_bw,
                relay_bw,
                eligible_relays,
                f"{direction} has no eligible relay paths",
            )

        direct_ms = self._transfer_ms(request_bytes, direct_bw)
        relay_ms = self._transfer_ms(request_bytes, relay_bw)
        pool_ms = self._transfer_ms(request_bytes, direct_bw + relay_bw)
        best_single_ms = min(direct_ms, relay_ms)
        pool_speedup = best_single_ms / pool_ms if pool_ms > 0.0 else 0.0
        relay_speedup = direct_ms / relay_ms if direct_ms > 0.0 and relay_ms > 0.0 else 0.0

        if request_bytes >= self.min_pool_bytes and pool_speedup >= self.min_pool_speedup:
            return self._decision(
                TransferMode.AUTO,
                TransferMode.POOL,
                request_bytes,
                request_chunks,
                direct_bw,
                relay_bw,
                eligible_relays,
                f"pool speedup {pool_speedup:.3f} >= {self.min_pool_speedup:.3f}",
            )
        if relay_speedup >= self.min_relay_speedup and relay_ms < direct_ms:
            return self._decision(
                TransferMode.AUTO,
                TransferMode.RELAY,
                request_bytes,
                request_chunks,
                direct_bw,
                relay_bw,
                eligible_relays,
                f"relay speedup {relay_speedup:.3f} >= {self.min_relay_speedup:.3f}",
            )
        return self._decision(
            TransferMode.AUTO,
            TransferMode.DIRECT,
            request_bytes,
            request_chunks,
            direct_bw,
            relay_bw,
            eligible_relays,
            "direct is the best single path",
        )

    @staticmethod
    def _transfer_ms(bytes_: int, bandwidth_gbps: float) -> float:
        if bytes_ <= 0 or bandwidth_gbps <= 0.0:
            return 0.0
        return (float(bytes_) / (bandwidth_gbps * 1e9)) * 1000.0

    def _eligible_relays(self, profile, direct_bw: float, direction: str):
        relay_attr = "effective_bw_gbps" if direction == "h2d" else "effective_d2h_bw_gbps"
        relays = []
        for relay in getattr(profile, "relays", []) or []:
            effective_bw = max(0.0, float(getattr(relay, relay_attr, 0.0) or 0.0))
            if direction != "h2d" and effective_bw <= 0.0:
                effective_bw = max(0.0, float(getattr(relay, "effective_bw_gbps", 0.0) or 0.0))
            if not getattr(relay, "p2p_enabled", False) or effective_bw <= 0.0:
                continue
            if effective_bw < self.relay_min_effective_bw_gbps:
                continue
            if (
                direct_bw > 0.0
                and self.relay_min_direct_ratio > 0.0
                and effective_bw < direct_bw * self.relay_min_direct_ratio
            ):
                continue
            relays.append(relay)
        return relays

    @staticmethod
    def _decision(
        requested_mode: TransferMode,
        resolved_mode: TransferMode,
        request_bytes: int,
        request_chunks: int,
        direct_bw: float,
        relay_bw: float,
        eligible_relays,
        reason: str,
    ) -> AutoTransferDecision:
        return AutoTransferDecision(
            requested_mode=requested_mode,
            resolved_mode=resolved_mode,
            request_bytes=int(request_bytes),
            request_chunks=int(request_chunks),
            direct_h2d_bw_gbps=float(direct_bw),
            relay_effective_bw_gbps=float(relay_bw),
            eligible_relay_devices=tuple(
                int(getattr(relay, "relay_device", -1)) for relay in eligible_relays
            ),
            reason=str(reason),
        )


__all__ = ["AutoTransferDecision", "AutoTransferSelector", "TransferMode"]
