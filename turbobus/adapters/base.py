from __future__ import annotations

from typing import Protocol


class FrameworkAdapter(Protocol):
    """Marker protocol for framework-facing TurboBus adapters."""


__all__ = ["FrameworkAdapter"]
