from __future__ import annotations

import sys as _sys

from .adapters import model_loading as _impl

_sys.modules[__name__] = _impl
