#!/usr/bin/env python3
""" Interactive display backends for training previews

The *display* side of FaceSwap's preview system: renders predicted-face previews produced during
and after training into a window, decoupled from the unit that generates them. At import time it
exposes these names

- ``Preview``: runtime-selected backend: :class:`PreviewTk` (embedded Tkinter GUI) unless importing
it fails, in which case :class:`PreviewCV` (OpenCV fallback) is used instead
- :class:`PreviewBuffer`: thread-safe store of named preview images consumed by a display backend
- ``TriggerType``: type alias mapping shared trigger events onto threading events
"""
from __future__ import annotations
import typing as T

from .preview import PreviewBuffer, TriggerType

if T.TYPE_CHECKING:
    from .preview import PreviewBase
    Preview: type[PreviewBase]

try:
    from .preview_tk import PreviewTk as Preview
except ImportError:
    from .preview_cv import PreviewCV as Preview
