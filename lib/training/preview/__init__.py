#!/usr/bin/env python3
""" Responsible for displaying training previews in a pop-up Windows """
from __future__ import annotations
import typing as T

from .preview_cv import PreviewBuffer, TriggerType

if T.TYPE_CHECKING:
    from .preview_cv import PreviewBase
    Preview: type[PreviewBase]

try:
    from .preview_tk import PreviewTk as Preview
except ImportError:
    from .preview_cv import PreviewCV as Preview
