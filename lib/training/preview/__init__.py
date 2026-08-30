#!/usr/bin/env python3
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
