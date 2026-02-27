#!/usr/bin/env python3
""" Default configurations for extract """

import gettext
import logging
import os

from lib.config import FaceswapConfig
from lib.config import ConfigItem

# LOCALES
_LANG = gettext.translation("plugins.extract.extract_config", localedir="locales", fallback=True)
_ = _LANG.gettext

logger = logging.getLogger(__name__)


class _Config(FaceswapConfig):
    """ Config File for Extraction """

    def set_defaults(self, helptext="") -> None:
        """ Set the default values for config """
        super().set_defaults(helptext=_("Options that apply to all extraction plugins"))
        self._defaults_from_plugin(os.path.dirname(__file__))


aligner_min_scale = ConfigItem(
    datatype=float,
    default=0.03,
    group=_("align"),
    info=_(
        "Filters out faces below this size. This is a multiplier of the minimum dimension of the "
        "frame (i.e. 1280x720 = 720). If the original face extract box is smaller than the "
        "minimum dimension times this multiplier, it is considered a false positive and "
        "discarded. Faces which are found to be unusually smaller than the frame tend to be "
        "misaligned images, except in extreme long-shots. These can be usually be safely "
        "discarded."),
    min_max=(0.0, 1.0),
    rounding=2)


aligner_max_scale = ConfigItem(
    datatype=float,
    default=4.00,
    group=_("align"),
    info=_(
        "Filters out faces above this size. This is a multiplier of the minimum dimension of the "
        "frame (i.e. 1280x720 = 720). If the original face extract box is larger than the minimum "
        "dimension times this multiplier, it is considered a false positive and discarded. Faces "
        "which are found to be unusually larger than the frame tend to be misaligned images "
        "except in extreme close-ups. These can be usually be safely discarded."),
    min_max=(0.0, 10.0),
    rounding=2)


aligner_distance = ConfigItem(
    datatype=float,
    default=40.0,
    group=_("align"),
    info=_(
        "Filters out faces who's landmarks are above this distance from an 'average' face. Values "
        "above 15 tend to be fairly safe. Values above 10 will remove more false positives, but "
        "may also filter out some faces at extreme angles."),
    min_max=(0.0, 45.0),
    rounding=1)


aligner_roll = ConfigItem(
    datatype=float,
    default=0.0,
    group=_("align"),
    info=_(
        "Filters out faces who's calculated roll is greater than zero +/- this value in degrees. "
        "Aligned faces should have a roll value close to zero. Values that are a significant "
        "distance from 0 degrees tend to be misaligned images. These can usually be safely "
        "disgarded."),
    min_max=(0.0, 90.0),
    rounding=1)


aligner_features = ConfigItem(
    datatype=bool,
    default=True,
    group=_("align"),
    info=_(
        "Filters out faces where the lowest point of the aligned face's eye or eyebrow is lower "
        "than the highest point of the aligned face's mouth. Any faces where this occurs are "
        "misaligned and can be safely disgarded."))


mask_storage_size = ConfigItem(
    datatype=int,
    default=128,
    group=_("mask"),
    info=_("The size to store masks at. Set to 0 to store at the mask model's output size."),
    min_max=(0, 1028),
    rounding=64)


# pylint:disable=duplicate-code
_IS_LOADED: bool = False


def load_config(config_file: str | None = None) -> None:
    """ Load the Extraction configuration .ini file

    Parameters
    ----------
    config_file : str | None, optional
        Path to a custom .ini configuration file to load. Default: ``None`` (use default
        configuration file)
    """
    global _IS_LOADED  # pylint:disable=global-statement
    if not _IS_LOADED:
        _Config(configfile=config_file)
    _IS_LOADED = True
