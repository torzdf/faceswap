#!/usr/bin/env python3
""" The default options for the faceswap Dfl_SAE Model plugin.

Defaults files should be named `<plugin_name>_defaults.py`

Any qualifying items placed into this file will automatically get added to the relevant config
.ini files within the faceswap/config folder and added to the relevant GUI settings page.

The following variable should be defined:

    Parameters
    ----------
    HELPTEXT: str
        A string describing what this plugin does

Further plugin configuration options are assigned using:
>>> <config_item> = ConfigItem(...)

where <config_item> is the name of the configuration option to be added (lower-case, alpha-numeric
+ underscore only) and ConfigItem(...) is the [`~lib.config.objects.ConfigItem`] data for the
option.

See the docstring/ReadtheDocs documentation required parameters for the ConfigItem object.
Items will be grouped together as per their `group` parameter, but otherwise will be processed in
the order that they are added to this module.
from lib.config import ConfigItem
"""
# pylint:disable=duplicate-code
from lib.config import ConfigItem


HELPTEXT = "Dfaker Model (Adapted from https://github.com/dfaker/df)"


output_size = ConfigItem(
    datatype=int,
    default=128,
    group="size",
    info="Resolution (in pixels) of the output image to generate on.\n"
         "BE AWARE Larger resolution will dramatically increase VRAM requirements.\n"
         "Must be 128 or 256.",
    rounding=128,
    min_max=(128, 256),
    fixed=True)

freeze_layers = ConfigItem(
    datatype=list,
    default=["encoder"],
    group="weights",
    info="If the command line option 'freeze-weights' is enabled, then the layers indicated "
         "here will be frozen the next time the model starts up.",
    choices=["encoder", "decoders.0", "decoders.1"],
    fixed=False)

load_layers = ConfigItem(
    datatype=list,
    default=["encoder"],
    group="weights",
    info="If the command line option 'load-weights' is populated, then the layers indicated "
         "here will be loaded from the given weights file if starting a new model.",
    choices=["encoder", "decoders.0", "decoders.1"],
    fixed=True)
