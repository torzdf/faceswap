#! /usr/env/bin/python3
""" The model build order for various Keras Application Encoders for porting weights """
from __future__ import annotations
import logging
import typing as T

from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from .legacy import LayerInfo

logger = logging.getLogger(__name__)


def inception_resnet_v2_reorder(layers: dict[str, LayerInfo]) -> dict[str, LayerInfo]:
    """ Re-orders imported layer names from Keras Applications InceptionResNetV2 from graph order
    to build order. Fairly straightforward as default naming is used for all problematic layers """
    retval: dict[str, LayerInfo] = {}
    model_prefix = "layers.functional.layers.functional.layers."
    reorder_layers = ["batch_normalization", "conv2d"]
    current = {"Conv2D": 0, "BatchNormalization": 0}
    backfill: dict[str, dict[int, LayerInfo]] = {"Conv2D": {}, "BatchNormalization": {}}
    for k, v in layers.items():
        if not k.startswith(model_prefix) or not any(v.layer_name.startswith(x)
                                                     for x in reorder_layers):
            logger.debug("Retaining layer '%s' ('%s')", k, v.layer_name)
            retval[k] = v
            continue
        while current[v.layer_type] in backfill[v.layer_type]:
            lyr = backfill[v.layer_type].pop(current[v.layer_type])
            logger.debug("Re-ordering layer '%s' ('%s')", lyr.weights_name, lyr.layer_name)
            retval[lyr.weights_name] = lyr
            current[v.layer_type] += 1

        str_idx = v.layer_name.rsplit("_", maxsplit=1)[-1]
        idx = int(str_idx) if str_idx.isdigit() else 0

        if idx == current[v.layer_type]:
            retval[k] = v
            current[v.layer_type] += 1
            logger.debug("Inserting layer '%s' ('%s')", k, v.layer_name)
            continue
        logger.debug("Holding layer '%s' ('%s')", k, v.layer_name)
        backfill[v.layer_type][idx] = v

    assert len(retval) == len(layers), "Not all layers handled"
    return retval


__all__ = get_module_objects(__name__)
