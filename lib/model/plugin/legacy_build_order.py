#! /usr/env/bin/python3
""" The model build order for various Keras Application Encoders for porting weights """
from __future__ import annotations
import logging
import typing as T

from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from .legacy import LayerInfo

logger = logging.getLogger(__name__)

_ENC_PREFIX = "layers.functional.layers.functional.layers."


def _inception_resnet_v2_reorder(layers: dict[str, LayerInfo]) -> dict[str, LayerInfo]:
    """ Re-orders imported layer names from Keras Applications InceptionResNetV2 from graph order
    to build order. Fairly straightforward as default naming is used for all problematic layers """
    reorder = ["batch_normalization", "conv2d"]
    current = {"Conv2D": 0, "BatchNormalization": 0}
    backfill: dict[str, dict[int, LayerInfo]] = {"Conv2D": {}, "BatchNormalization": {}}
    retval: dict[str, LayerInfo] = {}

    for k, v in layers.items():
        if not k.startswith(_ENC_PREFIX) or not any(v.layer_name.startswith(x) for x in reorder):
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


def _xception_reorder(layers: dict[str, LayerInfo]) -> dict[str, LayerInfo]:
    """ Re-orders imported layer names from Keras Applications Xception from graph order to build
    order. Skip layers need to be built prior to separable conv within each block """
    reorder_types = {"BatchNormalization", "Conv2D", "SeparableConv2D"}
    backfill: list[LayerInfo] = []
    current_block = 0

    def flush_backfill() -> None:
        while backfill:
            lyr = backfill.pop(0)
            logger.debug("Reordering layer '%s' ('%s')", lyr.weights_name, lyr.layer_name)
            retval[lyr.weights_name] = lyr

    retval: dict[str, LayerInfo] = {}
    for k, v in layers.items():
        if not k.startswith(_ENC_PREFIX) or v.layer_type not in reorder_types:
            logger.debug("Retaining layer '%s' ('%s')", k, v.layer_name)
            retval[k] = v
            continue

        if v.layer_name.startswith("block"):
            idx = int(v.layer_name.split("_")[0].replace("block", ""))
            if current_block != idx:
                flush_backfill()
                current_block += 1
            logger.debug("Holding layer '%s' ('%s')", k, v.layer_name)
            backfill.append(v)
            continue

        logger.debug("Inserting layer '%s' ('%s')", k, v.layer_name)
        retval[k] = v
        if v.layer_name.startswith("batch_normalization"):
            flush_backfill()
            current_block += 1

    flush_backfill()
    assert not backfill, "Not all layers allocated"
    return retval


def reorder_layers(model: str, layers: dict[str, LayerInfo]) -> dict[str, LayerInfo]:
    """ Re-order the layers from Keras graph order to Keras construction order for those models
    which require it for weight porting

    Parameters
    ----------
    model
        The name of the model that the layers belong to
    layers
        The layers of the model that require reordering

    Returns
    -------
    The reordered layers
    """
    functions = {"inception_resnet_v2": _inception_resnet_v2_reorder,
                 "xception": _xception_reorder}
    if model not in functions:
        return layers
    logger.info("Re-ordering layers for '%s'", model)
    return functions[model](layers)


__all__ = get_module_objects(__name__)
