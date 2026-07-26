#! /usr/env/bin/python3
""" The model build order for various FS Models where they differ significantly from config order
for porting weights """
from __future__ import annotations
import logging
import re
import typing as T

from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from .legacy import LayerInfo

logger = logging.getLogger(__name__)

_ENC_PREFIX = "layers.functional.layers.functional.layers."


def _iae_reorder(layers: dict[str, LayerInfo]) -> dict[str, LayerInfo]:
    """ Re-orders the intermediate layers for IAE models. Inters graph in order [both, B, A] but
    build in order [A, B, Both] """
    order = ["layers.input_layer",
             "layers.functional.",
             "layers.functional_2.",
             "layers.functional_1.",
             "layers.functional_3.",
             "layers.concatenate",
             "layers.functional_4."]
    return {k: v
            for model in order
            for k, v in layers.items()
            if k.startswith(model)}


def _inception_reorder(layers: dict[str, LayerInfo]) -> dict[str, LayerInfo]:
    """ Re-orders imported layer names from Keras Applications InceptionResNet models from graph
    order to build order. Fairly straightforward as default naming is used for all problematic
    layers """
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


def _nasnet_reorder(layers: dict[str, LayerInfo]  # pylint:disable=too-many-locals
                    ) -> dict[str, LayerInfo]:
    """ Re-orders imported layer names from Keras Applications NasNet from graph order to build
    order. Some fairly arbitrary re-ordering occurs, but fortunately layer labelling makes this a
    bit easier """
    sep_match = re.compile(r"^(separable)_conv_(\d)(?:_.*?(left|right)(\d))?.*_(\d+)$")
    adj_match = re.compile(r"^(adjust|reduction)_(conv|bn)_.*?(\d+)$")
    reorder_types = {"BatchNormalization", "Conv2D", "SeparableConv2D"}
    type_order = ["adjust", "reduction", "separable"]
    conv_order = ["conv", "bn"]
    side_order = ["left", "right"]

    sort_keys = {}
    for k, v in layers.items():
        if (not k.startswith("layers.functional.layers.functional.layers.")
                or v.layer_type not in reorder_types):
            continue
        lyr = v.layer_name
        block_add = 0 if "stem" in lyr else 3  # Numbering resets after stem
        match = next((m for m in (sep_match.match(lyr), adj_match.match(lyr)) if m), None)
        if not match:
            continue
        groups = match.groups()
        order = [int(groups[-1]) + block_add,   # Ensure blocks ordered
                 type_order.index(groups[0])]   # Ensure adjust, reduce, separable order
        if len(groups) == 3:  # adjust/reduce
            order.extend([conv_order.index(groups[1]), 0, 0])
        else:   # separable
            order.extend([int(groups[3]), side_order.index(groups[2]), int(groups[1])])
        sort_keys[k] = order
    sorted_keys = [k for k, _ in sorted(sort_keys.items(), key=lambda item: item[1])]

    retval: dict[str, LayerInfo] = {}
    while layers:
        k = list(layers)[0]
        if k not in sorted_keys:
            v = layers.pop(k)
            logger.debug("Retaining layer '%s' ('%s')", k, v.layer_name)
            retval[k] = v
            continue
        pos_idx = sorted_keys.index(k)
        for _ in range(pos_idx + 1):
            k = sorted_keys.pop(0)
            v = layers.pop(k)
            logger.debug("Reordering layer '%s' ('%s')", k, v.layer_name)
            retval[k] = v
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
    functions = {"iae": _iae_reorder,
                 "inception_resnet_v2": _inception_reorder,
                 "inception_v3": _inception_reorder,
                 "nasnet_large": _nasnet_reorder,
                 "nasnet_mobile": _nasnet_reorder,
                 "xception": _xception_reorder}

    if model not in functions:
        return layers
    logger.info("Re-ordering layers for '%s'", model)
    return functions[model](layers)


__all__ = get_module_objects(__name__)
