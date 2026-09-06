#! /usr/env/bin/python3
""" Analyse the topology of a Keras model to port its weights into Faceswap's Torch backend

This module inspects the layer graph of a legacy Keras model and locates the structures that change
when moving to channel-first ordering: the convolutions feeding pixel shuffler layers, the Dense
layers needing reshape around flattens and reshapes, and the output chains that split between image
and mask. Downstream migration code uses these results to select and reshape weights without clash.
"""
from __future__ import annotations

import logging

from lib.utils import get_module_objects

from .keras import LayerInfo

logger = logging.getLogger(__name__)


def get_pixel_shuffler_convs(layers: dict[str, LayerInfo]) -> dict[str, int]:
    """ Obtain a list of convolutions that lead into pixel shuffler layers for channel re-
    ordering

    When porting to Faceswap's channel-first backend, each conv feeding a pixel shuffler must be
    scaled by the ratio of its output and input channels. Walking back from every pixel shuffler
    yields those ratios so the weights can be reshaped without mismatch.

    Parameters
    ----------
    layers
        The list of standardized layer names within the keras model mapped to their inbound
    nodes

    Returns
    -------
    dict of convolution names that lead into pixel shuffler layers to the scale of the pixel
    shuffler layer
    """
    retval: dict[str, int] = {}
    for layer, info in layers.items():
        if not layer.rsplit(".", maxsplit=1)[-1].startswith("pixel_shuffler"):
            continue

        assert len(info.input_layers) == 1  # FS never has more than 1 input into a PS
        in_ = info.input_layers[0]
        in_size = info.input_shapes[0][0]
        in_conv = None

        while True:
            if in_.rsplit(".", maxsplit=1)[-1].startswith("conv2d"):
                logger.debug("Collected conv '%s' for pixel shuffler '%s'", in_, layer)
                in_conv = in_
                break

            logger.debug("Skipping non-conv '%s' for pixel shuffler '%s'", in_, layer)
            next_in = layers[in_].input_layers
            assert len(next_in) == 1
            in_ = next_in[0]

        assert in_conv is not None
        out_size = None

        for info in layers.values():
            if layer in info.input_layers:
                out_sizes = set(x[0] for x in info.input_shapes)
                assert len(out_sizes) == 1
                out_size = list(out_sizes)[0]
                break
        assert out_size is not None
        retval[in_conv] = out_size // in_size

    logger.debug("Pixel Shuffler convs and scales: %s", retval)
    return retval


def get_dense_reshapes(layers: dict[str, LayerInfo]
                       ) -> dict[str, tuple[bool, tuple[int, int, int]]]:
    """ Obtain the Dense layers that either follow a flatten or precede a reshape that require
    their weights reshaped for channel first ordering

    Channel-last Keras weights mismatch channel-first Torch layouts, so any adjacent Dense layer
    needs reshaping. Tracing these connections finds each target and whether its input or output
    channels need reshaping, plus the required tensor shape.

    Parameters
    ----------
    layers
        The standardized layer names with their inbound nodes and shapes

    Returns
    -------
    dict of dense layer names to tuple of (``True`` to reshape in_channels, ``False`` to
    reshape out_channels, (shape of input/output tensor))
    """
    retval: dict[str, tuple[bool, tuple[int, int, int]]] = {}
    reshapes = {lyr: shp
                for inf in layers.values()
                for lyr, shp in zip(inf.input_layers, inf.input_shapes)
                if lyr.rsplit(".", maxsplit=1)[-1].startswith("reshape")}

    for layer, info in layers.items():
        name = layer.rsplit(".", maxsplit=1)[-1]
        if not name.startswith(("dense", "reshape")):
            continue
        is_dense = name.startswith("dense")

        while True:
            assert len(info.input_layers) == 1  # FS never has more than 2 inputs into a Dense
            in_ = info.input_layers[0]
            if in_.rsplit(".", maxsplit=1)[-1].startswith("dropout"):  # move up from dropout
                logger.debug("Getting input to '%s' for layer '%s'", in_, layer)
                info = layers[in_]
                continue
            break

        # Reshape in
        if is_dense and not in_.rsplit(".", maxsplit=1)[-1].startswith("flatten"):
            logger.debug("Skipping in channel dense '%s' with input '%s'", layer, in_)
            continue

        if is_dense:
            assert len(layers[in_].input_layers) == 1
            shape = tuple(layers[in_].input_shapes[0])
            assert len(shape) == 3  # Must be H, W, C
            retval[layer] = (True, shape)
            logger.debug("Collected in channel reshape for '%s': %s", layer, shape)
            continue

        # Reshape out
        if not in_.rsplit(".", maxsplit=1)[-1].startswith("dense"):
            logger.debug("Skipping reshape '%s' with input '%s'", layer, in_)
            continue

        shape = tuple(reshapes[layer])
        assert len(shape) == 3  # Must be H, W, C
        retval[in_] = (False, shape)
        logger.debug("Collected out channel reshape for '%s': %s", in_, shape)

    logger.debug("Dense reshape weights: %s", retval)
    return retval


def _recurse_from_layer(layers: dict[str, list[str]],
                        current: list[str],
                        sub_model: str,
                        seen: set[str] | None = None) -> list[str]:
    """ From the given layers recurse backwards through all layers to the beginning of the sub-
    model

    Parameters
    ----------
    layers
        The full dict of standardized layer names with their inbound nodes
    current
        The list of layers to recurse backwards from
    sub_model
        The keras sub-model to collect the layers from
    seen
        layers that have already been collected. Default: ``None`` (first iteration)

    Returns
    -------
    list of unique layers that feed into the given layers
    """
    seen = set() if seen is None else seen
    retval: list[str] = []

    for lyr in current:
        if ".".join(lyr.split(".", maxsplit=2)[:2]) != sub_model:
            logger.debug("Exited sub-model '%s' at layer '%s'", sub_model, lyr)
            continue

        if lyr in seen:
            continue

        seen.add(lyr)
        retval.append(lyr)
        retval += _recurse_from_layer(layers, layers[lyr], sub_model, seen)

    return retval


def get_mask_layers(layers: dict[str, list[str]], weights: dict[str, tuple[int, ...]]
                    ) -> list[str]:
    """ Identify keras layer names that are part of the mask output chain.

    Keras interleaves creation of upscales between main image and mask when learn_mask is
    enabled, whilst Torch creates image upscales first then mask. This can cause shape clash
    when selecting weights to port.

    Parameters
    ----------
    layers
        The standardized layer names with their inbound nodes
    weights
        The standardized layer names to weight shapes for any conv layers in the model

    Returns
    -------
    list of layer names that are part of the mask output chain, if any
    """
    if list(weights.values())[-1][-1] != 1:  # Mask will always be last output in FS
        logger.debug("No mask output. Returning empty list")
        return []

    mod_msk_out = {".".join(k.split(".", maxsplit=2)[:2]): v  # Overwrites outputs at mod level
                   for k, v in weights.items() if v[-1] == 1}
    msk_outputs = [k.rsplit(".", maxsplit=2)[0] for k, v in weights.items()
                   if mod_msk_out.get(".".join(k.split(".", maxsplit=2)[:2])) == v]
    mod_img_out = {mod: v  # Overwrites outputs at model level
                   for k, v in weights.items() if v[-1] == 3
                   if (mod := ".".join(k.split(".", maxsplit=2)[:2])) in mod_msk_out}
    img_outputs = [k.rsplit(".", maxsplit=2)[0] for k, v in weights.items()
                   if mod_img_out.get(".".join(k.split(".", maxsplit=2)[:2])) == v]

    logger.debug("Selecting image output layers %s, mask output layers %s",
                 img_outputs, msk_outputs)

    img_layers = [y for x in img_outputs
                  for y in _recurse_from_layer(layers,
                                               [x],
                                               ".".join(x.split(".", maxsplit=2)[:2]))]
    msk_layers = [y for x in msk_outputs
                  for y in _recurse_from_layer(layers,
                                               [x],
                                               ".".join(x.split(".", maxsplit=2)[:2]))]

    retval = [x for x in msk_layers if x not in img_layers]
    logger.debug("Collected mask path layers: %s", retval)
    return retval


__all__ = get_module_objects(__name__)
