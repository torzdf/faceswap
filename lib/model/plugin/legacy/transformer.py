#! /usr/env/bin/python3
""" Handles grouping and processing of Keras weights for migration to torch

The ``KerasWeights`` class holds raw weight tensors and prepares each layer for Torch.
It applies ClipV embedding fusion, scale-less BatchNorm2D handling and SeparableConv2D splits.
MultiHeadAttention is fused before grouping, then the weights are grouped by layer and renamed
to match their Torch counterpart. Two helpers then reorder weights in place.
Dense weights are reordered by ``dense_reorder``, which converts them to channels-first layout.
``pixel_shuffle_reorder`` applies the same channels-first transpose for pixel-shuffle layers.
Grouped weights are then consumed one at a time by downstream migration code
"""
from __future__ import annotations

import logging
import typing as T
from collections import Counter

import numpy as np
import torch

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


def dense_reorder(name: str,
                  weights: dict[T.Literal["weight", "bias"], np.ndarray],
                  dense_reshapes: dict[str, tuple[bool, tuple[int, int, int]]]) -> None:
    """ Reorder a Dense layer's stored weights in place from channels-last to channels-first

    Undoes the space-to-depth or depth-to-space layout applied at runtime so the imported kernel
    matches the Torch weight ordering. The direction is read from ``dense_reshapes``; layers that
    are not present there are logged and skipped

    Parameters
    ----------
    name
        The standardized name of the Dense layer
    weights
        The ``weight`` and optional ``bias`` of a Dense layer imported from Keras, edited in place
    dense_reshapes
        Mapping of layer name to its reshape info as a ``(reshape_in, (height, width, channels))``
        tuple where ``reshape_in`` selects space-to-depth over the input channels
    """
    if name not in dense_reshapes:
        logger.debug("[KerasToTorch] Skipping unmapped Dense layer '%s'", name)
        return

    reshape_in, (height, width, channels) = dense_reshapes[name]
    out, in_ = weights["weight"].shape

    if reshape_in:  # Space to depth on input channel
        shape = (out, height, width, channels)
        trans = (0, 3, 1, 2)  # ch_first
    else:  # Depth to space on output channel
        shape = (height, width, channels, in_)
        trans = (2, 0, 1, 3)  # ch_first

    logger.debug("[KerasToTorch] Converting Dense weights for '%s'. Dense shape: %s, "
                 "Reshape: %s, Transpose: %s",
                 "Space to Depth" if reshape_in else "Depth to Space",
                 weights["weight"].shape,
                 shape,
                 trans)
    weights["weight"] = weights["weight"].reshape(shape).transpose(trans).reshape(out, in_)

    if not reshape_in and weights.get("bias") is not None:
        b_shape = shape[:-1]
        b_trans = trans[:-1]
        logger.debug("[KerasToTorch] Converting Dense bias for output. Bias shape: %s, "
                     "Reshape: %s, Transpose: %s",
                     weights["bias"].shape, b_shape, b_trans)
        weights["bias"] = weights["bias"].reshape(b_shape).transpose(b_trans).reshape(out)


def pixel_shuffle_reorder(weights: dict[T.Literal["weight", "bias"], np.ndarray],
                          scale: int) -> None:
    """ Reorder a layer's stored weights in place to channels-first ahead of the pixel shuffler

    The weight (and optional bias) rows are permuted with an index that maps each output channel
    to its sub-pixel position, so the values line up with what the Torch pixel-shuffle op expects

    Parameters
    ----------
    weights
        The ``weight`` and optional ``bias`` of a layer feeding a pixel shuffler, edited in place
    scale
        The pixel-shuffle scale factor used to derive the number of output channels
    """
    out_channels = weights["weight"].shape[0] // (scale * scale)
    trans = []
    for k_prime in range(scale * scale * out_channels):
        c = k_prime // (scale * scale)
        dh = (k_prime % (scale * scale)) // scale
        dw = k_prime % scale
        k = dh * scale * out_channels + dw * out_channels + c
        trans.append(k)
    logger.debug("[KerasToTorch] Permuting pixel-shuffler input weights of shape %s with "
                 "index of length %s", weights["weight"].shape, len(trans))
    weights["weight"] = weights["weight"][trans]
    if weights.get("bias") is not None:
        logger.debug("[KerasToTorch] Permuting pixel-shuffler input bias of shape %s with "
                     "index of length %s", weights["bias"].shape, len(trans))
        weights["bias"] = weights["bias"][trans]


class KerasWeights:
    """ Group and reshape the Keras weights of a legacy Keras model for migration to Torch

    Holds the raw Keras weight tensors, applies the layer-specific preparation they need (ClipV
    embeddings/projections, scale-less BatchNorm2D, SeparableConv2D splits and MultiHeadAttention
    fusion), then groups them by layer and renames each tensor to match its Torch counterpart.
    Grouped layers are consumed one at a time by ``get_next_weights``

    Parameters
    ----------
    weights
        The Keras weights loaded from the ``.keras`` model's hdf file
    mask_layers
        The names of the weight layers that belong to masks in the decoder
    is_clip
        ``True`` if the model contains a ClipV encoder and so needs special handling
    """
    def __init__(self,
                 weights: dict[str, np.ndarray],
                 mask_layers: list[str],
                 is_clip: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self._mask_layers = mask_layers
        self._is_clip = is_clip
        self._clip_group = ["class_embedding", "positional_embedding", "projection"]

        self._weights = self._prepare_weights(weights)
        """ The Keras weights with any required pre-processing applied """
        self._grouped: dict[str, dict[str, np.ndarray]] = {}
        """ The Keras weights formatted for torch grouped by layer """

    def __len__(self) -> int:
        """ Return the number of ungrouped Keras weight tensors """
        return len(self._weights)

    @property
    def len_grouped(self) -> int:
        """ The number of Keras layers grouped by weight type """
        return len(self._grouped)

    def _prepare_clipv(self, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ Reverse the dims of ClipV embedding and projection layers before downstream use

        Their storage order matches between Torch and Keras but would be reversed downstream, so
        they are transposed here to keep the later processing uniform

        Parameters
        ----------
        weights
            The Keras weights currently being prepared for migration

        Returns
        -------
        The weights with any ClipV embedding and projection dims reversed
        """
        if not self._is_clip:
            return weights

        retval: dict[str, np.ndarray] = {}
        for k, v in weights.items():
            if v.ndim != 2:
                retval[k] = v
                continue

            name = k.rsplit(".", maxsplit=3)[-3]
            if name not in ("positional_embedding", "projection"):
                retval[k] = v
                continue

            logger.debug("[KerasWeights] Reversing dims for clipV layer '%s'", name)
            retval[k] = v.T

        return retval

    @classmethod
    def _prepare_batch_norm(cls, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ Insert a ones-initialized weight into scale-less BatchNorm2D layers

        A ``BatchNorm2D`` built with ``scale=False`` has no gamma tensor in Keras, but Torch
        always expects one, so a weight of all 1.0 is added and the rest re-indexed

        Parameters
        ----------
        weights
            The Keras weights currently being prepared for migration

        Returns
        -------
        The weights with any scale-less BatchNorm2D layers adjusted
        """
        bn_count = Counter(k.rsplit(".", maxsplit=2)[0] for k in weights
                           if "batch_normalization" in k)
        bn_update = {k for k, v in bn_count.items() if v == 3}
        if not bn_update:
            return weights

        retval = {}
        for k, v in weights.items():
            prefix, w_idx = k.rsplit(".", maxsplit=1)
            if prefix.rsplit(".", maxsplit=1)[0] not in bn_update:
                retval[k] = v
                continue

            if w_idx == "0":
                retval[k] = np.ones_like(v)

            retval[f"{k.rsplit('.', maxsplit=1)[0]}.{int(w_idx) + 1}"] = v

        logger.debug("[KerasWeights] Inserted 'weight' to %s BatchNorms for 'scale=False'",
                     len(retval) - len(retval))
        return retval

    @classmethod
    def _prepare_separable_conv(cls, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ Split each SeparableConv2D into two convs and remap the weights in original order

        Torch has no direct equivalent of ``SeparableConv2D``, so each layer is renamed into an
        ``_a`` (depthwise) and ``_b`` (pointwise) pair, reducing the second conv's tensor index

        Parameters
        ----------
        weights
            The Keras weights currently being prepared for migration

        Returns
        -------
        The weights with any SeparableConv2D layers split into a two-conv pair
        """
        sep_weights = {k: v for k, v in weights.items() if "separable_conv2d" in k}
        if not sep_weights:
            return weights

        remap = {}
        for name in sep_weights:
            l_name, v_name, v_idx = name.rsplit(".", maxsplit=2)
            new_l_name = f"{l_name}_a" if v_idx == "0" else f"{l_name}_b"  # no bias 1st conv
            new_v_idx = v_idx if v_idx == "0" else str(int(v_idx) - 1)  # Reduce v_idx 2nd conv
            remap[name] = ".".join([new_l_name, v_name, new_v_idx])

        retval = {remap.get(k, k): v for k, v in weights.items()}
        logger.debug("[KerasWeights] Remapped SeparableConv2D keras weights: %s", remap)
        return retval

    @classmethod
    def _prepare_mha(cls, weights: dict[str, np.ndarray]  # pylint:disable=too-many-locals
                     ) -> dict[str, np.ndarray]:
        """ Fuse the separate Keras query, key and value weights of MultiHeadAttention layers

        Torch stores a single fused ``in_proj`` weight for multi-head attention, so the individual
        Q, K and V kernels and biases are concatenated into one matrix per layer

        Parameters
        ----------
        weights
            The Keras weights currently being prepared for migration

        Returns
        -------
        The weights with any MultiHeadAttention layers fused
        """
        mha_weights = {k: v for k, v in weights.items() if "multi_head_attention" in k}
        if not mha_weights:
            return weights

        mha_layers = {x.rsplit(".", maxsplit=3)[0] for x in mha_weights}
        remap = {}
        for layer in mha_layers:
            in_lyrs = [f"{layer}.{k}.vars." for k in ("query_dense", "key_dense", "value_dense")]
            out_lyr = f"{layer}.output_dense.vars."

            num, dim, feats = mha_weights[f"{out_lyr}0"].shape

            in_krn = np.concatenate([mha_weights[f"{k}0"].reshape(feats, num * dim)
                                     for k in in_lyrs],
                                    axis=1)
            in_bias = np.concatenate([mha_weights[f"{k}1"] for k in in_lyrs], axis=0).reshape(-1)
            out_krn = mha_weights[f"{out_lyr}0"].reshape(num * dim, feats)

            remap[layer] = {"in_proj.vars.0": in_krn,
                            "in_proj.vars.1": in_bias,
                            f"{out_lyr}0"[len(layer) + 1:]: out_krn,
                            f"{out_lyr}1"[len(layer) + 1:]: mha_weights[f"{out_lyr}1"]}

        retval: dict[str, np.ndarray] = {}
        for k, v in weights.items():
            lyr_name = k.rsplit(".", maxsplit=3)[0]
            if lyr_name not in mha_layers:
                retval[k] = v
                continue

            if any(x.startswith(lyr_name) for x in retval):
                continue

            for new_k, new_v in remap[lyr_name].items():
                retval[f"{lyr_name}.{new_k}"] = new_v

        logger.debug("[KerasWeights] Reshaped MultiHeadAttention keras weights: %s",
                     {k: {n: w.shape for n, w in v.items()} for k, v in remap.items()})
        return retval

    def _prepare_weights(self, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ Apply every Keras weight preparation step needed for porting to Torch

        Parameters
        ----------
        weights
            The original Keras weights extracted from the hdf file

        Returns
        -------
        The original Keras weights with all preparation steps applied
        """
        weights = self._prepare_clipv(weights)
        weights = self._prepare_batch_norm(weights)
        weights = self._prepare_separable_conv(weights)
        return self._prepare_mha(weights)

    @classmethod
    def _reshape(cls, layer: str, weights: np.ndarray) -> np.ndarray:
        """ Reshape a single weight tensor to match its PyTorch counterpart.

        Applies the layout transform for the given layer type: squeezing group-norm, expanding
        ConvNeXt layer-scale, and transposing separable/depthwise, 4-D conv and 2-D dense weights
        from Keras channels-last to Torch channels-first

        Parameters
        ----------
        layer
            The type of layer to be processed
        weights
            The weight tensor to be processed

        Returns
        -------
        The weight formatted for PyTorch, or the input unchanged when no transform applies
        """
        retval = weights
        reshaped = False
        if layer == "group_normalization":  # Group norm needs to be squeezed to 1 dim
            retval = weights.squeeze()
            reshaped = True
        elif layer == "layer_scale":  # ConvNeXt layer scale needs dims expanded:
            assert weights.ndim == 1, f"Keras layer_scale shape: {weights.shape}"
            retval = weights[:, None, None]
            reshaped = True
        elif (weights.ndim == 4
                and layer in ("separable_conv2d", "depthwise_conv2d")
                and weights.shape[0] != 1):
            new_shape = (weights.shape[2] * weights.shape[3], 1, *weights.shape[:2])
            retval = weights.transpose(2, 3, 0, 1).reshape(new_shape)
            reshaped = True
        elif weights.ndim == 4:
            retval = weights.transpose(3, 2, 0, 1)
            reshaped = True
        elif weights.ndim == 2:
            retval = weights.transpose(1, 0)
            reshaped = True

        if reshaped:
            logger.debug("[KerasWeights] Reshaped '%s' from %s to %s",
                         layer, weights.shape, retval.shape)

        return retval

    @classmethod
    def _get_layer_type(cls, name: str) -> str:
        """ Obtain the layer type from its name

        Parameters
        ----------
        name
            The name of the layer

        Returns
        -------
        The single canonical type string derived from the layer name
        """
        layer = name.rsplit(".", maxsplit=1)[-1]
        if layer.startswith("separable_conv2d_"):  # strip our manually added sep2d suffix
            layer = layer.rsplit("_", maxsplit=1)[0]
        layer_split = layer.rsplit("_", maxsplit=1)
        if len(layer_split) == 1:
            return layer
        if layer_split[-1].isdigit():
            return layer_split[0]
        return layer

    def group_weights(self) -> None:
        """ Group the prepared Keras weights by layer and rename them to their Torch names

        Each weight is reshaped and assigned the Torch tensor name for its index (weight, bias,
        running_mean or running_var), with per-layer-type overrides for layer-scale and in-
        projection tensors. ClipV layers are elevated to their parent layer
        """
        mapping = {0: "weight", 1: "bias", 2: "running_mean", 3: "running_var"}
        exceptions = {"layer_scale": {0: "layer_scale"},
                      "in_proj": {0: "in_proj_weight", 1: "in_proj_bias"}}
        if self._is_clip:
            exceptions |= {x: {0: x} for x in self._clip_group if x != "projection"}
            exceptions["projection"] = {0: "proj"}

        for lbl, weights in self._weights.items():
            name, w_idx = lbl.rsplit(".", maxsplit=1)
            name = name.rsplit(".", maxsplit=1)[0]  # Strip .vars from the end
            idx = int(w_idx)
            layer_type = self._get_layer_type(name)

            weights = self._reshape(layer_type, weights)
            w_type = (mapping | exceptions.get(layer_type, {}))[idx]  # keras to torch name
            if self._is_clip and layer_type in self._clip_group:
                logger.debug("[KerasWeights] elevating clipV layer '%s' to parent", name)
                name = name.rsplit(".", maxsplit=1)[0]
            self._grouped[name] = self._grouped.get(name, {}) | {w_type: weights}

    def get_next_weights(self,
                         is_mask: bool,
                         weight_key: str,
                         torch_weights: dict[str, torch.Tensor]
                         ) -> tuple[str, dict[str, np.ndarray]]:
        """ Pop and return the next Keras layer whose weights match the given Torch layer

        Scans the grouped weights for the first layer that is (or is not) a mask as requested,
        contains ``weight_key``, has the same number of tensors as ``torch_weights`` and matches
        its shape for that key, then removes and returns it

        Parameters
        ----------
        is_mask
            ``True`` if the requested weights are for a mask layer
        weight_key
            The key in the Torch and Keras weights used for the comparison
        torch_weights
            The Torch weights dict for the layer being matched

        Returns
        -------
        The name of the matched Keras layer and its grouped weight tensors
        """
        name = next(k for k, v in self._grouped.items()
                    if (is_mask and k in self._mask_layers
                        or not is_mask and k not in self._mask_layers)
                    and weight_key in v
                    and len(v) == len(torch_weights)
                    and torch_weights[weight_key].shape == v[weight_key].shape)
        weights = self._grouped.pop(name)
        logger.debug("[KerasWeights] Got %s weights '%s' for shape: %s. Remaining: %s",
                     len(weights), name, weights[weight_key].shape, self.len_grouped)
        return name, weights


__all__ = get_module_objects(__name__)
