#! /usr/env/bin/python3
""" Handles loading information from legacy .keras models """
from __future__ import annotations

import logging
import typing as T
from collections import Counter

import numpy as np
import torch

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .keras_model import KerasModel

if T.TYPE_CHECKING:
    from lib.model.plugin.handler import FaceswapModel
    from .keras_model import LayerInfo

logger = logging.getLogger(__name__)


def _get_pixel_shuffler_convs(layers: dict[str, LayerInfo]) -> dict[str, int]:
    """ Obtain a list of convolutions that lead into pixel shuffler layers for channel re-
    ordering

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


def _get_dense_reshapes(layers: dict[str, LayerInfo]
                        ) -> dict[str, tuple[bool, tuple[int, int, int]]]:
    """ Obtain the Dense layers that either follow a flatten or precede a reshape that require
    their weights reshaped for channel first ordering

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
    sub-model
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


def _get_mask_layers(layers: dict[str, list[str]], weights: dict[str, tuple[int, ...]]
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


class KerasWeights:
    """ Handles grouping and processing of Keras weights for migration to torch

    Parameters
    ----------
    weights
        The Keras weights as loaded from the hdf file stored in a .keras model
    mask_layers
        List of weight layer names that are for masks in the decoder
    is_clip
        ``True`` if the model contains a ClipV encoder, so requires special handling
    """
    def __init__(self, weights: dict[str, np.ndarray], mask_layers: list[str], is_clip: bool
                 ) -> None:
        logger.debug(parse_class_init(locals()))
        self._mask_layers = mask_layers
        self._is_clip = is_clip
        self._clip_group = ["class_embedding", "positional_embedding", "projection"]

        self._weights = self._prepare_weights(weights)
        """ The Keras weights with any required pre-processing applied """
        self._grouped: dict[str, dict[str, np.ndarray]] = {}
        """ The Keras weights formatted for torch grouped by layer """

    def __len__(self) -> int:
        """ The length of the ungrouped Keras weights """
        return len(self._weights)

    @property
    def len_grouped(self) -> int:
        """ The length of the grouped Keras weights """
        return len(self._grouped)

    def _prepare_clipv(self, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ ClipV encoder embedding + projection orders match between Torch + Keras but will get
        reversed downstream. To make the code more maintanble these are reversed here so no special
        processing is required later

        Parameters
        ----------
        weights
            The keras weights that are currently being prepared for migration

        Returns
        -------
        The weights with any processing applied for ClipV embedding + Projection layers
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
        """ BatchNorm2D with `scale=False` requires weights to be inserted initialized to 1.0 as
        this configuration option does not exist in torch so must be handled explicitly

        Parameters
        ----------
        weights
            The keras weights that are currently being prepared for migration

        Returns
        -------
        The weights with any processing applied for BatchNorm2D layers
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
        """ SeparableConv2D needs to be split into 2x convs and replaced in the weights file in
        original order

        Parameters
        ----------
        weights
            The keras weights that are currently being prepared for migration

        Returns
        -------
        The weights with any processing applied for SeparableConv2D layers
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
        """ MultiHeadAttention needs the separate Keras Q, K, V weights to be fused into a single
        weights matrix

        Parameters
        ----------
        weights
            The keras weights that are currently being prepared for migration

        Returns
        -------
        The weights with any processing applied for MultiHeadAttention layers
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
        """ Some Keras weights need preparation for porting. Specifically: ClipV
        embedding/projection layers. BatchNorm2D layers with scale=False. SeparableConv2D layers.
        MultiHeadAttention layers

        Parameters
        ----------
        weights
            The original keras weights extracted from the hdf file

        Returns
        -------
        The original keras weights with any processing applied.
        """
        weights = self._prepare_clipv(weights)
        weights = self._prepare_batch_norm(weights)
        weights = self._prepare_separable_conv(weights)
        return self._prepare_mha(weights)

    @classmethod
    def _reshape(cls, layer: str, weights: np.ndarray) -> np.ndarray:
        """ Reshape qualifying weights to match their PyTorch counterpart

        Parameters
        ----------
        layer
            The type of layer being processed
        weights
            The weight to be processed

        Returns
        -------
        The weights formatted for PyTorch
        """
        retval = weights
        reshaped = False
        if (weights.ndim == 4
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
        elif layer == "layer_scale":  # ConvNeXt layer scale needs dims expanded:
            assert weights.ndim == 1, f"Keras layer_scale shape: {weights.shape}"
            retval = weights[:, None, None]
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
            The name of the layer to obtain the type from

        Returns
            The type of the layer (the name with any following indices removed)
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
        """ Groups the Keras weights by layer, renames the weight names and weight shapes to match
        their torch counterparts """
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
        """ Obtain the next qualifying keras layer name and corresponding weights for the given
        weights shape

        Parameters
        ----------
        is_mask
            ``True`` if the requested weights are for a mask layer
        weight_key
            The key in the Torch and Keras weights to use for comparison
        torch_weights
            The torch weights dict for the layer
        """
        name = next(k for k, v in self._grouped.items()
                    if (is_mask and k in self._mask_layers
                        or not is_mask and k not in self._mask_layers)
                    and weight_key in v
                    and len(v) == len(torch_weights)
                    and torch_weights[weight_key].shape == v[weight_key].shape)
        weights = self._grouped.pop(name)
        logger.debug("[KerasWeights] Got %s weights '%s' for shape: %s. Remaining: %s",
                     len(weights), name, torch_weights[weight_key].shape, self.len_grouped)
        return name, weights


class KerasToTorch:
    """ Port weights from a keras trained Faceswap model to pyTorch format

    Parameters
    ----------
    torch_model
        The uninitialized Torch model plugin
    keras_file
        The fullpath to the keras model file
    """
    _remap_weight = ["layer_scale",  # ConvNext re-label in torch
                     "class_embedding", "positional_embedding", "proj",  # Clip re-label in torch
                     "in_proj_weight"]  # Clip re-label in torch
    _remap_bias = ["in_proj_bias"]  # Clip re-label in torch

    def __init__(self, torch_model: FaceswapModel, keras_file: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._keras = KerasModel(keras_file)
        self._torch = torch_model

        self._state_dict: dict[T.Literal["model", "state", "optimizer", "version"],
                               float | dict[str, T.Any]] = {}
        self._pixel_shuffler_convs = _get_pixel_shuffler_convs(self._keras.layers)
        self._dense_reshapes = _get_dense_reshapes(self._keras.layers)
        self._state = self._get_state()

    def _get_state(self) -> dict[str, T.Any]:
        """ Obtain the legacy state dict removing any keys that may break downstream dataclasses
        and updating any legacy items to be compatible with state version 2.0

        Returns
        -------
        The keras state file with legacy items fixed for import
        """
        retval = {k: "none" if v is None else v  # Nonetype used to be allowed
                  for k, v in self._keras.state.items()
                  if k not in ("mixed_precision_layers",  # Dropped
                               "sessions")}  # Handled later

        legacy_defaults = {  # If these do not exist then state file is v. old. Set sane defaults
            "centering": "legacy",
            "coverage": 62.5,
            "mask_loss_function": "mse",
            "optimizer": "adam"
        }
        for key, val in legacy_defaults.items():
            retval[key] = retval.get(key, val)

        if isinstance(retval.get("lowest_avg_loss"), dict):  # Loss used to be stored per side
            lowest_avg_loss = sum(T.cast(dict[str, float], retval["lowest_avg_loss"]).values())
            logger.debug("[KerasToTorch] Collating legacy lowest_avg_loss from %s to %s",
                         retval["lowest_avg_loss"], lowest_avg_loss)
            retval["lowest_avg_loss"] = lowest_avg_loss

        # Following keys no longer exist or map to new keys
        priors = ["dssim_loss", "mask_type", "mask_type", "l2_reg_term", "clipnorm", "autoclip"]
        new_items = ["loss_function", "learn_mask", "mask_type", "loss_function_2",
                     "gradient_clipping", "clipping"]
        for old, new in zip(priors, new_items):
            if old not in retval:
                logger.debug("[KerasToTorch] Legacy item '%s' not in state config. Skipping", old)
                continue

            if old == "dssim_loss":  # dssim_loss > loss_function
                retval[new] = "ssim" if retval[old] else "mae"
                del retval[old]
                logger.debug("[KerasToTorch] Updated state config from legacy dssim format. New"
                             "config loss function: '%s'", retval[new])
                continue

            if (old == "mask_type" and  # Replace removed masks with most similar equivalent
                    new == "mask_type" and
                    retval[old] in ("facehull", "dfl_full")):
                old_mask = retval[old]
                retval[new] = "components"
                logger.debug("[KerasToTorch] Updated 'mask_type' from '%s' to '%s' for this model",
                             old_mask, retval[new])

            if old == "l2_reg_term":  # Replace l2_reg_term with loss_2 func and update  weight
                retval[new] = "mse"
                retval["loss_weight_2"] = retval[old]
                del retval[old]
                logger.info("[KerasToTorch] Updated state config from legacy 'l2_reg_term' to "
                            "'loss_function_2'")

            if old == "clipnorm":  # Replace clipnorm with correct grad clip type and value
                retval[new] = "norm"
                del retval[old]
                logger.info("[KerasToTorch] Updated state config from legacy '%s' to  '%s: %s'",
                            old, new, old)

            if old == "autoclip":  # Replace autoclip with correct gradient clipping type
                retval[new] = old
                del retval[old]
                logger.info("[KerasToTorch] Updated state config from legacy '%s' to '%s: %s'",
                            old, new, old)

        retval["sessions"] = {int(i): {"batch_size" if k == "batchsize" else k: v
                                       for k, v in s.items() if k != "no_logs"}
                              for i, s in self._keras.state["sessions"].items()}
        retval["is_legacy"] = True
        retval["version"] = 2.0
        logger.debug("[KerasToTorch] Cleaned state: %s", retval)
        return retval

    def _group_weights(self, weights: dict[str, torch.Tensor]
                       ) -> dict[str, dict[str, torch.Tensor]]:
        """ Group the torch weights by layer

        Parameters
        ----------
        weights
            The weights to group, with separate items for weights and biases

        Returns
        -------
        Each layer of the model with a dictionary containing it's weights and biases
        """
        retval = {}
        for lbl, weight in weights.items():
            name, w_type = lbl.rsplit(".", maxsplit=1)
            retval[name] = retval.get(name, {}) | {w_type: weight}
        return retval

    def _dense_reorder(self,
                       name: str,
                       weights: dict[T.Literal["weight", "bias"], np.ndarray]) -> None:
        """ Shuffle the order that weights are stored for either the in-channels or out-channels
        for Dense operations from channels last to channels first in place.

        This handles the bottleneck for most existing Faceswap models fairly effectively

        Parameters
        ----------
        name
            The standardized name of the dense layer
        weights
            The weights and bias for a Dense layer being imported from Keras
        """
        if name not in self._dense_reshapes:
            logger.debug("[KerasToTorch] Skipping unmapped Dense layer '%s'", name)
            return

        reshape_in, (height, width, channels) = self._dense_reshapes[name]
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

    @classmethod
    def _pixel_shuffle_reorder(cls,
                               weights: dict[T.Literal["weight", "bias"], np.ndarray],
                               scale: int) -> None:
        """ Shuffle the order that weights are stored to channels first prior to feeding the pixel
        shuffler

        Parameters
        ----------
        weights
            The weights and bias for a conv layer being imported from Keras
        scale
            The scale of the pixel shuffler layer
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

    def _map_weights(self,  # pylint:disable=too-many-locals
                     torch_weights: dict[str, torch.Tensor],
                     keras_weights: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """ Convert the loaded keras weights to the format provided by the pre-existing torch
        weights and return as a compatible torch state_dict

        Returns
        -------
        The imported keras weights for importing into a torch plugin
        """
        is_clip = (self._state["name"] == "phaze_a" and
                   self._state["config"].get("enc_architecture", "").startswith("clipv_"))
        mask_layers = _get_mask_layers({k: v.input_layers
                                        for k, v in self._keras.layers.items()},
                                       {k: v.shape
                                        for k, v in self._keras.weights.items()
                                        if k.endswith(".0") and ".conv2d" in k})
        keras = KerasWeights(keras_weights, mask_layers, is_clip)
        torch_filtered = {k: v for k, v in torch_weights.items()  # Doesn't exist in keras
                          if not k.endswith("num_batches_tracked")}  # Reinserted at end
        logger.debug("[KerasToTorch] keras weights: %s, torch weights: %s",
                     len(keras), len(torch_filtered))
        if len(keras) != len(torch_filtered):
            raise RuntimeError(
                f"The number of weights within the keras file ({len(keras)}) differs from the "
                f"number of weights required by PyTorch ({len(torch_filtered)}). This is a bug "
                "and should be reported along with the model and configuration options used.")

        keras.group_weights()
        torch_grouped = self._group_weights(torch_filtered)
        logger.debug("[KerasToTorch] keras grouped weights: %s, torch grouped weights: %s",
                     keras.len_grouped, len(torch_grouped))

        if keras.len_grouped != len(torch_grouped):
            raise RuntimeError(
                f"The number of grouped weights within the keras file ({keras.len_grouped}) "
                f"differs from the number of weights required by PyTorch ({len(torch_grouped)}). "
                "This is a bug and should be reported along with the model and configuration "
                "options used.")

        # This logic goes through the loaded torch state_dict and searches forwards through the
        # keras model for where the first weight matches and pops it. This should be reasonably
        # robust as some tensors can drift a little, but not too far. Mask layer ordering is the
        # biggest barrier, so the search is filtered if learn_mask is enabled.
        # This will fail if match is not found.
        mapped: dict[str, torch.Tensor] = {}
        for lbl, weights in torch_grouped.items():
            weight_key = list(weights)[0]
            key, k_weights = keras.get_next_weights("mask" in lbl, weight_key, weights)

            if key.rsplit(".",
                          maxsplit=1)[-1].startswith("dense") and k_weights[weight_key].ndim == 2:
                self._dense_reorder(key,
                                    T.cast(dict[T.Literal["weight", "bias"], np.ndarray],
                                           k_weights))
            if key in self._pixel_shuffler_convs:
                self._pixel_shuffle_reorder(T.cast(dict[T.Literal["weight", "bias"], np.ndarray],
                                                   k_weights),
                                            self._pixel_shuffler_convs[key])

            logger.debug("[KerasToTorch] Mapped keras '%s' to torch '%s': %s",
                         key, lbl, k_weights[weight_key].shape)

            for k, v in k_weights.items():
                mapped[f"{lbl}.{k}"] = torch.from_numpy(v)

        retval: dict[str, torch.Tensor] = {}
        for k, v in torch_weights.items():
            if k.endswith("num_batches_tracked"):  # Re-insert non-existent batch norm tracking
                retval[k] = v
                continue
            retval[k] = mapped[k]  # Fail on unmatched

        logger.debug("[KerasToTorch] Mapped weights: %s", len(retval))
        return retval

    def _build_state_dict(self) -> None:
        """ Load the model state information to the plugin, initialize the plugin and map keras
        weights to the generated plugin's weights """
        # Initialize empty model with loaded state settings
        self._torch.load_state_dict({"state": self._state})
        self._state_dict = {"version": 1.0,
                            "state": self._state,
                            "model": self._map_weights(self._torch.plugin.state_dict(),
                                                       self._keras.weights)}
        if self._keras._optimizer:  # TODO
            pass

    def state_dict(self) -> dict[T.Literal["model", "state", "optimizer", "version"],
                                 float | dict[str, T.Any]]:
        """ Get the migrated state_dict from the old keras model """
        if not self._state_dict:
            self._build_state_dict()
        return self._state_dict


__all__ = get_module_objects(__name__)
