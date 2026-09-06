#! /usr/env/bin/python3
""" Migrate legacy Keras-era Faceswap models into the current PyTorch-based framework

FaceSwap moved from a TensorFlow backend to PyTorch, changing both the saved checkpoint format
and the in-memory state schema. To bridge that gap, this module reads checkpoints from the old
``.keras`` backend and converts them so their weights and training state can be reused with
modern models instead of being thrown away

It integrates with its neighbours to carry out this migration. It relies on ``.keras`` (via
``KerasModel``) to open a checkpoint and expose its layers and weights, on ``.topology`` for the
architecture helpers that identify pixel-shuffle and dense convolutions, and on ``.transformer``
for the ordering machinery that reshapes those weights into the layout PyTorch expects. On the
consumer side it works against a torch ``ModelPlugin``, loading the converted data back into it

The module defines a single migration driver, ``KerasToTorch``, which coordinates loading a
checkpoint, migrating its state forward to the current schema, and mapping its weights onto a
plugin's architecture. It also exposes two small helpers, one of which cleans and upgrades
legacy state; the other writes migrated data to disk via ``save_migrated_state_dict``.
"""
from __future__ import annotations

import logging
import os
import typing as T

import numpy as np
import torch

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .keras import KerasModel
from .topology import get_pixel_shuffler_convs, get_dense_reshapes, get_mask_layers

from .transformer import KerasWeights, dense_reorder, pixel_shuffle_reorder

if T.TYPE_CHECKING:
    from plugins.train.model.base import ModelPlugin

logger = logging.getLogger(__name__)


def save_migrated_state_dict(state_dict: dict[str, T.Any], checkpoint_path: str) -> str:
    """ Save a migrated state dict to disk, choosing the extension from its contents

    A full checkpoint carries an "optimizer" key and is written as ``{base}.ckpt``; weights-only
    state uses ``{base}.pth``. This rule is only guaranteed for legacy (Keras-era) migrations, so
    it lives here rather than in general persistence

    Parameters
    ----------
    state_dict
        The migrated state dict to save
    checkpoint_path
        Base path for the standard ``{model_name}.ckpt`` checkpoint file

    Returns
    -------
    The full path to the file that was written
    """
    ext = ".ckpt" if "optimizer" in state_dict else ".pth"
    output_path = f"{os.path.splitext(checkpoint_path)[0]}{ext}"
    logger.info("Saving migrated weights to '%s'", output_path)
    torch.save(state_dict, output_path)
    return output_path


def _get_state(keras_model: KerasModel) -> dict[str, T.Any]:
    """ Obtain the legacy state dict removing any keys that may break downstream dataclasses
    and updating any legacy items to be compatible with state version 2.0

    Parameters
    ----------
    keras_model
        The legacy Keras model whose state is cleaned and migrated to the current schema.

    Returns
    -------
    The cleansed state dictionary
    """
    state: dict[str, T.Any] = {"plugin_name": keras_model.state.pop("name").replace("_", "-"),
                               "plugin_version": 0.5}
    state |= {k: "none" if v is None else v  # Nonetype used to be allowed
              for k, v in keras_model.state.items()
              if k not in ("mixed_precision_layers",  # Dropped
                           "sessions")}  # Handled later

    legacy_defaults = {  # If these do not exist then state file is v. old. Set sane defaults
        "centering": "legacy",
        "coverage": 62.5,
        "mask_loss_function": "mse",
        "optimizer": "adam"
    }
    for key, val in legacy_defaults.items():
        state["config"][key] = state["config"].get(key, val)

    if isinstance(state.get("lowest_avg_loss"), dict):  # Loss used to be stored per side
        lowest_avg_loss = sum(T.cast(dict[str, float], state["lowest_avg_loss"]).values())
        logger.debug("[KerasToTorch] Collating legacy lowest_avg_loss from %s to %s",
                     state["lowest_avg_loss"], lowest_avg_loss)
        state["lowest_avg_loss"] = lowest_avg_loss

    # Following keys no longer exist or map to new keys
    priors = ["dssim_loss", "mask_type", "mask_type", "l2_reg_term", "clipnorm", "autoclip"]
    new_items = ["loss_function", "learn_mask", "mask_type", "loss_function_2",
                 "gradient_clipping", "clipping"]
    for old, new in zip(priors, new_items):
        if old not in state:
            logger.debug("[KerasToTorch] Legacy item '%s' not in state config. Skipping", old)
            continue

        if old == "dssim_loss":  # dssim_loss > loss_function
            state[new] = "ssim" if state[old] else "mae"
            del state[old]
            logger.debug("[KerasToTorch] Updated state config from legacy dssim format. New"
                         "config loss function: '%s'", state[new])
            continue

        if (old == "mask_type" and  # Replace removed masks with most similar equivalent
                new == "mask_type" and
                state[old] in ("facehull", "dfl_full")):
            old_mask = state[old]
            state[new] = "components"
            logger.debug("[KerasToTorch] Updated 'mask_type' from '%s' to '%s' for this model",
                         old_mask, state[new])

        if old == "l2_reg_term":  # Replace l2_reg_term with loss_2 func and update  weight
            state[new] = "mse"
            state["loss_weight_2"] = state[old]
            del state[old]
            logger.info("[KerasToTorch] Updated state config from legacy 'l2_reg_term' to "
                        "'loss_function_2'")

        if old == "clipnorm":  # Replace clipnorm with correct grad clip type and value
            state[new] = "norm"
            del state[old]
            logger.info("[KerasToTorch] Updated state config from legacy '%s' to '%s: %s'",
                        old, new, old)

        if old == "autoclip":  # Replace autoclip with correct gradient clipping type
            state[new] = old
            del state[old]
            logger.info("[KerasToTorch] Updated state config from legacy '%s' to '%s: %s'",
                        old, new, old)

    state["sessions"] = {int(i): {"batch_size" if k == "batchsize" else k: v
                                  for k, v in s.items() if k != "no_logs"}
                         for i, s in keras_model.state["sessions"].items()}
    logger.debug("[KerasToTorch] Cleaned state: %s", state)
    return state


class KerasToTorch:
    """ Port weights from a keras trained Faceswap model to pyTorch format

    Loads a legacy .keras model, migrates its state file forward to the current schema, and
    maps the Keras weights onto a torch plugin's existing architecture. The migrated state is
    exposed through the ``state`` property while weight mapping is handled by ``migrate``.

    Parameters
    ----------
    keras_file
        The fullpath to the keras model file
    """
    def __init__(self, keras_file: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._keras = KerasModel(keras_file)

        self._pixel_shuffler_convs = get_pixel_shuffler_convs(self._keras.layers)
        self._dense_reshapes = get_dense_reshapes(self._keras.layers)
        self._state = _get_state(self._keras)

    @property
    def state(self) -> dict[str, T.Any]:
        """ The model state file formatted as a Torch state_dict """
        return self._state

    def _group_torch_weights(self, weights: dict[str, torch.Tensor]
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

    def _group_weights(self,
                       torch_weights: dict[str, torch.Tensor],
                       keras: KerasWeights) -> dict[str, dict[str, torch.Tensor]]:
        """ Check that Keras and torch weight counts agree, then group weights by layer

        Parameters
        ----------
        torch_weights
            The torch plugin's weights with batch norm tracking counters removed
        keras
            The grouped Keras weights used to confirm the count matches PyTorch's expectations

        Returns
        -------
        Each model layer keyed by name, holding a dictionary of its weights and biases

        Raises
        ------
        RuntimeError
            If the number of Keras or grouped weights differs from what PyTorch requires
        """
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
        torch_grouped = self._group_torch_weights(torch_filtered)
        logger.debug("[KerasToTorch] keras grouped weights: %s, torch grouped weights: %s",
                     keras.len_grouped, len(torch_grouped))

        if keras.len_grouped != len(torch_grouped):
            raise RuntimeError(
                f"The number of grouped weights within the keras file ({keras.len_grouped}) "
                "differs from the number of grouped weights required by PyTorch "
                f"({len(torch_grouped)}). This is a bug and should be reported along with the "
                "model and configuration options used.")

        return torch_grouped

    def _map_weights(self,
                     torch_grouped: dict[str, dict[str, torch.Tensor]],
                     keras_weights: KerasWeights) -> dict[str, torch.Tensor]:
        """ Convert the loaded keras weights to the format provided by the pre-existing torch
        weights and return as a compatible torch state_dict

        Parameters
        ----------
        torch_grouped
            The torch weights grouped by layer, keyed by layer name.
        keras_weights
            The Keras weights used to find matching tensors as each weight is mapped.

        Returns
        -------
        The imported keras weights for importing into a torch plugin
        """
        # This logic goes through the loaded torch state_dict and searches forwards through the
        # keras model for where the first weight matches and pops it. This should be reasonably
        # robust as some tensors can drift a little, but not too far. Mask layer ordering is the
        # biggest barrier, so the search is filtered if learn_mask is enabled.
        # This will fail if match is not found.
        mapped: dict[str, torch.Tensor] = {}
        for lbl, weights in torch_grouped.items():
            weight_key = list(weights)[0]
            key, k_weights = keras_weights.get_next_weights("mask" in lbl, weight_key, weights)

            if key.rsplit(".",
                          maxsplit=1)[-1].startswith("dense") and k_weights[weight_key].ndim == 2:
                dense_reorder(key,
                              T.cast(dict[T.Literal["weight", "bias"], np.ndarray],
                                     k_weights),
                              self._dense_reshapes)
            if key in self._pixel_shuffler_convs:
                pixel_shuffle_reorder(T.cast(dict[T.Literal["weight", "bias"], np.ndarray],
                                             k_weights),
                                      self._pixel_shuffler_convs[key])

            logger.debug("[KerasToTorch] Mapped keras '%s' to torch '%s': %s",
                         key, lbl, k_weights[weight_key].shape)

            for k, v in k_weights.items():
                mapped[f"{lbl}.{k}"] = torch.from_numpy(v)
        return mapped

    def migrate(self, plugin: ModelPlugin) -> None:
        """ Load a keras model's weights into a torch plugin, migrating then mapping each one

        Maps the plugin's torch weights against the Keras weights, layer by layer. Reorder dense
        and pixel-shuffle layers as needed before loading them back, re-inserting batch norm
        tracking counters unchanged

        Parameters
        ----------
        plugin
            The torch model plugin whose weights are replaced with the migrated Keras weights
        """
        is_clip = (self._state["plugin_name"] == "phaze-a" and
                   self._state["config"].get("enc_architecture", "").startswith("clipv_"))
        mask_layers = get_mask_layers({k: v.input_layers
                                       for k, v in self._keras.layers.items()},
                                      {k: v.shape
                                       for k, v in self._keras.weights.items()
                                       if k.endswith(".0") and ".conv2d" in k})

        keras = KerasWeights(self._keras.weights, mask_layers, is_clip)
        torch_weights = plugin.state_dict()
        torch_grouped = self._group_weights(torch_weights, keras)

        mapped = self._map_weights(torch_grouped, keras)

        state: dict[str, torch.Tensor] = {}
        for k, v in torch_weights.items():
            if k.endswith("num_batches_tracked"):  # Re-insert non-existent batch norm tracking
                state[k] = v
                continue
            state[k] = mapped[k]  # Fail on unmatched

        logger.debug("[KerasToTorch] Mapped weights: %s", len(state))
        # TODO optimizer
        plugin.load_state_dict(state)


__all__ = get_module_objects(__name__)
