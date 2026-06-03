#! /usr/env/bin/python3
"""Handles loading information from legacy .keras models"""
from __future__ import annotations

import io
import json
import logging
import os
import typing as T
import zipfile

import h5py
import numpy as np
import torch

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from lib.model.faceswap import FaceswapModel

logger = logging.getLogger(__name__)


class KerasModel:
    """Loads data from a .keras model

    Parameters
    ----------
    model_path
        The full path to a keras model
    """
    def __init__(self, model_path: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._model_path = model_path
        self.state = self._load_state_file()
        """The keras model's state file"""
        self._config: dict[str, T.Any] = {}
        self.weights = {}
        """The stored layer name to numpy array for the loaded keras model"""
        self._optimizer: dict[T.Literal["version", "optimizer", "scale"], T.Any] = {}

        self._load_keras_model()

    def _get_weights(self,
                     entry: h5py.Group | h5py.Dataset,
                     collected: None | dict[str, np.ndarray] = None) -> dict[str, np.ndarray]:
        """Recurse through the data and collect model weights as numpy arrays

        Parameters
        ----------
        entry
            An h5py Group or Dataset entry from a Keras model
        collected
            Data collected so far. Default: ``None`` (first iteration)

        Returns
        -------
        The layer path names to layer weights in topological order in keras layout
        """
        assert entry.name is not None

        collected = {} if collected is None else collected
        if isinstance(entry, h5py.Dataset):
            collected |= {entry.name[1:].replace("/", "."): np.array(entry)}
            return collected

        if isinstance(entry, h5py.Group):
            collected |= {k: v
                          for e in entry
                          for k, v in self._get_weights(T.cast(h5py.Group | h5py.Dataset,
                                                               entry.get(e))).items()}

            return collected

        raise RuntimeError(f"Unhandled h5py file type '{entry.name}': {type(entry)}")

    def _load_state_file(self) -> dict[str, T.Any]:
        """Load the legacy state file"""
        state_path = f"{os.path.splitext(self._model_path)[0]}_state.json"
        if not os.path.exists(state_path):
            logger.warning("Legacy state file '%s' not found. Model training history not imported",
                           state_path)
            return {}

        with open(state_path, "r", encoding="utf-8") as s_file:
            retval = json.load(s_file)
        logger.debug("[KerasModel] Loaded state: %s", retval)
        return retval

    def _load_keras_model(self):
        """Load the objects we require out of the keras model file"""

        with zipfile.ZipFile(self._model_path, "r") as z_file:
            name_list = z_file.namelist()
            logger.debug("[KerasModel] zip file contents: %s", name_list)
            for fname in ("config.json", "model.weights.h5"):
                if fname not in name_list:
                    raise ValueError(f"Could not find key '{fname}' in "
                                     f"model file: {self._model_path}")

            self._config = json.loads(z_file.read("config.json"))
            logger.debug("[KerasModel] Loaded config: %s", self._config)

            weights = h5py.File(io.BytesIO(z_file.read("model.weights.h5")), "r")
            self.weights = self._get_weights(T.cast(h5py.Group, weights["layers"]))
            logger.debug("[KerasModel] Loaded weights: %s",
                         {k: v.shape for k, v in self.weights.items()})
            if "optimizer.pt" in name_list:
                self._optimizer = torch.load(io.BytesIO(z_file.read("optimizer.pt")))
                logger.debug("[KerasModel] Loaded optimizer state: %s",
                             {k: v if k == "version" else type(v)
                              for k, v in self._optimizer.items()})


ArrayT = T.TypeVar("ArrayT", torch.Tensor, np.ndarray)


class KerasToTorch:
    """Port weights from a keras trained Faceswap model to pyTorch format

    Parameters
    ----------
    torch_model
        The uninitialized corresponding Torch model plugin
    keras_file
        The fullpath to the keras model file
    """
    def __init__(self, torch_model: FaceswapModel, keras_file: str) -> None:
        logger.info(parse_class_init(locals()))
        self._keras = KerasModel(keras_file)
        self._torch = torch_model

        self._state_dict: dict[T.Literal["model", "state", "optimizer", "version"],
                               float | dict[str, T.Any]] = {}
        self._state = self._get_state()

    def _get_state(self) -> dict[str, T.Any]:
        """Obtain the legacy state dict removing any removed keys that may break downstream
        dataclasses

        Returns
        -------
        The keras state file with mixed_precision_layers and no_logs keys removed
        """
        retval = {k: v for k, v in self._keras.state.items() if k != "mixed_precision_layers"}
        retval["sessions"] = {int(i): {"batch_size" if k == "batchsize" else k: v
                                       for k, v in s.items() if k != "no_logs"}
                              for i, s in self._keras.state["sessions"].items()}
        logger.debug("[KerasToTorch] Cleaned state: %s", retval)
        return retval

    @classmethod
    def _group_layer_weights(cls,
                             weights: dict[str, ArrayT],
                             reshape_to_torch: bool
                             ) -> dict[str, dict[T.Literal["weight", "bias"], ArrayT]]:
        """Group the list of layer weights and biases by layer

        Parameters
        ----------
        weights
            The weights to group, with separate items for weights and biases
        reshape_to_torch
            ``True`` when input is keras so weights should be reshaped to Torch

        Returns
        -------
        Each layer of the model with a dictionary containing it's weights and biases
        """
        retval = {}
        for lbl, weight in weights.items():
            name, w_type = lbl.rsplit(".", maxsplit=1)
            if reshape_to_torch:
                w_type = "weight" if w_type == "0" else "bias"  # keras indexing to torch name
                assert isinstance(weight, np.ndarray)
                if weight.ndim == 4:
                    weight = weight.transpose(3, 2, 0, 1)
                elif weight.ndim == 2:
                    weight = weight.transpose(1, 0)
                elif weight.ndim != 1:
                    raise RuntimeError(f"Unhandled weight shape {weight.shape} for layer: "
                                       f"'{weight}'")

            assert w_type in ("weight", "bias")
            retval[name] = retval.get(name, {}) | {w_type: weight}
        return retval

    def _map_weights(self,
                     torch_weights: dict[str, torch.Tensor],
                     keras_weights: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """Convert the loaded keras weights to the format provided by the pre-existing torch
        weights and return as a compatible torch state_dict

        Returns
        -------
        The imported keras weights for importing into a torch plugin
        """
        # TODO Test this for all models as topological unlikely to always work
        if len(keras_weights) != len(torch_weights):
            raise RuntimeError(f"Keras weight count ({len(keras_weights)}) does not match Torch "
                               f"weight count ({len(torch_weights)})")

        keras_grouped = self._group_layer_weights(keras_weights, reshape_to_torch=True)
        torch_grouped = self._group_layer_weights(torch_weights, reshape_to_torch=False)
        if len(keras_grouped) != len(torch_grouped):
            raise RuntimeError(f"Keras weight count ({len(keras_grouped)}) does not match Torch "
                               f"weight count ({len(torch_grouped)})")

        # This logic goes through the loaded torch state_dict and searches forwards through the
        # keras model for where the first weight matches and pops it. This should be reasonably
        # robust as some tensors can drift a little, but not too far
        # This will fail if match is not found.
        retval: dict[str, torch.Tensor] = {}
        for lbl, weights in torch_grouped.items():
            key = next(k for k, v in keras_grouped.items()
                       if v["weight"].shape == weights["weight"].shape)
            val = keras_grouped.pop(key)
            if lbl == "encoder.dense1":
                # TODO need way of getting dense layers from flattened rather than hard coding
                # This will probably need to come from model_info once tracing properly implemented
                # H * W * C -> H, W, C -> C, H, W -> C * H * W
                val["weight"] = val["weight"].reshape(1024, 4, 4, 1024).transpose(0, 3, 1, 2).reshape(1024, 16384)

            if lbl == "encoder.dense2":
                # TODO need way of getting dense layers rather than hard coding
                # H * W * C -> H, W, C -> C, H, W -> C * H * W
                val["weight"] = val["weight"].reshape(4, 4, 1024, 1024).transpose(2, 0, 1, 3).reshape(16384, 1024)
                val["bias"] = val["bias"].reshape(4, 4, 1024).transpose(2, 0, 1).reshape(16384)

            logger.debug("[KerasToTorch] Mapped keras '%s' to torch '%s': %s",
                         key, lbl, val["weight"].shape)
            for w in ("weight", "bias"):
                retval[f"{lbl}.{w}"] = torch.from_numpy(val[w])

        logger.debug("[KerasToTorch] Mapped weights: %s", len(retval))
        return retval

    def _build_state_dict(self) -> None:
        """Load the model state information to the plugin, initialize the plugin and map keras
        weights to the generated plugin's weights"""
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
        """Get the migrated state_dict from the old keras model"""
        if not self._state_dict:
            self._build_state_dict()
        return self._state_dict


__all__ = get_module_objects(__name__)
