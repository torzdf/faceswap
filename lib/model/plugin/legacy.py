#! /usr/env/bin/python3
""" Handles loading information from legacy .keras models """
from __future__ import annotations

import io
import json
import logging
import os
import typing as T
import zipfile
from dataclasses import dataclass, field
from collections import Counter

import h5py
import numpy as np
import torch

from lib.logger import parse_class_init
from lib.utils import camel_to_snake_case, get_module_objects
from .legacy_build_order import reorder_layers

if T.TYPE_CHECKING:
    from .handler import FaceswapModel

logger = logging.getLogger(__name__)


@dataclass
class LayerInfo:
    """ Holds information about a layer collected from keras config file """
    layer_name: str
    """ The name of the original layer in the keras config """
    weights_name: str
    """ The derived name of the corresponding layer in the weights file """
    layer_type:  str
    """ The type (ClassName) of the layer """
    input_layers: list[str] = field(default_factory=list)
    """ List of inbound nodes to the layer """
    input_shapes: list[list[int]] = field(default_factory=list)
    """ List of input shapes to the layer corresponding to input_layers """


class KerasConfigParser:
    """ Parses a nested keras config dictionary to a flattened dictionary of standardized layer
    names as stored within the hdf file in config file order, mapped to layer information """

    @classmethod
    def _next_label(cls, cls_name: str, dst_name: str | None, counters: dict[str, int]) -> str:
        """ Compute the standardized (hdf-style) label for a single config node.

        Parameters
        ----------
        cls_name
            The Keras class name of the layer
        dst_name
            The currently building standardized layer name or ``None`` if this is the first
            iteration
        counters
            Count of how many times each standardized layer name has been seen.

        Returns
        -------
        The next available standardized name for the given layer
        """
        if dst_name is None:
            return "layers"  # Parent model always starts with "layers"

        base = f"{dst_name}.{camel_to_snake_case(cls_name)}"
        count = counters.get(base, 0)
        counters[base] = count + 1
        return base if count == 0 else f"{base}_{count}"

    @classmethod
    def _sub_model_output_name(cls, config: dict[str, T.Any]) -> str:
        """ Get the Keras name of a sub-model's output layer to ensure sub-model inputs are mapped
        to the relevant sub-model output layer. For Faceswap sub-models with multiple outputs,
        only the first output name is required """
        out_layers = config["config"]["output_layers"]
        return out_layers[0][0] if isinstance(out_layers[0], list) else out_layers[0]

    @classmethod
    def _map_sub_model_input(cls,
                             config: dict[str, T.Any],
                             mapping: dict[str, str],
                             outputs: dict[str, str],
                             is_sequential: bool) -> None:
        """ Map a sub-model's internal input layer name to whatever produces its input from the
        outer graph. """
        if is_sequential and config.get("inbound_nodes"):
            return  # top-level Sequential with no outer context to resolve

        inbounds: list[str] = list({arg["config"]["keras_history"][0]
                                    for node in config["inbound_nodes"]
                                    for arg in node["args"]})

        if is_sequential:
            assert len(inbounds) == 1  # Sequential models with multiple inputs not handled
            in_names = inbounds
        else:
            in_layers = config["config"]["input_layers"]
            in_layers = in_layers if isinstance(in_layers[0], list) else [in_layers]
            in_names = [i[0] for i in in_layers]
            assert len(in_names) == 1  # TODO sub-models with multiple inputs

        # Resolve a keras inbound node name to its standardized label, redirecting through the sub-
        # model's recorded output if 'name' is itself a sub-model.  We only need first input if
        # multi for our purposes
        mapped = next(mapping.get(outputs.get(name, name), name) for name in inbounds)
        logger.debug("[KerasConfigParser] Mapping '%s' to '%s' for sub-model '%s'",
                     in_names[0], mapped, config["name"])
        mapping[in_names[0]] = mapped

    @classmethod
    def _flatten_sub_model(cls,
                           config: dict[str, T.Any],
                           dst_label: str,
                           counters: dict[str, int],
                           mapping: dict[str, str],
                           outputs: dict[str, str]) -> dict[str, LayerInfo]:
        """ Flatten a nested Functional/Model layer, registering its input/output name mappings via
        recursing into its children.

        Parameters
        ----------
        config
            A keras config dictionary
        dst_label
            The currently building standardized layer name
        counters
            Count of how many times each standardized layer name has been seen
        mapping
            Mapping of keras layer names to standardized weight names
        outputs
            Mapping of keras functional sub-model names to their output layer names

        Returns
        -------
        dict of derived weights layer names in graph parsing order mapped to layer information
        """
        name = config.get("name")
        is_sequential = config["class_name"] == "Sequential"
        if name:
            # Fallback (eg top-level concatenates referencing this model by name. Corrected to
            # real output layer name when known at the end of the function):
            mapping[config["name"]] = dst_label
            cls._map_sub_model_input(config, mapping, outputs, is_sequential)

        # If there is no "." in the label then this is the main parent model
        child_dst = dst_label if "." not in dst_label else f"{dst_label}.layers"
        retval = {}

        if is_sequential:
            assert name is not None
            prev_label = mapping.get(name)
            for layer in config["config"]["layers"]:
                layer_result = cls.flatten(
                    layer, child_dst, counters, mapping, outputs, prev_label)
                retval |= layer_result
                if layer_result:
                    prev_label = next(reversed(layer_result))  # last inserted key
            # Record true output (last real layer) for downstream consumers
            if name and prev_label:
                outputs[name] = prev_label

        for layer in config["config"]["layers"]:
            retval |= cls.flatten(layer, child_dst, counters, mapping, outputs)

        # Replace functional output with actual layer name of the output of the sub-model:
        if name and name in outputs and outputs[name] in mapping:
            logger.debug("[KerasConfigParser] Remapping model '%s' to model output: '%s'",
                         mapping[name], outputs[name])
            mapping[name] = mapping[outputs[name]]
        return retval

    @classmethod
    def _extract_input_shapes(cls,
                              config: dict[str, T.Any],
                              mapping: dict[str, str],
                              prev_label: str | None) -> dict[str, list[int]]:
        """ Build {producer_label: input_shape} for a leaf layer from its inbound_nodes,
        normalizing Keras' inconsistent arg structure. Falls back to build_config + prev_label for
        Sequential models """
        if "inbound_nodes" not in config:  # Sequential model. Name from build_config
            if prev_label is None:
                return {}
            input_shape = config.get("build_config", {}).get("input_shape", [])
            return {prev_label: list(input_shape[1:])} if len(input_shape) > 1 else {}

        in_shapes = {}
        for node in config["inbound_nodes"]:
            for arg in node["args"]:
                tensors = arg if isinstance(arg, list) else [arg]  # Handle inconsistent arg types
                for tensor in tensors:
                    if not isinstance(tensor, dict):
                        continue
                    producer = tensor["config"]["keras_history"][0]
                    in_shapes[mapping[producer]] = tensor["config"]["shape"][1:]
        return in_shapes

    @classmethod
    def flatten(cls,
                config: dict[str, T.Any],
                parent: str | None = None,
                counters: dict[str, int] | None = None,
                mapping: dict[str, str] | None = None,
                outputs: dict[str, str] | None = None,
                prev_label: str | None = None) -> dict[str, LayerInfo]:
        """ Recurse through the config.json file flattening to a matching format to the hdf weights

        Parameters
        ----------
        config
            A keras config dictionary
        parent
            The parent model's standardized layer name. Default: ``None`` (first iteration)
        counters
            Count of how many times each standardized layer name has been seen. Default: ``None``
            (first iteration)
        mapping
            Mapping of keras layer names to standardized weight names. Default: ``None`` (first
            iteration)
        outputs
            Mapping of keras functional model names to their output layer names. Default: ``None``
            (first iteration)
        prev_label
            The dst_label of the preceding layer (used for Sequential models where inbound_nodes
            are absent). Default: ``None``

        Returns
        -------
        dict of derived weights layer names in graph parsing order mapped to layer information
        """
        counters = {} if counters is None else counters
        mapping = {} if mapping is None else mapping
        outputs = {} if outputs is None else outputs

        dst_label = cls._next_label(config["class_name"], parent, counters)

        if "layers" in config["config"]:
            return cls._flatten_sub_model(config, dst_label, counters, mapping, outputs)

        if "name" in config and not config["name"].startswith("input_layer"):
            mapping[config["name"]] = dst_label

        info = LayerInfo(config["name"] if "name" in config else config["class_name"],
                         dst_label,
                         config["class_name"])

        logger.debug("[KerasConfigParser] mapped layer '%s' to weight '%s'",
                     info.layer_name, dst_label)

        for k, v in cls._extract_input_shapes(config, mapping, prev_label).items():
            info.input_layers.append(k)
            info.input_shapes.append(v)
        return {dst_label: info}


class KerasModel:  # pylint:disable=too-few-public-methods
    """ Loads data from a .keras model

    Parameters
    ----------
    model_path
        The full path to a keras model
    """
    def __init__(self, model_path: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._model_path = model_path
        self.state = self._load_state_file()
        """ The keras model's state file """
        self.layers: dict[str, LayerInfo] = {}
        """ Flattened dict of standardized layer names within the model in config file order as
        derived from the model's config.json, standardized to h5 file weights labels format mapped
        to layer information """
        self.weights = {}
        """ The stored layer name to numpy array for the loaded keras model in model creation
        order """
        self._optimizer: dict[T.Literal["version", "optimizer", "scale"], T.Any] = {}

        self._load_keras_model()

    def _get_weights(self,
                     entry: h5py.Group | h5py.Dataset,
                     collected: None | dict[str, np.ndarray] = None) -> dict[str, np.ndarray]:
        """ Recurse through the data and collect model weights as numpy arrays

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

    def _sort_weights(self, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ Sort the weights into model construction order

        Parameters
        ----------
        weights
            The weights collected from the .h5 file

        Returns
        -------
        The weights sorted into model creation order
        """
        model = (self.state["config"]["enc_architecture"] if self.state["name"] == "phaze_a"
                 else self.state["name"])
        self.layers = reorder_layers(model, self.layers)

        lookup: dict[str, list[str]] = {}
        # Remove normalization weights from the beginning of EffNet  # TODO may actually need to prepend norms for legacy torch
        kap_enc_weights = [k for k in weights
                           if k.startswith("layers.functional.layers.functional.layers.")]
        for k in reversed(kap_enc_weights):
            if ".layers.normalization.vars." not in k:  # Always at end of encoder list
                break
            logger.info("[KerasModel] Removing KApp normalization weights: '%s'", k)
            del weights[k]
        for k in weights:
            lookup.setdefault(k.rsplit(".", maxsplit=2)[0], []).append(k)
        retval: dict[str, np.ndarray] = {}
        for k in self.layers:
            if k in lookup:
                keys = lookup[k]
                for key in keys:
                    retval[key] = weights.pop(key)
        assert len(weights) == 0, f"Not all weights mapped. Remaining: {len(weights)}"
        return retval

    def _load_state_file(self) -> dict[str, T.Any]:
        """ Load the legacy state file """
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
        """ Load the objects we require out of the keras model file """
        with zipfile.ZipFile(self._model_path, "r") as z_file:
            name_list = z_file.namelist()
            logger.debug("[KerasModel] zip file contents: %s", name_list)
            for fname in ("config.json", "model.weights.h5"):
                if fname not in name_list:
                    raise ValueError(f"Could not find key '{fname}' in "
                                     f"model file: {self._model_path}")

            # TODO remove
            # with open("/mnt/Data/fstest/train/conf.json", "w") as ofile:
            #     json.dump(json.loads(z_file.read("config.json")), ofile, indent=2)
            # exit()

            self.layers = KerasConfigParser.flatten(json.loads(z_file.read("config.json")))
            # TODO remove
            # for k, v in self.layers.items():
            #    print(k, v)
            # exit()
            logger.debug("[KerasModel] Flattened model layers: %s", self.layers)

            weights = h5py.File(io.BytesIO(z_file.read("model.weights.h5")), "r")
            self.weights = self._sort_weights(self._get_weights(T.cast(h5py.Group,
                                                                       weights["layers"])))
            logger.debug("[KerasModel] Loaded weights: %s",
                         {k: v.shape for k, v in self.weights.items()})
            if "optimizer.pt" in name_list:
                self._optimizer = torch.load(io.BytesIO(z_file.read("optimizer.pt")))
                logger.debug("[KerasModel] Loaded optimizer state: %s",
                             {k: v if k == "version" else type(v)
                              for k, v in self._optimizer.items()})


ArrayT = T.TypeVar("ArrayT", torch.Tensor, np.ndarray)


class KerasToTorch:
    """ Port weights from a keras trained Faceswap model to pyTorch format

    Parameters
    ----------
    torch_model
        The uninitialized corresponding Torch model plugin
    keras_file
        The fullpath to the keras model file
    """
    custom_mapping = {"iae": {"inter_side.0": "layers.functional_1",
                              "inter_both": "layers.functional_2"}}  # TODO move to legacy_build_order
    """ Mapping for instances where keras graph and build orders don't match up """

    def __init__(self, torch_model: FaceswapModel, keras_file: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._keras = KerasModel(keras_file)
        self._torch = torch_model

        self._state_dict: dict[T.Literal["model", "state", "optimizer", "version"],
                               float | dict[str, T.Any]] = {}

        self._pixel_shuffler_convs = self._get_pixel_shuffler_convs(self._keras.layers)
        self._dense_reshapes = self._get_dense_reshapes(self._keras.layers)
        self._mask_layers = self._get_mask_layers({k: v.input_layers
                                                   for k, v in self._keras.layers.items()},
                                                  {k: v.shape
                                                   for k, v in self._keras.weights.items()
                                                   if k.endswith(".0") and ".conv2d" in k})
        self._state = self._get_state()

    @classmethod
    def _get_pixel_shuffler_convs(cls, layers: dict[str, LayerInfo]) -> dict[str, int]:
        """ Obtain a list of convolutions that lead into pixel shuffler layers for channel re-
        ordering

        Parameters
        ----------
        layers
            The list of standardized layer names within the keras model mapped to there inbound
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
            assert len(info.input_layers) == 1  # FS never has more than 2 inputs into a PS
            in_ = info.input_layers[0]
            in_size = info.input_shapes[0][0]
            in_conv = None
            while True:
                if in_.rsplit(".", maxsplit=1)[-1].startswith("conv2d"):
                    logger.debug("[KerasToTorch] Collected conv '%s' for pixel shuffler '%s'",
                                 in_, layer)
                    in_conv = in_
                    break

                logger.debug("[KerasToTorch] Skipping non-conv '%s' for pixel shuffler '%s'",
                             in_, layer)
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

        logger.debug("[KerasToTorch] Pixel Shuffler convs and scales: %s", retval)
        return retval

    @classmethod
    def _get_dense_reshapes(cls, layers: dict[str, LayerInfo]
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
                    logger.debug("[KerasToTorch] Getting input to '%s' for layer '%s'", in_, layer)
                    info = layers[in_]
                    continue
                break

            # Reshape in
            if is_dense and not in_.rsplit(".", maxsplit=1)[-1].startswith("flatten"):
                logger.debug("[KerasToTorch] Skipping in channel dense '%s' with input '%s'",
                             layer, in_)
                continue
            if is_dense:
                assert len(layers[in_].input_layers) == 1
                shape = tuple(layers[in_].input_shapes[0])
                assert len(shape) == 3  # Must be H, W, C
                retval[layer] = (True, shape)
                logger.debug("[KerasToTorch] Collected in channel reshape for '%s': %s",
                             layer, shape)
                continue

            # Reshape out
            if not in_.rsplit(".", maxsplit=1)[-1].startswith("dense"):
                logger.debug("[KerasToTorch] Skipping reshape '%s' with input '%s'",
                             layer, in_)
                continue
            shape = tuple(reshapes[layer])
            assert len(shape) == 3  # Must be H, W, C
            retval[in_] = (False, shape)
            logger.debug("[KerasToTorch] Collected out channel reshape for '%s': %s",
                         in_, shape)
        logger.debug("[KerasToTorch] Dense reshape weights: %s", retval)
        return retval

    def _recurse_from_layer(self,
                            layers: dict[str, list[str]],
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
                logger.debug("[KerasToTorch] Exited sub-model '%s' at layer '%s'",
                             sub_model, lyr)
                continue
            if lyr in seen:
                continue
            seen.add(lyr)
            retval.append(lyr)
            retval += self._recurse_from_layer(layers, layers[lyr], sub_model, seen)

        return retval

    def _get_mask_layers(self,
                         layers: dict[str, list[str]],
                         weights: dict[str, tuple[int, ...]]) -> list[str]:
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
            logger.debug("[KerasToTorch] No mask output. Returning empty list")
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
        logger.debug("[KerasToTorch] Selecting image output layers %s, mask output layers %s",
                     img_outputs, msk_outputs)

        img_layers = [y for x in img_outputs
                      for y in self._recurse_from_layer(layers,
                                                        [x],
                                                        ".".join(x.split(".", maxsplit=2)[:2]))]
        msk_layers = [y for x in msk_outputs
                      for y in self._recurse_from_layer(layers,
                                                        [x],
                                                        ".".join(x.split(".", maxsplit=2)[:2]))]

        retval = [x for x in msk_layers if x not in img_layers]
        logger.debug("[KerasToTorch] Collected mask path layers: %s", retval)
        return retval

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

    @classmethod
    def _prepare_keras_weights(cls, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ Some Keras weights need preparation for porting. Specifically:

        - SeparableConv2D split to 2x convs (replaced in original order)
        - BatchNorm2D with `scale=False` requires weights to be inserted initialized to 1.0 as this
        configuration does not exist in torch.

        Parameters
        ----------
        weights
            The original keras weights extracted from the hdf file

        Returns
        -------
        The original keras weights with any processing applied.
        """
        # BatchNorm
        bn_count = Counter(k.rsplit(".", maxsplit=2)[0] for k in weights
                           if "batch_normalization" in k)
        bn_update = {k for k, v in bn_count.items() if v == 3}
        if bn_update:
            new_weights = {}
            for k, v in weights.items():
                prefix, w_idx = k.rsplit(".", maxsplit=1)
                if prefix.rsplit(".", maxsplit=1)[0] not in bn_update:
                    new_weights[k] = v
                    continue
                if w_idx == "0":
                    new_weights[k] = np.ones_like(v)
                new_weights[f"{k.rsplit('.', maxsplit=1)[0]}.{int(w_idx) + 1}"] = v
            logger.debug("[KerasToTorch] Inserted 'weight' to %s BatchNorms for 'scale=False'",
                         len(new_weights) - len(weights))
            weights = new_weights

        # SeparableConv2D
        src_weights = {k: v for k, v in weights.items() if "separable_conv2d" in k}
        if not src_weights:
            return weights

        remap = {}
        for name in src_weights:
            l_name, v_name, v_idx = name.rsplit(".", maxsplit=2)
            new_l_name = f"{l_name}_a" if v_idx == "0" else f"{l_name}_b"  # no bias on first conv
            new_v_idx = v_idx if v_idx == "0" else str(int(v_idx) - 1)  # Reduce v_idx on 2nd conv
            remap[name] = ".".join([new_l_name, v_name, new_v_idx])

        retval = {remap.get(k, k): v for k, v in weights.items()}
        logger.debug("[KerasToTorch] Remapped keras weights: %s", remap)
        return retval

    def _group_layer_weights(self,
                             weights: dict[str, ArrayT],
                             reshape_to_torch: bool
                             ) -> dict[str, dict[T.Literal["weight",
                                                           "bias",
                                                           "running_mean",
                                                           "running_var",
                                                           "num_batches_tracked"], ArrayT]]:
        """ Group the list of layer weights and biases by layer and remove trailing 'vars' label

        Parameters
        ----------
        weights
            The weights to group, with separate items for weights and biases
        reshape_to_torch
            ``True`` when input is keras so weights should be reshaped to Torch

        Returns
        -------
        Each layer of the model from the .h5 file with a dictionary containing it's weights and
        biases
        """
        retval = {}
        for lbl, weight in weights.items():
            name, w_type = lbl.rsplit(".", maxsplit=1)
            if reshape_to_torch:
                k_map = {"0": "weight", "1": "bias", "2": "running_mean", "3": "running_var"}
                name = name.rsplit(".", maxsplit=1)[0]  # Strip .vars from the end
                w_type = k_map[w_type]  # keras indexing to torch name
                assert isinstance(weight, np.ndarray)
                if (weight.ndim == 4
                        and any(x in name for x in ("separable_conv2d", "depthwise_conv2d"))
                        and weight.shape[0] != 1):
                    new_shape = (weight.shape[2] * weight.shape[3], 1, *weight.shape[:2])
                    weight = weight.transpose(2, 3, 0, 1).reshape(new_shape)
                elif weight.ndim == 4:
                    weight = weight.transpose(3, 2, 0, 1)
                elif weight.ndim == 2:
                    weight = weight.transpose(1, 0)
                elif ".layer_scale" in name:  # ConvNeXt layer scale needs dims expanded:
                    assert weight.ndim == 1, f"Keras layer_scale shape: {weight.shape}"
                    weight = weight[:, None, None]
                elif weight.ndim != 1:
                    raise RuntimeError(f"Unhandled weight shape {weight.shape} for layer: "
                                       f"'{weight}'")
            w_type = "weight" if w_type == "layer_scale" else w_type  # ConvNext re-label in torch
            assert w_type in ("weight",
                              "bias",
                              "running_mean",
                              "running_var",
                              "num_batches_tracked"), f"Unhandled weight type: {w_type}"
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

    def _get_keras_key(self,
                       torch_key: str,
                       weight_shape: torch.Size,
                       keras_grouped: dict[str, dict[T.Literal["weight",
                                                               "bias",
                                                               "running_mean",
                                                               "running_var",
                                                               "num_batches_tracked"], np.ndarray]]
                       ) -> str:
        """ Obtain the next qualifying weight key for the given torch key """
        re_mapping = self.custom_mapping.get(self._torch.name, {})
        re_map = next((x for x in [v if torch_key.startswith(k) else None
                                   for k, v in re_mapping.items()]
                       if x), None)
        if re_map:
            logger.debug("[KerasToTorch] Remapping '%s' to keras model '%s'", torch_key, re_map)
        retval = next(k for k, v in keras_grouped.items()
                      if ("mask" in torch_key and k in self._mask_layers
                          or "mask" not in torch_key and k not in self._mask_layers)
                      and (not re_map or k.startswith(re_map))
                      and v["weight"].shape == weight_shape)
        return retval

    def _map_weights(self,
                     torch_weights: dict[str, torch.Tensor],
                     keras_weights: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """ Convert the loaded keras weights to the format provided by the pre-existing torch
        weights and return as a compatible torch state_dict

        Returns
        -------
        The imported keras weights for importing into a torch plugin
        """
        keras_weights = self._prepare_keras_weights(keras_weights)
        torch_filtered = {k: v for k, v in torch_weights.items()  # Doesn't exist in keras
                          if not k.endswith("num_batches_tracked")}  # Reinserted at end

        if len(keras_weights) != len(torch_filtered):
            # TODO remove
            for i in range(max(len(keras_weights), len(torch_filtered))):
                if len(keras_weights) > i:
                    k_key = list(keras_weights)[i]
                    out = k_key + "@" + str(keras_weights[k_key].shape) + "@"
                else:
                    out = " @ @"
                if len(torch_filtered) > i:
                    t_key = list(torch_filtered)[i]
                    out += t_key + "@" + str(torch_filtered[t_key].cpu().numpy().shape) + "@"
                else:
                    out += " @ @"
                print(out)

            raise RuntimeError(f"Keras weight count ({len(keras_weights)}) does not match Torch "
                               f"weight count ({len(torch_filtered)})")

        keras_grouped = self._group_layer_weights(keras_weights, reshape_to_torch=True)
        torch_grouped = self._group_layer_weights(torch_filtered, reshape_to_torch=False)
        #  TODO remove this debug code
        # for i, w in enumerate((keras_grouped, torch_grouped)):
        #     print(f"\n{'keras' if i == 0 else "torch"}")
        #     for k, v in w.items():
        #         print(k, {x: y.shape for x, y in v.items()})
        # for (kn, kw), (tn, tw) in zip(*(keras_grouped.items(), torch_grouped.items())):
        #     k_shape = kw["weight"].shape
        #     t_shape = tw["weight"].cpu().numpy().shape
        #     print(tn, "@", t_shape, "@", kn, "@", k_shape, "@", k_shape == t_shape)
        # exit()
        if len(keras_grouped) != len(torch_grouped):
            raise RuntimeError(f"Keras weight count ({len(keras_grouped)}) does not match Torch "
                               f"weight count ({len(torch_grouped)})")

        # This logic goes through the loaded torch state_dict and searches forwards through the
        # keras model for where the first weight matches and pops it. This should be reasonably
        # robust as some tensors can drift a little, but not too far. Mask layer ordering is the
        # biggest barrier, so the search is filtered if learn_mask is enabled.
        # This will fail if match is not found.
        mapped: dict[str, torch.Tensor] = {}
        for lbl, weights in torch_grouped.items():
            try:
                key = self._get_keras_key(lbl, weights["weight"].shape, keras_grouped)
                lkup = key if "separable_conv2d" not in key else key[:-2]
                print(f"{lbl}@{self._keras.layers[lkup].layer_name}@{key}@{tuple(weights['weight'].shape)}")
            except:  # TODO remove
                print()
                # print(self._mask_layers)
                print(lbl, weights["weight"].shape)
                for idx, (k, v) in enumerate(keras_grouped.items()):
                    print()
                    print(k, v["weight"].shape)
                    print("mask" in lbl and k in self._mask_layers)
                    print("mask" not in lbl and k not in self._mask_layers)
                    print(v["weight"].shape == weights["weight"].shape)
                    print(list(v) == list(weights))
                    if idx == 5:
                        break
                raise

            val = keras_grouped.pop(key)
            if key.rsplit(".", maxsplit=1)[-1].startswith("dense") and val["weight"].ndim == 2:
                self._dense_reorder(key,
                                    T.cast(dict[T.Literal["weight", "bias"], np.ndarray], val))
            if key in self._pixel_shuffler_convs:
                self._pixel_shuffle_reorder(T.cast(dict[T.Literal["weight", "bias"], np.ndarray],
                                                   val),
                                            self._pixel_shuffler_convs[key])
            logger.debug("[KerasToTorch] Mapped keras '%s' to torch '%s': %s",
                         key, lbl, val["weight"].shape)
            if key.rsplit(".", maxsplit=1)[-1].startswith("layer_scale"):  # Rename for ConvNext
                val = {"layer_scale" if k == "weight" else k: v for k, v in val.items()}
            for k, v in val.items():
                mapped[f"{lbl}.{k}"] = torch.from_numpy(v)

        # exit()
        retval = {k: mapped.get(k, v) for k, v in torch_weights.items()}  # Re-insert BN
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
