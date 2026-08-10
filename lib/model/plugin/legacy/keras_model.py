#! /usr/env/bin/python3
""" Holds layer information and weights from a .keras model file """
from __future__ import annotations

import io
import json
import logging
import os
import re
import typing as T
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field

import h5py
import numpy as np
import torch

from lib.logger import parse_class_init
from lib.utils import camel_to_snake_case, get_module_objects


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


_ENC_PREFIX = "layers.functional.layers.functional.layers."


class LayerSorter:
    """ Sorts keras layers when graph order does not correspond to Torch build order. Weights
    matcher works by finding the next available Keras weights of the same shape as the currently
    processing Torch weights, so the order does not need to be exact, but it needs to be good
    enough for this algorhythm to select the correct weights

    Parameters
    ----------
    state
        The state dictionary for the keras model being migrated
    """
    def __init__(self, state: dict[str, T.Any]) -> None:
        logger.debug(parse_class_init(locals()))
        self._model = state["name"]
        self._encoder = state["config"]["enc_architecture"] if self._model == "phaze_a" else None
        self._functions = {"iae": self._iae_reorder,
                           "inception_resnet_v2": self._inception_reorder,
                           "inception_v3": self._inception_reorder,
                           "nasnet_large": self._nasnet_reorder,
                           "nasnet_mobile": self._nasnet_reorder,
                           "phaze_a": self._phaze_a_reorder,
                           "xception": self._xception_reorder}

    @classmethod
    def _order_layers(cls,
                      layers: dict[str, LayerInfo],
                      order: list[str]) -> dict[str, LayerInfo]:
        """ Re-orders the layers according to the provided order list. Layers not in the order
        list are returned in their original positions

        Parameters
        ----------
        layers
            The layers to be re-ordered
        order
            list of layer names that must be ordered in the returned dictionary

        Returns
        -------
        The layers re-ordered according to the provided order list
        """
        ordered = {k: v
                   for o in order
                   for k, v in layers.items()
                   if k == o}
        order_iter = iter(ordered)

        retval: dict[str, LayerInfo] = {}
        for k, v in layers.items():
            if k in ordered:
                next_k = next(order_iter)
                retval[next_k] = layers[next_k]
            else:
                retval[k] = v

        return retval

    def _iae_reorder(self, layers: dict[str, LayerInfo]) -> dict[str, LayerInfo]:
        """ Re-orders the intermediate layers for IAE models. Inters graph in order [both, B, A]
        but build in order [A, B, Both]

        Parameters
        ----------
        layers
            The layers to be re-ordered

        Returns
        -------
        The re-ordered layers
        """
        order = [layer
                 for prefix in ["layers.functional_2.", "layers.functional_1."]
                 for layer in layers
                 if layer.startswith(prefix)]
        return self._order_layers(layers, order)

    def _inception_reorder(self, layers: dict[str, LayerInfo], v3=False) -> dict[str, LayerInfo]:
        """ Re-orders imported layer names from Keras Applications InceptionResNet models from
        graph order to build order. Fairly straightforward as default naming is used for all
        problematic layers

        Parameters
        ----------
        layers
            The layers to be re-ordered
        v3
            ``True`` if the model is InceptionV3, ``False`` if InceptionResNetV2.
            Default: ``False``

        Returns
        -------
        The re-ordered layers
        """
        rename = {"mixed9_0": "mixed9",  # 1 badly named layer in inception v3
                  "mixed9": "mixed9_1",
                  "mixed9_1": "mixed9_2"} if v3 else {}

        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for key, layer in layers.items():
            name = layer.layer_name
            if not key.startswith(_ENC_PREFIX) or name.startswith("input_layer"):
                continue

            name = rename.get(name, name)
            head, sep, tail = name.rpartition("_")
            if sep and tail.isdigit():
                groups[head].append((int(tail), key))
            else:
                groups[name].append((0, key))

        order = [key
                 for group in groups.values()
                 for _, key in sorted(group)]
        return self._order_layers(layers, order)

    def _nasnet_reorder(self, layers: dict[str, LayerInfo]  # pylint:disable=too-many-locals
                        ) -> dict[str, LayerInfo]:
        """ Re-orders imported layer names from Keras Applications NasNet from graph order to build
        order. Some fairly arbitrary re-ordering occurs, but fortunately layer labelling makes this
        a bit easier

        Parameters
        ----------
        layers
            The layers to be re-ordered

        Returns
        -------
        The re-ordered layers
        """
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
        return self._order_layers(layers, sorted_keys)

    def _phaze_a_reorder(self, layers: dict[str, LayerInfo]) -> dict[str, LayerInfo]:
        """ Phaze-A has several configurations that require re-ordering

        Parameters
        ----------
        layers
            The layers to be re-ordered

        Returns
        -------
        The re-ordered layers
        """
        if self._encoder in self._functions:
            logger.debug("[LayerSorter] Sorting Phaze-A encoder: %s", self._encoder)
            kwargs = {"v3": True} if self._encoder == "inception_v3" else {}
            layers = self._functions[self._encoder](layers, **kwargs)

        return layers

    def _xception_reorder(self, layers: dict[str, LayerInfo]) -> dict[str, LayerInfo]:
        """ Re-orders imported layer names from Keras Applications Xception from graph order to
        build order. Skip layers need to be built prior to separable conv within each block

        Parameters
        ----------
        layers
            The layers to be re-ordered

        Returns
        -------
        The re-ordered layers
        """
        current_id = None
        order = []
        skips = []
        block = []

        for k, v in layers.items():
            name = v.layer_name

            if not k.startswith(_ENC_PREFIX) or name.startswith("input_layer"):
                continue

            if name.startswith("block"):
                block_id = int(name.split("_")[0][5:])
                if current_id is not None and block_id != current_id:
                    logger.debug("[LayerSorter] Sorted xception block_id %s: %s",
                                 current_id, [layers[k].layer_name for k in skips + block])
                    order.extend(skips + block)
                    skips.clear()
                    block.clear()

                current_id = block_id
                block.append(k)
            else:
                skips.append(k)

        logger.debug("[LayerSorter] Sorted xception block_id %s: %s",
                     current_id, [layers[k].layer_name for k in skips + block])
        order.extend(skips + block)
        return self._order_layers(layers, order)

    def sort(self, layers: dict[str, LayerInfo]) -> dict[str, LayerInfo]:
        """ Re-order the layers from Keras graph order to Keras construction order for those models
        which require it for weight porting

        Parameters
        ----------
        layers
            The layers of the model that require reordering

        Returns
        -------
        The reordered layers
        """
        if self._model not in self._functions:
            return layers
        logger.debug("[LayerSorter] Re-ordering layers: %s", self._model)
        return self._functions[self._model](layers)


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
        only the first output name is required

        Parameters
        ----------
        config
            A keras config dictionary

        Returns
        -------
        The name of the sub-model's output layer
        """
        out_layers = config["config"]["output_layers"]
        return out_layers[0][0] if isinstance(out_layers[0], list) else out_layers[0]

    @classmethod
    def _map_sub_model_input(cls,
                             config: dict[str, T.Any],
                             mapping: dict[str, str],
                             outputs: dict[str, str],
                             is_sequential: bool) -> None:
        """ Map a sub-model's internal input layer name to whatever produces its input from the
        outer graph.

        Parameters
        ----------
        config
            A keras config dictionary
        mapping
            Mapping of keras layer names to standardized weight names
        outputs
            Mapping of keras functional sub-model names to their output layer names
        is_sequential
            ``True`` if the input is a sequential model
        """
        if is_sequential and config.get("inbound_nodes"):
            return  # top-level Sequential with no outer context to resolve

        inbound_set: set[str] = set()
        for node in config["inbound_nodes"]:
            for args in node["args"]:
                args = args if isinstance(args, list) else [args]
                inbound_set.update(a["config"]["keras_history"][0] for a in args)
        inbounds = list(inbound_set)

        if is_sequential:
            assert len(inbounds) == 1  # Sequential models with multiple inputs not handled
            in_names = inbounds
        else:
            in_layers = config["config"]["input_layers"]
            in_layers = in_layers if isinstance(in_layers[0], list) else [in_layers]
            in_names = [i[0] for i in in_layers]

        # Resolve a keras inbound node name to its standardized label, redirecting through the sub-
        # model's recorded output if 'name' is itself a sub-model.  We only need first input if
        # multi for our purposes, so we zip to shortest (in_names in this case)
        for in_name, name in zip(in_names, inbounds):
            mapped = mapping.get(outputs.get(name, name), name)
            logger.debug("[KerasConfigParser] Mapping '%s' to '%s' for sub-model '%s'",
                         in_name, mapped, config["name"])
            mapping[in_name] = mapped

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
            if not is_sequential:
                outputs[name] = cls._sub_model_output_name(config)
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
        Sequential models

        Parameters
        ----------
        config
            A keras config dictionary
        mapping
            Mapping of keras layer names to standardized weight names
        prev_label
            The dst_label of the preceding layer (used for Sequential models where inbound_nodes
            are absent)

        Returns
        -------
        Mapping of producer label to input shape
        """
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
                    if not isinstance(tensor, dict) or tensor.get("class_name") == "__slice__":
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
        self.layers = LayerSorter(self.state).sort(self.layers)

        lookup: dict[str, list[str]] = {}
        # Remove normalization weights from the beginning of EffNet
        kap_enc_weights = [k for k in weights
                           if k.startswith("layers.functional.layers.functional.layers.")]

        for k in reversed(kap_enc_weights):
            if ".layers.normalization.vars." not in k:  # Always at end of encoder list
                break
            logger.debug("[KerasModel] Removing KApp normalization weights: '%s'", k)
            del weights[k]

        for k in weights:
            max_split = 3 if "multi_head_attention" in k else 2
            lookup.setdefault(k.rsplit(".", maxsplit=max_split)[0], []).append(k)

        retval: dict[str, np.ndarray] = {}
        for k in self.layers:
            if k in lookup:
                keys = lookup[k]
                for key in keys:
                    retval[key] = weights.pop(key)

        assert len(weights) == 0, f"Not all weights mapped. Remaining: {len(weights)}"
        return retval

    def _load_state_file(self) -> dict[str, T.Any]:
        """ Load the legacy state file

        Returns
        -------
        The loaded Faceswap state dictionary from the keras model file
        """
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

            self.layers = KerasConfigParser.flatten(json.loads(z_file.read("config.json")))
            logger.debug("[KerasModel] Flattened model layers: %s", self.layers)

            weights = h5py.File(io.BytesIO(z_file.read("model.weights.h5")), "r")
            self.weights = self._sort_weights(self._get_weights(T.cast(h5py.Group,
                                                                       weights["layers"])))
            logger.debug("[KerasModel] Loaded weights: %s",
                         {k: v.shape for k, v in self.weights.items()})

            if "optimizer.pt" in name_list:
                self._optimizer = torch.load(io.BytesIO(z_file.read("optimizer.pt")),
                                             map_location="cpu")
                logger.debug("[KerasModel] Loaded optimizer state: %s",
                             {k: v if k == "version" else type(v)
                              for k, v in self._optimizer.items()})


__all__ = get_module_objects(__name__)
