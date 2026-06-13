#!/usr/bin/env python3
"""Obtain summary information about a Faceswap Model"""
from __future__ import annotations

import logging
import os
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects, Tabulate

if T.TYPE_CHECKING:
    from torch.autograd.graph import Node
    from plugins.train.model.base import ModelPlugin

logger = logging.getLogger(__name__)


def _recurse_to_tensor(obj: list[torch.Tensor] | torch.Tensor,
                       return_attr: str | None = None
                       ) -> list[torch.Tensor] | list[T.Any] | torch.Tensor | T.Any:
    """Recurse through nested lists or tuples to obtain the contained tensors and return
    information in the same layout

    Parameters
    ----------
    obj
        The list or tuple object to recurse for the final size/tensor
    return_attr
        The attribute to return from the Tensor or ``None`` to return the tensor itself

    Returns
    -------
    The nested or final tensor(s)/tensor(s) attribute
    """
    if isinstance(obj, (tuple, list)):
        # TODO assert list/list[t]
        retval: list[T.Any] | list[torch.Tensor] = [
            _recurse_to_tensor(x, return_attr) for x in obj if x is not None]
        return retval
    if return_attr is None:
        return obj
    if return_attr == "shape":  # Convert from torch size to tuple with -1 batch dimension
        return (-1, *tuple(obj.shape[1:]))
    return getattr(obj, return_attr)


def _flatten_list(to_flatten: list[T.Any]) -> T.Generator[T.Any, None, None]:
    """Given a list of arbitrarily nested sub-lists yield each un-nested item

    Yields
    ------
    The next un-nested item from the lists
    """
    for x in to_flatten:
        if isinstance(x, list):
            yield from _flatten_list(x)
        else:
            yield x


class Layer:  # pylint:disable=too-many-instance-attributes
    """Hold's information about a Faceswap model's layer

    Parameters
    ----------
    name
        The module name of the layer
    layer_type: str
        The type (ClassName) of the layer
    is_plugin: bool
        ``True`` if the layer originates from plugins.train.model
    is_parent
        ``True`` if this module contains sub-modules
    input_shape: list[torch.Size]
        The shape of the layer input(s)
    output
        The output tensors from the layer, in output layout (eg all lists still nested)
    num_params
        The number of parameters in the layer
    num_buffers
        The number of (non-trainable) buffers in the layer
    param_bytes
        The number of bytes for the parameters
    buffer_bytes
        The number of bytes for the buffers
    requires_grad
        True if the layer requires grad
    """
    def __init__(self,  # pylint:disable=too-many-arguments,too-many-positional-arguments
                 name: str,
                 layer_type: str,
                 is_plugin: bool,
                 is_parent: bool,
                 input_shape: tuple[int, ...] | list[tuple[int, ...]],
                 output: torch.Tensor | list[torch.Tensor],
                 num_params: int,
                 num_buffers: int,
                 param_bytes: int,
                 buffer_bytes: int,
                 requires_grad: bool) -> None:
        self.name = name
        """The module name of the layer"""
        self.type = layer_type
        """The type (ClassName) of the layer"""
        self.is_plugin = is_plugin
        """``True`` if the layer originates from plugins.train.model"""
        self.is_parent = is_parent
        """``True`` if this module contains sub-modules"""
        self.input_shape = input_shape
        """The tensor information for the layer inputs"""
        self.output_shape = T.cast(tuple[int, ...] | list[tuple[int, ...]],
                                   _recurse_to_tensor(output, "shape"))
        """The output shape(s) from the layer"""
        self.num_params = num_params
        """The number of parameters in the layer"""
        self.num_buffers = num_buffers
        """The number of (non-trainable) buffers in the layer"""
        self.param_bytes = param_bytes
        """The number of bytes for the parameters"""
        self.buffer_bytes = buffer_bytes
        """The number of bytes for the buffers"""
        self.requires_grad = requires_grad
        """True if the layer requires grad"""
        self.call_count: int = 0
        """The number of times that the layer was called"""
        self.input_layers: list[str] = []
        """The input layer(s) to this layer"""
        self.grad_fn = T.cast("Node | list[Node] | None", _recurse_to_tensor(output, "grad_fn"))
        """The grad function(s) for the layer if exists and the backwards path has not been
        processed otherwise ``None``"""

        self._output_repr = (type(output), len(output))
        self._output: torch.Tensor | list[torch.Tensor] | None = output

    def __repr__(self) -> str:
        params = {k if k != "type" else "layer_type": self._output_repr if k == "output" else v
                  for k, v in self.__dict__.items()
                  if k not in ("call_count", "input_layers", "grad_fn")
                  and not k.startswith("_")}
        s_params = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    def release_references(self) -> None:
        """Release the references to the output Tensor and the grad_fn"""
        logger.debug("[Layer] releasing references for '%s'", self.name)
        del self._output
        del self.grad_fn
        self._output = None
        self.grad_fn = None


class _Structure:
    """Passes a tensor through a faceswap model to collect information about the model structure

    Parameters
    ----------
    model
        The Faceswap model plugin to parse
    """
    def __init__(self, model: ModelPlugin) -> None:
        logger.debug(parse_class_init(locals()))
        self._structure = self._get_structure(model)
        self.num_inputs = model.num_identities
        """The number of identities the model is configured for"""

    @property
    def structure(self) -> dict[str, Layer]:
        """The structure from tracing the model for the current configuration"""
        return self._structure

    def _add_forward_hook(self, summary: dict[str, Layer], name: str) -> T.Callable:
        """Add a forward hook to the model

        Parameters
        ----------
        summary
            The summary object to populate with model's layer information
        name
            The name of the module

        Returns
        -------
        The forward hook function
        """
        def hook_fn(module: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
            assert len(inputs) == 1
            layer = summary.get(
                name,
                Layer(name=name,
                      layer_type=module.__class__.__name__,
                      is_plugin=module.__module__.startswith("plugins.train.model."),
                      is_parent=len(list(module.children())) > 0,
                      input_shape=T.cast(tuple[int, ...] | list[tuple[int, ...]],
                                         _recurse_to_tensor(inputs[0], "shape")),
                      output=_recurse_to_tensor(outputs),
                      num_params=sum(p.numel() for p in module.parameters()),
                      num_buffers=sum(p.numel() for p in module.buffers()),
                      param_bytes=sum(p.numel() * p.element_size() for p in module.parameters()),
                      buffer_bytes=sum(p.numel() * p.element_size() for p in module.buffers()),
                      requires_grad=any(p.requires_grad for p in module.parameters())))
            layer.call_count += 1
            summary[name] = layer
        return hook_fn

    def _forward_trace(self, model: ModelPlugin) -> dict[str, Layer]:
        """Run a forward pass through the model and collect outputs for auditing

        Returns
        -------
        Information for each layer in the model
        """
        retval: dict[str, Layer] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []
        for name, module in model.named_modules():
            name = model.__class__.__name__ if not name else name
            hooks.append(module.register_forward_hook(self._add_forward_hook(retval, name)))
        inp = [torch.zeros([1, *model.input_shape], dtype=torch.float32)
               for _ in range(model.num_identities)]

        is_training = model.training
        model.eval()
        model(inp)

        for hook in hooks:
            hook.remove()

        if is_training:
            model.train()

        return retval

    @classmethod
    def _map_grad_fns_to_layers(cls, layers: dict[str, Layer]) -> dict[Node, str]:
        """Parse the collected layer's grad function and create a unique mapping from grad_fn back
        to original layer

        Parameters
        ----------
        layers
            The layer information from the forward trace

        Returns
        -------
        The grad_fn to layer mapping
        """
        retval: dict[Node, str] = {}
        for name, info in layers.items():
            if info.grad_fn is None:
                continue
            logger.debug("[_Structure] parsing layer '%s'. grad_fn: %s", name, info.grad_fn)
            grad_fns = info.grad_fn if isinstance(info.grad_fn, list) else [info.grad_fn]
            for f in _flatten_list(grad_fns):
                if f in retval:
                    logger.debug("[_Structure] skipping layer '%s' as seen grad_fn in '%s'",
                                 name, retval[f])
                    continue
                retval[f] = name
        return retval

    @classmethod
    def _find_producing_layer(cls,
                              grad_fn: Node | None,
                              grad_fn_to_layer: dict[Node, str],
                              visited: set[Node] | None = None) -> list[str]:
        visited = set() if visited is None else visited
        if grad_fn is None or grad_fn in visited:
            return []
        visited.add(grad_fn)

        if grad_fn in grad_fn_to_layer:
            return [grad_fn_to_layer[grad_fn]]

        results = []
        for parent_fn, _ in grad_fn.next_functions:
            results.extend(cls._find_producing_layer(parent_fn, grad_fn_to_layer, visited=visited))
        return results

    def _resolve_backwards(self, layers: dict[str, Layer]) -> None:
        """Pass through the grad functions to discover layer connectivity and update the layers
        dict in place

        Parameters
        ----------
        layers
            The layer information connected from the forward trace
        """
        grad_fn_to_layer = self._map_grad_fns_to_layers(layers)
        for name, info in layers.items():
            if info.grad_fn is None:
                continue
            grad_fns = info.grad_fn if isinstance(info.grad_fn, list) else [info.grad_fn]
            inputs = []
            for f in T.cast(list["Node"], _flatten_list(grad_fns)):
                for parent_fn, _ in f.next_functions:
                    inputs.extend(self._find_producing_layer(parent_fn, grad_fn_to_layer))
            layers[name].input_layers = list(dict.fromkeys(inputs))  # de-duped in order

    def _get_structure(self, model: ModelPlugin) -> dict[str, Layer]:
        """Process a sample tensor through the model and store information about each layer
        visited

        Returns
        -------
        Summary information for each layer in the model
        """
        layers = self._forward_trace(model)
        self._resolve_backwards(layers)
        for layer in layers.values():
            layer.release_references()
        return layers


class _Summary:
    """Generates model summary information from a collected structure

    Parameters
    ----------
    name
        The name of the model to summarize
    structure
        The traced structure of the model
    """
    def __init__(self, name: str, structure: _Structure) -> None:
        logger.debug(parse_class_init(locals()))
        self._name = name
        self._structure = structure
        self._header = [["Layer (type)", "Input Shape", "Output Shape", "Connected To", "Params"]]

    def _get_parents(self) -> dict[str, list[str]]:
        """Obtain the top-level module names and the number of instances of the module that exists
        within the model

        Returns
        -------
        The name of the parent module to list of names of modules that are of identical type for
        that module, in the order they appear within the model structure
        """
        plugin_modules = [x.name  # Modules defined within faceswap plugins
                          for x in self._structure.structure.values()
                          if x.is_plugin and x.is_parent and x.name != self._name]
        plugin_map = {}
        for name in plugin_modules:
            if not any(name.startswith(p + ".")  # Filter custom modules that are not top-level
                       for p in plugin_modules
                       if name != p):
                plugin_map.setdefault(self._structure.structure[name].type, []).append(name)
        retval = {v[0] if len(v) == 1 else os.path.commonprefix(v).rstrip("."): v
                  for v in plugin_map.values()}
        logger.debug("[_Summary] Parents: %s", retval)
        return retval

    def _generate_row(self,
                      name: str,
                      layer_type: str,
                      input_shapes: tuple[int, ...] | list[tuple[int, ...]],
                      output_shapes: tuple[int, ...] | list[tuple[int, ...]],
                      input_layers: list[str],
                      total_params: int,
                      instances: int) -> list[str]:
        """Generate a row of summary data for tabulation from the given information

        Parameters
        ----------
        name
            The name of the layer
        layer_type
            The type (class name) of the layer
        input_shapes
            The input shapes to the layer, can be nested
        output_shapes
            The output shapes to the layer, can be nested
        input_layers
            The layer names that feed this layer
        total_params
            The total number of parameters for the layer
        instances
            The number of times that this module appears in its parent module

        Returns
        -------
        The given data formatted for summary tabulation
        """
        return [
            f"{name}\n({layer_type})" if len(name) > 20 else f"{name} ({layer_type})",
            "\n".join(str(x[1:]) for x in _flatten_list([input_shapes])),
            "\n".join(str(x[1:]) for x in _flatten_list([output_shapes])),
            "\n".join(input_layers),
            (f"{total_params:,}" if instances == 1 or total_params == 0
             else "\n".join((f"Inst: {total_params:,}", f"Tot: {total_params * instances:,}")))
            ]

    def _sub_module_builder(self, module: str, instances: int) -> list[list[str]]:
        """Generate the rows for the model layers to be output to the summary table

        Parameters
        ----------
        module
            The parent module to build the row summary for
        instances
            The number of times that this module appears in its parent module

        Returns
        -------
        The rows of data to be output to the summary table
        """
        retval: list[list[str]] = []
        logger.debug("[_Summary] Building rows. module: '%s', instances: %s", module, instances)
        for name, info in self._structure.structure.items():
            if info.is_parent or "." not in name or not name.startswith(module):
                # Skip parents, top-levels and layers not belonging to given module
                logger.debug("[_Summary] '%s' Skipping layer: '%s'", module, name)
                continue
            input_layers = [tail if sep and head.isdigit() else head + sep + tail
                            for x in (x.split(".", maxsplit=1)[1]
                                      if x.startswith(f"{module}.") else x
                                      for x in info.input_layers)
                            for head, sep, tail in [x.partition(".")]]
            retval.append(self._generate_row(name[len(module) + 1:],  # strip parent + "."
                                             info.type,
                                             info.input_shape,
                                             info.output_shape,
                                             input_layers,
                                             info.num_params + info.num_buffers,
                                             instances))
        return retval

    def _top_level_builder(self, parents: list[str]) -> list[list[str]]:
        """Generate the rows for the top-level summary

        Parameters
        ----------
        parents
            The list of top-level module names to summarize for

        Returns
        -------
        The top-level rows of data to be output to the summary table
        """
        retval: list[list[str]] = []
        logger.debug("[_Summary] Building top-level rows")
        # TODO the input layers is very wrong for top-level
        for name, info in self._structure.structure.items():
            if name not in parents:
                continue
            retval.append(self._generate_row(name,
                                             info.type,
                                             info.input_shape,
                                             info.output_shape,
                                             info.input_layers,
                                             info.num_params + info.num_buffers,
                                             1))
        return retval

    def _get_param_info(self, module: str) -> tuple[int, int, float, float]:
        """Obtain the count and size in megabytes of trainable and non-trainable parameters for
        the given module

        Parameters
        ----------
        module
            The module to obtain the parameter summary for

        Returns
        -------
        trainable_parameters
            The total number of trainable parameters in the module
        non_trainable_parameters
            The total number of non-trainable parameters in the module
        trainable_megabytes
            The total size in megabytes of trainable parameters in the module
        non_trainable_megabytes
            The total size in megabytes of non trainable parameters in the module
        """
        retval = [0, 0, 0.0, 0.0]
        for name, info in self._structure.structure.items():
            if info.is_parent or "." not in name or not name.startswith(module):
                # Skip parents, top-levels and layers not belonging to given module
                logger.debug("[_Summary] '%s' Skipping layer: '%s'", module, name)
                continue
            idx = 0 if info.requires_grad else 1
            retval[idx] += info.num_params
            retval[1] += info.num_buffers
            retval[idx + 2] += info.param_bytes / (1024 ** 2)
            retval[3] += info.buffer_bytes / (1024 ** 2)
        logger.debug("[_Summary] Parameter information for '%s': %s", module, retval)
        return retval[0], retval[1], retval[2], retval[3]

    def _summarize_parameters(self,  # pylint:disable=too-many-locals
                              parameters: tuple[int, int, float, float],
                              instances: int,
                              print_fn: T.Callable[[str], T.Any]) -> None:
        """Generate the parameter count and memory summaries and output to print_fn

        Parameters
        ----------
        parameters
            the (trainable_params, non_trainable_params, trainable_bytes, non_trainable_bytes) to
            summarize
        instances
            The number of times that this module appears in model
        print_fn
            The function to print the parameter summary to
        """
        trainable, non_trainable, trainable_mb, non_trainable_mb = parameters
        total = trainable + non_trainable
        total_mb = trainable_mb + non_trainable_mb
        data: dict[str, list[list[str]]] = {}
        for title, info in zip(("Total", "Trainable", "Non-trainable"),
                               ((total, total_mb),
                                (trainable, trainable_mb),
                                (non_trainable, non_trainable_mb))):
            key = f" {title} parameters:"
            data[key] = [[f"{info[0]:,}"], [f"({info[1]:,.2f} MB)"]]
            if instances > 1:
                data[key][0] += [f"{int(info[0] * instances):,}"]
                data[key][1] += [f"({info[1] * instances:,.2f} MB)"]

        key_width = max(len(x) for x in data)
        col_widths = [max(max(map(len, col)) for col in entry) for entry in zip(*data.values())]
        for key, val in data.items():
            if len(val[0]) == 1:
                print_fn(f"{key.ljust(key_width)} "
                         f"{val[0][0].rjust(col_widths[0])} "
                         f"{val[1][0].rjust(col_widths[1])}")
                continue
            print_fn(
                f"{key.ljust(key_width)} "
                f"instance: {val[0][0].rjust(col_widths[0])} {val[1][0].rjust(col_widths[1])}, "
                f"total: {val[0][1].rjust(col_widths[1])}  {val[1][1].rjust(col_widths[1])}")

    def __call__(self, print_fn: T.Callable[[str], T.Any] | None = None) -> None:
        """Generate the preview

        Parameters
        ----------
        print_fn
            The function to print the summary to. Default: ``None`` (print to console)
        """
        print_fn = print if print_fn is None else print_fn
        parents = self._get_parents()
        all_params: dict[str, tuple[int, int, float, float]] = {}
        for name, modules in parents.items():
            count = len(modules)
            lookup = modules[0]
            rows = self._header + self._sub_module_builder(lookup, count)
            call_count = self._structure.structure[lookup].call_count
            print_fn(f"Model: {self._name}.{name} (Instances: {count}, "
                     f"Calls per instance: {call_count})")
            Tabulate(rows, padding=1, just=["ljust", "rjust", "rjust", "ljust", "rjust"])(print_fn)
            params = self._get_param_info(lookup)
            self._summarize_parameters(params, count, print_fn)
            all_params[name] = T.cast(tuple[int, int, float, float],
                                      tuple(x * count for x in params))

        parent_list = [y for x in parents.values() for y in x]
        rows = rows = self._header + self._top_level_builder(parent_list)
        sum_params = T.cast(tuple[int, int, float, float],
                            tuple(map(sum, zip(*all_params.values()))))
        print_fn(f"Model: {self._name} (Inputs: {self._structure.num_inputs})")
        Tabulate(rows, padding=1, just=["ljust", "rjust", "rjust", "ljust", "rjust"])(print_fn)
        self._summarize_parameters(sum_params, 1, print_fn)


class Info:
    """Obtain summary information about a Faceswap Model.

    Note: The information collected is correct at the time of calling. No model references are held
    so data is not updated if the model is changed subsequent to creating this class instance"""
    def __init__(self, model: ModelPlugin) -> None:
        logger.debug(parse_class_init(locals()))
        self._model_info = (model.__class__.__name__, repr(model))
        """The (name, repr) of the model"""
        self._device = next(model.named_parameters())[1].device
        self._structure = _Structure(model)
        self._summary = _Summary(model.__class__.__name__, self._structure)
        self._output_shapes = []
        self._input_shapes = []
        self._input_size: int = 0
        self._output_size: int = 0

    def __repr__(self) -> str:
        """Better logging"""
        return f"{self.__class__.__name__}(model={self._model_info[1]})"

    @property
    def device(self) -> torch.Device:
        """The device that the model resides on"""
        return self._device

    @property
    def output_shapes(self) -> list[list[tuple[int, int, int]]]:
        """The output sizes for each side of the model, excluding batch dimension. List of length
        num_identities, sub-list of length num_outputs, in shape (C, H, W)"""
        if not self._output_shapes:
            sizes = T.cast(list[list[torch.Size]],
                           self._structure.structure[self._model_info[0]].output_shape)
            assert isinstance(sizes, list)
            self._output_shapes = [[out[1:] for out in side] for side in sizes]
        return T.cast(list[list[tuple[int, int, int]]], self._output_shapes)

    @property
    def input_shapes(self) -> list[tuple[int, int, int]]:
        """The input sizes to the model. List of length num_identities containing inputs shaped
        (3, H, W)"""
        if not self._input_shapes:
            inputs = self._structure.structure[self._model_info[0]].input_shape
            assert isinstance(inputs, list)
            self._input_shapes = [i[1:] for i in inputs]
        return T.cast(list[tuple[int, int, int]], self._input_shapes)

    @property
    def input_size(self) -> int:
        """The pixel input size to the model, regardless of side"""
        if not self._input_size:
            input_sizes = set(x[1] for x in self.input_shapes)
            assert len(input_sizes) == 1, f"Multiple input sizes not supported. Got {input_sizes}"
            self._input_size = input_sizes.pop()
        return self._input_size

    @property
    def output_size(self) -> int:
        """The largest pixel image output size from the model, regardless of side"""
        if not self._output_size:
            max_out_sizes = set(max(out[1] for out in side if out[0] != 1)
                                for side in self.output_shapes)
            assert len(max_out_sizes) == 1, (
                f"All sides should have the same output size. Got {max_out_sizes}")
            self._output_size = max_out_sizes.pop()
        return self._output_size

    @property
    def structure(self) -> dict[str, Layer]:
        """The parsed model structure"""
        return self._structure.structure

    def summary(self, print_fn: T.Callable[[str], T.Any] | None = None) -> None:
        """Output the model summary table

        Parameters
        ----------
        print_fn
            The function to print the summary to. Default: ``None`` (print to console)
        """
        self._summary(print_fn)


__all__ = get_module_objects(__name__)
