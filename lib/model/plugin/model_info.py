#!/usr/bin/env python3
"""Obtain summary information about a Faceswap Model"""
from __future__ import annotations

import logging
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
    is_parent
        ``True`` if this module contains sub-modules
    input_shape: list[torch.Size]
        The shape of the layer input(s)
    output
        The output tensors from the layer, in output layout (eg all lists still nested)
    num_params
        The number of parameters in the layer
    requires_grad
        True if the layer requires grad
    """
    def __init__(self,
                 name: str,
                 layer_type: str,
                 is_parent: bool,
                 input_shape: tuple[int, ...] | list[tuple[int, ...]],
                 output: torch.Tensor | list[torch.Tensor],
                 num_params: int,
                 requires_grad: bool) -> None:
        self.name = name
        """The module name of the layer"""
        self.type = layer_type
        """The type (ClassName) of the layer"""
        self.is_parent: bool = is_parent
        """``True`` if this module contains sub-modules"""
        self.input_shape = input_shape
        """The tensor information for the layer inputs"""
        self.output_shape = T.cast(tuple[int, ...] | list[tuple[int, ...]],
                                   _recurse_to_tensor(output, "shape"))
        """The output shape(s) from the layer"""
        self.num_params = num_params
        """The number of parameters in the """
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
                  if k in ("name", "type", "is_parent", "input_shape", "output",
                           "num_params", "requires_grad")}
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
            layer = summary.get(name,
                                Layer(name=name,
                                      layer_type=module.__class__.__name__,
                                      is_parent=len(list(module.children())) > 0,
                                      input_shape=T.cast(tuple[int, ...] | list[tuple[int, ...]],
                                                         _recurse_to_tensor(inputs[0], "shape")),
                                      output=_recurse_to_tensor(outputs),
                                      num_params=sum(p.numel() for p in module.parameters()),
                                      requires_grad=any(p.requires_grad
                                                        for p in module.parameters())))
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
        # TODO Breakdowns by is_parent
        print_fn = print if print_fn is None else print_fn
        print_fn(f"Model: {self._model_info[0]}")
        rows = [["Layer (type)", "Input Shape", "Output Shape", "Connected To", "Params"]]
        rows.extend([f"{layer}\n({info.type})" if len(layer) > 20 else f"{layer} ({info.type})",
                     "\n".join(str(x[1:]) for x in _flatten_list([info.input_shape])),
                     "\n".join(str(x[1:]) for x in _flatten_list([info.output_shape])),
                     "\n".join(info.input_layers),
                     f"{info.num_params:,}"]
                    for layer, info in self._structure.structure.items()
                    if not info.is_parent)
        train_params = sum(v.num_params for v in self._structure.structure.values()
                           if not v.is_parent and v.requires_grad)
        non_train_params = sum(v.num_params for v in self._structure.structure.values()
                               if not v.is_parent and not v.requires_grad)
        Tabulate(rows, padding=0, just=["ljust", "rjust", "rjust", "ljust", "rjust"])(print_fn)
        print_fn(f"Total parameters: {train_params + non_train_params:,}")
        print_fn(f"Trainable parameters: {train_params:,}")
        print_fn(f"Non-trainable parameters: {non_train_params:,}")


__all__ = get_module_objects(__name__)
