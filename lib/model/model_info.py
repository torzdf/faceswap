#!/usr/bin/env python3
"""Obtain summary information about a Faceswap Model"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from plugins.train.model.base import ModelPlugin

logger = logging.getLogger(__name__)


@dataclass
class Layer:
    """Holds information about a Faceswap model's layer"""
    name: str
    """The module name of the layer"""
    type: str
    """The type (ClassName) of the layer"""
    input_shapes: list[torch.Size]
    """The shape of the inputs to the layer"""
    output_shapes: torch.Size | list[torch.Size]
    """The output shapes from the layer"""
    call_count: int = 0
    """The number of times that the layer was called"""


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

    def _recurse_to_tensor(self, obj: list[torch.Tensor] | torch.Tensor
                           ) -> list[torch.Size] | torch.Size:
        """Recurse through nested lists or tuples to obtain the contained tensors and return
        information in the same layout

        Parameters
        ----------
        obj
            The list or tuple object to recurse or the final size

        Returns
        -------
        The nested or final tensor size(s)
        """
        if isinstance(obj, (tuple, list)):
            # TODO assert list/list[t]
            return T.cast(list[torch.Size],
                          [self._recurse_to_tensor(x) for x in obj if x is not None])
        return T.cast(torch.Tensor, obj).shape[1:]

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
                                      type=module.__class__.__name__,
                                      input_shapes=T.cast(list[torch.Size],  # TODO confirm
                                                          self._recurse_to_tensor(inputs[0])),
                                      output_shapes=self._recurse_to_tensor(outputs)))
            layer.call_count += 1
            summary[name] = layer
        return hook_fn

    def _get_structure(self, model: ModelPlugin) -> dict[str, Layer]:
        """Process a sample tensor through the model and store information about each layer
        visited

        Returns
        -------
        Summary information for each layer in the model
        """
        # TODO
        summary: dict[str, Layer] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []
        for name, module in model.named_modules():
            name = model.__class__.__name__ if not name else name
            hooks.append(module.register_forward_hook(self._add_forward_hook(summary, name)))
        inp = [torch.zeros([1, *model.input_shape], dtype=torch.float32)
               for _ in range(model.num_identities)]
        # TODO put model back in correct state
        model.eval()
        model(inp)

        for hook in hooks:
            hook.remove()
        return summary


class Info:
    """Obtain summary information about a Faceswap Model"""
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
                           self._structure.structure[self._model_info[0]].output_shapes)
            assert isinstance(sizes, list)
            self._output_shapes = [[tuple(x for x in out) for out in side] for side in sizes]
        return T.cast(list[list[tuple[int, int, int]]], self._output_shapes)

    @property
    def input_shapes(self) -> list[tuple[int, int, int]]:
        """The input sizes to the model. List of length num_identities containing inputs shaped
        (3, H, W)"""
        if not self._input_shapes:
            inputs = self._structure.structure[self._model_info[0]].input_shapes
            self._input_shapes = [tuple(inp for inp in side) for side in inputs]
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
    def summary(self) -> str:  # TODO
        header = str.format("{0:<30}| {1:<30}| {2:<30}|",
                            "Module Name", "Input Size", "Output Size")
        retval = f"{header}\n" + "-" * len(header) + "\n"
        for name, summary in self._summary.items():
            inp = summary["input"]
            out = summary["output"]
            if isinstance(inp, list):
                inp = f"[{', '.join(x for x in inp)}]"
            if isinstance(out, list):
                out = f"[{', '.join(x for x in out)}]"

            retval += (f"{name:<30}|{inp:<30}|{out:<30}|\n")

        return retval


__all__ = get_module_objects(__name__)


if __name__ == "__main__":  # TODO remove
    from plugins.train.model.original import Original
    from lib.logger import log_setup
    log_setup("INFO", "", "")
    m = Original()
    i = Info(m)

    # for n, l in i._structure.structure.items():
    #     print(n, l.call_count, type(l.input_shapes), type(l.output_shapes))
    print(i)
    print(i.device)
    # for k, v in i.summary.items():
    #     print(k, v)
    # print(i.output_sizes)
