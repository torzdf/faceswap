#! /usr/env/bin/python3
"""Handles the collation, weighting masking and calculation of the selected Loss functions for
training Faceswap models"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import typing as T

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.losses import get_loss_function
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from .data import BatchMeta

logger = logging.getLogger(__name__)


@dataclass
class BatchLoss:
    """Dataclass for holding Loss values for a batch of data"""
    unweighted: list[dict[str, torch.Tensor]]
    """For each side output, the unweighted loss scalars for each function for each item in the
    batch"""
    weighted: list[dict[str, torch.Tensor]]
    """For each side output, the weighted loss scalars for each function for each item in the
    batch"""
    mask: torch.Tensor | None = None
    """The loss scalar for the mask for each item in the batch if learn_mask is selected otherwise
    ``None``. Default: ``None``"""
    _total: torch.Tensor | None = field(init=False, default=None)

    @property
    def total(self) -> torch.Tensor:
        """The total single weighted loss scalar for all items in the batch for backprop"""
        if self._total is None:
            total = T.cast(torch.Tensor, sum(sum(y.mean() for y in x.values())
                                             for x in self.weighted))
            if self.mask is not None:
                total += self.mask.mean()
            self._total = total
        return self._total

    def to_cpu(self) -> T.Self:
        """Detaches all contained loss values and moves them to CPU

        Returns
        -------
        This object with all tensors detached and moved to CPU
        """
        self._total = None if self._total is None else self._total.detach().cpu()
        self.unweighted = [{k: v.detach().cpu() for k, v in x.items()} for x in self.unweighted]
        self.weighted = [{k: v.detach().cpu() for k, v in x.items()} for x in self.weighted]
        self.mask = None if self.mask is None else self.mask.detach().cpu()
        return self


@dataclass
class LossConfig:
    """Dataclass to hold configuration options for Loss Functions

    Parameters
    ----------
    functions
        List of lost function names from configuration file to collate for loss calculation
    weights
        List of weights, corresponding to the the list of functions, to apply to each loss
        function
    use_mask
        ``True`` if loss should be masked as `penalize mask loss` has been selected
    eye_multiplier
        The amount of extra weighting to apply to the eye area
    mouth_multiplier
        The amount of extra weighting to apply to the mouth area
    mask_loss
        The loss function to use if learn_mask is enabled
    identity_loss
        The identity loss functions to use
    """
    functions: list[str]
    """List of lost function names from configuration file to collate for loss calculation"""
    weights: list[float]
    """List of weights, corresponding to the the list of functions, to apply to each loss
    function"""
    use_mask: bool
    """``True`` if loss should be masked as `penalize mask loss` has been selected"""
    eye_multiplier: float
    """The amount of extra weighting to apply to the eye area"""
    mouth_multiplier: float
    """The amount of extra weighting to apply to the mouth area"""
    mask_loss: str | None
    """The loss function to use if learn_mask is enabled"""
    identity_backend: T.Literal["ir-50", "ir-101"] | None
    """The backend to use for the identity loss functions. ``None`` for no identity loss"""
    identity_weight: float = 1.0
    """The weighting to use for identity loss"""


class LossCollator(nn.Module):  # pylint:disable=too-many-instance-attributes
    """Compiles the chosen loss functions and calculates the values in the training loop

    Parameters
    ----------
    config
        The loss configuration settings
    color_order
        The color order that the model is training in
    smallest_output
        The smallest output from the model. Required for initializing some loss functions
    image_counts
        The number of images contained within each dataset in input order
    """
    def __init__(self,
                 config: LossConfig,
                 color_order: T.Literal["bgr", "rgb"],
                 smallest_output: int,
                 image_counts: tuple[int, ...]) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._config = config
        self._color_order: T.Literal["bgr", "rgb"] = color_order
        self._smallest_output = smallest_output
        self._functions, self._weights = self._configure_functions()
        self._spatial, self._non_spatial = self._get_function_types()

        self._mask_loss_function = (
            None if config.mask_loss is None
            else self._functions[config.mask_loss] if config.mask_loss in self._functions
            else get_loss_function(config.mask_loss)
            )
        self._identity_function = (
            None if config.identity_backend is None
            else get_loss_function("identity",
                                   color_order=color_order,
                                   kwargs={"backbone": config.identity_backend,
                                           "image_counts": image_counts})
            )
        self._image_idx: int | None = None
        """The index of the final target image within y_true_all"""

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = ", ".join(f"{k[1:]}={repr(v)}" for k, v in self.__dict__.items()
                           if k in ("_config", "_color_order", "_smallest_output", "image_count"))
        return f"{self.__class__.__name__}({params})"

    def _configure_functions(self) -> tuple[nn.ModuleDict, dict[str, float]]:
        """Configure the selected loss functions and send to the correct device

        Returns
        -------
        functions
            ModuleDict of configured loss functions
        weights
            dict of loss names to weight to apply

        Raises
        ------
        ValueError
            If the number of function names and loss weights do not correspond
        """
        if len(self._config.functions) != len(self._config.weights):
            raise ValueError(f"Number of loss functions ({len(self._config.functions)}) and "
                             f"weights ({len(self._config.weights)}) should match")

        functions = nn.ModuleDict()
        weight_dict: dict[str, float] = {}
        for name, weight in zip(self._config.functions, self._config.weights):
            if name is None or name == "none" or weight <= 0.0:
                continue
            functions[name] = get_loss_function(name, self._color_order)
            weight_dict[name] = weight

        logger.debug("[Loss] Configured loss functions: %s",
                     {k: (functions[k].__class__.__name__, weight_dict[k]) for k in functions})
        return functions, weight_dict

    def _get_function_types(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Run a small tensor through each of the selected loss functions to determine which are
        spatial or non-spatial loss functions

        Returns
        -------
        spatial
            Tuple of loss names that produce spatial output
        non_spatial
            Tuple of loss names that produce non-spatial output
        """
        size = self._smallest_output
        dummy_a = torch.rand((1, 3, size, size), dtype=torch.float32)
        dummy_b = torch.rand((1, 3, size, size), dtype=torch.float32)
        spatial: list[str] = []
        non_spatial: list[str] = []
        for name, func in self._functions.items():
            out = func(dummy_a, dummy_b)
            dims = out.ndim
            if dims not in (1, 4):
                raise RuntimeError("Loss functions should return either spatial output per item "
                                   f"(N, C, H, W) (4 dims) or scalar per item (N, ) (1 dim). "
                                   f"Got {dims} dims for '{name}'")
            dst = spatial if dims == 4 else non_spatial
            dst.append(name)

        logger.debug("[Loss] spatial: %s, non-spatial: %s", spatial, non_spatial)
        return tuple(spatial), tuple(non_spatial)

    def _get_spatial_loss(self,
                          y_true: torch.Tensor,
                          y_pred: torch.Tensor,
                          meta: BatchMeta,
                          index: int) -> dict[str, torch.Tensor]:
        """Obtain the unweighted loss values for the spatial loss functions

        Parameters
        ----------
        y_true
            The ground truth batch of images
        y_pred
            The batch of model predictions
        meta
            The meta information for the batch
        index
            The output index for obtaining the correct meta data for the processing output

        Returns
        -------
        The unweighted loss for each spatial loss function with masks and multipliers applied
        """
        retval: dict[str, torch.Tensor] = {}
        for name in self._spatial:
            loss: torch.Tensor = self._functions[name](y_true, y_pred)
            if self._config.use_mask and meta.mask_face is not None:
                loss *= meta.mask_face[index]
            if self._config.eye_multiplier > 1. and meta.mask_eye is not None:
                loss += loss * meta.mask_eye[index] * self._config.eye_multiplier
            if self._config.mouth_multiplier > 1. and meta.mask_mouth is not None:
                loss += loss * meta.mask_mouth[index] * self._config.mouth_multiplier
            retval[name] = loss.mean(dim=tuple(range(1, loss.ndim)))
        logger.trace("[Loss] Spatial loss: %s", retval)  # type:ignore[attr-defined]
        return retval

    def _get_masked_inputs(self,
                           y_true: torch.Tensor,
                           y_pred: torch.Tensor,
                           meta: BatchMeta,
                           index: int
                           ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[float]]:
        """For non spatial loss functions the inputs need to be masked for each supplied masks

        Parameters
        ----------
        y_true
            The ground truth batch of images
        y_pred
            The batch of model predictions
        meta
            The meta information for the batch
        index
            The output index for obtaining the correct meta data for the processing output

        Returns
        -------
        inputs
            The (y_true, y_pred) inputs to the loss function for each supplied mask
        weights
            The weight to be applied for each masked input
        """
        weights = [1.0]
        assert meta.mask_face is not None
        face_mask = meta.mask_face[index]
        inputs = [(y_true * face_mask, y_pred * face_mask)]
        for m_type in ("eye", "mouth"):
            masks: list[torch.Tensor] | None = getattr(meta, f"mask_{m_type}")
            if masks is None:
                continue
            mask = masks[index]
            inputs.append((y_true * mask, y_pred * mask))
            weights.append(self._config.eye_multiplier if m_type == "eye"
                           else self._config.mouth_multiplier)
        logger.trace("[Loss] masked inputs: %s, weights: %s",  # type:ignore[attr-defined]
                     [[x.shape for x in i] for i in inputs], weights)
        return inputs, weights

    def _get_non_spatial_loss(self,
                              y_true: torch.Tensor,
                              y_pred: torch.Tensor,
                              meta: BatchMeta,
                              index: int) -> dict[str, torch.Tensor]:
        """Obtain the unweighted loss values for the non-spatial loss functions

        Parameters
        ----------
        y_true
            The ground truth batch of images
        y_pred
            The batch of model predictions
        meta
            The meta information for the batch
        index
            The output index for obtaining the correct meta data for the processing output

        Returns
        -------
        The unweighted loss for each non-spatial loss function with masks and multipliers applied
        """
        retval: dict[str, torch.Tensor] = {}
        if not self._config.use_mask:
            inputs = [(y_true, y_pred)]
            weights = [1.0]
        else:
            inputs, weights = self._get_masked_inputs(y_true, y_pred, meta, index)

        for name in self._non_spatial:
            losses = torch.stack([self._functions[name](inp_true, inp_pred) * weight
                                 for weight, (inp_true, inp_pred) in zip(weights, inputs)])
            retval[name] = losses.sum(dim=0)

        logger.trace("[Loss] Non-spatial loss: %s", retval)  # type:ignore[attr-defined]
        return retval

    def _get_identity_loss(self,
                           y_true_all: list[torch.Tensor],
                           swap_pred: list[torch.Tensor],
                           meta: BatchMeta) -> torch.Tensor:
        """Obtain the unweighted loss values for the identity loss function

        Parameters
        ----------
        y_true_all
            The ground truth batch of images for all outputs for a side of the model
        swap_pred
            The batch of swapped model predictions for each model input
        meta
            The meta information for the batch

        Returns
        -------
        The unweighted loss for the identity loss function
        """
        assert self._identity_function is not None
        assert len(meta.side_index) == 1
        if self._image_idx is None:
            self._image_idx = next(i for i in reversed(range(len(y_true_all)))
                                   if y_true_all[i].shape[1] != 1)
            logger.debug("[LossCollator] Set target image index to %s", self._image_idx)

        y_true = y_true_all[self._image_idx]
        # TODO receive stacked?
        id_loss = [self._identity_function(y_true,
                                           y_pred,
                                           meta.side_index[0],
                                           meta.image_indices)
                   for y_pred in swap_pred]

        return torch.mean((torch.stack(id_loss)), dim=0)

    def forward(self,
                y_true_all: list[torch.Tensor],
                y_pred_all: list[torch.Tensor],
                meta: BatchMeta,
                swap_pred: list[torch.Tensor]) -> BatchLoss:
        """Call the loss functions, reduce to batch dimension, apply masks and weighting and obtain
        the weighted and unweighted per function values and the weighted total loss scalar

        Parameters
        ----------
        y_true_all
            The ground truth batch of images for all outputs for a side of the model
        y_pred_all
            The batch of model predictions for all outputs for a side of the model
        meta
            The meta information for the batch
        swap_pred
            The swap predictions from the model if an identity loss is being used

        Returns
        -------
        The loss scalars for the batch
        """
        all_unweighted: list[dict[str, torch.Tensor]] = []
        all_weighted: list[dict[str, torch.Tensor]] = []
        mask_loss = None

        # TODO remove once channels first
        y_true_all = [x.permute(0, 3, 1, 2) for x in y_true_all]
        y_pred_all = [x.permute(0, 3, 1, 2) for x in y_pred_all]
        swap_pred = [x.permute(0, 3, 1, 2) for x in swap_pred]

        for idx, (y_true, y_pred) in enumerate(zip(y_true_all, y_pred_all)):

            if y_true.shape[1] == 1:
                assert self._mask_loss_function is not None
                mask_loss = T.cast(torch.Tensor, self._mask_loss_function(y_true, y_pred))
                mask_loss = mask_loss.mean(dim=tuple(range(1, mask_loss.ndim)))
                continue

            unweighted = self._get_spatial_loss(y_true, y_pred, meta, idx)
            unweighted |= self._get_non_spatial_loss(y_true, y_pred, meta, idx)
            all_unweighted.append(unweighted)
            all_weighted.append({k: v * self._weights[k] for k, v in unweighted.items()})

        if self._identity_function is not None:
            identity_loss = self._get_identity_loss(y_true_all, swap_pred, meta)
            all_unweighted[-1]["identity"] = identity_loss
            all_weighted[-1]["identity"] = identity_loss * self._config.identity_weight

        retval = BatchLoss(unweighted=all_unweighted,
                           weighted=all_weighted,
                           mask=mask_loss)
        logger.trace("[Loss] %s", retval)  # type:ignore[attr-defined]
        return retval


__all__ = get_module_objects(__name__)
