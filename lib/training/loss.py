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
    from lib.align.constants import CenteringType
    from .data import BatchMeta

logger = logging.getLogger(__name__)


# TODO y_true/y_pred order check

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

    def get_contributions(self) -> dict[T.Literal["unweighted", "weighted"],
                                        dict[str, torch.Tensor]]:
        """Obtain the contributions of each loss function to the total loss score for both weighted
        both weighted and unweighted scores

        Returns
        -------
        weighted and unweighted total contributions to the final loss cost
        """
        unweighted = {k: T.cast(torch.Tensor, sum(d[k].mean() for d in self.unweighted)).detach()
                      for k in self.unweighted[0]}
        weighted = {k: T.cast(torch.Tensor, sum(d[k].mean() for d in self.weighted)).detach()
                    for k in self.weighted[0]}
        if "identity" in list(self.unweighted)[-1]:
            unweighted["identity"] = self.unweighted[-1]["identity"].mean().detach()
            weighted["identity"] = self.weighted[-1]["identity"].mean().detach()
        return {"unweighted": unweighted, "weighted": weighted}

    def detach(self) -> T.Self:
        """Detaches all contained loss values"""
        self._total = None if self._total is None else self._total.detach()
        self.unweighted = [{k: v.detach() for k, v in x.items()} for x in self.unweighted]
        self.weighted = [{k: v.detach() for k, v in x.items()} for x in self.weighted]
        self.mask = None if self.mask is None else self.mask.detach()
        return self

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
class LossConfig:  # pylint:disable=too-many-instance-attributes
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
    eye_weight
        The amount of extra weighting to apply to the eye area
    mouth_weight
        The amount of extra weighting to apply to the mouth area
    mask_loss
        The loss function to use if learn_mask is enabled
    identity_backend
        The identity loss functions to use
    identity_weight
        The weighting to use for identity loss
    identity_dissimilarity_weight
        The weighting to use for identity dissimilarity loss
    identity_warmup
        The number of steps to warmup identity loss for
    centering
        The centering type that the model is training at (for identity loss)
    coverage
        The coverage that the model is training at (for identity loss)
    """
    functions: list[str]
    """List of lost function names from configuration file to collate for loss calculation"""
    weights: list[float]
    """List of weights, corresponding to the the list of functions, to apply to each loss
    function"""
    use_mask: bool
    """``True`` if loss should be masked as `penalize mask loss` has been selected"""
    eye_weight: float
    """The amount of extra weighting to apply to the eye area"""
    mouth_weight: float
    """The amount of extra weighting to apply to the mouth area"""
    mask_loss: str | None
    """The loss function to use if learn_mask is enabled"""
    identity_backend: T.Literal["ir-50", "ir-101"] | None
    """The backend to use for the identity loss functions. ``None`` for no identity loss"""
    identity_weight: float
    """The weighting to use for identity loss"""
    identity_dissimilarity_weight: float
    """The weighting to use for identity dissimilarity loss"""
    identity_warmup: int
    """The number of steps to warmup identity loss for"""
    centering: CenteringType
    """The centering type that the model is training at (for identity loss)"""
    coverage: float
    """The coverage that the model is training at (for identity loss)"""


class LossCollator(nn.Module):  # pylint:disable=too-many-instance-attributes
    """Compiles the chosen loss functions and calculates the values in the training loop

    Parameters
    ----------
    config
        The loss configuration settings
    color_order
        The color order that the model is training in
    output_sizes
        The image output sizes from the model
    image_counts
        The number of images contained within each dataset in input order
    """
    def __init__(self,
                 config: LossConfig,
                 color_order: T.Literal["bgr", "rgb"],
                 output_sizes: list[int],
                 image_counts: tuple[int, ...]) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._config = config
        self._color_order: T.Literal["bgr", "rgb"] = color_order
        self._output_sizes = output_sizes
        self._functions, self._weights = self._configure_functions()
        self._spatial, self._non_spatial = self._get_function_types()

        self._mask_loss_function = (
            None if config.mask_loss is None
            else self._functions[config.mask_loss] if config.mask_loss in self._functions
            else get_loss_function(config.mask_loss)
            )
        self._identity_function = (
            None if config.identity_backend is None
            or (config.identity_weight == 0.0 and config.identity_dissimilarity_weight == 0.0)
            else get_loss_function("identity",
                                   color_order=color_order,
                                   kwargs={"backbone": config.identity_backend,
                                           "warmup_steps": config.identity_warmup,
                                           "input_size": max(output_sizes),
                                           "centering": config.centering,
                                           "coverage": config.coverage,
                                           "image_counts": image_counts})
            )
        self._image_idx: int | None = None
        """The index of the final target image within y_true_all"""

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = ", ".join(f"{k[1:]}={repr(v)}" for k, v in self.__dict__.items()
                           if k in ("_config", "_color_order", "_output_sizes", "image_count"))
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
        size = min(self._output_sizes)
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
                          side_index: int,
                          meta: BatchMeta,
                          index: int) -> dict[str, torch.Tensor]:
        """Obtain the unweighted loss values for the spatial loss functions

        Parameters
        ----------
        y_true
            The ground truth batch of images
        y_pred
            The batch of model predictions
        side_index
            The side of the model that is calling the loss function
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
                loss *= meta.mask_face[index][:, side_index]
            if self._config.eye_weight > 1. and meta.mask_eye is not None:
                loss += loss * meta.mask_eye[index][:, side_index] * self._config.eye_weight
            if self._config.mouth_weight > 1. and meta.mask_mouth is not None:
                loss += loss * meta.mask_mouth[index][:, side_index] * self._config.mouth_weight
            retval[name] = loss.mean(dim=tuple(range(1, loss.ndim)))
        logger.trace("[Loss] Spatial loss: %s", retval)  # type:ignore[attr-defined]
        return retval

    def _get_masked_inputs(self,
                           y_true: torch.Tensor,
                           y_pred: torch.Tensor,
                           side_index: int,
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
        side_index
            The side of the model that is calling the loss function
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
        face_mask = meta.mask_face[index][:, side_index]
        inputs = [(y_true * face_mask, y_pred * face_mask)]
        for m_type in ("eye", "mouth"):
            masks: list[torch.Tensor] | None = getattr(meta, f"mask_{m_type}")
            if masks is None:
                continue
            mask = masks[index][:, side_index]
            inputs.append((y_true * mask, y_pred * mask))
            weights.append(self._config.eye_weight if m_type == "eye"
                           else self._config.mouth_weight)
        logger.trace("[Loss] masked inputs: %s, weights: %s",  # type:ignore[attr-defined]
                     [[x.shape for x in i] for i in inputs], weights)
        return inputs, weights

    def _get_non_spatial_loss(self,
                              y_true: torch.Tensor,
                              y_pred: torch.Tensor,
                              side_index: int,
                              meta: BatchMeta,
                              index: int) -> dict[str, torch.Tensor]:
        """Obtain the unweighted loss values for the non-spatial loss functions

        Parameters
        ----------
        y_true
            The ground truth batch of images
        y_pred
            The batch of model predictions
        side_index
            The side of the model that is calling the loss function
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
            inputs, weights = self._get_masked_inputs(y_true, y_pred, side_index, meta, index)

        for name in self._non_spatial:
            losses = torch.stack([self._functions[name](inp_true, inp_pred) * weight
                                 for weight, (inp_true, inp_pred) in zip(weights, inputs)])
            retval[name] = losses.sum(dim=0)

        logger.trace("[Loss] Non-spatial loss: %s", retval)  # type:ignore[attr-defined]
        return retval

    def _get_standard_loss(self,
                           y_true_all: list[torch.Tensor],
                           y_pred_all: list[torch.Tensor],
                           side_index: int,
                           meta: BatchMeta) -> tuple[list[dict[str, torch.Tensor]],
                                                     list[dict[str, torch.Tensor]],
                                                     torch.Tensor | None]:
        """Obtain the loss values for standard A->A, B->B, ... loss functions

        Parameters
        ----------
        y_true_all
            The ground truth batch of images for all outputs for a side of the model
        y_pred_all
            The batch of model predictions for all outputs for a side of the model
        side_index
            The side of the model that is calling the loss function
        meta
            The meta information for the batch

        Returns
        -------
        unweighted
            The unweighted loss per item for each standard loss function for each output
        weighted
            The weighted loss per item for each standard loss function for each output
        mask_loss
            The loss for mask output if learn_mask is selected
        """
        all_unweighted: list[dict[str, torch.Tensor]] = []
        all_weighted: list[dict[str, torch.Tensor]] = []
        mask_loss = None

        for idx, (y_true, y_pred) in enumerate(zip(y_true_all, y_pred_all)):

            if y_true.shape[1] == 1:
                assert self._mask_loss_function is not None
                mask_loss = T.cast(torch.Tensor, self._mask_loss_function(y_true, y_pred))
                mask_loss = mask_loss.mean(dim=tuple(range(1, mask_loss.ndim)))
                continue

            unweighted = self._get_spatial_loss(y_true, y_pred, side_index, meta, idx)
            unweighted |= self._get_non_spatial_loss(y_true, y_pred, side_index, meta, idx)
            all_unweighted.append(unweighted)
            all_weighted.append({k: v * self._weights[k] for k, v in unweighted.items()})

        logger.trace(  # type:ignore[attr-defined]
            "[LossCollator] unweighted: %s, weighted: %s, mask_loss: %s",
            all_unweighted, all_weighted, mask_loss)

        return all_unweighted, all_weighted, mask_loss

    def _get_identity_loss(self,
                           swap_true: torch.Tensor,
                           swap_pred: list[torch.Tensor],
                           side_index: int,
                           meta: BatchMeta,
                           dissim_indices: list[int]
                           ) -> dict[T.Literal["identity", "identity_dissim"], torch.Tensor]:
        """Obtain the unweighted loss values for the identity loss function

        Parameters
        ----------
        swap_true
            The ground truth batch of images for sides of the model in shape (batch_size,
            num_inputs, H, W, C)
        swap_pred
            The batch of swapped model predictions for all model inputs
        side_index
            The side of the model that is calling the loss function
        meta
            The meta information for the batch
        dissim_indices
            The input indices relating to swap_true that should be used for identity dissimilarity
            calculation if an identity loss is being used

        Returns
        -------
        The unweighted loss for the identity similarity and dissimilarity loss functions
        """
        assert self._identity_function is not None

        retval: dict[T.Literal["identity", "identity_dissim"], torch.Tensor] = {}

        if self._config.identity_weight > 0.0:
            id_loss = [self._identity_function(swap_true[:, side_index],
                                               y_pred,
                                               side_index,
                                               side_index,
                                               meta.image_indices[:, side_index],
                                               meta.offsets)
                       for y_pred in swap_pred]
            logger.trace("[LossCollator] identity loss: %s", id_loss)  # type:ignore[attr-defined]
            retval["identity"] = torch.stack(id_loss).mean(dim=0)

        if self._config.identity_dissimilarity_weight <= 0.0:
            return retval

        dissim_loss = [self._identity_function(swap_true[:, tgt_idx],
                                               y_pred,
                                               side_index,
                                               tgt_idx,
                                               meta.image_indices[:, tgt_idx],
                                               meta.offsets)
                       for (y_pred, tgt_idx) in zip(swap_pred, dissim_indices)]

        logger.trace(  # type:ignore[attr-defined]
            "[LossCollator] identity_dissim loss. input_index: %s dissim_indices: %s: %s",
            side_index, dissim_indices, dissim_loss)
        retval["identity_dissim"] = torch.stack(dissim_loss).mean(dim=0)
        return retval

    def _get_swap_loss(self,
                       swap_true: torch.Tensor | None,
                       swap_pred: list[torch.Tensor],
                       side_index: int,
                       meta: BatchMeta,
                       dissim_indices: list[int]
                       ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Obtain the loss values for swap A->B, B->A, ... loss functions

        Parameters
        ----------
        swap_true
            The ground truth batch of images for sides of the model in shape (batch_size,
            num_inputs, H, W, C)
        swap_pred
            The batch of swapped model predictions for all model inputs
        side_index
            The side of the model that is calling the loss function
        meta
            The meta information for the batch
        dissim_indices
            The input indices relating to swap_true that should be used for identity dissimilarity
            calculation if an identity loss is being used

        Returns
        -------
        unweighted
            The unweighted loss per item for each swap loss function
        weighted
            The weighted loss per item for each swap loss function
        """
        unweighted = {}
        weighted = {}

        if self._identity_function is not None and swap_true is not None:
            unweighted = T.cast(dict[str, torch.Tensor],
                                self._get_identity_loss(swap_true,
                                                        swap_pred,
                                                        side_index,
                                                        meta,
                                                        dissim_indices))
            mul = {"identity": self._config.identity_weight,
                   "identity_dissim": self._config.identity_dissimilarity_weight}
            weighted = {k: v * mul[k] for k, v in unweighted.items()}

        logger.trace(  # type:ignore[attr-defined]
            "[LossCollator] unweighted: %s, weighted : %s",
            unweighted, weighted)
        return unweighted, weighted

    def forward(self,
                y_true_all: list[torch.Tensor],
                y_pred_all: list[torch.Tensor],
                side_index: int,
                meta: BatchMeta,
                swap_true: torch.Tensor | None,
                swap_pred: list[torch.Tensor],
                dissim_indices: list[int]) -> BatchLoss:
        """Call the loss functions, reduce to batch dimension, apply masks and weighting and obtain
        the weighted and unweighted per function values and the weighted total loss scalar

        Parameters
        ----------
        y_true_all
            The ground truth batch of images for all outputs for a side of the model
        y_pred_all
            The batch of model predictions for all outputs for a side of the model
        side_index
            The side of the model that is calling the loss function
        meta
            The meta information for the batch
        swap_true
            The targets for identity loss for all model inputs in shape (batch_size, num_inputs,
            H, W, C) or ``None`` if identity loss is not enabled
        swap_pred
            The swap predictions from the model if an identity loss is being used
        dissim_indices
            The input indices relating to swap_true that should be used for identity dissimilarity
            calculation if an identity loss is being used

        Returns
        -------
        The loss scalars for the batch
        """
        # TODO remove once channels first
        y_true_all = [x.permute(0, 3, 1, 2) for x in y_true_all]
        y_pred_all = [x.permute(0, 3, 1, 2) for x in y_pred_all]
        swap_true = None if swap_true is None else swap_true.permute(0, 1, 4, 2, 3)
        swap_pred = [x.permute(0, 3, 1, 2) for x in swap_pred]

        unweighted, weighted, mask_loss = self._get_standard_loss(y_true_all,
                                                                  y_pred_all,
                                                                  side_index,
                                                                  meta)
        swap_unweighted, swap_weighted = self._get_swap_loss(swap_true,
                                                             swap_pred,
                                                             side_index,
                                                             meta,
                                                             dissim_indices)
        unweighted[-1] |= swap_unweighted
        weighted[-1] |= swap_weighted

        retval = BatchLoss(unweighted=unweighted,
                           weighted=weighted,
                           mask=mask_loss)
        logger.trace("[Loss] %s", retval)  # type:ignore[attr-defined]
        return retval


__all__ = get_module_objects(__name__)
