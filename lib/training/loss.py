#! /usr/env/bin/python3
""" Handles the collation, weighting, masking and calculation of the selected Loss functions for
training Faceswap models """
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
    """ Stores loss values and metadata for a single training batch

    This dataclass holds the unweighted and weighted loss computations for all configured loss
    functions during one forward pass. It provides lazy computation of total loss via the .total
    property, which sums contributions from spatial losses, non-spatial losses, and optional mask
    penalties. The stored values are detached tensors following the backwards pass

    Parameters
    ----------
    unweighted
        List of dictionaries mapping loss function names to their raw (unweighted) tensor losses.
        Each dictionary corresponds to one sample in the batch, with keys being loss function
        identifiers and values being scalar loss tensors for that sample
    weighted
        List of dictionaries mapping loss function names to their weighted tensor losses.
        Each entry is multiplied by its corresponding weight from LossCollator configuration,
        allowing differential emphasis on different loss components during optimization
    mask
        Optional mask loss tensor computed when `learn_mask` is enabled

    Notes
    -----
    The total property is computed lazily on first access via the @property decorator. This
    avoids unnecessary loss computation if the value isn't needed

    The detach() method recursively detaches all tensors

    The to_cpu() method is typically called after loss evaluation completes to free up GPU
    memory before the next iteration begins.
    """

    unweighted: list[dict[str, torch.Tensor]]
    """ List of dictionaries mapping loss function names to their raw (unweighted) tensor losses.
    Each dictionary corresponds to one sample in the batch, with keys being loss function
    identifiers and values being scalar loss tensors for that sample """

    weighted: list[dict[str, torch.Tensor]]
    """ List of dictionaries mapping loss function names to their weighted tensor losses.
    Each entry is multiplied by its corresponding weight from LossCollator configuration,
    allowing differential emphasis on different loss components during optimization """

    mask: torch.Tensor | None = None
    """ Optional mask loss tensor computed when `learn_mask` is enabled """

    _total: torch.Tensor | None = field(init=False, default=None)

    @property
    def total(self) -> torch.Tensor:
        """ Computed sum of all weighted losses plus optional mask contribution. Lazily evaluated
        on first access to avoid unnecessary computation if not needed """
        if self._total is None:
            total = T.cast(torch.Tensor, sum(sum(y.mean() for y in x.values())
                                             for x in self.weighted))
            if self.mask is not None:
                total += self.mask.mean()
            self._total = total
        return self._total

    def get_contributions(self) -> dict[T.Literal["unweighted", "weighted"],
                                        dict[str, torch.Tensor]]:
        """ Extract mean contributions from each loss function

        Computes average contribution per loss function across all batch samples and groups
        results into unweighted and weighted categories. Identity losses (if present) are
        extracted separately from the last sample if they exist in that position

        Returns
        -------
        Dictionary with keys "unweighted" and "weighted", each mapping loss function names
        (strings) to their mean contribution tensors across the batch

        Notes
        -----
        Identity losses are special-cased when they appear in the last position of either
        unweighted or weighted lists, as these represent identity-based comparison metrics
        """
        unweighted = {k: T.cast(torch.Tensor, sum(d[k].mean() for d in self.unweighted))
                      for k in self.unweighted[0]}
        weighted = {k: T.cast(torch.Tensor, sum(d[k].mean() for d in self.weighted))
                    for k in self.weighted[0]}
        if "identity" in list(self.unweighted)[-1]:
            unweighted["identity"] = self.unweighted[-1]["identity"].mean()
            weighted["identity"] = self.weighted[-1]["identity"].mean()
        return {"unweighted": unweighted, "weighted": weighted}

    def detach(self) -> T.Self:
        """ Detach all stored tensors from computation graph

        Recursively detaches the cached total loss and all entries in unweighted/weighted
        dictionaries as well as any mask tensor. Returns self for chaining

        Returns
        -------
        Self reference after detachment (enables method chaining)

        Notes
        -----
        Called at the end of each Optimizer backwards step
        """
        self._total = None if self._total is None else self._total.detach()
        self.unweighted = [{k: v.detach() for k, v in x.items()} for x in self.unweighted]
        self.weighted = [{k: v.detach() for k, v in x.items()} for x in self.weighted]
        self.mask = None if self.mask is None else self.mask.detach()
        return self

    def to_cpu(self) -> T.Self:
        """ Move all stored tensors to CPU memory.

        Detaches and transfers all loss tensors from GPU to CPU after computation completes,
        freeing up valuable GPU VRAM for the next iteration's batch processing. Returns self
        for chaining with other methods like detach()

        Returns
        -------
        Self reference after moving to CPU (enables method chaining)

        Notes
        -----
        Called at the end of each Optimizer backwards step
        """
        self._total = None if self._total is None else self._total.detach().cpu()
        self.unweighted = [{k: v.detach().cpu() for k, v in x.items()} for x in self.unweighted]
        self.weighted = [{k: v.detach().cpu() for k, v in x.items()} for x in self.weighted]
        self.mask = None if self.mask is None else self.mask.detach().cpu()
        return self


class LossCollator(nn.Module):  # pylint:disable=too-many-instance-attributes
    """ Configures and computes weighted/unweighted loss functions for training

    LossCollator inherits from nn.Module to provide the standard PyTorch forward() interface,
    enabling integration with DataLoader collators and training loops. It configures all loss
    functions specified in user configuration, applies appropriate weights, handles
    spatial vs non-spatial losses based on output dimensions, and manages mask-based penalties
    for face/eye/mouth regions when enabled

    The collator distinguishes between:
        - Spatial losses : Return per-pixel or region-wise loss (N, C, H, W) shape
        - Non-spatial losses: Return scalar loss values (N,) shape across batch

    This distinction affects how each loss contributes to gradient computation during
    backpropagation.

    Parameters
    ----------
    functions
        List of loss function names from user config. Names must be valid identifiers recognized by
        get_loss_function() or will be ignored if weight <= 0
    weights
        Corresponding weight values for each loss function. Only functions with positive weights
        are included in training computation. Zero-weighted functions are configured but excluded
        from loss summation
    color_order
        Either "bgr" (default) or "rgb" depending on input image channel order configuration.
        Affects how loss functions interpret color channels during comparison operations
    use_mask
        Whether a mask is used for training, be it for penalized mask loss, region multipliers or
        the model is learning a mask
    eye_multiplier
        Multiplier factor applied to eye region losses if > 1.0. Combined with use_mask=True
        to emphasize eye alignment during training by increasing their gradient contribution
    mouth_multiplier
        Multiplier factor applied to mouth region losses if > 1.0. Works similarly to
        eye_multiplier but for mouth alignment emphasis when mask_mouth is available in metadata
    smallest_output
        Width/height of the smallest output tensor among all loss functions. Used to create
        dummy tensors during function type detection (spatial vs non-spatial classification)
    mask_loss
        Optional name of additional mask-based loss function (e.g., "mae", "mse"). If provided,
        this loss is configured separately and computed for the 'learn_mask' additional output

    Notes
    -----
    Loss function registration: Functions with weight <= 0.0 are silently skipped during
    initialization to allow config flexibility without breaking training if certain loss types
    aren't desired

    Spatial vs Non-Spatial classification is performed once during __init__ using dummy tensors
    shaped (1, 3, smallest_output, smallest_output). This categorization determines how each loss
    contributes gradients - spatial losses sum across pixels first, non-spatial losses accumulate
    directly as scalars per batch item

    Mask handling: When use_mask=True and meta.mask_face is available, face regions are multiplied
    by mask values before loss computation. Eye/mouth multipliers add weighted penalties
    proportional to mask overlap for those specific regions if their masks exist in metadata

    Mask-based loss (mask_loss parameter): Only computed when learn_mask is enabled
    """
    def __init__(self,
                 functions: list[str],
                 weights: list[float],
                 color_order: T.Literal["bgr", "rgb"],
                 use_mask: bool,
                 eye_multiplier: float,
                 mouth_multiplier: float,
                 smallest_output: int,
                 mask_loss: str | None = None) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._color_order: T.Literal["bgr", "rgb"] = color_order
        self._use_mask = use_mask
        self._eye_multiplier = eye_multiplier
        self._mouth_multiplier = mouth_multiplier
        self._smallest_output = smallest_output
        self._mask_loss = mask_loss
        self._functions, self._weights = self._configure_functions(functions, weights)
        self._spatial, self._non_spatial = self._get_function_types()

        self._mask_loss_function = (
            None if mask_loss is None
            else self._functions[mask_loss] if mask_loss in self._functions
            else get_loss_function(mask_loss)
            )

    def __repr__(self) -> str:
        """ Return a string representation for logging purposes """
        params = {"functions": list(self._functions),
                  "weights": list(self._weights.values())}
        params |= {k[1:]: v for k, v in self.__dict__.items()
                   if k in ("_color_order", "_use_mask", "_eye_multiplier", "_mouth_multiplier",
                            "_smallest_output", "_mask_loss")}
        s_params = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    def _configure_functions(self,
                             names: list[str],
                             weights: list[float]) -> tuple[nn.ModuleDict, dict[str, float]]:
        """ Configure and initialize loss function modules from configuration values

        Parameters
        ----------
        names
            List of loss function names from user config to use
        weights
            List of weight values corresponding to each name

        Returns
        -------
        functions
            nn.ModuleDict keyed by function names with the loss function instances as values
        weight_mapping
            Dictionary mapping function names to their weight multipliers

        Raises
        ------
        ValueError
            If len(names) != len(weights), indicating mismatched configuration
        """

        if len(names) != len(weights):
            raise ValueError(f"Number of loss functions ({len(names)}) and weights "
                             f"({len(weights)}) should match")

        functions = nn.ModuleDict()
        weight_dict: dict[str, float] = {}
        for name, weight in zip(names, weights):
            if name is None or name == "none" or weight <= 0.0:
                continue
            functions[name] = get_loss_function(name, self._color_order)
            weight_dict[name] = weight

        logger.debug("[Loss] Configured loss functions: %s",
                     {k: (functions[k].__class__.__name__, weight_dict[k]) for k in functions})
        return functions, weight_dict

    def _get_function_types(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """ Classify loss functions as spatial or non-spatial based on output dimensions

        Returns
        -------
        spatial
            The function names of the spatial loss functions
        non_spatial
            The function names of the non-spatial loss functions

        Raises
        ------
        RuntimeError
            If any loss function returns output with ndim other than 1 or 4
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
                          y_pred: torch.Tensor,
                          y_true: torch.Tensor,
                          meta: BatchMeta,
                          index: int) -> dict[str, torch.Tensor]:
        """ Compute losses for spatial loss functions with optional mask/eye/mouth multipliers

        Parameters
        ----------
        y_pred
            Tensor of predicted feature maps from encoder (shape: [batch, channels, ...])
        y_true
            Tensor of ground truth feature maps for comparison (same shape as y_pred)
        meta
            Batch metadata containing mask_face, mask_eye, mask_mouth tensors if available
        index
            Sample index within the batch for accessing per-sample masks and metadata

        Returns
        -------
        Dictionary mapping loss function names to their computed mean loss values averaged over
        spatial dimensions
        """
        retval: dict[str, torch.Tensor] = {}
        for name in self._spatial:
            loss: torch.Tensor = self._functions[name](y_pred, y_true)
            if self._use_mask and meta.mask_face is not None:
                loss *= meta.mask_face[index]
            if self._eye_multiplier > 1. and meta.mask_eye is not None:
                loss += loss * meta.mask_eye[index] * self._eye_multiplier
            if self._mouth_multiplier > 1. and meta.mask_mouth is not None:
                loss += loss * meta.mask_mouth[index] * self._mouth_multiplier
            retval[name] = loss.mean(dim=tuple(range(1, loss.ndim)))
        logger.trace("[Loss] Spatial loss: %s", retval)  # type:ignore[attr-defined]
        return retval

    def _get_masked_inputs(self,
                           y_pred: torch.Tensor,
                           y_true: torch.Tensor,
                           meta: BatchMeta,
                           index: int
                           ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[float]]:
        """ Prepare masked input pairs for non-spatial loss computation

        Parameters
        ----------
        y_pred
            Tensor of predicted feature maps (shape: [batch, channels, ...])
        y_true
            Tensor of ground truth feature maps for comparison
        meta
            Batch metadata containing mask_face, mask_eye, mask_mouth tensors if available
        index
            Sample index within batch for accessing per-sample mask data

        Returns
        -------
        masked_input
            List of (masked_pred, masked_truth) pairs ready for feeding non-spatial loss functions
        multipliers
            List of weight multipliers for each input pair
        """
        weights = [1.0]
        assert meta.mask_face is not None
        face_mask = meta.mask_face[index]
        inputs = [(y_pred * face_mask, y_true * face_mask)]
        for m_type in ("eye", "mouth"):
            masks: list[torch.Tensor] | None = getattr(meta, f"mask_{m_type}")
            if masks is None:
                continue
            mask = masks[index]
            inputs.append((y_pred * mask, y_true * mask))
            weights.append(self._eye_multiplier if m_type == "eye" else self._mouth_multiplier)
        logger.trace("[Loss] masked inputs: %s, weights: %s",  # type:ignore[attr-defined]
                     [[x.shape for x in i] for i in inputs], weights)
        return inputs, weights

    def _get_non_spatial_loss(self,
                              y_pred: torch.Tensor,
                              y_true: torch.Tensor,
                              meta: BatchMeta,
                              index: int) -> dict[str, torch.Tensor]:
        """ Compute losses for non-spatial loss functions using raw or masked inputs

        Parameters
        ----------
        y_pred
            Tensor of predicted feature maps from encoder (shape: [batch, channels, ...])
        y_true
            Tensor of ground truth feature maps for comparison
        meta
            Batch metadata containing mask tensors if use_mask is enabled
        index
            Sample index within batch for accessing per-sample mask data or skipping masking

        Returns
        -------
        Dictionary mapping non-spatial loss function names to their computed scalar loss values
        """
        retval: dict[str, torch.Tensor] = {}
        if not self._use_mask:
            inputs = [(y_pred, y_true)]
            weights = [1.0]
        else:
            inputs, weights = self._get_masked_inputs(y_pred, y_true, meta, index)

        for name in self._non_spatial:
            losses = torch.stack([self._functions[name](inp_pred, inp_true) * weight
                                 for weight, (inp_pred, inp_true) in zip(weights, inputs)])
            retval[name] = losses.sum(dim=0)

        logger.trace("[Loss] Non-spatial loss: %s", retval)  # type:ignore[attr-defined]
        return retval

    def forward(self,
                y_pred_all: list[torch.Tensor],
                y_true_all: list[torch.Tensor],
                meta: BatchMeta) -> BatchLoss:
        """ Execute the full training step for a complete batch.

        Iterates through each sample in the batch computing spatial and non-spatial losses
        separately, accumulating results into unweighted/weighted lists. For samples with single-
        channel targets (binary masks), computes mask_loss using configured mask_loss_function if
        available instead of standard losses. Returns BatchLoss dataclass containing all computed
        values for later use by LossUnit or optimizer callbacks

        Parameters
        ----------
        y_pred_all
            List of predicted output tensors from each side of the model
        y_true_all
            List of ground truth tensors, matching length and order with y_pred_all
        meta
            BatchMeta object containing mask_face, mask_eye, mask_mouth for all samples in the
            batch

        Returns
        -------
        Dataclass instance containing:
            - unweighted list of loss dictionaries
            - weighted list of loss dictionaries (same structure, weights applied)
            - mask tensor (if mask_loss was computed during forward pass)

        Notes
        -----
        Spatial losses are summed across their output dimensions first before being added to the
        running total. Non-spatial losses maintain scalar form and are stacked then summed across
        samples. The mask_loss only gets returned in BatchLoss.mask if learn_mask is enabled
        """
        all_unweighted: list[dict[str, torch.Tensor]] = []
        all_weighted: list[dict[str, torch.Tensor]] = []
        mask_loss = None
        for idx, (y_pred, y_true) in enumerate(zip(y_pred_all, y_true_all)):

            if y_true.shape[1] == 1:
                assert self._mask_loss_function is not None
                mask_loss = T.cast(torch.Tensor, self._mask_loss_function(y_pred, y_true))
                mask_loss = mask_loss.mean(dim=tuple(range(1, mask_loss.ndim)))
                continue

            unweighted = self._get_spatial_loss(y_pred, y_true, meta, idx)
            unweighted |= self._get_non_spatial_loss(y_pred, y_true, meta, idx)
            all_unweighted.append(unweighted)
            all_weighted.append({k: v * self._weights[k] for k, v in unweighted.items()})

        retval = BatchLoss(unweighted=all_unweighted,
                           weighted=all_weighted,
                           mask=mask_loss)
        logger.trace("[Loss] %s", retval)  # type:ignore[attr-defined]
        return retval


__all__ = get_module_objects(__name__)
