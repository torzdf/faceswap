#!/usr/bin/env python3
"""Custom Loss Functions that act on the model's ability to swap identity for faceswap.py"""
from __future__ import annotations

import logging
import typing as T

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lib.model.networks.insightface_resnet import ir_50, ir_101
from lib.logger import parse_class_init
from lib.utils import get_module_objects
from lib.utils import GetModel

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from lib.model.networks.insightface_resnet import IRNet

logger = logging.getLogger(__name__)


class IdentityLoss(nn.Module):
    """Loss function that calculates the identity vectors of the swapped output of the model
    against the mean of the source dataset's identity vectors

    Parameters
    ----------
    backbone
        The model backbone to use
    color_order
        The `"bgr"` or `"rgb"` color order of the incoming images
    num_inputs
        The number of inputs that the model is being trained on
    image_counts
        The number of images contained within each dataset in input order
    """
    _input_size = 112
    _identities: torch.Tensor

    def __init__(self,
                 backbone: T.Literal["ir-50", "ir-101"],
                 color_order: T.Literal["bgr", "rgb"],
                 image_counts: tuple[int, ...]) -> None:
        logger.debug(parse_class_init(locals()))
        self._backbone = backbone
        self._color_order = color_order
        super().__init__()

        self._seen_targets = [set() for _ in range(len(image_counts))]
        self._cache_full = [False for _ in range(len(image_counts))]
        self._cache_counts = [0 for _ in range(len(image_counts))]
        self.register_buffer("_identities",
                             torch.zeros((len(image_counts), 512), dtype=torch.float32))

        self._net: IRNet = self._get_net()
        self._cosine_similarity = nn.CosineSimilarity(dim=1)

    def _get_net(self) -> IRNet:
        """Get the requested identity model and load the weights

        Returns
        -------
        The requested Identity model with weights loaded
        """
        model = ir_50 if self._backbone == "ir-50" else ir_101
        net = model(112)
        vers = 1 if self._backbone == "ir-50" else 2
        weights_path = GetModel(f"tface_v{vers}.pth", 33).model_path
        assert isinstance(weights_path, str)
        weights = torch.load(weights_path)
        net.load_state_dict(weights)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False

        logger.debug("[IdentityLoss] Loaded model with backbone '%s'", self._backbone)
        return net

    @classmethod
    def _prepare_images(cls, images: torch.Tensor) -> torch.Tensor:
        """Crop and resize the images for feeding through the identity model"""
        # TODO crop + align
        retval = F.interpolate(images, size=cls._input_size, mode="area")
        return retval

    def _cache_target_identities(self,
                                 targets: torch.Tensor,
                                 side_index: int,
                                 image_ids: npt.NDArray[np.int64]) -> None:
        """Obtain the identities for any images in the given batch that have not yet been cached
        and add to the running mean of identities for the side

        Parameters
        ----------
        targets
            The target identity images
        side_index
            The side index of the model that the target identity images belong to
        image_ids
            The image ids of the target identity images
        """
        indices_to_cache = set(image_ids.tolist()).difference(self._seen_targets[side_index])
        if not indices_to_cache:
            logger.info("[Identity Loss] Cache full for side %s", side_index)
            self._cache_full[side_index] = True
            return

        logger.trace(  # type:ignore[attr-defined]
            "[IdentityLoss] Caching identities for side %s: %s", side_index, indices_to_cache)

        feed = self._prepare_images(
            targets[np.isin(image_ids,  # pyright:ignore[reportArgumentType]
                            list(indices_to_cache))])
        identities = T.cast(torch.Tensor, self._net(feed)).mean(dim=0)
        self._cache_counts[side_index] += 1
        self._identities[side_index] += ((identities - self._identities[side_index]) /
                                         self._cache_counts[side_index])
        self._seen_targets[side_index].update(indices_to_cache)

    def forward(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor,
                side_index: int,
                image_ids: npt.NDArray[np.int64]) -> torch.Tensor:
        """Run forward pass through the identity loss function

        Parameters
        ----------
        y_true
            The target identity images
        y_pred
            The predicted swapped images generated from the original alternate identities
        side_index
            The side index of the model that the target identity images belong to
        image_ids
            The image ids of the target identity images

        Returns
        -------
        The similarity between the predicted swapped images and the target images
        """
        if not self._cache_full[side_index]:
            self._cache_target_identities(y_true, side_index, image_ids)
        swap_identities = self._net(self._prepare_images(y_pred))
        similarity = 1.0 - self._cosine_similarity(swap_identities, self._identities[side_index])
        return similarity


__all__ = get_module_objects(__name__)
