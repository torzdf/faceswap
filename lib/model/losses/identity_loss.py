#!/usr/bin/env python3
"""Custom Loss Functions that act on the model's ability to swap identity for faceswap.py"""
from __future__ import annotations

import logging
import typing as T

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lib.align.aligned_utils import batch_sub_crop_torch, get_base_scale, get_sub_crop_scale
from lib.model.networks.insightface_resnet import ir_50, ir_101
from lib.logger import parse_class_init
from lib.utils import get_module_objects
from lib.utils import GetModel

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from lib.align.constants import CenteringType
    from lib.model.networks.insightface_resnet import IRNet

logger = logging.getLogger(__name__)
# TODO y-offset


class IdentityLoss(nn.Module):  # pylint:disable=too-many-instance-attributes
    """Loss function that calculates the identity vectors of the swapped output of the model
    against the mean of the source dataset's identity vectors

    Parameters
    ----------
    color_order
        The `"bgr"` or `"rgb"` color order of the incoming images
    backbone
        The model backbone to use
    input_size
        The input size to the loss function (ie: the model output size)
    centering
        The centering that the model is trained at
    coverage
        The coverage that the model is trained at
    image_counts
        The number of images contained within each dataset in input order
    """
    _input_size = 112
    _identities: torch.Tensor
    _crop_grid_x: torch.Tensor
    _crop_grid_y: torch.Tensor

    def __init__(self,
                 color_order: T.Literal["bgr", "rgb"],
                 backbone: T.Literal["ir-50", "ir-101"],
                 input_size: int,
                 centering: CenteringType,
                 coverage: float,
                 image_counts: tuple[int, ...]) -> None:
        logger.debug(parse_class_init(locals()))
        self._backbone = backbone
        self._color_order = color_order
        self._image_counts = image_counts
        self._padding_diff, self._crop_size, self._base_size = self._get_crop_data(
            centering,
            coverage,
            input_size)
        super().__init__()

        self._seen_targets = [set() for _ in range(len(image_counts))]
        self._cache_full = [False for _ in range(len(image_counts))]
        self._cache_counts = [0 for _ in range(len(image_counts))]
        self.register_buffer("_identities",
                             torch.zeros((len(image_counts), 512), dtype=torch.float32))

        crop_grid = torch.meshgrid(torch.arange(self._crop_size, dtype=torch.long),
                                   torch.arange(self._crop_size, dtype=torch.long),
                                   indexing="ij")
        self.register_buffer("_crop_grid_x", crop_grid[0])
        self.register_buffer("_crop_grid_y", crop_grid[1])

        self._net: IRNet = self._get_net()
        self._cosine_similarity = nn.CosineSimilarity(dim=1)

    @classmethod
    def _get_crop_data(cls,
                       centering: CenteringType,
                       coverage: float,
                       input_size: int) -> tuple[int, int, int]:
        """Obtain the data required to crop a legacy 100% image from a training image for feeding
        the identity model, at model output scale

        Parameters
        ----------
        centering
            The centering that the model is trained at
        coverage
            The coverage that the model is trained at
        input_size
            The input size to the loss function (ie: the model output size)

        Returns
        -------
        padding_difference
            The difference in padding between a training image and a 100% coverage legacy sub-crop
        crop_size
            The size of the sub-crop to obtain from a training image
        base_size
            The size of the base aligned area with no padding applied
        """
        train_to_legacy_ratio = get_sub_crop_scale(centering, "legacy", coverage, 1.0) / 2
        out_size = 2 * int(np.rint(input_size * train_to_legacy_ratio))
        padding_diff = (input_size - out_size) // 2

        train_to_base_ratio = get_base_scale(centering, coverage) / 2
        base_size = 2 * int(np.rint(input_size * train_to_base_ratio))

        logger.debug("[IdentityLoss] padding_diff: %s, sub_crop size: %s, base_size: %s",
                     padding_diff, out_size, base_size)
        return padding_diff, out_size, base_size

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

    def _prepare_images(self, images: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Crop and resize the images for feeding through the identity model

        Parameters
        ----------
        targets
            The target identity images
        offsets
            The offsets to shift training images to legacy centering

        Returns
        -------
        The images prepared for feeding the identity model
        """
        # TODO color order
        scaled_offsets = torch.round(
            -offsets * self._base_size + self._padding_diff).to(torch.int32)
        cropped = batch_sub_crop_torch(images,
                                       scaled_offsets,
                                       self._crop_size,
                                       base_grid=(self._crop_grid_x, self._crop_grid_y))
        retval = F.interpolate(cropped, size=self._input_size, mode="area")
        return retval

    def _cache_target_identities(self,
                                 targets: torch.Tensor,
                                 target_index: int,
                                 image_ids: npt.NDArray[np.int64],
                                 offsets: torch.Tensor) -> None:
        """Obtain the identities for any images in the given batch that have not yet been cached
        and add to the running mean of identities for the side

        Parameters
        ----------
        targets
            The target identity images
        target_index
            The side index of the model that the target identity images belong to
        image_ids
            The image ids of the target identity images
        offsets
            The offsets to shift the target images to legacy centering
        """
        indices_to_cache = set(image_ids.tolist()).difference(self._seen_targets[target_index])
        if not indices_to_cache:
            return

        logger.debug(  # type:ignore[attr-defined]
            "[IdentityLoss] Caching identities for side %s: %s", target_index, indices_to_cache)

        mask = np.isin(image_ids, list(indices_to_cache))
        feed = self._prepare_images(targets[mask],  # pyright:ignore[reportArgumentType]
                                    offsets[mask])  # pyright:ignore[reportArgumentType]
        identities = T.cast(torch.Tensor, self._net(feed)).mean(dim=0)
        self._cache_counts[target_index] += 1
        self._identities[target_index] += ((identities - self._identities[target_index]) /
                                           self._cache_counts[target_index])
        self._seen_targets[target_index].update(indices_to_cache)

        if len(self._seen_targets[target_index]) == self._image_counts[target_index]:
            logger.info("[Identity Loss] Cache full for side %s", target_index)
            self._cache_full[target_index] = True

    def forward(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor,
                side_index: int,
                target_index: int,
                image_ids: npt.NDArray[np.int64],
                offsets: torch.Tensor) -> torch.Tensor:
        """Run forward pass through the identity loss function.

        Note: if side_index == target_index then this is a similarity calculation (that is y_true
        and y_pred should be resolving to the same identities). If side_index != target_index then
        this is a dissimilarity calculation(that is y_true and y_pred should be resolving to
        different identities)

        Parameters
        ----------
        y_true
            The target identity images
        y_pred
            The predicted swapped images generated from the original alternate identities
        side_index
            The side index of the model that y_pred belong to
        target_index
            The side index of the model that y_true belongs to
        image_ids
            The image ids of the target identity images
        offsets
            The offsets to shift training images to legacy centering

        Returns
        -------
        The similarity or dissimilarity between the predicted swapped images and the target images
        """
        if not self._cache_full[target_index]:
            self._cache_target_identities(y_true,
                                          target_index,
                                          image_ids,
                                          offsets[:, target_index])

        feed = self._prepare_images(y_pred, offsets[:, side_index])
        swap_identities = self._net(feed)
        retval = self._cosine_similarity(swap_identities, self._identities[target_index])
        if side_index == target_index:
            retval = 1.0 - retval
        return retval


__all__ = get_module_objects(__name__)
