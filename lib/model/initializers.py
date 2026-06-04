#!/usr/bin/env python3
"""Custom Initializers for faceswap.py"""
from __future__ import annotations

import logging
import typing as T

import numpy as np
import torch
from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


def icnr(tensor: torch.Tensor,
         initializer: T.Callable[[torch.Tensor], torch.Tensor] | None = None,
         scale: int = 2,
         *args,
         **kwargs) -> torch.Tensor:
    """ICNR initializer for checkerboard artifact free sub pixel convolution. Action is performed
    in place replacing the given input tensor

    Parameters
    ----------
    tensor
        The original weight tensor
    initializer
        The initializer used for sub kernels (orthogonal, glorot uniform, etc.). Default: ``None``
        (uses kaiming_normal)
    scale
        scaling factor of sub pixel convolution (up sampling from 8x8 to 16x16 is scale 2).
        Default: `2`
    args
        Any args for the original initializer
    kwargs
        Any kwargs for the original initializer

    Returns
    -------
    The modified kernel weights

    References
    ----------
    Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/pdf/1707.02937.pdf,  https://distill.pub/2016/deconv-checkerboard/
    https://gist.github.com/A03ki/2305398458cb8e2155e8e81333f0a965
    """
    initializer = nn.init.kaiming_normal_ if initializer is None else initializer
    scale_squared = scale * scale
    assert tensor.shape[0] % scale_squared == 0
    with torch.no_grad():
        sub_kernel = torch.empty(tensor.shape[0] // scale_squared, *tensor.shape[1:],
                                 dtype=tensor.dtype, device=tensor.device)
        initializer(sub_kernel, *args, **kwargs)
        tensor.copy_(sub_kernel.repeat_interleave(scale_squared, dim=0))

    logger.debug("ICNR Output shape: %s", tensor.shape)
    return tensor


def compute_fans(tensor: torch.Tensor) -> tuple[int, int]:
    """Calculate fan in/fan out for the given Tensor. Lifted from torch

    Parameters
    ----------
    tensor
        The tensor to calculate the fan for

    Returns
    -------
    fan_in
        The fan in shape
    fan_out
        The fan out shape
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_f_maps = tensor.size(1)
    num_output_f_maps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_f_maps * receptive_field_size
    fan_out = num_output_f_maps * receptive_field_size

    return fan_in, fan_out


class ConvolutionAware:
    """Initializer that generates orthogonal convolution filters in the Fourier space. If this
    initializer is passed a shape that is not 3D or 4D, orthogonal initialization will be used.

    Adapted, fixed and optimized from:
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/initializers/convaware.py

    Parameters
    ----------
    eps_std
        The Standard deviation for the random normal noise used to break symmetry in the inverse
        Fourier transform. Default: 0.05
    seed
        Used to seed the random generator. Default: ``None``
    initialized
        This should always be set to ``False``. To avoid Keras re-calculating the values every time
        the model is loaded, this parameter is internally set on first time initialization.
        Default:``False``

    Returns
    -------
    The modified kernel weights

    References
    ----------
    Armen Aghajanyan, https://arxiv.org/abs/1702.06295
    """
    def __init__(self, eps_std: float = 0.05, seed: int | None = None,) -> None:
        logger.debug(parse_class_init(locals()))
        self._eps_std = eps_std
        self._seed = seed

    @classmethod
    def _symmetrize(cls, inputs: torch.Tensor) -> torch.Tensor:
        """Make the given tensor symmetrical.

        Parameters
        ----------
        inputs
            The input tensor to make symmetrical

        Returns
        -------
        The symmetrical output
        """
        a = inputs.permute(0, 1, 3, 2)
        diag = a.diagonal(dim1=-2, dim2=-1)
        b = diag.diag_embed()
        retval = inputs + a - b
        logger.debug("[ConvolutionAware] Input shape: %s. Output shape: %s",
                     inputs.shape, retval.shape)
        return retval

    def _create_basis(self,
                      filters_size: int,
                      filters: int,
                      size: int,
                      dtype: torch.dtype) -> torch.Tensor:
        """Create the basis for convolutional aware initialization

        Parameters
        ----------
        filters_size
            The size of the filter
        filters
            The number of filters
        dtype
            The data type

        Returns
        -------
        The output array
        """
        if size == 1:
            return torch.normal(0.0, self._eps_std, (filters_size, filters, size))
        nbb = filters // size + 1
        a = torch.normal(0.0, 1.0, (filters_size, nbb, size, size))
        a = self._symmetrize(a)
        u: torch.Tensor = torch.linalg.svd(a)[0].permute(0, 1, 3, 2)
        retval = u.reshape(filters_size, nbb * size, size)[:, :filters, :]
        logger.debug("filters_size: %s, filters: %s, size: %s, dtype: %s, output: %s",
                     filters_size, filters, size, dtype, retval.shape)
        return retval

    @classmethod
    def _scale_filters(cls, filters: torch.Tensor, variance: float) -> torch.Tensor:
        """Scale the given filters.

        Parameters
        ----------
        filters
            The filters to scale
        variance
            The amount of variance

        Returns
        -------
        The scaled filters
        """
        c_var = torch.var(filters)
        p = torch.sqrt(variance / c_var)
        retval = filters * p
        logger.debug("Scaled filters (filters: %s, variance: %s, output: %s)",
                     filters.shape, variance, retval.shape)
        return retval

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Call function for the Convolutional Aware initializer.

        Parameters
        ----------
        tensor
            The original weight tensor

        Returns
        -------
        The modified kernel weights
        """
        shape = tensor.shape
        logger.debug("Calculating Convolution Aware Initializer for shape: %s", shape)
        rank = len(shape)
        if self._seed is not None:
            torch.manual_seed(self._seed)
            np.random.seed(self._seed)

        fan_in, _ = compute_fans(tensor)
        variance = 2 / fan_in

        kernel_shape: tuple[int, ...]
        correct_ifft: T.Callable
        correct_fft: T.Callable
        if rank == 3:
            filters_size, stack_size, row = shape
            kernel_shape = (row,)
            correct_ifft = lambda shape, s=[None]: torch.fft.irfft(shape, s[0])  # noqa:E731,E501 pylint:disable=unnecessary-lambda-assignment
            correct_fft = torch.fft.rfft
        elif rank == 4:
            filters_size, stack_size, row, column = shape
            kernel_shape = (row, column)
            correct_ifft = torch.fft.irfft2
            correct_fft = torch.fft.rfft2
        elif rank == 5:
            stack_size, filters_size, var_x, var_y, var_z = shape
            kernel_shape = (var_x, var_y, var_z)
            correct_fft = torch.fft.rfftn
            correct_ifft = torch.fft.irfftn
        else:
            return nn.init.orthogonal_(tensor)

        with torch.no_grad():
            kernel_fourier_shape = correct_fft(torch.zeros(kernel_shape)).shape
            basis = self._create_basis(filters_size,
                                       stack_size,
                                       T.cast(int, np.prod(kernel_fourier_shape)),
                                       tensor.dtype)
            basis = basis.reshape((filters_size, stack_size,) + kernel_fourier_shape)
            randoms = torch.normal(0, self._eps_std, basis.shape[:-2] + kernel_shape)
            init = correct_ifft(basis, kernel_shape) + randoms
            tensor.copy_(self._scale_filters(init, variance).to(tensor.dtype))
        logger.debug("ConvAware output: %s", (tensor.shape, tensor.dtype))
        return tensor


__all__ = get_module_objects(__name__)
