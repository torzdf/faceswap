""" Auto clipper for clipping gradients. """
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AutoClipper():  # pylint:disable=too-few-public-methods
    """ AutoClip: Adaptive Gradient Clipping for Source Separation Networks

    Parameters
    ----------
    clip_percentile: int
        The percentile to clip the gradients at
    history_size: int, optional
        The number of iterations of data to use to calculate the norm Default: ``10000``

    References
    ----------
    Adapted from: https://github.com/pseeth/autoclip
    original paper: https://arxiv.org/abs/2007.14469
    """
    def __init__(self, clip_percentile: int, history_size: int = 10000) -> None:
        logger.debug("Initializing %s (clip_percentile: %s, history_size: %s)",
                     self.__class__.__name__, clip_percentile, history_size)
        self._clip_percentile = clip_percentile
        self._history_size = history_size
        self._grad_history = []
        logger.debug("Initialized %s", self.__class__.__name__)

    def __call__(self, gradients: list[torch.Tensor]) -> list[torch.Tensor]:
        """ Call the AutoClip function.

        Parameters
        ----------
        gradients: list[:class:`torch.Tensor`]
            The list of gradient tensors for the optimizer

        Returns
        ----------
        list[:class:`torch.Tensor`]
            The autoclipped gradients
        """
        self._grad_history.append(sum([g.data.norm(2).item() ** 2
                                       for g in gradients if g is not None]) ** (1. / 2))
        self._grad_history = self._grad_history[-self._history_size:]
        clip_value = np.percentile(self._grad_history, self._clip_percentile)
        torch.nn.utils.clip_grad_norm_(gradients, clip_value)
        return gradients
