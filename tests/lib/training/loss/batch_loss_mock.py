"""Mock BatchLoss for testing LossUnit.

This module provides a simple mock implementation of BatchLoss for unit testing
LossUnit without requiring actual model training.
"""
from __future__ import annotations

from lib.utils import get_module_objects

import typing as T

import torch

if T.TYPE_CHECKING:
    import numpy.typing as npt


class BatchLossMock:
    """Mock BatchLoss for testing.

    This is a lightweight mock that provides the essential interface of BatchLoss
    for testing LossUnit without requiring actual model training.

    Parameters
    ----------
    unweighted
        List of dictionaries mapping loss function names to their raw (unweighted)
        tensor losses. Each dictionary corresponds to one sample in the batch.
    weighted
        List of dictionaries mapping loss function names to their weighted tensor losses.
    mask
        Optional mask loss tensor (if learn_mask is enabled)

    Notes
    -----
    This mock is designed to be compatible with LossUnit's expectations:
    - Each item in unweighted/weighted is a dict with loss function names as keys
    - Values are torch tensors (scalars or tensors)
    - mask is a torch tensor or None
    """

    def __init__(
        self,
        unweighted: list[dict[str, torch.Tensor]],
        weighted: list[dict[str, torch.Tensor]],
        mask: torch.Tensor | None = None,
    ) -> None:
        """Initialize mock BatchLoss."""
        # Ensure all tensors have proper batch dimension (1-d)
        self.unweighted = [{k: v if v.dim() > 0 else v.unsqueeze(0) for k, v in x.items()} for x in unweighted]
        self.weighted = [{k: v if v.dim() > 0 else v.unsqueeze(0) for k, v in x.items()} for x in weighted]
        self.mask = mask if mask is None or mask.dim() > 0 else mask

    @property
    def total(self) -> torch.Tensor:
        """Compute total loss."""
        total = sum(sum(y.mean() for y in x.values()) for x in self.weighted)
        if self.mask is not None:
            total += self.mask.mean()
        return total

    def get_contributions(self) -> dict[T.Literal["unweighted", "weighted"], dict[str, torch.Tensor]]:
        """Extract mean contributions from each loss function.
        
        Computes average contribution per loss function across all batch samples and groups
        results into unweighted and weighted categories.
        
        Returns
        -------
        Dictionary with keys "unweighted" and "weighted", each mapping loss function
        names (strings) to their mean contribution tensors across the batch.
        
        Notes
        -----
        Returns 1-d tensors for proper iteration in LossUnit._update_averages.
        """
        # Build contributions dict with proper structure
        contributions = {
            "unweighted": {},
            "weighted": {}
        }
        
        # Aggregate losses across all samples in the batch
        for sample in self.unweighted:
            for name, value in sample.items():
                if name not in contributions["unweighted"]:
                    contributions["unweighted"][name] = torch.zeros((1,), dtype=torch.float32)
                contributions["unweighted"][name] += value
        
        for sample in self.weighted:
            for name, value in sample.items():
                if name not in contributions["weighted"]:
                    contributions["weighted"][name] = torch.zeros((1,), dtype=torch.float32)
                contributions["weighted"][name] += value
        
        # Compute means (divide by number of samples)
        num_samples = len(self.weighted) if self.weighted else 1
        for key in ["unweighted", "weighted"]:
            for name in contributions[key]:
                contributions[key][name] = contributions[key][name] / num_samples
        
        return contributions

    def detach(self) -> "BatchLossMock":
        """Return detached copy."""
        return BatchLossMock(
            unweighted=[{k: v.detach() for k, v in x.items()} for x in self.unweighted],
            weighted=[{k: v.detach() for k, v in x.items()} for x in self.weighted],
            mask=self.mask.detach() if self.mask is not None else None,
        )

    def to_cpu(self) -> "BatchLossMock":
        """Return CPU copy."""
        return BatchLossMock(
            unweighted=[{k: v.cpu() for k, v in x.items()} for x in self.unweighted],
            weighted=[{k: v.cpu() for k, v in x.items()} for x in self.weighted],
            mask=self.mask.cpu() if self.mask is not None else None,
        )


__all__ = get_module_objects(__name__)
