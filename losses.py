from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    return F.one_hot(labels.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def soft_dice_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
    include_background: bool = False,
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    targets = one_hot(labels, num_classes=num_classes)

    if not include_background:
        probabilities = probabilities[:, 1:]
        targets = targets[:, 1:]

    reduce_dims = (0, 2, 3, 4)
    intersection = (probabilities * targets).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
    return (2.0 * intersection + smooth) / (denominator + smooth)


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int, weight_ce: float = 1.0, weight_dice: float = 1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.weight_ce = float(weight_ce)
        self.weight_dice = float(weight_dice)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim == 5 and labels.shape[1] == 1:
            ce_target = labels[:, 0]
        else:
            ce_target = labels

        ce_loss = F.cross_entropy(logits, ce_target.long())
        dice_loss = 1.0 - soft_dice_score(logits, labels, self.num_classes).mean()
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss
