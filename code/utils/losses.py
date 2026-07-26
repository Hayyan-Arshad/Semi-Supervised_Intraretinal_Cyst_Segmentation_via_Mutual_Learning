import torch
import torch.nn as nn


def sigmoid_dice_loss(inputs, targets):
    inputs = torch.sigmoid(inputs)
    targets = torch.sigmoid(targets) if targets.dtype.is_floating_point else targets.float()
    smooth = 1e-5
    intersection = torch.sum(inputs * targets)
    denominator = torch.sum(inputs) + torch.sum(targets)
    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets.float().unsqueeze(1)
        return self.bce(logits, targets) + sigmoid_dice_loss(logits, targets)

