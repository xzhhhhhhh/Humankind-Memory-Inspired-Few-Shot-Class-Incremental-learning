import torch
from torch import nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class cosLoss(nn.Module):
    def __init__(self):
        super(cosLoss, self).__init__()

    def forward(self, cos):
        return ((1 - cos).sum())

class distanceLoss(nn.Module):
    def __init__(self):
        super(distanceLoss, self).__init__()

    def forward(self, logits, labels):
        min = torch.min(logits)
        max = torch.max(logits)
        logits = 1 - ((logits - min) / (max - min))
        cos_sim = logits.sum(1) - torch.squeeze(logits.gather(0, torch.unsqueeze(labels, dim=1)))
        return - (cos_sim.sum(0) / logits.size(0) ** 2)

class SupContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(SupContrastive, self).__init__()
        self.reduction = reduction

    def get_oneHot(self, y_pred, y_true):
        y_true_oneHot = torch.zeros(y_pred.shape)
        for i, label in enumerate(y_true):
            y_true_oneHot[i][label] = 1
        return y_true_oneHot

    def forward(self, y_pred, y_true):
        y_true = self.get_oneHot(y_pred, y_true).to(device)
        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1).unsqueeze(1)
        sum_pos = (y_true * torch.exp(-y_pred))
        num_pos = y_true.sum(1)
        loss = torch.log(1 + sum_neg * sum_pos).sum(1) / num_pos

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss
