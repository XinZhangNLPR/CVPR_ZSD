import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss

@LOSSES.register_module
class ListMLELoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(ListMLELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                true_rank,
                **kwargs):
        #import pdb;pdb.set_trace()

        pred_sorted_by_true = cls_score.gather(dim=1, index=true_rank+1)
        max_pred_scores, _ = pred_sorted_by_true.max(dim=1, keepdim=True)
        pred_sorted_by_true_minus_max = pred_sorted_by_true - max_pred_scores
        cumsums = pred_sorted_by_true_minus_max.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + 1e-10) - pred_sorted_by_true_minus_max
        loss_cls = observation_loss.sum(dim=1).mean()
        loss_cls = loss_cls*self.loss_weight
        return loss_cls
