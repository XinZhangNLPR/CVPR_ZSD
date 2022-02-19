import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss
from itertools import product
from torch.nn import BCEWithLogitsLoss
@LOSSES.register_module
class PaiRankLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 topk = 5,
                 weight_by_diff = True):
        super(PaiRankLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.topk = topk
        self.weight_by_diff = weight_by_diff
        if self.weight_by_diff:
            self.scale = 0.1
            #self.scale = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)

    def forward(self,
                cls_score,
                labels,
                true_rank = None,
                vec_seen = None, 
                vec_unseen = None):
        #import pdb;pdb.set_trace()

        rank_value, rank_index = true_rank

        fg_score = cls_score[labels!=0]
        fg_labels = labels[labels!=0]-1
        scores = F.softmax(fg_score, dim=1) if fg_score is not None else None
        
        scores_pre = torch.mm(scores, vec_seen.t())
        fg_vec = torch.cat((vec_seen[:,1:],vec_unseen[:,1:]),1)
        scores_all = torch.mm(scores_pre, fg_vec)

        rank_index = torch.index_select(rank_index,0,fg_labels)
        rank_value = torch.index_select(rank_value,0,fg_labels)
        
        ###select top k 
        rank_index_topk = rank_index[:,:self.topk]

        y_pred = scores_all.gather(dim=1, index=rank_index_topk) #y_pred:N*topk

        y_true = rank_value[:,:self.topk] #y_true:N*topk        

        document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

        pairs_true = y_true[:, document_pairs_candidates]
        selected_pred = y_pred[:, document_pairs_candidates]  

        true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
        pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

        the_mask = (true_diffs > 0)
        pred_diffs = pred_diffs[the_mask]   

        weight = None
        if self.weight_by_diff:
            abs_diff = torch.abs(2*torch.sigmoid(true_diffs/self.scale)-1)
            weight = abs_diff[the_mask]

        true_diffs = (true_diffs > 0).type(torch.float32)
        true_diffs = true_diffs[the_mask]

        return self.loss_weight*BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)