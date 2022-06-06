import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init


from mmdet.core import delta2bbox, force_fp32, object_target, multi_apply
from mmdet.ops import nms
from ..registry import HEADS
from .obj_anchor_head import ObjAnchorHead
import numpy as np

@HEADS.register_module
class ObjRPNHead(ObjAnchorHead):

    def __init__(self, in_channels, freeze=False, **kwargs):
        self.freeze=freeze
        super(ObjRPNHead, self).__init__(2, in_channels, **kwargs)
    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls_conv_T = nn.Conv2d(self.feat_channels, self.semantic_dims, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        #self.rpn_obj = nn.Conv2d(self.feat_channels, self.num_anchors, 1)
        if self.freeze:
            for m in [self.rpn_conv, self.rpn_cls, self.rpn_reg]:
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls_conv_T, std=0.01)
        normal_init(self.rpn_reg, std=0.01)
        #normal_init(self.rpn_obj, std=0.01)
        
        normal_init(self.vec_fb)
        # with torch.no_grad():
        #     self.vec_fb.weight.data[0] = self.vec_bg_weight.unsqueeze(-1).unsqueeze(-1)
        #     self.vec_fb.weight.data[2] = self.vec_bg_weight.unsqueeze(-1).unsqueeze(-1)
        #     self.vec_fb.weight.data[4] = self.vec_bg_weight.unsqueeze(-1).unsqueeze(-1)

    def forward_single(self, x):
        #import pdb;pdb.set_trace()

        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        # B C(900) W H
        rpn_cls_score = self.rpn_cls_conv_T(x)
        if self.voc:
            rpn_cls_score = self.voc_conv(rpn_cls_score)
        if self.high_order:
            #import pdb;pdb.set_trace()
            if self.sinkhorn_arg:
                voc_select = self.voc_base[:,self.select_voc_index].permute(1,2,0).view(-1,self.semantic_dims)
                with torch.no_grad():
                    self.ho_conv.weight.data = voc_select.unsqueeze(-1).unsqueeze(-1)
            else:
                 self.ho_conv.weight.data = self.voc_base.unsqueeze(-1).unsqueeze(-1)
            ho_feat = self.ho_conv(rpn_cls_score) # feat <-> voc_select

            bg_fg_all = self.vec_fb.weight.data.squeeze(-1).squeeze(-1)
            cost_all =  torch.mm(bg_fg_all, self.voc_base) # bg_fg <-> voc_all
            ho_bg_fg_all = torch.gather(cost_all,1,self.select_voc_index.repeat(3,1)) #bg_fg <-> voc_select  [6,256]
            with torch.no_grad():
                self.ho_sim_bg.weight.data = ho_bg_fg_all[0::2].unsqueeze(-1).unsqueeze(-1)
                self.ho_sim_fg.weight.data = ho_bg_fg_all[1::2].unsqueeze(-1).unsqueeze(-1)
            ho_feat_bg, ho_feat_fg = ho_feat.split(256,1)
            bf_index = [0,3,1,4,2,5]
            rpn_cls_score = torch.cat([self.ho_sim_bg(ho_feat_bg),self.ho_sim_fg(ho_feat_fg)],1)[:,bf_index,:,:]
        else:
            rpn_cls_score = self.vec_fb(rpn_cls_score)
        rpn_bbox_pred = self.rpn_reg(x)
        #rpn_objectness_pred = self.rpn_obj(x)
        if self.sync_bg:
            return rpn_cls_score, rpn_bbox_pred,\
                (self.vec_fb.weight.data[0] + self.vec_fb.weight.data[2]+ self.vec_fb.weight.data[4]) / 3.0
        return rpn_cls_score, rpn_bbox_pred

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples= None , cfg= None  ):
        # classification loss

        #import pdb;pdb.set_trace()
           
        labels = labels.reshape(-1, self.cls_out_channels)
        if self.use_sigmoid_cls:
            label_weights = label_weights.reshape(-1, self.cls_out_channels)
        else:
            label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        # objectness loss
        #import pdb;pdb.set_trace() 
        return loss_cls, loss_bbox



    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_masks,
             gt_labels,
             img_metas, 
             cfg,
             gt_bboxes_ignore=None):
        #import pdb;pdb.set_trace() 
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_obj_targets = object_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            gt_masks,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            objectness_type = self.objectness_type)
        if cls_reg_obj_targets is None:
            return None
        #import pdb;pdb.set_trace()
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_obj_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

              
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        return dict(
            loss_rpn_cls=losses_cls, 
            loss_rpn_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        """
        Transform network output for a batch into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): size / scale info for each image
            cfg (mmcv.Config): test / postprocessing configuration
            rescale (bool): if True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the class index of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) 
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(
                cls_scores[i].size()[-2:],
                self.anchor_strides[i],
                device=device) for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list



    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            #import pdb;pdb.set_trace()
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                rpn_cls_score = rpn_cls_score.sigmoid()
                scores = rpn_cls_score
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]

            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals


