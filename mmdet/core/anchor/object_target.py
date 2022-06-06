import torch
import cv2
from mmdet.core.bbox.assign_sampling import build_sampler
import numpy as np
from ..bbox import PseudoSampler, assign_and_sample, bbox2delta, build_assigner
from ..utils import multi_apply
from mmdet.core.bbox import bbox_overlaps

def object_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  gt_masks_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True,
                  objectness_type = 'superpixel'):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """

    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         object_target_single,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_masks_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs,
         objectness_type = objectness_type)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    #import pdb;pdb.set_trace()
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)


    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def object_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_masks,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True,
                         objectness_type = 'superpixel'):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)


    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    #labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            #labels[pos_inds] = 1
            if objectness_type == 'superpixel':
                #import pdb;pdb.set_trace()
                #pos_objectness_targets1 = bbox_overlaps(objectness_sampling_result.pos_bboxes, objectness_sampling_result.pos_gt_bboxes,is_aligned=True)
                #pos_objectness_targets = compute_superpixel_scores(objectness_sampling_result.pos_bboxes, pos_assigned_gt_inds, gt_masks,0.761)             
                pos_objectness_targets = compute_superpixel_scores(sampling_result.pos_bboxes,
                                        sampling_result.pos_assigned_gt_inds, 
                                        gt_masks)            
            elif objectness_type == 'Centerness':
                pos_objectness_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                        sampling_result.pos_gt_bboxes,
                                        target_means, target_stds)
                valid_targets = torch.min(pos_objectness_bbox_targets,-1)[0] > 0
                pos_objectness_bbox_targets[valid_targets==False,:] = 0
                top_bottom = pos_objectness_bbox_targets[:,0:2]
                left_right = pos_objectness_bbox_targets[:,2:4]
                pos_objectness_targets = torch.sqrt(
                    (torch.min(top_bottom, -1)[0] / 
                        (torch.max(top_bottom, -1)[0] + 1e-12)) *
                    (torch.min(left_right, -1)[0] / 
                        (torch.max(left_right, -1)[0] + 1e-12)))
            elif objectness_type == 'BoxIoU':
                pos_objectness_targets = bbox_overlaps(
                    sampling_result.pos_bboxes,
                    sampling_result.pos_gt_bboxes,
                    is_aligned=True)
            #import pdb;pdb.set_trace()
            labels[pos_inds] = pos_objectness_targets.clone().detach()

        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight

    #import pdb;pdb.set_trace()



    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0





    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,neg_inds)


def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


def compute_superpixel_scores(pos_proposals, pos_assigned_gt_inds, gt_masks, score_thresh=None):
    scores = []
    proposals_np = pos_proposals.cpu().numpy()
    pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
    for i in range(proposals_np.shape[0]):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        fg_p_all = (gt_mask==1).sum()
        bbox = proposals_np[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)
        fg_p_inbox = (gt_mask[y1:y1 + h, x1:x1 + w]==1).sum()
        #import pdb;pdb.set_trace()
        score = fg_p_inbox/fg_p_all
        scores.append(score)
        # if score_thresh:
        #     if round(score,3) == score_thresh:
        #         rec_mask = cv2.rectangle(gt_mask*256, (int(x1), int(y1)), (int(x2), int(y2)),(255, 0, 255),2)
        #         rec_mask = cv2.putText(rec_mask, str(score), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        #         cv2.imwrite('rec_mask.jpg',rec_mask)
    scores = torch.from_numpy(np.stack(scores)).float().to(pos_proposals.device) 
    return scores  