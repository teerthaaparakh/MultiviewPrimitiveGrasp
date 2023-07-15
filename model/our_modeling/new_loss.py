import numpy as np 
from utils.other_configs import *
import torch
from torch.nn import functional as F 
from detectron2.structures import Instances, heatmaps_to_keypoints
from detectron2.layers import cat


def kgn_loss(pred, instances):
    num_ori = 9
    heatmap_pred = pred[:, : num_ori]  # shape 127, 9, 56, 56
    widths_pred = pred[:, num_ori: 2*num_ori] # shape 127, 9, 56, 56
    scales_pred = pred[:, 2*num_ori : 3*num_ori] # shape 127, 9, 56, 56
    kpts_offsets_pred = pred[:, 3*num_ori : 3*num_ori + 2*4*num_ori] # shape 127, 72, 56, 56
    center_reg_pred = pred[:, 11*num_ori :] ## shape 127, 2, 56, 56
    
    heatmaps_gt = []
    widths_gt = []
    scales_gt = []
    kpts_offsets_pred = []
    center_reg_pred = []
    
    for inst in instances: # batch 4 images, take one image
        
        gt_centerpoints = inst.gt_centerpoints
        gt_orientation = inst.gt_orientations
        gt_widths = inst.gt_widths
        gt_scales = inst.gt_scales
        prop_boxes = inst.proposal_boxes.tensor
        
        hm_gt, width_htgt = keypoints_to_heatmap(gt_centerpoints, gt_orientation, 
                                        gt_widths, gt_scales, prop_boxes)

        heatmaps_gt.append(hm_gt)
        widths_gt.append(width_htgt)
        
    # heatmap loss
    heatmaps_gt = torch.cat(heatmaps_gt, axis = 0)
    #per pixel cross entropy loss
    hm_loss = F.binary_cross_entropy(heatmap_pred, heatmaps_gt, reduction='mean')
    
    # widths loss
    widths_gt = torch.cat(widths_gt, axis = 0)
    width_loss = F.l1_loss(widths_pred, widths_gt)
    
    # scale loss
    # TODO (TP): get ground truth scales and calculate error
    
    # kpts offset loss
    # TODO (TP): ??
    
    # center reg loss
    # TODO (TP): ??
    
    loss = HM_WT * hm_loss + WD_WT * width_loss 
    
    return loss


def keypoints_to_heatmap(gt_centerpoints, gt_orientation, 
                                        gt_widths, prop_boxes, 
                                        gt_scales=None): # (M,C,56,56)
    
    heatmaps = []
    widths = []
    scales = []
    M, C, H, W = gt_centerpoints.shape
    for i in range(M):
        cen = gt_centerpoints[i]  # NCx3
        ori = gt_orientation[i]     # NCx1
        wi = gt_widths[i]           # NCx1
        box = prop_boxes[i]
        heatmap_ret = np.zeros(C, H, W)
        width_ret = np.zeros(C, H, W)
        
        NC = len(cen)
        for j in range(NC):
            single_center = cen[j]
            single_ori = ori[j]
            
            map_x, map_y, in_box = mapper(single_center, box, H)
            if in_box:
                heatmap_ret[single_ori, map_x, map_y] = 1
                width_ret[ori, map_x, map_y] = wi[j]
                
        heatmaps.append(heatmap_ret)
        widths.append(width_ret)
    
    heatmaps = np.array(heatmaps)
    widths = np.array(widths)
        
    return torch.from_numpy(heatmaps), torch.from_numpy(widths)


def mapper(center, box, hm_size):
    # center: 3 (x,y,v)
    # box: 4 (x,y,x,y)
    top_x, top_y, bottom_x, bottom_y = box
    scale_x =  hm_size/(bottom_x - top_x)
    scale_y = hm_size/(bottom_y - top_y)
    x = (center[0] - top_x) * scale_x
    x = x.floor().long()
    y = (center[1] - top_y) * scale_y
    y = y.floor().long()
    
    inside_xrange = x>=0 and x<=(hm_size-1)
    inside_yrange = y>=0 and y<=(hm_size-1)
    
    if inside_xrange and inside_yrange:
        return x,y,True
    else:
        return x,y,False
    
    
    
def keypoint_rcnn_inference(pred_keypoint_logits: torch.Tensor, pred_instances: List[Instances]):
    """
    Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score)
        and add it to the `pred_instances` as a `pred_keypoints` field.

    Args:
        pred_keypoint_logits (Tensor): A tensor of shape (R, K, S, S) where R is the total number
           of instances in the batch, K is the number of keypoints, and S is the side length of
           the keypoint heatmap. The values are spatial logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images.

    Returns:
        None. Each element in pred_instances will contain extra "pred_keypoints" and
            "pred_keypoint_heatmaps" fields. "pred_keypoints" is a tensor of shape
            (#instance, K, 3) where the last dimension corresponds to (x, y, score).
            The scores are larger than 0. "pred_keypoint_heatmaps" contains the raw
            keypoint logits as passed to this function.
    """
    # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
    bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

    pred_keypoint_logits = pred_keypoint_logits.detach()
    keypoint_results = heatmaps_to_keypoints(pred_keypoint_logits, bboxes_flat.detach())
    num_instances_per_image = [len(i) for i in pred_instances]
    keypoint_results = keypoint_results[:, :, [0, 1, 3]].split(num_instances_per_image, dim=0)
    heatmap_results = pred_keypoint_logits.split(num_instances_per_image, dim=0)

    for keypoint_results_per_image, heatmap_results_per_image, instances_per_image in zip(
        keypoint_results, heatmap_results, pred_instances
    ):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
        # heatmap_results_per_image is (num instances)x(num keypoints)x(side)x(side)
        instances_per_image.pred_keypoints = keypoint_results_per_image
        instances_per_image.pred_keypoint_heatmaps = heatmap_results_per_image
 





            
    
        
        
        

        
        