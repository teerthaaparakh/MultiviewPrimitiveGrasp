import sys, os, os.path as osp

sys.path.append(os.getenv("KGN_DIR"))
from utils.other_configs import *
import torch
from torch.nn import functional as F
from detectron2.structures import Instances
from detectron2.layers import cat
from detectron2.structures import Keypoints
from typing import List
import numpy as np
import itertools


def kgn_loss(pred, instances, normalizer=None):
    num_ori = NUM_BINS
    heatmap_pred = pred[:, :num_ori]  # shape 127, 9, 56, 56
    widths_pred = pred[:, num_ori : 2 * num_ori]  # shape 127, 9, 56, 56
    scales_pred = pred[:, 2 * num_ori : 3 * num_ori]  # shape 127, 9, 56, 56
    kpts_offsets_pred = pred[
        :, 3 * num_ori : 3 * num_ori + 2 * 4 * num_ori
    ]  # shape 127, 72, 56, 56
    center_reg_pred = pred[:, 11 * num_ori :]  # shape 127, 2, 56, 56

    heatmaps_gt = []
    widths_gt = []
    scales_gt = []
    kpts_offsets_pred = []
    center_reg_pred = []

    for inst in instances:  # batch 4 images, take one image
        assert isinstance(inst.gt_centerpoints[0], Keypoints)
        assert (
            inst.gt_centerpoints[0].tensor.shape[0] == inst.gt_orientations[0].shape[0]
        )
        assert inst.gt_centerpoints[0].tensor.shape[0] == inst.gt_widths[0].shape[0]

        gt_centerpoints = inst.gt_centerpoints
        gt_orientation = inst.gt_orientations
        gt_widths = inst.gt_widths
        prop_boxes = inst.proposal_boxes.tensor

        hm_gt, width_htgt = keypoints_to_heatmap(
            gt_centerpoints,
            gt_orientation,
            gt_widths,
            prop_boxes,
            heatmap_pred.shape[1:],
        )

        heatmaps_gt.append(hm_gt)
        widths_gt.append(width_htgt)

    # heatmap loss
    heatmaps_gt = torch.cat(heatmaps_gt, axis=0)
    # per pixel cross entropy loss
    # hm_loss = F.binary_cross_entropy(heatmap_pred, heatmaps_gt, reduction='sum')

    hm_loss = F.l1_loss(heatmap_pred, heatmaps_gt, reduction="sum")

    # widths loss
    widths_gt = torch.cat(widths_gt, axis=0)
    width_loss = F.l1_loss(widths_pred, widths_gt, reduction="sum")

    # scale loss
    # TODO (TP): get ground truth scales and calculate error

    # kpts offset loss
    # TODO (TP): ??

    # center reg loss
    # TODO (TP): ??

    if normalizer:
        hm_loss = hm_loss / normalizer
        width_loss = width_loss / normalizer

    return hm_loss, width_loss


def keypoints_to_heatmap(
    gt_centerpoints,
    gt_orientation,
    gt_widths,
    prop_boxes,
    heatmap_shape,
    gt_scales=None,
):  # (M,C,56,56)
    heatmaps = []
    widths = []
    scales = []
    C, H, W = heatmap_shape
    M = len(gt_centerpoints)
    for i in range(M):
        cen = gt_centerpoints[i].tensor  # NCx3
        ori = gt_orientation[i]  # NCx1
        wi = gt_widths[i]  # NCx1
        box = prop_boxes[i]
        heatmap_ret = np.zeros((C, H, W))
        width_ret = np.zeros((C, H, W))

        NC = len(cen)
        for j in range(NC):
            single_center = cen[j][0]
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
    scale_x = hm_size / (bottom_x - top_x)
    scale_y = hm_size / (bottom_y - top_y)
    x = (center[0] - top_x) * scale_x
    x = x.floor().long()
    y = (center[1] - top_y) * scale_y
    y = y.floor().long()

    inside_xrange = x >= 0 and x <= (hm_size - 1)
    inside_yrange = y >= 0 and y <= (hm_size - 1)

    if inside_xrange and inside_yrange:
        return x, y, True
    else:
        return x, y, False


def keypoint_rcnn_inference(
    pred_keypoint_logits: torch.Tensor, pred_instances: List[Instances]
):
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
    bboxes_flat = cat([b.proposal_boxes.tensor for b in pred_instances], dim=0)

    pred_keypoint_logits = pred_keypoint_logits.detach()
    num_instances_per_image = [len(i) for i in pred_instances]

    processed_results, selected_ids = heatmaps_to_keypoints(
        pred_keypoint_logits, bboxes_flat.detach(), num_instances_per_image
    )

    i = 0
    for processed_results_per_image, instances_per_image, selected_ids_per_image in zip(
        processed_results, pred_instances, selected_ids
    ):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
        # heatmap_results_per_image is (num instances)x(num keypoints)x(side)x(side)
        instances_per_image = instances_per_image[selected_ids_per_image]
        instances_per_image.pred_processed_results = processed_results_per_image
        pred_instances[i] = instances_per_image
        i += 1


@torch.jit.script_if_tracing
def heatmaps_to_keypoints(
    maps: torch.Tensor, rois: torch.Tensor, num_instances_per_image
) -> dict:
    num_instances = len(num_instances_per_image)  # num_instances_per_image = [4, 5, 7]
    cumsum_inst = np.insert(np.cumsum(num_instances_per_image), 0, 0)  # [0, 4, 9, 16]

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1)
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)

    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    width_corrections = widths / widths_ceil
    height_corrections = heights / heights_ceil

    num_rois, _, H, _ = maps.shape
    num_ori = NUM_BINS

    ret_list = [[]] * num_instances
    ret_list2 = [[]] * num_instances
    for instance_idx in range(num_instances):
        start = cumsum_inst[instance_idx]
        end = cumsum_inst[instance_idx + 1]

        heatmap_pred = maps[start:end, :num_ori]  # shape 127, 9, 56, 56
        widths_pred = maps[start:end, num_ori : 2 * num_ori]  # shape 127, 9, 56, 56
        scales_pred = maps[start:end, 2 * num_ori : 3 * num_ori]  # shape 127, 9, 56, 56
        kpts_offsets_pred = maps[
            start:end, 3 * num_ori : 3 * num_ori + 2 * 4 * num_ori
        ]  # shape 127, 72, 56, 56
        center_reg_pred = maps[start:end, 11 * num_ori :]  # shape 127, 2, 56, 56

        num_instances_this_image = num_instances_per_image[instance_idx]

        roi_heatmaps = []
        num_elements = []

        for i in range(num_instances_this_image):
            outsize = (int(heights_ceil[start + i]), int(widths_ceil[start + i]))
            roi_hmap = F.interpolate(
                heatmap_pred[[i]], size=outsize, mode="bicubic", align_corners=False
            )
            roi_heatmaps.append(roi_hmap[0])
            num_elements.append(roi_hmap.numel())

        selected_centers = top_centers(
            roi_heatmaps, num_elements
        )  # 100x4 (roi_num, ori_num, x, y)

        for idx in range(len(selected_centers)):
            data = selected_centers[idx]
            roi_num, ori_num, x_roi, y_roi = data

            x = (torch.tensor(x_roi).float() + 0.5) * width_corrections[start + roi_num]
            y = (torch.tensor(y_roi).float() + 0.5) * height_corrections[
                start + roi_num
            ]

            x = x + offset_x[start + roi_num]
            y = y + offset_y[start + roi_num]

            x_heat, y_heat, is_inside = mapper((x, y), rois[start + roi_num], H)

            if is_inside:
                width = widths_pred[roi_num, ori_num, x_heat, y_heat]
                scale = scales_pred[roi_num, ori_num, x_heat, y_heat]
                kpts_offset = kpts_offsets_pred[
                    roi_num, ori_num * 8 : (ori_num + 1) * 8, x_heat, y_heat
                ]
                center_reg = center_reg_pred[roi_num, :, x_heat, y_heat]
                ret_list[instance_idx].append(
                    {
                        "pixel_loc": (x, y),
                        "roi_num": roi_num,
                        "roi_box": rois[start + roi_num],
                        "width": width,
                        "scale": scale,
                        "kpts_offset": kpts_offset,
                        "center_reg": center_reg,
                    }
                )
                ret_list2[instance_idx].append(roi_num)

    return ret_list, torch.tensor(ret_list2)


def top_centers(hmap, num_elements, num=NUM_TOPS):
    # hmap shape 127, 9, 56, 56
    # return shape 100 x 4 (roi num, ori_num, x, y)
    ret = []
    num_rois = len(num_elements)
    num_elements_cumsum = np.cumsum(num_elements)
    flattened = [hmap[i].flatten() for i in range(num_rois)]

    flattened = torch.cat(flattened)

    idx = torch.topk(flattened, num).indices

    for i in range(num):
        instance_idx = np.where(((num_elements_cumsum > idx[i].numpy()) == True))[0]
        assert len(instance_idx) > 0, "index should less than total number of elements"

        instance_idx = instance_idx[0]

        if instance_idx == 0:
            idxx = idx[i] - 0
        else:
            idxx = idx[i] - num_elements_cumsum[instance_idx - 1]
        ret.append(
            (instance_idx, *np.unravel_index(idxx.numpy(), hmap[instance_idx].shape))
        )

    return ret


if __name__ == "__main__":
    import random

    num_rois = 2
    hmap = []
    num_elements = []
    for i in range(num_rois):
        height = random.randint(2, 5)
        width = random.randint(2, 5)
        hmap__ = torch.rand(2, height, width)
        hmap.append(hmap__)
        num_elements.append(2 * height * width)
    import pdb

    # pdb.set_trace()
    ret = top_centers(hmap, num_elements)
