from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, cat, interpolate
from detectron2.structures import Instances, heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling import ROI_KEYPOINT_HEAD_REGISTRY, BaseKeypointRCNNHead
from IPython import embed

from our_modeling.keypoint_structure import keypoints_to_heatmap
from torch.nn import functional as F

@ROI_KEYPOINT_HEAD_REGISTRY.register()
class MyKeypointHead(BaseKeypointRCNNHead, nn.Sequential):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    It is described in Sec. 5 of :paper:`Mask R-CNN`.
    """
    
    @configurable
    def __init__(self, input_shape, *, num_keypoints, conv_dims, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
        """
        super().__init__(num_keypoints=num_keypoints, **kwargs)
        embed()
        # default up_scale to 2.0 (this can be made an option)
        up_scale = 2.0
        in_channels = input_shape.channels

        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn{}".format(idx), module)
            self.add_module("conv_fcn_relu{}".format(idx), nn.ReLU())
            in_channels = layer_channels

        deconv_kernel = 4
        self.score_lowres = ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.up_scale = up_scale

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["input_shape"] = input_shape
        ret["conv_dims"] = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        return ret
    
    
    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input 4D region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses if in training. The predicted "instances" if in inference.
        """
        x = self.layers(x) 
        if self.training:
            num_images = len(instances)
            normalizer = (
                None if self.loss_normalizer == "visible" else num_images * self.loss_normalizer
            )
            return {
                "loss_keypoint": keypoint_rcnn_loss(x, instances, normalizer=normalizer)
                * self.loss_weight
            }
        else:
            keypoint_rcnn_inference(x, instances)
            return instances
        

    def layers(self, x): # yahi to hai layers jiska output ab num instance x 9+ 9+  x 56 x56 hoga
        for layer in self:
            x = layer(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x
    
    
def keypoint_rcnn_loss(pred, instances, normalizer):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.
            
    Returns a scalar tensor containing the loss.

    PSEUDOCODE:
    N = total_instances (num instances per image * batch size)
    C = num ori = 9
    M = num instances in an image
    NC = num centerpoints
    #TODO (TP): check if predicted heatmap va;ue is in between 0 and 1
    heatmap_pred: NxCx56x56                                   
    gt_centerpoints: MxNCx3                
    gt_orientation: MxNCx1  (one hot vector)
    
####################################################################
####################################################################
 
   def heatmap_loss(): 
        heatmapssss_gt = []
        ## converting ground truth centerpoints to heatmap
        for single_image in images:
            heatmap_gt, width_htgt, scale_htgt = keypoints_to_heatmap(gt_centerpoints, gt_orientation, 
                                        gt_widths, gt_scales, gt_boxes) # MxCx56x56
            heatmapssss_gt.append(heatmap_gt)
        heatmapssss_gt = torch.cat(heatmapssss_gt, axis=0) NxCx56x56
        heatmap_loss (NxCx56x56)= sum(gt*log(p) + (1-gt)*log(1-p) for each pixel)
        width_loss = (width_htgt - widths_pred)^2
        scale_loss = (scale_htgt - scales_pred)^2
        return heatmap_loss + width_loss + scale_loss
    

    def keypoints_to_heatmap(gt_centerpoints, gt_orientation, gt_widths, gt_scales, props): # MxCx56x56
        hh = []
        ww = []
        ss = []
        for i in range(M): 
            centers = gt_centerpoints[i]  # NCx3
            oris = gt_orientation[i]      # NCx1
            widths = gt_widths[i]          #NCx1
            scales = gt_scales[i]           #NCx1
            prop_box = props[i]
            heatmap_ret = np.zeros(C, 56, 56)
            width_ret = np.zeros(C, 56, 56)
            scale_ret = np.zeros(C, 56, 56)
            for j in range(NC):
                center = centers[j]  # 3
                ori = oris[j]        # ori index
                
                ## map this to heatmap_ret
                
                map_x, map_y, in_box = mapper(center, prop_box, heatmap_size) # map center to 56x56
                if in_box:
                    heatmap_ret[ori, map_x, map_y] = 1
                    width_ret[ori, map_x, map_y] = widths[j]
                    scale_ret[ori, map_x, map_y] = scales[j]
                    
            hh.append(heatmap_ret)
            ww.append(width_ret)
            ss.append(scale_ret)    
        
        return torch.tensor(hh), torch.tensor(ww), torch.tensor(ss)
            
                
    
    def mapper(center, prop_box, heatmap_size): 
        # center: 3 (x,y,v)
        # prop_box: 4 (x,y,x,y)
        top_x, top_y, bottom_x, bottom_y = prop_box
        scale_x =  heatmap_size/(bottom_x - top_x)
        scale_y = heatmap_size/(bottom_y - top_y)
        x = (center[0] - top_x) * scale_x
        x = x.floor().long()
        y = (center[1] - top_y) * scale_y
        y = y.floor().long()
        
        inside_xrange = x>=0 and x<=(heatmap_size-1)
        inside_yrange = y>=0 and y<=(heatmap_size-1)
        
        if inside_xrange and inside_yrange:
            return x,y,True
        else:
            return x,y,false
        
####################################################################
####################################################################

    def map_centerpoints(props, heatmaps): # for each image instance
        props: (num_instances, 4)  # num_instances are for 1 image, for eg 14
        heatmaps: (num_instances, num_orientations, 56, 56)
        
        offset_x = rois[:, 0]
        offset_y = rois[:, 1]
        
        widths = rois[:, 2] - rois[:, 0]
        heights = rois[:, 3] - rois[:, 1]
        ret = {}
        for i in range(num_instances):
            roi_inst = props[i]
            heatmap = heatmaps[i]
            off_x = offset_x[i]
            off_y = offset_y[i]
            w = widths[i]
            h = heights[i]
            
            roi_heatmap = F.interpolate([heatmaps], size = (h,w), mode = 'bicubic', align_corners = True) # (1,num_ori,h,w)
            ret[i] = {"roi_heatmap": roi_heatmap, "off_x": off_x, "off_y": off_y}
            
        return ret
    
        

        
    
        

        
            
    
    """
    heatmaps = []  
    valid = []
    num_ori = 9
    heatmap_pred = pred[:, : num_ori]  # shape 127, 9, 56, 56
    widths_pred = pred[:, num_ori: 2*num_ori] # shape 127, 9, 56, 56
    scales_pred = pred[:, 2*num_ori : 3*num_ori] # shape 127, 9, 56, 56
    kpts_offsets_pred = pred[:, 3*num_ori : 3*num_ori + 2*4*num_ori] # shape 127, 72, 56, 56
    center_reg_pred = pred[:, 11*num_ori :] ## shape 127, 2, 56, 56

    
    
    keypoint_side_len = pred.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        center_points = instances_per_image.gt_centerpoints
        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(
            center_points.tensor, instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred.sum() * 0

    N, K, H, W = pred.shape
    pred = pred.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        pred[valid], keypoint_targets[valid], reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    return keypoint_loss


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
 




