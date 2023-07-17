from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, interpolate
from detectron2.structures import Instances

from detectron2.modeling import ROI_KEYPOINT_HEAD_REGISTRY, BaseKeypointRCNNHead
from IPython import embed

from our_modeling.new_loss import kgn_loss, keypoint_rcnn_inference
from torch.nn import functional as F

@ROI_KEYPOINT_HEAD_REGISTRY.register()
class MyKeypointHead(BaseKeypointRCNNHead, nn.Sequential):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    It is described in Sec. 5 of :paper:`Mask R-CNN`.
    """

    @configurable
    def __init__(self, input_shape, *, num_outputs, loss_weight_tuple, conv_dims, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
        """
      
        super().__init__(**kwargs)
        
        self.loss_weight_tuple = loss_weight_tuple
        
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
            in_channels, num_outputs, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
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
        # ret = super().from_config(cfg, input_shape)
        ret = {}
        ret["input_shape"] = input_shape
        ret["conv_dims"] = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        if cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS:
            ret["num_keypoints"] = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        else:
            ret["num_keypoints"] = None
        ret["num_outputs"] = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_OUTPUTS
        ret["loss_weight_tuple"] = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT_TUPLE
    
        normalize_by_visible = (
            cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS
        )  # noqa
        if not normalize_by_visible:
            batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
            positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
            ret["loss_normalizer"] = (
                cfg.MODEL.ROI_HEADS.AVG_NUM_GRASPS * batch_size_per_image * positive_sample_fraction
            )
        else:
            ret["loss_normalizer"] = "visible"
            """
        loss_normalizer (float or str):
                If float, divide the loss by `loss_normalizer * #images`.
                If 'visible', the loss is normalized by the total number of
                visible keypoints across images."""
                
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
            
            hm_loss, width_loss = kgn_loss(x, instances, normalizer)
            
            return {
                "loss_hm": hm_loss * self.loss_weight_tuple[0],
                
                "loss_width": width_loss * self.loss_weight_tuple[1]
            }
        else:
            keypoint_rcnn_inference(x, instances)
            return instances
        

    def layers(self, x): # yahi to hai layers jiska output ab num instance x 9+ 9+  x 56 x56 hoga
        for layer in self:
            x = layer(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x
    
    
    
    



