import numpy as np
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, interpolate
from detectron2.structures import Instances

from detectron2.modeling import ROI_KEYPOINT_HEAD_REGISTRY, BaseKeypointRCNNHead
from IPython import embed

from model.our_modeling.new_loss import kgn_loss, keypoint_rcnn_inference
from torch.nn import functional as F

from model.our_modeling.encode_decode import Encoder, Decoder
from utils.util import get_grasp_features_v2, save_results

from utils.post_process import post_process


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class MyKeypointHead(BaseKeypointRCNNHead, nn.Sequential):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    It is described in Sec. 5 of :paper:`Mask R-CNN`.
    """

    @configurable
    def __init__(
        self,
        input_shape,
        *,
        num_outputs,
        loss_weight_tuple,
        conv_dims,
        use_vae,
        hidden_dims=None,
        latent_dim=None,
        grasp_input_dim=256,
        num_outputs_vae=None,
        eval_save_results=False,
        eval_output_dir=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.loss_weight_tuple = loss_weight_tuple
        self.use_vae = use_vae
        in_channels = input_shape.channels

        self.eval_save_results = eval_save_results
        self.eval_output_dir = eval_output_dir
        self.inference_latent_sampler = torch.distributions.normal.Normal(
            torch.zeros(latent_dim, dtype=torch.float32),
            torch.ones(latent_dim, dtype=torch.float32))

        if self.use_vae:
            self.avg_pool = nn.AvgPool2d((input_shape.height, input_shape.width))
            # self.encoder = Encoder(
            #     in_channels + num_outputs_vae, hidden_dims, latent_dim
            # )
            self.encoder = Encoder(
                in_channels + grasp_input_dim, hidden_dims, latent_dim
            )
            self.grasp_encoder = nn.Sequential(
                nn.Linear(10, 128),  # 10: 8 for kpts and 2 for center points
                # nn.Linear(2, 128), # assuming only training with centerpoints
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, grasp_input_dim),
            )
            self.decoder = Decoder(
                hidden_dims, latent_dim + in_channels, num_outputs_vae, offsets=True
            )
            self.latent_dim = latent_dim

        else:
            up_scale = 2.0

            for idx, layer_channels in enumerate(conv_dims, 1):
                module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
                self.add_module("conv_fcn{}".format(idx), module)
                self.add_module("conv_fcn_relu{}".format(idx), nn.ReLU())
                in_channels = layer_channels

            deconv_kernel = 4
            self.score_lowres = ConvTranspose2d(
                in_channels,
                num_outputs,
                deconv_kernel,
                stride=2,
                padding=deconv_kernel // 2 - 1,
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

        ret["use_vae"] = cfg.MODEL.ROI_KEYPOINT_HEAD.USE_VAE
        if ret["use_vae"]:
            ret["hidden_dims"] = cfg.MODEL.ROI_KEYPOINT_HEAD.VAE.HIDDEN_DIMS
            ret["latent_dim"] = cfg.MODEL.ROI_KEYPOINT_HEAD.VAE.LATENT_DIM
            ret["num_outputs_vae"] = cfg.MODEL.ROI_KEYPOINT_HEAD.VAE.NUM_OUTPUTS_VAE

        normalize_by_visible = (
            cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS
        )  # noqa
        if not normalize_by_visible:
            batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
            positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
            ret["loss_normalizer"] = (
                cfg.MODEL.ROI_HEADS.AVG_NUM_GRASPS
                * batch_size_per_image
                * positive_sample_fraction
            )
        else:
            ret["loss_normalizer"] = "visible"
            """
        loss_normalizer (float or str):
                If float, divide the loss by `loss_normalizer * #images`.
                If 'visible', the loss is normalized by the total number of
                visible keypoints across images."""

        ret["eval_save_results"] = cfg.TEST.EVAL_SAVE_RESULTS
        if ret["eval_save_results"]:
            ret["eval_output_dir"] = cfg.TEST.EVAL_OUTPUT_DIR

        return ret

    def forward(self, x_input: torch.Tensor, instances: List[Instances]):
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
        if not self.use_vae:
            x_output = self.layers(x_input)
            if self.training:
                num_images = len(instances)
                normalizer = (
                    None
                    if self.loss_normalizer == "visible"
                    else num_images * self.loss_normalizer
                )

                hm_loss, width_loss = kgn_loss(x_output, instances, normalizer)

                return {
                    "loss_hm": hm_loss * self.loss_weight_tuple[0],
                    "loss_width": width_loss * self.loss_weight_tuple[1],
                }
            else:
                keypoint_rcnn_inference(x_output, instances)
                return instances

        else:
            x_input = self.avg_pool(x_input)
            x_input_flattened = x_input.flatten(start_dim=1)

            if self.training:
                grasp_features = get_grasp_features_v2(instances, offsets=True)
                # grasp_features = get_grasp_features(
                #     instances
                # )  # kpts offset + centerpoints
                grasp_features_encoded = self.grasp_encoder(grasp_features)
                x_concat = torch.cat(
                    (x_input_flattened, grasp_features_encoded), axis=1
                )
                mu, log_var = self.encoder(x_concat)  # batch size x num_latents
                z = self.reparameterize(mu, log_var)  # batch size x num_latents

                concatenated_embeddings = torch.cat((x_input_flattened, z), axis=1)

                x_offset_output, x_cp_output = self.decoder(concatenated_embeddings)
                loss_dict = self.grasp_sampler_loss(
                    torch.cat((x_offset_output, x_cp_output), axis=1),
                    grasp_features,
                    mu,
                    log_var,
                )
                return loss_dict
            else:
                self.grasp_sampler_inference(input_image_features=x_input_flattened, instances=instances)
                return instances

    def layers(self, x):
        for layer in self:
            x = layer(x)
        x = interpolate(
            x, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        )
        return x

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def grasp_sampler_loss(
        self, grasp_features_output, grasp_features_input, mu, log_var
    ) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons_loss = F.mse_loss(grasp_features_output, grasp_features_input)
        # kld_loss = torch.mean(
        #     -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        # )
        
        
        kld_loss = torch.mean(
            torch.sum(log_var + (1+mu**2)/(2*log_var.exp()**2) - 0.5, dim=1), dim=0
        )
        # loss = recons_loss + kld_loss
        return {"reconstruction_loss": recons_loss, "KLDivergence_loss": kld_loss}

    def grasp_sampler_inference(self, input_image_features, instances):
        print("in test", instances)
        num_object_instances = len(input_image_features)
        # z = torch.randn(num_object_instances, self.latent_dim)
        z = self.inference_latent_sampler.sample((num_object_instances,))
        x_concat = torch.cat((input_image_features, z), axis=1)
        
        output = self.decoder(x_concat)
        post_process(output, instances)


#  x = dataset_dicts[0]; annotations = x["annotations"]; ann = annotations[0]
