import datetime
import logging
import os
import shutil
from utils.path_util import (
    get_config_file_path,
    get_output_dir,
    get_eval_output_dir,
    get_pretrained_resnet_path,
)
from model.configs.config import add_centernet_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.logger import log_every_n_seconds
from detectron2.config import CfgNode as CN
import detectron2.utils.comm as comm
from detectron2.data import DatasetCatalog

setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from dataset.dataset_mapper_v2 import mapper
from utils.other_configs import *
import wandb
from detectron2.engine.hooks import HookBase
import time
import random
import torch

# head 64x64 reduced image size: for every pixel: 9 orientations heatmap, 9 widths, 9 scales,
# 2 * dataset.num_grasp_kpts * opt.ori_num center offset, 2 center point regression


##################################################################################################
## train evaluation hook
##################################################################################################


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.data_pts = []
        for inputs in self._data_loader:
            self.data_pts.append(inputs)
        print("Eval period:", self._period)

    def _do_loss_eval(self):
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []

        k = min(len(self.data_pts), 100)
        sampled_pts = random.sample(self.data_pts, k=k)

        total = len(sampled_pts)
        num_warmup = min(5, total - 1)

        for idx, inputs in enumerate(sampled_pts):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (
                    time.perf_counter() - start_time
                ) / iters_after_start
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        wandb.log({"validation_loss": mean_loss})
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or ((self._period > 0) and ((next_iter % self._period) == 0)):
            print("Next iter-----------------------------", next_iter)
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class MyTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    lambda ddict: mapper(ddict, draw=False, is_test=True),
                ),
            ),
        )
        # swap the order of PeriodicWriter and ValidationLoss
        # code hangs with no GPUs > 1 if this line is removed
        hooks = hooks[:-2] + hooks[-2:][::-1]
        return hooks

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        m = lambda ddict: mapper(ddict, draw=False, is_test=True)
        return build_detection_test_loader(cfg, dataset_name, mapper=m)

    @classmethod
    def build_train_loader(cls, cfg):
        m = lambda ddict: mapper(ddict, draw=False, is_test=False)

        return build_detection_train_loader(
            cfg, mapper=m, dataset=DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        )

    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name):
    #     pass


def setup(device="cpu", config_fname=None):
    """
    Create configs and perform basic setups.
    """
    if config_fname:
        config_fname = config_fname
    else:
        config_fname = get_config_file_path()
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(config_fname)

    cfg.OUTPUT_DIR = get_output_dir()
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = get_pretrained_resnet_path()
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.MAX_ITER = 1  # 40000
    cfg.SOLVER.STEPS = (30000,)
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    cfg.MODEL.PIXEL_MEAN = (0, 0, 0, 0)  # (0.5, 0.5, 0.5, 0.1)
    cfg.MODEL.PIXEL_STD = (1, 1, 1, 1)  # (0.01, 0.01, 0.01, 0.01)
    cfg.MODEL.CENTERNET.NUM_CLASSES = 6
    # cfg.MODEL.CENTERNET.POST_NMS_TOPK_TRAIN =
    cfg.MODEL.CENTERNET.POST_NMS_TOPK_TEST = 50
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
    cfg.MODEL.KEYPOINT_ON = True
    cfg.MODEL.ROI_HEADS.NAME = "MyROIHeads"
    cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = (
        "MyKeypointHead"  # KRCNNConvDeconvUpsampleHead default
    )
    cfg.MODEL.ROI_KEYPOINT_HEAD.USE_VAE = True
    # if cfg.MODEL.ROI_KEYPOINT_HEAD.USE_VAE:
    cfg.MODEL.ROI_KEYPOINT_HEAD.VAE = CN()
    cfg.MODEL.ROI_KEYPOINT_HEAD.VAE.HIDDEN_DIMS = [32, 64, 128, 256, 256]
    cfg.MODEL.ROI_KEYPOINT_HEAD.VAE.LATENT_DIM = 100
    cfg.MODEL.ROI_KEYPOINT_HEAD.VAE.NUM_OUTPUTS_VAE = (
        10  # 2 center points, 8 keypoint offsets
    )
    cfg.DATASETS.TRAIN = ("KGN_VAE_train_dataset",)
    cfg.DATASETS.TEST = ("KGN_VAE_val_dataset",)
    # else:
    #     cfg.DATASETS.TRAIN = ("KGN_train_dataset",)
    #     cfg.DATASETS.TEST = ("KGN_test_dataset",)
    cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = False
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = None
    cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT_TUPLE = (HM_WT, WD_WT)
    cfg.DATASETS.NUM_BINS = NUM_BINS
    cfg.MODEL.ROI_HEADS.AVG_NUM_GRASPS = 10
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_OUTPUTS = (
        NUM_BINS + NUM_BINS + NUM_BINS + NUM_BINS * 2 * 4 + 2
    )  # hm + width + scale + keypoint offset + center reg
    cfg.TEST.EVAL_PERIOD = 5000000
    cfg.TEST.EVAL_SAVE_RESULTS = True
    if cfg.TEST.EVAL_SAVE_RESULTS:
        cfg.TEST.EVAL_OUTPUT_DIR = get_eval_output_dir()
        eval_output_dir = cfg.TEST.EVAL_OUTPUT_DIR
        if os.path.exists(eval_output_dir):
            shutil.rmtree(eval_output_dir)
        os.makedirs(eval_output_dir)

    cfg.freeze()
    return cfg
