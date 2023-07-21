import torch, detectron2

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.evaluation import inference_on_dataset
import detectron2.data.detection_utils as utils
from detectron2.utils.logger import create_small_table
from detectron2.config import configurable
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

import pycocotools.mask as mask_util
from PIL import Image
import copy
import logging

from typing import List, Optional, Union

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
import torch
import time
import datetime
import argparse

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import default_argument_parser, default_setup, launch
# from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator
#             from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import Transform, NoOpTransform
import albumentations as A
import logging

import wandb

from detectron2.evaluation import COCOEvaluator
import itertools
from tabulate import tabulate


def build_seg_aug(cfg):
    # augs = []
    additional_augs = None
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
            )
        )

    others = {
        'random_rotation': True,
        'random_resize': cfg.resize_aug
    }

    if cfg.augmentate:
        # augs.append(AlbumentationsWrapper(A.ToFloat(max_value=255.0, always_apply=True, p=1.0)))
        # return augs, None

        albumention_lst = [
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.8),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), p=0.8),
            A.FancyPCA (alpha=0.1, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            A.PixelDropout(dropout_prob=0.01, per_channel=False, drop_value=0, mask_drop_value=None, p=0.8),
            A.OneOf([A.Downscale(scale_min=0.25, scale_max=0.5, interpolation=2, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=1.0, p=1.0),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                    A.MotionBlur(blur_limit=9, allow_shifted=True, always_apply=False, p=1.0),
                    A.GaussNoise(var_limit=(10.0, 80.0), mean=0, per_channel=True, p=1.0)], 
                p=0.8
            ),
            A.OneOf([A.RandomBrightness(limit=0.2, always_apply=False, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=1.0),
                    A.RandomContrast(limit=0.2, p=1.0)],
                p=1.0
            )
        ]

        t1 = A.SomeOf(albumention_lst, 1, replace=False, p=1.0)
        t2 = A.SomeOf(albumention_lst, 1, replace=False, p=1.0)
        t3 = A.SomeOf(albumention_lst, 1, replace=False, p=1.0)
        t4 = A.SomeOf(albumention_lst, 1, replace=False, p=1.0)
        t5 = A.SomeOf(albumention_lst, 1, replace=False, p=1.0)

        additional_augs = A.Compose([A.OneOf([t1, t2, t3, t4, t5], p=0.5)])
        # augs.append(AlbumentationsWrapper(transform))
        augs.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))


        if others['random_rotation']:
            augs.append(T.RandomApply(T.RandomRotation([-90.0, 90.0], 
                                expand=True, center=[[0.3, 0.3], [0.7, 0.7]], 
                                sample_style="range"), prob=0.2))
        else:
            print('Random rotation disabled')

    return augs, additional_augs

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

        k = min(len(self.data_pts), 10)
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
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
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
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        wandb.log({'validation_loss': mean_loss})
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
        is_final = (next_iter == self.trainer.max_iter)
        if is_final or ((self._period > 0) and ((next_iter % self._period) == 0)):
            print("Next iter-----------------------------", next_iter)
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """taken from detectron, made one change"""
    boxes = (np.stack([BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos])
                    if len(annos)
                    else np.zeros((0, 4))
             )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            try:
                masks = PolygonMasks(segms)
            except ValueError as e:
                raise ValueError(
                    "Failed to use mask_format=='polygon' from the given annotations!"
                ) from e
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(np.array(x, copy=True))) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target


class MyDatasetMapper:
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
	additional_augs = None,
        resize_aug = False,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.additional_augs        = additional_augs
        self.resize_aug             = resize_aug

        if self.resize_aug:
            self.close_augmentations = copy.copy(augmentations)

            size_lst = [(720, 1280)]
            x, y = 720, 1280
            for factor in np.linspace(0.2, 0.8, 10):
                size_lst.append((int(x*factor), int(y*factor)))

            print("Resizing augmentation enabled, allowed sizes", size_lst)
            self.close_augmentations.append(T.RandomApply(T.RandomResize(size_lst), prob=0.7))
            self.close_augmentations = T.AugmentationList(self.close_augmentations)
        else:
            print("Resizing disabled.")

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        if self.additional_augs is not None:
            transformed = self.additional_augs(image=image)
            image = transformed['image']

        image_name = os.path.normpath(dataset_dict["file_name"]).split(os.sep)[-1]
        # print("Image name inside dataset mapper", image_name)
        if self.resize_aug and (not ('far' in image_name)):
            augs_to_apply = self.close_augmentations
        else:
            augs_to_apply = self.augmentations

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = augs_to_apply(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return MyCOCOEvaluator(dataset_name, tasks=['bbox', 'segm'], 
                            use_fast_impl=False, output_dir=cfg.OUTPUT_DIR)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                MyDatasetMapper(self.cfg,True)
            )
        ))
        # swap the order of PeriodicWriter and ValidationLoss
        # code hangs with no GPUs > 1 if this line is removed
        hooks = hooks[:-2] + hooks[-2:][::-1]
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        # if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
        augs, additional_augs = build_seg_aug(cfg)
        mapper = MyDatasetMapper(cfg, is_train=True, augmentations=augs,
					additional_augs=additional_augs,
                                        resize_aug=cfg.resize_aug)
        # else:
        #     mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)


    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)


class MyCOCOEvaluator(COCOEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._all_results_precisions = None

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        precisions = coco_eval.eval["precision"]
        print(precisions[:, :, 0, 0, -1])
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]
        self._all_results_precisions = precisions

        if class_names is None or len(class_names) <= 1:
            return results

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)
        results.update({"AP-" + name: ap for name, ap in results_per_category})

        return results
