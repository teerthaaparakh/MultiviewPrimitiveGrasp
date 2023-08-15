import wandb
import sys, os, os.path as osp
import shutil
import pickle
sys.path.append(os.getenv("KGN_DIR"))
from utils.path_util import get_config_file_path, get_output_dir, get_eval_output_dir, get_pretrained_resnet_path
from dataloader.dataloader_func import mapper
from dataloader.dataloader_func_vae import mapper_vae
from dataloader.dataset_func import dataset_function
from dataloader.dataset_func_vae import dataset_function_vae
from our_modeling import roi_head
from our_modeling import keypoint_head
from our_modeling import our_meta_arch
from model.configs.config import add_centernet_config
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage
import logging
from detectron2.data import DatasetCatalog
from detectron2.config import CfgNode as CN
from detectron2.evaluation import verify_results
from utils.path_util import get_pickled_dataset
from IPython import embed

setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog
from detectron2.data import build_detection_train_loader, build_detection_test_loader

from detectron2.structures import Instances

from model.our_modeling.instances import __newgetitem__

from utils.other_configs import *

# head 64x64 reduced image size: for every pixel: 9 orientations heatmap, 9 widths, 9 scales,
# 2 * dataset.num_grasp_kpts * opt.ori_num center offset, 2 center point regression

from IPython import embed
from detectron2.engine import default_argument_parser


class MyTrainer(DefaultTrainer):
    def train(self):
        self.iter = self.start_iter

        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    self.before_step()

                    self.run_step()
                    
                    self.after_step()

                    # if (
                    # self.cfg.TEST.EVAL_PERIOD > 0
                    # and self.iter % self.cfg.TEST.EVAL_PERIOD == 0
                    # and self.iter != self.max_iter
                    # ):
                    # test_results = self.test(self.cfg, self._trainer.model)
                    # comm.synchronize()

                    # print(test_results)

                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.MODEL.ROI_KEYPOINT_HEAD.USE_VAE:
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper_vae)
        else:
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.ROI_KEYPOINT_HEAD.USE_VAE:
            return build_detection_train_loader(cfg, mapper=mapper_vae)
        else:
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        
        pass


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
    cfg.SOLVER.MAX_ITER = 1 #40000
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
    if cfg.MODEL.ROI_KEYPOINT_HEAD.USE_VAE:
        cfg.MODEL.ROI_KEYPOINT_HEAD.VAE = CN()
        cfg.MODEL.ROI_KEYPOINT_HEAD.VAE.HIDDEN_DIMS = [32, 64, 128, 256, 256]
        cfg.MODEL.ROI_KEYPOINT_HEAD.VAE.LATENT_DIM = 100
        cfg.MODEL.ROI_KEYPOINT_HEAD.VAE.NUM_OUTPUTS_VAE = 10 # 2 center points, 8 keypoint offsets
        cfg.DATASETS.TRAIN = ("KGN_VAE_train_dataset",)
        cfg.DATASETS.TEST = ("KGN_VAE_test_dataset",)
    else:
        cfg.DATASETS.TRAIN = ("KGN_train_dataset",)
        cfg.DATASETS.TEST = ("KGN_test_dataset",)
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


def main_train(cfg, args):
    
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # TODO (TP): ROI pooling turn off and train, ROI pooling on with increased size of bounding box
    # TODO (TP): think of another way to map keypoint to heatmap in keypoint_rcnn_loss
    trainer.train()


def predict():
    pass


if __name__ == "__main__":
    args = default_argument_parser()
    Instances.__getitem__ = __newgetitem__
    

    use_pickled = True

    if use_pickled:
        
        def load_dataset(name):
            path = osp.join(get_pickled_dataset(), f"{name}.pkl")
            assert osp.exists(path), f"no pickled file at {path}.pkl found"
            with open(path, 'rb') as f:
                result = pickle.load(f)
            logging.info(f"Dataset {name} loaded. Length of dataset: {len(result)}")
            return result
        
        DatasetCatalog.register("KGN_train_dataset", lambda: load_dataset("sim_train"))
        DatasetCatalog.register("KGN_test_dataset", lambda: load_dataset("sim_test"))        
        DatasetCatalog.register("KGN_VAE_train_dataset", lambda: load_dataset("sim_train_vae"))
        DatasetCatalog.register("KGN_VAE_test_dataset", lambda: load_dataset("sim_test_vae"))

    else:
        DatasetCatalog.register("KGN_train_dataset", lambda: dataset_function(NUM_TRAINING_DATA))
        DatasetCatalog.register("KGN_test_dataset", lambda: dataset_function(NUM_TEST_DATA))        
        DatasetCatalog.register("KGN_VAE_train_dataset", lambda: dataset_function_vae(NUM_TRAINING_DATA))
        DatasetCatalog.register("KGN_VAE_test_dataset", lambda: dataset_function_vae(NUM_TEST_DATA))
    
    args = args.parse_args()
    args.num_gpus = 1
    if args.config_file:
        cfg = setup(args.config_file)
    else:
        cfg = setup()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # wandb.init(sync_tensorboard=True)
    wandb.tensorboard.patch(root_logdir=cfg.OUTPUT_DIR)
    wandb.init(name="KGN", project="Original KGN",
                settings=wandb.Settings(start_method="thread", console="off"), sync_tensorboard=True, mode="offline")

    main_train(cfg, args)

    wandb.finish()
