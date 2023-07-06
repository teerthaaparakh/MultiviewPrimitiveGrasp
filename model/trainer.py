import sys, os, os.path as osp

sys.path.append(os.getenv("KGN_DIR"))
from utils.path_util import get_config_file_path
from dataloader.dataloader_func import mapper
from dataloader import dataset_func
from model.configs.config import add_centernet_config
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog
from detectron2.data import build_detection_train_loader, build_detection_test_loader

MetadataCatalog.get("KGN_dataset").keypoint_names = ["uf", "ub", "lf", "lb"]
MetadataCatalog.get("KGN_dataset").num_keypoints = 4

# head 64x64 reduced image size: for every pixel: 9 orientations heatmap, 9 widths, 9 scales,
# 2 * dataset.num_grasp_kpts * opt.ori_num center offset, 2 center point regression

from IPython import embed


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)


def setup(device="cpu"):
    """
    Create configs and perform basic setups.
    """
    config_fname = get_config_file_path()
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(config_fname)

    cfg.DATASETS.TRAIN = ("KGN_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.DEVICE = device
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.MODEL.PIXEL_MEAN = (0.5, 0.5, 0.5, 0.1)
    cfg.MODEL.PIXEL_STD = (0.01, 0.01, 0.01, 0.01)

    # cfg.merge_from_list(args.opts)
    # if '/auto' in cfg.OUTPUT_DIR:
    #     file_name = os.path.basename(args.config_file)[:-5]
    #     cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
    # logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    # default_setup(cfg, args)
    return cfg


def main():
    cfg = setup()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # import pdb
    # pdb.set_trace()
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    import pdb

    pdb.set_trace()

    trainer.train()

    # ##### following are example checks for trainer
    # n = iter(trainer.data_loader)
    # d = next(n)
    # len(d)                // output: 2
    # d[0].keys()           //output: dict_keys(['image', 'height', 'width', 'instances'])
    # d[0]["image"].shape   //output: torch.Size(4, 480, 640)
    # d[0]["instances"]     //output: Instances(num_instances=1, image_height=480, image_width=640, fields=[gt_classes: tensor([0]), gt_keypoints: Keypoints(num_instances=1)])

    # embed()

    # trainer.train()


if __name__ == "__main__":
    main()
