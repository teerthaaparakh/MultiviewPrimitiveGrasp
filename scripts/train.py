import sys, os, os.path as osp

sys.path.append(os.getenv("KGN_DIR"))
from detectron2.data import DatasetCatalog
from detectron2.structures import Instances
import wandb

# from detectron2.data import MetadataCatalog
# from IPython import embed
from detectron2.engine import default_argument_parser
from model.our_modeling.instances import __newgetitem__
from dataset.dataset_function import load_dataset_wrapper
from model.trainer_utils import setup, MyTrainer

DatasetCatalog.register("KGN_VAE_train_dataset", lambda: load_dataset_wrapper(t="train"))

DatasetCatalog.register("KGN_VAE_val_dataset", lambda: load_dataset_wrapper(t="val"))

if __name__ == "__main__":
    args = default_argument_parser()
    Instances.__getitem__ = __newgetitem__

    args = args.parse_args()
    args.num_gpus = 1
    if args.config_file:
        cfg = setup(args.config_file)
    else:
        cfg = setup()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # wandb.init(sync_tensorboard=True)
    wandb.tensorboard.patch(root_logdir=cfg.OUTPUT_DIR)
    wandb.init(
        name="KGN",
        project="Original KGN",
        settings=wandb.Settings(start_method="thread", console="off"),
        sync_tensorboard=True,
        mode="offline",
    )

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    import pdb; pdb.set_trace()

    # TODO (TP): ROI pooling turn off and train,
    #            ROI pooling on with increased size of bounding box
    # TODO (TP): think of another way to map keypoint to heatmap
    #            in keypoint_rcnn_loss
    trainer.train()

    wandb.finish()
