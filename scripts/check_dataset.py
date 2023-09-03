import sys, os

sys.path.append(os.environ["KGN_DIR"])

from utils.path_util import get_data_dir
from dataset.dataset_function import load_dataset
from visualize.vis_server import VizScene
from dataset.dataset_mapper_v3 import mapper_v2

import torch
import numpy as np
from utils.util import set_seed

set_seed()

if __name__ == "__main__":
    # vis = VizScene(get_data_dir())
    # vis.overlay_grasp(1)

    dataset = load_dataset(get_data_dir(), num_samples=2)
    for data in dataset:
        mapper_v2(data, draw=True, is_test=False)
