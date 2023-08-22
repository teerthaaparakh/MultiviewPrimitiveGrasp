import sys
import os, os.path as osp

sys.path.append(os.getenv("KGN_DIR"))
from dataloader.dataset_func import dataset_function
from dataloader.dataset_func_vae import dataset_function_vae
from utils.other_configs import NUM_TRAINING_DATA, NUM_TEST_DATA
import pickle
from utils.path_util import get_pickled_dataset

from utils import path_util
from pathlib import Path

def create_pickled_data():
    os.makedirs(get_pickled_dataset(), exist_ok=True)

    path = osp.join(get_pickled_dataset(), "sim_train.pkl")
    result_training = dataset_function(NUM_TRAINING_DATA)
    with open(path, 'wb') as f:
        pickle.dump(result_training, f)

    path = osp.join(get_pickled_dataset(), "sim_test.pkl")
    result_training = dataset_function(NUM_TEST_DATA)
    with open(path, 'wb') as f:
        pickle.dump(result_training, f)

    path = osp.join(get_pickled_dataset(), "sim_train_vae.pkl")
    result_training = dataset_function_vae(NUM_TRAINING_DATA)
    with open(path, 'wb') as f:
        pickle.dump(result_training, f)

    path = osp.join(get_pickled_dataset(), "sim_test_vae.pkl")
    result_training = dataset_function_vae(NUM_TEST_DATA)
    with open(path, 'wb') as f:
        pickle.dump(result_training, f)


# dict_keys(['intrinsic', 'camera_poses', 'grasp_poses', 'grasp_widths', 'grasp_collision', 'obj_types', 'obj_dims', 'obj_poses'])
# def fix_data() -> None:
#     data_dir = path_util.get_data_dir()
#     scene_name = Path(data_dir).glob("*/color_images/color_image_*.png")
#     pass

