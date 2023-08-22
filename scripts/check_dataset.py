import sys, os
sys.path.append(os.environ["KGN_DIR"])

from utils.path_util import get_data_dir
from dataset.dataset_function import load_dataset
from visualize.vis_server import VizScene


if __name__ == "__main__":

    # vis = VizScene(get_data_dir())
    # vis.overlay_grasp(1)
    
    dataset = load_dataset(get_data_dir(), draw=True)
