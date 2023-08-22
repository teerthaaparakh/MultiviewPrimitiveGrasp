import os, sys

sys.path.append(os.environ["KGN_DIR"])
from dataset.generate_segments_info import SAM
from utils.path_util import get_data_dir

sam_checkpoint = "/Users/meenalp/Desktop/MEng/system_repos/new_system/sam_model/sam_vit_h_4b8939.pth"

sam = SAM(sam_checkpoint)
sam.run_sam(get_data_dir(), '1', '1')