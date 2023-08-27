import os, sys
from glob import glob

sys.path.append(os.environ["KGN_DIR"])
from dataset.generate_segments_info import SAM
from utils.path_util import get_data_dir

sam_checkpoint = (
    "/Users/meenalp/Desktop/MEng/system_repos/new_system/sam_model/sam_vit_h_4b8939.pth"
)


def get_scene_and_image_id(color_image_path):
    path_elements = color_image_path.split(os.sep)
    assert path_elements[-1].startswith(
        "color_image_"
    ), "the image file does not start with color_image_"
    scene_id = path_elements[-3]
    img_id = int(path_elements[-1].replace("color_image_", "")[:-4])
    return scene_id, img_id


if __name__ == "__main__":
    sam = SAM(sam_checkpoint)

    data_dir = get_data_dir()
    color_files_lst = sorted(
        glob(os.path.join(data_dir, "*/color_images/*.png"), recursive=True)
    )
    for idx, color_image_path in enumerate(color_files_lst):
        # print(f"Processing datapoint: {idx}")
        scene_id, img_id = get_scene_and_image_id(color_image_path)
        sam.run_sam(data_dir, scene_id, img_id)
