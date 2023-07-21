import os, os.path as osp


def get_src_dir():
    return os.environ["KGN_DIR"]


def get_dataset_dir():
    return osp.join(get_src_dir(), "dataset/ps_grasp_multi_1k")


def get_config_file_path():
    return osp.join(get_src_dir(), "model/configs/CenterNet2_R50_1x.yaml")

def get_debug_img_dir():
    return osp.join(get_src_dir(), "temp_images")

def get_output_dir():
    return osp.join(get_src_dir(), "output/KGN")



if __name__ == "__main__":
    print(get_dataset_dir())
