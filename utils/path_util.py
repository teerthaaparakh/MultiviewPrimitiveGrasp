import os, os.path as osp


def get_src_dir():
    return os.environ["KGN_DIR"]


def get_dataset_dir():
    return osp.join(get_src_dir(), "dataset/ps_grasp_multi_1k")


def get_config_file_path():
    return osp.join(get_src_dir(), "model/configs/CenterNet2_R50_1x.yaml")


def get_debug_img_dir():
    return osp.join(get_src_dir(), "temp_images")

def get_debug_img_dir_orig():
    return osp.join(get_src_dir(), "temp_images_orig")


def get_output_dir():
    return osp.join(get_src_dir(), "output/KGN")

def get_eval_output_dir():
    return osp.join(get_src_dir(), "eval_output")

def get_pretrained_resnet_path():
    return osp.join(get_src_dir(), "model/pretrained_resnet/R-50.pkl")


if __name__ == "__main__":
    print(get_dataset_dir())
