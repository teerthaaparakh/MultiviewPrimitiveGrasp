import os, os.path as osp


def get_src_dir():
    return os.environ["KGN_DIR"]


def get_dataset_dir():
    return os.path.join(get_src_dir(), "dataset/ps_grasp_multi_1k")


def get_config_file_path():
    return os.path.join(get_src_dir(), "model/configs/CenterNet2_R50_1x.yaml")


if __name__ == "__main__":
    print(get_dataset_dir())
