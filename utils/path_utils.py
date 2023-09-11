import os, os.path as osp


def get_src_dir():
    return os.environ["KGN_DIR"]

def get_pickled_data_dir():
    path = osp.join(get_src_dir(), "data", "pickled_datasets")
    return path

def get_data_dir():
    return osp.join(get_src_dir(), "data/ps_grasp_multi_1k")


