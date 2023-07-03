import os, os.path as osp

def get_src_dir():
    return os.environ["KGN_DIR"]

def get_dataset_dir():
    return os.path.join(get_src_dir(), "dataset/ps_grasp_multi_1k")



if __name__=="__main__":
    print(get_dataset_dir())