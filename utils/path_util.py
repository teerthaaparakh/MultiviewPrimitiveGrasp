import os, os.path as osp


def get_src_dir():
    return os.environ["KGN_DIR"]


def get_pickled_data_dir():
    return osp.join(get_src_dir(), "data", "pickled_datasets")


def get_pickled_normal_data_dir():
    return osp.join(get_pickled_data_dir(), "normal_dataset")


def get_pickled_aug_data_dir():
    return osp.join(get_pickled_data_dir(), "aug_dataset")


def get_data_dir():
    return osp.join(get_src_dir(), "data/ps_grasp_multi_1k")


def get_test_dataset_dir():
    return osp.join(get_src_dir(), "data/real_dataset")


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
    return osp.join(
        get_src_dir(), "model/pretrained_resnet/model_0000637.pth"
    )  # model/pretrained_resnet/R-50.pkl

def get_sam_model_path():
    return osp.join(get_src_dir(), "model/pretrained_sam/sam_vit_h_4b8939.pth")

# if __name__ == "__main__":
#     print(get_dataset_dir())
