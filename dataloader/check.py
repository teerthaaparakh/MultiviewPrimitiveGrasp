import sys, os
sys.path.append(os.getenv("KGN_DIR"))
from dataloader.dataloader_func import mapper
from dataloader.dataset_func import dataset_function


list_dict = dataset_function()

for indi_dict in list_dict:
    ret = mapper(indi_dict)
    print(ret["instances"].gt_keypoints.shape)
    print("\n")

