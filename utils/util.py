import numpy as np
import sys, os

sys.path.append(os.environ["KGN_DIR"])
from utils.other_configs import *
import random
import math, torch
from torch import functional as F
import pickle

def get_area(bbox):
    min_col, min_row, max_col, max_row = bbox
    height = max_row - min_row
    width = max_col - min_col
    return height * width


def get_orientation_class(kpts_2d, ori_range=[0, np.pi]):
    # kpts_2d: (num_grasps, 4, 2)
    kpt_2 = kpts_2d[:, 1, :]
    kpt_3 = kpts_2d[:, 2, :]

    kpt_2x = kpt_2[:, 0]
    kpt_2y = kpt_2[:, 1]

    kpt_3x = kpt_3[:, 0]
    kpt_3y = kpt_3[:, 1]

    delta_x = kpt_3x - kpt_2x
    delta_y = kpt_3y - kpt_2y

    angle = np.arctan2(delta_y, delta_x) 
    angle[angle < 0] += np.pi
    
    bin_size = (ori_range[1] - ori_range[0]) / NUM_BINS
    bin_index = np.floor(angle / bin_size).astype(int)
    return bin_index  # (num_grasps,)


    
def get_grasp_features(instances):
    ret = []
    for inst in range(len(instances)):
        # get the centers points location relative to bounding box
        instances[inst].gt_centerpoints.tensor[:,0][:,0] = instances[inst].gt_centerpoints.tensor[:,0][:,0] \
                                                            - instances[inst].gt_boxes.tensor[:,0]
        instances[inst].gt_centerpoints.tensor[:,0][:,1] = instances[inst].gt_centerpoints.tensor[:,0][:,1] \
                                                            - instances[inst].gt_boxes.tensor[:,1]
                                                    
        # # get the grasp keypoints location relative to bounding box
        # instances[inst].gt_keypoints.tensor[:, :, 0] = instances[inst].gt_keypoints.tensor[:, :, 0] \
        #                                             - instances[inst].gt_boxes.tensor[:,0][:, None]
                                                    
        # instances[inst].gt_keypoints.tensor[:, :, 1] = instances[inst].gt_keypoints.tensor[:, :, 1] \
        #                                             - instances[inst].gt_boxes.tensor[:,1][:, None]  
                                                    
        # scale centers and keypoints with heights and widths
        widths = instances[inst].gt_boxes.tensor[:,2] - instances[inst].gt_boxes.tensor[:,0]
        heights = instances[inst].gt_boxes.tensor[:,3] - instances[inst].gt_boxes.tensor[:,1]
        
        instances[inst].gt_centerpoints.tensor[:,0][:,0] = instances[inst].gt_centerpoints.tensor[:,0][:,0]/widths
        instances[inst].gt_centerpoints.tensor[:,0][:,1] = instances[inst].gt_centerpoints.tensor[:,0][:,1]/heights
        
        # instances[inst].gt_keypoints.tensor[:, :, 0] = instances[inst].gt_keypoints.tensor[:, :, 0]/widths[:,None]
        # instances[inst].gt_keypoints.tensor[:, :, 1] = instances[inst].gt_keypoints.tensor[:, :, 1]/heights[:,None]
        
        concat_features = torch.cat((instances[inst].gt_keypoints.tensor[:,:,0:2].flatten(start_dim = 1), 
                                            instances[inst].gt_centerpoints.tensor[:,0][:, 0:2]), axis = 1)
        ret.append(concat_features)
        
    return torch.cat(ret, axis=0)
        
        
def custom_random_generator(array, max_items):
    array = list(array)
    if len(array) >= max_items:
        return random.sample(array, max_items) 
    else:
        ret = [0]*max_items
        repetition = math.floor(max_items/len(array))
        total = repetition*len(array)
        remaining = max_items - total
        ret[0: total] = array*repetition
        ret[total: ] = random.sample(array, remaining)
        return ret


def kpts_to_hm(
    instances,
    heatmap_shape
):  # (M,C,56,56)
    heatmaps = []
    _, H, W = heatmap_shape
    C = NUM_BINS
    M = len(instances)
    # heatmap shape 27x9x14x14
    for i in range(M):
        cen = instances[i].gt_centerpoints.tensor
        ori = instances[i].gt_orientations
        NC = len(instances[i])
        heatmap_ret = np.zeros((NC, C, H, W))
        boxes = instances[0].proposal_boxes.tensor
        for j in range(NC):
            single_center = cen[j][0]
            single_ori = ori[j]
            box = boxes[j]
            map_x, map_y, in_box = mapper(single_center, box, H)
            if in_box:
                heatmap_ret[j, single_ori, map_x, map_y] = 1
        heatmaps.append(heatmap_ret)
    heatmaps = np.array(heatmaps)
    return torch.from_numpy(heatmaps)


def mapper(center, box, hm_size):
    # center: 3 (x,y,v)
    # box: 4 (x,y,x,y)
    top_x, top_y, bottom_x, bottom_y = box
    scale_x = hm_size / (bottom_x - top_x)
    scale_y = hm_size / (bottom_y - top_y)
    x = (center[0] - top_x) * scale_x
    x = x.floor().long()
    y = (center[1] - top_y) * scale_y
    y = y.floor().long()

    inside_xrange = x >= 0 and x <= (hm_size - 1)
    inside_yrange = y >= 0 and y <= (hm_size - 1)

    if inside_xrange and inside_yrange:
        return x, y, True
    else:
        return x, y, False


@torch.jit.script_if_tracing
def heatmaps_to_keypoints(
    maps: torch.Tensor, instances
) -> dict:
    
    num_instances = [len(inst) for inst in instances]
    maps = maps.split(num_instances)
    roi_heatmaps = []
    M = len(num_instances)
    for i in range(M):
        roi_hmaps = []
        heatmap = maps[i]
        rois = instances[i].proposal_boxes.tensor
        offset_x = rois[:, 0]
        offset_y = rois[:, 1]
        
        widths = (rois[:, 2] - rois[:, 0]).clamp(min=1)
        heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)
    
        widths_ceil = widths.ceil()
        heights_ceil = heights.ceil()
        
        width_corrections = widths / widths_ceil
        height_corrections = heights / heights_ceil
        
        for j in range(num_instances[i]):
            outsize = (int(heights_ceil[j]), int(widths_ceil[j]))
            roi_hmap = F.interpolate(
                heatmap[[j]], size=outsize, mode="bicubic", align_corners=False
            )
            roi_hmaps.append(roi_hmap)
            
        roi_heatmaps.append(roi_hmaps)
        
    return roi_heatmaps

def save_results(data, iter_no):
    print("HERE, HERE, HERE", data)
    with open(f'/Users/teerthaaparakh/Desktop/MultiviewPrimitiveGrasp/script_testing/inference_{iter_no}.pkl', 'wb') as file:
                    pickle.dump(data, file)
    
if __name__=="__main__":
    arr = range(10)
    max_items = 13
    print(custom_random_generator(arr, max_items))
    
    
    
