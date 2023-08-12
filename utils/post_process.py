import numpy as np
import torch
from detectron2.structures import Instances
import pickle

def post_process(output, instances: Instances):
    kpts_offset = output[0]
    kpts_center = output[1]
    split_idx = [len(inst) for inst in instances]
    kpts_offset = kpts_offset.split(split_idx)
    kpts_center = kpts_center.split(split_idx)
    
    
    
    for i in range(len(split_idx)):
        img_height, img_width = instances[i].image_size
        
        prop_boxes = instances[i].proposal_boxes.tensor
        offset = kpts_offset[i].reshape(-1, 4, 2)
        center = kpts_center[i]
        
        widths = prop_boxes[:, 2] - prop_boxes[:, 0]
        heights = prop_boxes[:, 3] - prop_boxes[:, 1]
        
        center[:, 0] = center[:, 0]*widths
        center[:, 1] = center[:, 1]*heights
        
        center[:, 0] += prop_boxes[:, 0]
        center[:, 1] += prop_boxes[:, 1]
        
        offset[:, :, 0] *= img_width
        offset[:, :, 1] *= img_height
        
        offset[:, :, 0] += center[:, 0][:, None]
        offset[:, :, 1] += center[:, 1][:, None]
        
        instances[i].grasp_kpts_pred = offset
        instances[i].center_pred = center
    
        
        
        
        
        