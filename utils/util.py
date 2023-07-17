import numpy as np
from utils.other_configs import *

def get_area(bbox):
    min_col, min_row, max_col, max_row = bbox
    height = max_row - min_row
    width = max_col - min_col
    return height*width
    
def get_ori_clss(kpts_2d, ori_range = [-np.pi/2, np.pi/2]):
    # kpts_2d: (num_grasps, 4, 2)
    kpt_2 = kpts_2d[:,1,:]
    kpt_3 = kpts_2d[:,2,:]
    
    kpt_2x = kpt_2[:,0]
    kpt_2y = kpt_2[:,1]
    
    kpt_3x = kpt_3[:,0]
    kpt_3y = kpt_3[:,1]
    
    delta_x = kpt_3x - kpt_2x
    delta_y = kpt_3y - kpt_2y
    
    angle = np.arctan(delta_y/delta_x) + np.pi/2
    
    bin_index = np.floor(angle/((ori_range[1]-ori_range[0])/NUM_BINS)).astype(int)
    return bin_index # (num_grasps,)
    

    
    
    
    