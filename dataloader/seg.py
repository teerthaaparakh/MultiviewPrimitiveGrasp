# imports
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from copy import copy


MARGIN = 5

def get_bb(mask):
    row_projection = np.sum(mask, axis=1)
    col_projection = np.sum(mask, axis=0)
    row_idx = np.nonzero(row_projection)[0]
    col_idx = np.nonzero(col_projection)[0]
    
    height, width = mask.shape
    if len(row_idx) == 0 or len(col_idx) == 0:
        return None
    min_row = max(np.min(row_idx) - MARGIN, 0)
    max_row = min(np.max(row_idx) + MARGIN, height)
    min_col = max(np.min(col_idx) - MARGIN, 0)
    max_col = min(np.max(col_idx) + MARGIN, width)  #check width and height
    
    return (min_row, min_col, max_row, max_col)

def draw_bb(ll, image):
    colors = [(255, 0, 0), (0, 255, 240), (0, 255, 0), (0, 0, 255), (240, 240, 0)]
    yellow = (255, 255, 240)

    image = copy(image)[:, :, ::-1].astype(np.uint8)
    for tu in ll:
        min_row, min_col, max_row, max_col = tu
        image = cv2.rectangle(
            image, (min_col, min_row), (max_col, max_row), color=yellow, thickness=2)
        
    return image
        
    

def generate_bbox(seg_img_path):    
    seg_img = cv2.imread(seg_img_path, cv2.IMREAD_UNCHANGED)
    indices = np.unique(seg_img)[1:]
    ll = {}
    for idx in indices:
        kernel = np.ones((3, 3), np.uint8)
        new_seg = seg_img == idx
        new_seg = cv2.erode(new_seg.astype(np.uint8), kernel)
        bb = get_bb(new_seg)
        if bb:
            ll[idx] = bb
    
    return ll
  
    
        
    
        
    
        
if __name__=="__main__":
    generate_bbox()
