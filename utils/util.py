import numpy as np

def get_area(bbox):
    min_col, min_row, max_col, max_row = bbox
    height = max_row - min_row
    width = max_col - min_col
    return height*width
    