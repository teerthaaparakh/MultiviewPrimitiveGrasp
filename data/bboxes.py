import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import copy


MARGIN = 5


def generate_bbox(seg_img_path):
    """
    for the particular image, returns a dictionary
    with keys being the object ids, and the values being the bbs for
    the corresponding mask
    """
    seg_img = cv2.imread(seg_img_path, cv2.IMREAD_UNCHANGED)
    return generate_bbox_from_seg(seg_img)

def generate_bbox_from_seg(seg_img):
    # assuming 0 is the background
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
    max_col = min(np.max(col_idx) + MARGIN, width)  # check width and height

    return (min_col, min_row, max_col, max_row)


def get_area(bbox):
    min_col, min_row, max_col, max_row = bbox
    height = max_row - min_row
    width = max_col - min_col
    return height * width

def draw_bb(ll, image):
    colors = [(255, 0, 0), (0, 255, 240), (0, 255, 0), (0, 0, 255), (240, 240, 0)]
    yellow = (255, 255, 240)

    image = copy(image)[:, :, ::-1].astype(np.uint8)
    vals = list(ll.values())

    for tu in vals:
        min_col, min_row, max_col, max_row = tu
        image = cv2.rectangle(
            image, (min_col, min_row), (max_col, max_row), color=yellow, thickness=2
        )

    return image

if __name__ == "__main__":
    scene_ind = 441
    seg_img_path = f"/Users/teerthaaparakh/Desktop/MultiviewPrimitiveGrasp/dataset/ps_grasp_multi_1k/{scene_ind}/seg_labels/segmask_label_0.jpg"

    img_path = f"/Users/teerthaaparakh/Desktop/MultiviewPrimitiveGrasp/dataset/ps_grasp_multi_1k/{scene_ind}/color_images/color_image_0.png"

    image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    ll = generate_bbox(seg_img_path)
    print(ll)
    image_ret = draw_bb(ll, image)

    from matplotlib import pyplot as plt

    plt.imshow(image_ret)
    plt.show()




