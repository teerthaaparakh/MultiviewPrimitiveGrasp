from typing import T
import os, glob
import os.path as osp
import cv2

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import pickle


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


class SAM:
    def __init__(self, sam_checkpoint, device=None):
        model_type = "vit_h"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print("SAM running on device", device)

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)

    # def run_sam(self, data_dir, scene_id, img_id):
    #     print(f"Scene: {scene_id}, Image: {img_id}")
    #     scene_path = osp.join(data_dir, f"{scene_id}")
    #     color_fname = osp.join(scene_path, f"color_images/color_image_{img_id}.png")
    #     rgb = cv2.cvtColor(cv2.imread(color_fname), cv2.COLOR_BGR2RGB)
    #     self.sam_predictor.set_image(rgb, image_format="RGB")

    #     sam_features = self.sam_predictor.get_image_embedding()
    #     print(sam_features.shape)
    #     os.makedirs(osp.join(scene_path, "sam_features"), exist_ok=True)
    #     with open(osp.join(scene_path, f"sam_features/{img_id}.pkl"), "wb") as f:
    #         pickle.dump(sam_features, f)

    def run(self, image):
        self.sam_predictor.set_image(image, image_format="RGB")

        sam_features = self.sam_predictor.get_image_embedding()
        return sam_features
