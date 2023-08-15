import numpy as np
from detectron2.modeling import build_model
import torch
import sys, os, glob, os.path as osp
sys.path.append(os.getenv("KGN_DIR"))
from utils.path_util import get_dataset_dir, get_test_dataset_dir
import cv2
from model.trainer import setup
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from detectron2.checkpoint import DetectionCheckpointer


class DefaultPredictor:

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.model.eval()
        

    def __call__(self, image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
          
            # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            _, height, width = image.shape
            
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
                
            return predictions
        
#check if the proper model is loaded
#check what predictions are returned
#results save
        
def main(real_dataset, visualize):
    cfg = setup()
    predict = DefaultPredictor(cfg)
    if real_dataset:
        root = get_test_dataset_dir()
        folder_name = "cup_over_bowl"
        rgb_paths = osp.join(root, folder_name, "images", "*.png")
        depth_paths = osp.join(root, folder_name, "depths", "*.png")
    
        
        
    else:
        root = get_dataset_dir()
        folder_name = "0"
        rgb_paths = osp.join(root, folder_name, "color_images", "*.png")
        depth_paths = osp.join(root, folder_name, "depth_img", "*.png")
        
    
    rgb_paths = sorted(glob.glob(rgb_paths))
    depth_paths = sorted(glob.glob(depth_paths))
    
    
    
    for i in range(len(rgb_paths)):
        
        rgb = cv2.imread(rgb_paths[i], cv2.IMREAD_UNCHANGED) / 255.0
        
        
        
        depth = 0.001 * cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)
        
        
        if not real_dataset:
            depth = depth[:,:,0]
        if len(depth.shape) == 2:
            depth = depth[..., None]
        if len(rgb.shape) == 2:
            rgb = rgb[..., None]

        image = np.concatenate((rgb, depth), axis=2)
        predictions = predict(image)
        
        center = predictions.center_pred
        bboxes = predictions.proposal_boxes.tensor
        if visualize:
            rgb = (rgb*255).astype(np.uint8)      
            
            plt.scatter(np.array(center[:,0]), np.array(center[:,1]), marker='.', color="black") 
            # print(offset_i_reshaped[idx][:,0], offset_i_reshaped[idx][:,1])
            # plt.scatter(offset_i_reshaped[idx][:,0], offset_i_reshaped[idx][:,1], marker='*', color="green")
            # plt.imsave(f"script_testing/centerpoints/{i}.png", img)
            plt.imshow(rgb)
            # for j in range(len(bboxes)):
            #     bbox = bboxes[j]
            #     wi = bbox[2] - bbox[0]
            #     he = bbox[3] - bbox[1]
            #     plt.gca().add_patch(Rectangle((bbox[0],bbox[1]),wi,he,
            #         edgecolor='red',
            #         facecolor='none',
            #         lw=2))
            plt.show()
        
        
if __name__=="__main__":
    real_dataset = True
    visualize = True
    main(real_dataset, visualize)
    
        
        

