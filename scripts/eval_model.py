import numpy as np
import sys, os
sys.path.append(os.getenv("KGN_DIR"))
from utils.path_util import get_trained_model_path, get_eval_output_dir
from detectron2.engine import default_argument_parser
from model.trainer_utils import setup
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch
import cv2
from matplotlib import pyplot as plt
import pickle


    
class DefaultPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        # if len(cfg.DATASETS.TEST):
        #     self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        # self.aug = T.ResizeShortestEdge(
        #     [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        # )

        self.input_format = "RGB"
        # assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, depth, visualize=False, save_output=False, **kwargs):
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
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
                
            height, width = original_image.shape[:2]
            # image = self.aug.get_transform(original_image).apply_image(original_image)
            image = original_image.transpose(2, 0, 1)
        
            rgbd = np.concatenate((image / 255.0, depth[None, ...]), axis=0)

            inputs = {"image": torch.from_numpy(rgbd).float(), "rgb": image, "depth":depth,
                            "height": height, "width": width}
            predictions = self.model([inputs])[0]
            
            if visualize:
                save_dir = os.path.join(get_eval_output_dir(), "images_output")
                os.makedirs(save_dir, exist_ok=True)
                file_name = kwargs["output_image_file_name"]
                fig, ax = plt.subplots()
                im = ax.imshow(original_image)

                cp = predictions.center_pred
                ax.plot(cp[:,0], cp[:,1], ls='dotted', linewidth=2, color='red')
                fig.savefig(os.path.join(save_dir, file_name))
                plt.close(fig)
                
                
            if save_output:
                save_dir = os.path.join(get_eval_output_dir(), "results")
                os.makedirs(save_dir, exist_ok=True)
                file_name = kwargs["output_dict_file_name"]
                file_path = os.path.join(save_dir, file_name)
                
                with open(file_path, 'wb') as f:
                    pickle.dump({"results":predictions}, f)
            
            return predictions
        

            
if __name__=="__main__":
    
    model_path = get_trained_model_path()
    
    args = default_argument_parser()
    args = args.parse_args()
    
    if torch.cuda.is_available():
        device = "cuda"
        args.num_gpus = torch.cuda.device_count()
        print(f"cuda found !!!!!!!!!, {args.num_gpus}")
    else:
        device = "cpu"
        args.num_gpus = 0
        
    if args.config_file:
        cfg = setup(device, args.config_file)
    else:
        cfg = setup(device)
        
    cfg.defrost()
    cfg.MODEL.WEIGHTS = model_path
    cfg.freeze()
    
    predictor = DefaultPredictor(cfg)
    
    rgb = "/Users/teerthaaparakh/Desktop/MultiviewPrimitiveGrasp/data/ps_grasp_multi_1k/0/color_images/color_image_0.png"
    depth = '/Users/teerthaaparakh/Desktop/MultiviewPrimitiveGrasp/data/ps_grasp_multi_1k/0/depth_raw/depth_raw_0.npy'
    
    rgb = cv2.imread(rgb)
    depth = np.load(depth) 
    
    output_image_file_name = "0.png"
    output_dict_file_name = "0.pkl"
    
    results = predictor(rgb, depth, visualize=True,save_output=True, output_image_file_name = output_image_file_name,
                        output_dict_file_name=output_dict_file_name)

    print(results)        
        
