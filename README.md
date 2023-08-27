TODO:
1. args.num_gpu: check whether GPU training is correctly setup or not
2. check the device in training setup
3. number of datapoints in training
4. check why flattening of keypoints is required in dataset (IMPORTANT)


Plan Forward:
1. Center points are working fine
2. Keypoint Offsets - ??
3. Bounding Box: can be fixed if needed (most likely yes)
4. Segmentation mask prediction??


Concrete Next Steps:
2. Only training center points, check 1 epoch
3. Run it on supercloud

Possibilities:

1. **Predict grasp pose in place of kpt offsets** 
1. Add more augmentations (random crop, random rotation)
2. Add better sim data
4. SAM features in place of RGB
5. output segmentation masks
5. Add Grasp evaluator
6. Small experiment showing the effect of using VAE, by taking a toy dataset
   and learning the distribution by vae and by increased density