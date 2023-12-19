
import cv2
import numpy as np
import os
def save_anns(anns,name):
    
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 1))
    img[:,:] = -1
    object_id = 1
    for ann in sorted_anns:
        m = ann['segmentation']
        
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = object_id
        object_id = object_id+1
    print(len(anns))
    
    cv2.imwrite(name,img)


import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=5,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=800,  # Requires open-cv to run post-processing
)

folder_path = '/data/jaswanth/OM-Gaussian-Splatting/data/garden/images/'
output_folder = '/data/jaswanth/OM-Gaussian-Splatting/data/garden/masks/'
images = os.listdir(folder_path)

for image_name in images:
    image_path = folder_path+image_name
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    output_path = output_folder +image_name
    save_anns(masks,output_path)

