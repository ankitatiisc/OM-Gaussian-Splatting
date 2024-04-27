import os
import numpy as np
from PIL import Image
import cv2
# from skimage.measure import label, regionprops
# from skimage.metrics import pairwise_iou
colors = np.random.randint(0, 255, size=(5000, 3))
def create_color_mask(binary_tensor):
    unique_codes = np.unique(binary_tensor)
    image = np.zeros((binary_tensor.shape[0],binary_tensor.shape[1], 3), dtype=np.float32)
    image_instance = np.zeros((binary_tensor.shape[0],binary_tensor.shape[1]), dtype=np.int32)
    # Fill the image with colors based on binary codes
    for i, code in enumerate(unique_codes):
        mask = binary_tensor == code
        image[mask] = colors[int(code)]
        image_instance[mask]= int(code)+1
    return image
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Function to find the maximum IoU between a ground truth mask and a set of predicted masks
def max_iou(gt_mask, pred_masks):
    max_iou = 0
    matching_pred_id = None
    for pred_id, pred_mask in pred_masks.items():
        iou = calculate_iou(gt_mask, pred_mask)
        if iou > max_iou:
            max_iou = iou
            matching_pred_id = pred_id
    return max_iou, matching_pred_id

# Path to the folders containing the ground truth masks and predicted masks
gt_folder = "/data/jaswanth/OM-Gaussian-Splatting-org/data/replica/room_0/rs_semantics"
pred_folder = "/data/jaswanth/OM-Gaussian-Splatting-org/output/replica/room0_1_with_mlp_3d_loss/test/ours_50000/pred_semantics"
output_folder = "/data/jaswanth/OM-Gaussian-Splatting-org/output/replica/room0_1_with_mlp_3d_loss/test/ours_50000/new_pred_semantics"
vis_out_folder = "/data/jaswanth/OM-Gaussian-Splatting-org/output/replica/room0_1_with_mlp_3d_loss/test/ours_50000/new_vis_semantics"
vis_gt_folder = "/data/jaswanth/OM-Gaussian-Splatting-org/output/replica/room0_1_with_mlp_3d_loss/test/ours_50000/gt_vis_semantics"
# os.mkdir(output_folder)
# os.mkdir(vis_out_folder)
# os.mkdir(vis_gt_folder)
# Iterate over each ground truth mask
for gt_mask_file in os.listdir(gt_folder):
    # Load ground truth mask
    
    gt_mask = np.array(Image.open(os.path.join(gt_folder, gt_mask_file)))
    
    # Load corresponding predicted mask
    pred_mask_file = gt_mask_file.replace("png", "npy")
    if(not os.path.exists(os.path.join(pred_folder, pred_mask_file))):
        continue
    pred_mask = np.load(os.path.join(pred_folder, pred_mask_file)) + 100
    gt_mask = cv2.resize(gt_mask, pred_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
    # Create a dictionary to store predicted masks with their corresponding IDs
    pred_masks = {pred_id: pred_mask==pred_id for pred_id in np.unique(pred_mask) if pred_id != 0}
    gt_unique = np.unique(gt_mask)
    # Iterate over each pixel in the ground truth mask
    # import pdb;pdb.set_trace()
    for i in gt_unique:
        if(1):
            gt_id = i
            
            # # Skip background pixels
            # if gt_id == 0:
            #     continue
            
            gt_mask_region = (gt_mask == gt_id)
            
            # Find the most similar region in the predicted mask
            max_iou_value, matching_pred_id = max_iou(gt_mask_region, pred_masks)
            
            # Replace the predicted mask ID with the ground truth ID
            if matching_pred_id is not None:
                pred_mask[pred_mask == matching_pred_id] = gt_id
    for i in np.unique([pred_mask]):
        if(i>98):
            pred_mask[pred_mask == i] = 101
    # import pdb;pdb.set_trace()
    color_img = create_color_mask(pred_mask)
    color_gt = create_color_mask(gt_mask)

    # import pdb;pdb.set_trace()
    cv2.imwrite(os.path.join(vis_out_folder, gt_mask_file),color_img)
    cv2.imwrite(os.path.join(vis_gt_folder, gt_mask_file),color_gt)
    # Save the updated predicted mask in the output folder
    output_mask_file = gt_mask_file.replace("png", "npy")
    output_path = os.path.join(output_folder, output_mask_file)
    np.save(output_path, pred_mask)
