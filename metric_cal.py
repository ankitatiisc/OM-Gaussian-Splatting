import numpy as np
import os
import cv2
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import tqdm
import torch
import numpy as np
from scipy.spatial.distance import cdist
import torch.nn.functional as F
def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred - image_gt) ** 2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr_hierarchy(image_pred, image_gt, valid_mask=None, reduction='mean'):
    psnr_coarse, psnr_fine = 100, 100
    if 'rgb_coarse' in image_pred:
        psnr_coarse = -10*torch.log10(mse(image_pred['rgb_coarse'].detach(), image_gt, valid_mask, reduction))
    if 'rgb_fine' in image_pred:
        psnr_fine = -10 * torch.log10(mse(image_pred['rgb_fine'].detach(), image_gt, valid_mask, reduction))
    return psnr_coarse, psnr_fine


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred.detach(), image_gt, valid_mask, reduction))


def get_non_robust_classes(confusion_matrix, robustness_thres):
    axis_0 = np.sum(confusion_matrix, axis=0)
    axis_1 = np.sum(confusion_matrix, axis=1)
    total_labels = axis_0.sum()
    non_robust_0 = axis_0 / total_labels < robustness_thres
    non_robust_1 = axis_1 / total_labels < robustness_thres
    return np.where(np.logical_and(non_robust_0, non_robust_1))[0].tolist()


def calculate_miou_1(mask1, mask2,map_ids):
    # Get unique object IDs from both masks
    unique_ids1 = np.unique(mask1)
    unique_ids2 = np.unique(mask2)
    
    # Initialize variables to store intersection and union
    intersection = 0
    union = 0
    
    # Compute intersection and union for each unique object ID pair
    for id1 in unique_ids1:
        # Calculate intersection
        intersection_mask = (mask1 == id1) & (mask2 == map_ids[id1])
        intersection += np.sum(intersection_mask)
        
        # Calculate union
        union_mask = (mask1 == id1) | (mask2 == map_ids[id1])
        union += np.sum(union_mask)
    
    # Calculate mean IoU
    miou = intersection / (union.astype(np.float32))
    
    return miou


def calculate_miou(confusion_matrix,diag_sum, ignore_class=None, robust=0.005):
    import pdb;pdb.set_trace()
    MIoU = np.divide(np.diag(diag_sum), (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)))
    if ignore_class is not None:
        ignore_class += get_non_robust_classes(confusion_matrix, robust)
        for i in ignore_class:
            MIoU[i] = float('nan')
    MIoU = np.nanmean(MIoU)
    return MIoU

def _generate_matrix(gt_image, pre_image):
    mask = (gt_image >= 0) & (gt_image < len(np.unique(gt_image)) )
    label = len(np.unique(pre_image)) * gt_image[mask].astype('int') + pre_image[mask]
    count = np.bincount(label, minlength=len(np.unique(gt_image))* len(np.unique(pre_image)))
    try:
        confusion_matrix = count.reshape(len(np.unique(gt_image)), len(np.unique(pre_image)))
    except:
        import pdb;pdb.set_trace()
    diag_sum = {i:np.argmax(confusion_matrix[i]) for i in np.unique(gt_image)}
    
    
    return confusion_matrix,diag_sum

def generate_matrix(gt_image,pre_image):
    conf_mat = np.zeros((len(np.unique(gt_image)),len(np.unique(pre_image))))
    for id1 in np.unique(gt_image):
        for id2 in np.unique(pre_image):
            conf_mat[id1][id2] = (np.sum((gt_image==id1)&(pre_image==id2))/np.sum((gt_image==id1)|(pre_image==id2)))
    diag_sum = {i:np.argmax(conf_mat[i]) for i in np.unique(gt_image)}
    import pdb;pdb.set_trace()
    return conf_mat,diag_sum


def add_batch(gt_image, pre_image, return_miou=False):
    # import pdb;pdb.set_trace()
    assert gt_image.shape == pre_image.shape
    confusion_matrix,diag_sum = generate_matrix(gt_image, pre_image)
    # import pdb;pdb.set_trace()
    if return_miou:
        return calculate_miou_1(gt_image,pre_image,diag_sum)




    

paths1 = ['/data/jaswanth/OM-Gaussian-Splatting/data/bedroom/train/semantic_instance',
            '/data/jaswanth/OM-Gaussian-Splatting/data/bathroom/train/semantic_instance',
            '/data/jaswanth/OM-Gaussian-Splatting/data/dinning/train/semantic_instance',
            '/data/jaswanth/OM-Gaussian-Splatting/data/office/train/semantic_instance',
            '/data/jaswanth/OM-Gaussian-Splatting/data/large_corridor_25/instance']
paths2 = ['/data/jaswanth/OM-Gaussian-Splatting/output/bedroom_17/train/ours_30000/instance_mask',
        '/data/jaswanth/OM-Gaussian-Splatting/output/bathroom_1/train/ours_30000/instance_mask',
        '/data/jaswanth/OM-Gaussian-Splatting/output/dinning_1/train/ours_30000/instance_mask',
        '/data/jaswanth/OM-Gaussian-Splatting/output/office_16/train/ours_30000/instance_mask',
        '/data/jaswanth/OM-Gaussian-Splatting/output/large_corridor_25_8/train/ours_30000/instance_mask']


dataset = ['bedroom','bathroom','dinning','office','large_corridor_25']

iddd =0
# for path1,path2 in zip(paths1,paths2):
iddd =4
path1 = paths1[iddd]
path2 = paths2[iddd]
gt_masks =sorted(os.listdir(path1))
pred_masks =sorted(os.listdir(path2))
mious = []
counting = 0
for gt_mask_name,pred_mask_name in zip(gt_masks,pred_masks):
    counting = counting+1
    # print(counting,end='\r')
    if(gt_mask_name.endswith('npy')):
        gt_mask = np.load(os.path.join(path1,gt_mask_name))
        # import pdb;pdb.set_trace()
    else:
        gt_mask = cv2.imread(os.path.join(path1,gt_mask_name))
    
    pred_mask = cv2.imread(os.path.join(path2,pred_mask_name))
    # import pdb;pdb.set_trace()
    pred_mask = pred_mask[...,0]
    # pred_mask = remove_least_common_ids(pred_mask,len(np.unique(gt_mask)))
    unique_ids = np.unique(pred_mask)
    pred_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    pred_mask = np.vectorize(pred_id_map.get)(pred_mask)

    unique_ids = np.unique(gt_mask)
    gt_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    gt_mask = np.vectorize(gt_id_map.get)(gt_mask)
    pred_mask = np.resize(pred_mask,gt_mask.shape)
    # import pdb;pdb.set_trace()
    iou = add_batch(gt_mask,pred_mask,return_miou=True)
    print(counting,iou,end='\r')
    mious.append(iou)
    

print(f'Mean IoU of {dataset[iddd]} is {np.array(mious).mean()}')