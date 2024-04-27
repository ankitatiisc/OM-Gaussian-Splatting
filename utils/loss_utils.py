#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch_scatter import scatter_mean
from info_nce import InfoNCE, info_nce
triplet_loss = nn.TripletMarginLoss(margin=1, p=2, eps=1e-7)
info_nceloss = InfoNCE(negative_mode='paired')

mlp_weights  = torch.from_numpy(np.load('/data/jaswanth/OM-Gaussian-Splatting-org/weights.npy'))
mlp_bias = torch.from_numpy(np.load('/data/jaswanth/OM-Gaussian-Splatting-org/bias.npy'))
def create_embedding_fn(input_X,
                        input_dims = 2,
                        include_input = False,
                        num_freq = 2,
                        log_sampling = True,
                        periodic_fns = [torch.sin, torch.cos]):
    
    max_freq = num_freq-1
    embed_fns = []
    d = input_dims
    out_dim = 0
    if include_input:
        embed_fns.append(lambda x : x)
        out_dim += d
        
    if log_sampling:
        freq_bands = 2.**torch.linspace(0., max_freq, steps=num_freq)
    else:
        freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=num_freq)
        
    for freq in freq_bands:
        for p_fn in periodic_fns:
            embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
            out_dim += d
    return  torch.cat([fn(input_X) for fn in embed_fns], -1)

# from torch_scatter import scatter_mean
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def decomp_loss(pred_ins, gt_labels, ins_num):
    # change label to one hot
    valid_gt_labels = torch.unique(gt_labels)
    gt_ins = torch.zeros(size=(gt_labels.shape[0], ins_num))

    valid_ins_num = len(valid_gt_labels)
    gt_ins[..., :valid_ins_num] = F.one_hot(gt_labels.long())[..., valid_gt_labels.long()]

    cost_ce, cost_siou, order_row, order_col = hungarian(pred_ins, gt_ins, valid_ins_num, ins_num)
    valid_ce = torch.mean(cost_ce[order_row, order_col[:valid_ins_num]])
    
    if not (len(order_col) == valid_ins_num):
        invalid_ce = torch.mean(pred_ins[:, order_col[valid_ins_num:]])
    else:
        invalid_ce = torch.tensor([0]).to(pred_ins.device)
    valid_siou = torch.mean(cost_siou[order_row, order_col[:valid_ins_num]])

    ins_loss_sum = valid_ce + invalid_ce + valid_siou
    return ins_loss_sum


# matching function
def hungarian(pred_ins, gt_ins, valid_ins_num, ins_num):
    @torch.no_grad()
    def reorder(cost_matrix, valid_ins_num):
        valid_scores = cost_matrix[:valid_ins_num]
        valid_scores = valid_scores.cpu().numpy()
        valid_scores = np.nan_to_num(valid_scores, nan=20.0)
        
        row_ind, col_ind = linear_sum_assignment(valid_scores)      
        unmapped = ins_num - valid_ins_num
        if unmapped > 0:
            unmapped_ind = np.array(list(set(range(ins_num)) - set(col_ind)))
            col_ind = np.concatenate([col_ind, unmapped_ind])
        return row_ind, col_ind
    # preprocess prediction and ground truth
    pred_ins = pred_ins.permute([1, 0])
    gt_ins = gt_ins.permute([1, 0])
    pred_ins = pred_ins[None, :, :]
    gt_ins = gt_ins[:, None, :].to(pred_ins.device)
    # import pdb;pdb.set_trace()
    cost_ce = torch.mean(-gt_ins * torch.log(pred_ins + 1e-8) - (1 - gt_ins) * torch.log(1 - pred_ins + 1e-8), dim=-1)

    # get soft iou score between prediction and ground truth, don't need do mean operation
    TP = torch.sum(pred_ins * gt_ins, dim=-1)
    FP = torch.sum(pred_ins, dim=-1) - TP
    FN = torch.sum(gt_ins, dim=-1) - TP
    cost_siou = TP / (TP + FP + FN + 1e-6)
    cost_siou = 1.0 - cost_siou

    # final score
    cost_matrix = cost_ce + cost_siou
    order_row, order_col = reorder(cost_matrix, valid_ins_num)

    return cost_ce, cost_siou, order_row, order_col


def ae_loss(features, instance_labels,valid_instances,gaussians,iteration, sigma=1.0):
   
    features = (features*valid_instances.unsqueeze(dim=-1))
    unique_instances, inverse_indices = torch.unique(instance_labels, return_inverse=True)
    centroids = scatter_mean(features, inverse_indices, dim=0, dim_size=unique_instances.shape[0])

    

    # # Pull loss: pull features towards their instance centroid
    pull_loss = (torch.abs(torch.pow(features - centroids[inverse_indices], 2).sum(dim=-1)-0.00001)*valid_instances).mean()
   


    # Push loss: push centroids away from each other
    # for each instance, compute distance to all other instances
    distances = torch.pow(centroids.unsqueeze(1) - centroids.unsqueeze(0), 2).sum(dim=-1) # (num_instances, num_instances)
    distances_nondiag = distances[~torch.eye(distances.shape[0], dtype=torch.bool, device=features.device)] # (num_instances * (num_instances - 1))
    if(distances_nondiag.shape[0]==0):
        return pull_loss
    
    push_loss = torch.exp(-distances_nondiag/sigma).mean()
    
    #Extra push loss, need to check if we need this or not
    # push_loss_1 = torch.exp(-2*torch.pow(features-0.5,2).sum(dim=-1)).mean()
    
    
    # 3D loss
    if(iteration>15000):
        with torch.no_grad():
            gaussian_means = gaussians.get_xyz
        gaussian_features = gaussians.get_object_ins
        x1 = torch.randperm(gaussian_means.shape[0])[:int(2000)]
        gaussian_means=  gaussian_means[x1]
        gaussian_features = gaussian_features[x1]@mlp_weights.to(features.device) + mlp_bias.to(features.device)
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        mean_dis = torch.pow(gaussian_means.unsqueeze(1) - gaussian_means.unsqueeze(0), 2).sum(dim=-1)
        features_dis = torch.pow(gaussian_features.unsqueeze(1) - gaussian_features.unsqueeze(0), 2).sum(dim=-1)
        mean_dis = (mean_dis/mean_dis.max()) + 0.0001
        features_dis = ((features_dis/features_dis.max()) + 0.0001).view(mean_dis.shape)
        loss_3d = torch.abs(torch.log(features_dis/mean_dis)).mean()
        return pull_loss + 2*push_loss + loss_3d*0.1
    # import pdb;pdb.set_trace()
            
    #Triplet Loss
    # import pdb;pdb.set_trace()
    if((iteration>15000) ):
        full_distances = torch.pow(features - centroids[inverse_indices], 2).sum(dim=-1)
        mean_distances = full_distances.mean()
        indices = full_distances>mean_distances
        anchors = features[indices]@mlp_weights.to(features.device) + mlp_bias.to(features.device)
        positive_indices = valid_instances>0.9
        positives = centroids[inverse_indices][indices]@mlp_weights.to(features.device) + mlp_bias.to(features.device)
        negative_indices =  (inverse_indices+torch.randint(1,unique_instances.shape[0] , size=(features.shape[0],)).to(inverse_indices.device))%unique_instances.shape[0]
        negatives = centroids[negative_indices][indices]@mlp_weights.to(features.device) + mlp_bias.to(features.device)
        riplet_loss_1 = triplet_loss(anchors,positives,negatives)
        return riplet_loss_1*0.2 +pull_loss + 2*push_loss
    
    
    return pull_loss + 2*push_loss +push_loss_1


