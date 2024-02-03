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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch_scatter import scatter_mean

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

# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from torch import nn
import torch.nn.functional as F
# from torch_scatter import scatter_mean


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.size_tensor(x[:, :, 1:, :]) + 1e-4
        count_w = self.size_tensor(x[:, :, :, 1:]) + 1e-4
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def size_tensor(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def get_semantic_weights(reweight_classes, fg_classes, num_semantic_classes):
    weights = torch.ones([num_semantic_classes]).float()
    if reweight_classes:
        weights[fg_classes] = 2
    return weights


class SCELoss(torch.nn.Module):

    def __init__(self, alpha, beta, class_weights):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def forward(self, pred, labels_probabilities):
        # CCE
        ce = self.cross_entropy(pred, labels_probabilities)

        # RCE
        weights = torch.tensor(self.class_weights, device=pred.device).unsqueeze(0)
        pred = F.softmax(pred * weights, dim=1)
        pred = torch.clamp(pred, min=1e-8, max=1.0)
        label_clipped = torch.clamp(labels_probabilities, min=1e-8, max=1.0)

        rce = torch.sum(-1 * (pred * torch.log(label_clipped) * weights), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss


def contrastive_loss(features, instance_labels, temperat):
    #import pdb;pdb.set_trace()
    bsize = features.size(0)
    masks = instance_labels.view(-1, 1).repeat(1, bsize).eq_(instance_labels.clone())
    masks = masks.fill_diagonal_(0, wrap=False)
    import pdb;pdb.set_trace()
    # compute similarity matrix based on Euclidean distance
    distance_sq = torch.pow(features.unsqueeze(1) - features.unsqueeze(0), 2).sum(dim=-1)
    import pdb;pdb.set_trace()
    # temperature = 1 for positive pairs and temperature for negative pairs
    temperature = torch.ones_like(distance_sq) * temperat
    temperature = torch.where(masks==1, temperature, torch.ones_like(temperature))
    import pdb;pdb.set_trace()
    similarity_kernel = torch.exp(-distance_sq/temperature)
    logits = torch.exp(similarity_kernel)

    p = torch.mul(logits, masks).sum(dim=-1)
    Z = logits.sum(dim=-1)
    import pdb;pdb.set_trace()
    prob = torch.div(p, Z)
    prob_masked = torch.masked_select(prob, prob.ne(0))
    import pdb;pdb.set_trace()
    loss = -prob_masked.log().sum()/bsize
    torch.cuda.empty_cache()
    return loss


def ae_loss(features, instance_labels, sigma=1.0):
    # get centroid of each instance
    # for instance in unique_instances: 
    #     centroid = mean(features[instance_labels == instance])
    # verctorized version:
    
    unique_instances, inverse_indices = torch.unique(instance_labels, return_inverse=True)
    centroids = scatter_mean(features, inverse_indices, dim=0, dim_size=unique_instances.shape[0])
    
    # Pull loss: pull features towards their instance centroid
    pull_loss = torch.pow(features - centroids[inverse_indices], 2).sum(dim=-1).mean()
    
    # Push loss: push centroids away from each other
    # for each instance, compute distance to all other instances
    distances = torch.pow(centroids.unsqueeze(1) - centroids.unsqueeze(0), 2).sum(dim=-1) # (num_instances, num_instances)
    
    distances_nondiag = distances[~torch.eye(distances.shape[0], dtype=torch.bool, device=features.device)] # (num_instances * (num_instances - 1))
    push_loss = torch.exp(-distances_nondiag/sigma).mean()
    
    return pull_loss + push_loss


def contrast_loss(features, instance_labels, temperature):
    # temp_features_1 = features[:20000]
    # temp_features_2 = features[20000:]
    # temp_instance_labels_1 = instance_labels[:20000]
    # temp_instance_labels_2 = instance_labels[20000:]
    loss =  ae_loss(features,instance_labels)
    for i in range(4):
        temp_features_1 = features[i*10000:(i+1)*10000]
        temp_instance_labels_1 = instance_labels[i*10000:(i+1)*10000]
        if loss == None:
            loss = ae_loss(temp_features_1,temp_instance_labels_1)
        else:
            loss = loss +  ae_loss(temp_features_1,temp_instance_labels_1,temperature)
    return loss
