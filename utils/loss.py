from math import e
from IPython.terminal.embed import embed
import numpy as np
from numpy.core.defchararray import array, center
import torch
import torch.nn as nn

stage_num = 3
out_up_blocks = 4

def cal_other_loss(predict,gt,mask):
    assert predict.size() == gt.size()
    batch_size = gt.shape[0]
    feature_chs = gt.shape[1]

    loss = 0.0
    pre_ = predict.clone()
    gt_ = gt.clone()
    mask_ = mask.clone()

    for i in range(batch_size):
        for j in range(feature_chs):
            nonzero_num = len(torch.nonzero(mask_[i,0]))
            pre_[i,j] *= mask_[i,0]

            tmp_loss = torch.sum((gt_[i,j] - pre_[i,j]) ** 2)
            if nonzero_num == 0:
                nonzero_num = 1
            loss = loss + (tmp_loss / nonzero_num)

    return loss / batch_size / feature_chs


def cal_center_loss(pre_center,gt_center,mask):
    batch_size = gt_center.shape[0]
    feature_chs = gt_center.shape[1]
    
    loss = 0.
    pre_center_ = pre_center.clone()
    gt_center_ = gt_center.clone()
    mask_ = mask.clone()

    for i in range(batch_size):
        for j in range(feature_chs):
            nonzero_num = len(torch.nonzero(mask[i,j]))
            pre_center_[i,j] *= mask_[i,j]

            tmp_loss = torch.sum(torch.abs(gt_center_[i,j] - pre_center_[i,j]))
            if nonzero_num == 0:
                nonzero_num = 1
            loss = loss + (tmp_loss / nonzero_num)

    return loss / batch_size / feature_chs


def cal_loss(pre_dict,gt_dict):

    #1.center_joint loss
    loss = 0.
    for i in range(stage_num):
        for j in range(out_up_blocks):
            tmp_offset_loss = cal_other_loss(pre_dict['offset_map'][i][j],gt_dict['offset_map'],gt_dict['mask'].cuda())
            # tmp_weight_loss = cal_weight_loss(pre_dict['offset_map_weight'][i][j],gt_dict['offset_map_weight'],loss_fn_1)
            tmp_depth_loss = cal_other_loss(pre_dict['rr_demap'][i][j],gt_dict['rr_demap'],gt_dict['mask'].cuda())
            tmp_center_loss = cal_center_loss(pre_dict['center_map'][i][j],gt_dict['center_map'],gt_dict['mask'].cuda())
            
            temp_loss =  tmp_offset_loss +  tmp_depth_loss +  tmp_center_loss

        loss += temp_loss / out_up_blocks
    return loss / stage_num

        