from numpy.core.fromnumeric import size
# from Dataset.pose import Pose
from re import L
from typing import ContextManager
from IPython import embed
import cv2
import PIL
import json
import numpy as np
from numpy.core.defchararray import _join_dispatcher, center
import torch
from torch.nn.functional import hinge_embedding_loss
import torch.optim as optim
import math
from torchvision.transforms.functional import scale
from operator import itemgetter

#CMU
levels = [[2,0,1],
         [1,15,16],
         [1,17,18],
         [2,3,4,5],
         [2,9,10,11],
         [2,6,7,8],
         [2,12,13,14]]
sort_index = [1,2,0,7,8,9,13,14,15,10,11,12,16,17,18,3,4,5,6]

def imread(path):
    # with open(path, 'rb') as f:
    #     with PIL.Image.open(f) as img:
    #         # img = img.resize(size=(540,960))
    #         return img.convert('RGB')

    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, (832,512))
    return img

def read_json(path):
    with open(path,'rb') as file:
        data = json.load(file)
    return data


def prepare_centers(ori_shape,out_shape,coors):  # coors --> pixel_coors
    """
    在这里要得到的是在像素平面中的中心点的位置
    """
    centers = []
    factory = ori_shape[0] / out_shape[0]  #shape顺序　--> (h,w) == (y,x)
    factorx = ori_shape[1] / out_shape[1]  #x,y方向上的缩放因子　--> 默认为4

    for coor in coors:
        ori_center = coor[:2,2]  #center顺序 --> (x,y) 
        scale_center = (ori_center[0] / factorx , ori_center[1] / factory)  #(x,y)
        centers.append(scale_center)
    return centers

def prepare_keypoints(ori_shape,out_shape,cam_coors,pixel_coors,dataset=None):
    """
    inplace operation
    像素坐标进行了scale
    """
    factory = ori_shape[0] / out_shape[0]  #shape顺序　--> (h,w) == (y,x)
    factorx = ori_shape[1] / out_shape[1]  #x,y方向上的缩放因子　--> 默认为4

    if dataset == 'cmu':
        # for cam_coor in cam_coors:
        #     cam_coor[[0,1],:] = cam_coor[[1,0],:]
        for pixel_coor in pixel_coors:
            # pixel_coor[[0,1],:] = pixel_coor[[1,0],:]
            pixel_coor[0,:] /= factorx  # x
            pixel_coor[1,:] /= factory  # y
    else:
        pass
    return cam_coors , pixel_coors

# def create_center_map(center_map,center,mask,sigma = 6):
#     """
#     高斯分布:一小块一小块的进行高斯
#     """
#     center_x , center_y = int(center[0]) , int(center[1]) #(x,y)
#     th = 4.6052
#     delta = math.sqrt(th * 2)

#     height = center_map.shape[0] 
#     width = center_map.shape[1]

#     x0 = int(max(0,center_x - delta * sigma + 0.5))
#     y0 = int(max(0,center_y - delta * sigma + 0.5))

#     x1 = int(min(width, center_x + delta * sigma + 0.5))
#     y1 = int(min(height, center_y + delta * sigma + 0.5))

#     if x0 > width or x1 < 0 or x1 <= x0:
#         return center_map,mask
#     if y0 > height or y1 < 0 or y1 <= y0:
#         return center_map,mask

#     ## fast way
#     arr_heat = center_map[y0:y1, x0:x1]  #　一整张图  center_map 只有一个channnel
#     exp_factorx = 1 / 2.0 / sigma / sigma # (1/2) * (1/sigma^2)
#     exp_factory = 1 / 2.0 / sigma / sigma
#     x_vec = (np.arange(x0, x1) - center_x) ** 2
#     y_vec = (np.arange(y0, y1) - center_y) ** 2
#     arr_sumx = exp_factorx * x_vec
#     arr_sumy = exp_factory * y_vec
#     xv, yv = np.meshgrid(arr_sumx, arr_sumy)   #这一步是进行网格化
#     # print(xv.shape,yv.shape)
#     # embed()
#     arr_sum = xv + yv
#     arr_exp = np.exp(-arr_sum)
#     arr_exp[arr_sum > th] = 0

#     center_map[y0:y1, x0:x1] = np.maximum(arr_heat, arr_exp)
    
#     mask[y0:y1, x0:x1] = 1
#     mask[y0:y1, x0:x1][arr_sum > th] = 0

#     return center_map,mask

def create_center_map(center_map,center,mask,centers_offset,sigma = 6):
    """
    高斯分布:一小块一小块的进行高斯
    """
    cen_offset = []
    center_x , center_y = int(center[0]) , int(center[1]) #(x,y)
    th = 4.3
    delta = math.sqrt(th * 2)

    height = center_map.shape[0] 
    width = center_map.shape[1]

    x0 = int(max(0,center_x - delta * sigma + 0.5))
    y0 = int(max(0,center_y - delta * sigma + 0.5))

    x1 = int(min(width, center_x + delta * sigma + 0.5))
    y1 = int(min(height, center_y + delta * sigma + 0.5))

    if x0 > width or x1 < 0 or x1 <= x0:
        return center_map,mask,cen_offset
    if y0 > height or y1 < 0 or y1 <= y0:
        return center_map,mask,cen_offset

    #store the offset relative to all centers
    up = center_y - y0
    bot = y1 - center_y
    left = center_x - x0
    right = x1 - center_x
    cen_offset.append((up,bot,left,right))
    ## fast way
    arr_heat = center_map[y0:y1, x0:x1]  #　一整张图  center_map 只有一个channnel

    exp_factorx = 1 / 2.0 / sigma / sigma # (1/2) * (1/sigma^2)
    exp_factory = 1 / 2.0 / sigma / sigma
    x_vec = (np.arange(x0, x1) - center_x) ** 2
    y_vec = (np.arange(y0, y1) - center_y) ** 2
    arr_sumx = exp_factorx * x_vec
    arr_sumy = exp_factory * y_vec
    xv, yv = np.meshgrid(arr_sumx, arr_sumy)   #这一步是进行网格化
    # print(xv.shape,yv.shape)

    arr_sum = xv + yv
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    mask_tmp = arr_exp.copy()
    mask_tmp[arr_sum < th] = 1

    center_map[y0:y1, x0:x1] = np.maximum(arr_heat, arr_exp)
    mask[y0:y1, x0:x1] = np.maximum(mask_tmp,mask[y0:y1, x0:x1])

    return center_map,mask,cen_offset

def create_heatmap(joints_heatmap,coors,joint_num,sigma = 5,mask=None):
    """
    高斯分布
    """
    th = 3
    delta = 2
    height = joints_heatmap.shape[1]
    width = joints_heatmap.shape[2]

    for coor in coors:

        coor = np.delete(coor,2,axis=1)  #root id --> 2

        for i in range(joint_num-1):
            joint_x , joint_y = int(coor[0,i]) , int(coor[1,i])
            
            x0 = int(max(0,joint_x - delta * sigma + 0.5))
            y0 = int(max(0,joint_y - delta * sigma + 0.5))
            x1 = int(min(width, joint_x + delta * sigma + 0.5))
            y1 = int(min(height, joint_y + delta * sigma + 0.5))
            # print(x0,x1,y0,y1)

            ## fast way
            arr_heat = joints_heatmap[i,y0:y1, x0:x1]  #　
            exp_factorx = 1 / 2.0 / sigma / sigma # (1/2) * (1/sigma^2)
            exp_factory = 1 / 2.0 / sigma / sigma
            x_vec = (np.arange(x0, x1) - joint_x) ** 2
            y_vec = (np.arange(y0, y1) - joint_y) ** 2
            arr_sumx = exp_factorx * x_vec
            arr_sumy = exp_factory * y_vec
            xv, yv = np.meshgrid(arr_sumx, arr_sumy)   #这一步是进行网格化
            arr_sum = xv + yv
            arr_exp = np.exp(-arr_sum)
            arr_exp[arr_sum > th] = 0

            print(arr_heat.shape,arr_exp.shape)

            joints_heatmap[i,y0:y1, x0:x1] = np.maximum(arr_heat, arr_exp)
    return joints_heatmap

def convert_coor_format():
    pass


def nms(center_map,th = 0.75,need_confi = False):
    """
    center_map: (1,1,270,480)
    """
    center_map = center_map.squeeze().cpu().numpy()  #-->把shape进行转换
    center_map[center_map < th] = 0  #小于阈值的首先置零

    height , width = int(center_map.shape[0]),int(center_map.shape[1])
    map_left = np.zeros((height,width),dtype=np.float32)
    map_right = np.zeros((height,width),dtype=np.float32)
    map_up = np.zeros((height,width),dtype=np.float32)
    map_bottom = np.zeros((height,width),dtype=np.float32)

    map_weight = np.zeros((height,width)) 

    map_left[:,:-1] = center_map[:,1:]
    map_right[:,1:] = center_map[:,:-1]
    map_up[:-1,:] = center_map[1:,:]
    map_bottom[1:,:] = center_map[:-1,:]

    map_weight[center_map >= map_left] = 1
    map_weight[center_map >= map_right] += 1
    map_weight[center_map >= map_up] += 1
    map_weight[center_map >= map_bottom] += 1
    map_weight[center_map >= th] += 1
    map_weight[map_weight != 5] = 0

    # peaks = np.argwhere(map_weight[1:(height - 1),1:(width - 1)] != 0)  #如果用torch --> torch.nonzero(..).cpu() 从2开始是排除在边缘的坐标点

    peaks = list(zip(np.nonzero(map_weight[1:(height - 1),1:(width - 1)])[1], np.nonzero(map_weight[1:(height - 1),1:(width - 1)])[0]))
    peaks = sorted(peaks, key=itemgetter(0))  #排序

    suppressed = np.zeros(len(peaks), np.uint8)
    keypoints_with_score = []

    #过滤距离较近的peak
    for i in range(len(peaks)):
        if suppressed[i] or center_map[peaks[i][1],peaks[i][0]] < th:
            continue
        for j in range(i+1,len(peaks)):
            if math.sqrt((peaks[i][0] - peaks[j][0]) ** 2 + (peaks[i][1] - peaks[j][1]) ** 2 ) < 6 or \
                          center_map[peaks[j][1],peaks[j][0]] < th :
                suppressed[j] = 1

        keypoints_with_score.append((peaks[i][1],peaks[i][0],center_map[peaks[i][1],peaks[i][0]]))      

    # if need_confi:
    #     confinces = [center_map[peak[0],peak[1]].item() for peak in peaks]
    #     return torch.Tensor(peaks),torch.Tensor(confinces)
    # return list(peaks)
    
    return keypoints_with_score

def nms_v2(center_map,th = 0.75,need_confi = False):
    center_map = center_map.squeeze().cpu().numpy()
    center_map[center_map < th] = 0
    center_map_with_borders = np.pad(center_map, [(2,2),(2,2)],mode='constant')
    map_center = center_map_with_borders[1:center_map_with_borders.shape[0]-1,1:center_map_with_borders.shape[1]-1]
    map_left = center_map_with_borders[1:center_map_with_borders.shape[0]-1,2:center_map_with_borders.shape[1]]
    map_right = center_map_with_borders[1:center_map_with_borders.shape[0]-1,0:center_map_with_borders.shape[1]-2]
    map_up = center_map_with_borders[2:center_map_with_borders.shape[0],1:center_map_with_borders.shape[1]-1]
    map_bottom = center_map_with_borders[0:center_map_with_borders.shape[0]-2, 1:center_map_with_borders.shape[1]-1]

    map_peaks = (map_center > map_left) &\
                (map_center > map_right) &\
                (map_center > map_up) &\
                (map_center > map_bottom)  #大于周围的四个像素点

    map_peaks = map_peaks[1:map_center.shape[0]-1, 1:map_center.shape[1]-1]
    keypoints = list(zip(np.nonzero(map_peaks)[1], np.nonzero(map_peaks)[0])) #zip自动打包
    keypoints = sorted(keypoints, key=itemgetter(0))  #排序

    #去除距离较近的不正确的点
    suppressed = np.zeros(len(keypoints), np.uint8)
    keypoints_with_score = []
    keypoint_num = 0
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i+1,len(keypoints)):
            if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 + \
                         (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                suppressed[j] = 1

        keypoints_with_score.append((keypoints[i][1],keypoints[i][0],center_map[keypoints[i][1],keypoints[i][0]]))


    return keypoints_with_score


#get the depth
def association(center_joints,offset_maps,depth_maps,out_shape=(128,208),ori_shape=(1080,1920)):
    """
    this function can associate every single pose ,and convert the pose to real pixel cooor
    center_joints:[center1,center_2,...]
    offset_maps:(1,36,128,208)
    do not carry refine....
    """
    root_id = 2
    Z = np.sqrt(out_shape[0] ** 2 + out_shape[1] ** 2)
    factorx = ori_shape[1] / out_shape[1]
    factory = ori_shape[0] / out_shape[0]
    scale_x = Z * factorx
    scale_y = Z * factory

    # center_joints = center_joints.cpu().numpy()
    # offset_maps = offset_maps.cpu().numpy()
    # depth_maps = depth_maps.cpu().numpy()

    poses = []  #all poses
    for center in center_joints:
        pose = [] #with depth
        root_depth = (depth_maps[0][center[:2]] * Z)
        root_x = center[1] * factorx
        root_y = center[0] * factory
        root_joint = (root_id,root_y,root_x,root_depth)
        pose.append(root_joint)  #store root joint

        offset_ch = 0
        depth_ch = 1
        for i, single_path in enumerate(levels):
            if i != 1 and i != 2:
                start_x = int(root_x)
                start_y = int(root_y)
                start_depth = root_depth
            if i == 1 or i == 2:
                start_x = int(pose[2][2])
                start_y = int(pose[2][1])
                start_depth = pose[2][3]

            for j in range(len(single_path) - 1):
                jtype = single_path[j+1]
                relative_depth = depth_maps[depth_ch,center[0],center[1]] * Z
                offset_y = offset_maps[offset_ch,center[0],center[1]] * scale_y
                offset_x = offset_maps[offset_ch+1,center[0],center[1]] * scale_x
                if j == 0:
                    current_joint_x = (start_x + offset_x)
                    current_joint_y = (start_y + offset_y)
                    current_joint_depth = (start_depth + relative_depth)
                if j != 0:
                    current_joint_x = (pose[-1][2] + offset_x)
                    current_joint_y = (pose[-1][1] + offset_y)
                    current_joint_depth = (pose[-1][3] + relative_depth)
                current_joint = (jtype,current_joint_y,current_joint_x,current_joint_depth)
                pose.append(current_joint)

                offset_ch += 2
                depth_ch += 1

        pose = np.array(pose)
        
        pose = pose[np.argsort(pose[:,0]),:]
        poses.append(pose)
        embed()
    return poses

def association_v2(center_joints,offset_maps,depth_maps,out_shape=(128,208),ori_shape=(1080,1920)):
    """
    this function can associate every single pose ,and convert the pose to real pixel cooor
    center_joints:[center1,center_2,...]
    offset_maps:(1,36,128,208)
    do not carry refine....
    """
    root_id = 2
    Z_norm = np.sqrt(out_shape[0] ** 2 + out_shape[1] ** 2)
    factorx = ori_shape[1] / out_shape[1]
    factory = ori_shape[0] / out_shape[0]
    scale_x = Z_norm * factorx
    scale_y = Z_norm * factory

    # center_joints = center_joints.cpu().numpy()
    # offset_maps = offset_maps.cpu().numpy()
    # depth_maps = depth_maps.cpu().numpy()

    poses = []  #all poses
    for center in center_joints:
        x = []
        y = []
        Z = []
        pose = [] #with depth
        root_depth = depth_maps[0][center[:2]] * Z_norm
        root_x = center[1] * factorx
        root_y = center[0] * factory
        x.append(root_x)
        y.append(root_y)
        Z.append(root_depth)

        offset_ch = 0
        depth_ch = 1
        for i, single_path in enumerate(levels):
            if i != 1 and i != 2:
                start_x = int(root_x)
                start_y = int(root_y)
                start_depth = root_depth
            if i == 1 or i == 2:
                start_x = int(x[2])
                start_y = int(y[2])
                start_depth = Z[2]

            for j in range(len(single_path) - 1):
                jtype = single_path[j+1]
                relative_depth = depth_maps[depth_ch,center[0],center[1]] * Z_norm
                offset_y = offset_maps[offset_ch,center[0],center[1]] * scale_y
                offset_x = offset_maps[offset_ch+1,center[0],center[1]] * scale_x
                if j == 0:
                    current_joint_x = (start_x + offset_x)
                    current_joint_y = (start_y + offset_y)
                    current_joint_depth = (start_depth + relative_depth)
                if j != 0:
                    current_joint_x = (x[-1] + offset_x)
                    current_joint_y = (y[-1] + offset_y)
                    current_joint_depth = (Z[-1] + relative_depth)
                x.append(current_joint_x)
                y.append(current_joint_y)
                Z.append(current_joint_depth)

                offset_ch += 2
                depth_ch += 1

        pose.append(x)
        pose.append(y)
        pose.append(Z) 
        pose = np.array(pose)
        pose[:,:] = pose[:,sort_index]
        poses.append(pose)
    
    return poses

def to_cam_3d(poses,cam):
    """
    poses: person_num * 3 * 19  3 -- (x,y,z)
    """
    cam_pose = []
    for pose in poses:
        tmp_pose = np.zeros(shape=pose.shape,dtype=np.float32)
        tmp_pose[0,:] = (pose[0,:] - cam[0,2]) * pose[2,:] / cam[0,0]  # x
        tmp_pose[1,:] = (pose[1,:] - cam[1,2]) * pose[2,:] / cam[1,1]  # y
        tmp_pose[2,:] = pose[2,:]

        cam_pose.append(tmp_pose)
    
    return cam_pose

def sum_features(features):
    stage_num = 3
    out_block = 4
    sum_feature = torch.zeros(size=features[2][3].shape).cuda()
    for i in range(stage_num):
        for j in range(out_block):
            sum_feature += features[i][j]
    
    return sum_feature / 3 / 4

if __name__ == '__main__':
    center_map = np.zeros(shape=(512,832),dtype=np.float32)
    mask = np.zeros(shape=(512,832),dtype=np.uint8)
    center_1 = (1,1)
    center_2 = (3,10)  #传进去的center坐标是（x,y） ,但是如果是图片的话，和图片中的坐标是一致的，NMS之后出来的是(y,x) 所以最后要对x\y进行转换
    center_3 = (100,100)
    center_4 = (20,5)
    offset = []
    center_map,mask,a_list = create_center_map(center_map,center_1,mask,offset)
    center_map,mask,a_list = create_center_map(center_map,center_2,mask,offset)
    # center_map,mask = create_center_map(center_map,center_3,mask,offset)
    # center_map,mask = create_center_map(center_map,center_4,mask,offset)
    embed()
    center_map = torch.from_numpy(center_map).unsqueeze(0).unsqueeze(0)
    k = nms(center_map)