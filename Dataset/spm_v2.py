"""
this version is to change the maps' format
"""
import enum
from IPython.terminal.embed import embed
import numpy as np
import math
from torch.nn.functional import hinge_embedding_loss
from torch.onnx import select_model_mode_for_export
from Dataset.pose import Pose
from utils.utils import create_center_map , create_heatmap
import torch

# joint_num = len(useful_points)

class SingleStageLabel():
    def __init__(self,height,width,centers,pcoors,ccoors,cnf) -> None:
        """
        pcoors --> pixel_coors（scaled）the third row is the depth(in world coordinate...)
        ccoors --> cam_coors
        """
        # self.dataset = 'cmu'
        self.level = []
        self.cnf = cnf
        self.centers = centers
        self.centers_offset = []
        self.pcoors = pcoors
        self.ccoors = ccoors
        self.height = height
        self.width = width
        self.Z = math.sqrt(self.height ** 2 + self.width ** 2)
        self.norm = math.sqrt(self.width + self.height)

        pose = Pose(dataset=self.cnf.data_format)
        self.joint_num = len(pose.useful_points)
        self.joint_num_ = len(pose.all_points)

        if self.joint_num != self.joint_num_:
            self.pcoors = pose.select_points(self.pcoors)
            self.ccoors = pose.select_points(self.ccoors)
        # self.Z = 1
        
        #center_map
        self.center_map = np.zeros(shape=(self.height,self.width), dtype=np.float32)
        self.center_mask = np.zeros(shape=(self.height,self.width),dtype=np.uint8)  #后面的mask都用这个就可以
        #offset_map
        self.offset_map = np.zeros(shape=((self.joint_num - 1) * 2 , self.height,self.width),
                                           dtype=np.float32)         #offset_map:offset放在center中心点的一定范围中 18*2 channels
        self.offset_map_tmp = self.offset_map.copy()
        self.kps_count = np.zeros(shape=((self.joint_num - 1) * 2 , self.height,self.width),
                                          dtype=np.int64)
        self.kps_count_tmp = self.kps_count.copy()
        #relative root map
        self.rel_demap = np.zeros(shape=(self.joint_num,self.height,self.width),
                                  dtype=np.float32)  #相对父节点深度图
        self.rel_demap_tmp = self.rel_demap.copy()
        #level
        self.level = pose.get_level(self.cnf.data_format)

    def create_2D_label(self):
        """
        centers:(x,y)
        pcoors:(x,y)
        """
        for i , center in enumerate(self.centers):  #对一张图中的所有人进行遍历
            #if center out of area , pass
            if center[0] < 0 or center[0] > self.width or \
                    center[1] < 0 or center[1] > self.height:
                continue
            self.center_map,self.center_mask,centers_offset_list = create_center_map(self.center_map,\
                                                                            center,self.center_mask,\
                                                                            self.centers_offset,sigma = self.cnf.sigma)
            self.body_joint_displacement(center,self.pcoors[i],centers_offset_list)
        
        self.kps_count[self.kps_count == 0] = 1
        self.offset_map = np.divide(self.offset_map , self.kps_count)
        # self.other_joints_heatmap = create_heatmap(self.other_joints_heatmap,self.pcoors,self.joint_num)

        return  self.center_map ,self.center_mask, self.offset_map

    def body_joint_displacement(self,center,coor,cen_oft_list):
        """
        在中心点一个范围内，都进行偏移的计算
        """
        ch_index = 0
        for single_path in self.level:   #遍历每一条通道    
            for i in range(len(single_path) - 1):

                start_id = single_path[i]
                end_id = single_path[i+1]
                # print(start_id , end_id)
                start_joint = coor[:2,start_id]
                end_joint = coor[:2,end_id]

                #父节点不在范围中，就直接pass掉
                if start_joint[0] < 0 or start_joint[0] > self.width or \
                        start_joint[1] < 0 or start_joint[1] > self.height:
                    ch_index += 2
                    break

                self.create_dense_displacement_map(start_joint,end_joint,center,ch_index,cen_oft_list)
                ch_index += 2    
        
    def create_dense_displacement_map(self,start_joint,end_joint,center,ch_index,cen_oft_list,sigmax = 6,sigmay = 6):
        
        start_x = int(start_joint[0])
        start_y = int(start_joint[1])
        center_x = int(center[0])
        center_y = int(center[1])

        th = 4.3
        delta = np.sqrt(th * 2)

        #这里是得到父节点附近的范围
        x0 = int(max(0, start_x - delta * sigmax + 0.5))          # (x0,y0), (x1,y1)为参考点周围点的最外层点坐标
        x1 = int(min(self.width, start_x + delta * sigmax + 0.5))
        y0 = int(max(0, start_y - delta * sigmay + 0.5))
        y1 = int(min(self.height, start_y + delta * sigmay + 0.5))

        up_tmp = start_y - y0
        bot_tmp = y1 - start_y
        left_tmp = start_x - x0
        right_tmp = x1 - start_x

        up = min(up_tmp,cen_oft_list[0][0])
        bot = min(bot_tmp,cen_oft_list[0][1])
        left = min(left_tmp,cen_oft_list[0][2])
        right = min(right_tmp,cen_oft_list[0][3])

        x_offset = 0
        y_offset = 0

        new_y0 = start_y - up
        new_y1 = start_y + bot
        new_x0 = start_x - left
        new_x1 = start_x + right
        
        #after carry the operation on the temp map
        #load the values to real map
        for x in range(new_x0,new_x1):
            for y in range(new_y0,new_y1):
                dis = np.sqrt((start_joint[0] - x)**2 + (start_joint[1] - y)**2)
                if dis > th:
                    continue
                else:
                    x_offset = (end_joint[0] - x) / self.Z
                    y_offset = (end_joint[1] - y) / self.Z

                self.offset_map_tmp[ch_index,y,x] += y_offset
                self.offset_map_tmp[ch_index+1,y,x] += x_offset
                # self.offset_map_weight_[ch_index:ch_index+2,y,x] = 1   #当在推理的时候，可以通过这个进行判断这个关节点是否可见,大于或者等于0.5就说明这个点是可见的
                #center周围点(x,y)不和关节点重合，则人数增加一个
                if end_joint[1] != y or end_joint[0] != x:
                    self.kps_count_tmp[ch_index:ch_index+2,y,x] += 1

        self.offset_map[ch_index:(ch_index+2),(center_y-up):(center_y+bot),(center_x-left):(center_x+right)] = \
                        self.offset_map_tmp[ch_index:(ch_index+2),new_y0:new_y1,new_x0:new_x1]
        self.kps_count[ch_index:(ch_index+2),(center_y-up):(center_y+bot),(center_x-left):(center_x+right)] = \
                        self.kps_count_tmp[ch_index:ch_index+2,new_y0:new_y1,new_x0:new_x1]

    def create_3D_label(self,sigmax = 6,sigmay = 6):
        """
        相对父节点深度图,直接放在center joint的位置
        在转换数据集便签格式的时候应该换成：3D --> (3,N) --> (X,Y,Z)
        这里面应该用的是ccoors,而不是pcoors....
        """

        for idx , pcoor in enumerate(self.pcoors):  #遍历所有的人
            all_joint_depth = pcoor[2,:]  #第三行所有,用的是相机坐标系下的深度...
            center = pcoor[:2,2]
            
            if center[0] < 0 or center[0] > self.width or \
                    center[1] < 0 or center[1] > self.height:
                continue
            
            th = 4.3
            delta = math.sqrt(th * 2)

            x0 = int(max(0,center[0] - delta * sigmax + 0.5))
            x1 = int(min(self.width,center[0] + delta * sigmax +0.5))
            y0 = int(max(0,center[1] - delta * sigmay + 0.5))
            y1 = int(min(self.height, center[1] + delta * sigmay + 0.5))

            #制作root joint的depth map
            for x in range(x0,x1):
                for y in range(y0,y1):
                    dis = np.sqrt((center[0] - x) ** 2 + (center[1] - y) ** 2)
                    if dis > th or self.rel_demap_tmp[0,y,x] != 0:     #后面判断条件是如果前面一个人已经占据了，下一个人的值就不准覆盖，设定最后推理的范围为3
                        continue
                    self.rel_demap_tmp[0,y,x] = all_joint_depth[2] / self.Z #针对root joint的id顺序                           

            ch_index = 1
            for single_path in self.level:
                for i in range(len(single_path) - 1):
                    
                    start_id = single_path[i]
                    end_id = single_path[i+1]

                    # if father_coor[0] < 0 or father_coor[1] > self.width or \
                    #         father_coor[1] < 0 or father_coor[1] > self.height :
                    #     if i == len(single_path) - 2:
                    #         ch_index += 1
                    #     continue

                    """
                    这里的问题：因为有的点是超出了范围，因此，0 和 1的范围可能会不一样，但是在训练的时候这个点不会出现就不会计算loss
                    """

                    start_dep = all_joint_depth[start_id]
                    end_dep = all_joint_depth[end_id]

                    rel_dep = (end_dep - start_dep) / self.Z  #normalize
                    # embed()
                    for x in range(x0,x1):
                        for y in range(y0,y1):
                            dis = np.sqrt((center[0] - x) ** 2 + (center[1] - y) ** 2)
                            if dis > th or self.rel_demap_tmp[ch_index,y,x] != 0:
                                continue
                            self.rel_demap_tmp[ch_index,y,x] = rel_dep
                    ch_index += 1 
            self.rel_demap[:,y0:y1,x0:x1] = self.rel_demap_tmp[:,y0:y1,x0:x1]
        return self.rel_demap

