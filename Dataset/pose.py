"""
dataset:cmu
'0': Neck
'1': Nose
'2': BodyCenter (center of hips)
'3': lShoulder
'4': lElbow
'5': lWrist,
'6': lHip
'7': lKnee
'8': lAnkle
'9': rShoulder
'10': rElbow
'11': rWrist
'12': rHip
'13': rKnee
'14': rAnkle
'15': lEye
'16': lEar
'17': rEye
'18': rEar
"""
import numpy as np
# all_points = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
# useful_points = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

# # inplace操作
# def select_points(poses):
#     for pose in poses:
#         for id in all_points:
#             if id not in useful_points:
#                 pose[0] = np.delete(pose[0],id,axis=1)  #删除一整列
#                 pose[1] = np.delete(pose[1],id,axis=1)

#     return poses

# def get_level(dataset = None):
#     """
#     选择不同数据集中的path

#         level: cmu
#         1:root -> neck -> nose
#         2:nose -> leye -> lear
#         3:nose -> reye -> rear
#         4:root -> lshoulder -> lelbow -> lwrist
#         5:root -> rshoulder -> relbow -> rwrist
#         6:root -> lhip -> lknee -> lankle
#         7:root -> rhip -> rknee -> rankle

#     """
#     level = {'cmu':[[2,0,1],
#                     [1,15,16],
#                     [1,17,18],
#                     [2,3,4,5],
#                     [2,9,10,11],
#                     [2,6,7,8],
#                     [2,12,13,14]]
#             }
#     return level[dataset]

class Pose():
    def __init__(self,dataset = 'cmu') -> None:
        # self.poses = poses
        self.dataset = dataset
        self.useful_points = []
        self.all_points = []

        if self.dataset == 'cmu':
            self.useful_points = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
            self.all_points = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
        else:
            pass

    # inplace操作
    def select_points(self,coors):
        for coor in coors:
            for id in self.all_points:
                if id not in self.useful_points:
                    coor[0] = np.delete(coor[0],id,axis=1)  #删除一整列
                    coor[1] = np.delete(coor[1],id,axis=1)

        return coors

    def get_level(self,dataset = None):
        """
        选择不同数据集中的path

        level: cmu
        1:root -> neck -> nose
        2:nose -> leye -> lear
        3:nose -> reye -> rear
        4:root -> lshoulder -> lelbow -> lwrist
        5:root -> rshoulder -> relbow -> rwrist
        6:root -> lhip -> lknee -> lankle
        7:root -> rhip -> rknee -> rankle

        """
        level = {'cmu':[[2,0,1],
                        [1,15,16],
                        [1,17,18],
                        [2,3,4,5],
                        [2,9,10,11],
                        [2,6,7,8],
                        [2,12,13,14]]
                }
        return level[dataset]
