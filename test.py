from IPython.terminal.embed import embed
import numpy as np
import torch
import torch.nn as nn

# x = np.array([[1,2,3],
#               [4,5,6],
#               [7,8,9]])

# x = np.delete(x,1,axis=1)
# print(x)

# cam_id = '00_16'
# lnum , rnum = int(cam_id.split('_')[0]) , int(cam_id.split('_')[1])
# print(lnum,rnum)

# x = np.random.rand(2,2,2)
# y = np.random.rand(2,2,2)
# embed()

# x = np.array([[[1,2],[3,4]]])
# y = np.array([[[2,3],[4,4]]])

# inputs = torch.autograd.Variable(torch.from_numpy(x))
# targets = torch.autograd.Variable(torch.from_numpy(y))

# loss_fn = nn.MSELoss(reduce='mean')
# loss_1 = loss_fn(inputs.float(), targets.float())
# embed()

# t = []
# x = (1,2,3)
# t.append(list(x))
# embed()

# x = np.array([[3,1,1],[0,1,1],[1,1,1],[15,1,1],[16,1,1]])
# x = x[np.lexsort(x[:,:].T)]


# x = torch.randn((2,36,20,20))

# x = x.view(1,36,20,20)

# x = np.array([[[1,2],[1,2]],
#               [[3,4],[3,4]]])

# y = np.array([[[2,2],[2,2]],
#               [[3,3],[3,3]]])

# for i in range(len(x)):
#     x[i] = x[i].T

# for i in range(len(x)):
#     for j in range(len(y)):
#         for k in range(2):
#             y_dis = np.sum((x[i][k] - y[j][k]) ** 2)
#             embed()

# from path import Path

# path_1 = Path('test/path')
# print(path_1 / f'1_{path_1}

# x = torch.rand((2,2,3,3))
# y = torch.rand((2,2,3,3))
# loss_fn = nn.MSELoss(reduction='mean')
# loss = loss_fn(x,y)

# x = np.ones((2,2,2,2))
# y = np.ones((2,2,2,2)) * 2

# x = torch.tensor(x)
# y = torch.tensor(y)

# loss_fn = nn.MSELoss(reduction='mean')
# loss = loss_fn(x,y)

# embed()

# x = [1,1,1,1]
# y = np.mean(x)
# embed()

# class test:
#     def __init__(self):
#         setattr(self, 'fn1', 20)
    
#     def get_value(self):
#         x = eval('self.fn1')
#         return x

# test = test()
# x = test.get_value()
# print(x)



# x = 3
# y = 2
# for i in range(x,y):
#     print('123')
# print('---')

# import random
# x = random.random()
# print(x)

# for i in range(2,-1):
#     print('13')

"""
x = np.zeros((5,5))
y = np.ones((5,5))

up = 1
bot = 1
left = 1
right = 1

center_x = 2
center_y = 2

tmp_up = 2
tmp_bot = 2
tmp_left = 2
tmp_right = 2

up = min(up,tmp_up)
bot = min(bot,tmp_bot)
left = min(left,tmp_left)
right = min(right,tmp_right)

begin_x = center_x - left
end_x = center_x + right
begin_y = center_y - up
end_y = center_y + bot
x[begin_y:end_y+1,begin_x:end_x+1] = y[begin_y:end_y+1,begin_x:end_x+1]
embed()

"""

# x = np.zeros((2,2))
# y = x.copy()
# y[0,0] = 1
# embed()

# import random

# x = random.randint(0,10)
# embed()
# y = random.randint(0,10)
# embed()

# a = []
# b = (2,1,2)
# a.append(b)


# def association(center_joints,offset_maps,weight_maps):
#     """
#     center_joints:这里出来的时候都是整数了
#     offset_maps: (1,36,128,208)
#     weight_maps: (1,36,128,208)
    
#     这里得到的center joint的坐标是基于图像中表示的　（y,x）,先用这种形势，因为到后面还要用这个得到深度值。。

#     return --> single([[jtype,y,x,score],[jtype,y,x,score],[jtype,y,x,score]....])
#                poses (single , single , single .....)
#     """
#     ##这里需不需要先将offset_maps的shape转换一遍？？
#     height = 128
#     width = 208
#     Z = math.sqrt(height ** 2 + width ** 2)
#     center_joints = center_joints.cpu().numpy()
#     offset_maps = offset_maps.squeeze().cpu().numpy()
#     weight_maps = weight_maps.squeeze().cpu().numpy()  
    
#     poses = []  #all-pose
#     for center_joint in center_joints:

#         single_pose = [] #a single complete person's pose
#         root_joint = (2,center_joint[0],center_joint[1]) #firstly,store the root-joint position 
#         single_pose.append(list(root_joint))

#         ch = 0
#         #计算同属于一个人的
#         for i , single_path in enumerate(levels):

#             if i != 1 and i != 2:
#                 x_ = int(center_joint[1])
#                 y_ = int(center_joint[0])
#             if i == 1 or i == 2:
#                 x_ = int(single_pose[2][2]) #这个时候就已经知道了1-th的位置是在2
#                 y_ = int(single_pose[2][1])

#             #1. first is root joint (center_joint here)
#             #2. first is the 1-th joint
#             for j in range(len(single_path)):
                
#                 if j == len(single_path) - 1:
#                     break

#                 current_jtype = single_path[j+1]
#                 if x_ != -1:
#                     x0 = int(max(0,x_ - 2 + 0.5))
#                     y0 = int(max(0,y_ - 2 + 0.5))
#                     x1 = int(min(width, x_ + 2 + 0.5))
#                     y1 = int(min(height, y_ + 2 + 0.5))
#                 else:
#                     x0 = x_
#                     x1 = x_
#                     y0 = y_
#                     y1 = y_

#                 if x0 > width or x1 < 0 or x1 <= x0 or y0 > height or y1 < 0 or y1 <= y0:
#                     if x_ < width and x_ > 0 and y_ < height and y_ > 0 : 
#                         offset_y = offset_maps[ch,y_,x_]
#                         offset_x = offset_maps[ch+1,y_,x_]
#                         score = weight_maps[ch,y_,x_]

#                         current_y = int(y_ + offset_y * Z)
#                         current_x = int(x_ + offset_x * Z)
#                     else:
#                         score = 0.
#                         current_y = -1
#                         current_x = -1

#                     # single_pose.append(list(current_jtype,current_y,current_x,score))
                    
#                 else:
#                     offset_y = np.mean(offset_maps[ch,y0:y1,x0:x1])
#                     offset_x = np.mean(offset_maps[ch+1,y0:y1,x0:x1])
#                     score = np.mean(weight_maps[ch,y0:y1,x0:x1])
                    
#                     current_y = int(y_ + offset_y * Z + 0.5)
#                     current_x = int(x_ + offset_x * Z + 0.5)
                    
#                 single_pose.append(list(current_jtype,current_y,current_x,score))

#                 x_ = current_x
#                 y_ = current_y

#                 ch += 2

#         #变换顺序
#         single_pose = single_pose[np.lexsort(single_pose[:,:].T)]
#         #保存
#         poses.append(single_pose)
#     return poses
# a = [[1,2],[3,4]]
# print(a[-1])


# import numpy as np
# colA = [2,5,1,8,1] # First column
# colB = [9,0,3,2,0] # Second column
# # Sort by ColA and then by colB
# sorted_index = np.lexsort((colB,colA))
# print(sorted_index)
# #print the result showing the
# #column values as pairs
# print ([(colA[i],colB[i]) for i in sorted_index])

# a=np.array([[2.00000000e+00, 7.67812500e+02, 7.66153846e+02, 2.23837173e+02],
#             [0.00000000e+00, 5.11959534e+02, 4.86981384e+02, 4.47674347e+02],
#             [1.00000000e+00, 4.43349121e+02, 4.11920441e+02, 4.40780640e+02],
#             [1.50000000e+01, 4.45317719e+02, 4.13535645e+02, 4.40681305e+02]])
# a=a[np.argsort(a[:,0]),:]
# embed()


# x = []
# sort_index = [0,2,1]
# ori_index = [0,1,2]
# a = [1,2,3]
# b = [4,5,6]
# c = [7,8,9]
# x.append(a)
# x.append(b)
# x.append(c)
# x = np.array(x)
# x[:,ori_index] = x[:,[0,2,1]]
# embed()

# x = np.zeros((2,3))
# y = np.ones((2,3))
# c = x * y
# embed()


# def dist(p1, p2, th):
#     """
#     type: (Seq, Seq, float) -> float
#     3D Point Distance
#     p1:predict point
#     p2:GT point
#     th:the max acceptable distance
#     return:euclidean distance between the positions of the two joints
#     """
#     if p1[0] != p2[0]:
#         return np.nan
#     d = np.linalg.norm(np.array(p1) - np.array(p2))
#     return d if d <= th else np.nan

# d = dist(p1 = [0,2,2], p2 = [0,1,1], th=7)
# embed()

# a = np.array([[1,2],
#               [3,4],
#               [5,6]])
# min = np.nanmin(a,axis=1)
# embed()          

# a = np.array([1,np.nan,2])

# x = len(a[~np.isnan(a)])
# embed()

# def not_nan_count(x):
#     """
#     :return: number of not np.nan elements of the array
#     返回的是一个数
#     """
#     return len(x[~np.isnan(x)])

# mat = np.array([[1,np.nan,3],
#                [np.nan,np.nan,5],
#                [4,6,np.nan]])

# nr = np.apply_along_axis(not_nan_count, 1, mat)   #返回的是每一行中的非nan的个数
# embed()

# a = np.zeros((1,1,128,208))

# z = torch.zeros(size=a.shape, requires_grad=False)
# y = torch.ones(size=a.shape,requires_grad=False)
# embed()

def change_pose(points_pre):
    """
    the function is to change the format to (X,Y,Z) , (X,Y,Z) ....
    """
    predict = []
    gt = []
    if len(points_pre) > 0:
        for pose in points_pre:
            for i in range(19):
                joint = []
                X = pose[0][i]
                Y = pose[1][i]
                Z = pose[2][i]
                joint.append(i)   #joint type
                joint.append(X)
                joint.append(Y)
                joint.append(Z)
                predict.append(joint)
    embed()       
    

x = np.array([ 
        [[-9.5815392e+00,  8.8624474e+01,  5.0636784e+01, -1.1106919e+02,
          8.0532494e+00, -5.4569408e+01,  1.0045500e+02,  1.1990183e+02,
          1.6416911e+02,  1.4151355e+02, -1.3018588e+02, -2.4417947e+02,
         -1.5096150e+02,  6.2668109e+02,  5.3569025e+02, -7.4804831e+00,
          8.5853127e+01, -9.8455620e+01, -2.6916736e+02],
        [ 1.7210104e+02,  4.8530052e+01,  2.4115734e+01,  3.0912695e+02,
          2.3358031e+02,  2.7749176e+02,  2.9982309e+01, -5.1853296e+02,
         -1.2262306e+02, -7.6493309e+01,  4.1650295e+01, -5.5960865e+00,
          2.5728527e+01, -6.4328285e+01,  1.8045760e+02,  1.2447727e-01,
          9.8773621e+01,  4.6609387e+00, -4.9698296e+01],
        [-8.7266510e+01, -2.1869252e+02, -7.7012016e+01,  1.5721777e+02,
          2.9937805e+02,  3.7309583e+02,  2.0874756e+02,  3.6594089e+02,
          1.8247908e+02, -1.3788985e+02,  4.2256058e+01,  1.0266022e+02,
         -1.5429031e+02,  2.8295575e+02,  2.2681512e+02, -1.3803497e+01,
          5.8100166e+01,  5.4536057e+01,  1.0739768e+02]],
        [[-1.4379854e+02,  1.1214600e+03,  6.9944077e+01, -1.8515214e+03,
          1.7018573e+03,  6.8844940e+02,  2.3858167e+03,  3.6402266e+03,
          3.4846670e+03,  5.9527148e+02, -7.4811127e+02, -2.8320061e+03,
         -2.1223079e+03,  8.6241953e+03,  8.3867510e+03,  1.0370318e+02,
          1.0352539e+03, -3.1127544e+03, -5.4584131e+03],
        [ 8.7929944e+02,  1.8031024e+02, -7.0500627e+00,  6.7009272e+03,
          4.6622627e+03,  6.1876064e+03,  2.3016025e+03, -4.6041543e+03,
          3.7327168e+01, -1.2954183e+03,  4.7038574e+02,  3.9271817e+02,
         -6.7859625e+02,  2.0352731e+03,  4.9403101e+03,  1.0409880e+02,
          1.9786310e+03, -1.1758493e+01, -2.0381228e+03],
        [-1.2073367e+02, -6.7090234e+02, -1.0743095e+02,  8.3505750e+02,
          1.5428772e+03,  1.8670879e+03,  8.4751575e+02,  1.3272065e+03,
          5.4678369e+02, -4.1125211e+02,  7.7623260e+01,  3.9345563e+02,
         -4.6711017e+02,  9.8021295e+02,  8.8107922e+02,  6.8454041e+01,
          2.2943546e+02,  4.4998718e+02,  5.6731897e+02]],
        [[  -4063.2266 ,   47346.68   ,     917.95233,  -17883.543  ,
           26094.023  ,  -36148.203  ,   39648.96   ,   80266.164  ,
           25577.383  ,   -3405.5098 ,   54287.19   ,    7787.149  ,
          -91603.61   ,  122606.44   ,  136512.02   ,    -341.12057,
           -3200.5054 ,  -60800.33   , -104030.81   ],
        [  51152.977  ,   20952.326  ,    -575.612  ,   66394.71   ,
           43117.32   ,   81508.71   ,   44931.684  ,  -45381.16   ,
            3233.1658 ,  -39454.18   ,  -54052.14   ,   -1230.8525 ,
          -16980.543  ,    8922.081  ,   65201.26   ,   -2209.857  ,
           -6708.2246 ,    -620.88873,  -28073.29   ],
        [  -1633.691  ,   -3553.4385 ,   -1618.6276 ,    2358.2002 ,
            5456.1504 ,    7141.202  ,    2661.2324 ,    4358.956  ,
             860.6123 ,   -3429.0647 ,   -1792.5665 ,    -343.485  ,
           -3743.5664 ,    2780.69   ,    3047.9072 ,    -360.83398,
            -214.06154,    1507.7402 ,    1941.2638 ]]])



embed()
