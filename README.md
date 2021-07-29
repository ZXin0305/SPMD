# SPMD
运行：
python3 main.py --> train

net.py --> 总的网络，stacked
base_net.py --> 包括最开始的两层网络，还有单独的沙漏块
2021.6.23
先从SPM（2D pose estimation）开始写，直接用CMU数据集

CMU数据集-->label中有19个关节点，（x,y,z） -- 都是在相机坐标系下的
关键点格式是：
3D Body Keypoint Data (coco19 keypoint definition)

0: Neck
1: Nose
2: BodyCenter (center of hips)
3: lShoulder
4: lElbow
5: lWrist,
6: lHip
7: lKnee
8: lAnkle
9: rShoulder
10: rElbow
11: rWrist
12: rHip
13: rKnee
14: rAnkle
15: lEye
16: lEar
17: rEye
18: rEar

CMU数据集的标签都是在世界坐标系下的，虽然有很多不同视角，但是都转换到了世界坐标系。所有在使用的时候应该把不同视角下的数据转换到各自的相机坐标系下，进一步可以转换到像素坐标系中、

coco数据集关键点顺序：
"nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"
————————————————
版权声明：本文为CSDN博主「pan_jinquan」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/guyuealian/article/details/116242320


center_map
这个地方有错误，利用高斯分布？？


2021.7.1　网络设计完成，数据集设计完成，没有写loss
但是网络参数为什么这么大？然后就超出内存了。。。

网络不知道为什么，太大了，只加上一个stage可以
================================================================
Total params: 19,665,576
Trainable params: 19,665,576
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 15.82
Forward/backward pass size (MB): 67607059928286.92
Params size (MB): 75.02
Estimated Total Size (MB): 67607059928377.76
----------------------------------------------------------------


2021.7.01
（1）SPM论文中说的对关节点偏移的(x,y)是存放在center joint(root joint)的附近像素点，由于没有源码，自己在写的时候觉得这样做有点麻烦，(后面进行了改正，都放在了center joint )好处可能是在后处理时读取关节点姿态，如果center joint坐标准确，那么得到其他关节点坐标就可以在每个offset map中对应于center　joint的位置进行读取。
换一种思路，就是将所有的偏移量放在了相对父节点的位置，这样写代码的时候方便(但是在读取的时候可能不太准确)，之后就可以从center joint 出发，依次得到每个人的姿态。（这是种比较直接的方法）
for x in range(x0,x1):
for y in range(y0,y1):
dis = np.sqrt((start_joint[0] - x)**2 + (start_joint[1] - y)**2)
if dis > 6:
continue
else:
x_offset = (end_joint[0] - x) / self.Z
y_offset = (end_joint[1] - y) / self.Z
 
self.offset_map[ch_index,y,x] += y_offset
self.offset_map[ch_index+1,y,x] += x_offset
self.offset_map_weight[ch_index:ch_index+2,y,x] = 1 #当在推理的时候，可以通过这个进行判断这个关节点是否可见 ,最后看看效果
#center周围点(x,y)不和关节点重合，则人数增加一个
if end_joint[1] != y or end_joint[0] != x:
self.kps_count[ch_index:ch_index+2,y,x] += 1
 
（2）想到的是直接利用SPM + Depth map（root relative depth map）,3D 分支设计的比较简单
def create_3D_label(self,sigmax = 6,sigmay = 6):
"""
相对父节点深度图
在转换数据集便签格式的时候应该换成：3D --> (3,N) --> (X,Y,Z)
"""
 
for idx , pcoor in enumerate(self.pcoors): #遍历所有的人
# all_rel_depth = []
all_joint_depth = pcoor[2,:] #第三行所有
# root_depth = all_joint_depth[2] #第2个为root joint
# all_rel_depth = (all_joint_depth - root_depth) / self.Z #进行normalize 这个都是相对根深度的值
#这里是为了制作root joint的depth map
for center in self.centers:
x0 = int(max(0,center[0] - sigmax + 0.5))
x1 = int(min(self.width,center[0] + sigmax +0.5))
y0 = int(max(0,center[1] - sigmay + 0.5))
y1 = int(min(self.height, center[1] + sigmay + 0.5))
for x in range(x0,x1):
for y in range(y0,y1):
dis = np.sqrt((center[0] - x) ** 2 + (center[1] - y) ** 2)
if dis > 6 or self.rel_demap[0,y,x] != 0: #后面判断条件是如果前面一个人已经占据了，下一个人的值就不准覆盖，设定最后推理的范围为3
continue
self.rel_demap[0,y,x] = all_joint_depth[2] #针对root joint的id顺序
ch_index = 1
for single_path in self.level:
# tmp_rel_dep = []
for i in range(len(single_path) - 1):
# if i == (len(single_path) - 1):
# break
start_id = single_path[i]
end_id = single_path[i+1]
child_coor = pcoor[:2,end_id]
"""
这里的问题：因为有的点是超出了范围，因此，0 和　1的范围可能会不一样，但是在训练的时候这个点不会出现就不会计算loss
"""
x0 = int(max(0,child_coor[0] - sigmax + 0.5))
x1 = int(min(self.width,child_coor[0] + sigmax +0.5))
y0 = int(max(0,child_coor[1] - sigmay + 0.5))
y1 = int(min(self.height, child_coor[1] + sigmay + 0.5))

start_dep = all_joint_depth[start_id]
end_dep = all_joint_depth[end_id]
rel_dep = (end_dep - start_dep) / self.Z #normalize

for x in range(x0,x1):
for y in range(y0,y1):
dis = np.sqrt((child_coor[0] - x) ** 2 + (child_coor[1] - y) ** 2)
if dis > 6 or self.rel_demap[ch_index,y,x] != 0:
continue
self.rel_demap[ch_index,y,x] = rel_dep
ch_index += 1 

return self.rel_demap
 
（3）参考了论文SMAP中的网络，写网络
SMAP中的网络主要如下：
图片: https://uploader.shimo.im/f/ZO1ahNNy82dLSZns.jpg?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJhY2Nlc3NfcmVzb3VyY2UiLCJleHAiOjE2Mjc1MjUxMDgsImciOiJ3SkNSVHdRV0doamtHRHFQIiwiaWF0IjoxNjI3NTI0ODA4LCJ1c2VySWQiOjY2MTMyMDYxfQ.tsERqCQsSOOgPERnrleEp7uXEszTbrK8soyVXXbrdvw

本身是一个沙漏网络的形式，增加了一些residule的分支，写好一个block之后，可以进行叠加操作，SMAP中设置的是3
 
对里面的某些层进行了参数改动，在下采样阶段中的每个小block中增加了res分支：
self.downsample = downsample_module(BottleNet,self.layers)
 
class downsample_module(nn.Module):
def __init__(self,block,layers):
super().__init__()
self.k = 3
self.s = 2
self.p = 0
# self.block = block
self.layer_1 = self.make_layer(block,32,layers[0])
self.layer_2 = self.make_layer(block,64,layers[1])
self.layer_3 = self.make_layer(block,128,layers[2])
self.layer_4 = self.make_layer(block,256,layers[3])
self.downsample_1 = conv_bn_relu(32,64,kernel_size=self.k,stride=self.s,padding=self.p,has_bn=True,has_relu=False)
self.downsample_2 = conv_bn_relu(64,128,kernel_size=self.k,stride=self.s,padding=self.p,has_bn=True,has_relu=True)
self.downsample_3 = conv_bn_relu(128,256,kernel_size=self.k,stride=self.s,padding=self.p,has_bn=True,has_relu=False)
self.downsample_4 = conv_bn_relu(256,512,kernel_size=self.k,stride=self.s,padding=self.p,has_bn=True,has_relu=False)
#权值初始化
for m in self.modules():
if isinstance(m,nn.Conv2d):
nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
elif isinstance(m,nn.BatchNorm2d):
nn.init.constant_(m.weight,1)
nn.init.constant_(m.bias,0)
 
def make_layer(self,block,in_ch,block_num):
layers = []
#首先设置一层将前面的通道数进行扩增,但是尺寸不变
out_ch = in_ch*BottleNet.expansion
if block_num != 1:
layers.append(block(in_ch,out_ch,first_block = True))
#如果只有一层，那么就到这一步
if block_num == 1:
layers.append(block(in_ch,out_ch,first_block = True))
layers.append(conv_bn_relu(out_ch,out_ch,kernel_size=3,stride=2,padding=0,has_bn=True,has_relu=True))
#进入主层,这个时候设置　in_ch　= out_ch
in_ch = out_ch 
for i in range(1,block_num):
if i != block_num - 1:
layers.append(block(in_ch,out_ch))
else:
layers.append(block(in_ch,out_ch,reduce_size = True))
return nn.Sequential(*layers)
def forward(self,x):
#layer_1
downsample_1 = self.downsample_1(x)
x1 = self.layer_1(x)
x1 += downsample_1
x1 = F.relu(x1)
#layer_2
downsample_2 = self.downsample_2(x1)
x2 = self.layer_2(x1)
x2 += downsample_2
x2 = F.relu(x2)
#layer_3
downsample_3 = self.downsample_3(x2)
x3 = self.layer_3(x2)
x3 += downsample_3
x3 = F.relu(x3)
#layer_4
downsample_4 = self.downsample_4(x3)
x4 = self.layer_4(x3)
x4 += downsample_4
x4 = F.relu(x4)
return x1,x2,x3,x4
 
上采样阶段是四个上采样阶段都会输出采样值，并返回，作为自监督的量：
class upsample_module(nn.Module):
def __init__(self,up_block,out_shape,ch_tuple):
"""
上采样模块 --> nn.ConvTransposed2d(in ,out,k,s,p,out_padding)
"""
super().__init__()
# self.up_block = up_block
self.in_ch = [512,256,128,64]
self.up_size = [(15,29),(32,59),(66,119),(134,239),(270,480)]
self.up1 = up_block(0,self.in_ch[0],self.up_size[0],out_shape,ch_tuple)
self.up2 = up_block(1,self.in_ch[1],self.up_size[1],out_shape,ch_tuple)
self.up3 = up_block(2,self.in_ch[2],self.up_size[2],out_shape,ch_tuple)
self.up4 = up_block(3,self.in_ch[3],self.up_size[3],out_shape,ch_tuple)
def forward(self,x4,x3,x2,x1):
out1,res_c1,res_o1,res_r1 = self.up1(x4,None) # (15,29)
out2,res_c2,res_o2,res_r2 = self.up2(x3,out1)
out3,res_c3,res_o3,res_r3 = self.up3(x2,out2)
out4,res_c4,res_o4,res_r4 = self.up4(x1,out3)
#没有joint heatmap
res_c = [res_c1,res_c2,res_c3,res_c4]
res_o = [res_o1,res_o2,res_o3,res_o4]
res_r = [res_r1,res_r2,res_r3,res_r4]
# return out4,res_c,res_j,res_o,res_r
return out4,res_c,res_o,res_r
 
最后在总的网络结构中，可以任意进行叠加：
#stage_1
stage_out1,res_c1,res_o1,res_r1 = self.stage_1(out) #return : a list
#stage_2
stage_out2,res_c2,res_o2,res_r2 = self.stage_2(stage_out1)
#stage_3
stage_out3,res_c3,res_o3,res_r3 = self.stage_3(stage_out2)
 
但是在测试的时候，发现显卡内存中总是满了，如果叠加3个的话，后来把每个网络块中输出的最大通道数从1024变到512,然后在bs为2的情况下，显卡内存能达到90％。。。。
（后面发现是自己直接原图像进行处理，这样计算量太大了，后面使用SMAP的网络的时候，就先resize成了(832,512)）

2021.07.04
昨天的时候尝试训练，当时发现loss非常大，试着去调整每个部分的loss的权值，还是非常大。
又将loss的计算方式进行了改变：之前是每个沙漏块上采样的部分的输出之间进行自监督的计算，只用每个块的最后一个输出和gt计算loss。后面发现没有必要，直接都与gt计算就可以了。并且只采用了最后两个块的输出。
但是这样做了loss还是非常大。。最后查看每个loss部分的具体值，发现offset map处的loss非常大，原因是在制作根节点深度图时并没有对数值进行规范化，即除以Z值，导致计算出来的loss非常大。规范化之后，loss变小了，并且开始的下降速度也提高了。
今天重新看了一下SPM的训练步骤，发现其中的一些步骤不太一样。。希望这样的可以训练的好一点吧。
目前的val和post-processing的部分还没有写完，只是先训练看看效果。

为什么在训练的时候程序会自己kill掉？
因为在训练的时候为了打印loss,就设置了一个list来保存每一个iteration的loss，
但是这个位置错了，不应该放在loss.backward()的前面，应该放在后面，不然pytorch会认为这个也是计算图中的一部分，会一块带着进行反向传播，这样会使计算图不断扩展，最终会内存泄漏。。。
在保存每个iteration的loss的时候，应将参数从torch中取出来，即写成：loss.item()
可以参考LoCO-master那个代码中的写法


2021.07.05
今天测试了训练的部分的代码（现在只应用了170407_haggling_a1中的'00_00','00_01','00_03'，测试的只用了'00_05'） 数据之后要进行扩充

非极大值抑制和分组部分写好了但是还没有测试 　－－－－－－－－－－
nms是针对center joint map的
当大于附近四个点并且自身的置信度大于阈值（这个阈值设置成0.75还行）时，才确定为center joint
注意：此时返回的center是（y,x）形式的

分组部分是按照level的形式进行的
返回的格式是          [  jtype  ,  y  ,  x  ,  score]
注意：这里的每个点的位置顺序依然是(y,x)形式的，且都没有乘以  factor !
子节点的2d位置是：(y,x)_child = (y,x)_farent + displacement
方法是在父节点周围一定区域中，计算平均偏移值
如果在某一条通路中，某一个点的坐标不正常，那么就让这个点的坐标值都赋为０
这个时候就说明必须要设计3d的refine-net了。。。

2021.7.18在分组这一块进行了改正：
因为是将offset放在了center joint的附近，所以就直接在center joint的周围进行读取就好了
返回的形式是
poses:[pose1,pose2,...poseN]
pose1,pose2,...,poseN --> [  [x1,x2,...,xj],
       [y1,y2,....,yj],
                                                         [Z1,Z2,....,Zj]]

将得到的点（含有-1值）都从像素平面到相机平面进行投影，2d位置不正常的点同样设置为-1，输入进修正网络进行修正。。。。

先不对这些错误点进行设置，在refine的时候进行就好了

2021.07.06
SMAP中的refine net 阶段是用前面自己训练的3D姿态识别器将人体的关节点数据保存成数据集
输入是（2d + 3d）* 15（关节点个数），输出是（3d）*  15(关节点个数)
现在先将前一阶段的识别器训练出来看看效果。。

SPM_pytorch中分组的方法是只关注每个具体像素点处点的位移，我想的是关注附近16个像素点的位移
2021.07.08
今天用了第一个epoch迭代5000和10000次的结果，但是为什么输出的center map的值总是为0？？？

现在改了loss，每一个stage中有四个上采样块，每个块都会输出四种特征图，都会和gt计算loss
但是center map的loss是根据gt中center joint 存在的位置和对应的预测出来的图中的位置做绝对差，然后除以总得有值的个数（这个时候没有利用mask）

不知道为什么center map为0，而且其他的图也不太对。。

2021.07.11
暑期已至，不知道你的创新点能不能够想得到。。
改变了center_map的loss的计算方式，本来是找到gt-center-map中不为0的点，就计算这些点的loss并除以不为零的点的个数，但是这样的毛病就是计算量太大了，在进行计算的时候，训练的速度非常慢。
后面改变了loss的计算方式，生成一个关于center_map的mask--标志着哪里存在着center joint，因为center-map是高斯图，因此在计算loss之前，将mask与预测出来的center-map相乘，只计算存在center-map的地方的loss,这样的计算量稍微小一点，不过在加大数据之后，训练时间非常长，十一个小时一个epoch


2021.07.13
改动：
对于offset_map \ center_map \ root relative map的loss计算都加上了mask..
每一层的feature map的大小
每个stage只在最后一个block才会有其他的输出
读取图片的时候直接resize成（２７０，４８０），并且在input_net时经过了空洞卷积，进行特征提取　-- > 　（135,240）
中间监督的size变化了

这个应该是弄不出来了，效果很差。。就这样把，想一下自己下一步要干什么了。。太难了叭。。这几天可以看看别人的好的demo,运行运行，好的就用


2021.07.19
改动：
１）：网络使用SMAP的，不过在每一个stage的输入时，加上了空洞卷积之后的特征图
２）：SMAP网络每一个stage在后面的上采样中的四层都有输出
３）：今天把offset_map和depth_map的值都放到了每一个center joint的附近
４）：loss的计算方式没有什么改变，不过下降的好慢阿
下一步把数据集都下载完成，从每个数据集中挑选图片进行训练

2021.07.28
前面的网络训练到epoch 4的时候，对于center map可以在一定程度上找到center joint，但是对于offset map的效果不太好
center map的loss计算是分batch_size \ feature_chs,依次进行计算的
而offset map 和 depth map则是根据mask，但是是除以总的元素的个数

之前的效果：
图片: https://uploader.shimo.im/f/11ecvnZwYmaFtOvD.jpg?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJhY2Nlc3NfcmVzb3VyY2UiLCJleHAiOjE2Mjc1MjUxMDgsImciOiJ3SkNSVHdRV0doamtHRHFQIiwiaWF0IjoxNjI3NTI0ODA4LCJ1c2VySWQiOjY2MTMyMDYxfQ.tsERqCQsSOOgPERnrleEp7uXEszTbrK8soyVXXbrdvw
图片: https://uploader.shimo.im/f/oAR9QvHqb4KtPJ72.jpg?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJhY2Nlc3NfcmVzb3VyY2UiLCJleHAiOjE2Mjc1MjUxMDgsImciOiJ3SkNSVHdRV0doamtHRHFQIiwiaWF0IjoxNjI3NTI0ODA4LCJ1c2VySWQiOjY2MTMyMDYxfQ.tsERqCQsSOOgPERnrleEp7uXEszTbrK8soyVXXbrdvw
图片: https://uploader.shimo.im/f/EAZhRM8VeKjh5sT0.jpg?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJhY2Nlc3NfcmVzb3VyY2UiLCJleHAiOjE2Mjc1MjUxMDgsImciOiJ3SkNSVHdRV0doamtHRHFQIiwiaWF0IjoxNjI3NTI0ODA4LCJ1c2VySWQiOjY2MTMyMDYxfQ.tsERqCQsSOOgPERnrleEp7uXEszTbrK8soyVXXbrdvw

这几张看着问题不大，只是可视化了前面的center joint和neck,后面可视化所有的点的时候，效果就不太好了。。

目前把转换到3D camera空间的函数写出来了
association得到的输出是按照：
poses:[pose1,pose2,...poseN]
pose1,pose2,...,poseN --> [  [x1,x2,...,xj],
       [y1,y2,....,yj],
       [Z1,Z2,....,Zj] ]
注意：这里前两行都是像素坐标，Z是在相机坐标系下面的。。。
后面要进行转换公式

2021.7.29
写了val，不过是在每一个epoch之后进行，就是要测试每一个epoch的准确度，把最好的保存下来(现在只使用了feature map的第3个stage的第4个block的输出，索引是[2][3])
val是根据LoCO-master写的，评价指标包括：recall,precision,f1-score
import numpy as np
from IPython import embed

def change_pose(points_pre, points_true):
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
                
    if len(points_true) > 0:
        for pose in points_true:
            for i in range(19):
                joint = []
                X = pose[0][i]
                Y = pose[1][i]
                Z = pose[2][i]
                joint.append(i)
                joint.append(X)
                joint.append(Y)
                joint.append(Z)
                gt.append(joint)

    return predict , gt

def dist(p1, p2, th):
    """
    type: (Seq, Seq, float) -> float
    3D Point Distance
    p1:predict point
    p2:GT point
    th:the max acceptable distance
    return:euclidean distance between the positions of the two joints
    """
    if p1[0] != p2[0]:
        return np.nan
    d = np.linalg.norm(np.array(p1[1:]) - np.array(p2[1:]))
    return d if d <= th else np.nan

def non_minima_suppression(x):
    """
    return:non-minima suppressed version of the input array
    supressed values become np.nan
    """
    min = np.nanmin(x)
    x[x != min] = np.nan
    if len(x[x == min]) > 1:
        ok = True
        for i in range(len(x)):
            if x[i] == min and ok:
                ok = False
            else:
                x[i] = np.nan
    return x

def not_nan_count(x):
    """
    :return: number of not np.nan elements of the array
    返回的是一个数
    """
    return len(x[~np.isnan(x)])


def joint_det_metrics(points_pre, points_true, th=7.0):
    """
    points_pre : the predict poses in camera coordinate
    points_true: the gt-truth poses in camera coordinate
    th:distance threshold; all distances > th will be considered 'np.nan'.
    return :  a dictionary of metrics, 'met', related to joint detection;
              the the available metrics are:
              (1) met['tp'] = number of True Positives
              (2) met['fn'] = number of False Negatives
              (3) met['fp'] = number of False Positives
              (4) met['pr'] = PRecision
              (5) met['re'] = REcall
              (6) met['f1'] = F1-score
    """
    predict, gt = change_pose(points_pre=points_pre, points_true=points_true)
    if len(predict) > 0 and len(gt) > 0:
        mat = []
        for p_true in gt:
            row = np.array([dist(p_pred, p_true, th=th) for p_pred in predict])
            mat.append(row)
        mat = np.array(mat)
        mat = np.apply_along_axis(non_minima_suppression, 1, mat)
        mat = np.apply_along_axis(non_minima_suppression, 0, mat)

        # calculate joint detection metrics
        nr = np.apply_along_axis(not_nan_count, 1, mat)
        tp = len(nr[nr != 0])   #number of true positives
        fn = len(nr[nr == 0])   #number of false negatives
        fp = len(predict) - tp
        pr = tp / (tp+fp)
        re = tp / (tp+fn)
        f1 = (2 * pr * re) / (pr + re)

    elif len(predict) == 0 and len(gt) == 0:
        tp = 0    #number of true positives
        fn = 0    #number of false negatives
        fp = 0    #number of false positive
        pr = 1.0
        re = 1.0
        f1 = 1.0
    elif len(predict) == 0:
        tp = 0
        fn = len(gt)
        fp = 0
        pr = 0.0
        re = 0.0
        f1 = 0.0
    else:
        tp = 0
        fn = 0
        fp = len(predict)
        pr = 0.0
        re = 0.0
        f1 = 0.0

    metrics = {
        'tp':tp, 'fn':fn, 'fp':fp,
        'pr':pr, 're':re, 'f1':f1,
    }

    return metrics

