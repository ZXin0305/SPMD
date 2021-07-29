import sys
sys.path.append('/home/xuchengjun/Desktop/zx/SPM_Depth')
from IPython.terminal.embed import embed
import torch
from Dataset.CMU import CMU_Dataset
import numpy as np
from utils.loss import cal_loss
from utils.utils import association,to_cam_3d,association_v2
import Config.config as conf
from torch.utils.data import DataLoader
from utils.utils import nms,imread,nms_v2,read_json
from utils.test_metric import change_pose
from torchvision import transforms
import cv2
from model.spmd import SPMD

ori_shape = [1080,1920]
feature_shape = [128,208]
factorx = ori_shape[1] / feature_shape[1]
factory = ori_shape[0] / feature_shape[0]
Z = np.sqrt(feature_shape[0] ** 2 + feature_shape[1] ** 2)
joint_type = {
'0': 'Neck',
'1': 'Nose',
'2': 'BodyCenter',
'3': 'lShoulder',
'4': 'lElbow',
'5': 'lWrist',
'6': 'lHip',
'7': 'lKnee',
'8': 'lAnkle',
'9': 'rShoulder',
'10': 'rElbow',
'11': 'rWrist',
'12': 'rHip',
'13': 'rKnee',
'14': 'rAnkle',
'15': "lEye",
'16': "lEar",
'17': "rEye",
'18': "rEar"
}


cam_ = np.matrix([
		[1633.6,0,943.253],
		[0,1628.87,556.807],
		[0,0,1]
      ])

def test(cnf,val_loader,net):

    val_epoch_len = len(val_loader)
    cen_ch = cnf.cen_ch
    joint_ch = cnf.joint_ch
    offset_ch = cnf.offset_ch
    rr_ch = cnf.rr_ch
    kps_ch = cnf.kps_ch
    loss_c = []
    loss_o = []
    loss_r = []
    loss_kps = []
    net.cuda()
    net.eval()
    for step , data in enumerate(val_loader):
        
        #data中应该包括重投影出来的像素坐标，只在像素坐标中进行验证
        
        img = data[0].cuda()
        center_map = data[1].cuda()
        offset_map = data[2].cuda()
        rr_demap = data[3].cuda()
        offset_map_weight = data[4].cuda()
        #应该在这里加上　with torch.no_grad(): 表示不用进行梯度计算了，不然内存会爆掉
        pre_c,pre_o,pre_r = net(img)
        embed()
        # with torch.no_grad():
        #     pre_c,pre_o,pre_r,pre_kps_weight = net(img)
        #     embed()

        #     #计算loss
        #     loss_1 = cal_center_loss(pre_c,center_map)
        #     loss_2 = cal_other_loss(pre_o,offset_map,offset_ch)
        #     loss_3 = cal_other_loss(pre_r,rr_demap,rr_ch)
        #     loss_4 = cal_other_loss(pre_kps_weight,offset_map_weight,kps_ch,is_kps_weight=True)

        #     loss_c.append(loss_1.data.item())
        #     loss_o.append(loss_2.data.item())
        #     loss_r.append(loss_3.data.item())
        #     loss_kps.append(loss_4.data.item())

        #     print('progress: {}/{} \n Loss: {:0.6f}/{:0.6f}/{:0.6f}/{:0.6f}'.format(
        #             step+1,val_epoch_len,
        #             np.mean(loss_c),np.mean(loss_o),
        #             np.mean(loss_r),np.mean(loss_kps),end=''))
def projectPoints(X, K, R, t, Kd):
        """ Projects points X (3xN) using camera intrinsics K (3x3),
        extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
        
        Roughly, x = K*(R*X + t) + distortion
        
        See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        or cv2.projectPoints
        """
        
        x = np.array(R*X + t) #x.shape --> (3,19) cam-coordinate
        cam_coors = np.array(x)
        
        x[0:2,:] = x[0:2,:]/x[2,:]
        
        r = x[0,:]*x[0,:] + x[1,:]*x[1,:]
        
        x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
        x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])

        x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
        x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2] 
        
        pixel_coors = x  #其实这里的第三行还是在相机坐标系下的

        return cam_coors , pixel_coors

def reproject(anno_file,cali_file,cam_id):
    cam_coors = []
    pixel_coors = []
    skel_with_conf = None
    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'],cam['node']):cam for cam in cali_file['cameras']}
    # Convert data into numpy arrays for convenience --> all
    for k,cam in cameras.items():
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3,1))

    # Select an HD camera (0,0) - (0,30), where the zero in the first index means HD camera 
    cam = cameras[cam_id]  #这个就是序号为:00_00的高清镜头的参数
    # print(cam)
    """
    Reproject 3D Body Keypoint onto the first HD camera
    """

    for body in anno_file['bodies']:
        skel_with_conf = np.array(body['joints19']).reshape((-1,4)).transpose() # (19,4) --> (4,19) 最后一行是置信度
        cam_coor , pixel_coor = projectPoints(skel_with_conf[0:3,:],cam['K'],cam['R'], cam['t'],cam['distCoef'])
        cam_coors.append(cam_coor)
        pixel_coors.append(pixel_coor)
        
    return cam_coors , pixel_coors , skel_with_conf , cam

def process_single_image(net):
    net.to('cpu')
    # embed()
    net.eval()
    img = imread('/media/xuchengjun/datasets/panoptic-toolbox/170407_haggling_a1/hdImgs/00_05/00_05_00003560.jpg')
    img_copy = cv2.imread('/media/xuchengjun/datasets/panoptic-toolbox/170407_haggling_a1/hdImgs/00_05/00_05_00003563.jpg')
    # img = imread('/media/xuchengjun/datasets/panoptic-toolbox/171204_pose1_sample/hdImgs/00_00/00_00_00000020.jpg')
    # img_copy = cv2.imread('/media/xuchengjun/datasets/panoptic-toolbox/171204_pose1_sample/hdImgs/00_00/00_00_00000020.jpg')

    pose_json = read_json(path='/media/xuchengjun/datasets/panoptic-toolbox/170407_haggling_a1/hdPose3d_stage1_coco19/body3DScene_00003560.json')
    cali_json = read_json(path='/media/xuchengjun/datasets/panoptic-toolbox/170407_haggling_a1/calibration_170407_haggling_a1.json')
    lnum = 0
    rnum = 0
    cam_coors_ori , pixel_coors_ori , skel_with_conf_ori , cam = reproject(anno_file=pose_json,cali_file=cali_json,cam_id=(lnum,rnum))
    # print("label -- \n",cam_coors_ori)

    # img_copy = img.copy()
    # img_copy = cv2.resize(img_copy,(1920,1080))
    img = transforms.ToTensor()(img)  #(3,1080,1920)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0)
    # img = img.cuda()
    with torch.no_grad():
        pre_dict = net(img)
        x = nms(pre_dict['center_map'][2][3])
        depth = torch.squeeze(pre_dict['rr_demap'][2][3])
        offset = torch.squeeze(pre_dict['offset_map'][2][3])
        poses = association_v2(x,offset,depth)
        # cam_pose = to_cam_3d(poses, cam_)
        # predict, gt = change_pose(points_pre=cam_pose, points_true=[])
        embed()

        # if len(x) > 0 :
        #     offset_x = []
        #     offset_y = []
        #     for i in range(len(x)):
        #         offset_y.append(offset[0][x[i][:2]] * Z * factory)
        #         offset_x.append(offset[1][x[i][:2]] * Z * factorx)

        # embed()

    print(x)
    # embed()
    for pose in poses:
        for i in range(19):
            cv2.circle(img_copy, center=(int(pose[0][i]+0.5),int(pose[1][i]+0.5)), radius=4, color=(255,0,255))

                # cv2.imshow('result', img_copy)
                # cv2.waitKey(0)
    # for i in range(len(x)):
    #     cv2.circle(img_copy,center=(int(x[i][:2][1]*factorx+offset_x[i]+0.5),int(x[i][:2][0]*factory+offset_y[i]+0.5)),radius=4,color=(0,255,255))
    # for coor in x:
    #     cv2.circle(img_copy, center=(int(coor[1]*factorx+0.5),int(coor[0]*factory+0.5)), radius=4, color=(0,0,255))  #cv中的形式不太一样，顺序是（x,y）
    # # cv2.imwrite('/home/xuchengjun/Desktop/zx/SPM_Depth/results/haggling_a1_00_05_00003563.jpg', img_copy)
    cv2.imshow('result', img_copy)
    cv2.waitKey(0)

if __name__ == '__main__':
    from collections import OrderedDict
    import cv2
    cnf = conf.set_param()
    val_dataset = CMU_Dataset(cnf,mode='val')
    val_loader = DataLoader(dataset=val_dataset,batch_size=1,shuffle=False)
    #load the model
    state_dict = torch.load('ite_0_10000.pth',map_location="cpu") #/home/xuchengjun/ZXin/pth/SPMD/
    # net = Global_Net(cnf)
    net = SPMD(cnf)
    new_state_dict = OrderedDict()
    for k,v in state_dict.items(): #k:键名，v:对应的权值参数
        name = k[7:]
        new_state_dict[name] = v 
    net.load_state_dict(new_state_dict)

    process_single_image(net)
    # test(cnf,val_loader,net)
