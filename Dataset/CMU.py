import os
import sys
from IPython.core.magic_arguments import real_name
sys.path.append('/home/xuchengjun/Desktop/zx/SPM_Depth')
import numpy as np
import Dataset.pose
import torch
from torch.utils import data
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
from path import Path
from IPython import embed
from Config import config
from utils import utils
from Dataset.spm_v2 import SingleStageLabel
import matplotlib.pyplot as plt
import cv2


useful_train_dirs = ['170407_haggling_a1','160422_ultimatum1'] # '170221_haggling_b1',
useful_val_dirs = ['160422_ultimatum1'] 
useful_img_dirs_train = ['00_00','00_01','00_02','00_03','00_04','00_05']
useful_img_dirs_val = ['00_16','00_30']
body_edges = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1

class CMU_Dataset(Dataset):

    def __init__(self,cnf,mode='train') -> None:   #cnf --> configuration
        super().__init__()

        self.cnf = cnf
        self.data_path = Path(cnf.data_path)
        self.sub_dirs = self.data_path.dirs()  #得到所有根路径下的子文件夹
        self.mode = mode

        if self.mode == 'train':
            self.data_list = self.get_train_data()
        elif self.mode == 'val':
            self.data_list = self.get_val_data()
    
    def get_train_data(self):
        data_list = []
        for sub_dir in self.sub_dirs:
            if sub_dir.basename() in useful_train_dirs:
                # print(f'{sub_dir.basename()}')
                img_dir_path = sub_dir / 'hdImgs'
                annotation_dir_path = sub_dir / 'hdPose3d_stage1_coco19'

                img_dirs = img_dir_path.dirs()
                annotation_files = annotation_dir_path.files()  #annotations 这里没有文件夹，是所有的

                for img_dir in img_dirs:
                    if img_dir.basename() in useful_img_dirs_train:
                        # imgs = img_dir.files()  #len(imgs) == 16716 所有的数据集
                        cali_file_path = sub_dir / ('calibration_' + sub_dir.basename() + '.json') #标定文件的路径
                        for idx in range(len(annotation_files)):
                            basename = annotation_files[idx].basename()
                            if basename.endswith('.json'):   #读取的时候有错误。。
                                anno_num = basename.split('.')[0].split('_')[1]  #只要这个标签的文件数值就好
                                img_path = img_dir / (img_dir.basename() +  '_' + anno_num + '.jpg')
                                data_list.append((img_path,annotation_files[idx],cali_file_path,img_dir.basename())) #img_dir.basename()　--> 主要是为了得到对应的相机参数


        return data_list
        
    def get_val_data(self):
        data_list = []
        for sub_dir in self.sub_dirs:
            if sub_dir.basename() in useful_val_dirs:
                # print(f'{sub_dir.basename()}')
                img_dir_path = sub_dir / 'hdImgs'
                annotation_dir_path = sub_dir / 'hdPose3d_stage1_coco19'

                img_dirs = img_dir_path.dirs()
                annotation_files = annotation_dir_path.files()  #annotations 这里没有文件夹，是所有的

                for img_dir in img_dirs:
                    if img_dir.basename() in useful_img_dirs_val:
                        # imgs = img_dir.files()  #len(imgs) == 16716 所有的数据集
                        cali_file_path = sub_dir / ('calibration_' + sub_dir.basename() + '.json') #标定文件的路径
                        for idx in range(len(annotation_files)):
                            basename = annotation_files[idx].basename()
                            if basename.endswith('.json'):   #读取的时候有错误。。
                                anno_num = basename.split('.')[0].split('_')[1]  #只要这个标签的文件数值就好
                                img_path = img_dir / (img_dir.basename() +  '_' + anno_num + '.jpg')

                                data_list.append((img_path,annotation_files[idx],cali_file_path,img_dir.basename())) #img_dir.basename()　--> 主要是为了得到对应的相机参数
        return data_list[0:4000]


    def __len__(self):
        # print(len(self.data_list))
        return len(self.data_list)

    
    def __getitem__(self, idx):
        #读取图片
        img_path , anno_path = self.data_list[idx][0] , self.data_list[idx][1]
        cali_path , cam_id = self.data_list[idx][2] , self.data_list[idx][3]
        img = utils.imread(img_path)       #(1080,1920,3) 图像中像素点是以(y,x)定位的
        img = transforms.ToTensor()(img)   #(3,1080,1920)
        img = transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])(img)
        # orih , oriw = img.shape[1:3] #图像的大小

        #读取json文件
        anno_file = utils.read_json(anno_path)
        cali_file = utils.read_json(cali_path)
        #先得到相机的序号参数
        cam_id = str(cam_id)
        lnum , rnum = int(cam_id.split('_')[0]) , int(cam_id.split('_')[1])
        cam_coors_ori , pixel_coors_ori , skel_with_conf_ori , cam = self.reproject(anno_file,cali_file,(lnum,rnum))  #将世界坐标系下的坐标转换到对应相机视角下的相机和像素坐标系中,dedao 

        # poses = pose.select_points(poses) #选择使用的点
        #制作label
        ori_shape = (1080,1920)
        out_shape = (self.cnf.outh,self.cnf.outw)
        centers = utils.prepare_centers(ori_shape = ori_shape,
                                        out_shape = out_shape,
                                        coors = pixel_coors_ori)  #(x,y)
        person_num = len(centers)


        cam_coors , pixel_coors = utils.prepare_keypoints(ori_shape = ori_shape,
                                                         out_shape = out_shape,
                                                         cam_coors = cam_coors_ori,
                                                         pixel_coors = pixel_coors_ori,
                                                         dataset=self.cnf.data_format) #(X,Y,Z) (x,y,Z)
        ssl = SingleStageLabel(self.cnf.outh,
                               self.cnf.outw,
                               centers,
                               pixel_coors,
                               cam_coors,
                               self.cnf)

        center_map , center_mask , offset_map = ssl.create_2D_label()
        # print(img_path)
        # self.show_center_map(center_map)
        # self.show_center_map(offset_map[2])
        rr_demap = ssl.create_3D_label()  # --> root_relative_depth_map 第一个channel是root joint的depth map

        if self.mode == 'train':
            return {'img':img,
                    'center_map':torch.Tensor(center_map).unsqueeze(0),
                    'offset_map':torch.Tensor(offset_map),
                    'rr_demap':torch.Tensor(rr_demap),
                    'mask':torch.Tensor(center_mask).unsqueeze(0)
            }
        if self.mode == 'val':
            # return img,torch.Tensor(center_map).unsqueeze(0),torch.Tensor(offset_map) , torch.Tensor(rr_demap),\
            #            torch.Tensor(offset_map_weight)
            return {
                "img":img,
                "cam_coors":torch.Tensor(cam_coors_ori),
                "cam_info":torch.Tensor(cam),
            }  # (3,19)

    """
    
    """
    def reproject(self,anno_file,cali_file,cam_id):
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
        """
        Reproject 3D Body Keypoint onto the first HD camera
        """

        for body in anno_file['bodies']:
            skel_with_conf = np.array(body['joints19']).reshape((-1,4)).transpose() # (19,4) --> (4,19) 最后一行是置信度
            cam_coor , pixel_coor = self.projectPoints(skel_with_conf[0:3,:],cam['K'],cam['R'], cam['t'],cam['distCoef'])
            cam_coors.append(cam_coor)
            pixel_coors.append(pixel_coor)
            
        return cam_coors , pixel_coors , skel_with_conf , cam['K']


    def projectPoints(self,X, K, R, t, Kd):
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

    def show_center_map(self,center_map):
        center_map = center_map * 255
        plt.subplot(111)
        plt.imshow(center_map)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    cnf = config.set_param()
    cmu = CMU_Dataset(cnf,mode='val')

    for i in range(len(cmu)):
        dict_ = cmu[i]
        embed()



        
        



        