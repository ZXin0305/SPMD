import os
import enum
from re import S
import torch
from path import Path
import torchvision
from Config import config
from IPython import embed
from model.spmd import SPMD
# from model.net import Global_Net
import torch.optim as optim
from Dataset.CMU import CMU_Dataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import numpy as np
from utils.loss import cal_loss
from torch.autograd import Variable
# from model.base_net import Single_hourglass_block,input_net
from datetime import datetime
from time import time
from utils.utils import nms,association,association_v2,sum_features,to_cam_3d
from utils.test_metric import joint_det_metrics


# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
# device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self,cnf) -> None:
        self.model_path = Path(cnf.model_path)
        self.device = cnf.device

        self.cen_ch = cnf.cen_ch
        # self.joint_ch = cnf.joint_ch
        self.offset_ch = cnf.offset_ch
        self.rr_ch = cnf.rr_ch
        self.kps_ch = cnf.kps_ch

        #init some values
        self.epoch = 0
        self.end_epoch = cnf.epoch
        self.bs = cnf.batch_size
        self.lr = cnf.lr
        self.best_test_f1 = None

        #build the net
        # self.net = Global_Net(cnf)
        self.net = SPMD(cnf)  #using SMAP's net
        self.net.cuda()  #先将模型load到第一块显卡
        self.net = torch.nn.DataParallel(self.net,device_ids = [0,1]) #将模型复制到多卡
        
        #the optimizer
        self.optimizer = optim.AdamW(params=self.net.parameters(),lr=self.lr,weight_decay=1e-2)
        # self.optimizer = optim.Adam(params=self.net.parameters(),lr=self.lr) #,betas=(0.9,0.999),eps=1e-08,weight_decay=8e-6
        # self.optimizer = optim.Adagrad(params=self.net.parameters(),lr=self.lr)
        #学习率调整器
        self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=5,gamma=0.88)

        #init dataset
        #CMU
        train_set = CMU_Dataset(cnf,mode='train')
        val_set = CMU_Dataset(cnf,mode='val')

        #init train/val loader
        self.train_loader = DataLoader(train_set,self.bs,shuffle=True,num_workers=cnf.num_workers)
        self.val_loader = DataLoader(val_set,batch_size=1,shuffle=False)
        self.epoch_len = len(self.train_loader)
        self.val_epoch_len = len(self.val_loader) 
        # embed()

        #init logging stuffs
        self.log_dir = Path(cnf.log_dir)
        self.sw = SummaryWriter(self.log_dir)

        #possibly load checkpoint
        self.load_ck()

    def load_ck(self):
        """
        loading training checkpoint
        """
        ck_path = self.log_dir / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path,map_location=torch.device('cpu'))
            print(f'[loading checkpoint --> {ck_path}')
            self.epoch = ck['epoch'] #last time's epoch
            self.net.load_state_dict(ck['model'])
            self.optimizer.load_state_dict(ck['optimizer'])
            self.scheduler.load_state_dict(ck['scheduler'])

    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            'epoch':self.epoch,
            'model':self.net.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        torch.save(ck,self.log_dir / 'training.ck')

    # def adjust_lr(self,last_lr):
    #     """
    #     设置每多少个epoch学习率除以多少
    #     """
    #     lr = last_lr * 0.1
    #     return lr

    def train(self):
        """
        start to train the model
        """
        self.net.train()
        train_loss = []
        start_time = time()

        for step , data in enumerate(self.train_loader):

            self.optimizer.zero_grad()

            img = data['img'].cuda()
            # center_map = data['center_map'].cuda()
            # offset_map = data['offset_map'].cuda()
            # rr_demap = data['rr_demap'].cuda()
            # offset_map_weight = data['offset_map_weight'].cuda()

            gt_dict = {
                'center_map':data['center_map'].cuda(),
                'offset_map':data['offset_map'].cuda(),
                'rr_demap':data['rr_demap'].cuda(),
                'mask':data['mask']
            }


            #predict the results through net
            # res_c,res_o,res_r = self.net(img)

            # pre_dict = {
            #     'center_map':res_c,
            #     'offset_map':res_o,
            #     'rr_demap':res_r
            # }


            #using SMAP's net , return a dict
            pre_dict = self.net(img) 
            total_loss = cal_loss(pre_dict,gt_dict)

            total_loss.backward()
            self.optimizer.step()
            train_loss.append(total_loss.data.item())  #.data可以获得该节点的值，Tensor类型,.item() --> 将torch中的值取出来

            #输出当前的进程
            print('\r[{}] Epoch: {} progerss: {} / {} Loss: {:0.8f}'.format(datetime.now().strftime("%m-%d@%H:%M"),
                    self.epoch,step+1,self.epoch_len,
                    torch.mean(torch.tensor(train_loss))),end='')
            
            if step % 5000 == 0 and step != 0:
                torch.save(self.net.state_dict(),self.model_path / f'ite_{self.epoch}_{step}.pth')

            if step >= self.epoch_len - 1:
                break
        
        # the epoch loss
        mean_epoch_loss = np.mean(train_loss)
        self.sw.add_scalar(tag='train/loss',scalar_value=mean_epoch_loss,global_step=self.epoch)
        print(f'\nTime: {time() - start_time:.2f} \n')
    
    def test(self):
        """
        start to val the model
        """
        self.net.eval()
        # self.net.requires_grad(False)
        t = time()
        test_prs = []   #precision
        test_res = []   #recall
        test_f1s = []   #f1-score

        for step, data in enumerate(self.val_loader):

            with torch.no_grad():
                img = data['img']
                gt_cam_coors = data['cam_coors'].squeeze(0)
                cam_info = data['cam_info']

                pre_dict_val = self.net(img)
                # center_map = sum_features(pre_dict_val['center_map'])  #返回的是sum之后的[2][3]
                # depth_map = sum_features(pre_dict_val['rr_demap'])
                # offset_map = sum_features(pre_dict_val['offset_map'])

                center_map = pre_dict_val['center_map'][2][3]
                depth_map = pre_dict_val['rr_demap'][2][3]
                offset_map = pre_dict_val['offset_map'][2][3]

                centers = nms(center_map)
                depth = torch.squeeze(depth_map)
                offset = torch.squeeze(offset_map)
                poses = association_v2(center_joints=centers, offset_maps=offset, depth_maps=depth)
                poses = np.array(poses)
                cam_info = np.matrix(cam_info[0])
                cam_pose = to_cam_3d(poses, cam_info)
                
                #calculate the metric
                metric_dict = joint_det_metrics(points_pre=cam_pose, points_true=np.array(gt_cam_coors), th=7.0)
                pr, re, f1 = metric_dict['pr'], metric_dict['re'], metric_dict['f1']
                # print(f'pr:{pr} \t re:{re} \t f1:{f1}',end='')
                test_prs.append(pr)
                test_res.append(re)
                test_f1s.append(f1)

                print('\r[{}] Epoch: {} progerss(val): {} / {} pr:{} re:{} f1:{}'.format(datetime.now().strftime("%m-%d@%H:%M"),
                    self.epoch,step+1,self.val_epoch_len,pr,re,f1,end=''))

        mean_test_pr = float(np.mean(test_prs))
        mean_test_re = float(np.mean(test_res))
        mean_test_f1 = float(np.mean(test_f1s))

        print(
            f'\t● AVG (PR, RE, F1) on TEST-set: '
            f'({mean_test_pr * 100:.2f}, '
            f'{mean_test_re * 100:.2f}, '
            f'{mean_test_f1 * 100:.2f}) ',
            end=''
        )
        print(f'│ Time: {time() - t:.2f} s')

        self.sw.add_scalar(tag='val/precision',scalar_value=mean_test_pr,global_step=self.epoch)
        self.sw.add_scalar(tag='val/recall',scalar_value=mean_test_re,global_step=self.epoch)
        self.sw.add_scalar(tag='val/f1-score',scalar_value=mean_test_f1,global_step=self.epoch)


        #save the best model
        if self.best_test_f1 is None or mean_test_f1 >= self.best_test_f1:
            self.best_test_f1 = mean_test_f1
            torch.save(self.net.state_dict(),self.model_path / f'epoch_{self.epoch}.pth')

        
    def run(self):
        for i in range(self.epoch,self.end_epoch):
            self.train()
            self.scheduler.step()
            self.test()
            self.epoch += 1
            self.save_ck()
            print()


if __name__ == "__main__":
    cnf = config.set_param()
    trainer = Trainer(cnf)
    trainer.test()