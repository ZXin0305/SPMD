import argparse

def set_param():
    parser= argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,default='/media/xuchengjun/datasets/panoptic-toolbox',help='the path to load the dataset')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--outh',type=int,default=128)  
    parser.add_argument('--outw',type=int,default=208)
    parser.add_argument('--sigma',type=int,default=6)   #这个参数是控制高斯分布值的
    parser.add_argument('--data_format',type=str,default='cmu') #choose different dataset's joint format
    parser.add_argument('--stage_num',type=int,default=3)

    parser.add_argument('--feature_ch',type=int,default=74)
    parser.add_argument('--cen_ch',type=int,default=1)
    # parser.add_argument('--joint_ch',type=int,default=18)
    parser.add_argument('--offset_ch',type=int,default=18*2)
    parser.add_argument('--rr_ch',type=int,default=19)  #root-relative channefuserl
    parser.add_argument('--kps_ch',type=int,default=36) #kps_weight
    parser.add_argument('--upsample_ch',type=int,default=256)

    parser.add_argument('--epoch',type=int,default=10)
    parser.add_argument('--log_dir',type=str,default='./log',help='the dir to store the summary')
    parser.add_argument('--lr',type=float,default=0.004) # learning rate
    parser.add_argument('--batch_size',type=int,default=4) #batch_size太小了，图片都是高清的
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--model_path',default='/home/xuchengjun/Desktop/zx/SPM_Depth/pth')

    arg = parser.parse_args()
    return arg

if __name__ == "__main__":
    from path import Path
    cnf = set_param()
    model_path = Path(cnf.model_path)
    epoch = 1
    print(model_path / f'epoch_{epoch}.pth')