import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from IPython import embed

class conv_bn_relu(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding,
                 has_bn=True,has_relu=True):
        super(conv_bn_relu,self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                stride=stride,padding=padding) 
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)


    def forward(self,x):
        def _func_factory(conv ,bn ,relu ,has_bn ,has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func

        func = _func_factory(self.conv, self.bn, self.relu, self.has_bn, self.has_relu )
        x = func(x)

        return x

class ResNet_top(nn.Module):
    def __init__(self):
        super(ResNet_top,self).__init__()
        self.conv = conv_bn_relu(3, 64, kernel_size=7, stride=2, padding=3,has_bn=True,has_relu=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    def forward(self,x):
        x1 = self.conv(x)
        x2 = self.maxpool(x1)

        return x1,x2


# ------------------------------------------------------------
# downsample module
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self,in_planes,planes,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        """
        每一个layer的通道数是中间小,两边大,而且大的是小的4倍
        """
        self.conv_bn_relu1 = conv_bn_relu(in_planes, planes, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True) 
        self.conv_bn_relu2 = conv_bn_relu(planes, planes, kernel_size=3,
                stride=stride, padding=1, has_bn=True, has_relu=True) 
        self.conv_bn_relu3 = conv_bn_relu(planes, planes * self.expansion,
                kernel_size=1, stride=1, padding=0, has_bn=True,
                has_relu=False) 
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self,x):
        out = self.conv_bn_relu1(x)
        out = self.conv_bn_relu2(out)
        out = self.conv_bn_relu3(out)

        if self.downsample is not None:    #其实每个layer都有downsample
            x = self.downsample(x) 

        out += x
        out = self.relu(out)

        return out

class ResNet_downsample_module(nn.Module):
    def __init__(self,block,layers,has_skip=False,zero_init_residual=False):
        super(ResNet_downsample_module,self).__init__()
        self.has_skip = has_skip
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        #权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = conv_bn_relu(self.in_planes, planes * block.expansion, 
                            kernel_size=1, stride=stride, padding=0, has_bn=True, has_relu=False)  

        layers = list()
        layers.append(block(self.in_planes,planes,stride,downsample))  #后面的都没有用stride
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))   

        return nn.Sequential(*layers)

    def forward(self, x, skip1, skip2):
        x1 = self.layer1(x)
        if self.has_skip:
            x1 = x1 + skip1[0] + skip2[0]
        x2 = self.layer2(x1)
        if self.has_skip:
            x2 = x2 + skip1[1] + skip2[1]
        x3 = self.layer3(x2)
        if self.has_skip:
            x3 = x3 + skip1[2] + skip2[2]
        x4 = self.layer4(x3)
        if self.has_skip:
            x4 = x4 + skip1[3] + skip2[3]     

        return x4, x3, x2 ,x1  

#------------------------------------------------------
# upsample module
class Upsample_unit(nn.Module):

    def __init__(self, idx, in_planes, up_size, output_ch_list, output_shape, upsample_ch=256, gen_skip=False, gen_cross_conv=False):
        super(Upsample_unit,self).__init__()
        self.output_shape = output_shape
        
        self.u_skip = conv_bn_relu(in_planes, upsample_ch, kernel_size=1, stride=1, padding=0,
                            has_bn=True, has_relu=False)
        self.relu = nn.ReLU(inplace=True)

        self.idx = idx
        if self.idx > 0:
            self.up_size = up_size
            self.up_conv = conv_bn_relu(upsample_ch, upsample_ch, kernel_size=1, stride=1, padding=0,
                            has_bn=True, has_relu=False)

        self.gen_skip = gen_skip
        if self.gen_skip:
            self.skip1 = conv_bn_relu(in_planes, in_planes, kernel_size=1, stride=1, padding=0, 
                            has_bn=True, has_relu=True)
            self.skip2 = conv_bn_relu(upsample_ch, in_planes, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True)      

        self.gen_cross_conv = gen_cross_conv
        if self.idx == 3 and self.gen_cross_conv:
            self.cross_conv = conv_bn_relu(upsample_ch, 64, kernel_size=1, stride=1, padding=0, has_bn=True, has_relu=True)

        #1.center_map
        self.center_map_conv1 = conv_bn_relu(upsample_ch,upsample_ch,kernel_size=1,stride=1,padding=0,
                                            has_bn=True,has_relu=True)
        self.center_map_conv2 = conv_bn_relu(upsample_ch,output_ch_list[0],kernel_size=1,stride=1,padding=0,
                                            has_bn=True,has_relu=False)
        #2.offset_map
        self.offset_map_conv1 = conv_bn_relu(upsample_ch,upsample_ch,kernel_size=1,stride=1,padding=0,
                                            has_bn=True,has_relu=True)
        self.offset_map_conv2 = conv_bn_relu(upsample_ch,output_ch_list[1],kernel_size=1,stride=1,padding=0,
                                            has_bn=True,has_relu=False)

        #3.realitive root depth map
        self.reldep_map_conv1 = conv_bn_relu(upsample_ch,upsample_ch,kernel_size=1,stride=1,padding=0,
                                            has_bn=True,has_relu=True)
        self.reldep_map_conv2 = conv_bn_relu(upsample_ch,output_ch_list[2],kernel_size=1,stride=1,padding=0,
                                            has_bn=True,has_relu=False)
                                               
    def forward(self, x, up_x):
        out = self.u_skip(x)  #--> x4, x3, x2, x1 // when is x4 , do not carry out the interpolate operation

        if self.idx > 0:
            up_x = F.interpolate(up_x, size=self.up_size, mode='bilinear', align_corners=True)
            up_x = self.up_conv(up_x)
            out += up_x
        out = self.relu(out)

        #1.center_map
        res_c = self.center_map_conv1(out)
        res_c = self.center_map_conv2(res_c)
        res_c = F.interpolate(res_c, size=self.output_shape, mode='bilinear', align_corners=True)

        #2.offset_map
        res_o = self.offset_map_conv1(out)
        res_o = self.offset_map_conv2(res_o)
        res_o = F.interpolate(res_o, size=self.output_shape, mode='bilinear', align_corners=True)

        #3.realitive root depth map
        res_r = self.reldep_map_conv1(out)
        res_r = self.reldep_map_conv2(res_r)
        res_r = F.interpolate(res_r, size=self.output_shape, mode='bilinear', align_corners=True)

        skip1 = None
        skip2 = None
        if self.gen_skip:
            skip1 = self.skip1(x)
            skip2 = self.skip2(out)

        cross_conv = None
        if self.idx == 3 and self.gen_cross_conv:
            cross_conv = self.cross_conv(out)

        return out, res_c, res_o, res_r, skip1, skip2, cross_conv



class Upsample_module(nn.Module):

    def __init__(self,output_ch_list, out_shape, upsample_ch=256, gen_skip=False, gen_cross_conv=False):
        super(Upsample_module,self).__init__()
        self.in_planes = [2048,1024,512,256]
        h , w = out_shape
        self.up_sizes = [(h // 8, w // 8), (h // 4, w // 4), (h // 2, w // 2), (h, w)]
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv

        self.up1 = Upsample_unit(0, self.in_planes[0], self.up_sizes[0], output_ch_list=output_ch_list, output_shape=out_shape, upsample_ch=upsample_ch,
                                gen_skip=gen_skip,gen_cross_conv=gen_cross_conv)
        self.up2 = Upsample_unit(1, self.in_planes[1], self.up_sizes[1], output_ch_list=output_ch_list, output_shape=out_shape, upsample_ch=upsample_ch,
                                gen_skip=gen_skip,gen_cross_conv=gen_cross_conv)
        self.up3 = Upsample_unit(2, self.in_planes[2], self.up_sizes[2], output_ch_list=output_ch_list, output_shape=out_shape, upsample_ch=upsample_ch,
                                gen_skip=gen_skip,gen_cross_conv=gen_cross_conv)
        self.up4 = Upsample_unit(3, self.in_planes[3], self.up_sizes[3], output_ch_list=output_ch_list, output_shape=out_shape, upsample_ch=upsample_ch,
                                gen_skip=gen_skip,gen_cross_conv=gen_cross_conv)

    def forward(self, x4, x3, x2, x1):
        out1, res_c1, res_o1, res_r1, skip1_1, skip2_1, _ = self.up1(x4, None)
        out2, res_c2, res_o2, res_r2, skip1_2, skip2_2, _ = self.up2(x3, out1)
        out3, res_c3, res_o3, res_r3, skip1_3, skip2_3, _ = self.up3(x2, out2)
        out4, res_c4, res_o4, res_r4, skip1_4, skip2_4, cross_conv = self.up4(x1, out3)

        res_c = [res_c1, res_c2, res_c3, res_c4]
        res_o = [res_o1, res_o2, res_o3, res_o4]
        res_r = [res_r1, res_r2, res_r3, res_r4]
        skip1 = [skip1_4, skip1_3, skip1_2, skip1_1]
        skip2 = [skip2_4, skip2_3, skip2_2, skip2_1]

        return cross_conv, res_c, res_o, res_r, skip1, skip2


class Single_stage_module(nn.Module):
    def __init__(self,output_ch_list,output_shape,has_skip=False,
                    gen_skip=False,gen_cross_conv=False,upsample_ch=256,
                    zero_init_residual=False,add_ori_sprvi=False):
        super(Single_stage_module, self).__init__()
        self.has_skip = has_skip
        self.gen_skip = gen_skip
        self.gen_cross_conv = gen_cross_conv
        self.add_ori_sprvi = add_ori_sprvi
        self.upsample_ch = upsample_ch
        self.zero_init_residual = zero_init_residual
        self.layers = [3,4,6,3]
        self.downsample = ResNet_downsample_module(Bottleneck,self.layers,
                            self.has_skip,self.zero_init_residual)
        self.upsample = Upsample_module(output_ch_list,output_shape,
                            self.upsample_ch,self.gen_skip,self.gen_cross_conv)
        self.dilateConv = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=2,dilation=2)

    def forward(self,x,skip1,skip2,ori_x):
        if self.add_ori_sprvi:
            sup_ = self.dilateConv(ori_x)
            x = sup_ + x
        
        x4,x3,x2,x1 = self.downsample(x,skip1,skip2)

        cross_conv,res_c,res_o,res_r,skip1,skip2 = self.upsample(x4,x3,x2,x1)
        return cross_conv,res_c,res_o,res_r,skip1,skip2


class SPMD(nn.Module):
    def __init__(self,cnf):
        super(SPMD,self).__init__()

        self.stage_num = cnf.stage_num
        self.center_ch = cnf.cen_ch
        self.offset_ch = cnf.offset_ch
        self.reroot_ch = cnf.rr_ch
        self.upsample_ch = cnf.upsample_ch
        self.output_shape = (cnf.outh,cnf.outw)

        self.top = ResNet_top()
        self.modules_stages = list()
        for i in range(self.stage_num):
            if i == 0:
                has_skip = False        #stage 1 没有残差连接
                add_ori_sprvi = False
            else:
                has_skip = True         #stage 2\3 有残差连接
                add_ori_sprvi = True    #这个是为了添加监督
            if i != self.stage_num - 1: #stage 1\2 产生残差 并且具有跨块的卷积操作
                gen_skip = True
                gen_cross_conv = True
            else:
                gen_skip = False        #stage 3不再产生残差,不进行跨块卷积
                gen_cross_conv = False

            self.modules_stages.append(
                Single_stage_module(
                    [self.center_ch,self.offset_ch,self.reroot_ch],self.output_shape,
                    has_skip=has_skip, gen_skip=gen_skip,
                    gen_cross_conv=gen_cross_conv,
                    upsample_ch=self.upsample_ch,
                    add_ori_sprvi=add_ori_sprvi
                )
            ) 

            setattr(self, 'stage%d' % i, self.modules_stages[i])  #设置属性,方便调用


    def forward(self,imgs):
        x1,x = self.top(imgs)
        ori_x = x1
        skip1 = None
        skip2 = None
        outputs = dict()
        outputs['center_map'] = list()
        outputs['offset_map'] = list()
        outputs['rr_demap'] = list()

        for i in range(self.stage_num):
            x,res_c,res_o,res_r,skip1,skip2 = eval('self.stage' + str(i))(x,skip1,skip2,ori_x)
            outputs['center_map'].append(res_c)
            outputs['offset_map'].append(res_o)
            outputs['rr_demap'].append(res_r)

        return outputs  #--> a dict
