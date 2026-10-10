from matplotlib.pyplot import axis
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .model_util import *
from train_util import *

# define the function includes in import *
__all__ = [
    'FSNet'
]


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
       
        self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, in_channels),
                nn.Sigmoid()
            )
     
    def forward(self, x):
        b, c, H, W = x.shape

        global_feat = self.global_avg_pool(x).view(b, c)  # (B, C)
        channel_weights = self.fc(global_feat).view(b, c, 1, 1)  # (B, C, 1, 1)
        return x * channel_weights  # 加权通道信息


class con(nn.Module):

    def __init__(self, inchannel, outchannel):
        super(con, self).__init__()
        self.conva = conv(True, inchannel, outchannel, kernel_size=3)
        self.convb = conv(True, outchannel, outchannel, kernel_size=3)
        self.conv1 = conv(True, inchannel, outchannel, kernel_size=1)
        self.conv2 = conv(True, outchannel, outchannel, kernel_size=1)

        self.conv_de = conv(True, outchannel * 2 + inchannel, outchannel, kernel_size=1)

        self.se_block = SEBlock(in_channels=inchannel)

    def forward(self, x):
        x0 = self.se_block(x)

        out1 = self.conva(x0)
        out2 = self.convb(out1)

        out3 = self.conv1(x0)
        out4 = self.conv2(out3)
        out = torch.cat([x0, out2, out4], 1)
        out = self.conv_de(out)

        return out

class Disentangle(nn.Module):
    def __init__(self, dim=64, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale

        # 边缘引导特征提取
        self.edge_extract = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 仅保留空间注意力（无通道注意力）
        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )               

        # 主干特征提取
        self.shared_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # 轻量卷积分支，提取 common / unique 特征
        self.common_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False)
        )
        self.unique_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False)
        )

    def forward(self, x):
        # 边缘注意力引导
        edge = self.edge_extract(x)
        spatial = self.spatial_att(x)

        # 边缘增强的空间注意力
        att = torch.sigmoid(spatial * (1 + edge))
        feat = self.shared_conv(x * att)

        # common / unique mask
        common_mask = torch.sigmoid(self.common_head(feat))
        unique_mask = torch.sigmoid(self.unique_head(feat))

        # 输出特征
        common = (x * common_mask) + self.res_scale * x
        unique = (x * unique_mask) + self.res_scale * x

        # 正交约束（互斥性损失）
        orth_loss = torch.mean((common_mask * unique_mask) ** 2)

        return unique, orth_loss


class water_diffusion(nn.Module):
    def __init__(self, dim=64):
        super(water_diffusion, self).__init__()
        self.conv_up = deconv(dim, dim)  # up ->2
        self.fusion = conv(True, dim*2, dim, kernel_size=3)

    def forward(self, input, flow):
        flow = self.conv_up(flow)
        _, _, h, w = flow.shape
        inp = F.interpolate(input, size=(h, w), mode='bilinear')
        out = self.fusion(torch.cat([inp, flow], dim=1))
        return out
class Embedder(nn.Module):
    def __init__(self, use_assist=False):
        super(Embedder, self).__init__()
        self.use_assist = use_assist
        self.head0 = conv(True, 64 * 5, 64, kernel_size=3)
        self.head1 = nn.Sequential(
            nn.Conv2d(64 + 3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

        )

        self.conv1 = con(3, 64)
        self.conv2 = con(64, 64)
        self.conv3 = con(64, 64)
        self.conv4 = con(64, 64)
        self.conv5 = con(64, 64)
        self.pool0 = nn.MaxPool2d(3, 2, 1)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        self.pool3 = nn.MaxPool2d(3, 2, 1)
        self.d1= Disentangle()
 

        self.diffusion_step = int(np.log2(16))
        self.up = nn.ModuleList([water_diffusion() for i in range(self.diffusion_step)])
        self.decoder1 = water_diffusion()
        self.decoder2 = water_diffusion()
        self.decoder3 = water_diffusion()
        self.decoder4 = water_diffusion()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(64 * 4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
    
        
        


    def forward(self, x):
        if self.use_assist:
            noise = torch.rand_like(x) * (torch.mean(x) * 0.001)
            x = (x + noise)

        x0 = self.conv1(x)
        x1 = self.pool0(x0)

        x1 = self.conv2(x1)
        x2 = self.pool1(x1)

        x2 = self.conv3(x2)
        x3 = self.pool2(x2)

        x3 = self.conv4(x3)
        x4 = self.pool3(x3)
        x4 = self.conv5(x4)

        
        


        x1_up = F.interpolate(x1, size=[x0.shape[2], x0.shape[3]])
        x2_up = F.interpolate(x2, size=[x0.shape[2], x0.shape[3]])
        x3_up = F.interpolate(x3, size=[x0.shape[2], x0.shape[3]])
        x4_up = F.interpolate(x4, size=[x0.shape[2], x0.shape[3]])

        out = torch.cat([x0, x1_up, x2_up, x3_up, x4_up], 1)

        out = self.head0(out)
        out = self.head1(torch.cat([out, x], dim=1))
        out1,loss0=self.d1(out)

        b, c, h, w = out1.shape
       
        spixel = (h // 16, w // 16)
        flow = F.interpolate(out1, size=spixel, mode='bilinear')


        flow1 = self.decoder1(x3, flow)
        flow2 = self.decoder2(x2, flow1)
        flow3 = self.decoder3(x1, flow2)
        flow4 = self.decoder4(x0, flow3)

        flow1_up = F.interpolate(flow1, size=[x.shape[2], x.shape[3]])
        flow2_up = F.interpolate(flow2, size=[x.shape[2], x.shape[3]])
        flow3_up = F.interpolate(flow3, size=[x.shape[2], x.shape[3]])

        fused = self.fuse_conv(torch.cat([flow1_up, flow2_up, flow3_up,  flow4], dim=1))

        loss=loss0


        return x4_up,out1,fused,loss



class FSNSpixelNet(nn.Module):
    expansion = 1

    def __init__(self, bn=True, use_assist=True):
        super(FSNSpixelNet, self).__init__()

        self.bn = bn
        self.assign_ch = 9
        self.use_assist = use_assist
        if self.use_assist and self.training:
            self.encoder_assit1 = Embedder(use_assist)
            self.encoder_assit2 = Embedder(use_assist)
            self.gap = nn.AdaptiveAvgPool2d((1))

        self.encoder = Embedder()

        self.head = nn.Sequential(
            predict_mask(64, self.assign_ch),
            nn.Softmax(1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x, br_assit=None, sen_assit=None):
        if self.training:
            return self._forward_train(x, br_assit, sen_assit)
        else:
            return self._forward_inference(x)

    def _forward_train(self, x, br_assit=None, sen_assit=None):
        if self.use_assist:
            encoder_br_assit,AD_br_assit,flow_br_assit,loss_br = self.encoder_assit1(br_assit)
            
            prob_br_assit = self.head(flow_br_assit)

            encoder_sen_assit, AD_sen_assit,flow_sen_assit,loss_sen= self.encoder_assit2(sen_assit)
            
            prob_sen_assit = self.head(flow_sen_assit)

            encoder_x,AD_x,flow,loss_x = self.encoder(x)
           
            prob = self.head(flow)
            loss = loss_br + loss_sen + loss_x
            

            return (loss,prob, prob_br_assit, prob_sen_assit,
            [encoder_x, encoder_br_assit], [encoder_x, encoder_sen_assit], [self.gap(encoder_x), self.gap(encoder_br_assit)], [self.gap(encoder_x), self.gap(encoder_sen_assit)],
            [AD_x, AD_br_assit], [AD_x, AD_sen_assit], [self.gap(AD_x), self.gap(AD_br_assit)], [self.gap(AD_x), self.gap(AD_sen_assit)],
            [flow, flow_br_assit], [flow, flow_sen_assit], [self.gap(flow), self.gap(flow_br_assit)], [self.gap(flow), self.gap(flow_sen_assit)])

    def _forward_inference(self, x):

        _,_,flow,loss = self.encoder(x)
        prob = self.head(flow)
        return prob

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def FSNet(data=None):
    # model with batch normalization
    model =  FSNSpixelNet(bn=True)
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model

