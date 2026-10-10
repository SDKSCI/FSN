import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_util import *
import kornia
def compute_semantic_pos_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]

    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

    # empirically we find timing 0.005 tend to better performance
    loss_sum =  0.005 * (loss_sem + loss_pos)
    loss_sem_sum =  0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum,  loss_pos_sum

def euclidean(x1,x2): # B,sp,C
	return torch.sum((x1-x2)**2, dim=2).sqrt().mean()

def compute_alignment_loss(content):
    contentA, contentB = content
    b,c,h,w = contentA.shape
    contentA = F.unfold(contentA, kernel_size=16, stride=16).permute(0,2,1).view(b, -1, c, 16 * 16).contiguous() #B, sp, dim, grid
    contentB = F.unfold(contentB, kernel_size=16, stride=16).permute(0,2,1).view(b, -1, c, 16 * 16).contiguous() #B, sp, dim, grid
    
    contentA = torch.mean(contentA, dim=3)
    contentB = torch.mean(contentB, dim=3)
    
    Pa = F.softmax(contentA, dim=2)
    Pb = F.softmax(contentB, dim=2)
    
    kl_divergence = torch.sum(Pa * torch.log(Pa / Pb), dim=2).mean() 
     
    return kl_divergence


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size=8):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


def compute_seg_and_edge_loss(output, edge_label, init_spixl_map_idx):
    """
    output: logits from model [B, C, H, W]
    label: int ground truth [B, H, W]
    edge_label: binary boundary label [B, H, W]
    """
    # ---------- 1. 语义分割损失 ----------
 
    # ---------- 2. 获取预测分割图 ----------
   # [B, H, W]
    pred_class = update_spixl_map(init_spixl_map_idx, output).to(dtype=torch.float32)
    # ---------- 3. 生成 boundary_pred ----------
    # 方法1: 用 Sobel 算子检测预测边缘
    pred_class = pred_class.float()  # [B, 1, H, W]
    boundary_pred = kornia.filters.sobel(pred_class)  # [B, 2, H, W]
    boundary_pred = torch.linalg.norm(boundary_pred, dim=1)  # [B, H, W]，计算梯度幅值

    # ---------- 4. 边界损失（只在 edge_label==1 上） ----------
    edge_label = edge_label.float()
    pos_mask = (edge_label == 1)
    # pos_mask = edge_label

    # Normalize boundary_pred to [0, 1]
    boundary_pred = (boundary_pred - boundary_pred.min()) / (boundary_pred.max() - boundary_pred.min() + 1e-6)

    bce_loss_map = F.binary_cross_entropy(boundary_pred.unsqueeze(1), edge_label, reduction='none')
    edge_loss = (bce_loss_map * pos_mask).sum() / (pos_mask.sum() + 1e-6)


    

    return edge_loss


# def compute_seg_and_edge_loss(output, edge_label, init_spixl_map_idx=None, edge_weight=0.01, smooth=0.1):
#     """
#     output: logits from model [B, C, H, W]
#     edge_label: binary boundary label [B, H, W]
#     init_spixl_map_idx: optional, not used in this version
#     edge_weight: scaling factor for BCE
#     smooth: label smoothing factor
#     """
#     # ---------- 1. 使用 softmax 概率图代替离散索引 ----------
#     prob = F.softmax(output, dim=1)              # [B, C, H, W]
#     pred_class = torch.argmax(prob, dim=1, keepdim=True).float()  # [B,1,H,W]

#     # ---------- 2. Sobel 提取边界 ----------
#     # kornia.sobel 输出 [B,2,H,W]，分别是 x, y 方向梯度
#     sobel_grad = kornia.filters.sobel(pred_class)  # [B,2,H,W]
#     boundary_pred = torch.linalg.norm(sobel_grad, dim=1, keepdim=True)  # [B,1,H,W]

#     # ---------- 3. 归一化到固定范围 ----------
#     boundary_pred = torch.clamp(boundary_pred / 8.0, 0.0, 1.0)  # [B,1,H,W]

#     # ---------- 4. 平滑边界标签 ----------
#     edge_label = edge_label.float()
#     edge_label_smooth = edge_label * (1 - smooth) + smooth * 0.5  # 0->0.05, 1->0.95

#     # ---------- 5. 计算 BCE 只在边界位置 ----------
#     pos_mask = (edge_label == 1).float()
#     bce_map = F.binary_cross_entropy(boundary_pred, edge_label_smooth, reduction='none')
#     edge_loss = (bce_map * pos_mask).sum() / (pos_mask.sum() + 1e-6)

#     # ---------- 6. 缩放 ----------
#     edge_loss = edge_loss * edge_weight

#     return edge_loss


def superpixel_reconstruction_mae_vectorized(image, spix_label_map, init_spixl_map_idx):
    """
    image:         [B, 3, H, W], float32, range [0, 1] or [0, 255]
    spix_label_map:[B, 1, H, W], int64
    init_spixl_map_idx: [B, K, H, W] (float), used in update_spixl_map
    """
    spix_label_map = update_spixl_map(init_spixl_map_idx, spix_label_map).to(dtype=torch.float32)
    spix_label_map = spix_label_map.squeeze(1).long()  # [B, H, W]
    num_spix = int(torch.max(spix_label_map).item()) + 1

    B, C, H, W = image.shape

    one_hot = F.one_hot(spix_label_map, num_classes=num_spix).permute(0, 3, 1, 2).float()  # [B, K, H, W]
    # one-hot [B, K, H, W]
    counts = one_hot.sum(dim=(2, 3), keepdim=False).unsqueeze(-1) + 1e-6  # [B, K, 1]



    image_exp = image.unsqueeze(1)  # [B, 1, 3, H, W]
    one_hot_exp = one_hot.unsqueeze(2)  # [B, K, 1, H, W]
    masked_image = image_exp * one_hot_exp  # [B, K, 3, H, W]

    spix_means = masked_image.sum(dim=(3, 4)) / counts  # [B, K, 3]
    spix_means = spix_means.permute(0, 2, 1)  # [B, 3, K]

    spix_label_flat = spix_label_map.view(B, -1)  # [B, H*W]
    spix_label_idx = spix_label_flat.unsqueeze(1).expand(-1, 3, -1)  # [B, 3, H*W]

    recon_flat = torch.gather(spix_means, dim=2, index=spix_label_idx)  # [B, 3, H*W]
    image_recon = recon_flat.view(B, 3, H, W)  # [B, 3, H, W]

    mae_loss = F.l1_loss(image_recon, image)
    return mae_loss