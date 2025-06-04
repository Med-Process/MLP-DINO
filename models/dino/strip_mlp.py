from ast import Pass
from email.quoprimime import body_check
import imp
from re import X

from matplotlib.pyplot import axis
from numpy import append

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np

from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from torch.nn.modules import BatchNorm2d

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy, random

from dataclasses import replace
from util.misc import NestedTensor



class BN_Activ_Conv(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=(1, 1), dilation=(1, 1), groups=1):
        super(BN_Activ_Conv, self).__init__()
        self.BN = nn.BatchNorm2d(out_channels)
        self.Activation = activation
        padding = [int((dilation[j] * (kernel_size[j] - 1) - stride[j] + 1) / 2) for j in range(2)]  # Same padding
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=False)

    def forward(self, img):
        img = self.BN(img)
        img = self.Activation(img)
        img = self.Conv(img)
        return img

class DepthWise_Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_merge = BN_Activ_Conv(channels, nn.GELU(), channels, (3, 3), groups=channels)

    def forward(self, img):
        img = self.conv_merge(img)
        return img



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim


        self.proj = nn.Conv2d(in_chans, int(embed_dim), kernel_size=patch_size, stride=patch_size, bias=False)


    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
   
        x = self.proj(x)

        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)
        trunc_normal_(self.embeddings_table, std=.02)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings



class SparseMLP_Block(nn.Module):
    def __init__(self, channels, H, W):
        super().__init__()
        assert W == H
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels//2)
        self.H = H
        if H == 56 or H == 28:
            self.g = int(H / 4)
        elif H == 14 or H == 7:
            self.g = 7
        self.ratio = 1; self.C = int(channels*0.5 / 2); self.chan = int(self.ratio * self.C)

        self.proj_h = nn.Conv2d(H*self.C, self.chan*H, (1, 3), stride=1, padding=(0, 1), groups=self.C,bias=True)
        self.proh_w = nn.Conv2d(self.C*W, self.chan*W, (1, 3), stride=1, padding=(0, 1), groups=self.C, bias=True)

        self.fuse_h = nn.Conv2d(channels, channels//2, (1,1), (1,1), bias=False)
        self.fuse_w = nn.Conv2d(channels, channels//2, (1,1), (1,1), bias=False)


        self.mlp=nn.Sequential(nn.Conv2d(channels, channels, 1, 1,bias=True),nn.BatchNorm2d(channels),nn.GELU())

        dim = channels // 2

        self.fc_h = nn.Conv2d(dim, dim, (3,7), stride=1, padding=(1,7//2), groups=dim, bias=False) 
        self.fc_w = nn.Conv2d(dim, dim, (7,3), stride=1, padding=(7//2,1), groups=dim, bias=False)

        self.reweight = Mlp(dim, dim // 2, dim * 3)

        self.fuse = nn.Conv2d(channels, channels, (1,1), (1,1), bias=False)


        self.relate_pos_h = RelativePosition(channels//2, H)
        self.relate_pos_w = RelativePosition(channels//2, W)

    def window_partion_overlap(self, data, unit, id_h, id_w):
        N, C, H, W = data.shape

        out = None
        for i in range(len(id_h)):
            st_h = id_h[i]
            for j in range(len(id_w)):
                st_w = id_w[j]
                temp = data[:, :, st_h:st_h+unit, st_w:st_w+unit]
                if out is None:
                    out = temp
                else:
                    out = torch.cat([out, temp], dim=0)

        return out

    def window_partion_right_corner(self, data, unit):
        N, C, H, W = data.shape
        h_t = H // unit; w_t = W // unit

        window = data[:, :, -unit*h_t:,-unit*w_t:].view(N, C, unit, h_t, unit, w_t)
        out = window.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, C, unit, unit)
        return out, h_t, w_t

    def process_unit_slidding(self, x):
        N, C, H, W = x.shape
        unit = self.H

        if H < unit:
            pd_size = unit - H; h_re = pd_size
            x = F.pad(x, (0,0,0,pd_size), "constant", 0)
            h_t = 1
        if W < unit:
            pd_size = unit - W; w_re = pd_size
            x = F.pad(x, (0,pd_size, 0,0), "constant", 0)
            w_t = 1

        N_, C_, H_, W_ = x.shape
        h_t = H_ / unit; w_t = W_ / unit

        if H_ == unit:
            h_t = 1; olp_h = 0
        else:
            h_t = round(H_ / unit + 0.5); olp_h = round((h_t * unit - H_) / (h_t - 1) - 0.5)    
        if W_ == unit:
            w_t = 1; olp_w = 0
        else:
            w_t = round(W_ / unit + 0.5); olp_w = round((w_t * unit - W_) / (w_t - 1) - 0.5)   


        # generate index
        last_overlap_h = 0; last_overlap_w = 0
        id_h = []; id_w = []
        for i in range(h_t):
            if i == 0:
                start = 0
            else:
                start = i * unit - i*olp_h
                if start + unit > H_:   
                    start = H_ - unit
                    last_overlap_h = id_h[-1] + unit - start
            id_h.append(start)

        for i in range(w_t):
            if i == 0:
                start = 0
            else:
                start = i * unit - i*olp_w
                if start + unit > W_:    
                    start = W_ - unit
                    last_overlap_w = id_w[-1] + unit - start
            id_w.append(start)

        out = self.window_partion_overlap(x, unit, id_h, id_w)

        return out, id_h, id_w, olp_h, olp_w, last_overlap_h, last_overlap_w


    def process_unit_right_corner_padding(self, x):
        N, C, H, W = x.shape
        unit = self.H

        h_t = H // unit; h_re = abs(h_t * unit - H)   
        w_t = W // unit; w_re = abs(w_t * unit - W)  


        if h_t < 1:
            pd_size = unit - H; h_re = pd_size
            x = F.pad(x, (0,0,pd_size,0), "constant", 0)
            h_t = 1
        
        if w_t < 1:
            pd_size = unit - W; w_re = pd_size
            x = F.pad(x, (pd_size,0, 0,0), "constant", 0)
            w_t = 1
     

        out, _, _ = self.window_partion_right_corner(x, unit)

        out_h = None; out_w = None
        h_ht, h_wt, w_ht, w_wt = 0, 0, 0, 0
        copy_x = copy.copy(x)
        if h_t > 1 and h_re > 0:
            copy_x = F.pad(x[:, :, :, :], (0,0,(unit-h_re), 0), "constant", 0)
        if w_t > 1 and w_re > 0:
            copy_x = F.pad(copy_x[:, :, :, :], ((unit-w_re), 0, 0,0), "constant", 0)

        if h_t > 1 and h_re > 0:
            out_h = copy_x[:, :, :unit, :]
            out_h, h_ht, h_wt = self.window_partion(out_h, unit)
        if w_t > 1 and w_re > 0:
            if h_re > 0:
                out_w = copy_x[:, :, unit:, :unit]
            else:
                out_w = copy_x[:, :, :, :unit]
            out_w, w_ht, w_wt = self.window_partion(out_w, unit)

        return out, out_h, out_w, h_re, w_re, h_t, w_t, h_ht, h_wt, w_ht, w_wt


    def restore_unit_slidding(self, x, N, id_h, id_w, olp_h, olp_w, last_overlap_h, last_overlap_w):
        unit = self.H
        m = 0

        row = []
        for i in range(len(id_h)):
            out = None
            for j in range(len(id_w)):   
                st_w = id_w[j]
                data = x[m*N:(m+1)*N, :, :, :]     
                m = m + 1
                if out is None:
                    out = data
                elif j < len(id_w) -1: 
                    if olp_w == 0:
                        out = torch.cat([out, data], dim=3)
                    else:
                        ori_l = out[:, :, :unit, :-olp_w]
                        ori_m = out[:, :, :unit, -olp_w:]
                        data_m = data[:, :, :, :olp_w]
                        data_r = data[:, :, :, olp_w:]
                        out = torch.cat([ori_l, ori_m*0.5+data_m*0.5, data_r], dim=3)
                elif last_overlap_w != 0:
                    ori_l = out[:, :, :unit, :-last_overlap_w]
                    ori_m = out[:, :, :unit, -last_overlap_w:]
                    data_m = data[:, :, :, :last_overlap_w]
                    data_r = data[:, :, :, last_overlap_w:]
                    out = torch.cat([ori_l, ori_m*0.5+data_m*0.5, data_r], dim=3)
                else:
                    if olp_w == 0:
                        out = torch.cat([out, data], dim=3)
                    else:
                        ori_l = out[:, :, :unit, :-olp_w]
                        ori_m = out[:, :, :unit, -olp_w:]
                        data_m = data[:, :, :, :olp_w]
                        data_r = data[:, :, :, olp_w:]
                        out = torch.cat([ori_l, ori_m*0.5+data_m*0.5, data_r], dim=3)
            row.append(out)


        out = None
        for i in range(len(id_h)):
            st_h = id_h[i]
            data = row[i]        
            if out is None:
                out = data
            elif i < len(id_h) -1:   
                if olp_h == 0:
                    out = torch.cat([out, data], dim=2)
                else:
                    ori_u = out[:, :, :-olp_h, :]
                    ori_m = out[:, :, -olp_h:, :]
                    data_m = data[:, :, :olp_h, :]
                    data_d = data[:, :, olp_h:, :]
                    out = torch.cat([ori_u, ori_m*0.5+data_m*0.5, data_d], dim=2)
            elif last_overlap_h != 0:
                ori_u = out[:, :, :-last_overlap_h, :]
                ori_m = out[:, :, -last_overlap_h:, :]
                data_m = data[:, :, :last_overlap_h, :]
                data_d = data[:, :, last_overlap_h:, :]
                out = torch.cat([ori_u, ori_m*0.5+data_m*0.5, data_d], dim=2)
            else:
                if olp_h == 0:
                    out = torch.cat([out, data], dim=2)
                else:
                    ori_u = out[:, :, :-olp_h, :]
                    ori_m = out[:, :, -olp_h:, :]
                    data_m = data[:, :, :olp_h, :]
                    data_d = data[:, :, olp_h:, :]
                    out = torch.cat([ori_u, ori_m*0.5+data_m*0.5, data_d], dim=2)

        return out

    def restore_unit_right_corner_padding(self, x, edge_h, edge_w, H, W, h_re, w_re, h_t, w_t):

        if edge_w is not None:
            x = torch.cat([edge_w[:, :, :, -w_re:], x], dim=3)    
        if edge_h is not None:
            if w_re > 0:
                x = torch.cat([edge_h[:, :, -h_re:, W-w_re:], x], dim=2)   
            else:
                x = torch.cat([edge_h[:, :, -h_re:, :], x], dim=2)  

        
        return x

    def forward(self, x, layer):
        N, C, H, W = x.shape

        x = self.mlp(x)

        x_1 = x[:, :C//2, :, :]
        x_2 = x[:, C//2:, :, :]
        
        x_1 = self.sparse_mlp(x_1, layer)

        x_w = self.fc_h(x_2)
        x_h = self.fc_w(x_2)
        att = F.adaptive_avg_pool2d(x_h + x_w + x_2, output_size=1)
        att = self.reweight(att).reshape(N, C//2, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x_2 = x_h * att[0] + x_w * att[1] + x_2 * att[2]

        x = self.fuse(torch.cat([x_1, x_2], dim=1))

        return x

    def sparse_mlp(self, x, layer):
        N_, C_, H_, W_ = x.shape
        
        short_cut = x

        core, id_h, id_w, olp_h, olp_w, last_overlap_h, last_overlap_w = self.process_unit_slidding(x)

        N, C, H, W = core.shape
        pos_h = self.relate_pos_h(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        pos_w = self.relate_pos_w(H, W).unsqueeze(0).permute(0, 3, 1, 2)

        C1 = int(C/self.C)

        x_w = self.CGSMM(core, pos_h, pos_w, C1)

        x = self.restore_unit_slidding(x_w, N_, id_h, id_w, olp_h, olp_w, last_overlap_h, last_overlap_w)
        n,c,h,w = short_cut.shape

        x = self.fuse_w(torch.cat([short_cut, x[:, :, :h, :w]], dim=1))

        return x


    def CGSMM(self, core, pos_h, pos_w, C1):
        N, C, H, W = core.shape
        x_h = core + pos_h
        x_h = x_h.view(N, C1, self.C, H, W)     
        x_h = x_h.permute(0, 1, 3, 2, 4).contiguous().view(N, C1, H, self.C*W)  

        x_h = self.proj_h(x_h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_h = x_h.view(N, C1, H, self.C, W).permute(0, 1, 3, 2, 4).contiguous().view(N, C, H, W) 

        x_h = self.fuse_h(torch.cat([x_h, core], dim=1))
        x_h = self.activation(self.BN(x_h)) + pos_w


        x_w = self.proh_w(x_h.view(N, C1, H*self.C, W).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_w = x_w.contiguous().view(N, C1, self.C, H, W).view(N, C, H, W)

        return x_w




class TokenMixing(nn.Module):
    r""" Token mixing of Sparse MLP

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, C, H, W):
        super().__init__()
        self.smlp_block = SparseMLP_Block(C, H, W)
        self.dwsc = DepthWise_Conv(C)

    
    def forward(self, x, layer):
        x = self.dwsc(x)
        x = self.smlp_block(x, layer)

        return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ChannelMixing(nn.Module):

    def __init__(self, in_channel, alpha, use_dropout=False, drop_rate=0):
        super().__init__()

        self.use_dropout = use_dropout

        self.conv_77 = nn.Conv2d(in_channel, in_channel, 7, 1, 3, groups=in_channel, bias=False)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.fc1 = nn.Linear(in_channel, alpha * in_channel)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(alpha * in_channel, in_channel)

        self.grn = GRN(3*in_channel)

    
    def forward(self, x):
        N, C, H, W = x.shape

        x = self.conv_77(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.grn(x)

        x = self.fc2(x)

        x = x.permute(0, 3, 1, 2)

        return x




class BasicBlock(nn.Module):
    def __init__(self, in_channel, H, W, alpha, use_dropout=0, drop_rate=0):
        super().__init__()

        self.layer = use_dropout
        self.token_mixing = TokenMixing(in_channel, H, W)
        self.channel_mixing = ChannelMixing(in_channel, alpha, use_dropout, drop_rate)
        
        drop_rate = 0.1

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.token_mixing(x, self.layer))
        x = x + self.drop_path(self.channel_mixing(x))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):  
        super().__init__()
        self.input_resolution = input_resolution

        self.max_pooling2d = nn.MaxPool2d(2, 2, 0)
        self.avg_pooling2d = nn.AvgPool2d(2, 2, 0)


    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, C, H_, W_ = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x1 = self.max_pooling2d(x)
        x2 = self.avg_pooling2d(x)

        # # cat
        x = torch.cat([x1, x2], dim=1)

        return x

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops



class StripMLPNet(nn.Module):
    """
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        layers (tuple(int)): Depth of each Swin Transformer layer.
        drop_rate (float): Dropout rate. Default: 0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=80, layers=[2, 8, 14, 2], drop_rate=0.5,
                 norm_layer=nn.BatchNorm2d, alpha=3, use_dropout=False, patch_norm=True, **kwargs):
        super(StripMLPNet, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(layers)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_norm = patch_norm
        self.drop_rate = drop_rate

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size, bias=False)
        
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.patches_resolution = patches_resolution

        self.avgpool = nn.AvgPool2d(2,2)

        self.blocks1 = nn.ModuleList()
        for i in range(layers[0]):
            basic = BasicBlock(embed_dim, self.patches_resolution[0], self.patches_resolution[1], alpha, use_dropout=i, drop_rate=drop_rate)
            self.blocks1.append(basic)

        self.blocks2 = nn.ModuleList()
        for i in range(layers[1]):
            basic = BasicBlock(embed_dim*2, int(self.patches_resolution[0]/2), int(self.patches_resolution[1]/2), alpha, use_dropout=i, drop_rate=drop_rate)
            self.blocks2.append(basic)
        
        self.blocks3 = nn.ModuleList()
        for i in range(layers[2]):
            basic = BasicBlock(embed_dim*4, int(self.patches_resolution[0]/4), int(self.patches_resolution[1]/4), alpha, use_dropout=i, drop_rate=drop_rate)
            self.blocks3.append(basic)

        self.blocks4 = nn.ModuleList()
        for i in range(layers[3]):
            basic = BasicBlock(embed_dim*8, int(self.patches_resolution[0]/8), int(self.patches_resolution[1]/8), alpha, use_dropout=i, drop_rate=drop_rate)
            self.blocks4.append(basic)

        self.merging1 = nn.Conv2d(embed_dim, embed_dim*2, 2, 2, bias=False)
        self.merging2 = nn.Conv2d(embed_dim*2, embed_dim*4, 2, 2, bias=False)
        self.merging3 = nn.Conv2d(embed_dim*4, embed_dim*8, 2, 2, bias=False)

        self.conv_s1_28 = nn.Conv2d(embed_dim*2, embed_dim*4, (2,2), 2, 0, groups=embed_dim*2, bias=False)
        self.conv_s1_14 = nn.Conv2d(embed_dim*4, embed_dim*8, (2,2), 2, 0, groups=embed_dim*4, bias=False)
        self.conv_s2_14 = nn.Conv2d(embed_dim*4, embed_dim*8, (2,2), 2, 0, groups=embed_dim*4, bias=False)
        
        self.norm = nn.BatchNorm2d(self.num_features)

    def forward(self, x_tens):
        x = x_tens.tensors
        out = []

        x = self.patch_embed(x)

        x = self.blocks(self.blocks1, x)

        x = self.merging1(x)

        x_s1_14 = self.conv_s1_28(x)   
        x_s1_7 = self.conv_s1_14(x_s1_14)

        x = self.blocks(self.blocks2, x)
        out.append(x)

        x = self.merging2(x)           

        x_s2_7 = self.conv_s2_14(x)

        x = self.blocks(self.blocks3, x + x_s1_14)
        out.append(x)
        
        x = self.merging3(x)

        x = self.blocks(self.blocks4, x + x_s1_7 + x_s2_7)

        x = self.norm(x)  # N C H W
        out.append(x)

        # collect for nesttensors        
        outs_dict = {}
        for idx, out_i in enumerate(out):
            m = x_tens.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(out_i, mask)

        return outs_dict

    def blocks(self, blocks, x):
        for b in blocks:
            x = b(x)
        return x


    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            msg = self.load_state_dict(state_dict['model'], strict=False)
            print("load pretrained model:", msg)
            print('Successfully load backbone ckpt.')

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


