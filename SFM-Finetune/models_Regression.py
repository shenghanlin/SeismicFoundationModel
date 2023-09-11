# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
import numpy as np
from util.msssim import MSSSIM
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False,Interpolation=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.interpolation = Interpolation
        self.decoder = DecoderCup(in_channels=[self.embed_dim,256,128,64])
        
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=self.num_classes,
            kernel_size=1
        )
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
            
    def generate_mask(self,input_tensor, ratio):
        mask = torch.zeros_like(input_tensor)
        indices = torch.randperm(mask.size(3)//16)[:int(mask.size(3)//16 * ratio)]
        sorted_indices = torch.sort(indices)[0]  # 对索引进行排序
        for i in range(0, len(sorted_indices)):
                mask[:, :, :, sorted_indices[i]*16:(sorted_indices[i]+1)*16] = 1
        return mask
    
    def forward_features(self, x):
        B,C,H,W = x.shape
        
        if self.interpolation:
            mask = self.generate_mask(x,0.75)
            x = x*mask
            img = x
        else:
            img = x
        x = self.patch_embed(x)
        _H,_W = H //self.patch_embed.patch_size[0],W //self.patch_embed.patch_size[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        x = self.decoder(x[:, 1:, :],img)
        x = self.segmentation_head(x)
        if self.interpolation:
            return x,mask
        return x
    
    def forward_Interpolationloss(self, imgs, pred, mask):
        loss1f = torch.nn.MSELoss()
        loss1 = loss1f(imgs, pred*(1-mask)+imgs*mask) 
        loss2f = MSSSIM()
        loss2 = loss2f(imgs, pred*(1-mask)+imgs*mask)
        a = 0.1
        loss = (1-a)*loss1+a*loss2
        return loss
    
    def forward(self, x):
        if self.interpolation:
            pred,mask = self.forward_features(x)
            loss = self.forward_Interpolationloss(x, pred, mask)
            return loss, pred, mask
        x = self.forward_features(x)
        
        return x

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=1, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self,in_channels=[1024,256,128,64]):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(
            1,
            32,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        skip_channels=[0,0,0,32]
        out_channels=[256,128,64,64]
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, img, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        skip_channels=[None,None,None,self.conv_more(img)]
        for i, decoder_block in enumerate(self.blocks):
            x = decoder_block(x, skip=skip_channels[i])
        return x

def forward_loss(imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        loss1f = torch.nn.MSELoss()
        loss1 = loss1f(imgs, pred) 
        loss2f = MSSSIM()
        loss2 = loss2f(imgs, pred)
        a = 0.5
        loss = (1-a)*loss1+a*loss2
        return loss
    

def mae_vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



