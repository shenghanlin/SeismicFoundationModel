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

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.decoder = VIT_MLAHead(mla_channels=self.embed_dim,num_classes=self.num_classes)
        
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=self.num_classes,
            kernel_size=3,
        )
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim) 
            del self.norm  # remove the original norm

    def forward_features(self, x):
        B,C,H,W = x.shape
        x = self.patch_embed(x)
        _H,_W = H //self.patch_embed.patch_size[0],W //self.patch_embed.patch_size[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        featureskip = []
        featureskipnum = 1
        for blk in self.blocks:
            x = blk(x)
            if featureskipnum%(len(self.blocks)//4)==0: 
                featureskip.append(x[:,1:,:])
                # print(featureskipnum)
            featureskipnum += 1
        
        x = self.decoder(featureskip[0],featureskip[1],featureskip[2],featureskip[3],h=_H,w=_W)
        return x
        
    def forward(self, x):
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
        # print(x.shape,skip.shape)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self):
        super().__init__()
        # self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            1024,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        
        decoder_channels = (256,128,64,16)


        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        # if self.config.n_skip != 0:
        #     skip_channels = self.config.skip_channels
        #     for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
        #         skip_channels[3-i]=0
        # else:
        #     skip_channels=[0,0,0,0]
        skip_channels=[512,256,128,64]
        self.conv_feature1 = Conv2dReLU(1024,skip_channels[0],kernel_size=3,padding=1,use_batchnorm=True)
        self.conv_feature2 = Conv2dReLU(1024,skip_channels[1],kernel_size=3,padding=1,use_batchnorm=True)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_feature3 = Conv2dReLU(1024,skip_channels[2],kernel_size=3,padding=1,use_batchnorm=True)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv_feature4 = Conv2dReLU(1024,skip_channels[3],kernel_size=3,padding=1,use_batchnorm=True)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=8)

        # skip_channels=[128,64,32,8]
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def TransShape(self,x,head_channels = 512,up=0):
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)

        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        if up==0:
            x = self.conv_feature1(x)
        elif up==1:
            x = self.conv_feature2(x)
            x = self.up2(x)
        elif up==2:
            x = self.conv_feature3(x)
            x = self.up3(x)
        elif up==3:
            x = self.conv_feature4(x)
            x = self.up4(x)
        return x

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        skip_channels=[512,256,128,64]
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = self.TransShape(features[i],head_channels=skip_channels[i],up=i)
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class MLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
            nn.Conv2d(
                                       mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels), nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        head2 = F.interpolate(self.head2(
            mla_p2), (4*mla_p2.shape[-2],4*mla_p2.shape[-1]), mode='bilinear', align_corners=True) 
        head3 = F.interpolate(self.head3(
            mla_p3), (4*mla_p3.shape[-2],4*mla_p3.shape[-1]), mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head4(
            mla_p4), (4*mla_p4.shape[-2],4*mla_p4.shape[-1]), mode='bilinear', align_corners=True)
        head5 = F.interpolate(self.head5(
            mla_p5), (4*mla_p5.shape[-2],4*mla_p5.shape[-1]), mode='bilinear', align_corners=True)
        return torch.cat([head2, head3, head4, head5], dim=1)


class VIT_MLAHead(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=768, mla_channels=256, mlahead_channels=128,num_classes=6,
                 norm_layer=nn.BatchNorm2d, norm_cfg=None, **kwargs):
        super(VIT_MLAHead, self).__init__(**kwargs)
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.num_classes = num_classes
        self.mlahead = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        self.cls = nn.Conv2d(4 * self.mlahead_channels,
                             self.num_classes, 3, padding=1)

    def forward(self, x1,x2,x3,x4,h=14,w=14):
        B,n_patch,hidden = x1.size()
        if h==w:
            h,w = int(np.sqrt(n_patch)),int(np.sqrt(n_patch))
        x1 = x1.permute(0,2,1)
        x1 = x1.contiguous().view(B,hidden,h,w)
        x2 = x2.permute(0,2,1)
        x2 = x2.contiguous().view(B,hidden,h,w)
        x3 = x3.permute(0,2,1)
        x3 = x3.contiguous().view(B,hidden,h,w)
        x4 = x4.permute(0,2,1)
        x4 = x4.contiguous().view(B,hidden,h,w)
        x = self.mlahead(x1,x2,x3,x4)
        x = self.cls(x)
        x = F.interpolate(x, size=(h*16,w*16), mode='bilinear',
                          align_corners=True)
        return x


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


