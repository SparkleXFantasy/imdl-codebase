import os
import torch
import torch.nn.functional as F
from torch import nn
from .swin_transformer import SwinTransformer


class UpsampleTransition(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.transit1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.transit2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.transit1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.transit2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        return x


class MapTransition(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.transit = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_chans)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transit(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        return x
    

class PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_q = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_k = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.randn(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_q = self.conv_q(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_k = self.conv_k(x).view(batch_size, -1, height * width)
        attention = self.softmax(torch.bmm(feat_q, feat_k))
        feat_v = self.conv_v(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_v, attention.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e

        return out
    

class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.randn(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e

        return out
    

class LGMNet(nn.Module):
    def __init__(self, 
                img_size=256,
                patch_size=4,
                in_chans=3,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                upsampler_mode='bicubic'):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.image_encoder = SwinTransformer(
            pretrain_img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample_transit_final = UpsampleTransition(24, 12)
        self.upsample_transit0 = UpsampleTransition(48, 24)
        self.upsample_transit1 = UpsampleTransition(96, 48)
        self.upsample_transit2 = UpsampleTransition(192, 96)
        self.upsample_transit3 = UpsampleTransition(384, 192)
        self.upsample_transit4 = UpsampleTransition(768, 384)
        self.map_transit_final = MapTransition(13, 1)
        self.map_transit0 = MapTransition(25, 1)
        self.map_transit1 = MapTransition(49, 1)
        self.map_transit2 = MapTransition(97, 1)
        self.map_transit3 = MapTransition(193, 1)
        self.map_transit4 = MapTransition(384, 1)
        self.upsampler_mode = upsampler_mode
        self.upsampler2 = nn.Upsample(scale_factor=2, mode=upsampler_mode, align_corners=True)
        self.pam_1 = PositionAttentionModule(48)
        self.cam_1 = ChannelAttentionModule()
        self.pam_2 = PositionAttentionModule(96)
        self.cam_2 = ChannelAttentionModule()
        self.pam_3 = PositionAttentionModule(192)
        self.cam_3 = ChannelAttentionModule()
        self.pam_4 = PositionAttentionModule(384)
        self.cam_4 = ChannelAttentionModule()

    def save_model(self, path, name):
        torch.save(self.state_dict(), os.path.join(path, name))

    def load_model(self, state_dict):
        self.load_state_dict(state_dict)

    def init_weights(self):
        def init_weight(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.apply(init_weight)

    def forward(self, x):
        H, W = x.shape[2:]
        img_feats = self.image_encoder(x)
        f1, f2, f3, f4 = img_feats
        # 384, 8, 8
        f4 = self.upsample_transit4(f4)
        f4_attn_fusion = f4 + self.pam_4(f4) + self.cam_4(f4)
        f4_map = self.map_transit4(f4_attn_fusion)
        # 384, 16, 16
        f4_up = self.upsampler2(f4_attn_fusion)
        # 192, 16, 16
        f3 = self.upsample_transit3(f3 + f4_up)
        f3_attn_fusion = f3 + self.pam_3(f3) + self.cam_3(f3)
        f3_map = self.map_transit3(torch.cat([self.upsampler2(f4_map), f3_attn_fusion], dim=1))
        # 192, 32, 32
        f3_up = self.upsampler2(f3_attn_fusion)
        # 96, 32, 32
        f2 = self.upsample_transit2(f2 + f3_up)
        f2_attn_fusion = f2 + self.pam_2(f2) + self.cam_2(f2)
        f2_map = self.map_transit2(torch.cat([self.upsampler2(f3_map), f2_attn_fusion], dim=1))
        # 96, 64, 64
        f2_up = self.upsampler2(f2_attn_fusion)
        # 48, 64, 64
        f1 = self.upsample_transit1(f1 + f2_up)
        f1_attn_fusion = f1 + self.pam_1(f1) + self.cam_1(f1)
        f1_map = self.map_transit1(torch.cat([self.upsampler2(f2_map), f1_attn_fusion], dim=1))
        # 48, 128, 128
        f1_up = self.upsampler2(f1_attn_fusion)
        # 24, 128, 128
        f0 = self.upsample_transit0(f1_up)
        f0_map = self.map_transit0(torch.cat([self.upsampler2(f1_map), f0], dim=1))
        # 24, 256, 256
        f0_up = self.upsampler2(f0)
        f_final = self.upsample_transit_final(f0_up)
        final_map = self.map_transit_final(torch.cat([self.upsampler2(f0_map), f_final], dim=1))
        return f4_map, f3_map, f2_map, f1_map, f0_map, final_map
