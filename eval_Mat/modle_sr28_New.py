import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# SE
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


# Channel MLP: Conv1*1 -> Conv1*1
class ChannelMLP(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mlp(x)


# MBConv: Conv1*1 -> DW Conv3*3 -> [SE] -> Conv1*1
class MBConv(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mbconv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mbconv(x)


# CCM
class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


# SAFM
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
        
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        
        # Activation
        self.act = nn.GELU() 

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h//2**i, w//2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        # Multiscale Block
        self.safm = SAFM(dim) 
        # Feedforward layer
        self.ccm = CCM(dim, ffn_scale) 

    def forward(self, x):
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=5):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CABY(nn.Module):
    def __init__(self, num_feat, compress_ratio=2, squeeze_factor=10):
        super(CABY, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class Channel_mixing8(nn.Module):
    def __init__(self, dim, growth_rate=2):
        super(Channel_mixing8, self).__init__()
        # self.FF = FeedForward1(dim, growth_rate, bias=True)
        # self.conv3_1x1 = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0), nn.GELU())

    def forward(self, x, y, z):
        b, c, h, w = x.size()
        q = x
        k = y
        k_embed = rearrange(k, 'b c h w -> (b h w) 1 c')
        q_embed = rearrange(q, 'b c h w -> (b h w) c 1')
        # print(k_embed.shape)

        v = z
        v_embed = rearrange(v, 'b c h w -> (b h w) c 1')

        kv = torch.matmul(v_embed, k_embed)  # (n*h*w, c, c)
        # kv = F.softmax(kv, dim=2)
        kv = F.softmax(kv, dim=1)

        qkv = torch.matmul(kv, q_embed)  # (n*h*w, c, 1)
        # qkv = F.softmax(qkv, dim=1)

        qkv_x = qkv.squeeze(2)  # (n*h*w, c)
        out = rearrange(qkv_x, '(b h w) c -> b c h w', b=b, h=h, w=w)
        # out = self.FF(qkv_x)
        return out

class SAFM8(nn.Module):
    def __init__(self, dim, n_levels=3, drop_path=0.0):
        super(SAFM8, self).__init__()
        self.n_levels = n_levels
        chunk_dim = dim ##// n_levels
        # self.conv_pw = nn.Conv2d(dim, dim*n_levels, 1, 1, 0)
        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])
        self.cm = Channel_mixing8(dim)
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        # Activation
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        h, w = x.size()[-2:]
        # x1 = self.conv_pw(x)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(x, p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](x)
            out.append(s)
        # print(out[0].shape)
        out = self.cm(out[0], out[1], out[2])
        out = self.drop_path(self.aggr(out))
        out = self.act(out) * x
        return out

class SAFM9(nn.Module):
    def __init__(self, dim, n_levels=4, drop_path=0.0):
        super(SAFM9, self).__init__()
        self.n_levels = n_levels
        chunk_dim = dim ##// n_levels
        # self.conv_pw = nn.Conv2d(dim, dim*n_levels, 1, 1, 0)
        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim//2**(i+1), chunk_dim//2**(i+1), 3, 1, 1, groups=chunk_dim//2**(i+1)) for i in range(self.n_levels-1)])
        self.mfr4 = nn.Conv2d(dim//2**(self.n_levels-1), dim//2**(self.n_levels-1), 3, 1, 1, groups=dim//2**(self.n_levels-1))
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        # Activation
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc1 = x.chunk(2, dim=1)
        # x1 = self.conv_pw(x)
        out = []
        x1 = self.mfr[0](xc1[0])
        out.append(x1)

        x2 = F.adaptive_max_pool2d(xc1[1], 2)
        xc2 = x2.chunk(2, dim=1)
        x2 = self.mfr[1](xc2[0])
        x2 = F.interpolate(x2, size=(h, w), mode='nearest')
        out.append(x2)

        x3 = F.adaptive_max_pool2d(xc2[1], 2)
        xc3 = x3.chunk(2, dim=1)
        x3 = self.mfr[2](xc3[0])
        x3 = F.interpolate(x3, size=(h, w), mode='nearest')
        out.append(x3)

        x4 = F.adaptive_max_pool2d(xc3[1], 2)
        x4 = self.mfr4(x4)
        x4 = F.interpolate(x4, size=(h, w), mode='nearest')
        out.append(x4)

        x_out = torch.cat(out, dim=1)
        # print(out[0].shape)
        out = self.drop_path(self.aggr(x_out))
        out = self.act(out) * x
        return out

class SAFM11(nn.Module):
    def __init__(self, dim, n_levels=4, drop_path=0.0):
        super(SAFM11, self).__init__()
        self.n_levels = n_levels
        chunk_dim = dim ##// n_levels
        # self.conv_pw = nn.Conv2d(dim, dim*n_levels, 1, 1, 0)
        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim//2**(i+1), chunk_dim//2**(i+1), 3, 1, 1, groups=chunk_dim//2**(i+1)) for i in range(self.n_levels-1)])
        self.mfr4 = nn.Conv2d(dim//2**(self.n_levels-1), dim//2**(self.n_levels-1), 3, 1, 1, groups=dim//2**(self.n_levels-1))
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        # Activation
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc1 = x.chunk(2, dim=1)
        # x1 = self.conv_pw(x)
        out = []
        x1 = self.mfr[0](xc1[0])
        out.append(x1)

        # x2 = F.max_pool2d(xc1[1], 6)
        p_size = (h // 4, w // 4)  ########
        x2 = F.adaptive_max_pool2d(xc1[1], p_size)   ######### p_size
        xc2 = x2.chunk(2, dim=1)
        x2 = self.mfr[1](xc2[0])
        x2 = F.interpolate(x2, size=(h, w), mode='nearest')
        out.append(x2)

        # x3 = F.max_pool2d(xc2[1], 6)
        p_size = (math.ceil(h / 16), math.ceil(w / 16))  ########
        x3 = F.adaptive_max_pool2d(xc2[1], p_size)   #########
        xc3 = x3.chunk(2, dim=1)
        x3 = self.mfr[2](xc3[0])
        x3 = F.interpolate(x3, size=(h, w), mode='nearest')
        out.append(x3)

        # x4 = F.max_pool2d(xc3[1], 6)
        p_size = (math.ceil(h / 64), math.ceil(w / 64))  ########
        x4 = F.adaptive_max_pool2d(xc3[1], p_size)   #########
        x4 = self.mfr4(x4)
        x4 = F.interpolate(x4, size=(h, w), mode='nearest')
        out.append(x4)

        x_out = torch.cat(out, dim=1)
        # print(out[0].shape)
        out = self.drop_path(self.aggr(x_out))
        out = self.act(out) * x
        return out

class N_maxpool(nn.Module):
    def __init__(self, dim, n_levels=4, drop_path=0.0):
        super(N_maxpool, self).__init__()
        self.n_levels = n_levels
        chunk_dim = dim ##// n_levels
        # self.conv_pw = nn.Conv2d(dim, dim*n_levels, 1, 1, 0)
        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim//2**(i+1), chunk_dim//2**(i+1), 3, 1, 1, groups=chunk_dim//2**(i+1)) for i in range(self.n_levels-1)])
        self.mfr4 = nn.Conv2d(dim//2**(self.n_levels-1), dim//2**(self.n_levels-1), 3, 1, 1, groups=dim//2**(self.n_levels-1))
        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        # Activation
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc1 = x.chunk(2, dim=1)
        # x1 = self.conv_pw(x)
        out = []
        x1 = self.mfr[0](xc1[0])
        out.append(x1)

        # x2 = F.max_pool2d(xc1[1], 6)
        p_size = (h // 4, w // 4)  ########
        # x2 = F.adaptive_max_pool2d(xc1[1], p_size)   ######### p_size
        xc2 = xc1[1].chunk(2, dim=1)
        x2 = self.mfr[1](xc2[0])
        # x2 = F.interpolate(x2, size=(h, w), mode='nearest')
        out.append(x2)

        # x3 = F.max_pool2d(xc2[1], 6)
        p_size = (math.ceil(h / 16), math.ceil(w / 16))  ########
        # x3 = F.adaptive_max_pool2d(xc2[1], p_size)   #########
        xc3 = xc2[1].chunk(2, dim=1)
        x3 = self.mfr[2](xc3[0])
        # x3 = F.interpolate(x3, size=(h, w), mode='nearest')
        out.append(x3)

        # x4 = F.max_pool2d(xc3[1], 6)
        p_size = (math.ceil(h / 64), math.ceil(w / 64))  ########
        # x4 = F.adaptive_max_pool2d(xc3[1], p_size)   #########
        x4 = self.mfr4(xc3[1])
        # x4 = F.interpolate(x4, size=(h, w), mode='nearest')
        out.append(x4)

        x_out = torch.cat(out, dim=1)
        # print(out[0].shape)
        out = self.drop_path(self.aggr(x_out))
        out = self.act(out) * x
        return out

class SFT31(nn.Module):  #
    def __init__(self, dim, growth_rate, bias, window_size):
        super(SFT31, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.feat_mix = SAFM8(dim)
        self.cab = CABY(dim)

    def forward(self, x):
        x1 = self.feat_mix(self.norm1(x)) + x
        x2 = self.cab(self.norm2(x1)) + x
        return x2 ##

class SFT32(nn.Module):  #
    def __init__(self, dim, growth_rate, bias, window_size):
        super(SFT32, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.feat_mix = SAFM9(dim)
        self.cab = CABY(dim)

    def forward(self, x):
        x1 = self.feat_mix(self.norm1(x)) + x
        x2 = self.cab(self.norm2(x1)) + x
        return x2 ##

class SFT33(nn.Module):  #
    def __init__(self, dim, growth_rate, bias, window_size):
        super(SFT33, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.feat_mix = SAFM9(dim)
        self.ccm = CCM(dim)

    def forward(self, x):
        x1 = self.feat_mix(self.norm1(x)) + x
        x2 = self.ccm(self.norm2(x1)) + x
        return x2 ##

class SFT38(nn.Module):  #
    def __init__(self, dim, growth_rate, bias, window_size):
        super(SFT38, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.feat_mix = SAFM11(dim)
        self.cab = CABY(dim)

    def forward(self, x):
        x1 = self.feat_mix(self.norm1(x)) + x
        x2 = self.cab(self.norm2(x1)) + x1
        return x2 ##

class None_GPD(nn.Module):  #
    def __init__(self, dim, growth_rate, bias, window_size):
        super(None_GPD, self).__init__()
        self.norm2 = LayerNorm(dim)
        self.cab = CABY(dim)

    def forward(self, x):
        x2 = self.cab(self.norm2(x)) + x
        return x2 ##

class None_CAB(nn.Module):  #
    def __init__(self, dim, growth_rate, bias, window_size):
        super(None_CAB, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.feat_mix = SAFM11(dim)

    def forward(self, x):
        x1 = self.feat_mix(self.norm1(x)) + x
        return x1 ##

class DCC_st38(nn.Module):
    def __init__(self, upscale=4, growth_rate=2, num_blocks=8, dim=36):
        super(DCC_st38, self).__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.feats = nn.Sequential(*[SFT38(dim, growth_rate=growth_rate, bias=True, window_size=7) for _ in range(num_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscale**2, 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x

class DCC_st39(nn.Module):
    def __init__(self, upscale=4, growth_rate=2, num_blocks=8, dim=36):
        super(DCC_st39, self).__init__()
        self.upscale = upscale
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.feats = nn.Sequential(
            *[SFT38(dim, growth_rate=growth_rate, bias=True, window_size=7) for _ in range(num_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscale ** 2, 3, 1, 1),
            nn.PixelShuffle(upscale)
        )
        # self.out_img = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.to_feat(x)
        x1 = self.feats(x1) + x1
        x2 = self.to_img(x1) + F.interpolate(x, size=(h * (self.upscale), w * (self.upscale)), mode='bicubic')
        # x2 = self.out_img(x2)
        return x2


class DCC_None_CAB(nn.Module):
    def __init__(self, upscale=4, growth_rate=2, num_blocks=8, dim=36):
        super(DCC_None_CAB, self).__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.feats = nn.Sequential(*[None_CAB(dim, growth_rate=growth_rate, bias=True, window_size=7) for _ in range(num_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscale**2, 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x

class DCC_None_GPD(nn.Module):
    def __init__(self, upscale=4, growth_rate=2, num_blocks=8, dim=36):
        super(DCC_None_GPD, self).__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.feats = nn.Sequential(*[None_GPD(dim, growth_rate=growth_rate, bias=True, window_size=7) for _ in range(num_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscale**2, 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x

class DCC_Nmaxpool(nn.Module):
    def __init__(self, upscale=4, growth_rate=2, num_blocks=8, dim=36):
        super(DCC_Nmaxpool, self).__init__()
        self.upscale = upscale
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.feats = nn.Sequential(
            *[SFT38(dim, growth_rate=growth_rate, bias=True, window_size=7) for _ in range(num_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscale ** 2, 3, 1, 1),
            nn.PixelShuffle(upscale)
        )
        # self.out_img = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        x1 = self.to_feat(x)
        x1 = self.feats(x1) + x1
        x2 = self.to_img(x1) + F.interpolate(x, size=(h * (self.upscale), w * (self.upscale)), mode='bicubic')
        # x2 = self.out_img(x2)
        return x2

if __name__== '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 3, 160, 211)
    # x = torch.randn(1, 3, 256, 256)
    # model = SAFM9(32, n_levels=4, drop_path=0.1)
    model = DCC_Nmaxpool(upscale=4, growth_rate=2, num_blocks=8, dim=32)
    # model = SAFMN(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)
    # import pytorch_lightning as pl

    # checkpoint = torch.load("../pretrained/DCC_st38_epoch=1039_val_psnr=28.33.ckpt", map_location=lambda storage, loc: storage)
    # print(checkpoint["hyper_parameters"])
    # # {"learning_rate": the_value, "another_parameter": the_other_value}


