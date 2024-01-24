import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import resnet
from einops import rearrange

def _AsppConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    asppconv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        nn.GroupNorm(out_channels // 32, out_channels),
        nn.ReLU()
    )
    return asppconv

class AsppModule(nn.Module):
    def __init__(self, output_stride=16):
        super(AsppModule, self).__init__()

        # output_stride choice
        if output_stride == 16:
            atrous_rates = [0, 6, 12, 18]
        elif output_stride == 8:
            atrous_rates = 2 * [0, 12, 24, 36]
        else:
            raise Warning("output_stride must be 8 or 16!")

        self._atrous_convolution1 = _AsppConv(2048, 256, 1, 1)
        self._atrous_convolution2 = _AsppConv(2048, 256, 3, 1, padding=atrous_rates[1], dilation=atrous_rates[1]
                                              )
        self._atrous_convolution3 = _AsppConv(2048, 256, 3, 1, padding=atrous_rates[2], dilation=atrous_rates[2]
                                              )
        self._atrous_convolution4 = _AsppConv(2048, 256, 3, 1, padding=atrous_rates[3], dilation=atrous_rates[3]
                                              )
        self._image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU()
        )

    def forward(self, input):
        input1 = self._atrous_convolution1(input)
        input2 = self._atrous_convolution2(input)
        input3 = self._atrous_convolution3(input)
        input4 = self._atrous_convolution4(input)
        input5 = self._image_pool(input)
        input5 = F.interpolate(input=input5, size=input4.size()[2:4], mode='bilinear', align_corners=True)

        return torch.cat((input1, input2, input3, input4, input5), dim=1)

class MSEncoder(nn.Module):
    def __init__(self, output_stride=16):
        super(MSEncoder, self).__init__()
        self.ASPP = AsppModule(output_stride=output_stride)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1280, 384, 1, bias=True)

    def forward(self, input):
        input = self.ASPP(input)
        input = self.conv1(input)
        return input


class GRT(nn.Module):
    def __init__(self, d_model,dim_feedforward=512,  nhead=4, dropout=0.1,
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.normalize_before = normalize_before
        self.feature_trans=nn.Sequential(nn.Conv2d(d_model*3,d_model*2,kernel_size=1,stride=1,padding=0,bias=True),nn.GroupNorm(1,d_model*2),nn.ReLU(),nn.Dropout(0.05),nn.Conv2d(d_model*2,d_model*2,kernel_size=1,stride=1,padding=0,bias=True),nn.GroupNorm(1,d_model*2),nn.ReLU(),nn.Conv2d(d_model*2,d_model,kernel_size=1,stride=1,padding=0,bias=True))

    def forward(self, srcs):
        src,srclow=srcs
        src=self.feature_trans(torch.cat([src,srclow],1))+src
        b,c,h,w=src.size()
        src=rearrange(src,'b c h w->(h w) b  c')
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        feat=rearrange(src,'(h w) b c->b c h w',h=h,w=w)
        return (feat,srclow)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LocalRTBLOCK(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.02,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

        self.passlayer=nn.Sequential(nn.Conv2d(dim,dim,3,1,1),nn.GroupNorm(1,dim),nn.ReLU(inplace=True),nn.Conv2d(dim,dim,3,1,1),nn.GroupNorm(1,dim),nn.ReLU(inplace=True),nn.Conv2d(dim,dim,1,1,0))

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        passfeat= self.passlayer(rearrange(x,'b (h w) c -> b c h w',h=H,w=W))
        passfeat=rearrange(passfeat,'b c h w -> b (h w) c')
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)+passfeat
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class LRT(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        self.blocks = nn.ModuleList([
            LocalRTBLOCK(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])


    def forward(self, x):
        B,C,H,W=x.shape

        x=rearrange(x,'b c h w -> b (h w) c')

        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        x=rearrange(x,'b (h w) c -> b c h w',h=H,w=W)
        return x


class downs(nn.Module):
    def __init__(self):
        super(downs, self).__init__()
        self.down1=nn.Upsample(scale_factor=1/4,mode='bilinear',align_corners=False)
        self.down=nn.Upsample(scale_factor=1/2,mode='bilinear',align_corners=False)
    def forward(self,x):
        with torch.no_grad():
            x=self.down1(x)
            x=(x>0.05).float()
            x=self.down(x)
            x=(x>0.05).float()
            x=self.down(x)
            x=(x>0.05).float()
        return x


class DecBlock01(nn.Module):
    def __init__(self, inc, midc, stride=1):
        super(DecBlock01, self).__init__()
        self.conv1 = nn.Conv2d(inc, midc, kernel_size=1, stride=1, padding=0, bias=True)
        self.gn1=nn.GroupNorm(16,midc)
        self.conv2 = nn.Conv2d(midc, midc, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn2=nn.GroupNorm(16,midc)
        self.conv3=nn.Conv2d(midc,inc,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self,x):
        x_=x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x+x_
        x = self.relu(x)
        return x


class DecBlock00(nn.Module):
    def __init__(self, inc, midc, stride=1):
        super(DecBlock00, self).__init__()
        self.conv1 = nn.Conv2d(inc, midc, kernel_size=1, stride=1, padding=0, bias=True)
        self.gn1=nn.GroupNorm(16,midc)
        self.conv2 = nn.Conv2d(midc, midc, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn2=nn.GroupNorm(16,midc)
        self.conv3=nn.Conv2d(midc,inc,kernel_size=1,stride=1,padding=0,bias=True)
        self.relu = nn.LeakyReLU(0.0)

    def forward(self,x):
        x_=x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x+x_
        x = self.relu(x)
        return x

class DCAM_ADB_TRI(nn.Module):
    def __init__(self):
        super(DCAM_ADB_TRI, self).__init__()
        self.backbone = resnet.ResNet50_MODTRI(os=16)
        self.encoder = MSEncoder(output_stride=16)
        self.pred1=nn.Sequential(nn.Conv2d(384,256,1,1,0),nn.ReLU(),nn.Conv2d(256,128,1,1,0),nn.ReLU(),nn.Dropout(0.2),nn.Conv2d(128,1,1,1,0))
        self.emb=nn.Sequential(nn.Conv2d(3,32,3,1,1),nn.ReLU(),nn.Conv2d(32,128,1,1,0))
        self.detail_emb=nn.Conv2d(512,256,3,2,1)
        self.detail_conv=nn.Conv2d(256+384,128,1,1)
        self.glb1=nn.Sequential(GRT(128,256,4,dropout=0.1),GRT(128,256,4,dropout=0.1))
        self.loc1=nn.Sequential(nn.Conv2d(128+512,128,1,1,0),LRT(128,2,2,drop_path=0.1))
        self.locfb1=nn.Conv2d(128,128,3,2,1)
        self.m1=nn.Conv2d(256,128,1,1,0)
        self.glb2=nn.Sequential(GRT(128,256,4,dropout=0.1),GRT(128,256,4,dropout=0.1))
        self.loc2=nn.Sequential(nn.Conv2d(128+512,128,1,1,0),LRT(128,2,2,drop_path=0.1))
        self.locfb2=nn.Conv2d(128,128,3,2,1)
        self.m2=nn.Conv2d(256,256,1,1,0)
        self.predg1=nn.Sequential(nn.Conv2d(128,128,1,1,0),nn.ReLU(),nn.Conv2d(128,1,1,1,0))
        self.predg2=nn.Sequential(nn.Conv2d(128,128,1,1,0),nn.ReLU(),nn.Conv2d(128,1,1,1,0))
        self.conv16=nn.Sequential(nn.Conv2d(256+1024,384,1,1,0),DecBlock01(384,192),DecBlock01(384,192))
        self.conv8=nn.Sequential(nn.Conv2d(384+512,256,1,1,0),DecBlock00(256,128),DecBlock00(256,128))
        self.conv4=nn.Sequential(nn.Conv2d(256+256,128,1,1,0),DecBlock00(128,64),DecBlock00(128,64))
        self.conv2=nn.Sequential(nn.Conv2d(128+64,64,3,1,1,bias=True),nn.PReLU(64),nn.Conv2d(64,64,3,1,1,bias=True),nn.PReLU(64))
        self.convo=nn.Sequential(nn.Conv2d(64+6+32,24,3,1,1,bias=True),nn.PReLU(24),nn.Conv2d(24,16,3,1,1,bias=True),nn.PReLU(16),nn.Conv2d(16,1,3,1,1,bias=True))
        self.down=downs()
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.up8=nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.up16=nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.ft1=nn.Conv2d(1024,256,1,1,0)
        self.ft2=nn.Conv2d(1024,256,1,1,0)

    def forward(self, x,tri):
        ins = torch.cat([x,tri],1)
        dm=self.down(tri)
        features, _ = self.backbone(ins, True)
        ms = self.encoder(features[-1])
        pred1=self.pred1(ms)
        low_feat=self.detail_emb(features[-3])
        feats=torch.cat([ms,low_feat],1)
        feats=self.detail_conv(feats)
        _,_,h,w=dm.shape
        dm=self.emb(dm)
        feats_=dm+feats
        ft1=self.ft1(features[-2])
        ft2=self.ft2(features[-2])
        feats,_=self.glb1((feats_,ft1))
        featsup=self.up(feats)
        featsup=torch.cat([featsup,features[-3]],1)
        featsup=self.loc1(featsup)
        predg1=self.predg1(featsup)
        feats=self.m1(torch.cat([feats,self.locfb1(featsup)],1) )
        feats,_=self.glb2((feats,ft2))
        featsup=self.up(feats)
        featsup=torch.cat([featsup,features[-3]],1)
        featsup=self.loc2(featsup)
        predg2=self.predg2(featsup)
        feats=self.m2(torch.cat([feats,self.locfb2(featsup)],1) )
        feats=self.conv16(torch.cat([feats,features[-2]],1))
        feats=self.up(feats)
        feats=torch.cat([feats,features[-3]],1)
        feats=self.conv8(feats)
        feats=self.up(feats)
        feats=torch.cat([feats,features[-4]],1)
        feats=self.conv4(feats)
        feats=self.up(feats)
        feats=torch.cat([feats,features[-5]],1)
        feats=self.conv2(feats)
        feats=self.up(feats)
        feats=torch.cat([feats,ins,features[-6]],1)
        alpha=self.convo(feats)
        pred1=self.up16(pred1)
        predg1=self.up8(predg1)
        predg2=self.up8(predg2)
        alpha=torch.clamp(alpha,0,1)
        return pred1,predg1,predg2,alpha

