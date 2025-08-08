"""
PhysTTT
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
from functools import partial
from timm.layers import trunc_normal_, lecun_normal_
from timm.layers import DropPath, to_2tuple
import math
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
from neural_methods.model.base.ttt_layer import TTTLayer
import cv2
import dlib

class Fusion_Stem(nn.Module):
    def __init__(self, apha=0.5, belta=0.5, dim=24):
        super(Fusion_Stem, self).__init__()

        self.stem11 = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem12 = nn.Sequential(
            nn.Conv2d(12, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem21 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem22 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/8,W/8]
        """
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:, :1, :, :, :], x[:, :1, :, :, :], x[:, :D - 2, :, :, :]], 1)
        x2 = torch.cat([x[:, :1, :, :, :], x[:, :D - 1, :, :, :]], 1)
        x3 = x
        x4 = torch.cat([x[:, 1:, :, :, :], x[:, D - 1:, :, :, :]], 1)
        x5 = torch.cat([x[:, 2:, :, :, :], x[:, D - 1:, :, :, :], x[:, D - 1:, :, :, :]], 1)
        x_diff = self.stem12(torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2).view(N * D, 12, H, W))

        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)  # 4*160, 12, 32, 32
        # print('stem11: ', x.shape)
        # fusion layer1
        x_path1 = self.apha * x + self.belta * x_diff
        x_path1 = self.stem21(x_path1)  # 4*160, 24, 16, 16
        # print('stem21: ', x_path1.shape)
        # fusion layer2
        x_path2 = self.stem22(x_diff)
        # print('stem22: ', x_path2.shape)
        x = self.apha * x_path1 + self.belta * x_path2  # 4*160, 24, 16, 16
        # print('fusion stem output: ', x.shape)
        return x


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()

        self.scale = 0.02
        self.dim = dim * mlp_ratio

        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))

        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm1d(dim * mlp_ratio),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        B, N, C = x.shape

        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)

        x_fre = torch.fft.fft(x, dim=1, norm='ortho')  # FFT on N dimension

        x_real = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.real, self.r) - \
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.i) + \
            self.rb
        )
        x_imag = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.r) + \
            torch.einsum('bnc,cc->bnc', x_fre.real, self.i) + \
            self.ib
        )

        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho")
        x = x.to(torch.float32)

        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)
        return x


# class SCAttention(nn.Module):
#     """支持4个输入张量的空间-通道注意力模块"""
#
#     def __init__(self, reduction_ratio=16):
#         super().__init__()
#
#         # 1. 通道注意力分支（对每个输入独立处理）
#         self.channel_attentions = nn.ModuleList([
#             nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(c, c // reduction_ratio, 1, bias=False),
#                 nn.ReLU(),
#                 nn.Conv2d(c // reduction_ratio, c, 1, bias=False),
#                 nn.Sigmoid()
#             ) for c in channels_list
#         ])
#
#         # 2. 空间注意力分支（多输入融合后处理）
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(sum(channels_list), 1, kernel_size=7, padding=3, bias=False),  # 融合所有输入
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         """输入: 4个张量 (B, C_i, H, W)"""
#
#         assert len(x) == 4, "需要4个输入张量的通道数列表"
#
#         # 1. 通道注意力
#         c1_att = self.channel_attentions[0](x[0]) * x[0]  # (B, C1, H, W)
#         c2_att = self.channel_attentions[1](x[1]) * x[1]  # (B, C2, H, W)
#         c3_att = self.channel_attentions[2](x[2]) * x[2]  # (B, C3, H, W)
#         c4_att = self.channel_attentions[3](x[3]) * x[3]  # (B, C4, H, W)
#
#         # 2. 多输入拼接用于空间注意力
#         x_concat = torch.cat([c1_att, c2_att, c3_att, c4_att], dim=1)  # (B, C1+C2+C3+C4, H, W)
#         s_att = self.spatial_attention(x_concat)  # (B, 1, H, W)
#
#         # 3. 应用空间注意力到每个输入
#         y1 = c1_att * s_att
#         y2 = c2_att * s_att
#         y3 = c3_att * s_att
#         y4 = c4_att * s_att
#
#         y_concat = torch.cat([y1, y2, y3, y4], dim=1)
#
#         return y_concat  # 返回增强后的4个张量

class SCAttention(nn.Module):
    def __init__(self, in_channels=2048, out_channels=64, reduction_ratio=16):
        super().__init__()
        # Channel attention layers
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        # Spatial attention layer
        self.conv_spatial = nn.Conv2d(1, 1, kernel_size=7, padding=3)
        # Projection layer
        self.conv_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.device = torch.device("cuda:0")
        self.weights = nn.Parameter(torch.ones(4))
        # self.weights = torch.tensor([0.1, 0.6, 0.2, 0.1]).to(self.device)  # 3 4 2 1  改成可学习的参数 混合高斯分布
        self.conv1 = nn.Conv1d(in_channels, in_channels // 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels // 8, out_channels, kernel_size=3, padding=1)


    def forward(self, x): # B T C
        weights = torch.distributions.Dirichlet(self.weights.abs()).rsample()
        x_stacked = torch.stack(x)  # shape: [4, 4, 300, 64]
        x_fusion = torch.einsum("i,ijkl->jkl", weights, x_stacked)
        x_fusion = x_fusion.permute(0, 2, 1) # B C T
        x_c1 = self.conv1(x_fusion)
        x_c2 = self.conv2(x_c1)  # B 64 T
        x_c2 = x_c2.permute(0, 2, 1)
        x_out = F.relu(x_c2)  # B T 64

        # Concatenate the 4 branches along the channel dimension
        # x = torch.cat([x[0], x[1], x[2], x[3]], dim=1)  # Shape: (batch_size, 4*C, H, W)
        #
        # # Channel attention
        # batch_size, C, H, W = x.shape
        # s = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, C)  # Global average pooling
        # s = F.relu(self.fc1(s))  # First FC layer with ReLU
        # s = torch.sigmoid(self.fc2(s)).view(batch_size, C, 1, 1)  # Second FC layer with Sigmoid
        # x_channel = x * s  # Apply channel attention
        #
        # # Spatial attention
        # spatial_avg = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average pooling
        # spatial_attention = torch.sigmoid(self.conv_spatial(spatial_avg))  # 7x7 conv with Sigmoid
        # x_attention = x_channel * spatial_attention  # Apply spatial attention
        #
        # # Projection to a single branch
        # x_out = self.conv_proj(x_attention)  # Shape: (batch_size, out_channels, H, W)
        return x_out

class TimesBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(TimesBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class Branch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.times_block = TimesBlock(in_channels=dim, out_channels=2048)
        self.conv = nn.Conv2d(in_channels=dim, out_channels=2048, kernel_size=(3, 5), stride=1)
        self.silu = nn.SiLU()
        self.ttt = TTTLayer(
            model_dim=2048,
            num_heads=8,
            method='ttt_mlp',
            mini_batch_size=10,
            ttt_base_lr=0.1,
            scan_checkpoint_group_size=4
        )

    def forward(self, x, seg):
        B, T, C = x.shape
        x = rearrange(x, "B T C -> B C T")
        x_b = x.reshape(B, C, T // seg, seg)  # B T N SEG
        # print('branch: ',x_b.shape)

        x_b = self.times_block(x_b)   # B T N SEG
        # print('times block: ',x_b.shape)
        x_b = self.silu(x_b)
        # print('silu: ',x_b.shape)
        x_b = x_b.view(B, x_b.shape[1], -1)
        # print(x_b.shape)
        x_b = rearrange(x_b, "B C T -> B T C")  # B T C_out 4 300 2048
        # print('pre ttt: ',x_b.shape)
        x_ttt = self.ttt(x_b)
        return x_ttt          # without TTT

class MambaTTTBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        # self.attn = MambaTTTLayer(dim)
        self.mlp = Frequencydomain_FFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

        self.branch = Branch(self.dim)
        self.attn = SCAttention()
        self.conv = nn.Conv1d(dim, 2048, kernel_size=3, padding=1)
        self.silu = nn.SiLU()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, T, C = x.size()
        if T != 300:
            x = F.pad(x, (0, 0, 0, 300-T), mode='constant', value=0)  # 只在C维末尾补2
        # Multi-temporal Parallelization
        x = self.norm1(x)
        segment = [75, 60, 30, 15]
        # segment = [150, 75, 60, 30, 15]
        branch_num = len(segment)
        x_r = x.unsqueeze(0).repeat(branch_num, 1, 1, 1)
        x_c = x_r.clone()
        x_b = []
        for i in range(branch_num):
            x_b.append(self.branch(x_c[i], segment[i]))
        # print(x_b[0].shape)  # 4, 300, 2048

        x_b0 = x
        x_b0 = x_b0.permute(0, 2, 1)  # 4 96 300
        x_b0 = self.conv(x_b0)  # 4 2048 300
        x_b0 = self.silu(x_b0)
        x_b0 = x_b0.permute(0, 2, 1)  # 4 300 2048

        x_b_ttt = []
        for i in range(branch_num):
            x_b_ttt.append(x_b0 * x_b[i])  # 4 4 300 2048

        x_attn = self.attn(x_b_ttt)
        x_out = x_attn + x  # residual
        x_out = self.norm2(x_out)

        # # 1. 输入扩展
        # x = self.in_proj(x)  # (B, L, expand*dim)
        #
        # # 2. 局部卷积
        # x = rearrange(x, "B L D -> B D L")  # 转为通道优先
        # x = self.conv(x)  # (B, expand*dim, L)
        # x = rearrange(x, "B D L -> B L D")  # 恢复形状
        #
        # # 3. 选择性SSM（简化版）
        # dt = self.dt_proj(x).sigmoid()  # (B, L, 1)
        # x_ssm = self.x_proj(x)  # (B, L, d_state)
        #
        # # 离散化状态矩阵（简化处理）
        # A_discrete = torch.exp(self.A * dt)  # (B, L, d_state, d_state)
        # out_ssm = torch.einsum("blij,blj->bli", A_discrete, x_ssm)  # (B, L, d_state)
        #
        # # 合并局部与全局信息
        # out = x + out_ssm  # 残差连接
        #
        # # 4. 输出投影
        # out = self.out_proj(out)  # (B, L, dim)

        return x_out


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class PhysTTT(nn.Module):
    def __init__(self,
                 depth=4,
                 embed_dim=64,
                 mlp_ratio=2,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 initializer_cfg=None,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        # self.device = device
        self.embed_dim = embed_dim

        self.Fusion_Stem = Fusion_Stem(dim=embed_dim // 4)
        self.attn_mask = Attention_mask()

        # self.stem3 = nn.Sequential(  # conv2d
        #     nn.Conv3d(embed_dim // 4, embed_dim, kernel_size=(2, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2)),
        #     nn.BatchNorm3d(embed_dim),
        # )

        self.stem3 = nn.Sequential(  # conv2d
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr

        self.blocks = nn.ModuleList([MambaTTTBlock(
            dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path=inter_dpr[i],
            norm_layer=nn.LayerNorm, )
            for i in range(depth)])

        self.upsample = nn.Upsample(scale_factor=2)
        self.ConvBlockLast = nn.Conv1d(embed_dim, 1, kernel_size=1, stride=1, padding=0)

        # init
        self.apply(segm_init_weights)
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, x):
        B, D, C, H, W = x.shape

        x = self.Fusion_Stem(x)  # [N*D C H/8 W/8] # 4*160, 24, 16, 16
        x = self.stem3(x)  # 640 96 16 16
        x = x.view(B, D, self.embed_dim, H // 8, W // 8)  # 4 160 96 16 16
        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask  # 4 96 80 16 16

        x = torch.mean(x, 4)
        x = torch.mean(x, 3)
        # x = rearrange(x, 'b c t -> b t c')
        # print('pretimeinput: ', x.shape)
        for blk in self.blocks:
            x = blk(x)

        rPPG = x.permute(0, 2, 1)  # B C T
        # rPPG = self.upsample(rPPG)
        rPPG = self.ConvBlockLast(rPPG)  # [N, 1, T]
        rPPG = rPPG.squeeze(1)

        return rPPG