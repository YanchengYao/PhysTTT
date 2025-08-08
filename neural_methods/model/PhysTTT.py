import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neural_methods.model.base.ttt_layer import TTTLayer

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
        diff = x3-x2
        torch.save(x, 'origin_face.pt')
        torch.save(diff, 'diff_face.pt')
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


class ResNetTTT(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super(ResNetTTT, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=dim1, out_channels=dim3, kernel_size=1, stride=1, padding=0)
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(in_channels=dim3,
                      out_channels=dim3,
                      kernel_size=3,
                      stride=1,
                      padding=1)
            for i in range(8)
        ])

        self.layer_norm = nn.LayerNorm(dim3)
        self.ttt = TTTLayer(
            model_dim=256,
            num_heads=4,
            method='ttt_mlp',
            mini_batch_size=4,
            ttt_base_lr=0.1,
            scan_checkpoint_group_size=4
        )

        self.fc = nn.Linear(dim3, dim2)
        self.fc2 = nn.Linear(dim2, dim1)
        self.fc3 = nn.Linear(dim1, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))

        x1 = F.relu(self.conv_blocks[0](x))
        x2 = F.relu(self.conv_blocks[1](x1))

        x3 = F.relu(self.conv_blocks[2](x2))
        x4 = F.relu(self.conv_blocks[3](x3))

        x5 = F.relu(self.conv_blocks[4](x4))
        x6 = F.relu(self.conv_blocks[5](x5))

        x7 = F.relu(self.conv_blocks[6](x6))
        x8 = F.relu(self.conv_blocks[7](x7))

        x = x2 + x4 + x6 + x8

        x = x.permute(0, 2, 1)  # (B, 300, 256)
        x = self.layer_norm(x)

        x = self.ttt(x)
        x = self.layer_norm(x)

        x = self.fc(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out

class PhysTTT(nn.Module):
    def __init__(self,
                 embed_dim=64,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()

        self.embed_dim = embed_dim

        self.Fusion_Stem = Fusion_Stem(dim=embed_dim // 4)
        self.attn_mask = Attention_mask()

        self.stem3 = nn.Sequential(  # conv2d
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
        )

        self.model = ResNetTTT(64, 128, 256)

    def forward(self, x):
        B, D, C, H, W = x.shape

        x = self.Fusion_Stem(x)
        x = self.stem3(x)
        x = x.view(B, D, self.embed_dim, H // 8, W // 8)
        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)

        x = x * mask
        x = torch.mean(x, 4)
        x = torch.mean(x, 3)
        rPPG = self.model(x)
        rPPG = rPPG.view(B, -1)
        return rPPG
