import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import numpy as np

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


def extract_face_features(video_frames):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    B, T, C, H, W = video_frames.shape
    device = video_frames.device if isinstance(video_frames, torch.Tensor) else 'cpu'

    features = torch.zeros((B, T, 9), device=device)

    for b in range(B):
        for t in range(T):
            frame = video_frames[b, t].permute(1, 2, 0).cpu().numpy()  # (128, 128, 3)
            frame = (frame * 255).astype(np.uint8)

            results = face_mesh.process(frame)

            if not results.multi_face_landmarks:
                continue

            landmarks = results.multi_face_landmarks[0].landmark

            points = [
                (landmarks[10].x, landmarks[10].y),
                (landmarks[50].x, landmarks[50].y),
                (landmarks[280].x, landmarks[280].y)
            ]

            pixel_points = [(int(x * W), int(y * H)) for x, y in points]

            rect_size = [[80,20], [20,20], [20,20]]

            channel_means = []
            for i, (px, py) in enumerate(pixel_points):
                x1 = max(0, px - rect_size[i][0] // 2)
                x2 = min(W, px + rect_size[i][0] // 2)
                y1 = max(0, py - rect_size[i][1] // 2)
                y2 = min(H, py + rect_size[i][1] // 2)

                region = video_frames[b, t, :, y1:y2, x1:x2]  # (3, h, w)
                region = torch.mean(region, dim=2)
                region = torch.mean(region, dim=1)

                channel_means.append(region)

            features[b, t] = torch.cat(channel_means)

    return features

class ConvLSTMResidual(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super(ConvLSTMResidual, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=dim1, out_channels=dim2, kernel_size=1, stride=1, padding=0)
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(in_channels=dim2 if i == 0 else dim3,
                      out_channels=dim3,
                      kernel_size=3,
                      stride=1,
                      padding=1)
            for i in range(8)
        ])

        self.lstm1 = nn.LSTM(input_size=dim3, hidden_size=dim3, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=dim3, hidden_size=dim3, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=dim3, hidden_size=dim3, batch_first=True)

        self.fc = nn.Linear(dim3, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 9, 300)

        x = F.relu(self.conv1(x))  # (B, 32, 300)

        for conv in self.conv_blocks:
            x = F.relu(conv(x))  # (B, 64, 300)

        x = x.permute(0, 2, 1)  # (B, 300, 64)

        lstm1_out, _ = self.lstm1(x)  # (B, 300, 64)
        lstm2_out, _ = self.lstm2(lstm1_out)  # (B, 300, 64)
        lstm2_out = lstm2_out + lstm1_out
        lstm3_out, _ = self.lstm3(lstm2_out)  # (B, 300, 64)

        out = self.fc(lstm3_out)  # (B, 300, 1)

        return out

class PhysioLSTM(nn.Module):
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

        self.model = ConvLSTMResidual(64, 128, 256)

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

        # x = rearrange(x, 'b c t -> b t c') 4 300 96
        rPPG = self.model(x)
        # rPPG = rPPG.squeeze(1)
        rPPG = rPPG.view(B, -1)
        return rPPG

