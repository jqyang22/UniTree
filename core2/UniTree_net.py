import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiScaleEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )

        self.conv7_dw = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.conv7_pw = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)

        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.relu = nn.ReLU(inplace=True)

        self.reduce = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.relu7(self.conv7_pw(self.conv7_dw(x)))

        out = torch.cat([x3, x5, x7], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.reduce(out)

        return out


class LightMultiScaleEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super().__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.relu = nn.ReLU(inplace=True)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 3, (out_channels * 3) // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d((out_channels * 3) // reduction, out_channels * 3, 1, bias=False),
            nn.Sigmoid()
        )

        self.reduce = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        out = torch.cat([x3, x5, x7], dim=1)
        out = self.bn(out)
        out = self.relu(out)

        se_weight = self.se(out)
        out = out * se_weight

        out = self.reduce(out)
        return out


class DeformableAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.offset_conv = nn.Conv2d(dim, 2, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        offset = self.offset_conv(x)
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_flat + attn_out
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

class BoundaryBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.conv(x))

def safe_downsample(tensor, scale_factor):
    h, w = tensor.shape[-2:]

    new_h = int(round(h * scale_factor))
    new_w = int(round(w * scale_factor))


    tensor_resized = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear',
                                   align_corners=False)
    return tensor_resized


class UniTree(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels[0], base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels)
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels * 2)
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels * 4)
        )

        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels * 8)
        )

        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, padding=1),
            nn.ReLU(),
            DeformableAttentionBlock(dim=base_channels * 16),
            nn.Conv2d(base_channels * 16, base_channels * 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels * 16)
        )

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_channels * 16 + base_channels * 8, base_channels * 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels * 8)
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 8 + base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels * 4)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4 + base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels * 2)
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2 + base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels)
        )

        self.out_mask = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.out_dens = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.out_edge = BoundaryBranch(base_channels)

    def forward(self, x):

        c1_hr = self.enc1(x)
        p1_hr = self.pool1(c1_hr)

        c2_hr = self.enc2(p1_hr)
        p2_hr = self.pool2(c2_hr)

        c3_hr = self.enc3(p2_hr)
        p3_hr = self.pool3(c3_hr)

        c4_hr = self.enc4(p3_hr)
        p4_hr = self.pool4(c4_hr)

        bn_hr = self.bottleneck(p4_hr)

        up4_hr = self.up4(bn_hr)
        d4_hr = self.dec4(torch.cat([up4_hr, c4_hr], dim=1))

        up3_hr = self.up3(d4_hr)
        d3_hr = self.dec3(torch.cat([up3_hr, c3_hr], dim=1))

        up2_hr = self.up2(d3_hr)
        d2_hr = self.dec2(torch.cat([up2_hr, c2_hr], dim=1))

        up1_hr = self.up1(d2_hr)
        d1_hr = self.dec1(torch.cat([up1_hr, c1_hr], dim=1))

        seg_hr = torch.sigmoid(self.out_mask(d1_hr))
        dens_hr = self.out_dens(d1_hr)
        edge_hr = self.out_edge(d1_hr)

        return c3_hr, seg_hr, self.out_mask(d1_hr), dens_hr, edge_hr

    def visualize_features(self, input_img, features1, features2, save_path):
        import os
        import matplotlib.pyplot as plt
        os.makedirs(save_path, exist_ok=True)

        import uuid
        import torchvision.utils as vutils

        unique_id = str(uuid.uuid4())[:8]
        save_dir = os.path.join(save_path, unique_id)
        os.makedirs(save_dir, exist_ok=True)

        img_np = input_img[0].detach().cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        plt.imsave(os.path.join(save_dir, "input_img.png"), img_np)

        for ch in range(min(3, features1.shape[1])):
            f = features1[0, ch].detach().cpu().numpy()
            f = (f - f.min()) / (f.max() - f.min())

            plt.imsave(os.path.join(save_dir, f"feature1_map_ch{ch}.png"), f)

        for ch in range(min(3, features2.shape[1])):
            f = features2[0, ch].detach().cpu().numpy()
            f = (f - f.min()) / (f.max() - f.min())
            plt.imsave(os.path.join(save_dir, f"feature2_map_ch{ch}.png"), f)
