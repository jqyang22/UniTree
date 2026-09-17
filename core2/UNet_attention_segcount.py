import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock2d(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock2d, self).__init__()
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, g):
        if x.shape[2:] != g.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='nearest')
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        f = self.relu(theta_x + phi_g)
        psi_f = self.psi(f)
        rate = self.sigmoid(psi_f)
        out = x * rate
        return out


def attention_up_and_concate(down_layer, layer):
    up = F.interpolate(down_layer, size=layer.shape[2:], mode='nearest')
    in_channels = layer.size(1)
    inter_channels = in_channels // 4 if in_channels >= 4 else 1
    att_block = AttentionBlock2d(in_channels, up.size(1), inter_channels).to(up.device)
    layer_att = att_block(layer, up)
    concate = torch.cat([up, layer_att], dim=1)
    return concate


class UNet(nn.Module):
    def __init__(self, input_shape, input_label_channel=1, layer_count=64, regularizers=None, gaussian_noise=0.1,
                 weight_file=None, inputBN=0):
        super(UNet, self).__init__()
        self.inputBN = inputBN
        if self.inputBN:
            self.bn_input = nn.BatchNorm2d(input_shape[0])
        self.conv1_1 = nn.Conv2d(input_shape[0], layer_count, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(layer_count, layer_count, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(layer_count)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(layer_count, 2 * layer_count, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(2 * layer_count, 2 * layer_count, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * layer_count)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(2 * layer_count, 4 * layer_count, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(4 * layer_count, 4 * layer_count, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * layer_count)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(4 * layer_count, 8 * layer_count, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(8 * layer_count, 8 * layer_count, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(8 * layer_count)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(8 * layer_count, 16 * layer_count, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(16 * layer_count, 16 * layer_count, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(16 * layer_count)

        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.att6 = AttentionBlock2d(8 * layer_count, 16 * layer_count, max(8 * layer_count // 4, 1))
        self.conv6_1 = nn.Conv2d(16 * layer_count + 8 * layer_count, 8 * layer_count, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(8 * layer_count, 8 * layer_count, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(8 * layer_count)

        self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.att7 = AttentionBlock2d(4 * layer_count, 8 * layer_count, max(4 * layer_count // 4, 1))
        self.conv7_1 = nn.Conv2d(8 * layer_count + 4 * layer_count, 4 * layer_count, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(4 * layer_count, 4 * layer_count, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(4 * layer_count)

        self.up8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.att8 = AttentionBlock2d(2 * layer_count, 4 * layer_count, max(2 * layer_count // 4, 1))
        self.conv8_1 = nn.Conv2d(4 * layer_count + 2 * layer_count, 2 * layer_count, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(2 * layer_count, 2 * layer_count, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(2 * layer_count)

        self.up9 = nn.Upsample(scale_factor=2, mode='nearest')
        self.att9 = AttentionBlock2d(layer_count, 2 * layer_count, max(layer_count // 4, 1))
        self.conv9_1 = nn.Conv2d(2 * layer_count + layer_count, layer_count, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(layer_count, layer_count, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(layer_count)

        self.out_seg = nn.Conv2d(layer_count, 1, kernel_size=1)
        self.out_dens = nn.Conv2d(layer_count, 1, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.inputBN:
            x = self.bn_input(x)
        c11 = F.relu(self.conv1_1(x))
        c11 = F.relu(self.conv1_2(c11))
        n11 = self.bn1(c11)
        p11 = self.pool1(n11)

        c2 = F.relu(self.conv2_1(p11))
        c2 = F.relu(self.conv2_2(c2))
        n2 = self.bn2(c2)
        p2 = self.pool2(n2)

        c3 = F.relu(self.conv3_1(p2))
        c3 = F.relu(self.conv3_2(c3))
        n3 = self.bn3(c3)
        p3 = self.pool3(n3)

        c4 = F.relu(self.conv4_1(p3))
        c4 = F.relu(self.conv4_2(c4))
        n4 = self.bn4(c4)
        p4 = self.pool4(n4)

        c5 = F.relu(self.conv5_1(p4))
        c5 = F.relu(self.conv5_2(c5))
        n5 = self.bn5(c5)

        up6 = self.up6(n5)
        att4 = self.att6(n4, n5)
        concat6 = torch.cat([up6, att4], dim=1)
        c6 = F.relu(self.conv6_1(concat6))
        c6 = F.relu(self.conv6_2(c6))
        n6 = self.bn6(c6)

        up7 = self.up7(n6)
        att3 = self.att7(n3, n6)
        concat7 = torch.cat([up7, att3], dim=1)
        c7 = F.relu(self.conv7_1(concat7))
        c7 = F.relu(self.conv7_2(c7))
        n7 = self.bn7(c7)

        up8 = self.up8(n7)
        att2 = self.att8(n2, n7)
        concat8 = torch.cat([up8, att2], dim=1)
        c8 = F.relu(self.conv8_1(concat8))
        c8 = F.relu(self.conv8_2(c8))
        n8 = self.bn8(c8)

        up9 = self.up9(n8)
        att1 = self.att9(n11, n8)
        concat9 = torch.cat([up9, att1], dim=1)
        c9 = F.relu(self.conv9_1(concat9))
        c9 = F.relu(self.conv9_2(c9))
        n9 = self.bn9(c9)

        seg = torch.sigmoid(self.out_seg(n9))
        dens = self.out_dens(n9)

        return seg, self.out_seg(n9), dens
