from torch import nn
import torch.nn.utils.spectral_norm as SN

def get_fc_discriminator(num_classes, base=64):
    return nn.Sequential(
        SN(nn.Conv2d(num_classes, base, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),

        SN(nn.Conv2d(base, base * 2, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),

        SN(nn.Conv2d(base * 2, base * 4, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(base * 4, 1, 4, 2, 1)
    )
