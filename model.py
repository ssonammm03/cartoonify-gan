import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c),
            nn.ReLU(True),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """
    Fast CartoonGAN-lite generator:
    Encoder (2 convs) -> 3 ResBlocks -> Decoder (2 deconvs) -> Tanh
    """
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.res = nn.Sequential(*[ResBlock(128) for _ in range(3)])
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.dec(self.res(self.enc(x)))
